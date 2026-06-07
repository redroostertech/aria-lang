//! Native (C) reverse-mode autodiff: a *traced* source-to-source compilation of
//! the function `f` passed to `grad(f, x)`.
//!
//! The interpreter implements `grad` with a runtime tape (a Wengert list) that
//! flows wherever a `Float`/`Vector` would *inside* `f` (see
//! `interp::Tape`/`Value::Tracing`/`Value::TracingVec`). This module reproduces
//! that on the native backend by emitting, per `grad` call site, a C helper
//!
//! ```c
//! static void* af_vec_grad$N(void* x);  /* AriaVector* -> AriaVector* (gradient) */
//! ```
//!
//! whose body builds one tape LEAF per coordinate of the input vector `x`,
//! evaluates a *traced* compilation of `f`'s body (each supported scalar/vector
//! op recorded on the `AriaTape` instead of computing a bare `double`), seeds the
//! scalar result's adjoint to 1, sweeps the tape in reverse, and returns the
//! leaves' adjoints as a fresh `AriaVector` — exactly the interpreter's
//! algorithm, numerically bit-matched (same f64 ops, same left-to-right
//! summation order; built with `-ffp-contract=off`).
//!
//! Supported op subset (a SOUND subset of the interpreter's — anything outside it
//! is rejected with a clean `grad: ...` error, never a wrong gradient, never a
//! panic):
//!   * scalars: Float/Int literals, Float-typed locals, `+ - * /`, unary `-`,
//!     and the scalar-producing vec builtins `vec_get`, `vec_dot`, `vec_norm`;
//!   * vectors: the input vector, vec-typed locals, `vec_add`, `vec_sub`,
//!     `vec_scale`, `vec_from_array([..])`, `vec_push`;
//!   * `let` bindings (in a `Block`) of either kind;
//!   * CONTROL FLOW: `if`/`else` and `match` (on a concrete Int/Bool/Float
//!     scrutinee, with Int/Bool/wildcard/var arms), differentiated through the
//!     TAKEN branch exactly as the interpreter does. The condition/scrutinee must
//!     reduce to a CONCRETE (non-differentiated) value — `vec_len`, captured
//!     scalars, literals, and `+ - * / %`, comparisons, `&& || !` over them — so
//!     the branch decision never observes a value being differentiated (a
//!     differentiated-value condition is a clean error in BOTH backends);
//!   * INTER-PROCEDURAL CALLS: a call to another (non-recursive) top-level
//!     function inside `f` is inlined — its parameters bound to the traced
//!     argument values and its body traced — so its ops record tape nodes,
//!     mirroring the interpreter stepping into the callee at runtime.
//! GATED (clean error): a condition/scrutinee that depends on a differentiated
//! value; `match` on constructors/records; (mutually) recursive calls inside `f`;
//! records/tuples, closures, other builtins, and any op on a non-{Float,Vector}
//! value. `f` itself may be a lambda literal `\v -> ..` OR a named top-level
//! function `grad(loss, x)` (its body is inlined) — as long as it stays within
//! this subset.

use std::collections::HashMap;
use std::fmt::Write;

use crate::ast::{Arm, BinOp, Expr, ExprKind, FnDecl, Item, PatternKind, Program, StmtKind, UnOp};

/// The C value produced by a traced expression: a scalar tape node-id, or a
/// traced vector (an `AriaTVec` of node-ids). Both are held in fresh C locals
/// whose names this enum carries.
#[derive(Clone)]
enum TVal {
    /// A scalar tape node: `int64_t <name>` holding a node-id.
    Scalar(String),
    /// A traced vector: `AriaTVec <name>` of node-ids.
    Vec(String),
    /// A concrete (non-differentiated) Int forward value: an `int64_t` C local.
    /// Only `let`-bound concrete Ints (loop counters / structural sizes) and
    /// captured Ints land here; they may feed `if`/`match` conditions and
    /// constant `vec_get` indices but never the tape. Mirrors the interpreter,
    /// where such values stay plain `Value::Int` (never `Tracing`).
    Int(String),
    /// A concrete Bool forward value: an `int` C local holding 0/1.
    Bool(String),
}

/// A concrete (NON-differentiated) forward value, used only for `if`/`match`
/// conditions and scrutinees (and constant indices). It is a C expression of a
/// concrete primitive type. Producing one of these NEVER records a tape node:
/// it mirrors the interpreter, where a control-flow condition is evaluated to a
/// plain `Value::Int`/`Value::Bool`/`Value::Float` (a comparison on a *traced*
/// scalar is a clean error in BOTH backends). The carried `String` is a C rvalue.
#[derive(Clone)]
enum FVal {
    /// A concrete `int64_t` C expression.
    Int(String),
    /// A concrete `int` (0/1) C expression.
    Bool(String),
    /// A concrete `double` C expression — only ever derived from other concrete
    /// forward values (literals, captured Floats), NEVER from a traced node, so
    /// it can be compared without observing a differentiated value.
    Float(String),
}

/// One discovered `grad` call site, rewritten to a `vec_grad$N(x, <captures..>)`
/// builtin call.
pub struct GradSite {
    /// The generated helper name, e.g. `vec_grad$0`. The `vec_` prefix routes it
    /// through the IR's `Bind::Call` path and the C backend's builtin dispatch.
    pub helper: String,
    /// The complete C definition of the helper (signature + body).
    pub c_def: String,
    /// Captured free variables of `f` (besides the input vector), in the order
    /// the helper takes them as extra parameters: `(name, is_vector)`. The call
    /// site passes these after `x`.
    pub captures: Vec<(String, bool)>,
}

/// Whether a captured free variable of `f` is used as a Vector or a Float.
#[derive(Clone, Copy)]
enum CapKind {
    Vector,
    Scalar,
}

struct Tracer<'a> {
    /// Top-level functions, for inlining a named `f` (or — not yet — calls).
    fns: &'a HashMap<String, &'a FnDecl>,
    /// In-scope traced locals: name -> its traced value.
    scope: HashMap<String, TVal>,
    /// Free variables of `f` referenced by the body — captured values passed in
    /// from the enclosing scope as extra helper parameters. Discovered lazily on
    /// first reference; a later inconsistent use (Vector then Float) is rejected.
    captures: HashMap<String, CapKind>,
    /// Capture discovery order (so the helper params and the call args line up).
    capture_order: Vec<String>,
    /// For a captured VECTOR, the name of the lifted `AriaTVec` local (created
    /// once on first use and reused for every later reference).
    cap_vecs: HashMap<String, String>,
    /// Fresh-temporary counter for C locals.
    tmp: usize,
    /// The emitted C statements for the helper body.
    body: String,
    /// A PROLOGUE emitted once, before the traced body, for statements that must
    /// dominate every later reference regardless of which `if`/`match` branch
    /// they appear in — currently the captured-Vector lifts (an `AriaTVec` local
    /// reused across branches). Emitting these into a branch-local buffer would
    /// leave them out of scope in sibling branches; the prologue hoists them.
    prologue: String,
    /// Inlining call stack: names of user functions currently being inlined
    /// inside `f`. Guards against (mutual) recursion — a function that calls
    /// itself, directly or indirectly, cannot be inlined into a finite trace, so
    /// we gate it cleanly (the interpreter `aria run` handles it).
    inlining: Vec<String>,
}

/// Maximum inter-procedural inlining depth inside `f`. A straight-line call
/// graph is typically shallow; this bounds pathological (but non-recursive)
/// nesting so a single `grad` trace cannot explode. Beyond it we gate cleanly.
const MAX_INLINE_DEPTH: usize = 64;

/// Rewrite every `grad(f, x)` in `program` to a `vec_grad$N(x)` call and return
/// the generated helper C definitions. On any unsupported `f`, return a clean
/// `Err` so the whole native compile fails with a specific message (the
/// interpreter `aria run` remains the fallback). A program with no `grad` calls
/// is returned unchanged with an empty site list.
pub fn rewrite_grad(program: &mut Program) -> Result<Vec<GradSite>, String> {
    // Index top-level functions for inlining a named `f`.
    let fns: HashMap<String, FnDecl> = program
        .items
        .iter()
        .filter_map(|it| match it {
            Item::Fn(f) => Some((f.name.clone(), f.clone())),
            _ => None,
        })
        .collect();
    let fn_refs: HashMap<String, &FnDecl> = fns.iter().map(|(k, v)| (k.clone(), v)).collect();

    let mut sites = Vec::new();
    let mut counter = 0usize;
    // Walk every function body, rewriting in place.
    for item in &mut program.items {
        if let Item::Fn(f) = item {
            rewrite_expr(&mut f.body, &fn_refs, &mut sites, &mut counter)?;
        }
    }
    Ok(sites)
}

/// Recursively rewrite `grad(...)` calls inside `e`.
fn rewrite_expr(
    e: &mut Expr,
    fns: &HashMap<String, &FnDecl>,
    sites: &mut Vec<GradSite>,
    counter: &mut usize,
) -> Result<(), String> {
    // First rewrite children, then this node (so nested grads are handled; the
    // tracer itself rejects a grad-inside-f as unsupported).
    match &mut e.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Str(_)
        | ExprKind::Unit
        | ExprKind::Var(_) => return Ok(()),
        ExprKind::Ctor(_, args) | ExprKind::Call(_, args) => {
            for a in args.iter_mut() {
                rewrite_expr(a, fns, sites, counter)?;
            }
        }
        ExprKind::Lambda(_, body, _) => rewrite_expr(body, fns, sites, counter)?,
        ExprKind::Apply(c, args, _) => {
            rewrite_expr(c, fns, sites, counter)?;
            for a in args.iter_mut() {
                rewrite_expr(a, fns, sites, counter)?;
            }
        }
        ExprKind::Record(_, fields) => {
            for (_, v) in fields.iter_mut() {
                rewrite_expr(v, fns, sites, counter)?;
            }
        }
        ExprKind::Field(o, _) => rewrite_expr(o, fns, sites, counter)?,
        ExprKind::Update(b, ups) => {
            rewrite_expr(b, fns, sites, counter)?;
            for (_, v) in ups.iter_mut() {
                rewrite_expr(v, fns, sites, counter)?;
            }
        }
        ExprKind::Unary(_, inner) => rewrite_expr(inner, fns, sites, counter)?,
        ExprKind::Binary(_, l, r) => {
            rewrite_expr(l, fns, sites, counter)?;
            rewrite_expr(r, fns, sites, counter)?;
        }
        ExprKind::If(c, t, el) => {
            rewrite_expr(c, fns, sites, counter)?;
            rewrite_expr(t, fns, sites, counter)?;
            rewrite_expr(el, fns, sites, counter)?;
        }
        ExprKind::Match(s, arms) => {
            rewrite_expr(s, fns, sites, counter)?;
            for arm in arms.iter_mut() {
                rewrite_expr(&mut arm.body, fns, sites, counter)?;
            }
        }
        ExprKind::Block(stmts, last) => {
            for s in stmts.iter_mut() {
                match &mut s.kind {
                    StmtKind::Let { value, .. } => rewrite_expr(value, fns, sites, counter)?,
                    StmtKind::Expr(ex) => rewrite_expr(ex, fns, sites, counter)?,
                }
            }
            rewrite_expr(last, fns, sites, counter)?;
        }
    }

    // Now, is THIS node a `grad(f, x)` call?
    if let ExprKind::Call(name, args) = &e.kind {
        if name == "grad" {
            if args.len() != 2 {
                return Err("grad expects (f: (Vector) -> Float, x: Vector)".into());
            }
            let (param, fbody) = resolve_f(&args[0], fns)?;
            let helper = format!("vec_grad${}", *counter);
            *counter += 1;
            let (c_def, captures) = compile_helper(&helper, &param, &fbody, fns)?;
            // Rewrite `grad(f, x)` -> `helper(x, <captures..>)`. Each captured
            // free variable becomes a trailing argument (a `Var` reference,
            // resolved in the enclosing scope exactly as the closure would).
            let mut call_args = vec![args[1].clone()];
            for (name, _) in &captures {
                call_args.push(Expr::synth(ExprKind::Var(name.clone())));
            }
            sites.push(GradSite { helper: helper.clone(), c_def, captures });
            e.kind = ExprKind::Call(helper, call_args);
        }
    }
    Ok(())
}

/// Resolve the `f` argument of `grad` to `(param_name, body)`. `f` is either a
/// lambda literal `\v -> ..` or a top-level function name (its single param +
/// body are inlined). Anything else is rejected.
fn resolve_f(f: &Expr, fns: &HashMap<String, &FnDecl>) -> Result<(String, Expr), String> {
    match &f.kind {
        ExprKind::Lambda(params, body, _) => {
            if params.len() != 1 {
                return Err("grad: `f` must take exactly one Vector argument".into());
            }
            Ok((params[0].0.clone(), (**body).clone()))
        }
        ExprKind::Var(name) => match fns.get(name) {
            Some(fd) if fd.params.len() == 1 => {
                Ok((fd.params[0].name.clone(), fd.body.clone()))
            }
            Some(_) => Err(format!(
                "grad: the function `{}` passed to grad must take exactly one Vector argument",
                name
            )),
            None => Err(format!(
                "grad: `f` must be a lambda literal or a top-level function, not `{}` \
                 (native backend); use the interpreter `aria run` for the general case",
                name
            )),
        },
        _ => Err(
            "grad: `f` must be a lambda literal `\\v -> ..` or a named top-level function \
             on the native backend; use the interpreter `aria run` for the general case"
                .into(),
        ),
    }
}

/// Emit the complete C definition of one grad helper. Returns the C source and
/// the ordered captured free variables `(name, is_vector)` the helper takes as
/// extra parameters after `x`.
fn compile_helper(
    helper: &str,
    param: &str,
    body: &Expr,
    fns: &HashMap<String, &FnDecl>,
) -> Result<(String, Vec<(String, bool)>), String> {
    let mut tr = Tracer {
        fns,
        scope: HashMap::new(),
        captures: HashMap::new(),
        capture_order: Vec::new(),
        cap_vecs: HashMap::new(),
        tmp: 0,
        body: String::new(),
        prologue: String::new(),
        inlining: Vec::new(),
    };
    // The input vector parameter is the traced leaf vector.
    tr.scope.insert(param.to_string(), TVal::Vec("$leaves".to_string()));
    let result = tr.trace(body)?;
    let out_id = match result {
        TVal::Scalar(s) => s,
        TVal::Vec(_) => {
            return Err("grad: `f` must return a Float (scalar), but it returns a Vector".into())
        }
        TVal::Int(_) | TVal::Bool(_) => {
            return Err(
                "grad: `f` must return a Float (differentiated scalar), but it returns a \
                 concrete Int/Bool"
                    .into(),
            )
        }
    };
    // The captured parameters, in discovery order.
    let captures: Vec<(String, bool)> = tr
        .capture_order
        .iter()
        .map(|n| (n.clone(), matches!(tr.captures[n], CapKind::Vector)))
        .collect();

    // Assemble the helper. `cfn` mangles the helper name identically to the
    // backend, so the emitted call resolves to this definition.
    let cname = crate::c_backend::cfn(helper);
    let mut out = String::new();
    let _ = writeln!(out, "/* reverse-mode autodiff: traced `grad` over the native AriaTape */");
    // Signature: `void* (void* x, <void* cap | double cap>...)`. Captured Vectors
    // are AriaVector* (void*); captured Floats are double.
    let mut sig = String::from("void* $x");
    for (name, is_vec) in &captures {
        if *is_vec {
            let _ = write!(sig, ", void* $cap_{}", name);
        } else {
            let _ = write!(sig, ", double $cap_{}", name);
        }
    }
    let _ = writeln!(out, "static void* {}({}) {{", cname, sig);
    out.push_str("    AriaTape $tp; aria_tape_init(&$tp);\n");
    out.push_str("    AriaVector* $xv = (AriaVector*)$x;\n");
    // One leaf per input coordinate.
    out.push_str("    AriaTVec $leaves = aria_tvec_alloc($xv->len);\n");
    out.push_str("    for (int64_t $i = 0; $i < $xv->len; $i++) $leaves.ids[$i] = aria_tape_leaf(&$tp, $xv->elems[$i]);\n");
    // Capture-Vector lifts, hoisted so they dominate every branch.
    out.push_str(&tr.prologue);
    // The traced body (declares its own temporaries).
    out.push_str(&tr.body);
    // Seed + sweep + extract gradient, then free everything traced.
    let _ = writeln!(out, "    aria_tape_backward(&$tp, {});", out_id);
    out.push_str("    void* $g = aria_tvec_to_grad(&$tp, &$leaves);\n");
    out.push_str("    aria_tvec_free(&$leaves);\n");
    out.push_str("    aria_tape_free(&$tp);\n");
    out.push_str("    aria_vec_drop($x);  /* grad consumes its input vector */\n");
    // Captured Vectors are passed by value (the call site dups them); release.
    for (name, is_vec) in &captures {
        if *is_vec {
            let _ = writeln!(out, "    aria_vec_drop($cap_{});", name);
        }
    }
    out.push_str("    return $g;\n");
    out.push_str("}\n");
    Ok((out, captures))
}

impl Tracer<'_> {
    fn fresh(&mut self) -> String {
        let n = self.tmp;
        self.tmp += 1;
        format!("$g{}", n)
    }

    /// Trace an expression, appending C statements that compute its tape value
    /// into a fresh local, and returning that local.
    fn trace(&mut self, e: &Expr) -> Result<TVal, String> {
        match &e.kind {
            // A scalar constant becomes a constant leaf.
            ExprKind::Float(f) => Ok(self.const_leaf(*f)),
            ExprKind::Int(n) => Ok(self.const_leaf(*n as f64)),
            ExprKind::Var(name) => {
                if let Some(tv) = self.scope.get(name) {
                    return Ok(tv.clone());
                }
                // An unknown name that is not a top-level function is a CAPTURED
                // free variable (a Vector or Float from the enclosing scope). Its
                // kind is resolved by the demanding context (`trace_scalar` /
                // `trace_vec`); a bare reference cannot disambiguate, so require
                // it to appear in a typed position.
                if self.fns.contains_key(name) {
                    return Err(format!(
                        "grad: `f` references the function `{}` as a value — not supported on \
                         the native backend",
                        name
                    ));
                }
                Err(format!(
                    "grad: the captured variable `{}` must be used directly as a Float or Vector \
                     argument in `f` on the native backend (e.g. `vec_dot(v, {})`)",
                    name, name
                ))
            }
            ExprKind::Unary(UnOp::Neg, inner) => {
                let x = self.trace_scalar(inner)?;
                let t = self.fresh();
                // r = -x : dr/dx = -1
                let _ = writeln!(
                    self.body,
                    "    int64_t {t} = aria_tape_unary(&$tp, {x}, -aria_tape_value(&$tp, {x}), -1.0);"
                );
                Ok(TVal::Scalar(t))
            }
            ExprKind::Binary(op, l, r) => self.trace_binary(*op, l, r),
            ExprKind::Call(name, args) => self.trace_call(name, args),
            ExprKind::Block(stmts, last) => {
                let saved = self.scope.clone();
                for s in stmts {
                    match &s.kind {
                        StmtKind::Let { name, value, .. } => {
                            let tv = self.trace(value)?;
                            self.scope.insert(name.clone(), tv);
                        }
                        StmtKind::Expr(ex) => {
                            // Evaluate for its (recorded) effect; discard result.
                            let _ = self.trace(ex)?;
                        }
                    }
                }
                let r = self.trace(last);
                self.scope = saved;
                r
            }
            ExprKind::Unary(UnOp::Not, _) => Err(
                "grad: boolean `!` is not differentiable in `f` on the native backend".into(),
            ),
            ExprKind::If(cond, then, els) => self.trace_if(cond, then, els),
            ExprKind::Match(scrut, arms) => self.trace_match(scrut, arms),
            _ => Err(format!(
                "grad: unsupported expression in `f` on the native backend: {}",
                describe(e)
            )),
        }
    }

    /// Resolve a bare `Var` that is a captured free variable, registering it
    /// with the demanded kind (and a fresh helper-parameter binding) on first
    /// use. Returns the bound `TVal` if `e` is such a var; `None` otherwise.
    fn try_capture(&mut self, e: &Expr, kind: CapKind) -> Option<Result<TVal, String>> {
        let name = match &e.kind {
            ExprKind::Var(n) if !self.scope.contains_key(n) && !self.fns.contains_key(n) => n,
            _ => return None,
        };
        // A capture's C parameter name mangles the source name safely.
        let cparam = format!("$cap_{}", name);
        match self.captures.get(name) {
            Some(CapKind::Vector) => match kind {
                CapKind::Vector => Some(Ok(TVal::Vec(self.cap_vecs[name].clone()))),
                CapKind::Scalar => Some(Err(format!(
                    "grad: captured `{}` is used as both a Vector and a Float in `f`",
                    name
                ))),
            },
            Some(CapKind::Scalar) => match kind {
                CapKind::Scalar => {
                    // A captured Float is lifted to a constant leaf on first need
                    // each time; bind a leaf node for this use.
                    let t = self.fresh();
                    let _ = writeln!(self.body, "    int64_t {t} = aria_tape_leaf(&$tp, {cparam});");
                    Some(Ok(TVal::Scalar(t)))
                }
                CapKind::Vector => Some(Err(format!(
                    "grad: captured `{}` is used as both a Float and a Vector in `f`",
                    name
                ))),
            },
            None => {
                self.captures.insert(name.clone(), kind);
                self.capture_order.push(name.clone());
                match kind {
                    CapKind::Vector => {
                        // Lift the captured AriaVector to a traced constant vector
                        // of leaves (its partials are discarded -- not an input),
                        // ONCE, into the PROLOGUE so the lifted `AriaTVec` local
                        // dominates every later reference (including ones in other
                        // `if`/`match` branches); reuse it for any later reference.
                        let t = self.fresh();
                        let _ = writeln!(
                            self.prologue,
                            "    AriaTVec {t} = aria_tvec_from_vec(&$tp, {cparam});"
                        );
                        self.cap_vecs.insert(name.clone(), t.clone());
                        Some(Ok(TVal::Vec(t)))
                    }
                    CapKind::Scalar => {
                        let t = self.fresh();
                        let _ = writeln!(self.body, "    int64_t {t} = aria_tape_leaf(&$tp, {cparam});");
                        Some(Ok(TVal::Scalar(t)))
                    }
                }
            }
        }
    }

    /// Trace `e`, requiring it to be a scalar (tape node-id).
    fn trace_scalar(&mut self, e: &Expr) -> Result<String, String> {
        if let Some(r) = self.try_capture(e, CapKind::Scalar) {
            return r.map(|tv| match tv {
                TVal::Scalar(s) => s,
                _ => unreachable!("scalar capture yields a Scalar TVal"),
            });
        }
        match self.trace(e)? {
            TVal::Scalar(s) => Ok(s),
            TVal::Vec(_) => Err(
                "grad: expected a Float (scalar) but got a Vector in `f` on the native backend"
                    .into(),
            ),
            TVal::Int(_) | TVal::Bool(_) => Err(
                "grad: a concrete Int/Bool (e.g. from `vec_len`) cannot be used as a \
                 differentiated Float in `f` on the native backend"
                    .into(),
            ),
        }
    }

    /// Trace `e`, requiring it to be a traced Vector.
    fn trace_vec(&mut self, e: &Expr) -> Result<String, String> {
        if let Some(r) = self.try_capture(e, CapKind::Vector) {
            return r.map(|tv| match tv {
                TVal::Vec(v) => v,
                _ => unreachable!("vector capture yields a Vec TVal"),
            });
        }
        match self.trace(e)? {
            TVal::Vec(v) => Ok(v),
            TVal::Scalar(_) => Err(
                "grad: expected a Vector but got a Float (scalar) in `f` on the native backend"
                    .into(),
            ),
            TVal::Int(_) | TVal::Bool(_) => Err(
                "grad: expected a Vector but got a concrete Int/Bool in `f` on the native backend"
                    .into(),
            ),
        }
    }

    /// A fresh constant-leaf scalar holding `v`.
    fn const_leaf(&mut self, v: f64) -> TVal {
        let t = self.fresh();
        let _ = writeln!(self.body, "    int64_t {t} = aria_tape_leaf(&$tp, {});", fmt_f64(v));
        TVal::Scalar(t)
    }

    fn trace_binary(&mut self, op: BinOp, l: &Expr, r: &Expr) -> Result<TVal, String> {
        let a = self.trace_scalar(l)?;
        let b = self.trace_scalar(r)?;
        let t = self.fresh();
        let (av, bv) = (
            format!("aria_tape_value(&$tp, {})", a),
            format!("aria_tape_value(&$tp, {})", b),
        );
        match op {
            // (a+b)' : da=1, db=1
            BinOp::Add => {
                let _ = writeln!(
                    self.body,
                    "    int64_t {t} = aria_tape_binary(&$tp, {a}, {b}, {av} + {bv}, 1.0, 1.0);"
                );
            }
            // (a-b)' : da=1, db=-1
            BinOp::Sub => {
                let _ = writeln!(
                    self.body,
                    "    int64_t {t} = aria_tape_binary(&$tp, {a}, {b}, {av} - {bv}, 1.0, -1.0);"
                );
            }
            // (a*b)' : da=b, db=a
            BinOp::Mul => {
                let _ = writeln!(
                    self.body,
                    "    int64_t {t} = aria_tape_binary(&$tp, {a}, {b}, {av} * {bv}, {bv}, {av});"
                );
            }
            // (a/b)' : da=1/b, db=-a/b^2 ; guard b==0 with a clean runtime trap.
            BinOp::Div => {
                let _ = writeln!(
                    self.body,
                    "    if ({bv} == 0.0) aria_trap_msg(\"grad: division by zero in a differentiated value\");"
                );
                let _ = writeln!(
                    self.body,
                    "    int64_t {t} = aria_tape_binary(&$tp, {a}, {b}, {av} / {bv}, 1.0 / ({bv}), -({av}) / (({bv}) * ({bv})));"
                );
            }
            _ => {
                return Err(format!(
                    "grad: operator `{:?}` is not differentiable in `f` on the native backend \
                     (only + - * / and unary -)",
                    op
                ))
            }
        }
        Ok(TVal::Scalar(t))
    }

    fn trace_call(&mut self, name: &str, args: &[Expr]) -> Result<TVal, String> {
        match name {
            "vec_get" => {
                if args.len() != 2 {
                    return Err("grad: vec_get expects (Vector, Int)".into());
                }
                let v = self.trace_vec(&args[0])?;
                // The index must be a compile-time-evaluable constant integer
                // expression of plain ints (it indexes the leaf array). Most
                // differentiable code uses a literal; support literals + simple
                // int arithmetic via a small evaluator.
                let idx = eval_const_int(&args[1]).ok_or_else(|| {
                    "grad: vec_get index must be a constant Int in `f` on the native backend"
                        .to_string()
                })?;
                let t = self.fresh();
                let _ = writeln!(
                    self.body,
                    "    if ({idx} < 0 || {idx} >= {v}.len) aria_trap_msg(\"vector index out of range\");"
                );
                // r = x[i]; identity, partial 1.
                let _ = writeln!(
                    self.body,
                    "    int64_t {t} = aria_tape_unary(&$tp, {v}.ids[{idx}], aria_tape_value(&$tp, {v}.ids[{idx}]), 1.0);"
                );
                Ok(TVal::Scalar(t))
            }
            "vec_dot" => {
                if args.len() != 2 {
                    return Err("grad: vec_dot expects (Vector, Vector)".into());
                }
                let a = self.trace_vec(&args[0])?;
                let b = self.trace_vec(&args[1])?;
                let t = self.fresh();
                let _ = writeln!(
                    self.body,
                    "    int64_t {t} = aria_trace_dot(&$tp, &{a}, &{b});"
                );
                Ok(TVal::Scalar(t))
            }
            "vec_norm" => {
                if args.len() != 1 {
                    return Err("grad: vec_norm expects (Vector)".into());
                }
                let a = self.trace_vec(&args[0])?;
                let t = self.fresh();
                let _ = writeln!(self.body, "    int64_t {t} = aria_trace_norm(&$tp, &{a});");
                Ok(TVal::Scalar(t))
            }
            "vec_add" | "vec_sub" => {
                if args.len() != 2 {
                    return Err(format!("grad: {} expects (Vector, Vector)", name));
                }
                let a = self.trace_vec(&args[0])?;
                let b = self.trace_vec(&args[1])?;
                let t = self.fresh();
                let (rv, db) = if name == "vec_sub" {
                    ("av - bv", "-1.0")
                } else {
                    ("av + bv", "1.0")
                };
                let _ = writeln!(self.body, "    AriaTVec {t} = aria_tvec_alloc({a}.len);");
                let _ = writeln!(
                    self.body,
                    "    if ({a}.len != {b}.len) aria_trap_msg(\"vector length mismatch\");"
                );
                let _ = writeln!(
                    self.body,
                    "    for (int64_t $j = 0; $j < {a}.len; $j++) {{ \
                     double av = aria_tape_value(&$tp, {a}.ids[$j]), bv = aria_tape_value(&$tp, {b}.ids[$j]); \
                     {t}.ids[$j] = aria_tape_binary(&$tp, {a}.ids[$j], {b}.ids[$j], {rv}, 1.0, {db}); }}"
                );
                Ok(TVal::Vec(t))
            }
            "vec_scale" => {
                if args.len() != 2 {
                    return Err("grad: vec_scale expects (Vector, Float)".into());
                }
                let a = self.trace_vec(&args[0])?;
                let s = self.trace_scalar(&args[1])?;
                let t = self.fresh();
                let _ = writeln!(self.body, "    AriaTVec {t} = aria_tvec_alloc({a}.len);");
                // r_i = s * a_i ; dr/da_i = s, dr/ds = a_i.
                let _ = writeln!(
                    self.body,
                    "    for (int64_t $j = 0; $j < {a}.len; $j++) {{ \
                     double av = aria_tape_value(&$tp, {a}.ids[$j]), sv = aria_tape_value(&$tp, {s}); \
                     {t}.ids[$j] = aria_tape_binary(&$tp, {a}.ids[$j], {s}, sv * av, sv, av); }}"
                );
                Ok(TVal::Vec(t))
            }
            "vec_from_array" => {
                if args.len() != 1 {
                    return Err("grad: vec_from_array expects (Array[Float])".into());
                }
                // Support only an inline array literal of scalar exprs (the common
                // case `vec_from_array([..])`). A non-literal array would require
                // tracing array values, which is outside the supported subset.
                let elems = array_literal_elems(&args[0]).ok_or_else(|| {
                    "grad: vec_from_array in `f` requires an inline array literal `[..]` on the \
                     native backend"
                        .to_string()
                })?;
                let mut ids = Vec::with_capacity(elems.len());
                for el in &elems {
                    ids.push(self.trace_scalar(el)?);
                }
                let t = self.fresh();
                let _ = writeln!(self.body, "    AriaTVec {t} = aria_tvec_alloc({});", ids.len());
                for (i, id) in ids.iter().enumerate() {
                    let _ = writeln!(self.body, "    {t}.ids[{i}] = {id};");
                }
                Ok(TVal::Vec(t))
            }
            "vec_push" => {
                if args.len() != 2 {
                    return Err("grad: vec_push expects (Vector, Float)".into());
                }
                let v = self.trace_vec(&args[0])?;
                let x = self.trace_scalar(&args[1])?;
                let t = self.fresh();
                let _ = writeln!(self.body, "    AriaTVec {t} = aria_tvec_alloc({v}.len + 1);");
                let _ = writeln!(
                    self.body,
                    "    for (int64_t $j = 0; $j < {v}.len; $j++) {t}.ids[$j] = {v}.ids[$j];"
                );
                let _ = writeln!(self.body, "    {t}.ids[{v}.len] = {x};");
                Ok(TVal::Vec(t))
            }
            "vec_len" => {
                // `vec_len` yields a CONCRETE Int (the interpreter returns a plain
                // `Value::Int` for `vec_len` of a TracingVec — line 1429 of
                // interp.rs). It is non-differentiable; it may feed `if`/`match`
                // conditions, a constant-foldable index, or a `let` binding. We
                // realize it into an `int64_t` C local.
                if args.len() != 1 {
                    return Err("grad: vec_len expects (Vector)".into());
                }
                let v = self.trace_vec(&args[0])?;
                let t = self.fresh();
                let _ = writeln!(self.body, "    int64_t {t} = {v}.len;");
                Ok(TVal::Int(t))
            }
            other => {
                // A call to a user function: INLINE it (trace through its body so
                // its ops record tape nodes), exactly reproducing the interpreter,
                // which steps into the callee with the same argument values.
                if self.fns.contains_key(other) {
                    self.trace_inline_call(other, args)
                } else {
                    Err(format!(
                        "grad: unsupported operation `{}` inside `f` on the native backend",
                        other
                    ))
                }
            }
        }
    }

    /// Inline a call `g(args..)` to a user function: bind each parameter to the
    /// traced value of the corresponding argument, then trace `g`'s body in that
    /// scope. This reproduces the interpreter EXACTLY (it evaluates the callee
    /// with the same argument values, recording the same tape nodes in the same
    /// order). Recursion is gated cleanly (it cannot be inlined into a finite,
    /// straight-line trace). Captures are resolved in the CALLER's scope before
    /// entering the callee, so the callee sees them as ordinary bound locals.
    fn trace_inline_call(&mut self, name: &str, args: &[Expr]) -> Result<TVal, String> {
        let fd = *self
            .fns
            .get(name)
            .ok_or_else(|| format!("grad: internal: unknown function `{}`", name))?;
        if self.inlining.iter().any(|n| n == name) {
            return Err(format!(
                "grad: `f` (transitively) calls the recursive function `{}` — recursion inside \
                 `f` cannot be inlined on the native backend; use the interpreter `aria run`",
                name
            ));
        }
        if self.inlining.len() >= MAX_INLINE_DEPTH {
            return Err(format!(
                "grad: inlining inside `f` exceeded depth {} at `{}` on the native backend; use \
                 the interpreter `aria run`",
                MAX_INLINE_DEPTH, name
            ));
        }
        if fd.params.len() != args.len() {
            return Err(format!(
                "grad: call to `{}` inside `f` has {} argument(s) but `{}` takes {}",
                name,
                args.len(),
                name,
                fd.params.len()
            ));
        }
        // Trace each argument FIRST, in the caller's scope (left-to-right, as the
        // interpreter evaluates arguments). A captured free var used only as an
        // argument is resolved here via try_capture so the kind is demanded by
        // the callee's parameter type would be unknown — instead we trace the arg
        // generically and bind whatever value (Scalar/Vec/Int/Bool) results.
        let mut bound: Vec<(String, TVal)> = Vec::with_capacity(args.len());
        for (p, a) in fd.params.iter().zip(args) {
            let tv = self.trace_arg(a)?;
            bound.push((p.name.clone(), tv));
        }
        // Enter the callee: a fresh scope holding ONLY the bound parameters (a
        // top-level function closes over nothing but globals, which are resolved
        // as captures/functions, not locals). Restore the caller scope after.
        let saved = std::mem::take(&mut self.scope);
        for (n, tv) in bound {
            self.scope.insert(n, tv);
        }
        self.inlining.push(name.to_string());
        let r = self.trace(&fd.body);
        self.inlining.pop();
        self.scope = saved;
        r
    }

    /// Trace a call argument whose demanded kind is not known up front (it is the
    /// callee parameter's type). Resolve a bare captured free var by peeking: a
    /// capture used as a function argument must already have a known kind, OR be
    /// disambiguated inside the callee. Since we cannot know here, we trace it as
    /// an ordinary expression; a bare unknown `Var` falls through to the capture
    /// machinery via the demanding op inside the callee. To keep that working, a
    /// bare captured Var passed straight through is deferred: we look it up as a
    /// Vector capture by default only if it is ALREADY known; otherwise we error
    /// with the standard "use it directly" message (same as `trace`).
    fn trace_arg(&mut self, e: &Expr) -> Result<TVal, String> {
        // A bare captured Vector/Float var as an argument: if its kind is already
        // discovered, bind that; if not, we cannot infer it from the call site,
        // so we require it to be used in a typed position (the same constraint as
        // a top-level bare capture). Most code passes `vec_*`/arithmetic exprs.
        if let ExprKind::Var(n) = &e.kind {
            if !self.scope.contains_key(n) && !self.fns.contains_key(n) {
                if let Some(kind) = self.captures.get(n).copied() {
                    return match kind {
                        CapKind::Vector => self.try_capture(e, CapKind::Vector).unwrap(),
                        CapKind::Scalar => self.try_capture(e, CapKind::Scalar).unwrap(),
                    };
                }
            }
        }
        self.trace(e)
    }

    /// Trace an `if`: evaluate the condition to a CONCRETE Bool from the forward
    /// computation (never a traced node — the interpreter likewise computes the
    /// condition concretely and a comparison on a differentiated value is an
    /// error in both backends), emit a real C `if`/`else`, and trace BOTH arms,
    /// each into its own buffer, assigning the chosen arm's traced value to a
    /// shared result local. Only the taken arm's tape nodes are recorded at
    /// runtime — exactly the interpreter's behaviour (it traces only the taken
    /// branch). Both arms must yield the SAME kind (Float or Vector).
    fn trace_if(&mut self, cond: &Expr, then: &Expr, els: &Expr) -> Result<TVal, String> {
        let c = self.forward_bool(cond)?;
        let (then_buf, then_val) = self.sub_trace(then)?;
        let (else_buf, else_val) = self.sub_trace(els)?;
        let result = self.join_branches(
            &[(format!("if ({c})"), then_buf, then_val), ("else".to_string(), else_buf, else_val)],
        )?;
        Ok(result)
    }

    /// Trace a `match` on a CONCRETE scalar scrutinee (Int/Bool/Float) using
    /// literal / wildcard / var patterns, lowered to a C if-else-if chain on the
    /// forward scrutinee value. ADT/record scrutinees (which would require
    /// tracing a traced ADT — outside the differentiable subset) stay gated.
    fn trace_match(&mut self, scrut: &Expr, arms: &[Arm]) -> Result<TVal, String> {
        if arms.is_empty() {
            return Err("grad: empty `match` inside `f`".into());
        }
        let sv = self.forward_eval(scrut)?;
        // Build (guard, body-buffer, body-value) per arm. A wildcard / var pattern
        // is the catch-all `else`. Each var pattern binds the concrete scrutinee.
        let mut branches: Vec<(String, String, TVal)> = Vec::with_capacity(arms.len());
        let mut saw_default = false;
        for arm in arms {
            if saw_default {
                // Arms after an irrefutable one are dead (the interpreter takes
                // the first match); reject so we never silently drop a branch.
                return Err(
                    "grad: `match` inside `f` has arm(s) after an irrefutable pattern on the \
                     native backend"
                        .into(),
                );
            }
            let guard = match &arm.pat.kind {
                PatternKind::Wild => {
                    saw_default = true;
                    "else".to_string()
                }
                PatternKind::Var(name) => {
                    // Bind the concrete scrutinee to `name` for this arm's body.
                    saw_default = true;
                    let (buf, val) = self.sub_trace_bound(&arm.body, name, &sv)?;
                    branches.push(("else".to_string(), buf, val));
                    continue;
                }
                PatternKind::Int(n) => match &sv {
                    FVal::Int(s) => format!("if ({s} == {n})"),
                    FVal::Bool(s) => format!("if ({s} == {n})"),
                    FVal::Float(_) => {
                        return Err(
                            "grad: `match` integer pattern against a Float scrutinee inside `f`"
                                .into(),
                        )
                    }
                },
                PatternKind::Bool(b) => match &sv {
                    FVal::Bool(s) => format!("if ({s} == {})", if *b { 1 } else { 0 }),
                    _ => {
                        return Err(
                            "grad: `match` boolean pattern against a non-Bool scrutinee inside `f`"
                                .into(),
                        )
                    }
                },
                PatternKind::Ctor(_, _) | PatternKind::Record(_, _) => {
                    return Err(
                        "grad: `match` on constructors/records inside `f` is not supported on the \
                         native backend (only Int/Bool/wildcard/var patterns on a concrete \
                         scalar scrutinee); use the interpreter `aria run`"
                            .into(),
                    )
                }
            };
            let (buf, val) = self.sub_trace(&arm.body)?;
            branches.push((guard, buf, val));
        }
        if !saw_default {
            // No irrefutable arm: a non-matching scrutinee would have no value.
            // The interpreter raises a runtime "no match arm" error; mirror it
            // with a runtime trap in a final `else`, after the typed arms.
            return self.join_branches_with_trap(&branches);
        }
        self.join_branches(&branches)
    }

    /// Trace `e` into a FRESH buffer (not appended to `self.body`), returning the
    /// buffer and the resulting value. Temporaries still draw from the shared
    /// counter so names stay globally unique across branches. The scope is saved
    /// and restored so branch-local `let`s do not leak.
    fn sub_trace(&mut self, e: &Expr) -> Result<(String, TVal), String> {
        let saved_body = std::mem::take(&mut self.body);
        let saved_scope = self.scope.clone();
        let r = self.trace(e);
        self.scope = saved_scope;
        let branch_body = std::mem::replace(&mut self.body, saved_body);
        Ok((branch_body, r?))
    }

    /// Like `sub_trace`, but first binds `name` to a copy of the concrete forward
    /// scrutinee `sv` (for a `match` var pattern) inside the branch buffer.
    fn sub_trace_bound(
        &mut self,
        e: &Expr,
        name: &str,
        sv: &FVal,
    ) -> Result<(String, TVal), String> {
        let saved_body = std::mem::take(&mut self.body);
        let saved_scope = self.scope.clone();
        // Materialize the scrutinee into a fresh local and bind `name` to it.
        let tv = match sv {
            FVal::Int(s) => {
                let t = self.fresh();
                let _ = writeln!(self.body, "    int64_t {t} = {s};");
                TVal::Int(t)
            }
            FVal::Bool(s) => {
                let t = self.fresh();
                let _ = writeln!(self.body, "    int {t} = {s};");
                TVal::Bool(t)
            }
            FVal::Float(s) => {
                // A Float scrutinee bound by a var pattern is concrete (forward),
                // not differentiated; it cannot feed the tape. Reject if used as a
                // differentiated Float — but binding it as a concrete value is fine
                // only for conditions. Lift to a constant leaf so simple identity
                // uses still differentiate to zero contribution (it is not an
                // input). We bind it as a Scalar leaf to keep arithmetic valid.
                let t = self.fresh();
                let _ = writeln!(self.body, "    int64_t {t} = aria_tape_leaf(&$tp, {s});");
                TVal::Scalar(t)
            }
        };
        self.scope.insert(name.to_string(), tv);
        let r = self.trace(e);
        self.scope = saved_scope;
        let branch_body = std::mem::replace(&mut self.body, saved_body);
        Ok((branch_body, r?))
    }

    /// Emit a C if/else chain over pre-traced branch buffers, each assigning its
    /// traced value to ONE shared result local, and return that local as a TVal.
    /// All branches must produce the same kind. The first element's guard is the
    /// leading `if (..)`; subsequent guards are `else if (..)` or `else`.
    fn join_branches(&mut self, branches: &[(String, String, TVal)]) -> Result<TVal, String> {
        self.emit_join(branches, false)
    }

    /// Like `join_branches`, but appends a final `else` that traps at runtime
    /// (for a non-exhaustive `match` with no irrefutable arm — mirrors the
    /// interpreter's "no match arm" runtime error).
    fn join_branches_with_trap(&mut self, branches: &[(String, String, TVal)]) -> Result<TVal, String> {
        self.emit_join(branches, true)
    }

    fn emit_join(
        &mut self,
        branches: &[(String, String, TVal)],
        trap_default: bool,
    ) -> Result<TVal, String> {
        // Determine the shared result kind (Scalar or Vec) — all arms must agree.
        let kind_is_vec = match &branches[0].2 {
            TVal::Vec(_) => true,
            TVal::Scalar(_) => false,
            TVal::Int(_) | TVal::Bool(_) => {
                return Err(
                    "grad: an `if`/`match` arm inside `f` produces a concrete Int/Bool, not a \
                     differentiated Float/Vector, on the native backend"
                        .into(),
                )
            }
        };
        for (_, _, v) in branches {
            let v_is_vec = match v {
                TVal::Vec(_) => true,
                TVal::Scalar(_) => false,
                TVal::Int(_) | TVal::Bool(_) => {
                    return Err(
                        "grad: an `if`/`match` arm inside `f` produces a concrete Int/Bool, not a \
                         differentiated Float/Vector, on the native backend"
                            .into(),
                    )
                }
            };
            if v_is_vec != kind_is_vec {
                return Err(
                    "grad: the arms of an `if`/`match` inside `f` must all produce the SAME kind \
                     (all Float, or all Vector) on the native backend"
                        .into(),
                );
            }
        }
        // The shared result local. A Vector result needs its length known to be
        // consistent; we store the AriaTVec produced by the taken arm directly.
        let res = self.fresh();
        if kind_is_vec {
            let _ = writeln!(self.body, "    AriaTVec {res};");
        } else {
            let _ = writeln!(self.body, "    int64_t {res};");
        }
        for (i, (guard, buf, val)) in branches.iter().enumerate() {
            let head = if i == 0 {
                format!("    {} {{", guard)
            } else if guard == "else" {
                "    else {".to_string()
            } else {
                // `else if (..)`
                format!("    else {} {{", guard)
            };
            let _ = writeln!(self.body, "{}", head);
            self.body.push_str(buf);
            let assign = match val {
                TVal::Vec(v) => format!("        {res} = {v};"),
                TVal::Scalar(s) => format!("        {res} = {s};"),
                _ => unreachable!("kind checked above"),
            };
            let _ = writeln!(self.body, "{}", assign);
            self.body.push_str("    }\n");
        }
        if trap_default {
            self.body.push_str(
                "    else { aria_trap_msg(\"grad: no matching `match` arm (non-exhaustive)\"); }\n",
            );
        }
        if kind_is_vec {
            Ok(TVal::Vec(res))
        } else {
            Ok(TVal::Scalar(res))
        }
    }

    /// Forward-evaluate `e` to a CONCRETE Bool C expression (for an `if`
    /// condition). Mirrors the interpreter: the condition must reduce to a plain
    /// Bool from the forward computation; a comparison on a differentiated
    /// (traced) value is a clean error in both backends.
    fn forward_bool(&mut self, e: &Expr) -> Result<String, String> {
        match self.forward_eval(e)? {
            FVal::Bool(s) => Ok(s),
            FVal::Int(_) | FVal::Float(_) => Err(
                "grad: an `if` condition inside `f` must be a Bool on the native backend".into(),
            ),
        }
    }

    /// Forward-evaluate a NON-differentiated expression to a concrete C value
    /// (Int/Bool/Float). This is the strict analogue of the interpreter
    /// evaluating a condition/scrutinee to a plain `Value`. It NEVER reads a
    /// traced node's adjoint and NEVER records on the tape; any subexpression
    /// that is a traced (differentiated) scalar/vector is rejected, exactly as
    /// the interpreter rejects a comparison/arithmetic that mixes a `Tracing`
    /// value into a control-flow decision.
    fn forward_eval(&mut self, e: &Expr) -> Result<FVal, String> {
        match &e.kind {
            ExprKind::Int(n) => Ok(FVal::Int(format!("(int64_t){}", n))),
            ExprKind::Bool(b) => Ok(FVal::Bool(if *b { "1".into() } else { "0".into() })),
            ExprKind::Float(f) => Ok(FVal::Float(fmt_f64(*f))),
            ExprKind::Var(name) => {
                // A local concrete Int/Bool binding (e.g. a `let n = vec_len(x)`).
                if let Some(tv) = self.scope.get(name) {
                    return match tv {
                        TVal::Int(s) => Ok(FVal::Int(s.clone())),
                        TVal::Bool(s) => Ok(FVal::Bool(s.clone())),
                        TVal::Scalar(_) | TVal::Vec(_) => Err(format!(
                            "grad: the differentiated value `{}` cannot be used in an `if`/`match` \
                             condition inside `f` (the branch must not depend on a value being \
                             differentiated) on the native backend",
                            name
                        )),
                    };
                }
                // A captured concrete free variable. Its kind is whatever the
                // enclosing scope holds; for conditions we accept a captured Int,
                // Bool, or Float (a captured *Vector* is not a scalar condition).
                if self.fns.contains_key(name) {
                    return Err(format!(
                        "grad: `f` references the function `{}` in a condition — not supported \
                         on the native backend",
                        name
                    ));
                }
                // Register/lookup as a SCALAR capture and read its concrete C
                // parameter. We do not know Int vs Float a priori; captured
                // condition variables are passed as `double $cap_<name>` (the
                // Scalar capture ABI), so compare as a double. An Int capture is
                // exact as a double for the magnitudes used in control flow.
                let cparam = format!("$cap_{}", name);
                match self.captures.get(name).copied() {
                    Some(CapKind::Scalar) => Ok(FVal::Float(cparam)),
                    Some(CapKind::Vector) => Err(format!(
                        "grad: captured Vector `{}` cannot be used directly in an `if`/`match` \
                         condition inside `f` on the native backend (use `vec_len({})`)",
                        name, name
                    )),
                    None => {
                        self.captures.insert(name.clone(), CapKind::Scalar);
                        self.capture_order.push(name.clone());
                        Ok(FVal::Float(cparam))
                    }
                }
            }
            ExprKind::Unary(UnOp::Neg, inner) => match self.forward_eval(inner)? {
                FVal::Int(s) => Ok(FVal::Int(format!("(-({}))", s))),
                FVal::Float(s) => Ok(FVal::Float(format!("(-({}))", s))),
                FVal::Bool(_) => Err("grad: cannot negate a Bool in a condition inside `f`".into()),
            },
            ExprKind::Unary(UnOp::Not, inner) => match self.forward_eval(inner)? {
                FVal::Bool(s) => Ok(FVal::Bool(format!("(!({}))", s))),
                _ => Err("grad: `!` expects a Bool in a condition inside `f`".into()),
            },
            ExprKind::Binary(op, l, r) => self.forward_binary(*op, l, r),
            ExprKind::Call(name, args) if name == "vec_len" => {
                if args.len() != 1 {
                    return Err("grad: vec_len expects (Vector)".into());
                }
                // `vec_len` reads the length of a TRACED or captured vector — a
                // concrete Int, with no dependence on the differentiated values.
                let v = self.trace_vec(&args[0])?;
                Ok(FVal::Int(format!("{}.len", v)))
            }
            ExprKind::Block(stmts, last) => {
                // A condition may be a small block of concrete `let`s. Bind each
                // (concrete) value and forward-eval the result.
                let saved = self.scope.clone();
                for s in stmts {
                    match &s.kind {
                        StmtKind::Let { name: n, value: v, .. } => {
                            let fv = self.forward_eval(v)?;
                            let t = self.fresh();
                            let tv = match fv {
                                FVal::Int(e) => {
                                    let _ = writeln!(self.body, "    int64_t {t} = {e};");
                                    TVal::Int(t)
                                }
                                FVal::Bool(e) => {
                                    let _ = writeln!(self.body, "    int {t} = {e};");
                                    TVal::Bool(t)
                                }
                                FVal::Float(_) => {
                                    // A concrete-Float `let` inside a condition is
                                    // rare and not tracked in `scope` (which only
                                    // holds Int/Bool/traced kinds); gate cleanly.
                                    self.scope = saved;
                                    return Err(
                                        "grad: a `let`-bound Float inside an `if`/`match` \
                                         condition is not supported on the native backend"
                                            .into(),
                                    );
                                }
                            };
                            self.scope.insert(n.clone(), tv);
                        }
                        StmtKind::Expr(_) => {
                            self.scope = saved;
                            return Err(
                                "grad: a statement-expression inside an `if`/`match` condition is \
                                 not supported on the native backend"
                                    .into(),
                            );
                        }
                    }
                }
                let r = self.forward_eval(last);
                self.scope = saved;
                r
            }
            _ => Err(
                "grad: this `if`/`match` condition/scrutinee is not a supported concrete \
                 Int/Bool/Float expression on the native backend (supported: literals, captured \
                 scalars, `vec_len`, and +-*/%, comparisons, &&/||/! over them)"
                    .into(),
            ),
        }
    }

    /// Forward-evaluate a concrete binary op (arithmetic / comparison / logic) on
    /// concrete operands. Comparisons yield a Bool; arithmetic preserves Int or
    /// Float; logic requires Bools. No tape interaction whatsoever.
    fn forward_binary(&mut self, op: BinOp, l: &Expr, r: &Expr) -> Result<FVal, String> {
        // Short-circuit logic.
        if matches!(op, BinOp::And | BinOp::Or) {
            let a = self.forward_eval(l)?;
            let b = self.forward_eval(r)?;
            let (FVal::Bool(a), FVal::Bool(b)) = (a, b) else {
                return Err("grad: `&&`/`||` expect Bools in a condition inside `f`".into());
            };
            let cop = if matches!(op, BinOp::And) { "&&" } else { "||" };
            return Ok(FVal::Bool(format!("(({}) {} ({}))", a, cop, b)));
        }
        let a = self.forward_eval(l)?;
        let b = self.forward_eval(r)?;
        // Determine a common numeric form for arithmetic/comparison.
        let cop = match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::And | BinOp::Or => unreachable!("handled above"),
        };
        let is_cmp = matches!(
            op,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
        );
        match (&a, &b) {
            (FVal::Int(x), FVal::Int(y)) => {
                if matches!(op, BinOp::Div | BinOp::Mod) {
                    // Guard division/modulo by zero with a clean runtime trap,
                    // matching the interpreter's catchable error.
                    let _ = writeln!(
                        self.body,
                        "    if (({}) == 0) aria_trap_msg(\"grad: division by zero in a condition\");",
                        y
                    );
                }
                let expr = format!("(({}) {} ({}))", x, cop, y);
                Ok(if is_cmp { FVal::Bool(expr) } else { FVal::Int(expr) })
            }
            // Any Float involved -> compare/compute as double. (Int promotes.)
            (FVal::Float(_) | FVal::Int(_), FVal::Float(_) | FVal::Int(_)) => {
                if matches!(op, BinOp::Mod) {
                    return Err("grad: `%` is not defined on Floats in a condition inside `f`".into());
                }
                let xe = fval_as_double(&a);
                let ye = fval_as_double(&b);
                let expr = format!("(({}) {} ({}))", xe, cop, ye);
                Ok(if is_cmp { FVal::Bool(expr) } else { FVal::Float(expr) })
            }
            (FVal::Bool(x), FVal::Bool(y)) if matches!(op, BinOp::Eq | BinOp::Ne) => {
                Ok(FVal::Bool(format!("(({}) {} ({}))", x, cop, y)))
            }
            _ => Err(
                "grad: mismatched operand kinds in an `if`/`match` condition inside `f` on the \
                 native backend"
                    .into(),
            ),
        }
    }
}

/// Render a concrete forward value as a C `double` expression (Int widens).
fn fval_as_double(v: &FVal) -> String {
    match v {
        FVal::Float(s) => s.clone(),
        FVal::Int(s) => format!("(double)({})", s),
        FVal::Bool(s) => format!("(double)({})", s),
    }
}

/// Format an f64 as a C `double` literal that round-trips exactly (hex float
/// would be ideal but is not portable across all C compilers; the shortest
/// decimal that round-trips is exact for IEEE-754 doubles via `{:?}`/`ryu`-style
/// Rust formatting, which we then make a valid C literal).
fn fmt_f64(v: f64) -> String {
    if v.is_infinite() {
        return if v > 0.0 { "(1.0/0.0)".into() } else { "(-1.0/0.0)".into() };
    }
    if v.is_nan() {
        return "(0.0/0.0)".into();
    }
    let s = format!("{:?}", v); // Rust's shortest round-tripping form, e.g. "2.0"
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{}.0", s)
    }
}

/// Evaluate a constant integer index expression (literal, or simple +,-,* of
/// literals). Returns `None` for anything non-constant.
fn eval_const_int(e: &Expr) -> Option<i64> {
    match &e.kind {
        ExprKind::Int(n) => Some(*n),
        ExprKind::Unary(UnOp::Neg, inner) => eval_const_int(inner).map(|n| -n),
        ExprKind::Binary(op, l, r) => {
            let a = eval_const_int(l)?;
            let b = eval_const_int(r)?;
            match op {
                BinOp::Add => a.checked_add(b),
                BinOp::Sub => a.checked_sub(b),
                BinOp::Mul => a.checked_mul(b),
                BinOp::Div if b != 0 => Some(a / b),
                _ => None,
            }
        }
        _ => None,
    }
}

/// If `e` is an inline array literal (`[..]`), return its element expressions.
/// Array literals lower to a `Call("array_lit", elems)` (or a monomorphized
/// `array_lit$f` form) by the time the C backend sees them.
fn array_literal_elems(e: &Expr) -> Option<Vec<Expr>> {
    match &e.kind {
        ExprKind::Call(name, args) if name == "array_lit" || name.starts_with("array_lit$") => {
            Some(args.clone())
        }
        _ => None,
    }
}

/// A short human description of an expression kind, for clean error messages.
fn describe(e: &Expr) -> &'static str {
    match &e.kind {
        ExprKind::Record(_, _) => "a record literal",
        ExprKind::Field(_, _) => "a record field access",
        ExprKind::Update(_, _) => "a record update",
        ExprKind::Lambda(_, _, _) => "a lambda",
        ExprKind::Apply(_, _, _) => "a closure application",
        ExprKind::Ctor(_, _) => "a constructor",
        ExprKind::Str(_) => "a String",
        ExprKind::Bool(_) => "a Bool",
        ExprKind::Unit => "Unit",
        _ => "an unsupported construct",
    }
}
