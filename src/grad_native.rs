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
//!   * `let` bindings (in a `Block`) of either kind.
//! GATED (clean error): `if`/`match`, calls to user functions or any other
//! builtin, records/tuples, closures, and any op on a non-{Float,Vector} value.
//! `f` itself may be a lambda literal `\v -> ..` OR a named top-level function
//! `grad(loss, x)` (its body is inlined) — as long as that body is straight-line
//! over the supported subset.

use std::collections::HashMap;
use std::fmt::Write;

use crate::ast::{BinOp, Expr, ExprKind, FnDecl, Item, Program, StmtKind, UnOp};

/// The C value produced by a traced expression: a scalar tape node-id, or a
/// traced vector (an `AriaTVec` of node-ids). Both are held in fresh C locals
/// whose names this enum carries.
#[derive(Clone)]
enum TVal {
    /// A scalar tape node: `int64_t <name>` holding a node-id.
    Scalar(String),
    /// A traced vector: `AriaTVec <name>` of node-ids.
    Vec(String),
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
}

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
    };
    // The input vector parameter is the traced leaf vector.
    tr.scope.insert(param.to_string(), TVal::Vec("$leaves".to_string()));
    let result = tr.trace(body)?;
    let out_id = match result {
        TVal::Scalar(s) => s,
        TVal::Vec(_) => {
            return Err("grad: `f` must return a Float (scalar), but it returns a Vector".into())
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
    // The traced body (declares its own temporaries, including capture lifts).
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
            ExprKind::If(_, _, _) | ExprKind::Match(_, _) => Err(
                "grad: control flow (`if`/`match`) inside `f` is not supported on the native \
                 backend; use a straight-line body, or the interpreter `aria run`"
                    .into(),
            ),
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
                        // ONCE; reuse the AriaTVec for any later reference.
                        let t = self.fresh();
                        let _ = writeln!(
                            self.body,
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
                TVal::Vec(_) => unreachable!(),
            });
        }
        match self.trace(e)? {
            TVal::Scalar(s) => Ok(s),
            TVal::Vec(_) => Err(
                "grad: expected a Float (scalar) but got a Vector in `f` on the native backend"
                    .into(),
            ),
        }
    }

    /// Trace `e`, requiring it to be a traced Vector.
    fn trace_vec(&mut self, e: &Expr) -> Result<String, String> {
        if let Some(r) = self.try_capture(e, CapKind::Vector) {
            return r.map(|tv| match tv {
                TVal::Vec(v) => v,
                TVal::Scalar(_) => unreachable!(),
            });
        }
        match self.trace(e)? {
            TVal::Vec(v) => Ok(v),
            TVal::Scalar(_) => Err(
                "grad: expected a Vector but got a Float (scalar) in `f` on the native backend"
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
            "vec_len" => Err(
                "grad: `vec_len` inside `f` is not supported on the native backend (its result \
                 is a non-differentiable Int used only for control flow)"
                    .into(),
            ),
            other => {
                // A call to a user function (not yet inlined) or an unsupported
                // builtin: gate cleanly.
                if self.fns.contains_key(other) {
                    Err(format!(
                        "grad: `f` calls the function `{}` — calls to other functions inside `f` \
                         are not supported on the native backend; inline it, or use the \
                         interpreter `aria run`",
                        other
                    ))
                } else {
                    Err(format!(
                        "grad: unsupported operation `{}` inside `f` on the native backend",
                        other
                    ))
                }
            }
        }
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
