//! Typed intermediate representation (A-normal form) — the shared foundation
//! for the memory-model proof-of-concept and the future WASM/native backends.
//!
//! The tree-walking interpreter has no notion of allocation, so it can neither
//! host Perceus-style reference counting nor lower to a real backend. This IR
//! fixes that: it is an explicit, sequenced form where every heap allocation
//! (an ADT constructor) and every intermediate value is named, so later passes
//! can insert `dup`/`drop` reference-count operations and reuse analysis.
//!
//! Scope of this stage: the *functional subset* — `Int`, `Bool`, `Float`,
//! `String`, algebraic data types, `let`, `if`, `match`, calls, and arithmetic.
//! Only ADT cells are heap-allocated and reference-counted; `Int`/`Bool`/
//! `Float`/`String`/`Unit` are unboxed value types and so are never `dup`/
//! `drop`ed (a `String` in an ADT field is freed by Rust when its cell is).
//! Tensors/RAG remain out of scope (they involve the opaque Tensor type).
//!
//! Stage 1 (this module today): lowering + an IR interpreter that is
//! differentially checked against the tree-walker, plus heap-allocation
//! counting. `dup`/`drop` insertion and reuse analysis come next.

use std::collections::HashMap;

use crate::ast::{BinOp, Expr, Item, Pattern, Program, Stmt, UnOp};

/// Atomic operand: a variable or an unboxed literal.
#[derive(Debug, Clone)]
pub enum Atom {
    Var(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
}

/// The right-hand side of a `let` binding.
#[derive(Debug, Clone)]
pub enum Bind {
    Atom(Atom),
    Prim(BinOp, Atom, Atom),
    Unary(UnOp, Atom),
    /// Heap allocation: an ADT constructor with already-named field atoms.
    Ctor(String, Vec<Atom>),
    /// Function or builtin call with named argument atoms.
    Call(String, Vec<Atom>),
    /// Constructor that may REUSE a freed cell in place: the first arg names a
    /// reuse token (from `DropReuse`). If the token holds an address, the cell
    /// is overwritten (no allocation); otherwise a fresh cell is allocated.
    CtorReuse(String, String, Vec<Atom>),
    If(Atom, Box<IExpr>, Box<IExpr>),
    /// Match on an ADT scrutinee. Arms are keyed by constructor; an arm with
    /// `ctor == None` is a catch-all (binds the whole scrutinee).
    Match(Atom, Vec<IArm>),
    /// Allocate a closure: a heap cell whose constructor tag is the lifted
    /// lambda's name and whose fields are the captured values. Behaves exactly
    /// like `Ctor` for reference counting (the captures are moved into the cell;
    /// the cell is dropped at its last use).
    MakeClosure(String, Vec<Atom>),
    /// Apply a closure value to arguments. The first atom evaluates to a closure
    /// cell; dispatch reads its lambda tag, re-binds the captured fields plus
    /// these argument atoms, and runs the lifted lambda body. Behaves like
    /// `Call` for reference counting (callee + args are consumed). The final
    /// field is the concrete result type (from monomorphization), which the
    /// backends need to type the result temporary; `None` on the untyped
    /// interpreter path.
    ApplyClosure(Atom, Vec<Atom>, Option<crate::ast::Ty>),
}

#[derive(Debug, Clone)]
pub struct IArm {
    pub ctor: Option<String>,
    pub binders: Vec<String>,
    pub body: IExpr,
}

/// An IR expression: a (possibly empty) sequence of `let`s ending in a return.
/// `Dup`/`Drop` are reference-count operations inserted by the `rc` pass; they
/// are absent in freshly-lowered IR and no-ops on unboxed (non-`Ref`) values.
#[derive(Debug, Clone)]
pub enum IExpr {
    Let(String, Bind, Box<IExpr>),
    Ret(Atom),
    /// Increment the refcount of `var`, then run the body.
    Dup(String, Box<IExpr>),
    /// Decrement the refcount of `var` (freeing recursively at 0), then the body.
    Drop(String, Box<IExpr>),
    /// Drop `scrut` but, if it becomes unique-and-dead, hand its cell address to
    /// the reuse `token` (binding it for a later `CtorReuse`) instead of fully
    /// freeing it. The token is "empty" when the cell was shared.
    DropReuse(String, String, Box<IExpr>),
    /// A self-tail-call turned into a loop back-edge: reassign the enclosing
    /// function's parameters to these argument atoms and re-enter its body from
    /// the top. Produced by the `tco` pass (see `tail_call_optimize`); the
    /// argument atoms correspond positionally to `IFn::params`. The backends
    /// evaluate ALL argument atoms into temporaries *before* overwriting any
    /// parameter (a parameter may appear in another argument), then jump to the
    /// loop header instead of performing a real call — giving constant stack for
    /// tail recursion. Ownership of the new args transfers to the params exactly
    /// as a real call would bind them, so the rc pass's dup/drop stays balanced.
    TailCall(Vec<Atom>),
}

#[derive(Debug, Clone)]
pub struct IFn {
    pub params: Vec<String>,
    /// Free variables captured from the enclosing scope, for a *lifted lambda*
    /// (empty for ordinary functions). A closure value is a heap cell whose
    /// constructor tag is this lambda's name and whose fields are the captured
    /// values, in this order. The closure calling convention passes the closure
    /// cell plus the call arguments; each backend's lambda prologue re-binds
    /// these capture names from the cell's fields (dup'ing each so the body owns
    /// its captures exactly like `params`), and the `rc` pass therefore treats
    /// `captures` as additional owned inputs alongside `params`.
    pub captures: Vec<String>,
    /// Concrete type signature for a lifted lambda (`None` for ordinary
    /// functions, whose signature the backends read from the source `FnDecl`).
    /// Lets a backend build a function-table entry and type the lambda body even
    /// though the lambda has no `FnDecl`.
    pub lam_sig: Option<LamSig>,
    pub body: IExpr,
    /// Set by the `tco` pass when `body` contains `IExpr::TailCall` back-edges.
    /// The backends then emit the body wrapped in a loop whose header is the
    /// re-entry target of every `TailCall`. Freshly-lowered / rc'd IR has this
    /// `false` (no `TailCall` nodes yet).
    pub tail_recursive: bool,
}

pub struct LowerError(pub String);

/// Concrete type signature of a lifted lambda: the parameter types, the captured
/// variable types (in closure-cell field order), and the return type.
#[derive(Debug, Clone)]
pub struct LamSig {
    pub param_tys: Vec<crate::ast::Ty>,
    pub capture_tys: Vec<crate::ast::Ty>,
    pub ret_ty: crate::ast::Ty,
}

/// Builtins the IR interpreter implements. Other declared builtins (tensor/RAG/
/// compression) are outside the IR subset and rejected at lowering.
const IR_BUILTINS: &[&str] = &[
    "print_int",
    "print_float",
    "print_bool",
    "print_str",
    "concat",
    "int_to_str",
    // Arrays: heap-allocated, reference-counted `$Array` cells handled directly
    // by the IR interpreter (see `array_builtin`), with FBIP in-place reuse.
    "array_new",
    "array_len",
    "array_get",
    "array_set",
    "array_push",
    // Bytes: a flat byte buffer, modeled as a heap cell whose fields are the
    // bytes (one `IValue::Int` each), with the same RC + FBIP in-place reuse as
    // arrays. Handled directly by the IR interpreter (see `bytes_builtin`).
    "bytes_new",
    "bytes_len",
    "bytes_get",
    "bytes_set",
    "bytes_push",
    "bytes_from_str",
    "bytes_to_str",
    // Vector / embedding builtins. The IR LOWERING lets them through to a
    // `Bind::Call` so the native (C) backend (which lowers via this same path)
    // can emit them. The IR tree-walking INTERPRETER (`aria mem`) does NOT
    // implement them — it gates them with a clean error in `Bind::Call` dispatch
    // (the tree-walking `interp::Interp` is the oracle for vectors instead).
    "vec_new",
    "vec_from_array",
    "vec_to_array",
    "vec_len",
    "vec_get",
    "vec_push",
    "vec_dot",
    "vec_norm",
    "vec_cosine",
    "vec_add",
    "vec_scale",
];

/// Heap-cell constructor tag for a functional array. Starts with `$` so it can
/// never collide with a user constructor (those are uppercase identifiers).
const ARRAY_TAG: &str = "$Array";

/// Heap-cell constructor tag for a `Bytes` buffer. Like `ARRAY_TAG`, starts with
/// `$` so it can never collide with a user constructor.
const BYTES_TAG: &str = "$Bytes";

/// Builtins the IR interpreter does NOT evaluate, but which the WASM backend
/// *does* compile (Phase 2f: tensor ops + `embed_similarity`). Lowering lets
/// these through to a `Bind::Call` so the wasm emitter can handle them; the IR
/// tree-walking interpreter never runs them (the tree-walking `interp::Interp`
/// is the reference oracle for these instead). The codec builtins
/// (`compressed_size`, `neural_bits_per_byte`) are intentionally absent — they
/// remain rejected at lowering (deferred).
const WASM_ONLY_BUILTINS: &[&str] = &[
    "tensor_zeros",
    "tensor_set",
    "tensor_get",
    "tensor_rows",
    "tensor_cols",
    "tensor_row",
    "tensor_from_rows",
    "matmul",
    "transpose",
    "softmax",
    "relu",
    "embed_similarity",
];

// ---- lowering (typed AST -> ANF IR) -------------------------------------

struct Lowerer {
    tmp: usize,
    /// Counter for fresh lifted-lambda names.
    lam: usize,
    /// Arity of every top-level function (used to tell a *global function* —
    /// called directly or wrapped as a value — from a *local* `Fn`-typed
    /// variable, which is applied as a closure).
    fn_arities: HashMap<String, usize>,
    /// Concrete (param types, return type) of each top-level function — used to
    /// give a value-wrapper closure its type signature.
    fn_sigs: HashMap<String, (Vec<crate::ast::Ty>, crate::ast::Ty)>,
    /// Lambdas (and top-level-function value wrappers) lifted to the top level
    /// during lowering. Merged into the program's function map afterwards.
    lifted: Vec<(String, IFn)>,
    /// Wrapper names already emitted for top-level functions used as values
    /// (so each global function gets exactly one `$fnval$` wrapper).
    wrappers: std::collections::HashSet<String>,
    /// Local names currently in scope (function parameters, `let` bindings, match
    /// binders, lambda parameters + captures). A name in this set is a *local*
    /// that shadows any same-named top-level function or builtin — exactly the
    /// tree-walker's scope-before-globals rule — so `Var`/`Call` resolve it as a
    /// local rather than a function value or by-name call. Managed with
    /// snapshot/restore around each scope; on a lowering error the whole pass is
    /// discarded, so an un-restored snapshot is harmless.
    bound: std::collections::HashSet<String>,
}

/// Variable names a pattern binds (used by the closure free-variable walk).
pub(crate) fn pattern_vars(p: &Pattern, acc: &mut std::collections::HashSet<String>) {
    match p {
        Pattern::Var(n) => {
            acc.insert(n.clone());
        }
        Pattern::Wild | Pattern::Int(_) | Pattern::Bool(_) => {}
        Pattern::Ctor(_, subs) => {
            for s in subs {
                pattern_vars(s, acc);
            }
        }
        Pattern::Record(_, fields) => {
            for (_, s) in fields {
                pattern_vars(s, acc);
            }
        }
    }
}

/// Collect the free variables of an expression — names referenced but not bound
/// within it. `bound` is the set of names already in scope (lambda params, lets,
/// match binders). A `Call`/`Apply` callee name counts as a use; constructor
/// names do not. This drives closure capture: a lambda captures exactly its free
/// variables that are not top-level functions or builtins.
pub(crate) fn ast_free(e: &Expr, bound: &std::collections::HashSet<String>, acc: &mut std::collections::HashSet<String>) {
    use std::collections::HashSet;
    match e {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Unit => {}
        Expr::Var(n) => {
            if !bound.contains(n) {
                acc.insert(n.clone());
            }
        }
        Expr::Ctor(_, args) => {
            for a in args {
                ast_free(a, bound, acc);
            }
        }
        Expr::Call(name, args) => {
            if !bound.contains(name) {
                acc.insert(name.clone());
            }
            for a in args {
                ast_free(a, bound, acc);
            }
        }
        Expr::Apply(callee, args, _) => {
            ast_free(callee, bound, acc);
            for a in args {
                ast_free(a, bound, acc);
            }
        }
        Expr::Record(_, fields) => {
            for (_, v) in fields {
                ast_free(v, bound, acc);
            }
        }
        Expr::Field(obj, _) => ast_free(obj, bound, acc),
        Expr::Update(base, updates) => {
            ast_free(base, bound, acc);
            for (_, v) in updates {
                ast_free(v, bound, acc);
            }
        }
        Expr::Unary(_, inner) => ast_free(inner, bound, acc),
        Expr::Binary(_, l, r) => {
            ast_free(l, bound, acc);
            ast_free(r, bound, acc);
        }
        Expr::If(c, t, e2) => {
            ast_free(c, bound, acc);
            ast_free(t, bound, acc);
            ast_free(e2, bound, acc);
        }
        Expr::Match(s, arms) => {
            ast_free(s, bound, acc);
            for arm in arms {
                let mut b = bound.clone();
                pattern_vars(&arm.pat, &mut b);
                ast_free(&arm.body, &b, acc);
            }
        }
        Expr::Lambda(params, body, _) => {
            let mut b = bound.clone();
            for (n, _) in params {
                b.insert(n.clone());
            }
            ast_free(body, &b, acc);
        }
        Expr::Block(stmts, last) => {
            let mut b: HashSet<String> = bound.clone();
            for s in stmts {
                match s {
                    Stmt::Let(name, _, v) => {
                        ast_free(v, &b, acc);
                        b.insert(name.clone());
                    }
                    Stmt::Expr(ex) => ast_free(ex, &b, acc),
                }
            }
            ast_free(last, &b, acc);
        }
    }
}

impl Lowerer {
    fn fresh(&mut self) -> String {
        let n = self.tmp;
        self.tmp += 1;
        format!("$t{}", n)
    }

    /// Return (creating on first use) the name of a zero-capture closure wrapper
    /// that forwards to the top-level function `name` — used when a function is
    /// referenced as a value rather than called. The wrapper's parameters forward
    /// positionally to a direct `Call`.
    fn fn_value_wrapper(&mut self, name: &str) -> String {
        let w = format!("$fnval${}", name);
        if !self.wrappers.contains(&w) {
            self.wrappers.insert(w.clone());
            let arity = *self.fn_arities.get(name).unwrap_or(&0);
            let params: Vec<String> = (0..arity).map(|i| format!("$a{}", i)).collect();
            let call_args: Vec<Atom> = params.iter().map(|p| Atom::Var(p.clone())).collect();
            let t = format!("$w{}", self.lam);
            self.lam += 1;
            let body = IExpr::Let(
                t.clone(),
                Bind::Call(name.to_string(), call_args),
                Box::new(IExpr::Ret(Atom::Var(t))),
            );
            let lam_sig = self.fn_sigs.get(name).map(|(ptys, ret)| LamSig {
                param_tys: ptys.clone(),
                capture_tys: vec![],
                ret_ty: ret.clone(),
            });
            self.lifted.push((
                w.clone(),
                IFn { params, captures: vec![], lam_sig, body, tail_recursive: false },
            ));
        }
        w
    }

    /// Lower `e` into a complete IExpr (lets ending in `Ret`).
    fn lower_block(&mut self, e: &Expr) -> Result<IExpr, LowerError> {
        let mut stmts: Vec<(String, Bind)> = Vec::new();
        let atom = self.lower(e, &mut stmts)?;
        // Fold the collected bindings into nested `Let`s, innermost = Ret(atom).
        let mut acc = IExpr::Ret(atom);
        for (name, bind) in stmts.into_iter().rev() {
            acc = IExpr::Let(name, bind, Box::new(acc));
        }
        Ok(acc)
    }

    /// Lower `e`, pushing any needed bindings to `stmts`, returning its atom.
    fn lower(&mut self, e: &Expr, stmts: &mut Vec<(String, Bind)>) -> Result<Atom, LowerError> {
        match e {
            Expr::Int(n) => Ok(Atom::Int(*n)),
            Expr::Float(f) => Ok(Atom::Float(*f)),
            Expr::Bool(b) => Ok(Atom::Bool(*b)),
            Expr::Str(s) => Ok(Atom::Str(s.clone())),
            Expr::Unit => Ok(Atom::Unit),
            // Records are interpreter-only so far; the IR/compiled backends do
            // not lower them yet (cleanly rejected, like tensors/arrays).
            Expr::Record(name, _) => Err(LowerError(format!(
                "records are not yet supported in the IR/compiled backends \
                 (use the interpreter `aria run`): record `{}`",
                name
            ))),
            Expr::Field(_, field) => Err(LowerError(format!(
                "record field access `.{}` is not yet supported in the IR/compiled \
                 backends (use the interpreter `aria run`)",
                field
            ))),
            Expr::Update(_, _) => Err(LowerError(
                "record update `{ r | .. }` is not yet supported in the IR/compiled \
                 backends (use the interpreter `aria run`)"
                    .to_string(),
            )),
            Expr::Var(n) => {
                // A bare top-level function name in value position (it would be an
                // `Expr::Call` if it were applied) becomes a closure over a
                // zero-capture wrapper that forwards to the direct call — UNLESS a
                // local of the same name shadows it (then it is an ordinary local).
                if !self.bound.contains(n) && self.fn_arities.contains_key(n) {
                    let w = self.fn_value_wrapper(n);
                    let t = self.fresh();
                    stmts.push((t.clone(), Bind::MakeClosure(w, vec![])));
                    Ok(Atom::Var(t))
                } else {
                    Ok(Atom::Var(n.clone()))
                }
            }

            Expr::Binary(op, l, r) => {
                // Short-circuit `&&` / `||` must NOT evaluate the rhs eagerly —
                // lower to control flow so the rhs runs only in the taken branch
                // (matching the interpreter's short-circuit semantics).
                if matches!(op, BinOp::And | BinOp::Or) {
                    let la = self.lower(l, stmts)?;
                    let rhs = self.lower_block(r)?;
                    let (then, els) = match op {
                        BinOp::And => (rhs, IExpr::Ret(Atom::Bool(false))),
                        _ => (IExpr::Ret(Atom::Bool(true)), rhs), // Or
                    };
                    let t = self.fresh();
                    stmts.push((t.clone(), Bind::If(la, Box::new(then), Box::new(els))));
                    return Ok(Atom::Var(t));
                }
                let la = self.lower(l, stmts)?;
                let ra = self.lower(r, stmts)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::Prim(*op, la, ra)));
                Ok(Atom::Var(t))
            }
            Expr::Unary(op, inner) => {
                let a = self.lower(inner, stmts)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::Unary(*op, a)));
                Ok(Atom::Var(t))
            }
            Expr::Ctor(name, args) => {
                let atoms = self.lower_all(args, stmts)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::Ctor(name.clone(), atoms)));
                Ok(Atom::Var(t))
            }
            Expr::Call(name, args) => {
                // A local binding (parameter / let / lambda capture) applied by
                // name is a closure application — the tree-walker resolves the
                // scope before the global function table, so a local shadows any
                // same-named top-level function. A non-local name that is neither a
                // top-level function nor a builtin is likewise a local `Fn` value.
                // Array builtins are routed to `Bind::Call`, not a closure value:
                // `array_lit` is a variadic internal builtin with no
                // `builtins::lookup` signature, and after monomorphization every
                // array op is renamed with an element-kind suffix (`array_get$i`,
                // `array_lit$r`, ...) which also has no `lookup` signature. The IR
                // interpreter's `array_builtin` (unsuffixed) and the native/wasm
                // backends (suffixed) handle them.
                if !name.starts_with("array_")
                    && !name.starts_with("map_")
                    && !name.starts_with("set_")
                    && !name.starts_with("vec_")
                    && (self.bound.contains(name)
                        || (!self.fn_arities.contains_key(name)
                            && crate::builtins::lookup(name).is_none()))
                {
                    let atoms = self.lower_all(args, stmts)?;
                    let t = self.fresh();
                    stmts.push((t.clone(), Bind::ApplyClosure(Atom::Var(name.clone()), atoms, None)));
                    return Ok(Atom::Var(t));
                }
                // Maps and Sets are FULLY supported by the interpreter and the
                // native (C) backend, but NOT by the IR memory path (`aria mem`)
                // or the wasm backend. The native path renames these to suffixed
                // names (`map_insert$i_i`) during monomorphization, which have no
                // `builtins::lookup` signature and so fall through to `Bind::Call`
                // below. An UNSUFFIXED map/set builtin reaching lowering means we
                // are on the IR-interpreter path: gate it with a clean error.
                if crate::builtins::lookup(name).is_some()
                    && (name.starts_with("map_") || name.starts_with("set_"))
                {
                    return Err(LowerError(format!(
                        "maps/sets are not yet supported in the IR memory path \
                         (`aria mem`) — use the interpreter `aria run` or the \
                         native backend `aria native-run`: builtin `{}`",
                        name
                    )));
                }
                // Builtins the IR doesn't implement (tensors/RAG/compression) are
                // outside the IR subset — reject at lowering with a clear message
                // rather than failing confusingly at run time.
                if crate::builtins::lookup(name).is_some()
                    && !IR_BUILTINS.contains(&name.as_str())
                    && !WASM_ONLY_BUILTINS.contains(&name.as_str())
                {
                    return Err(LowerError(format!(
                        "builtin `{}` is outside the IR subset (compression codecs not supported)",
                        name
                    )));
                }
                let atoms = self.lower_all(args, stmts)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::Call(name.clone(), atoms)));
                Ok(Atom::Var(t))
            }
            Expr::If(c, th, el) => {
                let ca = self.lower(c, stmts)?;
                let th_ir = self.lower_block(th)?;
                let el_ir = self.lower_block(el)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::If(ca, Box::new(th_ir), Box::new(el_ir))));
                Ok(Atom::Var(t))
            }
            Expr::Match(scrut, arms) => {
                let sa = self.lower(scrut, stmts)?;
                let bind = self.lower_match(sa, arms)?;
                let t = self.fresh();
                stmts.push((t.clone(), bind));
                Ok(Atom::Var(t))
            }
            Expr::Lambda(params, body, sig) => {
                // Lift the lambda to a top-level function and allocate a closure
                // cell holding its captured free variables. When monomorphization
                // has attached a concrete `ClosureSig`, take the capture list (and
                // types) from it so the cell layout, the lifted lambda's prologue,
                // and the backends' types all agree. Otherwise (the untyped IR
                // interpreter path) recover the captures syntactically.
                let (capture_names, lam_sig) = match sig {
                    Some(cs) => {
                        let names: Vec<String> = cs.captures.iter().map(|(n, _)| n.clone()).collect();
                        let sig = LamSig {
                            param_tys: params.iter().map(|(_, t)| t.clone()).collect(),
                            capture_tys: cs.captures.iter().map(|(_, t)| t.clone()).collect(),
                            ret_ty: cs.ret.clone(),
                        };
                        (names, Some(sig))
                    }
                    None => {
                        let mut param_set = std::collections::HashSet::new();
                        for (n, _) in params {
                            param_set.insert(n.clone());
                        }
                        let mut fvs = std::collections::HashSet::new();
                        ast_free(body, &param_set, &mut fvs);
                        let mut captures: Vec<String> = fvs
                            .into_iter()
                            .filter(|n| {
                                !self.fn_arities.contains_key(n)
                                    && crate::builtins::lookup(n).is_none()
                            })
                            .collect();
                        captures.sort();
                        (captures, None)
                    }
                };
                let lam_name = format!("$lam{}", self.lam);
                self.lam += 1;
                let lam_params: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
                // The lambda body is a fresh scope: its only locals are its
                // parameters and the captured variables (everything else it
                // references is a top-level function or builtin).
                let saved = std::mem::take(&mut self.bound);
                self.bound = lam_params.iter().cloned().chain(capture_names.iter().cloned()).collect();
                let lam_body = self.lower_block(body)?;
                self.bound = saved;
                self.lifted.push((
                    lam_name.clone(),
                    IFn {
                        params: lam_params,
                        captures: capture_names.clone(),
                        lam_sig,
                        body: lam_body,
                        tail_recursive: false,
                    },
                ));
                let cap_atoms = capture_names.into_iter().map(Atom::Var).collect();
                let t = self.fresh();
                stmts.push((t.clone(), Bind::MakeClosure(lam_name, cap_atoms)));
                Ok(Atom::Var(t))
            }
            Expr::Apply(callee, args, result_ty) => {
                let c = self.lower(callee, stmts)?;
                let atoms = self.lower_all(args, stmts)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::ApplyClosure(c, atoms, result_ty.clone())));
                Ok(Atom::Var(t))
            }
            Expr::Block(block_stmts, last) => {
                // A block opens a scope: its `let` bindings are local for the rest
                // of the block but must not leak to siblings. (`let` is
                // non-recursive — the value is lowered before the name is bound.)
                let block_saved = self.bound.clone();
                for s in block_stmts {
                    match s {
                        Stmt::Let(name, _ty, value) => {
                            let va = self.lower(value, stmts)?;
                            stmts.push((name.clone(), Bind::Atom(va)));
                            self.bound.insert(name.clone());
                        }
                        Stmt::Expr(ex) => {
                            // Evaluate for effect; bind to a discarded temp.
                            let a = self.lower(ex, stmts)?;
                            let t = self.fresh();
                            stmts.push((t, Bind::Atom(a)));
                        }
                    }
                }
                let r = self.lower(last, stmts);
                self.bound = block_saved;
                r
            }
        }
    }

    fn lower_all(&mut self, es: &[Expr], stmts: &mut Vec<(String, Bind)>) -> Result<Vec<Atom>, LowerError> {
        es.iter().map(|e| self.lower(e, stmts)).collect()
    }

    /// Lower a `match`. ADT (constructor) patterns become a `Match` bind;
    /// integer/bool literal patterns become an `if`-chain.
    /// If `arm` is a single irrefutable arm whose constructor pattern has NESTED
    /// constructor sub-patterns (e.g. tuple/record destructuring `((a,b),c)`),
    /// flatten it: bind each nested sub-pattern to a fresh variable and re-match
    /// it inside the body. The IR's flat `Match` then handles each level. Returns
    /// `None` if the arm is already flat or has a refutable (literal) sub-pattern.
    fn flatten_nested_arm(&mut self, arm: &crate::ast::Arm) -> Option<crate::ast::Arm> {
        use crate::ast::Arm;
        let (name, subs) = match &arm.pat {
            Pattern::Ctor(n, s) => (n, s),
            _ => return None,
        };
        if !subs.iter().any(|s| matches!(s, Pattern::Ctor(..))) {
            return None; // already flat
        }
        if subs.iter().any(|s| matches!(s, Pattern::Int(_) | Pattern::Bool(_))) {
            return None; // refutable nesting needs fall-through; leave to the error
        }
        let mut flat = Vec::with_capacity(subs.len());
        let mut body = arm.body.clone();
        for sub in subs {
            match sub {
                Pattern::Var(_) | Pattern::Wild => flat.push(sub.clone()),
                Pattern::Ctor(..) => {
                    let f = self.fresh();
                    flat.push(Pattern::Var(f.clone()));
                    body = Expr::Match(
                        Box::new(Expr::Var(f)),
                        vec![Arm { pat: sub.clone(), body }],
                    );
                }
                _ => return None,
            }
        }
        Some(Arm { pat: Pattern::Ctor(name.clone(), flat), body })
    }

    fn lower_match(&mut self, scrut: Atom, arms: &[crate::ast::Arm]) -> Result<Bind, LowerError> {
        // Flatten a single nested destructuring arm so the IR's flat matcher can
        // lower it (tuple/record patterns like `((a, b), c)`).
        if arms.len() == 1 {
            if let Some(flat) = self.flatten_nested_arm(&arms[0]) {
                return self.lower_match(scrut, std::slice::from_ref(&flat));
            }
        }
        let has_ctor = arms.iter().any(|a| matches!(a.pat, Pattern::Ctor(_, _)));
        if has_ctor {
            let mut iarms = Vec::new();
            for arm in arms {
                // Determine what the arm binds BEFORE lowering its body, so those
                // binders are in scope (they shadow same-named globals) while the
                // body is lowered.
                let (ctor, binders): (Option<String>, Vec<String>) = match &arm.pat {
                    Pattern::Ctor(name, subs) => {
                        let mut binders = Vec::new();
                        for sp in subs {
                            match sp {
                                Pattern::Var(n) => binders.push(n.clone()),
                                Pattern::Wild => binders.push(self.fresh()),
                                _ => {
                                    return Err(LowerError(
                                        "nested constructor patterns not supported in IR yet".into(),
                                    ))
                                }
                            }
                        }
                        (Some(name.clone()), binders)
                    }
                    Pattern::Var(n) => (None, vec![n.clone()]),
                    Pattern::Wild => (None, vec![self.fresh()]),
                    _ => {
                        return Err(LowerError(
                            "mixed literal/constructor patterns not supported in IR yet".into(),
                        ))
                    }
                };
                let saved = self.bound.clone();
                for b in &binders {
                    self.bound.insert(b.clone());
                }
                let body = self.lower_block(&arm.body)?;
                self.bound = saved;
                iarms.push(IArm { ctor, binders, body });
            }
            Ok(Bind::Match(scrut, iarms))
        } else {
            // Literal (Int/Bool) match -> if-chain in SOURCE ORDER, preserving
            // the interpreter's first-match-wins semantics. Fold arms
            // back-to-front: a catch-all (Var/Wild) REPLACES the accumulator (so
            // any later arms become unreachable, exactly as the interpreter
            // treats arms after a catch-all); a literal arm wraps the
            // accumulator in `if scrut == lit { body } else { acc }`. Folding the
            // first catch-all last makes it dominate later arms correctly.
            let mut chain = IExpr::Ret(Atom::Unit); // exhaustiveness => unreachable
            for arm in arms.iter().rev() {
                match &arm.pat {
                    Pattern::Var(_) | Pattern::Wild => {
                        chain = self.lower_catchall(&scrut, arm)?;
                    }
                    Pattern::Int(_) | Pattern::Bool(_) => {
                        let cond = self.lit_cond(&scrut, &arm.pat)?;
                        let then = self.lower_block(&arm.body)?;
                        let c = self.fresh();
                        let t = self.fresh();
                        let if_bind =
                            Bind::If(Atom::Var(c.clone()), Box::new(then), Box::new(chain));
                        let inner =
                            IExpr::Let(t.clone(), if_bind, Box::new(IExpr::Ret(Atom::Var(t))));
                        chain = IExpr::Let(c, cond, Box::new(inner));
                    }
                    _ => {
                        return Err(LowerError(
                            "mixed literal/constructor patterns not supported in IR yet".into(),
                        ))
                    }
                }
            }
            // lower_match returns a Bind; wrap the chain in an identity `if true`.
            Ok(Bind::If(Atom::Bool(true), Box::new(chain), Box::new(IExpr::Ret(Atom::Unit))))
        }
    }

    fn lower_catchall(&mut self, scrut: &Atom, arm: &crate::ast::Arm) -> Result<IExpr, LowerError> {
        // Bind the catch-all variable (if any) to the scrutinee, then the body —
        // with that variable in scope (it shadows a same-named global) while the
        // body is lowered.
        match &arm.pat {
            Pattern::Var(n) => {
                let saved = self.bound.clone();
                self.bound.insert(n.clone());
                let body = self.lower_block(&arm.body)?;
                self.bound = saved;
                Ok(IExpr::Let(n.clone(), Bind::Atom(scrut.clone()), Box::new(body)))
            }
            _ => self.lower_block(&arm.body),
        }
    }

    fn lit_cond(&mut self, scrut: &Atom, pat: &Pattern) -> Result<Bind, LowerError> {
        let rhs = match pat {
            Pattern::Int(i) => Atom::Int(*i),
            Pattern::Bool(b) => Atom::Bool(*b),
            _ => return Err(LowerError("unsupported literal pattern".into())),
        };
        Ok(Bind::Prim(BinOp::Eq, scrut.clone(), rhs))
    }
}

/// Lower an entire program's functions to IR. Returns an error if any function
/// uses a feature outside the IR subset.
pub fn lower_program(program: &Program) -> Result<HashMap<String, IFn>, String> {
    let mut fns = HashMap::new();
    let mut fn_arities = HashMap::new();
    let mut fn_sigs = HashMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            fn_arities.insert(f.name.clone(), f.params.len());
            fn_sigs.insert(
                f.name.clone(),
                (f.params.iter().map(|p| p.ty.clone()).collect::<Vec<_>>(), f.ret.clone()),
            );
        }
    }
    let mut lw = Lowerer {
        tmp: 0,
        lam: 0,
        fn_arities,
        fn_sigs,
        lifted: Vec::new(),
        wrappers: std::collections::HashSet::new(),
        bound: std::collections::HashSet::new(),
    };
    for item in &program.items {
        if let Item::Fn(f) = item {
            // Parameters are the function's initial in-scope locals.
            lw.bound = f.params.iter().map(|p| p.name.clone()).collect();
            let body = lw.lower_block(&f.body).map_err(|e| format!("fn `{}`: {}", f.name, e.0))?;
            let params = f.params.iter().map(|p| p.name.clone()).collect();
            fns.insert(
                f.name.clone(),
                IFn { params, captures: vec![], lam_sig: None, body, tail_recursive: false },
            );
        }
    }
    // Merge lifted lambdas / function-value wrappers into the program map.
    for (name, f) in lw.lifted {
        fns.insert(name, f);
    }
    Ok(fns)
}

// ---- self-tail-call optimization ----------------------------------------

/// Rewrite every SELF tail-call into a loop back-edge (`IExpr::TailCall`).
///
/// A function `f` is *self-tail-recursive* if, in tail position of its body,
/// it calls itself (`Bind::Call(f, args)`) with matching arity. Tail position
/// in the (rc'd) ANF IR flows through:
///   * the body's final `Ret`,
///   * a `Let(x, If(c, t, e), Ret(Var(x)))` — both branches are tail,
///   * a `Let(x, Match(s, arms), Ret(Var(x)))` — every arm body is tail,
///   * `Dup`/`Drop`/`DropReuse` wrappers (they execute, then continue), and
///   * the continuation of a non-tail `Let`.
/// A tail self-call has the shape `Let(t, Call(f, args), Ret(Var(t)))`; it is
/// replaced by `TailCall(args)`. The backends (and the IR interpreter) then
/// run the body in a loop, reassigning the parameters from `args` and
/// re-entering — constant stack for tail recursion. Only DIRECT self-calls are
/// handled; mutual recursion is out of scope.
///
/// Runs AFTER `rc::insert_rc`: the new arguments carry ownership transferred to
/// the parameters exactly as a real call would bind them, so dup/drop stays
/// balanced (the rc pass already dropped any parameter the recursive arguments
/// do not reuse, and dup'd any reused more than once).
pub fn tail_call_optimize(fns: HashMap<String, IFn>) -> HashMap<String, IFn> {
    fns.into_iter()
        .map(|(name, f)| {
            let arity = f.params.len();
            let mut found = false;
            let body = rewrite_tail(&name, arity, f.body, &mut found);
            (
                name,
                IFn {
                    params: f.params,
                    captures: f.captures,
                    lam_sig: f.lam_sig,
                    body,
                    tail_recursive: found,
                },
            )
        })
        .collect()
}

/// Rewrite an IExpr that is itself in TAIL position. `found` is set when at
/// least one self-tail-call is replaced.
fn rewrite_tail(self_name: &str, arity: usize, e: IExpr, found: &mut bool) -> IExpr {
    match e {
        // The canonical self-tail-call shape: `let t = self(args) in ret t`.
        IExpr::Let(ref x, Bind::Call(ref callee, ref args), ref cont)
            if callee == self_name
                && args.len() == arity
                && matches!(&**cont, IExpr::Ret(Atom::Var(v)) if v == x) =>
        {
            *found = true;
            IExpr::TailCall(args.clone())
        }
        // A tail `If`/`Match` bound to a temp returned immediately: its result
        // IS the function result, so both branches / every arm are tail.
        IExpr::Let(ref x, Bind::If(ref c, ref t, ref el), ref cont)
            if matches!(&**cont, IExpr::Ret(Atom::Var(v)) if v == x) =>
        {
            let t2 = rewrite_tail(self_name, arity, (**t).clone(), found);
            let e2 = rewrite_tail(self_name, arity, (**el).clone(), found);
            IExpr::Let(
                x.clone(),
                Bind::If(c.clone(), Box::new(t2), Box::new(e2)),
                cont.clone(),
            )
        }
        IExpr::Let(ref x, Bind::Match(ref s, ref arms), ref cont)
            if matches!(&**cont, IExpr::Ret(Atom::Var(v)) if v == x) =>
        {
            let arms2 = arms
                .iter()
                .map(|a| IArm {
                    ctor: a.ctor.clone(),
                    binders: a.binders.clone(),
                    body: rewrite_tail(self_name, arity, a.body.clone(), found),
                })
                .collect();
            IExpr::Let(
                x.clone(),
                Bind::Match(s.clone(), arms2),
                cont.clone(),
            )
        }
        // A non-tail `Let`: only its continuation is in tail position.
        IExpr::Let(x, bind, body) => {
            let body2 = rewrite_tail(self_name, arity, *body, found);
            IExpr::Let(x, bind, Box::new(body2))
        }
        // These execute, then continue to the tail.
        IExpr::Dup(v, body) => {
            IExpr::Dup(v, Box::new(rewrite_tail(self_name, arity, *body, found)))
        }
        IExpr::Drop(v, body) => {
            IExpr::Drop(v, Box::new(rewrite_tail(self_name, arity, *body, found)))
        }
        IExpr::DropReuse(s, t, body) => {
            IExpr::DropReuse(s, t, Box::new(rewrite_tail(self_name, arity, *body, found)))
        }
        // A plain return is the tail value; nothing to rewrite.
        other @ (IExpr::Ret(_) | IExpr::TailCall(_)) => other,
    }
}

// ---- IR interpreter with allocation counting ----------------------------

#[derive(Debug, Clone)]
pub enum IValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    /// A heap reference to an ADT cell.
    Ref(usize),
    /// A reuse token: `Some(addr)` if a freed cell is available for in-place
    /// reuse, `None` otherwise. Compiler-internal; never user-visible.
    Token(Option<usize>),
}

/// Discriminant for the kind of heap cell. The interpreter dispatches on this
/// (e.g. how `render` prints the cell) instead of string-comparing the `ctor`
/// tag, so each new cell shape (records, maps, …) is forced to be handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellKind {
    /// An ADT constructor or closure cell; `ctor` holds its tag name.
    Data,
    /// A functional array; `ctor` is `ARRAY_TAG`, `fields` are its elements.
    Array,
    /// A `Bytes` buffer; `ctor` is `BYTES_TAG`, `fields` are its bytes (each an
    /// `IValue::Int` 0..255). Same RC/FBIP shape as `Array`, but renders/compares
    /// as a distinct type.
    Bytes,
}

#[derive(Debug, Clone)]
struct Cell {
    kind: CellKind,
    ctor: String,
    fields: Vec<IValue>,
    rc: u32,
}

/// Metrics gathered while interpreting the IR.
#[derive(Debug, Default, Clone)]
pub struct Metrics {
    /// Total ADT cells constructed (gross allocations).
    pub allocations: usize,
    /// Cells freed by `drop` reaching refcount 0.
    pub frees: usize,
    /// Cells reused in place by `CtorReuse` (avoided a fresh allocation).
    pub reuses: usize,
    pub dups: usize,
    pub drops: usize,
    /// Currently-live cells (allocations - frees so far).
    pub live: usize,
    /// Maximum simultaneously-live cells during the run.
    pub peak_live: usize,
}

/// Max IR call-nesting before a catchable error. The IR interpreter, like the
/// AST one, is run on a large-stack thread (see main.rs), so this is generous;
/// it exists to turn non-terminating recursion into an error, not a crash.
const MAX_IR_CALL_DEPTH: usize = 100_000;

pub struct IrInterp {
    fns: HashMap<String, IFn>,
    /// `None` = a freed slot (used to detect use-after-free).
    heap: Vec<Option<Cell>>,
    depth: usize,
    pub metrics: Metrics,
    /// True when the IR has been through the rc pass (`dup`/`drop` inserted), so
    /// the interpreter actively manages reference counts. A `Bind::Prim(Eq|Ne)`
    /// on ADTs consumes its operands — but only the rc'd IR has dup'd a reused
    /// value beforehand, so the operand-drop is performed ONLY in this mode. On
    /// un-rc'd IR (no dups inserted), the comparison leaves operands alone, which
    /// avoids a spurious double-free for e.g. `v == v` (the un-rc'd baseline
    /// never reclaims memory anyway — it only checks the result value).
    manage_rc: bool,
}

type Env = HashMap<String, IValue>;

impl IrInterp {
    /// Construct an interpreter for rc'd IR (manages reference counts). This is
    /// the production path (the wasm backend always rc's its IR).
    pub fn new(fns: HashMap<String, IFn>) -> Self {
        IrInterp {
            fns,
            heap: Vec::new(),
            depth: 0,
            metrics: Metrics::default(),
            manage_rc: true,
        }
    }

    /// Construct an interpreter for un-rc'd IR (no dup/drop inserted). Used by
    /// the differential fuzz baseline; it does not reclaim memory and so does
    /// not perform the `Eq`/`Ne` operand-drop.
    pub fn new_no_rc(fns: HashMap<String, IFn>) -> Self {
        IrInterp {
            fns,
            heap: Vec::new(),
            depth: 0,
            metrics: Metrics::default(),
            manage_rc: false,
        }
    }

    /// Run `main` and return its value (plus collected metrics in `self`).
    pub fn run_main(&mut self) -> Result<IValue, String> {
        let main = self.fns.get("main").ok_or("no `main`")?.clone();
        let env = Env::new();
        self.eval(&main.body, env)
    }

    /// Render an IR value into the same textual form as `interp::Value::display`
    /// (recursively reading ADT cells from the heap), so IR and interpreter
    /// results can be compared structurally rather than by ad-hoc digit soup.
    pub fn render(&self, v: &IValue) -> String {
        match v {
            IValue::Int(n) => n.to_string(),
            // Match interp::Value::display exactly so the cross-check passes.
            IValue::Float(f) => format!("{}", f),
            IValue::Bool(b) => b.to_string(),
            IValue::Str(s) => s.clone(),
            IValue::Unit => "()".to_string(),
            IValue::Token(_) => "<token>".to_string(), // never a program result
            IValue::Ref(a) => {
                let c = self.heap[*a].as_ref().expect("render: freed cell escaped");
                match c.kind {
                    CellKind::Array => {
                        // Match interp::Value::Array display exactly: `[1, 2, 3]`
                        // (and `[]` for the empty array).
                        let inner: Vec<String> = c.fields.iter().map(|f| self.render(f)).collect();
                        format!("[{}]", inner.join(", "))
                    }
                    CellKind::Bytes => {
                        // Match interp::render_bytes exactly: `Bytes[00 01 ff]`.
                        let bs: Vec<u8> = c
                            .fields
                            .iter()
                            .map(|f| match f {
                                IValue::Int(n) => *n as u8,
                                _ => 0,
                            })
                            .collect();
                        crate::interp::render_bytes(&bs)
                    }
                    CellKind::Data if c.fields.is_empty() => c.ctor.clone(),
                    CellKind::Data => {
                        let inner: Vec<String> = c.fields.iter().map(|f| self.render(f)).collect();
                        format!("{}({})", c.ctor, inner.join(", "))
                    }
                }
            }
        }
    }

    fn atom(&self, a: &Atom, env: &Env) -> Result<IValue, String> {
        Ok(match a {
            Atom::Int(n) => IValue::Int(*n),
            Atom::Float(f) => IValue::Float(*f),
            Atom::Bool(b) => IValue::Bool(*b),
            Atom::Str(s) => IValue::Str(s.clone()),
            Atom::Unit => IValue::Unit,
            Atom::Var(n) => env.get(n).cloned().ok_or_else(|| format!("ir: unbound {}", n))?,
        })
    }

    fn eval(&mut self, e: &IExpr, mut env: Env) -> Result<IValue, String> {
        // Real trampoline over the `let`-chain: iterate instead of recursing, so
        // a long sequence of bindings does not grow the native stack.
        let mut e = e;
        loop {
            match e {
                IExpr::Ret(a) => return self.atom(a, &env),
                IExpr::Let(x, bind, body) => {
                    let v = self.eval_bind(bind, &env)?;
                    env.insert(x.clone(), v);
                    e = body;
                }
                IExpr::Dup(v, body) => {
                    let val = self.atom(&Atom::Var(v.clone()), &env)?;
                    self.dup_value(&val)?;
                    e = body;
                }
                IExpr::Drop(v, body) => {
                    let val = self.atom(&Atom::Var(v.clone()), &env)?;
                    self.drop_value(&val)?;
                    e = body;
                }
                IExpr::DropReuse(scrut, tok, body) => {
                    let token = match self.atom(&Atom::Var(scrut.clone()), &env)? {
                        IValue::Ref(a) => {
                            self.metrics.drops += 1;
                            self.drop_for_reuse(a)?
                        }
                        // Non-heap scrutinee can't be reused.
                        _ => IValue::Token(None),
                    };
                    env.insert(tok.clone(), token);
                    e = body;
                }
                // The IR tree-walking interpreter is the differential oracle and
                // runs rc'd-but-NOT-tco'd IR (the `tco` pass is applied only in
                // the wasm/native pipelines), so `TailCall` never reaches here.
                IExpr::TailCall(_) => {
                    return Err("ir: TailCall in non-TCO interpreter (internal)".into())
                }
            }
        }
    }

    /// Like `drop_cell`, but when the cell becomes unique-and-dead we KEEP its
    /// slot (releasing only its children) and return a reuse token holding the
    /// address, so a subsequent `CtorReuse` can overwrite it in place.
    fn drop_for_reuse(&mut self, addr: usize) -> Result<IValue, String> {
        let reached_zero = match &mut self.heap[addr] {
            Some(cell) => {
                if cell.rc == 0 {
                    return Err("ir: drop of cell with refcount 0 (double free)".into());
                }
                cell.rc -= 1;
                cell.rc == 0
            }
            None => return Err("ir: drop of already-freed cell".into()),
        };
        if reached_zero {
            // Release the children but retain the slot for reuse.
            let fields = std::mem::take(&mut self.heap[addr].as_mut().unwrap().fields);
            for f in &fields {
                if let IValue::Ref(a) = f {
                    self.drop_cell(*a)?;
                }
            }
            Ok(IValue::Token(Some(addr)))
        } else {
            Ok(IValue::Token(None))
        }
    }

    /// Structural equality matching `interp::values_equal`: scalars by value
    /// (`NaN != NaN`), ADT cells by constructor + fields read from the heap.
    fn ir_equal(&self, a: &IValue, b: &IValue) -> bool {
        match (a, b) {
            (IValue::Int(x), IValue::Int(y)) => x == y,
            (IValue::Float(x), IValue::Float(y)) => x == y,
            (IValue::Bool(x), IValue::Bool(y)) => x == y,
            (IValue::Str(x), IValue::Str(y)) => x == y,
            (IValue::Unit, IValue::Unit) => true,
            (IValue::Ref(x), IValue::Ref(y)) => match (&self.heap[*x], &self.heap[*y]) {
                (Some(cx), Some(cy)) => {
                    cx.ctor == cy.ctor
                        && cx.fields.len() == cy.fields.len()
                        && cx.fields.iter().zip(&cy.fields).all(|(p, q)| self.ir_equal(p, q))
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Decrement a cell's refcount; at zero, free it and recursively drop its
    /// `Ref` fields. This is the runtime side of Perceus reference counting.
    fn drop_cell(&mut self, addr: usize) -> Result<(), String> {
        let reached_zero = match &mut self.heap[addr] {
            Some(cell) => {
                if cell.rc == 0 {
                    return Err("ir: drop of cell with refcount 0 (double free)".into());
                }
                cell.rc -= 1;
                cell.rc == 0
            }
            None => return Err("ir: drop of already-freed cell".into()),
        };
        if reached_zero {
            // Take the cell out, free the slot, then drop child references.
            let cell = self.heap[addr].take().unwrap();
            self.metrics.frees += 1;
            self.metrics.live -= 1;
            for f in &cell.fields {
                if let IValue::Ref(a) = f {
                    self.drop_cell(*a)?;
                }
            }
        }
        Ok(())
    }

    /// Allocate a heap cell, updating allocation metrics; returns its address.
    fn push_cell(&mut self, kind: CellKind, ctor: &str, fields: Vec<IValue>) -> usize {
        self.metrics.allocations += 1;
        self.metrics.live += 1;
        self.metrics.peak_live = self.metrics.peak_live.max(self.metrics.live);
        self.heap.push(Some(Cell { kind, ctor: ctor.to_string(), fields, rc: 1 }));
        self.heap.len() - 1
    }

    /// Increment a value's refcount if it is a heap reference (no-op otherwise).
    fn dup_value(&mut self, v: &IValue) -> Result<(), String> {
        if let IValue::Ref(a) = v {
            match &mut self.heap[*a] {
                Some(c) => c.rc += 1,
                None => return Err("ir: dup of freed cell".into()),
            }
            self.metrics.dups += 1;
        }
        Ok(())
    }

    /// Drop a value (consuming a reference) if it is a heap reference.
    fn drop_value(&mut self, v: &IValue) -> Result<(), String> {
        if let IValue::Ref(a) = v {
            self.metrics.drops += 1;
            self.drop_cell(*a)?;
        }
        Ok(())
    }

    /// Array builtins. A `Bind::Call` transfers ownership of its argument Refs to
    /// the callee, so these consume (drop) the array argument and any consumed
    /// element, and produce an owned result — keeping the RC pass's dup/drop
    /// balanced. `set`/`push` perform FBIP in-place reuse when the array is
    /// uniquely owned (rc == 1), else copy-on-write. Returns `Ok(None)` if `name`
    /// is not an array builtin.
    fn array_builtin(&mut self, name: &str, args: &[IValue]) -> Result<Option<IValue>, String> {
        let as_ref = |v: &IValue| -> Result<usize, String> {
            match v {
                IValue::Ref(a) => Ok(*a),
                _ => Err("ir: array builtin expected an array".into()),
            }
        };
        let as_int = |v: &IValue| -> Result<i64, String> {
            match v {
                IValue::Int(n) => Ok(*n),
                _ => Err("ir: array builtin expected an Int index".into()),
            }
        };
        match name {
            "array_new" => Ok(Some(IValue::Ref(self.push_cell(CellKind::Array, ARRAY_TAG, Vec::new())))),
            "array_lit" => {
                // Array literal desugared (by the parser) to a single flat call
                // with all elements as arguments. A `Bind::Call` transfers
                // ownership of its argument Refs in, so each element is moved
                // straight into the new array cell — do NOT dup them. Builds
                // exactly one cell (an empty literal builds an empty array).
                let fields: Vec<IValue> = args.to_vec();
                Ok(Some(IValue::Ref(self.push_cell(CellKind::Array, ARRAY_TAG, fields))))
            }
            "array_len" => {
                let addr = as_ref(&args[0])?;
                let len = self.heap[addr].as_ref().ok_or("ir: array_len use-after-free")?.fields.len();
                self.drop_value(&IValue::Ref(addr))?; // consume the array arg
                Ok(Some(IValue::Int(len as i64)))
            }
            "array_get" => {
                let addr = as_ref(&args[0])?;
                let i = as_int(&args[1])?;
                let cell = self.heap[addr].as_ref().ok_or("ir: array_get use-after-free")?;
                if i < 0 || i as usize >= cell.fields.len() {
                    return Err(format!(
                        "ir: array_get index {} out of range for array of length {}",
                        i,
                        cell.fields.len()
                    ));
                }
                let elem = cell.fields[i as usize].clone();
                // The element is still owned by the array; give the caller its own
                // reference, then release the (consumed) array.
                self.dup_value(&elem)?;
                self.drop_value(&IValue::Ref(addr))?;
                Ok(Some(elem))
            }
            "array_set" => {
                let addr = as_ref(&args[0])?;
                let i = as_int(&args[1])?;
                let x = args[2].clone();
                let len = self.heap[addr].as_ref().ok_or("ir: array_set use-after-free")?.fields.len();
                if i < 0 || i as usize >= len {
                    return Err(format!(
                        "ir: array_set index {} out of range for array of length {}",
                        i, len
                    ));
                }
                let unique = self.heap[addr].as_ref().unwrap().rc == 1;
                if unique {
                    // FBIP: overwrite in place (x's ownership moves in); drop the
                    // displaced element. No allocation, no change to live count.
                    let old = std::mem::replace(
                        &mut self.heap[addr].as_mut().unwrap().fields[i as usize],
                        x,
                    );
                    self.drop_value(&old)?;
                    self.metrics.reuses += 1;
                    Ok(Some(IValue::Ref(addr)))
                } else {
                    // Copy-on-write: dup every retained field, move x into slot i
                    // (the original's old element stays with the still-shared
                    // original), then release our reference to the original.
                    let fields = self.heap[addr].as_ref().unwrap().fields.clone();
                    for (j, f) in fields.iter().enumerate() {
                        if j != i as usize {
                            self.dup_value(f)?;
                        }
                    }
                    let mut newfields = fields;
                    newfields[i as usize] = x;
                    let naddr = self.push_cell(CellKind::Array, ARRAY_TAG, newfields);
                    self.drop_value(&IValue::Ref(addr))?;
                    Ok(Some(IValue::Ref(naddr)))
                }
            }
            "array_push" => {
                let addr = as_ref(&args[0])?;
                let x = args[1].clone();
                let unique = self.heap[addr].as_ref().ok_or("ir: array_push use-after-free")?.rc == 1;
                if unique {
                    self.heap[addr].as_mut().unwrap().fields.push(x);
                    self.metrics.reuses += 1;
                    Ok(Some(IValue::Ref(addr)))
                } else {
                    let mut fields = self.heap[addr].as_ref().unwrap().fields.clone();
                    for f in &fields {
                        self.dup_value(f)?;
                    }
                    fields.push(x);
                    let naddr = self.push_cell(CellKind::Array, ARRAY_TAG, fields);
                    self.drop_value(&IValue::Ref(addr))?;
                    Ok(Some(IValue::Ref(naddr)))
                }
            }
            _ => Ok(None),
        }
    }

    /// Bytes builtins. Modeled on `array_builtin`: a `Bytes` is a heap cell whose
    /// fields are its bytes (each an `IValue::Int` 0..255). `set`/`push` reuse the
    /// cell in place when uniquely owned (rc == 1) — counting a reuse and leaving
    /// `live` unchanged — else copy-on-write. Bytes hold no nested heap refs, so
    /// there are no per-element dup/drop. Returns `Ok(None)` if not a bytes
    /// builtin. The byte-range policy: a value outside 0..255 on set/push is a
    /// runtime error (identical across all backends).
    fn bytes_builtin(&mut self, name: &str, args: &[IValue]) -> Result<Option<IValue>, String> {
        let as_ref = |v: &IValue| -> Result<usize, String> {
            match v {
                IValue::Ref(a) => Ok(*a),
                _ => Err("ir: bytes builtin expected a Bytes".into()),
            }
        };
        let as_int = |v: &IValue| -> Result<i64, String> {
            match v {
                IValue::Int(n) => Ok(*n),
                _ => Err("ir: bytes builtin expected an Int".into()),
            }
        };
        match name {
            "bytes_new" => {
                Ok(Some(IValue::Ref(self.push_cell(CellKind::Bytes, BYTES_TAG, Vec::new()))))
            }
            "bytes_len" => {
                let addr = as_ref(&args[0])?;
                let len = self.heap[addr].as_ref().ok_or("ir: bytes_len use-after-free")?.fields.len();
                self.drop_value(&IValue::Ref(addr))?; // consume the bytes arg
                Ok(Some(IValue::Int(len as i64)))
            }
            "bytes_get" => {
                let addr = as_ref(&args[0])?;
                let i = as_int(&args[1])?;
                let cell = self.heap[addr].as_ref().ok_or("ir: bytes_get use-after-free")?;
                if i < 0 || i as usize >= cell.fields.len() {
                    return Err(format!(
                        "ir: bytes_get index {} out of range for bytes of length {}",
                        i,
                        cell.fields.len()
                    ));
                }
                let b = match cell.fields[i as usize] {
                    IValue::Int(n) => n,
                    _ => 0,
                };
                self.drop_value(&IValue::Ref(addr))?;
                Ok(Some(IValue::Int(b)))
            }
            "bytes_set" => {
                let addr = as_ref(&args[0])?;
                let i = as_int(&args[1])?;
                let v = as_int(&args[2])?;
                let len = self.heap[addr].as_ref().ok_or("ir: bytes_set use-after-free")?.fields.len();
                if i < 0 || i as usize >= len {
                    return Err(format!(
                        "ir: bytes_set index {} out of range for bytes of length {}",
                        i, len
                    ));
                }
                if !(0..=255).contains(&v) {
                    return Err(format!("ir: bytes_set byte value {} out of range 0..255", v));
                }
                let unique = self.heap[addr].as_ref().unwrap().rc == 1;
                if unique {
                    // FBIP: overwrite in place. No allocation, live unchanged.
                    self.heap[addr].as_mut().unwrap().fields[i as usize] = IValue::Int(v);
                    self.metrics.reuses += 1;
                    Ok(Some(IValue::Ref(addr)))
                } else {
                    // Copy-on-write (bytes have no heap-ref fields to dup).
                    let mut newfields =
                        self.heap[addr].as_ref().unwrap().fields.clone();
                    newfields[i as usize] = IValue::Int(v);
                    let naddr = self.push_cell(CellKind::Bytes, BYTES_TAG, newfields);
                    self.drop_value(&IValue::Ref(addr))?;
                    Ok(Some(IValue::Ref(naddr)))
                }
            }
            "bytes_push" => {
                let addr = as_ref(&args[0])?;
                let v = as_int(&args[1])?;
                if !(0..=255).contains(&v) {
                    return Err(format!("ir: bytes_push byte value {} out of range 0..255", v));
                }
                let unique = self.heap[addr].as_ref().ok_or("ir: bytes_push use-after-free")?.rc == 1;
                if unique {
                    self.heap[addr].as_mut().unwrap().fields.push(IValue::Int(v));
                    self.metrics.reuses += 1;
                    Ok(Some(IValue::Ref(addr)))
                } else {
                    let mut fields = self.heap[addr].as_ref().unwrap().fields.clone();
                    fields.push(IValue::Int(v));
                    let naddr = self.push_cell(CellKind::Bytes, BYTES_TAG, fields);
                    self.drop_value(&IValue::Ref(addr))?;
                    Ok(Some(IValue::Ref(naddr)))
                }
            }
            "bytes_from_str" => {
                // The Str arg is a value type in the IR interpreter (IValue::Str),
                // not a heap Ref, so there is nothing to consume/drop.
                let s = match &args[0] {
                    IValue::Str(s) => s.clone(),
                    _ => return Err("ir: bytes_from_str expected a Str".into()),
                };
                let fields: Vec<IValue> =
                    s.as_bytes().iter().map(|b| IValue::Int(*b as i64)).collect();
                Ok(Some(IValue::Ref(self.push_cell(CellKind::Bytes, BYTES_TAG, fields))))
            }
            "bytes_to_str" => {
                let addr = as_ref(&args[0])?;
                let cell = self.heap[addr].as_ref().ok_or("ir: bytes_to_str use-after-free")?;
                let bs: Vec<u8> = cell
                    .fields
                    .iter()
                    .map(|f| match f {
                        IValue::Int(n) => *n as u8,
                        _ => 0,
                    })
                    .collect();
                let s = match String::from_utf8(bs) {
                    Ok(s) => s,
                    Err(_) => return Err("ir: bytes_to_str: invalid UTF-8".into()),
                };
                self.drop_value(&IValue::Ref(addr))?; // consume the bytes arg
                Ok(Some(IValue::Str(s)))
            }
            _ => Ok(None),
        }
    }

    fn eval_bind(&mut self, bind: &Bind, env: &Env) -> Result<IValue, String> {
        match bind {
            Bind::Atom(a) => self.atom(a, env),
            Bind::Unary(op, a) => {
                let v = self.atom(a, env)?;
                match (op, v) {
                    (UnOp::Neg, IValue::Int(n)) => {
                        Ok(IValue::Int(n.checked_neg().ok_or("ir: neg overflow")?))
                    }
                    (UnOp::Neg, IValue::Float(f)) => Ok(IValue::Float(-f)),
                    (UnOp::Not, IValue::Bool(b)) => Ok(IValue::Bool(!b)),
                    _ => Err("ir: bad unary".into()),
                }
            }
            Bind::Prim(op, a, b) => {
                let x = self.atom(a, env)?;
                let y = self.atom(b, env)?;
                // Eq/Ne work on ADTs too (typeck allows it); compare structurally
                // via the heap, exactly like interp::values_equal. prim() handles
                // the arithmetic/ordering operators on scalars.
                match op {
                    BinOp::Eq | BinOp::Ne => {
                        let equal = self.ir_equal(&x, &y);
                        // The rc pass marks `Eq`/`Ne` operands as CONSUMED (it
                        // dups any value reused later and inserts no drop here),
                        // so this comparison OWNS one reference to each operand
                        // and must release it — after `ir_equal` read the heap —
                        // to stay garbage-free, mirroring the wasm `__eq` site.
                        // Only in rc mode: the un-rc'd baseline never dup'd a
                        // reused operand, so dropping here would double-free e.g.
                        // `v == v`.
                        if self.manage_rc {
                            if let IValue::Ref(a) = x {
                                self.metrics.drops += 1;
                                self.drop_cell(a)?;
                            }
                            if let IValue::Ref(a) = y {
                                self.metrics.drops += 1;
                                self.drop_cell(a)?;
                            }
                        }
                        Ok(IValue::Bool(if *op == BinOp::Eq { equal } else { !equal }))
                    }
                    _ => prim(*op, x, y),
                }
            }
            Bind::Ctor(name, args) => {
                let fields = args.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                Ok(IValue::Ref(self.push_cell(CellKind::Data, name, fields)))
            }
            Bind::CtorReuse(tok, name, args) => {
                let fields: Vec<IValue> =
                    args.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                match self.atom(&Atom::Var(tok.clone()), env)? {
                    IValue::Token(Some(addr)) => {
                        // Reuse the freed slot in place — no allocation.
                        self.metrics.reuses += 1;
                        self.heap[addr] = Some(Cell { kind: CellKind::Data, ctor: name.clone(), fields, rc: 1 });
                        Ok(IValue::Ref(addr))
                    }
                    _ => {
                        // Token empty (cell was shared): allocate fresh.
                        Ok(IValue::Ref(self.push_cell(CellKind::Data, name, fields)))
                    }
                }
            }
            Bind::Call(name, args) => {
                let vals: Vec<IValue> =
                    args.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                // Vectors are FULLY supported by the tree-walking interpreter and
                // the native (C) backend, but NOT by the IR memory path (`aria
                // mem`). Lowering lets `vec_*` through (so the native backend, which
                // shares this lowering, can emit them); reaching the IR interpreter
                // here means we are on the `aria mem` path — gate with a clean error.
                if name.starts_with("vec_") && crate::builtins::lookup(name).is_some() {
                    return Err(format!(
                        "vectors are not yet supported in the IR memory path \
                         (`aria mem`) — use the interpreter `aria run` or the \
                         native backend `aria native-run`: builtin `{}`",
                        name
                    ));
                }
                // Array builtins manage heap cells + reference counts directly
                // (they need `&mut self`), so they are handled here rather than in
                // the `&self` `builtin` helper used for the unboxed builtins.
                if let Some(v) = self.array_builtin(name, &vals)? {
                    return Ok(v);
                }
                if let Some(v) = self.bytes_builtin(name, &vals)? {
                    return Ok(v);
                }
                if let Some(v) = self.builtin(name, &vals)? {
                    return Ok(v);
                }
                let f = self.fns.get(name).cloned().ok_or_else(|| format!("ir: unknown fn {}", name))?;
                if f.params.len() != vals.len() {
                    return Err(format!(
                        "ir: function `{}` expects {} argument(s), got {}",
                        name,
                        f.params.len(),
                        vals.len()
                    ));
                }
                let d = self.depth + 1;
                if d > MAX_IR_CALL_DEPTH {
                    return Err(format!("ir: maximum recursion depth ({}) exceeded", MAX_IR_CALL_DEPTH));
                }
                self.depth = d;
                let mut frame = Env::new();
                for (p, v) in f.params.iter().zip(vals.into_iter()) {
                    frame.insert(p.clone(), v);
                }
                let result = self.eval(&f.body, frame);
                self.depth = d - 1;
                result
            }
            Bind::If(c, then, els) => match self.atom(c, env)? {
                IValue::Bool(true) => self.eval(then, env.clone()),
                IValue::Bool(false) => self.eval(els, env.clone()),
                _ => Err("ir: if cond not bool".into()),
            },
            Bind::Match(scrut, arms) => {
                let v = self.atom(scrut, env)?;
                let addr = match v {
                    IValue::Ref(a) => a,
                    _ => return Err("ir: match on non-data".into()),
                };
                let cell = self
                    .heap[addr]
                    .clone()
                    .ok_or("ir: use-after-free in match")?;
                for arm in arms {
                    let matches = match &arm.ctor {
                        Some(c) => *c == cell.ctor,
                        None => true, // catch-all
                    };
                    if matches {
                        let mut frame = env.clone();
                        if arm.ctor.is_some() {
                            for (b, f) in arm.binders.iter().zip(cell.fields.iter()) {
                                frame.insert(b.clone(), f.clone());
                            }
                        } else if let Some(b) = arm.binders.first() {
                            frame.insert(b.clone(), IValue::Ref(addr));
                        }
                        return self.eval(&arm.body, frame);
                    }
                }
                Err(format!("ir: no match arm for {}", cell.ctor))
            }
            // A closure: a heap cell whose constructor tag is the lifted lambda's
            // name and whose fields are the captured values — exactly like `Ctor`.
            Bind::MakeClosure(lam, caps) => {
                let fields = caps.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                Ok(IValue::Ref(self.push_cell(CellKind::Data, lam, fields)))
            }
            // Apply a closure: read its lambda tag, bind the captured fields plus
            // the argument atoms, and run the lifted lambda body. The captures are
            // dup'd out of the cell so the body owns them exactly like parameters
            // (the body's rc'd drops then balance); the closure cell itself is
            // dropped by the rc pass's inserted `Drop`, not here.
            Bind::ApplyClosure(callee, args, _) => {
                let cv = self.atom(callee, env)?;
                let addr = match cv {
                    IValue::Ref(a) => a,
                    _ => return Err("ir: apply of non-closure value".into()),
                };
                let cell = self.heap[addr].clone().ok_or("ir: apply use-after-free")?;
                let f = self
                    .fns
                    .get(&cell.ctor)
                    .cloned()
                    .ok_or_else(|| format!("ir: apply of unknown lambda `{}`", cell.ctor))?;
                let argvals: Vec<IValue> =
                    args.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                if f.captures.len() != cell.fields.len() || f.params.len() != argvals.len() {
                    return Err(format!(
                        "ir: closure `{}` arity mismatch (captures {}/{}, params {}/{})",
                        cell.ctor,
                        f.captures.len(),
                        cell.fields.len(),
                        f.params.len(),
                        argvals.len()
                    ));
                }
                let d = self.depth + 1;
                if d > MAX_IR_CALL_DEPTH {
                    return Err(format!("ir: maximum recursion depth ({}) exceeded", MAX_IR_CALL_DEPTH));
                }
                self.depth = d;
                let mut frame = Env::new();
                for (cn, fv) in f.captures.iter().zip(cell.fields.iter()) {
                    if self.manage_rc {
                        if let IValue::Ref(a) = fv {
                            match &mut self.heap[*a] {
                                Some(c) => c.rc += 1,
                                None => return Err("ir: dup of freed capture".into()),
                            }
                            self.metrics.dups += 1;
                        }
                    }
                    frame.insert(cn.clone(), fv.clone());
                }
                for (p, v) in f.params.iter().zip(argvals.into_iter()) {
                    frame.insert(p.clone(), v);
                }
                let result = self.eval(&f.body, frame);
                self.depth = d - 1;
                let result = result?;
                // This application owns one reference to the closure cell (the rc
                // pass dup'd it for any further use); release it now. The body
                // borrowed the captures (dup'ing each), so this only drops this
                // application's hold; at rc 0 the captured fields are released too.
                if self.manage_rc {
                    self.metrics.drops += 1;
                    self.drop_cell(addr)?;
                }
                Ok(result)
            }
        }
    }

    fn builtin(&self, name: &str, args: &[IValue]) -> Result<Option<IValue>, String> {
        match name {
            "print_int" => match args {
                [IValue::Int(n)] => {
                    println!("{}", n);
                    Ok(Some(IValue::Unit))
                }
                _ => Err("ir: print_int expects Int".into()),
            },
            "print_float" => match args {
                [IValue::Float(f)] => {
                    println!("{}", f);
                    Ok(Some(IValue::Unit))
                }
                _ => Err("ir: print_float expects Float".into()),
            },
            "print_bool" => match args {
                [IValue::Bool(b)] => {
                    println!("{}", b);
                    Ok(Some(IValue::Unit))
                }
                _ => Err("ir: print_bool expects Bool".into()),
            },
            "print_str" => match args {
                [IValue::Str(s)] => {
                    println!("{}", s);
                    Ok(Some(IValue::Unit))
                }
                _ => Err("ir: print_str expects String".into()),
            },
            "concat" => match args {
                [IValue::Str(a), IValue::Str(b)] => Ok(Some(IValue::Str(format!("{}{}", a, b)))),
                _ => Err("ir: concat expects two Strings".into()),
            },
            "int_to_str" => match args {
                [IValue::Int(n)] => Ok(Some(IValue::Str(n.to_string()))),
                _ => Err("ir: int_to_str expects Int".into()),
            },
            _ => Ok(None),
        }
    }
}

fn prim(op: BinOp, x: IValue, y: IValue) -> Result<IValue, String> {
    use BinOp::*;
    use IValue::*;
    match (x, y) {
        (Int(a), Int(b)) => Ok(match op {
            Add => Int(a.checked_add(b).ok_or("ir: + overflow")?),
            Sub => Int(a.checked_sub(b).ok_or("ir: - overflow")?),
            Mul => Int(a.checked_mul(b).ok_or("ir: * overflow")?),
            Div => {
                if b == 0 {
                    return Err("division by zero".into());
                }
                Int(a.checked_div(b).ok_or("integer overflow in `/`")?)
            }
            Mod => {
                if b == 0 {
                    return Err("modulo by zero".into());
                }
                Int(a.checked_rem(b).ok_or("integer overflow in `%`")?)
            }
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            Lt => Bool(a < b),
            Le => Bool(a <= b),
            Gt => Bool(a > b),
            Ge => Bool(a >= b),
            And | Or => return Err("ir: logical op on Int".into()),
        }),
        // Float arithmetic mirrors interp::float_op: Div by 0.0 yields
        // infinity/NaN (NOT an error), and there is no float `Mod`.
        (Float(a), Float(b)) => Ok(match op {
            Add => Float(a + b),
            Sub => Float(a - b),
            Mul => Float(a * b),
            Div => Float(a / b),
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            Lt => Bool(a < b),
            Le => Bool(a <= b),
            Gt => Bool(a > b),
            Ge => Bool(a >= b),
            Mod | And | Or => return Err("ir: bad float op".into()),
        }),
        (Bool(a), Bool(b)) => Ok(match op {
            And => Bool(a && b),
            Or => Bool(a || b),
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            _ => return Err("ir: bad bool op".into()),
        }),
        // Strings support only equality (matching interp::values_equal).
        (Str(a), Str(b)) => Ok(match op {
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            _ => return Err("ir: bad string op".into()),
        }),
        _ => Err("ir: prim type mismatch".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interp, lexer, parser, typeck};

    // Lower + run through the IR interpreter, returning a STRUCTURAL string
    // identical to interp::Value::display (so Data/Bool/Unit compare faithfully).
    fn ir_run(src: &str) -> Result<String, String> {
        let toks = lexer::lex(src)?;
        let prog = parser::parse(toks)?;
        typeck::check(&prog).map_err(|e| e.join("; "))?;
        let fns = lower_program(&prog)?;
        let mut ir = IrInterp::new(fns);
        let v = ir.run_main()?;
        Ok(ir.render(&v))
    }

    // The tree-walker's result, for differential comparison.
    fn ast_run(src: &str) -> Result<String, String> {
        let toks = lexer::lex(src)?;
        let prog = parser::parse(toks)?;
        let it = interp::Interp::new(&prog)?;
        it.run_main().map(|v| v.display())
    }

    // The IR and interpreter must agree on the Ok/Err SHAPE and, when Ok, on the
    // exact structural value. (Both erroring counts as agreement; messages may
    // differ.)
    fn differential(src: &str) {
        let ir = ir_run(src);
        let ast = ast_run(src);
        match (&ir, &ast) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "value mismatch for:\n{}", src),
            (Err(_), Err(_)) => {}
            _ => panic!("Ok/Err shape mismatch for:\n{}\n  ir={:?}\n  ast={:?}", src, ir, ast),
        }
    }

    #[test]
    fn factorial_matches_interpreter() {
        differential("fn fac(n: Int) -> Int = match n { 0 => 1, _ => n * fac(n - 1), }\nfn main() -> Int = fac(6)");
    }

    #[test]
    fn function_values_lower_and_match_interpreter() {
        // Lambdas and higher-order calls now lower to closures (lifted lambdas +
        // closure cells) and the IR interpreter agrees with the tree-walker and
        // stays garbage-free — the native backend compiles these (the wasm
        // backend still rejects them for now).
        differential(
            "fn apply1(f: (Int) -> Int, x: Int) -> Int = f(x)\nfn main() -> Int = apply1(\\y -> y + 1, 41)",
        );
        // A captured variable, an immediately-applied lambda, and currying.
        differential("fn main() -> Int = (\\x -> x * 2)(21)");
        differential(
            "fn add(x: Int) -> (Int) -> Int = \\y -> x + y\nfn main() -> Int = add(3)(4)",
        );
    }

    #[test]
    fn list_sum_matches_interpreter_and_counts_allocs() {
        let src = "type L = | Nil | Cons(Int, L)\n\
                   fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
                   fn main() -> Int = sum(Cons(1, Cons(2, Cons(3, Nil))))";
        differential(src);
        let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
        let mut ir = IrInterp::new(lower_program(&prog).unwrap());
        ir.run_main().unwrap();
        assert_eq!(ir.metrics.allocations, 4, "expected 4 ADT allocations");
    }

    #[test]
    fn returns_constructed_value_structurally() {
        // main returns a Data value; render must match interp's display exactly.
        let src = "type P = | Pair(Int, Int)\nfn main() -> P = Pair(1, 2)";
        let ir = ir_run(src).unwrap();
        let ast = ast_run(src).unwrap();
        assert_eq!(ir, ast);
        assert_eq!(ir, "Pair(1, 2)");
    }

    #[test]
    fn if_and_bool_match() {
        differential("fn classify(n: Int) -> Int = if n < 0 { 0 } else { 1 }\nfn main() -> Int = classify(5)");
    }

    #[test]
    fn float_negation_matches_interpreter() {
        differential("fn main() -> Float = -2.5");
        differential("fn neg(x: Float) -> Float = -x\nfn main() -> Float = neg(3.5)");
    }

    #[test]
    fn adt_equality_matches_interpreter() {
        differential("type P = | P(Int, Int)\nfn main() -> Bool = P(1, 2) == P(1, 2)");
        differential("type P = | P(Int, Int)\nfn main() -> Bool = P(1, 2) == P(1, 9)");
        // structural over String fields too
        differential("type R = | R(String, Int)\nfn main() -> Bool = R(\"a\", 1) == R(\"a\", 1)");
    }

    #[test]
    fn tensor_builtin_lowers_for_wasm() {
        // Phase 2f: tensor builtins now lower (the wasm backend compiles them).
        let prog = parser::parse(
            lexer::lex("fn main() -> Float = tensor_get(tensor_zeros(2, 2), 0, 0)").unwrap(),
        )
        .unwrap();
        assert!(
            lower_program(&prog).is_ok(),
            "tensor builtins must lower for the wasm backend"
        );
    }

    #[test]
    fn codec_builtin_rejected_at_lowering() {
        // The rANS/predictor codec builtins remain outside the IR subset.
        let prog = parser::parse(
            lexer::lex("fn main() -> Int = compressed_size(\"hello\")").unwrap(),
        )
        .unwrap();
        assert!(
            lower_program(&prog).is_err(),
            "codec builtins must be rejected by IR lowering"
        );
    }

    // ---- regression tests for the adversarial-review findings ------------

    #[test]
    fn catchall_before_literal_first_match_wins() {
        // Interp takes the first (catch-all) arm; IR must too (not the literal).
        differential("fn f(n: Int) -> Int = match n { _ => 0, 1 => 99, }\nfn main() -> Int = f(1)");
    }

    #[test]
    fn literal_after_catchall_is_dead() {
        differential("fn f(n: Int) -> Int = match n { 1 => 10, x => 100, 2 => 20, }\nfn main() -> Int = f(2)");
    }

    #[test]
    fn first_of_two_catchalls_wins() {
        differential("fn f(n: Int) -> Int = match n { x => x, y => y + 1000, }\nfn main() -> Int = f(5)");
    }

    #[test]
    fn short_circuit_and_does_not_eval_rhs() {
        // false && (1/0 == 0): interp short-circuits to false (no div error); IR must too.
        differential("fn main() -> Int = { let b = false && (1 / 0 == 0); if b { 1 } else { 0 } }");
    }

    #[test]
    fn short_circuit_or_does_not_eval_rhs() {
        differential("fn main() -> Int = { let b = true || (1 / 0 == 0); if b { 1 } else { 0 } }");
    }

    #[test]
    fn div_by_zero_errors_in_both() {
        differential("fn main() -> Int = 1 / 0");
    }

    #[test]
    fn add_overflow_errors_in_both() {
        differential("fn main() -> Int = 9223372036854775807 + 1");
    }

    // ---- Float / String subset -----------------------------------------

    #[test]
    fn float_arithmetic_matches_interpreter() {
        // Float add/mul/div + a comparison; IR result must equal the interp's.
        differential("fn area(r: Float) -> Float = 3.14159 * r * r\nfn main() -> Float = area(2.0) / 2.0");
    }

    #[test]
    fn string_concat_and_int_to_str_matches_interpreter() {
        differential("fn main() -> String = concat(\"x = \", int_to_str(6 * 7))");
    }

    #[test]
    fn adt_with_string_fields_is_garbage_free() {
        // A list of records carrying String fields: after the run returns an
        // Int, every ADT cell (including those holding Strings) must be freed.
        let src = "type Rec = | R(String, Int)\n\
                   type L = | Nil | Cons(Rec, L)\n\
                   fn count(xs: L) -> Int = match xs { Nil => 0, Cons(_, r) => 1 + count(r), }\n\
                   fn main() -> Int = count(Cons(R(\"a\", 1), Cons(R(\"b\", 2), Nil)))";
        differential(src);
        let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
        typeck::check(&prog).unwrap();
        let fns = crate::rc::insert_rc(&lower_program(&prog).unwrap());
        let mut ir = IrInterp::new(fns);
        ir.run_main().unwrap();
        assert_eq!(ir.metrics.live, 0, "all String-carrying ADT cells must be freed");
    }

    #[test]
    fn deep_recursion_does_not_abort() {
        // Run on a large-stack thread (as the CLI does) and confirm a depth-3000
        // recursion returns the correct value rather than overflowing the stack.
        let src = "fn count(n: Int) -> Int = if n == 0 { 0 } else { count(n - 1) + 1 }\nfn main() -> Int = count(3000)";
        let out = std::thread::Builder::new()
            .stack_size(1 << 30)
            .spawn(move || {
                let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
                let mut ir = IrInterp::new(lower_program(&prog).unwrap());
                let v = ir.run_main().expect("ir run");
                ir.render(&v)
            })
            .unwrap()
            .join()
            .expect("ir thread must not abort");
        assert_eq!(out, "3000");
    }

    // ---- self-tail-call optimization pass ------------------------------

    /// Does an IExpr contain a `TailCall` anywhere?
    fn has_tail_call(e: &IExpr) -> bool {
        match e {
            IExpr::TailCall(_) => true,
            IExpr::Ret(_) => false,
            IExpr::Let(_, bind, body) => {
                let in_bind = match bind {
                    Bind::If(_, t, el) => has_tail_call(t) || has_tail_call(el),
                    Bind::Match(_, arms) => arms.iter().any(|a| has_tail_call(&a.body)),
                    _ => false,
                };
                in_bind || has_tail_call(body)
            }
            IExpr::Dup(_, b) | IExpr::Drop(_, b) | IExpr::DropReuse(_, _, b) => has_tail_call(b),
        }
    }

    #[test]
    fn tco_pass_marks_and_rewrites_self_tail_call() {
        // A self-tail-recursive accumulator: after `tail_call_optimize`, `go` is
        // flagged tail_recursive and its body contains a `TailCall`; the
        // non-recursive `main` is left untouched.
        let src = "fn go(n: Int, acc: Int) -> Int = if n == 0 { acc } else { go(n - 1, acc + n) }\n\
                   fn main() -> Int = go(10, 0)";
        let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
        let rcd = crate::rc::insert_rc(&lower_program(&prog).unwrap());
        let tco = tail_call_optimize(rcd);
        let go = tco.get("go").unwrap();
        assert!(go.tail_recursive, "`go` must be marked self-tail-recursive");
        assert!(has_tail_call(&go.body), "`go` body must contain a TailCall");
        let main = tco.get("main").unwrap();
        assert!(!main.tail_recursive, "`main` is not self-tail-recursive");
        assert!(!has_tail_call(&main.body));
    }

    #[test]
    fn tco_does_not_rewrite_non_tail_self_call() {
        // The recursive call is an operand of `+` (NOT in tail position), so the
        // pass must NOT mark the function or insert a TailCall.
        let src = "fn sumto(n: Int) -> Int = if n == 0 { 0 } else { n + sumto(n - 1) }\n\
                   fn main() -> Int = sumto(5)";
        let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
        let rcd = crate::rc::insert_rc(&lower_program(&prog).unwrap());
        let tco = tail_call_optimize(rcd);
        let f = tco.get("sumto").unwrap();
        assert!(!f.tail_recursive, "non-tail self-call must NOT be optimized");
        assert!(!has_tail_call(&f.body));
    }
}


