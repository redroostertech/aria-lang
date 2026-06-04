//! Static type checker for Aria.
//!
//! This is the keystone of the AI-native thesis: the compiler is the model's
//! correctness signal. The checker is sound and *bottom-up* — it synthesizes a
//! type for every expression and checks it against the declared annotations on
//! functions, constructors, and `let` bindings.
//!
//! Aria now supports parametric polymorphism (generics). Types may contain
//! unification variables (`Ty::Var`). The checker uses a Hindley-Milner-style
//! substitution map keyed on fresh-variable names: at each use of a generic
//! constructor or generic function, its declared type parameters are
//! INSTANTIATED with fresh unification variables, then unification drives the
//! variables to concrete types (or reports a clear mismatch).
//!
//! Beyond ordinary type mismatches it enforces the two things that most reduce
//! generated-code bugs:
//!   * exhaustive `match` — every constructor of an ADT must be handled (or a
//!     wildcard provided), so "forgot a case" is a compile error;
//!   * arity/field checks on calls and constructors.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::ast::*;

/// Render a type for diagnostics, resolving any solved variables first.
pub fn show(t: &Ty) -> String {
    match t {
        Ty::Int => "Int".to_string(),
        Ty::Float => "Float".to_string(),
        Ty::Bool => "Bool".to_string(),
        Ty::Str => "String".to_string(),
        Ty::Unit => "Unit".to_string(),
        Ty::Var(n) => n.clone(),
        Ty::Named(n, args) => {
            if args.is_empty() {
                n.clone()
            } else {
                let inner: Vec<String> = args.iter().map(show).collect();
                format!("{}[{}]", n, inner.join(", "))
            }
        }
        Ty::Fn(params, ret) => {
            let inner: Vec<String> = params.iter().map(show).collect();
            format!("({}) -> {}", inner.join(", "), show(ret))
        }
    }
}

type Scope = Vec<HashMap<String, Ty>>;

/// A type variable is a solvable unification variable iff its name starts with
/// `?` (produced by `fresh`). Declared generic parameters keep their source
/// name (e.g. `T`) and are therefore rigid — `unify` never binds them.
fn is_fresh(name: &str) -> bool {
    name.starts_with('?')
}

// A constructor's declared signature, retaining the owning type's generic
// parameters so each use can be instantiated with fresh variables.
#[derive(Clone)]
struct CtorSig {
    type_params: Vec<String>, // generic params of the owning type
    fields: Vec<Ty>,          // field types (may mention the params as Ty::Var)
    tyname: String,           // owning type name
}

// A function's declared signature, retaining its generic parameters.
#[derive(Clone)]
struct FnSig {
    type_params: Vec<String>,
    params: Vec<Ty>,
    ret: Ty,
}

struct Checker {
    fns: HashMap<String, FnSig>,
    ctors: HashMap<String, CtorSig>,
    types: HashMap<String, (Vec<String>, Vec<String>)>, // type -> (params, variant ctor names)
    // Union-find / substitution for unification variables, plus a fresh counter.
    subst: RefCell<HashMap<String, Ty>>,
    counter: RefCell<u64>,
}

/// Type-check a whole program. Returns every error found, not just the first.
pub fn check(program: &Program) -> Result<(), Vec<String>> {
    let mut fns: HashMap<String, FnSig> = HashMap::new();
    let mut ctors: HashMap<String, CtorSig> = HashMap::new();
    let mut types: HashMap<String, (Vec<String>, Vec<String>)> = HashMap::new();
    let mut errors: Vec<String> = Vec::new();

    // Pass 1: gather declarations.
    for item in &program.items {
        match item {
            Item::Type(t) => {
                let mut variants = Vec::new();
                for v in &t.variants {
                    variants.push(v.name.clone());
                    if ctors
                        .insert(
                            v.name.clone(),
                            CtorSig {
                                type_params: t.params.clone(),
                                fields: v.fields.clone(),
                                tyname: t.name.clone(),
                            },
                        )
                        .is_some()
                    {
                        errors.push(format!("duplicate constructor `{}`", v.name));
                    }
                }
                if types
                    .insert(t.name.clone(), (t.params.clone(), variants))
                    .is_some()
                {
                    errors.push(format!("duplicate type `{}`", t.name));
                }
            }
            Item::Fn(f) => {
                let params = f.params.iter().map(|p| p.ty.clone()).collect();
                if fns
                    .insert(
                        f.name.clone(),
                        FnSig {
                            type_params: f.type_params.clone(),
                            params,
                            ret: f.ret.clone(),
                        },
                    )
                    .is_some()
                {
                    errors.push(format!("duplicate function `{}`", f.name));
                }
            }
        }
    }

    // Pass 2: every Named type referenced must be defined, with the right arity,
    // and its arguments must in turn be valid. Type variables in scope are fine.
    fn known(
        t: &Ty,
        types: &HashMap<String, (Vec<String>, Vec<String>)>,
        params: &[String],
        errs: &mut Vec<String>,
        ctx: &str,
    ) {
        match t {
            Ty::Named(n, args) if BUILTIN_TYPES.contains(&n.as_str()) => {
                // Built-in nullary types (e.g. `Tensor`) take no arguments.
                if !args.is_empty() {
                    errs.push(format!(
                        "{}: built-in type `{}` takes no type arguments, got {}",
                        ctx,
                        n,
                        args.len()
                    ));
                }
            }
            Ty::Named(n, args) => {
                match types.get(n) {
                    None => errs.push(format!("{}: unknown type `{}`", ctx, n)),
                    Some((decl_params, _)) => {
                        if decl_params.len() != args.len() {
                            errs.push(format!(
                                "{}: type `{}` expects {} type argument(s), got {}",
                                ctx,
                                n,
                                decl_params.len(),
                                args.len()
                            ));
                        }
                    }
                }
                for a in args {
                    known(a, types, params, errs, ctx);
                }
            }
            Ty::Var(_) => {} // a declared generic parameter; fine
            Ty::Fn(fn_params, ret) => {
                for p in fn_params {
                    known(p, types, params, errs, ctx);
                }
                known(ret, types, params, errs, ctx);
            }
            _ => {}
        }
    }
    for item in &program.items {
        match item {
            Item::Fn(f) => {
                for p in &f.params {
                    known(
                        &p.ty,
                        &types,
                        &f.type_params,
                        &mut errors,
                        &format!("function `{}` parameter `{}`", f.name, p.name),
                    );
                }
                known(
                    &f.ret,
                    &types,
                    &f.type_params,
                    &mut errors,
                    &format!("function `{}` return type", f.name),
                );
            }
            Item::Type(t) => {
                for v in &t.variants {
                    for ft in &v.fields {
                        known(
                            ft,
                            &types,
                            &t.params,
                            &mut errors,
                            &format!("constructor `{}` field", v.name),
                        );
                    }
                }
            }
        }
    }

    let checker = Checker {
        fns,
        ctors,
        types,
        subst: RefCell::new(HashMap::new()),
        counter: RefCell::new(0),
    };

    // Pass 3: check each function body against its declared return type. Each
    // function gets a fresh unification context so leftover variables from one
    // body cannot leak into another.
    for item in &program.items {
        if let Item::Fn(f) = item {
            checker.subst.borrow_mut().clear();
            let mut scope: Scope = vec![HashMap::new()];
            for p in &f.params {
                scope[0].insert(p.name.clone(), p.ty.clone());
            }
            match checker.synth(&f.body, &mut scope) {
                Ok(t) => {
                    if let Err(e) = checker.unify(&t, &f.ret) {
                        errors.push(format!(
                            "function `{}`: body has type {} but return type is {} ({})",
                            f.name,
                            show(&checker.resolve(&t)),
                            show(&checker.resolve(&f.ret)),
                            e
                        ));
                    }
                }
                Err(e) => errors.push(format!("function `{}`: {}", f.name, e)),
            }
        }
    }

    // Pass 4: effect inference and `pure` checking.
    //
    // Aria has a single effect, `IO`, produced only by the four `print_*`
    // builtins. A function is IO iff it (transitively) calls a `print_*` builtin
    // or another IO function. We compute the IO set by a fixpoint over the call
    // graph (which naturally handles mutual recursion), then reject any function
    // annotated `pure` whose inferred effect set contains IO. Effects are erased
    // after this pass: it changes no runtime behavior and no backend codegen.
    let io = infer_io(program);
    for item in &program.items {
        if let Item::Fn(f) = item {
            if f.pure && io.contains(&f.name) {
                // Report a concrete cause: a directly-called `print_*` builtin if
                // there is one, otherwise the transitive nature of the violation.
                let reason = match direct_io_builtin(&f.body) {
                    Some(b) => format!("calls `{}`", b),
                    None => "transitively calls IO code".to_string(),
                };
                errors.push(format!(
                    "function `{}` is declared `pure` but performs IO ({})",
                    f.name, reason
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Names of the IO-producing builtins. Everything else is pure.
const IO_BUILTINS: &[&str] = &["print_int", "print_float", "print_bool", "print_str"];

/// Collect the names called (directly) inside an expression, into `out`. This is
/// a pure syntactic walk over calls; it records both builtin and user-function
/// callees by name.
fn collect_calls(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Unit
        | Expr::Var(_) => {}
        Expr::Ctor(_, args) => {
            for a in args {
                collect_calls(a, out);
            }
        }
        Expr::Call(name, args) => {
            out.insert(name.clone());
            for a in args {
                collect_calls(a, out);
            }
        }
        Expr::Lambda(_, body, _) => collect_calls(body, out),
        Expr::Apply(callee, args, _) => {
            collect_calls(callee, out);
            for a in args {
                collect_calls(a, out);
            }
        }
        Expr::Unary(_, x) => collect_calls(x, out),
        Expr::Binary(_, a, b) => {
            collect_calls(a, out);
            collect_calls(b, out);
        }
        Expr::If(c, t, e2) => {
            collect_calls(c, out);
            collect_calls(t, out);
            collect_calls(e2, out);
        }
        Expr::Match(scrut, arms) => {
            collect_calls(scrut, out);
            for a in arms {
                collect_calls(&a.body, out);
            }
        }
        Expr::Block(stmts, tail) => {
            for s in stmts {
                match s {
                    Stmt::Let(_, _, x) => collect_calls(x, out),
                    Stmt::Expr(x) => collect_calls(x, out),
                }
            }
            collect_calls(tail, out);
        }
    }
}

/// If the expression directly calls an IO builtin, return its name (for a clearer
/// error message). Only inspects direct calls, not transitive ones.
fn direct_io_builtin(e: &Expr) -> Option<String> {
    let mut calls = HashSet::new();
    collect_calls(e, &mut calls);
    IO_BUILTINS
        .iter()
        .find(|b| calls.contains(**b))
        .map(|b| b.to_string())
}

/// Infer the set of user-function names that perform IO. A function is IO iff it
/// (transitively) calls a `print_*` builtin or another IO function. Computed by a
/// least-fixpoint over the call graph: start with every function pure, then keep
/// marking functions IO whose body calls a `print_*` builtin or an already-IO
/// function, until nothing changes. This converges (the IO set only grows and is
/// bounded by the function count) and handles mutual recursion correctly.
fn infer_io(program: &Program) -> HashSet<String> {
    // Precompute each function's direct callee set.
    let mut callees: HashMap<String, HashSet<String>> = HashMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            let mut c = HashSet::new();
            collect_calls(&f.body, &mut c);
            callees.insert(f.name.clone(), c);
        }
    }

    let mut io: HashSet<String> = HashSet::new();
    loop {
        let mut changed = false;
        for (name, c) in &callees {
            if io.contains(name) {
                continue;
            }
            let performs_io = c.iter().any(|callee| {
                IO_BUILTINS.contains(&callee.as_str()) || io.contains(callee)
            });
            if performs_io {
                io.insert(name.clone());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    io
}

// Built-in nullary types that need no user declaration but must pass the
// "unknown type" check in pass 2. `Tensor` is the opaque AI-runtime handle.
// Builtin function signatures and built-in type names come from the shared
// `crate::builtins` source of truth so typeck and interp cannot drift.
use crate::builtins::{lookup as builtin_sig, BUILTIN_TYPES};

impl Checker {
    fn lookup_var(scope: &Scope, name: &str) -> Option<Ty> {
        for frame in scope.iter().rev() {
            if let Some(t) = frame.get(name) {
                return Some(t.clone());
            }
        }
        None
    }

    // ---- unification primitives -----------------------------------------

    // Allocate a fresh unification variable.
    fn fresh(&self) -> Ty {
        let mut c = self.counter.borrow_mut();
        let id = *c;
        *c += 1;
        Ty::Var(format!("?{}", id))
    }

    // Instantiate a list of declared parameter names with fresh variables,
    // returning the substitution to apply to a signature.
    fn instantiate_map(&self, params: &[String]) -> HashMap<String, Ty> {
        params.iter().map(|p| (p.clone(), self.fresh())).collect()
    }

    // Substitute declared parameter names (as `Ty::Var`) using `map`.
    fn apply_map(ty: &Ty, map: &HashMap<String, Ty>) -> Ty {
        match ty {
            Ty::Var(n) => map.get(n).cloned().unwrap_or_else(|| ty.clone()),
            Ty::Named(n, args) => {
                Ty::Named(n.clone(), args.iter().map(|a| Self::apply_map(a, map)).collect())
            }
            Ty::Fn(params, ret) => Ty::Fn(
                params.iter().map(|p| Self::apply_map(p, map)).collect(),
                Box::new(Self::apply_map(ret, map)),
            ),
            other => other.clone(),
        }
    }

    // Follow the substitution chain shallowly (one level) for a variable.
    fn prune(&self, ty: &Ty) -> Ty {
        if let Ty::Var(n) = ty {
            if let Some(bound) = self.subst.borrow().get(n).cloned() {
                return self.prune(&bound);
            }
        }
        ty.clone()
    }

    // Fully resolve a type by substituting all solved variables, for display.
    fn resolve(&self, ty: &Ty) -> Ty {
        let t = self.prune(ty);
        match t {
            Ty::Named(n, args) => {
                Ty::Named(n, args.iter().map(|a| self.resolve(a)).collect())
            }
            Ty::Fn(params, ret) => Ty::Fn(
                params.iter().map(|p| self.resolve(p)).collect(),
                Box::new(self.resolve(&ret)),
            ),
            other => other,
        }
    }

    fn occurs(&self, var: &str, ty: &Ty) -> bool {
        match self.prune(ty) {
            Ty::Var(n) => n == var,
            Ty::Named(_, args) => args.iter().any(|a| self.occurs(var, a)),
            Ty::Fn(params, ret) => {
                params.iter().any(|p| self.occurs(var, p)) || self.occurs(var, &ret)
            }
            _ => false,
        }
    }

    // Unify two types, recording variable bindings in the substitution.
    fn unify(&self, a: &Ty, b: &Ty) -> Result<(), String> {
        let a = self.prune(a);
        let b = self.prune(b);
        match (&a, &b) {
            (Ty::Int, Ty::Int)
            | (Ty::Float, Ty::Float)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Str, Ty::Str)
            | (Ty::Unit, Ty::Unit) => Ok(()),
            (Ty::Var(x), Ty::Var(y)) if x == y => Ok(()),
            // Only FRESH unification variables (`?N`) may be bound. A declared
            // generic parameter (e.g. `T`) is rigid/skolem: it unifies only with
            // itself (the equal-name case above) or with a fresh variable. This
            // keeps parametricity sound — a function body cannot silently
            // constrain its own type parameter to a concrete type.
            (Ty::Var(x), _) if is_fresh(x) => {
                if self.occurs(x, &b) {
                    return Err(format!("infinite type: {} occurs in {}", x, show(&self.resolve(&b))));
                }
                self.subst.borrow_mut().insert(x.clone(), b.clone());
                Ok(())
            }
            (_, Ty::Var(y)) if is_fresh(y) => {
                if self.occurs(y, &a) {
                    return Err(format!("infinite type: {} occurs in {}", y, show(&self.resolve(&a))));
                }
                self.subst.borrow_mut().insert(y.clone(), a.clone());
                Ok(())
            }
            (Ty::Named(n1, a1), Ty::Named(n2, a2)) if n1 == n2 && a1.len() == a2.len() => {
                for (x, y) in a1.iter().zip(a2.iter()) {
                    self.unify(x, y)?;
                }
                Ok(())
            }
            // Function types unify structurally: same arity, params + return.
            (Ty::Fn(p1, r1), Ty::Fn(p2, r2)) if p1.len() == p2.len() => {
                for (x, y) in p1.iter().zip(p2.iter()) {
                    self.unify(x, y)?;
                }
                self.unify(r1, r2)
            }
            _ => Err(format!(
                "expected {}, found {}",
                show(&self.resolve(&a)),
                show(&self.resolve(&b))
            )),
        }
    }

    // ---- synthesis -------------------------------------------------------

    fn synth(&self, e: &Expr, scope: &mut Scope) -> Result<Ty, String> {
        match e {
            Expr::Int(_) => Ok(Ty::Int),
            Expr::Float(_) => Ok(Ty::Float),
            Expr::Bool(_) => Ok(Ty::Bool),
            Expr::Str(_) => Ok(Ty::Str),
            Expr::Unit => Ok(Ty::Unit),

            Expr::Var(name) => {
                if let Some(t) = Checker::lookup_var(scope, name) {
                    Ok(t)
                } else if let Some(sig) = self.fns.get(name).cloned() {
                    // A bare top-level function name used as a VALUE: its type is
                    // its function type, with generic parameters instantiated.
                    let map = self.instantiate_map(&sig.type_params);
                    let params = sig.params.iter().map(|p| Self::apply_map(p, &map)).collect();
                    let ret = Self::apply_map(&sig.ret, &map);
                    Ok(Ty::Fn(params, Box::new(ret)))
                } else {
                    Err(format!("unbound variable `{}`", name))
                }
            }

            Expr::Ctor(name, args) => {
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown constructor `{}`", name))?;
                if args.len() != sig.fields.len() {
                    return Err(format!(
                        "constructor `{}` expects {} field(s), got {}",
                        name,
                        sig.fields.len(),
                        args.len()
                    ));
                }
                // Instantiate the owning type's parameters with fresh variables.
                let map = self.instantiate_map(&sig.type_params);
                for (i, (arg, ft)) in args.iter().zip(sig.fields.iter()).enumerate() {
                    let at = self.synth(arg, scope)?;
                    let expected = Self::apply_map(ft, &map);
                    self.unify(&expected, &at).map_err(|e| {
                        format!("constructor `{}` field {}: {}", name, i, e)
                    })?;
                }
                let type_args: Vec<Ty> = sig
                    .type_params
                    .iter()
                    .map(|p| map.get(p).cloned().unwrap())
                    .collect();
                Ok(Ty::Named(sig.tyname.clone(), type_args))
            }

            Expr::Call(name, args) => {
                // A local binding shadowing a name (e.g. a function-valued
                // parameter `f`) is applied as a function VALUE, not a by-name
                // top-level call. This makes `f(x)` work inside a HOF.
                if let Some(local) = Checker::lookup_var(scope, name) {
                    return self.synth_apply(&local, args, scope, &format!("`{}`", name));
                }
                let (params, ret, type_params) =
                    if let Some((p, r)) = builtin_sig(name) {
                        (p, r, Vec::new())
                    } else if let Some(sig) = self.fns.get(name).cloned() {
                        (sig.params, sig.ret, sig.type_params)
                    } else {
                        return Err(format!("unknown function `{}`", name));
                    };
                if args.len() != params.len() {
                    return Err(format!(
                        "function `{}` expects {} argument(s), got {}",
                        name,
                        params.len(),
                        args.len()
                    ));
                }
                // Instantiate the function's generic parameters with fresh vars.
                let map = self.instantiate_map(&type_params);
                for (i, (arg, pt)) in args.iter().zip(params.iter()).enumerate() {
                    let at = self.synth(arg, scope)?;
                    let expected = Self::apply_map(pt, &map);
                    self.unify(&expected, &at).map_err(|e| {
                        format!("function `{}` argument {}: {}", name, i, e)
                    })?;
                }
                Ok(Self::apply_map(&ret, &map))
            }

            Expr::Unary(op, inner) => {
                let t = self.prune(&self.synth(inner, scope)?);
                match (op, &t) {
                    (UnOp::Neg, Ty::Int) => Ok(Ty::Int),
                    (UnOp::Neg, Ty::Float) => Ok(Ty::Float),
                    (UnOp::Not, Ty::Bool) => Ok(Ty::Bool),
                    _ => Err(format!("cannot apply {:?} to {}", op, show(&self.resolve(&t)))),
                }
            }

            Expr::Binary(op, lhs, rhs) => {
                let lt = self.synth(lhs, scope)?;
                let rt = self.synth(rhs, scope)?;
                self.synth_binary(*op, &lt, &rt)
            }

            Expr::If(cond, then, els) => {
                let ct = self.synth(cond, scope)?;
                self.unify(&Ty::Bool, &ct)
                    .map_err(|_| format!("`if` condition must be Bool, got {}", show(&self.resolve(&ct))))?;
                let tt = self.synth(then, scope)?;
                let et = self.synth(els, scope)?;
                self.unify(&tt, &et).map_err(|_| {
                    format!(
                        "`if` branches have differing types: {} vs {}",
                        show(&self.resolve(&tt)),
                        show(&self.resolve(&et))
                    )
                })?;
                Ok(self.resolve(&tt))
            }

            Expr::Match(scrut, arms) => self.synth_match(scrut, arms, scope),

            Expr::Lambda(params, body, _) => {
                // Parameters are typed from their annotations. An unannotated
                // `\x -> ...` carries a parser-supplied placeholder var (`$lamN`);
                // give it a FRESH unification variable so its type can be solved
                // from the surrounding context (e.g. the function-type parameter
                // it is passed to). Check the body in the extended scope; the
                // lambda's type is the resulting `Ty::Fn`.
                let mut frame = HashMap::new();
                let mut param_tys: Vec<Ty> = Vec::new();
                for (n, t) in params {
                    let pt = match t {
                        Ty::Var(v) if v.starts_with("$lam") => self.fresh(),
                        other => other.clone(),
                    };
                    frame.insert(n.clone(), pt.clone());
                    param_tys.push(pt);
                }
                scope.push(frame);
                let body_ty = self.synth(body, scope);
                scope.pop();
                let body_ty = body_ty?;
                let param_tys: Vec<Ty> =
                    param_tys.iter().map(|t| self.resolve(t)).collect();
                Ok(Ty::Fn(param_tys, Box::new(self.resolve(&body_ty))))
            }

            Expr::Apply(callee, args, _) => {
                let ct = self.synth(callee, scope)?;
                self.synth_apply(&ct, args, scope, "value")
            }

            Expr::Block(stmts, last) => {
                scope.push(HashMap::new());
                let result = (|| {
                    for s in stmts {
                        match s {
                            Stmt::Let(name, ann, value) => {
                                let vt = self.synth(value, scope)?;
                                let bound = if let Some(a) = ann {
                                    self.unify(a, &vt).map_err(|_| {
                                        format!(
                                            "let `{}`: annotated {} but value is {}",
                                            name,
                                            show(&self.resolve(a)),
                                            show(&self.resolve(&vt))
                                        )
                                    })?;
                                    a.clone()
                                } else {
                                    vt
                                };
                                scope.last_mut().unwrap().insert(name.clone(), bound);
                            }
                            Stmt::Expr(e) => {
                                self.synth(e, scope)?;
                            }
                        }
                    }
                    self.synth(last, scope)
                })();
                scope.pop();
                result
            }
        }
    }

    // Type an application of a value of type `callee` to `args`. The callee must
    // unify with a function type of matching arity; each argument is checked
    // against the corresponding parameter type, and the application's type is the
    // function's return type.
    fn synth_apply(
        &self,
        callee: &Ty,
        args: &[Expr],
        scope: &mut Scope,
        what: &str,
    ) -> Result<Ty, String> {
        let pruned = self.prune(callee);
        // If the callee is already a concrete function type, check arity directly
        // for a clearer message; otherwise force it to a function type by
        // unifying with a fresh `Fn`.
        if let Ty::Fn(params, _) = &pruned {
            if params.len() != args.len() {
                return Err(format!(
                    "applying {} expects {} argument(s), got {}",
                    what,
                    params.len(),
                    args.len()
                ));
            }
        }
        let arg_vars: Vec<Ty> = (0..args.len()).map(|_| self.fresh()).collect();
        let ret_var = self.fresh();
        let want = Ty::Fn(arg_vars.clone(), Box::new(ret_var.clone()));
        self.unify(&want, &pruned).map_err(|_| {
            format!(
                "cannot apply {}: it is not a function of {} argument(s) (its type is {})",
                what,
                args.len(),
                show(&self.resolve(&pruned))
            )
        })?;
        for (i, (arg, pv)) in args.iter().zip(arg_vars.iter()).enumerate() {
            let at = self.synth(arg, scope)?;
            self.unify(pv, &at)
                .map_err(|e| format!("applying {} argument {}: {}", what, i, e))?;
        }
        Ok(self.resolve(&ret_var))
    }

    fn synth_match(&self, scrut: &Expr, arms: &[Arm], scope: &mut Scope) -> Result<Ty, String> {
        let s = self.synth(scrut, scope)?;
        let mut result: Option<Ty> = None;
        let mut covered_ctors: Vec<String> = Vec::new();
        let (mut saw_true, mut saw_false, mut saw_wild) = (false, false, false);

        for arm in arms {
            match &arm.pat {
                Pattern::Wild | Pattern::Var(_) => saw_wild = true,
                Pattern::Ctor(name, _) => covered_ctors.push(name.clone()),
                Pattern::Bool(true) => saw_true = true,
                Pattern::Bool(false) => saw_false = true,
                Pattern::Int(_) => {}
            }

            let mut binds = HashMap::new();
            self.check_pattern(&arm.pat, &s, &mut binds)?;
            scope.push(binds);
            let bt = self.synth(&arm.body, scope);
            scope.pop();
            let bt = bt?;

            match &result {
                None => result = Some(bt),
                Some(rt) => {
                    self.unify(rt, &bt).map_err(|_| {
                        format!(
                            "match arms have differing types: {} vs {}",
                            show(&self.resolve(rt)),
                            show(&self.resolve(&bt))
                        )
                    })?;
                }
            }
        }

        // Exhaustiveness. Keys on constructor names, so generics are unaffected.
        if !saw_wild {
            match self.prune(&s) {
                Ty::Named(tn, _) => {
                    if let Some((_, variants)) = self.types.get(&tn) {
                        let missing: Vec<String> = variants
                            .iter()
                            .filter(|v| !covered_ctors.contains(v))
                            .cloned()
                            .collect();
                        if !missing.is_empty() {
                            return Err(format!(
                                "non-exhaustive match on {}: missing case(s) {}",
                                tn,
                                missing.join(", ")
                            ));
                        }
                    }
                }
                Ty::Bool => {
                    if !(saw_true && saw_false) {
                        return Err("non-exhaustive match on Bool: handle both true and false (or add `_`)".into());
                    }
                }
                other => {
                    return Err(format!(
                        "non-exhaustive match on {}: add a wildcard `_` arm",
                        show(&self.resolve(&other))
                    ));
                }
            }
        }

        result
            .map(|t| self.resolve(&t))
            .ok_or_else(|| "match needs at least one arm".to_string())
    }

    fn check_pattern(
        &self,
        pat: &Pattern,
        expected: &Ty,
        binds: &mut HashMap<String, Ty>,
    ) -> Result<(), String> {
        match pat {
            Pattern::Wild => Ok(()),
            Pattern::Var(n) => {
                binds.insert(n.clone(), expected.clone());
                Ok(())
            }
            Pattern::Int(_) => self
                .unify(&Ty::Int, expected)
                .map_err(|_| format!("integer pattern matched against {}", show(&self.resolve(expected)))),
            Pattern::Bool(_) => self
                .unify(&Ty::Bool, expected)
                .map_err(|_| format!("boolean pattern matched against {}", show(&self.resolve(expected)))),
            Pattern::Ctor(name, subs) => {
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown constructor `{}`", name))?;
                // Instantiate the owning type's params, then unify the scrutinee
                // type with `Name[fresh...]` so field patterns get refined types.
                let map = self.instantiate_map(&sig.type_params);
                let type_args: Vec<Ty> = sig
                    .type_params
                    .iter()
                    .map(|p| map.get(p).cloned().unwrap())
                    .collect();
                let owner = Ty::Named(sig.tyname.clone(), type_args);
                self.unify(&owner, expected).map_err(|_| {
                    format!(
                        "constructor pattern `{}` (of type {}) matched against {}",
                        name,
                        sig.tyname,
                        show(&self.resolve(expected))
                    )
                })?;
                if subs.len() != sig.fields.len() {
                    return Err(format!(
                        "constructor pattern `{}` expects {} field(s), got {}",
                        name,
                        sig.fields.len(),
                        subs.len()
                    ));
                }
                for (sp, ft) in subs.iter().zip(sig.fields.iter()) {
                    let fty = Self::apply_map(ft, &map);
                    self.check_pattern(sp, &fty, binds)?;
                }
                Ok(())
            }
        }
    }

    fn synth_binary(&self, op: BinOp, lt: &Ty, rt: &Ty) -> Result<Ty, String> {
        use BinOp::*;
        let mut lt = self.prune(lt);
        let mut rt = self.prune(rt);
        // If exactly one operand is still an unsolved unification variable (e.g.
        // the parameter of an unannotated lambda whose type is fixed by the
        // other operand), unify them so the variable is driven to the concrete
        // operand's type. This makes `\x -> x + n` (with `n: Int`) infer `x: Int`
        // before the surrounding application pins it. We only do this when one
        // side is concrete to avoid prematurely tying two unknowns together.
        let l_open = matches!(&lt, Ty::Var(v) if is_fresh(v));
        let r_open = matches!(&rt, Ty::Var(v) if is_fresh(v));
        if l_open ^ r_open {
            let _ = self.unify(&lt, &rt);
            lt = self.prune(&lt);
            rt = self.prune(&rt);
        }
        let both = |t: &Ty| lt == *t && rt == *t;
        match op {
            And | Or => {
                if both(&Ty::Bool) {
                    Ok(Ty::Bool)
                } else {
                    Err(format!("`{:?}` needs Bool operands, got {} and {}", op, show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
            }
            Eq | Ne => {
                if self.unify(&lt, &rt).is_ok() {
                    Ok(Ty::Bool)
                } else {
                    Err(format!("cannot compare {} and {}", show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
            }
            Lt | Le | Gt | Ge => {
                if both(&Ty::Int) || both(&Ty::Float) {
                    Ok(Ty::Bool)
                } else {
                    Err(format!("`{:?}` needs two Ints or two Floats, got {} and {}", op, show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
            }
            Mod => {
                if both(&Ty::Int) {
                    Ok(Ty::Int)
                } else {
                    Err(format!("`%` needs Int operands, got {} and {}", show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
            }
            Add | Sub | Mul | Div => {
                if both(&Ty::Int) {
                    Ok(Ty::Int)
                } else if both(&Ty::Float) {
                    Ok(Ty::Float)
                } else {
                    Err(format!("`{:?}` needs two Ints or two Floats, got {} and {}", op, show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lexer, parser};

    fn check_src(src: &str) -> Result<(), Vec<String>> {
        let toks = lexer::lex(src).expect("lex");
        let prog = parser::parse(toks).expect("parse");
        check(&prog)
    }

    #[test]
    fn well_typed_program_ok() {
        let src = r#"
            type Shape = | Circle(Float) | Rect(Float, Float)
            fn area(s: Shape) -> Float =
              match s { Circle(r) => 3.14 * r * r, Rect(w, h) => w * h, }
            fn main() -> Int = { print_float(area(Circle(2.0))); 0 }
        "#;
        assert!(check_src(src).is_ok());
    }

    #[test]
    fn return_type_mismatch_caught() {
        let src = "fn f() -> Int = true";
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("return type")));
    }

    #[test]
    fn non_exhaustive_match_caught() {
        let src = r#"
            type Color = | Red | Green | Blue
            fn name(c: Color) -> Int = match c { Red => 0, Green => 1, }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("non-exhaustive") && e.contains("Blue")));
    }

    #[test]
    fn unbound_variable_caught() {
        let src = "fn f() -> Int = x + 1";
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("unbound variable `x`")));
    }

    #[test]
    fn arg_count_mismatch_caught() {
        let src = r#"
            fn g(a: Int, b: Int) -> Int = a + b
            fn main() -> Int = g(1)
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("expects 2 argument")));
    }

    #[test]
    fn constructor_field_type_mismatch_caught() {
        let src = r#"
            type Box = | B(Int)
            fn main() -> Int = { let x = B(true); 0 }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("field") && e.contains("Int")));
    }

    #[test]
    fn type_mismatch_in_arithmetic_caught() {
        let src = "fn f() -> Int = 1 + true";
        let errs = check_src(src).unwrap_err();
        assert!(!errs.is_empty());
    }

    #[test]
    fn generic_param_is_rigid() {
        // A body that constrains its own type parameter to a concrete type must
        // be rejected at definition time (parametricity / soundness). Without
        // rigid type params this type-checked, then crashed at runtime.
        let src = "fn bad[T](x: T) -> Int = { let y = x == 5; x + 1 }";
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("compare")));
    }

    #[test]
    fn generic_identity_still_checks() {
        // The legitimate fully-parametric case must still pass.
        assert!(check_src("fn id[T](x: T) -> T = x").is_ok());
    }

    // ---- generics --------------------------------------------------------

    #[test]
    fn generic_adt_instantiates() {
        // A generic List instantiated at Int should check, and main returns Int.
        let src = r#"
            type List[T] = | Nil | Cons(T, List[T])
            fn main() -> Int = { let xs = Cons(1, Cons(2, Nil)); 0 }
        "#;
        assert!(check_src(src).is_ok());
    }

    #[test]
    fn generic_type_argument_mismatch_caught() {
        // Cons(1, ...) fixes T = Int, so Cons(true, Nil) must be rejected.
        let src = r#"
            type List[T] = | Nil | Cons(T, List[T])
            fn main() -> Int = { let xs = Cons(1, Cons(true, Nil)); 0 }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("expected") && e.contains("found")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn generic_function_checks() {
        let src = r#"
            type Option[T] = | None | Some(T)
            fn is_some[T](o: Option[T]) -> Bool =
              match o { None => false, Some(_) => true, }
            fn main() -> Bool = is_some(Some(5))
        "#;
        assert!(check_src(src).is_ok());
    }

    #[test]
    fn exhaustiveness_fires_on_generic_type() {
        let src = r#"
            type Option[T] = | None | Some(T)
            fn unwrap_or(o: Option[Int]) -> Int = match o { Some(x) => x, }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("non-exhaustive") && e.contains("None")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn generic_return_type_inferred() {
        // A generic function returning Option[T] used at Int.
        let src = r#"
            type Option[T] = | None | Some(T)
            fn wrap[T](x: T) -> Option[T] = Some(x)
            fn main() -> Int = { let o = wrap(7); 0 }
        "#;
        assert!(check_src(src).is_ok());
    }

    // ---- effect system (purity) -----------------------------------------

    #[test]
    fn pure_with_direct_io_is_error() {
        let src = "pure fn f(x: Int) -> Int = { print_int(x); x }";
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("pure") && e.contains("print_int")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn genuinely_pure_function_ok() {
        let src = "pure fn g(x: Int) -> Int = x + 1";
        assert!(check_src(src).is_ok());
    }

    #[test]
    fn pure_calling_pure_function_ok() {
        let src = r#"
            pure fn g(x: Int) -> Int = x + 1
            pure fn h(x: Int) -> Int = g(x)
        "#;
        assert!(check_src(src).is_ok());
    }

    #[test]
    fn pure_recursive_calling_pure_ok() {
        // Recursive `pure` function that also calls another `pure` function.
        let src = r#"
            pure fn add(a: Int, b: Int) -> Int = a + b
            pure fn sum_to(n: Int) -> Int =
              match n { 0 => 0, _ => add(n, sum_to(n - 1)), }
        "#;
        assert!(check_src(src).is_ok(), "got: {:?}", check_src(src));
    }

    #[test]
    fn pure_with_transitive_io_is_error() {
        let src = r#"
            fn f2(x: Int) -> Int = { print_int(x); x }
            pure fn bad(x: Int) -> Int = f2(x)
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("`bad`") && e.contains("pure")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn pure_in_mutual_recursion_to_io_is_error() {
        // ping/pong mutually recurse; pong performs IO, so both are IO.
        let src = r#"
            pure fn ping(n: Int) -> Int =
              match n { 0 => 0, _ => pong(n - 1), }
            fn pong(n: Int) -> Int =
              match n { 0 => { print_int(0); 0 }, _ => ping(n - 1), }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("`ping`") && e.contains("pure")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn pure_mutual_recursion_cycle_without_io_ok() {
        let src = r#"
            pure fn even(n: Int) -> Bool =
              match n { 0 => true, _ => odd(n - 1), }
            pure fn odd(n: Int) -> Bool =
              match n { 0 => false, _ => even(n - 1), }
        "#;
        assert!(check_src(src).is_ok(), "got: {:?}", check_src(src));
    }

    #[test]
    fn io_inference_classifies_io_function() {
        // An un-annotated function that prints is inferred IO (no error), and
        // a `pure` caller of it is rejected, confirming the classification.
        let src = r#"
            fn logger(x: Int) -> Int = { print_int(x); x }
            pure fn caller(x: Int) -> Int = logger(x)
        "#;
        let prog = parser::parse(lexer::lex(src).expect("lex")).expect("parse");
        let io = infer_io(&prog);
        assert!(io.contains("logger"));
        assert!(io.contains("caller"));
        assert!(check(&prog).is_err());
    }

    #[test]
    fn main_may_be_io() {
        let src = "fn main() -> Int = { print_int(5); 0 }";
        assert!(check_src(src).is_ok());
    }

    // ---- first-class functions / closures --------------------------------

    #[test]
    fn lambda_and_application_check() {
        // A lambda value applied to an argument yields its return type.
        let src = "fn main() -> Int = (\\(x: Int) -> x + 1)(41)";
        assert!(check_src(src).is_ok(), "got: {:?}", check_src(src));
    }

    #[test]
    fn hof_with_function_type_param_checks() {
        let src = r#"
            type L = | Nil | Cons(Int, L)
            fn map(f: (Int) -> Int, xs: L) -> L =
              match xs { Nil => Nil, Cons(h, r) => Cons(f(h), map(f, r)), }
            fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }
            fn main() -> Int = {
                let n = 10;
                sum(map(\x -> x + n, Cons(1, Cons(2, Nil))))
            }
        "#;
        assert!(check_src(src).is_ok(), "got: {:?}", check_src(src));
    }

    #[test]
    fn function_passed_by_name_checks() {
        // A top-level function used as a value (not immediately called).
        let src = r#"
            type L = | Nil | Cons(Int, L)
            fn inc(x: Int) -> Int = x + 1
            fn map(f: (Int) -> Int, xs: L) -> L =
              match xs { Nil => Nil, Cons(h, r) => Cons(f(h), map(f, r)), }
            fn main() -> L = map(inc, Cons(1, Nil))
        "#;
        assert!(check_src(src).is_ok(), "got: {:?}", check_src(src));
    }

    #[test]
    fn generic_hof_checks() {
        // A fully generic higher-order map[T,U].
        let src = r#"
            type L[T] = | Nil | Cons(T, L[T])
            fn map[T, U](f: (T) -> U, xs: L[T]) -> L[U] =
              match xs { Nil => Nil, Cons(h, r) => Cons(f(h), map(f, r)), }
            fn main() -> Int = { let xs = map(\(x: Int) -> x == 0, Cons(1, Nil)); 0 }
        "#;
        assert!(check_src(src).is_ok(), "got: {:?}", check_src(src));
    }

    #[test]
    fn applying_a_non_function_is_error() {
        let src = "fn main() -> Int = { let x = 5; x(3) }";
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("not a function")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn lambda_wrong_argument_type_is_error() {
        // Applying an `(Int) -> Int` lambda to a Bool must be rejected.
        let src = "fn main() -> Int = (\\(x: Int) -> x + 1)(true)";
        let errs = check_src(src).unwrap_err();
        assert!(!errs.is_empty(), "expected a type error");
    }

    #[test]
    fn hof_wrong_function_type_is_error() {
        // Passing a `(Bool) -> Int` where `(Int) -> Int` is required.
        let src = r#"
            type L = | Nil | Cons(Int, L)
            fn map(f: (Int) -> Int, xs: L) -> L =
              match xs { Nil => Nil, Cons(h, r) => Cons(f(h), map(f, r)), }
            fn main() -> L = map(\(b: Bool) -> 0, Cons(1, Nil))
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(!errs.is_empty(), "expected a type error");
    }
}
