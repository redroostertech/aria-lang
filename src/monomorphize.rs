//! Monomorphization: turn a well-typed (possibly generic) `Program` into an
//! equivalent type-variable-free `Program` that the wasm backend can compile.
//!
//! Aria supports parametric polymorphism — generic ADTs like
//! `type List[T] = | Nil | Cons(T, List[T])` and generic functions like
//! `fn length[T](xs: List[T]) -> Int`. The interpreter handles these
//! dynamically, but the wasm backend needs CONCRETE layouts (a `List[Int]`
//! cell has an i64 field; a `List[V]` of an ADT has a Ref field). This pass
//! specializes every generic function and ADT for each set of concrete type
//! arguments actually reachable from `main`, producing a program whose every
//! type is a builtin or `Named(name, [])` and whose functions have no type
//! parameters — exactly the subset `wasm::compile` already lowers.
//!
//! ## Algorithm (specialization on demand from `main`)
//!
//! Start from `main` (concrete: `() -> Int`). Maintain a worklist of needed
//! function instantiations `(fn_name, concrete_type_args)` and type
//! instantiations `(type_name, concrete_type_args)`, plus the set of
//! already-emitted specializations.
//!
//! * To specialize a generic function at a call site we know the ACTUAL
//!   argument types (we are walking concrete code), so we unify the callee's
//!   declared parameter types (which mention its `type_params` as `Ty::Var`)
//!   against those actual types. The resulting substitution maps each type
//!   parameter to a concrete type; we substitute it through the body to get a
//!   specialized `FnDecl`, recursively enqueuing any generic calls/ctors it
//!   contains.
//! * To specialize a generic ADT used at concrete args (e.g. `List[Int]`) we
//!   emit a monomorphic `TypeDecl` with a mangled name (`List$Int`) whose
//!   variant fields are the substituted concrete types, and mangle its
//!   constructors per instantiation (`Cons$List$Int`, `Nil$List$Int`).
//!
//! ## Name mangling
//!
//! * type   `List` at `[Int]`        -> `List$Int`
//! * ctor   `Cons` of `List[Int]`    -> `Cons$List$Int`
//! * fn     `length` at type-args `[Int]` -> `length$Int`
//!
//! Type-argument components are mangled recursively so `List[List[Int]]`
//! becomes `List$List_Int` (nested args joined with `_`). A non-generic
//! function/type keeps its original name and is emitted once; its
//! generic-call/ctor sites are still rewritten to the mangled targets.

use std::collections::HashMap;

use crate::ast::*;

/// Monomorphize a well-typed program. On success the result contains no
/// `Ty::Var` and no remaining type parameters. Generic items unreachable from
/// `main` are dropped.
pub fn monomorphize(program: &Program) -> Result<Program, String> {
    Mono::new(program).run()
}

// ---- signatures gathered from the source program ---------------------------

#[derive(Clone)]
struct FnInfo {
    decl: FnDecl,
}

#[derive(Clone)]
struct CtorInfo {
    type_params: Vec<String>,
    fields: Vec<Ty>,
    tyname: String,
}

#[derive(Clone)]
struct TypeInfo {
    decl: TypeDecl,
}

struct Mono<'a> {
    program: &'a Program,
    fns: HashMap<String, FnInfo>,
    ctors: HashMap<String, CtorInfo>,
    types: HashMap<String, TypeInfo>,
    /// emitted function specializations, keyed by mangled name
    out_fns: HashMap<String, FnDecl>,
    /// emitted type specializations, keyed by mangled name
    out_types: HashMap<String, TypeDecl>,
    fn_order: Vec<String>,
    type_order: Vec<String>,
    fn_seen: std::collections::HashSet<String>,
    type_seen: std::collections::HashSet<String>,
    /// Maps a mangled monomorphic type name back to its generic origin
    /// `(original_name, concrete_args)`, so inference can unify a concrete
    /// argument type (already mangled) against a declared generic type.
    demangle: HashMap<String, (String, Vec<Ty>)>,
}

impl<'a> Mono<'a> {
    fn new(program: &'a Program) -> Self {
        let mut fns = HashMap::new();
        let mut ctors = HashMap::new();
        let mut types = HashMap::new();
        for item in &program.items {
            match item {
                Item::Fn(f) => {
                    fns.insert(f.name.clone(), FnInfo { decl: f.clone() });
                }
                Item::Type(t) => {
                    for v in &t.variants {
                        ctors.insert(
                            v.name.clone(),
                            CtorInfo {
                                type_params: t.params.clone(),
                                fields: v.fields.clone(),
                                tyname: t.name.clone(),
                            },
                        );
                    }
                    types.insert(t.name.clone(), TypeInfo { decl: t.clone() });
                }
            }
        }
        Mono {
            program,
            fns,
            ctors,
            types,
            out_fns: HashMap::new(),
            out_types: HashMap::new(),
            fn_order: Vec::new(),
            type_order: Vec::new(),
            fn_seen: std::collections::HashSet::new(),
            type_seen: std::collections::HashSet::new(),
            demangle: HashMap::new(),
        }
    }

    fn run(mut self) -> Result<Program, String> {
        // `main` is the root: concrete, no type params, returns Int.
        if !self.fns.contains_key("main") {
            // Leave it to the backend to report a missing main.
            return Ok(self.program.clone());
        }

        // Fast path: a program with no generics at all passes through verbatim
        // (preserves declaration order and avoids dropping unreferenced items,
        // matching prior behaviour for the existing concrete test corpus).
        let has_generics = self.program.items.iter().any(|it| match it {
            Item::Fn(f) => !f.type_params.is_empty(),
            Item::Type(t) => !t.params.is_empty(),
        });
        if !has_generics {
            return Ok(self.program.clone());
        }

        self.specialize_fn("main", &[])?;

        // Assemble output preserving the source declaration order for the
        // ORIGINAL (non-generic) items, then appending fresh specializations in
        // the order they were first needed. Types must precede functions only in
        // that the backend reads both up front; we keep types first.
        let mut items = Vec::new();
        for name in &self.type_order {
            items.push(Item::Type(self.out_types[name].clone()));
        }
        for name in &self.fn_order {
            items.push(Item::Fn(self.out_fns[name].clone()));
        }
        Ok(Program { items })
    }

    // ---- name mangling ------------------------------------------------------

    /// Render a CONCRETE type as a name component. Must be called only on types
    /// free of `Ty::Var`.
    fn ty_component(t: &Ty) -> String {
        match t {
            Ty::Int => "Int".to_string(),
            Ty::Float => "Float".to_string(),
            Ty::Bool => "Bool".to_string(),
            Ty::Str => "String".to_string(),
            Ty::Unit => "Unit".to_string(),
            Ty::Named(n, args) => {
                if args.is_empty() {
                    n.clone()
                } else {
                    let inner: Vec<String> = args.iter().map(Self::ty_component).collect();
                    format!("{}_{}", n, inner.join("_"))
                }
            }
            Ty::Var(v) => format!("?{}", v), // should not happen post-substitution
        }
    }

    fn mangle(base: &str, args: &[Ty]) -> String {
        if args.is_empty() {
            base.to_string()
        } else {
            let parts: Vec<String> = args.iter().map(Self::ty_component).collect();
            format!("{}${}", base, parts.join("$"))
        }
    }

    /// The mangled type name for `Named(name, args)`. Records the demangling so
    /// later inference can recover `(name, args)` from the mangled string, even
    /// before the type specialization itself is emitted.
    fn mangle_type_name(&mut self, name: &str, args: &[Ty]) -> String {
        let mangled = Self::mangle(name, args);
        if !args.is_empty() {
            self.demangle
                .entry(mangled.clone())
                .or_insert_with(|| (name.to_string(), args.to_vec()));
        }
        mangled
    }

    /// The mangled constructor name for ctor `cname` of owner `owner` at `args`.
    fn mangle_ctor_name(cname: &str, owner_mangled: &str) -> String {
        format!("{}${}", cname, owner_mangled)
    }

    // ---- substitution -------------------------------------------------------

    /// Substitute declared type parameters (as `Ty::Var`) with concrete types,
    /// AND rewrite every generic `Named(n, args)` (args non-empty) to the
    /// monomorphic `Named(mangled, [])`, enqueuing the type specialization.
    fn subst_ty(&mut self, ty: &Ty, map: &HashMap<String, Ty>) -> Result<Ty, String> {
        match ty {
            Ty::Var(n) => match map.get(n) {
                Some(t) => Ok(t.clone()),
                None => Err(format!("monomorphize: unbound type variable `{}`", n)),
            },
            Ty::Named(n, args) if args.is_empty() => Ok(Ty::Named(n.clone(), Vec::new())),
            Ty::Named(n, args) => {
                let cargs: Vec<Ty> = args
                    .iter()
                    .map(|a| self.subst_ty(a, map))
                    .collect::<Result<_, _>>()?;
                let mangled = self.mangle_type_name(n, &cargs);
                self.enqueue_type(n, &cargs)?;
                Ok(Ty::Named(mangled, Vec::new()))
            }
            other => Ok(other.clone()),
        }
    }

    // ---- type specialization ------------------------------------------------

    /// Ensure the monomorphic `TypeDecl` for `name[args]` is emitted. `args` are
    /// concrete (var-free). Idempotent.
    fn enqueue_type(&mut self, name: &str, args: &[Ty]) -> Result<(), String> {
        let info = match self.types.get(name) {
            Some(i) => i.clone(),
            // Builtins / unknown named types are left as-is by callers; nothing
            // to specialize.
            None => return Ok(()),
        };
        let mangled = self.mangle_type_name(name, args);
        // Record the demangling so inference can recover the generic owner from
        // a concrete (mangled) argument type.
        self.demangle
            .entry(mangled.clone())
            .or_insert_with(|| (name.to_string(), args.to_vec()));
        if self.type_seen.contains(&mangled) {
            return Ok(());
        }
        self.type_seen.insert(mangled.clone());

        if info.decl.params.len() != args.len() {
            return Err(format!(
                "monomorphize: type `{}` expects {} type arg(s), got {}",
                name,
                info.decl.params.len(),
                args.len()
            ));
        }
        let map: HashMap<String, Ty> = info
            .decl
            .params
            .iter()
            .cloned()
            .zip(args.iter().cloned())
            .collect();

        // Build specialized variants with mangled ctor names and substituted
        // field types. Recursive references to the same instantiation terminate
        // because `mangled` is already marked seen.
        let mut variants = Vec::new();
        for v in &info.decl.variants {
            let mut fields = Vec::new();
            for ft in &v.fields {
                fields.push(self.subst_ty(ft, &map)?);
            }
            variants.push(Variant {
                name: Self::mangle_ctor_name(&v.name, &mangled),
                fields,
            });
        }
        let decl = TypeDecl {
            name: mangled.clone(),
            params: Vec::new(),
            variants,
        };
        self.out_types.insert(mangled.clone(), decl);
        self.type_order.push(mangled);
        Ok(())
    }

    // ---- function specialization --------------------------------------------

    /// Ensure the monomorphic `FnDecl` for `name` at type-args `targs` is
    /// emitted; returns its mangled name.
    fn specialize_fn(&mut self, name: &str, targs: &[Ty]) -> Result<String, String> {
        let info = self
            .fns
            .get(name)
            .cloned()
            .ok_or_else(|| format!("monomorphize: unknown function `{}`", name))?;
        let mangled = Self::mangle(name, targs);
        if self.fn_seen.contains(&mangled) {
            return Ok(mangled);
        }
        self.fn_seen.insert(mangled.clone());

        if info.decl.type_params.len() != targs.len() {
            return Err(format!(
                "monomorphize: function `{}` expects {} type arg(s), got {}",
                name,
                info.decl.type_params.len(),
                targs.len()
            ));
        }
        let map: HashMap<String, Ty> = info
            .decl
            .type_params
            .iter()
            .cloned()
            .zip(targs.iter().cloned())
            .collect();

        // Build the local variable type environment from the substituted params.
        let mut env: HashMap<String, Ty> = HashMap::new();
        let mut params = Vec::new();
        for p in &info.decl.params {
            let pty = self.subst_ty(&p.ty, &map)?;
            env.insert(p.name.clone(), pty.clone());
            params.push(Param {
                name: p.name.clone(),
                ty: pty,
            });
        }
        let ret = self.subst_ty(&info.decl.ret, &map)?;

        // Rewrite the body: compute concrete types as we go (so nested generic
        // calls/ctors can be specialized) and rewrite ctor/call names. The
        // declared return type provides the expected type, letting nullary
        // generic constructors (e.g. `Nil`) recover their type args from context.
        let (body, _bt) = self.rewrite_expr(&info.decl.body, &mut env, &map, Some(&ret))?;

        let decl = FnDecl {
            name: mangled.clone(),
            type_params: Vec::new(),
            params,
            ret,
            body,
        };
        self.out_fns.insert(mangled.clone(), decl);
        self.fn_order.push(mangled.clone());
        Ok(mangled)
    }

    /// Produce a concrete, mangled expected type for a declared field/param
    /// type, using the partial substitution `sub`. Returns `None` if the type
    /// still mentions an unresolved variable (no useful expectation to push).
    /// Does NOT enqueue type specializations — it is a lookahead only.
    fn subst_ty_partial(&mut self, ty: &Ty, sub: &HashMap<String, Ty>) -> Option<Ty> {
        match ty {
            Ty::Var(n) => match sub.get(n) {
                Some(t) if !contains_var(t) => Some(self.mangle_concrete(t)),
                _ => None,
            },
            Ty::Named(n, args) if args.is_empty() => Some(Ty::Named(n.clone(), Vec::new())),
            Ty::Named(n, args) => {
                let cargs: Vec<Ty> = args
                    .iter()
                    .map(|a| self.subst_ty_partial(a, sub))
                    .collect::<Option<_>>()?;
                Some(Ty::Named(self.mangle_type_name(n, &cargs), Vec::new()))
            }
            other => Some(other.clone()),
        }
    }

    /// Re-mangle an already-concrete (var-free) type so nested generic `Named`s
    /// collapse to their monomorphic names. Used for lookahead expectations.
    fn mangle_concrete(&mut self, ty: &Ty) -> Ty {
        match ty {
            Ty::Named(n, args) if !args.is_empty() => {
                let cargs: Vec<Ty> = args.iter().map(|a| self.mangle_concrete(a)).collect();
                Ty::Named(self.mangle_type_name(n, &cargs), Vec::new())
            }
            other => other.clone(),
        }
    }

    // ---- expression rewriting + type synthesis ------------------------------

    /// Rewrite an expression in a monomorphic context, returning the rewritten
    /// expression and its CONCRETE type. `env` maps locals to concrete types;
    /// `tymap` maps the enclosing function's type params to concrete types (used
    /// only to substitute annotations). `expected` is the concrete type the
    /// surrounding context requires, if known — it lets under-constrained
    /// generic constructors (e.g. nullary `Nil`) and calls recover type args
    /// that their arguments alone cannot pin down.
    fn rewrite_expr(
        &mut self,
        e: &Expr,
        env: &mut HashMap<String, Ty>,
        tymap: &HashMap<String, Ty>,
        expected: Option<&Ty>,
    ) -> Result<(Expr, Ty), String> {
        match e {
            Expr::Int(v) => Ok((Expr::Int(*v), Ty::Int)),
            Expr::Float(v) => Ok((Expr::Float(*v), Ty::Float)),
            Expr::Bool(v) => Ok((Expr::Bool(*v), Ty::Bool)),
            Expr::Str(s) => Ok((Expr::Str(s.clone()), Ty::Str)),
            Expr::Unit => Ok((Expr::Unit, Ty::Unit)),

            Expr::Var(name) => {
                let ty = env
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("monomorphize: unbound variable `{}`", name))?;
                Ok((Expr::Var(name.clone()), ty))
            }

            Expr::Ctor(name, args) => {
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("monomorphize: unknown constructor `{}`", name))?;
                let mut sub: HashMap<String, Ty> = HashMap::new();
                // Seed inference from the expected type if it names this ctor's
                // owner: a nullary or partially-applied generic ctor (e.g. `Nil`)
                // gets its type args from context.
                if let Some(Ty::Named(m, margs)) = expected {
                    if margs.is_empty() {
                        if let Some((orig, cargs)) = self.demangle.get(m).cloned() {
                            if orig == sig.tyname && cargs.len() == sig.type_params.len() {
                                for (p, c) in sig.type_params.iter().zip(cargs.iter()) {
                                    sub.insert(p.clone(), c.clone());
                                }
                            }
                        }
                    } else if m == &sig.tyname && margs.len() == sig.type_params.len() {
                        for (p, c) in sig.type_params.iter().zip(margs.iter()) {
                            sub.insert(p.clone(), c.clone());
                        }
                    }
                }
                // Rewrite arguments, pushing each field's expected (substituted)
                // type down so nested nullary ctors resolve, and learning their
                // concrete types to refine inference.
                let mut rargs = Vec::new();
                for (a, ft) in args.iter().zip(sig.fields.iter()) {
                    let field_expect = self.subst_ty_partial(ft, &sub);
                    let (ra, at) = self.rewrite_expr(a, env, tymap, field_expect.as_ref())?;
                    self.unify_decl(ft, &at, &mut sub)?;
                    rargs.push(ra);
                }
                let type_args = self.solve_params(&sig.type_params, &sub, name)?;
                let owner_mangled = self.mangle_type_name(&sig.tyname, &type_args);
                self.enqueue_type(&sig.tyname, &type_args)?;
                let cname = if type_args.is_empty() {
                    name.clone()
                } else {
                    Self::mangle_ctor_name(name, &owner_mangled)
                };
                let result_ty = Ty::Named(owner_mangled, Vec::new());
                Ok((Expr::Ctor(cname, rargs), result_ty))
            }

            Expr::Call(name, args) => {
                // Builtin?  (Identified by NOT being a user function.) Builtins
                // are non-generic, so just rewrite args with no expectations.
                if !self.fns.contains_key(name) {
                    let mut rargs = Vec::new();
                    let mut arg_tys = Vec::new();
                    for a in args {
                        let (ra, at) = self.rewrite_expr(a, env, tymap, None)?;
                        rargs.push(ra);
                        arg_tys.push(at);
                    }
                    let rt = builtin_ret(name, &arg_tys);
                    return Ok((Expr::Call(name.clone(), rargs), rt));
                }
                let info = self.fns.get(name).cloned().unwrap();
                let mut sub: HashMap<String, Ty> = HashMap::new();
                // Seed from the expected return type if it pins type params.
                if let Some(exp) = expected {
                    let _ = self.unify_decl(&info.decl.ret, exp, &mut sub);
                }
                // Infer callee type args by unifying declared params against the
                // concrete actuals, pushing each param's (substituted) type down.
                let mut rargs = Vec::new();
                for (p, a) in info.decl.params.iter().zip(args.iter()) {
                    let pexp = self.subst_ty_partial(&p.ty, &sub);
                    let (ra, at) = self.rewrite_expr(a, env, tymap, pexp.as_ref())?;
                    self.unify_decl(&p.ty, &at, &mut sub)?;
                    rargs.push(ra);
                }
                let type_args = self.solve_params(&info.decl.type_params, &sub, name)?;
                let target = self.specialize_fn(name, &type_args)?;
                // The concrete return type: substitute inferred params, then
                // mangle any generic Named within.
                let callee_map: HashMap<String, Ty> = info
                    .decl
                    .type_params
                    .iter()
                    .cloned()
                    .zip(type_args.iter().cloned())
                    .collect();
                let rt = self.subst_ty(&info.decl.ret, &callee_map)?;
                Ok((Expr::Call(target, rargs), rt))
            }

            Expr::Unary(op, inner) => {
                let (ri, it) = self.rewrite_expr(inner, env, tymap, None)?;
                Ok((Expr::Unary(*op, Box::new(ri)), it))
            }

            Expr::Binary(op, l, r) => {
                let (rl, lt) = self.rewrite_expr(l, env, tymap, None)?;
                let (rr, _rt) = self.rewrite_expr(r, env, tymap, None)?;
                let ty = binary_ret(*op, &lt);
                Ok((Expr::Binary(*op, Box::new(rl), Box::new(rr)), ty))
            }

            Expr::If(c, t, e2) => {
                let (rc, _) = self.rewrite_expr(c, env, tymap, None)?;
                let (rt, tt) = self.rewrite_expr(t, env, tymap, expected)?;
                // The else branch shares the then branch's concrete type, which
                // is a stronger expectation than the caller's (it may have been
                // None).
                let (re, _) = self.rewrite_expr(e2, env, tymap, Some(&tt))?;
                Ok((Expr::If(Box::new(rc), Box::new(rt), Box::new(re)), tt))
            }

            Expr::Match(scrut, arms) => {
                let (rscrut, scrut_ty) = self.rewrite_expr(scrut, env, tymap, None)?;
                let mut rarms = Vec::new();
                let mut result_ty: Option<Ty> = None;
                for arm in arms {
                    let mut binds = env.clone();
                    let rpat = self.rewrite_pattern(&arm.pat, &scrut_ty, &mut binds)?;
                    // Prefer an already-determined arm type as the expectation
                    // (so later arms' nullary ctors resolve), else the caller's.
                    let arm_expect = result_ty.as_ref().or(expected);
                    let (rbody, bt) =
                        self.rewrite_expr(&arm.body, &mut binds, tymap, arm_expect)?;
                    result_ty.get_or_insert(bt);
                    rarms.push(Arm {
                        pat: rpat,
                        body: rbody,
                    });
                }
                let rt = result_ty.ok_or("monomorphize: empty match")?;
                Ok((Expr::Match(Box::new(rscrut), rarms), rt))
            }

            Expr::Block(stmts, last) => {
                let mut scope = env.clone();
                let mut rstmts = Vec::new();
                for s in stmts {
                    match s {
                        Stmt::Let(name, ann, value) => {
                            // An annotation gives a concrete expected type.
                            let ann_ty = match ann {
                                Some(a) => Some(self.subst_ty(a, tymap)?),
                                None => None,
                            };
                            let (rv, vt) =
                                self.rewrite_expr(value, &mut scope, tymap, ann_ty.as_ref())?;
                            let bound = ann_ty.clone().unwrap_or(vt);
                            scope.insert(name.clone(), bound.clone());
                            let rann = ann_ty.map(|_| bound);
                            rstmts.push(Stmt::Let(name.clone(), rann, rv));
                        }
                        Stmt::Expr(ex) => {
                            let (re, _) = self.rewrite_expr(ex, &mut scope, tymap, None)?;
                            rstmts.push(Stmt::Expr(re));
                        }
                    }
                }
                let (rlast, lt) = self.rewrite_expr(last, &mut scope, tymap, expected)?;
                Ok((Expr::Block(rstmts, Box::new(rlast)), lt))
            }
        }
    }

    /// Rewrite a pattern against a concrete scrutinee type, binding pattern
    /// variables into `binds` with their concrete types and mangling ctor names.
    fn rewrite_pattern(
        &mut self,
        pat: &Pattern,
        scrut_ty: &Ty,
        binds: &mut HashMap<String, Ty>,
    ) -> Result<Pattern, String> {
        match pat {
            Pattern::Wild => Ok(Pattern::Wild),
            Pattern::Var(n) => {
                binds.insert(n.clone(), scrut_ty.clone());
                Ok(Pattern::Var(n.clone()))
            }
            Pattern::Int(i) => Ok(Pattern::Int(*i)),
            Pattern::Bool(b) => Ok(Pattern::Bool(*b)),
            Pattern::Ctor(name, subs) => {
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("monomorphize: unknown constructor `{}`", name))?;
                // The scrutinee type is the (already mangled) owner. Recover the
                // concrete field types by re-deriving the owner's type args from
                // the scrutinee's mangled name via the type table. Simpler: infer
                // the type args by matching the owner's declared self-type. We
                // instead unify the constructor's OWN result type pattern.
                //
                // The scrutinee mangled name encodes the instantiation; we look
                // up the emitted TypeDecl to get the already-substituted variant
                // field types directly.
                let owner_mangled = match scrut_ty {
                    Ty::Named(n, _) => n.clone(),
                    _ => {
                        return Err(format!(
                            "monomorphize: constructor pattern `{}` on non-ADT {}",
                            name,
                            crate::typeck::show(scrut_ty)
                        ))
                    }
                };
                let cname = if self.types.get(&sig.tyname).map(|i| i.decl.params.is_empty())
                    == Some(true)
                {
                    name.clone()
                } else {
                    Self::mangle_ctor_name(name, &owner_mangled)
                };
                // Look up the emitted (monomorphic) variant to get field types.
                let field_tys: Vec<Ty> = self
                    .out_types
                    .get(&owner_mangled)
                    .and_then(|td| td.variants.iter().find(|v| v.name == cname))
                    .map(|v| v.fields.clone())
                    .or_else(|| {
                        // Non-generic owner: fields are already concrete.
                        if sig.type_params.is_empty() {
                            Some(sig.fields.clone())
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| {
                        format!(
                            "monomorphize: cannot resolve fields for `{}` of {}",
                            name, owner_mangled
                        )
                    })?;
                if subs.len() != field_tys.len() {
                    return Err(format!(
                        "monomorphize: constructor pattern `{}` arity mismatch",
                        name
                    ));
                }
                let mut rsubs = Vec::new();
                for (sp, ft) in subs.iter().zip(field_tys.iter()) {
                    rsubs.push(self.rewrite_pattern(sp, ft, binds)?);
                }
                Ok(Pattern::Ctor(cname, rsubs))
            }
        }
    }

    /// Unify a declared type `decl` (possibly mentioning the enclosing item's
    /// type parameters as `Ty::Var`) against a CONCRETE argument type `concrete`
    /// (which may already be a mangled monomorphic `Named`), recording param
    /// bindings in `sub`. When the declared side is a generic `Named(g, [vars])`
    /// and the concrete side is a mangled `Named(m, [])`, we recover `m`'s
    /// generic origin `(g, [args])` from `self.demangle` and unify args
    /// positionally.
    fn unify_decl(
        &self,
        decl: &Ty,
        concrete: &Ty,
        sub: &mut HashMap<String, Ty>,
    ) -> Result<(), String> {
        let decl = resolve(decl, sub);
        match (&decl, concrete) {
            (Ty::Var(v), _) => {
                if let Some(existing) = sub.get(v).cloned() {
                    self.unify_decl(&existing, concrete, sub)
                } else {
                    sub.insert(v.clone(), concrete.clone());
                    Ok(())
                }
            }
            (Ty::Int, Ty::Int)
            | (Ty::Float, Ty::Float)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Str, Ty::Str)
            | (Ty::Unit, Ty::Unit) => Ok(()),
            (Ty::Named(n1, a1), Ty::Named(n2, a2)) if n1 == n2 && a1.len() == a2.len() => {
                for (x, y) in a1.iter().zip(a2.iter()) {
                    self.unify_decl(x, y, sub)?;
                }
                Ok(())
            }
            // Declared generic `Named(g, [..])`, concrete mangled `Named(m, [])`.
            (Ty::Named(g, gargs), Ty::Named(m, margs))
                if !gargs.is_empty() && margs.is_empty() =>
            {
                if let Some((orig, cargs)) = self.demangle.get(m) {
                    if orig == g && cargs.len() == gargs.len() {
                        for (gv, cv) in gargs.iter().zip(cargs.iter()) {
                            self.unify_decl(gv, cv, sub)?;
                        }
                        return Ok(());
                    }
                }
                Err(format!(
                    "monomorphize: cannot unify declared `{}` with concrete `{}`",
                    crate::typeck::show(&decl),
                    crate::typeck::show(concrete)
                ))
            }
            _ => Err(format!(
                "monomorphize: cannot unify declared `{}` with concrete `{}`",
                crate::typeck::show(&decl),
                crate::typeck::show(concrete)
            )),
        }
    }

    /// Resolve declared params from a substitution, erroring if any is
    /// unconstrained (a truly-unused type parameter can't be monomorphized).
    fn solve_params(
        &self,
        params: &[String],
        sub: &HashMap<String, Ty>,
        site: &str,
    ) -> Result<Vec<Ty>, String> {
        let mut out = Vec::new();
        for p in params {
            match sub.get(p) {
                Some(t) if !contains_var(t) => out.push(resolve(t, sub)),
                _ => {
                    return Err(format!(
                        "monomorphize: could not infer type parameter `{}` at `{}` (phantom/unused type parameters are unsupported)",
                        p, site
                    ))
                }
            }
        }
        Ok(out)
    }
}

fn resolve(ty: &Ty, sub: &HashMap<String, Ty>) -> Ty {
    match ty {
        Ty::Var(v) => match sub.get(v) {
            Some(t) => resolve(t, sub),
            None => ty.clone(),
        },
        Ty::Named(n, args) => Ty::Named(n.clone(), args.iter().map(|a| resolve(a, sub)).collect()),
        other => other.clone(),
    }
}

fn contains_var(ty: &Ty) -> bool {
    match ty {
        Ty::Var(_) => true,
        Ty::Named(_, args) => args.iter().any(contains_var),
        _ => false,
    }
}

// ---- minimal type synthesis for builtins / operators ------------------------

/// Result type of a binary operator given the (concrete) left operand type.
fn binary_ret(op: BinOp, lt: &Ty) -> Ty {
    use BinOp::*;
    match op {
        And | Or | Eq | Ne | Lt | Le | Gt | Ge => Ty::Bool,
        Add | Sub | Mul | Div | Mod => lt.clone(),
    }
}

/// Result type of a builtin call. Only the builtins compiled by the wasm
/// backend matter here; others fall back to Unit and are rejected downstream.
fn builtin_ret(name: &str, args: &[Ty]) -> Ty {
    match name {
        "int_to_str" | "float_to_str" | "concat" => Ty::Str,
        "print_str" | "print_float" | "print_int" => Ty::Unit,
        // Fall back to the shared builtin signature table if present.
        _ => match crate::builtins::lookup(name) {
            Some((_, ret)) => ret,
            None => {
                let _ = args;
                Ty::Unit
            }
        },
    }
}
