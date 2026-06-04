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
    // Back-annotate the concrete types of any unannotated lambda parameters
    // (e.g. `let f = \x -> ..`) so specialization and the backends see a type
    // even when only the surrounding context fixes it.
    let mut program = program.clone();
    crate::typeck::annotate_lambda_params(&mut program);
    Mono::new(&program).run()
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
    /// `Some(names)` iff a record constructor — used to desugar `Expr::Record`/
    /// `Field`/`Update` and `Pattern::Record` to positional form.
    field_names: Option<Vec<String>>,
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
    /// Reconstructed trait/impl structure: a trait-method (dispatcher) call at a
    /// concrete receiver type resolves directly to the mangled impl function
    /// (static dispatch), so dispatchers are never emitted to the backends.
    traits: crate::traits::TraitIndex,
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
                                field_names: v.field_names.clone(),
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
            traits: crate::traits::index(program),
        }
    }

    fn run(mut self) -> Result<Program, String> {
        // `main` is the root: concrete, no type params, returns Int.
        if !self.fns.contains_key("main") {
            // Leave it to the backend to report a missing main.
            return Ok(self.program.clone());
        }

        // Fast path: a program with no generics needs no specialization, but its
        // function bodies are still rewritten in place so lambdas get their
        // concrete `ClosureSig` and applied local function values become typed
        // `Apply`s. With no generics, names/types pass through unchanged, so all
        // items and their declaration order are preserved (and no unreferenced
        // item is dropped).
        let has_generics = self.program.items.iter().any(|it| match it {
            Item::Fn(f) => !f.type_params.is_empty(),
            Item::Type(t) => !t.params.is_empty(),
        });
        if !has_generics {
            let empty: HashMap<String, Ty> = HashMap::new();
            let mut items = Vec::new();
            for it in &self.program.items {
                match it {
                    Item::Type(t) => items.push(Item::Type(t.clone())),
                    Item::Fn(f) => {
                        let mut env: HashMap<String, Ty> = f
                            .params
                            .iter()
                            .map(|p| (p.name.clone(), p.ty.clone()))
                            .collect();
                        let (body, _) =
                            self.rewrite_expr(&f.body, &mut env, &empty, Some(&f.ret))?;
                        items.push(Item::Fn(FnDecl {
                            name: f.name.clone(),
                            pure: f.pure,
                            type_params: Vec::new(),
                            bounds: Vec::new(),
                            params: f.params.clone(),
                            ret: f.ret.clone(),
                            body,
                        }));
                    }
                }
            }
            return Ok(Program { items });
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
            // Function types are outside the monomorphizing/compiled subset; they
            // are rejected before reaching codegen, but give a stable name here.
            Ty::Fn(params, ret) => {
                let inner: Vec<String> = params.iter().map(Self::ty_component).collect();
                format!("Fn_{}__{}", inner.join("_"), Self::ty_component(ret))
            }
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
            // `Array`/`Map`/`Set` are builtin generics, not user ADTs: keep them
            // as `Array[E]`/`Map[K,V]`/`Set[T]` (the backends recognize them and
            // the element-kind call suffix carries the element types). Do NOT
            // mangle the name or enqueue it as a type to specialize.
            Ty::Named(n, args) if n == "Array" || n == "Map" || n == "Set" => {
                let cargs: Vec<Ty> =
                    args.iter().map(|a| self.subst_ty(a, map)).collect::<Result<_, _>>()?;
                Ok(Ty::Named(n.clone(), cargs))
            }
            Ty::Named(n, args) => {
                let cargs: Vec<Ty> = args
                    .iter()
                    .map(|a| self.subst_ty(a, map))
                    .collect::<Result<_, _>>()?;
                let mangled = self.mangle_type_name(n, &cargs);
                self.enqueue_type(n, &cargs)?;
                Ok(Ty::Named(mangled, Vec::new()))
            }
            Ty::Fn(ps, r) => {
                let ps2 = ps.iter().map(|p| self.subst_ty(p, map)).collect::<Result<_, _>>()?;
                let r2 = self.subst_ty(r, map)?;
                Ok(Ty::Fn(ps2, Box::new(r2)))
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
                // Match the constructor names emitted by the `Ctor` arm and
                // `rewrite_pattern`: non-generic types keep their original ctor
                // names; only generic instantiations are mangled. (Previously this
                // always mangled, breaking any non-generic ADT — records included
                // — referenced inside a program that also uses generics.)
                name: if args.is_empty() {
                    v.name.clone()
                } else {
                    Self::mangle_ctor_name(&v.name, &mangled)
                },
                fields,
                field_names: v.field_names.clone(),
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
            pure: info.decl.pure,
            type_params: Vec::new(),
            bounds: Vec::new(),
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
            // `Array`/`Map`/`Set` are builtin generics: keep them as
            // `Array[E]`/`Map[K,V]`/`Set[T]`, never ADT-mangle (see `subst_ty`).
            Ty::Named(n, args) if n == "Array" || n == "Map" || n == "Set" => {
                let cargs: Vec<Ty> = args
                    .iter()
                    .map(|a| self.subst_ty_partial(a, sub))
                    .collect::<Option<_>>()?;
                Some(Ty::Named(n.clone(), cargs))
            }
            Ty::Named(n, args) => {
                let cargs: Vec<Ty> = args
                    .iter()
                    .map(|a| self.subst_ty_partial(a, sub))
                    .collect::<Option<_>>()?;
                Some(Ty::Named(self.mangle_type_name(n, &cargs), Vec::new()))
            }
            Ty::Fn(ps, r) => {
                let ps2: Vec<Ty> =
                    ps.iter().map(|p| self.subst_ty_partial(p, sub)).collect::<Option<_>>()?;
                let r2 = self.subst_ty_partial(r, sub)?;
                Some(Ty::Fn(ps2, Box::new(r2)))
            }
            other => Some(other.clone()),
        }
    }

    /// Re-mangle an already-concrete (var-free) type so nested generic `Named`s
    /// collapse to their monomorphic names. Used for lookahead expectations.
    fn mangle_concrete(&mut self, ty: &Ty) -> Ty {
        match ty {
            // `Array` is a builtin generic — never ADT-mangle it (keep
            // `Array[elem]`); the element-kind call suffix carries the element
            // type. Mangling it to `Array$Int` would hide the element from the
            // array-builtin element inference (e.g. an empty `[]` in a tuple).
            Ty::Named(n, args) if n == "Array" || n == "Map" || n == "Set" => {
                // Builtin generic collections — never ADT-mangle them (keep
                // `Array[E]`/`Map[K,V]`/`Set[T]`); the element-kind call suffix
                // carries the concrete element type to the backends. Mangling
                // would hide the element type from element inference.
                let cargs: Vec<Ty> = args.iter().map(|a| self.mangle_concrete(a)).collect();
                Ty::Named(n.clone(), cargs)
            }
            Ty::Named(n, args) if !args.is_empty() => {
                let cargs: Vec<Ty> = args.iter().map(|a| self.mangle_concrete(a)).collect();
                Ty::Named(self.mangle_type_name(n, &cargs), Vec::new())
            }
            Ty::Fn(ps, r) => Ty::Fn(
                ps.iter().map(|a| self.mangle_concrete(a)).collect(),
                Box::new(self.mangle_concrete(r)),
            ),
            other => other.clone(),
        }
    }

    /// Best-effort synthesis of an expression's CONCRETE (mangled) type WITHOUT
    /// rewriting it or enqueuing specializations. Returns `None` whenever the
    /// type cannot be determined cheaply and locally (e.g. an under-determined
    /// generic constructor, or a construct whose result type needs full
    /// rewriting). Used only to seed whole-call type-parameter inference, so a
    /// `None` here is never fatal — it just means this argument contributes
    /// nothing to the seed and is resolved in the rewriting pass instead.
    fn synth_ty(&mut self, e: &Expr, env: &HashMap<String, Ty>) -> Option<Ty> {
        match e {
            Expr::Int(_) => Some(Ty::Int),
            Expr::Float(_) => Some(Ty::Float),
            Expr::Bool(_) => Some(Ty::Bool),
            Expr::Str(_) => Some(Ty::Str),
            Expr::Unit => Some(Ty::Unit),
            Expr::Var(n) => env.get(n).cloned(),
            Expr::Unary(_, inner) => self.synth_ty(inner, env),
            Expr::Binary(op, l, _) => {
                let lt = self.synth_ty(l, env)?;
                Some(binary_ret(*op, &lt))
            }
            // Records are interpreter-only so far; the compiled pipeline does not
            // type or lower them (cleanly rejected in `rewrite_expr`).
            Expr::Record(..) | Expr::Field(..) | Expr::Update(..) => None,
            Expr::Ctor(name, args) => {
                // Only synthesizable if every owning type parameter is pinned by
                // the constructor's own field types (no expected-type context
                // available here). Otherwise return None.
                let sig = self.ctors.get(name).cloned()?;
                let mut sub: HashMap<String, Ty> = HashMap::new();
                for (a, ft) in args.iter().zip(sig.fields.iter()) {
                    if let Some(at) = self.synth_ty(a, env) {
                        let _ = self.unify_decl(ft, &at, &mut sub);
                    }
                }
                let mut targs = Vec::new();
                for p in &sig.type_params {
                    match sub.get(p) {
                        Some(t) if !contains_var(t) => targs.push(resolve(t, &sub)),
                        _ => return None,
                    }
                }
                Some(Ty::Named(self.mangle_type_name(&sig.tyname, &targs), Vec::new()))
            }
            Expr::Call(name, args) => {
                // `array_lit` (variadic array-literal desugaring): the element
                // type is the first argument's synthesized type.
                if name == "array_lit" {
                    let elem = match args.first() {
                        Some(a) => self.synth_ty(a, env)?,
                        None => return None,
                    };
                    return Some(Ty::Named("Array".to_string(), vec![elem]));
                }
                if !self.fns.contains_key(name) {
                    let mut arg_tys = Vec::new();
                    for a in args {
                        arg_tys.push(self.synth_ty(a, env)?);
                    }
                    let mut rt = builtin_ret(name, &arg_tys);
                    // Generic builtin: substitute its type vars from the concrete
                    // argument types so the seed type is var-free.
                    if contains_var(&rt) {
                        if let Some((sig_params, _)) = crate::builtins::lookup(name) {
                            let mut sub: HashMap<String, Ty> = HashMap::new();
                            for (pt, at) in sig_params.iter().zip(arg_tys.iter()) {
                                let _ = self.unify_decl(pt, at, &mut sub);
                            }
                            rt = resolve(&rt, &sub);
                        }
                    }
                    return Some(rt);
                }
                let info = self.fns.get(name).cloned()?;
                let mut sub: HashMap<String, Ty> = HashMap::new();
                for (p, a) in info.decl.params.iter().zip(args.iter()) {
                    if let Some(at) = self.synth_ty(a, env) {
                        let _ = self.unify_decl(&p.ty, &at, &mut sub);
                    }
                }
                let mut cmap: HashMap<String, Ty> = HashMap::new();
                for p in &info.decl.type_params {
                    match sub.get(p) {
                        Some(t) if !contains_var(t) => {
                            cmap.insert(p.clone(), resolve(t, &sub));
                        }
                        _ => return None,
                    }
                }
                Some(self.subst_ty_partial(&info.decl.ret, &cmap).unwrap_or_else(|| {
                    self.mangle_concrete(&resolve(&info.decl.ret, &cmap))
                }))
            }
            // If / Match / Block need full rewriting to type; skip for seeding.
            Expr::If(_, _, _) | Expr::Match(_, _) | Expr::Block(_, _) => None,
            // A lambda: best-effort `Fn` type from annotated params + synthesized
            // body. Unannotated params block synthesis (handled by full rewriting).
            Expr::Lambda(params, body, _) => {
                let mut ptys = Vec::new();
                for (_, ann) in params {
                    if contains_var(ann) {
                        return None;
                    }
                    ptys.push(ann.clone());
                }
                let mut inner = env.clone();
                for (n, ann) in params {
                    inner.insert(n.clone(), ann.clone());
                }
                let bt = self.synth_ty(body, &inner)?;
                Some(Ty::Fn(ptys, Box::new(bt)))
            }
            // An application: the callee's `Fn` return type, if synthesizable.
            Expr::Apply(callee, _, _) => match self.synth_ty(callee, env)? {
                Ty::Fn(_, ret) => Some(*ret),
                _ => None,
            },
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
    /// Given a concrete (possibly mangled) record type, return its constructor
    /// name (mangled to match the `Ctor` arm / `rewrite_pattern`), its declared
    /// field names, and the concrete field types. Used to desugar field access
    /// and functional update.
    fn record_shape(&mut self, ty: &Ty) -> Result<(String, Vec<String>, Vec<Ty>), String> {
        let tyname = match ty {
            Ty::Named(n, _) => n.clone(),
            _ => return Err("monomorphize: field access on a non-record value".to_string()),
        };
        // A generic record's type name is mangled (`Box$Box$Int`); recover its
        // origin + concrete args. Non-generic records pass through unmangled.
        let (orig, cargs) = self
            .demangle
            .get(&tyname)
            .cloned()
            .unwrap_or((tyname.clone(), Vec::new()));
        let sig = self
            .ctors
            .get(&orig)
            .cloned()
            .ok_or_else(|| format!("monomorphize: `{}` is not a record type", orig))?;
        let fnames = sig
            .field_names
            .clone()
            .ok_or_else(|| format!("monomorphize: `{}` is not a record", orig))?;
        let map: HashMap<String, Ty> =
            sig.type_params.iter().cloned().zip(cargs.iter().cloned()).collect();
        let ftys: Vec<Ty> =
            sig.fields.iter().map(|f| self.subst_ty(f, &map)).collect::<Result<_, _>>()?;
        let ctor = if cargs.is_empty() {
            orig.clone()
        } else {
            Self::mangle_ctor_name(&orig, &tyname)
        };
        Ok((ctor, fnames, ftys))
    }

    /// Rewrite an array builtin call. Threads the concrete element type INTO the
    /// array argument (so a nested `array_new()` resolves) and suffixes the
    /// emitted name with the element-kind tag the backends dispatch on. Returns
    /// `None` if `name` is not one of `array_new/get/set/push/len`.
    fn rewrite_array_op(
        &mut self,
        name: &str,
        args: &[Expr],
        env: &mut HashMap<String, Ty>,
        tymap: &HashMap<String, Ty>,
        expected: Option<&Ty>,
    ) -> Result<Option<(Expr, Ty)>, String> {
        let arr = |elem: Ty| Ty::Named("Array".to_string(), vec![elem]);
        match name {
            "array_new" => {
                // The element type comes from context (an `Array[E]` expected
                // type — a callee param, a `let` annotation, or a sibling arg).
                let elem = expected.and_then(array_elem_of).unwrap_or(Ty::Unit);
                let tag = array_elem_tag(&elem);
                Ok(Some((Expr::Call(format!("array_new${}", tag), Vec::new()), arr(elem))))
            }
            "array_push" => {
                // args = [array, value]: the value fixes `E`; push it down into
                // the array argument so a nested `array_new()` resolves to `E`.
                let (rval, elem) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let (rarr, _) = self.rewrite_expr(&args[0], env, tymap, Some(&arr(elem.clone())))?;
                let tag = array_elem_tag(&elem);
                Ok(Some((Expr::Call(format!("array_push${}", tag), vec![rarr, rval]), arr(elem))))
            }
            "array_set" => {
                // args = [array, index, value].
                let (rval, elem) = self.rewrite_expr(&args[2], env, tymap, None)?;
                let (rarr, _) = self.rewrite_expr(&args[0], env, tymap, Some(&arr(elem.clone())))?;
                let (ridx, _) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let tag = array_elem_tag(&elem);
                Ok(Some((
                    Expr::Call(format!("array_set${}", tag), vec![rarr, ridx, rval]),
                    arr(elem),
                )))
            }
            "array_get" => {
                // args = [array, index]; result is the element. An `expected`
                // element type means the array is `Array[expected]`.
                let arr_exp = expected.map(|e| arr(e.clone()));
                let (rarr, arr_ty) = self.rewrite_expr(&args[0], env, tymap, arr_exp.as_ref())?;
                let elem =
                    array_elem_of(&arr_ty).or_else(|| expected.cloned()).unwrap_or(Ty::Unit);
                let (ridx, _) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let tag = array_elem_tag(&elem);
                Ok(Some((Expr::Call(format!("array_get${}", tag), vec![rarr, ridx]), elem)))
            }
            "array_len" => {
                let (rarr, arr_ty) = self.rewrite_expr(&args[0], env, tymap, None)?;
                let elem = array_elem_of(&arr_ty).unwrap_or(Ty::Unit);
                let tag = array_elem_tag(&elem);
                Ok(Some((Expr::Call(format!("array_len${}", tag), vec![rarr]), Ty::Int)))
            }
            _ => Ok(None),
        }
    }

    /// Rewrite a Map/Set builtin call. Threads the concrete key/value (or
    /// element) types INTO the map/set argument (so a nested `map_new()`
    /// resolves) and suffixes the emitted name with `$<keytag>_<valtag>` (maps)
    /// or `$<elemtag>` (sets) — the element-kind contract the native backend
    /// dispatches on. Returns `None` if `name` is not a map/set builtin.
    fn rewrite_map_set_op(
        &mut self,
        name: &str,
        args: &[Expr],
        env: &mut HashMap<String, Ty>,
        tymap: &HashMap<String, Ty>,
        expected: Option<&Ty>,
    ) -> Result<Option<(Expr, Ty)>, String> {
        let mapty = |k: Ty, v: Ty| Ty::Named("Map".to_string(), vec![k, v]);
        let setty = |t: Ty| Ty::Named("Set".to_string(), vec![t]);
        match name {
            // ---- Map ----
            "map_new" => {
                let (k, v) = expected
                    .and_then(map_kv_of)
                    .unwrap_or((Ty::Unit, Ty::Unit));
                if !native_map_value_ok(&v) {
                    return Err(unsupported_map_value(&v));
                }
                let suffix = map_suffix(&k, &v);
                Ok(Some((
                    Expr::Call(format!("map_new${}", suffix), Vec::new()),
                    mapty(k, v),
                )))
            }
            "map_insert" => {
                // args = [map, key, value]: key & value fix K, V; push them into
                // the map argument so a nested `map_new()` resolves.
                let (rkey, k) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let (rval, v) = self.rewrite_expr(&args[2], env, tymap, None)?;
                if !native_map_value_ok(&v) {
                    return Err(unsupported_map_value(&v));
                }
                let (rmap, _) =
                    self.rewrite_expr(&args[0], env, tymap, Some(&mapty(k.clone(), v.clone())))?;
                let suffix = map_suffix(&k, &v);
                Ok(Some((
                    Expr::Call(format!("map_insert${}", suffix), vec![rmap, rkey, rval]),
                    mapty(k, v),
                )))
            }
            "map_get_or" => {
                // args = [map, key, default]; result is V. The default fixes V.
                let (rkey, k) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let (rdef, v) = self.rewrite_expr(&args[2], env, tymap, expected)?;
                if !native_map_value_ok(&v) {
                    return Err(unsupported_map_value(&v));
                }
                let (rmap, _) =
                    self.rewrite_expr(&args[0], env, tymap, Some(&mapty(k.clone(), v.clone())))?;
                let suffix = map_suffix(&k, &v);
                Ok(Some((
                    Expr::Call(format!("map_get_or${}", suffix), vec![rmap, rkey, rdef]),
                    v,
                )))
            }
            "map_has" => {
                let (rkey, k) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let (rmap, map_ty) = self.rewrite_expr(&args[0], env, tymap, None)?;
                let v = map_kv_of(&map_ty).map(|(_, v)| v).unwrap_or(Ty::Unit);
                let suffix = map_suffix(&k, &v);
                Ok(Some((
                    Expr::Call(format!("map_has${}", suffix), vec![rmap, rkey]),
                    Ty::Bool,
                )))
            }
            "map_len" => {
                let (rmap, map_ty) = self.rewrite_expr(&args[0], env, tymap, None)?;
                let (k, v) = map_kv_of(&map_ty).unwrap_or((Ty::Unit, Ty::Unit));
                let suffix = map_suffix(&k, &v);
                Ok(Some((
                    Expr::Call(format!("map_len${}", suffix), vec![rmap]),
                    Ty::Int,
                )))
            }
            "map_remove" => {
                let (rkey, k) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let (rmap, map_ty) =
                    self.rewrite_expr(&args[0], env, tymap, expected)?;
                let v = map_kv_of(&map_ty)
                    .map(|(_, v)| v)
                    .or_else(|| expected.and_then(map_kv_of).map(|(_, v)| v))
                    .unwrap_or(Ty::Unit);
                if !native_map_value_ok(&v) {
                    return Err(unsupported_map_value(&v));
                }
                let suffix = map_suffix(&k, &v);
                Ok(Some((
                    Expr::Call(format!("map_remove${}", suffix), vec![rmap, rkey]),
                    mapty(k, v),
                )))
            }
            "map_show" => {
                let (rmap, map_ty) = self.rewrite_expr(&args[0], env, tymap, None)?;
                let (k, v) = map_kv_of(&map_ty).unwrap_or((Ty::Unit, Ty::Unit));
                let suffix = map_suffix(&k, &v);
                Ok(Some((
                    Expr::Call(format!("map_show${}", suffix), vec![rmap]),
                    Ty::Str,
                )))
            }
            // ---- Set ----
            "set_new" => {
                let t = expected.and_then(set_elem_of).unwrap_or(Ty::Unit);
                let suffix = array_elem_tag(&t);
                Ok(Some((
                    Expr::Call(format!("set_new${}", suffix), Vec::new()),
                    setty(t),
                )))
            }
            "set_add" => {
                let (relem, t) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let (rset, _) = self.rewrite_expr(&args[0], env, tymap, Some(&setty(t.clone())))?;
                let suffix = array_elem_tag(&t);
                Ok(Some((
                    Expr::Call(format!("set_add${}", suffix), vec![rset, relem]),
                    setty(t),
                )))
            }
            "set_has" => {
                let (relem, t) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let (rset, _) = self.rewrite_expr(&args[0], env, tymap, Some(&setty(t.clone())))?;
                let suffix = array_elem_tag(&t);
                Ok(Some((
                    Expr::Call(format!("set_has${}", suffix), vec![rset, relem]),
                    Ty::Bool,
                )))
            }
            "set_len" => {
                let (rset, set_ty) = self.rewrite_expr(&args[0], env, tymap, None)?;
                let t = set_elem_of(&set_ty).unwrap_or(Ty::Unit);
                let suffix = array_elem_tag(&t);
                Ok(Some((
                    Expr::Call(format!("set_len${}", suffix), vec![rset]),
                    Ty::Int,
                )))
            }
            "set_remove" => {
                let (relem, t) = self.rewrite_expr(&args[1], env, tymap, None)?;
                let (rset, _) =
                    self.rewrite_expr(&args[0], env, tymap, Some(&setty(t.clone())))?;
                let suffix = array_elem_tag(&t);
                Ok(Some((
                    Expr::Call(format!("set_remove${}", suffix), vec![rset, relem]),
                    setty(t),
                )))
            }
            "set_show" => {
                let (rset, set_ty) = self.rewrite_expr(&args[0], env, tymap, None)?;
                let t = set_elem_of(&set_ty).unwrap_or(Ty::Unit);
                let suffix = array_elem_tag(&t);
                Ok(Some((
                    Expr::Call(format!("set_show${}", suffix), vec![rset]),
                    Ty::Str,
                )))
            }
            _ => Ok(None),
        }
    }

    /// Statically resolve a trait-method (dispatcher) call to the concrete impl
    /// function for the receiver's runtime type. The receiver's rewritten type is
    /// a concrete (possibly mangled) `Named(H, [])`; the impl function is named
    /// `m$Trait$Head` where `Head` is the ORIGINAL (unmangled) head type. The impl
    /// function is concrete (no type params), so we specialize it at no type args.
    fn rewrite_trait_call(
        &mut self,
        name: &str,
        args: &[Expr],
        env: &mut HashMap<String, Ty>,
        tymap: &HashMap<String, Ty>,
    ) -> Result<(Expr, Ty), String> {
        let mi = self
            .traits
            .methods
            .get(name)
            .cloned()
            .expect("caller checked this is a trait method");
        if args.is_empty() {
            return Err(format!(
                "monomorphize: trait method `{}` has no receiver argument",
                name
            ));
        }
        // Rewrite the receiver to learn its concrete type.
        let (rrecv, recv_ty) = self.rewrite_expr(&args[0], env, tymap, None)?;
        let head_mangled = match &recv_ty {
            Ty::Named(h, _) => h.clone(),
            other => {
                return Err(format!(
                    "monomorphize: trait method `{}` called on non-ADT receiver of type {}",
                    name,
                    crate::typeck::show(other)
                ))
            }
        };
        // Recover the original (source) head-type name from the mangled name.
        let head_orig = self
            .demangle
            .get(&head_mangled)
            .map(|(orig, _)| orig.clone())
            .unwrap_or(head_mangled);
        let impl_fn = crate::traits::impl_method_name(name, &mi.trait_name, &head_orig);
        let info = self.fns.get(&impl_fn).cloned().ok_or_else(|| {
            format!(
                "monomorphize: no impl of `{}` for `{}` (resolving call to `{}`)",
                mi.trait_name, head_orig, name
            )
        })?;
        // The impl method is concrete; rewrite each remaining argument against the
        // impl's (already-concrete) parameter types.
        let mut rargs = vec![rrecv];
        for (p, a) in info.decl.params.iter().zip(args.iter()).skip(1) {
            let pexp = self.mangle_concrete(&p.ty);
            let (ra, _at) = self.rewrite_expr(a, env, tymap, Some(&pexp))?;
            rargs.push(ra);
        }
        let target = self.specialize_fn(&impl_fn, &[])?;
        let rt = self.mangle_concrete(&info.decl.ret);
        Ok((Expr::Call(target, rargs), rt))
    }

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

            // Records desugar to positional ADT form HERE, where the receiver's
            // concrete record type is known. After this, the IR + backends see
            // only ordinary `Ctor`/`Match` (a record is a 1-variant ADT cell).
            //
            // A record literal is exactly a positional constructor application,
            // so reorder its fields and recurse through the `Ctor` arm (which
            // handles all the generic mangling / type-arg solving).
            Expr::Record(name, fields) => {
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("monomorphize: unknown record `{}`", name))?;
                let decl_names = sig
                    .field_names
                    .clone()
                    .ok_or_else(|| format!("monomorphize: `{}` is not a record", name))?;
                let ordered: Vec<Expr> = decl_names
                    .iter()
                    .map(|fname| fields.iter().find(|(n, _)| n == fname).unwrap().1.clone())
                    .collect();
                self.rewrite_expr(&Expr::Ctor(name.clone(), ordered), env, tymap, expected)
            }
            // `obj.field` -> `match obj { Ctor(b0,..,bn) => b_idx }`. `obj` is
            // rewritten first to learn its concrete (mangled) record type.
            Expr::Field(obj, field) => {
                let (robj, obj_ty) = self.rewrite_expr(obj, env, tymap, None)?;
                let (ctor, fnames, ftys) = self.record_shape(&obj_ty)?;
                let idx = fnames
                    .iter()
                    .position(|n| n == field)
                    .ok_or_else(|| format!("monomorphize: no field `{}` on `{}`", field, ctor))?;
                let binders = fresh_field_binders(fnames.len());
                let pat =
                    Pattern::Ctor(ctor, binders.iter().cloned().map(Pattern::Var).collect());
                let arm = Arm { pat, body: Expr::Var(binders[idx].clone()) };
                Ok((Expr::Match(Box::new(robj), vec![arm]), ftys[idx].clone()))
            }
            // `{ base | f = v }` -> `match base { Ctor(b0,..,bn) => Ctor(g0,..,gn) }`
            // where g_i is the new value for updated fields, else Var(b_i).
            Expr::Update(base, updates) => {
                let (rbase, base_ty) = self.rewrite_expr(base, env, tymap, None)?;
                let (ctor, fnames, _ftys) = self.record_shape(&base_ty)?;
                let binders = fresh_field_binders(fnames.len());
                let mut new_vals: HashMap<String, Expr> = HashMap::new();
                for (fname, val) in updates {
                    let (rv, _) = self.rewrite_expr(val, env, tymap, None)?;
                    new_vals.insert(fname.clone(), rv);
                }
                let rebuilt: Vec<Expr> = fnames
                    .iter()
                    .zip(binders.iter())
                    .map(|(fname, b)| new_vals.remove(fname).unwrap_or_else(|| Expr::Var(b.clone())))
                    .collect();
                let pat = Pattern::Ctor(
                    ctor.clone(),
                    binders.iter().cloned().map(Pattern::Var).collect(),
                );
                let arm = Arm { pat, body: Expr::Ctor(ctor, rebuilt) };
                Ok((Expr::Match(Box::new(rbase), vec![arm]), base_ty))
            }

            Expr::Var(name) => {
                if let Some(ty) = env.get(name).cloned() {
                    return Ok((Expr::Var(name.clone()), ty));
                }
                // A top-level function referenced as a value (e.g. passed by name
                // to a higher-order function). Specialize it — non-generic only,
                // since a generic function used as a bare value gives no way to
                // pick its type arguments — and yield its function type. Lowering
                // wraps it in a zero-capture closure.
                if let Some(info) = self.fns.get(name).cloned() {
                    if !info.decl.type_params.is_empty() {
                        return Err(format!(
                            "monomorphize: generic function `{}` used as a value is unsupported (apply it instead)",
                            name
                        ));
                    }
                    let target = self.specialize_fn(name, &[])?;
                    let pts: Vec<Ty> =
                        info.decl.params.iter().map(|p| self.mangle_concrete(&p.ty)).collect();
                    let rt = self.mangle_concrete(&info.decl.ret);
                    return Ok((Expr::Var(target), Ty::Fn(pts, Box::new(rt))));
                }
                Err(format!("monomorphize: unbound variable `{}`", name))
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
                // A local `Fn`-typed binding applied by name (e.g. a function
                // parameter `f` in `f(x)`) — the tree-walker resolves the scope
                // before the global function table, so do the same here. Keep it
                // an `Expr::Call`; lowering turns a non-global, non-builtin callee
                // into a closure application.
                if let Some(Ty::Fn(param_tys, ret)) = env.get(name).cloned() {
                    let mut rargs = Vec::new();
                    for (i, a) in args.iter().enumerate() {
                        let (ra, _at) = self.rewrite_expr(a, env, tymap, param_tys.get(i))?;
                        rargs.push(ra);
                    }
                    // Emit a typed application so the backends know the result
                    // type; lowering turns an applied local variable into a
                    // closure call.
                    return Ok((
                        Expr::Apply(
                            Box::new(Expr::Var(name.clone())),
                            rargs,
                            Some((*ret).clone()),
                        ),
                        (*ret).clone(),
                    ));
                }
                // Trait-method (dispatcher) call: resolve STATICALLY to the impl
                // method for the receiver's concrete type. The dispatcher itself
                // is never specialized/emitted — the backends only ever see the
                // mangled impl function `m$Trait$Head`.
                if self.traits.is_method(name) {
                    return self.rewrite_trait_call(name, args, env, tymap);
                }
                // `array_lit` is the variadic desugaring of an array literal. It
                // is not in the signature table, so type it directly: the result
                // is `Array[elem]` where `elem` is the (concrete) type of the
                // first rewritten argument. An empty literal has no element type
                // to recover here; fall back to an expected `Array[..]` if the
                // context supplies one, else leave the element unresolved.
                if name == "array_lit" {
                    let mut rargs = Vec::new();
                    let mut elem: Option<Ty> = None;
                    for a in args {
                        let (ra, at) = self.rewrite_expr(a, env, tymap, None)?;
                        if elem.is_none() {
                            elem = Some(at);
                        }
                        rargs.push(ra);
                    }
                    let elem = elem
                        .or_else(|| match expected {
                            Some(Ty::Named(n, eargs)) if n == "Array" && eargs.len() == 1 => {
                                Some(eargs[0].clone())
                            }
                            _ => None,
                        })
                        .unwrap_or(Ty::Unit);
                    // Suffix the emitted name with the element-kind tag so the
                    // native/wasm backends learn the concrete element type from
                    // the (post-monomorphization) call name. The IR interpreter
                    // never runs monomorphize, so it keeps using `array_lit`.
                    let suffixed = format!("array_lit${}", array_elem_tag(&elem));
                    let rt = Ty::Named("Array".to_string(), vec![elem]);
                    return Ok((Expr::Call(suffixed, rargs), rt));
                }
                // Builtin?  (Identified by NOT being a user function.) Rewrite the
                // arguments, then compute the result type. For a GENERIC builtin
                // (one whose signature mentions type variables, e.g.
                // `array_get: (Array[T], Int) -> T`), unify the signature's
                // parameter types against the concrete argument types to build a
                // substitution and apply it to the return type — exactly like the
                // user-fn path. This keeps `Ty::Var` from leaking into the
                // post-monomorphization IR.
                if !self.fns.contains_key(name) {
                    // Array builtins need their concrete element type threaded to
                    // the backends (as a name suffix) AND propagated INTO the array
                    // argument so a nested `array_new()` resolves its element type.
                    if let Some(r) = self.rewrite_array_op(name, args, env, tymap, expected)? {
                        return Ok(r);
                    }
                    // Map/Set builtins are threaded the same way: concrete key/
                    // value (element) types pushed into the collection argument
                    // and encoded as the call-name suffix the backends dispatch on.
                    if let Some(r) = self.rewrite_map_set_op(name, args, env, tymap, expected)? {
                        return Ok(r);
                    }
                    let mut rargs = Vec::new();
                    let mut arg_tys = Vec::new();
                    for a in args {
                        let (ra, at) = self.rewrite_expr(a, env, tymap, None)?;
                        rargs.push(ra);
                        arg_tys.push(at);
                    }
                    let mut rt = builtin_ret(name, &arg_tys);
                    if contains_var(&rt) {
                        if let Some((sig_params, sig_ret)) = crate::builtins::lookup(name) {
                            let mut sub: HashMap<String, Ty> = HashMap::new();
                            if let Some(exp) = expected {
                                let _ = self.unify_decl(&sig_ret, exp, &mut sub);
                            }
                            for (pt, at) in sig_params.iter().zip(arg_tys.iter()) {
                                let _ = self.unify_decl(pt, at, &mut sub);
                            }
                            rt = resolve(&rt, &sub);
                        }
                    }
                    return Ok((Expr::Call(name.clone(), rargs), rt));
                }
                let info = self.fns.get(name).cloned().unwrap();
                let mut sub: HashMap<String, Ty> = HashMap::new();
                // Seed from the expected return type if it pins type params.
                if let Some(exp) = expected {
                    let _ = self.unify_decl(&info.decl.ret, exp, &mut sub);
                }
                // FIRST PASS: solve the callee's type parameters by unifying its
                // declared parameter types against best-effort synthesized
                // argument types, ACROSS ALL arguments together. This lets one
                // argument determine a parameter that another argument leaves
                // free (the Either/`pick` case: `db: B = true` fixes `B`, so the
                // under-determined `e: E[A, B] = Lft(5)` gets a concrete
                // expected type below). Args we cannot synthesize cheaply are
                // simply skipped here; they will be unified in the second pass.
                for (p, a) in info.decl.params.iter().zip(args.iter()) {
                    if let Some(at) = self.synth_ty(a, env) {
                        let _ = self.unify_decl(&p.ty, &at, &mut sub);
                    }
                }
                // SECOND PASS: now substitute the solved params to get each
                // param's concrete type and thread it down as the EXPECTED type
                // while rewriting, refining `sub` from the rewritten arg types.
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

            Expr::Lambda(params, body, _) => {
                // Concrete parameter types come from the expected `Fn` type when
                // the lambda flows into a known position (a function argument or a
                // direct application); otherwise fall back to the (substituted)
                // annotation. The body is rewritten in the extended scope; the
                // lambda's type is the resulting `Ty::Fn`.
                let exp_params: Option<Vec<Ty>> = match expected {
                    Some(Ty::Fn(ps, _)) => Some(ps.clone()),
                    _ => None,
                };
                let exp_ret: Option<Ty> = match expected {
                    Some(Ty::Fn(_, r)) => Some((**r).clone()),
                    _ => None,
                };
                // Captures: free variables of the lambda that name an in-scope
                // local (globals/builtins are not in `env`). Computed against the
                // OUTER scope, with concrete types, in a stable (sorted) order.
                let mut param_set = std::collections::HashSet::new();
                for (n, _) in params {
                    param_set.insert(n.clone());
                }
                let mut fvs = std::collections::HashSet::new();
                crate::ir::ast_free(body, &param_set, &mut fvs);
                let mut captures: Vec<(String, Ty)> = fvs
                    .into_iter()
                    .filter_map(|n| env.get(&n).map(|t| (n, t.clone())))
                    .collect();
                captures.sort_by(|a, b| a.0.cmp(&b.0));
                // Rewrite the body with the parameters bound to concrete types.
                let mut body_scope = env.clone();
                let mut rparams: Vec<(String, Ty)> = Vec::new();
                for (i, (pn, pann)) in params.iter().enumerate() {
                    let pty = match exp_params.as_ref().and_then(|ps| ps.get(i)) {
                        Some(t) => t.clone(),
                        // No expectation from context: rely on the annotation. An
                        // unannotated parameter (a parser `$lam` placeholder) whose
                        // type only the surrounding context could fix — e.g. a bare
                        // `let f = \x -> ..` later passed to a typed position — is
                        // not yet supported by the compiled backends; ask for an
                        // annotation rather than emitting a confusing internal error.
                        None => match pann {
                            Ty::Var(v) if v.starts_with("$lam") => {
                                return Err(format!(
                                    "monomorphize: cannot infer the type of unannotated lambda parameter `{}` in this context — annotate it, e.g. `\\({}: Int) -> ...`",
                                    pn, pn
                                ))
                            }
                            other => self.subst_ty(other, tymap)?,
                        },
                    };
                    body_scope.insert(pn.clone(), pty.clone());
                    rparams.push((pn.clone(), pty));
                }
                let (rbody, bt) =
                    self.rewrite_expr(body, &mut body_scope, tymap, exp_ret.as_ref())?;
                let lam_ty = Ty::Fn(
                    rparams.iter().map(|(_, t)| t.clone()).collect(),
                    Box::new(bt.clone()),
                );
                let sig = crate::ast::ClosureSig { captures, ret: bt };
                Ok((Expr::Lambda(rparams, Box::new(rbody), Some(sig)), lam_ty))
            }

            Expr::Apply(callee, args, _) => {
                // Rewrite the arguments first so a directly-applied lambda
                // (`(\x -> ..)(5)`) can take its parameter types from them.
                let mut rargs = Vec::new();
                let mut arg_tys = Vec::new();
                for a in args {
                    let (ra, at) = self.rewrite_expr(a, env, tymap, None)?;
                    rargs.push(ra);
                    arg_tys.push(at);
                }
                let callee_expect = Ty::Fn(
                    arg_tys.clone(),
                    Box::new(expected.cloned().unwrap_or(Ty::Unit)),
                );
                let (rcallee, ct) =
                    self.rewrite_expr(callee, env, tymap, Some(&callee_expect))?;
                let ret_ty = match &ct {
                    Ty::Fn(_, r) => (**r).clone(),
                    other => {
                        return Err(format!(
                            "monomorphize: applying a non-function value of type {:?}",
                            other
                        ))
                    }
                };
                Ok((
                    Expr::Apply(Box::new(rcallee), rargs, Some(ret_ty.clone())),
                    ret_ty,
                ))
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
            // Record patterns are interpreter-only so far; cleanly rejected in the
            // compiled pipeline (records never reach here in practice, since a
            // record scrutinee comes from a record expression already gated).
            // `Ctor { f: p, .. }` -> positional `Ctor(p0, .., pn)` (unmentioned
            // fields become `_`), then recurse so the ctor name is mangled from
            // the scrutinee type and sub-patterns are bound.
            Pattern::Record(name, sub_fields) => {
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("monomorphize: unknown record `{}`", name))?;
                let fnames = sig
                    .field_names
                    .clone()
                    .ok_or_else(|| format!("monomorphize: `{}` is not a record", name))?;
                let ordered: Vec<Pattern> = fnames
                    .iter()
                    .map(|fname| {
                        sub_fields
                            .iter()
                            .find(|(n, _)| n == fname)
                            .map(|(_, p)| p.clone())
                            .unwrap_or(Pattern::Wild)
                    })
                    .collect();
                self.rewrite_pattern(&Pattern::Ctor(name.clone(), ordered), scrut_ty, binds)
            }
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
                    // Resolve the target first, then refuse an identity or cyclic
                    // binding (`v -> v`, or `v -> ...v...`). Such a binding makes
                    // `resolve` loop forever; it arises when an unresolved generic
                    // builtin result (e.g. `array_new`'s `Array[T]`) is unified
                    // against a signature reusing the same param name `T`. Skipping
                    // it lets a later, concrete binding (`T -> Int`) win.
                    let c = resolve(concrete, sub);
                    if !ty_mentions(&c, v) {
                        sub.insert(v.clone(), c);
                    }
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
            // Function types unify componentwise (closure-typed parameters).
            (Ty::Fn(p1, r1), Ty::Fn(p2, r2)) if p1.len() == p2.len() => {
                for (x, y) in p1.iter().zip(p2.iter()) {
                    self.unify_decl(x, y, sub)?;
                }
                self.unify_decl(r1, r2, sub)
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

/// Map a concrete array-element `Ty` to the one-char element-kind tag that the
/// native/wasm backends dispatch on. `Int`/`Bool` are unboxed integers (`"i"`),
/// `Float` an unboxed double (`"f"`), `Str` a heap string (`"s"`), and every
/// other element — ADTs, nested `Array`/`Tensor`, etc. — a boxed heap ref
/// (`"r"`). Used to suffix the six array-builtin call names (`array_get$i`, …).
fn array_elem_tag(elem: &Ty) -> &'static str {
    match elem {
        Ty::Int | Ty::Bool => "i",
        Ty::Float => "f",
        Ty::Str => "s",
        _ => "r",
    }
}

/// Fresh, collision-proof binder names for a desugared record destructuring.
static FIELD_BINDER_CTR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn fresh_field_binders(n: usize) -> Vec<String> {
    let base = FIELD_BINDER_CTR.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    (0..n).map(|i| format!("$rf{}", base + i)).collect()
}

/// The element type `E` of an `Array[E]`, if `ty` is one.
fn array_elem_of(ty: &Ty) -> Option<Ty> {
    match ty {
        Ty::Named(n, args) if n == "Array" && args.len() == 1 => Some(args[0].clone()),
        _ => None,
    }
}

/// The (key, value) types `(K, V)` of a `Map[K, V]`, if `ty` is one.
fn map_kv_of(ty: &Ty) -> Option<(Ty, Ty)> {
    match ty {
        Ty::Named(n, args) if n == "Map" && args.len() == 2 => {
            Some((args[0].clone(), args[1].clone()))
        }
        _ => None,
    }
}

/// The element type `T` of a `Set[T]`, if `ty` is one.
fn set_elem_of(ty: &Ty) -> Option<Ty> {
    match ty {
        Ty::Named(n, args) if n == "Set" && args.len() == 1 => Some(args[0].clone()),
        _ => None,
    }
}

/// The `$<keytag>_<valtag>` element-kind suffix for a map builtin, reusing the
/// array element-kind tags (`i`/`f`/`s`/`r`). The key tag is always `i` or `s`
/// (the checker restricts keys to Int/Str); the value tag is unrestricted.
fn map_suffix(k: &Ty, v: &Ty) -> String {
    format!("{}_{}", array_elem_tag(k), array_elem_tag(v))
}

/// Whether a concrete Map value type round-trips faithfully through the COMPILED
/// backends. Native maps store values under a coarse slot tag with no nested
/// type info, so only flat value kinds (Int/Float/Bool/Str/Bytes) survive
/// get/show/== without losing layout or comparing by raw pointer. The
/// interpreter supports any value type; this guards the compiled path only.
/// `Unit` is the not-yet-pinned placeholder (an unused `map_new()`), left to
/// fail later if the map is actually given a concrete bad value.
fn native_map_value_ok(v: &Ty) -> bool {
    matches!(v, Ty::Int | Ty::Float | Ty::Bool | Ty::Str | Ty::Unit)
        || matches!(v, Ty::Named(n, _) if n == "Bytes")
}

/// Clean error for an unsupported compiled-backend Map value type.
fn unsupported_map_value(v: &Ty) -> String {
    format!(
        "a Map value of type `{}` is not yet supported in the compiled backends \
         (supported value types: Int, Float, Bool, Str, Bytes); \
         use the interpreter `aria run` for richer value types",
        crate::typeck::show(v)
    )
}

fn resolve(ty: &Ty, sub: &HashMap<String, Ty>) -> Ty {
    match ty {
        Ty::Var(v) => match sub.get(v) {
            Some(t) => resolve(t, sub),
            None => ty.clone(),
        },
        Ty::Named(n, args) => Ty::Named(n.clone(), args.iter().map(|a| resolve(a, sub)).collect()),
        Ty::Fn(ps, r) => Ty::Fn(
            ps.iter().map(|a| resolve(a, sub)).collect(),
            Box::new(resolve(r, sub)),
        ),
        other => other.clone(),
    }
}

fn contains_var(ty: &Ty) -> bool {
    match ty {
        Ty::Var(_) => true,
        Ty::Named(_, args) => args.iter().any(contains_var),
        Ty::Fn(ps, r) => ps.iter().any(contains_var) || contains_var(r),
        _ => false,
    }
}

/// Does `ty` mention the type variable named `v`? Used as an occurs check before
/// inserting a substitution, so `unify_decl` never creates a cyclic binding that
/// would make `resolve` loop.
fn ty_mentions(ty: &Ty, v: &str) -> bool {
    match ty {
        Ty::Var(n) => n == v,
        Ty::Named(_, args) => args.iter().any(|a| ty_mentions(a, v)),
        Ty::Fn(ps, r) => ps.iter().any(|p| ty_mentions(p, v)) || ty_mentions(r, v),
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
