//! Traits / interfaces (Aria M3).
//!
//! Aria's traits are STATIC: an `interface` declares method signatures, an
//! `impl` provides them for one concrete head type, and dispatch is resolved by
//! monomorphization (no vtables, no inheritance, no subtyping). The method CALL
//! is an ordinary free-function call `show(p)` — `.` stays record-field-only.
//!
//! ## Lowering (this module's `lower`)
//!
//! Interfaces and impls never reach the backends. The parser collects them into
//! side vectors and calls `lower`, which rewrites them into ordinary
//! `Item::Fn`s so every downstream stage (typeck, interp, ir, c_backend, wasm)
//! sees only `Item::Fn`/`Item::Type`:
//!
//!   * each impl method `fn show(self: Point) -> String = ..` becomes a mangled
//!     free function `show$Show$Point`;
//!   * each trait method becomes a generic DISPATCHER `fn show[Self: Show](self:
//!     Self, ..) -> Ret = match self { <ctors of Point> => show$Show$Point(self,
//!     ..), <ctors of Color> => show$Show$Color(self, ..), .. }`. The dispatcher
//!     enumerates the constructors of every type with an `impl` of this trait.
//!
//! The interpreter runs the dispatcher directly: it picks the arm by the
//! receiver's runtime constructor, so no monomorphization is needed. The
//! compiled backends never see a dispatcher: the monomorphizer resolves a
//! dispatcher call at a CONCRETE receiver type straight to the mangled impl
//! function (see `monomorphize`).
//!
//! ## Reconstructing the trait index downstream
//!
//! After lowering only `Item::Fn`s remain, but typeck and monomorphize still
//! need the trait/impl structure. `index` recomputes it from the lowered
//! program purely from naming: an impl method is exactly a function whose name
//! is `method$Trait$Head` (the `$` separator cannot occur in a user identifier),
//! and a dispatcher is the plain-named generic function for that method.

use std::collections::{HashMap, HashSet};

use crate::ast::*;

/// The mangled impl-method name, e.g. `show$Show$Point`.
pub fn impl_method_name(method: &str, trait_name: &str, head: &str) -> String {
    format!("{}${}${}", method, trait_name, head)
}

/// Lower interfaces + impls into ordinary `Item::Fn`s, appended to `items`.
/// Performs the structural checks that are most natural here (so they surface as
/// clean parse-time errors rather than internal panics later): impl
/// completeness, no extra methods, coherence (one impl per (trait, head)), and
/// that every head type / referenced trait exists.
pub fn lower(
    items: &mut Vec<Item>,
    interfaces: &[InterfaceDecl],
    impls: &[ImplDecl],
) -> Result<(), String> {
    // Index interfaces by name.
    let mut traits: HashMap<String, &InterfaceDecl> = HashMap::new();
    for itf in interfaces {
        if traits.insert(itf.name.clone(), itf).is_some() {
            return Err(format!("duplicate interface `{}`", itf.name));
        }
    }

    // Every trait bound on an ordinary user function must name a declared
    // interface. (Bounds parse freely, so a typo'd or undeclared trait — e.g.
    // `fn f[T: Bogus](..)` — would otherwise be silently ignored: no dispatcher
    // exists for it, so no trait-method call could ever resolve through it.)
    // Checked here, before any dispatcher is appended, so `items` holds only the
    // user's own functions. Runs even when there are no interfaces/impls so a
    // lone bogus bound is still caught.
    for it in items.iter() {
        if let Item::Fn(f) = it {
            for (param, tr) in &f.bounds {
                if !traits.contains_key(tr) {
                    return Err(format!(
                        "function `{}`: trait bound `{}: {}` names an undeclared interface `{}`",
                        f.name, param, tr, tr
                    ));
                }
            }
        }
    }

    // Each interface method's receiver must be its FIRST parameter and must be
    // typed exactly as the interface's type parameter (`Self`). The dispatcher
    // routes on that first argument's runtime constructor, so a method that puts
    // the `Self`-typed parameter elsewhere — or never mentions `Self` at all
    // (e.g. `fn show(self: Int)`) — would dispatch on the wrong value and let an
    // incoherent impl type-check, then crash/diverge in the backends. Require it
    // up front with a clean error.
    for itf in interfaces {
        for m in &itf.methods {
            match m.params.first() {
                Some(p) if p.ty == Ty::Var(itf.self_param.clone()) => {}
                Some(_) => {
                    return Err(format!(
                        "interface `{}` method `{}`: the first parameter must be the receiver typed `{}` (the interface's type parameter)",
                        itf.name, m.name, itf.self_param
                    ))
                }
                None => {
                    return Err(format!(
                        "interface `{}` method `{}` must take a receiver typed `{}` as its first parameter",
                        itf.name, m.name, itf.self_param
                    ))
                }
            }
        }
    }

    if interfaces.is_empty() && impls.is_empty() {
        return Ok(());
    }

    // Index user types by name (so a dispatcher can enumerate their ctors). A
    // record type is a single-variant type whose variant name equals the type.
    let mut type_variants: HashMap<String, Vec<Variant>> = HashMap::new();
    for it in items.iter() {
        if let Item::Type(t) = it {
            type_variants.insert(t.name.clone(), t.variants.clone());
        }
    }

    // Coherence: at most one impl per (trait, head). Also collect, per trait, the
    // ordered list of head types that implement it (dispatch arms in source
    // order for determinism).
    let mut seen_impls: HashSet<(String, String)> = HashSet::new();
    let mut heads_of_trait: HashMap<String, Vec<String>> = HashMap::new();

    for imp in impls {
        let itf = traits.get(&imp.trait_name).ok_or_else(|| {
            format!("impl for unknown interface `{}`", imp.trait_name)
        })?;
        if !type_variants.contains_key(&imp.head_type) {
            return Err(format!(
                "impl `{}` for unknown type `{}`",
                imp.trait_name, imp.head_type
            ));
        }
        let key = (imp.trait_name.clone(), imp.head_type.clone());
        if !seen_impls.insert(key) {
            return Err(format!(
                "duplicate impl of `{}` for `{}`",
                imp.trait_name, imp.head_type
            ));
        }
        heads_of_trait
            .entry(imp.trait_name.clone())
            .or_default()
            .push(imp.head_type.clone());

        // Completeness + no-extras: the impl's method set must match the trait's.
        let trait_methods: HashSet<&str> =
            itf.methods.iter().map(|m| m.name.as_str()).collect();
        let mut provided: HashSet<&str> = HashSet::new();
        for m in &imp.methods {
            if !trait_methods.contains(m.name.as_str()) {
                return Err(format!(
                    "impl `{}` for `{}`: method `{}` is not part of interface `{}`",
                    imp.trait_name, imp.head_type, m.name, imp.trait_name
                ));
            }
            if !provided.insert(m.name.as_str()) {
                return Err(format!(
                    "impl `{}` for `{}`: duplicate method `{}`",
                    imp.trait_name, imp.head_type, m.name
                ));
            }
        }
        for tm in &itf.methods {
            if !provided.contains(tm.name.as_str()) {
                return Err(format!(
                    "impl `{}` for `{}`: missing method `{}`",
                    imp.trait_name, imp.head_type, tm.name
                ));
            }
        }

        // Emit each impl method as a mangled free function. Its signature must
        // match the trait method with `Self` := the head type; typeck verifies
        // that fully (here we just rename, preserving the author's body/params).
        for m in &imp.methods {
            let tm = itf
                .methods
                .iter()
                .find(|t| t.name == m.name)
                .expect("provided method is in trait set");
            // Arity match (a friendly early check; typeck does the full job).
            if m.params.len() != tm.params.len() {
                return Err(format!(
                    "impl `{}` for `{}`: method `{}` takes {} parameter(s) but the interface declares {}",
                    imp.trait_name,
                    imp.head_type,
                    m.name,
                    m.params.len(),
                    tm.params.len()
                ));
            }
            let mangled = impl_method_name(&m.name, &imp.trait_name, &imp.head_type);
            items.push(Item::Fn(FnDecl {
                name: mangled,
                // Preserve the source line of the impl method as written.
                line: m.line,
                pure: m.pure,
                type_params: Vec::new(),
                bounds: Vec::new(),
                params: m.params.clone(),
                ret: m.ret.clone(),
                body: m.body.clone(),
            }));
        }
    }

    // Emit a dispatcher per trait method, enumerating the constructors of every
    // head type that implements the trait.
    for itf in interfaces {
        let heads = heads_of_trait.get(&itf.name).cloned().unwrap_or_default();
        for tm in &itf.methods {
            // Dispatcher parameters: the trait method's own params, with the
            // receiver (`self`, the first parameter typed `Self`) kept as the
            // rigid trait parameter so the generic signature reads `[Self: Trait]
            // (self: Self, ..) -> Ret`.
            let params: Vec<Param> = tm.params.clone();
            let receiver = params
                .first()
                .map(|p| p.name.clone())
                .ok_or_else(|| {
                    format!(
                        "interface `{}` method `{}` must take a `self` receiver parameter",
                        itf.name, tm.name
                    )
                })?;

            // Build one match arm per (head type, its constructors). Each arm
            // re-applies the impl method to the dispatcher's own arguments.
            let mut arms: Vec<Arm> = Vec::new();
            for head in &heads {
                let variants = type_variants
                    .get(head)
                    .expect("head type exists (checked above)");
                let impl_fn = impl_method_name(&tm.name, &itf.name, head);
                let call_args: Vec<Expr> =
                    params.iter().map(|p| Expr::synth(ExprKind::Var(p.name.clone()))).collect();
                for v in variants {
                    let pat = Pattern::Ctor(
                        v.name.clone(),
                        v.fields.iter().map(|_| Pattern::Wild).collect(),
                    );
                    arms.push(Arm {
                        pat,
                        body: Expr::synth(ExprKind::Call(impl_fn.clone(), call_args.clone())),
                    });
                }
            }
            // A dispatcher with no impls at all still type-checks as a generic
            // function; but match needs >= 1 arm. If a trait has methods but no
            // impls, emit a body that simply errors at runtime is unnecessary —
            // any call would have failed the typeck's "no impl" check first. To
            // keep the AST well-formed, add a wildcard arm that recurses is wrong;
            // instead require at least one impl when a dispatcher is reachable.
            // We still must produce a valid body: with zero heads, fall back to a
            // single wildcard arm that re-dispatches to the first... there is
            // none, so use the receiver unchanged is ill-typed. Simplest: only
            // emit a dispatcher when there is at least one impl.
            if arms.is_empty() {
                continue;
            }
            let body = Expr::synth(ExprKind::Match(Box::new(Expr::synth(ExprKind::Var(receiver))), arms));
            items.push(Item::Fn(FnDecl {
                name: tm.name.clone(),
                // Compiler-generated trait dispatcher: no single source line.
                line: 0,
                pure: false,
                type_params: vec![itf.self_param.clone()],
                bounds: vec![(itf.self_param.clone(), itf.name.clone())],
                params,
                ret: tm.ret.clone(),
                body,
            }));
        }
    }

    Ok(())
}

// ===========================================================================
// Downstream reconstruction of the trait index from the LOWERED program.
// ===========================================================================

/// One trait method's signature, as recovered from the trait + dispatcher.
#[derive(Clone, Debug)]
pub struct MethodInfo {
    pub trait_name: String,
    /// Parameter types as declared on the dispatcher (the first is the receiver,
    /// typed `Ty::Var(self_param)`).
    pub params: Vec<Ty>,
    pub ret: Ty,
    /// The trait's `Self` type-parameter name (the dispatcher's type param).
    pub self_param: String,
}

/// The reconstructed trait/impl structure of a lowered program.
#[derive(Clone, Debug, Default)]
pub struct TraitIndex {
    /// method name -> its signature (one entry per trait method / dispatcher).
    pub methods: HashMap<String, MethodInfo>,
    /// (trait, head) pairs that have an impl.
    pub impls: HashSet<(String, String)>,
    /// trait -> ordered head types that implement it.
    pub heads: HashMap<String, Vec<String>>,
}

impl TraitIndex {
    /// Is `name` a trait method (i.e. a dispatcher)?
    pub fn is_method(&self, name: &str) -> bool {
        self.methods.contains_key(name)
    }
    /// Does an impl of `trait_name` exist for head type `head`?
    pub fn has_impl(&self, trait_name: &str, head: &str) -> bool {
        self.impls.contains(&(trait_name.to_string(), head.to_string()))
    }
}

/// Split a mangled impl-method name `method$Trait$Head` into its parts. Returns
/// `None` for any name that is not exactly a two-`$` impl-method name.
fn split_impl_name(name: &str) -> Option<(&str, &str, &str)> {
    let parts: Vec<&str> = name.split('$').collect();
    if parts.len() == 3 && !parts.iter().any(|p| p.is_empty()) {
        Some((parts[0], parts[1], parts[2]))
    } else {
        None
    }
}

/// Rebuild the trait index from a lowered program. Dispatchers carry the trait
/// method signatures (their generic param is `Self`, declared via `bounds`);
/// impl-method function names carry the (trait, head) coherence facts.
pub fn index(program: &Program) -> TraitIndex {
    let mut idx = TraitIndex::default();
    // Pass 1: collect impl facts from mangled impl-method names. This also yields
    // the set of `(method, trait)` pairs that actually have impls — the precise
    // marker for distinguishing a trait-method DISPATCHER (named exactly `m` with
    // a `[Self: Trait]` bound) from an ordinary BOUNDED user function (e.g.
    // `twice[T: Describe]`, which also carries a bound but is not a dispatcher).
    let mut method_traits: HashSet<(String, String)> = HashSet::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if let Some((m, tr, head)) = split_impl_name(&f.name) {
                idx.impls.insert((tr.to_string(), head.to_string()));
                // One head can appear under a multi-method trait once per method
                // fn; record it only once so `heads` lists each implementer once.
                let heads = idx.heads.entry(tr.to_string()).or_default();
                if !heads.iter().any(|h| h == head) {
                    heads.push(head.to_string());
                }
                method_traits.insert((m.to_string(), tr.to_string()));
            }
        }
    }
    // Pass 2: a dispatcher is a generic fn whose (name, bound-trait) pair has an
    // impl, and whose single bound is on its single type parameter.
    for item in &program.items {
        if let Item::Fn(f) = item {
            if split_impl_name(&f.name).is_some() {
                continue;
            }
            if f.type_params.len() == 1 && f.bounds.len() == 1 {
                let (param, trait_name) = &f.bounds[0];
                if param == &f.type_params[0]
                    && method_traits.contains(&(f.name.clone(), trait_name.clone()))
                {
                    idx.methods.insert(
                        f.name.clone(),
                        MethodInfo {
                            trait_name: trait_name.clone(),
                            params: f.params.iter().map(|p| p.ty.clone()).collect(),
                            ret: f.ret.clone(),
                            self_param: param.clone(),
                        },
                    );
                }
            }
        }
    }
    idx
}
