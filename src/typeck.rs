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
            // Synthetic tuple ADTs (`$TupleN`) print as `(A, B, ..)` — the surface
            // syntax — never leaking the internal `$TupleN` name to the user.
            if n.starts_with("$Tuple") && !args.is_empty() {
                let inner: Vec<String> = args.iter().map(show).collect();
                return format!("({})", inner.join(", "));
            }
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

/// Split a mangled impl-method name `method$Trait$Head` into its parts (the `$`
/// separator cannot occur in a user identifier). `None` otherwise.
fn split_impl(name: &str) -> Option<(&str, &str, &str)> {
    let parts: Vec<&str> = name.split('$').collect();
    if parts.len() == 3 && !parts.iter().any(|p| p.is_empty()) {
        Some((parts[0], parts[1], parts[2]))
    } else {
        None
    }
}

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
    field_names: Option<Vec<String>>, // Some iff a record constructor
}

/// A concrete (or partly-wildcard) value witnessing a non-exhaustive match,
/// used to produce a helpful "missing case `...`" error.
#[derive(Clone, Debug)]
enum Witness {
    Wild,
    Ctor(String, Vec<Witness>),
}

/// Build a column-witness vector from a freshly-constructed `head` witness and
/// the (optional) witness `tail` for the remaining columns. If the tail is
/// `None` (remaining columns were exhaustive), pad with wildcards so the
/// returned vector still has one entry per remaining column.
fn extend_witness(head: Witness, tail: Option<Vec<Witness>>, rest: usize) -> Vec<Witness> {
    let mut out = vec![head];
    match tail {
        Some(t) => out.extend(t),
        None => out.extend(std::iter::repeat(Witness::Wild).take(rest)),
    }
    out
}

// A function's declared signature, retaining its generic parameters.
#[derive(Clone)]
struct FnSig {
    type_params: Vec<String>,
    /// Trait bounds on the type parameters (`param -> trait`). At a call site,
    /// the concrete type a bounded parameter resolves to must have an `impl` of
    /// the named trait. Empty for builtins and unbounded functions.
    bounds: Vec<(String, String)>,
    params: Vec<Ty>,
    ret: Ty,
}

struct Checker {
    fns: HashMap<String, FnSig>,
    ctors: HashMap<String, CtorSig>,
    types: HashMap<String, (Vec<String>, Vec<String>)>, // type -> (params, variant ctor names)
    /// Reconstructed trait/impl structure (traits lowered to plain fns).
    traits: crate::traits::TraitIndex,
    /// Trait bounds of the function currently being checked: `param -> trait`.
    /// A trait-method call on a value of a bounded rigid param is allowed and
    /// deferred to monomorphization; on an unbounded param it is an error.
    cur_bounds: RefCell<HashMap<String, String>>,
    // Union-find / substitution for unification variables, plus a fresh counter.
    subst: RefCell<HashMap<String, Ty>>,
    counter: RefCell<u64>,
    // For each unannotated lambda parameter encountered while checking the
    // CURRENT function, the parser placeholder name (`$lamN`) paired with the
    // fresh unification variable assigned to it. After the function's body is
    // checked these resolve to concrete types, which `annotate_lambda_params`
    // writes back into the AST so the compiled backends can type unannotated
    // lambdas (e.g. `let f = \x -> ..`). Cleared per function.
    lam_vars: RefCell<Vec<(String, Ty)>>,
    /// The precise source span of the INNERMOST expression that produced the
    /// current per-function error, captured by `synth`/`check` so a structured
    /// diagnostic can point at the exact sub-expression rather than the
    /// function's definition line. Set on the FIRST error to surface (the
    /// deepest failing node sets it before any enclosing node), and cleared
    /// before each function body is checked. `None` (or a [`Span::none`] node)
    /// leaves the diagnostic's `col` unset.
    err_span: RefCell<Option<Span>>,
}

/// An error accumulator that records each message alongside an OPTIONAL precise
/// source span. Most checker errors are declaration-level and carry no span
/// (`push`); a body-level type error carries the span of the offending
/// sub-expression (`push_at`). `check` discards the spans (string-only public
/// API); `check_collect` keeps them for the structured `--json` path.
#[derive(Default)]
struct Errors {
    msgs: Vec<String>,
    spans: Vec<Option<Span>>,
}

impl Errors {
    /// Record a message with no precise location (declaration-level error).
    fn push(&mut self, m: String) {
        self.msgs.push(m);
        self.spans.push(None);
    }

    /// Record a message located at `span` (a body-level expression error). The
    /// no-location sentinel maps to `None` so the diagnostic's `col` stays unset.
    fn push_at(&mut self, m: String, span: Option<Span>) {
        self.msgs.push(m);
        self.spans.push(span.filter(|s| !s.is_none()));
    }

    /// Append plain (location-less) messages, e.g. tensor-shape diagnostics.
    fn extend_plain(&mut self, it: impl IntoIterator<Item = String>) {
        for m in it {
            self.push(m);
        }
    }

    fn is_empty(&self) -> bool {
        self.msgs.is_empty()
    }
}

/// Type-check a whole program. Returns every error found, not just the first.
/// The string messages are the single source of truth shared with the
/// structured (`--json`) path via [`check_collect`].
pub fn check(program: &Program) -> Result<(), Vec<String>> {
    match check_collect(program) {
        e if e.is_empty() => Ok(()),
        e => Err(e.msgs),
    }
}

/// The core checker: returns the [`Errors`] accumulator (messages + optional
/// precise spans). Both the string-only [`check`] and the structured
/// [`check_structured`] are thin wrappers over this, so their messages are
/// byte-for-byte identical.
fn check_collect(program: &Program) -> Errors {
    let mut fns: HashMap<String, FnSig> = HashMap::new();
    let mut ctors: HashMap<String, CtorSig> = HashMap::new();
    let mut types: HashMap<String, (Vec<String>, Vec<String>)> = HashMap::new();
    let mut errors = Errors::default();

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
                                field_names: v.field_names.clone(),
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
                // A user function may not shadow a built-in: a same-named user
                // `fn` (e.g. `array_get`, `concat`) would silently never run,
                // since calls resolve to the builtin first.
                if crate::builtins::lookup(&f.name).is_some() {
                    errors.push(format!(
                        "cannot redefine built-in function `{}`",
                        f.name
                    ));
                }
                let params = f.params.iter().map(|p| p.ty.clone()).collect();
                if fns
                    .insert(
                        f.name.clone(),
                        FnSig {
                            type_params: f.type_params.clone(),
                            bounds: f.bounds.clone(),
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
        errs: &mut Errors,
        ctx: &str,
    ) {
        match t {
            Ty::Named(n, args) if BUILTIN_TYPES.contains(&n.as_str()) => {
                // Built-in types have a fixed arity: `Array[T]` and `Set[T]`
                // take one type argument; `Map[K, V]` takes two; the others
                // (e.g. `Tensor`, `Bytes`) are nullary handles.
                let expected = match n.as_str() {
                    "Array" | "Set" => 1,
                    "Map" => 2,
                    _ => 0,
                };
                if args.len() != expected {
                    errs.push(format!(
                        "{}: built-in type `{}` takes {} type argument(s), got {}",
                        ctx,
                        n,
                        expected,
                        args.len()
                    ));
                }
                for a in args {
                    known(a, types, params, errs, ctx);
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
            Ty::Var(n) => {
                // A type variable in a SOURCE annotation must be one of the
                // declared generic parameters in scope. (Source types never
                // contain fresh `?N` unification vars, so this is safe.)
                if !params.contains(n) {
                    errs.push(format!("{}: unknown type parameter `{}`", ctx, n));
                }
            }
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

                // BUG E (backend parity): a declared type parameter that appears
                // in NONE of the parameter types can never be inferred at a call
                // site — typeck would silently leave it free while the compiled
                // backends reject it at monomorphization. Reject it up front for
                // parity. (Trait-method dispatchers, synthesized by lowering, are
                // exempt: their `Self` param is rigid, not call-site-inferred.)
                let is_dispatcher = f.bounds.iter().any(|(p, _)| f.type_params.contains(p));
                if !is_dispatcher {
                    for tp in &f.type_params {
                        if !f.params.iter().any(|p| ty_mentions_var(&p.ty, tp)) {
                            errors.push(format!(
                                "type parameter `{}` of `{}` is unused (cannot be inferred)",
                                tp, f.name
                            ));
                        }
                    }
                }

                // `main` is the entry point: it is called with no type arguments,
                // so it must not be generic.
                if f.name == "main" && !f.type_params.is_empty() {
                    errors.push("`main` must take no type parameters".to_string());
                }
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

    let traits = crate::traits::index(program);

    // Pass 2.5: validate impl methods against their trait signatures. Each impl
    // method `m$Trait$Head` must match the trait's declared `m` (the dispatcher)
    // with `Self` := the head type. The reconstructed `MethodInfo` carries the
    // dispatcher (= trait) signature; we substitute `Self := Head` and compare.
    for (name, sig) in &fns {
        if let Some((m, tr, head)) = split_impl(name) {
            // The dispatcher for this method holds the trait signature.
            let mi = match traits.methods.get(m) {
                Some(mi) if mi.trait_name == tr => mi,
                // No dispatcher (e.g. trait with no other impls reachable): the
                // lowering still emits one whenever an impl exists, so this only
                // fires for a genuinely malformed program; skip quietly.
                _ => continue,
            };
            let head_ty = Ty::Named(head.to_string(), Vec::new());
            let map: HashMap<String, Ty> =
                std::iter::once((mi.self_param.clone(), head_ty)).collect();
            if sig.params.len() != mi.params.len() {
                errors.push(format!(
                    "impl `{}` for `{}`: method `{}` takes {} parameter(s) but the interface declares {}",
                    tr, head, m, sig.params.len(), mi.params.len()
                ));
                continue;
            }
            for (i, (ip, tp)) in sig.params.iter().zip(mi.params.iter()).enumerate() {
                let want = Checker::apply_map(tp, &map);
                if &want != ip {
                    errors.push(format!(
                        "impl `{}` for `{}`: method `{}` parameter {} has type {} but the interface requires {}",
                        tr, head, m, i, show(ip), show(&want)
                    ));
                }
            }
            let want_ret = Checker::apply_map(&mi.ret, &map);
            if want_ret != sig.ret {
                errors.push(format!(
                    "impl `{}` for `{}`: method `{}` returns {} but the interface requires {}",
                    tr, head, m, show(&sig.ret), show(&want_ret)
                ));
            }
        }
    }

    let checker = Checker {
        fns,
        ctors,
        types,
        traits,
        cur_bounds: RefCell::new(HashMap::new()),
        subst: RefCell::new(HashMap::new()),
        counter: RefCell::new(0),
        lam_vars: RefCell::new(Vec::new()),
        err_span: RefCell::new(None),
    };

    // Pass 3: check each function body against its declared return type. Each
    // function gets a fresh unification context so leftover variables from one
    // body cannot leak into another.
    for item in &program.items {
        if let Item::Fn(f) = item {
            // A trait-method DISPATCHER (synthesized by lowering) routes a rigid
            // `Self` receiver by its runtime constructor — `match self { Point(..)
            // => .. }` deliberately matches a `Self`-typed scrutinee against a
            // concrete ctor, which ordinary checking would reject. Its arms are
            // generated mechanically from already-validated impls (Pass 2.5), so
            // skip body-checking it.
            if checker.traits.is_method(&f.name)
                && f.bounds.iter().any(|(p, _)| f.type_params.contains(p))
            {
                continue;
            }
            checker.subst.borrow_mut().clear();
            *checker.cur_bounds.borrow_mut() =
                f.bounds.iter().cloned().collect();
            let mut scope: Scope = vec![HashMap::new()];
            for p in &f.params {
                scope[0].insert(p.name.clone(), p.ty.clone());
            }
            // Clear the per-function error-span tracker; `synth`/`check` fill it
            // with the innermost offending sub-expression's span on error, so the
            // resulting diagnostic can point at the EXACT call site / operand
            // rather than the function's definition line.
            *checker.err_span.borrow_mut() = None;
            match checker.synth(&f.body, &mut scope) {
                Ok(t) => {
                    if let Err(e) = checker.unify(&t, &f.ret) {
                        // A whole-body return-type mismatch is located at the
                        // body's span (the function's result expression).
                        errors.push_at(
                            format!(
                                "function `{}`: body has type {} but return type is {} ({})",
                                f.name,
                                show(&checker.resolve(&t)),
                                show(&checker.resolve(&f.ret)),
                                e
                            ),
                            Some(f.body.span),
                        );
                    }
                }
                Err(e) => {
                    let span = *checker.err_span.borrow();
                    errors.push_at(format!("function `{}`: {}", f.name, e), span);
                }
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
    // The set of top-level function names — a callee in this set has its effects
    // tracked by `infer_io`; any *other* applied callee is a function value of
    // unknown effect (BUG D).
    let fnset: HashSet<String> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) => Some(f.name.clone()),
            _ => None,
        })
        .collect();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if !f.pure {
                continue;
            }
            if io.contains(&f.name) {
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
            } else if let Some(callee) = higher_order_effect_witness(&f.body, &fnset) {
                // Sound, conservative: Aria has no effect polymorphism, so a
                // `pure` function that applies a function VALUE cannot be proven
                // pure — the value's effects are unknown.
                errors.push(format!(
                    "cannot prove `{}` pure: it calls a function value `{}` whose effects are unknown",
                    f.name, callee
                ));
            }
        }
    }

    // Compile-time tensor shape checking. Run only on an otherwise well-typed
    // program, so shape inference sees consistent types and its messages are not
    // buried under unrelated type errors. Best-effort and false-positive-free:
    // it rejects a program only when statically-known tensor dimensions provably
    // do not line up (see `shape::check_program`).
    if errors.is_empty() {
        errors.extend_plain(crate::shape::check_program(program));
    }

    errors
}

/// Structured type/shape/purity/exhaustiveness checking for `aria check --json`.
///
/// Runs the SAME `check` pipeline that the human path uses (single source of
/// truth — the string messages are identical), then lifts each message into a
/// [`Diagnostic`], routing the sub-phases that historically share `check`'s
/// `Vec<String>` (purity, exhaustiveness, tensor shape) into their own `phase`
/// so consumers see a precise phase + a stable code. A clean program yields an
/// empty vector.
pub fn check_structured(program: &Program) -> Vec<crate::diagnostics::Diagnostic> {
    let errors = check_collect(program);
    errors
        .msgs
        .into_iter()
        .zip(errors.spans)
        .map(|(msg, span)| {
            let phase = phase_of_type_error(&msg);
            let mut d = crate::diagnostics::Diagnostic::error(phase, msg);
            // Attach the precise expression location when the checker knew it.
            // The `error` constructor already extracted any `line N:` prefix; a
            // real expression span supersedes it with both LINE and COLUMN (and
            // an end position) pointing at the exact sub-expression.
            if let Some(s) = span {
                d.set_span(s);
            }
            d
        })
        .collect()
}

/// Route one message from `typeck::check`'s flat error list to the phase that
/// actually produced it. Most are genuine `type` errors; a few sub-checks
/// (exhaustiveness, purity, tensor shape) flow through the same list and get
/// their own phase here so the structured output is precise.
fn phase_of_type_error(m: &str) -> &'static str {
    if m.contains("non-exhaustive match") {
        "exhaustiveness"
    } else if m.contains("is declared `pure` but performs IO") || m.starts_with("cannot prove `") {
        "purity"
    } else if is_shape_message(m) {
        "shape"
    } else {
        "type"
    }
}

/// Recognise a tensor-shape message (emitted by `shape::check_program`, which
/// prefixes `function `NAME`: ` then a shape-specific body). Matched on the
/// shape-vocabulary keywords those messages use, so it is robust to the
/// function-name prefix.
fn is_shape_message(m: &str) -> bool {
    m.contains("matmul expects")
        || m.contains("matmul inner dimensions")
        || m.contains("transpose expects")
        || m.contains("add requires identical shapes")
}

/// Infer and write back the concrete types of *unannotated* lambda parameters
/// into the AST, so later passes (monomorphization, the compiled backends) see a
/// type even for a bare `let f = \x -> ..` whose parameter type only the
/// surrounding context fixes. Best-effort: it assumes the program already
/// type-checked (callers run `check` first), re-runs body inference solely to
/// resolve each lambda placeholder, and rewrites `ExprKind::Lambda` parameter
/// annotations in place. Anything it cannot resolve is left untouched.
pub fn annotate_lambda_params(program: &mut Program) {
    // Pass 1 (mirrors `check`): gather declarations.
    let mut fns: HashMap<String, FnSig> = HashMap::new();
    let mut ctors: HashMap<String, CtorSig> = HashMap::new();
    let mut types: HashMap<String, (Vec<String>, Vec<String>)> = HashMap::new();
    for item in &program.items {
        match item {
            Item::Type(t) => {
                let mut variants = Vec::new();
                for v in &t.variants {
                    variants.push(v.name.clone());
                    ctors.entry(v.name.clone()).or_insert(CtorSig {
                        type_params: t.params.clone(),
                        fields: v.fields.clone(),
                        tyname: t.name.clone(),
                        field_names: v.field_names.clone(),
                    });
                }
                types.entry(t.name.clone()).or_insert((t.params.clone(), variants));
            }
            Item::Fn(f) => {
                let params = f.params.iter().map(|p| p.ty.clone()).collect();
                fns.entry(f.name.clone()).or_insert(FnSig {
                    type_params: f.type_params.clone(),
                    bounds: f.bounds.clone(),
                    params,
                    ret: f.ret.clone(),
                });
            }
        }
    }
    let traits = crate::traits::index(program);
    let checker = Checker {
        fns,
        ctors,
        types,
        traits,
        cur_bounds: RefCell::new(HashMap::new()),
        subst: RefCell::new(HashMap::new()),
        counter: RefCell::new(0),
        lam_vars: RefCell::new(Vec::new()),
        err_span: RefCell::new(None),
    };
    // Resolve each lambda placeholder by re-checking each body (a fresh context
    // per function, exactly like `check`'s Pass 3).
    let mut resolved: HashMap<String, Ty> = HashMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            // Skip dispatchers (their `match self { Ctor(..) => .. }` body is not
            // ordinarily well-typed; see Pass 3 in `check`).
            if checker.traits.is_method(&f.name)
                && f.bounds.iter().any(|(p, _)| f.type_params.contains(p))
            {
                continue;
            }
            checker.subst.borrow_mut().clear();
            *checker.cur_bounds.borrow_mut() = f.bounds.iter().cloned().collect();
            checker.lam_vars.borrow_mut().clear();
            let mut scope: Scope = vec![HashMap::new()];
            for p in &f.params {
                scope[0].insert(p.name.clone(), p.ty.clone());
            }
            if checker.synth(&f.body, &mut scope).is_ok() {
                for (name, var) in checker.lam_vars.borrow().iter() {
                    let r = checker.resolve(var);
                    // Only useful if it does not stay an unbound inference var.
                    if !matches!(&r, Ty::Var(v) if v.starts_with('?')) {
                        resolved.insert(name.clone(), r);
                    }
                }
            }
        }
    }
    if resolved.is_empty() {
        return;
    }
    for item in &mut program.items {
        if let Item::Fn(f) = item {
            annotate_expr(&mut f.body, &resolved);
        }
    }
}

/// Replace any unannotated lambda parameter annotation (`Ty::Var("$lamN")`) with
/// its resolved type, recursing through the whole expression tree.
fn annotate_expr(e: &mut Expr, resolved: &HashMap<String, Ty>) {
    match &mut e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::Unit | ExprKind::Var(_) => {}
        ExprKind::Ctor(_, args) | ExprKind::Call(_, args) => {
            for a in args {
                annotate_expr(a, resolved);
            }
        }
        ExprKind::Lambda(params, body, _) => {
            for (_, ty) in params.iter_mut() {
                if let Ty::Var(v) = ty {
                    if v.starts_with("$lam") {
                        if let Some(r) = resolved.get(v) {
                            *ty = r.clone();
                        }
                    }
                }
            }
            annotate_expr(body, resolved);
        }
        ExprKind::Apply(callee, args, _) => {
            annotate_expr(callee, resolved);
            for a in args {
                annotate_expr(a, resolved);
            }
        }
        ExprKind::Record(_, fields) => {
            for (_, v) in fields {
                annotate_expr(v, resolved);
            }
        }
        ExprKind::Field(obj, _) => annotate_expr(obj, resolved),
        ExprKind::Update(base, updates) => {
            annotate_expr(base, resolved);
            for (_, v) in updates {
                annotate_expr(v, resolved);
            }
        }
        ExprKind::Unary(_, inner) => annotate_expr(inner, resolved),
        ExprKind::Binary(_, l, r) => {
            annotate_expr(l, resolved);
            annotate_expr(r, resolved);
        }
        ExprKind::If(c, t, e2) => {
            annotate_expr(c, resolved);
            annotate_expr(t, resolved);
            annotate_expr(e2, resolved);
        }
        ExprKind::Match(scrut, arms) => {
            annotate_expr(scrut, resolved);
            for arm in arms {
                annotate_expr(&mut arm.body, resolved);
            }
        }
        ExprKind::Block(stmts, last) => {
            for s in stmts {
                match s {
                    Stmt::Let(_, _, v) => annotate_expr(v, resolved),
                    Stmt::Expr(ex) => annotate_expr(ex, resolved),
                }
            }
            annotate_expr(last, resolved);
        }
    }
}

/// Names of the IO-producing builtins. Everything else is pure.
const IO_BUILTINS: &[&str] = &["print_int", "print_float", "print_bool", "print_str"];

/// Does the declared type `ty` syntactically mention the type variable `v`?
/// Used to detect phantom/unused function type parameters (BUG E): a parameter
/// that appears in no argument type cannot be inferred at a call site.
fn ty_mentions_var(ty: &Ty, v: &str) -> bool {
    match ty {
        Ty::Var(n) => n == v,
        Ty::Named(_, args) => args.iter().any(|a| ty_mentions_var(a, v)),
        Ty::Fn(ps, r) => ps.iter().any(|p| ty_mentions_var(p, v)) || ty_mentions_var(r, v),
        _ => false,
    }
}

/// Collect the names called (directly) inside an expression, into `out`. This is
/// a pure syntactic walk over calls; it records both builtin and user-function
/// callees by name.
fn collect_calls(e: &Expr, out: &mut HashSet<String>) {
    match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::Unit
        | ExprKind::Var(_) => {}
        ExprKind::Ctor(_, args) => {
            for a in args {
                collect_calls(a, out);
            }
        }
        ExprKind::Call(name, args) => {
            out.insert(name.clone());
            for a in args {
                collect_calls(a, out);
            }
        }
        ExprKind::Lambda(_, body, _) => collect_calls(body, out),
        ExprKind::Apply(callee, args, _) => {
            collect_calls(callee, out);
            for a in args {
                collect_calls(a, out);
            }
        }
        ExprKind::Record(_, fields) => {
            for (_, v) in fields {
                collect_calls(v, out);
            }
        }
        ExprKind::Field(obj, _) => collect_calls(obj, out),
        ExprKind::Update(base, updates) => {
            collect_calls(base, out);
            for (_, v) in updates {
                collect_calls(v, out);
            }
        }
        ExprKind::Unary(_, x) => collect_calls(x, out),
        ExprKind::Binary(_, a, b) => {
            collect_calls(a, out);
            collect_calls(b, out);
        }
        ExprKind::If(c, t, e2) => {
            collect_calls(c, out);
            collect_calls(t, out);
            collect_calls(e2, out);
        }
        ExprKind::Match(scrut, arms) => {
            collect_calls(scrut, out);
            for a in arms {
                collect_calls(&a.body, out);
            }
        }
        ExprKind::Block(stmts, tail) => {
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

/// Find the first application of a function *value* with unknown effects inside
/// `e`, returning a name for the offending callee. Aria has no effect
/// polymorphism: when a `pure` function applies a function it received as a
/// parameter (or any captured/let-bound closure), the callee's effects are not
/// statically known, so the `pure` claim is unsound (the closure could perform
/// IO — see BUG D). We therefore treat *any* application of a non-top-level,
/// non-builtin callee as a potential effect.
///
/// Known-pure-or-tracked callees that do NOT count:
///   * a direct `ExprKind::Call` to a top-level function (its IO-ness is tracked by
///     `infer_io`) or to a builtin (IO builtins are caught separately),
///   * a directly-applied lambda literal `(\x -> ...)(a)` — its body is walked
///     in place, so its effects are visible.
fn higher_order_effect_witness(
    e: &Expr,
    fnset: &HashSet<String>,
) -> Option<String> {
    match &e.kind {
        ExprKind::Call(name, args) => {
            // A callee that is neither a top-level function nor a builtin must be
            // a function *value* in scope (a parameter or a let-bound closure):
            // applying it has unknown effects.
            if !fnset.contains(name) && builtin_sig(name).is_none() {
                return Some(name.clone());
            }
            for a in args {
                if let Some(w) = higher_order_effect_witness(a, fnset) {
                    return Some(w);
                }
            }
            None
        }
        ExprKind::Apply(callee, args, _) => {
            // Applying anything other than a lambda literal applies an unknown
            // function value.
            if !matches!(callee.kind, ExprKind::Lambda(..)) {
                let who = match &callee.kind {
                    ExprKind::Var(n) => n.clone(),
                    _ => "<closure value>".to_string(),
                };
                return Some(who);
            }
            if let Some(w) = higher_order_effect_witness(callee, fnset) {
                return Some(w);
            }
            for a in args {
                if let Some(w) = higher_order_effect_witness(a, fnset) {
                    return Some(w);
                }
            }
            None
        }
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::Unit
        | ExprKind::Var(_) => None,
        ExprKind::Ctor(_, args) => args.iter().find_map(|a| higher_order_effect_witness(a, fnset)),
        ExprKind::Lambda(_, body, _) => higher_order_effect_witness(body, fnset),
        ExprKind::Record(_, fields) => fields
            .iter()
            .find_map(|(_, v)| higher_order_effect_witness(v, fnset)),
        ExprKind::Field(obj, _) => higher_order_effect_witness(obj, fnset),
        ExprKind::Update(base, updates) => higher_order_effect_witness(base, fnset).or_else(|| {
            updates
                .iter()
                .find_map(|(_, v)| higher_order_effect_witness(v, fnset))
        }),
        ExprKind::Unary(_, x) => higher_order_effect_witness(x, fnset),
        ExprKind::Binary(_, a, b) => {
            higher_order_effect_witness(a, fnset).or_else(|| higher_order_effect_witness(b, fnset))
        }
        ExprKind::If(c, t, e2) => higher_order_effect_witness(c, fnset)
            .or_else(|| higher_order_effect_witness(t, fnset))
            .or_else(|| higher_order_effect_witness(e2, fnset)),
        ExprKind::Match(scrut, arms) => higher_order_effect_witness(scrut, fnset)
            .or_else(|| arms.iter().find_map(|a| higher_order_effect_witness(&a.body, fnset))),
        ExprKind::Block(stmts, tail) => {
            for s in stmts {
                let inner = match s {
                    Stmt::Let(_, _, x) => x,
                    Stmt::Expr(x) => x,
                };
                if let Some(w) = higher_order_effect_witness(inner, fnset) {
                    return Some(w);
                }
            }
            higher_order_effect_witness(tail, fnset)
        }
    }
}

// Built-in nullary types that need no user declaration but must pass the
// "unknown type" check in pass 2. `Tensor` is the opaque AI-runtime handle.
// Builtin function signatures and built-in type names come from the shared
// `crate::builtins` source of truth so typeck and interp cannot drift.
use crate::builtins::{lookup as builtin_sig, BUILTIN_TYPES};

/// Collect the distinct type-variable names appearing in a builtin's signature,
/// in order of first appearance. A builtin is generic over exactly these vars,
/// so the checker instantiates them fresh per call site (like a generic `fn`'s
/// `type_params`). Non-generic builtins return an empty list.
fn builtin_type_params(params: &[Ty], ret: &Ty) -> Vec<String> {
    fn walk(ty: &Ty, acc: &mut Vec<String>) {
        match ty {
            Ty::Var(n) => {
                if !acc.contains(n) {
                    acc.push(n.clone());
                }
            }
            Ty::Named(_, args) => args.iter().for_each(|a| walk(a, acc)),
            Ty::Fn(ps, r) => {
                ps.iter().for_each(|p| walk(p, acc));
                walk(r, acc);
            }
            _ => {}
        }
    }
    let mut acc = Vec::new();
    params.iter().for_each(|p| walk(p, &mut acc));
    walk(ret, &mut acc);
    acc
}

/// For a Map/Set builtin, the name of the signature type variable that carries
/// the (order-restricted) key / set-element type — `"K"` for `map_*`, `"T"` for
/// `set_*`. Returns `None` for any other builtin. The checker resolves this var
/// after argument unification and requires it to be Int or Str.
fn map_set_key_var(name: &str) -> Option<&'static str> {
    match name {
        "map_new" | "map_insert" | "map_get_or" | "map_has" | "map_len" | "map_remove"
        | "map_show" | "map_keys" | "map_values" => Some("K"),
        "set_new" | "set_add" | "set_has" | "set_len" | "set_remove" | "set_show"
        | "set_to_array" => Some("T"),
        _ => None,
    }
}

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

    /// Synthesize the type of `e`, recording the span of the innermost failing
    /// expression in `err_span` so a structured diagnostic can point at the exact
    /// sub-expression. The deepest failing node sets the span first; enclosing
    /// nodes do not overwrite it.
    fn synth(&self, e: &Expr, scope: &mut Scope) -> Result<Ty, String> {
        let r = self.synth_inner(e, scope);
        if r.is_err() {
            self.note_err_span(e.span);
        }
        r
    }

    /// Record `span` as the location of the current error, unless one is already
    /// set (keep the innermost) or `span` is the no-location sentinel.
    fn note_err_span(&self, span: Span) {
        if span.is_none() {
            return;
        }
        let mut slot = self.err_span.borrow_mut();
        if slot.is_none() {
            *slot = Some(span);
        }
    }

    fn synth_inner(&self, e: &Expr, scope: &mut Scope) -> Result<Ty, String> {
        match &e.kind {
            ExprKind::Int(_) => Ok(Ty::Int),
            ExprKind::Float(_) => Ok(Ty::Float),
            ExprKind::Bool(_) => Ok(Ty::Bool),
            ExprKind::Str(_) => Ok(Ty::Str),
            ExprKind::Unit => Ok(Ty::Unit),

            ExprKind::Var(name) => {
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

            ExprKind::Ctor(name, args) => {
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

            ExprKind::Record(name, fields) => {
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown record type `{}`", name))?;
                let decl_names = sig.field_names.clone().ok_or_else(|| {
                    format!("`{}` is not a record type (use `{}(..)` constructor syntax)", name, name)
                })?;
                // The provided field-name set must exactly match the declared one
                // (no missing, extra, or duplicate fields).
                for fname in &decl_names {
                    let count = fields.iter().filter(|(n, _)| n == fname).count();
                    if count == 0 {
                        return Err(format!("record `{}`: missing field `{}`", name, fname));
                    }
                    if count > 1 {
                        return Err(format!("record `{}`: duplicate field `{}`", name, fname));
                    }
                }
                for (n, _) in fields {
                    if !decl_names.contains(n) {
                        return Err(format!("record `{}` has no field `{}`", name, n));
                    }
                }
                // Instantiate the type's parameters, then check each named field's
                // value against its declared (substituted) type.
                let map = self.instantiate_map(&sig.type_params);
                for (fname, fty) in decl_names.iter().zip(sig.fields.iter()) {
                    let val = &fields.iter().find(|(n, _)| n == fname).unwrap().1;
                    let at = self.synth(val, scope)?;
                    let expected = Self::apply_map(fty, &map);
                    self.unify(&expected, &at)
                        .map_err(|e| format!("record `{}` field `{}`: {}", name, fname, e))?;
                }
                let type_args: Vec<Ty> =
                    sig.type_params.iter().map(|p| map.get(p).cloned().unwrap()).collect();
                Ok(Ty::Named(sig.tyname.clone(), type_args))
            }

            ExprKind::Field(obj, field) => {
                let ot = self.synth(obj, scope)?;
                let ot = self.prune(&ot);
                let (tyname, type_args) = match &ot {
                    Ty::Named(n, args) => (n.clone(), args.clone()),
                    _ => {
                        return Err(format!(
                            "field access `.{}` on a non-record value of type {}",
                            field,
                            show(&self.resolve(&ot))
                        ))
                    }
                };
                // The record type has a single constructor named after the type.
                let sig = self
                    .ctors
                    .get(&tyname)
                    .cloned()
                    .ok_or_else(|| format!("type `{}` is not a record", tyname))?;
                let decl_names = sig
                    .field_names
                    .clone()
                    .ok_or_else(|| format!("type `{}` is not a record", tyname))?;
                let idx = decl_names
                    .iter()
                    .position(|n| n == field)
                    .ok_or_else(|| format!("type `{}` has no field `{}`", tyname, field))?;
                // Substitute the object's concrete type arguments through the
                // declared field type (e.g. `Box[Int].value : Int`).
                let map: HashMap<String, Ty> =
                    sig.type_params.iter().cloned().zip(type_args).collect();
                Ok(Self::apply_map(&sig.fields[idx], &map))
            }

            ExprKind::Update(base, updates) => {
                let bt = self.synth(base, scope)?;
                let bt = self.prune(&bt);
                let (tyname, type_args) = match &bt {
                    Ty::Named(n, args) => (n.clone(), args.clone()),
                    _ => {
                        return Err(format!(
                            "record update on a non-record value of type {}",
                            show(&self.resolve(&bt))
                        ))
                    }
                };
                let sig = self
                    .ctors
                    .get(&tyname)
                    .cloned()
                    .ok_or_else(|| format!("type `{}` is not a record", tyname))?;
                let decl_names = sig
                    .field_names
                    .clone()
                    .ok_or_else(|| format!("type `{}` is not a record", tyname))?;
                let map: HashMap<String, Ty> =
                    sig.type_params.iter().cloned().zip(type_args).collect();
                for (i, (fname, val)) in updates.iter().enumerate() {
                    // Reject a field updated more than once (matches the record
                    // literal's duplicate-field rejection).
                    if updates[..i].iter().any(|(n, _)| n == fname) {
                        return Err(format!("record update: duplicate field `{}`", fname));
                    }
                    let idx = decl_names
                        .iter()
                        .position(|n| n == fname)
                        .ok_or_else(|| format!("type `{}` has no field `{}`", tyname, fname))?;
                    let at = self.synth(val, scope)?;
                    let expected = Self::apply_map(&sig.fields[idx], &map);
                    self.unify(&expected, &at)
                        .map_err(|e| format!("record update field `{}`: {}", fname, e))?;
                }
                // Update is type-preserving.
                Ok(bt)
            }

            ExprKind::Call(name, args) => {
                // A local binding shadowing a name (e.g. a function-valued
                // parameter `f`) is applied as a function VALUE, not a by-name
                // top-level call. This makes `f(x)` work inside a HOF.
                if let Some(local) = Checker::lookup_var(scope, name) {
                    return self.synth_apply(&local, args, scope, &format!("`{}`", name));
                }
                // Trait-method call (`show(p)`): the callee is a dispatcher. Type
                // it against the trait method signature, substituting `Self` with
                // the receiver's type — concrete (requiring an impl) or a bounded
                // rigid param (deferred to monomorphization). This keeps `Self`
                // rigid: parametricity is preserved and an unbounded `T` is
                // rejected.
                if let Some(mi) = self.traits.methods.get(name).cloned() {
                    return self.synth_trait_method(name, &mi, args, scope);
                }
                // `array_lit` is a variadic internal builtin (the desugaring of an
                // array literal `[e0, .., en]`). It is NOT in the signature table,
                // so handle it before the `builtin_sig`/`self.fns` lookup: every
                // element must share one element type `T`, and the result is
                // `Array[T]`.
                if name == "array_lit" {
                    let elem = self.fresh();
                    for (i, arg) in args.iter().enumerate() {
                        let at = self.synth(arg, scope)?;
                        self.unify(&elem, &at).map_err(|_| {
                            format!(
                                "array literal element {} has type {} but expected {}",
                                i,
                                show(&self.resolve(&at)),
                                show(&self.resolve(&elem))
                            )
                        })?;
                    }
                    return Ok(Ty::Named("Array".to_string(), vec![elem]));
                }
                let (params, ret, type_params, bounds) =
                    if let Some((p, r)) = builtin_sig(name) {
                        // A builtin whose signature mentions type variables
                        // (e.g. `array_get: (Array[T], Int) -> T`) is generic:
                        // collect those vars so they instantiate fresh per call.
                        let tps = builtin_type_params(&p, &r);
                        (p, r, tps, Vec::new())
                    } else if let Some(sig) = self.fns.get(name).cloned() {
                        (sig.params, sig.ret, sig.type_params, sig.bounds)
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
                // Non-lambda arguments first (to pin type parameters), then lambda
                // arguments bidirectionally — so a lambda's parameter types, fixed
                // by the function signature or a sibling argument, are in scope
                // before its body is checked.
                for (i, (arg, pt)) in args.iter().zip(params.iter()).enumerate() {
                    if !matches!(arg.kind, ExprKind::Lambda(..)) {
                        let at = self.synth(arg, scope)?;
                        let expected = Self::apply_map(pt, &map);
                        self.unify(&expected, &at).map_err(|e| {
                            format!("function `{}` argument {}: {}", name, i, e)
                        })?;
                    }
                }
                for (i, (arg, pt)) in args.iter().zip(params.iter()).enumerate() {
                    if matches!(arg.kind, ExprKind::Lambda(..)) {
                        let expected = Self::apply_map(pt, &map);
                        self.check(arg, &expected, scope).map_err(|e| {
                            format!("function `{}` argument {}: {}", name, i, e)
                        })?;
                    }
                }
                // Map/Set key & element types are restricted to the two
                // primitive totally-ordered types (Int, Str) so the ordered
                // representation has a deterministic, backend-agnostic order.
                // Resolve the relevant type variable now that the arguments have
                // pinned it; reject anything else (Float, Bool, ADT, Array, …)
                // with a clean error.
                if let Some(var) = map_set_key_var(name) {
                    if let Some(tv) = map.get(var) {
                        let kt = self.resolve(tv);
                        // A still-unresolved fresh variable (e.g. `map_new()`
                        // whose key type is pinned only by a later `insert`) is
                        // left to that later use to constrain; only reject a
                        // CONCRETE non-Int/Str key here.
                        let unresolved = matches!(&kt, Ty::Var(v) if is_fresh(v));
                        if !unresolved && !matches!(kt, Ty::Int | Ty::Str) {
                            let what = if name.starts_with("set_") {
                                "set element"
                            } else {
                                "map key"
                            };
                            return Err(format!(
                                "`{}`: {} type must be Int or Str (the totally-ordered \
                                 primitives), got {}",
                                name,
                                what,
                                show(&kt)
                            ));
                        }
                    }
                }
                // A `[T: Trait]`-bounded function may only be instantiated at a
                // type that has an `impl` of that trait — the obligation the bound
                // promises. Now that the arguments have pinned the type
                // parameters, check each bound against what `T` resolved to. A
                // concrete type must have an impl; a still-rigid type parameter
                // must itself carry the same bound in the enclosing function
                // (the obligation propagates); a still-unresolved fresh variable
                // (a phantom/unused parameter) is left to monomorphization.
                for (param, trait_name) in &bounds {
                    let Some(tv) = map.get(param) else { continue };
                    match self.resolve(tv) {
                        Ty::Named(h, _) => {
                            if !self.traits.has_impl(trait_name, &h) {
                                return Err(format!(
                                    "`{}` requires its type parameter `{}` to impl `{}`, but `{}` does not",
                                    name, param, trait_name, h
                                ));
                            }
                        }
                        Ty::Var(t) if !is_fresh(&t) => {
                            let ok = matches!(
                                self.cur_bounds.borrow().get(&t),
                                Some(b) if b == trait_name
                            );
                            if !ok {
                                return Err(format!(
                                    "`{}` requires its type parameter `{}` to impl `{}`, but `{}` is not bounded by `{}`",
                                    name, param, trait_name, t, trait_name
                                ));
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Self::apply_map(&ret, &map))
            }

            ExprKind::Unary(op, inner) => {
                let t = self.prune(&self.synth(inner, scope)?);
                match (op, &t) {
                    (UnOp::Neg, Ty::Int) => Ok(Ty::Int),
                    (UnOp::Neg, Ty::Float) => Ok(Ty::Float),
                    (UnOp::Not, Ty::Bool) => Ok(Ty::Bool),
                    _ => Err(format!("cannot apply {:?} to {}", op, show(&self.resolve(&t)))),
                }
            }

            ExprKind::Binary(op, lhs, rhs) => {
                let lt = self.synth(lhs, scope)?;
                let rt = self.synth(rhs, scope)?;
                self.synth_binary(*op, &lt, &rt)
            }

            ExprKind::If(cond, then, els) => {
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

            ExprKind::Match(scrut, arms) => self.synth_match(scrut, arms, scope),

            ExprKind::Lambda(params, body, _) => {
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
                        Ty::Var(v) if v.starts_with("$lam") => {
                            // Unannotated parameter: solve its type from context,
                            // recording the placeholder -> fresh-var link so it can
                            // be back-annotated into the AST after checking.
                            let fresh = self.fresh();
                            self.lam_vars.borrow_mut().push((v.clone(), fresh.clone()));
                            fresh
                        }
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

            ExprKind::Apply(callee, args, _) => {
                let ct = self.synth(callee, scope)?;
                self.synth_apply(&ct, args, scope, "value")
            }

            ExprKind::Block(stmts, last) => {
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
    /// Bidirectional check: verify `e` has the EXPECTED type, pushing that type
    /// inward. For a lambda checked against a function type this binds the
    /// parameters to the expected parameter types BEFORE checking the body, so a
    /// lambda whose parameter types are fixed only by the surrounding context
    /// (`apply2(\x -> \y -> x + y, ..)`) type-checks. Every other expression
    /// falls back to synthesis + unification.
    fn check(&self, e: &Expr, expected: &Ty, scope: &mut Scope) -> Result<(), String> {
        let r = self.check_inner(e, expected, scope);
        if r.is_err() {
            self.note_err_span(e.span);
        }
        r
    }

    fn check_inner(&self, e: &Expr, expected: &Ty, scope: &mut Scope) -> Result<(), String> {
        let exp = self.prune(expected);
        if let (ExprKind::Lambda(params, body, _), Ty::Fn(ep, er)) = (&e.kind, &exp) {
            if params.len() == ep.len() {
                let mut frame = HashMap::new();
                for ((n, t), pexp) in params.iter().zip(ep.iter()) {
                    // An unannotated parameter takes the expected type (recording
                    // the placeholder for AST back-annotation); an annotated one
                    // must unify with it.
                    let pt = match t {
                        Ty::Var(v) if v.starts_with("$lam") => {
                            let fresh = self.fresh();
                            self.lam_vars.borrow_mut().push((v.clone(), fresh.clone()));
                            fresh
                        }
                        other => other.clone(),
                    };
                    self.unify(&pt, pexp)?;
                    frame.insert(n.clone(), pt);
                }
                scope.push(frame);
                let r = self.check(body, er, scope);
                scope.pop();
                return r;
            }
        }
        let t = self.synth(e, scope)?;
        self.unify(&t, &exp)
    }

    /// Type a trait-method call `m(self, ..)` against the trait method signature
    /// `mi`, resolving `Self` from the receiver's type. `Self` is either a
    /// concrete head type with a required `impl`, or a rigid type parameter the
    /// enclosing function bounds by this trait (deferred to monomorphization).
    fn synth_trait_method(
        &self,
        name: &str,
        mi: &crate::traits::MethodInfo,
        args: &[Expr],
        scope: &mut Scope,
    ) -> Result<Ty, String> {
        if args.len() != mi.params.len() {
            return Err(format!(
                "trait method `{}` expects {} argument(s), got {}",
                name,
                mi.params.len(),
                args.len()
            ));
        }
        // Resolve `Self` from the receiver (first) argument.
        let recv_ty = self.resolve(&self.synth(&args[0], scope)?);
        let self_ty: Ty = match &recv_ty {
            Ty::Named(h, _) => {
                if !self.traits.has_impl(&mi.trait_name, h) {
                    return Err(format!(
                        "no impl of `{}` for `{}` (required by call to `{}`)",
                        mi.trait_name, h, name
                    ));
                }
                recv_ty.clone()
            }
            Ty::Var(t) if !is_fresh(t) => {
                // A rigid type parameter: it must be bounded by this trait in the
                // enclosing function, e.g. `fn f[T: Show](..)`. Otherwise calling
                // a trait method on an unconstrained `T` is unsound.
                match self.cur_bounds.borrow().get(t) {
                    Some(bound) if bound == &mi.trait_name => recv_ty.clone(),
                    _ => {
                        return Err(format!(
                            "type parameter `{}` is not bounded by `{}`; add `[{}: {}]` to call `{}` on it",
                            t, mi.trait_name, t, mi.trait_name, name
                        ))
                    }
                }
            }
            other => {
                return Err(format!(
                    "trait method `{}` of `{}` cannot be called on a value of type {}",
                    name,
                    mi.trait_name,
                    show(other)
                ))
            }
        };
        let map: HashMap<String, Ty> =
            std::iter::once((mi.self_param.clone(), self_ty)).collect();
        // Check each remaining argument against its (Self-substituted) param.
        for (i, (arg, pt)) in args.iter().zip(mi.params.iter()).enumerate().skip(1) {
            let at = self.synth(arg, scope)?;
            let expected = Self::apply_map(pt, &map);
            self.unify(&expected, &at)
                .map_err(|e| format!("trait method `{}` argument {}: {}", name, i, e))?;
        }
        Ok(Self::apply_map(&mi.ret, &map))
    }

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
        // Check non-lambda arguments first (they pin down any type parameters),
        // then lambda arguments bidirectionally against their now-resolved
        // expected types — so a lambda whose parameter types come from the
        // function's signature (or a sibling argument) is checked with those
        // types already in scope.
        for (i, (arg, pv)) in args.iter().zip(arg_vars.iter()).enumerate() {
            if !matches!(arg.kind, ExprKind::Lambda(..)) {
                let at = self.synth(arg, scope)?;
                self.unify(pv, &at)
                    .map_err(|e| format!("applying {} argument {}: {}", what, i, e))?;
            }
        }
        for (i, (arg, pv)) in args.iter().zip(arg_vars.iter()).enumerate() {
            if matches!(arg.kind, ExprKind::Lambda(..)) {
                self.check(arg, pv, scope)
                    .map_err(|e| format!("applying {} argument {}: {}", what, i, e))?;
            }
        }
        Ok(self.resolve(&ret_var))
    }

    fn synth_match(&self, scrut: &Expr, arms: &[Arm], scope: &mut Scope) -> Result<Ty, String> {
        let s = self.synth(scrut, scope)?;
        let mut result: Option<Ty> = None;

        for arm in arms {
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

        // Exhaustiveness via a recursive "matrix" usefulness check. This is
        // sound for *nested* patterns: a constructor arm only covers its
        // constructor when its sub-patterns are themselves exhaustive. We build
        // a one-column matrix (one row per arm) typed by the scrutinee type and
        // ask whether a wildcard row would still be useful — i.e. whether some
        // value escapes every arm. If so, we synthesize a witness for the error.
        let rows: Vec<Vec<Pattern>> = arms.iter().map(|a| vec![a.pat.clone()]).collect();
        let col_ty = self.resolve(&s);
        if let Some(witness) = self.missing_witness(&rows, &[col_ty.clone()]) {
            let wstr = witness
                .first()
                .map(Self::show_witness)
                .unwrap_or_else(|| "_".to_string());
            return Err(format!(
                "non-exhaustive match on {}: missing case `{}`",
                show(&col_ty),
                wstr
            ));
        }

        result
            .map(|t| self.resolve(&t))
            .ok_or_else(|| "match needs at least one arm".to_string())
    }

    /// The closed constructor space of a type, for exhaustiveness checking.
    /// Returns `Some(vec)` of `(ctor_name, field_types)` when the type has a
    /// *finite, known* set of constructors (a user ADT, or Bool encoded as the
    /// pseudo-ctors `true`/`false`). Field types are instantiated for the given
    /// type arguments. Returns `None` for "open" types whose value space is
    /// effectively infinite or unknown (Int, Float, Str, type variables,
    /// function types, builtin generics like List/Array/Map without variant
    /// constructors) — such columns are only covered by a wildcard.
    fn ctor_space(&self, ty: &Ty) -> Option<Vec<(String, Vec<Ty>)>> {
        match self.prune(ty) {
            Ty::Bool => Some(vec![
                ("true".to_string(), vec![]),
                ("false".to_string(), vec![]),
            ]),
            Ty::Named(tn, args) => {
                let (params, variants) = self.types.get(&tn)?;
                if variants.is_empty() {
                    return None;
                }
                let map: HashMap<String, Ty> = params
                    .iter()
                    .cloned()
                    .zip(args.iter().cloned())
                    .collect();
                let mut out = Vec::new();
                for cname in variants {
                    let sig = self.ctors.get(cname)?;
                    let fields = sig
                        .fields
                        .iter()
                        .map(|f| Self::apply_map(f, &map))
                        .collect();
                    out.push((cname.clone(), fields));
                }
                Some(out)
            }
            _ => None,
        }
    }

    /// The "head constructor" of a pattern for the matrix algorithm, or `None`
    /// for wildcard/variable patterns (which match any value). Record patterns
    /// are normalized to positional `Ctor` form so they share the ADT logic.
    fn pat_head(&self, pat: &Pattern) -> Option<(String, Vec<Pattern>)> {
        match pat {
            Pattern::Wild | Pattern::Var(_) => None,
            Pattern::Bool(b) => {
                Some((if *b { "true" } else { "false" }.to_string(), vec![]))
            }
            Pattern::Int(_) => Some((format!("{:?}", pat), vec![])),
            Pattern::Ctor(name, subs) => Some((name.clone(), subs.clone())),
            Pattern::Record(name, sub_fields) => {
                // Reorder named sub-patterns into declared field order, filling
                // omitted fields with wildcards.
                let decl = self
                    .ctors
                    .get(name)
                    .and_then(|s| s.field_names.clone())
                    .unwrap_or_default();
                let subs = decl
                    .iter()
                    .map(|fname| {
                        sub_fields
                            .iter()
                            .find(|(n, _)| n == fname)
                            .map(|(_, p)| p.clone())
                            .unwrap_or(Pattern::Wild)
                    })
                    .collect();
                Some((name.clone(), subs))
            }
        }
    }

    /// Recursive exhaustiveness: given a pattern `matrix` (each row has one
    /// pattern per column) and the `col_types` of those columns, return
    /// `Some(witness)` describing a value matched by no row, or `None` if the
    /// matrix is exhaustive. The standard usefulness algorithm (Maranget).
    fn missing_witness(
        &self,
        matrix: &[Vec<Pattern>],
        col_types: &[Ty],
    ) -> Option<Vec<Witness>> {
        // Base case: no columns left.
        if col_types.is_empty() {
            // A value reaches here iff some row exists (matches everything);
            // if no rows remain, the value is unmatched.
            return if matrix.is_empty() { Some(vec![]) } else { None };
        }

        let head_ty = &col_types[0];
        let rest_types = &col_types[1..];

        match self.ctor_space(head_ty) {
            // Closed, finite constructor space (ADT or Bool).
            Some(space) => {
                // The constructors that appear as a *concrete* pattern head in
                // this column (wildcards/vars are not roots — they are handled
                // by the default matrix).
                let present: std::collections::HashSet<String> = matrix
                    .iter()
                    .filter_map(|row| self.pat_head(&row[0]).map(|(n, _)| n))
                    .collect();
                let complete = space.iter().all(|(c, _)| present.contains(c));

                if complete {
                    // Complete signature: a value is unmatched iff it is
                    // unmatched under some specific constructor. Recurse per
                    // constructor (this terminates: each `specialize` is driven
                    // by the finite set of concrete sub-patterns).
                    for (cname, fields) in &space {
                        let spec = self.specialize(matrix, cname, fields.len());
                        let mut sub_types = fields.clone();
                        sub_types.extend_from_slice(rest_types);
                        if let Some(w) = self.missing_witness(&spec, &sub_types) {
                            let (sub, tail) = w.split_at(fields.len());
                            let head_w = Witness::Ctor(cname.clone(), sub.to_vec());
                            let mut out = vec![head_w];
                            out.extend_from_slice(tail);
                            return Some(out);
                        }
                    }
                    None
                } else {
                    // Incomplete signature: the column is exhaustive only via
                    // the *default* matrix (rows whose head is a wildcard). If
                    // the default matrix is non-exhaustive, build a witness whose
                    // head is a missing constructor (so the error names a real
                    // missing case) when one exists; otherwise a wildcard. This
                    // recursion drops a column and only keeps wildcard rows, so
                    // it always terminates — even on recursive types.
                    let default = self.default_matrix(matrix);
                    let tail = self.missing_witness(&default, rest_types)?;
                    let head_w = match space.iter().find(|(c, _)| !present.contains(c)) {
                        Some((cname, fields)) => Witness::Ctor(
                            cname.clone(),
                            fields.iter().map(|_| Witness::Wild).collect(),
                        ),
                        None => Witness::Wild,
                    };
                    let mut out = vec![head_w];
                    out.extend(tail);
                    Some(out)
                }
            }
            // Open / infinite column: only a wildcard row covers everything.
            None => {
                let has_wild = matrix
                    .iter()
                    .any(|row| self.pat_head(&row[0]).is_none());
                let default = self.default_matrix(matrix);
                let tail = self.missing_witness(&default, rest_types);
                if has_wild {
                    // Wildcards present: the column is covered for those rows;
                    // exhaustiveness depends purely on the remaining columns of
                    // the default matrix.
                    tail.map(|t| {
                        let mut out = vec![Witness::Wild];
                        out.extend(t);
                        out
                    })
                } else {
                    // No wildcard row: some value of this open column escapes.
                    Some(extend_witness(Witness::Wild, tail, rest_types.len()))
                }
            }
        }
    }

    /// Specialize the matrix to constructor `cname` with `arity` fields: keep
    /// rows whose head is `cname` (expanding sub-patterns into new leading
    /// columns) or a wildcard (expanding into `arity` wildcards).
    fn specialize(
        &self,
        matrix: &[Vec<Pattern>],
        cname: &str,
        arity: usize,
    ) -> Vec<Vec<Pattern>> {
        let mut out = Vec::new();
        for row in matrix {
            match self.pat_head(&row[0]) {
                Some((n, subs)) if n == cname => {
                    let mut new_row = subs;
                    new_row.extend_from_slice(&row[1..]);
                    out.push(new_row);
                }
                Some(_) => {} // different constructor: drop
                None => {
                    // wildcard/var head: expands to `arity` wildcards
                    let mut new_row = vec![Pattern::Wild; arity];
                    new_row.extend_from_slice(&row[1..]);
                    out.push(new_row);
                }
            }
        }
        out
    }

    /// The default matrix: keep only rows whose first pattern is a wildcard or
    /// variable, dropping that first column.
    fn default_matrix(&self, matrix: &[Vec<Pattern>]) -> Vec<Vec<Pattern>> {
        matrix
            .iter()
            .filter(|row| self.pat_head(&row[0]).is_none())
            .map(|row| row[1..].to_vec())
            .collect()
    }

    /// Render a witness value as readable source-like text for error messages.
    fn show_witness(w: &Witness) -> String {
        match w {
            Witness::Wild => "_".to_string(),
            Witness::Ctor(name, subs) => {
                if subs.is_empty() {
                    name.clone()
                } else {
                    format!(
                        "{}({})",
                        name,
                        subs.iter().map(Self::show_witness).collect::<Vec<_>>().join(", ")
                    )
                }
            }
        }
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
                    // Tuple patterns are synthetic ctors; phrase the error in
                    // tuple terms instead of leaking `$TupleN`.
                    if name.starts_with("$Tuple") {
                        format!(
                            "a {}-element tuple pattern cannot match {}",
                            subs.len(),
                            show(&self.resolve(expected))
                        )
                    } else {
                        format!(
                            "constructor pattern `{}` (of type {}) matched against {}",
                            name,
                            sig.tyname,
                            show(&self.resolve(expected))
                        )
                    }
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
            Pattern::Record(name, sub_fields) => {
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown record type `{}`", name))?;
                let decl_names = sig.field_names.clone().ok_or_else(|| {
                    format!("`{}` is not a record type", name)
                })?;
                let map = self.instantiate_map(&sig.type_params);
                let type_args: Vec<Ty> =
                    sig.type_params.iter().map(|p| map.get(p).cloned().unwrap()).collect();
                let owner = Ty::Named(sig.tyname.clone(), type_args);
                self.unify(&owner, expected).map_err(|_| {
                    format!(
                        "record pattern `{}` matched against {}",
                        name,
                        show(&self.resolve(expected))
                    )
                })?;
                for (fname, sp) in sub_fields {
                    let idx = decl_names
                        .iter()
                        .position(|n| n == fname)
                        .ok_or_else(|| format!("record `{}` has no field `{}`", name, fname))?;
                    let fty = Self::apply_map(&sig.fields[idx], &map);
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

    fn structured_src(src: &str) -> Vec<crate::diagnostics::Diagnostic> {
        let toks = lexer::lex(src).expect("lex");
        let prog = parser::parse(toks).expect("parse");
        check_structured(&prog)
    }

    #[test]
    fn structured_clean_program_is_empty_array() {
        let diags = structured_src("fn main() -> Int = 42");
        assert!(diags.is_empty());
        assert_eq!(crate::diagnostics::array_to_json(&diags), "[]");
    }

    #[test]
    fn structured_routes_phases_and_codes() {
        // Exhaustiveness.
        let d = structured_src(
            "type C = | R | G | B\nfn f(c: C) -> Int = match c { R => 0, G => 1 }\nfn main() -> Int = f(R)",
        );
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].phase, "exhaustiveness");
        assert_eq!(d[0].code, "E0203");
        assert_eq!(d[0].function.as_deref(), Some("f"));

        // Type mismatch.
        let d = structured_src("fn wrong() -> Int = true\nfn main() -> Int = 0");
        assert!(d.iter().any(|x| x.phase == "type"
            && x.code == "E0201"
            && x.function.as_deref() == Some("wrong")));

        // Unknown variable.
        let d = structured_src("fn main() -> Int = x");
        assert_eq!(d[0].phase, "type");
        assert_eq!(d[0].code, "E0200");

        // Purity (higher-order witness).
        let d = structured_src(
            "pure fn run(f: (Int) -> Int, x: Int) -> Int = f(x)\nfn main() -> Int = run(\\y -> y, 3)",
        );
        assert_eq!(d[0].phase, "purity");
        assert_eq!(d[0].code, "E0210");
        assert_eq!(d[0].function.as_deref(), Some("run"));

        // Tensor shape mismatch.
        let d = structured_src(
            "fn main() -> Tensor = matmul(tensor_zeros(2,3), tensor_zeros(4,5))",
        );
        assert_eq!(d[0].phase, "shape");
        assert_eq!(d[0].code, "E0300");
    }

    #[test]
    fn structured_array_is_valid_json_shape() {
        let d = structured_src(
            "fn wrong() -> Int = true\nfn bad(n: Int) -> Bool = n == \"five\"\nfn main() -> Int = 0",
        );
        let json = crate::diagnostics::array_to_json(&d);
        // Two diagnostics, comma-separated, no trailing comma, balanced brackets.
        assert!(json.starts_with("[{") && json.ends_with("}]"));
        assert!(json.contains("},{"));
        assert!(json.contains("\"code\":\"E0201\""));
        assert!(json.contains("\"phase\":\"type\""));
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
    fn bidirectional_lambda_argument_checking() {
        // A nested lambda with NO internal type hint, whose parameter types are
        // fixed only by the callee's signature, type-checks because the expected
        // function type is pushed into the lambda before its body is checked.
        assert!(check_src(
            "fn apply2(f: (Int) -> (Int) -> Int, a: Int, b: Int) -> Int = f(a)(b)\n\
             fn main() -> Int = apply2(\\x -> \\y -> x + y, 30, 12)"
        )
        .is_ok());
        // A higher-order lambda whose Int-ness comes from a SIBLING argument
        // (the list element type): non-lambda args are checked first.
        assert!(check_src(
            "type List = | Nil | Cons(Int, List)\n\
             fn map(f: (Int) -> Int, xs: List) -> List = match xs { Nil => Nil, Cons(h, t) => Cons(f(h), map(f, t)), }\n\
             fn main() -> List = map(\\x -> x + x, Cons(1, Nil))"
        )
        .is_ok());
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

    // --- BUG A: nested-pattern exhaustiveness (soundness) ---

    #[test]
    fn nested_non_exhaustive_match_rejected() {
        // Previously accepted as exhaustive because only the TOP-level ctor
        // `Wrap` was counted; `Wrap(F)` then crashed the interpreter at runtime.
        let src = r#"
            type B = | T | F
            type W = | Wrap(B)
            fn f(w: W) -> Int = match w { Wrap(T) => 1, }
            fn main() -> Int = f(Wrap(F))
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("non-exhaustive") && e.contains("Wrap(F)")),
            "expected a non-exhaustive error naming Wrap(F), got: {:?}",
            errs
        );
    }

    #[test]
    fn nested_non_exhaustive_two_field_rejected() {
        let src = r#"
            type B = | T | F
            type W = | Wrap(B, B)
            fn f(w: W) -> Int = match w { Wrap(T, T) => 1, Wrap(F, _) => 3 }
            fn main() -> Int = f(Wrap(F, T))
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("non-exhaustive") && e.contains("Wrap(T, F)")),
            "expected missing Wrap(T, F), got: {:?}",
            errs
        );
    }

    #[test]
    fn nested_recursive_non_exhaustive_rejected() {
        let src = r#"
            type Tree = | Leaf | Node(Tree, Tree)
            fn f(t: Tree) -> Int = match t { Node(Leaf, _) => 1, Node(_, _) => 2 }
            fn main() -> Int = f(Leaf)
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("non-exhaustive") && e.contains("Leaf")),
            "expected missing Leaf, got: {:?}",
            errs
        );
    }

    #[test]
    fn valid_nested_matches_still_accepted() {
        // Fully enumerated nested ctors.
        assert!(check_src(
            "type B = | T | F\n\
             type W = | Wrap(B)\n\
             fn f(w: W) -> Int = match w { Wrap(T) => 1, Wrap(F) => 2 }\n\
             fn main() -> Int = f(Wrap(F))"
        )
        .is_ok());
        // Wildcard sub-pattern covers all of B.
        assert!(check_src(
            "type B = | T | F\n\
             type W = | Wrap(B)\n\
             fn f(w: W) -> Int = match w { Wrap(_) => 1 }\n\
             fn main() -> Int = f(Wrap(F))"
        )
        .is_ok());
        // Two-field nested coverage with a wildcard tail.
        assert!(check_src(
            "type B = | T | F\n\
             type W = | Wrap(B, B)\n\
             fn f(w: W) -> Int = match w { Wrap(T, T) => 1, Wrap(T, F) => 2, Wrap(F, _) => 3 }\n\
             fn main() -> Int = f(Wrap(F, T))"
        )
        .is_ok());
        // Recursive type, exhaustively covered.
        assert!(check_src(
            "type Tree = | Leaf | Node(Tree, Tree)\n\
             fn f(t: Tree) -> Int = match t { Leaf => 0, Node(Leaf, _) => 1, Node(_, _) => 2 }\n\
             fn main() -> Int = f(Leaf)"
        )
        .is_ok());
    }

    // --- BUG E: phantom/unused type parameters (backend parity) ---

    #[test]
    fn unused_type_parameter_rejected() {
        // `U` appears in no parameter type, so it can't be inferred at a call
        // site — the backends reject it at monomorphization; typeck must too.
        let src = "fn f[T, U](x: T) -> T = x\nfn main() -> Int = f(5)";
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("type parameter `U`")
                && e.contains("unused")
                && e.contains("`f`")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn main_with_type_parameters_rejected() {
        let src = "fn main[T]() -> Int = 5";
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("`main` must take no type parameters")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn used_type_parameter_still_ok() {
        // Every type parameter appears in a parameter type — inferable, accepted.
        assert!(check_src("fn id[T](x: T) -> T = x\nfn main() -> Int = id(5)").is_ok());
        assert!(check_src(
            "fn pair[A, B](a: A, b: B) -> A = a\nfn main() -> Int = pair(1, 2)"
        )
        .is_ok());
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

    // ---- Bytes builtin type & signatures --------------------------------

    #[test]
    fn bytes_is_a_builtin_type_with_correct_signatures() {
        use crate::ast::Ty::*;
        use crate::builtins;
        // `Bytes` is a declared builtin (opaque) type name (nullary).
        assert!(builtins::BUILTIN_TYPES.contains(&"Bytes"));
        let bytes = Named("Bytes".to_string(), vec![]);
        // Each bytes builtin has the expected (params, ret) signature.
        let cases: &[(&str, Vec<Ty>, Ty)] = &[
            ("bytes_new", vec![], bytes.clone()),
            ("bytes_len", vec![bytes.clone()], Int),
            ("bytes_get", vec![bytes.clone(), Int], Int),
            ("bytes_set", vec![bytes.clone(), Int, Int], bytes.clone()),
            ("bytes_push", vec![bytes.clone(), Int], bytes.clone()),
            ("bytes_from_str", vec![Str], bytes.clone()),
            ("bytes_to_str", vec![bytes.clone()], Str),
        ];
        for (name, params, ret) in cases {
            let (p, r) = builtins::lookup(name)
                .unwrap_or_else(|| panic!("builtin `{}` missing from signature table", name));
            assert_eq!(&p, params, "param mismatch for `{}`", name);
            assert_eq!(&r, ret, "return mismatch for `{}`", name);
        }
    }

    #[test]
    fn bytes_program_type_checks_and_bytes_neq_str_is_rejected() {
        // A representative Bytes program type-checks.
        let ok = r#"
            fn main() -> Int = {
                let b = bytes_set(bytes_push(bytes_from_str("hi"), 33), 0, 104);
                bytes_len(b) + bytes_get(b, 0)
            }
        "#;
        assert!(check_src(ok).is_ok(), "got: {:?}", check_src(ok));

        // `Bytes` and `Str` are distinct types, so comparing them is a type error
        // (the runtime distinct-tag invariant is also enforced at the type level).
        let bad = r#"fn main() -> Bool = bytes_from_str("x") == "x""#;
        let errs = check_src(bad).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("cannot compare")),
            "expected a `cannot compare` error, got: {:?}",
            errs
        );
    }

    // ---- Vector builtin type & signatures -------------------------------

    #[test]
    fn vector_is_a_builtin_type_with_correct_signatures() {
        use crate::ast::Ty::*;
        use crate::builtins;
        // `Vector` is a declared builtin (opaque) type name (nullary).
        assert!(builtins::BUILTIN_TYPES.contains(&"Vector"));
        let vector = Named("Vector".to_string(), vec![]);
        let float_array = Named("Array".to_string(), vec![Float]);
        // Each vector builtin has the expected (params, ret) signature.
        let cases: &[(&str, Vec<Ty>, Ty)] = &[
            ("vec_new", vec![], vector.clone()),
            ("vec_from_array", vec![float_array.clone()], vector.clone()),
            ("vec_to_array", vec![vector.clone()], float_array.clone()),
            ("vec_len", vec![vector.clone()], Int),
            ("vec_get", vec![vector.clone(), Int], Float),
            ("vec_push", vec![vector.clone(), Float], vector.clone()),
            ("vec_dot", vec![vector.clone(), vector.clone()], Float),
            ("vec_norm", vec![vector.clone()], Float),
            ("vec_cosine", vec![vector.clone(), vector.clone()], Float),
            ("vec_add", vec![vector.clone(), vector.clone()], vector.clone()),
            ("vec_scale", vec![vector.clone(), Float], vector.clone()),
        ];
        for (name, params, ret) in cases {
            let (p, r) = builtins::lookup(name)
                .unwrap_or_else(|| panic!("builtin `{}` missing from signature table", name));
            assert_eq!(&p, params, "param mismatch for `{}`", name);
            assert_eq!(&r, ret, "return mismatch for `{}`", name);
        }
    }

    #[test]
    fn vector_program_type_checks() {
        // A representative Vector program type-checks: build via from_array and
        // push, compute dot/cosine/norm (Float).
        let ok = r#"
            fn main() -> Float = {
                let a = vec_from_array([1.0, 2.0, 3.0]);
                let b = vec_push(vec_new(), 4.0);
                vec_dot(a, a) + vec_norm(a) + vec_cosine(a, b)
            }
        "#;
        assert!(check_src(ok).is_ok(), "got: {:?}", check_src(ok));

        // `vec_to_array` round-trips to an Array[Float]; indexing yields Float.
        let ok2 = r#"
            fn main() -> Float = {
                let a = vec_from_array([5.0, 6.0]);
                let xs = vec_to_array(a);
                xs[0]
            }
        "#;
        assert!(check_src(ok2).is_ok(), "got: {:?}", check_src(ok2));
    }

    #[test]
    fn vec_from_array_requires_float_array_not_int_array() {
        // `vec_from_array` requires `Array[Float]`; an `Array[Int]` is a type
        // error (the float-array parameter type does not unify with Array[Int]).
        let bad = r#"fn main() -> Vector = vec_from_array([1, 2, 3])"#;
        let errs = check_src(bad).unwrap_err();
        assert!(
            !errs.is_empty(),
            "expected a type error for vec_from_array(Array[Int]), got OK"
        );
    }

    #[test]
    fn vector_neq_array_float_is_rejected() {
        // A `Vector` and an `Array[Float]` are distinct types, so comparing them
        // is a type error (the runtime distinct-tag invariant is enforced at the
        // type level too).
        let bad = r#"fn main() -> Bool = vec_from_array([1.0]) == [1.0]"#;
        let errs = check_src(bad).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("cannot compare") || e.contains("compare")),
            "expected a compare error, got: {:?}",
            errs
        );
    }

    // ---- Map / Set builtin types, signatures & key restriction ----------

    #[test]
    fn map_set_are_builtin_types_with_correct_signatures() {
        use crate::ast::Ty::*;
        use crate::builtins;
        assert!(builtins::BUILTIN_TYPES.contains(&"Map"));
        assert!(builtins::BUILTIN_TYPES.contains(&"Set"));
        let map = Named("Map".to_string(), vec![Var("K".into()), Var("V".into())]);
        let set = Named("Set".to_string(), vec![Var("T".into())]);
        let cases: &[(&str, Vec<Ty>, Ty)] = &[
            ("map_new", vec![], map.clone()),
            ("map_insert", vec![map.clone(), Var("K".into()), Var("V".into())], map.clone()),
            ("map_get_or", vec![map.clone(), Var("K".into()), Var("V".into())], Var("V".into())),
            ("map_has", vec![map.clone(), Var("K".into())], Bool),
            ("map_len", vec![map.clone()], Int),
            ("map_remove", vec![map.clone(), Var("K".into())], map.clone()),
            ("set_new", vec![], set.clone()),
            ("set_add", vec![set.clone(), Var("T".into())], set.clone()),
            ("set_has", vec![set.clone(), Var("T".into())], Bool),
            ("set_len", vec![set.clone()], Int),
            ("set_remove", vec![set.clone(), Var("T".into())], set.clone()),
            // Enumeration builtins return a plain Array of the key/value/element.
            ("map_keys", vec![map.clone()], Named("Array".into(), vec![Var("K".into())])),
            ("map_values", vec![map.clone()], Named("Array".into(), vec![Var("V".into())])),
            ("set_to_array", vec![set.clone()], Named("Array".into(), vec![Var("T".into())])),
        ];
        for (name, params, ret) in cases {
            let (p, r) = builtins::lookup(name)
                .unwrap_or_else(|| panic!("builtin `{}` missing from signature table", name));
            assert_eq!(&p, params, "param mismatch for `{}`", name);
            assert_eq!(&r, ret, "return mismatch for `{}`", name);
        }
    }

    #[test]
    fn map_set_programs_type_check_with_int_and_str_keys() {
        // Int-keyed map, including `map_get_or`'s default matching V.
        let ok_int = r#"
            fn main() -> Int = {
                let m = map_insert(map_insert(map_new(), 1, 10), 2, 20);
                map_get_or(m, 1, 0) + map_len(m)
            }
        "#;
        assert!(check_src(ok_int).is_ok(), "got: {:?}", check_src(ok_int));
        // Str-keyed map.
        let ok_str = r#"
            fn main() -> Int = map_get_or(map_insert(map_new(), "a", 7), "a", 0)
        "#;
        assert!(check_src(ok_str).is_ok(), "got: {:?}", check_src(ok_str));
        // Set of Str.
        let ok_set = r#"
            fn main() -> Bool = set_has(set_add(set_new(), "x"), "x")
        "#;
        assert!(check_src(ok_set).is_ok(), "got: {:?}", check_src(ok_set));
    }

    #[test]
    fn enumeration_builtins_type_check() {
        // map_keys yields Array[K] (Int), map_values Array[V] (here also Int),
        // both usable with the array primitives.
        let ok_int = r#"
            fn main() -> Int = {
                let m = map_insert(map_insert(map_new(), 1, 10), 2, 20);
                array_len(map_keys(m)) + array_get(map_values(m), 0)
            }
        "#;
        assert!(check_src(ok_int).is_ok(), "got: {:?}", check_src(ok_int));
        // map_values preserves a non-key value type (Float) into Array[Float].
        let ok_fval = r#"
            fn main() -> Float =
                array_get(map_values(map_insert(map_new(), 1, 2.5)), 0)
        "#;
        assert!(check_src(ok_fval).is_ok(), "got: {:?}", check_src(ok_fval));
        // Str keys -> Array[String]; element used where a String is expected.
        let ok_skeys = r#"
            fn main() -> String =
                array_get(map_keys(map_insert(map_new(), "a", 1)), 0)
        "#;
        assert!(check_src(ok_skeys).is_ok(), "got: {:?}", check_src(ok_skeys));
        // set_to_array yields Array[T].
        let ok_set = r#"
            fn main() -> Int = array_len(set_to_array(set_add(set_new(), 3)))
        "#;
        assert!(check_src(ok_set).is_ok(), "got: {:?}", check_src(ok_set));
    }

    #[test]
    fn enumeration_result_element_type_is_checked() {
        // map_keys of an Int-keyed map is Array[Int]; treating an element as a
        // String must be a type error (proves the element type is threaded).
        let bad = r#"
            fn main() -> String =
                array_get(map_keys(map_insert(map_new(), 1, 10)), 0)
        "#;
        assert!(check_src(bad).is_err(), "expected an element-type error");
        // A non-Int/Str key is still rejected through the enumeration builtin's
        // Map argument (the key restriction is inherited).
        let bad_key = r#"
            fn main() -> Int = array_len(map_keys(map_insert(map_new(), true, 1)))
        "#;
        let errs = check_src(bad_key).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("must be Int or Str")),
            "expected a key-restriction error, got: {:?}",
            errs
        );
    }

    #[test]
    fn non_int_str_key_is_rejected() {
        // A Bool key (a `Map[Bool, Int]`) is rejected: keys must be Int or Str.
        let bad_bool = r#"fn main() -> Bool = map_has(map_insert(map_new(), true, 1), true)"#;
        let errs = check_src(bad_bool).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("must be Int or Str")),
            "expected a key-restriction error, got: {:?}",
            errs
        );
        // A Float set element is likewise rejected.
        let bad_float = r#"fn main() -> Bool = set_has(set_add(set_new(), 1.5), 1.5)"#;
        let errs = check_src(bad_float).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("must be Int or Str")),
            "expected a set-element-restriction error, got: {:?}",
            errs
        );
        // An ADT-keyed map is rejected.
        let bad_adt = r#"
            type K = | A | B
            fn main() -> Bool = map_has(map_insert(map_new(), A, 1), A)
        "#;
        let errs = check_src(bad_adt).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("must be Int or Str")),
            "expected an ADT-key-restriction error, got: {:?}",
            errs
        );
    }

    #[test]
    fn map_get_or_default_must_match_value_type() {
        // The default (3rd arg) must have the map's value type V. A Str default
        // for an `Int`-valued map is a type error.
        let bad = r#"
            fn main() -> Int = map_get_or(map_insert(map_new(), 1, 10), 1, "oops")
        "#;
        assert!(check_src(bad).is_err(), "expected a value-type mismatch error");
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

    // --- BUG D: `pure` must not be preserved under higher-order calls ---

    #[test]
    fn pure_applying_function_value_is_rejected() {
        // `run` is declared `pure` but applies its function PARAMETER `f`, whose
        // effects are unknown (the caller passes an IO-performing lambda). Aria
        // has no effect polymorphism, so this `pure` claim is unsound.
        let src = "pure fn run(f: (Int) -> Int, x: Int) -> Int = f(x)";
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("cannot prove `run` pure")
                && e.contains("function value `f`")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn pure_applying_let_bound_closure_is_rejected() {
        // Even a `let`-bound closure has unknown effects when applied (Aria does
        // not yet prove let-bound closure purity), so be conservatively sound.
        let src = "pure fn run(x: Int) -> Int = { let g = \\y -> y + 1; g(x) }";
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("cannot prove `run` pure")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn pure_hof_free_function_still_ok() {
        // A genuinely pure function (no applied function values, only arithmetic
        // and a pure top-level call / pure builtin) must still pass.
        assert!(check_src(
            "pure fn add(a: Int, b: Int) -> Int = a + b\n\
             pure fn area(w: Float, h: Float) -> Float = w * h\n\
             pure fn use_them(a: Int, b: Int) -> Int = add(a, b)\n\
             pure fn len(xs: Array[Int]) -> Int = array_len(xs)"
        )
        .is_ok());
    }

    #[test]
    fn non_pure_hof_is_fine() {
        // The same higher-order function WITHOUT the `pure` annotation is
        // perfectly legal — we only reject the unsound `pure` claim.
        let src = "fn run(f: (Int) -> Int, x: Int) -> Int = f(x)\n\
             fn main() -> Int = { print_int(run(\\x -> { print_int(x); x }, 5)); 0 }";
        assert!(check_src(src).is_ok(), "got: {:?}", check_src(src));
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

    // ---- traits / interfaces --------------------------------------------

    #[test]
    fn trait_impl_and_bounded_call_check() {
        // An interface, two impls (an ADT + a record), and a bounded generic
        // function calling the trait method all type-check.
        let src = r#"
            interface Describe[T] { fn code(self: T) -> Int }
            type Shape = | Circle | Square
            impl Describe for Shape { fn code(self: Shape) -> Int = match self { Circle => 1, Square => 2, } }
            type Point = { x: Int, y: Int }
            impl Describe for Point { fn code(self: Point) -> Int = self.x + self.y }
            fn twice[T: Describe](v: T) -> Int = code(v) + code(v)
            fn main() -> Int = { print_int(code(Circle)); print_int(twice(Point { x: 1, y: 2 })); 0 }
        "#;
        assert!(check_src(src).is_ok(), "got: {:?}", check_src(src));
    }

    #[test]
    fn trait_missing_impl_method_caught() {
        let src = r#"
            interface Describe[T] { fn code(self: T) -> Int }
            type Shape = | Circle
            impl Describe for Shape { }
            fn main() -> Int = 0
        "#;
        // Lowering rejects this at parse time with a clean message.
        let err = lexer::lex(src).and_then(parser::parse).unwrap_err();
        assert!(err.contains("missing method") && err.contains("code"), "got: {}", err);
    }

    #[test]
    fn trait_duplicate_impl_caught() {
        let src = r#"
            interface Describe[T] { fn code(self: T) -> Int }
            type Shape = | Circle
            impl Describe for Shape { fn code(self: Shape) -> Int = 1 }
            impl Describe for Shape { fn code(self: Shape) -> Int = 2 }
            fn main() -> Int = 0
        "#;
        let err = lexer::lex(src).and_then(parser::parse).unwrap_err();
        assert!(err.contains("duplicate impl"), "got: {}", err);
    }

    #[test]
    fn trait_call_on_type_without_impl_caught() {
        let src = r#"
            interface Describe[T] { fn code(self: T) -> Int }
            type Shape = | Circle
            type Color = | Red
            impl Describe for Shape { fn code(self: Shape) -> Int = 1 }
            fn main() -> Int = { print_int(code(Red)); 0 }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("no impl of `Describe` for `Color`")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn trait_method_on_unbounded_param_caught() {
        let src = r#"
            interface Describe[T] { fn code(self: T) -> Int }
            type Shape = | Circle
            impl Describe for Shape { fn code(self: Shape) -> Int = 1 }
            fn bad[T](v: T) -> Int = code(v)
            fn main() -> Int = 0
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("not bounded by `Describe`")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn trait_impl_signature_mismatch_caught() {
        // An impl method whose return type does not match the interface.
        let src = r#"
            interface Describe[T] { fn code(self: T) -> Int }
            type Shape = | Circle
            impl Describe for Shape { fn code(self: Shape) -> Bool = true }
            fn main() -> Int = 0
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("interface requires") || e.contains("return")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn trait_interface_receiver_must_be_self_typed() {
        // An interface method whose receiver isn't typed as the trait's type
        // parameter would dispatch on the wrong value and let an incoherent impl
        // (`fn show(self: Int)` for an ADT) slip through. Rejected at lowering.
        let src = r#"
            interface Show[T] { fn show(self: Int) -> Int }
            type Color = | Red | Green
            impl Show for Color { fn show(self: Int) -> Int = self + 1 }
            fn main() -> Int = show(Red)
        "#;
        let err = lexer::lex(src).and_then(parser::parse).unwrap_err();
        assert!(
            err.contains("first parameter must be the receiver typed `T`"),
            "got: {}",
            err
        );
    }

    #[test]
    fn trait_bound_on_undeclared_interface_caught() {
        // A `[T: Trait]` bound naming a trait that was never declared is a typo,
        // not a silently-ignored no-op.
        let src = r#"
            fn foo[T: Bogus](v: T) -> T = v
            fn main() -> Int = { foo(5); 0 }
        "#;
        let err = lexer::lex(src).and_then(parser::parse).unwrap_err();
        assert!(
            err.contains("undeclared interface `Bogus`"),
            "got: {}",
            err
        );
    }

    #[test]
    fn trait_bounded_call_with_unimpld_type_caught() {
        // Passing a concrete type with no `impl` of the bound to a `[T: Trait]`
        // function is caught at the CALL site (not deferred to monomorphization /
        // the interpreter, where the two backends would diverge).
        let src = r#"
            interface D[T] { fn d(self: T) -> Int }
            type Color = | Red | Green
            impl D for Color { fn d(self: Color) -> Int = 1 }
            type Hue = | Warm | Cool
            fn use_d[T: D](x: T) -> Int = d(x)
            fn main() -> Int = use_d(Warm)
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("requires its type parameter")
                && e.contains("impl `D`")
                && e.contains("Hue")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn tensor_shape_mismatch_surfaces_through_typeck() {
        // The shape checker is wired into `check`, so a provable matmul
        // dimension mismatch is reported as a (type-check-time) error.
        let src = "fn main() -> Float = {\n\
            let a = tensor_zeros(32, 64);\n\
            let b = tensor_zeros(128, 10);\n\
            tensor_get(matmul(a, b), 0, 0)\n\
        }\n";
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("shape error")
                && e.contains("inner dimensions do not match")),
            "got: {:?}",
            errs
        );
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
