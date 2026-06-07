//! `aria analyze` — a FIRST-CLASS STATIC CALL GRAPH for Aria programs.
//!
//! This is the *analysis half* of the AI-native thesis: where `aria check`
//! answers "is this program well-typed?" and the runtime stack trace answers
//! "where did it fail?", `aria analyze` answers "how is this program STRUCTURED?"
//! — the callers/callees of every function, dead code, recursion, and mutually
//! recursive cycles. It is a static code analyzer whose output (a stable JSON
//! object) gives an AI tool precise structural understanding of the code:
//!
//!   - "what does this function depend on?"      -> callees
//!   - "what breaks if I change this function?"  -> callers / fan_in
//!   - "is this function dead?"                  -> unused / unreachable
//!   - "is there a recursion I must terminate?"  -> recursive / cycles
//!
//! The pass runs over the TYPE-CHECKED AST (lex/parse/typeck first), so it only
//! ever analyzes well-formed programs. It reports on USER functions (the program
//! the human/model wrote); prelude functions and trait dispatchers are present in
//! the lowered program but are flagged separately and never counted as "unused
//! user code". A call to a builtin (`print_int`, `array_get`, ...) or a prelude
//! function is recorded as a callee under a separate list, so the user call graph
//! stays clean while still exposing the full dependency surface.
//!
//! v1 scope (honest): the graph is built from `Expr::Call` NAMES walked over each
//! function body. Higher-order/first-class calls (`Expr::Apply` of a
//! function-valued variable, e.g. `f(x)` where `f` is a parameter) cannot be
//! resolved statically to a target and are not edges; a top-level function passed
//! by NAME as a value (`array_map(xs, helper)`) IS recorded as a callee (we treat
//! a bare function-name reference as a use). Constructor applications are not part
//! of the call graph (they are data, not control flow).

use crate::ast::{Expr, ExprKind, FnDecl, Item, Program, Span, StmtKind};
use crate::diagnostics::json_escape;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// The declared TYPE SIGNATURE of a function, rendered as clean type strings via
/// [`crate::typeck::show`]. This is the metadata that turns the call graph into a
/// TYPED program model: alongside who-calls-whom, every node carries the types it
/// operates over. Purely declared (no inference): it reads what the source wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    /// Declared generic type parameters, e.g. `["T", "U"]` for `fn id[T,U](..)`.
    /// Empty for a non-generic function.
    pub type_params: Vec<String>,
    /// Trait bounds on the type parameters, e.g. `[("T", "Show")]` for
    /// `fn p[T: Show](..)`. Empty for an unbounded function.
    pub bounds: Vec<(String, String)>,
    /// Each parameter's name and its declared type (rendered), in source order.
    pub params: Vec<(String, String)>,
    /// The declared return type (rendered), e.g. `"Int"` or `"(Int, Bool)"`.
    pub ret: String,
}

impl Signature {
    /// Render `f`'s declared signature into clean type strings (no inference).
    fn of(f: &FnDecl) -> Signature {
        Signature {
            type_params: f.type_params.clone(),
            bounds: f.bounds.clone(),
            params: f
                .params
                .iter()
                .map(|p| (p.name.clone(), crate::typeck::show(&p.ty)))
                .collect(),
            ret: crate::typeck::show(&f.ret),
        }
    }

    /// Render this signature in Aria surface syntax, e.g.
    /// `fn id[T](x: T) -> T` or `fn add(a: Int, b: Int) -> Int`. Used by the
    /// human summary so each node shows its types inline.
    pub fn render(&self, name: &str) -> String {
        let mut s = String::from("fn ");
        s.push_str(name);
        if !self.type_params.is_empty() {
            // Attach any bound to its parameter, e.g. `[T: Show, U]`.
            let parts: Vec<String> = self
                .type_params
                .iter()
                .map(|tp| match self.bounds.iter().find(|(v, _)| v == tp) {
                    Some((_, tr)) => format!("{}: {}", tp, tr),
                    None => tp.clone(),
                })
                .collect();
            s.push('[');
            s.push_str(&parts.join(", "));
            s.push(']');
        }
        let ps: Vec<String> =
            self.params.iter().map(|(n, t)| format!("{}: {}", n, t)).collect();
        s.push('(');
        s.push_str(&ps.join(", "));
        s.push(')');
        s.push_str(" -> ");
        s.push_str(&self.ret);
        s
    }

    /// JSON-encode this signature as a stable object:
    /// `{"type_params":[..],"bounds":[["T","Show"]],"params":[{"name":..,"type":..}],"ret":".."}`.
    fn to_json(&self) -> String {
        let mut s = String::from("{\"type_params\":");
        s.push_str(&str_array_json(&self.type_params));
        s.push_str(",\"bounds\":[");
        for (i, (v, tr)) in self.bounds.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("[\"{}\",\"{}\"]", json_escape(v), json_escape(tr)));
        }
        s.push_str("],\"params\":[");
        for (i, (n, t)) in self.params.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"name\":\"{}\",\"type\":\"{}\"}}",
                json_escape(n),
                json_escape(t)
            ));
        }
        s.push_str("],\"ret\":");
        s.push_str(&format!("\"{}\"", json_escape(&self.ret)));
        s.push('}');
        s
    }
}

/// One TYPED CALL EDGE at a precise site: the callee, the exact `(line, col)` of
/// the call expression, the INFERRED concrete type of each argument expression at
/// THIS site (in source order), and the inferred type of the call's result. This
/// is what turns the call graph into a typed-edge program model: not just *who
/// calls whom from where*, but *what types actually flow on this call* — including
/// the concrete instantiation of a generic at this site (`id(2)` -> `["Int"]`/
/// `"Int"`, `id("hi")` -> `["String"]`/`"String"`). A type is `None` only when it
/// is unavailable for that span (a synthesized node, or a non-`Call` edge such as
/// a function passed by name with no applied arguments).
#[derive(Debug, Clone, PartialEq)]
pub struct TypedSite {
    /// The callee name (user, library, or builtin) — same name space as `call_sites`.
    pub callee: String,
    /// 1-based source line of the call expression.
    pub line: u32,
    /// 1-based source column of the call expression.
    pub col: u32,
    /// The inferred concrete type of each argument expression at this site, in
    /// source order. `None` for an argument whose span carried no recorded type.
    pub arg_types: Vec<Option<String>>,
    /// The inferred concrete type of the call's RESULT at this site, or `None` if
    /// the call expression's own span carried no recorded type.
    pub result_type: Option<String>,
}

/// Per-function call-graph facts.
#[derive(Debug, Clone, PartialEq)]
pub struct FnInfo {
    /// Function name.
    pub name: String,
    /// The declared TYPE SIGNATURE (params + types, return type, generics +
    /// bounds), rendered via [`crate::typeck::show`]. This is what makes the call
    /// graph a TYPED program model: each node carries the types it works over.
    pub signature: Signature,
    /// 1-based source line of the definition (`fn` keyword); 0 for
    /// compiler-generated functions (trait dispatchers / lowered impl methods).
    pub line: usize,
    /// `true` for functions the USER wrote (vs prelude / synthetic). Only user
    /// functions are eligible to be reported as `unused`.
    pub user: bool,
    /// User functions this function calls, sorted, de-duplicated.
    pub callees: Vec<String>,
    /// Builtin / prelude functions this function calls, sorted, de-duplicated.
    /// Kept separate so the user-to-user graph is clean while the full dependency
    /// surface is still visible.
    pub lib_callees: Vec<String>,
    /// User functions that call this function (the inverse of `callees`), sorted.
    pub callers: Vec<String>,
    /// `true` if this function calls itself directly (a self-edge in `callees`).
    pub recursive: bool,
    /// Number of `callers` (user functions that call this one).
    pub fan_in: usize,
    /// Number of `callees` (distinct user functions this one calls).
    pub fan_out: usize,
    /// Precise CALL-SITE locations of every edge OUT of this function: for each
    /// callee NAME (user, library, or builtin) that this function calls, the
    /// sorted, de-duplicated list of `(line, col)` positions where the call
    /// appears in this function's body. A source-located call graph: a consumer
    /// can jump to the exact `callee(..)` rather than only knowing that the edge
    /// exists. Synthesized calls (no source span) contribute no site. Keyed by
    /// callee name, sorted for stable output.
    pub call_sites: Vec<(String, Vec<(u32, u32)>)>,
    /// TYPED CALL EDGES out of this function: for every located call, the
    /// inferred argument types and result type AT THAT SITE (see [`TypedSite`]).
    /// This is the per-call-site argument-type-inference view — the precise
    /// "what types flow on this call" model, with generics concretely
    /// instantiated per site. Sorted by `(line, col)` then callee for stable
    /// output. Empty when no type map was supplied (an untyped analysis) or the
    /// function makes no located calls.
    pub typed_sites: Vec<TypedSite>,
}

/// Whole-program call-graph analysis.
#[derive(Debug, Clone)]
pub struct CallGraph {
    /// Per-function facts, in source order (user functions first as they appear).
    pub functions: Vec<FnInfo>,
    /// The entry point (`main`) if present.
    pub entry: Option<String>,
    /// User functions with no callers and which are not `main` — dead code.
    pub unused: Vec<String>,
    /// User functions not statically reachable from `main` (transitively). A
    /// superset relationship with `unused`: an unused function is unreachable, but
    /// a function may be reachable only via an unused function and thus also
    /// unreachable. Empty if there is no `main`.
    pub unreachable: Vec<String>,
    /// Mutually-recursive groups: strongly-connected components of size > 1, plus
    /// self-recursive singletons. Each inner Vec is one cycle (sorted), and the
    /// outer list is sorted for stable output.
    pub cycles: Vec<Vec<String>>,
}

/// Build the static call graph for a parsed (and ideally type-checked) program.
/// `prelude_names` is the set of prelude function names (so they can be flagged
/// as library, not user, code). Equivalent to [`analyze_typed`] with no type map
/// (so every `typed_sites` list is empty); kept for callers that only want the
/// structural graph.
pub fn analyze(program: &Program, prelude_names: &HashSet<String>) -> CallGraph {
    analyze_typed(program, prelude_names, None)
}

/// Build the static call graph, optionally enriched with PER-CALL-SITE argument
/// and result types from `types` (a span -> rendered-concrete-type map produced
/// by [`crate::typeck::check_with_types`]). When `types` is `Some`, every located
/// call edge also carries its inferred argument types and result type at that
/// site (see [`TypedSite`] / [`FnInfo::typed_sites`]); when `None`, the graph is
/// purely structural and `typed_sites` is empty everywhere. The type map is
/// METADATA: it never changes the structural fields (callees/callers/cycles/
/// unused) — those are computed identically with or without it.
pub fn analyze_typed(
    program: &Program,
    prelude_names: &HashSet<String>,
    types: Option<&HashMap<Span, String>>,
) -> CallGraph {
    // The set of all top-level function names (user + prelude + synthetic) and
    // their decls, so a call to one is a graph node and a call to anything else
    // is a builtin.
    let mut decls: BTreeMap<String, (usize, bool)> = BTreeMap::new(); // name -> (line, user)
    let mut sigs: BTreeMap<String, Signature> = BTreeMap::new(); // name -> declared signature
    let mut order: Vec<String> = Vec::new();
    let builtin: HashSet<&str> = crate::builtins::names().into_iter().collect();

    for item in &program.items {
        if let Item::Fn(f) = item {
            // A function is "user" code iff it is neither a prelude function nor a
            // compiler-generated one. Trait dispatchers / lowered impl methods are
            // the synthetic functions traits::lower emits; they are not prelude
            // and we mark them non-user by treating line-0 OR a mangled `$` name
            // as synthetic. (Mangled impl methods carry their source line but a
            // `$` in the name, so the name check catches them.)
            let is_prelude = prelude_names.contains(&f.name);
            let is_synthetic = f.name.contains('$');
            let user = !is_prelude && !is_synthetic;
            if !decls.contains_key(&f.name) {
                order.push(f.name.clone());
            }
            decls.insert(f.name.clone(), (f.line, user));
            sigs.insert(f.name.clone(), Signature::of(f));
        }
    }

    // Compute callees per function by walking its body for `ExprKind::Call`
    // names and bare function-name `ExprKind::Var` references (a function used as
    // a value). Alongside the names, collect each call's precise SOURCE SITE so
    // the graph is source-located (every edge knows the `(line, col)` it occurs
    // at). Keyed by caller, then by callee name.
    let mut callees_user: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut callees_lib: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut call_sites: BTreeMap<String, BTreeMap<String, BTreeSet<(u32, u32)>>> = BTreeMap::new();
    // Per-function TYPED call edges, collected only when a type map is supplied.
    let mut typed_sites: BTreeMap<String, Vec<TypedSite>> = BTreeMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            // The function's PARAMS are in-scope locals that shadow any top-level
            // function of the same name for the whole body. Seed the lexical scope
            // with them before walking.
            let mut scope: Scope = Scope::new();
            for p in &f.params {
                scope.push(&p.name);
            }
            let mut names: BTreeSet<String> = BTreeSet::new();
            collect_call_names(&f.body, &mut names, &mut scope);
            let sites = call_sites.entry(f.name.clone()).or_default();
            collect_call_sites(&f.body, sites, &mut scope);
            // The typed walk re-uses a fresh scope (same shadowing rules), and only
            // records when a type map is present.
            if let Some(tm) = types {
                let mut tscope: Scope = Scope::new();
                for p in &f.params {
                    tscope.push(&p.name);
                }
                let ts = typed_sites.entry(f.name.clone()).or_default();
                collect_typed_sites(&f.body, ts, &mut tscope, tm);
            }
            let u = callees_user.entry(f.name.clone()).or_default();
            let l = callees_lib.entry(f.name.clone()).or_default();
            for n in names {
                if decls.contains_key(&n) {
                    // A call to a known function node: user or library depending on
                    // the callee's own classification.
                    if decls[&n].1 {
                        u.insert(n);
                    } else {
                        l.insert(n);
                    }
                } else if builtin.contains(n.as_str()) || n == "grad" {
                    l.insert(n);
                }
                // Anything else (an unknown name) is not an edge.
            }
        }
    }

    // Invert the USER-to-USER edges to get callers.
    let mut callers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (caller, callee_set) in &callees_user {
        for callee in callee_set {
            callers.entry(callee.clone()).or_default().insert(caller.clone());
        }
    }

    // Assemble per-function info in source order.
    let mut functions: Vec<FnInfo> = Vec::new();
    for name in &order {
        let (line, user) = decls[name];
        let user_callees: Vec<String> =
            callees_user.get(name).into_iter().flatten().cloned().collect();
        let lib_callees: Vec<String> =
            callees_lib.get(name).into_iter().flatten().cloned().collect();
        let fn_callers: Vec<String> =
            callers.get(name).into_iter().flatten().cloned().collect();
        let recursive = user_callees.iter().any(|c| c == name);
        // Flatten this function's per-callee call sites into a sorted list,
        // keeping ONLY real edges: a callee that is a known function node (user
        // or library) or a builtin. A bare local-variable `Var` reference (a
        // param/`let`, e.g. `x`) is not a call and is filtered out — mirroring
        // how the name-based edge computation ignores non-function names.
        let fn_call_sites: Vec<(String, Vec<(u32, u32)>)> = call_sites
            .get(name)
            .into_iter()
            .flatten()
            .filter(|(callee, _)| {
                decls.contains_key(callee.as_str())
                    || builtin.contains(callee.as_str())
                    || callee.as_str() == "grad"
            })
            .map(|(callee, sites)| (callee.clone(), sites.iter().copied().collect()))
            .collect();
        // The typed sites, filtered to the SAME real-edge set as `call_sites`
        // (drop a local function-valued application, which is not a graph edge),
        // sorted by position then callee for stable output.
        let mut fn_typed_sites: Vec<TypedSite> = typed_sites
            .get(name)
            .into_iter()
            .flatten()
            .filter(|s| {
                decls.contains_key(s.callee.as_str())
                    || builtin.contains(s.callee.as_str())
                    || s.callee.as_str() == "grad"
            })
            .cloned()
            .collect();
        fn_typed_sites.sort_by(|a, b| {
            (a.line, a.col, &a.callee).cmp(&(b.line, b.col, &b.callee))
        });
        functions.push(FnInfo {
            name: name.clone(),
            signature: sigs.get(name).cloned().unwrap_or(Signature {
                type_params: Vec::new(),
                bounds: Vec::new(),
                params: Vec::new(),
                ret: String::new(),
            }),
            line,
            user,
            fan_out: user_callees.len(),
            fan_in: fn_callers.len(),
            recursive,
            callees: user_callees,
            lib_callees,
            callers: fn_callers,
            call_sites: fn_call_sites,
            typed_sites: fn_typed_sites,
        });
    }

    let entry = if decls.contains_key("main") { Some("main".to_string()) } else { None };

    // Unused USER functions: no callers and not `main`.
    let mut unused: Vec<String> = functions
        .iter()
        .filter(|f| f.user && f.name != "main" && f.callers.is_empty())
        .map(|f| f.name.clone())
        .collect();
    unused.sort();

    // Reachability from `main` over the USER-to-USER edges.
    let reachable = reachable_from_main(&callees_user, &decls);
    let mut unreachable: Vec<String> = if entry.is_some() {
        functions
            .iter()
            .filter(|f| f.user && !reachable.contains(&f.name))
            .map(|f| f.name.clone())
            .collect()
    } else {
        Vec::new()
    };
    unreachable.sort();

    // Strongly-connected components over the USER-to-USER edges (Tarjan), keeping
    // components of size > 1 (mutual recursion) and self-recursive singletons.
    let cycles = recursive_cycles(&callees_user, &decls);

    CallGraph { functions, entry, unused, unreachable, cycles }
}

/// A lexical scope of IN-SCOPE LOCAL binders (function params, `let` bindings,
/// lambda params, and match-arm pattern variables). A name present here SHADOWS
/// any top-level function of the same name, so a reference to it is NOT a call-
/// graph edge. Implemented as a stack of names with a parallel count map for
/// O(1) "is this name shadowed?" queries that respect nested re-bindings of the
/// same name (push/pop are balanced per lexical region).
struct Scope {
    /// Push order, so a block/lambda/arm can `truncate` back to its entry depth.
    stack: Vec<String>,
    /// How many times each name is currently in scope (handles shadowing of the
    /// same name across nested regions). Non-zero ⇒ shadowed.
    counts: HashMap<String, usize>,
}

impl Scope {
    fn new() -> Scope {
        Scope { stack: Vec::new(), counts: HashMap::new() }
    }
    /// Current binder depth — capture before entering a region, restore after.
    fn depth(&self) -> usize {
        self.stack.len()
    }
    fn push(&mut self, name: &str) {
        self.stack.push(name.to_string());
        *self.counts.entry(name.to_string()).or_insert(0) += 1;
    }
    /// Pop every binder introduced after `depth` (leaving the region).
    fn truncate(&mut self, depth: usize) {
        while self.stack.len() > depth {
            let name = self.stack.pop().unwrap();
            if let Some(c) = self.counts.get_mut(&name) {
                *c -= 1;
                if *c == 0 {
                    self.counts.remove(&name);
                }
            }
        }
    }
    /// Is `name` currently bound by an in-scope local (and thus shadows a
    /// top-level function of the same name)?
    fn is_local(&self, name: &str) -> bool {
        self.counts.get(name).copied().unwrap_or(0) > 0
    }
    /// Bind every variable a pattern introduces (record-field binders, ctor
    /// sub-binders, var patterns) into scope for the arm body.
    fn bind_pattern(&mut self, pat: &crate::ast::Pattern) {
        use crate::ast::PatternKind;
        match &pat.kind {
            PatternKind::Var(name) => self.push(name),
            PatternKind::Ctor(_, subs) => {
                for s in subs {
                    self.bind_pattern(s);
                }
            }
            PatternKind::Record(_, fields) => {
                for (name, sub) in fields {
                    match &sub.kind {
                        // `Point { x }` shorthand binds the field name itself.
                        PatternKind::Var(v) => self.push(v),
                        // `Point { x: <pat> }` binds whatever the nested pattern does.
                        _ => {
                            let _ = name;
                            self.bind_pattern(sub);
                        }
                    }
                }
            }
            PatternKind::Wild | PatternKind::Int(_) | PatternKind::Bool(_) => {}
        }
    }
}

/// Walk `e`, inserting every `Expr::Call` callee name and every bare function-name
/// reference (an `Expr::Var` used as a value) into `out` — but ONLY when the name
/// is NOT shadowed by an in-scope local binding. `scope` tracks the lexical
/// binders (function params, `let`s, lambda params, match-arm pattern variables);
/// a name that resolves to a local is not a top-level-function reference and is
/// skipped. Constructor applications (`Expr::Ctor`) are data, not control flow,
/// and are deliberately skipped.
fn collect_call_names(e: &Expr, out: &mut BTreeSet<String>, scope: &mut Scope) {
    match &e.kind {
        ExprKind::Call(name, args) => {
            // A `Call(name)` where `name` is a shadowing local (a function-valued
            // param/`let` invoked as `f(x)`) is NOT a top-level edge.
            if !scope.is_local(name) {
                out.insert(name.clone());
            }
            for a in args {
                collect_call_names(a, out, scope);
            }
        }
        // A bare lowercase identifier that is NOT a local binding may be a
        // top-level function used as a value (`array_map(xs, helper)`). A name
        // bound by an in-scope local is just a variable read and not an edge.
        ExprKind::Var(name) => {
            if !scope.is_local(name) {
                out.insert(name.clone());
            }
        }
        ExprKind::Ctor(_, args) => {
            for a in args {
                collect_call_names(a, out, scope);
            }
        }
        ExprKind::Record(_, fields) => {
            for (_, v) in fields {
                collect_call_names(v, out, scope);
            }
        }
        ExprKind::Field(obj, _) => collect_call_names(obj, out, scope),
        ExprKind::Update(base, updates) => {
            collect_call_names(base, out, scope);
            for (_, v) in updates {
                collect_call_names(v, out, scope);
            }
        }
        ExprKind::Lambda(params, body, _) => {
            // Lambda params are in scope for the body only.
            let depth = scope.depth();
            for (name, _) in params {
                scope.push(name);
            }
            collect_call_names(body, out, scope);
            scope.truncate(depth);
        }
        ExprKind::Apply(callee, args, _) => {
            collect_call_names(callee, out, scope);
            for a in args {
                collect_call_names(a, out, scope);
            }
        }
        ExprKind::Unary(_, inner) => collect_call_names(inner, out, scope),
        ExprKind::Binary(_, lhs, rhs) => {
            collect_call_names(lhs, out, scope);
            collect_call_names(rhs, out, scope);
        }
        ExprKind::If(c, t, e2) => {
            collect_call_names(c, out, scope);
            collect_call_names(t, out, scope);
            collect_call_names(e2, out, scope);
        }
        ExprKind::Match(scrut, arms) => {
            collect_call_names(scrut, out, scope);
            for arm in arms {
                // The arm's pattern variables are in scope for the arm body only.
                let depth = scope.depth();
                scope.bind_pattern(&arm.pat);
                collect_call_names(&arm.body, out, scope);
                scope.truncate(depth);
            }
        }
        ExprKind::Block(stmts, last) => {
            // A `let` binding is in scope for the REMAINDER of the block. Capture
            // the entry depth and drop all block-local binders when leaving.
            let depth = scope.depth();
            for s in stmts {
                match &s.kind {
                    StmtKind::Let { name, value, .. } => {
                        // The bound expression is evaluated in the PRE-binding
                        // scope; then the name comes into scope for the rest.
                        collect_call_names(value, out, scope);
                        scope.push(name);
                    }
                    StmtKind::Expr(ex) => collect_call_names(ex, out, scope),
                }
            }
            collect_call_names(last, out, scope);
            scope.truncate(depth);
        }
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::Unit => {}
    }
}

/// Walk `e`, recording the precise SOURCE SITE (`(line, col)` start of the call
/// expression) of every call by callee NAME, into `out` (keyed by callee). This
/// mirrors [`collect_call_names`] but keeps each call's location so the graph is
/// source-located. A `Call` records the `Call` expression's span; a bare
/// function-name `Var` records the `Var`'s span. Synthesized nodes (no span)
/// contribute nothing. The caller filters which names are real graph nodes.
fn collect_call_sites(
    e: &Expr,
    out: &mut BTreeMap<String, BTreeSet<(u32, u32)>>,
    scope: &mut Scope,
) {
    let note = |name: &str, span: crate::ast::Span, out: &mut BTreeMap<String, BTreeSet<(u32, u32)>>| {
        if !span.is_none() {
            out.entry(name.to_string())
                .or_default()
                .insert((span.start_line, span.start_col));
        }
    };
    match &e.kind {
        ExprKind::Call(name, args) => {
            // A `Call` of a shadowing local (function-valued param/`let`) is not a
            // top-level edge and contributes no site.
            if !scope.is_local(name) {
                note(name, e.span, out);
            }
            for a in args {
                collect_call_sites(a, out, scope);
            }
        }
        ExprKind::Var(name) => {
            if !scope.is_local(name) {
                note(name, e.span, out);
            }
        }
        ExprKind::Ctor(_, args) => {
            for a in args {
                collect_call_sites(a, out, scope);
            }
        }
        ExprKind::Record(_, fields) => {
            for (_, v) in fields {
                collect_call_sites(v, out, scope);
            }
        }
        ExprKind::Field(obj, _) => collect_call_sites(obj, out, scope),
        ExprKind::Update(base, updates) => {
            collect_call_sites(base, out, scope);
            for (_, v) in updates {
                collect_call_sites(v, out, scope);
            }
        }
        ExprKind::Lambda(params, body, _) => {
            let depth = scope.depth();
            for (name, _) in params {
                scope.push(name);
            }
            collect_call_sites(body, out, scope);
            scope.truncate(depth);
        }
        ExprKind::Apply(callee, args, _) => {
            collect_call_sites(callee, out, scope);
            for a in args {
                collect_call_sites(a, out, scope);
            }
        }
        ExprKind::Unary(_, inner) => collect_call_sites(inner, out, scope),
        ExprKind::Binary(_, lhs, rhs) => {
            collect_call_sites(lhs, out, scope);
            collect_call_sites(rhs, out, scope);
        }
        ExprKind::If(c, t, e2) => {
            collect_call_sites(c, out, scope);
            collect_call_sites(t, out, scope);
            collect_call_sites(e2, out, scope);
        }
        ExprKind::Match(scrut, arms) => {
            collect_call_sites(scrut, out, scope);
            for arm in arms {
                let depth = scope.depth();
                scope.bind_pattern(&arm.pat);
                collect_call_sites(&arm.body, out, scope);
                scope.truncate(depth);
            }
        }
        ExprKind::Block(stmts, last) => {
            let depth = scope.depth();
            for s in stmts {
                match &s.kind {
                    StmtKind::Let { name, value, .. } => {
                        collect_call_sites(value, out, scope);
                        scope.push(name);
                    }
                    StmtKind::Expr(ex) => collect_call_sites(ex, out, scope),
                }
            }
            collect_call_sites(last, out, scope);
            scope.truncate(depth);
        }
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::Unit => {}
    }
}

/// Walk `e`, emitting a [`TypedSite`] for every located call edge with its
/// INFERRED argument and result types looked up from `tm` (the span -> rendered-
/// type table). Mirrors [`collect_call_sites`]'s scope/shadowing rules so the
/// same set of edges is produced; the only addition is per-site types:
///
///   * `Call(name, args)` where `name` is a real (non-shadowed) callee records
///     each direct argument's span -> type as `arg_types`, and the `Call`
///     expression's own span -> type as `result_type`. This is the headline:
///     a generic call records its CONCRETE instantiation at this site.
///   * a bare function-name `Var` used as a value records a zero-argument site
///     whose `result_type` is the (instantiated) function type — a typed edge
///     with no arguments.
///
/// A type missing for a span (synthesized node, or an unmapped span) yields
/// `None` for that slot — never a fabricated type.
fn collect_typed_sites(
    e: &Expr,
    out: &mut Vec<TypedSite>,
    scope: &mut Scope,
    tm: &HashMap<Span, String>,
) {
    let lookup = |span: Span| -> Option<String> {
        if span.is_none() {
            None
        } else {
            tm.get(&span).cloned()
        }
    };
    match &e.kind {
        ExprKind::Call(name, args) => {
            if !scope.is_local(name) && !e.span.is_none() {
                out.push(TypedSite {
                    callee: name.clone(),
                    line: e.span.start_line,
                    col: e.span.start_col,
                    arg_types: args.iter().map(|a| lookup(a.span)).collect(),
                    result_type: lookup(e.span),
                });
            }
            for a in args {
                collect_typed_sites(a, out, scope, tm);
            }
        }
        ExprKind::Var(name) => {
            // A bare top-level function name used as a value: a zero-arg typed
            // edge whose result type is the function's (instantiated) type.
            if !scope.is_local(name) && !e.span.is_none() {
                out.push(TypedSite {
                    callee: name.clone(),
                    line: e.span.start_line,
                    col: e.span.start_col,
                    arg_types: Vec::new(),
                    result_type: lookup(e.span),
                });
            }
        }
        ExprKind::Ctor(_, args) => {
            for a in args {
                collect_typed_sites(a, out, scope, tm);
            }
        }
        ExprKind::Record(_, fields) => {
            for (_, v) in fields {
                collect_typed_sites(v, out, scope, tm);
            }
        }
        ExprKind::Field(obj, _) => collect_typed_sites(obj, out, scope, tm),
        ExprKind::Update(base, updates) => {
            collect_typed_sites(base, out, scope, tm);
            for (_, v) in updates {
                collect_typed_sites(v, out, scope, tm);
            }
        }
        ExprKind::Lambda(params, body, _) => {
            let depth = scope.depth();
            for (name, _) in params {
                scope.push(name);
            }
            collect_typed_sites(body, out, scope, tm);
            scope.truncate(depth);
        }
        ExprKind::Apply(callee, args, _) => {
            collect_typed_sites(callee, out, scope, tm);
            for a in args {
                collect_typed_sites(a, out, scope, tm);
            }
        }
        ExprKind::Unary(_, inner) => collect_typed_sites(inner, out, scope, tm),
        ExprKind::Binary(_, lhs, rhs) => {
            collect_typed_sites(lhs, out, scope, tm);
            collect_typed_sites(rhs, out, scope, tm);
        }
        ExprKind::If(c, t, e2) => {
            collect_typed_sites(c, out, scope, tm);
            collect_typed_sites(t, out, scope, tm);
            collect_typed_sites(e2, out, scope, tm);
        }
        ExprKind::Match(scrut, arms) => {
            collect_typed_sites(scrut, out, scope, tm);
            for arm in arms {
                let depth = scope.depth();
                scope.bind_pattern(&arm.pat);
                collect_typed_sites(&arm.body, out, scope, tm);
                scope.truncate(depth);
            }
        }
        ExprKind::Block(stmts, last) => {
            let depth = scope.depth();
            for s in stmts {
                match &s.kind {
                    StmtKind::Let { name, value, .. } => {
                        collect_typed_sites(value, out, scope, tm);
                        scope.push(name);
                    }
                    StmtKind::Expr(ex) => collect_typed_sites(ex, out, scope, tm),
                }
            }
            collect_typed_sites(last, out, scope, tm);
            scope.truncate(depth);
        }
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::Unit => {}
    }
}

/// The set of USER functions reachable from `main` over user-to-user edges.
fn reachable_from_main(
    callees_user: &BTreeMap<String, BTreeSet<String>>,
    decls: &BTreeMap<String, (usize, bool)>,
) -> HashSet<String> {
    let mut seen: HashSet<String> = HashSet::new();
    if !decls.contains_key("main") {
        return seen;
    }
    let mut stack = vec!["main".to_string()];
    while let Some(n) = stack.pop() {
        if !seen.insert(n.clone()) {
            continue;
        }
        if let Some(cs) = callees_user.get(&n) {
            for c in cs {
                if !seen.contains(c) {
                    stack.push(c.clone());
                }
            }
        }
    }
    seen
}

/// Strongly-connected components (Tarjan) over the USER-to-USER call edges,
/// returning the mutually-recursive groups (SCCs of size > 1) and self-recursive
/// singletons (a node with a self-edge), each sorted, with the outer list sorted.
fn recursive_cycles(
    callees_user: &BTreeMap<String, BTreeSet<String>>,
    decls: &BTreeMap<String, (usize, bool)>,
) -> Vec<Vec<String>> {
    // Build the node list (user functions only) and an index.
    let nodes: Vec<String> =
        decls.iter().filter(|(_, (_, u))| *u).map(|(n, _)| n.clone()).collect();
    let index_of: HashMap<&str, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    let n = nodes.len();
    let adj: Vec<Vec<usize>> = nodes
        .iter()
        .map(|name| {
            callees_user
                .get(name)
                .into_iter()
                .flatten()
                .filter_map(|c| index_of.get(c.as_str()).copied())
                .collect()
        })
        .collect();

    // Iterative Tarjan to avoid deep recursion on large graphs.
    let mut idx = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut counter = 0usize;
    let mut sccs: Vec<Vec<usize>> = Vec::new();

    // Explicit DFS frame: (node, next-child-index).
    for start in 0..n {
        if idx[start] != usize::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(start, 0)];
        while let Some(&(v, ci)) = call.last() {
            if ci == 0 {
                idx[v] = counter;
                low[v] = counter;
                counter += 1;
                stack.push(v);
                on_stack[v] = true;
            }
            if ci < adj[v].len() {
                let w = adj[v][ci];
                call.last_mut().unwrap().1 += 1;
                if idx[w] == usize::MAX {
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(idx[w]);
                }
            } else {
                // Done with v: if it's a root, pop an SCC.
                if low[v] == idx[v] {
                    let mut comp: Vec<usize> = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        comp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(comp);
                }
                call.pop();
                if let Some(&(parent, _)) = call.last() {
                    low[parent] = low[parent].min(low[v]);
                }
            }
        }
    }

    // Keep SCCs of size > 1, plus self-recursive singletons.
    let mut cycles: Vec<Vec<String>> = Vec::new();
    for comp in sccs {
        if comp.len() > 1 {
            let mut names: Vec<String> = comp.iter().map(|&i| nodes[i].clone()).collect();
            names.sort();
            cycles.push(names);
        } else {
            let v = comp[0];
            if adj[v].contains(&v) {
                cycles.push(vec![nodes[v].clone()]);
            }
        }
    }
    cycles.sort();
    cycles
}

impl CallGraph {
    /// Emit the call graph as a stable, well-formed JSON object. Schema (all
    /// arrays sorted/deterministic for diffable output):
    ///
    /// ```json
    /// {
    ///   "entry": "main" | null,
    ///   "functions": [
    ///     { "name": "...",
    ///       "signature": { "type_params": ["T"], "bounds": [["T","Show"]],
    ///                      "params": [{"name":"x","type":"T"}], "ret": "T" },
    ///       "line": N, "user": true,
    ///       "callees": ["..."], "lib_callees": ["..."], "callers": ["..."],
    ///       "recursive": false, "fan_in": N, "fan_out": N,
    ///       "call_sites": {"callee": [[line, col]]} }
    ///   ],
    ///   "unused":      ["..."],   // user fns with no callers, not main (dead code)
    ///   "unreachable": ["..."],   // user fns not reachable from main
    ///   "cycles":      [["a","b"]] // mutually-recursive groups + self-recursion
    /// }
    /// ```
    pub fn to_json(&self) -> String {
        let mut s = String::from("{");
        // entry
        s.push_str("\"entry\":");
        match &self.entry {
            Some(e) => s.push_str(&format!("\"{}\"", json_escape(e))),
            None => s.push_str("null"),
        }
        // functions
        s.push_str(",\"functions\":[");
        for (i, f) in self.functions.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"name\":\"{}\",\"signature\":{},\"line\":{},\"user\":{},\"callees\":{},\"lib_callees\":{},\"callers\":{},\"recursive\":{},\"fan_in\":{},\"fan_out\":{},\"call_sites\":{},\"typed_call_sites\":{}}}",
                json_escape(&f.name),
                f.signature.to_json(),
                f.line,
                f.user,
                str_array_json(&f.callees),
                str_array_json(&f.lib_callees),
                str_array_json(&f.callers),
                f.recursive,
                f.fan_in,
                f.fan_out,
                call_sites_json(&f.call_sites),
                typed_sites_json(&f.typed_sites),
            ));
        }
        s.push(']');
        // derived facts
        s.push_str(&format!(",\"unused\":{}", str_array_json(&self.unused)));
        s.push_str(&format!(",\"unreachable\":{}", str_array_json(&self.unreachable)));
        s.push_str(",\"cycles\":[");
        for (i, c) in self.cycles.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&str_array_json(c));
        }
        s.push(']');
        s.push('}');
        s
    }

    /// A readable human summary of the call graph for `aria analyze` (no `--json`).
    pub fn to_human(&self) -> String {
        let mut s = String::new();
        match &self.entry {
            Some(e) => s.push_str(&format!("entry: {}\n", e)),
            None => s.push_str("entry: (no `main`)\n"),
        }
        s.push_str(&format!("functions: {}\n\n", self.functions.len()));
        for f in &self.functions {
            let kind = if f.user { "" } else { " [library]" };
            let line = if f.line == 0 {
                "(generated)".to_string()
            } else {
                format!("line {}", f.line)
            };
            s.push_str(&format!("{}  ({}){}\n", f.signature.render(&f.name), line, kind));
            s.push_str(&format!(
                "  fan_in={} fan_out={}{}\n",
                f.fan_in,
                f.fan_out,
                if f.recursive { " recursive" } else { "" }
            ));
            if !f.callees.is_empty() {
                s.push_str(&format!("  calls:     {}\n", f.callees.join(", ")));
            }
            if !f.lib_callees.is_empty() {
                s.push_str(&format!("  uses(lib): {}\n", f.lib_callees.join(", ")));
            }
            if !f.callers.is_empty() {
                s.push_str(&format!("  called by: {}\n", f.callers.join(", ")));
            }
            // Show a couple of TYPED call edges (the inferred argument/result types
            // at the actual call sites) when available — kept short so the summary
            // stays readable. Each line reads `callee(Int, Int) -> Int  @ L:C`.
            for site in f.typed_sites.iter().take(2) {
                let args: Vec<String> = site
                    .arg_types
                    .iter()
                    .map(|a| a.clone().unwrap_or_else(|| "?".to_string()))
                    .collect();
                let ret = site.result_type.clone().unwrap_or_else(|| "?".to_string());
                s.push_str(&format!(
                    "  call:      {}({}) -> {}  @ {}:{}\n",
                    site.callee,
                    args.join(", "),
                    ret,
                    site.line,
                    site.col
                ));
            }
            if f.typed_sites.len() > 2 {
                s.push_str(&format!(
                    "             (+{} more typed call site(s))\n",
                    f.typed_sites.len() - 2
                ));
            }
        }
        s.push('\n');
        if self.unused.is_empty() {
            s.push_str("unused: (none)\n");
        } else {
            s.push_str(&format!("unused (dead code): {}\n", self.unused.join(", ")));
        }
        if !self.unreachable.is_empty() {
            s.push_str(&format!("unreachable from main: {}\n", self.unreachable.join(", ")));
        }
        if self.cycles.is_empty() {
            s.push_str("recursive cycles: (none)\n");
        } else {
            s.push_str("recursive cycles:\n");
            for c in &self.cycles {
                if c.len() == 1 {
                    s.push_str(&format!("  - {} (self-recursive)\n", c[0]));
                } else {
                    s.push_str(&format!("  - {} (mutual)\n", c.join(" <-> ")));
                }
            }
        }
        s
    }
}

/// JSON-encode a slice of strings as an array literal, escaping each element.
fn str_array_json(xs: &[String]) -> String {
    let mut s = String::from("[");
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\"{}\"", json_escape(x)));
    }
    s.push(']');
    s
}

/// JSON-encode a function's call sites as an object keyed by callee name, each
/// value an array of `[line, col]` pairs (the precise positions that callee is
/// called from in this function), e.g. `{"helper":[[3,5],[4,9]]}`. Empty `{}`
/// when the function makes no located calls.
fn call_sites_json(sites: &[(String, Vec<(u32, u32)>)]) -> String {
    let mut s = String::from("{");
    for (i, (callee, locs)) in sites.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\"{}\":[", json_escape(callee)));
        for (j, (line, col)) in locs.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&format!("[{},{}]", line, col));
        }
        s.push(']');
    }
    s.push('}');
    s
}

/// JSON-encode a function's TYPED call edges as an array of objects, each:
/// `{"callee":"add","line":3,"col":20,"arg_types":["Int","Int"],"result_type":"Int"}`.
/// `arg_types` is an array (per argument, in source order); a type that was
/// unavailable for a span is encoded as JSON `null`. `result_type` is the call's
/// inferred result type, or `null`. Empty `[]` when the function makes no located
/// calls or the analysis carried no type map. This is the per-call-site argument-
/// type-inference view: the SAME edges as `call_sites`, enriched with the concrete
/// types that flow on each call (generics instantiated per site).
fn typed_sites_json(sites: &[TypedSite]) -> String {
    let opt = |t: &Option<String>| match t {
        Some(s) => format!("\"{}\"", json_escape(s)),
        None => "null".to_string(),
    };
    let mut s = String::from("[");
    for (i, site) in sites.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"callee\":\"{}\",\"line\":{},\"col\":{},\"arg_types\":[",
            json_escape(&site.callee),
            site.line,
            site.col
        ));
        for (j, at) in site.arg_types.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&opt(at));
        }
        s.push_str("],\"result_type\":");
        s.push_str(&opt(&site.result_type));
        s.push('}');
    }
    s.push(']');
    s
}

/// The set of prelude function NAMES, derived by parsing the prelude source on its
/// own. Used to classify a function in the combined (user + prelude) program as
/// library code. Computed once.
pub fn prelude_fn_names() -> HashSet<String> {
    let mut names = HashSet::new();
    if let Ok(toks) = crate::lexer::lex(crate::prelude::SOURCE) {
        if let Ok(prog) = crate::parser::parse(toks) {
            for item in &prog.items {
                if let Item::Fn(f) = item {
                    names.insert(f.name.clone());
                }
            }
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a user program THROUGH the prelude (exactly as the CLI does) and
    /// build its call graph.
    fn graph_of(src: &str) -> CallGraph {
        let toks = crate::lexer::lex(&crate::prelude::wrap(src)).expect("lex");
        let prog = crate::parser::parse(toks).expect("parse");
        // Type-check to mirror `aria analyze` (analysis runs on a checked AST).
        crate::typeck::check(&prog).expect("typeck");
        analyze(&prog, &prelude_fn_names())
    }

    fn info<'a>(g: &'a CallGraph, name: &str) -> &'a FnInfo {
        g.functions.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("no fn {}", name))
    }

    /// Like [`graph_of`] but TYPED: runs `check_with_types` to get the span->type
    /// table and threads it through `analyze_typed`, so each call edge carries its
    /// per-site argument/result types.
    fn typed_graph_of(src: &str) -> CallGraph {
        let toks = crate::lexer::lex(&crate::prelude::wrap(src)).expect("lex");
        let prog = crate::parser::parse(toks).expect("parse");
        let types = crate::typeck::check_with_types(&prog).expect("typeck");
        analyze_typed(&prog, &prelude_fn_names(), Some(&types))
    }

    /// The typed call sites of `name` to `callee`, as `(arg_types, result_type)`
    /// with each type rendered (a missing type becomes `"?"`).
    fn typed_sites_to(g: &CallGraph, name: &str, callee: &str) -> Vec<(Vec<String>, String)> {
        info(g, name)
            .typed_sites
            .iter()
            .filter(|s| s.callee == callee)
            .map(|s| {
                let args = s
                    .arg_types
                    .iter()
                    .map(|a| a.clone().unwrap_or_else(|| "?".to_string()))
                    .collect();
                (args, s.result_type.clone().unwrap_or_else(|| "?".to_string()))
            })
            .collect()
    }

    #[test]
    fn chain_callees_and_callers() {
        let src = "\
fn c() -> Int = 3
fn b() -> Int = c()
fn a() -> Int = b()
fn main() -> Int = a()
";
        let g = graph_of(src);
        assert_eq!(g.entry.as_deref(), Some("main"));
        assert_eq!(info(&g, "a").callees, vec!["b".to_string()]);
        assert_eq!(info(&g, "b").callees, vec!["c".to_string()]);
        assert_eq!(info(&g, "main").callees, vec!["a".to_string()]);
        // callers are the inverse.
        assert_eq!(info(&g, "b").callers, vec!["a".to_string()]);
        assert_eq!(info(&g, "a").callers, vec!["main".to_string()]);
        // fan_in / fan_out.
        assert_eq!(info(&g, "b").fan_in, 1);
        assert_eq!(info(&g, "a").fan_out, 1);
        // No unused, no cycles.
        assert!(g.unused.is_empty(), "unused: {:?}", g.unused);
        assert!(g.cycles.is_empty(), "cycles: {:?}", g.cycles);
    }

    #[test]
    fn unused_and_unreachable_dead_code() {
        let src = "\
fn used() -> Int = 1
fn unused() -> Int = 2
fn main() -> Int = used()
";
        let g = graph_of(src);
        assert_eq!(g.unused, vec!["unused".to_string()]);
        assert_eq!(g.unreachable, vec!["unused".to_string()]);
        assert!(info(&g, "used").callers.contains(&"main".to_string()));
    }

    #[test]
    fn self_recursion_flagged() {
        let src = "\
fn fact(n: Int) -> Int = if n == 0 { 1 } else { n * fact(n - 1) }
fn main() -> Int = fact(5)
";
        let g = graph_of(src);
        assert!(info(&g, "fact").recursive, "fact should be recursive");
        assert!(info(&g, "fact").callees.contains(&"fact".to_string()));
        assert_eq!(g.cycles, vec![vec!["fact".to_string()]]);
    }

    #[test]
    fn mutual_recursion_is_a_cycle() {
        let src = "\
fn even(n: Int) -> Bool = if n == 0 { true } else { odd(n - 1) }
fn odd(n: Int) -> Bool = if n == 0 { false } else { even(n - 1) }
fn main() -> Int = if even(10) { 1 } else { 0 }
";
        let g = graph_of(src);
        // even <-> odd form one SCC of size 2.
        assert_eq!(g.cycles, vec![vec!["even".to_string(), "odd".to_string()]]);
        // Neither is "directly" recursive (no self-edge).
        assert!(!info(&g, "even").recursive);
        assert!(!info(&g, "odd").recursive);
        assert!(info(&g, "even").callees.contains(&"odd".to_string()));
        assert!(info(&g, "odd").callees.contains(&"even".to_string()));
    }

    #[test]
    fn builtins_are_lib_callees_not_user_edges() {
        let src = "\
fn main() -> Int = { print_int(7); 7 }
";
        let g = graph_of(src);
        let m = info(&g, "main");
        assert!(m.callees.is_empty(), "no user callees: {:?}", m.callees);
        assert!(m.lib_callees.contains(&"print_int".to_string()));
    }

    #[test]
    fn prelude_call_is_lib_not_user() {
        let src = "\
fn main() -> Int = array_len(range(3))
";
        let g = graph_of(src);
        let m = info(&g, "main");
        // `range` is a prelude function -> a lib callee, not a user callee.
        assert!(m.callees.is_empty(), "user callees should be empty: {:?}", m.callees);
        assert!(m.lib_callees.contains(&"range".to_string()));
        // `range` itself is present but flagged non-user, so it's never "unused".
        assert!(!g.unused.contains(&"range".to_string()));
    }

    #[test]
    fn function_passed_by_name_is_a_callee() {
        let src = "\
fn dbl(x: Int) -> Int = x * 2
fn main() -> Int = { let ys = array_map(range(3), dbl); array_len(ys) }
";
        let g = graph_of(src);
        // `dbl` is referenced by NAME as a value -> recorded as a user callee, so
        // it is NOT dead code.
        assert!(info(&g, "main").callees.contains(&"dbl".to_string()));
        assert!(!g.unused.contains(&"dbl".to_string()), "dbl is used as a value");
    }

    #[test]
    fn json_is_well_formed_and_stable() {
        let src = "\
fn helper() -> Int = 1
fn unused() -> Int = 9
fn main() -> Int = helper()
";
        let g = graph_of(src);
        let json = g.to_json();
        // Spot-check structure and content.
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("\"entry\":\"main\""));
        assert!(json.contains("\"unused\":[\"unused\"]"));
        assert!(json.contains("\"name\":\"helper\""));
        // Each function carries a `call_sites` object; `main` calls `helper` at a
        // precise `[line, col]` (line 3). The empty-call function reports `{}`.
        assert!(json.contains("\"call_sites\":"));
        assert!(json.contains("\"call_sites\":{\"helper\":[[3,"), "got {}", json);
        // Roundtrip-ish: balanced braces/brackets.
        let braces = json.chars().filter(|&c| c == '{').count();
        let close = json.chars().filter(|&c| c == '}').count();
        assert_eq!(braces, close, "balanced braces in {}", json);
    }

    #[test]
    fn call_sites_record_precise_locations_and_filter_locals() {
        // Each edge OUT of a function carries the precise `(line, col)` it occurs
        // at; a bare local-variable reference (`x`) is NOT an edge and must not
        // appear among the call sites.
        let src = "\
fn helper(x: Int) -> Int = x + 1
fn main() -> Int = helper(1) + helper(2)
";
        let g = graph_of(src);
        let m = info(&g, "main");
        let sites: &Vec<(u32, u32)> = &m
            .call_sites
            .iter()
            .find(|(c, _)| c == "helper")
            .expect("helper edge has call sites")
            .1;
        // Two calls to `helper` on line 2, at the two precise columns.
        assert_eq!(sites, &vec![(2, 20), (2, 32)], "precise call sites of helper");
        // `helper` itself only references its param `x`, which is not a call edge.
        let h = info(&g, "helper");
        assert!(
            h.call_sites.is_empty(),
            "local var `x` must not be a call site: {:?}",
            h.call_sites
        );
    }

    // ---- typed call graph: declared signatures ------------------------

    #[test]
    fn signature_renders_generic_multi_arg_recursive_and_unit() {
        let src = "\
fn id[T](x: T) -> T = x
fn fib(n: Int) -> Int = if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
fn add(a: Int, b: Int) -> Int = a + b
fn noop() -> Unit = ()
fn main() -> Int = { let z = noop(); add(id(fib(5)), 0) }
";
        let g = graph_of(src);
        // Generic single-param identity: type_params=[T], param x: T, ret T.
        let id = info(&g, "id");
        assert_eq!(id.signature.type_params, vec!["T".to_string()]);
        assert_eq!(id.signature.params, vec![("x".to_string(), "T".to_string())]);
        assert_eq!(id.signature.ret, "T");
        assert_eq!(id.signature.render("id"), "fn id[T](x: T) -> T");
        // Recursive Int->Int.
        let fib = info(&g, "fib");
        assert_eq!(fib.signature.render("fib"), "fn fib(n: Int) -> Int");
        assert!(fib.recursive);
        // Multi-arg.
        let add = info(&g, "add");
        assert_eq!(add.signature.render("add"), "fn add(a: Int, b: Int) -> Int");
        // No-param, Unit return.
        let noop = info(&g, "noop");
        assert!(noop.signature.params.is_empty());
        assert_eq!(noop.signature.ret, "Unit");
        assert_eq!(noop.signature.render("noop"), "fn noop() -> Unit");
    }

    #[test]
    fn signature_renders_bounds() {
        let src = "\
interface Show[T] { fn show(self: T) -> String }
type Color = | Red
impl Show for Color { fn show(self: Color) -> String = \"red\" }
fn label[T: Show](x: T) -> String = show(x)
fn main() -> Int = { let z = label(Red); 0 }
";
        let g = graph_of(src);
        let lbl = info(&g, "label");
        assert_eq!(lbl.signature.type_params, vec!["T".to_string()]);
        assert_eq!(lbl.signature.bounds, vec![("T".to_string(), "Show".to_string())]);
        // The bound attaches to its type param in the rendered form.
        assert_eq!(lbl.signature.render("label"), "fn label[T: Show](x: T) -> String");
    }

    #[test]
    fn json_carries_signatures_and_is_well_formed() {
        let src = "\
fn id[T](x: T) -> T = x
fn fib(n: Int) -> Int = if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
fn even(n: Int) -> Bool = if n == 0 { true } else { odd(n - 1) }
fn odd(n: Int) -> Bool = if n == 0 { false } else { even(n - 1) }
fn add(a: Int, b: Int) -> Int = a + b
fn dead() -> Int = 0
fn main() -> Int = { let z = even(id(fib(add(1, 2)))); 0 }
";
        let g = graph_of(src);
        let json = g.to_json();
        // Each function carries a `signature` object.
        assert!(json.contains("\"signature\":{\"type_params\":"));
        // The generic id renders its type param + param type + ret.
        assert!(json.contains("\"name\":\"id\",\"signature\":{\"type_params\":[\"T\"],\"bounds\":[],\"params\":[{\"name\":\"x\",\"type\":\"T\"}],\"ret\":\"T\"}"), "got {}", json);
        // The multi-arg add.
        assert!(json.contains("\"params\":[{\"name\":\"a\",\"type\":\"Int\"},{\"name\":\"b\",\"type\":\"Int\"}]"));
        // Bool return on even/odd.
        assert!(json.contains("\"ret\":\"Bool\""));
        // Structure preserved alongside the new signatures: mutual + self cycles,
        // and dead code. (fib is self-recursive; even<->odd is a mutual cycle.)
        assert!(json.contains("\"cycles\":[[\"even\",\"odd\"],[\"fib\"]]"), "got {}", json);
        assert!(json.contains("\"unused\":[\"dead\"]"));
        // Balanced braces (well-formedness spot-check; full check is python json.tool).
        let open = json.chars().filter(|&c| c == '{').count();
        let close = json.chars().filter(|&c| c == '}').count();
        assert_eq!(open, close, "balanced braces in {}", json);
        let ob = json.chars().filter(|&c| c == '[').count();
        let cb = json.chars().filter(|&c| c == ']').count();
        assert_eq!(ob, cb, "balanced brackets in {}", json);
    }

    #[test]
    fn human_summary_shows_signatures() {
        let src = "\
fn fib(n: Int) -> Int = if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
fn main() -> Int = fib(5)
";
        let g = graph_of(src);
        let h = g.to_human();
        assert!(h.contains("fn fib(n: Int) -> Int"), "human summary should show signatures:\n{}", h);
        assert!(h.contains("fn main() -> Int"));
    }

    #[test]
    fn human_summary_is_readable() {
        let src = "\
fn even(n: Int) -> Bool = if n == 0 { true } else { odd(n - 1) }
fn odd(n: Int) -> Bool = if n == 0 { false } else { even(n - 1) }
fn dead() -> Int = 0
fn main() -> Int = if even(4) { 1 } else { 0 }
";
        let g = graph_of(src);
        let h = g.to_human();
        assert!(h.contains("entry: main"));
        assert!(h.contains("unused (dead code): dead"));
        assert!(h.contains("recursive cycles:"));
        assert!(h.contains("even <-> odd") || h.contains("odd <-> even"));
    }

    // ---- scope tracking: a local binder shadows a top-level function ----

    #[test]
    fn shadowed_let_is_not_a_call_edge() {
        // `h` binds a local `let g`, so the reference to `g` is the local, NOT the
        // top-level `fn g`. `g` therefore has no callers and is unused.
        let src = "\
fn g() -> Int = 5
fn h() -> Int = { let g = 1; g + 1 }
fn main() -> Int = h()
";
        let g = graph_of(src);
        assert!(info(&g, "h").callees.is_empty(), "h calls nothing: {:?}", info(&g, "h").callees);
        assert!(info(&g, "g").callers.is_empty(), "g has no callers");
        assert!(g.unused.contains(&"g".to_string()), "g is unused: {:?}", g.unused);
    }

    #[test]
    fn shadowed_param_is_not_a_call_edge() {
        // The param `inc` shadows the top-level `fn inc`, so `inc(x)` in `use_it`
        // calls the LOCAL value, not the top-level function.
        let src = "\
fn inc(n: Int) -> Int = n + 1
fn use_it(inc: Int) -> Int = inc + 1
fn main() -> Int = use_it(3)
";
        let g = graph_of(src);
        assert!(info(&g, "use_it").callees.is_empty(), "use_it calls nothing");
        assert!(g.unused.contains(&"inc".to_string()), "inc is unused: {:?}", g.unused);
    }

    #[test]
    fn shadowed_param_local_recursion_is_not_self_recursive() {
        // `loopy` binds a local `let loopy`; the reference is the local, so `loopy`
        // is NOT self-recursive and forms no cycle.
        let src = "\
fn loopy(n: Int) -> Int = { let loopy = n + 1; loopy }
fn main() -> Int = loopy(3)
";
        let g = graph_of(src);
        assert!(!info(&g, "loopy").recursive, "loopy is not recursive");
        assert!(g.cycles.is_empty(), "no cycles: {:?}", g.cycles);
    }

    #[test]
    fn shadowed_lambda_param_is_not_a_call_edge() {
        // The lambda param `target` shadows top-level `fn target`, so the body's
        // reference is the param, not the function.
        let src = "\
fn target(x: Int) -> Int = x
fn run() -> Int = { let f = \\(target: Int) -> target + 1; f(2) }
fn main() -> Int = run()
";
        let g = graph_of(src);
        assert!(info(&g, "run").callees.is_empty(), "run calls nothing: {:?}", info(&g, "run").callees);
        assert!(g.unused.contains(&"target".to_string()), "target is unused: {:?}", g.unused);
    }

    #[test]
    fn shadowed_match_var_is_not_a_call_edge() {
        // The match-arm var `helper` shadows top-level `fn helper`; the arm body's
        // reference is the bound pattern variable.
        let src = "\
fn helper() -> Int = 7
fn pick(n: Int) -> Int = match n { helper => helper, }
fn main() -> Int = pick(1)
";
        let g = graph_of(src);
        assert!(info(&g, "pick").callees.is_empty(), "pick calls nothing: {:?}", info(&g, "pick").callees);
        assert!(g.unused.contains(&"helper".to_string()), "helper is unused: {:?}", g.unused);
    }

    #[test]
    fn user_fn_colliding_with_prelude_local_has_no_phantom_caller() {
        // A user `fn x` collides in NAME with locals used inside prelude bodies
        // (e.g. `x`/`f`/`n` in array combinators). Walking the prelude must NOT
        // create a phantom caller for the user's `x`: it is unused.
        let src = "\
fn x(n: Int) -> Int = n + 1
fn main() -> Int = 0
";
        let g = graph_of(src);
        assert!(info(&g, "x").callers.is_empty(), "x has no callers: {:?}", info(&g, "x").callers);
        assert!(g.unused.contains(&"x".to_string()), "x is unused: {:?}", g.unused);
    }

    #[test]
    fn function_valued_local_called_is_not_a_user_edge() {
        // `apply` takes a function-valued param `f` and calls `f(x)`. That call is
        // to the LOCAL value, not a top-level function, so `apply` has no callee.
        let src = "\
fn dbl(x: Int) -> Int = x * 2
fn apply(f: (Int) -> Int, x: Int) -> Int = f(x)
fn main() -> Int = apply(dbl, 3)
";
        let g = graph_of(src);
        // `apply`'s body call `f(x)` is the param, not an edge.
        assert!(info(&g, "apply").callees.is_empty(), "apply has no user callee: {:?}", info(&g, "apply").callees);
        // `dbl` is passed by NAME to `apply` from main -> it is a callee of main and used.
        assert!(info(&g, "main").callees.contains(&"apply".to_string()));
        assert!(info(&g, "main").callees.contains(&"dbl".to_string()));
        assert!(!g.unused.contains(&"dbl".to_string()), "dbl is used as a value");
    }

    // ---- typed call edges: per-call-site argument & result types -------

    #[test]
    fn typed_sites_record_concrete_arg_and_result_types() {
        let src = "\
fn add(a: Int, b: Int) -> Int = a + b
fn main() -> Int = add(1, 2)
";
        let g = typed_graph_of(src);
        let sites = typed_sites_to(&g, "main", "add");
        assert_eq!(sites, vec![(vec!["Int".to_string(), "Int".to_string()], "Int".to_string())]);
    }

    #[test]
    fn typed_sites_generic_instantiation_differs_per_site() {
        // The headline: one generic `id`, two call sites, DIFFERENT concrete types.
        let src = "\
fn id[T](x: T) -> T = x
fn main() -> Int = { let a = id(2); let s = id(\"hi\"); a }
";
        let g = typed_graph_of(src);
        let sites = typed_sites_to(&g, "main", "id");
        // Two sites: one instantiated at Int, one at String.
        assert!(
            sites.contains(&(vec!["Int".to_string()], "Int".to_string())),
            "id(2) is Int->Int: {:?}",
            sites
        );
        assert!(
            sites.contains(&(vec!["String".to_string()], "String".to_string())),
            "id(\"hi\") is String->String: {:?}",
            sites
        );
        assert_eq!(sites.len(), 2);
    }

    #[test]
    fn typed_sites_nested_call_argument_types() {
        // `add(id(2), 3)`: the inner `id(2)` is Int->Int, and the outer `add` sees
        // both arguments as Int.
        let src = "\
fn add(a: Int, b: Int) -> Int = a + b
fn id[T](x: T) -> T = x
fn main() -> Int = add(id(2), 3)
";
        let g = typed_graph_of(src);
        let add_sites = typed_sites_to(&g, "main", "add");
        assert_eq!(
            add_sites,
            vec![(vec!["Int".to_string(), "Int".to_string()], "Int".to_string())]
        );
        let id_sites = typed_sites_to(&g, "main", "id");
        assert_eq!(id_sites, vec![(vec!["Int".to_string()], "Int".to_string())]);
    }

    #[test]
    fn typed_sites_builtin_call_concrete_collection_types() {
        // A builtin (Vector/Array) call carries the concrete instantiation.
        let src = "\
fn main() -> Int = { let v = [10, 20, 30]; array_len(v) }
";
        let g = typed_graph_of(src);
        let sites = typed_sites_to(&g, "main", "array_len");
        assert_eq!(
            sites,
            vec![(vec!["Array[Int]".to_string()], "Int".to_string())]
        );
    }

    #[test]
    fn typed_sites_record_and_float_argument_types() {
        let src = "\
type Point = { x: Float, y: Float }
fn scale(p: Point, k: Float) -> Float = p.x + k
fn main() -> Float = scale(Point { x: 1.0, y: 2.0 }, 3.5)
";
        let g = typed_graph_of(src);
        let sites = typed_sites_to(&g, "main", "scale");
        assert_eq!(
            sites,
            vec![(vec!["Point".to_string(), "Float".to_string()], "Float".to_string())]
        );
    }

    #[test]
    fn typed_sites_map_builtin_concrete_types() {
        let src = "\
fn main() -> Int = { let m0 = map_new(); let m1 = map_insert(m0, 1, 100); map_len(m1) }
";
        let g = typed_graph_of(src);
        let new_sites = typed_sites_to(&g, "main", "map_new");
        assert_eq!(new_sites, vec![(vec![], "Map[Int, Int]".to_string())]);
        let ins_sites = typed_sites_to(&g, "main", "map_insert");
        assert_eq!(
            ins_sites,
            vec![(
                vec![
                    "Map[Int, Int]".to_string(),
                    "Int".to_string(),
                    "Int".to_string()
                ],
                "Map[Int, Int]".to_string()
            )]
        );
    }

    #[test]
    fn untyped_analyze_has_empty_typed_sites() {
        // The structural-only `analyze` (no type map) leaves `typed_sites` empty,
        // and the structural fields are identical to before.
        let src = "\
fn helper(x: Int) -> Int = x + 1
fn main() -> Int = helper(1)
";
        let g = graph_of(src);
        assert!(info(&g, "main").typed_sites.is_empty());
        // Structural fields unchanged.
        assert!(info(&g, "main").callees.contains(&"helper".to_string()));
    }

    #[test]
    fn typed_sites_graceful_null_when_type_unavailable() {
        // A type map missing a span yields `None` (JSON null), never a fabricated
        // type. Build a graph with an EMPTY type map and confirm null surfaces.
        let toks = crate::lexer::lex(&crate::prelude::wrap(
            "fn helper(x: Int) -> Int = x + 1\nfn main() -> Int = helper(1)",
        ))
        .expect("lex");
        let prog = crate::parser::parse(toks).expect("parse");
        let empty: HashMap<Span, String> = HashMap::new();
        let g = analyze_typed(&prog, &prelude_fn_names(), Some(&empty));
        let site = info(&g, "main")
            .typed_sites
            .iter()
            .find(|s| s.callee == "helper")
            .expect("helper edge");
        assert_eq!(site.arg_types, vec![None]);
        assert_eq!(site.result_type, None);
    }

    #[test]
    fn typed_call_sites_json_is_well_formed_and_concrete() {
        let src = "\
fn id[T](x: T) -> T = x
fn main() -> Int = { let a = id(2); let s = id(\"hi\"); a }
";
        let g = typed_graph_of(src);
        let json = g.to_json();
        // The new key is present and carries the per-site concrete instantiations.
        assert!(json.contains("\"typed_call_sites\":"));
        assert!(
            json.contains("\"callee\":\"id\",\"line\":2,") && json.contains("\"arg_types\":[\"Int\"],\"result_type\":\"Int\""),
            "id(2) typed site: {}",
            json
        );
        assert!(
            json.contains("\"arg_types\":[\"String\"],\"result_type\":\"String\""),
            "id(\"hi\") typed site: {}",
            json
        );
        // Balanced braces/brackets (well-formedness spot-check).
        let ob = json.chars().filter(|&c| c == '{').count();
        let cb = json.chars().filter(|&c| c == '}').count();
        assert_eq!(ob, cb, "balanced braces");
        let os = json.chars().filter(|&c| c == '[').count();
        let cs = json.chars().filter(|&c| c == ']').count();
        assert_eq!(os, cs, "balanced brackets");
    }

    #[test]
    fn typed_human_summary_shows_call_types() {
        let src = "\
fn add(a: Int, b: Int) -> Int = a + b
fn main() -> Int = add(1, 2)
";
        let g = typed_graph_of(src);
        let h = g.to_human();
        assert!(h.contains("call:      add(Int, Int) -> Int"), "human shows typed call:\n{}", h);
    }
}
