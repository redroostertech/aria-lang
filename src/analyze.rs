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

use crate::ast::{Expr, ExprKind, Item, Program, Stmt};
use crate::diagnostics::json_escape;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Per-function call-graph facts.
#[derive(Debug, Clone, PartialEq)]
pub struct FnInfo {
    /// Function name.
    pub name: String,
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
/// as library, not user, code).
pub fn analyze(program: &Program, prelude_names: &HashSet<String>) -> CallGraph {
    // The set of all top-level function names (user + prelude + synthetic) and
    // their decls, so a call to one is a graph node and a call to anything else
    // is a builtin.
    let mut decls: BTreeMap<String, (usize, bool)> = BTreeMap::new(); // name -> (line, user)
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
    for item in &program.items {
        if let Item::Fn(f) = item {
            let mut names: BTreeSet<String> = BTreeSet::new();
            collect_call_names(&f.body, &mut names);
            let sites = call_sites.entry(f.name.clone()).or_default();
            collect_call_sites(&f.body, sites);
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
        functions.push(FnInfo {
            name: name.clone(),
            line,
            user,
            fan_out: user_callees.len(),
            fan_in: fn_callers.len(),
            recursive,
            callees: user_callees,
            lib_callees,
            callers: fn_callers,
            call_sites: fn_call_sites,
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

/// Walk `e`, inserting every `Expr::Call` callee name and every bare function-name
/// reference (an `Expr::Var` used as a value) into `out`. Constructor applications
/// (`Expr::Ctor`) are data, not control flow, and are deliberately skipped.
fn collect_call_names(e: &Expr, out: &mut BTreeSet<String>) {
    match &e.kind {
        ExprKind::Call(name, args) => {
            out.insert(name.clone());
            for a in args {
                collect_call_names(a, out);
            }
        }
        // A bare lowercase identifier that is NOT a local binding may be a
        // top-level function used as a value (`array_map(xs, helper)`). We record
        // every `Var` name; the caller filters by which names are actual function
        // nodes, so a plain variable reference simply finds no node and is ignored.
        ExprKind::Var(name) => {
            out.insert(name.clone());
        }
        ExprKind::Ctor(_, args) => {
            for a in args {
                collect_call_names(a, out);
            }
        }
        ExprKind::Record(_, fields) => {
            for (_, v) in fields {
                collect_call_names(v, out);
            }
        }
        ExprKind::Field(obj, _) => collect_call_names(obj, out),
        ExprKind::Update(base, updates) => {
            collect_call_names(base, out);
            for (_, v) in updates {
                collect_call_names(v, out);
            }
        }
        ExprKind::Lambda(_, body, _) => collect_call_names(body, out),
        ExprKind::Apply(callee, args, _) => {
            collect_call_names(callee, out);
            for a in args {
                collect_call_names(a, out);
            }
        }
        ExprKind::Unary(_, inner) => collect_call_names(inner, out),
        ExprKind::Binary(_, lhs, rhs) => {
            collect_call_names(lhs, out);
            collect_call_names(rhs, out);
        }
        ExprKind::If(c, t, e2) => {
            collect_call_names(c, out);
            collect_call_names(t, out);
            collect_call_names(e2, out);
        }
        ExprKind::Match(scrut, arms) => {
            collect_call_names(scrut, out);
            for arm in arms {
                collect_call_names(&arm.body, out);
            }
        }
        ExprKind::Block(stmts, last) => {
            for s in stmts {
                match s {
                    Stmt::Let(_, _, v) => collect_call_names(v, out),
                    Stmt::Expr(ex) => collect_call_names(ex, out),
                }
            }
            collect_call_names(last, out);
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
fn collect_call_sites(e: &Expr, out: &mut BTreeMap<String, BTreeSet<(u32, u32)>>) {
    let note = |name: &str, span: crate::ast::Span, out: &mut BTreeMap<String, BTreeSet<(u32, u32)>>| {
        if !span.is_none() {
            out.entry(name.to_string())
                .or_default()
                .insert((span.start_line, span.start_col));
        }
    };
    match &e.kind {
        ExprKind::Call(name, args) => {
            note(name, e.span, out);
            for a in args {
                collect_call_sites(a, out);
            }
        }
        ExprKind::Var(name) => {
            note(name, e.span, out);
        }
        ExprKind::Ctor(_, args) => {
            for a in args {
                collect_call_sites(a, out);
            }
        }
        ExprKind::Record(_, fields) => {
            for (_, v) in fields {
                collect_call_sites(v, out);
            }
        }
        ExprKind::Field(obj, _) => collect_call_sites(obj, out),
        ExprKind::Update(base, updates) => {
            collect_call_sites(base, out);
            for (_, v) in updates {
                collect_call_sites(v, out);
            }
        }
        ExprKind::Lambda(_, body, _) => collect_call_sites(body, out),
        ExprKind::Apply(callee, args, _) => {
            collect_call_sites(callee, out);
            for a in args {
                collect_call_sites(a, out);
            }
        }
        ExprKind::Unary(_, inner) => collect_call_sites(inner, out),
        ExprKind::Binary(_, lhs, rhs) => {
            collect_call_sites(lhs, out);
            collect_call_sites(rhs, out);
        }
        ExprKind::If(c, t, e2) => {
            collect_call_sites(c, out);
            collect_call_sites(t, out);
            collect_call_sites(e2, out);
        }
        ExprKind::Match(scrut, arms) => {
            collect_call_sites(scrut, out);
            for arm in arms {
                collect_call_sites(&arm.body, out);
            }
        }
        ExprKind::Block(stmts, last) => {
            for s in stmts {
                match s {
                    Stmt::Let(_, _, v) => collect_call_sites(v, out),
                    Stmt::Expr(ex) => collect_call_sites(ex, out),
                }
            }
            collect_call_sites(last, out);
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
    ///     { "name": "...", "line": N, "user": true,
    ///       "callees": ["..."], "lib_callees": ["..."], "callers": ["..."],
    ///       "recursive": false, "fan_in": N, "fan_out": N }
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
                "{{\"name\":\"{}\",\"line\":{},\"user\":{},\"callees\":{},\"lib_callees\":{},\"callers\":{},\"recursive\":{},\"fan_in\":{},\"fan_out\":{},\"call_sites\":{}}}",
                json_escape(&f.name),
                f.line,
                f.user,
                str_array_json(&f.callees),
                str_array_json(&f.lib_callees),
                str_array_json(&f.callers),
                f.recursive,
                f.fan_in,
                f.fan_out,
                call_sites_json(&f.call_sites),
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
            s.push_str(&format!("{} ({}){}\n", f.name, line, kind));
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
}
