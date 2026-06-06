//! `aria analyze` data-flow layer — per-function VARIABLE DATA-FLOW.
//!
//! Where the call-graph half of `aria analyze` (see `analyze.rs`) answers "how
//! do functions depend on each other?", this layer answers "how does DATA move
//! inside one function?": for every local binding — a parameter, a `let`, a
//! lambda parameter, or a match-arm pattern variable — it computes the exact
//! DEF (where the binding is introduced) and the exact set of USES (every place
//! the variable is read), plus the binding's type and whether it is dead.
//!
//! Aria is PURE / IMMUTABLE: every binding is assigned EXACTLY ONCE (there is no
//! mutation, no reassignment, no `var`). That single-assignment property makes
//! def-use chains EXACT — there is no reaching-definitions lattice, no
//! flow-sensitive merge: a use binds to the one lexically-innermost binding of
//! that name, full stop. The result is a precise, deterministic data-flow model
//! an AI/static-analysis tool can rely on:
//!
//!   - "where is this variable used?"            -> uses
//!   - "is this binding dead?"                   -> unused (use_count == 0)
//!   - "does this name shadow an outer one?"     -> shadows
//!
//! The analysis runs on the TYPE-CHECKED AST (the same precondition as the call
//! graph) and reuses the EXACT lexical-scope walk from `analyze.rs` (params,
//! `let`-for-rest-of-block, lambda params, match-arm pattern binders including
//! record-field shorthand). A binding's TYPE is looked up from the span->type
//! table `typeck::check_with_types` produces: parameters from their declared
//! type, `let`/lambda/match binders from the inferred type of the binding's
//! value/use site. Binder DEF spans come from the parser's [`BinderSpans`] side
//! table (the AST does not carry binder spans for `let`/lambda/match binders).
//!
//! Top-level function names, prelude functions, and builtins are NOT local
//! bindings and are never reported here (the call graph already models them).

use crate::ast::{BinderSpans, Expr, ExprKind, FnDecl, Item, Pattern, Program, Span, Stmt, Ty};
use crate::diagnostics::json_escape;
use std::collections::HashMap;

/// The kind of a local binding, mirroring the four ways Aria introduces a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindKind {
    /// A function parameter.
    Param,
    /// A `let` binding inside a block.
    Let,
    /// A lambda parameter (`\x -> ..` / `\(x: T) -> ..`).
    LambdaParam,
    /// A match-arm pattern variable (incl. record-field shorthand / ctor binders).
    MatchBinder,
}

impl BindKind {
    /// The stable string used in the JSON `kind` field.
    pub fn as_str(self) -> &'static str {
        match self {
            BindKind::Param => "param",
            BindKind::Let => "let",
            BindKind::LambdaParam => "lambda_param",
            BindKind::MatchBinder => "match_binder",
        }
    }
}

/// One local binding's data-flow facts: its name, kind, definition site, type,
/// and the precise set of read locations (uses). Because Aria bindings are
/// single-assignment, `uses` is the COMPLETE, EXACT def-use chain of this
/// binding — every read of the variable, and nothing else.
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    /// The bound variable's name.
    pub name: String,
    /// How the binding was introduced (param / let / lambda_param / match_binder).
    pub kind: BindKind,
    /// 1-based `(line, col)` of the binder's definition site. `(0, 0)` only if the
    /// binder span was unavailable (a synthesized binder).
    pub def: (u32, u32),
    /// The full binder def [`Span`] (start+end), so a lint can attach a precise
    /// range. [`Span::none`] when unavailable.
    pub def_span: Span,
    /// The binding's rendered type, if known from the span->type table (declared
    /// for params, inferred for the rest). `None` when no type was recorded.
    pub ty: Option<String>,
    /// Every source location where this variable is READ, sorted, de-duplicated.
    pub uses: Vec<(u32, u32)>,
    /// `true` iff the binding is never read (`uses` is empty) — a dead binding.
    pub unused: bool,
}

impl Binding {
    /// The number of distinct read sites.
    pub fn use_count(&self) -> usize {
        self.uses.len()
    }
}

/// A shadowing report: an inner binding whose `name` shadows an outer in-scope
/// binding of the same name. Records both definition sites so a tool can show
/// "`x` at line 6 shadows `x` at line 2".
#[derive(Debug, Clone, PartialEq)]
pub struct Shadow {
    /// The shadowed name.
    pub name: String,
    /// `(line, col)` of the INNER (shadowing) binding's def.
    pub inner: (u32, u32),
    /// `(line, col)` of the OUTER (shadowed) binding's def.
    pub outer: (u32, u32),
}

/// Per-function data-flow: every local binding with its def-use facts, plus the
/// whole-function derived lists (dead bindings, shadowing bindings).
#[derive(Debug, Clone, PartialEq)]
pub struct FnDataFlow {
    /// Function name.
    pub name: String,
    /// All local bindings, in the order they are introduced (params first, then
    /// `let`/lambda/match binders in source order of their def).
    pub bindings: Vec<Binding>,
    /// Names of dead bindings (def'd, never used), in binding order, with each
    /// binding's def location — the unused-binding subset of `bindings`.
    pub unused_bindings: Vec<(String, (u32, u32))>,
    /// Shadowing bindings discovered while walking the function's scopes.
    pub shadows: Vec<Shadow>,
}

/// Compute per-function data-flow for every USER function in `program`. `types`
/// is the span->rendered-type table from [`crate::typeck::check_with_types`]
/// (used to type each binding); `binders` is the parser's [`BinderSpans`] side
/// table (used to locate `let`/lambda/match binder defs). Library/prelude and
/// compiler-synthetic functions (a `$` in the name) are skipped: only the code
/// the user wrote is modeled.
pub fn analyze(
    program: &Program,
    prelude_names: &std::collections::HashSet<String>,
    types: &HashMap<Span, String>,
    binders: &BinderSpans,
) -> Vec<FnDataFlow> {
    let mut out = Vec::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            let is_prelude = prelude_names.contains(&f.name);
            let is_synthetic = f.name.contains('$');
            if is_prelude || is_synthetic {
                continue;
            }
            out.push(analyze_fn(f, types, binders));
        }
    }
    out
}

/// A single in-scope binding while walking (an entry in the scope stack). `id`
/// indexes into the `bindings` accumulator so a `Var` use is attributed to the
/// exact innermost binding.
#[derive(Clone)]
struct ScopeEntry {
    name: String,
    id: usize,
    def: (u32, u32),
}

/// The walk state: the accumulating binding list and a lexical scope stack of
/// in-scope entries. A `Var(name)` read resolves to the LAST (innermost) entry
/// with that name; pushing a binding whose name already has an in-scope entry
/// records a [`Shadow`].
struct DfWalk<'a> {
    types: &'a HashMap<Span, String>,
    binders: &'a BinderSpans,
    bindings: Vec<Binding>,
    scope: Vec<ScopeEntry>,
    shadows: Vec<Shadow>,
}

impl<'a> DfWalk<'a> {
    fn lookup_type(&self, span: Span) -> Option<String> {
        if span.is_none() {
            None
        } else {
            self.types.get(&span).cloned()
        }
    }

    /// Introduce a new binding into both the accumulator and the live scope,
    /// recording a shadow if an in-scope binding of the same name already exists.
    /// Returns the new binding's id.
    fn declare(
        &mut self,
        name: &str,
        kind: BindKind,
        def_span: Span,
        ty: Option<String>,
    ) -> usize {
        let def = (def_span.start_line, def_span.start_col);
        if let Some(outer) = self.scope.iter().rev().find(|e| e.name == name) {
            self.shadows.push(Shadow {
                name: name.to_string(),
                inner: def,
                outer: outer.def,
            });
        }
        let id = self.bindings.len();
        self.bindings.push(Binding {
            name: name.to_string(),
            kind,
            def,
            def_span,
            ty,
            uses: Vec::new(),
            unused: true,
        });
        self.scope.push(ScopeEntry { name: name.to_string(), id, def });
        id
    }

    /// Record a read of `name` at `span` against the innermost in-scope binding,
    /// if any. A name that resolves to no local binding (a top-level function,
    /// prelude function, builtin, or unknown) is not a local use and is ignored.
    fn use_var(&mut self, name: &str, span: Span) {
        if span.is_none() {
            return;
        }
        if let Some(e) = self.scope.iter().rev().find(|e| e.name == name) {
            let id = e.id;
            self.bindings[id].uses.push((span.start_line, span.start_col));
        }
    }

    /// Bind every variable a match pattern introduces into scope (a match
    /// binder), looking up each binder's def span from the side table (keyed by
    /// the arm body span) and its type from the span->type table at the def site.
    fn bind_pattern(&mut self, pat: &Pattern, arm_body: Span) {
        match pat {
            Pattern::Var(name) => {
                let def = self
                    .binders
                    .match_binders
                    .get(&(arm_body, name.clone()))
                    .copied()
                    .unwrap_or(Span::none());
                let ty = self.lookup_type(def);
                self.declare(name, BindKind::MatchBinder, def, ty);
            }
            Pattern::Ctor(_, subs) => {
                for s in subs {
                    self.bind_pattern(s, arm_body);
                }
            }
            Pattern::Record(_, fields) => {
                for (_, sub) in fields {
                    match sub {
                        Pattern::Var(v) => {
                            let def = self
                                .binders
                                .match_binders
                                .get(&(arm_body, v.clone()))
                                .copied()
                                .unwrap_or(Span::none());
                            let ty = self.lookup_type(def);
                            self.declare(v, BindKind::MatchBinder, def, ty);
                        }
                        _ => self.bind_pattern(sub, arm_body),
                    }
                }
            }
            Pattern::Wild | Pattern::Int(_) | Pattern::Bool(_) => {}
        }
    }

    fn walk(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Var(name) => self.use_var(name, e.span),
            ExprKind::Call(name, args) => {
                // The callee NAME might resolve to a function-valued local binding
                // (`let f = ..; f(x)`) — that is a read of the local. A call to a
                // top-level function/builtin resolves to no local and is ignored.
                self.use_var(name, e.span);
                for a in args {
                    self.walk(a);
                }
            }
            ExprKind::Ctor(_, args) => {
                for a in args {
                    self.walk(a);
                }
            }
            ExprKind::Record(_, fields) => {
                for (_, v) in fields {
                    self.walk(v);
                }
            }
            ExprKind::Field(obj, _) => self.walk(obj),
            ExprKind::Update(base, updates) => {
                self.walk(base);
                for (_, v) in updates {
                    self.walk(v);
                }
            }
            ExprKind::Lambda(params, body, _) => {
                let depth = self.scope.len();
                for (i, (name, ty)) in params.iter().enumerate() {
                    let def = self
                        .binders
                        .lambda_params
                        .get(&(body.span, i))
                        .copied()
                        .unwrap_or(Span::none());
                    // A lambda parameter's declared/inferred annotation is on the
                    // AST; render it directly (the span table rarely keys binders).
                    let rendered = render_ty(ty).or_else(|| self.lookup_type(def));
                    self.declare(name, BindKind::LambdaParam, def, rendered);
                }
                self.walk(body);
                self.scope.truncate(depth);
            }
            ExprKind::Apply(callee, args, _) => {
                self.walk(callee);
                for a in args {
                    self.walk(a);
                }
            }
            ExprKind::Unary(_, inner) => self.walk(inner),
            ExprKind::Binary(_, lhs, rhs) => {
                self.walk(lhs);
                self.walk(rhs);
            }
            ExprKind::If(c, t, e2) => {
                self.walk(c);
                self.walk(t);
                self.walk(e2);
            }
            ExprKind::Match(scrut, arms) => {
                self.walk(scrut);
                for arm in arms {
                    let depth = self.scope.len();
                    self.bind_pattern(&arm.pat, arm.body.span);
                    self.walk(&arm.body);
                    self.scope.truncate(depth);
                }
            }
            ExprKind::Block(stmts, last) => {
                let depth = self.scope.len();
                for s in stmts {
                    match s {
                        Stmt::Let(name, ann, v) => {
                            // The RHS is evaluated in the PRE-binding scope.
                            self.walk(v);
                            let def = self
                                .binders
                                .lets
                                .get(&v.span)
                                .copied()
                                .unwrap_or(Span::none());
                            // A `let`'s type: prefer an explicit annotation, else
                            // the inferred type of its RHS value expression.
                            let ty = ann
                                .as_ref()
                                .and_then(render_ty)
                                .or_else(|| self.lookup_type(v.span))
                                .or_else(|| self.lookup_type(def));
                            self.declare(name, BindKind::Let, def, ty);
                        }
                        Stmt::Expr(ex) => self.walk(ex),
                    }
                }
                self.walk(last);
                self.scope.truncate(depth);
            }
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Bool(_)
            | ExprKind::Str(_)
            | ExprKind::Unit => {}
        }
    }
}

/// Render a `Ty` to a clean type string, returning `None` for a bare placeholder
/// type variable (an unsolved lambda-parameter annotation), which would be noise.
fn render_ty(t: &Ty) -> Option<String> {
    match t {
        Ty::Var(_) => None,
        _ => Some(crate::typeck::show(t)),
    }
}

/// Compute the data-flow of a single function: seed the scope with its
/// parameters, walk the body collecting uses, then finalize each binding's
/// `unused` flag and the derived dead/shadow lists.
fn analyze_fn(f: &FnDecl, types: &HashMap<Span, String>, binders: &BinderSpans) -> FnDataFlow {
    let mut w = DfWalk {
        types,
        binders,
        bindings: Vec::new(),
        scope: Vec::new(),
        shadows: Vec::new(),
    };
    // Parameters: declared type comes straight off the AST; def span is on Param.
    for p in &f.params {
        let ty = render_ty(&p.ty);
        w.declare(&p.name, BindKind::Param, p.span, ty);
    }
    w.walk(&f.body);

    // Finalize: sort/de-dup each binding's uses and set `unused`. When a binding
    // got no type from its def span (common for match binders, whose def span is
    // not keyed in the span->type table), fall back to the inferred type at its
    // first USE site — a read of the variable is typed in the table.
    for b in &mut w.bindings {
        b.uses.sort_unstable();
        b.uses.dedup();
        b.unused = b.uses.is_empty();
        if b.ty.is_none() {
            if let Some(&(ul, uc)) = b.uses.first() {
                b.ty = types
                    .iter()
                    .find(|(s, _)| s.start_line == ul && s.start_col == uc)
                    .map(|(_, t)| t.clone());
            }
        }
    }
    let unused_bindings: Vec<(String, (u32, u32))> = w
        .bindings
        .iter()
        .filter(|b| b.unused)
        .map(|b| (b.name.clone(), b.def))
        .collect();

    FnDataFlow {
        name: f.name.clone(),
        bindings: w.bindings,
        unused_bindings,
        shadows: w.shadows,
    }
}

/// JSON-encode a single `(line, col)` pair as `[line,col]`.
fn loc_json(loc: (u32, u32)) -> String {
    format!("[{},{}]", loc.0, loc.1)
}

/// JSON-encode an optional rendered type as a string or `null`.
fn opt_ty_json(t: &Option<String>) -> String {
    match t {
        Some(s) => format!("\"{}\"", json_escape(s)),
        None => "null".to_string(),
    }
}

impl FnDataFlow {
    /// Emit this function's data-flow as a stable JSON object. Schema:
    ///
    /// ```json
    /// {
    ///   "bindings": [
    ///     { "name": "x", "kind": "param", "def": [line,col], "type": "Int"|null,
    ///       "uses": [[line,col]], "use_count": N, "unused": false }
    ///   ],
    ///   "unused_bindings": [ {"name":"tmp","def":[line,col]} ],
    ///   "shadows": [ {"name":"x","def":[line,col],"shadows":[line,col]} ]
    /// }
    /// ```
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\"bindings\":[");
        for (i, b) in self.bindings.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"name\":\"{}\",\"kind\":\"{}\",\"def\":{},\"type\":{},\"uses\":[",
                json_escape(&b.name),
                b.kind.as_str(),
                loc_json(b.def),
                opt_ty_json(&b.ty),
            ));
            for (j, u) in b.uses.iter().enumerate() {
                if j > 0 {
                    s.push(',');
                }
                s.push_str(&loc_json(*u));
            }
            s.push_str(&format!(
                "],\"use_count\":{},\"unused\":{}}}",
                b.use_count(),
                b.unused
            ));
        }
        s.push_str("],\"unused_bindings\":[");
        for (i, (name, def)) in self.unused_bindings.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"name\":\"{}\",\"def\":{}}}",
                json_escape(name),
                loc_json(*def)
            ));
        }
        s.push_str("],\"shadows\":[");
        for (i, sh) in self.shadows.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"name\":\"{}\",\"def\":{},\"shadows\":{}}}",
                json_escape(&sh.name),
                loc_json(sh.inner),
                loc_json(sh.outer)
            ));
        }
        s.push_str("]}");
        s
    }

    /// A concise human one-liner summary of this function's data-flow for the
    /// `aria analyze` (non-`--json`) output, e.g.
    /// `unused: tmp (line 4); shadows: x (line 6 shadows line 2)`. Returns an
    /// empty string when there is nothing notable to report.
    pub fn to_human(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.unused_bindings.is_empty() {
            let items: Vec<String> = self
                .unused_bindings
                .iter()
                .map(|(n, d)| format!("`{}` (line {})", n, d.0))
                .collect();
            parts.push(format!("unused: {}", items.join(", ")));
        }
        if !self.shadows.is_empty() {
            let items: Vec<String> = self
                .shadows
                .iter()
                .map(|sh| format!("`{}` (line {} shadows line {})", sh.name, sh.inner.0, sh.outer.0))
                .collect();
            parts.push(format!("shadows: {}", items.join(", ")));
        }
        parts.join("; ")
    }
}

/// Compute the unused-binding warnings for a WELL-FORMED program directly from
/// its source `text` (the user program, before prelude wrapping). Returns an
/// empty list if the program does not lex/parse/type-check (lint warnings are
/// only meaningful on otherwise-clean code; hard errors are reported separately
/// by the caller). This is the entry point both `aria check --json` and the LSP
/// use to obtain warnings without re-implementing the pipeline.
pub fn warnings_for_source(text: &str) -> Vec<crate::diagnostics::Diagnostic> {
    let wrapped = crate::prelude::wrap(text);
    let toks = match crate::lexer::lex(&wrapped) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let (prog, binders) = match crate::parser::parse_with_binders(toks) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };
    let types = match crate::typeck::check_with_types(&prog) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let flows = analyze(&prog, &crate::analyze::prelude_fn_names(), &types, &binders);
    unused_let_warnings(&flows)
}

/// The unused-binding LINT codes (warning severity). Documented in
/// `docs/DIAGNOSTICS.md`.
///
/// `W0001` — unused variable: a `let` binding that is never read. This is the
/// default lint. Because Aria is single-assignment, an unused `let` is provably
/// dead (the value is computed and discarded), so the warning is exact.
pub const W_UNUSED_VARIABLE: &str = "W0001";
/// `W0002` — unused parameter. NOT emitted by default: a parameter can be
/// legitimately unused because a signature / interface / trait method requires
/// it, so flagging it would be noisy. Reserved for an explicit opt-in lint mode.
pub const W_UNUSED_PARAMETER: &str = "W0002";

/// Emit unused-binding WARNING diagnostics from per-function data-flow. By
/// default this flags only unused `let` bindings (`W0001`): a `let` whose value
/// is computed and never read is provably dead under Aria's single-assignment
/// model. Unused PARAMETERS (`W0002`) are deliberately NOT emitted — a parameter
/// is frequently unused for a legitimate reason (a signature/interface/trait
/// method requires it), so warning on them would be false-positive noise. Match
/// binders and lambda parameters are likewise skipped (pattern destructuring and
/// callback shapes routinely leave some unused). Warnings are advisory: they do
/// not affect any exit code (see `docs/DIAGNOSTICS.md`).
pub fn unused_let_warnings(flows: &[FnDataFlow]) -> Vec<crate::diagnostics::Diagnostic> {
    let mut out = Vec::new();
    for f in flows {
        for b in &f.bindings {
            if b.kind == BindKind::Let && b.unused {
                out.push(crate::diagnostics::Diagnostic::warning(
                    W_UNUSED_VARIABLE,
                    format!("unused variable `{}`", b.name),
                    b.def_span,
                    Some(f.name.clone()),
                ));
            }
        }
    }
    out
}

/// JSON-encode a whole-program data-flow result as an object keyed by function
/// name, each value a [`FnDataFlow::to_json`] object. Functions appear in source
/// order; the object is stable for diffable output.
pub fn to_json(flows: &[FnDataFlow]) -> String {
    let mut s = String::from("{");
    for (i, f) in flows.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\"{}\":{}", json_escape(&f.name), f.to_json()));
    }
    s.push('}');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flows_of(src: &str) -> Vec<FnDataFlow> {
        let toks = crate::lexer::lex(&crate::prelude::wrap(src)).expect("lex");
        let (prog, binders) = crate::parser::parse_with_binders(toks).expect("parse");
        let types = crate::typeck::check_with_types(&prog).expect("typeck");
        analyze(&prog, &crate::analyze::prelude_fn_names(), &types, &binders)
    }

    fn flow<'a>(fs: &'a [FnDataFlow], name: &str) -> &'a FnDataFlow {
        fs.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("no fn {}", name))
    }

    fn binding<'a>(f: &'a FnDataFlow, name: &str) -> &'a Binding {
        f.bindings.iter().find(|b| b.name == name).unwrap_or_else(|| panic!("no binding {}", name))
    }

    #[test]
    fn param_used_twice_records_two_uses() {
        let src = "fn add(x: Int) -> Int = x + x\nfn main() -> Int = add(1)\n";
        let fs = flows_of(src);
        let f = flow(&fs, "add");
        let x = binding(f, "x");
        assert_eq!(x.kind, BindKind::Param);
        assert_eq!(x.use_count(), 2, "x used twice: {:?}", x.uses);
        assert!(!x.unused);
        assert_eq!(x.ty.as_deref(), Some("Int"));
        // def points at the param binder on line 1.
        assert_eq!(x.def.0, 1);
    }

    #[test]
    fn unused_let_is_dead() {
        let src = "fn f() -> Int = { let tmp = 5; 1 }\nfn main() -> Int = f()\n";
        let fs = flows_of(src);
        let f = flow(&fs, "f");
        let tmp = binding(f, "tmp");
        assert_eq!(tmp.kind, BindKind::Let);
        assert!(tmp.unused, "tmp should be unused");
        assert_eq!(tmp.use_count(), 0);
        assert_eq!(tmp.ty.as_deref(), Some("Int"));
        assert_eq!(f.unused_bindings.len(), 1);
        assert_eq!(f.unused_bindings[0].0, "tmp");
    }

    #[test]
    fn used_let_records_use() {
        let src = "fn f() -> Int = { let y = 5; y + 1 }\nfn main() -> Int = f()\n";
        let fs = flows_of(src);
        let f = flow(&fs, "f");
        let y = binding(f, "y");
        assert!(!y.unused);
        assert_eq!(y.use_count(), 1);
        assert!(f.unused_bindings.is_empty());
    }

    #[test]
    fn inner_shadow_does_not_count_for_outer() {
        // Outer `x` (param) shadowed by an inner `let x`; the inner use binds to
        // the inner binding only. The outer param is then unused.
        let src = "\
fn g(x: Int) -> Int = {
  let x = 99;
  x + 1
}
fn main() -> Int = g(3)
";
        let fs = flows_of(src);
        let f = flow(&fs, "g");
        // Two bindings named `x`: the param and the inner let.
        let xs: Vec<&Binding> = f.bindings.iter().filter(|b| b.name == "x").collect();
        assert_eq!(xs.len(), 2);
        let param_x = xs.iter().find(|b| b.kind == BindKind::Param).unwrap();
        let let_x = xs.iter().find(|b| b.kind == BindKind::Let).unwrap();
        // The inner `x + 1` use binds to the LET, not the param.
        assert_eq!(let_x.use_count(), 1, "inner use binds to let-x");
        assert_eq!(param_x.use_count(), 0, "param x is shadowed + unused");
        assert!(param_x.unused);
        // And a shadow is reported (inner def line 2 shadows outer def line 1).
        assert_eq!(f.shadows.len(), 1);
        assert_eq!(f.shadows[0].name, "x");
        assert_eq!(f.shadows[0].inner.0, 2);
        assert_eq!(f.shadows[0].outer.0, 1);
    }

    #[test]
    fn lambda_param_and_match_binder() {
        let src = "\
type Box = | Box(Int)
fn h(b: Box) -> Int = {
  let f = \\(n: Int) -> n + 1;
  match b {
    Box(v) => f(v)
  }
}
fn main() -> Int = h(Box(2))
";
        let fs = flows_of(src);
        let f = flow(&fs, "h");
        let n = binding(f, "n");
        assert_eq!(n.kind, BindKind::LambdaParam);
        assert_eq!(n.use_count(), 1, "lambda param n used once");
        assert_eq!(n.ty.as_deref(), Some("Int"));
        let v = binding(f, "v");
        assert_eq!(v.kind, BindKind::MatchBinder);
        assert_eq!(v.use_count(), 1, "match binder v used once");
        // `f` (the let-bound lambda) is used once in the arm.
        let fb = binding(f, "f");
        assert_eq!(fb.use_count(), 1);
    }

    #[test]
    fn record_field_shorthand_binder() {
        let src = "\
type Point = { x: Int, y: Int }
fn px(p: Point) -> Int = match p { Point { x, y } => x }
fn main() -> Int = px(Point { x: 1, y: 2 })
";
        let fs = flows_of(src);
        let f = flow(&fs, "px");
        let x = binding(f, "x");
        assert_eq!(x.kind, BindKind::MatchBinder);
        assert_eq!(x.use_count(), 1);
        // `y` is bound by the shorthand but never used -> dead.
        let y = binding(f, "y");
        assert!(y.unused, "y should be unused");
        assert!(f.unused_bindings.iter().any(|(n, _)| n == "y"));
    }

    #[test]
    fn json_is_well_formed() {
        let src = "fn f(x: Int) -> Int = { let tmp = 5; x }\nfn main() -> Int = f(1)\n";
        let fs = flows_of(src);
        let j = to_json(&fs);
        assert!(j.starts_with('{') && j.ends_with('}'));
        assert!(j.contains("\"kind\":\"param\""));
        assert!(j.contains("\"kind\":\"let\""));
        assert!(j.contains("\"unused\":true"));
        assert!(j.contains("\"use_count\":"));
    }

    #[test]
    fn unused_let_warning_emitted_params_not() {
        // An unused `let` produces a W0001 warning; an unused PARAM does NOT (it
        // may be legitimately required by a signature) — no false-positive noise.
        let src = "fn f(unused_p: Int) -> Int = { let tmp = 5; 1 }\nfn main() -> Int = f(2)\n";
        let ws = warnings_for_source(src);
        assert_eq!(ws.len(), 1, "only the let warns: {:?}", ws);
        assert_eq!(ws[0].code, "W0001");
        assert_eq!(ws[0].severity, "warning");
        assert!(ws[0].message.contains("tmp"));
        assert_eq!(ws[0].function.as_deref(), Some("f"));
        // Precise span on the binder.
        assert_eq!(ws[0].line, Some(1));
    }

    #[test]
    fn clean_program_has_no_warnings() {
        let src = "fn f() -> Int = { let y = 5; y }\nfn main() -> Int = f()\n";
        assert!(warnings_for_source(src).is_empty());
    }

    #[test]
    fn warnings_empty_on_ill_typed_source() {
        // Lint only runs on well-formed code; a type error yields no warnings here
        // (errors are reported separately by the caller).
        let src = "fn f() -> Int = true\n";
        assert!(warnings_for_source(src).is_empty());
    }

    #[test]
    fn shadow_in_nested_block_only() {
        // A use of the OUTER var before the inner shadow binds to the outer.
        let src = "\
fn k(x: Int) -> Int = {
  let a = x + 1;
  let x = a;
  x
}
fn main() -> Int = k(1)
";
        let fs = flows_of(src);
        let f = flow(&fs, "k");
        let param_x = f.bindings.iter().find(|b| b.name == "x" && b.kind == BindKind::Param).unwrap();
        let let_x = f.bindings.iter().find(|b| b.name == "x" && b.kind == BindKind::Let).unwrap();
        // `x + 1` (in `let a`) is BEFORE the inner `let x`, so binds to the param.
        assert_eq!(param_x.use_count(), 1, "param x used once before shadow");
        // The final `x` binds to the let.
        assert_eq!(let_x.use_count(), 1);
        assert!(!param_x.unused);
    }
}
