//! Recursive-descent parser with Pratt-style precedence for binary operators.
//!
//! The grammar is LL(1) apart from the trivial "ident, then maybe `(`" lookahead,
//! which keeps it friendly to grammar-constrained decoding later on.

use crate::ast::*;
use crate::lexer::{Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
    /// Generic type parameters of the function currently being parsed, so that
    /// type annotations inside its body (e.g. `let y: T = ...`) resolve `T` to a
    /// type variable rather than a bogus nullary named type.
    type_params: Vec<String>,
    /// Counter for naming the inferred type variable of an unannotated lambda
    /// parameter (`\x -> ...`). Kept unique per parse.
    lambda_counter: usize,
    /// When true, an `Upper {` is NOT parsed as a record literal — used while
    /// parsing the head of `if`/`match`, where the `{` opens the block/arms (so
    /// `match Nil { .. }` is a match on `Nil`, not a record literal `Nil { .. }`).
    /// Reset to false inside any delimited sub-expression (parens, args, blocks,
    /// array/record literals), so `if (P { x: 1 }).x { .. }` still works.
    no_record_literal: bool,
    /// Tuple arities used anywhere in the source, so `parse_program` can inject a
    /// synthetic `$TupleN` ADT for exactly those (and no others).
    tuple_arities: std::collections::HashSet<usize>,
    /// Current recursion depth of the recursive-descent expression/type parser.
    /// Bounded by `MAX_NESTING_DEPTH` so pathological nesting (e.g. 100k nested
    /// parens) yields a CLEAN parse error instead of overflowing the Rust stack
    /// (which would abort the process with SIGSEGV/exit 134 and break the
    /// "`--json` always emits valid JSON or a clean error" contract).
    depth: usize,
    /// Side table of precise LAMBDA-PARAMETER binder spans — the one binder kind
    /// the AST cannot carry a span for directly (see [`crate::ast::BinderSpans`]).
    /// Pure analysis METADATA, consumed only by the data-flow analyzer.
    binder_spans: BinderSpans,
}

/// Maximum nesting depth for the expression / type parser. Set far above any
/// real program (deeply nested but human-authored code is nowhere near this)
/// while still bounding the Rust call stack well below an overflow.
const MAX_NESTING_DEPTH: usize = 2048;

/// RAII guard returned by `Parser::enter_depth`: decrements the parser's nesting
/// counter when it drops, so the depth is restored on every control-flow path
/// (normal return or `?` early-exit) without manual bookkeeping. It holds a raw
/// pointer (not a `&mut`) so the parser body can keep using `&mut self` freely
/// while the guard is alive; the pointer targets `self.depth`, which outlives
/// the guard (the guard is a local in a `&mut self` method), so the deref is
/// sound and single-threaded.
struct DepthGuard {
    depth: *mut usize,
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        // SAFETY: `depth` points at the live `Parser::depth` field for the
        // duration of the borrowing method call (the guard never outlives it),
        // and the parser is single-threaded, so this is the only access.
        unsafe {
            *self.depth -= 1;
        }
    }
}

fn is_upper(name: &str) -> bool {
    name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
}

/// The synthetic ADT name/constructor for an `n`-tuple (`$` prefix can't collide
/// with a user identifier).
fn tuple_type_name(n: usize) -> String {
    format!("$Tuple{}", n)
}

/// The synthetic generic ADT declaration for an `n`-tuple, e.g.
/// `type $Tuple2[$t0, $t1] = | $Tuple2($t0, $t1)`. Tuples desugar into values of
/// these, so they flow through the whole existing ADT machinery.
fn synthetic_tuple_type(n: usize) -> Item {
    let params: Vec<String> = (0..n).map(|i| format!("$t{}", i)).collect();
    let fields: Vec<Ty> = params.iter().map(|p| Ty::Var(p.clone())).collect();
    Item::Type(TypeDecl {
        name: tuple_type_name(n),
        params,
        variants: vec![Variant { name: tuple_type_name(n), fields, field_names: None }],
    })
}

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser {
            toks,
            pos: 0,
            type_params: Vec::new(),
            lambda_counter: 0,
            no_record_literal: false,
            tuple_arities: std::collections::HashSet::new(),
            depth: 0,
            binder_spans: BinderSpans::default(),
        }
    }

    /// The precise span of the current (not-yet-consumed) token, as a [`Span`].
    /// Used to record a binder's definition site (the identifier's exact extent).
    fn cur_span(&self) -> Span {
        let t = &self.toks[self.pos];
        Span {
            start_line: t.line as u32,
            start_col: t.col as u32,
            end_line: t.end_line as u32,
            end_col: t.end_col as u32,
        }
    }

    /// Enter one level of expression/type recursion, erroring cleanly if the
    /// nesting limit is exceeded. The returned guard decrements on drop, so the
    /// counter is correct on every path (including early `?` returns).
    fn enter_depth(&mut self) -> Result<DepthGuard, String> {
        self.depth += 1;
        if self.depth > MAX_NESTING_DEPTH {
            // Decrement before bailing so a recovering caller sees a consistent
            // counter (the guard is not constructed on this path).
            self.depth -= 1;
            return Err(format!(
                "line {}: expression nesting too deep (limit {})",
                self.line(),
                MAX_NESTING_DEPTH
            ));
        }
        Ok(DepthGuard { depth: &mut self.depth as *mut usize })
    }

    /// Record that an `n`-tuple is used and return its synthetic ADT name.
    fn tuple_name(&mut self, n: usize) -> String {
        self.tuple_arities.insert(n);
        tuple_type_name(n)
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn line(&self) -> usize {
        self.toks[self.pos].line
    }

    /// The (line, col) START of the current (not-yet-consumed) token — the start
    /// of whatever node is about to be parsed.
    fn here(&self) -> (usize, usize) {
        let t = &self.toks[self.pos];
        (t.line, t.col)
    }

    /// The (line, col) END (one past the last char) of the MOST RECENTLY consumed
    /// token, i.e. `toks[pos-1]`. Used as the end of a node whose last token was
    /// just consumed. Falls back to the current token's start at position 0.
    fn prev_end(&self) -> (usize, usize) {
        if self.pos == 0 {
            let t = &self.toks[0];
            (t.line, t.col)
        } else {
            let t = &self.toks[self.pos - 1];
            (t.end_line, t.end_col)
        }
    }

    /// Build an `Expr` from a `kind` and a start position captured BEFORE parsing
    /// began, ending at the most recently consumed token. This is the canonical
    /// way the parser attaches a precise span to a node.
    fn spanned(&self, start: (usize, usize), kind: ExprKind) -> Expr {
        let (el, ec) = self.prev_end();
        Expr::new(
            kind,
            Span {
                start_line: start.0 as u32,
                start_col: start.1 as u32,
                end_line: el as u32,
                end_col: ec as u32,
            },
        )
    }

    /// Build a `Pattern` from a `kind` and a start position captured BEFORE the
    /// pattern was parsed, ending at the most recently consumed token — the same
    /// span discipline `spanned` uses for expressions.
    fn spanned_pat(&self, start: (usize, usize), kind: PatternKind) -> Pattern {
        let (el, ec) = self.prev_end();
        Pattern::new(
            kind,
            Span {
                start_line: start.0 as u32,
                start_col: start.1 as u32,
                end_line: el as u32,
                end_col: ec as u32,
            },
        )
    }

    /// Build a `Stmt` from a `kind` and a start position captured BEFORE the
    /// statement was parsed, ending at the most recently consumed token (the
    /// terminating `;` for a `let`/expression statement).
    fn spanned_stmt(&self, start: (usize, usize), kind: StmtKind) -> Stmt {
        let (el, ec) = self.prev_end();
        Stmt::new(
            kind,
            Span {
                start_line: start.0 as u32,
                start_col: start.1 as u32,
                end_line: el as u32,
                end_col: ec as u32,
            },
        )
    }

    fn advance(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, want: &Tok) -> Result<(), String> {
        if self.peek() == want {
            self.advance();
            Ok(())
        } else {
            Err(format!(
                "line {}: expected {:?}, found {:?}",
                self.line(),
                want,
                self.peek()
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.peek().clone() {
            Tok::Ident(s) => {
                self.advance();
                Ok(s)
            }
            other => Err(format!("line {}: expected identifier, found {:?}", self.line(), other)),
        }
    }

    // ---- top level -------------------------------------------------------

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut items = Vec::new();
        let mut interfaces: Vec<InterfaceDecl> = Vec::new();
        let mut impls: Vec<ImplDecl> = Vec::new();
        while *self.peek() != Tok::Eof {
            match self.peek() {
                Tok::Interface => interfaces.push(self.parse_interface()?),
                Tok::Impl => impls.push(self.parse_impl()?),
                _ => items.push(self.parse_item()?),
            }
        }
        // Prepend a synthetic tuple ADT for each arity ACTUALLY used (`(a, b)`
        // desugars to `$Tuple2(a, b)`). Tuples reuse the entire ADT pipeline —
        // generics, reference counting, every backend — with no downstream
        // special-casing. Injecting only used arities keeps tuple-free programs
        // unchanged (so they still take the no-generics fast path).
        let mut arities: Vec<usize> = self.tuple_arities.iter().cloned().collect();
        arities.sort_unstable();
        let mut all: Vec<Item> = arities.into_iter().map(synthetic_tuple_type).collect();
        all.extend(items);
        // Lower interfaces + impls to ordinary `Item::Fn`s (mangled impl methods
        // + per-method dispatchers) so no downstream stage ever sees a new `Item`
        // variant. This is the keystone of the trait design: STATIC dispatch via
        // monomorphization in the compiled backends, runtime constructor dispatch
        // in the interpreter — both routed through the same lowered functions.
        crate::traits::lower(&mut all, &interfaces, &impls)?;
        Ok(Program { items: all })
    }

    fn parse_item(&mut self) -> Result<Item, String> {
        match self.peek() {
            Tok::Fn | Tok::Pure => Ok(Item::Fn(self.parse_fn()?)),
            Tok::Type => Ok(Item::Type(self.parse_type_decl()?)),
            other => Err(format!(
                "line {}: expected `fn`, `pure`, `type`, `interface`, or `impl`, found {:?}",
                self.line(),
                other
            )),
        }
    }

    /// Parse `interface Name[T] { fn m(self: T, ..) -> R, .. }`. The single type
    /// parameter is `Self` (the implementing type). Each method is a SIGNATURE
    /// only (no body); methods are separated by `,` or newlines and the list may
    /// have a trailing comma.
    fn parse_interface(&mut self) -> Result<InterfaceDecl, String> {
        self.expect(&Tok::Interface)?;
        let name = self.expect_ident()?;
        let params = self.parse_type_params()?;
        if params.len() != 1 {
            return Err(format!(
                "line {}: interface `{}` must declare exactly one type parameter (the implementing type), e.g. `interface {}[T]`",
                self.line(),
                name,
                name
            ));
        }
        let self_param = params.into_iter().next().unwrap();
        let tps = vec![self_param.clone()];
        self.expect(&Tok::LBrace)?;
        let mut methods = Vec::new();
        while *self.peek() != Tok::RBrace {
            self.expect(&Tok::Fn)?;
            let mname = self.expect_ident()?;
            self.expect(&Tok::LParen)?;
            let mut mparams = Vec::new();
            if *self.peek() != Tok::RParen {
                loop {
                    let pspan = self.cur_span();
                    let pname = self.expect_ident()?;
                    self.expect(&Tok::Colon)?;
                    let ty = self.parse_type(&tps)?;
                    mparams.push(Param { name: pname, ty, span: pspan });
                    if *self.peek() == Tok::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Tok::RParen)?;
            self.expect(&Tok::Arrow)?;
            let ret = self.parse_type(&tps)?;
            methods.push(MethodSig { name: mname, params: mparams, ret });
            if *self.peek() == Tok::Comma {
                self.advance();
            }
        }
        self.expect(&Tok::RBrace)?;
        if methods.is_empty() {
            return Err(format!(
                "line {}: interface `{}` must declare at least one method",
                self.line(),
                name
            ));
        }
        Ok(InterfaceDecl { name, self_param, methods })
    }

    /// Parse `impl Trait for Head { fn m(self: Head, ..) -> R = body, .. }`.
    /// The head type is a concrete named type (no generic impl heads); each
    /// method is a full function declaration whose body is provided.
    fn parse_impl(&mut self) -> Result<ImplDecl, String> {
        self.expect(&Tok::Impl)?;
        let trait_name = self.expect_ident()?;
        // `for` is spelled as an identifier (no dedicated keyword needed).
        match self.peek().clone() {
            Tok::Ident(s) if s == "for" => {
                self.advance();
            }
            other => {
                return Err(format!(
                    "line {}: expected `for` in `impl {} for ..`, found {:?}",
                    self.line(),
                    trait_name,
                    other
                ))
            }
        }
        let head_type = self.expect_ident()?;
        self.expect(&Tok::LBrace)?;
        let mut methods = Vec::new();
        while *self.peek() != Tok::RBrace {
            methods.push(self.parse_fn()?);
            if *self.peek() == Tok::Comma {
                self.advance();
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(ImplDecl { trait_name, head_type, methods })
    }

    fn parse_fn(&mut self) -> Result<FnDecl, String> {
        // 1-based source line of the function (the `pure`/`fn` keyword). Captured
        // before consuming any tokens so it points at the function's start.
        let line = self.line();
        // Optional `pure` annotation before `fn`. Canonical form: `pure` may
        // only appear immediately before `fn` and nowhere else.
        let pure = if *self.peek() == Tok::Pure {
            self.advance();
            true
        } else {
            false
        };
        self.expect(&Tok::Fn)?;
        let name = self.expect_ident()?;
        let (type_params, bounds) = self.parse_type_params_bounded()?;
        self.expect(&Tok::LParen)?;
        let mut params = Vec::new();
        if *self.peek() != Tok::RParen {
            loop {
                let pspan = self.cur_span();
                let pname = self.expect_ident()?;
                self.expect(&Tok::Colon)?;
                let ty = self.parse_type(&type_params)?;
                params.push(Param { name: pname, ty, span: pspan });
                if *self.peek() == Tok::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen)?;
        self.expect(&Tok::Arrow)?;
        let ret = self.parse_type(&type_params)?;
        self.expect(&Tok::Eq)?;
        // Make the generics visible to annotations inside the body, then clear.
        self.type_params = type_params.clone();
        let body = self.parse_expr(0)?;
        self.type_params = Vec::new();
        Ok(FnDecl {
            name,
            line,
            pure,
            type_params,
            bounds,
            params,
            ret,
            body,
        })
    }

    // Parse an optional `[T, U: Trait, ...]` type-parameter list, returning the
    // parameter names and any `param: Trait` bounds. Bounds are how a generic
    // function declares it may call a trait's methods on that parameter (static
    // dispatch resolved at monomorphization). A parameter may carry at most one
    // bound here (the surface syntax `T: Show`); multiple traits are not yet
    // supported.
    fn parse_type_params_bounded(&mut self) -> Result<(Vec<String>, Vec<(String, String)>), String> {
        let mut params = Vec::new();
        let mut bounds = Vec::new();
        if *self.peek() == Tok::LBracket {
            self.advance();
            if *self.peek() != Tok::RBracket {
                loop {
                    let p = self.expect_ident()?;
                    if *self.peek() == Tok::Colon {
                        self.advance();
                        let tr = self.expect_ident()?;
                        bounds.push((p.clone(), tr));
                    }
                    params.push(p);
                    if *self.peek() == Tok::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Tok::RBracket)?;
        }
        Ok((params, bounds))
    }

    // Parse an optional `[T, U, ...]` type-parameter list on a declaration.
    fn parse_type_params(&mut self) -> Result<Vec<String>, String> {
        let mut params = Vec::new();
        if *self.peek() == Tok::LBracket {
            self.advance();
            if *self.peek() != Tok::RBracket {
                loop {
                    params.push(self.expect_ident()?);
                    if *self.peek() == Tok::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Tok::RBracket)?;
        }
        Ok(params)
    }

    fn parse_type_decl(&mut self) -> Result<TypeDecl, String> {
        self.expect(&Tok::Type)?;
        let name = self.expect_ident()?;
        let params = self.parse_type_params()?;
        self.expect(&Tok::Eq)?;
        // A `{` after `=` introduces a RECORD type: a single constructor (named
        // after the type) with named fields. Otherwise it is a sum type, whose
        // first variant must be preceded by `|` (canonical form, one spelling).
        if *self.peek() == Tok::LBrace {
            self.advance();
            let mut fields = Vec::new();
            let mut names = Vec::new();
            if *self.peek() != Tok::RBrace {
                loop {
                    let fname = self.expect_ident()?;
                    self.expect(&Tok::Colon)?;
                    let fty = self.parse_type(&params)?;
                    names.push(fname);
                    fields.push(fty);
                    if *self.peek() == Tok::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Tok::RBrace)?;
            return Ok(TypeDecl {
                name: name.clone(),
                params,
                variants: vec![Variant { name, fields, field_names: Some(names) }],
            });
        }
        self.expect(&Tok::Pipe)?;
        let mut variants = Vec::new();
        loop {
            let vname = self.expect_ident()?;
            let mut fields = Vec::new();
            if *self.peek() == Tok::LParen {
                self.advance();
                if *self.peek() != Tok::RParen {
                    loop {
                        fields.push(self.parse_type(&params)?);
                        if *self.peek() == Tok::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RParen)?;
            }
            variants.push(Variant { name: vname, fields, field_names: None });
            if *self.peek() == Tok::Pipe {
                self.advance();
            } else {
                break;
            }
        }
        Ok(TypeDecl {
            name,
            params,
            variants,
        })
    }

    // Parse a type expression. `tparams` lists the generic parameters in scope;
    // a bare name found there becomes a `Ty::Var`, builtins map to their concrete
    // types, and anything else is a `Ty::Named` with optional `[..]` arguments.
    fn parse_type(&mut self, tparams: &[String]) -> Result<Ty, String> {
        // Bound the recursion depth (nested function/parameterised types) so
        // pathological nesting yields a clean error, not a stack overflow.
        let _guard = self.enter_depth()?;
        // Function type: `(T1, T2, ...) -> R`. The leading `(` unambiguously
        // distinguishes it from a named/builtin type (which starts with an
        // identifier).
        if *self.peek() == Tok::LParen {
            self.advance();
            let mut params = Vec::new();
            if *self.peek() != Tok::RParen {
                loop {
                    params.push(self.parse_type(tparams)?);
                    if *self.peek() == Tok::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Tok::RParen)?;
            // `(T1, ..) -> R` is a function type; without the arrow, the paren
            // list is a TUPLE type `(A, B, ..)` (>= 2 elements), a grouped type
            // `(T)` (1 element), or `()` = Unit (0 elements).
            if *self.peek() == Tok::Arrow {
                self.advance();
                let ret = self.parse_type(tparams)?;
                return Ok(Ty::Fn(params, Box::new(ret)));
            }
            return Ok(match params.len() {
                0 => Ty::Unit,
                1 => params.into_iter().next().unwrap(),
                n => Ty::Named(self.tuple_name(n), params),
            });
        }
        let name = self.expect_ident()?;
        // A bare builtin name (no brackets) stays a concrete builtin type.
        let builtin = match name.as_str() {
            "Int" => Some(Ty::Int),
            "Float" => Some(Ty::Float),
            "Bool" => Some(Ty::Bool),
            "String" => Some(Ty::Str),
            "Unit" => Some(Ty::Unit),
            _ => None,
        };
        let mut args = Vec::new();
        if *self.peek() == Tok::LBracket {
            self.advance();
            if *self.peek() != Tok::RBracket {
                loop {
                    args.push(self.parse_type(tparams)?);
                    if *self.peek() == Tok::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Tok::RBracket)?;
        }
        if args.is_empty() {
            if let Some(b) = builtin {
                return Ok(b);
            }
            if tparams.iter().any(|p| p == &name) {
                return Ok(Ty::Var(name));
            }
        }
        Ok(Ty::Named(name, args))
    }

    // ---- expressions (Pratt) --------------------------------------------

    fn bin_power(tok: &Tok) -> Option<(BinOp, u8, u8)> {
        // (operator, left binding power, right binding power); left-assoc.
        Some(match tok {
            Tok::OrOr => (BinOp::Or, 1, 2),
            Tok::AndAnd => (BinOp::And, 3, 4),
            Tok::EqEq => (BinOp::Eq, 5, 6),
            Tok::NotEq => (BinOp::Ne, 5, 6),
            Tok::Lt => (BinOp::Lt, 5, 6),
            Tok::Le => (BinOp::Le, 5, 6),
            Tok::Gt => (BinOp::Gt, 5, 6),
            Tok::Ge => (BinOp::Ge, 5, 6),
            Tok::Plus => (BinOp::Add, 7, 8),
            Tok::Minus => (BinOp::Sub, 7, 8),
            Tok::Star => (BinOp::Mul, 9, 10),
            Tok::Slash => (BinOp::Div, 9, 10),
            Tok::Percent => (BinOp::Mod, 9, 10),
            _ => return None,
        })
    }

    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, String> {
        // Bound the recursive-descent depth: every nested sub-expression (parens,
        // operands, args, blocks) re-enters here, so a pathological nest (e.g.
        // 100k nested `(`) hits the limit and returns a clean parse error instead
        // of overflowing the Rust stack (which would SIGSEGV / exit 134 and break
        // the `--json` "valid JSON or clean error" contract).
        let _guard = self.enter_depth()?;
        // The left operand's start is the start of the whole binary expression.
        let start = self.here();
        let mut lhs = self.parse_unary()?;
        loop {
            let (op, lbp, rbp) = match Parser::bin_power(self.peek()) {
                Some(x) => x,
                None => break,
            };
            if lbp < min_bp {
                break;
            }
            self.advance(); // consume operator
            let rhs = self.parse_expr(rbp)?;
            lhs = self.spanned(start, ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        let start = self.here();
        match self.peek() {
            Tok::Minus => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(self.spanned(start, ExprKind::Unary(UnOp::Neg, Box::new(inner))))
            }
            Tok::Bang => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(self.spanned(start, ExprKind::Unary(UnOp::Not, Box::new(inner))))
            }
            Tok::Backslash => self.parse_lambda(),
            _ => self.parse_postfix(),
        }
    }

    // A lambda: `\x -> body` or `\(x: Int, y: Int) -> body`. Canonical form uses
    // a leading backslash. The single-parameter shorthand `\x -> ...` infers no
    // type, so its parameter type is left as a fresh type variable for the
    // checker to solve from use; the parenthesized form carries explicit
    // annotations. The body extends as far right as possible.
    fn parse_lambda(&mut self) -> Result<Expr, String> {
        let start = self.here();
        self.expect(&Tok::Backslash)?;
        let mut params = Vec::new();
        // Parallel record of each parameter binder's precise def span, recorded
        // below (keyed by the lambda body span) for the data-flow analyzer.
        let mut pspans: Vec<Span> = Vec::new();
        if *self.peek() == Tok::LParen {
            self.advance();
            if *self.peek() != Tok::RParen {
                loop {
                    let pspan = self.cur_span();
                    let pname = self.expect_ident()?;
                    self.expect(&Tok::Colon)?;
                    let tps = self.type_params.clone();
                    let ty = self.parse_type(&tps)?;
                    params.push((pname, ty));
                    pspans.push(pspan);
                    if *self.peek() == Tok::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Tok::RParen)?;
        } else {
            // Single unannotated parameter shorthand: `\x -> body`.
            let pspan = self.cur_span();
            let pname = self.expect_ident()?;
            params.push((pname, Ty::Var(self.fresh_lambda_var())));
            pspans.push(pspan);
        }
        self.expect(&Tok::Arrow)?;
        let body = self.parse_expr(0)?;
        // Record each lambda parameter's def span, keyed by `(body span, index)`.
        if !body.span.is_none() {
            for (i, ps) in pspans.into_iter().enumerate() {
                self.binder_spans.lambda_params.insert((body.span, i), ps);
            }
        }
        Ok(self.spanned(start, ExprKind::Lambda(params, Box::new(body), None)))
    }

    // Parse an atom, then any trailing `(args)` applications. A trailing call on
    // a bare lowercase identifier stays `Expr::Call` (top-level call by name);
    // any other callee (a lambda, a parenthesized expression, or a further
    // application) becomes `Expr::Apply`.
    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let start = self.here();
        let mut e = self.parse_atom()?;
        // `parse_atom` already consumes the first `(args)` for `name(args)`
        // (yielding a by-name Call/Ctor). Any further `(args)` here — or an
        // application of a non-name expression such as `(\x -> ...)(5)` — is a
        // general application of the callee value. A trailing `[i]` is array
        // indexing, desugared to `array_get(e, i)`. Both chain: `f(x)[0][1]`.
        loop {
            match self.peek() {
                Tok::LParen => {
                    let args = self.parse_args()?;
                    e = self.spanned(start, ExprKind::Apply(Box::new(e), args, None));
                }
                Tok::LBracket => {
                    self.advance(); // `[`
                    let idx = self.parse_sub_expr(0)?;
                    self.expect(&Tok::RBracket)?;
                    e = self.spanned(start, ExprKind::Call("array_get".to_string(), vec![e, idx]));
                }
                Tok::Dot => {
                    self.advance(); // `.`
                    let field = self.expect_ident()?;
                    e = self.spanned(start, ExprKind::Field(Box::new(e), field));
                }
                _ => break,
            }
        }
        Ok(e)
    }

    // A fresh, source-invisible type-variable name for an unannotated lambda
    // parameter. Prefixed so it never collides with a user generic parameter.
    fn fresh_lambda_var(&mut self) -> String {
        let n = self.lambda_counter;
        self.lambda_counter += 1;
        format!("$lam{}", n)
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, String> {
        self.expect(&Tok::LParen)?;
        let mut args = Vec::new();
        if *self.peek() != Tok::RParen {
            loop {
                args.push(self.parse_sub_expr(0)?);
                if *self.peek() == Tok::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen)?;
        Ok(args)
    }

    /// `Name { field: expr, ... }` — a record literal. The leading `Name` and the
    /// opening `{` have already been recognized by `parse_atom`.
    fn parse_record_literal(&mut self, name: String, start: (usize, usize)) -> Result<Expr, String> {
        self.expect(&Tok::LBrace)?;
        let mut fields = Vec::new();
        if *self.peek() != Tok::RBrace {
            loop {
                let fname = self.expect_ident()?;
                self.expect(&Tok::Colon)?;
                let val = self.parse_sub_expr(0)?;
                fields.push((fname, val));
                if *self.peek() == Tok::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(self.spanned(start, ExprKind::Record(name, fields)))
    }

    fn parse_atom(&mut self) -> Result<Expr, String> {
        let start = self.here();
        match self.peek().clone() {
            Tok::Int(v) => {
                self.advance();
                Ok(self.spanned(start, ExprKind::Int(v)))
            }
            Tok::Float(v) => {
                self.advance();
                Ok(self.spanned(start, ExprKind::Float(v)))
            }
            Tok::Str(s) => {
                self.advance();
                Ok(self.spanned(start, ExprKind::Str(s)))
            }
            Tok::True => {
                self.advance();
                Ok(self.spanned(start, ExprKind::Bool(true)))
            }
            Tok::False => {
                self.advance();
                Ok(self.spanned(start, ExprKind::Bool(false)))
            }
            Tok::If => self.parse_if(),
            Tok::Match => self.parse_match(),
            Tok::LBrace => self.parse_block(),
            // Array literal `[e0, e1, ...]` desugars to a single flat,
            // variadic `array_lit` builtin call — no new AST node, so the
            // checker (which special-cases `array_lit` by name), IR and all
            // backends see only an ordinary builtin call. An empty `[]` is
            // `array_lit()`.
            Tok::LBracket => {
                self.advance(); // `[`
                let mut elems = Vec::new();
                if *self.peek() != Tok::RBracket {
                    loop {
                        elems.push(self.parse_sub_expr(0)?);
                        if *self.peek() == Tok::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RBracket)?;
                Ok(self.spanned(start, ExprKind::Call("array_lit".to_string(), elems)))
            }
            Tok::LParen => {
                self.advance();
                // `()` = Unit; `(e)` = grouping; `(a, b, ..)` = a tuple value
                // (the synthetic `$TupleN` constructor).
                if *self.peek() == Tok::RParen {
                    self.advance();
                    return Ok(self.spanned(start, ExprKind::Unit));
                }
                let mut elems = vec![self.parse_sub_expr(0)?];
                while *self.peek() == Tok::Comma {
                    self.advance();
                    elems.push(self.parse_sub_expr(0)?);
                }
                self.expect(&Tok::RParen)?;
                if elems.len() == 1 {
                    // A parenthesized single expression keeps its OWN span (the
                    // inner node), not the paren extent — the parens are pure
                    // grouping with no node of their own.
                    Ok(elems.into_iter().next().unwrap())
                } else {
                    let n = elems.len();
                    let tname = self.tuple_name(n);
                    Ok(self.spanned(start, ExprKind::Ctor(tname, elems)))
                }
            }
            Tok::Ident(name) => {
                self.advance();
                // `Upper { field: expr, ... }` is a record literal (unless we're
                // in an `if`/`match` head, where `{` opens the block/arms).
                if is_upper(&name)
                    && *self.peek() == Tok::LBrace
                    && !self.no_record_literal
                {
                    return self.parse_record_literal(name, start);
                }
                let has_args = *self.peek() == Tok::LParen;
                if has_args {
                    let args = self.parse_args()?;
                    if is_upper(&name) {
                        Ok(self.spanned(start, ExprKind::Ctor(name, args)))
                    } else {
                        Ok(self.spanned(start, ExprKind::Call(name, args)))
                    }
                } else if is_upper(&name) {
                    Ok(self.spanned(start, ExprKind::Ctor(name, Vec::new())))
                } else {
                    Ok(self.spanned(start, ExprKind::Var(name)))
                }
            }
            other => Err(format!(
                "line {}: unexpected token {:?} in expression",
                self.line(),
                other
            )),
        }
    }

    /// Parse an expression with record literals SUPPRESSED (`if`/`match` head),
    /// restoring the previous setting afterward.
    fn parse_head_expr(&mut self) -> Result<Expr, String> {
        let prev = self.no_record_literal;
        self.no_record_literal = true;
        let r = self.parse_expr(0);
        self.no_record_literal = prev;
        r
    }

    /// Parse an expression with record literals ALLOWED (inside a delimited
    /// sub-expression: parens, args, blocks, array/record literals).
    fn parse_sub_expr(&mut self, min_bp: u8) -> Result<Expr, String> {
        let prev = self.no_record_literal;
        self.no_record_literal = false;
        let r = self.parse_expr(min_bp);
        self.no_record_literal = prev;
        r
    }

    fn parse_if(&mut self) -> Result<Expr, String> {
        let start = self.here();
        self.expect(&Tok::If)?;
        let cond = self.parse_head_expr()?;
        let then = self.parse_block()?;
        self.expect(&Tok::Else)?;
        let els = self.parse_block()?;
        Ok(self.spanned(start, ExprKind::If(Box::new(cond), Box::new(then), Box::new(els))))
    }

    fn parse_match(&mut self) -> Result<Expr, String> {
        let start = self.here();
        self.expect(&Tok::Match)?;
        let scrut = self.parse_head_expr()?;
        self.expect(&Tok::LBrace)?;
        let mut arms = Vec::new();
        while *self.peek() != Tok::RBrace {
            // Each match-arm pattern now carries its binders' precise spans on the
            // pattern nodes themselves (a `PatternKind::Var`'s span is its def
            // site), so no side-table bookkeeping is needed here.
            let pat = self.parse_pattern()?;
            self.expect(&Tok::FatArrow)?;
            // An arm body is a fresh expression context: record literals are
            // allowed even when the enclosing `match` is in an `if`/`match` head.
            let body = self.parse_sub_expr(0)?;
            arms.push(Arm { pat, body });
            if *self.peek() == Tok::Comma {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        if arms.is_empty() {
            return Err(format!("line {}: match needs at least one arm", self.line()));
        }
        Ok(self.spanned(start, ExprKind::Match(Box::new(scrut), arms)))
    }

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        // The pattern's start is the start of the current token; every variant
        // closes its span at the most recently consumed token via `spanned_pat`.
        let start = self.here();
        match self.peek().clone() {
            // Tuple pattern `(p, q, ..)` -> the synthetic `$TupleN` ctor pattern;
            // `(p)` is just grouping.
            Tok::LParen => {
                self.advance();
                let mut subs = vec![self.parse_pattern()?];
                while *self.peek() == Tok::Comma {
                    self.advance();
                    subs.push(self.parse_pattern()?);
                }
                self.expect(&Tok::RParen)?;
                if subs.len() == 1 {
                    // A parenthesized single pattern keeps its OWN span (the inner
                    // node), not the paren extent — the parens are pure grouping.
                    Ok(subs.into_iter().next().unwrap())
                } else {
                    let n = subs.len();
                    let tname = self.tuple_name(n);
                    Ok(self.spanned_pat(start, PatternKind::Ctor(tname, subs)))
                }
            }
            Tok::Underscore => {
                self.advance();
                Ok(self.spanned_pat(start, PatternKind::Wild))
            }
            Tok::Int(v) => {
                self.advance();
                Ok(self.spanned_pat(start, PatternKind::Int(v)))
            }
            Tok::True => {
                self.advance();
                Ok(self.spanned_pat(start, PatternKind::Bool(true)))
            }
            Tok::False => {
                self.advance();
                Ok(self.spanned_pat(start, PatternKind::Bool(false)))
            }
            Tok::Ident(name) => {
                self.advance();
                if is_upper(&name) {
                    // Record pattern `Name { x, y }` (shorthand binds each field
                    // to a same-named var) or `Name { x: pat, ... }`.
                    if *self.peek() == Tok::LBrace {
                        self.advance();
                        let mut fields = Vec::new();
                        if *self.peek() != Tok::RBrace {
                            loop {
                                // The field-name shorthand `{ x }` binds a variable
                                // whose pattern span is the field-name identifier.
                                let fstart = self.here();
                                let fname = self.expect_ident()?;
                                let sub = if *self.peek() == Tok::Colon {
                                    self.advance();
                                    self.parse_pattern()?
                                } else {
                                    self.spanned_pat(fstart, PatternKind::Var(fname.clone()))
                                };
                                fields.push((fname, sub));
                                if *self.peek() == Tok::Comma {
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(&Tok::RBrace)?;
                        return Ok(self.spanned_pat(start, PatternKind::Record(name, fields)));
                    }
                    let mut subs = Vec::new();
                    if *self.peek() == Tok::LParen {
                        self.advance();
                        if *self.peek() != Tok::RParen {
                            loop {
                                subs.push(self.parse_pattern()?);
                                if *self.peek() == Tok::Comma {
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(&Tok::RParen)?;
                    }
                    Ok(self.spanned_pat(start, PatternKind::Ctor(name, subs)))
                } else {
                    // A bare lowercase variable pattern: its span IS its def site.
                    Ok(self.spanned_pat(start, PatternKind::Var(name)))
                }
            }
            other => Err(format!("line {}: invalid pattern {:?}", self.line(), other)),
        }
    }

    fn parse_block(&mut self) -> Result<Expr, String> {
        // The block's span starts at its opening `{`.
        let start = self.here();
        self.expect(&Tok::LBrace)?;
        // Inside a block, record literals are allowed again (an `if`/`match` head
        // suppressed them only up to its block).
        let saved_no_rec = self.no_record_literal;
        self.no_record_literal = false;
        let result = self.parse_block_body(start);
        self.no_record_literal = saved_no_rec;
        result
    }

    fn parse_block_body(&mut self, start: (usize, usize)) -> Result<Expr, String> {
        let mut stmts = Vec::new();
        // Empty block evaluates to Unit. The synthesized `Unit` result has no
        // source token, so it carries the no-location sentinel span.
        if *self.peek() == Tok::RBrace {
            self.advance();
            return Ok(self.spanned(start, ExprKind::Block(stmts, Box::new(Expr::synth(ExprKind::Unit)))));
        }
        // Functional record update `{ base | field = expr, ... }`: if the block
        // does not start with `let`, parse the first expression; a following `|`
        // (single pipe, never a binary operator) marks an update. `parse_expr`
        // stops at the `|` since it is not an operator.
        if *self.peek() != Tok::Let {
            let first = self.parse_expr(0)?;
            if *self.peek() == Tok::Pipe {
                // The update spans the base through `}`; its start is the base's.
                let base_start = (first.span.start_line as usize, first.span.start_col as usize);
                return self.parse_update_tail(first, base_start);
            }
            // Not an update: `first` is a statement (`;`) or the block result.
            if *self.peek() == Tok::Semi {
                // The expression statement spans from the expression's start
                // through the terminating `;` (consumed next).
                let stmt_start = (first.span.start_line as usize, first.span.start_col as usize);
                self.advance();
                stmts.push(self.spanned_stmt(stmt_start, StmtKind::Expr(first)));
                if *self.peek() == Tok::RBrace {
                    self.advance();
                    return Ok(self.spanned(start, ExprKind::Block(stmts, Box::new(Expr::synth(ExprKind::Unit)))));
                }
            } else {
                self.expect(&Tok::RBrace)?;
                return Ok(self.spanned(start, ExprKind::Block(stmts, Box::new(first))));
            }
        }
        let final_expr;
        loop {
            if *self.peek() == Tok::Let {
                // The `let` statement spans from the `let` keyword through its
                // terminating `;`. `name_span` is the binder identifier alone.
                let stmt_start = self.here();
                self.advance();
                let name_span = self.cur_span();
                let name = self.expect_ident()?;
                let mut ann = None;
                if *self.peek() == Tok::Colon {
                    self.advance();
                    let tps = self.type_params.clone();
                    ann = Some(self.parse_type(&tps)?);
                }
                self.expect(&Tok::Eq)?;
                let value = self.parse_expr(0)?;
                self.expect(&Tok::Semi)?;
                stmts.push(self.spanned_stmt(
                    stmt_start,
                    StmtKind::Let { name, name_span, ann, value },
                ));
                continue;
            }
            let e = self.parse_expr(0)?;
            if *self.peek() == Tok::Semi {
                let stmt_start = (e.span.start_line as usize, e.span.start_col as usize);
                self.advance();
                stmts.push(self.spanned_stmt(stmt_start, StmtKind::Expr(e)));
                if *self.peek() == Tok::RBrace {
                    // trailing `;` then close: result is Unit (synthesized, no span).
                    final_expr = Expr::synth(ExprKind::Unit);
                    break;
                }
                continue;
            } else if *self.peek() == Tok::Pipe {
                // `{ stmts...; base | f = v, ... }` — the block result is an update.
                let base_start = (e.span.start_line as usize, e.span.start_col as usize);
                let upd = self.parse_update_tail(e, base_start)?;
                return Ok(self.spanned(start, ExprKind::Block(stmts, Box::new(upd))));
            } else {
                final_expr = e;
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(self.spanned(start, ExprKind::Block(stmts, Box::new(final_expr))))
    }

    /// Parse the tail of a functional update `| field = expr, ... }` given the
    /// already-parsed `base`. The leading `|` is the next token; consumes through
    /// the closing `}`. `start` is the position the whole update expression began.
    fn parse_update_tail(&mut self, base: Expr, start: (usize, usize)) -> Result<Expr, String> {
        self.expect(&Tok::Pipe)?;
        let mut updates = Vec::new();
        loop {
            let fname = self.expect_ident()?;
            self.expect(&Tok::Eq)?;
            let val = self.parse_expr(0)?;
            updates.push((fname, val));
            if *self.peek() == Tok::Comma {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(self.spanned(start, ExprKind::Update(Box::new(base), updates)))
    }
}

pub fn parse(toks: Vec<Token>) -> Result<Program, String> {
    Parser::new(toks).parse_program()
}

/// Parse like [`parse`], additionally returning the [`BinderSpans`] side table of
/// precise LAMBDA-PARAMETER definition spans — the one binder kind the AST cannot
/// carry a span for directly (`let` names, match-arm pattern variables, and
/// function parameters now carry their binder span IN the AST). Used by the
/// data-flow analyzer (`aria analyze`) so it can report each local binding's
/// definition site; ordinary compilation uses [`parse`] and ignores the table.
pub fn parse_with_binders(toks: Vec<Token>) -> Result<(Program, BinderSpans), String> {
    let mut p = Parser::new(toks);
    let prog = p.parse_program()?;
    Ok((prog, p.binder_spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_src(src: &str) -> Result<Program, String> {
        let toks = crate::lexer::lex(src)?;
        parse(toks)
    }

    #[test]
    fn fn_decl_line_is_the_fn_keyword_line() {
        // Each function's `line` is the 1-based source line of its `fn`/`pure`
        // keyword. Blank lines and comments are counted, and a `pure` prefix does
        // not shift the line off the keyword.
        let src = "\
fn a() -> Int = 1

-- a comment on line 3
fn b() -> Int = a()

pure fn c() -> Int = 9
";
        let prog = parse_src(src).expect("parse");
        let line_of = |name: &str| -> usize {
            prog.items
                .iter()
                .find_map(|it| match it {
                    Item::Fn(f) if f.name == name => Some(f.line),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("no fn {}", name))
        };
        assert_eq!(line_of("a"), 1);
        assert_eq!(line_of("b"), 4);
        assert_eq!(line_of("c"), 6);
    }

    #[test]
    fn deeply_nested_parens_yield_clean_error_not_overflow() {
        // ~200k nested parens previously overflowed the recursive-descent stack
        // (SIGSEGV / exit 134, no output). The depth guard must instead return a
        // clean parse error (a normal `Err`, never a crash).
        let n = 200_000;
        let src = format!(
            "fn main() -> Int = {}1{}",
            "(".repeat(n),
            ")".repeat(n)
        );
        let err = parse_src(&src).expect_err("deep nesting should be a clean parse error");
        assert!(
            err.contains("expression nesting too deep"),
            "expected the nesting-depth error, got `{}`",
            err
        );
    }

    #[test]
    fn deeply_nested_types_yield_clean_error_not_overflow() {
        // The type parser is mutually recursive too; nest parameterised types
        // pathologically and confirm a clean error rather than a stack overflow.
        let n = 100_000;
        let src = format!(
            "fn main() -> {}Int{} = 0",
            "Array[".repeat(n),
            "]".repeat(n)
        );
        let err = parse_src(&src).expect_err("deep type nesting should be a clean parse error");
        assert!(
            err.contains("expression nesting too deep"),
            "expected the nesting-depth error, got `{}`",
            err
        );
    }

    #[test]
    fn normal_nesting_parses_fine() {
        // A few hundred levels deep is comfortably under the limit and must parse.
        let n = 300;
        let src = format!(
            "fn main() -> Int = {}1{}",
            "(".repeat(n),
            ")".repeat(n)
        );
        let prog = parse_src(&src).expect("normal nesting should parse");
        assert!(
            prog.items.iter().any(|it| matches!(it, Item::Fn(f) if f.name == "main")),
            "parsed program should contain `main`"
        );
    }

    // ---- precise statement / pattern spans -------------------------------
    //
    // These mirror the expression-span style (the lexer's
    // `tokens_carry_precise_line_and_column` and the `synth`/`spanned` discipline):
    // each assertion pins an EXACT 1-based `(start_line, start_col)` ..
    // `(end_line, end_col)` half-open range so a regression that drifts a span by a
    // column is caught.

    /// The body block of function `name`, as `(stmts, last)`.
    fn fn_block<'a>(prog: &'a Program, name: &str) -> (&'a [Stmt], &'a Expr) {
        let f = prog
            .items
            .iter()
            .find_map(|it| match it {
                Item::Fn(f) if f.name == name => Some(f),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no fn {}", name));
        match &f.body.kind {
            ExprKind::Block(stmts, last) => (stmts.as_slice(), last),
            other => panic!("fn {} body is not a block: {:?}", name, other),
        }
    }

    fn span_tuple(s: Span) -> (u32, u32, u32, u32) {
        (s.start_line, s.start_col, s.end_line, s.end_col)
    }

    #[test]
    fn let_statement_carries_precise_span_and_binder_span() {
        // Line 2 is `  let x = 1 + 2;` (two leading spaces).
        //  col: 1   2='l' of `let` at col 3 ... the terminating `;` ends at col 17.
        //  `let` starts at col 3; `x` (the binder) is at col 7..8.
        let src = "\
fn f() -> Int = {
  let x = 1 + 2;
  x
}
";
        let prog = parse_src(src).expect("parse");
        let (stmts, last) = fn_block(&prog, "f");
        assert_eq!(stmts.len(), 1, "one let statement");
        let s = &stmts[0];
        let (name, name_span) = match &s.kind {
            StmtKind::Let { name, name_span, .. } => (name.as_str(), *name_span),
            other => panic!("expected a let, got {:?}", other),
        };
        assert_eq!(name, "x");
        // The whole `let` statement spans `let` (line 2, col 3) through the `;`
        // (which ends at col 17 — `  let x = 1 + 2;` is 16 chars, `;` at col 16,
        // ending one-past at col 17).
        assert_eq!(span_tuple(s.span), (2, 3, 2, 17), "let-statement span");
        // The binder `x` alone sits at line 2, cols 7..8.
        assert_eq!(span_tuple(name_span), (2, 7, 2, 8), "let binder span");
        // The block result `x` is on line 3, col 3..4 (sanity that spans flow).
        assert_eq!(span_tuple(last.span), (3, 3, 3, 4), "block result span");
    }

    #[test]
    fn expression_statement_span_runs_through_its_semicolon() {
        // `  print_int(0);` — the statement spans the call through the `;`.
        let src = "\
fn f() -> Int = {
  print_int(0);
  1
}
";
        let prog = parse_src(src).expect("parse");
        let (stmts, _last) = fn_block(&prog, "f");
        assert_eq!(stmts.len(), 1);
        match &stmts[0].kind {
            StmtKind::Expr(_) => {}
            other => panic!("expected an expression statement, got {:?}", other),
        }
        // `print_int(0)` starts at col 3; the `;` ends one-past at col 16
        // (`  print_int(0);` is 15 chars, `;` at col 15, end col 16).
        assert_eq!(span_tuple(stmts[0].span), (2, 3, 2, 16), "expr-stmt span");
    }

    #[test]
    fn match_arm_patterns_carry_precise_spans() {
        // A constructor pattern with a variable sub-binder, and a wildcard arm.
        // `    Some(v) => v,` and `    None    => 0`.
        let src = "\
type Opt = | Some(Int) | None
fn g(o: Opt) -> Int = match o {
  Some(v) => v,
  None => 0
}
";
        let prog = parse_src(src).expect("parse");
        let f = prog
            .items
            .iter()
            .find_map(|it| match it {
                Item::Fn(f) if f.name == "g" => Some(f),
                _ => None,
            })
            .expect("fn g");
        let arms = match &f.body.kind {
            ExprKind::Match(_, arms) => arms,
            other => panic!("g body is not a match: {:?}", other),
        };
        assert_eq!(arms.len(), 2);
        // Arm 0: `Some(v)` at line 3, cols 3..10 (`Some(v)` is 7 chars at col 3).
        assert_eq!(span_tuple(arms[0].pat.span), (3, 3, 3, 10), "Some(v) ctor pattern span");
        // Its sub-pattern `v` is at line 3, cols 8..9.
        let v_span = match &arms[0].pat.kind {
            PatternKind::Ctor(name, subs) => {
                assert_eq!(name, "Some");
                assert_eq!(subs.len(), 1);
                subs[0].span
            }
            other => panic!("expected Some(..) ctor pattern, got {:?}", other),
        };
        assert_eq!(span_tuple(v_span), (3, 8, 3, 9), "binder `v` sub-pattern span");
        // Arm 1: `None` (nullary ctor) at line 4, cols 3..7.
        assert_eq!(span_tuple(arms[1].pat.span), (4, 3, 4, 7), "None ctor pattern span");
    }

    #[test]
    fn record_field_shorthand_binder_span_is_the_field_name() {
        // `Point { x, y }` — the shorthand binders `x`/`y` get the field-name span.
        let src = "\
type Point = { x: Int, y: Int }
fn px(p: Point) -> Int = match p {
  Point { x, y } => x
}
";
        let prog = parse_src(src).expect("parse");
        let f = prog
            .items
            .iter()
            .find_map(|it| match it {
                Item::Fn(f) if f.name == "px" => Some(f),
                _ => None,
            })
            .expect("fn px");
        let arms = match &f.body.kind {
            ExprKind::Match(_, arms) => arms,
            other => panic!("px body is not a match: {:?}", other),
        };
        let fields = match &arms[0].pat.kind {
            PatternKind::Record(name, fields) => {
                assert_eq!(name, "Point");
                fields
            }
            other => panic!("expected a record pattern, got {:?}", other),
        };
        // `Point { x, y }`: `x` is at line 3, col 11..12; `y` at col 14..15.
        let x = fields.iter().find(|(n, _)| n == "x").map(|(_, p)| p.span).unwrap();
        let y = fields.iter().find(|(n, _)| n == "y").map(|(_, p)| p.span).unwrap();
        assert_eq!(span_tuple(x), (3, 11, 3, 12), "shorthand binder `x` span");
        assert_eq!(span_tuple(y), (3, 14, 3, 15), "shorthand binder `y` span");
        // The whole record pattern spans `Point { x, y }`: col 3 through col 17.
        assert_eq!(span_tuple(arms[0].pat.span), (3, 3, 3, 17), "record pattern span");
    }

    #[test]
    fn literal_and_wildcard_pattern_spans() {
        // Integer-literal and wildcard patterns get precise single-token spans.
        let src = "\
fn h(n: Int) -> Int = match n {
  0 => 10,
  _ => 20
}
";
        let prog = parse_src(src).expect("parse");
        let f = prog
            .items
            .iter()
            .find_map(|it| match it {
                Item::Fn(f) if f.name == "h" => Some(f),
                _ => None,
            })
            .expect("fn h");
        let arms = match &f.body.kind {
            ExprKind::Match(_, arms) => arms,
            other => panic!("h body is not a match: {:?}", other),
        };
        // `0` at line 2, col 3..4.
        assert!(matches!(arms[0].pat.kind, PatternKind::Int(0)));
        assert_eq!(span_tuple(arms[0].pat.span), (2, 3, 2, 4), "int pattern span");
        // `_` at line 3, col 3..4.
        assert!(matches!(arms[1].pat.kind, PatternKind::Wild));
        assert_eq!(span_tuple(arms[1].pat.span), (3, 3, 3, 4), "wildcard pattern span");
    }
}
