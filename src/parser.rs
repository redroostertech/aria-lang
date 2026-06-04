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
        }
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
                    let pname = self.expect_ident()?;
                    self.expect(&Tok::Colon)?;
                    let ty = self.parse_type(&tps)?;
                    mparams.push(Param { name: pname, ty });
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
                let pname = self.expect_ident()?;
                self.expect(&Tok::Colon)?;
                let ty = self.parse_type(&type_params)?;
                params.push(Param { name: pname, ty });
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
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Tok::Minus => {
                self.advance();
                Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()?)))
            }
            Tok::Bang => {
                self.advance();
                Ok(Expr::Unary(UnOp::Not, Box::new(self.parse_unary()?)))
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
        self.expect(&Tok::Backslash)?;
        let mut params = Vec::new();
        if *self.peek() == Tok::LParen {
            self.advance();
            if *self.peek() != Tok::RParen {
                loop {
                    let pname = self.expect_ident()?;
                    self.expect(&Tok::Colon)?;
                    let tps = self.type_params.clone();
                    let ty = self.parse_type(&tps)?;
                    params.push((pname, ty));
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
            let pname = self.expect_ident()?;
            params.push((pname, Ty::Var(self.fresh_lambda_var())));
        }
        self.expect(&Tok::Arrow)?;
        let body = self.parse_expr(0)?;
        Ok(Expr::Lambda(params, Box::new(body), None))
    }

    // Parse an atom, then any trailing `(args)` applications. A trailing call on
    // a bare lowercase identifier stays `Expr::Call` (top-level call by name);
    // any other callee (a lambda, a parenthesized expression, or a further
    // application) becomes `Expr::Apply`.
    fn parse_postfix(&mut self) -> Result<Expr, String> {
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
                    e = Expr::Apply(Box::new(e), args, None);
                }
                Tok::LBracket => {
                    self.advance(); // `[`
                    let idx = self.parse_sub_expr(0)?;
                    self.expect(&Tok::RBracket)?;
                    e = Expr::Call("array_get".to_string(), vec![e, idx]);
                }
                Tok::Dot => {
                    self.advance(); // `.`
                    let field = self.expect_ident()?;
                    e = Expr::Field(Box::new(e), field);
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
    fn parse_record_literal(&mut self, name: String) -> Result<Expr, String> {
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
        Ok(Expr::Record(name, fields))
    }

    fn parse_atom(&mut self) -> Result<Expr, String> {
        match self.peek().clone() {
            Tok::Int(v) => {
                self.advance();
                Ok(Expr::Int(v))
            }
            Tok::Float(v) => {
                self.advance();
                Ok(Expr::Float(v))
            }
            Tok::Str(s) => {
                self.advance();
                Ok(Expr::Str(s))
            }
            Tok::True => {
                self.advance();
                Ok(Expr::Bool(true))
            }
            Tok::False => {
                self.advance();
                Ok(Expr::Bool(false))
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
                Ok(Expr::Call("array_lit".to_string(), elems))
            }
            Tok::LParen => {
                self.advance();
                // `()` = Unit; `(e)` = grouping; `(a, b, ..)` = a tuple value
                // (the synthetic `$TupleN` constructor).
                if *self.peek() == Tok::RParen {
                    self.advance();
                    return Ok(Expr::Unit);
                }
                let mut elems = vec![self.parse_sub_expr(0)?];
                while *self.peek() == Tok::Comma {
                    self.advance();
                    elems.push(self.parse_sub_expr(0)?);
                }
                self.expect(&Tok::RParen)?;
                if elems.len() == 1 {
                    Ok(elems.into_iter().next().unwrap())
                } else {
                    let n = elems.len();
                    Ok(Expr::Ctor(self.tuple_name(n), elems))
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
                    return self.parse_record_literal(name);
                }
                let has_args = *self.peek() == Tok::LParen;
                if has_args {
                    let args = self.parse_args()?;
                    if is_upper(&name) {
                        Ok(Expr::Ctor(name, args))
                    } else {
                        Ok(Expr::Call(name, args))
                    }
                } else if is_upper(&name) {
                    Ok(Expr::Ctor(name, Vec::new()))
                } else {
                    Ok(Expr::Var(name))
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
        self.expect(&Tok::If)?;
        let cond = self.parse_head_expr()?;
        let then = self.parse_block()?;
        self.expect(&Tok::Else)?;
        let els = self.parse_block()?;
        Ok(Expr::If(Box::new(cond), Box::new(then), Box::new(els)))
    }

    fn parse_match(&mut self) -> Result<Expr, String> {
        self.expect(&Tok::Match)?;
        let scrut = self.parse_head_expr()?;
        self.expect(&Tok::LBrace)?;
        let mut arms = Vec::new();
        while *self.peek() != Tok::RBrace {
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
        Ok(Expr::Match(Box::new(scrut), arms))
    }

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
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
                    Ok(subs.into_iter().next().unwrap())
                } else {
                    let n = subs.len();
                    Ok(Pattern::Ctor(self.tuple_name(n), subs))
                }
            }
            Tok::Underscore => {
                self.advance();
                Ok(Pattern::Wild)
            }
            Tok::Int(v) => {
                self.advance();
                Ok(Pattern::Int(v))
            }
            Tok::True => {
                self.advance();
                Ok(Pattern::Bool(true))
            }
            Tok::False => {
                self.advance();
                Ok(Pattern::Bool(false))
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
                                let fname = self.expect_ident()?;
                                let sub = if *self.peek() == Tok::Colon {
                                    self.advance();
                                    self.parse_pattern()?
                                } else {
                                    Pattern::Var(fname.clone())
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
                        return Ok(Pattern::Record(name, fields));
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
                    Ok(Pattern::Ctor(name, subs))
                } else {
                    Ok(Pattern::Var(name))
                }
            }
            other => Err(format!("line {}: invalid pattern {:?}", self.line(), other)),
        }
    }

    fn parse_block(&mut self) -> Result<Expr, String> {
        self.expect(&Tok::LBrace)?;
        // Inside a block, record literals are allowed again (an `if`/`match` head
        // suppressed them only up to its block).
        let saved_no_rec = self.no_record_literal;
        self.no_record_literal = false;
        let result = self.parse_block_body();
        self.no_record_literal = saved_no_rec;
        result
    }

    fn parse_block_body(&mut self) -> Result<Expr, String> {
        let mut stmts = Vec::new();
        // Empty block evaluates to Unit.
        if *self.peek() == Tok::RBrace {
            self.advance();
            return Ok(Expr::Block(stmts, Box::new(Expr::Unit)));
        }
        // Functional record update `{ base | field = expr, ... }`: if the block
        // does not start with `let`, parse the first expression; a following `|`
        // (single pipe, never a binary operator) marks an update. `parse_expr`
        // stops at the `|` since it is not an operator.
        if *self.peek() != Tok::Let {
            let first = self.parse_expr(0)?;
            if *self.peek() == Tok::Pipe {
                return self.parse_update_tail(first);
            }
            // Not an update: `first` is a statement (`;`) or the block result.
            if *self.peek() == Tok::Semi {
                self.advance();
                stmts.push(Stmt::Expr(first));
                if *self.peek() == Tok::RBrace {
                    self.advance();
                    return Ok(Expr::Block(stmts, Box::new(Expr::Unit)));
                }
            } else {
                self.expect(&Tok::RBrace)?;
                return Ok(Expr::Block(stmts, Box::new(first)));
            }
        }
        let final_expr;
        loop {
            if *self.peek() == Tok::Let {
                self.advance();
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
                stmts.push(Stmt::Let(name, ann, value));
                continue;
            }
            let e = self.parse_expr(0)?;
            if *self.peek() == Tok::Semi {
                self.advance();
                stmts.push(Stmt::Expr(e));
                if *self.peek() == Tok::RBrace {
                    // trailing `;` then close: result is Unit.
                    final_expr = Expr::Unit;
                    break;
                }
                continue;
            } else if *self.peek() == Tok::Pipe {
                // `{ stmts...; base | f = v, ... }` — the block result is an update.
                let upd = self.parse_update_tail(e)?;
                return Ok(Expr::Block(stmts, Box::new(upd)));
            } else {
                final_expr = e;
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(Expr::Block(stmts, Box::new(final_expr)))
    }

    /// Parse the tail of a functional update `| field = expr, ... }` given the
    /// already-parsed `base`. The leading `|` is the next token; consumes through
    /// the closing `}`.
    fn parse_update_tail(&mut self, base: Expr) -> Result<Expr, String> {
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
        Ok(Expr::Update(Box::new(base), updates))
    }
}

pub fn parse(toks: Vec<Token>) -> Result<Program, String> {
    Parser::new(toks).parse_program()
}
