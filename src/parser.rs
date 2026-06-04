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
}

fn is_upper(name: &str) -> bool {
    name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
}

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser { toks, pos: 0, type_params: Vec::new(), lambda_counter: 0 }
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
        while *self.peek() != Tok::Eof {
            items.push(self.parse_item()?);
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self) -> Result<Item, String> {
        match self.peek() {
            Tok::Fn | Tok::Pure => Ok(Item::Fn(self.parse_fn()?)),
            Tok::Type => Ok(Item::Type(self.parse_type_decl()?)),
            other => Err(format!(
                "line {}: expected `fn`, `pure`, or `type`, found {:?}",
                self.line(),
                other
            )),
        }
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
        let type_params = self.parse_type_params()?;
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
            params,
            ret,
            body,
        })
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
        // Canonical form: the first variant must be preceded by `|`, so a sum
        // type has exactly one spelling (no optional leading pipe).
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
            variants.push(Variant { name: vname, fields });
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
            self.expect(&Tok::Arrow)?;
            let ret = self.parse_type(tparams)?;
            return Ok(Ty::Fn(params, Box::new(ret)));
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
        Ok(Expr::Lambda(params, Box::new(body)))
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
        // general application of the callee value.
        while *self.peek() == Tok::LParen {
            let args = self.parse_args()?;
            e = Expr::Apply(Box::new(e), args);
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
                args.push(self.parse_expr(0)?);
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
            Tok::LParen => {
                self.advance();
                let e = self.parse_expr(0)?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            Tok::Ident(name) => {
                self.advance();
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

    fn parse_if(&mut self) -> Result<Expr, String> {
        self.expect(&Tok::If)?;
        let cond = self.parse_expr(0)?;
        let then = self.parse_block()?;
        self.expect(&Tok::Else)?;
        let els = self.parse_block()?;
        Ok(Expr::If(Box::new(cond), Box::new(then), Box::new(els)))
    }

    fn parse_match(&mut self) -> Result<Expr, String> {
        self.expect(&Tok::Match)?;
        let scrut = self.parse_expr(0)?;
        self.expect(&Tok::LBrace)?;
        let mut arms = Vec::new();
        while *self.peek() != Tok::RBrace {
            let pat = self.parse_pattern()?;
            self.expect(&Tok::FatArrow)?;
            let body = self.parse_expr(0)?;
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
        let mut stmts = Vec::new();
        // Empty block evaluates to Unit.
        if *self.peek() == Tok::RBrace {
            self.advance();
            return Ok(Expr::Block(stmts, Box::new(Expr::Unit)));
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
            } else {
                final_expr = e;
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(Expr::Block(stmts, Box::new(final_expr)))
    }
}

pub fn parse(toks: Vec<Token>) -> Result<Program, String> {
    Parser::new(toks).parse_program()
}
