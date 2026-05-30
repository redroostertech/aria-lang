//! Abstract syntax tree for Aria.
//!
//! The AST is deliberately small and regular. Every construct is an
//! expression, identifiers are disambiguated by case at parse time
//! (Uppercase = constructor/type, lowercase = value/function), and there
//! is exactly one way to spell each thing.

#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Int,
    Float,
    Bool,
    Str,
    Unit,
    Named(String),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    Var(String),
    /// Constructor application, e.g. `Circle(2.0)` or nullary `Nil`.
    Ctor(String, Vec<Expr>),
    /// Function or builtin call, e.g. `factorial(5)`.
    Call(String, Vec<Expr>),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Match(Box<Expr>, Vec<Arm>),
    /// A sequence of statements ending in a result expression.
    Block(Vec<Stmt>, Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Option<Ty>, Expr),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Arm {
    pub pat: Pattern,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wild,
    Var(String),
    Int(i64),
    Bool(bool),
    Ctor(String, Vec<Pattern>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Ty,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Ty,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Ty>,
}

#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub name: String,
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Fn(FnDecl),
    Type(TypeDecl),
}

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}
