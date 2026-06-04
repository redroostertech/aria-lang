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
    /// A named type, possibly with type arguments, e.g. `List[Int]` or `Option[T]`.
    /// A non-generic named type has an empty argument list.
    Named(String, Vec<Ty>),
    /// A type variable, either a declared generic parameter (`T`) or a fresh
    /// unification variable introduced by the checker.
    Var(String),
    /// A first-class function type: parameter types and a return type, e.g.
    /// `(Int, Bool) -> Int`. Only the interpreter executes these; the compiled
    /// backends reject any program mentioning a `Ty::Fn`.
    Fn(Vec<Ty>, Box<Ty>),
}

/// Concrete closure type information attached to an `Expr::Lambda` by the
/// monomorphizer: the captured free variables (in the order they are stored in
/// the closure cell) with their concrete types, and the lambda's return type.
#[derive(Debug, Clone)]
pub struct ClosureSig {
    pub captures: Vec<(String, Ty)>,
    pub ret: Ty,
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
    /// A lambda with typed parameters and a body, e.g. `\(x: Int) -> x + 1`.
    /// Evaluates to a closure capturing the defining environment. The third field
    /// is concrete closure type information (captured variables with their types,
    /// and the return type); it is `None` as produced by the parser and filled in
    /// by `monomorphize` so the compiled backends can lay out the closure cell and
    /// type the lifted lambda. The interpreter ignores it.
    Lambda(Vec<(String, Ty)>, Box<Expr>, Option<ClosureSig>),
    /// Application of an arbitrary expression (a lambda, a function-valued
    /// variable, or a call result) to arguments, e.g. `f(3)` where `f` is a
    /// local function value, or `(\x -> x)(5)`. The third field is the concrete
    /// result type, filled in by `monomorphize` (`None` from the parser).
    Apply(Box<Expr>, Vec<Expr>, Option<Ty>),
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
    /// `true` if the function carries a `pure` annotation. The effect checker
    /// (`typeck`) verifies such a function performs no IO. Erased afterwards.
    pub pure: bool,
    /// Declared generic type parameters, e.g. `[T, U]`.
    pub type_params: Vec<String>,
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
    /// Declared generic type parameters, e.g. `[T]`.
    pub params: Vec<String>,
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
