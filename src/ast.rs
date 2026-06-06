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

/// A precise source span: the 1-based start and end (line, column) of a token
/// or a parsed node. Columns are 1-based and count Unicode scalar values from
/// the start of the line. `end_line`/`end_col` point one past the last
/// character of the node (a half-open `[start, end)` extent in column terms),
/// so a single-character token at line 4 col 7 has `end_col == 8`.
///
/// A compiler-synthesized node (one with no single source location: a
/// monomorphizer rewrite, a desugared tuple/record/array op, a lowered trait
/// dispatcher, a prelude-internal construct) carries [`Span::none`], whose
/// fields are all zero. Spans are pure METADATA: no backend, the monomorphizer,
/// or the evaluator reads them, so they never affect codegen or a program's
/// result. They feed diagnostics, runtime stack traces, the call graph, and the
/// LSP, which point at the EXACT sub-expression rather than the function's
/// definition line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

impl Span {
    /// The sentinel span for a compiler-synthesized node with no source
    /// location: all fields zero. Consumers treat a zero `start_line` as
    /// "no precise location" and fall back to the function-definition line.
    pub const fn none() -> Span {
        Span { start_line: 0, start_col: 0, end_line: 0, end_col: 0 }
    }

    /// `true` for the [`Span::none`] sentinel (no precise source location).
    pub fn is_none(&self) -> bool {
        self.start_line == 0
    }
}

/// An expression: its `kind` (the shape of the node) plus the precise source
/// `span` it was parsed from. Splitting the span out of every variant means a
/// single field carries the location for the whole AST, and matching is done on
/// `&expr.kind`. Synthesized expressions use [`Span::none`].
#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    /// Build an expression node from its kind and span.
    pub fn new(kind: ExprKind, span: Span) -> Expr {
        Expr { kind, span }
    }

    /// Build a compiler-synthesized expression (no source location): used by the
    /// monomorphizer, trait lowering, and any pass that fabricates a node.
    pub fn synth(kind: ExprKind) -> Expr {
        Expr { kind, span: Span::none() }
    }
}

#[derive(Debug, Clone)]
pub enum ExprKind {
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
    /// A record literal, e.g. `Point { x: 1.0, y: 2.0 }`. Fields may be written
    /// in any order; the checker validates the set against the declared record
    /// and the interpreter reorders them into declared field order. A record is
    /// a single-constructor type whose constructor shares the type's name.
    Record(String, Vec<(String, Expr)>),
    /// Field access, e.g. `p.x`. Resolved against the record type of the object.
    Field(Box<Expr>, String),
    /// Functional record update, e.g. `{ p | x = 3.0 }`: a copy of the base
    /// record with the listed fields replaced. Type-preserving.
    Update(Box<Expr>, Vec<(String, Expr)>),
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
    /// A record pattern, e.g. `Point { x, y }`, binding each named field to a
    /// same-named variable. Unmentioned fields are ignored.
    Record(String, Vec<(String, Pattern)>),
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
    /// Precise source span of the parameter's BINDER (the parameter name
    /// identifier), used by the data-flow analyzer to report the binding's
    /// definition site. [`Span::none`] for compiler-synthesized parameters
    /// (monomorphizer clones / lowered dispatchers).
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: String,
    /// 1-based source line of the `fn` keyword that introduced this function.
    /// Used by runtime stack traces and the static call-graph analyzer to report
    /// the definition site of a function. `0` for compiler-generated functions
    /// (trait dispatchers / lowered impl methods / monomorphized clones) that
    /// have no single source line.
    pub line: usize,
    /// `true` if the function carries a `pure` annotation. The effect checker
    /// (`typeck`) verifies such a function performs no IO. Erased afterwards.
    pub pure: bool,
    /// Declared generic type parameters, e.g. `[T, U]`.
    pub type_params: Vec<String>,
    /// Trait bounds on the type parameters, e.g. `[T: Show]` gives `("T", "Show")`.
    /// A bound declares that any concrete `T` substituted at a call site must have
    /// an `impl` of the named trait; inside the body, a trait method of that trait
    /// may be called on a value of type `T` (deferred until monomorphization picks
    /// the concrete impl). Empty for ordinary (unbounded) functions.
    pub bounds: Vec<(String, String)>,
    pub params: Vec<Param>,
    pub ret: Ty,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Ty>,
    /// `Some(names)` iff this is a record-style variant (`type P = { x: T, .. }`):
    /// the field names, positionally aligned with `fields`. `None` for ordinary
    /// positional sum-type variants. A record type has exactly one such variant
    /// whose name equals the type name.
    pub field_names: Option<Vec<String>>,
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

/// A single method signature inside an `interface` declaration, e.g.
/// `fn show(self: T) -> String`. The first parameter's type is the trait's
/// `Self` type variable (named after the interface's type parameter), so when an
/// `impl` provides the method for a concrete type the signature is checked with
/// `Self` := that type.
#[derive(Debug, Clone)]
pub struct MethodSig {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Ty,
}

/// An `interface Show[T] { fn show(self: T) -> String, .. }` — a trait /
/// typeclass. `self_param` is the name of the type variable (`T`) standing for
/// the implementing type; it appears as `Ty::Var(self_param)` in the method
/// signatures. Interfaces never reach the backends: the parser lowers them (and
/// their impls) to ordinary `Item::Fn` dispatchers + mangled impl functions.
#[derive(Debug, Clone)]
pub struct InterfaceDecl {
    pub name: String,
    pub self_param: String,
    pub methods: Vec<MethodSig>,
}

/// An `impl Show for Point { fn show(self: Point) -> String = .. }` — a concrete
/// implementation of an interface for one head type. `methods` are full function
/// declarations; lowering mangles each to `show$Show$Point` and synthesizes a
/// dispatcher that routes by the runtime constructor.
#[derive(Debug, Clone)]
pub struct ImplDecl {
    pub trait_name: String,
    pub head_type: String,
    pub methods: Vec<FnDecl>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

/// A parser-built SIDE TABLE of precise binder source spans for the binders that
/// the AST does not itself carry a span for (`let` names, lambda parameters, and
/// match-arm pattern variables). It is pure METADATA consumed only by the
/// data-flow analyzer to report each local binding's definition site; no backend,
/// the monomorphizer, or the evaluator reads it, so it never affects codegen or a
/// program's result.
///
/// Each binder is keyed by an in-AST span that survives unchanged from parse to
/// analysis (the program is not monomorphized before `aria analyze`):
///   * a `let` binder by the span of its right-hand-side value expression,
///   * a lambda parameter by `(lambda-body span, parameter index)`,
///   * a match-arm pattern variable by `(arm-body span, binder name)`.
/// (Function parameters carry their own span directly on [`Param`].)
#[derive(Debug, Clone, Default)]
pub struct BinderSpans {
    /// `let` binder span, keyed by the let's RHS value expression span.
    pub lets: std::collections::HashMap<Span, Span>,
    /// Lambda parameter binder span, keyed by `(lambda body span, param index)`.
    pub lambda_params: std::collections::HashMap<(Span, usize), Span>,
    /// Match-arm pattern-variable binder span, keyed by `(arm body span, name)`.
    pub match_binders: std::collections::HashMap<(Span, String), Span>,
}
