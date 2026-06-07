//! Compile-time tensor shape checker — a tiny, self-contained type system for
//! tensor expressions, independent of Aria's main parser.
//!
//! The premise: a dimension mismatch should be a *compile error*, not a 3am
//! crash. We model a small expression IR (`TExpr`) over `Shape`s and walk it
//! statically with `infer`. Shapes may carry concrete integer dimensions
//! (`Dim::Known`) or symbolic ones (`Dim::Named`, e.g. `Batch`), so an
//! attention block like `softmax(matmul(Q, transpose(K)))` can be verified
//! before a single number is ever computed.

use std::fmt;

/// A single dimension of a tensor shape.
///
/// `Known` is a concrete extent; `Named` is a symbolic placeholder (such as
/// `Batch` or `SeqLen`) that is checked by identity rather than by value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Known(usize),
    Named(String),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Known(n) => write!(f, "{}", n),
            Dim::Named(s) => write!(f, "{}", s),
        }
    }
}

/// An ordered list of dimensions describing a tensor's shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<Dim>,
}

impl Shape {
    /// Build a shape from a list of dimensions.
    pub fn new(dims: Vec<Dim>) -> Shape {
        Shape { dims }
    }

    /// Number of dimensions (rank) of this shape.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

/// A node in the tensor-expression IR.
///
/// Every variant is shape-checkable: `infer` propagates shapes bottom-up and
/// rejects any operation whose operands do not line up.
#[derive(Debug, Clone)]
pub enum TExpr {
    /// A named input tensor with a declared shape (a leaf of the IR).
    Input { name: String, shape: Shape },
    /// Matrix multiply: `[m, k] x [k, n] -> [m, n]`; inner dims must match.
    MatMul(Box<TExpr>, Box<TExpr>),
    /// Transpose of a 2D tensor: `[a, b] -> [b, a]`.
    Transpose(Box<TExpr>),
    /// Elementwise add: both operands must have identical shapes.
    Add(Box<TExpr>, Box<TExpr>),
    /// Softmax: shape-preserving.
    Softmax(Box<TExpr>),
}

/// A shape error carrying a human-readable explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeError {
    pub message: String,
}

impl ShapeError {
    fn new(message: impl Into<String>) -> ShapeError {
        ShapeError { message: message.into() }
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shape error: {}", self.message)
    }
}

impl std::error::Error for ShapeError {}

/// Infer the result shape of a tensor expression, or report the first shape
/// mismatch as a descriptive `ShapeError`.
pub fn infer(e: &TExpr) -> Result<Shape, ShapeError> {
    match e {
        TExpr::Input { shape, .. } => Ok(shape.clone()),

        TExpr::MatMul(lhs, rhs) => {
            let a = infer(lhs)?;
            let b = infer(rhs)?;
            if a.rank() != 2 || b.rank() != 2 {
                return Err(ShapeError::new(format!(
                    "matmul expects two 2D tensors, got {} (rank {}) x {} (rank {})",
                    a, a.rank(), b, b.rank()
                )));
            }
            // a = [m, k], b = [k, n]; the inner dims (k) must match.
            let inner_l = &a.dims[1];
            let inner_r = &b.dims[0];
            if inner_l != inner_r {
                return Err(ShapeError::new(format!(
                    "matmul inner dimensions do not match: lhs {} has inner dim {} \
                     but rhs {} has inner dim {}",
                    a, inner_l, b, inner_r
                )));
            }
            Ok(Shape::new(vec![a.dims[0].clone(), b.dims[1].clone()]))
        }

        TExpr::Transpose(inner) => {
            let s = infer(inner)?;
            if s.rank() != 2 {
                return Err(ShapeError::new(format!(
                    "transpose expects a 2D tensor, got {} (rank {})",
                    s, s.rank()
                )));
            }
            Ok(Shape::new(vec![s.dims[1].clone(), s.dims[0].clone()]))
        }

        TExpr::Add(lhs, rhs) => {
            let a = infer(lhs)?;
            let b = infer(rhs)?;
            if a != b {
                return Err(ShapeError::new(format!(
                    "add requires identical shapes, but lhs is {} and rhs is {}",
                    a, b
                )));
            }
            Ok(a)
        }

        TExpr::Softmax(inner) => infer(inner),
    }
}

// ===========================================================================
// Wiring the shape checker into the language: a compile-time pass over real
// `.aria` programs that catches tensor dimension mismatches before runtime.
// ===========================================================================

use crate::ast::{Expr, ExprKind, Item, Pattern, PatternKind, Program, StmtKind};
use std::collections::HashMap;

/// A statically-known tensor shape, or `None` when it cannot be determined.
type MaybeShape = Option<Shape>;

/// Compile-time tensor shape checking for a whole program.
///
/// This is a BEST-EFFORT, NO-FALSE-POSITIVE analysis: it tracks the statically
/// known shape of each tensor-valued binding through a function body (shapes
/// originate from `tensor_zeros(r, c)` with integer-literal dimensions and flow
/// through `matmul`/`transpose`/`softmax`/`relu`/`tensor_set`), and reports a
/// mismatch — e.g. `matmul([m, k], [k', n])` with `k != k'` — as an error,
/// reusing `infer`'s rules and messages. Any tensor whose shape is not
/// determinable (a function parameter, a non-literal dimension, a value from a
/// user function, an `if`/`match` whose branches disagree) is treated as UNKNOWN
/// and any operation consuming it is skipped, so a program is never rejected
/// unless its dimensions provably do not line up. Runs only on otherwise
/// well-typed programs (called from `typeck::check`).
pub fn check_program(program: &Program) -> Vec<String> {
    let mut errors = Vec::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            // Parameters have unknown shapes — we do not track shapes across
            // call boundaries (an intraprocedural analysis).
            let mut env: HashMap<String, MaybeShape> = HashMap::new();
            for p in &f.params {
                env.insert(p.name.clone(), None);
            }
            let mut ctx = ShapeCtx { fn_name: &f.name, errors: &mut errors };
            ctx.visit(&f.body, &env);
        }
    }
    errors
}

struct ShapeCtx<'a> {
    fn_name: &'a str,
    errors: &'a mut Vec<String>,
}

impl ShapeCtx<'_> {
    /// Walk an expression, recursing into EVERY subexpression so a nested tensor
    /// op (e.g. inside `tensor_get(matmul(a, b), 0, 0)`) is still checked, and
    /// return the statically-known shape of the expression if it is a tensor
    /// with a determinable shape. `env` is read-only; child scopes clone it so an
    /// inner `let`/pattern binding cannot leak a shape into an outer scope.
    fn visit(&mut self, e: &Expr, env: &HashMap<String, MaybeShape>) -> MaybeShape {
        match &e.kind {
            ExprKind::Var(v) => env.get(v).cloned().flatten(),
            ExprKind::Call(name, args) => self.visit_call(name, args, env),
            ExprKind::Ctor(_, args) => {
                for a in args {
                    self.visit(a, env);
                }
                None
            }
            ExprKind::Lambda(params, body, _) => {
                let mut child = env.clone();
                for (pn, _) in params {
                    child.insert(pn.clone(), None);
                }
                self.visit(body, &child);
                None
            }
            ExprKind::Apply(f, args, _) => {
                self.visit(f, env);
                for a in args {
                    self.visit(a, env);
                }
                None
            }
            ExprKind::Record(_, fields) => {
                for (_, ex) in fields {
                    self.visit(ex, env);
                }
                None
            }
            ExprKind::Field(o, _) => {
                self.visit(o, env);
                None
            }
            ExprKind::Update(o, fields) => {
                self.visit(o, env);
                for (_, ex) in fields {
                    self.visit(ex, env);
                }
                None
            }
            ExprKind::Unary(_, a) => {
                self.visit(a, env);
                None
            }
            ExprKind::Binary(_, a, b) => {
                self.visit(a, env);
                self.visit(b, env);
                None
            }
            ExprKind::If(c, t, f) => {
                self.visit(c, env);
                let st = self.visit(t, env);
                let sf = self.visit(f, env);
                // Join: only a shape both branches agree on is statically known.
                match (st, sf) {
                    (Some(a), Some(b)) if a == b => Some(a),
                    _ => None,
                }
            }
            ExprKind::Match(scrut, arms) => {
                self.visit(scrut, env);
                let mut joined: MaybeShape = None;
                let mut agree = true;
                for (i, arm) in arms.iter().enumerate() {
                    // Pattern bindings shadow with unknown shapes (so an inner
                    // `t` does not inherit an outer tensor's shape).
                    let mut child = env.clone();
                    for v in pattern_vars(&arm.pat) {
                        child.insert(v, None);
                    }
                    let s = self.visit(&arm.body, &child);
                    if i == 0 {
                        joined = s;
                    } else if joined != s {
                        agree = false;
                    }
                }
                if agree {
                    joined
                } else {
                    None
                }
            }
            ExprKind::Block(stmts, result) => {
                let mut child = env.clone();
                for s in stmts {
                    match &s.kind {
                        StmtKind::Let { name, value, .. } => {
                            let sh = self.visit(value, &child);
                            child.insert(name.clone(), sh);
                        }
                        StmtKind::Expr(ex) => {
                            self.visit(ex, &child);
                        }
                    }
                }
                self.visit(result, &child)
            }
            // Literals carry no tensor shape.
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::Unit => None,
        }
    }

    /// Shape of a builtin/function call, after recursing into its arguments.
    fn visit_call(
        &mut self,
        name: &str,
        args: &[Expr],
        env: &HashMap<String, MaybeShape>,
    ) -> MaybeShape {
        // Visit every argument first so nested tensor ops are always checked,
        // regardless of whether this call is itself a tracked tensor op.
        let shapes: Vec<MaybeShape> = args.iter().map(|a| self.visit(a, env)).collect();
        match (name, args.len()) {
            // A fresh tensor whose dimensions are integer literals has a known
            // shape; with non-literal dims it is unknown.
            ("tensor_zeros", 2) => match (int_lit(&args[0]), int_lit(&args[1])) {
                (Some(r), Some(c)) => Some(Shape::new(vec![Dim::Known(r), Dim::Known(c)])),
                _ => None,
            },
            // matmul / transpose: reuse `infer` for the rule + error message, but
            // only when the operand shapes are statically known.
            ("matmul", 2) => match (&shapes[0], &shapes[1]) {
                (Some(a), Some(b)) => self.run_infer(TExpr::MatMul(
                    Box::new(input_of(a)),
                    Box::new(input_of(b)),
                )),
                _ => None,
            },
            ("transpose", 1) => match &shapes[0] {
                Some(a) => self.run_infer(TExpr::Transpose(Box::new(input_of(a)))),
                None => None,
            },
            // Shape-preserving ops propagate the operand's (maybe-known) shape.
            ("softmax", 1) | ("relu", 1) | ("tensor_set", 4) => shapes[0].clone(),
            _ => None,
        }
    }

    /// Run `infer` on a small `TExpr` built from known operand shapes, recording
    /// any mismatch against the current function.
    fn run_infer(&mut self, te: TExpr) -> MaybeShape {
        match infer(&te) {
            Ok(s) => Some(s),
            Err(e) => {
                self.errors.push(format!("function `{}`: {}", self.fn_name, e));
                None
            }
        }
    }
}

/// A leaf `TExpr` carrying a known shape, for feeding `infer`.
fn input_of(s: &Shape) -> TExpr {
    TExpr::Input { name: "_".to_string(), shape: s.clone() }
}

/// A non-negative integer literal as a dimension extent, if `e` is one.
fn int_lit(e: &Expr) -> Option<usize> {
    match &e.kind {
        ExprKind::Int(n) if *n >= 0 => Some(*n as usize),
        _ => None,
    }
}

/// The variables a pattern binds (so an arm body can shadow them as unknown).
fn pattern_vars(p: &Pattern) -> Vec<String> {
    match &p.kind {
        PatternKind::Var(v) => vec![v.clone()],
        PatternKind::Ctor(_, sub) => sub.iter().flat_map(pattern_vars).collect(),
        PatternKind::Record(_, fields) => {
            fields.iter().flat_map(|(_, sp)| pattern_vars(sp)).collect()
        }
        _ => Vec::new(),
    }
}

/// Convenience: a known integer dimension.
fn known(n: usize) -> Dim {
    Dim::Known(n)
}

/// Convenience: a symbolic dimension.
fn named(s: &str) -> Dim {
    Dim::Named(s.to_string())
}

/// Convenience: a named input expression with a 2D shape.
fn input(name: &str, a: Dim, b: Dim) -> TExpr {
    TExpr::Input { name: name.to_string(), shape: Shape::new(vec![a, b]) }
}

/// Demonstrate compile-time shape checking: one expression that type-checks
/// and one that is rejected before any computation happens.
pub fn demo() {
    println!("=== Aria compile-time shape checking ===");

    // (1) A valid attention-like block: softmax(matmul(Q, transpose(K))).
    //   Q : [Batch, 64]
    //   K : [Batch, 64]  ->  transpose(K) : [64, Batch]
    //   matmul(Q, K^T)   : [Batch, Batch]
    //   softmax(...)     : [Batch, Batch]
    let q = input("Q", named("Batch"), known(64));
    let k = input("K", named("Batch"), known(64));
    let attention = TExpr::Softmax(Box::new(TExpr::MatMul(
        Box::new(q),
        Box::new(TExpr::Transpose(Box::new(k))),
    )));

    match infer(&attention) {
        Ok(shape) => println!("[ok]  softmax(matmul(Q, transpose(K))) : {}", shape),
        Err(err) => println!("[unexpected] {}", err),
    }

    // (2) A deliberately mismatched matmul, caught at "compile time".
    //   A : [32, 64], B : [128, 10]  ->  inner dims 64 != 128.
    let a = input("A", known(32), known(64));
    let b = input("B", known(128), known(10));
    let bad = TExpr::MatMul(Box::new(a), Box::new(b));

    match infer(&bad) {
        Ok(shape) => println!("[unexpected ok] {}", shape),
        Err(err) => println!("[rejected] {}", err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_attention_infers_expected_shape() {
        let q = input("Q", named("Batch"), known(64));
        let k = input("K", named("Batch"), known(64));
        let attention = TExpr::Softmax(Box::new(TExpr::MatMul(
            Box::new(q),
            Box::new(TExpr::Transpose(Box::new(k))),
        )));

        let shape = infer(&attention).expect("attention should type-check");
        assert_eq!(
            shape,
            Shape::new(vec![named("Batch"), named("Batch")])
        );
    }

    #[test]
    fn mismatched_matmul_is_rejected() {
        let a = input("A", known(32), known(64));
        let b = input("B", known(128), known(10));
        let bad = TExpr::MatMul(Box::new(a), Box::new(b));

        let err = infer(&bad).expect_err("inner dims 64 != 128 must fail");
        assert!(err.message.contains("inner dimensions do not match"));
        assert!(err.message.contains("64"));
        assert!(err.message.contains("128"));
    }

    #[test]
    fn mismatched_add_is_rejected() {
        let a = input("A", known(2), known(3));
        let b = input("B", known(2), known(4));
        let bad = TExpr::Add(Box::new(a), Box::new(b));

        let err = infer(&bad).expect_err("differing shapes must fail");
        assert!(err.message.contains("identical shapes"));
    }

    #[test]
    fn transpose_of_transpose_is_identity() {
        let m = input("M", known(7), named("Cols"));
        let original = infer(&m).unwrap();
        let twice = TExpr::Transpose(Box::new(TExpr::Transpose(Box::new(m))));

        assert_eq!(infer(&twice).unwrap(), original);
    }

    #[test]
    fn matmul_with_symbolic_inner_dims_checks() {
        // [M, Batch] x [Batch, N] -> [M, N] using symbolic inner dims.
        let a = input("A", named("M"), named("Batch"));
        let b = input("B", named("Batch"), named("N"));
        let prod = TExpr::MatMul(Box::new(a), Box::new(b));

        assert_eq!(
            infer(&prod).unwrap(),
            Shape::new(vec![named("M"), named("N")])
        );
    }

    // ---- program-level shape checking (`check_program`) -----------------

    /// Parse + shape-check a program (independent of the rest of typeck).
    fn shape_errors(src: &str) -> Vec<String> {
        let prog = crate::parser::parse(crate::lexer::lex(src).expect("lex")).expect("parse");
        check_program(&prog)
    }

    #[test]
    fn program_valid_attention_has_no_shape_error() {
        let src = "fn main() -> Float = {\n\
            let q = tensor_zeros(4, 8);\n\
            let k = tensor_zeros(4, 8);\n\
            tensor_get(softmax(matmul(q, transpose(k))), 0, 0)\n\
        }\n";
        assert!(shape_errors(src).is_empty());
    }

    #[test]
    fn program_mismatched_matmul_is_rejected() {
        let src = "fn main() -> Float = {\n\
            let a = tensor_zeros(32, 64);\n\
            let b = tensor_zeros(128, 10);\n\
            tensor_get(matmul(a, b), 0, 0)\n\
        }\n";
        let errs = shape_errors(src);
        assert_eq!(errs.len(), 1, "expected one shape error, got {:?}", errs);
        assert!(errs[0].contains("inner dimensions do not match"));
        assert!(errs[0].contains("64") && errs[0].contains("128"));
    }

    #[test]
    fn program_nested_matmul_in_arg_is_checked() {
        // The bad matmul is nested inside a non-tensor call argument.
        let src =
            "fn main() -> Float = tensor_get(matmul(tensor_zeros(3,5), tensor_zeros(7,2)), 0, 0)\n";
        let errs = shape_errors(src);
        assert_eq!(errs.len(), 1, "got {:?}", errs);
        assert!(errs[0].contains("inner dimensions do not match"));
    }

    #[test]
    fn program_unknown_shapes_are_not_false_positives() {
        // Tensor parameters (unknown shapes) and non-literal dimensions must not
        // produce any error — the analysis only rejects what it can prove wrong.
        let params = "fn use_t(a: Tensor, b: Tensor) -> Float = tensor_get(matmul(a, b), 0, 0)\n\
                      fn main() -> Float = use_t(tensor_zeros(2,3), tensor_zeros(9,9))\n";
        assert!(shape_errors(params).is_empty(), "{:?}", shape_errors(params));

        let dyn_dim =
            "fn mk(n: Int) -> Float = tensor_get(matmul(tensor_zeros(n, 4), tensor_zeros(9, 2)), 0, 0)\n\
             fn main() -> Float = mk(3)\n";
        assert!(shape_errors(dyn_dim).is_empty(), "{:?}", shape_errors(dyn_dim));
    }

    #[test]
    fn program_inner_let_shadow_does_not_corrupt_outer_shape() {
        // The inner `x = [5,5]` must not leak into the outer scope, where `x` is
        // [2,3] and `matmul(x, [3,4])` is valid (inner dims 3 == 3).
        let src = "fn main() -> Float = {\n\
            let x = tensor_zeros(2, 3);\n\
            let y = { let x = tensor_zeros(5, 5); tensor_get(x, 0, 0) };\n\
            tensor_get(matmul(x, tensor_zeros(3, 4)), 0, 0)\n\
        }\n";
        assert!(shape_errors(src).is_empty(), "{:?}", shape_errors(src));
    }

    #[test]
    fn program_transpose_makes_matmul_valid() {
        // [4,8] x transpose([4,8])=[8,4] -> [4,4] (inner 8 == 8).
        let ok = "fn main() -> Float =\n\
            tensor_get(matmul(tensor_zeros(4, 8), transpose(tensor_zeros(4, 8))), 0, 0)\n";
        assert!(shape_errors(ok).is_empty(), "{:?}", shape_errors(ok));
        // Without the transpose, [4,8] x [4,8] is a mismatch (8 != 4).
        let bad = "fn main() -> Float =\n\
            tensor_get(matmul(tensor_zeros(4, 8), tensor_zeros(4, 8)), 0, 0)\n";
        assert_eq!(shape_errors(bad).len(), 1);
    }
}
