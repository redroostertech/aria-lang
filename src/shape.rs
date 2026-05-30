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
}
