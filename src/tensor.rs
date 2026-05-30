//! Shaped-tensor runtime — the numeric foundation for Aria's model/inference
//! work and its neural compression tier.
//!
//! Tensors are row-major `f32` with an explicit shape. This module is the
//! shared API that the transformer, quantization, and neural-compression
//! layers all build on. Kernels here are plain, correct Rust (zero deps);
//! SIMD/Metal/GPU lowering is a later optimization that can replace these
//! bodies without changing the API.

/// A dense, row-major f32 tensor with an explicit shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "shape {:?} does not match {} elements", shape, data.len());
        Tensor { shape, data }
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor { shape: shape.to_vec(), data: vec![0.0; n] }
    }

    pub fn filled(shape: &[usize], v: f32) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor { shape: shape.to_vec(), data: vec![v; n] }
    }

    /// Build a 2D tensor (rows x cols) from a closure of (row, col).
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Tensor {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Tensor { shape: vec![rows, cols], data }
    }

    pub fn rows(&self) -> usize {
        assert_eq!(self.shape.len(), 2, "rows() needs a 2D tensor");
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        assert_eq!(self.shape.len(), 2, "cols() needs a 2D tensor");
        self.shape[1]
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols() + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        let cols = self.cols();
        self.data[r * cols + c] = v;
    }

    /// Matrix multiply: (m,k) x (k,n) -> (m,n).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k) = (self.rows(), self.cols());
        let (k2, n) = (other.rows(), other.cols());
        assert_eq!(k, k2, "matmul shape mismatch: (.,{}) x ({},.)", k, k2);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                let row = &other.data[p * n..p * n + n];
                let dst = &mut out[i * n..i * n + n];
                for j in 0..n {
                    dst[j] += a * row[j];
                }
            }
        }
        Tensor { shape: vec![m, n], data: out }
    }

    /// 2D transpose.
    pub fn transpose(&self) -> Tensor {
        let (m, n) = (self.rows(), self.cols());
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor { shape: vec![n, m], data: out }
    }

    /// Elementwise add (identical shapes).
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "add shape mismatch");
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Tensor { shape: self.shape.clone(), data }
    }

    /// Multiply every element by a scalar.
    pub fn scale(&self, s: f32) -> Tensor {
        let data = self.data.iter().map(|x| x * s).collect();
        Tensor { shape: self.shape.clone(), data }
    }

    /// Add a per-column bias vector (length = cols) to every row of a 2D tensor.
    pub fn add_rowvec(&self, bias: &[f32]) -> Tensor {
        let (m, n) = (self.rows(), self.cols());
        assert_eq!(bias.len(), n, "bias length must equal cols");
        let mut out = self.data.clone();
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] += bias[j];
            }
        }
        Tensor { shape: vec![m, n], data: out }
    }

    /// Apply a function to every element.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        let data = self.data.iter().map(|x| f(*x)).collect();
        Tensor { shape: self.shape.clone(), data }
    }

    pub fn relu(&self) -> Tensor {
        self.map(|x| if x > 0.0 { x } else { 0.0 })
    }

    /// Row-wise softmax over the last dimension of a 2D tensor (numerically stable).
    pub fn softmax_rows(&self) -> Tensor {
        let (m, n) = (self.rows(), self.cols());
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let row = &self.data[i * n..i * n + n];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..n {
                let e = (row[j] - max).exp();
                out[i * n + j] = e;
                sum += e;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for j in 0..n {
                out[i * n + j] *= inv;
            }
        }
        Tensor { shape: vec![m, n], data: out }
    }

    /// Row-wise layer normalization with learned scale/shift (length = cols).
    pub fn layernorm_rows(&self, gamma: &[f32], beta: &[f32], eps: f32) -> Tensor {
        let (m, n) = (self.rows(), self.cols());
        assert_eq!(gamma.len(), n);
        assert_eq!(beta.len(), n);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let row = &self.data[i * n..i * n + n];
            let mean = row.iter().sum::<f32>() / n as f32;
            let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for j in 0..n {
                out[i * n + j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
        Tensor { shape: vec![m, n], data: out }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        let a = Tensor::from_fn(2, 3, |r, c| (r * 3 + c) as f32);
        let id = Tensor::from_fn(3, 3, |r, c| if r == c { 1.0 } else { 0.0 });
        assert_eq!(a.matmul(&id), a);
    }

    #[test]
    fn matmul_known() {
        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(a.matmul(&b).data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn transpose_roundtrip() {
        let a = Tensor::from_fn(2, 3, |r, c| (r * 3 + c) as f32);
        assert_eq!(a.transpose().transpose(), a);
    }

    #[test]
    fn softmax_sums_to_one() {
        let a = Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let s = a.softmax_rows();
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
