//! Tensor extensions — quantization and weight compression for Aria.
//!
//! This module builds on the shared `Tensor` runtime to add the kernels the
//! inference tier needs once models are too big to ship in raw f32: extra
//! activations (`gelu`, `rmsnorm_rows`), a cache-blocked matmul that is bit-for
//! -bit identical to the reference kernel, and an INT8 symmetric quantization
//! path (`QuantTensor` + `qmatmul`). Like `tensor.rs`, the bodies are plain,
//! correct Rust (zero deps); SIMD/Metal lowering can replace them later without
//! touching the API.

use crate::tensor::Tensor;

/// GELU activation using the tanh approximation (same form most transformers
/// ship with). Applied elementwise; shape is preserved.
pub fn gelu(x: &Tensor) -> Tensor {
    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    x.map(|v| {
        let inner = C * (v + 0.044715 * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    })
}

/// Row-wise RMSNorm with a per-column learned weight (length = cols).
///
/// Each row is scaled by the reciprocal RMS of its elements, then multiplied
/// elementwise by `weight`. Unlike LayerNorm there is no mean subtraction and
/// no bias — this matches the LLaMA-style normalization used by Aria's models.
pub fn rmsnorm_rows(x: &Tensor, weight: &[f32], eps: f32) -> Tensor {
    let (m, n) = (x.rows(), x.cols());
    assert_eq!(weight.len(), n, "rmsnorm weight length must equal cols");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let row = &x.data[i * n..i * n + n];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (ms + eps).sqrt();
        for j in 0..n {
            out[i * n + j] = row[j] * inv_rms * weight[j];
        }
    }
    Tensor { shape: vec![m, n], data: out }
}

/// Cache-blocked matrix multiply: (m,k) x (k,n) -> (m,n).
///
/// Produces results identical to `Tensor::matmul` by using the exact same
/// accumulation order (i, p, j) — blocking only changes the *iteration*
/// grouping, never the summation order, so the f32 result is bit-for-bit equal.
/// `block` is the tile size along each dimension; `0` is treated as `1`.
pub fn matmul_tiled(a: &Tensor, b: &Tensor, block: usize) -> Tensor {
    let (m, k) = (a.rows(), a.cols());
    let (k2, n) = (b.rows(), b.cols());
    assert_eq!(k, k2, "matmul_tiled shape mismatch: (.,{}) x ({},.)", k, k2);
    let bs = block.max(1);
    let mut out = vec![0.0f32; m * n];
    let mut i0 = 0;
    while i0 < m {
        let i_end = (i0 + bs).min(m);
        let mut p0 = 0;
        while p0 < k {
            let p_end = (p0 + bs).min(k);
            let mut j0 = 0;
            while j0 < n {
                let j_end = (j0 + bs).min(n);
                for i in i0..i_end {
                    for p in p0..p_end {
                        let av = a.data[i * k + p];
                        let row = &b.data[p * n..p * n + n];
                        let dst = &mut out[i * n..i * n + n];
                        for j in j0..j_end {
                            dst[j] += av * row[j];
                        }
                    }
                }
                j0 = j_end;
            }
            p0 = p_end;
        }
        i0 = i_end;
    }
    Tensor { shape: vec![m, n], data: out }
}

/// An INT8-quantized 2D tensor with one symmetric f32 scale per row.
///
/// Reconstructed value is `data[i*cols+j] as f32 * scale[i]`. Symmetric means
/// zero maps to zero (no zero-point), which keeps `qmatmul` a clean i32 dot
/// product scaled afterward.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantTensor {
    pub shape: Vec<usize>,
    pub data: Vec<i8>,
    pub scale: Vec<f32>, // one entry per row
}

impl QuantTensor {
    fn rows(&self) -> usize {
        self.shape[0]
    }
    fn cols(&self) -> usize {
        self.shape[1]
    }

    /// Reconstruct an f32 tensor from the quantized representation.
    pub fn dequantize(&self) -> Tensor {
        let (m, n) = (self.rows(), self.cols());
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let s = self.scale[i];
            for j in 0..n {
                out[i * n + j] = self.data[i * n + j] as f32 * s;
            }
        }
        Tensor { shape: vec![m, n], data: out }
    }
}

/// Quantize a 2D f32 tensor to INT8 with per-row symmetric scaling.
///
/// For each row the scale is `max(|x|) / 127`; values are rounded to the nearest
/// integer and clamped into `[-127, 127]`. A zero row gets scale `0`.
pub fn quantize(x: &Tensor) -> QuantTensor {
    let (m, n) = (x.rows(), x.cols());
    let mut data = vec![0i8; m * n];
    let mut scale = vec![0.0f32; m];
    for i in 0..m {
        let row = &x.data[i * n..i * n + n];
        let amax = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let s = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for j in 0..n {
            let q = (row[j] * inv).round();
            let q = q.clamp(-127.0, 127.0);
            data[i * n + j] = q as i8;
        }
        scale[i] = s;
    }
    QuantTensor { shape: vec![m, n], data, scale }
}

/// INT8 matmul: (m,k) x (k,n) -> (m,n).
///
/// Each i8*i8 product is exact, but because `b` carries a *per-row* scale
/// `b.scale[p]` that varies along the contraction axis `k`, the term scale
/// `a.scale[i] * b.scale[p]` differs for every `p`. The reduction over `k`
/// therefore accumulates in f32 (one scaled add per term), not in a single i32
/// accumulator. To get a true rounded-once i32 dot product, `b` would need a
/// single per-tensor (or per-column) scale; that is a deliberate future change.
pub fn qmatmul(a: &QuantTensor, b: &QuantTensor) -> Tensor {
    let (m, k) = (a.rows(), a.cols());
    let (k2, n) = (b.rows(), b.cols());
    assert_eq!(k, k2, "qmatmul shape mismatch: (.,{}) x ({},.)", k, k2);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let a_scale = a.scale[i];
        for p in 0..k {
            let av = a.data[i * k + p] as i32;
            if av == 0 {
                continue;
            }
            // b's row p shares one scale across all of row p.
            let term_scale = a_scale * b.scale[p];
            let row = &b.data[p * n..p * n + n];
            let dst = &mut out[i * n..i * n + n];
            for j in 0..n {
                // Exact i8*i8 product, scaled by this term's scale and summed
                // into the f32 accumulator (the scale varies along k, so the
                // reduction is necessarily in f32 — see the function doc).
                let prod = av * row[j] as i32;
                dst[j] += prod as f32 * term_scale;
            }
        }
    }
    Tensor { shape: vec![m, n], data: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny deterministic LCG so tests are "random-ish" but reproducible.
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map high bits to [-1, 1).
        let x = (*seed >> 40) as f32 / (1u64 << 24) as f32;
        x * 2.0 - 1.0
    }

    fn rand_tensor(rows: usize, cols: usize, seed: &mut u64) -> Tensor {
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(lcg(seed) * 3.0);
        }
        Tensor { shape: vec![rows, cols], data }
    }

    #[test]
    fn quant_roundtrip_within_tol() {
        let mut seed = 0x1234_5678u64;
        let x = rand_tensor(5, 8, &mut seed);
        let q = quantize(&x);
        let r = q.dequantize();
        for (a, b) in x.data.iter().zip(&r.data) {
            // Quantization step is amax/127; per-element error < step/2 <= |amax|/254.
            assert!((a - b).abs() < 0.05, "roundtrip too far: {} vs {}", a, b);
        }
    }

    #[test]
    fn qmatmul_approx_matmul() {
        let mut seed = 0x9e37_79b9u64;
        let a = rand_tensor(4, 6, &mut seed);
        let b = rand_tensor(6, 5, &mut seed);
        let reference = a.matmul(&b);
        let qa = quantize(&a);
        let qb = quantize(&b);
        let got = qmatmul(&qa, &qb);
        assert_eq!(got.shape, reference.shape);
        for (g, r) in got.data.iter().zip(&reference.data) {
            assert!((g - r).abs() < 0.2, "qmatmul off: {} vs {}", g, r);
        }
    }

    #[test]
    fn tiled_exactly_equals_matmul() {
        let mut seed = 0xdead_beefu64;
        let a = rand_tensor(7, 9, &mut seed);
        let b = rand_tensor(9, 11, &mut seed);
        let reference = a.matmul(&b);
        for &block in &[1usize, 2, 3, 4, 8, 16, 100] {
            let tiled = matmul_tiled(&a, &b, block);
            assert_eq!(tiled.data, reference.data, "block {} differed", block);
        }
    }

    #[test]
    fn gelu_sanity() {
        let x = Tensor::new(vec![1, 3], vec![-10.0, 0.0, 10.0]);
        let g = gelu(&x);
        assert!(g.data[0].abs() < 1e-3, "gelu(-10) ~ 0");
        assert!(g.data[1].abs() < 1e-6, "gelu(0) == 0");
        assert!((g.data[2] - 10.0).abs() < 1e-3, "gelu(10) ~ 10");
        // Monotonic across the non-saturated middle.
        let m = gelu(&Tensor::new(vec![1, 3], vec![-1.0, 0.5, 2.0]));
        assert!(m.data[0] < m.data[1] && m.data[1] < m.data[2]);
    }

    #[test]
    fn rmsnorm_sanity() {
        // Unit weights: output row should have RMS ~ 1.
        let x = Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let w = vec![1.0; 4];
        let y = rmsnorm_rows(&x, &w, 1e-6);
        let ms = y.data.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((ms.sqrt() - 1.0).abs() < 1e-3, "rms should be ~1, got {}", ms.sqrt());

        // Weight scales the result linearly.
        let w2 = vec![2.0; 4];
        let y2 = rmsnorm_rows(&x, &w2, 1e-6);
        for (a, b) in y.data.iter().zip(&y2.data) {
            assert!((b - 2.0 * a).abs() < 1e-4);
        }
    }
}
