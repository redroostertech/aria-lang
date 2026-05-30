//! Minimal transformer forward pass (inference only) built on the shared
//! `crate::tensor::Tensor` runtime.
//!
//! This implements the classic decoder-style block end to end with zero
//! external dependencies:
//!   * token embedding lookup + sinusoidal positional encoding
//!   * multi-head causal self-attention: softmax(QK^T / sqrt(d_head)) V
//!   * residual + row-wise layernorm
//!   * position-wise MLP (linear -> gelu -> linear)
//!   * residual + row-wise layernorm
//!   * final linear projection to vocab logits + softmax over the last position
//!
//! Weights are not trained here; `demo()` synthesizes deterministic
//! pseudo-random weights with a tiny LCG so the forward pass is reproducible.
//! The math is plain, correct Rust; performance lowering is a later concern.

use crate::tensor::Tensor;

/// Deterministic linear-congruential generator producing f32 weights in a
/// small symmetric range. Used only to fabricate reproducible demo weights.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Lcg {
        // Avoid a zero state; any nonzero seed is fine.
        Lcg { state: seed | 1 }
    }

    // Numerical Recipes LCG constants.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform f32 in [-scale, scale).
    fn next_f32(&mut self, scale: f32) -> f32 {
        // Use the top 24 bits for a [0,1) mantissa, then center it.
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        let unit = bits as f32 / (1u32 << 24) as f32; // [0,1)
        (unit * 2.0 - 1.0) * scale
    }

    /// 2D tensor of pseudo-random weights, scaled by 1/sqrt(fan_in).
    fn weights(&mut self, rows: usize, cols: usize) -> Tensor {
        let scale = 1.0 / (rows as f32).sqrt();
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(self.next_f32(scale));
        }
        Tensor::new(vec![rows, cols], data)
    }

    /// Bias vector of pseudo-random values.
    fn bias(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32(scale)).collect()
    }
}

/// Gaussian Error Linear Unit (tanh approximation), applied elementwise.
fn gelu(x: f32) -> f32 {
    // 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) ))
    const C: f32 = 0.7978845608028654; // sqrt(2/pi)
    let inner = C * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// A single attention head's projection weights (no biases on Q/K/V).
struct AttentionHead {
    wq: Tensor, // (d_model, d_head)
    wk: Tensor, // (d_model, d_head)
    wv: Tensor, // (d_model, d_head)
}

/// Multi-head causal self-attention with an output projection.
struct MultiHeadAttention {
    heads: Vec<AttentionHead>,
    wo: Tensor, // (n_heads * d_head, d_model)
    d_head: usize,
}

impl MultiHeadAttention {
    /// Causal self-attention over `x` (seq_len, d_model) -> (seq_len, d_model).
    fn forward(&self, x: &Tensor) -> Tensor {
        let seq = x.rows();
        let d_head = self.d_head;
        let scale = 1.0 / (d_head as f32).sqrt();

        // Concatenated per-head context vectors: (seq, n_heads * d_head).
        let concat_cols = self.heads.len() * d_head;
        let mut concat = Tensor::zeros(&[seq, concat_cols]);

        for (h, head) in self.heads.iter().enumerate() {
            let q = x.matmul(&head.wq); // (seq, d_head)
            let k = x.matmul(&head.wk); // (seq, d_head)
            let v = x.matmul(&head.wv); // (seq, d_head)

            // Scores (seq, seq) = Q K^T / sqrt(d_head), with causal masking.
            let scores = q.matmul(&k.transpose()).scale(scale);
            let mut masked = scores.clone();
            for i in 0..seq {
                for j in 0..seq {
                    if j > i {
                        masked.set(i, j, f32::NEG_INFINITY);
                    }
                }
            }
            let attn = masked.softmax_rows(); // (seq, seq)
            let context = attn.matmul(&v); // (seq, d_head)

            // Place this head's output into the concatenated buffer.
            let off = h * d_head;
            for i in 0..seq {
                for d in 0..d_head {
                    concat.set(i, off + d, context.at(i, d));
                }
            }
        }

        // Output projection back to d_model.
        concat.matmul(&self.wo)
    }
}

/// Position-wise feed-forward network: linear -> gelu -> linear.
struct Mlp {
    w1: Tensor,    // (d_model, d_ff)
    b1: Vec<f32>,  // (d_ff)
    w2: Tensor,    // (d_ff, d_model)
    b2: Vec<f32>,  // (d_model)
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = x.matmul(&self.w1).add_rowvec(&self.b1).map(gelu);
        h.matmul(&self.w2).add_rowvec(&self.b2)
    }
}

/// One transformer decoder block: attention sublayer then MLP sublayer, each
/// wrapped in a residual connection followed by row-wise layernorm.
struct Block {
    attn: MultiHeadAttention,
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,
    mlp: Mlp,
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
}

impl Block {
    fn forward(&self, x: &Tensor) -> Tensor {
        const EPS: f32 = 1e-5;
        // Attention sublayer (post-norm): x = LN(x + Attn(x)).
        let a = self.attn.forward(x);
        let x = x.add(&a).layernorm_rows(&self.ln1_gamma, &self.ln1_beta, EPS);
        // MLP sublayer (post-norm): x = LN(x + MLP(x)).
        let m = self.mlp.forward(&x);
        x.add(&m).layernorm_rows(&self.ln2_gamma, &self.ln2_beta, EPS)
    }
}

/// A complete (tiny) decoder-style transformer for inference.
pub struct Transformer {
    pub vocab: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub max_seq: usize,
    embedding: Tensor, // (vocab, d_model)
    pos_enc: Tensor,   // (max_seq, d_model)
    blocks: Vec<Block>,
    w_out: Tensor,     // (d_model, vocab)
    b_out: Vec<f32>,   // (vocab)
}

/// Standard sinusoidal positional encoding, (max_seq, d_model).
fn sinusoidal_positional_encoding(max_seq: usize, d_model: usize) -> Tensor {
    Tensor::from_fn(max_seq, d_model, |pos, i| {
        // Pair up dimensions: even -> sin, odd -> cos, sharing a frequency.
        let pair = i / 2;
        let exponent = (2 * pair) as f32 / d_model as f32;
        let freq = 1.0 / 10000f32.powf(exponent);
        let angle = pos as f32 * freq;
        if i % 2 == 0 {
            angle.sin()
        } else {
            angle.cos()
        }
    })
}

impl Transformer {
    /// Build a deterministic model from an LCG seed.
    pub fn new(
        seed: u64,
        vocab: usize,
        d_model: usize,
        n_heads: usize,
        n_blocks: usize,
        d_ff: usize,
        max_seq: usize,
    ) -> Transformer {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        let d_head = d_model / n_heads;
        let mut g = Lcg::new(seed);

        let embedding = g.weights(vocab, d_model);
        let pos_enc = sinusoidal_positional_encoding(max_seq, d_model);

        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let mut heads = Vec::with_capacity(n_heads);
            for _ in 0..n_heads {
                heads.push(AttentionHead {
                    wq: g.weights(d_model, d_head),
                    wk: g.weights(d_model, d_head),
                    wv: g.weights(d_model, d_head),
                });
            }
            let attn = MultiHeadAttention {
                heads,
                wo: g.weights(n_heads * d_head, d_model),
                d_head,
            };
            let mlp = Mlp {
                w1: g.weights(d_model, d_ff),
                b1: g.bias(d_ff, 0.0),
                w2: g.weights(d_ff, d_model),
                b2: g.bias(d_model, 0.0),
            };
            blocks.push(Block {
                attn,
                ln1_gamma: vec![1.0; d_model],
                ln1_beta: vec![0.0; d_model],
                mlp,
                ln2_gamma: vec![1.0; d_model],
                ln2_beta: vec![0.0; d_model],
            });
        }

        let w_out = g.weights(d_model, vocab);
        let b_out = g.bias(vocab, 0.0);

        Transformer {
            vocab,
            d_model,
            n_heads,
            max_seq,
            embedding,
            pos_enc,
            blocks,
            w_out,
            b_out,
        }
    }

    /// Embed a token sequence: lookup + add positional encoding.
    /// Returns (seq_len, d_model).
    fn embed(&self, tokens: &[usize]) -> Tensor {
        let seq = tokens.len();
        assert!(seq <= self.max_seq, "sequence longer than max_seq");
        Tensor::from_fn(seq, self.d_model, |pos, c| {
            let tok = tokens[pos];
            assert!(tok < self.vocab, "token id out of vocab range");
            self.embedding.at(tok, c) + self.pos_enc.at(pos, c)
        })
    }

    /// Full forward pass. Returns logits (seq_len, vocab) over every position.
    pub fn forward(&self, tokens: &[usize]) -> Tensor {
        let mut x = self.embed(tokens);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        // Project hidden states to vocab logits.
        x.matmul(&self.w_out).add_rowvec(&self.b_out)
    }

    /// Next-token distribution: softmax over the logits of the last position.
    /// Returns a length-`vocab` probability vector.
    pub fn next_token_probs(&self, tokens: &[usize]) -> Vec<f32> {
        let logits = self.forward(tokens);
        let last = logits.rows() - 1;
        // Extract the last row, then softmax it via a (1, vocab) tensor.
        let row: Vec<f32> = (0..self.vocab).map(|c| logits.at(last, c)).collect();
        let probs = Tensor::new(vec![1, self.vocab], row).softmax_rows();
        probs.data
    }
}

/// Index of the maximum element (ties resolved to the lowest index).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

/// Build a tiny deterministic model, run a forward pass on a short token
/// sequence, and print the logits shape and the argmax next token.
pub fn demo() {
    let vocab = 16;
    let d_model = 8;
    let n_heads = 2;
    let n_blocks = 2;
    let d_ff = 32;
    let max_seq = 32;

    let model = Transformer::new(0xA1A0, vocab, d_model, n_heads, n_blocks, d_ff, max_seq);

    let tokens = [1usize, 5, 2, 7, 3];
    let logits = model.forward(&tokens);
    let probs = model.next_token_probs(&tokens);
    let next = argmax(&probs);

    println!("transformer demo:");
    println!(
        "  config: vocab={} d_model={} n_heads={} n_blocks={}",
        vocab, d_model, n_heads, n_blocks
    );
    println!("  input tokens: {:?}", tokens);
    println!("  logits shape: {:?}", logits.shape);
    println!(
        "  next-token argmax: {} (p={:.4})",
        next, probs[next]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> Transformer {
        // vocab=16, d_model=8, 2 heads, 2 blocks, d_ff=32, max_seq=32.
        Transformer::new(0xA1A0, 16, 8, 2, 2, 32, 32)
    }

    #[test]
    fn forward_output_shapes() {
        let model = tiny_model();
        let tokens = [1usize, 5, 2, 7, 3];
        let logits = model.forward(&tokens);
        // (seq_len, vocab)
        assert_eq!(logits.shape, vec![tokens.len(), model.vocab]);

        let probs = model.next_token_probs(&tokens);
        assert_eq!(probs.len(), model.vocab);
    }

    #[test]
    fn embedding_includes_positional_encoding() {
        let model = tiny_model();
        let x = model.embed(&[3usize, 3]);
        // Same token at different positions must differ (positional encoding).
        let differ = (0..model.d_model).any(|c| (x.at(0, c) - x.at(1, c)).abs() > 1e-9);
        assert!(differ, "positional encoding should distinguish positions");
    }

    #[test]
    fn next_token_probs_sum_to_one() {
        let model = tiny_model();
        let probs = model.next_token_probs(&[1usize, 5, 2, 7, 3]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum was {}", sum);
        // All probabilities are valid.
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn attention_softmax_rows_sum_to_one() {
        // Verify the masked-attention weights themselves form distributions.
        let model = tiny_model();
        let x = model.embed(&[1usize, 5, 2, 7]);
        let head = &model.blocks[0].attn.heads[0];
        let d_head = model.blocks[0].attn.d_head;
        let scale = 1.0 / (d_head as f32).sqrt();
        let q = x.matmul(&head.wq);
        let k = x.matmul(&head.wk);
        let mut scores = q.matmul(&k.transpose()).scale(scale);
        let seq = x.rows();
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    scores.set(i, j, f32::NEG_INFINITY);
                }
            }
        }
        let attn = scores.softmax_rows();
        for i in 0..seq {
            let s: f32 = (0..seq).map(|j| attn.at(i, j)).sum();
            assert!((s - 1.0).abs() < 1e-5, "attn row {} summed to {}", i, s);
            // Causal mask: future positions must have ~zero weight.
            for j in (i + 1)..seq {
                assert!(attn.at(i, j) < 1e-6, "masked weight leaked at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn forward_is_deterministic() {
        let tokens = [2usize, 4, 6, 8, 1, 0];
        let a = Transformer::new(0x1234, 16, 8, 2, 2, 32, 32).forward(&tokens);
        let b = Transformer::new(0x1234, 16, 8, 2, 2, 32, 32).forward(&tokens);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.data, b.data, "forward pass must be deterministic");
    }
}
