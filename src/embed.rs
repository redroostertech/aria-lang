//! A real, learned distributional word-embedding model — zero dependencies.
//!
//! This is NOT a hash. It is the canonical *count-based* distributional method
//! (Levy & Goldberg, 2014: "Neural Word Embedding as Implicit Matrix
//! Factorization"), which is provably competitive with word2vec:
//!
//!   1. Tokenize a BUNDLED English corpus (`data/corpus.txt`, embedded at
//!      compile time via `include_str!`).
//!   2. Build a symmetric window-based word-word CO-OCCURRENCE matrix.
//!   3. Re-weight it with PPMI (positive pointwise mutual information), the
//!      association measure that turns raw counts into a semantic signal.
//!   4. Reduce the PPMI matrix to `DIM` dimensions with a hand-rolled TRUNCATED
//!      SVD (deterministic power/subspace iteration — no external linear
//!      algebra crate, no RNG nondeterminism: a fixed-seed PRNG seeds the
//!      starting subspace, so the table is byte-identical across runs/builds).
//!
//! The result is a learned vocabulary -> vector table where words appearing in
//! similar contexts get similar vectors. That is *real distributional
//! semantics*: `cat` and `dog` land near each other because they share
//! contexts ("the ___ is a pet", "feed the ___") — something a hash provably
//! cannot do. It is deliberately a SMALL model on a SMALL corpus: genuine
//! distributional structure, not a large pretrained language model.
//!
//! The table is trained LAZILY on first use and cached in a `OnceLock`, so the
//! one-time PPMI+SVD cost is paid once per process and every later `embed` /
//! `embed_similarity` call is a fast table lookup + average.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The bundled training corpus, embedded into the binary at compile time.
/// Source: an original, hand-curated public-domain-style English corpus written
/// for this project (`data/corpus.txt`). It is plain declarative sentences over
/// everyday vocabulary — animals, foods, colors, family, royalty, places,
/// occupations, weather, vehicles, and actions — arranged so that semantically
/// related words share neighboring contexts. No copyrighted text is included.
const CORPUS: &str = include_str!("../data/corpus.txt");

/// Embedding dimensionality (the truncated-SVD rank).
pub const DIM: usize = 64;

/// Symmetric co-occurrence window radius (words within +/- WINDOW count).
const WINDOW: usize = 4;

/// Minimum corpus frequency for a token to enter the vocabulary. Rare words
/// have too few contexts to place meaningfully and only add noise.
const MIN_COUNT: usize = 2;

/// The trained model: a vocabulary and its row-per-word embedding table.
pub struct EmbedModel {
    /// token -> row index in `vectors`.
    vocab: HashMap<String, usize>,
    /// `vectors[i]` is the DIM-dim learned, L2-normalized embedding of word i.
    vectors: Vec<[f32; DIM]>,
}

impl EmbedModel {
    /// The learned vector for a single (already lowercased) word, if in-vocab.
    pub fn word_vector(&self, word: &str) -> Option<&[f32; DIM]> {
        self.vocab.get(word).map(|&i| &self.vectors[i])
    }

    /// Vocabulary size.
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// Embed a free-text string as the L2-normalized average of its in-vocab
    /// word vectors. Out-of-vocabulary words contribute nothing (the unknown
    /// token is the zero vector). Text with no known words yields the zero
    /// vector (cosine against anything is then 0.0, never NaN).
    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut acc = vec![0.0f32; DIM];
        let mut n = 0usize;
        for tok in tokenize(text) {
            if let Some(v) = self.word_vector(&tok) {
                for d in 0..DIM {
                    acc[d] += v[d];
                }
                n += 1;
            }
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            for x in acc.iter_mut() {
                *x *= inv;
            }
            l2_normalize(&mut acc);
        }
        acc
    }

    /// The `k` nearest in-vocab words to `word` by cosine similarity (excluding
    /// `word` itself). Returns `(word, cosine)` sorted by descending cosine.
    /// Used by the semantic-structure tests and `docs/EMBEDDINGS.md`.
    pub fn nearest_words(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        let qv = match self.word_vector(word) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let mut scored: Vec<(String, f32)> = self
            .vocab
            .iter()
            .filter(|(w, _)| w.as_str() != word)
            .map(|(w, &i)| (w.clone(), cosine(qv, &self.vectors[i])))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }
}

/// The process-wide cached model, trained once on first use.
static MODEL: OnceLock<EmbedModel> = OnceLock::new();

/// Borrow the trained model, training it (deterministically) on first access.
pub fn model() -> &'static EmbedModel {
    MODEL.get_or_init(|| train(CORPUS))
}

/// Lowercase + ASCII-word tokenizer. Splits on any non-alphabetic character and
/// lowercases, so punctuation and digits are dropped and `Cat,`/`cat` unify.
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Train the count-based PPMI+SVD model from a corpus string. Pure and
/// deterministic: same corpus -> byte-identical table.
pub fn train(corpus: &str) -> EmbedModel {
    // ---- 1. Vocabulary (frequency-filtered, deterministic order) -----------
    let mut counts: HashMap<String, usize> = HashMap::new();
    // Tokenize per line so windows do not span across unrelated sentences.
    let lines: Vec<Vec<String>> = corpus
        .lines()
        .map(tokenize)
        .filter(|l| !l.is_empty())
        .collect();
    for line in &lines {
        for tok in line {
            *counts.entry(tok.clone()).or_insert(0) += 1;
        }
    }
    // Sort the vocabulary so row indices are deterministic regardless of the
    // HashMap iteration order: by descending count, then lexicographically.
    let mut kept: Vec<(String, usize)> =
        counts.into_iter().filter(|(_, c)| *c >= MIN_COUNT).collect();
    kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut vocab: HashMap<String, usize> = HashMap::new();
    for (i, (w, _)) in kept.iter().enumerate() {
        vocab.insert(w.clone(), i);
    }
    let v = vocab.len();
    if v == 0 {
        return EmbedModel {
            vocab,
            vectors: Vec::new(),
        };
    }

    // ---- 2. Symmetric window co-occurrence matrix --------------------------
    // Dense `v x v` is fine for a small vocabulary (a few hundred words).
    let mut cooc = vec![0.0f64; v * v];
    let mut word_total = vec![0.0f64; v]; // marginal counts (row sums)
    let mut grand_total = 0.0f64;
    for line in &lines {
        let ids: Vec<Option<usize>> = line.iter().map(|w| vocab.get(w).copied()).collect();
        for i in 0..ids.len() {
            let wi = match ids[i] {
                Some(x) => x,
                None => continue,
            };
            let lo = i.saturating_sub(WINDOW);
            let hi = (i + WINDOW).min(ids.len().saturating_sub(1));
            for j in lo..=hi {
                if j == i {
                    continue;
                }
                if let Some(wj) = ids[j] {
                    cooc[wi * v + wj] += 1.0;
                    word_total[wi] += 1.0;
                    grand_total += 1.0;
                }
            }
        }
    }

    // ---- 3. PPMI re-weighting ----------------------------------------------
    // PPMI(i, j) = max(0, log( P(i, j) / (P(i) * P(j)) )).
    // With counts: log( cooc_ij * total / (sum_i * sum_j) ), clamped at 0.
    let mut ppmi = vec![0.0f64; v * v];
    if grand_total > 0.0 {
        for i in 0..v {
            if word_total[i] == 0.0 {
                continue;
            }
            for j in 0..v {
                let c = cooc[i * v + j];
                if c <= 0.0 || word_total[j] == 0.0 {
                    continue;
                }
                let pmi = (c * grand_total / (word_total[i] * word_total[j])).ln();
                if pmi > 0.0 {
                    ppmi[i * v + j] = pmi;
                }
            }
        }
    }

    // ---- 4. Truncated SVD of the (symmetric) PPMI matrix -------------------
    // PPMI is symmetric, so its SVD coincides with its symmetric eigendecomp:
    // M = U S U^T. We recover the top-DIM eigenpairs by deterministic subspace
    // (block power) iteration with Gram-Schmidt re-orthonormalization, then form
    // word vectors W = U * sqrt(S) — the standard Levy-Goldberg embedding.
    let k = DIM.min(v);
    let (eigvecs, eigvals) = truncated_eig_sym(&ppmi, v, k);

    let mut vectors = vec![[0.0f32; DIM]; v];
    for (col, &lambda) in eigvals.iter().enumerate() {
        // Negative eigenvalues (numerical noise on a PSD-ish matrix) -> 0.
        let scale = if lambda > 0.0 { lambda.sqrt() } else { 0.0 };
        for row in 0..v {
            vectors[row][col] = (eigvecs[row * k + col] * scale) as f32;
        }
    }
    // L2-normalize each word vector so cosine is a plain dot product downstream
    // and averaging behaves well.
    for vec in vectors.iter_mut() {
        l2_normalize(vec);
    }

    EmbedModel { vocab, vectors }
}

/// Deterministic top-`k` symmetric eigendecomposition of a dense `n x n`
/// symmetric matrix `m` (row-major), by block power (subspace) iteration with
/// Gram-Schmidt re-orthonormalization. Returns `(eigvecs, eigvals)` where
/// `eigvecs` is row-major `n x k` (column c is eigenvector c) and `eigvals[c]`
/// is the matching Rayleigh-quotient eigenvalue, sorted by descending |value|.
///
/// This is the textbook deterministic routine for the leading invariant
/// subspace of a symmetric matrix. The starting block is seeded by a fixed-seed
/// PRNG (no OS entropy), so the result is identical on every run/build.
fn truncated_eig_sym(m: &[f64], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let k = k.min(n);
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    // Starting block q: n x k, columns seeded deterministically then
    // orthonormalized.
    let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15);
    let mut q = vec![0.0f64; n * k];
    for x in q.iter_mut() {
        // Symmetric in [-1, 1).
        *x = rng.next_f64() * 2.0 - 1.0;
    }
    gram_schmidt(&mut q, n, k);

    // Iterate Z = M Q ; Q = orthonormalize(Z). 100 iterations is ample
    // convergence for a well-separated leading subspace of this size and keeps
    // training well under a millisecond for a few-hundred-word vocabulary.
    let mut z = vec![0.0f64; n * k];
    for _ in 0..100 {
        // z = m * q  (n x n times n x k)
        for r in 0..n {
            let mrow = &m[r * n..r * n + n];
            for c in 0..k {
                let mut s = 0.0;
                for t in 0..n {
                    s += mrow[t] * q[t * k + c];
                }
                z[r * k + c] = s;
            }
        }
        std::mem::swap(&mut q, &mut z);
        gram_schmidt(&mut q, n, k);
    }

    // Rayleigh quotients lambda_c = q_c^T M q_c give the eigenvalues.
    let mut eigvals = vec![0.0f64; k];
    // Reuse z as M*q.
    for r in 0..n {
        let mrow = &m[r * n..r * n + n];
        for c in 0..k {
            let mut s = 0.0;
            for t in 0..n {
                s += mrow[t] * q[t * k + c];
            }
            z[r * k + c] = s;
        }
    }
    for c in 0..k {
        let mut s = 0.0;
        for r in 0..n {
            s += q[r * k + c] * z[r * k + c];
        }
        eigvals[c] = s;
    }

    // Sort eigenpairs by descending eigenvalue (largest variance first).
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut sorted_vecs = vec![0.0f64; n * k];
    let mut sorted_vals = vec![0.0f64; k];
    for (newc, &oldc) in order.iter().enumerate() {
        sorted_vals[newc] = eigvals[oldc];
        for r in 0..n {
            sorted_vecs[r * k + newc] = q[r * k + oldc];
        }
    }
    (sorted_vecs, sorted_vals)
}

/// In-place modified Gram-Schmidt orthonormalization of the columns of a
/// row-major `n x k` matrix `q`. Degenerate (near-zero) columns are reseeded to
/// a canonical basis vector so the block never collapses in rank.
fn gram_schmidt(q: &mut [f64], n: usize, k: usize) {
    for c in 0..k {
        // Subtract projections onto previous columns.
        for p in 0..c {
            let mut dot = 0.0;
            for r in 0..n {
                dot += q[r * k + c] * q[r * k + p];
            }
            for r in 0..n {
                q[r * k + c] -= dot * q[r * k + p];
            }
        }
        // Normalize.
        let mut norm = 0.0;
        for r in 0..n {
            norm += q[r * k + c] * q[r * k + c];
        }
        norm = norm.sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for r in 0..n {
                q[r * k + c] *= inv;
            }
        } else {
            // Reseed to canonical basis vector e_c (deterministic) and
            // re-orthogonalize against previous columns on the next pass.
            for r in 0..n {
                q[r * k + c] = if r == c % n { 1.0 } else { 0.0 };
            }
            for p in 0..c {
                let mut dot = 0.0;
                for r in 0..n {
                    dot += q[r * k + c] * q[r * k + p];
                }
                for r in 0..n {
                    q[r * k + c] -= dot * q[r * k + p];
                }
            }
            let mut nrm = 0.0;
            for r in 0..n {
                nrm += q[r * k + c] * q[r * k + c];
            }
            nrm = nrm.sqrt();
            if nrm > 1e-12 {
                let inv = 1.0 / nrm;
                for r in 0..n {
                    q[r * k + c] *= inv;
                }
            }
        }
    }
}

/// A tiny deterministic 64-bit LCG, used only to seed the SVD starting block.
/// No OS entropy is ever consulted, so training is fully reproducible.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// A double in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Cosine similarity of two equal-length f32 slices. 0.0 on zero magnitude.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Scale a vector in place to unit L2 norm (no-op if all zeros).
fn l2_normalize(v: &mut [f32]) {
    let mut sum = 0.0f32;
    for &x in v.iter() {
        sum += x * x;
    }
    let norm = sum.sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Embed `text` as a learned `Vec<f32>` of length `DIM` (the public entry the
/// runtime calls). Uses the lazily-trained, cached model.
pub fn embed(text: &str) -> Vec<f32> {
    model().embed_text(text)
}

/// Cosine similarity of the learned embeddings of two texts.
pub fn embed_similarity(a: &str, b: &str) -> f32 {
    let m = model();
    let va = m.embed_text(a);
    let vb = m.embed_text(b);
    cosine(&va, &vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_is_nonempty_and_reasonable() {
        let m = model();
        assert!(
            m.vocab_len() > 150,
            "vocab too small: {} words",
            m.vocab_len()
        );
        // Common seed words must be present.
        for w in ["cat", "dog", "king", "queen", "apple", "red"] {
            assert!(m.word_vector(w).is_some(), "missing vocab word {w}");
        }
    }

    #[test]
    fn vectors_are_l2_normalized() {
        let m = model();
        for w in ["cat", "king", "river", "bread"] {
            let v = m.word_vector(w).unwrap();
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "{w} not normalized: |v|={n}");
        }
    }

    #[test]
    fn training_is_deterministic() {
        // Two independent trainings of the same corpus must be byte-identical:
        // no RNG nondeterminism leaks in.
        let a = train(CORPUS);
        let b = train(CORPUS);
        assert_eq!(a.vocab_len(), b.vocab_len());
        for w in ["cat", "dog", "king", "queen", "river", "bread", "red", "blue"] {
            let va = a.word_vector(w).unwrap();
            let vb = b.word_vector(w).unwrap();
            assert_eq!(va, vb, "vectors for {w} differ across trainings");
        }
    }

    #[test]
    fn semantic_ordering_word_pairs() {
        // The proof it is REAL distributional semantics, not a hash: words that
        // share contexts in the corpus score higher than unrelated ones.
        let m = model();
        let sim = |a: &str, b: &str| {
            cosine(m.word_vector(a).unwrap(), m.word_vector(b).unwrap())
        };

        // Animals that share "pet/animal/farm" contexts vs an unrelated object.
        assert!(
            sim("cat", "dog") > sim("cat", "car"),
            "cat~dog {} should beat cat~car {}",
            sim("cat", "dog"),
            sim("cat", "car")
        );
        // Royalty pair vs royalty/food.
        assert!(
            sim("king", "queen") > sim("king", "bread"),
            "king~queen {} should beat king~bread {}",
            sim("king", "queen"),
            sim("king", "bread")
        );
        // Fruits cluster.
        assert!(
            sim("apple", "orange") > sim("apple", "king"),
            "apple~orange {} should beat apple~king {}",
            sim("apple", "orange"),
            sim("apple", "king")
        );
    }

    #[test]
    fn nearest_neighbors_are_semantic() {
        let m = model();
        // The top neighbors of "cat" should include other animals, not e.g.
        // "bread" or "river" at the very top. We assert at least one clearly
        // related word appears in the top 8.
        let nbrs = m.nearest_words("cat", 8);
        let words: Vec<&str> = nbrs.iter().map(|(w, _)| w.as_str()).collect();
        let animalish = ["dog", "horse", "cow", "pig", "rabbit", "fox", "pet", "animal", "kitten", "puppy"];
        assert!(
            words.iter().any(|w| animalish.contains(w)),
            "cat neighbors not animal-like: {words:?}"
        );
    }

    #[test]
    fn embed_text_related_beats_unrelated() {
        // Sentence-level: related sentences out-score unrelated ones.
        let m = model();
        let s1 = m.embed_text("the cat is a small pet animal");
        let s2 = m.embed_text("the dog is a loyal pet animal");
        let s3 = m.embed_text("the king rules the kingdom from his throne");
        let related = cosine(&s1, &s2);
        let unrelated = cosine(&s1, &s3);
        assert!(
            related > unrelated,
            "cat/dog sentence sim {related} should beat cat/king {unrelated}"
        );
    }

    #[test]
    fn out_of_vocab_text_is_zero_vector() {
        let m = model();
        let v = m.embed_text("zzqqxx wuggle florbnak");
        assert_eq!(v.len(), DIM);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn cosine_basic_properties() {
        let a = vec![0.3, 0.1, 0.7, 0.5];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
        let x = vec![1.0, 0.0];
        let y = vec![0.0, 1.0];
        assert!(cosine(&x, &y).abs() < 1e-6);
    }
}

