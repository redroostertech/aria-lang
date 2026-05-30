//! Native retrieval — RAG as a first-class runtime primitive.
//!
//! Retrieval in Aria is not a library bolted on top of the language: it is a
//! built-in capability backed by the same numeric core as the rest of the
//! runtime. This module provides a small, dependency-free vector store with
//! cosine-similarity search and a deterministic hashing embedder, so a demo
//! (and the tests) need no external model or network access.
//!
//! Embeddings are plain `Vec<f32>` rows. Swapping in real model embeddings
//! later only changes how vectors are produced, not how they are stored or
//! searched.

/// One stored document: an identifier, its raw text, and its embedding.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// An in-memory collection of embedded documents supporting top-k retrieval.
#[derive(Debug, Default, Clone)]
pub struct EmbeddingStore {
    docs: Vec<Document>,
}

impl EmbeddingStore {
    /// Create an empty store.
    pub fn new() -> EmbeddingStore {
        EmbeddingStore { docs: Vec::new() }
    }

    /// Number of stored documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the store holds no documents.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Insert a document. `id` is anything convertible to a `String`, so both
    /// integer-like and textual identifiers work at the call site.
    pub fn add(&mut self, id: impl Into<String>, text: impl Into<String>, embedding: Vec<f32>) {
        self.docs.push(Document {
            id: id.into(),
            text: text.into(),
            embedding,
        });
    }

    /// Return the `k` most similar documents to `query` by cosine similarity,
    /// sorted by score descending. Each entry is `(score, id, text)`.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(f32, String, String)> {
        let mut scored: Vec<(f32, String, String)> = self
            .docs
            .iter()
            .map(|d| (cosine_similarity(query, &d.embedding), d.id.clone(), d.text.clone()))
            .collect();

        // Sort by score descending. `f32` is not `Ord`, so compare manually and
        // treat NaN as the smallest value so it never wins a ranking.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Cosine similarity of two equal-length vectors.
///
/// Returns 0.0 if either vector has zero magnitude (or they differ in length),
/// which keeps degenerate inputs from producing NaN in rankings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// Deterministic 64-bit FNV-1a hash of a token.
///
/// FNV-1a is tiny, dependency-free, and stable across runs/platforms, which is
/// exactly what a reproducible embedder needs.
fn fnv1a(token: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in token.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Deterministic "bag of hashed tokens" embedder.
///
/// Tokenizes on whitespace, lowercases each token, hashes it into one of
/// `dim` buckets (with a sign bit so tokens can cancel as well as add), then
/// L2-normalizes the resulting vector. No model or randomness involved: the
/// same text always maps to the same unit vector.
pub fn hash_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim.max(1)];
    let dim = v.len();
    for token in text.split_whitespace() {
        let token = token.to_lowercase();
        if token.is_empty() {
            continue;
        }
        let h = fnv1a(&token);
        let bucket = (h % dim as u64) as usize;
        // Use a high bit as a sign so distinct tokens don't only ever add up.
        let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        v[bucket] += sign;
    }
    l2_normalize(&mut v);
    v
}

/// Scale a vector in place to unit L2 norm (no-op if it is all zeros).
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

/// Build a tiny corpus, embed a query, and print the ranked retrieval results.
pub fn demo() {
    const DIM: usize = 64;

    let corpus = [
        ("d1", "the cat sat on the warm windowsill in the sun"),
        ("d2", "dogs are loyal companions that love long walks"),
        ("d3", "rust is a systems programming language with no garbage collector"),
        ("d4", "vectors and cosine similarity power semantic search"),
        ("d5", "the quick brown fox jumps over the lazy dog"),
        ("d6", "neural networks learn embeddings from large text corpora"),
        ("d7", "a balanced breakfast includes fruit and whole grains"),
    ];

    let mut store = EmbeddingStore::new();
    for (id, text) in corpus.iter() {
        store.add(*id, *text, hash_embed(text, DIM));
    }

    let query = "semantic search with cosine similarity over vectors";
    let q = hash_embed(query, DIM);

    println!("RAG demo: {} docs, dim={}", store.len(), DIM);
    println!("query: {:?}", query);
    println!("top-3 results:");
    for (rank, (score, id, text)) in store.top_k(&q, 3).into_iter().enumerate() {
        println!("  {}. [{:.4}] {}: {}", rank + 1, score, id, text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![0.3, 0.1, 0.7, 0.5];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn top_k_finds_lexically_overlapping_doc() {
        const DIM: usize = 128;
        let mut store = EmbeddingStore::new();
        store.add("a", "bananas grow in tropical climates", hash_embed("bananas grow in tropical climates", DIM));
        store.add("b", "cosine similarity ranks vectors by angle", hash_embed("cosine similarity ranks vectors by angle", DIM));
        store.add("c", "the weather today is cold and rainy", hash_embed("the weather today is cold and rainy", DIM));

        let q = hash_embed("rank vectors by cosine similarity angle", DIM);
        let results = store.top_k(&q, 3);

        assert_eq!(results.len(), 3);
        // The doc that shares the most tokens with the query must rank first.
        assert_eq!(results[0].1, "b");
        // And scores must be in descending order.
        assert!(results[0].0 >= results[1].0);
        assert!(results[1].0 >= results[2].0);
    }

    #[test]
    fn hash_embed_is_deterministic() {
        let a = hash_embed("the quick brown fox", 64);
        let b = hash_embed("the quick brown fox", 64);
        assert_eq!(a, b);
    }

    #[test]
    fn hash_embed_is_l2_normalized() {
        let v = hash_embed("retrieval augmented generation in aria", 64);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm was {}", norm);
    }

    #[test]
    fn hash_embed_empty_text_is_zero_vector() {
        let v = hash_embed("   ", 16);
        assert_eq!(v.len(), 16);
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
