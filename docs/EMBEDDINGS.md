# Embeddings in Aria — a real learned distributional model

Aria's `embed` / `embed_similarity` builtins are backed by a **real, learned,
count-based distributional embedding model** — not a hash. This document
describes the method, the corpus, the determinism guarantees, and how to use it.

## Honest scope

This is deliberately a **small model on a small corpus**: a ~13 KB bundled
English corpus and a 229-word vocabulary, reduced to 64-dimensional vectors. It
captures **genuine distributional semantics** (words that occur in similar
contexts get similar vectors), which is provably impossible for a hash. It is
**not** a large pretrained language model, and it only "knows" words that appear
in the bundled corpus. Out-of-vocabulary words contribute nothing.

The point is correctness of *kind*, not scale: this is the canonical count-based
word-embedding method (Levy & Goldberg, 2014, *"Neural Word Embedding as Implicit
Matrix Factorization"*), which is provably competitive with word2vec, implemented
with **zero external dependencies and no network access**.

## The method (`src/embed.rs`)

1. **Tokenize** the bundled corpus (`data/corpus.txt`, embedded at compile time
   via `include_str!`): lowercase, split on non-alphabetic characters, one
   sentence per line. Words occurring fewer than `MIN_COUNT` (2) times are
   dropped. Vocabulary order is fully deterministic (sorted by descending count,
   then lexicographically), independent of hash-map iteration order.
2. **Co-occurrence matrix**: a symmetric word×word count over a ±4-word window
   (windows never span across lines, so unrelated sentences don't bleed
   together).
3. **PPMI** (positive pointwise mutual information):
   `PPMI(i,j) = max(0, log( c_ij · N / (c_i · c_j) ))`. This turns raw counts
   into an association score — the standard weighting that makes the matrix
   semantic.
4. **Truncated SVD** of the (symmetric) PPMI matrix, hand-rolled with
   deterministic **subspace / block power iteration** plus Gram–Schmidt
   re-orthonormalization (`truncated_eig_sym`). The leading `DIM = 64`
   eigenpairs give word vectors `W = U · √Λ` — the Levy–Goldberg embedding.
   Each row is then L2-normalized.

A free-text string is embedded as the **L2-normalized mean** of its in-vocab
word vectors. `embed_similarity(a, b)` is the cosine of those two means.

Training is **lazy and cached**: the PPMI+SVD is computed once on first use and
stored in a `OnceLock`, so every later `embed` / `embed_similarity` call is a
fast table lookup + average. Training a corpus this size is sub-millisecond.

## Determinism

The embedding table is **byte-identical across runs and builds**. The only
source of randomness — the starting block of the SVD subspace iteration — is
seeded by a fixed-seed LCG (`Lcg::new(0x9E3779B97F4A7C15)`); no OS entropy is
ever consulted. The test `embed::tests::training_is_deterministic` trains the
corpus twice and asserts the word vectors are equal bit-for-bit.

## The builtins

| Builtin | Signature | Notes |
|---|---|---|
| `embed` | `(String) -> Vector` | The learned embedding as a first-class `Vector` (length 64). Composes with the retrieval prelude. |
| `embed_similarity` | `(String, String) -> Float` | Cosine of the two texts' learned embeddings. |

Both are **interpreter-only**. The learned vocabulary→vector table is a
Rust-runtime artifact, so the compiled backends (hand-emitted WASM, native-C)
reject `embed`/`embed_similarity` with a clean `Err` — the same
interpreter-only boundary used for the codec builtins (`compressed_size`,
`neural_bits_per_byte`). The `Vector` type itself and all `vec_*` ops *are*
supported on every backend; only the *production* of a learned embedding is
interpreter-side.

## Proof it is real (semantic structure, not a hash)

Word-level cosine over the learned table (a hash could not order these):

```
cat ~ dog   = 0.85     cat ~ car   = -0.10
king ~ queen = 0.86    king ~ bread = 0.01
apple ~ orange = 0.98  apple ~ king = 0.04
```

Nearest neighbors of seed words are semantically coherent:

```
cat:   dog, mouse, house, sleeps, ...
king:  queen, rules, kingdom, castle, ...
apple: orange, banana, sweet, fruit, ...
mother: father, family, daughter, ...
car:   road, drives, fast, long, ...
```

Sentence-level (`embed_similarity`):

```
embed_similarity("the cat is a small pet animal",
                 "the dog is a loyal pet animal")  = 0.94   (related)
embed_similarity("the cat is a small pet animal",
                 "the king rules the kingdom ...")  = 0.34   (unrelated)
```

## End-to-end retrieval (real embeddings → `Vector` → cosine search)

`embed` produces a `Vector`, which the retrieval prelude (`nearest`,
`nearest_score`, `similarities`) searches over an `Array[Vector]`:

```aria
fn main() -> Int = {
  let store: Array[Vector] = [
    embed("the dog is a loyal pet animal"),
    embed("the king rules the kingdom")
  ];
  -- a cat query retrieves the dog (pet/animal) document, not the king one
  nearest(store, embed("a cat is a small pet"))   -- => 0
}
```

`aria demo rag` runs a larger version of this over a 7-document corpus and
prints the cosine-ranked results.

## The corpus (`data/corpus.txt`)

An original, hand-curated, public-domain-style English corpus written for this
project. It is plain declarative sentences over everyday vocabulary — animals,
foods, colors, family, royalty, places, occupations, weather, vehicles, and
actions — arranged so that semantically related words share neighboring
contexts (e.g. `cat`/`dog` both appear as "the ___ is a pet"). No copyrighted
text is included. Editing the corpus and rebuilding regenerates the table
deterministically (the model trains from `include_str!("../data/corpus.txt")`).
