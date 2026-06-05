//! The Aria prelude: a small standard library of iteration / higher-order
//! operations over collections, written in Aria itself.
//!
//! These are ordinary generic, tail-recursive functions built on the array
//! primitives (`array_len`/`array_get`/`array_push`/`array_new`). Because they
//! are plain Aria source they monomorphize per use and run on EVERY backend with
//! no runtime support — the interpreter, the native (C) backend, and wasm all
//! get them for free. Unused prelude functions are dropped by monomorphization
//! (they are unreachable from `main`), so they add nothing to compiled output.
//!
//! Accumulator arrays are always seeded INTERNALLY (inside a function where the
//! element type parameter is in scope), never passed in as a bare `array_new()`
//! — that keeps every empty array's element type statically known to the
//! compiled backends.
//!
//! The prelude is APPENDED after the user's program (Aria resolves forward
//! references, and `main` may call a function defined later), so user source
//! keeps its original line numbers in error messages.

/// The prelude source, appended to every program by the CLI driver.
pub const SOURCE: &str = r#"
-- ===== Aria prelude: iteration & higher-order operations =====

-- Map `f` over an array, returning a new array of the results.
fn array_map[A, B](xs: Array[A], f: (A) -> B) -> Array[B] =
  array__map_go(xs, f, 0, array_new())
fn array__map_go[A, B](xs: Array[A], f: (A) -> B, i: Int, acc: Array[B]) -> Array[B] =
  if i >= array_len(xs) { acc }
  else { array__map_go(xs, f, i + 1, array_push(acc, f(array_get(xs, i)))) }

-- Left fold: `f(... f(f(init, xs[0]), xs[1]) ..., xs[n-1])`.
fn array_fold[A, B](xs: Array[A], init: B, f: (B, A) -> B) -> B =
  array__fold_go(xs, 0, init, f)
fn array__fold_go[A, B](xs: Array[A], i: Int, acc: B, f: (B, A) -> B) -> B =
  if i >= array_len(xs) { acc }
  else { array__fold_go(xs, i + 1, f(acc, array_get(xs, i)), f) }

-- Keep only the elements for which `keep` returns true.
fn array_filter[A](xs: Array[A], keep: (A) -> Bool) -> Array[A] =
  array__filter_go(xs, 0, array_new(), keep)
fn array__filter_go[A](xs: Array[A], i: Int, acc: Array[A], keep: (A) -> Bool) -> Array[A] =
  if i >= array_len(xs) { acc }
  else {
    let x = array_get(xs, i);
    if keep(x) { array__filter_go(xs, i + 1, array_push(acc, x), keep) }
    else { array__filter_go(xs, i + 1, acc, keep) }
  }

-- The integers `[0, 1, ..., n-1]` (empty when n <= 0).
fn range(n: Int) -> Array[Int] = array__range_go(0, n, array_new())
fn array__range_go(i: Int, n: Int, acc: Array[Int]) -> Array[Int] =
  if i >= n { acc } else { array__range_go(i + 1, n, array_push(acc, i)) }

-- ===== Embedding retrieval: nearest-neighbour search over a Vector store =====
-- An embedding STORE is an `Array[Vector]` (e.g. one Vector per document); a
-- QUERY is a Vector. These find the most similar stored embedding by cosine
-- similarity — the core retrieval step of a RAG / semantic-search pipeline —
-- built purely on `vec_cosine` and the array primitives.

-- Index of the store embedding most similar to `query` (-1 for an empty store).
fn nearest(store: Array[Vector], query: Vector) -> Int =
  embed__argmax_go(store, query, 0, 0 - 1, 0.0 - 2.0)
-- The best cosine similarity itself (-2.0 for an empty store, below cosine's -1).
fn nearest_score(store: Array[Vector], query: Vector) -> Float =
  embed__max_go(store, query, 0, 0.0 - 2.0)
-- The cosine of `query` against every stored embedding, in store order.
fn similarities(store: Array[Vector], query: Vector) -> Array[Float] =
  embed__sims_go(store, query, 0, array_new())

fn embed__argmax_go(store: Array[Vector], query: Vector, i: Int, bi: Int, bs: Float) -> Int =
  if i >= array_len(store) { bi }
  else {
    let s = vec_cosine(query, array_get(store, i));
    if s > bs { embed__argmax_go(store, query, i + 1, i, s) }
    else { embed__argmax_go(store, query, i + 1, bi, bs) }
  }
fn embed__max_go(store: Array[Vector], query: Vector, i: Int, bs: Float) -> Float =
  if i >= array_len(store) { bs }
  else {
    let s = vec_cosine(query, array_get(store, i));
    if s > bs { embed__max_go(store, query, i + 1, s) }
    else { embed__max_go(store, query, i + 1, bs) }
  }
fn embed__sims_go(store: Array[Vector], query: Vector, i: Int, acc: Array[Float]) -> Array[Float] =
  if i >= array_len(store) { acc }
  else {
    embed__sims_go(store, query, i + 1, array_push(acc, vec_cosine(query, array_get(store, i))))
  }
"#;

/// Append the prelude to a user program. Returns the combined source to lex.
pub fn wrap(user_src: &str) -> String {
    let mut s = String::with_capacity(user_src.len() + SOURCE.len() + 1);
    s.push_str(user_src);
    s.push('\n');
    s.push_str(SOURCE);
    s
}
