//! The single source of truth for Aria's built-in functions and types.
//!
//! Both backends consume this:
//!   * the type checker (`typeck`) reads `lookup` for a builtin's signature and
//!     `BUILTIN_TYPES` to accept built-in type names;
//!   * the interpreter (`interp`) implements each builtin's *behavior*, and a
//!     test asserts every name declared here is actually implemented.
//!
//! Keeping the signatures in one place means the two backends cannot silently
//! drift (e.g. a builtin the checker accepts but the interpreter doesn't know,
//! which would type-check and then fail at runtime).

use crate::ast::Ty;
use std::sync::OnceLock;

/// Built-in (opaque) type names that need no user `type` declaration.
/// `Array` is generic (`Array[T]`); `Tensor` and `Bytes` are nullary opaque
/// handles. `Bytes` is a flat, growable byte buffer (a byte = Int 0..255).
pub const BUILTIN_TYPES: &[&str] = &["Tensor", "Array", "Bytes", "Map", "Set", "Vector"];

/// Cached signature table, built once on first access. The table is on a hot
/// path (`lookup`/`names` are called per call-expression during lowering and
/// type-checking), so we avoid rebuilding (and re-allocating the owned `Ty`
/// strings) on every call.
static SIGNATURES: OnceLock<Vec<(&'static str, Vec<Ty>, Ty)>> = OnceLock::new();

/// Borrow the cached signature table, building it once on first use.
fn signatures_cached() -> &'static Vec<(&'static str, Vec<Ty>, Ty)> {
    SIGNATURES.get_or_init(build_signatures)
}

/// Every built-in function as `(name, parameter types, return type)`.
///
/// Returns a clone of the cached table for backward-compatible behavior; this
/// is called rarely, so the clone is acceptable.
pub fn signatures() -> Vec<(&'static str, Vec<Ty>, Ty)> {
    signatures_cached().clone()
}

/// Construct the signature table. Called once via `signatures_cached`.
fn build_signatures() -> Vec<(&'static str, Vec<Ty>, Ty)> {
    use Ty::*;
    // The opaque tensor handle, shared across all tensor builtins.
    let tensor = || Named("Tensor".to_string(), vec![]);
    // Generic array element type `T` and the array type `Array[T]`. A builtin
    // signature mentioning a `Ty::Var` is treated as generic by the checker,
    // which instantiates the var fresh per call site (see typeck `Expr::Call`).
    let elem = || Var("T".to_string());
    let array = || Named("Array".to_string(), vec![elem()]);
    // The nullary opaque byte-buffer handle, shared across the bytes builtins.
    let bytes = || Named("Bytes".to_string(), vec![]);
    // Ordered Map[K, V] and Set[T]. `K`/`T` are restricted to Int/Str by the
    // type checker (the two primitive totally-ordered types); `V` is fully
    // generic. The generic vars instantiate fresh per call site, like Array.
    let mkey = || Var("K".to_string());
    let mval = || Var("V".to_string());
    let map = || Named("Map".to_string(), vec![mkey(), mval()]);
    let setelem = || Var("T".to_string());
    let set = || Named("Set".to_string(), vec![setelem()]);
    // The nullary opaque dense-float-vector / embedding handle, shared across the
    // vector builtins. A `Vector` is an immutable dense buffer of `Float` (f64).
    let vector = || Named("Vector".to_string(), vec![]);
    let float_array = || Named("Array".to_string(), vec![Float]);
    let vector_array = || Named("Array".to_string(), vec![vector()]);
    vec![
        ("print_int", vec![Int], Unit),
        ("print_float", vec![Float], Unit),
        ("print_bool", vec![Bool], Unit),
        ("print_str", vec![Str], Unit),
        ("concat", vec![Str, Str], Str),
        ("int_to_str", vec![Int], Str),
        // ---- AI runtime primitives -----------------------------------------
        ("tensor_zeros", vec![Int, Int], tensor()),
        ("tensor_set", vec![tensor(), Int, Int, Float], tensor()),
        ("tensor_get", vec![tensor(), Int, Int], Float),
        ("tensor_rows", vec![tensor()], Int),
        ("tensor_cols", vec![tensor()], Int),
        // ---- Tensor <-> Vector bridge --------------------------------------
        // `tensor_row(t, i)` pulls row i of a 2D tensor out as a length-cols
        // Vector, widening each stored f32 to f64 (exact). Out-of-range i traps.
        // `tensor_from_rows(rows)` stacks an `Array[Vector]` of equal-length
        // vectors into a `[len, L]` tensor, narrowing each f64 to f32. Unequal
        // lengths trap; an empty array yields a 0x0 tensor. Identical on all
        // three backends (interp/native/wasm).
        ("tensor_row", vec![tensor(), Int], vector()),
        ("tensor_from_rows", vec![vector_array()], tensor()),
        ("matmul", vec![tensor(), tensor()], tensor()),
        ("transpose", vec![tensor()], tensor()),
        ("softmax", vec![tensor()], tensor()),
        ("relu", vec![tensor()], tensor()),
        ("embed_similarity", vec![Str, Str], Float),
        ("compressed_size", vec![Str], Int),
        ("neural_bits_per_byte", vec![Str], Float),
        // ---- Arrays (generic, functional with FBIP in-place reuse) ---------
        ("array_new", vec![], array()),
        ("array_len", vec![array()], Int),
        ("array_get", vec![array(), Int], elem()),
        ("array_set", vec![array(), Int, elem()], array()),
        ("array_push", vec![array(), elem()], array()),
        // ---- Bytes (flat byte buffer, FBIP in-place reuse like Array) -------
        // A byte is an Int 0..255. `set`/`push` of an out-of-range Int trap at
        // run time (range policy applied identically across all backends).
        ("bytes_new", vec![], bytes()),
        ("bytes_len", vec![bytes()], Int),
        ("bytes_get", vec![bytes(), Int], Int),
        ("bytes_set", vec![bytes(), Int, Int], bytes()),
        ("bytes_push", vec![bytes(), Int], bytes()),
        ("bytes_from_str", vec![Str], bytes()),
        ("bytes_to_str", vec![bytes()], Str),
        // ---- Ordered Map[K, V] (sorted by key; K is Int or Str) -------------
        // The read API is TOTAL (no Option type exists): `map_get_or` returns
        // its 3rd argument when the key is absent. `map_insert` replaces an
        // existing key's value. Entries are kept sorted by key for a
        // deterministic display/equality/iteration order across all backends.
        ("map_new", vec![], map()),
        ("map_insert", vec![map(), mkey(), mval()], map()),
        ("map_get_or", vec![map(), mkey(), mval()], mval()),
        ("map_has", vec![map(), mkey()], Bool),
        ("map_len", vec![map()], Int),
        ("map_remove", vec![map(), mkey()], map()),
        // Canonical textual rendering `Map[k1: v1, k2: v2]` (ascending key
        // order, empty `Map[]`) — identical byte-for-byte in every backend, so a
        // whole-map can be printed via `print_str(map_show(m))`.
        ("map_show", vec![map()], Str),
        // Enumeration into a plain `Array` so a map can be iterated with the
        // prelude array HOFs. Both arrays come out in ASCENDING key order (the
        // same deterministic order used for display/equality), so `map_keys` and
        // `map_values` are index-aligned. Consumes the map.
        ("map_keys", vec![map()], Named("Array".to_string(), vec![mkey()])),
        ("map_values", vec![map()], Named("Array".to_string(), vec![mval()])),
        // ---- Ordered Set[T] (sorted by element; T is Int or Str) ------------
        ("set_new", vec![], set()),
        ("set_add", vec![set(), setelem()], set()),
        ("set_has", vec![set(), setelem()], Bool),
        ("set_len", vec![set()], Int),
        ("set_remove", vec![set(), setelem()], set()),
        // Canonical textual rendering `Set[a, b, c]` (ascending order).
        ("set_show", vec![set()], Str),
        // Enumeration into a plain `Array` (ascending element order, the same
        // deterministic order used for display/equality) so a set can be iterated
        // with the prelude array HOFs. Consumes the set.
        ("set_to_array", vec![set()], Named("Array".to_string(), vec![setelem()])),
        // ---- Vector / Embedding (dense, immutable buffer of Float) ----------
        // A flat heap buffer of f64. `push`/`add`/`scale` are functional (the
        // oracle copies; the native backend reuses in place when unique).
        // `dot`/`cosine`/`add` on two vectors of UNEQUAL length is a clean
        // runtime error/trap in every backend. `cosine` returns 0.0 when either
        // operand has L2 norm 0 (never divides by zero). Out-of-range `get` traps.
        ("vec_new", vec![], vector()),
        ("vec_from_array", vec![float_array()], vector()),
        ("vec_to_array", vec![vector()], float_array()),
        ("vec_len", vec![vector()], Int),
        ("vec_get", vec![vector(), Int], Float),
        ("vec_push", vec![vector(), Float], vector()),
        ("vec_dot", vec![vector(), vector()], Float),
        ("vec_norm", vec![vector()], Float),
        ("vec_cosine", vec![vector(), vector()], Float),
        ("vec_add", vec![vector(), vector()], vector()),
        ("vec_sub", vec![vector(), vector()], vector()),
        ("vec_scale", vec![vector(), Float], vector()),
        // ---- Reverse-mode automatic differentiation -------------------------
        // `grad(f, x)` returns the gradient `∂f/∂x` of a scalar function of a
        // Vector, at the point `x`, by ONE reverse-mode (tape) backward pass —
        // O(1) in the number of inputs, the correct asymptotics for training.
        //
        // INTERPRETER-ONLY. `f` is function-typed (`Ty::Fn`), which the compiled
        // backends already reject (only the interpreter executes function
        // values). The interpreter evaluates `f` over a tracing Vector, records a
        // Wengert tape of every differentiable scalar/vector op, seeds the scalar
        // output's adjoint to 1.0, and sweeps the tape backward to read the input
        // adjoints. Supported differentiable op set (anything else inside `f`
        // raises a clean "grad: unsupported operation" error, never a panic):
        // Float `+ - * /` and unary negate; `vec_get`, `vec_dot`, `vec_add`,
        // `vec_sub`, `vec_scale`, `vec_norm`, and `vec_from_array`/`vec_push`
        // built from tracing elements.
        ("grad", vec![Fn(vec![vector()], Box::new(Float)), vector()], vector()),
    ]
}

/// Look up a builtin's signature by name.
pub fn lookup(name: &str) -> Option<(Vec<Ty>, Ty)> {
    signatures_cached()
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, params, ret)| (params.clone(), ret.clone()))
}

/// All built-in function names.
pub fn names() -> Vec<&'static str> {
    signatures_cached().iter().map(|(n, _, _)| *n).collect()
}
