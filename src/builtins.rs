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
pub const BUILTIN_TYPES: &[&str] = &["Tensor", "Array", "Bytes"];

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
