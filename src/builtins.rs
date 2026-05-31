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

/// Built-in (opaque) type names that need no user `type` declaration.
pub const BUILTIN_TYPES: &[&str] = &["Tensor"];

/// Every built-in function as `(name, parameter types, return type)`.
pub fn signatures() -> Vec<(&'static str, Vec<Ty>, Ty)> {
    use Ty::*;
    // The opaque tensor handle, shared across all tensor builtins.
    let tensor = || Named("Tensor".to_string(), vec![]);
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
    ]
}

/// Look up a builtin's signature by name.
pub fn lookup(name: &str) -> Option<(Vec<Ty>, Ty)> {
    signatures()
        .into_iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, params, ret)| (params, ret))
}

/// All built-in function names.
pub fn names() -> Vec<&'static str> {
    signatures().into_iter().map(|(n, _, _)| n).collect()
}
