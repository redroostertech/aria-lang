//! Aria's hand-rolled WebAssembly emitter for the ANF IR (`src/ir.rs`).
//!
//! Scope (Phase 2a) — pure, heap-free:
//!   * Types: `Int -> i64`, `Bool -> i32`.
//!   * `IExpr`: `Let`, `Ret`.  `Bind`: `Atom`, `Prim`, `Unary`, `Call`, `If`.
//!
//! Scope (Phase 2b, this module) — heap-allocated ADTs with reference counting:
//!   * A `Ty::Named` (non-generic) ADT becomes a heap `Ref` (an i32 wasm32
//!     address). `Bind::Ctor` allocates a cell; `Bind::Match` loads the tag and
//!     dispatches an if/else chain, binding fields out of the cell; `IExpr::Dup`
//!     / `IExpr::Drop` are the reference-count ops (no-ops on unboxed values).
//!   * The backend runs `rc::insert_rc` so the compiled module manages its heap
//!     exactly like the IR interpreter (garbage-free: `__live() == 0` after an
//!     Int-returning `main`).
//!   * Linear memory: a Memory section (256 pages) + a bookkeeping region (bump
//!     pointer, live counter, per-arity free-list heads) + emitted runtime
//!     helpers `__alloc`/`__free`/`__dup`/`__drop`/`__live`. A cell at `p` uses
//!     8-byte slots: `[p+0]=rc`, `[p+8]=tag`, `[p+16+8*i]=field i`. `__drop`'s
//!     per-tag Ref-field knowledge is compiled in from the typed AST.
//!
//! In-place cell REUSE (FBIP): the rc pass emits same-arity `DropReuse`/
//! `CtorReuse` pairs; codegen exploits them. `__drop_reuse(ptr)` decrements rc
//! and, if the cell becomes unique-and-dead, releases its Ref CHILDREN but
//! RETAINS the slot, returning the pointer as a reuse token (else 0).
//! `CtorReuse` overwrites that retained slot in place (no `__alloc`, no `__live`
//! change, `__reuses`++); a null token falls back to a fresh `__alloc`. This
//! matches the IR interpreter exactly and preserves garbage-freeness.
//!
//! DEFERRED (clean `Err`, never a panic): Float / String / Unit, ADT fields of
//! those types, generic ADTs, structural ADT `==`/`!=`, builtin calls.
//!
//! Value types (i64 vs i32) come from two places:
//!   * Function signatures: read directly off the typed AST `Program`
//!     (`Item::Fn`'s `params`/`ret`), mapping `Ty::Int -> i64`, `Ty::Bool -> i32`.
//!   * Local (let-bound) variables: inferred structurally from each `Bind`
//!     (see `bind_type`), given the already-known types of params + earlier lets.
//!
//! The wasm binary is emitted by hand, byte for byte, using only the standard
//! library — no `wat`, no `walrus`, no external crates. LEB128 (both unsigned
//! and signed) is implemented below. The output is validated by *differential
//! testing*: the compiled module is run under Node's built-in `WebAssembly` and
//! its result must equal the tree-walking interpreter's (the reference oracle).
//!
//! Overflow semantics: integer overflow is a *defined error* in Aria. The IR
//! interpreter does checked arithmetic (overflow -> `Err`); to match it, this
//! backend routes `Add`/`Sub`/`Mul` and unary `Neg` (on `Int`) through emitted
//! helper functions (`__add_ovf` etc.) that detect signed-i64 overflow and
//! execute `unreachable` (a wasm trap) on overflow. A trap surfaces in Node as a
//! thrown error, which `aria wasm-run` reports as `TRAP` — i.e. agreement with
//! the interpreter's `Err`. Division/remainder already trap natively in wasm on
//! `/0` and `i64::MIN / -1`, matching the interpreter, so they are left as-is.

use std::collections::HashMap;

use crate::ast::{BinOp, Item, Program, Ty, UnOp};
use crate::ir::{self, Atom, Bind, IExpr, IFn};

/// A wasm-level value type. On the operand stack we keep `Int` as i64, `Bool`
/// as i32, and a heap reference (`Ref`) as i32 (a wasm32 linear-memory address).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum WType {
    I64, // Aria Int
    I32, // Aria Bool
    F64, // Aria Float (unboxed IEEE-754 double)
    Ref, // pointer into linear memory (i32 address) — Phase 2b ADT cell
    Str, // pointer into linear memory (i32 address) — Phase 2d String object
    Tensor, // pointer into linear memory (i32 address) — Phase 2f Tensor object
    F32, // INTERNAL ONLY: a scalar f32 used for Tensor-helper locals/scratch.
         // No Aria value ever has this type (Tensor data is f32 but is read out
         // as f64); it exists purely so helper bodies can declare f32 locals.
}

impl WType {
    /// The valtype byte used in the binary (`0x7E` = i64, `0x7F` = i32,
    /// `0x7C` = f64). `Ref`/`Str` are wasm32 addresses, so they are i32.
    fn byte(self) -> u8 {
        match self {
            WType::I64 => 0x7E,
            WType::F64 => 0x7C,
            WType::F32 => 0x7D,
            WType::I32 | WType::Ref | WType::Str | WType::Tensor => 0x7F,
        }
    }

    /// Map an AST type to a wasm value type. `Int -> i64`, `Bool -> i32`,
    /// `Float -> f64`, and a *non-generic* named ADT type -> `Ref` (a heap
    /// pointer). Generic type variables and Unit remain outside the subset.
    fn from_ty(ty: &Ty) -> Result<WType, String> {
        match ty {
            Ty::Int => Ok(WType::I64),
            Ty::Bool => Ok(WType::I32),
            Ty::Float => Ok(WType::F64),
            // An immutable, reference-counted heap String (Phase 2d).
            Ty::Str => Ok(WType::Str),
            // The opaque, reference-counted heap Tensor (Phase 2f). It is the
            // builtin ADT name `Tensor`; distinguish it before the generic
            // named-ADT case so it gets its own heap object kind.
            Ty::Named(n, args) if n == "Tensor" && args.is_empty() => Ok(WType::Tensor),
            // A named ADT becomes a heap reference. (Generics — args present —
            // are out of the 2b subset and rejected below.)
            Ty::Named(_, args) if args.is_empty() => Ok(WType::Ref),
            // A closure value is a heap reference (cell: tag = lambda id,
            // fields = captures).
            Ty::Fn(_, _) => Ok(WType::Ref),
            other => Err(format!(
                "wasm backend: unsupported type `{:?}` (subset: Int/Bool/String and non-generic ADTs)",
                other
            )),
        }
    }
}

/// A function's wasm-level signature, derived from the typed AST.
struct FnSig {
    params: Vec<WType>,
    ret: WType,
}

// ---- ADT / heap metadata (Phase 2b) -------------------------------------

/// Static information about one constructor, compiled in from `Item::Type`.
#[derive(Debug, Clone)]
struct CtorInfo {
    /// Distinct integer tag, assigned per program in declaration order.
    tag: i64,
    /// wasm type of each field (Int/Bool/Ref). Length = field count = arity.
    field_types: Vec<WType>,
}

impl CtorInfo {
    fn arity(&self) -> usize {
        self.field_types.len()
    }
    /// Indices of the fields that are heap references (must be dropped via
    /// `__drop`).
    fn ref_fields(&self) -> Vec<usize> {
        self.field_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| if *t == WType::Ref { Some(i) } else { None })
            .collect()
    }
    /// Indices of the fields that are heap Strings (dropped via `__drop_str`).
    fn str_fields(&self) -> Vec<usize> {
        self.field_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| if *t == WType::Str { Some(i) } else { None })
            .collect()
    }
    /// True if this constructor has any reference-managed field (Ref or Str),
    /// i.e. dropping a dead cell of this tag must release children.
    fn has_managed_fields(&self) -> bool {
        !self.ref_fields().is_empty() || !self.str_fields().is_empty()
    }
}

/// Program-wide ADT layout knowledge built from every `Item::Type`.
struct CtorTable {
    /// constructor name -> info (tag + field types).
    by_name: HashMap<String, CtorInfo>,
    /// Maximum constructor arity in the whole program (sizes the free-list array).
    max_arity: usize,
}

impl CtorTable {
    /// Build from the typed AST. Each ADT field must be Int/Bool/non-generic
    /// ADT; anything else (Float/String/generic) yields a clean `Err`.
    fn build(program: &Program) -> Result<CtorTable, String> {
        let mut by_name = HashMap::new();
        let mut max_arity = 0usize;
        let mut tag: i64 = 0;
        for item in &program.items {
            if let Item::Type(t) = item {
                if !t.params.is_empty() {
                    return Err(format!(
                        "wasm backend: generic type `{}` is outside the 2b subset",
                        t.name
                    ));
                }
                for v in &t.variants {
                    let field_types = v
                        .fields
                        .iter()
                        .map(WType::from_ty)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| format!("type `{}` ctor `{}`: {}", t.name, v.name, e))?;
                    // A Tensor field inside an ADT cell is outside the subset:
                    // __drop has no per-tag knowledge to release Tensor children,
                    // so it would leak. Reject cleanly (never a panic).
                    if field_types.contains(&WType::Tensor) {
                        return Err(format!(
                            "wasm backend: type `{}` ctor `{}` has a Tensor field (outside the subset)",
                            t.name, v.name
                        ));
                    }
                    max_arity = max_arity.max(field_types.len());
                    by_name.insert(v.name.clone(), CtorInfo { tag, field_types });
                    tag += 1;
                }
            }
        }
        Ok(CtorTable { by_name, max_arity })
    }

    fn get(&self, name: &str) -> Result<&CtorInfo, String> {
        self.by_name
            .get(name)
            .ok_or_else(|| format!("wasm backend: unknown constructor `{}`", name))
    }
}

// ---- linear-memory layout (Phase 2b) ------------------------------------
//
// We reserve a small bookkeeping region at the very start of linear memory:
//   [0]  bump pointer (i32): next never-yet-allocated address.
//   [8]  live-cell counter (i64): incremented on a fresh alloc, decremented on
//        free. Exported via `__live`.
//   [16] reuse counter (i64): incremented on each in-place cell reuse (FBIP).
//        Exported via `__reuses`.
//   [24] free-list heads, one i32 per arity 0..=max_arity (segregated by size).
// Cells are allocated above `HEAP_BASE` (after the free-list array).
//
// A cell at pointer `p` uses 8-byte slots:
//   [p+0]  rc   (i64)
//   [p+8]  tag  (i64)
//   [p+16+8*i] field i (always 8 bytes; Int=i64, Bool=0/1 in i64, Ref=zero-
//          extended i32 address). A freed cell reuses [p+8] as its free-list
//          "next" link.

// 1024 pages = 64 MiB of linear memory. Sized so a ~1M-cell heap program (e.g.
// building then folding a 1,000,000-element list) fits without exhausting the
// bump allocator and trapping; tail-recursion itself uses constant stack.
const MEM_PAGES: u64 = 1024;
const BUMP_PTR_ADDR: u64 = 0; // i32
const LIVE_ADDR: u64 = 8; // i64 (8-aligned)
const REUSES_ADDR: u64 = 16; // i64 (8-aligned): in-place reuse counter (FBIP)
const FREELIST_BASE: u64 = 24; // i32 per arity, 4 bytes each
const CELL_HEADER: u64 = 16; // rc(8) + tag(8)
const SLOT: u64 = 8;

// ---- String heap objects (Phase 2d) -------------------------------------
//
// A String is an immutable, reference-counted heap object, distinct from an
// ADT cell. Layout at pointer `p`:
//   [p+0]  rc   (i64)
//   [p+8]  len  (i64)   — UTF-8 byte length
//   [p+16 .. p+16+len]  the raw UTF-8 bytes
// A String has no Ref/Str children, so dropping it at rc 0 just frees it.
//
// Distinguishing Strings from ADT cells in `__drop`: rather than overload the
// tag word (which a String repurposes as `len`), Strings get their OWN runtime
// helpers `__dup_str`/`__drop_str`. The backend knows statically (from the AST
// type) whether a value is an ADT (`Ref`) or a String (`Str`), so it calls the
// matching dup/drop. For a String FIELD inside an ADT cell, `__drop`'s per-tag
// recursive field-release calls `__drop_str` on Str fields and `__drop` on Ref
// fields (the per-field kind is compiled in from the typed AST).
//
// Allocation: Strings are variable-size. `__alloc_str(len)` rounds the total
// object size (16 + len) up to an 8-byte boundary, computes a "size class" =
// (rounded_total - CELL_HEADER) / SLOT (i.e. an effective field count), and
// reuses the existing segregated free-list `__alloc`/`__free` machinery keyed
// on that size class. Thus String allocs/frees flow through the SAME bump +
// free-list allocator and the SAME `__live` counter as ADT cells (+1 on alloc,
// -1 on free), keeping the garbage-free invariant intact.
const STR_HEADER: u64 = 16; // rc(8) + len(8)

/// Largest String size class that gets an exact-size free-list bucket. The
/// free-list array is sized to `max(max_arity, STR_MAX_CLASS) + 1`. A String
/// whose size class exceeds this is bump-allocated and, on free, decrements
/// `__live` but is not returned to a bucket (still garbage-free by `__live`,
/// just not reclaimed). Covers strings up to ~`(STR_MAX_CLASS*8 - 8)` bytes.
const STR_MAX_CLASS: u64 = 64;

// ---- Tensor heap objects (Phase 2f) -------------------------------------
//
// A Tensor is an opaque, reference-counted heap object, distinct from ADT
// cells and Strings. It mirrors the interpreter's `crate::tensor::Tensor`
// (row-major, f32 storage). Layout at pointer `p`:
//   [p+0]   rc    (i64)
//   [p+8]   rows  (i64)
//   [p+16]  cols  (i64)
//   [p+24 .. p+24+4*rows*cols]  row-major f32 elements
// We store f32 (not f64) so `tensor_get` reproduces the interpreter's exact
// stored precision (it does `*v as f32` on set and `t.at(..) as f64` on get).
//
// A Tensor has no Ref/Str/Tensor children, so dropping it at rc 0 just frees
// it (decrementing `__live`). Distinguishing Tensors from ADT cells / Strings
// in drop: like Strings, Tensors get their OWN runtime helpers
// `__alloc_tensor`/`__drop_tensor`. The backend knows statically (from the AST
// type) whether a value is a Tensor (`WType::Tensor`) and calls the matching
// dup/drop (a Tensor reuses `__dup_str`'s rc-bump shape via `DupStr`-like code,
// but for clarity gets its own helpers).
//
// Allocation: a Tensor is variable-size. `__alloc_tensor(rows, cols)` rounds
// the total object size (24 + 4*rows*cols) up to an 8-byte boundary, computes a
// size class = (rounded_total - CELL_HEADER) / SLOT, clamps it to
// TENSOR_MAX_CLASS, and reuses the segregated `__alloc`/`__free` machinery on
// that class — so Tensor allocs/frees flow through the SAME bump + free-list
// allocator and the SAME `__live` counter (garbage-free invariant intact).
const TENSOR_HEADER: u64 = 24; // rc(8) + rows(8) + cols(8)

/// Largest Tensor size class that gets an exact-size free-list bucket. The
/// free-list array is sized to `max(max_arity, STR_MAX_CLASS, TENSOR_MAX_CLASS)
/// + 1`. A Tensor whose class exceeds this is bump-allocated and, on free,
/// decrements `__live` but is not bucketed (still garbage-free by `__live`).
const TENSOR_MAX_CLASS: u64 = 256;

/// The emitted checked-arithmetic helper functions. Each detects signed-i64
/// overflow and executes `unreachable` (a wasm trap) on overflow; otherwise it
/// returns the wrapped result. They are appended to the module after all user
/// functions, in this fixed order, so their function indices are
/// `n_user_fns + (helper as offset)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OvfHelper {
    Add,
    Sub,
    Mul,
    Neg,
}

impl OvfHelper {
    /// Fixed order in which helpers are appended after user functions.
    const ALL: [OvfHelper; 4] = [OvfHelper::Add, OvfHelper::Sub, OvfHelper::Mul, OvfHelper::Neg];

    /// Slot offset of this helper relative to the first helper index.
    fn offset(self) -> u32 {
        match self {
            OvfHelper::Add => 0,
            OvfHelper::Sub => 1,
            OvfHelper::Mul => 2,
            OvfHelper::Neg => 3,
        }
    }

    /// `(params, ret)` wasm signature of the helper.
    fn sig(self) -> (Vec<WType>, WType) {
        match self {
            OvfHelper::Add | OvfHelper::Sub | OvfHelper::Mul => {
                (vec![WType::I64, WType::I64], WType::I64)
            }
            OvfHelper::Neg => (vec![WType::I64], WType::I64),
        }
    }
}

/// The emitted heap-runtime helper functions (Phase 2b). Appended after the
/// overflow helpers, in this fixed order, so their indices are
/// `heap_base + offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeapHelper {
    Alloc,     // (nfields:i32) -> ptr:i32
    Free,      // (ptr:i32, nfields:i32) -> ()  [we give it an i32 dummy return]
    Dup,       // (ptr:i32) -> ()
    Drop,      // (ptr:i32) -> ()
    Live,      // () -> i64
    DropReuse, // (ptr:i32) -> token:i32  (the cell ptr if unique-and-dead, else 0)
    Reuses,    // () -> i64  (in-place reuse counter)
    // ---- String runtime (Phase 2d) ----
    AllocStr, // (len:i32) -> ptr:i32   alloc a String object (rc=1, len set, bytes uninit)
    DupStr,   // (ptr:i32) -> ()        rc++ on a String
    DropStr,  // (ptr:i32) -> ()        rc--; free at 0 (no children)
    Concat,   // (a:i32, b:i32) -> ptr:i32   new String = a ++ b
    IntToStr, // (n:i64) -> ptr:i32     decimal UTF-8 of n
    StrEq,    // (a:i32, b:i32) -> i32   len+byte equality (1/0)
    StrLit,   // (data_addr:i32, len:i32) -> ptr:i32   alloc + copy a literal
    // ---- structural ADT equality (Phase 2e) ----
    Eq, // (a:i32, b:i32) -> i32   recursive structural ADT equality (1/0)
    // ---- Tensor runtime (Phase 2f) ----
    AllocTensor, // (rows:i32, cols:i32) -> ptr:i32  alloc a Tensor (rc=1, hdr set, data uninit)
    DupTensor,   // (ptr:i32) -> i32   rc++ on a Tensor
    DropTensor,  // (ptr:i32) -> i32   rc--; free at 0 (no children)
    TensorZeros, // (rows:i64, cols:i64) -> ptr:i32  zeroed Tensor (traps on negative/overflow)
    TensorSet,   // (t:i32, r:i64, c:i64, v:f64) -> ptr:i32  clone + write one element
    TensorGet,   // (t:i32, r:i64, c:i64) -> f64   read one element (traps OOB), consumes t
    TensorRows,  // (t:i32) -> i64    rows header, consumes t
    TensorCols,  // (t:i32) -> i64    cols header, consumes t
    Matmul,      // (a:i32, b:i32) -> ptr:i32   matrix multiply, consumes a and b (traps on shape)
    Transpose,   // (t:i32) -> ptr:i32   2D transpose, consumes t
    Relu,        // (t:i32) -> ptr:i32   elementwise relu, consumes t
    Softmax,     // (t:i32) -> ptr:i32   row-wise softmax (uses env.exp), consumes t
    EmbedSim,    // (a:i32, b:i32) -> f64  embed_similarity, consumes both Strings
    HashEmbed,   // (s:i32, vec_addr:i32) -> () write a dim-64 normalized embedding to vec_addr
}

impl HeapHelper {
    const ALL: [HeapHelper; 29] = [
        HeapHelper::Alloc,
        HeapHelper::Free,
        HeapHelper::Dup,
        HeapHelper::Drop,
        HeapHelper::Live,
        HeapHelper::DropReuse,
        HeapHelper::Reuses,
        HeapHelper::AllocStr,
        HeapHelper::DupStr,
        HeapHelper::DropStr,
        HeapHelper::Concat,
        HeapHelper::IntToStr,
        HeapHelper::StrEq,
        HeapHelper::StrLit,
        HeapHelper::Eq,
        HeapHelper::AllocTensor,
        HeapHelper::DupTensor,
        HeapHelper::DropTensor,
        HeapHelper::TensorZeros,
        HeapHelper::TensorSet,
        HeapHelper::TensorGet,
        HeapHelper::TensorRows,
        HeapHelper::TensorCols,
        HeapHelper::Matmul,
        HeapHelper::Transpose,
        HeapHelper::Relu,
        HeapHelper::Softmax,
        HeapHelper::EmbedSim,
        HeapHelper::HashEmbed,
    ];

    fn offset(self) -> u32 {
        match self {
            HeapHelper::Alloc => 0,
            HeapHelper::Free => 1,
            HeapHelper::Dup => 2,
            HeapHelper::Drop => 3,
            HeapHelper::Live => 4,
            HeapHelper::DropReuse => 5,
            HeapHelper::Reuses => 6,
            HeapHelper::AllocStr => 7,
            HeapHelper::DupStr => 8,
            HeapHelper::DropStr => 9,
            HeapHelper::Concat => 10,
            HeapHelper::IntToStr => 11,
            HeapHelper::StrEq => 12,
            HeapHelper::StrLit => 13,
            HeapHelper::Eq => 14,
            HeapHelper::AllocTensor => 15,
            HeapHelper::DupTensor => 16,
            HeapHelper::DropTensor => 17,
            HeapHelper::TensorZeros => 18,
            HeapHelper::TensorSet => 19,
            HeapHelper::TensorGet => 20,
            HeapHelper::TensorRows => 21,
            HeapHelper::TensorCols => 22,
            HeapHelper::Matmul => 23,
            HeapHelper::Transpose => 24,
            HeapHelper::Relu => 25,
            HeapHelper::Softmax => 26,
            HeapHelper::EmbedSim => 27,
            HeapHelper::HashEmbed => 28,
        }
    }

    /// `(params, ret)` wasm signature. `Free`/`Dup`/`Drop`/`DupStr`/`DropStr`
    /// are logically void; to keep the module's "exactly one result" invariant
    /// we make them return an i32 (a dummy 0), and the caller `drop`s it.
    fn sig(self) -> (Vec<WType>, WType) {
        match self {
            HeapHelper::Alloc => (vec![WType::I32], WType::I32),
            HeapHelper::Free => (vec![WType::I32, WType::I32], WType::I32),
            HeapHelper::Dup => (vec![WType::I32], WType::I32),
            HeapHelper::Drop => (vec![WType::I32], WType::I32),
            HeapHelper::Live => (vec![], WType::I64),
            HeapHelper::DropReuse => (vec![WType::I32], WType::I32),
            HeapHelper::Reuses => (vec![], WType::I64),
            HeapHelper::AllocStr => (vec![WType::I32], WType::I32),
            HeapHelper::DupStr => (vec![WType::I32], WType::I32),
            HeapHelper::DropStr => (vec![WType::I32], WType::I32),
            HeapHelper::Concat => (vec![WType::I32, WType::I32], WType::I32),
            HeapHelper::IntToStr => (vec![WType::I64], WType::I32),
            HeapHelper::StrEq => (vec![WType::I32, WType::I32], WType::I32),
            HeapHelper::StrLit => (vec![WType::I32, WType::I32], WType::I32),
            HeapHelper::Eq => (vec![WType::I32, WType::I32], WType::I32),
            HeapHelper::AllocTensor => (vec![WType::I32, WType::I32], WType::I32),
            HeapHelper::DupTensor => (vec![WType::I32], WType::I32),
            HeapHelper::DropTensor => (vec![WType::I32], WType::I32),
            HeapHelper::TensorZeros => (vec![WType::I64, WType::I64], WType::I32),
            HeapHelper::TensorSet => {
                (vec![WType::I32, WType::I64, WType::I64, WType::F64], WType::I32)
            }
            HeapHelper::TensorGet => (vec![WType::I32, WType::I64, WType::I64], WType::F64),
            HeapHelper::TensorRows => (vec![WType::I32], WType::I64),
            HeapHelper::TensorCols => (vec![WType::I32], WType::I64),
            HeapHelper::Matmul => (vec![WType::I32, WType::I32], WType::I32),
            HeapHelper::Transpose => (vec![WType::I32], WType::I32),
            HeapHelper::Relu => (vec![WType::I32], WType::I32),
            HeapHelper::Softmax => (vec![WType::I32], WType::I32),
            HeapHelper::EmbedSim => (vec![WType::I32, WType::I32], WType::F64),
            HeapHelper::HashEmbed => (vec![WType::I32, WType::I32], WType::I32),
        }
    }
}

const I64_MIN: i64 = -9223372036854775808;

// ---- LEB128 (implemented by hand, no deps) ------------------------------

/// Unsigned LEB128.
fn leb_u(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
}

/// Signed LEB128.
fn leb_s(mut v: i64, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7; // arithmetic shift (sign-extending)
        let sign_bit_set = byte & 0x40 != 0;
        let done = (v == 0 && !sign_bit_set) || (v == -1 && sign_bit_set);
        out.push(if done { byte } else { byte | 0x80 });
        if done {
            break;
        }
    }
}

// ---- linear-memory access opcodes ---------------------------------------
// memarg = align (LEB u32) then offset (LEB u32). We use natural alignment.

/// `i64.store` with the given byte offset (3 = align 2^3 = 8 bytes).
fn i64_store(offset: u64, out: &mut Vec<u8>) {
    out.push(0x37); // i64.store
    leb_u(3, out); // align
    leb_u(offset, out); // offset
}

/// `i64.load` with the given byte offset.
fn i64_load(offset: u64, out: &mut Vec<u8>) {
    out.push(0x29); // i64.load
    leb_u(3, out);
    leb_u(offset, out);
}

/// `f64.store` with the given byte offset (3 = align 2^3 = 8 bytes).
fn f64_store(offset: u64, out: &mut Vec<u8>) {
    out.push(0x39); // f64.store
    leb_u(3, out); // align
    leb_u(offset, out); // offset
}

/// `f64.load` with the given byte offset (align 2^3 = 8 bytes).
fn f64_load(offset: u64, out: &mut Vec<u8>) {
    out.push(0x2B); // f64.load
    leb_u(3, out);
    leb_u(offset, out);
}

/// `f32.store` with the given byte offset (2 = align 2^2 = 4 bytes).
fn f32_store(offset: u64, out: &mut Vec<u8>) {
    out.push(0x38); // f32.store
    leb_u(2, out);
    leb_u(offset, out);
}

/// `f32.load` with the given byte offset (align 2^2 = 4 bytes).
fn f32_load(offset: u64, out: &mut Vec<u8>) {
    out.push(0x2A); // f32.load
    leb_u(2, out);
    leb_u(offset, out);
}

/// `i32.store` with the given byte offset (2 = align 2^2 = 4 bytes).
fn i32_store(offset: u64, out: &mut Vec<u8>) {
    out.push(0x36); // i32.store
    leb_u(2, out);
    leb_u(offset, out);
}

/// `i32.load` with the given byte offset.
fn i32_load(offset: u64, out: &mut Vec<u8>) {
    out.push(0x28); // i32.load
    leb_u(2, out);
    leb_u(offset, out);
}

/// `i32.store8` (single byte) with the given offset (align 2^0 = 1).
fn i32_store8(offset: u64, out: &mut Vec<u8>) {
    out.push(0x3A); // i32.store8
    leb_u(0, out);
    leb_u(offset, out);
}

/// `i32.load8_u` (zero-extended byte) with the given offset (align 2^0 = 1).
fn i32_load8_u(offset: u64, out: &mut Vec<u8>) {
    out.push(0x2D); // i32.load8_u
    leb_u(0, out);
    leb_u(offset, out);
}

/// Emit a length-prefixed byte vector (LEB u32 count of *bytes*, then bytes).
fn vec_bytes(content: &[u8], out: &mut Vec<u8>) {
    leb_u(content.len() as u64, out);
    out.extend_from_slice(content);
}

/// Emit a section: id byte, then LEB length of the content, then the content.
fn section(id: u8, content: &[u8], out: &mut Vec<u8>) {
    out.push(id);
    leb_u(content.len() as u64, out);
    out.extend_from_slice(content);
}

// ---- type environment for a single function body ------------------------

/// Tracks, per IR variable, its wasm type and its wasm local index. Params
/// occupy the first local slots (in order), then let-bound variables.
struct LocalEnv<'a> {
    types: HashMap<String, WType>,
    index: HashMap<String, u32>,
    /// Types of the *additional* locals (the let-bound ones), in slot order.
    locals: Vec<WType>,
    n_params: u32,
    sigs: &'a HashMap<String, FnSig>,
    /// Function index of the first overflow helper (`OvfHelper::Add`).
    ovf_base: u32,
    /// Function index of the first heap helper (`HeapHelper::Alloc`).
    heap_base: u32,
    /// Program-wide ADT layout (constructor tags + field types).
    ctors: &'a CtorTable,
    /// Scratch i32 local index used to hold a cell pointer while storing its
    /// fields. Allocated lazily (the function may have no constructors).
    scratch_ptr: Option<u32>,
    /// String-literal pool: literal bytes -> the data-segment address of the
    /// raw UTF-8 bytes (in read-only data). A literal is materialized at runtime
    /// by `__alloc_str(len)` + a copy from this address.
    str_lits: &'a HashMap<Vec<u8>, u64>,
    /// Wasm function index of the imported `env.print_str` (always 0).
    print_str_idx: u32,
    /// Wasm function index of the imported `env.print_float` (always 1).
    print_float_idx: u32,
    /// Wasm function index of the imported `env.print_int` (always 2).
    print_int_idx: u32,
    /// Wasm function index of the imported `env.print_bool` (always 3).
    print_bool_idx: u32,
    /// Wasm function index of the imported `env.exp` (always 4); used by the
    /// Tensor `softmax` helper for a libm-faithful `exp`.
    exp_idx: u32,
    /// Current control-block nesting depth during code emission (each `if`/`loop`
    /// pushes a level). Used to compute the `br` relative target of a `TailCall`
    /// back to the enclosing tail-loop. Only meaningful during codegen.
    block_depth: u32,
    /// Set IFF the function being emitted is self-tail-recursive: the param
    /// `(local_index, wasm_type)` list and the `block_depth` of the enclosing
    /// `loop` (its `br` target). A `TailCall` reassigns the params and `br`s to it.
    tail: Option<TailCtx>,
    /// The enclosing function's declared return type. Used as the result type of
    /// an `if`/`match` ALL of whose branches diverge (every arm a `TailCall`):
    /// the value is never produced, but the block still needs a blocktype, and
    /// the function return type is the consistent choice.
    fn_ret: WType,
    /// Closure (lifted-lambda) dispatch metadata: the closure-tag base, each
    /// lambda's closure tag, and the `call_indirect` type index for every
    /// closure machine-signature.
    closures: &'a ClosureWasm,
}

/// Lifted-lambda dispatch metadata for the wasm backend.
struct ClosureWasm {
    /// First closure tag (one past the last constructor tag) so closure tags
    /// never collide with ADT constructor tags. A closure cell stores
    /// `base + table_index` in its tag word; `call_indirect` recovers the table
    /// index as `tag - base`.
    base: i64,
    /// Lambda name -> its closure tag (`base + table_index`).
    tags: HashMap<String, i64>,
    /// Closure machine-signature `(i32 closure :: params, ret)` -> the wasm type
    /// index used by `call_indirect`. Since a defined function's type index
    /// equals its function index, this is the first lifted lambda with that
    /// signature.
    sig_typeidx: HashMap<(Vec<WType>, WType), u32>,
}

/// Per-closure-tag information `__drop` needs to release a dead closure cell:
/// its tag, capture count (for `__free`), and the indices + kinds of its
/// reference-counted captures (`Ref` -> `__drop`, `Str` -> `__drop_str`).
struct ClosureDropInfo {
    tag: i64,
    ncaps: usize,
    managed: Vec<(usize, WType)>,
}

/// Tail-loop context for a self-tail-recursive wasm function.
#[derive(Clone)]
struct TailCtx {
    /// `(local index, wasm type)` of each parameter, in order.
    params: Vec<(u32, WType)>,
    /// The `block_depth` at which the enclosing `loop` opened (its label level).
    loop_depth: u32,
    /// The function's declared return type (the loop block's result valtype).
    ret: WType,
}

impl<'a> LocalEnv<'a> {
    /// The function index of an overflow helper.
    fn ovf_index(&self, h: OvfHelper) -> u32 {
        self.ovf_base + h.offset()
    }

    /// The function index of a heap helper.
    fn heap_index(&self, h: HeapHelper) -> u32 {
        self.heap_base + h.offset()
    }

    /// Obtain (allocating on first use) the scratch i32 local for cell pointers.
    fn scratch(&mut self) -> u32 {
        if let Some(s) = self.scratch_ptr {
            return s;
        }
        let idx = self.n_params + self.locals.len() as u32;
        self.locals.push(WType::I32);
        self.scratch_ptr = Some(idx);
        idx
    }

    /// Allocate a brand-new anonymous i32 local slot (e.g. a string-compare
    /// temporary). Always fresh, never reused, so callers can hold two at once.
    fn fresh_i32(&mut self) -> u32 {
        let idx = self.n_params + self.locals.len() as u32;
        self.locals.push(WType::I32);
        idx
    }

    fn var_type(&self, name: &str) -> Result<WType, String> {
        self.types
            .get(name)
            .copied()
            .ok_or_else(|| format!("wasm backend: unbound variable `{}`", name))
    }

    fn var_index(&self, name: &str) -> Result<u32, String> {
        self.index
            .get(name)
            .copied()
            .ok_or_else(|| format!("wasm backend: unbound variable `{}`", name))
    }

    /// Allocate a fresh local slot for a let-bound variable of the given type.
    fn add_local(&mut self, name: &str, ty: WType) {
        let idx = self.n_params + self.locals.len() as u32;
        self.locals.push(ty);
        self.types.insert(name.to_string(), ty);
        self.index.insert(name.to_string(), idx);
    }
}

// ---- atom / type inference ----------------------------------------------

/// The wasm type of an atom. Float/Str/Unit literals are unsupported.
fn atom_type(a: &Atom, env: &LocalEnv) -> Result<WType, String> {
    match a {
        Atom::Int(_) => Ok(WType::I64),
        Atom::Bool(_) => Ok(WType::I32),
        Atom::Var(n) => env.var_type(n),
        Atom::Float(_) => Ok(WType::F64),
        Atom::Str(_) => Ok(WType::Str),
        Atom::Unit => Err("wasm backend: Unit is outside the 2a subset".into()),
    }
}

/// Infer the wasm result type of a `Bind` without emitting code.
fn bind_type(bind: &Bind, env: &LocalEnv) -> Result<WType, String> {
    match bind {
        Bind::Atom(a) => atom_type(a, env),
        // A closure value is a heap reference; applying one yields the lambda's
        // return type, attached by monomorphization.
        Bind::MakeClosure(_, _) => Ok(WType::Ref),
        Bind::ApplyClosure(_, _, ret) => match ret {
            Some(t) => WType::from_ty(t),
            None => Err("wasm backend: closure application missing its result type".into()),
        },
        Bind::Prim(op, l, _) => Ok(match op {
            // Arithmetic keeps the operand type: Int -> i64, Float -> f64.
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => atom_type(l, env)?,
            // Comparisons and logical ops produce a Bool (i32).
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And
            | BinOp::Or => WType::I32,
        }),
        Bind::Unary(op, a) => match op {
            UnOp::Neg => atom_type(a, env), // numeric negation keeps the operand type
            UnOp::Not => Ok(WType::I32),    // Bool Not
        },
        Bind::Call(name, _) => {
            if is_str_builtin(name) {
                return Ok(str_builtin_ret(name));
            }
            if let Some(ret) = tensor_builtin_ret(name) {
                return Ok(ret);
            }
            let sig = env
                .sigs
                .get(name)
                .ok_or_else(|| format!("wasm backend: call to unsupported/unknown fn `{}`", name))?;
            Ok(sig.ret)
        }
        Bind::If(_, then, els) => {
            // The IR's lowering of integer `match` wraps the if-chain in an
            // identity `if true { chain } else { Ret(Unit) }`, and a bare
            // literal-match chain bottoms out in an unreachable `Ret(Unit)`.
            // Those Unit branches are dead; the live branch dictates the type.
            let t = iexpr_type(then, env);
            let e = iexpr_type(els, env);
            match (t, e) {
                (Ok(t), Ok(e)) => {
                    if t != e {
                        return Err(format!(
                            "wasm backend: if-branches have differing types ({:?} vs {:?})",
                            t, e
                        ));
                    }
                    Ok(t)
                }
                // One branch diverges (unreachable Unit, or a `TailCall` loop
                // back-edge that yields no value); the other branch wins.
                (Ok(t), Err(_)) if is_diverging(els) => Ok(t),
                (Err(_), Ok(e)) if is_diverging(then) => Ok(e),
                // Both branches diverge (e.g. each arm is a `TailCall`): use the
                // function return type as the (never-materialized) blocktype.
                (Err(_), Err(_)) if is_diverging(then) && is_diverging(els) => Ok(env.fn_ret),
                (Err(err), _) => Err(err),
                (_, Err(err)) => Err(err),
            }
        }
        Bind::Ctor(name, _) => {
            // Validate the constructor exists; a cell pointer is always a Ref.
            env.ctors.get(name)?;
            Ok(WType::Ref)
        }
        Bind::CtorReuse(_tok, name, _) => {
            // Validate the constructor exists; a (possibly reused) cell is a Ref.
            env.ctors.get(name)?;
            Ok(WType::Ref)
        }
        Bind::Match(scrut, arms) => match_type(scrut, arms, env),
    }
}

/// Infer the wasm result type of a `Bind::Match`: every (live) arm must agree.
/// Arm binders are bound to the matched constructor's field types for the
/// purpose of inferring that arm's body type.
fn match_type(scrut: &Atom, arms: &[ir::IArm], env: &LocalEnv) -> Result<WType, String> {
    // The scrutinee must be a heap reference.
    let st = atom_type(scrut, env)?;
    if st != WType::Ref {
        return Err("wasm backend: `match` scrutinee must be an ADT (Ref)".into());
    }
    let mut result: Option<WType> = None;
    for arm in arms {
        // Build a probe env with the arm's binders bound to field types.
        let mut types = env.types.clone();
        match &arm.ctor {
            Some(cname) => {
                let info = env.ctors.get(cname)?;
                for (b, ft) in arm.binders.iter().zip(info.field_types.iter()) {
                    types.insert(b.clone(), *ft);
                }
            }
            None => {
                // Catch-all binds the scrutinee pointer (a Ref).
                if let Some(b) = arm.binders.first() {
                    types.insert(b.clone(), WType::Ref);
                }
            }
        }
        let probe = LocalEnv {
            types,
            index: env.index.clone(),
            locals: env.locals.clone(),
            n_params: env.n_params,
            sigs: env.sigs,
            ovf_base: env.ovf_base,
            heap_base: env.heap_base,
            ctors: env.ctors,
            scratch_ptr: env.scratch_ptr,
            str_lits: env.str_lits,
            print_str_idx: env.print_str_idx,
            print_float_idx: env.print_float_idx,
            print_int_idx: env.print_int_idx,
            print_bool_idx: env.print_bool_idx,
            exp_idx: env.exp_idx,
            block_depth: env.block_depth,
            tail: env.tail.clone(),
            fn_ret: env.fn_ret,
            closures: env.closures,
        };
        // A diverging arm (dead Unit marker, or one ending in a `TailCall` loop
        // back-edge) yields no value and imposes no type constraint; skip it.
        if is_diverging(&arm.body) {
            continue;
        }
        let at = iexpr_type(&arm.body, &probe)?;
        match result {
            None => result = Some(at),
            Some(prev) if prev != at => {
                return Err(format!(
                    "wasm backend: `match` arms have differing types ({:?} vs {:?})",
                    prev, at
                ));
            }
            _ => {}
        }
    }
    // Every arm diverged (each a `TailCall`): use the function return type as the
    // (never-materialized) blocktype.
    Ok(result.unwrap_or(env.fn_ret))
}

/// True when an IExpr is a bare `Ret(Unit)` — the IR's marker for the dead
/// fall-through branch of a lowered integer `match`. Such a branch produces no
/// real value (it is statically unreachable), so the backend compiles it to the
/// wasm `unreachable` instruction, which validates under any block type.
fn is_unreachable_unit(e: &IExpr) -> bool {
    match e {
        IExpr::Ret(Atom::Unit) => true,
        // The rc pass may wrap the dead branch in `dup`/`drop`s (e.g. dropping a
        // param the live branch consumed). Those are dead too: see through them.
        IExpr::Dup(_, b) | IExpr::Drop(_, b) => is_unreachable_unit(b),
        _ => false,
    }
}

/// True when an IExpr does NOT yield a value at this position: the dead
/// `Ret(Unit)` marker, OR a branch ending in a `TailCall` (a loop back-edge
/// that re-enters the function). Such a branch imposes no type constraint on a
/// sibling `if`/`match` arm. Codegen still distinguishes the two: a TailCall
/// emits the loop back-edge; only the dead marker emits `unreachable`.
fn is_diverging(e: &IExpr) -> bool {
    match e {
        IExpr::Ret(Atom::Unit) => true,
        IExpr::TailCall(_) => true,
        // A tail `if`/`match` (its result is returned immediately) diverges when
        // EVERY branch diverges — e.g. an `if` both of whose arms are TailCalls.
        IExpr::Let(x, Bind::If(_, t, el), cont)
            if matches!(&**cont, IExpr::Ret(Atom::Var(v)) if v == x) =>
        {
            is_diverging(t) && is_diverging(el)
        }
        IExpr::Let(x, Bind::Match(_, arms), cont)
            if matches!(&**cont, IExpr::Ret(Atom::Var(v)) if v == x) =>
        {
            arms.iter().all(|a| is_diverging(&a.body))
        }
        IExpr::Dup(_, b) | IExpr::Drop(_, b) | IExpr::DropReuse(_, _, b) | IExpr::Let(_, _, b) => {
            is_diverging(b)
        }
        _ => false,
    }
}

/// Infer the wasm result type of an `IExpr`. `If`-branches may introduce their
/// own let-bound locals, but those are scoped to the branch; for *type*
/// inference we evaluate against a shallow clone so we don't pollute the parent
/// env. (Code generation, in contrast, threads real local slots through.)
fn iexpr_type(e: &IExpr, env: &LocalEnv) -> Result<WType, String> {
    match e {
        IExpr::Ret(a) => atom_type(a, env),
        IExpr::Let(x, bind, body) => {
            let t = bind_type(bind, env)?;
            // A temporary, lightweight extension for inference only.
            let mut types = env.types.clone();
            types.insert(x.clone(), t);
            let probe = LocalEnv {
                types,
                index: env.index.clone(),
                locals: env.locals.clone(),
                n_params: env.n_params,
                sigs: env.sigs,
                ovf_base: env.ovf_base,
                heap_base: env.heap_base,
                ctors: env.ctors,
                scratch_ptr: env.scratch_ptr,
                str_lits: env.str_lits,
                print_str_idx: env.print_str_idx,
            print_float_idx: env.print_float_idx,
            print_int_idx: env.print_int_idx,
            print_bool_idx: env.print_bool_idx,
            exp_idx: env.exp_idx,
            block_depth: env.block_depth,
            tail: env.tail.clone(),
            fn_ret: env.fn_ret,
            closures: env.closures,
            };
            iexpr_type(body, &probe)
        }
        // dup/drop don't introduce bindings; the type is the body's type.
        IExpr::Dup(_, body) | IExpr::Drop(_, body) => iexpr_type(body, env),
        // DropReuse binds the reuse token (an i32) for the body to reference.
        IExpr::DropReuse(_scrut, tok, body) => {
            let mut types = env.types.clone();
            types.insert(tok.clone(), WType::I32);
            let probe = LocalEnv {
                types,
                index: env.index.clone(),
                locals: env.locals.clone(),
                n_params: env.n_params,
                sigs: env.sigs,
                ovf_base: env.ovf_base,
                heap_base: env.heap_base,
                ctors: env.ctors,
                scratch_ptr: env.scratch_ptr,
                str_lits: env.str_lits,
                print_str_idx: env.print_str_idx,
            print_float_idx: env.print_float_idx,
            print_int_idx: env.print_int_idx,
            print_bool_idx: env.print_bool_idx,
            exp_idx: env.exp_idx,
            block_depth: env.block_depth,
            tail: env.tail.clone(),
            fn_ret: env.fn_ret,
            closures: env.closures,
            };
            iexpr_type(body, &probe)
        }
        // A `TailCall` is a loop back-edge: it yields no value here. Report it as
        // "no type"; sibling-branch inference treats it as diverging (skips it).
        IExpr::TailCall(_) => Err("wasm backend: TailCall has no value type".into()),
    }
}

// ---- code generation -----------------------------------------------------

/// Push the value of an atom onto the wasm operand stack.
fn emit_atom(a: &Atom, env: &LocalEnv, code: &mut Vec<u8>) -> Result<WType, String> {
    match a {
        Atom::Int(n) => {
            code.push(0x42); // i64.const
            leb_s(*n, code);
            Ok(WType::I64)
        }
        Atom::Bool(b) => {
            code.push(0x41); // i32.const
            leb_s(if *b { 1 } else { 0 }, code);
            Ok(WType::I32)
        }
        Atom::Var(n) => {
            code.push(0x20); // local.get
            leb_u(env.var_index(n)? as u64, code);
            env.var_type(n)
        }
        Atom::Float(f) => {
            // f64.const: opcode 0x44 followed by the 8 raw little-endian
            // IEEE-754 bytes (a fixed 8-byte immediate, NOT LEB-encoded).
            code.push(0x44); // f64.const
            code.extend_from_slice(&f.to_le_bytes());
            Ok(WType::F64)
        }
        Atom::Str(s) => {
            // Materialize the literal at runtime: push (data_addr, len) and call
            // `__str_lit`, which allocs a String object and copies the raw UTF-8
            // bytes from the read-only data region. Keeps `emit_atom` free of any
            // local allocation (the copy loop lives inside the helper).
            let bytes = s.as_bytes();
            let data_addr = *env.str_lits.get(bytes).ok_or_else(|| {
                "wasm backend: string literal missing from pool (internal)".to_string()
            })?;
            code.push(0x41); // i32.const data_addr
            leb_s(data_addr as i64, code);
            code.push(0x41); // i32.const len
            leb_s(bytes.len() as i64, code);
            code.push(0x10); // call __str_lit
            leb_u(env.heap_index(HeapHelper::StrLit) as u64, code);
            Ok(WType::Str)
        }
        Atom::Unit => Err("wasm backend: Unit is outside the 2a subset".into()),
    }
}

/// Emit a `Bind`, leaving its single result value on the operand stack.
fn emit_bind(bind: &Bind, env: &mut LocalEnv, code: &mut Vec<u8>) -> Result<WType, String> {
    match bind {
        Bind::Atom(a) => emit_atom(a, env, code),
        Bind::MakeClosure(lam, caps) => emit_make_closure(lam, caps, env, code),
        Bind::ApplyClosure(callee, args, ret) => {
            emit_apply_closure(callee, args, ret.as_ref(), env, code)
        }
        Bind::Prim(op, l, r) => {
            // String `==`/`!=`: structural (len + byte) compare via `__streq`.
            // Operand ownership mirrors the rc pass: a Prim does NOT consume its
            // variable operands (they are borrowed and dropped at their real last
            // use), but a String *literal* operand allocates a temporary that the
            // rc pass never sees, so we drop it here to stay garbage-free.
            if matches!(op, BinOp::Eq | BinOp::Ne) && atom_type(l, env)? == WType::Str {
                if atom_type(r, env)? != WType::Str {
                    return Err("wasm backend: == / != on mismatched types".into());
                }
                return emit_str_eq(*op, l, r, env, code);
            }
            // ADT `==`/`!=`: recursive structural compare via `__eq`. Like the
            // String case, the comparison OWNS its operands (the rc pass marks
            // Eq/Ne operands consumed) and must drop both ADT pointers after.
            if matches!(op, BinOp::Eq | BinOp::Ne) && atom_type(l, env)? == WType::Ref {
                if atom_type(r, env)? != WType::Ref {
                    return Err("wasm backend: == / != on mismatched types".into());
                }
                return emit_adt_eq(*op, l, r, env, code);
            }
            let lt = emit_atom(l, env, code)?;
            let rt = emit_atom(r, env, code)?;
            // Add/Sub/Mul on Int are checked: route through an emitted helper
            // that traps (`unreachable`) on signed-i64 overflow, matching the
            // interpreter. Everything else (Div/Mod/comparisons/logical) stays
            // inline. Div/Mod already trap natively on /0 and MIN/-1.
            match op {
                // Int Add/Sub/Mul are checked (overflow traps). Float
                // Add/Sub/Mul are plain f64 ops (no overflow concept).
                BinOp::Add | BinOp::Sub | BinOp::Mul if lt == WType::I64 || rt == WType::I64 => {
                    if lt != WType::I64 || rt != WType::I64 {
                        return Err("wasm backend: arithmetic expects Int operands".into());
                    }
                    let helper = match op {
                        BinOp::Add => OvfHelper::Add,
                        BinOp::Sub => OvfHelper::Sub,
                        BinOp::Mul => OvfHelper::Mul,
                        _ => unreachable!(),
                    };
                    code.push(0x10); // call
                    leb_u(env.ovf_index(helper) as u64, code);
                    Ok(WType::I64)
                }
                _ => emit_prim(*op, lt, rt, code),
            }
        }
        Bind::Unary(op, a) => match op {
            UnOp::Neg => {
                let t = atom_type(a, env)?;
                match t {
                    WType::I64 => {
                        // i64 negation is checked: traps on `-i64::MIN`, which
                        // overflows, matching the interpreter.
                        emit_atom(a, env, code)?;
                        code.push(0x10); // call __neg_ovf
                        leb_u(env.ovf_index(OvfHelper::Neg) as u64, code);
                        Ok(WType::I64)
                    }
                    WType::F64 => {
                        // f64.neg: no overflow concept for floats.
                        emit_atom(a, env, code)?;
                        code.push(0x9A); // f64.neg
                        Ok(WType::F64)
                    }
                    WType::I32 | WType::Ref | WType::Str | WType::Tensor | WType::F32 => {
                        Err("wasm backend: numeric negation requires an Int or Float".into())
                    }
                }
            }
            UnOp::Not => {
                let t = emit_atom(a, env, code)?;
                if t != WType::I32 {
                    return Err("wasm backend: logical `not` expects a Bool".into());
                }
                code.push(0x45); // i32.eqz
                Ok(WType::I32)
            }
        },
        Bind::Call(name, args) => {
            // String builtins are emitted inline (they consume their String
            // arguments, matching the rc pass's "Call args are consumed" rule).
            if is_str_builtin(name) {
                return emit_str_builtin(name, args, env, code);
            }
            // Tensor / embedding builtins are emitted as calls into the heap
            // helpers (they consume their heap arguments, matching the rc pass).
            if tensor_builtin_ret(name).is_some() {
                return emit_tensor_builtin(name, args, env, code);
            }
            let sig_ret;
            let sig_params;
            {
                let sig = env.sigs.get(name).ok_or_else(|| {
                    format!("wasm backend: call to unsupported/unknown fn `{}`", name)
                })?;
                sig_ret = sig.ret;
                sig_params = sig.params.clone();
            }
            if sig_params.len() != args.len() {
                return Err(format!(
                    "wasm backend: call to `{}` has {} args, expected {}",
                    name,
                    args.len(),
                    sig_params.len()
                ));
            }
            for a in args {
                emit_atom(a, env, code)?;
            }
            code.push(0x10); // call
            let idx = *env
                .index
                .get(&fn_index_key(name))
                .ok_or_else(|| format!("wasm backend: no index for fn `{}`", name))?;
            leb_u(idx as u64, code);
            Ok(sig_ret)
        }
        Bind::If(c, then, els) => {
            // Push the condition (must be i32 / Bool).
            let ct = emit_atom(c, env, code)?;
            if ct != WType::I32 {
                return Err("wasm backend: `if` condition must be a Bool".into());
            }
            let result_ty = bind_type(bind, env)?;
            code.push(0x04); // if
            code.push(result_ty.byte()); // blocktype = result valtype
            // Each branch opens a new control level: bump the block depth so a
            // nested `TailCall` computes its `br` target relative to the loop.
            env.block_depth += 1;
            emit_if_branch(then, result_ty, env, code)?;
            code.push(0x05); // else
            emit_if_branch(els, result_ty, env, code)?;
            code.push(0x0B); // end
            env.block_depth -= 1;
            Ok(result_ty)
        }
        Bind::Ctor(name, fields) => emit_ctor(name, fields, env, code),
        Bind::CtorReuse(tok, name, fields) => emit_ctor_reuse(tok, name, fields, env, code),
        Bind::Match(scrut, arms) => emit_match(scrut, arms, env, code),
    }
}

/// Validate a constructor's name/arity, returning its (cloned) `CtorInfo`.
fn ctor_info_checked(
    name: &str,
    fields: &[Atom],
    env: &LocalEnv,
) -> Result<CtorInfo, String> {
    let info = env.ctors.get(name)?.clone();
    if fields.len() != info.arity() {
        return Err(format!(
            "wasm backend: ctor `{}` got {} fields, expected {}",
            name,
            fields.len(),
            info.arity()
        ));
    }
    Ok(info)
}

/// Write a constructor cell's tag + fields into the cell whose pointer is held
/// in the `scratch` local. The cell's slot must already exist (freshly
/// allocated, or a reused slot); rc is set by the caller (`__alloc` for a fresh
/// cell, or explicitly for a reused one). Used by both `Ctor` and `CtorReuse`.
fn emit_store_cell_fields(
    name: &str,
    info: &CtorInfo,
    fields: &[Atom],
    scratch: u32,
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<(), String> {
    // store tag at [ptr+8]
    code.push(0x20); // local.get scratch
    leb_u(scratch as u64, code);
    code.push(0x42); // i64.const tag
    leb_s(info.tag, code);
    i64_store(8, code);
    // store each field at [ptr + 16 + 8*i]
    for (i, (a, fty)) in fields.iter().zip(info.field_types.iter()).enumerate() {
        code.push(0x20); // local.get scratch (address)
        leb_u(scratch as u64, code);
        let t = emit_atom(a, env, code)?;
        if t != *fty {
            return Err(format!(
                "wasm backend: ctor `{}` field {} type mismatch (got {:?}, expected {:?})",
                name, i, t, fty
            ));
        }
        let off = CELL_HEADER + SLOT * i as u64;
        // Store into the 8-byte slot per field kind. A Float field is an
        // unboxed f64 (f64.store); Int/Bool/Ref/Str go through the i64 slot
        // (Bool/Ref/Str zero-extended from i32).
        match fty {
            WType::F64 => f64_store(off, code),
            WType::I64 => i64_store(off, code),
            WType::I32 | WType::Ref | WType::Str | WType::Tensor => {
                code.push(0xAD); // i64.extend_i32_u
                i64_store(off, code);
            }
            // F32 is an internal helper-local type, never an ADT field.
            WType::F32 => unreachable!("F32 is not an Aria field type"),
        }
    }
    Ok(())
}

/// The builtins the wasm backend implements inline. `concat`/`int_to_str`
/// produce/consume Strings; `print_str`/`print_float`/`print_int`/`print_bool`
/// are host-imported effectful prints. (`print_float` takes an unboxed f64,
/// `print_int` an i64, `print_bool` an i32.)
fn is_str_builtin(name: &str) -> bool {
    matches!(
        name,
        "concat" | "int_to_str" | "print_str" | "print_float" | "print_int" | "print_bool"
    )
}

/// The wasm result type of an inline builtin. `print_str`/`print_float` are
/// logically Unit; we give them a dummy i32 result (never used).
fn str_builtin_ret(name: &str) -> WType {
    match name {
        "concat" | "int_to_str" => WType::Str,
        _ => WType::I32, // print_str -> dummy i32
    }
}

/// Emit a String builtin call. Per the rc pass's "Call arguments are consumed"
/// rule, each builtin CONSUMES (drops) its String arguments:
///   * `concat(a, b)` -> `__concat(a, b)` (the helper copies both runs, then
///     drops `a` and `b`); result is a fresh String (rc=1).
///   * `int_to_str(n)` -> `__int_to_str(n)` (n is unboxed Int, nothing to drop).
///   * `print_str(s)` -> imported `env.print_str(ptr+16, len)`, then `__drop_str(s)`;
///     yields a dummy i32 0.
fn emit_str_builtin(
    name: &str,
    args: &[Atom],
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    match name {
        "concat" => {
            if args.len() != 2 {
                return Err("wasm backend: concat expects 2 arguments".into());
            }
            if atom_type(&args[0], env)? != WType::Str || atom_type(&args[1], env)? != WType::Str {
                return Err("wasm backend: concat expects two Strings".into());
            }
            emit_atom(&args[0], env, code)?;
            emit_atom(&args[1], env, code)?;
            code.push(0x10); // call __concat (consumes both args internally)
            leb_u(env.heap_index(HeapHelper::Concat) as u64, code);
            Ok(WType::Str)
        }
        "int_to_str" => {
            if args.len() != 1 || atom_type(&args[0], env)? != WType::I64 {
                return Err("wasm backend: int_to_str expects one Int".into());
            }
            emit_atom(&args[0], env, code)?;
            code.push(0x10); // call __int_to_str
            leb_u(env.heap_index(HeapHelper::IntToStr) as u64, code);
            Ok(WType::Str)
        }
        "print_str" => {
            if args.len() != 1 || atom_type(&args[0], env)? != WType::Str {
                return Err("wasm backend: print_str expects one String".into());
            }
            // Evaluate the String once into a temp, call the host import with
            // (ptr+16, len), then drop the String (this call consumes it).
            let s = env.fresh_i32();
            let print_idx = env.print_str_idx;
            let drop_str = env.heap_index(HeapHelper::DropStr);
            emit_atom(&args[0], env, code)?;
            code.push(0x21); // local.set s
            leb_u(s as u64, code);
            // env.print_str(ptr + 16, len)
            code.push(0x20); // local.get s
            leb_u(s as u64, code);
            code.push(0x41); // i32.const STR_HEADER
            leb_s(STR_HEADER as i64, code);
            code.push(0x6A); // i32.add  -> bytes pointer
            code.push(0x20); // local.get s
            leb_u(s as u64, code);
            i64_load(8, code); // len (i64)
            code.push(0xA7); // i32.wrap_i64 -> len:i32
            code.push(0x10); // call env.print_str
            leb_u(print_idx as u64, code);
            // drop the String
            code.push(0x20); // local.get s
            leb_u(s as u64, code);
            code.push(0x10); // call __drop_str
            leb_u(drop_str as u64, code);
            code.push(0x1A); // drop dummy result
            // dummy i32 result for the let-binding
            code.push(0x41); // i32.const 0
            leb_s(0, code);
            Ok(WType::I32)
        }
        "print_float" => {
            if args.len() != 1 || atom_type(&args[0], env)? != WType::F64 {
                return Err("wasm backend: print_float expects one Float".into());
            }
            // Push the f64 and call the host import `env.print_float(f64)`.
            let print_idx = env.print_float_idx;
            emit_atom(&args[0], env, code)?;
            code.push(0x10); // call env.print_float
            leb_u(print_idx as u64, code);
            // dummy i32 result for the let-binding (print_float is Unit-like)
            code.push(0x41); // i32.const 0
            leb_s(0, code);
            Ok(WType::I32)
        }
        "print_int" => {
            if args.len() != 1 || atom_type(&args[0], env)? != WType::I64 {
                return Err("wasm backend: print_int expects one Int".into());
            }
            // Push the i64 and call the host import `env.print_int(i64)`.
            let print_idx = env.print_int_idx;
            emit_atom(&args[0], env, code)?;
            code.push(0x10); // call env.print_int
            leb_u(print_idx as u64, code);
            // dummy i32 result for the let-binding (print_int is Unit-like)
            code.push(0x41); // i32.const 0
            leb_s(0, code);
            Ok(WType::I32)
        }
        "print_bool" => {
            if args.len() != 1 || atom_type(&args[0], env)? != WType::I32 {
                return Err("wasm backend: print_bool expects one Bool".into());
            }
            // Push the i32 (0/1) and call the host import `env.print_bool(i32)`.
            let print_idx = env.print_bool_idx;
            emit_atom(&args[0], env, code)?;
            code.push(0x10); // call env.print_bool
            leb_u(print_idx as u64, code);
            // dummy i32 result for the let-binding (print_bool is Unit-like)
            code.push(0x41); // i32.const 0
            leb_s(0, code);
            Ok(WType::I32)
        }
        _ => Err(format!("wasm backend: unknown string builtin `{}`", name)),
    }
}

/// The Tensor / embedding builtins the wasm backend implements (Phase 2f).
/// Returns the wasm result type of `name`, or `None` if it is not one of them.
/// The codec builtins (`compressed_size`, `neural_bits_per_byte`) are
/// intentionally NOT here — they remain a clean Err (deferred).
fn tensor_builtin_ret(name: &str) -> Option<WType> {
    Some(match name {
        "tensor_zeros" | "tensor_set" | "matmul" | "transpose" | "softmax" | "relu" => {
            WType::Tensor
        }
        "tensor_get" | "embed_similarity" => WType::F64,
        "tensor_rows" | "tensor_cols" => WType::I64,
        _ => return None,
    })
}

/// Emit a Tensor / embedding builtin call. Each maps to one heap helper that
/// CONSUMES its heap arguments (Tensor/String pointers), matching the rc pass's
/// "Call arguments are consumed" rule, so no separate drop is emitted here.
fn emit_tensor_builtin(
    name: &str,
    args: &[Atom],
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    // Helper: push every argument atom, checking its expected wasm type.
    let push_args = |env: &mut LocalEnv, code: &mut Vec<u8>, expect: &[WType]| -> Result<(), String> {
        if args.len() != expect.len() {
            return Err(format!(
                "wasm backend: `{}` expects {} args, got {}",
                name,
                expect.len(),
                args.len()
            ));
        }
        for (a, want) in args.iter().zip(expect.iter()) {
            let got = emit_atom(a, env, code)?;
            if got != *want {
                return Err(format!(
                    "wasm backend: `{}` arg type mismatch (got {:?}, expected {:?})",
                    name, got, want
                ));
            }
        }
        Ok(())
    };
    let (expect, helper, ret): (Vec<WType>, HeapHelper, WType) = match name {
        "tensor_zeros" => (
            vec![WType::I64, WType::I64],
            HeapHelper::TensorZeros,
            WType::Tensor,
        ),
        "tensor_set" => (
            vec![WType::Tensor, WType::I64, WType::I64, WType::F64],
            HeapHelper::TensorSet,
            WType::Tensor,
        ),
        "tensor_get" => (
            vec![WType::Tensor, WType::I64, WType::I64],
            HeapHelper::TensorGet,
            WType::F64,
        ),
        "tensor_rows" => (vec![WType::Tensor], HeapHelper::TensorRows, WType::I64),
        "tensor_cols" => (vec![WType::Tensor], HeapHelper::TensorCols, WType::I64),
        "matmul" => (
            vec![WType::Tensor, WType::Tensor],
            HeapHelper::Matmul,
            WType::Tensor,
        ),
        "transpose" => (vec![WType::Tensor], HeapHelper::Transpose, WType::Tensor),
        "relu" => (vec![WType::Tensor], HeapHelper::Relu, WType::Tensor),
        "softmax" => (vec![WType::Tensor], HeapHelper::Softmax, WType::Tensor),
        "embed_similarity" => (
            vec![WType::Str, WType::Str],
            HeapHelper::EmbedSim,
            WType::F64,
        ),
        _ => return Err(format!("wasm backend: unknown tensor builtin `{}`", name)),
    };
    push_args(env, code, &expect)?;
    code.push(0x10); // call helper
    leb_u(env.heap_index(helper) as u64, code);
    Ok(ret)
}

/// Emit a String `==` / `!=`. Both operands are evaluated into fresh i32 temps
/// (so a literal operand's pointer is captured), `__streq` compares them, the
/// result is negated for `!=`, and BOTH operands are then dropped via
/// `__drop_str`. The comparison CONSUMES its String operands: the rc pass marks
/// `Eq`/`Ne` operands as consumed (so it dups a variable that is used again and
/// never separately drops it here), and a literal operand has its own rc=1
/// temp; either way the comparison owns one reference and releases it, keeping
/// the heap garbage-free.
fn emit_str_eq(
    op: BinOp,
    l: &Atom,
    r: &Atom,
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    let lt = env.fresh_i32();
    let rt = env.fresh_i32();
    let streq = env.heap_index(HeapHelper::StrEq);
    let drop_str = env.heap_index(HeapHelper::DropStr);
    // lt = l ; rt = r
    emit_atom(l, env, code)?;
    code.push(0x21); // local.set lt
    leb_u(lt as u64, code);
    emit_atom(r, env, code)?;
    code.push(0x21); // local.set rt
    leb_u(rt as u64, code);
    // result = __streq(lt, rt)
    code.push(0x20);
    leb_u(lt as u64, code);
    code.push(0x20);
    leb_u(rt as u64, code);
    code.push(0x10); // call __streq
    leb_u(streq as u64, code);
    if op == BinOp::Ne {
        code.push(0x45); // i32.eqz (negate)
    }
    // The comparison result is now on the stack. The comparison consumes BOTH
    // operands (the rc pass dup'd any variable still needed later and inserts no
    // drop for these operands), so release each here — after `__streq` read the
    // bytes. A literal temp and a moved-in variable both held one owned ref.
    let drop_one = |_a: &Atom, slot: u32, code: &mut Vec<u8>| {
        code.push(0x20); // local.get slot
        leb_u(slot as u64, code);
        code.push(0x10); // call __drop_str
        leb_u(drop_str as u64, code);
        code.push(0x1A); // drop dummy result
    };
    drop_one(l, lt, code);
    drop_one(r, rt, code);
    Ok(WType::I32)
}

/// Emit an ADT `==` / `!=`. Both operands are evaluated into fresh i32 temps,
/// `__eq` compares them structurally (recursively), the result is negated for
/// `!=`, and BOTH operand pointers are then dropped via `__drop`. `__eq` only
/// READS the heap, so dropping afterwards is safe. The comparison CONSUMES its
/// ADT operands: the rc pass marks `Eq`/`Ne` operands as consumed (dup'ing a
/// value that is reused later, and inserting no separate drop for these
/// operands), so the comparison owns exactly one reference to each operand and
/// releases it here — keeping the heap garbage-free, and the dup'd reuse path
/// from double-freeing.
fn emit_adt_eq(
    op: BinOp,
    l: &Atom,
    r: &Atom,
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    let lt = env.fresh_i32();
    let rt = env.fresh_i32();
    let eq = env.heap_index(HeapHelper::Eq);
    let drop_ref = env.heap_index(HeapHelper::Drop);
    // lt = l ; rt = r
    emit_atom(l, env, code)?;
    code.push(0x21); // local.set lt
    leb_u(lt as u64, code);
    emit_atom(r, env, code)?;
    code.push(0x21); // local.set rt
    leb_u(rt as u64, code);
    // result = __eq(lt, rt)
    code.push(0x20);
    leb_u(lt as u64, code);
    code.push(0x20);
    leb_u(rt as u64, code);
    code.push(0x10); // call __eq
    leb_u(eq as u64, code);
    if op == BinOp::Ne {
        code.push(0x45); // i32.eqz (negate)
    }
    // Release each operand now that `__eq` has finished reading the heap.
    let drop_one = |slot: u32, code: &mut Vec<u8>| {
        code.push(0x20); // local.get slot
        leb_u(slot as u64, code);
        code.push(0x10); // call __drop
        leb_u(drop_ref as u64, code);
        code.push(0x1A); // drop dummy result
    };
    drop_one(lt, code);
    drop_one(rt, code);
    Ok(WType::I32)
}

/// Emit a `Bind::Ctor`: allocate a cell, store its tag and fields, yield the
/// pointer (an i32 Ref) on the stack.
fn emit_ctor(
    name: &str,
    fields: &[Atom],
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    let info = ctor_info_checked(name, fields, env)?;
    let scratch = env.scratch();
    let alloc = env.heap_index(HeapHelper::Alloc);
    // ptr = __alloc(nfields)  (sets rc=1, live++)
    code.push(0x41); // i32.const nfields
    leb_s(info.arity() as i64, code);
    code.push(0x10); // call __alloc
    leb_u(alloc as u64, code);
    code.push(0x21); // local.set scratch  (ptr)
    leb_u(scratch as u64, code);
    emit_store_cell_fields(name, &info, fields, scratch, env, code)?;
    // result = ptr
    code.push(0x20); // local.get scratch
    leb_u(scratch as u64, code);
    Ok(WType::Ref)
}

/// Emit a `Bind::MakeClosure(lam, caps)`: allocate a closure cell — a heap cell
/// whose tag word is the lifted lambda's closure tag and whose fields are the
/// captured values — and yield the pointer (an i32 Ref). Mirrors `emit_ctor`,
/// except the tag identifies a lambda (for `call_indirect`) rather than an ADT
/// constructor, and the field types come from the captured atoms themselves.
fn emit_make_closure(
    lam: &str,
    caps: &[Atom],
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    let tag = *env
        .closures
        .tags
        .get(lam)
        .ok_or_else(|| format!("wasm backend: unknown lambda `{}`", lam))?;
    let scratch = env.scratch();
    let alloc = env.heap_index(HeapHelper::Alloc);
    // ptr = __alloc(ncaps)  (sets rc=1, live++)
    code.push(0x41); // i32.const ncaps
    leb_s(caps.len() as i64, code);
    code.push(0x10); // call __alloc
    leb_u(alloc as u64, code);
    code.push(0x21); // local.set scratch (ptr)
    leb_u(scratch as u64, code);
    // store tag at [ptr+8]
    code.push(0x20); // local.get scratch
    leb_u(scratch as u64, code);
    code.push(0x42); // i64.const tag
    leb_s(tag, code);
    i64_store(8, code);
    // store each capture at [ptr + 16 + 8*i]
    for (i, a) in caps.iter().enumerate() {
        code.push(0x20); // local.get scratch (address)
        leb_u(scratch as u64, code);
        let t = emit_atom(a, env, code)?;
        let off = CELL_HEADER + SLOT * i as u64;
        match t {
            WType::F64 => f64_store(off, code),
            WType::I64 => i64_store(off, code),
            WType::I32 | WType::Ref | WType::Str | WType::Tensor => {
                code.push(0xAD); // i64.extend_i32_u
                i64_store(off, code);
            }
            WType::F32 => unreachable!("F32 is not an Aria capture type"),
        }
    }
    // result = ptr
    code.push(0x20); // local.get scratch
    leb_u(scratch as u64, code);
    Ok(WType::Ref)
}

/// Emit a `Bind::ApplyClosure(callee, args, ret)`: dispatch a closure value
/// through the function table with `call_indirect`. The lifted lambda's wasm
/// signature is `(i32 closure, args...) -> ret`; we push the closure pointer,
/// the arguments, and finally the table index recovered from the closure cell's
/// tag (`tag - closure_base`), then `call_indirect`. This application owns one
/// reference to the closure (the rc pass dup'd it for any further use), so we
/// `__drop` it afterwards — the lambda body borrowed the captures (dup'ing each),
/// so freeing the cell here only releases this application's hold.
fn emit_apply_closure(
    callee: &Atom,
    args: &[Atom],
    ret: Option<&Ty>,
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    let ret_ty = ret.ok_or("wasm backend: closure application missing its result type")?;
    let ret_wty = WType::from_ty(ret_ty)?;
    // Evaluate the closure pointer once into a fresh local (used as arg 0, for the
    // table index, and for the trailing drop).
    let clo = env.fresh_i32();
    let ct = emit_atom(callee, env, code)?;
    if ct != WType::Ref {
        return Err("wasm backend: applying a non-closure value".into());
    }
    code.push(0x21); // local.set clo
    leb_u(clo as u64, code);
    // arg 0 = closure pointer
    code.push(0x20); // local.get clo
    leb_u(clo as u64, code);
    // remaining args, recording their wasm types for the call_indirect signature
    let mut sig_params = vec![WType::I32]; // closure ptr
    for a in args {
        let t = emit_atom(a, env, code)?;
        sig_params.push(t);
    }
    // table index = (tag - closure_base) as i32
    code.push(0x20); // local.get clo
    leb_u(clo as u64, code);
    i64_load(8, code); // tag
    code.push(0x42); // i64.const base
    leb_s(env.closures.base, code);
    code.push(0x7D); // i64.sub
    code.push(0xA7); // i32.wrap_i64
    // call_indirect typeidx tableidx
    let key = (sig_params, ret_wty);
    let typeidx = *env.closures.sig_typeidx.get(&key).ok_or_else(|| {
        "wasm backend: no function-table type for this closure signature (internal)".to_string()
    })?;
    code.push(0x11); // call_indirect
    leb_u(typeidx as u64, code);
    code.push(0x00); // table index 0
    // drop our reference to the closure cell.
    let drop_idx = env.heap_index(HeapHelper::Drop);
    code.push(0x20); // local.get clo
    leb_u(clo as u64, code);
    code.push(0x10); // call __drop
    leb_u(drop_idx as u64, code);
    code.push(0x1A); // drop dummy result
    Ok(ret_wty)
}

/// Emit a `Bind::CtorReuse(tok, name, fields)`: if the reuse token `tok` (an i32
/// local) is non-zero it is a retained cell slot of the right arity — overwrite
/// it in place (set rc=1, write tag + fields), bump the `__reuses` counter, and
/// do NOT call `__alloc` or touch `__live` (the slot was kept, not freed). If
/// `tok == 0` (cell was shared, or scrutinee wasn't a Ref), allocate fresh —
/// exactly the empty-token fallback of the IR interpreter. Mirrors the IR
/// interpreter's `Bind::CtorReuse` handler.
fn emit_ctor_reuse(
    tok: &str,
    name: &str,
    fields: &[Atom],
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    let info = ctor_info_checked(name, fields, env)?;
    let tok_idx = env.var_index(tok)?;
    if env.var_type(tok)? != WType::I32 {
        return Err("wasm backend: reuse token must be an i32".into());
    }
    let scratch = env.scratch();
    let alloc = env.heap_index(HeapHelper::Alloc);
    // if (result i32)  tok != 0  { reuse in place }  else  { alloc fresh }
    code.push(0x20); // local.get tok
    leb_u(tok_idx as u64, code);
    code.push(0x45); // i32.eqz
    code.push(0x45); // i32.eqz  (tok != 0)
    code.push(0x04); // if
    code.push(WType::Ref.byte()); // blocktype = i32 (the resulting Ref)
    // --- reuse branch: scratch = tok; rc = 1; __reuses++ ---
    code.push(0x20); // local.get tok
    leb_u(tok_idx as u64, code);
    code.push(0x21); // local.set scratch
    leb_u(scratch as u64, code);
    // rc = 1 at [scratch]
    code.push(0x20); // local.get scratch
    leb_u(scratch as u64, code);
    code.push(0x42); // i64.const 1
    leb_s(1, code);
    i64_store(0, code);
    // __reuses++  (LIVE accounting unchanged: reuse is net 0)
    code.push(0x41); // i32.const REUSES_ADDR
    leb_s(REUSES_ADDR as i64, code);
    code.push(0x41); // i32.const REUSES_ADDR
    leb_s(REUSES_ADDR as i64, code);
    i64_load(0, code);
    code.push(0x42); // i64.const 1
    leb_s(1, code);
    code.push(0x7C); // i64.add
    i64_store(0, code);
    emit_store_cell_fields(name, &info, fields, scratch, env, code)?;
    code.push(0x20); // local.get scratch  (result)
    leb_u(scratch as u64, code);
    code.push(0x05); // else
    // --- fresh branch: scratch = __alloc(nfields) (sets rc=1, live++) ---
    code.push(0x41); // i32.const nfields
    leb_s(info.arity() as i64, code);
    code.push(0x10); // call __alloc
    leb_u(alloc as u64, code);
    code.push(0x21); // local.set scratch
    leb_u(scratch as u64, code);
    emit_store_cell_fields(name, &info, fields, scratch, env, code)?;
    code.push(0x20); // local.get scratch  (result)
    leb_u(scratch as u64, code);
    code.push(0x0B); // end
    Ok(WType::Ref)
}

/// Emit a `Bind::Match`: load the scrutinee's tag and dispatch via an if/else
/// chain, binding each arm's field variables by loading the cell slots. Mirrors
/// the IR interpreter's first-match-wins arm selection.
fn emit_match(
    scrut: &Atom,
    arms: &[ir::IArm],
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    let st = atom_type(scrut, env)?;
    if st != WType::Ref {
        return Err("wasm backend: `match` scrutinee must be an ADT (Ref)".into());
    }
    let result_ty = match_type(scrut, arms, env)?;
    emit_match_chain(scrut, arms, 0, result_ty, env, code)?;
    Ok(result_ty)
}

/// Recursively emit the if/else chain for match arms from index `i` onward.
/// A constructor arm compares the scrutinee tag; a catch-all (ctor None) is the
/// final unconditional branch.
fn emit_match_chain(
    scrut: &Atom,
    arms: &[ir::IArm],
    i: usize,
    result_ty: WType,
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<(), String> {
    if i >= arms.len() {
        // Exhaustive matches never reach here; emit a trap to satisfy validation.
        code.push(0x00); // unreachable
        return Ok(());
    }
    let arm = &arms[i];
    match &arm.ctor {
        None => {
            // Catch-all: bind the scrutinee pointer (if a binder is present).
            emit_arm_body(scrut, arm, None, result_ty, env, code)
        }
        Some(cname) => {
            let info = env.ctors.get(cname)?.clone();
            // Load tag from [scrut+8] and compare to this ctor's tag.
            emit_atom(scrut, env, code)?; // i32 address
            i64_load(8, code); // tag (i64)
            code.push(0x42); // i64.const tag
            leb_s(info.tag, code);
            code.push(0x51); // i64.eq
            code.push(0x04); // if
            code.push(result_ty.byte());
            // The arm body and the fall-through chain run one control level
            // deeper; bump the block depth so a nested `TailCall` `br`s correctly.
            env.block_depth += 1;
            emit_arm_body(scrut, arm, Some(&info), result_ty, env, code)?;
            code.push(0x05); // else
            emit_match_chain(scrut, arms, i + 1, result_ty, env, code)?;
            code.push(0x0B); // end
            env.block_depth -= 1;
            Ok(())
        }
    }
}

/// Emit one match arm's body. For a constructor arm, bind each used field by
/// loading `[scrut+16+8*i]` into a fresh local (converted to the field's stack
/// type). For a catch-all, bind the scrutinee pointer.
fn emit_arm_body(
    scrut: &Atom,
    arm: &ir::IArm,
    info: Option<&CtorInfo>,
    result_ty: WType,
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<(), String> {
    match info {
        Some(info) => {
            for (idx, b) in arm.binders.iter().enumerate() {
                let fty = info.field_types[idx];
                // Allocate a local for the binder and load the field into it.
                env.add_local(b, fty);
                let slot = env.var_index(b)?;
                let off = CELL_HEADER + SLOT * idx as u64;
                emit_atom(scrut, env, code)?; // address
                // Load the 8-byte slot per field kind, converting to the field's
                // stack type. A Float field is an f64 (f64.load); Int stays i64;
                // Bool/Ref/Str are wrapped back down to i32.
                match fty {
                    WType::F64 => f64_load(off, code),
                    WType::I64 => i64_load(off, code),
                    WType::I32 | WType::Ref | WType::Str | WType::Tensor => {
                        i64_load(off, code);
                        code.push(0xA7); // i32.wrap_i64
                    }
                    WType::F32 => unreachable!("F32 is not an Aria field type"),
                }
                code.push(0x21); // local.set slot
                leb_u(slot as u64, code);
            }
        }
        None => {
            if let Some(b) = arm.binders.first() {
                env.add_local(b, WType::Ref);
                let slot = env.var_index(b)?;
                emit_atom(scrut, env, code)?; // the pointer itself
                code.push(0x21); // local.set
                leb_u(slot as u64, code);
            }
        }
    }
    let bt = emit_iexpr(&arm.body, env, code)?;
    if bt != result_ty {
        return Err(format!(
            "wasm backend: match arm body type {:?} != expected {:?}",
            bt, result_ty
        ));
    }
    Ok(())
}

/// Emit an arithmetic/comparison/logical primitive given operand types.
fn emit_prim(op: BinOp, lt: WType, rt: WType, code: &mut Vec<u8>) -> Result<WType, String> {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
            if lt == WType::F64 || rt == WType::F64 {
                if lt != WType::F64 || rt != WType::F64 {
                    return Err("wasm backend: arithmetic expects matching operands".into());
                }
                // Float arithmetic: plain f64 ops. Div by 0.0 yields inf/NaN
                // (no trap), matching the interpreter; there is no Float `Mod`.
                code.push(match op {
                    BinOp::Add => 0xA0, // f64.add
                    BinOp::Sub => 0xA1, // f64.sub
                    BinOp::Mul => 0xA2, // f64.mul
                    BinOp::Div => 0xA3, // f64.div
                    BinOp::Mod => return Err("wasm backend: Float has no `%`".into()),
                    _ => unreachable!(),
                });
                return Ok(WType::F64);
            }
            if lt != WType::I64 || rt != WType::I64 {
                return Err("wasm backend: arithmetic expects Int operands".into());
            }
            code.push(match op {
                BinOp::Add => 0x7C,
                BinOp::Sub => 0x7D,
                BinOp::Mul => 0x7E,
                BinOp::Div => 0x7F, // i64.div_s
                BinOp::Mod => 0x81, // i64.rem_s
                _ => unreachable!(),
            });
            Ok(WType::I64)
        }
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
            if lt == WType::F64 || rt == WType::F64 {
                if lt != WType::F64 || rt != WType::F64 {
                    return Err("wasm backend: ordering comparisons expect matching operands".into());
                }
                // f64 comparisons yield i32 (Bool); they already give 0 for any
                // NaN operand, matching the interpreter's `NaN`-aware ordering.
                code.push(match op {
                    BinOp::Lt => 0x63, // f64.lt
                    BinOp::Gt => 0x64, // f64.gt
                    BinOp::Le => 0x65, // f64.le
                    BinOp::Ge => 0x66, // f64.ge
                    _ => unreachable!(),
                });
                return Ok(WType::I32);
            }
            if lt != WType::I64 || rt != WType::I64 {
                return Err("wasm backend: ordering comparisons expect Int operands".into());
            }
            code.push(match op {
                BinOp::Lt => 0x53, // i64.lt_s
                BinOp::Gt => 0x55, // i64.gt_s
                BinOp::Le => 0x57, // i64.le_s
                BinOp::Ge => 0x59, // i64.ge_s
                _ => unreachable!(),
            });
            Ok(WType::I32)
        }
        BinOp::Eq | BinOp::Ne => {
            if lt != rt {
                return Err("wasm backend: == / != on mismatched types".into());
            }
            match lt {
                WType::I64 => code.push(if op == BinOp::Eq { 0x51 } else { 0x52 }),
                WType::I32 => code.push(if op == BinOp::Eq { 0x46 } else { 0x47 }),
                // f64.eq (0x61) / f64.ne (0x62); f64.eq gives 0 for NaN==NaN,
                // matching the interpreter (`x == y` so `NaN != NaN`).
                WType::F64 => code.push(if op == BinOp::Eq { 0x61 } else { 0x62 }),
                // String equality is handled in `emit_bind` (it needs `env` to
                // call `__streq` + drop literal temps), so it never reaches here.
                WType::Str => {
                    return Err("wasm backend: internal — String `==` should be routed earlier".into())
                }
                // Structural ADT equality needs a recursive compare (and would
                // interact with reference counting); out of the 2b subset.
                WType::Ref => {
                    return Err(
                        "wasm backend: structural `==`/`!=` on ADTs is outside the 2b subset".into(),
                    )
                }
                // Tensor structural `==` would need an elementwise compare;
                // outside the wasm subset (clean Err, never a panic).
                WType::Tensor => {
                    return Err(
                        "wasm backend: `==`/`!=` on Tensor is outside the wasm subset".into(),
                    )
                }
                WType::F32 => unreachable!("F32 is not an Aria value type"),
            }
            Ok(WType::I32)
        }
        // `&&` / `||` are lowered to `Bind::If` by the IR, so a raw And/Or Prim
        // should not normally appear; support it anyway for completeness (both
        // operands are Bool / i32).
        BinOp::And | BinOp::Or => {
            if lt != WType::I32 || rt != WType::I32 {
                return Err("wasm backend: logical op expects Bool operands".into());
            }
            code.push(if op == BinOp::And { 0x71 } else { 0x72 }); // i32.and / i32.or
            Ok(WType::I32)
        }
    }
}

/// Emit one branch of an `if` whose result valtype is `result_ty`. The dead
/// `Ret(Unit)` marker becomes `unreachable`. A branch ending in a `TailCall`
/// diverges via `br` (stack-polymorphic) — emit it without a type check. A
/// normal branch must leave a value of `result_ty`.
fn emit_if_branch(
    branch: &IExpr,
    result_ty: WType,
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<(), String> {
    if is_unreachable_unit(branch) {
        code.push(0x00); // unreachable (statically-dead branch)
        return Ok(());
    }
    if is_diverging(branch) {
        // Ends in a `TailCall` (loop back-edge); the branch leaves via `br`.
        emit_iexpr(branch, env, code)?;
        return Ok(());
    }
    let t = emit_iexpr(branch, env, code)?;
    if t != result_ty {
        return Err("wasm backend: `if` branch type disagrees".into());
    }
    Ok(())
}

/// Emit an `IExpr`, leaving its result value on the operand stack.
fn emit_iexpr(e: &IExpr, env: &mut LocalEnv, code: &mut Vec<u8>) -> Result<WType, String> {
    match e {
        IExpr::Ret(a) => emit_atom(a, env, code),
        IExpr::Let(x, bind, body) => {
            let t = emit_bind(bind, env, code)?;
            env.add_local(x, t);
            let idx = env.var_index(x)?;
            code.push(0x21); // local.set
            leb_u(idx as u64, code);
            emit_iexpr(body, env, code)
        }
        IExpr::Dup(v, body) => {
            // dup is a no-op on unboxed values; ADT cells use `__dup`, Strings
            // `__dup_str`, Tensors `__dup_tensor` (all bump the rc at [ptr+0]).
            let h = match env.var_type(v)? {
                WType::Ref => Some(HeapHelper::Dup),
                WType::Str => Some(HeapHelper::DupStr),
                WType::Tensor => Some(HeapHelper::DupTensor),
                _ => None,
            };
            if let Some(h) = h {
                code.push(0x20); // local.get v
                leb_u(env.var_index(v)? as u64, code);
                code.push(0x10); // call __dup / __dup_str / __dup_tensor
                leb_u(env.heap_index(h) as u64, code);
                code.push(0x1A); // drop the dummy i32 result
            }
            emit_iexpr(body, env, code)
        }
        IExpr::Drop(v, body) => {
            let h = match env.var_type(v)? {
                WType::Ref => Some(HeapHelper::Drop),
                WType::Str => Some(HeapHelper::DropStr),
                WType::Tensor => Some(HeapHelper::DropTensor),
                _ => None,
            };
            if let Some(h) = h {
                code.push(0x20); // local.get v
                leb_u(env.var_index(v)? as u64, code);
                code.push(0x10); // call __drop / __drop_str / __drop_tensor
                leb_u(env.heap_index(h) as u64, code);
                code.push(0x1A); // drop the dummy i32 result
            }
            emit_iexpr(body, env, code)
        }
        IExpr::DropReuse(scrut, tok, body) => {
            // Bind a fresh i32 local for the reuse token. If `scrut` is a Ref,
            // `tok = __drop_reuse(scrut)` (the cell ptr if unique-and-dead, else
            // 0). For a non-Ref scrutinee the cell can't be reused: tok = 0.
            env.add_local(tok, WType::I32);
            let tok_idx = env.var_index(tok)?;
            if env.var_type(scrut)? == WType::Ref {
                code.push(0x20); // local.get scrut
                leb_u(env.var_index(scrut)? as u64, code);
                code.push(0x10); // call __drop_reuse
                leb_u(env.heap_index(HeapHelper::DropReuse) as u64, code);
            } else {
                code.push(0x41); // i32.const 0 (no reusable cell)
                leb_s(0, code);
            }
            code.push(0x21); // local.set tok
            leb_u(tok_idx as u64, code);
            emit_iexpr(body, env, code)
        }
        IExpr::TailCall(args) => {
            // Self-tail-call -> loop back-edge. Push ALL new argument values onto
            // the stack FIRST (each arg reads the OLD param locals, so this is
            // safe even when a parameter appears in another argument), then pop
            // them into the param locals in REVERSE order, then `br` to the loop.
            // Ownership transfers to the params exactly as a real call's binding
            // (rc-balanced); the old param values are overwritten by the new ones.
            let tail = env
                .tail
                .clone()
                .ok_or("wasm backend: TailCall outside a tail-recursive function (internal)")?;
            if tail.params.len() != args.len() {
                return Err("wasm backend: TailCall arity mismatch (internal)".into());
            }
            for a in args {
                emit_atom(a, env, code)?;
            }
            for (idx, _ty) in tail.params.iter().rev() {
                code.push(0x21); // local.set <param>
                leb_u(*idx as u64, code);
            }
            // `br` to the enclosing loop: relative target = (current depth) -
            // (loop's depth). Directly inside the loop body that is 0.
            let target = env.block_depth - tail.loop_depth;
            code.push(0x0C); // br
            leb_u(target as u64, code);
            // After an unconditional `br` the operand stack is polymorphic; report
            // the function's return type so any value-checking caller is satisfied.
            Ok(tail.ret)
        }
    }
}

/// Build the complete code-section *entry* (locals declaration + body +
/// trailing `end`) for a checked-arithmetic helper. Each helper computes the
/// wrapping result, checks the signed-i64 overflow condition, and executes
/// `unreachable` (trap) on overflow; otherwise it returns the result.
///
/// Helper signatures (params are locals 0..n):
///   * Add/Sub/Mul(a: i64, b: i64) -> i64
///   * Neg(a: i64) -> i64
fn emit_ovf_helper(h: OvfHelper) -> Vec<u8> {
    // Opcode aliases for readability.
    const I64_CONST: u8 = 0x42;
    const I64_ADD: u8 = 0x7C;
    const I64_SUB: u8 = 0x7D;
    const I64_MUL: u8 = 0x7E;
    const I64_DIV_S: u8 = 0x7F;
    const I64_XOR: u8 = 0x85;
    const I64_AND: u8 = 0x83;
    const I64_LT_S: u8 = 0x53;
    const I64_EQ: u8 = 0x51;
    const LOCAL_GET: u8 = 0x20;
    const LOCAL_TEE: u8 = 0x22;
    const IF: u8 = 0x04;
    const ELSE: u8 = 0x05;
    const END: u8 = 0x0B;
    const UNREACHABLE: u8 = 0x00;
    const RETURN: u8 = 0x0F;
    const BT_I64: u8 = 0x7E; // blocktype result = i64

    // How many extra i64 locals (beyond params) each helper needs to hold its
    // wrapped result `r`.
    let (n_params, extra_locals): (u32, u32) = match h {
        OvfHelper::Add | OvfHelper::Sub | OvfHelper::Mul => (2, 1), // a,b + r
        OvfHelper::Neg => (1, 0),                                   // a only
    };
    // Local index of the result temp `r` (first slot after params).
    let r = n_params;

    let mut body: Vec<u8> = Vec::new();
    let const_i64 = |v: i64, out: &mut Vec<u8>| {
        out.push(I64_CONST);
        leb_s(v, out);
    };

    match h {
        OvfHelper::Add => {
            // r = a + b
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            body.push(I64_ADD);
            body.push(LOCAL_TEE);
            leb_u(r as u64, &mut body); // r = a+b, leave r on stack
            // overflow iff ((a ^ r) & (b ^ r)) < 0
            // currently stack: [r]; build (a^r):
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a   stack: [r, a]
            body.push(I64_XOR); //         (a ^ r)  -- xor is commutative
            // (b ^ r):
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            body.push(LOCAL_GET);
            leb_u(r as u64, &mut body); // r
            body.push(I64_XOR); // (b ^ r)
            body.push(I64_AND); // (a^r) & (b^r)
            const_i64(0, &mut body);
            body.push(I64_LT_S); // < 0  ?
            body.push(IF);
            body.push(BT_I64);
            body.push(UNREACHABLE); // overflow -> trap
            body.push(ELSE);
            body.push(LOCAL_GET);
            leb_u(r as u64, &mut body); // return r
            body.push(END);
        }
        OvfHelper::Sub => {
            // r = a - b
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            body.push(I64_SUB);
            body.push(0x21); // local.set r = a-b  (stack now empty)
            leb_u(r as u64, &mut body);
            // overflow iff ((a ^ b) & (a ^ r)) < 0
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            body.push(I64_XOR); // (a ^ b)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            body.push(LOCAL_GET);
            leb_u(r as u64, &mut body); // r
            body.push(I64_XOR); // (a ^ r)
            body.push(I64_AND); // (a^b) & (a^r)
            const_i64(0, &mut body);
            body.push(I64_LT_S);
            body.push(IF);
            body.push(BT_I64);
            body.push(UNREACHABLE);
            body.push(ELSE);
            body.push(LOCAL_GET);
            leb_u(r as u64, &mut body);
            body.push(END);
        }
        OvfHelper::Mul => {
            // if a == 0 -> 0
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            const_i64(0, &mut body);
            body.push(I64_EQ);
            body.push(IF);
            body.push(BT_I64);
            const_i64(0, &mut body); // result 0
            body.push(RETURN);
            body.push(ELSE);
            // else if a == -1 -> overflow iff b == i64::MIN, else -b
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            const_i64(-1, &mut body);
            body.push(I64_EQ);
            body.push(IF);
            body.push(BT_I64);
            // a == -1
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            const_i64(I64_MIN, &mut body);
            body.push(I64_EQ);
            body.push(IF);
            body.push(BT_I64);
            body.push(UNREACHABLE); // -i64::MIN overflows
            body.push(ELSE);
            // result = 0 - b
            const_i64(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            body.push(I64_SUB);
            body.push(RETURN);
            body.push(END); // end inner (b==MIN) if
            // (the a==-1 if-block must yield i64; the then-branch returned, the
            //  else-branch returned — but wasm still needs a value for the block
            //  type on fall-through. Both arms `return`, so this point is
            //  unreachable; emit `unreachable` to satisfy validation.)
            body.push(UNREACHABLE);
            body.push(ELSE);
            // general case: a != 0 and a != -1, so div_s is safe.
            // r = a * b
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            body.push(I64_MUL);
            body.push(LOCAL_TEE);
            leb_u(r as u64, &mut body); // r  stack:[r]
            // overflow iff (r / a) != b
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a    stack:[r, a]
            body.push(I64_DIV_S); // r / a
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            body.push(I64_EQ); // (r/a) == b ?
            body.push(IF);
            body.push(BT_I64);
            body.push(LOCAL_GET);
            leb_u(r as u64, &mut body); // ok -> return r
            body.push(ELSE);
            body.push(UNREACHABLE); // overflow -> trap
            body.push(END); // end (r/a)==b if
            body.push(END); // end a==-1 if
            body.push(END); // end a==0 if
        }
        OvfHelper::Neg => {
            // overflow iff a == i64::MIN; else 0 - a
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            const_i64(I64_MIN, &mut body);
            body.push(I64_EQ);
            body.push(IF);
            body.push(BT_I64);
            body.push(UNREACHABLE);
            body.push(ELSE);
            const_i64(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            body.push(I64_SUB); // 0 - a
            body.push(END);
        }
    }

    // Assemble the entry: locals declaration + body + trailing `end`.
    let mut entry = Vec::new();
    if extra_locals == 0 {
        leb_u(0, &mut entry); // no local groups
    } else {
        leb_u(1, &mut entry); // one group
        leb_u(extra_locals as u64, &mut entry); // count
        entry.push(WType::I64.byte());
    }
    entry.extend_from_slice(&body);
    entry.push(END);
    entry
}

// ---- heap runtime helpers (Phase 2b) ------------------------------------

/// Wrap a finished helper body (`body`, WITHOUT a trailing `end`) and a list of
/// extra-local *types* into a code-section entry (locals decl + body + end).
fn helper_entry(extra_locals: &[WType], mut body: Vec<u8>) -> Vec<u8> {
    let mut entry = Vec::new();
    // One local group per local (simple, no run-length encoding).
    leb_u(extra_locals.len() as u64, &mut entry);
    for lty in extra_locals {
        leb_u(1, &mut entry);
        entry.push(lty.byte());
    }
    entry.append(&mut body);
    entry.push(0x0B); // end
    entry
}

/// Emit one heap-runtime helper's code-section entry. `heap_base` is the wasm
/// function index of `HeapHelper::Alloc`; the helpers call each other through
/// it. The per-tag Ref-field knowledge in `__drop` is compiled in from `ctors`.
fn emit_heap_helper(
    h: HeapHelper,
    ctors: &CtorTable,
    heap_base: u32,
    exp_idx: u32,
    embed_scratch: u64,
    closure_drops: &[ClosureDropInfo],
) -> Vec<u8> {
    const LOCAL_GET: u8 = 0x20;
    const LOCAL_SET: u8 = 0x21;
    const LOCAL_TEE: u8 = 0x22;
    const I32_CONST: u8 = 0x41;
    const I64_CONST: u8 = 0x42;
    const I32_ADD: u8 = 0x6A;
    const I32_SUB: u8 = 0x6B;
    const I32_MUL: u8 = 0x6C;
    const I32_NE: u8 = 0x47;
    const I32_WRAP_I64: u8 = 0xA7;
    const I64_EXTEND_I32_U: u8 = 0xAD;
    const I64_ADD: u8 = 0x7C;
    const I64_SUB: u8 = 0x7D;
    const I64_EQZ: u8 = 0x50;
    const I64_EQ: u8 = 0x51;
    const CALL: u8 = 0x10;
    const DROP: u8 = 0x1A;
    const IF: u8 = 0x04;
    const ELSE: u8 = 0x05;
    const END: u8 = 0x0B;
    const BT_VOID: u8 = 0x40; // empty block type

    let drop_idx = heap_base + HeapHelper::Drop.offset();
    let free_idx = heap_base + HeapHelper::Free.offset();
    let drop_str_idx = heap_base + HeapHelper::DropStr.offset();
    let alloc_idx = heap_base + HeapHelper::Alloc.offset();
    let alloc_str_idx = heap_base + HeapHelper::AllocStr.offset();
    let streq_idx = heap_base + HeapHelper::StrEq.offset();
    let eq_idx = heap_base + HeapHelper::Eq.offset();
    let alloc_tensor_idx = heap_base + HeapHelper::AllocTensor.offset();
    let drop_tensor_idx = heap_base + HeapHelper::DropTensor.offset();
    let hash_embed_idx = heap_base + HeapHelper::HashEmbed.offset();

    let mut body: Vec<u8> = Vec::new();
    match h {
        HeapHelper::Alloc => {
            // params: nfields(0). locals: fl_addr(1,i32), head(2,i32), ptr(3,i32)
            // fl_addr = FREELIST_BASE + nfields*4
            body.push(I32_CONST);
            leb_s(FREELIST_BASE as i64, &mut body);
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // nfields
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(1, &mut body); // fl_addr
            // head = i32.load[fl_addr]
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            i32_load(0, &mut body);
            body.push(LOCAL_TEE);
            leb_u(2, &mut body); // head
            // if head != 0 { pop free-list } else { bump }
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(I32_NE);
            body.push(IF);
            body.push(BT_VOID);
            // free-list head reuse: next = i32.wrap(i64.load[head+8]); store back
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // fl_addr
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // head
            i64_load(8, &mut body);
            body.push(I32_WRAP_I64);
            i32_store(0, &mut body); // fl_addr <- next
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // head
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // ptr = head
            body.push(ELSE);
            // bump: ptr = bump; bump += 16 + 8*nfields
            body.push(I32_CONST);
            leb_s(BUMP_PTR_ADDR as i64, &mut body);
            i32_load(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // ptr
            // bump = ptr + CELL_HEADER + 8*nfields
            body.push(I32_CONST);
            leb_s(BUMP_PTR_ADDR as i64, &mut body); // address for the store
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // ptr
            body.push(I32_CONST);
            leb_s(CELL_HEADER as i64, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // nfields
            body.push(I32_CONST);
            leb_s(SLOT as i64, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD); // ptr + 16 + 8*nfields
            i32_store(0, &mut body); // bump <- new value
            body.push(END); // end if
            // rc = 1 at [ptr]
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // ptr
            body.push(I64_CONST);
            leb_s(1, &mut body);
            i64_store(0, &mut body);
            // live++
            body.push(I32_CONST);
            leb_s(LIVE_ADDR as i64, &mut body);
            body.push(I32_CONST);
            leb_s(LIVE_ADDR as i64, &mut body);
            i64_load(0, &mut body);
            body.push(I64_CONST);
            leb_s(1, &mut body);
            body.push(I64_ADD);
            i64_store(0, &mut body);
            // return ptr
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            helper_entry(&[WType::I32, WType::I32, WType::I32], body)
        }
        HeapHelper::Free => {
            // params: ptr(0), nfields(1). local: fl_addr(2,i32)
            body.push(I32_CONST);
            leb_s(FREELIST_BASE as i64, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // nfields
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // fl_addr
            // [ptr+8] = extend(old head)   (link)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // fl_addr
            i32_load(0, &mut body); // old head
            body.push(I64_EXTEND_I32_U);
            i64_store(8, &mut body);
            // fl_addr <- ptr
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // fl_addr
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            i32_store(0, &mut body);
            // live--
            body.push(I32_CONST);
            leb_s(LIVE_ADDR as i64, &mut body);
            body.push(I32_CONST);
            leb_s(LIVE_ADDR as i64, &mut body);
            i64_load(0, &mut body);
            body.push(I64_CONST);
            leb_s(1, &mut body);
            body.push(I64_SUB);
            i64_store(0, &mut body);
            // return 0 (dummy)
            body.push(I32_CONST);
            leb_s(0, &mut body);
            helper_entry(&[WType::I32], body)
        }
        HeapHelper::Dup => {
            // params: ptr(0). rc = i64.load[ptr]; store rc+1.
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr (address for store)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            i64_load(0, &mut body);
            body.push(I64_CONST);
            leb_s(1, &mut body);
            body.push(I64_ADD);
            i64_store(0, &mut body);
            body.push(I32_CONST);
            leb_s(0, &mut body); // dummy return
            helper_entry(&[], body)
        }
        HeapHelper::Drop => {
            // params: ptr(0). local: rc(1, i64).
            // rc = i64.load[ptr] - 1; store back.
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr (address)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            i64_load(0, &mut body);
            body.push(I64_CONST);
            leb_s(1, &mut body);
            body.push(I64_SUB);
            body.push(LOCAL_TEE);
            leb_u(1, &mut body); // rc (leave on stack for the store)
            i64_store(0, &mut body);
            // if rc == 0 { drop ref fields per tag; free }
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // rc
            body.push(I64_EQZ);
            body.push(IF);
            body.push(BT_VOID);
            // Per-tag dispatch: for each ctor with >=1 ref field OR to free, emit
            // `if tag == T { drop ref fields; free(ptr, arity) }`. We must free
            // EVERY tag (even with no ref fields), so we emit a branch per ctor.
            // Order: independent ifs (not else-chained) keyed on tag equality.
            for info in ctor_infos_sorted(ctors) {
                // load tag
                body.push(LOCAL_GET);
                leb_u(0, &mut body); // ptr
                i64_load(8, &mut body); // tag
                body.push(I64_CONST);
                leb_s(info.tag, &mut body);
                body.push(I64_EQ);
                body.push(IF);
                body.push(BT_VOID);
                // recursively drop each Ref field (ADT child) via __drop
                for fi in info.ref_fields() {
                    body.push(LOCAL_GET);
                    leb_u(0, &mut body); // ptr
                    i64_load(CELL_HEADER + SLOT * fi as u64, &mut body);
                    body.push(I32_WRAP_I64); // field address
                    body.push(CALL);
                    leb_u(drop_idx as u64, &mut body);
                    body.push(DROP); // dummy result
                }
                // drop each String field via __drop_str (a String field is
                // reference-managed, just like a Ref field)
                for fi in info.str_fields() {
                    body.push(LOCAL_GET);
                    leb_u(0, &mut body); // ptr
                    i64_load(CELL_HEADER + SLOT * fi as u64, &mut body);
                    body.push(I32_WRAP_I64); // String field address
                    body.push(CALL);
                    leb_u(drop_str_idx as u64, &mut body);
                    body.push(DROP); // dummy result
                }
                // free(ptr, arity)
                body.push(LOCAL_GET);
                leb_u(0, &mut body); // ptr
                body.push(I32_CONST);
                leb_s(info.arity() as i64, &mut body);
                body.push(CALL);
                leb_u(free_idx as u64, &mut body);
                body.push(DROP); // dummy result
                body.push(END); // end if tag==T
            }
            // Closure cells: per closure tag, release the reference-counted
            // captures (`Ref` -> __drop, `Str` -> __drop_str), then free the cell
            // (arity = capture count). Same shape as the constructor branches.
            for cd in closure_drops {
                body.push(LOCAL_GET);
                leb_u(0, &mut body); // ptr
                i64_load(8, &mut body); // tag
                body.push(I64_CONST);
                leb_s(cd.tag, &mut body);
                body.push(I64_EQ);
                body.push(IF);
                body.push(BT_VOID);
                for (i, t) in &cd.managed {
                    body.push(LOCAL_GET);
                    leb_u(0, &mut body); // ptr
                    i64_load(CELL_HEADER + SLOT * *i as u64, &mut body);
                    body.push(I32_WRAP_I64); // capture address
                    body.push(CALL);
                    let idx = match t {
                        WType::Str => drop_str_idx,
                        _ => drop_idx,
                    };
                    leb_u(idx as u64, &mut body);
                    body.push(DROP); // dummy result
                }
                // free(ptr, ncaps)
                body.push(LOCAL_GET);
                leb_u(0, &mut body); // ptr
                body.push(I32_CONST);
                leb_s(cd.ncaps as i64, &mut body);
                body.push(CALL);
                leb_u(free_idx as u64, &mut body);
                body.push(DROP); // dummy result
                body.push(END); // end if tag==closure_tag
            }
            body.push(END); // end if rc==0
            body.push(I32_CONST);
            leb_s(0, &mut body); // dummy return
            helper_entry(&[WType::I64], body)
        }
        HeapHelper::Live => {
            body.push(I32_CONST);
            leb_s(LIVE_ADDR as i64, &mut body);
            i64_load(0, &mut body);
            helper_entry(&[], body)
        }
        HeapHelper::DropReuse => {
            // params: ptr(0). local: rc(1, i64).
            // rc = i64.load[ptr] - 1; store back.
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr (address)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            i64_load(0, &mut body);
            body.push(I64_CONST);
            leb_s(1, &mut body);
            body.push(I64_SUB);
            body.push(LOCAL_TEE);
            leb_u(1, &mut body); // rc (leave on stack for the store)
            i64_store(0, &mut body);
            // if rc == 0 { drop ref CHILDREN (per tag), KEEP this slot; return ptr }
            // else { return 0 }.
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // rc
            body.push(I64_EQZ);
            body.push(IF);
            body.push(0x7F); // blocktype result = i32 (the returned token)
            // Per-tag dispatch: recursively drop each Ref child (reusing __drop's
            // per-tag Ref-field knowledge), but DO NOT free or decrement __live
            // for THIS cell — its slot is retained for reuse.
            for info in ctor_infos_sorted(ctors) {
                if !info.has_managed_fields() {
                    continue; // no children to release for this tag
                }
                body.push(LOCAL_GET);
                leb_u(0, &mut body); // ptr
                i64_load(8, &mut body); // tag
                body.push(I64_CONST);
                leb_s(info.tag, &mut body);
                body.push(I64_EQ);
                body.push(IF);
                body.push(BT_VOID);
                for fi in info.ref_fields() {
                    body.push(LOCAL_GET);
                    leb_u(0, &mut body); // ptr
                    i64_load(CELL_HEADER + SLOT * fi as u64, &mut body);
                    body.push(I32_WRAP_I64); // child address
                    body.push(CALL);
                    leb_u(drop_idx as u64, &mut body);
                    body.push(DROP); // dummy result
                }
                for fi in info.str_fields() {
                    body.push(LOCAL_GET);
                    leb_u(0, &mut body); // ptr
                    i64_load(CELL_HEADER + SLOT * fi as u64, &mut body);
                    body.push(I32_WRAP_I64); // String child address
                    body.push(CALL);
                    leb_u(drop_str_idx as u64, &mut body);
                    body.push(DROP); // dummy result
                }
                body.push(END); // end if tag==T
            }
            // token = ptr (slot retained for reuse)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            body.push(ELSE);
            // shared cell: just decremented; no reusable slot.
            body.push(I32_CONST);
            leb_s(0, &mut body); // token = 0 (null)
            body.push(END); // end if rc==0
            helper_entry(&[WType::I64], body)
        }
        HeapHelper::Reuses => {
            body.push(I32_CONST);
            leb_s(REUSES_ADDR as i64, &mut body);
            i64_load(0, &mut body);
            helper_entry(&[], body)
        }
        HeapHelper::AllocStr => {
            // (len:i32) -> ptr:i32. Round (STR_HEADER + len) up to 8; size class
            // = (rounded - CELL_HEADER) / SLOT, clamped to STR_MAX_CLASS. Alloc a
            // cell of that class via __alloc (rc=1, live++), then store len at
            // [ptr+8]. Locals: cls(1,i32), ptr(2,i32).
            // cls = ((STR_HEADER + len + 7) & ~7 - CELL_HEADER) / SLOT
            body.push(I32_CONST);
            leb_s(STR_HEADER as i64, &mut body);
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // len
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(7, &mut body);
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(-8, &mut body); // i32 0xFFFFFFF8 (= ~7), signed-LEB encoded
            body.push(0x71); // i32.and  -> rounded total
            body.push(I32_CONST);
            leb_s(CELL_HEADER as i64, &mut body);
            body.push(I32_SUB); // rounded - 16
            body.push(I32_CONST);
            leb_s(SLOT as i64, &mut body);
            body.push(0x6E); // i32.div_u  -> size class
            // clamp to STR_MAX_CLASS: if cls > MAX { cls = MAX }
            body.push(LOCAL_TEE);
            leb_u(1, &mut body); // cls
            body.push(I32_CONST);
            leb_s(STR_MAX_CLASS as i64, &mut body);
            body.push(0x4B); // i32.gt_u
            body.push(IF);
            body.push(BT_VOID);
            body.push(I32_CONST);
            leb_s(STR_MAX_CLASS as i64, &mut body);
            body.push(LOCAL_SET);
            leb_u(1, &mut body); // cls = MAX
            body.push(END);
            // ptr = __alloc(cls)
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // cls
            body.push(CALL);
            leb_u(alloc_idx as u64, &mut body);
            body.push(LOCAL_TEE);
            leb_u(2, &mut body); // ptr
            // store len at [ptr+8]
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // len (i32)
            body.push(I64_EXTEND_I32_U); // -> i64
            i64_store(8, &mut body);
            // return ptr
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            helper_entry(&[WType::I32, WType::I32], body)
        }
        HeapHelper::DupStr => {
            // (ptr:i32) -> i32. rc++ (same as __dup).
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr (address)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            i64_load(0, &mut body);
            body.push(I64_CONST);
            leb_s(1, &mut body);
            body.push(I64_ADD);
            i64_store(0, &mut body);
            body.push(I32_CONST);
            leb_s(0, &mut body); // dummy return
            helper_entry(&[], body)
        }
        HeapHelper::DropStr => {
            // (ptr:i32) -> i32. rc--; at 0 free the String (no children). The
            // size class to free with is recomputed from len: same formula as
            // AllocStr. Locals: rc(1,i64), cls(2,i32).
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr (address)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            i64_load(0, &mut body);
            body.push(I64_CONST);
            leb_s(1, &mut body);
            body.push(I64_SUB);
            body.push(LOCAL_TEE);
            leb_u(1, &mut body); // rc
            i64_store(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // rc
            body.push(I64_EQZ);
            body.push(IF);
            body.push(BT_VOID);
            // cls = clamp(((STR_HEADER + len + 7)&~7 - 16)/8, MAX)
            body.push(I32_CONST);
            leb_s(STR_HEADER as i64, &mut body);
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            i64_load(8, &mut body); // len (i64)
            body.push(I32_WRAP_I64); // -> i32
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(7, &mut body);
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(-8, &mut body); // i32 0xFFFFFFF8 (= ~7), signed-LEB encoded
            body.push(0x71); // i32.and
            body.push(I32_CONST);
            leb_s(CELL_HEADER as i64, &mut body);
            body.push(I32_SUB);
            body.push(I32_CONST);
            leb_s(SLOT as i64, &mut body);
            body.push(0x6E); // i32.div_u
            body.push(LOCAL_TEE);
            leb_u(2, &mut body); // cls
            body.push(I32_CONST);
            leb_s(STR_MAX_CLASS as i64, &mut body);
            body.push(0x4B); // i32.gt_u
            body.push(IF);
            body.push(BT_VOID);
            body.push(I32_CONST);
            leb_s(STR_MAX_CLASS as i64, &mut body);
            body.push(LOCAL_SET);
            leb_u(2, &mut body);
            body.push(END);
            // __free(ptr, cls)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // ptr
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // cls
            body.push(CALL);
            leb_u(free_idx as u64, &mut body);
            body.push(DROP); // dummy
            body.push(END); // end if rc==0
            body.push(I32_CONST);
            leb_s(0, &mut body); // dummy return
            helper_entry(&[WType::I64, WType::I32], body)
        }
        HeapHelper::Concat => {
            // (a:i32, b:i32) -> ptr:i32. result = alloc_str(len(a)+len(b));
            // copy a's bytes then b's bytes; drop a; drop b; return result.
            // Locals: la(2,i32), lb(3,i32), out(4,i32), i(5,i32).
            // la = len(a)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            i64_load(8, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // la
            // lb = len(b)
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            i64_load(8, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // lb
            // out = __alloc_str(la + lb)
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(I32_ADD);
            body.push(CALL);
            leb_u(alloc_str_idx as u64, &mut body);
            body.push(LOCAL_SET);
            leb_u(4, &mut body); // out
            // copy a: i = 0; while i < la { out[16+i] = a[16+i]; i++ }
            emit_byte_copy_loop(&mut body, /*src*/ 0, /*dst*/ 4, /*len*/ 2, /*dst_off*/ 0, /*i*/ 5);
            // copy b into out at offset la: while i<lb { out[16+la+i]=b[16+i] }
            // We reuse i; dst byte index = la + i, handled via a running dst ptr.
            emit_byte_copy_loop_offset(&mut body, /*src*/ 1, /*dst*/ 4, /*len*/ 3, /*dst_base_extra*/ 2, /*i*/ 5);
            // drop a, drop b (this builtin consumes its args)
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(CALL);
            leb_u(drop_str_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(CALL);
            leb_u(drop_str_idx as u64, &mut body);
            body.push(DROP);
            // return out
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            helper_entry(&[WType::I32, WType::I32, WType::I32, WType::I32], body)
        }
        HeapHelper::IntToStr => {
            // (n:i64) -> ptr:i32. Format n in decimal UTF-8. Matches Rust's
            // i64::to_string: "0", negatives prefixed with '-'. Algorithm:
            //   neg = n < 0; if neg, work with magnitude (use i64, careful with
            //   i64::MIN: we operate on the value via unsigned-safe digit extract
            //   by negating per-digit). We compute digits into a 24-byte scratch
            //   buffer high-to-low at SCRATCH, then alloc_str and copy.
            // Simpler robust approach: handle sign, then repeatedly take
            //   d = n % 10 (could be negative), char = '0' + |d|, n = n / 10.
            // This works for i64::MIN too since we never negate the whole value.
            // Locals: neg(1,i32), buf(2,i32 scratch addr), p(3,i32 write ptr),
            //   len(4,i32), d(5,i64), ptr(6,i32 result), n stays in 0.
            //
            // We use a fixed scratch region in linear memory at SCRATCH_ADDR to
            // build digits low-to-high, then reverse-copy into the String.
            // n == 0 special-case.
            // buf = SCRATCH (a 32-byte scratch at a fixed high-ish bookkeeping
            // address — we reuse REUSES? no. We use the free area just after the
            // freelist; but that's the heap. Instead build digits on the wasm
            // stack is awkward; use a small fixed scratch in low memory that is
            // never otherwise used: bytes [ITOA_SCRATCH .. +24).)
            const ITOA_SCRATCH: i64 = 4; // bytes 4..8 are unused padding before LIVE_ADDR? No.
            let _ = ITOA_SCRATCH;
            // Build into the to-be-allocated String directly is hard (length
            // unknown up front). Two-pass: pass 1 count digits, pass 2 fill.
            // count: tmp=n; cnt=0; if neg cnt++ ; do { cnt++; tmp/=10 } while tmp!=0
            // Locals indices: neg(1,i32) cnt(2,i32) tmp(3,i64) ptr(4,i32) wp(5,i32) dig(6,i64)
            // neg = n < 0
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x53); // i64.lt_s
            body.push(LOCAL_SET);
            leb_u(1, &mut body); // neg
            // cnt = 0 ; tmp = n
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // cnt
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // tmp
            // loop: cnt++ ; tmp = tmp / 10 ; if tmp != 0 continue
            body.push(0x03); // loop
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // cnt
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // cnt++
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // tmp
            body.push(I64_CONST);
            leb_s(10, &mut body);
            body.push(0x7F); // i64.div_s
            body.push(LOCAL_TEE);
            leb_u(3, &mut body); // tmp
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x52); // i64.ne
            body.push(0x0D); // br_if
            leb_u(0, &mut body); // -> loop header
            body.push(END); // end loop
            // if neg cnt++
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // neg
            body.push(IF);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // cnt++
            body.push(END);
            // ptr = __alloc_str(cnt)
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // cnt
            body.push(CALL);
            leb_u(alloc_str_idx as u64, &mut body);
            body.push(LOCAL_SET);
            leb_u(4, &mut body); // ptr
            // wp = ptr + 16 + cnt   (write digits backwards from the end)
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(I32_CONST);
            leb_s(STR_HEADER as i64, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // cnt
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(5, &mut body); // wp (one past last byte)
            // tmp = n
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // tmp
            // loop: dig = tmp % 10 (signed, may be <=0 for negatives);
            //   ch = '0' + |dig| ; wp-- ; store ch ; tmp /= 10 ; if tmp!=0 cont
            body.push(0x03); // loop
            body.push(BT_VOID);
            // wp = wp - 1
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_SUB);
            body.push(LOCAL_SET);
            leb_u(5, &mut body); // wp--
            // dig = tmp % 10
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // tmp
            body.push(I64_CONST);
            leb_s(10, &mut body);
            body.push(0x81); // i64.rem_s
            body.push(LOCAL_SET);
            leb_u(6, &mut body); // dig (in [-9,9])
            // |dig|: if dig < 0 dig = -dig
            body.push(LOCAL_GET);
            leb_u(6, &mut body);
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x53); // i64.lt_s
            body.push(IF);
            body.push(BT_VOID);
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(6, &mut body);
            body.push(I64_SUB); // 0 - dig
            body.push(LOCAL_SET);
            leb_u(6, &mut body);
            body.push(END);
            // store byte: [wp] = '0' + dig  (i32.store8). address = wp, value:
            body.push(LOCAL_GET);
            leb_u(5, &mut body); // wp (addr)
            body.push(I32_CONST);
            leb_s('0' as i64, &mut body);
            body.push(LOCAL_GET);
            leb_u(6, &mut body); // dig (i64)
            body.push(I32_WRAP_I64);
            body.push(I32_ADD); // '0' + dig
            i32_store8(0, &mut body);
            // tmp = tmp / 10 ; if tmp != 0 continue
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(I64_CONST);
            leb_s(10, &mut body);
            body.push(0x7F); // i64.div_s
            body.push(LOCAL_TEE);
            leb_u(3, &mut body);
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x52); // i64.ne
            body.push(0x0D); // br_if
            leb_u(0, &mut body);
            body.push(END); // end loop
            // if neg: wp-- ; [wp] = '-'
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // neg
            body.push(IF);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_SUB);
            body.push(LOCAL_TEE);
            leb_u(5, &mut body); // wp--
            body.push(I32_CONST);
            leb_s('-' as i64, &mut body);
            i32_store8(0, &mut body);
            body.push(END);
            // return ptr
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            helper_entry(
                &[WType::I32, WType::I32, WType::I64, WType::I32, WType::I32, WType::I64],
                body,
            )
        }
        HeapHelper::StrEq => {
            // (a:i32, b:i32) -> i32. 1 if len(a)==len(b) and all bytes equal.
            // Locals: la(2,i32), i(3,i32).
            // if len(a) != len(b) return 0
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            i64_load(8, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            i64_load(8, &mut body);
            body.push(0x52); // i64.ne
            body.push(IF);
            body.push(BT_VOID); // we only `return` inside
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(0x0F); // return
            body.push(END);
            // la = len(a) (i32)
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(8, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // la
            // i = 0
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body);
            // loop while i < la: if a[16+i] != b[16+i] return 0; i++
            body.push(0x03); // loop
            body.push(BT_VOID);
            // if i < la
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(0x49); // i32.lt_u
            body.push(IF);
            body.push(BT_VOID);
            // compare bytes: a[16+i] vs b[16+i]
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // i
            body.push(I32_ADD);
            i32_load8_u(STR_HEADER, &mut body); // a byte
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // i
            body.push(I32_ADD);
            i32_load8_u(STR_HEADER, &mut body); // b byte
            body.push(I32_NE);
            body.push(IF);
            body.push(BT_VOID); // one-armed if; body only `return`s, no value
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(0x0F); // return 0
            body.push(END);
            // i++
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(3, &mut body);
            body.push(0x0C); // br to loop header (continue)
            leb_u(1, &mut body); // depth 1 = the loop (0 = inner if-block)
            body.push(END); // end if i<la
            body.push(END); // end loop
            // all equal
            body.push(I32_CONST);
            leb_s(1, &mut body);
            helper_entry(&[WType::I32, WType::I32], body)
        }
        HeapHelper::StrLit => {
            // (data_addr:i32, len:i32) -> ptr:i32. ptr = __alloc_str(len);
            // copy len bytes from data_addr into [ptr+16..]; return ptr.
            // Locals: ptr(2,i32), i(3,i32).
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // len
            body.push(CALL);
            leb_u(alloc_str_idx as u64, &mut body);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // ptr
            // i = 0; while i < len { [ptr+16+i] = [data_addr+i]; i++ }
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body);
            body.push(0x03); // loop
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // i
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // len
            body.push(0x49); // i32.lt_u
            body.push(IF);
            body.push(BT_VOID);
            // dst = ptr + i ; (offset STR_HEADER folded into store8)
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // ptr
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // i
            body.push(I32_ADD);
            // src byte = [data_addr + i]
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // data_addr
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // i
            body.push(I32_ADD);
            i32_load8_u(0, &mut body);
            i32_store8(STR_HEADER, &mut body); // [ptr + i + 16]
            // i++
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(3, &mut body);
            body.push(0x0C); // br loop
            leb_u(1, &mut body);
            body.push(END); // end if
            body.push(END); // end loop
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // return ptr
            helper_entry(&[WType::I32, WType::I32], body)
        }
        HeapHelper::Eq => {
            // (a:i32, b:i32) -> i32. Structural ADT equality, recursive.
            // This helper only READS the heap; it never dups/drops. The caller
            // (`emit_adt_eq`) owns and drops both operands after the comparison.
            //
            // 1. If tag(a) != tag(b) return 0.
            // 2. Per tag T: `if tag(a) == T { return AND of field comparisons }`
            //    (no fields -> return 1). Fields compared by their static type:
            //    Int/Bool -> i64.eq (Bool is zero-extended in its slot),
            //    Float -> f64.eq, String -> __streq, Ref -> recursive __eq.
            const I64_NE: u8 = 0x52;
            const F64_EQ: u8 = 0x61;
            const I32_AND: u8 = 0x71;
            const RETURN: u8 = 0x0F;
            // if tag(a) != tag(b) { return 0 }
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            i64_load(8, &mut body); // tag(a)
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            i64_load(8, &mut body); // tag(b)
            body.push(I64_NE);
            body.push(IF);
            body.push(BT_VOID);
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(RETURN);
            body.push(END);
            // Per-tag dispatch (tags now known equal; key on tag(a)).
            for info in ctor_infos_sorted(ctors) {
                body.push(LOCAL_GET);
                leb_u(0, &mut body); // a
                i64_load(8, &mut body); // tag(a)
                body.push(I64_CONST);
                leb_s(info.tag, &mut body);
                body.push(I64_EQ);
                body.push(IF);
                body.push(BT_VOID); // one-armed if; body only `return`s its result
                // Accumulate field equalities, ANDed onto an initial `1`.
                body.push(I32_CONST);
                leb_s(1, &mut body); // start: all-equal so far
                for (fi, fty) in info.field_types.iter().enumerate() {
                    let off = CELL_HEADER + SLOT * fi as u64;
                    match fty {
                        // Int and Bool both live in the i64 slot (Bool is
                        // zero-extended), so i64.eq compares them correctly.
                        WType::I64 | WType::I32 => {
                            body.push(LOCAL_GET);
                            leb_u(0, &mut body); // a
                            i64_load(off, &mut body);
                            body.push(LOCAL_GET);
                            leb_u(1, &mut body); // b
                            i64_load(off, &mut body);
                            body.push(I64_EQ);
                        }
                        WType::F64 => {
                            body.push(LOCAL_GET);
                            leb_u(0, &mut body); // a
                            f64_load(off, &mut body);
                            body.push(LOCAL_GET);
                            leb_u(1, &mut body); // b
                            f64_load(off, &mut body);
                            body.push(F64_EQ);
                        }
                        WType::Str => {
                            // __streq(a.field, b.field)
                            body.push(LOCAL_GET);
                            leb_u(0, &mut body); // a
                            i64_load(off, &mut body);
                            body.push(I32_WRAP_I64);
                            body.push(LOCAL_GET);
                            leb_u(1, &mut body); // b
                            i64_load(off, &mut body);
                            body.push(I32_WRAP_I64);
                            body.push(CALL);
                            leb_u(streq_idx as u64, &mut body);
                        }
                        WType::Ref => {
                            // __eq(a.field, b.field)  (recursive)
                            body.push(LOCAL_GET);
                            leb_u(0, &mut body); // a
                            i64_load(off, &mut body);
                            body.push(I32_WRAP_I64);
                            body.push(LOCAL_GET);
                            leb_u(1, &mut body); // b
                            i64_load(off, &mut body);
                            body.push(I32_WRAP_I64);
                            body.push(CALL);
                            leb_u(eq_idx as u64, &mut body);
                        }
                        // Tensor fields are rejected from ADTs (CtorTable::build),
                        // so this arm is unreachable; emit a constant-true to keep
                        // the match exhaustive without affecting any real program.
                        WType::Tensor => {
                            body.push(I32_CONST);
                            leb_s(1, &mut body);
                        }
                        WType::F32 => unreachable!("F32 is not an Aria field type"),
                    }
                    body.push(I32_AND); // fold into the accumulator
                }
                body.push(RETURN); // return the accumulated equality
                body.push(END); // end if tag==T
            }
            // Unreachable in practice (tags matched one of the known ctors).
            body.push(I32_CONST);
            leb_s(0, &mut body);
            helper_entry(&[], body)
        }

        // ---- Tensor runtime (Phase 2f) ----------------------------------
        HeapHelper::AllocTensor => {
            // (rows:i32, cols:i32) -> ptr:i32. Alloc a Tensor object (rc=1,
            // header set, data UNINITIALIZED). cls = round8(24 + 4*rows*cols)
            // reduced to a free-list size class, clamped to TENSOR_MAX_CLASS.
            // Locals: n(2,i32), cls(3,i32), ptr(4,i32).
            // n = rows * cols
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(I32_MUL);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // n
            // cls = ((TENSOR_HEADER + 4*n + 7) & ~7 - CELL_HEADER) / SLOT
            body.push(I32_CONST);
            leb_s(TENSOR_HEADER as i64, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // n
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL); // 4*n
            body.push(I32_ADD); // 24 + 4n
            body.push(I32_CONST);
            leb_s(7, &mut body);
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(-8, &mut body);
            body.push(0x71); // i32.and -> rounded total
            body.push(I32_CONST);
            leb_s(CELL_HEADER as i64, &mut body);
            body.push(I32_SUB);
            body.push(I32_CONST);
            leb_s(SLOT as i64, &mut body);
            body.push(0x6E); // i32.div_u -> size class
            body.push(LOCAL_TEE);
            leb_u(3, &mut body); // cls
            // clamp to TENSOR_MAX_CLASS
            body.push(I32_CONST);
            leb_s(TENSOR_MAX_CLASS as i64, &mut body);
            body.push(0x4B); // i32.gt_u
            body.push(IF);
            body.push(BT_VOID);
            body.push(I32_CONST);
            leb_s(TENSOR_MAX_CLASS as i64, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body);
            body.push(END);
            // ptr = __alloc(cls)
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(CALL);
            leb_u(alloc_idx as u64, &mut body);
            body.push(LOCAL_TEE);
            leb_u(4, &mut body); // ptr
            // store rows at [ptr+8]
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(I64_EXTEND_I32_U);
            i64_store(8, &mut body);
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // ptr (addr for cols)
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(I64_EXTEND_I32_U);
            i64_store(16, &mut body);
            // return ptr
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            helper_entry(&[WType::I32, WType::I32, WType::I32], body)
        }
        HeapHelper::DupTensor => {
            // (ptr:i32) -> i32. rc++.
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(0, &mut body);
            body.push(I64_CONST);
            leb_s(1, &mut body);
            body.push(I64_ADD);
            i64_store(0, &mut body);
            body.push(I32_CONST);
            leb_s(0, &mut body);
            helper_entry(&[], body)
        }
        HeapHelper::DropTensor => {
            // (ptr:i32) -> i32. rc--; at 0, free with the class recomputed from
            // rows*cols. Locals: rc(1,i64), cls(2,i32).
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(0, &mut body);
            body.push(I64_CONST);
            leb_s(1, &mut body);
            body.push(I64_SUB);
            body.push(LOCAL_TEE);
            leb_u(1, &mut body); // rc
            i64_store(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(I64_EQZ);
            body.push(IF);
            body.push(BT_VOID);
            // cls from rows*cols (i64 -> i32)
            body.push(I32_CONST);
            leb_s(TENSOR_HEADER as i64, &mut body);
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(8, &mut body); // rows
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(16, &mut body); // cols
            body.push(0x7E); // i64.mul
            body.push(I32_WRAP_I64); // n (i32)
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD); // 24 + 4n
            body.push(I32_CONST);
            leb_s(7, &mut body);
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(-8, &mut body);
            body.push(0x71); // i32.and
            body.push(I32_CONST);
            leb_s(CELL_HEADER as i64, &mut body);
            body.push(I32_SUB);
            body.push(I32_CONST);
            leb_s(SLOT as i64, &mut body);
            body.push(0x6E); // i32.div_u
            body.push(LOCAL_TEE);
            leb_u(2, &mut body); // cls
            body.push(I32_CONST);
            leb_s(TENSOR_MAX_CLASS as i64, &mut body);
            body.push(0x4B); // i32.gt_u
            body.push(IF);
            body.push(BT_VOID);
            body.push(I32_CONST);
            leb_s(TENSOR_MAX_CLASS as i64, &mut body);
            body.push(LOCAL_SET);
            leb_u(2, &mut body);
            body.push(END);
            // __free(ptr, cls)
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(CALL);
            leb_u(free_idx as u64, &mut body);
            body.push(DROP);
            body.push(END); // end if rc==0
            body.push(I32_CONST);
            leb_s(0, &mut body);
            helper_entry(&[WType::I64, WType::I32], body)
        }
        HeapHelper::TensorZeros => {
            // (rows:i64, cols:i64) -> ptr:i32. Trap (unreachable) on negative
            // dims or an element count exceeding the interpreter's cap; else
            // alloc and zero the data. Locals: n(2,i64), ptr(3,i32), i(4,i32).
            const MAX_TENSOR_ELEMS: i64 = 64 * 1024 * 1024;
            // if rows < 0 || cols < 0 -> trap
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x53); // i64.lt_s
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x53); // i64.lt_s
            body.push(0x72); // i32.or
            body.push(IF);
            body.push(BT_VOID);
            body.push(0x00); // unreachable
            body.push(END);
            // n = rows * cols (i64). The interpreter caps n at 64M, well below
            // i64 overflow for non-negative dims that each fit usize, so a plain
            // i64.mul then a magnitude check matches its behavior.
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(0x7E); // i64.mul
            body.push(LOCAL_TEE);
            leb_u(2, &mut body); // n
            // if n > MAX -> trap (also catches negative wrap from huge dims)
            body.push(I64_CONST);
            leb_s(MAX_TENSOR_ELEMS, &mut body);
            body.push(0x55); // i64.gt_s
            body.push(IF);
            body.push(BT_VOID);
            body.push(0x00); // unreachable
            body.push(END);
            // ptr = __alloc_tensor(wrap rows, wrap cols)
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(I32_WRAP_I64);
            body.push(CALL);
            leb_u(alloc_tensor_idx as u64, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // ptr
            // zero the data: i = 0; while i < n { f32.store [ptr+24+4*i] = 0 }
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(4, &mut body); // i
            body.push(0x03); // loop
            body.push(BT_VOID);
            // if i < (i32)n
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(I32_WRAP_I64); // (i32)n
            body.push(0x48); // i32.lt_s
            body.push(IF);
            body.push(BT_VOID);
            // addr = ptr + 24 + 4*i
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            // value 0.0f32
            body.push(0x43); // f32.const
            body.extend_from_slice(&0.0f32.to_le_bytes());
            f32_store(TENSOR_HEADER, &mut body);
            // i++
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(4, &mut body);
            body.push(0x0C); // br loop
            leb_u(1, &mut body);
            body.push(END); // end if
            body.push(END); // end loop
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // return ptr
            helper_entry(&[WType::I64, WType::I32, WType::I32], body)
        }
        HeapHelper::TensorSet => {
            // (t:i32, r:i64, c:i64, v:f64) -> ptr:i32. Pure/immutable: CLONE t
            // (a fresh tensor with the same shape + copied data), then write one
            // element; trap on an out-of-range index (the interpreter errors).
            // Consumes t (drops it after copying). Locals: rows(4,i64),
            // cols(4?)... use: rows(4,i64), cols(5,i64), n(6,i32), out(7,i32),
            // i(8,i32).
            // rows = t.rows ; cols = t.cols
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(8, &mut body);
            body.push(LOCAL_SET);
            leb_u(4, &mut body); // rows
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(16, &mut body);
            body.push(LOCAL_SET);
            leb_u(5, &mut body); // cols
            // bounds: if r<0 || c<0 || r>=rows || c>=cols -> trap
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // r
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x53); // r < 0
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // c
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x53); // c < 0
            body.push(0x72); // or
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // r
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // rows
            body.push(0x59); // r >= rows  (i64.ge_s)
            body.push(0x72); // or
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // c
            body.push(LOCAL_GET);
            leb_u(5, &mut body); // cols
            body.push(0x59); // c >= cols
            body.push(0x72); // or
            body.push(IF);
            body.push(BT_VOID);
            body.push(0x00); // unreachable
            body.push(END);
            // out = __alloc_tensor(rows, cols)
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_WRAP_I64);
            body.push(CALL);
            leb_u(alloc_tensor_idx as u64, &mut body);
            body.push(LOCAL_SET);
            leb_u(7, &mut body); // out
            // n = rows*cols (i32)
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(0x7E); // i64.mul
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(6, &mut body); // n
            // copy data: i=0; while i<n { out[24+4i] = t[24+4i] } (raw 4-byte)
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(8, &mut body); // i
            body.push(0x03); // loop
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(8, &mut body);
            body.push(LOCAL_GET);
            leb_u(6, &mut body);
            body.push(0x48); // i32.lt_s
            body.push(IF);
            body.push(BT_VOID);
            // out + 4*i  (addr), then load t's f32 and store
            body.push(LOCAL_GET);
            leb_u(7, &mut body); // out
            body.push(LOCAL_GET);
            leb_u(8, &mut body);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // t
            body.push(LOCAL_GET);
            leb_u(8, &mut body);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            f32_load(TENSOR_HEADER, &mut body); // t[24+4i]
            f32_store(TENSOR_HEADER, &mut body); // out[24+4i]
            body.push(LOCAL_GET);
            leb_u(8, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(8, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body);
            body.push(END); // end if
            body.push(END); // end loop
            // write the element: out[24 + 4*(r*cols + c)] = (f32)v
            body.push(LOCAL_GET);
            leb_u(7, &mut body); // out
            // offset = 4 * (r*cols + c)
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // r
            body.push(LOCAL_GET);
            leb_u(5, &mut body); // cols
            body.push(0x7E); // i64.mul
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // c
            body.push(I64_ADD);
            body.push(I32_WRAP_I64);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD); // out + 4*idx
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // v (f64)
            body.push(0xB6); // f32.demote_f64 -> (f32)v
            f32_store(TENSOR_HEADER, &mut body);
            // drop t (this builtin consumes it)
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(CALL);
            leb_u(drop_tensor_idx as u64, &mut body);
            body.push(DROP);
            // return out
            body.push(LOCAL_GET);
            leb_u(7, &mut body);
            helper_entry(
                &[WType::I64, WType::I64, WType::I32, WType::I32, WType::I32],
                body,
            )
        }
        HeapHelper::TensorGet => {
            // (t:i32, r:i64, c:i64) -> f64. Trap OOB; else load f32 -> f64.
            // Consumes t (drops it after reading). Locals: rows(3,i64),
            // cols(4,i64), val(5,f64).
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(8, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // rows
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(16, &mut body);
            body.push(LOCAL_SET);
            leb_u(4, &mut body); // cols
            // bounds check (same as set)
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x53);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(I64_CONST);
            leb_s(0, &mut body);
            body.push(0x53);
            body.push(0x72);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(0x59);
            body.push(0x72);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(0x59);
            body.push(0x72);
            body.push(IF);
            body.push(BT_VOID);
            body.push(0x00); // unreachable
            body.push(END);
            // val = (f64) t[24 + 4*(r*cols + c)]
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // t
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // r
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // cols
            body.push(0x7E); // i64.mul
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // c
            body.push(I64_ADD);
            body.push(I32_WRAP_I64);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            f32_load(TENSOR_HEADER, &mut body);
            body.push(0xBB); // f64.promote_f32
            body.push(LOCAL_SET);
            leb_u(5, &mut body); // val
            // drop t
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(CALL);
            leb_u(drop_tensor_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(5, &mut body); // return val
            helper_entry(&[WType::I64, WType::I64, WType::F64], body)
        }
        HeapHelper::TensorRows | HeapHelper::TensorCols => {
            // (t:i32) -> i64. Load the header word, then consume (drop) t.
            // Local: out(1,i64).
            let off = if h == HeapHelper::TensorRows { 8 } else { 16 };
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(off, &mut body);
            body.push(LOCAL_SET);
            leb_u(1, &mut body); // out
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(CALL);
            leb_u(drop_tensor_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            helper_entry(&[WType::I64], body)
        }
        HeapHelper::Matmul => {
            // (a:i32, b:i32) -> ptr:i32. (m,k) x (k,n) -> (m,n). Trap on shape
            // mismatch (a.cols != b.rows). Accumulate in f32 to match the
            // interpreter's f32 kernel. Consumes a and b. Locals:
            // m(2,i32), k(3,i32), n(4,i32), out(5,i32), i(6,i32), j(7,i32),
            // p(8,i32), acc(9,f32).
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(8, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // m = a.rows
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(16, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // k = a.cols
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            i64_load(16, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(4, &mut body); // n = b.cols
            // shape check: a.cols (k) != b.rows -> trap
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // k
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            i64_load(8, &mut body);
            body.push(I32_WRAP_I64); // b.rows
            body.push(I32_NE);
            body.push(IF);
            body.push(BT_VOID);
            body.push(0x00); // unreachable
            body.push(END);
            // out = __alloc_tensor(m, n)
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(CALL);
            leb_u(alloc_tensor_idx as u64, &mut body);
            body.push(LOCAL_SET);
            leb_u(5, &mut body); // out
            // for i in 0..m
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(6, &mut body); // i
            body.push(0x03); // loop i
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(6, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // m
            body.push(0x48); // i32.lt_s
            body.push(IF);
            body.push(BT_VOID);
            // for j in 0..n
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(7, &mut body); // j
            body.push(0x03); // loop j
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(7, &mut body);
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // n
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            // acc = 0.0f32
            body.push(0x43);
            body.extend_from_slice(&0.0f32.to_le_bytes());
            body.push(LOCAL_SET);
            leb_u(9, &mut body); // acc
            // for p in 0..k
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(8, &mut body); // p
            body.push(0x03); // loop p
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(8, &mut body);
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // k
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            // acc += a[i*k+p] * b[p*n+j]
            body.push(LOCAL_GET);
            leb_u(9, &mut body); // acc
            // a element: a + 4*(i*k + p)
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // a
            body.push(LOCAL_GET);
            leb_u(6, &mut body); // i
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // k
            body.push(I32_MUL);
            body.push(LOCAL_GET);
            leb_u(8, &mut body); // p
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            f32_load(TENSOR_HEADER, &mut body); // a[i*k+p]
            // b element: b + 4*(p*n + j)
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // b
            body.push(LOCAL_GET);
            leb_u(8, &mut body); // p
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // n
            body.push(I32_MUL);
            body.push(LOCAL_GET);
            leb_u(7, &mut body); // j
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            f32_load(TENSOR_HEADER, &mut body); // b[p*n+j]
            body.push(0x94); // f32.mul
            body.push(0x92); // f32.add (acc + a*b)
            body.push(LOCAL_SET);
            leb_u(9, &mut body); // acc
            // p++
            body.push(LOCAL_GET);
            leb_u(8, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(8, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body); // continue loop p
            body.push(END); // end if p<k
            body.push(END); // end loop p
            // out[i*n+j] = acc
            body.push(LOCAL_GET);
            leb_u(5, &mut body); // out
            body.push(LOCAL_GET);
            leb_u(6, &mut body); // i
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // n
            body.push(I32_MUL);
            body.push(LOCAL_GET);
            leb_u(7, &mut body); // j
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            body.push(LOCAL_GET);
            leb_u(9, &mut body); // acc
            f32_store(TENSOR_HEADER, &mut body);
            // j++
            body.push(LOCAL_GET);
            leb_u(7, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(7, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body); // continue loop j
            body.push(END); // end if j<n
            body.push(END); // end loop j
            // i++
            body.push(LOCAL_GET);
            leb_u(6, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(6, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body); // continue loop i
            body.push(END); // end if i<m
            body.push(END); // end loop i
            // drop a, drop b
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(CALL);
            leb_u(drop_tensor_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(CALL);
            leb_u(drop_tensor_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(5, &mut body); // return out
            helper_entry(
                &[
                    WType::I32, // m (2)
                    WType::I32, // k (3)
                    WType::I32, // n (4)
                    WType::I32, // out (5)
                    WType::I32, // i (6)
                    WType::I32, // j (7)
                    WType::I32, // p (8)
                    WType::F32, // acc (9)
                ],
                body,
            )
        }
        HeapHelper::Transpose => {
            // (t:i32) -> ptr:i32. out(n,m); out[j*m+i] = t[i*n+j]. Consumes t.
            // Locals: m(1,i32), n(2,i32), out(3,i32), i(4,i32), j(5,i32).
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(8, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(1, &mut body); // m
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(16, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // n
            // out = __alloc_tensor(n, m)
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(CALL);
            leb_u(alloc_tensor_idx as u64, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // out
            // for i in 0..m
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(4, &mut body);
            body.push(0x03);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // m
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            // for j in 0..n
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x03);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // n
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            // out[j*m+i] = t[i*n+j]
            body.push(LOCAL_GET);
            leb_u(3, &mut body); // out
            body.push(LOCAL_GET);
            leb_u(5, &mut body); // j
            body.push(LOCAL_GET);
            leb_u(1, &mut body); // m
            body.push(I32_MUL);
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // i
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            body.push(LOCAL_GET);
            leb_u(0, &mut body); // t
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // i
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // n
            body.push(I32_MUL);
            body.push(LOCAL_GET);
            leb_u(5, &mut body); // j
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            f32_load(TENSOR_HEADER, &mut body);
            f32_store(TENSOR_HEADER, &mut body);
            // j++
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body);
            body.push(END);
            body.push(END); // end loop j
            // i++
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(4, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body);
            body.push(END);
            body.push(END); // end loop i
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(CALL);
            leb_u(drop_tensor_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            helper_entry(
                &[WType::I32, WType::I32, WType::I32, WType::I32, WType::I32],
                body,
            )
        }
        HeapHelper::Relu => {
            // (t:i32) -> ptr:i32. out[i] = max(t[i], 0) per the interpreter's
            // `if x > 0 { x } else { 0 }`. Consumes t. Locals: n(1,i32),
            // out(2,i32), i(3,i32), x(4,f32).
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(8, &mut body);
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(16, &mut body);
            body.push(0x7E); // i64.mul (rows*cols)
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(1, &mut body); // n
            // out = __alloc_tensor(t.rows, t.cols)
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(8, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(16, &mut body);
            body.push(I32_WRAP_I64);
            body.push(CALL);
            leb_u(alloc_tensor_idx as u64, &mut body);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // out
            // i = 0; while i < n
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body);
            body.push(0x03);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            // x = t[24+4i]
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            f32_load(TENSOR_HEADER, &mut body);
            body.push(LOCAL_SET);
            leb_u(4, &mut body); // x
            // out[24+4i] = (x > 0) ? x : 0
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // out
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            // value via if (x > 0.0)
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // x
            body.push(0x43);
            body.extend_from_slice(&0.0f32.to_le_bytes());
            body.push(0x5E); // f32.gt
            body.push(IF);
            body.push(0x7D); // blocktype = f32
            body.push(LOCAL_GET);
            leb_u(4, &mut body); // x
            body.push(ELSE);
            body.push(0x43);
            body.extend_from_slice(&0.0f32.to_le_bytes());
            body.push(END);
            f32_store(TENSOR_HEADER, &mut body);
            // i++
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(3, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body);
            body.push(END);
            body.push(END); // end loop
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(CALL);
            leb_u(drop_tensor_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            helper_entry(&[WType::I32, WType::I32, WType::I32, WType::F32], body)
        }
        HeapHelper::Softmax => {
            // (t:i32) -> ptr:i32. Row-wise, numerically stable, matching
            // `softmax_rows`: per row compute max; e = exp(x-max) via env.exp on
            // the f64-promoted (x-max), demoted back to f32 (mirroring the f32
            // kernel's per-element rounding); accumulate sum; then multiply each
            // by 1/sum (or 0 when sum<=0). Consumes t. Locals:
            //   m(1,i32) n(2,i32) out(3,i32) i(4,i32) j(5,i32) mx(6,f32)
            //   sum(7,f32) e(8,f32) base(9,i32) addr(10,i32)
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(8, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(1, &mut body); // m
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            i64_load(16, &mut body);
            body.push(I32_WRAP_I64);
            body.push(LOCAL_SET);
            leb_u(2, &mut body); // n
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(CALL);
            leb_u(alloc_tensor_idx as u64, &mut body);
            body.push(LOCAL_SET);
            leb_u(3, &mut body); // out
            // for i in 0..m
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(4, &mut body);
            body.push(0x03);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            // base = i*n
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(I32_MUL);
            body.push(LOCAL_SET);
            leb_u(9, &mut body);
            // mx = -inf
            body.push(0x43);
            body.extend_from_slice(&f32::NEG_INFINITY.to_le_bytes());
            body.push(LOCAL_SET);
            leb_u(6, &mut body);
            // for j: mx = f32.max(mx, t[base+j])
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x03);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(6, &mut body);
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(9, &mut body);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            f32_load(TENSOR_HEADER, &mut body);
            body.push(0x97); // f32.max
            body.push(LOCAL_SET);
            leb_u(6, &mut body);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body);
            body.push(END);
            body.push(END); // end max loop
            // sum = 0
            body.push(0x43);
            body.extend_from_slice(&0.0f32.to_le_bytes());
            body.push(LOCAL_SET);
            leb_u(7, &mut body);
            // for j: e = demote(exp(promote(t[base+j]-mx))); out[base+j]=e; sum+=e
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x03);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            // e = demote(exp(promote( t[base+j] - mx )))
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(LOCAL_GET);
            leb_u(9, &mut body);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            f32_load(TENSOR_HEADER, &mut body);
            body.push(LOCAL_GET);
            leb_u(6, &mut body); // mx
            body.push(0x93); // f32.sub
            body.push(0xBB); // f64.promote_f32
            body.push(CALL);
            leb_u(exp_idx as u64, &mut body);
            body.push(0xB6); // f32.demote_f64
            body.push(LOCAL_SET);
            leb_u(8, &mut body); // e
            // out[base+j] = e
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(LOCAL_GET);
            leb_u(9, &mut body);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            body.push(LOCAL_GET);
            leb_u(8, &mut body);
            f32_store(TENSOR_HEADER, &mut body);
            // sum += e
            body.push(LOCAL_GET);
            leb_u(7, &mut body);
            body.push(LOCAL_GET);
            leb_u(8, &mut body);
            body.push(0x92);
            body.push(LOCAL_SET);
            leb_u(7, &mut body);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body);
            body.push(END);
            body.push(END); // end exp loop
            // inv = (sum > 0) ? 1/sum : 0   (store into local 6 = mx, now free)
            body.push(LOCAL_GET);
            leb_u(7, &mut body);
            body.push(0x43);
            body.extend_from_slice(&0.0f32.to_le_bytes());
            body.push(0x5E); // f32.gt
            body.push(IF);
            body.push(0x7D);
            body.push(0x43);
            body.extend_from_slice(&1.0f32.to_le_bytes());
            body.push(LOCAL_GET);
            leb_u(7, &mut body);
            body.push(0x95); // f32.div
            body.push(ELSE);
            body.push(0x43);
            body.extend_from_slice(&0.0f32.to_le_bytes());
            body.push(END);
            body.push(LOCAL_SET);
            leb_u(6, &mut body); // inv
            // for j: out[base+j] *= inv
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x03);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            // addr = out + 4*(base+j)
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(LOCAL_GET);
            leb_u(9, &mut body);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_ADD);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(10, &mut body); // addr
            body.push(LOCAL_GET);
            leb_u(10, &mut body);
            body.push(LOCAL_GET);
            leb_u(10, &mut body);
            f32_load(TENSOR_HEADER, &mut body);
            body.push(LOCAL_GET);
            leb_u(6, &mut body); // inv
            body.push(0x94); // f32.mul
            f32_store(TENSOR_HEADER, &mut body);
            // j++
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body);
            body.push(END);
            body.push(END); // end scale loop
            // i++
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(4, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body);
            body.push(END);
            body.push(END); // end row loop
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(CALL);
            leb_u(drop_tensor_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            helper_entry(
                &[
                    WType::I32, // m
                    WType::I32, // n
                    WType::I32, // out
                    WType::I32, // i
                    WType::I32, // j
                    WType::F32, // mx / inv
                    WType::F32, // sum
                    WType::F32, // e
                    WType::I32, // base
                    WType::I32, // addr
                ],
                body,
            )
        }
        HeapHelper::EmbedSim => {
            // (a:i32, b:i32) -> f64. cosine(hash_embed(a,64), hash_embed(b,64)).
            // Compute both embeddings into the fixed EMBED_SCRATCH region (two
            // dim-64 f32 vectors), then cosine. Consumes a and b. Locals:
            // dot(2,f32), na(3,f32), nb(4,f32), i(5,i32), va(6,f32), vb(7,f32),
            // denom(8,f32).
            let va_addr = embed_scratch;
            let vb_addr = embed_scratch + 64 * 4;
            // hash_embed(a, va_addr) ; hash_embed(b, vb_addr)
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(I32_CONST);
            leb_s(va_addr as i64, &mut body);
            body.push(CALL);
            leb_u(hash_embed_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(I32_CONST);
            leb_s(vb_addr as i64, &mut body);
            body.push(CALL);
            leb_u(hash_embed_idx as u64, &mut body);
            body.push(DROP);
            // dot=na=nb=0
            for l in [2u32, 3, 4] {
                body.push(0x43);
                body.extend_from_slice(&0.0f32.to_le_bytes());
                body.push(LOCAL_SET);
                leb_u(l as u64, &mut body);
            }
            // i = 0; while i < 64
            body.push(I32_CONST);
            leb_s(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x03);
            body.push(BT_VOID);
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(64, &mut body);
            body.push(0x48);
            body.push(IF);
            body.push(BT_VOID);
            // va = va_addr[i] ; vb = vb_addr[i]
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_CONST);
            leb_s(va_addr as i64, &mut body);
            body.push(I32_ADD);
            f32_load(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(6, &mut body); // va
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(4, &mut body);
            body.push(I32_MUL);
            body.push(I32_CONST);
            leb_s(vb_addr as i64, &mut body);
            body.push(I32_ADD);
            f32_load(0, &mut body);
            body.push(LOCAL_SET);
            leb_u(7, &mut body); // vb
            // dot += va*vb
            body.push(LOCAL_GET);
            leb_u(2, &mut body);
            body.push(LOCAL_GET);
            leb_u(6, &mut body);
            body.push(LOCAL_GET);
            leb_u(7, &mut body);
            body.push(0x94); // f32.mul
            body.push(0x92); // f32.add
            body.push(LOCAL_SET);
            leb_u(2, &mut body);
            // na += va*va
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(LOCAL_GET);
            leb_u(6, &mut body);
            body.push(LOCAL_GET);
            leb_u(6, &mut body);
            body.push(0x94);
            body.push(0x92);
            body.push(LOCAL_SET);
            leb_u(3, &mut body);
            // nb += vb*vb
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(LOCAL_GET);
            leb_u(7, &mut body);
            body.push(LOCAL_GET);
            leb_u(7, &mut body);
            body.push(0x94);
            body.push(0x92);
            body.push(LOCAL_SET);
            leb_u(4, &mut body);
            // i++
            body.push(LOCAL_GET);
            leb_u(5, &mut body);
            body.push(I32_CONST);
            leb_s(1, &mut body);
            body.push(I32_ADD);
            body.push(LOCAL_SET);
            leb_u(5, &mut body);
            body.push(0x0C);
            leb_u(1, &mut body);
            body.push(END);
            body.push(END); // end loop
            // denom = sqrt(na) * sqrt(nb)
            body.push(LOCAL_GET);
            leb_u(3, &mut body);
            body.push(0x91); // f32.sqrt
            body.push(LOCAL_GET);
            leb_u(4, &mut body);
            body.push(0x91); // f32.sqrt
            body.push(0x94); // f32.mul
            body.push(LOCAL_SET);
            leb_u(8, &mut body); // denom
            // drop a, drop b (consumes the Strings)
            body.push(LOCAL_GET);
            leb_u(0, &mut body);
            body.push(CALL);
            leb_u(drop_str_idx as u64, &mut body);
            body.push(DROP);
            body.push(LOCAL_GET);
            leb_u(1, &mut body);
            body.push(CALL);
            leb_u(drop_str_idx as u64, &mut body);
            body.push(DROP);
            // result = (denom == 0) ? 0.0 : dot/denom ; promoted to f64
            body.push(LOCAL_GET);
            leb_u(8, &mut body);
            body.push(0x43);
            body.extend_from_slice(&0.0f32.to_le_bytes());
            body.push(0x5B); // f32.eq
            body.push(IF);
            body.push(0x7C); // f64 result
            body.push(0x44); // f64.const 0.0
            body.extend_from_slice(&0.0f64.to_le_bytes());
            body.push(ELSE);
            body.push(LOCAL_GET);
            leb_u(2, &mut body); // dot
            body.push(LOCAL_GET);
            leb_u(8, &mut body); // denom
            body.push(0x95); // f32.div
            body.push(0xBB); // f64.promote_f32
            body.push(END);
            helper_entry(
                &[
                    WType::F32,
                    WType::F32,
                    WType::F32,
                    WType::I32,
                    WType::F32,
                    WType::F32,
                    WType::F32,
                ],
                body,
            )
        }
        HeapHelper::HashEmbed => {
            // (s:i32, vec_addr:i32) -> i32(dummy). Build a dim-64 f32 embedding
            // at [vec_addr..vec_addr+256], matching `rag::hash_embed(text, 64)`:
            //   zero the 64 buckets; for each whitespace-delimited token,
            //   lowercase (ASCII) each byte, FNV-1a hash the token, bucket =
            //   h % 64, sign = (h>>63)&1 ? -1 : +1, v[bucket] += sign; then
            //   L2-normalize. NOTE: rag uses `str::to_lowercase` (Unicode);
            //   we replicate ASCII lowercasing, which matches for ASCII inputs
            //   (the differential tests + demo use ASCII).
            // Locals: len(2,i32), i(3,i32), hash(4,i64), tokstart(5,i32),
            //   byte(6,i32), bucket(4?)... use: norm(7,f32), j(8,i32),
            //   sign(9,f32), bytep(10,i32).
            emit_hash_embed_body(&mut body);
            helper_entry(
                &[
                    WType::I32, // len (2)
                    WType::I32, // i (3)
                    WType::I64, // hash (4)
                    WType::I32, // in_token (5)
                    WType::I32, // byte (6)
                    WType::F32, // norm (7)
                    WType::I32, // bucket (8)
                    WType::F32, // sign (9)
                ],
                body,
            )
        }
    }
}

/// Emit a byte-copy loop: `while i < <len_local> { dst[16 + i] = src[16 + i]; i++ }`,
/// where `src`/`dst` are i32 locals holding String pointers, `len_local` the
/// byte count, and `i_local` a scratch i32 counter. `_dst_off` is unused (kept
/// for signature symmetry). Copies into dst's byte region starting at index 0.
fn emit_byte_copy_loop(
    body: &mut Vec<u8>,
    src: u32,
    dst: u32,
    len_local: u32,
    _dst_off: u32,
    i_local: u32,
) {
    // i = 0
    body.push(0x41);
    leb_s(0, body);
    body.push(0x21);
    leb_u(i_local as u64, body);
    body.push(0x03); // loop
    body.push(0x40);
    // if i < len
    body.push(0x20);
    leb_u(i_local as u64, body);
    body.push(0x20);
    leb_u(len_local as u64, body);
    body.push(0x49); // i32.lt_u
    body.push(0x04); // if
    body.push(0x40);
    // dst + i
    body.push(0x20);
    leb_u(dst as u64, body);
    body.push(0x20);
    leb_u(i_local as u64, body);
    body.push(0x6A); // i32.add
    // src byte [src + i + 16]
    body.push(0x20);
    leb_u(src as u64, body);
    body.push(0x20);
    leb_u(i_local as u64, body);
    body.push(0x6A);
    i32_load8_u(STR_HEADER, body);
    i32_store8(STR_HEADER, body); // [dst + i + 16]
    // i++
    body.push(0x20);
    leb_u(i_local as u64, body);
    body.push(0x41);
    leb_s(1, body);
    body.push(0x6A);
    body.push(0x21);
    leb_u(i_local as u64, body);
    body.push(0x0C); // br loop
    leb_u(1, body);
    body.push(0x0B); // end if
    body.push(0x0B); // end loop
}

/// Like `emit_byte_copy_loop` but the destination index is offset by the value
/// of `dst_extra_local` (e.g. the length already written): copies
/// `src[16+i]` -> `dst[16 + dst_extra + i]` for i in 0..len.
fn emit_byte_copy_loop_offset(
    body: &mut Vec<u8>,
    src: u32,
    dst: u32,
    len_local: u32,
    dst_extra_local: u32,
    i_local: u32,
) {
    // i = 0
    body.push(0x41);
    leb_s(0, body);
    body.push(0x21);
    leb_u(i_local as u64, body);
    body.push(0x03); // loop
    body.push(0x40);
    body.push(0x20);
    leb_u(i_local as u64, body);
    body.push(0x20);
    leb_u(len_local as u64, body);
    body.push(0x49); // i32.lt_u
    body.push(0x04); // if
    body.push(0x40);
    // dst + dst_extra + i  (address; +16 folded into store8 offset)
    body.push(0x20);
    leb_u(dst as u64, body);
    body.push(0x20);
    leb_u(dst_extra_local as u64, body);
    body.push(0x6A); // i32.add
    body.push(0x20);
    leb_u(i_local as u64, body);
    body.push(0x6A); // + i
    // src byte
    body.push(0x20);
    leb_u(src as u64, body);
    body.push(0x20);
    leb_u(i_local as u64, body);
    body.push(0x6A);
    i32_load8_u(STR_HEADER, body);
    i32_store8(STR_HEADER, body);
    // i++
    body.push(0x20);
    leb_u(i_local as u64, body);
    body.push(0x41);
    leb_s(1, body);
    body.push(0x6A);
    body.push(0x21);
    leb_u(i_local as u64, body);
    body.push(0x0C);
    leb_u(1, body);
    body.push(0x0B); // end if
    body.push(0x0B); // end loop
}

/// Emit the body of the `__hash_embed(s:i32, vec_addr:i32) -> i32` helper,
/// matching `crate::rag::hash_embed(text, 64)` for ASCII input. Params: s(0),
/// vec_addr(1). Locals: len(2,i32), i(3,i32), hash(4,i64), in_token(5,i32),
/// byte(6,i32), norm(7,f32), bucket(8,i32), sign(9,f32).
///
/// We stream FNV-1a over each whitespace-delimited, ASCII-lowercased token (no
/// buffering): on a token boundary we finalize `hash` into a bucket/sign and
/// accumulate; at end-of-string we finalize a trailing token. Then L2-normalize.
fn emit_hash_embed_body(body: &mut Vec<u8>) {
    const LOCAL_GET: u8 = 0x20;
    const LOCAL_SET: u8 = 0x21;
    const I32_CONST: u8 = 0x41;
    const I64_CONST: u8 = 0x42;
    const I32_ADD: u8 = 0x6A;
    const I32_MUL: u8 = 0x6C;
    const IF: u8 = 0x04;
    const ELSE: u8 = 0x05;
    const END: u8 = 0x0B;
    const BT_VOID: u8 = 0x40;
    const FNV_OFFSET: i64 = 0xcbf2_9ce4_8422_2325u64 as i64;
    const FNV_PRIME: i64 = 0x0000_0100_0000_01b3;

    // Append the "finalize current token" sequence: bucket = hash % 64;
    // sign = (hash < 0) ? -1 : +1; vec[bucket] += sign; in_token = 0.
    let finalize = |body: &mut Vec<u8>| {
        // bucket = (i32) (hash % 64)   (unsigned remainder)
        body.push(LOCAL_GET);
        leb_u(4, body); // hash
        body.push(I64_CONST);
        leb_s(64, body);
        body.push(0x82); // i64.rem_u
        body.push(0xA7); // i32.wrap_i64
        body.push(LOCAL_SET);
        leb_u(8, body); // bucket
        // sign = (hash < 0) ? -1.0 : 1.0
        body.push(LOCAL_GET);
        leb_u(4, body); // hash
        body.push(I64_CONST);
        leb_s(0, body);
        body.push(0x53); // i64.lt_s  (top bit set)
        body.push(IF);
        body.push(0x7D); // f32 result
        body.push(0x43);
        body.extend_from_slice(&(-1.0f32).to_le_bytes());
        body.push(ELSE);
        body.push(0x43);
        body.extend_from_slice(&1.0f32.to_le_bytes());
        body.push(END);
        body.push(LOCAL_SET);
        leb_u(9, body); // sign
        // addr = vec_addr + 4*bucket ; vec[bucket] += sign
        body.push(LOCAL_GET);
        leb_u(1, body);
        body.push(LOCAL_GET);
        leb_u(8, body);
        body.push(I32_CONST);
        leb_s(4, body);
        body.push(I32_MUL);
        body.push(I32_ADD); // store address
        body.push(LOCAL_GET);
        leb_u(1, body);
        body.push(LOCAL_GET);
        leb_u(8, body);
        body.push(I32_CONST);
        leb_s(4, body);
        body.push(I32_MUL);
        body.push(I32_ADD);
        f32_load(0, body);
        body.push(LOCAL_GET);
        leb_u(9, body); // sign
        body.push(0x92); // f32.add
        f32_store(0, body);
        // in_token = 0
        body.push(I32_CONST);
        leb_s(0, body);
        body.push(LOCAL_SET);
        leb_u(5, body);
    };

    // 1. Zero the 64 buckets.
    body.push(I32_CONST);
    leb_s(0, body);
    body.push(LOCAL_SET);
    leb_u(3, body); // i
    body.push(0x03);
    body.push(BT_VOID);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(64, body);
    body.push(0x48);
    body.push(IF);
    body.push(BT_VOID);
    body.push(LOCAL_GET);
    leb_u(1, body);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(4, body);
    body.push(I32_MUL);
    body.push(I32_ADD);
    body.push(0x43);
    body.extend_from_slice(&0.0f32.to_le_bytes());
    f32_store(0, body);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(1, body);
    body.push(I32_ADD);
    body.push(LOCAL_SET);
    leb_u(3, body);
    body.push(0x0C);
    leb_u(1, body);
    body.push(END);
    body.push(END);

    // 2. len = s.len ; in_token = 0
    body.push(LOCAL_GET);
    leb_u(0, body);
    i64_load(8, body);
    body.push(0xA7); // i32.wrap_i64
    body.push(LOCAL_SET);
    leb_u(2, body); // len
    body.push(I32_CONST);
    leb_s(0, body);
    body.push(LOCAL_SET);
    leb_u(5, body); // in_token

    // 3. scan
    body.push(I32_CONST);
    leb_s(0, body);
    body.push(LOCAL_SET);
    leb_u(3, body); // i
    body.push(0x03);
    body.push(BT_VOID);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(LOCAL_GET);
    leb_u(2, body);
    body.push(0x48);
    body.push(IF);
    body.push(BT_VOID);
    // byte = s[16 + i]
    body.push(LOCAL_GET);
    leb_u(0, body);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_ADD);
    i32_load8_u(STR_HEADER, body);
    body.push(LOCAL_SET);
    leb_u(6, body); // byte
    // is_ws = (byte==0x20) || (0x09<=byte<=0x0d)
    body.push(LOCAL_GET);
    leb_u(6, body);
    body.push(I32_CONST);
    leb_s(0x20, body);
    body.push(0x46); // i32.eq
    body.push(LOCAL_GET);
    leb_u(6, body);
    body.push(I32_CONST);
    leb_s(0x09, body);
    body.push(0x4E); // i32.ge_s
    body.push(LOCAL_GET);
    leb_u(6, body);
    body.push(I32_CONST);
    leb_s(0x0d, body);
    body.push(0x4C); // i32.le_s
    body.push(0x71); // and
    body.push(0x72); // or
    body.push(IF);
    body.push(BT_VOID);
    // whitespace: if in_token finalize
    body.push(LOCAL_GET);
    leb_u(5, body);
    body.push(IF);
    body.push(BT_VOID);
    finalize(body);
    body.push(END);
    body.push(ELSE);
    // non-whitespace: start token if needed
    body.push(LOCAL_GET);
    leb_u(5, body);
    body.push(0x45); // i32.eqz
    body.push(IF);
    body.push(BT_VOID);
    body.push(I64_CONST);
    leb_s(FNV_OFFSET, body);
    body.push(LOCAL_SET);
    leb_u(4, body); // hash
    body.push(I32_CONST);
    leb_s(1, body);
    body.push(LOCAL_SET);
    leb_u(5, body); // in_token = 1
    body.push(END);
    // lowercase ASCII
    body.push(LOCAL_GET);
    leb_u(6, body);
    body.push(I32_CONST);
    leb_s('A' as i64, body);
    body.push(0x4E); // ge_s
    body.push(LOCAL_GET);
    leb_u(6, body);
    body.push(I32_CONST);
    leb_s('Z' as i64, body);
    body.push(0x4C); // le_s
    body.push(0x71); // and
    body.push(IF);
    body.push(BT_VOID);
    body.push(LOCAL_GET);
    leb_u(6, body);
    body.push(I32_CONST);
    leb_s(32, body);
    body.push(I32_ADD);
    body.push(LOCAL_SET);
    leb_u(6, body);
    body.push(END);
    // hash = (hash ^ lc) * FNV_PRIME
    body.push(LOCAL_GET);
    leb_u(4, body);
    body.push(LOCAL_GET);
    leb_u(6, body);
    body.push(0xAD); // i64.extend_i32_u
    body.push(0x85); // i64.xor
    body.push(I64_CONST);
    leb_s(FNV_PRIME, body);
    body.push(0x7E); // i64.mul
    body.push(LOCAL_SET);
    leb_u(4, body);
    body.push(END); // end if-ws/else
    // i++
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(1, body);
    body.push(I32_ADD);
    body.push(LOCAL_SET);
    leb_u(3, body);
    body.push(0x0C);
    leb_u(1, body);
    body.push(END);
    body.push(END); // end scan loop

    // 4. trailing token
    body.push(LOCAL_GET);
    leb_u(5, body);
    body.push(IF);
    body.push(BT_VOID);
    finalize(body);
    body.push(END);

    // 5. L2 normalize. sum = sum vec[i]^2
    body.push(0x43);
    body.extend_from_slice(&0.0f32.to_le_bytes());
    body.push(LOCAL_SET);
    leb_u(7, body); // norm (sum)
    body.push(I32_CONST);
    leb_s(0, body);
    body.push(LOCAL_SET);
    leb_u(3, body);
    body.push(0x03);
    body.push(BT_VOID);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(64, body);
    body.push(0x48);
    body.push(IF);
    body.push(BT_VOID);
    body.push(LOCAL_GET);
    leb_u(7, body);
    body.push(LOCAL_GET);
    leb_u(1, body);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(4, body);
    body.push(I32_MUL);
    body.push(I32_ADD);
    f32_load(0, body);
    body.push(LOCAL_GET);
    leb_u(1, body);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(4, body);
    body.push(I32_MUL);
    body.push(I32_ADD);
    f32_load(0, body);
    body.push(0x94); // f32.mul
    body.push(0x92); // f32.add
    body.push(LOCAL_SET);
    leb_u(7, body);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(1, body);
    body.push(I32_ADD);
    body.push(LOCAL_SET);
    leb_u(3, body);
    body.push(0x0C);
    leb_u(1, body);
    body.push(END);
    body.push(END);
    // norm = sqrt(sum)
    body.push(LOCAL_GET);
    leb_u(7, body);
    body.push(0x91); // f32.sqrt
    body.push(LOCAL_SET);
    leb_u(7, body);
    // if norm > 0 divide
    body.push(LOCAL_GET);
    leb_u(7, body);
    body.push(0x43);
    body.extend_from_slice(&0.0f32.to_le_bytes());
    body.push(0x5E); // f32.gt
    body.push(IF);
    body.push(BT_VOID);
    body.push(I32_CONST);
    leb_s(0, body);
    body.push(LOCAL_SET);
    leb_u(3, body);
    body.push(0x03);
    body.push(BT_VOID);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(64, body);
    body.push(0x48);
    body.push(IF);
    body.push(BT_VOID);
    body.push(LOCAL_GET);
    leb_u(1, body);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(4, body);
    body.push(I32_MUL);
    body.push(I32_ADD);
    body.push(LOCAL_SET);
    leb_u(8, body); // addr (reuse bucket local)
    body.push(LOCAL_GET);
    leb_u(8, body);
    body.push(LOCAL_GET);
    leb_u(8, body);
    f32_load(0, body);
    body.push(LOCAL_GET);
    leb_u(7, body);
    body.push(0x95); // f32.div
    f32_store(0, body);
    body.push(LOCAL_GET);
    leb_u(3, body);
    body.push(I32_CONST);
    leb_s(1, body);
    body.push(I32_ADD);
    body.push(LOCAL_SET);
    leb_u(3, body);
    body.push(0x0C);
    leb_u(1, body);
    body.push(END);
    body.push(END);
    body.push(END); // end if norm>0

    // dummy return 0
    body.push(I32_CONST);
    leb_s(0, body);
}

/// Constructor infos sorted by tag (deterministic codegen for `__drop`).
fn ctor_infos_sorted(ctors: &CtorTable) -> Vec<CtorInfo> {
    let mut v: Vec<CtorInfo> = ctors.by_name.values().cloned().collect();
    v.sort_by_key(|c| c.tag);
    v
}

/// Key under which a function's index is stored in `LocalEnv::index`. Prefixed
/// to avoid colliding with a same-named local variable.
fn fn_index_key(name: &str) -> String {
    format!("\u{1}fn:{}", name)
}

// ---- string-literal pool -------------------------------------------------

/// Walk an `IExpr` and collect every distinct `Atom::Str` literal's bytes.
fn collect_str_lits_iexpr(e: &IExpr, out: &mut Vec<Vec<u8>>) {
    match e {
        IExpr::Ret(a) => collect_str_lits_atom(a, out),
        IExpr::Let(_, b, body) => {
            collect_str_lits_bind(b, out);
            collect_str_lits_iexpr(body, out);
        }
        IExpr::Dup(_, body) | IExpr::Drop(_, body) | IExpr::DropReuse(_, _, body) => {
            collect_str_lits_iexpr(body, out)
        }
        IExpr::TailCall(args) => {
            for a in args {
                collect_str_lits_atom(a, out);
            }
        }
    }
}

fn collect_str_lits_bind(b: &Bind, out: &mut Vec<Vec<u8>>) {
    match b {
        Bind::Atom(a) | Bind::Unary(_, a) => collect_str_lits_atom(a, out),
        Bind::MakeClosure(_, atoms) => {
            for a in atoms {
                collect_str_lits_atom(a, out);
            }
        }
        Bind::ApplyClosure(callee, atoms, _) => {
            collect_str_lits_atom(callee, out);
            for a in atoms {
                collect_str_lits_atom(a, out);
            }
        }
        Bind::Prim(_, l, r) => {
            collect_str_lits_atom(l, out);
            collect_str_lits_atom(r, out);
        }
        Bind::Ctor(_, args) | Bind::Call(_, args) | Bind::CtorReuse(_, _, args) => {
            for a in args {
                collect_str_lits_atom(a, out);
            }
        }
        Bind::If(c, then, els) => {
            collect_str_lits_atom(c, out);
            collect_str_lits_iexpr(then, out);
            collect_str_lits_iexpr(els, out);
        }
        Bind::Match(scrut, arms) => {
            collect_str_lits_atom(scrut, out);
            for arm in arms {
                collect_str_lits_iexpr(&arm.body, out);
            }
        }
    }
}

fn collect_str_lits_atom(a: &Atom, out: &mut Vec<Vec<u8>>) {
    if let Atom::Str(s) = a {
        let bytes = s.as_bytes().to_vec();
        if !out.contains(&bytes) {
            out.push(bytes);
        }
    }
}

// ---- top-level driver ----------------------------------------------------

/// Compile a type-checked `Program` to a WebAssembly binary (subset 2a).
/// Returns a clean `Err` for any feature outside the subset; never panics.
pub fn compile(program: &Program) -> Result<Vec<u8>, String> {
    // 0. Monomorphization (wasm-backend-only pre-pass): specialize every
    //    generic function and ADT reachable from `main` per the concrete type
    //    arguments actually used, producing a type-variable-free program. A
    //    program with no generics passes through unchanged. The rest of the
    //    backend then sees only concrete Int/Bool/Float/String/ADT types.
    let mono = crate::monomorphize::monomorphize(program)?;
    let program = &mono;

    // 1. Collect function signatures from the typed AST, in declaration order,
    //    assigning deterministic wasm function indices.
    // We import five host functions: `env.print_str` (index 0),
    // `env.print_float` (index 1), `env.print_int` (index 2),
    // `env.print_bool` (index 3), and `env.exp` (index 4, a libm-faithful
    // `exp(f64)->f64` used by Tensor softmax). Every DEFINED function index is
    // offset by N_IMPORTS.
    const N_IMPORTS: u32 = 5;
    const PRINT_STR_IDX: u32 = 0;
    const PRINT_FLOAT_IDX: u32 = 1;
    const PRINT_INT_IDX: u32 = 2;
    const PRINT_BOOL_IDX: u32 = 3;
    const EXP_IDX: u32 = 4;

    let mut sigs: HashMap<String, FnSig> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut fn_index: HashMap<String, u32> = HashMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            let params = f
                .params
                .iter()
                .map(|p| WType::from_ty(&p.ty))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("fn `{}`: {}", f.name, e))?;
            let ret = WType::from_ty(&f.ret).map_err(|e| format!("fn `{}`: {}", f.name, e))?;
            // The wasm function index = N_IMPORTS + declaration ordinal.
            let idx = N_IMPORTS + order.len() as u32;
            fn_index.insert(f.name.clone(), idx);
            order.push(f.name.clone());
            sigs.insert(f.name.clone(), FnSig { params, ret });
        }
    }

    if !sigs.contains_key("main") {
        return Err("wasm backend: no `main` function".into());
    }
    {
        let m = &sigs["main"];
        // `main` takes no params and returns Int (existing), Float (an f64
        // export decodes to a JS number), or a heap String (Phase 2d); the Node
        // runner decodes the result accordingly.
        if !m.params.is_empty()
            || (m.ret != WType::I64 && m.ret != WType::F64 && m.ret != WType::Str)
        {
            return Err(
                "wasm backend: `main` must take no params and return Int, Float, or String".into(),
            );
        }
    }

    // 1b. Build the ADT layout table (constructor tags + field wasm types).
    //     Rejects generic types / out-of-subset field types with a clean Err.
    let ctors = CtorTable::build(program)?;

    // 2. Lower the whole program to ANF IR, then insert reference-count
    //    operations (Phase 2b: the compiled module manages a real heap, so it
    //    needs the dup/drop the IR interpreter relies on for garbage-freeness).
    let lowered: HashMap<String, IFn> = ir::lower_program(program)?;
    // In-place cell reuse (FBIP): the rc pass opportunistically emits
    // `DropReuse`/`CtorReuse` pairs (always same-arity ctors), and codegen now
    // exploits them — a `match` that consumes a UNIQUE cell and rebuilds a
    // same-arity constructor reuses that cell's memory in place, matching the IR
    // interpreter. A shared cell yields a null token, so `CtorReuse` falls back
    // to a fresh allocation; either way the heap stays garbage-free.
    let rcd: HashMap<String, IFn> = crate::rc::insert_rc(&lowered);
    // Self-tail-call elimination: rewrite each self-tail-recursive function into
    // a loop (`TailCall` back-edges -> a wasm `loop` + `br`). Runs after rc so
    // ownership of the new args transfers to the params as a real call's binding.
    let fns: HashMap<String, IFn> = ir::tail_call_optimize(rcd);

    // 2c. Closure (lifted-lambda) table. Every lowered function carrying a
    //     `lam_sig` is a lifted lambda, emitted as a wasm function with the
    //     closure calling convention `(i32 closure, params...) -> ret` and
    //     dispatched via `call_indirect`. Lambda wasm functions are placed right
    //     after the user functions (so their indices stay deterministic) and
    //     before the runtime helpers. A defined function's wasm type index equals
    //     its function index, so each lambda's signature doubles as the
    //     `call_indirect` type for its machine-signature.
    let mut lam_names: Vec<String> =
        fns.iter().filter(|(_, f)| f.lam_sig.is_some()).map(|(n, _)| n.clone()).collect();
    lam_names.sort();
    let closure_base = ctors.by_name.len() as i64;
    let mut closure_tags: HashMap<String, i64> = HashMap::new();
    let mut sig_typeidx: HashMap<(Vec<WType>, WType), u32> = HashMap::new();
    // Per-lambda wasm layout, in `lam_names` (table-index) order.
    struct LamLayout {
        name: String,
        tag: i64,
        param_wtys: Vec<WType>,
        ret_wty: WType,
        cap_wtys: Vec<WType>,
        fn_index: u32,
    }
    let mut lam_layouts: Vec<LamLayout> = Vec::new();
    for (j, name) in lam_names.iter().enumerate() {
        let sig = fns[name]
            .lam_sig
            .as_ref()
            .ok_or_else(|| format!("wasm backend: lambda `{}` missing its signature", name))?;
        let param_wtys = sig
            .param_tys
            .iter()
            .map(WType::from_ty)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("lambda `{}`: {}", name, e))?;
        let ret_wty = WType::from_ty(&sig.ret_ty).map_err(|e| format!("lambda `{}`: {}", name, e))?;
        let cap_wtys = sig
            .capture_tys
            .iter()
            .map(WType::from_ty)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("lambda `{}`: {}", name, e))?;
        let tag = closure_base + j as i64;
        let fn_index = N_IMPORTS + order.len() as u32 + j as u32;
        closure_tags.insert(name.clone(), tag);
        // call_indirect signature: (i32 closure :: params) -> ret. Keep the FIRST
        // lambda with each signature (its type index == its function index).
        let mut key_params = vec![WType::I32];
        key_params.extend(param_wtys.iter().copied());
        sig_typeidx.entry((key_params, ret_wty)).or_insert(fn_index);
        lam_layouts.push(LamLayout {
            name: name.clone(),
            tag,
            param_wtys,
            ret_wty,
            cap_wtys,
            fn_index,
        });
    }
    let closures = ClosureWasm { base: closure_base, tags: closure_tags, sig_typeidx };
    let n_lambdas = lam_layouts.len() as u32;

    // Helper function indices come after the imports, the user functions, AND the
    // lifted lambdas:
    //   [imports] [user fns] [lambdas] [overflow helpers x4] [heap helpers...]
    let ovf_base = N_IMPORTS + order.len() as u32 + n_lambdas;
    let heap_base = ovf_base + OvfHelper::ALL.len() as u32;

    // Collect the program's distinct String literals (their raw UTF-8 bytes),
    // including those inside lifted lambda bodies.
    let mut lit_list: Vec<Vec<u8>> = Vec::new();
    for name in order.iter().chain(lam_names.iter()) {
        if let Some(ifn) = fns.get(name) {
            collect_str_lits_iexpr(&ifn.body, &mut lit_list);
        }
    }

    // Read-only data region: the bookkeeping area (bump ptr + live + reuses +
    // free-list heads) rounded to 8, followed by the String-literal bytes.
    // String literals live BELOW the heap so their addresses never alias a cell.
    // The free-list array must cover every size class an allocation can request:
    // ADT arities 0..=max_arity AND String size classes 0..=STR_MAX_CLASS.
    let n_freelist_slots = (ctors.max_arity as u64)
        .max(STR_MAX_CLASS)
        .max(TENSOR_MAX_CLASS)
        + 1;
    // A fixed scratch region (after the free-list array, before the read-only
    // literal pool) used by `embed_similarity`: two dim-64 f32 vectors = 512
    // bytes; reserve 1024 to be safe. Its base is `EMBED_SCRATCH`.
    let embed_scratch = {
        let raw = FREELIST_BASE + 4 * n_freelist_slots;
        (raw + 7) & !7
    };
    const EMBED_SCRATCH_BYTES: u64 = 1024;
    let ro_base = embed_scratch + EMBED_SCRATCH_BYTES;
    // Assign each literal a data-segment address; `str_lits` maps bytes -> addr.
    let mut str_lits: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut lit_addr = ro_base;
    for bytes in &lit_list {
        str_lits.insert(bytes.clone(), lit_addr);
        lit_addr += bytes.len() as u64;
    }
    // The heap (bump allocator) starts after all literal bytes, 8-aligned.
    let heap_base_addr = (lit_addr + 7) & !7;

    // 3. Generate a code-section entry per function, in `order`.
    let mut code_entries: Vec<Vec<u8>> = Vec::new();
    let mut type_section_funcs: Vec<(Vec<WType>, WType)> = Vec::new();
    for name in &order {
        let sig = &sigs[name];
        let ifn = fns
            .get(name)
            .ok_or_else(|| format!("wasm backend: function `{}` missing from IR", name))?;
        if ifn.params.len() != sig.params.len() {
            return Err(format!(
                "wasm backend: fn `{}` IR arity {} != signature arity {}",
                name,
                ifn.params.len(),
                sig.params.len()
            ));
        }

        // Seed the local environment with the params (slots 0..n).
        let mut types = HashMap::new();
        let mut index = HashMap::new();
        for (i, (pname, pty)) in ifn.params.iter().zip(sig.params.iter()).enumerate() {
            types.insert(pname.clone(), *pty);
            index.insert(pname.clone(), i as u32);
        }
        // Register every function's index so calls can resolve.
        for (fname, fidx) in &fn_index {
            index.insert(fn_index_key(fname), *fidx);
        }
        let mut env = LocalEnv {
            types,
            index,
            locals: Vec::new(),
            n_params: sig.params.len() as u32,
            sigs: &sigs,
            ovf_base,
            heap_base,
            ctors: &ctors,
            scratch_ptr: None,
            str_lits: &str_lits,
            print_str_idx: PRINT_STR_IDX,
            print_float_idx: PRINT_FLOAT_IDX,
            print_int_idx: PRINT_INT_IDX,
            print_bool_idx: PRINT_BOOL_IDX,
            exp_idx: EXP_IDX,
            block_depth: 0,
            tail: None,
            fn_ret: sig.ret,
            closures: &closures,
        };
        // Self-tail-recursive functions are emitted with their body wrapped in a
        // `loop` (result type = the function's return type). A `TailCall`
        // reassigns the param locals and `br`s back to the loop top — constant
        // stack for tail recursion.
        if ifn.tail_recursive {
            let params: Vec<(u32, WType)> = ifn
                .params
                .iter()
                .zip(sig.params.iter())
                .enumerate()
                .map(|(i, (_pn, pt))| (i as u32, *pt))
                .collect();
            env.tail = Some(TailCtx { params, loop_depth: 1, ret: sig.ret });
        }

        // Emit the body instructions; the result is left on the stack.
        let mut body_code = Vec::new();
        if ifn.tail_recursive {
            body_code.push(0x03); // loop
            body_code.push(sig.ret.byte()); // blocktype = result valtype
            env.block_depth = 1;
        }
        let result_ty = emit_iexpr(&ifn.body, &mut env, &mut body_code)?;
        if ifn.tail_recursive {
            body_code.push(0x0B); // end loop
            env.block_depth = 0;
        }
        if result_ty != sig.ret {
            return Err(format!(
                "wasm backend: fn `{}` body produces {:?} but is declared to return {:?}",
                name, result_ty, sig.ret
            ));
        }

        // Build the code entry: locals declaration + body + end (0x0B).
        // Local groups: one group per local (simple, correct; no run-length).
        let mut entry = Vec::new();
        leb_u(env.locals.len() as u64, &mut entry); // local-group count
        for lty in &env.locals {
            leb_u(1, &mut entry); // count in this group
            entry.push(lty.byte());
        }
        entry.extend_from_slice(&body_code);
        entry.push(0x0B); // end

        // Wrap with its byte-length prefix (the Code section stores sized funcs).
        let mut sized = Vec::new();
        vec_bytes(&entry, &mut sized);
        code_entries.push(sized);

        type_section_funcs.push((sig.params.clone(), sig.ret));
    }

    // 3a'. Append a code entry per lifted lambda, in `lam_layouts` order, with the
    //      closure calling convention: param 0 is the closure cell pointer (i32),
    //      params 1.. are the lambda's value parameters, and the captured values
    //      are loaded from the cell into fresh locals (dup'ing the
    //      reference-counted ones) in a prologue before the body runs.
    for layout in &lam_layouts {
        let ifn = fns
            .get(&layout.name)
            .ok_or_else(|| format!("wasm backend: lambda `{}` missing from IR", layout.name))?;
        if ifn.params.len() != layout.param_wtys.len() || ifn.captures.len() != layout.cap_wtys.len() {
            return Err(format!("wasm backend: lambda `{}` arity/signature mismatch", layout.name));
        }
        let mut types = HashMap::new();
        let mut index = HashMap::new();
        // Lambda value parameters occupy locals 1.. (local 0 is the closure ptr).
        for (i, (pn, pty)) in ifn.params.iter().zip(layout.param_wtys.iter()).enumerate() {
            types.insert(pn.clone(), *pty);
            index.insert(pn.clone(), (i + 1) as u32);
        }
        for (fname, fidx) in &fn_index {
            index.insert(fn_index_key(fname), *fidx);
        }
        let n_params = 1 + ifn.params.len() as u32;
        let mut env = LocalEnv {
            types,
            index,
            locals: Vec::new(),
            n_params,
            sigs: &sigs,
            ovf_base,
            heap_base,
            ctors: &ctors,
            scratch_ptr: None,
            str_lits: &str_lits,
            print_str_idx: PRINT_STR_IDX,
            print_float_idx: PRINT_FLOAT_IDX,
            print_int_idx: PRINT_INT_IDX,
            print_bool_idx: PRINT_BOOL_IDX,
            exp_idx: EXP_IDX,
            block_depth: 0,
            tail: None,
            fn_ret: layout.ret_wty,
            closures: &closures,
        };
        let mut body_code = Vec::new();
        // Prologue: load each capture from the closure cell into a fresh local,
        // dup'ing the reference-counted ones so the body owns them.
        let dup_idx = env.heap_index(HeapHelper::Dup);
        let dup_str_idx = env.heap_index(HeapHelper::DupStr);
        for (i, (cn, ct)) in ifn.captures.iter().zip(layout.cap_wtys.iter()).enumerate() {
            env.add_local(cn, *ct);
            let slot = env.var_index(cn)?;
            body_code.push(0x20); // local.get 0 (closure ptr)
            leb_u(0, &mut body_code);
            let off = CELL_HEADER + SLOT * i as u64;
            match ct {
                WType::F64 => f64_load(off, &mut body_code),
                WType::I64 => i64_load(off, &mut body_code),
                WType::I32 | WType::Ref | WType::Str | WType::Tensor => {
                    i64_load(off, &mut body_code);
                    body_code.push(0xA7); // i32.wrap_i64
                }
                WType::F32 => unreachable!("F32 is not an Aria capture type"),
            }
            body_code.push(0x21); // local.set slot
            leb_u(slot as u64, &mut body_code);
            match ct {
                WType::Ref => {
                    body_code.push(0x20);
                    leb_u(slot as u64, &mut body_code);
                    body_code.push(0x10);
                    leb_u(dup_idx as u64, &mut body_code);
                    body_code.push(0x1A); // drop dummy
                }
                WType::Str => {
                    body_code.push(0x20);
                    leb_u(slot as u64, &mut body_code);
                    body_code.push(0x10);
                    leb_u(dup_str_idx as u64, &mut body_code);
                    body_code.push(0x1A); // drop dummy
                }
                _ => {}
            }
        }
        let result_ty = emit_iexpr(&ifn.body, &mut env, &mut body_code)?;
        if result_ty != layout.ret_wty {
            return Err(format!(
                "wasm backend: lambda `{}` body produces {:?} but its type is {:?}",
                layout.name, result_ty, layout.ret_wty
            ));
        }
        let mut entry = Vec::new();
        leb_u(env.locals.len() as u64, &mut entry);
        for lty in &env.locals {
            leb_u(1, &mut entry);
            entry.push(lty.byte());
        }
        entry.extend_from_slice(&body_code);
        entry.push(0x0B); // end
        let mut sized = Vec::new();
        vec_bytes(&entry, &mut sized);
        code_entries.push(sized);
        // Type: (i32 closure, params...) -> ret.
        let mut params = vec![WType::I32];
        params.extend(layout.param_wtys.iter().copied());
        type_section_funcs.push((params, layout.ret_wty));
    }

    // 3b. Append the checked-arithmetic helper functions, in `OvfHelper::ALL`
    //     order, so their indices line up with `ovf_base + offset`.
    for h in OvfHelper::ALL {
        type_section_funcs.push(h.sig());
        let entry = emit_ovf_helper(h);
        let mut sized = Vec::new();
        vec_bytes(&entry, &mut sized);
        code_entries.push(sized);
    }

    // Closure-cell drop information: `__drop` must release a dead closure cell's
    // reference-counted captures, then free it, per closure tag.
    let closure_drops: Vec<ClosureDropInfo> = lam_layouts
        .iter()
        .map(|l| ClosureDropInfo {
            tag: l.tag,
            ncaps: l.cap_wtys.len(),
            managed: l
                .cap_wtys
                .iter()
                .enumerate()
                .filter(|(_, t)| matches!(t, WType::Ref | WType::Str))
                .map(|(i, t)| (i, *t))
                .collect(),
        })
        .collect();

    // 3c. Append the heap-runtime helpers (alloc/free/dup/drop/live), in
    //     `HeapHelper::ALL` order, so indices line up with `heap_base + offset`.
    for h in HeapHelper::ALL {
        type_section_funcs.push(h.sig());
        let entry = emit_heap_helper(h, &ctors, heap_base, EXP_IDX, embed_scratch, &closure_drops);
        let mut sized = Vec::new();
        vec_bytes(&entry, &mut sized);
        code_entries.push(sized);
    }

    // 4. Assemble the module.
    let mut out = Vec::new();
    // Header: magic + version.
    out.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);

    // Type section (id 1): type index 0 is `env.print_str`'s signature
    // `(i32, i32) -> ()`; type index 1 is `env.print_float`'s `(f64) -> ()`;
    // type index 2 is `env.print_int`'s `(i64) -> ()`; type index 3 is
    // `env.print_bool`'s `(i32) -> ()`; type indices 4.. are the defined
    // functions' single-result signatures.
    {
        let mut content = Vec::new();
        leb_u((N_IMPORTS as usize + type_section_funcs.len()) as u64, &mut content);
        // type 0: env.print_str(ptr:i32, len:i32) -> ()  (no results)
        content.push(0x60); // func type tag
        leb_u(2, &mut content); // 2 params
        content.push(WType::I32.byte());
        content.push(WType::I32.byte());
        leb_u(0, &mut content); // zero results
        // type 1: env.print_float(f64) -> ()  (no results)
        content.push(0x60); // func type tag
        leb_u(1, &mut content); // 1 param
        content.push(WType::F64.byte());
        leb_u(0, &mut content); // zero results
        // type 2: env.print_int(i64) -> ()  (no results)
        content.push(0x60); // func type tag
        leb_u(1, &mut content); // 1 param
        content.push(WType::I64.byte());
        leb_u(0, &mut content); // zero results
        // type 3: env.print_bool(i32) -> ()  (no results)
        content.push(0x60); // func type tag
        leb_u(1, &mut content); // 1 param
        content.push(WType::I32.byte());
        leb_u(0, &mut content); // zero results
        // type 4: env.exp(f64) -> f64
        content.push(0x60); // func type tag
        leb_u(1, &mut content); // 1 param
        content.push(WType::F64.byte());
        leb_u(1, &mut content); // one result
        content.push(WType::F64.byte());
        // types 5..: the defined functions.
        for (params, ret) in &type_section_funcs {
            content.push(0x60); // func type tag
            leb_u(params.len() as u64, &mut content);
            for p in params {
                content.push(p.byte());
            }
            leb_u(1, &mut content); // exactly one result
            content.push(ret.byte());
        }
        section(1, &content, &mut out);
    }

    // Import section (id 2): import `env.print_str` (type 0),
    // `env.print_float` (type 1), `env.print_int` (type 2),
    // `env.print_bool` (type 3), and `env.exp` (type 4).
    {
        let mut content = Vec::new();
        leb_u(N_IMPORTS as u64, &mut content); // five imports
        let m = b"env";
        // env.print_str : type 0
        leb_u(m.len() as u64, &mut content);
        content.extend_from_slice(m);
        let nm = b"print_str";
        leb_u(nm.len() as u64, &mut content);
        content.extend_from_slice(nm);
        content.push(0x00); // import kind = func
        leb_u(0, &mut content); // type index 0
        // env.print_float : type 1
        leb_u(m.len() as u64, &mut content);
        content.extend_from_slice(m);
        let nf = b"print_float";
        leb_u(nf.len() as u64, &mut content);
        content.extend_from_slice(nf);
        content.push(0x00); // import kind = func
        leb_u(1, &mut content); // type index 1
        // env.print_int : type 2
        leb_u(m.len() as u64, &mut content);
        content.extend_from_slice(m);
        let ni = b"print_int";
        leb_u(ni.len() as u64, &mut content);
        content.extend_from_slice(ni);
        content.push(0x00); // import kind = func
        leb_u(2, &mut content); // type index 2
        // env.print_bool : type 3
        leb_u(m.len() as u64, &mut content);
        content.extend_from_slice(m);
        let nb = b"print_bool";
        leb_u(nb.len() as u64, &mut content);
        content.extend_from_slice(nb);
        content.push(0x00); // import kind = func
        leb_u(3, &mut content); // type index 3
        // env.exp : type 4
        leb_u(m.len() as u64, &mut content);
        content.extend_from_slice(m);
        let ne = b"exp";
        leb_u(ne.len() as u64, &mut content);
        content.extend_from_slice(ne);
        content.push(0x00); // import kind = func
        leb_u(4, &mut content); // type index 4
        section(2, &content, &mut out);
    }

    // Function section (id 3): type index per DEFINED function. Type indices
    // 0..N_IMPORTS are the imports; defined function `i` uses type index
    // `N_IMPORTS + i`.
    {
        let n = type_section_funcs.len();
        let mut content = Vec::new();
        leb_u(n as u64, &mut content);
        for i in 0..n {
            leb_u((N_IMPORTS as usize + i) as u64, &mut content); // type index
        }
        section(3, &content, &mut out);
    }

    // Table section (id 4): one funcref table sized to hold every lifted lambda,
    // used by `call_indirect` for closure dispatch. Omitted when there are no
    // lambdas (no closures in the program).
    if n_lambdas > 0 {
        let mut content = Vec::new();
        leb_u(1, &mut content); // one table
        content.push(0x70); // element type = funcref
        content.push(0x00); // limits: flags=0 (min only)
        leb_u(n_lambdas as u64, &mut content); // min size
        section(4, &content, &mut out);
    }

    // Memory section (id 5): one memory, min `MEM_PAGES` pages, no max.
    {
        let mut content = Vec::new();
        leb_u(1, &mut content); // one memory
        content.push(0x00); // limits: flags=0 (min only)
        leb_u(MEM_PAGES, &mut content); // min pages
        section(5, &content, &mut out);
    }

    // Export section (id 7): `main`, `__live`, `__reuses`, and `memory`.
    {
        let main_idx = fn_index["main"];
        let live_idx = heap_base + HeapHelper::Live.offset();
        let reuses_idx = heap_base + HeapHelper::Reuses.offset();
        let mut content = Vec::new();
        leb_u(4, &mut content); // four exports
        // main (func)
        let n = b"main";
        leb_u(n.len() as u64, &mut content);
        content.extend_from_slice(n);
        content.push(0x00);
        leb_u(main_idx as u64, &mut content);
        // __live (func)
        let n = b"__live";
        leb_u(n.len() as u64, &mut content);
        content.extend_from_slice(n);
        content.push(0x00);
        leb_u(live_idx as u64, &mut content);
        // __reuses (func)
        let n = b"__reuses";
        leb_u(n.len() as u64, &mut content);
        content.extend_from_slice(n);
        content.push(0x00);
        leb_u(reuses_idx as u64, &mut content);
        // memory (mem index 0)
        let n = b"memory";
        leb_u(n.len() as u64, &mut content);
        content.extend_from_slice(n);
        content.push(0x02); // export kind = memory
        leb_u(0, &mut content);
        section(7, &content, &mut out);
    }

    // Element section (id 9): initialize the funcref table — `table[j]` is the
    // wasm function index of the j-th lifted lambda (a closure cell of tag
    // `closure_base + j` dispatches to it). One active segment at table offset 0.
    if n_lambdas > 0 {
        let mut content = Vec::new();
        leb_u(1, &mut content); // one element segment
        content.push(0x00); // flags=0: active, table 0, i32.const offset
        content.push(0x41); // i32.const 0
        leb_s(0, &mut content);
        content.push(0x0B); // end of offset expr
        leb_u(n_lambdas as u64, &mut content); // function count
        for layout in &lam_layouts {
            leb_u(layout.fn_index as u64, &mut content);
        }
        section(9, &content, &mut out);
    }

    // Code section (id 10): vec of sized function bodies.
    {
        let mut content = Vec::new();
        leb_u(code_entries.len() as u64, &mut content);
        for e in &code_entries {
            content.extend_from_slice(e);
        }
        section(10, &content, &mut out);
    }

    // Data section (id 11): initialize the bump pointer to HEAP_BASE. (Linear
    // memory is zero-initialized, so the live word and free-list heads start at
    // 0 — exactly what we want.) An active data segment at BUMP_PTR_ADDR writes
    // the 4-byte little-endian initial bump value.
    {
        let mut content = Vec::new();
        // segment count = 1 (bump pointer) + one per String literal.
        leb_u((1 + lit_list.len()) as u64, &mut content);
        // segment 0: the initial bump-pointer value at BUMP_PTR_ADDR.
        content.push(0x00); // flags=0: active, memory 0, i32.const offset
        content.push(0x41); // i32.const
        leb_s(BUMP_PTR_ADDR as i64, &mut content);
        content.push(0x0B); // end of offset expr
        let bytes = (heap_base_addr as u32).to_le_bytes();
        leb_u(bytes.len() as u64, &mut content);
        content.extend_from_slice(&bytes);
        // one segment per String literal: raw UTF-8 bytes at their pool address.
        for bytes in &lit_list {
            let addr = str_lits[bytes];
            content.push(0x00); // flags=0: active, memory 0
            content.push(0x41); // i32.const
            leb_s(addr as i64, &mut content);
            content.push(0x0B); // end of offset expr
            leb_u(bytes.len() as u64, &mut content);
            content.extend_from_slice(bytes);
        }
        section(11, &content, &mut out);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interp, lexer, parser, typeck};
    use std::process::Command;

    fn node_available() -> bool {
        Command::new("node")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Run the compiled wasm under Node, returning the printed result or "TRAP".
    fn run_wasm(bytes: &[u8]) -> Result<String, String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir();
        // Unique per call: tests run in parallel within one process, so a
        // PID-only name would let concurrent runs clobber each other's file.
        let path = dir.join(format!(
            "aria_wasm_test_{}_{}.wasm",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
        let script = format!(
            "const fs=require('fs');\
             const dec=new TextDecoder();\
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}},exp:Math.exp}}}};\
             try{{const b=fs.readFileSync({:?});\
             WebAssembly.instantiate(b,imp).then(r=>{{\
             try{{const ex=r.instance.exports;const v=ex.main();\
             if(typeof v==='bigint'){{process.stdout.write(String(v));}}\
             else{{const mem=new Uint8Array(ex.memory.buffer);\
             const dv=new DataView(ex.memory.buffer);\
             const len=Number(dv.getBigInt64(v+8,true));\
             process.stdout.write(dec.decode(mem.subarray(v+16,v+16+len)));}}}}\
             catch(e){{process.stdout.write('TRAP');}}\
             }}).catch(e=>{{process.stdout.write('TRAP');}});}}\
             catch(e){{process.stdout.write('TRAP');}}",
            path.to_string_lossy()
        );
        let out = Command::new("node")
            .arg("-e")
            .arg(&script)
            .output()
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&path);
        if !out.status.success() {
            return Ok("TRAP".to_string());
        }
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    }

    fn interp_result(src: &str) -> Result<String, String> {
        let toks = lexer::lex(src)?;
        let prog = parser::parse(toks)?;
        typeck::check(&prog).map_err(|e| e.join("; "))?;
        let it = interp::Interp::new(&prog)?;
        it.run_main().map(|v| v.display())
    }

    fn compile_src(src: &str) -> Result<Vec<u8>, String> {
        let toks = lexer::lex(src)?;
        let prog = parser::parse(toks)?;
        typeck::check(&prog).map_err(|e| e.join("; "))?;
        compile(&prog)
    }

    /// Differential check: compiled-wasm-via-Node must equal the interpreter.
    fn differential(src: &str) {
        let interp = interp_result(src).expect("interpreter should succeed on battery");
        let bytes = compile_src(src).expect("compile should succeed on battery");
        if !node_available() {
            return; // gate gracefully when node is missing
        }
        let wasm = run_wasm(&bytes).expect("running wasm");
        assert_eq!(interp, wasm, "wasm != interpreter for:\n{}", src);
    }

    /// Differential + garbage-free for closures: the compiled wasm result must
    /// equal the interpreter AND leave `__live == 0` (every closure/ADT cell
    /// freed). Handles both Int- and String-returning `main` (closure programs
    /// that print).
    fn differential_gc(src: &str) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let interp = interp_result(src).expect("interpreter should succeed on battery");
        let bytes = compile_src(src).expect("compile should succeed on battery");
        if !node_available() {
            return;
        }
        let path = std::env::temp_dir().join(format!(
            "aria_wasm_gc_{}_{}.wasm",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, &bytes).expect("write wasm");
        let script = format!(
            "const fs=require('fs');\
             const dec=new TextDecoder();\
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}},exp:Math.exp}}}};\
             WebAssembly.instantiate(fs.readFileSync({:?}),imp).then(r=>{{\
             const ex=r.instance.exports;const v=ex.main();\
             let s;if(typeof v==='bigint'){{s=String(v);}}\
             else{{const mem=new Uint8Array(ex.memory.buffer);\
             const dv=new DataView(ex.memory.buffer);\
             const len=Number(dv.getBigInt64(v+8,true));\
             s=dec.decode(mem.subarray(v+16,v+16+len));}}\
             process.stdout.write(s+'\\x00'+String(ex.__live()));\
             }}).catch(e=>process.stdout.write('TRAP\\x000'));",
            path.to_string_lossy()
        );
        let out = Command::new("node").arg("-e").arg(&script).output().expect("node run");
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&out.stdout).to_string();
        let (wasm, live) = s.split_once('\u{0}').expect("harness output");
        assert_eq!(interp, wasm, "wasm != interpreter for:\n{}", src);
        assert_eq!(live.parse::<i64>().unwrap_or(-1), 0, "wasm not garbage-free for:\n{}", src);
    }

    #[test]
    fn wasm_closures_match_and_are_garbage_free() {
        // Immediate application, currying via a returned closure, and a captured
        // local — all dispatched through the function table garbage-free.
        differential_gc("fn main() -> Int = (\\x -> x * 2)(21)");
        differential_gc(
            "fn add(x: Int) -> (Int) -> Int = \\y -> x + y\n\
             fn main() -> Int = add(3)(4)",
        );
        // Generic higher-order map with a capturing lambda and a function by name,
        // plus a Ref-captured heap list released when the closure cell dies.
        differential_gc(
            "type List = | Nil | Cons(Int, List)\n\
             fn map(f: (Int) -> Int, xs: List) -> List = match xs { Nil => Nil, Cons(h, t) => Cons(f(h), map(f, t)), }\n\
             fn sum(xs: List) -> Int = match xs { Nil => 0, Cons(h, t) => h + sum(t), }\n\
             fn dbl(x: Int) -> Int = x * 2\n\
             fn main() -> Int = {\n\
               let n = 10;\n\
               let xs = Cons(1, Cons(2, Cons(3, Nil)));\n\
               sum(map(\\x -> x + n, xs)) + sum(map(dbl, xs))\n\
             }",
        );
        // A closure stored then applied twice, and compose (a closure capturing
        // two other closures).
        differential_gc(
            "fn twice(f: (Int) -> Int, x: Int) -> Int = f(f(x))\n\
             fn main() -> Int = twice(\\n -> n + 5, 100)",
        );
        differential_gc(
            "fn compose(f: (Int)->Int, g: (Int)->Int) -> (Int)->Int = \\x -> f(g(x))\n\
             fn main() -> Int = {\n\
               let h = compose(\\(a: Int) -> a + 1, \\(b: Int) -> b * 2);\n\
               h(10) + h(20)\n\
             }",
        );
    }

    #[test]
    fn wasm_matches_interpreter() {
        // Curated PURE Int/Bool battery, all within non-overflowing ranges
        // (wasm i64 wraps; the interpreter checks — so we stay small).
        let battery = [
            // factorial via integer match (lowered to nested if)
            "fn fac(n: Int) -> Int = match n { 0 => 1, _ => n * fac(n - 1), }\n\
             fn main() -> Int = fac(10)",
            // fibonacci, two-branch recursion
            "fn fib(n: Int) -> Int = if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\n\
             fn main() -> Int = fib(15)",
            // plain arithmetic with precedence + div + mod
            "fn main() -> Int = (1 + 2 * 3 - 4) / 1 + 17 % 5",
            // nested if returning Int
            "fn sign(n: Int) -> Int = if n < 0 { 0 - 1 } else { if n > 0 { 1 } else { 0 } }\n\
             fn main() -> Int = sign(0 - 42) + sign(7) + sign(0)",
            // Int-literal match with several arms + catch-all
            "fn name(n: Int) -> Int = match n { 1 => 100, 2 => 200, 3 => 300, _ => 0, }\n\
             fn main() -> Int = name(1) + name(2) + name(3) + name(9)",
            // 2-arg recursion (Ackermann-ish gcd)
            "fn gcd(a: Int, b: Int) -> Int = if b == 0 { a } else { gcd(b, a % b) }\n\
             fn main() -> Int = gcd(1071, 462)",
            // boolean / comparison feeding an if; unary neg + not exercised
            "fn pos(n: Int) -> Int = if !(n < 0) { 1 } else { 0 }\n\
             fn main() -> Int = pos(-3) + pos(3)",
            // let-bindings inside a block
            "fn main() -> Int = { let a = 6; let b = 7; let c = a * b; c - 2 }",
            // short-circuit && lowered to control flow
            "fn main() -> Int = { let b = (3 > 1) && (2 < 4); if b { 1 } else { 0 } }",
            // power by repeated recursion
            "fn pow(b: Int, e: Int) -> Int = if e == 0 { 1 } else { b * pow(b, e - 1) }\n\
             fn main() -> Int = pow(2, 20)",
        ];
        for src in battery {
            differential(src);
        }
    }

    #[test]
    fn trap_on_div_by_zero_treated_as_error() {
        // wasm div_s by 0 traps -> "TRAP"; the interpreter errors. Both => error.
        let src = "fn main() -> Int = 1 / 0";
        let interp = interp_result(src);
        assert!(interp.is_err(), "interpreter should error on div by zero");
        let bytes = compile_src(src).expect("compiles fine");
        if node_available() {
            assert_eq!(run_wasm(&bytes).unwrap(), "TRAP");
        }
    }

    /// Both backends must treat an overflowing program as an error: the
    /// interpreter returns `Err`, and the compiled wasm traps (`unreachable`),
    /// which Node surfaces as `TRAP`. Both are "error" => agreement.
    fn assert_overflow_agrees(src: &str) {
        let interp = interp_result(src);
        assert!(
            interp.is_err(),
            "interpreter should error on overflow for:\n{}",
            src
        );
        let bytes = compile_src(src).expect("overflowing program still compiles");
        if node_available() {
            assert_eq!(
                run_wasm(&bytes).unwrap(),
                "TRAP",
                "wasm should trap on overflow for:\n{}",
                src
            );
        }
    }

    #[test]
    fn add_overflow_errors_in_both_backends() {
        // i64::MAX + 1 overflows.
        assert_overflow_agrees("fn main() -> Int = 9223372036854775807 + 1");
    }

    #[test]
    fn mul_overflow_errors_in_both_backends() {
        // 3037000500^2 exceeds i64::MAX.
        assert_overflow_agrees("fn main() -> Int = 3037000500 * 3037000500");
    }

    #[test]
    fn neg_overflow_errors_in_both_backends() {
        // Negating i64::MIN overflows. Build MIN via (0 - MAX) - 1, then negate.
        assert_overflow_agrees(
            "fn neg(x: Int) -> Int = -x\n\
             fn main() -> Int = neg((0 - 9223372036854775807) - 1)",
        );
        // Sub-form overflow: 0 - i64::MIN (the lowered unary-neg of a literal).
        assert_overflow_agrees("fn main() -> Int = 0 - ((0 - 9223372036854775807) - 1)");
    }

    #[test]
    fn near_boundary_non_overflow_still_agrees() {
        // (i64::MAX - 1) + 1 == i64::MAX: right at the edge, must NOT trap and
        // must agree between backends. Also a checked multiply that fits, and a
        // safe unary negation.
        differential("fn main() -> Int = 9223372036854775806 + 1");
        differential("fn main() -> Int = 3037000499 * 3037000499");
        differential(
            "fn neg(x: Int) -> Int = -x\n\
             fn main() -> Int = neg(42) + neg(0 - 7)",
        );
    }

    #[test]
    fn unsupported_programs_return_err_not_panic() {
        // Unit-returning `main` (Unit is outside the compilable result subset).
        let s5 = "fn main() -> Unit = ()";
        // `main` returning Bool: the Node harness decodes only Int/Float/String
        // results, so a Bool-returning `main` is rejected (even though structural
        // ADT `==` itself is now supported — here it's main's RETURN type that is
        // out of subset).
        let s8 = "type P = | P(Int, Int)\nfn main() -> Bool = P(1, 2) == P(1, 2)";
        for src in [s5, s8] {
            let r = compile_src(src);
            assert!(r.is_err(), "expected Err (no panic) for:\n{}", src);
        }
    }

    // ---- Float (f64) support --------------------------------------------

    /// Run a `main -> Float` compiled module under Node, returning the raw f64
    /// bit pattern (so the comparison is NUMERIC/bitwise, never via printed
    /// text — Rust `{}` and JS `Number.toString()` differ for some f64s). Also
    /// returns `__live`. A trap surfaces as `Err`.
    fn run_wasm_f64_bits(bytes: &[u8]) -> Result<(u64, i64), String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "aria_wasm_f64_{}_{}.wasm",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
        // Write the returned f64 into a Float64Array and read its raw bits via a
        // BigUint64Array view, so we compare bit patterns (NaN-safe, exact).
        let script = format!(
            "const fs=require('fs');\
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}},exp:Math.exp}}}};\
             const b=fs.readFileSync({:?});\
             WebAssembly.instantiate(b,imp).then(r=>{{\
             const ex=r.instance.exports;const v=ex.main();\
             const fb=new Float64Array(1);fb[0]=v;\
             const ub=new BigUint64Array(fb.buffer);\
             process.stdout.write(String(ub[0])+'|'+String(ex.__live()));\
             }}).catch(e=>process.stdout.write('TRAP|0'));",
            path.to_string_lossy()
        );
        let out = Command::new("node").arg("-e").arg(&script).output().map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&out.stdout).to_string();
        if s.starts_with("TRAP") {
            return Err("TRAP".into());
        }
        let (bits, live) = s.split_once('|').ok_or("bad harness output")?;
        Ok((
            bits.parse::<u64>().map_err(|e| e.to_string())?,
            live.parse::<i64>().unwrap_or(-1),
        ))
    }

    /// Differential check for a `main -> Float` program: parse the interpreter's
    /// printed float and the wasm's returned f64 and assert they are BITWISE
    /// equal (NaN bit patterns compared raw too), never by printed text.
    fn differential_float(src: &str) {
        let interp = interp_result(src).expect("interpreter should succeed");
        let expected: f64 = interp.parse().expect("interpreter float parses");
        let bytes = compile_src(src).expect("compile should succeed");
        if !node_available() {
            return;
        }
        let (wasm_bits, live) = run_wasm_f64_bits(&bytes).expect("running wasm");
        assert_eq!(
            expected.to_bits(),
            wasm_bits,
            "wasm f64 bits != interpreter for:\n{}\n  interp={} ({:#x}) wasm={:#x}",
            src,
            expected,
            expected.to_bits(),
            wasm_bits
        );
        assert_eq!(live, 0, "float program leaked {} live cell(s) in:\n{}", live, src);
    }

    #[test]
    fn float_computation_through_int_and_bool() {
        // Exact Int-valued results from float computation compare cleanly (no
        // float text formatting involved). `main` returns Int (the wasm harness
        // decodes an i64 result); the float work happens internally.
        differential("fn main() -> Int = if 1.5 * 2.0 == 3.0 { 1 } else { 0 }");
        differential("fn main() -> Int = if 0.1 + 0.2 > 0.3 { 1 } else { 0 }");
        differential("fn main() -> Int = if 2.5 - 1.5 == 1.0 { 1 } else { 0 }");
        differential("fn main() -> Int = if 7.0 / 2.0 == 3.5 { 1 } else { 0 }");
        // Float division by 0.0 yields +inf (no trap): inf > 1000000.0 is true.
        differential("fn main() -> Int = if 1.0 / 0.0 > 1000000.0 { 1 } else { 0 }");
        // NaN comparisons: 0.0/0.0 is NaN; NaN == NaN is false, NaN != NaN true.
        differential("fn main() -> Int = if (0.0 / 0.0) == (0.0 / 0.0) { 1 } else { 0 }");
        differential("fn main() -> Int = if (0.0 / 0.0) != (0.0 / 0.0) { 1 } else { 0 }");
        // Ordering ops (<, <=, >, >=) combined.
        differential(
            "fn main() -> Int = if 1.25 < 1.5 && 2.0 >= 2.0 && 3.0 <= 3.0 { 1 } else { 0 }",
        );
    }

    #[test]
    fn float_negation() {
        differential("fn main() -> Int = if -2.5 + 2.5 == 0.0 { 1 } else { 0 }");
        differential(
            "fn neg(x: Float) -> Float = -x\n\
             fn main() -> Int = if neg(3.5) == -3.5 { 1 } else { 0 }",
        );
        // main -> Float, numeric (bitwise) comparison.
        differential_float("fn main() -> Float = -2.5");
        differential_float("fn neg(x: Float) -> Float = -x\nfn main() -> Float = neg(0.0 - 7.5)");
    }

    #[test]
    fn float_main_returns_f64_numeric() {
        // main -> Float results compared by f64 bits, not printed text.
        differential_float("fn area(r: Float) -> Float = 3.14159 * r * r\nfn main() -> Float = area(2.0) / 2.0");
        differential_float("fn main() -> Float = 1.0 / 0.0"); // +inf
        differential_float("fn main() -> Float = { let a = 1.5; let b = 2.0; a + b }");
    }

    #[test]
    fn float_field_in_adt_is_garbage_free() {
        // An ADT carrying Float fields: sum/compare them, discard the cell. The
        // Float field is NOT reference-managed, so __drop just frees the cell;
        // the heap must end garbage-free (__live == 0). Result is a Bool.
        differential_heap(
            "type V = | V(Float, Float)\n\
             fn main() -> Int = match V(1.5, 2.25) { V(a, b) => if a + b == 3.75 { 1 } else { 0 }, }",
        );
        // A list of Float-field records, summed via a Bool/Int result.
        differential_heap(
            "type P = | P(Float, Float)\n\
             type L = | Nil | Cons(P, L)\n\
             fn cnt(xs: L) -> Int = match xs { Nil => 0, Cons(_, r) => 1 + cnt(r), }\n\
             fn main() -> Int = cnt(Cons(P(1.0, 2.0), Cons(P(3.0, 4.0), Nil)))",
        );
    }

    /// Approximate float differential for Tensor ops whose value can round
    /// slightly differently between the interpreter's `f32::exp` and the wasm
    /// backend's `Math.exp` host import (softmax) or accumulated f32 rounding.
    /// Compares as f64 within a small absolute tolerance, and still asserts the
    /// program is garbage-free (`__live == 0`).
    fn differential_float_approx(src: &str, tol: f64) {
        let interp = interp_result(src).expect("interpreter should succeed");
        let expected: f64 = interp.parse().expect("interpreter float parses");
        let bytes = compile_src(src).expect("compile should succeed");
        if !node_available() {
            return;
        }
        let (wasm_bits, live) = run_wasm_f64_bits(&bytes).expect("running wasm");
        let got = f64::from_bits(wasm_bits);
        assert!(
            (got - expected).abs() <= tol,
            "wasm float {} not within {} of interpreter {} for:\n{}",
            got,
            tol,
            expected,
            src
        );
        assert_eq!(live, 0, "tensor program leaked {} live cell(s) in:\n{}", live, src);
    }

    #[test]
    fn tensor_zeros_set_get_matches_interpreter() {
        // tensor_zeros / tensor_set / tensor_get round-trip a single element.
        differential_float("fn main() -> Float = tensor_get(tensor_set(tensor_zeros(2, 2), 1, 1, 4.25), 1, 1)");
        // An untouched element stays 0.
        differential_float("fn main() -> Float = tensor_get(tensor_zeros(3, 4), 2, 3)");
        // tensor_rows / tensor_cols reduce to Int (compared exactly).
        differential("fn main() -> Int = tensor_rows(tensor_zeros(5, 7)) + tensor_cols(tensor_zeros(5, 7))");
    }

    #[test]
    fn tensor_matmul_identity_matches_interpreter() {
        // identity(2x2) * M, read element (0,1) == 3.0 (the prompt's example).
        differential_float(
            "fn main() -> Float = tensor_get(matmul(tensor_set(tensor_set(tensor_zeros(2,2),0,0,1.0),1,1,1.0), tensor_set(tensor_zeros(2,2),0,1,3.0)), 0, 1)",
        );
        // A non-trivial 2x2 * 2x2 product element.
        differential_float(
            "fn main() -> Float = {\n\
               let a = tensor_set(tensor_set(tensor_set(tensor_set(tensor_zeros(2,2),0,0,1.0),0,1,2.0),1,0,3.0),1,1,4.0);\n\
               let b = tensor_set(tensor_set(tensor_set(tensor_set(tensor_zeros(2,2),0,0,5.0),0,1,6.0),1,0,7.0),1,1,8.0);\n\
               tensor_get(matmul(a, b), 1, 1)\n\
             }",
        );
    }

    #[test]
    fn tensor_relu_transpose_matches_interpreter() {
        // relu zeroes a negative, transpose moves it; read the moved element.
        differential_float(
            "fn main() -> Float = {\n\
               let a = tensor_set(tensor_set(tensor_zeros(2,2),0,0,-2.0),0,1,5.0);\n\
               let t = transpose(relu(a));\n\
               tensor_get(t, 1, 0)\n\
             }",
        );
        // relu of a negative element is exactly 0.0.
        differential_float(
            "fn main() -> Float = tensor_get(relu(tensor_set(tensor_zeros(1,1),0,0,-9.0)), 0, 0)",
        );
    }

    #[test]
    fn tensor_softmax_matches_interpreter() {
        // Row-wise softmax; the largest input gets the largest probability.
        // exp via the host import may round in the last ulp vs the interpreter's
        // f32::exp, so compare within a small tolerance (still garbage-free).
        differential_float_approx(
            "fn main() -> Float = {\n\
               let a = tensor_set(tensor_set(tensor_zeros(1,3),0,0,1.0),0,2,3.0);\n\
               tensor_get(softmax(a), 0, 2)\n\
             }",
            1e-5,
        );
    }

    #[test]
    fn embed_similarity_matches_interpreter() {
        // Identical strings -> cosine ~1.0; both backends agree (approx, since
        // FNV/normalization is f32 and division may round slightly).
        differential_float_approx(
            "fn main() -> Float = embed_similarity(\"cosine similarity over vectors\", \"cosine similarity over vectors\")",
            1e-5,
        );
        // Unrelated strings -> a much smaller similarity; agreement within tol.
        differential_float_approx(
            "fn main() -> Float = embed_similarity(\"cosine similarity over vectors\", \"the weather is cold and rainy today\")",
            1e-5,
        );
    }

    #[test]
    fn adt_structural_equality_matches_interpreter() {
        // Structural ADT `==`/`!=` compiled to the recursive `__eq` helper must
        // agree with the interpreter AND end garbage-free (`__live == 0`). The
        // result is an Int derived from the comparison (main can't return Bool).

        // Nullary constructors: equal vs unequal tags, `==` and `!=`.
        differential_heap(
            "type C = | Red | Green | Blue\n\
             fn main() -> Int = if Red == Red { 1 } else { 0 }",
        );
        differential_heap(
            "type C = | Red | Green | Blue\n\
             fn main() -> Int = if Red == Green { 1 } else { 0 }",
        );
        differential_heap(
            "type C = | Red | Green | Blue\n\
             fn main() -> Int = if Red != Blue { 1 } else { 0 }",
        );

        // Int fields: equal, unequal, and a different constructor.
        differential_heap(
            "type P = | A(Int, Int) | B(Int)\n\
             fn main() -> Int = if A(1, 2) == A(1, 2) { 1 } else { 0 }",
        );
        differential_heap(
            "type P = | A(Int, Int) | B(Int)\n\
             fn main() -> Int = if A(1, 2) == A(1, 9) { 1 } else { 0 }",
        );
        differential_heap(
            "type P = | A(Int, Int) | B(Int)\n\
             fn main() -> Int = if A(1, 2) != B(1) { 1 } else { 0 }",
        );

        // Bool fields.
        differential_heap(
            "type Q = | Q(Bool, Bool)\n\
             fn main() -> Int = if Q(true, false) == Q(true, false) { 1 } else { 0 }",
        );
        differential_heap(
            "type Q = | Q(Bool, Bool)\n\
             fn main() -> Int = if Q(true, false) == Q(true, true) { 1 } else { 0 }",
        );

        // Float fields.
        differential_heap(
            "type F = | F(Float, Float)\n\
             fn main() -> Int = if F(1.5, 2.5) == F(1.5, 2.5) { 1 } else { 0 }",
        );
        differential_heap(
            "type F = | F(Float, Float)\n\
             fn main() -> Int = if F(1.5, 2.5) == F(1.5, 9.0) { 1 } else { 0 }",
        );

        // String fields (compared via `__streq` inside `__eq`).
        differential_heap(
            "type S = | S(String, Int)\n\
             fn main() -> Int = if S(\"hi\", 3) == S(\"hi\", 3) { 1 } else { 0 }",
        );
        differential_heap(
            "type S = | S(String, Int)\n\
             fn main() -> Int = if S(\"hi\", 3) == S(\"bye\", 3) { 1 } else { 0 }",
        );

        // NESTED ADTs (Ref fields): recursive `__eq` descends both lists.
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn main() -> Int = if Cons(1, Cons(2, Nil)) == Cons(1, Cons(2, Nil)) { 1 } else { 0 }",
        );
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn main() -> Int = if Cons(1, Cons(2, Nil)) == Cons(1, Cons(3, Nil)) { 1 } else { 0 }",
        );
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn main() -> Int = if Cons(1, Nil) != Cons(1, Cons(2, Nil)) { 1 } else { 0 }",
        );
    }

    #[test]
    fn adt_equality_shared_operand_no_double_free() {
        // A value compared (consumed by `==`) and then USED AGAIN: the rc pass
        // dups it before the comparison, the `==` site drops its one owned
        // reference, and the later use drops the other. No double-free; the heap
        // ends garbage-free (`__live == 0`).
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn lenL(xs: L) -> Int = match xs { Nil => 0, Cons(_, r) => 1 + lenL(r), }\n\
             fn main() -> Int = {\n\
               let xs = Cons(1, Cons(2, Nil));\n\
               let ys = Cons(1, Cons(2, Nil));\n\
               let eq = if xs == ys { 1 } else { 0 };\n\
               eq + lenL(xs)\n\
             }",
        );
        // Both operands shared and reused after the comparison.
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn lenL(xs: L) -> Int = match xs { Nil => 0, Cons(_, r) => 1 + lenL(r), }\n\
             fn main() -> Int = {\n\
               let xs = Cons(1, Cons(2, Nil));\n\
               let ys = Cons(1, Cons(2, Nil));\n\
               let eq = if xs == ys { 10 } else { 0 };\n\
               eq + lenL(xs) + lenL(ys)\n\
             }",
        );
    }

    #[test]
    fn float_main_returns_f64() {
        // A Float-field ADT whose matched field flows out as main's f64 result.
        differential_float(
            "type V = | V(Float, Float)\n\
             fn main() -> Float = match V(1.5, 2.25) { V(a, b) => a + b, }",
        );
    }

    #[test]
    fn print_float_simple_value_matches() {
        // print_float on values where Rust `{}` (the interpreter) and JS
        // Number.toString (the Node harness) agree (small exact decimals). The
        // interpreter prints via println!("{}", f); the expected text is exactly
        // `format!("{}", f)`, compared to the wasm's captured stdout. NOTE:
        // float text formatting can differ between Rust and JS for some values,
        // so this is restricted to exact decimals where they agree.
        for (src, value) in [
            ("fn main() -> Int = { print_float(1.5); 0 }", 1.5f64),
            ("fn main() -> Int = { print_float(2.0 + 0.5); 0 }", 2.5f64),
            ("fn main() -> Int = { print_float(-3.25); 0 }", -3.25f64),
        ] {
            let expected = format!("{}\n", value);
            let bytes = compile_src(src).expect("compile should succeed");
            if !node_available() {
                continue;
            }
            let wasm_out = run_wasm_capture_stdout(&bytes).expect("running wasm");
            assert_eq!(expected, wasm_out, "print_float output mismatch for:\n{}", src);
        }
    }

    /// Run a compiled module capturing the host print side effects' stdout (the
    /// `print_float` import writes String(x)+"\n"), ignoring the Int result.
    fn run_wasm_capture_stdout(bytes: &[u8]) -> Result<String, String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "aria_wasm_pf_{}_{}.wasm",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
        let script = format!(
            "const fs=require('fs');\
             const imp={{env:{{print_str:(p,n)=>{{}},\
             print_float:(x)=>{{process.stdout.write(String(x));process.stdout.write('\\n');}},\
             print_int:(n)=>{{process.stdout.write(String(n));process.stdout.write('\\n');}},\
             print_bool:(b)=>{{process.stdout.write(b?'true':'false');process.stdout.write('\\n');}},\
             exp:Math.exp}}}};\
             const b=fs.readFileSync({:?});\
             WebAssembly.instantiate(b,imp).then(r=>{{r.instance.exports.main();}})\
             .catch(e=>process.stdout.write('TRAP'));",
            path.to_string_lossy()
        );
        let out = Command::new("node").arg("-e").arg(&script).output().map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&path);
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    }

    // ---- Phase 2d: heap-allocated Strings -------------------------------

    /// Run a String-returning compiled module under Node, returning
    /// `(decoded_string, live_cells)`. The host `env.print_str` import is a
    /// no-op here (these programs don't print); `__live` is read after `main`.
    fn run_wasm_str_live(bytes: &[u8]) -> Result<(String, i64), String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "aria_wasm_str_{}_{}.wasm",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
        let script = format!(
            "const fs=require('fs');\
             const dec=new TextDecoder();\
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}},exp:Math.exp}}}};\
             const b=fs.readFileSync({:?});\
             WebAssembly.instantiate(b,imp).then(r=>{{\
             const ex=r.instance.exports;const v=ex.main();\
             const dv=new DataView(ex.memory.buffer);\
             const len=Number(dv.getBigInt64(v+8,true));\
             const s=dec.decode(new Uint8Array(ex.memory.buffer).subarray(v+16,v+16+len));\
             process.stdout.write(s+'\\u0000'+String(ex.__live()));\
             }}).catch(e=>process.stdout.write('TRAP\\u00000'));",
            path.to_string_lossy()
        );
        let out = Command::new("node").arg("-e").arg(&script).output().map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&out.stdout).to_string();
        let (res, live) = s.split_once('\u{0}').ok_or("bad harness output")?;
        Ok((res.to_string(), live.parse::<i64>().unwrap_or(-1)))
    }

    /// Differential check for a String-returning program: the decoded wasm
    /// result must equal the interpreter's `display`, and exactly ONE heap
    /// object (the returned String) may remain live (garbage-free otherwise).
    fn differential_str(src: &str) {
        let interp = interp_result(src).expect("interpreter should succeed");
        let bytes = compile_src(src).expect("compile should succeed");
        if !node_available() {
            return;
        }
        let (wasm, live) = run_wasm_str_live(&bytes).expect("running wasm");
        assert_eq!(interp, wasm, "wasm != interpreter for:\n{}", src);
        assert_eq!(live, 1, "expected exactly the result String live, got {} in:\n{}", live, src);
    }

    #[test]
    fn string_literal_returned() {
        differential_str("fn main() -> String = \"hello, world\"");
    }

    #[test]
    fn string_concat_and_int_to_str() {
        differential_str("fn main() -> String = concat(\"hello, \", int_to_str(42))");
    }

    #[test]
    fn string_int_to_str_negative_and_zero() {
        differential_str("fn main() -> String = concat(int_to_str(0), int_to_str(-12345))");
        // i64::MIN must format correctly (the per-digit algorithm never negates
        // the whole value). MIN is built at runtime as (0 - i64::MAX) - 1, since
        // the literal `9223372036854775808` is out of range for the lexer.
        differential_str(
            "fn min() -> Int = (0 - 9223372036854775807) - 1\n\
             fn main() -> String = int_to_str(min())",
        );
    }

    #[test]
    fn string_concat_chain_is_garbage_free() {
        // Multiple concats build and free intermediate Strings; only the final
        // result remains live.
        differential_str(
            "fn main() -> String = concat(concat(\"a\", \"b\"), concat(int_to_str(7), \"z\"))",
        );
    }

    #[test]
    fn string_equality_matches_interpreter() {
        // String `==` / `!=`: equal, unequal (same len), and length-mismatch.
        differential(
            "fn main() -> Int = if \"abc\" == \"abc\" { 1 } else { 0 }",
        );
        differential(
            "fn main() -> Int = if \"abc\" == \"abd\" { 1 } else { 0 }",
        );
        differential(
            "fn main() -> Int = if \"abc\" != \"ab\" { 1 } else { 0 }",
        );
        // Through int_to_str, and the `!=` negation path.
        differential(
            "fn main() -> Int = if int_to_str(42) == \"42\" { 100 } else { 0 }",
        );
    }

    #[test]
    fn string_equality_is_garbage_free() {
        // After comparing two literal Strings, both temporaries must be freed
        // (the result is an unboxed Bool), so `__live` ends at 0.
        let src = "fn main() -> Int = if \"hello\" == \"hello\" { 1 } else { 0 }";
        let bytes = compile_src(src).expect("compile should succeed");
        if !node_available() {
            return;
        }
        let (wasm, live) = run_wasm_live(&bytes).expect("running wasm");
        assert_eq!(interp_result(src).unwrap(), wasm, "wasm != interpreter");
        assert_eq!(live, 0, "string == leaked: {} live", live);
    }

    #[test]
    fn string_field_in_adt_is_dropped() {
        // An ADT carrying a String field: matching it out and discarding the
        // cell must drop the String too, leaving the heap garbage-free.
        differential_heap(
            "type R = | R(String, Int)\n\
             fn main() -> Int = match R(concat(\"a\", \"b\"), 5) { R(s, n) => n, }",
        );
    }

    #[test]
    fn string_returned_from_branch() {
        // `main -> String` selected by a Bool, exercising the if-result String
        // type and the harness's String decode.
        differential_str(
            "fn pick(b: Bool) -> String = if b { \"yes\" } else { concat(\"n\", \"o\") }\n\
             fn main() -> String = pick(false)",
        );
    }

    #[test]
    fn emits_valid_module_header() {
        let bytes = compile_src("fn main() -> Int = 42").unwrap();
        assert_eq!(&bytes[0..8], &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
        if node_available() {
            assert_eq!(run_wasm(&bytes).unwrap(), "42");
        }
    }

    #[test]
    fn leb128_signed_roundtrips() {
        // Spot-check the hand-rolled signed LEB encoder against known vectors.
        let mut v = Vec::new();
        leb_s(0, &mut v);
        assert_eq!(v, [0x00]);
        v.clear();
        leb_s(-1, &mut v);
        assert_eq!(v, [0x7f]);
        v.clear();
        leb_s(63, &mut v);
        assert_eq!(v, [0x3f]);
        v.clear();
        leb_s(64, &mut v);
        assert_eq!(v, [0xc0, 0x00]);
        v.clear();
        leb_s(-64, &mut v);
        assert_eq!(v, [0x40]);
    }

    // ---- Phase 2b: heap-allocated ADTs ----------------------------------

    /// Run the compiled wasm under Node, returning `(main_result, live_cells)`.
    /// `live` is the value of the exported `__live()` after `main` returns.
    fn run_wasm_live(bytes: &[u8]) -> Result<(String, i64), String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "aria_wasm2b_{}_{}.wasm",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
        let script = format!(
            "const fs=require('fs');\
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}},exp:Math.exp}}}};\
             const b=fs.readFileSync({:?});\
             WebAssembly.instantiate(b,imp).then(r=>{{\
             const m=String(r.instance.exports.main());\
             const l=String(r.instance.exports.__live());\
             process.stdout.write(m+'|'+l);\
             }}).catch(e=>process.stdout.write('TRAP|0'));",
            path.to_string_lossy()
        );
        let out = Command::new("node")
            .arg("-e")
            .arg(&script)
            .output()
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&out.stdout).to_string();
        let (res, live) = s.split_once('|').ok_or("bad harness output")?;
        Ok((res.to_string(), live.parse::<i64>().unwrap_or(-1)))
    }

    /// Differential check for an Int-returning heap program: the compiled wasm
    /// result must equal the interpreter AND `__live()` must be 0 (garbage-free).
    fn differential_heap(src: &str) {
        let interp = interp_result(src).expect("interpreter should succeed");
        let bytes = compile_src(src).expect("compile should succeed");
        if !node_available() {
            return;
        }
        let (wasm, live) = run_wasm_live(&bytes).expect("running wasm");
        assert_eq!(interp, wasm, "wasm != interpreter for:\n{}", src);
        assert_eq!(live, 0, "leak: {} live cell(s) after main in:\n{}", live, src);
    }

    // ---- Generics via monomorphization ----------------------------------

    #[test]
    fn generic_option_get_at_int() {
        // A generic `Opt[T]` and a generic `get[T]`, both specialized to Int.
        // Nullary `None`'s type args are recovered from the param's expected
        // type. Result equals the interpreter and the heap ends garbage-free.
        differential_heap(
            "type Opt[T] = | None | Some(T)\n\
             fn get[T](o: Opt[T], d: T) -> T = match o { None => d, Some(x) => x, }\n\
             fn main() -> Int = get(Some(7), 0)",
        );
        differential_heap(
            "type Opt[T] = | None | Some(T)\n\
             fn or_else(o: Opt[Int], d: Int) -> Int = match o { None => d, Some(x) => x, }\n\
             fn main() -> Int = or_else(None, 99) + or_else(Some(5), 0)",
        );
    }

    #[test]
    fn generic_list_sum_and_length_at_int() {
        // Generic `List[T]` used at `List[Int]`: a monomorphic `sum` over the
        // concrete instantiation plus a generic `length[T]` specialized to Int.
        differential_heap(
            "type List[T] = | Nil | Cons(T, List[T])\n\
             fn sum(xs: List[Int]) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
             fn length[T](xs: List[T]) -> Int = match xs { Nil => 0, Cons(_, r) => 1 + length(r), }\n\
             fn main() -> Int = { let xs = Cons(1, Cons(2, Cons(3, Cons(4, Nil)))); sum(xs) + length(xs) }",
        );
    }

    #[test]
    fn generic_fn_instantiated_at_two_types() {
        // The SAME generic function/type instantiated at TWO concrete types in
        // one program (`List[Int]` and `List[Bool]`): each gets its own
        // specialization (length$Int / length$Bool over List$Int / List$Bool).
        differential_heap(
            "type List[T] = | Nil | Cons(T, List[T])\n\
             fn length[T](xs: List[T]) -> Int = match xs { Nil => 0, Cons(_, r) => 1 + length(r), }\n\
             fn main() -> Int = {\n\
               let xs = Cons(1, Cons(2, Nil));\n\
               let bs = Cons(true, Cons(false, Cons(true, Nil)));\n\
               length(xs) + length(bs)\n\
             }",
        );
    }

    #[test]
    fn generic_wrap_returns_specialized_type() {
        // A generic function returning `Option[T]`, instantiated at Int; the
        // result type is inferred from the call's argument.
        differential_heap(
            "type Option[T] = | None | Some(T)\n\
             fn wrap[T](x: T) -> Option[T] = Some(x)\n\
             fn unwrap(o: Option[Int], d: Int) -> Int = match o { None => d, Some(x) => x, }\n\
             fn main() -> Int = unwrap(wrap(42), 0)",
        );
    }

    #[test]
    fn generic_either_phantom_param_resolved_by_context() {
        // Either/Result shape: a constructor that does NOT mention every type
        // parameter (`Lft(A)` leaves `B` free, `Rgt(B)` leaves `A` free). At a
        // call `pick(Lft(5), 0, true)`, `pick`'s `B` is fixed by the third arg
        // (`db: B = true`), so `e: E[A, B]` is `E[Int, Bool]` and `Lft(5)` must
        // monomorphize at `E[Int, Bool]` — its free `B` recovered from the
        // call's resolved expected type, not from the constructor alone.
        differential_heap(
            "type E[A, B] = | Lft(A) | Rgt(B)\n\
             fn pick[A, B](e: E[A, B], da: A, db: B) -> A = match e { Lft(a) => a, Rgt(b) => da, }\n\
             fn main() -> Int = pick(Lft(5), 0, true) + pick(Rgt(true), 9, false)",
        );
    }

    #[test]
    fn generic_result_at_two_param_pairs() {
        // A `Result[T, E]` with `Ok(T)`/`Err(E)` used at two DIFFERENT `(T, E)`
        // pairs — `(Int, Bool)` and `(Int, String)`. Each call's free
        // parameter (the one its scrutinee constructor doesn't pin) is fixed by
        // a sibling argument (`dt: T`, `de: E`), so both `Ok` and `Err` get a
        // fully concrete owning type. Each specialization must agree with the
        // interpreter and end garbage-free.
        differential_heap(
            "type Result[T, E] = | Ok(T) | Err(E)\n\
             fn choose[T, E](r: Result[T, E], dt: T, de: E) -> T = match r { Ok(v) => v, Err(e) => dt, }\n\
             fn main() -> Int = choose(Ok(7), 0, false) + choose(Err(\"x\"), 100, \"y\")",
        );
    }

    #[test]
    fn generic_ctor_using_only_second_param() {
        // A constructor (`Second(B)`) that uses ONLY the second type parameter:
        // its owning type's first parameter `A` is left free by the field and
        // must come from context. `get_b(Second(42), true, 0)` pins `A` via
        // `da: A = true` and `B` via both the field and `db: B = 0`, giving
        // `Tagged[Bool, Int]`. Agrees with the interpreter and garbage-free.
        differential_heap(
            "type Tagged[A, B] = | First(A) | Second(B)\n\
             fn get_b[A, B](t: Tagged[A, B], da: A, db: B) -> B = match t { First(a) => db, Second(b) => b, }\n\
             fn main() -> Int = get_b(Second(42), true, 0) + get_b(First(false), true, 9)",
        );
    }

    #[test]
    fn generic_true_phantom_is_clean_err() {
        // A genuinely-unresolvable phantom: `A` is mentioned in `mk`'s signature
        // but used by NO constructor of `Phantom[A]` and fixed by no argument or
        // return context. Monomorphization must reject it with a clean `Err`
        // (not a panic), while the interpreter — which never needs `A` concrete
        // — still runs.
        let src = "type Phantom[A] = | Only(Int)\n\
                   fn mk[A]() -> Phantom[A] = Only(1)\n\
                   fn main() -> Int = match mk() { Only(n) => n, }";
        // Interpreter handles the unconstrained parameter dynamically.
        assert!(interp_result(src).is_ok());
        // The wasm pipeline must surface a clean error, not panic.
        match compile_src(src) {
            Err(msg) => assert!(
                msg.contains("could not infer type parameter"),
                "expected a clean phantom-parameter error, got: {}",
                msg
            ),
            Ok(_) => panic!("expected monomorphization to reject the true phantom"),
        }
    }

    #[test]
    fn heap_cons_list_sum() {
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
             fn main() -> Int = sum(Cons(1, Cons(2, Cons(3, Nil))))",
        );
    }

    #[test]
    fn heap_cons_list_length() {
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn len(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => 1 + len(r), }\n\
             fn main() -> Int = len(Cons(5, Cons(6, Cons(7, Cons(8, Nil)))))",
        );
    }

    #[test]
    fn heap_map_then_sum() {
        // Build a range, map +1 into a fresh list, then sum it: every cell freed.
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn range(n: Int, acc: L) -> L = if n == 0 { acc } else { range(n - 1, Cons(n, acc)) }\n\
             fn inc(xs: L) -> L = match xs { Nil => Nil, Cons(h, r) => Cons(h + 1, inc(r)), }\n\
             fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
             fn main() -> Int = sum(inc(range(20, Nil)))",
        );
    }

    #[test]
    fn heap_binary_tree_total() {
        differential_heap(
            "type T = | Leaf | Node(T, Int, T)\n\
             fn total(t: T) -> Int = match t { Leaf => 0, Node(l, v, r) => total(l) + v + total(r), }\n\
             fn main() -> Int = total(Node(Node(Leaf, 1, Leaf), 2, Node(Leaf, 3, Leaf)))",
        );
    }

    #[test]
    fn heap_ref_used_across_match() {
        // The scrutinee is matched AND used again afterward (a Ref field threaded
        // across a match): the match must borrow it, and it must still be freed.
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn len(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => 1 + len(r), }\n\
             fn f(xs: L) -> Int = { let n = match xs { Nil => 0, Cons(h, r) => h, }; n + len(xs) }\n\
             fn main() -> Int = f(Cons(7, Cons(8, Cons(9, Nil))))",
        );
    }

    #[test]
    fn heap_shared_reference() {
        // `xs` consumed twice -> requires a dup; all cells must net to freed.
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn len(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => 1 + len(r), }\n\
             fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
             fn use_twice(xs: L) -> Int = sum(xs) + len(xs)\n\
             fn main() -> Int = use_twice(Cons(10, Cons(20, Cons(30, Nil))))",
        );
    }

    #[test]
    fn heap_unused_value_is_freed() {
        // Built then never used -> must be dropped, leaving __live() == 0.
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn main() -> Int = { let tmp = Cons(1, Cons(2, Nil)); 7 }",
        );
    }

    #[test]
    fn heap_branch_only_uses_value_in_one_arm() {
        // Consumed in one branch, dropped in the other; garbage-free either way.
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn pick(b: Bool, xs: L) -> Int = if b { match xs { Nil => 0, Cons(h, r) => h, } } else { 99 }\n\
             fn main() -> Int = pick(false, Cons(5, Cons(6, Nil)))",
        );
    }

    // ---- Phase 2c: in-place cell REUSE (FBIP) ---------------------------

    /// Run the compiled wasm under Node, returning `(main_result, live, reuses)`
    /// from the exported `main`/`__live`/`__reuses`.
    fn run_wasm_reuses(bytes: &[u8]) -> Result<(String, i64, i64), String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "aria_wasm_reuse_{}_{}.wasm",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
        let script = format!(
            "const fs=require('fs');\
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}},exp:Math.exp}}}};\
             const b=fs.readFileSync({:?});\
             WebAssembly.instantiate(b,imp).then(r=>{{\
             const m=String(r.instance.exports.main());\
             const l=String(r.instance.exports.__live());\
             const u=String(r.instance.exports.__reuses());\
             process.stdout.write(m+'|'+l+'|'+u);\
             }}).catch(e=>process.stdout.write('TRAP|0|0'));",
            path.to_string_lossy()
        );
        let out = Command::new("node")
            .arg("-e")
            .arg(&script)
            .output()
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&out.stdout).to_string();
        let mut it = s.splitn(3, '|');
        let res = it.next().ok_or("bad harness output")?.to_string();
        let live = it.next().and_then(|x| x.parse::<i64>().ok()).unwrap_or(-1);
        let reuses = it.next().and_then(|x| x.parse::<i64>().ok()).unwrap_or(-1);
        Ok((res, live, reuses))
    }

    #[test]
    fn reuse_unique_list_map_reuses_cells_in_place() {
        // A unique list mapped element-wise (`inc`): the `match`/rebuild reuses
        // each consumed Cons cell in place. With a 30-element list, `inc` must
        // reuse >= 30 cells, the wasm result must equal the interpreter, and the
        // heap must end garbage-free (`__live == 0`).
        let src = "type L = | Nil | Cons(Int, L)\n\
                   fn rng(n: Int, a: L) -> L = if n == 0 { a } else { rng(n - 1, Cons(n, a)) }\n\
                   fn inc(xs: L) -> L = match xs { Nil => Nil, Cons(h, r) => Cons(h + 1, inc(r)), }\n\
                   fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
                   fn main() -> Int = sum(inc(rng(30, Nil)))";
        let interp = interp_result(src).expect("interpreter should succeed");
        let bytes = compile_src(src).expect("compile should succeed");
        if !node_available() {
            return;
        }
        let (wasm, live, reuses) = run_wasm_reuses(&bytes).expect("running wasm");
        assert_eq!(interp, wasm, "wasm != interpreter for unique-list map");
        assert_eq!(live, 0, "leak: {} live cell(s) after unique-list map", live);
        assert!(
            reuses >= 30,
            "expected >= 30 in-place reuses (list length), got {}",
            reuses
        );
    }

    #[test]
    fn reuse_shared_list_does_not_wrongly_reuse() {
        // `xs` is consumed twice (by `inc` AND `len`), so it is shared: the rc
        // pass dups it, `__drop_reuse` returns null while it is still referenced,
        // and `CtorReuse` must allocate fresh. The result must still match the
        // interpreter and the heap must end garbage-free.
        let src = "type L = | Nil | Cons(Int, L)\n\
                   fn rng(n: Int, a: L) -> L = if n == 0 { a } else { rng(n - 1, Cons(n, a)) }\n\
                   fn inc(xs: L) -> L = match xs { Nil => Nil, Cons(h, r) => Cons(h + 1, inc(r)), }\n\
                   fn len(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => 1 + len(r), }\n\
                   fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
                   fn use_shared(xs: L) -> Int = sum(inc(xs)) + len(xs)\n\
                   fn main() -> Int = use_shared(rng(10, Nil))";
        let interp = interp_result(src).expect("interpreter should succeed");
        let bytes = compile_src(src).expect("compile should succeed");
        if !node_available() {
            return;
        }
        let (wasm, live, _reuses) = run_wasm_reuses(&bytes).expect("running wasm");
        assert_eq!(interp, wasm, "wasm != interpreter for shared list");
        assert_eq!(live, 0, "leak: {} live cell(s) after shared-list use", live);
    }

    #[test]
    fn reuse_tree_map_is_garbage_free() {
        // A unique binary tree rebuilt node-for-node (same-arity `Node` reuse):
        // reuses fire for the interior nodes and the heap ends garbage-free.
        let src = "type T = | Leaf | Node(T, Int, T)\n\
                   fn build(n: Int) -> T = if n == 0 { Leaf } else { Node(build(n - 1), n, build(n - 1)) }\n\
                   fn inc(t: T) -> T = match t { Leaf => Leaf, Node(l, v, r) => Node(inc(l), v + 1, inc(r)), }\n\
                   fn total(t: T) -> Int = match t { Leaf => 0, Node(l, v, r) => total(l) + v + total(r), }\n\
                   fn main() -> Int = total(inc(build(4)))";
        let interp = interp_result(src).expect("interpreter should succeed");
        let bytes = compile_src(src).expect("compile should succeed");
        if !node_available() {
            return;
        }
        let (wasm, live, reuses) = run_wasm_reuses(&bytes).expect("running wasm");
        assert_eq!(interp, wasm, "wasm != interpreter for unique-tree map");
        assert_eq!(live, 0, "leak: {} live cell(s) after unique-tree map", live);
        assert!(reuses > 0, "expected in-place node reuse, got {}", reuses);
    }

    // ---- self-tail-call optimization (wasm) ----------------------------

    #[test]
    fn deep_tail_accumulator_wasm() {
        // 1,000,000-deep tail-recursive accumulator. Self-tail-call elimination
        // emits a wasm `loop` + `br` (param reassignment), so this runs in
        // constant wasm stack under Node's DEFAULT stack (no `--stack-size`) and
        // agrees with the interpreter (= 500000500000).
        differential(
            "fn go(n: Int, acc: Int) -> Int = if n == 0 { acc } else { go(n - 1, acc + n) }\n\
             fn main() -> Int = go(1000000, 0)",
        );
    }

    #[test]
    fn deep_tail_call_in_match_wasm() {
        // Self-tail-call in a `match` arm body, 1,000,000 deep, with a small flat
        // ADT scrutinee (each iteration allocates+frees one `Step` cell). Must
        // agree with the interpreter and end garbage-free (__live==0).
        differential_heap(
            "type Step = | Done | More(Int)\n\
             fn step(n: Int) -> Step = if n == 0 { Done } else { More(n) }\n\
             fn go(n: Int, acc: Int) -> Int = \
                match step(n) { Done => acc, More(k) => go(k - 1, acc + k), }\n\
             fn main() -> Int = go(1000000, 0)",
        );
    }

    #[test]
    fn heap_list_tail_recursion_is_garbage_free_wasm() {
        // Build then fold a cons-list; both functions are self-tail-recursive and
        // pass the HEAP list parameter through the tail call. The compiled module
        // reassigns that heap param under TCO and must end garbage-free. Depth is
        // modest because the interpreter oracle deep-clones the list (O(n^2)).
        differential_heap(
            "type L = | Nil | Cons(Int, L)\n\
             fn build(n: Int, acc: L) -> L = if n == 0 { acc } else { build(n - 1, Cons(n, acc)) }\n\
             fn length(xs: L, acc: Int) -> Int = \
                match xs { Nil => acc, Cons(_, r) => length(r, acc + 1), }\n\
             fn main() -> Int = length(build(300, Nil), 0)",
        );
    }

    #[test]
    fn tail_call_swapping_args_wasm() {
        // gcd by subtraction: the new tail-call args read the OTHER old param, so
        // all args must be computed before any param local is overwritten.
        differential(
            "fn gcd(a: Int, b: Int) -> Int = \
                if b == 0 { a } else { if a < b { gcd(b, a) } else { gcd(a - b, b) } }\n\
             fn main() -> Int = gcd(1071, 462)",
        );
    }
}
