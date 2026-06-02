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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WType {
    I64, // Aria Int
    I32, // Aria Bool
    F64, // Aria Float (unboxed IEEE-754 double)
    Ref, // pointer into linear memory (i32 address) — Phase 2b ADT cell
    Str, // pointer into linear memory (i32 address) — Phase 2d String object
}

impl WType {
    /// The valtype byte used in the binary (`0x7E` = i64, `0x7F` = i32,
    /// `0x7C` = f64). `Ref`/`Str` are wasm32 addresses, so they are i32.
    fn byte(self) -> u8 {
        match self {
            WType::I64 => 0x7E,
            WType::F64 => 0x7C,
            WType::I32 | WType::Ref | WType::Str => 0x7F,
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
            // A named ADT becomes a heap reference. (Generics — args present —
            // are out of the 2b subset and rejected below.)
            Ty::Named(_, args) if args.is_empty() => Ok(WType::Ref),
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

const MEM_PAGES: u64 = 256;
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
}

impl HeapHelper {
    const ALL: [HeapHelper; 15] = [
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
                // One branch is an unreachable Unit return; the other wins.
                (Ok(t), Err(_)) if is_unreachable_unit(els) => Ok(t),
                (Err(_), Ok(e)) if is_unreachable_unit(then) => Ok(e),
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
        };
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
    result.ok_or_else(|| "wasm backend: `match` with no arms".into())
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
            };
            iexpr_type(body, &probe)
        }
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
                    WType::I32 | WType::Ref | WType::Str => {
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
            // then-branch
            if is_unreachable_unit(then) {
                code.push(0x00); // unreachable (dead branch)
            } else {
                let tt = emit_iexpr(then, env, code)?;
                if tt != result_ty {
                    return Err("wasm backend: `if` then-branch type disagrees".into());
                }
            }
            code.push(0x05); // else
            // else-branch
            if is_unreachable_unit(els) {
                code.push(0x00); // unreachable (dead branch)
            } else {
                let et = emit_iexpr(els, env, code)?;
                if et != result_ty {
                    return Err("wasm backend: `if` else-branch type disagrees".into());
                }
            }
            code.push(0x0B); // end
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
            WType::I32 | WType::Ref | WType::Str => {
                code.push(0xAD); // i64.extend_i32_u
                i64_store(off, code);
            }
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
            emit_arm_body(scrut, arm, Some(&info), result_ty, env, code)?;
            code.push(0x05); // else
            emit_match_chain(scrut, arms, i + 1, result_ty, env, code)?;
            code.push(0x0B); // end
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
                    WType::I32 | WType::Ref | WType::Str => {
                        i64_load(off, code);
                        code.push(0xA7); // i32.wrap_i64
                    }
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
            // use `__dup_str` (both just bump the rc word at [ptr+0]).
            match env.var_type(v)? {
                WType::Ref | WType::Str => {
                    let h = if env.var_type(v)? == WType::Str {
                        HeapHelper::DupStr
                    } else {
                        HeapHelper::Dup
                    };
                    code.push(0x20); // local.get v
                    leb_u(env.var_index(v)? as u64, code);
                    code.push(0x10); // call __dup / __dup_str
                    leb_u(env.heap_index(h) as u64, code);
                    code.push(0x1A); // drop the dummy i32 result
                }
                _ => {}
            }
            emit_iexpr(body, env, code)
        }
        IExpr::Drop(v, body) => {
            match env.var_type(v)? {
                WType::Ref | WType::Str => {
                    let h = if env.var_type(v)? == WType::Str {
                        HeapHelper::DropStr
                    } else {
                        HeapHelper::Drop
                    };
                    code.push(0x20); // local.get v
                    leb_u(env.var_index(v)? as u64, code);
                    code.push(0x10); // call __drop / __drop_str
                    leb_u(env.heap_index(h) as u64, code);
                    code.push(0x1A); // drop the dummy i32 result
                }
                _ => {}
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
fn emit_heap_helper(h: HeapHelper, ctors: &CtorTable, heap_base: u32) -> Vec<u8> {
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
    }
}

fn collect_str_lits_bind(b: &Bind, out: &mut Vec<Vec<u8>>) {
    match b {
        Bind::Atom(a) | Bind::Unary(_, a) => collect_str_lits_atom(a, out),
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
    // We import four host functions: `env.print_str` (index 0),
    // `env.print_float` (index 1), `env.print_int` (index 2), and
    // `env.print_bool` (index 3). Every DEFINED function index is offset by
    // N_IMPORTS.
    const N_IMPORTS: u32 = 4;
    const PRINT_STR_IDX: u32 = 0;
    const PRINT_FLOAT_IDX: u32 = 1;
    const PRINT_INT_IDX: u32 = 2;
    const PRINT_BOOL_IDX: u32 = 3;

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
    let fns: HashMap<String, IFn> = crate::rc::insert_rc(&lowered);

    // Helper function indices come after the imports and every user function:
    //   [imports...] [user fns...] [overflow helpers x4] [heap helpers...]
    let ovf_base = N_IMPORTS + order.len() as u32;
    let heap_base = ovf_base + OvfHelper::ALL.len() as u32;

    // Collect the program's distinct String literals (their raw UTF-8 bytes).
    let mut lit_list: Vec<Vec<u8>> = Vec::new();
    for name in &order {
        if let Some(ifn) = fns.get(name) {
            collect_str_lits_iexpr(&ifn.body, &mut lit_list);
        }
    }

    // Read-only data region: the bookkeeping area (bump ptr + live + reuses +
    // free-list heads) rounded to 8, followed by the String-literal bytes.
    // String literals live BELOW the heap so their addresses never alias a cell.
    // The free-list array must cover every size class an allocation can request:
    // ADT arities 0..=max_arity AND String size classes 0..=STR_MAX_CLASS.
    let n_freelist_slots = (ctors.max_arity as u64).max(STR_MAX_CLASS) + 1;
    let ro_base = {
        let raw = FREELIST_BASE + 4 * n_freelist_slots;
        (raw + 7) & !7
    };
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
        };

        // Emit the body instructions; the result is left on the stack.
        let mut body_code = Vec::new();
        let result_ty = emit_iexpr(&ifn.body, &mut env, &mut body_code)?;
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

    // 3b. Append the checked-arithmetic helper functions, in `OvfHelper::ALL`
    //     order, so their indices line up with `ovf_base + offset`.
    for h in OvfHelper::ALL {
        type_section_funcs.push(h.sig());
        let entry = emit_ovf_helper(h);
        let mut sized = Vec::new();
        vec_bytes(&entry, &mut sized);
        code_entries.push(sized);
    }

    // 3c. Append the heap-runtime helpers (alloc/free/dup/drop/live), in
    //     `HeapHelper::ALL` order, so indices line up with `heap_base + offset`.
    for h in HeapHelper::ALL {
        type_section_funcs.push(h.sig());
        let entry = emit_heap_helper(h, &ctors, heap_base);
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
        // types 4..: the defined functions.
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
    // `env.print_float` (type 1), `env.print_int` (type 2), and
    // `env.print_bool` (type 3).
    {
        let mut content = Vec::new();
        leb_u(N_IMPORTS as u64, &mut content); // four imports
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
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}}}}}};\
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
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}}}}}};\
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
             print_bool:(b)=>{{process.stdout.write(b?'true':'false');process.stdout.write('\\n');}}}}}};\
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
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}}}}}};\
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
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}}}}}};\
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
             const imp={{env:{{print_str:(p,n)=>{{}},print_float:(x)=>{{}},print_int:(n)=>{{}},print_bool:(b)=>{{}}}}}};\
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
}
