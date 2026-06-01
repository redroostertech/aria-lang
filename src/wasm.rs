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
//! DEFERRED (clean `Err`, never a panic): Float / String / Unit, ADT fields of
//! those types, generic ADTs, in-place reuse (`CtorReuse`/`DropReuse` are
//! desugared to fresh `Ctor`/`Drop`), structural ADT `==`/`!=`, builtin calls.
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
    Ref, // pointer into linear memory (i32 address) — Phase 2b ADT cell
}

impl WType {
    /// The valtype byte used in the binary (`0x7E` = i64, `0x7F` = i32).
    /// `Ref` is a wasm32 address, so it is an i32 at the wasm level.
    fn byte(self) -> u8 {
        match self {
            WType::I64 => 0x7E,
            WType::I32 | WType::Ref => 0x7F,
        }
    }

    /// Map an AST type to a wasm value type. `Int -> i64`, `Bool -> i32`, and a
    /// *non-generic* named ADT type -> `Ref` (a heap pointer). Generic type
    /// variables, Float, String and Unit remain outside the 2b subset.
    fn from_ty(ty: &Ty) -> Result<WType, String> {
        match ty {
            Ty::Int => Ok(WType::I64),
            Ty::Bool => Ok(WType::I32),
            // A named ADT becomes a heap reference. (Generics — args present —
            // are out of the 2b subset and rejected below.)
            Ty::Named(_, args) if args.is_empty() => Ok(WType::Ref),
            other => Err(format!(
                "wasm backend: unsupported type `{:?}` (2b subset: Int/Bool and non-generic ADTs)",
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
    /// Indices of the fields that are heap references (must be dropped).
    fn ref_fields(&self) -> Vec<usize> {
        self.field_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| if *t == WType::Ref { Some(i) } else { None })
            .collect()
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
//   [4]  live-cell counter (i64): incremented on a fresh alloc, decremented on
//        free. Exported via `__live`.
//   [16] free-list heads, one i32 per arity 0..=max_arity (segregated by size).
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
const FREELIST_BASE: u64 = 16; // i32 per arity, 4 bytes each
const CELL_HEADER: u64 = 16; // rc(8) + tag(8)
const SLOT: u64 = 8;

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
    Alloc, // (nfields:i32) -> ptr:i32
    Free,  // (ptr:i32, nfields:i32) -> ()  [we give it an i32 dummy return]
    Dup,   // (ptr:i32) -> ()
    Drop,  // (ptr:i32) -> ()
    Live,  // () -> i64
}

impl HeapHelper {
    const ALL: [HeapHelper; 5] = [
        HeapHelper::Alloc,
        HeapHelper::Free,
        HeapHelper::Dup,
        HeapHelper::Drop,
        HeapHelper::Live,
    ];

    fn offset(self) -> u32 {
        match self {
            HeapHelper::Alloc => 0,
            HeapHelper::Free => 1,
            HeapHelper::Dup => 2,
            HeapHelper::Drop => 3,
            HeapHelper::Live => 4,
        }
    }

    /// `(params, ret)` wasm signature. `Free`/`Dup`/`Drop` are logically void;
    /// to keep the module's "exactly one result" invariant we make them return
    /// an i32 (a dummy 0), and the caller `drop`s the result.
    fn sig(self) -> (Vec<WType>, WType) {
        match self {
            HeapHelper::Alloc => (vec![WType::I32], WType::I32),
            HeapHelper::Free => (vec![WType::I32, WType::I32], WType::I32),
            HeapHelper::Dup => (vec![WType::I32], WType::I32),
            HeapHelper::Drop => (vec![WType::I32], WType::I32),
            HeapHelper::Live => (vec![], WType::I64),
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
        Atom::Float(_) => Err("wasm backend: Float literals are outside the 2a subset".into()),
        Atom::Str(_) => Err("wasm backend: String literals are outside the 2a subset".into()),
        Atom::Unit => Err("wasm backend: Unit is outside the 2a subset".into()),
    }
}

/// Infer the wasm result type of a `Bind` without emitting code.
fn bind_type(bind: &Bind, env: &LocalEnv) -> Result<WType, String> {
    match bind {
        Bind::Atom(a) => atom_type(a, env),
        Bind::Prim(op, _, _) => Ok(match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => WType::I64,
            // Comparisons and logical ops produce a Bool (i32).
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And
            | BinOp::Or => WType::I32,
        }),
        Bind::Unary(op, a) => match op {
            UnOp::Neg => atom_type(a, env), // numeric negation keeps the operand type
            UnOp::Not => Ok(WType::I32),    // Bool Not
        },
        Bind::Call(name, _) => {
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
        Bind::CtorReuse(..) => Err(
            "wasm backend: in-place constructor reuse (`CtorReuse`) is outside the 2b subset (deferred)"
                .into(),
        ),
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
            };
            iexpr_type(body, &probe)
        }
        // dup/drop don't introduce bindings; the type is the body's type.
        IExpr::Dup(_, body) | IExpr::Drop(_, body) => iexpr_type(body, env),
        IExpr::DropReuse(_, _, _) => Err(
            "wasm backend: reuse tokens (`DropReuse`) are outside the 2b subset (deferred)".into(),
        ),
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
        Atom::Float(_) => Err("wasm backend: Float literals are outside the 2a subset".into()),
        Atom::Str(_) => Err("wasm backend: String literals are outside the 2a subset".into()),
        Atom::Unit => Err("wasm backend: Unit is outside the 2a subset".into()),
    }
}

/// Emit a `Bind`, leaving its single result value on the operand stack.
fn emit_bind(bind: &Bind, env: &mut LocalEnv, code: &mut Vec<u8>) -> Result<WType, String> {
    match bind {
        Bind::Atom(a) => emit_atom(a, env, code),
        Bind::Prim(op, l, r) => {
            let lt = emit_atom(l, env, code)?;
            let rt = emit_atom(r, env, code)?;
            // Add/Sub/Mul on Int are checked: route through an emitted helper
            // that traps (`unreachable`) on signed-i64 overflow, matching the
            // interpreter. Everything else (Div/Mod/comparisons/logical) stays
            // inline. Div/Mod already trap natively on /0 and MIN/-1.
            match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul => {
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
                    WType::I32 | WType::Ref => {
                        Err("wasm backend: numeric negation requires an Int".into())
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
        Bind::CtorReuse(..) => Err(
            "wasm backend: in-place constructor reuse (`CtorReuse`) is outside the 2b subset (deferred)"
                .into(),
        ),
        Bind::Match(scrut, arms) => emit_match(scrut, arms, env, code),
    }
}

/// Push an atom and convert it to an i64 suitable for an 8-byte field slot.
/// Int stays i64; Bool (i32 0/1) and Ref (i32 address) are zero-extended.
fn emit_atom_as_slot(a: &Atom, env: &LocalEnv, code: &mut Vec<u8>) -> Result<(), String> {
    let t = emit_atom(a, env, code)?;
    match t {
        WType::I64 => {}
        WType::I32 | WType::Ref => code.push(0xAD), // i64.extend_i32_u
    }
    Ok(())
}

/// Emit a `Bind::Ctor`: allocate a cell, store its tag and fields, yield the
/// pointer (an i32 Ref) on the stack.
fn emit_ctor(
    name: &str,
    fields: &[Atom],
    env: &mut LocalEnv,
    code: &mut Vec<u8>,
) -> Result<WType, String> {
    let info = env.ctors.get(name)?.clone();
    if fields.len() != info.arity() {
        return Err(format!(
            "wasm backend: ctor `{}` got {} fields, expected {}",
            name,
            fields.len(),
            info.arity()
        ));
    }
    let scratch = env.scratch();
    let alloc = env.heap_index(HeapHelper::Alloc);
    // ptr = __alloc(nfields)
    code.push(0x41); // i32.const nfields
    leb_s(info.arity() as i64, code);
    code.push(0x10); // call __alloc
    leb_u(alloc as u64, code);
    code.push(0x21); // local.set scratch  (ptr)
    leb_u(scratch as u64, code);
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
        // Convert the operand to the field's i64 slot encoding.
        match fty {
            WType::I64 => {}
            WType::I32 | WType::Ref => code.push(0xAD), // i64.extend_i32_u
        }
        let off = CELL_HEADER + SLOT * i as u64;
        i64_store(off, code);
    }
    // result = ptr
    code.push(0x20); // local.get scratch
    leb_u(scratch as u64, code);
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
                emit_atom(scrut, env, code)?; // address
                i64_load(CELL_HEADER + SLOT * idx as u64, code); // raw i64 slot
                // Convert i64 slot -> field stack type.
                match fty {
                    WType::I64 => {}
                    WType::I32 | WType::Ref => code.push(0xA7), // i32.wrap_i64
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
            // dup is a no-op on non-Ref (unboxed) values, exactly like the IR
            // interpreter; only heap references are reference-counted.
            if env.var_type(v)? == WType::Ref {
                code.push(0x20); // local.get v
                leb_u(env.var_index(v)? as u64, code);
                code.push(0x10); // call __dup
                leb_u(env.heap_index(HeapHelper::Dup) as u64, code);
                code.push(0x1A); // drop the dummy i32 result
            }
            emit_iexpr(body, env, code)
        }
        IExpr::Drop(v, body) => {
            if env.var_type(v)? == WType::Ref {
                code.push(0x20); // local.get v
                leb_u(env.var_index(v)? as u64, code);
                code.push(0x10); // call __drop
                leb_u(env.heap_index(HeapHelper::Drop) as u64, code);
                code.push(0x1A); // drop the dummy i32 result
            }
            emit_iexpr(body, env, code)
        }
        IExpr::DropReuse(_, _, _) => Err(
            "wasm backend: reuse tokens (`DropReuse`) are outside the 2b subset (deferred)".into(),
        ),
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
                // recursively drop each Ref field
                for fi in info.ref_fields() {
                    body.push(LOCAL_GET);
                    leb_u(0, &mut body); // ptr
                    i64_load(CELL_HEADER + SLOT * fi as u64, &mut body);
                    body.push(I32_WRAP_I64); // field address
                    body.push(CALL);
                    leb_u(drop_idx as u64, &mut body);
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
    }
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

/// Rewrite the (compiler-internal) in-place-reuse IR nodes into their
/// allocation-based equivalents, so the 2b backend never sees them:
///   * `DropReuse(scrut, _tok, body)` -> `Drop(scrut, body)` (the token is
///     unused once reuse is disabled).
///   * `CtorReuse(_tok, name, fields)` -> `Ctor(name, fields)` (always allocate
///     fresh — exactly the empty-token fallback of the IR interpreter).
/// Semantics are preserved (reuse is an optimization); only allocations differ.
fn desugar_reuse(e: &IExpr) -> IExpr {
    match e {
        IExpr::Ret(a) => IExpr::Ret(a.clone()),
        IExpr::Dup(v, b) => IExpr::Dup(v.clone(), Box::new(desugar_reuse(b))),
        IExpr::Drop(v, b) => IExpr::Drop(v.clone(), Box::new(desugar_reuse(b))),
        IExpr::DropReuse(scrut, _tok, b) => {
            IExpr::Drop(scrut.clone(), Box::new(desugar_reuse(b)))
        }
        IExpr::Let(x, bind, body) => {
            IExpr::Let(x.clone(), desugar_reuse_bind(bind), Box::new(desugar_reuse(body)))
        }
    }
}

fn desugar_reuse_bind(bind: &Bind) -> Bind {
    match bind {
        Bind::CtorReuse(_tok, name, fields) => Bind::Ctor(name.clone(), fields.clone()),
        Bind::If(c, t, e) => Bind::If(
            c.clone(),
            Box::new(desugar_reuse(t)),
            Box::new(desugar_reuse(e)),
        ),
        Bind::Match(s, arms) => Bind::Match(
            s.clone(),
            arms.iter()
                .map(|a| ir::IArm {
                    ctor: a.ctor.clone(),
                    binders: a.binders.clone(),
                    body: desugar_reuse(&a.body),
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

// ---- top-level driver ----------------------------------------------------

/// Compile a type-checked `Program` to a WebAssembly binary (subset 2a).
/// Returns a clean `Err` for any feature outside the subset; never panics.
pub fn compile(program: &Program) -> Result<Vec<u8>, String> {
    // 1. Collect function signatures from the typed AST, in declaration order,
    //    assigning deterministic wasm function indices.
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
            let idx = order.len() as u32;
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
        if !m.params.is_empty() || m.ret != WType::I64 {
            return Err("wasm backend: `main` must take no params and return Int".into());
        }
    }

    // 1b. Build the ADT layout table (constructor tags + field wasm types).
    //     Rejects generic types / out-of-subset field types with a clean Err.
    let ctors = CtorTable::build(program)?;

    // 2. Lower the whole program to ANF IR, then insert reference-count
    //    operations (Phase 2b: the compiled module manages a real heap, so it
    //    needs the dup/drop the IR interpreter relies on for garbage-freeness).
    let lowered: HashMap<String, IFn> = ir::lower_program(program)?;
    let rced: HashMap<String, IFn> = crate::rc::insert_rc(&lowered);
    // In-place cell reuse (`CtorReuse`/`DropReuse`, FBIP) is a later phase. The
    // rc pass emits it opportunistically, but it is *semantically optional*:
    // `DropReuse` degrades to a plain `Drop` and `CtorReuse` to a fresh `Ctor`
    // (its empty-token fallback). Desugar them away so 2b stays correct and
    // garbage-free without the in-place optimization.
    let fns: HashMap<String, IFn> = rced
        .into_iter()
        .map(|(n, f)| {
            (
                n,
                IFn {
                    params: f.params,
                    body: desugar_reuse(&f.body),
                },
            )
        })
        .collect();

    // Helper function indices come after every user function:
    //   [user fns...] [overflow helpers x4] [heap helpers x5]
    let ovf_base = order.len() as u32;
    let heap_base = ovf_base + OvfHelper::ALL.len() as u32;

    // HEAP_BASE: cells are bump-allocated above the bookkeeping region (bump
    // pointer + live word + one free-list head i32 per arity 0..=max_arity),
    // rounded up to an 8-byte boundary.
    let heap_base_addr = {
        let raw = FREELIST_BASE + 4 * (ctors.max_arity as u64 + 1);
        (raw + 7) & !7
    };

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

    // Type section (id 1): one functype per function, in order.
    {
        let mut content = Vec::new();
        leb_u(type_section_funcs.len() as u64, &mut content);
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

    // Function section (id 3): type index per function (1:1 with type section).
    // Covers user functions *and* the appended overflow helpers.
    {
        let n = type_section_funcs.len();
        let mut content = Vec::new();
        leb_u(n as u64, &mut content);
        for i in 0..n {
            leb_u(i as u64, &mut content); // type index == function index here
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

    // Export section (id 7): `main`, `__live`, and `memory`.
    {
        let main_idx = fn_index["main"];
        let live_idx = heap_base + HeapHelper::Live.offset();
        let mut content = Vec::new();
        leb_u(3, &mut content); // three exports
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
        leb_u(1, &mut content); // one segment
        content.push(0x00); // flags=0: active, memory 0, i32.const offset
        content.push(0x41); // i32.const
        leb_s(BUMP_PTR_ADDR as i64, &mut content);
        content.push(0x0B); // end of offset expr
        let bytes = (heap_base_addr as u32).to_le_bytes();
        leb_u(bytes.len() as u64, &mut content);
        content.extend_from_slice(&bytes);
        section(11, &content, &mut out);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interp, ir, lexer, parser, typeck};
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
             try{{const b=fs.readFileSync({:?});\
             WebAssembly.instantiate(b).then(r=>{{\
             try{{process.stdout.write(String(r.instance.exports.main()));}}\
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
        // String result.
        let s1 = "fn main() -> String = concat(\"a\", \"b\")";
        // Float signature.
        let s3 = "fn main() -> Float = 3.5";
        // print_int builtin call.
        let s4 = "fn main() -> Int = { print_int(1); 0 }";
        // Unit-returning function.
        let s5 = "fn main() -> Unit = ()";
        // 2b-deferred: an ADT carrying a Float field (only Int/Bool/Ref fields).
        let s6 = "type B = | B(Float)\nfn main() -> Int = match B(1.5) { B(x) => 0, }";
        // 2b-deferred: an ADT carrying a String field.
        let s7 = "type R = | R(String, Int)\n\
                  fn main() -> Int = match R(\"a\", 1) { R(s, n) => n, }";
        // 2b-deferred: structural ADT equality (needs a recursive compare).
        let s8 = "type P = | P(Int, Int)\nfn main() -> Bool = P(1, 2) == P(1, 2)";
        for src in [s1, s3, s4, s5, s6, s7, s8] {
            let r = compile_src(src);
            assert!(r.is_err(), "expected Err (no panic) for:\n{}", src);
        }
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
             const b=fs.readFileSync({:?});\
             WebAssembly.instantiate(b).then(r=>{{\
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
}
