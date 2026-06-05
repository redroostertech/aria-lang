//! Aria's NATIVE backend: a C transpiler for the ANF IR (`src/ir.rs`).
//!
//! This backend mirrors the STRUCTURE of the WebAssembly emitter (`src/wasm.rs`)
//! but emits portable C source instead of a hand-encoded binary. The whole
//! existing pipeline is REUSED: `monomorphize::monomorphize` (strips generics),
//! `ir::lower_program` (typed AST -> ANF IR), and `rc::insert_rc` (inserts the
//! Perceus-style `dup`/`drop` + in-place reuse). Only the final codegen step is
//! new — and C is far easier than wasm (native recursion, `malloc`/`free`,
//! `printf`, no binary encoding).
//!
//! Runtime model (matches the IR interpreter and the wasm backend exactly):
//!   * Values: `Int -> int64_t`, `Bool -> int64_t` (0/1), `Float -> double`,
//!     `Ref`/`String -> void*` (a heap pointer).
//!   * ADT cell: `malloc`'d `{ int64_t rc; int64_t tag; int64_t fields[]; }`,
//!     one 64-bit slot per field — Int stored directly, Bool 0/1, Float via a
//!     bit-reinterpret union, Ref/Str as the pointer cast through `uintptr_t`.
//!     The backend knows each field's static type from the constructor table,
//!     so it casts correctly on store/load.
//!   * String object: `{ int64_t rc; int64_t len; char bytes[]; }`.
//!   * Allocator: `malloc` + a live-cell counter (`aria_live`) bumped on alloc,
//!     decremented on free — the native analogue of wasm's `__live`. A
//!     value-returning `main` is garbage-free iff `aria_live == 0` at exit.
//!   * `aria_dup`/`aria_drop`: drop decrements rc and, at 0, recursively drops
//!     the cell's Ref/Str fields per its tag, then `free`s and decrements the
//!     live counter. `aria_drop_reuse` returns the pointer if the cell became
//!     unique-and-dead (children dropped, slot retained) else NULL; `CtorReuse`
//!     overwrites that slot or allocs fresh.
//!   * Builtins: `aria_streq`, `aria_eq` (structural per-tag ADT equality),
//!     `aria_concat`, `aria_int_to_str`, and `print_int/bool/float/str`.
//!
//! Overflow: integer `+`/`-`/`*`/unary-`-` are checked with
//! `__builtin_*_overflow`; on overflow (or `/0`, `INT64_MIN / -1`) the program
//! calls `aria_trap` (writes "TRAP" then aborts), matching the interpreter's
//! `Err` and the wasm trap.
//!
//! Float formatting: `aria_print_float` (see `aria_fmt_float` in the runtime)
//! renders the SHORTEST decimal that round-trips to the same f64, in plain
//! notation — matching the interpreter's Rust `format!("{}", f)` and the wasm
//! backend, so all backends print identical float output.
//!
//! Out-of-subset features (tensors/RAG/compression builtins, Unit results) yield
//! a clean `Err` from `compile` — never a panic.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::ast::{BinOp, Item, Program, Ty, UnOp};
use crate::ir::{self, Atom, Bind, IExpr, IFn};

/// The concrete element kind of a native array, encoding both the slot storage
/// and whether elements are heap references (needing a recursive drop). This
/// mirrors the one-char suffix the monomorphizer attaches to array builtins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElemKind {
    Int,    // Int — stored directly in an int64 slot
    Bool,   // Bool — stored inline in the int64 slot like Int (no dup/drop), but
            // typed as Bool so `array_get` yields a usable Bool. Tagged `o`,
            // code 8 (aligned with `SlotKind::Bool`).
    Float,  // Float — f64 bit-reinterpreted into an int64 slot
    Str,    // heap String object (ref counted, recursive drop)
    Ref,    // boxed heap value: ADT cell / closure (recursive drop via aria_drop)
    Bytes,  // heap AriaBytes buffer (aria_bytes_drop)
    Array,  // heap AriaArray — a nested array (aria_array_drop)
    Map,    // heap AriaMap (aria_map_drop)
    Set,    // heap AriaSet (aria_set_drop)
    Vector, // heap AriaVector / embedding (aria_vec_drop)
}

impl ElemKind {
    /// Parse the monomorphizer's one-char element-kind suffix. The codes match
    /// `SlotKind`'s so array elements can route through the shared
    /// `aria_slot_dup`/`aria_slot_drop` runtime dispatch.
    fn from_tag(c: char) -> Result<ElemKind, String> {
        match c {
            'i' => Ok(ElemKind::Int),
            'o' => Ok(ElemKind::Bool),
            'f' => Ok(ElemKind::Float),
            's' => Ok(ElemKind::Str),
            'r' => Ok(ElemKind::Ref),
            'b' => Ok(ElemKind::Bytes),
            'a' => Ok(ElemKind::Array),
            'm' => Ok(ElemKind::Map),
            'e' => Ok(ElemKind::Set),
            'v' => Ok(ElemKind::Vector),
            other => Err(format!("c backend: bad array element-kind tag `{}`", other)),
        }
    }

    /// The header `kind` code stored in `AriaArray.kind` (drives the kind-aware
    /// runtime drop). These MATCH the `SlotKind` codes so the per-element dup/drop
    /// can be dispatched through `aria_slot_dup`/`aria_slot_drop`.
    fn code(self) -> i64 {
        match self {
            ElemKind::Int => 0,
            ElemKind::Float => 1,
            ElemKind::Str => 2,
            ElemKind::Ref => 3,
            ElemKind::Bytes => 4,
            ElemKind::Array => 5,
            ElemKind::Map => 6,
            ElemKind::Set => 7,
            ElemKind::Bool => 8,
            ElemKind::Vector => 9,
        }
    }

    /// The C value type of an element of this kind. For the heap-container kinds
    /// (nested Array, Map, Set) the inner element kind is not threaded through the
    /// one-char tag, so we report a COARSE inner kind (Ref); this only affects the
    /// static type of a retrieved container — its C declaration is `void*` either
    /// way, and the container's own runtime header carries the precise inner kind.
    /// The FLAT heap kinds (Bytes, Vector) round-trip exactly.
    fn elem_ctype(self) -> CType {
        match self {
            ElemKind::Int => CType::Int,
            ElemKind::Bool => CType::Bool,
            ElemKind::Float => CType::Float,
            ElemKind::Str => CType::Str,
            ElemKind::Ref => CType::Ref,
            ElemKind::Bytes => CType::Bytes,
            ElemKind::Vector => CType::Vector,
            ElemKind::Array => CType::Array(ElemKind::Ref),
            ElemKind::Map => CType::Map(SlotKind::Ref, SlotKind::Ref),
            ElemKind::Set => CType::Set(SlotKind::Ref),
        }
    }

    /// The element kind that holds a value of the given C value type. Each tagged
    /// heap type gets its OWN kind (so the per-element dup/drop uses the correct
    /// runtime function — no type confusion treating an AriaVector/AriaBytes as an
    /// AriaCell).
    fn from_ctype(ct: &CType) -> ElemKind {
        match ct {
            CType::Int => ElemKind::Int,
            CType::Bool => ElemKind::Bool,
            CType::Float => ElemKind::Float,
            CType::Str => ElemKind::Str,
            CType::Ref => ElemKind::Ref,
            CType::Bytes => ElemKind::Bytes,
            CType::Vector => ElemKind::Vector,
            CType::Array(_) => ElemKind::Array,
            CType::Map(..) => ElemKind::Map,
            CType::Set(_) => ElemKind::Set,
        }
    }
}

/// A richer per-slot kind for the Map/Set runtime: a map key/value or set
/// element may be ANY supported Aria type, and each needs the CORRECT recursive
/// drop (a Str, a Bytes buffer, an Array, an ADT cell, or a nested Map/Set use
/// different drop functions). The integer `code()` is stored in the AriaMap /
/// AriaSet header and drives the kind-aware runtime dup/drop. Map keys are only
/// ever `Int` or `Str` (the checker's restriction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotKind {
    Int,   // Int — stored directly, no drop
    Float, // Float — f64 bit-reinterpreted, no drop
    Str,   // AriaStr* — aria_str_drop
    Ref,   // ADT cell / closure — aria_drop
    Bytes, // AriaBytes* — aria_bytes_drop
    Array, // AriaArray* — aria_array_drop
    Map,   // AriaMap* — aria_map_drop
    Set,   // AriaSet* — aria_set_drop
    Bool,  // Bool — stored directly (0/1); distinct from Int so DISPLAY renders
           // `true`/`false` (matching the interpreter) rather than `1`/`0`.
    Vector, // AriaVector* — aria_vec_drop
}

impl SlotKind {
    fn code(self) -> i64 {
        match self {
            SlotKind::Int => 0,
            SlotKind::Float => 1,
            SlotKind::Str => 2,
            SlotKind::Ref => 3,
            SlotKind::Bytes => 4,
            SlotKind::Array => 5,
            SlotKind::Map => 6,
            SlotKind::Set => 7,
            SlotKind::Bool => 8,
            SlotKind::Vector => 9,
        }
    }
    fn from_ctype(ct: CType) -> SlotKind {
        match ct {
            CType::Int => SlotKind::Int,
            CType::Bool => SlotKind::Bool,
            CType::Float => SlotKind::Float,
            CType::Str => SlotKind::Str,
            CType::Ref => SlotKind::Ref,
            CType::Bytes => SlotKind::Bytes,
            CType::Array(_) => SlotKind::Array,
            CType::Map(..) => SlotKind::Map,
            CType::Set(_) => SlotKind::Set,
            CType::Vector => SlotKind::Vector,
        }
    }
    /// Encode an evaluated C expression of value type `t` into an int64 slot.
    fn encode(self, t: CType, ex: &str) -> String {
        match self {
            SlotKind::Int | SlotKind::Bool => format!("(int64_t)({})", ex),
            SlotKind::Float => format!("aria_f2i({})", ex),
            _ => {
                let _ = t;
                format!("(int64_t)(uintptr_t)({})", ex)
            }
        }
    }
    /// Decode an int64 slot back to a C value expression of the slot's ctype.
    fn decode(self, slot: &str) -> String {
        match self {
            SlotKind::Int | SlotKind::Bool => slot.to_string(),
            SlotKind::Float => format!("aria_i2f({})", slot),
            _ => format!("(void*)(uintptr_t)({})", slot),
        }
    }
}

/// A C-level value type for an Aria value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CType {
    Int,                 // int64_t
    Bool,                // int64_t (0/1)
    Float,               // double
    Ref,                 // void* — heap ADT cell
    Str,                 // void* — heap String object
    Bytes,               // void* — heap AriaBytes buffer (flat byte buffer)
    Array(ElemKind),     // void* — heap AriaArray of the given element kind
    Map(SlotKind, SlotKind), // void* — heap AriaMap (key kind, value kind)
    Set(SlotKind),       // void* — heap AriaSet (element kind)
    Vector,              // void* — heap AriaVector buffer (flat f64 vector / embedding)
}

impl CType {
    /// The C type name used to declare a local / parameter of this kind.
    fn decl(self) -> &'static str {
        match self {
            CType::Int | CType::Bool => "int64_t",
            CType::Float => "double",
            CType::Ref
            | CType::Str
            | CType::Bytes
            | CType::Array(_)
            | CType::Map(..)
            | CType::Set(_)
            | CType::Vector => "void*",
        }
    }

    /// Map an AST type to a C value type. Generic type variables and Unit are
    /// outside the subset (rejected with a clean Err).
    fn from_ty(ty: &Ty) -> Result<CType, String> {
        match ty {
            Ty::Int => Ok(CType::Int),
            Ty::Bool => Ok(CType::Bool),
            Ty::Float => Ok(CType::Float),
            Ty::Str => Ok(CType::Str),
            Ty::Named(n, args) if n == "Array" && args.len() == 1 => {
                let elem = CType::from_ty(&args[0])?;
                Ok(CType::Array(ElemKind::from_ctype(&elem)))
            }
            Ty::Named(n, _) if n == "Bytes" => Ok(CType::Bytes),
            // The opaque, reference-counted dense float vector / embedding.
            Ty::Named(n, _) if n == "Vector" => Ok(CType::Vector),
            Ty::Named(n, args) if n == "Map" && args.len() == 2 => {
                let k = SlotKind::from_ctype(CType::from_ty(&args[0])?);
                let v_ct = CType::from_ty(&args[1])?;
                // The native AriaMap stores each value under a COARSE slot-kind
                // tag with no nested type info, so its runtime get/show/== can
                // only faithfully handle FLAT value kinds. A non-flat value
                // (Array, tuple/ADT (Ref), nested Map/Set) would lose its element
                // layout on retrieval and render/compare by raw pointer —
                // diverging from the interpreter. Reject cleanly (the
                // interpreter supports these; this is a native limitation).
                if !matches!(
                    v_ct,
                    CType::Int | CType::Bool | CType::Float | CType::Str | CType::Bytes
                ) {
                    return Err(format!(
                        "c backend: a Map value of type `{}` is not yet supported \
                         (native maps support value types Int, Float, Bool, Str, Bytes); \
                         use the interpreter `aria run` for richer value types",
                        crate::typeck::show(&args[1])
                    ));
                }
                Ok(CType::Map(k, SlotKind::from_ctype(v_ct)))
            }
            Ty::Named(n, args) if n == "Set" && args.len() == 1 => {
                let e = SlotKind::from_ctype(CType::from_ty(&args[0])?);
                Ok(CType::Set(e))
            }
            Ty::Named(_, args) if args.is_empty() => Ok(CType::Ref),
            // A closure value is a heap cell (tag = lambda id, fields = captures).
            Ty::Fn(_, _) => Ok(CType::Ref),
            other => Err(format!(
                "c backend: unsupported type `{:?}` (subset: Int/Bool/Float/String, Array, and non-generic ADTs)",
                other
            )),
        }
    }
}

/// Parse a suffixed array-builtin name (`array_get$r`, `array_lit$f`, …) into its
/// base operation and concrete element kind. The UNSUFFIXED names never reach the
/// backend (they are interpreter/IR-only). Returns `None` if not an array builtin.
fn parse_array_builtin(name: &str) -> Option<(&'static str, ElemKind)> {
    let (base, tag) = name.rsplit_once('$')?;
    let kind = ElemKind::from_tag(tag.chars().next()?).ok()?;
    let base = match base {
        "array_new" => "array_new",
        "array_lit" => "array_lit",
        "array_len" => "array_len",
        "array_get" => "array_get",
        "array_set" => "array_set",
        "array_push" => "array_push",
        _ => return None,
    };
    Some((base, kind))
}

/// Parse a suffixed map/set builtin name (`map_insert$i_r`, `set_add$s`, …) into
/// its base operation. The element kinds are recovered from the operands' static
/// C types at emit time (which carry the PRECISE kind, unlike the coarse `$r`
/// name suffix). Returns `None` if not a map/set builtin.
fn parse_map_set_builtin(name: &str) -> Option<&'static str> {
    let (base, _suffix) = name.rsplit_once('$')?;
    match base {
        "map_new" => Some("map_new"),
        "map_insert" => Some("map_insert"),
        "map_get_or" => Some("map_get_or"),
        "map_has" => Some("map_has"),
        "map_len" => Some("map_len"),
        "map_remove" => Some("map_remove"),
        "map_show" => Some("map_show"),
        "map_keys" => Some("map_keys"),
        "map_values" => Some("map_values"),
        "set_new" => Some("set_new"),
        "set_add" => Some("set_add"),
        "set_has" => Some("set_has"),
        "set_len" => Some("set_len"),
        "set_remove" => Some("set_remove"),
        "set_show" => Some("set_show"),
        "set_to_array" => Some("set_to_array"),
        _ => None,
    }
}

/// A function's C-level signature, derived from the typed AST.
struct FnSig {
    params: Vec<CType>,
    ret: CType,
}

// ---- ADT / constructor metadata -----------------------------------------

#[derive(Debug, Clone)]
struct CtorInfo {
    tag: i64,
    field_types: Vec<CType>,
}

impl CtorInfo {
    fn arity(&self) -> usize {
        self.field_types.len()
    }
}

struct CtorTable {
    by_name: HashMap<String, CtorInfo>,
}

impl CtorTable {
    /// Build the program-wide constructor table from every `Item::Type`. After
    /// monomorphization there are no generic types; any out-of-subset field type
    /// (e.g. a generic var) yields a clean Err.
    fn build(program: &Program) -> Result<CtorTable, String> {
        let mut by_name = HashMap::new();
        let mut tag: i64 = 0;
        for item in &program.items {
            if let Item::Type(t) = item {
                if !t.params.is_empty() {
                    return Err(format!(
                        "c backend: generic type `{}` survived monomorphization",
                        t.name
                    ));
                }
                for v in &t.variants {
                    let field_types = v
                        .fields
                        .iter()
                        .map(CType::from_ty)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| format!("type `{}` ctor `{}`: {}", t.name, v.name, e))?;
                    by_name.insert(v.name.clone(), CtorInfo { tag, field_types });
                    tag += 1;
                }
            }
        }
        Ok(CtorTable { by_name })
    }

    fn get(&self, name: &str) -> Result<&CtorInfo, String> {
        self.by_name
            .get(name)
            .ok_or_else(|| format!("c backend: unknown constructor `{}`", name))
    }

    fn sorted(&self) -> Vec<(&String, &CtorInfo)> {
        let mut v: Vec<_> = self.by_name.iter().collect();
        v.sort_by_key(|(_, c)| c.tag);
        v
    }
}

// ---- C identifier sanitization ------------------------------------------

/// Map an Aria variable name (e.g. `$t3`, `x`) to a valid, collision-free C
/// identifier. Aria identifiers are `[A-Za-z_][A-Za-z0-9_]*`; the lowerer also
/// produces `$tN` temporaries. We prefix with `v_` and escape any non-alnum
/// byte, so distinct Aria names never collide.
fn cvar(name: &str) -> String {
    let mut s = String::from("v_");
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() || b == b'_' {
            s.push(b as char);
        } else {
            // Escape (e.g. `$` -> `_24`) so `$t0` and `_t0` can't collide.
            let _ = write!(s, "_{:02x}", b);
        }
    }
    s
}

/// Map an Aria function name to its emitted C function name.
fn cfn(name: &str) -> String {
    let mut s = String::from("af_");
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() || b == b'_' {
            s.push(b as char);
        } else {
            let _ = write!(s, "_{:02x}", b);
        }
    }
    s
}

// ---- per-function codegen environment -----------------------------------

struct Env<'a> {
    /// Static C type of every in-scope variable (params + let-bound).
    types: HashMap<String, CType>,
    sigs: &'a HashMap<String, FnSig>,
    ctors: &'a CtorTable,
    /// Monotone counter for fresh C temporaries (if/match result vars, etc.).
    tmp: usize,
    /// String-literal pool: bytes -> the C global array name holding them.
    str_lits: &'a HashMap<Vec<u8>, String>,
    /// The enclosing function's `(param_name, type)` list, in order, IFF that
    /// function is self-tail-recursive (its body was wrapped in a loop). A
    /// `TailCall` reassigns these and `goto`s the loop top. Empty otherwise.
    tail_params: &'a [(String, CType)],
    /// The enclosing function's declared return type. Used as the result type of
    /// an `if`/`match` ALL of whose branches diverge (every arm is a `TailCall`):
    /// the temp is never read, but C/wasm still need a type, and the function
    /// return type is the consistent choice (these are tail constructs).
    fn_ret: CType,
    /// Closure (lifted-lambda) table: maps each lambda's name to its closure
    /// tag and records the tag base, so `MakeClosure` can tag the cell and
    /// `ApplyClosure` can index the function-pointer table.
    closures: &'a ClosureTable,
}

/// Lifted-lambda dispatch metadata shared across the C codegen.
struct ClosureTable {
    /// First closure tag (one past the last constructor tag), so closure tags
    /// never collide with ADT constructor tags.
    base: i64,
    /// Lambda names in tag order; index `i` has tag `base + i` and lives at
    /// `__aria_lam_table[i]`.
    names: Vec<String>,
    /// Lambda name -> closure tag.
    tags: HashMap<String, i64>,
}

impl<'a> Env<'a> {
    fn fresh(&mut self) -> String {
        let n = self.tmp;
        self.tmp += 1;
        format!("t_{}", n)
    }

    fn var_type(&self, name: &str) -> Result<CType, String> {
        self.types
            .get(name)
            .copied()
            .ok_or_else(|| format!("c backend: unbound variable `{}`", name))
    }
}

// ---- atom / type inference ----------------------------------------------

fn atom_type(a: &Atom, env: &Env) -> Result<CType, String> {
    match a {
        Atom::Int(_) => Ok(CType::Int),
        Atom::Bool(_) => Ok(CType::Bool),
        Atom::Float(_) => Ok(CType::Float),
        Atom::Str(_) => Ok(CType::Str),
        Atom::Var(n) => env.var_type(n),
        Atom::Unit => Err("c backend: Unit value is outside the subset".into()),
    }
}

/// Render an atom as a C expression (no side effects). For string literals this
/// MATERIALIZES a fresh String object via `aria_str_lit` — but materialization
/// is effectful (alloc), so string-literal atoms must be emitted through
/// `emit_atom_stmt` when they need a stable binding. Here we only handle the
/// pure cases; callers that may see a `Str` literal use `emit_atom`.
fn atom_expr(a: &Atom, _env: &Env) -> Result<String, String> {
    match a {
        Atom::Int(n) => Ok(format!("INT64_C({})", n)),
        Atom::Bool(b) => Ok(if *b { "1".into() } else { "0".into() }),
        Atom::Float(f) => Ok(c_float_lit(*f)),
        Atom::Var(n) => Ok(cvar(n)),
        Atom::Str(_) => Err("c backend: string literal needs statement context".into()),
        Atom::Unit => Err("c backend: Unit value is outside the subset".into()),
    }
}

/// Emit an atom as a C expression, materializing a string literal if needed.
/// Returns the C expression text and the atom's type.
fn emit_atom(a: &Atom, env: &Env, _out: &mut String) -> Result<(CType, String), String> {
    if let Atom::Str(s) = a {
        let g = env
            .str_lits
            .get(s.as_bytes())
            .ok_or("c backend: string literal missing from pool (internal)")?;
        // aria_str_lit copies the bytes into a fresh rc=1 String object.
        return Ok((
            CType::Str,
            format!("aria_str_lit({}, {})", g, s.as_bytes().len()),
        ));
    }
    Ok((atom_type(a, env)?, atom_expr(a, env)?))
}

/// A finite, parseable C double literal. Handles the non-finite cases that have
/// no portable literal syntax.
fn c_float_lit(f: f64) -> String {
    if f.is_nan() {
        "(0.0/0.0)".into()
    } else if f.is_infinite() {
        if f < 0.0 { "(-1.0/0.0)".into() } else { "(1.0/0.0)".into() }
    } else {
        // 17 significant digits round-trips an f64 exactly.
        format!("{:.17e}", f)
    }
}

/// Infer the C result type of a `Bind` without emitting code (mirrors wasm's
/// `bind_type`).
fn bind_type(bind: &Bind, env: &Env) -> Result<CType, String> {
    match bind {
        Bind::Atom(a) => atom_type(a, env),
        Bind::Prim(op, l, _) => Ok(match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => atom_type(l, env)?,
            _ => CType::Bool,
        }),
        Bind::Unary(op, a) => match op {
            UnOp::Neg => atom_type(a, env),
            UnOp::Not => Ok(CType::Bool),
        },
        Bind::Call(name, args) => {
            if let Some((base, kind)) = parse_array_builtin(name) {
                return Ok(array_builtin_ret(base, kind));
            }
            if let Some(base) = parse_map_set_builtin(name) {
                return map_set_builtin_ret(base, name, args, env);
            }
            if let Some(t) = builtin_ret(name) {
                return Ok(t);
            }
            env.sigs
                .get(name)
                .map(|s| s.ret)
                .ok_or_else(|| format!("c backend: call to unknown fn `{}`", name))
        }
        Bind::Ctor(name, _) | Bind::CtorReuse(_, name, _) => {
            env.ctors.get(name)?;
            Ok(CType::Ref)
        }
        // A closure value is a heap cell.
        Bind::MakeClosure(_, _) => Ok(CType::Ref),
        // The result of applying a closure is the lambda's return type, attached
        // by monomorphization.
        Bind::ApplyClosure(_, _, ret) => match ret {
            Some(t) => CType::from_ty(t),
            None => Err("c backend: closure application missing its result type".into()),
        },
        Bind::If(_, then, els) => {
            let t = iexpr_type(then, env);
            let e = iexpr_type(els, env);
            match (t, e) {
                (Ok(t), Ok(e)) if t == e => Ok(t),
                (Ok(t), _) if is_diverging(els) => Ok(t),
                (_, Ok(e)) if is_diverging(then) => Ok(e),
                // Both branches diverge (e.g. each arm is a `TailCall`): the temp
                // is never read; use the function return type for consistency.
                (_, _) if is_diverging(then) && is_diverging(els) => Ok(env.fn_ret),
                (Ok(t), Ok(e)) => Err(format!(
                    "c backend: if-branches differ in type ({:?} vs {:?})",
                    t, e
                )),
                (Err(err), _) => Err(err),
                (_, Err(err)) => Err(err),
            }
        }
        Bind::Match(scrut, arms) => match_type(scrut, arms, env),
    }
}

fn match_type(scrut: &Atom, arms: &[ir::IArm], env: &Env) -> Result<CType, String> {
    if atom_type(scrut, env)? != CType::Ref {
        return Err("c backend: `match` scrutinee must be an ADT (Ref)".into());
    }
    let mut result: Option<CType> = None;
    for arm in arms {
        // Extend a probe env with this arm's binders.
        let mut types = env.types.clone();
        if let Some(cname) = &arm.ctor {
            let info = env.ctors.get(cname)?;
            for (b, fty) in arm.binders.iter().zip(info.field_types.iter()) {
                types.insert(b.clone(), *fty);
            }
        } else if let Some(b) = arm.binders.first() {
            types.insert(b.clone(), CType::Ref);
        }
        let probe = Env { types, sigs: env.sigs, ctors: env.ctors, tmp: env.tmp, str_lits: env.str_lits, tail_params: env.tail_params, fn_ret: env.fn_ret, closures: env.closures };
        match iexpr_type(&arm.body, &probe) {
            Ok(t) => {
                if let Some(prev) = result {
                    if prev != t {
                        return Err(format!(
                            "c backend: match arms differ in type ({:?} vs {:?})",
                            prev, t
                        ));
                    }
                } else {
                    result = Some(t);
                }
            }
            Err(_) if is_diverging(&arm.body) => {}
            Err(e) => return Err(e),
        }
    }
    // Every arm diverged (each a `TailCall`): the temp is never read; use the
    // function return type for consistency.
    Ok(result.unwrap_or(env.fn_ret))
}

/// True when an IExpr is the IR's dead `Ret(Unit)` fall-through marker (the
/// only genuinely unreachable branch — codegen emits a trap for it).
fn is_unreachable_unit(e: &IExpr) -> bool {
    match e {
        IExpr::Ret(Atom::Unit) => true,
        IExpr::Dup(_, b) | IExpr::Drop(_, b) | IExpr::DropReuse(_, _, b) => is_unreachable_unit(b),
        _ => false,
    }
}

/// True when an IExpr does NOT yield a value at this position: it is the dead
/// `Ret(Unit)` marker OR it ends in a `TailCall` (a loop back-edge that
/// re-enters the function). Such a branch imposes no type constraint on a
/// sibling `if`/`match` arm, so TYPE inference takes the type from the
/// value-producing arm. (Codegen still distinguishes the two: a TailCall
/// emits the loop back-edge, only the dead marker emits a trap.)
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

fn iexpr_type(e: &IExpr, env: &Env) -> Result<CType, String> {
    match e {
        IExpr::Ret(a) => atom_type(a, env),
        IExpr::Let(x, bind, body) => {
            let t = bind_type(bind, env)?;
            let mut types = env.types.clone();
            types.insert(x.clone(), t);
            let probe = Env { types, sigs: env.sigs, ctors: env.ctors, tmp: env.tmp, str_lits: env.str_lits, tail_params: env.tail_params, fn_ret: env.fn_ret, closures: env.closures };
            iexpr_type(body, &probe)
        }
        IExpr::Dup(_, body) | IExpr::Drop(_, body) => iexpr_type(body, env),
        IExpr::DropReuse(_, tok, body) => {
            let mut types = env.types.clone();
            types.insert(tok.clone(), CType::Ref); // a reuse token is a void* (cell ptr or NULL)
            let probe = Env { types, sigs: env.sigs, ctors: env.ctors, tmp: env.tmp, str_lits: env.str_lits, tail_params: env.tail_params, fn_ret: env.fn_ret, closures: env.closures };
            iexpr_type(body, &probe)
        }
        // A `TailCall` is a loop back-edge: it yields no value here. Report it as
        // "no type"; sibling-branch inference treats it like a diverging branch
        // (`is_unreachable_unit`) and takes the type from the value-producing arm.
        IExpr::TailCall(_) => Err("c backend: TailCall has no value type".into()),
    }
}

// ---- builtins -----------------------------------------------------------

/// The builtins the C backend implements inline. Returns the C result type, or
/// `None` if `name` is not a supported builtin.
fn builtin_ret(name: &str) -> Option<CType> {
    match name {
        "concat" | "int_to_str" | "bytes_to_str" => Some(CType::Str),
        // print_* are logically Unit; we model them as Int 0 (never used).
        "print_int" | "print_bool" | "print_float" | "print_str" => Some(CType::Int),
        // Bytes builtins (non-generic; no element-kind suffix).
        "bytes_new" | "bytes_set" | "bytes_push" | "bytes_from_str" => Some(CType::Bytes),
        "bytes_len" | "bytes_get" => Some(CType::Int),
        // Vector / embedding builtins (non-generic; no element-kind suffix).
        "vec_new" | "vec_from_array" | "vec_push" | "vec_add" | "vec_scale" => {
            Some(CType::Vector)
        }
        "vec_to_array" => Some(CType::Array(ElemKind::Float)),
        "vec_len" => Some(CType::Int),
        "vec_get" | "vec_dot" | "vec_norm" | "vec_cosine" => Some(CType::Float),
        _ => None,
    }
}

fn is_builtin(name: &str) -> bool {
    builtin_ret(name).is_some()
        || parse_array_builtin(name).is_some()
        || parse_map_set_builtin(name).is_some()
}

/// A COARSE `SlotKind` from a single element-kind tag char (`i`/`f`/`s`/`r`).
/// The monomorphizer collapses all heap-ref types to `r`, so this returns
/// `SlotKind::Ref` for any non-scalar — the PRECISE kind is recovered from the
/// operand's static `CType` at emit time. Used only where no operand is
/// available (`map_new`/`set_new`'s header kinds, harmless as they hold no
/// elements until reconciled on first insert).
fn slot_kind_from_tag(c: char) -> SlotKind {
    match c {
        'i' => SlotKind::Int,
        'f' => SlotKind::Float,
        's' => SlotKind::Str,
        'b' => SlotKind::Bytes,
        'v' => SlotKind::Vector,
        'a' => SlotKind::Array,
        'm' => SlotKind::Map,
        'e' => SlotKind::Set,
        'o' => SlotKind::Bool,
        _ => SlotKind::Ref,
    }
}

/// Parse the `$<keytag>_<valtag>` (map) or `$<elemtag>` (set) name suffix into
/// coarse (key, value) / (element) slot kinds.
fn map_set_suffix_kinds(name: &str) -> (SlotKind, SlotKind) {
    let suffix = name.rsplit_once('$').map(|(_, s)| s).unwrap_or("");
    match suffix.split_once('_') {
        Some((k, v)) => (
            slot_kind_from_tag(k.chars().next().unwrap_or('i')),
            slot_kind_from_tag(v.chars().next().unwrap_or('i')),
        ),
        None => {
            let e = slot_kind_from_tag(suffix.chars().next().unwrap_or('i'));
            (e, e)
        }
    }
}

/// The C result type of a map/set builtin. Container-producing ops recover the
/// precise key/value (element) kinds from their operand's static `CType` when
/// available (a `map_*`/`set_*` op takes the container as arg 0), falling back to
/// the coarse name-suffix kinds for the empty constructors. `map_get_or` yields
/// the value type; the predicates/`*_len` yield Bool/Int.
fn map_set_builtin_ret(base: &str, name: &str, args: &[Atom], env: &Env) -> Result<CType, String> {
    let (sk, sv) = map_set_suffix_kinds(name);
    // Recover precise container kinds from the first argument when present.
    let container_kinds = |env: &Env| -> Option<(SlotKind, SlotKind)> {
        match args.first().map(|a| atom_type(a, env)) {
            Some(Ok(CType::Map(k, v))) => Some((k, v)),
            Some(Ok(CType::Set(e))) => Some((e, e)),
            _ => None,
        }
    };
    match base {
        "map_new" => Ok(CType::Map(sk, sv)),
        "map_insert" => {
            // Precise value kind from the inserted value (arg 2); precise key
            // kind from arg 1.
            let k = match args.get(1).map(|a| atom_type(a, env)) {
                Some(Ok(t)) => SlotKind::from_ctype(t),
                _ => sk,
            };
            let v = match args.get(2).map(|a| atom_type(a, env)) {
                Some(Ok(t)) => SlotKind::from_ctype(t),
                _ => sv,
            };
            Ok(CType::Map(k, v))
        }
        "map_remove" => {
            let (k, v) = container_kinds(env).unwrap_or((sk, sv));
            Ok(CType::Map(k, v))
        }
        "map_get_or" => {
            // The value type: prefer the default (arg 2)'s precise type.
            match args.get(2).map(|a| atom_type(a, env)) {
                Some(Ok(t)) => Ok(t),
                _ => Ok(slot_kind_ctype(sv)),
            }
        }
        "map_has" | "set_has" => Ok(CType::Bool),
        "map_len" | "set_len" => Ok(CType::Int),
        "map_show" | "set_show" => Ok(CType::Str),
        // Enumeration into an Array. The element kind is the map key (`map_keys`),
        // map value (`map_values`), or set element (`set_to_array`), recovered
        // precisely from the container's static type (falling back to the coarse
        // name-suffix kind for an empty constructor with no header info).
        "map_keys" => {
            let (k, _) = container_kinds(env).unwrap_or((sk, sv));
            Ok(CType::Array(ElemKind::from_ctype(&slot_kind_ctype(k))))
        }
        "map_values" => {
            let (_, v) = container_kinds(env).unwrap_or((sk, sv));
            Ok(CType::Array(ElemKind::from_ctype(&slot_kind_ctype(v))))
        }
        "set_to_array" => {
            let (e, _) = container_kinds(env).unwrap_or((sk, sv));
            Ok(CType::Array(ElemKind::from_ctype(&slot_kind_ctype(e))))
        }
        "set_new" => Ok(CType::Set(sk)),
        "set_add" => {
            let e = match args.get(1).map(|a| atom_type(a, env)) {
                Some(Ok(t)) => SlotKind::from_ctype(t),
                _ => sk,
            };
            Ok(CType::Set(e))
        }
        "set_remove" => {
            let (e, _) = container_kinds(env).unwrap_or((sk, sv));
            Ok(CType::Set(e))
        }
        _ => Err(format!("c backend: unknown map/set builtin `{}`", base)),
    }
}

/// A representative `CType` for a `SlotKind` (used when no operand pins the
/// precise type — only the scalar/Str/Ref distinction matters there).
fn slot_kind_ctype(k: SlotKind) -> CType {
    match k {
        SlotKind::Int => CType::Int,
        SlotKind::Bool => CType::Bool,
        SlotKind::Float => CType::Float,
        SlotKind::Str => CType::Str,
        SlotKind::Bytes => CType::Bytes,
        SlotKind::Array => CType::Array(ElemKind::Ref),
        SlotKind::Map => CType::Map(SlotKind::Ref, SlotKind::Ref),
        SlotKind::Set => CType::Set(SlotKind::Ref),
        SlotKind::Vector => CType::Vector,
        SlotKind::Ref => CType::Ref,
    }
}

/// The C result type of an array builtin: builders yield an `Array(kind)`,
/// `array_get` the element ctype, `array_len` an Int.
fn array_builtin_ret(base: &str, kind: ElemKind) -> CType {
    match base {
        "array_new" | "array_lit" | "array_set" | "array_push" => CType::Array(kind),
        "array_get" => kind.elem_ctype(),
        "array_len" => CType::Int,
        _ => CType::Array(kind),
    }
}

// ---- code generation -----------------------------------------------------

/// Emit the C statements for an `IExpr`, writing into `out` at indent `ind`.
/// `result` names a (pre-declared) C lvalue to assign the IExpr's value to, and
/// `result_ty` is its type. The caller declares `result` so that `if`/`match`
/// (statements in C) can assign into it from each branch.
fn emit_iexpr(
    e: &IExpr,
    result: &str,
    result_ty: CType,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    match e {
        IExpr::Ret(a) => {
            let (_, ex) = emit_atom(a, env, out)?;
            let _ = writeln!(out, "{}{} = {};", ind, result, ex);
            Ok(())
        }
        IExpr::Let(x, bind, body) => {
            let t = bind_type(bind, env)?;
            let cv = cvar(x);
            let _ = writeln!(out, "{}{} {};", ind, t.decl(), cv);
            emit_bind(bind, &cv, t, env, ind, out)?;
            env.types.insert(x.clone(), t);
            emit_iexpr(body, result, result_ty, env, ind, out)
        }
        IExpr::Dup(v, body) => {
            match env.var_type(v)? {
                CType::Ref => {
                    let _ = writeln!(out, "{}aria_dup({});", ind, cvar(v));
                }
                CType::Str => {
                    let _ = writeln!(out, "{}aria_str_dup({});", ind, cvar(v));
                }
                CType::Array(_) => {
                    let _ = writeln!(out, "{}aria_array_dup({});", ind, cvar(v));
                }
                CType::Bytes => {
                    let _ = writeln!(out, "{}aria_bytes_dup({});", ind, cvar(v));
                }
                CType::Map(..) => {
                    let _ = writeln!(out, "{}aria_map_dup({});", ind, cvar(v));
                }
                CType::Set(_) => {
                    let _ = writeln!(out, "{}aria_set_dup({});", ind, cvar(v));
                }
                CType::Vector => {
                    let _ = writeln!(out, "{}aria_vec_dup({});", ind, cvar(v));
                }
                _ => {}
            }
            emit_iexpr(body, result, result_ty, env, ind, out)
        }
        IExpr::Drop(v, body) => {
            match env.var_type(v)? {
                CType::Ref => {
                    let _ = writeln!(out, "{}aria_drop({});", ind, cvar(v));
                }
                CType::Str => {
                    let _ = writeln!(out, "{}aria_str_drop({});", ind, cvar(v));
                }
                CType::Array(_) => {
                    let _ = writeln!(out, "{}aria_array_drop({});", ind, cvar(v));
                }
                CType::Bytes => {
                    let _ = writeln!(out, "{}aria_bytes_drop({});", ind, cvar(v));
                }
                CType::Map(..) => {
                    let _ = writeln!(out, "{}aria_map_drop({});", ind, cvar(v));
                }
                CType::Set(_) => {
                    let _ = writeln!(out, "{}aria_set_drop({});", ind, cvar(v));
                }
                CType::Vector => {
                    let _ = writeln!(out, "{}aria_vec_drop({});", ind, cvar(v));
                }
                _ => {}
            }
            emit_iexpr(body, result, result_ty, env, ind, out)
        }
        IExpr::DropReuse(scrut, tok, body) => {
            let tv = cvar(tok);
            if env.var_type(scrut)? == CType::Ref {
                let _ = writeln!(out, "{}void* {} = aria_drop_reuse({});", ind, tv, cvar(scrut));
            } else {
                let _ = writeln!(out, "{}void* {} = NULL;", ind, tv);
            }
            env.types.insert(tok.clone(), CType::Ref);
            emit_iexpr(body, result, result_ty, env, ind, out)
        }
        IExpr::TailCall(args) => {
            // Self-tail-call -> loop back-edge. Evaluate ALL new argument atoms
            // into fresh temporaries FIRST (a parameter may appear in another
            // argument), then overwrite the parameters and jump to the loop top.
            // Ownership of each new arg transfers to its parameter exactly as a
            // real call would bind it (rc-balanced: the rc pass already dropped
            // any parameter the args do not reuse).
            let params: Vec<(String, CType)> = env.tail_params.to_vec();
            if params.len() != args.len() {
                return Err("c backend: TailCall arity mismatch (internal)".into());
            }
            let mut temps = Vec::with_capacity(args.len());
            for (a, (_, pty)) in args.iter().zip(params.iter()) {
                let (_, ex) = emit_atom(a, env, out)?;
                let tn = env.fresh();
                let _ = writeln!(out, "{}{} {} = {};", ind, pty.decl(), cvar(&tn), ex);
                temps.push(tn);
            }
            for ((pname, _), tn) in params.iter().zip(temps.iter()) {
                let _ = writeln!(out, "{}{} = {};", ind, cvar(pname), cvar(tn));
            }
            let _ = writeln!(out, "{}goto aria_loop_top;", ind);
            Ok(())
        }
    }
}

/// Emit the C statements for a `Bind`, assigning its value to the (already
/// declared) lvalue `dst` of type `dst_ty`.
fn emit_bind(
    bind: &Bind,
    dst: &str,
    dst_ty: CType,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    match bind {
        Bind::Atom(a) => {
            let (_, ex) = emit_atom(a, env, out)?;
            let _ = writeln!(out, "{}{} = {};", ind, dst, ex);
            Ok(())
        }
        Bind::Prim(op, l, r) => emit_prim(*op, l, r, dst, env, ind, out),
        Bind::Unary(op, a) => emit_unary(*op, a, dst, env, ind, out),
        Bind::Call(name, args) => emit_call(name, args, dst, dst_ty, env, ind, out),
        Bind::Ctor(name, fields) => emit_ctor(name, fields, dst, env, ind, out),
        Bind::CtorReuse(tok, name, fields) => emit_ctor_reuse(tok, name, fields, dst, env, ind, out),
        Bind::If(c, then, els) => {
            let (ct, cx) = emit_atom(c, env, out)?;
            if ct != CType::Bool {
                return Err("c backend: `if` condition must be a Bool".into());
            }
            let _ = writeln!(out, "{}if ({}) {{", ind, cx);
            let inner = format!("{}    ", ind);
            if is_unreachable_unit(then) {
                let _ = writeln!(out, "{}aria_trap_msg(\"unreachable expression evaluated\");", inner);
            } else {
                emit_iexpr(then, dst, dst_ty, env, &inner, out)?;
            }
            let _ = writeln!(out, "{}}} else {{", ind);
            if is_unreachable_unit(els) {
                let _ = writeln!(out, "{}aria_trap_msg(\"unreachable expression evaluated\");", inner);
            } else {
                emit_iexpr(els, dst, dst_ty, env, &inner, out)?;
            }
            let _ = writeln!(out, "{}}}", ind);
            Ok(())
        }
        Bind::Match(scrut, arms) => emit_match(scrut, arms, dst, dst_ty, env, ind, out),
        Bind::MakeClosure(lam, caps) => emit_make_closure(lam, caps, dst, env, ind, out),
        Bind::ApplyClosure(callee, args, ret) => {
            emit_apply_closure(callee, args, ret.as_ref(), dst, env, ind, out)
        }
    }
}

/// Allocate a closure cell: a heap cell tagged with the lifted lambda's closure
/// tag, whose fields are the captured values (stored by their static type).
fn emit_make_closure(
    lam: &str,
    caps: &[Atom],
    dst: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    let tag = *env
        .closures
        .tags
        .get(lam)
        .ok_or_else(|| format!("c backend: unknown lambda `{}`", lam))?;
    let _ = writeln!(out, "{}{} = aria_alloc({});", ind, dst, caps.len());
    let _ = writeln!(out, "{}aria_set_tag({}, INT64_C({}));", ind, dst, tag);
    for (i, a) in caps.iter().enumerate() {
        let (t, ex) = emit_atom(a, env, out)?;
        match t {
            CType::Int | CType::Bool => {
                let _ = writeln!(out, "{}aria_field({}, {}) = (int64_t)({});", ind, dst, i, ex);
            }
            CType::Float => {
                let _ = writeln!(out, "{}aria_field({}, {}) = aria_f2i({});", ind, dst, i, ex);
            }
            CType::Ref | CType::Str | CType::Bytes | CType::Array(_) | CType::Map(..) | CType::Set(_) | CType::Vector => {
                let _ = writeln!(out, "{}aria_field({}, {}) = (int64_t)(uintptr_t)({});", ind, dst, i, ex);
            }
        }
    }
    Ok(())
}

/// Apply a closure: read its tag, index the function-pointer table, and call the
/// lifted lambda with the closure cell followed by the argument values. The cast
/// reconstructs the lambda's C signature `ret (void*, args...)` from the call
/// site's statically-known argument and result types.
fn emit_apply_closure(
    callee: &Atom,
    args: &[Atom],
    ret: Option<&Ty>,
    dst: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    let ret_ty = ret.ok_or("c backend: closure application missing its result type")?;
    let ret_ct = CType::from_ty(ret_ty)?;
    let (ct, cx) = emit_atom(callee, env, out)?;
    if ct != CType::Ref {
        return Err("c backend: applying a non-closure value".into());
    }
    // Evaluate arguments and collect their C types for the function-pointer cast.
    let mut arg_exprs = Vec::new();
    let mut arg_ctys = Vec::new();
    for a in args {
        let (at, ax) = emit_atom(a, env, out)?;
        arg_ctys.push(at.decl().to_string());
        arg_exprs.push(ax);
    }
    let mut sig_params = vec!["void*".to_string()];
    sig_params.extend(arg_ctys);
    let fnptr_ty = format!("{} (*)({})", ret_ct.decl(), sig_params.join(", "));
    let mut call_args = vec![format!("(void*){}", cx)];
    call_args.extend(arg_exprs);
    let idx = format!("(aria_tag({}) - INT64_C({}))", cx, env.closures.base);
    let _ = writeln!(
        out,
        "{}{} = (({}) __aria_lam_table[{}])({});",
        ind,
        dst,
        fnptr_ty,
        idx,
        call_args.join(", ")
    );
    // This application owns one reference to the closure (the rc pass dup'd it if
    // it is used again); release it now. The lambda body borrowed the cell's
    // captures (dup'ing each), so freeing the cell here releases only this
    // application's hold — at rc 0 the captures are released too.
    let _ = writeln!(out, "{}aria_drop((void*){});", ind, cx);
    Ok(())
}

fn emit_prim(
    op: BinOp,
    l: &Atom,
    r: &Atom,
    dst: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    let (lt, lx) = emit_atom(l, env, out)?;
    let (rt, rx) = emit_atom(r, env, out)?;
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul => {
            if lt == CType::Float && rt == CType::Float {
                let cop = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    _ => "*",
                };
                let _ = writeln!(out, "{}{} = {} {} {};", ind, dst, lx, cop, rx);
                return Ok(());
            }
            if lt != CType::Int || rt != CType::Int {
                return Err("c backend: arithmetic expects matching Int/Float operands".into());
            }
            // Checked overflow -> trap, matching the interpreter's Err.
            let bi = match op {
                BinOp::Add => "__builtin_add_overflow",
                BinOp::Sub => "__builtin_sub_overflow",
                _ => "__builtin_mul_overflow",
            };
            let _ = writeln!(out, "{}if ({}({}, {}, &{})) aria_trap_msg(\"integer overflow\");", ind, bi, lx, rx, dst);
            Ok(())
        }
        BinOp::Div | BinOp::Mod => {
            if lt == CType::Float && rt == CType::Float {
                if op == BinOp::Mod {
                    return Err("c backend: Float has no `%`".into());
                }
                // Float division: IEEE (no trap on /0), matching the interpreter.
                let _ = writeln!(out, "{}{} = {} / {};", ind, dst, lx, rx);
                return Ok(());
            }
            if lt != CType::Int || rt != CType::Int {
                return Err("c backend: arithmetic expects Int operands".into());
            }
            let cop = if op == BinOp::Div { "/" } else { "%" };
            // Trap on /0 and INT64_MIN / -1 (UB in C) to match interp's Err.
            let _ = writeln!(
                out,
                "{}if ({} == 0 || ({} == INT64_MIN && {} == -1)) aria_trap_msg(\"division by zero or overflow\");",
                ind, rx, lx, rx
            );
            let _ = writeln!(out, "{}{} = {} {} {};", ind, dst, lx, cop, rx);
            Ok(())
        }
        BinOp::Eq | BinOp::Ne => {
            // String / ADT equality is handled by emit_call's eq path? No — Eq/Ne
            // on Str/Ref appear as Prim here. Compare structurally and consume.
            if lt == CType::Str && rt == CType::Str {
                let cmp = if op == BinOp::Eq { "" } else { "!" };
                let _ = writeln!(
                    out,
                    "{}{} = {}aria_streq_consume({}, {});",
                    ind, dst, cmp, lx, rx
                );
                return Ok(());
            }
            if lt == CType::Ref && rt == CType::Ref {
                let cmp = if op == BinOp::Eq { "" } else { "!" };
                let _ = writeln!(
                    out,
                    "{}{} = {}aria_eq_consume({}, {});",
                    ind, dst, cmp, lx, rx
                );
                return Ok(());
            }
            if lt == CType::Bytes && rt == CType::Bytes {
                let cmp = if op == BinOp::Eq { "" } else { "!" };
                let _ = writeln!(
                    out,
                    "{}{} = {}aria_byteseq_consume({}, {});",
                    ind, dst, cmp, lx, rx
                );
                return Ok(());
            }
            if matches!(lt, CType::Map(..)) && matches!(rt, CType::Map(..)) {
                let cmp = if op == BinOp::Eq { "" } else { "!" };
                let _ = writeln!(out, "{}{} = {}aria_map_eq_consume({}, {});", ind, dst, cmp, lx, rx);
                return Ok(());
            }
            if matches!(lt, CType::Set(_)) && matches!(rt, CType::Set(_)) {
                let cmp = if op == BinOp::Eq { "" } else { "!" };
                let _ = writeln!(out, "{}{} = {}aria_set_eq_consume({}, {});", ind, dst, cmp, lx, rx);
                return Ok(());
            }
            if lt == CType::Vector && rt == CType::Vector {
                let cmp = if op == BinOp::Eq { "" } else { "!" };
                let _ = writeln!(out, "{}{} = {}aria_veceq_consume({}, {});", ind, dst, cmp, lx, rx);
                return Ok(());
            }
            // Scalar ==/!= (Int/Bool/Float) — direct C comparison.
            let cop = if op == BinOp::Eq { "==" } else { "!=" };
            let _ = writeln!(out, "{}{} = ({} {} {});", ind, dst, lx, cop, rx);
            Ok(())
        }
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
            let cop = match op {
                BinOp::Lt => "<",
                BinOp::Le => "<=",
                BinOp::Gt => ">",
                _ => ">=",
            };
            let _ = writeln!(out, "{}{} = ({} {} {});", ind, dst, lx, cop, rx);
            Ok(())
        }
        BinOp::And => {
            // Short-circuit `&&`/`||` are lowered to `if` by the IR, so a Prim
            // And/Or here is a plain boolean op on already-evaluated operands.
            let _ = writeln!(out, "{}{} = ({} && {});", ind, dst, lx, rx);
            Ok(())
        }
        BinOp::Or => {
            let _ = writeln!(out, "{}{} = ({} || {});", ind, dst, lx, rx);
            Ok(())
        }
    }
}

fn emit_unary(
    op: UnOp,
    a: &Atom,
    dst: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    let (t, ax) = emit_atom(a, env, out)?;
    match op {
        UnOp::Neg => {
            if t == CType::Float {
                let _ = writeln!(out, "{}{} = -{};", ind, dst, ax);
            } else if t == CType::Int {
                // Checked negation: INT64_MIN negation overflows -> trap.
                let _ = writeln!(out, "{}if (__builtin_sub_overflow((int64_t)0, {}, &{})) aria_trap_msg(\"integer overflow\");", ind, ax, dst);
            } else {
                return Err("c backend: unary `-` expects Int/Float".into());
            }
            Ok(())
        }
        UnOp::Not => {
            if t != CType::Bool {
                return Err("c backend: unary `!` expects Bool".into());
            }
            let _ = writeln!(out, "{}{} = !{};", ind, dst, ax);
            Ok(())
        }
    }
}

fn emit_call(
    name: &str,
    args: &[Atom],
    dst: &str,
    _dst_ty: CType,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    if is_builtin(name) {
        return emit_builtin(name, args, dst, env, ind, out);
    }
    let sig = env
        .sigs
        .get(name)
        .ok_or_else(|| format!("c backend: call to unknown fn `{}`", name))?;
    if sig.params.len() != args.len() {
        return Err(format!(
            "c backend: fn `{}` got {} args, expected {}",
            name,
            args.len(),
            sig.params.len()
        ));
    }
    let mut parts = Vec::new();
    for a in args {
        let (_, ex) = emit_atom(a, env, out)?;
        parts.push(ex);
    }
    let _ = writeln!(out, "{}{} = {}({});", ind, dst, cfn(name), parts.join(", "));
    Ok(())
}

/// Emit an inline builtin. Per the rc pass's "Call arguments are consumed" rule,
/// String-consuming builtins (`concat`, `print_str`, structural eq) release
/// their String/Ref operands; `int_to_str` takes an unboxed Int (nothing to
/// drop). The emitted helpers do the consuming internally.
fn emit_builtin(
    name: &str,
    args: &[Atom],
    dst: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    if let Some((base, kind)) = parse_array_builtin(name) {
        return emit_array_builtin(base, kind, args, dst, env, ind, out);
    }
    if let Some(base) = parse_map_set_builtin(name) {
        return emit_map_set_builtin(base, name, args, dst, env, ind, out);
    }
    match name {
        "concat" => {
            if args.len() != 2
                || atom_type(&args[0], env)? != CType::Str
                || atom_type(&args[1], env)? != CType::Str
            {
                return Err("c backend: concat expects two Strings".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            let (_, b) = emit_atom(&args[1], env, out)?;
            let _ = writeln!(out, "{}{} = aria_concat({}, {});", ind, dst, a, b);
            Ok(())
        }
        "int_to_str" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Int {
                return Err("c backend: int_to_str expects one Int".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_int_to_str({});", ind, dst, a);
            Ok(())
        }
        "print_int" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Int {
                return Err("c backend: print_int expects one Int".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}aria_print_int({});", ind, a);
            let _ = writeln!(out, "{}{} = 0;", ind, dst);
            Ok(())
        }
        "print_bool" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Bool {
                return Err("c backend: print_bool expects one Bool".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}aria_print_bool({});", ind, a);
            let _ = writeln!(out, "{}{} = 0;", ind, dst);
            Ok(())
        }
        "print_float" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Float {
                return Err("c backend: print_float expects one Float".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}aria_print_float({});", ind, a);
            let _ = writeln!(out, "{}{} = 0;", ind, dst);
            Ok(())
        }
        "print_str" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Str {
                return Err("c backend: print_str expects one String".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            // print then consume the String (it is an argument -> consumed).
            let _ = writeln!(out, "{}aria_print_str({});", ind, a);
            let _ = writeln!(out, "{}{} = 0;", ind, dst);
            Ok(())
        }
        // ---- Bytes builtins (each helper consumes its Bytes argument(s)) ----
        "bytes_new" => {
            if !args.is_empty() {
                return Err("c backend: bytes_new expects no arguments".into());
            }
            let _ = writeln!(out, "{}{} = aria_bytes_new();", ind, dst);
            Ok(())
        }
        "bytes_len" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Bytes {
                return Err("c backend: bytes_len expects one Bytes".into());
            }
            let (_, b) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_bytes_len({});", ind, dst, b);
            Ok(())
        }
        "bytes_get" => {
            if args.len() != 2 || atom_type(&args[0], env)? != CType::Bytes {
                return Err("c backend: bytes_get expects (Bytes, Int)".into());
            }
            let (_, b) = emit_atom(&args[0], env, out)?;
            let (it, idx) = emit_atom(&args[1], env, out)?;
            if it != CType::Int {
                return Err("c backend: bytes_get index must be Int".into());
            }
            let _ = writeln!(out, "{}{} = aria_bytes_get({}, {});", ind, dst, b, idx);
            Ok(())
        }
        "bytes_set" => {
            if args.len() != 3 || atom_type(&args[0], env)? != CType::Bytes {
                return Err("c backend: bytes_set expects (Bytes, Int, Int)".into());
            }
            let (_, b) = emit_atom(&args[0], env, out)?;
            let (it, idx) = emit_atom(&args[1], env, out)?;
            let (vt, v) = emit_atom(&args[2], env, out)?;
            if it != CType::Int || vt != CType::Int {
                return Err("c backend: bytes_set index/value must be Int".into());
            }
            let _ = writeln!(out, "{}{} = aria_bytes_set({}, {}, {});", ind, dst, b, idx, v);
            Ok(())
        }
        "bytes_push" => {
            if args.len() != 2 || atom_type(&args[0], env)? != CType::Bytes {
                return Err("c backend: bytes_push expects (Bytes, Int)".into());
            }
            let (_, b) = emit_atom(&args[0], env, out)?;
            let (vt, v) = emit_atom(&args[1], env, out)?;
            if vt != CType::Int {
                return Err("c backend: bytes_push value must be Int".into());
            }
            let _ = writeln!(out, "{}{} = aria_bytes_push({}, {});", ind, dst, b, v);
            Ok(())
        }
        "bytes_from_str" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Str {
                return Err("c backend: bytes_from_str expects one String".into());
            }
            let (_, s) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_bytes_from_str({});", ind, dst, s);
            Ok(())
        }
        "bytes_to_str" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Bytes {
                return Err("c backend: bytes_to_str expects one Bytes".into());
            }
            let (_, b) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_bytes_to_str({});", ind, dst, b);
            Ok(())
        }
        // ---- Vector builtins (each helper consumes its Vector argument(s)) ----
        "vec_new" => {
            if !args.is_empty() {
                return Err("c backend: vec_new expects no arguments".into());
            }
            let _ = writeln!(out, "{}{} = aria_vec_new();", ind, dst);
            Ok(())
        }
        "vec_from_array" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Array(ElemKind::Float) {
                return Err("c backend: vec_from_array expects one Array[Float]".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_vec_from_array({});", ind, dst, a);
            Ok(())
        }
        "vec_to_array" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Vector {
                return Err("c backend: vec_to_array expects one Vector".into());
            }
            let (_, v) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_vec_to_array({});", ind, dst, v);
            Ok(())
        }
        "vec_len" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Vector {
                return Err("c backend: vec_len expects one Vector".into());
            }
            let (_, v) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_vec_len({});", ind, dst, v);
            Ok(())
        }
        "vec_get" => {
            if args.len() != 2 || atom_type(&args[0], env)? != CType::Vector {
                return Err("c backend: vec_get expects (Vector, Int)".into());
            }
            let (_, v) = emit_atom(&args[0], env, out)?;
            let (it, idx) = emit_atom(&args[1], env, out)?;
            if it != CType::Int {
                return Err("c backend: vec_get index must be Int".into());
            }
            let _ = writeln!(out, "{}{} = aria_vec_get({}, {});", ind, dst, v, idx);
            Ok(())
        }
        "vec_push" => {
            if args.len() != 2 || atom_type(&args[0], env)? != CType::Vector {
                return Err("c backend: vec_push expects (Vector, Float)".into());
            }
            let (_, v) = emit_atom(&args[0], env, out)?;
            let (ft, f) = emit_atom(&args[1], env, out)?;
            if ft != CType::Float {
                return Err("c backend: vec_push value must be Float".into());
            }
            let _ = writeln!(out, "{}{} = aria_vec_push({}, {});", ind, dst, v, f);
            Ok(())
        }
        "vec_dot" => {
            if args.len() != 2
                || atom_type(&args[0], env)? != CType::Vector
                || atom_type(&args[1], env)? != CType::Vector
            {
                return Err("c backend: vec_dot expects (Vector, Vector)".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            let (_, b) = emit_atom(&args[1], env, out)?;
            let _ = writeln!(out, "{}{} = aria_vec_dot({}, {});", ind, dst, a, b);
            Ok(())
        }
        "vec_norm" => {
            if args.len() != 1 || atom_type(&args[0], env)? != CType::Vector {
                return Err("c backend: vec_norm expects one Vector".into());
            }
            let (_, v) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_vec_norm({});", ind, dst, v);
            Ok(())
        }
        "vec_cosine" => {
            if args.len() != 2
                || atom_type(&args[0], env)? != CType::Vector
                || atom_type(&args[1], env)? != CType::Vector
            {
                return Err("c backend: vec_cosine expects (Vector, Vector)".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            let (_, b) = emit_atom(&args[1], env, out)?;
            let _ = writeln!(out, "{}{} = aria_vec_cosine({}, {});", ind, dst, a, b);
            Ok(())
        }
        "vec_add" => {
            if args.len() != 2
                || atom_type(&args[0], env)? != CType::Vector
                || atom_type(&args[1], env)? != CType::Vector
            {
                return Err("c backend: vec_add expects (Vector, Vector)".into());
            }
            let (_, a) = emit_atom(&args[0], env, out)?;
            let (_, b) = emit_atom(&args[1], env, out)?;
            let _ = writeln!(out, "{}{} = aria_vec_add({}, {});", ind, dst, a, b);
            Ok(())
        }
        "vec_scale" => {
            if args.len() != 2 || atom_type(&args[0], env)? != CType::Vector {
                return Err("c backend: vec_scale expects (Vector, Float)".into());
            }
            let (_, v) = emit_atom(&args[0], env, out)?;
            let (ft, s) = emit_atom(&args[1], env, out)?;
            if ft != CType::Float {
                return Err("c backend: vec_scale factor must be Float".into());
            }
            let _ = writeln!(out, "{}{} = aria_vec_scale({}, {});", ind, dst, v, s);
            Ok(())
        }
        _ => Err(format!("c backend: unsupported builtin `{}`", name)),
    }
}

/// Encode an evaluated element value (C expression `ex` of type `t`) into the
/// int64 slot representation used by `AriaArray.elems[]`, per the array's kind.
fn encode_elem_slot(kind: ElemKind, t: CType, ex: &str) -> Result<String, String> {
    // A `Ref`-kind element is any boxed heap value stored as a pointer — an ADT
    // cell, a string, or a NESTED array — so accept all pointer-typed values
    // there (this is what makes `Array[Array[..]]` / `Array[String]` work). The
    // scalar kinds must match exactly so the encoding is correct.
    let ok = match kind {
        ElemKind::Int => matches!(t, CType::Int),
        ElemKind::Bool => matches!(t, CType::Bool),
        ElemKind::Float => t == CType::Float,
        ElemKind::Str => t == CType::Str,
        ElemKind::Bytes => t == CType::Bytes,
        ElemKind::Vector => t == CType::Vector,
        ElemKind::Array => matches!(t, CType::Array(_)),
        ElemKind::Map => matches!(t, CType::Map(..)),
        ElemKind::Set => matches!(t, CType::Set(_)),
        // A `Ref`-kind element is any boxed ADT cell / closure stored as a pointer.
        ElemKind::Ref => matches!(t, CType::Ref),
    };
    if !ok {
        return Err(format!(
            "c backend: array element type mismatch (got {:?}, expected {:?})",
            t,
            kind.elem_ctype()
        ));
    }
    Ok(match kind {
        // Bool stores inline in the int64 slot like Int (0/1), no dup/drop.
        ElemKind::Int | ElemKind::Bool => format!("(int64_t)({})", ex),
        ElemKind::Float => format!("aria_f2i({})", ex),
        // every heap kind is stored as a pointer cast through uintptr_t.
        ElemKind::Str
        | ElemKind::Ref
        | ElemKind::Bytes
        | ElemKind::Array
        | ElemKind::Map
        | ElemKind::Set
        | ElemKind::Vector => format!("(int64_t)(uintptr_t)({})", ex),
    })
}

/// Decode an int64 slot C expression `slot` back to the element's C value type.
fn decode_elem_slot(kind: ElemKind, slot: &str) -> String {
    match kind {
        // Bool decodes like Int: the slot already holds 0/1 as an int64, usable
        // directly as a C bool (`int`).
        ElemKind::Int | ElemKind::Bool => slot.to_string(),
        ElemKind::Float => format!("aria_i2f({})", slot),
        ElemKind::Str
        | ElemKind::Ref
        | ElemKind::Bytes
        | ElemKind::Array
        | ElemKind::Map
        | ElemKind::Set
        | ElemKind::Vector => format!("(void*)(uintptr_t)({})", slot),
    }
}

/// Emit a native array builtin. The monomorphizer suffixes each name with the
/// concrete element kind; we dispatch to the `aria_array_*` runtime, encoding
/// Float via f2i and Str/Ref via pointer casts.
fn emit_array_builtin(
    base: &str,
    kind: ElemKind,
    args: &[Atom],
    dst: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    match base {
        "array_new" => {
            if !args.is_empty() {
                return Err("c backend: array_new expects no arguments".into());
            }
            let _ = writeln!(out, "{}{} = aria_array_new(INT64_C({}));", ind, dst, kind.code());
            Ok(())
        }
        "array_lit" => {
            // Build an array, then push each element. Each element's ownership
            // moves into the array (the IR does not dup them), so storing the
            // raw slot is correct.
            let _ = writeln!(out, "{}{} = aria_array_new(INT64_C({}));", ind, dst, kind.code());
            for a in args {
                let (t, ex) = emit_atom(a, env, out)?;
                let slot = encode_elem_slot(kind, t, &ex)?;
                let _ = writeln!(
                    out,
                    "{}{} = aria_array_push({}, {}, INT64_C({}));",
                    ind, dst, dst, slot, kind.code()
                );
            }
            Ok(())
        }
        "array_len" => {
            if args.len() != 1 {
                return Err("c backend: array_len expects one array".into());
            }
            let (_, arr) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_array_len({});", ind, dst, arr);
            Ok(())
        }
        "array_get" => {
            if args.len() != 2 {
                return Err("c backend: array_get expects (array, index)".into());
            }
            let (_, arr) = emit_atom(&args[0], env, out)?;
            let (it, idx) = emit_atom(&args[1], env, out)?;
            if it != CType::Int {
                return Err("c backend: array_get index must be Int".into());
            }
            let slot = format!("aria_array_get({}, {})", arr, idx);
            let _ = writeln!(out, "{}{} = {};", ind, dst, decode_elem_slot(kind, &slot));
            Ok(())
        }
        "array_set" => {
            if args.len() != 3 {
                return Err("c backend: array_set expects (array, index, value)".into());
            }
            let (_, arr) = emit_atom(&args[0], env, out)?;
            let (it, idx) = emit_atom(&args[1], env, out)?;
            if it != CType::Int {
                return Err("c backend: array_set index must be Int".into());
            }
            let (t, ex) = emit_atom(&args[2], env, out)?;
            let slot = encode_elem_slot(kind, t, &ex)?;
            let _ = writeln!(out, "{}{} = aria_array_set({}, {}, {});", ind, dst, arr, idx, slot);
            Ok(())
        }
        "array_push" => {
            if args.len() != 2 {
                return Err("c backend: array_push expects (array, value)".into());
            }
            let (_, arr) = emit_atom(&args[0], env, out)?;
            let (t, ex) = emit_atom(&args[1], env, out)?;
            let slot = encode_elem_slot(kind, t, &ex)?;
            let _ = writeln!(
                out,
                "{}{} = aria_array_push({}, {}, INT64_C({}));",
                ind, dst, arr, slot, kind.code()
            );
            Ok(())
        }
        _ => Err(format!("c backend: unsupported array builtin `{}`", base)),
    }
}

/// Emit a native map/set builtin. The precise key/value/element kinds are read
/// from the operands' static `CType`s and passed to the `aria_map_*`/`aria_set_*`
/// runtime, which keeps the container sorted by key/element and does the
/// kind-aware dup/drop. Each helper CONSUMES its container argument (rc-balanced:
/// the rc pass dup'd it if reused), mirroring the array runtime.
fn emit_map_set_builtin(
    base: &str,
    name: &str,
    args: &[Atom],
    dst: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    // The coarse header kinds for the empty constructors; precise kinds come
    // from operands for every op that has one.
    let (sk, sv) = map_set_suffix_kinds(name);
    match base {
        "map_new" => {
            if !args.is_empty() {
                return Err("c backend: map_new expects no arguments".into());
            }
            let _ = writeln!(
                out,
                "{}{} = aria_map_new(INT64_C({}), INT64_C({}));",
                ind, dst, sk.code(), sv.code()
            );
            Ok(())
        }
        "map_insert" => {
            if args.len() != 3 {
                return Err("c backend: map_insert expects (map, key, value)".into());
            }
            let (mt, m) = emit_atom(&args[0], env, out)?;
            let (kk, vk) = map_kinds_of(mt)?;
            let (kt, kex) = emit_atom(&args[1], env, out)?;
            let (vt, vex) = emit_atom(&args[2], env, out)?;
            // Use the precise kinds of the actual key/value operands.
            let kk = unify_slot(kk, SlotKind::from_ctype(kt));
            let vk = unify_slot(vk, SlotKind::from_ctype(vt));
            let kslot = kk.encode(kt, &kex);
            let vslot = vk.encode(vt, &vex);
            let _ = writeln!(
                out,
                "{}{} = aria_map_insert({}, {}, {}, INT64_C({}), INT64_C({}));",
                ind, dst, m, kslot, vslot, kk.code(), vk.code()
            );
            Ok(())
        }
        "map_get_or" => {
            if args.len() != 3 {
                return Err("c backend: map_get_or expects (map, key, default)".into());
            }
            let (mt, m) = emit_atom(&args[0], env, out)?;
            let (kk, vk) = map_kinds_of(mt)?;
            let (kt, kex) = emit_atom(&args[1], env, out)?;
            let (vt, vex) = emit_atom(&args[2], env, out)?;
            let kk = unify_slot(kk, SlotKind::from_ctype(kt));
            let vk = unify_slot(vk, SlotKind::from_ctype(vt));
            let kslot = kk.encode(kt, &kex);
            let vslot = vk.encode(vt, &vex);
            // Returns the value slot (the stored value dup'd, or the default).
            let call = format!(
                "aria_map_get_or({}, {}, {}, INT64_C({}), INT64_C({}))",
                m, kslot, vslot, kk.code(), vk.code()
            );
            let _ = writeln!(out, "{}{} = {};", ind, dst, vk.decode(&call));
            Ok(())
        }
        "map_has" => {
            if args.len() != 2 {
                return Err("c backend: map_has expects (map, key)".into());
            }
            let (mt, m) = emit_atom(&args[0], env, out)?;
            let (kk, _) = map_kinds_of(mt)?;
            let (kt, kex) = emit_atom(&args[1], env, out)?;
            let kk = unify_slot(kk, SlotKind::from_ctype(kt));
            let kslot = kk.encode(kt, &kex);
            let _ = writeln!(
                out,
                "{}{} = aria_map_has({}, {}, INT64_C({}));",
                ind, dst, m, kslot, kk.code()
            );
            Ok(())
        }
        "map_len" => {
            if args.len() != 1 {
                return Err("c backend: map_len expects (map)".into());
            }
            let (_, m) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_map_len({});", ind, dst, m);
            Ok(())
        }
        "map_show" => {
            if args.len() != 1 {
                return Err("c backend: map_show expects (map)".into());
            }
            let (_, m) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_map_show({});", ind, dst, m);
            Ok(())
        }
        "map_keys" | "map_values" => {
            if args.len() != 1 {
                return Err(format!("c backend: {} expects (map)", base));
            }
            let (mt, m) = emit_atom(&args[0], env, out)?;
            let (kk, vk) = map_kinds_of(mt)?;
            // The array's element kind: Bool collapses to the Int slot (no-op
            // dup/drop), matching the result CType (ElemKind has no Bool).
            let (src_kind, out_kind) = if base == "map_keys" {
                (kk, ElemKind::from_ctype(&slot_kind_ctype(kk)).code())
            } else {
                (vk, ElemKind::from_ctype(&slot_kind_ctype(vk)).code())
            };
            let helper = if base == "map_keys" { "aria_map_keys" } else { "aria_map_values" };
            let _ = writeln!(
                out,
                "{}{} = {}({}, INT64_C({}), INT64_C({}));",
                ind, dst, helper, m, src_kind.code(), out_kind
            );
            Ok(())
        }
        "set_to_array" => {
            if args.len() != 1 {
                return Err("c backend: set_to_array expects (set)".into());
            }
            let (st, s) = emit_atom(&args[0], env, out)?;
            let ek = set_kind_of(st)?;
            let out_kind = ElemKind::from_ctype(&slot_kind_ctype(ek)).code();
            let _ = writeln!(
                out,
                "{}{} = aria_set_to_array({}, INT64_C({}), INT64_C({}));",
                ind, dst, s, ek.code(), out_kind
            );
            Ok(())
        }
        "map_remove" => {
            if args.len() != 2 {
                return Err("c backend: map_remove expects (map, key)".into());
            }
            let (mt, m) = emit_atom(&args[0], env, out)?;
            let (kk, _) = map_kinds_of(mt)?;
            let (kt, kex) = emit_atom(&args[1], env, out)?;
            let kk = unify_slot(kk, SlotKind::from_ctype(kt));
            let kslot = kk.encode(kt, &kex);
            let _ = writeln!(
                out,
                "{}{} = aria_map_remove({}, {}, INT64_C({}));",
                ind, dst, m, kslot, kk.code()
            );
            Ok(())
        }
        "set_new" => {
            if !args.is_empty() {
                return Err("c backend: set_new expects no arguments".into());
            }
            let _ = writeln!(out, "{}{} = aria_set_new(INT64_C({}));", ind, dst, sk.code());
            Ok(())
        }
        "set_add" => {
            if args.len() != 2 {
                return Err("c backend: set_add expects (set, elem)".into());
            }
            let (st, s) = emit_atom(&args[0], env, out)?;
            let ek = set_kind_of(st)?;
            let (et, eex) = emit_atom(&args[1], env, out)?;
            let ek = unify_slot(ek, SlotKind::from_ctype(et));
            let eslot = ek.encode(et, &eex);
            let _ = writeln!(
                out,
                "{}{} = aria_set_add({}, {}, INT64_C({}));",
                ind, dst, s, eslot, ek.code()
            );
            Ok(())
        }
        "set_has" => {
            if args.len() != 2 {
                return Err("c backend: set_has expects (set, elem)".into());
            }
            let (st, s) = emit_atom(&args[0], env, out)?;
            let ek = set_kind_of(st)?;
            let (et, eex) = emit_atom(&args[1], env, out)?;
            let ek = unify_slot(ek, SlotKind::from_ctype(et));
            let eslot = ek.encode(et, &eex);
            let _ = writeln!(
                out,
                "{}{} = aria_set_has({}, {}, INT64_C({}));",
                ind, dst, s, eslot, ek.code()
            );
            Ok(())
        }
        "set_len" => {
            if args.len() != 1 {
                return Err("c backend: set_len expects (set)".into());
            }
            let (_, s) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_set_len({});", ind, dst, s);
            Ok(())
        }
        "set_show" => {
            if args.len() != 1 {
                return Err("c backend: set_show expects (set)".into());
            }
            let (_, s) = emit_atom(&args[0], env, out)?;
            let _ = writeln!(out, "{}{} = aria_set_show({});", ind, dst, s);
            Ok(())
        }
        "set_remove" => {
            if args.len() != 2 {
                return Err("c backend: set_remove expects (set, elem)".into());
            }
            let (st, s) = emit_atom(&args[0], env, out)?;
            let ek = set_kind_of(st)?;
            let (et, eex) = emit_atom(&args[1], env, out)?;
            let ek = unify_slot(ek, SlotKind::from_ctype(et));
            let eslot = ek.encode(et, &eex);
            let _ = writeln!(
                out,
                "{}{} = aria_set_remove({}, {}, INT64_C({}));",
                ind, dst, s, eslot, ek.code()
            );
            Ok(())
        }
        _ => Err(format!("c backend: unsupported map/set builtin `{}`", base)),
    }
}

/// The (key, value) slot kinds of a `CType::Map`.
fn map_kinds_of(t: CType) -> Result<(SlotKind, SlotKind), String> {
    match t {
        CType::Map(k, v) => Ok((k, v)),
        other => Err(format!("c backend: expected a Map, got {:?}", other)),
    }
}

/// The element slot kind of a `CType::Set`.
fn set_kind_of(t: CType) -> Result<SlotKind, String> {
    match t {
        CType::Set(e) => Ok(e),
        other => Err(format!("c backend: expected a Set, got {:?}", other)),
    }
}

/// Reconcile a container's recorded slot kind with the actual key/value OPERAND
/// kind for an op that has one (`map_insert`/`map_get_or`/...). The operand is
/// post-typeck concrete and is the AUTHORITATIVE kind for the slot being
/// stored/retrieved, so it wins whenever it is known. The container's recorded
/// kind is only a fallback: an unannotated `map_new()` built across separate
/// `let` bindings can carry a stale default kind (e.g. `Int`/`Ref` from
/// `$i_i`/`$r_r`) that does NOT match the value actually inserted — trusting the
/// container there mis-encodes the slot (the BUG-2b / empty-container-default
/// interaction). For a well-typed program the operand and a precise container
/// kind always agree, so preferring the operand is at least as correct.
fn unify_slot(container: SlotKind, operand: SlotKind) -> SlotKind {
    let _ = container;
    operand
}

fn check_ctor<'a>(
    name: &str,
    fields: &[Atom],
    env: &'a Env,
) -> Result<&'a CtorInfo, String> {
    let info = env.ctors.get(name)?;
    if fields.len() != info.arity() {
        return Err(format!(
            "c backend: ctor `{}` got {} fields, expected {}",
            name,
            fields.len(),
            info.arity()
        ));
    }
    Ok(info)
}

/// Store the tag + each field into the cell whose pointer is the C expression
/// `cellptr` (a `void*`). Field stores cast per static type into the 64-bit slot.
fn emit_store_fields(
    name: &str,
    fields: &[Atom],
    cellptr: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    let info = env.ctors.get(name)?.clone();
    let _ = writeln!(out, "{}aria_set_tag({}, INT64_C({}));", ind, cellptr, info.tag);
    for (i, (a, fty)) in fields.iter().zip(info.field_types.iter()).enumerate() {
        let (t, ex) = emit_atom(a, env, out)?;
        if t != *fty {
            return Err(format!(
                "c backend: ctor `{}` field {} type mismatch (got {:?}, expected {:?})",
                name, i, t, fty
            ));
        }
        match fty {
            CType::Int | CType::Bool => {
                let _ = writeln!(out, "{}aria_field({}, {}) = (int64_t)({});", ind, cellptr, i, ex);
            }
            CType::Float => {
                let _ = writeln!(out, "{}aria_field({}, {}) = aria_f2i({});", ind, cellptr, i, ex);
            }
            CType::Ref | CType::Str | CType::Bytes | CType::Array(_) | CType::Map(..) | CType::Set(_) | CType::Vector => {
                let _ = writeln!(out, "{}aria_field({}, {}) = (int64_t)(uintptr_t)({});", ind, cellptr, i, ex);
            }
        }
    }
    Ok(())
}

fn emit_ctor(
    name: &str,
    fields: &[Atom],
    dst: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    let info = check_ctor(name, fields, env)?.clone();
    let _ = writeln!(out, "{}{} = aria_alloc({});", ind, dst, info.arity());
    emit_store_fields(name, fields, dst, env, ind, out)?;
    Ok(())
}

fn emit_ctor_reuse(
    tok: &str,
    name: &str,
    fields: &[Atom],
    dst: &str,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    let info = check_ctor(name, fields, env)?.clone();
    let tv = cvar(tok);
    // If the token is a retained slot, reuse it (rc=1, bump reuse counter); else
    // alloc fresh. Mirrors the IR interpreter's CtorReuse handler.
    let _ = writeln!(out, "{}if ({} != NULL) {{ {} = aria_reuse({}); }} else {{ {} = aria_alloc({}); }}", ind, tv, dst, tv, dst, info.arity());
    emit_store_fields(name, fields, dst, env, ind, out)?;
    Ok(())
}

fn emit_match(
    scrut: &Atom,
    arms: &[ir::IArm],
    dst: &str,
    dst_ty: CType,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    if atom_type(scrut, env)? != CType::Ref {
        return Err("c backend: `match` scrutinee must be an ADT (Ref)".into());
    }
    let (_, sx) = emit_atom(scrut, env, out)?;
    // Hold the scrutinee in a stable temp (it may be a complex expr / var).
    let sv = env.fresh();
    let _ = writeln!(out, "{}void* {} = {};", ind, sv, sx);
    emit_match_chain(&sv, arms, 0, dst, dst_ty, env, ind, out)
}

fn emit_match_chain(
    sv: &str,
    arms: &[ir::IArm],
    i: usize,
    dst: &str,
    dst_ty: CType,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    if i >= arms.len() {
        let _ = writeln!(out, "{}aria_trap_msg(\"no matching pattern (non-exhaustive match)\");", ind);
        return Ok(());
    }
    let arm = &arms[i];
    match &arm.ctor {
        None => emit_arm_body(sv, arm, None, dst, dst_ty, env, ind, out),
        Some(cname) => {
            let info = env.ctors.get(cname)?.clone();
            let _ = writeln!(out, "{}if (aria_tag({}) == INT64_C({})) {{", ind, sv, info.tag);
            let inner = format!("{}    ", ind);
            emit_arm_body(sv, arm, Some(&info), dst, dst_ty, env, &inner, out)?;
            let _ = writeln!(out, "{}}} else {{", ind);
            emit_match_chain(sv, arms, i + 1, dst, dst_ty, env, &inner, out)?;
            let _ = writeln!(out, "{}}}", ind);
            Ok(())
        }
    }
}

fn emit_arm_body(
    sv: &str,
    arm: &ir::IArm,
    info: Option<&CtorInfo>,
    dst: &str,
    dst_ty: CType,
    env: &mut Env,
    ind: &str,
    out: &mut String,
) -> Result<(), String> {
    match info {
        Some(info) => {
            for (idx, b) in arm.binders.iter().enumerate() {
                let fty = info.field_types[idx];
                let cv = cvar(b);
                let load = match fty {
                    CType::Int | CType::Bool => {
                        format!("aria_field({}, {})", sv, idx)
                    }
                    CType::Float => format!("aria_i2f(aria_field({}, {}))", sv, idx),
                    CType::Ref | CType::Str | CType::Bytes | CType::Array(_) | CType::Map(..) | CType::Set(_) | CType::Vector => {
                        format!("(void*)(uintptr_t)aria_field({}, {})", sv, idx)
                    }
                };
                let _ = writeln!(out, "{}{} {} = {};", ind, fty.decl(), cv, load);
                env.types.insert(b.clone(), fty);
            }
        }
        None => {
            if let Some(b) = arm.binders.first() {
                let cv = cvar(b);
                let _ = writeln!(out, "{}void* {} = {};", ind, cv, sv);
                env.types.insert(b.clone(), CType::Ref);
            }
        }
    }
    emit_iexpr(&arm.body, dst, dst_ty, env, ind, out)
}

// ---- string-literal collection ------------------------------------------

fn collect_lits_iexpr(e: &IExpr, out: &mut Vec<Vec<u8>>) {
    match e {
        IExpr::Ret(a) => collect_lit_atom(a, out),
        IExpr::Let(_, b, body) => {
            collect_lits_bind(b, out);
            collect_lits_iexpr(body, out);
        }
        IExpr::Dup(_, b) | IExpr::Drop(_, b) | IExpr::DropReuse(_, _, b) => collect_lits_iexpr(b, out),
        IExpr::TailCall(args) => {
            for a in args {
                collect_lit_atom(a, out);
            }
        }
    }
}

fn collect_lits_bind(b: &Bind, out: &mut Vec<Vec<u8>>) {
    match b {
        Bind::Atom(a) | Bind::Unary(_, a) => collect_lit_atom(a, out),
        Bind::Prim(_, l, r) => {
            collect_lit_atom(l, out);
            collect_lit_atom(r, out);
        }
        Bind::Ctor(_, fs) | Bind::Call(_, fs) | Bind::CtorReuse(_, _, fs) | Bind::MakeClosure(_, fs) => {
            for a in fs {
                collect_lit_atom(a, out);
            }
        }
        Bind::ApplyClosure(callee, fs, _) => {
            collect_lit_atom(callee, out);
            for a in fs {
                collect_lit_atom(a, out);
            }
        }
        Bind::If(c, t, e) => {
            collect_lit_atom(c, out);
            collect_lits_iexpr(t, out);
            collect_lits_iexpr(e, out);
        }
        Bind::Match(s, arms) => {
            collect_lit_atom(s, out);
            for a in arms {
                collect_lits_iexpr(&a.body, out);
            }
        }
    }
}

fn collect_lit_atom(a: &Atom, out: &mut Vec<Vec<u8>>) {
    if let Atom::Str(s) = a {
        let b = s.as_bytes().to_vec();
        if !out.contains(&b) {
            out.push(b);
        }
    }
}

// ---- the C runtime prelude ----------------------------------------------

/// The fixed C runtime: cell/string layout, allocator + live counter,
/// dup/drop/reuse, structural equality, concat/int_to_str, and the print
/// helpers. The per-tag recursive field release (`aria_drop`) is emitted
/// program-specifically below; this prelude declares it.
const RUNTIME: &str = r#"#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- live-cell accounting (the native analogue of wasm __live) ---- */
static int64_t aria_live = 0;
static int64_t aria_reuses = 0;

/* ---- ADT cell: { int64_t rc; int64_t tag; int64_t fields[]; } ---- */
typedef struct { int64_t rc; int64_t tag; int64_t fields[]; } AriaCell;

static void aria_trap_msg(const char* msg) {
    /* A defined Aria runtime error. Print a descriptive message to stderr in the
       same `runtime error: ...` form the interpreter uses, then abort with a
       non-zero status (which runners detect as a trap, independent of output). */
    fflush(stdout);
    fprintf(stderr, "runtime error: %s\n", msg);
    fflush(stderr);
    exit(70);
}
/* Generic trap with no specific context (allocation failure, internal invariant). */
static void aria_trap(void) { aria_trap_msg("aborted"); }

static void* aria_alloc(int64_t nfields) {
    AriaCell* c = (AriaCell*)malloc(sizeof(AriaCell) + (size_t)nfields * sizeof(int64_t));
    if (!c) aria_trap();
    c->rc = 1;
    c->tag = 0;
    aria_live++;
    return (void*)c;
}

/* Reuse a retained slot (from aria_drop_reuse): rc=1, count the reuse. The slot
   was kept (not freed), so aria_live is unchanged. */
static void* aria_reuse(void* p) {
    AriaCell* c = (AriaCell*)p;
    c->rc = 1;
    aria_reuses++;
    return p;
}

#define aria_field(p, i)  (((AriaCell*)(p))->fields[(i)])
static inline void aria_set_tag(void* p, int64_t t) { ((AriaCell*)(p))->tag = t; }
static inline int64_t aria_tag(void* p) { return ((AriaCell*)(p))->tag; }

/* Float <-> int64 bit reinterpret for storing Float fields in a 64-bit slot. */
static inline int64_t aria_f2i(double d) { int64_t i; memcpy(&i, &d, 8); return i; }
static inline double aria_i2f(int64_t i) { double d; memcpy(&d, &i, 8); return d; }

/* ---- String object: { int64_t rc; int64_t len; char bytes[]; } ---- */
typedef struct { int64_t rc; int64_t len; char bytes[]; } AriaStr;

static void* aria_str_alloc(int64_t len) {
    AriaStr* s = (AriaStr*)malloc(sizeof(AriaStr) + (size_t)len);
    if (!s) aria_trap();
    s->rc = 1;
    s->len = len;
    aria_live++;
    return (void*)s;
}
static inline void aria_str_dup(void* p) { ((AriaStr*)p)->rc++; }
static void aria_str_drop(void* p) {
    AriaStr* s = (AriaStr*)p;
    if (--s->rc == 0) { aria_live--; free(s); }
}
static void* aria_str_lit(const char* bytes, int64_t len) {
    AriaStr* s = (AriaStr*)aria_str_alloc(len);
    memcpy(s->bytes, bytes, (size_t)len);
    return (void*)s;
}

/* ---- dup/drop/reuse for ADT cells (per-tag field release below) ---- */
static inline void aria_dup(void* p) { ((AriaCell*)p)->rc++; }
static void aria_drop_children(void* p);  /* program-specific, emitted below */

static void aria_drop(void* p) {
    AriaCell* c = (AriaCell*)p;
    if (--c->rc == 0) {
        aria_drop_children(p);
        aria_live--;
        free(c);
    }
}

/* Drop for reuse: if the cell becomes unique-and-dead, release its children but
   RETAIN the slot, returning the pointer; otherwise return NULL. */
static void* aria_drop_reuse(void* p) {
    AriaCell* c = (AriaCell*)p;
    if (--c->rc == 0) {
        aria_drop_children(p);
        return p;  /* slot retained for a same-arity CtorReuse */
    }
    return NULL;
}

/* ---- native array: { rc; kind; len; cap; int64_t elems[] } ----
   kind codes match SlotKind: 0=Int/Bool, 1=Float, 2=Str, 3=Ref (ADT/closure),
   4=Bytes, 5=Array (nested), 6=Map, 7=Set, 9=Vector. Elements are stored one per
   int64 slot: Int directly, Float via aria_f2i, every heap kind as the pointer
   cast through uintptr_t. Per-element dup/drop dispatch through the shared
   aria_slot_dup/aria_slot_drop (forward-declared) so each heap kind uses its
   CORRECT recursive drop. FBIP: set/push mutate in place when rc==1, else COW. */
static void aria_slot_dup(int64_t kind, int64_t slot);
static void aria_slot_drop(int64_t kind, int64_t slot);
typedef struct { int64_t rc; int64_t kind; int64_t len; int64_t cap; int64_t elems[]; } AriaArray;

static void* aria_array_alloc(int64_t kind, int64_t cap) {
    AriaArray* a = (AriaArray*)malloc(sizeof(AriaArray) + (size_t)cap * sizeof(int64_t));
    if (!a) aria_trap();
    a->rc = 1; a->kind = kind; a->len = 0; a->cap = cap;
    aria_live++;
    return (void*)a;
}
static void* aria_array_new(int64_t kind) { return aria_array_alloc(kind, 0); }
static inline void aria_array_dup(void* p) { ((AriaArray*)p)->rc++; }
static void aria_array_drop(void* p);  /* defined below */
static int64_t aria_array_len(void* p) {
    int64_t n = ((AriaArray*)p)->len;
    aria_array_drop(p);  /* array_len consumes its array argument */
    return n;
}

/* Dup a single element slot (used when copy-on-write clones a buffer). The
   kind-aware dispatch lives in aria_slot_dup (Str/Ref/Bytes/Array/Map/Set/Vector;
   Int/Float are no-ops). */
static void aria_array_dup_elem(int64_t kind, int64_t slot) {
    aria_slot_dup(kind, slot);
}
/* Drop a single element slot. */
static void aria_array_drop_elem(int64_t kind, int64_t slot) {
    aria_slot_drop(kind, slot);
}

static void aria_array_drop(void* p) {
    AriaArray* a = (AriaArray*)p;
    if (--a->rc == 0) {
        /* Int(0)/Float(1)/Bool(8) elements store inline — nothing to drop. */
        if (a->kind != 0 && a->kind != 1 && a->kind != 8) {
            for (int64_t i = 0; i < a->len; i++) aria_slot_drop(a->kind, a->elems[i]);
        }
        aria_live--;
        free(a);
    }
}

static int64_t aria_array_get(void* p, int64_t i) {
    AriaArray* a = (AriaArray*)p;
    if (i < 0 || i >= a->len) aria_trap_msg("array index out of range");
    int64_t slot = a->elems[i];
    /* The element is still owned by the array; hand the caller its own
       reference, then release the (consumed) array. */
    aria_array_dup_elem(a->kind, slot);
    aria_array_drop(p);
    return slot;
}

/* Clone an array's buffer (dup'ing each Str/Ref element) for copy-on-write. */
static void* aria_array_clone(AriaArray* a) {
    int64_t cap = a->len > 0 ? a->len : 1;
    AriaArray* n = (AriaArray*)aria_array_alloc(a->kind, cap);
    n->len = a->len;
    for (int64_t i = 0; i < a->len; i++) {
        n->elems[i] = a->elems[i];
        aria_array_dup_elem(a->kind, a->elems[i]);
    }
    return (void*)n;
}

static void* aria_array_set(void* p, int64_t i, int64_t x) {
    AriaArray* a = (AriaArray*)p;
    if (i < 0 || i >= a->len) aria_trap_msg("array index out of range");
    if (a->rc == 1) {
        /* FBIP: overwrite in place; drop the displaced element. */
        aria_array_drop_elem(a->kind, a->elems[i]);
        a->elems[i] = x;
        aria_reuses++;
        return p;
    }
    /* Copy-on-write: clone (dup'ing each element), then overwrite slot i with x
       (releasing the freshly-dup'd element that x replaces), drop original. */
    AriaArray* n = (AriaArray*)aria_array_clone(a);
    aria_array_drop_elem(n->kind, n->elems[i]);
    n->elems[i] = x;
    aria_array_drop(p);
    return (void*)n;
}

/* Grow an array (rc==1, owned) so it has room for at least one more element.
   When `cap==0` (a fresh `array_new`) start at a sane minimum (4). The WHOLE
   AriaArray object (header + inline `elems[]`) is reallocated together, so the
   returned pointer may have moved — callers must use it. */
static AriaArray* aria_array_grow(AriaArray* a) {
    if (a->len < a->cap) return a;
    int64_t ncap = a->cap > 0 ? a->cap * 2 : 4;
    AriaArray* g = (AriaArray*)realloc(a, sizeof(AriaArray) + (size_t)ncap * sizeof(int64_t));
    if (!g) aria_trap();
    g->cap = ncap;
    return g;
}

/* `kind` is the AUTHORITATIVE element kind from the (correctly-suffixed) push
   call site. An empty `array_new` may have been tagged with a stale kind by the
   monomorphizer; reconcile the header here while the array still has no live
   element references (len==0), so the kind-aware dup/drop stay correct. */
static void* aria_array_push(void* p, int64_t x, int64_t kind) {
    AriaArray* a = (AriaArray*)p;
    if (a->len == 0) a->kind = kind;
    if (a->rc == 1) {
        /* FBIP: realloc-grow the whole object in place. */
        a = aria_array_grow(a);
        a->elems[a->len++] = x;
        aria_reuses++;
        return (void*)a;
    }
    /* Copy-on-write: clone, append x, drop original. */
    AriaArray* n = (AriaArray*)aria_array_clone(a);
    n->kind = kind;
    n = aria_array_grow(n);
    n->elems[n->len++] = x;
    aria_array_drop(p);
    return (void*)n;
}

/* ---- Bytes: a flat, growable, reference-counted byte buffer ----
   Layout: { rc; len; cap; unsigned char bytes[] }. A byte is an Int 0..255.
   Modeled on the String object plus a `cap` for growth. FBIP: set/push mutate
   in place when rc==1, else copy-on-write. Range policy: bytes_set/bytes_push
   with a value outside 0..255 trap (identical across all backends). */
typedef struct { int64_t rc; int64_t len; int64_t cap; unsigned char bytes[]; } AriaBytes;

static void* aria_bytes_alloc(int64_t cap) {
    AriaBytes* b = (AriaBytes*)malloc(sizeof(AriaBytes) + (size_t)cap);
    if (!b) aria_trap();
    b->rc = 1; b->len = 0; b->cap = cap;
    aria_live++;
    return (void*)b;
}
static void* aria_bytes_new(void) { return aria_bytes_alloc(0); }
static inline void aria_bytes_dup(void* p) { ((AriaBytes*)p)->rc++; }
static void aria_bytes_drop(void* p) {
    AriaBytes* b = (AriaBytes*)p;
    if (--b->rc == 0) { aria_live--; free(b); }
}
static int64_t aria_bytes_len(void* p) {
    int64_t n = ((AriaBytes*)p)->len;
    aria_bytes_drop(p);  /* bytes_len consumes its argument */
    return n;
}
static int64_t aria_bytes_get(void* p, int64_t i) {
    AriaBytes* b = (AriaBytes*)p;
    if (i < 0 || i >= b->len) aria_trap_msg("bytes index out of range");
    int64_t v = (int64_t)b->bytes[i];
    aria_bytes_drop(p);  /* bytes_get consumes its argument */
    return v;
}
static AriaBytes* aria_bytes_clone(AriaBytes* b) {
    int64_t cap = b->len > 0 ? b->len : 1;
    AriaBytes* n = (AriaBytes*)aria_bytes_alloc(cap);
    n->len = b->len;
    memcpy(n->bytes, b->bytes, (size_t)b->len);
    return n;
}
static void* aria_bytes_set(void* p, int64_t i, int64_t v) {
    AriaBytes* b = (AriaBytes*)p;
    if (i < 0 || i >= b->len) aria_trap_msg("bytes index out of range");
    if (v < 0 || v > 255) aria_trap_msg("byte value out of range (must be 0..255)");
    if (b->rc == 1) {
        b->bytes[i] = (unsigned char)v;
        aria_reuses++;
        return p;
    }
    AriaBytes* n = aria_bytes_clone(b);
    n->bytes[i] = (unsigned char)v;
    aria_bytes_drop(p);
    return (void*)n;
}
static AriaBytes* aria_bytes_grow(AriaBytes* b) {
    if (b->len < b->cap) return b;
    int64_t ncap = b->cap > 0 ? b->cap * 2 : 4;
    AriaBytes* g = (AriaBytes*)realloc(b, sizeof(AriaBytes) + (size_t)ncap);
    if (!g) aria_trap();
    g->cap = ncap;
    return g;
}
static void* aria_bytes_push(void* p, int64_t v) {
    AriaBytes* b = (AriaBytes*)p;
    if (v < 0 || v > 255) aria_trap_msg("byte value out of range (must be 0..255)");
    if (b->rc == 1) {
        b = aria_bytes_grow(b);
        b->bytes[b->len++] = (unsigned char)v;
        aria_reuses++;
        return (void*)b;
    }
    AriaBytes* n = aria_bytes_clone(b);
    n = aria_bytes_grow(n);
    n->bytes[n->len++] = (unsigned char)v;
    aria_bytes_drop(p);
    return (void*)n;
}
static void* aria_bytes_from_str(void* p) {
    AriaStr* s = (AriaStr*)p;
    AriaBytes* b = (AriaBytes*)aria_bytes_alloc(s->len > 0 ? s->len : 0);
    b->len = s->len;
    memcpy(b->bytes, s->bytes, (size_t)s->len);
    aria_str_drop(p);  /* the Str argument is consumed */
    return (void*)b;
}
/* Minimal UTF-8 validation: trap on an ill-formed sequence (matches the
   interpreter's clean error on invalid UTF-8). */
static int aria_utf8_valid(const unsigned char* s, int64_t n) {
    int64_t i = 0;
    while (i < n) {
        unsigned char c = s[i];
        int64_t need;            /* number of continuation bytes that follow c */
        unsigned char lo = 0x80, hi = 0xBF;  /* allowed range of the 1st cont. */
        if (c < 0x80) { i++; continue; }
        else if (c >= 0xC2 && c <= 0xDF) need = 1;
        else if (c == 0xE0) { need = 2; lo = 0xA0; }
        else if (c >= 0xE1 && c <= 0xEC) need = 2;
        else if (c == 0xED) { need = 2; hi = 0x9F; }
        else if (c >= 0xEE && c <= 0xEF) need = 2;
        else if (c == 0xF0) { need = 3; lo = 0x90; }
        else if (c >= 0xF1 && c <= 0xF3) need = 3;
        else if (c == 0xF4) { need = 3; hi = 0x8F; }
        else return 0;
        if (i + need >= n) return 0;   /* truncated multibyte sequence */
        for (int64_t k = 1; k <= need; k++) {
            unsigned char cc = s[i + k];
            unsigned char l = (k == 1) ? lo : 0x80;
            unsigned char h = (k == 1) ? hi : 0xBF;
            if (cc < l || cc > h) return 0;
        }
        i += need + 1;
    }
    return 1;
}
static void* aria_bytes_to_str(void* p) {
    AriaBytes* b = (AriaBytes*)p;
    if (!aria_utf8_valid(b->bytes, b->len)) aria_trap_msg("bytes are not valid UTF-8");
    AriaStr* s = (AriaStr*)aria_str_alloc(b->len);
    memcpy(s->bytes, b->bytes, (size_t)b->len);
    aria_bytes_drop(p);  /* the Bytes argument is consumed */
    return (void*)s;
}
/* Structural Bytes equality (content). Consumes neither operand here; the
   `_consume` wrapper releases both for the ==/!= operators. */
static int64_t aria_byteseq(void* a, void* b) {
    AriaBytes* x = (AriaBytes*)a; AriaBytes* y = (AriaBytes*)b;
    if (x->len != y->len) return 0;
    return memcmp(x->bytes, y->bytes, (size_t)x->len) == 0;
}
static int64_t aria_byteseq_consume(void* a, void* b) {
    int64_t r = aria_byteseq(a, b);
    aria_bytes_drop(a); aria_bytes_drop(b);
    return r;
}
/* Print the canonical `Bytes[..]` rendering (does NOT consume the buffer). */
static void aria_print_bytes_value(void* p) {
    AriaBytes* b = (AriaBytes*)p;
    fputs("Bytes[", stdout);
    for (int64_t i = 0; i < b->len; i++) {
        if (i > 0) fputc(' ', stdout);
        printf("%02x", (unsigned)b->bytes[i]);
    }
    fputs("]\n", stdout);
}

/* ---- Vector / Embedding (dense, immutable buffer of f64) ------------------
   An AriaVector is a flat heap buffer of `double`, modeled exactly on AriaBytes
   but with f64 elements. push/add/scale are FBIP: they mutate in place when
   rc==1, else copy-on-write — always garbage-free. dot/cosine/add on two vectors
   of UNEQUAL length trap (clean runtime error, matching the interpreter). cosine
   returns 0.0 when either operand has L2 norm 0 (no divide-by-zero -> NaN). The
   `Vector[..]` display uses the SAME shortest-round-trip `aria_fmt_float` as a
   scalar Float, so it is byte-for-byte identical to the interpreter. */
typedef struct { int64_t rc; int64_t len; int64_t cap; double elems[]; } AriaVector;
static void aria_fmt_float(double d, char* buf, size_t cap);            /* fwd */

static void* aria_vec_alloc(int64_t cap) {
    AriaVector* v = (AriaVector*)malloc(sizeof(AriaVector) + (size_t)cap * sizeof(double));
    if (!v) aria_trap();
    v->rc = 1; v->len = 0; v->cap = cap;
    aria_live++;
    return (void*)v;
}
static void* aria_vec_new(void) { return aria_vec_alloc(0); }
static inline void aria_vec_dup(void* p) { ((AriaVector*)p)->rc++; }
static void aria_vec_drop(void* p) {
    AriaVector* v = (AriaVector*)p;
    if (--v->rc == 0) { aria_live--; free(v); }
}
static int64_t aria_vec_len(void* p) {
    int64_t n = ((AriaVector*)p)->len;
    aria_vec_drop(p);  /* vec_len consumes its argument */
    return n;
}
static double aria_vec_get(void* p, int64_t i) {
    AriaVector* v = (AriaVector*)p;
    if (i < 0 || i >= v->len) aria_trap_msg("vector index out of range");
    double x = v->elems[i];
    aria_vec_drop(p);  /* vec_get consumes its argument */
    return x;
}
static AriaVector* aria_vec_clone(AriaVector* v) {
    int64_t cap = v->len > 0 ? v->len : 1;
    AriaVector* n = (AriaVector*)aria_vec_alloc(cap);
    n->len = v->len;
    memcpy(n->elems, v->elems, (size_t)v->len * sizeof(double));
    return n;
}
static AriaVector* aria_vec_grow(AriaVector* v) {
    if (v->len < v->cap) return v;
    int64_t ncap = v->cap > 0 ? v->cap * 2 : 4;
    AriaVector* g = (AriaVector*)realloc(v, sizeof(AriaVector) + (size_t)ncap * sizeof(double));
    if (!g) aria_trap();
    g->cap = ncap;
    return g;
}
static void* aria_vec_push(void* p, double x) {
    AriaVector* v = (AriaVector*)p;
    if (v->rc == 1) {
        v = aria_vec_grow(v);
        v->elems[v->len++] = x;
        aria_reuses++;
        return (void*)v;
    }
    AriaVector* n = aria_vec_clone(v);
    n = aria_vec_grow(n);
    n->elems[n->len++] = x;
    aria_vec_drop(p);
    return (void*)n;
}
/* Build a Vector from an Array[Float] (kind code 1). Consumes the array. */
static void* aria_vec_from_array(void* p) {
    AriaArray* a = (AriaArray*)p;
    AriaVector* v = (AriaVector*)aria_vec_alloc(a->len > 0 ? a->len : 0);
    v->len = a->len;
    for (int64_t i = 0; i < a->len; i++) v->elems[i] = aria_i2f(a->elems[i]);
    aria_array_drop(p);  /* the Array argument is consumed */
    return (void*)v;
}
/* Build an Array[Float] (kind code 1) from a Vector. Consumes the vector. */
static void* aria_vec_to_array(void* p) {
    AriaVector* v = (AriaVector*)p;
    void* a = aria_array_new(INT64_C(1));
    for (int64_t i = 0; i < v->len; i++) a = aria_array_push(a, aria_f2i(v->elems[i]), INT64_C(1));
    aria_vec_drop(p);  /* the Vector argument is consumed */
    return a;
}
/* Sum of elementwise products (left-to-right, matching the interpreter's order
   so the float result is byte-for-byte identical). */
static double aria_vec_dot_raw(AriaVector* x, AriaVector* y) {
    double acc = 0.0;
    for (int64_t i = 0; i < x->len; i++) acc += x->elems[i] * y->elems[i];
    return acc;
}
static double aria_vec_dot(void* a, void* b) {
    AriaVector* x = (AriaVector*)a; AriaVector* y = (AriaVector*)b;
    if (x->len != y->len) aria_trap_msg("vector length mismatch");  /* clean trap */
    double r = aria_vec_dot_raw(x, y);
    aria_vec_drop(a); aria_vec_drop(b);  /* both operands consumed */
    return r;
}
static double aria_vec_norm(void* a) {
    AriaVector* x = (AriaVector*)a;
    double r = sqrt(aria_vec_dot_raw(x, x));
    aria_vec_drop(a);
    return r;
}
static double aria_vec_cosine(void* a, void* b) {
    AriaVector* x = (AriaVector*)a; AriaVector* y = (AriaVector*)b;
    if (x->len != y->len) aria_trap_msg("vector length mismatch");  /* clean trap */
    double nx = sqrt(aria_vec_dot_raw(x, x));
    double ny = sqrt(aria_vec_dot_raw(y, y));
    double r;
    /* Zero-norm policy: return 0.0 (never divide by zero -> NaN). */
    if (nx == 0.0 || ny == 0.0) r = 0.0;
    else r = aria_vec_dot_raw(x, y) / (nx * ny);
    aria_vec_drop(a); aria_vec_drop(b);
    return r;
}
/* Elementwise add. FBIP: reuse the first operand in place when it is unique. */
static void* aria_vec_add(void* a, void* b) {
    AriaVector* x = (AriaVector*)a; AriaVector* y = (AriaVector*)b;
    if (x->len != y->len) aria_trap_msg("vector length mismatch");  /* clean trap */
    if (x->rc == 1) {
        for (int64_t i = 0; i < x->len; i++) x->elems[i] += y->elems[i];
        aria_reuses++;
        aria_vec_drop(b);
        return a;
    }
    AriaVector* n = aria_vec_clone(x);
    for (int64_t i = 0; i < n->len; i++) n->elems[i] += y->elems[i];
    aria_vec_drop(a); aria_vec_drop(b);
    return (void*)n;
}
/* Multiply every element by a scalar. FBIP when the operand is unique. */
static void* aria_vec_scale(void* a, double s) {
    AriaVector* x = (AriaVector*)a;
    if (x->rc == 1) {
        for (int64_t i = 0; i < x->len; i++) x->elems[i] *= s;
        aria_reuses++;
        return a;
    }
    AriaVector* n = aria_vec_clone(x);
    for (int64_t i = 0; i < n->len; i++) n->elems[i] *= s;
    aria_vec_drop(a);
    return (void*)n;
}
/* Structural Vector equality (length + exact element bits). Does NOT consume. */
static int64_t aria_veceq(void* a, void* b) {
    AriaVector* x = (AriaVector*)a; AriaVector* y = (AriaVector*)b;
    if (x->len != y->len) return 0;
    for (int64_t i = 0; i < x->len; i++) if (x->elems[i] != y->elems[i]) return 0;
    return 1;
}
static int64_t aria_veceq_consume(void* a, void* b) {
    int64_t r = aria_veceq(a, b);
    aria_vec_drop(a); aria_vec_drop(b);
    return r;
}
/* Print the canonical `Vector[..]` rendering (does NOT consume the buffer). */
static void aria_print_vec_value(void* p) {
    AriaVector* v = (AriaVector*)p;
    char tmp[320];
    fputs("Vector[", stdout);
    for (int64_t i = 0; i < v->len; i++) {
        if (i > 0) fputs(", ", stdout);
        aria_fmt_float(v->elems[i], tmp, sizeof tmp);
        fputs(tmp, stdout);
    }
    fputs("]\n", stdout);
}

/* ---- Ordered Map and Set -------------------------------------------------
   An AriaMap is kept SORTED BY KEY ascending; an AriaSet sorted by element.
   This makes iteration / display / equality deterministic and identical to the
   interpreter (the differential oracle). Storage is an insertion-sorted array of
   int64 slots: a Map stores interleaved [k0,v0,k1,v1,...], a Set stores
   [e0,e1,...]. Each slot is encoded like the array runtime (Int directly, Float
   via aria_f2i, heap pointer via uintptr_t). `kkind`/`vkind` (the SlotKind codes
   0=Int,1=Float,2=Str,3=Ref,4=Bytes,5=Array,6=Map,7=Set) drive the kind-aware
   dup/drop. Keys are only ever Int(0) or Str(2). FBIP: insert/add/remove mutate
   in place when rc==1, else copy-on-write — always garbage-free. */
static int64_t aria_eq(void* a, void* b);
static int64_t aria_byteseq(void* a, void* b);
static int64_t aria_streq(void* a, void* b);
static void aria_print_int_inline(int64_t n);
static void aria_print_float_inline(double d);
static void aria_map_dup(void* p);
static void aria_map_drop(void* p);
static void aria_set_dup(void* p);
static void aria_set_drop(void* p);
static int64_t aria_map_eq(void* a, void* b);
static int64_t aria_set_eq(void* a, void* b);
static void aria_print_map_value(void* p);
static void aria_print_set_value(void* p);

/* Dup / drop a single slot by its SlotKind code. */
static void aria_slot_dup(int64_t kind, int64_t slot) {
    switch (kind) {
        case 2: aria_str_dup((void*)(uintptr_t)slot); break;
        case 3: aria_dup((void*)(uintptr_t)slot); break;
        case 4: aria_bytes_dup((void*)(uintptr_t)slot); break;
        case 5: aria_array_dup((void*)(uintptr_t)slot); break;
        case 6: aria_map_dup((void*)(uintptr_t)slot); break;
        case 7: aria_set_dup((void*)(uintptr_t)slot); break;
        case 9: aria_vec_dup((void*)(uintptr_t)slot); break;
        default: break;  /* Int(0)/Float(1)/Bool(8): nothing to dup */
    }
}
static void aria_slot_drop(int64_t kind, int64_t slot) {
    switch (kind) {
        case 2: aria_str_drop((void*)(uintptr_t)slot); break;
        case 3: aria_drop((void*)(uintptr_t)slot); break;
        case 4: aria_bytes_drop((void*)(uintptr_t)slot); break;
        case 5: aria_array_drop((void*)(uintptr_t)slot); break;
        case 6: aria_map_drop((void*)(uintptr_t)slot); break;
        case 7: aria_set_drop((void*)(uintptr_t)slot); break;
        case 9: aria_vec_drop((void*)(uintptr_t)slot); break;
        default: break;  /* Int(0)/Float(1)/Bool(8): nothing to drop */
    }
}
/* Total ordering on keys/elements. kind is 0 (Int) or 2 (Str). Returns <0/0/>0. */
static int aria_key_cmp(int64_t kind, int64_t a, int64_t b) {
    if (kind == 2) {
        AriaStr* x = (AriaStr*)(uintptr_t)a; AriaStr* y = (AriaStr*)(uintptr_t)b;
        int64_t n = x->len < y->len ? x->len : y->len;
        int c = memcmp(x->bytes, y->bytes, (size_t)n);
        if (c != 0) return c;
        if (x->len < y->len) return -1;
        if (x->len > y->len) return 1;
        return 0;
    }
    /* Int (kind 0): signed numeric order. */
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}
/* Compare two value slots for equality, kind-aware. */
static int64_t aria_slot_eq(int64_t kind, int64_t a, int64_t b) {
    switch (kind) {
        case 1: return aria_i2f(a) == aria_i2f(b);                  /* Float */
        case 2: return aria_streq((void*)(uintptr_t)a, (void*)(uintptr_t)b);
        case 3: return aria_eq((void*)(uintptr_t)a, (void*)(uintptr_t)b);
        case 4: return aria_byteseq((void*)(uintptr_t)a, (void*)(uintptr_t)b);
        case 6: return aria_map_eq((void*)(uintptr_t)a, (void*)(uintptr_t)b);
        case 7: return aria_set_eq((void*)(uintptr_t)a, (void*)(uintptr_t)b);
        default: return a == b;                                      /* Int */
    }
}
/* Print a single value slot inline (no trailing newline), matching the
   interpreter's display for that value type. */
static void aria_print_slot(int64_t kind, int64_t slot) {
    switch (kind) {
        case 1: aria_print_float_inline(aria_i2f(slot)); break;
        case 2: { AriaStr* s = (AriaStr*)(uintptr_t)slot; fwrite(s->bytes, 1, (size_t)s->len, stdout); break; }
        case 4: aria_print_bytes_value((void*)(uintptr_t)slot); break;  /* Bytes already has its own form */
        case 6: aria_print_map_value((void*)(uintptr_t)slot); break;
        case 7: aria_print_set_value((void*)(uintptr_t)slot); break;
        case 8: fputs(slot ? "true" : "false", stdout); break;  /* Bool */
        default: aria_print_int_inline(slot); break;  /* Int */
    }
}

typedef struct { int64_t rc; int64_t kkind; int64_t vkind; int64_t len; int64_t cap; int64_t slots[]; } AriaMap;

static void* aria_map_alloc(int64_t kkind, int64_t vkind, int64_t cap) {
    AriaMap* m = (AriaMap*)malloc(sizeof(AriaMap) + (size_t)cap * 2 * sizeof(int64_t));
    if (!m) aria_trap();
    m->rc = 1; m->kkind = kkind; m->vkind = vkind; m->len = 0; m->cap = cap;
    aria_live++;
    return (void*)m;
}
static void* aria_map_new(int64_t kkind, int64_t vkind) { return aria_map_alloc(kkind, vkind, 0); }
static void aria_map_dup(void* p) { ((AriaMap*)p)->rc++; }
static void aria_map_drop(void* p) {
    AriaMap* m = (AriaMap*)p;
    if (--m->rc == 0) {
        for (int64_t i = 0; i < m->len; i++) {
            aria_slot_drop(m->kkind, m->slots[2*i]);
            aria_slot_drop(m->vkind, m->slots[2*i+1]);
        }
        aria_live--;
        free(m);
    }
}
static int64_t aria_map_len(void* p) {
    int64_t n = ((AriaMap*)p)->len;
    aria_map_drop(p);  /* map_len consumes its argument */
    return n;
}
/* Binary search for `key` (kind kkind); returns the index if present (sets
   *found=1) or the insertion point (*found=0). */
static int64_t aria_map_find(AriaMap* m, int64_t key, int* found) {
    int64_t lo = 0, hi = m->len;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int c = aria_key_cmp(m->kkind, m->slots[2*mid], key);
        if (c == 0) { *found = 1; return mid; }
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    *found = 0;
    return lo;
}
static AriaMap* aria_map_clone(AriaMap* m) {
    int64_t cap = m->len > 0 ? m->len : 1;
    AriaMap* n = (AriaMap*)aria_map_alloc(m->kkind, m->vkind, cap);
    n->len = m->len;
    for (int64_t i = 0; i < m->len; i++) {
        n->slots[2*i]   = m->slots[2*i];   aria_slot_dup(m->kkind, m->slots[2*i]);
        n->slots[2*i+1] = m->slots[2*i+1]; aria_slot_dup(m->vkind, m->slots[2*i+1]);
    }
    return n;
}
static AriaMap* aria_map_grow(AriaMap* m) {
    if (m->len < m->cap) return m;
    int64_t ncap = m->cap > 0 ? m->cap * 2 : 4;
    AriaMap* g = (AriaMap*)realloc(m, sizeof(AriaMap) + (size_t)ncap * 2 * sizeof(int64_t));
    if (!g) aria_trap();
    g->cap = ncap;
    return g;
}
/* Insert (or replace) key->val. Ownership of `key` and `val` transfers in. A
   replaced key's old value is dropped; the duplicate key (on replace) is dropped.
   FBIP in place when rc==1, else copy-on-write. The argument map is consumed. */
static void* aria_map_insert(void* p, int64_t key, int64_t val, int64_t kkind, int64_t vkind) {
    AriaMap* m = (AriaMap*)p;
    if (m->len == 0) { m->kkind = kkind; m->vkind = vkind; }
    if (m->rc != 1) {
        AriaMap* n = aria_map_clone(m);
        aria_map_drop(p);
        m = n;
    } else {
        aria_reuses++;  /* FBIP: mutate the uniquely-owned buffer in place */
    }
    int found = 0;
    int64_t idx = aria_map_find(m, key, &found);
    if (found) {
        aria_slot_drop(m->vkind, m->slots[2*idx+1]);  /* drop displaced value */
        m->slots[2*idx+1] = val;
        aria_slot_drop(m->kkind, key);                /* duplicate key not stored */
        return (void*)m;
    }
    m = aria_map_grow(m);
    /* Shift entries [idx..len) right by one to open a slot at idx. */
    for (int64_t i = m->len; i > idx; i--) {
        m->slots[2*i]   = m->slots[2*(i-1)];
        m->slots[2*i+1] = m->slots[2*(i-1)+1];
    }
    m->slots[2*idx] = key;
    m->slots[2*idx+1] = val;
    m->len++;
    return (void*)m;
}
/* Total read: returns the stored value (dup'd) if present, else `dflt`. The
   not-taken branch's slot is dropped so exactly one value reference is returned.
   The map argument is consumed. */
static int64_t aria_map_get_or(void* p, int64_t key, int64_t dflt, int64_t kkind, int64_t vkind) {
    AriaMap* m = (AriaMap*)p;
    (void)kkind;
    int found = 0;
    int64_t idx = aria_map_find(m, key, &found);
    int64_t result;
    if (found) {
        result = m->slots[2*idx+1];
        aria_slot_dup(vkind, result);     /* hand caller its own reference */
        aria_slot_drop(vkind, dflt);      /* default not used */
    } else {
        result = dflt;                    /* ownership of dflt passes to caller */
    }
    aria_slot_drop(kkind, key);           /* the lookup key is consumed */
    aria_map_drop(p);
    return result;
}
static int64_t aria_map_has(void* p, int64_t key, int64_t kkind) {
    AriaMap* m = (AriaMap*)p;
    int found = 0;
    (void)aria_map_find(m, key, &found);
    aria_slot_drop(kkind, key);
    aria_map_drop(p);
    return found ? 1 : 0;
}
/* Remove key if present. FBIP in place when rc==1, else copy-on-write. */
static void* aria_map_remove(void* p, int64_t key, int64_t kkind) {
    AriaMap* m = (AriaMap*)p;
    if (m->rc != 1) {
        AriaMap* n = aria_map_clone(m);
        aria_map_drop(p);
        m = n;
    } else {
        aria_reuses++;
    }
    int found = 0;
    int64_t idx = aria_map_find(m, key, &found);
    if (found) {
        aria_slot_drop(m->kkind, m->slots[2*idx]);
        aria_slot_drop(m->vkind, m->slots[2*idx+1]);
        for (int64_t i = idx; i + 1 < m->len; i++) {
            m->slots[2*i]   = m->slots[2*(i+1)];
            m->slots[2*i+1] = m->slots[2*(i+1)+1];
        }
        m->len--;
    }
    aria_slot_drop(kkind, key);  /* the lookup key is consumed */
    return (void*)m;
}
static int64_t aria_map_eq(void* a, void* b) {
    AriaMap* x = (AriaMap*)a; AriaMap* y = (AriaMap*)b;
    if (x->len != y->len) return 0;
    for (int64_t i = 0; i < x->len; i++) {
        if (aria_key_cmp(x->kkind, x->slots[2*i], y->slots[2*i]) != 0) return 0;
        if (!aria_slot_eq(x->vkind, x->slots[2*i+1], y->slots[2*i+1])) return 0;
    }
    return 1;
}
static int64_t aria_map_eq_consume(void* a, void* b) {
    int64_t r = aria_map_eq(a, b);
    aria_map_drop(a); aria_map_drop(b);
    return r;
}
static void aria_print_map_value(void* p) {
    AriaMap* m = (AriaMap*)p;
    fputs("Map[", stdout);
    for (int64_t i = 0; i < m->len; i++) {
        if (i > 0) fputs(", ", stdout);
        aria_print_slot(m->kkind, m->slots[2*i]);
        fputs(": ", stdout);
        aria_print_slot(m->vkind, m->slots[2*i+1]);
    }
    fputs("]", stdout);
}

typedef struct { int64_t rc; int64_t ekind; int64_t len; int64_t cap; int64_t slots[]; } AriaSet;

static void* aria_set_alloc(int64_t ekind, int64_t cap) {
    AriaSet* s = (AriaSet*)malloc(sizeof(AriaSet) + (size_t)cap * sizeof(int64_t));
    if (!s) aria_trap();
    s->rc = 1; s->ekind = ekind; s->len = 0; s->cap = cap;
    aria_live++;
    return (void*)s;
}
static void* aria_set_new(int64_t ekind) { return aria_set_alloc(ekind, 0); }
static void aria_set_dup(void* p) { ((AriaSet*)p)->rc++; }
static void aria_set_drop(void* p) {
    AriaSet* s = (AriaSet*)p;
    if (--s->rc == 0) {
        for (int64_t i = 0; i < s->len; i++) aria_slot_drop(s->ekind, s->slots[i]);
        aria_live--;
        free(s);
    }
}
static int64_t aria_set_len(void* p) {
    int64_t n = ((AriaSet*)p)->len;
    aria_set_drop(p);
    return n;
}
static int64_t aria_set_find(AriaSet* s, int64_t e, int* found) {
    int64_t lo = 0, hi = s->len;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int c = aria_key_cmp(s->ekind, s->slots[mid], e);
        if (c == 0) { *found = 1; return mid; }
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    *found = 0;
    return lo;
}
static AriaSet* aria_set_clone(AriaSet* s) {
    int64_t cap = s->len > 0 ? s->len : 1;
    AriaSet* n = (AriaSet*)aria_set_alloc(s->ekind, cap);
    n->len = s->len;
    for (int64_t i = 0; i < s->len; i++) { n->slots[i] = s->slots[i]; aria_slot_dup(s->ekind, s->slots[i]); }
    return n;
}
static AriaSet* aria_set_grow(AriaSet* s) {
    if (s->len < s->cap) return s;
    int64_t ncap = s->cap > 0 ? s->cap * 2 : 4;
    AriaSet* g = (AriaSet*)realloc(s, sizeof(AriaSet) + (size_t)ncap * sizeof(int64_t));
    if (!g) aria_trap();
    g->cap = ncap;
    return g;
}
/* Add an element (ownership transfers in). An already-present element is a
   no-op (its incoming reference is dropped). FBIP in place when rc==1. */
static void* aria_set_add(void* p, int64_t e, int64_t ekind) {
    AriaSet* s = (AriaSet*)p;
    if (s->len == 0) s->ekind = ekind;
    if (s->rc != 1) {
        AriaSet* n = aria_set_clone(s);
        aria_set_drop(p);
        s = n;
    } else {
        aria_reuses++;
    }
    int found = 0;
    int64_t idx = aria_set_find(s, e, &found);
    if (found) { aria_slot_drop(s->ekind, e); return (void*)s; }
    s = aria_set_grow(s);
    for (int64_t i = s->len; i > idx; i--) s->slots[i] = s->slots[i-1];
    s->slots[idx] = e;
    s->len++;
    return (void*)s;
}
static int64_t aria_set_has(void* p, int64_t e, int64_t ekind) {
    AriaSet* s = (AriaSet*)p;
    int found = 0;
    (void)aria_set_find(s, e, &found);
    aria_slot_drop(ekind, e);
    aria_set_drop(p);
    return found ? 1 : 0;
}
static void* aria_set_remove(void* p, int64_t e, int64_t ekind) {
    AriaSet* s = (AriaSet*)p;
    if (s->rc != 1) {
        AriaSet* n = aria_set_clone(s);
        aria_set_drop(p);
        s = n;
    } else {
        aria_reuses++;
    }
    int found = 0;
    int64_t idx = aria_set_find(s, e, &found);
    if (found) {
        aria_slot_drop(s->ekind, s->slots[idx]);
        for (int64_t i = idx; i + 1 < s->len; i++) s->slots[i] = s->slots[i+1];
        s->len--;
    }
    aria_slot_drop(ekind, e);
    return (void*)s;
}
static int64_t aria_set_eq(void* a, void* b) {
    AriaSet* x = (AriaSet*)a; AriaSet* y = (AriaSet*)b;
    if (x->len != y->len) return 0;
    for (int64_t i = 0; i < x->len; i++)
        if (aria_key_cmp(x->ekind, x->slots[i], y->slots[i]) != 0) return 0;
    return 1;
}
static int64_t aria_set_eq_consume(void* a, void* b) {
    int64_t r = aria_set_eq(a, b);
    aria_set_drop(a); aria_set_drop(b);
    return r;
}
static void aria_print_set_value(void* p) {
    AriaSet* s = (AriaSet*)p;
    fputs("Set[", stdout);
    for (int64_t i = 0; i < s->len; i++) {
        if (i > 0) fputs(", ", stdout);
        aria_print_slot(s->ekind, s->slots[i]);
    }
    fputs("]", stdout);
}

/* ---- map_show / set_show: render the canonical string into an AriaStr ----
   A growable byte buffer mirrors `aria_print_slot`, so the produced text is
   byte-for-byte identical to the interpreter's `Value::display`. */
static void aria_fmt_float(double d, char* buf, size_t cap);  /* fwd */
typedef struct { char* p; size_t len; size_t cap; } AriaSB;
static void aria_sb_reserve(AriaSB* b, size_t extra) {
    if (b->len + extra + 1 > b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 32;
        while (b->len + extra + 1 > ncap) ncap *= 2;
        char* np = (char*)realloc(b->p, ncap);
        if (!np) aria_trap();
        b->p = np; b->cap = ncap;
    }
}
static void aria_sb_puts(AriaSB* b, const char* s, size_t n) {
    aria_sb_reserve(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
}
static void aria_sb_cstr(AriaSB* b, const char* s) { aria_sb_puts(b, s, strlen(s)); }
/* Append a single value slot's rendering (no separators). */
static void aria_sb_slot(AriaSB* b, int64_t kind, int64_t slot) {
    char tmp[330];
    switch (kind) {
        case 1: { aria_fmt_float(aria_i2f(slot), tmp, sizeof tmp); aria_sb_cstr(b, tmp); break; }
        case 2: { AriaStr* s = (AriaStr*)(uintptr_t)slot; aria_sb_puts(b, s->bytes, (size_t)s->len); break; }
        case 4: { /* Bytes: `Bytes[xx xx]` */
            AriaBytes* by = (AriaBytes*)(uintptr_t)slot;
            aria_sb_cstr(b, "Bytes[");
            for (int64_t i = 0; i < by->len; i++) {
                if (i > 0) aria_sb_cstr(b, " ");
                snprintf(tmp, sizeof tmp, "%02x", (unsigned)by->bytes[i]);
                aria_sb_cstr(b, tmp);
            }
            aria_sb_cstr(b, "]");
            break;
        }
        case 8: aria_sb_cstr(b, slot ? "true" : "false"); break;  /* Bool */
        default: { snprintf(tmp, sizeof tmp, "%lld", (long long)slot); aria_sb_cstr(b, tmp); break; }
    }
}
static AriaStr* aria_sb_finish(AriaSB* b) {
    AriaStr* s = (AriaStr*)aria_str_alloc((int64_t)b->len);
    memcpy(s->bytes, b->p, b->len);
    free(b->p);
    return s;
}
static void* aria_map_show(void* p) {
    AriaMap* m = (AriaMap*)p;
    AriaSB b = {0,0,0};
    aria_sb_cstr(&b, "Map[");
    for (int64_t i = 0; i < m->len; i++) {
        if (i > 0) aria_sb_cstr(&b, ", ");
        aria_sb_slot(&b, m->kkind, m->slots[2*i]);
        aria_sb_cstr(&b, ": ");
        aria_sb_slot(&b, m->vkind, m->slots[2*i+1]);
    }
    aria_sb_cstr(&b, "]");
    AriaStr* s = aria_sb_finish(&b);
    aria_map_drop(p);  /* map_show consumes its argument */
    return (void*)s;
}
static void* aria_set_show(void* p) {
    AriaSet* st = (AriaSet*)p;
    AriaSB b = {0,0,0};
    aria_sb_cstr(&b, "Set[");
    for (int64_t i = 0; i < st->len; i++) {
        if (i > 0) aria_sb_cstr(&b, ", ");
        aria_sb_slot(&b, st->ekind, st->slots[i]);
    }
    aria_sb_cstr(&b, "]");
    AriaStr* s = aria_sb_finish(&b);
    aria_set_drop(p);
    return (void*)s;
}

/* ---- map_keys / map_values / set_to_array: enumerate into an AriaArray ----
   Build a FRESH AriaArray of element kind `out_kind` from the map's keys /
   values or the set's elements, IN ASCENDING (key-/element-sorted) order — the
   same deterministic order as display/equality, so iteration is stable across
   backends. Each element is dup'd into the new array using the source slot kind
   (`src_kind`), then the source map/set is consumed (dropped), leaving no
   garbage. `out_kind` is the array element kind (Bool collapses to Int(0), a
   no-op for dup/drop). */
static void* aria_map_keys(void* p, int64_t src_kind, int64_t out_kind) {
    AriaMap* m = (AriaMap*)p;
    void* a = aria_array_new(out_kind);
    for (int64_t i = 0; i < m->len; i++) {
        int64_t slot = m->slots[2*i];
        aria_slot_dup(src_kind, slot);
        a = aria_array_push(a, slot, out_kind);
    }
    aria_map_drop(p);  /* map_keys consumes its argument */
    return a;
}
static void* aria_map_values(void* p, int64_t src_kind, int64_t out_kind) {
    AriaMap* m = (AriaMap*)p;
    void* a = aria_array_new(out_kind);
    for (int64_t i = 0; i < m->len; i++) {
        int64_t slot = m->slots[2*i+1];
        aria_slot_dup(src_kind, slot);
        a = aria_array_push(a, slot, out_kind);
    }
    aria_map_drop(p);  /* map_values consumes its argument */
    return a;
}
static void* aria_set_to_array(void* p, int64_t src_kind, int64_t out_kind) {
    AriaSet* s = (AriaSet*)p;
    void* a = aria_array_new(out_kind);
    for (int64_t i = 0; i < s->len; i++) {
        int64_t slot = s->slots[i];
        aria_slot_dup(src_kind, slot);
        a = aria_array_push(a, slot, out_kind);
    }
    aria_set_drop(p);  /* set_to_array consumes its argument */
    return a;
}

/* ---- structural equality (per-tag, emitted below) ---- */

static int64_t aria_streq(void* a, void* b) {
    AriaStr* x = (AriaStr*)a; AriaStr* y = (AriaStr*)b;
    if (x->len != y->len) return 0;
    return memcmp(x->bytes, y->bytes, (size_t)x->len) == 0;
}

/* The ==/!= operators consume their operands (the rc pass dups reused values),
   so these variants compare then release both. */
static int64_t aria_streq_consume(void* a, void* b) {
    int64_t r = aria_streq(a, b);
    aria_str_drop(a); aria_str_drop(b);
    return r;
}
static int64_t aria_eq_consume(void* a, void* b) {
    int64_t r = aria_eq(a, b);
    aria_drop(a); aria_drop(b);
    return r;
}

/* ---- concat / int_to_str ---- */
static void* aria_concat(void* a, void* b) {
    AriaStr* x = (AriaStr*)a; AriaStr* y = (AriaStr*)b;
    AriaStr* r = (AriaStr*)aria_str_alloc(x->len + y->len);
    memcpy(r->bytes, x->bytes, (size_t)x->len);
    memcpy(r->bytes + x->len, y->bytes, (size_t)y->len);
    aria_str_drop(a); aria_str_drop(b);  /* arguments are consumed */
    return (void*)r;
}
static void* aria_int_to_str(int64_t n) {
    char buf[24];
    int len = snprintf(buf, sizeof(buf), "%lld", (long long)n);
    AriaStr* r = (AriaStr*)aria_str_alloc(len);
    memcpy(r->bytes, buf, (size_t)len);
    return (void*)r;
}

/* ---- print helpers (match the interpreter's formatting) ---- */
static void aria_print_int(int64_t n) { printf("%lld\n", (long long)n); }
static void aria_print_bool(int64_t b) { fputs(b ? "true\n" : "false\n", stdout); }
/* Format `d` as the SHORTEST decimal that round-trips back to the same f64, in
   plain (non-scientific) notation -- matching the interpreter's Rust
   `format!("{}", f)` and the wasm backend's JS rendering. C's `%g` would pick a
   shortest significant-digit count but may switch to scientific notation
   (`100` -> `1e+02`), which Rust never does. So: find the shortest sig-digit
   count whose `%.*e` round-trips, then lay the digits out as a plain decimal. */
static void aria_fmt_float(double d, char* buf, size_t cap) {
    if (d != d) { snprintf(buf, cap, "NaN"); return; }
    if (d == 1.0/0.0) { snprintf(buf, cap, "inf"); return; }
    if (d == -1.0/0.0) { snprintf(buf, cap, "-inf"); return; }
    char sci[40];
    int p = 17;
    for (int q = 1; q <= 17; q++) {
        snprintf(sci, sizeof sci, "%.*e", q - 1, d);
        if (strtod(sci, NULL) == d) { p = q; break; }
    }
    (void)p;
    const char* s = sci;
    char* out = buf;
    if (*s == '-') { *out++ = '-'; s++; }
    char digits[20];
    int nd = 0;
    digits[nd++] = *s++;
    if (*s == '.') { s++; while (*s >= '0' && *s <= '9') digits[nd++] = *s++; }
    int exp10 = 0;
    if (*s == 'e' || *s == 'E') exp10 = (int)strtol(s + 1, NULL, 10);
    while (nd > 1 && digits[nd - 1] == '0') nd--;
    int point = exp10 + 1;
    if (point <= 0) {
        *out++ = '0'; *out++ = '.';
        for (int i = 0; i < -point; i++) *out++ = '0';
        for (int i = 0; i < nd; i++) *out++ = digits[i];
    } else if (point >= nd) {
        for (int i = 0; i < nd; i++) *out++ = digits[i];
        for (int i = nd; i < point; i++) *out++ = '0';
    } else {
        for (int i = 0; i < nd; i++) { if (i == point) *out++ = '.'; *out++ = digits[i]; }
    }
    *out = '\0';
}
static void aria_print_float(double d) {
    char buf[320];
    aria_fmt_float(d, buf, sizeof buf);
    printf("%s\n", buf);
}
/* Inline (no-newline) scalar renderers used by the Map/Set value printer. */
static void aria_print_int_inline(int64_t n) { printf("%lld", (long long)n); }
static void aria_print_float_inline(double d) {
    char buf[320];
    aria_fmt_float(d, buf, sizeof buf);
    fputs(buf, stdout);
}
static void aria_print_str(void* p) {
    AriaStr* s = (AriaStr*)p;
    fwrite(s->bytes, 1, (size_t)s->len, stdout);
    fputc('\n', stdout);
    aria_str_drop(p);  /* argument consumed */
}
"#;

// ---- top-level compile ---------------------------------------------------

/// Compile a typed `Program` to portable C source, or return a clean `Err` for
/// programs outside the supported subset.
pub fn compile(program: &Program) -> Result<String, String> {
    // 0. Monomorphize: specialize every generic function/ADT reachable from main
    //    so the rest of the backend sees only concrete types.
    let mono = crate::monomorphize::monomorphize(program)?;
    let program = &mono;

    // 1. Function signatures (declaration order).
    let mut sigs: HashMap<String, FnSig> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            let params = f
                .params
                .iter()
                .map(|p| CType::from_ty(&p.ty))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("fn `{}`: {}", f.name, e))?;
            let ret = CType::from_ty(&f.ret).map_err(|e| format!("fn `{}`: {}", f.name, e))?;
            order.push(f.name.clone());
            sigs.insert(f.name.clone(), FnSig { params, ret });
        }
    }
    if !sigs.contains_key("main") {
        return Err("c backend: no `main` function".into());
    }
    {
        let m = &sigs["main"];
        if !m.params.is_empty() {
            return Err("c backend: `main` must take no parameters".into());
        }
        // main may return Int, Float, Bool, String, or Bytes; the runner prints it.
        if !matches!(
            m.ret,
            CType::Int | CType::Float | CType::Bool | CType::Str | CType::Bytes | CType::Vector
        ) {
            return Err(
                "c backend: `main` must return Int, Bool, Float, String, Bytes, or Vector".into(),
            );
        }
    }

    // 1b. Constructor table (tags + field types). Rejects generics / bad fields.
    let ctors = CtorTable::build(program)?;

    // 2. Lower to ANF IR and insert reference-count + reuse ops — REUSING the
    //    existing pipeline exactly like the wasm backend.
    let lowered: HashMap<String, IFn> = ir::lower_program(program)?;
    let rcd: HashMap<String, IFn> = crate::rc::insert_rc(&lowered);
    // Self-tail-call elimination: rewrite each self-tail-recursive function into
    // a loop (`TailCall` back-edges). Runs after rc so ownership of the new args
    // transfers to the params exactly as a real call's binding would.
    let fns: HashMap<String, IFn> = ir::tail_call_optimize(rcd);

    // 2b. Closure table: every lowered function carrying a `lam_sig` is a lifted
    //     lambda. Assign each a closure tag past the last constructor tag (so the
    //     tag never collides with an ADT tag) and a function-table index.
    let mut lam_names: Vec<String> = fns
        .iter()
        .filter(|(_, f)| f.lam_sig.is_some())
        .map(|(n, _)| n.clone())
        .collect();
    lam_names.sort();
    let closure_base = ctors.by_name.len() as i64;
    let mut tags = HashMap::new();
    for (i, n) in lam_names.iter().enumerate() {
        tags.insert(n.clone(), closure_base + i as i64);
    }
    let closures = ClosureTable { base: closure_base, names: lam_names.clone(), tags };

    // 3. Collect string literals -> a stable C global per distinct literal.
    let mut lit_list: Vec<Vec<u8>> = Vec::new();
    for name in order.iter().chain(lam_names.iter()) {
        if let Some(ifn) = fns.get(name) {
            collect_lits_iexpr(&ifn.body, &mut lit_list);
        }
    }
    let mut str_lits: HashMap<Vec<u8>, String> = HashMap::new();
    let mut lit_decls = String::new();
    for (i, bytes) in lit_list.iter().enumerate() {
        let gname = format!("aria_lit_{}", i);
        let _ = write!(lit_decls, "static const char {}[] = {{", gname);
        for (j, b) in bytes.iter().enumerate() {
            if j > 0 {
                lit_decls.push(',');
            }
            let _ = write!(lit_decls, "{}", b);
        }
        // A trailing 0 keeps zero-length arrays legal; len is passed explicitly.
        if !bytes.is_empty() {
            lit_decls.push(',');
        }
        lit_decls.push('0');
        lit_decls.push_str("};\n");
        str_lits.insert(bytes.clone(), gname);
    }

    let mut src = String::new();
    src.push_str(RUNTIME);
    src.push('\n');
    src.push_str(&lit_decls);
    src.push('\n');

    // 4. Forward-declare every user function.
    for name in &order {
        let sig = &sigs[name];
        let params = if sig.params.is_empty() {
            "void".to_string()
        } else {
            sig.params.iter().map(|t| t.decl().to_string()).collect::<Vec<_>>().join(", ")
        };
        let _ = writeln!(src, "static {} {}({});", sig.ret.decl(), cfn(name), params);
    }
    // 4b. Forward-declare every lifted lambda with the closure calling
    //     convention `ret (void* closure, params...)`, then build the
    //     function-pointer dispatch table indexed by `tag - closure_base`.
    for name in &lam_names {
        let ifn = &fns[name];
        let (ret_ct, param_cts, _) = lam_c_types(ifn)?;
        let mut decls = vec!["void*".to_string()];
        decls.extend(param_cts.iter().map(|t| t.decl().to_string()));
        let _ = writeln!(src, "static {} {}({});", ret_ct.decl(), cfn(name), decls.join(", "));
    }
    let _ = write!(src, "static void* __aria_lam_table[] = {{");
    for (i, name) in lam_names.iter().enumerate() {
        if i > 0 {
            src.push_str(", ");
        }
        let _ = write!(src, "(void*){}", cfn(name));
    }
    // A trailing dummy keeps a zero-lambda table a legal non-empty array.
    if lam_names.is_empty() {
        src.push_str("0");
    }
    src.push_str("};\n");
    src.push('\n');

    // 5. Emit the per-tag structural-equality and child-release helpers.
    emit_eq_helper(&ctors, &mut src);
    emit_drop_children_helper(&ctors, &closures, &fns, &mut src)?;

    // 6. Emit each user function body.
    for name in &order {
        let sig = &sigs[name];
        let ifn = fns
            .get(name)
            .ok_or_else(|| format!("c backend: function `{}` missing from IR", name))?;
        if ifn.params.len() != sig.params.len() {
            return Err(format!(
                "c backend: fn `{}` IR arity {} != signature arity {}",
                name,
                ifn.params.len(),
                sig.params.len()
            ));
        }
        let mut types = HashMap::new();
        let param_decls: Vec<String> = ifn
            .params
            .iter()
            .zip(sig.params.iter())
            .map(|(pn, pt)| {
                types.insert(pn.clone(), *pt);
                format!("{} {}", pt.decl(), cvar(pn))
            })
            .collect();
        let params = if param_decls.is_empty() {
            "void".to_string()
        } else {
            param_decls.join(", ")
        };
        let _ = writeln!(src, "static {} {}({}) {{", sig.ret.decl(), cfn(name), params);
        // For a self-tail-recursive function, expose its parameters so a
        // `TailCall` can reassign them, and emit a loop-top label the back-edge
        // `goto`s. The function parameters are themselves the mutable loop
        // induction variables; re-entering re-executes the body's declarations.
        let tail_params: Vec<(String, CType)> = if ifn.tail_recursive {
            ifn.params.iter().zip(sig.params.iter()).map(|(pn, pt)| (pn.clone(), *pt)).collect()
        } else {
            Vec::new()
        };
        let mut env = Env {
            types,
            sigs: &sigs,
            ctors: &ctors,
            tmp: 0,
            str_lits: &str_lits,
            tail_params: &tail_params,
            fn_ret: sig.ret,
            closures: &closures,
        };
        let _ = writeln!(src, "    {} aria_ret;", sig.ret.decl());
        if ifn.tail_recursive {
            let _ = writeln!(src, "    aria_loop_top:;");
        }
        emit_iexpr(&ifn.body, "aria_ret", sig.ret, &mut env, "    ", &mut src)?;
        let _ = writeln!(src, "    return aria_ret;");
        let _ = writeln!(src, "}}");
        src.push('\n');
    }

    // 6b. Emit each lifted lambda body with the closure calling convention.
    for name in &lam_names {
        let ifn = &fns[name];
        emit_lambda(name, ifn, &sigs, &ctors, &str_lits, &closures, &mut src)?;
    }

    // 7. The C `main`: run aria_main, print its result, and report the live
    //    count on stderr (garbage-free <=> aria_live == 0 for value results).
    let ret = sigs["main"].ret;
    let _ = writeln!(src, "int main(void) {{");
    match ret {
        CType::Int => {
            let _ = writeln!(src, "    int64_t r = {}();", cfn("main"));
            let _ = writeln!(src, "    printf(\"%lld\\n\", (long long)r);");
        }
        CType::Bool => {
            let _ = writeln!(src, "    int64_t r = {}();", cfn("main"));
            let _ = writeln!(src, "    fputs(r ? \"true\\n\" : \"false\\n\", stdout);");
        }
        CType::Float => {
            let _ = writeln!(src, "    double r = {}();", cfn("main"));
            let _ = writeln!(src, "    aria_print_float(r);");
        }
        CType::Str => {
            let _ = writeln!(src, "    void* r = {}();", cfn("main"));
            let _ = writeln!(src, "    AriaStr* s = (AriaStr*)r;");
            let _ = writeln!(src, "    fwrite(s->bytes, 1, (size_t)s->len, stdout);");
            let _ = writeln!(src, "    fputc('\\n', stdout);");
            let _ = writeln!(src, "    aria_str_drop(r);");
        }
        CType::Bytes => {
            // Print the canonical `Bytes[..]` rendering (byte-for-byte identical to
            // the interpreter/IR/wasm), then consume the buffer.
            let _ = writeln!(src, "    void* r = {}();", cfn("main"));
            let _ = writeln!(src, "    aria_print_bytes_value(r);");
            let _ = writeln!(src, "    aria_bytes_drop(r);");
        }
        CType::Map(..) => {
            // Print the canonical `Map[k: v, ..]` rendering (sorted by key), then
            // consume the map.
            let _ = writeln!(src, "    void* r = {}();", cfn("main"));
            let _ = writeln!(src, "    aria_print_map_value(r);");
            let _ = writeln!(src, "    aria_map_drop(r);");
        }
        CType::Set(_) => {
            let _ = writeln!(src, "    void* r = {}();", cfn("main"));
            let _ = writeln!(src, "    aria_print_set_value(r);");
            let _ = writeln!(src, "    aria_set_drop(r);");
        }
        CType::Vector => {
            // Print the canonical `Vector[..]` rendering (byte-for-byte identical
            // to the interpreter), then consume the buffer.
            let _ = writeln!(src, "    void* r = {}();", cfn("main"));
            let _ = writeln!(src, "    aria_print_vec_value(r);");
            let _ = writeln!(src, "    aria_vec_drop(r);");
        }
        CType::Ref | CType::Array(_) => {
            // An ADT / Array result is outside the printed-result subset (it has
            // no canonical printed form in this backend). Clean error, no panic.
            return Err(
                "c backend: `main` must return a printable value \
                 (Int/Bool/Float/Str/Bytes/Map/Set)".into(),
            );
        }
    }
    let _ = writeln!(src, "    fprintf(stderr, \"aria_live=%lld aria_reuses=%lld\\n\", (long long)aria_live, (long long)aria_reuses);");
    let _ = writeln!(src, "    return 0;");
    let _ = writeln!(src, "}}");

    Ok(src)
}

/// Emit `aria_eq`: structural ADT equality, per-tag, recursing into Ref fields
/// and comparing Str fields with `aria_streq` (mirrors wasm's `__eq`).
fn emit_eq_helper(ctors: &CtorTable, out: &mut String) {
    out.push_str("static int64_t aria_eq(void* a, void* b) {\n");
    out.push_str("    if (aria_tag(a) != aria_tag(b)) return 0;\n");
    for (_, info) in ctors.sorted() {
        let _ = writeln!(out, "    if (aria_tag(a) == INT64_C({})) {{", info.tag);
        out.push_str("        int64_t eq = 1;\n");
        for (i, fty) in info.field_types.iter().enumerate() {
            match fty {
                CType::Int | CType::Bool => {
                    let _ = writeln!(out, "        eq = eq && (aria_field(a, {}) == aria_field(b, {}));", i, i);
                }
                CType::Float => {
                    let _ = writeln!(out, "        eq = eq && (aria_i2f(aria_field(a, {})) == aria_i2f(aria_field(b, {})));", i, i);
                }
                CType::Str => {
                    let _ = writeln!(out, "        eq = eq && aria_streq((void*)(uintptr_t)aria_field(a, {}), (void*)(uintptr_t)aria_field(b, {}));", i, i);
                }
                CType::Ref => {
                    let _ = writeln!(out, "        eq = eq && aria_eq((void*)(uintptr_t)aria_field(a, {}), (void*)(uintptr_t)aria_field(b, {}));", i, i);
                }
                CType::Array(_) => {
                    // Array fields compare by pointer identity (structural array
                    // equality is outside the subset).
                    let _ = writeln!(out, "        eq = eq && (aria_field(a, {}) == aria_field(b, {}));", i, i);
                }
                CType::Bytes => {
                    // Bytes fields compare structurally (content), like Strings.
                    let _ = writeln!(out, "        eq = eq && aria_byteseq((void*)(uintptr_t)aria_field(a, {}), (void*)(uintptr_t)aria_field(b, {}));", i, i);
                }
                CType::Map(..) => {
                    // Map fields compare structurally (ordered contents).
                    let _ = writeln!(out, "        eq = eq && aria_map_eq((void*)(uintptr_t)aria_field(a, {}), (void*)(uintptr_t)aria_field(b, {}));", i, i);
                }
                CType::Set(_) => {
                    // Set fields compare structurally (ordered contents).
                    let _ = writeln!(out, "        eq = eq && aria_set_eq((void*)(uintptr_t)aria_field(a, {}), (void*)(uintptr_t)aria_field(b, {}));", i, i);
                }
                CType::Vector => {
                    // Vector fields compare structurally (length + elements).
                    let _ = writeln!(out, "        eq = eq && aria_veceq((void*)(uintptr_t)aria_field(a, {}), (void*)(uintptr_t)aria_field(b, {}));", i, i);
                }
            }
        }
        out.push_str("        return eq;\n");
        out.push_str("    }\n");
    }
    out.push_str("    return 0;\n");
    out.push_str("}\n\n");
}

/// Emit `aria_drop_children`: per-tag release of a dead cell's Ref/Str fields,
/// recursing via `aria_drop`/`aria_str_drop`.
/// The C value types of a lifted lambda: (return, parameters, captures).
fn lam_c_types(ifn: &IFn) -> Result<(CType, Vec<CType>, Vec<CType>), String> {
    let sig = ifn
        .lam_sig
        .as_ref()
        .ok_or("c backend: lifted lambda missing its type signature")?;
    let ret = CType::from_ty(&sig.ret_ty)?;
    let params = sig.param_tys.iter().map(CType::from_ty).collect::<Result<Vec<_>, _>>()?;
    let caps = sig.capture_tys.iter().map(CType::from_ty).collect::<Result<Vec<_>, _>>()?;
    Ok((ret, params, caps))
}

/// Emit a lifted lambda as a C function `ret f(void* closure, params...)`. The
/// prologue loads each captured value from the closure cell and `dup`s the
/// reference-counted ones, so the body owns its captures exactly like parameters
/// (the cell retains its own reference until it is itself dropped).
fn emit_lambda(
    name: &str,
    ifn: &IFn,
    sigs: &HashMap<String, FnSig>,
    ctors: &CtorTable,
    str_lits: &HashMap<Vec<u8>, String>,
    closures: &ClosureTable,
    out: &mut String,
) -> Result<(), String> {
    let (ret_ct, param_cts, cap_cts) = lam_c_types(ifn)?;
    if ifn.params.len() != param_cts.len() || ifn.captures.len() != cap_cts.len() {
        return Err(format!("c backend: lambda `{}` arity/signature mismatch", name));
    }
    let mut decls = vec!["void* __aria_clo".to_string()];
    for (pn, pt) in ifn.params.iter().zip(param_cts.iter()) {
        decls.push(format!("{} {}", pt.decl(), cvar(pn)));
    }
    let _ = writeln!(out, "static {} {}({}) {{", ret_ct.decl(), cfn(name), decls.join(", "));
    let mut types = HashMap::new();
    for (i, (cn, ct)) in ifn.captures.iter().zip(cap_cts.iter()).enumerate() {
        types.insert(cn.clone(), *ct);
        let load = match ct {
            CType::Int | CType::Bool => format!("(int64_t)aria_field(__aria_clo, {})", i),
            CType::Float => format!("aria_i2f(aria_field(__aria_clo, {}))", i),
            CType::Ref | CType::Str | CType::Bytes | CType::Array(_) | CType::Map(..) | CType::Set(_) | CType::Vector => format!("(void*)(uintptr_t)aria_field(__aria_clo, {})", i),
        };
        let _ = writeln!(out, "    {} {} = {};", ct.decl(), cvar(cn), load);
        match ct {
            CType::Ref => {
                let _ = writeln!(out, "    aria_dup({});", cvar(cn));
            }
            CType::Str => {
                let _ = writeln!(out, "    aria_str_dup({});", cvar(cn));
            }
            CType::Array(_) => {
                let _ = writeln!(out, "    aria_array_dup({});", cvar(cn));
            }
            CType::Bytes => {
                let _ = writeln!(out, "    aria_bytes_dup({});", cvar(cn));
            }
            // A captured Map/Set is dropped by the closure's drop-children
            // helper, so its load MUST dup too — otherwise the refcount is one
            // short and the container is freed while still live in the enclosing
            // scope (use-after-free). Mirrors the Str/Array/Bytes arms above.
            CType::Map(..) => {
                let _ = writeln!(out, "    aria_map_dup({});", cvar(cn));
            }
            CType::Set(_) => {
                let _ = writeln!(out, "    aria_set_dup({});", cvar(cn));
            }
            CType::Vector => {
                let _ = writeln!(out, "    aria_vec_dup({});", cvar(cn));
            }
            _ => {}
        }
    }
    for (pn, pt) in ifn.params.iter().zip(param_cts.iter()) {
        types.insert(pn.clone(), *pt);
    }
    let tail_params: Vec<(String, CType)> = Vec::new();
    let mut env = Env {
        types,
        sigs,
        ctors,
        tmp: 0,
        str_lits,
        tail_params: &tail_params,
        fn_ret: ret_ct,
        closures,
    };
    let _ = writeln!(out, "    {} aria_ret;", ret_ct.decl());
    emit_iexpr(&ifn.body, "aria_ret", ret_ct, &mut env, "    ", out)?;
    let _ = writeln!(out, "    return aria_ret;");
    let _ = writeln!(out, "}}");
    out.push('\n');
    Ok(())
}

fn emit_drop_children_helper(
    ctors: &CtorTable,
    closures: &ClosureTable,
    fns: &HashMap<String, IFn>,
    out: &mut String,
) -> Result<(), String> {
    out.push_str("static void aria_drop_children(void* p) {\n");
    out.push_str("    int64_t tag = aria_tag(p);\n");
    for (_, info) in ctors.sorted() {
        let managed: Vec<(usize, CType)> = info
            .field_types
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, CType::Ref | CType::Str | CType::Bytes | CType::Array(_) | CType::Map(..) | CType::Set(_) | CType::Vector))
            .map(|(i, t)| (i, *t))
            .collect();
        if managed.is_empty() {
            continue;
        }
        let _ = writeln!(out, "    if (tag == INT64_C({})) {{", info.tag);
        for (i, t) in managed {
            emit_drop_managed_field(t, i, out);
        }
        out.push_str("        return;\n");
        out.push_str("    }\n");
    }
    // Closure cells: release each reference-counted captured value.
    for name in &closures.names {
        let ifn = fns
            .get(name)
            .ok_or_else(|| format!("c backend: lambda `{}` missing from IR", name))?;
        let (_, _, cap_cts) = lam_c_types(ifn)?;
        let managed: Vec<(usize, CType)> = cap_cts
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, CType::Ref | CType::Str | CType::Bytes | CType::Array(_) | CType::Map(..) | CType::Set(_) | CType::Vector))
            .map(|(i, t)| (i, *t))
            .collect();
        if managed.is_empty() {
            continue;
        }
        let tag = closures.tags[name];
        let _ = writeln!(out, "    if (tag == INT64_C({})) {{", tag);
        for (i, t) in managed {
            emit_drop_managed_field(t, i, out);
        }
        out.push_str("        return;\n");
        out.push_str("    }\n");
    }
    out.push_str("    (void)tag;\n");
    out.push_str("}\n\n");
    Ok(())
}

/// Emit the release of a managed (heap-ref) field `i` from a dead cell, per type.
fn emit_drop_managed_field(t: CType, i: usize, out: &mut String) {
    match t {
        CType::Ref => {
            let _ = writeln!(out, "        aria_drop((void*)(uintptr_t)aria_field(p, {}));", i);
        }
        CType::Str => {
            let _ = writeln!(out, "        aria_str_drop((void*)(uintptr_t)aria_field(p, {}));", i);
        }
        CType::Array(_) => {
            let _ = writeln!(out, "        aria_array_drop((void*)(uintptr_t)aria_field(p, {}));", i);
        }
        CType::Bytes => {
            let _ = writeln!(out, "        aria_bytes_drop((void*)(uintptr_t)aria_field(p, {}));", i);
        }
        CType::Map(..) => {
            let _ = writeln!(out, "        aria_map_drop((void*)(uintptr_t)aria_field(p, {}));", i);
        }
        CType::Set(_) => {
            let _ = writeln!(out, "        aria_set_drop((void*)(uintptr_t)aria_field(p, {}));", i);
        }
        CType::Vector => {
            let _ = writeln!(out, "        aria_vec_drop((void*)(uintptr_t)aria_field(p, {}));", i);
        }
        _ => {}
    }
}

// ---- differential tests --------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interp, lexer, parser, typeck};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Is a system C compiler (`cc`) available? Tests gate on this and skip
    /// gracefully when it is not (e.g. a minimal CI image).
    fn cc_available() -> bool {
        std::process::Command::new("cc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn typed_program(src: &str) -> Result<Program, String> {
        let toks = lexer::lex(src)?;
        let prog = parser::parse(toks)?;
        typeck::check(&prog).map_err(|e| e.join("; "))?;
        Ok(prog)
    }


    /// The interpreter's canonical result string (the reference oracle). Run on
    /// a large-stack thread so deep (but finite) recursion in the battery does
    /// not overflow the small default test-thread stack (the native program has
    /// its own ample stack).
    fn interp_result(src: &str) -> Result<String, String> {
        let src = src.to_string();
        std::thread::Builder::new()
            .stack_size(1 << 30)
            .spawn(move || {
                let prog = typed_program(&src)?;
                let it = interp::Interp::new(&prog)?;
                it.run_main().map(|v| v.display())
            })
            .expect("spawn interp thread")
            .join()
            .unwrap_or_else(|_| Err("interpreter thread panicked".into()))
    }

    fn compile_src(src: &str) -> Result<String, String> {
        compile(&typed_program(src)?)
    }

    /// Build the C source with `cc`, run it, and return `(stdout, stderr)`.
    fn build_and_run(c_src: &str) -> Result<(String, String), String> {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let c_path = dir.join(format!("aria_ct_{}_{}.c", std::process::id(), n));
        let exe = dir.join(format!("aria_ce_{}_{}", std::process::id(), n));
        std::fs::write(&c_path, c_src).map_err(|e| e.to_string())?;
        let cc = std::process::Command::new("cc")
            .arg("-O2").arg("-std=c11").arg("-o").arg(&exe).arg(&c_path)
            .output().map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&c_path);
        if !cc.status.success() {
            let _ = std::fs::remove_file(&exe);
            return Err(format!("cc failed: {}", String::from_utf8_lossy(&cc.stderr)));
        }
        let run = std::process::Command::new(&exe).output().map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&exe);
        Ok((
            String::from_utf8_lossy(&run.stdout).into_owned(),
            String::from_utf8_lossy(&run.stderr).into_owned(),
        ))
    }

    /// Differential + garbage-free check: the native program's printed result
    /// must equal the interpreter's, and `aria_live` must be 0 at exit.
    fn differential(src: &str) {
        let want = interp_result(src).expect("interpreter should succeed on battery");
        let c_src = compile_src(src).expect("c compile should succeed on battery");
        if !cc_available() {
            return; // gate gracefully when `cc` is missing
        }
        let (stdout, stderr) = build_and_run(&c_src).expect("build+run native");
        // `main`'s result is the first stdout line.
        let got = stdout.lines().next().unwrap_or("").to_string();
        assert_eq!(want, got, "native != interpreter for:\n{}", src);
        assert!(
            stderr.contains("aria_live=0"),
            "expected garbage-free (aria_live=0), got stderr `{}` for:\n{}",
            stderr.trim(),
            src
        );
    }

    #[test]
    fn factorial() {
        differential(
            "fn fac(n: Int) -> Int = match n { 0 => 1, _ => n * fac(n - 1), }\n\
             fn main() -> Int = fac(10)",
        );
    }

    #[test]
    fn closure_immediate_and_curry() {
        // An immediately-applied lambda, and currying via a returned closure.
        differential("fn main() -> Int = (\\x -> x * 2)(21)");
        differential(
            "fn add(x: Int) -> (Int) -> Int = \\y -> x + y\n\
             fn main() -> Int = add(3)(4)",
        );
    }

    #[test]
    fn closure_captures_and_higher_order() {
        // A closure passed to a generic higher-order function, capturing a local,
        // plus a function used by name — all reference-counted garbage-free.
        differential(
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
    }

    #[test]
    fn closure_unannotated_let_bound() {
        // An unannotated lambda bound to a bare `let` (its parameter type fixed
        // only by a later use) — typeck back-annotates the inferred type so the
        // backend can compile it.
        differential(
            "fn compose(f: (Int)->Int, g: (Int)->Int) -> (Int)->Int = \\x -> f(g(x))\n\
             fn main() -> Int = {\n\
               let inc = \\x -> x + 1;\n\
               let dbl = \\x -> x * 2;\n\
               compose(inc, dbl)(10)\n\
             }",
        );
    }

    #[test]
    fn closure_thunk_escape_and_reuse() {
        // A zero-parameter thunk capturing a value, applied twice.
        differential(
            "fn force(t: () -> Int) -> Int = t()\n\
             fn main() -> Int = { let k = 77; let t = \\() -> k + 1; force(t) + force(t) }",
        );
        // Closures escaping from both branches of an `if`, then applied.
        differential(
            "fn choose(c: Bool, n: Int) -> (Int) -> Int = if c { \\x -> x + n } else { \\x -> x - n }\n\
             fn main() -> Int = choose(true, 10)(100) + choose(false, 3)(100)",
        );
        // FBIP in-place reuse of a cell that carries a closure field.
        differential(
            "type Box = | B(Int, (Int) -> Int)\n\
             fn bump(bx: Box) -> Box = match bx { B(n, f) => B(n + 1, f), }\n\
             fn run(bx: Box) -> Int = match bx { B(n, f) => f(n), }\n\
             fn main() -> Int = run(bump(bump(B(10, \\x -> x * 2))))",
        );
    }

    #[test]
    fn local_binding_shadows_global_function() {
        // A lambda parameter, a `let`, and a match binder may each shadow a
        // top-level function of the same name — the backend must resolve the
        // local (scope-before-globals), not emit a function value / by-name call.
        differential(
            "fn helper(n: Int) -> Int = n * 1000\n\
             fn ap(f: (Int) -> Int, x: Int) -> Int = f(x)\n\
             fn main() -> Int = ap(\\helper -> helper + 1, 41)",
        );
        differential(
            "fn helper(n: Int) -> Int = n * 1000\n\
             fn main() -> Int = { let helper = 5; helper + 1 }",
        );
        // A block-scoped `let` shadowing a function must not leak to a sibling
        // branch that calls the real function.
        differential(
            "fn helper(n: Int) -> Int = n * 1000\n\
             fn pick(c: Bool) -> Int = if c { let helper = 1; helper + 1 } else { helper(2) }\n\
             fn main() -> Int = pick(true) + pick(false)",
        );
    }

    #[test]
    fn closure_bidirectional_context_typed() {
        // A curried lambda with no internal type hint, fully typed by the callee
        // signature (bidirectional checking pushes the expected type inward).
        differential(
            "fn apply2(f: (Int) -> (Int) -> Int, a: Int, b: Int) -> Int = f(a)(b)\n\
             fn main() -> Int = apply2(\\x -> \\y -> x + y, 30, 12)",
        );
    }

    #[test]
    fn closure_applied_twice_and_composed() {
        // A closure stored, then applied twice (rc dup), and a closure that
        // captures two other closures (Ref captures released on the cell's drop).
        differential(
            "fn twice(f: (Int) -> Int, x: Int) -> Int = f(f(x))\n\
             fn main() -> Int = twice(\\n -> n + 5, 100)",
        );
        differential(
            "fn compose(f: (Int)->Int, g: (Int)->Int) -> (Int)->Int = \\x -> f(g(x))\n\
             fn main() -> Int = {\n\
               let h = compose(\\(a: Int) -> a + 1, \\(b: Int) -> b * 2);\n\
               h(10) + h(20)\n\
             }",
        );
    }

    #[test]
    fn list_sum_and_length() {
        differential(
            "type List = | Nil | Cons(Int, List)\n\
             fn sum(l: List) -> Int = match l { Nil => 0, Cons(h, t) => h + sum(t), }\n\
             fn len(l: List) -> Int = match l { Nil => 0, Cons(h, t) => 1 + len(t), }\n\
             fn build(n: Int) -> List = match n { 0 => Nil, _ => Cons(n, build(n - 1)), }\n\
             fn main() -> Int = sum(build(100)) + len(build(10))",
        );
    }

    #[test]
    fn binary_tree() {
        differential(
            "type Tree = | Leaf(Int) | Node(Tree, Tree)\n\
             fn total(t: Tree) -> Int = match t { Leaf(n) => n, Node(l, r) => total(l) + total(r), }\n\
             fn mk(d: Int) -> Tree = match d { 0 => Leaf(1), _ => Node(mk(d - 1), mk(d - 1)), }\n\
             fn main() -> Int = total(mk(8))",
        );
    }

    #[test]
    fn generic_option_list() {
        differential(
            "type Option[T] = | None | Some(T)\n\
             type Lst[T] = | Nil | Cons(T, Lst[T])\n\
             fn unwrap(o: Option[Int], d: Int) -> Int = match o { None => d, Some(x) => x, }\n\
             fn first(l: Lst[Int]) -> Option[Int] = match l { Nil => None, Cons(h, t) => Some(h), }\n\
             fn main() -> Int = unwrap(first(Cons(7, Cons(8, Nil))), 0) + unwrap(first(Nil), 99)",
        );
    }

    #[test]
    fn string_concat_and_int_to_str() {
        differential(
            "fn main() -> String = \
             concat(concat(\"count=\", int_to_str(42)), concat(\" neg=\", int_to_str(-7)))",
        );
    }

    #[test]
    fn float_through_int() {
        // Float arithmetic whose RESULT is an Int/Bool, dodging the documented
        // Rust-vs-C float formatting difference.
        differential(
            "fn area(r: Float) -> Float = r * r * 3.0\n\
             fn main() -> Int = if area(2.0) > 11.0 { 1 } else { 0 }",
        );
    }

    #[test]
    fn reuse_heavy_map() {
        // A list `map` (inc) that the rc pass turns into in-place CtorReuse:
        // garbage-free AND the result must agree with the interpreter.
        differential(
            "type List = | Nil | Cons(Int, List)\n\
             fn inc(l: List) -> List = match l { Nil => Nil, Cons(h, t) => Cons(h + 1, inc(t)), }\n\
             fn sum(l: List) -> Int = match l { Nil => 0, Cons(h, t) => h + sum(t), }\n\
             fn build(n: Int) -> List = match n { 0 => Nil, _ => Cons(n, build(n - 1)), }\n\
             fn main() -> Int = sum(inc(inc(build(50))))",
        );
    }

    #[test]
    fn adt_structural_equality() {
        differential(
            "type P = | A(Int, Int) | B(Int)\n\
             fn main() -> Int = if A(1, 2) == A(1, 2) { 1 } else { 0 }",
        );
    }

    #[test]
    fn reuse_actually_fires() {
        // Sanity: the reuse-heavy program must report a non-zero reuse count,
        // proving the FBIP in-place reuse path is exercised by the C backend.
        if !cc_available() {
            return;
        }
        let src = "type List = | Nil | Cons(Int, List)\n\
             fn inc(l: List) -> List = match l { Nil => Nil, Cons(h, t) => Cons(h + 1, inc(t)), }\n\
             fn sum(l: List) -> Int = match l { Nil => 0, Cons(h, t) => h + sum(t), }\n\
             fn build(n: Int) -> List = match n { 0 => Nil, _ => Cons(n, build(n - 1)), }\n\
             fn main() -> Int = sum(inc(build(50)))";
        let c_src = compile_src(src).expect("compile");
        let (_, stderr) = build_and_run(&c_src).expect("build+run");
        assert!(
            !stderr.contains("aria_reuses=0"),
            "expected in-place reuse to fire, stderr: {}",
            stderr.trim()
        );
    }

    // ---- native arrays -------------------------------------------------

    #[test]
    fn array_set_compiles_and_runs() {
        // The headline example: build, set index 1 to 99, then read back. Returns
        // 10 + 99 + 30 = 139 (the task brief's "129" is an arithmetic slip; we
        // assert the value the interpreter oracle also produces). `differential`
        // also checks it is garbage-free.
        differential(
            "fn main() -> Int = { let a = array_set([10,20,30], 1, 99); a[0] + a[1] + a[2] }",
        );
        let src = "fn main() -> Int = { let a = array_set([10,20,30], 1, 99); a[0] + a[1] + a[2] }";
        let c_src = compile_src(src).expect("array program should compile");
        if !cc_available() {
            return;
        }
        let (stdout, _) = build_and_run(&c_src).expect("build+run native array");
        assert_eq!(stdout.lines().next().unwrap_or(""), "139");
    }

    #[test]
    fn native_float_printing_matches_interpreter() {
        // Native floats must print the shortest round-tripping decimal, identical
        // to the interpreter's Rust `{}` formatting (no lossy `%g`).
        if !cc_available() {
            return;
        }
        let src = "fn main() -> Int = { print_float(3.14); print_float(12.566360); \
                   print_float(0.670820415019989); print_float(3.141592653589793); \
                   print_float(100.0); print_float(0.00001); 0 }";
        let expected = [
            format!("{}", 3.14_f64),
            format!("{}", 12.566360_f64),
            format!("{}", 0.670820415019989_f64),
            format!("{}", 3.141592653589793_f64),
            format!("{}", 100.0_f64),
            format!("{}", 0.00001_f64),
        ];
        let c_src = compile_src(src).expect("float program should compile");
        let (stdout, _) = build_and_run(&c_src).expect("build+run native float");
        let lines: Vec<&str> = stdout.lines().collect();
        for (i, e) in expected.iter().enumerate() {
            assert_eq!(lines.get(i).copied().unwrap_or(""), e, "float line {}", i);
        }
    }

    #[test]
    fn tuples_compile_and_are_garbage_free() {
        // Tuples desugar to synthetic generic ADTs: construction, tuple-typed fn
        // params/returns, and destructuring patterns must compile, match the
        // interpreter, and be garbage-free in native.
        differential(
            "fn swap(p: (Int, Bool)) -> (Bool, Int) = match p { (a, b) => (b, a), }\n\
             fn third(t: (Int, Int, Int)) -> Int = match t { (a, b, c) => c, }\n\
             fn main() -> Int = {\n\
               let s = swap((7, true));\n\
               third((10, 20, 30)) + match s { (b, n) => n, }\n\
             }",
        );
    }

    #[test]
    fn records_compile_and_are_garbage_free() {
        // Records desugar (in monomorphize) to positional ADT cells: literal,
        // field access, functional update, record patterns, and a GENERIC record
        // must all compile, match the interpreter, and be garbage-free in native.
        differential(
            "type Box[T] = { value: T, tag: Int }\n\
             type P = { x: Int, y: Int }\n\
             fn unwrap[T](b: Box[T]) -> T = b.value\n\
             fn describe(p: P) -> Int = match p { P { x, y } => x * 100 + y, }\n\
             fn main() -> Int = {\n\
               let b = Box { value: 7, tag: 1 };\n\
               let p = P { x: 2, y: 3 };\n\
               let q = { p | x = 9 };\n\
               unwrap(b) + describe(p) + describe(q) + b.tag\n\
             }",
        );
    }

    #[test]
    fn array_int_get_set_push_sum() {
        // get/set/push and a recursive sum over an Int array, garbage-free.
        differential(
            "fn sum(a: Array[Int], i: Int, acc: Int) -> Int =\n\
               if i == array_len(a) { acc } else { sum(a, i + 1, acc + a[i]) }\n\
             fn main() -> Int = {\n\
               let a = array_push(array_push(array_set([1,2,3], 0, 10), 4), 5);\n\
               sum(a, 0, 0)\n\
             }",
        );
    }

    #[test]
    fn array_new_inline_push() {
        // `array_push(array_new(), x)` where a callee parameter fixes the element
        // type: the monomorphizer threads `E` into the nested `array_new()`, so it
        // resolves to `$i`. (When the empty `array_new()` is instead tagged `$r`,
        // e.g. via a bare annotated `let`, the runtime reconciles the header kind
        // from the push call site — see `empty_array_grow_from_array_new`.)
        differential(
            "fn first(a: Array[Int]) -> Int = a[0]\n\
             fn main() -> Int = first(array_push(array_new(), 42))",
        );
    }

    #[test]
    fn empty_array_grow_from_array_new() {
        // Regression: `let a: Array[Int] = array_new()` then TWO pushes used to
        // segfault. The monomorphizer can tag the empty `array_new` with a stale
        // element kind (`$r`); the runtime now reconciles the header `kind` from
        // the authoritative push call site (and grows cap 0 -> 4 by reallocating
        // the whole header+elems object), so dup/drop on the grown array stay
        // correct. Must agree with the interpreter AND be garbage-free.
        differential(
            "fn main() -> Int = {\n\
               let a: Array[Int] = array_new();\n\
               let b = array_push(array_push(a, 7), 8);\n\
               array_len(b) + b[1]\n\
             }",
        );
        // Build from empty to length 10 by repeated push, then sum (= 55).
        differential(
            "fn build(a: Array[Int], n: Int) -> Array[Int] =\n\
               if n == 0 { a } else { build(array_push(a, n), n - 1) }\n\
             fn sum(a: Array[Int], i: Int, acc: Int) -> Int =\n\
               if i == array_len(a) { acc } else { sum(a, i + 1, acc + a[i]) }\n\
             fn main() -> Int = {\n\
               let a: Array[Int] = array_new();\n\
               sum(build(a, 10), 0, 0)\n\
             }",
        );
        // An empty Array[String] grown by pushes from array_new — every String
        // element must be released on drop (aria_live=0).
        differential(
            "fn main() -> String = {\n\
               let a: Array[String] = array_new();\n\
               let b = array_push(array_push(array_push(a, \"x\"), \"y\"), \"z\");\n\
               concat(concat(b[0], b[1]), b[2])\n\
             }",
        );
        // array_len of a freshly-annotated empty array is 0.
        differential(
            "fn main() -> Int = { let a: Array[Int] = array_new(); array_len(a) }",
        );
    }

    #[test]
    fn array_string_garbage_free() {
        // An Array[String]: build, read, concat — drop must release every String
        // element (aria_live=0).
        differential(
            "fn main() -> String = {\n\
               let a = array_push([\"hello\", \"world\"], \"!\");\n\
               concat(concat(a[0], a[1]), a[2])\n\
             }",
        );
    }

    #[test]
    fn array_of_bool_is_usable() {
        // Regression for BUG 3: `Array[Bool]` collapsed Bool into the Int element
        // kind, so `array_get` returned Int and `print_bool`/`if` rejected it.
        // Bool is now a first-class array element kind (`o`, code 8, inline slot).
        // build/get/use-in-if (and print_bool).
        differential(
            "fn main() -> Int = {\n\
               let xs = array_push(array_push(array_new(), true), false);\n\
               (if xs[0] { 1 } else { 0 }) + (if xs[1] { 10 } else { 0 })\n\
             }",
        );
        // An array literal of Bools.
        differential(
            "fn main() -> Int = {\n\
               let xs = [true, false, true];\n\
               (if xs[0] { 1 } else { 0 }) + (if xs[2] { 100 } else { 0 })\n\
             }",
        );
    }

    #[test]
    fn array_of_adt_garbage_free() {
        // An Array of an ADT (`$r` elements): drop must recursively release each
        // boxed cell, leaving no garbage.
        differential(
            "type Color = | Red | Green | Blue | Shade(Int)\n\
             fn rank(c: Color) -> Int = match c { Red => 1, Green => 2, Blue => 3, Shade(n) => n, }\n\
             fn main() -> Int = {\n\
               let a = array_push([Red, Green, Shade(40)], Blue);\n\
               rank(a[0]) + rank(a[1]) + rank(a[2]) + rank(a[3])\n\
             }",
        );
    }

    #[test]
    fn array_build_drop_is_garbage_free() {
        // Building an array and then dropping it (the result is an unrelated Int)
        // must leave aria_live=0.
        if !cc_available() {
            return;
        }
        let src = "fn use_it(a: Array[Int]) -> Int = 99\n\
                   fn main() -> Int = { let a = array_push([1,2,3], 4); use_it(a) }";
        let c_src = compile_src(src).expect("compile");
        let (stdout, stderr) = build_and_run(&c_src).expect("build+run");
        assert_eq!(stdout.lines().next().unwrap_or(""), "99");
        assert!(stderr.contains("aria_live=0"), "expected garbage-free, got `{}`", stderr.trim());
    }

    #[test]
    fn unsupported_tensor_returns_err_not_panic() {
        // A tensor builtin is outside the IR subset; compilation must return a
        // clean Err (never panic).
        let src = "fn t() -> Tensor = tensor_zeros(2, 2)\n\
                   fn main() -> Int = 0";
        let r = compile_src(src);
        assert!(r.is_err(), "expected clean Err for a tensor program, got Ok");
    }

    // ---- self-tail-call optimization (native) --------------------------

    #[test]
    fn deep_tail_accumulator_native() {
        // 1,000,000-deep tail-recursive accumulator. Self-tail-call elimination
        // turns it into a C `while`-loop (param reassignment + `goto`), so the
        // native program runs in constant C stack and agrees with the
        // interpreter (= 500000500000), garbage-free.
        differential(
            "fn go(n: Int, acc: Int) -> Int = if n == 0 { acc } else { go(n - 1, acc + n) }\n\
             fn main() -> Int = go(1000000, 0)",
        );
    }

    #[test]
    fn deep_tail_call_in_match_native() {
        // Self-tail-call in a `match` arm body (tail position flows through every
        // arm), 1,000,000 deep, with a small flat-ADT scrutinee (so the
        // interpreter oracle's per-iteration clone is O(1)). Result agrees with
        // the interpreter (= 500000500000); the native side allocates/frees a
        // `Step` cell each iteration and must end garbage-free.
        differential(
            "type Step = | Done | More(Int)\n\
             fn step(n: Int) -> Step = if n == 0 { Done } else { More(n) }\n\
             fn go(n: Int, acc: Int) -> Int = \
                match step(n) { Done => acc, More(k) => go(k - 1, acc + k), }\n\
             fn main() -> Int = go(1000000, 0)",
        );
    }

    #[test]
    fn heap_list_tail_recursion_is_garbage_free_native() {
        // Build then fold a cons-list; both functions are self-tail-recursive and
        // pass a HEAP parameter (the list) through the tail call. The native
        // program reassigns that heap param under TCO and must free every cell
        // (aria_live=0). Depth kept modest because the interpreter oracle
        // deep-clones the list each step (O(n^2)); the 1M heap case is covered by
        // the flat-ADT `deep_tail_call_in_match_native` test above.
        differential(
            "type L = | Nil | Cons(Int, L)\n\
             fn build(n: Int, acc: L) -> L = if n == 0 { acc } else { build(n - 1, Cons(n, acc)) }\n\
             fn length(xs: L, acc: Int) -> Int = \
                match xs { Nil => acc, Cons(_, r) => length(r, acc + 1), }\n\
             fn main() -> Int = length(build(300, Nil), 0)",
        );
    }

    #[test]
    fn tail_call_swapping_args_native() {
        // gcd by subtraction: a tail call whose new args read the OTHER old param.
        // The loop reassigns via temporaries, so args see the OLD values.
        differential(
            "fn gcd(a: Int, b: Int) -> Int = \
                if b == 0 { a } else { if a < b { gcd(b, a) } else { gcd(a - b, b) } }\n\
             fn main() -> Int = gcd(1071, 462)",
        );
    }
}
