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
//! Float formatting note: Rust's `{}` and C's `printf` format floats
//! differently, so the differential tests route Float results through Int/Bool
//! (the same choice the wasm backend documented). `print_float` uses `%g`.
//!
//! Out-of-subset features (tensors/RAG/compression builtins, Unit results) yield
//! a clean `Err` from `compile` — never a panic.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::ast::{BinOp, Item, Program, Ty, UnOp};
use crate::ir::{self, Atom, Bind, IExpr, IFn};

/// A C-level value type for an Aria value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CType {
    Int,   // int64_t
    Bool,  // int64_t (0/1)
    Float, // double
    Ref,   // void* — heap ADT cell
    Str,   // void* — heap String object
}

impl CType {
    /// The C type name used to declare a local / parameter of this kind.
    fn decl(self) -> &'static str {
        match self {
            CType::Int | CType::Bool => "int64_t",
            CType::Float => "double",
            CType::Ref | CType::Str => "void*",
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
            Ty::Named(_, args) if args.is_empty() => Ok(CType::Ref),
            // A closure value is a heap cell (tag = lambda id, fields = captures).
            Ty::Fn(_, _) => Ok(CType::Ref),
            other => Err(format!(
                "c backend: unsupported type `{:?}` (subset: Int/Bool/Float/String and non-generic ADTs)",
                other
            )),
        }
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
        Bind::Call(name, _) => {
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
        "concat" | "int_to_str" => Some(CType::Str),
        // print_* are logically Unit; we model them as Int 0 (never used).
        "print_int" | "print_bool" | "print_float" | "print_str" => Some(CType::Int),
        _ => None,
    }
}

fn is_builtin(name: &str) -> bool {
    builtin_ret(name).is_some()
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
                let _ = writeln!(out, "{}aria_trap();", inner);
            } else {
                emit_iexpr(then, dst, dst_ty, env, &inner, out)?;
            }
            let _ = writeln!(out, "{}}} else {{", ind);
            if is_unreachable_unit(els) {
                let _ = writeln!(out, "{}aria_trap();", inner);
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
            CType::Ref | CType::Str => {
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
            let _ = writeln!(out, "{}if ({}({}, {}, &{})) aria_trap();", ind, bi, lx, rx, dst);
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
                "{}if ({} == 0 || ({} == INT64_MIN && {} == -1)) aria_trap();",
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
                let _ = writeln!(out, "{}if (__builtin_sub_overflow((int64_t)0, {}, &{})) aria_trap();", ind, ax, dst);
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
        _ => Err(format!("c backend: unsupported builtin `{}`", name)),
    }
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
            CType::Ref | CType::Str => {
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
        let _ = writeln!(out, "{}aria_trap();", ind);
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
                    CType::Ref | CType::Str => {
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

/* ---- live-cell accounting (the native analogue of wasm __live) ---- */
static int64_t aria_live = 0;
static int64_t aria_reuses = 0;

/* ---- ADT cell: { int64_t rc; int64_t tag; int64_t fields[]; } ---- */
typedef struct { int64_t rc; int64_t tag; int64_t fields[]; } AriaCell;

static void aria_trap(void) {
    /* Print TRAP (stdout) so a runner can detect it, then abort like a wasm
       trap / the interpreter's Err. */
    fputs("TRAP\n", stdout);
    fflush(stdout);
    exit(70);
}

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

/* ---- structural equality (per-tag, emitted below) ---- */
static int64_t aria_eq(void* a, void* b);

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
static void aria_print_float(double d) { printf("%g\n", d); }
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
        // main may return Int, Float, Bool, or String; the runner prints it.
        if !matches!(m.ret, CType::Int | CType::Float | CType::Bool | CType::Str) {
            return Err("c backend: `main` must return Int, Bool, Float, or String".into());
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
            let _ = writeln!(src, "    printf(\"%g\\n\", r);");
        }
        CType::Str => {
            let _ = writeln!(src, "    void* r = {}();", cfn("main"));
            let _ = writeln!(src, "    AriaStr* s = (AriaStr*)r;");
            let _ = writeln!(src, "    fwrite(s->bytes, 1, (size_t)s->len, stdout);");
            let _ = writeln!(src, "    fputc('\\n', stdout);");
            let _ = writeln!(src, "    aria_str_drop(r);");
        }
        _ => unreachable!(),
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
            CType::Ref | CType::Str => format!("(void*)(uintptr_t)aria_field(__aria_clo, {})", i),
        };
        let _ = writeln!(out, "    {} {} = {};", ct.decl(), cvar(cn), load);
        match ct {
            CType::Ref => {
                let _ = writeln!(out, "    aria_dup({});", cvar(cn));
            }
            CType::Str => {
                let _ = writeln!(out, "    aria_str_dup({});", cvar(cn));
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
            .filter(|(_, t)| matches!(t, CType::Ref | CType::Str))
            .map(|(i, t)| (i, *t))
            .collect();
        if managed.is_empty() {
            continue;
        }
        let _ = writeln!(out, "    if (tag == INT64_C({})) {{", info.tag);
        for (i, t) in managed {
            match t {
                CType::Ref => {
                    let _ = writeln!(out, "        aria_drop((void*)(uintptr_t)aria_field(p, {}));", i);
                }
                CType::Str => {
                    let _ = writeln!(out, "        aria_str_drop((void*)(uintptr_t)aria_field(p, {}));", i);
                }
                _ => {}
            }
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
            .filter(|(_, t)| matches!(t, CType::Ref | CType::Str))
            .map(|(i, t)| (i, *t))
            .collect();
        if managed.is_empty() {
            continue;
        }
        let tag = closures.tags[name];
        let _ = writeln!(out, "    if (tag == INT64_C({})) {{", tag);
        for (i, t) in managed {
            match t {
                CType::Ref => {
                    let _ = writeln!(out, "        aria_drop((void*)(uintptr_t)aria_field(p, {}));", i);
                }
                CType::Str => {
                    let _ = writeln!(out, "        aria_str_drop((void*)(uintptr_t)aria_field(p, {}));", i);
                }
                _ => {}
            }
        }
        out.push_str("        return;\n");
        out.push_str("    }\n");
    }
    out.push_str("    (void)tag;\n");
    out.push_str("}\n\n");
    Ok(())
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
