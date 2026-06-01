//! Aria's first *compiled* backend: a hand-rolled WebAssembly emitter for the
//! pure numeric/control subset of the ANF IR (`src/ir.rs`).
//!
//! Scope (Phase 2a) — pure, heap-free:
//!   * Types: `Int -> i64`, `Bool -> i32`. Float / String / ADT / Unit are OUT.
//!   * `IExpr`: `Let`, `Ret`.  `Bind`: `Atom`, `Prim`, `Unary`, `Call`, `If`.
//!   * Everything else (`Ctor`/`CtorReuse`/`Match`, `Dup`/`Drop`/`DropReuse`,
//!     Float/Str/Unit atoms, non-Int/Bool signatures, builtin calls) is rejected
//!     with a clean `Err(String)` — this emitter NEVER panics on valid IR.
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

/// A wasm numeric value type, restricted to the subset we support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WType {
    I64, // Aria Int
    I32, // Aria Bool
}

impl WType {
    /// The valtype byte used in the binary (`0x7E` = i64, `0x7F` = i32).
    fn byte(self) -> u8 {
        match self {
            WType::I64 => 0x7E,
            WType::I32 => 0x7F,
        }
    }

    fn from_ty(ty: &Ty) -> Result<WType, String> {
        match ty {
            Ty::Int => Ok(WType::I64),
            Ty::Bool => Ok(WType::I32),
            other => Err(format!(
                "wasm backend: unsupported type `{:?}` (only Int/Bool are in the 2a subset)",
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
}

impl<'a> LocalEnv<'a> {
    /// The function index of an overflow helper.
    fn ovf_index(&self, h: OvfHelper) -> u32 {
        self.ovf_base + h.offset()
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
        Bind::Ctor(..) | Bind::CtorReuse(..) => {
            Err("wasm backend: ADT constructors are outside the 2a subset".into())
        }
        Bind::Match(..) => Err("wasm backend: `match` on ADTs is outside the 2a subset".into()),
    }
}

/// True when an IExpr is a bare `Ret(Unit)` — the IR's marker for the dead
/// fall-through branch of a lowered integer `match`. Such a branch produces no
/// real value (it is statically unreachable), so the backend compiles it to the
/// wasm `unreachable` instruction, which validates under any block type.
fn is_unreachable_unit(e: &IExpr) -> bool {
    matches!(e, IExpr::Ret(Atom::Unit))
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
            };
            iexpr_type(body, &probe)
        }
        IExpr::Dup(_, _) | IExpr::Drop(_, _) | IExpr::DropReuse(_, _, _) => {
            Err("wasm backend: reference-counting ops are outside the 2a subset".into())
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
                    WType::I32 => {
                        Err("wasm backend: numeric negation of a Bool is invalid".into())
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
        Bind::Ctor(..) | Bind::CtorReuse(..) => {
            Err("wasm backend: ADT constructors are outside the 2a subset".into())
        }
        Bind::Match(..) => Err("wasm backend: `match` on ADTs is outside the 2a subset".into()),
    }
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
        IExpr::Dup(_, _) | IExpr::Drop(_, _) | IExpr::DropReuse(_, _, _) => {
            Err("wasm backend: reference-counting ops are outside the 2a subset".into())
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

/// Key under which a function's index is stored in `LocalEnv::index`. Prefixed
/// to avoid colliding with a same-named local variable.
fn fn_index_key(name: &str) -> String {
    format!("\u{1}fn:{}", name)
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

    // 2. Lower the whole program to ANF IR (reusing the shared lowering pass).
    let fns: HashMap<String, IFn> = ir::lower_program(program)?;

    // The checked-arithmetic helpers are appended after every user function, so
    // the first helper (`Add`) gets index == number of user functions.
    let ovf_base = order.len() as u32;

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

    // Export section (id 7): export `main`.
    {
        let main_idx = fn_index["main"];
        let mut content = Vec::new();
        leb_u(1, &mut content); // one export
        let name = b"main";
        leb_u(name.len() as u64, &mut content);
        content.extend_from_slice(name);
        content.push(0x00); // export kind = func
        leb_u(main_idx as u64, &mut content);
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
        // ADT constructor / Match.
        let s2 = "type L = | Nil | Cons(Int, L)\n\
                  fn main() -> Int = match Cons(1, Nil) { Nil => 0, Cons(h, _) => h, }";
        // Float signature.
        let s3 = "fn main() -> Float = 3.5";
        // print_int builtin call.
        let s4 = "fn main() -> Int = { print_int(1); 0 }";
        // Unit-returning function.
        let s5 = "fn main() -> Unit = ()";
        for src in [s1, s2, s3, s4, s5] {
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
}
