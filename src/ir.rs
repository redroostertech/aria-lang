//! Typed intermediate representation (A-normal form) — the shared foundation
//! for the memory-model proof-of-concept and the future WASM/native backends.
//!
//! The tree-walking interpreter has no notion of allocation, so it can neither
//! host Perceus-style reference counting nor lower to a real backend. This IR
//! fixes that: it is an explicit, sequenced form where every heap allocation
//! (an ADT constructor) and every intermediate value is named, so later passes
//! can insert `dup`/`drop` reference-count operations and reuse analysis.
//!
//! Scope of this stage: the *functional subset* — `Int`, `Bool`, `Float`,
//! `String`, algebraic data types, `let`, `if`, `match`, calls, and arithmetic.
//! Only ADT cells are heap-allocated and reference-counted; `Int`/`Bool`/
//! `Float`/`String`/`Unit` are unboxed value types and so are never `dup`/
//! `drop`ed (a `String` in an ADT field is freed by Rust when its cell is).
//! Tensors/RAG remain out of scope (they involve the opaque Tensor type).
//!
//! Stage 1 (this module today): lowering + an IR interpreter that is
//! differentially checked against the tree-walker, plus heap-allocation
//! counting. `dup`/`drop` insertion and reuse analysis come next.

use std::collections::HashMap;

use crate::ast::{BinOp, Expr, Item, Pattern, Program, Stmt, UnOp};

/// Atomic operand: a variable or an unboxed literal.
#[derive(Debug, Clone)]
pub enum Atom {
    Var(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
}

/// The right-hand side of a `let` binding.
#[derive(Debug, Clone)]
pub enum Bind {
    Atom(Atom),
    Prim(BinOp, Atom, Atom),
    Unary(UnOp, Atom),
    /// Heap allocation: an ADT constructor with already-named field atoms.
    Ctor(String, Vec<Atom>),
    /// Function or builtin call with named argument atoms.
    Call(String, Vec<Atom>),
    /// Constructor that may REUSE a freed cell in place: the first arg names a
    /// reuse token (from `DropReuse`). If the token holds an address, the cell
    /// is overwritten (no allocation); otherwise a fresh cell is allocated.
    CtorReuse(String, String, Vec<Atom>),
    If(Atom, Box<IExpr>, Box<IExpr>),
    /// Match on an ADT scrutinee. Arms are keyed by constructor; an arm with
    /// `ctor == None` is a catch-all (binds the whole scrutinee).
    Match(Atom, Vec<IArm>),
}

#[derive(Debug, Clone)]
pub struct IArm {
    pub ctor: Option<String>,
    pub binders: Vec<String>,
    pub body: IExpr,
}

/// An IR expression: a (possibly empty) sequence of `let`s ending in a return.
/// `Dup`/`Drop` are reference-count operations inserted by the `rc` pass; they
/// are absent in freshly-lowered IR and no-ops on unboxed (non-`Ref`) values.
#[derive(Debug, Clone)]
pub enum IExpr {
    Let(String, Bind, Box<IExpr>),
    Ret(Atom),
    /// Increment the refcount of `var`, then run the body.
    Dup(String, Box<IExpr>),
    /// Decrement the refcount of `var` (freeing recursively at 0), then the body.
    Drop(String, Box<IExpr>),
    /// Drop `scrut` but, if it becomes unique-and-dead, hand its cell address to
    /// the reuse `token` (binding it for a later `CtorReuse`) instead of fully
    /// freeing it. The token is "empty" when the cell was shared.
    DropReuse(String, String, Box<IExpr>),
}

#[derive(Debug, Clone)]
pub struct IFn {
    pub params: Vec<String>,
    pub body: IExpr,
}

pub struct LowerError(pub String);

/// Builtins the IR interpreter implements. Other declared builtins (tensor/RAG/
/// compression) are outside the IR subset and rejected at lowering.
const IR_BUILTINS: &[&str] = &[
    "print_int",
    "print_float",
    "print_bool",
    "print_str",
    "concat",
    "int_to_str",
];

/// Builtins the IR interpreter does NOT evaluate, but which the WASM backend
/// *does* compile (Phase 2f: tensor ops + `embed_similarity`). Lowering lets
/// these through to a `Bind::Call` so the wasm emitter can handle them; the IR
/// tree-walking interpreter never runs them (the tree-walking `interp::Interp`
/// is the reference oracle for these instead). The codec builtins
/// (`compressed_size`, `neural_bits_per_byte`) are intentionally absent — they
/// remain rejected at lowering (deferred).
const WASM_ONLY_BUILTINS: &[&str] = &[
    "tensor_zeros",
    "tensor_set",
    "tensor_get",
    "tensor_rows",
    "tensor_cols",
    "matmul",
    "transpose",
    "softmax",
    "relu",
    "embed_similarity",
];

// ---- lowering (typed AST -> ANF IR) -------------------------------------

struct Lowerer {
    tmp: usize,
}

impl Lowerer {
    fn fresh(&mut self) -> String {
        let n = self.tmp;
        self.tmp += 1;
        format!("$t{}", n)
    }

    /// Lower `e` into a complete IExpr (lets ending in `Ret`).
    fn lower_block(&mut self, e: &Expr) -> Result<IExpr, LowerError> {
        let mut stmts: Vec<(String, Bind)> = Vec::new();
        let atom = self.lower(e, &mut stmts)?;
        // Fold the collected bindings into nested `Let`s, innermost = Ret(atom).
        let mut acc = IExpr::Ret(atom);
        for (name, bind) in stmts.into_iter().rev() {
            acc = IExpr::Let(name, bind, Box::new(acc));
        }
        Ok(acc)
    }

    /// Lower `e`, pushing any needed bindings to `stmts`, returning its atom.
    fn lower(&mut self, e: &Expr, stmts: &mut Vec<(String, Bind)>) -> Result<Atom, LowerError> {
        match e {
            Expr::Int(n) => Ok(Atom::Int(*n)),
            Expr::Float(f) => Ok(Atom::Float(*f)),
            Expr::Bool(b) => Ok(Atom::Bool(*b)),
            Expr::Str(s) => Ok(Atom::Str(s.clone())),
            Expr::Unit => Ok(Atom::Unit),
            Expr::Var(n) => Ok(Atom::Var(n.clone())),

            Expr::Binary(op, l, r) => {
                // Short-circuit `&&` / `||` must NOT evaluate the rhs eagerly —
                // lower to control flow so the rhs runs only in the taken branch
                // (matching the interpreter's short-circuit semantics).
                if matches!(op, BinOp::And | BinOp::Or) {
                    let la = self.lower(l, stmts)?;
                    let rhs = self.lower_block(r)?;
                    let (then, els) = match op {
                        BinOp::And => (rhs, IExpr::Ret(Atom::Bool(false))),
                        _ => (IExpr::Ret(Atom::Bool(true)), rhs), // Or
                    };
                    let t = self.fresh();
                    stmts.push((t.clone(), Bind::If(la, Box::new(then), Box::new(els))));
                    return Ok(Atom::Var(t));
                }
                let la = self.lower(l, stmts)?;
                let ra = self.lower(r, stmts)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::Prim(*op, la, ra)));
                Ok(Atom::Var(t))
            }
            Expr::Unary(op, inner) => {
                let a = self.lower(inner, stmts)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::Unary(*op, a)));
                Ok(Atom::Var(t))
            }
            Expr::Ctor(name, args) => {
                let atoms = self.lower_all(args, stmts)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::Ctor(name.clone(), atoms)));
                Ok(Atom::Var(t))
            }
            Expr::Call(name, args) => {
                // Builtins the IR doesn't implement (tensors/RAG/compression) are
                // outside the IR subset — reject at lowering with a clear message
                // rather than failing confusingly at run time.
                if crate::builtins::lookup(name).is_some()
                    && !IR_BUILTINS.contains(&name.as_str())
                    && !WASM_ONLY_BUILTINS.contains(&name.as_str())
                {
                    return Err(LowerError(format!(
                        "builtin `{}` is outside the IR subset (compression codecs not supported)",
                        name
                    )));
                }
                let atoms = self.lower_all(args, stmts)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::Call(name.clone(), atoms)));
                Ok(Atom::Var(t))
            }
            Expr::If(c, th, el) => {
                let ca = self.lower(c, stmts)?;
                let th_ir = self.lower_block(th)?;
                let el_ir = self.lower_block(el)?;
                let t = self.fresh();
                stmts.push((t.clone(), Bind::If(ca, Box::new(th_ir), Box::new(el_ir))));
                Ok(Atom::Var(t))
            }
            Expr::Match(scrut, arms) => {
                let sa = self.lower(scrut, stmts)?;
                let bind = self.lower_match(sa, arms)?;
                let t = self.fresh();
                stmts.push((t.clone(), bind));
                Ok(Atom::Var(t))
            }
            Expr::Block(block_stmts, last) => {
                for s in block_stmts {
                    match s {
                        Stmt::Let(name, _ty, value) => {
                            let va = self.lower(value, stmts)?;
                            stmts.push((name.clone(), Bind::Atom(va)));
                        }
                        Stmt::Expr(ex) => {
                            // Evaluate for effect; bind to a discarded temp.
                            let a = self.lower(ex, stmts)?;
                            let t = self.fresh();
                            stmts.push((t, Bind::Atom(a)));
                        }
                    }
                }
                self.lower(last, stmts)
            }
        }
    }

    fn lower_all(&mut self, es: &[Expr], stmts: &mut Vec<(String, Bind)>) -> Result<Vec<Atom>, LowerError> {
        es.iter().map(|e| self.lower(e, stmts)).collect()
    }

    /// Lower a `match`. ADT (constructor) patterns become a `Match` bind;
    /// integer/bool literal patterns become an `if`-chain.
    fn lower_match(&mut self, scrut: Atom, arms: &[crate::ast::Arm]) -> Result<Bind, LowerError> {
        let has_ctor = arms.iter().any(|a| matches!(a.pat, Pattern::Ctor(_, _)));
        if has_ctor {
            let mut iarms = Vec::new();
            for arm in arms {
                let body = self.lower_block(&arm.body)?;
                match &arm.pat {
                    Pattern::Ctor(name, subs) => {
                        let mut binders = Vec::new();
                        for sp in subs {
                            match sp {
                                Pattern::Var(n) => binders.push(n.clone()),
                                Pattern::Wild => binders.push(self.fresh()),
                                _ => {
                                    return Err(LowerError(
                                        "nested constructor patterns not supported in IR yet".into(),
                                    ))
                                }
                            }
                        }
                        iarms.push(IArm { ctor: Some(name.clone()), binders, body });
                    }
                    Pattern::Var(n) => {
                        iarms.push(IArm { ctor: None, binders: vec![n.clone()], body })
                    }
                    Pattern::Wild => {
                        iarms.push(IArm { ctor: None, binders: vec![self.fresh()], body })
                    }
                    _ => {
                        return Err(LowerError(
                            "mixed literal/constructor patterns not supported in IR yet".into(),
                        ))
                    }
                }
            }
            Ok(Bind::Match(scrut, iarms))
        } else {
            // Literal (Int/Bool) match -> if-chain in SOURCE ORDER, preserving
            // the interpreter's first-match-wins semantics. Fold arms
            // back-to-front: a catch-all (Var/Wild) REPLACES the accumulator (so
            // any later arms become unreachable, exactly as the interpreter
            // treats arms after a catch-all); a literal arm wraps the
            // accumulator in `if scrut == lit { body } else { acc }`. Folding the
            // first catch-all last makes it dominate later arms correctly.
            let mut chain = IExpr::Ret(Atom::Unit); // exhaustiveness => unreachable
            for arm in arms.iter().rev() {
                match &arm.pat {
                    Pattern::Var(_) | Pattern::Wild => {
                        chain = self.lower_catchall(&scrut, arm)?;
                    }
                    Pattern::Int(_) | Pattern::Bool(_) => {
                        let cond = self.lit_cond(&scrut, &arm.pat)?;
                        let then = self.lower_block(&arm.body)?;
                        let c = self.fresh();
                        let t = self.fresh();
                        let if_bind =
                            Bind::If(Atom::Var(c.clone()), Box::new(then), Box::new(chain));
                        let inner =
                            IExpr::Let(t.clone(), if_bind, Box::new(IExpr::Ret(Atom::Var(t))));
                        chain = IExpr::Let(c, cond, Box::new(inner));
                    }
                    _ => {
                        return Err(LowerError(
                            "mixed literal/constructor patterns not supported in IR yet".into(),
                        ))
                    }
                }
            }
            // lower_match returns a Bind; wrap the chain in an identity `if true`.
            Ok(Bind::If(Atom::Bool(true), Box::new(chain), Box::new(IExpr::Ret(Atom::Unit))))
        }
    }

    fn lower_catchall(&mut self, scrut: &Atom, arm: &crate::ast::Arm) -> Result<IExpr, LowerError> {
        // Bind the catch-all variable (if any) to the scrutinee, then the body.
        let body = self.lower_block(&arm.body)?;
        match &arm.pat {
            Pattern::Var(n) => Ok(IExpr::Let(n.clone(), Bind::Atom(scrut.clone()), Box::new(body))),
            _ => Ok(body),
        }
    }

    fn lit_cond(&mut self, scrut: &Atom, pat: &Pattern) -> Result<Bind, LowerError> {
        let rhs = match pat {
            Pattern::Int(i) => Atom::Int(*i),
            Pattern::Bool(b) => Atom::Bool(*b),
            _ => return Err(LowerError("unsupported literal pattern".into())),
        };
        Ok(Bind::Prim(BinOp::Eq, scrut.clone(), rhs))
    }
}

/// Lower an entire program's functions to IR. Returns an error if any function
/// uses a feature outside the IR subset.
pub fn lower_program(program: &Program) -> Result<HashMap<String, IFn>, String> {
    let mut fns = HashMap::new();
    let mut lw = Lowerer { tmp: 0 };
    for item in &program.items {
        if let Item::Fn(f) = item {
            let body = lw.lower_block(&f.body).map_err(|e| format!("fn `{}`: {}", f.name, e.0))?;
            let params = f.params.iter().map(|p| p.name.clone()).collect();
            fns.insert(f.name.clone(), IFn { params, body });
        }
    }
    Ok(fns)
}

// ---- IR interpreter with allocation counting ----------------------------

#[derive(Debug, Clone)]
pub enum IValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    /// A heap reference to an ADT cell.
    Ref(usize),
    /// A reuse token: `Some(addr)` if a freed cell is available for in-place
    /// reuse, `None` otherwise. Compiler-internal; never user-visible.
    Token(Option<usize>),
}

#[derive(Debug, Clone)]
struct Cell {
    ctor: String,
    fields: Vec<IValue>,
    rc: u32,
}

/// Metrics gathered while interpreting the IR.
#[derive(Debug, Default, Clone)]
pub struct Metrics {
    /// Total ADT cells constructed (gross allocations).
    pub allocations: usize,
    /// Cells freed by `drop` reaching refcount 0.
    pub frees: usize,
    /// Cells reused in place by `CtorReuse` (avoided a fresh allocation).
    pub reuses: usize,
    pub dups: usize,
    pub drops: usize,
    /// Currently-live cells (allocations - frees so far).
    pub live: usize,
    /// Maximum simultaneously-live cells during the run.
    pub peak_live: usize,
}

/// Max IR call-nesting before a catchable error. The IR interpreter, like the
/// AST one, is run on a large-stack thread (see main.rs), so this is generous;
/// it exists to turn non-terminating recursion into an error, not a crash.
const MAX_IR_CALL_DEPTH: usize = 100_000;

pub struct IrInterp {
    fns: HashMap<String, IFn>,
    /// `None` = a freed slot (used to detect use-after-free).
    heap: Vec<Option<Cell>>,
    depth: usize,
    pub metrics: Metrics,
    /// True when the IR has been through the rc pass (`dup`/`drop` inserted), so
    /// the interpreter actively manages reference counts. A `Bind::Prim(Eq|Ne)`
    /// on ADTs consumes its operands — but only the rc'd IR has dup'd a reused
    /// value beforehand, so the operand-drop is performed ONLY in this mode. On
    /// un-rc'd IR (no dups inserted), the comparison leaves operands alone, which
    /// avoids a spurious double-free for e.g. `v == v` (the un-rc'd baseline
    /// never reclaims memory anyway — it only checks the result value).
    manage_rc: bool,
}

type Env = HashMap<String, IValue>;

impl IrInterp {
    /// Construct an interpreter for rc'd IR (manages reference counts). This is
    /// the production path (the wasm backend always rc's its IR).
    pub fn new(fns: HashMap<String, IFn>) -> Self {
        IrInterp {
            fns,
            heap: Vec::new(),
            depth: 0,
            metrics: Metrics::default(),
            manage_rc: true,
        }
    }

    /// Construct an interpreter for un-rc'd IR (no dup/drop inserted). Used by
    /// the differential fuzz baseline; it does not reclaim memory and so does
    /// not perform the `Eq`/`Ne` operand-drop.
    pub fn new_no_rc(fns: HashMap<String, IFn>) -> Self {
        IrInterp {
            fns,
            heap: Vec::new(),
            depth: 0,
            metrics: Metrics::default(),
            manage_rc: false,
        }
    }

    /// Run `main` and return its value (plus collected metrics in `self`).
    pub fn run_main(&mut self) -> Result<IValue, String> {
        let main = self.fns.get("main").ok_or("no `main`")?.clone();
        let env = Env::new();
        self.eval(&main.body, env)
    }

    /// Render an IR value into the same textual form as `interp::Value::display`
    /// (recursively reading ADT cells from the heap), so IR and interpreter
    /// results can be compared structurally rather than by ad-hoc digit soup.
    pub fn render(&self, v: &IValue) -> String {
        match v {
            IValue::Int(n) => n.to_string(),
            // Match interp::Value::display exactly so the cross-check passes.
            IValue::Float(f) => format!("{}", f),
            IValue::Bool(b) => b.to_string(),
            IValue::Str(s) => s.clone(),
            IValue::Unit => "()".to_string(),
            IValue::Token(_) => "<token>".to_string(), // never a program result
            IValue::Ref(a) => {
                let c = self.heap[*a].as_ref().expect("render: freed cell escaped");
                if c.fields.is_empty() {
                    c.ctor.clone()
                } else {
                    let inner: Vec<String> = c.fields.iter().map(|f| self.render(f)).collect();
                    format!("{}({})", c.ctor, inner.join(", "))
                }
            }
        }
    }

    fn atom(&self, a: &Atom, env: &Env) -> Result<IValue, String> {
        Ok(match a {
            Atom::Int(n) => IValue::Int(*n),
            Atom::Float(f) => IValue::Float(*f),
            Atom::Bool(b) => IValue::Bool(*b),
            Atom::Str(s) => IValue::Str(s.clone()),
            Atom::Unit => IValue::Unit,
            Atom::Var(n) => env.get(n).cloned().ok_or_else(|| format!("ir: unbound {}", n))?,
        })
    }

    fn eval(&mut self, e: &IExpr, mut env: Env) -> Result<IValue, String> {
        // Real trampoline over the `let`-chain: iterate instead of recursing, so
        // a long sequence of bindings does not grow the native stack.
        let mut e = e;
        loop {
            match e {
                IExpr::Ret(a) => return self.atom(a, &env),
                IExpr::Let(x, bind, body) => {
                    let v = self.eval_bind(bind, &env)?;
                    env.insert(x.clone(), v);
                    e = body;
                }
                IExpr::Dup(v, body) => {
                    if let IValue::Ref(a) = self.atom(&Atom::Var(v.clone()), &env)? {
                        match &mut self.heap[a] {
                            Some(cell) => cell.rc += 1,
                            None => return Err(format!("ir: dup of freed cell `{}`", v)),
                        }
                        self.metrics.dups += 1;
                    }
                    e = body;
                }
                IExpr::Drop(v, body) => {
                    if let IValue::Ref(a) = self.atom(&Atom::Var(v.clone()), &env)? {
                        self.metrics.drops += 1;
                        self.drop_cell(a)?;
                    }
                    e = body;
                }
                IExpr::DropReuse(scrut, tok, body) => {
                    let token = match self.atom(&Atom::Var(scrut.clone()), &env)? {
                        IValue::Ref(a) => {
                            self.metrics.drops += 1;
                            self.drop_for_reuse(a)?
                        }
                        // Non-heap scrutinee can't be reused.
                        _ => IValue::Token(None),
                    };
                    env.insert(tok.clone(), token);
                    e = body;
                }
            }
        }
    }

    /// Like `drop_cell`, but when the cell becomes unique-and-dead we KEEP its
    /// slot (releasing only its children) and return a reuse token holding the
    /// address, so a subsequent `CtorReuse` can overwrite it in place.
    fn drop_for_reuse(&mut self, addr: usize) -> Result<IValue, String> {
        let reached_zero = match &mut self.heap[addr] {
            Some(cell) => {
                if cell.rc == 0 {
                    return Err("ir: drop of cell with refcount 0 (double free)".into());
                }
                cell.rc -= 1;
                cell.rc == 0
            }
            None => return Err("ir: drop of already-freed cell".into()),
        };
        if reached_zero {
            // Release the children but retain the slot for reuse.
            let fields = std::mem::take(&mut self.heap[addr].as_mut().unwrap().fields);
            for f in &fields {
                if let IValue::Ref(a) = f {
                    self.drop_cell(*a)?;
                }
            }
            Ok(IValue::Token(Some(addr)))
        } else {
            Ok(IValue::Token(None))
        }
    }

    /// Structural equality matching `interp::values_equal`: scalars by value
    /// (`NaN != NaN`), ADT cells by constructor + fields read from the heap.
    fn ir_equal(&self, a: &IValue, b: &IValue) -> bool {
        match (a, b) {
            (IValue::Int(x), IValue::Int(y)) => x == y,
            (IValue::Float(x), IValue::Float(y)) => x == y,
            (IValue::Bool(x), IValue::Bool(y)) => x == y,
            (IValue::Str(x), IValue::Str(y)) => x == y,
            (IValue::Unit, IValue::Unit) => true,
            (IValue::Ref(x), IValue::Ref(y)) => match (&self.heap[*x], &self.heap[*y]) {
                (Some(cx), Some(cy)) => {
                    cx.ctor == cy.ctor
                        && cx.fields.len() == cy.fields.len()
                        && cx.fields.iter().zip(&cy.fields).all(|(p, q)| self.ir_equal(p, q))
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Decrement a cell's refcount; at zero, free it and recursively drop its
    /// `Ref` fields. This is the runtime side of Perceus reference counting.
    fn drop_cell(&mut self, addr: usize) -> Result<(), String> {
        let reached_zero = match &mut self.heap[addr] {
            Some(cell) => {
                if cell.rc == 0 {
                    return Err("ir: drop of cell with refcount 0 (double free)".into());
                }
                cell.rc -= 1;
                cell.rc == 0
            }
            None => return Err("ir: drop of already-freed cell".into()),
        };
        if reached_zero {
            // Take the cell out, free the slot, then drop child references.
            let cell = self.heap[addr].take().unwrap();
            self.metrics.frees += 1;
            self.metrics.live -= 1;
            for f in &cell.fields {
                if let IValue::Ref(a) = f {
                    self.drop_cell(*a)?;
                }
            }
        }
        Ok(())
    }

    fn eval_bind(&mut self, bind: &Bind, env: &Env) -> Result<IValue, String> {
        match bind {
            Bind::Atom(a) => self.atom(a, env),
            Bind::Unary(op, a) => {
                let v = self.atom(a, env)?;
                match (op, v) {
                    (UnOp::Neg, IValue::Int(n)) => {
                        Ok(IValue::Int(n.checked_neg().ok_or("ir: neg overflow")?))
                    }
                    (UnOp::Neg, IValue::Float(f)) => Ok(IValue::Float(-f)),
                    (UnOp::Not, IValue::Bool(b)) => Ok(IValue::Bool(!b)),
                    _ => Err("ir: bad unary".into()),
                }
            }
            Bind::Prim(op, a, b) => {
                let x = self.atom(a, env)?;
                let y = self.atom(b, env)?;
                // Eq/Ne work on ADTs too (typeck allows it); compare structurally
                // via the heap, exactly like interp::values_equal. prim() handles
                // the arithmetic/ordering operators on scalars.
                match op {
                    BinOp::Eq | BinOp::Ne => {
                        let equal = self.ir_equal(&x, &y);
                        // The rc pass marks `Eq`/`Ne` operands as CONSUMED (it
                        // dups any value reused later and inserts no drop here),
                        // so this comparison OWNS one reference to each operand
                        // and must release it — after `ir_equal` read the heap —
                        // to stay garbage-free, mirroring the wasm `__eq` site.
                        // Only in rc mode: the un-rc'd baseline never dup'd a
                        // reused operand, so dropping here would double-free e.g.
                        // `v == v`.
                        if self.manage_rc {
                            if let IValue::Ref(a) = x {
                                self.metrics.drops += 1;
                                self.drop_cell(a)?;
                            }
                            if let IValue::Ref(a) = y {
                                self.metrics.drops += 1;
                                self.drop_cell(a)?;
                            }
                        }
                        Ok(IValue::Bool(if *op == BinOp::Eq { equal } else { !equal }))
                    }
                    _ => prim(*op, x, y),
                }
            }
            Bind::Ctor(name, args) => {
                let fields = args.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                self.metrics.allocations += 1;
                self.metrics.live += 1;
                self.metrics.peak_live = self.metrics.peak_live.max(self.metrics.live);
                self.heap.push(Some(Cell { ctor: name.clone(), fields, rc: 1 }));
                Ok(IValue::Ref(self.heap.len() - 1))
            }
            Bind::CtorReuse(tok, name, args) => {
                let fields: Vec<IValue> =
                    args.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                match self.atom(&Atom::Var(tok.clone()), env)? {
                    IValue::Token(Some(addr)) => {
                        // Reuse the freed slot in place — no allocation.
                        self.metrics.reuses += 1;
                        self.heap[addr] = Some(Cell { ctor: name.clone(), fields, rc: 1 });
                        Ok(IValue::Ref(addr))
                    }
                    _ => {
                        // Token empty (cell was shared): allocate fresh.
                        self.metrics.allocations += 1;
                        self.metrics.live += 1;
                        self.metrics.peak_live = self.metrics.peak_live.max(self.metrics.live);
                        self.heap.push(Some(Cell { ctor: name.clone(), fields, rc: 1 }));
                        Ok(IValue::Ref(self.heap.len() - 1))
                    }
                }
            }
            Bind::Call(name, args) => {
                let vals: Vec<IValue> =
                    args.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                if let Some(v) = self.builtin(name, &vals)? {
                    return Ok(v);
                }
                let f = self.fns.get(name).cloned().ok_or_else(|| format!("ir: unknown fn {}", name))?;
                if f.params.len() != vals.len() {
                    return Err(format!(
                        "ir: function `{}` expects {} argument(s), got {}",
                        name,
                        f.params.len(),
                        vals.len()
                    ));
                }
                let d = self.depth + 1;
                if d > MAX_IR_CALL_DEPTH {
                    return Err(format!("ir: maximum recursion depth ({}) exceeded", MAX_IR_CALL_DEPTH));
                }
                self.depth = d;
                let mut frame = Env::new();
                for (p, v) in f.params.iter().zip(vals.into_iter()) {
                    frame.insert(p.clone(), v);
                }
                let result = self.eval(&f.body, frame);
                self.depth = d - 1;
                result
            }
            Bind::If(c, then, els) => match self.atom(c, env)? {
                IValue::Bool(true) => self.eval(then, env.clone()),
                IValue::Bool(false) => self.eval(els, env.clone()),
                _ => Err("ir: if cond not bool".into()),
            },
            Bind::Match(scrut, arms) => {
                let v = self.atom(scrut, env)?;
                let addr = match v {
                    IValue::Ref(a) => a,
                    _ => return Err("ir: match on non-data".into()),
                };
                let cell = self
                    .heap[addr]
                    .clone()
                    .ok_or("ir: use-after-free in match")?;
                for arm in arms {
                    let matches = match &arm.ctor {
                        Some(c) => *c == cell.ctor,
                        None => true, // catch-all
                    };
                    if matches {
                        let mut frame = env.clone();
                        if arm.ctor.is_some() {
                            for (b, f) in arm.binders.iter().zip(cell.fields.iter()) {
                                frame.insert(b.clone(), f.clone());
                            }
                        } else if let Some(b) = arm.binders.first() {
                            frame.insert(b.clone(), IValue::Ref(addr));
                        }
                        return self.eval(&arm.body, frame);
                    }
                }
                Err(format!("ir: no match arm for {}", cell.ctor))
            }
        }
    }

    fn builtin(&self, name: &str, args: &[IValue]) -> Result<Option<IValue>, String> {
        match name {
            "print_int" => match args {
                [IValue::Int(n)] => {
                    println!("{}", n);
                    Ok(Some(IValue::Unit))
                }
                _ => Err("ir: print_int expects Int".into()),
            },
            "print_float" => match args {
                [IValue::Float(f)] => {
                    println!("{}", f);
                    Ok(Some(IValue::Unit))
                }
                _ => Err("ir: print_float expects Float".into()),
            },
            "print_bool" => match args {
                [IValue::Bool(b)] => {
                    println!("{}", b);
                    Ok(Some(IValue::Unit))
                }
                _ => Err("ir: print_bool expects Bool".into()),
            },
            "print_str" => match args {
                [IValue::Str(s)] => {
                    println!("{}", s);
                    Ok(Some(IValue::Unit))
                }
                _ => Err("ir: print_str expects String".into()),
            },
            "concat" => match args {
                [IValue::Str(a), IValue::Str(b)] => Ok(Some(IValue::Str(format!("{}{}", a, b)))),
                _ => Err("ir: concat expects two Strings".into()),
            },
            "int_to_str" => match args {
                [IValue::Int(n)] => Ok(Some(IValue::Str(n.to_string()))),
                _ => Err("ir: int_to_str expects Int".into()),
            },
            _ => Ok(None),
        }
    }
}

fn prim(op: BinOp, x: IValue, y: IValue) -> Result<IValue, String> {
    use BinOp::*;
    use IValue::*;
    match (x, y) {
        (Int(a), Int(b)) => Ok(match op {
            Add => Int(a.checked_add(b).ok_or("ir: + overflow")?),
            Sub => Int(a.checked_sub(b).ok_or("ir: - overflow")?),
            Mul => Int(a.checked_mul(b).ok_or("ir: * overflow")?),
            Div => {
                if b == 0 {
                    return Err("division by zero".into());
                }
                Int(a.checked_div(b).ok_or("integer overflow in `/`")?)
            }
            Mod => {
                if b == 0 {
                    return Err("modulo by zero".into());
                }
                Int(a.checked_rem(b).ok_or("integer overflow in `%`")?)
            }
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            Lt => Bool(a < b),
            Le => Bool(a <= b),
            Gt => Bool(a > b),
            Ge => Bool(a >= b),
            And | Or => return Err("ir: logical op on Int".into()),
        }),
        // Float arithmetic mirrors interp::float_op: Div by 0.0 yields
        // infinity/NaN (NOT an error), and there is no float `Mod`.
        (Float(a), Float(b)) => Ok(match op {
            Add => Float(a + b),
            Sub => Float(a - b),
            Mul => Float(a * b),
            Div => Float(a / b),
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            Lt => Bool(a < b),
            Le => Bool(a <= b),
            Gt => Bool(a > b),
            Ge => Bool(a >= b),
            Mod | And | Or => return Err("ir: bad float op".into()),
        }),
        (Bool(a), Bool(b)) => Ok(match op {
            And => Bool(a && b),
            Or => Bool(a || b),
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            _ => return Err("ir: bad bool op".into()),
        }),
        // Strings support only equality (matching interp::values_equal).
        (Str(a), Str(b)) => Ok(match op {
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            _ => return Err("ir: bad string op".into()),
        }),
        _ => Err("ir: prim type mismatch".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interp, lexer, parser, typeck};

    // Lower + run through the IR interpreter, returning a STRUCTURAL string
    // identical to interp::Value::display (so Data/Bool/Unit compare faithfully).
    fn ir_run(src: &str) -> Result<String, String> {
        let toks = lexer::lex(src)?;
        let prog = parser::parse(toks)?;
        typeck::check(&prog).map_err(|e| e.join("; "))?;
        let fns = lower_program(&prog)?;
        let mut ir = IrInterp::new(fns);
        let v = ir.run_main()?;
        Ok(ir.render(&v))
    }

    // The tree-walker's result, for differential comparison.
    fn ast_run(src: &str) -> Result<String, String> {
        let toks = lexer::lex(src)?;
        let prog = parser::parse(toks)?;
        let it = interp::Interp::new(&prog)?;
        it.run_main().map(|v| v.display())
    }

    // The IR and interpreter must agree on the Ok/Err SHAPE and, when Ok, on the
    // exact structural value. (Both erroring counts as agreement; messages may
    // differ.)
    fn differential(src: &str) {
        let ir = ir_run(src);
        let ast = ast_run(src);
        match (&ir, &ast) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "value mismatch for:\n{}", src),
            (Err(_), Err(_)) => {}
            _ => panic!("Ok/Err shape mismatch for:\n{}\n  ir={:?}\n  ast={:?}", src, ir, ast),
        }
    }

    #[test]
    fn factorial_matches_interpreter() {
        differential("fn fac(n: Int) -> Int = match n { 0 => 1, _ => n * fac(n - 1), }\nfn main() -> Int = fac(6)");
    }

    #[test]
    fn list_sum_matches_interpreter_and_counts_allocs() {
        let src = "type L = | Nil | Cons(Int, L)\n\
                   fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
                   fn main() -> Int = sum(Cons(1, Cons(2, Cons(3, Nil))))";
        differential(src);
        let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
        let mut ir = IrInterp::new(lower_program(&prog).unwrap());
        ir.run_main().unwrap();
        assert_eq!(ir.metrics.allocations, 4, "expected 4 ADT allocations");
    }

    #[test]
    fn returns_constructed_value_structurally() {
        // main returns a Data value; render must match interp's display exactly.
        let src = "type P = | Pair(Int, Int)\nfn main() -> P = Pair(1, 2)";
        let ir = ir_run(src).unwrap();
        let ast = ast_run(src).unwrap();
        assert_eq!(ir, ast);
        assert_eq!(ir, "Pair(1, 2)");
    }

    #[test]
    fn if_and_bool_match() {
        differential("fn classify(n: Int) -> Int = if n < 0 { 0 } else { 1 }\nfn main() -> Int = classify(5)");
    }

    #[test]
    fn float_negation_matches_interpreter() {
        differential("fn main() -> Float = -2.5");
        differential("fn neg(x: Float) -> Float = -x\nfn main() -> Float = neg(3.5)");
    }

    #[test]
    fn adt_equality_matches_interpreter() {
        differential("type P = | P(Int, Int)\nfn main() -> Bool = P(1, 2) == P(1, 2)");
        differential("type P = | P(Int, Int)\nfn main() -> Bool = P(1, 2) == P(1, 9)");
        // structural over String fields too
        differential("type R = | R(String, Int)\nfn main() -> Bool = R(\"a\", 1) == R(\"a\", 1)");
    }

    #[test]
    fn tensor_builtin_lowers_for_wasm() {
        // Phase 2f: tensor builtins now lower (the wasm backend compiles them).
        let prog = parser::parse(
            lexer::lex("fn main() -> Float = tensor_get(tensor_zeros(2, 2), 0, 0)").unwrap(),
        )
        .unwrap();
        assert!(
            lower_program(&prog).is_ok(),
            "tensor builtins must lower for the wasm backend"
        );
    }

    #[test]
    fn codec_builtin_rejected_at_lowering() {
        // The rANS/predictor codec builtins remain outside the IR subset.
        let prog = parser::parse(
            lexer::lex("fn main() -> Int = compressed_size(\"hello\")").unwrap(),
        )
        .unwrap();
        assert!(
            lower_program(&prog).is_err(),
            "codec builtins must be rejected by IR lowering"
        );
    }

    // ---- regression tests for the adversarial-review findings ------------

    #[test]
    fn catchall_before_literal_first_match_wins() {
        // Interp takes the first (catch-all) arm; IR must too (not the literal).
        differential("fn f(n: Int) -> Int = match n { _ => 0, 1 => 99, }\nfn main() -> Int = f(1)");
    }

    #[test]
    fn literal_after_catchall_is_dead() {
        differential("fn f(n: Int) -> Int = match n { 1 => 10, x => 100, 2 => 20, }\nfn main() -> Int = f(2)");
    }

    #[test]
    fn first_of_two_catchalls_wins() {
        differential("fn f(n: Int) -> Int = match n { x => x, y => y + 1000, }\nfn main() -> Int = f(5)");
    }

    #[test]
    fn short_circuit_and_does_not_eval_rhs() {
        // false && (1/0 == 0): interp short-circuits to false (no div error); IR must too.
        differential("fn main() -> Int = { let b = false && (1 / 0 == 0); if b { 1 } else { 0 } }");
    }

    #[test]
    fn short_circuit_or_does_not_eval_rhs() {
        differential("fn main() -> Int = { let b = true || (1 / 0 == 0); if b { 1 } else { 0 } }");
    }

    #[test]
    fn div_by_zero_errors_in_both() {
        differential("fn main() -> Int = 1 / 0");
    }

    #[test]
    fn add_overflow_errors_in_both() {
        differential("fn main() -> Int = 9223372036854775807 + 1");
    }

    // ---- Float / String subset -----------------------------------------

    #[test]
    fn float_arithmetic_matches_interpreter() {
        // Float add/mul/div + a comparison; IR result must equal the interp's.
        differential("fn area(r: Float) -> Float = 3.14159 * r * r\nfn main() -> Float = area(2.0) / 2.0");
    }

    #[test]
    fn string_concat_and_int_to_str_matches_interpreter() {
        differential("fn main() -> String = concat(\"x = \", int_to_str(6 * 7))");
    }

    #[test]
    fn adt_with_string_fields_is_garbage_free() {
        // A list of records carrying String fields: after the run returns an
        // Int, every ADT cell (including those holding Strings) must be freed.
        let src = "type Rec = | R(String, Int)\n\
                   type L = | Nil | Cons(Rec, L)\n\
                   fn count(xs: L) -> Int = match xs { Nil => 0, Cons(_, r) => 1 + count(r), }\n\
                   fn main() -> Int = count(Cons(R(\"a\", 1), Cons(R(\"b\", 2), Nil)))";
        differential(src);
        let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
        typeck::check(&prog).unwrap();
        let fns = crate::rc::insert_rc(&lower_program(&prog).unwrap());
        let mut ir = IrInterp::new(fns);
        ir.run_main().unwrap();
        assert_eq!(ir.metrics.live, 0, "all String-carrying ADT cells must be freed");
    }

    #[test]
    fn deep_recursion_does_not_abort() {
        // Run on a large-stack thread (as the CLI does) and confirm a depth-3000
        // recursion returns the correct value rather than overflowing the stack.
        let src = "fn count(n: Int) -> Int = if n == 0 { 0 } else { count(n - 1) + 1 }\nfn main() -> Int = count(3000)";
        let out = std::thread::Builder::new()
            .stack_size(1 << 30)
            .spawn(move || {
                let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
                let mut ir = IrInterp::new(lower_program(&prog).unwrap());
                let v = ir.run_main().expect("ir run");
                ir.render(&v)
            })
            .unwrap()
            .join()
            .expect("ir thread must not abort");
        assert_eq!(out, "3000");
    }
}


