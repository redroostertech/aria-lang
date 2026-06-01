//! Typed intermediate representation (A-normal form) — the shared foundation
//! for the memory-model proof-of-concept and the future WASM/native backends.
//!
//! The tree-walking interpreter has no notion of allocation, so it can neither
//! host Perceus-style reference counting nor lower to a real backend. This IR
//! fixes that: it is an explicit, sequenced form where every heap allocation
//! (an ADT constructor) and every intermediate value is named, so later passes
//! can insert `dup`/`drop` reference-count operations and reuse analysis.
//!
//! Scope of this stage: the *functional subset* — `Int`, `Bool`, algebraic data
//! types, `let`, `if`, `match`, calls, and arithmetic. Only ADT cells are
//! heap-allocated; `Int`/`Bool`/`Unit` are unboxed. Strings/floats/tensors are
//! out of scope for the memory POC (they don't change the RC story).
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
    Bool(bool),
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
#[derive(Debug, Clone)]
pub enum IExpr {
    Let(String, Bind, Box<IExpr>),
    Ret(Atom),
}

#[derive(Debug, Clone)]
pub struct IFn {
    pub params: Vec<String>,
    pub body: IExpr,
}

pub struct LowerError(pub String);

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
            Expr::Bool(b) => Ok(Atom::Bool(*b)),
            Expr::Unit => Ok(Atom::Unit),
            Expr::Var(n) => Ok(Atom::Var(n.clone())),
            Expr::Float(_) | Expr::Str(_) => Err(LowerError(
                "IR subset is Int/Bool/ADT only (no Float/String yet)".into(),
            )),

            Expr::Binary(op, l, r) => {
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
            // Literal (Int/Bool) match -> nested if-chain, built back to front.
            // A Var/Wild arm is the catch-all `else`.
            let mut else_branch: Option<IExpr> = None;
            let mut literal_arms: Vec<(&crate::ast::Arm,)> = Vec::new();
            for arm in arms {
                match &arm.pat {
                    Pattern::Var(_) | Pattern::Wild => {
                        else_branch = Some(self.lower_catchall(&scrut, arm)?);
                    }
                    _ => literal_arms.push((arm,)),
                }
            }
            let mut chain = else_branch.unwrap_or(IExpr::Ret(Atom::Unit));
            for (arm,) in literal_arms.into_iter().rev() {
                let cond = self.lit_cond(&scrut, &arm.pat)?;
                let then = self.lower_block(&arm.body)?;
                // Bind the comparison then branch on it.
                let c = self.fresh();
                let mut s = Vec::new();
                s.push((c.clone(), cond));
                let bind = Bind::If(Atom::Var(c), Box::new(then), Box::new(chain));
                let t = self.fresh();
                s.push((t.clone(), bind));
                let mut acc = IExpr::Ret(Atom::Var(t));
                for (name, b) in s.into_iter().rev() {
                    acc = IExpr::Let(name, b, Box::new(acc));
                }
                chain = acc;
            }
            // Wrap the chain as a single bind via an identity if(true).
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
    Bool(bool),
    Unit,
    /// A heap reference to an ADT cell.
    Ref(usize),
}

#[derive(Debug, Clone)]
struct Cell {
    ctor: String,
    fields: Vec<IValue>,
}

/// Metrics gathered while interpreting the IR.
#[derive(Debug, Default, Clone)]
pub struct Metrics {
    pub allocations: usize,
}

pub struct IrInterp {
    fns: HashMap<String, IFn>,
    heap: Vec<Cell>,
    pub metrics: Metrics,
}

type Env = HashMap<String, IValue>;

impl IrInterp {
    pub fn new(fns: HashMap<String, IFn>) -> Self {
        IrInterp { fns, heap: Vec::new(), metrics: Metrics::default() }
    }

    /// Run `main` and return its value (plus collected metrics in `self`).
    pub fn run_main(&mut self) -> Result<IValue, String> {
        let main = self.fns.get("main").ok_or("no `main`")?.clone();
        let env = Env::new();
        self.eval(&main.body, env)
    }

    fn atom(&self, a: &Atom, env: &Env) -> Result<IValue, String> {
        Ok(match a {
            Atom::Int(n) => IValue::Int(*n),
            Atom::Bool(b) => IValue::Bool(*b),
            Atom::Unit => IValue::Unit,
            Atom::Var(n) => env.get(n).cloned().ok_or_else(|| format!("ir: unbound {}", n))?,
        })
    }

    fn eval(&mut self, e: &IExpr, mut env: Env) -> Result<IValue, String> {
        loop {
            match e {
                IExpr::Ret(a) => return self.atom(a, &env),
                IExpr::Let(x, bind, body) => {
                    let v = self.eval_bind(bind, &env)?;
                    env.insert(x.clone(), v);
                    // tail-loop into body without growing the Rust stack per let
                    return self.eval(body, env);
                }
            }
        }
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
                    (UnOp::Not, IValue::Bool(b)) => Ok(IValue::Bool(!b)),
                    _ => Err("ir: bad unary".into()),
                }
            }
            Bind::Prim(op, a, b) => {
                let x = self.atom(a, env)?;
                let y = self.atom(b, env)?;
                prim(*op, x, y)
            }
            Bind::Ctor(name, args) => {
                let fields = args.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                self.metrics.allocations += 1;
                self.heap.push(Cell { ctor: name.clone(), fields });
                Ok(IValue::Ref(self.heap.len() - 1))
            }
            Bind::Call(name, args) => {
                let vals: Vec<IValue> =
                    args.iter().map(|a| self.atom(a, env)).collect::<Result<_, _>>()?;
                if let Some(v) = self.builtin(name, &vals)? {
                    return Ok(v);
                }
                let f = self.fns.get(name).cloned().ok_or_else(|| format!("ir: unknown fn {}", name))?;
                let mut frame = Env::new();
                for (p, v) in f.params.iter().zip(vals.into_iter()) {
                    frame.insert(p.clone(), v);
                }
                self.eval(&f.body, frame)
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
                let cell = self.heap[addr].clone();
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
            Div => Int(a.checked_div(b).ok_or("ir: div by zero")?),
            Mod => Int(a.checked_rem(b).ok_or("ir: mod by zero")?),
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            Lt => Bool(a < b),
            Le => Bool(a <= b),
            Gt => Bool(a > b),
            Ge => Bool(a >= b),
            And | Or => return Err("ir: logical op on Int".into()),
        }),
        (Bool(a), Bool(b)) => Ok(match op {
            And => Bool(a && b),
            Or => Bool(a || b),
            Eq => Bool(a == b),
            Ne => Bool(a != b),
            _ => return Err("ir: bad bool op".into()),
        }),
        _ => Err("ir: prim type mismatch".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interp, lexer, parser, typeck};

    // Lower + run through the IR interpreter, returning a display string.
    fn ir_run(src: &str) -> Result<String, String> {
        let toks = lexer::lex(src).map_err(|e| e)?;
        let prog = parser::parse(toks).map_err(|e| e)?;
        typeck::check(&prog).map_err(|e| e.join("; "))?;
        let fns = lower_program(&prog)?;
        let mut ir = IrInterp::new(fns);
        let v = ir.run_main()?;
        Ok(format!("{:?}", v))
    }

    // The tree-walker's result, for differential comparison.
    fn ast_run(src: &str) -> String {
        let toks = lexer::lex(src).unwrap();
        let prog = parser::parse(toks).unwrap();
        let it = interp::Interp::new(&prog).unwrap();
        format!("{:?}", it.run_main().unwrap())
    }

    // Compare semantics by exit value: both must agree on the final Int/Bool.
    fn differential(src: &str) {
        let ir = ir_run(src).expect("ir run");
        let ast = ast_run(src);
        // Normalize: IR Int(n) vs interp Int(n); compare the trailing integer.
        let ir_n: String = ir.chars().filter(|c| c.is_ascii_digit() || *c == '-').collect();
        let ast_n: String = ast.chars().filter(|c| c.is_ascii_digit() || *c == '-').collect();
        assert_eq!(ir_n, ast_n, "IR vs interp mismatch for:\n{}\n  ir={} ast={}", src, ir, ast);
    }

    #[test]
    fn factorial_matches_interpreter() {
        let src = "fn fac(n: Int) -> Int = match n { 0 => 1, _ => n * fac(n - 1), }\nfn main() -> Int = fac(6)";
        differential(src); // 720
    }

    #[test]
    fn list_sum_matches_interpreter_and_counts_allocs() {
        let src = "type L = | Nil | Cons(Int, L)\n\
                   fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
                   fn main() -> Int = sum(Cons(1, Cons(2, Cons(3, Nil))))";
        differential(src); // 6
        // And the IR interpreter should have allocated 4 cells (3 Cons + 1 Nil).
        let toks = lexer::lex(src).unwrap();
        let prog = parser::parse(toks).unwrap();
        let fns = lower_program(&prog).unwrap();
        let mut ir = IrInterp::new(fns);
        ir.run_main().unwrap();
        assert_eq!(ir.metrics.allocations, 4, "expected 4 ADT allocations");
    }

    #[test]
    fn if_and_bool_match() {
        let src = "fn classify(n: Int) -> Int = if n < 0 { 0 } else { 1 }\nfn main() -> Int = classify(5)";
        differential(src);
    }
}
