//! Precise reference-count insertion (Perceus-style) over the ANF IR.
//!
//! Stage 2 of the memory model. Given lowered IR with no reference-count
//! operations, this pass inserts `dup`/`drop` so that — with NO programmer
//! annotations — every heap cell is freed exactly once: dropped at its last
//! use, dup'd at non-last uses, and released in the branch that does not use it.
//! Correctness is validated at runtime by the IR interpreter's garbage-free
//! check (allocations == frees for value-returning programs; no use-after-free).
//!
//! Only ADT cells are reference-counted; `dup`/`drop` on unboxed values are
//! runtime no-ops, so the pass can treat all variables uniformly without types.
//! The analysis relies on the invariant that, on entry to `rc(e, live)`, the set
//! of owned variables is exactly `fv(e) ∪ live`.

use std::collections::{HashMap, HashSet};

use crate::ir::{Atom, Bind, IArm, IExpr, IFn};

/// Insert reference-count operations into every function body.
pub fn insert_rc(fns: &HashMap<String, IFn>) -> HashMap<String, IFn> {
    fns.iter()
        .map(|(name, f)| {
            let mut body = rc(&f.body, &HashSet::new());
            // Parameters are owned on entry; drop any the body never uses.
            let body_fv = fv_expr(&f.body);
            for p in f.params.iter().rev() {
                if !body_fv.contains(p) {
                    body = IExpr::Drop(p.clone(), Box::new(body));
                }
            }
            (name.clone(), IFn { params: f.params.clone(), body })
        })
        .collect()
}

// ---- free-variable analysis ---------------------------------------------

fn add_atom(a: &Atom, acc: &mut HashSet<String>) {
    if let Atom::Var(v) = a {
        acc.insert(v.clone());
    }
}

fn fv_expr(e: &IExpr) -> HashSet<String> {
    let mut acc = HashSet::new();
    collect_fv(e, &mut acc);
    acc
}

fn collect_fv(e: &IExpr, acc: &mut HashSet<String>) {
    match e {
        IExpr::Ret(a) => add_atom(a, acc),
        IExpr::Dup(_, b) | IExpr::Drop(_, b) => collect_fv(b, acc),
        IExpr::Let(x, bind, body) => {
            collect_fv_bind(bind, acc);
            let mut b = HashSet::new();
            collect_fv(body, &mut b);
            b.remove(x);
            acc.extend(b);
        }
    }
}

fn collect_fv_bind(bind: &Bind, acc: &mut HashSet<String>) {
    match bind {
        Bind::Atom(a) => add_atom(a, acc),
        Bind::Prim(_, a, b) => {
            add_atom(a, acc);
            add_atom(b, acc);
        }
        Bind::Unary(_, a) => add_atom(a, acc),
        Bind::Ctor(_, atoms) | Bind::Call(_, atoms) => {
            for a in atoms {
                add_atom(a, acc);
            }
        }
        Bind::If(c, t, e) => {
            add_atom(c, acc);
            collect_fv(t, acc);
            collect_fv(e, acc);
        }
        Bind::Match(s, arms) => {
            add_atom(s, acc);
            for arm in arms {
                let mut a = HashSet::new();
                collect_fv(&arm.body, &mut a);
                for b in &arm.binders {
                    a.remove(b);
                }
                acc.extend(a);
            }
        }
    }
}

/// Operand variables whose ownership a bind transfers (moves a heap reference):
/// aliasing, constructor fields, and call arguments. Prim/Unary operate on
/// unboxed values and `If`/`Match` are handled structurally, so none here.
fn consumed(bind: &Bind) -> Vec<String> {
    match bind {
        Bind::Atom(Atom::Var(v)) => vec![v.clone()],
        Bind::Ctor(_, atoms) | Bind::Call(_, atoms) => atoms
            .iter()
            .filter_map(|a| if let Atom::Var(v) = a { Some(v.clone()) } else { None })
            .collect(),
        _ => vec![],
    }
}

// ---- the transform -------------------------------------------------------

fn maybe_drop(x: &str, dead: bool, body: IExpr) -> IExpr {
    if dead {
        IExpr::Drop(x.to_string(), Box::new(body))
    } else {
        body
    }
}

fn with_drops(vars: Vec<String>, mut e: IExpr) -> IExpr {
    for v in vars {
        e = IExpr::Drop(v, Box::new(e));
    }
    e
}

/// `a \ (b ∪ c)`.
fn diff(a: &HashSet<String>, b: &HashSet<String>, c: &HashSet<String>) -> Vec<String> {
    a.iter()
        .filter(|v| !b.contains(*v) && !c.contains(*v))
        .cloned()
        .collect()
}

/// `live` = variables the continuation after `e` will consume (owned, must
/// survive). Invariant: owned-on-entry = `fv(e) ∪ live`.
fn rc(e: &IExpr, live: &HashSet<String>) -> IExpr {
    match e {
        IExpr::Ret(a) => {
            // The result escapes (moved to the caller / this let's binder); vars
            // in `live` pass through. If the returned var is also needed later,
            // dup it.
            if let Atom::Var(v) = a {
                if live.contains(v) {
                    return IExpr::Dup(v.clone(), Box::new(IExpr::Ret(a.clone())));
                }
            }
            IExpr::Ret(a.clone())
        }

        IExpr::Let(x, bind, body) => {
            // Variables needed during/after the body (i.e. after this bind).
            let mut after = fv_expr(body);
            after.remove(x);
            for v in live {
                after.insert(v.clone());
            }
            let body2 = rc(body, live);
            let x_dead = !fv_expr(body).contains(x) && !live.contains(x);

            match bind {
                Bind::If(c, then, els) => {
                    let fvt = fv_expr(then);
                    let fve = fv_expr(els);
                    // Drop in the branch that doesn't use a var (and won't need
                    // it after), to keep ownership balanced across branches.
                    let then2 = with_drops(diff(&fve, &fvt, &after), rc(then, &after));
                    let els2 = with_drops(diff(&fvt, &fve, &after), rc(els, &after));
                    let nb = Bind::If(c.clone(), Box::new(then2), Box::new(els2));
                    IExpr::Let(x.clone(), nb, Box::new(maybe_drop(x, x_dead, body2)))
                }

                Bind::Match(scrut, arms) => {
                    let sname = if let Atom::Var(s) = scrut { Some(s.clone()) } else { None };
                    let arm_fv: Vec<HashSet<String>> = arms
                        .iter()
                        .map(|arm| {
                            let mut a = fv_expr(&arm.body);
                            for b in &arm.binders {
                                a.remove(b);
                            }
                            a
                        })
                        .collect();
                    let mut new_arms = Vec::new();
                    for (i, arm) in arms.iter().enumerate() {
                        let mut others = HashSet::new();
                        for (j, fvj) in arm_fv.iter().enumerate() {
                            if j != i {
                                others.extend(fvj.iter().cloned());
                            }
                        }
                        let this = &arm_fv[i];
                        let arm_body_fv = fv_expr(&arm.body);
                        // Outer vars used only in OTHER arms (and not after, not
                        // the scrutinee) must be dropped at this arm's entry.
                        let drop_set: Vec<String> = others
                            .into_iter()
                            .filter(|v| {
                                !this.contains(v)
                                    && !after.contains(v)
                                    && sname.as_ref() != Some(v)
                            })
                            .collect();
                        let mut ab = rc(&arm.body, &after);
                        ab = with_drops(drop_set, ab);
                        match &arm.ctor {
                            Some(_) => {
                                // Release the matched cell (its fields were dup'd
                                // for the binders that use them).
                                if let Some(s) = &sname {
                                    ab = IExpr::Drop(s.clone(), Box::new(ab));
                                }
                                for b in &arm.binders {
                                    if arm_body_fv.contains(b) {
                                        ab = IExpr::Dup(b.clone(), Box::new(ab));
                                    }
                                }
                            }
                            None => {
                                // Catch-all binder aliases the scrutinee; drop it
                                // (releasing the cell) only if the body never uses it.
                                if let Some(b) = arm.binders.first() {
                                    if !arm_body_fv.contains(b) {
                                        ab = IExpr::Drop(b.clone(), Box::new(ab));
                                    }
                                }
                            }
                        }
                        new_arms.push(IArm {
                            ctor: arm.ctor.clone(),
                            binders: arm.binders.clone(),
                            body: ab,
                        });
                    }
                    let nb = Bind::Match(scrut.clone(), new_arms);
                    IExpr::Let(x.clone(), nb, Box::new(maybe_drop(x, x_dead, body2)))
                }

                _ => {
                    // Simple bind: Atom / Prim / Unary / Ctor / Call.
                    let inner = maybe_drop(x, x_dead, body2);
                    let mut e2 = IExpr::Let(x.clone(), bind.clone(), Box::new(inner));
                    // Dup each consumed operand once per occurrence beyond the
                    // one owned reference, plus one more if it must survive.
                    let mut counts: HashMap<String, usize> = HashMap::new();
                    for v in consumed(bind) {
                        *counts.entry(v).or_insert(0) += 1;
                    }
                    for (v, k) in counts {
                        let need = k + if after.contains(&v) { 1 } else { 0 };
                        for _ in 0..need.saturating_sub(1) {
                            e2 = IExpr::Dup(v.clone(), Box::new(e2));
                        }
                    }
                    e2
                }
            }
        }

        IExpr::Dup(..) | IExpr::Drop(..) => e.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interp, ir, lexer, parser, typeck};

    /// Lower, insert RC, run, and return (result string, allocations, frees).
    fn run_rc(src: &str) -> (String, usize, usize) {
        let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
        typeck::check(&prog).expect("typeck");
        let fns = insert_rc(&ir::lower_program(&prog).unwrap());
        let mut runner = ir::IrInterp::new(fns);
        let v = runner.run_main().expect("ir run");
        (runner.render(&v), runner.metrics.allocations, runner.metrics.frees)
    }

    fn ast_result(src: &str) -> String {
        let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
        interp::Interp::new(&prog).unwrap().run_main().unwrap().display()
    }

    /// For an Int/Bool-returning program, the RC pass must be garbage-free:
    /// every allocation is freed, and the result matches the interpreter.
    fn assert_garbage_free(src: &str) {
        let (ir_res, allocs, frees) = run_rc(src);
        assert_eq!(ir_res, ast_result(src), "value mismatch:\n{}", src);
        assert_eq!(allocs, frees, "leak: {} allocations, {} frees in:\n{}", allocs, frees, src);
    }

    #[test]
    fn list_sum_is_garbage_free() {
        assert_garbage_free(
            "type L = | Nil | Cons(Int, L)\n\
             fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
             fn main() -> Int = sum(Cons(1, Cons(2, Cons(3, Nil))))",
        );
    }

    #[test]
    fn map_then_sum_is_garbage_free() {
        // Builds a list, maps +1 (fresh list), sums it: all cells must be freed.
        assert_garbage_free(
            "type L = | Nil | Cons(Int, L)\n\
             fn range(n: Int, acc: L) -> L = if n == 0 { acc } else { range(n - 1, Cons(n, acc)) }\n\
             fn inc(xs: L) -> L = match xs { Nil => Nil, Cons(h, r) => Cons(h + 1, inc(r)), }\n\
             fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
             fn main() -> Int = sum(inc(range(50, Nil)))",
        );
    }

    #[test]
    fn shared_reference_is_garbage_free() {
        // `xs` used twice (shared) -> requires a dup; both uses + the original
        // must net to zero frees-vs-allocations.
        assert_garbage_free(
            "type L = | Nil | Cons(Int, L)\n\
             fn len(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => 1 + len(r), }\n\
             fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }\n\
             fn use_twice(xs: L) -> Int = sum(xs) + len(xs)\n\
             fn main() -> Int = use_twice(Cons(10, Cons(20, Nil)))",
        );
    }

    #[test]
    fn unused_value_is_freed() {
        // `tmp` is built then never used -> must be dropped, not leaked.
        assert_garbage_free(
            "type L = | Nil | Cons(Int, L)\n\
             fn main() -> Int = { let tmp = Cons(1, Cons(2, Nil)); 7 }",
        );
    }

    #[test]
    fn branch_only_uses_value_in_one_arm() {
        // The list is consumed in one branch and dropped in the other.
        assert_garbage_free(
            "type L = | Nil | Cons(Int, L)\n\
             fn pick(b: Bool, xs: L) -> Int = if b { match xs { Nil => 0, Cons(h, r) => h, } } else { 99 }\n\
             fn main() -> Int = pick(false, Cons(5, Nil))",
        );
    }

    #[test]
    fn factorial_no_heap_is_garbage_free() {
        assert_garbage_free(
            "fn fac(n: Int) -> Int = match n { 0 => 1, _ => n * fac(n - 1), }\nfn main() -> Int = fac(6)",
        );
    }
}
