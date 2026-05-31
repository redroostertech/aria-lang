//! Static type checker for Aria.
//!
//! This is the keystone of the AI-native thesis: the compiler is the model's
//! correctness signal. The checker is sound and *bottom-up* — it synthesizes a
//! type for every expression and checks it against the declared annotations on
//! functions, constructors, and `let` bindings. Because Aria has no generics
//! yet, every type is concrete and no unification is needed.
//!
//! Beyond ordinary type mismatches it enforces the two things that most reduce
//! generated-code bugs:
//!   * exhaustive `match` — every constructor of an ADT must be handled (or a
//!     wildcard provided), so "forgot a case" is a compile error;
//!   * arity/field checks on calls and constructors.

use std::collections::HashMap;

use crate::ast::*;

pub fn show(t: &Ty) -> String {
    match t {
        Ty::Int => "Int".to_string(),
        Ty::Float => "Float".to_string(),
        Ty::Bool => "Bool".to_string(),
        Ty::Str => "String".to_string(),
        Ty::Unit => "Unit".to_string(),
        Ty::Named(n) => n.clone(),
    }
}

type Scope = Vec<HashMap<String, Ty>>;

struct Checker {
    fns: HashMap<String, (Vec<Ty>, Ty)>,
    ctors: HashMap<String, (Vec<Ty>, String)>, // ctor -> (field types, owning type)
    types: HashMap<String, Vec<String>>,        // type -> variant ctor names
}

/// Type-check a whole program. Returns every error found, not just the first.
pub fn check(program: &Program) -> Result<(), Vec<String>> {
    let mut fns: HashMap<String, (Vec<Ty>, Ty)> = HashMap::new();
    let mut ctors: HashMap<String, (Vec<Ty>, String)> = HashMap::new();
    let mut types: HashMap<String, Vec<String>> = HashMap::new();
    let mut errors: Vec<String> = Vec::new();

    // Pass 1: gather declarations.
    for item in &program.items {
        match item {
            Item::Type(t) => {
                let mut variants = Vec::new();
                for v in &t.variants {
                    variants.push(v.name.clone());
                    if ctors
                        .insert(v.name.clone(), (v.fields.clone(), t.name.clone()))
                        .is_some()
                    {
                        errors.push(format!("duplicate constructor `{}`", v.name));
                    }
                }
                if types.insert(t.name.clone(), variants).is_some() {
                    errors.push(format!("duplicate type `{}`", t.name));
                }
            }
            Item::Fn(f) => {
                let params = f.params.iter().map(|p| p.ty.clone()).collect();
                if fns.insert(f.name.clone(), (params, f.ret.clone())).is_some() {
                    errors.push(format!("duplicate function `{}`", f.name));
                }
            }
        }
    }

    // Pass 2: every Named type referenced must be defined.
    let known = |t: &Ty, errs: &mut Vec<String>, ctx: &str| {
        if let Ty::Named(n) = t {
            if !types.contains_key(n) {
                errs.push(format!("{}: unknown type `{}`", ctx, n));
            }
        }
    };
    for item in &program.items {
        match item {
            Item::Fn(f) => {
                for p in &f.params {
                    known(&p.ty, &mut errors, &format!("function `{}` parameter `{}`", f.name, p.name));
                }
                known(&f.ret, &mut errors, &format!("function `{}` return type", f.name));
            }
            Item::Type(t) => {
                for v in &t.variants {
                    for ft in &v.fields {
                        known(ft, &mut errors, &format!("constructor `{}` field", v.name));
                    }
                }
            }
        }
    }

    let checker = Checker { fns, ctors, types };

    // Pass 3: check each function body against its declared return type.
    for item in &program.items {
        if let Item::Fn(f) = item {
            let mut scope: Scope = vec![HashMap::new()];
            for p in &f.params {
                scope[0].insert(p.name.clone(), p.ty.clone());
            }
            match checker.synth(&f.body, &mut scope) {
                Ok(t) => {
                    if t != f.ret {
                        errors.push(format!(
                            "function `{}`: body has type {} but return type is {}",
                            f.name,
                            show(&t),
                            show(&f.ret)
                        ));
                    }
                }
                Err(e) => errors.push(format!("function `{}`: {}", f.name, e)),
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn builtin_sig(name: &str) -> Option<(Vec<Ty>, Ty)> {
    use Ty::*;
    Some(match name {
        "print_int" => (vec![Int], Unit),
        "print_float" => (vec![Float], Unit),
        "print_bool" => (vec![Bool], Unit),
        "print_str" => (vec![Str], Unit),
        "concat" => (vec![Str, Str], Str),
        "int_to_str" => (vec![Int], Str),
        _ => return None,
    })
}

impl Checker {
    fn lookup_var(scope: &Scope, name: &str) -> Option<Ty> {
        for frame in scope.iter().rev() {
            if let Some(t) = frame.get(name) {
                return Some(t.clone());
            }
        }
        None
    }

    fn lookup_fn(&self, name: &str) -> Option<(Vec<Ty>, Ty)> {
        builtin_sig(name).or_else(|| self.fns.get(name).cloned())
    }

    fn synth(&self, e: &Expr, scope: &mut Scope) -> Result<Ty, String> {
        match e {
            Expr::Int(_) => Ok(Ty::Int),
            Expr::Float(_) => Ok(Ty::Float),
            Expr::Bool(_) => Ok(Ty::Bool),
            Expr::Str(_) => Ok(Ty::Str),
            Expr::Unit => Ok(Ty::Unit),

            Expr::Var(name) => Checker::lookup_var(scope, name)
                .ok_or_else(|| format!("unbound variable `{}`", name)),

            Expr::Ctor(name, args) => {
                let (fields, tyname) = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown constructor `{}`", name))?;
                if args.len() != fields.len() {
                    return Err(format!(
                        "constructor `{}` expects {} field(s), got {}",
                        name,
                        fields.len(),
                        args.len()
                    ));
                }
                for (i, (arg, ft)) in args.iter().zip(fields.iter()).enumerate() {
                    let at = self.synth(arg, scope)?;
                    if at != *ft {
                        return Err(format!(
                            "constructor `{}` field {} expects {}, got {}",
                            name,
                            i,
                            show(ft),
                            show(&at)
                        ));
                    }
                }
                Ok(Ty::Named(tyname))
            }

            Expr::Call(name, args) => {
                let (params, ret) = self
                    .lookup_fn(name)
                    .ok_or_else(|| format!("unknown function `{}`", name))?;
                if args.len() != params.len() {
                    return Err(format!(
                        "function `{}` expects {} argument(s), got {}",
                        name,
                        params.len(),
                        args.len()
                    ));
                }
                for (i, (arg, pt)) in args.iter().zip(params.iter()).enumerate() {
                    let at = self.synth(arg, scope)?;
                    if at != *pt {
                        return Err(format!(
                            "function `{}` argument {} expects {}, got {}",
                            name,
                            i,
                            show(pt),
                            show(&at)
                        ));
                    }
                }
                Ok(ret)
            }

            Expr::Unary(op, inner) => {
                let t = self.synth(inner, scope)?;
                match (op, &t) {
                    (UnOp::Neg, Ty::Int) => Ok(Ty::Int),
                    (UnOp::Neg, Ty::Float) => Ok(Ty::Float),
                    (UnOp::Not, Ty::Bool) => Ok(Ty::Bool),
                    _ => Err(format!("cannot apply {:?} to {}", op, show(&t))),
                }
            }

            Expr::Binary(op, lhs, rhs) => {
                let lt = self.synth(lhs, scope)?;
                let rt = self.synth(rhs, scope)?;
                synth_binary(*op, &lt, &rt)
            }

            Expr::If(cond, then, els) => {
                let ct = self.synth(cond, scope)?;
                if ct != Ty::Bool {
                    return Err(format!("`if` condition must be Bool, got {}", show(&ct)));
                }
                let tt = self.synth(then, scope)?;
                let et = self.synth(els, scope)?;
                if tt != et {
                    return Err(format!(
                        "`if` branches have differing types: {} vs {}",
                        show(&tt),
                        show(&et)
                    ));
                }
                Ok(tt)
            }

            Expr::Match(scrut, arms) => self.synth_match(scrut, arms, scope),

            Expr::Block(stmts, last) => {
                scope.push(HashMap::new());
                let result = (|| {
                    for s in stmts {
                        match s {
                            Stmt::Let(name, ann, value) => {
                                let vt = self.synth(value, scope)?;
                                let bound = if let Some(a) = ann {
                                    if *a != vt {
                                        return Err(format!(
                                            "let `{}`: annotated {} but value is {}",
                                            name,
                                            show(a),
                                            show(&vt)
                                        ));
                                    }
                                    a.clone()
                                } else {
                                    vt
                                };
                                scope.last_mut().unwrap().insert(name.clone(), bound);
                            }
                            Stmt::Expr(e) => {
                                self.synth(e, scope)?;
                            }
                        }
                    }
                    self.synth(last, scope)
                })();
                scope.pop();
                result
            }
        }
    }

    fn synth_match(&self, scrut: &Expr, arms: &[Arm], scope: &mut Scope) -> Result<Ty, String> {
        let s = self.synth(scrut, scope)?;
        let mut result: Option<Ty> = None;
        let mut covered_ctors: Vec<String> = Vec::new();
        let (mut saw_true, mut saw_false, mut saw_wild) = (false, false, false);

        for arm in arms {
            match &arm.pat {
                Pattern::Wild | Pattern::Var(_) => saw_wild = true,
                Pattern::Ctor(name, _) => covered_ctors.push(name.clone()),
                Pattern::Bool(true) => saw_true = true,
                Pattern::Bool(false) => saw_false = true,
                Pattern::Int(_) => {}
            }

            let mut binds = HashMap::new();
            self.check_pattern(&arm.pat, &s, &mut binds)?;
            scope.push(binds);
            let bt = self.synth(&arm.body, scope);
            scope.pop();
            let bt = bt?;

            match &result {
                None => result = Some(bt),
                Some(rt) => {
                    if *rt != bt {
                        return Err(format!(
                            "match arms have differing types: {} vs {}",
                            show(rt),
                            show(&bt)
                        ));
                    }
                }
            }
        }

        // Exhaustiveness.
        if !saw_wild {
            match &s {
                Ty::Named(tn) => {
                    if let Some(variants) = self.types.get(tn) {
                        let missing: Vec<String> = variants
                            .iter()
                            .filter(|v| !covered_ctors.contains(v))
                            .cloned()
                            .collect();
                        if !missing.is_empty() {
                            return Err(format!(
                                "non-exhaustive match on {}: missing case(s) {}",
                                tn,
                                missing.join(", ")
                            ));
                        }
                    }
                }
                Ty::Bool => {
                    if !(saw_true && saw_false) {
                        return Err("non-exhaustive match on Bool: handle both true and false (or add `_`)".into());
                    }
                }
                other => {
                    return Err(format!(
                        "non-exhaustive match on {}: add a wildcard `_` arm",
                        show(other)
                    ));
                }
            }
        }

        result.ok_or_else(|| "match needs at least one arm".to_string())
    }

    fn check_pattern(
        &self,
        pat: &Pattern,
        expected: &Ty,
        binds: &mut HashMap<String, Ty>,
    ) -> Result<(), String> {
        match pat {
            Pattern::Wild => Ok(()),
            Pattern::Var(n) => {
                binds.insert(n.clone(), expected.clone());
                Ok(())
            }
            Pattern::Int(_) => {
                if *expected == Ty::Int {
                    Ok(())
                } else {
                    Err(format!("integer pattern matched against {}", show(expected)))
                }
            }
            Pattern::Bool(_) => {
                if *expected == Ty::Bool {
                    Ok(())
                } else {
                    Err(format!("boolean pattern matched against {}", show(expected)))
                }
            }
            Pattern::Ctor(name, subs) => {
                let (fields, tyname) = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown constructor `{}`", name))?;
                match expected {
                    Ty::Named(n) if *n == tyname => {}
                    _ => {
                        return Err(format!(
                            "constructor pattern `{}` (of type {}) matched against {}",
                            name,
                            tyname,
                            show(expected)
                        ))
                    }
                }
                if subs.len() != fields.len() {
                    return Err(format!(
                        "constructor pattern `{}` expects {} field(s), got {}",
                        name,
                        fields.len(),
                        subs.len()
                    ));
                }
                for (sp, ft) in subs.iter().zip(fields.iter()) {
                    self.check_pattern(sp, ft, binds)?;
                }
                Ok(())
            }
        }
    }
}

fn synth_binary(op: BinOp, lt: &Ty, rt: &Ty) -> Result<Ty, String> {
    use BinOp::*;
    let both = |t: &Ty| lt == t && rt == t;
    match op {
        And | Or => {
            if both(&Ty::Bool) {
                Ok(Ty::Bool)
            } else {
                Err(format!("`{:?}` needs Bool operands, got {} and {}", op, show(lt), show(rt)))
            }
        }
        Eq | Ne => {
            if lt == rt {
                Ok(Ty::Bool)
            } else {
                Err(format!("cannot compare {} and {}", show(lt), show(rt)))
            }
        }
        Lt | Le | Gt | Ge => {
            if both(&Ty::Int) || both(&Ty::Float) {
                Ok(Ty::Bool)
            } else {
                Err(format!("`{:?}` needs two Ints or two Floats, got {} and {}", op, show(lt), show(rt)))
            }
        }
        Mod => {
            if both(&Ty::Int) {
                Ok(Ty::Int)
            } else {
                Err(format!("`%` needs Int operands, got {} and {}", show(lt), show(rt)))
            }
        }
        Add | Sub | Mul | Div => {
            if both(&Ty::Int) {
                Ok(Ty::Int)
            } else if both(&Ty::Float) {
                Ok(Ty::Float)
            } else {
                Err(format!("`{:?}` needs two Ints or two Floats, got {} and {}", op, show(lt), show(rt)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lexer, parser};

    fn check_src(src: &str) -> Result<(), Vec<String>> {
        let toks = lexer::lex(src).expect("lex");
        let prog = parser::parse(toks).expect("parse");
        check(&prog)
    }

    #[test]
    fn well_typed_program_ok() {
        let src = r#"
            type Shape = | Circle(Float) | Rect(Float, Float)
            fn area(s: Shape) -> Float =
              match s { Circle(r) => 3.14 * r * r, Rect(w, h) => w * h, }
            fn main() -> Int = { print_float(area(Circle(2.0))); 0 }
        "#;
        assert!(check_src(src).is_ok());
    }

    #[test]
    fn return_type_mismatch_caught() {
        let src = "fn f() -> Int = true";
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("return type")));
    }

    #[test]
    fn non_exhaustive_match_caught() {
        let src = r#"
            type Color = | Red | Green | Blue
            fn name(c: Color) -> Int = match c { Red => 0, Green => 1, }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("non-exhaustive") && e.contains("Blue")));
    }

    #[test]
    fn unbound_variable_caught() {
        let src = "fn f() -> Int = x + 1";
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("unbound variable `x`")));
    }

    #[test]
    fn arg_count_mismatch_caught() {
        let src = r#"
            fn g(a: Int, b: Int) -> Int = a + b
            fn main() -> Int = g(1)
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("expects 2 argument")));
    }

    #[test]
    fn constructor_field_type_mismatch_caught() {
        let src = r#"
            type Box = | B(Int)
            fn main() -> Int = { let x = B(true); 0 }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("field") && e.contains("Int")));
    }

    #[test]
    fn type_mismatch_in_arithmetic_caught() {
        let src = "fn f() -> Int = 1 + true";
        let errs = check_src(src).unwrap_err();
        assert!(!errs.is_empty());
    }
}
