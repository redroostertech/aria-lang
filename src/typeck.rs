//! Static type checker for Aria.
//!
//! This is the keystone of the AI-native thesis: the compiler is the model's
//! correctness signal. The checker is sound and *bottom-up* — it synthesizes a
//! type for every expression and checks it against the declared annotations on
//! functions, constructors, and `let` bindings.
//!
//! Aria now supports parametric polymorphism (generics). Types may contain
//! unification variables (`Ty::Var`). The checker uses a Hindley-Milner-style
//! substitution map keyed on fresh-variable names: at each use of a generic
//! constructor or generic function, its declared type parameters are
//! INSTANTIATED with fresh unification variables, then unification drives the
//! variables to concrete types (or reports a clear mismatch).
//!
//! Beyond ordinary type mismatches it enforces the two things that most reduce
//! generated-code bugs:
//!   * exhaustive `match` — every constructor of an ADT must be handled (or a
//!     wildcard provided), so "forgot a case" is a compile error;
//!   * arity/field checks on calls and constructors.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::ast::*;

/// Render a type for diagnostics, resolving any solved variables first.
pub fn show(t: &Ty) -> String {
    match t {
        Ty::Int => "Int".to_string(),
        Ty::Float => "Float".to_string(),
        Ty::Bool => "Bool".to_string(),
        Ty::Str => "String".to_string(),
        Ty::Unit => "Unit".to_string(),
        Ty::Var(n) => n.clone(),
        Ty::Named(n, args) => {
            if args.is_empty() {
                n.clone()
            } else {
                let inner: Vec<String> = args.iter().map(show).collect();
                format!("{}[{}]", n, inner.join(", "))
            }
        }
    }
}

type Scope = Vec<HashMap<String, Ty>>;

// A constructor's declared signature, retaining the owning type's generic
// parameters so each use can be instantiated with fresh variables.
#[derive(Clone)]
struct CtorSig {
    type_params: Vec<String>, // generic params of the owning type
    fields: Vec<Ty>,          // field types (may mention the params as Ty::Var)
    tyname: String,           // owning type name
}

// A function's declared signature, retaining its generic parameters.
#[derive(Clone)]
struct FnSig {
    type_params: Vec<String>,
    params: Vec<Ty>,
    ret: Ty,
}

struct Checker {
    fns: HashMap<String, FnSig>,
    ctors: HashMap<String, CtorSig>,
    types: HashMap<String, (Vec<String>, Vec<String>)>, // type -> (params, variant ctor names)
    // Union-find / substitution for unification variables, plus a fresh counter.
    subst: RefCell<HashMap<String, Ty>>,
    counter: RefCell<u64>,
}

/// Type-check a whole program. Returns every error found, not just the first.
pub fn check(program: &Program) -> Result<(), Vec<String>> {
    let mut fns: HashMap<String, FnSig> = HashMap::new();
    let mut ctors: HashMap<String, CtorSig> = HashMap::new();
    let mut types: HashMap<String, (Vec<String>, Vec<String>)> = HashMap::new();
    let mut errors: Vec<String> = Vec::new();

    // Pass 1: gather declarations.
    for item in &program.items {
        match item {
            Item::Type(t) => {
                let mut variants = Vec::new();
                for v in &t.variants {
                    variants.push(v.name.clone());
                    if ctors
                        .insert(
                            v.name.clone(),
                            CtorSig {
                                type_params: t.params.clone(),
                                fields: v.fields.clone(),
                                tyname: t.name.clone(),
                            },
                        )
                        .is_some()
                    {
                        errors.push(format!("duplicate constructor `{}`", v.name));
                    }
                }
                if types
                    .insert(t.name.clone(), (t.params.clone(), variants))
                    .is_some()
                {
                    errors.push(format!("duplicate type `{}`", t.name));
                }
            }
            Item::Fn(f) => {
                let params = f.params.iter().map(|p| p.ty.clone()).collect();
                if fns
                    .insert(
                        f.name.clone(),
                        FnSig {
                            type_params: f.type_params.clone(),
                            params,
                            ret: f.ret.clone(),
                        },
                    )
                    .is_some()
                {
                    errors.push(format!("duplicate function `{}`", f.name));
                }
            }
        }
    }

    // Pass 2: every Named type referenced must be defined, with the right arity,
    // and its arguments must in turn be valid. Type variables in scope are fine.
    fn known(
        t: &Ty,
        types: &HashMap<String, (Vec<String>, Vec<String>)>,
        params: &[String],
        errs: &mut Vec<String>,
        ctx: &str,
    ) {
        match t {
            Ty::Named(n, args) if BUILTIN_TYPES.contains(&n.as_str()) => {
                // Built-in nullary types (e.g. `Tensor`) take no arguments.
                if !args.is_empty() {
                    errs.push(format!(
                        "{}: built-in type `{}` takes no type arguments, got {}",
                        ctx,
                        n,
                        args.len()
                    ));
                }
            }
            Ty::Named(n, args) => {
                match types.get(n) {
                    None => errs.push(format!("{}: unknown type `{}`", ctx, n)),
                    Some((decl_params, _)) => {
                        if decl_params.len() != args.len() {
                            errs.push(format!(
                                "{}: type `{}` expects {} type argument(s), got {}",
                                ctx,
                                n,
                                decl_params.len(),
                                args.len()
                            ));
                        }
                    }
                }
                for a in args {
                    known(a, types, params, errs, ctx);
                }
            }
            Ty::Var(_) => {} // a declared generic parameter; fine
            _ => {}
        }
    }
    for item in &program.items {
        match item {
            Item::Fn(f) => {
                for p in &f.params {
                    known(
                        &p.ty,
                        &types,
                        &f.type_params,
                        &mut errors,
                        &format!("function `{}` parameter `{}`", f.name, p.name),
                    );
                }
                known(
                    &f.ret,
                    &types,
                    &f.type_params,
                    &mut errors,
                    &format!("function `{}` return type", f.name),
                );
            }
            Item::Type(t) => {
                for v in &t.variants {
                    for ft in &v.fields {
                        known(
                            ft,
                            &types,
                            &t.params,
                            &mut errors,
                            &format!("constructor `{}` field", v.name),
                        );
                    }
                }
            }
        }
    }

    let checker = Checker {
        fns,
        ctors,
        types,
        subst: RefCell::new(HashMap::new()),
        counter: RefCell::new(0),
    };

    // Pass 3: check each function body against its declared return type. Each
    // function gets a fresh unification context so leftover variables from one
    // body cannot leak into another.
    for item in &program.items {
        if let Item::Fn(f) = item {
            checker.subst.borrow_mut().clear();
            let mut scope: Scope = vec![HashMap::new()];
            for p in &f.params {
                scope[0].insert(p.name.clone(), p.ty.clone());
            }
            match checker.synth(&f.body, &mut scope) {
                Ok(t) => {
                    if let Err(e) = checker.unify(&t, &f.ret) {
                        errors.push(format!(
                            "function `{}`: body has type {} but return type is {} ({})",
                            f.name,
                            show(&checker.resolve(&t)),
                            show(&checker.resolve(&f.ret)),
                            e
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

// Built-in nullary types that need no user declaration but must pass the
// "unknown type" check in pass 2. `Tensor` is the opaque AI-runtime handle.
const BUILTIN_TYPES: &[&str] = &["Tensor"];

fn builtin_sig(name: &str) -> Option<(Vec<Ty>, Ty)> {
    use Ty::*;
    // The opaque tensor handle, shared across all tensor builtins below.
    let tensor = || Named("Tensor".to_string(), vec![]);
    Some(match name {
        "print_int" => (vec![Int], Unit),
        "print_float" => (vec![Float], Unit),
        "print_bool" => (vec![Bool], Unit),
        "print_str" => (vec![Str], Unit),
        "concat" => (vec![Str, Str], Str),
        "int_to_str" => (vec![Int], Str),

        // ---- AI runtime primitives ---------------------------------------
        // Tensors are opaque values built and queried purely via builtins.
        "tensor_zeros" => (vec![Int, Int], tensor()),
        "tensor_set" => (vec![tensor(), Int, Int, Float], tensor()),
        "tensor_get" => (vec![tensor(), Int, Int], Float),
        "tensor_rows" => (vec![tensor()], Int),
        "tensor_cols" => (vec![tensor()], Int),
        "matmul" => (vec![tensor(), tensor()], tensor()),
        "transpose" => (vec![tensor()], tensor()),
        "softmax" => (vec![tensor()], tensor()),
        "relu" => (vec![tensor()], tensor()),
        "embed_similarity" => (vec![Str, Str], Float),
        "compressed_size" => (vec![Str], Int),
        "neural_bits_per_byte" => (vec![Str], Float),
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

    // ---- unification primitives -----------------------------------------

    // Allocate a fresh unification variable.
    fn fresh(&self) -> Ty {
        let mut c = self.counter.borrow_mut();
        let id = *c;
        *c += 1;
        Ty::Var(format!("?{}", id))
    }

    // Instantiate a list of declared parameter names with fresh variables,
    // returning the substitution to apply to a signature.
    fn instantiate_map(&self, params: &[String]) -> HashMap<String, Ty> {
        params.iter().map(|p| (p.clone(), self.fresh())).collect()
    }

    // Substitute declared parameter names (as `Ty::Var`) using `map`.
    fn apply_map(ty: &Ty, map: &HashMap<String, Ty>) -> Ty {
        match ty {
            Ty::Var(n) => map.get(n).cloned().unwrap_or_else(|| ty.clone()),
            Ty::Named(n, args) => {
                Ty::Named(n.clone(), args.iter().map(|a| Self::apply_map(a, map)).collect())
            }
            other => other.clone(),
        }
    }

    // Follow the substitution chain shallowly (one level) for a variable.
    fn prune(&self, ty: &Ty) -> Ty {
        if let Ty::Var(n) = ty {
            if let Some(bound) = self.subst.borrow().get(n).cloned() {
                return self.prune(&bound);
            }
        }
        ty.clone()
    }

    // Fully resolve a type by substituting all solved variables, for display.
    fn resolve(&self, ty: &Ty) -> Ty {
        let t = self.prune(ty);
        match t {
            Ty::Named(n, args) => {
                Ty::Named(n, args.iter().map(|a| self.resolve(a)).collect())
            }
            other => other,
        }
    }

    fn occurs(&self, var: &str, ty: &Ty) -> bool {
        match self.prune(ty) {
            Ty::Var(n) => n == var,
            Ty::Named(_, args) => args.iter().any(|a| self.occurs(var, a)),
            _ => false,
        }
    }

    // Unify two types, recording variable bindings in the substitution.
    fn unify(&self, a: &Ty, b: &Ty) -> Result<(), String> {
        let a = self.prune(a);
        let b = self.prune(b);
        match (&a, &b) {
            (Ty::Int, Ty::Int)
            | (Ty::Float, Ty::Float)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Str, Ty::Str)
            | (Ty::Unit, Ty::Unit) => Ok(()),
            (Ty::Var(x), Ty::Var(y)) if x == y => Ok(()),
            (Ty::Var(x), _) => {
                if self.occurs(x, &b) {
                    return Err(format!("infinite type: {} occurs in {}", x, show(&self.resolve(&b))));
                }
                self.subst.borrow_mut().insert(x.clone(), b.clone());
                Ok(())
            }
            (_, Ty::Var(y)) => {
                if self.occurs(y, &a) {
                    return Err(format!("infinite type: {} occurs in {}", y, show(&self.resolve(&a))));
                }
                self.subst.borrow_mut().insert(y.clone(), a.clone());
                Ok(())
            }
            (Ty::Named(n1, a1), Ty::Named(n2, a2)) if n1 == n2 && a1.len() == a2.len() => {
                for (x, y) in a1.iter().zip(a2.iter()) {
                    self.unify(x, y)?;
                }
                Ok(())
            }
            _ => Err(format!(
                "expected {}, found {}",
                show(&self.resolve(&a)),
                show(&self.resolve(&b))
            )),
        }
    }

    // ---- synthesis -------------------------------------------------------

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
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown constructor `{}`", name))?;
                if args.len() != sig.fields.len() {
                    return Err(format!(
                        "constructor `{}` expects {} field(s), got {}",
                        name,
                        sig.fields.len(),
                        args.len()
                    ));
                }
                // Instantiate the owning type's parameters with fresh variables.
                let map = self.instantiate_map(&sig.type_params);
                for (i, (arg, ft)) in args.iter().zip(sig.fields.iter()).enumerate() {
                    let at = self.synth(arg, scope)?;
                    let expected = Self::apply_map(ft, &map);
                    self.unify(&expected, &at).map_err(|e| {
                        format!("constructor `{}` field {}: {}", name, i, e)
                    })?;
                }
                let type_args: Vec<Ty> = sig
                    .type_params
                    .iter()
                    .map(|p| map.get(p).cloned().unwrap())
                    .collect();
                Ok(Ty::Named(sig.tyname.clone(), type_args))
            }

            Expr::Call(name, args) => {
                let (params, ret, type_params) =
                    if let Some((p, r)) = builtin_sig(name) {
                        (p, r, Vec::new())
                    } else if let Some(sig) = self.fns.get(name).cloned() {
                        (sig.params, sig.ret, sig.type_params)
                    } else {
                        return Err(format!("unknown function `{}`", name));
                    };
                if args.len() != params.len() {
                    return Err(format!(
                        "function `{}` expects {} argument(s), got {}",
                        name,
                        params.len(),
                        args.len()
                    ));
                }
                // Instantiate the function's generic parameters with fresh vars.
                let map = self.instantiate_map(&type_params);
                for (i, (arg, pt)) in args.iter().zip(params.iter()).enumerate() {
                    let at = self.synth(arg, scope)?;
                    let expected = Self::apply_map(pt, &map);
                    self.unify(&expected, &at).map_err(|e| {
                        format!("function `{}` argument {}: {}", name, i, e)
                    })?;
                }
                Ok(Self::apply_map(&ret, &map))
            }

            Expr::Unary(op, inner) => {
                let t = self.prune(&self.synth(inner, scope)?);
                match (op, &t) {
                    (UnOp::Neg, Ty::Int) => Ok(Ty::Int),
                    (UnOp::Neg, Ty::Float) => Ok(Ty::Float),
                    (UnOp::Not, Ty::Bool) => Ok(Ty::Bool),
                    _ => Err(format!("cannot apply {:?} to {}", op, show(&self.resolve(&t)))),
                }
            }

            Expr::Binary(op, lhs, rhs) => {
                let lt = self.synth(lhs, scope)?;
                let rt = self.synth(rhs, scope)?;
                self.synth_binary(*op, &lt, &rt)
            }

            Expr::If(cond, then, els) => {
                let ct = self.synth(cond, scope)?;
                self.unify(&Ty::Bool, &ct)
                    .map_err(|_| format!("`if` condition must be Bool, got {}", show(&self.resolve(&ct))))?;
                let tt = self.synth(then, scope)?;
                let et = self.synth(els, scope)?;
                self.unify(&tt, &et).map_err(|_| {
                    format!(
                        "`if` branches have differing types: {} vs {}",
                        show(&self.resolve(&tt)),
                        show(&self.resolve(&et))
                    )
                })?;
                Ok(self.resolve(&tt))
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
                                    self.unify(a, &vt).map_err(|_| {
                                        format!(
                                            "let `{}`: annotated {} but value is {}",
                                            name,
                                            show(&self.resolve(a)),
                                            show(&self.resolve(&vt))
                                        )
                                    })?;
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
                    self.unify(rt, &bt).map_err(|_| {
                        format!(
                            "match arms have differing types: {} vs {}",
                            show(&self.resolve(rt)),
                            show(&self.resolve(&bt))
                        )
                    })?;
                }
            }
        }

        // Exhaustiveness. Keys on constructor names, so generics are unaffected.
        if !saw_wild {
            match self.prune(&s) {
                Ty::Named(tn, _) => {
                    if let Some((_, variants)) = self.types.get(&tn) {
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
                        show(&self.resolve(&other))
                    ));
                }
            }
        }

        result
            .map(|t| self.resolve(&t))
            .ok_or_else(|| "match needs at least one arm".to_string())
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
            Pattern::Int(_) => self
                .unify(&Ty::Int, expected)
                .map_err(|_| format!("integer pattern matched against {}", show(&self.resolve(expected)))),
            Pattern::Bool(_) => self
                .unify(&Ty::Bool, expected)
                .map_err(|_| format!("boolean pattern matched against {}", show(&self.resolve(expected)))),
            Pattern::Ctor(name, subs) => {
                let sig = self
                    .ctors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown constructor `{}`", name))?;
                // Instantiate the owning type's params, then unify the scrutinee
                // type with `Name[fresh...]` so field patterns get refined types.
                let map = self.instantiate_map(&sig.type_params);
                let type_args: Vec<Ty> = sig
                    .type_params
                    .iter()
                    .map(|p| map.get(p).cloned().unwrap())
                    .collect();
                let owner = Ty::Named(sig.tyname.clone(), type_args);
                self.unify(&owner, expected).map_err(|_| {
                    format!(
                        "constructor pattern `{}` (of type {}) matched against {}",
                        name,
                        sig.tyname,
                        show(&self.resolve(expected))
                    )
                })?;
                if subs.len() != sig.fields.len() {
                    return Err(format!(
                        "constructor pattern `{}` expects {} field(s), got {}",
                        name,
                        sig.fields.len(),
                        subs.len()
                    ));
                }
                for (sp, ft) in subs.iter().zip(sig.fields.iter()) {
                    let fty = Self::apply_map(ft, &map);
                    self.check_pattern(sp, &fty, binds)?;
                }
                Ok(())
            }
        }
    }

    fn synth_binary(&self, op: BinOp, lt: &Ty, rt: &Ty) -> Result<Ty, String> {
        use BinOp::*;
        let lt = self.prune(lt);
        let rt = self.prune(rt);
        let both = |t: &Ty| lt == *t && rt == *t;
        match op {
            And | Or => {
                if both(&Ty::Bool) {
                    Ok(Ty::Bool)
                } else {
                    Err(format!("`{:?}` needs Bool operands, got {} and {}", op, show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
            }
            Eq | Ne => {
                if self.unify(&lt, &rt).is_ok() {
                    Ok(Ty::Bool)
                } else {
                    Err(format!("cannot compare {} and {}", show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
            }
            Lt | Le | Gt | Ge => {
                if both(&Ty::Int) || both(&Ty::Float) {
                    Ok(Ty::Bool)
                } else {
                    Err(format!("`{:?}` needs two Ints or two Floats, got {} and {}", op, show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
            }
            Mod => {
                if both(&Ty::Int) {
                    Ok(Ty::Int)
                } else {
                    Err(format!("`%` needs Int operands, got {} and {}", show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
            }
            Add | Sub | Mul | Div => {
                if both(&Ty::Int) {
                    Ok(Ty::Int)
                } else if both(&Ty::Float) {
                    Ok(Ty::Float)
                } else {
                    Err(format!("`{:?}` needs two Ints or two Floats, got {} and {}", op, show(&self.resolve(&lt)), show(&self.resolve(&rt))))
                }
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

    // ---- generics --------------------------------------------------------

    #[test]
    fn generic_adt_instantiates() {
        // A generic List instantiated at Int should check, and main returns Int.
        let src = r#"
            type List[T] = | Nil | Cons(T, List[T])
            fn main() -> Int = { let xs = Cons(1, Cons(2, Nil)); 0 }
        "#;
        assert!(check_src(src).is_ok());
    }

    #[test]
    fn generic_type_argument_mismatch_caught() {
        // Cons(1, ...) fixes T = Int, so Cons(true, Nil) must be rejected.
        let src = r#"
            type List[T] = | Nil | Cons(T, List[T])
            fn main() -> Int = { let xs = Cons(1, Cons(true, Nil)); 0 }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("expected") && e.contains("found")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn generic_function_checks() {
        let src = r#"
            type Option[T] = | None | Some(T)
            fn is_some[T](o: Option[T]) -> Bool =
              match o { None => false, Some(_) => true, }
            fn main() -> Bool = is_some(Some(5))
        "#;
        assert!(check_src(src).is_ok());
    }

    #[test]
    fn exhaustiveness_fires_on_generic_type() {
        let src = r#"
            type Option[T] = | None | Some(T)
            fn unwrap_or(o: Option[Int]) -> Int = match o { Some(x) => x, }
        "#;
        let errs = check_src(src).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("non-exhaustive") && e.contains("None")),
            "got: {:?}",
            errs
        );
    }

    #[test]
    fn generic_return_type_inferred() {
        // A generic function returning Option[T] used at Int.
        let src = r#"
            type Option[T] = | None | Some(T)
            fn wrap[T](x: T) -> Option[T] = Some(x)
            fn main() -> Int = { let o = wrap(7); 0 }
        "#;
        assert!(check_src(src).is_ok());
    }
}
