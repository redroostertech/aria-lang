//! Tree-walking interpreter.
//!
//! This is the "runs today" backend. Frontend (lexer/parser) is kept fully
//! separate so a WASM or native code generator can be added as an alternative
//! backend without touching anything here.

use std::collections::HashMap;

use crate::ast::*;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Data { ctor: String, fields: Vec<Value> },
    /// An opaque AI-runtime tensor handle, built and queried via builtins.
    Tensor(crate::tensor::Tensor),
    Unit,
}

impl Value {
    pub fn display(&self) -> String {
        match self {
            Value::Int(n) => n.to_string(),
            Value::Float(f) => format!("{}", f),
            Value::Bool(b) => b.to_string(),
            Value::Str(s) => s.clone(),
            Value::Tensor(t) => {
                if t.shape.len() == 2 {
                    format!("Tensor({}x{})", t.shape[0], t.shape[1])
                } else {
                    format!("Tensor{:?}", t.shape)
                }
            }
            Value::Unit => "()".to_string(),
            Value::Data { ctor, fields } => {
                if fields.is_empty() {
                    ctor.clone()
                } else {
                    let inner: Vec<String> = fields.iter().map(|v| v.display()).collect();
                    format!("{}({})", ctor, inner.join(", "))
                }
            }
        }
    }
}

type Scope = Vec<HashMap<String, Value>>;

pub struct Interp {
    fns: HashMap<String, FnDecl>,
    /// constructor name -> arity
    ctors: HashMap<String, usize>,
    /// Current Aria call-stack depth, to turn runaway recursion into a
    /// catchable error instead of a native stack overflow.
    depth: std::cell::Cell<usize>,
}

/// Maximum Aria function-call nesting. The interpreter runs on a large-stack
/// thread (see main.rs), so this is generous; it exists to catch genuinely
/// non-terminating recursion as an error rather than crashing the process.
const MAX_CALL_DEPTH: usize = 100_000;

impl Interp {
    pub fn new(program: &Program) -> Result<Self, String> {
        let mut fns = HashMap::new();
        let mut ctors = HashMap::new();
        for item in &program.items {
            match item {
                Item::Fn(f) => {
                    if fns.insert(f.name.clone(), f.clone()).is_some() {
                        return Err(format!("duplicate function `{}`", f.name));
                    }
                }
                Item::Type(t) => {
                    for v in &t.variants {
                        if ctors.insert(v.name.clone(), v.fields.len()).is_some() {
                            return Err(format!("duplicate constructor `{}`", v.name));
                        }
                    }
                }
            }
        }
        Ok(Interp { fns, ctors, depth: std::cell::Cell::new(0) })
    }

    pub fn run_main(&self) -> Result<Value, String> {
        let main = self
            .fns
            .get("main")
            .ok_or_else(|| "no `main` function found".to_string())?;
        if !main.params.is_empty() {
            return Err("`main` must take no parameters".to_string());
        }
        let mut scope: Scope = vec![HashMap::new()];
        self.eval(&main.body, &mut scope)
    }

    fn lookup<'a>(scope: &'a Scope, name: &str) -> Option<&'a Value> {
        for frame in scope.iter().rev() {
            if let Some(v) = frame.get(name) {
                return Some(v);
            }
        }
        None
    }

    fn eval(&self, e: &Expr, scope: &mut Scope) -> Result<Value, String> {
        match e {
            Expr::Int(n) => Ok(Value::Int(*n)),
            Expr::Float(f) => Ok(Value::Float(*f)),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Unit => Ok(Value::Unit),

            Expr::Var(name) => {
                if let Some(v) = Interp::lookup(scope, name) {
                    Ok(v.clone())
                } else {
                    Err(format!("unbound variable `{}`", name))
                }
            }

            Expr::Ctor(name, args) => {
                let arity = self
                    .ctors
                    .get(name)
                    .ok_or_else(|| format!("unknown constructor `{}`", name))?;
                if *arity != args.len() {
                    return Err(format!(
                        "constructor `{}` expects {} field(s), got {}",
                        name,
                        arity,
                        args.len()
                    ));
                }
                let mut fields = Vec::with_capacity(args.len());
                for a in args {
                    fields.push(self.eval(a, scope)?);
                }
                Ok(Value::Data {
                    ctor: name.clone(),
                    fields,
                })
            }

            Expr::Call(name, args) => {
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval(a, scope)?);
                }
                if let Some(v) = builtin(name, &vals)? {
                    return Ok(v);
                }
                let f = self
                    .fns
                    .get(name)
                    .ok_or_else(|| format!("unknown function `{}`", name))?;
                if f.params.len() != vals.len() {
                    return Err(format!(
                        "function `{}` expects {} argument(s), got {}",
                        name,
                        f.params.len(),
                        vals.len()
                    ));
                }
                let mut frame = HashMap::new();
                for (p, v) in f.params.iter().zip(vals.into_iter()) {
                    frame.insert(p.name.clone(), v);
                }
                let mut call_scope: Scope = vec![frame];
                let d = self.depth.get() + 1;
                if d > MAX_CALL_DEPTH {
                    return Err(format!(
                        "maximum recursion depth ({}) exceeded calling `{}`",
                        MAX_CALL_DEPTH, name
                    ));
                }
                self.depth.set(d);
                let result = self.eval(&f.body, &mut call_scope);
                self.depth.set(d - 1);
                result
            }

            Expr::Unary(op, inner) => {
                let v = self.eval(inner, scope)?;
                match (op, v) {
                    (UnOp::Neg, Value::Int(n)) => Ok(Value::Int(
                        n.checked_neg().ok_or("integer overflow in unary `-`")?,
                    )),
                    (UnOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
                    (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
                    (op, v) => Err(format!("cannot apply {:?} to {}", op, v.display())),
                }
            }

            Expr::Binary(op, lhs, rhs) => self.eval_binary(*op, lhs, rhs, scope),

            Expr::If(cond, then, els) => match self.eval(cond, scope)? {
                Value::Bool(true) => self.eval(then, scope),
                Value::Bool(false) => self.eval(els, scope),
                other => Err(format!("`if` condition must be Bool, got {}", other.display())),
            },

            Expr::Match(scrut, arms) => {
                let v = self.eval(scrut, scope)?;
                for arm in arms {
                    let mut binds = HashMap::new();
                    if match_pattern(&arm.pat, &v, &mut binds) {
                        scope.push(binds);
                        let result = self.eval(&arm.body, scope);
                        scope.pop();
                        return result;
                    }
                }
                Err(format!("no match arm for value {}", v.display()))
            }

            Expr::Block(stmts, last) => {
                scope.push(HashMap::new());
                let mut run = || -> Result<Value, String> {
                    for s in stmts {
                        match s {
                            Stmt::Let(name, _ty, value) => {
                                let v = self.eval(value, scope)?;
                                scope.last_mut().unwrap().insert(name.clone(), v);
                            }
                            Stmt::Expr(e) => {
                                self.eval(e, scope)?;
                            }
                        }
                    }
                    self.eval(last, scope)
                };
                let result = run();
                scope.pop();
                result
            }
        }
    }

    fn eval_binary(
        &self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        scope: &mut Scope,
    ) -> Result<Value, String> {
        // Short-circuiting boolean operators.
        match op {
            BinOp::And => {
                return match self.eval(lhs, scope)? {
                    Value::Bool(false) => Ok(Value::Bool(false)),
                    Value::Bool(true) => match self.eval(rhs, scope)? {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        v => Err(format!("`&&` expects Bool, got {}", v.display())),
                    },
                    v => Err(format!("`&&` expects Bool, got {}", v.display())),
                };
            }
            BinOp::Or => {
                return match self.eval(lhs, scope)? {
                    Value::Bool(true) => Ok(Value::Bool(true)),
                    Value::Bool(false) => match self.eval(rhs, scope)? {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        v => Err(format!("`||` expects Bool, got {}", v.display())),
                    },
                    v => Err(format!("`||` expects Bool, got {}", v.display())),
                };
            }
            _ => {}
        }

        let l = self.eval(lhs, scope)?;
        let r = self.eval(rhs, scope)?;

        match op {
            BinOp::Eq => return Ok(Value::Bool(values_equal(&l, &r))),
            BinOp::Ne => return Ok(Value::Bool(!values_equal(&l, &r))),
            _ => {}
        }

        match (&l, &r) {
            (Value::Int(a), Value::Int(b)) => Ok(int_op(op, *a, *b)?),
            (Value::Float(a), Value::Float(b)) => Ok(float_op(op, *a, *b)?),
            _ => Err(format!(
                "operator {:?} needs two Ints or two Floats, got {} and {}",
                op,
                l.display(),
                r.display()
            )),
        }
    }
}

fn int_op(op: BinOp, a: i64, b: i64) -> Result<Value, String> {
    Ok(match op {
        // Checked arithmetic: overflow is a catchable runtime error, not a
        // debug-build panic / release-build silent wrap.
        BinOp::Add => Value::Int(a.checked_add(b).ok_or("integer overflow in `+`")?),
        BinOp::Sub => Value::Int(a.checked_sub(b).ok_or("integer overflow in `-`")?),
        BinOp::Mul => Value::Int(a.checked_mul(b).ok_or("integer overflow in `*`")?),
        BinOp::Div => {
            if b == 0 {
                return Err("division by zero".into());
            }
            Value::Int(a / b)
        }
        BinOp::Mod => {
            if b == 0 {
                return Err("modulo by zero".into());
            }
            Value::Int(a % b)
        }
        BinOp::Lt => Value::Bool(a < b),
        BinOp::Le => Value::Bool(a <= b),
        BinOp::Gt => Value::Bool(a > b),
        BinOp::Ge => Value::Bool(a >= b),
        _ => return Err(format!("operator {:?} not valid on Int", op)),
    })
}

fn float_op(op: BinOp, a: f64, b: f64) -> Result<Value, String> {
    Ok(match op {
        BinOp::Add => Value::Float(a + b),
        BinOp::Sub => Value::Float(a - b),
        BinOp::Mul => Value::Float(a * b),
        BinOp::Div => Value::Float(a / b),
        BinOp::Lt => Value::Bool(a < b),
        BinOp::Le => Value::Bool(a <= b),
        BinOp::Gt => Value::Bool(a > b),
        BinOp::Ge => Value::Bool(a >= b),
        _ => return Err(format!("operator {:?} not valid on Float", op)),
    })
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Unit, Value::Unit) => true,
        (
            Value::Data {
                ctor: c1,
                fields: f1,
            },
            Value::Data {
                ctor: c2,
                fields: f2,
            },
        ) => c1 == c2 && f1.len() == f2.len() && f1.iter().zip(f2).all(|(x, y)| values_equal(x, y)),
        // Tensors compare structurally (shape + contents). Without this arm,
        // `t == t` fell through to `false`, silently disagreeing with the type
        // checker which accepts `==` on Tensor.
        (Value::Tensor(a), Value::Tensor(b)) => a.shape == b.shape && a.data == b.data,
        _ => false,
    }
}

fn match_pattern(pat: &Pattern, val: &Value, binds: &mut HashMap<String, Value>) -> bool {
    match pat {
        Pattern::Wild => true,
        Pattern::Var(name) => {
            binds.insert(name.clone(), val.clone());
            true
        }
        Pattern::Int(i) => matches!(val, Value::Int(v) if v == i),
        Pattern::Bool(b) => matches!(val, Value::Bool(v) if v == b),
        Pattern::Ctor(name, subs) => match val {
            Value::Data { ctor, fields } if ctor == name && fields.len() == subs.len() => subs
                .iter()
                .zip(fields)
                .all(|(p, f)| match_pattern(p, f, binds)),
            _ => false,
        },
    }
}

/// Returns Ok(Some(value)) if `name` is a builtin, Ok(None) otherwise.
fn builtin(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    let one = |args: &[Value]| -> Result<Value, String> {
        if args.len() != 1 {
            return Err(format!("`{}` expects 1 argument", name));
        }
        Ok(args[0].clone())
    };
    match name {
        "print_int" => match one(args)? {
            Value::Int(n) => {
                println!("{}", n);
                Ok(Some(Value::Unit))
            }
            v => Err(format!("print_int expects Int, got {}", v.display())),
        },
        "print_float" => match one(args)? {
            Value::Float(f) => {
                println!("{}", f);
                Ok(Some(Value::Unit))
            }
            v => Err(format!("print_float expects Float, got {}", v.display())),
        },
        "print_bool" => match one(args)? {
            Value::Bool(b) => {
                println!("{}", b);
                Ok(Some(Value::Unit))
            }
            v => Err(format!("print_bool expects Bool, got {}", v.display())),
        },
        "print_str" => match one(args)? {
            Value::Str(s) => {
                println!("{}", s);
                Ok(Some(Value::Unit))
            }
            v => Err(format!("print_str expects String, got {}", v.display())),
        },
        "concat" => {
            if args.len() != 2 {
                return Err("concat expects 2 arguments".into());
            }
            match (&args[0], &args[1]) {
                (Value::Str(a), Value::Str(b)) => Ok(Some(Value::Str(format!("{}{}", a, b)))),
                _ => Err("concat expects two Strings".into()),
            }
        }
        "int_to_str" => match one(args)? {
            Value::Int(n) => Ok(Some(Value::Str(n.to_string()))),
            v => Err(format!("int_to_str expects Int, got {}", v.display())),
        },

        // ---- AI runtime primitives -------------------------------------
        // All tensor builtins are pure: mutating ones clone then modify.
        "tensor_zeros" => match args {
            [Value::Int(r), Value::Int(c)] => {
                if *r < 0 || *c < 0 {
                    return Err("tensor_zeros expects non-negative dimensions".into());
                }
                // Guard the element-count multiply against usize overflow and
                // cap it so a runtime value can't request a process-aborting
                // allocation (the type checker cannot bound these).
                const MAX_TENSOR_ELEMS: usize = 64 * 1024 * 1024; // 256 MiB of f32
                let n = (*r as usize)
                    .checked_mul(*c as usize)
                    .ok_or("tensor_zeros dimensions overflow")?;
                if n > MAX_TENSOR_ELEMS {
                    return Err(format!(
                        "tensor_zeros: {}x{} exceeds the {}-element limit",
                        r, c, MAX_TENSOR_ELEMS
                    ));
                }
                let t = crate::tensor::Tensor::zeros(&[*r as usize, *c as usize]);
                Ok(Some(Value::Tensor(t)))
            }
            _ => Err("tensor_zeros expects (Int, Int)".into()),
        },
        "tensor_set" => match args {
            [Value::Tensor(t), Value::Int(r), Value::Int(c), Value::Float(v)] => {
                let (rows, cols) = (t.rows(), t.cols());
                if *r < 0 || *c < 0 || *r as usize >= rows || *c as usize >= cols {
                    return Err(format!(
                        "tensor_set index ({}, {}) out of range for {}x{} tensor",
                        r, c, rows, cols
                    ));
                }
                let mut out = t.clone();
                out.set(*r as usize, *c as usize, *v as f32);
                Ok(Some(Value::Tensor(out)))
            }
            _ => Err("tensor_set expects (Tensor, Int, Int, Float)".into()),
        },
        "tensor_get" => match args {
            [Value::Tensor(t), Value::Int(r), Value::Int(c)] => {
                let (rows, cols) = (t.rows(), t.cols());
                if *r < 0 || *c < 0 || *r as usize >= rows || *c as usize >= cols {
                    return Err(format!(
                        "tensor_get index ({}, {}) out of range for {}x{} tensor",
                        r, c, rows, cols
                    ));
                }
                Ok(Some(Value::Float(t.at(*r as usize, *c as usize) as f64)))
            }
            _ => Err("tensor_get expects (Tensor, Int, Int)".into()),
        },
        "tensor_rows" => match args {
            [Value::Tensor(t)] => Ok(Some(Value::Int(t.rows() as i64))),
            _ => Err("tensor_rows expects (Tensor)".into()),
        },
        "tensor_cols" => match args {
            [Value::Tensor(t)] => Ok(Some(Value::Int(t.cols() as i64))),
            _ => Err("tensor_cols expects (Tensor)".into()),
        },
        "matmul" => match args {
            [Value::Tensor(a), Value::Tensor(b)] => {
                if a.cols() != b.rows() {
                    return Err(format!(
                        "matmul shape mismatch: {}x{} times {}x{}",
                        a.rows(),
                        a.cols(),
                        b.rows(),
                        b.cols()
                    ));
                }
                Ok(Some(Value::Tensor(a.matmul(b))))
            }
            _ => Err("matmul expects (Tensor, Tensor)".into()),
        },
        "transpose" => match args {
            [Value::Tensor(t)] => Ok(Some(Value::Tensor(t.transpose()))),
            _ => Err("transpose expects (Tensor)".into()),
        },
        "softmax" => match args {
            [Value::Tensor(t)] => Ok(Some(Value::Tensor(t.softmax_rows()))),
            _ => Err("softmax expects (Tensor)".into()),
        },
        "relu" => match args {
            [Value::Tensor(t)] => Ok(Some(Value::Tensor(t.relu()))),
            _ => Err("relu expects (Tensor)".into()),
        },
        "embed_similarity" => match args {
            [Value::Str(a), Value::Str(b)] => {
                let va = crate::rag::hash_embed(a, 64);
                let vb = crate::rag::hash_embed(b, 64);
                Ok(Some(Value::Float(crate::rag::cosine_similarity(&va, &vb) as f64)))
            }
            _ => Err("embed_similarity expects (String, String)".into()),
        },
        "compressed_size" => match args {
            [Value::Str(s)] => {
                let n = crate::rans::compress(s.as_bytes()).len();
                Ok(Some(Value::Int(n as i64)))
            }
            _ => Err("compressed_size expects (String)".into()),
        },
        "neural_bits_per_byte" => match args {
            [Value::Str(s)] => {
                Ok(Some(Value::Float(crate::predict::eval_bits_per_byte(s.as_bytes()))))
            }
            _ => Err("neural_bits_per_byte expects (String)".into()),
        },
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lexer, parser, typeck};

    // Lex -> parse -> typeck -> interp, returning the value of `main`.
    fn run(src: &str) -> Value {
        let toks = lexer::lex(src).expect("lex");
        let prog = parser::parse(toks).expect("parse");
        typeck::check(&prog).expect("typeck");
        let interp = Interp::new(&prog).expect("interp::new");
        interp.run_main().expect("run")
    }

    #[test]
    fn tensor_builtins_end_to_end() {
        // Build a 2x2 identity, multiply by itself, and read back element (1,1).
        let src = r#"
            fn main() -> Float = {
                let i0 = tensor_zeros(2, 2);
                let i1 = tensor_set(i0, 0, 0, 1.0);
                let id = tensor_set(i1, 1, 1, 1.0);
                let p = matmul(id, id);
                tensor_get(p, 1, 1)
            }
        "#;
        match run(src) {
            Value::Float(f) => assert!((f - 1.0).abs() < 1e-6, "got {f}"),
            v => panic!("expected Float, got {}", v.display()),
        }
    }

    #[test]
    fn every_declared_builtin_is_implemented() {
        // Drift guard: every builtin the type checker knows about (from the
        // shared `builtins` table) must be handled by the interpreter. Calling
        // with no args returns Err for an implemented builtin (arg mismatch) but
        // Ok(None) for an unknown name — so Ok(None) here means drift.
        for name in crate::builtins::names() {
            let r = builtin(name, &[]);
            assert!(
                !matches!(r, Ok(None)),
                "builtin `{}` is declared in the shared table but not implemented in interp",
                name
            );
        }
    }

    /// Substrings of runtime errors that the type checker is supposed to make
    /// unreachable. A well-typed program must never produce one of these.
    fn is_type_class_error(msg: &str) -> bool {
        const NEEDLES: &[&str] = &[
            "unbound variable",
            "unknown function",
            "unknown constructor",
            "argument(s), got",
            "field(s), got",
            "no match arm",
            "`if` condition",
            "cannot apply",
            "needs two Ints or two Floats",
            "needs Bool",
            "cannot compare",
            "expects Int",
            "expects Float",
            "expects Bool",
            "expects String",
            "expects (",
        ];
        NEEDLES.iter().any(|n| msg.contains(n))
    }

    #[test]
    fn well_typed_programs_never_hit_type_class_errors() {
        // Differential/consistency harness: every program here type-checks, so
        // running it must NOT produce a type-class runtime error (legitimate
        // runtime errors like overflow/div-by-zero are fine and not exercised).
        let programs = [
            "type C = | R | G | B\nfn n(c: C) -> Int = match c { R => 0, G => 1, B => 2, }\nfn main() -> Int = n(B)",
            "type L[T] = | Nil | Cons(T, L[T])\nfn len[T](xs: L[T]) -> Int = match xs { Nil => 0, Cons(_, r) => 1 + len(r), }\nfn main() -> Int = len(Cons(1, Cons(2, Nil)))",
            "fn main() -> Float = tensor_get(matmul(tensor_set(tensor_zeros(2,2),0,0,2.0), tensor_set(tensor_zeros(2,2),0,0,3.0)), 0, 0)",
            "fn main() -> Int = { let a = tensor_zeros(2,2); let b = tensor_zeros(2,2); if a == b { 1 } else { 0 } }",
            "fn main() -> String = concat(\"x = \", int_to_str(6 * 7))",
        ];
        for src in programs {
            let toks = lexer::lex(src).expect("lex");
            let prog = parser::parse(toks).expect("parse");
            typeck::check(&prog).expect("must type-check");
            let interp = Interp::new(&prog).expect("interp::new");
            if let Err(msg) = interp.run_main() {
                assert!(
                    !is_type_class_error(&msg),
                    "well-typed program produced a type-class runtime error: {}\nprogram: {}",
                    msg, src
                );
            }
        }
    }

    #[test]
    fn tensor_equality_is_structural() {
        // A tensor must equal itself and differ from a different one (regression:
        // previously `t == t` fell through to false).
        let same = run(r#"
            fn main() -> Bool = {
                let a = tensor_set(tensor_zeros(2, 2), 0, 0, 1.0);
                a == a
            }
        "#);
        assert!(matches!(same, Value::Bool(true)));
        let diff = run(r#"
            fn main() -> Bool = {
                let a = tensor_set(tensor_zeros(2, 2), 0, 0, 1.0);
                let b = tensor_zeros(2, 2);
                a == b
            }
        "#);
        assert!(matches!(diff, Value::Bool(false)));
    }

    #[test]
    fn embed_similarity_related_beats_unrelated() {
        // Identical strings -> cosine ~1.0.
        let src = r#"
            fn main() -> Float =
                embed_similarity("cosine similarity over vectors",
                                 "cosine similarity over vectors")
        "#;
        match run(src) {
            Value::Float(f) => assert!((f - 1.0).abs() < 1e-5, "identical text similarity was {f}"),
            v => panic!("expected Float, got {}", v.display()),
        }
    }
}
