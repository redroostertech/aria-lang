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
    /// A functional array (`Array[T]`). The interpreter is the reference oracle,
    /// so it uses a plain `Vec` with copy-on-write semantics; the FBIP in-place
    /// reuse that the compiled backends perform is an optimization that cannot
    /// change observable results, so the oracle need not model it.
    Array(Vec<Value>),
    /// A flat, growable byte buffer (`Bytes`). A byte is an Int 0..255. Distinct
    /// from `Str` (own type tag) so `==`/display never conflate the two even when
    /// they hold identical bytes.
    Bytes(Vec<u8>),
    /// An ordered map (`Map[K, V]`), kept SORTED BY KEY ascending so iteration,
    /// display, and equality are deterministic and identical across all
    /// backends. Keys are restricted (by the checker) to Int or Str.
    Map(Vec<(Value, Value)>),
    /// An ordered set (`Set[T]`), kept SORTED ascending. Elements are Int or Str.
    Set(Vec<Value>),
    /// An opaque AI-runtime tensor handle, built and queried via builtins.
    Tensor(crate::tensor::Tensor),
    /// A first-class function value. A lambda captures the environment in which
    /// it was created; a bare top-level function name becomes a closure with an
    /// empty captured environment. Boxed (via `Arc`) so adding closures does not
    /// enlarge `Value` and blow the recursive-interpreter stack.
    Closure(std::sync::Arc<ClosureData>),
    Unit,
}

/// The payload of a `Value::Closure`, heap-allocated behind an `Arc`.
#[derive(Debug)]
pub struct ClosureData {
    pub params: Vec<String>,
    pub body: Expr,
    pub env: HashMap<String, Value>,
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
            Value::Array(xs) => {
                let inner: Vec<String> = xs.iter().map(|v| v.display()).collect();
                format!("[{}]", inner.join(", "))
            }
            Value::Bytes(bs) => render_bytes(bs),
            Value::Map(entries) => {
                let inner: Vec<String> = entries
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k.display(), v.display()))
                    .collect();
                format!("Map[{}]", inner.join(", "))
            }
            Value::Set(elems) => {
                let inner: Vec<String> = elems.iter().map(|v| v.display()).collect();
                format!("Set[{}]", inner.join(", "))
            }
            Value::Closure(c) => {
                format!("<closure/{}>", c.params.len())
            }
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

/// The ONE canonical textual rendering of a `Bytes` value, emitted byte-for-byte
/// identically by every backend (interp / IR / native / wasm): the literal
/// `Bytes[`, then each byte as lowercase two-digit hex separated by single
/// spaces, then `]`. Empty is `Bytes[]`.
pub fn render_bytes(bs: &[u8]) -> String {
    let inner: Vec<String> = bs.iter().map(|b| format!("{:02x}", b)).collect();
    format!("Bytes[{}]", inner.join(" "))
}

/// Total ordering on map keys / set elements. Keys are restricted to Int and Str
/// by the type checker; Ints order numerically, Strs lexicographically by bytes.
/// A mixed comparison cannot arise (a Map/Set is homogeneous), but is given a
/// stable fallback so the function is total.
fn key_cmp(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x.cmp(y),
        (Value::Str(x), Value::Str(y)) => x.as_bytes().cmp(y.as_bytes()),
        // Unreachable for well-typed programs; keep a deterministic fallback.
        (Value::Int(_), _) => Ordering::Less,
        (_, Value::Int(_)) => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

type Scope = Vec<HashMap<String, Value>>;

/// The result of evaluating a function body in tail position: either a final
/// value, or a request to re-enter the SAME function (self-tail-call) with these
/// already-evaluated argument values. Drives the interpreter's TCO loop.
enum TailOutcome {
    Value(Value),
    SelfCall(Vec<Value>),
}

pub struct Interp {
    fns: HashMap<String, FnDecl>,
    /// constructor name -> arity
    ctors: HashMap<String, usize>,
    /// record constructor name -> declared field names (in declared order), used
    /// to reorder record-literal fields and resolve `.field` access by index.
    record_fields: HashMap<String, Vec<String>>,
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
        let mut record_fields = HashMap::new();
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
                        if let Some(names) = &v.field_names {
                            record_fields.insert(v.name.clone(), names.clone());
                        }
                    }
                }
            }
        }
        Ok(Interp { fns, ctors, record_fields, depth: std::cell::Cell::new(0) })
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
        self.eval_fn_body("main", &main.body, &mut scope)
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
                } else if let Some(c) = self.fn_as_closure(name) {
                    // A bare top-level function name used as a value.
                    Ok(c)
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

            Expr::Record(name, fields) => {
                let decl = self
                    .record_fields
                    .get(name)
                    .ok_or_else(|| format!("unknown record type `{}`", name))?
                    .clone();
                // Evaluate field values in SOURCE (left-to-right) order so side
                // effects are observed as written, then assemble the positional
                // `Data` in DECLARED field order (the canonical layout).
                let mut evaled: Vec<(String, Value)> = Vec::with_capacity(fields.len());
                for (fname, val_expr) in fields {
                    evaled.push((fname.clone(), self.eval(val_expr, scope)?));
                }
                let mut vals = Vec::with_capacity(decl.len());
                for fname in &decl {
                    let v = evaled
                        .iter()
                        .find(|(n, _)| n == fname)
                        .ok_or_else(|| format!("record `{}`: missing field `{}`", name, fname))?
                        .1
                        .clone();
                    vals.push(v);
                }
                Ok(Value::Data { ctor: name.clone(), fields: vals })
            }

            Expr::Field(obj, field) => {
                let v = self.eval(obj, scope)?;
                match v {
                    Value::Data { ctor, fields } => {
                        let decl = self
                            .record_fields
                            .get(&ctor)
                            .ok_or_else(|| format!("type `{}` is not a record", ctor))?;
                        let idx = decl
                            .iter()
                            .position(|n| n == field)
                            .ok_or_else(|| format!("type `{}` has no field `{}`", ctor, field))?;
                        Ok(fields[idx].clone())
                    }
                    other => Err(format!(
                        "field access `.{}` on a non-record value {}",
                        field,
                        other.display()
                    )),
                }
            }

            Expr::Update(base, updates) => {
                let v = self.eval(base, scope)?;
                match v {
                    Value::Data { ctor, mut fields } => {
                        let decl = self
                            .record_fields
                            .get(&ctor)
                            .ok_or_else(|| format!("type `{}` is not a record", ctor))?
                            .clone();
                        for (fname, val_expr) in updates {
                            let idx = decl
                                .iter()
                                .position(|n| n == fname)
                                .ok_or_else(|| format!("type `{}` has no field `{}`", ctor, fname))?;
                            fields[idx] = self.eval(val_expr, scope)?;
                        }
                        Ok(Value::Data { ctor, fields })
                    }
                    other => Err(format!(
                        "record update on a non-record value {}",
                        other.display()
                    )),
                }
            }

            Expr::Call(name, args) => {
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval(a, scope)?);
                }
                // A local binding that shadows the name (e.g. a function-valued
                // parameter) is applied as a closure value.
                if let Some(v) = Interp::lookup(scope, name) {
                    let callee = v.clone();
                    return self.apply_value(callee, vals, name);
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
                // Evaluate the body with SELF-tail-call elimination: a tail call
                // to `name` reuses this frame (a loop) instead of recursing, so
                // tail recursion runs in constant stack and never trips the depth
                // guard. Only this single nested call frame is on the stack.
                let result = self.eval_fn_body(name, &f.body, &mut call_scope);
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
                    if match_pattern(&arm.pat, &v, &mut binds, &self.record_fields) {
                        scope.push(binds);
                        let result = self.eval(&arm.body, scope);
                        scope.pop();
                        return result;
                    }
                }
                Err(format!("no match arm for value {}", v.display()))
            }

            Expr::Lambda(params, body, _) => {
                // Capture the current environment by flattening all in-scope
                // frames (inner shadowing outer) into the closure's env.
                let mut env = HashMap::new();
                for frame in scope.iter() {
                    for (k, v) in frame {
                        env.insert(k.clone(), v.clone());
                    }
                }
                Ok(Value::Closure(std::sync::Arc::new(ClosureData {
                    params: params.iter().map(|(n, _)| n.clone()).collect(),
                    body: (**body).clone(),
                    env,
                })))
            }

            Expr::Apply(callee, args, _) => {
                let f = self.eval(callee, scope)?;
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval(a, scope)?);
                }
                self.apply_value(f, vals, "value")
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

    /// Evaluate the body of the function named `self_name` with SELF-tail-call
    /// elimination. `scope` holds exactly the call frame (params bound at its
    /// base). We loop: evaluate toward the tail; if the tail is a direct call to
    /// `self_name` with matching arity, evaluate its argument values, REBIND the
    /// parameters, and re-iterate the same body (a loop) instead of recursing.
    /// All new argument values are computed BEFORE any parameter is rebound (a
    /// parameter may appear in another argument). Non-tail calls — and tail calls
    /// to OTHER functions — go through the ordinary depth-guarded path.
    fn eval_fn_body(
        &self,
        self_name: &str,
        body: &Expr,
        scope: &mut Scope,
    ) -> Result<Value, String> {
        loop {
            match self.eval_tail(self_name, body, scope)? {
                TailOutcome::Value(v) => return Ok(v),
                TailOutcome::SelfCall(args) => {
                    // Rebind the parameters in the (single) call frame and loop.
                    // The frame is `scope[0]`; tail position never leaves extra
                    // frames pushed (Block/Match frames are popped before we
                    // surface a SelfCall — see `eval_tail`).
                    let params = &self.fns[self_name].params;
                    let frame = scope.first_mut().expect("call frame present");
                    for (p, v) in params.iter().zip(args.into_iter()) {
                        frame.insert(p.name.clone(), v);
                    }
                }
            }
        }
    }

    /// Evaluate `e`, which is in TAIL position of the function named
    /// `self_name`. Returns either the value, or a self-tail-call request
    /// carrying the already-evaluated argument values. Tail position flows
    /// through `if` branches, `match` arm bodies, and the final expression of a
    /// `Block`; any other expression is evaluated normally via `self.eval`.
    fn eval_tail(
        &self,
        self_name: &str,
        e: &Expr,
        scope: &mut Scope,
    ) -> Result<TailOutcome, String> {
        match e {
            Expr::Call(name, args) => {
                // A direct self-call (not shadowed by a local of the same name,
                // not a builtin) with matching arity is a self-tail-call.
                let shadowed = Interp::lookup(scope, name).is_some();
                let is_self = name == self_name
                    && !shadowed
                    && self
                        .fns
                        .get(name)
                        .map(|f| f.params.len() == args.len())
                        .unwrap_or(false);
                if is_self {
                    let mut vals = Vec::with_capacity(args.len());
                    for a in args {
                        vals.push(self.eval(a, scope)?);
                    }
                    Ok(TailOutcome::SelfCall(vals))
                } else {
                    self.eval(e, scope).map(TailOutcome::Value)
                }
            }
            Expr::If(cond, then, els) => match self.eval(cond, scope)? {
                Value::Bool(true) => self.eval_tail(self_name, then, scope),
                Value::Bool(false) => self.eval_tail(self_name, els, scope),
                other => Err(format!("`if` condition must be Bool, got {}", other.display())),
            },
            Expr::Match(scrut, arms) => {
                let v = self.eval(scrut, scope)?;
                for arm in arms {
                    let mut binds = HashMap::new();
                    if match_pattern(&arm.pat, &v, &mut binds, &self.record_fields) {
                        scope.push(binds);
                        let result = self.eval_tail(self_name, &arm.body, scope);
                        scope.pop();
                        return result;
                    }
                }
                Err(format!("no match arm for value {}", v.display()))
            }
            Expr::Block(stmts, last) => {
                scope.push(HashMap::new());
                let run = |me: &Self, scope: &mut Scope| -> Result<TailOutcome, String> {
                    for s in stmts {
                        match s {
                            Stmt::Let(name, _ty, value) => {
                                let val = me.eval(value, scope)?;
                                scope.last_mut().unwrap().insert(name.clone(), val);
                            }
                            Stmt::Expr(ex) => {
                                me.eval(ex, scope)?;
                            }
                        }
                    }
                    me.eval_tail(self_name, last, scope)
                };
                let result = run(self, scope);
                // A `SelfCall` carries fully-evaluated argument values, so it is
                // safe to pop this block's frame before returning it: the values
                // no longer reference any local bound in the block.
                scope.pop();
                result
            }
            // Not a tail-position construct: evaluate normally.
            _ => self.eval(e, scope).map(TailOutcome::Value),
        }
    }

    /// Build a closure value for a top-level function used as a value. The
    /// captured environment is empty: a top-level function closes over nothing
    /// (it can still reference other top-level functions, resolved at call time).
    fn fn_as_closure(&self, name: &str) -> Option<Value> {
        let f = self.fns.get(name)?;
        Some(Value::Closure(std::sync::Arc::new(ClosureData {
            params: f.params.iter().map(|p| p.name.clone()).collect(),
            body: f.body.clone(),
            env: HashMap::new(),
        })))
    }

    /// Apply a function VALUE (a closure) to already-evaluated arguments. Binds
    /// the parameters in the closure's captured environment and evaluates the
    /// body, honoring the recursion-depth guard.
    fn apply_value(&self, callee: Value, args: Vec<Value>, what: &str) -> Result<Value, String> {
        let data = match callee {
            Value::Closure(c) => c,
            other => {
                return Err(format!(
                    "cannot apply {} `{}`: it is not a function",
                    what,
                    other.display()
                ))
            }
        };
        if data.params.len() != args.len() {
            return Err(format!(
                "function value expects {} argument(s), got {}",
                data.params.len(),
                args.len()
            ));
        }
        let mut frame = data.env.clone();
        for (p, v) in data.params.iter().zip(args.into_iter()) {
            frame.insert(p.clone(), v);
        }
        let mut call_scope: Scope = vec![frame];
        let d = self.depth.get() + 1;
        if d > MAX_CALL_DEPTH {
            return Err(format!(
                "maximum recursion depth ({}) exceeded applying {}",
                MAX_CALL_DEPTH, what
            ));
        }
        self.depth.set(d);
        let result = self.eval(&data.body, &mut call_scope);
        self.depth.set(d - 1);
        result
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
            Value::Int(a.checked_div(b).ok_or("integer overflow in `/`")?)
        }
        BinOp::Mod => {
            if b == 0 {
                return Err("modulo by zero".into());
            }
            Value::Int(a.checked_rem(b).ok_or("integer overflow in `%`")?)
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
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| values_equal(a, b))
        }
        // Bytes compare by content. The (Bytes, Str) and (Str, Bytes) pairs fall
        // through to `false` below: a Bytes never equals a Str (distinct tags),
        // and the type checker rejects such a comparison up front anyway.
        (Value::Bytes(x), Value::Bytes(y)) => x == y,
        // Maps/Sets are kept sorted, so structural element-wise comparison of the
        // ordered contents is exact equality of the abstract map/set.
        (Value::Map(x), Value::Map(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|((k1, v1), (k2, v2))| values_equal(k1, k2) && values_equal(v1, v2))
        }
        (Value::Set(x), Value::Set(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| values_equal(a, b))
        }
        // Tensors compare structurally (shape + contents). Without this arm,
        // `t == t` fell through to `false`, silently disagreeing with the type
        // checker which accepts `==` on Tensor.
        (Value::Tensor(a), Value::Tensor(b)) => {
            // Reflexive even with NaN elements (NaN != NaN in IEEE, but `t == t`
            // should hold): treat two NaNs as equal.
            a.shape == b.shape
                && a.data.len() == b.data.len()
                && a.data.iter().zip(&b.data).all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
        }
        _ => false,
    }
}

fn match_pattern(
    pat: &Pattern,
    val: &Value,
    binds: &mut HashMap<String, Value>,
    record_fields: &HashMap<String, Vec<String>>,
) -> bool {
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
                .all(|(p, f)| match_pattern(p, f, binds, record_fields)),
            _ => false,
        },
        Pattern::Record(name, sub_fields) => match val {
            Value::Data { ctor, fields } if ctor == name => {
                let decl = match record_fields.get(name) {
                    Some(d) => d,
                    None => return false,
                };
                sub_fields.iter().all(|(fname, subpat)| {
                    match decl.iter().position(|n| n == fname) {
                        Some(i) => match_pattern(subpat, &fields[i], binds, record_fields),
                        None => false,
                    }
                })
            }
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

        // ---- Arrays --------------------------------------------------------
        // Functional API: `set`/`push` return a new array (the oracle copies;
        // the compiled backends reuse in place when the array is unique).
        // Out-of-bounds `get`/`set` are runtime errors (the type checker cannot
        // bound an index), matching how tensor get/set guard their indices.
        "array_new" => match args {
            [] => Ok(Some(Value::Array(Vec::new()))),
            _ => Err("array_new expects no arguments".into()),
        },
        "array_len" => match args {
            [Value::Array(xs)] => Ok(Some(Value::Int(xs.len() as i64))),
            _ => Err("array_len expects (Array)".into()),
        },
        "array_get" => match args {
            [Value::Array(xs), Value::Int(i)] => {
                if *i < 0 || *i as usize >= xs.len() {
                    return Err(format!(
                        "array_get index {} out of range for array of length {}",
                        i,
                        xs.len()
                    ));
                }
                Ok(Some(xs[*i as usize].clone()))
            }
            _ => Err("array_get expects (Array, Int)".into()),
        },
        "array_set" => match args {
            [Value::Array(xs), Value::Int(i), v] => {
                if *i < 0 || *i as usize >= xs.len() {
                    return Err(format!(
                        "array_set index {} out of range for array of length {}",
                        i,
                        xs.len()
                    ));
                }
                let mut out = xs.clone();
                out[*i as usize] = v.clone();
                Ok(Some(Value::Array(out)))
            }
            _ => Err("array_set expects (Array, Int, T)".into()),
        },
        "array_push" => match args {
            [Value::Array(xs), v] => {
                let mut out = xs.clone();
                out.push(v.clone());
                Ok(Some(Value::Array(out)))
            }
            _ => Err("array_push expects (Array, T)".into()),
        },
        // Array literal: variadic, desugared by the parser to a single flat
        // `Call("array_lit", [e0,...,en])`. Builds the array from all argument
        // values in order; empty args -> empty array. Not in `signatures()`
        // (it is variadic), so it bypasses the drift table/test.
        "array_lit" => Ok(Some(Value::Array(args.to_vec()))),

        // ---- Bytes ---------------------------------------------------------
        // A flat byte buffer. `set`/`push` are functional (the oracle copies; the
        // compiled backends reuse in place when unique). Out-of-range index on
        // get/set is a runtime error. A byte value outside 0..255 on set/push is
        // REJECTED with a runtime error (the policy is identical in every
        // backend). `to_str` errors on invalid UTF-8.
        "bytes_new" => match args {
            [] => Ok(Some(Value::Bytes(Vec::new()))),
            _ => Err("bytes_new expects no arguments".into()),
        },
        "bytes_len" => match args {
            [Value::Bytes(bs)] => Ok(Some(Value::Int(bs.len() as i64))),
            _ => Err("bytes_len expects (Bytes)".into()),
        },
        "bytes_get" => match args {
            [Value::Bytes(bs), Value::Int(i)] => {
                if *i < 0 || *i as usize >= bs.len() {
                    return Err(format!(
                        "bytes_get index {} out of range for bytes of length {}",
                        i,
                        bs.len()
                    ));
                }
                Ok(Some(Value::Int(bs[*i as usize] as i64)))
            }
            _ => Err("bytes_get expects (Bytes, Int)".into()),
        },
        "bytes_set" => match args {
            [Value::Bytes(bs), Value::Int(i), Value::Int(v)] => {
                if *i < 0 || *i as usize >= bs.len() {
                    return Err(format!(
                        "bytes_set index {} out of range for bytes of length {}",
                        i,
                        bs.len()
                    ));
                }
                if *v < 0 || *v > 255 {
                    return Err(format!("bytes_set byte value {} out of range 0..255", v));
                }
                let mut out = bs.clone();
                out[*i as usize] = *v as u8;
                Ok(Some(Value::Bytes(out)))
            }
            _ => Err("bytes_set expects (Bytes, Int, Int)".into()),
        },
        "bytes_push" => match args {
            [Value::Bytes(bs), Value::Int(v)] => {
                if *v < 0 || *v > 255 {
                    return Err(format!("bytes_push byte value {} out of range 0..255", v));
                }
                let mut out = bs.clone();
                out.push(*v as u8);
                Ok(Some(Value::Bytes(out)))
            }
            _ => Err("bytes_push expects (Bytes, Int)".into()),
        },
        "bytes_from_str" => match args {
            [Value::Str(s)] => Ok(Some(Value::Bytes(s.as_bytes().to_vec()))),
            _ => Err("bytes_from_str expects (Str)".into()),
        },
        "bytes_to_str" => match args {
            [Value::Bytes(bs)] => match std::str::from_utf8(bs) {
                Ok(s) => Ok(Some(Value::Str(s.to_string()))),
                Err(_) => Err("bytes_to_str: invalid UTF-8".into()),
            },
            _ => Err("bytes_to_str expects (Bytes)".into()),
        },

        // ---- Ordered Map ---------------------------------------------------
        // A Map is a Vec of (key, value) kept sorted by key. The interpreter is
        // the oracle, so it copies (the compiled backends reuse in place when
        // unique). `insert` replaces an existing key; the read API is total
        // (`get_or` returns its default when the key is absent).
        "map_new" => match args {
            [] => Ok(Some(Value::Map(Vec::new()))),
            _ => Err("map_new expects no arguments".into()),
        },
        "map_insert" => match args {
            [Value::Map(entries), k, v] => {
                let mut out = entries.clone();
                match out.binary_search_by(|(ek, _)| key_cmp(ek, k)) {
                    Ok(i) => out[i].1 = v.clone(),          // replace existing value
                    Err(i) => out.insert(i, (k.clone(), v.clone())),
                }
                Ok(Some(Value::Map(out)))
            }
            _ => Err("map_insert expects (Map, K, V)".into()),
        },
        "map_get_or" => match args {
            [Value::Map(entries), k, default] => {
                let r = match entries.binary_search_by(|(ek, _)| key_cmp(ek, k)) {
                    Ok(i) => entries[i].1.clone(),
                    Err(_) => default.clone(),
                };
                Ok(Some(r))
            }
            _ => Err("map_get_or expects (Map, K, V)".into()),
        },
        "map_has" => match args {
            [Value::Map(entries), k] => Ok(Some(Value::Bool(
                entries.binary_search_by(|(ek, _)| key_cmp(ek, k)).is_ok(),
            ))),
            _ => Err("map_has expects (Map, K)".into()),
        },
        "map_len" => match args {
            [Value::Map(entries)] => Ok(Some(Value::Int(entries.len() as i64))),
            _ => Err("map_len expects (Map)".into()),
        },
        "map_remove" => match args {
            [Value::Map(entries), k] => {
                let mut out = entries.clone();
                if let Ok(i) = out.binary_search_by(|(ek, _)| key_cmp(ek, k)) {
                    out.remove(i);
                }
                Ok(Some(Value::Map(out)))
            }
            _ => Err("map_remove expects (Map, K)".into()),
        },
        "map_show" => match args {
            [m @ Value::Map(_)] => Ok(Some(Value::Str(m.display()))),
            _ => Err("map_show expects (Map)".into()),
        },

        // ---- Ordered Set ---------------------------------------------------
        "set_new" => match args {
            [] => Ok(Some(Value::Set(Vec::new()))),
            _ => Err("set_new expects no arguments".into()),
        },
        "set_add" => match args {
            [Value::Set(elems), x] => {
                let mut out = elems.clone();
                if let Err(i) = out.binary_search_by(|e| key_cmp(e, x)) {
                    out.insert(i, x.clone()); // adding an existing element is a no-op
                }
                Ok(Some(Value::Set(out)))
            }
            _ => Err("set_add expects (Set, T)".into()),
        },
        "set_has" => match args {
            [Value::Set(elems), x] => Ok(Some(Value::Bool(
                elems.binary_search_by(|e| key_cmp(e, x)).is_ok(),
            ))),
            _ => Err("set_has expects (Set, T)".into()),
        },
        "set_len" => match args {
            [Value::Set(elems)] => Ok(Some(Value::Int(elems.len() as i64))),
            _ => Err("set_len expects (Set)".into()),
        },
        "set_remove" => match args {
            [Value::Set(elems), x] => {
                let mut out = elems.clone();
                if let Ok(i) = out.binary_search_by(|e| key_cmp(e, x)) {
                    out.remove(i);
                }
                Ok(Some(Value::Set(out)))
            }
            _ => Err("set_remove expects (Set, T)".into()),
        },
        "set_show" => match args {
            [s @ Value::Set(_)] => Ok(Some(Value::Str(s.display()))),
            _ => Err("set_show expects (Set)".into()),
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
    fn record_literal_field_access_and_order_independence() {
        // Field order in the literal must not matter; `.field` reads by name.
        let src = "type P = { x: Int, y: Int }\n\
                   fn main() -> Int = { let p = P { y: 20, x: 10 }; p.x + p.y * 2 }";
        assert_eq!(run(src).display(), "50");
    }

    #[test]
    fn generic_record_field_through_type_param() {
        let src = "type Box[T] = { value: T, tag: Int }\n\
                   fn unwrap[T](b: Box[T]) -> T = b.value\n\
                   fn main() -> Int = { let b = Box { value: 7, tag: 1 }; unwrap(b) + b.tag }";
        assert_eq!(run(src).display(), "8");
    }

    #[test]
    fn functional_update_is_non_destructive() {
        let src = "type P = { x: Int, y: Int }\n\
                   fn main() -> Int = {\n\
                     let p = P { x: 1, y: 2 };\n\
                     let q = { p | y = 9 };\n\
                     p.y * 100 + q.y\n\
                   }";
        assert_eq!(run(src).display(), "209"); // p.y=2 unchanged, q.y=9
    }

    #[test]
    fn record_pattern_binds_fields() {
        let src = "type P = { a: Int, b: Int, c: Int }\n\
                   fn f(p: P) -> Int = match p { P { a: 0, b, c } => b + c, P { a, b, c } => a, }\n\
                   fn main() -> Int = f(P { a: 0, b: 3, c: 4 }) + f(P { a: 5, b: 0, c: 0 })";
        assert_eq!(run(src).display(), "12"); // 7 + 5
    }

    #[test]
    fn record_literals_not_confused_with_match_arms() {
        // Regression: `match` on a nullary ctor must parse as a match, not a
        // record literal `Nil { .. }`.
        let src = "type L = | Nil | Cons(Int, L)\n\
                   fn empty(xs: L) -> Int = match xs { Nil => 1, Cons(h, t) => 0, }\n\
                   fn main() -> Int = empty(Nil)";
        assert_eq!(run(src).display(), "1");
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

    // A representative value of a declared parameter/return type, for driving
    // builtins in the drift test.
    fn dummy(ty: &crate::ast::Ty) -> Value {
        use crate::ast::Ty::*;
        match ty {
            Int => Value::Int(0),
            Float => Value::Float(0.0),
            Bool => Value::Bool(false),
            Str => Value::Str(String::new()),
            Unit => Value::Unit,
            Named(n, _) if n == "Tensor" => Value::Tensor(crate::tensor::Tensor::zeros(&[1, 1])),
            Named(n, _) if n == "Bytes" => Value::Bytes(vec![0]),
            Named(n, args) if n == "Array" => {
                // A one-element array of the (concrete) element type, so generic
                // array builtins have something to index/return.
                Value::Array(vec![dummy(&args[0])])
            }
            // A one-entry map / set, so the generic builtins have something to
            // read. The drift test drives map/set builtins at Int key/element;
            // the contained key/element is deliberately DISTINCT from the
            // dummy(K) lookup key (which is `Int(0)`/empty `Str`) so a `remove`
            // is a no-op and the container keeps its element type (an emptied
            // container would lose its element types and fail the exact-type
            // drift check).
            Named(n, args) if n == "Map" => {
                Value::Map(vec![(distinct_dummy_key(&args[0]), dummy(&args[1]))])
            }
            Named(n, args) if n == "Set" => Value::Set(vec![distinct_dummy_key(&args[0])]),
            // A generic element position: any concrete value will do.
            Var(_) => Value::Int(0),
            other => panic!("drift test has no dummy for {}", crate::typeck::show(other)),
        }
    }

    // A dummy key/element value DISTINCT from `dummy(ty)`, so a `remove`/lookup
    // driven with `dummy(ty)` misses and leaves a one-element container intact.
    fn distinct_dummy_key(ty: &crate::ast::Ty) -> Value {
        use crate::ast::Ty::*;
        match ty {
            Str => Value::Str("x".to_string()),
            // Int keys, and a generic key var (drift drives these at Int): use 1
            // (dummy(Int)/dummy(Var) is 0).
            _ => Value::Int(1),
        }
    }

    // Map a runtime value back to its type, to check a builtin's declared return.
    fn value_ty(v: &Value) -> crate::ast::Ty {
        use crate::ast::Ty::*;
        match v {
            Value::Int(_) => Int,
            Value::Float(_) => Float,
            Value::Bool(_) => Bool,
            Value::Str(_) => Str,
            Value::Unit => Unit,
            Value::Tensor(_) => Named("Tensor".into(), vec![]),
            Value::Bytes(_) => Named("Bytes".into(), vec![]),
            Value::Array(xs) => Named(
                "Array".into(),
                vec![xs.first().map(value_ty).unwrap_or(Var("T".into()))],
            ),
            Value::Map(entries) => Named(
                "Map".into(),
                match entries.first() {
                    Some((k, v)) => vec![value_ty(k), value_ty(v)],
                    None => vec![Var("K".into()), Var("V".into())],
                },
            ),
            Value::Set(elems) => Named(
                "Set".into(),
                vec![elems.first().map(value_ty).unwrap_or(Var("T".into()))],
            ),
            Value::Data { ctor, .. } => Named(ctor.clone(), vec![]),
            Value::Closure(c) => {
                Fn(c.params.iter().map(|_| Unit).collect(), Box::new(Unit))
            }
        }
    }

    // --- Substitution machinery for the drift guard -----------------------
    //
    // To check a generic builtin's declared return type we infer how its type
    // variables are instantiated by unifying its declared *parameter* types
    // against the (concrete) types of the dummy arguments, apply the resulting
    // substitution to the declared return type, and require an EXACT structural
    // match against the actual returned value's type. A declared `Var` only
    // matches a `Var` if it stays genuinely free after substitution (e.g.
    // `array_new`'s element type, which has no argument to constrain it).

    use std::collections::HashMap;

    type Subst = HashMap<String, crate::ast::Ty>;

    // Unify a declared type against a concrete actual type, accumulating
    // variable bindings into `s`. Returns false on a structural mismatch.
    fn unify(declared: &crate::ast::Ty, actual: &crate::ast::Ty, s: &mut Subst) -> bool {
        use crate::ast::Ty::*;
        match (declared, actual) {
            (Var(v), _) => match s.get(v) {
                Some(bound) => bound == actual,
                None => {
                    s.insert(v.clone(), actual.clone());
                    true
                }
            },
            (Named(n, a), Named(m, b)) => {
                n == m && a.len() == b.len() && a.iter().zip(b).all(|(x, y)| unify(x, y, s))
            }
            (Fn(p1, r1), Fn(p2, r2)) => {
                p1.len() == p2.len()
                    && p1.iter().zip(p2).all(|(x, y)| unify(x, y, s))
                    && unify(r1, r2, s)
            }
            (Int, Int) | (Float, Float) | (Bool, Bool) | (Str, Str) | (Unit, Unit) => true,
            _ => false,
        }
    }

    // Apply a substitution to a type, leaving unbound variables in place.
    fn apply(ty: &crate::ast::Ty, s: &Subst) -> crate::ast::Ty {
        use crate::ast::Ty::*;
        match ty {
            Var(v) => s.get(v).cloned().unwrap_or_else(|| ty.clone()),
            Named(n, a) => Named(n.clone(), a.iter().map(|t| apply(t, s)).collect()),
            Fn(p, r) => Fn(
                p.iter().map(|t| apply(t, s)).collect(),
                Box::new(apply(r, s)),
            ),
            Int | Float | Bool | Str | Unit => ty.clone(),
        }
    }

    // Exact structural type equality, with the single relaxation that a
    // declared free `Var` matches an actual free `Var` (the element type is
    // genuinely unconstrained, e.g. `array_new`'s `Array[T]`).
    fn ty_exact(expected: &crate::ast::Ty, actual: &crate::ast::Ty) -> bool {
        use crate::ast::Ty::*;
        match (expected, actual) {
            (Var(_), Var(_)) => true,
            (Named(n, a), Named(m, b)) => {
                n == m && a.len() == b.len() && a.iter().zip(b).all(|(x, y)| ty_exact(x, y))
            }
            (Fn(p1, r1), Fn(p2, r2)) => {
                p1.len() == p2.len()
                    && p1.iter().zip(p2).all(|(x, y)| ty_exact(x, y))
                    && ty_exact(r1, r2)
            }
            (Int, Int) | (Float, Float) | (Bool, Bool) | (Str, Str) | (Unit, Unit) => true,
            _ => false,
        }
    }

    #[test]
    fn declared_builtins_implemented_with_matching_signature() {
        // Drift guard, both directions: every builtin in the shared table must
        // be implemented AND return a value of its declared type when driven
        // with correctly-typed arguments. The return type is checked by first
        // inferring the type-variable substitution from the (concrete) dummy
        // argument types, applying it to the declared return type, and then
        // requiring an exact match. This catches a generic builtin declared to
        // return `T` (e.g. `array_get` over `Array[Int]` -> `Int`) that instead
        // returns the wrong concrete type, while still letting a genuinely
        // unconstrained return var (e.g. `array_new` -> `Array[T]`) pass.
        for (name, params, ret) in crate::builtins::signatures() {
            let args: Vec<Value> = params.iter().map(dummy).collect();
            match builtin(name, &args) {
                Ok(Some(v)) => {
                    // Infer the substitution from declared params vs. dummy types.
                    let mut s: Subst = HashMap::new();
                    for (p, a) in params.iter().zip(&args) {
                        assert!(
                            unify(p, &value_ty(a), &mut s),
                            "builtin `{}`: dummy arg type does not unify with declared param {}",
                            name,
                            crate::typeck::show(p)
                        );
                    }
                    let expected = apply(&ret, &s);
                    assert!(
                        ty_exact(&expected, &value_ty(&v)),
                        "builtin `{}` returned {:?} (type {}), declared {} (instantiated {})",
                        name,
                        v.display(),
                        crate::typeck::show(&value_ty(&v)),
                        crate::typeck::show(&ret),
                        crate::typeck::show(&expected)
                    );
                }
                Ok(None) => panic!("builtin `{}` is declared but not implemented in interp", name),
                Err(e) => panic!("builtin `{}` errored on valid dummy args: {}", name, e),
            }
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

    // Lex -> parse -> typeck -> interp, returning `main`'s runtime error message.
    fn run_err(src: &str) -> String {
        let toks = lexer::lex(src).expect("lex");
        let prog = parser::parse(toks).expect("parse");
        typeck::check(&prog).expect("typeck");
        let interp = Interp::new(&prog).expect("interp::new");
        interp.run_main().expect_err("expected a runtime error")
    }

    #[test]
    fn bytes_build_get_set_push_and_roundtrip() {
        // from_str -> push -> set (in place) -> len/get -> to_str round-trip.
        let v = run(r#"
            fn main() -> Int = {
                let b = bytes_from_str("Hi");
                let b2 = bytes_push(b, 33);
                let b3 = bytes_set(b2, 0, 104);
                bytes_len(b3) + bytes_get(b3, 2)
            }
        "#);
        // len 3 + byte at index 2 ('!') = 33  -> 36
        assert!(matches!(v, Value::Int(36)), "got {}", v.display());

        let s = run(r#"fn main() -> String = bytes_to_str(bytes_from_str("hello"))"#);
        assert_eq!(s.display(), "hello");
    }

    #[test]
    fn bytes_canonical_display_and_equality() {
        // The canonical rendering: `Bytes[00 ab ff]` (lowercase hex, empty `[]`).
        let v = run(r#"
            fn main() -> Bytes = {
                let b = bytes_push(bytes_push(bytes_push(bytes_new(), 0), 171), 255);
                b
            }
        "#);
        assert_eq!(v.display(), "Bytes[00 ab ff]");
        assert_eq!(run("fn main() -> Bytes = bytes_new()").display(), "Bytes[]");

        // `==` compares contents; a Bytes never equals a Str with the same bytes
        // (distinct type tag — the type checker also rejects that comparison).
        let eq = run(r#"fn main() -> Bool = bytes_from_str("ok") == bytes_from_str("ok")"#);
        assert!(matches!(eq, Value::Bool(true)));
        let ne = run(r#"fn main() -> Bool = bytes_from_str("ok") == bytes_from_str("no")"#);
        assert!(matches!(ne, Value::Bool(false)));
        // Distinct tags: a Bytes and a Str with identical bytes are NOT equal.
        assert!(!values_equal(
            &Value::Bytes(b"hi".to_vec()),
            &Value::Str("hi".to_string())
        ));
    }

    #[test]
    fn bytes_error_cases_are_clean_runtime_errors() {
        // Index out of range on get / set.
        assert!(run_err("fn main() -> Int = bytes_get(bytes_from_str(\"ab\"), 5)")
            .contains("out of range"));
        assert!(run_err("fn main() -> Bytes = bytes_set(bytes_from_str(\"ab\"), 9, 1)")
            .contains("out of range"));
        // Byte value outside 0..255 on set / push is rejected.
        assert!(run_err("fn main() -> Bytes = bytes_push(bytes_new(), 300)")
            .contains("out of range 0..255"));
        assert!(run_err("fn main() -> Bytes = bytes_set(bytes_from_str(\"a\"), 0, -1)")
            .contains("out of range 0..255"));
        // Invalid UTF-8 in to_str.
        assert!(
            run_err("fn main() -> String = bytes_to_str(bytes_push(bytes_new(), 255))")
                .contains("invalid UTF-8")
        );
    }

    // ---- Map / Set (interpreter oracle behavior) ------------------------

    #[test]
    fn map_ordered_display_and_replacement() {
        // Out-of-order inserts render in ASCENDING key order; an insert of an
        // existing key REPLACES its value (does not duplicate it).
        let v = run(r#"
            fn main() -> Map[Int, Int] = {
                let m = map_insert(map_insert(map_insert(map_new(), 30, 3), 10, 1), 20, 2);
                map_insert(m, 10, 111)
            }
        "#);
        assert_eq!(v.display(), "Map[10: 111, 20: 2, 30: 3]");
        assert_eq!(run("fn main() -> Map[Int, Int] = map_new()").display(), "Map[]");
    }

    #[test]
    fn map_get_or_total_read_and_default() {
        // Present key returns its value; absent key returns the default (no
        // Option type exists, so the read is total).
        let present = run(r#"fn main() -> Int = map_get_or(map_insert(map_new(), 7, 42), 7, -1)"#);
        assert!(matches!(present, Value::Int(42)));
        let absent = run(r#"fn main() -> Int = map_get_or(map_insert(map_new(), 7, 42), 9, -1)"#);
        assert!(matches!(absent, Value::Int(-1)));
    }

    #[test]
    fn map_has_len_remove() {
        let has = run(r#"fn main() -> Bool = map_has(map_insert(map_new(), 5, 1), 5)"#);
        assert!(matches!(has, Value::Bool(true)));
        let missing = run(r#"fn main() -> Bool = map_has(map_insert(map_new(), 5, 1), 6)"#);
        assert!(matches!(missing, Value::Bool(false)));
        let len = run(r#"
            fn main() -> Int = {
                let m = map_insert(map_insert(map_insert(map_new(), 1, 1), 2, 2), 1, 9);
                map_len(m)
            }
        "#);
        assert!(matches!(len, Value::Int(2)), "got {}", len.display()); // key 1 replaced, not added
        let removed = run(r#"
            fn main() -> Map[Int, Int] =
                map_remove(map_insert(map_insert(map_new(), 1, 10), 2, 20), 1)
        "#);
        assert_eq!(removed.display(), "Map[2: 20]");
        // Removing an absent key is a no-op.
        let noop = run(r#"fn main() -> Int = map_len(map_remove(map_insert(map_new(), 1, 1), 99))"#);
        assert!(matches!(noop, Value::Int(1)));
    }

    #[test]
    fn map_str_keys_sorted_and_equality() {
        let v = run(r#"
            fn main() -> Map[String, Int] =
                map_insert(map_insert(map_insert(map_new(), "pear", 3), "apple", 1), "fig", 2)
        "#);
        assert_eq!(v.display(), "Map[apple: 1, fig: 2, pear: 3]");
        // Two maps built in different insertion orders are equal (sorted contents).
        let eq = run(r#"
            fn main() -> Bool = {
                let a = map_insert(map_insert(map_new(), 1, 10), 2, 20);
                let b = map_insert(map_insert(map_new(), 2, 20), 1, 10);
                a == b
            }
        "#);
        assert!(matches!(eq, Value::Bool(true)));
        let ne = run(r#"
            fn main() -> Bool = {
                let a = map_insert(map_new(), 1, 10);
                let b = map_insert(map_new(), 1, 99);
                a == b
            }
        "#);
        assert!(matches!(ne, Value::Bool(false)));
    }

    #[test]
    fn set_ordered_dedup_display_and_ops() {
        // Out-of-order, duplicate adds render sorted & deduped; add of an
        // existing element is a no-op.
        let v = run(r#"
            fn main() -> Set[Int] =
                set_add(set_add(set_add(set_add(set_new(), 30), 10), 20), 10)
        "#);
        assert_eq!(v.display(), "Set[10, 20, 30]");
        assert_eq!(run("fn main() -> Set[Int] = set_new()").display(), "Set[]");
        let len = run(r#"fn main() -> Int = set_len(set_add(set_add(set_new(), 1), 1))"#);
        assert!(matches!(len, Value::Int(1))); // dedup
        let has = run(r#"fn main() -> Bool = set_has(set_add(set_new(), 5), 5)"#);
        assert!(matches!(has, Value::Bool(true)));
        let removed = run(r#"fn main() -> Set[Int] = set_remove(set_add(set_add(set_new(), 1), 2), 1)"#);
        assert_eq!(removed.display(), "Set[2]");
        // Str set, sorted; equality independent of insertion order.
        let strs = run(r#"
            fn main() -> Set[String] = set_add(set_add(set_add(set_new(), "b"), "a"), "c")
        "#);
        assert_eq!(strs.display(), "Set[a, b, c]");
        let eq = run(r#"
            fn main() -> Bool = {
                let a = set_add(set_add(set_new(), 1), 2);
                let b = set_add(set_add(set_new(), 2), 1);
                a == b
            }
        "#);
        assert!(matches!(eq, Value::Bool(true)));
    }

    #[test]
    fn tensor_equality_is_reflexive_with_nan() {
        // A NaN-containing tensor must still equal itself (`==` is reflexive even
        // though NaN != NaN in IEEE).
        let same = run(r#"
            fn main() -> Bool = {
                let a = tensor_set(tensor_zeros(1, 1), 0, 0, 0.0 / 0.0);
                a == a
            }
        "#);
        assert!(matches!(same, Value::Bool(true)));
    }

    #[test]
    fn closure_captures_a_variable() {
        // The lambda `\x -> x + n` must capture `n` from the enclosing scope.
        let v = run(r#"
            fn apply1(f: (Int) -> Int, x: Int) -> Int = f(x)
            fn main() -> Int = { let n = 100; apply1(\x -> x + n, 5) }
        "#);
        assert!(matches!(v, Value::Int(105)), "got {}", v.display());
    }

    #[test]
    fn map_with_a_lambda_closure() {
        // map with a closure capturing `n`, summed: (1+10)+(2+10) = 23.
        let v = run(r#"
            type L = | Nil | Cons(Int, L)
            fn map(f: (Int) -> Int, xs: L) -> L =
              match xs { Nil => Nil, Cons(h, r) => Cons(f(h), map(f, r)), }
            fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }
            fn main() -> Int = {
                let n = 10;
                sum(map(\x -> x + n, Cons(1, Cons(2, Nil))))
            }
        "#);
        assert!(matches!(v, Value::Int(23)), "got {}", v.display());
    }

    #[test]
    fn filter_with_a_lambda() {
        // Keep odd numbers from [1,2,3,4], then sum: 1 + 3 = 4.
        let v = run(r#"
            type L = | Nil | Cons(Int, L)
            fn filter(p: (Int) -> Bool, xs: L) -> L =
              match xs {
                Nil => Nil,
                Cons(h, r) => if p(h) { Cons(h, filter(p, r)) } else { filter(p, r) },
              }
            fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }
            fn main() -> Int =
              sum(filter(\x -> x % 2 == 1, Cons(1, Cons(2, Cons(3, Cons(4, Nil))))))
        "#);
        assert!(matches!(v, Value::Int(4)), "got {}", v.display());
    }

    #[test]
    fn fold_with_a_lambda() {
        // Left fold (+) over [1,2,3,4] starting at 0 = 10.
        let v = run(r#"
            type L = | Nil | Cons(Int, L)
            fn fold(f: (Int, Int) -> Int, acc: Int, xs: L) -> Int =
              match xs { Nil => acc, Cons(h, r) => fold(f, f(acc, h), r), }
            fn main() -> Int =
              fold(\(a: Int, b: Int) -> a + b, 0, Cons(1, Cons(2, Cons(3, Cons(4, Nil)))))
        "#);
        assert!(matches!(v, Value::Int(10)), "got {}", v.display());
    }

    #[test]
    fn function_passed_by_name_runs() {
        // A top-level function name handed to a HOF as a value.
        let v = run(r#"
            type L = | Nil | Cons(Int, L)
            fn dbl(x: Int) -> Int = x * 2
            fn map(f: (Int) -> Int, xs: L) -> L =
              match xs { Nil => Nil, Cons(h, r) => Cons(f(h), map(f, r)), }
            fn sum(xs: L) -> Int = match xs { Nil => 0, Cons(h, r) => h + sum(r), }
            fn main() -> Int = sum(map(dbl, Cons(1, Cons(2, Cons(3, Nil)))))
        "#);
        assert!(matches!(v, Value::Int(12)), "got {}", v.display());
    }

    #[test]
    fn immediately_applied_lambda() {
        let v = run("fn main() -> Int = (\\(x: Int) -> x * x)(7)");
        assert!(matches!(v, Value::Int(49)), "got {}", v.display());
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

    // ---- self-tail-call optimization -----------------------------------

    #[test]
    fn deep_tail_accumulator_no_recursion_limit() {
        // A tail-recursive accumulator 1,000,000 deep — FAR past MAX_CALL_DEPTH.
        // With self-tail-call elimination this runs as a loop in constant stack,
        // so it must NOT hit the depth guard and must return 500000500000. Run on
        // the DEFAULT (small) test-thread stack to prove the stack stays flat.
        assert!(MAX_CALL_DEPTH < 1_000_000);
        let v = run(
            "fn go(n: Int, acc: Int) -> Int = if n == 0 { acc } else { go(n - 1, acc + n) }\n\
             fn main() -> Int = go(1000000, 0)",
        );
        assert!(matches!(v, Value::Int(500000500000)), "got {}", v.display());
    }

    #[test]
    fn deep_tail_call_in_match_no_recursion_limit() {
        // A self-tail-call in a `match` arm body (tail position flows through
        // every arm), 1,000,000 deep. Scrutinee is a small flat ADT so each
        // iteration's clone is O(1); TCO gives constant stack.
        let v = run(
            "type Step = | Done | More(Int)\n\
             fn step(n: Int) -> Step = if n == 0 { Done } else { More(n) }\n\
             fn go(n: Int, acc: Int) -> Int = \
                match step(n) { Done => acc, More(k) => go(k - 1, acc + k), }\n\
             fn main() -> Int = go(1000000, 0)",
        );
        assert!(matches!(v, Value::Int(500000500000)), "got {}", v.display());
    }

    #[test]
    fn tail_call_arg_references_other_param() {
        // Swap-style tail call where each new argument reads the OTHER (old)
        // parameter: all args must be evaluated BEFORE any param is rebound.
        // gcd by subtraction is a clean check (gcd(48, 36) = 12).
        let v = run(
            "fn gcd(a: Int, b: Int) -> Int = \
                if b == 0 { a } else { if a < b { gcd(b, a) } else { gcd(a - b, b) } }\n\
             fn main() -> Int = gcd(48, 36)",
        );
        assert!(matches!(v, Value::Int(12)), "got {}", v.display());
    }

    #[test]
    fn non_tail_recursion_unchanged() {
        // A NON-tail call (the recursive call is an operand of `+`, not in tail
        // position) is NOT turned into a loop: it still uses a real call frame.
        // It must keep producing the right answer. Run on a large-stack thread
        // (as the CLI does) since non-tail recursion consumes native stack per
        // call; the TCO transform does not change that.
        let src = "fn sumto(n: Int) -> Int = if n == 0 { 0 } else { n + sumto(n - 1) }\n\
                   fn main() -> Int = sumto(2000)";
        let v = std::thread::Builder::new()
            .stack_size(1 << 30)
            .spawn(move || {
                let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
                Interp::new(&prog).unwrap().run_main().unwrap()
            })
            .unwrap()
            .join()
            .expect("non-tail recursion must not crash at this depth");
        assert!(matches!(v, Value::Int(2001000)), "got {}", v.display());
    }
}
