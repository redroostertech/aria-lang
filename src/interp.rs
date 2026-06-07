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
    /// A dense, immutable float vector / embedding (`Vector`). A flat buffer of
    /// `f64`. Distinct type tag from `Array[Float]`, so `==`/display never
    /// conflate the two even with identical elements.
    Vector(Vec<f64>),
    /// An opaque AI-runtime tensor handle, built and queried via builtins.
    Tensor(crate::tensor::Tensor),
    /// A first-class function value. A lambda captures the environment in which
    /// it was created; a bare top-level function name becomes a closure with an
    /// empty captured environment. Boxed (via `Arc`) so adding closures does not
    /// enlarge `Value` and blow the recursive-interpreter stack.
    Closure(std::sync::Arc<ClosureData>),
    /// A REVERSE-MODE AUTODIFF tracing scalar. Holds an index into the active
    /// `grad` tape (a Wengert list); flows everywhere a `Float` would *inside* a
    /// `grad(f, x)` evaluation of `f`. It NEVER exists outside a `grad` call —
    /// normal programs never construct or observe one, so non-grad evaluation is
    /// completely unaffected (the scalar/vector ops only branch into the tracing
    /// path when an operand is already `Tracing`/`TracingVec`).
    Tracing(usize),
    /// A reverse-mode tracing Vector: a dense vector of tape-node ids (one per
    /// coordinate). The `Vector` argument that `grad` feeds to `f` is one of
    /// these, so `vec_get`/`vec_dot`/… on it produce `Tracing` scalars. Like
    /// `Tracing`, it only ever exists inside a `grad` call.
    TracingVec(Vec<usize>),
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
            Value::Vector(xs) => render_vector(xs),
            Value::Closure(c) => {
                format!("<closure/{}>", c.params.len())
            }
            // Tracing values only ever exist transiently inside a `grad` call
            // and never reach user-visible display; render them descriptively
            // for diagnostics if one ever surfaces in an error message.
            Value::Tracing(_) => "<grad scalar>".to_string(),
            Value::TracingVec(ids) => format!("<grad vector/{}>", ids.len()),
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

/// The ONE canonical textual rendering of a `Vector` value, emitted byte-for-byte
/// identically by the interpreter and the native backend: the literal `Vector[`,
/// then each element via the SAME shortest-round-trip float formatter used for a
/// scalar `Float` (`format!("{}", f)` here; `aria_fmt_float` in native), comma +
/// space separated, then `]`. Empty is `Vector[]`.
pub fn render_vector(xs: &[f64]) -> String {
    let inner: Vec<String> = xs.iter().map(|f| format!("{}", f)).collect();
    format!("Vector[{}]", inner.join(", "))
}

// ===========================================================================
// Reverse-mode automatic differentiation — the `grad` builtin's tape.
//
// A `Tape` is a Wengert list: an append-only `Vec<Node>` where each node holds
// its computed forward `value` and, for each parent it depends on, the *local
// partial derivative* of this node w.r.t. that parent. `grad(f, x)` evaluates
// `f` over a tracing Vector (one leaf node per input coordinate); every
// differentiable scalar/vector op that sees a tracing operand pushes a node
// recording its inputs and local partials. After the forward trace yields a
// single scalar output node, we seed its adjoint to 1.0 and sweep the tape in
// reverse, accumulating `parent.adj += node.adj * local_partial` (the standard
// vector-Jacobian product). The gradient is the vector of leaf adjoints.
//
// The tape is owned by the `grad` call (a thread-local set only for the
// duration of `f`'s evaluation) — the mutation is scoped entirely to the
// builtin and never observable from Aria, mirroring how `matmul` hides its
// mutable out-buffer. Outside a `grad` call the thread-local is `None` and the
// tracing path is never entered.
// ===========================================================================

/// One node of the reverse-mode tape. `value` is the forward result; `parents`
/// pairs each input node id with the local partial ∂(this)/∂(that input).
struct TapeNode {
    value: f64,
    parents: Vec<(usize, f64)>,
}

/// The append-only Wengert list for one `grad` call.
struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    fn new() -> Self {
        Tape { nodes: Vec::new() }
    }
    /// Push a leaf (input) node holding a concrete value, with no parents.
    fn leaf(&mut self, value: f64) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TapeNode { value, parents: Vec::new() });
        id
    }
    /// Push an interior node: its forward value and its (parent, local-partial)
    /// list. Returns the new node id.
    fn op(&mut self, value: f64, parents: Vec<(usize, f64)>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TapeNode { value, parents });
        id
    }
    fn value(&self, id: usize) -> f64 {
        self.nodes[id].value
    }
    /// Reverse sweep from `output` (seeded adjoint 1.0). Returns the adjoint of
    /// every node, indexed by node id.
    fn backward(&self, output: usize) -> Vec<f64> {
        let mut adj = vec![0.0_f64; self.nodes.len()];
        adj[output] = 1.0;
        // Nodes are in topological order (a node only references earlier ids),
        // so a single reverse pass over the ids suffices.
        for id in (0..self.nodes.len()).rev() {
            let a = adj[id];
            if a == 0.0 {
                continue;
            }
            for &(p, partial) in &self.nodes[id].parents {
                adj[p] += a * partial;
            }
        }
        adj
    }
}

thread_local! {
    /// The active `grad` tape, present only while evaluating the function `f`
    /// passed to `grad`. `None` everywhere else, so the tracing branches in
    /// `eval_binary`/the vector builtins are inert for normal programs.
    static GRAD_TAPE: std::cell::RefCell<Option<Tape>> = const { std::cell::RefCell::new(None) };

    /// The active OUTPUT-CAPTURE buffer. `None` (the default) means the `print_*`
    /// builtins write to this process's stdout EXACTLY as before — normal `aria
    /// run` and every existing test/example is unaffected. `Some(buf)` (set only
    /// for the duration of a `run_main_capturing` call) makes the same builtins
    /// APPEND their formatted line (identical formatting + trailing newline) to
    /// `buf` instead of touching stdout, so a caller (the agent loop / benchmark)
    /// can observe what a program PRINTED. Thread-local so it never crosses the
    /// large-stack worker thread the interpreter runs on, and so capture is
    /// strictly scoped to the capturing run.
    static OUTPUT_CAPTURE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };

    /// The active RUNTIME CALL STACK, present only while a `run_main`-family call
    /// is in flight. `None` (the default) means stack tracking is OFF, so normal
    /// `aria run` SUCCESS paths and non-interpreter code pay ZERO overhead and see
    /// no behavior change. `Some(stack)` (installed for the duration of a run)
    /// makes every user-function entry PUSH a `{function, line}` frame and every
    /// SUCCESSFUL return POP it. On an ERROR the frames are deliberately LEFT on
    /// the stack as they unwind, so when the run returns `Err` the stack holds the
    /// exact call chain from `main` down to the function that trapped — that is the
    /// stack trace. Consecutive identical frames (a non-tail self-recursive call)
    /// are collapsed at render time so runaway/deep recursion does not produce a
    /// million-line trace. Thread-local so it never crosses the large-stack worker
    /// thread the interpreter runs on and stays strictly scoped to one run.
    static CALL_STACK: std::cell::RefCell<Option<Vec<Frame>>> = const { std::cell::RefCell::new(None) };
}

/// One entry in a runtime stack trace: the called function's name plus the
/// precise CALL-SITE location (the `(line, col)` of the call expression that
/// entered this function). `call_line == 0` means the call site is unknown (a
/// compiler-synthesized call with no source span, or the synthetic entry into
/// `main`), in which case rendering falls back to the function's DEFINITION
/// line `def_line`. `def_line == 0` marks a compiler-generated callee (a trait
/// dispatcher / lowered impl method / closure value) that has no single source
/// definition line.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub function: String,
    /// 1-based DEFINITION line of the callee (`fn` keyword); 0 = generated.
    pub def_line: usize,
    /// 1-based CALL-SITE line (where this function was called); 0 = unknown.
    pub call_line: usize,
    /// 1-based CALL-SITE column; meaningful only when `call_line != 0`.
    pub call_col: usize,
}

/// A structured runtime error: the trap message plus the call stack at the point
/// of failure (most-recent call first). Produced by `run_main_traced` /
/// `run_main_capturing_traced` so the agent loop and diagnostics can consume the
/// frames, and rendered to a human string by `RuntimeError::render` for
/// `aria run` / `aria agent` output.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeError {
    pub message: String,
    /// Frames, MOST-RECENT call first (the function that trapped is `frames[0]`,
    /// `main` is last). Empty if the error occurred before any user frame was
    /// entered (e.g. "no `main` function found").
    pub frames: Vec<Frame>,
}

impl RuntimeError {
    /// Render the error followed by an indented, most-recent-first stack trace.
    /// Each frame reports the PRECISE CALL SITE (`line:col`) where that function
    /// was called, e.g.
    ///   runtime error: division by zero
    ///     at `inner` (line 12:5)
    ///     at `middle` (line 8:3)
    ///     at `main` (line 3)
    /// When the call site is unknown (the synthetic entry into `main`, or a
    /// synthesized call with no source span) the frame falls back to the
    /// function's DEFINITION line (`line N`). A compiler-generated callee with
    /// neither a call site nor a definition line prints `(generated)`.
    pub fn render(&self) -> String {
        let mut s = format!("runtime error: {}", self.message);
        for f in &self.frames {
            if f.call_line != 0 {
                s.push_str(&format!(
                    "\n  at `{}` (line {}:{})",
                    f.function, f.call_line, f.call_col
                ));
            } else if f.def_line != 0 {
                s.push_str(&format!("\n  at `{}` (line {})", f.function, f.def_line));
            } else {
                s.push_str(&format!("\n  at `{}` (generated)", f.function));
            }
        }
        s
    }
}

/// Install a fresh (empty) call stack for the duration of `f`, capturing the
/// frames left behind on the error path into a `RuntimeError`. Restores the
/// previous stack state (`None` normally) on the way out, success OR error, so a
/// subsequent run is unaffected. Consecutive identical frames are collapsed.
fn with_call_stack<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, RuntimeError> {
    let prev = CALL_STACK.with(|c| c.borrow_mut().replace(Vec::new()));
    let result = f();
    let stack = CALL_STACK.with(|c| std::mem::replace(&mut *c.borrow_mut(), prev));
    match result {
        Ok(v) => Ok(v),
        Err(message) => {
            let raw = stack.unwrap_or_default();
            // Most-recent call first: the deepest (last-pushed) frame is the one
            // that trapped. Collapse runs of identical consecutive frames so a
            // non-tail self-recursion does not explode the trace.
            let mut frames: Vec<Frame> = Vec::new();
            for fr in raw.into_iter().rev() {
                if frames.last().map(|p| p == &fr).unwrap_or(false) {
                    continue;
                }
                frames.push(fr);
            }
            Err(RuntimeError { message, frames })
        }
    }
}

/// Push a user-function frame onto the active call stack (no-op when tracking is
/// off). Called on entry to a user function in the interpreter. `def_line` is the
/// callee's definition line (0 = generated); `call_span` is the span of the call
/// EXPRESSION at the call site (its start `(line, col)` is recorded; a
/// `Span::none` call span leaves the call site unknown, falling back to the
/// definition line when rendered).
fn push_frame(function: &str, def_line: usize, call_span: crate::ast::Span) {
    CALL_STACK.with(|c| {
        if let Some(stack) = c.borrow_mut().as_mut() {
            let (call_line, call_col) = if call_span.is_none() {
                (0, 0)
            } else {
                (call_span.start_line as usize, call_span.start_col as usize)
            };
            stack.push(Frame {
                function: function.to_string(),
                def_line,
                call_line,
                call_col,
            });
        }
    });
}

/// Pop the top frame on a SUCCESSFUL return (no-op when tracking is off). The
/// error path intentionally does NOT pop, leaving the chain in place for the
/// trace.
fn pop_frame() {
    CALL_STACK.with(|c| {
        if let Some(stack) = c.borrow_mut().as_mut() {
            stack.pop();
        }
    });
}

/// Emit one already-formatted output LINE from a `print_*` builtin: append it to
/// the active capture buffer when capturing, else write it (plus a newline) to
/// stdout exactly as `println!` did before. The single choke point that keeps
/// the captured text byte-for-byte identical to the printed text.
fn emit_line(line: &str) {
    OUTPUT_CAPTURE.with(|c| {
        let mut b = c.borrow_mut();
        match b.as_mut() {
            Some(buf) => {
                buf.push_str(line);
                buf.push('\n');
            }
            None => {
                println!("{}", line);
            }
        }
    });
}

/// Read the forward value of a tracing scalar from the active tape. Errors
/// cleanly if there is no active tape (should never happen for a well-formed
/// `Tracing` value, which only exists inside a `grad` call).
fn tracing_value(id: usize) -> Result<f64, String> {
    GRAD_TAPE.with(|t| {
        t.borrow()
            .as_ref()
            .map(|tape| tape.value(id))
            .ok_or_else(|| "grad: tracing value used outside a `grad` call".to_string())
    })
}

/// Coerce a value to a tape node id, materializing a `Float`/`Int` as a fresh
/// constant leaf so a mixed op (`tracing * constant`) records correctly. Any
/// other value type inside a differentiated computation is a clean error.
fn as_node(v: &Value) -> Result<usize, String> {
    match v {
        Value::Tracing(id) => Ok(*id),
        Value::Float(f) => GRAD_TAPE.with(|t| {
            let mut b = t.borrow_mut();
            let tape = b
                .as_mut()
                .ok_or_else(|| "grad: internal tape missing".to_string())?;
            Ok(tape.leaf(*f))
        }),
        Value::Int(n) => GRAD_TAPE.with(|t| {
            let mut b = t.borrow_mut();
            let tape = b
                .as_mut()
                .ok_or_else(|| "grad: internal tape missing".to_string())?;
            Ok(tape.leaf(*n as f64))
        }),
        other => Err(format!(
            "grad: unsupported operation on a differentiated value of type {}",
            other.display()
        )),
    }
}

/// Record a unary differentiable op: result value `v`, with local partial
/// `d_dx` of the result w.r.t. the single input `x`. Returns a `Tracing` value.
fn record_unary(x: usize, v: f64, d_dx: f64) -> Value {
    GRAD_TAPE.with(|t| {
        let mut b = t.borrow_mut();
        let tape = b.as_mut().expect("tape present in tracing op");
        Value::Tracing(tape.op(v, vec![(x, d_dx)]))
    })
}

/// Record a binary differentiable op: result value `v`, with local partials
/// `da`/`db` of the result w.r.t. inputs `a`/`b`. Returns a `Tracing` value.
fn record_binary(a: usize, b: usize, v: f64, da: f64, db: f64) -> Value {
    GRAD_TAPE.with(|t| {
        let mut bt = t.borrow_mut();
        let tape = bt.as_mut().expect("tape present in tracing op");
        Value::Tracing(tape.op(v, vec![(a, da), (b, db)]))
    })
}

/// Record an op with an arbitrary parent/partial list (used by `vec_dot`).
fn record_many(parents: Vec<(usize, f64)>, v: f64) -> Value {
    GRAD_TAPE.with(|t| {
        let mut b = t.borrow_mut();
        let tape = b.as_mut().expect("tape present in tracing op");
        Value::Tracing(tape.op(v, parents))
    })
}

/// True iff the binary float op `op` on `(l, r)` involves a tracing operand and
/// so must be recorded on the tape. Only the four arithmetic ops are
/// differentiable; comparisons/logic on tracing scalars are unsupported.
fn is_tracing(v: &Value) -> bool {
    matches!(v, Value::Tracing(_))
}

/// Record one of the four differentiable Float arithmetic ops on tracing
/// operands. Returns the resulting `Tracing` value, or a clean error for a
/// non-differentiable operator. The caller guarantees at least one operand is
/// `Tracing`.
fn tracing_float_op(op: BinOp, l: &Value, r: &Value) -> Result<Value, String> {
    let a = as_node(l)?;
    let b = as_node(r)?;
    let (av, bv) = (tracing_value(a)?, tracing_value(b)?);
    match op {
        // (a+b)' : da=1, db=1
        BinOp::Add => Ok(record_binary(a, b, av + bv, 1.0, 1.0)),
        // (a-b)' : da=1, db=-1
        BinOp::Sub => Ok(record_binary(a, b, av - bv, 1.0, -1.0)),
        // (a*b)' : da=b, db=a
        BinOp::Mul => Ok(record_binary(a, b, av * bv, bv, av)),
        // (a/b)' : da=1/b, db=-a/b^2
        BinOp::Div => {
            if bv == 0.0 {
                return Err("grad: division by zero in a differentiated value".into());
            }
            Ok(record_binary(a, b, av / bv, 1.0 / bv, -av / (bv * bv)))
        }
        other => Err(format!(
            "grad: unsupported operation `{:?}` on a differentiated value",
            other
        )),
    }
}

/// Sum of elementwise products of two equal-length float slices (the caller
/// guarantees equal lengths). The native backend uses the identical left-to-right
/// summation order so the float result is byte-for-byte identical.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (a, b) in x.iter().zip(y) {
        acc += a * b;
    }
    acc
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

/// Maximum Aria function-call nesting (NON-tail; tail calls are eliminated and
/// never count). This is the real bound that turns runaway recursion into a clean
/// `Err` instead of crashing the process. For the guard to win the race against a
/// native stack overflow, the interpreter must run on a thread whose stack holds
/// this many frames: each Aria call consumes native stack, and worst-case
/// non-tail recursion costs ~26 KiB/frame in a DEBUG (unoptimized) build, so 100k
/// frames need ~2.6 GiB. The CLI therefore runs the interpreter on a 4 GiB stack
/// (`main::INTERP_STACK_SIZE`) so this guard fires cleanly in BOTH debug and
/// release before any `stack overflow`/`Abort trap`.
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
        // Push the `main` frame so a trace bottoms out at `main` (run_main is the
        // entry point and does not go through the `ExprKind::Call` push path).
        // `main` has no call SITE (it is the synthetic entry), so its frame uses
        // a none call span and renders with its definition line. No-op when stack
        // tracking is off.
        push_frame("main", main.line, crate::ast::Span::none());
        let mut scope: Scope = vec![HashMap::new()];
        let result = self.eval_fn_body("main", &main.body, &mut scope);
        if result.is_ok() {
            pop_frame();
        }
        result
    }

    /// Run `main` with RUNTIME STACK TRACKING on, returning a structured
    /// `RuntimeError` (message + call frames, most-recent first) on failure. The
    /// SUCCESS path is byte-for-byte identical to `run_main` (stack tracking only
    /// records frames; it never alters values or output), so `aria run` success
    /// output is unchanged. Used by `aria run` and the agent loop's error path.
    pub fn run_main_traced(&self) -> Result<Value, RuntimeError> {
        with_call_stack(|| self.run_main())
    }

    /// `run_main_capturing` with stack tracking on. Returns `(value, captured
    /// stdout)` on success, or a structured `RuntimeError` (with the captured
    /// stdout discarded — the partial output before the trap is not the signal we
    /// surface) on failure. Used by the agent loop to RUN a clean-checking program
    /// and feed back the runtime error + trace if it traps.
    pub fn run_main_capturing_traced(&self) -> Result<(Value, String), RuntimeError> {
        with_call_stack(|| self.run_main_capturing())
    }

    /// Like [`run_main_capturing_traced`] but ALSO returns whatever the program
    /// printed BEFORE it trapped: the second tuple element is the captured stdout
    /// even on the error path. Used by the agent loop so a program that prints
    /// some lines and THEN fails can surface those lines (the full picture) in the
    /// runtime feedback to the model. On success the value + full output are
    /// returned exactly as [`run_main_capturing_traced`].
    pub fn run_main_capturing_traced_partial(
        &self,
    ) -> (Result<Value, RuntimeError>, String) {
        let mut captured = String::new();
        let result = with_call_stack(|| {
            let (res, out) = self.run_main_capturing_keep_output();
            // Stash the partial output regardless of outcome, then propagate the
            // inner `Result<Value, String>` so `with_call_stack` builds the trace.
            captured = out;
            res
        });
        (result, captured)
    }

    /// Run `main` with output capture on, returning BOTH `main`'s
    /// `Result<Value, String>` AND the captured stdout — even on a runtime error,
    /// so a partial print-then-trap output is preserved. Mirrors
    /// [`run_main_capturing`] but keeps the buffer on the error path.
    fn run_main_capturing_keep_output(&self) -> (Result<Value, String>, String) {
        let prev = OUTPUT_CAPTURE.with(|c| c.borrow_mut().replace(String::new()));
        let result = self.run_main();
        let captured = OUTPUT_CAPTURE.with(|c| std::mem::replace(&mut *c.borrow_mut(), prev));
        (result, captured.unwrap_or_default())
    }

    /// Run `main` with OUTPUT CAPTURE on: every `print_*` builtin appends its
    /// formatted line to a buffer instead of writing to stdout. Returns BOTH
    /// `main`'s value AND the captured stdout `String` (byte-for-byte what a
    /// normal run would have printed). The capture buffer is installed for the
    /// duration of this call only and torn down afterward — success OR error — so
    /// a subsequent `run_main` (or any other code) prints to real stdout exactly
    /// as before. Used by the agent loop / benchmark to grade what a program
    /// PRINTS, not just what `main` returns.
    pub fn run_main_capturing(&self) -> Result<(Value, String), String> {
        // Install a fresh capture buffer. Nesting is not expected (the agent
        // loop runs one program at a time), but if one were already active we
        // would still restore it below, so the outer capture is preserved.
        let prev = OUTPUT_CAPTURE.with(|c| c.borrow_mut().replace(String::new()));
        let result = self.run_main();
        // Always retrieve our buffer and restore the EXACT previous capture
        // state (`None` if there was none), whether `main` succeeded or failed.
        let captured = OUTPUT_CAPTURE.with(|c| std::mem::replace(&mut *c.borrow_mut(), prev));
        let captured = captured.unwrap_or_default();
        result.map(|v| (v, captured))
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
        match &e.kind {
            ExprKind::Int(n) => Ok(Value::Int(*n)),
            ExprKind::Float(f) => Ok(Value::Float(*f)),
            ExprKind::Bool(b) => Ok(Value::Bool(*b)),
            ExprKind::Str(s) => Ok(Value::Str(s.clone())),
            ExprKind::Unit => Ok(Value::Unit),

            ExprKind::Var(name) => {
                if let Some(v) = Interp::lookup(scope, name) {
                    Ok(v.clone())
                } else if let Some(c) = self.fn_as_closure(name) {
                    // A bare top-level function name used as a value.
                    Ok(c)
                } else {
                    Err(format!("unbound variable `{}`", name))
                }
            }

            ExprKind::Ctor(name, args) => {
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

            ExprKind::Record(name, fields) => {
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

            ExprKind::Field(obj, field) => {
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

            ExprKind::Update(base, updates) => {
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

            ExprKind::Call(name, args) => {
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval(a, scope)?);
                }
                // A local binding that shadows the name (e.g. a function-valued
                // parameter) is applied as a closure value.
                if let Some(v) = Interp::lookup(scope, name) {
                    let callee = v.clone();
                    return self.apply_value(callee, vals, name, e.span);
                }
                // `grad` is the reverse-mode autodiff builtin. It must invoke the
                // closure argument (`apply_value`), so it lives on `Interp`
                // rather than in the free `builtin` helper.
                if name == "grad" {
                    return self.grad_builtin(vals);
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
                let fn_line = f.line;
                let mut frame = HashMap::new();
                for (p, v) in f.params.iter().zip(vals.into_iter()) {
                    frame.insert(p.name.clone(), v);
                }
                let mut call_scope: Scope = vec![frame];
                // Push this function's frame BEFORE the depth check so a
                // recursion-limit error's trace includes the function that hit
                // it. The call SITE is this `Call` expression's span, so the
                // trace points at the exact `name(..)` in the caller's body.
                push_frame(name, fn_line, e.span);
                let d = self.depth.get() + 1;
                if d > MAX_CALL_DEPTH {
                    // Leave the frame on the stack (error path) for the trace.
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
                // On success, pop our frame; on error, leave it for the trace.
                if result.is_ok() {
                    pop_frame();
                }
                result
            }

            ExprKind::Unary(op, inner) => {
                let v = self.eval(inner, scope)?;
                match (op, v) {
                    (UnOp::Neg, Value::Int(n)) => Ok(Value::Int(
                        n.checked_neg().ok_or("integer overflow in unary `-`")?,
                    )),
                    (UnOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
                    // Reverse-mode AD: negate a tracing scalar. (-x)' : dx = -1.
                    (UnOp::Neg, Value::Tracing(id)) => {
                        let v = tracing_value(id)?;
                        Ok(record_unary(id, -v, -1.0))
                    }
                    (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
                    (op, v) => Err(format!("cannot apply {:?} to {}", op, v.display())),
                }
            }

            ExprKind::Binary(op, lhs, rhs) => self.eval_binary(*op, lhs, rhs, scope),

            ExprKind::If(cond, then, els) => match self.eval(cond, scope)? {
                Value::Bool(true) => self.eval(then, scope),
                Value::Bool(false) => self.eval(els, scope),
                other => Err(format!("`if` condition must be Bool, got {}", other.display())),
            },

            ExprKind::Match(scrut, arms) => {
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

            ExprKind::Lambda(params, body, _) => {
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

            ExprKind::Apply(callee, args, _) => {
                let f = self.eval(callee, scope)?;
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval(a, scope)?);
                }
                self.apply_value(f, vals, "value", e.span)
            }

            ExprKind::Block(stmts, last) => {
                scope.push(HashMap::new());
                let mut run = || -> Result<Value, String> {
                    for s in stmts {
                        match &s.kind {
                            StmtKind::Let { name, value, .. } => {
                                let v = self.eval(value, scope)?;
                                scope.last_mut().unwrap().insert(name.clone(), v);
                            }
                            StmtKind::Expr(e) => {
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
        match &e.kind {
            ExprKind::Call(name, args) => {
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
            ExprKind::If(cond, then, els) => match self.eval(cond, scope)? {
                Value::Bool(true) => self.eval_tail(self_name, then, scope),
                Value::Bool(false) => self.eval_tail(self_name, els, scope),
                other => Err(format!("`if` condition must be Bool, got {}", other.display())),
            },
            ExprKind::Match(scrut, arms) => {
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
            ExprKind::Block(stmts, last) => {
                scope.push(HashMap::new());
                let run = |me: &Self, scope: &mut Scope| -> Result<TailOutcome, String> {
                    for s in stmts {
                        match &s.kind {
                            StmtKind::Let { name, value, .. } => {
                                let val = me.eval(value, scope)?;
                                scope.last_mut().unwrap().insert(name.clone(), val);
                            }
                            StmtKind::Expr(ex) => {
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
    fn apply_value(
        &self,
        callee: Value,
        args: Vec<Value>,
        what: &str,
        call_span: crate::ast::Span,
    ) -> Result<Value, String> {
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
        // A closure has no single source definition line (it may be an anonymous
        // lambda or a top-level function used as a value), so its frame carries
        // def_line 0; the CALL SITE span still locates where it was applied.
        push_frame(what, 0, call_span);
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
        if result.is_ok() {
            pop_frame();
        }
        result
    }

    /// The reverse-mode autodiff builtin `grad(f, x) -> Vector`. Evaluates the
    /// scalar function `f` over a tracing Vector built from `x`'s coordinates,
    /// recording a tape; seeds the scalar output's adjoint to 1.0 and sweeps the
    /// tape backward; returns the vector of input-leaf adjoints (∂f/∂x).
    fn grad_builtin(&self, args: Vec<Value>) -> Result<Value, String> {
        let (callee, x) = match args.as_slice() {
            [c, Value::Vector(x)] => (c.clone(), x.clone()),
            [_, other] => {
                return Err(format!(
                    "grad: second argument must be a Vector, got {}",
                    other.display()
                ))
            }
            _ => return Err("grad expects (f: (Vector) -> Float, x: Vector)".into()),
        };

        // Install a fresh tape for the duration of this `grad` call. A nested
        // `grad` (f calls grad) would clobber the outer tape; reject that
        // cleanly rather than mis-differentiate.
        let already = GRAD_TAPE.with(|t| t.borrow().is_some());
        if already {
            return Err("grad: nested `grad` calls are not supported".into());
        }

        // Build one leaf node per input coordinate, then a tracing Vector of
        // those node ids to feed to `f`.
        let leaves: Vec<usize> = GRAD_TAPE.with(|t| {
            *t.borrow_mut() = Some(Tape::new());
            let mut b = t.borrow_mut();
            let tape = b.as_mut().unwrap();
            x.iter().map(|&v| tape.leaf(v)).collect()
        });
        let tracing_x = Value::TracingVec(leaves.clone());

        // Evaluate f(tracing_x). On ANY error, tear down the tape before
        // propagating so a later normal evaluation never sees a stale tape.
        let result =
            self.apply_value(callee, vec![tracing_x], "grad function `f`", crate::ast::Span::none());

        let grad_or_err = result.and_then(|out| match out {
            // The scalar output must be a single tracing node. A bare constant
            // output (f ignores x) yields a zero gradient.
            Value::Tracing(out_id) => GRAD_TAPE.with(|t| {
                let b = t.borrow();
                let tape = b.as_ref().unwrap();
                let adj = tape.backward(out_id);
                Ok(Value::Vector(leaves.iter().map(|&l| adj[l]).collect()))
            }),
            Value::Float(_) | Value::Int(_) => {
                // f returned a constant independent of x: gradient is all zeros.
                Ok(Value::Vector(vec![0.0; leaves.len()]))
            }
            other => Err(format!(
                "grad: function `f` must return a Float, got {}",
                other.display()
            )),
        });

        // Always clear the tape, success or failure.
        GRAD_TAPE.with(|t| *t.borrow_mut() = None);
        grad_or_err
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

        // Reverse-mode autodiff: if either operand is a tracing scalar (only
        // possible inside a `grad` call), record the arithmetic op on the tape
        // instead of computing a plain float. Zero overhead otherwise.
        if is_tracing(&l) || is_tracing(&r) {
            return tracing_float_op(op, &l, &r);
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
        // A Vector compares by length + elements exactly (bitwise float `==`, so
        // NaN != NaN, matching scalar Float equality). A Vector never equals an
        // Array[Float] (distinct tags; falls through to `false`).
        (Value::Vector(x), Value::Vector(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| a == b)
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
    match &pat.kind {
        PatternKind::Wild => true,
        PatternKind::Var(name) => {
            binds.insert(name.clone(), val.clone());
            true
        }
        PatternKind::Int(i) => matches!(val, Value::Int(v) if v == i),
        PatternKind::Bool(b) => matches!(val, Value::Bool(v) if v == b),
        PatternKind::Ctor(name, subs) => match val {
            Value::Data { ctor, fields } if ctor == name && fields.len() == subs.len() => subs
                .iter()
                .zip(fields)
                .all(|(p, f)| match_pattern(p, f, binds, record_fields)),
            _ => false,
        },
        PatternKind::Record(name, sub_fields) => match val {
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
/// Differentiable Vector builtins for reverse-mode AD: invoked only when a
/// `grad` trace passes a `TracingVec` (or, for `vec_scale`, a `Tracing` scalar
/// factor) to a vector builtin. Each op records the appropriate tape nodes and
/// returns a `Tracing`/`TracingVec` result. Returns `Ok(None)` for a builtin
/// that is not part of the differentiable vector op set, so the caller can
/// surface a clean "unsupported operation" error rather than silently mis-
/// differentiating. Length-mismatch policy matches the concrete builtins.
fn tracing_vec_builtin(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    // Convert a value expected to be a tracing Vector into its node-id slice.
    // A *concrete* `Vector` operand (e.g. a constant `c` in `vec_dot(x, c)`) is
    // lifted to fresh constant leaf nodes so the mixed op differentiates
    // correctly (its partials w.r.t. those leaves are simply discarded — they
    // are not input leaves).
    let as_vec_nodes = |v: &Value| -> Result<Vec<usize>, String> {
        match v {
            Value::TracingVec(ids) => Ok(ids.clone()),
            Value::Vector(xs) => GRAD_TAPE.with(|t| {
                let mut b = t.borrow_mut();
                let tape = b
                    .as_mut()
                    .ok_or_else(|| "grad: internal tape missing".to_string())?;
                Ok(xs.iter().map(|f| tape.leaf(*f)).collect())
            }),
            other => Err(format!(
                "grad: `{}` expected a Vector, got {}",
                name,
                other.display()
            )),
        }
    };
    match name {
        "vec_get" => match args {
            [vv, Value::Int(i)] => {
                let ids = as_vec_nodes(vv)?;
                if *i < 0 || *i as usize >= ids.len() {
                    return Err(format!(
                        "vec_get index {} out of range for vector of length {}",
                        i,
                        ids.len()
                    ));
                }
                // Identity: result = x[i]; partial w.r.t. x[i] is 1.
                let node = ids[*i as usize];
                let v = tracing_value(node)?;
                Ok(Some(record_unary(node, v, 1.0)))
            }
            _ => Err("vec_get expects (Vector, Int)".into()),
        },
        "vec_len" => match args {
            [Value::TracingVec(ids)] => Ok(Some(Value::Int(ids.len() as i64))),
            _ => Err("vec_len expects (Vector)".into()),
        },
        "vec_dot" => match args {
            [a, b] => {
                let xa = as_vec_nodes(a)?;
                let xb = as_vec_nodes(b)?;
                if xa.len() != xb.len() {
                    return Err(format!(
                        "vec_dot length mismatch: {} vs {}",
                        xa.len(),
                        xb.len()
                    ));
                }
                // dot = Σ a_i*b_i ; ∂dot/∂a_i = b_i, ∂dot/∂b_i = a_i.
                let mut acc = 0.0;
                let mut parents = Vec::with_capacity(xa.len() * 2);
                for (&ai, &bi) in xa.iter().zip(xb.iter()) {
                    let av = tracing_value(ai)?;
                    let bv = tracing_value(bi)?;
                    acc += av * bv; // same left-to-right order as `dot`
                    parents.push((ai, bv));
                    parents.push((bi, av));
                }
                Ok(Some(record_many(parents, acc)))
            }
            _ => Err("vec_dot expects (Vector, Vector)".into()),
        },
        "vec_norm" => match args {
            [a] => {
                let xa = as_vec_nodes(a)?;
                // norm = sqrt(Σ x_i^2); ∂norm/∂x_i = x_i / norm. At norm==0 the
                // gradient is undefined — keep it 0 (a clean, non-NaN choice).
                let mut sumsq = 0.0;
                let mut vals = Vec::with_capacity(xa.len());
                for &xi in &xa {
                    let v = tracing_value(xi)?;
                    vals.push(v);
                    sumsq += v * v;
                }
                let norm = sumsq.sqrt();
                let parents: Vec<(usize, f64)> = xa
                    .iter()
                    .zip(vals.iter())
                    .map(|(&xi, &v)| (xi, if norm == 0.0 { 0.0 } else { v / norm }))
                    .collect();
                Ok(Some(record_many(parents, norm)))
            }
            _ => Err("vec_norm expects (Vector)".into()),
        },
        "vec_add" | "vec_sub" => match args {
            [a, b] => {
                let xa = as_vec_nodes(a)?;
                let xb = as_vec_nodes(b)?;
                if xa.len() != xb.len() {
                    return Err(format!(
                        "{} length mismatch: {} vs {}",
                        name,
                        xa.len(),
                        xb.len()
                    ));
                }
                let sub = name == "vec_sub";
                let mut out = Vec::with_capacity(xa.len());
                for (&ai, &bi) in xa.iter().zip(xb.iter()) {
                    let av = tracing_value(ai)?;
                    let bv = tracing_value(bi)?;
                    // add: r=a+b, da=1, db=1 ; sub: r=a-b, da=1, db=-1.
                    let (rv, db) = if sub { (av - bv, -1.0) } else { (av + bv, 1.0) };
                    let node = match record_binary(ai, bi, rv, 1.0, db) {
                        Value::Tracing(id) => id,
                        _ => unreachable!(),
                    };
                    out.push(node);
                }
                Ok(Some(Value::TracingVec(out)))
            }
            _ => Err(format!("{} expects (Vector, Vector)", name).into()),
        },
        "vec_scale" => match args {
            [a, s] => {
                let xa = as_vec_nodes(a)?;
                // The scalar factor may itself be tracing (e.g. `vec_scale(v, t)`
                // with `t` a differentiated scalar) or a plain Float.
                let snode = as_node(s)?;
                let sv = tracing_value(snode)?;
                let mut out = Vec::with_capacity(xa.len());
                for &ai in &xa {
                    let av = tracing_value(ai)?;
                    // r = s*a ; ∂r/∂a = s, ∂r/∂s = a.
                    let node = match record_binary(ai, snode, sv * av, sv, av) {
                        Value::Tracing(id) => id,
                        _ => unreachable!(),
                    };
                    out.push(node);
                }
                Ok(Some(Value::TracingVec(out)))
            }
            _ => Err("vec_scale expects (Vector, Float)".into()),
        },
        "vec_push" => match args {
            [vv, x] => {
                let mut ids = as_vec_nodes(vv)?;
                ids.push(as_node(x)?);
                Ok(Some(Value::TracingVec(ids)))
            }
            _ => Err("vec_push expects (Vector, Float)".into()),
        },
        "vec_from_array" => match args {
            [Value::Array(xs)] => {
                // Build a tracing Vector from an array that contains tracing
                // scalars (and/or plain floats lifted to constant leaves).
                let mut ids = Vec::with_capacity(xs.len());
                for v in xs {
                    ids.push(as_node(v)?);
                }
                Ok(Some(Value::TracingVec(ids)))
            }
            _ => Err("vec_from_array expects (Array[Float])".into()),
        },
        // Any other builtin applied to a tracing value is outside the supported
        // differentiable op set: signal `None` so the caller raises a clean,
        // specific error (never a panic, never a silently-wrong gradient).
        _ => Ok(None),
    }
}

fn builtin(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    // Reverse-mode AD: a Vector builtin applied to a tracing Vector (only
    // possible inside a `grad` call) is handled by the tape recorder, which
    // produces `Tracing`/`TracingVec` results. We branch here, before the normal
    // (concrete) builtin dispatch, only when a tracing operand is present — so
    // ordinary programs pay nothing and behave identically.
    if args
        .iter()
        .any(|a| matches!(a, Value::TracingVec(_) | Value::Tracing(_)))
    {
        match tracing_vec_builtin(name, args)? {
            Some(v) => return Ok(Some(v)),
            // A tracing operand reached a builtin outside the differentiable op
            // set — fail cleanly rather than fall through to the concrete
            // dispatch (which would mis-type the tracing value).
            None => {
                return Err(format!(
                    "grad: unsupported operation `{}` on a differentiated value",
                    name
                ))
            }
        }
    }
    let one = |args: &[Value]| -> Result<Value, String> {
        if args.len() != 1 {
            return Err(format!("`{}` expects 1 argument", name));
        }
        Ok(args[0].clone())
    };
    match name {
        "print_int" => match one(args)? {
            Value::Int(n) => {
                emit_line(&n.to_string());
                Ok(Some(Value::Unit))
            }
            v => Err(format!("print_int expects Int, got {}", v.display())),
        },
        "print_float" => match one(args)? {
            Value::Float(f) => {
                emit_line(&format!("{}", f));
                Ok(Some(Value::Unit))
            }
            v => Err(format!("print_float expects Float, got {}", v.display())),
        },
        "print_bool" => match one(args)? {
            Value::Bool(b) => {
                emit_line(&b.to_string());
                Ok(Some(Value::Unit))
            }
            v => Err(format!("print_bool expects Bool, got {}", v.display())),
        },
        "print_str" => match one(args)? {
            Value::Str(s) => {
                emit_line(&s);
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
        // Pull row `i` of a 2D tensor out as a length-cols Vector, widening
        // each stored f32 to f64 (exact). Out-of-range row index traps.
        "tensor_row" => match args {
            [Value::Tensor(t), Value::Int(i)] => {
                let (rows, cols) = (t.rows(), t.cols());
                if *i < 0 || *i as usize >= rows {
                    return Err(format!(
                        "tensor_row index {} out of range for {}x{} tensor",
                        i, rows, cols
                    ));
                }
                let r = *i as usize;
                let out: Vec<f64> = (0..cols).map(|c| t.at(r, c) as f64).collect();
                Ok(Some(Value::Vector(out)))
            }
            _ => Err("tensor_row expects (Tensor, Int)".into()),
        },
        // Stack an Array[Vector] of equal-length vectors into a [n, L] tensor,
        // narrowing each f64 element to f32. Unequal lengths trap; an empty
        // array yields a 0x0 tensor.
        "tensor_from_rows" => match args {
            [Value::Array(rows)] => {
                if rows.is_empty() {
                    return Ok(Some(Value::Tensor(crate::tensor::Tensor::zeros(&[0, 0]))));
                }
                let l = match &rows[0] {
                    Value::Vector(v) => v.len(),
                    _ => return Err("tensor_from_rows expects (Array[Vector])".into()),
                };
                let mut data: Vec<f32> = Vec::with_capacity(rows.len() * l);
                for row in rows {
                    match row {
                        Value::Vector(v) => {
                            if v.len() != l {
                                return Err(
                                    "tensor_from_rows: rows must have equal length".into()
                                );
                            }
                            for x in v {
                                data.push(*x as f32);
                            }
                        }
                        _ => return Err("tensor_from_rows expects (Array[Vector])".into()),
                    }
                }
                Ok(Some(Value::Tensor(crate::tensor::Tensor::new(
                    vec![rows.len(), l],
                    data,
                ))))
            }
            _ => Err("tensor_from_rows expects (Array[Vector])".into()),
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
                // Cosine over the LEARNED count-based (PPMI + truncated-SVD)
                // distributional embeddings of each text — real semantics, not
                // a hash. Related texts score higher than unrelated ones.
                Ok(Some(Value::Float(crate::embed::embed_similarity(a, b) as f64)))
            }
            _ => Err("embed_similarity expects (String, String)".into()),
        },
        "embed" => match args {
            [Value::Str(s)] => {
                // The learned embedding of `s` as a first-class Vector (length
                // crate::embed::DIM), so it composes with the retrieval prelude
                // (`nearest`/`similarities` over `Array[Vector]`).
                let v = crate::embed::embed(s);
                Ok(Some(Value::Vector(v.into_iter().map(|f| f as f64).collect())))
            }
            _ => Err("embed expects (String)".into()),
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

        // ---- Vector / Embedding (dense, immutable buffer of Float) ----------
        // The interpreter is the reference oracle. `push`/`add`/`scale` are
        // functional (a fresh Vec); the native backend's FBIP in-place reuse is a
        // pure optimization. Length-mismatch on dot/cosine/add and OOB get are
        // clean runtime errors; cosine of a zero-norm operand is 0.0 (no /0).
        "vec_new" => match args {
            [] => Ok(Some(Value::Vector(Vec::new()))),
            _ => Err("vec_new expects no arguments".into()),
        },
        "vec_from_array" => match args {
            [Value::Array(xs)] => {
                let mut out = Vec::with_capacity(xs.len());
                for v in xs {
                    match v {
                        Value::Float(f) => out.push(*f),
                        _ => return Err("vec_from_array expects (Array[Float])".into()),
                    }
                }
                Ok(Some(Value::Vector(out)))
            }
            _ => Err("vec_from_array expects (Array[Float])".into()),
        },
        "vec_to_array" => match args {
            [Value::Vector(xs)] => {
                Ok(Some(Value::Array(xs.iter().map(|f| Value::Float(*f)).collect())))
            }
            _ => Err("vec_to_array expects (Vector)".into()),
        },
        "vec_len" => match args {
            [Value::Vector(xs)] => Ok(Some(Value::Int(xs.len() as i64))),
            _ => Err("vec_len expects (Vector)".into()),
        },
        "vec_get" => match args {
            [Value::Vector(xs), Value::Int(i)] => {
                if *i < 0 || *i as usize >= xs.len() {
                    return Err(format!(
                        "vec_get index {} out of range for vector of length {}",
                        i,
                        xs.len()
                    ));
                }
                Ok(Some(Value::Float(xs[*i as usize])))
            }
            _ => Err("vec_get expects (Vector, Int)".into()),
        },
        "vec_push" => match args {
            [Value::Vector(xs), Value::Float(v)] => {
                let mut out = xs.clone();
                out.push(*v);
                Ok(Some(Value::Vector(out)))
            }
            _ => Err("vec_push expects (Vector, Float)".into()),
        },
        "vec_dot" => match args {
            [Value::Vector(x), Value::Vector(y)] => {
                if x.len() != y.len() {
                    return Err(format!(
                        "vec_dot length mismatch: {} vs {}",
                        x.len(),
                        y.len()
                    ));
                }
                Ok(Some(Value::Float(dot(x, y))))
            }
            _ => Err("vec_dot expects (Vector, Vector)".into()),
        },
        "vec_norm" => match args {
            [Value::Vector(x)] => Ok(Some(Value::Float(dot(x, x).sqrt()))),
            _ => Err("vec_norm expects (Vector)".into()),
        },
        "vec_cosine" => match args {
            [Value::Vector(x), Value::Vector(y)] => {
                if x.len() != y.len() {
                    return Err(format!(
                        "vec_cosine length mismatch: {} vs {}",
                        x.len(),
                        y.len()
                    ));
                }
                let nx = dot(x, x).sqrt();
                let ny = dot(y, y).sqrt();
                // Zero-norm policy: return 0.0 (never divide by zero -> NaN).
                if nx == 0.0 || ny == 0.0 {
                    Ok(Some(Value::Float(0.0)))
                } else {
                    Ok(Some(Value::Float(dot(x, y) / (nx * ny))))
                }
            }
            _ => Err("vec_cosine expects (Vector, Vector)".into()),
        },
        "vec_add" => match args {
            [Value::Vector(x), Value::Vector(y)] => {
                if x.len() != y.len() {
                    return Err(format!(
                        "vec_add length mismatch: {} vs {}",
                        x.len(),
                        y.len()
                    ));
                }
                Ok(Some(Value::Vector(
                    x.iter().zip(y).map(|(a, b)| a + b).collect(),
                )))
            }
            _ => Err("vec_add expects (Vector, Vector)".into()),
        },
        "vec_sub" => match args {
            [Value::Vector(x), Value::Vector(y)] => {
                if x.len() != y.len() {
                    return Err(format!(
                        "vec_sub length mismatch: {} vs {}",
                        x.len(),
                        y.len()
                    ));
                }
                Ok(Some(Value::Vector(
                    x.iter().zip(y).map(|(a, b)| a - b).collect(),
                )))
            }
            _ => Err("vec_sub expects (Vector, Vector)".into()),
        },
        "vec_scale" => match args {
            [Value::Vector(x), Value::Float(s)] => {
                Ok(Some(Value::Vector(x.iter().map(|a| a * s).collect())))
            }
            _ => Err("vec_scale expects (Vector, Float)".into()),
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
        // Enumerate the map's keys / values into an Array, in ascending key order
        // (entries are already kept sorted by key), so the two are index-aligned.
        "map_keys" => match args {
            [Value::Map(entries)] => {
                Ok(Some(Value::Array(entries.iter().map(|(k, _)| k.clone()).collect())))
            }
            _ => Err("map_keys expects (Map)".into()),
        },
        "map_values" => match args {
            [Value::Map(entries)] => {
                Ok(Some(Value::Array(entries.iter().map(|(_, v)| v.clone()).collect())))
            }
            _ => Err("map_values expects (Map)".into()),
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
        // Enumerate the set's elements into an Array, in ascending order (the set
        // is already kept sorted by element).
        "set_to_array" => match args {
            [Value::Set(elems)] => Ok(Some(Value::Array(elems.clone()))),
            _ => Err("set_to_array expects (Set)".into()),
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

    // Lex -> parse -> typeck -> interp, returning (main's value, captured stdout).
    fn run_capturing(src: &str) -> (Value, String) {
        let toks = lexer::lex(src).expect("lex");
        let prog = parser::parse(toks).expect("parse");
        typeck::check(&prog).expect("typeck");
        let interp = Interp::new(&prog).expect("interp::new");
        interp.run_main_capturing().expect("run")
    }

    // Lex -> parse -> typeck -> interp with STACK TRACKING, expecting a runtime
    // error and returning the structured `RuntimeError`.
    fn run_traced_err(src: &str) -> RuntimeError {
        let toks = lexer::lex(src).expect("lex");
        let prog = parser::parse(toks).expect("parse");
        typeck::check(&prog).expect("typeck");
        let interp = Interp::new(&prog).expect("interp::new");
        interp.run_main_traced().expect_err("expected a runtime error")
    }

    // ---- runtime stack traces (Feature 1) -----------------------------

    #[test]
    fn stack_trace_three_frames_div_by_zero() {
        // main -> outer -> inner, where inner divides by zero. The trace is
        // most-recent first; each frame now reports the PRECISE CALL SITE
        // (`line:col`) where the function was called, not its definition line.
        let src = "\
fn inner(x: Int) -> Int = 1 / 0
fn outer(x: Int) -> Int = inner(x)
fn main() -> Int = outer(5)
";
        let err = run_traced_err(src);
        assert_eq!(err.message, "division by zero");
        assert_eq!(
            err.frames,
            vec![
                // `inner` was called at line 2 col 27 (`inner(x)` in `outer`).
                Frame { function: "inner".into(), def_line: 1, call_line: 2, call_col: 27 },
                // `outer` was called at line 3 col 20 (`outer(5)` in `main`).
                Frame { function: "outer".into(), def_line: 2, call_line: 3, call_col: 20 },
                // `main` is the synthetic entry: no call site, falls back to def line 3.
                Frame { function: "main".into(), def_line: 3, call_line: 0, call_col: 0 },
            ]
        );
        // Rendered form: call sites as `line:col`, `main` as its definition line.
        let r = err.render();
        assert!(r.starts_with("runtime error: division by zero"));
        assert!(r.contains("\n  at `inner` (line 2:27)"), "got:\n{}", r);
        assert!(r.contains("\n  at `outer` (line 3:20)"), "got:\n{}", r);
        assert!(r.contains("\n  at `main` (line 3)"), "got:\n{}", r);
    }

    #[test]
    fn stack_trace_index_out_of_bounds_carries_chain() {
        // An out-of-bounds `array_get` is a RUNTIME trap (it type-checks), so its
        // call chain is traced. (A non-exhaustive `match` is a COMPILE error in
        // Aria and never reaches the interpreter, so it is not a runtime trace.)
        let src = "\
fn get_it(i: Int) -> Int = array_get(array_push(array_new(), 10), i)
fn main() -> Int = get_it(5)
";
        let err = run_traced_err(src);
        // Index-out-of-bounds message from the array builtin.
        assert!(
            err.message.contains("out of") || err.message.contains("index"),
            "msg: {}",
            err.message
        );
        let names: Vec<&str> = err.frames.iter().map(|f| f.function.as_str()).collect();
        assert_eq!(names, vec!["get_it", "main"]);
    }

    #[test]
    fn stack_trace_collapses_nontail_self_recursion() {
        // A non-tail self-recursive function that traps must NOT produce one frame
        // per recursive call: consecutive frames with the SAME call site collapse.
        // `boom` recurses in a non-tail position (the `+ 1` makes it non-tail),
        // bottoming out by dividing by zero at n == 0. The 20 recursive calls all
        // share the call site `boom(n - 1)` (line 1) so they collapse to ONE
        // frame; the OUTERMOST `boom` was called from `main` (line 2) — a distinct
        // call site — so it is reported separately. This is strictly MORE precise
        // than the old definition-line trace (which merged all `boom`s).
        let src = "\
fn boom(n: Int) -> Int = if n == 0 { 1 / 0 } else { boom(n - 1) + 1 }
fn main() -> Int = boom(20)
";
        let err = run_traced_err(src);
        assert_eq!(err.message, "division by zero");
        let names: Vec<&str> = err.frames.iter().map(|f| f.function.as_str()).collect();
        // The recursive boom calls (call site line 1) collapse to one; the entry
        // boom (call site line 2, from `main`) stays distinct. Not 21 frames.
        assert_eq!(names, vec!["boom", "boom", "main"]);
        // The two boom frames carry their two distinct call sites.
        assert_eq!(err.frames[0].call_line, 1, "inner recursion call site");
        assert_eq!(err.frames[1].call_line, 2, "entry call from main");
    }

    #[test]
    fn successful_run_leaves_no_trace_and_is_unchanged() {
        // The SUCCESS path through the traced runner returns the same value as the
        // plain runner — stack tracking adds no behavior change.
        let src = "fn main() -> Int = { print_int(3); 42 }";
        let toks = lexer::lex(src).expect("lex");
        let prog = parser::parse(toks).expect("parse");
        typeck::check(&prog).expect("typeck");
        let interp = Interp::new(&prog).expect("interp::new");
        let v = interp.run_main_traced().expect("clean run");
        assert_eq!(v.display(), "42");
    }

    // ---- output capture (Part 1) --------------------------------------

    #[test]
    fn capture_collects_print_lines_with_formatting_and_newlines() {
        // Each print_* appends ONE line (its formatted value + '\n'), in order.
        let src = "fn main() -> Int = {\n\
                     print_int(7);\n\
                     print_bool(true);\n\
                     print_str(\"hi\");\n\
                     print_float(1.5);\n\
                     42\n\
                   }";
        let (v, out) = run_capturing(src);
        assert_eq!(v.display(), "42");
        assert_eq!(out, "7\ntrue\nhi\n1.5\n");
    }

    #[test]
    fn capture_of_program_with_no_prints_is_empty() {
        let (v, out) = run_capturing("fn main() -> Int = 1 + 2");
        assert_eq!(v.display(), "3");
        assert_eq!(out, "");
    }

    #[test]
    fn capture_does_not_leak_into_a_later_noncapturing_run() {
        // After a capturing run, the thread-local sink must be back to `None` so
        // the print builtins resume writing to stdout (default behavior). We
        // can't observe stdout here, but we CAN prove a subsequent capturing run
        // starts from an empty buffer (i.e. the prior capture didn't persist).
        let _ = run_capturing("fn main() -> Int = { print_int(1); 0 }");
        let (_v, out) = run_capturing("fn main() -> Int = { print_int(2); 0 }");
        assert_eq!(out, "2\n", "each capturing run starts fresh");
    }

    #[test]
    fn capture_str_value_is_byte_for_byte_with_trailing_newline() {
        // print_str of a String main() returns: value and captured text agree on
        // the content (capture adds exactly the println newline, nothing else).
        let src = "fn main() -> String = {\n\
                     print_str(concat(\"x = \", int_to_str(6 * 7)));\n\
                     \"done\"\n\
                   }";
        let (v, out) = run_capturing(src);
        assert_eq!(v.display(), "done");
        assert_eq!(out, "x = 42\n");
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
            // A one-element float vector, so the vector builtins have something to
            // index/operate on. `vec_from_array` is driven with `dummy(Array[Float])`
            // (a one-element `Array` of `Float(0.0)`), so it succeeds.
            Named(n, _) if n == "Vector" => Value::Vector(vec![0.0]),
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
            Value::Vector(_) => Named("Vector".into(), vec![]),
            Value::Data { ctor, .. } => Named(ctor.clone(), vec![]),
            Value::Closure(c) => {
                Fn(c.params.iter().map(|_| Unit).collect(), Box::new(Unit))
            }
            // Reverse-mode AD tracing values only exist transiently inside a
            // `grad` call and never reach this test helper; map them to the
            // scalar/Vector type they stand in for.
            Value::Tracing(_) => Float,
            Value::TracingVec(_) => Named("Vector".into(), vec![]),
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
            // `grad` is the one builtin whose implementation lives on the
            // evaluator (`Interp::grad_builtin`), not the free `builtin` helper,
            // because it must APPLY its closure argument. It is exercised
            // directly by the autodiff tests instead; the free `builtin` table
            // legitimately does not contain it.
            if name == "grad" {
                continue;
            }
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

    // ---- Vector / Embedding (interpreter oracle behavior) ---------------

    #[test]
    fn vector_build_dot_norm_and_roundtrip() {
        // from_array -> dot / norm; to_array round-trip preserves elements.
        let v = run(r#"
            fn main() -> Float = {
                let a = vec_from_array([1.0, 2.0, 3.0]);
                let b = vec_from_array([4.0, 5.0, 6.0]);
                vec_dot(a, b)
            }
        "#);
        // 1*4 + 2*5 + 3*6 = 32
        assert!(matches!(v, Value::Float(f) if f == 32.0), "got {}", v.display());

        let n = run("fn main() -> Float = vec_norm(vec_from_array([3.0, 4.0]))");
        // sqrt(9 + 16) = 5
        assert!(matches!(n, Value::Float(f) if f == 5.0), "got {}", n.display());

        // to_array round-trip: index 1 of the array of a Vector.
        let r = run(r#"
            fn main() -> Float = {
                let a = vec_from_array([7.0, 8.0, 9.0]);
                let xs = vec_to_array(a);
                xs[1]
            }
        "#);
        assert!(matches!(r, Value::Float(f) if f == 8.0), "got {}", r.display());

        // push / len / get.
        let g = run(r#"
            fn main() -> Float = {
                let a = vec_push(vec_push(vec_new(), 1.5), 2.5);
                vec_get(a, 1)
            }
        "#);
        assert!(matches!(g, Value::Float(f) if f == 2.5), "got {}", g.display());
    }

    #[test]
    fn vector_cosine_parallel_orthogonal_and_zero_norm() {
        // cosine of identical (parallel) unit vectors is exactly 1.0.
        let par = run(r#"fn main() -> Float =
            vec_cosine(vec_from_array([1.0, 0.0]), vec_from_array([1.0, 0.0]))"#);
        assert!(matches!(par, Value::Float(f) if f == 1.0), "got {}", par.display());

        // cosine of orthogonal vectors is exactly 0.0.
        let orth = run(r#"fn main() -> Float =
            vec_cosine(vec_from_array([1.0, 0.0]), vec_from_array([0.0, 1.0]))"#);
        assert!(matches!(orth, Value::Float(f) if f == 0.0), "got {}", orth.display());

        // ZERO-NORM policy: an all-zero (or empty) operand yields 0.0, NOT NaN.
        let zn = run(r#"fn main() -> Float =
            vec_cosine(vec_from_array([1.0, 2.0]), vec_from_array([0.0, 0.0]))"#);
        assert!(matches!(zn, Value::Float(f) if f == 0.0), "got {}", zn.display());
        // The result must not be NaN.
        if let Value::Float(f) = zn {
            assert!(!f.is_nan(), "zero-norm cosine produced NaN");
        }
    }

    #[test]
    fn vector_add_scale_display_and_equality() {
        // add (elementwise) + canonical `Vector[..]` display (shortest round-trip
        // floats, comma+space separated; empty `Vector[]`).
        let v = run(r#"fn main() -> Vector =
            vec_add(vec_from_array([1.0, 2.0, 3.0]), vec_from_array([4.0, 5.0, 6.0]))"#);
        assert_eq!(v.display(), "Vector[5, 7, 9]");
        assert_eq!(run("fn main() -> Vector = vec_new()").display(), "Vector[]");

        let s = run("fn main() -> Vector = vec_scale(vec_from_array([1.5, 2.0]), 2.0)");
        assert_eq!(s.display(), "Vector[3, 4]");

        // `==` compares length + elements exactly.
        let eq = run(r#"fn main() -> Bool =
            vec_from_array([1.0, 2.0]) == vec_from_array([1.0, 2.0])"#);
        assert!(matches!(eq, Value::Bool(true)));
        let ne = run(r#"fn main() -> Bool =
            vec_from_array([1.0, 2.0]) == vec_from_array([1.0, 3.0])"#);
        assert!(matches!(ne, Value::Bool(false)));
        // Different length -> not equal.
        let nl = run(r#"fn main() -> Bool =
            vec_from_array([1.0]) == vec_from_array([1.0, 2.0])"#);
        assert!(matches!(nl, Value::Bool(false)));
        // Distinct tags: a Vector and an Array[Float] with identical elements are
        // NOT equal (even though the type checker also rejects comparing them).
        assert!(!values_equal(
            &Value::Vector(vec![1.0, 2.0]),
            &Value::Array(vec![Value::Float(1.0), Value::Float(2.0)])
        ));
    }

    #[test]
    fn vector_error_cases_are_clean_runtime_errors() {
        // OOB index on get -> clean runtime error (not a panic).
        assert!(run_err("fn main() -> Float = vec_get(vec_from_array([1.0, 2.0]), 5)")
            .contains("out of range"));
        assert!(run_err("fn main() -> Float = vec_get(vec_from_array([1.0]), -1)")
            .contains("out of range"));
        // Length mismatch on dot / cosine / add -> clean runtime error.
        assert!(run_err(
            "fn main() -> Float = vec_dot(vec_from_array([1.0, 2.0]), vec_from_array([1.0]))"
        )
        .contains("length mismatch"));
        assert!(run_err(
            "fn main() -> Float = vec_cosine(vec_from_array([1.0, 2.0]), vec_from_array([1.0]))"
        )
        .contains("length mismatch"));
        assert!(run_err(
            "fn main() -> Vector = vec_add(vec_from_array([1.0, 2.0]), vec_from_array([1.0]))"
        )
        .contains("length mismatch"));
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
    fn map_keys_values_set_to_array_enumeration() {
        // map_keys / map_values come out in ASCENDING key order (entries are kept
        // sorted), regardless of insertion order, and are index-aligned.
        let keys = run(r#"
            fn main() -> Array[Int] =
                map_keys(map_insert(map_insert(map_insert(map_new(), 30, 3), 10, 1), 20, 2))
        "#);
        assert_eq!(keys.display(), "[10, 20, 30]");
        let vals = run(r#"
            fn main() -> Array[Int] =
                map_values(map_insert(map_insert(map_insert(map_new(), 30, 3), 10, 1), 20, 2))
        "#);
        assert_eq!(vals.display(), "[1, 2, 3]"); // index-aligned with sorted keys
        // Str keys sort lexicographically; values follow the same order.
        let skeys = run(r#"
            fn main() -> Array[String] =
                map_keys(map_insert(map_insert(map_insert(map_new(), "pear", 9), "apple", 7), "fig", 8))
        "#);
        assert_eq!(skeys.display(), "[apple, fig, pear]");
        let svals = run(r#"
            fn main() -> Array[Int] =
                map_values(map_insert(map_insert(map_insert(map_new(), "pear", 9), "apple", 7), "fig", 8))
        "#);
        assert_eq!(svals.display(), "[7, 8, 9]");
        // set_to_array: ascending, deduped.
        let elems = run(r#"
            fn main() -> Array[Int] =
                set_to_array(set_add(set_add(set_add(set_add(set_new(), 30), 10), 20), 10))
        "#);
        assert_eq!(elems.display(), "[10, 20, 30]");
        let strs = run(r#"
            fn main() -> Array[String] =
                set_to_array(set_add(set_add(set_add(set_new(), "b"), "a"), "c"))
        "#);
        assert_eq!(strs.display(), "[a, b, c]");
        // A non-Int/Str value type is fine for map_values (the oracle supports
        // any value type): Float values preserved in key order.
        let fvals = run(r#"
            fn main() -> Array[Float] =
                map_values(map_insert(map_insert(map_new(), 2, 2.5), 1, 1.5))
        "#);
        assert_eq!(fvals.display(), "[1.5, 2.5]");
    }

    #[test]
    fn enumeration_of_empty_map_set_is_empty_array() {
        // An empty map/set enumerates to an empty array (not a crash).
        assert_eq!(run("fn main() -> Array[Int] = map_keys(map_new())").display(), "[]");
        assert_eq!(run("fn main() -> Array[Int] = map_values(map_new())").display(), "[]");
        assert_eq!(run("fn main() -> Array[Int] = set_to_array(set_new())").display(), "[]");
        assert_eq!(run("fn main() -> Int = array_len(map_keys(map_new()))").display(), "0");
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
        // `embed_similarity` now uses the LEARNED count-based (PPMI + truncated
        // SVD) distributional model: identical text -> cosine ~1.0, AND a
        // semantically related pair must out-score an unrelated pair (a hash
        // could not do this — it is the proof the embedding is real).
        let identical = run(r#"
            fn main() -> Float =
                embed_similarity("the cat is a small pet animal",
                                 "the cat is a small pet animal")
        "#);
        match identical {
            Value::Float(f) => assert!((f - 1.0).abs() < 1e-4, "identical text similarity was {f}"),
            v => panic!("expected Float, got {}", v.display()),
        }

        let related = match run(r#"
            fn main() -> Float =
                embed_similarity("the cat is a small pet animal",
                                 "the dog is a loyal pet animal")
        "#) {
            Value::Float(f) => f,
            v => panic!("expected Float, got {}", v.display()),
        };
        let unrelated = match run(r#"
            fn main() -> Float =
                embed_similarity("the cat is a small pet animal",
                                 "the king rules the kingdom from his throne")
        "#) {
            Value::Float(f) => f,
            v => panic!("expected Float, got {}", v.display()),
        };
        assert!(
            related > unrelated,
            "related pair {related} should out-score unrelated pair {unrelated}"
        );
    }

    #[test]
    fn embed_builtin_returns_vector_and_retrieves() {
        // `embed(text) -> Vector` produces a first-class learned embedding that
        // composes with the retrieval prelude (`nearest`/`similarities` over an
        // `Array[Vector]`). End-to-end real-embedding retrieval: a cat query
        // must retrieve the dog document (index 0) over the king document
        // (index 1).
        // `nearest` comes from the retrieval prelude, so wrap the program.
        let v = run(&crate::prelude::wrap(r#"
            fn main() -> Int = {
                let store: Array[Vector] = [ embed("the dog is a loyal pet animal"),
                                             embed("the king rules the kingdom") ];
                nearest(store, embed("a cat is a small pet"))
            }
        "#));
        assert!(matches!(v, Value::Int(0)), "expected nearest = 0 (dog doc), got {}", v.display());

        // `embed` yields a Vector of the model dimension.
        let d = run(r#"fn main() -> Int = vec_len(embed("the cat"))"#);
        assert!(matches!(d, Value::Int(64)), "expected dim 64, got {}", d.display());
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

    #[test]
    fn runaway_non_tail_recursion_hits_guard_cleanly_not_stack_overflow() {
        // Runaway NON-tail recursion (the call is an operand of `+`, so it cannot
        // be TCO'd and consumes one native frame per level) must hit the depth
        // guard as a clean `Err` — NOT a native stack overflow / Abort trap.
        // We run on the SAME 4 GiB stack the CLI reserves for the interpreter
        // (`main::INTERP_STACK_SIZE`), which is what lets the 100k-frame guard win
        // the race against an overflow even in a worst-case debug build. This is
        // the regression for FIX 2 (the guard must fire before the worker stack
        // overflows in debug too). `boom(200000)` is well past MAX_CALL_DEPTH, so
        // the guard fires; `boom(40000)`/`boom(99000)` (below the guard) instead
        // run to completion on this stack, which the CLI-level check covers.
        const INTERP_STACK_SIZE: usize = 1 << 32; // mirror main::INTERP_STACK_SIZE
        let src = "fn boom(n: Int) -> Int = if n == 0 { 0 } else { 1 + boom(n - 1) }\n\
                   fn main() -> Int = boom(200000)";
        let res = std::thread::Builder::new()
            .stack_size(INTERP_STACK_SIZE)
            .spawn(move || {
                let prog = parser::parse(lexer::lex(src).unwrap()).unwrap();
                Interp::new(&prog).unwrap().run_main()
            })
            .unwrap()
            .join()
            .expect("runaway recursion must NOT abort the process (clean Err, not overflow)");
        match res {
            Err(msg) => assert!(
                msg.contains("maximum recursion depth"),
                "expected a clean recursion-depth error, got: {msg}"
            ),
            Ok(v) => panic!("expected a recursion-depth Err, got Ok({})", v.display()),
        }
    }
}
