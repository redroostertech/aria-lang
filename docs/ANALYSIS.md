# Aria — Runtime Stack Traces & Static Call Graph

> Two first-class, **AI-facing** capabilities that give a model both *richer
> runtime feedback* (what failed and through which call chain) and *structural
> code understanding* (callers/callees, dead code, recursion). This document is
> the contract for both: the stack-trace format and the `aria analyze`
> call-graph JSON schema.

There are two features here:

1. **Runtime stack traces** — when an interpreted program *errors*, the runtime
   error carries the call chain (functions + lines) from `main` down to the
   function that trapped.
2. **A static call graph** (`aria analyze`) — a static-analysis pass over the
   type-checked AST that reports callers/callees per function plus whole-program
   facts (entry, dead code, unreachable functions, recursive cycles).

---

## 1. Runtime stack traces

### Why (AI capability)

`aria check` answers *"is this well-typed?"*. The stack trace answers *"where
did it fail at runtime?"*. An LLM authoring loop that gets

```
runtime error: division by zero
  at `inner` (line 12:5)
  at `middle` (line 8:3)
  at `main` (line 3)
```

knows exactly which function trapped, **the precise call site** (`line:col`)
that reached it, and the chain back to `main` — a far stronger fix signal than
the bare message. The agent loop (`aria agent`) feeds this trace back to the
model on a clean-checking but runtime-failing program, closing the loop
**write -> check -> RUN -> fix**.

### Format

`aria run` on an **erroring** program prints, to **stderr**:

```
runtime error: <message>
  at `<function>` (line <L>:<C>)
  at `<function>` (line <L>:<C>)
  ...
```

- **Most-recent call first**: the function that trapped is the first `at` line;
  `main` is last.
- The location is the **precise CALL SITE** — the `line:col` of the call
  expression that entered this function (e.g. `inner` shows where `inner(..)`
  was written in the caller's body), 1-based. This points at the **exact call**,
  not merely the callee's definition line.
- When the call site is unknown — the synthetic entry into `main`, or a
  compiler-synthesized call with no source span — the frame falls back to the
  function's **definition line** (`line <N>`, no column).
- A compiler-generated callee with neither a call site nor a definition line (a
  trait dispatcher, or an applied closure / lambda) prints `(generated)`.
- **Consecutive identical frames are collapsed.** A non-tail self-recursive
  function that recurses deeply and then traps yields **one** frame for that
  function, not one per recursive call — so a 100k-deep recursion error does not
  produce a 100k-line trace. Tail-recursive self-calls are eliminated by the
  interpreter's TCO (they reuse a single call frame) and so already contribute a
  single frame.

### Structured form

The interpreter exposes a structured runtime error so tools (the agent loop,
diagnostics, an LSP) can consume the frames directly rather than parse text:

```rust
pub struct Frame {
    pub function: String,
    pub def_line: usize,   // callee's definition line; 0 => generated callee
    pub call_line: usize,  // precise CALL-SITE line; 0 => call site unknown
    pub call_col: usize,   // precise CALL-SITE column (when call_line != 0)
}
pub struct RuntimeError { pub message: String, pub frames: Vec<Frame> }
//                         frames are MOST-RECENT call first
```

- `interp::Interp::run_main_traced() -> Result<Value, RuntimeError>`
- `interp::Interp::run_main_capturing_traced() -> Result<(Value, String), RuntimeError>`
- `RuntimeError::render()` produces the human string shown above.

The plain `run_main` / `run_main_capturing` (which return `Result<_, String>`)
are unchanged for callers that don't want a trace.

### Cost & scope (honest notes)

- **Zero overhead / no behavior change on success.** Stack tracking is a
  thread-local that is `None` everywhere except inside a `*_traced` run; on the
  success path it only pushes/pops frames and never alters a value or any output.
  Normal `aria run` success output, and every existing example/test, is
  unchanged — only the **error** path gains the trace.
- **Interpreter-side only (v1).** The compiled **native (C)** and **wasm**
  backends keep their existing bare trap messages; they do **not** emit a stack
  trace in v1. The interpreter is the oracle and what the agent loop runs, so
  that is where the trace lives.
- **Frame lines are PRECISE CALL SITES.** Every AST expression carries a source
  span (1-based start/end line+column), and a frame records the span of the
  `Call`/`Apply` expression at the call site — so the trace points at the exact
  `callee(..)` that failed, not the callee's definition line. The same
  precise-span foundation powers the LSP's exact ranges and the structured
  diagnostics' `line`+`col`. (`main`, the synthetic entry, has no call site and
  falls back to its definition line.)

---

## 2. Static call graph — `aria analyze`

### Invocation

```sh
aria analyze <file.aria>          # human-readable summary
aria analyze --json <file.aria>   # stable JSON for AI tools / static analyzers
```

The pass runs **lex -> parse -> typeck first**, so it only analyzes well-formed
programs (a type error is reported and analysis is skipped). Like the rest of the
toolchain it analyzes the program **with the prelude wrapped in**, but it reports
on **user** functions and flags prelude / compiler-generated functions
separately (they are never counted as "unused user code").

The JSON is compact (one line), hand-rolled, and reuses `diagnostics::json_escape`
so it is always valid — pipe it through `python3 -m json.tool` to confirm.

### A TYPED, source-located program model

`aria analyze --json` is the **program model an AI tool / static analyzer
consumes**. Every function node carries:

- its **declared type signature** (`signature`) — parameter names + types, the
  return type, and any generic `type_params` / `bounds`, rendered with the same
  pretty-printer (`typeck::show`) the diagnostics use;
- **who it calls** (`callees` / `lib_callees`) and **who calls it** (`callers`);
- **where** each call happens, to the exact `[line, col]` (`call_sites`).

So the result is a typed, source-located call graph: *who calls whom, from which
line:col, and the type signature of every function* — exactly what a tool needs
to reason about the code precisely (e.g. "this function takes an `Array[Int]`
and returns an `Int`; it is called from `main` at line 9:5; it is recursive").

Function `signature`s are **declared types**: the analyzer renders what the
source wrote (it runs after type-checking, so the program is well-formed). On top
of that, each call EDGE now also carries **per-call-site INFERRED types** —
`typed_call_sites` (see below) — so the model is not only *who calls whom* but
**what types actually flow on each call**, with generics concretely instantiated
*at the site*. That makes `aria analyze --json` a **typed-EDGE** program model:
the precise dataflow-of-types across calls.

### Typed edges: per-call-site argument & result types

Where a function's `signature` is its *declared* shape, a call site's
`typed_call_sites` entry is the **inferred, fully-resolved** type that flows on
that specific call. This is produced by the type checker
(`typeck::check_with_types`), which records each expression's span → its inferred
type and re-resolves it through the final substitution, so every recorded type is
**concrete** (no leftover unification variables; an ambiguous/never-pinned type
renders as `"?"`, never a raw internal id).

The headline is **generic instantiation per site**. The same generic `id` called
twice records *different* concrete types at each call:

```aria
fn id[T](x: T) -> T = x
fn main() -> Int = { let a = id(2); let s = id("hi"); a }
```

```json
"typed_call_sites": [
  {"callee": "id", "line": 2, "col": 30, "arg_types": ["Int"],    "result_type": "Int"},
  {"callee": "id", "line": 2, "col": 45, "arg_types": ["String"], "result_type": "String"}
]
```

`id(2)` is `Int -> Int` *at its site*; `id("hi")` is `String -> String` *at
its*. Nested calls resolve correctly too (`add(id(2), 3)` records `add`'s
arguments as `["Int","Int"]` and the inner `id(2)` as `["Int"]`), and builtin
generics are concrete (`array_len([10,20,30])` → `arg_types ["Array[Int]"]`,
`result_type "Int"`; `map_new()` → `result_type "Map[Int, Int]"`). Record and
`Float` arguments render as their concrete types (`scale(Point {..}, 3.5)` →
`["Point","Float"]`).

The typed table is **metadata**: it never changes what typeck accepts/rejects,
nor any backend codegen or program result. The structural fields
(`callees`/`callers`/`cycles`/`unused`) are computed identically with or without
it. A type unavailable for a span (a synthesized node, or a non-`Call` edge such
as a function passed by name with no applied arguments) is emitted as JSON `null`,
never a fabricated type.

### JSON schema

```json
{
  "entry": "main",
  "functions": [
    {
      "name": "b",
      "signature": {
        "type_params": [],
        "bounds":      [],
        "params":      [{"name": "n", "type": "Int"}],
        "ret":         "Int"
      },
      "line": 2,
      "user": true,
      "callees":     ["c"],
      "lib_callees": ["array_get"],
      "callers":     ["a"],
      "recursive":   false,
      "fan_in":      1,
      "fan_out":     1,
      "call_sites":  {"c": [[2, 27]], "array_get": [[2, 35]]},
      "typed_call_sites": [
        {"callee": "c",         "line": 2, "col": 27, "arg_types": [],      "result_type": "Int"},
        {"callee": "array_get", "line": 2, "col": 35, "arg_types": ["Array[Int]", "Int"], "result_type": "Int"}
      ]
    }
  ],
  "unused":      ["dead"],
  "unreachable": ["dead"],
  "cycles":      [["even","odd"], ["fact"]]
}
```

#### Field meanings

| Field | Meaning |
| --- | --- |
| `entry` | The program's entry point (`"main"`), or `null` if there is no `main`. |
| `functions[].name` | Function name. |
| `functions[].signature` | The function's **declared type signature** (rendered via `typeck::show`): `type_params` (generic params, e.g. `["T"]`, empty if none), `bounds` (trait bounds as `[var, trait]` pairs, e.g. `[["T","Show"]]`, empty if none), `params` (an ordered array of `{"name", "type"}` for each parameter), and `ret` (the return type). A no-param function has `params: []`; a unit return is `"ret": "Unit"`. These are *declared* types read off the source — no inference. |
| `functions[].line` | 1-based definition line; `0` for compiler-generated functions (trait dispatchers, lowered impl methods). |
| `functions[].user` | `true` for functions the human/model wrote; `false` for prelude / synthetic (trait) functions. Only `user` functions are eligible to be `unused`. |
| `functions[].callees` | The set of **user** functions this function calls, sorted and de-duplicated. A top-level function passed by **name** as a value (`array_map(xs, helper)`) counts as a callee (so it is not flagged dead). |
| `functions[].lib_callees` | The set of **builtin / prelude** functions this function calls, sorted. Kept separate so the user-to-user graph stays clean while the full dependency surface is visible. |
| `functions[].callers` | The inverse: user functions that call this one, sorted. |
| `functions[].recursive` | `true` iff the function calls itself directly. |
| `functions[].fan_in` | `callers.len()` — how many functions depend on this one. |
| `functions[].fan_out` | `callees.len()` — how many user functions this one depends on. |
| `functions[].call_sites` | **Precise call-site locations** of every edge out of this function: an object keyed by callee NAME (user, library, or builtin), each value the sorted, de-duplicated list of `[line, col]` positions where that callee is called in this function's body. A source-located call graph — jump to the exact `callee(..)`, not just the fact that the edge exists. Bare local-variable references are NOT calls and never appear here. Synthesized calls (no source span) contribute no site; a function with no located calls reports `{}`. |
| `functions[].typed_call_sites` | **Typed call EDGES**: the per-call-site INFERRED types, as an array (one object per located call, sorted by `(line, col)` then callee). Each object is `{"callee", "line", "col", "arg_types", "result_type"}`: `callee` is the same name space as `call_sites`; `line`/`col` is the call expression's position; `arg_types` is the inferred CONCRETE type of each argument expression, in source order (a generic call shows its concrete instantiation *at this site*); `result_type` is the call's inferred result type. A type unavailable for a span is `null`. A function passed by NAME as a value (no applied args) records a zero-`arg_types` site whose `result_type` is its (instantiated) function type. Empty `[]` when the function makes no located calls. This is the precise "what types flow on this call" view, with generics instantiated per site. |
| `unused` | User functions with **no callers** and not `main` — dead code. |
| `unreachable` | User functions **not statically reachable** from `main` (transitive closure over user-to-user edges). Empty if there is no `main`. A function can be unreachable without being directly `unused` (e.g. only called by an unused function). |
| `cycles` | Recursive groups: strongly-connected components of size > 1 (mutual recursion), plus self-recursive singletons. Each inner array is one cycle (sorted); the outer list is sorted for stable, diffable output. Computed with Tarjan's SCC algorithm (iterative, no external deps). |

### How an AI tool / static analyzer uses it

- **Typed reasoning** — `signature` gives the model the *types* each function
  operates over (params, return, generics + bounds) without re-running the
  checker: it can verify an edge is type-compatible, suggest a call, or report a
  signature mismatch precisely.
- **Dataflow-of-types across calls** — `typed_call_sites` gives the *inferred*
  type that flows on each individual call: the model sees that `id` is used at
  `Int` here and `String` there (the concrete generic instantiation per site),
  that a nested `add(id(2), 3)` carries `Int` arguments, and that a builtin call
  resolves to `Array[Int]` / `Map[Int, Int]`. It can trace a value's type from
  caller to callee without re-running inference, spot a site where a generic is
  instantiated unexpectedly, or confirm an argument's concrete type before
  suggesting a refactor.
- **Impact analysis** — *"what breaks if I change `f`?"* -> `callers` / `fan_in`,
  and the transitive closure of callers.
- **Dependency understanding** — *"what does `f` rely on?"* -> `callees` +
  `lib_callees`.
- **Dead-code detection** — `unused` / `unreachable` tell the model (or a linter)
  which functions can be deleted, or signal a missing call it forgot to wire up.
- **Termination / recursion review** — `recursive` and `cycles` flag the
  functions that must have a terminating base case; an AI reviewer can focus
  there.
- **Refactoring hot-spots** — high `fan_in` functions are shared utilities
  (change carefully); high `fan_out` functions are orchestrators.

### Human output

Without `--json`, `aria analyze` prints a readable summary: the entry point and
function count, then per function its **rendered signature** (e.g.
`fn fib(n: Int) -> Int` or `fn id[T](x: T) -> T`) with its definition line,
followed by `fan_in`/`fan_out`, `recursive`, the `calls` / `uses(lib)` /
`called by` lists (library functions tagged `[library]`), and up to a couple of
**typed call lines** (`call: callee(ArgT, ..) -> RetT @ L:C`) showing the
inferred per-site types (extra sites summarized as `(+N more …)` to keep it
readable), then the `unused`, `unreachable`, and `recursive cycles` sections
(cycles rendered as `a <-> b (mutual)` or `f (self-recursive)`). For example:

```text
fn main() -> Int  (line 3)
  fan_in=0 fan_out=2
  calls:     add, id
  call:      add(Int, Int) -> Int  @ 3:20
  call:      id(Int) -> Int  @ 3:24
fn id[T](x: T) -> T  (line 2)
  fan_in=1 fan_out=0
  called by: main
```

### Scope (honest notes)

- Edges are built from `Expr::Call` **names** (and bare function-name references
  used as values) walked over each body. **Higher-order calls** through a
  function-valued parameter (`f(x)` where `f` is a parameter) cannot be resolved
  to a static target and are **not** edges. A function passed by name *is* an
  edge.
- Constructor applications are **data**, not control flow, and are not part of
  the call graph.
- A call to a builtin or prelude function appears under `lib_callees`, never as a
  user edge, and those functions are never reported as "unused user code".
- `typed_call_sites` types are the checker's INFERRED, fully-resolved types at the
  site. A type the checker could not pin to a concrete type (rare on a well-typed
  program — e.g. a result fed nowhere) renders as `"?"` in the type string, and a
  span with no recorded type at all is `null`; neither fabricates a type. The
  types are observation only and never alter what typeck accepts or any backend's
  output.

## Data-flow (`dataflow`)

Alongside the call graph, `aria analyze --json` emits a top-level **`dataflow`**
object that models how DATA moves *inside* each user function: the exact
**def-use chain** of every local binding. Where the call graph answers "how do
functions depend on each other?", the data-flow layer answers "how is each
variable defined and used?".

### The single-assignment exactness property

Aria is **pure / immutable**: every binding — a function parameter, a `let`, a
lambda parameter, or a match-arm pattern variable — is assigned **exactly once**.
There is no mutation, no reassignment, no `var`. This makes def-use chains
**EXACT**: a variable read binds to the one lexically-innermost binding of that
name, with no reaching-definitions lattice and no flow-sensitive merge. The
reported `uses` of a binding are therefore its **complete** set of reads — every
read, and nothing but reads of *that* binding. (A name shadowed by an inner
binding of the same name does not collect the inner uses; they belong to the
inner binding.)

### JSON schema

`dataflow` is an object keyed by **function name**; each value is:

```json
{
  "bindings": [
    { "name": "x", "kind": "param", "def": [line, col], "type": "Int" | null,
      "uses": [[line, col], ...], "use_count": N, "unused": false }
  ],
  "unused_bindings": [ { "name": "tmp", "def": [line, col] } ],
  "shadows": [ { "name": "x", "def": [line, col], "shadows": [line, col] } ]
}
```

- **`kind`** is one of `"param"`, `"let"`, `"lambda_param"`, `"match_binder"`.
- **`def`** is the 1-based `[line, col]` of the binder's definition site.
- **`type`** is the binding's rendered type: the *declared* type for a parameter
  or lambda parameter, the *inferred* type for a `let` (or a match binder,
  resolved from its first use). `null` when no type was recorded for that span.
- **`uses`** lists every read location (sorted, de-duplicated); **`use_count`**
  is its length; **`unused`** is `use_count == 0`.
- **`unused_bindings`** is the dead-binding subset (defined, never read), each
  with its def location. **`shadows`** reports every binding whose name shadows
  an in-scope outer binding, giving the inner def (`def`) and the outer def it
  shadows (`shadows`).

Scoping mirrors the call graph's lexical walk exactly: parameters are in scope
for the whole body, a `let` for the remainder of its block, lambda parameters for
the lambda body, and match-arm pattern variables (including `Point { x, y }`
record-field shorthand and constructor sub-binders) for the arm body. Top-level
function names, prelude functions, and builtins are **not** local bindings and
never appear here.

### Human output

The human `aria analyze` output appends a concise `data-flow:` section listing,
per function with notable facts, its unused bindings and shadows, e.g.

```text
data-flow:
  g: unused: `tmp` (line 2); shadows: `x` (line 3 shadows line 1)
```

### AI / tooling usage

The `dataflow` object gives an AI or static-analysis tool a precise, deterministic
view of intra-function data movement: "where is this variable used?" (`uses`),
"is this binding dead?" (`unused` / `unused_bindings`), and "does this name
shadow an outer one?" (`shadows`). Because of single-assignment exactness, a tool
can rely on `uses` being the literal, complete read set — safe to drive a rename,
a dead-code removal, or a shadowing cleanup. Dead `let` bindings are also surfaced
as **`W0001` warnings** in `aria check --json` and the LSP (see
`docs/DIAGNOSTICS.md`). The data-flow layer is pure **metadata**: it never changes
any call-graph field, what `typeck` accepts, an exit code, or a backend's output.
