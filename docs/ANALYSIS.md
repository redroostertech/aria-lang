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
  at `inner` (line 12)
  at `middle` (line 8)
  at `main` (line 3)
```

knows exactly which function (and definition line) trapped and the chain that
reached it — a far stronger fix signal than the bare message. The agent loop
(`aria agent`) feeds this trace back to the model on a clean-checking but
runtime-failing program, closing the loop **write -> check -> RUN -> fix**.

### Format

`aria run` on an **erroring** program prints, to **stderr**:

```
runtime error: <message>
  at `<function>` (line <N>)
  at `<function>` (line <N>)
  ...
```

- **Most-recent call first**: the function that trapped is the first `at` line;
  `main` is last.
- The line is the function's **definition line** (the `fn` keyword), 1-based.
- A compiler-generated function (a trait dispatcher, or an applied closure /
  lambda that has no single source line) prints `(generated)` instead of a line
  number.
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
pub struct Frame { pub function: String, pub line: usize }   // line 0 => generated
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
- **Frame lines are FUNCTION-DEFINITION lines in v1.** Exact **call-site** line
  precision (the line of the specific call that failed) is the planned next step
  — the same precise-span work that will upgrade the LSP and structured
  diagnostics. It requires threading source spans onto `Expr::Call`, which is an
  explicit follow-on and intentionally **not** done here.

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

### JSON schema

```json
{
  "entry": "main",
  "functions": [
    {
      "name": "b",
      "line": 2,
      "user": true,
      "callees":     ["c"],
      "lib_callees": ["array_get"],
      "callers":     ["a"],
      "recursive":   false,
      "fan_in":      1,
      "fan_out":     1
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
| `functions[].line` | 1-based definition line; `0` for compiler-generated functions (trait dispatchers, lowered impl methods). |
| `functions[].user` | `true` for functions the human/model wrote; `false` for prelude / synthetic (trait) functions. Only `user` functions are eligible to be `unused`. |
| `functions[].callees` | The set of **user** functions this function calls, sorted and de-duplicated. A top-level function passed by **name** as a value (`array_map(xs, helper)`) counts as a callee (so it is not flagged dead). |
| `functions[].lib_callees` | The set of **builtin / prelude** functions this function calls, sorted. Kept separate so the user-to-user graph stays clean while the full dependency surface is visible. |
| `functions[].callers` | The inverse: user functions that call this one, sorted. |
| `functions[].recursive` | `true` iff the function calls itself directly. |
| `functions[].fan_in` | `callers.len()` — how many functions depend on this one. |
| `functions[].fan_out` | `callees.len()` — how many user functions this one depends on. |
| `unused` | User functions with **no callers** and not `main` — dead code. |
| `unreachable` | User functions **not statically reachable** from `main` (transitive closure over user-to-user edges). Empty if there is no `main`. A function can be unreachable without being directly `unused` (e.g. only called by an unused function). |
| `cycles` | Recursive groups: strongly-connected components of size > 1 (mutual recursion), plus self-recursive singletons. Each inner array is one cycle (sorted); the outer list is sorted for stable, diffable output. Computed with Tarjan's SCC algorithm (iterative, no external deps). |

### How an AI tool / static analyzer uses it

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
function count, then per-function `fan_in`/`fan_out`, `recursive`, and the
`calls` / `uses(lib)` / `called by` lists (library functions tagged
`[library]`), followed by the `unused`, `unreachable`, and `recursive cycles`
sections (cycles rendered as `a <-> b (mutual)` or `f (self-recursive)`).

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
