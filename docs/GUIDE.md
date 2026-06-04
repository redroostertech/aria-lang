# Aria — Guide & Language Reference (v0)

> This documents **what actually works today** in the prototype. Every snippet
> here was run against `./target/release/aria`. For the honest status of bigger
> claims see [CLAIMS.md](CLAIMS.md); for where the language is going see
> [ROADMAP.md](ROADMAP.md).

---

## 1. What is Aria, and what is it for?

Aria is an experiment in designing a programming language **optimized for large
language models to write correctly**, while compiling to fast native code. The
bet is that the properties that reduce *human* error (familiarity, flexibility)
and the properties that reduce *machine* cost (density, low-level control) are
both different from the properties that reduce *LLM* error.

The LLM-error-reducing properties Aria is built around:

- **One canonical form.** A single comment style (`--`), required leading `|` on
  sum types, mandatory return types — fewer stylistic degrees of freedom for a
  model to be inconsistent about.
- **Case = meaning.** `Uppercase` is always a type or constructor; `lowercase`
  is always a value or function. Zero-ambiguity name resolution.
- **Everything is an expression.** `if`, `match`, and blocks all return values.
- **Immutability.** `let` only; there is no assignment syntax at all, so line N
  never depends on hidden mutation.
- **ADTs + exhaustive `match`.** The single highest-leverage feature for correct
  generation — the compiler rejects a `match` that forgets a case.
- **A grammar designed for constrained decoding.** `aria gbnf` emits a real
  GBNF grammar so a model *cannot* emit a syntax error when decoding under it.

**Who is it for right now?** Researchers and tinkerers exploring
language-design-for-LLMs, AOT compilation via C, and the Perceus reference-
counting memory model. It is **not** production-ready (no package manager, no
collections, no IO beyond `print_*`, no editor tooling — see
[ROADMAP.md](ROADMAP.md)).

### What you can build with it today (honestly)

Pure, typed, compute-only programs over `Int`/`Float`/`Bool`/`String` and your
own algebraic data types: recursive algorithms, interpreters, numeric kernels,
tree/list transforms. Programs compile to a **standalone ~9 KB native binary**
with no runtime and no GC pauses (precise reference counting). What you *cannot*
build yet: anything needing files, network, arrays, maps, real IO, or
concurrency.

---

## 2. Install & build

Requires a Rust toolchain (and, for the compiling backends, a C compiler `cc`
and/or `node`).

```sh
cargo build --release          # produces ./target/release/aria
./target/release/aria          # prints the subcommand list
```

---

## 3. Your first programs (all verified)

### Hello, values

```aria
fn main() -> Int = {
  print_int(42);
  print_str("hello world");
  print_bool(true);
  print_float(3.5);
  0
}
```
```sh
$ aria run hello.aria
42
hello world
true
3.5
```

> **Gotcha that bites everyone:** `main`'s returned `Int` becomes the process
> **exit code** (`value & 0xff`). `fn main() -> Int = fib(10)` prints nothing and
> exits `55`. To *see* a result, `print_int` it and return `0`.

### Algebraic data types + match

```aria
type Shape =
  | Circle(Float)
  | Rect(Float, Float)

fn area(s: Shape) -> Float =
  match s {
    Circle(r)  => 3.14159 * r * r,
    Rect(w, h) => w * h,
  }

fn main() -> Int = { print_float(area(Circle(2.0))); 0 }
```

Forget the `Rect` arm and the compiler refuses to build:
`non-exhaustive match on Shape: missing case(s) Rect`.

### Recursion instead of loops

Aria has **no loop syntax**. You iterate with recursion; write it
tail-recursively and the native backend turns it into a `goto` loop (no stack
growth):

```aria
fn sum_to(n: Int, acc: Int) -> Int =
  if n == 0 { acc } else { sum_to(n - 1, acc + n) }

fn main() -> Int = { print_int(sum_to(1000000, 0)); 0 }
```

### Generics + a hand-rolled list

There are **no built-in collections**. A list is an ADT you declare:

```aria
type List[T] = | Nil | Cons(T, List[T])

fn length[T](xs: List[T]) -> Int =
  match xs { Nil => 0, Cons(_, rest) => 1 + length(rest), }

fn sum(xs: List[Int]) -> Int =
  match xs { Nil => 0, Cons(h, t) => h + sum(t), }

fn main() -> Int = {
  let xs = Cons(1, Cons(2, Cons(3, Cons(4, Nil))));
  print_int(sum(xs));      -- 10
  print_int(length(xs));   -- 4
  0
}
```

Type arguments are inferred (Hindley–Milner) — you write `Cons(1, Nil)`, never
`Cons[Int](1, Nil[Int])`.

### Higher-order functions & lambdas

```aria
fn apply(f: (Int) -> Int, x: Int) -> Int = f(x)
fn main() -> Int = apply(\n -> n * n, 6)   -- exit code 36
```

Lambdas are `\x -> e` (inferred param) or `\(x: Int) -> e` (annotated); they
close over scope.

---

## 4. Language reference (the whole surface)

| Category | What exists | Notes |
|---|---|---|
| **Comments** | `-- to end of line` | no block comments |
| **Types** | `Int` (i64), `Float` (f64), `Bool`, `String`/`Str`, `Unit`, user ADTs, `Tensor` (opaque), function types `(A,B)->C` | no arrays/lists/maps/tuples/records/bytes |
| **Type decls** | `type Name[T,..] = \| A \| B(T1,T2)` | leading `\|` required; Uppercase ctors |
| **Functions** | `fn f[T](x: T) -> R = expr`, optional `pure fn` | return type mandatory; recursion + mutual recursion |
| **Expressions** | `if c {..} else {..}`, `match`, `let x = e;`, blocks `{ s; s; result }`, calls, ctor application, lambdas | everything is an expression |
| **Patterns** | `_`, var, Int literal, Bool literal, `Ctor(p, ..)` | no float/string-literal patterns |
| **Operators** | `+ - * / %`, `== != < <= > >=`, `&& \|\| !`, unary `-` | same ops for Int and Float; **no Int/Float mixing** |
| **Literals** | `42`, `3.5`, `true`/`false`, `"...\n\t\\\""` | no hex/underscore/exponent int literals |
| **Builtins** | `print_int/float/bool/str`, `concat`, `int_to_str`, plus the AI builtins below | only `print_*` are effectful (`IO`) |

**AI builtins** (interpreter; tensors/embed also run on WASM — see matrix):
`tensor_zeros/set/get/rows/cols`, `matmul`, `transpose`, `softmax`, `relu`
(typed `Tensor`); `embed_similarity` (lexical cosine); `compressed_size` (rANS
byte count); `neural_bits_per_byte` (context-mixing predictor).

---

## 5. The toolchain (every subcommand)

| Command | Purpose | Example |
|---|---|---|
| `aria run <f>` | Typecheck + tree-walk interpret | `aria run examples/intro.aria` |
| `aria check <f>` | Typecheck only (~instant) | `aria check examples/hof.aria` |
| `aria ast <f>` | Dump parsed AST | `aria ast prog.aria` |
| `aria mem <f>` | Lower to IR + RC, run, print alloc/reuse + garbage-free | `aria mem examples/mem_bench.aria` |
| `aria native <f> <out>` | Transpile to C, build native exe via `cc -O2` | `aria native prog.aria prog.bin` |
| `aria native-run <f>` | native compile + run | `aria native-run prog.aria` |
| `aria wasm <f> <out.wasm>` | Emit a real `.wasm` binary | `aria wasm prog.aria prog.wasm` |
| `aria wasm-run <f>` | wasm compile + run under `node` | `aria wasm-run prog.aria` |
| `aria gbnf [out]` | Emit a GBNF grammar of Aria's syntax | `aria gbnf grammar.gbnf` |
| `aria pack/unpack <in> <out>` | rANS compress / decompress any file | `aria pack README.md r.bin` |
| `aria npack/nunpack <in> <out>` | neural-codec compress / decompress | `aria npack notes.txt n.bin` |
| `aria bench` | Compression benchmark on synthetic telemetry | `aria bench` |
| `aria demo [transformer\|predict\|shape\|rag]` | Rust-side AI demos | `aria demo transformer` |

There is **no** `--help`, REPL, formatter, LSP, or file-watcher yet.

### Backend capability matrix (probed empirically)

| Feature | interpret (`run`) | WASM (`wasm-run`) | native (`native-run`) |
|---|:--:|:--:|:--:|
| Int/Bool/Float, comparisons, boolean ops | ✅ | ✅ | ✅ |
| String, `concat`, `int_to_str` | ✅ | ✅ | ✅ |
| User ADTs + `match` | ✅ | ✅ | ✅ |
| Lambdas / HOFs / closures | ✅ | ✅ | ✅ |
| Tensors, `matmul`, `embed_similarity` | ✅ | ✅ | ❌ (`unknown fn`) |
| Compression builtins | ✅ | ❌ | ❌ |

The **interpreter is the full surface**; WASM matches it minus the compression
builtins; native (C) additionally lacks tensors/embeddings. All three agree on
the Int/Bool/Float/String/ADT/closure core and are cross-checked by differential
fuzzing.

---

## 6. Aria vs other tools — where it fits

These are honest, narrow comparisons for the *current* prototype.

**Same recursive algorithm, three languages** (`fib`, `loopsum`, `collatz` in
[`benchmarks/`](../benchmarks/)): Aria's native binary runs **~30–500× faster
than CPython and ~3–7× faster than Node** on compute-bound integer code, while an
allocation-heavy linked-list sum runs **~1.3× slower than Node** (precise RC pays
per-allocation where V8's nursery GC bump-allocates). Reproduce with
`python3 benchmarks/run.py`.

| If you want… | Use instead of Aria, because… |
|---|---|
| A model to emit code that won't even parse-error | **Aria + GBNF** is purpose-built for this; mainstream langs have no canonical grammar for constrained decoding. This is Aria's clearest differentiator. |
| Fast numeric/array work today | **NumPy / Julia / Rust** — Aria has no arrays/vectors as values yet. |
| To ship a real service | **Go / Rust / Python** — Aria has no IO, networking, packages, or concurrency. |
| A tiny dependency-free AOT binary for a pure algorithm | **Aria native** is genuinely nice here: ~9 KB, no runtime, no GC pauses. |
| Verified memory safety without GC pauses | **Aria's Perceus RC** (garbage-free verified) or **Rust** — Aria is simpler but far less capable. |

**Bottom line:** today Aria is a compelling *research* vehicle for
LLM-friendly syntax + AOT-via-C + precise RC. It is not yet a general-purpose
language. See [ROADMAP.md](ROADMAP.md) for what would change that.

---

## 7. Developer workflow & error quality

Edit `.aria`, then `aria run file.aria` (or `aria check` for a fast typecheck-
only pass, `aria ast` to inspect parsing, `aria mem` to see allocation behavior).
Errors are specific and carry a line number (lex/parse) or function name
(type/runtime), and type errors are batched:

```
type errors in examples/broken.aria:
  - function `code`: non-exhaustive match on Color: missing case(s) Blue
  - function `wrong_return`: body has type Bool but return type is Int
  - function `bad_compare`: cannot compare Int and String
```

`aria check` is effectively instant; a native build (including `cc -O2`) is
~0.4 s for a small program.

---

## 8. Tests

```sh
RUST_MIN_STACK=67108864 cargo test     # 214 tests pass
```

The stack bump is needed because one debug-mode RC test recurses deeply; the
production CLI already runs on a large-stack thread. The suite includes a
**type-directed differential fuzzer** (random well-typed programs run through the
interpreter oracle vs the IR+RC backend, requiring identical results **and**
garbage-freeness), plus lexer/parser panic-fuzzing and WASM/codec round-trip
fuzzing.
