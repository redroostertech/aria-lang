# Aria

**An AI-native programming language — a research prototype.**

Aria is an experiment in designing a programming language *optimized for large
language model consumption*: a language whose syntax and semantics minimize the
kinds of mistakes models make, while staying efficient for computers to compile
and run.

> Status: early prototype. The frontend (lexer + parser + type checker), a
> tree-walking interpreter, a typed IR with Perceus reference counting, and
> **two compiling backends** work today: a WASM backend and a native backend
> (Aria → C → `cc -O2`). The native backend currently covers a subset —
> `Int`/`Bool`, functions, recursion, and algebraic data types — and is what the
> [Performance](#performance) benchmarks below measure.

## Why another language?

Languages optimized for *humans* (familiar, flexible, forgiving) and languages
optimized for *machines* (dense, low-level, explicit) are both poorly suited to
*models*. The properties that actually reduce LLM error are different:

| Design rule | Why it's AI-native |
|---|---|
| **One canonical form** — `;` terminates statements, single comment style (`--`), a required leading `\|` on sum types (no optional spellings) | Removes stylistic degrees of freedom that cause inconsistent generation. *(Full canonicalization — e.g. trailing commas — is being formalized in the spec.)* |
| **Case = meaning** — `Uppercase` is always a type/constructor, `lowercase` always a value/function | Zero-ambiguity identifier resolution; friendly to grammar-constrained decoding |
| **Everything is an expression** — `if`, `match`, blocks all return values | Uniform reasoning; no statement/expression split to track |
| **Immutable `let` only** | Local reasoning — line N never depends on hidden mutation |
| **Algebraic data types + exhaustive `match`** | The highest-leverage feature for correct code generation |
| **Regular, LL(1)-ish grammar** | *Designed* to drive a GBNF grammar for constrained decoding (so a model could not emit a syntax error). **GBNF export is implemented** (`aria gbnf`) and validated against the parser on the example programs. |
| **Zero-dependency, fast compile** | Tight feedback loop; efficient compiling/bundling from day one |

## Quick start

Requires a Rust toolchain.

```sh
cargo build --release
./target/release/aria run   examples/intro.aria
./target/release/aria run   examples/list.aria
./target/release/aria check examples/broken.aria  # see the type checker reject bad code
./target/release/aria ast   examples/intro.aria   # dump the parsed AST
```

`aria run` type-checks before executing. `aria check` type-checks only. The
checker catches type mismatches, arity/field errors, unbound variables, and
**non-exhaustive `match`** — e.g. on the intentionally-broken example:

```
type errors in examples/broken.aria:
  - function `code`: non-exhaustive match on Color: missing case(s) Blue
  - function `wrong_return`: body has type Bool but return type is Int
  - function `bad_compare`: cannot compare Int and String
```

## A taste

```aria
-- Uppercase = type/constructor, lowercase = value/function.
type Shape =
  | Circle(Float)
  | Rect(Float, Float)

fn area(s: Shape) -> Float =
  match s {
    Circle(r)  => 3.14159 * r * r,
    Rect(w, h) => w * h,
  }

fn main() -> Int = {
  print_float(area(Circle(2.0)));
  0
}
```

## Performance

Aria's native backend compiles a program to C and builds it with `cc -O2`, so
these numbers compare **AOT-compiled native code** (Aria) against the **CPython
bytecode interpreter** (Python 3.11) and a **JIT** (V8, in Node 20). The point
is not "Aria beats Python/Node" — it's that the *same algorithm*, written in
each language and producing *byte-identical output*, runs at native speed once
it leaves Aria's compiled subset.

Methodology, stated plainly so you can reproduce or distrust it:

- **Same algorithm in all three languages**, outputs verified identical before timing (`fib(38)=39088169`, etc.).
- **Aria = native binary**, compiled once (`aria native x.aria x.bin`), then the binary is timed — this is clang `-O2` codegen, not an interpreter.
- **Best-of-3 wall-clock**, program output suppressed; sizes chosen so compute dominates process startup (startup baselines below).
- Machine: Intel i7-9750H, macOS (x86_64). Python 3.11.0, Node v20.16.0, Apple clang 17 (`cc -O2 -std=c11`).
- Sources + harness live in [`benchmarks/`](benchmarks/) (`python3 benchmarks/run.py` reproduces the table).

Because the test machine is a thermally-variable 2019 laptop, absolute
millisecond figures drifted run-to-run (Aria's own `fib` ranged 191–304 ms
across five runs). The **speed-up ratios are the robust signal** — all three
languages run on the same loaded machine, so load cancels out. We report a
representative run's times plus the **observed ratio range across 5 runs**:

| benchmark | stresses | Aria | Python 3.11 | Node 20 | vs Python | vs Node |
|---|---|---:|---:|---:|---:|---:|
| `fib(38)` | recursion / call overhead | ~190 ms | ~6,100 ms | ~630 ms | **31–32× faster** | **3.2–3.6× faster** |
| `loopsum` (Σ 1‥100M) | tight integer loop¹ | ~48 ms | ~15,000 ms | ~190 ms | **300–500× faster** | **3.8–6.3× faster** |
| `collatz` (1‥1M) | branches + integer math | ~230 ms | ~11,300 ms | ~1,600 ms | **45–52× faster** | **6.0–7.3× faster** |
| `listsum` (20M cons cells) | allocation / memory model | ~1,700 ms | ~11,300 ms | ~1,200 ms | **6.7–7.0× faster** | **0.7–0.8× — *slower*** |

¹ Aria has no loop syntax in this subset; the loop is written as self-tail-recursion, which the C backend lowers to a `goto` loop (no stack growth).

**The honest result.** On compute-bound integer code Aria is **~30–500× faster
than CPython and ~3–7× faster than Node** — expected for native code vs an
interpreter, and a real win even over V8's JIT. But `listsum` is the interesting
one: Aria is **~1.3× *slower* than Node**, and we're leaving it in the table.
That benchmark allocates and frees 20 million cons cells, and Aria's Perceus
reference counting does 20M individual `malloc`/`free` pairs, while V8's
generational GC bump-allocates short-lived garbage in a nursery and reclaims it
in bulk. That's the genuine trade-off of precise RC: **deterministic,
pause-free, immediate reclamation (and garbage-free-verified — `aria_live==0` at
exit), but a per-allocation cost** a bump-and-sweep GC amortizes away. Aria is
not magic; it's fast where native compilation pays off and competitive-to-behind
where allocator strategy dominates.

Supporting figures (same machine, representative):

- **Process startup** (empty program, best of 8): Aria native binary **~17 ms**, Python **~152 ms**, Node **~200 ms**. The benchmarks above are sized so this fixed cost doesn't flatter Aria.
- **Aria compile time** (parse → typecheck → monomorphize → IR → RC/reuse → emit C → `cc -O2`): **~0.4 s** per benchmark; resulting binaries are **~8.7 KB**.

**Caveats.** These are microbenchmarks on the compiled subset (`Int`/`Bool`,
functions, recursion, ADTs) — not the whole language, not application workloads,
and a single machine. Python and Node are general-purpose runtimes doing far
more than raw arithmetic; this measures identical algorithms at the compute
level and nothing beyond that.

## Limitations & honest status

Aria is a **research prototype**. An independent adversarial audit of every claim
in this README (run against the source and by executing the tools) is written up
in **[docs/CLAIMS.md](docs/CLAIMS.md)**. The headline findings — including where
we *overstated* — are:

**What's genuinely real (verified):** the type checker (HM generics, exhaustive
`match`, rigid type params, a purity effect system), the typed IR with
zero-annotation Perceus reference counting (garbage-free verified at runtime, 50%
allocations eliminated by reuse on the map benchmark), three backends
(interpreter, hand-emitted WASM, native-via-C) that agree under differential
fuzzing, a correct order-0 rANS coder, a real PAQ-style context-mixing predictor,
a correct (tiny, untrained) causal-transformer forward pass, and a validated GBNF
exporter. The compression win (2.5× gzip) reproduces exactly.

**Where we corrected or softened claims during the audit:**
- The neural codec is **~21% larger than `gzip -9`** on the test corpus (not "on par") and runs at only **~1 MB/s**.
- "Embeddings" (`embed_similarity`, RAG demo) are **FNV-1a token hashing** — lexical bag-of-words, *not* a learned/semantic model.
- The "type-aware" compression win is a **hand-coded transform for one synthetic dataset**, not driven by the language's type system; off-domain (source code, text, random) the order-0 rANS coder **loses 1.5–3.7× to gzip**.
- The transformer uses **untrained random weights** — a correctness demo, not a useful model.

**The biggest gap — the data model.** For a self-described *AI-native* language,
the data structures are the weakest part today:

| Missing / partial | Status |
|---|---|
| **Loops** | No `for`/`while`. Iteration is recursion (native tail-calls → `goto` loop, no stack growth). |
| **Arrays / lists / tuples / records** | None built-in. Linked lists are hand-rolled ADTs (O(n), one heap cell/element). |
| **Maps / sets** | None. |
| **Vectors / embeddings** | `Tensor` is an opaque 2-D float matrix (interpreter + WASM only, no native, no indexing syntax). No 1-D vector type; embeddings are never exposed as a value. |
| **Bytes / binary buffers** | None. `Str` is the only byte-sequence type. |
| **Compression as a language API** | The codec is a standalone Rust library; Aria reaches it only via two `Str`-only, interpreter-only builtins. |

Closing this gap (a first-class `Array`/`Vector`/`Bytes`/`Map` layer, with the
shape checker in `src/shape.rs` wired into the type system) is the **top
priority** to make the "AI-native" claim real. The full production-readiness plan
— data model, tooling (LSP, formatter, REPL, Cursor/VS Code plugins), deployment,
and FFI/"drivers" — is in **[docs/ROADMAP.md](docs/ROADMAP.md)**. A how-to and
language reference is in **[docs/GUIDE.md](docs/GUIDE.md)**.

## AI-native primitives (callable from Aria)

LLM/numeric operations are exposed as language-runtime builtins — not a library
bolted on — so you write them directly in typed Aria code:

```aria
fn main() -> Int = {
  let a   = tensor_set(tensor_zeros(2, 2), 0, 0, 1.0);  -- opaque Tensor handles
  let prod = matmul(a, a);
  print_float(tensor_get(prod, 0, 0));
  print_float(embed_similarity("cosine similarity", "vectors by cosine"));  -- RAG
  print_int(compressed_size("abababababab"));                                -- compression
  print_float(neural_bits_per_byte("the quick brown fox the quick"));        -- predictor
  0
}
```

Builtins: `tensor_zeros/set/get/rows/cols`, `matmul`, `transpose`, `softmax`,
`relu` (typed `Tensor`); `embed_similarity` (RAG cosine); `compressed_size`
(rANS); `neural_bits_per_byte` (predictive model). See `examples/ai.aria`.

## Compression engine (research focus)

A core thesis of Aria: because the language knows the *shape* of data, it can
compress far better than byte-blind tools like `.zip`. The engine is layered as
**model → entropy coder**, where smarter models plug into the same back end:

- **Entropy back end:** rANS (the coder behind Zstandard/FSE) — near-optimal bits, faster than Huffman. (`src/rans.rs`)
- **Type-aware model:** columnar split + delta + zig-zag transforms driven by data shape. (`src/pack.rs`)
- **Roadmap:** context-modeling, then a **neural/predictive** tier (a model predicts the next token, an arithmetic coder records only the surprise) — fully **lossless**, the "LLMs are SOTA compressors" frontier.

```sh
cargo run --release -- bench          # benchmark vs gzip on synthetic telemetry
cargo run --release -- pack   in out  # compress any file (rANS, order-0 entropy)
cargo run --release -- unpack in out  # decompress
cargo run --release -- npack  in out  # compress with the predictive (neural) codec
cargo run --release -- nunpack in out # decompress
cargo run --release -- demo           # run the AI-native runtime demos:
cargo run --release -- demo transformer  # tiny transformer forward pass
cargo run --release -- demo predict      # context-mixing predictor (bits/byte)
cargo run --release -- demo shape        # compile-time tensor shape checking
cargo run --release -- demo rag          # native retrieval (embedding top-k)
cargo run --release -- mem    file.aria  # lower the Int/Bool/ADT subset to IR,
                                         # cross-check vs interpreter, count ADT allocations
```

> `aria mem` inserts precise Perceus-style `dup`/`drop` **and reuse analysis**,
> then reports fresh allocations vs in-place reuses, frees, and peak live cells.
> It verifies **garbage-freeness** (no cell live at exit, zero annotations) and
> cross-checks the result against the interpreter. On the map benchmark, reuse
> eliminates 50% of allocations (a unique list is mutated in place). This is a
> POC on the Int/Bool/ADT subset — not yet the whole language or a native backend.

Sample run (200k-row synthetic telemetry, 3 × i64 columns). Sizes are
deterministic; times are from one representative run and vary by machine/load:

| method | size | ratio | time (representative) |
|---|---:|---:|---:|
| raw (i64 row-major) | 4,800,000 | 100.0% | – |
| gzip -9 (zip-class) | 1,021,751 | 21.3% | several seconds |
| Aria rANS (entropy only) | 1,685,558 | 35.1% | sub-second |
| **Aria type-aware + rANS** | **406,180** | **8.5%** | sub-second |

→ **2.5× smaller than `gzip -9`** (deterministic; reproduced across runs) **and much faster**
— sub-second vs gzip's multiple seconds on this dataset (the exact speed multiple, ~25–40×,
varies by machine/load, so we don't pin a single number). Fully lossless (round-trip verified).
*Caveat: this is a synthetic best-case for columnar data (monotonic/cyclic/slow-drifting
integer columns) — exactly where delta+columnar beats a byte-blind tool. The codec is also a
standalone Rust library today; it is not yet driven by an Aria program's own type information.*

**Predictive (neural) codec** — a context-mixing predictor feeds a binary arithmetic coder
(`model + entropy coder = optimal compression`, the "LLMs are compressors" architecture).
On a mixed text corpus (repo prose + Rust source, 143 KB) it is **1.8× smaller than the
order-0 rANS codec**, lossless — but **~21% *larger* than `gzip -9`** on that corpus, so it
does **not** yet match gzip (an earlier "on par with gzip" claim was an overstatement). It
also runs at only **~1 MB/s** (two-plus orders of magnitude slower than gzip), which is the
dominant practical caveat. It loses to gzip because the predictor lacks LZ-style long-match
modeling — the next upgrade is a match model / higher-order contexts, then a true neural
(transformer) predictor. ("Neural" here means a PAQ-style context-mixing predictor with an
online-trained logistic mixer — a real adaptive statistical model, not a neural network.)

## Architecture

```
src/
  lexer.rs       hand-written tokenizer (comments, literals, operators)
  ast.rs         the abstract syntax tree
  parser.rs      recursive descent + Pratt precedence
  typeck.rs      static type checker + match exhaustiveness
  interp.rs      tree-walking interpreter (the reference backend)
  ir.rs          typed ANF IR (differentially checked vs the tree-walker)
  rc.rs          Perceus reference counting + reuse analysis (FBIP)
  wasm.rs        WASM backend (hand-emitted .wasm)
  c_backend.rs   native backend — lowers the IR to C, built with `cc -O2`
  gbnf.rs        GBNF grammar export for constrained decoding
  main.rs        CLI dispatch
```

The frontend is kept strictly separate from the backends: the interpreter, the
WASM backend, and the native C backend all consume the same checked IR, so a new
code generator can be added without touching the parser or AST.

## Roadmap

- [x] Lexer, parser, AST
- [x] Tree-walking interpreter (functions, recursion, ADTs, pattern matching, blocks)
- [x] Static type checker (bottom-up synthesis + checking against annotations)
- [x] Exhaustiveness checking for `match`
- [x] **Generics** — generic ADTs + generic functions with Hindley–Milner inference and rigid (skolem) type params (`examples/generic.aria`)
- [ ] `let`-generalization (let-bound values are not yet generalized to polymorphic schemes)
- [x] Typed ANF IR + IR interpreter (differentially checked vs the tree-walker)
- [x] Precise Perceus-style reference counting (zero-annotation, garbage-free verified)
- [x] Reuse analysis (FBIP) — unique cells mutated in place; **50% of allocations eliminated** on the map benchmark, zero annotations
- [x] rANS entropy coder + type-aware compression (beats gzip on structured data)
- [x] Shaped-tensor runtime (matmul/softmax/layernorm) + INT8 quantization
- [x] Transformer forward pass (inference) running on the tensor core
- [x] Arithmetic coder + context-mixing predictor (predictive-compression building blocks)
- [x] Native RAG primitives (embedding store + cosine top-k retrieval)
- [x] Compile-time tensor shape checking — *standalone prototype* (`src/shape.rs`, run via `aria demo shape`); not yet wired into the language's own type checker
- [x] Wire predictor + arithmetic coder into an end-to-end neural codec (`aria npack`)
- [ ] Add a match model / higher-order contexts (beat gzip), then a transformer predictor
- [x] **WASM backend (Phase 2a)** — compiles the pure Int/Bool/function subset to a real `.wasm` binary (hand-emitted), runs via Node, differentially tested vs the interpreter (`aria wasm` / `aria wasm-run`). Integer overflow is a defined error: the compiled wasm **traps** on `+`/`-`/`*`/negation overflow, matching the interpreter's checked-error semantics (the two backends agree).
- [x] **WASM Phase 2b** — ADTs + **strings** on linear memory with reference counting (`dup`/`drop`) and **in-place reuse (FBIP)** lowered to wasm: a bump+free-list allocator, per-constructor recursive drop, `concat`/`int_to_str`/string-`==`, `print_str` via a host import. **Garbage-free verified in the compiled output** (`__live()==0`); reuse eliminates ~50% of allocations in compiled code (`__reuses` counter); continuously fuzzed against the interpreter oracle (sampled `wasm == interp` + garbage-free).
- [x] **Native backend** — lowers the `Int`/`Bool`/function/ADT subset to C, builds with `cc -O2` (`aria native` / `aria native-run`); self-tail-recursion becomes a `goto` loop, ADTs use Perceus RC (garbage-free verified). **~30–500× faster than CPython, ~3–7× faster than Node** on compute-bound integer code — see [Performance](#performance).
- [x] **GBNF grammar export** for constrained decoding (`aria gbnf` emits a grammar for the full surface syntax)
- [x] **Purity effect system** — a single `IO` effect inferred by least-fixpoint over the call graph; `pure fn` is compiler-proven (`examples/pure.aria`)
- [ ] Full capability/effect system (beyond the binary pure/IO distinction)
- [ ] **First-class data: arrays/vectors, bytes, maps/sets** — *currently missing; the top priority for an AI-native language* (see [Limitations](#limitations--honest-status))
- [ ] Structured, machine-parseable compiler diagnostics
- [ ] Developer tooling: LSP, formatter, REPL, editor/Cursor plugins (see [docs/ROADMAP.md](docs/ROADMAP.md))

## License

MIT — see [LICENSE](LICENSE).
