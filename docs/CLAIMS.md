# Aria — Claims Audit

Every claim Aria makes, checked against the actual code/tests/benchmarks on this
machine. Verdicts: ✅ supported · ⚠️ partial / softened · 🔧 fixed during audit.
Re-run the checks with `cargo test`, `aria bench`, and the probes noted below.

## Language properties

| Claim | How it was checked | Verdict |
|---|---|---|
| **Zero dependencies, fast compile** | `Cargo.toml` has no `[dependencies]`; `Cargo.lock` lists only `aria` | ✅ |
| **Immutable `let` only** (no hidden mutation) | `x = 2` after `let x = 1` is a **parse error** — no assignment syntax exists | ✅ |
| **Everything is an expression** | `if`/`match`/blocks used as values in a program; runs and returns the expected value | ✅ |
| **Case = meaning** (Uppercase=ctor, lowercase=value) | enforced in the parser; identifiers restricted to ASCII so the rule is always well-defined | ✅ |
| **ADTs + exhaustive `match`** | `examples/broken.aria` — a missing case is a compile error (`non-exhaustive match … missing Blue`) | ✅ |
| **Static types gate execution** | `aria run` type-checks first; the generic-soundness exploit is now *rejected* at type-check time | ✅ |
| **Generics with HM inference** | `examples/generic.aria` + typeck tests (instantiation, mismatch caught, rigid params) | ✅ |
| **AI primitives callable from Aria** | `examples/ai.aria` runs `matmul`/`embed_similarity`/`compressed_size`; interp tests | ✅ |
| **One canonical form — no optional syntax** | Was **false**: both `type T = A\|B` and `type T = \| A\|B` parsed. | 🔧 Leading `\|` is now **required**; one spelling. Trailing-comma optionality remains and is deferred to the grammar spec. |
| **Regular grammar → GBNF "model can't emit a syntax error"** | `aria gbnf` emits a complete GBNF grammar; `gbnf.rs` ships its own acceptor and tests that the grammar accepts the real example programs and rejects malformed input | ✅ **Now implemented and validated against the parser** (was previously unbuilt) |

## Compression

| Claim | Measured (this machine) | Verdict |
|---|---|---|
| **Type-aware codec 2.5× smaller than gzip -9** | 406,180 B (8.5%) vs gzip 1,021,751 B (21.3%) = **2.52×**, deterministic across runs | ✅ |
| **"~42× faster"** | type-aware sub-second vs gzip multiple seconds; multiple measured ~25–40×, machine/load dependent | ⚠️ Softened — size ratio pinned, speed stated as a range |
| **Type-aware codec is lossless** | `bench` asserts `restored == cols`; round-trip verified | ✅ |
| **Neural codec 1.8× smaller than rANS, ~on par with gzip** | corpus: rANS 117,472 → neural 63,911 = **1.84×**; neural vs gzip **0.86×**; round-trip **lossless** | ✅ |
| **"the language knows the shape, so it compresses better"** | the codec is a standalone Rust library; no Aria type info flows into it today | 🔧 README corrected to say so |

## Tooling / positioning

| Claim | Reality | Verdict |
|---|---|---|
| **Compile-time tensor shape checking** | `src/shape.rs` is a standalone demo IR, reachable only via `aria demo shape`; it is *not* wired into the language type checker | 🔧 README marks it a prototype |
| **Memory model (Perceus-RC)** | **implemented** in `rc.rs` over the typed IR: zero-annotation `dup`/`drop` + reuse analysis, garbage-free verified at runtime, cross-checked vs the interpreter, and lowered to *both* compiled backends | ✅ **Now real** (was "researched, not implemented") — 50% allocations eliminated on the map benchmark |
| **"compiled" language** | **two compiling backends exist**: hand-emitted WASM (`wasm.rs`) and native-via-C (`c_backend.rs`, `cc -O2`), both consuming the same checked IR and agreeing under differential fuzzing | ✅ **Now real** (was "interpreter only; compiler is the next milestone") |
| **`aria mem` / memory POC** | lowers the Int/Bool/ADT subset to ANF IR, inserts precise Perceus-style `dup`/`drop` **+ reuse analysis**, runs it (cross-checked against the tree-walker), and reports fresh allocations / reuses / frees / peak-live | ✅ Garbage-free (no cell live at exit, zero annotations) verified across list-sum, map-then-sum, shared refs, unused values, branch-only-use, scrutinee-used-after-match, heap-field-after-borrow, and trees. **Reuse eliminates 50% of allocations** on the map benchmark (unique cells mutated in place). Scope: the functional subset + an IR interpreter — *not* the whole language or a native backend yet. |

## Compiled backend (WASM, Phase 2a)

| Claim | Reality | Verdict |
|---|---|---|
| **Aria compiles to WebAssembly** | `src/wasm.rs` hand-emits a real `.wasm` for the pure Int/Bool/function subset (arithmetic, comparisons, `if`, integer `match`, recursion, short-circuit `&&`/`\|\|`); runs via Node | ✅ Differentially tested vs the interpreter oracle; unsupported features (ADTs/strings/floats/builtins) rejected with a clean error, not a panic. Heap data is Phase 2b. |
| **Compiled and interpreted Aria agree** | verified across a curated battery + adversarial edge cases (LEB128 limits, multi-byte section lengths, many functions/locals, negatives, large i64) | ✅ Integer overflow is a *defined error*: wasm **traps** on `+`/`-`/`*`/negation overflow and div/rem-by-zero (and `MIN/-1`), matching the interpreter's checked-error semantics. No silent-wrap divergence. |

### Backend agreement caveats (honest)

- **Recursion depth (wasm):** compiled wasm recursion is bounded by the host runtime's stack. Under Node it traps on ~20k-deep non-tail recursion; the `wasm-run` harness passes `--stack-size=8000` to raise this so it matches the interpreter (1 GiB thread) and native (C stack) on the cases tested (50k+). This is a *runtime stack limit*, not a miscompilation — a different wasm runtime would differ.
- **`softmax` / `embed_similarity`:** agree with the interpreter within ~1e-5, not bit-exact, because `exp` (Rust `f32::exp` vs JS `Math.exp`) is not bit-portable. All other tensor ops (`matmul`/`transpose`/`relu`/`get`) bit-match.
- **Closures / higher-order functions:** lambdas, captures, currying, and applied function values now **compile and run in BOTH compiled backends** — native (C) via a function-pointer table, and wasm via a `funcref` table + `call_indirect`. Lambdas are lambda-lifted to top-level functions and captured into reference-counted closure cells (cell tag = lambda id, fields = captures), so dup/drop/free reuse the existing Perceus machinery. Verified differentially **three-way** (interpreter / wasm / native all agree) and **garbage-free** (`__live`/`aria_live == 0`): immediate application, currying, scalar AND heap captures, generic higher-order `map` with capture + function-by-name, applied-twice, `compose` (closures capturing closures), a closure stored in an ADT then dropped *without* being applied (exercising per-tag capture release in `__drop`), and **unannotated** lambdas bound to bare `let`s (`let f = \x -> ..`) — typeck back-annotates the inferred parameter types into the AST before compilation. Lambda **arguments** are checked *bidirectionally*: the expected function type (from the callee's signature, or pinned by a sibling argument) is pushed into the lambda before its body is checked, so curried/nested lambdas typed only by context — `apply2(\x -> \y -> x + y, 30, 12)` — need no annotation. (A bare `let sq = \x -> x * x` with no context remains correctly rejected: `*` is overloaded over Int/Float and Aria has no numeric defaulting, so it is genuinely ambiguous and asks for an annotation.)
- **Codec builtins** (`compressed_size`, `neural_bits_per_byte`) are a **deliberate interpreter-only boundary**, not a stopgap. They are *measurement* builtins backed by ~1,200 lines of floating-point neural model (`transformer`/`predict`) plus an rANS entropy coder (`rans`); the interpreter is their reference implementation. Because their result depends on non-bit-portable `f32`/`exp` arithmetic (see the `softmax` caveat above), a ported backend could not even be guaranteed bit-identical — so duplicating the coder across hand-emitted wasm and C would add large surface area and divergence risk for an analysis helper that is never on a hot path. In the compiled backends they return a clean `Err` directing the user to the interpreter.

## Net

The core *language-property* claims (immutability, expressions, case rule,
exhaustiveness, zero-deps, static typing, generics, callable AI primitives) and
the *compression size* claims are all supported by code and reproducible. The
items that outran reality — optional leading pipe, the pinned speed multiple,
GBNF, shape-checking-as-a-language-feature, and "the language knows the shape" —
were either fixed or reworded to match what the code actually does.

---

## v0 adversarial audit (latest pass)

A fresh end-to-end audit — every README claim re-checked against the source and
by running the tools (5 parallel auditors). Verdicts: ✅ true · ⚠️ partial /
overstated · ❌ false.

### Confirmed real (often under-claimed)

| Claim | Evidence | Verdict |
|---|---|---|
| Static type checker (HM generics, exhaustive `match`, rigid type params) | `typeck.rs`; `aria check examples/broken.aria` reports 3 batched errors; 214 tests pass (with `RUST_MIN_STACK` bump) | ✅ + a **purity effect system** the README under-credits |
| **Generics with HM inference** | `examples/generic.aria` runs; rigid (skolem) params prevent a body constraining its own `T` | ✅ README's `[ ]` checkbox was **misleading** — fixed; only `let`-generalization is unfinished |
| Typed ANF IR, differentially checked vs tree-walker | check runs on **every** `aria mem`, not just in tests | ✅ stronger than implied |
| Perceus RC, zero-annotation, garbage-free | `aria mem examples/mem_bench.aria` → garbage-free, **50.0% reuse (1001/2002)** | ✅ |
| Native backend (Aria → C → `cc -O2`) | self-tail-recursion → `goto` loop; ADTs use RC; ~9 KB binaries | ✅ **~30–500× CPython, ~3–7× Node** on integer code (benchmarked) |
| WASM backend (2a + 2b) | real `.wasm`, ADTs/strings on linear memory, overflow traps, fuzzed vs interpreter | ✅ |
| rANS coder; **type-aware codec 2.5× gzip -9** | `aria bench` reproduces 406,180 vs 1,021,751 = 2.52×, deterministic; ~30–40× faster | ✅ |
| Transformer forward pass | `aria demo transformer` — real causal attention + MLP + layernorm | ✅ **but tiny & untrained** (random weights) — a correctness demo |
| Context-mixing predictor (`neural_bits_per_byte`, `aria demo predict`) | real PAQ-style integer predictor with online logistic mixer | ✅ (it is *statistical*, not a neural net) |
| GBNF export | `aria gbnf`; validated against parser on examples | ✅ |
| INT8 quantization, shape checker (`shape.rs`) | real; shape checker honest that it's standalone, not wired into typeck | ✅ |

### Overstated / corrected this pass

| Claim | Reality measured | Action |
|---|---|---|
| Neural codec "roughly on par with `gzip -9`" | **~21% larger** than gzip on the 143 KB corpus; only **~1 MB/s** throughput | ⚠️ README corrected to "does not yet match gzip"; speed caveat added |
| "Embeddings" (`embed_similarity`, RAG) | **FNV-1a token hashing** — lexical bag-of-words, not a learned/semantic model | ⚠️ documented as lexical; never exposed as a value |
| "Type-aware" compression win | hand-coded transform for **one synthetic dataset**; off-domain (source/text/random) order-0 rANS **loses 1.5–3.7× to gzip** | ⚠️ disclosed in README; the win is not type-system-driven |
| README line 30 "GBNF … not yet built" vs line 251 "[x] GBNF" | internal contradiction; GBNF **is** built | 🔧 line 30 fixed |
| Benchmark absolute ms | drift up to ~60% run-to-run on a thermally-variable laptop | 🔧 README reports **ratio ranges** across 5 runs, not single numbers |

### The structural gap (not a claim, but the key finding)

For an "AI-native" language the **data model is the weakest part**: no loops, no
arrays/lists/maps/sets/tuples/records, no `Bytes`, no first-class vector/embedding
type (only the opaque interpreter+WASM `Tensor`), and the compression engine is a
standalone Rust library reachable only via two `Str`-only interpreter builtins.
The AI primitives are real *kernels* wired to the interpreter, not a programmable
first-class data layer. Closing this is Tier 0 of [ROADMAP.md](ROADMAP.md).

### Honesty issues to fix in the repo itself

- `cargo test` does **not** pass out-of-the-box — needs `RUST_MIN_STACK=67108864`
  (a debug-mode deep-recursion test overflows the default stack; the CLI itself is
  fine because it runs on a large-stack thread).
- Native release binaries print an `aria_live=… aria_reuses=…` diagnostic to
  stderr — should be suppressed outside debug/`mem`.
