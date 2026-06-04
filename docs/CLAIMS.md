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
| **Regular grammar → GBNF "model can't emit a syntax error"** | GBNF export is not implemented | ⚠️ Reworded as design intent (planned), not a current fact |

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
| **Memory model (Perceus-RC) "decided"** | researched + documented in `docs/MEMORY.md`, but **not implemented**; whole-program ownership inference is an explicit unproven bet | ⚠️ Honest in MEMORY.md; remains the project's biggest open risk |
| **"compiled" language** | runs on a tree-walking interpreter; no compiled backend yet | ✅ README does not claim "compiled" — it says compiler is the next milestone |
| **`aria mem` / memory POC** | lowers the Int/Bool/ADT subset to ANF IR, inserts precise Perceus-style `dup`/`drop` **+ reuse analysis**, runs it (cross-checked against the tree-walker), and reports fresh allocations / reuses / frees / peak-live | ✅ Garbage-free (no cell live at exit, zero annotations) verified across list-sum, map-then-sum, shared refs, unused values, branch-only-use, scrutinee-used-after-match, heap-field-after-borrow, and trees. **Reuse eliminates 50% of allocations** on the map benchmark (unique cells mutated in place). Scope: the functional subset + an IR interpreter — *not* the whole language or a native backend yet. |

## Compiled backend (WASM, Phase 2a)

| Claim | Reality | Verdict |
|---|---|---|
| **Aria compiles to WebAssembly** | `src/wasm.rs` hand-emits a real `.wasm` for the pure Int/Bool/function subset (arithmetic, comparisons, `if`, integer `match`, recursion, short-circuit `&&`/`\|\|`); runs via Node | ✅ Differentially tested vs the interpreter oracle; unsupported features (ADTs/strings/floats/builtins) rejected with a clean error, not a panic. Heap data is Phase 2b. |
| **Compiled and interpreted Aria agree** | verified across a curated battery + adversarial edge cases (LEB128 limits, multi-byte section lengths, many functions/locals, negatives, large i64) | ✅ Integer overflow is a *defined error*: wasm **traps** on `+`/`-`/`*`/negation overflow and div/rem-by-zero (and `MIN/-1`), matching the interpreter's checked-error semantics. No silent-wrap divergence. |

### Backend agreement caveats (honest)

- **Recursion depth (wasm):** compiled wasm recursion is bounded by the host runtime's stack. Under Node it traps on ~20k-deep non-tail recursion; the `wasm-run` harness passes `--stack-size=8000` to raise this so it matches the interpreter (1 GiB thread) and native (C stack) on the cases tested (50k+). This is a *runtime stack limit*, not a miscompilation — a different wasm runtime would differ.
- **`softmax` / `embed_similarity`:** agree with the interpreter within ~1e-5, not bit-exact, because `exp` (Rust `f32::exp` vs JS `Math.exp`) is not bit-portable. All other tensor ops (`matmul`/`transpose`/`relu`/`get`) bit-match.
- **Codec builtins** (`compressed_size`, `neural_bits_per_byte`) remain interpreter-only in the compiled backends (porting the whole coders is deferred; they return a clean `Err`).

## Net

The core *language-property* claims (immutability, expressions, case rule,
exhaustiveness, zero-deps, static typing, generics, callable AI primitives) and
the *compression size* claims are all supported by code and reproducible. The
items that outran reality — optional leading pipe, the pinned speed multiple,
GBNF, shape-checking-as-a-language-feature, and "the language knows the shape" —
were either fixed or reworded to match what the code actually does.
