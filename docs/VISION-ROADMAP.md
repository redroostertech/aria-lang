# Aria — Vision Roadmap: Toward a "Living, Breathing" Language

> **North star (owner's words):** *"if this language can be used for AI and
> applications built on top of it to be able to performantly RETRAIN, and 'LEARN'
> and ADAPT — essentially a LIVING and BREATHING language — then my goals will be
> realized."*
>
> Read concretely, that is **two intertwined capabilities**:
> 1. **Programs that learn/adapt at runtime** — Aria apps that train, fine-tune,
>    update weights/embeddings, and adapt online, not just run a frozen model.
> 2. **A codebase that evolves** — an LLM writes/extends/repairs Aria *under
>    grammar + type + test constraint*, so the system co-evolves with its author.
>
> This document is a research-grade, honest roadmap grounded in a read of the
> current source (June 2026). It is deliberate about what is *straightforward
> engineering*, what is *hard but known*, and what is *open research*. Every claim
> traces to a file. **No code was modified to produce this.**

---

## A. Honest starting point — what exists today (verified against source)

The repository's own docs (`README.md`, `docs/CLAIMS.md`, `docs/GUIDE.md`) have
historically *understated* the data model and *overstated* the AI models. The
ground truth from the current source is:

### What Aria genuinely has toward the vision

- **A real, multi-type data model — far past what the docs claim.** `src/builtins.rs`
  now exposes first-class, generically-typed, backend-portable
  `Array[T]`, `Bytes`, `Map[K,V]`, `Set[T]`, **`Vector` (dense f64 embedding)**,
  and `Tensor` (f32 matrix), plus records and tuples (`examples/record.aria`,
  `examples/tuple.aria`) and static traits/interfaces (`src/traits.rs`,
  `examples/trait.aria`). The GUIDE's "no arrays/maps/vectors" line is stale.
- **Tensors AND vectors compile to native.** `src/c_backend.rs` contains a real C
  runtime: `AriaTensor` (refcounted `{rc, rows, cols, float data[]}`,
  lines ~2959+) and `AriaVector` (refcounted `{rc, len, cap, double elems[]}`,
  lines ~2789+), with `aria_tensor_matmul/transpose/softmax/relu` and
  `aria_vec_dot/cosine/norm/add/scale`. The GUIDE's capability matrix ("native:
  no tensors") is also stale.
- **Bit-exact floating point across backends.** `cc` is invoked with
  `-ffp-contract=off` (`src/main.rs` `cc_build`) and the C matmul uses the *same*
  `i-p-j` loop order and f32 accumulation as the interpreter (`c_backend.rs`
  ~3072) so results are bit-identical, not just close. This is a real asset for a
  numeric language — reproducible training is a property most ML stacks lack.
- **Tensor↔Vector bridge** (`tensor_row`, `tensor_from_rows`) lets the matmul
  world and the embedding world compose (`examples/bridge.aria`).
- **Embedding retrieval in pure Aria.** `src/prelude.rs` ships `nearest` /
  `nearest_score` / `similarities` over an `Array[Vector]` store, built only on
  `vec_cosine` + array primitives — the retrieval core of RAG, running on all
  three backends (`examples/rag.aria`).
- **Compile-time shape checking, now wired in.** `src/shape.rs::check_program` is
  called from `typeck::check` (`src/typeck.rs:417`). `matmul([m,k],[k',n])` with
  `k≠k'` is a *compile error* in real `.aria` programs (intraprocedural,
  no-false-positive). This is a genuine differentiator vs PyTorch's runtime shape
  crashes.
- **Garbage-free Perceus RC with FBIP reuse** (`src/rc.rs`): zero-annotation
  `dup`/`drop`, and **in-place reuse** that covers ADT cells *and* `Array`,
  `Bytes`, `Vector`, and `Tensor` (`relu` reuses when `rc==1`, `c_backend.rs`
  ~3108). 50% of allocations eliminated on the map benchmark; garbage-free
  verified (`aria_live==0`).
- **Four differentially-tested backends** that must agree bit-for-bit (interp, IR,
  WASM, native) under a type-directed fuzzer (`src/proptest.rs`, ~2761 lines).
- **A validated GBNF grammar** (`src/gbnf.rs`) covering the full surface incl.
  traits/records, tested to accept real examples and reject malformed input.
- **A single, inferred `IO` effect** by least-fixpoint over the call graph
  (`src/typeck.rs` ~385–417); `pure fn` is compiler-proven.

### What is conspicuously absent for "learning / adapting"

- **No automatic differentiation. No gradients. No `backward`/`grad`.** Nothing in
  the tree computes a derivative. This is *the* gap for Capability 1.
- **Scalar f32 kernels only.** `src/tensor.rs` and the C runtime are triple-nested
  scalar loops — **no BLAS, no SIMD, no Accelerate/Metal/CUDA.** Fine for tiny
  demos, orders of magnitude off real training.
- **No mutation, no `ref` cell, no in-place state beyond FBIP.** There is *no
  assignment syntax at all* (verified: `src/ast.rs`/`src/parser.rs` have no
  Assign node). Optimizer/weight state can only be threaded functionally or
  reused via FBIP when a binding is unique.
- **No real models.** The transformer (`src/transformer.rs`) runs on
  **untrained LCG-random weights** — a correct forward pass, not a useful model.
  `embed_similarity` is **FNV-1a token hashing** (`src/rag.rs`), not learned
  embeddings. The "neural" codec is a **PAQ-style integer context-mixing
  predictor** (`src/predict.rs`) — adaptive and statistical, but not a neural net.
- **No IO beyond `print_*`, no file/serialization.** You cannot load or save a
  weight tensor. The single `IO` effect has exactly four producers. Training is
  pointless if you can't persist the result.
- **No inference-stack integration.** `aria gbnf` emits a *string*. Nothing wires
  it into llama.cpp/vLLM for constrained decoding.
- **No structured diagnostics, no LSP.** Errors are human strings
  (`typeck::check` returns `Vec<String>`); there is no JSON span/code channel and
  no language server — the two things an *agent* feedback loop needs most.

**One-line summary:** Aria has an unusually strong *substrate* for numeric,
memory-safe, reproducible computation, and a real grammar — but it is missing the
*derivative*, the *speed*, the *persistence*, and the *agent feedback surface*
that the vision requires.

---

## B. Capability 1 — Programs that learn / adapt

This is the crux and the hardest part. Autodiff in a **pure, immutable,
monomorphized, AOT** functional language is a real design problem. Below are the
options, how each collides with Aria's specifics, and a recommendation.

### B.1 Automatic differentiation — the design space

The thing that makes "learn" possible is the gradient `∂loss/∂params`. Three
classical approaches, assessed against Aria's actual constraints (immutability,
HM types, monomorphization, the `Tensor`/`Vector` types):

**Option 1 — Reverse-mode via a runtime tape (Wengert list).**
The standard ML approach (PyTorch autograd). Tensor ops record a graph at runtime;
`backward()` walks it in reverse accumulating gradients.
- *Fit with Aria:* Mechanically the easiest because Aria already has refcounted
  `AriaTensor` handles; a "tracing tensor" would be a tagged variant that also
  records its parents and a local VJP closure. Closures already compile (function
  table in `c_backend.rs`).
- *Collision:* The tape is **mutable shared state** that Aria's effect/RC model
  doesn't express. You'd implement the tape *inside the runtime* (Rust/C), not in
  Aria — i.e. `grad` becomes a **builtin** like `matmul`, with the tape hidden in
  the runtime, exposed as a pure-looking Aria function `grad(f, x) -> Tensor`.
  This keeps Aria pure at the surface while the impurity lives in the runtime
  (exactly how `matmul` already hides a mutable `out` buffer).
- *Verdict:* **Hard but known.** Lowest risk to a *first* gradient. The cost is
  that differentiable code must be expressed through the tracing builtins, not
  arbitrary Aria control flow.

**Option 2 — Forward-mode via dual numbers.**
Carry `(value, derivative)` pairs; one pass per input direction.
- *Fit:* Beautiful fit for a pure language — a dual is just a 2-field record
  (Aria has records now), and the rules are local, no tape, no mutation. Could
  even be written *in Aria itself* over a `Dual` type once operator overloading /
  trait-based numerics exist (`src/traits.rs` is the hook).
- *Collision:* Forward-mode costs O(#inputs) passes. For training (millions of
  params, scalar loss) that's catastrophic — reverse-mode is O(1) in #inputs.
  Forward-mode is the *wrong complexity* for learning, though great for
  Jacobian-vector products and for *validating* a reverse-mode impl.
- *Verdict:* **Straightforward engineering, wrong asymptotics for training.** Build
  it early as a **correctness oracle** for Option 1/3, not as the training path.

**Option 3 — Source-to-source transformation of Aria functions.**
The research frontier (JAX/`grad`, Zygote.jl, Enzyme, Tapenade). Transform a
typed Aria function `f : Tensor -> Float` into `f' : Tensor -> (Float, Tensor)`
at compile time, emitting the adjoint program, which then monomorphizes and
compiles to C like any other Aria code.
- *Fit with Aria's strengths:* This is where Aria's architecture is genuinely
  *advantageous*. Immutability means **no aliasing analysis is needed** — the
  single hardest part of source-to-source AD (Enzyme spends enormous effort on it)
  largely evaporates when there is no mutation. The typed ANF IR (`src/ir.rs`) is
  *already* the normalized form AD transforms want (ANF/SSA-like, explicit
  let-bindings, no nested effects). Reverse-mode AD on pure ANF is a well-studied,
  *clean* transformation: each `let x = op(a,b)` gets a dual `let (da,db) +=
  vjp_op(x̄, a, b)` in the reversed pass.
- *Collisions / open problems:*
  - **Monomorphization ordering.** AD should run on the *typed* IR before or
    interleaved with monomorphization so the adjoint is itself monomorphized. The
    transform must be threaded into the `ir.rs → monomorphize.rs` pipeline.
  - **Control flow (the reverse pass needs the forward "trace").** `if`/`match`
    require remembering which branch ran; recursion (Aria's only loop) requires a
    stack of intermediate values for the reverse sweep. In a pure language this is
    the **checkpoint/trace** problem — known, but real work, and it interacts with
    Aria's tail-call-to-`goto` optimization (the `goto` loop discards the
    per-iteration values AD must retain).
  - **Higher-order functions / closures.** Differentiating through `array_map(f,
    xs)` means differentiating `f`. Doable (the adjoint of `map` is a `map` of
    adjoints) but requires AD to be closure-aware.
- *Verdict:* **The right long-term answer and Aria's strongest differentiator** —
  *because* immutability removes aliasing AD's worst pain. But it is **open-ish
  research** in this exact setting; do not start here.

**Recommendation.** **Tape-based reverse-mode as a runtime builtin first
(Option 1), validated by a dual-number oracle (Option 2), with source-to-source
(Option 3) as the flagship Phase-3 research bet.** Concretely:

```aria
-- Phase-1 surface (tape hidden in the runtime, like matmul's out-buffer):
fn loss(w: Tensor, x: Tensor, y: Tensor) -> Float = ...
let g = grad(loss, w, x, y);     -- builtin: returns ∂loss/∂w as a Tensor
```

`grad` is a builtin whose Rust/C implementation runs `loss` under a tracing
tensor and returns the accumulated gradient — pure at the Aria surface, impure in
the runtime, exactly mirroring how `matmul` already hides mutation. This gives a
*real gradient on the native backend* with the least new machinery, and the
dual-number `Dual` record (writable in Aria) cross-checks it the way the
differential fuzzer already cross-checks backends.

### B.2 Training loops & adaptation in an immutable language

There is **no assignment** in Aria, so an SGD step `w ← w − lr·g` must be
expressed as one of:

1. **Functional threading + FBIP reuse (recommended).** Write the step as
   `let w2 = vec_sub(w, vec_scale(g, lr))` and let Perceus reuse `w`'s buffer
   in place because it is unique (`rc==1`). This is *already* how `array_set`,
   `vec_add`, and `relu` behave — `src/rc.rs` + the `aria_vec_*`/`aria_tensor_*`
   "clone-if-shared, mutate-if-unique" pattern in `c_backend.rs`. **The training
   loop is just tail recursion**, which the native backend turns into a `goto`
   loop with no stack growth. So: *the immutable loop is already efficient when
   the weights are linearly threaded.* This is a quietly large asset.
   - *Caveat:* it relies on the optimizer state staying unique. The moment you
     also keep a momentum buffer, an Adam `(m, v)` pair, and the params, you must
     thread a tuple/record of state and trust reuse on each field. Worth a
     dedicated benchmark (RC churn per step is a real risk — see §F).
2. **A controlled `ref`/mutable cell (deliberate, scoped escape hatch).** Add a
   `Ref[T]` builtin (heap cell, `ref_new/ref_get/ref_set`) gated behind the
   effect system as a new `Mut` effect, so purity is still tracked. This is the
   pragmatic option for *online* services that update weights across requests
   (§B.2.3). It is a real language-design decision: keep it *opt-in and
   effect-tracked* so the "immutable by default, local reasoning" property
   survives.
3. **State-monad / explicit `step : State -> State`.** Most faithful to purity;
   most boilerplate. Best reserved for library code, not the user surface.

**Online adaptation (a running service that keeps learning).** This needs (a)
persistent mutable weight state across calls — the `Ref`/`Mut` path above — and
(b) IO to receive new data and to checkpoint. It is the most "living and
breathing" demo but depends on *both* the autodiff work and the IO/serialization
work below.

**Persisting / loading model state.** Today impossible (`IO` has 4 producers, all
`print_*`). Required additions:
- File IO as an effect (`read_bytes`/`write_bytes : ... -> Bytes` under `IO`).
  Aria already has a first-class `Bytes` buffer, so the serialization *target*
  exists.
- A tensor/vector (de)serializer to/from `Bytes` (`tensor_to_bytes` /
  `tensor_from_bytes`). Straightforward given the flat `data[]` layout.
This is **straightforward engineering** and is a prerequisite for *any* claim of
"retrain": a model you cannot save has not learned anything you can use.

### B.3 Performant numerics — from scalar loops to real speed

The interface is already right and stable: `matmul`/`softmax`/`relu`/`vec_*` are
builtins with fixed signatures, and the kernel bodies are isolated in the runtime
(`tensor.rs` for the oracle, `c_backend.rs` for native). **The kernel can be
swapped without touching a single `.aria` program or the type checker.** The path:

| Step | Mechanism | Effort | Dep | Risk to bit-exactness |
|---|---|---|---|---|
| SIMD the C matmul | `#pragma omp simd` / intrinsics / `-O3 -march=native` in `cc_build` | **M** | none (C backend exists) | Must keep `-ffp-contract=off` semantics; tiling can change accumulation order → would break the 4-backend bit-exact invariant. **Decide: relax bit-exactness for tensors, or keep a "reference" slow path.** |
| BLAS via FFI | `extern "C"` decls → link `-lAccelerate`/`-lopenblas`; route `matmul` to `cblas_sgemm` | **L** | **C FFI** (does not exist yet) | sgemm accumulation order ≠ the i-p-j oracle → bit divergence is expected; needs an explicit "fast vs reference" policy. |
| GPU (Metal/CUDA) | FFI to a kernel; async/streams; device memory mgmt | **XL** | C FFI + a memory/device model | Large; nondeterministic; furthest from Aria's current value props. |

**Honest take:** SIMD (M) is the high-ROI near-term win. BLAS (L) is the right
"real speed" target *and the forcing function for C FFI* — which the production
roadmap (`docs/ROADMAP.md` Tier 4) already wants for everything else. **The key
tension to resolve explicitly is bit-exactness vs speed**: Aria's headline
correctness property (4 backends agree bit-for-bit) is in direct conflict with
optimized GEMM. The clean resolution is a *typed/flagged* distinction: a
`matmul` that promises bit-exact reference semantics vs a `matmul_fast` that
promises only IEEE-correct-to-tolerance — and the differential fuzzer compares
the fast path within an epsilon (it *already* does this for `softmax`/`exp`,
which are documented as ~1e-5, not bit-exact — see `docs/CLAIMS.md`).

### B.4 The smallest real "a program that learns" demo

**Target:** train a **logistic-regression / 1-layer classifier** on a tiny toy
dataset (e.g. 2-D XOR-ish or linearly-separable points) by gradient descent,
end-to-end, on the **native backend**, and show the loss going down + the learned
weights classifying held-out points.

Minimum feature set to make *that* work (and nothing more):
1. `grad` builtin for a fixed, small op set: `matmul`, `+` (vec/tensor add),
   `relu`/`sigmoid`, and an MSE or cross-entropy loss. (Tape-based reverse-mode,
   §B.1 Option 1.) — **the one genuinely new piece.**
2. The training loop as tail recursion threading `(w, b)` with FBIP reuse (§B.2.1)
   — **already works** once `grad` exists.
3. `vec_scale`/`vec_sub` (have `scale`/`add`; need `sub` — trivial) for the SGD
   update.
4. A `sigmoid` builtin (trivial, mirrors `relu`).
5. Data as `Array[Vector]` + `Array[Float]` labels — **already supported.**
6. *Optional but recommended:* file IO to save the trained weights, so it's a
   "retrain" not just a "fit-in-RAM."

Everything except item 1 (and optionally 6) **exists today.** That is why `grad`
is the recommended first milestone (§E).

---

## C. Capability 2 — A codebase that evolves (the AI-native authoring loop)

This is **nearer-term, lower-risk, and the more defensible differentiator.** Aria
already has the two hardest prerequisites that most languages lack: a *canonical,
constrained-decodable grammar* and a *fast, precise type checker*. What's missing
is the plumbing that turns those into an agent feedback loop.

### C.1 GBNF → real constrained decoding (the missing wire)

Today `aria gbnf` (`src/gbnf.rs`) emits a real, validated GBNF string. The grammar
is *already in the format llama.cpp consumes.* What does **not** exist is the
integration. Needed:
- A thin harness that loads `grammar()` into **llama.cpp** (`--grammar-file`) or
  **vLLM** (via its grammar/Outlines backend) so a local model is *forced* to emit
  token sequences the Aria parser accepts.
- A demo: prompt a small local model to "write an Aria function that …", decode
  **under the grammar**, and show that the output **parses 100% of the time** —
  the property no mainstream language can offer. (Effort **S–M**; it's wiring +
  a script, the grammar is done.)
- *Honest caveat:* a grammar guarantees *syntactic* validity, not *type*
  correctness — that's exactly why §C.2/§C.4 matter. The grammar removes parse
  errors; the type checker removes the next layer.

### C.2 Structured diagnostics (the feedback channel)

`typeck::check` returns `Vec<String>` — human-readable, function-scoped, batched,
but **not machine-parseable**. For an agent loop, add a parallel structured
channel:
- A `Diagnostic { code, severity, message, span:{line,col,len}, function }`
  emitted as **JSON** (`aria check --json`). The information *already exists* in
  the checker (function names; lex/parse errors already carry line numbers per
  `src/main.rs`); spans need to be threaded through the AST (the lexer has
  positions; the AST currently discards most of them — *this is the main work*).
- Stable **error codes** (e.g. `E-NONEXHAUSTIVE`, `E-SHAPE-MISMATCH`,
  `E-ARITY`) so an agent can pattern-match and a fine-tune can learn repair
  policies per code. The shape checker (`src/shape.rs`) already produces
  distinctive messages worth coding.
- Effort **M** (span threading is the cost); **this is the single
  highest-leverage authoring-loop investment** because it is the agent's eyes.

### C.3 LSP + editor/Cursor extension

- **LSP server** wrapping `typeck::check`: diagnostics (reuse §C.2 JSON), hover
  types (the checker infers them), go-to-def, completion. The checker is *fast*
  (`aria check` ~instant), so a responsive server is realistic. Effort **L**
  (LSP boilerplate + the span work from §C.2, which it shares).
- **VS Code / Cursor extension**: a TextMate grammar (cheap given case=meaning) +
  an LSP client, and crucially **ship the `.gbnf`** so Cursor's model can be
  constrained. This is the concrete, shippable form of "a model can't emit a
  syntax error." Effort **M** atop the LSP.
- **Tree-sitter grammar** for Neovim/Helix — cheap given the LL(1)-ish design.

### C.4 The agent loop (the actual differentiator)

The thing that makes this more than "just a grammar file" is a **closed loop**:

```
  LLM (decoding UNDER the GBNF grammar)  →  emits syntactically-valid Aria
     →  `aria check --json`              →  structured type/shape diagnostics
     →  agent reads codes+spans, repairs (re-decode the offending span)
     →  `aria native-run` / `aria run`   →  captures output / exit code
     →  agent compares to the goal/test  →  repair or accept
```

To make this real and demonstrable, build:
1. A `aria check --json` mode (§C.2) and a `--json` run mode that reports
   structured runtime outcome (ok / trap-with-reason / value).
2. A small **driver** (Rust or Python) that runs the loop against a local model
   with grammar-constrained decoding (§C.1) and structured feedback (§C.2).
3. A **task suite + scorer**: "implement `fn f(...) -> ...` so that these
   assertions pass," graded by compile-success → type-success → test-pass. This
   *measures* the loop and produces training data for self-improvement.

**The differentiator vs other languages:** the loop's first two stages are *near
hard guarantees* in Aria — the model **cannot** emit a parse error (grammar), and
the type checker rejects whole classes of semantic errors **including tensor shape
mismatches** (`shape.rs`) before any run. Most agent-coding loops burn turns on
syntax/shape errors a model can't see coming; Aria makes those *structurally
impossible or statically caught*. That is the honest, defensible claim.

---

## D. Synthesis — the "living, breathing" system

The two capabilities compose into the north star:

> **An agent (Capability 2) writes an Aria program that itself trains/adapts a
> model (Capability 1), runs it, reads structured results, and rewrites itself.**

Concretely, the converged system is a loop where:
- An LLM, decoding under Aria's grammar and steered by structured diagnostics,
  emits/edits an Aria training pipeline (a `loss`, a model, a `grad`-based SGD
  loop).
- That pipeline **runs natively**, trains on data, and **persists weights**
  (`Bytes` + file IO), reporting metrics through the structured channel.
- The agent observes loss/accuracy and **rewrites the program** (architecture,
  hyperparameters, the loss) — the *code* adapting in response to what the
  *program* learned. Program-level learning and code-level evolution share one
  feedback substrate.

**What is genuinely novel here:**
- A language where **shape errors and syntax errors are caught/forbidden before
  the agent ever runs the code** — a structurally tighter authoring loop than
  PyTorch+Python, where both are runtime surprises.
- **Bit-exact, garbage-free, native** numeric execution — reproducible training
  runs and pause-free online adaptation, properties the Python/PyTorch stack does
  not offer.
- **Immutability-as-an-AD-asset**: the property that makes Aria pleasant for LLMs
  (no hidden mutation) is the *same* property that makes source-to-source autodiff
  tractable (no aliasing). That alignment is the real intellectual core of the
  project.

**What is hype to avoid:**
- "Self-improving AI that rewrites itself" — only honest once the agent loop
  measurably raises a *test-pass rate* on a held-out suite (§C.4). Until then it's
  a scaffold, not a result.
- "Trains models" — only honest once a real (even tiny) model's loss provably
  decreases on the native backend (§B.4). The current transformer is *untrained*;
  don't conflate a forward pass with learning.
- "Real embeddings / neural compression" — `embed_similarity` is FNV-1a hashing;
  the codec is statistical context-mixing, not a neural net. Keep the existing
  honesty (`docs/CLAIMS.md` already does this well).

---

## E. Phased plan

Each phase yields a **demonstrable artifact**. Effort: S(days) M(1–2wk)
L(weeks) XL(month+). "DoD" = definition-of-done demo.

### Phase 0 — Make the authoring loop real (lowest risk, highest leverage)
*Theme: Capability 2. Build the agent's eyes and hands before anything else.*

| Milestone | Requires (vs existing) | Effort | DoD demo |
|---|---|---|---|
| **0a. Structured diagnostics** | thread spans through AST/lexer (positions exist); JSON + error codes over the existing `Vec<String>` checker | **M** | `aria check --json broken.aria` emits coded spans incl. `E-SHAPE-MISMATCH` from `shape.rs` |
| **0b. GBNF → llama.cpp constrained decode** | wire existing `gbnf::grammar()` into llama.cpp; a script | **S–M** | A local model emits 50/50 Aria snippets that **all parse** (vs an unconstrained baseline that doesn't) |
| **0c. Agent loop driver + task suite** | 0a+0b; `--json` run outcome; a scorer | **M** | Loop solves N "make these asserts pass" tasks; report compile→type→test pass-rate |
| **0d. LSP + Cursor extension** | 0a spans; LSP boilerplate; ship `.gbnf` | **L** | Hover types + live diagnostics in Cursor; grammar shipped |

**RECOMMENDED FIRST MILESTONE: 0a (structured diagnostics).** Justification:
lowest risk (no new semantics, the information already exists), unblocks
*everything* downstream (the agent loop, the LSP, and later the training-metrics
channel all consume it), and immediately demonstrable. It converts Aria's existing
strong checker — including the already-wired shape checker — into a machine
feedback surface, which is the precondition for "a codebase that evolves."

### Phase 1 — The first program that *learns*
*Theme: Capability 1, smallest real demo (§B.4).*

| Milestone | Requires | Effort | DoD demo |
|---|---|---|---|
| **1a. `grad` builtin (tape reverse-mode)** | new runtime tape over `AriaTensor`/`AriaVector`; VJPs for matmul/add/relu/sigmoid/loss | **L** | `grad(loss, w)` returns correct gradient, cross-checked vs a dual-number `Dual` oracle |
| **1b. SGD on toy data** | 1a; `vec_sub`/`sigmoid` (trivial); FBIP-reused loop (exists) | **S** (after 1a) | Logistic regression trains on native backend; **loss decreases**, held-out points classified |
| **1c. File IO + tensor (de)serialize** | new `IO` effect producers; `Bytes`↔tensor (Bytes exists) | **M** | Train, **save** weights, reload, resume — a real "retrain" |

**DoD for the phase:** `aria native-run train.aria` prints a decreasing loss curve
and a saved-then-reloaded model that classifies held-out data. *This is the first
honest "a program that learns" claim.*

### Phase 2 — Performance + online adaptation (make learning *performant*)
*Theme: the "performantly retrain" in the owner's words.*

| Milestone | Requires | Effort | DoD demo |
|---|---|---|---|
| **2a. SIMD/`-O3` tensor kernels** | edit C runtime + `cc_build`; epsilon-compare in fuzzer | **M** | matmul Nx faster; fuzzer green within tolerance |
| **2b. C FFI (`extern`)** | new `extern` decl + linker plumbing (native backend transpiles to C already) | **L** | Aria calls a C function and links it |
| **2c. BLAS-backed `matmul_fast`** | 2b; `cblas_sgemm` route; fast-vs-reference policy | **L** | Large matmul at BLAS speed; reference path still bit-exact |
| **2d. `Ref[T]`/`Mut` effect + online updater** | effect-system extension (single `IO` effect today is the hook); 1a/1c | **L** | A long-running Aria process updates weights across inputs (online adaptation) |

### Phase 3 — Source-to-source AD + self-evolving pipelines (the research bet)
*Theme: the flagship differentiator and the full north star.*

| Milestone | Requires | Effort | DoD demo |
|---|---|---|---|
| **3a. Source-to-source reverse AD on the ANF IR** | AD pass in `ir.rs`→`monomorphize.rs`; checkpointing for `match`/recursion; closure-aware | **XL** (research) | `grad` of *arbitrary* pure Aria functions (not just builtins), validated vs 1a's tape and the dual oracle |
| **3b. Differentiable transformer, then trained** | 3a + Phase-2 speed; real init + data | **XL** | A *small but trained* model whose loss provably drops (retires the "untrained weights" caveat) |
| **3c. Self-evolving pipeline** | Phase-0 loop + 1/2/3 | **XL** | Agent writes a training pipeline, reads metrics via the structured channel, and rewrites it to improve a measured score |

**Sequencing rationale.** Phase 0 makes the system *observable and steerable*
(cheap, unblocks all). Phase 1 proves *learning* with the least new math (tape
`grad`). Phase 2 makes it *fast* and *online* (the owner's "performantly"). Phase 3
is the research payoff that only makes sense once 0–2 exist. Each phase ends in a
runnable artifact.

---

## F. Risk / feasibility register (honest)

| Risk | Class | Why it threatens the vision | Mitigation / where Aria has an edge |
|---|---|---|---|
| **Source-to-source AD in a monomorphized AOT functional lang** | **Open research** | Checkpointing across `match`/recursion, closure-aware adjoints, AD×monomorphization ordering are unsolved *in this exact setting* | Start with tape `grad` (known). Aria's **immutability removes aliasing analysis** — AD's worst pain (Enzyme/Zygote) — so the research is *easier here than elsewhere*. The **ANF IR is already AD-shaped.** |
| **RC churn per training step** | Hard but known | Each SGD step allocs/frees; `listsum` already shows RC loses to a nursery GC on alloc-heavy churn (`README` Performance) | FBIP **reuses** unique weight/optimizer buffers in place (no alloc) — *already implemented* for Array/Bytes/Vector/Tensor. Needs a per-step benchmark to confirm reuse holds with multi-field optimizer state. |
| **Scalar kernels → real speed** | Hard but known | Triple-loop matmul is orders of magnitude off BLAS; "performantly retrain" fails without it | Kernel is **isolated behind a stable builtin interface**; swap to SIMD→BLAS→GPU without touching `.aria` code or types. SIMD is **M**, BLAS gated on C FFI (already a roadmap item). |
| **Bit-exactness vs optimized GEMM** | Design tension | Aria's headline "4 backends agree bit-for-bit" conflicts with tiled/BLAS accumulation order | Adopt the existing `softmax`/`exp` precedent (**epsilon-compared, documented** in `CLAIMS.md`): a bit-exact *reference* path + a tolerance-checked *fast* path. |
| **The "real models" gap** | Engineering + data | Transformer is untrained; embeddings are FNV-1a; nothing has *learned* | Don't claim learning until Phase-1 loss provably drops. The substrate (bit-exact native tensors, retrieval, shape checking) is real and ready to host a real model. |
| **Persistence/IO absent** | Straightforward eng | "Retrain" is meaningless if weights can't be saved/loaded | `Bytes` exists as the serialization target; add file-IO effect producers (Phase 1c). Low risk. |
| **Agent loop = scaffold, not result** | Evaluation risk | Easy to *demo*, hard to *prove* it improves anything | Tie every claim to a **test-pass-rate on a held-out suite** (§C.4). Grammar+shape-checking give a genuinely tighter loop than PyTorch+Python — but only the number proves it. |
| **GPU** | XL / strategic | Far from current value props (tiny native binaries, determinism, RC) | Defer past Phase 2. CPU BLAS may be "performant enough" for the adaptive/online use cases the vision actually targets; revisit GPU only if scale demands it. |

### Where Aria genuinely beats "just reach for PyTorch"
1. **Shape errors are compile-time** (`shape.rs` wired into `typeck`), not 3am
   runtime crashes — and an LLM author *cannot* emit a tensor-shape bug that
   compiles.
2. **Syntax errors are structurally impossible** under GBNF-constrained decoding —
   no Python stack can promise that.
3. **Bit-exact, deterministic, garbage-free, pause-free** native execution —
   reproducible training and clean online adaptation.
4. **Immutability is simultaneously the LLM-friendliness property and the
   autodiff-tractability property** — a rare alignment that is the project's real
   thesis.

These are not reasons Aria is *better than* PyTorch today (it is far less
capable). They are reasons the *path* is worth walking: Aria can offer a
**verifiably-correct, agent-authorable, reproducible** learning substrate that the
incumbent stack structurally cannot — *if* the autodiff, speed, and persistence
gaps are closed in the order above.

---

### TL;DR
- **Strongest assets today:** first-class native tensors/vectors with bit-exact
  fp, compile-time shape checking wired into typeck, garbage-free RC + FBIP reuse,
  4 agreeing backends, a validated GBNF grammar, and pure immutability.
- **The one thing blocking "learn":** no autodiff. Recommend **tape-based reverse
  `grad` as a builtin first**, dual numbers as oracle, source-to-source AD as the
  Phase-3 research flagship (where immutability is a decisive advantage).
- **The one thing blocking "a codebase that evolves":** no structured feedback.
  Recommend **structured JSON diagnostics (Milestone 0a) as the first thing
  built** — lowest risk, unblocks the agent loop, the LSP, and training metrics.
- **Smallest honest "it learns" demo:** logistic regression trained by `grad`-SGD
  on the native backend, weights saved/reloaded — almost entirely buildable on
  what exists *once `grad` lands.*
