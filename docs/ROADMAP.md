# Aria — Production-Readiness Roadmap

> What it would take to move Aria from *research prototype* to a language devs can
> actually adopt, experiment with, and deploy. Grounded in an audit of the current
> code (see [CLAIMS.md](CLAIMS.md)). Ordered by leverage.

## Where we are (one paragraph)

Aria has a real frontend (lexer/parser/typechecker with HM generics, exhaustive
`match`, a purity effect system), a typed ANF IR with zero-annotation Perceus
reference counting (garbage-free verified), and **three agreeing backends**
(interpreter, hand-emitted WASM, native-via-C). What it lacks is everything
*around* a language: a real data model, IO, packaging, tooling, and deployment
story. The compiler core is further along than the ecosystem — which is the
normal and fixable shape for a prototype.

---

## Tier 0 — The data model (do this first; nothing else matters without it)

For a self-described **AI-native** language, the absence of first-class
collections and binary/vector types is the single biggest blocker. Today the only
compound types are the opaque `Tensor` and hand-rolled linked-list ADTs.

1. **`Array[T]` / `Vector[T]`** — contiguous, O(1)-indexed, the foundation
   everything else builds on. Needs: literal syntax, an indexing operator, bounds
   semantics (checked, trapping like the existing overflow checks), and lowering
   in all three backends. The native backend already does `malloc`/`free` + RC for
   ADT cells, so a growable buffer is a natural extension.
2. **`Bytes` / binary buffers** — a first-class byte sequence with literals and
   slicing. This is the type the compression codecs *should* consume.
3. **`Map[K,V]` / `Set[T]`** — hashing infrastructure already exists internally
   (FNV-1a in `rag.rs`); expose it as a real container.
4. **Iteration that isn't only recursion** — either a `for x in xs` form that
   desugars to the existing tail-recursion (which the native backend already turns
   into a `goto` loop), or first-class iterators. Keep immutability; add ergonomics.
5. **A real `Vector`/`Embedding` value + wire in the shape checker.** `src/shape.rs`
   is a working compile-time tensor-shape type system that is *not connected* to
   the language. Promote `Tensor` from an opaque handle to a shaped, indexable
   value type and route `shape.rs` into `typeck.rs` so `matmul` dimension errors
   are caught at compile time in real `.aria` programs.
6. **Replace toy embeddings.** `embed_similarity` is FNV-1a token hashing. Either
   expose a `Vector` type so users bring their own model embeddings, or add a real
   embedding builtin. Same for the "neural" branding — it's a context-mixing
   predictor; either ship a true neural predictor or rename honestly.

**Why this tier is "AI-native":** vectors, byte buffers, tensors-as-values, and a
typed compression API are exactly the primitives LLM/embedding/RAG workloads need.
Today they exist only as interpreter-wired demos. Making them first-class,
typed, and backend-portable is what turns the tagline into a fact.

---

## Tier 1 — A usable language (IO, modules, errors)

- **IO beyond `print_*`.** File read/write, stdin/args, stdout/stderr as real
  typed effects (the purity effect system is the right hook — extend the single
  `IO` effect into a small capability set).
- **Modules & imports.** Today everything is one file. Need a module system,
  visibility, and a path/namespace story before any nontrivial program.
- **Error handling.** A `Result[T,E]` ADT plus `?`-style propagation (the type
  system can already express it; add sugar). Currently runtime errors (`division
  by zero`) just abort.
- **`let`-generalization.** Generics work, but let-bound values aren't generalized
  to polymorphic schemes — finish HM inference here.
- **Float/Int ergonomics.** Mixing is a type error with no conversion builtins;
  add `int_to_float`/`float_to_int` and friends.

---

## Tier 2 — Tooling devs expect (this is where "adoptable" is won)

This is the part that most determines whether anyone *tries* the language.

- **Language Server (LSP).** The single highest-ROI tool: diagnostics, hover
  types, go-to-def, completion. The typechecker already produces precise,
  function-scoped errors — wrap it in an LSP server. This powers **VS Code,
  Cursor, JetBrains, Neovim** simultaneously, since they all speak LSP.
- **Editor / Cursor extensions.**
  - *VS Code / Cursor extension*: TextMate grammar for syntax highlighting + an
    LSP client. Cursor is VS Code-based, so one extension covers both. Ship the
    `.gbnf` grammar so Cursor's model can be constrained to emit valid Aria — this
    is a genuinely novel selling point unique to Aria.
  - *Tree-sitter grammar*: powers Neovim/Helix highlighting and structural editing;
    relatively cheap given the LL(1)-ish grammar.
- **Formatter (`aria fmt`).** The language already aims at "one canonical form" —
  a formatter makes that real and kills diff noise. Low effort, high trust.
- **REPL (`aria repl`).** Interactive evaluation on top of the interpreter.
- **Better diagnostics.** Structured, machine-parseable JSON diagnostics (already
  on the roadmap) — doubles as the LSP payload and as feedback for LLM agents.
- **`aria test`.** A built-in test runner so libraries can ship tests.

---

## Tier 3 — Packaging, deployment & running real programs

**How you'd run/deploy Aria programs (target end state):**

- **Native binaries (works today, needs hardening).** `aria native f.aria out`
  already emits a standalone ~9 KB executable via `cc -O2`. To productionize:
  cross-compilation targets, static linking options, reproducible builds, and
  dropping the `aria_live=…` debug line from release output. This is the primary
  deployment path: ship a single dependency-free binary.
- **WASM (works today, needs a host story).** `aria wasm` emits a real `.wasm`
  module run under Node. To deploy: a documented host ABI (imports/exports),
  browser + WASI hosts, and a thin JS/host shim so Aria can target edge/serverless
  and the browser. Strong fit given the AI/web angle.
- **A package manager + registry (`aria add`).** Modules → packages →
  dependency resolution → a registry. Required before a real ecosystem.
- **Build manifest.** A project file (`aria.toml`) describing entrypoints,
  targets, and deps.
- **CI-friendly exit codes & logging.** Separate program output from the RC
  diagnostics; stable, documented exit semantics (note today `main`'s return
  value *is* the exit code — that needs a clearer convention for services).

---

## Tier 4 — FFI / "drivers" / talking to the outside world

"Drivers" in the systems sense (DB clients, HTTP, GPU, OS APIs) all reduce to
**one missing primitive: a foreign function interface.** The native backend's
path makes this tractable:

- **C FFI.** Aria already *transpiles to C and links with `cc`*. Add `extern`
  declarations so Aria can call C symbols and be called from C. This instantly
  unlocks the entire C ABI — sqlite, libcurl, OpenSSL, BLAS, CUDA/Metal via their
  C APIs, OS syscalls.
- **A standard library built on FFI.** Files, sockets, time, env, process — thin
  typed wrappers over libc/OS, gated by the capability/effect system.
- **Accelerator "drivers" for the AI angle.** The tensor core is pure-Rust scalar
  code today. Real workloads need BLAS/Accelerate/cuBLAS/Metal backends behind the
  same `matmul`/`softmax` interface — reachable once C FFI exists.
- **Host ABI for WASM.** The mirror of FFI on the WASM side: a documented import
  table so a JS/WASI host provides IO, fetch, and model calls.

---

## Tier 5 — Trust, correctness & community

- **`cargo test` green out-of-the-box** (today needs `RUST_MIN_STACK`; fix the
  deep-recursion debug test or bump the harness stack).
- **Keep the differential fuzzer and extend it** to every new backend/feature —
  it's the project's strongest correctness asset.
- **A spec.** Formalize the grammar (the GBNF export is a great start), the type
  system, and the canonical-form rules (trailing-comma optionality is still
  undecided).
- **Docs site + examples gallery + a "why Aria" page** centered on the one thing
  nothing else has: *constrained decoding so a model can't emit a syntax error.*
- **Honest benchmarks** (done — see the README Performance section and
  `benchmarks/`); keep them reproducible and caveated.

---

## Suggested sequencing

1. **Tier 0.1–0.2** (`Array`, `Bytes`) + **Tier 2 LSP/extension** in parallel —
   the data model makes it *useful*, the tooling makes it *tryable*. Ship the
   Cursor/VS Code extension with GBNF-constrained generation as the hook.
2. **Tier 1** (IO, modules, `Result`) — turns toys into programs.
3. **Tier 4 C FFI** — unlocks the stdlib and "drivers" without writing them all by hand.
4. **Tier 3 packaging** — once there's something worth packaging.
5. **Tier 0.5 + shape checker** + accelerator backends — deliver the AI-native
   promise end-to-end.

The compiler core is genuinely solid; the work ahead is breadth (data model, IO,
FFI, tooling), not a rewrite.
