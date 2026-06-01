# Aria

**An AI-native programming language — a research prototype.**

Aria is an experiment in designing a programming language *optimized for large
language model consumption*: a language whose syntax and semantics minimize the
kinds of mistakes models make, while staying efficient for computers to compile
and run.

> Status: early prototype. Frontend (lexer + parser) and a tree-walking
> interpreter work today. WASM / native code generation is the next milestone.

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
| **Regular, LL(1)-ish grammar** | *Designed* to drive a GBNF grammar for constrained decoding (so a model could not emit a syntax error). GBNF export is planned, not yet built. |
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
order-0 rANS codec** and roughly on par with `gzip -9`, lossless. It doesn't yet beat gzip
because the predictor lacks LZ-style long-match modeling — the next upgrade is a match model
/ higher-order contexts, then a true neural (transformer) predictor.

## Architecture

```
src/
  lexer.rs    hand-written tokenizer (comments, literals, operators)
  ast.rs      the abstract syntax tree
  parser.rs   recursive descent + Pratt precedence
  interp.rs   tree-walking interpreter (the "runs today" backend)
  main.rs     CLI: `aria run` / `aria ast`
```

The frontend is kept strictly separate from the backend so that a WASM or
native (LLVM/Cranelift) code generator can be added as an alternative backend
without touching the parser or AST.

## Roadmap

- [x] Lexer, parser, AST
- [x] Tree-walking interpreter (functions, recursion, ADTs, pattern matching, blocks)
- [x] Static type checker (bottom-up synthesis + checking against annotations)
- [x] Exhaustiveness checking for `match`
- [ ] Type inference for `let`-generalization / generics (polymorphism)
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
- [x] **WASM Phase 2b** — ADTs on linear memory with reference counting (`dup`/`drop`) lowered to wasm: a bump+free-list allocator, per-constructor recursive drop, **garbage-free verified in the compiled output** (`__live()==0`), differentially tested vs the interpreter. (Strings + in-place reuse in wasm are the next slice.)
- [ ] Effect / capability system
- [ ] Native backend (Cranelift or LLVM)
- [ ] GBNF grammar export for constrained decoding
- [ ] Structured, machine-parseable compiler diagnostics

## License

MIT — see [LICENSE](LICENSE).
