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
| **One canonical form** — no optional syntax, `;` terminates statements, single comment style (`--`) | Removes stylistic degrees of freedom that cause inconsistent generation |
| **Case = meaning** — `Uppercase` is always a type/constructor, `lowercase` always a value/function | Zero-ambiguity identifier resolution; friendly to grammar-constrained decoding |
| **Everything is an expression** — `if`, `match`, blocks all return values | Uniform reasoning; no statement/expression split to track |
| **Immutable `let` only** | Local reasoning — line N never depends on hidden mutation |
| **Algebraic data types + exhaustive `match`** | The highest-leverage feature for correct code generation |
| **Regular, LL(1)-ish grammar** | Can drive a GBNF grammar so a model *cannot* emit a syntax error |
| **Zero-dependency, fast compile** | Tight feedback loop; efficient compiling/bundling from day one |

## Quick start

Requires a Rust toolchain.

```sh
cargo build --release
./target/release/aria run examples/intro.aria
./target/release/aria run examples/list.aria
./target/release/aria ast examples/intro.aria   # dump the parsed AST
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
- [ ] Static type checker (Hindley–Milner-style inference)
- [ ] Exhaustiveness checking for `match`
- [ ] Effect / capability system
- [ ] WASM backend
- [ ] Native backend (Cranelift or LLVM)
- [ ] GBNF grammar export for constrained decoding
- [ ] Structured, machine-parseable compiler diagnostics

## License

MIT — see [LICENSE](LICENSE).
