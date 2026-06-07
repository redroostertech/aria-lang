# Aria — Structured Diagnostics (`aria check --json`)

> The machine-readable feedback channel an LLM authoring loop (and a future LSP)
> reads to self-correct. This document is the **contract**: the JSON schema and
> the stable code table consumers may depend on.

## Invocation

```sh
aria check --json <file.aria>
```

- Emits a JSON **array** of diagnostic objects to **stdout** — one object per
  error **or warning**.
- Exits **non-zero (1)** if there are any **errors**, **0** otherwise.
  **Warnings do NOT affect the exit code**: a program with only warnings still
  exits 0 and is "clean" for compilation purposes.
- A clean program prints exactly `[]` and exits 0. A program that type-checks but
  has lint warnings (e.g. an unused `let`) prints an array of `"warning"` objects
  and **still exits 0**.
- The `--json` flag is accepted in any position (e.g.
  `aria check --json f.aria` or `aria check f.aria --json`) and is only valid
  for the `check` command.
- The human-readable `aria check <file.aria>` (no flag) is unchanged.

The output is compact (one line). Strings are JSON-escaped (quotes, backslashes,
newlines, and other control characters), so the array is always valid JSON —
pipe it through `python3 -m json.tool` to confirm.

## Schema

The top level is **just the array** (kept simple and stable). Each element is an
object:

```json
{
  "severity": "error",
  "phase": "exhaustiveness",
  "code": "E0203",
  "message": "function `code`: non-exhaustive match on Color: missing case `Blue`",
  "line": null,
  "col": null,
  "function": "code"
}
```

| field      | type            | meaning                                                                 |
|------------|-----------------|-------------------------------------------------------------------------|
| `severity` | string          | `"error"` (a hard compile error) or `"warning"` (an advisory lint that does **not** fail compilation / change the exit code). |
| `phase`    | string          | Compiler phase: `lex`, `parse`, `type`, `shape`, `purity`, `exhaustiveness`, `io` (file read), or `lint` (a warning). |
| `code`     | string          | **Stable** short code per error *category* (see table). Match on this.  |
| `message`  | string          | Human-readable text (identical to the non-`--json` path).               |
| `line`     | int or `null`   | 1-based source line if known, else `null`.                              |
| `col`      | int or `null`   | 1-based column if known, else `null`. **Populated for expression-level type/shape errors** from the precise span of the offending sub-expression. |
| `end_line` | int or `null`   | 1-based END line of the offending expression's span (precise spans only), else `null`. |
| `end_col`  | int or `null`   | 1-based END column (one past the last character) of the span, else `null`. With `line`/`col` this gives an exact range. |
| `function` | string or `null`| Enclosing function name if known, else `null`.                          |

### Forward compatibility (consumer rules)

The schema is designed so precise spans can be added later without breaking
consumers. A consumer MUST:

- **Ignore unknown object fields** (new fields may be added).
- **Tolerate `null`** for `line`, `col`, `end_line`, `end_col`, and `function`.
- **Key off `code`** (and optionally `phase`) for programmatic handling, not off
  the `message` text (messages may be reworded).

## Code table

Codes are grouped by category. The code — not the classification heuristic — is
the stable contract.

| code    | phase            | category                                                              |
|---------|------------------|-----------------------------------------------------------------------|
| `E0001` | `lex`            | Lexing error (bad character, malformed number, bad string/escape).    |
| `E0002` | `io`             | File I/O error (the source file cannot be read / does not exist).     |
| `E0100` | `parse`          | Parse error (unexpected token, missing delimiter, bad pattern).       |
| `E0200` | `type`           | Unknown / undefined name (unbound variable, unknown function/constructor/record). |
| `E0201` | `type`           | Type mismatch (return/body, comparison, application, branch, `let`…). |
| `E0202` | `type`           | Constructor / record fields: wrong arity (arguments, method params, type args) **and** named-field shape (missing / duplicate / unknown field). |
| `E0203` | `exhaustiveness` | Non-exhaustive `match` (a constructor / nested case is unhandled).     |
| `E0204` | `type`           | Unknown type or type parameter.                                       |
| `E0205` | `type`           | Unused / un-inferable (phantom) type parameter.                       |
| `E0206` | `type`           | Trait / interface / impl: bound resolution failure (method not callable, missing bound) **and** interface/impl method-arity mismatch (reported as `type`/E0206 even though it is raised during parser-time trait lowering). |
| `E0207` | `type`           | Duplicate or illegal redefinition (type, ctor, fn, builtin).          |
| `E0210` | `purity`         | Purity violation (a `pure` fn performs / may perform IO).             |
| `E0300` | `shape`          | Tensor shape mismatch (matmul/transpose/add dimension error).         |
| `E0900` | (any)            | Uncategorized (reserved fallback; should not occur in practice).      |

New categories will get new `E####` codes; existing codes keep their meaning.

### Warning codes (`W####`)

Warnings are **advisory lints** (severity `"warning"`, phase `lint`) surfaced on
otherwise **well-formed** programs. They are emitted from the data-flow analysis
(see `docs/ANALYSIS.md`). A warning carries a **precise span** (`line`/`col`/
`end_line`/`end_col`) on the flagged construct and the enclosing `function`.

| code    | phase  | severity  | category                                                        |
|---------|--------|-----------|-----------------------------------------------------------------|
| `W0001` | `lint` | `warning` | **Unused variable**: a `let` binding that is never read. Because Aria is single-assignment, an unused `let` is provably dead. |
| `W0002` | `lint` | `warning` | **Unused parameter**. *Reserved / opt-in — NOT emitted by default*, because a parameter is frequently unused for a legitimate reason (a signature / interface / trait method requires it), which would make it false-positive noise. |

By default only `W0001` (unused `let`) is emitted. Unused **parameters**,
**lambda parameters**, and **match binders** are deliberately *not* warned about
(pattern destructuring and callback shapes routinely leave some unused), to keep
the lint low-noise for an AI authoring loop.

### Where warnings are surfaced

- **`aria check --json`**: warning objects are appended to the diagnostics array
  (only when the program is otherwise error-free). They **do not change the exit
  code** — a program with only warnings exits 0.
- **`aria lsp`**: published as LSP diagnostics with **`severity: 2` (Warning)** and
  the precise range, so editors underline dead variables and an agent loop sees
  them.
- **Human `aria check` (no `--json`)**: **unchanged** — warnings are NOT printed
  on the human path. They are a machine/editor channel only. (`aria analyze` also
  reports unused bindings in its `data-flow:` summary and `dataflow` JSON.)

## Location precision (what is populated today)

Every AST **expression**, **statement**, and **pattern** now carries a precise
source **span** (1-based start/end line+column), threaded from the lexer (which
tracks line **and** column per token) through the parser. The type/shape checker
records the span of the **innermost offending node** on each error, so
diagnostics point at the EXACT operand / call site / `let` statement / `match`
pattern, not just the function.

- **`lex` / `parse` errors:** `line` IS populated (these messages carry a
  `line N:` prefix). `col`/`end_line`/`end_col` are `null` — a lex/parse failure
  is reported at a token, not a fully-parsed expression span.
- **`type` / `shape` errors at the expression level:** `line` **and** `col`
  (and `end_line`/`end_col`) are populated from the offending sub-expression's
  span — e.g. `fn f() -> Int = 1 + true` reports the precise location of
  `1 + true`, and `1 + (2 * true)` reports the inner `2 * true`. A whole-body
  return-type mismatch is located at the function's result expression.
- **`let`-statement and `match`-pattern errors:** a `let x: Int = true;`
  annotation mismatch is located at the **whole `let` statement** span (the `let`
  keyword through the terminating `;`), and a `match`-arm pattern of the wrong
  type (`Box(v)` against an `Int` scrutinee) is located at the **pattern node's**
  span — e.g. `Box(v)` exactly, recorded innermost-first so a nested sub-pattern
  pins the deepest failing node.
- **`function`** is populated **for function-scoped errors** (extracted from the
  message, which names the enclosing function). It is `null` for
  declaration-level errors not inside any function body — e.g.
  duplicate-declaration / redefinition errors (`duplicate type`,
  `duplicate function`, `cannot redefine built-in`) and `io` file-read errors.
- **Declaration-level and unlocatable errors** (duplicate declarations, unknown
  types in signatures, purity/exhaustiveness messages the checker cannot tie to
  a single expression) leave `col`/`end_*` `null` — better unset than wrong.

Consumers must always tolerate `null` for any location field (per the
forward-compatibility rules).

## Examples

Three known type errors (`examples/broken.aria`):

```sh
$ aria check --json examples/broken.aria
[{"severity":"error","phase":"exhaustiveness","code":"E0203","message":"function `code`: non-exhaustive match on Color: missing case `Blue`","line":12,"col":3,"end_line":15,"end_col":4,"function":"code"},{"severity":"error","phase":"type","code":"E0201","message":"function `wrong_return`: body has type Bool but return type is Int (expected Bool, found Int)","line":18,"col":28,"end_line":18,"end_col":32,"function":"wrong_return"},{"severity":"error","phase":"type","code":"E0201","message":"function `bad_compare`: cannot compare Int and String","line":21,"col":34,"end_line":21,"end_col":45,"function":"bad_compare"}]
$ echo $?
1
```

Each error now carries a precise `line`+`col` (and `end_line`/`end_col`)
pointing at the offending sub-expression. A type error mid-expression locates
the exact operand:

```sh
$ aria check --json -   # fn f() -> Int = 1 + true
[{"severity":"error","phase":"type","code":"E0201","message":"function `f`: `Add` needs two Ints or two Floats, got Int and Bool","line":1,"col":17,"end_line":1,"end_col":25,"function":"f"}]
```

A clean program:

```sh
$ aria check --json clean.aria
[]
$ echo $?
0
```

A parse error (note populated `line`):

```sh
$ aria check --json parse_err.aria
[{"severity":"error","phase":"parse","code":"E0100","message":"line 2: unexpected token Let in expression","line":2,"col":null,"end_line":null,"end_col":null,"function":null}]
```

A program that type-checks but has an **unused `let`** — a `W0001` warning with a
precise span, and the exit code is still **0** (a warning is not a compile
failure):

```sh
$ aria check --json unused.aria   # fn f() -> Int = { let tmp = 99; 1 }
[{"severity":"warning","phase":"lint","code":"W0001","message":"unused variable `tmp`","line":1,"col":23,"end_line":1,"end_col":26,"function":"f"}]
$ echo $?
0
```
