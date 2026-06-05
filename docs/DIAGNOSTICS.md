# Aria — Structured Diagnostics (`aria check --json`)

> The machine-readable feedback channel an LLM authoring loop (and a future LSP)
> reads to self-correct. This document is the **contract**: the JSON schema and
> the stable code table consumers may depend on.

## Invocation

```sh
aria check --json <file.aria>
```

- Emits a JSON **array** of diagnostic objects to **stdout** — one object per
  error.
- Exits **non-zero (1)** if there are any diagnostics, **0** if the program is
  clean.
- A clean program prints exactly `[]` and exits 0.
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
| `severity` | string          | `"error"` today. Room for `"warning"` later.                            |
| `phase`    | string          | Compiler phase: `lex`, `parse`, `type`, `shape`, `purity`, `exhaustiveness`. |
| `code`     | string          | **Stable** short code per error *category* (see table). Match on this.  |
| `message`  | string          | Human-readable text (identical to the non-`--json` path).               |
| `line`     | int or `null`   | 1-based source line if known, else `null`.                              |
| `col`      | int or `null`   | 1-based column if known, else `null` (currently always `null`).         |
| `function` | string or `null`| Enclosing function name if known, else `null`.                          |

### Forward compatibility (consumer rules)

The schema is designed so precise spans can be added later without breaking
consumers. A consumer MUST:

- **Ignore unknown object fields** (new fields may be added).
- **Tolerate `null`** for `line`, `col`, and `function`.
- **Key off `code`** (and optionally `phase`) for programmatic handling, not off
  the `message` text (messages may be reworded).

## Code table

Codes are grouped by category. The code — not the classification heuristic — is
the stable contract.

| code    | phase            | category                                                              |
|---------|------------------|-----------------------------------------------------------------------|
| `E0001` | `lex`            | Lexing error (bad character, malformed number, bad string/escape).    |
| `E0100` | `parse`          | Parse error (unexpected token, missing delimiter, bad pattern).       |
| `E0200` | `type`           | Unknown / undefined name (unbound variable, unknown function/constructor/record). |
| `E0201` | `type`           | Type mismatch (return/body, comparison, application, branch, `let`…). |
| `E0202` | `type`           | Wrong arity (arguments, constructor/record fields, method params, type args). |
| `E0203` | `exhaustiveness` | Non-exhaustive `match` (a constructor / nested case is unhandled).     |
| `E0204` | `type`           | Unknown type or type parameter.                                       |
| `E0205` | `type`           | Unused / un-inferable (phantom) type parameter.                       |
| `E0206` | `type`           | Trait / bound resolution failure (method not callable, missing bound).|
| `E0207` | `type`           | Duplicate or illegal redefinition (type, ctor, fn, builtin).          |
| `E0210` | `purity`         | Purity violation (a `pure` fn performs / may perform IO).             |
| `E0300` | `shape`          | Tensor shape mismatch (matmul/transpose/add dimension error).         |
| `E0900` | (any)            | Uncategorized (reserved fallback; should not occur in practice).      |

New categories will get new `E####` codes; existing codes keep their meaning.

## Location precision (what is populated today)

- **`lex` / `parse` errors:** `line` IS populated (the lexer tracks a 1-based
  line per token; these messages carry a `line N:` prefix). `col` is `null` —
  the lexer does not track columns yet.
- **`type` / `shape` / `purity` / `exhaustiveness` errors:** `function` IS
  populated (extracted from the message, which already names the enclosing
  function). `line`/`col` are `null` for these: attaching a precise source span
  would require threading a span through every AST node, which is deferred. The
  schema already has the fields, so adding precision later is non-breaking.

This is a deliberate first-milestone scope: line-level for lex/parse,
function-level for the semantic phases.

## Examples

Three known type errors (`examples/broken.aria`):

```sh
$ aria check --json examples/broken.aria
[{"severity":"error","phase":"exhaustiveness","code":"E0203","message":"function `code`: non-exhaustive match on Color: missing case `Blue`","line":null,"col":null,"function":"code"},{"severity":"error","phase":"type","code":"E0201","message":"function `wrong_return`: body has type Bool but return type is Int (expected Bool, found Int)","line":null,"col":null,"function":"wrong_return"},{"severity":"error","phase":"type","code":"E0201","message":"function `bad_compare`: cannot compare Int and String","line":null,"col":null,"function":"bad_compare"}]
$ echo $?
1
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
[{"severity":"error","phase":"parse","code":"E0100","message":"line 2: unexpected token Let in expression","line":2,"col":null,"function":null}]
```
