# Aria — Language Server (`aria lsp`)

> The **editor half** of the AI-native authoring loop. `aria lsp` is a stdio
> Language Server that surfaces the Aria compiler's structured diagnostics
> **live** in any LSP editor (VS Code / Cursor / Neovim) and to LLM agents that
> drive an editor. It is the same diagnostics contract as
> [`aria check --json`](DIAGNOSTICS.md) — the squiggles a human sees and the
> JSON an agent reads are one and the same.

## What it supports (v1)

- **Diagnostics only.** On open / change / save the server type-checks the
  in-memory document and publishes diagnostics. No hover, completion, or
  go-to-definition yet (those need precise spans — see *Limitations*).
- **Full document sync** (`textDocumentSync = 1`): each change sends the whole
  new document, which the server re-checks.
- **Transport:** standard LSP over **stdio** — every message is framed as
  `Content-Length: N\r\n\r\n<N bytes of UTF-8 JSON>` (JSON-RPC 2.0). The framing,
  a minimal incoming-JSON parser, and the outgoing JSON are all hand-rolled
  (Aria has zero external dependencies).

### Methods handled

| method                          | kind         | behavior                                                            |
|---------------------------------|--------------|---------------------------------------------------------------------|
| `initialize`                    | request      | replies with `capabilities.textDocumentSync = 1` and `serverInfo` (`aria-lsp` / `0.1.0`). |
| `initialized`                   | notification | no-op.                                                              |
| `textDocument/didOpen`          | notification | checks `params.textDocument.text`, publishes diagnostics for the uri. |
| `textDocument/didChange`        | notification | re-checks the **last** `contentChanges[].text` (Full sync = full document), re-publishes. |
| `textDocument/didSave`          | notification | re-checks `params.text` if the client included it (best-effort).    |
| `textDocument/didClose`         | notification | no-op.                                                              |
| `shutdown`                      | request      | returns `null`.                                                     |
| `exit`                          | notification | terminates the server (exit code 0 after `shutdown`, else 1).       |
| *unknown request*               | request      | replies with JSON-RPC error `-32601` (method not found).            |
| *unknown notification*          | notification | ignored.                                                            |

The server **never panics** on malformed input: an unparseable message is logged
to stderr and dropped (it cannot be answered because its `id` is unknown), and a
broken `Content-Length` frame logs and stops. It will not crash the editor
session.

## Diagnostic → LSP mapping

Each Aria [`Diagnostic`](DIAGNOSTICS.md#schema) maps to one LSP `Diagnostic`:

| LSP field   | value                                                                                  |
|-------------|----------------------------------------------------------------------------------------|
| `range`     | a **whole-line** range: `start = {line: L0, character: 0}`, `end = {line: L0, character: 1000000}`. |
| `L0`        | `diagnostic.line - 1` (LSP lines are 0-based), clamped `>= 0` and `<= last document line`. A `null` line maps to line 0. |
| `severity`  | `1` (Error).                                                                           |
| `source`    | `"aria"`.                                                                              |
| `code`      | the stable Aria code (e.g. `"E0201"`) — see the code table in [DIAGNOSTICS.md](DIAGNOSTICS.md#code-table). |
| `message`   | the human-readable message (identical to `aria check`).                                |

When the program is **clean**, the server publishes an **empty** `diagnostics`
array, so the editor clears old squiggles.

The checker runs on the in-memory text via the same pipeline as
`aria check --json`: `prelude::wrap(text)` → lex → parse → `check_structured`.
The prelude is *appended*, so user line numbers are preserved; a diagnostic whose
line somehow falls past the user's document is clamped to the document's last
line.

## Editor configuration

The server is the command `aria lsp` (build the binary with `cargo build`; it is
`target/debug/aria` or install `aria` on your `PATH`). Associate it with the
`.aria` file extension / `aria` language id.

### VS Code / Cursor

There is no published extension yet. The simplest path is a tiny extension that
starts the server over stdio using
[`vscode-languageclient`](https://www.npmjs.com/package/vscode-languageclient):

```ts
import { LanguageClient, TransportKind } from "vscode-languageclient/node";

export function activate() {
  const client = new LanguageClient(
    "aria",
    "Aria Language Server",
    {
      command: "aria",          // or an absolute path to the built binary
      args: ["lsp"],
      transport: TransportKind.stdio,
    },
    {
      documentSelector: [{ scheme: "file", language: "aria" }],
    }
  );
  client.start();
}
```

Register the language + extension in the extension's `package.json`:

```json
{
  "contributes": {
    "languages": [
      { "id": "aria", "extensions": [".aria"] }
    ]
  }
}
```

### Neovim (built-in LSP)

```lua
vim.filetype.add({ extension = { aria = "aria" } })

vim.api.nvim_create_autocmd("FileType", {
  pattern = "aria",
  callback = function(args)
    vim.lsp.start({
      name = "aria",
      cmd = { "aria", "lsp" },   -- or an absolute path to the built binary
      root_dir = vim.fs.dirname(args.file),
    })
  end,
})
```

## Limitations

- **Diagnostics only.** No hover, completion, signature help, or
  go-to-definition yet.
- **Whole-line ranges.** The compiler currently tracks **line, not column**, for
  most errors (lex/parse carry a line; semantic errors carry the enclosing
  function but no span), so diagnostics highlight the whole line. This ties
  directly into the future **precise-spans** work noted in
  [DIAGNOSTICS.md](DIAGNOSTICS.md#location-precision-what-is-populated-today):
  once spans are threaded through the AST, the LSP `range` can narrow to the
  exact token, and hover/go-to-def become tractable.
- **Full sync only** (`textDocumentSync = 1`); no incremental updates. Documents
  are small, so re-checking the whole text on each change is cheap.
- No workspace/multi-file awareness — each document is checked on its own (with
  the Aria prelude appended, as in `aria check`).

## Why this matters

`aria check --json` gives an LLM agent a machine-readable feedback channel to
self-correct. `aria lsp` is the **editor-facing** counterpart of that same
channel: it puts those diagnostics live in front of a human author *and* into any
agent that operates through an editor — closing the AI-native authoring loop on
the editor side.
