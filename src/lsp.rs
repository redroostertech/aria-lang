//! A minimal stdio Language Server Protocol (LSP) server for Aria — `aria lsp`.
//!
//! This is the *editor half* of the AI-native authoring loop: it surfaces the
//! compiler's structured diagnostics (`src/diagnostics.rs`) live in any LSP
//! editor (VS Code / Cursor / Neovim) and to LLM agents driving an editor. It
//! builds directly on `typeck::check_structured` — the SAME diagnostics
//! `aria check --json` emits — so the squiggles an author sees and the JSON an
//! agent reads are one and the same contract.
//!
//! Scope for v1 is **diagnostics only**. Hover / completion / go-to-definition
//! are future work (they need precise spans, tracked in `docs/DIAGNOSTICS.md`).
//!
//! The project has ZERO external dependencies, so the JSON-RPC framing, a small
//! incoming-JSON parser, and the outgoing JSON are all hand-rolled here. The
//! diagnostic→JSON-string escaping reuses `diagnostics::json_escape`.
//!
//! Transport: standard LSP over stdio — each message is
//! `Content-Length: N\r\n\r\n<N bytes of UTF-8 JSON>` (JSON-RPC 2.0). See
//! `docs/LSP.md` for the editor configuration and the diagnostic mapping.

use crate::diagnostics::{json_escape, Diagnostic};
use std::io::{BufRead, Write};

/// Run the stdio LSP server: read framed messages from stdin, dispatch by
/// `method`, and write framed responses/notifications to stdout. Loops until an
/// `exit` notification (or stdin EOF). Never panics on malformed input — a bad
/// frame or message is logged to stderr and skipped so the editor session is
/// never crashed. Returns the process exit code (0 on a clean shutdown/exit).
pub fn run() -> i32 {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let stdout = std::io::stdout();
    let mut writer = stdout.lock();
    serve(&mut reader, &mut writer)
}

/// The transport loop, generic over the reader/writer so tests can drive it with
/// in-memory buffers. Returns the intended process exit code.
fn serve<R: BufRead, W: Write>(reader: &mut R, writer: &mut W) -> i32 {
    // Tracks whether a `shutdown` request was received; after `shutdown` the
    // spec says the only valid follow-up is `exit` (exit code 0), and any other
    // request should error — but for a minimal server we simply honor `exit`.
    let mut got_shutdown = false;
    loop {
        match read_message(reader) {
            Ok(Some(body)) => {
                match handle_message(&body, writer, &mut got_shutdown) {
                    Action::Continue => {}
                    Action::Exit => return if got_shutdown { 0 } else { 1 },
                }
            }
            // Clean EOF: editor closed the pipe. Treat as exit.
            Ok(None) => return if got_shutdown { 0 } else { 1 },
            Err(e) => {
                // Malformed framing: log and stop (we can no longer trust the
                // stream's byte boundaries). This is rare and unrecoverable.
                eprintln!("aria lsp: framing error: {}", e);
                return 1;
            }
        }
    }
}

enum Action {
    Continue,
    Exit,
}

/// Read one LSP message: parse the `Content-Length` header block, then read
/// exactly that many bytes of body. Returns `Ok(None)` on a clean EOF before any
/// header, `Ok(Some(body))` on success, and `Err` only on an I/O error or a
/// header block that ends mid-stream / lacks a usable `Content-Length`.
fn read_message<R: BufRead>(reader: &mut R) -> Result<Option<String>, String> {
    let mut content_length: Option<usize> = None;
    let mut saw_any_header = false;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("read header: {}", e))?;
        if n == 0 {
            // EOF. If we hadn't started a header block this is a clean shutdown;
            // otherwise the stream was truncated mid-headers.
            if saw_any_header {
                return Err("unexpected EOF in header block".to_string());
            }
            return Ok(None);
        }
        // A header line ends in CRLF; the blank line `\r\n` terminates the block.
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            // End of header block.
            break;
        }
        saw_any_header = true;
        if let Some(rest) = header_value(trimmed, "content-length") {
            content_length = rest.trim().parse::<usize>().ok();
        }
        // Other headers (e.g. Content-Type) are ignored.
    }
    let len = content_length.ok_or_else(|| "missing Content-Length header".to_string())?;
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("read body: {}", e))?;
    String::from_utf8(buf)
        .map(Some)
        .map_err(|e| format!("body is not UTF-8: {}", e))
}

/// Case-insensitively match a `Header-Name: value` line and return the value
/// part (everything after the first colon). `None` if `name` doesn't match.
fn header_value<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let colon = line.find(':')?;
    let (k, v) = line.split_at(colon);
    if k.trim().eq_ignore_ascii_case(name) {
        Some(&v[1..]) // skip the ':'
    } else {
        None
    }
}

/// Write `body` as a framed LSP message (`Content-Length` header + CRLFCRLF +
/// the UTF-8 body). The body is a complete JSON-RPC object string.
fn write_message<W: Write>(writer: &mut W, body: &str) -> std::io::Result<()> {
    write!(writer, "Content-Length: {}\r\n\r\n{}", body.len(), body)?;
    writer.flush()
}

/// Dispatch one incoming JSON-RPC message. Writes any response/notification to
/// `writer`. Returns `Action::Exit` only for the `exit` notification.
fn handle_message<W: Write>(
    body: &str,
    writer: &mut W,
    got_shutdown: &mut bool,
) -> Action {
    let json = match Json::parse(body) {
        Ok(j) => j,
        Err(e) => {
            // A message we cannot even parse is dropped (logged). We can't reply
            // with an error because we don't know the request `id`.
            eprintln!("aria lsp: dropping unparseable message: {}", e);
            return Action::Continue;
        }
    };

    let method = json.get_str("method").unwrap_or_default();
    let method = method.as_str();
    // `id` is present on requests, absent on notifications. We preserve it as a
    // RAW JSON token so a string id round-trips as a string and a number id as a
    // number.
    let id = json.get_raw("id");

    match method {
        "initialize" => {
            if let Some(id) = id {
                let result = r#"{"capabilities":{"textDocumentSync":1},"serverInfo":{"name":"aria-lsp","version":"0.1.0"}}"#;
                let _ = write_message(writer, &rpc_result(&id, result));
            }
        }
        "initialized" => { /* notification, no-op */ }
        "shutdown" => {
            *got_shutdown = true;
            if let Some(id) = id {
                let _ = write_message(writer, &rpc_result(&id, "null"));
            }
        }
        "exit" => {
            return Action::Exit;
        }
        "textDocument/didOpen" => {
            if let (Some(uri), Some(text)) = (
                json.get_path_str(&["params", "textDocument", "uri"]),
                json.get_path_str(&["params", "textDocument", "text"]),
            ) {
                publish_for(writer, &uri, &text);
            }
        }
        "textDocument/didChange" => {
            if let Some(uri) = json.get_path_str(&["params", "textDocument", "uri"]) {
                // Full sync (textDocumentSync = 1): the LAST content change holds
                // the full new document text.
                if let Some(text) = json.last_content_change_text() {
                    publish_for(writer, &uri, &text);
                }
            }
        }
        "textDocument/didSave" => {
            // With Full sync didChange already covers edits; handle didSave too in
            // case the client only re-validates on save. `text` is present only if
            // the server registered `save.includeText` — we didn't, so re-check
            // is best-effort using any text the client sent.
            if let Some(uri) = json.get_path_str(&["params", "textDocument", "uri"]) {
                if let Some(text) = json.get_path_str(&["params", "text"]) {
                    publish_for(writer, &uri, &text);
                }
            }
        }
        "textDocument/didClose" => { /* nothing persisted; no-op */ }
        other => {
            // Unknown request: reply with a JSON-RPC "method not found" so the
            // client isn't left waiting. Unknown notifications are ignored.
            if let Some(id) = id {
                let _ = write_message(
                    writer,
                    &rpc_error(&id, -32601, &format!("method not found: {}", other)),
                );
            }
        }
    }
    Action::Continue
}

/// Run the checker on `text` and publish a `textDocument/publishDiagnostics`
/// notification for `uri` (an EMPTY array when the program is clean, so the
/// editor clears old squiggles).
fn publish_for<W: Write>(writer: &mut W, uri: &str, text: &str) {
    let diags = check_text(text);
    let body = publish_diagnostics_notification(uri, &diags, text);
    let _ = write_message(writer, &body);
}

/// Run the Aria pipeline on in-memory `text` (NOT a file path), mirroring the
/// `aria check --json` path: `prelude::wrap` → `lexer::lex` → `parser::parse` →
/// `typeck::check_structured`. Lex and parse fail fast (one diagnostic each).
pub fn check_text(text: &str) -> Vec<Diagnostic> {
    let wrapped = crate::prelude::wrap(text);
    match crate::lexer::lex(&wrapped) {
        Err(e) => vec![Diagnostic::error("lex", e)],
        Ok(toks) => match crate::parser::parse(toks) {
            Err(e) => vec![Diagnostic::error("parse", e)],
            Ok(program) => {
                let mut diags = crate::typeck::check_structured(&program);
                // On an otherwise-clean program, surface advisory lint warnings
                // (unused `let` bindings) as severity-2 LSP diagnostics so editors
                // underline dead variables. Only lint when there are no errors.
                if diags.is_empty() {
                    diags.extend(crate::dataflow::warnings_for_source(text));
                }
                diags
            }
        },
    }
}

/// Build a `textDocument/publishDiagnostics` notification (a JSON-RPC
/// notification: no `id`). `text` is the document, used to clamp diagnostic line
/// numbers into the user's document range (so a stray prelude-region line — which
/// shouldn't happen for user errors, but we guard it — never points past the
/// document's last line).
pub fn publish_diagnostics_notification(uri: &str, diags: &[Diagnostic], text: &str) -> String {
    // 0-based index of the document's last line. An empty document still has one
    // line (line 0).
    let last_line = text.lines().count().saturating_sub(1);
    let mut items = String::from("[");
    for (i, d) in diags.iter().enumerate() {
        if i > 0 {
            items.push(',');
        }
        items.push_str(&lsp_diagnostic_json(d, last_line));
    }
    items.push(']');

    format!(
        r#"{{"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{{"uri":"{}","diagnostics":{}}}}}"#,
        json_escape(uri),
        items
    )
}

/// Map one Aria [`Diagnostic`] to an LSP `Diagnostic` JSON object.
///
/// - When the diagnostic carries a PRECISE expression span (both `line` and
///   `col`, from the type/shape checker), the `range` is the EXACT extent of the
///   offending sub-expression: `{start:{line,character}, end:{line,character}}`,
///   converted to LSP's 0-based coordinates (`line-1`, `col-1`). The end uses
///   the diagnostic's end position when present, else a one-character span.
/// - Otherwise (only a `line` is known, e.g. a lex/parse error, or no location)
///   the `range` falls back to the WHOLE LINE
///   (`character 0 .. BIG`, which editors clamp to the line length).
/// - All lines are clamped into `[0, last_line]` so the range can never point
///   past the user's document. A `null` line maps to line 0.
/// - `severity` = 1 (Error) for an error diagnostic, 2 (Warning) for a lint
///   warning. `source` = "aria". `code` = the Aria code.
fn lsp_diagnostic_json(d: &Diagnostic, last_line: usize) -> String {
    // LSP lines are 0-based; Aria lines are 1-based (or None → line 0).
    let l0 = match d.line {
        Some(n) => n.saturating_sub(1),
        None => 0,
    };
    let l0 = l0.min(last_line);
    const EOL: u32 = 1_000_000;
    let (start_char, end_line, end_char) = match d.col {
        // Precise span: column known → an exact range. Convert to 0-based.
        Some(c) => {
            let sc = (c.saturating_sub(1)) as u32;
            // End position: the span's end if known, else one char past start.
            let el = match d.end_line {
                Some(n) => n.saturating_sub(1).min(last_line),
                None => l0,
            };
            let ec = match d.end_col {
                Some(n) => (n.saturating_sub(1)) as u32,
                None => sc + 1,
            };
            (sc, el, ec)
        }
        // No column: fall back to the whole-line range.
        None => (0, l0, EOL),
    };
    // LSP DiagnosticSeverity: 1 = Error, 2 = Warning. Map from the Aria severity.
    let severity = if d.severity == "warning" { 2 } else { 1 };
    format!(
        r#"{{"range":{{"start":{{"line":{sl},"character":{sc}}},"end":{{"line":{el},"character":{ec}}}}},"severity":{sev},"source":"aria","code":"{code}","message":"{msg}"}}"#,
        sl = l0,
        sc = start_char,
        el = end_line,
        ec = end_char,
        sev = severity,
        code = d.code,
        msg = json_escape(&d.message),
    )
}

/// JSON-RPC 2.0 success response with the given raw `id` token and raw `result`
/// JSON value.
fn rpc_result(id: &str, result_json: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","id":{},"result":{}}}"#, id, result_json)
}

/// JSON-RPC 2.0 error response with the given raw `id` token, error `code`, and
/// `message`.
fn rpc_error(id: &str, code: i64, message: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":{},"error":{{"code":{},"message":"{}"}}}}"#,
        id,
        code,
        json_escape(message)
    )
}

// ---------------------------------------------------------------------------
// Minimal incoming-JSON parser.
//
// Enough to read the fields the LSP requests we handle carry: strings, numbers,
// objects, arrays, booleans, and null. It builds a small owned tree so callers
// can navigate by key/index. Escaped strings in incoming JSON ARE decoded
// (\" \\ \/ \b \f \n \r \t \uXXXX, including surrogate pairs). Correctness over
// speed; the documents are small.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// Numbers are kept as their source text (we never need arithmetic on them;
    /// `id` may be an int or — per spec — a string, and we only echo it back).
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parse a complete JSON document. Trailing whitespace is allowed; trailing
    /// non-whitespace is an error.
    pub fn parse(s: &str) -> Result<Json, String> {
        let bytes = s.as_bytes();
        let mut p = Parser { b: bytes, i: 0 };
        p.skip_ws();
        let v = p.value()?;
        p.skip_ws();
        if p.i != p.b.len() {
            return Err(format!("trailing data at byte {}", p.i));
        }
        Ok(v)
    }

    /// Look up `key` in an object, returning a borrowed value reference.
    fn get<'a>(&'a self, key: &str) -> Option<&'a Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Get a top-level string field by key (decoded).
    fn get_str(&self, key: &str) -> Option<String> {
        match self.get(key) {
            Some(Json::Str(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Follow a path of object keys and return the final value as a string.
    fn get_path_str(&self, path: &[&str]) -> Option<String> {
        let mut cur = self;
        for k in path {
            cur = cur.get(k)?;
        }
        match cur {
            Json::Str(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Return the RAW JSON text for a top-level field, re-serialized minimally.
    /// Used for `id`, which must round-trip with its original JSON type (a string
    /// id stays a string, a number id stays a number).
    fn get_raw(&self, key: &str) -> Option<String> {
        self.get(key).map(|v| v.to_raw())
    }

    /// `params.contentChanges` is an array; with Full sync the LAST element's
    /// `text` is the full new document. Return that text (decoded).
    fn last_content_change_text(&self) -> Option<String> {
        let changes = self.get("params")?.get("contentChanges")?;
        match changes {
            Json::Arr(items) => match items.last()? {
                obj @ Json::Obj(_) => match obj.get("text") {
                    Some(Json::Str(s)) => Some(s.clone()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// Re-serialize this value to compact JSON text (used only for echoing `id`).
    fn to_raw(&self) -> String {
        match self {
            Json::Null => "null".to_string(),
            Json::Bool(b) => b.to_string(),
            Json::Num(n) => n.clone(),
            Json::Str(s) => format!("\"{}\"", json_escape(s)),
            Json::Arr(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.to_raw()).collect();
                format!("[{}]", inner.join(","))
            }
            Json::Obj(pairs) => {
                let inner: Vec<String> = pairs
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", json_escape(k), v.to_raw()))
                    .collect();
                format!("{{{}}}", inner.join(","))
            }
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() {
            match self.b[self.i] {
                b' ' | b'\t' | b'\n' | b'\r' => self.i += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') | Some(b'f') => self.boolean(),
            Some(b'n') => self.null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!("unexpected byte {:?} at {}", c as char, self.i)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // consume '{'
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected object key at {}", self.i));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at {}", self.i));
            }
            self.i += 1; // consume ':'
            let val = self.value()?;
            pairs.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    continue;
                }
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.i)),
            }
        }
        Ok(Json::Obj(pairs))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // consume '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let val = self.value()?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    continue;
                }
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.i)),
            }
        }
        Ok(Json::Arr(items))
    }

    /// Parse a JSON string starting at the opening quote, decoding all escapes.
    fn string(&mut self) -> Result<String, String> {
        debug_assert_eq!(self.peek(), Some(b'"'));
        self.i += 1; // consume opening quote
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let esc = self.peek().ok_or("unterminated escape")?;
                    self.i += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let cp = self.hex4()?;
                            // Handle a UTF-16 surrogate pair.
                            if (0xD800..=0xDBFF).contains(&cp) {
                                // high surrogate; expect a following \uXXXX low.
                                if self.peek() == Some(b'\\') {
                                    self.i += 1;
                                    if self.peek() == Some(b'u') {
                                        self.i += 1;
                                        let low = self.hex4()?;
                                        if (0xDC00..=0xDFFF).contains(&low) {
                                            let c = 0x10000
                                                + ((cp - 0xD800) << 10)
                                                + (low - 0xDC00);
                                            out.push(
                                                char::from_u32(c)
                                                    .ok_or("invalid surrogate pair")?,
                                            );
                                        } else {
                                            return Err("invalid low surrogate".to_string());
                                        }
                                    } else {
                                        return Err("expected \\u low surrogate".to_string());
                                    }
                                } else {
                                    return Err("lone high surrogate".to_string());
                                }
                            } else if (0xDC00..=0xDFFF).contains(&cp) {
                                return Err("lone low surrogate".to_string());
                            } else {
                                out.push(char::from_u32(cp).ok_or("invalid \\u escape")?);
                            }
                        }
                        other => {
                            return Err(format!("invalid escape \\{}", other as char))
                        }
                    }
                }
                // A raw byte: collect this and any following continuation bytes
                // into a UTF-8 char. JSON strings are UTF-8; we already have a
                // valid `&str`, so multi-byte sequences are contiguous.
                _ => {
                    // Determine the UTF-8 length from the lead byte.
                    let len = utf8_len(c);
                    let start = self.i - 1;
                    let end = start + len;
                    if end > self.b.len() {
                        return Err("truncated UTF-8 in string".to_string());
                    }
                    let s = std::str::from_utf8(&self.b[start..end])
                        .map_err(|_| "invalid UTF-8 in string")?;
                    out.push_str(s);
                    self.i = end;
                }
            }
        }
        Ok(out)
    }

    /// Read exactly 4 hex digits as a u32 code point.
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("truncated \\u escape".to_string());
        }
        let hex = std::str::from_utf8(&self.b[self.i..self.i + 4])
            .map_err(|_| "bad \\u escape")?;
        let v = u32::from_str_radix(hex, 16).map_err(|_| "bad \\u hex")?;
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.i += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad number")?;
        if s.is_empty() {
            return Err("empty number".to_string());
        }
        Ok(Json::Num(s.to_string()))
    }

    fn boolean(&mut self) -> Result<Json, String> {
        if self.b[self.i..].starts_with(b"true") {
            self.i += 4;
            Ok(Json::Bool(true))
        } else if self.b[self.i..].starts_with(b"false") {
            self.i += 5;
            Ok(Json::Bool(false))
        } else {
            Err(format!("invalid literal at {}", self.i))
        }
    }

    fn null(&mut self) -> Result<Json, String> {
        if self.b[self.i..].starts_with(b"null") {
            self.i += 4;
            Ok(Json::Null)
        } else {
            Err(format!("invalid literal at {}", self.i))
        }
    }
}

/// UTF-8 byte length implied by a lead byte.
fn utf8_len(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead >> 5 == 0b110 {
        2
    } else if lead >> 4 == 0b1110 {
        3
    } else if lead >> 3 == 0b11110 {
        4
    } else {
        1 // invalid lead; from_utf8 will reject downstream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Frame reader/writer round-trip -----------------------------------

    #[test]
    fn frame_write_then_read_round_trips() {
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, body).unwrap();
        // The framed bytes start with the Content-Length header.
        let text = String::from_utf8(buf.clone()).unwrap();
        assert!(text.starts_with(&format!("Content-Length: {}\r\n\r\n", body.len())));
        // And reading it back yields the original body.
        let mut cursor = std::io::Cursor::new(buf);
        let got = read_message(&mut cursor).unwrap().unwrap();
        assert_eq!(got, body);
    }

    #[test]
    fn reader_handles_multiple_messages_and_eof() {
        let a = r#"{"a":1}"#;
        let b = r#"{"b":2}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, a).unwrap();
        write_message(&mut buf, b).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        assert_eq!(read_message(&mut cur).unwrap().unwrap(), a);
        assert_eq!(read_message(&mut cur).unwrap().unwrap(), b);
        // Clean EOF.
        assert_eq!(read_message(&mut cur).unwrap(), None);
    }

    #[test]
    fn reader_is_case_insensitive_and_ignores_extra_headers() {
        let body = r#"{"ok":true}"#;
        let framed = format!(
            "content-length: {}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n{}",
            body.len(),
            body
        );
        let mut cur = std::io::Cursor::new(framed.into_bytes());
        assert_eq!(read_message(&mut cur).unwrap().unwrap(), body);
    }

    // --- Incoming-JSON field extraction (incl. escaped strings) -----------

    #[test]
    fn extracts_method_and_nested_uri_text() {
        let msg = r#"{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///x.aria","text":"fn main() -> Int = 1"}}}"#;
        let j = Json::parse(msg).unwrap();
        assert_eq!(j.get_str("method").unwrap(), "textDocument/didOpen");
        assert_eq!(
            j.get_path_str(&["params", "textDocument", "uri"]).unwrap(),
            "file:///x.aria"
        );
        assert_eq!(
            j.get_path_str(&["params", "textDocument", "text"]).unwrap(),
            "fn main() -> Int = 1"
        );
    }

    #[test]
    fn decodes_escaped_strings_in_incoming_json() {
        // Newlines, quotes, backslashes, tab, and a \u escape.
        let msg = r#"{"text":"line1\nline2\t\"q\"\\doneA"}"#;
        let j = Json::parse(msg).unwrap();
        assert_eq!(j.get_str("text").unwrap(), "line1\nline2\t\"q\"\\doneA");
    }

    #[test]
    fn decodes_surrogate_pair() {
        // U+1F600 GRINNING FACE encoded as a UTF-16 surrogate pair.
        let msg = r#"{"text":"😀"}"#;
        let j = Json::parse(msg).unwrap();
        assert_eq!(j.get_str("text").unwrap(), "\u{1F600}");
    }

    #[test]
    fn id_round_trips_as_number_and_string() {
        let n = Json::parse(r#"{"id":42}"#).unwrap();
        assert_eq!(n.get_raw("id").unwrap(), "42");
        let s = Json::parse(r#"{"id":"abc"}"#).unwrap();
        assert_eq!(s.get_raw("id").unwrap(), "\"abc\"");
        // A notification has no id.
        let none = Json::parse(r#"{"method":"initialized"}"#).unwrap();
        assert_eq!(none.get_raw("id"), None);
    }

    #[test]
    fn last_content_change_text_for_full_sync() {
        let msg = r#"{"method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///a"},"contentChanges":[{"text":"old"},{"text":"new full doc"}]}}"#;
        let j = Json::parse(msg).unwrap();
        assert_eq!(j.last_content_change_text().unwrap(), "new full doc");
    }

    // --- Diagnostic -> LSP JSON mapping -----------------------------------

    #[test]
    fn maps_diagnostic_to_lsp_range_severity_code() {
        // A parse error carries a 1-based line; LSP is 0-based.
        let d = Diagnostic::error("parse", "line 3: unexpected token".into());
        assert_eq!(d.line, Some(3));
        let js = lsp_diagnostic_json(&d, 100);
        assert!(js.contains(r#""start":{"line":2,"character":0}"#), "{}", js);
        assert!(js.contains(r#""severity":1"#));
        assert!(js.contains(r#""source":"aria""#));
        assert!(js.contains(r#""code":"E0100""#));
        assert!(js.contains("unexpected token"));
    }

    #[test]
    fn null_line_maps_to_zero_and_clamps_to_last_line() {
        // A semantic error with no line maps to LSP line 0.
        let d = Diagnostic::error("type", "function `f`: cannot compare Int and String".into());
        assert_eq!(d.line, None);
        let js = lsp_diagnostic_json(&d, 5);
        assert!(js.contains(r#""line":0"#), "{}", js);

        // A line beyond the document is clamped down to last_line.
        let d2 = Diagnostic::error("parse", "line 999: oops".into());
        let js2 = lsp_diagnostic_json(&d2, 4);
        assert!(js2.contains(r#""start":{"line":4,"character":0}"#), "{}", js2);
    }

    // --- Full in-process check: broken -> diagnostics, clean -> [] --------

    #[test]
    fn broken_program_yields_expected_lsp_diagnostics() {
        let broken = r#"type Color = | Red | Green | Blue
fn code(c: Color) -> Int =
  match c {
    Red   => 0,
    Green => 1,
  }
fn wrong_return() -> Int = true
fn bad_compare(n: Int) -> Bool = n == "five"
fn main() -> Int = code(Blue)
"#;
        let diags = check_text(broken);
        assert!(!diags.is_empty(), "broken program should have diagnostics");
        // Expect the three documented errors.
        let codes: Vec<&str> = diags.iter().map(|d| d.code).collect();
        assert!(codes.contains(&"E0203"), "non-exhaustive match: {:?}", codes);
        assert!(
            codes.iter().filter(|c| **c == "E0201").count() >= 2,
            "two type mismatches: {:?}",
            codes
        );

        // And the published notification is well-formed and non-empty.
        let note = publish_diagnostics_notification("file:///broken.aria", &diags, broken);
        assert!(note.contains(r#""method":"textDocument/publishDiagnostics""#));
        assert!(note.contains(r#""uri":"file:///broken.aria""#));
        assert!(note.contains(r#""severity":1"#));
        assert!(note.contains(r#""code":"E0203""#));
        assert!(!note.contains(r#""diagnostics":[]"#));
    }

    #[test]
    fn unused_let_published_as_warning_severity_2() {
        // A program that type-checks cleanly but has an unused `let` yields a
        // single severity-2 (Warning) LSP diagnostic with code W0001 and a
        // precise range — and NOT an error (severity 1).
        let src = "fn f() -> Int = { let tmp = 99; 1 }\nfn main() -> Int = f()";
        let diags = check_text(src);
        assert_eq!(diags.len(), 1, "exactly one warning: {:?}", diags);
        assert_eq!(diags[0].severity, "warning");
        assert_eq!(diags[0].code, "W0001");
        let note = publish_diagnostics_notification("file:///u.aria", &diags, src);
        assert!(note.contains(r#""severity":2"#), "must be severity 2: {}", note);
        assert!(note.contains(r#""code":"W0001""#), "{}", note);
        // Precise range, not the whole-line fallback.
        assert!(!note.contains("1000000"), "warning should carry a precise span: {}", note);
    }

    #[test]
    fn clean_program_yields_empty_array() {
        let clean = "fn main() -> Int = 1\n";
        let diags = check_text(clean);
        assert!(diags.is_empty(), "clean program: {:?}", diags);
        let note = publish_diagnostics_notification("file:///clean.aria", &diags, clean);
        assert!(note.contains(r#""diagnostics":[]"#), "{}", note);
    }

    // --- End-to-end serve() loop ------------------------------------------

    #[test]
    fn serve_initialize_didopen_didchange_exit() {
        let mut input = Vec::new();
        let push = |buf: &mut Vec<u8>, body: &str| {
            write_message(buf, body).unwrap();
        };
        push(&mut input, r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#);
        push(&mut input, r#"{"jsonrpc":"2.0","method":"initialized","params":{}}"#);
        push(
            &mut input,
            r#"{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///b.aria","text":"fn wrong_return() -> Int = true\nfn main() -> Int = 1"}}}"#,
        );
        push(
            &mut input,
            r#"{"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///b.aria"},"contentChanges":[{"text":"fn main() -> Int = 1"}]}}"#,
        );
        push(&mut input, r#"{"jsonrpc":"2.0","id":2,"method":"shutdown"}"#);
        push(&mut input, r#"{"jsonrpc":"2.0","method":"exit"}"#);

        let mut reader = std::io::Cursor::new(input);
        let mut out: Vec<u8> = Vec::new();
        let code = serve(&mut reader, &mut out);
        assert_eq!(code, 0, "clean shutdown+exit should be 0");

        let text = String::from_utf8(out).unwrap();
        // initialize response advertises Full textDocumentSync.
        assert!(text.contains(r#""capabilities":{"textDocumentSync":1}"#), "{}", text);
        assert!(text.contains(r#""id":1"#));
        // didOpen publishes a diagnostic (the broken wrong_return).
        assert!(text.contains(r#""code":"E0201""#), "didOpen should publish E0201: {}", text);
        // didChange to a clean program clears squiggles (empty array).
        assert!(text.contains(r#""diagnostics":[]"#), "didChange should clear: {}", text);
        // shutdown returns null.
        assert!(text.contains(r#""id":2,"result":null"#), "{}", text);
    }

    #[test]
    fn precise_expression_error_publishes_exact_range_not_whole_line() {
        // A type error mid-expression (`1 + true`) now carries a precise span, so
        // the LSP must emit an EXACT range (line:char start..end) rather than the
        // whole-line `character:0 .. 1000000` fallback. `1 + true` sits at columns
        // 17..25 on line 1, i.e. 0-based LSP `line:0, char:16 .. 24`.
        let mut input = Vec::new();
        let push = |buf: &mut Vec<u8>, body: &str| {
            write_message(buf, body).unwrap();
        };
        push(&mut input, r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#);
        push(
            &mut input,
            r#"{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///p.aria","text":"fn f() -> Int = 1 + true\nfn main() -> Int = f()"}}}"#,
        );
        push(&mut input, r#"{"jsonrpc":"2.0","id":2,"method":"shutdown"}"#);
        push(&mut input, r#"{"jsonrpc":"2.0","method":"exit"}"#);

        let mut reader = std::io::Cursor::new(input);
        let mut out: Vec<u8> = Vec::new();
        let code = serve(&mut reader, &mut out);
        assert_eq!(code, 0);
        let text = String::from_utf8(out).unwrap();
        // The published diagnostic carries the EXACT range, not a whole-line one.
        assert!(
            text.contains(r#""range":{"start":{"line":0,"character":16},"end":{"line":0,"character":24}}"#),
            "expected an exact range, got: {}",
            text
        );
        // And it is NOT the whole-line fallback (the big sentinel end character).
        assert!(!text.contains("1000000"), "should not be a whole-line range: {}", text);
        assert!(text.contains(r#""code":"E0201""#), "{}", text);
    }

    #[test]
    fn parse_error_still_falls_back_to_whole_line_range() {
        // A lex/parse error knows only a line (no column), so the LSP keeps the
        // whole-line range. (`@` is an unexpected character on line 1.)
        let d = Diagnostic::error("lex", "line 1: unexpected character `@`".into());
        let js = lsp_diagnostic_json(&d, 10);
        assert!(js.contains(r#""start":{"line":0,"character":0}"#), "{}", js);
        assert!(js.contains("1000000"), "whole-line fallback expected: {}", js);
    }

    #[test]
    fn pattern_type_error_publishes_exact_pattern_range_not_whole_line() {
        // A pattern of the wrong type now carries the pattern NODE's precise span,
        // so the LSP emits an EXACT range over the pattern, not a whole-line one.
        // Line 2 (0-based line 1) is `  Box(v) => v,`; `Box(v)` is cols 3..9
        // (1-based), i.e. 0-based LSP characters 2..8.
        let src = "fn g(n: Int) -> Int = match n {\n  Box(v) => v,\n  _ => 0\n}\ntype Boxed = | Box(Int)\nfn main() -> Int = g(1)";
        let diags = check_text(src);
        assert!(!diags.is_empty(), "expected a pattern type error");
        let note = publish_diagnostics_notification("file:///pat.aria", &diags, src);
        assert!(
            note.contains(r#""range":{"start":{"line":1,"character":2},"end":{"line":1,"character":8}}"#),
            "expected an exact pattern range, got: {}",
            note
        );
        assert!(!note.contains("1000000"), "should not be a whole-line range: {}", note);
    }

    #[test]
    fn let_annotation_error_publishes_exact_statement_range() {
        // A `let x: Int = true;` annotation mismatch carries the let STATEMENT's
        // precise span, so the LSP range covers the whole statement exactly. Line 2
        // (0-based line 1) is `  let x: Int = true;`; the statement is 1-based cols
        // 3..21, i.e. 0-based LSP characters 2..20.
        let src = "fn f() -> Int = {\n  let x: Int = true;\n  x\n}\nfn main() -> Int = f()";
        let diags = check_text(src);
        assert!(!diags.is_empty(), "expected a let-annotation type error");
        let note = publish_diagnostics_notification("file:///let.aria", &diags, src);
        assert!(
            note.contains(r#""range":{"start":{"line":1,"character":2},"end":{"line":1,"character":20}}"#),
            "expected an exact let-statement range, got: {}",
            note
        );
        assert!(!note.contains("1000000"), "should not be a whole-line range: {}", note);
    }

    #[test]
    fn unknown_request_gets_method_not_found_error() {
        let mut out: Vec<u8> = Vec::new();
        let mut got_shutdown = false;
        let body = r#"{"jsonrpc":"2.0","id":7,"method":"textDocument/hover","params":{}}"#;
        handle_message(body, &mut out, &mut got_shutdown);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains(r#""id":7"#));
        assert!(text.contains(r#""code":-32601"#), "{}", text);
    }

    #[test]
    fn malformed_message_is_dropped_not_panicked() {
        let mut out: Vec<u8> = Vec::new();
        let mut got_shutdown = false;
        // Not valid JSON.
        let action = handle_message("{not json", &mut out, &mut got_shutdown);
        assert!(matches!(action, Action::Continue));
        assert!(out.is_empty(), "no response for unparseable input");
    }
}
