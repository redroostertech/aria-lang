//! Machine-readable structured diagnostics for `aria check --json`.
//!
//! This is the *contract* an AI authoring loop (and a future LSP) reads to
//! self-correct: every compiler error is reported as a [`Diagnostic`] with a
//! STABLE short `code` per error CATEGORY, a `phase`, a human `message`, and —
//! where tractable — a source `line`/`col` and the enclosing `function`.
//!
//! The JSON is HAND-WRITTEN (the project has zero external dependencies, so no
//! serde): the only structured output is a top-level array — `[]` when the
//! program is clean, `[{...},{...}]` otherwise. The schema is forward-compatible:
//! consumers must ignore unknown fields and tolerate `null` for `line`/`col`/
//! `function`, so precise spans can be added later without breaking them.
//!
//! See `docs/DIAGNOSTICS.md` for the full schema + code table.

/// One structured compiler diagnostic. Field set is intentionally minimal and
/// stable; new fields may be added later (consumers must ignore unknowns).
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    /// `"error"` today; room for `"warning"` later.
    pub severity: &'static str,
    /// The compiler phase that produced this: one of `lex`, `parse`, `type`,
    /// `shape`, `purity`, `exhaustiveness`.
    pub phase: &'static str,
    /// A STABLE short code identifying the error CATEGORY (e.g. `"E0201"`).
    pub code: &'static str,
    /// Human-readable message (the same text the non-`--json` path prints).
    pub message: String,
    /// 1-based source line, if known; else `None`.
    pub line: Option<usize>,
    /// 1-based column, if known; else `None`. Currently always `None`
    /// (the lexer/parser track line but not column) — reserved for precise
    /// spans.
    pub col: Option<usize>,
    /// Enclosing function name, if known; else `None`.
    pub function: Option<String>,
}

impl Diagnostic {
    /// Build an error-severity diagnostic, classifying `message` (in `phase`)
    /// into a stable `code`, and extracting the enclosing `function` name and a
    /// source `line` from the message text when they are embedded there.
    pub fn error(phase: &'static str, message: String) -> Diagnostic {
        let function = extract_function(&message);
        let line = extract_line(&message);
        let code = classify(phase, &message);
        Diagnostic {
            severity: "error",
            phase,
            code,
            message,
            line,
            col: None,
            function,
        }
    }

    /// Serialize this single diagnostic as a JSON object.
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push('{');
        s.push_str(&format!("\"severity\":\"{}\",", self.severity));
        s.push_str(&format!("\"phase\":\"{}\",", self.phase));
        s.push_str(&format!("\"code\":\"{}\",", self.code));
        s.push_str(&format!("\"message\":\"{}\",", json_escape(&self.message)));
        s.push_str(&format!("\"line\":{},", json_num_or_null(self.line)));
        s.push_str(&format!("\"col\":{},", json_num_or_null(self.col)));
        match &self.function {
            Some(f) => s.push_str(&format!("\"function\":\"{}\"", json_escape(f))),
            None => s.push_str("\"function\":null"),
        }
        s.push('}');
        s
    }
}

/// Serialize a slice of diagnostics as a JSON array (the top-level output).
/// `[]` when empty. Compact, one object per element, no trailing comma.
pub fn array_to_json(diags: &[Diagnostic]) -> String {
    let mut s = String::from("[");
    for (i, d) in diags.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&d.to_json());
    }
    s.push(']');
    s
}

fn json_num_or_null(n: Option<usize>) -> String {
    match n {
        Some(v) => v.to_string(),
        None => "null".to_string(),
    }
}

/// Escape a string for embedding inside a JSON string literal: quotes,
/// backslashes, the standard short escapes, and any remaining control
/// character (< 0x20) as a `\u00XX` sequence. Produces valid JSON for any
/// input.
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// If `message` embeds an enclosing function name in one of the well-known
/// shapes the checker emits, return it. Best-effort; `None` otherwise.
///
/// Recognised shapes (all produced by `typeck`/`shape`):
///   - `` function `NAME`: ... ``                (type / shape / generic-arg)
///   - `` function `NAME` ... ``                  (return type, arity, pure, ...)
///   - `` cannot prove `NAME` pure: ... ``        (higher-order purity)
///   - `` type parameter `P` of `NAME` is unused``
fn extract_function(message: &str) -> Option<String> {
    // `cannot prove `NAME` pure` — the function is the FIRST backtick group.
    if message.starts_with("cannot prove `") {
        return backtick_after(message, "cannot prove ");
    }
    // `type parameter `P` of `NAME` is unused` — the function is after `of `.
    if message.starts_with("type parameter `") {
        if let Some(idx) = message.find("` of `") {
            return backtick_after(message, &message[..idx + "` of ".len()]);
        }
        return None;
    }
    // The common `function `NAME` ...` / `function `NAME`: ...` shape.
    if message.starts_with("function `") {
        return backtick_after(message, "function ");
    }
    None
}

/// Given `prefix` immediately followed by a `` `NAME` `` backtick group in
/// `message`, return `NAME`.
fn backtick_after(message: &str, prefix: &str) -> Option<String> {
    let rest = message.strip_prefix(prefix)?;
    let rest = rest.strip_prefix('`')?;
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
}

/// Extract a 1-based source line from a `line N: ...` prefix (lex/parse
/// errors carry it). `None` if absent.
fn extract_line(message: &str) -> Option<usize> {
    let rest = message.strip_prefix("line ")?;
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end].parse().ok()
}

/// Classify a `(phase, message)` pair into a STABLE category `code`.
///
/// Classification is by message-text inspection within a phase. This is a
/// deliberate, documented design choice for the first milestone: the string
/// messages are the existing source of truth, so a small classifier over them
/// avoids a large refactor while still giving consumers stable codes. The codes
/// (not the heuristics) are the contract.
pub fn classify(phase: &str, message: &str) -> &'static str {
    match phase {
        "lex" => "E0001",
        "parse" => "E0100",
        "shape" => "E0300",
        "purity" => "E0210",
        "exhaustiveness" => "E0203",
        "type" => classify_type(message),
        _ => "E0900",
    }
}

/// Sub-classify a `type`-phase message. Several distinct phases historically
/// flow through `typeck::check`'s `Vec<String>` (type, exhaustiveness, purity,
/// shape); `check_structured` routes the clearly-separable ones (shape, purity,
/// exhaustiveness) to their own phase before calling this, but this function is
/// also robust to seeing them, so codes stay stable regardless.
fn classify_type(m: &str) -> &'static str {
    // Purity (also reachable directly as the `purity` phase).
    if m.contains("is declared `pure` but performs IO")
        || m.starts_with("cannot prove `")
    {
        return "E0210";
    }
    // Exhaustiveness.
    if m.contains("non-exhaustive match") {
        return "E0203";
    }
    // Unknown / undefined names.
    if m.contains("unbound variable")
        || m.contains("unknown function")
        || m.contains("unknown constructor")
        || m.contains("unknown record type")
    {
        return "E0200";
    }
    // Unknown type, type parameter, or constructor space.
    if m.contains("unknown type") || m.contains("unknown type parameter") {
        return "E0204";
    }
    // Unused / un-inferable type parameter (phantom).
    if m.contains("is unused (cannot be inferred)") {
        return "E0205";
    }
    // Arity: wrong number of arguments / fields / parameters / type arguments.
    if m.contains("expects") && (m.contains("argument(s)") || m.contains("field(s)"))
        || m.contains("takes") && m.contains("parameter(s)")
        || m.contains("type argument(s)")
        || m.contains("element tuple pattern cannot match")
    {
        return "E0202";
    }
    // Trait / bound resolution failures.
    if m.contains("requires its type parameter")
        || m.contains("is not bounded by")
        || m.contains("trait method")
    {
        return "E0206";
    }
    // Duplicate / redefinition declarations.
    if m.starts_with("duplicate ") || m.contains("cannot redefine built-in") {
        return "E0207";
    }
    // Everything else in the type phase is a type-mismatch family error
    // (`body has type .. but return type is ..`, `cannot compare ..`,
    //  `cannot apply ..`, `expected .., found ..`, ...).
    "E0201"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_array_is_brackets() {
        assert_eq!(array_to_json(&[]), "[]");
    }

    #[test]
    fn escape_quotes_newlines_backslashes_controls() {
        let s = "a\"b\\c\nd\te\r\u{0}f";
        let e = json_escape(s);
        assert_eq!(e, "a\\\"b\\\\c\\nd\\te\\r\\u0000f");
    }

    #[test]
    fn single_diag_is_well_formed_object() {
        let d = Diagnostic::error("type", "function `f`: cannot compare Int and String".into());
        let j = d.to_json();
        assert!(j.contains("\"severity\":\"error\""));
        assert!(j.contains("\"phase\":\"type\""));
        assert!(j.contains("\"code\":\"E0201\""));
        assert!(j.contains("\"function\":\"f\""));
        assert!(j.contains("\"line\":null"));
        assert!(j.contains("\"col\":null"));
        // Braces balance.
        assert!(j.starts_with('{') && j.ends_with('}'));
    }

    #[test]
    fn array_separates_with_commas_no_trailing() {
        let d1 = Diagnostic::error("lex", "line 3: unexpected character `@`".into());
        let d2 = Diagnostic::error("parse", "line 5: expected RParen, found Eof".into());
        let arr = array_to_json(&[d1, d2]);
        assert!(arr.starts_with("[{"));
        assert!(arr.ends_with("}]"));
        assert!(arr.contains("},{"));
        assert!(!arr.contains(",]"));
    }

    #[test]
    fn line_extracted_from_parse_lex() {
        let d = Diagnostic::error("parse", "line 7: unexpected token".into());
        assert_eq!(d.line, Some(7));
        assert_eq!(d.code, "E0100");
        let l = Diagnostic::error("lex", "line 2: unexpected character `@`".into());
        assert_eq!(l.line, Some(2));
        assert_eq!(l.code, "E0001");
    }

    #[test]
    fn function_extracted_from_type_messages() {
        let d = Diagnostic::error(
            "type",
            "function `bad_compare`: cannot compare Int and String".into(),
        );
        assert_eq!(d.function.as_deref(), Some("bad_compare"));
        let r = Diagnostic::error(
            "type",
            "function `wrong_return`: body has type Bool but return type is Int (expected Bool, found Int)".into(),
        );
        assert_eq!(r.function.as_deref(), Some("wrong_return"));
        assert_eq!(r.code, "E0201");
    }

    #[test]
    fn function_extracted_from_pure_witness() {
        let d = Diagnostic::error(
            "purity",
            "cannot prove `run` pure: it calls a function value `f` whose effects are unknown".into(),
        );
        assert_eq!(d.function.as_deref(), Some("run"));
        assert_eq!(d.code, "E0210");
    }

    #[test]
    fn classifier_maps_representative_messages() {
        assert_eq!(classify("lex", "line 1: bad"), "E0001");
        assert_eq!(classify("parse", "line 1: expected"), "E0100");
        assert_eq!(classify("shape", "matmul inner dimensions"), "E0300");
        assert_eq!(
            classify("type", "function `c`: non-exhaustive match on Color: missing case `Blue`"),
            "E0203"
        );
        assert_eq!(classify("exhaustiveness", "anything"), "E0203");
        assert_eq!(classify("type", "unbound variable `x`"), "E0200");
        assert_eq!(classify("type", "function `f`: unknown function `g`"), "E0200");
        assert_eq!(
            classify("type", "function `f` return type: unknown type `Foo`"),
            "E0204"
        );
        assert_eq!(
            classify("type", "type parameter `T` of `f` is unused (cannot be inferred)"),
            "E0205"
        );
        assert_eq!(
            classify("type", "function `f` expects 2 argument(s), got 3"),
            "E0202"
        );
        assert_eq!(
            classify("type", "function `f`: cannot compare Int and String"),
            "E0201"
        );
        assert_eq!(classify("type", "duplicate function `f`"), "E0207");
        assert_eq!(
            classify("purity", "function `g` is declared `pure` but performs IO (calls `print_int`)"),
            "E0210"
        );
    }
}
