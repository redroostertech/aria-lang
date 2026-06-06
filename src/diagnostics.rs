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
    /// 1-based column, if known; else `None`. Populated (together with `line`)
    /// for expression-level type/shape errors from the precise source span of
    /// the offending sub-expression; `None` for declaration-level errors and
    /// errors the checker cannot locate to a single expression.
    pub col: Option<usize>,
    /// 1-based END line of the offending expression's span (one past its last
    /// character in column terms), if a precise span is known; else `None`.
    pub end_line: Option<usize>,
    /// 1-based END column of the offending expression's span, if known; else
    /// `None`. With `line`/`col` this gives consumers (the LSP) an exact range.
    pub end_col: Option<usize>,
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
        // Some errors are RAISED in one phase but logically belong to another
        // (e.g. a file-read failure arrives via the `lex` construction site; an
        // interface/impl arity mismatch is raised during parser-time trait
        // lowering). Reclassify by message content so the reported (phase, code)
        // matches the documented bucket regardless of where it was raised.
        let (phase, code) = reclassify(phase, &message);
        Diagnostic {
            severity: "error",
            phase,
            code,
            message,
            line,
            col: None,
            end_line: None,
            end_col: None,
            function,
        }
    }

    /// Overwrite this diagnostic's location with a precise source span (1-based
    /// start and end line/column), as recorded by the type/shape checker for the
    /// offending sub-expression. Supersedes any line extracted from the message
    /// text and fills in the column + end position the LSP renders as an exact
    /// range. A [`crate::ast::Span::none`] span is ignored (its `start_line` is 0).
    pub fn set_span(&mut self, span: crate::ast::Span) {
        if span.is_none() {
            return;
        }
        self.line = Some(span.start_line as usize);
        self.col = Some(span.start_col as usize);
        self.end_line = Some(span.end_line as usize);
        self.end_col = Some(span.end_col as usize);
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
        s.push_str(&format!("\"end_line\":{},", json_num_or_null(self.end_line)));
        s.push_str(&format!("\"end_col\":{},", json_num_or_null(self.end_col)));
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
    reclassify(phase, message).1
}

/// Returns true for messages that are file/stream I/O failures (a missing or
/// unreadable source file), which are NOT a compiler phase.
fn is_io_message(m: &str) -> bool {
    m.starts_with("cannot read ")
}

/// Returns true for interface/impl/trait-resolution SEMANTIC messages. These can
/// arrive via the parse path (parser-time trait lowering raises with phase
/// `parse`) but logically belong to the type phase under the trait/bound code
/// (E0206). The markers are deliberately SPECIFIC: a bare mention of the word
/// "interface" is NOT enough, because a structural PARSE error lists `interface`
/// as one of the keywords it expected (e.g. "expected `fn`, `pure`, `type`,
/// `interface`, or `impl`, found ...") — that is a genuine parse error (E0100),
/// not a trait error. We therefore match only phrasings the trait/impl checker
/// itself emits (a quoted `impl`/`interface` head, an explicit declares/method/
/// bound clause), never the keyword in isolation.
fn is_trait_message(m: &str) -> bool {
    m.contains("the interface declares")
        || m.contains("impl `")
        || m.contains("duplicate interface `")
        // A genuine trait-checker complaint about a specific interface METHOD
        // (`interface `Show` method `show`: ...`). Structural PARSE errors about a
        // malformed interface DECLARATION (`interface `Show` must declare ...`)
        // deliberately do NOT match — they stay parse/E0100.
        || (m.contains("interface `") && m.contains(" method `"))
        || m.contains("requires its type parameter")
        || m.contains("is not bounded by")
        || m.contains("trait method")
        || m.contains("trait bound")
}

/// Map a raised `(phase, message)` to the documented `(phase, code)`, letting
/// message content override the raising phase for cross-cutting categories
/// (I/O, trait/interface) that don't belong to the phase that produced them.
fn reclassify(phase: &str, message: &str) -> (&'static str, &'static str) {
    // File I/O: distinct `io` phase + `E0002` (NOT a lex error).
    if is_io_message(message) {
        return ("io", "E0002");
    }
    // Trait / interface / impl: type phase + E0206, even via the parse path.
    if is_trait_message(message) {
        return ("type", "E0206");
    }
    (static_phase(phase), classify_by_phase(phase, message))
}

/// The canonical `phase` string (interned) for a raised phase, used when no
/// content-based reclassification applies.
fn static_phase(phase: &str) -> &'static str {
    match phase {
        "lex" => "lex",
        "parse" => "parse",
        "type" => "type",
        "shape" => "shape",
        "purity" => "purity",
        "exhaustiveness" => "exhaustiveness",
        "io" => "io",
        _ => "type",
    }
}

/// Phase-driven code for the non-reclassified path.
fn classify_by_phase(phase: &str, message: &str) -> &'static str {
    match phase {
        "lex" => "E0001",
        "parse" => "E0100",
        "shape" => "E0300",
        "purity" => "E0210",
        "exhaustiveness" => "E0203",
        "io" => "E0002",
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
    // Arity / constructor-and-record field shape: wrong number of arguments /
    // fields / parameters / type arguments, AND named-field shape errors
    // (missing / duplicate / unknown field) all live in the constructor/record
    // fields bucket (E0202).
    if m.contains("expects") && (m.contains("argument(s)") || m.contains("field(s)"))
        || m.contains("takes") && m.contains("parameter(s)")
        || m.contains("type argument(s)")
        || m.contains("element tuple pattern cannot match")
        || m.contains("missing field `")
        || m.contains("duplicate field `")
        || m.contains("has no field `")
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

    #[test]
    fn record_field_shape_errors_are_e0202() {
        // Record named-field shape errors live in the constructor/record-fields
        // bucket (E0202), NOT the type-mismatch family (E0201).
        for m in [
            "record `Point`: missing field `y`",
            "record `Point`: duplicate field `x`",
            "record `Point` has no field `z`",
            "type `Point` has no field `z`",
            "record update: duplicate field `x`",
        ] {
            let d = Diagnostic::error("type", m.into());
            assert_eq!(d.code, "E0202", "expected E0202 for `{}`, got {}", m, d.code);
            assert_eq!(d.phase, "type");
        }
    }

    #[test]
    fn interface_impl_arity_is_type_phase_e0206_even_via_parse() {
        // The interface/impl method-arity mismatch is RAISED during parser-time
        // trait lowering (phase `parse`), but must be reported as a trait/bound
        // error: phase `type`, code E0206.
        let m = "impl `Show` for `Point`: method `show` takes 2 parameter(s) but the interface declares 1";
        let d = Diagnostic::error("parse", m.into());
        assert_eq!(d.phase, "type", "interface arity should be type phase");
        assert_eq!(d.code, "E0206", "interface arity should be E0206");
    }

    #[test]
    fn parse_error_listing_interface_keyword_is_not_e0206() {
        // A structural PARSE error whose message merely LISTS `interface` as one of
        // the expected keywords (e.g. a file containing just `x`) must stay phase
        // `parse` / E0100 — it is NOT a trait/impl semantic error. (Regression for
        // the `is_trait_message` bare-"interface" false positive.)
        let m = "line 1: expected `fn`, `pure`, `type`, `interface`, or `impl`, found Ident(\"x\")";
        let d = Diagnostic::error("parse", m.into());
        assert_eq!(d.phase, "parse", "must remain a parse error: {}", m);
        assert_eq!(d.code, "E0100", "must remain E0100: {}", m);
    }

    #[test]
    fn malformed_interface_declaration_is_a_parse_error_not_e0206() {
        // A structural complaint about a malformed interface DECLARATION
        // (`interface `Show` must declare ...`) is a parse-phase error, not a
        // trait-resolution semantic error. Only a complaint about a specific
        // interface METHOD (`interface `Show` method `show`: ...`) is E0206.
        let decl = "line 1: interface `Show` must declare exactly one type parameter (the implementing type), e.g. `interface Show[T]`";
        let d = Diagnostic::error("parse", decl.into());
        assert_eq!(d.phase, "parse", "interface decl shape is a parse error");
        assert_eq!(d.code, "E0100");

        // ...whereas a method-level interface complaint IS a trait error.
        let method = "interface `Show` method `show` must take a `self` receiver parameter";
        let d2 = Diagnostic::error("parse", method.into());
        assert_eq!(d2.phase, "type", "interface method error is a trait error");
        assert_eq!(d2.code, "E0206");
    }

    #[test]
    fn file_read_error_is_io_phase_e0002() {
        // A missing/unreadable source file is an I/O failure, not a lex error.
        let d = Diagnostic::error("lex", "cannot read foo.aria: No such file or directory (os error 2)".into());
        assert_eq!(d.phase, "io", "file read error should be io phase");
        assert_eq!(d.code, "E0002", "file read error should be E0002");
    }

    #[test]
    fn declaration_level_errors_have_null_function() {
        // Declaration-level errors legitimately carry `function: null` (the doc's
        // softened wording): they are not scoped to a function body.
        for m in [
            "duplicate type `Point`",
            "duplicate function `f`",
            "cannot redefine built-in `print_int`",
        ] {
            let d = Diagnostic::error("type", m.into());
            assert!(d.function.is_none(), "`{}` should have function: null", m);
        }
    }
}
