//! Hand-written lexer for Aria.
//!
//! The token set is intentionally tiny and unambiguous. Comments start with
//! `--` and run to end of line. There is no significant whitespace.

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    // Literals
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),
    // Keywords
    Fn,
    Pure,
    Type,
    Interface,
    Impl,
    Let,
    Match,
    If,
    Else,
    True,
    False,
    // Punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    Dot, // .  (record field access)
    Pipe,
    Underscore,
    Arrow,    // ->
    FatArrow, // =>
    Backslash, // \  (introduces a lambda)
    Eq,       // =
    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    Bang,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    /// 1-based source line of the token's first character.
    pub line: usize,
    /// 1-based source column (in Unicode scalar values from the line start) of
    /// the token's first character.
    pub col: usize,
    /// 1-based source line of the position ONE PAST the token's last character.
    /// For a token that does not span a newline this equals `line`.
    pub end_line: usize,
    /// 1-based source column ONE PAST the token's last character. For a
    /// single-character token at `col` this is `col + 1`, giving a half-open
    /// `[col, end_col)` extent.
    pub end_col: usize,
}

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut line = 1usize;
    // 1-based column of `chars[i]`. Advanced in lockstep with `i` and reset to 1
    // after each newline, so every token can record a precise start/end column.
    let mut col = 1usize;
    let mut out: Vec<Token> = Vec::new();

    // Emit a token spanning [(sl, sc) .. (el, ec)) where the end is the current
    // `(line, col)` position (one past the last consumed character).
    let push = |out: &mut Vec<Token>, tok: Tok, sl: usize, sc: usize, el: usize, ec: usize| {
        out.push(Token { tok, line: sl, col: sc, end_line: el, end_col: ec })
    };

    while i < n {
        let c = chars[i];

        if c == '\n' {
            line += 1;
            col = 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            col += 1;
            continue;
        }

        // The token starts here; remember its 1-based start column.
        let start_col = col;

        // Line comment: -- ... \n
        if c == '-' && i + 1 < n && chars[i + 1] == '-' {
            while i < n && chars[i] != '\n' {
                i += 1;
                col += 1;
            }
            continue;
        }

        // Two-character operators.
        if i + 1 < n {
            let two = (c, chars[i + 1]);
            let two_tok = match two {
                ('-', '>') => Some(Tok::Arrow),
                ('=', '>') => Some(Tok::FatArrow),
                ('=', '=') => Some(Tok::EqEq),
                ('!', '=') => Some(Tok::NotEq),
                ('<', '=') => Some(Tok::Le),
                ('>', '=') => Some(Tok::Ge),
                ('&', '&') => Some(Tok::AndAnd),
                ('|', '|') => Some(Tok::OrOr),
                _ => None,
            };
            if let Some(t) = two_tok {
                i += 2;
                col += 2;
                push(&mut out, t, line, start_col, line, col);
                continue;
            }
        }

        // Single-character tokens.
        let single = match c {
            '(' => Some(Tok::LParen),
            ')' => Some(Tok::RParen),
            '{' => Some(Tok::LBrace),
            '}' => Some(Tok::RBrace),
            '[' => Some(Tok::LBracket),
            ']' => Some(Tok::RBracket),
            ',' => Some(Tok::Comma),
            ';' => Some(Tok::Semi),
            ':' => Some(Tok::Colon),
            // A bare `.` (the number scanner has already consumed any `.` that is
            // part of a float literal, and rejects a trailing one) is field access.
            '.' => Some(Tok::Dot),
            '|' => Some(Tok::Pipe),
            '=' => Some(Tok::Eq),
            '+' => Some(Tok::Plus),
            '-' => Some(Tok::Minus),
            '*' => Some(Tok::Star),
            '/' => Some(Tok::Slash),
            '%' => Some(Tok::Percent),
            '<' => Some(Tok::Lt),
            '>' => Some(Tok::Gt),
            '!' => Some(Tok::Bang),
            '\\' => Some(Tok::Backslash),
            _ => None,
        };
        if let Some(t) = single {
            i += 1;
            col += 1;
            push(&mut out, t, line, start_col, line, col);
            continue;
        }

        // String literal.
        if c == '"' {
            i += 1;
            col += 1; // opening quote
            let mut s = String::new();
            while i < n && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < n {
                    let esc = chars[i + 1];
                    let r = match esc {
                        'n' => '\n',
                        't' => '\t',
                        '\\' => '\\',
                        '"' => '"',
                        other => return Err(format!("line {}: invalid escape \\{}", line, other)),
                    };
                    s.push(r);
                    i += 2;
                    col += 2;
                } else {
                    if chars[i] == '\n' {
                        return Err(format!("line {}: unterminated string", line));
                    }
                    s.push(chars[i]);
                    i += 1;
                    col += 1;
                }
            }
            if i >= n {
                return Err(format!("line {}: unterminated string", line));
            }
            i += 1; // closing quote
            col += 1;
            push(&mut out, Tok::Str(s), line, start_col, line, col);
            continue;
        }

        // Number literal (Int or Float).
        if c.is_ascii_digit() {
            let start = i;
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            let mut is_float = false;
            if i + 1 < n && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                is_float = true;
                i += 1; // dot
                while i < n && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            // Reject a letter/underscore/dot glued to the number (e.g. `1e10`,
            // `0x1F`, `1_000`, `1.`) instead of silently splitting into two
            // tokens. Exponents/hex/separators are not part of the grammar.
            if i < n && (chars[i].is_alphabetic() || chars[i] == '_' || chars[i] == '.') {
                let bad: String = chars[start..=i].iter().collect();
                return Err(format!("line {}: malformed number literal `{}`", line, bad));
            }
            let text: String = chars[start..i].iter().collect();
            // The number scanned only ASCII digits/`.`, so one char == one column.
            col += i - start;
            if is_float {
                let f: f64 = text
                    .parse()
                    .map_err(|_| format!("line {}: bad float `{}`", line, text))?;
                push(&mut out, Tok::Float(f), line, start_col, line, col);
            } else {
                let v: i64 = text
                    .parse()
                    .map_err(|_| format!("line {}: integer `{}` out of range", line, text))?;
                push(&mut out, Tok::Int(v), line, start_col, line, col);
            }
            continue;
        }

        // Identifier or keyword. Restricted to ASCII so the case-based rule
        // (Uppercase = constructor/type, lowercase = value/function) is always
        // well-defined; caseless scripts would otherwise be silently mis-classed.
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            // Identifiers/keywords are ASCII, so one char == one column.
            col += i - start;
            let tok = match word.as_str() {
                "fn" => Tok::Fn,
                "pure" => Tok::Pure,
                "type" => Tok::Type,
                "interface" => Tok::Interface,
                "impl" => Tok::Impl,
                "let" => Tok::Let,
                "match" => Tok::Match,
                "if" => Tok::If,
                "else" => Tok::Else,
                "true" => Tok::True,
                "false" => Tok::False,
                "_" => Tok::Underscore,
                _ => Tok::Ident(word),
            };
            push(&mut out, tok, line, start_col, line, col);
            continue;
        }

        return Err(format!("line {}: unexpected character `{}`", line, c));
    }

    // The synthetic EOF sits one past the last character (a zero-width span).
    push(&mut out, Tok::Eof, line, col, line, col);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_carry_precise_line_and_column() {
        // `  fn f` — two leading spaces, so `fn` starts at column 3 and `f` at 6.
        let toks = lex("  fn f").expect("lex");
        // [Fn, Ident("f"), Eof]
        assert_eq!(toks[0].tok, Tok::Fn);
        assert_eq!((toks[0].line, toks[0].col), (1, 3));
        // `fn` is two chars wide: [3, 5).
        assert_eq!((toks[0].end_line, toks[0].end_col), (1, 5));
        assert_eq!(toks[1].tok, Tok::Ident("f".to_string()));
        assert_eq!((toks[1].line, toks[1].col), (1, 6));
        assert_eq!((toks[1].end_line, toks[1].end_col), (1, 7));
    }

    #[test]
    fn columns_reset_each_line_and_track_multiline() {
        // Line 1 holds `x = 1`; line 2 holds `  yy`. Confirm the column resets and
        // that a token on line 2 reports line 2 with its own column.
        let toks = lex("x = 1\n  yy").expect("lex");
        // Find the `yy` identifier.
        let yy = toks
            .iter()
            .find(|t| t.tok == Tok::Ident("yy".to_string()))
            .expect("yy token");
        assert_eq!((yy.line, yy.col), (2, 3));
        assert_eq!((yy.end_line, yy.end_col), (2, 5));
        // The `1` on line 1 is at column 5.
        let one = toks.iter().find(|t| t.tok == Tok::Int(1)).expect("int token");
        assert_eq!((one.line, one.col), (1, 5));
        assert_eq!((one.end_line, one.end_col), (1, 6));
    }
}
