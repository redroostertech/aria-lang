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
    Type,
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
    Pipe,
    Underscore,
    Arrow,    // ->
    FatArrow, // =>
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
    pub line: usize,
}

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut line = 1usize;
    let mut out: Vec<Token> = Vec::new();

    let push = |out: &mut Vec<Token>, tok: Tok, line: usize| out.push(Token { tok, line });

    while i < n {
        let c = chars[i];

        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Line comment: -- ... \n
        if c == '-' && i + 1 < n && chars[i + 1] == '-' {
            while i < n && chars[i] != '\n' {
                i += 1;
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
                push(&mut out, t, line);
                i += 2;
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
            _ => None,
        };
        if let Some(t) = single {
            push(&mut out, t, line);
            i += 1;
            continue;
        }

        // String literal.
        if c == '"' {
            i += 1;
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
                } else {
                    if chars[i] == '\n' {
                        return Err(format!("line {}: unterminated string", line));
                    }
                    s.push(chars[i]);
                    i += 1;
                }
            }
            if i >= n {
                return Err(format!("line {}: unterminated string", line));
            }
            i += 1; // closing quote
            push(&mut out, Tok::Str(s), line);
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
            if is_float {
                let f: f64 = text
                    .parse()
                    .map_err(|_| format!("line {}: bad float `{}`", line, text))?;
                push(&mut out, Tok::Float(f), line);
            } else {
                let v: i64 = text
                    .parse()
                    .map_err(|_| format!("line {}: integer `{}` out of range", line, text))?;
                push(&mut out, Tok::Int(v), line);
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
            let tok = match word.as_str() {
                "fn" => Tok::Fn,
                "type" => Tok::Type,
                "let" => Tok::Let,
                "match" => Tok::Match,
                "if" => Tok::If,
                "else" => Tok::Else,
                "true" => Tok::True,
                "false" => Tok::False,
                "_" => Tok::Underscore,
                _ => Tok::Ident(word),
            };
            push(&mut out, tok, line);
            continue;
        }

        return Err(format!("line {}: unexpected character `{}`", line, c));
    }

    push(&mut out, Tok::Eof, line);
    Ok(out)
}
