//! GBNF grammar export for Aria.
//!
//! Aria's AI-native claim is that an LLM doing CONSTRAINED DECODING against the
//! language's grammar literally cannot emit a syntax error. GBNF (the GGML BNF
//! grammar format used by llama.cpp and similar local-LLM stacks) is the
//! standard way to do that. This module emits a complete, well-formed GBNF
//! grammar describing Aria's concrete syntax, mirroring exactly what `lexer.rs`
//! and `parser.rs` accept.
//!
//! The operator-precedence hierarchy is encoded as a layered rule chain so the
//! generated text is unambiguous and well-formed:
//!
//!   expr -> or-expr -> and-expr -> cmp-expr -> add-expr
//!        -> mul-expr -> unary -> app -> atom
//!
//! `ws` (whitespace + `--` line comments) is threaded between tokens. The
//! grammar is real GBNF: usable by llama.cpp, and validated in this module's
//! tests against the actual Aria example programs.

/// The Aria GBNF grammar as a single string, ready to feed to llama.cpp.
pub fn grammar() -> String {
    // Each rule is `name ::= ...`. Whitespace/comments are explicit via `ws`
    // (zero-or-more) so the grammar matches the tokenizer's "no significant
    // whitespace" rule. Token spellings come straight from `lexer.rs`.
    let rules: &[&str] = &[
        // ---- program / items -------------------------------------------
        // A program is a sequence of items surrounded by optional whitespace.
        "root ::= ws item ( ws item )* ws | ws",
        "item ::= fn-decl | type-decl | interface-decl | impl-decl",

        // fn-decl: optional `pure`, `fn`, name, optional (bounded) type params,
        // params, `->` return type, `=`, body expression.
        "fn-decl ::= ( \"pure\" ws )? \"fn\" ws lower-ident ws bound-params? ws \"(\" ws params? ws \")\" ws \"->\" ws type ws \"=\" ws expr",

        // interface-decl: `interface Name[T] { fn m(self: T, ..) -> R, .. }`.
        // Methods are signatures only (no body), comma-separated, trailing OK.
        "interface-decl ::= \"interface\" ws upper-ident ws type-params ws \"{\" ws method-sig ( ws \",\"? ws method-sig )* ( ws \",\" )? ws \"}\"",
        "method-sig ::= \"fn\" ws lower-ident ws \"(\" ws params? ws \")\" ws \"->\" ws type",

        // impl-decl: `impl Trait for Head { fn m(self: Head, ..) -> R = body, .. }`.
        "impl-decl ::= \"impl\" ws upper-ident ws \"for\" ws upper-ident ws \"{\" ws fn-decl ( ws \",\"? ws fn-decl )* ( ws \",\" )? ws \"}\"",

        // Bounded type params: `[T, U: Trait, ..]` — a bound is `: Trait` on a
        // parameter (the trait the impl-resolved methods come from).
        "bound-params ::= \"[\" ws bound-param ( ws \",\" ws bound-param )* ws \"]\"",
        "bound-param ::= ident ( ws \":\" ws upper-ident )?",

        // type-decl: `type Name[params] = | Ctor(..) | ...` with REQUIRED
        // leading pipe before the first variant.
        "type-decl ::= \"type\" ws upper-ident ws type-params? ws \"=\" ws ( record-body | sum-body )",
        "sum-body ::= \"|\" ws variant ( ws \"|\" ws variant )*",
        "record-body ::= \"{\" ws ( field-decl ( ws \",\" ws field-decl )* ws )? \"}\"",
        "field-decl ::= lower-ident ws \":\" ws type",
        "variant ::= upper-ident ( ws \"(\" ws type ( ws \",\" ws type )* ws \")\" )?",

        // type-params: `[T, U, ...]` (identifiers).
        "type-params ::= \"[\" ws ident ( ws \",\" ws ident )* ws \"]\"",

        // params: `name: Type, ...`.
        "params ::= param ( ws \",\" ws param )*",
        "param ::= lower-ident ws \":\" ws type",

        // ---- types ------------------------------------------------------
        // A type is a function type `(T, ...) -> R` or a named/builtin type
        // with optional generic `[args]`.
        "type ::= fn-type | named-type",
        "fn-type ::= \"(\" ws ( type ( ws \",\" ws type )* ws )? \")\" ws \"->\" ws type",
        "named-type ::= ident ( ws \"[\" ws type ( ws \",\" ws type )* ws \"]\" )?",

        // ---- expressions (precedence-layered) ---------------------------
        // expr -> or -> and -> cmp -> add -> mul -> unary -> app -> atom.
        "expr ::= or-expr",
        "or-expr ::= and-expr ( ws \"||\" ws and-expr )*",
        "and-expr ::= cmp-expr ( ws \"&&\" ws cmp-expr )*",
        "cmp-expr ::= add-expr ( ws cmp-op ws add-expr )*",
        "cmp-op ::= \"==\" | \"!=\" | \"<=\" | \">=\" | \"<\" | \">\"",
        "add-expr ::= mul-expr ( ws add-op ws mul-expr )*",
        "add-op ::= \"+\" | \"-\"",
        "mul-expr ::= unary-expr ( ws mul-op ws unary-expr )*",
        "mul-op ::= \"*\" | \"/\" | \"%\"",

        // Unary `-`/`!` (prefix, right-recursive) and lambdas bind tighter than
        // the binary operators but looser than application.
        "unary-expr ::= \"-\" ws unary-expr | \"!\" ws unary-expr | lambda | app-expr",

        // A lambda: `\x -> e` or `\(x: T, ...) -> e`. The body runs to the right.
        "lambda ::= \"\\\\\" ws lambda-params ws \"->\" ws expr",
        "lambda-params ::= \"(\" ws ( param ( ws \",\" ws param )* ws )? \")\" | lower-ident",

        // Application: an atom followed by any number of `(args)` calls,
        // `[index]` array subscripts, or `.field` record accesses.
        "app-expr ::= atom ( ws \"(\" ws args? ws \")\" | ws \"[\" ws expr ws \"]\" | ws \".\" ws lower-ident )*",
        "args ::= expr ( ws \",\" ws expr )*",

        // ---- atoms ------------------------------------------------------
        "atom ::= if-expr | match-expr | update-expr | block | paren-expr | array-lit | record-lit | literal | ctor-expr | call-or-var",

        // Array literal `[e, e, ...]` (possibly empty).
        "array-lit ::= \"[\" ws ( expr ( ws \",\" ws expr )* ws )? \"]\"",

        // Record literal `Name { field: expr, ... }`.
        "record-lit ::= upper-ident ws \"{\" ws ( field-init ( ws \",\" ws field-init )* ws )? \"}\"",
        "field-init ::= lower-ident ws \":\" ws expr",

        // Functional record update `{ base | field = expr, ... }` (a block form).
        "update-expr ::= \"{\" ws expr ws \"|\" ws field-set ( ws \",\" ws field-set )* ws \"}\"",
        "field-set ::= lower-ident ws \"=\" ws expr",

        "paren-expr ::= \"(\" ws expr ws \")\"",

        // `if c { e } else { e }`.
        "if-expr ::= \"if\" ws expr ws block ws \"else\" ws block",

        // `match e { pat => e, ... }` (at least one arm).
        "match-expr ::= \"match\" ws expr ws \"{\" ws arm ( ws \",\" ws arm )* ( ws \",\" )? ws \"}\"",
        "arm ::= pattern ws \"=>\" ws expr",

        // A block: `{ let-stmt* expr? }`. Statements end in `;`; the optional
        // final expression has no trailing `;` (a trailing `;` yields Unit).
        "block ::= \"{\" ws ( stmt ws )* ( expr ws )? \"}\"",
        "stmt ::= let-stmt | expr ws \";\"",
        "let-stmt ::= \"let\" ws lower-ident ws ( \":\" ws type ws )? \"=\" ws expr ws \";\"",

        // Uppercase identifier = constructor (nullary or applied).
        "ctor-expr ::= upper-ident ( ws \"(\" ws args? ws \")\" )?",
        // Lowercase identifier = variable, or a by-name call `name(args)`.
        "call-or-var ::= lower-ident ( ws \"(\" ws args? ws \")\" )?",

        // ---- patterns ---------------------------------------------------
        // `_`, int literal, bool literal, variable, or `Ctor(subpats)`.
        "pattern ::= \"_\" | int | bool | record-pattern | ctor-pattern | lower-ident",
        "ctor-pattern ::= upper-ident ( ws \"(\" ws pattern ( ws \",\" ws pattern )* ws \")\" )?",
        "record-pattern ::= upper-ident ws \"{\" ws ( field-pat ( ws \",\" ws field-pat )* ws )? \"}\"",
        "field-pat ::= lower-ident ( ws \":\" ws pattern )?",

        // ---- literals ---------------------------------------------------
        "literal ::= float | int | string | bool",
        "bool ::= \"true\" | \"false\"",
        // Float requires a fractional part (the lexer rejects `1.`).
        "float ::= digit+ \".\" digit+",
        "int ::= digit+",
        // String with the four supported escapes (\n \t \\ \").
        "string ::= \"\\\"\" str-char* \"\\\"\"",
        "str-char ::= [^\"\\\\] | \"\\\\\" [nt\\\\\"]",

        // ---- identifiers ------------------------------------------------
        // ASCII alnum/underscore; case of the first letter is significant.
        "ident ::= [A-Za-z_] [A-Za-z0-9_]*",
        "lower-ident ::= [a-z_] [A-Za-z0-9_]*",
        "upper-ident ::= [A-Z] [A-Za-z0-9_]*",
        "digit ::= [0-9]",

        // ---- whitespace & comments -------------------------------------
        // Spaces/tabs/newlines and `--` line comments, any amount.
        "ws ::= ( [ \\t\\r\\n] | comment )*",
        "comment ::= \"--\" [^\\n]*",
    ];

    let mut out = String::new();
    for r in rules {
        out.push_str(r);
        out.push('\n');
    }
    out
}

// ===========================================================================
// GBNF acceptor — a small recursive-descent matcher over the SUBSET of GBNF
// that `grammar()` uses. It exists to keep the grammar honest against the real
// parser: tests below feed it the actual Aria example programs (which must be
// accepted) and deliberately malformed snippets (which must be rejected).
//
// Supported GBNF features: rule defs (`name ::= ...`), alternation `|`,
// sequence, string terminals `"..."`, char classes `[...]` and negated
// `[^...]`, repetition `* + ?`, grouping `( ... )`, and rule references.
// ===========================================================================

#[cfg(test)]
mod acceptor {
    use std::collections::HashMap;

    /// A parsed GBNF expression (the right-hand side of a rule, or a fragment).
    #[derive(Debug, Clone)]
    pub enum Node {
        /// Ordered alternation: first matching branch wins (with backtracking).
        Alt(Vec<Node>),
        /// Sequence of nodes matched in order.
        Seq(Vec<Node>),
        /// Literal string terminal.
        Lit(String),
        /// Character class; `neg` flips membership.
        Class { ranges: Vec<(char, char)>, neg: bool },
        /// Reference to another rule by name.
        Ref(String),
        Star(Box<Node>),
        Plus(Box<Node>),
        Opt(Box<Node>),
    }

    pub struct Grammar {
        pub rules: HashMap<String, Node>,
    }

    // ---- GBNF text parser -------------------------------------------------

    struct GParser {
        chars: Vec<char>,
        i: usize,
    }

    impl GParser {
        fn new(s: &str) -> Self {
            GParser { chars: s.chars().collect(), i: 0 }
        }
        fn peek(&self) -> Option<char> {
            self.chars.get(self.i).copied()
        }
        fn bump(&mut self) -> Option<char> {
            let c = self.peek();
            if c.is_some() {
                self.i += 1;
            }
            c
        }
        /// Skip spaces/tabs only (newlines separate rules at the top level, but
        /// within a single rule RHS we keep them on one logical line).
        fn skip_inline_ws(&mut self) {
            while let Some(c) = self.peek() {
                if c == ' ' || c == '\t' {
                    self.i += 1;
                } else {
                    break;
                }
            }
        }

        /// Parse an alternation (lowest precedence) until end-of-input or `)`.
        fn parse_alt(&mut self) -> Node {
            let mut branches = vec![self.parse_seq()];
            loop {
                self.skip_inline_ws();
                if self.peek() == Some('|') {
                    self.bump();
                    branches.push(self.parse_seq());
                } else {
                    break;
                }
            }
            if branches.len() == 1 {
                branches.pop().unwrap()
            } else {
                Node::Alt(branches)
            }
        }

        /// Parse a sequence of postfix terms until `|`, `)`, or end.
        fn parse_seq(&mut self) -> Node {
            let mut items = Vec::new();
            loop {
                self.skip_inline_ws();
                match self.peek() {
                    None | Some('|') | Some(')') => break,
                    _ => {}
                }
                items.push(self.parse_postfix());
            }
            if items.len() == 1 {
                items.pop().unwrap()
            } else {
                Node::Seq(items)
            }
        }

        fn parse_postfix(&mut self) -> Node {
            let mut n = self.parse_primary();
            loop {
                match self.peek() {
                    Some('*') => {
                        self.bump();
                        n = Node::Star(Box::new(n));
                    }
                    Some('+') => {
                        self.bump();
                        n = Node::Plus(Box::new(n));
                    }
                    Some('?') => {
                        self.bump();
                        n = Node::Opt(Box::new(n));
                    }
                    _ => break,
                }
            }
            n
        }

        fn parse_primary(&mut self) -> Node {
            self.skip_inline_ws();
            match self.peek() {
                Some('(') => {
                    self.bump();
                    let inner = self.parse_alt();
                    self.skip_inline_ws();
                    assert_eq!(self.bump(), Some(')'), "unbalanced ( in grammar rule");
                    inner
                }
                Some('"') => self.parse_lit(),
                Some('[') => self.parse_class(),
                Some(c) if c.is_ascii_alphabetic() || c == '_' => self.parse_ref(),
                other => panic!("unexpected char {:?} in GBNF rule", other),
            }
        }

        /// Read a GBNF escape after a backslash, returning the real character.
        fn read_escape(&mut self) -> char {
            match self.bump() {
                Some('n') => '\n',
                Some('t') => '\t',
                Some('r') => '\r',
                Some('\\') => '\\',
                Some('"') => '"',
                Some(']') => ']',
                Some('[') => '[',
                Some(other) => other,
                None => panic!("dangling escape in grammar"),
            }
        }

        fn parse_lit(&mut self) -> Node {
            assert_eq!(self.bump(), Some('"'));
            let mut s = String::new();
            loop {
                match self.bump() {
                    Some('"') => break,
                    Some('\\') => s.push(self.read_escape()),
                    Some(c) => s.push(c),
                    None => panic!("unterminated string literal in grammar"),
                }
            }
            Node::Lit(s)
        }

        fn parse_class(&mut self) -> Node {
            assert_eq!(self.bump(), Some('['));
            let neg = if self.peek() == Some('^') {
                self.bump();
                true
            } else {
                false
            };
            let mut ranges: Vec<(char, char)> = Vec::new();
            // Read members until the closing `]`.
            loop {
                match self.peek() {
                    Some(']') => {
                        self.bump();
                        break;
                    }
                    None => panic!("unterminated char class in grammar"),
                    _ => {}
                }
                let lo = match self.bump().unwrap() {
                    '\\' => self.read_escape(),
                    c => c,
                };
                // Range `a-z`? A `-` not immediately before `]` starts a range.
                if self.peek() == Some('-') && self.chars.get(self.i + 1) != Some(&']') {
                    self.bump(); // consume '-'
                    let hi = match self.bump().unwrap() {
                        '\\' => self.read_escape(),
                        c => c,
                    };
                    ranges.push((lo, hi));
                } else {
                    ranges.push((lo, lo));
                }
            }
            Node::Class { ranges, neg }
        }

        fn parse_ref(&mut self) -> Node {
            let mut name = String::new();
            while let Some(c) = self.peek() {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    name.push(c);
                    self.bump();
                } else {
                    break;
                }
            }
            Node::Ref(name)
        }
    }

    /// Parse a full grammar (one `name ::= rhs` per line; blank lines ignored).
    pub fn parse_grammar(text: &str) -> Grammar {
        let mut rules = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let sep = line.find("::=").expect("rule missing ::=");
            let name = line[..sep].trim().to_string();
            let rhs = &line[sep + 3..];
            let mut p = GParser::new(rhs);
            let node = p.parse_alt();
            p.skip_inline_ws();
            assert!(p.peek().is_none(), "trailing junk in rule `{}`", name);
            rules.insert(name, node);
        }
        Grammar { rules }
    }

    // ---- matcher ----------------------------------------------------------
    //
    // The matcher is a general CFG recognizer that computes, for a node at a
    // start position, the SET of end positions it can reach. Returning a set
    // (rather than threading a continuation) keeps recursion depth bounded by
    // the grammar's nesting depth instead of the input length, and memoizing
    // rule references on `(rule, pos)` makes the whole thing polynomial and
    // immune to the exponential backtracking the precedence layers would
    // otherwise cause. `accepts` succeeds iff `root` can end exactly at EOF.

    use std::cell::RefCell;
    use std::collections::BTreeSet;

    impl Grammar {
        /// Collect every rule name referenced anywhere (for dangling-ref check).
        pub fn referenced(&self) -> Vec<String> {
            let mut out = Vec::new();
            for node in self.rules.values() {
                collect_refs(node, &mut out);
            }
            out
        }

        /// Does the grammar accept the WHOLE of `input` starting from `root`?
        pub fn accepts(&self, input: &str) -> bool {
            let chars: Vec<char> = input.chars().collect();
            let root = self.rules.get("root").expect("no root rule");
            let memo: RefCell<HashMap<(String, usize), BTreeSet<usize>>> =
                RefCell::new(HashMap::new());
            let ends = self.ends_of(root, &chars, 0, &memo);
            ends.contains(&chars.len())
        }

        /// All positions at which `node`, starting at `pos`, can finish.
        fn ends_of(
            &self,
            node: &Node,
            chars: &[char],
            pos: usize,
            memo: &RefCell<HashMap<(String, usize), BTreeSet<usize>>>,
        ) -> BTreeSet<usize> {
            let mut out = BTreeSet::new();
            match node {
                Node::Lit(s) => {
                    let lit: Vec<char> = s.chars().collect();
                    if pos + lit.len() <= chars.len() && chars[pos..pos + lit.len()] == lit[..] {
                        out.insert(pos + lit.len());
                    }
                }
                Node::Class { ranges, neg } => {
                    if pos < chars.len() {
                        let c = chars[pos];
                        let inside = ranges.iter().any(|(lo, hi)| c >= *lo && c <= *hi);
                        if inside != *neg {
                            out.insert(pos + 1);
                        }
                    }
                }
                Node::Ref(name) => {
                    let key = (name.clone(), pos);
                    if let Some(cached) = memo.borrow().get(&key) {
                        return cached.clone();
                    }
                    // Mark in-progress as empty to break any reference cycle.
                    memo.borrow_mut().insert(key.clone(), BTreeSet::new());
                    let r = self
                        .rules
                        .get(name)
                        .unwrap_or_else(|| panic!("undefined rule `{}`", name));
                    let ends = self.ends_of(r, chars, pos, memo);
                    memo.borrow_mut().insert(key, ends.clone());
                    return ends;
                }
                Node::Alt(branches) => {
                    for b in branches {
                        out.extend(self.ends_of(b, chars, pos, memo));
                    }
                }
                Node::Seq(items) => {
                    let mut frontier = BTreeSet::new();
                    frontier.insert(pos);
                    for it in items {
                        let mut next = BTreeSet::new();
                        for &p in &frontier {
                            next.extend(self.ends_of(it, chars, p, memo));
                        }
                        frontier = next;
                        if frontier.is_empty() {
                            break;
                        }
                    }
                    out = frontier;
                }
                Node::Opt(inner) => {
                    out.insert(pos);
                    out.extend(self.ends_of(inner, chars, pos, memo));
                }
                Node::Star(inner) => {
                    out = self.repeat(inner, chars, pos, false, memo);
                }
                Node::Plus(inner) => {
                    out = self.repeat(inner, chars, pos, true, memo);
                }
            }
            out
        }

        /// Reachable end positions for `inner*` (or `inner+` when `one_plus`).
        fn repeat(
            &self,
            inner: &Node,
            chars: &[char],
            pos: usize,
            one_plus: bool,
            memo: &RefCell<HashMap<(String, usize), BTreeSet<usize>>>,
        ) -> BTreeSet<usize> {
            let mut reached: BTreeSet<usize> = BTreeSet::new();
            if !one_plus {
                reached.insert(pos); // zero repetitions
            }
            let mut frontier: BTreeSet<usize> = BTreeSet::new();
            frontier.insert(pos);
            // Expand repetitions breadth-first; each new end position is visited
            // once (guarded by `reached`), so this terminates.
            while !frontier.is_empty() {
                let mut next = BTreeSet::new();
                for &p in &frontier {
                    for e in self.ends_of(inner, chars, p, memo) {
                        if e != p && reached.insert(e) {
                            next.insert(e);
                        }
                    }
                }
                frontier = next;
            }
            reached
        }
    }

    fn collect_refs(node: &Node, out: &mut Vec<String>) {
        match node {
            Node::Ref(n) => out.push(n.clone()),
            Node::Seq(xs) | Node::Alt(xs) => {
                for x in xs {
                    collect_refs(x, out);
                }
            }
            Node::Star(x) | Node::Plus(x) | Node::Opt(x) => collect_refs(x, out),
            Node::Lit(_) | Node::Class { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::acceptor::parse_grammar;
    use super::grammar;
    use std::collections::HashSet;

    fn g() -> super::acceptor::Grammar {
        parse_grammar(&grammar())
    }

    #[test]
    fn well_formed_no_dangling_refs() {
        let gram = g();
        let defined: HashSet<&String> = gram.rules.keys().collect();
        assert!(defined.contains(&"root".to_string()), "root rule must exist");
        for r in gram.referenced() {
            assert!(
                defined.contains(&r),
                "grammar references undefined rule `{}`",
                r
            );
        }
    }

    #[test]
    fn accepts_simple_programs() {
        let gram = g();
        assert!(gram.accepts("fn main() -> Int = 0\n"));
        assert!(gram.accepts("pure fn add(a: Int, b: Int) -> Int = a + b\n"));
        assert!(gram.accepts("type Bool2 =\n  | T\n  | F\n"));
        assert!(gram.accepts("fn id[T](x: T) -> T = x\n"));
        // Traits: an interface, an impl, and a bounded generic function.
        assert!(gram.accepts("interface Show[T] { fn show(self: T) -> Int }\n"));
        assert!(gram.accepts(
            "impl Show for Point { fn show(self: Point) -> Int = 1 }\n"
        ));
        assert!(gram.accepts("fn all[T: Show](x: T) -> Int = show(x)\n"));
        assert!(gram.accepts("fn f() -> Int = if true { 1 } else { 2 }\n"));
        assert!(gram.accepts(
            "fn f(x: Int) -> Int = match x { 0 => 1, _ => x, }\n"
        ));
        assert!(gram.accepts("fn f() -> Int = { let x = 1; x }\n"));
        assert!(gram.accepts("fn f() -> (Int) -> Int = \\x -> x + 1\n"));
        assert!(gram.accepts(
            "fn f() -> Int = (\\(a: Int, b: Int) -> a + b)(1, 2)\n"
        ));
        assert!(gram.accepts("fn f() -> Int = 1 + 2 * 3 - 4 % 5\n"));
        assert!(gram.accepts("fn f() -> Bool = !true && 1 < 2 || false\n"));
        assert!(gram.accepts("fn f() -> String = \"hi\\n\"\n"));
        assert!(gram.accepts("fn f() -> Float = 3.14\n"));
        assert!(gram.accepts("-- comment\nfn f() -> Int = -5\n"));
    }

    #[test]
    fn accepts_real_examples() {
        let gram = g();
        for name in ["intro", "list", "generic", "hof", "pure", "mem_bench", "array", "record", "trait", "bytes", "map", "set", "shape", "vector", "iter"] {
            let path = format!(
                "{}/examples/{}.aria",
                env!("CARGO_MANIFEST_DIR"),
                name
            );
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
            assert!(
                gram.accepts(&src),
                "grammar rejected real example `{}.aria`",
                name
            );
        }
    }

    #[test]
    fn rejects_malformed() {
        let gram = g();
        // Missing closing paren.
        assert!(!gram.accepts("fn f() -> Int = g(1, 2\n"));
        // fn without `=`/body.
        assert!(!gram.accepts("fn f() -> Int\n"));
        // type decl missing the required leading `|`.
        assert!(!gram.accepts("type T = A | B\n"));
        // Unclosed lambda (no body after `->`).
        assert!(!gram.accepts("fn f() -> Int = \\x ->\n"));
        // Garbage.
        assert!(!gram.accepts("fn 123 = ?\n"));
        // Empty input is not a valid program-with-an-item... but the grammar
        // does allow an empty (all-whitespace) program, so check a true error:
        assert!(!gram.accepts("fn\n"));
    }

    #[test]
    fn grammar_string_is_nonempty_and_has_root() {
        let s = grammar();
        assert!(s.contains("root ::="));
        assert!(s.contains("lower-ident ::="));
        assert!(s.contains("ws ::="));
    }
}
