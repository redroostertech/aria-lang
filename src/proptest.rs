//! Exploratory, deterministic property-based / differential test suite.
//!
//! Everything here is test-only (`#[cfg(test)] mod proptest;`). It uses only
//! the Rust standard library: randomness comes from a tiny seeded LCG, never an
//! external crate.
//!
//! The keystone is a *type-directed* random program generator that emits Aria
//! source text that is well-typed by construction. Two independent backends —
//! the tree-walking interpreter (`interp`, the oracle) and the IR + RC pipeline
//! (`ir::lower_program` -> `rc::insert_rc` -> `ir::IrInterp`) — are run on each
//! program and required to agree (Ok/Err shape, and on the rendered value), with
//! the IR run additionally required to be garbage-free.

#![cfg(test)]

use crate::{interp, ir, lexer, parser, rc, typeck, wasm};

// ---------------------------------------------------------------------------
// Tiny deterministic PRNG (a 64-bit linear congruential generator).
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // Avoid a degenerate zero state.
        Lcg { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    /// Knuth's MMIX LCG constants.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Mix high bits down (raw LCG low bits are weak).
        let x = self.state;
        x ^ (x >> 31)
    }

    /// Uniform in `[0, n)` for `n > 0`.
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn choice(&mut self, n: usize) -> usize {
        (self.below(n as u64)) as usize
    }
}

// ---------------------------------------------------------------------------
// Type-directed program generator.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Int,
    Bool,
    Float,
    Str,
    List,
    /// `Array[Int]`. The generator only ever builds NON-empty arrays, so
    /// index-0 `get`/`set` never trap and the garbage-free check stays active.
    Array,
    /// `(Int, Int)` — exercises tuple construction, the RC'd tuple cell, and
    /// destructuring patterns. (Tuples work in every backend.)
    Tuple,
}

/// The fixed prelude prepended to every generated program. Only these functions
/// recurse, and only over finite generated lists, so every program terminates.
const PRELUDE: &str = "\
type IntList = | Nil | Cons(Int, IntList)
fn sumL(xs: IntList) -> Int = match xs { Nil => 0, Cons(h, r) => h + sumL(r), }
fn lenL(xs: IntList) -> Int = match xs { Nil => 0, Cons(h, r) => 1 + lenL(r), }
fn incL(xs: IntList) -> IntList = match xs { Nil => Nil, Cons(h, r) => Cons(h + 1, incL(r)), }
";

struct Gen<'a> {
    rng: &'a mut Lcg,
    /// In-scope variables: (name, type).
    scope: Vec<(String, Ty)>,
    /// Monotonic counter for fresh variable names.
    fresh: usize,
    /// When true, the generator stays within the wasm 2a/2b subset: only
    /// Int/Bool/IntList types are introduced (no Float/String). Used by the
    /// compiled-backend differential fuzz so most seeds actually reach codegen.
    wasm_subset: bool,
}

impl<'a> Gen<'a> {
    fn new(rng: &'a mut Lcg) -> Self {
        Gen { rng, scope: Vec::new(), fresh: 0, wasm_subset: false }
    }

    fn new_wasm(rng: &'a mut Lcg) -> Self {
        Gen { rng, scope: Vec::new(), fresh: 0, wasm_subset: true }
    }

    fn fresh_name(&mut self) -> String {
        let n = self.fresh;
        self.fresh += 1;
        format!("v{}", n)
    }

    /// Pick a random in-scope variable of the requested type, if any.
    fn var_of(&mut self, ty: Ty) -> Option<String> {
        let names: Vec<&String> =
            self.scope.iter().filter(|(_, t)| *t == ty).map(|(n, _)| n).collect();
        if names.is_empty() {
            None
        } else {
            let i = self.rng.choice(names.len());
            Some(names[i].clone())
        }
    }

    /// Generate an expression of the given type. `fuel` bounds recursion depth;
    /// at zero fuel only atomic (leaf) productions are used so generation always
    /// terminates with small, finite programs.
    fn expr(&mut self, ty: Ty, fuel: u32) -> String {
        if fuel == 0 {
            return self.leaf(ty);
        }
        match ty {
            Ty::Int => self.gen_int(fuel),
            Ty::Bool => self.gen_bool(fuel),
            Ty::Float => self.gen_float(fuel),
            Ty::Str => self.gen_str(fuel),
            Ty::List => self.gen_list(fuel),
            Ty::Array => self.gen_array(fuel),
            Ty::Tuple => self.gen_tuple(fuel),
        }
    }

    /// Smallest closed expression of a type (used at zero fuel / as a base case).
    fn leaf(&mut self, ty: Ty) -> String {
        match ty {
            Ty::Int => {
                if let Some(v) = self.var_of(Ty::Int) {
                    v
                } else {
                    format!("{}", self.rng.below(10))
                }
            }
            Ty::Bool => {
                if let Some(v) = self.var_of(Ty::Bool) {
                    v
                } else if self.rng.below(2) == 0 {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            Ty::Float => {
                if let Some(v) = self.var_of(Ty::Float) {
                    v
                } else {
                    // Always one decimal place so it parses as a float literal.
                    format!("{}.{}", self.rng.below(10), self.rng.below(10))
                }
            }
            Ty::Str => {
                if let Some(v) = self.var_of(Ty::Str) {
                    v
                } else {
                    let words = ["a", "bc", "hi", "", "z9"];
                    format!("\"{}\"", words[self.rng.choice(words.len())])
                }
            }
            Ty::List => {
                if let Some(v) = self.var_of(Ty::List) {
                    v
                } else {
                    "Nil".to_string()
                }
            }
            Ty::Array => {
                if let Some(v) = self.var_of(Ty::Array) {
                    v
                } else {
                    // A non-empty single-element array literal.
                    format!("[{}]", self.rng.below(10))
                }
            }
            Ty::Tuple => {
                if let Some(v) = self.var_of(Ty::Tuple) {
                    v
                } else {
                    format!("({}, {})", self.rng.below(10), self.rng.below(10))
                }
            }
        }
    }

    /// Generate inside a fresh `let`-block: `{ let <fresh> = <T>; <body:ty> }`.
    /// The bound variable is in scope only for the body.
    fn let_block(&mut self, ty: Ty, fuel: u32) -> String {
        let bind_ty = self.random_ty();
        let rhs = self.expr(bind_ty, fuel - 1);
        let name = self.fresh_name();
        self.scope.push((name.clone(), bind_ty));
        let body = self.expr(ty, fuel - 1);
        self.scope.pop();
        format!("{{ let {} = {}; {} }}", name, rhs, body)
    }

    fn random_ty(&mut self) -> Ty {
        if self.wasm_subset {
            // Restricted universe: Int/Bool/IntList/String AND Array[Int]. No
            // Float, which stays outside the wasm backend's compilable subset.
            return match self.rng.below(6) {
                0 => Ty::Int,
                1 => Ty::Bool,
                2 => Ty::Str,
                3 => Ty::List,
                4 => Ty::Array,
                _ => Ty::Tuple,
            };
        }
        match self.rng.below(7) {
            0 => Ty::Int,
            1 => Ty::Bool,
            2 => Ty::Float,
            3 => Ty::Str,
            4 => Ty::List,
            5 => Ty::Array,
            _ => Ty::Tuple,
        }
    }

    fn gen_int(&mut self, fuel: u32) -> String {
        // Weighted toward leaves to keep programs small; `below` picks a rule.
        // Arrays are now supported by every backend, so the array consumers run
        // in both the full and wasm-subset generators.
        match self.rng.below(12) {
            0 | 1 => self.leaf(Ty::Int),
            2 => {
                let op = ["+", "-", "*"][self.rng.choice(3)];
                format!("({} {} {})", self.expr(Ty::Int, fuel - 1), op, self.expr(Ty::Int, fuel - 1))
            }
            3 => format!(
                "if {} {{ {} }} else {{ {} }}",
                self.expr(Ty::Bool, fuel - 1),
                self.expr(Ty::Int, fuel - 1),
                self.expr(Ty::Int, fuel - 1)
            ),
            4 => format!("sumL({})", self.expr(Ty::List, fuel - 1)),
            5 => format!("lenL({})", self.expr(Ty::List, fuel - 1)),
            6 | 7 => {
                // match over an IntList; the Cons arm binds h:Int, r:IntList.
                let scrut = self.expr(Ty::List, fuel - 1);
                let nil_arm = self.expr(Ty::Int, fuel - 1);
                let h = self.fresh_name();
                let r = self.fresh_name();
                self.scope.push((h.clone(), Ty::Int));
                self.scope.push((r.clone(), Ty::List));
                let cons_arm = self.expr(Ty::Int, fuel - 1);
                self.scope.pop();
                self.scope.pop();
                format!(
                    "match {} {{ Nil => {}, Cons({}, {}) => {}, }}",
                    scrut, nil_arm, h, r, cons_arm
                )
            }
            // Array consumers. Generated arrays are always non-empty, so the
            // index-0 `get` never traps.
            8 => format!("array_len({})", self.expr(Ty::Array, fuel - 1)),
            9 => format!("array_get({}, 0)", self.expr(Ty::Array, fuel - 1)),
            10 => {
                // Destructure a tuple: `match <tuple> { (a, b) => <int> }`.
                let scrut = self.expr(Ty::Tuple, fuel - 1);
                let a = self.fresh_name();
                let b = self.fresh_name();
                self.scope.push((a.clone(), Ty::Int));
                self.scope.push((b.clone(), Ty::Int));
                let body = self.expr(Ty::Int, fuel - 1);
                self.scope.pop();
                self.scope.pop();
                format!("match {} {{ ({}, {}) => {}, }}", scrut, a, b, body)
            }
            _ => self.let_block(Ty::Int, fuel),
        }
    }

    fn gen_bool(&mut self, fuel: u32) -> String {
        match self.rng.below(9) {
            0 | 1 => self.leaf(Ty::Bool),
            2 => {
                let op = ["<", "<=", ">", ">=", "==", "!="][self.rng.choice(6)];
                format!("({} {} {})", self.expr(Ty::Int, fuel - 1), op, self.expr(Ty::Int, fuel - 1))
            }
            8 => {
                // String structural equality (== / !=): exercises the wasm
                // backend's `__streq` against the interpreter's `values_equal`.
                let op = ["==", "!="][self.rng.choice(2)];
                format!("({} {} {})", self.expr(Ty::Str, fuel - 1), op, self.expr(Ty::Str, fuel - 1))
            }
            7 => {
                // Structural ADT equality (== / !=) on two generated IntLists:
                // exercises the wasm backend's recursive `__eq` helper against
                // the interpreter's `values_equal` (and that the compared cells
                // end garbage-free).
                let op = ["==", "!="][self.rng.choice(2)];
                format!(
                    "({} {} {})",
                    self.expr(Ty::List, fuel - 1),
                    op,
                    self.expr(Ty::List, fuel - 1)
                )
            }
            3 => format!("!{}", self.paren_bool(fuel - 1)),
            4 | 5 => {
                let op = ["&&", "||"][self.rng.choice(2)];
                format!(
                    "({} {} {})",
                    self.expr(Ty::Bool, fuel - 1),
                    op,
                    self.expr(Ty::Bool, fuel - 1)
                )
            }
            6 => format!(
                "if {} {{ {} }} else {{ {} }}",
                self.expr(Ty::Bool, fuel - 1),
                self.expr(Ty::Bool, fuel - 1),
                self.expr(Ty::Bool, fuel - 1)
            ),
            _ => self.let_block(Ty::Bool, fuel),
        }
    }

    /// A bool expression that is safe to put directly after `!` (parenthesized
    /// or atomic), avoiding any precedence surprises.
    fn paren_bool(&mut self, fuel: u32) -> String {
        if fuel == 0 || self.rng.below(2) == 0 {
            self.leaf(Ty::Bool)
        } else {
            format!("({})", self.expr(Ty::Bool, fuel))
        }
    }

    fn gen_float(&mut self, fuel: u32) -> String {
        match self.rng.below(6) {
            0 | 1 | 2 => self.leaf(Ty::Float),
            3 => {
                let op = ["+", "-", "*"][self.rng.choice(3)];
                format!(
                    "({} {} {})",
                    self.expr(Ty::Float, fuel - 1),
                    op,
                    self.expr(Ty::Float, fuel - 1)
                )
            }
            4 => format!("-{}", self.leaf(Ty::Float)),
            _ => self.let_block(Ty::Float, fuel),
        }
    }

    fn gen_str(&mut self, fuel: u32) -> String {
        match self.rng.below(6) {
            0 | 1 | 2 => self.leaf(Ty::Str),
            3 => format!(
                "concat({}, {})",
                self.expr(Ty::Str, fuel - 1),
                self.expr(Ty::Str, fuel - 1)
            ),
            4 => format!("int_to_str({})", self.expr(Ty::Int, fuel - 1)),
            _ => self.let_block(Ty::Str, fuel),
        }
    }

    fn gen_list(&mut self, fuel: u32) -> String {
        match self.rng.below(7) {
            0 | 1 => self.leaf(Ty::List),
            2 | 3 => format!(
                "Cons({}, {})",
                self.expr(Ty::Int, fuel - 1),
                self.expr(Ty::List, fuel - 1)
            ),
            4 => format!("incL({})", self.expr(Ty::List, fuel - 1)),
            5 => format!(
                "if {} {{ {} }} else {{ {} }}",
                self.expr(Ty::Bool, fuel - 1),
                self.expr(Ty::List, fuel - 1),
                self.expr(Ty::List, fuel - 1)
            ),
            _ => self.let_block(Ty::List, fuel),
        }
    }

    /// Generate an `Array[Int]` expression. Every rule yields a NON-empty array
    /// (so index-0 `get`/`set` never trap), exercising literals, FBIP `push`/
    /// `set`, branching, and binding.
    fn gen_array(&mut self, fuel: u32) -> String {
        match self.rng.below(7) {
            0 | 1 => self.leaf(Ty::Array),
            2 => {
                // A 1-3 element literal (always non-empty).
                let n = 1 + self.rng.below(3);
                let elems: Vec<String> =
                    (0..n).map(|_| self.expr(Ty::Int, fuel - 1)).collect();
                format!("[{}]", elems.join(", "))
            }
            3 => format!(
                "array_push({}, {})",
                self.expr(Ty::Array, fuel - 1),
                self.expr(Ty::Int, fuel - 1)
            ),
            4 => format!(
                "array_set({}, 0, {})",
                self.expr(Ty::Array, fuel - 1),
                self.expr(Ty::Int, fuel - 1)
            ),
            5 => format!(
                "if {} {{ {} }} else {{ {} }}",
                self.expr(Ty::Bool, fuel - 1),
                self.expr(Ty::Array, fuel - 1),
                self.expr(Ty::Array, fuel - 1)
            ),
            _ => self.let_block(Ty::Array, fuel),
        }
    }

    /// Generate an `(Int, Int)` tuple expression: literal, branch, or binding.
    fn gen_tuple(&mut self, fuel: u32) -> String {
        match self.rng.below(5) {
            0 | 1 | 2 => format!(
                "({}, {})",
                self.expr(Ty::Int, fuel - 1),
                self.expr(Ty::Int, fuel - 1)
            ),
            3 => format!(
                "if {} {{ {} }} else {{ {} }}",
                self.expr(Ty::Bool, fuel - 1),
                self.expr(Ty::Tuple, fuel - 1),
                self.expr(Ty::Tuple, fuel - 1)
            ),
            _ => self.let_block(Ty::Tuple, fuel),
        }
    }
}

/// Build a full, well-typed-by-construction program for a given seed. `main`
/// always returns `Int` (chosen because the differential value comparison is
/// then a plain integer string, but the body exercises every type internally).
fn gen_program(seed: u64) -> String {
    let mut rng = Lcg::new(seed);
    let mut g = Gen::new(&mut rng);
    let body = g.expr(Ty::Int, 4);
    format!("{}fn main() -> Int = {}\n", PRELUDE, body)
}

/// Like [`gen_program`] but restricted to the wasm backend's compilable subset
/// (Int/Bool/IntList only — no Float/String), so a large fraction of seeds
/// actually reach codegen instead of being rejected as out-of-subset.
fn gen_program_wasm(seed: u64) -> String {
    let mut rng = Lcg::new(seed);
    let mut g = Gen::new_wasm(&mut rng);
    let body = g.expr(Ty::Int, 4);
    format!("{}fn main() -> Int = {}\n", PRELUDE, body)
}

// ---------------------------------------------------------------------------
// Pipeline helpers.
// ---------------------------------------------------------------------------

/// Run a program through the tree-walking interpreter (the oracle).
fn ast_run(src: &str) -> Result<String, String> {
    let toks = lexer::lex(src)?;
    let prog = parser::parse(toks)?;
    let it = interp::Interp::new(&prog)?;
    it.run_main().map(|v| v.display())
}

/// Run a program through the IR + RC pipeline, returning the rendered value and
/// the number of cells still live at the end (0 == garbage-free).
fn ir_run_rc(src: &str) -> Result<(String, usize), String> {
    let toks = lexer::lex(src)?;
    let prog = parser::parse(toks)?;
    let fns = rc::insert_rc(&ir::lower_program(&prog)?);
    let mut runner = ir::IrInterp::new(fns);
    let v = runner.run_main()?;
    Ok((runner.render(&v), runner.metrics.live))
}

/// Run a program through the IR pipeline WITHOUT rc insertion.
fn ir_run_no_rc(src: &str) -> Result<String, String> {
    let toks = lexer::lex(src)?;
    let prog = parser::parse(toks)?;
    let fns = ir::lower_program(&prog)?;
    let mut runner = ir::IrInterp::new_no_rc(fns);
    let v = runner.run_main()?;
    Ok(runner.render(&v))
}

/// True iff the program type-checks (and lexes/parses).
fn well_typed(src: &str) -> bool {
    match lexer::lex(src).and_then(|t| parser::parse(t)) {
        Ok(prog) => typeck::check(&prog).is_ok(),
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// 2) Differential: interpreter (oracle) vs. IR+RC pipeline.
// ---------------------------------------------------------------------------

#[test]
fn differential_interp_vs_ir() {
    const SEEDS: u64 = 500;
    let mut skipped = 0u64;
    let mut checked = 0u64;

    for seed in 0..SEEDS {
        let src = gen_program(seed);
        if !well_typed(&src) {
            skipped += 1;
            continue;
        }
        checked += 1;

        let ast = ast_run(&src);
        let ir = ir_run_rc(&src);

        match (&ast, &ir) {
            (Ok(a), Ok((b, live))) => {
                assert_eq!(
                    a, b,
                    "seed {}: value mismatch\n--- program ---\n{}\n--- interp={:?} ir={:?}",
                    seed, src, a, b
                );
                assert_eq!(
                    *live, 0,
                    "seed {}: IR run leaked {} live cell(s)\n--- program ---\n{}",
                    seed, live, src
                );
            }
            (Err(_), Err(_)) => {}
            _ => panic!(
                "seed {}: Ok/Err shape mismatch\n--- program ---\n{}\n--- interp={:?}\n--- ir={:?}",
                seed, src, ast, ir
            ),
        }
    }

    assert!(checked > 0, "generator produced no well-typed programs");
    // The generator must be mostly correct: fewer than half the seeds skipped.
    assert!(
        skipped * 2 < SEEDS,
        "generator too weak: {}/{} seeds skipped (>= 50%)",
        skipped,
        SEEDS
    );
}

// ---------------------------------------------------------------------------
// 3) RC / reuse must not change observable behavior.
// ---------------------------------------------------------------------------

#[test]
fn reuse_preserves_result() {
    const SEEDS: u64 = 500;
    let mut checked = 0u64;

    for seed in 0..SEEDS {
        // Offset the seeds so this exercises different programs than the
        // differential test (still fully deterministic).
        let src = gen_program(seed.wrapping_add(1_000_000));
        if !well_typed(&src) {
            continue;
        }
        checked += 1;

        let without = ir_run_no_rc(&src);
        let with = ir_run_rc(&src).map(|(s, _)| s);

        match (&without, &with) {
            (Ok(a), Ok(b)) => assert_eq!(
                a, b,
                "seed {}: rc/reuse changed the result\n--- program ---\n{}\n no_rc={:?} rc={:?}",
                seed, src, a, b
            ),
            (Err(_), Err(_)) => {}
            _ => panic!(
                "seed {}: rc/reuse changed Ok/Err shape\n--- program ---\n{}\n no_rc={:?} rc={:?}",
                seed, src, without, with
            ),
        }
    }

    assert!(checked > 0, "generator produced no well-typed programs");
}

// ---------------------------------------------------------------------------
// 4) Frontend fuzzing: lex + parse must never panic on arbitrary input.
// ---------------------------------------------------------------------------

/// Build a random-ish source string mixing source-like fragments, operators,
/// keywords, and arbitrary (valid UTF-8) characters.
fn random_source(rng: &mut Lcg) -> String {
    const FRAGMENTS: &[&str] = &[
        "fn", "main", "let", "match", "if", "else", "type", "Int", "Bool", "Cons", "Nil", "true",
        "false", "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "(", ")", "{", "}", ",", ";", "+",
        "-", "*", "/", "%", "!", "<", ">", "=", "|", ".", "\"", "\\", "0", "42", "1.5", "  ", "\n",
        "\t", ":", "abc", "_", "@", "#", "λ", "→", "🦀",
    ];
    let len = rng.choice(40);
    let mut s = String::new();
    for _ in 0..len {
        if rng.below(4) == 0 {
            // Arbitrary scalar value (kept in a sane range, valid UTF-8).
            let cp = rng.below(0x300) as u32;
            if let Some(c) = char::from_u32(cp) {
                s.push(c);
            }
        } else {
            s.push_str(FRAGMENTS[rng.choice(FRAGMENTS.len())]);
        }
    }
    s
}

#[test]
fn lexer_parser_never_panics() {
    const SEEDS: u64 = 1000;
    // Silence panic output while we intentionally probe for panics.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut failures: Vec<String> = Vec::new();
    for seed in 0..SEEDS {
        let mut rng = Lcg::new(seed.wrapping_mul(0x1234_5678).wrapping_add(7));
        let input = random_source(&mut rng);
        let probe = input.clone();
        let result = std::panic::catch_unwind(move || {
            // Both stages must return Ok/Err, never panic.
            if let Ok(toks) = lexer::lex(&probe) {
                let _ = parser::parse(toks);
            }
        });
        if result.is_err() {
            failures.push(format!("seed {}: panicked on input {:?}", seed, input));
        }
    }

    std::panic::set_hook(prev_hook);
    assert!(failures.is_empty(), "lex/parse panicked:\n{}", failures.join("\n"));
}

// ---------------------------------------------------------------------------
// 5) Codec fuzzing: no panics on arbitrary input + round-trip identity.
// ---------------------------------------------------------------------------

fn random_bytes(rng: &mut Lcg) -> Vec<u8> {
    // Keep inputs small: the bit-level entropy coders are O(len) per byte and
    // we run hundreds of seeds, so large buffers dominate test time without
    // adding coverage.
    let len = rng.choice(48);
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        // Bias toward a small alphabet some of the time (more compressible),
        // arbitrary bytes the rest, to stress both code paths.
        if rng.below(2) == 0 {
            v.push((rng.below(4)) as u8);
        } else {
            v.push((rng.below(256)) as u8);
        }
    }
    v
}

#[test]
fn codec_fuzz_roundtrip_and_no_panic() {
    use crate::{arith, neural_codec, rans};

    const SEEDS: u64 = 300;
    // The neural codec builds a multi-megabyte context-mixing model on every
    // `Predictor::new()`, so each compress/decompress is far heavier than rANS.
    // Exercise it on a representative subset of seeds to keep the suite fast,
    // while rANS and the arithmetic coder run on all SEEDS.
    const NEURAL_EVERY: u64 = 12;
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut failures: Vec<String> = Vec::new();

    for seed in 0..SEEDS {
        let mut rng = Lcg::new(seed.wrapping_mul(0x9E37_79B1).wrapping_add(3));
        let data = random_bytes(&mut rng);

        // (a) decompress must never panic on arbitrary input.
        let d1 = data.clone();
        if std::panic::catch_unwind(move || {
            let _ = rans::decompress(&d1);
        })
        .is_err()
        {
            failures.push(format!("seed {}: rans::decompress panicked on {:?}", seed, data));
        }
        let d3 = data.clone();
        if std::panic::catch_unwind(move || {
            let _ = arith::decompress_adaptive(&d3, 0);
        })
        .is_err()
        {
            failures.push(format!("seed {}: arith::decompress_adaptive panicked", seed));
        }

        // (b) round-trip identity for rANS (all seeds).
        let d4 = data.clone();
        let rt_rans = std::panic::catch_unwind(move || rans::decompress(&rans::compress(&d4)));
        match rt_rans {
            Ok(Ok(out)) => {
                if out != data {
                    failures.push(format!("seed {}: rans round-trip mismatch", seed));
                }
            }
            _ => failures.push(format!("seed {}: rans round-trip errored/panicked", seed)),
        }

        // The expensive neural codec: no-panic on arbitrary input + round-trip,
        // on a sampled subset of seeds.
        if seed % NEURAL_EVERY == 0 {
            let d2 = data.clone();
            if std::panic::catch_unwind(move || {
                let _ = neural_codec::decompress(&d2);
            })
            .is_err()
            {
                failures.push(format!("seed {}: neural_codec::decompress panicked", seed));
            }

            let d5 = data.clone();
            let rt_neural = std::panic::catch_unwind(move || {
                neural_codec::decompress(&neural_codec::compress(&d5))
            });
            match rt_neural {
                Ok(Ok(out)) => {
                    if out != data {
                        failures.push(format!("seed {}: neural_codec round-trip mismatch", seed));
                    }
                }
                _ => {
                    failures.push(format!("seed {}: neural_codec round-trip errored/panicked", seed))
                }
            }
        }
    }

    std::panic::set_hook(prev_hook);
    assert!(failures.is_empty(), "codec fuzz failures:\n{}", failures.join("\n"));
}

// ---------------------------------------------------------------------------
// 6) Compiled-backend differential: WASM (via Node) vs. interpreter (oracle).
// ---------------------------------------------------------------------------

/// True iff `node --version` succeeds; mirrors the gate in `wasm.rs` tests so
/// the suite skips gracefully (never fails) when Node is unavailable.
fn node_available() -> bool {
    std::process::Command::new("node")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run compiled wasm under Node where `main` returns an Int or a heap String,
/// returning `(rendered_result, live_cells)`. An i64 (bigint) result is an Int;
/// any other (an i32 String pointer) is decoded from the String object. Used by
/// the maps/sets differential (whose programs return an Int or the canonical
/// ordered-display String — never a Float, so the bigint/else split is
/// unambiguous). Mirrors the `wasm.rs` `differential_gc` harness.
fn run_wasm_live_str(bytes: &[u8]) -> Result<(String, i64), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "aria_proptest_wasm_str_{}_{}.wasm",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
    let script = format!(
        "const fs=require('fs');\
         const dec=new TextDecoder();\
         let memref=null;\
         const imp={{env:{{print_str:(p,n)=>{{\
         process.stdout.write(dec.decode(new Uint8Array(memref.buffer).subarray(p,p+n)));\
         process.stdout.write('\\n');}},\
         print_float:(x)=>{{process.stdout.write(String(x));process.stdout.write('\\n');}},\
         print_int:(n)=>{{process.stdout.write(String(n));process.stdout.write('\\n');}},\
         print_bool:(b)=>{{process.stdout.write(b?'true':'false');process.stdout.write('\\n');}},\
         exp:Math.exp}}}};\
         const b=fs.readFileSync({:?});\
         WebAssembly.instantiate(b,imp).then(r=>{{\
         const ex=r.instance.exports;memref=ex.memory;\
         const v=ex.main();\
         let m;if(typeof v==='bigint'){{m=String(v);}}\
         else{{const dv=new DataView(ex.memory.buffer);\
         const len=Number(dv.getBigInt64(v+8,true));\
         m=dec.decode(new Uint8Array(ex.memory.buffer).subarray(v+16,v+16+len));\
         ex.__drop_str(v);}}\
         const l=String(ex.__live());\
         process.stdout.write(m+'|'+l);\
         }}).catch(e=>process.stdout.write('TRAP|0'));",
        path.to_string_lossy()
    );
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(&script)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    let (res, live) = s.split_once('|').ok_or("bad harness output")?;
    Ok((res.to_string(), live.parse::<i64>().unwrap_or(-1)))
}

/// Run compiled wasm under Node, returning `(main_result, live_cells)`.
/// `live` is the value of the exported `__live()` after `main` returns; a TRAP
/// (e.g. an overflow `unreachable`) surfaces as `("TRAP", _)`. Mirrors the
/// minimal Node invocation used by the `wasm.rs` tests' `run_wasm_live`.
fn run_wasm_live(bytes: &[u8]) -> Result<(String, i64), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "aria_proptest_wasm_{}_{}.wasm",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
    let script = format!(
        "const fs=require('fs');\
         const dec=new TextDecoder();\
         let memref=null;\
         const imp={{env:{{print_str:(p,n)=>{{\
         process.stdout.write(dec.decode(new Uint8Array(memref.buffer).subarray(p,p+n)));\
         process.stdout.write('\\n');}},\
         print_float:(x)=>{{process.stdout.write(String(x));process.stdout.write('\\n');}},\
         print_int:(n)=>{{process.stdout.write(String(n));process.stdout.write('\\n');}},\
         print_bool:(b)=>{{process.stdout.write(b?'true':'false');process.stdout.write('\\n');}},\
         exp:Math.exp}}}};\
         const b=fs.readFileSync({:?});\
         WebAssembly.instantiate(b,imp).then(r=>{{\
         const ex=r.instance.exports;memref=ex.memory;\
         const v=ex.main();\
         let m;if(typeof v==='bigint'){{m=String(v);}}\
         else if(typeof v==='number'){{m=String(v);}}\
         else{{const dv=new DataView(ex.memory.buffer);\
         const len=Number(dv.getBigInt64(v+8,true));\
         m=dec.decode(new Uint8Array(ex.memory.buffer).subarray(v+16,v+16+len));}}\
         const l=String(ex.__live());\
         process.stdout.write(m+'|'+l);\
         }}).catch(e=>process.stdout.write('TRAP|0'));",
        path.to_string_lossy()
    );
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(&script)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    let (res, live) = s.split_once('|').ok_or("bad harness output")?;
    Ok((res.to_string(), live.parse::<i64>().unwrap_or(-1)))
}

#[test]
fn wasm_matches_interpreter_fuzz() {
    // Node-gated: skip the whole test gracefully if Node is missing.
    if !node_available() {
        return;
    }

    // Keep it fast: each compiled seed shells out to Node, so we sample a
    // bounded number of seeds. The wasm-subset generator keeps most of them
    // compilable.
    const SEEDS: u64 = 120;
    let mut skipped = 0u64; // not well-typed, or out-of-wasm-subset
    let mut overflow = 0u64; // interpreter overflow -> wasm traps (expected)
    let mut checked = 0u64; // actually exercised through wasm + compared

    for seed in 0..SEEDS {
        let src = gen_program_wasm(seed);
        if !well_typed(&src) {
            skipped += 1;
            continue;
        }

        let bytes = match wasm::compile(
            &parser::parse(lexer::lex(&src).expect("lex")).expect("parse"),
        ) {
            Ok(b) => b,
            // Out-of-subset (Float/String/etc.) is expected, not a failure.
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        // Oracle: the tree-walking interpreter. Overflow is a *defined error*
        // in Aria; the interpreter returns Err while wasm traps. When the
        // interpreter errors we can't meaningfully compare a value, so we skip
        // (but still confirm wasm trapped rather than producing a bogus value).
        let interp = ast_run(&src);
        let (wasm_res, live) = run_wasm_live(&bytes)
            .unwrap_or_else(|e| panic!("seed {}: node runner failed: {}\n{}", seed, e, src));

        match interp {
            Ok(expected) => {
                assert_eq!(
                    expected, wasm_res,
                    "seed {}: wasm != interpreter\n--- program ---\n{}\n--- interp={:?} wasm={:?}",
                    seed, src, expected, wasm_res
                );
                assert_eq!(
                    live, 0,
                    "seed {}: wasm leaked {} live cell(s)\n--- program ---\n{}",
                    seed, live, src
                );
                checked += 1;
            }
            Err(_) => {
                // Interpreter errored (overflow). Wasm must trap, not silently
                // return a wrapped value.
                assert_eq!(
                    wasm_res, "TRAP",
                    "seed {}: interpreter errored but wasm did not trap (={:?})\n{}",
                    seed, wasm_res, src
                );
                overflow += 1;
            }
        }
    }

    eprintln!(
        "wasm_matches_interpreter_fuzz: {} seeds -> {} checked, {} overflow-trap, {} skipped",
        SEEDS, checked, overflow, skipped
    );

    // Guard against vacuous success: a healthy number of seeds must have run
    // all the way through wasm and agreed with the interpreter.
    assert!(
        checked >= 20,
        "too few programs exercised through wasm: {} (skipped {}, overflow {})",
        checked,
        skipped,
        overflow
    );
}

// ---------------------------------------------------------------------------
// 7) Compiled-backend differential: native C vs. interpreter (oracle).
//    Uses the FULL generator (includes arrays), so this is the only fuzzer that
//    exercises the native array runtime (AriaArray, FBIP) end to end.
// ---------------------------------------------------------------------------

fn cc_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build C source with `cc -O2`, run it, and return `(main_result, live_cells)`.
/// The native binary prints `main`'s value to stdout and `aria_live=N` to
/// stderr; a runtime trap (overflow / div-by-zero / OOB → `abort`) surfaces as
/// `("TRAP", 0)`.
fn run_native_live(c_src: &str) -> Result<(String, i64), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let cpath = dir.join(format!("aria_ptc_{}_{}.c", std::process::id(), n));
    let exe = dir.join(format!("aria_pte_{}_{}", std::process::id(), n));
    std::fs::write(&cpath, c_src).map_err(|e| e.to_string())?;
    let cc = std::process::Command::new("cc")
        .arg("-O2").arg("-std=c11").arg("-o").arg(&exe).arg(&cpath).arg("-lm")
        .output().map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&cpath);
    if !cc.status.success() {
        let _ = std::fs::remove_file(&exe);
        return Err(format!("cc failed: {}", String::from_utf8_lossy(&cc.stderr)));
    }
    let run = std::process::Command::new(&exe).output().map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&exe);
    if !run.status.success() {
        // Non-zero exit / signal = a defined Aria runtime error trapped via abort.
        return Ok(("TRAP".to_string(), 0));
    }
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    let result = stdout.lines().last().unwrap_or("").trim().to_string();
    let live = stderr
        .split("aria_live=")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(-1);
    Ok((result, live))
}

#[test]
fn native_matches_interpreter_fuzz() {
    // cc-gated: skip gracefully when no C compiler is available.
    if !cc_available() {
        return;
    }
    // Each seed shells out to `cc` + runs a binary, so sample a bounded count.
    const SEEDS: u64 = 80;
    let (mut skipped, mut overflow, mut checked) = (0u64, 0u64, 0u64);

    for seed in 0..SEEDS {
        let src = gen_program(seed); // full generator: includes arrays
        if !well_typed(&src) {
            skipped += 1;
            continue;
        }
        let prog = parser::parse(lexer::lex(&src).expect("lex")).expect("parse");
        let c_src = match crate::c_backend::compile(&prog) {
            Ok(c) => c,
            // Out of the native subset (e.g. compression builtins) — expected.
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let interp = ast_run(&src);
        let (nat, live) = run_native_live(&c_src)
            .unwrap_or_else(|e| panic!("seed {}: native runner failed: {}\n{}", seed, e, src));

        match interp {
            Ok(expected) => {
                assert_eq!(
                    expected, nat,
                    "seed {}: native != interpreter\n--- program ---\n{}\n--- interp={:?} native={:?}",
                    seed, src, expected, nat
                );
                assert_eq!(
                    live, 0,
                    "seed {}: native leaked {} live cell(s)\n--- program ---\n{}",
                    seed, live, src
                );
                checked += 1;
            }
            Err(_) => {
                assert_eq!(
                    nat, "TRAP",
                    "seed {}: interpreter errored but native did not trap (={:?})\n{}",
                    seed, nat, src
                );
                overflow += 1;
            }
        }
    }

    eprintln!(
        "native_matches_interpreter_fuzz: {} seeds -> {} checked, {} trap, {} skipped",
        SEEDS, checked, overflow, skipped
    );
    assert!(
        checked >= 15,
        "too few programs exercised through native: {} (skipped {})",
        checked, skipped
    );
}

// ---------------------------------------------------------------------------
// 8) Traits / interfaces (M3): differential interp vs. native vs. wasm.
//    Static dispatch through monomorphization must agree with the interpreter's
//    runtime-constructor dispatch, for a trait over an ADT, over a record, and
//    through a bounded generic function.
// ---------------------------------------------------------------------------

/// Trait programs whose `main` RETURNS the value produced through trait
/// dispatch (so the compiled runners' "last stdout line" comparison is exact).
/// Each pairs the source with its expected `main` result.
fn trait_diff_programs() -> Vec<(String, &'static str)> {
    vec![
        // Trait over an ADT and a record; direct + bounded-generic dispatch.
        (
            r#"
            interface Describe[T] { fn code(self: T) -> Int }
            type Shape = | Circle | Square | Triangle
            impl Describe for Shape { fn code(self: Shape) -> Int = match self { Circle => 1, Square => 2, Triangle => 3, } }
            type Point = { x: Int, y: Int }
            impl Describe for Point { fn code(self: Point) -> Int = self.x + self.y }
            fn twice[T: Describe](v: T) -> Int = code(v) + code(v)
            fn main() -> Int = code(Triangle) + code(Point { x: 10, y: 7 }) + twice(Square) + twice(Point { x: 1, y: 2 })
            "#
            .to_string(),
            // 3 + 17 + 4 + 6 = 30
            "30",
        ),
        // Trait over a generic ADT instantiated at a concrete type, plus a
        // multi-method interface.
        (
            r#"
            interface Sz[T] { fn sz(self: T) -> Int, fn dbl(self: T) -> Int }
            type Box = | Empty | Full(Int)
            impl Sz for Box { fn sz(self: Box) -> Int = match self { Empty => 0, Full(n) => n, }, fn dbl(self: Box) -> Int = sz(self) * 2 }
            fn main() -> Int = sz(Full(21)) + dbl(Full(4))
            "#
            .to_string(),
            // 21 + 8 = 29
            "29",
        ),
    ]
}

#[test]
fn traits_interp_matches_compiled() {
    let progs = trait_diff_programs();
    let cc = cc_available();
    let node = node_available();
    let mut native_checked = 0u64;
    let mut wasm_checked = 0u64;

    for (src, expected) in &progs {
        // Oracle: tree-walking interpreter. `main` returns an Int value.
        let interp = ast_run(src).unwrap_or_else(|e| panic!("interp failed: {}\n{}", e, src));
        assert_eq!(
            &interp, expected,
            "interpreter result mismatch\n{}\n got={} want={}",
            src, interp, expected
        );

        let prog = parser::parse(lexer::lex(src).expect("lex")).expect("parse");
        assert!(typeck::check(&prog).is_ok(), "type error\n{}", src);

        // Native (C) backend.
        if cc {
            let c_src = crate::c_backend::compile(&prog)
                .unwrap_or_else(|e| panic!("c_backend failed: {}\n{}", e, src));
            let (nat, live) = run_native_live(&c_src)
                .unwrap_or_else(|e| panic!("native runner failed: {}\n{}", e, src));
            assert_eq!(nat, *expected, "native != expected\n{}\n native={}", src, nat);
            assert_eq!(live, 0, "native leaked {} cell(s)\n{}", live, src);
            native_checked += 1;
        }

        // WASM backend.
        if node {
            let bytes = wasm::compile(&prog)
                .unwrap_or_else(|e| panic!("wasm compile failed: {}\n{}", e, src));
            let (w, live) = run_wasm_live(&bytes)
                .unwrap_or_else(|e| panic!("wasm runner failed: {}\n{}", e, src));
            assert_eq!(w, *expected, "wasm != expected\n{}\n wasm={}", src, w);
            assert_eq!(live, 0, "wasm leaked {} cell(s)\n{}", live, src);
            wasm_checked += 1;
        }
    }

    eprintln!(
        "traits_interp_matches_compiled: {} programs ({} native, {} wasm)",
        progs.len(),
        native_checked,
        wasm_checked
    );
}

// ---------------------------------------------------------------------------
// 8) Bytes differential: interp (oracle) vs native (cc-gated) vs wasm
//    (node-gated). Exercises bytes_new/len/get/set/push/from_str/to_str + `==`,
//    requiring identical results AND a garbage-free heap (live == 0) on both
//    compiled backends. Each `main` returns an Int (the harness compares the
//    final value); the Bytes builtins run mid-program, so all are exercised.
// ---------------------------------------------------------------------------

/// Hand-written Bytes programs and their expected Int results.
fn bytes_diff_programs() -> Vec<(String, &'static str)> {
    vec![
        // from_str + len + get + a sum over the buffer.
        (
            "fn sumb(b: Bytes, i: Int, acc: Int) -> Int =\n\
               if i == bytes_len(b) { acc } else { sumb(b, i + 1, acc + bytes_get(b, i)) }\n\
             fn main() -> Int = sumb(bytes_from_str(\"ABC\"), 0, 0)\n"
                .to_string(),
            // 65 + 66 + 67
            "198",
        ),
        // push grows the buffer; set overwrites in place (FBIP on a unique buffer).
        (
            "fn main() -> Int = {\n\
               let b = bytes_push(bytes_push(bytes_new(), 10), 20);\n\
               let b2 = bytes_set(b, 0, 100);\n\
               bytes_get(b2, 0) + bytes_get(b2, 1) + bytes_len(b2)\n\
             }\n"
                .to_string(),
            // 100 + 20 + 2
            "122",
        ),
        // round-trip from_str -> to_str -> from_str, then length.
        (
            "fn main() -> Int = bytes_len(bytes_from_str(concat(\"he\", \"llo\")))\n"
                .to_string(),
            "5",
        ),
        // structural `==`: equal contents -> 1, different -> 0 (distinct buffers).
        (
            "fn main() -> Int = {\n\
               let a = if bytes_from_str(\"xy\") == bytes_from_str(\"xy\") { 1 } else { 0 };\n\
               let c = if bytes_from_str(\"xy\") == bytes_from_str(\"zz\") { 10 } else { 0 };\n\
               a + c\n\
             }\n"
                .to_string(),
            // equal -> 1, unequal -> 0
            "1",
        ),
        // a shared buffer forces copy-on-write on set (still garbage-free).
        (
            "fn pair(b: Bytes) -> Int = bytes_get(bytes_set(b, 0, 1), 0) + bytes_get(b, 0)\n\
             fn main() -> Int = pair(bytes_push(bytes_new(), 9))\n"
                .to_string(),
            // set(copy)->1 at idx0, original idx0 still 9
            "10",
        ),
    ]
}

#[test]
fn bytes_interp_matches_compiled() {
    let progs = bytes_diff_programs();
    let cc = cc_available();
    let node = node_available();
    let mut native_checked = 0u64;
    let mut wasm_checked = 0u64;

    for (src, expected) in &progs {
        // Oracle: tree-walking interpreter.
        let interp = ast_run(src).unwrap_or_else(|e| panic!("interp failed: {}\n{}", e, src));
        assert_eq!(
            &interp, expected,
            "interpreter result mismatch\n{}\n got={} want={}",
            src, interp, expected
        );

        // IR + RC pipeline must agree AND be garbage-free (no live cells).
        let (ir_res, ir_live) =
            ir_run_rc(src).unwrap_or_else(|e| panic!("ir failed: {}\n{}", e, src));
        assert_eq!(&ir_res, expected, "ir != expected\n{}\n ir={}", src, ir_res);
        assert_eq!(ir_live, 0, "ir leaked {} cell(s)\n{}", ir_live, src);

        let prog = parser::parse(lexer::lex(src).expect("lex")).expect("parse");
        assert!(typeck::check(&prog).is_ok(), "type error\n{}", src);

        // Native (C) backend: identical result + garbage-free.
        if cc {
            let c_src = crate::c_backend::compile(&prog)
                .unwrap_or_else(|e| panic!("c_backend failed: {}\n{}", e, src));
            let (nat, live) = run_native_live(&c_src)
                .unwrap_or_else(|e| panic!("native runner failed: {}\n{}", e, src));
            assert_eq!(nat, *expected, "native != expected\n{}\n native={}", src, nat);
            assert_eq!(live, 0, "native leaked {} cell(s)\n{}", live, src);
            native_checked += 1;
        }

        // WASM backend: identical result + garbage-free.
        if node {
            let bytes = wasm::compile(&prog)
                .unwrap_or_else(|e| panic!("wasm compile failed: {}\n{}", e, src));
            let (w, live) = run_wasm_live(&bytes)
                .unwrap_or_else(|e| panic!("wasm runner failed: {}\n{}", e, src));
            assert_eq!(w, *expected, "wasm != expected\n{}\n wasm={}", src, w);
            assert_eq!(live, 0, "wasm leaked {} cell(s)\n{}", live, src);
            wasm_checked += 1;
        }
    }

    eprintln!(
        "bytes_interp_matches_compiled: {} programs ({} native, {} wasm)",
        progs.len(),
        native_checked,
        wasm_checked
    );
}

// ---------------------------------------------------------------------------
// 9) Maps & Sets differential: interp (oracle) vs native (cc-gated). Maps/Sets
//    are NOT supported by the wasm backend, so wasm is intentionally excluded
//    here. Each program exercises build / insert / lookup(get_or) / has / len /
//    remove plus the ORDERED display (`map_show`/`set_show`), with Str keys, key
//    replacement, removal, and out-of-order inserts. The interpreter and the
//    native backend must produce IDENTICAL results AND the native heap must be
//    garbage-free (live == 0). Programs return either an Int or a String (the
//    canonical ordered rendering); `run_native_live` compares the final value.
// ---------------------------------------------------------------------------

/// Hand-written Map/Set programs and their expected results (Int or the
/// canonical ordered-display String).
fn map_set_diff_programs() -> Vec<(String, &'static str)> {
    vec![
        // Out-of-order inserts render in ascending key order; replacement keeps
        // the latest value.
        (
            "fn main() -> String = {\n\
               let m = map_insert(map_insert(map_insert(map_new(), 30, 3), 10, 1), 20, 2);\n\
               map_show(map_insert(m, 10, 111))\n\
             }\n"
                .to_string(),
            "Map[10: 111, 20: 2, 30: 3]",
        ),
        // Str-keyed map, out of order -> sorted display.
        (
            "fn main() -> String =\n\
               map_show(map_insert(map_insert(map_insert(map_new(), \"pear\", 3), \"apple\", 1), \"fig\", 2))\n"
                .to_string(),
            "Map[apple: 1, fig: 2, pear: 3]",
        ),
        // Bool VALUES must render as `true`/`false` (not `1`/`0`) in every backend.
        (
            "fn main() -> String =\n\
               map_show(map_insert(map_insert(map_new(), 1, true), 2, false))\n"
                .to_string(),
            "Map[1: true, 2: false]",
        ),
        // Total read: present -> value, absent -> default; len after a replace.
        (
            "fn main() -> Int = {\n\
               let m = map_insert(map_insert(map_insert(map_new(), 1, 10), 2, 20), 1, 99);\n\
               map_get_or(m, 1, 0) + map_get_or(m, 7, 5) + map_len(m)\n\
             }\n"
                .to_string(),
            // 99 (replaced) + 5 (absent default) + 2 (len) = 106
            "106",
        ),
        // has + remove; removed key absent, survivor present.
        (
            "fn main() -> String = {\n\
               let m = map_insert(map_insert(map_insert(map_new(), 1, 10), 2, 20), 3, 30);\n\
               map_show(map_remove(m, 2))\n\
             }\n"
                .to_string(),
            "Map[1: 10, 3: 30]",
        ),
        // map equality independent of insertion order (returned as Int).
        (
            "fn main() -> Int = {\n\
               let a = map_insert(map_insert(map_new(), 1, 10), 2, 20);\n\
               let b = map_insert(map_insert(map_new(), 2, 20), 1, 10);\n\
               if a == b { 1 } else { 0 }\n\
             }\n"
                .to_string(),
            "1",
        ),
        // A shared map forces copy-on-write on insert (still garbage-free).
        (
            "fn use2(m: Map[Int, Int]) -> Int =\n\
               map_get_or(map_insert(m, 5, 50), 5, 0) + map_get_or(m, 1, -1)\n\
             fn main() -> Int = use2(map_insert(map_new(), 1, 10))\n"
                .to_string(),
            // 50 (inserted copy) + 10 (original key 1 intact) = 60
            "60",
        ),
        // Set: out-of-order, duplicate adds -> sorted, deduped display.
        (
            "fn main() -> String =\n\
               set_show(set_add(set_add(set_add(set_add(set_new(), 30), 10), 20), 10))\n"
                .to_string(),
            "Set[10, 20, 30]",
        ),
        // Str set, sorted; remove drops an element.
        (
            "fn main() -> String =\n\
               set_show(set_remove(set_add(set_add(set_add(set_new(), \"b\"), \"a\"), \"c\"), \"a\"))\n"
                .to_string(),
            "Set[b, c]",
        ),
        // Set has / len / dedup as an Int.
        (
            "fn main() -> Int = {\n\
               let s = set_add(set_add(set_add(set_new(), 1), 1), 2);\n\
               let h = if set_has(s, 2) { 100 } else { 0 };\n\
               h + set_len(s)\n\
             }\n"
                .to_string(),
            // 100 + 2 (deduped) = 102
            "102",
        ),
        // A Map captured in a closure must be dup'd on load (it is dropped by the
        // closure's drop-children helper). Regression for a native use-after-free
        // where the captured map was freed while still live in the caller.
        (
            "fn apply(f: (Int) -> Int, x: Int) -> Int = f(x)\n\
             fn main() -> Int = {\n\
               let m = map_insert(map_insert(map_new(), 1, 10), 2, 20);\n\
               let g = \\x -> map_len(m);\n\
               apply(g, 0) + map_len(m)\n\
             }\n"
                .to_string(),
            // closure reads len 2, then the original map is still live -> 2 + 2
            "4",
        ),
        // Likewise for a captured Set.
        (
            "fn apply(f: (Int) -> Int, x: Int) -> Int = f(x)\n\
             fn main() -> Int = {\n\
               let s = set_add(set_add(set_new(), 1), 2);\n\
               let g = \\x -> set_len(s);\n\
               apply(g, 0) + set_len(s)\n\
             }\n"
                .to_string(),
            "4",
        ),
    ]
}

#[test]
fn maps_sets_interp_matches_compiled() {
    let progs = map_set_diff_programs();
    let cc = cc_available();
    let node = node_available();
    let mut native_checked = 0u64;
    let mut wasm_ran = 0u64;

    for (src, expected) in &progs {
        // Oracle: tree-walking interpreter.
        let interp = ast_run(src).unwrap_or_else(|e| panic!("interp failed: {}\n{}", e, src));
        assert_eq!(
            &interp, expected,
            "interpreter result mismatch\n{}\n got={} want={}",
            src, interp, expected
        );

        let prog = parser::parse(lexer::lex(src).expect("lex")).expect("parse");
        assert!(typeck::check(&prog).is_ok(), "type error\n{}", src);

        // Native (C) backend: identical result + garbage-free (live == 0).
        if cc {
            let c_src = crate::c_backend::compile(&prog)
                .unwrap_or_else(|e| panic!("c_backend failed: {}\n{}", e, src));
            let (nat, live) = run_native_live(&c_src)
                .unwrap_or_else(|e| panic!("native runner failed: {}\n{}", e, src));
            assert_eq!(nat, *expected, "native != expected\n{}\n native={}", src, nat);
            assert_eq!(live, 0, "native leaked {} cell(s)\n{}", live, src);
            native_checked += 1;
        }

        // The wasm backend now RUNS ordered maps/sets fully: every map_*/set_* op
        // is real hand-emitted wasm. These programs return an Int or the canonical
        // ordered-display String (Int/Str keys, Int values, map_show/set_show) —
        // all supported. The result must match the interpreter and the heap must
        // be garbage-free (`__live == 0`, the returned String released by the
        // harness, exactly as native drops its printed result).
        let bytes = wasm::compile(&prog)
            .unwrap_or_else(|e| panic!("wasm compile failed (should run): {}\n{}", e, src));
        if node {
            let (w, live) = run_wasm_live_str(&bytes)
                .unwrap_or_else(|e| panic!("wasm runner failed: {}\n{}", e, src));
            assert_eq!(w, *expected, "wasm != expected\n{}\n wasm={}", src, w);
            assert_eq!(live, 0, "wasm leaked {} cell(s)\n{}", live, src);
            wasm_ran += 1;
        }
    }

    eprintln!(
        "maps_sets_interp_matches_compiled: {} programs ({} native, {} wasm ran)",
        progs.len(),
        native_checked,
        wasm_ran
    );
}

/// A Float- or Bytes-VALUED `map_show`/`set_show` is the one gated wasm case:
/// the wasm backend has no in-wasm shortest-round-trip float / hex formatter
/// (the same gap that gates a Vector/Bytes-returning main). The gate is a CLEAN
/// compile error — never a panic or a wrong number — while every NON-show op on
/// the same Float/Bytes-valued map still compiles and runs.
#[test]
fn wasm_gates_float_valued_map_show() {
    // Float-valued map_show: gated cleanly under wasm.
    let show_src = "fn main() -> String = map_show(map_insert(map_new(), 1, 1.5))\n";
    let prog = parser::parse(lexer::lex(show_src).expect("lex")).expect("parse");
    assert!(typeck::check(&prog).is_ok(), "type error\n{}", show_src);
    // The interpreter renders it fine (the oracle); wasm gates the rendering.
    assert!(ast_run(show_src).is_ok(), "interpreter should run map_show\n{}", show_src);
    let err = wasm::compile(&prog)
        .err()
        .unwrap_or_else(|| panic!("wasm should gate a Float-valued map_show\n{}", show_src));
    assert!(
        err.contains("map_show") && err.contains("Float"),
        "expected a clean Float-show gate, got: {}\n{}",
        err,
        show_src
    );

    // ...but the NON-show ops on a Float-valued map compile + run under wasm and
    // match the interpreter, garbage-free.
    let ops_src = "fn main() -> Int = {\n\
                     let m = map_insert(map_insert(map_new(), 2, 2.5), 1, 1.5);\n\
                     let v = map_get_or(m, 1, 0.0);\n\
                     (if v == 1.5 { 100 } else { 0 }) + (if map_has(m, 2) { 10 } else { 0 }) + map_len(m)\n\
                   }\n";
    let oprog = parser::parse(lexer::lex(ops_src).expect("lex")).expect("parse");
    assert!(typeck::check(&oprog).is_ok(), "type error\n{}", ops_src);
    let expected = ast_run(ops_src).expect("interp float ops");
    assert_eq!(expected, "112");
    let bytes = wasm::compile(&oprog)
        .unwrap_or_else(|e| panic!("wasm should compile Float-valued non-show ops: {}\n{}", e, ops_src));
    if node_available() {
        let (w, live) = run_wasm_live_str(&bytes).expect("wasm run float ops");
        assert_eq!(w, expected, "wasm float-ops != interp\n{}", ops_src);
        assert_eq!(live, 0, "wasm leaked on float ops\n{}", ops_src);
    }
}

// ---------------------------------------------------------------------------
// 9b) Collection enumeration differential: `map_keys` / `map_values` /
//     `set_to_array` turn a Map/Set into a plain Array (in the same ascending,
//     deterministic order used for display/equality) so it can be iterated with
//     the prelude array HOFs (`array_fold`/`array_map`). Each program is wrapped
//     with the real prelude (so the HOFs are in scope) and run through interp
//     (oracle) vs native (cc-gated) vs wasm (node-gated); results must be
//     IDENTICAL and both compiled heaps garbage-free (live == 0). (The `aria mem`
//     IR path still rejects maps/sets — covered elsewhere.)
// ---------------------------------------------------------------------------

/// User programs (prelude appended at run time) combining the enumeration
/// builtins with the array HOFs, plus their expected Int / String results.
fn enum_hof_diff_programs() -> Vec<(String, &'static str)> {
    vec![
        // map_values folded: sum of an out-of-order Int->Int map = 1+2+3+4 = 10.
        (
            "fn main() -> Int = {\n\
               let m = map_insert(map_insert(map_insert(map_insert(map_new(), 30, 3), 10, 1), 20, 2), 40, 4);\n\
               array_fold(map_values(m), 0, \\(a: Int, x: Int) -> a + x)\n\
             }\n"
                .to_string(),
            "10",
        ),
        // map_keys folded: keys come out ASCENDING regardless of insert order;
        // sum 10+20+30+40 = 100.
        (
            "fn main() -> Int = {\n\
               let m = map_insert(map_insert(map_insert(map_insert(map_new(), 30, 3), 10, 1), 20, 2), 40, 4);\n\
               array_fold(map_keys(m), 0, \\(a: Int, x: Int) -> a + x)\n\
             }\n"
                .to_string(),
            "100",
        ),
        // map_keys mapped then folded: double each key, sum = 200.
        (
            "fn main() -> Int = {\n\
               let m = map_insert(map_insert(map_insert(map_insert(map_new(), 30, 3), 10, 1), 20, 2), 40, 4);\n\
               array_fold(array_map(map_keys(m), \\x -> x * 2), 0, \\(a: Int, x: Int) -> a + x)\n\
             }\n"
                .to_string(),
            "200",
        ),
        // Str-keyed map: keys ASCENDING -> concat = "applefigpear".
        (
            "fn main() -> String = {\n\
               let m = map_insert(map_insert(map_insert(map_new(), \"pear\", 9), \"apple\", 7), \"fig\", 8);\n\
               array_fold(map_keys(m), \"\", \\(a: String, x: String) -> concat(a, x))\n\
             }\n"
                .to_string(),
            "applefigpear",
        ),
        // Str-keyed map: values index-aligned with the sorted keys, sum = 24.
        (
            "fn main() -> Int = {\n\
               let m = map_insert(map_insert(map_insert(map_new(), \"pear\", 9), \"apple\", 7), \"fig\", 8);\n\
               array_fold(map_values(m), 0, \\(a: Int, x: Int) -> a + x)\n\
             }\n"
                .to_string(),
            "24",
        ),
        // Int set (out of order, with a duplicate) -> ascending array, sum = 18.
        (
            "fn main() -> Int = {\n\
               let s = set_add(set_add(set_add(set_add(set_add(set_new(), 5), 1), 3), 1), 9);\n\
               array_fold(set_to_array(s), 0, \\(a: Int, x: Int) -> a + x)\n\
             }\n"
                .to_string(),
            "18",
        ),
        // Str set -> ASCENDING array -> concat = "alphacharliedelta".
        (
            "fn main() -> String = {\n\
               let s = set_add(set_add(set_add(set_add(set_new(), \"delta\"), \"alpha\"), \"charlie\"), \"alpha\");\n\
               array_fold(set_to_array(s), \"\", \\(a: String, x: String) -> concat(a, x))\n\
             }\n"
                .to_string(),
            "alphacharliedelta",
        ),
        // Empty map/set -> empty array -> fold returns the seed unchanged. The
        // element type is pinned via an annotated builder fn (an UNANNOTATED bare
        // `map_new()` whose value type is never determined is the usual
        // whole-language `array_new()`-style limitation, out of scope here).
        (
            "fn empty_im() -> Map[Int, Int] = map_new()\n\
             fn main() -> Int =\n\
               array_fold(map_values(empty_im()), 7, \\(a: Int, x: Int) -> a + x)\n"
                .to_string(),
            "7",
        ),
        (
            "fn empty_is() -> Set[Int] = set_new()\n\
             fn main() -> Int =\n\
               array_fold(set_to_array(empty_is()), 5, \\(a: Int, x: Int) -> a + x) + array_len(set_to_array(empty_is()))\n"
                .to_string(),
            "5",
        ),
        // Float-valued map: map_values yields an Array[Float]; sum = 1.5+2.5 = 4.0.
        (
            "fn main() -> Int = {\n\
               let m = map_insert(map_insert(map_new(), 2, 2.5), 1, 1.5);\n\
               let xs = map_values(m);\n\
               if array_get(xs, 0) + array_get(xs, 1) == 4.0 { 1 } else { 0 }\n\
             }\n"
                .to_string(),
            "1",
        ),
    ]
}

#[test]
fn enum_hofs_match_across_backends() {
    let progs = enum_hof_diff_programs();
    let cc = cc_available();
    let node = node_available();
    let mut native_checked = 0u64;
    let mut wasm_ran = 0u64;

    for (user_src, expected) in &progs {
        // Wrap with the real prelude (exactly as the CLI does) so the array HOFs
        // are in scope.
        let src = crate::prelude::wrap(user_src);

        // Oracle: tree-walking interpreter.
        let interp =
            ast_run(&src).unwrap_or_else(|e| panic!("interp failed: {}\n{}", e, user_src));
        assert_eq!(
            &interp, expected,
            "interpreter result mismatch\n{}\n got={} want={}",
            user_src, interp, expected
        );

        let prog = parser::parse(lexer::lex(&src).expect("lex")).expect("parse");
        assert!(typeck::check(&prog).is_ok(), "type error\n{}", user_src);

        // Native (C) backend: identical result + garbage-free (live == 0).
        if cc {
            let c_src = crate::c_backend::compile(&prog)
                .unwrap_or_else(|e| panic!("c_backend failed: {}\n{}", e, user_src));
            let (nat, live) = run_native_live(&c_src)
                .unwrap_or_else(|e| panic!("native runner failed: {}\n{}", e, user_src));
            assert_eq!(nat, *expected, "native != expected\n{}\n native={}", user_src, nat);
            assert_eq!(live, 0, "native leaked {} cell(s)\n{}", live, user_src);
            native_checked += 1;
        }

        // Wasm backend: `map_keys`/`map_values`/`set_to_array` build a real wasm
        // Array enumerated in ascending order, folded by the prelude HOFs. Each
        // program returns an Int or a String (never a Float/Bytes show), so wasm
        // runs them and must match the interpreter, garbage-free.
        let bytes = wasm::compile(&prog)
            .unwrap_or_else(|e| panic!("wasm compile failed (should run): {}\n{}", e, user_src));
        if node {
            let (w, live) = run_wasm_live_str(&bytes)
                .unwrap_or_else(|e| panic!("wasm runner failed: {}\n{}", e, user_src));
            assert_eq!(w, *expected, "wasm != expected\n{}\n wasm={}", user_src, w);
            assert_eq!(live, 0, "wasm leaked {} cell(s)\n{}", live, user_src);
            wasm_ran += 1;
        }
    }

    eprintln!(
        "enum_hofs_match_across_backends: {} programs ({} native, {} wasm ran)",
        progs.len(),
        native_checked,
        wasm_ran
    );
}

/// A Map whose VALUE type is non-flat (Array, nested Map, tuple/ADT) is not
/// faithfully representable in the native backend (coarse value slots), so the
/// compiled path must REJECT it cleanly — but the interpreter still supports it.
/// This prevents silent interp-vs-native divergence (pointer display / identity
/// equality / lost element layout on retrieval).
#[test]
fn native_rejects_non_flat_map_values() {
    let cases = [
        // Array value: retrieving + indexing would decode the element wrong.
        "fn main() -> Int = {\n\
           let a = array_push(array_new(), 7);\n\
           let m = map_insert(map_new(), 1, a);\n\
           array_get(map_get_or(m, 1, array_new()), 0)\n\
         }\n",
        // Nested-map value.
        "fn main() -> Int = {\n\
           let inner = map_insert(map_new(), 1, 2);\n\
           map_len(map_insert(map_new(), 1, inner))\n\
         }\n",
    ];
    for src in cases {
        let prog = parser::parse(lexer::lex(src).expect("lex")).expect("parse");
        // Type-checks (the interpreter supports these value types)...
        assert!(typeck::check(&prog).is_ok(), "should type-check\n{}", src);
        assert!(ast_run(src).is_ok(), "interpreter should run it\n{}", src);
        // ...but the native (C) backend rejects it cleanly, no panic.
        let err = crate::c_backend::compile(&prog).unwrap_err();
        assert!(
            err.contains("Map value of type") && err.contains("not yet supported"),
            "expected a clean native rejection, got: {}\n{}",
            err,
            src
        );
    }
}

// ---------------------------------------------------------------------------
// 10) Vectors / embeddings differential: interp (oracle) vs native (cc-gated).
//     Vectors are NOT supported by the wasm backend, so wasm is intentionally
//     excluded (and asserted to be cleanly rejected). Each program exercises
//     from_array/to_array round-trip, dot, cosine of parallel/orthogonal/zero-
//     norm vectors, norm, add, scale, push/get/len, and the canonical
//     `Vector[..]` display. The interpreter and the native backend must produce
//     IDENTICAL results AND the native heap must be garbage-free (live == 0).
//     Programs return an Int, a Float, or the canonical Vector display String;
//     `run_native_live` compares the final value.
// ---------------------------------------------------------------------------

/// Hand-written Vector programs and their expected results (Int / Float-as-string
/// / canonical `Vector[..]` String).
fn vector_diff_programs() -> Vec<(String, &'static str)> {
    vec![
        // dot product.
        (
            "fn main() -> Float =\n\
               vec_dot(vec_from_array([1.0, 2.0, 3.0]), vec_from_array([4.0, 5.0, 6.0]))\n"
                .to_string(),
            "32",
        ),
        // L2 norm (sqrt(9 + 16) = 5).
        (
            "fn main() -> Float = vec_norm(vec_from_array([3.0, 4.0]))\n".to_string(),
            "5",
        ),
        // A bare empty array literal must infer its element type (Array[Float])
        // so the native backend accepts it, matching the interpreter. Regression
        // for an interp-vs-native divergence where `[]` fell back to Unit.
        (
            "fn main() -> Int = vec_len(vec_from_array([]))\n".to_string(),
            "0",
        ),
        // cosine of identical (parallel) vectors is 1.0.
        (
            "fn main() -> Float =\n\
               vec_cosine(vec_from_array([1.0, 0.0]), vec_from_array([1.0, 0.0]))\n"
                .to_string(),
            "1",
        ),
        // cosine of orthogonal vectors is 0.0.
        (
            "fn main() -> Float =\n\
               vec_cosine(vec_from_array([1.0, 0.0]), vec_from_array([0.0, 1.0]))\n"
                .to_string(),
            "0",
        ),
        // ZERO-NORM policy: cosine with an all-zero operand is 0.0, not NaN.
        (
            "fn main() -> Float =\n\
               vec_cosine(vec_from_array([1.0, 2.0]), vec_from_array([0.0, 0.0]))\n"
                .to_string(),
            "0",
        ),
        // elementwise add -> canonical Vector display.
        (
            "fn main() -> Vector =\n\
               vec_add(vec_from_array([1.0, 2.0, 3.0]), vec_from_array([4.0, 5.0, 6.0]))\n"
                .to_string(),
            "Vector[5, 7, 9]",
        ),
        // scale -> canonical Vector display (shortest-round-trip floats).
        (
            "fn main() -> Vector = vec_scale(vec_from_array([1.5, 2.0]), 2.0)\n".to_string(),
            "Vector[3, 4]",
        ),
        // from_array/to_array round-trip + indexing (Float through Array).
        (
            "fn main() -> Float = {\n\
               let a = vec_from_array([7.0, 8.0, 9.0]);\n\
               let xs = vec_to_array(a);\n\
               xs[1]\n\
             }\n"
                .to_string(),
            "8",
        ),
        // push / len / get (FBIP push, then read back).
        (
            "fn main() -> Float = {\n\
               let a = vec_push(vec_push(vec_new(), 1.5), 2.5);\n\
               vec_get(a, 1)\n\
             }\n"
                .to_string(),
            "2.5",
        ),
        // empty vector displays as `Vector[]`.
        (
            "fn main() -> Vector = vec_new()\n".to_string(),
            "Vector[]",
        ),
        // equality of two equal vectors (as an Int).
        (
            "fn main() -> Int = {\n\
               let a = vec_from_array([1.0, 2.0]);\n\
               let b = vec_from_array([1.0, 2.0]);\n\
               if a == b { 1 } else { 0 }\n\
             }\n"
                .to_string(),
            "1",
        ),
        // A shared vector forces copy-on-write on scale (still garbage-free):
        // the scaled copy and the original both contribute.
        (
            "fn use2(v: Vector) -> Float =\n\
               vec_get(vec_scale(v, 10.0), 0) + vec_get(v, 0)\n\
             fn main() -> Float = use2(vec_from_array([3.0, 0.0]))\n"
                .to_string(),
            // scaled copy idx0 = 30, original idx0 = 3 -> 33
            "33",
        ),
    ]
}

#[test]
fn vectors_interp_matches_compiled() {
    let progs = vector_diff_programs();
    let cc = cc_available();
    let node = node_available();
    let mut native_checked = 0u64;
    let mut wasm_ran = 0u64;
    let mut wasm_gated = 0u64;

    for (src, expected) in &progs {
        // Oracle: tree-walking interpreter.
        let interp = ast_run(src).unwrap_or_else(|e| panic!("interp failed: {}\n{}", e, src));
        assert_eq!(
            &interp, expected,
            "interpreter result mismatch\n{}\n got={} want={}",
            src, interp, expected
        );

        let prog = parser::parse(lexer::lex(src).expect("lex")).expect("parse");
        assert!(typeck::check(&prog).is_ok(), "type error\n{}", src);

        // Native (C) backend: identical result + garbage-free (live == 0).
        if cc {
            let c_src = crate::c_backend::compile(&prog)
                .unwrap_or_else(|e| panic!("c_backend failed: {}\n{}", e, src));
            let (nat, live) = run_native_live(&c_src)
                .unwrap_or_else(|e| panic!("native runner failed: {}\n{}", e, src));
            assert_eq!(nat, *expected, "native != expected\n{}\n native={}", src, nat);
            assert_eq!(live, 0, "native leaked {} cell(s)\n{}", live, src);
            native_checked += 1;
        }

        // The wasm backend now RUNS vectors fully INSIDE a program: every vec_*
        // op is real wasm. A program whose `main` RETURNS a Vector is the one
        // gated case (the shared Node harness only renders Int/Float/String — a
        // `Vector[..]` rendering needs an in-wasm float formatter, out of scope);
        // such programs are rejected cleanly (never a panic / wrong number).
        // Detect the gated case by the expected output being a `Vector[..]`
        // rendering, and assert the gate; otherwise compile, run (node-gated),
        // and require an identical result with a garbage-free heap (live == 0).
        let main_returns_vector = expected.starts_with("Vector[");
        let bytes = wasm::compile(&prog);
        if main_returns_vector {
            assert!(
                bytes.is_err(),
                "wasm should gate a Vector-returning main\n{}",
                src
            );
            let msg = bytes.unwrap_err();
            assert!(
                msg.contains("Vector return is gated") || msg.contains("Bytes or Vector return"),
                "wasm gate message should mention the gated Vector return\n{}\n got: {}",
                src,
                msg
            );
            wasm_gated += 1;
        } else {
            let bytes =
                bytes.unwrap_or_else(|e| panic!("wasm compile failed (should run): {}\n{}", e, src));
            if node {
                let (w, live) = run_wasm_live(&bytes)
                    .unwrap_or_else(|e| panic!("wasm runner failed: {}\n{}", e, src));
                assert_eq!(w, *expected, "wasm != expected\n{}\n wasm={}", src, w);
                assert_eq!(live, 0, "wasm leaked {} cell(s)\n{}", live, src);
                wasm_ran += 1;
            }
        }
    }

    eprintln!(
        "vectors_interp_matches_compiled: {} programs ({} native, {} wasm ran, {} wasm gated)",
        progs.len(),
        native_checked,
        wasm_ran,
        wasm_gated
    );
}

/// Vectors USED inside a wasm program (built, mutated FBIP, dot/cosine/norm/add/
/// scale, compared, printed element-by-element, round-tripped through an Array)
/// run fully under wasm and match the interpreter oracle byte-for-byte, with a
/// garbage-free heap (`__live == 0`). These each return a SCALAR so the shared
/// Node harness renders them — exercising the full vec_* surface that the gated
/// Vector-return cases in `vectors_interp_matches_compiled` can't show end-to-end.
/// Also covers the clean TRAP on a length mismatch / out-of-range index.
#[test]
fn vectors_run_in_wasm_matches_interp() {
    if !node_available() {
        return;
    }
    // (source, expected). Each `main` returns a SCALAR (so the Node harness
    // renders it cleanly — no intermediate prints to pollute stdout). Trap cases
    // use "TRAP".
    let cases: &[(&str, &str)] = &[
        // zero-norm cosine policy (all-zero operand) -> 0.0, never NaN.
        (
            "fn main() -> Float = vec_cosine(vec_from_array([1.0,2.0]), vec_from_array([0.0,0.0]))\n",
            "0",
        ),
        // cosine of parallel vectors = 1.
        (
            "fn main() -> Float = vec_cosine(vec_from_array([1.0,0.0]), vec_from_array([1.0,0.0]))\n",
            "1",
        ),
        // cosine of orthogonal vectors = 0.
        (
            "fn main() -> Float = vec_cosine(vec_from_array([1.0,0.0]), vec_from_array([0.0,1.0]))\n",
            "0",
        ),
        // norm: sqrt(9+16) = 5.
        (
            "fn main() -> Float = vec_norm(vec_from_array([3.0, 4.0]))\n",
            "5",
        ),
        // dot product = 32.
        (
            "fn main() -> Float = vec_dot(vec_from_array([1.0,2.0,3.0]), vec_from_array([4.0,5.0,6.0]))\n",
            "32",
        ),
        // vec_add (FBIP in-place, a unique) then read the last element back = 9.
        (
            "fn main() -> Float = {\n\
               let c = vec_add(vec_from_array([1.0,2.0,3.0]), vec_from_array([4.0,5.0,6.0]));\n\
               vec_get(c, 0) + vec_get(c, 1) + vec_get(c, 2)\n\
             }\n",
            "21",
        ),
        // scale (FBIP) then dot; 2*[1,2,3]·[4,5,6] = 64.
        (
            "fn main() -> Float =\n\
               vec_dot(vec_scale(vec_from_array([1.0,2.0,3.0]), 2.0), vec_from_array([4.0,5.0,6.0]))\n",
            "64",
        ),
        // copy-on-write: a shared vector scaled, original unchanged. 30 + 3 = 33.
        (
            "fn use2(v: Vector) -> Float = vec_get(vec_scale(v, 10.0), 0) + vec_get(v, 0)\n\
             fn main() -> Float = use2(vec_from_array([3.0, 0.0]))\n",
            "33",
        ),
        // copy-on-write on vec_add: original first element survives. a0=1, b0=11.
        (
            "fn main() -> Float = {\n\
               let a = vec_from_array([1.0, 2.0, 3.0]);\n\
               let a2 = a;\n\
               let b = vec_add(a, vec_from_array([10.0, 10.0, 10.0]));\n\
               vec_get(a2, 0) + vec_get(b, 0)\n\
             }\n",
            "12",
        ),
        // from_array/to_array round-trip + index.
        (
            "fn main() -> Float = {\n\
               let xs = vec_to_array(vec_from_array([7.0, 8.0, 9.0]));\n\
               xs[1]\n\
             }\n",
            "8",
        ),
        // FBIP push loop builds 100 ones; sum = 100.
        (
            "fn build(n: Int, acc: Vector) -> Vector =\n\
               if n == 0 { acc } else { build(n - 1, vec_push(acc, 1.0)) }\n\
             fn sumv(v: Vector, i: Int, n: Int, acc: Float) -> Float =\n\
               if i == n { acc } else { sumv(v, i + 1, n, acc + vec_get(v, i)) }\n\
             fn main() -> Float = sumv(build(100, vec_new()), 0, 100, 0.0)\n",
            "100",
        ),
        // equality holds (-> 1); a Vector never equals one of a different length.
        (
            "fn main() -> Int =\n\
               if vec_from_array([1.0,2.0]) == vec_from_array([1.0,2.0]) { 1 } else { 0 }\n",
            "1",
        ),
        // inequality (different element) -> 0.
        (
            "fn main() -> Int =\n\
               if vec_from_array([1.0,2.0]) == vec_from_array([1.0,3.0]) { 1 } else { 0 }\n",
            "0",
        ),
        // length mismatch in `==` -> not equal -> 0.
        (
            "fn main() -> Int =\n\
               if vec_from_array([1.0]) == vec_from_array([1.0,2.0]) { 1 } else { 0 }\n",
            "0",
        ),
        // length-mismatch dot -> clean TRAP.
        (
            "fn main() -> Float = vec_dot(vec_from_array([1.0,2.0]), vec_from_array([1.0,2.0,3.0]))\n",
            "TRAP",
        ),
        // out-of-range index -> clean TRAP.
        (
            "fn main() -> Float = vec_get(vec_from_array([1.0,2.0]), 5)\n",
            "TRAP",
        ),
    ];
    for (src, expected) in cases {
        // Interp oracle (skip for the trap cases, which are runtime errors).
        if *expected != "TRAP" {
            let interp = ast_run(src).unwrap_or_else(|e| panic!("interp failed: {}\n{}", e, src));
            assert_eq!(&interp, expected, "interp mismatch\n{}", src);
        } else {
            assert!(ast_run(src).is_err(), "interp should error (trap)\n{}", src);
        }
        let prog = parser::parse(lexer::lex(src).expect("lex")).expect("parse");
        assert!(typeck::check(&prog).is_ok(), "type error\n{}", src);
        let bytes = wasm::compile(&prog)
            .unwrap_or_else(|e| panic!("wasm compile failed: {}\n{}", e, src));
        let (w, live) =
            run_wasm_live(&bytes).unwrap_or_else(|e| panic!("wasm runner failed: {}\n{}", e, src));
        if *expected == "TRAP" {
            assert_eq!(w, "TRAP", "wasm should trap cleanly\n{}\n got={}", src, w);
        } else {
            assert_eq!(w, *expected, "wasm != expected\n{}\n wasm={}", src, w);
            assert_eq!(live, 0, "wasm leaked {} cell(s)\n{}", live, src);
        }
    }
    eprintln!(
        "vectors_run_in_wasm_matches_interp: {} programs ran under wasm",
        cases.len()
    );
}

/// The prelude's higher-order operations (`array_map`/`array_filter`/
/// `array_fold`/`range`) are ordinary Aria functions, so they must agree across
/// ALL THREE backends. Each program is wrapped with the real prelude (exactly as
/// the CLI does) and run through interp (oracle), native (cc-gated), and wasm
/// (node-gated); results must be identical and the native/wasm heaps garbage-free.
#[test]
fn prelude_hofs_match_across_backends() {
    let progs: &[(&str, &str)] = &[
        // map then fold: sum of x*10 over [1,2,3] = 60.
        (
            "fn main() -> Int = array_fold(array_map(array_push(array_push(array_push(array_new(), 1), 2), 3), \\x -> x * 10), 0, \\(a: Int, x: Int) -> a + x)\n",
            "60",
        ),
        // range + map + filter + fold: even squares in [0,6) = 0+4+16 = 20.
        (
            "fn main() -> Int = array_fold(array_filter(array_map(range(6), \\x -> x * x), \\x -> x % 2 == 0), 0, \\(a: Int, x: Int) -> a + x)\n",
            "20",
        ),
        // range length, and fold over a plain range: sum 0..10 = 45.
        (
            "fn main() -> Int = array_fold(range(10), 0, \\(a: Int, x: Int) -> a + x)\n",
            "45",
        ),
        // filter keeps a subset; len via fold of 1s. [0..10) > 4 -> 5 elements.
        (
            "fn main() -> Int = array_fold(array_filter(range(10), \\x -> x > 4), 0, \\(a: Int, x: Int) -> a + 1)\n",
            "5",
        ),
    ];
    let cc = cc_available();
    let node = node_available();
    let (mut nat_n, mut wasm_n) = (0u64, 0u64);
    for (user_src, expected) in progs {
        let src = crate::prelude::wrap(user_src);
        let interp = ast_run(&src).unwrap_or_else(|e| panic!("interp failed: {}\n{}", e, user_src));
        assert_eq!(&interp, expected, "interp mismatch\n{}", user_src);

        let prog = parser::parse(lexer::lex(&src).expect("lex")).expect("parse");
        assert!(typeck::check(&prog).is_ok(), "type error\n{}", user_src);

        if cc {
            let c_src = crate::c_backend::compile(&prog)
                .unwrap_or_else(|e| panic!("c_backend failed: {}\n{}", e, user_src));
            let (nat, live) = run_native_live(&c_src)
                .unwrap_or_else(|e| panic!("native runner failed: {}\n{}", e, user_src));
            assert_eq!(nat, *expected, "native != expected\n{}", user_src);
            assert_eq!(live, 0, "native leaked {} cell(s)\n{}", live, user_src);
            nat_n += 1;
        }
        if node {
            let bytes = wasm::compile(&prog)
                .unwrap_or_else(|e| panic!("wasm compile failed: {}\n{}", e, user_src));
            let (w, live) = run_wasm_live(&bytes)
                .unwrap_or_else(|e| panic!("wasm runner failed: {}\n{}", e, user_src));
            assert_eq!(w, *expected, "wasm != expected\n{}", user_src);
            assert_eq!(live, 0, "wasm leaked {} cell(s)\n{}", live, user_src);
            wasm_n += 1;
        }
    }
    eprintln!(
        "prelude_hofs_match_across_backends: {} programs ({} native, {} wasm)",
        progs.len(),
        nat_n,
        wasm_n
    );
}

/// `Array[Vector]` and `Array[Bytes]` differential: an array whose ELEMENT is a
/// tagged heap type (Vector / Bytes) must keep its PRECISE element type through
/// `array_get` (so `vec_*` / `bytes_*` ops on a retrieved element type-check in
/// the native backend, matching the interpreter) and must dup/drop each element
/// with the CORRECT kind-aware runtime function — never the generic ADT
/// `aria_drop` (a type-confusion). Interp (oracle) vs native (cc-gated): identical
/// result AND garbage-free heap (live == 0). Wasm intentionally rejects both
/// (vectors gated; a Bytes array element exceeds the wasm subset) — asserted
/// cleanly (no panic).
#[test]
fn arrays_of_tagged_heap_elems_interp_matches_compiled() {
    let cc = cc_available();
    // (source, expected interp result). Each builds an array of 2-3 tagged-heap
    // elements, retrieves elements with `array_get`, and operates on them.
    let cases: &[(&str, &str)] = &[
        // --- Array[Vector]: an embedding store. The headline case — the same
        //     `vec_cosine(array_get(vs, i), array_get(vs, j))` the issue cited. ---
        // cosine of two stored, orthogonal embeddings = 0.
        (
            "fn main() -> Float = {\n\
               let vs: Array[Vector] = [vec_from_array([1.0, 0.0]), vec_from_array([0.0, 1.0])];\n\
               vec_cosine(array_get(vs, 0), array_get(vs, 1))\n\
             }\n",
            "0",
        ),
        // dot of two stored embeddings (1*4 + 2*5 + 3*6 = 32).
        (
            "fn main() -> Float = {\n\
               let vs: Array[Vector] = [vec_from_array([1.0, 2.0, 3.0]), vec_from_array([4.0, 5.0, 6.0])];\n\
               vec_dot(array_get(vs, 0), array_get(vs, 1))\n\
             }\n",
            "32",
        ),
        // vec_add of two stored embeddings, then norm (||[4,6]|| = sqrt(52)).
        (
            "fn main() -> Float = {\n\
               let vs: Array[Vector] = [vec_from_array([1.0, 2.0]), vec_from_array([3.0, 4.0])];\n\
               vec_norm(vec_add(array_get(vs, 0), array_get(vs, 1)))\n\
             }\n",
            "7.211102550927978",
        ),
        // array_len + per-element vec_len of a 3-embedding store.
        (
            "fn main() -> Int = {\n\
               let vs: Array[Vector] = [vec_from_array([1.0]), vec_from_array([1.0, 2.0]), vec_from_array([1.0, 2.0, 3.0])];\n\
               array_len(vs) * 100 + vec_len(array_get(vs, 2)) * 10 + vec_len(array_get(vs, 0))\n\
             }\n",
            "331",
        ),
        // --- Array[Bytes] ---
        // build two byte buffers, get one, bytes_len + bytes_get it.
        (
            "fn main() -> Int = {\n\
               let bs: Array[Bytes] = [bytes_push(bytes_push(bytes_new(), 10), 20), bytes_push(bytes_new(), 99)];\n\
               array_len(bs) * 1000 + bytes_len(array_get(bs, 0)) * 100 + bytes_get(array_get(bs, 0), 1) * 10 + bytes_get(array_get(bs, 1), 0) / 11\n\
             }\n",
            "2409",
        ),
    ];

    let mut native_checked = 0u64;
    for (src, expected) in cases {
        // Oracle: tree-walking interpreter.
        let interp = ast_run(src).unwrap_or_else(|e| panic!("interp failed: {}\n{}", e, src));
        assert_eq!(
            &interp, expected,
            "interpreter result mismatch\n{}\n got={} want={}",
            src, interp, expected
        );

        let prog = parser::parse(lexer::lex(src).expect("lex")).expect("parse");
        assert!(typeck::check(&prog).is_ok(), "type error\n{}", src);

        // Native (C) backend: identical result + garbage-free (live == 0).
        if cc {
            let c_src = crate::c_backend::compile(&prog)
                .unwrap_or_else(|e| panic!("c_backend failed: {}\n{}", e, src));
            let (nat, live) = run_native_live(&c_src)
                .unwrap_or_else(|e| panic!("native runner failed: {}\n{}", e, src));
            assert_eq!(nat, *expected, "native != expected\n{}\n native={}", src, nat);
            assert_eq!(live, 0, "native leaked {} cell(s)\n{}", live, src);
            native_checked += 1;
        }

        // The wasm backend must cleanly REJECT both (vector gate / Bytes array
        // element outside the subset) — never compile or panic.
        let bytes = wasm::compile(&prog);
        assert!(
            bytes.is_err(),
            "wasm backend unexpectedly accepted a tagged-heap array program\n{}",
            src
        );
    }

    eprintln!(
        "arrays_of_tagged_heap_elems_interp_matches_compiled: {} programs ({} native)",
        cases.len(),
        native_checked
    );
}

/// Error cases (length mismatch on dot/add/cosine, OOB vec_get) must surface as a
/// CLEAN runtime error in the interpreter AND a CLEAN trap (`abort` -> "TRAP", no
/// panic) in the native backend — never a panic or silent wrong answer.
#[test]
fn vector_error_cases_clean_on_interp_and_native() {
    let cc = cc_available();
    let cases = [
        // length mismatch on dot.
        "fn main() -> Float = vec_dot(vec_from_array([1.0, 2.0]), vec_from_array([1.0]))\n",
        // length mismatch on add.
        "fn main() -> Vector = vec_add(vec_from_array([1.0, 2.0]), vec_from_array([1.0]))\n",
        // length mismatch on cosine.
        "fn main() -> Float = vec_cosine(vec_from_array([1.0, 2.0]), vec_from_array([1.0]))\n",
        // OOB vec_get.
        "fn main() -> Float = vec_get(vec_from_array([1.0, 2.0]), 5)\n",
    ];
    for src in cases {
        // Interpreter: a clean Err (no panic).
        assert!(
            ast_run(src).is_err(),
            "interpreter should reject with a clean error\n{}",
            src
        );
        // Native: compiles, then TRAPs at run time (clean abort), live accounting
        // intact (the trap aborts before the leak check, surfaced as "TRAP").
        if cc {
            let prog = parser::parse(lexer::lex(src).expect("lex")).expect("parse");
            assert!(typeck::check(&prog).is_ok(), "type error\n{}", src);
            let c_src = crate::c_backend::compile(&prog)
                .unwrap_or_else(|e| panic!("c_backend failed: {}\n{}", e, src));
            let (nat, _live) = run_native_live(&c_src)
                .unwrap_or_else(|e| panic!("native runner failed: {}\n{}", e, src));
            assert_eq!(nat, "TRAP", "native should trap cleanly\n{}\n got={}", src, nat);
        }
    }
}
