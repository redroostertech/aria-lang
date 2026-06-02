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
            // Restricted universe: Int/Bool/IntList AND String (Phase 2d). No
            // Float, which stays outside the wasm backend's compilable subset.
            return match self.rng.below(4) {
                0 => Ty::Int,
                1 => Ty::Bool,
                2 => Ty::Str,
                _ => Ty::List,
            };
        }
        match self.rng.below(5) {
            0 => Ty::Int,
            1 => Ty::Bool,
            2 => Ty::Float,
            3 => Ty::Str,
            _ => Ty::List,
        }
    }

    fn gen_int(&mut self, fuel: u32) -> String {
        // Weighted toward leaves to keep programs small; `below` picks a rule.
        match self.rng.below(9) {
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
         print_bool:(b)=>{{process.stdout.write(b?'true':'false');process.stdout.write('\\n');}}}}}};\
         const b=fs.readFileSync({:?});\
         WebAssembly.instantiate(b,imp).then(r=>{{\
         const ex=r.instance.exports;memref=ex.memory;\
         const v=ex.main();\
         let m;if(typeof v==='bigint'){{m=String(v);}}\
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
