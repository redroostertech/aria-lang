//! Aria CLI.
//!
//! Usage:
//!   aria run   <file.aria>          parse and execute `main`
//!   aria ast   <file.aria>          print the parsed AST (debugging)
//!   aria pack  <in> <out>           compress any file (rANS entropy coder)
//!   aria unpack <in> <out>          decompress an Aria-packed file
//!   aria bench                      run the compression benchmark
//!   aria wasm   <file.aria> <out>   compile the pure Int/Bool subset to wasm
//!   aria wasm-run <file.aria>       compile to wasm and run it via Node
//!   aria native <file.aria> <out>   transpile to C and build a native exe via cc
//!   aria native-run <file.aria>     transpile to C, build, run, print the result
//!   aria gbnf  [<file.out>]         emit a GBNF grammar for Aria's syntax

// Many runtime modules expose library-style APIs not all wired into the CLI yet.
#![allow(dead_code)]

mod arith;
mod ast;
mod builtins;
mod c_backend;
mod gbnf;
mod interp;
mod ir;
mod lexer;
mod monomorphize;
mod neural_codec;
mod pack;
mod parser;
mod predict;
mod prelude;
#[cfg(test)]
mod proptest;
mod rag;
mod rc;
mod rans;
mod shape;
mod tensor;
mod tensor_ext;
mod traits;
mod transformer;
mod typeck;
mod wasm;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: aria <run|check|ast|pack|unpack|npack|nunpack|bench|demo|mem|wasm|wasm-run|native|native-run|gbnf> [args...]");
        return ExitCode::from(2);
    }

    match args[1].as_str() {
        "run" | "ast" | "check" => run_source(&args),
        "pack" => pack_file(&args, Codec::Rans, true),
        "unpack" => pack_file(&args, Codec::Rans, false),
        "npack" => pack_file(&args, Codec::Neural, true),
        "nunpack" => pack_file(&args, Codec::Neural, false),
        "bench" => {
            pack::bench();
            ExitCode::SUCCESS
        }
        "demo" => run_demo(args.get(2).map(|s| s.as_str())),
        "mem" => run_mem(&args),
        "wasm" => run_wasm_compile(&args),
        "wasm-run" => run_wasm_run(&args),
        "native" => run_native_compile(&args),
        "native-run" => run_native_run(&args),
        "gbnf" => run_gbnf(&args),
        other => {
            eprintln!("unknown command `{}`", other);
            ExitCode::from(2)
        }
    }
}

/// Lower a program to the IR, run the IR interpreter, and report heap-allocation
/// metrics. (Stage 1 of the memory-model work — dup/drop + reuse come next.)
fn run_mem(args: &[String]) -> ExitCode {
    if args.len() < 3 {
        eprintln!("usage: aria mem <file.aria>");
        return ExitCode::from(2);
    }
    let src = match std::fs::read_to_string(&args[2]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", args[2], e);
            return ExitCode::from(2);
        }
    };
    let prog = match lexer::lex(&prelude::wrap(&src)).and_then(parser::parse) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(errs) = typeck::check(&prog) {
        for e in errs {
            eprintln!("type error: {}", e);
        }
        return ExitCode::FAILURE;
    }
    let fns = match ir::lower_program(&prog) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("lowering error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    // Insert precise reference-count operations (Stage 2).
    let fns = rc::insert_rc(&fns);

    // Run both the IR and the tree-walking interpreter on a large-stack thread
    // and cross-check them, so `aria mem` can never silently report a result the
    // two backends disagree on.
    let outcome = std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || {
            let mut runner = ir::IrInterp::new(fns);
            let ir_res = runner.run_main().map(|v| runner.render(&v));
            let metrics = runner.metrics.clone();
            let ast_res = match interp::Interp::new(&prog) {
                Ok(it) => it.run_main().map(|v| v.display()),
                Err(e) => Err(e),
            };
            (ir_res, ast_res, metrics)
        })
        .expect("spawn mem thread")
        .join()
        .unwrap_or_else(|_| (Err("ir thread panicked".into()), Err("".into()), ir::Metrics::default()));

    let (ir_res, ast_res, m) = outcome;
    match (&ir_res, &ast_res) {
        (Ok(ir), Ok(ast)) if ir == ast => {
            let gross = m.allocations + m.reuses;
            eprintln!("ir == interpreter: {} (agree)", ir);
            eprintln!(
                "fresh allocations: {}  in-place reuses: {}  gross (no reuse): {}",
                m.allocations, m.reuses, gross
            );
            eprintln!(
                "frees: {}  peak live: {}  (dups: {}, drops: {})",
                m.frees, m.peak_live, m.dups, m.drops
            );
            if m.live == 0 {
                eprintln!("garbage-free: yes (no cells live at exit)");
                if m.reuses > 0 && gross > 0 {
                    eprintln!(
                        "reuse eliminated {:.1}% of allocations ({} of {})",
                        100.0 * m.reuses as f64 / gross as f64,
                        m.reuses,
                        gross
                    );
                }
            } else {
                eprintln!("{} cell(s) still live (reachable from the result)", m.live);
            }
            ExitCode::SUCCESS
        }
        (Ok(ir), Ok(ast)) => {
            eprintln!("DIVERGENCE: ir result {:?} != interpreter result {:?}", ir, ast);
            ExitCode::FAILURE
        }
        (Err(e), _) => {
            eprintln!("ir runtime error: {}", e);
            ExitCode::FAILURE
        }
        (Ok(_), Err(e)) => {
            eprintln!("interpreter error (ir succeeded): {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Parse + type-check a file and compile it to a WebAssembly binary (subset 2a).
fn compile_to_wasm(path: &str) -> Result<Vec<u8>, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    let prog = lexer::lex(&prelude::wrap(&src)).and_then(parser::parse)?;
    typeck::check(&prog).map_err(|errs| errs.join("; "))?;
    wasm::compile(&prog)
}

/// Coarsely classify `main`'s declared return type for the wasm-run harness:
/// `"str"` for a heap String result (which is an i32 pointer the harness must
/// decode + free), `"scalar"` otherwise (Int/Float come back as a JS
/// bigint/number directly). Best-effort: any parse hiccup falls back to scalar.
fn main_return_kind(path: &str) -> String {
    let kind = (|| -> Option<String> {
        let src = std::fs::read_to_string(path).ok()?;
        let prog = lexer::lex(&prelude::wrap(&src)).and_then(parser::parse).ok()?;
        let main = prog.items.iter().find_map(|it| match it {
            ast::Item::Fn(f) if f.name == "main" => Some(f),
            _ => None,
        })?;
        Some(match &main.ret {
            ast::Ty::Str => "str".to_string(),
            _ => "scalar".to_string(),
        })
    })();
    kind.unwrap_or_else(|| "scalar".to_string())
}

/// `aria wasm <file.aria> <out.wasm>`: compile and write the wasm binary.
fn run_wasm_compile(args: &[String]) -> ExitCode {
    if args.len() < 4 {
        eprintln!("usage: aria wasm <file.aria> <out.wasm>");
        return ExitCode::from(2);
    }
    let bytes = match compile_to_wasm(&args[2]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::write(&args[3], &bytes) {
        eprintln!("error: cannot write {}: {}", args[3], e);
        return ExitCode::FAILURE;
    }
    eprintln!("wrote {} ({} bytes)", args[3], bytes.len());
    ExitCode::SUCCESS
}

/// `aria wasm-run <file.aria>`: compile to a temp .wasm, run it under Node, and
/// print `main`'s result (or `TRAP` on a wasm trap such as div-by-zero).
fn run_wasm_run(args: &[String]) -> ExitCode {
    if args.len() < 3 {
        eprintln!("usage: aria wasm-run <file.aria>");
        return ExitCode::from(2);
    }
    let bytes = match compile_to_wasm(&args[2]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    // Determine `main`'s return kind so the Node harness decodes the result
    // correctly (Int/Float come back as JS bigint/number; a String/Map/Set is an
    // i32 heap pointer — also a JS number, so they cannot be told apart by JS
    // type alone). `"str"` => decode the heap String and free it (so `__live`
    // reaches 0, matching native dropping its printed result).
    let main_ret = main_return_kind(&args[2]);
    let path = std::env::temp_dir().join(format!("aria_{}.wasm", std::process::id()));
    if let Err(e) = std::fs::write(&path, &bytes) {
        eprintln!("error: cannot write temp wasm: {}", e);
        return ExitCode::FAILURE;
    }
    // Run `main`, print its result to stdout, and (for Phase 2b heap programs
    // that export `__live`) report the live-cell count on stderr so leaks are
    // visible. A wasm trap surfaces as `TRAP`, matching the interpreter's `Err`.
    let script = format!(
        "const fs=require('fs');\
         const dec=new TextDecoder();\
         let memref=null;\
         const decodeStr=(p)=>{{const mem=new Uint8Array(memref.buffer);\
         const dv=new DataView(memref.buffer);\
         const len=Number(dv.getBigInt64(p+8,true));\
         return dec.decode(mem.subarray(p+16,p+16+len));}};\
         const fmtFloat=(x)=>{{\
         if(Number.isNaN(x)){{return 'NaN';}}\
         if(x===Infinity){{return 'inf';}}\
         if(x===-Infinity){{return '-inf';}}\
         if(x===0&&1/x===-Infinity){{return '-0';}}\
         return String(x);}};\
         const imp={{env:{{print_str:(p,n)=>{{\
         const mem=new Uint8Array(memref.buffer);\
         process.stdout.write(dec.decode(mem.subarray(p,p+n)));\
         process.stdout.write('\\n');}},\
         print_float:(x)=>{{process.stdout.write(fmtFloat(x));\
         process.stdout.write('\\n');}},\
         print_int:(n)=>{{process.stdout.write(String(n));\
         process.stdout.write('\\n');}},\
         print_bool:(b)=>{{process.stdout.write(b?'true':'false');\
         process.stdout.write('\\n');}},\
         exp:Math.exp}}}};\
         try{{const b=fs.readFileSync({:?});\
         WebAssembly.instantiate(b,imp).then(r=>{{\
         try{{const ex=r.instance.exports;memref=ex.memory;\
         const isStr={};\
         const v=ex.main();\
         if(isStr){{process.stdout.write(decodeStr(v));\
         if(ex.__drop_str){{ex.__drop_str(v);}}}}\
         else if(typeof v==='bigint'){{process.stdout.write(String(v));}}\
         else{{process.stdout.write(String(v));}}\
         if(ex.__live){{process.stderr.write('__live='+String(ex.__live()));}}\
         if(ex.__reuses){{process.stderr.write(' __reuses='+String(ex.__reuses()));}}}}\
         catch(e){{process.stdout.write('TRAP');process.exitCode=1;}}\
         }}).catch(e=>{{process.stdout.write('TRAP');process.exitCode=1;}});}}\
         catch(e){{process.stdout.write('TRAP');process.exitCode=1;}}",
        path.to_string_lossy(),
        if main_ret == "str" { "true" } else { "false" }
    );
    // `--stack-size` raises V8's stack so compiled wasm can recurse as deeply as
    // the interpreter / native backend (wasm under Node's default stack traps on
    // ~20k-deep non-tail recursion — a runtime stack limit, not a miscompilation).
    let out = std::process::Command::new("node")
        .arg("--stack-size=8000")
        .arg("-e")
        .arg(&script)
        .output();
    let _ = std::fs::remove_file(&path);
    match out {
        Ok(o) if o.status.success() => {
            use std::io::Write;
            let _ = std::io::stdout().write_all(&o.stdout);
            println!();
            // Forward the `__live=` diagnostic (if any) to stderr.
            if !o.stderr.is_empty() {
                eprintln!("{}", String::from_utf8_lossy(&o.stderr));
            }
            ExitCode::SUCCESS
        }
        Ok(o) => {
            // A wasm trap (e.g. overflow / div-by-zero / array OOB): the harness
            // wrote partial output + `TRAP` to stdout and set a non-zero exit.
            // Surface that (and any stderr) and fail, matching native/interp.
            use std::io::Write;
            let _ = std::io::stdout().write_all(&o.stdout);
            if !o.stdout.is_empty() {
                println!();
            }
            if !o.stderr.is_empty() {
                eprintln!("{}", String::from_utf8_lossy(&o.stderr));
            }
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("error: could not run node (is it installed?): {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Parse + type-check a file and transpile it to C source (native backend).
fn compile_to_c(path: &str) -> Result<String, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    let prog = lexer::lex(&prelude::wrap(&src)).and_then(parser::parse)?;
    typeck::check(&prog).map_err(|errs| errs.join("; "))?;
    c_backend::compile(&prog)
}

/// Invoke the system C compiler `cc` to build `c_src` into the executable at
/// `out_path`. Returns the `cc` stderr on failure.
fn cc_build(c_src: &str, out_path: &std::path::Path) -> Result<(), String> {
    let c_path = std::env::temp_dir().join(format!("aria_{}.c", std::process::id()));
    std::fs::write(&c_path, c_src).map_err(|e| format!("cannot write temp C: {}", e))?;
    let status = std::process::Command::new("cc")
        .arg("-O2")
        .arg("-std=c11")
        .arg("-o")
        .arg(out_path)
        .arg(&c_path)
        // Link libm for the math functions used by the Vector runtime (sqrt).
        .arg("-lm")
        .output();
    let _ = std::fs::remove_file(&c_path);
    match status {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(format!("cc failed: {}", String::from_utf8_lossy(&o.stderr))),
        Err(e) => Err(format!("could not run cc (is a C compiler installed?): {}", e)),
    }
}

/// `aria native <file.aria> <out>`: transpile to C and build a native exe.
fn run_native_compile(args: &[String]) -> ExitCode {
    if args.len() < 4 {
        eprintln!("usage: aria native <file.aria> <out>");
        return ExitCode::from(2);
    }
    let c_src = match compile_to_c(&args[2]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let out = std::path::Path::new(&args[3]);
    if let Err(e) = cc_build(&c_src, out) {
        eprintln!("error: {}", e);
        return ExitCode::FAILURE;
    }
    eprintln!("built {}", args[3]);
    ExitCode::SUCCESS
}

/// `aria native-run <file.aria>`: transpile to C, build in a temp dir, run, and
/// print `main`'s result. The native live-cell count is forwarded on stderr.
fn run_native_run(args: &[String]) -> ExitCode {
    if args.len() < 3 {
        eprintln!("usage: aria native-run <file.aria>");
        return ExitCode::from(2);
    }
    let c_src = match compile_to_c(&args[2]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let exe = std::env::temp_dir().join(format!("aria_exe_{}", std::process::id()));
    if let Err(e) = cc_build(&c_src, &exe) {
        eprintln!("error: {}", e);
        return ExitCode::FAILURE;
    }
    let out = std::process::Command::new(&exe).output();
    let _ = std::fs::remove_file(&exe);
    match out {
        Ok(o) => {
            use std::io::Write;
            let _ = std::io::stdout().write_all(&o.stdout);
            if !o.stderr.is_empty() {
                eprint!("{}", String::from_utf8_lossy(&o.stderr));
            }
            if o.status.success() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("error: could not run native executable: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// `aria gbnf [<file.out>]`: emit a GBNF grammar describing Aria's concrete
/// syntax. With no path the grammar is printed to stdout; with a path it is
/// written there. The grammar is real GBNF (usable by llama.cpp) so an LLM
/// constrained-decoding against it cannot emit an Aria syntax error.
fn run_gbnf(args: &[String]) -> ExitCode {
    let g = gbnf::grammar();
    if let Some(out) = args.get(2) {
        if let Err(e) = std::fs::write(out, &g) {
            eprintln!("error: cannot write {}: {}", out, e);
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {} ({} bytes)", out, g.len());
    } else {
        print!("{}", g);
    }
    ExitCode::SUCCESS
}

fn run_demo(which: Option<&str>) -> ExitCode {
    match which {
        Some("transformer") => transformer::demo(),
        Some("predict") => predict::demo(),
        Some("shape") => shape::demo(),
        Some("rag") => rag::demo(),
        Some(other) => {
            eprintln!("unknown demo `{}` (try: transformer, predict, shape, rag)", other);
            return ExitCode::from(2);
        }
        None => {
            println!("== transformer ==");
            transformer::demo();
            println!("\n== predict ==");
            predict::demo();
            println!("\n== shape ==");
            shape::demo();
            println!("\n== rag ==");
            rag::demo();
        }
    }
    ExitCode::SUCCESS
}

fn run_source(args: &[String]) -> ExitCode {
    if args.len() < 3 {
        eprintln!("usage: aria {} <file.aria>", args[1]);
        return ExitCode::from(2);
    }
    let path = &args[2];
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path, e);
            return ExitCode::from(2);
        }
    };

    let toks = match lexer::lex(&prelude::wrap(&src)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("lex error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let program = match parser::parse(toks) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("parse error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if args[1] == "ast" {
        println!("{:#?}", program);
        return ExitCode::SUCCESS;
    }

    // Type-check before running — the compiler is the correctness signal.
    if let Err(errors) = typeck::check(&program) {
        eprintln!("type error{} in {}:", if errors.len() == 1 { "" } else { "s" }, path);
        for e in &errors {
            eprintln!("  - {}", e);
        }
        return ExitCode::FAILURE;
    }

    if args[1] == "check" {
        println!("{}: type-checks OK", path);
        return ExitCode::SUCCESS;
    }

    let interp = match interp::Interp::new(&program) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    // Run on a large-stack thread: the tree-walking interpreter uses native
    // stack per Aria call, so deep (but finite) recursion won't overflow; the
    // interpreter's own depth guard catches genuinely infinite recursion.
    let result = std::thread::Builder::new()
        .stack_size(1 << 30) // 1 GiB
        .spawn(move || interp.run_main())
        .expect("spawn interpreter thread")
        .join()
        .unwrap_or_else(|_| Err("interpreter thread panicked".into()));
    match result {
        Ok(v) => {
            // Print `main`'s result value, matching the native backend
            // (`aria native-run`) so a program's full output is identical across
            // backends. An `Int` result additionally maps to the process exit
            // code (low 8 bits), preserving the historical convention.
            println!("{}", v.display());
            match v {
                interp::Value::Int(n) => ExitCode::from((n & 0xff) as u8),
                _ => ExitCode::SUCCESS,
            }
        }
        Err(e) => {
            eprintln!("runtime error: {}", e);
            ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Copy)]
enum Codec {
    Rans,
    Neural,
}

fn pack_file(args: &[String], codec: Codec, compress: bool) -> ExitCode {
    if args.len() < 4 {
        eprintln!("usage: aria {} <in> <out>", args[1]);
        return ExitCode::from(2);
    }
    let input = match std::fs::read(&args[2]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", args[2], e);
            return ExitCode::from(2);
        }
    };
    let decompressed = if compress {
        Ok(match codec {
            Codec::Rans => rans::compress(&input),
            Codec::Neural => neural_codec::compress(&input),
        })
    } else {
        match codec {
            Codec::Rans => rans::decompress(&input),
            Codec::Neural => neural_codec::decompress(&input),
        }
    };
    let output = match decompressed {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::write(&args[3], &output) {
        eprintln!("error: cannot write {}: {}", args[3], e);
        return ExitCode::FAILURE;
    }
    if compress {
        let ratio = 100.0 * output.len() as f64 / input.len().max(1) as f64;
        eprintln!(
            "packed {} -> {} bytes ({:.1}% of original)",
            input.len(),
            output.len(),
            ratio
        );
    } else {
        eprintln!("unpacked {} -> {} bytes", input.len(), output.len());
    }
    ExitCode::SUCCESS
}
