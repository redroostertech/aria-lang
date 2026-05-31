//! Aria CLI.
//!
//! Usage:
//!   aria run   <file.aria>          parse and execute `main`
//!   aria ast   <file.aria>          print the parsed AST (debugging)
//!   aria pack  <in> <out>           compress any file (rANS entropy coder)
//!   aria unpack <in> <out>          decompress an Aria-packed file
//!   aria bench                      run the compression benchmark

// Many runtime modules expose library-style APIs not all wired into the CLI yet.
#![allow(dead_code)]

mod arith;
mod ast;
mod builtins;
mod interp;
mod lexer;
mod neural_codec;
mod pack;
mod parser;
mod predict;
mod rag;
mod rans;
mod shape;
mod tensor;
mod tensor_ext;
mod transformer;
mod typeck;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: aria <run|check|ast|pack|unpack|npack|nunpack|bench|demo> [args...]");
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
        other => {
            eprintln!("unknown command `{}`", other);
            ExitCode::from(2)
        }
    }
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

    let toks = match lexer::lex(&src) {
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
        Ok(interp::Value::Int(n)) => ExitCode::from((n & 0xff) as u8),
        Ok(_) => ExitCode::SUCCESS,
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
