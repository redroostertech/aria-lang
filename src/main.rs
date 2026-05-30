//! Aria CLI.
//!
//! Usage:
//!   aria run <file.aria>     parse and execute `main`
//!   aria ast <file.aria>     print the parsed AST (debugging)

mod ast;
mod interp;
mod lexer;
mod parser;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: aria <run|ast> <file.aria>");
        return ExitCode::from(2);
    }
    let cmd = &args[1];
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

    match cmd.as_str() {
        "ast" => {
            println!("{:#?}", program);
            ExitCode::SUCCESS
        }
        "run" => {
            let interp = match interp::Interp::new(&program) {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("error: {}", e);
                    return ExitCode::FAILURE;
                }
            };
            match interp.run_main() {
                Ok(v) => {
                    // `main` returning an Int sets the process exit code.
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
        other => {
            eprintln!("unknown command `{}` (use run or ast)", other);
            ExitCode::from(2)
        }
    }
}
