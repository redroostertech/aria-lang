//! `aria agent` — a provider-agnostic AI authoring loop for Aria.
//!
//! This is the *authoring half* of the AI-native thesis: an LLM (local, cloud,
//! or an agentic CLI) writes an Aria program, the compiler CHECKS it, and the
//! STRUCTURED DIAGNOSTICS (`typeck::check_structured`, the same channel as
//! `aria check --json`) are fed back so the model FIXES it — looping until the
//! program type-checks and runs. The compiler is the ground-truth feedback;
//! the loop converges to working code.
//!
//! The loop, end to end:
//!
//! ```text
//!   ┌─ PRIMER + TASK ────────────────────────────────────────────────┐
//!   │  assemble a tight, accurate Aria primer + the user's task +     │
//!   │  "return ONLY a complete Aria program"                          │
//!   └────────────────────────────────────────────────────────────────┘
//!                              │  prompt
//!                              ▼
//!                 ┌───────────────────────┐
//!                 │   PROVIDER.complete    │  (mock / cmd / claude / codex /
//!                 │   raw model text       │   llama / anthropic / openai)
//!                 └───────────────────────┘
//!                              │  raw text
//!                              ▼
//!                 ┌───────────────────────┐
//!                 │   EXTRACT program      │  strip ``` fences / prose;
//!                 │   (pick the `fn main`) │  pick the `fn main` block
//!                 └───────────────────────┘
//!                              │  program
//!                              ▼
//!                 ┌───────────────────────┐   diagnostics
//!                 │  check_structured()    │ ───────────────┐
//!                 └───────────────────────┘                │ FEEDBACK
//!                       │ clean                            ▼  (append to
//!                       ▼                          ┌──────────────┐  transcript,
//!                 ┌───────────────────────┐        │  build error │  loop back to
//!                 │  interp run_main()     │        │  message     │  PROVIDER)
//!                 │  SUCCESS: program +    │        └──────────────┘
//!                 │  result + iterations   │
//!                 └───────────────────────┘
//! ```
//!
//! SAFETY BY CONSTRUCTION. The loop RUNS model-generated Aria via the in-process
//! tree-walking interpreter (`interp::Interp`). This is safe to do with
//! UNTRUSTED LLM output because Aria has **no I/O, no FFI, no network, and no
//! filesystem access** — the only "effects" are the `print_*` builtins (write to
//! this process's stdout) and pure computation. There is no `rm -rf`, no
//! exfiltration, no shelling out: executing a hallucinated program cannot harm
//! the host. The one residual cost is unbounded computation (a program can
//! loop/recurse), and that is bounded by the interpreter's call-depth guard
//! (`MAX_CALL_DEPTH` = 100_000 Aria frames) which turns runaway recursion into a
//! clean `Err` rather than a crash. For the guard to win the race against a
//! native stack overflow, the interpreter must run on a thread whose stack can
//! hold that many frames: each Aria call consumes native stack, and worst-case
//! non-tail recursion costs ~26 KiB/frame in an UNOPTIMIZED (debug) build, so
//! 100k frames need ~2.6 GiB. The CLI therefore reserves a 4 GiB interpreter
//! stack (`main::INTERP_STACK_SIZE`) so the depth guard fires cleanly in BOTH
//! debug and release before any `stack overflow`/`Abort trap`. (Tail recursion
//! is unbounded regardless: TCO reuses one frame, so it never grows the stack and
//! is intentionally NOT subject to the depth guard.) This is a genuine,
//! honestly-stated advantage of an effect-free language for AI authoring.
//!
//! NOTE on real providers: the `cmd:` / preset providers shell out to external
//! tools (`claude`, `codex`, `llama-cli`, `curl`). Those are NOT invoked in the
//! test suite (they may be unavailable or recursive); the tests cover command
//! CONSTRUCTION and drive the full loop with the deterministic built-in `mock`.

use crate::diagnostics::{array_to_json, Diagnostic};
use crate::{interp, lexer, parser, prelude, typeck};

/// A completion provider: turn a prompt into the model's raw text response.
/// Errors are plain strings (never panics) — a failed provider fails the loop
/// gracefully rather than crashing.
pub trait Provider: Send {
    fn complete(&self, prompt: &str) -> Result<String, String>;
    /// Short human label for transcripts/reporting.
    fn label(&self) -> String;
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

/// The universal escape hatch: run `sh -c "<shell>"`, write the PROMPT to the
/// child's STDIN, read its STDOUT as the response. One mechanism covers a
/// curl-to-cloud command, a local `llama-cli` invocation, or an agentic CLI.
///
/// `grammar_file`, when set, is a temp file holding `gbnf::grammar()` that a
/// preset (`llama:`) referenced in its command for constrained decoding; we keep
/// the handle so the file outlives the run. It has no effect on execution here.
pub struct CmdProvider {
    pub shell: String,
    pub name: String,
    /// Kept alive for the provider's lifetime (constrained-decoding grammar).
    _grammar_file: Option<TempFile>,
}

impl CmdProvider {
    pub fn new(shell: impl Into<String>) -> CmdProvider {
        let shell = shell.into();
        CmdProvider { name: format!("cmd:{}", shell), shell, _grammar_file: None }
    }

    fn named(name: impl Into<String>, shell: impl Into<String>) -> CmdProvider {
        CmdProvider { name: name.into(), shell: shell.into(), _grammar_file: None }
    }
}

impl Provider for CmdProvider {
    fn complete(&self, prompt: &str) -> Result<String, String> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new("sh")
            .arg("-c")
            .arg(&self.shell)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("could not spawn provider `{}`: {}", self.shell, e))?;

        // Write the prompt to the child's stdin, then close it so the child sees
        // EOF. Take the handle out so the `child` borrow ends before `wait`.
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(prompt.as_bytes())
                .map_err(|e| format!("could not write prompt to provider: {}", e))?;
            // Dropping `stdin` here closes the pipe (EOF for the child).
        }

        let out = child
            .wait_with_output()
            .map_err(|e| format!("provider `{}` failed: {}", self.shell, e))?;
        if !out.status.success() {
            return Err(format!(
                "provider `{}` exited with {}: {}",
                self.shell,
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn label(&self) -> String {
        self.name.clone()
    }
}

/// Built-in DETERMINISTIC provider that PROVES the loop works with no model.
///
/// Call 1 returns an Aria program with a deliberate, check-detectable error
/// (`main` returns a `String` where it declared `Int`). Once it has been fed
/// FEEDBACK containing the diagnostics (i.e. a later call whose prompt mentions
/// our error channel), it returns the CORRECTED program — so the test proves the
/// diagnostics drove the fix. As a fallback it advances by call count.
pub struct MockProvider {
    /// Number of times `complete` has been called (interior mutability so the
    /// trait's `&self` signature is honoured).
    calls: std::cell::Cell<usize>,
}

impl MockProvider {
    pub fn new() -> MockProvider {
        MockProvider { calls: std::cell::Cell::new(0) }
    }

    /// First attempt: declares `-> Int` but the body is a `String` — a clean
    /// `E0201` type mismatch the checker catches.
    pub fn buggy_program() -> &'static str {
        "fn sum_to(n: Int) -> Int =\n  \
           if n == 0 { 0 } else { n + sum_to(n - 1) }\n\n\
         fn main() -> Int = {\n  \
           print_int(sum_to(10));\n  \
           \"done\"\n}\n"
    }

    /// Corrected attempt: `main` now returns an `Int`. Checks clean and runs.
    pub fn fixed_program() -> &'static str {
        "fn sum_to(n: Int) -> Int =\n  \
           if n == 0 { 0 } else { n + sum_to(n - 1) }\n\n\
         fn main() -> Int = {\n  \
           print_int(sum_to(10));\n  \
           sum_to(10)\n}\n"
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for MockProvider {
    fn complete(&self, prompt: &str) -> Result<String, String> {
        let n = self.calls.get();
        self.calls.set(n + 1);
        // Drive the fix off having RECEIVED feedback (the feedback message
        // embeds the diagnostics array and the phrase below), so the test proves
        // the structured diagnostics caused the correction. Fall back to call
        // count for the very first call (no feedback yet).
        let got_feedback = prompt.contains("Your program had these errors");
        if n == 0 && !got_feedback {
            // Wrap in a fence + prose to also exercise the extractor.
            Ok(format!(
                "Here is the program:\n```aria\n{}```\n",
                Self::buggy_program()
            ))
        } else {
            Ok(format!("```aria\n{}```", Self::fixed_program()))
        }
    }

    fn label(&self) -> String {
        "mock".to_string()
    }
}

/// A fixed-program provider: every `complete` call returns the SAME program,
/// fenced so it also exercises the extractor. The benchmark constructs one of
/// these per task from that task's `reference` solution, so `--provider
/// reference` drives the whole agent loop + grader OFFLINE (no model) and must
/// converge + grade-correct in a single iteration — the end-to-end self-test.
pub struct FixedProvider {
    program: String,
    name: String,
}

impl FixedProvider {
    pub fn new(program: impl Into<String>, name: impl Into<String>) -> FixedProvider {
        FixedProvider { program: program.into(), name: name.into() }
    }
}

impl Provider for FixedProvider {
    fn complete(&self, _prompt: &str) -> Result<String, String> {
        Ok(format!("```aria\n{}```", self.program))
    }
    fn label(&self) -> String {
        self.name.clone()
    }
}

/// A temp file whose path is exposed and which is removed on drop. Used to
/// materialise the GBNF grammar for the `llama:` preset's `--grammar-file`.
pub struct TempFile {
    path: std::path::PathBuf,
}

impl TempFile {
    /// Write `contents` to a uniquely-named temp file and return a handle.
    pub fn write(prefix: &str, contents: &str) -> Result<TempFile, String> {
        // Unique-enough name: pid + a monotonically increasing counter.
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let id = N.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir()
            .join(format!("{}_{}_{}.gbnf", prefix, std::process::id(), id));
        std::fs::write(&path, contents)
            .map_err(|e| format!("cannot write temp grammar {}: {}", path.display(), e))?;
        Ok(TempFile { path })
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ---------------------------------------------------------------------------
// Provider selection (`--provider <spec>`)
// ---------------------------------------------------------------------------

/// Parse a `--provider` spec into a boxed [`Provider`]. The presets all build a
/// `CmdProvider` with the right shell command; `mock` is the built-in
/// deterministic provider; `cmd:<shell>` is the raw escape hatch.
///
/// Recognised specs:
///   - `mock`                         the deterministic built-in (no model)
///   - `cmd:<shell>`                  run `sh -c "<shell>"`, prompt on stdin
///   - `claude`                       Claude Code CLI (`claude -p`)
///   - `codex`                        Codex CLI (`codex exec`)
///   - `llama:<model.gguf>`           local llama.cpp, grammar-constrained
///   - `anthropic`                    cloud Anthropic via `curl` (best-effort)
///   - `openai`                       cloud OpenAI via `curl` (best-effort)
pub fn provider_from_spec(spec: &str) -> Result<Box<dyn Provider>, String> {
    if spec == "mock" {
        return Ok(Box::new(MockProvider::new()));
    }
    if let Some(shell) = spec.strip_prefix("cmd:") {
        if shell.trim().is_empty() {
            return Err("provider `cmd:` requires a shell command".to_string());
        }
        return Ok(Box::new(CmdProvider::new(shell)));
    }
    if spec == "claude" {
        return Ok(Box::new(CmdProvider::named("claude", claude_command())));
    }
    if spec == "codex" {
        return Ok(Box::new(CmdProvider::named("codex", codex_command())));
    }
    if let Some(model) = spec.strip_prefix("llama:") {
        if model.trim().is_empty() {
            return Err("provider `llama:` requires a path to a .gguf model".to_string());
        }
        // Materialise the GBNF grammar to a temp file the command references via
        // `--grammar-file`, so the LOCAL model is constrained to emit
        // syntactically-valid Aria (constrained decoding).
        let gf = TempFile::write("aria_grammar", &crate::gbnf::grammar())?;
        let shell = llama_command(model, &gf.path().to_string_lossy());
        return Ok(Box::new(CmdProvider {
            name: format!("llama:{}", model),
            shell,
            _grammar_file: Some(gf),
        }));
    }
    if let Some(model) = spec.strip_prefix("ollama:") {
        if model.trim().is_empty() {
            return Err("provider `ollama:` requires a model name".to_string());
        }
        // NOTE: ollama has NO GBNF/constrained-decoding support, so an `ollama:`
        // model is NOT syntactically constrained — it relies on the structured-
        // diagnostics FEEDBACK LOOP alone to reach valid Aria. The prompt arrives
        // on stdin (`ollama run` reads a prompt from stdin when not a TTY).
        let shell = ollama_command(model);
        return Ok(Box::new(CmdProvider::named(format!("ollama:{}", model), shell)));
    }
    if spec == "anthropic" {
        return Ok(Box::new(CmdProvider::named("anthropic", anthropic_command())));
    }
    if spec == "openai" {
        return Ok(Box::new(CmdProvider::named("openai", openai_command())));
    }
    Err(format!(
        "unknown provider `{}` (try: mock, cmd:<shell>, claude, codex, llama:<model.gguf>, ollama:<model>, anthropic, openai)",
        spec
    ))
}

/// The Claude Code CLI command: `-p <prompt>` reads from stdin in our loop, so
/// we pass `-p` (print/non-interactive) and let the prompt arrive on stdin.
fn claude_command() -> String {
    "claude -p".to_string()
}

/// The Codex CLI: `codex exec` runs a one-shot task; the prompt arrives on
/// stdin via the `cmd:` mechanism.
fn codex_command() -> String {
    "codex exec".to_string()
}

/// A local llama.cpp command constrained by Aria's GBNF grammar. The model is
/// loaded from `model`; `--grammar-file <grammar>` forces SYNTACTICALLY-VALID
/// Aria (constrained decoding — the local model literally cannot emit a syntax
/// error). The prompt arrives on stdin (`-f /dev/stdin`).
///
/// BINARY: we invoke `llama-completion`, llama.cpp's one-shot TEXT-COMPLETION
/// tool. (Older llama.cpp shipped only `llama-cli`; modern builds split a
/// non-interactive completion binary out as `llama-completion` — `llama-cli` in
/// those builds drops into an interactive loop that floods stdout. If your
/// llama.cpp predates the split, substitute `llama-cli`; the flags are identical.)
///
/// Flags chosen so the child's STDOUT is JUST the completion the extractor reads:
///   - `-no-cnv`            one-shot completion, not interactive conversation;
///   - `--no-display-prompt` do NOT echo the prompt back to stdout;
///   - `--simple-io`        plain IO for better behaviour under a piped subprocess;
///   - `-n 1024`            a sane token budget (enough for a whole program);
///   - `--temp 0`           greedy/deterministic decoding for reproducible numbers;
///   - `--no-warmup`        skip warmup for a faster cold start;
///   - `2>/dev/null`        llama.cpp's load/perf logs go to stderr — drop them so
///                          only the model's text reaches the provider.
fn llama_command(model: &str, grammar: &str) -> String {
    format!(
        "llama-completion -m {} --grammar-file {} -no-cnv --no-display-prompt --simple-io \
         --no-warmup --temp 0 -n 1024 -f /dev/stdin 2>/dev/null",
        shell_quote(model),
        shell_quote(grammar)
    )
}

/// A local ollama command. The prompt arrives on stdin (`ollama run <model>`
/// reads its prompt from stdin when stdout is not a TTY). UNLIKE `llama:`, ollama
/// offers NO GBNF/constrained decoding, so syntactic correctness relies on the
/// feedback loop alone — documented in `docs/BENCHMARKING.md`.
fn ollama_command(model: &str) -> String {
    format!("ollama run {}", shell_quote(model))
}

/// Best-effort cloud Anthropic command: POST the prompt (read from stdin) to the
/// Messages API using `$ANTHROPIC_API_KEY` and a default model, returning the
/// raw JSON — the program extractor then pulls the code out of the text. Cloud
/// models cannot be GBNF-constrained, so syntax relies on the feedback loop.
fn anthropic_command() -> String {
    // `jq -Rs .` JSON-encodes the whole stdin prompt into a string literal; the
    // body is assembled with that as the user message content.
    "PROMPT=$(jq -Rs .); \
     curl -s https://api.anthropic.com/v1/messages \
       -H \"x-api-key: $ANTHROPIC_API_KEY\" \
       -H \"anthropic-version: 2023-06-01\" \
       -H \"content-type: application/json\" \
       -d \"{\\\"model\\\":\\\"${ANTHROPIC_MODEL:-claude-3-5-sonnet-latest}\\\",\\\"max_tokens\\\":2048,\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":$PROMPT}]}\""
        .to_string()
}

/// Best-effort cloud OpenAI command, symmetric to [`anthropic_command`], using
/// `$OPENAI_API_KEY` and the chat-completions API.
fn openai_command() -> String {
    "PROMPT=$(jq -Rs .); \
     curl -s https://api.openai.com/v1/chat/completions \
       -H \"Authorization: Bearer $OPENAI_API_KEY\" \
       -H \"content-type: application/json\" \
       -d \"{\\\"model\\\":\\\"${OPENAI_MODEL:-gpt-4o}\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":$PROMPT}]}\""
        .to_string()
}

/// Single-quote `s` for safe embedding in a `sh -c` command (handles embedded
/// quotes the POSIX way: close, escaped-quote, reopen).
fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

// ---------------------------------------------------------------------------
// Prompt assembly
// ---------------------------------------------------------------------------

/// The tight, accurate Aria language primer fed to the model. Facts are derived
/// from the real language (verified against `aria check`/`aria run`): comment
/// `--`, everything-is-an-expression, no mutation/loops (recursion), lambdas,
/// the prelude HOFs, the data model, and the `main` signature rule.
pub fn primer() -> &'static str {
    "\
You are writing a program in Aria, a small, pure, statically-typed functional language.

CORE RULES
- Comments start with `--`.
- EVERYTHING IS AN EXPRESSION: `if`/`else`, `match`, and blocks all yield values.
- NO mutation, NO loops, NO statements: use RECURSION and expressions instead.
- A block is `{ e1; e2; ...; last }` and evaluates to its LAST expression.
- Functions: `fn name(p: T, q: U) -> R = <expr>`. Generics: `fn id[T](x: T) -> T = x`.
- `pure fn` may be declared; ordinary `fn` may call the `print_*` builtins.
- Bindings: `let x = <expr>; <rest>` (no reassignment).
- Conditionals: `if cond { a } else { b }` (both branches required, same type).
- Lambdas: `\\x -> body` (one param) or `\\(x: T, y: U) -> body` (annotated).

TYPES & DATA
- Primitives: `Int`, `Float`, `Bool`, `String`, `Unit`.
- Algebraic data types (uppercase names):
    `type Color = | Red | Green | Blue`
    `type Shape = | Circle(Float) | Rect(Float, Float)`
    `type List[T] = | Nil | Cons(T, List[T])`   -- generic, `[T]` type params
- Records:  `type Point = { x: Int, y: Int }`  built as `Point { x: 1, y: 2 }`,
  accessed as `p.x`.
- `match` MUST be EXHAUSTIVE (cover every constructor or use `_`):
    `match s { Circle(r) => 3.14 * r * r, Rect(w, h) => w * h }`
- Data-model builtins: Array, Map, Set, Vector, Tensor, e.g.
    `array_new()`, `array_push(xs, x)`, `array_get(xs, i)`, `array_len(xs)`,
    `map_new()`, `map_insert(m, k, v)`, `set_new()`, `vec_new(...)`, tensors.

PRELUDE HIGHER-ORDER FUNCTIONS (use these instead of loops)
- `range(n) -> Array[Int]`                      -- [0, 1, ..., n-1]
- `array_map(xs, \\x -> ...) -> Array[B]`
- `array_filter(xs, \\x -> <Bool>) -> Array[A]`
- `array_fold(xs, init, \\(acc, x) -> ...) -> B`   -- note: pass a NAMED fn for
  multi-param folds if a bare lambda will not infer; e.g. `fn add(a:Int,b:Int)->Int=a+b`.

OUTPUT BUILTINS
- `print_int(n)`, `print_float(x)`, `print_str(s)`, `print_bool(b)` print a line.
- `int_to_str(n) -> String`, `concat(a, b) -> String` for building strings.

PROGRAM SHAPE
- The entry point is `fn main() -> Int` (or `-> Float`/`-> String`). It takes NO
  parameters. Its return value is the program's result. Use `print_*` for output
  and end `main` with a final expression of the declared return type.
"
}

/// Assemble the full prompt: the primer, the task, and a strict instruction to
/// emit ONLY a complete program. `transcript` carries the running feedback from
/// prior failed attempts (empty on the first iteration).
pub fn build_prompt(task: &str, transcript: &[String]) -> String {
    let mut s = String::new();
    s.push_str(primer());
    s.push_str("\nTASK\n");
    s.push_str(task.trim());
    s.push('\n');
    if !transcript.is_empty() {
        s.push_str("\nPRIOR ATTEMPTS & COMPILER FEEDBACK\n");
        for msg in transcript {
            s.push_str(msg);
            s.push('\n');
        }
    }
    s.push_str(
        "\nReturn ONLY a complete Aria program (no markdown, no prose, no explanation).\n",
    );
    s
}

/// Build the FEEDBACK message appended to the transcript when a candidate fails
/// to check: the JSON diagnostics plus the program the model produced, with an
/// instruction to return the corrected full program. The literal phrase
/// "Your program had these errors" is also what the mock keys its fix on.
pub fn build_feedback(program: &str, diags: &[Diagnostic]) -> String {
    format!(
        "Your program had these errors: {}\nHere is the program you wrote:\n{}\nReturn the corrected full Aria program (only the program).",
        array_to_json(diags),
        program
    )
}

/// Build the RUNTIME-error feedback appended to the transcript when a candidate
/// TYPE-CHECKS CLEAN but FAILS AT RUNTIME. It carries the runtime error message,
/// the most-recent-first STACK TRACE (the exact call chain the model needs to
/// localize the fix), and the program, with an instruction to return a corrected
/// program. This closes the loop over runtime failures (write -> check -> RUN ->
/// fix), not just type errors. The literal phrase "failed at runtime" lets a
/// mock/test key its corrected response on having received runtime feedback.
///
/// `output` is whatever the program PRINTED before it trapped (captured `print_*`
/// output). When non-empty it is surfaced FIRST ("Your program printed: ...")
/// so the model has the full picture — what the program produced AND where it
/// then failed — not just the error. A program that printed nothing adds no such
/// preamble.
pub fn build_runtime_feedback(
    program: &str,
    err: &interp::RuntimeError,
    output: Option<&str>,
) -> String {
    // `err.render()` is `runtime error: <msg>` + the indented frames.
    match output {
        Some(o) if !o.is_empty() => format!(
            "Your program type-checks and printed:\n{}\nthen failed at runtime:\n{}\nHere is the program you wrote:\n{}\nReturn the corrected full Aria program (only the program).",
            o,
            err.render(),
            program
        ),
        _ => format!(
            "Your program type-checks but failed at runtime:\n{}\nHere is the program you wrote:\n{}\nReturn the corrected full Aria program (only the program).",
            err.render(),
            program
        ),
    }
}

// ---------------------------------------------------------------------------
// Program extraction
// ---------------------------------------------------------------------------

/// Extract a complete Aria program from a model's raw text response. Handles:
///   - a fenced ```aria block, or a fenced ``` block;
///   - multiple fenced blocks (picks the one defining `fn main`);
///   - prose around a fenced block;
///   - a bare program with no fences (returns it as-is, trimmed).
/// Never panics; on an empty response returns an empty string.
pub fn extract_program(response: &str) -> String {
    // Cloud providers (`anthropic`/`openai`) return the model's text wrapped in a
    // JSON envelope. If the response looks like such an envelope, unwrap the
    // assistant text first, THEN run the normal fence/prose extraction on it.
    let response = &unwrap_json_response(response);
    let blocks = fenced_blocks(response);
    if !blocks.is_empty() {
        // Prefer a block that defines `fn main`; else the last block (models tend
        // to put the final program last).
        if let Some(b) = blocks.iter().rev().find(|b| defines_main(b)) {
            return b.trim().to_string();
        }
        return blocks.last().unwrap().trim().to_string();
    }
    // No fences: treat the whole response as a bare program.
    response.trim().to_string()
}

/// True if `s` contains an Aria `fn main` definition (allowing whitespace /
/// generic-free signature variation).
fn defines_main(s: &str) -> bool {
    s.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with("fn main") || t.starts_with("pure fn main")
    })
}

/// Collect the bodies of all triple-backtick fenced code blocks in `text`. An
/// optional language tag on the opening fence (e.g. ```` ```aria ````) is
/// stripped. Robust to an unterminated final fence (takes to end of text).
fn fenced_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("```") {
        // Advance past the opening fence and its optional language tag line.
        let after_open = &rest[open + 3..];
        // The language tag (if any) runs to the next newline; skip it.
        let body_start = match after_open.find('\n') {
            Some(nl) => nl + 1,
            None => {
                // Opening fence with no newline after it — nothing usable left.
                break;
            }
        };
        let body_region = &after_open[body_start..];
        match body_region.find("```") {
            Some(close) => {
                blocks.push(body_region[..close].to_string());
                rest = &body_region[close + 3..];
            }
            None => {
                // Unterminated fence: take the remainder as the block body.
                blocks.push(body_region.to_string());
                break;
            }
        }
    }
    blocks
}

/// Unwrap a cloud-API JSON envelope to the assistant's TEXT, so the normal
/// fence/prose extractor can then run on it. Handles the two shapes our cloud
/// presets produce:
///   - Anthropic Messages API: `{"content":[{"type":"text","text":"<code>"}],...}`
///   - OpenAI chat completions: `{"choices":[{"message":{"content":"<code>"}}],...}`
/// A non-JSON response (a local model's bare text, or already-extracted code) is
/// returned unchanged — this is a no-op unless the input is clearly such an
/// envelope. We do a tiny, dependency-free scan for the relevant key and decode
/// the JSON string that follows it; if anything is off we return the input as-is
/// (the downstream check loop then drives the fix), never panicking.
fn unwrap_json_response(response: &str) -> String {
    let trimmed = response.trim_start();
    if !trimmed.starts_with('{') {
        return response.to_string();
    }
    // Anthropic: the text lives under a `"text":` key inside the `content` array.
    // OpenAI: under a `"content":` key inside `choices[].message`. We look for the
    // OpenAI key first (more specific), then Anthropic's `"text"`.
    if let Some(s) = json_string_after_key(trimmed, "\"content\"") {
        // OpenAI's `message.content` is a plain string; Anthropic's top-level
        // `content` is an ARRAY (so `json_string_after_key` returns None and we
        // fall through to the `"text"` key below).
        return s;
    }
    if let Some(s) = json_string_after_key(trimmed, "\"text\"") {
        return s;
    }
    response.to_string()
}

/// Find `key` in `json`, and if it is immediately followed (modulo whitespace and
/// a `:`) by a JSON STRING, decode and return that string. Returns `None` if the
/// key is absent or its value is not a string (e.g. an array/object). This is a
/// deliberately small, allocation-light scan — not a full JSON parser — because
/// we only need to lift one text field out of a known-shape API response.
fn json_string_after_key(json: &str, key: &str) -> Option<String> {
    let mut search_from = 0;
    while let Some(rel) = json[search_from..].find(key) {
        let kpos = search_from + rel;
        let mut rest = json[kpos + key.len()..].trim_start();
        if let Some(after_colon) = rest.strip_prefix(':') {
            rest = after_colon.trim_start();
            if rest.starts_with('"') {
                if let Some(decoded) = decode_json_string(rest) {
                    return Some(decoded);
                }
            }
            // Value is present but not a string (array/object/number) — this key
            // isn't the text we want; keep scanning for a later occurrence.
        }
        search_from = kpos + key.len();
    }
    None
}

/// Decode a JSON string literal at the START of `s` (which must begin with `"`),
/// honouring the standard JSON escapes (`\" \\ \/ \n \r \t \b \f` and `\uXXXX`).
/// Returns the decoded contents (without the surrounding quotes), or `None` if
/// the literal is unterminated/malformed.
fn decode_json_string(s: &str) -> Option<String> {
    let mut chars = s.chars();
    if chars.next() != Some('"') {
        return None;
    }
    let mut out = String::new();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'b' => out.push('\u{0008}'),
                'f' => out.push('\u{000C}'),
                'u' => {
                    let mut code = 0u32;
                    for _ in 0..4 {
                        let h = chars.next()?;
                        code = code * 16 + h.to_digit(16)?;
                    }
                    out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                }
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    // Unterminated string.
    None
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

/// The outcome of one authoring run.
pub struct AgentOutcome {
    pub success: bool,
    /// The best (final) program produced.
    pub program: String,
    /// Diagnostics remaining on the final program (`[]` iff it checked clean).
    pub diagnostics: Vec<Diagnostic>,
    /// `main`'s rendered result value, if the program checked clean AND ran.
    pub result: Option<String>,
    /// What the program PRINTED (captured `print_*` output) on a successful run.
    /// `None` if the program never reached a clean+running state. This is the
    /// program's observable OUTPUT, distinct from `main`'s return `result`.
    pub output: Option<String>,
    /// A runtime error string, if the program checked clean but failed to run.
    pub runtime_error: Option<String>,
    /// How many provider iterations were used.
    pub iterations: usize,
    /// The running transcript (prompts/feedback) for reporting on failure.
    pub transcript: Vec<String>,
}

/// Run the write -> check -> fix -> run loop against `provider` for `task`, up to
/// `max_iters` iterations. Pure orchestration: it never panics on provider
/// failure or malformed output — those surface as a failed [`AgentOutcome`].
///
/// SAFETY: clean programs are executed in-process via `interp` — safe by
/// construction (Aria has no I/O/FFI/network/filesystem; see the module docs).
pub fn run_loop(
    provider: &dyn Provider,
    task: &str,
    max_iters: usize,
    verbose: bool,
) -> AgentOutcome {
    let mut transcript: Vec<String> = Vec::new();
    let mut last_program = String::new();
    let mut last_diags: Vec<Diagnostic> = Vec::new();
    let mut last_runtime_error: Option<String> = None;
    let mut last_output: Option<String> = None;
    let iters = max_iters.max(1);

    for iter in 1..=iters {
        let prompt = build_prompt(task, &transcript);
        if verbose {
            eprintln!("--- iteration {} ---", iter);
            eprintln!("[prompt {} bytes]", prompt.len());
        }

        let raw = match provider.complete(&prompt) {
            Ok(r) => r,
            Err(e) => {
                // Provider failure: fail gracefully with what we have.
                return AgentOutcome {
                    success: false,
                    program: last_program,
                    diagnostics: last_diags,
                    result: None,
                    output: None,
                    runtime_error: Some(format!("provider error: {}", e)),
                    iterations: iter,
                    transcript,
                };
            }
        };

        let program = extract_program(&raw);
        last_program = program.clone();
        if verbose {
            eprintln!("[extracted {} bytes of program]", program.len());
        }

        // CHECK in-process via the same path as `aria check --json`.
        let diags = check_program(&program);
        last_diags = diags.clone();

        if diags.is_empty() {
            // Clean: RUN it CAPTURING its printed output (safe by construction —
            // no I/O/FFI/etc.). We grade what the program PRINTS, so the loop
            // carries the captured stdout alongside `main`'s return value.
            let (result, output, runtime_error) = run_program_traced(&program);
            match runtime_error {
                None => {
                    // Clean check AND clean run: converged.
                    return AgentOutcome {
                        success: true,
                        program,
                        diagnostics: Vec::new(),
                        result,
                        output,
                        runtime_error: None,
                        iterations: iter,
                        transcript,
                    };
                }
                Some(err) => {
                    // CHECKS CLEAN but FAILS AT RUNTIME: not yet converged. Feed
                    // the runtime error + STACK TRACE back and retry within budget
                    // (write -> check -> RUN -> fix). If this was the last
                    // iteration, fall through to the budget-exhausted outcome
                    // below, surfacing the runtime error.
                    if verbose {
                        eprintln!("[runtime error] {}", err.render());
                    }
                    last_runtime_error = Some(err.render());
                    last_output = output.clone();
                    if iter < iters {
                        // Include any output the program printed BEFORE it trapped,
                        // so the model sees the full picture (printed-then-failed).
                        let feedback =
                            build_runtime_feedback(&program, &err, output.as_deref());
                        transcript.push(feedback);
                        continue;
                    }
                    // Last iteration: report the runtime failure, surfacing any
                    // output the program printed before it trapped.
                    return AgentOutcome {
                        success: false,
                        program,
                        diagnostics: Vec::new(),
                        result: None,
                        output,
                        runtime_error: Some(err.render()),
                        iterations: iter,
                        transcript,
                    };
                }
            }
        }

        // Not clean: record feedback and loop (unless this was the last iter).
        if verbose {
            eprintln!("[{} diagnostic(s)]", diags.len());
            for d in &diags {
                eprintln!("  {} {}: {}", d.code, d.phase, d.message);
            }
        }
        let feedback = build_feedback(&program, &diags);
        transcript.push(feedback);
    }

    // Budget exhausted without a converging run. `last_runtime_error` carries the
    // most recent clean-check-but-runtime-failure (if any), so the report can
    // distinguish "never type-checked" from "checked but kept trapping".
    AgentOutcome {
        success: false,
        program: last_program,
        diagnostics: last_diags,
        result: None,
        output: last_output,
        runtime_error: last_runtime_error,
        iterations: iters,
        transcript,
    }
}

/// Lex -> parse -> type-check a candidate program in-process, returning the
/// structured diagnostics (the same data `aria check --json` emits). Lex/parse
/// failures become single diagnostics, mirroring `run_check_json`.
pub fn check_program(program: &str) -> Vec<Diagnostic> {
    match lexer::lex(&prelude::wrap(program)) {
        Err(e) => vec![Diagnostic::error("lex", e)],
        Ok(toks) => match parser::parse(toks) {
            Err(e) => vec![Diagnostic::error("parse", e)],
            Ok(prog) => typeck::check_structured(&prog),
        },
    }
}

/// Run a program that has ALREADY been checked clean, CAPTURING its printed
/// output. Returns `(Some(result), Some(printed_output), None)` on success or
/// `(None, None, Some(error))` on a runtime error / construction failure. The
/// program is re-lexed/parsed (cheap) so this is self-contained. Unlike a normal
/// `aria run`, the `print_*` output is BUFFERED (not sent to stdout) so the loop
/// can report — and the benchmark grade — what the program PRINTED.
pub fn run_program(program: &str) -> (Option<String>, Option<String>, Option<String>) {
    let (res, out, err) = run_program_traced(program);
    // Flatten the structured runtime error back to a plain string (the rendered
    // trace) for callers that only want the legacy 3-tuple shape.
    (res, out, err.map(|e| e.render()))
}

/// Like [`run_program`] but returns a STRUCTURED [`interp::RuntimeError`] (message
/// + stack frames) on a runtime failure, so the loop can both render the trace as
/// feedback for the model AND inspect the frames. Construction / lex-parse
/// failures (which carry no Aria call stack) become a frame-less `RuntimeError`.
///
/// The captured OUTPUT (second element) is returned on BOTH paths: on success it
/// is everything the program printed; on a runtime error it is whatever the
/// program printed BEFORE it trapped (the partial print-then-trap output), so the
/// loop can include it in the feedback to the model. It is `None` only for a
/// lex/parse/construction failure (the program never ran).
pub fn run_program_traced(
    program: &str,
) -> (Option<String>, Option<String>, Option<interp::RuntimeError>) {
    let no_frames = |message: String| interp::RuntimeError { message, frames: Vec::new() };
    let prog = match lexer::lex(&prelude::wrap(program)).and_then(parser::parse) {
        Ok(p) => p,
        Err(e) => return (None, None, Some(no_frames(e))),
    };
    let interp = match interp::Interp::new(&prog) {
        Ok(i) => i,
        Err(e) => return (None, None, Some(no_frames(e))),
    };
    let (result, out) = interp.run_main_capturing_traced_partial();
    match result {
        Ok(v) => (Some(v.display()), Some(out), None),
        // Keep the partial output the program printed before it trapped.
        Err(e) => (None, Some(out), Some(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- program extraction -------------------------------------------

    #[test]
    fn extract_fenced_aria_block() {
        let r = "blah\n```aria\nfn main() -> Int = 0\n```\ntrailing";
        assert_eq!(extract_program(r), "fn main() -> Int = 0");
    }

    #[test]
    fn extract_fenced_plain_block() {
        let r = "```\nfn main() -> Int = 1\n```";
        assert_eq!(extract_program(r), "fn main() -> Int = 1");
    }

    #[test]
    fn extract_bare_program() {
        let r = "fn main() -> Int = 2\n";
        assert_eq!(extract_program(r), "fn main() -> Int = 2");
    }

    #[test]
    fn extract_prose_then_block() {
        let r = "Sure! Here you go:\n\n```aria\nfn main() -> Int = 3\n```\n\nHope that helps.";
        assert_eq!(extract_program(r), "fn main() -> Int = 3");
    }

    #[test]
    fn extract_multiple_blocks_picks_fn_main() {
        // A helper block then the real program — must pick the `fn main` one.
        let r = "First a helper:\n```aria\nfn helper(x: Int) -> Int = x + 1\n```\n\
                 Now the program:\n```aria\nfn helper(x: Int) -> Int = x + 1\n\
                 fn main() -> Int = helper(41)\n```";
        let p = extract_program(r);
        assert!(p.contains("fn main"), "picked the wrong block: {}", p);
        assert!(p.contains("helper(41)"));
    }

    #[test]
    fn extract_handles_unterminated_fence() {
        let r = "```aria\nfn main() -> Int = 9";
        assert_eq!(extract_program(r), "fn main() -> Int = 9");
    }

    #[test]
    fn extract_empty_response_is_empty() {
        assert_eq!(extract_program(""), "");
        assert_eq!(extract_program("   \n  "), "");
    }

    // ---- JSON cloud-response unwrap (no network; canned API JSON) ------

    #[test]
    fn extract_from_anthropic_json_response() {
        // A realistic Anthropic Messages API response: the program is in
        // `content[0].text`, fenced, with the newlines JSON-escaped as `\n`.
        let json = r#"{"id":"msg_01","type":"message","role":"assistant","model":"claude-3-5-sonnet","content":[{"type":"text","text":"Here you go:\n```aria\nfn main() -> Int = {\n  print_int(42);\n  0\n}\n```\n"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":20}}"#;
        let p = extract_program(json);
        assert!(p.contains("fn main() -> Int"), "got: {}", p);
        assert!(p.contains("print_int(42)"), "got: {}", p);
        // And the unwrapped+extracted program actually checks clean.
        assert!(check_program(&p).is_empty(), "extracted program should check: {}", p);
    }

    #[test]
    fn extract_from_openai_json_response() {
        // OpenAI chat-completions: program in `choices[0].message.content`.
        let json = r#"{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"```aria\nfn main() -> Int = 7\n```"},"finish_reason":"stop"}],"usage":{"total_tokens":30}}"#;
        let p = extract_program(json);
        assert_eq!(p, "fn main() -> Int = 7");
    }

    #[test]
    fn extract_from_openai_bare_program_in_json() {
        // OpenAI content with NO fences — a bare program string with escaped
        // newlines. The JSON unwrap must still surface the program.
        let json = r#"{"choices":[{"message":{"content":"fn main() -> Int = {\n  print_int(1);\n  0\n}\n"}}]}"#;
        let p = extract_program(json);
        assert!(p.contains("fn main"), "got: {}", p);
        assert!(p.contains("print_int(1)"), "got: {}", p);
    }

    #[test]
    fn unwrap_json_handles_unicode_and_quote_escapes() {
        // `\uXXXX` and `\"` in the content must decode correctly.
        let json = r#"{"choices":[{"message":{"content":"fn main() -> String = \"hi!\""}}]}"#;
        let p = extract_program(json);
        assert_eq!(p, "fn main() -> String = \"hi!\"");
    }

    #[test]
    fn unwrap_non_json_is_passthrough() {
        // A local model's bare response is NOT JSON; it must pass through intact.
        assert_eq!(unwrap_json_response("fn main() -> Int = 0\n"), "fn main() -> Int = 0\n");
        // A fenced response (no JSON envelope) is also untouched by the unwrap.
        let fenced = "```aria\nfn main() -> Int = 0\n```";
        assert_eq!(unwrap_json_response(fenced), fenced);
    }

    #[test]
    fn decode_json_string_basic_escapes() {
        assert_eq!(decode_json_string(r#""a\nb""#).as_deref(), Some("a\nb"));
        assert_eq!(decode_json_string(r#""tab\there""#).as_deref(), Some("tab\there"));
        assert_eq!(decode_json_string(r#""quote\"end""#).as_deref(), Some("quote\"end"));
        // Unterminated -> None (never panics).
        assert_eq!(decode_json_string(r#""unterminated"#), None);
    }

    // ---- prompt assembly ----------------------------------------------

    #[test]
    fn prompt_contains_primer_task_and_only_instruction() {
        let p = build_prompt("print the number 7", &[]);
        // Primer essentials.
        assert!(p.contains("EVERYTHING IS AN EXPRESSION"));
        assert!(p.contains("fn main() -> Int"));
        assert!(p.contains("match"));
        // The task.
        assert!(p.contains("print the number 7"));
        // The strict instruction.
        assert!(p.contains("Return ONLY a complete Aria program"));
    }

    #[test]
    fn prompt_includes_transcript_feedback() {
        let fb = vec!["Your program had these errors: [...]".to_string()];
        let p = build_prompt("task", &fb);
        assert!(p.contains("PRIOR ATTEMPTS & COMPILER FEEDBACK"));
        assert!(p.contains("Your program had these errors"));
    }

    #[test]
    fn feedback_contains_diagnostic_codes() {
        let diags = check_program("fn main() -> Int = \"hi\"");
        assert!(!diags.is_empty());
        let fb = build_feedback("fn main() -> Int = \"hi\"", &diags);
        assert!(fb.contains("Your program had these errors"));
        assert!(fb.contains("E0201"), "feedback should embed the diagnostic code");
        assert!(fb.contains("fn main"));
    }

    // ---- preset command construction (NO real model invoked) ----------

    #[test]
    fn preset_claude_builds_expected_command() {
        let p = provider_from_spec("claude").unwrap();
        assert_eq!(p.label(), "claude");
    }

    #[test]
    fn preset_codex_builds_expected_command() {
        let p = provider_from_spec("codex").unwrap();
        assert_eq!(p.label(), "codex");
    }

    #[test]
    fn claude_and_codex_command_strings() {
        assert_eq!(claude_command(), "claude -p");
        assert_eq!(codex_command(), "codex exec");
    }

    #[test]
    fn preset_llama_includes_grammar_file() {
        let cmd = llama_command("/models/m.gguf", "/tmp/g.gbnf");
        assert!(cmd.contains("llama-completion"), "use the one-shot completion binary");
        assert!(cmd.contains("-m '/models/m.gguf'"));
        assert!(cmd.contains("--grammar-file '/tmp/g.gbnf'"));
        // Hardened flags: one-shot, prompt-on-stdin, quiet stdout, token budget.
        assert!(cmd.contains("-no-cnv"), "must be one-shot, not conversational");
        assert!(cmd.contains("--no-display-prompt"), "must not echo the prompt");
        assert!(cmd.contains("-f /dev/stdin"), "prompt arrives on stdin");
        assert!(cmd.contains("-n 1024"), "must set a token budget");
        assert!(cmd.contains("--temp 0"), "deterministic decoding");
        assert!(cmd.contains("2>/dev/null"), "drop llama.cpp logs from stdout");
        // And the spec path materialises a real grammar file + provider.
        let p = provider_from_spec("llama:/models/m.gguf").unwrap();
        assert!(p.label().starts_with("llama:"));
    }

    #[test]
    fn preset_llama_materialises_real_grammar_file() {
        // The `llama:` spec must WRITE the GBNF grammar to a temp file and embed
        // that exact path in the command — proving constrained decoding is wired.
        let gf = TempFile::write("aria_grammar_test", &crate::gbnf::grammar()).unwrap();
        let path = gf.path().to_string_lossy().to_string();
        assert!(std::path::Path::new(&path).exists(), "grammar temp file written");
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("root ::="), "temp file holds the real GBNF grammar");
        let cmd = llama_command("/models/m.gguf", &path);
        assert!(cmd.contains(&format!("--grammar-file '{}'", path)));
    }

    #[test]
    fn preset_ollama_builds_run_command_and_notes_no_grammar() {
        let cmd = ollama_command("qwen2.5:1.5b");
        assert_eq!(cmd, "ollama run 'qwen2.5:1.5b'");
        let p = provider_from_spec("ollama:qwen2.5:1.5b").unwrap();
        assert_eq!(p.label(), "ollama:qwen2.5:1.5b");
        // Empty model name is rejected.
        assert!(provider_from_spec("ollama:").is_err());
    }

    #[test]
    fn preset_cloud_commands_use_curl_and_keys() {
        let a = anthropic_command();
        assert!(a.contains("curl"));
        assert!(a.contains("api.anthropic.com"));
        assert!(a.contains("ANTHROPIC_API_KEY"));
        let o = openai_command();
        assert!(o.contains("curl"));
        assert!(o.contains("api.openai.com"));
        assert!(o.contains("OPENAI_API_KEY"));
        // And the specs construct providers.
        assert_eq!(provider_from_spec("anthropic").unwrap().label(), "anthropic");
        assert_eq!(provider_from_spec("openai").unwrap().label(), "openai");
    }

    #[test]
    fn cmd_provider_spec_parses() {
        let p = provider_from_spec("cmd:cat").unwrap();
        assert_eq!(p.label(), "cmd:cat");
    }

    #[test]
    fn unknown_and_empty_provider_specs_error() {
        assert!(provider_from_spec("nope").is_err());
        assert!(provider_from_spec("cmd:").is_err());
        assert!(provider_from_spec("cmd:   ").is_err());
        assert!(provider_from_spec("llama:").is_err());
    }

    #[test]
    fn shell_quote_escapes_single_quotes() {
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("a'b"), "'a'\\''b'");
    }

    // ---- the cmd provider, exercised with a harmless local shell ------

    #[test]
    fn cmd_provider_pipes_prompt_through_stdin() {
        // `cat` echoes the prompt back — no model, no network. Proves the
        // stdin->stdout plumbing.
        let p = CmdProvider::new("cat");
        let out = p.complete("hello agent").unwrap();
        assert_eq!(out, "hello agent");
    }

    #[test]
    fn cmd_provider_nonzero_exit_is_error() {
        let p = CmdProvider::new("exit 3");
        assert!(p.complete("anything").is_err());
    }

    // ---- the full loop, driven by the mock ----------------------------

    #[test]
    fn mock_buggy_program_fails_check_with_e0201() {
        // The mock's first program must be caught by check_structured.
        let diags = check_program(MockProvider::buggy_program());
        assert!(!diags.is_empty(), "buggy program should not check clean");
        assert!(
            diags.iter().any(|d| d.code == "E0201"),
            "expected a type-mismatch E0201, got {:?}",
            diags.iter().map(|d| d.code).collect::<Vec<_>>()
        );
    }

    #[test]
    fn mock_fixed_program_checks_clean_and_runs() {
        let diags = check_program(MockProvider::fixed_program());
        assert!(diags.is_empty(), "fixed program should check clean: {:?}", diags);
        let (res, out, err) = run_program(MockProvider::fixed_program());
        assert!(err.is_none(), "fixed program should run: {:?}", err);
        // sum_to(10) == 55 (the return value).
        assert_eq!(res.as_deref(), Some("55"));
        // The fixed program also PRINTS sum_to(10) == 55 via print_int.
        assert_eq!(out.as_deref(), Some("55\n"));
    }

    #[test]
    fn loop_converges_via_diagnostic_feedback() {
        let provider = MockProvider::new();
        let outcome = run_loop(&provider, "print the sum of 1..10", 5, false);

        // Converged, and used exactly 2 iterations (buggy -> feedback -> fixed).
        assert!(outcome.success, "loop should converge: {:?}", outcome.diagnostics);
        assert_eq!(outcome.iterations, 2, "should take 2 iterations");
        assert!(outcome.diagnostics.is_empty());
        assert_eq!(outcome.result.as_deref(), Some("55"));
        // The loop CAPTURED what the program printed (print_int(sum_to(10))).
        assert_eq!(outcome.output.as_deref(), Some("55\n"));
        assert!(outcome.program.contains("fn main"));

        // The transcript (feedback that drove the fix) must contain the
        // diagnostic code caught on iteration 1 — proving diagnostics drove it.
        assert_eq!(outcome.transcript.len(), 1, "one feedback message");
        assert!(
            outcome.transcript[0].contains("E0201"),
            "feedback must carry the diagnostic code"
        );
        assert!(outcome.transcript[0].contains("Your program had these errors"));
    }

    #[test]
    fn loop_reports_failure_when_budget_exhausted() {
        // A provider that always returns the same broken program never converges.
        struct AlwaysBroken;
        impl Provider for AlwaysBroken {
            fn complete(&self, _p: &str) -> Result<String, String> {
                Ok("fn main() -> Int = \"nope\"".to_string())
            }
            fn label(&self) -> String {
                "broken".to_string()
            }
        }
        let outcome = run_loop(&AlwaysBroken, "task", 3, false);
        assert!(!outcome.success);
        assert_eq!(outcome.iterations, 3);
        assert!(!outcome.diagnostics.is_empty());
        // Three failed attempts => three feedback messages in the transcript.
        assert_eq!(outcome.transcript.len(), 3);
    }

    #[test]
    fn loop_handles_provider_failure_gracefully() {
        struct Failing;
        impl Provider for Failing {
            fn complete(&self, _p: &str) -> Result<String, String> {
                Err("boom".to_string())
            }
            fn label(&self) -> String {
                "failing".to_string()
            }
        }
        let outcome = run_loop(&Failing, "task", 5, false);
        assert!(!outcome.success);
        assert!(outcome.runtime_error.as_deref().unwrap().contains("boom"));
    }

    // ---- runtime-error feedback path (Feature 1b) ---------------------

    #[test]
    fn run_program_traced_surfaces_stack_trace() {
        // A clean-checking program that divides by zero three calls deep yields a
        // structured runtime error whose frames are the call chain.
        let src = "\
fn inner(x: Int) -> Int = 1 / 0
fn outer(x: Int) -> Int = inner(x)
fn main() -> Int = outer(5)
";
        let (res, _out, err) = run_program_traced(src);
        assert!(res.is_none());
        let err = err.expect("runtime error");
        assert!(err.message.contains("division by zero"), "msg: {}", err.message);
        let names: Vec<&str> = err.frames.iter().map(|f| f.function.as_str()).collect();
        assert_eq!(names, vec!["inner", "outer", "main"], "most-recent first");
        // The rendered form embeds the trace the model will see. Frames now carry
        // the precise CALL SITE (`line:col`): `inner` is called at line 2 col 27
        // (inside `outer`); `main` is the synthetic entry, shown by its def line.
        let rendered = err.render();
        assert!(rendered.contains("at `inner` (line 2:27)"), "got:\n{}", rendered);
        assert!(rendered.contains("at `main` (line 3)"), "got:\n{}", rendered);
    }

    #[test]
    fn loop_feeds_back_runtime_error_and_reconverges() {
        // A provider whose FIRST program type-checks but TRAPS at runtime, and
        // whose corrected program (returned once it has SEEN runtime feedback) is
        // clean. Proves the loop treats a runtime failure as non-converged and
        // closes the write -> check -> RUN -> fix loop.
        struct RuntimeThenFixed;
        impl Provider for RuntimeThenFixed {
            fn complete(&self, prompt: &str) -> Result<String, String> {
                if prompt.contains("failed at runtime") {
                    // Corrected program: no division by zero, returns 7.
                    Ok("```aria\nfn main() -> Int = 7\n```".to_string())
                } else {
                    // Clean-checking but traps: 1/0 is well-typed Int division.
                    Ok("```aria\nfn main() -> Int = 1 / 0\n```".to_string())
                }
            }
            fn label(&self) -> String {
                "rt".to_string()
            }
        }
        let outcome = run_loop(&RuntimeThenFixed, "task", 5, false);
        assert!(outcome.success, "should reconverge after runtime feedback");
        assert_eq!(outcome.iterations, 2, "trap -> feedback -> fix");
        assert_eq!(outcome.result.as_deref(), Some("7"));
        // The transcript's feedback carried the runtime error + stack trace.
        assert_eq!(outcome.transcript.len(), 1);
        assert!(outcome.transcript[0].contains("failed at runtime"));
        assert!(outcome.transcript[0].contains("division by zero"));
        assert!(
            outcome.transcript[0].contains("at `main`"),
            "feedback must carry the stack trace: {}",
            outcome.transcript[0]
        );
    }

    #[test]
    fn run_program_traced_keeps_output_printed_before_a_trap() {
        // A program that PRINTS two lines and THEN divides by zero: the captured
        // output before the trap must survive on the runtime-error path.
        let src = "\
fn main() -> Int = {
  print_int(1);
  print_int(2);
  1 / 0
}
";
        let (res, out, err) = run_program_traced(src);
        assert!(res.is_none());
        assert!(err.expect("runtime error").message.contains("division by zero"));
        // The two printed lines are preserved even though the run trapped.
        assert_eq!(out.as_deref(), Some("1\n2\n"), "partial output before trap");
    }

    #[test]
    fn runtime_feedback_includes_printed_output() {
        // When a program printed before trapping, the feedback surfaces BOTH the
        // captured output AND the runtime error + trace.
        let err = interp::RuntimeError {
            message: "division by zero".to_string(),
            frames: Vec::new(),
        };
        let fb = build_runtime_feedback("fn main() -> Int = 0", &err, Some("1\n2\n"));
        assert!(fb.contains("printed:\n1\n2\n"), "feedback shows printed output: {}", fb);
        assert!(fb.contains("then failed at runtime"));
        assert!(fb.contains("division by zero"));
        // With NO output, no "printed" preamble — just the runtime error.
        let fb2 = build_runtime_feedback("fn main() -> Int = 0", &err, None);
        assert!(!fb2.contains("printed:"), "no output -> no preamble: {}", fb2);
        assert!(fb2.contains("failed at runtime"));
        let fb3 = build_runtime_feedback("fn main() -> Int = 0", &err, Some(""));
        assert!(!fb3.contains("printed:"), "empty output -> no preamble: {}", fb3);
    }

    #[test]
    fn loop_feeds_back_printed_output_with_runtime_trace() {
        // A provider whose first program PRINTS then traps; its corrected program
        // (returned once it has seen runtime feedback) is clean. The feedback that
        // drove the fix must carry BOTH the printed lines AND the stack trace.
        struct PrintThenTrap;
        impl Provider for PrintThenTrap {
            fn complete(&self, prompt: &str) -> Result<String, String> {
                if prompt.contains("failed at runtime") {
                    Ok("```aria\nfn main() -> Int = { print_int(42); 42 }\n```".to_string())
                } else {
                    Ok("```aria\nfn main() -> Int = { print_int(1); print_int(2); 1 / 0 }\n```"
                        .to_string())
                }
            }
            fn label(&self) -> String {
                "pt".to_string()
            }
        }
        let outcome = run_loop(&PrintThenTrap, "task", 5, false);
        assert!(outcome.success, "should reconverge after print-then-trap feedback");
        assert_eq!(outcome.iterations, 2);
        // The single feedback message carries the printed output AND the trace.
        assert_eq!(outcome.transcript.len(), 1);
        let fb = &outcome.transcript[0];
        assert!(fb.contains("printed:\n1\n2\n"), "feedback must include printed output: {}", fb);
        assert!(fb.contains("then failed at runtime"), "got: {}", fb);
        assert!(fb.contains("division by zero"), "got: {}", fb);
        assert!(fb.contains("at `main`"), "feedback must carry the stack trace: {}", fb);
        // The converged program prints 42 and returns 42.
        assert_eq!(outcome.result.as_deref(), Some("42"));
        assert_eq!(outcome.output.as_deref(), Some("42\n"));
    }

    #[test]
    fn successful_outcome_surfaces_captured_output() {
        // On a converged run the outcome carries the program's OUTPUT distinct from
        // main's return value: this program PRINTS "hi" but RETURNS 0.
        struct PrintsHi;
        impl Provider for PrintsHi {
            fn complete(&self, _p: &str) -> Result<String, String> {
                Ok("```aria\nfn main() -> Int = { print_str(\"hi\"); 0 }\n```".to_string())
            }
            fn label(&self) -> String {
                "hi".to_string()
            }
        }
        let outcome = run_loop(&PrintsHi, "task", 3, false);
        assert!(outcome.success);
        assert_eq!(outcome.result.as_deref(), Some("0"), "main returns 0");
        assert_eq!(outcome.output.as_deref(), Some("hi\n"), "but prints `hi`");
    }

    #[test]
    fn loop_reports_runtime_failure_when_budget_exhausted() {
        // A provider that ALWAYS returns a clean-checking program that traps never
        // converges; the budget-exhausted outcome surfaces the runtime error.
        struct AlwaysTraps;
        impl Provider for AlwaysTraps {
            fn complete(&self, _p: &str) -> Result<String, String> {
                Ok("fn main() -> Int = 1 / 0".to_string())
            }
            fn label(&self) -> String {
                "traps".to_string()
            }
        }
        let outcome = run_loop(&AlwaysTraps, "task", 3, false);
        assert!(!outcome.success);
        assert_eq!(outcome.iterations, 3);
        assert!(outcome.diagnostics.is_empty(), "it type-checks, so no diagnostics");
        let rt = outcome.runtime_error.expect("runtime error surfaced");
        assert!(rt.contains("division by zero"));
        // Two runtime-feedback messages precede the final (3rd) attempt.
        assert_eq!(outcome.transcript.len(), 2);
    }
}
