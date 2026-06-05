# Aria — `aria agent` (the AI authoring loop)

> The **authoring half** of Aria's "living, breathing" AI-native vision: an LLM
> writes an Aria program, the **compiler is the ground-truth feedback**, and the
> loop converges to working, safe-to-run code. Where `aria check --json` is the
> structured *feedback channel*, `aria agent` is the *loop* that drives a model
> with it.

## What it does

`aria agent` is a provider-agnostic **write → check → fix → run** loop:

1. **Assemble a prompt** — a tight, accurate Aria language primer + your task +
   a strict instruction to return *only* a complete Aria program.
2. **Invoke the provider** — a local model, a cloud model, or an agentic CLI —
   to get the model's raw text.
3. **Extract the program** — strip ```` ``` ```` / ```` ```aria ```` fences and
   surrounding prose; with multiple code blocks, pick the one defining `fn main`;
   a bare (un-fenced) program is accepted as-is.
4. **Check it in-process** via `typeck::check_structured` — the *same* path as
   `aria check --json`. If there are diagnostics, build a feedback message
   embedding the **JSON diagnostics** + the program, append it to the transcript,
   and loop back to step 2.
5. **Run it** in-process via the interpreter when the check is clean. If the
   program **type-checks but fails at runtime** (e.g. division by zero), the loop
   treats it as *not yet converged*: it builds a runtime-error feedback message
   carrying the error **and the stack trace** (the exact call chain — see
   [ANALYSIS.md](ANALYSIS.md)), appends it to the transcript, and loops back to
   step 2. This closes the loop over runtime failures too —
   **write → check → RUN → fix** — and the trace is exactly the signal the model
   needs to localize the bug. On a clean check *and* a clean run, report the
   program, `main`'s result, the captured output, and the iteration count.
6. **Give up gracefully** after `--max-iters` (default 5) without a converging
   run: report the best attempt, its remaining diagnostics *or* the last runtime
   error, and the transcript.

```text
   PRIMER + TASK ──▶ PROVIDER ──▶ EXTRACT ──▶ check_structured ──┐ diagnostics
        ▲                          program         │ clean       │
        │                                          ▼             ▼
        │                                    interp run     build FEEDBACK
        │                                    SUCCESS:        (JSON diagnostics
        └──────────── append feedback ◀──────────────────── + program)
                      to transcript, loop
```

## Invocation

```sh
aria agent [--provider <spec>] [--max-iters N] [--out <file.aria>] [--verbose] "<task>"
```

- `--provider <spec>` — which model/CLI to drive (default `mock`).
- `--max-iters N` — iteration budget (default `5`).
- `--out <file.aria>` — on success, write the final program to this file.
- `--verbose` — print each iteration's prompt size, extracted program size, and
  the diagnostics caught.

The progress, final program, result, and diagnostics are printed to **stderr**
(so `--out` keeps stdout for the program's own `print_*` output). Exit code is
`0` on success, `1` on failure, `2` on a usage error.

### Offline demo (no model required)

```sh
aria agent --provider mock --verbose "write a program that prints the sum of 1..10"
```

The built-in **`mock`** provider scripts the loop deterministically: iteration 1
returns a program that declares `fn main() -> Int` but returns a `String`
(a clean `E0201` type mismatch the checker catches); the diagnostics are fed
back; iteration 2 returns the corrected program, which checks clean and runs,
printing `55`. This proves the *whole* loop end-to-end with no network and no
model.

## Providers

A provider is anything that turns a prompt into raw model text
(`fn complete(&self, prompt: &str) -> Result<String, String>`). Specs:

| Spec | What it drives | How |
| --- | --- | --- |
| `mock` | built-in, deterministic | no model — scripts a buggy→fixed sequence for the demo/tests |
| `cmd:<shell>` | **anything** (escape hatch) | runs `sh -c "<shell>"`, writes the prompt to **stdin**, reads **stdout** |
| `claude` | the Claude Code CLI | builds `claude -p` (prompt on stdin) |
| `codex` | the Codex CLI | builds `codex exec` (prompt on stdin) |
| `llama:<model.gguf>` | local llama.cpp | builds a `llama-cli` command with **`--grammar-file`** = Aria's GBNF (constrained decoding) |
| `anthropic` | cloud Anthropic | builds a `curl` to the Messages API using `$ANTHROPIC_API_KEY` (best-effort) |
| `openai` | cloud OpenAI | builds a `curl` to the chat-completions API using `$OPENAI_API_KEY` (best-effort) |

### `cmd:<shell>` — the universal escape hatch

Every preset is just sugar over `cmd:`. The mechanism: spawn `sh -c "<shell>"`,
write the full prompt to the child's **stdin**, read its **stdout** as the
response. That single contract covers a curl-to-cloud command, a local
`llama.cpp` invocation, or an agentic coder. If a preset doesn't fit your setup,
write the `cmd:` yourself — it is the reliable path.

```sh
# Pipe the prompt to any tool that reads stdin and writes the program to stdout:
aria agent --provider 'cmd:my-llm --stdin' "reverse a list of Ints"
```

### Local model, grammar-constrained (`llama:`)

```sh
aria agent --provider llama:/models/llama-3-8b.gguf "factorial of 6"
```

constructs (roughly):

```sh
llama-cli -m '/models/llama-3-8b.gguf' --grammar-file '<tmp>.gbnf' -no-cnv -f /dev/stdin
```

where `<tmp>.gbnf` is `aria gbnf` (the GBNF grammar from `src/gbnf.rs`), written
to a temp file for the run. **Constrained decoding** means the local model
*cannot emit an Aria syntax error* — every token it produces stays inside the
grammar.

### Cloud model via `curl` / `cmd:`

```sh
ANTHROPIC_API_KEY=sk-... aria agent --provider anthropic "sum a list of Floats"
# or roll your own with full control:
aria agent --provider 'cmd:curl -s https://api.example/v1 -H "..." -d @-' "..."
```

The cloud presets emit the model's raw JSON response on stdout; the program
**extractor** then pulls the Aria code out of the message text. These are
**best-effort** (they assume `jq` + `curl` are present and the API shape is
current); the `cmd:` escape hatch is the dependable route.

## Constrained decoding vs. the feedback loop

Two complementary mechanisms keep generated programs correct:

- **Constrained decoding (GBNF)** makes **syntax errors impossible** — but only
  where the stack supports it (local `llama.cpp` via `--grammar-file`). It
  cannot catch *semantic* errors (type mismatches, non-exhaustive `match`,
  arity, purity).
- **The structured-diagnostics feedback loop** fixes **semantic errors
  everywhere** — every provider, cloud or local — by feeding
  `typeck::check_structured`'s JSON back to the model until it converges.

Together: the grammar (where available) eliminates a whole class of failures up
front, and the compiler's structured diagnostics drive the model to fix the
rest. The compiler is the ground truth either way.

## Safety by construction

The loop **runs model-generated Aria** via the in-process interpreter. This is
**safe to do with untrusted LLM output** because Aria has **no I/O, no FFI, no
network, and no filesystem access** — the only effects are the `print_*`
builtins (which write to the process's own stdout) and pure computation. There
is no `rm -rf`, no exfiltration, no shelling out: executing a hallucinated
program **cannot harm the host.**

This is a genuine, honestly-stated advantage of an *effect-free* language for AI
authoring — you can let a model's output run as part of the loop without a
sandbox. The one residual cost is **unbounded computation** (a generated program
can loop/recurse). That is bounded by the interpreter's call-depth guard
(`MAX_CALL_DEPTH`) plus the large-but-finite worker stack, which turn runaway
recursion into a clean error rather than a crash — not a security boundary, but
a liveness one.

(Note: the *provider* side — `cmd:`, `claude`, `curl`, `llama-cli` — does shell
out, and is exactly as trusted as the command you supply. The safety claim is
about executing the *generated Aria program*, not about the provider you choose.)

## Measuring a provider — `aria agent-bench`

The loop converges a model to code that **checks clean and runs**. But "runs" is
not "right": a program can check clean, run, and still print the **wrong answer**
("converged but incorrect"). `aria agent-bench` turns *"does this provider write
Aria correctly?"* into a **measured number** — a pass-rate per provider.

```sh
aria agent-bench [--provider <spec>] [--max-iters N] [--task <name>] [--verbose]
```

For each task in a fixed **authoring suite** (`src/agent_tasks.rs`), the runner:

1. drives the agent loop (write → check → fix → run) with the chosen provider,
   recording whether it **converged** and how many **iterations** it took, then
2. **grades** the converged program against the task's **out-of-band oracle**
   (`src/agent_bench.rs`) — running it *capturing* its printed output and
   comparing the observed output and/or `main` return value to the expected one.

It then prints a per-task table and an aggregate summary to **stdout** (progress
on stderr). Defaults: `--provider reference`, `--max-iters 5`. Exit code is `0`
iff every task graded correct.

### The metrics

| Metric | Meaning |
| --- | --- |
| **convergence rate** | fraction of tasks the loop got to *check clean + run* within budget. Measures the loop, not correctness. |
| **correctness pass-rate** | **the headline number** — fraction of tasks whose produced program actually printed/returned the *right* answer. This is the real "writes Aria correctly" score. |
| **iterations-to-green** | mean / median iterations used over the converged tasks — how many compiler-feedback rounds the provider needed. Lower is better. |

A high convergence with a low correctness rate is the **converged-but-incorrect**
signal: the provider reliably produces *runnable* Aria that computes the *wrong*
thing. The two numbers are reported separately so this is visible.

The report is greppable: every per-task line starts `TASK `, every aggregate
line starts `BENCH ` (e.g. `BENCH correctness 100.0% (15/15)`).

### Grading is external — no leak

The expected answer is **never** placed in the prompt the provider sees — the
loop only ever gets the task's natural-language `prompt`. The grader applies the
oracle **out of band** after the program is written and run. So the pass-rate
measures genuine author-correctness, not the model's ability to echo a test.

### `--provider reference` — the offline self-test

The suite ships a **known-correct reference solution** per task (never shown to a
real provider). With `--provider reference`, the runner feeds each task its *own*
reference (via `agent::FixedProvider`), driving the **entire** harness — output
capture, loop, grader, runner, report — **offline with no model**. It must report
**~100% converged + 100% correct in 1 iteration each**, proving the machinery end
to end:

```text
== aria authoring benchmark :: provider `reference` ==
TASK name               converged  iters  correct  note
TASK constant                 yes      1      yes
TASK sum_1_to_100             yes      1      yes
...
TASK string_build             yes      1      yes
---
BENCH convergence 100.0% (15/15)
BENCH correctness 100.0% (15/15)  <- author-correctness pass-rate
BENCH iters-to-green mean 1.00 median 1.0 (over 15 converged)
BENCH counts total=15 converged=15 correct=15 incorrect=0 nonconverged=0
```

To measure a *real* provider, swap the spec — e.g. `aria agent-bench --provider
claude` or `--provider 'cmd:my-llm --stdin'`. The same suite + grader then yield
that provider's pass-rate and iterations-to-green. That is how we turn "LLMs
write Aria correctly" into a number **per provider**.

### Output capture

Grading needs to see what a program **prints**, not just what `main` returns. The
interpreter has an opt-in capture mode (`Interp::run_main_capturing`): when on,
the `print_*` builtins append their formatted line to a buffer instead of writing
to stdout (identical formatting + trailing newline). Normal `aria run` is
unaffected — only the loop/benchmark use the capturing path, and the `AgentOutcome`
now carries the program's printed **output** alongside `main`'s return value
(shown under `--- output ---` in `aria agent --verbose`).

## Limitations

- **Cloud providers are best-effort**: they assume `curl` + `jq` and a current
  API shape. Prefer `cmd:` for anything non-trivial.
- **Cloud models can't be GBNF-constrained** (the APIs don't accept GBNF), so
  for them syntax correctness relies entirely on the feedback loop.
- **`print_*` output** from a successful run goes to the process stdout; the
  structured "result" reported by the loop is `main`'s **return value**.
- The loop's quality is bounded by the model: a model that never produces a
  checkable program will exhaust the iteration budget and report failure with
  the best attempt and its diagnostics.

## Implementation

- `src/agent.rs` — the `Provider` trait, the `mock`/`cmd`/preset/`FixedProvider`
  providers, the prompt assembly, the program extractor, `run_loop`, and
  `run_program` (the capturing run).
- `src/agent_tasks.rs` — the authoring task suite (prompt + oracle + reference
  per task) and the external `grade` function.
- `src/agent_bench.rs` — the benchmark runner: drive the loop per task, grade,
  aggregate, and render the report.
- `src/interp.rs` — `Interp::run_main_capturing` (output capture for grading).
- Wired into the CLI in `src/main.rs` (`aria agent`, `aria agent-bench`).
- The check path reuses `typeck::check_structured` (see `docs/DIAGNOSTICS.md`);
  the grammar reuses `src/gbnf.rs` (see `aria gbnf`).
- Tested entirely **offline**: command *construction* for every preset, the
  extractor, the prompt, and the full loop driven by the deterministic `mock`
  (real models/CLIs are never invoked in the test suite).
