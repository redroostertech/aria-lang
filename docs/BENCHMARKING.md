# Aria — LIVE LLM AUTHORING BENCHMARK

> How **"LLMs write Aria correctly"** becomes a *measured, reproducible number*
> per model. This is the live benchmark infrastructure for `aria agent` (see
> [AGENT.md](AGENT.md)): point it at a real model and it reports that model's
> **author-correctness pass-rate** over a fixed task suite.

## TL;DR — the one number

```
aria agent-bench --provider <PROVIDER>
```

drives the **write → check → fix → run** loop ([AGENT.md](AGENT.md)) over the
15-task suite ([`src/agent_tasks.rs`](../src/agent_tasks.rs)) and prints:

```
BENCH convergence  <pct>%   (<n>/15)   <- got to clean-checking + running code
BENCH correctness  <pct>%   (<n>/15)   <- author-correctness PASS-RATE (headline)
BENCH iters-to-green mean <m> median <md>
```

The **correctness** line is the headline: the fraction of tasks whose produced
program is *actually correct* against an **out-of-band oracle** (the expected
answer is never in the prompt — see "Honesty" below). Convergence is the weaker
"did it produce code that checks-and-runs" number.

---

## Two distinct mechanisms (why Aria gets honest numbers)

1. **Grammar-constrained decoding kills SYNTAX errors — locally.**
   For a local `llama.cpp` model, Aria exports its concrete syntax as a **GBNF
   grammar** ([`src/gbnf.rs`](../src/gbnf.rs)) and passes it to `llama-cli` via
   `--grammar-file`. The decoder can then only emit token sequences the grammar
   accepts, so the model **literally cannot produce a parse error**. Cloud APIs
   (`anthropic`/`openai`) and `ollama` have **no** GBNF support, so they get no
   syntactic guarantee — they rely on mechanism 2 alone.

2. **The feedback loop fixes SEMANTICS — everywhere.**
   Whatever the provider, every candidate is type-checked via
   `typeck::check_structured` (the same channel as `aria check --json`) and, if
   clean, **run**. The structured diagnostics — and, for a clean-but-trapping
   program, the runtime error **plus stack trace** — are fed back so the model
   fixes the bug, looping until the program checks clean and runs. This corrects
   type errors, wrong recursion, runtime traps, etc., for *all* providers.

So: **grammar → no syntax errors (local)**, **feedback → no semantic errors
(everywhere)**. The benchmark measures what survives both: real correctness.

## Safe by construction (why running model output is fine)

The loop **runs** model-written Aria in-process. This is safe with *untrusted*
LLM output because **Aria has no I/O, no FFI, no network, and no filesystem
access** — the only effects are the `print_*` builtins and pure computation.
There is no `rm -rf`, no exfiltration, no shelling out: executing a hallucinated
program cannot harm the host. Unbounded computation is bounded by the
interpreter's call-depth guard (run on a large stack so the guard fires cleanly).
See the safety note in [`src/agent.rs`](../src/agent.rs).

---

## Getting a REAL number — exact commands

### (a) Local llama.cpp + a small GGUF + grammar-constrained decoding

This is the fully-offline, no-API-key path, and the only one with the
syntactic guarantee.

**One-time setup** — install `llama-completion` and fetch a small instruct GGUF:

```bash
# llama.cpp (provides `llama-completion`; older builds call it `llama-cli`):
brew install llama.cpp            # or build from https://github.com/ggml-org/llama.cpp

# A small instruct model (~1.1 GB). Any instruct/coder GGUF works; bigger = better.
mkdir -p ~/.llama-models
curl -L -o ~/.llama-models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
```

**Run the benchmark** (one command):

```bash
scripts/bench-llm.sh ~/.llama-models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

or directly:

```bash
RUST_MIN_STACK=536870912 \
  aria agent-bench --provider "llama:$HOME/.llama-models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
```

The `llama:` provider writes Aria's GBNF grammar to a temp file and builds:

```
llama-completion -m <model.gguf> --grammar-file <grammar> -no-cnv \
  --no-display-prompt --simple-io --no-warmup --temp 0 -n 1024 -f /dev/stdin 2>/dev/null
```

(`llama-completion` is llama.cpp's non-interactive one-shot completion binary;
older llama.cpp builds shipped only `llama-cli` — substitute it if needed.)

(`--grammar-file` = constrained decoding; `--no-display-prompt`/`--simple-io`/
`2>/dev/null` keep stdout to just the completion; `--temp 0` = deterministic.)

### (b) Cloud API via `curl` (no Rust HTTP dependency)

No GBNF here — syntax relies on the feedback loop. The response is JSON; the
program extractor unwraps the assistant text out of the API envelope
(`content[].text` for Anthropic, `choices[].message.content` for OpenAI)
automatically.

```bash
# Anthropic (needs `curl` + `jq`):
ANTHROPIC_API_KEY=sk-ant-... \
  aria agent-bench --provider anthropic
# optional: ANTHROPIC_MODEL=claude-3-5-sonnet-latest

# OpenAI:
OPENAI_API_KEY=sk-... \
  aria agent-bench --provider openai
# optional: OPENAI_MODEL=gpt-4o
```

### (c) An agentic coder CLI

```bash
aria agent-bench --provider claude      # builds `claude -p`, prompt on stdin
aria agent-bench --provider codex       # builds `codex exec`, prompt on stdin
```

### (d) `ollama` (local, but NOT grammar-constrained)

```bash
aria agent-bench --provider ollama:qwen2.5:1.5b
```

ollama offers no GBNF, so this relies on the feedback loop only (expect more
iterations / lower convergence than the `llama:` path on the same model).

---

## What was actually executed in this sandbox (honesty)

This repo ships a **fully offline end-to-end proof** of the entire measurement
pipeline that does NOT require any LLM, network, or API key:

- [`scripts/stub_model.sh`](../scripts/stub_model.sh) is a deterministic external
  "model": it reads the agent prompt on stdin and emits a valid Aria program,
  dispatching on the task. It is wired through the **genuine external-subprocess
  provider** (the same `cmd:` path a real local model uses — *not* the built-in
  `mock`/`reference`):

  ```bash
  aria agent-bench --provider "cmd:bash scripts/stub_model.sh"
  ```

- This produces a **real, non-trivial report** (the offline "real number"):

  ```
  BENCH convergence 93.3% (14/15)
  BENCH correctness 40.0% (6/15)  <- author-correctness pass-rate
  BENCH iters-to-green mean 1.07 median 1.0 (over 14 converged)
  ```

  By construction the stub solves 6 tasks correctly (`constant`, `sum_1_to_100`,
  `factorial_10`, `is_prime_97`, `record_field`, `gcd`); takes **2 iterations**
  on `sum_1_to_100` (it emits a buggy `E0201` program first and the fix only
  after seeing the compiler feedback — proving the loop); converges on `fib_20`
  but grades **incorrect** (it prints `fib(19)`, not `fib(20)` — proving "runs"
  ≠ "right"); and never converges on `string_build` (a true failure row). This
  exercises the *full* pipeline: external provider → grammar/feedback → check →
  run → grade → aggregate. It is asserted in
  [`src/agent_bench.rs`](../src/agent_bench.rs)
  (`cmd_stub_model_produces_expected_real_report`).

- The provider command construction and the JSON cloud-response extractor are
  unit-tested with realistic **canned** API JSON (no network) in
  [`src/agent.rs`](../src/agent.rs).

**What still needs YOUR model/key for a genuine LLM number:** the actual
`llama:`/`anthropic`/`openai`/`claude`/`codex` runs above. The infrastructure,
command construction, extraction, grading, and aggregation are all proven
offline; plugging in a real provider yields that model's real pass-rate via the
identical code path.

---

## The task suite & grading (no leak)

15 tasks ([`src/agent_tasks.rs`](../src/agent_tasks.rs)) spanning recursion,
ADTs + `match`, records, Array/Map/Set + prelude HOFs, and string building. Each
task has a natural-language `prompt`, an **out-of-band oracle**
(`expected_output`/`expected_return`), and a known-correct `reference`.

**Honesty:** the oracle is **never** placed in the prompt — `agent::build_prompt`
only ever gets the task's `prompt`. The grader runs the produced program,
captures its `print_*` output, and compares it to the oracle *out of band*. So
the pass-rate measures real author-correctness, not the model's ability to echo a
test. The `reference` solutions are used only for the offline self-test
(`--provider reference` must score 100%/100%) and never shown to a real provider.
