#!/usr/bin/env bash
#
# bench-llm.sh — one-command runner to get a REAL author-correctness number for
# Aria from a LOCAL llama.cpp model with GBNF-constrained decoding.
#
# Usage:
#     scripts/bench-llm.sh <path-to-model.gguf> [max_iters]
#
# It builds `aria` (release for speed) and runs the full authoring benchmark with
# the `llama:<model>` provider, which:
#   - invokes `llama-completion` (llama.cpp's one-shot completion binary; on older
#     llama.cpp this was `llama-cli`), writing Aria's GBNF grammar to a temp file
#     and passing it via `--grammar-file`, so the local model is CONSTRAINED to
#     emit syntactically valid Aria (it literally cannot produce a parse error);
#   - feeds the compiler's structured diagnostics + stack traces back to fix
#     semantic errors, looping until each program checks clean and runs.
#
# The result is a reproducible, per-model pass-rate: convergence%, the headline
# author-CORRECTNESS%, and iterations-to-green.
#
# ---------------------------------------------------------------------------
# Getting a small instruct GGUF (do this once; a ~1-2 GB download):
#
#   mkdir -p ~/.llama-models
#   # Qwen2.5-1.5B-Instruct (Q4_K_M, ~1.1 GB) from the Bartowski GGUF repo:
#   curl -L -o ~/.llama-models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
#     https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
#
#   # Then benchmark it:
#   scripts/bench-llm.sh ~/.llama-models/qwen2.5-1.5b-instruct-q4_k_m.gguf
#
# (Any instruct-tuned GGUF works; larger/coder models score higher. You also need
# llama.cpp's `llama-cli` on PATH: `brew install llama.cpp` or build from source.)
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL="${1:-}"
MAX_ITERS="${2:-5}"

if [ -z "$MODEL" ]; then
  echo "usage: scripts/bench-llm.sh <path-to-model.gguf> [max_iters]" >&2
  exit 2
fi
if [ ! -f "$MODEL" ]; then
  echo "error: model file not found: $MODEL" >&2
  exit 2
fi
if ! command -v llama-completion >/dev/null 2>&1; then
  echo "error: llama-completion (llama.cpp) not found on PATH." >&2
  echo "  install with: brew install llama.cpp   (or build llama.cpp from source)" >&2
  echo "  (older llama.cpp builds call this binary 'llama-cli' instead)" >&2
  exit 2
fi

# Run from the repo root regardless of where the script was invoked.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ">> building aria (release) ..." >&2
cargo build --release >/dev/null 2>&1

echo ">> benchmarking llama:$MODEL  (max-iters=$MAX_ITERS, GBNF-constrained) ..." >&2
# RUST_MIN_STACK keeps the interpreter's deep-recursion guard winning the race
# against a native stack overflow when it runs model-generated Aria.
RUST_MIN_STACK=536870912 \
  ./target/release/aria agent-bench \
    --provider "llama:$MODEL" \
    --max-iters "$MAX_ITERS"
