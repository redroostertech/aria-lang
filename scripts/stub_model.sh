#!/usr/bin/env bash
#
# stub_model.sh — a DETERMINISTIC, OFFLINE "model" for proving the Aria authoring
# benchmark pipeline end to end with NO real LLM, NO network, and NO recursion
# into `claude`/`codex`.
#
# It reads the full agent PROMPT on stdin (the primer + the task + any prior
# compiler feedback) and emits a syntactically-valid Aria program on stdout — the
# exact same contract a real model satisfies through the `cmd:` provider path. It
# is wired into the benchmark with:
#
#     aria agent-bench --provider "cmd:bash scripts/stub_model.sh"
#
# This exercises the genuine external-subprocess provider (NOT the built-in
# mock/reference): prompt on stdin -> program on stdout -> compiler check -> run
# -> grade -> aggregate. The pass-rate it produces is a REAL number from a real
# external command.
#
# BY DESIGN it is not perfect: it SOLVES a handful of tasks correctly, FLUBS a
# couple (one converged-but-incorrect, one that never type-checks), and on one
# task emits a buggy program FIRST and the fix only after it has SEEN the
# compiler's feedback — so the report shows a non-trivial pass-rate AND the
# write -> check -> fix feedback loop in action.

set -euo pipefail

# Slurp the entire prompt from stdin.
full_prompt="$(cat)"

# Isolate the TASK text. The agent prompt is `<primer>\nTASK\n<task>\n...`; the
# primer ITSELF contains illustrative Aria (e.g. `type Point = { x: Int, y: Int }`),
# so we must match ONLY the task description, not the whole prompt. Take everything
# after the `TASK` line marker emitted by `agent::build_prompt`.
prompt="${full_prompt#*$'\nTASK\n'}"

# Did this prompt already carry compiler feedback from a prior failed attempt?
# (`agent::build_feedback` emits the literal phrase below.) Used to demonstrate
# the feedback loop on one task.
saw_feedback=0
case "$prompt" in
  *"Your program had these errors"*) saw_feedback=1 ;;
  *"failed at runtime"*) saw_feedback=1 ;;
esac

emit() { printf '%s\n' "$1"; }

# Dispatch on a DISTINCTIVE phrase from each task's natural-language prompt.
# (These match agent_tasks.rs; the expected answer is never in the prompt, so the
# stub must actually encode a correct computation, not echo a constant.)
case "$prompt" in

  # --- constant: print 42, return 0 (CORRECT) ---------------------------
  *"prints the integer 42"*)
    emit 'fn main() -> Int = {
  print_int(42);
  0
}'
    ;;

  # --- sum_1_to_100: FEEDBACK-LOOP DEMO -------------------------------------
  # First attempt declares `-> Int` but ends in a String (a clean E0201 the
  # checker catches). After feedback, emit the correct recursive sum.
  *"sum of the integers from 1"*)
    if [ "$saw_feedback" -eq 1 ]; then
      emit 'fn sum_to(n: Int) -> Int =
  if n == 0 { 0 } else { n + sum_to(n - 1) }

fn main() -> Int = {
  print_int(sum_to(100));
  0
}'
    else
      # Buggy: body is a String where Int is declared -> E0201.
      emit 'fn sum_to(n: Int) -> Int =
  if n == 0 { 0 } else { n + sum_to(n - 1) }

fn main() -> Int = {
  print_int(sum_to(100));
  "done"
}'
    fi
    ;;

  # --- factorial_10: 10! (CORRECT) -----------------------------------------
  *"ten factorial"*)
    emit 'fn fact(n: Int) -> Int =
  if n == 0 { 1 } else { n * fact(n - 1) }

fn main() -> Int = {
  print_int(fact(10));
  0
}'
    ;;

  # --- gcd: Euclid (CORRECT) -----------------------------------------------
  *"greatest common divisor"*)
    emit 'fn gcd(a: Int, b: Int) -> Int =
  if b == 0 { a } else { gcd(b, a % b) }

fn main() -> Int = {
  print_int(gcd(48, 36));
  0
}'
    ;;

  # --- is_prime_97: trial division (CORRECT) -------------------------------
  *"97 is a prime"*)
    emit 'fn divides(d: Int, n: Int) -> Bool = n % d == 0
fn has_factor(n: Int, d: Int) -> Bool =
  if d * d > n { false }
  else { if divides(d, n) { true } else { has_factor(n, d + 1) } }
fn is_prime(n: Int) -> Bool =
  if n < 2 { false } else { !has_factor(n, 2) }

fn main() -> Int = {
  print_bool(is_prime(97));
  0
}'
    ;;

  # --- record_field: Point.x + Point.y (CORRECT) ---------------------------
  *"type Point = { x: Int, y: Int }"*)
    emit 'type Point = { x: Int, y: Int }

fn main() -> Int = {
  let p = Point { x: 3, y: 4 };
  print_int(p.x + p.y);
  0
}'
    ;;

  # --- factorial via fib phrasing: FLUB #1 (converged-but-INCORRECT) -------
  # We "answer" the Fibonacci task with a clean-checking program that prints the
  # WRONG number (fib(19) instead of fib(20)). It converges (checks + runs) but
  # grades INCORRECT — proving the runner separates "runs" from "right".
  *"Fibonacci"*)
    emit 'fn fib(n: Int) -> Int =
  if n < 2 { n } else { fib(n - 1) + fib(n - 2) }

fn main() -> Int = {
  print_int(fib(19));
  0
}'
    ;;

  # --- string_build: FLUB #2 (never type-checks -> non-converged) ----------
  # Deliberately broken: `int_to_str` is fed a String, and the program never
  # checks clean, so within budget it never converges (a true failure row).
  *"sum = "*)
    emit 'fn main() -> Int = {
  print_str(concat("sum = ", int_to_str("oops")));
  0
}'
    ;;

  # --- everything else: a syntactically valid but generic program ----------
  # Checks clean and runs (so it "converges") but will grade INCORRECT for any
  # task whose oracle it does not match — an honest stand-in for "the model did
  # not solve this one".
  *)
    emit 'fn main() -> Int = {
  print_int(0);
  0
}'
    ;;
esac
