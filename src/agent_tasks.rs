//! The Aria AI-AUTHORING BENCHMARK: a fixed suite of natural-language authoring
//! TASKS, each with a correctness ORACLE and a known-correct REFERENCE solution,
//! plus the GRADER that turns "did the model write Aria correctly?" into a
//! deterministic pass/fail.
//!
//! This is the MEASUREMENT layer for the agent loop (`src/agent.rs`). The loop
//! makes a provider WRITE a program and CONVERGE it to something that checks
//! clean and runs; this module asks the orthogonal, harder question: is the
//! converged program ACTUALLY CORRECT for the task? A program can check clean,
//! run, and still print the wrong answer — that is "converged but incorrect".
//!
//! HOW GRADING STAYS HONEST (no leak): the `expected` oracle is NEVER placed in
//! the prompt the provider sees — `agent::build_prompt` only ever gets the task's
//! natural-language `prompt`. The grader runs the produced program (capturing its
//! printed output via `interp::run_main_capturing`) and compares the OBSERVED
//! output/return against the oracle out-of-band. So the pass-rate measures real
//! author-correctness, not the provider's ability to echo a test.
//!
//! The `reference` field is a KNOWN-CORRECT Aria program for each task. It is
//! used to (a) SELF-TEST the suite + grader offline (`--provider reference`
//! feeds each task its own reference and must score ~100%), and (b) prove, in
//! tests, that the grader passes correct solutions and FAILS wrong ones. It is
//! never shown to a real provider.
//!
//! Everything here stays within the current language (pure, recursion, ADTs +
//! match, records, tuples-as-records, Array/Map/Set + the prelude HOFs, the
//! `print_*`/`int_to_str`/`concat` builtins) — no IO, loops, mutation, or
//! strings beyond `concat`. Zero external dependencies.

use crate::agent;

/// One authoring task: a natural-language spec, an out-of-band correctness
/// oracle, and a known-correct reference solution.
pub struct Task {
    /// Stable, greppable task name (also the `--task <name>` selector).
    pub name: &'static str,
    /// The natural-language spec shown to the provider. It must pin down the
    /// expected OUTPUT precisely enough that grading is deterministic.
    pub prompt: &'static str,
    /// Expected captured stdout (exact, including trailing newlines), or `None`
    /// if this task is graded only on `main`'s return value.
    pub expected_output: Option<&'static str>,
    /// Expected rendering of `main`'s return value (`Value::display`), or `None`
    /// if this task is graded only on printed output.
    pub expected_return: Option<&'static str>,
    /// A known-correct Aria program. NEVER shown to a real provider; used by the
    /// `reference` provider for the offline self-test and by the suite's tests.
    pub reference: &'static str,
}

/// The grade of a produced program against a task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Grade {
    /// Output and/or return value matched the oracle.
    Correct,
    /// The program ran but produced the wrong output/return (converged but
    /// incorrect), or it failed to run / construct. `why` is a short reason.
    Incorrect { why: String },
}

impl Grade {
    pub fn is_correct(&self) -> bool {
        matches!(self, Grade::Correct)
    }
}

/// GRADE a produced Aria `program` against `task`: run it CAPTURING its printed
/// output (the same path the agent loop uses), then compare the observed output
/// and/or return value against the task's oracle. CORRECT iff every expectation
/// the task specifies is met. Never panics: a runtime/construction failure is a
/// clean `Incorrect`, so one bad program never aborts a benchmark sweep.
///
/// The oracle is applied OUT OF BAND here — it was never in the prompt — so this
/// measures real author-correctness, not test-passing.
pub fn grade(task: &Task, program: &str) -> Grade {
    let (ret, out, err) = agent::run_program(program);
    if let Some(e) = err {
        return Grade::Incorrect { why: format!("runtime error: {}", e) };
    }
    // A clean check is the loop's job; here the program already ran. `out`/`ret`
    // are `Some` on a successful run (see `agent::run_program`).
    if let Some(expected) = task.expected_output {
        let got = out.as_deref().unwrap_or("");
        if got != expected {
            return Grade::Incorrect {
                why: format!("output mismatch: expected {:?}, got {:?}", expected, got),
            };
        }
    }
    if let Some(expected) = task.expected_return {
        let got = ret.as_deref().unwrap_or("");
        if got != expected {
            return Grade::Incorrect {
                why: format!("return mismatch: expected {:?}, got {:?}", expected, got),
            };
        }
    }
    Grade::Correct
}

/// The full benchmark suite. Spread of difficulty, all within the current
/// language. Each prompt is written to make the expected OUTPUT unambiguous so
/// grading is deterministic, while never revealing the literal expected value
/// any more than the spec itself requires.
pub fn tasks() -> Vec<Task> {
    vec![
        Task {
            name: "constant",
            prompt: "Write a program whose `main` prints the integer 42 on its own line \
                     (using print_int) and returns 0.",
            expected_output: Some("42\n"),
            expected_return: Some("0"),
            reference: "fn main() -> Int = {\n  print_int(42);\n  0\n}\n",
        },
        Task {
            name: "sum_1_to_100",
            prompt: "Write a program whose `main` prints the sum of the integers from 1 \
                     to 100 inclusive (using print_int) and returns 0.",
            expected_output: Some("5050\n"),
            expected_return: Some("0"),
            reference: "fn sum_to(n: Int) -> Int =\n  \
                          if n == 0 { 0 } else { n + sum_to(n - 1) }\n\n\
                        fn main() -> Int = {\n  print_int(sum_to(100));\n  0\n}\n",
        },
        Task {
            name: "factorial_10",
            prompt: "Write a program whose `main` prints 10! (ten factorial = the product \
                     1*2*...*10) using print_int, and returns 0.",
            expected_output: Some("3628800\n"),
            expected_return: Some("0"),
            reference: "fn fact(n: Int) -> Int =\n  \
                          if n == 0 { 1 } else { n * fact(n - 1) }\n\n\
                        fn main() -> Int = {\n  print_int(fact(10));\n  0\n}\n",
        },
        Task {
            name: "fib_20",
            prompt: "Write a program whose `main` prints the 20th Fibonacci number using \
                     print_int, and returns 0. Define Fibonacci with fib(0) = 0, \
                     fib(1) = 1, fib(n) = fib(n-1) + fib(n-2), and print fib(20).",
            expected_output: Some("6765\n"),
            expected_return: Some("0"),
            reference: "fn fib(n: Int) -> Int =\n  \
                          if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\n\n\
                        fn main() -> Int = {\n  print_int(fib(20));\n  0\n}\n",
        },
        Task {
            name: "is_prime_97",
            prompt: "Write a program whose `main` prints whether 97 is a prime number, \
                     using print_bool (so it prints `true` if 97 is prime, `false` \
                     otherwise), and returns 0.",
            expected_output: Some("true\n"),
            expected_return: Some("0"),
            reference: "fn divides(d: Int, n: Int) -> Bool = n % d == 0\n\
                        fn has_factor(n: Int, d: Int) -> Bool =\n  \
                          if d * d > n { false }\n  \
                          else { if divides(d, n) { true } else { has_factor(n, d + 1) } }\n\
                        fn is_prime(n: Int) -> Bool =\n  \
                          if n < 2 { false } else { !has_factor(n, 2) }\n\n\
                        fn main() -> Int = {\n  print_bool(is_prime(97));\n  0\n}\n",
        },
        Task {
            name: "reverse_list",
            prompt: "Define an algebraic list type `type List[T] = | Nil | Cons(T, List[T])`. \
                     Build the list 1, 2, 3, 4 (in that order), reverse it, and have `main` \
                     print the reversed list's elements with print_int, one per line, in \
                     the reversed order (last element first). Return 0.",
            expected_output: Some("4\n3\n2\n1\n"),
            expected_return: Some("0"),
            reference: "type List[T] = | Nil | Cons(T, List[T])\n\n\
                        fn rev_onto(xs: List[Int], acc: List[Int]) -> List[Int] =\n  \
                          match xs {\n    \
                            Nil => acc,\n    \
                            Cons(h, t) => rev_onto(t, Cons(h, acc)),\n  \
                          }\n\
                        fn print_all(xs: List[Int]) -> Int =\n  \
                          match xs {\n    \
                            Nil => 0,\n    \
                            Cons(h, t) => { print_int(h); print_all(t) },\n  \
                          }\n\n\
                        fn main() -> Int = {\n  \
                          let xs = Cons(1, Cons(2, Cons(3, Cons(4, Nil))));\n  \
                          let r = rev_onto(xs, Nil);\n  \
                          print_all(r);\n  0\n}\n",
        },
        Task {
            name: "sum_array",
            prompt: "Write a program that builds the array [10, 20, 30, 40] (using \
                     array_new and array_push) and has `main` print the sum of its \
                     elements with print_int, then returns 0.",
            expected_output: Some("100\n"),
            expected_return: Some("0"),
            reference: "fn main() -> Int = {\n  \
                          let a0 = array_new();\n  \
                          let a1 = array_push(a0, 10);\n  \
                          let a2 = array_push(a1, 20);\n  \
                          let a3 = array_push(a2, 30);\n  \
                          let a4 = array_push(a3, 40);\n  \
                          print_int(array_fold(a4, 0, \\(acc: Int, x: Int) -> acc + x));\n  \
                          0\n}\n",
        },
        Task {
            name: "count_evens",
            prompt: "Write a program that, using range(10) to get [0,1,...,9], counts how \
                     many of those numbers are even (divisible by 2) with array_filter and \
                     array_fold, and has `main` print that count with print_int, then \
                     returns 0.",
            expected_output: Some("5\n"),
            expected_return: Some("0"),
            reference: "fn main() -> Int = {\n  \
                          let evens = array_filter(range(10), \\x -> x % 2 == 0);\n  \
                          print_int(array_fold(evens, 0, \\(acc: Int, x: Int) -> acc + 1));\n  \
                          0\n}\n",
        },
        Task {
            name: "map_squares",
            prompt: "Write a program that maps x -> x*x over range(5) (i.e. over \
                     [0,1,2,3,4]) with array_map, sums the resulting squares with \
                     array_fold, and has `main` print that sum with print_int, then \
                     returns 0.",
            expected_output: Some("30\n"),
            expected_return: Some("0"),
            reference: "fn main() -> Int = {\n  \
                          let sq = array_map(range(5), \\x -> x * x);\n  \
                          print_int(array_fold(sq, 0, \\(acc: Int, x: Int) -> acc + x));\n  \
                          0\n}\n",
        },
        Task {
            name: "map_sum_values",
            prompt: "Build a Map (with map_new and map_insert) holding the entries \
                     1 -> 100, 2 -> 200, 3 -> 300. Have `main` print the sum of its \
                     values with print_int (using map_values and array_fold), then return \
                     0.",
            expected_output: Some("600\n"),
            expected_return: Some("0"),
            reference: "fn main() -> Int = {\n  \
                          let m0 = map_new();\n  \
                          let m1 = map_insert(m0, 1, 100);\n  \
                          let m2 = map_insert(m1, 2, 200);\n  \
                          let m3 = map_insert(m2, 3, 300);\n  \
                          print_int(array_fold(map_values(m3), 0, \\(acc: Int, x: Int) -> acc + x));\n  \
                          0\n}\n",
        },
        Task {
            name: "max_of_list",
            prompt: "Define `type List[T] = | Nil | Cons(T, List[T])`, build the list \
                     3, 7, 2, 9, 4, and have `main` print the maximum element with \
                     print_int using recursion and match, then return 0.",
            expected_output: Some("9\n"),
            expected_return: Some("0"),
            reference: "type List[T] = | Nil | Cons(T, List[T])\n\n\
                        fn max2(a: Int, b: Int) -> Int = if a > b { a } else { b }\n\
                        fn max_list(xs: List[Int], best: Int) -> Int =\n  \
                          match xs {\n    \
                            Nil => best,\n    \
                            Cons(h, t) => max_list(t, max2(best, h)),\n  \
                          }\n\n\
                        fn main() -> Int = {\n  \
                          let xs = Cons(3, Cons(7, Cons(2, Cons(9, Cons(4, Nil)))));\n  \
                          print_int(max_list(xs, 0 - 1000000));\n  0\n}\n",
        },
        Task {
            name: "record_field",
            prompt: "Define a record `type Point = { x: Int, y: Int }`, construct the point \
                     with x = 3 and y = 4, and have `main` print the sum x + y with \
                     print_int via field access, then return 0.",
            expected_output: Some("7\n"),
            expected_return: Some("0"),
            reference: "type Point = { x: Int, y: Int }\n\n\
                        fn main() -> Int = {\n  \
                          let p = Point { x: 3, y: 4 };\n  \
                          print_int(p.x + p.y);\n  0\n}\n",
        },
        Task {
            name: "gcd",
            prompt: "Write a program that computes the greatest common divisor (GCD) of \
                     48 and 36 using Euclid's algorithm (recursion with the modulo \
                     operator %), and has `main` print it with print_int, then return 0.",
            expected_output: Some("12\n"),
            expected_return: Some("0"),
            reference: "fn gcd(a: Int, b: Int) -> Int =\n  \
                          if b == 0 { a } else { gcd(b, a % b) }\n\n\
                        fn main() -> Int = {\n  print_int(gcd(48, 36));\n  0\n}\n",
        },
        Task {
            name: "set_dedup_count",
            prompt: "Build a Set (with set_new and set_add) by adding the numbers \
                     5, 3, 5, 1, 3, 5 in that order (with duplicates). Have `main` print \
                     the number of DISTINCT elements with print_int (using set_len), then \
                     return 0.",
            expected_output: Some("3\n"),
            expected_return: Some("0"),
            reference: "fn main() -> Int = {\n  \
                          let s0 = set_new();\n  \
                          let s1 = set_add(s0, 5);\n  \
                          let s2 = set_add(s1, 3);\n  \
                          let s3 = set_add(s2, 5);\n  \
                          let s4 = set_add(s3, 1);\n  \
                          let s5 = set_add(s4, 3);\n  \
                          let s6 = set_add(s5, 5);\n  \
                          print_int(set_len(s6));\n  0\n}\n",
        },
        Task {
            name: "string_build",
            prompt: "Write a program whose `main` computes the sum 1+2+3+4+5 and uses \
                     concat and int_to_str to build a string of the form `sum = N` where \
                     N is that sum (the literal text `sum = ` followed by the number), \
                     prints it with print_str on its own line, then returns 0.",
            expected_output: Some("sum = 15\n"),
            expected_return: Some("0"),
            reference: "fn main() -> Int = {\n  \
                          print_str(concat(\"sum = \", int_to_str(1 + 2 + 3 + 4 + 5)));\n  \
                          0\n}\n",
        },
    ]
}

/// Look up a task by exact name.
pub fn task_by_name(name: &str) -> Option<Task> {
    tasks().into_iter().find(|t| t.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every reference solution must (a) check CLEAN, (b) RUN, and (c) GRADE
    // CORRECT against its own oracle. This validates the TASKS themselves: a
    // task whose oracle disagreed with its (known-correct) reference is a bug in
    // the task, caught here.
    #[test]
    fn every_reference_checks_runs_and_grades_correct() {
        for t in tasks() {
            let diags = agent::check_program(t.reference);
            assert!(
                diags.is_empty(),
                "task `{}` reference should check clean, got: {:?}",
                t.name,
                diags.iter().map(|d| (d.code, d.message.clone())).collect::<Vec<_>>()
            );
            let (ret, out, err) = agent::run_program(t.reference);
            assert!(err.is_none(), "task `{}` reference should run: {:?}", t.name, err);
            // The reference must reproduce whatever the oracle pins down.
            if let Some(eo) = t.expected_output {
                assert_eq!(
                    out.as_deref(),
                    Some(eo),
                    "task `{}` reference output mismatch",
                    t.name
                );
            }
            if let Some(er) = t.expected_return {
                assert_eq!(
                    ret.as_deref(),
                    Some(er),
                    "task `{}` reference return mismatch",
                    t.name
                );
            }
            assert!(
                grade(&t, t.reference).is_correct(),
                "task `{}` reference should grade CORRECT",
                t.name
            );
        }
    }

    // The grader is not trivially passing: a deliberately WRONG program — one
    // that checks clean and RUNS but prints/returns the wrong thing — must be
    // graded INCORRECT. (Negative check required by the spec.)
    #[test]
    fn grader_fails_a_wrong_program() {
        let t = task_by_name("sum_1_to_100").expect("task exists");
        // Clean + runs, but prints 9999 instead of 5050 — converged but WRONG.
        let wrong = "fn main() -> Int = {\n  print_int(9999);\n  0\n}\n";
        assert!(agent::check_program(wrong).is_empty(), "wrong prog still checks clean");
        let g = grade(&t, wrong);
        assert!(!g.is_correct(), "wrong output must grade incorrect: {:?}", g);
        match g {
            Grade::Incorrect { why } => assert!(
                why.contains("output mismatch"),
                "expected an output-mismatch reason, got: {}",
                why
            ),
            Grade::Correct => panic!("should be incorrect"),
        }
    }

    // A program that checks clean but RETURNS the wrong value (right output,
    // wrong return) is also incorrect — proves the return-value oracle bites.
    #[test]
    fn grader_fails_wrong_return_value() {
        let t = task_by_name("constant").expect("task exists");
        // Prints 42 (right) but returns 1 (wrong — oracle wants 0).
        let wrong = "fn main() -> Int = {\n  print_int(42);\n  1\n}\n";
        assert!(agent::check_program(wrong).is_empty());
        let g = grade(&t, wrong);
        assert!(!g.is_correct(), "wrong return must grade incorrect: {:?}", g);
    }

    // A program that fails to RUN (e.g. division by zero) grades incorrect with a
    // runtime-error reason — never a panic.
    #[test]
    fn grader_fails_a_runtime_error_cleanly() {
        let t = task_by_name("constant").expect("task exists");
        let boom = "fn main() -> Int = {\n  print_int(1 / 0);\n  0\n}\n";
        let g = grade(&t, boom);
        assert!(!g.is_correct());
        match g {
            Grade::Incorrect { why } => assert!(why.contains("runtime error")),
            Grade::Correct => panic!("should be incorrect"),
        }
    }

    #[test]
    fn task_names_are_unique_and_findable() {
        let ts = tasks();
        let mut names: Vec<&str> = ts.iter().map(|t| t.name).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "task names must be unique");
        assert!(task_by_name("gcd").is_some());
        assert!(task_by_name("does_not_exist").is_none());
        assert!(n >= 12, "expected a spread of >=12 tasks, got {}", n);
    }

    // No prompt may STATE its expected answer in prose ("the sum is 100",
    // "gcd(48,36) = 12", ...): such phrasing lets a model echo the constant
    // (`print_int(100)`) and grade CORRECT without computing, inflating the
    // author-correctness measure. We check that none of the specific answer-
    // revealing phrases the prompts previously carried remain. (A bare answer
    // digit that legitimately appears among the INPUTS — e.g. `9` in the list
    // `3,7,2,9,4` for max_of_list — is not a leak; only an answer STATEMENT is.)
    #[test]
    fn no_prompt_states_its_answer() {
        // (task, banned answer-revealing substrings that must NOT appear).
        let banned: &[(&str, &[&str])] = &[
            ("fib_20", &["6765"]),
            ("sum_array", &["sum is 100", "= 100", "is 100"]),
            ("count_evens", &["5 even", "are 5"]),
            ("map_squares", &["sum to 30", "= 30", "is 30", "0,1,4,9,16"]),
            ("map_sum_values", &["is 600", "= 600", "sum of the values is"]),
            ("max_of_list", &["maximum is 9", "is 9", "= 9"]),
            ("record_field", &["i.e. 7", "= 7", "is 7"]),
            ("gcd", &["= 12", "is 12", "gcd(48, 36) ="]),
            ("set_dedup_count", &["3 distinct", "are 3", "(1, 3, 5)"]),
            ("string_build", &["15"]),
            ("reverse_list", &["4 then 3 then 2 then 1"]),
        ];
        for (name, subs) in banned {
            let t = task_by_name(name).unwrap_or_else(|| panic!("task `{}` exists", name));
            for s in *subs {
                assert!(
                    !t.prompt.contains(s),
                    "task `{}` prompt reveals its answer via {:?}:\n{}",
                    name,
                    s,
                    t.prompt
                );
            }
        }
    }

    // Spot-check: a constant-echo program (`print_int(<answer>)`) is NOT derivable
    // straight from the prompt text — the literal answer is absent — so a model
    // cannot copy it without doing the computation. (Grading is out-of-band, so
    // such an echo still grades CORRECT if it prints the right value; the point is
    // the PROMPT can no longer be the source of the constant.)
    #[test]
    fn constant_echo_not_copyable_from_prompt() {
        // These answers do NOT coincide with any number appearing in their prompt.
        for (name, answer) in [("sum_array", "100"), ("count_evens", "5"), ("gcd", "12")] {
            let t = task_by_name(name).expect("task exists");
            assert_eq!(t.expected_output, Some(format!("{}\n", answer).as_str()));
            assert!(
                !t.prompt.contains(answer),
                "task `{}` prompt still contains the constant {:?}",
                name,
                answer
            );
        }
    }
}
