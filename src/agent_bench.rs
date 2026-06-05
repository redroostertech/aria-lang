//! The AUTHORING BENCHMARK RUNNER: drive the agent loop over the task suite with
//! any provider, then GRADE each converged program, and report a single headline
//! number — the provider's author-correctness PASS-RATE — alongside convergence
//! and iterations-to-green.
//!
//! This is the measurement that turns "LLMs write Aria correctly" into a number.
//! For each task we (1) run the write -> check -> fix -> run loop
//! (`agent::run_loop`) with the chosen provider up to a budget, recording whether
//! it CONVERGED (checked clean + ran) and how many ITERATIONS it took, then
//! (2) GRADE the converged program against the task's out-of-band oracle
//! (`agent_tasks::grade`) — the expected answer is never in the prompt, so this
//! measures real author-correctness, not test-passing.
//!
//! `--provider reference` feeds each task its OWN reference solution (via
//! `agent::FixedProvider`), so the whole harness — capture, loop, grader, runner
//! — runs OFFLINE with no model and must report ~100% converged + 100% correct in
//! 1 iteration each. That is the self-test proving the machinery end to end.

use crate::agent::{self, FixedProvider, Provider};
use crate::agent_tasks::{self, Grade, Task};

/// The benchmark result for a single task.
pub struct TaskResult {
    pub name: &'static str,
    /// Did the loop reach a clean-checking, successfully-running program within
    /// the iteration budget?
    pub converged: bool,
    /// Iterations the loop used (to convergence, or the full budget on failure).
    pub iterations: usize,
    /// The grade of the converged program (`None` if it never converged, so
    /// there was nothing to grade).
    pub grade: Option<Grade>,
    /// A short note for the report (a runtime/provider error, or the grader's
    /// reason when incorrect). Empty when correct.
    pub note: String,
}

impl TaskResult {
    /// Did the produced program grade CORRECT? (Converged AND correct.)
    pub fn correct(&self) -> bool {
        matches!(self.grade, Some(Grade::Correct))
    }
}

/// The aggregate over a whole benchmark sweep — the headline metrics.
pub struct Aggregate {
    pub total: usize,
    pub converged: usize,
    pub correct: usize,
    /// Iteration counts of the CONVERGED tasks (for mean/median to-green).
    pub converged_iters: Vec<usize>,
}

impl Aggregate {
    pub fn convergence_rate(&self) -> f64 {
        pct(self.converged, self.total)
    }
    /// The HEADLINE number: fraction of tasks whose produced program was correct.
    pub fn correctness_rate(&self) -> f64 {
        pct(self.correct, self.total)
    }
    pub fn mean_iters(&self) -> Option<f64> {
        if self.converged_iters.is_empty() {
            return None;
        }
        let sum: usize = self.converged_iters.iter().sum();
        Some(sum as f64 / self.converged_iters.len() as f64)
    }
    pub fn median_iters(&self) -> Option<f64> {
        if self.converged_iters.is_empty() {
            return None;
        }
        let mut v = self.converged_iters.clone();
        v.sort_unstable();
        let n = v.len();
        Some(if n % 2 == 1 {
            v[n / 2] as f64
        } else {
            (v[n / 2 - 1] + v[n / 2]) as f64 / 2.0
        })
    }
}

fn pct(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

/// Build the provider for a task. For the special `reference` spec we hand the
/// task its OWN reference solution via a `FixedProvider` (the offline self-test);
/// for any other spec we build the ordinary provider from `agent::provider_from_spec`
/// (e.g. `mock`, `cmd:...`, `claude`, ...). Returns a fresh provider per task so
/// stateful providers (like `mock`) start clean for each.
fn provider_for_task(spec: &str, task: &Task) -> Result<Box<dyn Provider>, String> {
    if spec == "reference" {
        Ok(Box::new(FixedProvider::new(task.reference, "reference")))
    } else {
        agent::provider_from_spec(spec)
    }
}

/// Run the benchmark for ONE task: drive the loop, then grade. Never panics — a
/// provider/loop/grade failure is recorded as a non-converged or incorrect
/// result and the sweep continues.
pub fn run_task(spec: &str, task: &Task, max_iters: usize, verbose: bool) -> TaskResult {
    let provider = match provider_for_task(spec, task) {
        Ok(p) => p,
        Err(e) => {
            return TaskResult {
                name: task.name,
                converged: false,
                iterations: 0,
                grade: None,
                note: format!("provider error: {}", e),
            };
        }
    };

    let outcome = agent::run_loop(provider.as_ref(), task.prompt, max_iters, verbose);

    if !outcome.success {
        // Did not converge (budget exhausted, runtime error, or provider error).
        let note = outcome
            .runtime_error
            .clone()
            .unwrap_or_else(|| "did not converge within budget".to_string());
        return TaskResult {
            name: task.name,
            converged: false,
            iterations: outcome.iterations,
            grade: None,
            note,
        };
    }

    // Converged: grade the produced program against the task's oracle.
    let grade = agent_tasks::grade(task, &outcome.program);
    let note = match &grade {
        Grade::Correct => String::new(),
        Grade::Incorrect { why } => why.clone(),
    };
    TaskResult {
        name: task.name,
        converged: true,
        iterations: outcome.iterations,
        grade: Some(grade),
        note,
    }
}

/// Run the benchmark over `tasks` and return the per-task results + aggregate.
pub fn run(spec: &str, tasks: &[Task], max_iters: usize, verbose: bool) -> (Vec<TaskResult>, Aggregate) {
    let mut results = Vec::with_capacity(tasks.len());
    for t in tasks {
        results.push(run_task(spec, t, max_iters, verbose));
    }
    let total = results.len();
    let converged = results.iter().filter(|r| r.converged).count();
    let correct = results.iter().filter(|r| r.correct()).count();
    let converged_iters: Vec<usize> =
        results.iter().filter(|r| r.converged).map(|r| r.iterations).collect();
    let agg = Aggregate { total, converged, correct, converged_iters };
    (results, agg)
}

/// Render the full, greppable report (per-task lines + the aggregate summary) as
/// a `String`, so it can be both printed and asserted on in tests. Each per-task
/// line is prefixed `TASK ` and each aggregate line `BENCH `, for easy grepping.
pub fn render_report(spec: &str, results: &[TaskResult], agg: &Aggregate) -> String {
    let mut s = String::new();
    s.push_str(&format!("== aria authoring benchmark :: provider `{}` ==\n", spec));
    // Column header for the per-task table.
    s.push_str(&format!(
        "TASK {:<18} {:>9} {:>6} {:>8}  note\n",
        "name", "converged", "iters", "correct"
    ));
    for r in results {
        let converged = if r.converged { "yes" } else { "no" };
        let correct = match &r.grade {
            Some(Grade::Correct) => "yes",
            Some(Grade::Incorrect { .. }) => "no",
            None => "-",
        };
        s.push_str(&format!(
            "TASK {:<18} {:>9} {:>6} {:>8}  {}\n",
            r.name, converged, r.iterations, correct, r.note
        ));
    }
    // Aggregate summary — the headline is the CORRECTNESS pass-rate.
    let mean = agg
        .mean_iters()
        .map(|m| format!("{:.2}", m))
        .unwrap_or_else(|| "n/a".to_string());
    let median = agg
        .median_iters()
        .map(|m| format!("{:.1}", m))
        .unwrap_or_else(|| "n/a".to_string());
    s.push_str("---\n");
    s.push_str(&format!(
        "BENCH convergence {:.1}% ({}/{})\n",
        agg.convergence_rate(),
        agg.converged,
        agg.total
    ));
    s.push_str(&format!(
        "BENCH correctness {:.1}% ({}/{})  <- author-correctness pass-rate\n",
        agg.correctness_rate(),
        agg.correct,
        agg.total
    ));
    s.push_str(&format!(
        "BENCH iters-to-green mean {} median {} (over {} converged)\n",
        mean,
        median,
        agg.converged_iters.len()
    ));
    s.push_str(&format!(
        "BENCH counts total={} converged={} correct={} incorrect={} nonconverged={}\n",
        agg.total,
        agg.converged,
        agg.correct,
        agg.converged - agg.correct,
        agg.total - agg.converged,
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    // The reference provider must drive the WHOLE harness — capture, loop,
    // grader, runner — to ~100% converged + 100% correct in 1 iteration each,
    // entirely offline. This is the end-to-end self-test.
    #[test]
    fn reference_provider_scores_full_marks_offline() {
        let tasks = agent_tasks::tasks();
        let (results, agg) = run("reference", &tasks, 3, false);
        assert_eq!(agg.total, tasks.len());
        assert_eq!(agg.converged, agg.total, "reference should converge every task");
        assert_eq!(agg.correct, agg.total, "reference should be correct on every task");
        assert!((agg.correctness_rate() - 100.0).abs() < 1e-9);
        // Each converged in exactly ONE iteration (the reference checks clean
        // immediately — no feedback round needed).
        for r in &results {
            assert!(r.converged, "task `{}` should converge", r.name);
            assert!(r.correct(), "task `{}` should be correct", r.name);
            assert_eq!(r.iterations, 1, "task `{}` should take 1 iteration", r.name);
        }
        assert_eq!(agg.mean_iters(), Some(1.0));
        assert_eq!(agg.median_iters(), Some(1.0));
    }

    // The report is sane and greppable: it carries the headline correctness line
    // at 100% and a per-task line for every task.
    #[test]
    fn reference_report_is_greppable_and_full_marks() {
        let tasks = agent_tasks::tasks();
        let (results, agg) = run("reference", &tasks, 3, false);
        let report = render_report("reference", &results, &agg);
        assert!(report.contains("BENCH correctness 100.0%"), "report:\n{}", report);
        assert!(report.contains("author-correctness pass-rate"));
        assert!(report.contains("BENCH convergence 100.0%"));
        // One TASK line per task (plus the header line, which also starts TASK).
        let task_lines = report.lines().filter(|l| l.starts_with("TASK ")).count();
        assert_eq!(task_lines, tasks.len() + 1, "one header + one line per task");
        // Spot-check a couple of task names appear.
        assert!(report.contains("gcd"));
        assert!(report.contains("sum_1_to_100"));
    }

    // A provider that converges to a WRONG program is measured as converged-but-
    // incorrect: convergence high, correctness low. This proves the runner
    // distinguishes "runs" from "right". We use a FixedProvider feeding a clean-
    // checking program that prints the wrong number for `sum_1_to_100`.
    #[test]
    fn converged_but_incorrect_is_measured() {
        let task = agent_tasks::task_by_name("sum_1_to_100").unwrap();
        // This is NOT the reference: it checks clean and runs, but prints 1.
        struct WrongButClean;
        impl Provider for WrongButClean {
            fn complete(&self, _p: &str) -> Result<String, String> {
                Ok("```aria\nfn main() -> Int = { print_int(1); 0 }\n```".to_string())
            }
            fn label(&self) -> String {
                "wrong".to_string()
            }
        }
        let provider = WrongButClean;
        let outcome = agent::run_loop(&provider, task.prompt, 3, false);
        assert!(outcome.success, "wrong-but-clean program still converges");
        let grade = agent_tasks::grade(&task, &outcome.program);
        assert!(!grade.is_correct(), "but it must grade INCORRECT");
    }

    // The runner never panics on a provider that always fails: the task is
    // recorded as non-converged, the sweep produces a sane (zero-correct)
    // aggregate.
    #[test]
    fn failing_provider_yields_sane_zero_aggregate() {
        // `cmd:exit 7` spawns a shell that exits non-zero -> provider error every
        // call -> the loop fails -> non-converged. (No model, no network.)
        let tasks: Vec<Task> = agent_tasks::tasks().into_iter().take(2).collect();
        let (results, agg) = run("cmd:exit 7", &tasks, 2, false);
        assert_eq!(agg.total, 2);
        assert_eq!(agg.converged, 0);
        assert_eq!(agg.correct, 0);
        assert!(agg.mean_iters().is_none(), "no converged tasks -> no mean");
        assert!(results.iter().all(|r| !r.converged));
        // The report still renders cleanly with n/a iterations.
        let report = render_report("cmd:exit 7", &results, &agg);
        assert!(report.contains("BENCH correctness 0.0%"));
        assert!(report.contains("iters-to-green mean n/a"));
    }
}
