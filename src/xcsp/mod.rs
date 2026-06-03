//! XCSP3-core front-end: parse via `xcsp3-rust-parser`, build through
//! [`callback::Model`], solve, and report competition-style output.

mod callback;

use std::fmt::Display;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use xcsp3_rust_parser::xcsp_runner::XcspRunner;

use crate::ids::VarId;
use crate::parallel::{normalize_options, solve_cop, worker_roles, ParallelOutcome};
use crate::problem::{Objective, Problem};
use crate::search::{optimize_seeded, solve_interruptible_seeded, SearchControl, SolveStats};

pub use crate::parallel::RunOptions;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct StagedXml(PathBuf);

impl Drop for StagedXml {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

const MAX_VALUE_LINE_LEN: usize = 4096;

fn write_tokens<W: Write>(w: &mut W, tokens: impl IntoIterator<Item = impl Display>) -> std::io::Result<()> {
    let mut line = String::from("v");
    for token in tokens {
        let token = token.to_string();
        if line.len() > 1 && line.len() + token.len() + 1 > MAX_VALUE_LINE_LEN {
            writeln!(w, "{line}")?;
            line.truncate(1);
        }
        line.push(' ');
        line.push_str(&token);
    }
    if line.len() > 1 {
        writeln!(w, "{line}")?;
    }
    Ok(())
}

fn write_improvement<W: Write>(w: &mut W, verbose: bool, error: &mut Option<std::io::Error>, value: impl Display) {
    if verbose && error.is_none() {
        *error = writeln!(w, "o {value}").and_then(|_| writeln!(w, "c incumbent {value} source sequential")).and_then(|_| w.flush()).err();
    }
}

fn write_optimization_result<W: Write>(
    w: &mut W,
    output: &SolutionOutput,
    best: Option<(Vec<i32>, impl Display)>,
    stats: SolveStats,
    complete: bool,
    interrupted: bool,
    verbose: bool,
) -> Result<(), String> {
    let to_err = |e: std::io::Error| e.to_string();
    if verbose {
        writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures).map_err(to_err)?;
    }
    match best {
        Some((solution, value)) => {
            if !verbose {
                writeln!(w, "o {value}").map_err(to_err)?;
            }
            writeln!(w, "{}", if complete { "s OPTIMUM FOUND" } else { "s SATISFIABLE" }).map_err(to_err)?;
            output.write(w, &solution).map_err(to_err)?;
        }
        None if interrupted => writeln!(w, "s UNKNOWN").map_err(to_err)?,
        None => writeln!(w, "s UNSATISFIABLE").map_err(to_err)?,
    }
    Ok(())
}

fn stage_xml(xml: &str) -> Result<StagedXml, String> {
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("qayd-{}-{}.xml", std::process::id(), n));
    std::fs::write(&path, xml).map_err(|e| e.to_string())?;
    Ok(StagedXml(path))
}

#[derive(Clone)]
struct SolutionOutput {
    names: Vec<String>,
    entries: Vec<Option<usize>>,
}

impl SolutionOutput {
    fn write<W: Write>(&self, w: &mut W, solution: &[i32]) -> std::io::Result<()> {
        writeln!(w, "v <instantiation>")?;
        writeln!(w, "v <list>")?;
        write_tokens(w, &self.names)?;
        writeln!(w, "v </list>")?;
        writeln!(w, "v <values>")?;
        write_tokens(w, self.entries.iter().map(|&position| position.map_or_else(|| "*".to_string(), |i| solution[i].to_string())))?;
        writeln!(w, "v </values>")?;
        writeln!(w, "v </instantiation>")
    }
}

struct XcspProblem {
    problem: Problem,
    output: SolutionOutput,
}

fn parse_problem(path: &Path) -> Result<XcspProblem, String> {
    let mut model = callback::Model::new();
    let path_str = path.to_str().ok_or("bad temp path")?;
    let parsed = XcspRunner::run(path_str, &mut model);
    parsed.map_err(|e| e.to_string())?;
    if let Some(e) = model.error.take() {
        return Err(e);
    }
    let mut relevant = (0..model.solver.store.num_vars()).map(|i| model.solver.store.is_relevant(VarId(i as u32))).collect::<Vec<_>>();
    if let Some(objective) = &model.objective {
        match objective {
            Objective::Var(_, var) => relevant[var.index()] = true,
            Objective::Linear(_, _, vars) => {
                for &var in vars {
                    relevant[var.index()] = true;
                }
            }
            Objective::Expr(_, expr) => {
                let mut vars = Vec::new();
                expr.collect_vars(&mut vars);
                for var in vars {
                    relevant[var.index()] = true;
                }
            }
        }
    }
    let search = relevant.iter().enumerate().filter_map(|(i, &used)| used.then_some(VarId(i as u32))).collect::<Vec<_>>();
    let mut positions = vec![None; model.solver.store.num_vars()];
    for (i, &var) in search.iter().enumerate() {
        positions[var.index()] = Some(i);
    }
    let (names, entries) = model.declared.iter().map(|(name, var)| (name.clone(), positions[var.index()])).unzip();
    let output = SolutionOutput { names, entries };
    let problem = Problem { solver: model.solver, search, objective: model.objective };
    Ok(XcspProblem { problem, output })
}

/// Parse, build, and solve an XCSP3 instance, returning competition-style
/// output lines as a string. No `c` comments.
pub fn run(xml: &str) -> Result<String, String> {
    let mut buf = Vec::new();
    let never = AtomicBool::new(false);
    run_to(xml, false, &never, &mut buf)?;
    String::from_utf8(buf).map_err(|e| e.to_string())
}

/// Solve an XCSP3 instance, streaming output to `w`. When `verbose`, emits `c`
/// comment lines. Search halts as soon as `stop` is set (time limit / Ctrl+C).
pub fn run_to<W: Write>(xml: &str, verbose: bool, stop: &AtomicBool, w: &mut W) -> Result<(), String> {
    run_to_with_options(xml, verbose, stop, w, RunOptions::default())
}

/// Like [`run_to`], with explicit seed and portfolio-worker settings.
pub fn run_to_with_options<W: Write>(xml: &str, verbose: bool, stop: &AtomicBool, w: &mut W, options: RunOptions) -> Result<(), String> {
    let start = Instant::now();
    let options = RunOptions { workers: options.workers.max(1), ..options };
    let path = stage_xml(xml)?;
    let xcsp = parse_problem(&path.0)?;
    let has_objective = xcsp.problem.objective.is_some();
    let var_objective = xcsp.problem.var_objective().is_some();
    let symbolic_probes_disabled = has_objective && !var_objective && options.probes > 0;
    let options = normalize_options(has_objective, var_objective, options);
    let to_err = |e: std::io::Error| e.to_string();

    if verbose {
        let kind = if xcsp.problem.objective.is_some() { "COP" } else { "CSP" };
        writeln!(w, "c qayd XCSP3").map_err(to_err)?;
        writeln!(w, "c type {kind}").map_err(to_err)?;
        writeln!(w, "c variables {}", xcsp.problem.solver.store.num_vars()).map_err(to_err)?;
        writeln!(w, "c sparse domains {}", xcsp.problem.solver.store.num_sparse_domains()).map_err(to_err)?;
        writeln!(w, "c bounds domains {}", xcsp.problem.solver.store.num_bounds_domains()).map_err(to_err)?;
        writeln!(w, "c search variables {}", xcsp.problem.search.len()).map_err(to_err)?;
        writeln!(w, "c propagators {}", xcsp.problem.solver.num_propagators()).map_err(to_err)?;
        writeln!(w, "c seed {}", options.seed).map_err(to_err)?;
        writeln!(w, "c workers {} ({})", options.workers, worker_roles(has_objective, options)).map_err(to_err)?;
        writeln!(w, "c split {}", options.split).map_err(to_err)?;
        if symbolic_probes_disabled {
            writeln!(w, "c probes disabled (probe worker only supports a materialized objective variable)").map_err(to_err)?;
        }
    }

    if options.workers == 1 || !has_objective {
        solve_single(xcsp, verbose, stop, w, options.seed)?;
    } else {
        let XcspProblem { problem, output } = xcsp;
        let result = solve_cop(problem, verbose, stop, w, options)?;
        write_parallel_result(w, &output, result, verbose)?;
    }

    if verbose {
        if stop.load(Ordering::Relaxed) {
            writeln!(w, "c interrupted").map_err(to_err)?;
        }
        writeln!(w, "c time {:.3}s", start.elapsed().as_secs_f64()).map_err(to_err)?;
    }
    w.flush().map_err(to_err)?;
    Ok(())
}

fn solve_single<W: Write>(xcsp: XcspProblem, verbose: bool, stop: &AtomicBool, w: &mut W, seed: u64) -> Result<(), String> {
    let XcspProblem { problem, output } = xcsp;
    let mut solver = problem.solver;
    let vars = problem.search;
    let to_err = |e: std::io::Error| e.to_string();

    match problem.objective {
        None => {
            let mut sol = None;
            let stats = solve_interruptible_seeded(
                &mut solver,
                &vars,
                |s| {
                    let active = vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>();
                    sol = Some(active);
                    SearchControl::Stop
                },
                stop,
                seed,
            );
            if verbose {
                writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures).map_err(to_err)?;
            }
            match sol {
                Some(s) => {
                    writeln!(w, "s SATISFIABLE").map_err(to_err)?;
                    output.write(w, &s).map_err(to_err)?;
                }
                None if stop.load(Ordering::Relaxed) => writeln!(w, "s UNKNOWN").map_err(to_err)?,
                None => writeln!(w, "s UNSATISFIABLE").map_err(to_err)?,
            }
            Ok(())
        }
        Some(objective) => {
            let minimizing = objective.minimizing();
            let mut io_err: Option<std::io::Error> = None;
            let (best, stats, complete) =
                optimize_seeded(&mut solver, &vars, objective.search(), minimizing, stop, seed, None, None, &[], None, |v, _| {
                    write_improvement(w, verbose, &mut io_err, v)
                });
            if let Some(e) = io_err {
                return Err(e.to_string());
            }
            let interrupted = stop.load(Ordering::Relaxed);
            write_optimization_result(w, &output, best, stats, complete, interrupted, verbose)
        }
    }
}

fn write_parallel_result<W: Write>(w: &mut W, output: &SolutionOutput, result: ParallelOutcome, verbose: bool) -> Result<(), String> {
    let to_err = |e: std::io::Error| e.to_string();
    if verbose {
        writeln!(w, "c nodes {} failures {}", result.stats.nodes, result.stats.failures).map_err(to_err)?;
        writeln!(w, "c shared clauses {} imported {}", result.shared_clauses, result.imported_clauses).map_err(to_err)?;
        if let Some((split, completed)) = result.split_jobs {
            writeln!(w, "c split jobs {split} completed {completed}").map_err(to_err)?;
        }
        if let Some((attempts, unsat)) = result.probe_stats {
            writeln!(w, "c probes attempts {attempts} unsat {unsat}").map_err(to_err)?;
        }
        if let Some((attempts, improved)) = result.lns_stats {
            writeln!(w, "c lns attempts {attempts} improved {improved}").map_err(to_err)?;
        }
    }
    match result.best {
        Some((solution, value)) => {
            if !verbose {
                writeln!(w, "o {value}").map_err(to_err)?;
            }
            writeln!(w, "{}", if result.proved { "s OPTIMUM FOUND" } else { "s SATISFIABLE" }).map_err(to_err)?;
            output.write(w, &solution).map_err(to_err)?;
        }
        None if result.interrupted => writeln!(w, "s UNKNOWN").map_err(to_err)?,
        None if result.proved => writeln!(w, "s UNSATISFIABLE").map_err(to_err)?,
        None => writeln!(w, "s UNKNOWN").map_err(to_err)?,
    }
    Ok(())
}
