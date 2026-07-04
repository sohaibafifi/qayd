//! XCSP3-core front-end: parse via `xcsp3-rust-parser`, build through
//! [`callback::Model`], solve, and report competition-style output.

mod callback;

use std::fmt::Display;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use xcsp3_rust_parser::xcsp_runner::XcspRunner;

use crate::engines::ls::cop::{solve_ls, LocalSearchOutcome, LocalSearchSpec, LsConfig};
use crate::ids::VarId;
use crate::parallel::{normalize_options, solve_cop, solve_csp, worker_roles, CspOutcome, ParallelOutcome};
use crate::problem::{Objective, Problem};
use crate::search::{decide_sat_seeded, find_one_seeded, optimize_seeded, SolveStats};

pub use crate::parallel::{Mode, RunOptions};

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

fn io_err(e: std::io::Error) -> String {
    e.to_string()
}

/// Write a result's tail: the optional `o {value}` line (non-verbose runs with a
/// known objective), the status line, and the instantiation. `sat`/`none` are
/// the status lines for the solved / unsolved cases; `value` is `None` for CSP.
fn write_solution_tail<W: Write>(
    w: &mut W,
    output: &SolutionOutput,
    best: Option<(Vec<i32>, Option<impl Display>)>,
    sat: &str,
    none: &str,
    verbose: bool,
) -> Result<(), String> {
    match best {
        Some((solution, value)) => {
            if !verbose {
                if let Some(value) = value {
                    writeln!(w, "o {value}").map_err(io_err)?;
                }
            }
            writeln!(w, "{sat}").map_err(io_err)?;
            output.write(w, &solution).map_err(io_err)?;
        }
        None => writeln!(w, "{none}").map_err(io_err)?,
    }
    Ok(())
}

fn write_improvement_from<W: Write>(w: &mut W, verbose: bool, error: &mut Option<std::io::Error>, value: impl Display, source: &str) {
    if verbose && error.is_none() {
        *error = writeln!(w, "o {value}").and_then(|_| writeln!(w, "c incumbent {value} source {source}")).and_then(|_| w.flush()).err();
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
    if verbose {
        writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures).map_err(io_err)?;
        write_inprocessing_stats(w, stats).map_err(io_err)?;
    }
    let sat = if complete { "s OPTIMUM FOUND" } else { "s SATISFIABLE" };
    let none = if interrupted { "s UNKNOWN" } else { "s UNSATISFIABLE" };
    write_solution_tail(w, output, best.map(|(s, v)| (s, Some(v))), sat, none, verbose)
}

fn write_inprocessing_stats<W: Write>(w: &mut W, stats: SolveStats) -> std::io::Result<()> {
    if stats.vivified_clauses > 0 {
        writeln!(w, "c inprocessing vivified {} removed {}", stats.vivified_clauses, stats.vivified_lits)?;
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
    entries: Vec<SolutionEntry>,
}

#[derive(Clone, Copy)]
enum SolutionEntry {
    Star,
    Search(usize),
    Fixed(i32),
}

impl SolutionOutput {
    fn write<W: Write>(&self, w: &mut W, solution: &[i32]) -> std::io::Result<()> {
        writeln!(w, "v <instantiation>")?;
        writeln!(w, "v <list>")?;
        write_tokens(w, &self.names)?;
        writeln!(w, "v </list>")?;
        writeln!(w, "v <values>")?;
        write_tokens(
            w,
            self.entries.iter().map(|entry| match *entry {
                SolutionEntry::Star => "*".to_string(),
                SolutionEntry::Search(i) => solution[i].to_string(),
                SolutionEntry::Fixed(value) => value.to_string(),
            }),
        )?;
        writeln!(w, "v </values>")?;
        writeln!(w, "v </instantiation>")
    }

    fn remap_after_presolve(&mut self, old_search: &[VarId], problem: &Problem) {
        let mut positions = vec![None; problem.solver.store.num_vars()];
        for (i, &var) in problem.search.iter().enumerate() {
            positions[var.index()] = Some(i);
        }
        for entry in &mut self.entries {
            let SolutionEntry::Search(old_position) = *entry else {
                continue;
            };
            let var = old_search[old_position];
            *entry = if let Some(new_position) = positions[var.index()] {
                SolutionEntry::Search(new_position)
            } else if problem.solver.store.is_fixed(var) {
                SolutionEntry::Fixed(problem.solver.store.value(var))
            } else {
                SolutionEntry::Star
            };
        }
    }
}

struct XcspProblem {
    problem: Problem,
    local: LocalSearchSpec,
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
    let (names, entries) = model
        .declared
        .iter()
        .map(|(name, var)| {
            let entry = positions[var.index()].map_or(SolutionEntry::Star, SolutionEntry::Search);
            (name.clone(), entry)
        })
        .unzip();
    let output = SolutionOutput { names, entries };
    let problem = Problem { solver: model.solver, search, objective: model.objective };
    Ok(XcspProblem { problem, local: model.local, output })
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
    crate::mem::set_limit_mb(options.mem_limit.unwrap_or(0));
    crate::mem::set_verbose(verbose && options.mem_limit.is_some());
    let path = stage_xml(xml)?;
    let mut xcsp = parse_problem(&path.0)?;
    let has_objective = xcsp.problem.objective.is_some();
    let var_objective = xcsp.problem.var_objective().is_some();
    if options.local_search() && !has_objective {
        return Err("--ls requires a COP instance".to_string());
    }
    let symbolic_probes_disabled = has_objective && !var_objective && options.probes > 0;
    let options = normalize_options(has_objective, var_objective, options);
    let old_search = xcsp.problem.search.clone();
    let presolve = xcsp.problem.presolve(stop);
    xcsp.output.remap_after_presolve(&old_search, &xcsp.problem);

    if verbose {
        let kind = if xcsp.problem.objective.is_some() { "COP" } else { "CSP" };
        writeln!(w, "c qayd XCSP3").map_err(io_err)?;
        writeln!(w, "c type {kind}").map_err(io_err)?;
        if options.local_search() {
            writeln!(w, "c mode ls").map_err(io_err)?;
            writeln!(w, "c effort incumbent").map_err(io_err)?;
        }
        writeln!(w, "c variables {}", xcsp.problem.solver.store.num_vars()).map_err(io_err)?;
        writeln!(w, "c sparse domains {}", xcsp.problem.solver.store.num_sparse_domains()).map_err(io_err)?;
        writeln!(w, "c bounds domains {}", xcsp.problem.solver.store.num_bounds_domains()).map_err(io_err)?;
        writeln!(w, "c search variables {}", xcsp.problem.search.len()).map_err(io_err)?;
        writeln!(w, "c propagators {}", xcsp.problem.solver.num_propagators()).map_err(io_err)?;
        writeln!(w, "c seed {}", options.seed).map_err(io_err)?;
        writeln!(w, "c workers {} ({})", options.workers, worker_roles(has_objective, options)).map_err(io_err)?;
        if !has_objective {
            writeln!(w, "c csp learning {}", !options.no_learn_csp).map_err(io_err)?;
        }
        writeln!(w, "c split {}", options.split).map_err(io_err)?;
        if presolve.failed {
            writeln!(w, "c presolve failed").map_err(io_err)?;
        } else if presolve.stopped {
            writeln!(w, "c presolve stopped").map_err(io_err)?;
        } else if presolve.search_before != presolve.search_after || presolve.fixed > 0 {
            writeln!(w, "c presolve reduced search {} -> {} fixed {}", presolve.search_before, presolve.search_after, presolve.fixed)
                .map_err(io_err)?;
        }
        if symbolic_probes_disabled {
            writeln!(w, "c probes disabled (probe worker only supports a materialized objective variable)").map_err(io_err)?;
        }
    }

    if presolve.failed || presolve.stopped {
        if verbose {
            writeln!(w, "c nodes 0 failures {}", usize::from(presolve.failed)).map_err(io_err)?;
        }
        writeln!(w, "{}", if presolve.failed { "s UNSATISFIABLE" } else { "s UNKNOWN" }).map_err(io_err)?;
        if verbose {
            if stop.load(Ordering::Relaxed) {
                writeln!(w, "c interrupted").map_err(io_err)?;
            }
            writeln!(w, "c time {:.3}s", start.elapsed().as_secs_f64()).map_err(io_err)?;
        }
        w.flush().map_err(io_err)?;
        return Ok(());
    }

    // `--ls` is local-search-first: always run the LS engine, even single worker.
    if options.local_search() {
        let XcspProblem { problem, local, output } = xcsp;
        let result = solve_ls_parallel(problem, local, verbose, stop, w, options)?;
        write_ls_result(w, &output, result, verbose)?;
    } else if options.workers == 1 {
        solve_single(xcsp, verbose, stop, w, options)?;
    } else if !has_objective {
        let XcspProblem { problem, output, .. } = xcsp;
        let result = solve_csp(problem, stop, options);
        write_csp_result(w, &output, result, verbose)?;
    } else {
        let XcspProblem { problem, local, output } = xcsp;
        let result = solve_cop(problem, local, verbose, stop, w, options)?;
        write_parallel_result(w, &output, result, verbose)?;
    }

    if verbose {
        if stop.load(Ordering::Relaxed) {
            writeln!(w, "c interrupted").map_err(io_err)?;
        }
        writeln!(w, "c time {:.3}s", start.elapsed().as_secs_f64()).map_err(io_err)?;
        let peak = crate::mem::peak();
        if peak > 0 {
            writeln!(w, "c peak memory {} MB", peak / (1024 * 1024)).map_err(io_err)?;
        }
    }
    w.flush().map_err(io_err)?;
    Ok(())
}

enum LsMsg {
    Improved { value: i64, source: &'static str, worker: usize },
    Done(LocalSearchOutcome),
}

fn solve_ls_parallel<W: Write>(
    problem: Problem,
    local: LocalSearchSpec,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
    options: RunOptions,
) -> Result<LocalSearchOutcome, String> {
    let minimizing = problem.objective.as_ref().is_none_or(Objective::minimizing);
    // The `--ls` engine runs the autonomous local-search upgrades: GLS to escape
    // local minima and min-conflicts value selection on large domains.
    let ls_config = LsConfig { gls: true, min_conflicts: true, kick_bandit: false };
    let (tx, rx) = mpsc::channel();
    let mut printed = None;
    let mut summary = None;
    let mut error = None;

    std::thread::scope(|scope| {
        for worker in 0..options.workers {
            let tx = tx.clone();
            let problem = problem.clone();
            let local = local.clone();
            scope.spawn(move || {
                let seed = options.seed.wrapping_add(worker as u64);
                let result = solve_ls(problem, local, stop, seed, ls_config, |value, _, source| {
                    let _ = tx.send(LsMsg::Improved { value, source, worker });
                });
                let _ = tx.send(LsMsg::Done(result));
            });
        }
        drop(tx);

        for msg in rx {
            match msg {
                LsMsg::Improved { value, source, worker, .. }
                    if printed.is_none_or(|old| if minimizing { value < old } else { value > old }) =>
                {
                    printed = Some(value);
                    if verbose && error.is_none() {
                        if let Err(e) = writeln!(w, "o {value}")
                            .and_then(|_| writeln!(w, "c incumbent {value} source {source} worker {worker}"))
                            .and_then(|_| w.flush())
                        {
                            error = Some(e.to_string());
                            stop.store(true, Ordering::Relaxed);
                        }
                    }
                }
                LsMsg::Improved { .. } => {}
                LsMsg::Done(result) => merge_ls_result(&mut summary, result, minimizing),
            }
        }
    });

    if let Some(e) = error {
        return Err(e);
    }
    Ok(summary.unwrap_or(LocalSearchOutcome {
        best: None,
        iterations: 0,
        moves: 0,
        restarts: 0,
        constraints: 0,
        functionals: 0,
        unsupported: 1,
    }))
}

fn merge_ls_result(summary: &mut Option<LocalSearchOutcome>, mut result: LocalSearchOutcome, minimizing: bool) {
    let Some(current) = summary else {
        *summary = Some(result);
        return;
    };
    current.iterations += result.iterations;
    current.moves += result.moves;
    current.restarts += result.restarts;
    current.constraints = current.constraints.max(result.constraints);
    current.functionals = current.functionals.max(result.functionals);
    current.unsupported = current.unsupported.max(result.unsupported);
    if let Some((solution, value)) = result.best.take() {
        if current.best.as_ref().is_none_or(|(_, old)| if minimizing { value < *old } else { value > *old }) {
            current.best = Some((solution, value));
        }
    }
}

fn write_ls_result<W: Write>(w: &mut W, output: &SolutionOutput, result: LocalSearchOutcome, verbose: bool) -> Result<(), String> {
    if verbose {
        writeln!(w, "c local constraints {} functionals {}", result.constraints, result.functionals).map_err(io_err)?;
        if result.unsupported > 0 {
            writeln!(w, "c local unsupported {}", result.unsupported).map_err(io_err)?;
        }
        if result.iterations > 0 {
            writeln!(w, "c local iterations {} moves {} restarts {}", result.iterations, result.moves, result.restarts).map_err(io_err)?;
        }
    }
    write_solution_tail(w, output, result.best.map(|(s, v)| (s, Some(v))), "s SATISFIABLE", "s UNKNOWN", verbose)
}

fn solve_single<W: Write>(xcsp: XcspProblem, verbose: bool, stop: &AtomicBool, w: &mut W, options: RunOptions) -> Result<(), String> {
    let XcspProblem { problem, output, .. } = xcsp;
    let mut solver = problem.solver;
    let vars = problem.search;

    match problem.objective {
        None => {
            // Find-one CSP learns by default: CDCL with restarts and root probing
            // dominates plain chronological DFS, and never enumerates, so the
            // soundness caveat on learned blocking clauses does not apply here.
            // (Enumeration/counting uses the separate non-learning `enumerate`.)
            // `--no-learn-csp` opts out to the non-learning DFS driver for ablation.
            let (sol, stats, complete) = if options.no_learn_csp {
                find_one_seeded(&mut solver, &vars, stop, options.seed)
            } else {
                decide_sat_seeded(&mut solver, &vars, stop, options.seed)
            };
            if verbose {
                writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures).map_err(io_err)?;
                write_inprocessing_stats(w, stats).map_err(io_err)?;
            }
            match sol {
                Some(s) => {
                    writeln!(w, "s SATISFIABLE").map_err(io_err)?;
                    output.write(w, &s).map_err(io_err)?;
                }
                None if !complete => writeln!(w, "s UNKNOWN").map_err(io_err)?,
                None => writeln!(w, "s UNSATISFIABLE").map_err(io_err)?,
            }
            Ok(())
        }
        Some(objective) => {
            let minimizing = objective.minimizing();
            let mut io_err: Option<std::io::Error> = None;
            // Single-worker COP is always the proof engine here; `--ls` routes to
            // the local-search portfolio before reaching this path.
            let (best, stats, complete) = optimize_seeded(
                &mut solver,
                &vars,
                objective.search(),
                minimizing,
                stop,
                options.seed,
                None,
                None,
                &[],
                None,
                Vec::new(),
                |v, _| write_improvement_from(w, verbose, &mut io_err, v, "sequential"),
            );
            if let Some(e) = io_err {
                return Err(e.to_string());
            }
            let interrupted = stop.load(Ordering::Relaxed);
            write_optimization_result(w, &output, best, stats, complete, interrupted, verbose)
        }
    }
}

fn write_parallel_result<W: Write>(w: &mut W, output: &SolutionOutput, result: ParallelOutcome, verbose: bool) -> Result<(), String> {
    if verbose {
        writeln!(w, "c nodes {} failures {}", result.stats.nodes, result.stats.failures).map_err(io_err)?;
        write_inprocessing_stats(w, result.stats).map_err(io_err)?;
        if result.shared_clauses > 0 || result.imported_clauses > 0 {
            writeln!(w, "c shared clauses {} imported {}", result.shared_clauses, result.imported_clauses).map_err(io_err)?;
        }
        if let Some((split, completed)) = result.split_jobs {
            writeln!(w, "c split jobs {split} completed {completed}").map_err(io_err)?;
        }
        if let Some((attempts, unsat)) = result.probe_stats {
            writeln!(w, "c probes attempts {attempts} unsat {unsat}").map_err(io_err)?;
        }
        if let Some((attempts, improved)) = result.lns_stats {
            writeln!(w, "c lns attempts {attempts} improved {improved}").map_err(io_err)?;
        }
    }
    let sat = if result.proved { "s OPTIMUM FOUND" } else { "s SATISFIABLE" };
    let none = if result.proved && !result.interrupted { "s UNSATISFIABLE" } else { "s UNKNOWN" };
    write_solution_tail(w, output, result.best.map(|(s, v)| (s, Some(v))), sat, none, verbose)
}

fn write_csp_result<W: Write>(w: &mut W, output: &SolutionOutput, result: CspOutcome, verbose: bool) -> Result<(), String> {
    if verbose {
        writeln!(w, "c nodes {} failures {}", result.stats.nodes, result.stats.failures).map_err(io_err)?;
        write_inprocessing_stats(w, result.stats).map_err(io_err)?;
        if result.shared_clauses > 0 || result.imported_clauses > 0 {
            writeln!(w, "c shared clauses {} imported {}", result.shared_clauses, result.imported_clauses).map_err(io_err)?;
        }
    }
    let none = if result.decided { "s UNSATISFIABLE" } else { "s UNKNOWN" };
    write_solution_tail(w, output, result.solution.map(|s| (s, None::<i32>)), "s SATISFIABLE", none, verbose)
}
