//! XCSP3-core front-end: parse via `xcsp3-rust-parser`, build through
//! [`callback::Model`], solve, and report competition-style output.

mod callback;

use std::collections::BTreeSet;
use std::fmt::Display;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use xcsp3_rust_parser::xcsp_runner::XcspRunner;

use crate::model::ModelPackage;
use crate::orchestrator::{
    solve_model_with_external_stop, CpControls, EventCallback, EventControl, LinearControls, SolveError, SolveEvent, SolveLimits,
    SolveMode, SolveRequest, SolveResult, SolveStatus,
};

/// Search strategy selected by the XCSP command line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Complete CP search.
    #[default]
    Default,
    /// Incumbent-only local search.
    Ls,
}

/// Public XCSP runtime options retained for CLI and Rust API compatibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunOptions {
    pub seed: u64,
    pub workers: usize,
    pub mode: Mode,
    pub split: bool,
    pub probes: usize,
    pub lns: usize,
    pub no_learn_csp: bool,
    /// Prefer declared semantic variables before lowering auxiliaries.
    /// This strict experimental policy is opt-in because its benefit depends
    /// on the model structure.
    pub semantic_branching: bool,
    pub force_scope_reasons: bool,
    pub shared_pool_capacity: usize,
    pub time_limit: Option<Duration>,
    pub mem_limit: Option<usize>,
    pub linear: LinearControls,
}

impl RunOptions {
    fn local_search(self) -> bool {
        self.mode == Mode::Ls
    }
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            workers: 1,
            mode: Mode::Default,
            split: false,
            probes: 0,
            lns: 0,
            no_learn_csp: false,
            semantic_branching: false,
            force_scope_reasons: false,
            shared_pool_capacity: 1 << 14,
            time_limit: None,
            mem_limit: None,
            linear: LinearControls::default(),
        }
    }
}

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
    Value(usize),
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
                SolutionEntry::Value(i) => solution[i].to_string(),
            }),
        )?;
        writeln!(w, "v </values>")?;
        writeln!(w, "v </instantiation>")
    }
}

struct ParsedXcsp {
    package: ModelPackage,
    search: Vec<usize>,
    output: SolutionOutput,
    stats: ModelStats,
    has_objective: bool,
}

#[derive(Clone, Copy)]
struct ModelStats {
    variables: usize,
    sparse_domains: usize,
    bounds_domains: usize,
    constraints: usize,
}

fn parse_problem(path: &Path) -> Result<ParsedXcsp, String> {
    let mut model = callback::Model::new();
    let path_str = path.to_str().ok_or("bad temp path")?;
    let parsed = XcspRunner::run(path_str, &mut model);
    parsed.map_err(|e| e.to_string())?;
    if let Some(e) = model.error.take() {
        return Err(e);
    }
    let relevant = model.relevant_variables();
    let mut seen = BTreeSet::new();
    let search = model
        .declared
        .iter()
        .filter_map(|(_, variable)| (relevant[variable.0] && seen.insert(variable.0)).then_some(variable.0))
        .collect::<Vec<_>>();
    let (names, entries) = model
        .declared
        .iter()
        .map(|(name, var)| {
            let entry = if relevant[var.0] { SolutionEntry::Value(var.0) } else { SolutionEntry::Star };
            (name.clone(), entry)
        })
        .unzip();
    let output = SolutionOutput { names, entries };
    let stats = ModelStats {
        variables: model.num_variables(),
        sparse_domains: model.num_sparse_domains(),
        bounds_domains: model.num_bounds_domains(),
        constraints: model.num_constraints(),
    };
    let has_objective = model.has_objective();
    let package = model.into_package();
    Ok(ParsedXcsp { package, search, output, stats, has_objective })
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
    if options.local_search() && options.semantic_branching {
        return Err("semantic branching requires exact mode".to_string());
    }
    let path = stage_xml(xml)?;
    let ParsedXcsp { package, search, output, stats, has_objective } = parse_problem(&path.0)?;

    if verbose {
        let kind = if has_objective { "COP" } else { "CSP" };
        writeln!(w, "c qayd XCSP3").map_err(io_err)?;
        writeln!(w, "c type {kind}").map_err(io_err)?;
        if options.local_search() {
            writeln!(w, "c mode ls").map_err(io_err)?;
            writeln!(w, "c effort incumbent").map_err(io_err)?;
        }
        writeln!(w, "c variables {}", stats.variables).map_err(io_err)?;
        writeln!(w, "c sparse domains {}", stats.sparse_domains).map_err(io_err)?;
        writeln!(w, "c bounds domains {}", stats.bounds_domains).map_err(io_err)?;
        writeln!(w, "c semantic variables {}", search.len()).map_err(io_err)?;
        writeln!(w, "c propagators {}", stats.constraints).map_err(io_err)?;
        writeln!(w, "c seed {}", options.seed).map_err(io_err)?;
        writeln!(w, "c workers {}", options.workers).map_err(io_err)?;
        if !has_objective {
            writeln!(w, "c csp learning {}", !options.no_learn_csp).map_err(io_err)?;
        }
        writeln!(w, "c semantic branching {}", options.semantic_branching).map_err(io_err)?;
        writeln!(w, "c split {}", options.split).map_err(io_err)?;
    }

    // The optional semantic scope never changes the output projection or the
    // requirement that the engine complete every lowering auxiliary.
    let request = solve_request(options, search);
    let result = {
        let mut events = EventCallback(|event| {
            if let SolveEvent::Progress { engine, objectives, .. } = event {
                if let Some(value) = objectives.first().filter(|_| verbose) {
                    writeln!(w, "o {value}")
                        .and_then(|_| writeln!(w, "c incumbent {value} source {}", engine.name()))
                        .and_then(|_| w.flush())
                        .map_err(|error| SolveError::Engine(error.to_string()))?;
                }
            }
            Ok(EventControl::Continue)
        });
        solve_model_with_external_stop(&package, &request, stop, &mut events).map_err(|error| error.to_string())?
    };
    let elapsed = result.elapsed();
    write_result(w, &output, &result, verbose)?;

    if verbose {
        if stop.load(Ordering::Relaxed) {
            writeln!(w, "c interrupted").map_err(io_err)?;
        }
        writeln!(w, "c time {:.3}s", elapsed.as_secs_f64()).map_err(io_err)?;
        let peak = crate::mem::peak();
        if peak > 0 {
            writeln!(w, "c peak memory {} MB", peak / (1024 * 1024)).map_err(io_err)?;
        }
    }
    w.flush().map_err(io_err)?;
    Ok(())
}

fn solve_request(options: RunOptions, primary_branch_scope: Vec<usize>) -> SolveRequest {
    let memory_bytes = options.mem_limit.map(|megabytes| u64::try_from(megabytes).unwrap_or(u64::MAX).saturating_mul(1024 * 1024));
    SolveRequest {
        mode: if options.local_search() { SolveMode::LocalSearch } else { SolveMode::Exact },
        seed: options.seed,
        threads: options.workers,
        limits: SolveLimits { time: options.time_limit, memory_bytes, ..SolveLimits::default() },
        linear: options.linear,
        cp: CpControls {
            split: options.split,
            probes: options.probes,
            lns: options.lns,
            no_learn_csp: options.no_learn_csp,
            force_scope_reasons: options.force_scope_reasons,
            shared_pool_capacity: options.shared_pool_capacity,
        },
        primary_branch_scope: (!options.local_search() && options.semantic_branching).then_some(primary_branch_scope),
        ..SolveRequest::default()
    }
}

fn write_result<W: Write>(w: &mut W, output: &SolutionOutput, result: &SolveResult, verbose: bool) -> Result<(), String> {
    if verbose {
        write_report(w, result).map_err(io_err)?;
        if let Some(message) = result.message() {
            writeln!(w, "c {message}").map_err(io_err)?;
        }
    }

    let solution = result.primal().map(candidate_values).transpose()?;
    if !verbose {
        if let Some(value) = result.primal().and_then(|candidate| candidate.objectives().first()) {
            writeln!(w, "o {value}").map_err(io_err)?;
        }
    }
    let status = match result.status() {
        SolveStatus::Optimal => "s OPTIMUM FOUND",
        SolveStatus::Satisfiable => "s SATISFIABLE",
        SolveStatus::Unsatisfiable => "s UNSATISFIABLE",
        SolveStatus::Unknown | SolveStatus::Unsupported => "s UNKNOWN",
    };
    writeln!(w, "{status}").map_err(io_err)?;
    if let Some(solution) = solution {
        output.write(w, &solution).map_err(io_err)?;
    }
    Ok(())
}

fn candidate_values(candidate: &crate::orchestrator::CandidateSolution) -> Result<Vec<i32>, String> {
    candidate
        .assignment()
        .integers
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let value = value.ok_or_else(|| format!("verified XCSP candidate is missing integer value {index}"))?;
            i32::try_from(value).map_err(|_| format!("verified XCSP candidate value {value} is outside the i32 range"))
        })
        .collect()
}

fn write_report<W: Write>(w: &mut W, result: &SolveResult) -> std::io::Result<()> {
    let stats = result.aggregate_search_stats();
    writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures)?;
    if stats.vivified_clauses > 0 {
        writeln!(w, "c inprocessing vivified {} removed {}", stats.vivified_clauses, stats.vivified_lits)?;
    }
    if stats.lp_rows > 0 {
        writeln!(
            w,
            "c lp rows {} root_bound {} solves {} certified {} timeouts {} refactorizations {} time_ms {:.3}",
            stats.lp_rows,
            stats.lp_root_bound.map_or_else(|| "?".to_string(), |bound| bound.to_string()),
            stats.lp_solves,
            stats.lp_certified,
            stats.lp_timeouts,
            stats.lp_refactorizations,
            stats.lp_micros as f64 / 1000.0,
        )?;
    }

    let value =
        |key| result.reports().iter().flat_map(|report| &report.metadata).find_map(|(name, value)| (name == key).then_some(value.as_str()));
    if let (Some(fixed), Some(before), Some(after)) =
        (value("presolve_fixed"), value("presolve_search_before"), value("presolve_search_after"))
    {
        if fixed != "0" || before != after {
            writeln!(w, "c presolve reduced search {before} -> {after} fixed {fixed}")?;
        }
    }
    if let (Some(constraints), Some(functionals)) = (value("ls_constraints"), value("ls_functionals")) {
        writeln!(w, "c local constraints {constraints} functionals {functionals}")?;
        if let Some(unsupported) = value("ls_unsupported").filter(|unsupported| *unsupported != "0") {
            writeln!(w, "c local unsupported {unsupported}")?;
        }
        if let Some(moves) = value("ls_moves").filter(|_| stats.nodes > 0) {
            writeln!(w, "c local iterations {} moves {moves} restarts {}", stats.nodes, stats.failures)?;
        }
    }
    if let (Some(shared), Some(imported)) = (value("shared_clauses"), value("imported_clauses")) {
        if shared != "0" || imported != "0" {
            writeln!(w, "c shared clauses {shared} imported {imported}")?;
        }
    }
    if let (Some(jobs), Some(completed)) = (value("split_jobs"), value("completed_jobs")) {
        writeln!(w, "c split jobs {jobs} completed {completed}")?;
    }
    if let (Some(attempts), Some(unsatisfiable)) = (value("probe_attempts"), value("probe_unsat")) {
        writeln!(w, "c probes attempts {attempts} unsat {unsatisfiable}")?;
    }
    if let (Some(attempts), Some(improved)) = (value("lns_attempts"), value("lns_improved")) {
        writeln!(w, "c lns attempts {attempts} improved {improved}")?;
    }
    Ok(())
}
