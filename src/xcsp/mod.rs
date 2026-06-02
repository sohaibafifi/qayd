//! XCSP3-core front-end: parse via `xcsp3-rust-parser`, build through
//! [`callback::Model`], solve, and report competition-style output.

mod callback;

use std::collections::VecDeque;
use std::fmt::Display;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use xcsp3_rust_parser::xcsp_runner::XcspRunner;

use crate::ids::VarId;
use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::lcg::lit::Lit;
use crate::search::{
    optimize_seeded, probe_seeded, solve_interruptible_seeded, split_cube_seeded,
    Objective as SearchObjective, SearchControl, SolveStats,
};
use crate::store::Solver;
use callback::Objective;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
const LNS_CONFLICT_BUDGET: u64 = 10_000;
const LNS_RELAX_DIVISOR: u64 = 5;

struct StagedXml(PathBuf);

impl Drop for StagedXml {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// XCSP execution settings. Parallel search is opt-in with `workers > 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunOptions {
    /// Reproducible base seed. Worker `w` uses `seed + w`.
    pub seed: u64,
    /// Number of independent portfolio workers.
    pub workers: usize,
    /// Split the search space into disjoint proof jobs.
    pub split: bool,
    /// Number of workers dedicated to optimistic objective probes.
    pub probes: usize,
    /// Number of workers dedicated to incumbent-driven LNS.
    pub lns: usize,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            workers: 1,
            split: false,
            probes: 0,
            lns: 0,
        }
    }
}

const MAX_VALUE_LINE_LEN: usize = 4096;

fn write_tokens<W: Write>(
    w: &mut W,
    tokens: impl IntoIterator<Item = impl Display>,
) -> std::io::Result<()> {
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

fn write_improvement<W: Write>(
    w: &mut W,
    verbose: bool,
    error: &mut Option<std::io::Error>,
    value: impl Display,
) {
    if verbose && error.is_none() {
        *error = writeln!(w, "o {value}")
            .and_then(|_| writeln!(w, "c incumbent {value} source sequential"))
            .and_then(|_| w.flush())
            .err();
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
            writeln!(
                w,
                "{}",
                if complete {
                    "s OPTIMUM FOUND"
                } else {
                    "s SATISFIABLE"
                }
            )
            .map_err(to_err)?;
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

/// Immutable root model. Portfolio workers clone it instead of reparsing XML.
#[derive(Clone)]
struct Problem {
    solver: Solver,
    search: Vec<VarId>,
    output: SolutionOutput,
    objective: Option<Objective>,
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
        write_tokens(
            w,
            self.entries.iter().map(|&position| {
                position.map_or_else(|| "*".to_string(), |i| solution[i].to_string())
            }),
        )?;
        writeln!(w, "v </values>")?;
        writeln!(w, "v </instantiation>")
    }
}

impl Problem {
    fn var_objective(&self) -> Option<(bool, VarId)> {
        self.objective
            .as_ref()
            .and_then(|objective| objective.var().map(|obj| (objective.minimizing(), obj)))
    }

    fn objective_dir(&self) -> Option<bool> {
        self.objective.as_ref().map(Objective::minimizing)
    }
}

impl Objective {
    fn minimizing(&self) -> bool {
        match self {
            Self::Var(minimizing, _)
            | Self::Linear(minimizing, _, _)
            | Self::Expr(minimizing, _) => *minimizing,
        }
    }

    fn var(&self) -> Option<VarId> {
        match self {
            Self::Var(_, var) => Some(*var),
            Self::Linear(_, _, _) | Self::Expr(_, _) => None,
        }
    }

    fn search(&self) -> SearchObjective<'_> {
        match self {
            Self::Var(_, var) => SearchObjective::Var(*var),
            Self::Linear(_, coeffs, vars) => SearchObjective::Linear { coeffs, vars },
            Self::Expr(_, expr) => SearchObjective::Expr(expr),
        }
    }
}

type Cube = Vec<Lit>;

struct WorkState {
    jobs: VecDeque<Cube>,
    active: usize,
    split: u64,
    completed: u64,
}

struct WorkQueue {
    state: Mutex<WorkState>,
    ready: Condvar,
}

impl WorkQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(WorkState {
                jobs: VecDeque::from([Vec::new()]),
                active: 0,
                split: 0,
                completed: 0,
            }),
            ready: Condvar::new(),
        }
    }

    fn take(&self, cancel: &AtomicBool) -> Option<Cube> {
        let mut state = self.state.lock().unwrap();
        loop {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            if let Some(job) = state.jobs.pop_front() {
                state.active += 1;
                return Some(job);
            }
            if state.active == 0 {
                return None;
            }
            state = self.ready.wait(state).unwrap();
        }
    }

    fn needs_split(&self, target: usize, limit: u64) -> bool {
        let state = self.state.lock().unwrap();
        state.jobs.len() < target && state.split < limit
    }

    fn try_push_split(&self, sibling: Cube, target: usize, limit: u64) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.jobs.len() >= target || state.split >= limit {
            return false;
        }
        state.jobs.push_back(sibling);
        state.split += 1;
        self.ready.notify_one();
        true
    }

    fn finish(&self, completed: bool) -> bool {
        let mut state = self.state.lock().unwrap();
        state.active -= 1;
        state.completed += u64::from(completed);
        let finished = completed && state.active == 0 && state.jobs.is_empty();
        self.ready.notify_all();
        finished
    }

    fn wake_all(&self) {
        self.ready.notify_all();
    }

    fn stats(&self) -> (u64, u64) {
        let state = self.state.lock().unwrap();
        (state.split, state.completed)
    }
}

fn parse_problem(path: &Path) -> Result<Problem, String> {
    let mut model = callback::Model::new();
    let path_str = path.to_str().ok_or("bad temp path")?;
    let parsed = XcspRunner::run(path_str, &mut model);
    parsed.map_err(|e| e.to_string())?;
    if let Some(e) = model.error.take() {
        return Err(e);
    }
    let mut relevant = (0..model.solver.store.num_vars())
        .map(|i| model.solver.store.is_relevant(VarId(i as u32)))
        .collect::<Vec<_>>();
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
    let search = relevant
        .iter()
        .enumerate()
        .filter_map(|(i, &used)| used.then_some(VarId(i as u32)))
        .collect::<Vec<_>>();
    let mut positions = vec![None; model.solver.store.num_vars()];
    for (i, &var) in search.iter().enumerate() {
        positions[var.index()] = Some(i);
    }
    let (names, entries) = model
        .declared
        .iter()
        .map(|(name, var)| (name.clone(), positions[var.index()]))
        .unzip();
    let output = SolutionOutput { names, entries };
    Ok(Problem {
        solver: model.solver,
        search,
        output,
        objective: model.objective,
    })
}

struct CopShared {
    cancel: AtomicBool,
    proved: AtomicBool,
    best: AtomicI64,
    solution: Mutex<Option<(Vec<i32>, i64)>>,
    clauses: Arc<SharedClausePool>,
    work: WorkQueue,
    minimizing: bool,
    probe_lower: AtomicI64,
    probe_upper: AtomicI64,
    probe_attempts: AtomicU64,
    probe_unsat: AtomicU64,
    lns_attempts: AtomicU64,
}

impl CopShared {
    fn stop(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.work.wake_all();
    }

    fn report_improvement(
        &self,
        tx: &mpsc::Sender<WorkerMsg>,
        value: i64,
        solution: &[i32],
        source: WorkerSource,
    ) {
        let mut current = self.best.load(Ordering::Relaxed);
        while if self.minimizing {
            value < current
        } else {
            value > current
        } {
            match self.best.compare_exchange_weak(
                current,
                value,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let mut slot = self.solution.lock().unwrap();
                    if self.best.load(Ordering::Acquire) == value {
                        *slot = Some((solution.to_vec(), value));
                    }
                    let _ = tx.send(WorkerMsg::Improved { value, source });
                    return;
                }
                Err(actual) => current = actual,
            }
        }
    }

    fn incumbent(&self) -> Option<Vec<i32>> {
        self.solution
            .lock()
            .unwrap()
            .as_ref()
            .map(|(solution, _)| solution.clone())
    }

    fn next_probe(&self) -> Option<i32> {
        let best = self.best.load(Ordering::Relaxed);
        let lower = self.probe_lower.load(Ordering::Relaxed);
        let upper = self.probe_upper.load(Ordering::Relaxed);
        let (lo, hi) = if self.minimizing {
            (lower, if best == i64::MAX { upper } else { best - 1 })
        } else {
            (if best == i64::MIN { lower } else { best + 1 }, upper)
        };
        (lo <= hi).then(|| (lo + (hi - lo) / 2) as i32)
    }

    fn record_probe_unsat(&self, target: i32) {
        if self.minimizing {
            self.probe_lower
                .fetch_max(target as i64 + 1, Ordering::Relaxed);
        } else {
            self.probe_upper
                .fetch_min(target as i64 - 1, Ordering::Relaxed);
        }
        self.probe_unsat.fetch_add(1, Ordering::Relaxed);
    }

    fn finish_if_probed(&self) -> bool {
        let best = self.best.load(Ordering::Relaxed);
        let lower = self.probe_lower.load(Ordering::Relaxed);
        let upper = self.probe_upper.load(Ordering::Relaxed);
        let proved = if self.minimizing {
            if best == i64::MAX {
                lower > upper
            } else {
                lower >= best
            }
        } else if best == i64::MIN {
            upper < lower
        } else {
            upper <= best
        };
        if proved {
            self.proved.store(true, Ordering::Relaxed);
            self.stop();
        }
        proved
    }
}

#[derive(Clone, Copy)]
struct WorkerSource {
    kind: &'static str,
    worker: usize,
}

enum WorkerMsg {
    Improved { value: i64, source: WorkerSource },
    Done(SolveStats),
}

fn merge_stats(total: &mut SolveStats, part: SolveStats) {
    total.solutions += part.solutions;
    total.nodes += part.nodes;
    total.failures += part.failures;
    total.learned_lits += part.learned_lits;
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
pub fn run_to<W: Write>(
    xml: &str,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
) -> Result<(), String> {
    run_to_with_options(xml, verbose, stop, w, RunOptions::default())
}

/// Like [`run_to`], with explicit seed and portfolio-worker settings.
pub fn run_to_with_options<W: Write>(
    xml: &str,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
    options: RunOptions,
) -> Result<(), String> {
    let start = Instant::now();
    let options = RunOptions {
        workers: options.workers.max(1),
        ..options
    };
    let path = stage_xml(xml)?;
    let problem = parse_problem(&path.0)?;
    let has_objective = problem.objective.is_some();
    let var_objective = problem.var_objective().is_some();
    let symbolic_probes_disabled = has_objective && !var_objective && options.probes > 0;
    let workers = if has_objective { options.workers } else { 1 };
    let probes = if var_objective {
        options.probes.min(workers.saturating_sub(1))
    } else {
        0
    };
    let options = RunOptions {
        workers,
        split: options.split && var_objective,
        probes,
        lns: options.lns.min(workers.saturating_sub(probes + 1)),
        ..options
    };
    let to_err = |e: std::io::Error| e.to_string();

    if verbose {
        let kind = if problem.objective.is_some() {
            "COP"
        } else {
            "CSP"
        };
        writeln!(w, "c qayd XCSP3").map_err(to_err)?;
        writeln!(w, "c type {kind}").map_err(to_err)?;
        writeln!(w, "c variables {}", problem.solver.store.num_vars()).map_err(to_err)?;
        writeln!(
            w,
            "c sparse domains {}",
            problem.solver.store.num_sparse_domains()
        )
        .map_err(to_err)?;
        writeln!(
            w,
            "c bounds domains {}",
            problem.solver.store.num_bounds_domains()
        )
        .map_err(to_err)?;
        writeln!(w, "c search variables {}", problem.search.len()).map_err(to_err)?;
        writeln!(w, "c propagators {}", problem.solver.num_propagators()).map_err(to_err)?;
        writeln!(w, "c seed {}", options.seed).map_err(to_err)?;
        writeln!(w, "c workers {}", options.workers).map_err(to_err)?;
        writeln!(w, "c split {}", options.split).map_err(to_err)?;
        if symbolic_probes_disabled {
            writeln!(
                w,
                "c probes 0 (probe worker only supports a materialized objective variable)"
            )
            .map_err(to_err)?;
        } else {
            writeln!(w, "c probes {}", options.probes).map_err(to_err)?;
        }
        writeln!(w, "c lns {}", options.lns).map_err(to_err)?;
        w.flush().map_err(to_err)?;
    }

    if options.workers == 1 || !has_objective {
        solve_single(problem, verbose, stop, w, options.seed)?;
    } else {
        solve_parallel_cop(problem, verbose, stop, w, options)?;
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

fn solve_single<W: Write>(
    problem: Problem,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
    seed: u64,
) -> Result<(), String> {
    let mut solver = problem.solver;
    let vars = problem.search;
    let output = problem.output;
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
                writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures)
                    .map_err(to_err)?;
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
            let (best, stats, complete) = optimize_seeded(
                &mut solver,
                &vars,
                objective.search(),
                minimizing,
                stop,
                seed,
                None,
                None,
                &[],
                None,
                |v, _| write_improvement(w, verbose, &mut io_err, v),
            );
            if let Some(e) = io_err {
                return Err(e.to_string());
            }
            let interrupted = stop.load(Ordering::Relaxed);
            write_optimization_result(w, &output, best, stats, complete, interrupted, verbose)
        }
    }
}

fn run_probe_worker(
    model: Problem,
    worker: usize,
    obj: VarId,
    minimizing: bool,
    seed: u64,
    shared: &CopShared,
    tx: &mpsc::Sender<WorkerMsg>,
) {
    let vars = &model.search;
    let mut stats = SolveStats::default();
    let mut attempt = 0;
    while !shared.cancel.load(Ordering::Relaxed) {
        if shared.finish_if_probed() {
            break;
        }
        let Some(target) = shared.next_probe() else {
            break;
        };
        shared.probe_attempts.fetch_add(1, Ordering::Relaxed);
        let mut solver = model.solver.clone();
        let (found, part, complete) = probe_seeded(
            &mut solver,
            vars,
            obj,
            minimizing,
            target,
            &shared.cancel,
            seed.wrapping_add(attempt),
            Some(ClauseSharing::new(Arc::clone(&shared.clauses), worker)),
        );
        attempt += 1;
        merge_stats(&mut stats, part);
        if !complete {
            break;
        }
        match found {
            Some((solution, value)) => shared.report_improvement(
                tx,
                value as i64,
                &solution,
                WorkerSource {
                    kind: "probe",
                    worker,
                },
            ),
            None => shared.record_probe_unsat(target),
        }
    }
    shared.finish_if_probed();
    let _ = tx.send(WorkerMsg::Done(stats));
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn fix_lns_neighborhood(
    solver: &mut Solver,
    vars: &[VarId],
    solution: &[i32],
    candidates: &[usize],
    seed: u64,
) -> bool {
    let pivot = candidates[mix64(seed) as usize % candidates.len()];
    candidates.iter().copied().all(|i| {
        i == pivot
            || mix64(seed ^ i as u64).is_multiple_of(LNS_RELAX_DIVISOR)
            || solver.store.fix(vars[i], solution[i]).is_ok()
    })
}

#[allow(clippy::too_many_arguments)]
fn optimize_worker(
    solver: &mut Solver,
    vars: &[VarId],
    objective: Objective,
    seed: u64,
    shared: &CopShared,
    tx: &mpsc::Sender<WorkerMsg>,
    source: WorkerSource,
    clause_worker: Option<usize>,
    conflict_budget: Option<u64>,
) -> (SolveStats, bool) {
    let minimizing = objective.minimizing();
    let sharing = objective.var().and_then(|_| {
        clause_worker.map(|worker| ClauseSharing::new(Arc::clone(&shared.clauses), worker))
    });
    let (_, stats, complete) = optimize_seeded(
        solver,
        vars,
        objective.search(),
        minimizing,
        &shared.cancel,
        seed,
        Some(&shared.best),
        sharing,
        &[],
        conflict_budget,
        |value, solution| shared.report_improvement(tx, value, solution, source),
    );
    (stats, complete)
}

fn run_lns_worker(
    model: Problem,
    worker: usize,
    seed: u64,
    shared: &CopShared,
    tx: &mpsc::Sender<WorkerMsg>,
) {
    let vars = &model.search;
    let objective = model.objective.clone().expect("COP lost its objective");
    let objective_var = model.var_objective().map(|(_, obj)| obj);
    let candidates = vars
        .iter()
        .enumerate()
        .filter_map(|(i, &var)| {
            (Some(var) != objective_var && !model.solver.store.is_fixed(var)).then_some(i)
        })
        .collect::<Vec<_>>();
    let mut stats = SolveStats::default();
    let mut attempt = 0u64;
    while !candidates.is_empty() && !shared.cancel.load(Ordering::Relaxed) {
        let Some(solution) = shared.incumbent() else {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        };
        let attempt_seed = seed.wrapping_add(attempt);
        attempt += 1;
        let mut solver = model.solver.clone();
        if !fix_lns_neighborhood(&mut solver, vars, &solution, &candidates, attempt_seed) {
            continue;
        }
        shared.lns_attempts.fetch_add(1, Ordering::Relaxed);
        let (part, _) = optimize_worker(
            &mut solver,
            vars,
            objective.clone(),
            attempt_seed,
            shared,
            tx,
            WorkerSource {
                kind: "lns",
                worker,
            },
            None,
            Some(LNS_CONFLICT_BUDGET),
        );
        merge_stats(&mut stats, part);
    }
    let _ = tx.send(WorkerMsg::Done(stats));
}

fn solve_parallel_cop<W: Write>(
    problem: Problem,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
    options: RunOptions,
) -> Result<(), String> {
    let minimizing = problem.objective_dir().expect("COP lost its objective");
    let var_objective = problem.var_objective();
    let (probe_lower, probe_upper) = var_objective
        .map(|(_, obj)| {
            (
                problem.solver.store.min(obj) as i64,
                problem.solver.store.max(obj) as i64,
            )
        })
        .unwrap_or((i64::MIN, i64::MAX));
    let shared = Arc::new(CopShared {
        cancel: AtomicBool::new(false),
        proved: AtomicBool::new(false),
        best: AtomicI64::new(if minimizing { i64::MAX } else { i64::MIN }),
        solution: Mutex::new(None),
        clauses: Arc::new(SharedClausePool::default()),
        work: WorkQueue::new(),
        minimizing,
        probe_lower: AtomicI64::new(probe_lower),
        probe_upper: AtomicI64::new(probe_upper),
        probe_attempts: AtomicU64::new(0),
        probe_unsat: AtomicU64::new(0),
        lns_attempts: AtomicU64::new(0),
    });
    let (tx, rx) = mpsc::channel();
    let mut stats = SolveStats::default();
    let mut printed = None;
    let mut error = None;
    let output = problem.output.clone();
    let mut models = Vec::with_capacity(options.workers);
    models.push(problem);
    for _ in 1..options.workers {
        models.push(models[0].clone());
    }
    let regular_workers = options.workers - options.probes - options.lns;
    let lns_end = regular_workers + options.lns;

    std::thread::scope(|scope| {
        let monitor = Arc::clone(&shared);
        scope.spawn(move || {
            while !monitor.cancel.load(Ordering::Relaxed) {
                if stop.load(Ordering::Relaxed) {
                    monitor.stop();
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        for (worker, model) in models.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let tx = tx.clone();
            scope.spawn(move || {
                let seed = options.seed.wrapping_add(worker as u64);
                if worker >= lns_end {
                    let (_, obj) = var_objective.expect("symbolic objective cannot use probes");
                    run_probe_worker(model, worker, obj, minimizing, seed, &shared, &tx);
                    return;
                }
                if worker >= regular_workers {
                    run_lns_worker(model, worker, seed, &shared, &tx);
                    return;
                }
                let vars = &model.search;
                if !options.split {
                    let mut solver = model.solver;
                    let (worker_stats, complete) = optimize_worker(
                        &mut solver,
                        vars,
                        model.objective.unwrap(),
                        seed,
                        &shared,
                        &tx,
                        WorkerSource {
                            kind: "portfolio",
                            worker,
                        },
                        Some(worker),
                        None,
                    );
                    if complete {
                        shared.proved.store(true, Ordering::Relaxed);
                        shared.stop();
                    }
                    let _ = tx.send(WorkerMsg::Done(worker_stats));
                    return;
                }
                let split_target = 2 * regular_workers;
                let split_limit = 4 * regular_workers as u64;
                let (_, obj) = var_objective.expect("symbolic objective cannot split");
                let mut worker_stats = SolveStats::default();
                while let Some(mut cube) = shared.work.take(&shared.cancel) {
                    while shared.work.needs_split(split_target, split_limit) {
                        let mut probe = model.solver.clone();
                        let Some(lit) = split_cube_seeded(
                            &mut probe,
                            vars,
                            &cube,
                            &shared.cancel,
                            seed,
                            Some(shared.clauses.lazy_atoms()),
                        ) else {
                            break;
                        };
                        let mut sibling = cube.clone();
                        sibling.push(lit.negate());
                        if !shared
                            .work
                            .try_push_split(sibling, split_target, split_limit)
                        {
                            break;
                        }
                        cube.push(lit);
                    }
                    let (job_stats, complete) = if shared.cancel.load(Ordering::Relaxed) {
                        (SolveStats::default(), false)
                    } else {
                        let mut solver = model.solver.clone();
                        let (_, job_stats, complete) = optimize_seeded(
                            &mut solver,
                            vars,
                            SearchObjective::Var(obj),
                            minimizing,
                            &shared.cancel,
                            seed,
                            Some(&shared.best),
                            Some(ClauseSharing::new(Arc::clone(&shared.clauses), worker)),
                            &cube,
                            None,
                            |value, solution| {
                                shared.report_improvement(
                                    &tx,
                                    value,
                                    solution,
                                    WorkerSource {
                                        kind: "split",
                                        worker,
                                    },
                                );
                            },
                        );
                        (job_stats, complete)
                    };
                    merge_stats(&mut worker_stats, job_stats);
                    if shared.work.finish(complete) {
                        shared.proved.store(true, Ordering::Relaxed);
                        shared.stop();
                    }
                    if !complete {
                        break;
                    }
                    if shared.proved.load(Ordering::Relaxed) {
                        break;
                    }
                }
                let _ = tx.send(WorkerMsg::Done(worker_stats));
            });
        }
        drop(tx);
        for msg in rx {
            match msg {
                WorkerMsg::Improved { value, source }
                    if printed.is_none_or(
                        |old| {
                            if minimizing {
                                value < old
                            } else {
                                value > old
                            }
                        },
                    ) =>
                {
                    printed = Some(value);
                    if verbose && error.is_none() {
                        if let Err(e) = writeln!(w, "o {value}")
                            .and_then(|_| {
                                writeln!(
                                    w,
                                    "c incumbent {value} source {} worker {}",
                                    source.kind, source.worker
                                )
                            })
                            .and_then(|_| w.flush())
                        {
                            error = Some(e.to_string());
                            shared.stop();
                        }
                    }
                }
                WorkerMsg::Improved { .. } => {}
                WorkerMsg::Done(worker_stats) => {
                    merge_stats(&mut stats, worker_stats);
                }
            }
        }
        shared.stop();
    });

    if let Some(e) = error {
        return Err(e);
    }
    let to_err = |e: std::io::Error| e.to_string();
    if verbose {
        writeln!(w, "c nodes {} failures {}", stats.nodes, stats.failures).map_err(to_err)?;
        writeln!(
            w,
            "c shared clauses {} imported {}",
            shared.clauses.len(),
            shared.clauses.imported()
        )
        .map_err(to_err)?;
        if options.split {
            let (split, completed) = shared.work.stats();
            writeln!(w, "c split jobs {split} completed {completed}").map_err(to_err)?;
        }
        if options.probes > 0 {
            writeln!(
                w,
                "c probes attempts {} unsat {}",
                shared.probe_attempts.load(Ordering::Relaxed),
                shared.probe_unsat.load(Ordering::Relaxed)
            )
            .map_err(to_err)?;
        }
        if options.lns > 0 {
            writeln!(
                w,
                "c lns attempts {}",
                shared.lns_attempts.load(Ordering::Relaxed)
            )
            .map_err(to_err)?;
        }
    }
    let interrupted = stop.load(Ordering::Relaxed);
    let proved = shared.proved.load(Ordering::Relaxed);
    match shared.solution.lock().unwrap().clone() {
        Some((sol, value)) => {
            if !verbose {
                writeln!(w, "o {value}").map_err(to_err)?;
            }
            let status = if proved {
                "s OPTIMUM FOUND"
            } else {
                "s SATISFIABLE"
            };
            writeln!(w, "{status}").map_err(to_err)?;
            output.write(w, &sol).map_err(to_err)?;
        }
        None if interrupted => writeln!(w, "s UNKNOWN").map_err(to_err)?,
        None if proved => writeln!(w, "s UNSATISFIABLE").map_err(to_err)?,
        None => writeln!(w, "s UNKNOWN").map_err(to_err)?,
    }
    Ok(())
}
