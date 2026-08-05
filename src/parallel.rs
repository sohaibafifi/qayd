//! Parallel optimization orchestration independent of any input format.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Duration;

use crate::engines::ls::cop::{solve_ls, LocalSearchSpec, LsConfig};
use crate::ids::VarId;
use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::lcg::lit::Lit;
use crate::lns::{fix_neighborhood, LnsState};
use crate::problem::{Objective, Problem};
use crate::search::{
    decide_sat_shared_seeded, find_one_seeded, optimize_seeded, probe_seeded, split_cube_seeded, Objective as SearchObjective, SolveStats,
};
use crate::store::Solver;

/// Search strategy selected on the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// Full solver: propagation + branch-and-bound / LCG; proves optimality.
    #[default]
    Default,
    /// Local-search-first incumbent mode (GLS + min-conflicts); does not prove
    /// optimality.
    Ls,
}

/// Solver execution settings. Parallel search is opt-in with `workers > 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunOptions {
    /// Reproducible base seed. Worker `w` uses `seed + w`.
    pub seed: u64,
    /// Number of independent portfolio workers.
    pub workers: usize,
    /// Search strategy.
    pub mode: Mode,
    /// Split the search space into disjoint proof jobs.
    pub split: bool,
    /// Number of workers dedicated to optimistic objective probes.
    pub probes: usize,
    /// Number of workers dedicated to incumbent-driven LNS.
    pub lns: usize,
    /// Ablation opt-out: disable CDCL learning for CSP find-one/UNSAT solving,
    /// routing every worker to the chronological-DFS driver. Learning is on by
    /// default; plain DFS wins only on highly symmetric models.
    pub no_learn_csp: bool,
    /// Ablation control: use whole propagator scopes instead of tight reasons.
    pub force_scope_reasons: bool,
    /// Requested shared-clause ring capacity. Rounded down to a power of two.
    pub shared_pool_capacity: usize,
    /// Hard memory budget in megabytes; `None` = unlimited. Enforced only when the
    /// tracking allocator is installed (the `qayd` binary).
    pub mem_limit: Option<usize>,
}

impl RunOptions {
    /// Whether the local-search engine is selected (`--ls`).
    pub(crate) fn local_search(&self) -> bool {
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
            force_scope_reasons: false,
            shared_pool_capacity: 1 << 14,
            mem_limit: None,
        }
    }
}

pub(crate) struct ParallelOutcome {
    pub(crate) best: Option<(Vec<i32>, i64)>,
    pub(crate) stats: SolveStats,
    pub(crate) proved: bool,
    pub(crate) interrupted: bool,
    pub(crate) shared_clauses: usize,
    pub(crate) imported_clauses: u64,
    pub(crate) split_jobs: Option<(u64, u64)>,
    pub(crate) probe_stats: Option<(u64, u64)>,
    pub(crate) lns_stats: Option<(u64, u64)>,
}

pub(crate) fn normalize_options(has_objective: bool, var_objective: bool, options: RunOptions) -> RunOptions {
    let options = RunOptions { workers: options.workers.max(1), ..options };
    let workers = options.workers;
    if !has_objective {
        // CSP runs a clause-sharing find-one/UNSAT portfolio; none of the
        // COP-only worker roles apply.
        return RunOptions { workers, split: false, probes: 0, lns: 0, ..options };
    }
    if options.local_search() {
        return RunOptions { workers, split: false, probes: 0, lns: 0, ..options };
    }
    if workers == 1 {
        return RunOptions { workers, split: false, probes: 0, lns: 0, ..options };
    }
    let fast_incumbents = usize::from(options.probes.saturating_add(options.lns) < workers.saturating_sub(1));
    let reserved_probe = usize::from(workers > 3 && var_objective);
    let probes = if var_objective { options.probes.max(reserved_probe).min(workers.saturating_sub(fast_incumbents + 1)) } else { 0 };
    let reserved_lns = usize::from(workers > 2);
    RunOptions {
        workers,
        split: options.split && var_objective,
        probes,
        lns: options.lns.max(reserved_lns).min(workers.saturating_sub(probes + fast_incumbents + 1)),
        ..options
    }
}

pub(crate) fn worker_roles(has_objective: bool, options: RunOptions) -> String {
    if has_objective && options.local_search() {
        return format!("ls {}", options.workers);
    }
    if !has_objective {
        return if options.workers == 1 { "sequential 1".to_string() } else { format!("portfolio {}", options.workers) };
    }
    if options.workers == 1 {
        return "sequential 1".to_string();
    }
    let incumbent = fast_incumbent_workers(has_objective, options);
    let regular = options.workers - options.probes - options.lns - incumbent;
    let mut roles = vec![format!("{} {regular}", if options.split { "split" } else { "portfolio" })];
    if options.lns > 0 {
        roles.push(format!("lns {}", options.lns));
    }
    if options.probes > 0 {
        roles.push(format!("probe {}", options.probes));
    }
    if incumbent > 0 {
        roles.push(format!("incumbent {incumbent}"));
    }
    roles.join(", ")
}

fn fast_incumbent_workers(has_objective: bool, options: RunOptions) -> usize {
    usize::from(has_objective && !options.local_search() && options.workers > options.probes + options.lns + 1)
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
            state: Mutex::new(WorkState { jobs: VecDeque::from([Vec::new()]), active: 0, split: 0, completed: 0 }),
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
    lns_improved: AtomicU64,
}

impl CopShared {
    fn stop(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.work.wake_all();
    }

    fn report_improvement(&self, tx: &mpsc::Sender<WorkerMsg>, value: i64, solution: &[i32], source: WorkerSource) -> bool {
        let mut current = self.best.load(Ordering::Relaxed);
        while if self.minimizing { value < current } else { value > current } {
            match self.best.compare_exchange_weak(current, value, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => {
                    let mut slot = self.solution.lock().unwrap();
                    if self.best.load(Ordering::Acquire) == value {
                        *slot = Some((solution.to_vec(), value));
                    }
                    let _ = tx.send(WorkerMsg::Improved { value, source });
                    return true;
                }
                Err(actual) => current = actual,
            }
        }
        false
    }

    fn incumbent(&self) -> Option<Vec<i32>> {
        self.solution.lock().unwrap().as_ref().map(|(solution, _)| solution.clone())
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
            self.probe_lower.fetch_max(target as i64 + 1, Ordering::Relaxed);
        } else {
            self.probe_upper.fetch_min(target as i64 - 1, Ordering::Relaxed);
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
    total.vivified_clauses += part.vivified_clauses;
    total.vivified_lits += part.vivified_lits;
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
            Some((solution, value)) => {
                shared.report_improvement(tx, value as i64, &solution, WorkerSource { kind: "probe", worker });
            }
            None => shared.record_probe_unsat(target),
        }
    }
    shared.finish_if_probed();
    let _ = tx.send(WorkerMsg::Done(stats));
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
) -> (SolveStats, bool, bool) {
    let minimizing = objective.minimizing();
    let sharing = objective.var().and_then(|_| clause_worker.map(|worker| ClauseSharing::new(Arc::clone(&shared.clauses), worker)));
    let mut improved = false;
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
        Vec::new(),
        Vec::new(),
        |value, solution| improved |= shared.report_improvement(tx, value, solution, source),
    );
    (stats, complete, improved)
}

fn run_lns_worker(model: Problem, worker: usize, seed: u64, shared: &CopShared, tx: &mpsc::Sender<WorkerMsg>) {
    let vars = &model.search;
    let objective = model.objective.clone().expect("COP lost its objective");
    let objective_var = model.var_objective().map(|(_, obj)| obj);
    let candidates = vars
        .iter()
        .enumerate()
        .filter_map(|(i, &var)| (Some(var) != objective_var && !model.solver.store.is_fixed(var)).then_some(i))
        .collect::<Vec<_>>();
    let mut stats = SolveStats::default();
    let mut attempt = 0u64;
    let mut lns = LnsState::new(seed);
    while !candidates.is_empty() && !shared.cancel.load(Ordering::Relaxed) {
        let Some(solution) = shared.incumbent() else {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        };
        let attempt_seed = seed.wrapping_add(attempt);
        attempt += 1;
        let mut solver = model.solver.clone();
        if !fix_neighborhood(&mut solver, vars, &solution, &candidates, attempt_seed, lns.relax_percent()) {
            continue;
        }
        shared.lns_attempts.fetch_add(1, Ordering::Relaxed);
        let (part, complete, improved) = optimize_worker(
            &mut solver,
            vars,
            objective.clone(),
            attempt_seed,
            shared,
            tx,
            WorkerSource { kind: "lns", worker },
            None,
            Some(lns.budget()),
        );
        if improved {
            shared.lns_improved.fetch_add(1, Ordering::Relaxed);
        }
        lns.record(improved, complete);
        merge_stats(&mut stats, part);
    }
    let _ = tx.send(WorkerMsg::Done(stats));
}

fn validated_incumbent(model: &Problem, solution: &[i32]) -> Option<(Vec<i32>, i64)> {
    if solution.len() != model.search.len() {
        return None;
    }
    let mut solver = model.solver.clone();
    solver.enqueue_all();
    for (&var, &value) in model.search.iter().zip(solution) {
        solver.store.fix(var, value).ok()?;
    }
    solver.propagate().ok()?;
    let value = objective_value(&solver, model.objective.as_ref()?)?;
    let solution = model.search.iter().map(|&var| solver.store.value(var)).collect();
    Some((solution, value))
}

fn objective_value(solver: &Solver, objective: &Objective) -> Option<i64> {
    match objective {
        Objective::Var(_, var) => fixed_value(solver, *var),
        Objective::Linear(_, coeffs, vars) => {
            coeffs.iter().zip(vars).try_fold(0i64, |sum, (&coeff, &var)| Some(sum + coeff * fixed_value(solver, var)?))
        }
        Objective::Expr(_, expr) => {
            let mut vars = Vec::new();
            expr.collect_vars(&mut vars);
            for var in vars {
                fixed_value(solver, var)?;
            }
            expr.eval(&|var| solver.store.value(var) as i64)
        }
    }
}

fn fixed_value(solver: &Solver, var: VarId) -> Option<i64> {
    solver.store.is_fixed(var).then(|| solver.store.value(var) as i64)
}

fn run_incumbent_worker(
    model: Problem,
    local: LocalSearchSpec,
    worker: usize,
    seed: u64,
    shared: &CopShared,
    tx: &mpsc::Sender<WorkerMsg>,
) {
    let validator = model.clone();
    solve_ls(model, local, &shared.cancel, seed, LsConfig::default(), |_, solution, _| {
        if let Some((solution, value)) = validated_incumbent(&validator, solution) {
            shared.report_improvement(tx, value, &solution, WorkerSource { kind: "incumbent", worker });
        }
    });
    let _ = tx.send(WorkerMsg::Done(SolveStats::default()));
}

/// Outcome of a CSP portfolio: a solution, a UNSAT proof, or (if interrupted) neither.
pub(crate) struct CspOutcome {
    pub(crate) solution: Option<Vec<i32>>,
    pub(crate) stats: SolveStats,
    /// `true` once some worker found a solution or exhausted the tree (SAT/UNSAT
    /// decided); `false` only when every worker was interrupted first.
    pub(crate) decided: bool,
    pub(crate) shared_clauses: usize,
    pub(crate) imported_clauses: u64,
}

/// Shared state for the CSP find-one/UNSAT portfolio.
struct CspShared {
    cancel: AtomicBool,
    /// First solution found by any worker.
    solution: Mutex<Option<Vec<i32>>>,
    /// Set once SAT (a solution) or UNSAT (an exhausted worker) is determined.
    decided: AtomicBool,
    clauses: Arc<SharedClausePool>,
}

/// Race `workers` diversified find-one workers on the same CSP. Worker 0 runs
/// plain non-learning DFS, which wins on highly symmetric models (e.g. graph
/// colouring) where clause learning thrashes; the rest run CDCL, differing by
/// seed (value-endpoint choice) and restart cadence (odd workers restart faster)
/// and cross-pollinating learned clauses (sound model consequences). The first
/// worker to find a solution, or to exhaust the tree (proving UNSAT), decides the
/// answer and cancels the rest. UNSAT and SAT are mutually exclusive across
/// workers solving the identical model, so the first decision is authoritative.
///
/// Workers clone their private solver from the immutable model. Because a
/// propagator is `Send` but not `Sync`, the source cannot be shared by reference,
/// so the clones are made serially up front; the loop polls the stop flag so a
/// slow clone of a large model cannot overrun the deadline before search begins.
pub(crate) fn solve_csp(mut problem: Problem, stop: &AtomicBool, options: RunOptions) -> CspOutcome {
    problem.solver.set_force_scope_reasons(options.force_scope_reasons);
    let shared = Arc::new(CspShared {
        cancel: AtomicBool::new(false),
        solution: Mutex::new(None),
        decided: AtomicBool::new(false),
        clauses: Arc::new(SharedClausePool::with_capacity(options.shared_pool_capacity)),
    });
    let mut models = Vec::with_capacity(options.workers);
    models.push(problem);
    for _ in 1..options.workers {
        if stop.load(Ordering::Relaxed) {
            break; // out of time before search even starts; run with what we have
        }
        models.push(models[0].clone());
    }
    let (tx, rx) = mpsc::channel();
    let mut stats = SolveStats::default();

    std::thread::scope(|scope| {
        let monitor = Arc::clone(&shared);
        scope.spawn(move || {
            while !monitor.cancel.load(Ordering::Relaxed) {
                if stop.load(Ordering::Relaxed) {
                    monitor.cancel.store(true, Ordering::Relaxed);
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
                let mut solver = model.solver;
                let vars = &model.search;
                let (solution, worker_stats, complete) = if worker == 0 || options.no_learn_csp {
                    // Strategy diversity: a non-learning DFS worker. `--no-learn-csp`
                    // turns the whole portfolio into this ablation.
                    find_one_seeded(&mut solver, vars, &shared.cancel, seed)
                } else {
                    let sharing = ClauseSharing::new(Arc::clone(&shared.clauses), worker);
                    decide_sat_shared_seeded(&mut solver, vars, &shared.cancel, seed, Some(sharing), worker % 2 == 1)
                };
                if solution.is_some() {
                    *shared.solution.lock().unwrap() = solution;
                    shared.decided.store(true, Ordering::Relaxed);
                    shared.cancel.store(true, Ordering::Relaxed);
                } else if complete {
                    // Exhausted, not cancelled: a sound UNSAT proof.
                    shared.decided.store(true, Ordering::Relaxed);
                    shared.cancel.store(true, Ordering::Relaxed);
                }
                let _ = tx.send(worker_stats);
            });
        }
        drop(tx);
        for worker_stats in rx {
            merge_stats(&mut stats, worker_stats);
        }
        shared.cancel.store(true, Ordering::Relaxed);
    });

    let solution = shared.solution.lock().unwrap().clone();
    CspOutcome {
        solution,
        stats,
        decided: shared.decided.load(Ordering::Relaxed),
        shared_clauses: shared.clauses.len(),
        imported_clauses: shared.clauses.imported(),
    }
}

pub(crate) fn solve_cop<W: Write>(
    mut problem: Problem,
    local: LocalSearchSpec,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
    options: RunOptions,
) -> Result<ParallelOutcome, String> {
    problem.solver.set_force_scope_reasons(options.force_scope_reasons);
    let minimizing = problem.objective_dir().expect("COP lost its objective");
    let var_objective = problem.var_objective();
    let (probe_lower, probe_upper) = var_objective
        .map(|(_, obj)| (problem.solver.store.min(obj) as i64, problem.solver.store.max(obj) as i64))
        .unwrap_or((i64::MIN, i64::MAX));
    let shared = Arc::new(CopShared {
        cancel: AtomicBool::new(false),
        proved: AtomicBool::new(false),
        best: AtomicI64::new(if minimizing { i64::MAX } else { i64::MIN }),
        solution: Mutex::new(None),
        clauses: Arc::new(SharedClausePool::with_capacity(options.shared_pool_capacity)),
        work: WorkQueue::new(),
        minimizing,
        probe_lower: AtomicI64::new(probe_lower),
        probe_upper: AtomicI64::new(probe_upper),
        probe_attempts: AtomicU64::new(0),
        probe_unsat: AtomicU64::new(0),
        lns_attempts: AtomicU64::new(0),
        lns_improved: AtomicU64::new(0),
    });
    let (tx, rx) = mpsc::channel();
    let mut stats = SolveStats::default();
    let mut printed = None;
    let mut error = None;
    let mut models = Vec::with_capacity(options.workers);
    models.push(problem);
    for _ in 1..options.workers {
        models.push(models[0].clone());
    }
    let incumbent_workers = fast_incumbent_workers(true, options);
    let regular_workers = options.workers - options.probes - options.lns - incumbent_workers;
    let lns_end = regular_workers + options.lns;
    let probe_end = lns_end + options.probes;

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
            let local = local.clone();
            scope.spawn(move || {
                let seed = options.seed.wrapping_add(worker as u64);
                if worker >= probe_end {
                    run_incumbent_worker(model, local, worker, seed, &shared, &tx);
                    return;
                }
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
                    let (worker_stats, complete, _) = optimize_worker(
                        &mut solver,
                        vars,
                        model.objective.unwrap(),
                        seed,
                        &shared,
                        &tx,
                        WorkerSource { kind: "portfolio", worker },
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
                        let Some(lit) = split_cube_seeded(&mut probe, vars, &cube, &shared.cancel, seed, Some(shared.clauses.lazy_atoms()))
                        else {
                            break;
                        };
                        let mut sibling = cube.clone();
                        sibling.push(lit.negate());
                        if !shared.work.try_push_split(sibling, split_target, split_limit) {
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
                            Vec::new(),
                            Vec::new(),
                            |value, solution| {
                                shared.report_improvement(&tx, value, solution, WorkerSource { kind: "split", worker });
                            },
                        );
                        (job_stats, complete)
                    };
                    merge_stats(&mut worker_stats, job_stats);
                    if shared.work.finish(complete) {
                        shared.proved.store(true, Ordering::Relaxed);
                        shared.stop();
                    }
                    if !complete || shared.proved.load(Ordering::Relaxed) {
                        break;
                    }
                }
                let _ = tx.send(WorkerMsg::Done(worker_stats));
            });
        }
        drop(tx);
        for msg in rx {
            match msg {
                WorkerMsg::Improved { value, source } if printed.is_none_or(|old| if minimizing { value < old } else { value > old }) => {
                    printed = Some(value);
                    if verbose && error.is_none() {
                        if let Err(e) = writeln!(w, "o {value}")
                            .and_then(|_| writeln!(w, "c incumbent {value} source {} worker {}", source.kind, source.worker))
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
    let split_jobs = options.split.then(|| shared.work.stats());
    let probe_stats =
        (options.probes > 0).then(|| (shared.probe_attempts.load(Ordering::Relaxed), shared.probe_unsat.load(Ordering::Relaxed)));
    let lns_stats = (options.lns > 0).then(|| (shared.lns_attempts.load(Ordering::Relaxed), shared.lns_improved.load(Ordering::Relaxed)));
    let best = shared.solution.lock().unwrap().clone();
    Ok(ParallelOutcome {
        best,
        stats,
        proved: shared.proved.load(Ordering::Relaxed),
        interrupted: stop.load(Ordering::Relaxed),
        shared_clauses: shared.clauses.len(),
        imported_clauses: shared.clauses.imported(),
        split_jobs,
        probe_stats,
        lns_stats,
    })
}
