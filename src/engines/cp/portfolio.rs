//! Parallel CP portfolio search.
//!
//! This module owns CP-specific cooperation: shared clauses, split cubes,
//! objective probes, incumbent-driven LNS, and exact proof completion. Generic
//! worker lifecycle and bounded progress delivery live in the orchestrator.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::ids::VarId;
use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::lcg::lit::Lit;
use crate::lns::{fix_neighborhood, LnsState};
use crate::orchestrator::{execute_workers, execute_workers_silent, merge_search_stats, EventControl, WorkerContext};
use crate::problem::{Objective, Problem};
use crate::search::{
    decide_sat_assuming_seeded_with_scope, decide_sat_shared_seeded, find_one_seeded_with_scope, optimize_seeded_with_scope,
    probe_seeded_with_scope, split_cube_seeded_with_scope, Objective as SearchObjective, SolveStats,
};
use crate::store::Solver;

/// Solver execution settings. Parallel search is opt-in with `workers > 1`.
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
    /// Disable CDCL learning for CSP find-one/UNSAT solving.
    pub no_learn_csp: bool,
    /// Use whole propagator scopes instead of tight reasons.
    pub force_scope_reasons: bool,
    /// Requested shared-clause ring capacity. Rounded down to a power of two.
    pub shared_pool_capacity: usize,
    /// Total conflict ceiling shared by all complete workers.
    pub conflict_limit: Option<u64>,
}

#[derive(Clone, Default)]
pub(crate) struct SearchGuidance {
    pub initial_phase: Vec<Option<i32>>,
    pub branch_order: Vec<VarId>,
    pub primary_branch_scope: Option<Vec<VarId>>,
}

impl SearchGuidance {
    fn is_empty(&self) -> bool {
        self.initial_phase.iter().all(Option::is_none) && self.branch_order.is_empty() && self.primary_branch_scope.is_none()
    }

    fn is_chronological_dfs_compatible(&self) -> bool {
        self.initial_phase.iter().all(Option::is_none) && self.branch_order.is_empty()
    }
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            workers: 1,
            split: false,
            probes: 0,
            lns: 0,
            no_learn_csp: false,
            force_scope_reasons: false,
            shared_pool_capacity: 1 << 14,
            conflict_limit: None,
        }
    }
}

pub(crate) struct ParallelOutcome {
    pub(crate) best: Option<(Vec<i32>, i64)>,
    pub(crate) stats: SolveStats,
    pub(crate) proved: bool,
    pub(crate) shared_clauses: usize,
    pub(crate) imported_clauses: u64,
    pub(crate) split_jobs: Option<(u64, u64)>,
    pub(crate) probe_stats: Option<(u64, u64)>,
    pub(crate) lns_stats: Option<(u64, u64)>,
}

struct CopWorkerResult {
    stats: SolveStats,
}

pub(crate) fn normalize_options(has_objective: bool, var_objective: bool, options: RunOptions) -> RunOptions {
    let options = RunOptions { workers: options.workers.max(1), ..options };
    let workers = options.workers;
    if !has_objective {
        return RunOptions { workers, split: false, probes: 0, lns: 0, ..options };
    }
    if workers == 1 {
        return RunOptions { workers, split: false, probes: 0, lns: 0, ..options };
    }
    if options.conflict_limit.is_some() {
        return RunOptions { workers, split: false, probes: 0, lns: 0, ..options };
    }
    let reserved_probe = usize::from(workers > 3 && var_objective);
    let probes = if !var_objective {
        0
    } else if options.probes > 0 {
        options.probes
    } else if options.lns == 0 {
        reserved_probe
    } else {
        0
    }
    .min(workers.saturating_sub(1));
    let reserved_lns = usize::from(workers > 2);
    let available_lns = workers.saturating_sub(probes + 1);
    let lns = if options.lns > 0 { options.lns } else { reserved_lns }.min(available_lns);
    debug_assert!(probes.saturating_add(lns) < workers);
    RunOptions { workers, split: options.split && var_objective, probes, lns, ..options }
}

fn worker_conflict_quota(total: Option<u64>, worker: usize, workers: usize) -> Option<u64> {
    total.map(|total| {
        let workers = u64::try_from(workers).unwrap_or(u64::MAX).max(1);
        let worker = u64::try_from(worker).unwrap_or(u64::MAX);
        total / workers + u64::from(worker < total % workers)
    })
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
            if cancel.load(Ordering::Acquire) {
                return None;
            }
            if let Some(job) = state.jobs.pop_front() {
                state.active += 1;
                return Some(job);
            }
            if state.active == 0 {
                return None;
            }
            state = self.ready.wait_timeout(state, Duration::from_millis(5)).unwrap().0;
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

#[derive(Clone, Copy)]
struct WorkerSource {
    kind: &'static str,
    worker: usize,
}

#[derive(Clone)]
struct Improvement {
    value: i64,
    source: WorkerSource,
    solution: Option<Vec<i32>>,
}

struct CopShared {
    cancel: Arc<AtomicBool>,
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
    publish_incumbents: bool,
}

impl CopShared {
    fn stop(&self) {
        self.cancel.store(true, Ordering::Release);
        self.work.wake_all();
    }

    fn report_improvement(
        &self,
        context: &WorkerContext<Improvement, CopWorkerResult>,
        value: i64,
        solution: &[i32],
        source: WorkerSource,
    ) -> bool {
        let mut current = self.best.load(Ordering::Relaxed);
        while if self.minimizing { value < current } else { value > current } {
            match self.best.compare_exchange_weak(current, value, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => {
                    let mut slot = self.solution.lock().unwrap();
                    if self.best.load(Ordering::Acquire) == value {
                        *slot = Some((solution.to_vec(), value));
                    }
                    context.publish_latest(Improvement { value, source, solution: self.publish_incumbents.then(|| solution.to_vec()) });
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
        let best = self.best.load(Ordering::Acquire);
        let lower = self.probe_lower.load(Ordering::Acquire);
        let upper = self.probe_upper.load(Ordering::Acquire);
        let (lo, hi) = if self.minimizing {
            (lower, if best == i64::MAX { upper } else { best - 1 })
        } else {
            (if best == i64::MIN { lower } else { best + 1 }, upper)
        };
        (lo <= hi).then(|| (lo + (hi - lo) / 2) as i32)
    }

    fn record_probe_unsat(&self, target: i32) {
        if self.minimizing {
            self.probe_lower.fetch_max(target as i64 + 1, Ordering::AcqRel);
        } else {
            self.probe_upper.fetch_min(target as i64 - 1, Ordering::AcqRel);
        }
        self.probe_unsat.fetch_add(1, Ordering::Relaxed);
    }

    fn finish_if_probed(&self) -> bool {
        let best = self.best.load(Ordering::Acquire);
        let lower = self.probe_lower.load(Ordering::Acquire);
        let upper = self.probe_upper.load(Ordering::Acquire);
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
            self.proved.store(true, Ordering::Release);
            self.stop();
        }
        proved
    }
}

fn run_probe_worker(
    model: Problem,
    obj: VarId,
    minimizing: bool,
    shared: &CopShared,
    context: &WorkerContext<Improvement, CopWorkerResult>,
    guidance: &SearchGuidance,
) -> CopWorkerResult {
    let vars = &model.search;
    let mut stats = SolveStats::default();
    let mut attempt = 0;
    while !context.is_cancelled() {
        if shared.finish_if_probed() {
            break;
        }
        let Some(target) = shared.next_probe() else {
            break;
        };
        shared.probe_attempts.fetch_add(1, Ordering::Relaxed);
        let mut solver = model.solver.clone();
        let (found, part, complete) = probe_seeded_with_scope(
            &mut solver,
            vars,
            guidance.primary_branch_scope.as_deref(),
            obj,
            minimizing,
            target,
            context.stop(),
            context.seed().wrapping_add(attempt),
            Some(ClauseSharing::new(Arc::clone(&shared.clauses), context.worker())),
        );
        attempt += 1;
        merge_search_stats(&mut stats, part);
        if !complete {
            break;
        }
        match found {
            Some((solution, value)) => {
                shared.report_improvement(context, value as i64, &solution, WorkerSource { kind: "probe", worker: context.worker() });
            }
            None => shared.record_probe_unsat(target),
        }
    }
    shared.finish_if_probed();
    CopWorkerResult { stats }
}

#[allow(clippy::too_many_arguments)]
fn optimize_worker(
    solver: &mut Solver,
    vars: &[VarId],
    objective: Objective,
    shared: &CopShared,
    context: &WorkerContext<Improvement, CopWorkerResult>,
    source: WorkerSource,
    clause_worker: Option<usize>,
    conflict_budget: Option<u64>,
    guidance: &SearchGuidance,
) -> (SolveStats, bool, bool) {
    let minimizing = objective.minimizing();
    let materialized = objective.var().is_some();
    let sharing = if materialized { clause_worker.map(|worker| ClauseSharing::new(Arc::clone(&shared.clauses), worker)) } else { None };
    // Importing an incumbent while a symbolic objective-bound propagator is
    // already active can invalidate that worker's root state. Symbolic workers
    // therefore search independently and only merge their verified incumbents;
    // materialized objectives retain cooperative bound and clause sharing.
    let shared_bound = materialized.then_some(&shared.best);
    let mut improved = false;
    let (_, stats, complete) = optimize_seeded_with_scope(
        solver,
        vars,
        guidance.primary_branch_scope.as_deref(),
        objective.search(),
        minimizing,
        context.stop(),
        context.seed(),
        shared_bound,
        sharing,
        &[],
        conflict_budget,
        guidance.initial_phase.clone(),
        guidance.branch_order.clone(),
        |value, solution| improved |= shared.report_improvement(context, value, solution, source),
    );
    (stats, complete, improved)
}

fn run_lns_worker(
    model: Problem,
    shared: &CopShared,
    context: &WorkerContext<Improvement, CopWorkerResult>,
    guidance: &SearchGuidance,
) -> CopWorkerResult {
    let vars = &model.search;
    let objective = model.objective.clone().expect("COP lost its objective");
    let objective_var = model.var_objective().map(|(_, obj)| obj);
    let primary_members = guidance.primary_branch_scope.as_ref().map(|scope| {
        let mut members = vec![false; model.solver.store.num_vars()];
        for &variable in scope {
            members[variable.index()] = true;
        }
        members
    });
    let candidates = vars
        .iter()
        .enumerate()
        .filter_map(|(i, &var)| {
            let is_primary = primary_members.as_ref().is_none_or(|members| members[var.index()]);
            (is_primary && Some(var) != objective_var && !model.solver.store.is_fixed(var)).then_some(i)
        })
        .collect::<Vec<_>>();
    let mut stats = SolveStats::default();
    let mut attempt = 0u64;
    let mut lns = LnsState::new(context.seed());
    while !candidates.is_empty() && !context.is_cancelled() {
        let Some(solution) = shared.incumbent() else {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        };
        let attempt_seed = context.seed().wrapping_add(attempt);
        attempt += 1;
        let mut solver = model.solver.clone();
        if !fix_neighborhood(&mut solver, vars, &solution, &candidates, attempt_seed, lns.relax_percent()) {
            continue;
        }
        shared.lns_attempts.fetch_add(1, Ordering::Relaxed);
        let source = WorkerSource { kind: "lns", worker: context.worker() };
        let (part, complete, improved) = optimize_worker_with_seed(
            &mut solver,
            vars,
            objective.clone(),
            attempt_seed,
            shared,
            context,
            source,
            None,
            Some(lns.budget()),
            guidance,
        );
        if improved {
            shared.lns_improved.fetch_add(1, Ordering::Relaxed);
        }
        lns.record(improved, complete);
        merge_search_stats(&mut stats, part);
    }
    CopWorkerResult { stats }
}

#[allow(clippy::too_many_arguments)]
fn optimize_worker_with_seed(
    solver: &mut Solver,
    vars: &[VarId],
    objective: Objective,
    seed: u64,
    shared: &CopShared,
    context: &WorkerContext<Improvement, CopWorkerResult>,
    source: WorkerSource,
    clause_worker: Option<usize>,
    conflict_budget: Option<u64>,
    guidance: &SearchGuidance,
) -> (SolveStats, bool, bool) {
    let minimizing = objective.minimizing();
    let materialized = objective.var().is_some();
    let sharing = if materialized { clause_worker.map(|worker| ClauseSharing::new(Arc::clone(&shared.clauses), worker)) } else { None };
    let shared_bound = materialized.then_some(&shared.best);
    let mut improved = false;
    let (_, stats, complete) = optimize_seeded_with_scope(
        solver,
        vars,
        guidance.primary_branch_scope.as_deref(),
        objective.search(),
        minimizing,
        context.stop(),
        seed,
        shared_bound,
        sharing,
        &[],
        conflict_budget,
        Vec::new(),
        Vec::new(),
        |value, solution| improved |= shared.report_improvement(context, value, solution, source),
    );
    (stats, complete, improved)
}

/// Outcome of a CSP portfolio: a solution, an UNSAT proof, or interruption.
pub(crate) struct CspOutcome {
    pub(crate) solution: Option<Vec<i32>>,
    pub(crate) stats: SolveStats,
    pub(crate) decided: bool,
    pub(crate) shared_clauses: usize,
    pub(crate) imported_clauses: u64,
    pub(crate) search_kind: &'static str,
}

struct CspWorkerResult {
    stats: SolveStats,
}

struct CspShared {
    solution: Mutex<Option<(usize, Vec<i32>)>>,
    decided: AtomicBool,
    clauses: Arc<SharedClausePool>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RootPreparation {
    Ready,
    Inconsistent,
    Interrupted,
}

fn prepare_root(problem: &mut Problem, stop: &AtomicBool) -> RootPreparation {
    if stop.load(Ordering::Acquire) {
        return RootPreparation::Interrupted;
    }
    problem.solver.enqueue_all();
    let propagated = problem.solver.propagate_until(|| stop.load(Ordering::Acquire));
    problem.solver.store.events.clear();
    if stop.load(Ordering::Acquire) {
        RootPreparation::Interrupted
    } else if propagated.is_err() {
        RootPreparation::Inconsistent
    } else {
        RootPreparation::Ready
    }
}

/// Race diversified find-one workers on the same CSP.
pub(crate) fn solve_csp(mut problem: Problem, stop: &AtomicBool, options: RunOptions, guidance: SearchGuidance) -> CspOutcome {
    problem.solver.set_force_scope_reasons(options.force_scope_reasons);
    let chronological_compatible = guidance.is_chronological_dfs_compatible() && options.conflict_limit.is_none();
    let scoped_no_learn = options.no_learn_csp && chronological_compatible;
    let search_kind = if scoped_no_learn || (options.workers == 1 && chronological_compatible) {
        "chronological-dfs"
    } else if chronological_compatible {
        "hybrid"
    } else {
        "cdcl"
    };
    match prepare_root(&mut problem, stop) {
        RootPreparation::Ready => {}
        RootPreparation::Inconsistent => {
            return CspOutcome {
                solution: None,
                stats: SolveStats { failures: 1, ..SolveStats::default() },
                decided: true,
                shared_clauses: 0,
                imported_clauses: 0,
                search_kind,
            };
        }
        RootPreparation::Interrupted => {
            return CspOutcome {
                solution: None,
                stats: SolveStats::default(),
                decided: false,
                shared_clauses: 0,
                imported_clauses: 0,
                search_kind,
            };
        }
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let shared = Arc::new(CspShared {
        solution: Mutex::new(None),
        decided: AtomicBool::new(false),
        clauses: Arc::new(SharedClausePool::with_capacity(options.shared_pool_capacity)),
    });
    let mut models = Vec::with_capacity(options.workers);
    models.push(problem);
    for _ in 1..options.workers {
        if stop.load(Ordering::Acquire) {
            break;
        }
        models.push(models[0].clone());
    }

    let execution = execute_workers_silent(models, stop, cancel, options.seed, {
        let shared = Arc::clone(&shared);
        move |context, model| {
            let Problem { mut solver, search, objective: _ } = model;
            let vars = &search;
            let use_chronological_dfs = scoped_no_learn || (context.worker() == 0 && chronological_compatible);
            let (solution, stats, complete) = if use_chronological_dfs {
                find_one_seeded_with_scope(&mut solver, vars, guidance.primary_branch_scope.as_deref(), context.stop(), context.seed())
            } else {
                let sharing = ClauseSharing::new(Arc::clone(&shared.clauses), context.worker());
                if guidance.is_empty() && options.conflict_limit.is_none() {
                    decide_sat_shared_seeded(&mut solver, vars, context.stop(), context.seed(), Some(sharing), context.worker() % 2 == 1)
                } else {
                    decide_sat_assuming_seeded_with_scope(
                        &mut solver,
                        vars,
                        guidance.primary_branch_scope.as_deref(),
                        &[],
                        context.stop(),
                        context.seed(),
                        Some(sharing),
                        worker_conflict_quota(options.conflict_limit, context.worker(), options.workers),
                        guidance.initial_phase.clone(),
                        guidance.branch_order.clone(),
                    )
                }
            };
            if let Some(solution) = solution {
                let mut slot = shared.solution.lock().unwrap();
                if slot.as_ref().is_none_or(|(worker, _)| context.worker() < *worker) {
                    *slot = Some((context.worker(), solution));
                }
                shared.decided.store(true, Ordering::Release);
                context.cancel();
            } else if complete {
                shared.decided.store(true, Ordering::Release);
                context.cancel();
            }
            CspWorkerResult { stats }
        }
    });

    let mut stats = SolveStats::default();
    for report in execution.reports {
        merge_search_stats(&mut stats, report.result.stats);
    }
    let solution = shared.solution.lock().unwrap().as_ref().map(|(_, solution)| solution.clone());
    CspOutcome {
        solution,
        stats,
        decided: shared.decided.load(Ordering::Acquire),
        shared_clauses: shared.clauses.len(),
        imported_clauses: shared.clauses.imported(),
        search_kind,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cop_with_progress<W: Write>(
    mut problem: Problem,
    verbose: bool,
    stop: &AtomicBool,
    w: &mut W,
    options: RunOptions,
    guidance: SearchGuidance,
    publish_incumbents: bool,
    on_improvement: &mut dyn FnMut(i64, Option<&[i32]>),
) -> Result<ParallelOutcome, String> {
    problem.solver.set_force_scope_reasons(options.force_scope_reasons);
    let minimizing = problem.objective_dir().expect("COP lost its objective");
    match prepare_root(&mut problem, stop) {
        RootPreparation::Ready => {}
        RootPreparation::Inconsistent => {
            return Ok(ParallelOutcome {
                best: None,
                stats: SolveStats { failures: 1, ..SolveStats::default() },
                proved: true,
                shared_clauses: 0,
                imported_clauses: 0,
                split_jobs: options.split.then_some((0, 0)),
                probe_stats: (options.probes > 0).then_some((0, 0)),
                lns_stats: (options.lns > 0).then_some((0, 0)),
            });
        }
        RootPreparation::Interrupted => {
            return Ok(ParallelOutcome {
                best: None,
                stats: SolveStats::default(),
                proved: false,
                shared_clauses: 0,
                imported_clauses: 0,
                split_jobs: options.split.then_some((0, 0)),
                probe_stats: (options.probes > 0).then_some((0, 0)),
                lns_stats: (options.lns > 0).then_some((0, 0)),
            });
        }
    }
    let var_objective = problem.var_objective();
    let (probe_lower, probe_upper) = var_objective
        .map(|(_, obj)| (problem.solver.store.min(obj) as i64, problem.solver.store.max(obj) as i64))
        .unwrap_or((i64::MIN, i64::MAX));
    let cancel = Arc::new(AtomicBool::new(false));
    let shared = Arc::new(CopShared {
        cancel: Arc::clone(&cancel),
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
        publish_incumbents,
    });
    let mut models = Vec::with_capacity(options.workers);
    models.push(problem);
    for _ in 1..options.workers {
        models.push(models[0].clone());
    }
    let regular_workers = options.workers - options.probes - options.lns;
    let lns_end = regular_workers + options.lns;
    let mut printed = None;

    let execution = execute_workers(
        models,
        stop,
        cancel,
        options.seed,
        {
            let shared = Arc::clone(&shared);
            move |context: WorkerContext<Improvement, CopWorkerResult>, mut model| {
                if context.worker() >= lns_end {
                    let (_, obj) = var_objective.expect("symbolic objective cannot use probes");
                    return run_probe_worker(model, obj, minimizing, &shared, &context, &guidance);
                }
                if context.worker() >= regular_workers {
                    return run_lns_worker(model, &shared, &context, &guidance);
                }
                let vars = &model.search;
                if !options.split {
                    let mut solver = std::mem::take(&mut model.solver);
                    let source = WorkerSource { kind: "portfolio", worker: context.worker() };
                    let (stats, complete, _) = optimize_worker(
                        &mut solver,
                        vars,
                        model.objective.clone().expect("COP lost its objective"),
                        &shared,
                        &context,
                        source,
                        Some(context.worker()),
                        worker_conflict_quota(options.conflict_limit, context.worker(), regular_workers),
                        &guidance,
                    );
                    if complete {
                        shared.proved.store(true, Ordering::Release);
                        shared.stop();
                    }
                    return CopWorkerResult { stats };
                }

                let split_target = 2 * regular_workers;
                let split_limit = 4 * regular_workers as u64;
                let (_, obj) = var_objective.expect("symbolic objective cannot split");
                let mut stats = SolveStats::default();
                while let Some(mut cube) = shared.work.take(context.stop()) {
                    while shared.work.needs_split(split_target, split_limit) {
                        let mut probe = model.solver.clone();
                        let Some(lit) = split_cube_seeded_with_scope(
                            &mut probe,
                            vars,
                            guidance.primary_branch_scope.as_deref(),
                            &cube,
                            context.stop(),
                            context.seed(),
                            Some(shared.clauses.lazy_atoms()),
                        ) else {
                            break;
                        };
                        let mut sibling = cube.clone();
                        sibling.push(lit.negate());
                        if !shared.work.try_push_split(sibling, split_target, split_limit) {
                            break;
                        }
                        cube.push(lit);
                    }
                    let (job_stats, complete) = if context.is_cancelled() {
                        (SolveStats::default(), false)
                    } else {
                        let mut solver = model.solver.clone();
                        let (_, job_stats, complete) = optimize_seeded_with_scope(
                            &mut solver,
                            vars,
                            guidance.primary_branch_scope.as_deref(),
                            SearchObjective::Var(obj),
                            minimizing,
                            context.stop(),
                            context.seed(),
                            Some(&shared.best),
                            Some(ClauseSharing::new(Arc::clone(&shared.clauses), context.worker())),
                            &cube,
                            worker_conflict_quota(options.conflict_limit, context.worker(), regular_workers),
                            guidance.initial_phase.clone(),
                            guidance.branch_order.clone(),
                            |value, solution| {
                                shared.report_improvement(
                                    &context,
                                    value,
                                    solution,
                                    WorkerSource { kind: "split", worker: context.worker() },
                                );
                            },
                        );
                        (job_stats, complete)
                    };
                    merge_search_stats(&mut stats, job_stats);
                    if shared.work.finish(complete) {
                        shared.proved.store(true, Ordering::Release);
                        shared.stop();
                    }
                    if !complete || shared.proved.load(Ordering::Acquire) {
                        break;
                    }
                }
                CopWorkerResult { stats }
            }
        },
        |event| {
            let Improvement { value, source, solution } = event.payload;
            if printed.is_none_or(|old| if minimizing { value < old } else { value > old }) {
                printed = Some(value);
                on_improvement(value, solution.as_deref());
                if verbose {
                    if let Err(error) = writeln!(w, "o {value}")
                        .and_then(|_| writeln!(w, "c incumbent {value} source {} worker {}", source.kind, source.worker))
                        .and_then(|_| w.flush())
                    {
                        shared.stop();
                        return Err(error.to_string());
                    }
                }
            }
            Ok(EventControl::Continue)
        },
    )?;
    shared.stop();

    let mut stats = SolveStats::default();
    for report in execution.reports {
        merge_search_stats(&mut stats, report.result.stats);
    }
    let split_jobs = options.split.then(|| shared.work.stats());
    let probe_stats =
        (options.probes > 0).then(|| (shared.probe_attempts.load(Ordering::Relaxed), shared.probe_unsat.load(Ordering::Relaxed)));
    let lns_stats = (options.lns > 0).then(|| (shared.lns_attempts.load(Ordering::Relaxed), shared.lns_improved.load(Ordering::Relaxed)));
    let best = shared.solution.lock().unwrap().clone();
    Ok(ParallelOutcome {
        best,
        stats,
        proved: shared.proved.load(Ordering::Acquire),
        shared_clauses: shared.clauses.len(),
        imported_clauses: shared.clauses.imported(),
        split_jobs,
        probe_stats,
        lns_stats,
    })
}
