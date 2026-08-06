//! Multithread portfolio for ordered-list local search.
//!
//! Worker zero deliberately uses the sequential profile, the caller seed, and
//! no incumbent injection. Extra workers diversify the ALNS parameters and may
//! adopt a strictly better shared sequence at a local optimum.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex, MutexGuard};

use super::alns::SearchProfile;
use super::local_search::{solve_collection_capped_worker, Score};
use super::metrics::ListSearchMetrics;
use super::moves::better;
use crate::engines::dual;
use crate::mix64;
use crate::model::list::{CollectionModel, CollectionSolution, ObjectiveTier};

const WORKER_SEED_STEP: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Clone)]
struct SharedBest {
    source: usize,
    lists: Vec<Vec<i32>>,
    score: Score,
    feasible: bool,
}

pub(super) struct SharedListIncumbent {
    best: Mutex<Option<SharedBest>>,
    generation: AtomicU64,
    publications: AtomicU64,
    injections: AtomicU64,
}

impl SharedListIncumbent {
    fn new() -> Self {
        Self { best: Mutex::new(None), generation: AtomicU64::new(0), publications: AtomicU64::new(0), injections: AtomicU64::new(0) }
    }

    fn lock_best(&self) -> MutexGuard<'_, Option<SharedBest>> {
        self.best.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn publish(&self, source: usize, lists: &[Vec<i32>], score: &Score, feasible: bool) -> bool {
        let mut incumbent = self.lock_best();
        if incumbent.as_ref().is_some_and(|current| !better(feasible, score, current.feasible, &current.score)) {
            return false;
        }
        *incumbent = Some(SharedBest { source, lists: lists.to_vec(), score: score.clone(), feasible });
        self.publications.fetch_add(1, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Release);
        true
    }
}

pub(super) struct WorkerCoordination<'a> {
    shared: &'a SharedListIncumbent,
    worker: usize,
    injection_period: u64,
    last_seen_generation: u64,
}

impl<'a> WorkerCoordination<'a> {
    fn new(shared: &'a SharedListIncumbent, worker: usize, profile: SearchProfile, items: usize) -> Self {
        let root = integer_sqrt(items.max(1)) as u64;
        let injection_period = match profile {
            SearchProfile::Intensify => 1,
            SearchProfile::Balanced => root.max(1),
            SearchProfile::Sequential | SearchProfile::Diversify => 0,
        };
        Self { shared, worker, injection_period, last_seen_generation: 0 }
    }

    pub(super) fn publish(&self, lists: &[Vec<i32>], score: &Score, feasible: bool) -> bool {
        self.shared.publish(self.worker, lists, score, feasible)
    }

    pub(super) fn maybe_inject(&mut self, local_optimum: u64, incumbent_score: &Score, incumbent_feasible: bool) -> Option<Vec<Vec<i32>>> {
        if self.injection_period == 0 || !local_optimum.is_multiple_of(self.injection_period) {
            return None;
        }
        let generation = self.shared.generation.load(Ordering::Acquire);
        if generation == self.last_seen_generation {
            return None;
        }
        self.last_seen_generation = generation;
        let incumbent = self.shared.lock_best();
        let shared = incumbent.as_ref()?;
        if shared.source == self.worker || !better(shared.feasible, &shared.score, incumbent_feasible, incumbent_score) {
            return None;
        }
        self.shared.injections.fetch_add(1, Ordering::Relaxed);
        Some(shared.lists.clone())
    }
}

/// Metrics for one worker in a list-search portfolio.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ListPortfolioWorkerMetrics {
    pub worker: usize,
    pub seed: u64,
    pub profile: String,
    pub feasible: bool,
    pub objectives: Vec<i64>,
    pub search: ListSearchMetrics,
}

/// Aggregate metrics for a multithread ordered-list search.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ListPortfolioMetrics {
    pub workers: usize,
    pub publications: u64,
    pub injections: u64,
    pub best_worker: Option<usize>,
    pub worker_metrics: Vec<ListPortfolioWorkerMetrics>,
}

impl fmt::Display for ListPortfolioMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "qayd list-LS portfolio: workers={} publications={} injections={} best-worker={}",
            self.workers,
            self.publications,
            self.injections,
            self.best_worker.map_or_else(|| "none".to_owned(), |worker| worker.to_string())
        )?;
        for worker in &self.worker_metrics {
            writeln!(
                f,
                "  worker={} seed={} profile={} feasible={} objectives={:?} candidates={}",
                worker.worker, worker.seed, worker.profile, worker.feasible, worker.objectives, worker.search.candidates
            )?;
        }
        Ok(())
    }
}

enum PortfolioMessage {
    Progress(i64),
    Finished { worker: usize, seed: u64, profile: SearchProfile, solution: CollectionSolution, search: Box<ListSearchMetrics> },
}

/// Solve with `workers` independent list-search trajectories and a shared
/// incumbent. `workers == 1` delegates directly to the sequential engine.
pub fn solve_collection_parallel(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    workers: usize,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
) -> CollectionSolution {
    solve_collection_parallel_internal(model, seed, stop, workers, u64::MAX, hint, report, false, true).0
}

/// Portfolio entry point for a frontend that already validated the collection
/// model against the shared solve budget.
#[cfg(feature = "python")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_collection_parallel_validated(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    workers: usize,
    max_iters: u64,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
    profile: bool,
) -> (CollectionSolution, ListPortfolioMetrics) {
    solve_collection_parallel_internal(model, seed, stop, workers, max_iters, hint, report, profile, false)
}

/// Deterministic iteration-capped, profiled portfolio entry point.
pub fn solve_collection_parallel_capped_profiled(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    workers: usize,
    max_iters: u64,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ListPortfolioMetrics) {
    solve_collection_parallel_internal(model, seed, stop, workers, max_iters, hint, report, true, true)
}

#[allow(clippy::too_many_arguments)]
fn solve_collection_parallel_internal(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    workers: usize,
    max_iters: u64,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
    metrics_enabled: bool,
    validate_model: bool,
) -> (CollectionSolution, ListPortfolioMetrics) {
    let workers = workers.max(1);
    if workers == 1 {
        let (mut solution, search) = solve_collection_capped_worker(
            model,
            seed,
            stop,
            max_iters,
            hint,
            report,
            metrics_enabled,
            validate_model,
            SearchProfile::Sequential,
            None,
        );
        let dual_bound = if !stop.load(Ordering::Relaxed) { dual::compute(model, stop) } else { None };
        dual::attach(model, &mut solution, dual_bound);
        let worker_metrics = ListPortfolioWorkerMetrics {
            worker: 0,
            seed,
            profile: SearchProfile::Sequential.name().to_owned(),
            feasible: solution.feasible,
            objectives: solution.objectives.clone(),
            search,
        };
        return (
            solution,
            ListPortfolioMetrics { workers: 1, publications: 0, injections: 0, best_worker: Some(0), worker_metrics: vec![worker_metrics] },
        );
    }

    let shared = SharedListIncumbent::new();
    let (sender, receiver) = mpsc::channel();
    let mut completed = Vec::with_capacity(workers);
    std::thread::scope(|scope| {
        for worker in 0..workers {
            let sender = sender.clone();
            let profile = worker_profile(worker);
            let worker_seed = portfolio_seed(seed, worker);
            let worker_hint = (worker == 0).then_some(hint).flatten();
            let coordination = Some(WorkerCoordination::new(&shared, worker, profile, model.items.len()));
            scope.spawn(move || {
                let mut progress = |objective| {
                    let _ = sender.send(PortfolioMessage::Progress(objective));
                };
                let (solution, search) = solve_collection_capped_worker(
                    model,
                    worker_seed,
                    stop,
                    max_iters,
                    worker_hint,
                    &mut progress,
                    metrics_enabled,
                    validate_model,
                    profile,
                    coordination,
                );
                let _ = sender.send(PortfolioMessage::Finished { worker, seed: worker_seed, profile, solution, search: Box::new(search) });
            });
        }
        drop(sender);

        let minimize_primary = model.objectives.first().is_none_or(|tier| tier.minimize);
        let mut reported: Option<i64> = None;
        for message in receiver {
            match message {
                PortfolioMessage::Progress(objective) => {
                    let improved = reported.is_none_or(|current| if minimize_primary { objective < current } else { objective > current });
                    if improved {
                        reported = Some(objective);
                        report(objective);
                    }
                }
                PortfolioMessage::Finished { worker, seed, profile, solution, search } => {
                    completed.push((worker, seed, profile, solution, *search));
                }
            }
        }
    });

    completed.sort_by_key(|entry| entry.0);
    let best_idx = completed
        .iter()
        .enumerate()
        .fold(None, |best: Option<usize>, (idx, candidate)| match best {
            Some(current) if !solution_better(&candidate.3, &completed[current].3, model) => Some(current),
            _ => Some(idx),
        })
        .expect("a non-empty portfolio must complete at least one worker");
    let mut solution = completed[best_idx].3.clone();
    let dual_bound = if !stop.load(Ordering::Relaxed) { dual::compute(model, stop) } else { None };
    dual::attach(model, &mut solution, dual_bound);
    let best_worker = completed[best_idx].0;
    let worker_metrics = completed
        .into_iter()
        .map(|(worker, seed, profile, solution, search)| ListPortfolioWorkerMetrics {
            worker,
            seed,
            profile: profile.name().to_owned(),
            feasible: solution.feasible,
            objectives: solution.objectives,
            search,
        })
        .collect();
    let metrics = ListPortfolioMetrics {
        workers,
        publications: shared.publications.load(Ordering::Relaxed),
        injections: shared.injections.load(Ordering::Relaxed),
        best_worker: Some(best_worker),
        worker_metrics,
    };
    (solution, metrics)
}

fn solution_better(candidate: &CollectionSolution, incumbent: &CollectionSolution, model: &CollectionModel) -> bool {
    if candidate.feasible != incumbent.feasible {
        return candidate.feasible;
    }
    if !candidate.feasible {
        return false;
    }
    if model.schedule.is_some() {
        return candidate.objectives < incumbent.objectives;
    }
    for ((candidate_value, incumbent_value), tier) in candidate.objectives.iter().zip(&incumbent.objectives).zip(&model.objectives) {
        if candidate_value == incumbent_value {
            continue;
        }
        return if tier.minimize { candidate_value < incumbent_value } else { candidate_value > incumbent_value };
    }
    false
}

fn worker_profile(worker: usize) -> SearchProfile {
    match worker % 4 {
        0 => SearchProfile::Sequential,
        1 => SearchProfile::Intensify,
        2 => SearchProfile::Diversify,
        _ => SearchProfile::Balanced,
    }
}

fn portfolio_seed(seed: u64, worker: usize) -> u64 {
    if worker == 0 {
        seed
    } else {
        mix64(seed ^ (worker as u64).wrapping_mul(WORKER_SEED_STEP))
    }
}

fn integer_sqrt(value: usize) -> usize {
    if value < 2 {
        return value;
    }
    let mut lo = 1usize;
    let mut hi = value.min(1usize << (usize::BITS / 2));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if mid <= value / mid {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Deterministic oracle for publication and controlled injection semantics.
#[doc(hidden)]
pub fn audit_incumbent_exchange() -> bool {
    let shared = SharedListIncumbent::new();
    let weak = Score { violation: 1, tiers: vec![100].into() };
    let strong = Score { violation: 0, tiers: vec![50].into() };
    let mut receiver = WorkerCoordination::new(&shared, 1, SearchProfile::Intensify, 16);
    shared.publish(0, &[vec![1, 2]], &weak, false)
        && receiver.maybe_inject(1, &weak, false).is_none()
        && shared.publish(0, &[vec![2, 1]], &strong, true)
        && receiver.maybe_inject(2, &weak, false) == Some(vec![vec![2, 1]])
        && shared.publications.load(Ordering::Relaxed) == 2
        && shared.injections.load(Ordering::Relaxed) == 1
}

/// Deterministic oracle for feasibility and mixed-sense lexicographic merging.
#[doc(hidden)]
pub fn audit_portfolio_merge() -> bool {
    let model = CollectionModel {
        items: Vec::new(),
        lists: 1,
        objectives: vec![
            ObjectiveTier { minimize: false, terms: Vec::new(), max_terms: None },
            ObjectiveTier { minimize: true, terms: Vec::new(), max_terms: None },
        ],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let solution = |feasible, objectives| CollectionSolution {
        lists: vec![Vec::new()],
        objectives,
        feasible,
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        bound: None,
    };
    let incumbent = solution(true, vec![5, 10]);
    solution_better(&solution(true, vec![6, 999]), &incumbent, &model)
        && solution_better(&solution(true, vec![5, 9]), &incumbent, &model)
        && !solution_better(&solution(true, vec![5, 11]), &incumbent, &model)
        && solution_better(&incumbent, &solution(false, vec![100, 0]), &model)
}
