//! Assumption-aware execution for persistent CP sessions.
//!
//! This is a search adapter, not a second solve control plane. Status, proof,
//! bounds, events, and semantic replay remain owned by the canonical CP
//! orchestrator.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::orchestrator::{execute_workers, execute_workers_silent, merge_search_stats, EventControl, WorkerContext};
use crate::problem::Problem;
use crate::search::{self, Assumption, SharedObjectiveBound, SolveStats};

use super::portfolio;
use super::portfolio::{CspOutcome, ParallelOutcome, SearchGuidance};

pub(crate) struct IncrementalSearch<'a> {
    assumptions: &'a [Assumption],
    clauses: Arc<SharedClausePool>,
    next_worker: &'a mut usize,
    workers: usize,
    guidance: SearchGuidance,
}

impl<'a> IncrementalSearch<'a> {
    pub(crate) fn new(
        assumptions: &'a [Assumption],
        clauses: Arc<SharedClausePool>,
        next_worker: &'a mut usize,
        workers: usize,
        guidance: SearchGuidance,
    ) -> Self {
        debug_assert!(workers > 0);
        Self { assumptions, clauses, next_worker, workers, guidance }
    }

    pub(crate) fn next_worker(&self) -> usize {
        *self.next_worker
    }

    pub(crate) fn solve_csp(&mut self, problem: Problem, stop: &AtomicBool, seed: u64, conflict_limit: Option<u64>) -> CspOutcome {
        let before_clauses = self.clauses.len();
        let before_imported = self.clauses.imported();
        if stop.load(Ordering::Acquire) {
            return CspOutcome {
                solution: None,
                stats: SolveStats::default(),
                decided: false,
                shared_clauses: 0,
                imported_clauses: 0,
                search_kind: "incremental-cdcl",
            };
        }
        let worker_count = self.workers;
        // Reserve the whole epoch before any fallible preparation. Once an
        // epoch has started, abandoned worker ids are never reused, keeping
        // persistent clause ownership monotone across errors, panics and
        // interrupted preparation. A pre-armed stop returns above and
        // deliberately consumes no ids.
        let first_worker = self.take_workers(worker_count);
        let Some(models) = clone_worker_models(problem, worker_count, stop) else {
            return CspOutcome {
                solution: None,
                stats: SolveStats::default(),
                decided: false,
                shared_clauses: 0,
                imported_clauses: 0,
                search_kind: if worker_count == 1 { "incremental-cdcl" } else { "incremental-cdcl-portfolio" },
            };
        };
        let shared = Arc::new(IncrementalCspShared { solution: Mutex::new(None), decided: AtomicBool::new(false) });
        let cancel = Arc::new(AtomicBool::new(false));
        let assumptions = self.assumptions;
        let clauses = Arc::clone(&self.clauses);
        let guidance = self.guidance.clone();
        let execution = execute_workers_silent(models, stop, cancel, seed, {
            let shared = Arc::clone(&shared);
            move |context, model| {
                let local_worker = context.worker();
                let worker = first_worker.checked_add(local_worker).expect("session worker range was preflighted");
                let Problem { mut solver, search, objective: _ } = model;
                let conflict_budget = portfolio::worker_conflict_quota(conflict_limit, local_worker, worker_count);
                if conflict_budget == Some(0) {
                    return IncrementalCspWorker { stats: SolveStats::default() };
                }
                let (solution, stats, complete) = search::decide_sat_assuming_seeded_with_scope(
                    &mut solver,
                    &search,
                    guidance.primary_branch_scope.as_deref(),
                    assumptions,
                    context.stop(),
                    context.seed(),
                    Some(ClauseSharing::new(Arc::clone(&clauses), worker)),
                    conflict_budget,
                    guidance.initial_phase.clone(),
                    guidance.branch_order.clone(),
                    guidance.search_phases.clone(),
                );
                if let Some(solution) = &solution {
                    let mut slot = shared.solution.lock().unwrap();
                    if slot.as_ref().is_none_or(|(current, _)| worker < *current) {
                        *slot = Some((worker, solution.clone()));
                    }
                    shared.decided.store(true, Ordering::Release);
                    context.cancel();
                } else if complete {
                    shared.decided.store(true, Ordering::Release);
                    context.cancel();
                }
                IncrementalCspWorker { stats }
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
            shared_clauses: self.clauses.len().saturating_sub(before_clauses),
            imported_clauses: self.clauses.imported().saturating_sub(before_imported),
            search_kind: if worker_count == 1 { "incremental-cdcl" } else { "incremental-cdcl-portfolio" },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_cop(
        &mut self,
        problem: Problem,
        scoped_assumptions: &[Assumption],
        stop: &AtomicBool,
        seed: u64,
        conflict_limit: Option<u64>,
        guidance: &SearchGuidance,
        publish_incumbents: bool,
        on_improvement: &mut dyn FnMut(i64, Option<&[i32]>),
    ) -> ParallelOutcome {
        let before_clauses = self.clauses.len();
        let before_imported = self.clauses.imported();
        if stop.load(Ordering::Acquire) {
            return empty_cop_outcome(false, SolveStats::default());
        }
        let worker_count = self.workers;
        // Keep the same monotone reservation contract as satisfaction epochs:
        // only a stop observed before the epoch starts preserves the range.
        let first_worker = self.take_workers(worker_count);
        let Some(models) = clone_worker_models(problem, worker_count, stop) else {
            return empty_cop_outcome(false, SolveStats::default());
        };
        let objective = models[0].objective.as_ref().expect("incremental COP lost its objective");
        assert!(objective.var().is_some(), "persistent clause sharing requires a materialized objective");
        let minimizing = objective.minimizing();
        let shared = Arc::new(IncrementalCopShared {
            best: SharedObjectiveBound::new(None),
            solution: Mutex::new(None),
            proved: AtomicBool::new(false),
            minimizing,
            publish_incumbents,
        });
        let cancel = Arc::new(AtomicBool::new(false));
        let mut assumptions = Vec::with_capacity(self.assumptions.len().saturating_add(scoped_assumptions.len()));
        assumptions.extend_from_slice(self.assumptions);
        assumptions.extend_from_slice(scoped_assumptions);
        let assumptions = Arc::new(assumptions);
        let clauses = Arc::clone(&self.clauses);
        let guidance = guidance.clone();
        let execution = execute_workers(
            models,
            stop,
            cancel,
            seed,
            {
                let shared = Arc::clone(&shared);
                move |context: WorkerContext<IncrementalImprovement, IncrementalCopWorker>, model| {
                    let local_worker = context.worker();
                    let worker = first_worker.checked_add(local_worker).expect("session worker range was preflighted");
                    let Problem { mut solver, search, objective } = model;
                    let objective = objective.expect("incremental COP lost its objective");
                    let conflict_budget = portfolio::worker_conflict_quota(conflict_limit, local_worker, worker_count);
                    if conflict_budget == Some(0) {
                        return IncrementalCopWorker { stats: SolveStats::default() };
                    }
                    let (_, stats, complete) = search::optimize_assuming_seeded_with_scope_and_relaxation(
                        &mut solver,
                        &search,
                        guidance.primary_branch_scope.as_deref(),
                        assumptions.as_slice(),
                        objective.search(),
                        minimizing,
                        context.stop(),
                        context.seed(),
                        Some(&shared.best),
                        Some(ClauseSharing::new(Arc::clone(&clauses), worker)),
                        conflict_budget,
                        guidance.initial_phase.clone(),
                        guidance.branch_order.clone(),
                        guidance.search_phases.clone(),
                        guidance.linear.clone(),
                        |value, solution| shared.report_improvement(&context, worker, value, solution),
                    );
                    if complete {
                        shared.proved.store(true, Ordering::Release);
                        context.cancel();
                    }
                    IncrementalCopWorker { stats }
                }
            },
            |event| {
                let improvement = event.payload;
                on_improvement(improvement.value, improvement.solution.as_deref());
                Ok::<_, ()>(if stop.load(Ordering::Acquire) { EventControl::Stop } else { EventControl::Continue })
            },
        )
        .expect("the incremental progress handler is infallible");

        let mut stats = SolveStats::default();
        for report in execution.reports {
            merge_search_stats(&mut stats, report.result.stats);
        }
        let best = shared.solution.lock().unwrap().as_ref().map(|(_, solution, value)| (solution.clone(), *value));
        ParallelOutcome {
            best,
            stats,
            proved: shared.proved.load(Ordering::Acquire),
            certified_bound: false,
            shared_clauses: self.clauses.len().saturating_sub(before_clauses),
            imported_clauses: self.clauses.imported().saturating_sub(before_imported),
            split_jobs: None,
            probe_stats: None,
            lns_stats: None,
        }
    }

    fn take_workers(&mut self, count: usize) -> usize {
        let first = *self.next_worker;
        *self.next_worker = first.checked_add(count).expect("session worker range was preflighted");
        first
    }
}

struct IncrementalCspShared {
    solution: Mutex<Option<(usize, Vec<i32>)>>,
    decided: AtomicBool,
}

struct IncrementalCspWorker {
    stats: SolveStats,
}

#[derive(Clone)]
struct IncrementalImprovement {
    value: i64,
    solution: Option<Vec<i32>>,
}

struct IncrementalCopWorker {
    stats: SolveStats,
}

struct IncrementalCopShared {
    best: SharedObjectiveBound,
    solution: Mutex<Option<(usize, Vec<i32>, i64)>>,
    proved: AtomicBool,
    minimizing: bool,
    publish_incumbents: bool,
}

impl IncrementalCopShared {
    fn report_improvement(
        &self,
        context: &WorkerContext<IncrementalImprovement, IncrementalCopWorker>,
        worker: usize,
        value: i64,
        solution: &[i32],
    ) {
        let mut slot = self.solution.lock().unwrap();
        let strictly_better = slot.as_ref().is_none_or(|(_, _, current)| if self.minimizing { value < *current } else { value > *current });
        let equal_but_deterministic = slot.as_ref().is_some_and(|(current_worker, current_solution, current)| {
            value == *current && (worker < *current_worker || (worker == *current_worker && solution < current_solution.as_slice()))
        });
        if !strictly_better && !equal_but_deterministic {
            return;
        }
        if strictly_better {
            self.best.publish(value);
        }
        *slot = Some((worker, solution.to_vec(), value));
        drop(slot);
        if strictly_better {
            context.publish_latest(IncrementalImprovement { value, solution: self.publish_incumbents.then(|| solution.to_vec()) });
        }
    }
}

fn clone_worker_models(problem: Problem, workers: usize, stop: &AtomicBool) -> Option<Vec<Problem>> {
    let mut models = Vec::with_capacity(workers);
    models.push(problem);
    for _ in 1..workers {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        models.push(models[0].clone());
    }
    Some(models)
}

fn empty_cop_outcome(proved: bool, stats: SolveStats) -> ParallelOutcome {
    ParallelOutcome {
        best: None,
        stats,
        proved,
        certified_bound: false,
        shared_clauses: 0,
        imported_clauses: 0,
        split_jobs: None,
        probe_stats: None,
        lns_stats: None,
    }
}
