//! Assumption-aware execution for persistent CP sessions.
//!
//! This is a search adapter, not a second solve control plane. Status, proof,
//! bounds, events, and semantic replay remain owned by the canonical CP
//! orchestrator.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::problem::Problem;
use crate::search::{self, Assumption, SolveStats};

use super::portfolio;
use super::portfolio::{CspOutcome, ParallelOutcome, RootPreparation, SearchGuidance};

pub(crate) struct IncrementalSearch<'a> {
    assumptions: &'a [Assumption],
    clauses: Arc<SharedClausePool>,
    next_worker: usize,
    guidance: SearchGuidance,
}

impl<'a> IncrementalSearch<'a> {
    pub(crate) fn new(
        assumptions: &'a [Assumption],
        clauses: Arc<SharedClausePool>,
        first_worker: usize,
        guidance: SearchGuidance,
    ) -> Self {
        Self { assumptions, clauses, next_worker: first_worker, guidance }
    }

    pub(crate) fn next_worker(&self) -> usize {
        self.next_worker
    }

    pub(crate) fn solve_csp(&mut self, mut problem: Problem, stop: &AtomicBool, seed: u64, conflict_limit: Option<u64>) -> CspOutcome {
        let before_clauses = self.clauses.len();
        let before_imported = self.clauses.imported();
        match portfolio::prepare_root(&mut problem, stop) {
            RootPreparation::Ready => {}
            RootPreparation::Inconsistent => {
                return CspOutcome {
                    solution: None,
                    stats: SolveStats { failures: 1, ..SolveStats::default() },
                    decided: true,
                    shared_clauses: 0,
                    imported_clauses: 0,
                    search_kind: "incremental-cdcl",
                };
            }
            RootPreparation::Interrupted => {
                return CspOutcome {
                    solution: None,
                    stats: SolveStats::default(),
                    decided: false,
                    shared_clauses: 0,
                    imported_clauses: 0,
                    search_kind: "incremental-cdcl",
                };
            }
        }

        let worker = self.take_worker();
        let (solution, stats, decided) = search::decide_sat_assuming_seeded_with_scope(
            &mut problem.solver,
            &problem.search,
            self.guidance.primary_branch_scope.as_deref(),
            self.assumptions,
            stop,
            seed,
            Some(ClauseSharing::new(Arc::clone(&self.clauses), worker)),
            conflict_limit,
            self.guidance.initial_phase.clone(),
            self.guidance.branch_order.clone(),
        );
        CspOutcome {
            solution,
            stats,
            decided,
            shared_clauses: self.clauses.len().saturating_sub(before_clauses),
            imported_clauses: self.clauses.imported().saturating_sub(before_imported),
            search_kind: "incremental-cdcl",
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_cop(
        &mut self,
        mut problem: Problem,
        stop: &AtomicBool,
        seed: u64,
        conflict_limit: Option<u64>,
        advisory_phase: &[Option<i32>],
        publish_incumbents: bool,
        on_improvement: &mut dyn FnMut(i64, Option<&[i32]>),
    ) -> ParallelOutcome {
        let before_clauses = self.clauses.len();
        let before_imported = self.clauses.imported();
        match portfolio::prepare_root(&mut problem, stop) {
            RootPreparation::Ready => {}
            RootPreparation::Inconsistent => {
                return empty_cop_outcome(true, SolveStats { failures: 1, ..SolveStats::default() });
            }
            RootPreparation::Interrupted => return empty_cop_outcome(false, SolveStats::default()),
        }

        let objective = problem.objective.as_ref().expect("incremental COP lost its objective");
        let mut initial_phase = self.guidance.initial_phase.clone();
        if initial_phase.len() != problem.solver.store.num_vars() {
            initial_phase.resize(problem.solver.store.num_vars(), None);
        }
        if advisory_phase.len() == initial_phase.len() {
            for (current, &proposed) in initial_phase.iter_mut().zip(advisory_phase) {
                if current.is_none() {
                    *current = proposed;
                }
            }
        }
        let worker = self.take_worker();
        let (best, stats, proved) = search::optimize_assuming_seeded_with_scope(
            &mut problem.solver,
            &problem.search,
            self.guidance.primary_branch_scope.as_deref(),
            self.assumptions,
            objective.search(),
            objective.minimizing(),
            stop,
            seed,
            None,
            Some(ClauseSharing::new(Arc::clone(&self.clauses), worker)),
            conflict_limit,
            initial_phase,
            self.guidance.branch_order.clone(),
            |value, assignment| on_improvement(value, publish_incumbents.then_some(assignment)),
        );
        ParallelOutcome {
            best,
            stats,
            proved,
            shared_clauses: self.clauses.len().saturating_sub(before_clauses),
            imported_clauses: self.clauses.imported().saturating_sub(before_imported),
            split_jobs: None,
            probe_stats: None,
            lns_stats: None,
        }
    }

    fn take_worker(&mut self) -> usize {
        let worker = self.next_worker;
        self.next_worker = self.next_worker.saturating_add(1);
        worker
    }
}

fn empty_cop_outcome(proved: bool, stats: SolveStats) -> ParallelOutcome {
    ParallelOutcome {
        best: None,
        stats,
        proved,
        shared_clauses: 0,
        imported_clauses: 0,
        split_jobs: None,
        probe_stats: None,
        lns_stats: None,
    }
}
