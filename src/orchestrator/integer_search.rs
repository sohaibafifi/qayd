//! Integer local-search orchestration adapter.
//!
//! The engine owns lowering and neighborhood search. This adapter owns shared
//! budgets, CP repair, semantic verification, events, and public result status.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::engines::ls::cop::{solve_ls_capped, solve_ls_capped_borrowed, LocalSearchOutcome, LocalSearchSpec, LsConfig};
use crate::engines::ls::disjunctive_schedule::DisjunctiveSchedulePlan;
use crate::engines::ls::integer::{IntegerLocalSearchPlan, IntegerWarmStartKind, IntegerWarmStartPlan};
use crate::engines::ls::scenario_schedule::{ConstructionLimits, ScenarioSchedulePlan};
use crate::expr::Expr;
use crate::ids::VarId;
use crate::model::{CompiledCp, Model};
use crate::problem::Objective as PhysicalObjective;

use super::{
    execute_workers, CandidateSolution, EngineKind, EngineReport, EventControl, EventSink, SolveBudget, SolveError, SolveEvent,
    SolveRequest, SolveResult, SolveStatus, TerminationReason, VerificationLevel,
};

pub(crate) struct IntegerWarmStart {
    pub(crate) candidate: CandidateSolution,
    pub(crate) physical_solution: Vec<i32>,
    pub(crate) physical_objective: i64,
    pub(crate) report: EngineReport,
}

#[derive(Clone)]
struct Improvement {
    objective: i64,
    assignment: Vec<i32>,
}

const INTERNAL_REPLAY_INTERVAL: Duration = Duration::from_millis(25);
const STRUCTURAL_WARM_START_ITERATIONS: u64 = 160;
const SIGNED_PRODUCT_SQUARES_WARM_START_ITERATIONS: u64 = 50_000;

struct RepairedCandidate {
    candidate: CandidateSolution,
    physical_solution: Vec<i32>,
    physical_objective: Option<i64>,
}

struct RepairContext<'a> {
    model: &'a Model,
    compiled: &'a CompiledCp,
    spec: &'a LocalSearchSpec,
    request: &'a SolveRequest,
    budget: &'a SolveBudget,
    stop: &'a AtomicBool,
}

struct SemanticRepairContext<'a> {
    model: &'a Model,
    compiled: &'a CompiledCp,
    request: &'a SolveRequest,
    budget: &'a SolveBudget,
    stop: &'a AtomicBool,
}

impl RepairContext<'_> {
    fn candidate(&self, values: &[i32], seed: u64, verification: VerificationLevel) -> Result<Option<CandidateSolution>, SolveError> {
        Ok(repair_candidate(self, values, seed, verification)?.map(|repaired| repaired.candidate))
    }
}

pub(crate) fn warm_start(
    model: &Model,
    compiled: &CompiledCp,
    plan: &IntegerWarmStartPlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    search_stop: &AtomicBool,
    transfer_stop: &AtomicBool,
) -> Result<Option<IntegerWarmStart>, SolveError> {
    match plan {
        IntegerWarmStartPlan::Local(plan) => local_warm_start(model, compiled, plan, request, budget, search_stop, transfer_stop),
        IntegerWarmStartPlan::ScenarioSchedule(plan) => {
            scenario_schedule_warm_start(model, compiled, plan, request, budget, search_stop, transfer_stop)
        }
        IntegerWarmStartPlan::DisjunctiveSchedule(plan) => {
            disjunctive_schedule_warm_start(model, compiled, plan, request, budget, search_stop, transfer_stop)
        }
        IntegerWarmStartPlan::Fallbacks(plans) => {
            for plan in plans {
                if budget.expired()
                    || search_stop.load(std::sync::atomic::Ordering::Acquire)
                    || transfer_stop.load(std::sync::atomic::Ordering::Acquire)
                {
                    return Ok(None);
                }
                if let Some(candidate) = warm_start(model, compiled, plan, request, budget, search_stop, transfer_stop)? {
                    return Ok(Some(candidate));
                }
            }
            Ok(None)
        }
    }
}

fn local_warm_start(
    model: &Model,
    compiled: &CompiledCp,
    plan: &IntegerLocalSearchPlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    search_stop: &AtomicBool,
    transfer_stop: &AtomicBool,
) -> Result<Option<IntegerWarmStart>, SolveError> {
    let Some(warm_start_kind) = plan.warm_start else {
        return Ok(None);
    };
    if budget.expired() {
        return Ok(None);
    }

    let started = Instant::now();
    let max_iterations = match warm_start_kind {
        IntegerWarmStartKind::Structural => STRUCTURAL_WARM_START_ITERATIONS,
        IntegerWarmStartKind::SignedProductSquares => SIGNED_PRODUCT_SQUARES_WARM_START_ITERATIONS,
    };
    // LocalModel owns one normalized spec copy, covered by estimated_bytes. The
    // much larger physical CP root remains borrowed throughout this warm start.
    let outcome = solve_ls_capped_borrowed(
        compiled.problem(),
        plan.spec.clone(),
        search_stop,
        request.seed,
        LsConfig { gls: true, min_conflicts: true, kick_bandit: false },
        max_iterations,
        |_, _, _| {},
    );
    let LocalSearchOutcome { best, iterations, moves, restarts, constraints, functionals, unsupported } = outcome;
    let Some((values, local_objective)) = best else {
        return Ok(None);
    };

    let repair = RepairContext { model, compiled, spec: &plan.spec, request, budget, stop: transfer_stop };
    let Some(repaired) = repair_candidate(&repair, &values, request.seed, VerificationLevel::Transfer)? else {
        return Ok(None);
    };
    let physical_objective = repaired
        .physical_objective
        .ok_or_else(|| SolveError::InvalidResult("local-search warm start has no physical objective".to_string()))?;
    if physical_objective != local_objective {
        return Err(SolveError::InvalidResult(format!(
            "integer warm start scored objective {local_objective}, physical replay produced {}",
            physical_objective
        )));
    }
    if repaired.candidate.objectives() != [physical_objective] {
        return Err(SolveError::InvalidResult(format!(
            "integer warm-start physical objective {} does not match canonical replay {:?}",
            physical_objective,
            repaired.candidate.objectives()
        )));
    }

    Ok(Some(IntegerWarmStart {
        candidate: repaired.candidate,
        physical_solution: repaired.physical_solution,
        physical_objective,
        report: EngineReport {
            engine: Some(EngineKind::IntegerLocalSearch),
            search: crate::search::SolveStats {
                solutions: 1,
                nodes: iterations,
                failures: restarts,
                ..crate::search::SolveStats::default()
            },
            elapsed: started.elapsed(),
            improvements: 1,
            metadata: vec![
                ("ls_role".to_string(), warm_start_kind.role().to_string()),
                ("ls_moves".to_string(), moves.to_string()),
                ("ls_constraints".to_string(), constraints.to_string()),
                ("ls_functionals".to_string(), functionals.to_string()),
                ("ls_unsupported".to_string(), unsupported.to_string()),
            ],
        },
    }))
}

fn scenario_schedule_warm_start(
    model: &Model,
    compiled: &CompiledCp,
    plan: &ScenarioSchedulePlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    search_stop: &AtomicBool,
    transfer_stop: &AtomicBool,
) -> Result<Option<IntegerWarmStart>, SolveError> {
    if budget.expired() {
        return Ok(None);
    }
    let started = Instant::now();
    let Some(solution) = plan.construct(model, search_stop, ConstructionLimits::default()) else {
        return Ok(None);
    };
    let repair = SemanticRepairContext { model, compiled, request, budget, stop: transfer_stop };
    let Some(repaired) = repair_semantic_candidate(&repair, &solution.values, request.seed, VerificationLevel::Transfer)? else {
        return Ok(None);
    };
    let physical_objective = repaired
        .physical_objective
        .ok_or_else(|| SolveError::InvalidResult("scenario-schedule warm start has no physical objective".to_string()))?;
    if physical_objective != solution.objective || repaired.candidate.objectives() != [solution.objective] {
        return Err(SolveError::InvalidResult(format!(
            "scenario-schedule warm start scored {}, physical replay produced {}, canonical replay produced {:?}",
            solution.objective,
            physical_objective,
            repaired.candidate.objectives()
        )));
    }
    Ok(Some(IntegerWarmStart {
        candidate: repaired.candidate,
        physical_solution: repaired.physical_solution,
        physical_objective,
        report: EngineReport {
            engine: Some(EngineKind::IntegerLocalSearch),
            search: crate::search::SolveStats { solutions: 1, nodes: solution.configurations, ..crate::search::SolveStats::default() },
            elapsed: started.elapsed(),
            improvements: solution.improvements.saturating_add(1),
            metadata: vec![
                ("ls_role".to_string(), "scenario_schedule_warm_start".to_string()),
                ("ls_configurations".to_string(), solution.configurations.to_string()),
                ("ls_candidate_visits".to_string(), solution.candidate_visits.to_string()),
            ],
        },
    }))
}

fn disjunctive_schedule_warm_start(
    model: &Model,
    compiled: &CompiledCp,
    plan: &DisjunctiveSchedulePlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    search_stop: &AtomicBool,
    transfer_stop: &AtomicBool,
) -> Result<Option<IntegerWarmStart>, SolveError> {
    if budget.expired() {
        return Ok(None);
    }
    let started = Instant::now();
    let Some(construction_budget) = plan.construction_budget() else {
        return Ok(None);
    };
    let Some(construction) = plan.construct(model, request.seed, search_stop, construction_budget) else {
        return Ok(None);
    };
    let minimizing = model.objectives()[0].is_minimize();
    let repair = SemanticRepairContext { model, compiled, request, budget, stop: transfer_stop };
    let mut best: Option<(RepairedCandidate, usize)> = None;
    let mut repaired_count = 0u64;
    let mut improvements = 0u64;
    for (index, assignment) in construction.assignments.iter().enumerate() {
        if budget.expired() || transfer_stop.load(Ordering::Acquire) {
            break;
        }
        let Some(repaired) = repair_partial_semantic_candidate(
            &repair,
            assignment,
            request.seed.wrapping_add(u64::try_from(index).unwrap_or(u64::MAX)),
            VerificationLevel::Transfer,
        )?
        else {
            continue;
        };
        repaired_count = repaired_count.saturating_add(1);
        let physical_objective = repaired
            .physical_objective
            .ok_or_else(|| SolveError::InvalidResult("disjunctive-schedule warm start has no physical objective".to_string()))?;
        let [canonical_objective] = repaired.candidate.objectives() else {
            return Err(SolveError::InvalidResult(format!(
                "disjunctive-schedule canonical replay produced {:?}, expected one objective",
                repaired.candidate.objectives()
            )));
        };
        if physical_objective != *canonical_objective {
            return Err(SolveError::InvalidResult(format!(
                "disjunctive-schedule physical replay produced {physical_objective}, canonical replay produced {canonical_objective}"
            )));
        }
        let improves = best.as_ref().is_none_or(|(current, _)| {
            let current = current.physical_objective.expect("a retained disjunctive candidate has an objective");
            if minimizing {
                physical_objective < current
            } else {
                physical_objective > current
            }
        });
        if improves {
            best = Some((repaired, index));
            improvements = improvements.saturating_add(1);
        }
    }
    let Some((repaired, selected)) = best else {
        return Ok(None);
    };
    let physical_objective = repaired.physical_objective.expect("a retained disjunctive candidate has a physical objective");
    Ok(Some(IntegerWarmStart {
        candidate: repaired.candidate,
        physical_solution: repaired.physical_solution,
        physical_objective,
        report: EngineReport {
            engine: Some(EngineKind::IntegerLocalSearch),
            search: crate::search::SolveStats {
                solutions: 1,
                nodes: construction.work,
                failures: u64::try_from(construction.assignments.len()).unwrap_or(u64::MAX).saturating_sub(repaired_count),
                ..crate::search::SolveStats::default()
            },
            elapsed: started.elapsed(),
            improvements,
            metadata: vec![
                ("ls_role".to_string(), "disjunctive_schedule_warm_start".to_string()),
                ("ls_construction_candidates".to_string(), construction.assignments.len().to_string()),
                ("ls_construction_work".to_string(), construction.work.to_string()),
                ("ls_construction_checkpoints".to_string(), construction.checkpoints.to_string()),
                ("ls_repaired_candidates".to_string(), repaired_count.to_string()),
                ("ls_selected_candidate".to_string(), selected.to_string()),
            ],
        },
    }))
}

pub(crate) fn solve(
    model: &Model,
    compiled: &CompiledCp,
    plan: &IntegerLocalSearchPlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    engine_stop: &AtomicBool,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    let started = Instant::now();
    let config = LsConfig { gls: true, min_conflicts: true, kick_bandit: false };
    let mut problem = compiled.problem().clone();
    if problem.objective.is_none() {
        problem.objective = Some(crate::problem::Objective::Expr(true, Expr::Const(0)));
    }
    let max_iterations = request.limits.iterations.unwrap_or(u64::MAX);
    let inputs = (0..request.threads)
        .map(|worker| (problem.clone(), plan.spec.clone(), worker_iteration_quota(max_iterations, worker, request.threads)))
        .collect();
    let publish_assignments = request.publish_incumbent_assignments;
    let repair = RepairContext { model, compiled, spec: &plan.spec, request, budget, stop: engine_stop };
    let minimizing = model.objectives().first().is_none_or(crate::model::Objective::is_minimize);
    let mut checkpoint_best: Option<CandidateSolution> = None;
    let mut checkpoint_replays = 0u64;
    let mut last_checkpoint = None;
    let execution = execute_workers(
        inputs,
        engine_stop,
        Arc::new(AtomicBool::new(false)),
        request.seed,
        |context, (problem, spec, worker_iterations)| {
            solve_ls_capped(problem, spec, context.stop(), context.seed(), config, worker_iterations, |objective, assignment, _source| {
                context.publish_latest(Improvement { objective, assignment: assignment.to_vec() });
            })
        },
        |event| {
            let now = Instant::now();
            // A coalesced engine incumbent that improves the last canonically
            // replayed objective must be checked immediately. Otherwise an
            // external stop can leave an older verified assignment behind even
            // though the final progress event already announced a better one.
            let improves_verified_objective =
                checkpoint_best.as_ref().and_then(|candidate| candidate.objectives().first()).is_some_and(|incumbent| {
                    if minimizing {
                        event.payload.objective < *incumbent
                    } else {
                        event.payload.objective > *incumbent
                    }
                });
            let replay = checkpoint_best.is_none()
                || publish_assignments
                || improves_verified_objective
                || last_checkpoint.is_none_or(|previous: Instant| now.duration_since(previous) >= INTERNAL_REPLAY_INTERVAL);
            if replay {
                last_checkpoint = Some(now);
                if let Some(candidate) = repair.candidate(
                    &event.payload.assignment,
                    request.seed.wrapping_add(event.worker as u64),
                    VerificationLevel::Transfer,
                )? {
                    checkpoint_replays = checkpoint_replays.saturating_add(1);
                    let improved = checkpoint_best.as_ref().is_none_or(|incumbent| candidate_better(&candidate, incumbent, minimizing));
                    if improved {
                        checkpoint_best = Some(candidate.clone());
                    }
                    if publish_assignments && improved && sink.emit(SolveEvent::Candidate(candidate))? == EventControl::Stop {
                        budget.cancel_with(TerminationReason::EventSink);
                        return Ok(EventControl::Stop);
                    }
                }
            }
            let control = sink.emit(SolveEvent::Progress {
                engine: EngineKind::IntegerLocalSearch,
                objectives: vec![event.payload.objective],
                elapsed: budget.elapsed(),
            })?;
            if control == EventControl::Stop {
                budget.cancel_with(TerminationReason::EventSink);
            }
            Ok(control)
        },
    )?;

    let mut best = checkpoint_best.map(promote_checkpoint_candidate);
    let mut iterations = 0u64;
    let mut moves = 0u64;
    let mut restarts = 0u64;
    let mut constraints = 0usize;
    let mut functionals = 0usize;
    let mut unsupported = 0usize;
    let mut rejected = 0usize;
    let mut last_rejection = None;

    for report in execution.reports {
        let LocalSearchOutcome {
            best: local_best,
            iterations: local_iterations,
            moves: local_moves,
            restarts: local_restarts,
            constraints: local_constraints,
            functionals: local_functionals,
            unsupported: local_unsupported,
        } = report.result;
        iterations = iterations.saturating_add(local_iterations);
        moves = moves.saturating_add(local_moves);
        restarts = restarts.saturating_add(local_restarts);
        constraints = constraints.max(local_constraints);
        functionals = functionals.max(local_functionals);
        unsupported = unsupported.max(local_unsupported);
        let Some((values, _)) = local_best else {
            continue;
        };
        match repair.candidate(&values, report.seed, VerificationLevel::Final) {
            Ok(Some(candidate)) if best.as_ref().is_none_or(|incumbent| candidate_better(&candidate, incumbent, minimizing)) => {
                best = Some(candidate);
            }
            Ok(_) => {}
            Err(error) => {
                rejected = rejected.saturating_add(1);
                last_rejection = Some(error.to_string());
            }
        }
    }

    let iteration_limit_reached = request.limits.iterations.is_some_and(|limit| iterations >= limit) && !budget.expired();

    let status = if best.is_some() { SolveStatus::Satisfiable } else { SolveStatus::Unknown };
    let mut metadata = vec![
        ("ls_moves".to_string(), moves.to_string()),
        ("ls_constraints".to_string(), constraints.to_string()),
        ("ls_functionals".to_string(), functionals.to_string()),
        ("ls_unsupported".to_string(), unsupported.to_string()),
        ("ls_rejected_incumbents".to_string(), rejected.to_string()),
        ("ls_checkpoint_replays".to_string(), checkpoint_replays.to_string()),
        ("workers".to_string(), request.threads.to_string()),
    ];
    if let Some(error) = &last_rejection {
        metadata.push(("ls_last_rejection".to_string(), error.clone()));
    }
    let message = if iteration_limit_reached {
        Some("integer local search reached the shared IterationLimit".to_string())
    } else if unsupported > 0 {
        Some(format!("integer local search declined {unsupported} model constructs"))
    } else if rejected > 0 && best.is_none() {
        Some(format!(
            "canonical replay rejected {rejected} local-search incumbents{}",
            last_rejection.as_ref().map_or(String::new(), |error| format!(": {error}"))
        ))
    } else if budget.expired() && best.is_none() {
        Some(format!("integer local search stopped: {:?}", budget.termination_reason()))
    } else {
        None
    };
    Ok(SolveResult {
        status,
        primal: best,
        bounds: Vec::new(),
        proof: None,
        reports: vec![EngineReport {
            engine: Some(EngineKind::IntegerLocalSearch),
            search: crate::search::SolveStats {
                solutions: u64::from(status == SolveStatus::Satisfiable),
                nodes: iterations,
                failures: restarts,
                ..crate::search::SolveStats::default()
            },
            elapsed: started.elapsed(),
            improvements: u64::from(status == SolveStatus::Satisfiable),
            metadata,
        }],
        message,
    })
}

fn candidate_better(candidate: &CandidateSolution, incumbent: &CandidateSolution, minimizing: bool) -> bool {
    match (candidate.objectives().first(), incumbent.objectives().first()) {
        (None, None) => false,
        (Some(candidate), Some(incumbent)) if minimizing => candidate < incumbent,
        (Some(candidate), Some(incumbent)) => candidate > incumbent,
        _ => false,
    }
}

fn promote_checkpoint_candidate(candidate: CandidateSolution) -> CandidateSolution {
    CandidateSolution::verified(
        candidate.assignment().clone(),
        candidate.objectives().to_vec(),
        candidate.source(),
        VerificationLevel::Final,
    )
}

fn worker_iteration_quota(total: u64, worker: usize, workers: usize) -> u64 {
    if total == u64::MAX {
        return u64::MAX;
    }
    let workers = u64::try_from(workers).unwrap_or(u64::MAX).max(1);
    let worker = u64::try_from(worker).unwrap_or(u64::MAX);
    total / workers + u64::from(worker < total % workers)
}

fn repair_candidate(
    context: &RepairContext<'_>,
    values: &[i32],
    seed: u64,
    verification: VerificationLevel,
) -> Result<Option<RepairedCandidate>, SolveError> {
    if context.budget.expired() && verification == VerificationLevel::Transfer {
        return Ok(None);
    }
    let stop = if verification == VerificationLevel::Final { context.budget.stop() } else { context.stop };
    if stop.load(Ordering::Acquire) {
        return Ok(None);
    }
    let problem = context.compiled.problem();
    if values.len() != problem.search.len() {
        return Err(SolveError::InvalidResult(format!(
            "integer local-search assignment has {} values, expected {}",
            values.len(),
            problem.search.len()
        )));
    }
    let mut solver = problem.solver.clone();
    for (&variable, &value) in problem.search.iter().zip(values) {
        if context.spec.is_decision(variable) && !context.spec.is_derived(variable) {
            solver
                .store
                .fix(variable, value)
                .map_err(|_| SolveError::InvalidResult("local-search decision violates its CP domain".to_string()))?;
        }
    }
    solver.enqueue_all();
    let propagation = solver.propagate_until(|| stop.load(Ordering::Acquire));
    if stop.load(Ordering::Acquire) {
        return Ok(None);
    }
    propagation.map_err(|_| SolveError::InvalidResult("local-search decisions violate the canonical CP root".to_string()))?;
    let completed = if problem.search.iter().all(|variable| solver.store.is_fixed(*variable)) {
        problem.search.iter().map(|variable| solver.store.value(*variable)).collect()
    } else {
        let conflict_limit = Some(context.request.limits.conflicts.unwrap_or(10_000).min(10_000));
        let (solution, _, complete) = crate::search::decide_sat_assuming_seeded(
            &mut solver,
            &problem.search,
            &[],
            stop,
            seed,
            None,
            conflict_limit,
            Vec::new(),
            Vec::new(),
        );
        let Some(solution) = cp_repair_completion(solution, complete, true)? else {
            return Ok(None);
        };
        solution
    };
    let physical_objective = physical_objective_value(problem, &completed)?;
    let candidate =
        super::cp::candidate_if_running_with_stop(context.model, context.compiled, &completed, context.budget, stop, verification)?.map(
            |candidate| {
                CandidateSolution::verified(
                    candidate.assignment().clone(),
                    candidate.objectives().to_vec(),
                    EngineKind::IntegerLocalSearch,
                    verification,
                )
            },
        );
    if let Some(candidate) = &candidate {
        super::cp::verify_assumptions_interruptible(candidate, &context.request.assumptions, stop)?;
    }
    Ok(candidate.map(|candidate| RepairedCandidate { candidate, physical_solution: completed, physical_objective }))
}

fn repair_semantic_candidate(
    context: &SemanticRepairContext<'_>,
    values: &[i32],
    seed: u64,
    verification: VerificationLevel,
) -> Result<Option<RepairedCandidate>, SolveError> {
    let values = values.iter().copied().map(Some).collect::<Vec<_>>();
    repair_partial_semantic_candidate(context, &values, seed, verification)
}

fn repair_partial_semantic_candidate(
    context: &SemanticRepairContext<'_>,
    values: &[Option<i32>],
    seed: u64,
    verification: VerificationLevel,
) -> Result<Option<RepairedCandidate>, SolveError> {
    let stop = if verification == VerificationLevel::Final { context.budget.stop() } else { context.stop };
    if stop.load(Ordering::Acquire) || (context.budget.expired() && verification == VerificationLevel::Transfer) {
        return Ok(None);
    }
    let map = context.compiled.int_variables();
    if values.len() != map.len() {
        return Err(SolveError::InvalidResult(format!(
            "semantic warm-start assignment has {} values, expected {}",
            values.len(),
            map.len()
        )));
    }
    let fully_specified = values.iter().all(Option::is_some);
    let problem = context.compiled.problem();
    let mut solver = problem.solver.clone();
    for (&variable, &value) in map.iter().zip(values) {
        if value.is_some_and(|value| solver.store.fix(variable, value).is_err()) {
            return Ok(None);
        }
    }
    solver.enqueue_all();
    let propagation = solver.propagate_until(|| stop.load(Ordering::Acquire));
    if stop.load(Ordering::Acquire) {
        return Ok(None);
    }
    if propagation.is_err() {
        return Ok(None);
    }
    let completed = if problem.search.iter().all(|variable| solver.store.is_fixed(*variable)) {
        problem.search.iter().map(|variable| solver.store.value(*variable)).collect()
    } else {
        let conflict_limit = Some(context.request.limits.conflicts.unwrap_or(10_000).min(10_000));
        let (solution, _, complete) = crate::search::decide_sat_assuming_seeded(
            &mut solver,
            &problem.search,
            &[],
            stop,
            seed,
            None,
            conflict_limit,
            Vec::new(),
            Vec::new(),
        );
        let Some(solution) = cp_repair_completion(solution, complete, fully_specified)? else {
            return Ok(None);
        };
        solution
    };
    let physical_objective = physical_objective_value(problem, &completed)?;
    let candidate =
        super::cp::candidate_if_running_with_stop(context.model, context.compiled, &completed, context.budget, stop, verification)?.map(
            |candidate| {
                CandidateSolution::verified(
                    candidate.assignment().clone(),
                    candidate.objectives().to_vec(),
                    EngineKind::IntegerLocalSearch,
                    verification,
                )
            },
        );
    if let Some(candidate) = &candidate {
        super::cp::verify_assumptions_interruptible(candidate, &context.request.assumptions, stop)?;
    }
    Ok(candidate.map(|candidate| RepairedCandidate { candidate, physical_solution: completed, physical_objective }))
}

fn cp_repair_completion(solution: Option<Vec<i32>>, complete: bool, fully_specified: bool) -> Result<Option<Vec<i32>>, SolveError> {
    match (solution, complete) {
        (Some(solution), _) => Ok(Some(solution)),
        (None, false) => Ok(None),
        (None, true) if fully_specified => {
            Err(SolveError::InvalidResult("local-search decisions have no completion in the canonical CP model".to_string()))
        }
        (None, true) => Ok(None),
    }
}

#[cfg(test)]
pub(super) fn audit_cp_repair_completion(complete: bool, fully_specified: bool) -> Result<bool, SolveError> {
    cp_repair_completion(None, complete, fully_specified).map(|solution| solution.is_some())
}

#[cfg(test)]
pub(super) fn audit_partial_cp_repair_search_rejection() -> Result<bool, SolveError> {
    let mut solver = crate::Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    for (x_value, y_value) in [(1, 1), (1, 0), (0, 1), (0, 0)] {
        crate::constraints::intension::intension(
            &mut solver,
            Expr::Or(vec![
                Expr::Eq(Box::new(Expr::Var(x)), Box::new(Expr::Const(x_value))),
                Expr::Eq(Box::new(Expr::Var(y)), Box::new(Expr::Const(y_value))),
            ]),
        );
    }
    solver.enqueue_all();
    solver.propagate().map_err(|_| SolveError::InvalidResult("test formula was rejected before repair search".to_string()))?;
    let stop = AtomicBool::new(false);
    let (solution, _, complete) =
        crate::search::decide_sat_assuming_seeded(&mut solver, &[x, y], &[], &stop, 0, None, None, Vec::new(), Vec::new());
    let repaired = cp_repair_completion(solution, complete, false)?;
    Ok(complete && repaired.is_none())
}

#[cfg(test)]
pub(super) fn audit_prearmed_cp_repair_interruption() -> Result<bool, SolveError> {
    let mut model = Model::new();
    model.bool_var();
    let compiling = AtomicBool::new(false);
    let compiled = CompiledCp::compile_interruptible(&model, &compiling)
        .map_err(|error| SolveError::Compile(error.reason))?
        .ok_or_else(|| SolveError::Interrupted("test CP compilation was interrupted".to_string()))?;
    let budget = SolveBudget::new(None);
    let stop = AtomicBool::new(true);
    let request = SolveRequest::default();
    let repair = SemanticRepairContext { model: &model, compiled: &compiled, request: &request, budget: &budget, stop: &stop };
    Ok(repair_partial_semantic_candidate(&repair, &[Some(0)], 0, VerificationLevel::Transfer)?.is_none())
}

fn physical_objective_value(problem: &crate::problem::Problem, values: &[i32]) -> Result<Option<i64>, SolveError> {
    let Some(objective) = problem.objective.as_ref() else {
        return Ok(None);
    };
    let by_variable = problem.search.iter().copied().zip(values.iter().copied()).collect::<HashMap<_, _>>();
    let value = |variable: VarId| by_variable.get(&variable).copied().map(i64::from);
    let objective_value = match objective {
        PhysicalObjective::Var(_, variable) | PhysicalObjective::VarWithAffine(_, variable, _, _) => value(*variable).ok_or_else(|| {
            SolveError::InvalidResult(format!("physical objective variable {} is absent from the search solution", variable.index()))
        }),
        PhysicalObjective::Linear(_, coefficients, variables) => coefficients
            .iter()
            .zip(variables)
            .try_fold(0i64, |sum, (&coefficient, &variable)| {
                let variable_value = value(variable).ok_or(())?;
                let term = coefficient.checked_mul(variable_value).ok_or(())?;
                sum.checked_add(term).ok_or(())
            })
            .map_err(|()| SolveError::InvalidResult("physical warm-start objective is incomplete or overflows i64".to_string())),
        PhysicalObjective::Expr(_, expression) => {
            let mut variables = Vec::new();
            expression.collect_vars(&mut variables);
            if let Some(variable) = variables.into_iter().find(|variable| !by_variable.contains_key(variable)) {
                return Err(SolveError::InvalidResult(format!(
                    "physical objective variable {} is absent from the search solution",
                    variable.index()
                )));
            }
            expression
                .eval(&|variable| value(variable).expect("objective variables were checked above"))
                .ok_or_else(|| SolveError::InvalidResult("physical warm-start objective expression is undefined".to_string()))
        }
    }?;
    Ok(Some(objective_value))
}
