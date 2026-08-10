//! Orchestration for parser frontends that still lower directly to a physical
//! CP store. This compatibility boundary owns search, budgets, status, clause
//! sessions, events, and canonical replay while those parsers migrate to
//! [`crate::model::ModelPackage`].

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use crate::ids::VarId;
use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::problem::{Objective as ProblemObjective, Problem};
use crate::search::{self, Assumption, SolveStats};

use super::{
    merge_search_stats, Assignment, Bound, CandidateSolution, EngineKind, EngineReport, EventControl, EventSink, ProofClaim,
    ProvenConclusion, SolveBudget, SolveError, SolveEvent, SolveMode, SolveRequest, SolveResult, SolveStatus, TerminationReason,
    VerificationLevel,
};

#[derive(Clone)]
pub(crate) struct PhysicalObjectiveTier {
    pub objective: ProblemObjective,
}

/// Fully lowered exact CP input used only by compatibility frontends.
pub(crate) struct PhysicalSolveInput {
    pub problem: Problem,
    /// Size of the frontend-visible integer arena. Search auxiliaries outside
    /// this prefix are verified but do not escape in the typed assignment.
    pub visible_variables: usize,
    pub objectives: Vec<PhysicalObjectiveTier>,
    pub assumptions: Vec<Assumption>,
    pub hints: Vec<(VarId, i32)>,
    pub primary_branch_scope: Option<Vec<VarId>>,
    pub branch_order: Vec<VarId>,
    pub shared_clauses: Option<Arc<SharedClausePool>>,
    pub first_worker: usize,
}

pub(crate) struct PhysicalSolveOutput {
    pub result: SolveResult,
    pub next_worker: usize,
}

/// Execute an exact physical model under the same budget and result contract
/// as semantic plans. Lexicographic tiers are solved sequentially and fixed by
/// assumptions before the next tier starts.
pub(crate) fn solve_physical_exact_with_budget(
    input: PhysicalSolveInput,
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<PhysicalSolveOutput, SolveError> {
    request.validate()?;
    if request.mode == SolveMode::LocalSearch {
        return Err(SolveError::InvalidRequest("physical exact orchestration does not accept local-search mode".to_string()));
    }
    if request.threads != 1 {
        return Err(SolveError::InvalidRequest("physical lexicographic orchestration currently uses one exact worker".to_string()));
    }
    execute_physical_exact(input, request, budget, sink)
}

fn execute_physical_exact(
    input: PhysicalSolveInput,
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<PhysicalSolveOutput, SolveError> {
    validate_input(&input)?;
    if budget.expired() {
        return Ok(PhysicalSolveOutput { result: SolveResult::unknown(), next_worker: input.first_worker });
    }

    let search_stop = budget.search_stop();
    let engine_stop = search_stop.flag();
    let started = Instant::now();
    let mut worker = input.first_worker;
    let mut remaining_conflicts = request.limits.conflicts;
    let phase = initial_phase(&input)?;
    let mut total_stats = SolveStats::default();
    let active_assumptions = input.assumptions.clone();
    let mut root_solver = input.problem.solver.clone();
    let mut best_assignment: Option<Vec<i32>> = None;
    let mut objective_values = Vec::with_capacity(input.objectives.len());
    let mut all_complete = true;
    let mut event_error = None;
    let mut conflict_limit_reached = false;

    if input.objectives.is_empty() {
        let mut solver = root_solver.clone();
        let sharing = next_sharing(input.shared_clauses.as_ref(), &mut worker);
        let (assignment, stats, complete) = search::decide_sat_assuming_seeded_with_scope(
            &mut solver,
            &input.problem.search,
            input.primary_branch_scope.as_deref(),
            &active_assumptions,
            engine_stop,
            request.seed,
            sharing,
            remaining_conflicts,
            phase,
            input.branch_order.clone(),
        );
        if let Some(limit) = remaining_conflicts.as_mut() {
            *limit = limit.saturating_sub(stats.failures);
        }
        total_stats = stats;
        let primal = assignment
            .as_deref()
            .map(|values| {
                super::verify_final_with_budget(budget, |stop| {
                    replay_candidate(&input, values, &active_assumptions, &[], VerificationLevel::Final, stop)
                })
            })
            .transpose()?;
        let status = if primal.is_some() {
            SolveStatus::Satisfiable
        } else if complete {
            SolveStatus::Unsatisfiable
        } else {
            SolveStatus::Unknown
        };
        let conflict_limit_reached = remaining_conflicts == Some(0) && !complete && !budget.expired();
        let proof = (status == SolveStatus::Unsatisfiable)
            .then(|| ProofClaim::complete_search(EngineKind::IntegerExact, ProvenConclusion::Unsatisfiable, 0));
        let message = (!complete).then(|| {
            let reason = if conflict_limit_reached { TerminationReason::ConflictLimit } else { budget.termination_reason() };
            format!("physical CP decision search stopped: {reason:?}")
        });
        let result = SolveResult {
            status,
            primal,
            bounds: Vec::new(),
            proof,
            reports: vec![physical_report(total_stats, started.elapsed(), 0, &input, worker)],
            message,
        };
        result.validate_contract()?;
        return Ok(PhysicalSolveOutput { result, next_worker: worker });
    }

    for (tier_index, objective) in input.objectives.iter().enumerate() {
        if remaining_conflicts == Some(0) || engine_stop.load(std::sync::atomic::Ordering::Acquire) {
            conflict_limit_reached = remaining_conflicts == Some(0) && !budget.expired();
            all_complete = false;
            break;
        }
        let mut solver = root_solver.clone();
        let sharing = next_sharing(input.shared_clauses.as_ref(), &mut worker);
        let prior_values = objective_values.clone();
        let mut tier_error = None;
        let (best, stats, complete) = search::optimize_assuming_seeded_with_scope(
            &mut solver,
            &input.problem.search,
            input.primary_branch_scope.as_deref(),
            &active_assumptions,
            objective.objective.search(),
            objective.objective.minimizing(),
            engine_stop,
            request.seed.wrapping_add(tier_index as u64),
            None,
            sharing,
            remaining_conflicts,
            phase.clone(),
            input.branch_order.clone(),
            |value, assignment| {
                if tier_error.is_some() {
                    return;
                }
                let mut objectives = prior_values.clone();
                objectives.push(value);
                if request.publish_incumbent_assignments {
                    match replay_candidate(&input, assignment, &active_assumptions, &objectives, VerificationLevel::Transfer, budget.stop())
                        .and_then(|candidate| emit_intermediate(sink, budget, SolveEvent::Candidate(candidate)))
                    {
                        Ok(true) => {}
                        Ok(false) => return,
                        Err(error) => {
                            tier_error = Some(error);
                            budget.cancel_with(TerminationReason::Engine);
                        }
                    }
                }
                if tier_error.is_none() {
                    match emit_intermediate(
                        sink,
                        budget,
                        SolveEvent::Progress { engine: EngineKind::IntegerExact, objectives, elapsed: budget.elapsed() },
                    ) {
                        Ok(true) => {}
                        Ok(false) => {}
                        Err(error) => {
                            tier_error = Some(error);
                            budget.cancel_with(TerminationReason::Engine);
                        }
                    }
                }
            },
        );
        if let Some(error) = tier_error {
            event_error = Some(error);
            break;
        }
        if let Some(limit) = remaining_conflicts.as_mut() {
            *limit = limit.saturating_sub(stats.failures);
        }
        merge_search_stats(&mut total_stats, stats);
        let Some((assignment, value)) = best else {
            if !complete {
                conflict_limit_reached = remaining_conflicts == Some(0) && !budget.expired();
            }
            all_complete = tier_index == 0 && complete;
            break;
        };
        best_assignment = Some(assignment);
        objective_values.push(value);
        post_objective_equality(&mut root_solver, &objective.objective, value)?;
        if !complete {
            conflict_limit_reached = remaining_conflicts == Some(0) && !budget.expired();
            all_complete = false;
            break;
        }
    }
    if let Some(error) = event_error {
        return Err(error);
    }

    let primal = best_assignment
        .as_deref()
        .map(|values| {
            super::verify_final_with_budget(budget, |stop| {
                replay_candidate(&input, values, &active_assumptions, &objective_values, VerificationLevel::Final, stop)
            })
        })
        .transpose()?;
    let exhausted_without_primal = primal.is_none() && objective_values.is_empty() && all_complete;
    let solved_all_tiers = primal.is_some() && all_complete && objective_values.len() == input.objectives.len();
    let status = if solved_all_tiers {
        SolveStatus::Optimal
    } else if primal.is_some() {
        SolveStatus::Satisfiable
    } else if exhausted_without_primal {
        SolveStatus::Unsatisfiable
    } else {
        SolveStatus::Unknown
    };
    let proof = match status {
        SolveStatus::Optimal => {
            Some(ProofClaim::complete_search(EngineKind::IntegerExact, ProvenConclusion::Optimal, input.objectives.len()))
        }
        SolveStatus::Unsatisfiable => {
            Some(ProofClaim::complete_search(EngineKind::IntegerExact, ProvenConclusion::Unsatisfiable, input.objectives.len()))
        }
        SolveStatus::Satisfiable | SolveStatus::Unknown | SolveStatus::Unsupported => None,
    };
    let bounds = if solved_all_tiers {
        objective_values
            .iter()
            .enumerate()
            .map(|(tier, &value)| Bound { tier, value, method: "complete lexicographic CP search".to_string() })
            .collect()
    } else {
        Vec::new()
    };
    let improvements = total_stats.solutions;
    let message = (!all_complete).then(|| {
        let reason = if conflict_limit_reached { TerminationReason::ConflictLimit } else { budget.termination_reason() };
        format!("physical lexicographic CP search stopped: {reason:?}")
    });
    let result = SolveResult {
        status,
        primal,
        bounds,
        proof,
        reports: vec![physical_report(total_stats, started.elapsed(), improvements, &input, worker)],
        message,
    };
    result.validate_contract()?;
    Ok(PhysicalSolveOutput { result, next_worker: worker })
}

fn validate_input(input: &PhysicalSolveInput) -> Result<(), SolveError> {
    let variables = input.problem.solver.store.num_vars();
    if input.visible_variables > variables {
        return Err(SolveError::InvalidRequest("visible physical variable prefix exceeds the solver variable arena".to_string()));
    }
    for variable in input
        .problem
        .search
        .iter()
        .chain(input.assumptions.iter().map(|assumption| &assumption.var))
        .chain(input.hints.iter().map(|(variable, _)| variable))
        .chain(input.primary_branch_scope.iter().flatten())
        .chain(input.branch_order.iter())
    {
        if variable.index() >= variables {
            return Err(SolveError::InvalidRequest(format!("physical request references unknown variable {}", variable.index())));
        }
    }
    if let Some(scope) = &input.primary_branch_scope {
        let search = input.problem.search.iter().copied().collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        for &variable in scope {
            if !search.contains(&variable) {
                return Err(SolveError::InvalidRequest(format!(
                    "primary physical branch scope references non-search variable {}",
                    variable.index()
                )));
            }
            if !seen.insert(variable) {
                return Err(SolveError::InvalidRequest(format!(
                    "physical variable {} appears twice in primary branch scope",
                    variable.index()
                )));
            }
        }
        if let Some(variable) = input.branch_order.iter().find(|variable| !seen.contains(variable)) {
            return Err(SolveError::InvalidRequest(format!(
                "physical branch-order variable {} is outside the primary branch scope",
                variable.index()
            )));
        }
    }
    for objective in &input.objectives {
        for variable in objective_variables(&objective.objective) {
            if variable.index() >= variables {
                return Err(SolveError::InvalidRequest(format!("physical objective references unknown variable {}", variable.index())));
            }
        }
    }
    Ok(())
}

fn initial_phase(input: &PhysicalSolveInput) -> Result<Vec<Option<i32>>, SolveError> {
    let mut phase = vec![None; input.problem.solver.store.num_vars()];
    for &(variable, value) in &input.hints {
        if phase[variable.index()].is_some() {
            return Err(SolveError::InvalidRequest(format!("physical variable {} has more than one hint", variable.index())));
        }
        if !input.problem.solver.store.contains(variable, value) {
            return Err(SolveError::InvalidRequest(format!("hint value {value} is outside variable {}'s domain", variable.index())));
        }
        phase[variable.index()] = Some(value);
    }
    Ok(phase)
}

fn next_sharing(pool: Option<&Arc<SharedClausePool>>, worker: &mut usize) -> Option<ClauseSharing> {
    pool.map(|pool| {
        let sharing = ClauseSharing::new(Arc::clone(pool), *worker);
        *worker = worker.saturating_add(1);
        sharing
    })
}

fn replay_candidate(
    input: &PhysicalSolveInput,
    values: &[i32],
    assumptions: &[Assumption],
    claimed_objectives: &[i64],
    verification: VerificationLevel,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<CandidateSolution, SolveError> {
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return Err(SolveError::Interrupted("physical assignment replay was interrupted".to_string()));
    }
    if values.len() != input.problem.search.len() {
        return Err(SolveError::InvalidResult(format!(
            "physical assignment has {} values, expected {}",
            values.len(),
            input.problem.search.len()
        )));
    }
    let mut solver = input.problem.solver.clone();
    solver.enqueue_all();
    for (index, (&variable, &value)) in input.problem.search.iter().zip(values).enumerate() {
        if index & 0xff == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SolveError::Interrupted("physical assignment replay was interrupted".to_string()));
        }
        solver
            .store
            .fix(variable, value)
            .map_err(|_| SolveError::InvalidResult("physical assignment violates a variable domain".to_string()))?;
    }
    solver
        .propagate_until(|| stop.load(std::sync::atomic::Ordering::Acquire))
        .map_err(|_| SolveError::InvalidResult("physical assignment violates a posted constraint".to_string()))?;
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return Err(SolveError::Interrupted("physical assignment propagation was interrupted".to_string()));
    }
    for (index, assumption) in assumptions.iter().enumerate() {
        if index & 0xff == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SolveError::Interrupted("physical assumption replay was interrupted".to_string()));
        }
        let value = solver.store.value(assumption.var);
        if !assumption_holds(value, assumption) {
            return Err(SolveError::InvalidResult(format!(
                "physical assignment violates assumption on variable {}",
                assumption.var.index()
            )));
        }
    }
    let mut actual = Vec::with_capacity(input.objectives.len());
    for (index, objective) in input.objectives.iter().enumerate() {
        if index & 0x3f == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SolveError::Interrupted("physical objective replay was interrupted".to_string()));
        }
        actual.push(objective_value(&objective.objective, &solver)?);
    }
    if actual.get(..claimed_objectives.len()) != Some(claimed_objectives) {
        return Err(SolveError::InvalidResult(format!(
            "physical objective mismatch: engine reported {claimed_objectives:?}, replay produced {actual:?}"
        )));
    }
    let mut by_variable = std::collections::BTreeMap::new();
    for (index, (&variable, &value)) in input.problem.search.iter().zip(values).enumerate() {
        if index & 0xff == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SolveError::Interrupted("physical assignment projection was interrupted".to_string()));
        }
        by_variable.insert(variable, value);
    }
    let mut integers = Vec::with_capacity(input.visible_variables);
    for index in 0..input.visible_variables {
        if index & 0xff == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SolveError::Interrupted("physical assignment projection was interrupted".to_string()));
        }
        integers.push(by_variable.get(&VarId(index as u32)).copied().map(i64::from));
    }
    Ok(CandidateSolution::verified(
        Assignment { integers, sets: Vec::new(), lists: Vec::new(), intervals: Vec::new() },
        actual,
        EngineKind::IntegerExact,
        verification,
    ))
}

fn objective_variables(objective: &ProblemObjective) -> Vec<VarId> {
    match objective {
        ProblemObjective::Var(_, variable) => vec![*variable],
        ProblemObjective::Linear(_, _, variables) => variables.clone(),
        ProblemObjective::Expr(_, expression) => {
            let mut variables = Vec::new();
            expression.collect_vars(&mut variables);
            variables
        }
    }
}

fn objective_value(objective: &ProblemObjective, solver: &crate::Solver) -> Result<i64, SolveError> {
    match objective {
        ProblemObjective::Var(_, variable) => Ok(i64::from(solver.store.value(*variable))),
        ProblemObjective::Linear(_, coefficients, variables) => coefficients
            .iter()
            .zip(variables)
            .try_fold(0i64, |sum, (&coefficient, &variable)| {
                sum.checked_add(coefficient.saturating_mul(i64::from(solver.store.value(variable))))
            })
            .ok_or_else(|| SolveError::InvalidResult("physical objective value overflowed i64".to_string())),
        ProblemObjective::Expr(_, expression) => expression
            .eval(&|variable| i64::from(solver.store.value(variable)))
            .ok_or_else(|| SolveError::InvalidResult("physical objective expression is undefined".to_string())),
    }
}

fn post_objective_equality(solver: &mut crate::Solver, objective: &ProblemObjective, value: i64) -> Result<(), SolveError> {
    let expression = match objective {
        ProblemObjective::Var(_, variable) => crate::expr::Expr::Var(*variable),
        ProblemObjective::Linear(_, coefficients, variables) => crate::expr::Expr::Add(
            coefficients
                .iter()
                .zip(variables)
                .map(|(&coefficient, &variable)| {
                    crate::expr::Expr::Mul(vec![crate::expr::Expr::Const(coefficient), crate::expr::Expr::Var(variable)])
                })
                .collect(),
        ),
        ProblemObjective::Expr(_, expression) => expression.clone(),
    };
    crate::constraints::intension::intension(
        solver,
        crate::expr::Expr::Eq(Box::new(expression), Box::new(crate::expr::Expr::Const(value))),
    );
    Ok(())
}

fn assumption_holds(value: i32, assumption: &Assumption) -> bool {
    use search::AssumptionOp;
    match assumption.op {
        AssumptionOp::Eq => value == assumption.value,
        AssumptionOp::Ne => value != assumption.value,
        AssumptionOp::Le => value <= assumption.value,
        AssumptionOp::Lt => value < assumption.value,
        AssumptionOp::Ge => value >= assumption.value,
        AssumptionOp::Gt => value > assumption.value,
    }
}

fn physical_report(
    search: SolveStats,
    elapsed: std::time::Duration,
    improvements: u64,
    input: &PhysicalSolveInput,
    next_worker: usize,
) -> EngineReport {
    EngineReport {
        engine: Some(EngineKind::IntegerExact),
        search,
        elapsed,
        improvements,
        metadata: vec![
            ("lexicographic_tiers".to_string(), input.objectives.len().to_string()),
            ("clause_session_workers".to_string(), next_worker.saturating_sub(input.first_worker).to_string()),
        ],
    }
}

fn emit_intermediate(sink: &mut dyn EventSink, budget: &SolveBudget, event: SolveEvent) -> Result<bool, SolveError> {
    if sink.emit(event)? == EventControl::Stop {
        budget.cancel_with(TerminationReason::EventSink);
        return Ok(false);
    }
    Ok(true)
}
