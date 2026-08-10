//! Semantic integer local-search compilation and execution.
//!
//! The frontend-neutral model is lowered once to the same physical CP root as
//! exact search. The local-search scorer is a specialized view of that root;
//! every assignment crossing the worker boundary is repaired on the CP root
//! and replayed against the semantic model before publication.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::constraints::linear::Relation as PhysicalRelation;
use crate::constraints::table::{Dfa, Mdd, MddArc, STAR};
use crate::engines::ls::cop::{solve_ls_capped, LocalRhs, LocalSearchOutcome, LocalSearchSpec, LsConfig};
use crate::expr::Expr;
use crate::ids::VarId;
use crate::model::{
    BoolLiteral, CompiledCp, Constraint, IntDomain, IntExpr as SemanticIntExpr, IntGlobalConstraint, IntVarRef, Model,
    Objective as SemanticObjective, Relation, SetVarRef,
};
use crate::problem::Objective as PhysicalObjective;

use super::{
    execute_workers, CandidateSolution, EngineKind, EngineReport, EventControl, EventSink, SolveBudget, SolveError, SolveEvent,
    SolveRequest, SolveResult, SolveStatus, TerminationReason, VerificationLevel,
};

#[derive(Clone)]
pub(crate) struct IntegerLocalSearchPlan {
    spec: LocalSearchSpec,
    guarded_warm_start: bool,
}

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
const GUARDED_WARM_START_ITERATIONS: u64 = 160;

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
}

impl RepairContext<'_> {
    fn candidate(&self, values: &[i32], seed: u64, verification: VerificationLevel) -> Result<Option<CandidateSolution>, SolveError> {
        Ok(repair_candidate(self, values, seed, verification)?.map(|repaired| repaired.candidate))
    }
}

pub(crate) fn may_support_guarded_warm_start(model: &Model) -> bool {
    semantic_guarded_warm_start(model)
}

impl IntegerLocalSearchPlan {
    pub(crate) fn supports_guarded_warm_start(&self) -> bool {
        self.guarded_warm_start
    }
}

pub(crate) fn compile(model: &Model, compiled: &CompiledCp) -> Result<IntegerLocalSearchPlan, SolveError> {
    if compiled.objectives().len() > 1 {
        return Err(SolveError::InvalidRequest("integer local search currently supports at most one objective tier".to_string()));
    }

    let mut spec = LocalSearchSpec::default();
    for &variable in compiled.int_variables() {
        spec.add_var(variable);
    }
    for set in compiled.sets() {
        for &membership in &set.membership {
            spec.add_var(membership);
        }
    }
    for constraint in model.constraints() {
        compile_constraint(&mut spec, model, compiled, constraint)?;
    }
    if spec.unsupported() > 0 {
        return Err(SolveError::Unsupported(format!(
            "integer local-search compilation rejected {} unsupported model construct(s)",
            spec.unsupported()
        )));
    }
    let guarded_warm_start = semantic_guarded_warm_start(model) && spec.has_guarded_word_structure(compiled.problem());
    Ok(IntegerLocalSearchPlan { spec, guarded_warm_start })
}

pub(crate) fn warm_start(
    model: &Model,
    compiled: &CompiledCp,
    plan: &IntegerLocalSearchPlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    engine_stop: &AtomicBool,
) -> Result<Option<IntegerWarmStart>, SolveError> {
    if !plan.supports_guarded_warm_start() || budget.expired() {
        return Ok(None);
    }

    let started = Instant::now();
    let problem = compiled.problem().clone();
    let outcome = solve_ls_capped(
        problem,
        plan.spec.clone(),
        engine_stop,
        request.seed,
        LsConfig { gls: true, min_conflicts: true, kick_bandit: false },
        GUARDED_WARM_START_ITERATIONS,
        |_, _, _| {},
    );
    let LocalSearchOutcome { best, iterations, moves, restarts, constraints, functionals, unsupported } = outcome;
    let Some((values, local_objective)) = best else {
        return Ok(None);
    };

    let repair = RepairContext { model, compiled, spec: &plan.spec, request, budget };
    let Some(repaired) = repair_candidate(&repair, &values, request.seed, VerificationLevel::Transfer)? else {
        return Ok(None);
    };
    let physical_objective = repaired
        .physical_objective
        .ok_or_else(|| SolveError::InvalidResult("guarded integer warm start has no physical objective".to_string()))?;
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
                ("ls_role".to_string(), "guarded_warm_start".to_string()),
                ("ls_moves".to_string(), moves.to_string()),
                ("ls_constraints".to_string(), constraints.to_string()),
                ("ls_functionals".to_string(), functionals.to_string()),
                ("ls_unsupported".to_string(), unsupported.to_string()),
            ],
        },
    }))
}

fn semantic_guarded_warm_start(model: &Model) -> bool {
    let [SemanticObjective::IntExpr { minimize, expr: objective }] = model.objectives() else {
        return false;
    };
    if semantic_direct_guarded_warm_start(model, objective, *minimize) {
        return true;
    }
    if !model.constraints().iter().any(|constraint| {
        matches!(
            constraint,
            Constraint::IntegerGlobal(IntGlobalConstraint::Table { positive: true, .. })
                | Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { .. })
        )
    }) {
        return false;
    }

    model.constraints().iter().any(|constraint| {
        let Constraint::Intension(expression) = constraint else {
            return false;
        };
        let Some((guard, counter)) = semantic_guarded_mismatch_bound(expression) else {
            return false;
        };
        model.int_vars().get(guard.0).is_some_and(semantic_binary_domain)
            && semantic_objective_rewards_guard(model, objective, *minimize, guard)
            && semantic_counter_uses_common_element_array(model, counter)
    })
}

fn semantic_direct_guarded_warm_start(model: &Model, objective: &SemanticIntExpr, minimizing: bool) -> bool {
    let mut requirements: HashMap<IntVarRef, Vec<(IntVarRef, i32)>> = HashMap::new();
    for constraint in model.constraints() {
        let guarded_value = match constraint {
            Constraint::Intension(expression) => semantic_direct_guarded_value(expression),
            Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, tuples, positive: true }) => {
                semantic_direct_guarded_table_value(variables, tuples)
            }
            _ => None,
        };
        if let Some((guard, target, value)) = guarded_value {
            requirements.entry(guard).or_default().push((target, value));
        }
    }

    requirements.into_iter().any(|(guard, mut values)| {
        values.sort_unstable();
        values.dedup();
        let has_conflicting_value = values.windows(2).any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1);
        !values.is_empty()
            && !has_conflicting_value
            && values.iter().all(|&(target, _)| target != guard)
            && model.int_vars().get(guard.0).is_some_and(semantic_binary_domain)
            && semantic_objective_rewards_guard(model, objective, minimizing, guard)
            && semantic_guarded_elements_have_positive_construction(model, &values)
    })
}

fn semantic_binary_domain(domain: &IntDomain) -> bool {
    match domain {
        IntDomain::Bool | IntDomain::Range { lo: 0, hi: 1 } => true,
        IntDomain::Set(values) => values.len() == 2 && values.contains(&0) && values.contains(&1),
        IntDomain::Range { .. } => false,
    }
}

fn semantic_direct_guarded_value(expression: &SemanticIntExpr) -> Option<(IntVarRef, IntVarRef, i32)> {
    let SemanticIntExpr::Or(parts) = expression else {
        return None;
    };
    let [left, right] = parts.as_slice() else {
        return None;
    };
    semantic_direct_guarded_parts(left, right).or_else(|| semantic_direct_guarded_parts(right, left))
}

fn semantic_direct_guarded_parts(guard: &SemanticIntExpr, target: &SemanticIntExpr) -> Option<(IntVarRef, IntVarRef, i32)> {
    let guard = semantic_zero_guard(guard)?;
    let (target, value) = semantic_eq_variable_constant(target)?;
    Some((guard, target, value))
}

fn semantic_eq_variable_constant(expression: &SemanticIntExpr) -> Option<(IntVarRef, i32)> {
    let SemanticIntExpr::Eq(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), SemanticIntExpr::Constant(value))
        | (SemanticIntExpr::Constant(value), SemanticIntExpr::Variable(variable)) => Some((*variable, i32::try_from(*value).ok()?)),
        _ => None,
    }
}

fn semantic_direct_guarded_table_value(variables: &[IntVarRef], tuples: &[Vec<i32>]) -> Option<(IntVarRef, IntVarRef, i32)> {
    let [guard, target] = variables else {
        return None;
    };
    let mut inactive_free = false;
    let mut active_value = None;
    for tuple in tuples {
        let [guard_value, target_value] = tuple.as_slice() else {
            return None;
        };
        match (*guard_value, *target_value) {
            (0, STAR) => inactive_free = true,
            (1, value) if value != STAR && active_value.is_none_or(|active| active == value) => active_value = Some(value),
            _ => return None,
        }
    }
    inactive_free.then_some((*guard, *target, active_value?))
}

fn semantic_guarded_elements_have_positive_construction(model: &Model, requirements: &[(IntVarRef, i32)]) -> bool {
    let first_target = requirements[0].0;
    let mut candidate_arrays: Vec<&[IntVarRef]> = Vec::new();
    for constraint in model.constraints() {
        let Constraint::IntegerGlobal(IntGlobalConstraint::Element { array, value, .. }) = constraint else {
            continue;
        };
        if *value == first_target && !candidate_arrays.contains(&array.as_slice()) {
            candidate_arrays.push(array);
        }
    }

    candidate_arrays.into_iter().any(|array| {
        let mut indexes = Vec::with_capacity(requirements.len());
        for &(target, _) in requirements {
            let mut matching_indexes = model
                .constraints()
                .iter()
                .filter_map(|constraint| {
                    let Constraint::IntegerGlobal(IntGlobalConstraint::Element { array: candidate_array, index, value }) = constraint
                    else {
                        return None;
                    };
                    (*value == target && candidate_array.as_slice() == array).then_some(*index)
                })
                .collect::<Vec<_>>();
            matching_indexes.sort_unstable();
            matching_indexes.dedup();
            let [index] = matching_indexes.as_slice() else {
                return false;
            };
            indexes.push(*index);
        }
        let distinct_indexes = indexes.iter().copied().collect::<HashSet<_>>();
        distinct_indexes.len() == indexes.len() && semantic_positive_tables_connect_indexes(model, &indexes)
    })
}

fn semantic_positive_tables_connect_indexes(model: &Model, indexes: &[IntVarRef]) -> bool {
    if indexes.is_empty() {
        return false;
    }
    let positions = indexes.iter().enumerate().map(|(position, &index)| (index, position)).collect::<HashMap<_, _>>();
    let mut adjacency = vec![Vec::new(); indexes.len()];
    let mut covered = vec![false; indexes.len()];
    for constraint in model.constraints() {
        let Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, tuples, positive: true }) = constraint else {
            continue;
        };
        if tuples.is_empty() {
            continue;
        }
        let mut table_positions = variables
            .iter()
            .enumerate()
            .filter_map(|(column, variable)| {
                positions
                    .get(variable)
                    .copied()
                    .filter(|_| tuples.iter().any(|tuple| tuple.get(column).is_some_and(|&value| value != STAR)))
            })
            .collect::<Vec<_>>();
        table_positions.sort_unstable();
        table_positions.dedup();
        for &position in &table_positions {
            covered[position] = true;
        }
        for &left in &table_positions {
            for &right in &table_positions {
                if left != right {
                    adjacency[left].push(right);
                }
            }
        }
    }
    if !covered.iter().all(|&value| value) {
        return false;
    }

    let mut reached = vec![false; indexes.len()];
    let mut pending = vec![0usize];
    reached[0] = true;
    while let Some(position) = pending.pop() {
        for &next in &adjacency[position] {
            if !reached[next] {
                reached[next] = true;
                pending.push(next);
            }
        }
    }
    reached.iter().all(|&value| value)
}

fn semantic_guarded_mismatch_bound(expression: &SemanticIntExpr) -> Option<(IntVarRef, IntVarRef)> {
    let SemanticIntExpr::Or(parts) = expression else {
        return None;
    };
    let [left, right] = parts.as_slice() else {
        return None;
    };
    semantic_zero_guard(left)
        .zip(semantic_nonnegative_bound(right))
        .or_else(|| semantic_zero_guard(right).zip(semantic_nonnegative_bound(left)))
}

fn semantic_zero_guard(expression: &SemanticIntExpr) -> Option<IntVarRef> {
    let SemanticIntExpr::Eq(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), SemanticIntExpr::Constant(0))
        | (SemanticIntExpr::Constant(0), SemanticIntExpr::Variable(variable)) => Some(*variable),
        _ => None,
    }
}

fn semantic_nonnegative_bound(expression: &SemanticIntExpr) -> Option<IntVarRef> {
    let SemanticIntExpr::Le(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), SemanticIntExpr::Constant(value)) if *value >= 0 => Some(*variable),
        _ => None,
    }
}

fn semantic_objective_rewards_guard(model: &Model, objective: &SemanticIntExpr, minimizing: bool, guard: IntVarRef) -> bool {
    let direct = semantic_affine_coefficient(objective, guard).filter(|&coefficient| coefficient != 0);
    let coefficient = direct.or_else(|| {
        let SemanticIntExpr::Variable(objective_variable) = objective else {
            return None;
        };
        semantic_materialized_objective_coefficient(model, *objective_variable, guard)
    });
    coefficient.is_some_and(|coefficient| if minimizing { coefficient < 0 } else { coefficient > 0 })
}

fn semantic_materialized_objective_coefficient(model: &Model, objective: IntVarRef, guard: IntVarRef) -> Option<i128> {
    let mut candidates = model.constraints().iter().filter_map(|constraint| {
        let Constraint::Linear { terms, relation: Relation::Eq, .. } = constraint else {
            return None;
        };
        let mut objective_coefficient = 0i128;
        let mut guard_coefficient = 0i128;
        for &(coefficient, variable) in terms {
            if variable == objective {
                objective_coefficient = objective_coefficient.checked_add(i128::from(coefficient))?;
            }
            if variable == guard {
                guard_coefficient = guard_coefficient.checked_add(i128::from(coefficient))?;
            }
        }
        (objective_coefficient.abs() == 1 && guard_coefficient != 0).then(|| -guard_coefficient / objective_coefficient)
    });
    let coefficient = candidates.next()?;
    candidates.all(|candidate| candidate == coefficient).then_some(coefficient)
}

fn semantic_affine_coefficient(expression: &SemanticIntExpr, target: IntVarRef) -> Option<i128> {
    match expression {
        SemanticIntExpr::Constant(_) => Some(0),
        SemanticIntExpr::Variable(variable) => Some(i128::from(*variable == target)),
        SemanticIntExpr::Neg(value) => semantic_affine_coefficient(value, target)?.checked_neg(),
        SemanticIntExpr::Add(values) => {
            values.iter().try_fold(0i128, |sum, value| sum.checked_add(semantic_affine_coefficient(value, target)?))
        }
        SemanticIntExpr::Sub(left, right) => {
            semantic_affine_coefficient(left, target)?.checked_sub(semantic_affine_coefficient(right, target)?)
        }
        SemanticIntExpr::Mul(values) => {
            let mut scale = 1i128;
            let mut nonconstant = None;
            for value in values {
                if let Some(constant) = semantic_constant_value(value) {
                    scale = scale.checked_mul(constant)?;
                } else if nonconstant.replace(value).is_some() {
                    return None;
                }
            }
            nonconstant.map_or(Some(0), |value| semantic_affine_coefficient(value, target)?.checked_mul(scale))
        }
        _ => None,
    }
}

fn semantic_constant_value(expression: &SemanticIntExpr) -> Option<i128> {
    match expression {
        SemanticIntExpr::Constant(value) => Some(i128::from(*value)),
        SemanticIntExpr::Neg(value) => semantic_constant_value(value)?.checked_neg(),
        SemanticIntExpr::Add(values) => values.iter().try_fold(0i128, |sum, value| sum.checked_add(semantic_constant_value(value)?)),
        SemanticIntExpr::Sub(left, right) => semantic_constant_value(left)?.checked_sub(semantic_constant_value(right)?),
        SemanticIntExpr::Mul(values) => {
            values.iter().try_fold(1i128, |product, value| product.checked_mul(semantic_constant_value(value)?))
        }
        _ => None,
    }
}

fn semantic_counter_uses_common_element_array(model: &Model, counter: IntVarRef) -> bool {
    model.constraints().iter().any(|constraint| {
        let Constraint::Linear { terms, relation: Relation::Eq, rhs: 0 } = constraint else {
            return false;
        };
        let mut saw_counter = false;
        let mut mismatches = Vec::new();
        for &(coefficient, variable) in terms {
            if variable == counter {
                if saw_counter || coefficient != -1 {
                    return false;
                }
                saw_counter = true;
            } else if coefficient == 1 {
                mismatches.push(variable);
            } else if coefficient != 0 {
                return false;
            }
        }
        saw_counter && !mismatches.is_empty() && semantic_mismatches_share_element_array(model, &mismatches)
    })
}

fn semantic_mismatches_share_element_array(model: &Model, mismatches: &[IntVarRef]) -> bool {
    let mut common_arrays: Option<Vec<Vec<IntVarRef>>> = None;
    for &mismatch in mismatches {
        let sources = model
            .constraints()
            .iter()
            .filter_map(|constraint| {
                let Constraint::Intension(expression) = constraint else {
                    return None;
                };
                semantic_mismatch_source(expression, mismatch)
            })
            .collect::<Vec<_>>();
        if sources.is_empty() {
            return false;
        }
        let mut arrays = model
            .constraints()
            .iter()
            .filter_map(|constraint| {
                let Constraint::IntegerGlobal(IntGlobalConstraint::Element { array, value, .. }) = constraint else {
                    return None;
                };
                sources.contains(value).then(|| array.clone())
            })
            .collect::<Vec<_>>();
        arrays.sort_unstable();
        arrays.dedup();
        if arrays.is_empty() {
            return false;
        }
        match &mut common_arrays {
            None => common_arrays = Some(arrays),
            Some(common) => common.retain(|array| arrays.binary_search(array).is_ok()),
        }
        if common_arrays.as_ref().is_none_or(Vec::is_empty) {
            return false;
        }
    }
    common_arrays.is_some_and(|arrays| !arrays.is_empty())
}

fn semantic_mismatch_source(expression: &SemanticIntExpr, mismatch: IntVarRef) -> Option<IntVarRef> {
    let SemanticIntExpr::Eq(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), value) if *variable == mismatch => semantic_ne_variable_constant(value),
        (value, SemanticIntExpr::Variable(variable)) if *variable == mismatch => semantic_ne_variable_constant(value),
        _ => None,
    }
}

fn semantic_ne_variable_constant(expression: &SemanticIntExpr) -> Option<IntVarRef> {
    let SemanticIntExpr::Ne(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), SemanticIntExpr::Constant(_))
        | (SemanticIntExpr::Constant(_), SemanticIntExpr::Variable(variable)) => Some(*variable),
        _ => None,
    }
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
    let repair = RepairContext { model, compiled, spec: &plan.spec, request, budget };
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
    let stop = context.budget.stop();
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
    solver.propagate().map_err(|_| SolveError::InvalidResult("local-search decisions violate the canonical CP root".to_string()))?;
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
        let Some(solution) = cp_repair_completion(solution, complete)? else {
            return Ok(None);
        };
        solution
    };
    let physical_objective = physical_objective_value(problem, &completed)?;
    let candidate =
        super::cp::candidate_if_running(context.model, context.compiled, &completed, context.budget, verification)?.map(|candidate| {
            CandidateSolution::verified(
                candidate.assignment().clone(),
                candidate.objectives().to_vec(),
                EngineKind::IntegerLocalSearch,
                verification,
            )
        });
    if let Some(candidate) = &candidate {
        super::cp::verify_assumptions(candidate, &context.request.assumptions)?;
    }
    Ok(candidate.map(|candidate| RepairedCandidate { candidate, physical_solution: completed, physical_objective }))
}

fn cp_repair_completion(solution: Option<Vec<i32>>, complete: bool) -> Result<Option<Vec<i32>>, SolveError> {
    match (solution, complete) {
        (Some(solution), _) => Ok(Some(solution)),
        (None, false) => Ok(None),
        (None, true) => Err(SolveError::InvalidResult("local-search decisions have no completion in the canonical CP model".to_string())),
    }
}

#[cfg(test)]
pub(super) fn audit_cp_repair_completion(complete: bool) -> Result<bool, SolveError> {
    cp_repair_completion(None, complete).map(|solution| solution.is_some())
}

fn physical_objective_value(problem: &crate::problem::Problem, values: &[i32]) -> Result<Option<i64>, SolveError> {
    let Some(objective) = problem.objective.as_ref() else {
        return Ok(None);
    };
    let by_variable = problem.search.iter().copied().zip(values.iter().copied()).collect::<HashMap<_, _>>();
    let value = |variable: VarId| by_variable.get(&variable).copied().map(i64::from);
    let objective_value = match objective {
        PhysicalObjective::Var(_, variable) => value(*variable).ok_or_else(|| {
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

fn compile_constraint(spec: &mut LocalSearchSpec, model: &Model, compiled: &CompiledCp, constraint: &Constraint) -> Result<(), SolveError> {
    let map = compiled.int_variables();
    match constraint {
        Constraint::Intension(expression) => spec.add_expr(expression_of(compiled, expression)?),
        Constraint::Selected { selector, constraint } => {
            let start = spec.begin_guarded_constraints();
            let result = compile_constraint(spec, model, compiled, constraint);
            spec.finish_guarded_constraints(start, map[selector.0]);
            result?;
        }
        Constraint::Linear { terms, relation, rhs } => spec.add_linear(
            terms.iter().map(|(coefficient, _)| *coefficient).collect(),
            terms.iter().map(|(_, variable)| map[variable.0]).collect(),
            physical_relation(*relation),
            *rhs,
        ),
        Constraint::Clause(literals) => spec.add_expr(clause_expression(map, literals)),
        Constraint::IntegerGlobal(global) => compile_global(spec, compiled, global)?,
        Constraint::SetSubset { subset, superset } => {
            for value in set_values(model, [*subset, *superset]) {
                match (membership(compiled, *subset, value), membership(compiled, *superset, value)) {
                    (Some(left), Some(right)) => spec.add_expr(Expr::Imp(
                        Box::new(Expr::Eq(Box::new(Expr::Var(left)), Box::new(Expr::Const(1)))),
                        Box::new(Expr::Eq(Box::new(Expr::Var(right)), Box::new(Expr::Const(1)))),
                    )),
                    (Some(left), None) => spec.add_linear(vec![1], vec![left], PhysicalRelation::Eq, 0),
                    (None, _) => {}
                }
            }
        }
        Constraint::SetDisjoint { left, right } => {
            for value in set_values(model, [*left, *right]) {
                if let (Some(left), Some(right)) = (membership(compiled, *left, value), membership(compiled, *right, value)) {
                    spec.add_linear(vec![1, 1], vec![left, right], PhysicalRelation::Le, 1);
                }
            }
        }
        Constraint::SetCardinality { set, min, max } => {
            let variables = compiled.sets()[set.0].membership.clone();
            spec.add_linear(vec![1; variables.len()], variables.clone(), PhysicalRelation::Ge, *min as i64);
            spec.add_linear(vec![1; variables.len()], variables, PhysicalRelation::Le, *max as i64);
        }
        Constraint::ListPartition { .. }
        | Constraint::ListPartitionWithCoverage { .. }
        | Constraint::SameList { .. }
        | Constraint::ItemPrecedence { .. }
        | Constraint::CollectionGlobal(_)
        | Constraint::ListLength { .. }
        | Constraint::ListItemSum { .. }
        | Constraint::ListReduction(_)
        | Constraint::IntervalPrecedence { .. }
        | Constraint::IntervalAlternative { .. }
        | Constraint::IntervalEndpointRelation { .. }
        | Constraint::IntervalResource(_) => {
            return Err(SolveError::Compile("integer local-search compiler received a collection or interval constraint".to_string()));
        }
    }
    Ok(())
}

fn compile_global(spec: &mut LocalSearchSpec, compiled: &CompiledCp, global: &IntGlobalConstraint) -> Result<(), SolveError> {
    let map = compiled.int_variables();
    let vars = |ids: &[crate::model::IntVarRef]| ids.iter().map(|variable| map[variable.0]).collect::<Vec<_>>();
    match global {
        IntGlobalConstraint::AllDifferent { variables, except } if except.is_empty() => spec.add_all_different(vars(variables)),
        IntGlobalConstraint::AllDifferent { variables, except } => spec.add_all_different_except(vars(variables), except.clone()),
        IntGlobalConstraint::AllEqual(variables) => spec.add_all_equal(vars(variables)),
        IntGlobalConstraint::Ordered { variables, relation } => {
            for pair in vars(variables).windows(2) {
                spec.add_linear(vec![1, -1], pair.to_vec(), physical_relation(*relation), 0);
            }
        }
        IntGlobalConstraint::Instantiation { variables, values } => {
            for (variable, value) in vars(variables).into_iter().zip(values) {
                spec.add_linear(vec![1], vec![variable], PhysicalRelation::Eq, i64::from(*value));
            }
        }
        IntGlobalConstraint::Minimum { target, variables } | IntGlobalConstraint::Maximum { target, variables } => {
            let values = vars(variables).into_iter().map(Expr::Var).collect();
            let extremum = if matches!(global, IntGlobalConstraint::Minimum { .. }) { Expr::Min(values) } else { Expr::Max(values) };
            spec.add_expr(Expr::Eq(Box::new(Expr::Var(map[target.0])), Box::new(extremum)));
        }
        IntGlobalConstraint::Element { array, index, value } => {
            spec.add_element(vars(array), map[index.0], map[value.0], 0);
        }
        IntGlobalConstraint::ElementConst { array, index, value } => {
            let tuples = array.iter().enumerate().map(|(index, value)| vec![index as i32, *value]).collect();
            spec.add_extension(vec![map[index.0], map[value.0]], tuples, true);
        }
        IntGlobalConstraint::Count { variables, value, relation, count } => {
            spec.add_count(vars(variables), vec![*value], physical_relation(*relation), LocalRhs::Const(*count));
        }
        IntGlobalConstraint::Cardinality { variables, values, lower, upper, closed } => {
            spec.add_cardinality(vars(variables), values.clone(), lower.clone(), upper.clone(), *closed);
        }
        IntGlobalConstraint::NValues { variables, relation, count } => {
            spec.add_n_values(vars(variables), physical_relation(*relation), LocalRhs::Const(*count));
        }
        IntGlobalConstraint::Table { variables, tuples, positive } => {
            spec.add_extension(vars(variables), tuples.clone(), *positive);
        }
        IntGlobalConstraint::Regular { variables, automaton } => spec.add_regular(
            vars(variables),
            Dfa {
                n_states: automaton.states,
                start: automaton.start,
                accept: automaton.accepting.clone(),
                transitions: automaton.transitions.clone(),
            },
        ),
        IntGlobalConstraint::Mdd { variables, mdd } => spec.add_mdd(
            vars(variables),
            Mdd {
                layers: mdd
                    .layers
                    .iter()
                    .map(|layer| layer.iter().map(|arc| MddArc { from: arc.from, value: arc.value, to: arc.to }).collect())
                    .collect(),
                nodes_per_layer: mdd.nodes_per_layer.clone(),
            },
        ),
        IntGlobalConstraint::Lex { left, right, strict } => {
            spec.add_lex_chain(vec![vars(left), vars(right)], *strict);
        }
        IntGlobalConstraint::LexChain { rows, strict } => {
            spec.add_lex_chain(rows.iter().map(|row| vars(row)).collect(), *strict);
        }
        IntGlobalConstraint::Channel { left, right } => spec.add_channel_inverse(vars(left), 0, vars(right), 0),
        IntGlobalConstraint::Circuit { successors, .. } => spec.add_circuit(vars(successors)),
        IntGlobalConstraint::NoOverlap { starts, durations } => spec.add_no_overlap(
            vars(starts).into_iter().map(|start| vec![start]).collect(),
            durations.iter().copied().map(|duration| vec![Expr::Const(duration)]).collect(),
            false,
        ),
        IntGlobalConstraint::OptionalNoOverlap { starts, durations, presences } => spec.add_no_overlap(
            vars(starts).into_iter().map(|start| vec![start]).collect(),
            durations
                .iter()
                .zip(presences)
                .map(|(&duration, presence)| {
                    vec![presence
                        .map_or(Expr::Const(duration), |presence| Expr::Mul(vec![Expr::Const(duration), Expr::Var(map[presence.0])]))]
                })
                .collect(),
            true,
        ),
        IntGlobalConstraint::AlternativeChannel { shared_start, starts, presences, .. } => {
            let presences = vars(presences);
            spec.add_linear(vec![1; presences.len()], presences.clone(), PhysicalRelation::Eq, 1);
            for (&start, &presence) in starts.iter().zip(&presences) {
                spec.add_expr(Expr::Imp(
                    Box::new(Expr::Eq(Box::new(Expr::Var(presence)), Box::new(Expr::Const(1)))),
                    Box::new(Expr::Eq(Box::new(Expr::Var(map[shared_start.0])), Box::new(Expr::Var(map[start.0])))),
                ));
            }
        }
        IntGlobalConstraint::Cumulative { starts, durations, demands, capacity } => spec.add_cumulative_rhs(
            vars(starts),
            durations.iter().copied().map(LocalRhs::Const).collect(),
            demands.iter().copied().map(LocalRhs::Const).collect(),
            LocalRhs::Const(*capacity),
        ),
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity } => {
            spec.add_cumulative(vars(starts), vars(durations), vars(demands), LocalRhs::Var(map[capacity.0]))
        }
        IntGlobalConstraint::BinPacking { items, sizes, capacities } => {
            spec.add_bin_packing(vars(items), sizes.clone(), capacities.iter().copied().map(LocalRhs::Const).collect(), false)
        }
        IntGlobalConstraint::BinLoads { items, sizes, loads } => {
            spec.add_bin_packing(vars(items), sizes.clone(), vars(loads).into_iter().map(LocalRhs::Var).collect(), true)
        }
        IntGlobalConstraint::Knapsack { variables, weights, profits, weight_relation, weight_limit, profit_relation, profit_limit } => {
            let variables = vars(variables);
            spec.add_linear(weights.clone(), variables.clone(), physical_relation(*weight_relation), *weight_limit);
            spec.add_linear(profits.clone(), variables, physical_relation(*profit_relation), *profit_limit);
        }
        IntGlobalConstraint::ValuePrecedence { variables, values, covered } => {
            spec.add_precedence(vars(variables), values.clone(), *covered);
        }
    }
    Ok(())
}

fn expression_of(compiled: &CompiledCp, expression: &crate::model::IntExpr) -> Result<Expr, SolveError> {
    compiled.compile_expression(expression).map_err(|error| SolveError::Compile(error.reason))
}

fn physical_relation(relation: Relation) -> PhysicalRelation {
    match relation {
        Relation::Eq => PhysicalRelation::Eq,
        Relation::Ne => PhysicalRelation::Ne,
        Relation::Le => PhysicalRelation::Le,
        Relation::Lt => PhysicalRelation::Lt,
        Relation::Ge => PhysicalRelation::Ge,
        Relation::Gt => PhysicalRelation::Gt,
    }
}

fn clause_expression(map: &[VarId], literals: &[BoolLiteral]) -> Expr {
    Expr::Or(
        literals
            .iter()
            .map(|literal| {
                let variable = Expr::Var(map[literal.variable.0]);
                if literal.positive {
                    variable
                } else {
                    Expr::Not(Box::new(variable))
                }
            })
            .collect(),
    )
}

fn membership(compiled: &CompiledCp, set: SetVarRef, value: i32) -> Option<VarId> {
    let set = &compiled.sets()[set.0];
    set.values.binary_search(&value).ok().map(|index| set.membership[index])
}

fn set_values<const N: usize>(model: &Model, sets: [SetVarRef; N]) -> std::collections::BTreeSet<i32> {
    sets.iter().flat_map(|set| model.sets()[set.0].possible.iter().copied()).collect()
}
