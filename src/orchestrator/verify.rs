//! Canonical semantic solution replay.

use std::collections::{BTreeSet, HashSet};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(test)]
use std::sync::{mpsc, Arc};
#[cfg(test)]
use std::time::Duration;

use crate::model::{
    CompiledCollection, Constraint, IndependentComponent, IndependentDecomposition, IntExpr, IntGlobalConstraint, IntVarRef, Model,
    Objective, PartitionCoverage, Relation,
};

use super::{Assignment, SolveBudget, SolveError, TerminationReason};

#[cfg(test)]
const FINAL_REPLAY_POLL: Duration = Duration::from_millis(2);
#[cfg(test)]
const MAX_SUPERVISED_FINAL_REPLAYS: usize = 4;

#[cfg(test)]
static ACTIVE_SUPERVISED_FINAL_REPLAYS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
enum FinalReplayAttempt<T> {
    Completed(Result<T, SolveError>),
    Stopped,
    Unavailable,
}

/// Run one final canonical replay under the request budget. A soft deadline
/// may trigger exactly one fresh replay under the bounded finalization grace.
/// Hard cancellation is checked by the supervisor as well as the replay, so a
/// callback that does not cooperate cannot delay publication indefinitely.
pub(crate) fn verify_final_with_budget<T, F>(budget: &SolveBudget, operation: F) -> Result<T, SolveError>
where
    F: Fn(&AtomicBool) -> Result<T, SolveError>,
{
    match operation(budget.stop()) {
        Err(SolveError::Interrupted(_)) if soft_deadline_allows_grace(budget) => {}
        result => return result,
    }

    let grace = budget.finalization_grace_stop();
    let result = operation(grace.flag());
    if budget.hard_cancelled() || grace.flag().load(Ordering::Acquire) {
        Err(final_replay_interrupted(budget))
    } else {
        result
    }
}

/// Supervise the first replay as well as its grace. This variant is reserved
/// for models that can invoke user-provided external callbacks, since those
/// callbacks cannot be forced to poll the cancellation token.
#[cfg(test)]
pub(crate) fn verify_final_supervised_with_budget<T, F>(budget: &SolveBudget, operation: F) -> Result<T, SolveError>
where
    T: Send + 'static,
    F: Fn(&AtomicBool) -> Result<T, SolveError> + Clone + Send + 'static,
{
    match run_final_replay_attempt(budget, budget.stop_handle(), operation.clone()) {
        FinalReplayAttempt::Completed(Err(SolveError::Interrupted(_))) if soft_deadline_allows_grace(budget) => {}
        FinalReplayAttempt::Completed(result) => return result,
        // A stopped supervisor may have detached a non-cooperative callback.
        // Never invoke it a second time concurrently during the grace period.
        FinalReplayAttempt::Stopped | FinalReplayAttempt::Unavailable => return Err(final_replay_interrupted(budget)),
    }

    run_supervised_final_replay_grace(budget, operation)
}

#[cfg(test)]
fn run_supervised_final_replay_grace<T, F>(budget: &SolveBudget, operation: F) -> Result<T, SolveError>
where
    T: Send + 'static,
    F: Fn(&AtomicBool) -> Result<T, SolveError> + Send + 'static,
{
    let grace = budget.finalization_grace_stop();
    let outcome = run_final_replay_attempt(budget, grace.flag_handle(), operation);
    match outcome {
        FinalReplayAttempt::Completed(result) if !budget.hard_cancelled() => result,
        FinalReplayAttempt::Completed(_) | FinalReplayAttempt::Stopped | FinalReplayAttempt::Unavailable => {
            Err(final_replay_interrupted(budget))
        }
    }
}

#[cfg(test)]
fn run_final_replay_attempt<T, F>(budget: &SolveBudget, stop: Arc<AtomicBool>, operation: F) -> FinalReplayAttempt<T>
where
    T: Send + 'static,
    F: Fn(&AtomicBool) -> Result<T, SolveError> + Send + 'static,
{
    let Some(permit) = SupervisedReplayPermit::acquire() else {
        return FinalReplayAttempt::Unavailable;
    };
    let (sender, receiver) = mpsc::sync_channel(1);
    let worker_stop = Arc::clone(&stop);
    let worker = std::thread::Builder::new().name("qayd-final-replay".to_string()).spawn(move || {
        let _permit = permit;
        let _ = sender.send(operation(&worker_stop));
    });
    if worker.is_err() {
        return FinalReplayAttempt::Unavailable;
    }

    loop {
        match receiver.recv_timeout(FINAL_REPLAY_POLL) {
            Ok(result) => {
                if budget.hard_cancelled() || stop.load(Ordering::Acquire) {
                    return FinalReplayAttempt::Stopped;
                }
                return FinalReplayAttempt::Completed(result);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if budget.hard_cancelled() || stop.load(Ordering::Acquire) {
                    return FinalReplayAttempt::Stopped;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return FinalReplayAttempt::Completed(Err(SolveError::Engine(
                    "canonical final replay worker terminated without a result".to_string(),
                )));
            }
        }
    }
}

#[cfg(test)]
struct SupervisedReplayPermit;

#[cfg(test)]
impl SupervisedReplayPermit {
    fn acquire() -> Option<Self> {
        ACTIVE_SUPERVISED_FINAL_REPLAYS
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| (active < MAX_SUPERVISED_FINAL_REPLAYS).then_some(active + 1))
            .ok()
            .map(|_| Self)
    }
}

#[cfg(test)]
impl Drop for SupervisedReplayPermit {
    fn drop(&mut self) {
        ACTIVE_SUPERVISED_FINAL_REPLAYS.fetch_sub(1, Ordering::AcqRel);
    }
}

fn soft_deadline_allows_grace(budget: &SolveBudget) -> bool {
    budget.termination_reason() == TerminationReason::Deadline && !budget.hard_cancelled()
}

fn final_replay_interrupted(budget: &SolveBudget) -> SolveError {
    SolveError::Interrupted(format!("canonical final replay exceeded its grace or was cancelled: {:?}", budget.termination_reason()))
}

pub fn verify_semantic_assignment(model: &Model, assignment: &Assignment, claimed_objectives: &[i64]) -> Result<Vec<i64>, SolveError> {
    verify_semantic_assignment_interruptible(model, assignment, claimed_objectives, &AtomicBool::new(false))
}

static VERIFICATION_CALLS: AtomicU64 = AtomicU64::new(0);

#[doc(hidden)]
pub fn audit_semantic_verification_calls() -> u64 {
    VERIFICATION_CALLS.load(Ordering::Relaxed)
}

pub(crate) fn verify_semantic_assignment_interruptible(
    model: &Model,
    assignment: &Assignment,
    claimed_objectives: &[i64],
    stop: &AtomicBool,
) -> Result<Vec<i64>, SolveError> {
    VERIFICATION_CALLS.fetch_add(1, Ordering::Relaxed);
    check_stop(stop)?;
    if !model.validate_interruptible(stop).map_err(|errors| SolveError::Compile(errors.join("; ")))? {
        return Err(SolveError::Interrupted("canonical semantic verification was interrupted".to_string()));
    }
    check_stop(stop)?;
    verify_semantic_assignment_validated(model, assignment, claimed_objectives, stop)
}

/// Replay an assignment against a model already validated at the orchestrator
/// boundary. This skips only declaration/reference validation; all assignment,
/// constraint, objective, and interruption checks remain canonical.
pub(crate) fn verify_semantic_assignment_validated_interruptible(
    model: &Model,
    assignment: &Assignment,
    claimed_objectives: &[i64],
    stop: &AtomicBool,
) -> Result<Vec<i64>, SolveError> {
    VERIFICATION_CALLS.fetch_add(1, Ordering::Relaxed);
    check_stop(stop)?;
    verify_semantic_assignment_validated(model, assignment, claimed_objectives, stop)
}

fn verify_semantic_assignment_validated(
    model: &Model,
    assignment: &Assignment,
    claimed_objectives: &[i64],
    stop: &AtomicBool,
) -> Result<Vec<i64>, SolveError> {
    let has_integer = !model.int_vars().is_empty() || !model.sets().is_empty();
    let has_collection = !model.lists().is_empty() || !model.intervals().is_empty();
    let collection_families = usize::from(!model.lists().is_empty()) + usize::from(!model.intervals().is_empty());
    let mixed_families = (has_integer && has_collection) || collection_families > 1;
    // Integer and set assignments can always be replayed directly against the
    // canonical model. Discovering and cloning their connected components here
    // only duplicates work after exact CP search. Collection models still need
    // decomposition because independently compiled list or interval components
    // may not have a valid monolithic physical representation.
    if has_collection {
        match model.independent_family_components_interruptible(stop) {
            IndependentDecomposition::Components(components) => {
                return verify_decomposed_assignment(model, assignment, claimed_objectives, components, stop);
            }
            IndependentDecomposition::Interrupted => {
                return Err(SolveError::Interrupted("canonical semantic decomposition was interrupted".to_string()));
            }
            IndependentDecomposition::NotApplicable => {}
        }
    }
    if mixed_families {
        return Err(SolveError::InvalidResult("mixed assignment cannot be canonically decomposed".to_string()));
    }
    if !model.lists().is_empty() || !model.intervals().is_empty() {
        return verify_collection_assignment(model, assignment, claimed_objectives, stop);
    }
    verify_integer_shape(model, assignment)?;
    let integer = |reference: IntVarRef| assignment.integers[reference.0];
    for (index, domain) in model.int_vars().iter().enumerate() {
        poll_stop(stop, index)?;
        let value =
            assignment.integers[index].ok_or_else(|| SolveError::InvalidResult(format!("integer variable {index} is unassigned")))?;
        if !domain.contains(value) {
            check_stop(stop)?;
            return Err(SolveError::InvalidResult(format!("integer variable {index} has value {value} outside its domain")));
        }
    }
    for (index, declaration) in model.sets().iter().enumerate() {
        check_stop(stop)?;
        let value = &assignment.sets[index];
        let mut sorted = true;
        for (position, pair) in value.windows(2).enumerate() {
            poll_stop(stop, position)?;
            if pair[0] >= pair[1] {
                sorted = false;
                break;
            }
        }
        check_stop(stop)?;
        if !sorted {
            return Err(SolveError::InvalidResult(format!("set variable {index} is not strictly sorted and duplicate-free")));
        }
        let mut bounds_hold = true;
        for (position, required) in declaration.required.iter().enumerate() {
            poll_stop(stop, position)?;
            if value.binary_search(required).is_err() {
                bounds_hold = false;
                break;
            }
        }
        if bounds_hold {
            for (position, assigned) in value.iter().enumerate() {
                poll_stop(stop, position)?;
                if !declaration.possible.contains(assigned) {
                    bounds_hold = false;
                    break;
                }
            }
        }
        check_stop(stop)?;
        if !bounds_hold {
            return Err(SolveError::InvalidResult(format!("set variable {index} violates its lower or upper bound")));
        }
    }
    for (index, constraint) in model.constraints().iter().enumerate() {
        poll_stop(stop, index)?;
        if !constraint_holds(model, constraint, assignment, &integer, stop)? {
            check_stop(stop)?;
            return Err(SolveError::InvalidResult(format!("constraint {index} is violated")));
        }
    }
    let mut objectives = Vec::with_capacity(model.objectives().len());
    for (index, objective) in model.objectives().iter().enumerate() {
        poll_stop(stop, index)?;
        let value = match objective {
            Objective::IntExpr { expr, .. } => evaluate_int_expr_interruptible(expr, &integer, stop)?
                .ok_or_else(|| SolveError::InvalidResult("integer objective is undefined or overflows".to_string())),
            Objective::ListTerms { .. } | Objective::Makespan { .. } => {
                Err(SolveError::InvalidResult("non-integer objective in an integer assignment".to_string()))
            }
        }?;
        check_stop(stop)?;
        objectives.push(value);
    }
    check_stop(stop)?;
    if objectives != claimed_objectives {
        return Err(SolveError::InvalidResult(format!(
            "objective mismatch: engine reported {claimed_objectives:?}, canonical replay produced {objectives:?}"
        )));
    }
    check_stop(stop)?;
    Ok(objectives)
}

fn verify_decomposed_assignment(
    model: &Model,
    assignment: &Assignment,
    claimed_objectives: &[i64],
    components: Vec<IndependentComponent>,
    stop: &AtomicBool,
) -> Result<Vec<i64>, SolveError> {
    if components.len() < 2 {
        return Err(SolveError::InvalidResult("semantic decomposition made no recursive verification progress".to_string()));
    }
    for (index, component) in components.iter().enumerate() {
        poll_stop(stop, index)?;
        if component.model.int_vars().len() == model.int_vars().len()
            && component.model.sets().len() == model.sets().len()
            && component.model.lists().len() == model.lists().len()
            && component.model.intervals().len() == model.intervals().len()
            && component.model.constraints().len() == model.constraints().len()
            && component.model.objectives().len() == model.objectives().len()
        {
            return Err(SolveError::InvalidResult("semantic decomposition contains a component identical to its parent".to_string()));
        }
    }
    if assignment.integers.len() != model.int_vars().len()
        || assignment.sets.len() != model.sets().len()
        || assignment.lists.len() != model.lists().len()
        || assignment.intervals.len() != model.intervals().len()
    {
        return Err(SolveError::InvalidResult("decomposed assignment has the wrong semantic arena shape".to_string()));
    }
    if claimed_objectives.len() != model.objectives().len() {
        return Err(SolveError::InvalidResult(format!(
            "decomposed candidate reports {} objective tiers but the semantic model has {}",
            claimed_objectives.len(),
            model.objectives().len()
        )));
    }
    let mut integer_coverage = vec![false; model.int_vars().len()];
    let mut set_coverage = vec![false; model.sets().len()];
    let mut list_coverage = vec![false; model.lists().len()];
    let mut interval_coverage = vec![false; model.intervals().len()];
    let mut mode_coverage = vec![false; model.interval_modes().len()];
    let mut objective_coverage = vec![false; claimed_objectives.len()];
    for component in components {
        check_stop(stop)?;
        let projected = project_component_assignment(
            &component,
            assignment,
            &mut integer_coverage,
            &mut set_coverage,
            &mut list_coverage,
            &mut interval_coverage,
            &mut mode_coverage,
            stop,
        )?;
        let mut local_claims = Vec::with_capacity(component.objective_tiers.len());
        for (index, &tier) in component.objective_tiers.iter().enumerate() {
            poll_stop(stop, index)?;
            let claimed = claimed_objectives
                .get(tier)
                .copied()
                .ok_or_else(|| SolveError::InvalidResult(format!("component maps to unknown semantic objective tier {tier}")))?;
            let covered = objective_coverage
                .get_mut(tier)
                .ok_or_else(|| SolveError::InvalidResult(format!("component maps to unknown semantic objective tier {tier}")))?;
            if std::mem::replace(covered, true) {
                return Err(SolveError::InvalidResult(format!(
                    "semantic objective tier {tier} is owned by more than one decomposition component"
                )));
            }
            local_claims.push(claimed);
        }
        verify_semantic_assignment_validated_interruptible(&component.model, &projected, &local_claims, stop)?;
    }
    for (kind, coverage) in [
        ("integer", integer_coverage.as_slice()),
        ("set", set_coverage.as_slice()),
        ("list", list_coverage.as_slice()),
        ("interval", interval_coverage.as_slice()),
        ("interval mode", mode_coverage.as_slice()),
        ("objective tier", objective_coverage.as_slice()),
    ] {
        if let Some(index) = first_uncovered(coverage, stop)? {
            return Err(SolveError::InvalidResult(format!("semantic {kind} {index} has no decomposition owner")));
        }
    }
    check_stop(stop)?;
    Ok(claimed_objectives.to_vec())
}

#[allow(clippy::too_many_arguments)]
fn project_component_assignment(
    component: &IndependentComponent,
    assignment: &Assignment,
    integer_coverage: &mut [bool],
    set_coverage: &mut [bool],
    list_coverage: &mut [bool],
    interval_coverage: &mut [bool],
    mode_coverage: &mut [bool],
    stop: &AtomicBool,
) -> Result<Assignment, SolveError> {
    let mut projected = Assignment {
        integers: vec![None; component.model.int_vars().len()],
        sets: vec![Vec::new(); component.model.sets().len()],
        lists: vec![Vec::new(); component.model.lists().len()],
        intervals: vec![Default::default(); component.model.intervals().len()],
    };
    let mut local_integers = vec![false; projected.integers.len()];
    let mut local_sets = vec![false; projected.sets.len()];
    let mut local_lists = vec![false; projected.lists.len()];
    let mut local_intervals = vec![false; projected.intervals.len()];
    let mut local_modes = vec![false; component.model.interval_modes().len()];
    for (index, &(original, local)) in component.integers.iter().enumerate() {
        poll_stop(stop, index)?;
        mark_projection(integer_coverage, &mut local_integers, original, local, "integer")?;
        projected.integers[local] = assignment.integers[original];
    }
    for (index, &(original, local)) in component.sets.iter().enumerate() {
        poll_stop(stop, index)?;
        mark_projection(set_coverage, &mut local_sets, original, local, "set")?;
        projected.sets[local] = assignment.sets[original].clone();
    }
    for (index, &(original, local)) in component.lists.iter().enumerate() {
        poll_stop(stop, index)?;
        mark_projection(list_coverage, &mut local_lists, original, local, "list")?;
        projected.lists[local] = assignment.lists[original].clone();
    }
    for (index, &(original, local)) in component.intervals.iter().enumerate() {
        poll_stop(stop, index)?;
        mark_projection(interval_coverage, &mut local_intervals, original, local, "interval")?;
        let mut value = assignment.intervals[original];
        if let Some(original_mode) = value.mode {
            let mut selected = None;
            for (mode_index, &(candidate, local)) in component.interval_modes.iter().enumerate() {
                poll_stop(stop, mode_index)?;
                if candidate == original_mode {
                    selected = Some(local);
                    break;
                }
            }
            value.mode = Some(selected.ok_or_else(|| {
                SolveError::InvalidResult(format!(
                    "semantic interval {original} selects mode {original_mode} outside its decomposition component"
                ))
            })?);
        }
        projected.intervals[local] = value;
    }
    for (index, &(original, local)) in component.interval_modes.iter().enumerate() {
        poll_stop(stop, index)?;
        mark_projection(mode_coverage, &mut local_modes, original, local, "interval mode")?;
    }
    for (kind, coverage) in [
        ("integer", local_integers.as_slice()),
        ("set", local_sets.as_slice()),
        ("list", local_lists.as_slice()),
        ("interval", local_intervals.as_slice()),
        ("interval mode", local_modes.as_slice()),
    ] {
        if let Some(local) = first_uncovered(coverage, stop)? {
            return Err(SolveError::InvalidResult(format!("component-local {kind} {local} has no semantic owner")));
        }
    }
    check_stop(stop)?;
    Ok(projected)
}

fn first_uncovered(coverage: &[bool], stop: &AtomicBool) -> Result<Option<usize>, SolveError> {
    for (index, &covered) in coverage.iter().enumerate() {
        poll_stop(stop, index)?;
        if !covered {
            return Ok(Some(index));
        }
    }
    check_stop(stop)?;
    Ok(None)
}

fn mark_projection(
    coverage: &mut [bool],
    local_coverage: &mut [bool],
    original: usize,
    local: usize,
    kind: &str,
) -> Result<(), SolveError> {
    let Some(covered) = coverage.get_mut(original) else {
        return Err(SolveError::InvalidResult(format!("component maps unknown semantic {kind} {original}")));
    };
    if std::mem::replace(covered, true) {
        return Err(SolveError::InvalidResult(format!("semantic {kind} {original} is owned by more than one component")));
    }
    let Some(local_covered) = local_coverage.get_mut(local) else {
        return Err(SolveError::InvalidResult(format!("component maps {kind} {original} to unknown local index {local}")));
    };
    if std::mem::replace(local_covered, true) {
        return Err(SolveError::InvalidResult(format!("component-local {kind} {local} has more than one semantic owner")));
    }
    Ok(())
}

fn check_stop(stop: &AtomicBool) -> Result<(), SolveError> {
    if stop.load(Ordering::Acquire) {
        Err(SolveError::Interrupted("canonical semantic verification was interrupted".to_string()))
    } else {
        Ok(())
    }
}

pub(crate) fn evaluate_int_expr_interruptible(
    expression: &IntExpr,
    value: &impl Fn(IntVarRef) -> Option<i64>,
    stop: &AtomicBool,
) -> Result<Option<i64>, SolveError> {
    check_stop(stop)?;
    let result = match expression {
        IntExpr::Constant(number) => Some(*number),
        IntExpr::Variable(variable) => value(*variable),
        IntExpr::Neg(inner) => evaluate_int_expr_interruptible(inner, value, stop)?.and_then(i64::checked_neg),
        IntExpr::Abs(inner) => evaluate_int_expr_interruptible(inner, value, stop)?.and_then(i64::checked_abs),
        IntExpr::Add(items) => {
            let mut total = Some(0i64);
            for (index, item) in items.iter().enumerate() {
                poll_stop(stop, index)?;
                total = total.zip(evaluate_int_expr_interruptible(item, value, stop)?).and_then(|(left, right)| left.checked_add(right));
                if total.is_none() {
                    break;
                }
            }
            total
        }
        IntExpr::Sub(left, right) => evaluate_int_expr_interruptible(left, value, stop)?
            .zip(evaluate_int_expr_interruptible(right, value, stop)?)
            .and_then(|(left, right)| left.checked_sub(right)),
        IntExpr::Mul(items) => {
            let mut product = Some(1i64);
            for (index, item) in items.iter().enumerate() {
                poll_stop(stop, index)?;
                product =
                    product.zip(evaluate_int_expr_interruptible(item, value, stop)?).and_then(|(left, right)| left.checked_mul(right));
                if product.is_none() {
                    break;
                }
            }
            product
        }
        IntExpr::Div(left, right) => evaluate_int_expr_interruptible(left, value, stop)?
            .zip(evaluate_int_expr_interruptible(right, value, stop)?)
            .and_then(|(left, right)| left.checked_div(right)),
        IntExpr::Mod(left, right) => evaluate_int_expr_interruptible(left, value, stop)?
            .zip(evaluate_int_expr_interruptible(right, value, stop)?)
            .and_then(|(left, right)| left.checked_rem(right)),
        IntExpr::Min(items) | IntExpr::Max(items) => {
            let mut result = None;
            for (index, item) in items.iter().enumerate() {
                poll_stop(stop, index)?;
                let Some(item) = evaluate_int_expr_interruptible(item, value, stop)? else {
                    return Ok(None);
                };
                result = Some(match (expression, result) {
                    (IntExpr::Min(_), Some(current)) => item.min(current),
                    (IntExpr::Max(_), Some(current)) => item.max(current),
                    (IntExpr::Min(_) | IntExpr::Max(_), None) => item,
                    _ => unreachable!("expression match preserves min/max variant"),
                });
            }
            result
        }
        IntExpr::Eq(left, right)
        | IntExpr::Ne(left, right)
        | IntExpr::Lt(left, right)
        | IntExpr::Le(left, right)
        | IntExpr::Gt(left, right)
        | IntExpr::Ge(left, right)
        | IntExpr::Iff(left, right) => {
            let operands = evaluate_int_expr_interruptible(left, value, stop)?.zip(evaluate_int_expr_interruptible(right, value, stop)?);
            operands.map(|(left, right)| match expression {
                IntExpr::Eq(_, _) => i64::from(left == right),
                IntExpr::Ne(_, _) => i64::from(left != right),
                IntExpr::Lt(_, _) => i64::from(left < right),
                IntExpr::Le(_, _) => i64::from(left <= right),
                IntExpr::Gt(_, _) => i64::from(left > right),
                IntExpr::Ge(_, _) => i64::from(left >= right),
                IntExpr::Iff(_, _) => i64::from((left != 0) == (right != 0)),
                _ => unreachable!("expression match preserves binary variant"),
            })
        }
        IntExpr::Not(inner) => evaluate_int_expr_interruptible(inner, value, stop)?.map(|number| i64::from(number == 0)),
        IntExpr::And(items) => {
            let mut all = true;
            for (index, item) in items.iter().enumerate() {
                poll_stop(stop, index)?;
                if evaluate_int_expr_interruptible(item, value, stop)?.is_none_or(|number| number == 0) {
                    all = false;
                    break;
                }
            }
            Some(i64::from(all))
        }
        IntExpr::Or(items) => {
            let mut any = false;
            for (index, item) in items.iter().enumerate() {
                poll_stop(stop, index)?;
                if evaluate_int_expr_interruptible(item, value, stop)?.is_some_and(|number| number != 0) {
                    any = true;
                    break;
                }
            }
            Some(i64::from(any))
        }
        IntExpr::Imp(left, right) => {
            let Some(left) = evaluate_int_expr_interruptible(left, value, stop)? else {
                return Ok(None);
            };
            if left == 0 {
                Some(1)
            } else {
                evaluate_int_expr_interruptible(right, value, stop)?.map(|right| i64::from(right != 0))
            }
        }
        IntExpr::IfThenElse(condition, then_value, else_value) => {
            let Some(condition) = evaluate_int_expr_interruptible(condition, value, stop)? else {
                return Ok(None);
            };
            if condition != 0 {
                evaluate_int_expr_interruptible(then_value, value, stop)?
            } else {
                evaluate_int_expr_interruptible(else_value, value, stop)?
            }
        }
    };
    check_stop(stop)?;
    Ok(result)
}

fn verify_collection_assignment(
    model: &Model,
    assignment: &Assignment,
    claimed_objectives: &[i64],
    stop: &AtomicBool,
) -> Result<Vec<i64>, SolveError> {
    if !model.int_vars().is_empty() || !model.sets().is_empty() {
        return Err(SolveError::InvalidResult("mixed integer/set and collection assignment needs a decomposition plan".to_string()));
    }
    let compiled = CompiledCollection::compile_interruptible(model, stop)
        .map_err(|error| SolveError::Compile(error.reason))?
        .ok_or_else(|| SolveError::Interrupted("canonical collection verification was interrupted".to_string()))?;
    let physical = compiled.as_model();
    let mut objectives = Vec::with_capacity(claimed_objectives.len());
    for (index, &objective) in claimed_objectives.iter().enumerate() {
        poll_stop(stop, index)?;
        objectives.push(objective);
    }
    let solution = if physical.schedule.is_some() {
        if assignment.intervals.len() != model.intervals().len() || !assignment.lists.is_empty() {
            return Err(SolveError::InvalidResult("schedule assignment has the wrong arena shape".to_string()));
        }
        let mut starts = Vec::with_capacity(assignment.intervals.len());
        let mut presences = Vec::with_capacity(assignment.intervals.len());
        let mut machines = Vec::with_capacity(assignment.intervals.len());
        let mut modes = Vec::with_capacity(assignment.intervals.len());
        for (index, value) in assignment.intervals.iter().enumerate() {
            poll_stop(stop, index)?;
            starts.push(value.start.unwrap_or(0));
            presences.push(value.present);
            machines.push(value.machine.and_then(|machine| i64::try_from(machine).ok()).unwrap_or(-1));
            modes.push(value.mode);
        }
        crate::model::list::CollectionSolution {
            lists: Vec::new(),
            objectives,
            feasible: true,
            starts,
            presences,
            machines,
            modes,
            bound: None,
        }
    } else {
        if assignment.lists.len() != model.lists().len() || !assignment.intervals.is_empty() {
            return Err(SolveError::InvalidResult("list assignment has the wrong arena shape".to_string()));
        }
        let mut lists = Vec::with_capacity(assignment.lists.len());
        for (list_index, source) in assignment.lists.iter().enumerate() {
            poll_stop(stop, list_index)?;
            let mut list = Vec::with_capacity(source.len());
            for (item_index, &item) in source.iter().enumerate() {
                poll_stop(stop, item_index)?;
                list.push(item);
            }
            lists.push(list);
        }
        crate::model::list::CollectionSolution {
            lists,
            objectives,
            feasible: true,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
            modes: Vec::new(),
            bound: None,
        }
    };
    crate::model::list::verify_collection_solution_interruptible(physical, &solution, stop).map_err(|reason| {
        if stop.load(Ordering::Acquire) {
            SolveError::Interrupted(reason)
        } else {
            SolveError::InvalidResult(reason)
        }
    })
}

fn verify_integer_shape(model: &Model, assignment: &Assignment) -> Result<(), SolveError> {
    if assignment.integers.len() != model.int_vars().len()
        || assignment.sets.len() != model.sets().len()
        || !assignment.lists.is_empty()
        || !assignment.intervals.is_empty()
    {
        return Err(SolveError::InvalidResult("integer assignment has the wrong arena shape".to_string()));
    }
    Ok(())
}

fn constraint_holds(
    model: &Model,
    constraint: &Constraint,
    assignment: &Assignment,
    integer: &impl Fn(IntVarRef) -> Option<i64>,
    stop: &AtomicBool,
) -> Result<bool, SolveError> {
    check_stop(stop)?;
    let holds = match constraint {
        Constraint::Intension(expr) => evaluate_int_expr_interruptible(expr, integer, stop)?.is_some_and(|value| value != 0),
        Constraint::Selected { selector, constraint } => {
            integer(*selector) != Some(1) || constraint_holds(model, constraint, assignment, integer, stop)?
        }
        Constraint::Linear { terms, relation, rhs } => {
            let mut left = Some(0i128);
            for (index, (coefficient, variable)) in terms.iter().enumerate() {
                poll_stop(stop, index)?;
                left = left.zip(integer(*variable)).map(|(sum, value)| sum + i128::from(*coefficient) * i128::from(value));
                if left.is_none() {
                    break;
                }
            }
            left.is_some_and(|left| compare_i128(left, *relation, i128::from(*rhs)))
        }
        Constraint::Clause(literals) => {
            let mut satisfied = false;
            for (index, literal) in literals.iter().enumerate() {
                poll_stop(stop, index)?;
                if integer(literal.variable).is_some_and(|value| (value != 0) == literal.positive) {
                    satisfied = true;
                    break;
                }
            }
            satisfied
        }
        Constraint::IntegerGlobal(global) => global_holds(global, integer, stop)?,
        Constraint::SetSubset { subset, superset } => set_subset_holds(&assignment.sets[subset.0], &assignment.sets[superset.0], stop)?,
        Constraint::SetDisjoint { left, right } => set_disjoint_holds(&assignment.sets[left.0], &assignment.sets[right.0], stop)?,
        Constraint::SetCardinality { set: reference, min, max } => (*min..=*max).contains(&assignment.sets[reference.0].len()),
        Constraint::ListPartition { .. }
        | Constraint::ListPartitionWithCoverage { coverage: PartitionCoverage::Exact | PartitionCoverage::Partial, .. }
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
            return Err(SolveError::InvalidResult(format!(
                "collection constraint reached integer verification in a model with {} lists and {} intervals",
                model.lists().len(),
                model.intervals().len()
            )));
        }
    };
    check_stop(stop)?;
    Ok(holds)
}

fn global_holds(global: &IntGlobalConstraint, value: &impl Fn(IntVarRef) -> Option<i64>, stop: &AtomicBool) -> Result<bool, SolveError> {
    check_stop(stop)?;
    let get = |reference: IntVarRef| {
        value(reference).ok_or_else(|| SolveError::InvalidResult(format!("integer variable {} is unassigned", reference.0)))
    };
    let values = |variables: &[IntVarRef]| collect_values(variables, &get, stop);
    let holds = match global {
        IntGlobalConstraint::AllDifferent { variables, except } => {
            let assigned = values(variables)?;
            let mut excluded = HashSet::with_capacity(except.len());
            for (index, except) in except.iter().enumerate() {
                poll_stop(stop, index)?;
                excluded.insert(i64::from(*except));
            }
            let mut seen = HashSet::with_capacity(assigned.len());
            let mut distinct = true;
            for (index, assigned) in assigned.into_iter().enumerate() {
                poll_stop(stop, index)?;
                if !excluded.contains(&assigned) && !seen.insert(assigned) {
                    distinct = false;
                    break;
                }
            }
            distinct
        }
        IntGlobalConstraint::AllEqual(variables) => adjacent_values_hold(&values(variables)?, |left, right| left == right, stop)?,
        IntGlobalConstraint::Ordered { variables, relation } => {
            adjacent_values_hold(&values(variables)?, |left, right| compare_i128(i128::from(left), *relation, i128::from(right)), stop)?
        }
        IntGlobalConstraint::Instantiation { variables, values: expected } => equal_expected_values(&values(variables)?, expected, stop)?,
        IntGlobalConstraint::Minimum { target, variables } => extreme_holds(&values(variables)?, get(*target)?, true, stop)?,
        IntGlobalConstraint::Maximum { target, variables } => extreme_holds(&values(variables)?, get(*target)?, false, stop)?,
        IntGlobalConstraint::Element { array, index, value: target } => {
            if let Some(variable) = usize::try_from(get(*index)?).ok().and_then(|index| array.get(index)) {
                get(*variable)? == get(*target)?
            } else {
                false
            }
        }
        IntGlobalConstraint::ElementConst { array, index, value: target } => {
            if let Some(element) = usize::try_from(get(*index)?).ok().and_then(|index| array.get(index)) {
                i64::from(*element) == get(*target)?
            } else {
                false
            }
        }
        IntGlobalConstraint::Count { variables, value: expected, relation, count } => {
            let assigned = values(variables)?;
            let mut actual = 0i128;
            for (index, assigned) in assigned.iter().enumerate() {
                poll_stop(stop, index)?;
                actual += i128::from(*assigned == i64::from(*expected));
            }
            compare_i128(actual, *relation, i128::from(*count))
        }
        IntGlobalConstraint::Cardinality { variables, values: counted, lower, upper, closed } => {
            let assigned = values(variables)?;
            cardinality_holds(&assigned, counted, lower, upper, *closed, stop)?
        }
        IntGlobalConstraint::NValues { variables, relation, count } => {
            let assigned = values(variables)?;
            let mut distinct = BTreeSet::new();
            for (index, assigned) in assigned.into_iter().enumerate() {
                poll_stop(stop, index)?;
                distinct.insert(assigned);
            }
            let distinct = distinct.len() as i128;
            compare_i128(distinct, *relation, i128::from(*count))
        }
        IntGlobalConstraint::Table { variables, tuples, positive } => table_holds(&values(variables)?, tuples, *positive, stop)?,
        IntGlobalConstraint::Regular { variables, automaton } => regular_holds(&values(variables)?, automaton, stop)?,
        IntGlobalConstraint::Mdd { variables, mdd } => mdd_holds(&values(variables)?, mdd, stop)?,
        IntGlobalConstraint::Lex { left, right, strict } => lex_holds(&values(left)?, &values(right)?, *strict, stop)?,
        IntGlobalConstraint::LexChain { rows, strict } => lex_chain_holds(rows, *strict, &get, stop)?,
        IntGlobalConstraint::Channel { left, right } => {
            let left = values(left)?;
            let right = values(right)?;
            channel_holds(&left, &right, stop)?
        }
        IntGlobalConstraint::Circuit { successors, .. } => circuit_holds(&values(successors)?, stop)?,
        IntGlobalConstraint::NoOverlap { starts, durations } => no_overlap_holds(&values(starts)?, durations, stop)?,
        IntGlobalConstraint::OptionalNoOverlap { starts, durations, presences } => {
            let starts = values(starts)?;
            let mut active_starts = Vec::new();
            let mut active_durations = Vec::new();
            for (index, ((&start, &duration), presence)) in starts.iter().zip(durations).zip(presences).enumerate() {
                poll_stop(stop, index)?;
                let active = match presence {
                    Some(presence) => get(*presence)? == 1,
                    None => true,
                };
                if active {
                    active_starts.push(start);
                    active_durations.push(duration);
                }
            }
            no_overlap_holds(&active_starts, &active_durations, stop)?
        }
        IntGlobalConstraint::AlternativeChannel { shared_start, starts, presences, .. } => {
            let starts = values(starts)?;
            let presences = values(presences)?;
            alternative_channel_holds(get(*shared_start)?, &starts, &presences, stop)?
        }
        IntGlobalConstraint::Cumulative { starts, durations, demands, capacity } => {
            cumulative_holds(&values(starts)?, durations, demands, *capacity, stop)?
        }
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity } => {
            cumulative_holds(&values(starts)?, &values(durations)?, &values(demands)?, get(*capacity)?, stop)?
        }
        IntGlobalConstraint::BinPacking { items, sizes, capacities } => {
            let mut loads = vec![0i128; capacities.len()];
            let mut valid = true;
            for (index, (&bin, &size)) in values(items)?.iter().zip(sizes).enumerate() {
                poll_stop(stop, index)?;
                if let Ok(bin) = usize::try_from(bin) {
                    if let Some(load) = loads.get_mut(bin) {
                        *load += i128::from(size);
                    } else {
                        valid = false;
                    }
                } else {
                    valid = false;
                }
            }
            if valid {
                for (index, (load, capacity)) in loads.iter().zip(capacities).enumerate() {
                    poll_stop(stop, index)?;
                    if *load > i128::from(*capacity) {
                        valid = false;
                        break;
                    }
                }
            }
            valid
        }
        IntGlobalConstraint::BinLoads { items, sizes, loads: targets } => {
            let target_values = values(targets)?;
            let mut loads = vec![0i128; targets.len()];
            let mut valid = true;
            for (index, (&bin, &size)) in values(items)?.iter().zip(sizes).enumerate() {
                poll_stop(stop, index)?;
                if let Ok(bin) = usize::try_from(bin) {
                    if let Some(load) = loads.get_mut(bin) {
                        *load += i128::from(size);
                    } else {
                        valid = false;
                    }
                } else {
                    valid = false;
                }
            }
            if valid {
                for (index, (load, target)) in loads.iter().zip(target_values).enumerate() {
                    poll_stop(stop, index)?;
                    if *load != i128::from(target) {
                        valid = false;
                        break;
                    }
                }
            }
            valid
        }
        IntGlobalConstraint::Knapsack { variables, weights, profits, weight_relation, weight_limit, profit_relation, profit_limit } => {
            let assigned = values(variables)?;
            let weight = dot(&assigned, weights, stop)?;
            let profit = dot(&assigned, profits, stop)?;
            compare_i128(weight, *weight_relation, i128::from(*weight_limit))
                && compare_i128(profit, *profit_relation, i128::from(*profit_limit))
        }
        IntGlobalConstraint::ValuePrecedence { variables, values: ordered, covered } => {
            precedence_holds(&values(variables)?, ordered, *covered, stop)?
        }
    };
    check_stop(stop)?;
    Ok(holds)
}

const VERIFICATION_POLL_MASK: usize = 0xff;

fn poll_stop(stop: &AtomicBool, progress: usize) -> Result<(), SolveError> {
    if progress & VERIFICATION_POLL_MASK == 0 {
        check_stop(stop)?;
    }
    Ok(())
}

fn collect_values(
    variables: &[IntVarRef],
    get: &impl Fn(IntVarRef) -> Result<i64, SolveError>,
    stop: &AtomicBool,
) -> Result<Vec<i64>, SolveError> {
    let mut assigned = Vec::with_capacity(variables.len());
    for (index, variable) in variables.iter().enumerate() {
        poll_stop(stop, index)?;
        assigned.push(get(*variable)?);
    }
    check_stop(stop)?;
    Ok(assigned)
}

fn adjacent_values_hold(values: &[i64], predicate: impl Fn(i64, i64) -> bool, stop: &AtomicBool) -> Result<bool, SolveError> {
    for (index, pair) in values.windows(2).enumerate() {
        poll_stop(stop, index)?;
        if !predicate(pair[0], pair[1]) {
            check_stop(stop)?;
            return Ok(false);
        }
    }
    check_stop(stop)?;
    Ok(true)
}

fn equal_expected_values(values: &[i64], expected: &[i32], stop: &AtomicBool) -> Result<bool, SolveError> {
    if values.len() != expected.len() {
        check_stop(stop)?;
        return Ok(false);
    }
    for (index, (assigned, expected)) in values.iter().zip(expected).enumerate() {
        poll_stop(stop, index)?;
        if *assigned != i64::from(*expected) {
            check_stop(stop)?;
            return Ok(false);
        }
    }
    check_stop(stop)?;
    Ok(true)
}

fn extreme_holds(values: &[i64], target: i64, minimum: bool, stop: &AtomicBool) -> Result<bool, SolveError> {
    let Some((&first, rest)) = values.split_first() else {
        check_stop(stop)?;
        return Ok(false);
    };
    let mut extreme = first;
    for (index, value) in rest.iter().enumerate() {
        poll_stop(stop, index)?;
        if (minimum && *value < extreme) || (!minimum && *value > extreme) {
            extreme = *value;
        }
    }
    check_stop(stop)?;
    Ok(extreme == target)
}

fn cardinality_holds(
    assigned: &[i64],
    counted: &[i32],
    lower: &[i64],
    upper: &[i64],
    closed: bool,
    stop: &AtomicBool,
) -> Result<bool, SolveError> {
    let mut progress = 0usize;
    for ((counted, lower), upper) in counted.iter().zip(lower).zip(upper) {
        let mut count = 0i128;
        for assigned in assigned {
            poll_stop(stop, progress)?;
            progress = progress.wrapping_add(1);
            count += i128::from(*assigned == i64::from(*counted));
        }
        if !(i128::from(*lower)..=i128::from(*upper)).contains(&count) {
            check_stop(stop)?;
            return Ok(false);
        }
    }
    if closed {
        for assigned in assigned {
            let mut present = false;
            for counted in counted {
                poll_stop(stop, progress)?;
                progress = progress.wrapping_add(1);
                if *assigned == i64::from(*counted) {
                    present = true;
                    break;
                }
            }
            if !present {
                check_stop(stop)?;
                return Ok(false);
            }
        }
    }
    check_stop(stop)?;
    Ok(true)
}

fn table_holds(assigned: &[i64], tuples: &[Vec<i32>], positive: bool, stop: &AtomicBool) -> Result<bool, SolveError> {
    // XCSP's parser normalizes `*` to `i32::MIN`, the same reserved sentinel
    // used by the table propagators.  Keep that marker as a pattern here: the
    // canonical replay must apply exactly the same support/conflict semantics
    // as the engine whose candidate it verifies.
    const STAR: i32 = i32::MIN;

    let mut progress = 0usize;
    let mut present = false;
    for tuple in tuples {
        poll_stop(stop, progress)?;
        progress = progress.wrapping_add(1);
        if tuple.len() != assigned.len() {
            continue;
        }
        let mut matches = true;
        for (expected, assigned) in tuple.iter().zip(assigned) {
            poll_stop(stop, progress)?;
            progress = progress.wrapping_add(1);
            if *expected != STAR && i64::from(*expected) != *assigned {
                matches = false;
                break;
            }
        }
        if matches {
            present = true;
            break;
        }
    }
    check_stop(stop)?;
    Ok(present == positive)
}

fn regular_holds(assigned: &[i64], automaton: &crate::model::Automaton, stop: &AtomicBool) -> Result<bool, SolveError> {
    let mut state = automaton.start;
    let mut progress = 0usize;
    for symbol in assigned {
        let mut next_state = None;
        for (from, candidate, next) in &automaton.transitions {
            poll_stop(stop, progress)?;
            progress = progress.wrapping_add(1);
            if *from == state && i64::from(*candidate) == *symbol {
                next_state = Some(*next);
                break;
            }
        }
        let Some(next) = next_state else {
            check_stop(stop)?;
            return Ok(false);
        };
        state = next;
    }
    for (index, accepting) in automaton.accepting.iter().enumerate() {
        poll_stop(stop, index)?;
        if *accepting == state {
            check_stop(stop)?;
            return Ok(true);
        }
    }
    check_stop(stop)?;
    Ok(false)
}

fn mdd_holds(assigned: &[i64], mdd: &crate::model::Mdd, stop: &AtomicBool) -> Result<bool, SolveError> {
    let mut reachable = BTreeSet::from([0usize]);
    let mut progress = 0usize;
    for (layer, assigned) in mdd.layers.iter().zip(assigned) {
        let mut next = BTreeSet::new();
        for arc in layer {
            poll_stop(stop, progress)?;
            progress = progress.wrapping_add(1);
            if reachable.contains(&arc.from) && i64::from(arc.value) == *assigned {
                next.insert(arc.to);
            }
        }
        reachable = next;
    }
    check_stop(stop)?;
    Ok(!reachable.is_empty())
}

fn lex_holds(left: &[i64], right: &[i64], strict: bool, stop: &AtomicBool) -> Result<bool, SolveError> {
    let mut ordering = None;
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        poll_stop(stop, index)?;
        let pair_ordering = left.cmp(right);
        if !pair_ordering.is_eq() {
            ordering = Some(pair_ordering);
            break;
        }
    }
    let ordering = ordering.unwrap_or_else(|| left.len().cmp(&right.len()));
    check_stop(stop)?;
    Ok(ordering.is_lt() || (!strict && ordering.is_eq()))
}

fn lex_chain_holds(
    rows: &[Vec<IntVarRef>],
    strict: bool,
    get: &impl Fn(IntVarRef) -> Result<i64, SolveError>,
    stop: &AtomicBool,
) -> Result<bool, SolveError> {
    let Some((first, rest)) = rows.split_first() else {
        check_stop(stop)?;
        return Ok(true);
    };
    let mut previous = collect_values(first, get, stop)?;
    for (index, row) in rest.iter().enumerate() {
        poll_stop(stop, index)?;
        let current = collect_values(row, get, stop)?;
        if !lex_holds(&previous, &current, strict, stop)? {
            return Ok(false);
        }
        previous = current;
    }
    check_stop(stop)?;
    Ok(true)
}

fn channel_holds(left: &[i64], right: &[i64], stop: &AtomicBool) -> Result<bool, SolveError> {
    for (index, target) in left.iter().enumerate() {
        poll_stop(stop, index)?;
        if usize::try_from(*target).ok().and_then(|target| right.get(target)).is_none_or(|back| *back != index as i64) {
            check_stop(stop)?;
            return Ok(false);
        }
    }
    check_stop(stop)?;
    Ok(true)
}

fn set_subset_holds(subset: &[i32], superset: &[i32], stop: &AtomicBool) -> Result<bool, SolveError> {
    let mut superset_index = 0usize;
    for (index, subset) in subset.iter().enumerate() {
        poll_stop(stop, index.wrapping_add(superset_index))?;
        while superset_index < superset.len() && superset[superset_index] < *subset {
            poll_stop(stop, superset_index)?;
            superset_index += 1;
        }
        if superset.get(superset_index) != Some(subset) {
            check_stop(stop)?;
            return Ok(false);
        }
    }
    check_stop(stop)?;
    Ok(true)
}

fn set_disjoint_holds(left: &[i32], right: &[i32], stop: &AtomicBool) -> Result<bool, SolveError> {
    let (mut left_index, mut right_index, mut progress) = (0usize, 0usize, 0usize);
    while left_index < left.len() && right_index < right.len() {
        poll_stop(stop, progress)?;
        progress = progress.wrapping_add(1);
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                check_stop(stop)?;
                return Ok(false);
            }
        }
    }
    check_stop(stop)?;
    Ok(true)
}

fn compare_i128(left: i128, relation: Relation, right: i128) -> bool {
    match relation {
        Relation::Eq => left == right,
        Relation::Ne => left != right,
        Relation::Le => left <= right,
        Relation::Lt => left < right,
        Relation::Ge => left >= right,
        Relation::Gt => left > right,
    }
}

fn alternative_channel_holds(shared_start: i64, starts: &[i64], presences: &[i64], stop: &AtomicBool) -> Result<bool, SolveError> {
    let mut selected = None;
    for (index, presence) in presences.iter().enumerate() {
        poll_stop(stop, index)?;
        if *presence == 1 && selected.replace(index).is_some() {
            check_stop(stop)?;
            return Ok(false);
        }
    }
    let holds = selected.and_then(|selected| starts.get(selected)).is_some_and(|start| *start == shared_start);
    check_stop(stop)?;
    Ok(holds)
}

fn circuit_holds(successors: &[i64], stop: &AtomicBool) -> Result<bool, SolveError> {
    let length = successors.len();
    if length == 0 {
        check_stop(stop)?;
        return Ok(true);
    }
    let mut normalized = Vec::with_capacity(length);
    let mut unique = HashSet::with_capacity(length);
    for (index, successor) in successors.iter().enumerate() {
        poll_stop(stop, index)?;
        let Ok(successor) = usize::try_from(*successor) else {
            check_stop(stop)?;
            return Ok(false);
        };
        if successor >= length || !unique.insert(successor) {
            check_stop(stop)?;
            return Ok(false);
        }
        normalized.push(successor);
    }
    let mut seen = vec![false; length];
    let mut current = 0;
    for index in 0..length {
        poll_stop(stop, index)?;
        if seen[current] {
            check_stop(stop)?;
            return Ok(false);
        }
        seen[current] = true;
        current = normalized[current];
    }
    if current != 0 {
        check_stop(stop)?;
        return Ok(false);
    }
    for (index, seen) in seen.into_iter().enumerate() {
        poll_stop(stop, index)?;
        if !seen {
            check_stop(stop)?;
            return Ok(false);
        }
    }
    check_stop(stop)?;
    Ok(true)
}

fn no_overlap_holds(starts: &[i64], durations: &[i64], stop: &AtomicBool) -> Result<bool, SolveError> {
    let length = starts.len().min(durations.len());
    let mut progress = 0usize;
    for left in 0..length {
        let start = starts[left];
        let duration = durations[left];
        for right in left + 1..length {
            poll_stop(stop, progress)?;
            progress = progress.wrapping_add(1);
            let other_start = starts[right];
            let other_duration = durations[right];
            if i128::from(start) + i128::from(duration) > i128::from(other_start)
                && i128::from(other_start) + i128::from(other_duration) > i128::from(start)
            {
                check_stop(stop)?;
                return Ok(false);
            }
        }
    }
    check_stop(stop)?;
    Ok(true)
}

fn cumulative_holds(starts: &[i64], durations: &[i64], demands: &[i64], capacity: i64, stop: &AtomicBool) -> Result<bool, SolveError> {
    let mut points = BTreeSet::new();
    for (index, (&start, &duration)) in starts.iter().zip(durations).enumerate() {
        poll_stop(stop, index)?;
        points.insert(i128::from(start));
        points.insert(i128::from(start) + i128::from(duration));
    }
    let mut progress = 0usize;
    for time in points {
        let mut load = 0i128;
        for ((&start, &duration), &demand) in starts.iter().zip(durations).zip(demands) {
            poll_stop(stop, progress)?;
            progress = progress.wrapping_add(1);
            if i128::from(start) <= time && time < i128::from(start) + i128::from(duration) {
                load += i128::from(demand);
            }
        }
        if load > i128::from(capacity) {
            check_stop(stop)?;
            return Ok(false);
        }
    }
    check_stop(stop)?;
    Ok(true)
}

fn dot(values: &[i64], coefficients: &[i64], stop: &AtomicBool) -> Result<i128, SolveError> {
    let mut result = 0i128;
    for (index, (&value, &coefficient)) in values.iter().zip(coefficients).enumerate() {
        poll_stop(stop, index)?;
        result += i128::from(value) * i128::from(coefficient);
    }
    check_stop(stop)?;
    Ok(result)
}

fn precedence_holds(assigned: &[i64], ordered: &[i32], covered: bool, stop: &AtomicBool) -> Result<bool, SolveError> {
    if ordered.is_empty() {
        check_stop(stop)?;
        return Ok(true);
    }
    let mut positions = Vec::with_capacity(ordered.len());
    let mut progress = 0usize;
    for ordered in ordered {
        let mut position = None;
        for (index, assigned) in assigned.iter().enumerate() {
            poll_stop(stop, progress)?;
            progress = progress.wrapping_add(1);
            if *assigned == i64::from(*ordered) {
                position = Some(index);
                break;
            }
        }
        positions.push(position);
    }
    if covered {
        for (index, position) in positions.iter().enumerate() {
            poll_stop(stop, index)?;
            if position.is_none() {
                check_stop(stop)?;
                return Ok(false);
            }
        }
    }
    for (index, pair) in positions.windows(2).enumerate() {
        poll_stop(stop, index)?;
        let holds = match pair {
            [Some(left), Some(right)] => left < right,
            [None, Some(_)] => false,
            _ => true,
        };
        if !holds {
            check_stop(stop)?;
            return Ok(false);
        }
    }
    check_stop(stop)?;
    Ok(true)
}
