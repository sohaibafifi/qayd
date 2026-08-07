//! Canonical semantic solution replay.

use std::collections::{BTreeSet, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::model::{
    CompiledCollection, Constraint, IndependentComponent, IndependentDecomposition, IntGlobalConstraint, Model, Objective,
    PartitionCoverage, Relation,
};
use crate::model::{IntVarRef, SetVarRef};

use super::{Assignment, SolveError};

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
    match model.independent_family_components_interruptible(stop) {
        IndependentDecomposition::Components(components) => {
            return verify_decomposed_assignment(model, assignment, claimed_objectives, components, stop);
        }
        IndependentDecomposition::Interrupted => {
            return Err(SolveError::Interrupted("canonical semantic decomposition was interrupted".to_string()));
        }
        IndependentDecomposition::NotApplicable => {}
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
        check_stop(stop)?;
        let value =
            assignment.integers[index].ok_or_else(|| SolveError::InvalidResult(format!("integer variable {index} is unassigned")))?;
        if !domain.contains(value) {
            return Err(SolveError::InvalidResult(format!("integer variable {index} has value {value} outside its domain")));
        }
    }
    for (index, declaration) in model.sets().iter().enumerate() {
        check_stop(stop)?;
        let value = &assignment.sets[index];
        if value.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SolveError::InvalidResult(format!("set variable {index} is not strictly sorted and duplicate-free")));
        }
        let value = value.iter().copied().collect::<BTreeSet<_>>();
        if !declaration.required.is_subset(&value) || !value.is_subset(&declaration.possible) {
            return Err(SolveError::InvalidResult(format!("set variable {index} violates its lower or upper bound")));
        }
    }
    for (index, constraint) in model.constraints().iter().enumerate() {
        check_stop(stop)?;
        if !constraint_holds(model, constraint, assignment, &integer)? {
            return Err(SolveError::InvalidResult(format!("constraint {index} is violated")));
        }
    }
    let objectives = model
        .objectives()
        .iter()
        .map(|objective| match objective {
            Objective::IntExpr { expr, .. } => {
                expr.evaluate(&integer).ok_or_else(|| SolveError::InvalidResult("integer objective is undefined or overflows".to_string()))
            }
            Objective::ListTerms { .. } | Objective::Makespan { .. } => {
                Err(SolveError::InvalidResult("non-integer objective in an integer assignment".to_string()))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if objectives != claimed_objectives {
        return Err(SolveError::InvalidResult(format!(
            "objective mismatch: engine reported {claimed_objectives:?}, canonical replay produced {objectives:?}"
        )));
    }
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
    if components.iter().any(|component| {
        component.model.int_vars().len() == model.int_vars().len()
            && component.model.sets().len() == model.sets().len()
            && component.model.lists().len() == model.lists().len()
            && component.model.intervals().len() == model.intervals().len()
            && component.model.constraints().len() == model.constraints().len()
            && component.model.objectives().len() == model.objectives().len()
    }) {
        return Err(SolveError::InvalidResult("semantic decomposition contains a component identical to its parent".to_string()));
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
        )?;
        let local_claims = component
            .objective_tiers
            .iter()
            .map(|&tier| {
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
                Ok(claimed)
            })
            .collect::<Result<Vec<_>, _>>()?;
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
        if let Some(index) = coverage.iter().position(|covered| !covered) {
            return Err(SolveError::InvalidResult(format!("semantic {kind} {index} has no decomposition owner")));
        }
    }
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
    for &(original, local) in &component.integers {
        mark_projection(integer_coverage, &mut local_integers, original, local, "integer")?;
        projected.integers[local] = assignment.integers[original];
    }
    for &(original, local) in &component.sets {
        mark_projection(set_coverage, &mut local_sets, original, local, "set")?;
        projected.sets[local] = assignment.sets[original].clone();
    }
    for &(original, local) in &component.lists {
        mark_projection(list_coverage, &mut local_lists, original, local, "list")?;
        projected.lists[local] = assignment.lists[original].clone();
    }
    for &(original, local) in &component.intervals {
        mark_projection(interval_coverage, &mut local_intervals, original, local, "interval")?;
        let mut value = assignment.intervals[original];
        if let Some(original_mode) = value.mode {
            value.mode = Some(
                component.interval_modes.iter().find_map(|&(original, local)| (original == original_mode).then_some(local)).ok_or_else(
                    || {
                        SolveError::InvalidResult(format!(
                            "semantic interval {original} selects mode {original_mode} outside its decomposition component"
                        ))
                    },
                )?,
            );
        }
        projected.intervals[local] = value;
    }
    for &(original, local) in &component.interval_modes {
        mark_projection(mode_coverage, &mut local_modes, original, local, "interval mode")?;
    }
    for (kind, coverage) in [
        ("integer", local_integers.as_slice()),
        ("set", local_sets.as_slice()),
        ("list", local_lists.as_slice()),
        ("interval", local_intervals.as_slice()),
        ("interval mode", local_modes.as_slice()),
    ] {
        if let Some(local) = coverage.iter().position(|covered| !covered) {
            return Err(SolveError::InvalidResult(format!("component-local {kind} {local} has no semantic owner")));
        }
    }
    Ok(projected)
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
    let solution = if physical.schedule.is_some() {
        if assignment.intervals.len() != model.intervals().len() || !assignment.lists.is_empty() {
            return Err(SolveError::InvalidResult("schedule assignment has the wrong arena shape".to_string()));
        }
        crate::model::list::CollectionSolution {
            lists: Vec::new(),
            objectives: claimed_objectives.to_vec(),
            feasible: true,
            starts: assignment.intervals.iter().map(|value| value.start.unwrap_or(0)).collect(),
            presences: assignment.intervals.iter().map(|value| value.present).collect(),
            machines: assignment
                .intervals
                .iter()
                .map(|value| value.machine.and_then(|machine| i64::try_from(machine).ok()).unwrap_or(-1))
                .collect(),
            modes: assignment.intervals.iter().map(|value| value.mode).collect(),
            bound: None,
        }
    } else {
        if assignment.lists.len() != model.lists().len() || !assignment.intervals.is_empty() {
            return Err(SolveError::InvalidResult("list assignment has the wrong arena shape".to_string()));
        }
        crate::model::list::CollectionSolution {
            lists: assignment.lists.clone(),
            objectives: claimed_objectives.to_vec(),
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
) -> Result<bool, SolveError> {
    Ok(match constraint {
        Constraint::Intension(expr) => expr.evaluate(integer).is_some_and(|value| value != 0),
        Constraint::Selected { selector, constraint } => {
            integer(*selector) != Some(1) || constraint_holds(model, constraint, assignment, integer)?
        }
        Constraint::Linear { terms, relation, rhs } => {
            let left = terms
                .iter()
                .try_fold(0i128, |sum, (coefficient, variable)| Some(sum + i128::from(*coefficient) * i128::from(integer(*variable)?)));
            left.is_some_and(|left| compare_i128(left, *relation, i128::from(*rhs)))
        }
        Constraint::Clause(literals) => {
            literals.iter().any(|literal| integer(literal.variable).is_some_and(|value| (value != 0) == literal.positive))
        }
        Constraint::IntegerGlobal(global) => global_holds(global, integer)?,
        Constraint::SetSubset { subset, superset } => set(assignment, *subset).is_subset(&set(assignment, *superset)),
        Constraint::SetDisjoint { left, right } => set(assignment, *left).is_disjoint(&set(assignment, *right)),
        Constraint::SetCardinality { set: reference, min, max } => (*min..=*max).contains(&set(assignment, *reference).len()),
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
    })
}

fn global_holds(global: &IntGlobalConstraint, value: &impl Fn(IntVarRef) -> Option<i64>) -> Result<bool, SolveError> {
    let get = |reference: IntVarRef| {
        value(reference).ok_or_else(|| SolveError::InvalidResult(format!("integer variable {} is unassigned", reference.0)))
    };
    let values = |variables: &[IntVarRef]| variables.iter().map(|variable| get(*variable)).collect::<Result<Vec<_>, _>>();
    Ok(match global {
        IntGlobalConstraint::AllDifferent { variables, except } => {
            let values = values(variables)?;
            let except = except.iter().map(|value| i64::from(*value)).collect::<HashSet<_>>();
            values.iter().enumerate().all(|(index, value)| except.contains(value) || !values[index + 1..].contains(value))
        }
        IntGlobalConstraint::AllEqual(variables) => values(variables)?.windows(2).all(|pair| pair[0] == pair[1]),
        IntGlobalConstraint::Ordered { variables, relation } => {
            values(variables)?.windows(2).all(|pair| compare_i128(i128::from(pair[0]), *relation, i128::from(pair[1])))
        }
        IntGlobalConstraint::Instantiation { variables, values: expected } => {
            values(variables)?.iter().copied().eq(expected.iter().map(|value| i64::from(*value)))
        }
        IntGlobalConstraint::Minimum { target, variables } => values(variables)?.into_iter().min() == Some(get(*target)?),
        IntGlobalConstraint::Maximum { target, variables } => values(variables)?.into_iter().max() == Some(get(*target)?),
        IntGlobalConstraint::Element { array, index, value: target } => usize::try_from(get(*index)?)
            .ok()
            .and_then(|index| array.get(index))
            .is_some_and(|variable| get(*variable).ok() == Some(get(*target).unwrap_or(i64::MIN))),
        IntGlobalConstraint::ElementConst { array, index, value: target } => usize::try_from(get(*index)?)
            .ok()
            .and_then(|index| array.get(index))
            .is_some_and(|value| i64::from(*value) == get(*target).unwrap_or(i64::MIN)),
        IntGlobalConstraint::Count { variables, value: expected, relation, count } => {
            let actual = values(variables)?.iter().filter(|value| **value == i64::from(*expected)).count() as i128;
            compare_i128(actual, *relation, i128::from(*count))
        }
        IntGlobalConstraint::Cardinality { variables, values: counted, lower, upper, closed } => {
            let assigned = values(variables)?;
            let bounds_hold = counted.iter().zip(lower).zip(upper).all(|((&counted, &lower), &upper)| {
                let count = assigned.iter().filter(|value| **value == i64::from(counted)).count() as i128;
                (i128::from(lower)..=i128::from(upper)).contains(&count)
            });
            bounds_hold && (!closed || assigned.iter().all(|value| counted.iter().any(|counted| i64::from(*counted) == *value)))
        }
        IntGlobalConstraint::NValues { variables, relation, count } => {
            let distinct = values(variables)?.into_iter().collect::<BTreeSet<_>>().len() as i128;
            compare_i128(distinct, *relation, i128::from(*count))
        }
        IntGlobalConstraint::Table { variables, tuples, positive } => {
            let assigned = values(variables)?;
            let present = tuples.iter().any(|tuple| tuple.iter().map(|value| i64::from(*value)).eq(assigned.iter().copied()));
            present == *positive
        }
        IntGlobalConstraint::Regular { variables, automaton } => {
            let mut state = automaton.start;
            let mut valid = true;
            for symbol in values(variables)? {
                if let Some((_, _, next)) =
                    automaton.transitions.iter().find(|(from, candidate, _)| *from == state && i64::from(*candidate) == symbol)
                {
                    state = *next;
                } else {
                    valid = false;
                    break;
                }
            }
            valid && automaton.accepting.contains(&state)
        }
        IntGlobalConstraint::Mdd { variables, mdd } => {
            let mut reachable = BTreeSet::from([0usize]);
            for (layer, assigned) in mdd.layers.iter().zip(values(variables)?) {
                reachable = layer
                    .iter()
                    .filter(|arc| reachable.contains(&arc.from) && i64::from(arc.value) == assigned)
                    .map(|arc| arc.to)
                    .collect();
            }
            !reachable.is_empty()
        }
        IntGlobalConstraint::Lex { left, right, strict } => lex_holds(&values(left)?, &values(right)?, *strict),
        IntGlobalConstraint::LexChain { rows, strict } => {
            let rows = rows.iter().map(|row| values(row)).collect::<Result<Vec<_>, _>>()?;
            rows.windows(2).all(|rows| lex_holds(&rows[0], &rows[1], *strict))
        }
        IntGlobalConstraint::Channel { left, right } => {
            let left = values(left)?;
            let right = values(right)?;
            left.iter().enumerate().all(|(index, target)| {
                usize::try_from(*target).ok().and_then(|target| right.get(target)).is_some_and(|back| *back == index as i64)
            })
        }
        IntGlobalConstraint::Circuit { successors, .. } => circuit_holds(&values(successors)?),
        IntGlobalConstraint::NoOverlap { starts, durations } => no_overlap_holds(&values(starts)?, durations),
        IntGlobalConstraint::OptionalNoOverlap { starts, durations, presences } => {
            let starts = values(starts)?;
            let mut active_starts = Vec::new();
            let mut active_durations = Vec::new();
            for ((&start, &duration), presence) in starts.iter().zip(durations).zip(presences) {
                let active = match presence {
                    Some(presence) => get(*presence)? == 1,
                    None => true,
                };
                if active {
                    active_starts.push(start);
                    active_durations.push(duration);
                }
            }
            no_overlap_holds(&active_starts, &active_durations)
        }
        IntGlobalConstraint::AlternativeChannel { shared_start, starts, presences, .. } => {
            let starts = values(starts)?;
            let presences = values(presences)?;
            let selected = presences.iter().enumerate().filter(|(_, presence)| **presence == 1).collect::<Vec<_>>();
            selected.len() == 1 && starts[selected[0].0] == get(*shared_start)?
        }
        IntGlobalConstraint::Cumulative { starts, durations, demands, capacity } => {
            cumulative_holds(&values(starts)?, durations, demands, *capacity)
        }
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity } => {
            cumulative_holds(&values(starts)?, &values(durations)?, &values(demands)?, get(*capacity)?)
        }
        IntGlobalConstraint::BinPacking { items, sizes, capacities } => {
            let mut loads = vec![0i128; capacities.len()];
            let mut valid = true;
            for (&bin, &size) in values(items)?.iter().zip(sizes) {
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
            valid && loads.iter().zip(capacities).all(|(load, capacity)| *load <= i128::from(*capacity))
        }
        IntGlobalConstraint::BinLoads { items, sizes, loads: targets } => {
            let target_values = values(targets)?;
            let mut loads = vec![0i128; targets.len()];
            let mut valid = true;
            for (&bin, &size) in values(items)?.iter().zip(sizes) {
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
            valid && loads.iter().zip(target_values).all(|(load, target)| *load == i128::from(target))
        }
        IntGlobalConstraint::Knapsack { variables, weights, profits, weight_relation, weight_limit, profit_relation, profit_limit } => {
            let assigned = values(variables)?;
            let weight = dot(&assigned, weights);
            let profit = dot(&assigned, profits);
            compare_i128(weight, *weight_relation, i128::from(*weight_limit))
                && compare_i128(profit, *profit_relation, i128::from(*profit_limit))
        }
        IntGlobalConstraint::ValuePrecedence { variables, values: ordered, covered } => {
            precedence_holds(&values(variables)?, ordered, *covered)
        }
    })
}

fn set(assignment: &Assignment, reference: SetVarRef) -> BTreeSet<i32> {
    assignment.sets[reference.0].iter().copied().collect()
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

fn lex_holds(left: &[i64], right: &[i64], strict: bool) -> bool {
    left < right || (!strict && left == right)
}

fn circuit_holds(successors: &[i64]) -> bool {
    let length = successors.len();
    if length == 0 {
        return true;
    }
    let Some(successors) =
        successors.iter().map(|value| usize::try_from(*value).ok().filter(|value| *value < length)).collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    if successors.iter().copied().collect::<BTreeSet<_>>().len() != length {
        return false;
    }
    let mut seen = vec![false; length];
    let mut current = 0;
    for _ in 0..length {
        if seen[current] {
            return false;
        }
        seen[current] = true;
        current = successors[current];
    }
    current == 0 && seen.into_iter().all(|value| value)
}

fn no_overlap_holds(starts: &[i64], durations: &[i64]) -> bool {
    starts.iter().zip(durations).enumerate().all(|(left, (&start, &duration))| {
        starts.iter().zip(durations).skip(left + 1).all(|(&other_start, &other_duration)| {
            i128::from(start) + i128::from(duration) <= i128::from(other_start)
                || i128::from(other_start) + i128::from(other_duration) <= i128::from(start)
        })
    })
}

fn cumulative_holds(starts: &[i64], durations: &[i64], demands: &[i64], capacity: i64) -> bool {
    let points = starts
        .iter()
        .zip(durations)
        .flat_map(|(&start, &duration)| [i128::from(start), i128::from(start) + i128::from(duration)])
        .collect::<BTreeSet<_>>();
    points.into_iter().all(|time| {
        starts
            .iter()
            .zip(durations)
            .zip(demands)
            .filter(|((&start, &duration), _)| i128::from(start) <= time && time < i128::from(start) + i128::from(duration))
            .map(|(_, demand)| i128::from(*demand))
            .sum::<i128>()
            <= i128::from(capacity)
    })
}

fn dot(values: &[i64], coefficients: &[i64]) -> i128 {
    values.iter().zip(coefficients).map(|(&value, &coefficient)| i128::from(value) * i128::from(coefficient)).sum()
}

fn precedence_holds(assigned: &[i64], ordered: &[i32], covered: bool) -> bool {
    if ordered.is_empty() {
        return true;
    }
    let positions = ordered.iter().map(|value| assigned.iter().position(|assigned| *assigned == i64::from(*value))).collect::<Vec<_>>();
    if covered && positions.iter().any(Option::is_none) {
        return false;
    }
    positions.windows(2).all(|pair| match pair {
        [Some(left), Some(right)] => left < right,
        [None, Some(_)] => false,
        _ => true,
    })
}
