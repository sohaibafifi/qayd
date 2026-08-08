//! Canonical replay of complete collection assignments.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) const VERIFICATION_INTERRUPTED: &str = "canonical verification was interrupted";

use super::{eval_reduction_on_lists_interruptible, CollectionModel, CollectionSolution, GlobalConstraint, Op, Reduction, Resource};

/// Replay a collection solution independently of a search engine and return its
/// canonical lexicographic objective vector.
#[cfg(test)]
pub(crate) fn verify_collection_solution(model: &CollectionModel, solution: &CollectionSolution) -> Result<Vec<i64>, String> {
    verify_collection_solution_interruptible(model, solution, &AtomicBool::new(false))
}

#[cfg(test)]
thread_local! {
    static VERIFICATION_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[doc(hidden)]
#[cfg(test)]
pub(crate) fn audit_verification_calls() -> u64 {
    VERIFICATION_CALLS.get()
}

#[cfg(test)]
pub(crate) fn audit_record_final_verification_boundary() {
    VERIFICATION_CALLS.set(VERIFICATION_CALLS.get().saturating_add(1));
}

pub(crate) fn verify_collection_solution_interruptible(
    model: &CollectionModel,
    solution: &CollectionSolution,
    stop: &AtomicBool,
) -> Result<Vec<i64>, String> {
    check_stop(stop)?;
    if !model.validate_interruptible(stop)? {
        return Err(VERIFICATION_INTERRUPTED.to_string());
    }
    check_stop(stop)?;
    let objectives = if let Some(schedule) = &model.schedule {
        verify_schedule(schedule, solution, stop)?
    } else {
        verify_lists(model, solution, stop)?
    };
    if !solution.objectives.is_empty() && solution.objectives != objectives {
        return Err(format!("reported objectives {:?} differ from canonical objectives {:?}", solution.objectives, objectives));
    }
    Ok(objectives)
}

fn check_stop(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err(VERIFICATION_INTERRUPTED.to_string())
    } else {
        Ok(())
    }
}

fn verify_lists(model: &CollectionModel, solution: &CollectionSolution, stop: &AtomicBool) -> Result<Vec<i64>, String> {
    if solution.lists.len() != model.lists {
        return Err(format!("solution has {} lists, expected {}", solution.lists.len(), model.lists));
    }
    let mut universe = HashSet::with_capacity(model.items.len());
    for &item in &model.items {
        check_stop(stop)?;
        universe.insert(item);
    }
    let mut owner = HashMap::with_capacity(model.items.len());
    for (list_index, contents) in solution.lists.iter().enumerate() {
        check_stop(stop)?;
        for &item in contents {
            check_stop(stop)?;
            if !universe.contains(&item) {
                return Err(format!("solution contains item {item} outside the model universe"));
            }
            if owner.insert(item, list_index).is_some() {
                return Err(format!("solution assigns item {item} more than once"));
            }
        }
    }
    if owner.len() != model.items.len() {
        let mut missing = None;
        for &item in &model.items {
            check_stop(stop)?;
            if !owner.contains_key(&item) {
                missing = Some(item);
                break;
            }
        }
        let missing = missing.unwrap_or_default();
        return Err(format!("solution does not assign item {missing}"));
    }

    for constraint in &model.constraints {
        check_stop(stop)?;
        let value = eval_reduction_on_lists_interruptible(&constraint.reduction, &solution.lists, stop)
            .map_err(|()| VERIFICATION_INTERRUPTED.to_string())?
            .ok_or_else(|| "a collection constraint is undefined on the candidate".to_string())?;
        let satisfied = match constraint.op {
            Op::Le => value <= constraint.rhs,
            Op::Ge => value >= constraint.rhs,
            Op::Eq => value == constraint.rhs,
        };
        if !satisfied {
            return Err(format!(
                "collection constraint evaluates to {value}, relation {:?} {} is false",
                op_name(constraint.op),
                constraint.rhs
            ));
        }
    }
    for global in &model.globals {
        check_stop(stop)?;
        verify_global(global, &owner, stop)?;
    }

    let objectives = verify_list_objectives(model, &solution.lists, stop)?;
    check_stop(stop)?;
    Ok(objectives)
}

fn op_name(op: Op) -> &'static str {
    match op {
        Op::Le => "<=",
        Op::Ge => ">=",
        Op::Eq => "==",
    }
}

fn verify_global(global: &GlobalConstraint, owner: &HashMap<i32, usize>, stop: &AtomicBool) -> Result<(), String> {
    let list = |item: i32| owner.get(&item).copied().ok_or_else(|| format!("global constraint references unassigned item {item}"));
    let satisfied = match global {
        GlobalConstraint::ListLe { before, after } => list(*before)? <= list(*after)?,
        GlobalConstraint::SameList { a, b } => list(*a)? == list(*b)?,
        GlobalConstraint::DifferentList { a, b } => list(*a)? != list(*b)?,
        GlobalConstraint::AllSameList { items } => {
            let first = list(items[0])?;
            let mut same = true;
            for &item in items.iter().skip(1) {
                check_stop(stop)?;
                if list(item)? != first {
                    same = false;
                    break;
                }
            }
            same
        }
        GlobalConstraint::AllDifferentLists { items } => {
            let mut values = HashSet::with_capacity(items.len());
            let mut different = true;
            for &item in items.iter() {
                check_stop(stop)?;
                if !values.insert(list(item)?) {
                    different = false;
                    break;
                }
            }
            different
        }
        GlobalConstraint::ListDistance { a, b, min, max } => {
            let distance = list(*a)?.abs_diff(list(*b)?);
            *min <= distance && distance <= *max
        }
    };
    if satisfied {
        Ok(())
    } else {
        Err("cross-list global constraint is violated".to_string())
    }
}

fn verify_list_objectives(model: &CollectionModel, lists: &[Vec<i32>], stop: &AtomicBool) -> Result<Vec<i64>, String> {
    let mut objectives = Vec::with_capacity(model.objectives.len());
    for tier in &model.objectives {
        check_stop(stop)?;
        let mut value = verify_objective_terms(&tier.terms, lists, tier.minimize, stop)?;
        if let Some(max_terms) = &tier.max_terms {
            for term in max_terms {
                check_stop(stop)?;
                let mut component = None;
                for group in &term.groups {
                    let group = verify_objective_terms(group, lists, tier.minimize, stop)?;
                    component = Some(component.map_or(group, |current: i64| current.max(group)));
                }
                value = value.saturating_add(component.unwrap_or(0).saturating_mul(term.coeff));
            }
        }
        objectives.push(value);
    }
    Ok(objectives)
}

fn verify_objective_terms(terms: &[Reduction], lists: &[Vec<i32>], minimize: bool, stop: &AtomicBool) -> Result<i64, String> {
    let undefined = if minimize { i64::MAX / 4 } else { i64::MIN / 4 };
    let mut value = 0i64;
    for term in terms {
        check_stop(stop)?;
        let term = eval_reduction_on_lists_interruptible(term, lists, stop)
            .map_err(|()| VERIFICATION_INTERRUPTED.to_string())?
            .unwrap_or(undefined);
        value = value.saturating_add(term);
    }
    Ok(value)
}

fn verify_schedule(schedule: &super::Schedule, solution: &CollectionSolution, stop: &AtomicBool) -> Result<Vec<i64>, String> {
    let n = schedule.intervals.len();
    if solution.starts.len() != n || solution.presences.len() != n || solution.machines.len() != n || solution.modes.len() != n {
        return Err(format!(
            "schedule assignment dimensions are starts={}, presences={}, machines={}, modes={}, expected {n}",
            solution.starts.len(),
            solution.presences.len(),
            solution.machines.len(),
            solution.modes.len()
        ));
    }

    let mut durations = vec![0i64; n];
    let mut makespan = 0i64;
    for (index, interval) in schedule.intervals.iter().enumerate() {
        check_stop(stop)?;
        let present = solution.presences[index];
        if !present {
            if !interval.optional {
                return Err(format!("mandatory interval {index} is absent"));
            }
            if solution.machines[index] != -1 {
                return Err(format!("absent interval {index} has a selected machine"));
            }
            if solution.modes[index].is_some() {
                return Err(format!("absent interval {index} has a selected mode"));
            }
            continue;
        }
        let duration = if interval.modes.is_empty() {
            if solution.machines[index] != -1 {
                return Err(format!("fixed interval {index} unexpectedly selects a machine"));
            }
            if solution.modes[index].is_some() {
                return Err(format!("fixed interval {index} unexpectedly selects a mode"));
            }
            interval.duration
        } else {
            let machine = usize::try_from(solution.machines[index]).map_err(|_| format!("moded interval {index} has no machine"))?;
            let mode = select_mode(interval, index, machine, solution.modes[index])?;
            let start = solution.starts[index];
            if !(mode.start_window.0..=mode.start_window.1).contains(&start) {
                return Err(format!(
                    "moded interval {index} starts at {start}, outside mode window {}..{}",
                    mode.start_window.0, mode.start_window.1
                ));
            }
            mode.duration
        };
        durations[index] = duration;
        let start = solution.starts[index];
        let end = start.checked_add(duration).ok_or_else(|| format!("interval {index} end overflows i64"))?;
        if interval.modes.is_empty() && (start < 0 || end > interval.horizon) {
            return Err(format!("interval {index} lies outside [0, {}]", interval.horizon));
        }
        if end > interval.horizon {
            return Err(format!("interval {index} ends beyond horizon {}", interval.horizon));
        }
        makespan = makespan.max(end);
    }

    for &(before, after) in &schedule.precedences {
        check_stop(stop)?;
        if solution.presences[before]
            && solution.presences[after]
            && solution.starts[before].saturating_add(durations[before]) > solution.starts[after]
        {
            return Err(format!("precedence {before} -> {after} is violated"));
        }
    }
    for resource in &schedule.resources {
        check_stop(stop)?;
        match resource {
            Resource::NoOverlap(intervals) => verify_no_overlap(intervals, solution, &durations, stop)?,
            Resource::MachineNoOverlap => {
                for left in 0..n {
                    check_stop(stop)?;
                    for right in (left + 1)..n {
                        if solution.machines[left] >= 0 && solution.machines[left] == solution.machines[right] {
                            verify_pair(left, right, solution, &durations)?;
                        }
                    }
                }
            }
            Resource::Cumulative { demands, capacity } => verify_cumulative(demands, *capacity, solution, &durations, stop)?,
        }
    }
    Ok(schedule.minimize_makespan.then_some(makespan).into_iter().collect())
}

fn select_mode(
    interval: &super::IntervalVar,
    interval_index: usize,
    machine: usize,
    reference: Option<usize>,
) -> Result<&super::Mode, String> {
    let selected = if let Some(reference) = reference {
        let matching = interval.modes.iter().filter(|mode| mode.reference == Some(reference)).collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(format!("moded interval {interval_index} references semantic mode {reference}, found {} matches", matching.len()));
        }
        matching[0]
    } else {
        if interval.modes.iter().any(|mode| mode.reference.is_some()) {
            return Err(format!("moded interval {interval_index} omits its semantic mode identity"));
        }
        let matching = interval.modes.iter().filter(|mode| mode.machine == machine).collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(format!(
                "legacy moded interval {interval_index} must identify exactly one mode by machine; found {} matches",
                matching.len()
            ));
        }
        matching[0]
    };
    if selected.machine != machine {
        return Err(format!(
            "moded interval {interval_index} reports machine {machine}, but semantic mode {:?} uses machine {}",
            selected.reference, selected.machine
        ));
    }
    Ok(selected)
}

fn verify_no_overlap(intervals: &[usize], solution: &CollectionSolution, durations: &[i64], stop: &AtomicBool) -> Result<(), String> {
    for left in 0..intervals.len() {
        check_stop(stop)?;
        for right in (left + 1)..intervals.len() {
            verify_pair(intervals[left], intervals[right], solution, durations)?;
        }
    }
    Ok(())
}

fn verify_pair(left: usize, right: usize, solution: &CollectionSolution, durations: &[i64]) -> Result<(), String> {
    if !solution.presences[left] || !solution.presences[right] {
        return Ok(());
    }
    let left_end = solution.starts[left].saturating_add(durations[left]);
    let right_end = solution.starts[right].saturating_add(durations[right]);
    if solution.starts[left] < right_end && solution.starts[right] < left_end {
        Err(format!("intervals {left} and {right} overlap"))
    } else {
        Ok(())
    }
}

fn verify_cumulative(
    demands: &[(usize, i64)],
    capacity: i64,
    solution: &CollectionSolution,
    durations: &[i64],
    stop: &AtomicBool,
) -> Result<(), String> {
    let mut events = Vec::with_capacity(demands.len() * 2);
    for &(interval, _) in demands {
        check_stop(stop)?;
        if solution.presences[interval] {
            events.push(solution.starts[interval]);
            events.push(solution.starts[interval].saturating_add(durations[interval]));
        }
    }
    events.sort_unstable();
    events.dedup();
    for &time in &events {
        check_stop(stop)?;
        let mut usage = 0i128;
        for &(interval, demand) in demands {
            check_stop(stop)?;
            if solution.presences[interval]
                && solution.starts[interval] <= time
                && time < solution.starts[interval].saturating_add(durations[interval])
            {
                usage += i128::from(demand);
            }
        }
        if usage > i128::from(capacity) {
            return Err(format!("cumulative usage {usage} exceeds capacity {capacity} at time {time}"));
        }
    }
    Ok(())
}
