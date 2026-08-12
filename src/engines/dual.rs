//! Certified objective bounds for collection models.
//!
//! Every method in this module is a relaxation. It may return no bound, but it
//! must never return a heuristic estimate as a certificate.

#[cfg(test)]
use std::cell::Cell;
#[cfg(feature = "lp-relaxation")]
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(feature = "lp-relaxation")]
use std::time::Instant;

use crate::engines::ls::lists::eval::eval_expr;
use crate::model::list::{
    BoundReport, CollectionModel, CollectionSolution, Expr, ExprArena, ExprId, GlobalConstraint, Iterable, Op, ReduceOp, Reduction,
    Resource,
};
use crate::orchestrator::LinearControls;
#[cfg(feature = "lp-relaxation")]
use crate::search::LinearModelStatus;
use crate::search::SolveStats;

const MAX_COLUMN_GENERATION_CUSTOMERS: usize = 16;
const COLUMN_DUAL_SCALE: i128 = 1 << 12;
const MAX_EXACT_PACKING_ITEMS: usize = 35;
const PACKING_NODE_LIMIT: u64 = 20_000;
const MAX_ENERGY_INTERVAL_PAIRS: usize = 50_000;
const MAX_VRPTW_METRIC_NODES: usize = 128;
const MAX_ROUTING_RELAXATION_NODES: usize = 2_048;
const INF: i64 = i64::MAX;
#[cfg(feature = "lp-relaxation")]
const MAX_ROUTE_MASTER_ROUNDS: usize = 64;
#[cfg(feature = "lp-relaxation")]
const MAX_NG_ROUTE_BASE_STATES: usize = 2_000_000;
#[cfg(feature = "lp-relaxation")]
const MAX_NG_ROUTE_BASE_TRANSITIONS: usize = 200_000_000;
#[cfg(feature = "lp-relaxation")]
const ROUTE_DUAL_PROJECTION_SCALE: usize = 1 << 10;

/// A certified bound before a primal solution is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualBound {
    pub value: i64,
    pub method: &'static str,
    pub stats: SolveStats,
}

/// Compute the strongest supported bound on the primary objective.
pub fn compute(model: &CollectionModel, stop: &AtomicBool) -> Option<DualBound> {
    compute_internal(model, None, stop)
}

/// Compute structural bounds and the optional certified route-master LP.
/// Collection engines call this variant so the same explicit controls used by
/// integer CP also govern the routing relaxation.
pub fn compute_with_linear(model: &CollectionModel, controls: LinearControls, stop: &AtomicBool) -> Option<DualBound> {
    compute_internal(model, Some(controls), stop)
}

fn compute_internal(model: &CollectionModel, controls: Option<LinearControls>, stop: &AtomicBool) -> Option<DualBound> {
    if stop.load(Ordering::Relaxed) {
        return None;
    }
    if let Some(schedule) = &model.schedule {
        return schedule_bound(schedule, stop);
    }
    let tier = model.objectives.first()?;
    let mut best = used_list_bound(model, stop).or_else(|| additive_assignment_bound(model, stop));
    if tier.minimize {
        if let Some(routing) = RoutingRelaxation::from_model(model, stop) {
            best = stronger(best, routing.assignment_bound(stop), true);
            if routing.routes == 1 && routing.symmetric {
                best = stronger(best, routing.held_karp_bound(stop), true);
            }
            best = stronger(best, routing.column_generation_bound(stop), true);
            if let Some(controls) = controls {
                best = stronger(best, routing.route_master_lp_bound(controls, stop), true);
            }
        }
    }
    best
}

/// Attach a candidate dual to a feasible solution, rejecting any orientation
/// mismatch instead of reporting a negative or unsound gap.
pub fn attach(model: &CollectionModel, solution: &mut CollectionSolution, dual: Option<DualBound>) {
    let Some(primal) = solution.feasible.then(|| solution.objectives.first().copied()).flatten() else {
        solution.bound = None;
        return;
    };
    let minimizing = model.schedule.as_ref().is_some_and(|schedule| schedule.minimize_makespan)
        || model.objectives.first().is_some_and(|tier| tier.minimize);
    solution.bound = dual.and_then(|dual| make_report(primal, dual.value, minimizing, dual.method.to_owned()));
}

/// Attach a previously computed certificate to a new incumbent.
pub fn attach_value(model: &CollectionModel, solution: &mut CollectionSolution, dual: i64, method: String) {
    let Some(primal) = solution.feasible.then(|| solution.objectives.first().copied()).flatten() else {
        solution.bound = None;
        return;
    };
    let minimizing = model.schedule.as_ref().is_some_and(|schedule| schedule.minimize_makespan)
        || model.objectives.first().is_some_and(|tier| tier.minimize);
    solution.bound = make_report(primal, dual, minimizing, method);
}

/// Exact completion closes the gap independently of the structural relaxation.
pub fn attach_exact(solution: &mut CollectionSolution, method: &'static str) {
    let Some(primal) = solution.feasible.then(|| solution.objectives.first().copied()).flatten() else {
        solution.bound = None;
        return;
    };
    solution.bound = make_report(primal, primal, true, method.to_owned());
}

/// Build a checked report from compatible primal and dual values.
pub fn make_report(primal: i64, dual: i64, minimizing: bool, method: String) -> Option<BoundReport> {
    let gap = if minimizing { i128::from(primal) - i128::from(dual) } else { i128::from(dual) - i128::from(primal) };
    if gap < 0 {
        return None;
    }
    let absolute_gap = u64::try_from(gap).unwrap_or(u64::MAX);
    let scale = primal.unsigned_abs().max(dual.unsigned_abs()).max(1) as f64;
    Some(BoundReport { dual, absolute_gap, relative_gap: absolute_gap as f64 / scale, method })
}

fn stronger(current: Option<DualBound>, candidate: Option<DualBound>, minimizing: bool) -> Option<DualBound> {
    match (current, candidate) {
        (None, other) | (other, None) => other,
        (Some(mut current), Some(mut candidate)) => {
            if (minimizing && candidate.value > current.value) || (!minimizing && candidate.value < current.value) {
                merge_lp_stats(&mut candidate.stats, current.stats);
                Some(candidate)
            } else {
                merge_lp_stats(&mut current.stats, candidate.stats);
                Some(current)
            }
        }
    }
}

fn merge_lp_stats(total: &mut SolveStats, part: SolveStats) {
    total.lp_rows = total.lp_rows.saturating_add(part.lp_rows);
    if total.lp_model_status == crate::search::LinearModelStatus::NotAttempted {
        total.lp_model_status = part.lp_model_status;
    }
    total.lp_variables = total.lp_variables.saturating_add(part.lp_variables);
    total.lp_columns = total.lp_columns.saturating_add(part.lp_columns);
    total.lp_covered_variables = total.lp_covered_variables.saturating_add(part.lp_covered_variables);
    total.lp_objective_variables = total.lp_objective_variables.saturating_add(part.lp_objective_variables);
    total.lp_objective_covered_variables = total.lp_objective_covered_variables.saturating_add(part.lp_objective_covered_variables);
    total.lp_source_rows = total.lp_source_rows.saturating_add(part.lp_source_rows);
    total.lp_nonzeros = total.lp_nonzeros.saturating_add(part.lp_nonzeros);
    total.lp_solves = total.lp_solves.saturating_add(part.lp_solves);
    total.lp_certified = total.lp_certified.saturating_add(part.lp_certified);
    total.lp_route_ng_size = total.lp_route_ng_size.max(part.lp_route_ng_size);
    total.lp_timeouts = total.lp_timeouts.saturating_add(part.lp_timeouts);
    total.lp_refactorizations = total.lp_refactorizations.saturating_add(part.lp_refactorizations);
    total.lp_micros = total.lp_micros.saturating_add(part.lp_micros);
    if total.lp_root_bound.is_none() {
        total.lp_root_bound = part.lp_root_bound;
    }
}

#[cfg(test)]
thread_local! {
    static ROUTING_RELAXATION_EDGE_EVALUATIONS: Cell<Option<u64>> = const { Cell::new(None) };
}

#[cfg(test)]
pub(crate) struct DualAuditGuard;

#[cfg(test)]
impl DualAuditGuard {
    pub(crate) fn acquire() -> Self {
        ROUTING_RELAXATION_EDGE_EVALUATIONS.with(|count| count.set(Some(0)));
        Self
    }
}

#[cfg(test)]
impl Drop for DualAuditGuard {
    fn drop(&mut self) {
        ROUTING_RELAXATION_EDGE_EVALUATIONS.with(|count| count.set(None));
    }
}

#[cfg(test)]
pub(crate) fn audit_routing_relaxation_edge_evaluations() -> u64 {
    ROUTING_RELAXATION_EDGE_EVALUATIONS.with(|count| count.get().unwrap_or(0))
}

#[cfg(test)]
pub(crate) fn audit_reset_routing_relaxation_edge_evaluations() {
    ROUTING_RELAXATION_EDGE_EVALUATIONS.with(|count| {
        if count.get().is_some() {
            count.set(Some(0));
        }
    });
}

#[cfg(test)]
fn audit_note_routing_relaxation_edge_evaluation() {
    ROUTING_RELAXATION_EDGE_EVALUATIONS.with(|count| count.set(count.get().map(|value| value + 1)));
}

fn additive_assignment_bound(model: &CollectionModel, stop: &AtomicBool) -> Option<DualBound> {
    let tier = model.objectives.first()?;
    if tier.max_terms.as_ref().is_some_and(|terms| !terms.is_empty()) || model.lists == 0 {
        return None;
    }
    let mut contributions = Vec::with_capacity(model.lists);
    for _ in 0..model.lists {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        contributions.push(vec![0i128; model.items.len()]);
    }
    let mut magnitude = 0i128;
    for reduction in &tier.terms {
        if stop.load(Ordering::Relaxed)
            || !matches!(reduction.op, ReduceOp::Sum)
            || !matches!(reduction.iterable, Iterable::Items(_) | Iterable::SetItems(_))
        {
            return None;
        }
        let list = reduction.iterable.list();
        if list >= model.lists {
            return None;
        }
        for (item_index, &item) in model.items.iter().enumerate() {
            if item_index.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                return None;
            }
            let raw = i128::from(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(item)]));
            let value = raw.checked_mul(i128::from(reduction.coeff))?;
            contributions[list][item_index] = contributions[list][item_index].checked_add(value)?;
            magnitude = magnitude.checked_add(value.abs())?;
        }
    }
    if magnitude > i128::from(i64::MAX) {
        return None;
    }
    let value = (0..model.items.len()).try_fold(0i128, |total, item| {
        let selected = if tier.minimize {
            contributions.iter().map(|list| list[item]).min()
        } else {
            contributions.iter().map(|list| list[item]).max()
        }?;
        total.checked_add(selected)
    })?;
    Some(DualBound { value: i64::try_from(value).ok()?, method: "assignment relaxation", stats: SolveStats::default() })
}

fn used_list_bound(model: &CollectionModel, stop: &AtomicBool) -> Option<DualBound> {
    let tier = model.objectives.first()?;
    if !tier.minimize || tier.max_terms.as_ref().is_some_and(|terms| !terms.is_empty()) || tier.terms.is_empty() {
        return None;
    }
    let mut coefficients = vec![0i128; model.lists];
    for reduction in &tier.terms {
        if stop.load(Ordering::Relaxed) || !matches!(reduction.op, ReduceOp::Used) {
            return None;
        }
        let list = reduction.iterable.list();
        let coefficient = coefficients.get_mut(list)?;
        *coefficient = coefficient.checked_add(i128::from(reduction.coeff))?;
    }
    if coefficients.iter().try_fold(0i128, |sum, coefficient| sum.checked_add(coefficient.abs()))? > i128::from(i64::MAX) {
        return None;
    }
    let capacity = capacity_relaxation(model, stop);
    let (min_routes, method) = if model.items.is_empty() {
        (0, "used-list relaxation")
    } else {
        let mut bound = capacity.as_ref().map_or(1, |capacity| capacity.min_routes.max(1));
        let mut method = capacity.as_ref().map_or("used-list relaxation", |capacity| capacity.method);
        let conflicts = list_conflicts(model, capacity.as_ref(), stop)?;
        if let Some(capacity) = &capacity {
            let candidate = conflict_packing_bound(&capacity.demands, capacity.capacity, &conflicts, stop);
            if candidate > bound {
                bound = candidate;
                method = "conflict-aware bin-packing relaxation";
            }
        }
        if let Some(vrptw) = VrptwRelaxation::from_model(model, capacity.as_ref(), stop) {
            if let Some(candidate) = vrptw.fleet_bound(stop) {
                if candidate.value >= bound {
                    bound = candidate.value;
                    method = candidate.method;
                }
            }
        }
        (bound, method)
    };
    let max_routes = model.items.len().min(model.lists);
    if min_routes > max_routes {
        return None;
    }
    coefficients.sort_unstable();
    let mut prefix = 0i128;
    let mut best = None;
    for (used, coefficient) in coefficients.iter().take(max_routes).enumerate() {
        prefix = prefix.checked_add(*coefficient)?;
        if used + 1 >= min_routes {
            best = Some(best.map_or(prefix, |value: i128| value.min(prefix)));
        }
    }
    if min_routes == 0 {
        best = Some(best.map_or(0, |value| value.min(0)));
    }
    Some(DualBound { value: i64::try_from(best?).ok()?, method, stats: SolveStats::default() })
}

#[derive(Clone)]
struct CapacityRelaxation {
    demands: Vec<i64>,
    capacity: i64,
    min_routes: usize,
    method: &'static str,
}

fn capacity_relaxation(model: &CollectionModel, stop: &AtomicBool) -> Option<CapacityRelaxation> {
    struct Family {
        demands: Vec<i64>,
        capacity: i64,
        lists: Vec<bool>,
    }
    let mut families: Vec<Family> = Vec::new();
    for constraint in &model.constraints {
        if !matches!(constraint.op, Op::Le)
            || !matches!(constraint.reduction.op, ReduceOp::Sum)
            || !matches!(constraint.reduction.iterable, Iterable::Items(_) | Iterable::SetItems(_))
            || constraint.rhs < 0
        {
            continue;
        }
        let list = constraint.reduction.iterable.list();
        if constraint.reduction.coeff < 0 {
            continue;
        }
        let mut demands = Vec::with_capacity(model.items.len());
        let mut raw_total = 0i128;
        let mut demand_total = 0i128;
        for (item_index, &item) in model.items.iter().enumerate() {
            if item_index.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                return None;
            }
            let raw = eval_expr(&constraint.reduction.arena.exprs, constraint.reduction.body, &[i64::from(item)]);
            if raw < 0 {
                demands.clear();
                break;
            }
            raw_total = raw_total.checked_add(i128::from(raw))?;
            let demand = raw.checked_mul(constraint.reduction.coeff)?;
            demand_total = demand_total.checked_add(i128::from(demand))?;
            demands.push(demand);
        }
        if demands.len() != model.items.len() || raw_total > i128::from(i64::MAX) || demand_total > i128::from(i64::MAX) {
            continue;
        }
        if let Some(family) = families.iter_mut().find(|family| family.capacity == constraint.rhs && family.demands == demands) {
            *family.lists.get_mut(list)? = true;
        } else {
            let mut lists = vec![false; model.lists];
            *lists.get_mut(list)? = true;
            families.push(Family { demands, capacity: constraint.rhs, lists });
        }
    }
    families
        .into_iter()
        .filter(|family| family.lists.iter().all(|&seen| seen))
        .filter_map(|family| {
            let total = family.demands.iter().try_fold(0i128, |sum, &demand| sum.checked_add(i128::from(demand)))?;
            if family.capacity == 0 {
                return (total == 0).then_some(CapacityRelaxation {
                    demands: family.demands,
                    capacity: 0,
                    min_routes: 0,
                    method: "capacity assignment relaxation",
                });
            }
            let volume = usize::try_from((total + i128::from(family.capacity) - 1) / i128::from(family.capacity)).ok()?;
            let min_routes = bin_packing_bound(&family.demands, family.capacity, stop).max(volume);
            let method = if min_routes > volume { "bin-packing relaxation" } else { "capacity assignment relaxation" };
            Some(CapacityRelaxation { demands: family.demands, capacity: family.capacity, min_routes, method })
        })
        .max_by_key(|family| family.min_routes)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PackingSearch {
    Feasible,
    Infeasible,
    Limit,
    Interrupted,
}

fn ceil_ratio(total: i128, capacity: i128) -> Option<usize> {
    if total <= 0 {
        return Some(0);
    }
    usize::try_from((total + capacity - 1) / capacity).ok()
}

fn packing_seed_bound(sizes: &[i64], capacity: i64) -> usize {
    if capacity <= 0 {
        return usize::from(sizes.iter().any(|&size| size > 0));
    }
    let positive = sizes.iter().copied().filter(|&size| size > 0).collect::<Vec<_>>();
    if positive.iter().any(|&size| size > capacity) {
        return positive.len().saturating_add(1);
    }
    let total = positive.iter().map(|&size| i128::from(size)).sum::<i128>();
    let mut bound = ceil_ratio(total, i128::from(capacity)).unwrap_or(usize::MAX);
    bound = bound.max(positive.iter().filter(|&&size| i128::from(size) * 2 > i128::from(capacity)).count());

    let mut thresholds = positive.iter().copied().filter(|&size| i128::from(size) * 2 <= i128::from(capacity)).collect::<Vec<_>>();
    thresholds.sort_unstable();
    thresholds.dedup();
    for threshold in thresholds {
        let transformed = positive
            .iter()
            .map(|&size| {
                if size > capacity - threshold {
                    capacity
                } else if size < threshold {
                    0
                } else {
                    size
                }
            })
            .map(i128::from)
            .sum::<i128>();
        bound = bound.max(ceil_ratio(transformed, i128::from(capacity)).unwrap_or(usize::MAX));
    }
    bound
}

fn bin_packing_bound(sizes: &[i64], capacity: i64, stop: &AtomicBool) -> usize {
    let mut sizes = sizes.iter().copied().filter(|&size| size > 0).collect::<Vec<_>>();
    let seed = packing_seed_bound(&sizes, capacity);
    if capacity <= 0 || sizes.is_empty() || sizes.len() > MAX_EXACT_PACKING_ITEMS || seed > sizes.len() {
        return seed;
    }
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    for bins in seed..=sizes.len() {
        let mut loads = vec![0i64; bins];
        let mut nodes = 0u64;
        match pack_plain(&sizes, capacity, 0, &mut loads, &mut nodes, stop) {
            PackingSearch::Feasible => return bins,
            PackingSearch::Infeasible => {}
            PackingSearch::Limit | PackingSearch::Interrupted => return bins,
        }
    }
    sizes.len()
}

fn pack_plain(sizes: &[i64], capacity: i64, item: usize, loads: &mut [i64], nodes: &mut u64, stop: &AtomicBool) -> PackingSearch {
    *nodes = nodes.saturating_add(1);
    if *nodes > PACKING_NODE_LIMIT {
        return PackingSearch::Limit;
    }
    if (*nodes).is_multiple_of(256) && stop.load(Ordering::Relaxed) {
        return PackingSearch::Interrupted;
    }
    if item == sizes.len() {
        return PackingSearch::Feasible;
    }
    let size = sizes[item];
    let mut tried = Vec::new();
    for bin in 0..loads.len() {
        let load = loads[bin];
        if load.checked_add(size).is_none_or(|next| next > capacity) || tried.contains(&load) {
            continue;
        }
        tried.push(load);
        loads[bin] += size;
        match pack_plain(sizes, capacity, item + 1, loads, nodes, stop) {
            PackingSearch::Feasible => return PackingSearch::Feasible,
            PackingSearch::Limit => {
                loads[bin] -= size;
                return PackingSearch::Limit;
            }
            PackingSearch::Interrupted => {
                loads[bin] -= size;
                return PackingSearch::Interrupted;
            }
            PackingSearch::Infeasible => {}
        }
        loads[bin] -= size;
        if load == 0 {
            break;
        }
    }
    PackingSearch::Infeasible
}

#[allow(clippy::needless_range_loop)]
fn list_conflicts(model: &CollectionModel, capacity: Option<&CapacityRelaxation>, stop: &AtomicBool) -> Option<Vec<Vec<bool>>> {
    let count = model.items.len();
    let mut conflicts = Vec::with_capacity(count);
    for _ in 0..count {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        conflicts.push(vec![false; count]);
    }
    if let Some(capacity) = capacity {
        for left in 0..count {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            for right in left + 1..count {
                if capacity.demands[left].checked_add(capacity.demands[right]).is_none_or(|sum| sum > capacity.capacity) {
                    conflicts[left][right] = true;
                    conflicts[right][left] = true;
                }
            }
        }
    }
    let positions = model.items.iter().enumerate().map(|(index, &item)| (item, index)).collect::<HashMap<_, _>>();
    let mut mark = |a: i32, b: i32| {
        if let (Some(&left), Some(&right)) = (positions.get(&a), positions.get(&b)) {
            conflicts[left][right] = true;
            conflicts[right][left] = true;
        }
    };
    for constraint in &model.globals {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        match constraint {
            GlobalConstraint::DifferentList { a, b } | GlobalConstraint::ListDistance { a, b, min: 1.., .. } => mark(*a, *b),
            GlobalConstraint::AllDifferentLists { items } => {
                for left in 0..items.len() {
                    if stop.load(Ordering::Relaxed) {
                        return None;
                    }
                    for right in left + 1..items.len() {
                        mark(items[left], items[right]);
                    }
                }
            }
            GlobalConstraint::ListLe { .. }
            | GlobalConstraint::SameList { .. }
            | GlobalConstraint::AllSameList { .. }
            | GlobalConstraint::ListDistance { .. } => {}
        }
    }
    Some(conflicts)
}

fn greedy_clique_bound(conflicts: &[Vec<bool>], active: &[bool], stop: &AtomicBool) -> Option<usize> {
    let mut best = usize::from(active.iter().any(|&enabled| enabled));
    for start in 0..conflicts.len() {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        if !active[start] {
            continue;
        }
        let mut candidates = (0..conflicts.len()).filter(|&node| active[node] && conflicts[start][node]).collect::<Vec<_>>();
        let mut size = 1usize;
        while !candidates.is_empty() {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let mut selected = None;
            for (position, &node) in candidates.iter().enumerate() {
                let degree = candidates.iter().filter(|&&other| conflicts[node][other]).count();
                if selected.is_none_or(|(_, best_degree)| degree > best_degree) {
                    selected = Some((position, degree));
                }
            }
            let (position, _) = selected.expect("non-empty candidate set");
            let picked = candidates[position];
            let _ = candidates.swap_remove(position);
            size += 1;
            candidates.retain(|&node| conflicts[picked][node]);
        }
        best = best.max(size);
    }
    Some(best)
}

fn conflict_packing_bound(sizes: &[i64], capacity: i64, conflicts: &[Vec<bool>], stop: &AtomicBool) -> usize {
    let active = sizes.iter().map(|&size| size > 0).collect::<Vec<_>>();
    let seed = packing_seed_bound(sizes, capacity);
    let bound = seed.max(greedy_clique_bound(conflicts, &active, stop).unwrap_or(seed));
    let mut order = (0..sizes.len()).filter(|&item| active[item]).collect::<Vec<_>>();
    if capacity <= 0 || order.is_empty() || order.len() > MAX_EXACT_PACKING_ITEMS || bound > order.len() {
        return bound;
    }
    order.sort_unstable_by(|&left, &right| sizes[right].cmp(&sizes[left]).then_with(|| right.cmp(&left)));
    for bins in bound..=order.len() {
        let mut loads = vec![0i64; bins];
        let mut members = vec![Vec::<usize>::new(); bins];
        let mut nodes = 0u64;
        match pack_conflicts(&order, sizes, capacity, conflicts, 0, &mut loads, &mut members, &mut nodes, stop) {
            PackingSearch::Feasible => return bins,
            PackingSearch::Infeasible => {}
            PackingSearch::Limit | PackingSearch::Interrupted => return bins,
        }
    }
    order.len()
}

#[allow(clippy::too_many_arguments)]
fn pack_conflicts(
    order: &[usize],
    sizes: &[i64],
    capacity: i64,
    conflicts: &[Vec<bool>],
    position: usize,
    loads: &mut [i64],
    members: &mut [Vec<usize>],
    nodes: &mut u64,
    stop: &AtomicBool,
) -> PackingSearch {
    *nodes = nodes.saturating_add(1);
    if *nodes > PACKING_NODE_LIMIT {
        return PackingSearch::Limit;
    }
    if (*nodes).is_multiple_of(256) && stop.load(Ordering::Relaxed) {
        return PackingSearch::Interrupted;
    }
    if position == order.len() {
        return PackingSearch::Feasible;
    }
    let item = order[position];
    for bin in 0..loads.len() {
        let load = loads[bin];
        if load.checked_add(sizes[item]).is_none_or(|next| next > capacity) || members[bin].iter().any(|&other| conflicts[item][other]) {
            continue;
        }
        loads[bin] += sizes[item];
        members[bin].push(item);
        match pack_conflicts(order, sizes, capacity, conflicts, position + 1, loads, members, nodes, stop) {
            PackingSearch::Feasible => return PackingSearch::Feasible,
            PackingSearch::Limit => {
                let _ = members[bin].pop();
                loads[bin] -= sizes[item];
                return PackingSearch::Limit;
            }
            PackingSearch::Interrupted => {
                let _ = members[bin].pop();
                loads[bin] -= sizes[item];
                return PackingSearch::Interrupted;
            }
            PackingSearch::Infeasible => {}
        }
        let _ = members[bin].pop();
        loads[bin] -= sizes[item];
        if load == 0 {
            break;
        }
    }
    PackingSearch::Infeasible
}

#[derive(Clone, PartialEq, Eq)]
struct VrptwData {
    depot: i32,
    initial_time: i64,
    travel: Vec<Vec<i64>>,
    earliest: Vec<i64>,
    latest: Vec<i64>,
    service: Vec<i64>,
}

struct VrptwRelaxation {
    items: Vec<i32>,
    data: VrptwData,
    capacity: CapacityRelaxation,
}

#[derive(Clone, Copy)]
struct FleetBound {
    value: usize,
    method: &'static str,
}

fn arena_expr(arena: &ExprArena, id: ExprId) -> Option<&Expr> {
    arena.exprs.get(id.0 as usize)
}

fn is_arg(arena: &ExprArena, id: ExprId, argument: u8) -> bool {
    matches!(arena_expr(arena, id), Some(Expr::Arg(found)) if *found == argument)
}

fn is_zero(arena: &ExprArena, id: ExprId) -> bool {
    matches!(arena_expr(arena, id), Some(Expr::Const(0)))
}

fn array_on_arg(arena: &ExprArena, id: ExprId, argument: u8) -> Option<Vec<i64>> {
    let Expr::Array(values, index) = arena_expr(arena, id)? else { return None };
    is_arg(arena, *index, argument).then(|| values.as_ref().clone())
}

fn matrix_on_args(arena: &ExprArena, id: ExprId, row_argument: u8, column_argument: u8) -> Option<Vec<Vec<i64>>> {
    let Expr::Matrix(values, row, column) = arena_expr(arena, id)? else { return None };
    (is_arg(arena, *row, row_argument) && is_arg(arena, *column, column_argument)).then(|| values.as_ref().clone())
}

fn expr_equivalent(left_arena: &ExprArena, left: ExprId, right_arena: &ExprArena, right: ExprId) -> bool {
    match (arena_expr(left_arena, left), arena_expr(right_arena, right)) {
        (Some(Expr::Const(a)), Some(Expr::Const(b))) => a == b,
        (Some(Expr::Arg(a)), Some(Expr::Arg(b))) => a == b,
        (Some(Expr::Array(left_values, left_index)), Some(Expr::Array(right_values, right_index))) => {
            (Arc::ptr_eq(left_values, right_values) || left_values.as_ref() == right_values.as_ref())
                && expr_equivalent(left_arena, *left_index, right_arena, *right_index)
        }
        (Some(Expr::Matrix(left_values, left_row, left_col)), Some(Expr::Matrix(right_values, right_row, right_col))) => {
            (Arc::ptr_eq(left_values, right_values) || left_values.as_ref() == right_values.as_ref())
                && expr_equivalent(left_arena, *left_row, right_arena, *right_row)
                && expr_equivalent(left_arena, *left_col, right_arena, *right_col)
        }
        (Some(Expr::Add(la, lb)), Some(Expr::Add(ra, rb)))
        | (Some(Expr::Sub(la, lb)), Some(Expr::Sub(ra, rb)))
        | (Some(Expr::Mul(la, lb)), Some(Expr::Mul(ra, rb)))
        | (Some(Expr::Mod(la, lb)), Some(Expr::Mod(ra, rb)))
        | (Some(Expr::Min(la, lb)), Some(Expr::Min(ra, rb)))
        | (Some(Expr::Max(la, lb)), Some(Expr::Max(ra, rb)))
        | (Some(Expr::Div(la, lb)), Some(Expr::Div(ra, rb)))
        | (Some(Expr::Lt(la, lb)), Some(Expr::Lt(ra, rb)))
        | (Some(Expr::Le(la, lb)), Some(Expr::Le(ra, rb)))
        | (Some(Expr::Eq(la, lb)), Some(Expr::Eq(ra, rb)))
        | (Some(Expr::Ne(la, lb)), Some(Expr::Ne(ra, rb))) => {
            expr_equivalent(left_arena, *la, right_arena, *ra) && expr_equivalent(left_arena, *lb, right_arena, *rb)
        }
        (Some(Expr::Pow(left_base, left_exp)), Some(Expr::Pow(right_base, right_exp))) => {
            left_exp == right_exp && expr_equivalent(left_arena, *left_base, right_arena, *right_base)
        }
        (Some(Expr::MulScaled(la, lb, ls)), Some(Expr::MulScaled(ra, rb, rs)))
        | (Some(Expr::DivScaled(la, lb, ls)), Some(Expr::DivScaled(ra, rb, rs))) => {
            ls == rs && expr_equivalent(left_arena, *la, right_arena, *ra) && expr_equivalent(left_arena, *lb, right_arena, *rb)
        }
        (Some(Expr::Abs(left_inner)), Some(Expr::Abs(right_inner))) => expr_equivalent(left_arena, *left_inner, right_arena, *right_inner),
        (Some(Expr::IfThenElse(lc, lt, lo)), Some(Expr::IfThenElse(rc, rt, ro))) => {
            expr_equivalent(left_arena, *lc, right_arena, *rc)
                && expr_equivalent(left_arena, *lt, right_arena, *rt)
                && expr_equivalent(left_arena, *lo, right_arena, *ro)
        }
        (
            Some(Expr::PiecewiseLinear { input: left_input, points: left_points }),
            Some(Expr::PiecewiseLinear { input: right_input, points: right_points }),
        ) => {
            (Arc::ptr_eq(left_points, right_points) || left_points.as_ref() == right_points.as_ref())
                && expr_equivalent(left_arena, *left_input, right_arena, *right_input)
        }
        (Some(Expr::External { name: left_name, args: left_args }), Some(Expr::External { name: right_name, args: right_args })) => {
            left_name == right_name
                && left_args.len() == right_args.len()
                && left_args
                    .iter()
                    .zip(right_args.iter())
                    .all(|(&left_arg, &right_arg)| expr_equivalent(left_arena, left_arg, right_arena, right_arg))
        }
        _ => false,
    }
}

fn same_routing_reduction_set(left: &[&Reduction], right: &[&Reduction]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right.iter()).all(|(left, right)| {
            left.coeff == right.coeff
                && matches!(left.op, ReduceOp::Sum)
                && matches!(right.op, ReduceOp::Sum)
                && expr_equivalent(&left.arena, left.body, &right.arena, right.body)
        })
}

fn build_routing_cost_matrix(nodes: &[i32], reductions: &[&Reduction], customers: usize, stop: &AtomicBool) -> Option<Vec<Vec<i64>>> {
    let mut max_raw = vec![0i128; reductions.len()];
    let mut matrix = Vec::with_capacity(nodes.len());
    for _ in 0..nodes.len() {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        matrix.push(vec![0i64; nodes.len()]);
    }
    for (from_index, &from) in nodes.iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        for (to_index, &to) in nodes.iter().enumerate() {
            if to_index.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                return None;
            }
            let mut cost = 0i128;
            for (reduction_index, reduction) in reductions.iter().enumerate() {
                #[cfg(test)]
                audit_note_routing_relaxation_edge_evaluation();
                let raw = eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(from), i64::from(to)]);
                if raw < 0 {
                    return None;
                }
                max_raw[reduction_index] = max_raw[reduction_index].max(i128::from(raw));
                let contribution = i128::from(raw).checked_mul(i128::from(reduction.coeff))?;
                i64::try_from(contribution).ok()?;
                cost = cost.checked_add(contribution)?;
            }
            matrix[from_index][to_index] = i64::try_from(cost).ok()?;
        }
    }
    let max_route_edges = i128::try_from(customers.saturating_add(1)).ok()?;
    for (&raw, reduction) in max_raw.iter().zip(reductions) {
        let worst = raw.checked_mul(max_route_edges)?.checked_mul(i128::from(reduction.coeff))?;
        i64::try_from(worst).ok()?;
    }
    (matrix[0][0] == 0 && matrix.iter().flatten().all(|&cost| cost >= 0)).then_some(matrix)
}

type ParsedVrptwStep = (Vec<Vec<i64>>, Vec<i64>, Vec<i64>);

fn parse_vrptw_step(arena: &ExprArena, root: ExprId) -> Option<ParsedVrptwStep> {
    let Expr::Add(left, right) = arena_expr(arena, root)? else { return None };
    for (clock, service_expr) in [(*left, *right), (*right, *left)] {
        let Some(service) = array_on_arg(arena, service_expr, 0) else { continue };
        let Some(Expr::Max(first, second)) = arena_expr(arena, clock) else { continue };
        for (release_expr, advance) in [(*first, *second), (*second, *first)] {
            let Some(earliest) = array_on_arg(arena, release_expr, 0) else { continue };
            let Some(Expr::Add(a, b)) = arena_expr(arena, advance) else { continue };
            for (accumulator, travel_expr) in [(*a, *b), (*b, *a)] {
                if !is_arg(arena, accumulator, 1) {
                    continue;
                }
                if let Some(travel) = matrix_on_args(arena, travel_expr, 2, 0) {
                    return Some((travel, earliest, service));
                }
            }
        }
    }
    None
}

fn parse_vrptw_emit(arena: &ExprArena, root: ExprId) -> Option<(Vec<i64>, Vec<i64>)> {
    let Expr::Max(left, right) = arena_expr(arena, root)? else { return None };
    let lateness = if is_zero(arena, *left) {
        *right
    } else if is_zero(arena, *right) {
        *left
    } else {
        return None;
    };
    let Expr::Sub(without_service, latest_expr) = arena_expr(arena, lateness)? else { return None };
    let latest = array_on_arg(arena, *latest_expr, 0)?;
    let Expr::Sub(accumulator, service_expr) = arena_expr(arena, *without_service)? else { return None };
    if !is_arg(arena, *accumulator, 1) {
        return None;
    }
    Some((array_on_arg(arena, *service_expr, 0)?, latest))
}

fn parse_vrptw_constraint(constraint: &crate::model::list::Constraint) -> Option<(usize, VrptwData)> {
    if !matches!(constraint.op, Op::Le)
        || constraint.rhs != 0
        || !matches!(constraint.reduction.op, ReduceOp::Sum)
        || constraint.reduction.coeff != 1
    {
        return None;
    }
    let Iterable::Scan { list, init, boundary, step, end: Some(end) } = constraint.reduction.iterable else { return None };
    if boundary != end {
        return None;
    }
    let arena = &constraint.reduction.arena;
    let (travel, earliest, service) = parse_vrptw_step(arena, step)?;
    let (emit_service, latest) = parse_vrptw_emit(arena, constraint.reduction.body)?;
    if service != emit_service {
        return None;
    }
    Some((list, VrptwData { depot: boundary, initial_time: init, travel, earliest, latest, service }))
}

impl VrptwRelaxation {
    fn from_model(model: &CollectionModel, capacity: Option<&CapacityRelaxation>, stop: &AtomicBool) -> Option<Self> {
        let capacity = capacity?.clone();
        if stop.load(Ordering::Relaxed) || capacity.capacity <= 0 || model.lists == 0 || model.items.is_empty() {
            return None;
        }
        let mut by_list = vec![None; model.lists];
        for constraint in &model.constraints {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let Some((list, data)) = parse_vrptw_constraint(constraint) else { continue };
            let slot = by_list.get_mut(list)?;
            if slot.as_ref().is_some_and(|known| known != &data) {
                return None;
            }
            *slot = Some(data);
        }
        let data = by_list.first()?.clone()?;
        if by_list.iter().any(|entry| entry.as_ref() != Some(&data)) {
            return None;
        }
        let depot = usize::try_from(data.depot).ok()?;
        let mut node_indices = Vec::with_capacity(model.items.len());
        for &item in &model.items {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            node_indices.push(usize::try_from(item).ok()?);
        }
        let max_node = node_indices.iter().copied().chain(std::iter::once(depot)).max()?;
        if data.earliest.len() <= max_node
            || data.latest.len() <= max_node
            || data.service.len() <= max_node
            || data.travel.len() <= max_node
            || data.travel.iter().any(|row| row.len() <= max_node)
            || data.initial_time != data.earliest[depot]
        {
            return None;
        }
        let nodes = std::iter::once(depot).chain(node_indices.iter().copied()).collect::<Vec<_>>();
        for &node in &nodes {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            if data.earliest[node] > data.latest[node] || data.service[node] < 0 || data.travel[node][node] != 0 {
                return None;
            }
            for &other in &nodes {
                if data.travel[node][other] < 0 {
                    return None;
                }
            }
        }
        if nodes.len() > MAX_VRPTW_METRIC_NODES {
            return None;
        }
        for &from in &nodes {
            for &via in &nodes {
                if stop.load(Ordering::Relaxed) {
                    return None;
                }
                for &to in &nodes {
                    if i128::from(data.travel[from][to]) > i128::from(data.travel[from][via]) + i128::from(data.travel[via][to]) {
                        return None;
                    }
                }
            }
        }
        Some(Self { items: model.items.clone(), data, capacity })
    }

    #[allow(clippy::needless_range_loop)]
    fn fleet_bound(&self, stop: &AtomicBool) -> Option<FleetBound> {
        let mut best = FleetBound { value: self.capacity.min_routes.max(1), method: self.capacity.method };
        let mut conflicts = Vec::with_capacity(self.items.len());
        for _ in 0..self.items.len() {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            conflicts.push(vec![false; self.items.len()]);
        }
        for left in 0..self.items.len() {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            for right in left + 1..self.items.len() {
                let incompatible =
                    self.capacity.demands[left].checked_add(self.capacity.demands[right]).is_none_or(|sum| sum > self.capacity.capacity)
                        || (!self.route_feasible(&[left, right]) && !self.route_feasible(&[right, left]));
                conflicts[left][right] = incompatible;
                conflicts[right][left] = incompatible;
            }
        }
        let packing = conflict_packing_bound(&self.capacity.demands, self.capacity.capacity, &conflicts, stop);
        if packing >= best.value {
            best = FleetBound { value: packing, method: "VRPTW conflict-aware bin-packing relaxation" };
        }
        if let Some(energy) = self.energy_bound(&conflicts, stop) {
            if energy.value >= best.value {
                best = energy;
            }
        }
        if let Some(route_cover) = self.route_cover_bound(stop) {
            if route_cover >= best.value {
                best = FleetBound { value: route_cover, method: "exact VRPTW route-cover dual" };
            }
        }
        Some(best)
    }

    fn node(&self, customer: usize) -> Option<usize> {
        usize::try_from(*self.items.get(customer)?).ok()
    }

    fn route_feasible(&self, customers: &[usize]) -> bool {
        let Ok(depot) = usize::try_from(self.data.depot) else { return false };
        let mut departure = self.data.initial_time;
        let mut previous = depot;
        for &customer in customers {
            let Some(node) = self.node(customer) else { return false };
            let Some(arrival) = departure.checked_add(self.data.travel[previous][node]) else { return false };
            let start = self.data.earliest[node].max(arrival);
            if start > self.data.latest[node] {
                return false;
            }
            let Some(next) = start.checked_add(self.data.service[node]) else { return false };
            departure = next;
            previous = node;
        }
        departure.checked_add(self.data.travel[previous][depot]).is_some_and(|arrival| arrival <= self.data.latest[depot])
    }

    fn route_cover_bound(&self, stop: &AtomicBool) -> Option<usize> {
        let customers = self.items.len();
        if customers == 0 || customers > MAX_COLUMN_GENERATION_CUSTOMERS {
            return None;
        }
        let depot = usize::try_from(self.data.depot).ok()?;
        let states = 1usize.checked_shl(u32::try_from(customers).ok()?)?;
        let mut loads = vec![0i64; states];
        for mask in 1..states {
            if mask.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                return None;
            }
            let bit = mask.trailing_zeros() as usize;
            loads[mask] = loads[mask & (mask - 1)].checked_add(self.capacity.demands[bit])?;
        }
        let mut completion = vec![INF; states.checked_mul(customers)?];
        for customer in 0..customers {
            let node = self.node(customer)?;
            let arrival = self.data.initial_time.checked_add(self.data.travel[depot][node])?;
            let start = self.data.earliest[node].max(arrival);
            if self.capacity.demands[customer] <= self.capacity.capacity && start <= self.data.latest[node] {
                completion[(1usize << customer) * customers + customer] = start.checked_add(self.data.service[node])?;
            }
        }
        for mask in 1..states {
            if mask.is_multiple_of(256) && stop.load(Ordering::Relaxed) {
                return None;
            }
            if loads[mask] > self.capacity.capacity {
                continue;
            }
            for last in 0..customers {
                if mask & (1usize << last) == 0 {
                    continue;
                }
                let previous = mask ^ (1usize << last);
                if previous == 0 {
                    continue;
                }
                let last_node = self.node(last)?;
                let mut best = INF;
                for before in 0..customers {
                    if previous & (1usize << before) == 0 {
                        continue;
                    }
                    let prefix = completion[previous * customers + before];
                    if prefix == INF {
                        continue;
                    }
                    let before_node = self.node(before)?;
                    let Some(arrival) = prefix.checked_add(self.data.travel[before_node][last_node]) else { continue };
                    let start = self.data.earliest[last_node].max(arrival);
                    if start <= self.data.latest[last_node] {
                        if let Some(value) = start.checked_add(self.data.service[last_node]) {
                            best = best.min(value);
                        }
                    }
                }
                completion[mask * customers + last] = best;
            }
        }
        let mut feasible = vec![false; states];
        for mask in 1..states {
            if mask.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                return None;
            }
            if loads[mask] > self.capacity.capacity {
                continue;
            }
            feasible[mask] = (0..customers).any(|last| {
                if mask & (1usize << last) == 0 {
                    return false;
                }
                let prefix = completion[mask * customers + last];
                let Some(node) = self.node(last) else { return false };
                prefix != INF && prefix.checked_add(self.data.travel[node][depot]).is_some_and(|arrival| arrival <= self.data.latest[depot])
            });
        }
        if (0..customers).any(|customer| !feasible[1usize << customer]) {
            return None;
        }
        covering_dual_bound(&feasible, customers, stop)
    }

    fn energy_bound(&self, conflicts: &[Vec<bool>], stop: &AtomicBool) -> Option<FleetBound> {
        let depot = usize::try_from(self.data.depot).ok()?;
        let count = self.items.len();
        let nodes = (0..count).map(|customer| self.node(customer)).collect::<Option<Vec<_>>>()?;
        let mut dprime = Vec::with_capacity(count);
        for _ in 0..count {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            dprime.push(vec![None; count]);
        }
        let mut outgoing = vec![0i128; count];
        for left in 0..count {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let from = nodes[left];
            let mut best = i128::from(self.data.travel[from][depot]);
            for right in 0..count {
                if left == right {
                    continue;
                }
                let to = nodes[right];
                let earliest_departure = i128::from(self.data.earliest[from]) + i128::from(self.data.service[from]);
                if earliest_departure + i128::from(self.data.travel[from][to]) > i128::from(self.data.latest[to]) {
                    continue;
                }
                let forced_wait =
                    i128::from(self.data.earliest[to]) - (i128::from(self.data.latest[from]) + i128::from(self.data.service[from]));
                let weight = i128::from(self.data.travel[from][to]).max(forced_wait);
                dprime[left][right] = Some(weight);
                best = best.min(weight);
            }
            outgoing[left] = best;
        }
        let mut est = vec![0i128; count];
        let mut lst = vec![0i128; count];
        let mut duration = vec![0i128; count];
        for right in 0..count {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let node = nodes[right];
            let mut incoming = i128::from(self.data.travel[depot][node]);
            for left in 0..count {
                if let Some(weight) = dprime[left][right] {
                    incoming = incoming.min(weight - outgoing[left]);
                }
            }
            incoming = incoming.max(0);
            est[right] = i128::from(self.data.earliest[node])
                .max(i128::from(self.data.initial_time) + i128::from(self.data.travel[depot][node]))
                - incoming;
            lst[right] = i128::from(self.data.latest[node])
                .min(i128::from(self.data.latest[depot]) - i128::from(self.data.service[node]) - i128::from(self.data.travel[node][depot]))
                - incoming;
            duration[right] = i128::from(self.data.service[node]) + outgoing[right] + incoming;
            if duration[right] < 0 || est[right] > lst[right] {
                return None;
            }
        }
        let mut starts = vec![i128::from(self.data.initial_time)];
        let mut ends = vec![i128::from(self.data.latest[depot])];
        for customer in 0..count {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            starts.extend([est[customer], lst[customer], est[customer] + duration[customer]]);
            ends.extend([lst[customer], est[customer] + duration[customer], lst[customer] + duration[customer]]);
        }
        starts.sort_unstable();
        starts.dedup();
        ends.sort_unstable();
        ends.dedup();
        let total_pairs = starts.len().saturating_mul(ends.len());
        let mut stride = 1usize;
        while total_pairs > MAX_ENERGY_INTERVAL_PAIRS.saturating_mul(stride).saturating_mul(stride) {
            stride += 1;
        }
        let mut best = FleetBound { value: 1, method: "VRPTW interval-energy relaxation" };
        let mut examined = 0usize;
        for &left in starts.iter().step_by(stride) {
            for &right in ends.iter().step_by(stride) {
                if left >= right {
                    continue;
                }
                examined += 1;
                if examined.is_multiple_of(256) && stop.load(Ordering::Relaxed) {
                    return Some(best);
                }
                let width = right - left;
                let work = (0..count)
                    .map(|customer| {
                        interval_overlap_i128(est[customer], duration[customer], left, right).min(interval_overlap_i128(
                            lst[customer],
                            duration[customer],
                            left,
                            right,
                        ))
                    })
                    .collect::<Vec<_>>();
                let total = work.iter().sum::<i128>();
                let energy = ceil_ratio(total, width)?;
                if energy > best.value {
                    best.value = energy;
                    best.method = "VRPTW interval-energy relaxation";
                }
                if let (Ok(width), Some(work)) =
                    (i64::try_from(width), work.iter().map(|&value| i64::try_from(value).ok()).collect::<Option<Vec<_>>>())
                {
                    let active = work.iter().map(|&value| value > 0).collect::<Vec<_>>();
                    let seed = packing_seed_bound(&work, width);
                    let packing = seed.max(greedy_clique_bound(conflicts, &active, stop).unwrap_or(seed));
                    if packing > best.value {
                        best.value = packing;
                        best.method = "VRPTW interval-energy/BPPC relaxation";
                    }
                }
            }
        }
        Some(best)
    }
}

fn interval_overlap_i128(start: i128, duration: i128, left: i128, right: i128) -> i128 {
    0.max((start + duration).min(right) - start.max(left))
}

fn covering_dual_bound(feasible: &[bool], customers: usize, stop: &AtomicBool) -> Option<usize> {
    let states = feasible.len();
    let mut active = vec![false; states];
    for customer in 0..customers {
        active[1usize << customer] = true;
    }
    let mut dual = vec![0i128; customers];
    let mut center = dual.clone();
    let mut best_value = 0i128;
    for iteration in 0..48usize {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let mut proposal = dual.clone();
        for offset in 0..customers {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let customer = (offset + iteration) % customers;
            let mut slack = None;
            for (mask, &is_active) in active.iter().enumerate().take(states).skip(1) {
                if mask.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                    return None;
                }
                if is_active && mask & (1usize << customer) != 0 {
                    let value = COLUMN_DUAL_SCALE - subset_sum(mask, &proposal);
                    slack = Some(slack.map_or(value, |known: i128| known.min(value)));
                }
            }
            let slack = slack?;
            if slack > 0 {
                proposal[customer] += if iteration % 4 == 3 { slack } else { 3 * slack / 4 };
            }
        }
        let mut priced = None;
        for (mask, &is_feasible) in feasible.iter().enumerate().take(states).skip(1) {
            if mask.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                return None;
            }
            if is_feasible {
                let reduced_cost = COLUMN_DUAL_SCALE - subset_sum(mask, &proposal);
                if priced.is_none_or(|(_, known)| reduced_cost < known) {
                    priced = Some((mask, reduced_cost));
                }
            }
        }
        let (priced_mask, reduced_cost) = priced?;
        let shift = reduced_cost.min(0);
        let corrected = proposal.iter().map(|&value| value + shift).collect::<Vec<_>>();
        let corrected_value = corrected.iter().sum::<i128>();
        if corrected_value > best_value {
            best_value = corrected_value;
            center = corrected;
        }
        if reduced_cost < 0 {
            active[priced_mask] = true;
        }
        dual = proposal.iter().zip(&center).map(|(&value, &anchor)| (3 * value + anchor).div_euclid(4)).collect();
    }
    ceil_ratio(best_value, COLUMN_DUAL_SCALE)
}

struct RoutingRelaxation {
    costs: Vec<Vec<i64>>,
    routes: usize,
    symmetric: bool,
    capacity: Option<CapacityRelaxation>,
}

impl RoutingRelaxation {
    fn from_model(model: &CollectionModel, stop: &AtomicBool) -> Option<Self> {
        let tier = model.objectives.first()?;
        if !tier.minimize || tier.max_terms.as_ref().is_some_and(|terms| !terms.is_empty()) || model.lists == 0 || model.items.is_empty() {
            return None;
        }
        let mut by_list = vec![Vec::new(); model.lists];
        let mut depot = None;
        for reduction in &tier.terms {
            if !matches!(reduction.op, ReduceOp::Sum) {
                return None;
            }
            let Iterable::Edges { list, start, end } = reduction.iterable else { return None };
            if start != end || depot.is_some_and(|known| known != start) {
                return None;
            }
            depot = Some(start);
            by_list.get_mut(list)?.push(reduction);
        }
        if by_list.iter().any(Vec::is_empty) {
            return None;
        }
        let depot = depot?;
        if model.items.contains(&depot) {
            return None;
        }
        let mut unique = model.items.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != model.items.len() {
            return None;
        }
        let mut nodes = Vec::with_capacity(model.items.len() + 1);
        nodes.push(depot);
        nodes.extend_from_slice(&model.items);
        if nodes.len() > MAX_ROUTING_RELAXATION_NODES {
            return None;
        }
        let mut reference: Option<Vec<Vec<i64>>> = None;
        let mut seen_reduction_sets: Vec<&[&Reduction]> = Vec::new();
        for reductions in &by_list {
            if reductions.iter().any(|reduction| reduction.coeff < 0) {
                return None;
            }
            if seen_reduction_sets.iter().any(|known| same_routing_reduction_set(known, reductions)) {
                continue;
            }
            let matrix = build_routing_cost_matrix(&nodes, reductions, model.items.len(), stop)?;
            if let Some(reference) = &reference {
                for (row_index, (left, right)) in reference.iter().zip(&matrix).enumerate() {
                    if row_index.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                        return None;
                    }
                    if left != right {
                        return None;
                    }
                }
            } else {
                reference = Some(matrix.clone());
            }
            seen_reduction_sets.push(reductions);
        }
        let costs = reference?;
        let max_edges = i128::try_from(model.items.len().saturating_add(model.lists)).ok()?;
        let max_cost = i128::from(costs.iter().flatten().copied().max().unwrap_or(0));
        i64::try_from(max_cost.checked_mul(max_edges)?).ok()?;
        let mut symmetric = true;
        'symmetry: for (i, row) in costs.iter().enumerate() {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            for (&forward, reverse_row) in row.iter().skip(i + 1).zip(costs.iter().skip(i + 1)) {
                if forward != reverse_row[i] {
                    symmetric = false;
                    break 'symmetry;
                }
            }
        }
        Some(Self { costs, routes: model.lists, symmetric, capacity: capacity_relaxation(model, stop) })
    }

    fn min_routes(&self) -> usize {
        self.capacity.as_ref().map_or(1, |capacity| capacity.min_routes.max(1))
    }

    fn assignment_bound(&self, stop: &AtomicBool) -> Option<DualBound> {
        let customers = self.costs.len().checked_sub(1)?;
        let routes = self.min_routes();
        if routes > customers || routes > self.routes {
            return None;
        }
        let mut outgoing = 0i128;
        let mut incoming = 0i128;
        for customer in 1..=customers {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            outgoing =
                outgoing.checked_add(i128::from((0..=customers).filter(|&to| to != customer).map(|to| self.costs[customer][to]).min()?))?;
            incoming = incoming
                .checked_add(i128::from((0..=customers).filter(|&from| from != customer).map(|from| self.costs[from][customer]).min()?))?;
        }
        let mut starts = (1..=customers).map(|customer| self.costs[0][customer]).collect::<Vec<_>>();
        let mut ends = (1..=customers).map(|customer| self.costs[customer][0]).collect::<Vec<_>>();
        starts.sort_unstable();
        ends.sort_unstable();
        outgoing = starts.iter().take(routes).try_fold(outgoing, |sum, &cost| sum.checked_add(i128::from(cost)))?;
        incoming = ends.iter().take(routes).try_fold(incoming, |sum, &cost| sum.checked_add(i128::from(cost)))?;
        Some(DualBound {
            value: i64::try_from(outgoing.max(incoming)).ok()?,
            method: "directed assignment relaxation",
            stats: SolveStats::default(),
        })
    }

    fn held_karp_bound(&self, stop: &AtomicBool) -> Option<DualBound> {
        let mut penalties = vec![0i128; self.costs.len()];
        let mut center = penalties.clone();
        let max_cost = self.costs.iter().flatten().copied().max().unwrap_or(1).max(1);
        let mut step = ((i128::from(max_cost) + 3) / 4).max(1);
        let mut best = i128::MIN;
        for iteration in 0..64usize {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let (modified, degrees) = minimum_one_tree(&self.costs, &penalties, stop)?;
            let bound = modified.checked_sub(2 * penalties.iter().sum::<i128>())?;
            best = best.max(bound);
            if degrees.iter().all(|&degree| degree == 2) {
                break;
            }
            let proposal =
                penalties.iter().zip(&degrees).map(|(&penalty, &degree)| penalty + step * i128::from(degree - 2)).collect::<Vec<_>>();
            penalties = proposal.iter().zip(&center).map(|(&value, &anchor)| (3 * value + anchor).div_euclid(4)).collect();
            if iteration % 8 == 7 {
                center.clone_from(&penalties);
                step = (step * 3 / 4).max(1);
            }
        }
        Some(DualBound { value: i64::try_from(best).ok()?, method: "Held-Karp 1-tree", stats: SolveStats::default() })
    }

    fn column_generation_bound(&self, stop: &AtomicBool) -> Option<DualBound> {
        let customers = self.costs.len().checked_sub(1)?;
        if customers == 0 || customers > MAX_COLUMN_GENERATION_CUSTOMERS {
            return None;
        }
        let (demands, capacity) = match &self.capacity {
            Some(capacity) if capacity.capacity > 0 => (capacity.demands.clone(), capacity.capacity),
            Some(capacity) if capacity.demands.iter().any(|&demand| demand > 0) => return None,
            _ => (vec![1; customers], i64::try_from(customers).ok()?),
        };
        let route_cost = route_columns(&self.costs, &demands, capacity, stop)?;
        let states = route_cost.len();
        let mut active = vec![false; states];
        for customer in 0..customers {
            let mask = 1usize << customer;
            if route_cost[mask] == INF {
                return None;
            }
            active[mask] = true;
        }
        let mut dual = vec![0i128; customers];
        let mut center = dual.clone();
        let mut best_value = 0i128;
        for iteration in 0..32usize {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let mut proposal = dual.clone();
            for offset in 0..customers {
                if stop.load(Ordering::Relaxed) {
                    return None;
                }
                let customer = (offset + iteration) % customers;
                let mut slack = None;
                for mask in 1..states {
                    if mask.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                        return None;
                    }
                    if active[mask] && mask & (1usize << customer) != 0 {
                        let value = i128::from(route_cost[mask]) * COLUMN_DUAL_SCALE - subset_sum(mask, &proposal);
                        slack = Some(slack.map_or(value, |known: i128| known.min(value)));
                    }
                }
                let slack = slack?;
                if slack > 0 {
                    proposal[customer] += if iteration % 4 == 3 { slack } else { 3 * slack / 4 };
                }
            }
            let mut priced = None;
            for (mask, &cost) in route_cost.iter().enumerate().take(states).skip(1) {
                if mask.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                    return None;
                }
                if cost != INF {
                    let reduced_cost = i128::from(cost) * COLUMN_DUAL_SCALE - subset_sum(mask, &proposal);
                    if priced.is_none_or(|(_, known)| reduced_cost < known) {
                        priced = Some((mask, reduced_cost));
                    }
                }
            }
            let (priced_mask, reduced_cost) = priced?;
            let shift = reduced_cost.min(0);
            let feasible = proposal.iter().map(|&value| value + shift).collect::<Vec<_>>();
            let feasible_value = feasible.iter().sum::<i128>();
            if feasible_value > best_value {
                best_value = feasible_value;
                center = feasible;
            }
            if reduced_cost < 0 {
                active[priced_mask] = true;
            }
            dual = proposal.iter().zip(&center).map(|(&value, &anchor)| (3 * value + anchor).div_euclid(4)).collect();
        }
        Some(DualBound {
            value: i64::try_from(best_value.div_euclid(COLUMN_DUAL_SCALE)).ok()?,
            method: "stabilized VRP column generation",
            stats: SolveStats::default(),
        })
    }

    #[cfg(not(feature = "lp-relaxation"))]
    fn route_master_lp_bound(&self, _controls: LinearControls, _stop: &AtomicBool) -> Option<DualBound> {
        None
    }

    #[cfg(feature = "lp-relaxation")]
    fn route_master_lp_bound(&self, controls: LinearControls, stop: &AtomicBool) -> Option<DualBound> {
        use crate::engines::linear::{solve_advisory, AdvisoryLinearProblem, AdvisoryLinearRow, AdvisoryLinearStatus};
        use crate::orchestrator::LinearBackendMode;

        let customers = self.costs.len().checked_sub(1)?;
        let master_variables = customers.checked_add(2)?;
        if customers <= MAX_COLUMN_GENERATION_CUSTOMERS
            || controls.backend == LinearBackendMode::Native
            || controls.root_time.is_zero()
            || master_variables > controls.max_variables
            || customers > controls.max_rows
            || self.costs.iter().flatten().any(|&cost| cost < 0)
            || !(1..=16).contains(&controls.route_ng_size)
            || controls.route_max_labels == 0
            || !(1..=100).contains(&controls.route_dual_stabilization_percent)
        {
            return None;
        }
        let min_routes = self.min_routes();
        let max_routes = self.routes.min(customers);
        if min_routes > max_routes {
            return None;
        }
        let (demands, capacity) = match &self.capacity {
            Some(capacity) if capacity.capacity > 0 => (capacity.demands.clone(), capacity.capacity),
            Some(_) => return None,
            None => (vec![1; customers], i64::try_from(customers).ok()?),
        };
        if demands.len() != customers || demands.iter().any(|&demand| demand <= 0 || demand > capacity) {
            return None;
        }
        let divisor = demands.iter().copied().fold(capacity, gcd_positive).max(1);
        let scaled_capacity = usize::try_from(capacity / divisor).ok()?;
        let scaled_demands = demands.iter().map(|&demand| usize::try_from(demand / divisor).ok()).collect::<Option<Vec<_>>>()?;
        let states = scaled_capacity.checked_add(1)?.checked_mul(customers)?;
        let transitions = scaled_capacity.checked_mul(customers)?.checked_mul(customers)?;
        if states > MAX_NG_ROUTE_BASE_STATES || transitions > MAX_NG_ROUTE_BASE_TRANSITIONS {
            return None;
        }

        let singleton_costs = (1..=customers)
            .map(|customer| i128::from(self.costs[0][customer]).checked_add(i128::from(self.costs[customer][0])))
            .collect::<Option<Vec<_>>>()?;
        if singleton_costs.iter().any(|&cost| cost < 0 || cost.unsigned_abs() > u128::from(1u64 << 53)) {
            return None;
        }
        let fleet_dual_limit = singleton_costs.iter().try_fold(0i128, |sum, &cost| sum.checked_add(cost))?;
        if fleet_dual_limit.unsigned_abs() > u128::from(1u64 << 53)
            || singleton_costs
                .iter()
                .any(|&cost| cost.checked_add(fleet_dual_limit).is_none_or(|limit| limit.unsigned_abs() > u128::from(1u64 << 53)))
        {
            return None;
        }
        let mut columns = singleton_costs
            .iter()
            .enumerate()
            .map(|(customer, &cost)| {
                let mut counts = vec![0u32; customers];
                counts[customer] = 1;
                RouteColumn { counts, cost }
            })
            .collect::<Vec<_>>();
        let generated_reserve = controls.max_rows.saturating_sub(columns.len()).min(MAX_ROUTE_MASTER_ROUNDS);
        let pair_slots = controls.max_rows.saturating_sub(columns.len()).saturating_sub(generated_reserve);
        let mut pairs = Vec::new();
        for left in 0..customers {
            for right in left + 1..customers {
                if scaled_demands[left].saturating_add(scaled_demands[right]) > scaled_capacity {
                    continue;
                }
                let forward = i128::from(self.costs[0][left + 1])
                    .checked_add(i128::from(self.costs[left + 1][right + 1]))?
                    .checked_add(i128::from(self.costs[right + 1][0]))?;
                let reverse = i128::from(self.costs[0][right + 1])
                    .checked_add(i128::from(self.costs[right + 1][left + 1]))?
                    .checked_add(i128::from(self.costs[left + 1][0]))?;
                let cost = forward.min(reverse);
                let savings = cost.checked_sub(singleton_costs[left].checked_add(singleton_costs[right])?)?;
                let mut counts = vec![0u32; customers];
                counts[left] = 1;
                counts[right] = 1;
                pairs.push((savings, RouteColumn { counts, cost }));
            }
        }
        pairs.sort_unstable_by_key(|(savings, _)| *savings);
        columns.extend(pairs.into_iter().take(pair_slots).map(|(_, column)| column));
        let mut column_index = columns.iter().enumerate().map(|(index, column)| (column.counts.clone(), index)).collect::<HashMap<_, _>>();
        let started_at = Instant::now();
        let deadline = started_at.checked_add(controls.root_time)?;
        let generation_deadline = started_at.checked_add(controls.root_time.mul_f32(0.75)).unwrap_or(deadline).min(deadline);
        let q_neighborhoods = NgNeighborhoods::build(&self.costs, 1, generation_deadline, stop)?;
        let generation_pricing = RoutePricing {
            costs: &self.costs,
            demands: &scaled_demands,
            capacity: scaled_capacity,
            max_labels: controls.route_max_labels,
            deadline: generation_deadline,
            stop,
        };
        let base_dual = RouteMasterDual { customers: route_master_base_dual(&self.costs)?, fleet_lower: 0, fleet_upper: 0 };
        let base_value = base_dual.objective(min_routes, max_routes)?;
        let mut best_scaled = base_value;
        let mut best = i64::try_from(ceil_scaled(best_scaled, COLUMN_DUAL_SCALE)?).ok()?;
        let mut center = base_dual.clone();
        let mut last_dual = None;
        let mut stabilization_percent = controls.route_dual_stabilization_percent;
        let mut stats = SolveStats {
            lp_model_status: LinearModelStatus::Ready,
            lp_variables: u64::try_from(master_variables).unwrap_or(u64::MAX),
            lp_columns: u64::try_from(master_variables).unwrap_or(u64::MAX),
            lp_covered_variables: u64::try_from(customers).unwrap_or(u64::MAX),
            lp_objective_variables: u64::try_from(customers).unwrap_or(u64::MAX),
            lp_objective_covered_variables: u64::try_from(customers).unwrap_or(u64::MAX),
            lp_certified: 1,
            lp_root_bound: Some(best),
            ..SolveStats::default()
        };

        for _ in 0..MAX_ROUTE_MASTER_ROUNDS {
            if stop.load(Ordering::Acquire) || Instant::now() >= generation_deadline || columns.len() > controls.max_rows {
                break;
            }
            let nonzeros =
                columns.iter().map(|column| column.counts.iter().filter(|&&count| count != 0).count().saturating_add(2)).sum::<usize>();
            if nonzeros > controls.max_nonzeros {
                break;
            }
            let problem = AdvisoryLinearProblem {
                objective: {
                    let mut objective = vec![-1; customers];
                    objective.push(-i128::try_from(min_routes).ok()?);
                    objective.push(i128::try_from(max_routes).ok()?);
                    objective
                },
                bounds: {
                    let mut bounds = singleton_costs
                        .iter()
                        .map(|&upper| upper.checked_add(fleet_dual_limit).map(|limit| (0, limit)))
                        .collect::<Option<Vec<_>>>()?;
                    bounds.extend([(0, fleet_dual_limit), (0, fleet_dual_limit)]);
                    bounds
                },
                rows: columns
                    .iter()
                    .map(|column| AdvisoryLinearRow {
                        terms: {
                            let mut terms = column
                                .counts
                                .iter()
                                .enumerate()
                                .filter_map(|(customer, &count)| (count != 0).then_some((customer, i128::from(count))))
                                .collect::<Vec<_>>();
                            terms.extend([(customers, 1), (customers + 1, -1)]);
                            terms
                        },
                        lower: None,
                        upper: Some(column.cost),
                    })
                    .collect(),
            };
            let remaining = generation_deadline.saturating_duration_since(Instant::now());
            let started = Instant::now();
            let Some(solution) = solve_advisory(&problem, controls.backend, remaining, stop) else {
                break;
            };
            stats.lp_solves = stats.lp_solves.saturating_add(1);
            stats.lp_micros = stats.lp_micros.saturating_add(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
            stats.lp_refactorizations = stats.lp_refactorizations.saturating_add(solution.refactorizations);
            match solution.status {
                AdvisoryLinearStatus::Optimal => {}
                AdvisoryLinearStatus::TimeLimit => {
                    stats.lp_timeouts = stats.lp_timeouts.saturating_add(1);
                    break;
                }
                AdvisoryLinearStatus::Other => break,
            }
            if solution.primal.len() != master_variables {
                break;
            }
            let Some(raw_dual) = RouteMasterDual::from_primal(&solution.primal, customers) else { break };
            last_dual = Some(raw_dual.clone());
            let Some(proposal) = center.blend(&raw_dual, stabilization_percent, 100) else { break };
            let Some(priced) = generation_pricing.price(&proposal, &q_neighborhoods) else {
                break;
            };
            stats.lp_route_ng_size = stats.lp_route_ng_size.max(1);

            // Shifting every customer multiplier by the most negative reduced
            // cost makes it feasible for every nonempty q-route. The q-route
            // family contains every elementary CVRP route, so the result also
            // certifies the semantic route master, including its fleet rows.
            let shift = priced.reduced_cost.min(0);
            let Some(certified_dual) = proposal.shift_customers(shift) else { break };
            let Some(certified_value) = certified_dual.objective(min_routes, max_routes) else { break };
            let Some(certified) = ceil_scaled(certified_value, COLUMN_DUAL_SCALE).and_then(|value| i64::try_from(value).ok()) else {
                break;
            };
            if certified_value > best_scaled {
                best_scaled = certified_value;
                best = certified;
                center = certified_dual;
                stabilization_percent = stabilization_percent.saturating_add(5).min(95);
            } else {
                stabilization_percent = stabilization_percent.saturating_sub(5).max(50);
            }
            stats.lp_certified = stats.lp_certified.saturating_add(1);
            stats.lp_root_bound = Some(best);
            if priced.reduced_cost >= 0 {
                break;
            }
            let next_nonzeros = nonzeros.saturating_add(priced.counts.iter().filter(|&&count| count != 0).count().saturating_add(2));
            if columns.len() >= controls.max_rows || next_nonzeros > controls.max_nonzeros {
                break;
            }
            if let Some(&index) = column_index.get(&priced.counts) {
                if priced.cost < columns[index].cost {
                    columns[index].cost = priced.cost;
                } else {
                    break;
                }
            } else {
                column_index.insert(priced.counts.clone(), columns.len());
                columns.push(RouteColumn { counts: priced.counts, cost: priced.cost });
            }
        }
        if let Some(candidate) = last_dual {
            let projection_pricing = RoutePricing {
                costs: &self.costs,
                demands: &scaled_demands,
                capacity: scaled_capacity,
                max_labels: controls.route_max_labels,
                deadline,
                stop,
            };
            let mut projection_base = center;
            let mut previous_ng_size = 0;
            for ng_size in [1, 2.min(controls.route_ng_size), 4.min(controls.route_ng_size), controls.route_ng_size] {
                if ng_size == previous_ng_size {
                    continue;
                }
                previous_ng_size = ng_size;
                let Some(projected) = projection_pricing.project(&projection_base, &candidate, ng_size) else {
                    if stop.load(Ordering::Acquire) || Instant::now() >= deadline {
                        stats.lp_timeouts = stats.lp_timeouts.saturating_add(1);
                    }
                    break;
                };
                let projected_value = projected.objective(min_routes, max_routes)?;
                if projected_value > best_scaled {
                    best_scaled = projected_value;
                    best = i64::try_from(ceil_scaled(best_scaled, COLUMN_DUAL_SCALE)?).ok()?;
                }
                projection_base = projected;
                stats.lp_route_ng_size = stats.lp_route_ng_size.max(u64::try_from(ng_size).unwrap_or(u64::MAX));
                stats.lp_certified = stats.lp_certified.saturating_add(1);
                stats.lp_root_bound = Some(best);
            }
        }
        stats.lp_rows = u64::try_from(columns.len()).unwrap_or(u64::MAX);
        stats.lp_source_rows = stats.lp_rows;
        stats.lp_nonzeros = u64::try_from(
            columns.iter().map(|column| column.counts.iter().filter(|&&count| count != 0).count().saturating_add(2)).sum::<usize>(),
        )
        .unwrap_or(u64::MAX);
        Some(DualBound { value: best, method: "stabilized ng-route fleet LP relaxation", stats })
    }
}

#[cfg(feature = "lp-relaxation")]
#[derive(Clone)]
struct RouteColumn {
    counts: Vec<u32>,
    cost: i128,
}

#[cfg(feature = "lp-relaxation")]
#[derive(Clone)]
struct RouteMasterDual {
    customers: Vec<i128>,
    fleet_lower: i128,
    fleet_upper: i128,
}

#[cfg(feature = "lp-relaxation")]
impl RouteMasterDual {
    fn from_primal(primal: &[f64], customers: usize) -> Option<Self> {
        if primal.len() != customers.checked_add(2)? {
            return None;
        }
        Some(Self {
            customers: primal[..customers].iter().map(|&value| scaled_floor(value, COLUMN_DUAL_SCALE)).collect::<Option<Vec<_>>>()?,
            fleet_lower: scaled_floor(primal[customers], COLUMN_DUAL_SCALE)?.max(0),
            fleet_upper: scaled_ceil(primal[customers + 1], COLUMN_DUAL_SCALE)?.max(0),
        })
    }

    fn objective(&self, min_routes: usize, max_routes: usize) -> Option<i128> {
        self.customers
            .iter()
            .try_fold(0i128, |sum, &value| sum.checked_add(value))?
            .checked_add(self.fleet_lower.checked_mul(i128::try_from(min_routes).ok()?)?)?
            .checked_sub(self.fleet_upper.checked_mul(i128::try_from(max_routes).ok()?)?)
    }

    fn blend(&self, target: &Self, numerator: usize, denominator: usize) -> Option<Self> {
        if self.customers.len() != target.customers.len() || numerator > denominator || denominator == 0 {
            return None;
        }
        let blend = |anchor: i128, value: i128| {
            value
                .checked_sub(anchor)?
                .checked_mul(i128::try_from(numerator).ok()?)?
                .div_euclid(i128::try_from(denominator).ok()?)
                .checked_add(anchor)
        };
        Some(Self {
            customers: self
                .customers
                .iter()
                .zip(&target.customers)
                .map(|(&anchor, &value)| blend(anchor, value))
                .collect::<Option<Vec<_>>>()?,
            fleet_lower: blend(self.fleet_lower, target.fleet_lower)?.max(0),
            fleet_upper: blend(self.fleet_upper, target.fleet_upper)?.max(0),
        })
    }

    fn shift_customers(&self, shift: i128) -> Option<Self> {
        Some(Self {
            customers: self.customers.iter().map(|&value| value.checked_add(shift)).collect::<Option<Vec<_>>>()?,
            fleet_lower: self.fleet_lower,
            fleet_upper: self.fleet_upper,
        })
    }
}

#[cfg(feature = "lp-relaxation")]
struct PricedNgRoute {
    reduced_cost: i128,
    counts: Vec<u32>,
    cost: i128,
}

#[cfg(feature = "lp-relaxation")]
#[derive(Clone, Copy)]
struct NgLabel {
    reduced_cost: i128,
    trace: usize,
}

#[cfg(feature = "lp-relaxation")]
#[derive(Clone, Copy)]
struct NgTrace {
    customer: usize,
    predecessor: Option<usize>,
}

#[cfg(feature = "lp-relaxation")]
struct NgNeighborhoods {
    members: Vec<Vec<usize>>,
    positions: Vec<u8>,
    customers: usize,
}

#[cfg(feature = "lp-relaxation")]
impl NgNeighborhoods {
    fn build(costs: &[Vec<i64>], requested_size: usize, deadline: Instant, stop: &AtomicBool) -> Option<Self> {
        let customers = costs.len().checked_sub(1)?;
        if customers == 0 || requested_size == 0 || requested_size > 16 || costs.iter().any(|row| row.len() != costs.len()) {
            return None;
        }
        let size = requested_size.min(customers);
        let mut members = Vec::with_capacity(customers);
        let mut positions = vec![u8::MAX; customers.checked_mul(customers)?];
        for customer in 0..customers {
            if stop.load(Ordering::Acquire) || Instant::now() >= deadline {
                return None;
            }
            let mut neighborhood = Vec::with_capacity(size);
            neighborhood.push(customer);
            if size > 1 {
                let mut nearest = (0..customers)
                    .filter(|&other| other != customer)
                    .map(|other| {
                        let distance = i128::from(costs[customer + 1][other + 1]) + i128::from(costs[other + 1][customer + 1]);
                        (distance, other)
                    })
                    .collect::<Vec<_>>();
                nearest.sort_unstable();
                neighborhood.extend(nearest.into_iter().take(size - 1).map(|(_, other)| other));
            }
            for (position, &member) in neighborhood.iter().enumerate() {
                positions[customer * customers + member] = u8::try_from(position).ok()?;
            }
            members.push(neighborhood);
        }
        Some(Self { members, positions, customers })
    }

    fn position(&self, neighborhood: usize, customer: usize) -> Option<u8> {
        let position = *self.positions.get(neighborhood.checked_mul(self.customers)?.checked_add(customer)?)?;
        (position != u8::MAX).then_some(position)
    }

    fn extend(&self, last: usize, memory: u16, next: usize) -> Option<u16> {
        if self.position(last, next).is_some_and(|position| memory & (1u16 << position) != 0) {
            return None;
        }
        let mut extended = 1u16 << self.position(next, next)?;
        for (source_position, &remembered) in self.members.get(last)?.iter().enumerate() {
            if memory & (1u16 << source_position) == 0 {
                continue;
            }
            if let Some(target_position) = self.position(next, remembered) {
                extended |= 1u16 << target_position;
            }
        }
        Some(extended)
    }
}

#[cfg(feature = "lp-relaxation")]
fn gcd_positive(mut left: i64, mut right: i64) -> i64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.abs()
}

#[cfg(feature = "lp-relaxation")]
fn scaled_floor(value: f64, scale: i128) -> Option<i128> {
    if !value.is_finite() {
        return None;
    }
    let scaled = value * scale as f64;
    if !scaled.is_finite() || scaled < i128::MIN as f64 || scaled > i128::MAX as f64 {
        return None;
    }
    Some(scaled.floor() as i128)
}

#[cfg(feature = "lp-relaxation")]
fn scaled_ceil(value: f64, scale: i128) -> Option<i128> {
    if !value.is_finite() {
        return None;
    }
    let scaled = value * scale as f64;
    if !scaled.is_finite() || scaled < i128::MIN as f64 || scaled > i128::MAX as f64 {
        return None;
    }
    Some(scaled.ceil() as i128)
}

#[cfg(feature = "lp-relaxation")]
fn ceil_scaled(value: i128, scale: i128) -> Option<i128> {
    let quotient = value.div_euclid(scale);
    let remainder = value.rem_euclid(scale);
    quotient.checked_add(i128::from(remainder != 0))
}

#[cfg(feature = "lp-relaxation")]
fn route_master_base_dual(costs: &[Vec<i64>]) -> Option<Vec<i128>> {
    let customers = costs.len().checked_sub(1)?;
    if costs.iter().any(|row| row.len() != costs.len()) {
        return None;
    }
    (1..=customers)
        .map(|customer| {
            let outgoing = (0..=customers).filter(|&next| next != customer).map(|next| costs[customer][next]).min()?;
            let incoming = (0..=customers).filter(|&previous| previous != customer).map(|previous| costs[previous][customer]).min()?;
            let sum = i128::from(outgoing).checked_add(i128::from(incoming))?.checked_mul(COLUMN_DUAL_SCALE)?;
            Some(sum.div_euclid(2))
        })
        .collect()
}

#[cfg(feature = "lp-relaxation")]
struct RoutePricing<'a> {
    costs: &'a [Vec<i64>],
    demands: &'a [usize],
    capacity: usize,
    max_labels: usize,
    deadline: Instant,
    stop: &'a AtomicBool,
}

#[cfg(feature = "lp-relaxation")]
impl RoutePricing<'_> {
    fn project(&self, base: &RouteMasterDual, candidate: &RouteMasterDual, ng_size: usize) -> Option<RouteMasterDual> {
        let neighborhoods = NgNeighborhoods::build(self.costs, ng_size, self.deadline, self.stop)?;
        let priced = self.price(candidate, &neighborhoods)?;
        if priced.reduced_cost >= 0 {
            return Some(candidate.clone());
        }
        let mut feasible_numerator = 0usize;
        let mut infeasible_numerator = ROUTE_DUAL_PROJECTION_SCALE;
        let mut feasible = base.clone();
        while feasible_numerator + 1 < infeasible_numerator {
            if self.stop.load(Ordering::Acquire) || Instant::now() >= self.deadline {
                return None;
            }
            let middle = (feasible_numerator + infeasible_numerator) / 2;
            let trial = base.blend(candidate, middle, ROUTE_DUAL_PROJECTION_SCALE)?;
            if self.price(&trial, &neighborhoods)?.reduced_cost >= 0 {
                feasible_numerator = middle;
                feasible = trial;
            } else {
                infeasible_numerator = middle;
            }
        }
        Some(feasible)
    }

    fn price(&self, dual: &RouteMasterDual, neighborhoods: &NgNeighborhoods) -> Option<PricedNgRoute> {
        let customers = self.demands.len();
        if customers == 0
            || self.capacity == 0
            || self.demands.iter().any(|&demand| demand == 0 || demand > self.capacity)
            || dual.customers.len() != customers
            || self.costs.len() != customers + 1
            || neighborhoods.customers != customers
            || self.max_labels == 0
        {
            return None;
        }
        let states = self.capacity.checked_add(1)?.checked_mul(customers)?;
        let mut labels = vec![BTreeMap::<u16, NgLabel>::new(); states];
        let mut traces = Vec::new();
        for customer in 0..customers {
            let load = self.demands[customer];
            let index = load.checked_mul(customers)?.checked_add(customer)?;
            let memory = 1u16 << neighborhoods.position(customer, customer)?;
            let trace = traces.len();
            traces.push(NgTrace { customer, predecessor: None });
            labels[index].insert(
                memory,
                NgLabel {
                    reduced_cost: i128::from(self.costs[0][customer + 1])
                        .checked_mul(COLUMN_DUAL_SCALE)?
                        .checked_sub(dual.customers[customer])?,
                    trace,
                },
            );
            if traces.len() > self.max_labels {
                return None;
            }
        }
        let mut visited_transitions = 0usize;
        for load in 1..=self.capacity {
            for last in 0..customers {
                let index = load.checked_mul(customers)?.checked_add(last)?;
                if labels[index].is_empty() {
                    continue;
                }
                let current = labels[index].iter().map(|(&memory, label)| (memory, *label)).collect::<Vec<_>>();
                for (memory, label) in current {
                    for next in 0..customers {
                        visited_transitions = visited_transitions.saturating_add(1);
                        if visited_transitions.is_multiple_of(4_096)
                            && (self.stop.load(Ordering::Acquire) || Instant::now() >= self.deadline)
                        {
                            return None;
                        }
                        let Some(next_memory) = neighborhoods.extend(last, memory, next) else { continue };
                        let next_load = load.checked_add(self.demands[next])?;
                        if next_load > self.capacity {
                            continue;
                        }
                        let candidate = label
                            .reduced_cost
                            .checked_add(i128::from(self.costs[last + 1][next + 1]).checked_mul(COLUMN_DUAL_SCALE)?)?
                            .checked_sub(dual.customers[next])?;
                        let next_index = next_load.checked_mul(customers)?.checked_add(next)?;
                        if labels[next_index]
                            .iter()
                            .any(|(&known_memory, known)| known_memory & next_memory == known_memory && known.reduced_cost <= candidate)
                        {
                            continue;
                        }
                        let dominated = labels[next_index]
                            .iter()
                            .filter_map(|(&known_memory, known)| {
                                (known_memory & next_memory == next_memory && known.reduced_cost >= candidate).then_some(known_memory)
                            })
                            .collect::<Vec<_>>();
                        for known_memory in dominated {
                            labels[next_index].remove(&known_memory);
                        }
                        let trace = traces.len();
                        traces.push(NgTrace { customer: next, predecessor: Some(label.trace) });
                        labels[next_index].insert(next_memory, NgLabel { reduced_cost: candidate, trace });
                        if traces.len() > self.max_labels {
                            return None;
                        }
                    }
                }
            }
        }
        let mut best = None;
        for load in 1..=self.capacity {
            if self.stop.load(Ordering::Acquire) || Instant::now() >= self.deadline {
                return None;
            }
            for last in 0..customers {
                let index = load.checked_mul(customers)?.checked_add(last)?;
                for label in labels[index].values() {
                    let closed = label
                        .reduced_cost
                        .checked_add(i128::from(self.costs[last + 1][0]).checked_mul(COLUMN_DUAL_SCALE)?)?
                        .checked_sub(dual.fleet_lower)?
                        .checked_add(dual.fleet_upper)?;
                    if best.is_none_or(|(_, known)| closed < known) {
                        best = Some((label.trace, closed));
                    }
                }
            }
        }
        let (mut trace, reduced_cost) = best?;
        let mut reversed = Vec::new();
        loop {
            let current = *traces.get(trace)?;
            reversed.push(current.customer);
            let Some(previous) = current.predecessor else {
                break;
            };
            trace = previous;
        }
        reversed.reverse();
        let mut counts = vec![0u32; customers];
        for &customer in &reversed {
            counts[customer] = counts[customer].checked_add(1)?;
        }
        let first = *reversed.first()?;
        let last = *reversed.last()?;
        let mut cost = i128::from(self.costs[0][first + 1]);
        for pair in reversed.windows(2) {
            cost = cost.checked_add(i128::from(self.costs[pair[0] + 1][pair[1] + 1]))?;
        }
        cost = cost.checked_add(i128::from(self.costs[last + 1][0]))?;
        Some(PricedNgRoute { reduced_cost, counts, cost })
    }
}

fn minimum_one_tree(costs: &[Vec<i64>], penalties: &[i128], stop: &AtomicBool) -> Option<(i128, Vec<i32>)> {
    let customers = costs.len().checked_sub(1)?;
    if customers == 0 {
        return Some((0, vec![0]));
    }
    let edge = |from: usize, to: usize| i128::from(costs[from][to]) + penalties[from] + penalties[to];
    let mut degrees = vec![0i32; costs.len()];
    let mut total = 0i128;
    if customers > 1 {
        let mut included = vec![false; costs.len()];
        let mut key = vec![i128::MAX; costs.len()];
        let mut parent = vec![0usize; costs.len()];
        key[1] = 0;
        for _ in 0..customers {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let node = (1..costs.len()).filter(|&node| !included[node]).min_by_key(|&node| key[node])?;
            included[node] = true;
            if node != 1 {
                total = total.checked_add(key[node])?;
                degrees[node] += 1;
                degrees[parent[node]] += 1;
            }
            for next in 1..costs.len() {
                let weight = edge(node, next);
                if !included[next] && node != next && weight < key[next] {
                    key[next] = weight;
                    parent[next] = node;
                }
            }
        }
    }
    let mut depot_edges = (1..costs.len()).map(|customer| (edge(0, customer), customer)).collect::<Vec<_>>();
    depot_edges.sort_unstable();
    if customers == 1 {
        let (weight, customer) = depot_edges[0];
        total = total.checked_add(2 * weight)?;
        degrees[0] = 2;
        degrees[customer] = 2;
    } else {
        for &(weight, customer) in depot_edges.iter().take(2) {
            total = total.checked_add(weight)?;
            degrees[0] += 1;
            degrees[customer] += 1;
        }
    }
    Some((total, degrees))
}

fn route_columns(costs: &[Vec<i64>], demands: &[i64], capacity: i64, stop: &AtomicBool) -> Option<Vec<i64>> {
    let customers = demands.len();
    let states = 1usize.checked_shl(u32::try_from(customers).ok()?)?;
    let mut loads = vec![0i64; states];
    for mask in 1..states {
        if mask.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
            return None;
        }
        let bit = mask.trailing_zeros() as usize;
        loads[mask] = loads[mask & (mask - 1)].checked_add(demands[bit])?;
    }
    let mut paths = vec![INF; states.checked_mul(customers)?];
    for customer in 0..customers {
        if demands[customer] <= capacity {
            paths[(1usize << customer) * customers + customer] = costs[0][customer + 1];
        }
    }
    for mask in 1..states {
        if mask.is_multiple_of(256) && stop.load(Ordering::Relaxed) {
            return None;
        }
        if loads[mask] > capacity {
            continue;
        }
        for last in 0..customers {
            if mask & (1usize << last) == 0 {
                continue;
            }
            let previous = mask ^ (1usize << last);
            if previous == 0 {
                continue;
            }
            let best = (0..customers)
                .filter(|&before| previous & (1usize << before) != 0)
                .filter_map(|before| {
                    let prefix = paths[previous * customers + before];
                    (prefix != INF).then(|| prefix.checked_add(costs[before + 1][last + 1])).flatten()
                })
                .min()
                .unwrap_or(INF);
            paths[mask * customers + last] = best;
        }
    }
    let mut route_cost = vec![INF; states];
    for mask in 1..states {
        if mask.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
            return None;
        }
        if loads[mask] > capacity {
            continue;
        }
        route_cost[mask] = (0..customers)
            .filter(|&last| mask & (1usize << last) != 0)
            .filter_map(|last| {
                let path = paths[mask * customers + last];
                (path != INF).then(|| path.checked_add(costs[last + 1][0])).flatten()
            })
            .min()
            .unwrap_or(INF);
    }
    Some(route_cost)
}

fn subset_sum(mask: usize, values: &[i128]) -> i128 {
    values.iter().enumerate().filter(|&(index, _)| mask & (1usize << index) != 0).map(|(_, &value)| value).sum()
}

fn schedule_bound(schedule: &crate::model::list::Schedule, stop: &AtomicBool) -> Option<DualBound> {
    if !schedule.minimize_makespan || schedule.intervals.is_empty() {
        return None;
    }
    let mut durations = Vec::with_capacity(schedule.intervals.len());
    for interval in &schedule.intervals {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        durations.push(if interval.optional {
            0
        } else if interval.modes.is_empty() {
            interval.duration
        } else {
            let mut shortest = i64::MAX;
            for (mode_index, mode) in interval.modes.iter().enumerate() {
                if mode_index.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                    return None;
                }
                shortest = shortest.min(mode.duration);
            }
            shortest
        });
    }
    let mut outgoing = Vec::with_capacity(durations.len());
    for _ in 0..durations.len() {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        outgoing.push(Vec::new());
    }
    let mut indegree = vec![0usize; durations.len()];
    for (edge, &(before, after)) in schedule.precedences.iter().enumerate() {
        if edge.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
            return None;
        }
        if schedule.intervals.get(before)?.optional || schedule.intervals.get(after)?.optional {
            continue;
        }
        outgoing[before].push(after);
        indegree[after] += 1;
    }
    let mut queue = Vec::new();
    for (interval, &degree) in indegree.iter().enumerate().take(durations.len()) {
        if interval.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
            return None;
        }
        if degree == 0 {
            queue.push(interval);
        }
    }
    let mut earliest = vec![0i64; durations.len()];
    let mut visited = 0usize;
    while let Some(interval) = queue.pop() {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        visited += 1;
        let end = earliest[interval].checked_add(durations[interval])?;
        for &next in &outgoing[interval] {
            earliest[next] = earliest[next].max(end);
            indegree[next] -= 1;
            if indegree[next] == 0 {
                queue.push(next);
            }
        }
    }
    if visited != durations.len() {
        return None;
    }
    let mut bound = 0i64;
    for (index, (&start, &duration)) in earliest.iter().zip(&durations).enumerate() {
        if index.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
            return None;
        }
        bound = bound.max(start.saturating_add(duration));
    }
    for resource in &schedule.resources {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let resource_bound = match resource {
            Resource::NoOverlap(intervals) => {
                let mut seen = vec![false; durations.len()];
                let mut total = 0i64;
                for (index, &interval) in intervals.iter().enumerate() {
                    if index.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                        return None;
                    }
                    if interval >= durations.len() {
                        return None;
                    }
                    if !seen[interval] && !schedule.intervals[interval].optional {
                        seen[interval] = true;
                        total = total.checked_add(durations[interval])?;
                    }
                }
                total
            }
            Resource::Cumulative { demands, capacity } if *capacity > 0 => {
                let mut energy = 0i128;
                for (index, &(interval, demand)) in demands.iter().enumerate() {
                    if index.is_multiple_of(1_024) && stop.load(Ordering::Relaxed) {
                        return None;
                    }
                    if schedule.intervals.get(interval).is_some_and(|spec| !spec.optional) {
                        energy = energy.checked_add(i128::from(durations[interval]).checked_mul(i128::from(demand))?)?;
                    }
                }
                i64::try_from((energy + i128::from(*capacity) - 1) / i128::from(*capacity)).ok()?
            }
            Resource::Cumulative { .. } | Resource::MachineNoOverlap => 0,
        };
        bound = bound.max(resource_bound);
    }
    Some(DualBound { value: bound, method: "critical-path/resource relaxation", stats: SolveStats::default() })
}
