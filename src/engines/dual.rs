//! Certified objective bounds for collection models.
//!
//! Every method in this module is a relaxation. It may return no bound, but it
//! must never return a heuristic estimate as a certificate.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::engines::ls::lists::eval::eval_expr;
use crate::model::list::{
    BoundReport, CollectionModel, CollectionSolution, Expr, ExprArena, ExprId, GlobalConstraint, Iterable, Op, ReduceOp, Resource,
};

const MAX_COLUMN_GENERATION_CUSTOMERS: usize = 16;
const COLUMN_DUAL_SCALE: i128 = 1 << 12;
const MAX_EXACT_PACKING_ITEMS: usize = 35;
const PACKING_NODE_LIMIT: u64 = 20_000;
const MAX_ENERGY_INTERVAL_PAIRS: usize = 50_000;
const MAX_VRPTW_METRIC_NODES: usize = 128;
const MAX_ROUTING_RELAXATION_NODES: usize = 2_048;
const INF: i64 = i64::MAX;

/// A certified bound before a primal solution is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualBound {
    pub value: i64,
    pub method: &'static str,
}

/// Compute the strongest supported bound on the primary objective.
pub fn compute(model: &CollectionModel, stop: &AtomicBool) -> Option<DualBound> {
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
        (Some(current), Some(candidate)) => {
            if (minimizing && candidate.value > current.value) || (!minimizing && candidate.value < current.value) {
                Some(candidate)
            } else {
                Some(current)
            }
        }
    }
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
    Some(DualBound { value: i64::try_from(value).ok()?, method: "assignment relaxation" })
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
    Some(DualBound { value: i64::try_from(best?).ok()?, method })
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
        for reductions in by_list {
            if reductions.iter().any(|reduction| reduction.coeff < 0) {
                return None;
            }
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
            let max_route_edges = i128::try_from(model.items.len().saturating_add(1)).ok()?;
            for (&raw, reduction) in max_raw.iter().zip(&reductions) {
                let worst = raw.checked_mul(max_route_edges)?.checked_mul(i128::from(reduction.coeff))?;
                i64::try_from(worst).ok()?;
            }
            if matrix[0][0] != 0 || matrix.iter().flatten().any(|&cost| cost < 0) {
                return None;
            }
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
                reference = Some(matrix);
            }
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
        Some(DualBound { value: i64::try_from(outgoing.max(incoming)).ok()?, method: "directed assignment relaxation" })
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
        Some(DualBound { value: i64::try_from(best).ok()?, method: "Held-Karp 1-tree" })
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
        Some(DualBound { value: i64::try_from(best_value.div_euclid(COLUMN_DUAL_SCALE)).ok()?, method: "stabilized VRP column generation" })
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
    Some(DualBound { value: bound, method: "critical-path/resource relaxation" })
}
