//! Certified objective bounds for collection models.
//!
//! Every method in this module is a relaxation. It may return no bound, but it
//! must never return a heuristic estimate as a certificate.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::engines::ls::lists::eval::eval_expr;
use crate::model::list::{BoundReport, CollectionModel, CollectionSolution, Iterable, Op, ReduceOp, Resource};

const MAX_COLUMN_GENERATION_CUSTOMERS: usize = 16;
const COLUMN_DUAL_SCALE: i128 = 1 << 12;
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
            best = stronger(best, routing.assignment_bound(), true);
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
    let mut contributions = vec![vec![0i128; model.items.len()]; model.lists];
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
    let min_routes = if model.items.is_empty() { 0 } else { capacity_relaxation(model).map_or(1, |capacity| capacity.min_routes.max(1)) };
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
    Some(DualBound {
        value: i64::try_from(best?).ok()?,
        method: if min_routes > 1 { "capacity assignment relaxation" } else { "used-list relaxation" },
    })
}

#[derive(Clone)]
struct CapacityRelaxation {
    demands: Vec<i64>,
    capacity: i64,
    min_routes: usize,
}

fn capacity_relaxation(model: &CollectionModel) -> Option<CapacityRelaxation> {
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
        for &item in &model.items {
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
                return (total == 0).then_some(CapacityRelaxation { demands: family.demands, capacity: 0, min_routes: 0 });
            }
            let min_routes = usize::try_from((total + i128::from(family.capacity) - 1) / i128::from(family.capacity)).ok()?;
            Some(CapacityRelaxation { demands: family.demands, capacity: family.capacity, min_routes })
        })
        .max_by_key(|family| family.min_routes)
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
        let mut reference = None;
        for reductions in by_list {
            if reductions.iter().any(|reduction| reduction.coeff < 0) {
                return None;
            }
            let mut max_raw = vec![0i128; reductions.len()];
            let mut matrix = vec![vec![0i64; nodes.len()]; nodes.len()];
            for (from_index, &from) in nodes.iter().enumerate() {
                if stop.load(Ordering::Relaxed) {
                    return None;
                }
                for (to_index, &to) in nodes.iter().enumerate() {
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
                if reference != &matrix {
                    return None;
                }
            } else {
                reference = Some(matrix);
            }
        }
        let costs = reference?;
        let max_edges = i128::try_from(model.items.len().saturating_add(model.lists)).ok()?;
        let max_cost = i128::from(costs.iter().flatten().copied().max().unwrap_or(0));
        i64::try_from(max_cost.checked_mul(max_edges)?).ok()?;
        let symmetric = (0..costs.len()).all(|i| (0..costs.len()).all(|j| costs[i][j] == costs[j][i]));
        Some(Self { costs, routes: model.lists, symmetric, capacity: capacity_relaxation(model) })
    }

    fn min_routes(&self) -> usize {
        self.capacity.as_ref().map_or(1, |capacity| capacity.min_routes.max(1))
    }

    fn assignment_bound(&self) -> Option<DualBound> {
        let customers = self.costs.len().checked_sub(1)?;
        let routes = self.min_routes();
        if routes > customers || routes > self.routes {
            return None;
        }
        let mut outgoing = 0i128;
        let mut incoming = 0i128;
        for customer in 1..=customers {
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
            let (modified, degrees) = minimum_one_tree(&self.costs, &penalties)?;
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
                let customer = (offset + iteration) % customers;
                let slack = (1..states)
                    .filter(|&mask| active[mask] && mask & (1usize << customer) != 0)
                    .map(|mask| i128::from(route_cost[mask]) * COLUMN_DUAL_SCALE - subset_sum(mask, &proposal))
                    .min()?;
                if slack > 0 {
                    proposal[customer] += if iteration % 4 == 3 { slack } else { 3 * slack / 4 };
                }
            }
            let (priced_mask, reduced_cost) = (1..states)
                .filter(|&mask| route_cost[mask] != INF)
                .map(|mask| (mask, i128::from(route_cost[mask]) * COLUMN_DUAL_SCALE - subset_sum(mask, &proposal)))
                .min_by_key(|&(_, reduced_cost)| reduced_cost)?;
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

fn minimum_one_tree(costs: &[Vec<i64>], penalties: &[i128]) -> Option<(i128, Vec<i32>)> {
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
    let durations = schedule
        .intervals
        .iter()
        .map(|interval| {
            if interval.optional {
                0
            } else if interval.modes.is_empty() {
                interval.duration
            } else {
                interval.modes.iter().map(|mode| mode.duration).min().unwrap_or(0)
            }
        })
        .collect::<Vec<_>>();
    let mut outgoing = vec![Vec::new(); durations.len()];
    let mut indegree = vec![0usize; durations.len()];
    for &(before, after) in &schedule.precedences {
        if schedule.intervals.get(before)?.optional || schedule.intervals.get(after)?.optional {
            continue;
        }
        outgoing[before].push(after);
        indegree[after] += 1;
    }
    let mut queue = (0..durations.len()).filter(|&interval| indegree[interval] == 0).collect::<Vec<_>>();
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
    let mut bound = earliest.iter().zip(&durations).map(|(&start, &duration)| start.saturating_add(duration)).max().unwrap_or(0);
    for resource in &schedule.resources {
        let resource_bound = match resource {
            Resource::NoOverlap(intervals) => {
                let mut seen = vec![false; durations.len()];
                let mut total = 0i64;
                for &interval in intervals {
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
                let energy = demands
                    .iter()
                    .filter(|&&(interval, _)| schedule.intervals.get(interval).is_some_and(|spec| !spec.optional))
                    .try_fold(0i128, |sum, &(interval, demand)| {
                    sum.checked_add(i128::from(durations[interval]).checked_mul(i128::from(demand))?)
                })?;
                i64::try_from((energy + i128::from(*capacity) - 1) / i128::from(*capacity)).ok()?
            }
            Resource::Cumulative { .. } | Resource::MachineNoOverlap => 0,
        };
        bound = bound.max(resource_bound);
    }
    Some(DualBound { value: bound, method: "critical-path/resource relaxation" })
}
