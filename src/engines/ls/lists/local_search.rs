//! Private list local-search heuristic implementation.
//!
//! This module scores and moves over an already-declared model. It must not be
//! the only place where a new modeling feature exists: add the feature to the
//! shared Rust model and backend classifier first, then teach this heuristic to
//! search it as a fallback or incumbent generator.

use std::cell::{Cell, RefCell};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use smallvec::SmallVec;

use super::alns::{
    build_candidate, build_candidate_bounded, build_macro_candidate_bounded, routing_compound_structural_floor,
    routing_relink_structural_floor, routing_route_elimination_floor, AcceptanceKind, AlnsBuildRun, AlnsBuildStatus, AlnsController,
    AlnsWorkBudget, MacroOperator, SearchProfile,
};
use super::elite::{elite_archive_budget, elite_selection_budget, path_relink_bounded, EliteOperationStatus, ElitePool, PathRelinkStatus};
use super::eval::{eval_reduction, violation_of, INFEASIBLE};
use super::incremental::{EvalScratch, EvaluationInterrupted, InsertView, ListView, ReductionCache};
use super::metrics::{ListSearchMetrics, MetricsRecorder};
use super::moves::{
    apply_move, best_improving_move, better, search_routing_neighborhood, shuffle, snapshot, trial_list_score_view, NeighborhoodKind,
    RoutingIndexCache, RoutingScanMemory, RoutingScanWorkspace, ScanMode, ScanOutcome, SearchMemory, WorkBudget,
};
use super::portfolio::WorkerCoordination;
use super::routing_search::{RoutingSearchControl, SliceKind};
use crate::engines::dual;
use crate::mix64;
use crate::model::list::scan::{time_window_scan_signature, TimeWindowScanSpec};
use crate::model::list::{
    CollectionModel, CollectionSolution, Constraint, Expr, ExprId, GlobalConstraint, Iterable, MaxTerm, ReduceOp, Reduction,
};

pub(super) struct Globals {
    cons: Vec<GlobalConstraint>,
    pub(super) value_to_idx: HashMap<i32, usize>,
    of_idx: Vec<Vec<usize>>,
}

impl Globals {
    pub(super) fn build(model: &CollectionModel, stop: &AtomicBool) -> Self {
        let mut value_to_idx: HashMap<i32, usize> = HashMap::with_capacity(model.items.len());
        for (index, &value) in model.items.iter().enumerate() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            value_to_idx.insert(value, index);
        }
        let mut of_idx = vec![Vec::new(); model.items.len()];
        for (g, c) in model.globals.iter().enumerate() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            for v in c.items() {
                if let Some(&i) = value_to_idx.get(&v) {
                    if !of_idx[i].contains(&g) {
                        of_idx[i].push(g);
                    }
                }
            }
        }
        Self { cons: model.globals.clone(), value_to_idx, of_idx }
    }

    /// List index of `value`, honouring any `(value, list)` overrides.
    fn list_of(&self, item_list: &[usize], value: i32, ov: &[(i32, usize)]) -> usize {
        for &(v, l) in ov {
            if v == value {
                return l;
            }
        }
        self.value_to_idx.get(&value).map_or(0, |&i| item_list[i])
    }

    fn list_of_interruptible(
        &self,
        item_list: &[usize],
        value: i32,
        overrides: &[(i32, usize)],
        stop: &AtomicBool,
    ) -> Result<usize, EvaluationInterrupted> {
        for (index, &(candidate, list)) in overrides.iter().enumerate() {
            if index.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                return Err(EvaluationInterrupted);
            }
            if candidate == value {
                return Ok(list);
            }
        }
        if stop.load(Ordering::Relaxed) {
            Err(EvaluationInterrupted)
        } else {
            Ok(self.value_to_idx.get(&value).map_or(0, |&index| item_list[index]))
        }
    }

    /// Violation of one global constraint under the current `item_list` plus
    /// overrides. `ListLe` penalises by how many lists out of order; `SameList`
    /// by the list-index distance, so both are smooth for local search.
    fn one(&self, c: &GlobalConstraint, item_list: &[usize], ov: &[(i32, usize)]) -> i64 {
        match c {
            GlobalConstraint::ListLe { before, after } => {
                let lb = self.list_of(item_list, *before, ov) as i64;
                let la = self.list_of(item_list, *after, ov) as i64;
                (lb - la).max(0)
            }
            GlobalConstraint::SameList { a, b } => {
                let la = self.list_of(item_list, *a, ov) as i64;
                let lb = self.list_of(item_list, *b, ov) as i64;
                (la - lb).abs()
            }
            GlobalConstraint::DifferentList { a, b } => i64::from(self.list_of(item_list, *a, ov) == self.list_of(item_list, *b, ov)),
            GlobalConstraint::AllSameList { items } => {
                let mut owners = items.iter().map(|item| self.list_of(item_list, *item, ov) as i64);
                let Some(first) = owners.next() else { return 0 };
                let (lo, hi) = owners.fold((first, first), |(lo, hi), owner| (lo.min(owner), hi.max(owner)));
                hi - lo
            }
            GlobalConstraint::AllDifferentLists { items } => {
                let mut seen = HashSet::with_capacity(items.len());
                items
                    .iter()
                    .fold(0i64, |violation, item| violation.saturating_add(i64::from(!seen.insert(self.list_of(item_list, *item, ov)))))
            }
            GlobalConstraint::ListDistance { a, b, min, max } => {
                let distance = self.list_of(item_list, *a, ov).abs_diff(self.list_of(item_list, *b, ov));
                i64::try_from(min.saturating_sub(distance).saturating_add(distance.saturating_sub(*max))).unwrap_or(i64::MAX)
            }
        }
    }

    fn one_interruptible(
        &self,
        constraint: &GlobalConstraint,
        item_list: &[usize],
        overrides: &[(i32, usize)],
        stop: &AtomicBool,
    ) -> Result<i64, EvaluationInterrupted> {
        let list_of = |value| self.list_of_interruptible(item_list, value, overrides, stop);
        match constraint {
            GlobalConstraint::ListLe { before, after } => {
                let before = list_of(*before)? as i64;
                let after = list_of(*after)? as i64;
                Ok((before - after).max(0))
            }
            GlobalConstraint::SameList { a, b } => {
                let left = list_of(*a)? as i64;
                let right = list_of(*b)? as i64;
                Ok((left - right).abs())
            }
            GlobalConstraint::DifferentList { a, b } => Ok(i64::from(list_of(*a)? == list_of(*b)?)),
            GlobalConstraint::AllSameList { items } => {
                let Some(&first) = items.first() else { return Ok(0) };
                let first = list_of(first)? as i64;
                let (mut lo, mut hi) = (first, first);
                for (index, &item) in items.iter().enumerate().skip(1) {
                    if index.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                        return Err(EvaluationInterrupted);
                    }
                    let owner = list_of(item)? as i64;
                    lo = lo.min(owner);
                    hi = hi.max(owner);
                }
                Ok(hi - lo)
            }
            GlobalConstraint::AllDifferentLists { items } => {
                let mut seen = HashSet::with_capacity(items.len());
                let mut violation = 0i64;
                for (index, &item) in items.iter().enumerate() {
                    if index.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                        return Err(EvaluationInterrupted);
                    }
                    violation = violation.saturating_add(i64::from(!seen.insert(list_of(item)?)));
                }
                Ok(violation)
            }
            GlobalConstraint::ListDistance { a, b, min, max } => {
                let distance = list_of(*a)?.abs_diff(list_of(*b)?);
                Ok(i64::try_from(min.saturating_sub(distance).saturating_add(distance.saturating_sub(*max))).unwrap_or(i64::MAX))
            }
        }
    }

    /// Total violation over all global constraints.
    pub(super) fn total(&self, item_list: &[usize]) -> i64 {
        self.cons.iter().fold(0i64, |acc, c| acc.saturating_add(self.one(c, item_list, &[])))
    }

    /// Change in total global violation if the listed items moved to new lists.
    pub(super) fn delta(&self, item_list: &[usize], overrides: &[(i32, usize)]) -> i64 {
        let stop = AtomicBool::new(false);
        self.delta_interruptible(item_list, overrides, &stop).expect("an uninterrupted global delta must complete")
    }

    pub(super) fn delta_interruptible(
        &self,
        item_list: &[usize],
        overrides: &[(i32, usize)],
        stop: &AtomicBool,
    ) -> Result<i64, EvaluationInterrupted> {
        if stop.load(Ordering::Relaxed) {
            return Err(EvaluationInterrupted);
        }
        let mut affected: Vec<usize> = Vec::new();
        let mut affected_seen = HashSet::new();
        let mut work = 0usize;
        for &(v, _) in overrides {
            if work.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                return Err(EvaluationInterrupted);
            }
            if let Some(&i) = self.value_to_idx.get(&v) {
                for &g in &self.of_idx[i] {
                    if work.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                        return Err(EvaluationInterrupted);
                    }
                    if affected_seen.insert(g) {
                        affected.push(g);
                    }
                    work = work.saturating_add(1);
                }
            }
            work = work.saturating_add(1);
        }
        let mut d = 0i64;
        for &g in &affected {
            if work.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                return Err(EvaluationInterrupted);
            }
            let c = &self.cons[g];
            let next = self.one_interruptible(c, item_list, overrides, stop)?;
            let old = self.one_interruptible(c, item_list, &[], stop)?;
            d = d.saturating_add(next.saturating_sub(old));
            work = work.saturating_add(1);
        }
        if stop.load(Ordering::Relaxed) {
            Err(EvaluationInterrupted)
        } else {
            Ok(d)
        }
    }
}

/// Objective reductions (grouped by tier) and constraints, both grouped by the
/// list they read so a move only rescores the lists it touches. `senses[t]` is
/// true when tier `t` is minimised.
pub(super) struct PerList {
    pub(super) objective: Vec<Vec<Vec<Reduction>>>,
    pub(super) max_objective: Vec<Vec<MaxTerm>>,
    pub(super) objective_delta: Vec<Vec<Vec<ReductionDeltaKind>>>,
    pub(super) constraints: Vec<Vec<Constraint>>,
    pub(super) constraint_delta: Vec<Vec<ReductionDeltaKind>>,
    pub(super) senses: Vec<bool>,
    pub(super) tiers: usize,
    pub(super) globals: Globals,
    /// Whether route-edge reductions exist. Routing-only moves such as Or-opt
    /// are useful here and wasteful on order-independent packing models.
    pub(super) has_edges: bool,
    /// Per-list route boundaries from the first edge reduction on that list.
    pub(super) route_bounds: Vec<Option<(i32, i32)>>,
    /// Nearest-neighbor candidate edges for routing moves, when a direct matrix
    /// edge objective is available.
    pub(super) candidates: Option<CandidateNeighbors>,
    /// Routing data parsed once from the reductions and reused by construction.
    routing: Option<RoutingSignature>,
    /// Whether cost-refiner moves (2-opt* / cross / reverse) may prune by the
    /// geometric candidate lists even while a route still overflows.
    pub(super) infeas_cand: bool,
    /// Whether permuting the list variables leaves the model unchanged. Search
    /// then keeps a single representative empty list in each neighbourhood,
    /// avoiding the large symmetry created by optional vehicle fleets.
    pub(super) interchangeable_lists: Cell<bool>,
    penalties: GlsPenalties,
    routing_gls: Option<RoutingGls>,
    pub(super) metrics: MetricsRecorder,
}

struct GlsPenalties {
    constraints: RefCell<Vec<Vec<i64>>>,
    objectives: RefCell<Vec<Vec<Vec<i64>>>>,
}

struct RoutingGls {
    matrix: Arc<Vec<Vec<i64>>>,
    penalties: RefCell<Vec<Vec<i64>>>,
    lambda: Cell<i64>,
    coeff: i64,
    symmetric: bool,
}

struct RoutingSignature {
    depot: i32,
    matrix: Arc<Vec<Vec<i64>>>,
    demands: Option<Arc<Vec<i64>>>,
    capacity: Option<i64>,
    has_time_windows: bool,
    time_windows: Option<TimeWindowScanSpec>,
    has_fleet_objective: bool,
    reverse_equivalent: bool,
    distance_scale: u64,
    demand_scale: u64,
    time_scale: u64,
    slack_scale: u64,
}

pub(super) struct CandidateNeighbors {
    /// Directed, cost-ordered k-nearest neighbours used by construction and
    /// destroy/repair operators.
    map: HashMap<i32, Vec<i32>>,
    /// Symmetric candidate graph used by routing neighbourhoods.  It contains
    /// at most two adjacency entries per directed kNN edge, including route
    /// boundary nodes such as the depot.
    routing_map: HashMap<i32, Vec<i32>>,
    /// Semantic neighbors derived from a recognized time-window scan.
    semantic_map: HashMap<i32, Vec<i32>>,
}

impl CandidateNeighbors {
    fn build(
        model: &CollectionModel,
        matrix: Arc<Vec<Vec<i64>>>,
        routing: Option<&RoutingSignature>,
        limit: usize,
        stop: &AtomicBool,
    ) -> Option<Self> {
        let n = matrix.len();
        if n == 0 || matrix.iter().any(|row| row.len() != n) {
            return None;
        }
        let mut values = model.items.clone();
        for reduction in model.objectives.iter().flat_map(|tier| tier.reductions()) {
            if let Iterable::Edges { start, end, .. } = &reduction.iterable {
                values.push(*start);
                values.push(*end);
            }
        }
        for constraint in &model.constraints {
            if let Iterable::Edges { start, end, .. } = &constraint.reduction.iterable {
                values.push(*start);
                values.push(*end);
            }
        }
        values.sort_unstable();
        values.dedup();
        if values.iter().any(|&value| usize::try_from(value).ok().is_none_or(|idx| idx >= n)) {
            return None;
        }

        let mut targets = model.items.clone();
        targets.extend(values.iter().copied());
        targets.sort_unstable();
        targets.dedup();
        let keep = limit.min(targets.len().saturating_sub(1));

        let mut map = HashMap::with_capacity(values.len());
        let mut semantic_map = HashMap::with_capacity(values.len());
        for &from in &values {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let from_idx = usize::try_from(from).ok()?;
            // Keep only the best `limit` entries while scanning the row.  A
            // full sort here used to make candidate construction O(n² log n)
            // and left one whole row uninterruptible.
            let mut heap = BinaryHeap::with_capacity(keep);
            let mut semantic_heap = BinaryHeap::with_capacity(keep);
            for (target_at, &to) in targets.iter().enumerate() {
                if target_at.is_multiple_of(256) && stop.load(Ordering::Relaxed) {
                    return None;
                }
                if to == from {
                    continue;
                }
                let to_idx = usize::try_from(to).ok()?;
                let entry = (matrix[from_idx][to_idx], to);
                if heap.len() < keep {
                    heap.push(entry);
                } else if heap.peek().is_some_and(|worst| entry < *worst) {
                    heap.pop();
                    heap.push(entry);
                }
                if let Some(routing) = routing.filter(|signature| signature.time_windows.is_some()) {
                    let semantic_entry = (routing_relatedness(routing, from, to, true), to);
                    if semantic_heap.len() < keep {
                        semantic_heap.push(semantic_entry);
                    } else if semantic_heap.peek().is_some_and(|worst| semantic_entry < *worst) {
                        semantic_heap.pop();
                        semantic_heap.push(semantic_entry);
                    }
                }
            }
            let mut near = heap.into_vec();
            near.sort_unstable_by_key(|&(cost, to)| (cost, to));
            map.insert(from, near.into_iter().map(|(_, to)| to).collect());
            if !semantic_heap.is_empty() {
                let mut related = semantic_heap.into_vec();
                related.sort_unstable();
                semantic_map.insert(from, related.into_iter().map(|(_, to)| to).collect());
            }
        }

        // Routing treats a candidate edge as undirected.  Build that view in
        // deterministic source/cost order without allowing mutual directed
        // arcs to duplicate work in every granular scan.
        let directed_edges = map.values().map(Vec::len).sum();
        let mut routing_map: HashMap<i32, Vec<i32>> = HashMap::with_capacity(values.len());
        let mut seen = HashSet::with_capacity(directed_edges);
        let mut visited = 0usize;
        for &from in &values {
            for &to in map.get(&from).map_or(&[][..], Vec::as_slice) {
                if visited.is_multiple_of(256) && stop.load(Ordering::Relaxed) {
                    return None;
                }
                visited += 1;
                let edge = if from < to { (from, to) } else { (to, from) };
                if seen.insert(edge) {
                    routing_map.entry(from).or_default().push(to);
                    routing_map.entry(to).or_default().push(from);
                }
            }
        }
        Some(Self { map, routing_map, semantic_map })
    }

    pub(super) fn contains(&self, a: i32, b: i32) -> bool {
        self.map.get(&a).is_some_and(|near| near.contains(&b)) || self.map.get(&b).is_some_and(|near| near.contains(&a))
    }

    pub(super) fn neighbors(&self, item: i32) -> &[i32] {
        self.map.get(&item).map_or(&[], Vec::as_slice)
    }

    pub(super) fn routing_neighbors(&self, item: i32) -> &[i32] {
        self.routing_map.get(&item).map_or(&[], Vec::as_slice)
    }

    pub(super) fn semantic_neighbors(&self, item: i32) -> &[i32] {
        self.semantic_map.get(&item).map_or(&[], Vec::as_slice)
    }

    pub(super) fn contains_semantic(&self, left: i32, right: i32) -> bool {
        self.contains(left, right)
            || self.semantic_map.get(&left).is_some_and(|near| near.contains(&right))
            || self.semantic_map.get(&right).is_some_and(|near| near.contains(&left))
    }
}

fn interleaved_construction_neighbors(neighbors: &CandidateNeighbors, item: i32, limit: usize) -> Vec<i32> {
    let geometric = neighbors.neighbors(item);
    let semantic = neighbors.semantic_neighbors(item);
    let mut combined = Vec::with_capacity(limit.min(geometric.len().saturating_add(semantic.len())));
    let mut seen = HashSet::with_capacity(combined.capacity());
    for index in 0..geometric.len().max(semantic.len()) {
        for candidate in [semantic.get(index), geometric.get(index)].into_iter().flatten() {
            if combined.len() >= limit {
                return combined;
            }
            if seen.insert(*candidate) {
                combined.push(*candidate);
            }
        }
    }
    combined
}

fn normalized_difference(left: i64, right: i64, scale: u64) -> u64 {
    let difference = (i128::from(left) - i128::from(right)).unsigned_abs();
    u64::try_from(difference.saturating_mul(1_000) / u128::from(scale.max(1))).unwrap_or(u64::MAX)
}

impl PerList {
    pub(super) fn minimizes_fleet(&self) -> bool {
        self.routing.as_ref().is_some_and(|routing| routing.has_fleet_objective)
    }

    /// Deterministic Shaw relatedness assembled from model semantics. Every
    /// component is normalized before weighting, so distance units cannot drown
    /// window, slack, demand, or route-membership information.
    pub(super) fn shaw_relatedness(&self, left: i32, right: i32, same_route: bool) -> u64 {
        let Some(routing) = &self.routing else {
            return u64::from(!same_route).saturating_mul(1_000);
        };
        routing_relatedness(routing, left, right, same_route)
    }

    /// Remaining normalized resource margin after a feasible routing edit.
    /// Fleet reduction maximizes this before distance so early insertions do
    /// not consume the only room needed by later customers.
    pub(super) fn routing_headroom(&self, route: &[i32]) -> i64 {
        let Some(routing) = &self.routing else { return 0 };
        let mut headroom = 0i128;
        if let (Some(demands), Some(capacity)) = (&routing.demands, routing.capacity) {
            let load = route.iter().fold(0i64, |total, &item| {
                total.saturating_add(usize::try_from(item).ok().and_then(|index| demands.get(index)).copied().unwrap_or(0))
            });
            let remaining = capacity.saturating_sub(load).max(0);
            headroom = headroom.saturating_add(i128::from(remaining).saturating_mul(1_000) / i128::from(capacity.max(1)));
        }
        if let Some(windows) = &routing.time_windows {
            let Ok(mut previous) = usize::try_from(routing.depot) else { return 0 };
            let mut departure = windows.earliest.get(previous).copied().unwrap_or(0);
            let mut minimum = i64::MAX;
            for &item in route.iter().chain(std::iter::once(&routing.depot)) {
                let Ok(current) = usize::try_from(item) else { return 0 };
                let travel = windows.travel.get(previous).and_then(|row| row.get(current)).copied().unwrap_or(i64::MAX);
                let earliest = windows.earliest.get(current).copied().unwrap_or(0);
                let latest = windows.latest_start.get(current).copied().unwrap_or(i64::MIN);
                let start = earliest.max(departure.saturating_add(travel));
                minimum = minimum.min(latest.saturating_sub(start));
                departure = start.saturating_add(windows.service.get(current).copied().unwrap_or(0));
                previous = current;
            }
            let time_margin = i128::from(minimum.max(0)).saturating_mul(1_000) / i128::from(routing.time_scale.max(1));
            headroom = headroom.saturating_add(time_margin);
        }
        i64::try_from(headroom).unwrap_or(i64::MAX)
    }
}

fn routing_relatedness(routing: &RoutingSignature, left: i32, right: i32, same_route: bool) -> u64 {
    let (Ok(left_index), Ok(right_index)) = (usize::try_from(left), usize::try_from(right)) else {
        return u64::MAX;
    };
    let spatial = routing
        .matrix
        .get(left_index)
        .and_then(|row| row.get(right_index))
        .copied()
        .map_or(1_000, |distance| normalized_difference(distance, 0, routing.distance_scale));
    let demand = routing.demands.as_ref().map_or(0, |demands| {
        demands
            .get(left_index)
            .copied()
            .zip(demands.get(right_index).copied())
            .map_or(0, |(left, right)| normalized_difference(left, right, routing.demand_scale))
    });
    let (window, slack, temporal_arc) = routing.time_windows.as_ref().map_or((0, 0, 0), |windows| {
        let Some((&left_earliest, &right_earliest, &left_latest, &right_latest)) = windows
            .earliest
            .get(left_index)
            .zip(windows.earliest.get(right_index))
            .zip(windows.latest_start.get(left_index).zip(windows.latest_start.get(right_index)))
            .map(|((left_earliest, right_earliest), (left_latest, right_latest))| {
                (left_earliest, right_earliest, left_latest, right_latest)
            })
        else {
            return (0, 0, 0);
        };
        let center_left = i128::from(left_earliest).saturating_add(i128::from(left_latest));
        let center_right = i128::from(right_earliest).saturating_add(i128::from(right_latest));
        let window = u64::try_from((center_left - center_right).unsigned_abs().saturating_mul(500) / u128::from(routing.time_scale.max(1)))
            .unwrap_or(u64::MAX);
        let slack = normalized_difference(
            left_latest.saturating_sub(left_earliest),
            right_latest.saturating_sub(right_earliest),
            routing.slack_scale,
        );
        let temporal_arc = windows.travel.get(left_index).and_then(|row| row.get(right_index)).copied().map_or(0, |travel| {
            let service = windows.service.get(left_index).copied().unwrap_or(0);
            normalized_difference(service.saturating_add(travel), 0, routing.time_scale)
        });
        (window, slack, temporal_arc)
    });
    spatial
        .saturating_mul(9)
        .saturating_add(window.saturating_mul(3))
        .saturating_add(slack.saturating_mul(2))
        .saturating_add(temporal_arc.saturating_mul(2))
        .saturating_add(demand.saturating_mul(2))
        .saturating_add(u64::from(!same_route).saturating_mul(5_000))
}

#[derive(Clone, Copy)]
pub(super) enum ReductionDeltaKind {
    ItemsCount,
    Used,
    Unsupported,
}

fn expr_is_arg(exprs: &[Expr], id: ExprId, arg: u8) -> bool {
    matches!(exprs.get(id.0 as usize), Some(Expr::Arg(a)) if *a == arg)
}

fn direct_edge_matrix(r: &Reduction) -> Option<Arc<Vec<Vec<i64>>>> {
    match r.arena.exprs.get(r.body.0 as usize) {
        Some(Expr::Matrix(matrix, row, col)) => {
            let direct_args = expr_is_arg(&r.arena.exprs, *row, 0) && expr_is_arg(&r.arena.exprs, *col, 1);
            direct_args.then(|| Arc::clone(matrix))
        }
        _ => None,
    }
}

fn direct_item_array(reduction: &Reduction) -> Option<Arc<Vec<i64>>> {
    let Expr::Array(values, index) = reduction.arena.exprs.get(reduction.body.0 as usize)? else {
        return None;
    };
    expr_is_arg(&reduction.arena.exprs, *index, 0).then(|| Arc::clone(values))
}

fn same_matrix(left: &Arc<Vec<Vec<i64>>>, right: &Arc<Vec<Vec<i64>>>) -> bool {
    Arc::ptr_eq(left, right) || left.as_ref() == right.as_ref()
}

fn symmetric_matrix(matrix: &[Vec<i64>]) -> bool {
    matrix.iter().enumerate().all(|(row, values)| {
        values.len() == matrix.len() && values.iter().take(row).enumerate().all(|(column, &value)| matrix[column].get(row) == Some(&value))
    })
}

fn reversal_invariant_reduction(reduction: &Reduction) -> bool {
    match &reduction.iterable {
        Iterable::Items(_) | Iterable::SetItems(_) => true,
        Iterable::Edges { start, end, .. } => start == end && direct_edge_matrix(reduction).is_some_and(|matrix| symmetric_matrix(&matrix)),
        Iterable::Scan { .. } | Iterable::Windows { .. } | Iterable::Pairs(_) => false,
    }
}

fn routing_signature(model: &CollectionModel) -> Option<RoutingSignature> {
    if !homogeneous_minimization_routing(model) {
        return None;
    }
    let mut route_bounds = vec![None; model.lists];
    let mut matrix = None;
    let mut inspect = |reduction: &Reduction, objective: bool| -> Option<()> {
        let Iterable::Edges { list, start, end, .. } = &reduction.iterable else {
            return Some(());
        };
        if *list >= model.lists {
            return None;
        }
        let bounds = (*start, *end);
        if route_bounds[*list].is_some_and(|existing| existing != bounds) {
            return None;
        }
        route_bounds[*list] = Some(bounds);
        if objective && matrix.is_none() {
            matrix = direct_edge_matrix(reduction);
        }
        Some(())
    };
    for tier in &model.objectives {
        for reduction in &tier.terms {
            inspect(reduction, true)?;
        }
        for reduction in tier.max_terms.iter().flatten().flat_map(|term| term.groups.iter().flatten()) {
            inspect(reduction, true)?;
        }
    }
    for constraint in &model.constraints {
        inspect(&constraint.reduction, false)?;
    }

    let matrix = matrix?;
    let (depot, end) = route_bounds.iter().flatten().next().copied()?;
    if depot != end
        || route_bounds.iter().any(Option::is_none)
        || route_bounds.iter().flatten().any(|&(start, finish)| start != depot || finish != depot)
    {
        return None;
    }
    let mut guided_lists = vec![false; model.lists];
    let mut inspect_objective_reduction = |reduction: &Reduction| -> Option<()> {
        let Iterable::Edges { list, start, end, .. } = &reduction.iterable else {
            return Some(());
        };
        if *start != depot || *end != depot || *list >= model.lists {
            return None;
        }
        if matches!(reduction.op, ReduceOp::Sum)
            && reduction.coeff > 0
            && direct_edge_matrix(reduction).is_some_and(|other| same_matrix(&other, &matrix))
        {
            guided_lists[*list] = true;
        }
        Some(())
    };
    for tier in &model.objectives {
        for reduction in &tier.terms {
            inspect_objective_reduction(reduction)?;
        }
        for reduction in tier.max_terms.iter().flatten().flat_map(|term| term.groups.iter().flatten()) {
            inspect_objective_reduction(reduction)?;
        }
    }
    if guided_lists.iter().any(|guided| !guided) {
        return None;
    }
    for constraint in &model.constraints {
        if let Iterable::Edges { start, end, .. } = &constraint.reduction.iterable {
            if *start != depot || *end != depot {
                return None;
            }
        }
    }
    let capacity_constraint = model.constraints.iter().find(|constraint| {
        matches!(constraint.op, crate::model::list::Op::Le)
            && matches!(constraint.reduction.op, ReduceOp::Sum)
            && matches!(constraint.reduction.iterable, Iterable::Items(_))
            && direct_item_array(&constraint.reduction).is_some()
    });
    let demands = capacity_constraint.and_then(|constraint| direct_item_array(&constraint.reduction));
    let capacity = capacity_constraint.map(|constraint| constraint.rhs);
    let has_time_windows = model.constraints.iter().any(|constraint| matches!(constraint.reduction.iterable, Iterable::Scan { .. }));
    let time_windows = model.constraints.iter().find_map(time_window_scan_signature);
    let has_fleet_objective = model.objectives.first().is_some_and(|tier| {
        tier.minimize
            && tier.max_terms.as_ref().is_none_or(|terms| terms.is_empty())
            && tier.terms.len() == model.lists
            && tier.terms.iter().all(|reduction| {
                matches!(reduction.op, ReduceOp::Used) && matches!(reduction.iterable, Iterable::Items(_)) && reduction.coeff > 0
            })
    });
    let reverse_equivalent = model
        .objectives
        .iter()
        .flat_map(|tier| tier.reductions())
        .chain(model.constraints.iter().map(|constraint| &constraint.reduction))
        .all(reversal_invariant_reduction);
    let distance_scale = matrix.iter().flat_map(|row| row.iter()).map(|value| value.unsigned_abs()).max().unwrap_or(1).max(1);
    let demand_scale = demands
        .as_ref()
        .and_then(|values| {
            let minimum = values.iter().copied().min()?;
            let maximum = values.iter().copied().max()?;
            Some(maximum.saturating_sub(minimum).unsigned_abs().max(1))
        })
        .unwrap_or(1);
    let (time_scale, slack_scale) = time_windows.as_ref().map_or((1, 1), |windows| {
        let earliest = windows.earliest.iter().copied().min().unwrap_or(0);
        let latest = windows.latest_start.iter().copied().max().unwrap_or(earliest);
        let slack =
            windows.earliest.iter().zip(windows.latest_start.iter()).map(|(&start, &end)| end.saturating_sub(start)).max().unwrap_or(0);
        (latest.saturating_sub(earliest).unsigned_abs().max(1), slack.unsigned_abs().max(1))
    });
    Some(RoutingSignature {
        depot,
        matrix,
        demands,
        capacity,
        has_time_windows,
        time_windows,
        has_fleet_objective,
        reverse_equivalent,
        distance_scale,
        demand_scale,
        time_scale,
        slack_scale,
    })
}

/// Whether the specialized sliced routing trajectory can consume this physical
/// list model. The orchestrator and the engine use this single capability test,
/// so a stage is never labelled routing while executing the generic list loop.
pub(crate) fn routing_search_supported(model: &CollectionModel) -> bool {
    routing_signature(model).is_some()
}

/// How a reduction can be scored incrementally from the old list plus a local
/// edit. Symmetric edge-cost detection is cached here, when the per-list index
/// is built, so candidate scoring does not inspect the expression tree.
fn reduction_delta_kind(r: &Reduction) -> ReductionDeltaKind {
    match (r.op, &r.iterable) {
        (ReduceOp::Count, Iterable::Items(_)) => ReductionDeltaKind::ItemsCount,
        (ReduceOp::Used, Iterable::Items(_)) => ReductionDeltaKind::Used,
        _ => ReductionDeltaKind::Unsupported,
    }
}

fn same_iterable_shape(a: &Iterable, b: &Iterable) -> bool {
    match (a, b) {
        (Iterable::Items(_), Iterable::Items(_))
        | (Iterable::SetItems(_), Iterable::SetItems(_))
        | (Iterable::Pairs(_), Iterable::Pairs(_)) => true,
        (Iterable::Edges { start: a_start, end: a_end, .. }, Iterable::Edges { start: b_start, end: b_end, .. }) => {
            a_start == b_start && a_end == b_end
        }
        (
            Iterable::Scan { init: a_init, boundary: a_boundary, step: a_step, end: a_end, .. },
            Iterable::Scan { init: b_init, boundary: b_boundary, step: b_step, end: b_end, .. },
        ) => a_init == b_init && a_boundary == b_boundary && a_step == b_step && a_end == b_end,
        (Iterable::Windows { size: a_size, inner: a_inner, .. }, Iterable::Windows { size: b_size, inner: b_inner, .. }) => {
            a_size == b_size && a_inner == b_inner
        }
        _ => false,
    }
}

fn same_reduction_shape(a: &Reduction, b: &Reduction) -> bool {
    let same_op = match (a.op, b.op) {
        (ReduceOp::Sum, ReduceOp::Sum)
        | (ReduceOp::Count, ReduceOp::Count)
        | (ReduceOp::Used, ReduceOp::Used)
        | (ReduceOp::Min, ReduceOp::Min)
        | (ReduceOp::Max, ReduceOp::Max) => true,
        (ReduceOp::SelectKth(a), ReduceOp::SelectKth(b)) => a == b,
        _ => false,
    };
    same_op && a.arena == b.arena && a.body == b.body && a.coeff == b.coeff && same_iterable_shape(&a.iterable, &b.iterable)
}

fn homogeneous_minimization_routing(model: &CollectionModel) -> bool {
    if model.lists == 0
        || model.objectives.is_empty()
        || model.objectives.iter().any(|tier| !tier.minimize || tier.max_terms.as_ref().is_some_and(|terms| !terms.is_empty()))
    {
        return false;
    }
    for tier in &model.objectives {
        let mut by_list = vec![Vec::new(); model.lists];
        for reduction in &tier.terms {
            let list = reduction.iterable.list();
            if list >= model.lists {
                return false;
            }
            by_list[list].push(reduction);
        }
        if (1..model.lists).any(|list| {
            by_list[list].len() != by_list[0].len()
                || by_list[list].iter().zip(&by_list[0]).any(|(left, right)| !same_reduction_shape(left, right))
        }) {
            return false;
        }
    }
    let mut constraints = vec![Vec::new(); model.lists];
    for constraint in &model.constraints {
        let list = constraint.reduction.iterable.list();
        if list >= model.lists {
            return false;
        }
        constraints[list].push(constraint);
    }
    (1..model.lists).all(|list| {
        constraints[list].len() == constraints[0].len()
            && constraints[list].iter().zip(&constraints[0]).all(|(left, right)| {
                left.op == right.op && left.rhs == right.rhs && same_reduction_shape(&left.reduction, &right.reduction)
            })
    })
}

fn lists_are_interchangeable(
    model: &CollectionModel,
    objective: &[Vec<Vec<Reduction>>],
    constraints: &[Vec<Constraint>],
    max_objective: &[Vec<MaxTerm>],
) -> bool {
    if model.lists < 2 || !model.globals.is_empty() || max_objective.iter().any(|terms| !terms.is_empty()) {
        return false;
    }
    (1..model.lists).all(|list| {
        objective[0]
            .iter()
            .zip(&objective[list])
            .all(|(left, right)| left.len() == right.len() && left.iter().zip(right).all(|(a, b)| same_reduction_shape(a, b)))
            && constraints[0].len() == constraints[list].len()
            && constraints[0]
                .iter()
                .zip(&constraints[list])
                .all(|(a, b)| a.op == b.op && a.rhs == b.rhs && same_reduction_shape(&a.reduction, &b.reduction))
    })
}

fn routing_gls(objective: &[Vec<Vec<Reduction>>], senses: &[bool], stop: &AtomicBool) -> Option<RoutingGls> {
    if objective.is_empty() || senses.first().copied() != Some(true) {
        return None;
    }
    let first = objective.first()?.first()?.first()?;
    let matrix = direct_edge_matrix(first)?;
    if !matches!(first.op, ReduceOp::Sum) || first.coeff <= 0 || matrix.is_empty() || matrix.iter().any(|row| row.len() != matrix.len()) {
        return None;
    }
    for list in objective {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let tier = list.first()?;
        if tier.len() != 1
            || !matches!(tier[0].iterable, Iterable::Edges { .. })
            || !matches!(tier[0].op, ReduceOp::Sum)
            || tier[0].coeff != first.coeff
        {
            return None;
        }
        let other = direct_edge_matrix(&tier[0])?;
        if !Arc::ptr_eq(&other, &matrix) {
            for (left, right) in other.iter().zip(matrix.iter()) {
                if stop.load(Ordering::Relaxed) || left != right {
                    return None;
                }
            }
        }
    }
    let mut symmetric = true;
    for row in 0..matrix.len() {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        for col in 0..row {
            if matrix[row][col] != matrix[col][row] {
                symmetric = false;
                break;
            }
        }
    }
    let mut penalties = Vec::with_capacity(matrix.len());
    for _ in 0..matrix.len() {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        penalties.push(vec![0; matrix.len()]);
    }
    Some(RoutingGls { penalties: RefCell::new(penalties), matrix, lambda: Cell::new(1), coeff: first.coeff, symmetric })
}

/// Active representatives of interchangeable routes. Every nonempty route is
/// retained, plus the first empty route. If a move fills it, a different empty
/// route becomes the representative on the next neighbourhood scan.
pub(super) fn active_list_indices(lists: &[Vec<i32>], interchangeable: bool) -> Vec<usize> {
    if !interchangeable {
        return (0..lists.len()).collect();
    }
    let first_empty = lists.iter().position(Vec::is_empty);
    (0..lists.len()).filter(|&list| !lists[list].is_empty() || Some(list) == first_empty).collect()
}

impl PerList {
    pub(super) fn build(model: &CollectionModel) -> Self {
        let stop = AtomicBool::new(false);
        Self::build_profiled(model, false, 24, true, false, &stop)
    }

    fn build_profiled(
        model: &CollectionModel,
        metrics_enabled: bool,
        candidate_limit: usize,
        enable_routing_gls: bool,
        diversify_descent: bool,
        stop: &AtomicBool,
    ) -> Self {
        let tiers = model.objectives.len();
        let mut objective = vec![vec![Vec::new(); tiers]; model.lists];
        let mut max_objective = vec![Vec::new(); tiers];
        let mut objective_delta = vec![vec![Vec::new(); tiers]; model.lists];
        let mut constraints = vec![Vec::new(); model.lists];
        let mut constraint_delta = vec![Vec::new(); model.lists];
        let mut senses = vec![true; tiers];
        let mut has_edges = false;
        let mut route_bounds = vec![None; model.lists];
        let mut candidate_matrix: Option<Arc<Vec<Vec<i64>>>> = None;
        for (t, tier) in model.objectives.iter().enumerate() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            senses[t] = tier.minimize;
            for r in &tier.terms {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let list = r.iterable.list();
                if let Iterable::Edges { start, end, .. } = &r.iterable {
                    has_edges = true;
                    route_bounds[list].get_or_insert((*start, *end));
                    if candidate_matrix.is_none() {
                        candidate_matrix = direct_edge_matrix(r);
                    }
                }
                objective_delta[list][t].push(reduction_delta_kind(r));
                objective[list][t].push(r.clone());
            }
            if let Some(max_terms) = &tier.max_terms {
                for term in max_terms {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    for r in term.groups.iter().flatten() {
                        let list = r.iterable.list();
                        if let Iterable::Edges { start, end, .. } = &r.iterable {
                            has_edges = true;
                            route_bounds[list].get_or_insert((*start, *end));
                            if candidate_matrix.is_none() {
                                candidate_matrix = direct_edge_matrix(r);
                            }
                        }
                    }
                }
                max_objective[t].extend(max_terms.iter().cloned());
            }
        }
        for c in &model.constraints {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let list = c.reduction.iterable.list();
            if let Iterable::Edges { start, end, .. } = &c.reduction.iterable {
                has_edges = true;
                route_bounds[list].get_or_insert((*start, *end));
                if candidate_matrix.is_none() {
                    candidate_matrix = direct_edge_matrix(&c.reduction);
                }
            }
            constraint_delta[list].push(reduction_delta_kind(&c.reduction));
            constraints[list].push(c.clone());
        }
        let penalties = GlsPenalties {
            constraints: RefCell::new(constraints.iter().map(|list| vec![1; list.len()]).collect()),
            objectives: RefCell::new(objective.iter().map(|list| list.iter().map(|tier| vec![1; tier.len()]).collect()).collect()),
        };
        let interchangeable_lists =
            (enable_routing_gls || diversify_descent) && lists_are_interchangeable(model, &objective, &constraints, &max_objective);
        let routing_gls = enable_routing_gls.then(|| routing_gls(&objective, &senses, stop)).flatten();
        let routing = has_edges.then(|| routing_signature(model)).flatten();
        let construction_limit = ((model.items.len() as f64).sqrt().ceil() as usize).clamp(8, 64);
        let candidates = candidate_matrix
            .and_then(|matrix| CandidateNeighbors::build(model, matrix, routing.as_ref(), candidate_limit.max(construction_limit), stop));
        Self {
            objective,
            max_objective,
            objective_delta,
            constraints,
            constraint_delta,
            senses,
            tiers: model.objectives.len(),
            globals: Globals::build(model, stop),
            has_edges,
            route_bounds,
            candidates,
            routing,
            infeas_cand: true,
            interchangeable_lists: Cell::new(interchangeable_lists),
            penalties,
            routing_gls,
            metrics: MetricsRecorder::new(metrics_enabled),
        }
    }

    pub(super) fn edge_penalty(&self, list: usize, contents: &(impl ListView + ?Sized)) -> i64 {
        let stop = AtomicBool::new(false);
        self.edge_penalty_interruptible(list, contents, &stop).expect("an uninterrupted edge-penalty evaluation must complete")
    }

    pub(super) fn edge_penalty_interruptible(
        &self,
        list: usize,
        contents: &(impl ListView + ?Sized),
        stop: &AtomicBool,
    ) -> Result<i64, EvaluationInterrupted> {
        let Some(gls) = &self.routing_gls else { return Ok(0) };
        let Some((start, end)) = self.route_bounds[list] else { return Ok(0) };
        let penalties = gls.penalties.borrow();
        let mut penalty = 0i64;
        for pos in 0..=contents.len() {
            if pos.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                return Err(EvaluationInterrupted);
            }
            let from = if pos == 0 { start } else { contents.at(pos - 1) };
            let to = if pos == contents.len() { end } else { contents.at(pos) };
            let (Ok(from), Ok(to)) = (usize::try_from(from), usize::try_from(to)) else { continue };
            penalty = penalty.saturating_add(penalties.get(from).and_then(|row| row.get(to)).copied().unwrap_or(0));
        }
        if stop.load(Ordering::Relaxed) {
            Err(EvaluationInterrupted)
        } else {
            Ok(penalty)
        }
    }

    fn bump_routing_gls(&self, state: &State, primary_objective: i64) -> Option<usize> {
        let gls = self.routing_gls.as_ref()?;
        let item_count = state.lists.iter().map(Vec::len).sum::<usize>().max(1);
        let average_edge = primary_objective.saturating_abs() / i64::try_from(item_count).unwrap_or(i64::MAX).max(1);
        gls.lambda.set((average_edge / 10).max(1));

        let penalties = gls.penalties.borrow();
        let mut best: Option<(i64, i64)> = None;
        let mut edges = Vec::new();
        for (list, contents) in state.lists.iter().enumerate() {
            let Some((start, end)) = self.route_bounds[list] else { continue };
            for pos in 0..=contents.len() {
                let from_value = if pos == 0 { start } else { contents[pos - 1] };
                let to_value = if pos == contents.len() { end } else { contents[pos] };
                let (Ok(from), Ok(to)) = (usize::try_from(from_value), usize::try_from(to_value)) else { continue };
                let Some(cost) = gls.matrix.get(from).and_then(|row| row.get(to)).copied() else { continue };
                let penalty = penalties[from][to];
                let utility = cost.saturating_mul(gls.coeff).max(0);
                if utility == 0 {
                    continue;
                }
                if best.is_none_or(|(best_value, best_denominator)| {
                    i128::from(utility) * i128::from(best_denominator) > i128::from(best_value) * i128::from(penalty.saturating_add(1))
                }) {
                    best = Some((utility, penalty.saturating_add(1)));
                }
                edges.push((from, to, utility, penalty.saturating_add(1)));
            }
        }
        drop(penalties);
        let (best_value, best_denominator) = best?;
        let mut penalties = gls.penalties.borrow_mut();
        let mut bumped = 0usize;
        let mut seen = HashSet::new();
        for (from, to, utility, denominator) in edges {
            if i128::from(utility) * i128::from(best_denominator) == i128::from(best_value) * i128::from(denominator)
                && seen.insert(if gls.symmetric && from > to { (to, from) } else { (from, to) })
            {
                penalties[from][to] = penalties[from][to].saturating_add(1);
                if gls.symmetric && from != to {
                    penalties[to][from] = penalties[to][from].saturating_add(1);
                }
                bumped += 1;
            }
        }
        Some(bumped)
    }

    fn reset_routing_gls(&self) -> bool {
        let Some(gls) = &self.routing_gls else { return false };
        for row in gls.penalties.borrow_mut().iter_mut() {
            row.fill(0);
        }
        gls.lambda.set(1);
        true
    }

    pub(super) fn has_max_objective(&self) -> bool {
        self.max_objective.iter().any(|terms| !terms.is_empty())
    }

    /// Increase the penalties of the reductions with maximum GLS utility.
    /// Constraint reductions are used while infeasible. Once feasible, the
    /// first lexicographic objective tier becomes the feature set.
    pub(super) fn bump_gls(&self, state: &State) -> usize {
        let raw = full_score_raw(self, state);
        if raw.violation > 0 {
            self.interchangeable_lists.set(false);
            let mut weights = self.penalties.constraints.borrow_mut();
            let mut best: Option<(i64, i64)> = None;
            for (list, score) in state.scores.iter().enumerate() {
                for (idx, &value) in score.constraint_violations.iter().enumerate() {
                    if value <= 0 {
                        continue;
                    }
                    let weight = weights[list][idx];
                    if best.is_none_or(|(best_value, best_weight)| {
                        i128::from(value) * i128::from(best_weight) > i128::from(best_value) * i128::from(weight)
                    }) {
                        best = Some((value, weight));
                    }
                }
            }
            let Some((best_value, best_weight)) = best else { return 0 };
            let mut bumped = 0usize;
            for (list, score) in state.scores.iter().enumerate() {
                for (idx, &value) in score.constraint_violations.iter().enumerate() {
                    let weight = weights[list][idx];
                    if value > 0 && i128::from(value) * i128::from(best_weight) == i128::from(best_value) * i128::from(weight) {
                        weights[list][idx] = weight.saturating_add(1);
                        bumped += 1;
                    }
                }
            }
            return bumped;
        }

        if self.tiers == 0 {
            return 0;
        }
        if let Some(bumped) = self.bump_routing_gls(state, raw.tiers[0]) {
            return bumped;
        }
        self.interchangeable_lists.set(false);
        let minimize = self.senses[0];
        let mut weights = self.penalties.objectives.borrow_mut();
        let mut best: Option<(i64, i64)> = None;
        for (list, score) in state.scores.iter().enumerate() {
            for (idx, value) in score.objective_reductions[0].iter().enumerate() {
                let Some(value) = value else { continue };
                let utility = if minimize { *value } else { value.saturating_neg() }.max(0);
                if utility == 0 {
                    continue;
                }
                let weight = weights[list][0][idx];
                if best.is_none_or(|(best_value, best_weight)| {
                    i128::from(utility) * i128::from(best_weight) > i128::from(best_value) * i128::from(weight)
                }) {
                    best = Some((utility, weight));
                }
            }
        }
        let Some((best_value, best_weight)) = best else { return 0 };
        let mut bumped = 0usize;
        for (list, score) in state.scores.iter().enumerate() {
            for (idx, value) in score.objective_reductions[0].iter().enumerate() {
                let Some(value) = value else { continue };
                let utility = if minimize { *value } else { value.saturating_neg() }.max(0);
                let weight = weights[list][0][idx];
                if utility > 0 && i128::from(utility) * i128::from(best_weight) == i128::from(best_value) * i128::from(weight) {
                    weights[list][0][idx] = weight.saturating_add(1);
                    bumped += 1;
                }
            }
        }
        bumped
    }
}

/// The comparable score of a state: violation first, then the objective tiers
/// (each already signed so smaller is better), compared lexicographically.
pub(super) type TierValues = SmallVec<[i64; 4]>;
pub(super) type ConstraintViolations = SmallVec<[i64; 4]>;
pub(super) type ReductionValues = SmallVec<[Option<i64>; 4]>;
pub(super) type ObjectiveReductionValues = SmallVec<[ReductionValues; 4]>;

pub(super) fn tier_values(len: usize, value: i64) -> TierValues {
    std::iter::repeat_n(value, len).collect()
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Score {
    pub(super) violation: i64,
    pub(super) tiers: TierValues,
}

/// Cached per-list contribution: its violation and its raw (unsigned) value in
/// each objective tier.
#[derive(Clone)]
pub(super) struct ListScore {
    pub(super) violation: i64,
    pub(super) objectives: TierValues,
    pub(super) constraint_violations: ConstraintViolations,
    pub(super) objective_reductions: ObjectiveReductionValues,
    pub(super) undefined_violation: i64,
    pub(super) edge_penalty: i64,
}

pub(super) struct ListReductionCaches {
    pub(super) objective: Vec<Vec<ReductionCache>>,
    pub(super) constraints: Vec<ReductionCache>,
}

pub(super) struct TrialList<'a> {
    pub(super) list: usize,
    pub(super) score: &'a ListScore,
    pub(super) contents: &'a dyn ListView,
}

fn reduction_cache(per: &PerList, reduction: &Reduction, contents: &[i32]) -> ReductionCache {
    per.metrics.measure_full(reduction, contents.len(), || ReductionCache::build(reduction, contents))
}

fn reduction_cache_interruptible(per: &PerList, reduction: &Reduction, contents: &[i32], stop: &AtomicBool) -> Option<ReductionCache> {
    per.metrics.measure_full(reduction, contents.len(), || ReductionCache::build_interruptible(reduction, contents, stop))
}

impl ListReductionCaches {
    pub(super) fn build(per: &PerList, idx: usize, contents: &[i32]) -> Self {
        let objective = per.objective[idx]
            .iter()
            .map(|tier| tier.iter().map(|reduction| reduction_cache(per, reduction, contents)).collect())
            .collect();
        let constraints = per.constraints[idx].iter().map(|constraint| reduction_cache(per, &constraint.reduction, contents)).collect();
        Self { objective, constraints }
    }

    pub(super) fn build_interruptible(per: &PerList, idx: usize, contents: &[i32], stop: &AtomicBool) -> Option<Self> {
        let objective = per.objective[idx]
            .iter()
            .map(|tier| {
                tier.iter().map(|reduction| reduction_cache_interruptible(per, reduction, contents, stop)).collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?;
        let constraints = per.constraints[idx]
            .iter()
            .map(|constraint| reduction_cache_interruptible(per, &constraint.reduction, contents, stop))
            .collect::<Option<Vec<_>>>()?;
        Some(Self { objective, constraints })
    }

    pub(super) fn score(&self, per: &PerList, idx: usize, contents: &dyn ListView) -> ListScore {
        let mut violation = 0i64;
        let mut undefined_violation = 0i64;
        let mut objectives = tier_values(per.tiers, 0);
        let mut objective_reductions = ObjectiveReductionValues::with_capacity(per.tiers);
        for (tier, slot) in objectives.iter_mut().enumerate() {
            let mut values = ReductionValues::with_capacity(self.objective[tier].len());
            for cache in &self.objective[tier] {
                match cache.value() {
                    Some(value) => {
                        *slot = slot.saturating_add(value);
                        values.push(Some(value));
                    }
                    None => {
                        violation = violation.saturating_add(INFEASIBLE);
                        undefined_violation = undefined_violation.saturating_add(INFEASIBLE);
                        values.push(None);
                    }
                }
            }
            objective_reductions.push(values);
        }
        let mut constraint_violations = ConstraintViolations::with_capacity(self.constraints.len());
        for (constraint, cache) in per.constraints[idx].iter().zip(&self.constraints) {
            match cache.value() {
                Some(value) => {
                    let constraint_violation = violation_of(value, constraint.op, constraint.rhs);
                    violation = violation.saturating_add(constraint_violation);
                    constraint_violations.push(constraint_violation);
                }
                None => {
                    violation = violation.saturating_add(INFEASIBLE);
                    undefined_violation = undefined_violation.saturating_add(INFEASIBLE);
                    constraint_violations.push(0);
                }
            }
        }
        let edge_penalty = per.edge_penalty(idx, contents);
        ListScore { violation, objectives, constraint_violations, objective_reductions, undefined_violation, edge_penalty }
    }
}

/// Independent full evaluator retained as the incremental-cache oracle.
pub(super) fn list_score_exact(per: &PerList, idx: usize, contents: &[i32]) -> ListScore {
    let mut violation = 0i64;
    let mut undefined_violation = 0i64;
    let mut objectives = tier_values(per.tiers, 0);
    let mut objective_reductions = ObjectiveReductionValues::with_capacity(per.tiers);
    for (tier, slot) in objectives.iter_mut().enumerate() {
        let mut values = ReductionValues::with_capacity(per.objective[idx][tier].len());
        for reduction in &per.objective[idx][tier] {
            match eval_reduction(reduction, contents) {
                Some(value) => {
                    *slot = slot.saturating_add(value);
                    values.push(Some(value));
                }
                None => {
                    violation = violation.saturating_add(INFEASIBLE);
                    undefined_violation = undefined_violation.saturating_add(INFEASIBLE);
                    values.push(None);
                }
            }
        }
        objective_reductions.push(values);
    }
    let mut constraint_violations = ConstraintViolations::with_capacity(per.constraints[idx].len());
    for constraint in &per.constraints[idx] {
        match eval_reduction(&constraint.reduction, contents) {
            Some(value) => {
                let constraint_violation = violation_of(value, constraint.op, constraint.rhs);
                violation = violation.saturating_add(constraint_violation);
                constraint_violations.push(constraint_violation);
            }
            None => {
                violation = violation.saturating_add(INFEASIBLE);
                undefined_violation = undefined_violation.saturating_add(INFEASIBLE);
                constraint_violations.push(0);
            }
        }
    }
    let edge_penalty = per.edge_penalty(idx, contents);
    ListScore { violation, objectives, constraint_violations, objective_reductions, undefined_violation, edge_penalty }
}

fn build_max_caches(per: &PerList, lists: &[Vec<i32>]) -> Vec<Vec<Vec<Vec<ReductionCache>>>> {
    per.max_objective
        .iter()
        .map(|terms| {
            terms
                .iter()
                .map(|term| {
                    term.groups
                        .iter()
                        .map(|group| {
                            group.iter().map(|reduction| reduction_cache(per, reduction, &lists[reduction.iterable.list()])).collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn build_max_caches_interruptible(per: &PerList, lists: &[Vec<i32>], stop: &AtomicBool) -> Option<Vec<Vec<Vec<Vec<ReductionCache>>>>> {
    per.max_objective
        .iter()
        .map(|terms| {
            terms
                .iter()
                .map(|term| {
                    term.groups
                        .iter()
                        .map(|group| {
                            group
                                .iter()
                                .map(|reduction| reduction_cache_interruptible(per, reduction, &lists[reduction.iterable.list()], stop))
                                .collect::<Option<Vec<_>>>()
                        })
                        .collect::<Option<Vec<_>>>()
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()
}

/// Apply each tier's optimisation direction to raw tier sums (smaller better).
pub(super) fn signed(per: &PerList, violation: i64, raw: TierValues) -> Score {
    let mut tiers = tier_values(per.tiers, 0);
    for ((slot, &r), &minimize) in tiers.iter_mut().zip(raw.iter()).zip(per.senses.iter()) {
        *slot = if minimize { r } else { r.saturating_neg() };
    }
    Score { violation, tiers }
}

/// Raw (unsigned) totals across all lists: violation and per-tier sums.
pub(super) fn base_totals(scores: &[ListScore], tiers: usize) -> (i64, TierValues) {
    let mut violation = 0i64;
    let mut raw = tier_values(tiers, 0);
    for s in scores {
        violation = violation.saturating_add(s.violation);
        for (r, &o) in raw.iter_mut().zip(s.objectives.iter()) {
            *r = r.saturating_add(o);
        }
    }
    (violation, raw)
}

fn max_objective_totals_interruptible<'a>(
    per: &PerList,
    state: &State,
    replacements: &'a [TrialList<'a>],
    scratch: &mut EvalScratch,
    stop: &AtomicBool,
) -> Result<(i64, TierValues), EvaluationInterrupted> {
    let mut violation = 0i64;
    let mut raw = tier_values(per.tiers, 0);
    let mut work = 0usize;
    for (tier, terms) in per.max_objective.iter().enumerate() {
        for (term_idx, term) in terms.iter().enumerate() {
            let mut best = None;
            for (group_idx, group) in term.groups.iter().enumerate() {
                let mut group_value = 0i64;
                let mut defined = true;
                for (reduction_idx, reduction) in group.iter().enumerate() {
                    if work.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                        return Err(EvaluationInterrupted);
                    }
                    let cache = &state.max_caches[tier][term_idx][group_idx][reduction_idx];
                    let list = reduction.iterable.list();
                    let value = if let Some(replacement) = replacements.iter().find(|replacement| replacement.list == list) {
                        let value = per.metrics.measure_delta(reduction, || {
                            cache.candidate_value_interruptible(reduction, &state.lists[list], replacement.contents, scratch, stop)
                        })?;
                        if matches!(reduction.iterable, Iterable::Scan { .. }) {
                            per.metrics.record_incremental_scan(scratch.recomputed_scan_steps());
                        }
                        value
                    } else {
                        cache.value()
                    };
                    match value {
                        Some(value) => group_value = group_value.saturating_add(value),
                        None => {
                            defined = false;
                            violation = violation.saturating_add(INFEASIBLE);
                        }
                    }
                    work = work.saturating_add(1);
                }
                if defined {
                    best = Some(best.map_or(group_value, |value: i64| value.max(group_value)));
                }
            }
            if let Some(best) = best {
                raw[tier] = raw[tier].saturating_add(best.saturating_mul(term.coeff));
            }
        }
    }
    if stop.load(Ordering::Relaxed) {
        Err(EvaluationInterrupted)
    } else {
        Ok((violation, raw))
    }
}

fn score_with_replacements_mode<'a>(
    per: &PerList,
    state: &'a State,
    replacements: &'a [TrialList<'a>],
    global_delta: i64,
    scratch: &mut EvalScratch,
    guided: bool,
) -> Score {
    let stop = AtomicBool::new(false);
    score_with_replacements_mode_interruptible(per, state, replacements, global_delta, scratch, guided, &stop)
        .expect("an uninterrupted replacement score must complete")
}

fn score_with_replacements_mode_interruptible<'a>(
    per: &PerList,
    state: &'a State,
    replacements: &'a [TrialList<'a>],
    global_delta: i64,
    scratch: &mut EvalScratch,
    guided: bool,
    stop: &AtomicBool,
) -> Result<Score, EvaluationInterrupted> {
    // Fold in list order, just like a full evaluation. Subtracting an old
    // contribution from a saturated total and adding the replacement is not
    // reversible at i64::MIN/MAX.
    let mut violation = 0i64;
    let mut raw = tier_values(per.tiers, 0);
    let constraint_weights = guided.then(|| per.penalties.constraints.borrow());
    let objective_weights = guided.then(|| per.penalties.objectives.borrow());
    let mut work = 0usize;
    for (list, cached) in state.scores.iter().enumerate() {
        if work.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
            return Err(EvaluationInterrupted);
        }
        let score = replacements.iter().find(|replacement| replacement.list == list).map_or(cached, |replacement| replacement.score);
        if let Some(weights) = &constraint_weights {
            let mut weighted = score.undefined_violation;
            for (&value, &weight) in score.constraint_violations.iter().zip(&weights[list]) {
                if work.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                    return Err(EvaluationInterrupted);
                }
                weighted = weighted.saturating_add(value.saturating_mul(weight));
                work = work.saturating_add(1);
            }
            violation = violation.saturating_add(weighted);
        } else {
            violation = violation.saturating_add(score.violation);
        }
        for (tier, (slot, &value)) in raw.iter_mut().zip(score.objectives.iter()).enumerate() {
            *slot = slot.saturating_add(value);
            if let Some(weights) = &objective_weights {
                for (&reduction_value, &weight) in score.objective_reductions[tier].iter().zip(&weights[list][tier]) {
                    if work.is_multiple_of(64) && stop.load(Ordering::Relaxed) {
                        return Err(EvaluationInterrupted);
                    }
                    if let Some(reduction_value) = reduction_value {
                        *slot = slot.saturating_add(reduction_value.saturating_mul(weight.saturating_sub(1)));
                    }
                    work = work.saturating_add(1);
                }
            }
            if tier == 0 && guided {
                if let Some(gls) = &per.routing_gls {
                    *slot = slot.saturating_add(score.edge_penalty.saturating_mul(gls.lambda.get()));
                }
            }
            work = work.saturating_add(1);
        }
        work = work.saturating_add(1);
    }
    violation = violation.saturating_add(state.global_viol).saturating_add(global_delta);
    if per.has_max_objective() {
        let (max_violation, max_raw) = max_objective_totals_interruptible(per, state, replacements, scratch, stop)?;
        violation = violation.saturating_add(max_violation);
        for (slot, value) in raw.iter_mut().zip(max_raw) {
            *slot = slot.saturating_add(value);
        }
    }
    if stop.load(Ordering::Relaxed) {
        Err(EvaluationInterrupted)
    } else {
        Ok(signed(per, violation, raw))
    }
}

pub(super) fn score_with_replacements<'a>(
    per: &PerList,
    state: &'a State,
    replacements: &'a [TrialList<'a>],
    global_delta: i64,
    scratch: &mut EvalScratch,
) -> Score {
    score_with_replacements_mode(per, state, replacements, global_delta, scratch, true)
}

pub(super) fn score_with_replacements_interruptible<'a>(
    per: &PerList,
    state: &'a State,
    replacements: &'a [TrialList<'a>],
    global_delta: i64,
    scratch: &mut EvalScratch,
    stop: &AtomicBool,
) -> Result<Score, EvaluationInterrupted> {
    score_with_replacements_mode_interruptible(per, state, replacements, global_delta, scratch, true, stop)
}

/// Exact unpenalized score after replacing an arbitrary set of lists. This is
/// the speculative counterpart of `full_score_raw`: untouched lists and their
/// reduction caches are reused, while only the supplied views are rescored.
pub(super) fn score_with_replacements_raw<'a>(
    per: &PerList,
    state: &'a State,
    replacements: &'a [TrialList<'a>],
    global_delta: i64,
    scratch: &mut EvalScratch,
) -> Score {
    score_with_replacements_mode(per, state, replacements, global_delta, scratch, false)
}

pub(super) fn score_with_replacements_raw_interruptible<'a>(
    per: &PerList,
    state: &'a State,
    replacements: &'a [TrialList<'a>],
    global_delta: i64,
    scratch: &mut EvalScratch,
    stop: &AtomicBool,
) -> Result<Score, EvaluationInterrupted> {
    score_with_replacements_mode_interruptible(per, state, replacements, global_delta, scratch, false, stop)
}

/// Full score including the cross-list global violation.
pub(super) fn full_score(per: &PerList, state: &State) -> Score {
    score_with_replacements(per, state, &[], 0, &mut EvalScratch::default())
}

pub(super) fn full_score_interruptible(per: &PerList, state: &State, stop: &AtomicBool) -> Result<Score, EvaluationInterrupted> {
    score_with_replacements_interruptible(per, state, &[], 0, &mut EvalScratch::default(), stop)
}

/// Exact, unpenalized score used for incumbent tracking and public objectives.
pub(super) fn full_score_raw(per: &PerList, state: &State) -> Score {
    score_with_replacements_mode(per, state, &[], 0, &mut EvalScratch::default(), false)
}

pub(super) fn full_score_exact_lists(per: &PerList, lists: &[Vec<i32>], global_violation: i64) -> Score {
    let scores: Vec<ListScore> = lists.iter().enumerate().map(|(idx, contents)| list_score_exact(per, idx, contents)).collect();
    let (mut violation, mut raw) = base_totals(&scores, per.tiers);
    violation = violation.saturating_add(global_violation);
    for (tier, terms) in per.max_objective.iter().enumerate() {
        for term in terms {
            let mut best = None;
            for group in &term.groups {
                let mut value = 0i64;
                let mut defined = true;
                for reduction in group {
                    match eval_reduction(reduction, &lists[reduction.iterable.list()]) {
                        Some(reduction_value) => value = value.saturating_add(reduction_value),
                        None => {
                            defined = false;
                            violation = violation.saturating_add(INFEASIBLE);
                        }
                    }
                }
                if defined {
                    best = Some(best.map_or(value, |old: i64| old.max(value)));
                }
            }
            if let Some(best) = best {
                raw[tier] = raw[tier].saturating_add(best.saturating_mul(term.coeff));
            }
        }
    }
    signed(per, violation, raw)
}

/// State of the search: list contents, cached per-list scores, and (for global
/// constraints) each item's current list plus the total global violation.
pub(super) struct State {
    pub(super) lists: Vec<Vec<i32>>,
    pub(super) scores: Vec<ListScore>,
    pub(super) caches: Vec<ListReductionCaches>,
    max_caches: Vec<Vec<Vec<Vec<ReductionCache>>>>,
    pub(super) item_list: Vec<usize>,
    pub(super) global_viol: i64,
}

impl State {
    /// Greedy construction: insert each item (in `order`) into the list whose
    /// resulting per-list score increases least, lexicographically (violation
    /// first, then objective). This yields a feasibility-leaning start, so items
    /// avoid lists they would push over a capacity bound, and empty lists with a
    /// `min`/`max` constraint get filled, far better than a blind round robin
    /// on tight instances, while different `order`s give diverse seeded starts.
    /// Global constraints are ignored here; the search repairs them.
    fn greedy(
        model: &CollectionModel,
        per: &PerList,
        order: &[usize],
        seed: u64,
        randomized_ties: bool,
        stop: &AtomicBool,
    ) -> Option<Self> {
        let k = model.lists.max(1);
        let mut state = Self::from_lists_interruptible(model, per, vec![Vec::new(); k], stop)?;
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let mut scratch = EvalScratch::default();
        for &item_idx in order {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let item = model.items[item_idx];
            let mut best_l = 0;
            let mut best_key = Score { violation: i64::MAX, tiers: tier_values(per.tiers, i64::MAX) };
            let mut best_tie = u64::MAX;
            for l in active_list_indices(&state.lists, per.interchangeable_lists.get()) {
                let candidate = InsertView::new(&state.lists[l], state.lists[l].len(), item);
                per.metrics.record_candidate();
                let sc = trial_list_score_view(per, &state, l, &candidate, None, &mut scratch);
                // Lexicographic increment of placing the item in list l.
                let dv = sc.violation.saturating_sub(state.scores[l].violation);
                let mut draw = tier_values(per.tiers, 0);
                for ((d, &new), &old) in draw.iter_mut().zip(sc.objectives.iter()).zip(state.scores[l].objectives.iter()) {
                    *d = new.saturating_sub(old);
                }
                let key = signed(per, dv, draw);
                let tie = mix64(seed ^ mix64(item_idx as u64) ^ mix64(l as u64));
                if key < best_key || (randomized_ties && key == best_key && tie < best_tie) {
                    best_key = key;
                    best_tie = tie;
                    best_l = l;
                }
            }
            state.lists[best_l].push(item);
            if !state.rescore_interruptible(per, best_l, stop) {
                return None;
            }
            state.item_list[item_idx] = best_l;
        }
        state.global_viol = per.globals.total(&state.item_list);
        Some(state)
    }

    pub(super) fn from_lists(model: &CollectionModel, per: &PerList, mut lists: Vec<Vec<i32>>) -> Self {
        let k = model.lists.max(1);
        lists.truncate(k);
        lists.resize_with(k, Vec::new);
        let caches: Vec<ListReductionCaches> = (0..k).map(|idx| ListReductionCaches::build(per, idx, &lists[idx])).collect();
        let scores: Vec<ListScore> = (0..k).map(|idx| caches[idx].score(per, idx, &lists[idx])).collect();
        let max_caches = build_max_caches(per, &lists);
        let mut item_list = vec![0usize; model.items.len()];
        for (l, contents) in lists.iter().enumerate() {
            for &value in contents {
                if let Some(&idx) = per.globals.value_to_idx.get(&value) {
                    item_list[idx] = l;
                }
            }
        }
        let global_viol = per.globals.total(&item_list);
        Self { lists, scores, caches, max_caches, item_list, global_viol }
    }

    pub(super) fn from_lists_interruptible(
        model: &CollectionModel,
        per: &PerList,
        mut lists: Vec<Vec<i32>>,
        stop: &AtomicBool,
    ) -> Option<Self> {
        let k = model.lists.max(1);
        lists.truncate(k);
        lists.resize_with(k, Vec::new);
        let caches = (0..k).map(|idx| ListReductionCaches::build_interruptible(per, idx, &lists[idx], stop)).collect::<Option<Vec<_>>>()?;
        let scores = (0..k).map(|idx| caches[idx].score(per, idx, &lists[idx])).collect();
        let max_caches = build_max_caches_interruptible(per, &lists, stop)?;
        let mut item_list = vec![0usize; model.items.len()];
        for (list_index, contents) in lists.iter().enumerate() {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            for (item_index, &value) in contents.iter().enumerate() {
                if item_index.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                    return None;
                }
                if let Some(&idx) = per.globals.value_to_idx.get(&value) {
                    item_list[idx] = list_index;
                }
            }
        }
        let global_viol = per.globals.total(&item_list);
        (!stop.load(Ordering::Relaxed)).then_some(Self { lists, scores, caches, max_caches, item_list, global_viol })
    }

    pub(super) fn rescore_interruptible(&mut self, per: &PerList, idx: usize, stop: &AtomicBool) -> bool {
        let Some(cache) = ListReductionCaches::build_interruptible(per, idx, &self.lists[idx], stop) else {
            return false;
        };
        let score = cache.score(per, idx, &self.lists[idx]);
        let mut max_updates = Vec::new();
        for (tier, terms) in per.max_objective.iter().enumerate() {
            for (term_idx, term) in terms.iter().enumerate() {
                for (group_idx, group) in term.groups.iter().enumerate() {
                    for (reduction_idx, reduction) in group.iter().enumerate() {
                        if reduction.iterable.list() == idx {
                            let Some(reduction_cache) = reduction_cache_interruptible(per, reduction, &self.lists[idx], stop) else {
                                return false;
                            };
                            max_updates.push((tier, term_idx, group_idx, reduction_idx, reduction_cache));
                        }
                    }
                }
            }
        }
        self.caches[idx] = cache;
        self.scores[idx] = score;
        for (tier, term_idx, group_idx, reduction_idx, cache) in max_updates {
            self.max_caches[tier][term_idx][group_idx][reduction_idx] = cache;
        }
        true
    }

    /// Record that `value` now lives in list `l`, for global-constraint tracking.
    pub(super) fn set_item_list(&mut self, per: &PerList, value: i32, l: usize) {
        if let Some(&i) = per.globals.value_to_idx.get(&value) {
            self.item_list[i] = l;
        }
    }

    fn refresh_edge_penalties(&mut self, per: &PerList) {
        for (list, score) in self.scores.iter_mut().enumerate() {
            score.edge_penalty = per.edge_penalty(list, &self.lists[list]);
        }
    }
}

struct InitialConstruction {
    state: State,
    incumbent: Option<State>,
    name: &'static str,
    elapsed: std::time::Duration,
    feasible_history: Vec<InitialFeasible>,
    candidates: u64,
    reported: bool,
}

#[derive(Clone)]
struct InitialFeasible {
    elapsed: std::time::Duration,
    score: Score,
    fleet: usize,
    candidates: u64,
}

struct RoutingConstructionWork<'a> {
    candidates: &'a mut u64,
    candidate_budget: u64,
    feasible_history: &'a mut Vec<InitialFeasible>,
    started: &'a Instant,
    per: &'a PerList,
    report: &'a mut dyn FnMut(i64),
    fallback_lists: &'a mut Option<(Vec<Vec<i32>>, &'static str)>,
}

impl RoutingConstructionWork<'_> {
    fn exhausted(&self) -> bool {
        *self.candidates >= self.candidate_budget
    }

    fn consume_candidate(&mut self) {
        *self.candidates = self.candidates.saturating_add(1);
    }

    fn observe(&mut self, state: &State, score: Score) -> bool {
        let improved =
            observe_construction_incumbent(self.feasible_history, state, score.clone(), self.started.elapsed(), *self.candidates);
        if improved {
            *self.fallback_lists = Some((state.lists.clone(), "parallel-savings"));
        }
        if improved && self.per.tiers > 0 {
            (self.report)(tier_value(self.per, &score, 0));
        }
        improved
    }
}

fn initial_feasible(state: &State, score: Score, elapsed: std::time::Duration, candidates: u64) -> InitialFeasible {
    InitialFeasible { elapsed, score, fleet: state.lists.iter().filter(|route| !route.is_empty()).count(), candidates }
}

fn observe_construction_incumbent(
    history: &mut Vec<InitialFeasible>,
    state: &State,
    score: Score,
    elapsed: Duration,
    candidates: u64,
) -> bool {
    if score.violation != 0 || history.last().is_some_and(|incumbent| score >= incumbent.score) {
        return false;
    }
    history.push(initial_feasible(state, score, elapsed, candidates));
    true
}

#[derive(Clone)]
struct InsertionPlacement {
    list: usize,
    position: usize,
    score: Score,
}

/// Lexicographic opportunity loss between a best placement and an alternative.
/// Earlier score components dominate later ones without flattening arbitrary
/// i64 objective values into a fixed-width scalar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ScoreRegret {
    forced: bool,
    component: usize,
    delta: u128,
}

impl ScoreRegret {
    const fn forced() -> Self {
        Self { forced: true, component: 0, delta: u128::MAX }
    }
}

impl Ord for ScoreRegret {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.forced
            .cmp(&other.forced)
            // A difference in violation (component zero), or an earlier
            // objective tier, is lexicographically more important.
            .then_with(|| other.component.cmp(&self.component))
            .then_with(|| self.delta.cmp(&other.delta))
    }
}

impl PartialOrd for ScoreRegret {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub(super) fn score_regret(alternative: &Score, best: &Score) -> ScoreRegret {
    let components =
        std::iter::once((alternative.violation, best.violation)).chain(alternative.tiers.iter().copied().zip(best.tiers.iter().copied()));
    for (component, (worse, better)) in components.enumerate() {
        if worse != better {
            return ScoreRegret { forced: false, component, delta: (i128::from(worse) - i128::from(better)).unsigned_abs() };
        }
    }
    ScoreRegret { forced: false, component: usize::MAX, delta: 0 }
}

#[cfg(test)]
pub(crate) fn audit_lexicographic_regret() -> bool {
    let score = |violation, tiers: &[i64]| Score { violation, tiers: tiers.iter().copied().collect() };
    let feasible = score(0, &[0, 0]);
    let infeasible = score(1, &[i64::MIN, i64::MIN]);
    let primary = score(0, &[1, i64::MIN]);
    let secondary = score(0, &[0, i64::MAX]);
    ScoreRegret::forced() > score_regret(&infeasible, &feasible)
        && score_regret(&infeasible, &feasible) > score_regret(&primary, &feasible)
        && score_regret(&primary, &feasible) > score_regret(&secondary, &feasible)
}

fn construction_positions(per: &PerList, route: &[i32], item: i32) -> Vec<usize> {
    let all: Vec<usize> = (0..=route.len()).collect();
    let Some(neighbors) = &per.candidates else {
        return all;
    };
    let granular: Vec<usize> = all
        .iter()
        .copied()
        .filter(|&position| {
            position.checked_sub(1).is_some_and(|before| neighbors.contains_semantic(item, route[before]))
                || route.get(position).is_some_and(|&after| neighbors.contains_semantic(item, after))
                || route.is_empty()
        })
        .collect();
    if granular.is_empty() {
        all
    } else {
        granular
    }
}

fn insertion_placements(
    per: &PerList,
    state: &State,
    item: i32,
    scratch: &mut EvalScratch,
    candidates: &mut u64,
    candidate_budget: u64,
) -> Vec<InsertionPlacement> {
    let mut placements = Vec::new();
    for list in active_list_indices(&state.lists, per.interchangeable_lists.get()) {
        for position in construction_positions(per, &state.lists[list], item) {
            if *candidates >= candidate_budget {
                return placements;
            }
            *candidates = candidates.saturating_add(1);
            per.metrics.record_candidate();
            let candidate = InsertView::new(&state.lists[list], position, item);
            let list_score = trial_list_score_view(per, state, list, &candidate, None, scratch);
            if list_score.violation != 0 {
                continue;
            }
            let global_delta = per.globals.delta(&state.item_list, &[(item, list)]);
            let score = score_with_replaced_list(per, state, list, &list_score, &candidate, global_delta, scratch);
            placements.push(InsertionPlacement { list, position, score });
        }
    }
    placements.sort_by(|left, right| {
        left.score.cmp(&right.score).then_with(|| left.list.cmp(&right.list)).then_with(|| left.position.cmp(&right.position))
    });
    placements
}

fn apply_insertion(state: &mut State, per: &PerList, item: i32, placement: &InsertionPlacement, stop: &AtomicBool) -> bool {
    state.lists[placement.list].insert(placement.position, item);
    if !state.rescore_interruptible(per, placement.list, stop) {
        return false;
    }
    state.set_item_list(per, item, placement.list);
    state.global_viol = per.globals.total(&state.item_list);
    true
}

fn cheapest_insertion_state(
    model: &CollectionModel,
    per: &PerList,
    order: &[usize],
    stop: &AtomicBool,
    candidates: &mut u64,
    candidate_budget: u64,
) -> Option<State> {
    let mut state = State::from_lists_interruptible(model, per, vec![Vec::new(); model.lists.max(1)], stop)?;
    let mut scratch = EvalScratch::default();
    for &index in order {
        if stop.load(Ordering::Relaxed) || *candidates >= candidate_budget {
            return None;
        }
        let item = model.items[index];
        let placement = insertion_placements(per, &state, item, &mut scratch, candidates, candidate_budget).into_iter().next()?;
        if !apply_insertion(&mut state, per, item, &placement, stop) {
            return None;
        }
    }
    Some(state)
}

fn regret_insertion_state(
    model: &CollectionModel,
    per: &PerList,
    order: &[usize],
    seed: u64,
    stop: &AtomicBool,
    candidates: &mut u64,
    candidate_budget: u64,
) -> Option<State> {
    let mut state = State::from_lists_interruptible(model, per, vec![Vec::new(); model.lists.max(1)], stop)?;
    let mut remaining: Vec<i32> = order.iter().map(|&index| model.items[index]).collect();
    let mut scratch = EvalScratch::default();
    while !remaining.is_empty() {
        if stop.load(Ordering::Relaxed) || *candidates >= candidate_budget {
            return None;
        }
        let mut selected: Option<(ScoreRegret, u64, usize, InsertionPlacement)> = None;
        for (remaining_index, &item) in remaining.iter().enumerate() {
            let placements = insertion_placements(per, &state, item, &mut scratch, candidates, candidate_budget);
            let Some(best) = placements.first().cloned() else {
                continue;
            };
            let regret = placements.get(1).map_or_else(ScoreRegret::forced, |second| score_regret(&second.score, &best.score));
            let tie = mix64(seed ^ mix64(item as u64));
            if selected
                .as_ref()
                .is_none_or(|&(old_regret, old_tie, _, _)| (regret, std::cmp::Reverse(tie)) > (old_regret, std::cmp::Reverse(old_tie)))
            {
                selected = Some((regret, tie, remaining_index, best));
            }
        }
        let (_, _, remaining_index, placement) = selected?;
        let item = remaining.swap_remove(remaining_index);
        if !apply_insertion(&mut state, per, item, &placement, stop) {
            return None;
        }
    }
    Some(state)
}

fn singleton_state(model: &CollectionModel, per: &PerList, stop: &AtomicBool) -> Option<State> {
    if model.lists < model.items.len() {
        return None;
    }
    let mut lists = vec![Vec::new(); model.lists.max(1)];
    for (list, &item) in model.items.iter().enumerate() {
        lists[list].push(item);
    }
    State::from_lists_interruptible(model, per, lists, stop)
}

fn route_for_item(per: &PerList, state: &State, item: i32) -> Option<usize> {
    per.globals.value_to_idx.get(&item).map(|&index| state.item_list[index])
}

fn oriented_route(route: &[i32], endpoint: i32, at_end: bool, may_reverse: bool) -> Option<Vec<i32>> {
    let already_oriented = if at_end { route.last() == Some(&endpoint) } else { route.first() == Some(&endpoint) };
    if already_oriented {
        return Some(route.to_vec());
    }
    let reversible = if at_end { route.first() == Some(&endpoint) } else { route.last() == Some(&endpoint) };
    if may_reverse && reversible {
        return Some(route.iter().rev().copied().collect());
    }
    None
}

fn matrix_cost(matrix: &[Vec<i64>], from: i32, to: i32) -> Option<i64> {
    let from = usize::try_from(from).ok()?;
    let to = usize::try_from(to).ok()?;
    matrix.get(from)?.get(to).copied()
}

fn savings_state(
    model: &CollectionModel,
    per: &PerList,
    mut state: State,
    seed: u64,
    stop: &AtomicBool,
    work: &mut RoutingConstructionWork<'_>,
) -> State {
    let Some(routing) = &per.routing else {
        return state;
    };
    let mut savings = Vec::new();
    let mut seen = HashSet::new();
    let base_neighbor_limit = ((model.items.len() as f64).sqrt().ceil() as usize).clamp(8, 64);
    let neighbor_limit = if routing.has_time_windows { base_neighbor_limit.saturating_mul(2).min(64) } else { base_neighbor_limit };
    for &from in &model.items {
        let nearby: Vec<i32> = if let Some(neighbors) = &per.candidates {
            interleaved_construction_neighbors(neighbors, from, neighbor_limit)
        } else {
            model.items.iter().copied().filter(|&item| item != from).take(neighbor_limit).collect()
        };
        for to in nearby {
            if from == to || !seen.insert((from, to)) {
                continue;
            }
            let Some(value) = matrix_cost(&routing.matrix, routing.depot, from)
                .zip(matrix_cost(&routing.matrix, to, routing.depot))
                .zip(matrix_cost(&routing.matrix, from, to))
                .map(|((out, back), direct)| out.saturating_add(back).saturating_sub(direct))
            else {
                continue;
            };
            savings.push((value, mix64(seed ^ mix64(from as u64) ^ to as u64), from, to));
        }
    }
    savings.sort_unstable_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    // Reversal is a candidate orientation, not an assumption of symmetry. The
    // complete merged route is checked below, so time-window-infeasible reverse
    // orientations are rejected exactly.
    let may_reverse = true;
    let mut scratch = EvalScratch::default();
    for (_, _, from, to) in savings {
        if stop.load(Ordering::Relaxed) || work.exhausted() {
            break;
        }
        let (Some(left), Some(right)) = (route_for_item(per, &state, from), route_for_item(per, &state, to)) else {
            continue;
        };
        if left == right {
            continue;
        }
        let (Some(mut left_route), Some(right_route)) =
            (oriented_route(&state.lists[left], from, true, may_reverse), oriented_route(&state.lists[right], to, false, may_reverse))
        else {
            continue;
        };
        left_route.extend_from_slice(&right_route);
        let empty: Vec<i32> = Vec::new();
        work.consume_candidate();
        per.metrics.record_candidate();
        let left_score = trial_list_score_view(per, &state, left, &left_route, None, &mut scratch);
        let right_score = trial_list_score_view(per, &state, right, &empty, None, &mut scratch);
        if left_score.violation != 0 || right_score.violation != 0 {
            continue;
        }
        let overrides: Vec<(i32, usize)> = state.lists[right].iter().map(|&item| (item, left)).collect();
        let global_delta = per.globals.delta(&state.item_list, &overrides);
        let candidate_score = score_with_replacements_mode(
            per,
            &state,
            &[
                TrialList { list: left, score: &left_score, contents: &left_route },
                TrialList { list: right, score: &right_score, contents: &empty },
            ],
            global_delta,
            &mut scratch,
            false,
        );
        if candidate_score.violation != 0 || candidate_score >= full_score_raw(per, &state) {
            continue;
        }
        state.lists[left] = left_route;
        state.lists[right].clear();
        if !state.rescore_interruptible(per, left, stop) || !state.rescore_interruptible(per, right, stop) {
            break;
        }
        for &(item, list) in &overrides {
            state.set_item_list(per, item, list);
        }
        state.global_viol = per.globals.total(&state.item_list);
        let score = full_score_raw(per, &state);
        work.observe(&state, score);
    }
    state
}

/// Clarke-Wright construction for a model whose route count is smaller than
/// the customer count. The singleton routes live only in this temporary
/// list per customer. Every merge is replayed through the regular per-list
/// evaluator before it is accepted, which keeps capacity and scan constraints
/// hard during construction.
fn savings_from_singletons_state(
    model: &CollectionModel,
    per: &PerList,
    seed: u64,
    stop: &AtomicBool,
    work: &mut RoutingConstructionWork<'_>,
) -> Option<State> {
    let routing = per.routing.as_ref()?;
    if model.lists == 0 || model.items.is_empty() {
        return State::from_lists_interruptible(model, per, vec![Vec::new(); model.lists.max(1)], stop);
    }
    let mut routes: Vec<Vec<i32>> = model.items.iter().map(|&item| vec![item]).collect();
    if routes.iter().any(|route| list_score_exact(per, 0, route).violation != 0) {
        return None;
    }
    let mut owner: HashMap<i32, usize> = model.items.iter().enumerate().map(|(index, &item)| (item, index)).collect();
    let mut savings = Vec::new();
    let mut seen = HashSet::new();
    let base_neighbor_limit = ((model.items.len() as f64).sqrt().ceil() as usize).clamp(8, 64);
    let neighbor_limit = if routing.has_time_windows { base_neighbor_limit.saturating_mul(2).min(64) } else { base_neighbor_limit };
    for &from in &model.items {
        let nearby: Vec<i32> = if let Some(neighbors) = &per.candidates {
            interleaved_construction_neighbors(neighbors, from, neighbor_limit)
        } else {
            model.items.iter().copied().filter(|&item| item != from).take(neighbor_limit).collect()
        };
        for to in nearby {
            if from == to || !seen.insert((from, to)) {
                continue;
            }
            let value = matrix_cost(&routing.matrix, routing.depot, from)
                .zip(matrix_cost(&routing.matrix, to, routing.depot))
                .zip(matrix_cost(&routing.matrix, from, to))
                .map(|((out, back), direct)| out.saturating_add(back).saturating_sub(direct))?;
            savings.push((value, mix64(seed ^ mix64(from as u64) ^ to as u64), from, to));
        }
    }
    savings.sort_unstable_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    // Try both endpoint orientations even with time windows. Exact list
    // scoring below is the authority on whether the oriented merge is feasible.
    let may_reverse = true;
    let mut active_routes = routes.len();
    let mut best_feasible: Option<(State, Score)> = None;
    for (_, _, from, to) in savings {
        if stop.load(Ordering::Relaxed) || work.exhausted() {
            break;
        }
        let (Some(&left), Some(&right)) = (owner.get(&from), owner.get(&to)) else {
            continue;
        };
        if left == right {
            continue;
        }
        let (Some(mut merged), Some(right_route)) =
            (oriented_route(&routes[left], from, true, may_reverse), oriented_route(&routes[right], to, false, may_reverse))
        else {
            continue;
        };
        merged.extend_from_slice(&right_route);
        work.consume_candidate();
        per.metrics.record_candidate();
        if list_score_exact(per, 0, &merged).violation != 0 {
            continue;
        }
        routes[left] = merged;
        let moved = std::mem::take(&mut routes[right]);
        for item in moved {
            owner.insert(item, left);
        }
        active_routes -= 1;
        if active_routes <= model.lists {
            let mut candidate_lists: Vec<Vec<i32>> = routes.iter().filter(|route| !route.is_empty()).cloned().collect();
            candidate_lists.resize(model.lists.max(1), Vec::new());
            let candidate = State::from_lists_interruptible(model, per, candidate_lists, stop)?;
            let score = full_score_raw(per, &candidate);
            if score.violation == 0 && best_feasible.as_ref().is_none_or(|(_, incumbent)| score < *incumbent) {
                work.observe(&candidate, score.clone());
                best_feasible = Some((candidate, score));
            }
        }
    }
    if active_routes > model.lists {
        return None;
    }
    best_feasible.map(|(state, _)| state)
}

fn routing_construction_orders(model: &CollectionModel, routing: &RoutingSignature, randomized: &[usize]) -> Vec<Vec<usize>> {
    let stable: Vec<usize> = (0..model.items.len()).collect();
    let mut reversed = stable.clone();
    reversed.reverse();
    let mut orders = vec![stable];
    if let Some(windows) = routing.time_windows.as_ref() {
        let item_value = |index: usize| model.items[index];
        let time = |values: &[i64], index: usize| {
            usize::try_from(item_value(index)).ok().and_then(|item| values.get(item)).copied().unwrap_or(i64::MAX)
        };
        let mut earliest = (0..model.items.len()).collect::<Vec<_>>();
        earliest.sort_by_key(|&index| (time(&windows.earliest, index), item_value(index)));
        let mut latest = (0..model.items.len()).collect::<Vec<_>>();
        latest.sort_by_key(|&index| (time(&windows.latest_start, index), item_value(index)));
        let mut tight = (0..model.items.len()).collect::<Vec<_>>();
        tight.sort_by_key(|&index| {
            (
                time(&windows.latest_start, index).saturating_sub(time(&windows.earliest, index)),
                time(&windows.earliest, index),
                item_value(index),
            )
        });
        orders.extend([earliest, latest, tight]);
    }
    orders.extend([randomized.to_vec(), reversed]);
    let mut unique = Vec::with_capacity(orders.len());
    for order in orders {
        if !unique.contains(&order) {
            unique.push(order);
        }
    }
    unique
}

fn sequential_insertion_state(
    model: &CollectionModel,
    per: &PerList,
    attempt: u64,
    seed: u64,
    stop: &AtomicBool,
    candidates: &mut u64,
    candidate_budget: u64,
) -> Option<State> {
    let routing = per.routing.as_ref()?;
    let windows = routing.time_windows.as_ref()?;
    let depot = usize::try_from(routing.depot).ok()?;
    let depot_distance = |item: i32| {
        let item = usize::try_from(item).ok()?;
        routing.matrix.get(depot)?.get(item).copied()
    };
    let depot_round_trip = |item: i32| {
        let item = usize::try_from(item).ok()?;
        routing
            .matrix
            .get(depot)?
            .get(item)
            .copied()
            .zip(routing.matrix.get(item)?.get(depot).copied())
            .map(|(outbound, inbound)| outbound.saturating_add(inbound))
    };
    let latest = |item: i32| usize::try_from(item).ok().and_then(|item| windows.latest_start.get(item)).copied().unwrap_or(i64::MAX);
    let slack = |item: i32| {
        usize::try_from(item)
            .ok()
            .and_then(|item| windows.earliest.get(item).zip(windows.latest_start.get(item)))
            .map_or(i64::MAX, |(&earliest, &latest)| latest.saturating_sub(earliest))
    };
    let mut remaining = model.items.clone();
    let mut routes = Vec::new();
    while !remaining.is_empty() {
        if stop.load(Ordering::Relaxed) || *candidates >= candidate_budget || routes.len() >= model.lists {
            return None;
        }
        let classical = attempt < 6;
        let old_variant = attempt.saturating_sub(6);
        let seed_rule = if classical {
            attempt % 2
        } else {
            match old_variant {
                0 => 1,
                1 => 2,
                2 => 0,
                _ => 2,
            }
        };
        let seed_at = match seed_rule {
            0 => remaining
                .iter()
                .enumerate()
                .max_by_key(|&(_, &item)| (depot_distance(item).unwrap_or(i64::MIN), std::cmp::Reverse(latest(item)), item))
                .map(|(index, _)| index),
            1 => remaining
                .iter()
                .enumerate()
                .min_by_key(|&(_, &item)| (latest(item), std::cmp::Reverse(depot_distance(item)), item))
                .map(|(index, _)| index),
            _ => remaining
                .iter()
                .enumerate()
                .min_by_key(|&(_, &item)| (slack(item), latest(item), std::cmp::Reverse(depot_distance(item)), item))
                .map(|(index, _)| index),
        }?;
        let route_seed = remaining.swap_remove(seed_at);
        let mut route = vec![route_seed];
        let (mut route_metrics, mut route_starts, mut return_start) = fixed_route_schedule(routing, &route)?;
        let (distance_weight, delay_weight) = match (attempt / 2) % 3 {
            0 => (2i128, 0i128),
            1 => (1, 1),
            _ => (0, 2),
        };

        loop {
            if stop.load(Ordering::Relaxed) || *candidates >= candidate_budget {
                return None;
            }
            let mut selected: Option<(i128, i64, u64, usize, usize, FixedRouteMetrics)> = None;
            for (remaining_at, &item) in remaining.iter().enumerate() {
                for position in 0..=route.len() {
                    if *candidates >= candidate_budget {
                        return None;
                    }
                    *candidates = candidates.saturating_add(1);
                    per.metrics.record_candidate();
                    let mut inserted = Vec::with_capacity(route.len().saturating_add(1));
                    inserted.extend_from_slice(&route[..position]);
                    inserted.push(item);
                    inserted.extend_from_slice(&route[position..]);
                    let observed = (position < route.len()).then_some(position.saturating_add(1));
                    let Some((metrics, successor_start, candidate_return)) = fixed_route_probe(routing, &inserted, observed) else {
                        continue;
                    };
                    let distance_delta = metrics.distance.saturating_sub(route_metrics.distance);
                    let delay = if position < route.len() {
                        successor_start.unwrap_or(i64::MAX).saturating_sub(route_starts[position]).max(0)
                    } else {
                        candidate_return.saturating_sub(return_start).max(0)
                    };
                    let radial = depot_distance(item).unwrap_or(0);
                    let desirability = if classical {
                        let insertion_cost = distance_weight
                            .saturating_mul(i128::from(distance_delta))
                            .saturating_add(delay_weight.saturating_mul(i128::from(delay)));
                        i128::from(radial).saturating_mul(2).saturating_sub(insertion_cost)
                    } else {
                        let slack_loss = route_metrics.minimum_slack.saturating_sub(metrics.minimum_slack).max(0);
                        let lambda = i128::from(if old_variant >= 2 { 2 } else { 1 });
                        lambda
                            .saturating_mul(i128::from(depot_round_trip(item).unwrap_or(0)))
                            .saturating_sub(i128::from(distance_delta).saturating_mul(2))
                            .saturating_sub(i128::from(slack_loss))
                    };
                    let tie = mix64(seed ^ mix64(attempt) ^ mix64(item as i64 as u64));
                    let rank = (desirability, metrics.minimum_slack, std::cmp::Reverse(tie));
                    if selected.as_ref().is_none_or(|current| rank > (current.0, current.1, std::cmp::Reverse(current.2))) {
                        selected = Some((desirability, metrics.minimum_slack, tie, remaining_at, position, metrics));
                    }
                }
            }
            let Some((_, _, _, remaining_at, position, metrics)) = selected else { break };
            let item = remaining.swap_remove(remaining_at);
            route.insert(position, item);
            (route_metrics, route_starts, return_start) = fixed_route_schedule(routing, &route)?;
            debug_assert_eq!(route_metrics.distance, metrics.distance);
        }
        routes.push(route);
    }
    routes.resize(model.lists.max(1), Vec::new());
    let state = State::from_lists_interruptible(model, per, routes, stop)?;
    (full_score_raw(per, &state).violation == 0).then_some(state)
}

fn routing_construction(
    model: &CollectionModel,
    per: &PerList,
    order: &[usize],
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> Option<InitialConstruction> {
    let routing = per.routing.as_ref()?;
    if let (Some(demands), Some(capacity)) = (&routing.demands, routing.capacity) {
        if model
            .items
            .iter()
            .any(|&item| usize::try_from(item).ok().and_then(|index| demands.get(index)).is_some_and(|&demand| demand > capacity))
        {
            return None;
        }
    }
    let started = Instant::now();
    let root = ((model.items.len() as f64).sqrt().ceil() as u64).clamp(8, 64);
    let candidate_budget = (model.items.len() as u64).saturating_mul(root).saturating_mul(7_680).clamp(40_000, 30_000_000);
    let mut candidates = 0u64;
    let mut feasible_history = Vec::new();
    let mut best: Option<(State, &'static str)> = None;
    let mut fallback_lists = None;
    let construction_orders = routing_construction_orders(model, routing, order);

    if let Some(singletons) = singleton_state(model, per, stop) {
        let singleton_score = full_score_raw(per, &singletons);
        if singleton_score.violation == 0 {
            let improved =
                observe_construction_incumbent(&mut feasible_history, &singletons, singleton_score.clone(), started.elapsed(), candidates);
            if improved {
                fallback_lists = Some((singletons.lists.clone(), "parallel-savings"));
                if per.tiers > 0 {
                    report(tier_value(per, &singleton_score, 0));
                }
            }
            let saved = savings_state(
                model,
                per,
                singletons,
                seed,
                stop,
                &mut RoutingConstructionWork {
                    candidates: &mut candidates,
                    candidate_budget,
                    feasible_history: &mut feasible_history,
                    started: &started,
                    per,
                    report,
                    fallback_lists: &mut fallback_lists,
                },
            );
            if stop.load(Ordering::Relaxed) {
                best = fallback_lists.take().map(|(lists, name)| (State::from_lists(model, per, lists), name));
            } else {
                let saved_score = full_score_raw(per, &saved);
                if observe_construction_incumbent(&mut feasible_history, &saved, saved_score.clone(), started.elapsed(), candidates)
                    && per.tiers > 0
                {
                    report(tier_value(per, &saved_score, 0));
                }
                best = Some((saved, "parallel-savings"));
            }
        }
    } else if let Some(saved) = savings_from_singletons_state(
        model,
        per,
        seed,
        stop,
        &mut RoutingConstructionWork {
            candidates: &mut candidates,
            candidate_budget,
            feasible_history: &mut feasible_history,
            started: &started,
            per,
            report,
            fallback_lists: &mut fallback_lists,
        },
    ) {
        let score = full_score_raw(per, &saved);
        if observe_construction_incumbent(&mut feasible_history, &saved, score.clone(), started.elapsed(), candidates) && per.tiers > 0 {
            report(tier_value(per, &score, 0));
        }
        best = Some((saved, "parallel-savings"));
    }

    if model.items.len() <= 512 && candidates < candidate_budget {
        let sequential_budget =
            u64::try_from(model.items.len()).unwrap_or(u64::MAX).saturating_pow(2).saturating_mul(128).clamp(100_000, 20_000_000);
        let sequential_limit = candidates.saturating_add(sequential_budget).min(candidate_budget);
        for attempt in 0..10 {
            if stop.load(Ordering::Relaxed) || candidates >= sequential_limit {
                break;
            }
            let Some(candidate) = sequential_insertion_state(model, per, attempt, seed, stop, &mut candidates, sequential_limit) else {
                continue;
            };
            let score = full_score_raw(per, &candidate);
            if observe_construction_incumbent(&mut feasible_history, &candidate, score.clone(), started.elapsed(), candidates) {
                fallback_lists = Some((candidate.lists.clone(), "sequential-insertion"));
                if per.tiers > 0 {
                    report(tier_value(per, &score, 0));
                }
            }
            if best.as_ref().is_none_or(|(incumbent, _)| score < full_score_raw(per, incumbent)) {
                best = Some((candidate, "sequential-insertion"));
            }
        }
    }

    if model.items.len() <= 512 || best.is_none() {
        let insertion_budget = (model.items.len() as u64).saturating_mul(root).saturating_mul(640).clamp(10_000, 2_000_000);
        let insertion_limit = candidates.saturating_add(insertion_budget).min(candidate_budget);
        for attempt in &construction_orders {
            if stop.load(Ordering::Relaxed) || candidates >= insertion_limit {
                break;
            }
            if let Some(cheapest) = cheapest_insertion_state(model, per, attempt, stop, &mut candidates, insertion_limit) {
                let score = full_score_raw(per, &cheapest);
                if score.violation == 0 {
                    let improved =
                        observe_construction_incumbent(&mut feasible_history, &cheapest, score.clone(), started.elapsed(), candidates);
                    if improved {
                        fallback_lists = Some((cheapest.lists.clone(), "cheapest-insertion"));
                        if per.tiers > 0 {
                            report(tier_value(per, &score, 0));
                        }
                    }
                    if best.as_ref().is_none_or(|(incumbent, _)| score < full_score_raw(per, incumbent)) {
                        best = Some((cheapest, "cheapest-insertion"));
                    }
                }
            }
        }
    }

    if model.items.len() <= 160 && candidates < candidate_budget {
        for (attempt_index, attempt) in construction_orders.iter().enumerate() {
            if stop.load(Ordering::Relaxed) || candidates >= candidate_budget {
                break;
            }
            let attempt_seed = seed ^ mix64(u64::try_from(attempt_index).unwrap_or(u64::MAX));
            let Some(regret) = regret_insertion_state(model, per, attempt, attempt_seed, stop, &mut candidates, candidate_budget) else {
                continue;
            };
            let score = full_score_raw(per, &regret);
            if score.violation == 0 {
                let improved = observe_construction_incumbent(&mut feasible_history, &regret, score.clone(), started.elapsed(), candidates);
                if improved {
                    fallback_lists = Some((regret.lists.clone(), "regret-insertion"));
                    if per.tiers > 0 {
                        report(tier_value(per, &score, 0));
                    }
                }
                if best.as_ref().is_none_or(|(incumbent, _)| score < full_score_raw(per, incumbent)) {
                    best = Some((regret, "regret-insertion"));
                }
            }
        }
    }

    if stop.load(Ordering::Relaxed) {
        best = fallback_lists.take().map(|(lists, name)| (State::from_lists(model, per, lists), name)).or(best);
    }
    let (state, name) = best?;
    let reported = per.tiers > 0 && !feasible_history.is_empty();
    Some(InitialConstruction { state, incumbent: None, name, elapsed: started.elapsed(), feasible_history, candidates, reported })
}

fn combine_routing_constructions(per: &PerList, mut primary: InitialConstruction, mut stable: InitialConstruction) -> InitialConstruction {
    let primary_score = full_score_raw(per, &primary.state);
    let stable_score = full_score_raw(per, &stable.state);
    let primary_fleet = primary.state.lists.iter().filter(|route| !route.is_empty()).count();
    let stable_fleet = stable.state.lists.iter().filter(|route| !route.is_empty()).count();
    let search_stable = stable_fleet <= primary_fleet;
    let primary_is_best = primary_score < stable_score;
    let best_reported = primary_is_best && primary.reported;

    let offset = primary.elapsed;
    let candidate_offset = primary.candidates;
    for snapshot in &mut stable.feasible_history {
        snapshot.elapsed = offset.saturating_add(snapshot.elapsed);
        snapshot.candidates = candidate_offset.saturating_add(snapshot.candidates);
    }
    primary.feasible_history.extend(stable.feasible_history);
    primary.candidates = primary.candidates.saturating_add(stable.candidates);
    primary.elapsed = primary.elapsed.saturating_add(stable.elapsed);
    primary.reported = best_reported;
    primary.name = "routing-portfolio";

    if search_stable {
        if primary_is_best {
            stable.incumbent = Some(primary.state);
        }
        stable.feasible_history = primary.feasible_history;
        stable.candidates = primary.candidates;
        stable.elapsed = primary.elapsed;
        stable.reported = primary.reported;
        stable.name = primary.name;
        stable
    } else {
        if !primary_is_best {
            primary.incumbent = Some(stable.state);
        }
        primary
    }
}

/// Raw (unsigned) value of a tier, undoing the maximisation sign flip.
fn tier_value(per: &PerList, score: &Score, tier: usize) -> i64 {
    if per.senses[tier] {
        score.tiers[tier]
    } else {
        score.tiers[tier].saturating_neg()
    }
}

/// The raw objective values of every declared tier, in model order.
fn objective_values(per: &PerList, score: &Score) -> Vec<i64> {
    (0..per.tiers).map(|t| tier_value(per, score, t)).collect()
}

fn record_state(
    per: &PerList,
    state: &State,
    best_lists: &mut Vec<Vec<i32>>,
    best_score: &mut Score,
    best_feasible: &mut bool,
    report: &mut dyn FnMut(i64),
) -> bool {
    let score = full_score_raw(per, state);
    let feasible = score.violation == 0;
    if !better(feasible, &score, *best_feasible, best_score) {
        return false;
    }
    *best_lists = state.lists.clone();
    *best_score = score;
    *best_feasible = feasible;
    if feasible && per.tiers > 0 {
        report(tier_value(per, best_score, 0));
    }
    true
}

pub(super) fn score_with_replaced_list(
    per: &PerList,
    state: &State,
    list: usize,
    replacement: &ListScore,
    contents: &dyn ListView,
    global_delta: i64,
    scratch: &mut EvalScratch,
) -> Score {
    score_with_replacements(per, state, &[TrialList { list, score: replacement, contents }], global_delta, scratch)
}

#[allow(clippy::too_many_arguments)]
fn run_generic_search(
    model: &CollectionModel,
    per: &PerList,
    state: &mut State,
    stop: &AtomicBool,
    max_iters: u64,
    seed: u64,
    profile: SearchProfile,
    memory: &mut SearchMemory,
    best_lists: &mut Vec<Vec<i32>>,
    best_score: &mut Score,
    best_feasible: &mut bool,
    report: &mut dyn FnMut(i64),
    alns: &mut AlnsController,
    coordination: &mut Option<WorkerCoordination<'_>>,
) -> bool {
    let mut stagnant = 0u64;
    let mut iter = 0u64;
    let mut local_optima = 0u64;

    while !stop.load(Ordering::Relaxed) && iter < max_iters {
        iter += 1;
        let scan_seed = if profile.diversify_initial_descent() { seed ^ mix64(iter) } else { 0 };
        match best_improving_move(per, state, stop, memory, scan_seed) {
            Some(mv) => {
                if !apply_move(per, state, mv, stop) {
                    return false;
                }
                memory.reset_touched(mv);
                if record_state(per, state, best_lists, best_score, best_feasible, report) {
                    if coordination.as_ref().is_some_and(|shared| shared.publish(best_lists, best_score, *best_feasible)) {
                        alns.record_shared_publication();
                    }
                    stagnant = 0;
                }
            }
            None => {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                local_optima += 1;
                if record_state(per, state, best_lists, best_score, best_feasible, report) {
                    if coordination.as_ref().is_some_and(|shared| shared.publish(best_lists, best_score, *best_feasible)) {
                        alns.record_shared_publication();
                    }
                    stagnant = 0;
                } else {
                    stagnant = stagnant.saturating_add(1);
                }

                if let Some(lists) = coordination.as_mut().and_then(|shared| shared.maybe_inject(local_optima, best_score, *best_feasible))
                {
                    let Some(injected_state) = State::from_lists_interruptible(model, per, lists, stop) else {
                        return false;
                    };
                    *state = injected_state;
                    let injected = record_state(per, state, best_lists, best_score, best_feasible, report);
                    debug_assert!(injected, "a shared incumbent must strictly improve the worker incumbent");
                    alns.record_shared_injection();
                    alns.reset_after_injection(best_score);
                    memory.reset_all();
                    stagnant = 0;
                    continue;
                }

                let elite_period = profile.elite_restart_period();
                if stagnant > 0 && stagnant.is_multiple_of(elite_period) {
                    let Some(elite_state) = State::from_lists_interruptible(model, per, best_lists.clone(), stop) else {
                        return false;
                    };
                    *state = elite_state;
                    memory.reset_all();
                    alns.reset_after_injection(best_score);
                }

                let penalties = per.bump_gls(state);
                state.refresh_edge_penalties(per);
                alns.record_gls(penalties);

                let operator_seed = seed ^ mix64(iter) ^ mix64(stagnant);
                let choice = alns.choose(model.items.len(), stagnant, operator_seed, iter);
                let current_guided = full_score(per, state);
                let current_raw = full_score_raw(per, state);
                let Some(candidate) = build_candidate(model, per, state, choice, operator_seed, stop) else {
                    alns.record_failed(choice.destroy, choice.repair);
                    continue;
                };
                let candidate_guided = full_score(per, &candidate);
                let candidate_raw = full_score_raw(per, &candidate);
                let improved_current = candidate_raw < current_raw;
                let global_best = record_state(per, &candidate, best_lists, best_score, best_feasible, report);
                if global_best && coordination.as_ref().is_some_and(|shared| shared.publish(best_lists, best_score, *best_feasible)) {
                    alns.record_shared_publication();
                }
                let acceptance = alns.accept(&current_guided, &candidate_guided, &current_raw, &candidate_raw, operator_seed, iter);
                if !matches!(acceptance, AcceptanceKind::Rejected) {
                    *state = candidate;
                    memory.reset_all();
                }
                alns.record(choice.destroy, choice.repair, acceptance, improved_current, global_best);
                if global_best {
                    stagnant = 0;
                }
            }
        }
    }
    true
}

fn scaled_work(items: usize, multiplier: u64, minimum: u64, maximum: u64) -> u64 {
    u64::try_from(items).unwrap_or(u64::MAX).saturating_mul(multiplier).clamp(minimum, maximum)
}

fn routing_move_budget(items: usize) -> WorkBudget {
    let work = scaled_work(items, 8, 512, 8_192);
    WorkBudget::new(work, work)
}

fn routing_global_budget(items: usize) -> WorkBudget {
    let work = scaled_work(items, 2, 256, 2_048);
    WorkBudget::new(work, work)
}

fn routing_alns_budget(items: usize) -> AlnsWorkBudget {
    let work = scaled_work(items, 32, 2_048, 32_768);
    AlnsWorkBudget::new(work, work)
}

fn routing_exploration_budget(model: &CollectionModel) -> AlnsWorkBudget {
    // Collection states are exact partitions of the model items. Use the
    // canonical cardinality so budget admission itself stays O(1).
    let items = model.items.len();
    let structural_scale = items.saturating_add(model.lists);
    let generated = scaled_work(structural_scale, 32, 4_096, 65_536);
    let evaluated = scaled_work(items, 12, 512, 12_288);
    AlnsWorkBudget::new(generated, evaluated)
}

fn routing_macro_budget(
    model: &CollectionModel,
    per: &PerList,
    state: &State,
    operator: MacroOperator,
    stop: &AtomicBool,
) -> Option<AlnsWorkBudget> {
    let exploratory = routing_exploration_budget(model);
    let required = match operator {
        MacroOperator::RouteElimination => {
            let greedy = routing_route_elimination_floor(model, per, state, stop)?;
            AlnsWorkBudget::new(greedy.generated.saturating_mul(8), greedy.evaluated.saturating_mul(8))
        }
        MacroOperator::EjectionChain | MacroOperator::ChainRelocate | MacroOperator::GuidedSegmentExchange => {
            AlnsWorkBudget::new(routing_compound_structural_floor(model, per, state, stop)?, 1)
        }
    };
    Some(AlnsWorkBudget::new(exploratory.generated.max(required.generated), exploratory.evaluated.max(required.evaluated)))
}

#[derive(Clone, Copy)]
struct FixedRouteMetrics {
    distance: i64,
    minimum_slack: i64,
}

#[derive(Clone)]
struct FixedFleetPartial {
    routes: Vec<Arc<Vec<i32>>>,
    metrics: Vec<Option<FixedRouteMetrics>>,
}

type FixedFleetRank = (usize, i64, std::cmp::Reverse<i64>, usize, u64);

impl FixedFleetPartial {
    fn rank(&self, target_used: usize, seed: u64) -> FixedFleetRank {
        let used = self.routes.iter().filter(|route| !route.is_empty()).count();
        let distance = self.metrics.iter().flatten().fold(0i64, |total, metrics| total.saturating_add(metrics.distance));
        let minimum_slack = self.metrics.iter().flatten().map(|metrics| metrics.minimum_slack).min().unwrap_or(i64::MAX);
        let minimum_len = self.routes.iter().map(|route| route.len()).min().unwrap_or(0);
        let maximum_len = self.routes.iter().map(|route| route.len()).max().unwrap_or(0);
        let tie = self.routes.iter().enumerate().fold(seed, |hash, (list, route)| {
            mix64(hash ^ mix64(list as u64) ^ mix64(route.last().copied().unwrap_or(-1) as i64 as u64) ^ mix64(route.len() as u64))
        });
        (target_used.saturating_sub(used), distance, std::cmp::Reverse(minimum_slack), maximum_len.saturating_sub(minimum_len), tie)
    }

    fn signature(&self) -> Vec<(usize, i32)> {
        self.routes.iter().map(|route| (route.len(), route.last().copied().unwrap_or(-1))).collect()
    }
}

fn fixed_route_probe(
    routing: &RoutingSignature,
    route: &[i32],
    observed_position: Option<usize>,
) -> Option<(FixedRouteMetrics, Option<i64>, i64)> {
    if route.is_empty() {
        return None;
    }
    let windows = routing.time_windows.as_ref()?;
    let depot = usize::try_from(routing.depot).ok()?;
    let mut previous = depot;
    let mut departure = windows.earliest.get(depot).copied()?;
    let mut minimum_slack = i64::MAX;
    let mut distance = 0i64;
    let mut load = 0i64;
    let mut observed_start = None;
    for (position, &item) in route.iter().enumerate() {
        let current = usize::try_from(item).ok()?;
        if let Some(demands) = &routing.demands {
            load = load.saturating_add(demands.get(current).copied()?);
            if routing.capacity.is_some_and(|capacity| load > capacity) {
                return None;
            }
        }
        let travel = windows.travel.get(previous)?.get(current).copied()?;
        let start = windows.earliest.get(current).copied()?.max(departure.saturating_add(travel));
        let latest = windows.latest_start.get(current).copied()?;
        if start > latest {
            return None;
        }
        if observed_position == Some(position) {
            observed_start = Some(start);
        }
        minimum_slack = minimum_slack.min(latest.saturating_sub(start));
        departure = start.saturating_add(windows.service.get(current).copied()?);
        distance = distance.saturating_add(routing.matrix.get(previous)?.get(current).copied()?);
        previous = current;
    }
    let return_start =
        windows.earliest.get(depot).copied()?.max(departure.saturating_add(windows.travel.get(previous)?.get(depot).copied()?));
    let depot_latest = windows.latest_start.get(depot).copied()?;
    if return_start > depot_latest {
        return None;
    }
    minimum_slack = minimum_slack.min(depot_latest.saturating_sub(return_start));
    distance = distance.saturating_add(routing.matrix.get(previous)?.get(depot).copied()?);
    Some((FixedRouteMetrics { distance, minimum_slack }, observed_start, return_start))
}

fn fixed_route_metrics(routing: &RoutingSignature, route: &[i32]) -> Option<FixedRouteMetrics> {
    fixed_route_probe(routing, route, None).map(|(metrics, _, _)| metrics)
}

fn fixed_route_schedule(routing: &RoutingSignature, route: &[i32]) -> Option<(FixedRouteMetrics, Vec<i64>, i64)> {
    let (metrics, _, return_start) = fixed_route_probe(routing, route, None)?;
    let windows = routing.time_windows.as_ref()?;
    let mut previous = usize::try_from(routing.depot).ok()?;
    let mut departure = windows.earliest.get(previous).copied()?;
    let mut starts = Vec::with_capacity(route.len());
    for &item in route {
        let current = usize::try_from(item).ok()?;
        let start =
            windows.earliest.get(current).copied()?.max(departure.saturating_add(windows.travel.get(previous)?.get(current).copied()?));
        starts.push(start);
        departure = start.saturating_add(windows.service.get(current).copied()?);
        previous = current;
    }
    Some((metrics, starts, return_start))
}

fn fixed_fleet_order(items: &[i32], routing: &RoutingSignature, attempt: u64, seed: u64) -> Vec<i32> {
    let mut order = items.to_vec();
    let depot = usize::try_from(routing.depot).unwrap_or(0);
    let windows = routing.time_windows.as_ref().expect("fixed-fleet rebuilding requires recognized time windows");
    let depot_distance = |item: i32| {
        usize::try_from(item).ok().and_then(|index| routing.matrix.get(depot).and_then(|row| row.get(index))).copied().unwrap_or(i64::MIN)
    };
    let earliest = |item: i32| usize::try_from(item).ok().and_then(|index| windows.earliest.get(index)).copied().unwrap_or(i64::MAX);
    let latest = |item: i32| usize::try_from(item).ok().and_then(|index| windows.latest_start.get(index)).copied().unwrap_or(i64::MAX);
    match attempt % 6 {
        0 => order.sort_unstable_by_key(|&item| (latest(item), std::cmp::Reverse(depot_distance(item)), item)),
        1 => order.sort_unstable_by_key(|&item| (earliest(item), latest(item), std::cmp::Reverse(depot_distance(item)), item)),
        2 => order.sort_unstable_by_key(|&item| {
            (latest(item).saturating_sub(earliest(item)), latest(item), std::cmp::Reverse(depot_distance(item)), item)
        }),
        3 => order.sort_unstable_by_key(|&item| (std::cmp::Reverse(depot_distance(item)), latest(item), item)),
        4 => order.sort_unstable_by_key(|&item| (mix64(seed ^ mix64(item as i64 as u64)), item)),
        _ => order.sort_unstable_by_key(|&item| (std::cmp::Reverse(latest(item)), std::cmp::Reverse(depot_distance(item)), item)),
    }
    order
}

#[allow(clippy::too_many_arguments)]
fn fixed_fleet_beam_state(
    model: &CollectionModel,
    per: &PerList,
    target: usize,
    attempt: u64,
    seed: u64,
    stop: &AtomicBool,
    candidates: &mut u64,
    candidate_budget: u64,
) -> Option<State> {
    let routing = per.routing.as_ref()?;
    if target == 0 || model.items.len() < target {
        return None;
    }
    let mut lists = fixed_fleet_beam_lists(&model.items, routing, &per.metrics, target, attempt, seed, stop, candidates, candidate_budget)?;
    lists.resize_with(model.lists.max(1), Vec::new);
    let state = State::from_lists_interruptible(model, per, lists, stop)?;
    (full_score_raw(per, &state).violation == 0 && state.lists.iter().filter(|route| !route.is_empty()).count() == target).then_some(state)
}

#[allow(clippy::too_many_arguments)]
fn fixed_fleet_beam_lists(
    items: &[i32],
    routing: &RoutingSignature,
    metrics: &MetricsRecorder,
    target: usize,
    attempt: u64,
    seed: u64,
    stop: &AtomicBool,
    candidates: &mut u64,
    candidate_budget: u64,
) -> Option<Vec<Vec<i32>>> {
    routing.time_windows.as_ref()?;
    if target == 0 || items.len() < target {
        return None;
    }
    let width = match (items.len() >= 256, attempt % 3) {
        (true, 0) => 16,
        (true, 1) => 48,
        (true, _) => 32,
        (false, 0) => 64,
        (false, 1) => 256,
        (false, _) => 128,
    };
    let empty = Arc::new(Vec::new());
    let mut beam = vec![FixedFleetPartial { routes: vec![empty; target], metrics: vec![None; target] }];
    let order = fixed_fleet_order(items, routing, attempt, seed);
    for (inserted, item) in order.into_iter().enumerate() {
        if stop.load(Ordering::Relaxed) || *candidates >= candidate_budget {
            return None;
        }
        let mut next = Vec::with_capacity(beam.len().saturating_mul(target).saturating_mul(8));
        for partial in &beam {
            for list in 0..target {
                let route = &partial.routes[list];
                for position in 0..=route.len() {
                    if *candidates >= candidate_budget {
                        return None;
                    }
                    *candidates = candidates.saturating_add(1);
                    metrics.record_candidate();
                    let mut replacement = Vec::with_capacity(route.len().saturating_add(1));
                    replacement.extend_from_slice(&route[..position]);
                    replacement.push(item);
                    replacement.extend_from_slice(&route[position..]);
                    let Some(metrics) = fixed_route_metrics(routing, &replacement) else { continue };
                    let mut candidate = partial.clone();
                    candidate.routes[list] = Arc::new(replacement);
                    candidate.metrics[list] = Some(metrics);
                    next.push(candidate);
                }
            }
        }
        if next.is_empty() {
            return None;
        }
        let target_used = target.min(inserted.saturating_add(1));
        next.sort_unstable_by_key(|candidate| candidate.rank(target_used, seed));
        let mut signatures = HashSet::with_capacity(width);
        beam.clear();
        for candidate in next {
            if signatures.insert(candidate.signature()) {
                beam.push(candidate);
                if beam.len() >= width {
                    break;
                }
            }
        }
    }
    beam.sort_unstable_by_key(|candidate| candidate.rank(target, seed));
    beam.into_iter().next().map(|partial| partial.routes.into_iter().map(|route| route.as_ref().clone()).collect())
}

fn fixed_fleet_rebuild(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    attempt: u64,
    seed: u64,
    stop: &AtomicBool,
) -> AlnsBuildRun {
    let started = Instant::now();
    let fleet = current.lists.iter().filter(|route| !route.is_empty()).count();
    let target = fleet.saturating_sub(1);
    let mut candidates = 0u64;
    let candidate_budget = u64::try_from(model.items.len())
        .unwrap_or(u64::MAX)
        .saturating_pow(2)
        .saturating_mul(u64::try_from(target.max(1)).unwrap_or(u64::MAX))
        .saturating_mul(32)
        .clamp(250_000, 4_000_000);
    let candidate = (target > 0)
        .then(|| fixed_fleet_beam_state(model, per, target, attempt, seed, stop, &mut candidates, candidate_budget))
        .flatten()
        .filter(|state| {
            full_score_raw(per, state).violation == 0 && state.lists.iter().filter(|route| !route.is_empty()).count() <= target
        });
    let status = if candidate.is_some() {
        AlnsBuildStatus::Built
    } else if stop.load(Ordering::Relaxed) {
        AlnsBuildStatus::Interrupted
    } else if candidates >= candidate_budget {
        AlnsBuildStatus::BudgetExhausted
    } else {
        AlnsBuildStatus::Infeasible
    };
    let cpu_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    AlnsBuildRun {
        status,
        candidate,
        generated: candidates,
        evaluated: candidates,
        work_units: candidates.saturating_mul(2),
        structural_work: 0,
        repair_index_work: 0,
        canonical_rebuilds: u64::from(matches!(status, AlnsBuildStatus::Built)),
        cpu_nanos,
        destroy_generated: 0,
        destroy_evaluated: 0,
        destroy_cpu_nanos: 0,
        repair_generated: candidates,
        repair_evaluated: candidates,
        repair_cpu_nanos: cpu_nanos,
        destroy_executed: false,
        repair_executed: true,
        removed: model.items.len(),
        eliminated_route: matches!(status, AlnsBuildStatus::Built),
    }
}

fn fixed_pair_rebuild(model: &CollectionModel, per: &PerList, current: &State, attempt: u64, seed: u64, stop: &AtomicBool) -> AlnsBuildRun {
    let started = Instant::now();
    let routes: Vec<usize> = current.lists.iter().enumerate().filter_map(|(list, route)| (!route.is_empty()).then_some(list)).collect();
    let mut pairs: Vec<(u64, u64, usize, usize)> = routes
        .iter()
        .enumerate()
        .flat_map(|(left_at, &left)| {
            routes.iter().skip(left_at + 1).map(move |&right| {
                let relatedness = current.lists[left]
                    .iter()
                    .flat_map(|&left_item| {
                        current.lists[right].iter().map(move |&right_item| per.shaw_relatedness(left_item, right_item, false))
                    })
                    .min()
                    .unwrap_or(u64::MAX);
                let tie = mix64(seed ^ mix64(left as u64) ^ mix64(right as u64));
                (relatedness, tie, left, right)
            })
        })
        .collect();
    pairs.sort_unstable();
    let pair_count = pairs.len();
    let attempt_index = usize::try_from(attempt).unwrap_or(usize::MAX);
    let selected = pairs.get(attempt_index % pair_count.max(1)).map(|&(_, _, left, right)| (left, right));
    let beam_attempt = attempt_index / pair_count.max(1);
    let mut candidates = 0u64;
    let selected_items = selected.map_or(0, |(left, right)| current.lists[left].len().saturating_add(current.lists[right].len()));
    let candidate_budget = u64::try_from(selected_items).unwrap_or(u64::MAX).saturating_pow(2).saturating_mul(64).clamp(100_000, 1_000_000);
    let candidate = selected
        .and_then(|(left, right)| {
            let items: Vec<i32> = current.lists[left].iter().chain(&current.lists[right]).copied().collect();
            let routing = per.routing.as_ref()?;
            let rebuilt = fixed_fleet_beam_lists(
                &items,
                routing,
                &per.metrics,
                2,
                u64::try_from(beam_attempt).unwrap_or(u64::MAX),
                seed ^ mix64(attempt),
                stop,
                &mut candidates,
                candidate_budget,
            )?;
            let mut lists = current.lists.clone();
            lists[left] = rebuilt[0].clone();
            lists[right] = rebuilt[1].clone();
            State::from_lists_interruptible(model, per, lists, stop)
        })
        .filter(|state| {
            let score = full_score_raw(per, state);
            score.violation == 0
                && score < full_score_raw(per, current)
                && state.lists.iter().filter(|route| !route.is_empty()).count()
                    == current.lists.iter().filter(|route| !route.is_empty()).count()
        });
    let status = if candidate.is_some() {
        AlnsBuildStatus::Built
    } else if stop.load(Ordering::Relaxed) {
        AlnsBuildStatus::Interrupted
    } else if candidates >= candidate_budget {
        AlnsBuildStatus::BudgetExhausted
    } else {
        AlnsBuildStatus::Infeasible
    };
    let cpu_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    AlnsBuildRun {
        status,
        candidate,
        generated: candidates,
        evaluated: candidates,
        work_units: candidates.saturating_mul(2),
        structural_work: 0,
        repair_index_work: 0,
        canonical_rebuilds: u64::from(matches!(status, AlnsBuildStatus::Built)),
        cpu_nanos,
        destroy_generated: 0,
        destroy_evaluated: 0,
        destroy_cpu_nanos: 0,
        repair_generated: candidates,
        repair_evaluated: candidates,
        repair_cpu_nanos: cpu_nanos,
        destroy_executed: false,
        repair_executed: true,
        removed: selected_items,
        eliminated_route: false,
    }
}

fn routing_relink_budget(model: &CollectionModel, per: &PerList, state: &State, stop: &AtomicBool) -> Option<AlnsWorkBudget> {
    let exploratory = routing_exploration_budget(model);
    Some(AlnsWorkBudget::new(
        exploratory.generated.max(routing_relink_structural_floor(model, per, state, stop)?),
        exploratory.evaluated.max(1),
    ))
}

/// Regression oracle for models whose cache-aware materialization cost exceeds
/// the capped exploration allowance. Speculative macros must still score, route
/// elimination must complete both canonical rebuilds, and relinking must build
/// one canonical step.
#[doc(hidden)]
pub fn audit_size_safe_routing_compound_budget() -> bool {
    use crate::model::list::{ExprArena, ObjectiveTier};

    const ITEMS: usize = 3_000;
    const LISTS: usize = 3_000;
    const TERMS_PER_LIST: usize = 3;

    let mut arena = ExprArena::default();
    let body = arena.constant(1);
    let terms = (0..LISTS)
        .flat_map(|list| {
            std::iter::repeat_with({
                let arena = arena.clone();
                move || Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena: arena.clone(), body, coeff: 1 }
            })
            .take(TERMS_PER_LIST)
        })
        .collect();
    let model = CollectionModel {
        items: (1..=i32::try_from(ITEMS).expect("audit size fits i32")).collect(),
        lists: LISTS,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let per = PerList::build(&model);
    let mut lists = vec![Vec::new(); LISTS];
    for (index, item) in model.items.iter().copied().enumerate() {
        lists[index / 2].push(item);
    }
    let state = State::from_lists(&model, &per, lists);
    let exploratory = routing_exploration_budget(&model);
    let stop = AtomicBool::new(false);
    let compound_floor = routing_compound_structural_floor(&model, &per, &state, &stop).expect("the audit is not interrupted");
    let operators = [MacroOperator::ChainRelocate, MacroOperator::GuidedSegmentExchange, MacroOperator::EjectionChain];
    let capped_runs = operators.map(|operator| build_macro_candidate_bounded(&model, &per, &state, operator, 17, exploratory, &stop));
    let safe_budgets =
        operators.map(|operator| routing_macro_budget(&model, &per, &state, operator, &stop).expect("the audit is not interrupted"));
    let safe_runs: [_; 3] =
        std::array::from_fn(|index| build_macro_candidate_bounded(&model, &per, &state, operators[index], 17, safe_budgets[index], &stop));

    let route_floor = routing_route_elimination_floor(&model, &per, &state, &stop).expect("the audit is not interrupted");
    let capped_route = build_macro_candidate_bounded(&model, &per, &state, MacroOperator::RouteElimination, 17, exploratory, &stop);
    let safe_route_budget =
        routing_macro_budget(&model, &per, &state, MacroOperator::RouteElimination, &stop).expect("the audit is not interrupted");
    let safe_route = build_macro_candidate_bounded(&model, &per, &state, MacroOperator::RouteElimination, 17, safe_route_budget, &stop);

    let mut target_lists = state.lists.clone();
    target_lists[0].swap(0, 1);
    let target_state = State::from_lists(&model, &per, target_lists);
    let mut elite = ElitePool::new(6, false, false);
    let target_score = full_score_raw(&per, &target_state);
    let archive = elite.consider_bounded(&target_state, &target_score, elite_archive_budget(&model), &stop);
    let inserted = archive.status == EliteOperationStatus::Complete && archive.inserted;
    let relink_floor = routing_relink_structural_floor(&model, &per, &state, &stop).expect("the audit is not interrupted");
    let safe_relink_budget = routing_relink_budget(&model, &per, &state, &stop).expect("the audit is not interrupted");
    let selection = elite.select_target_bounded(&state, 17, elite_selection_budget(&model), &stop);
    let relink = selection.target.map(|target| path_relink_bounded(&model, &per, &state, target, safe_relink_budget, &stop));

    compound_floor > exploratory.generated
        && capped_runs.iter().all(|run| run.status == AlnsBuildStatus::BudgetExhausted && run.evaluated == 0 && run.canonical_rebuilds == 0)
        && safe_runs.iter().all(|run| {
            run.status == AlnsBuildStatus::BudgetExhausted
                && run.evaluated > 0
                && run.canonical_rebuilds == 0
                && run.generated <= safe_budgets[0].generated
        })
        && route_floor.generated > exploratory.generated
        && capped_route.status == AlnsBuildStatus::BudgetExhausted
        && capped_route.evaluated == 0
        && capped_route.canonical_rebuilds < 2
        && safe_route_budget.generated >= route_floor.generated
        && safe_route_budget.evaluated >= route_floor.evaluated
        && safe_route.status == AlnsBuildStatus::Built
        && safe_route.evaluated > 0
        && safe_route.canonical_rebuilds == 2
        && safe_route.candidate.is_some()
        && inserted
        && relink_floor > exploratory.generated
        && safe_relink_budget.generated == relink_floor
        && relink.is_some_and(|run| {
            run.evaluated == 1
                && run.steps == 1
                && run.structural_work > 0
                && run.structural_work < run.generated
                && run.canonical_rebuilds == 1
        })
}

fn observe_routing_checkpoint(
    control: &mut RoutingSearchControl,
    started: Option<&Instant>,
    per: &PerList,
    best_lists: &[Vec<i32>],
    best_score: &Score,
    best_feasible: bool,
) {
    if !control.checkpoints_enabled() {
        return;
    }
    let Some(started) = started else { return };
    control.observe_checkpoint(
        started.elapsed(),
        best_feasible,
        &objective_values(per, best_score),
        best_lists.iter().filter(|route| !route.is_empty()).count(),
    );
}

fn reset_routing_scans(memories: &mut [RoutingScanMemory]) {
    for memory in memories {
        memory.reset();
    }
}

fn invalidate_routing_scans(
    index_cache: &mut RoutingIndexCache,
    granular_scans: &mut [RoutingScanMemory],
    global_scans: &mut [RoutingScanMemory],
) {
    index_cache.reset();
    reset_routing_scans(granular_scans);
    reset_routing_scans(global_scans);
}

fn consider_elite_candidate(
    model: &CollectionModel,
    elite: &mut ElitePool,
    control: &mut RoutingSearchControl,
    state: &State,
    score: &Score,
    stop: &AtomicBool,
) -> EliteOperationStatus {
    let run = elite.consider_bounded(state, score, elite_archive_budget(model), stop);
    control.record_elite_archive(run.status, run.inserted, run.work_units, run.cpu_nanos);
    run.status
}

#[allow(clippy::too_many_arguments)]
fn run_routing_search(
    model: &CollectionModel,
    per: &PerList,
    state: &mut State,
    stop: &AtomicBool,
    max_iters: u64,
    seed: u64,
    best_lists: &mut Vec<Vec<i32>>,
    best_score: &mut Score,
    best_feasible: &mut bool,
    report: &mut dyn FnMut(i64),
    alns: &mut AlnsController,
    coordination: &mut Option<WorkerCoordination<'_>>,
    metrics_enabled: bool,
    started: Option<&Instant>,
    construction_offset: Duration,
    construction_elapsed: Duration,
    construction_history: &[InitialFeasible],
    construction_candidates: u64,
) -> bool {
    let items = model.items.len();
    let mut control = RoutingSearchControl::new(metrics_enabled, construction_candidates);
    if control.checkpoints_enabled() {
        for incumbent in construction_history {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            control.observe_checkpoint_with_candidates(
                construction_offset.saturating_add(incumbent.elapsed),
                true,
                &objective_values(per, &incumbent.score),
                incumbent.fleet,
                incumbent.candidates,
            );
        }
        control.observe_checkpoint_with_candidates(
            construction_offset.saturating_add(construction_elapsed),
            *best_feasible,
            &objective_values(per, best_score),
            best_lists.iter().filter(|route| !route.is_empty()).count(),
            construction_candidates,
        );
    }
    let mut elite =
        ElitePool::new(8, !per.interchangeable_lists.get(), per.routing.as_ref().is_some_and(|routing| routing.reverse_equivalent));
    let initial_state_score = full_score_raw(per, state);
    if initial_state_score.violation == 0 {
        let _ = consider_elite_candidate(model, &mut elite, &mut control, state, &initial_state_score, stop);
    }
    let mut granular_scans: [RoutingScanMemory; 6] = std::array::from_fn(|_| RoutingScanMemory::new());
    let mut global_scans: [RoutingScanMemory; 6] = std::array::from_fn(|_| RoutingScanMemory::new());
    let mut routing_index = RoutingIndexCache::new();
    let mut stagnant = 0u64;
    let mut slice = 0u64;
    let mut state_consistent = true;
    let mut best_fleet = best_lists.iter().filter(|route| !route.is_empty()).count();
    let mut fleet_stagnant = 0u64;
    let mut fleet_rebuild_attempts = 0u64;
    let mut fleet_beam_refinement = (items < 200 || best_fleet <= items.max(1).isqrt()).then_some(best_fleet);
    let fleet_window = u64::try_from(items.saturating_mul(2).clamp(64, 512)).unwrap_or(512);
    let stable_warmup = u64::try_from(items.saturating_mul(8).clamp(64, 2_048)).unwrap_or(2_048);

    'routing: while !stop.load(Ordering::Relaxed) && slice < max_iters {
        slice = slice.saturating_add(1);
        observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);

        if let Some(lists) = coordination.as_mut().and_then(|shared| shared.maybe_inject(slice, best_score, *best_feasible)) {
            let Some(injected) = State::from_lists_interruptible(model, per, lists, stop) else {
                state_consistent = false;
                break;
            };
            *state = injected;
            observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
            let improved = record_state(per, state, best_lists, best_score, best_feasible, report);
            debug_assert!(improved, "a shared incumbent must strictly improve the worker incumbent");
            alns.record_shared_injection();
            alns.reset_after_injection(best_score);
            invalidate_routing_scans(&mut routing_index, &mut granular_scans, &mut global_scans);
            if *best_feasible {
                let score = full_score_raw(per, state);
                if consider_elite_candidate(model, &mut elite, &mut control, state, &score, stop) == EliteOperationStatus::Interrupted {
                    break 'routing;
                }
            }
            observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
            stagnant = 0;
        }

        let observed_fleet = best_lists.iter().filter(|route| !route.is_empty()).count();
        if observed_fleet < best_fleet {
            best_fleet = observed_fleet;
            fleet_stagnant = 0;
            fleet_rebuild_attempts = 0;
            fleet_beam_refinement = (items < 200 || best_fleet <= items.max(1).isqrt()).then_some(best_fleet);
            if per.reset_routing_gls() {
                state.refresh_edge_penalties(per);
                invalidate_routing_scans(&mut routing_index, &mut granular_scans, &mut global_scans);
            }
        } else {
            fleet_stagnant = fleet_stagnant.saturating_add(1);
        }
        let fleet_focus =
            per.minimizes_fleet() && best_fleet > 1 && fleet_stagnant >= fleet_window && fleet_stagnant.is_multiple_of(fleet_window);
        if fleet_focus {
            alns.reset_after_injection(&full_score_raw(per, state));
            reset_routing_scans(&mut granular_scans);
            reset_routing_scans(&mut global_scans);
        }
        let slice_kind = if fleet_focus { SliceKind::Macro } else { control.next_slice(stagnant, elite.len() > 1) };
        if stagnant > 0 && slice.is_multiple_of(16) {
            let penalties = per.bump_gls(state);
            state.refresh_edge_penalties(per);
            alns.record_gls(penalties);
            if penalties > 0 {
                reset_routing_scans(&mut granular_scans);
                reset_routing_scans(&mut global_scans);
            }
        }
        // Start from a bounded deterministic intensification trajectory, then
        // hand tie-breaking back to the caller seed. This gives every run a
        // reproducible baseline while preserving long-run diversity.
        let trajectory_seed = if slice <= stable_warmup { 0 } else { seed };
        let operator_seed = trajectory_seed ^ mix64(slice) ^ mix64(stagnant);

        match slice_kind {
            SliceKind::Descent | SliceKind::Global => {
                let current_raw = full_score_raw(per, state);
                let infeasible = current_raw.violation != 0;
                let kind = if infeasible {
                    [NeighborhoodKind::Relocate, NeighborhoodKind::Swap, NeighborhoodKind::OrOpt][(mix64(operator_seed) % 3) as usize]
                } else {
                    control.choose_neighborhood(operator_seed)
                };
                let (mode, budget, memory) = if matches!(slice_kind, SliceKind::Global) || infeasible {
                    (ScanMode::Global, routing_global_budget(items), &mut global_scans[kind.index()])
                } else {
                    (ScanMode::Granular, routing_move_budget(items), &mut granular_scans[kind.index()])
                };
                let run = search_routing_neighborhood(
                    per,
                    state,
                    stop,
                    RoutingScanWorkspace::new(&mut routing_index, memory),
                    kind,
                    mode,
                    budget,
                );
                let mut accepted_move = false;
                let mut improved_current = false;
                let mut global_best = false;
                match run.outcome {
                    ScanOutcome::Improved(mv) => {
                        if !apply_move(per, state, mv, stop) {
                            if matches!(slice_kind, SliceKind::Global) {
                                control.record_global(kind, run, false, false, false);
                            } else {
                                control.record_descent(kind, run, false, false, false);
                            }
                            observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                            state_consistent = false;
                            break;
                        }
                        accepted_move = true;
                        invalidate_routing_scans(&mut routing_index, &mut granular_scans, &mut global_scans);
                        let state_score = full_score_raw(per, state);
                        improved_current = state_score < current_raw;
                        observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                        global_best = record_state(per, state, best_lists, best_score, best_feasible, report);
                        if global_best && coordination.as_ref().is_some_and(|shared| shared.publish(best_lists, best_score, *best_feasible))
                        {
                            alns.record_shared_publication();
                        }
                        if state_score.violation == 0
                            && consider_elite_candidate(model, &mut elite, &mut control, state, &state_score, stop)
                                == EliteOperationStatus::Interrupted
                        {
                            if matches!(slice_kind, SliceKind::Global) {
                                control.record_global(kind, run, improved_current, global_best, accepted_move);
                            } else {
                                control.record_descent(kind, run, improved_current, global_best, accepted_move);
                            }
                            observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                            break 'routing;
                        }
                    }
                    ScanOutcome::Interrupted => {
                        if matches!(slice_kind, SliceKind::Global) {
                            control.record_global(kind, run, false, false, false);
                        } else {
                            control.record_descent(kind, run, false, false, false);
                        }
                        observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                        break;
                    }
                    ScanOutcome::Complete | ScanOutcome::BudgetExhausted => {}
                }
                if matches!(slice_kind, SliceKind::Global) {
                    control.record_global(kind, run, improved_current, global_best, accepted_move);
                } else {
                    control.record_descent(kind, run, improved_current, global_best, accepted_move);
                }
                observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                stagnant = if global_best { 0 } else { stagnant.saturating_add(1) };
            }
            SliceKind::Alns => {
                let current_guided = full_score(per, state);
                let current_raw = full_score_raw(per, state);
                let choice = alns.choose(items, stagnant, operator_seed, slice);
                let mut run = build_candidate_bounded(model, per, state, choice, operator_seed, routing_alns_budget(items), stop);
                observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                control.record_alns(run.evaluated);
                if run.status == AlnsBuildStatus::Interrupted {
                    alns.record_failed_bounded(choice, &run);
                    observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                    break;
                }
                let Some(candidate) = run.candidate.take() else {
                    alns.record_failed_bounded(choice, &run);
                    observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                    stagnant = stagnant.saturating_add(1);
                    continue;
                };
                let candidate_guided = full_score(per, &candidate);
                let candidate_raw = full_score_raw(per, &candidate);
                let improved_current = candidate_raw < current_raw;
                let global_best = record_state(per, &candidate, best_lists, best_score, best_feasible, report);
                if global_best && coordination.as_ref().is_some_and(|shared| shared.publish(best_lists, best_score, *best_feasible)) {
                    alns.record_shared_publication();
                }
                let worsens_fleet = per.minimizes_fleet()
                    && current_raw.violation == 0
                    && candidate_raw.violation == 0
                    && candidate.lists.iter().filter(|route| !route.is_empty()).count()
                        > state.lists.iter().filter(|route| !route.is_empty()).count();
                let acceptance = if worsens_fleet {
                    alns.reject_semantic_worsening(&current_raw)
                } else {
                    alns.accept(&current_guided, &candidate_guided, &current_raw, &candidate_raw, operator_seed, slice)
                };
                let archive_interrupted = candidate_raw.violation == 0
                    && consider_elite_candidate(model, &mut elite, &mut control, &candidate, &candidate_raw, stop)
                        == EliteOperationStatus::Interrupted;
                if !matches!(acceptance, AcceptanceKind::Rejected) {
                    *state = candidate;
                    invalidate_routing_scans(&mut routing_index, &mut granular_scans, &mut global_scans);
                }
                alns.record_bounded(choice, acceptance, improved_current, global_best, &run);
                observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                if archive_interrupted {
                    break 'routing;
                }
                stagnant = if global_best { 0 } else { stagnant.saturating_add(1) };
            }
            SliceKind::Macro => {
                let operator = if fleet_focus { MacroOperator::RouteElimination } else { control.choose_macro(operator_seed) };
                let smallest_route = state.lists.iter().filter(|route| !route.is_empty()).map(Vec::len).min().unwrap_or(0);
                let root = items.max(1).isqrt();
                let dense_fleet = best_fleet > 4 && smallest_route > root;
                let pair_count = best_fleet.saturating_mul(best_fleet.saturating_sub(1)).saturating_div(2);
                let pair_refinement_limit = pair_count.min(best_fleet.saturating_mul(2).clamp(6, 32)) as u64;
                let run = if fleet_focus && dense_fleet && fleet_rebuild_attempts < 6 {
                    let attempt = fleet_rebuild_attempts;
                    fleet_rebuild_attempts = fleet_rebuild_attempts.saturating_add(1);
                    let run = fixed_fleet_rebuild(model, per, state, attempt, operator_seed, stop);
                    if let Some(candidate) = &run.candidate {
                        fleet_beam_refinement = Some(candidate.lists.iter().filter(|route| !route.is_empty()).count());
                    }
                    run
                } else if fleet_focus && fleet_beam_refinement == Some(best_fleet) && fleet_rebuild_attempts < pair_refinement_limit {
                    let attempt = fleet_rebuild_attempts;
                    fleet_rebuild_attempts = fleet_rebuild_attempts.saturating_add(1);
                    fixed_pair_rebuild(model, per, state, attempt, operator_seed, stop)
                } else {
                    let Some(budget) = routing_macro_budget(model, per, state, operator, stop) else {
                        observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                        break;
                    };
                    build_macro_candidate_bounded(model, per, state, operator, operator_seed, budget, stop)
                };
                observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                if run.status == AlnsBuildStatus::Interrupted {
                    control.record_macro(operator, run.status, run.generated, run.evaluated, run.cpu_nanos, false, false, false);
                    observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                    break;
                }
                let mut improved_current = false;
                let mut global_best = false;
                let mut archive_interrupted = false;
                let accepted = if let Some(candidate) = run.candidate {
                    let candidate_score = full_score_raw(per, &candidate);
                    improved_current = candidate_score < full_score_raw(per, state);
                    global_best = record_state(per, &candidate, best_lists, best_score, best_feasible, report);
                    if global_best && coordination.as_ref().is_some_and(|shared| shared.publish(best_lists, best_score, *best_feasible)) {
                        alns.record_shared_publication();
                    }
                    if candidate_score.violation == 0 {
                        archive_interrupted = consider_elite_candidate(model, &mut elite, &mut control, &candidate, &candidate_score, stop)
                            == EliteOperationStatus::Interrupted;
                    }
                    *state = candidate;
                    invalidate_routing_scans(&mut routing_index, &mut granular_scans, &mut global_scans);
                    true
                } else {
                    false
                };
                control.record_macro(
                    operator,
                    run.status,
                    run.generated,
                    run.evaluated,
                    run.cpu_nanos,
                    improved_current,
                    global_best,
                    accepted,
                );
                observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                if archive_interrupted {
                    break 'routing;
                }
                stagnant = if global_best { 0 } else { stagnant.saturating_add(1) };
            }
            SliceKind::Relink => {
                let selection = elite.select_target_bounded(state, operator_seed, elite_selection_budget(model), stop);
                control.record_elite_selection(selection.status, selection.target.is_some(), selection.work_units, selection.cpu_nanos);
                if selection.status == EliteOperationStatus::Interrupted {
                    control.record_relink_slice_without_target();
                    observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                    break 'routing;
                }
                let Some(target) = selection.target else {
                    control.record_relink_slice_without_target();
                    observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                    stagnant = stagnant.saturating_add(1);
                    continue;
                };
                let Some(budget) = routing_relink_budget(model, per, state, stop) else {
                    control.record_relink_slice_without_target();
                    observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                    break;
                };
                let run = path_relink_bounded(model, per, state, target, budget, stop);
                observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                if run.status == PathRelinkStatus::Interrupted {
                    control.record_relink(
                        run.status,
                        run.steps,
                        run.work_units,
                        run.generated,
                        run.evaluated,
                        run.cpu_nanos,
                        false,
                        false,
                        false,
                    );
                    observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                    break;
                }
                let mut improved_current = false;
                let mut global_best = false;
                let mut archive_interrupted = false;
                let accepted = if let Some(candidate) = run.candidate {
                    let candidate_score = full_score_raw(per, &candidate);
                    improved_current = candidate_score < full_score_raw(per, state);
                    global_best = record_state(per, &candidate, best_lists, best_score, best_feasible, report);
                    if global_best && coordination.as_ref().is_some_and(|shared| shared.publish(best_lists, best_score, *best_feasible)) {
                        alns.record_shared_publication();
                    }
                    if candidate_score.violation == 0 {
                        archive_interrupted = consider_elite_candidate(model, &mut elite, &mut control, &candidate, &candidate_score, stop)
                            == EliteOperationStatus::Interrupted;
                    }
                    *state = candidate;
                    invalidate_routing_scans(&mut routing_index, &mut granular_scans, &mut global_scans);
                    true
                } else {
                    false
                };
                control.record_relink(
                    run.status,
                    run.steps,
                    run.work_units,
                    run.generated,
                    run.evaluated,
                    run.cpu_nanos,
                    improved_current,
                    global_best,
                    accepted,
                );
                observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
                if archive_interrupted {
                    break 'routing;
                }
                stagnant = if global_best { 0 } else { stagnant.saturating_add(1) };
            }
        }
    }

    observe_routing_checkpoint(&mut control, started, per, best_lists, best_score, *best_feasible);
    per.metrics.record_routing(control.finish());
    state_consistent
}

/// Solve a collection model with constraint-based local search until `stop`.
/// `report` is called with the objective each time a strictly better *feasible*
/// incumbent is found, for progress output; pass `&mut |_| {}` to ignore it.
pub fn solve_collection(model: &CollectionModel, seed: u64, stop: &AtomicBool, report: &mut dyn FnMut(i64)) -> CollectionSolution {
    solve_collection_capped(model, seed, stop, u64::MAX, None, report)
}

/// Run [`solve_collection`] with profiling enabled and return the metrics
/// without printing them. This is intended for benchmark harnesses.
pub fn solve_collection_profiled(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ListSearchMetrics) {
    solve_collection_capped_internal(model, seed, stop, u64::MAX, None, report, true, true)
}

/// Like [`solve_collection`], but seeds the initial incumbent from `hint` -- one
/// visiting-order sequence per list variable, from a caller's constructive
/// heuristic -- instead of the greedy random partition. Universe items the hint
/// omits are placed in the last list (the pool on an optional model, so unhinted
/// nodes stay droppable). The hint seeds the initial incumbent; adaptive
/// destroy/repair passes diversify from it, and the best incumbent (possibly the
/// hint itself) is always retained.
pub fn solve_collection_hinted(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    hint: &[Vec<i32>],
    report: &mut dyn FnMut(i64),
) -> CollectionSolution {
    solve_collection_capped(model, seed, stop, u64::MAX, Some(hint), report)
}

/// Frontend entry point after the model has already passed budget-aware
/// validation. Avoids validating the same large model again inside the search
/// worker while preserving validation on every public engine entry point.
pub(crate) fn solve_collection_validated(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    max_iters: u64,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
    profile: bool,
) -> (CollectionSolution, ListSearchMetrics) {
    solve_collection_capped_worker(model, seed, stop, max_iters, hint, report, profile, false, SearchProfile::Sequential, None)
}

/// Complete a warm-start hint into a full `k`-list partition for [`State::from_lists`]:
/// copy each provided sequence (defensively dropping duplicates and any value
/// outside the universe), then place every universe item the hint omits in the
/// last list -- the unreferenced pool on an optional model (so unhinted nodes stay
/// droppable), or the last route otherwise. Guarantees a complete partition so no
/// item is silently left unassigned.
fn hint_partition(model: &CollectionModel, hint: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let k = model.lists.max(1);
    let universe: HashSet<i32> = model.items.iter().copied().collect();
    let mut placed: HashSet<i32> = HashSet::with_capacity(model.items.len());
    let mut lists: Vec<Vec<i32>> = vec![Vec::new(); k];
    for (l, seq) in hint.iter().take(k).enumerate() {
        for &value in seq {
            if universe.contains(&value) && placed.insert(value) {
                lists[l].push(value);
            }
        }
    }
    for &item in &model.items {
        if !placed.contains(&item) {
            lists[k - 1].push(item);
        }
    }
    lists
}

/// Like [`solve_collection`], but stops after at most `max_iters` local-search
/// iterations as well as when `stop` is set. Used to get a quick bounded
/// incumbent (e.g. to warm-start the exact routing backend) even when no time
/// limit / stop flag is in effect.
pub fn solve_collection_capped(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    max_iters: u64,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
) -> CollectionSolution {
    solve_collection_capped_internal(model, seed, stop, max_iters, hint, report, false, true).0
}

/// Deterministic iteration-capped variant of [`solve_collection_profiled`].
/// It always collects metrics and leaves presentation to the caller.
pub fn solve_collection_capped_profiled(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    max_iters: u64,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ListSearchMetrics) {
    solve_collection_capped_internal(model, seed, stop, max_iters, hint, report, true, true)
}

#[allow(clippy::too_many_arguments)]
fn solve_collection_capped_internal(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    max_iters: u64,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
    metrics_enabled: bool,
    validate_model: bool,
) -> (CollectionSolution, ListSearchMetrics) {
    let (mut solution, metrics) = solve_collection_capped_worker(
        model,
        seed,
        stop,
        max_iters,
        hint,
        report,
        metrics_enabled,
        validate_model,
        SearchProfile::Sequential,
        None,
    );
    // Construction owns the time-to-first-incumbent contract. Structural
    // bounds are useful after that point, but must never consume the solve
    // budget before a cheap feasible fallback is published.
    let dual_bound = if !stop.load(Ordering::Relaxed) { dual::compute(model, stop) } else { None };
    dual::attach(model, &mut solution, dual_bound);
    (solution, metrics)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve_collection_capped_worker(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    max_iters: u64,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
    metrics_enabled: bool,
    validate_model: bool,
    profile: SearchProfile,
    mut coordination: Option<WorkerCoordination<'_>>,
) -> (CollectionSolution, ListSearchMetrics) {
    let started = metrics_enabled.then(Instant::now);
    let no_solution = || {
        let solution = CollectionSolution {
            lists: vec![Vec::new(); model.lists.max(1)],
            objectives: Vec::new(),
            feasible: false,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
            modes: Vec::new(),
            bound: None,
        };
        let metrics = MetricsRecorder::new(metrics_enabled).snapshot(started.map(|instant| instant.elapsed()).unwrap_or_default());
        (solution, metrics)
    };
    if stop.load(Ordering::Relaxed) {
        return no_solution();
    }
    // Guard the search path: an invalid model would otherwise panic (bad list
    // index) or read silent zeros (out-of-range table index). Callers like the
    // Python frontend validate first to raise a precise error; this is the
    // backstop for direct Rust callers so the engine never panics or corrupts.
    if validate_model {
        match model.validate_interruptible(stop) {
            Ok(true) => {}
            Ok(false) => return no_solution(),
            Err(_e) => {
                debug_assert!(false, "solve_collection called on an invalid model: {_e}");
                return no_solution();
            }
        }
    }
    // Scheduling has its own physical engine and is dispatched directly by the
    // orchestrator. A list-search worker never launches that engine.
    if model.schedule.is_some() {
        return no_solution();
    }
    let per = PerList::build_profiled(
        model,
        metrics_enabled,
        profile.candidate_limit(),
        profile.use_arc_gls(),
        profile.diversify_initial_descent(),
        stop,
    );
    if stop.load(Ordering::Relaxed) {
        return no_solution();
    }
    let n = model.items.len();
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(&mut order, seed);
    // Warm start from the caller's hint when given (completed to a full
    // partition); otherwise the greedy random construction. Either way the search
    // loop, ALNS controller, and incumbent tracking below are identical.
    let construction_started = Instant::now();
    let construction_offset = started.as_ref().map_or(Duration::ZERO, Instant::elapsed);
    let construction = match hint {
        Some(h) => State::from_lists_interruptible(model, &per, hint_partition(model, h), stop).map(|state| {
            let score = full_score_raw(&per, &state);
            let elapsed = construction_started.elapsed();
            let feasible_history = (score.violation == 0).then(|| initial_feasible(&state, score, elapsed, 0)).into_iter().collect();
            InitialConstruction { state, incumbent: None, name: "warm-start", elapsed, feasible_history, candidates: 0, reported: false }
        }),
        None => {
            let primary = routing_construction(model, &per, &order, seed, stop, report);
            let construction = if seed != 0 && primary.is_some() && per.minimizes_fleet() && !stop.load(Ordering::Relaxed) {
                let mut stable_order: Vec<usize> = (0..n).collect();
                shuffle(&mut stable_order, 0);
                let mut silent = |_| {};
                let stable = routing_construction(model, &per, &stable_order, 0, stop, &mut silent);
                match (primary, stable) {
                    (Some(primary), Some(stable)) => Some(combine_routing_constructions(&per, primary, stable)),
                    (primary, _) => primary,
                }
            } else {
                primary
            };
            construction.or_else(|| {
                State::greedy(model, &per, &order, seed, profile.diversify_initial_descent(), stop).map(|state| {
                    let score = full_score_raw(&per, &state);
                    let elapsed = construction_started.elapsed();
                    let feasible_history =
                        (score.violation == 0).then(|| initial_feasible(&state, score, elapsed, 0)).into_iter().collect();
                    InitialConstruction {
                        state,
                        incumbent: None,
                        name: "generic-greedy",
                        elapsed,
                        feasible_history,
                        candidates: 0,
                        reported: false,
                    }
                })
            })
        }
    };
    let Some(construction) = construction else {
        return no_solution();
    };
    let InitialConstruction {
        mut state,
        incumbent,
        name: constructor_name,
        elapsed: construction_elapsed,
        feasible_history,
        candidates: construction_candidates,
        reported: construction_reported,
    } = construction;
    let construction_incumbent = incumbent.as_ref().unwrap_or(&state);
    let construction_score = full_score_raw(&per, construction_incumbent);
    let construction_objectives = objective_values(&per, &construction_score);
    let fleet = Some(construction_incumbent.lists.iter().filter(|route| !route.is_empty()).count());
    let constructor_cost = per
        .routing
        .as_ref()
        .and_then(|routing| {
            if routing.has_fleet_objective {
                construction_objectives.last().copied()
            } else {
                construction_objectives.first().copied()
            }
        })
        .or_else(|| construction_objectives.first().copied());
    per.metrics.record_construction(
        constructor_name,
        construction_elapsed,
        feasible_history.first().map(|snapshot| snapshot.elapsed),
        construction_candidates,
        fleet,
        constructor_cost,
    );
    let (mut best_lists, mut best_score, mut best_feasible) = snapshot(&per, construction_incumbent);
    let mut alns = if per.routing.is_some() {
        AlnsController::new_routing_profile(n, &full_score_raw(&per, &state), profile)
    } else {
        AlnsController::new_profile(n, &best_score, profile)
    };
    if coordination.as_ref().is_some_and(|shared| shared.publish(&best_lists, &best_score, best_feasible)) {
        alns.record_shared_publication();
    }
    if best_feasible && per.tiers > 0 && !construction_reported {
        report(tier_value(&per, &best_score, 0));
    }
    let state_consistent = if per.routing.is_some() {
        run_routing_search(
            model,
            &per,
            &mut state,
            stop,
            max_iters,
            seed,
            &mut best_lists,
            &mut best_score,
            &mut best_feasible,
            report,
            &mut alns,
            &mut coordination,
            metrics_enabled,
            started.as_ref(),
            construction_offset,
            construction_elapsed,
            &feasible_history,
            construction_candidates,
        )
    } else {
        let mut memory = SearchMemory::new(model.lists.max(1));
        run_generic_search(
            model,
            &per,
            &mut state,
            stop,
            max_iters,
            seed,
            profile,
            &mut memory,
            &mut best_lists,
            &mut best_score,
            &mut best_feasible,
            report,
            &mut alns,
            &mut coordination,
        )
    };

    if state_consistent
        && record_state(&per, &state, &mut best_lists, &mut best_score, &mut best_feasible, report)
        && coordination.as_ref().is_some_and(|shared| shared.publish(&best_lists, &best_score, best_feasible))
    {
        alns.record_shared_publication();
    }

    // Report objective values from the exact unpenalized incumbent score. GLS
    // only guides moves, so it can never leak into the public result. When
    // infeasible these values are best-effort; `feasible` is the signal to trust.
    let objectives = objective_values(&per, &best_score);
    let solution = CollectionSolution {
        lists: best_lists,
        objectives,
        feasible: best_feasible,
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        modes: Vec::new(),
        bound: None,
    };
    per.metrics.record_alns(alns.metrics());
    let metrics = per.metrics.snapshot(started.map(|instant| instant.elapsed()).unwrap_or_default());
    (solution, metrics)
}
