//! Private list local-search heuristic implementation.
//!
//! This module scores and moves over an already-declared model. It must not be
//! the only place where a new modeling feature exists: add the feature to the
//! shared Rust model and backend classifier first, then teach this heuristic to
//! search it as a fallback or incumbent generator.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use smallvec::SmallVec;

use super::alns::{build_candidate, AcceptanceKind, AlnsController, SearchProfile};
use super::eval::{eval_reduction, violation_of, INFEASIBLE};
use super::incremental::{EvalScratch, InsertView, ListView, ReductionCache};
use super::metrics::{metrics_enabled_from_env, ListSearchMetrics, MetricsRecorder};
use super::moves::{apply_move, best_improving_move, better, shuffle, snapshot, trial_list_score_view, SearchMemory};
use super::portfolio::WorkerCoordination;
use super::schedule_ls::solve_schedule;
use crate::mix64;
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

    /// Total violation over all global constraints.
    pub(super) fn total(&self, item_list: &[usize]) -> i64 {
        self.cons.iter().fold(0i64, |acc, c| acc.saturating_add(self.one(c, item_list, &[])))
    }

    /// Change in total global violation if the listed items moved to new lists.
    pub(super) fn delta(&self, item_list: &[usize], overrides: &[(i32, usize)]) -> i64 {
        let mut affected: Vec<usize> = Vec::new();
        for &(v, _) in overrides {
            if let Some(&i) = self.value_to_idx.get(&v) {
                for &g in &self.of_idx[i] {
                    if !affected.contains(&g) {
                        affected.push(g);
                    }
                }
            }
        }
        let mut d = 0i64;
        for &g in &affected {
            let c = &self.cons[g];
            d = d.saturating_add(self.one(c, item_list, overrides).saturating_sub(self.one(c, item_list, &[])));
        }
        d
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
    /// Whether cost-refiner moves (2-opt* / cross / reverse) may prune by the
    /// geometric candidate lists even while a route still overflows. On by
    /// default (lets an infeasible-heavy instance afford more passes); set
    /// `QAYD_LS_INFEAS_CAND=0` to revert to the conservative feasible-only gate.
    pub(super) infeas_cand: bool,
    penalties: GlsPenalties,
    pub(super) metrics: MetricsRecorder,
}

struct GlsPenalties {
    constraints: RefCell<Vec<Vec<i64>>>,
    objectives: RefCell<Vec<Vec<Vec<i64>>>>,
}

pub(super) struct CandidateNeighbors {
    map: HashMap<i32, Vec<i32>>,
}

impl CandidateNeighbors {
    fn build(model: &CollectionModel, matrix: Arc<Vec<Vec<i64>>>, stop: &AtomicBool) -> Option<Self> {
        const LIMIT: usize = 24;
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
        for &value in &values {
            if !targets.contains(&value) {
                targets.push(value);
            }
        }
        targets.sort_unstable();
        targets.dedup();

        let mut map = HashMap::with_capacity(values.len());
        for &from in &values {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let from_idx = usize::try_from(from).ok()?;
            let mut near = targets
                .iter()
                .copied()
                .filter(|&to| to != from)
                .map(|to| {
                    let to_idx = usize::try_from(to).ok()?;
                    Some((matrix[from_idx][to_idx], to))
                })
                .collect::<Option<Vec<_>>>()?;
            near.sort_unstable_by_key(|&(cost, to)| (cost, to));
            near.truncate(LIMIT.min(near.len()));
            map.insert(from, near.into_iter().map(|(_, to)| to).collect());
        }
        Some(Self { map })
    }

    pub(super) fn contains(&self, a: i32, b: i32) -> bool {
        self.map.get(&a).is_some_and(|near| near.contains(&b)) || self.map.get(&b).is_some_and(|near| near.contains(&a))
    }

    /// The nearest neighbour of `from` (by edge cost) that is currently present and
    /// not already removed -- the relatedness ranking used by Shaw removal.
    pub(super) fn nearest_present(&self, from: i32, removed: &HashSet<i32>, present: &HashSet<i32>) -> Option<i32> {
        self.map.get(&from)?.iter().copied().find(|to| present.contains(to) && !removed.contains(to))
    }
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

impl PerList {
    pub(super) fn build(model: &CollectionModel) -> Self {
        let stop = AtomicBool::new(false);
        Self::build_profiled(model, false, &stop)
    }

    fn build_profiled(model: &CollectionModel, metrics_enabled: bool, stop: &AtomicBool) -> Self {
        let tiers = model.objectives.len();
        let mut objective = vec![vec![Vec::new(); tiers]; model.lists];
        let mut max_objective = vec![Vec::new(); tiers];
        let mut objective_delta = vec![vec![Vec::new(); tiers]; model.lists];
        let mut constraints = vec![Vec::new(); model.lists];
        let mut constraint_delta = vec![Vec::new(); model.lists];
        let mut senses = vec![true; tiers];
        let mut has_edges = false;
        let mut route_bounds = vec![None; model.lists];
        let mut candidate_matrix = None;
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
                    candidate_matrix.get_or_insert_with(|| direct_edge_matrix(r));
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
                            candidate_matrix.get_or_insert_with(|| direct_edge_matrix(r));
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
                candidate_matrix.get_or_insert_with(|| direct_edge_matrix(&c.reduction));
            }
            constraint_delta[list].push(reduction_delta_kind(&c.reduction));
            constraints[list].push(c.clone());
        }
        let penalties = GlsPenalties {
            constraints: RefCell::new(constraints.iter().map(|list| vec![1; list.len()]).collect()),
            objectives: RefCell::new(objective.iter().map(|list| list.iter().map(|tier| vec![1; tier.len()]).collect()).collect()),
        };
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
            candidates: candidate_matrix.flatten().and_then(|matrix| CandidateNeighbors::build(model, matrix, stop)),
            infeas_cand: std::env::var("QAYD_LS_INFEAS_CAND").as_deref() != Ok("0"),
            penalties,
            metrics: MetricsRecorder::new(metrics_enabled),
        }
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

    pub(super) fn score(&self, per: &PerList, idx: usize) -> ListScore {
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
        ListScore { violation, objectives, constraint_violations, objective_reductions, undefined_violation }
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
    ListScore { violation, objectives, constraint_violations, objective_reductions, undefined_violation }
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

fn max_objective_totals<'a>(
    per: &PerList,
    state: &State,
    replacements: &'a [TrialList<'a>],
    scratch: &mut EvalScratch,
) -> (i64, TierValues) {
    let mut violation = 0i64;
    let mut raw = tier_values(per.tiers, 0);
    for (tier, terms) in per.max_objective.iter().enumerate() {
        for (term_idx, term) in terms.iter().enumerate() {
            let mut best = None;
            for (group_idx, group) in term.groups.iter().enumerate() {
                let mut group_value = 0i64;
                let mut defined = true;
                for (reduction_idx, reduction) in group.iter().enumerate() {
                    let cache = &state.max_caches[tier][term_idx][group_idx][reduction_idx];
                    let list = reduction.iterable.list();
                    let value = if let Some(replacement) = replacements.iter().find(|replacement| replacement.list == list) {
                        let value = per.metrics.measure_delta(reduction, || {
                            cache.candidate_value(reduction, &state.lists[list], replacement.contents, scratch)
                        });
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
    (violation, raw)
}

fn score_with_replacements_mode<'a>(
    per: &PerList,
    state: &'a State,
    replacements: &'a [TrialList<'a>],
    global_delta: i64,
    scratch: &mut EvalScratch,
    guided: bool,
) -> Score {
    // Fold in list order, just like a full evaluation. Subtracting an old
    // contribution from a saturated total and adding the replacement is not
    // reversible at i64::MIN/MAX.
    let mut violation = 0i64;
    let mut raw = tier_values(per.tiers, 0);
    let constraint_weights = guided.then(|| per.penalties.constraints.borrow());
    let objective_weights = guided.then(|| per.penalties.objectives.borrow());
    for (list, cached) in state.scores.iter().enumerate() {
        let score = replacements.iter().find(|replacement| replacement.list == list).map_or(cached, |replacement| replacement.score);
        if let Some(weights) = &constraint_weights {
            let weighted = score
                .constraint_violations
                .iter()
                .zip(&weights[list])
                .fold(score.undefined_violation, |sum, (&value, &weight)| sum.saturating_add(value.saturating_mul(weight)));
            violation = violation.saturating_add(weighted);
        } else {
            violation = violation.saturating_add(score.violation);
        }
        for (tier, (slot, &value)) in raw.iter_mut().zip(score.objectives.iter()).enumerate() {
            *slot = slot.saturating_add(value);
            if let Some(weights) = &objective_weights {
                for (&reduction_value, &weight) in score.objective_reductions[tier].iter().zip(&weights[list][tier]) {
                    if let Some(reduction_value) = reduction_value {
                        *slot = slot.saturating_add(reduction_value.saturating_mul(weight.saturating_sub(1)));
                    }
                }
            }
        }
    }
    violation = violation.saturating_add(state.global_viol).saturating_add(global_delta);
    if per.has_max_objective() {
        let (max_violation, max_raw) = max_objective_totals(per, state, replacements, scratch);
        violation = violation.saturating_add(max_violation);
        for (slot, value) in raw.iter_mut().zip(max_raw) {
            *slot = slot.saturating_add(value);
        }
    }
    signed(per, violation, raw)
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

/// Full score including the cross-list global violation.
pub(super) fn full_score(per: &PerList, state: &State) -> Score {
    score_with_replacements(per, state, &[], 0, &mut EvalScratch::default())
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
    fn greedy(model: &CollectionModel, per: &PerList, order: &[usize], stop: &AtomicBool) -> Option<Self> {
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
            for l in 0..k {
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
                if key < best_key {
                    best_key = key;
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
        let scores: Vec<ListScore> = (0..k).map(|idx| caches[idx].score(per, idx)).collect();
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
        let scores = (0..k).map(|idx| caches[idx].score(per, idx)).collect();
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
        let score = cache.score(per, idx);
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

/// Flatten a lexicographic [`Score`] into a single comparable cost for regret
/// arithmetic. Violation dominates the primary objective tier.
pub(super) fn score_scalar(score: &Score) -> i128 {
    (score.violation as i128) * (1i128 << 50) + i128::from(score.tiers.first().copied().unwrap_or(0))
}

/// Solve a collection model with constraint-based local search until `stop`.
/// `report` is called with the objective each time a strictly better *feasible*
/// incumbent is found, for progress output; pass `&mut |_| {}` to ignore it.
pub fn solve_collection(model: &CollectionModel, seed: u64, stop: &AtomicBool, report: &mut dyn FnMut(i64)) -> CollectionSolution {
    // `QAYD_LS_MAX_ITERS` caps the local-search iterations for *deterministic,
    // machine-load-independent* benchmarking: with a large time limit the
    // iteration count (not the wall clock) becomes the binding budget, so the
    // exact same trajectory runs every time regardless of host speed. Unset in
    // normal use (the wall-clock stop flag is the only budget).
    let max_iters = std::env::var("QAYD_LS_MAX_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);
    solve_collection_capped(model, seed, stop, max_iters, None, report)
}

/// Run [`solve_collection`] with profiling enabled and return the metrics
/// without printing them. This is intended for benchmark harnesses.
pub fn solve_collection_profiled(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ListSearchMetrics) {
    let max_iters = std::env::var("QAYD_LS_MAX_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);
    solve_collection_capped_internal(model, seed, stop, max_iters, None, report, true, true)
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
    let max_iters = std::env::var("QAYD_LS_MAX_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);
    solve_collection_capped(model, seed, stop, max_iters, Some(hint), report)
}

/// Frontend entry point after the model has already passed budget-aware
/// validation. Avoids validating the same large model again inside the search
/// worker while preserving validation on every public engine entry point.
#[cfg(feature = "python")]
pub(crate) fn solve_collection_validated(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
) -> CollectionSolution {
    let max_iters = std::env::var("QAYD_LS_MAX_ITERS").ok().and_then(|value| value.parse().ok()).unwrap_or(u64::MAX);
    let metrics_enabled = metrics_enabled_from_env();
    let (solution, metrics) = solve_collection_capped_internal(model, seed, stop, max_iters, hint, report, metrics_enabled, false);
    if metrics_enabled {
        eprint!("{metrics}");
    }
    solution
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
    let metrics_enabled = metrics_enabled_from_env();
    let (solution, metrics) = solve_collection_capped_internal(model, seed, stop, max_iters, hint, report, metrics_enabled, true);
    if metrics_enabled {
        eprint!("{metrics}");
    }
    solution
}

/// Deterministic iteration-capped variant of [`solve_collection_profiled`].
/// It always collects metrics, independently of `QAYD_LS_METRICS`, and leaves
/// presentation to the caller.
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
    solve_collection_capped_worker(
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
    )
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
    if let Some(sched) = &model.schedule {
        let solution = solve_schedule(sched, seed, stop, report);
        let metrics = MetricsRecorder::new(metrics_enabled).snapshot(started.map(|instant| instant.elapsed()).unwrap_or_default());
        return (solution, metrics);
    }
    let per = PerList::build_profiled(model, metrics_enabled, stop);
    if stop.load(Ordering::Relaxed) {
        return no_solution();
    }
    let n = model.items.len();
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(&mut order, seed);
    // Warm start from the caller's hint when given (completed to a full
    // partition); otherwise the greedy random construction. Either way the search
    // loop, ALNS controller, and incumbent tracking below are identical.
    let state = match hint {
        Some(h) => State::from_lists_interruptible(model, &per, hint_partition(model, h), stop),
        None => State::greedy(model, &per, &order, stop),
    };
    let Some(mut state) = state.filter(|_| !stop.load(Ordering::Relaxed)) else {
        return no_solution();
    };
    let mut memory = SearchMemory::new(model.lists.max(1));

    let (mut best_lists, mut best_score, mut best_feasible) = snapshot(&per, &state);
    let mut alns = AlnsController::new_profile(n, &best_score, profile);
    if coordination.as_ref().is_some_and(|shared| shared.publish(&best_lists, &best_score, best_feasible)) {
        alns.record_shared_publication();
    }
    if best_feasible && per.tiers > 0 {
        report(tier_value(&per, &best_score, 0));
    }
    let mut stagnant = 0u64;
    let mut iter = 0u64;
    let mut state_consistent = true;
    let (mut local_optima, mut alns_candidates, mut alns_accepted) = (0u64, 0u64, 0u64);

    while !stop.load(Ordering::Relaxed) && iter < max_iters {
        iter += 1;
        match best_improving_move(&per, &state, stop, &mut memory) {
            Some(mv) => {
                if !apply_move(&per, &mut state, mv, stop) {
                    state_consistent = false;
                    break;
                }
                memory.reset_touched(mv);
                if record_state(&per, &state, &mut best_lists, &mut best_score, &mut best_feasible, report) {
                    if coordination.as_ref().is_some_and(|shared| shared.publish(&best_lists, &best_score, best_feasible)) {
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
                if record_state(&per, &state, &mut best_lists, &mut best_score, &mut best_feasible, report) {
                    if coordination.as_ref().is_some_and(|shared| shared.publish(&best_lists, &best_score, best_feasible)) {
                        alns.record_shared_publication();
                    }
                    stagnant = 0;
                } else {
                    stagnant = stagnant.saturating_add(1);
                }

                if let Some(lists) = coordination.as_mut().and_then(|shared| shared.maybe_inject(local_optima, &best_score, best_feasible))
                {
                    let Some(injected_state) = State::from_lists_interruptible(model, &per, lists, stop) else {
                        break;
                    };
                    state = injected_state;
                    let injected = record_state(&per, &state, &mut best_lists, &mut best_score, &mut best_feasible, report);
                    debug_assert!(injected, "a shared incumbent must strictly improve the worker incumbent");
                    alns.record_shared_injection();
                    alns.reset_after_injection(&best_score);
                    memory.reset_all();
                    stagnant = 0;
                    continue;
                }

                let penalties = per.bump_gls(&state);
                alns.record_gls(penalties);

                let operator_seed = seed ^ mix64(iter) ^ mix64(stagnant);
                let choice = alns.choose(n, stagnant, operator_seed, iter);
                let current_guided = full_score(&per, &state);
                let current_raw = full_score_raw(&per, &state);
                let Some(candidate) = build_candidate(model, &per, &state, choice, operator_seed, stop) else {
                    alns.record_failed(choice.destroy, choice.repair);
                    continue;
                };
                alns_candidates = alns_candidates.saturating_add(1);
                let candidate_guided = full_score(&per, &candidate);
                let candidate_raw = full_score_raw(&per, &candidate);
                let improved_current = candidate_raw < current_raw;
                let global_best = record_state(&per, &candidate, &mut best_lists, &mut best_score, &mut best_feasible, report);
                if global_best && coordination.as_ref().is_some_and(|shared| shared.publish(&best_lists, &best_score, best_feasible)) {
                    alns.record_shared_publication();
                }
                let acceptance = alns.accept(&current_guided, &candidate_guided, &current_raw, &candidate_raw, operator_seed, iter);
                if !matches!(acceptance, AcceptanceKind::Rejected) {
                    alns_accepted = alns_accepted.saturating_add(1);
                    state = candidate;
                    memory.reset_all();
                }
                alns.record(choice.destroy, choice.repair, acceptance, improved_current, global_best);
                if global_best {
                    stagnant = 0;
                }
            }
        }
    }

    if state_consistent
        && record_state(&per, &state, &mut best_lists, &mut best_score, &mut best_feasible, report)
        && coordination.as_ref().is_some_and(|shared| shared.publish(&best_lists, &best_score, best_feasible))
    {
        alns.record_shared_publication();
    }

    if std::env::var("QAYD_LS_DEBUG").is_ok() {
        eprintln!("LS: iters={iter} local_optima={local_optima} alns_candidates={alns_candidates} alns_accepted={alns_accepted}");
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
    };
    per.metrics.record_alns(alns.metrics());
    let metrics = per.metrics.snapshot(started.map(|instant| instant.elapsed()).unwrap_or_default());
    (solution, metrics)
}
