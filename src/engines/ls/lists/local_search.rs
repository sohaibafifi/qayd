//! Private list local-search heuristic implementation.
//!
//! This module scores and moves over an already-declared model. It must not be
//! the only place where a new modeling feature exists: add the feature to the
//! shared Rust model and backend classifier first, then teach this heuristic to
//! search it as a fallback or incumbent generator.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use smallvec::SmallVec;

use super::eval::{eval_reduction, violation_of, INFEASIBLE};
use super::incremental::{EvalScratch, InsertView, ListView, ReductionCache};
use super::metrics::{metrics_enabled_from_env, ListSearchMetrics, MetricsRecorder};
use super::moves::{apply_move, best_improving_move, better, random_kick, shuffle, snapshot, trial_list_score_view, SearchMemory};
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
    pub(super) fn build(model: &CollectionModel) -> Self {
        let value_to_idx: HashMap<i32, usize> = model.items.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut of_idx = vec![Vec::new(); model.items.len()];
        for (g, c) in model.globals.iter().enumerate() {
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
    pub(super) metrics: MetricsRecorder,
}

pub(super) struct CandidateNeighbors {
    map: HashMap<i32, Vec<i32>>,
}

impl CandidateNeighbors {
    fn build(model: &CollectionModel, matrix: Arc<Vec<Vec<i64>>>) -> Option<Self> {
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
        Self::build_profiled(model, false)
    }

    fn build_profiled(model: &CollectionModel, metrics_enabled: bool) -> Self {
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
            senses[t] = tier.minimize;
            for r in &tier.terms {
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
            let list = c.reduction.iterable.list();
            if let Iterable::Edges { start, end, .. } = &c.reduction.iterable {
                has_edges = true;
                route_bounds[list].get_or_insert((*start, *end));
                candidate_matrix.get_or_insert_with(|| direct_edge_matrix(&c.reduction));
            }
            constraint_delta[list].push(reduction_delta_kind(&c.reduction));
            constraints[list].push(c.clone());
        }
        Self {
            objective,
            max_objective,
            objective_delta,
            constraints,
            constraint_delta,
            senses,
            tiers: model.objectives.len(),
            globals: Globals::build(model),
            has_edges,
            route_bounds,
            candidates: candidate_matrix.flatten().and_then(|matrix| CandidateNeighbors::build(model, matrix)),
            infeas_cand: std::env::var("QAYD_LS_INFEAS_CAND").as_deref() != Ok("0"),
            metrics: MetricsRecorder::new(metrics_enabled),
        }
    }

    pub(super) fn has_max_objective(&self) -> bool {
        self.max_objective.iter().any(|terms| !terms.is_empty())
    }
}

/// The comparable score of a state: violation first, then the objective tiers
/// (each already signed so smaller is better), compared lexicographically.
pub(super) type TierValues = SmallVec<[i64; 4]>;

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

impl ListReductionCaches {
    pub(super) fn build(per: &PerList, idx: usize, contents: &[i32]) -> Self {
        let objective = per.objective[idx]
            .iter()
            .map(|tier| tier.iter().map(|reduction| reduction_cache(per, reduction, contents)).collect())
            .collect();
        let constraints = per.constraints[idx].iter().map(|constraint| reduction_cache(per, &constraint.reduction, contents)).collect();
        Self { objective, constraints }
    }

    pub(super) fn score(&self, per: &PerList, idx: usize) -> ListScore {
        let mut violation = 0i64;
        let mut objectives = tier_values(per.tiers, 0);
        for (tier, slot) in objectives.iter_mut().enumerate() {
            for cache in &self.objective[tier] {
                match cache.value() {
                    Some(value) => *slot = slot.saturating_add(value),
                    None => violation = violation.saturating_add(INFEASIBLE),
                }
            }
        }
        for (constraint, cache) in per.constraints[idx].iter().zip(&self.constraints) {
            match cache.value() {
                Some(value) => violation = violation.saturating_add(violation_of(value, constraint.op, constraint.rhs)),
                None => violation = violation.saturating_add(INFEASIBLE),
            }
        }
        ListScore { violation, objectives }
    }
}

/// Independent full evaluator retained as the incremental-cache oracle.
pub(super) fn list_score_exact(per: &PerList, idx: usize, contents: &[i32]) -> ListScore {
    let mut violation = 0i64;
    let mut objectives = tier_values(per.tiers, 0);
    for (tier, slot) in objectives.iter_mut().enumerate() {
        for reduction in &per.objective[idx][tier] {
            match eval_reduction(reduction, contents) {
                Some(value) => *slot = slot.saturating_add(value),
                None => violation = violation.saturating_add(INFEASIBLE),
            }
        }
    }
    for constraint in &per.constraints[idx] {
        match eval_reduction(&constraint.reduction, contents) {
            Some(value) => violation = violation.saturating_add(violation_of(value, constraint.op, constraint.rhs)),
            None => violation = violation.saturating_add(INFEASIBLE),
        }
    }
    ListScore { violation, objectives }
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

pub(super) fn score_with_replacements<'a>(
    per: &PerList,
    state: &'a State,
    replacements: &'a [TrialList<'a>],
    global_delta: i64,
    scratch: &mut EvalScratch,
) -> Score {
    // Fold in list order, just like a full evaluation. Subtracting an old
    // contribution from a saturated total and adding the replacement is not
    // reversible at i64::MIN/MAX.
    let mut violation = 0i64;
    let mut raw = tier_values(per.tiers, 0);
    for (list, cached) in state.scores.iter().enumerate() {
        let score = replacements.iter().find(|replacement| replacement.list == list).map_or(cached, |replacement| replacement.score);
        violation = violation.saturating_add(score.violation);
        for (slot, &value) in raw.iter_mut().zip(score.objectives.iter()) {
            *slot = slot.saturating_add(value);
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

/// Full score including the cross-list global violation.
pub(super) fn full_score(per: &PerList, state: &State) -> Score {
    score_with_replacements(per, state, &[], 0, &mut EvalScratch::default())
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
    /// on tight instances, while different `order`s give diverse restarts.
    /// Global constraints are ignored here; the search repairs them.
    fn greedy(model: &CollectionModel, per: &PerList, order: &[usize]) -> Self {
        let k = model.lists.max(1);
        let mut state = Self::from_lists(model, per, vec![Vec::new(); k]);
        let mut scratch = EvalScratch::default();
        for &item_idx in order {
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
            state.rescore(per, best_l);
            state.item_list[item_idx] = best_l;
        }
        state.global_viol = per.globals.total(&state.item_list);
        state
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

    pub(super) fn rescore(&mut self, per: &PerList, idx: usize) {
        self.caches[idx] = ListReductionCaches::build(per, idx, &self.lists[idx]);
        self.scores[idx] = self.caches[idx].score(per, idx);
        for (tier, terms) in per.max_objective.iter().enumerate() {
            for (term_idx, term) in terms.iter().enumerate() {
                for (group_idx, group) in term.groups.iter().enumerate() {
                    for (reduction_idx, reduction) in group.iter().enumerate() {
                        if reduction.iterable.list() == idx {
                            self.max_caches[tier][term_idx][group_idx][reduction_idx] = reduction_cache(per, reduction, &self.lists[idx]);
                        }
                    }
                }
            }
        }
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
    let score = full_score(per, state);
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

fn score_with_replaced_list(
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

fn shuffle_values(values: &mut [i32], seed: u64) {
    for i in (1..values.len()).rev() {
        let j = (mix64(seed.wrapping_add(i as u64)) % (i as u64 + 1)) as usize;
        values.swap(i, j);
    }
}

fn destroy_route_segments(lists: &mut [Vec<i32>], target: usize, seed: u64) -> Vec<i32> {
    let mut removed = Vec::with_capacity(target);
    let mut step = 0u64;
    while removed.len() < target {
        let total: usize = lists.iter().map(Vec::len).sum();
        if total == 0 {
            break;
        }
        let mut pick = (mix64(seed ^ mix64(step)) % total as u64) as usize;
        let mut route = 0usize;
        let mut pos = 0usize;
        for (idx, list) in lists.iter().enumerate() {
            if pick < list.len() {
                route = idx;
                pos = pick;
                break;
            }
            pick -= list.len();
        }

        let remaining = target - removed.len();
        let max_len = lists[route].len().min(remaining).min(8);
        let len = 1 + (mix64(seed.wrapping_add(step).wrapping_add(0x9E37_79B9_7F4A_7C15)) % max_len as u64) as usize;
        let start = if pos + len <= lists[route].len() { pos } else { lists[route].len() - len };
        removed.extend(lists[route].drain(start..start + len));
        step = step.wrapping_add(1);
    }
    shuffle_values(&mut removed, seed ^ 0xD1B5_4A32_D192_ED03);
    removed
}

/// Shaw removal: grow a cluster of *related* (here: nearby) customers. Seed with a
/// random customer, then repeatedly remove the nearest still-present customer to a
/// random already-removed one. Removing a related cluster (vs scattered random
/// nodes) gives the repair room to re-route a whole neighbourhood.
fn destroy_shaw(lists: &mut [Vec<i32>], candidates: &CandidateNeighbors, target: usize, seed: u64) -> Vec<i32> {
    let present: HashSet<i32> = lists.iter().flatten().copied().collect();
    if present.is_empty() {
        return Vec::new();
    }
    let all: Vec<i32> = lists.iter().flatten().copied().collect();
    let mut removed_set: HashSet<i32> = HashSet::with_capacity(target);
    let mut order = Vec::with_capacity(target);
    let seed_c = all[(mix64(seed) % all.len() as u64) as usize];
    removed_set.insert(seed_c);
    order.push(seed_c);
    let mut step = 0u64;
    while removed_set.len() < target && removed_set.len() < present.len() {
        let pivot = order[(mix64(seed ^ mix64(step)) % order.len() as u64) as usize];
        let next =
            candidates.nearest_present(pivot, &removed_set, &present).or_else(|| all.iter().copied().find(|c| !removed_set.contains(c)));
        let Some(next) = next else { break };
        removed_set.insert(next);
        order.push(next);
        step = step.wrapping_add(1);
    }
    let mut removed = Vec::with_capacity(removed_set.len());
    for list in lists.iter_mut() {
        list.retain(|c| {
            if removed_set.contains(c) {
                removed.push(*c);
                false
            } else {
                true
            }
        });
    }
    shuffle_values(&mut removed, seed ^ 0xD1B5_4A32_D192_ED03);
    removed
}

fn repair_lns(per: &PerList, state: &mut State, removed: &[i32], seed: u64, stop: &AtomicBool) -> bool {
    let mut scratch = EvalScratch::default();
    let mut polled = 0u32;
    for &item in removed {
        let mut best: Option<(Score, u64, usize, usize, ListScore)> = None;
        for list in 0..state.lists.len() {
            for pos in 0..=state.lists[list].len() {
                polled = polled.wrapping_add(1);
                if polled.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                    return false;
                }
                let candidate = InsertView::new(&state.lists[list], pos, item);
                per.metrics.record_candidate();
                let next = trial_list_score_view(per, state, list, &candidate, None, &mut scratch);
                let global_delta = per.globals.delta(&state.item_list, &[(item, list)]);
                let score = score_with_replaced_list(per, state, list, &next, &candidate, global_delta, &mut scratch);
                let tie = mix64(seed ^ (item as i64 as u64) ^ ((list as u64) << 32) ^ pos as u64);
                if best
                    .as_ref()
                    .is_none_or(|(best_score, best_tie, _, _, _)| score < *best_score || (score == *best_score && tie < *best_tie))
                {
                    best = Some((score, tie, list, pos, next));
                }
            }
        }
        let Some((_, _, list, pos, score)) = best else {
            return false;
        };
        state.lists[list].insert(pos, item);
        state.rescore(per, list);
        debug_assert_eq!(state.scores[list].violation, score.violation);
        debug_assert_eq!(state.scores[list].objectives, score.objectives);
        state.set_item_list(per, item, list);
        state.global_viol = per.globals.total(&state.item_list);
    }
    true
}

/// Flatten a lexicographic [`Score`] into a single comparable cost for regret
/// arithmetic: violation dominates (feasibility first), then the primary
/// objective tier. Exact for the routing case (one tier); a heuristic ordering
/// for multi-tier models, which is fine since regret only *orders* insertions.
fn score_scalar(score: &Score) -> i128 {
    (score.violation as i128) * (1i128 << 50) + score.tiers[0] as i128
}

/// Regret-`k` insertion: repeatedly place the removed item that would "regret"
/// most if deferred -- the one whose cheapest insertion is far better than its
/// next-best `k-1` alternatives, i.e. the item with the fewest good homes left.
/// Items with fewer than `k` feasible spots get an inflated regret so they are
/// placed while options remain. Stronger than greedy cheapest-insertion (which
/// ignores the competition for each slot) at O(pending² · positions) cost, which
/// is bounded because `removed` is a small fraction of the instance.
fn repair_regret(per: &PerList, state: &mut State, removed: &[i32], k: usize, seed: u64, stop: &AtomicBool) -> bool {
    let mut pending: Vec<i32> = removed.to_vec();
    let mut scratch = EvalScratch::default();
    let mut polled = 0u32;
    while !pending.is_empty() {
        // The pending item with the largest regret, and its best placement.
        let mut choice: Option<(usize, i128, u64, usize, usize, ListScore)> = None;
        for (idx, &item) in pending.iter().enumerate() {
            let mut topk: Vec<i128> = Vec::with_capacity(k + 1);
            let mut best_place: Option<(i128, u64, usize, usize, ListScore)> = None;
            for list in 0..state.lists.len() {
                for pos in 0..=state.lists[list].len() {
                    polled = polled.wrapping_add(1);
                    if polled.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                        return false;
                    }
                    let candidate = InsertView::new(&state.lists[list], pos, item);
                    per.metrics.record_candidate();
                    let next = trial_list_score_view(per, state, list, &candidate, None, &mut scratch);
                    let global_delta = per.globals.delta(&state.item_list, &[(item, list)]);
                    let cost = score_scalar(&score_with_replaced_list(per, state, list, &next, &candidate, global_delta, &mut scratch));
                    let tie = mix64(seed ^ (item as i64 as u64) ^ ((list as u64) << 32) ^ pos as u64);
                    let ins = topk.partition_point(|&c| c <= cost);
                    if ins < k {
                        topk.insert(ins, cost);
                        topk.truncate(k);
                    }
                    if best_place.as_ref().is_none_or(|&(bc, bt, _, _, _)| cost < bc || (cost == bc && tie < bt)) {
                        best_place = Some((cost, tie, list, pos, next));
                    }
                }
            }
            let Some((bcost, btie, blist, bpos, bscore)) = best_place else {
                return false; // nowhere to put this item -> repair fails
            };
            // Sum of (i-th best - best) over the next k-1 homes; a missing
            // alternative is charged a large penalty so scarce-option items win.
            const MISS: i128 = 1 << 60;
            let mut regret: i128 = 0;
            for i in 1..k {
                let c = topk.get(i).copied().unwrap_or(bcost + MISS);
                regret = regret.saturating_add(c - bcost);
            }
            if choice.as_ref().is_none_or(|&(_, r, t, _, _, _)| regret > r || (regret == r && btie < t)) {
                choice = Some((idx, regret, btie, blist, bpos, bscore));
            }
        }
        let (idx, _, _, list, pos, score) = choice.expect("pending is non-empty");
        let item = pending.swap_remove(idx);
        state.lists[list].insert(pos, item);
        state.rescore(per, list);
        debug_assert_eq!(state.scores[list].violation, score.violation);
        debug_assert_eq!(state.scores[list].objectives, score.objectives);
        state.set_item_list(per, item, list);
        state.global_viol = per.globals.total(&state.item_list);
    }
    true
}

fn descend_lns_candidate(per: &PerList, state: &mut State, stop: &AtomicBool, max_steps: usize) {
    let mut memory = SearchMemory::new(state.lists.len());
    for _ in 0..max_steps {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let Some(mv) = best_improving_move(per, state, stop, &mut memory) else {
            return;
        };
        apply_move(per, state, mv);
        memory.reset_touched(mv);
    }
}

fn routing_lns(
    model: &CollectionModel,
    per: &PerList,
    incumbent: &[Vec<i32>],
    seed: u64,
    since_improve: u64,
    stop: &AtomicBool,
) -> Option<State> {
    if !per.has_edges || stop.load(Ordering::Relaxed) {
        return None;
    }
    let total: usize = incumbent.iter().map(Vec::len).sum();
    if total == 0 {
        return None;
    }

    let pressure = since_improve.min(20) as usize;
    let jitter = (mix64(seed) % 13) as usize;
    let destroy_pct = (12 + pressure + jitter).min(45);
    let target = (total * destroy_pct).div_ceil(100).clamp(1, total);
    let mut lists = incumbent.to_vec();
    // Shaw removal when a candidate (edge-distance) structure is available, else
    // fall back to random segment destroy. `QAYD_ROUTING_SHAW=0` forces random.
    let removed = match &per.candidates {
        Some(candidates) if std::env::var("QAYD_ROUTING_SHAW").as_deref() != Ok("0") => destroy_shaw(&mut lists, candidates, target, seed),
        _ => destroy_route_segments(&mut lists, target, seed),
    };
    if removed.is_empty() {
        return None;
    }

    let mut state = State::from_lists(model, per, lists);
    // Regret-k repair when `QAYD_ROUTING_REGRET=k` (k>=2); otherwise greedy
    // cheapest-insertion. Regret looks ahead at the competition for each slot.
    let repair_seed = seed ^ 0xA076_1D64_78BD_642F;
    let repaired = match std::env::var("QAYD_ROUTING_REGRET").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(k) if k >= 2 => repair_regret(per, &mut state, &removed, k, repair_seed, stop),
        _ => repair_lns(per, &mut state, &removed, repair_seed, stop),
    };
    if !repaired {
        return None;
    }
    state = State::from_lists(model, per, state.lists);
    debug_assert_eq!(state.lists.iter().map(Vec::len).sum::<usize>(), model.items.len());

    let descent_steps = (removed.len() * 2).clamp(8, 64);
    descend_lns_candidate(per, &mut state, stop, descent_steps);
    Some(state)
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
    solve_collection_capped_internal(model, seed, stop, max_iters, None, report, true)
}

/// Like [`solve_collection`], but seeds the initial incumbent from `hint` -- one
/// visiting-order sequence per list variable, from a caller's constructive
/// heuristic -- instead of the greedy random partition. Universe items the hint
/// omits are placed in the last list (the pool on an optional model, so unhinted
/// nodes stay droppable). The hint seeds only the first incumbent; GRASP restarts
/// still diversify from fresh random partitions, and the best incumbent (possibly
/// the hint itself) is always retained.
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
    let (solution, metrics) = solve_collection_capped_internal(model, seed, stop, max_iters, hint, report, metrics_enabled);
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
    solve_collection_capped_internal(model, seed, stop, max_iters, hint, report, true)
}

fn solve_collection_capped_internal(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    max_iters: u64,
    hint: Option<&[Vec<i32>]>,
    report: &mut dyn FnMut(i64),
    metrics_enabled: bool,
) -> (CollectionSolution, ListSearchMetrics) {
    let started = metrics_enabled.then(Instant::now);
    // Guard the search path: an invalid model would otherwise panic (bad list
    // index) or read silent zeros (out-of-range table index). Callers like the
    // Python frontend validate first to raise a precise error; this is the
    // backstop for direct Rust callers so the engine never panics or corrupts.
    if let Err(_e) = model.validate() {
        debug_assert!(false, "solve_collection called on an invalid model: {_e}");
        let solution = CollectionSolution {
            lists: vec![Vec::new(); model.lists.max(1)],
            objectives: Vec::new(),
            feasible: false,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
        };
        let metrics = MetricsRecorder::new(metrics_enabled).snapshot(started.map(|instant| instant.elapsed()).unwrap_or_default());
        return (solution, metrics);
    }
    if let Some(sched) = &model.schedule {
        let solution = solve_schedule(sched, seed, stop, report);
        let metrics = MetricsRecorder::new(metrics_enabled).snapshot(started.map(|instant| instant.elapsed()).unwrap_or_default());
        return (solution, metrics);
    }
    let per = PerList::build_profiled(model, metrics_enabled);
    let n = model.items.len();
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(&mut order, seed);
    // Warm start from the caller's hint when given (completed to a full
    // partition); otherwise the greedy random construction. Either way the search
    // loop, restarts, and incumbent tracking below are identical.
    let mut state = match hint {
        Some(h) => State::from_lists(model, &per, hint_partition(model, h)),
        None => State::greedy(model, &per, &order),
    };
    let mut memory = SearchMemory::new(model.lists.max(1));

    let (mut best_lists, mut best_score, mut best_feasible) = snapshot(&per, &state);
    if best_feasible && per.tiers > 0 {
        report(tier_value(&per, &best_score, 0));
    }
    // Local optima visited since the incumbent last improved. Descent moves do
    // NOT reset it (otherwise a kick that descent immediately undoes would keep
    // it at zero and the restart below would never fire). After enough fruitless
    // local optima, restart from a fresh random partition (GRASP-style); until
    // then, kick harder the longer the search has been stuck.
    const RESTART_AFTER: u64 = 25;
    const ROUTING_LNS_AFTER: u64 = 8;
    let mut since_improve = 0u64;
    let mut iter = 0u64;
    // When eager (default), a non-improving LNS pass does NOT reset the stuck
    // counter, so LNS keeps firing every local optimum in [8, 25) and the GRASP
    // restart can still eventually trigger; `QAYD_LS_LNS_EAGER=0` restores the
    // original "reset on any LNS pass" behaviour (LNS fires once per stuck cycle).
    let lns_eager = std::env::var("QAYD_LS_LNS_EAGER").as_deref() != Ok("0");
    // Env-gated (`QAYD_LS_DEBUG`) diagnostics: how often the search reaches a
    // local optimum, and how the perturbation budget splits across LNS / restart
    // / kick. Zero runtime cost when the counters are optimised out of the hot
    // path; printed once at loop exit.
    let (mut local_optima, mut lns_calls, mut lns_ok, mut restarts, mut kicks) = (0u64, 0u64, 0u64, 0u64, 0u64);

    while !stop.load(Ordering::Relaxed) && iter < max_iters {
        iter += 1;
        match best_improving_move(&per, &state, stop, &mut memory) {
            Some(mv) => {
                apply_move(&per, &mut state, mv);
                memory.reset_touched(mv);
                if record_state(&per, &state, &mut best_lists, &mut best_score, &mut best_feasible, report) {
                    since_improve = 0;
                }
            }
            None => {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                local_optima += 1;
                if record_state(&per, &state, &mut best_lists, &mut best_score, &mut best_feasible, report) {
                    since_improve = 0;
                } else {
                    since_improve += 1;
                }
                if best_feasible && since_improve >= ROUTING_LNS_AFTER {
                    lns_calls += 1;
                    if let Some(candidate) =
                        routing_lns(model, &per, &best_lists, seed ^ mix64(iter) ^ mix64(since_improve), since_improve, stop)
                    {
                        lns_ok += 1;
                        // Eager: reset the stuck counter ONLY when the LNS actually
                        // improved the incumbent, so a run of non-improving passes lets
                        // `since_improve` climb toward `RESTART_AFTER` instead of pinning
                        // it below and starving the GRASP restart. Non-eager: reset on
                        // any LNS pass (original behaviour, one LNS per stuck cycle).
                        let improved = record_state(&per, &candidate, &mut best_lists, &mut best_score, &mut best_feasible, report);
                        if improved || !lns_eager {
                            since_improve = 0;
                        }
                        state = candidate;
                        memory.reset_all();
                        continue;
                    }
                }
                if since_improve >= RESTART_AFTER {
                    restarts += 1;
                    shuffle(&mut order, seed ^ mix64(iter));
                    state = State::greedy(model, &per, &order);
                    memory.reset_all();
                    since_improve = 0;
                } else {
                    kicks += 1;
                    let strength = 1 + (since_improve / 5) as usize;
                    random_kick(&per, &mut state, seed ^ mix64(iter), strength);
                    memory.reset_all();
                }
            }
        }
    }

    record_state(&per, &state, &mut best_lists, &mut best_score, &mut best_feasible, report);

    if std::env::var("QAYD_LS_DEBUG").is_ok() {
        eprintln!("LS: iters={iter} local_optima={local_optima} lns_calls={lns_calls} lns_ok={lns_ok} restarts={restarts} kicks={kicks}");
    }

    // Report the objective values from the same score that drove the search, so
    // they can never disagree with the accepted solution. When infeasible they
    // are best-effort; `feasible` is the signal to trust.
    let objectives = objective_values(&per, &best_score);
    let solution = CollectionSolution {
        lists: best_lists,
        objectives,
        feasible: best_feasible,
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
    };
    let metrics = per.metrics.snapshot(started.map(|instant| instant.elapsed()).unwrap_or_default());
    (solution, metrics)
}
