use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use super::eval::{eval_reduction, violation_of, INFEASIBLE};
use super::model::{
    CollectionModel, CollectionSolution, Constraint, Expr, ExprId, GlobalConstraint, Iterable, ReduceOp, Reduction, MAX_TIERS,
};
use super::moves::{apply_move, best_improving_move, better, random_kick, shuffle, snapshot};
use super::schedule::solve_schedule;
use crate::mix64;

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
    pub(super) objective: Vec<[Vec<Reduction>; MAX_TIERS]>,
    pub(super) objective_delta: Vec<[Vec<ReductionDeltaKind>; MAX_TIERS]>,
    pub(super) constraints: Vec<Vec<Constraint>>,
    pub(super) constraint_delta: Vec<Vec<ReductionDeltaKind>>,
    pub(super) senses: [bool; MAX_TIERS],
    pub(super) tiers: usize,
    pub(super) globals: Globals,
    /// Whether every reduction on list `l` supports O(edit) incremental scoring,
    /// so a candidate move can be evaluated without rebuilding/rescanning the
    /// list. Lists with an unsupported reduction (Pairs, Scan, Windows, Min, Max)
    /// fall back to full recomputation.
    pub(super) list_incremental: Vec<bool>,
}

#[derive(Clone, Copy)]
pub(super) enum ReductionDeltaKind {
    ItemsSum,
    ItemsCount,
    Used,
    Edges { symmetric: bool },
    Unsupported,
}

impl ReductionDeltaKind {
    fn supported(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

fn expr_is_arg(exprs: &[Expr], id: ExprId, arg: u8) -> bool {
    matches!(exprs.get(id.0 as usize), Some(Expr::Arg(a)) if *a == arg)
}

fn matrix_is_symmetric(matrix: &[Vec<i64>]) -> bool {
    let n = matrix.len();
    if matrix.iter().any(|row| row.len() != n) {
        return false;
    }
    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate().skip(i + 1) {
            if value != matrix[j][i] {
                return false;
            }
        }
    }
    true
}

fn direct_symmetric_edge_matrix(r: &Reduction) -> bool {
    match r.arena.exprs.get(r.body.0 as usize) {
        Some(Expr::Matrix(matrix, row, col)) => {
            let direct_args = expr_is_arg(&r.arena.exprs, *row, 0) && expr_is_arg(&r.arena.exprs, *col, 1);
            let reversed_args = expr_is_arg(&r.arena.exprs, *row, 1) && expr_is_arg(&r.arena.exprs, *col, 0);
            (direct_args || reversed_args) && matrix_is_symmetric(matrix)
        }
        _ => false,
    }
}

/// How a reduction can be scored incrementally from the old list plus a local
/// edit. Symmetric edge-cost detection is cached here, when the per-list index
/// is built, so candidate scoring does not inspect the expression tree.
fn reduction_delta_kind(r: &Reduction) -> ReductionDeltaKind {
    match (r.op, &r.iterable) {
        (ReduceOp::Sum, Iterable::Items(_)) => ReductionDeltaKind::ItemsSum,
        (ReduceOp::Count, Iterable::Items(_)) => ReductionDeltaKind::ItemsCount,
        (ReduceOp::Used, Iterable::Items(_)) => ReductionDeltaKind::Used,
        (ReduceOp::Sum, Iterable::Edges { .. }) => ReductionDeltaKind::Edges { symmetric: direct_symmetric_edge_matrix(r) },
        _ => ReductionDeltaKind::Unsupported,
    }
}

impl PerList {
    pub(super) fn build(model: &CollectionModel) -> Self {
        let mut objective: Vec<[Vec<Reduction>; MAX_TIERS]> = (0..model.lists).map(|_| std::array::from_fn(|_| Vec::new())).collect();
        let mut objective_delta: Vec<[Vec<ReductionDeltaKind>; MAX_TIERS]> =
            (0..model.lists).map(|_| std::array::from_fn(|_| Vec::new())).collect();
        let mut constraints = vec![Vec::new(); model.lists];
        let mut constraint_delta = vec![Vec::new(); model.lists];
        let mut senses = [true; MAX_TIERS];
        for (t, tier) in model.objectives.iter().enumerate() {
            senses[t] = tier.minimize;
            for r in &tier.terms {
                let list = r.iterable.list();
                objective_delta[list][t].push(reduction_delta_kind(r));
                objective[list][t].push(r.clone());
            }
        }
        for c in &model.constraints {
            let list = c.reduction.iterable.list();
            constraint_delta[list].push(reduction_delta_kind(&c.reduction));
            constraints[list].push(c.clone());
        }
        let list_incremental = (0..model.lists)
            .map(|l| {
                objective_delta[l].iter().all(|tier| tier.iter().all(|kind| kind.supported()))
                    && constraint_delta[l].iter().all(|kind| kind.supported())
            })
            .collect();
        Self {
            objective,
            objective_delta,
            constraints,
            constraint_delta,
            senses,
            tiers: model.objectives.len(),
            globals: Globals::build(model),
            list_incremental,
        }
    }
}

/// The comparable score of a state: violation first, then the objective tiers
/// (each already signed so smaller is better), compared lexicographically.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Score {
    pub(super) violation: i64,
    pub(super) tiers: [i64; MAX_TIERS],
}

/// Cached per-list contribution: its violation and its raw (unsigned) value in
/// each objective tier.
#[derive(Clone, Copy)]
pub(super) struct ListScore {
    pub(super) violation: i64,
    pub(super) objectives: [i64; MAX_TIERS],
}

pub(super) fn list_score(per: &PerList, idx: usize, contents: &[i32]) -> ListScore {
    let mut violation = 0i64;
    let mut objectives = [0i64; MAX_TIERS];
    for (t, tier) in per.objective[idx].iter().enumerate() {
        for r in tier {
            match eval_reduction(r, contents) {
                Some(v) => objectives[t] = objectives[t].saturating_add(v),
                None => violation = violation.saturating_add(INFEASIBLE),
            }
        }
    }
    for c in &per.constraints[idx] {
        match eval_reduction(&c.reduction, contents) {
            Some(v) => violation = violation.saturating_add(violation_of(v, c.op, c.rhs)),
            None => violation = violation.saturating_add(INFEASIBLE),
        }
    }
    ListScore { violation, objectives }
}

/// Apply each tier's optimisation direction to raw tier sums (smaller better).
pub(super) fn signed(per: &PerList, violation: i64, raw: [i64; MAX_TIERS]) -> Score {
    let mut tiers = [0i64; MAX_TIERS];
    for ((slot, &r), &minimize) in tiers.iter_mut().zip(raw.iter()).zip(per.senses.iter()) {
        *slot = if minimize { r } else { r.saturating_neg() };
    }
    Score { violation, tiers }
}

/// Raw (unsigned) totals across all lists: violation and per-tier sums.
pub(super) fn base_totals(scores: &[ListScore]) -> (i64, [i64; MAX_TIERS]) {
    let mut violation = 0i64;
    let mut raw = [0i64; MAX_TIERS];
    for s in scores {
        violation = violation.saturating_add(s.violation);
        for (r, &o) in raw.iter_mut().zip(s.objectives.iter()) {
            *r = r.saturating_add(o);
        }
    }
    (violation, raw)
}

pub(super) fn total_score(per: &PerList, scores: &[ListScore]) -> Score {
    let (violation, raw) = base_totals(scores);
    signed(per, violation, raw)
}

/// Full score including the cross-list global violation.
pub(super) fn full_score(per: &PerList, state: &State) -> Score {
    let mut s = total_score(per, &state.scores);
    s.violation = s.violation.saturating_add(state.global_viol);
    s
}

/// State of the search: list contents, cached per-list scores, and (for global
/// constraints) each item's current list plus the total global violation.
pub(super) struct State {
    pub(super) lists: Vec<Vec<i32>>,
    pub(super) scores: Vec<ListScore>,
    /// Raw value of each constraint reduction per list, so a candidate move can
    /// update the (nonlinear) constraint violation incrementally.
    pub(super) con_vals: Vec<Vec<Option<i64>>>,
    pub(super) item_list: Vec<usize>,
    pub(super) global_viol: i64,
}

/// Raw value of every constraint reduction on a list (`None` = undefined).
pub(super) fn compute_con_vals(per: &PerList, idx: usize, contents: &[i32]) -> Vec<Option<i64>> {
    per.constraints[idx].iter().map(|c| eval_reduction(&c.reduction, contents)).collect()
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
        let mut lists: Vec<Vec<i32>> = vec![Vec::new(); k];
        let mut scores: Vec<ListScore> = (0..k).map(|idx| list_score(per, idx, &[])).collect();
        let mut item_list = vec![0usize; model.items.len()];
        let mut buf: Vec<i32> = Vec::new();
        for &item_idx in order {
            let item = model.items[item_idx];
            let mut best_l = 0;
            let mut best_key = Score { violation: i64::MAX, tiers: [i64::MAX; MAX_TIERS] };
            let mut best_sc = scores[0];
            for l in 0..k {
                buf.clear();
                buf.extend_from_slice(&lists[l]);
                buf.push(item);
                let sc = list_score(per, l, &buf);
                // Lexicographic increment of placing the item in list l.
                let dv = sc.violation.saturating_sub(scores[l].violation);
                let mut draw = [0i64; MAX_TIERS];
                for ((d, &new), &old) in draw.iter_mut().zip(sc.objectives.iter()).zip(scores[l].objectives.iter()) {
                    *d = new.saturating_sub(old);
                }
                let key = signed(per, dv, draw);
                if key < best_key {
                    best_key = key;
                    best_l = l;
                    best_sc = sc;
                }
            }
            lists[best_l].push(item);
            scores[best_l] = best_sc;
            item_list[item_idx] = best_l;
        }
        let global_viol = per.globals.total(&item_list);
        let con_vals = (0..k).map(|l| compute_con_vals(per, l, &lists[l])).collect();
        Self { lists, scores, con_vals, item_list, global_viol }
    }

    pub(super) fn rescore(&mut self, per: &PerList, idx: usize) {
        self.scores[idx] = list_score(per, idx, &self.lists[idx]);
        self.con_vals[idx] = compute_con_vals(per, idx, &self.lists[idx]);
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

/// Solve a collection model with constraint-based local search until `stop`.
/// `report` is called with the objective each time a strictly better *feasible*
/// incumbent is found, for progress output; pass `&mut |_| {}` to ignore it.
pub fn solve_collection(model: &CollectionModel, seed: u64, stop: &AtomicBool, report: &mut dyn FnMut(i64)) -> CollectionSolution {
    // Guard the search path: an invalid model would otherwise panic (bad list
    // index) or read silent zeros (out-of-range table index). Callers like the
    // Python frontend validate first to raise a precise error; this is the
    // backstop for direct Rust callers so the engine never panics or corrupts.
    if let Err(_e) = model.validate() {
        debug_assert!(false, "solve_collection called on an invalid model: {_e}");
        return CollectionSolution {
            lists: vec![Vec::new(); model.lists.max(1)],
            objectives: Vec::new(),
            feasible: false,
            starts: Vec::new(),
            machines: Vec::new(),
        };
    }
    if let Some(sched) = &model.schedule {
        return solve_schedule(sched, seed, stop, report);
    }
    let per = PerList::build(model);
    let n = model.items.len();
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(&mut order, seed);
    let mut state = State::greedy(model, &per, &order);

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
    let mut since_improve = 0u64;
    let mut iter = 0u64;

    while !stop.load(Ordering::Relaxed) {
        iter += 1;
        match best_improving_move(&per, &state, stop) {
            Some(mv) => apply_move(&per, &mut state, mv),
            None => {
                let (lists, score, feasible) = snapshot(&per, &state);
                if better(feasible, score, best_feasible, best_score) {
                    best_lists = lists;
                    best_score = score;
                    best_feasible = feasible;
                    if feasible && per.tiers > 0 {
                        report(tier_value(&per, &best_score, 0));
                    }
                    since_improve = 0;
                } else {
                    since_improve += 1;
                }
                if since_improve >= RESTART_AFTER {
                    shuffle(&mut order, seed ^ mix64(iter));
                    state = State::greedy(model, &per, &order);
                    since_improve = 0;
                } else {
                    let strength = 1 + (since_improve / 5) as usize;
                    random_kick(&per, &mut state, seed ^ mix64(iter), strength);
                }
            }
        }
    }

    let (lists, score, feasible) = snapshot(&per, &state);
    if better(feasible, score, best_feasible, best_score) {
        best_lists = lists;
        best_score = score;
        best_feasible = feasible;
    }

    // Report the objective values from the same score that drove the search, so
    // they can never disagree with the accepted solution. When infeasible they
    // are best-effort; `feasible` is the signal to trust.
    let objectives = objective_values(&per, &best_score);
    CollectionSolution { lists: best_lists, objectives, feasible: best_feasible, starts: Vec::new(), machines: Vec::new() }
}
