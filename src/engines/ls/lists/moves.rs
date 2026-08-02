use std::sync::atomic::{AtomicBool, Ordering};

use super::eval::{eval_expr, violation_of};
use super::local_search::{
    compute_con_vals, full_score, list_score, score_with_replacements, ListScore, PerList, ReductionDeltaKind, Score, State, TrialList,
};
use crate::mix64;
use crate::model::list::{CollectionModel, Iterable, Reduction};

pub(super) fn snapshot(per: &PerList, state: &State) -> (Vec<Vec<i32>>, Score, bool) {
    let score = full_score(per, state);
    let feasible = score.violation == 0;
    (state.lists.clone(), score, feasible)
}

/// Prefer feasible over infeasible, then better lexicographic score.
pub(super) fn better(feasible: bool, score: Score, best_feasible: bool, best_score: Score) -> bool {
    match (feasible, best_feasible) {
        (true, false) => true,
        (false, true) => false,
        _ => score < best_score,
    }
}

const MAX_OR_OPT: usize = 3;

#[derive(Clone, Copy)]
pub(super) enum Move {
    Relocate { src: usize, src_pos: usize, dst: usize, dst_pos: usize },
    OrOpt { src: usize, start: usize, len: usize, dst: usize, dst_pos: usize },
    TwoOptStar { a: usize, cut_a: usize, b: usize, cut_b: usize },
    CrossExchange { a: usize, start_a: usize, len_a: usize, b: usize, start_b: usize, len_b: usize },
    Swap { a: usize, a_pos: usize, b: usize, b_pos: usize },
    Reverse { list: usize, i: usize, j: usize },
}

/// Score of the state if one list were replaced by candidate contents.
fn score_one(per: &PerList, state: &State, list: usize, score: ListScore, contents: &[i32], global_delta: i64) -> Score {
    per.metrics.record_candidate();
    score_with_replacements(per, state, &[TrialList { list, score, contents }], global_delta)
}

/// Score of the state if two lists were replaced by candidate contents.
fn score_two<'a>(per: &PerList, state: &'a State, left: TrialList<'a>, right: TrialList<'a>, global_delta: i64) -> Score {
    per.metrics.record_candidate();
    score_with_replacements(per, state, &[left, right], global_delta)
}

/// Copy `list` into `buf` (reusing its capacity) with one edit applied.
fn build_removed(buf: &mut Vec<i32>, list: &[i32], pos: usize) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.remove(pos);
}
fn build_inserted(buf: &mut Vec<i32>, list: &[i32], pos: usize, item: i32) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.insert(pos.min(list.len()), item);
}
fn build_replaced(buf: &mut Vec<i32>, list: &[i32], pos: usize, item: i32) {
    buf.clear();
    buf.extend_from_slice(list);
    buf[pos] = item;
}
fn build_reversed(buf: &mut Vec<i32>, list: &[i32], i: usize, j: usize) {
    buf.clear();
    buf.extend_from_slice(list);
    buf[i..=j].reverse();
}
fn build_segment_removed(buf: &mut Vec<i32>, list: &[i32], start: usize, len: usize) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.drain(start..start + len);
}
fn build_segment_inserted(buf: &mut Vec<i32>, list: &[i32], pos: usize, items: &[i32]) {
    buf.clear();
    buf.extend_from_slice(list);
    let pos = pos.min(buf.len());
    for (offset, &item) in items.iter().enumerate() {
        buf.insert(pos + offset, item);
    }
}
fn build_segment_moved_within(buf: &mut Vec<i32>, list: &[i32], start: usize, len: usize, to: usize) {
    let items = segment_items(list, start, len);
    build_segment_removed(buf, list, start, len);
    let pos = to.min(buf.len());
    for (offset, &item) in items[..len].iter().enumerate() {
        buf.insert(pos + offset, item);
    }
}
fn build_two_opt_star(left: &mut Vec<i32>, right: &mut Vec<i32>, a: &[i32], cut_a: usize, b: &[i32], cut_b: usize) {
    left.clear();
    left.extend_from_slice(&a[..cut_a]);
    left.extend_from_slice(&b[cut_b..]);

    right.clear();
    right.extend_from_slice(&b[..cut_b]);
    right.extend_from_slice(&a[cut_a..]);
}
fn build_cross_exchange(left: &mut Vec<i32>, right: &mut Vec<i32>, a: (&[i32], usize, usize), b: (&[i32], usize, usize)) {
    let (a, start_a, len_a) = a;
    let (b, start_b, len_b) = b;
    left.clear();
    left.extend_from_slice(&a[..start_a]);
    left.extend_from_slice(&b[start_b..start_b + len_b]);
    left.extend_from_slice(&a[start_a + len_a..]);

    right.clear();
    right.extend_from_slice(&b[..start_b]);
    right.extend_from_slice(&a[start_a..start_a + len_a]);
    right.extend_from_slice(&b[start_b + len_b..]);
}
fn build_moved_within(buf: &mut Vec<i32>, list: &[i32], from: usize, to: usize, item: i32) {
    buf.clear();
    buf.extend_from_slice(list);
    buf.remove(from);
    buf.insert(to.min(buf.len()), item);
}

fn segment_items(list: &[i32], start: usize, len: usize) -> [i32; MAX_OR_OPT] {
    let mut items = [0; MAX_OR_OPT];
    items[..len].copy_from_slice(&list[start..start + len]);
    items
}

/// A single local edit to one list. Carries enough to (a) compute the affected
/// reductions' delta in O(edit) and (b) materialise the edited list for the
/// full-recompute fallback. Positions match the `build_*` helpers exactly.
#[derive(Clone, Copy, Debug)]
enum Edit {
    Remove { pos: usize },
    Insert { pos: usize, item: i32 },
    MoveWithin { from: usize, to: usize },
    Replace { pos: usize, item: i32 },
    Reverse { i: usize, j: usize },
    SegmentRemove { start: usize, len: usize },
    SegmentInsert { pos: usize, items: [i32; MAX_OR_OPT], len: usize },
    SegmentMoveWithin { start: usize, len: usize, to: usize },
}

impl Edit {
    /// Length of the list after the edit.
    fn new_len(&self, len: usize) -> usize {
        match self {
            Edit::Remove { .. } => len - 1,
            Edit::Insert { .. } => len + 1,
            Edit::SegmentRemove { len: segment_len, .. } => len - segment_len,
            Edit::SegmentInsert { len: segment_len, .. } => len + segment_len,
            _ => len,
        }
    }

    /// Materialise the edited list into `buf` (the fallback path).
    fn apply(&self, buf: &mut Vec<i32>, list: &[i32]) {
        match *self {
            Edit::Remove { pos } => build_removed(buf, list, pos),
            Edit::Insert { pos, item } => build_inserted(buf, list, pos, item),
            Edit::MoveWithin { from, to } => build_moved_within(buf, list, from, to, list[from]),
            Edit::Replace { pos, item } => build_replaced(buf, list, pos, item),
            Edit::Reverse { i, j } => build_reversed(buf, list, i, j),
            Edit::SegmentRemove { start, len } => build_segment_removed(buf, list, start, len),
            Edit::SegmentInsert { pos, items, len } => build_segment_inserted(buf, list, pos, &items[..len]),
            Edit::SegmentMoveWithin { start, len, to } => build_segment_moved_within(buf, list, start, len, to),
        }
    }
}

pub(super) struct SearchMemory {
    inactive: Vec<bool>,
}

impl SearchMemory {
    pub(super) fn new(routes: usize) -> Self {
        Self { inactive: vec![false; routes] }
    }

    pub(super) fn reset_all(&mut self) {
        self.inactive.fill(false);
    }

    pub(super) fn reset_touched(&mut self, mv: Move) {
        for list in mv.touched_lists() {
            if let Some(inactive) = self.inactive.get_mut(list) {
                *inactive = false;
            }
        }
    }

    fn skip(&self, list: usize) -> bool {
        self.inactive.get(list).copied().unwrap_or(false)
    }

    fn mark_inactive(&mut self, list: usize) {
        if let Some(inactive) = self.inactive.get_mut(list) {
            *inactive = true;
        }
    }
}

impl Move {
    fn touched_lists(self) -> [usize; 2] {
        match self {
            Move::Relocate { src, dst, .. } | Move::OrOpt { src, dst, .. } => [src, dst],
            Move::TwoOptStar { a, b, .. } | Move::CrossExchange { a, b, .. } | Move::Swap { a, b, .. } => [a, b],
            Move::Reverse { list, .. } => [list, list],
        }
    }
}

/// Delta of a per-item (order-independent) value reduction under `edit`, where
/// `value(item)` is the body contribution of one item.
fn items_value_delta(list: &[i32], edit: &Edit, value: impl Fn(i32) -> i64) -> i64 {
    match *edit {
        Edit::Remove { pos } => -value(list[pos]),
        Edit::Insert { item, .. } => value(item),
        Edit::Replace { pos, item } => value(item).saturating_sub(value(list[pos])),
        Edit::SegmentRemove { start, len } => list[start..start + len].iter().fold(0i64, |delta, &item| delta.saturating_sub(value(item))),
        Edit::SegmentInsert { items, len, .. } => items[..len].iter().fold(0i64, |delta, &item| delta.saturating_add(value(item))),
        Edit::MoveWithin { .. } | Edit::Reverse { .. } | Edit::SegmentMoveWithin { .. } => 0,
    }
}

/// Delta of a closed-tour edge-sum reduction (`Edges { start, end }`) under
/// `edit`, where `edge(from, to)` is the body contribution of one edge. The tour
/// is `[start, list.., end]`; only edges local to the edit are touched.
fn edges_value_delta(list: &[i32], start: i32, end: i32, edit: &Edit, symmetric: bool, edge: impl Fn(i32, i32) -> i64) -> i64 {
    let n = list.len();
    // The i-th tour node: 0 -> start, n+1 -> end, else list[i-1].
    let node = |i: usize| -> i32 {
        if i == 0 {
            start
        } else if i == n + 1 {
            end
        } else {
            list[i - 1]
        }
    };
    match *edit {
        Edit::Remove { pos } => {
            let t = pos + 1;
            edge(node(t - 1), node(t + 1)).saturating_sub(edge(node(t - 1), node(t))).saturating_sub(edge(node(t), node(t + 1)))
        }
        Edit::Insert { pos, item } => {
            let (a, b) = (node(pos), node(pos + 1));
            edge(a, item).saturating_add(edge(item, b)).saturating_sub(edge(a, b))
        }
        Edit::Replace { pos, item } => {
            let t = pos + 1;
            let (l, r, old) = (node(t - 1), node(t + 1), node(t));
            edge(l, item).saturating_add(edge(item, r)).saturating_sub(edge(l, old)).saturating_sub(edge(old, r))
        }
        Edit::Reverse { i, j } => {
            let before = node(i);
            let after = node(j + 2);
            if symmetric {
                return edge(before, list[j])
                    .saturating_add(edge(list[i], after))
                    .saturating_sub(edge(before, list[i]))
                    .saturating_sub(edge(list[j], after));
            }
            let mut old = 0i64;
            for t in i..=j + 1 {
                old = old.saturating_add(edge(node(t), node(t + 1)));
            }
            // New node order across positions i..=j+2: before, list[j..=i], after.
            let mut new = edge(before, list[j]);
            for t in (i + 1..=j).rev() {
                new = new.saturating_add(edge(list[t], list[t - 1]));
            }
            new = new.saturating_add(edge(list[i], after));
            new.saturating_sub(old)
        }
        Edit::MoveWithin { from, to } => {
            let item = list[from];
            // Removal delta on the original tour.
            let t = from + 1;
            let rem = edge(node(t - 1), node(t + 1)).saturating_sub(edge(node(t - 1), node(t))).saturating_sub(edge(node(t), node(t + 1)));
            // Insertion delta into the post-removal tour L_rm (length n-1).
            let m = n - 1;
            let to2 = to.min(m);
            let lrm = |i: usize| -> i32 { list[if i < from { i } else { i + 1 }] };
            let node_rm = |i: usize| -> i32 {
                if i == 0 {
                    start
                } else if i == m + 1 {
                    end
                } else {
                    lrm(i - 1)
                }
            };
            let (a, b) = (node_rm(to2), node_rm(to2 + 1));
            let ins = edge(a, item).saturating_add(edge(item, b)).saturating_sub(edge(a, b));
            rem.saturating_add(ins)
        }
        Edit::SegmentRemove { start: segment_start, len } => {
            let first = segment_start + 1;
            let last = segment_start + len;
            edge(node(first - 1), node(last + 1)).saturating_sub(segment_path_cost(
                node(first - 1),
                node(last + 1),
                &list[segment_start..segment_start + len],
                &edge,
            ))
        }
        Edit::SegmentInsert { pos, items, len } => {
            let (a, b) = (node(pos), node(pos + 1));
            segment_insert_delta(a, b, &items[..len], &edge)
        }
        Edit::SegmentMoveWithin { start: segment_start, len, to } => {
            let first = segment_start + 1;
            let last = segment_start + len;
            let rem = edge(node(first - 1), node(last + 1)).saturating_sub(segment_path_cost(
                node(first - 1),
                node(last + 1),
                &list[segment_start..segment_start + len],
                &edge,
            ));

            let m = n - len;
            let to2 = to.min(m);
            let lrm = |i: usize| -> i32 { list[if i < segment_start { i } else { i + len }] };
            let node_rm = |i: usize| -> i32 {
                if i == 0 {
                    start
                } else if i == m + 1 {
                    end
                } else {
                    lrm(i - 1)
                }
            };
            let (a, b) = (node_rm(to2), node_rm(to2 + 1));
            rem.saturating_add(segment_insert_delta(a, b, &list[segment_start..segment_start + len], &edge))
        }
    }
}

fn segment_insert_delta(a: i32, b: i32, items: &[i32], edge: &impl Fn(i32, i32) -> i64) -> i64 {
    segment_path_cost(a, b, items, edge).saturating_sub(edge(a, b))
}

fn segment_path_cost(a: i32, b: i32, items: &[i32], edge: &impl Fn(i32, i32) -> i64) -> i64 {
    if items.is_empty() {
        return 0;
    }
    let mut added = edge(a, items[0]);
    for pair in items.windows(2) {
        added = added.saturating_add(edge(pair[0], pair[1]));
    }
    added = added.saturating_add(edge(*items.last().unwrap_or(&items[0]), b));
    added
}

/// Delta of one supported reduction's raw value under `edit`.
fn reduction_delta(r: &Reduction, kind: ReductionDeltaKind, list: &[i32], edit: &Edit) -> i64 {
    let delta = match kind {
        ReductionDeltaKind::ItemsSum => items_value_delta(list, edit, |item| eval_expr(&r.arena.exprs, r.body, &[i64::from(item)])),
        ReductionDeltaKind::ItemsCount => {
            items_value_delta(list, edit, |item| i64::from(eval_expr(&r.arena.exprs, r.body, &[i64::from(item)]) != 0))
        }
        ReductionDeltaKind::Used => {
            let old = list.len();
            i64::from(edit.new_len(old) > 0) - i64::from(old > 0)
        }
        ReductionDeltaKind::Edges { symmetric } => match &r.iterable {
            Iterable::Edges { start, end, .. } => edges_value_delta(list, *start, *end, edit, symmetric, |from, to| {
                eval_expr(&r.arena.exprs, r.body, &[i64::from(from), i64::from(to)])
            }),
            _ => unreachable!("edge delta kind on a non-edge reduction"),
        },
        ReductionDeltaKind::Unsupported => unreachable!("reduction_delta on an unsupported reduction"),
    };
    delta.saturating_mul(r.coeff)
}

/// Trial score of list `idx` after `edit`, computed incrementally from the cached
/// base score and constraint values when the list is fully incremental, else by
/// materialising the edited list and rescoring it in full.
fn trial_list_score(per: &PerList, state: &State, idx: usize, edit: Edit, scratch: &mut Vec<i32>) -> ListScore {
    if per.has_max_objective() || !per.list_incremental[idx] {
        per.metrics.record_full_trial();
        edit.apply(scratch, &state.lists[idx]);
        list_score(per, idx, scratch)
    } else {
        per.metrics.record_incremental_trial();
        let list = &state.lists[idx];
        let base = &state.scores[idx];
        let mut objectives = base.objectives;
        for (t, slot) in objectives.iter_mut().enumerate() {
            for (r, kind) in per.objective[idx][t].iter().zip(&per.objective_delta[idx][t]) {
                let delta = per.metrics.measure_delta(r, || reduction_delta(r, *kind, list, &edit));
                *slot = slot.saturating_add(delta);
            }
        }
        let mut violation = base.violation;
        for (ci, (c, kind)) in per.constraints[idx].iter().zip(&per.constraint_delta[idx]).enumerate() {
            let old = state.con_vals[idx][ci].expect("supported constraint reduction is always defined");
            let delta = per.metrics.measure_delta(&c.reduction, || reduction_delta(&c.reduction, *kind, list, &edit));
            let new = old.saturating_add(delta);
            violation = violation.saturating_sub(violation_of(old, c.op, c.rhs)).saturating_add(violation_of(new, c.op, c.rhs));
        }
        ListScore { violation, objectives }
    }
}

/// Test oracle: for `model` with the given list contents, assert the incremental
/// trial score of every single-list edit equals a full recompute of the edited
/// list. Panics on the first mismatch; returns the number of edits checked. The
/// `model` must use only incremental-supported reductions (so the delta paths,
/// not the fallback, are exercised). Exposed for the integration oracle, which
/// keeps the actual test out of `src/`.
#[doc(hidden)]
pub fn audit_incremental(model: &CollectionModel, lists: &[Vec<i32>]) -> usize {
    let per = PerList::build(model);
    let k = lists.len();
    assert!((0..k).all(|i| per.list_incremental[i]), "audit model must be fully incremental");
    let scores: Vec<ListScore> = (0..k).map(|i| list_score(&per, i, &lists[i])).collect();
    let con_vals: Vec<Vec<Option<i64>>> = (0..k).map(|i| compute_con_vals(&per, i, &lists[i])).collect();
    let mut item_list = vec![0usize; model.items.len()];
    for (l, lst) in lists.iter().enumerate() {
        for &v in lst {
            if let Some(&i) = per.globals.value_to_idx.get(&v) {
                item_list[i] = l;
            }
        }
    }
    let global_viol = per.globals.total(&item_list);
    let state = State { lists: lists.to_vec(), scores, con_vals, item_list, global_viol };

    let mut scratch = Vec::new();
    let mut full_buf = Vec::new();
    let mut checked = 0usize;
    let same = |a: &ListScore, b: &ListScore| a.violation == b.violation && a.objectives == b.objectives;
    for (idx, list) in lists.iter().enumerate() {
        let n = list.len();
        let mut edits: Vec<Edit> = Vec::new();
        for pos in 0..n {
            edits.push(Edit::Remove { pos });
        }
        for &item in &model.items {
            for pos in 0..=n {
                edits.push(Edit::Insert { pos, item });
            }
            for pos in 0..n {
                edits.push(Edit::Replace { pos, item });
            }
        }
        for from in 0..n {
            for to in 0..n {
                if from != to {
                    edits.push(Edit::MoveWithin { from, to });
                }
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                edits.push(Edit::Reverse { i, j });
            }
        }
        for len in 2..=MAX_OR_OPT.min(n) {
            for start in 0..=n - len {
                edits.push(Edit::SegmentRemove { start, len });
                let items = segment_items(list, start, len);
                for pos in 0..=n {
                    edits.push(Edit::SegmentInsert { pos, items, len });
                }
                for to in 0..=n - len {
                    if to != start {
                        edits.push(Edit::SegmentMoveWithin { start, len, to });
                    }
                }
            }
        }
        for edit in edits {
            let inc = trial_list_score(&per, &state, idx, edit, &mut scratch);
            edit.apply(&mut full_buf, list);
            let full = list_score(&per, idx, &full_buf);
            assert!(
                same(&inc, &full),
                "incremental != full for list {idx} edit {edit:?}: inc=({},{:?}) full=({},{:?})",
                inc.violation,
                inc.objectives,
                full.violation,
                full.objectives
            );
            checked += 1;
        }
    }
    checked
}

fn route_prev_next(per: &PerList, state: &State, list: usize, pos: usize) -> Option<(i32, i32)> {
    let (start, end) = per.route_bounds.get(list).copied().flatten()?;
    let route = state.lists.get(list)?;
    let prev = if pos == 0 { start } else { *route.get(pos - 1)? };
    let next = if pos == route.len() { end } else { *route.get(pos)? };
    Some((prev, next))
}

fn route_before_after(per: &PerList, state: &State, list: usize, start_pos: usize, len: usize) -> Option<(i32, i32)> {
    let (start, end) = per.route_bounds.get(list).copied().flatten()?;
    let route = state.lists.get(list)?;
    let before = if start_pos == 0 { start } else { *route.get(start_pos - 1)? };
    let after_pos = start_pos + len;
    let after = if after_pos == route.len() { end } else { *route.get(after_pos)? };
    Some((before, after))
}

fn candidate_edge(per: &PerList, a: i32, b: i32) -> bool {
    per.candidates.as_ref().is_none_or(|candidates| candidates.contains(a, b))
}

fn candidate_insert(per: &PerList, prev: i32, next: i32, first: i32, last: i32) -> bool {
    candidate_edge(per, prev, first) || candidate_edge(per, last, next)
}

fn candidate_segment_insert(per: &PerList, state: &State, list: usize, pos: usize, first: i32, last: i32) -> bool {
    route_prev_next(per, state, list, pos).is_none_or(|(prev, next)| candidate_insert(per, prev, next, first, last))
}

fn candidate_two_opt_star(per: &PerList, state: &State, a: usize, cut_a: usize, b: usize, cut_b: usize) -> bool {
    match (route_prev_next(per, state, a, cut_a), route_prev_next(per, state, b, cut_b)) {
        (Some((a_prev, a_next)), Some((b_prev, b_next))) => candidate_edge(per, a_prev, b_next) || candidate_edge(per, b_prev, a_next),
        _ => true,
    }
}

fn candidate_cross_exchange(per: &PerList, state: &State, a: (usize, usize, usize), b: (usize, usize, usize)) -> bool {
    let (a, start_a, len_a) = a;
    let (b, start_b, len_b) = b;
    let route_a = &state.lists[a];
    let route_b = &state.lists[b];
    let first_a = route_a[start_a];
    let last_a = route_a[start_a + len_a - 1];
    let first_b = route_b[start_b];
    let last_b = route_b[start_b + len_b - 1];
    match (route_before_after(per, state, a, start_a, len_a), route_before_after(per, state, b, start_b, len_b)) {
        (Some((a_before, a_after)), Some((b_before, b_after))) => {
            candidate_edge(per, a_before, first_b)
                || candidate_edge(per, last_b, a_after)
                || candidate_edge(per, b_before, first_a)
                || candidate_edge(per, last_a, b_after)
        }
        _ => true,
    }
}

/// First-improvement local move: return the first relocate / swap / or-opt /
/// 2-opt* / cross-exchange / reversal that strictly lowers the score, or `None`
/// at a local optimum (or when `stop` fires). The neighbourhood is scanned one
/// source route at a time: a route whose *entire* neighbourhood yields no
/// improvement is marked inactive (don't-look bit) and skipped until a later
/// applied move touches it again, so settled routes are not re-scanned every
/// pass. Candidate buffers are reused, so no allocation happens per candidate.
/// `stop` is polled per candidate so even very long lists honour the time limit.
pub(super) fn best_improving_move(per: &PerList, state: &State, stop: &AtomicBool, memory: &mut SearchMemory) -> Option<Move> {
    let current = full_score(per, state);
    // Repair-capable moves (relocate / or-opt insert filters) only prune by the
    // geometric candidate lists once feasible: while a route still overflows, the
    // repair destination may not be a geometric neighbour, so they need the full
    // neighbourhood. Cost-refiner moves (2-opt* / cross / reverse) are the
    // O(k²·len^…) cost of a pass and never the *only* way to repair load (plain
    // relocate is a complete membership neighbourhood), so they prune by
    // candidates even while infeasible - that is what lets an infeasible-heavy
    // instance afford enough passes to reach feasibility.
    let use_candidates = per.has_edges && current.violation == 0 && per.candidates.is_some();
    let cand_cost = if per.infeas_cand { per.has_edges && per.candidates.is_some() } else { use_candidates };
    let k = state.lists.len();
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut overrides = Vec::new();
    let mut polled = 0u32;
    let mut stopped = || {
        polled = polled.wrapping_add(1);
        polled.is_multiple_of(1024) && stop.load(Ordering::Relaxed)
    };
    // Scan one source route at a time so a settled route can be skipped wholesale
    // via its don't-look bit. A route is marked inactive only once its *entire*
    // neighbourhood (relocate, swap, and for routing: or-opt, 2-opt*, cross,
    // reverse, all originating from it) yields no improving move. Pair moves are
    // scanned from both endpoints, so a pair (src, y) is still examined when src
    // is active even if y is inactive; the redundant active/active double-scan is
    // bounded because relocate returns early while the search is still dense.
    for src in 0..k {
        if memory.skip(src) {
            continue;
        }
        // --- Relocate a single item out of `src`. ---
        for src_pos in 0..state.lists[src].len() {
            let item = state.lists[src][src_pos];
            for dst in 0..k {
                if dst == src {
                    let len = state.lists[src].len();
                    for dst_pos in 0..len {
                        if dst_pos == src_pos {
                            continue;
                        }
                        if stopped() {
                            return None;
                        }
                        let nl = trial_list_score(per, state, src, Edit::MoveWithin { from: src_pos, to: dst_pos }, &mut a);
                        // Within-list move: no item changes list, so gdelta = 0.
                        if score_one(per, state, src, nl, &a, 0) < current {
                            return Some(Move::Relocate { src, src_pos, dst, dst_pos });
                        }
                    }
                } else {
                    let na = trial_list_score(per, state, src, Edit::Remove { pos: src_pos }, &mut a);
                    let gd = per.globals.delta(&state.item_list, &[(item, dst)]);
                    for dst_pos in 0..=state.lists[dst].len() {
                        if stopped() {
                            return None;
                        }
                        if use_candidates && !candidate_segment_insert(per, state, dst, dst_pos, item, item) {
                            continue;
                        }
                        let nb = trial_list_score(per, state, dst, Edit::Insert { pos: dst_pos, item }, &mut b);
                        if score_two(
                            per,
                            state,
                            TrialList { list: src, score: na, contents: &a },
                            TrialList { list: dst, score: nb, contents: &b },
                            gd,
                        ) < current
                        {
                            return Some(Move::Relocate { src, src_pos, dst, dst_pos });
                        }
                    }
                }
            }
        }
        // --- Swap one item of `src` with one item of another route. ---
        for y in 0..k {
            if y == src {
                continue;
            }
            for xp in 0..state.lists[src].len() {
                for yp in 0..state.lists[y].len() {
                    if stopped() {
                        return None;
                    }
                    let (vx, vy) = (state.lists[src][xp], state.lists[y][yp]);
                    let na = trial_list_score(per, state, src, Edit::Replace { pos: xp, item: vy }, &mut a);
                    let nb = trial_list_score(per, state, y, Edit::Replace { pos: yp, item: vx }, &mut b);
                    let gd = per.globals.delta(&state.item_list, &[(vx, y), (vy, src)]);
                    if score_two(
                        per,
                        state,
                        TrialList { list: src, score: na, contents: &a },
                        TrialList { list: y, score: nb, contents: &b },
                        gd,
                    ) < current
                    {
                        return Some(Move::Swap { a: src, a_pos: xp, b: y, b_pos: yp });
                    }
                }
            }
        }
        // --- Routing moves originating from `src` (need edge costs). ---
        if per.has_edges {
            // Or-opt: relocate a 2..=MAX_OR_OPT segment of `src`.
            let src_len = state.lists[src].len();
            for len in 2..=MAX_OR_OPT.min(src_len) {
                for start in 0..=src_len - len {
                    let items = segment_items(&state.lists[src], start, len);
                    for dst in 0..k {
                        if dst == src {
                            let post_len = src_len - len;
                            for dst_pos in 0..=post_len {
                                if dst_pos == start {
                                    continue;
                                }
                                if stopped() {
                                    return None;
                                }
                                let nl = trial_list_score(per, state, src, Edit::SegmentMoveWithin { start, len, to: dst_pos }, &mut a);
                                if score_one(per, state, src, nl, &a, 0) < current {
                                    return Some(Move::OrOpt { src, start, len, dst, dst_pos });
                                }
                            }
                        } else {
                            let na = trial_list_score(per, state, src, Edit::SegmentRemove { start, len }, &mut a);
                            let mut overrides = [(0, 0); MAX_OR_OPT];
                            for slot in 0..len {
                                overrides[slot] = (items[slot], dst);
                            }
                            let gd = per.globals.delta(&state.item_list, &overrides[..len]);
                            for dst_pos in 0..=state.lists[dst].len() {
                                if stopped() {
                                    return None;
                                }
                                if use_candidates && !candidate_segment_insert(per, state, dst, dst_pos, items[0], items[len - 1]) {
                                    continue;
                                }
                                let nb = trial_list_score(per, state, dst, Edit::SegmentInsert { pos: dst_pos, items, len }, &mut b);
                                if score_two(
                                    per,
                                    state,
                                    TrialList { list: src, score: na, contents: &a },
                                    TrialList { list: dst, score: nb, contents: &b },
                                    gd,
                                ) < current
                                {
                                    return Some(Move::OrOpt { src, start, len, dst, dst_pos });
                                }
                            }
                        }
                    }
                }
            }
            // 2-opt*: swap the tails of `src` and another route.
            for y in 0..k {
                if y == src {
                    continue;
                }
                let lx = state.lists[src].len();
                let ly = state.lists[y].len();
                for cut_x in 0..=lx {
                    for cut_y in 0..=ly {
                        if cut_x == lx && cut_y == ly {
                            continue;
                        }
                        if stopped() {
                            return None;
                        }
                        if cand_cost && !candidate_two_opt_star(per, state, src, cut_x, y, cut_y) {
                            continue;
                        }
                        build_two_opt_star(&mut a, &mut b, &state.lists[src], cut_x, &state.lists[y], cut_y);
                        per.metrics.record_full_trial();
                        let na = list_score(per, src, &a);
                        per.metrics.record_full_trial();
                        let nb = list_score(per, y, &b);
                        overrides.clear();
                        overrides.extend(state.lists[src][cut_x..].iter().map(|&item| (item, y)));
                        overrides.extend(state.lists[y][cut_y..].iter().map(|&item| (item, src)));
                        let gd = per.globals.delta(&state.item_list, &overrides);
                        if score_two(
                            per,
                            state,
                            TrialList { list: src, score: na, contents: &a },
                            TrialList { list: y, score: nb, contents: &b },
                            gd,
                        ) < current
                        {
                            return Some(Move::TwoOptStar { a: src, cut_a: cut_x, b: y, cut_b: cut_y });
                        }
                    }
                }
            }
            // Cross-exchange: swap a segment of `src` with a segment of another route.
            for y in 0..k {
                if y == src {
                    continue;
                }
                let lx = state.lists[src].len();
                let ly = state.lists[y].len();
                for len_x in 1..=MAX_OR_OPT.min(lx) {
                    for start_x in 0..=lx - len_x {
                        for len_y in 1..=MAX_OR_OPT.min(ly) {
                            if len_x == 1 && len_y == 1 {
                                continue;
                            }
                            for start_y in 0..=ly - len_y {
                                if stopped() {
                                    return None;
                                }
                                if cand_cost && !candidate_cross_exchange(per, state, (src, start_x, len_x), (y, start_y, len_y)) {
                                    continue;
                                }
                                build_cross_exchange(
                                    &mut a,
                                    &mut b,
                                    (&state.lists[src], start_x, len_x),
                                    (&state.lists[y], start_y, len_y),
                                );
                                per.metrics.record_full_trial();
                                let na = list_score(per, src, &a);
                                per.metrics.record_full_trial();
                                let nb = list_score(per, y, &b);
                                overrides.clear();
                                overrides.extend(state.lists[src][start_x..start_x + len_x].iter().map(|&item| (item, y)));
                                overrides.extend(state.lists[y][start_y..start_y + len_y].iter().map(|&item| (item, src)));
                                let gd = per.globals.delta(&state.item_list, &overrides);
                                if score_two(
                                    per,
                                    state,
                                    TrialList { list: src, score: na, contents: &a },
                                    TrialList { list: y, score: nb, contents: &b },
                                    gd,
                                ) < current
                                {
                                    return Some(Move::CrossExchange {
                                        a: src,
                                        start_a: start_x,
                                        len_a: len_x,
                                        b: y,
                                        start_b: start_y,
                                        len_b: len_y,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            // 2-opt: reverse a segment within `src`.
            let len = state.lists[src].len();
            for i in 0..len {
                for j in (i + 1)..len {
                    if stopped() {
                        return None;
                    }
                    if cand_cost {
                        let Some((before, after)) = route_before_after(per, state, src, i, j + 1 - i) else {
                            continue;
                        };
                        if !candidate_edge(per, before, state.lists[src][j]) && !candidate_edge(per, state.lists[src][i], after) {
                            continue;
                        }
                    }
                    let nl = trial_list_score(per, state, src, Edit::Reverse { i, j }, &mut a);
                    if score_one(per, state, src, nl, &a, 0) < current {
                        return Some(Move::Reverse { list: src, i, j });
                    }
                }
            }
        }
        memory.mark_inactive(src);
    }
    None
}

pub(super) fn apply_move(per: &PerList, state: &mut State, mv: Move) {
    match mv {
        Move::Relocate { src, src_pos, dst, dst_pos } => {
            let item = state.lists[src].remove(src_pos);
            let pos = dst_pos.min(state.lists[dst].len());
            state.lists[dst].insert(pos, item);
            state.rescore(per, src);
            state.rescore(per, dst);
            if src != dst {
                state.set_item_list(per, item, dst);
                state.global_viol = per.globals.total(&state.item_list);
            }
        }
        Move::OrOpt { src, start, len, dst, dst_pos } => {
            let segment: Vec<i32> = state.lists[src].drain(start..start + len).collect();
            let pos = dst_pos.min(state.lists[dst].len());
            for (offset, &item) in segment.iter().enumerate() {
                state.lists[dst].insert(pos + offset, item);
            }
            state.rescore(per, src);
            if src != dst {
                state.rescore(per, dst);
                for &item in &segment {
                    state.set_item_list(per, item, dst);
                }
                state.global_viol = per.globals.total(&state.item_list);
            }
        }
        Move::TwoOptStar { a, cut_a, b, cut_b } => {
            let tail_a = state.lists[a].split_off(cut_a);
            let tail_b = state.lists[b].split_off(cut_b);
            state.lists[a].extend(tail_b.iter().copied());
            state.lists[b].extend(tail_a.iter().copied());
            state.rescore(per, a);
            state.rescore(per, b);
            for item in tail_a {
                state.set_item_list(per, item, b);
            }
            for item in tail_b {
                state.set_item_list(per, item, a);
            }
            state.global_viol = per.globals.total(&state.item_list);
        }
        Move::CrossExchange { a, start_a, len_a, b, start_b, len_b } => {
            let seg_a: Vec<i32> = state.lists[a].drain(start_a..start_a + len_a).collect();
            let seg_b: Vec<i32> = state.lists[b].drain(start_b..start_b + len_b).collect();
            for (offset, &item) in seg_b.iter().enumerate() {
                state.lists[a].insert(start_a + offset, item);
            }
            for (offset, &item) in seg_a.iter().enumerate() {
                state.lists[b].insert(start_b + offset, item);
            }
            state.rescore(per, a);
            state.rescore(per, b);
            for &item in &seg_a {
                state.set_item_list(per, item, b);
            }
            for &item in &seg_b {
                state.set_item_list(per, item, a);
            }
            state.global_viol = per.globals.total(&state.item_list);
        }
        Move::Swap { a, a_pos, b, b_pos } => {
            let tmp = state.lists[a][a_pos];
            state.lists[a][a_pos] = state.lists[b][b_pos];
            state.lists[b][b_pos] = tmp;
            state.rescore(per, a);
            state.rescore(per, b);
            // After the swap, lists[a][a_pos] holds the item that was in b.
            state.set_item_list(per, state.lists[a][a_pos], a);
            state.set_item_list(per, state.lists[b][b_pos], b);
            state.global_viol = per.globals.total(&state.item_list);
        }
        Move::Reverse { list, i, j } => {
            state.lists[list][i..=j].reverse();
            state.rescore(per, list);
            // Reversal keeps every item in the same list, so globals are unchanged.
        }
    }
}

pub(super) fn shuffle(order: &mut [usize], seed: u64) {
    for i in (1..order.len()).rev() {
        let j = (mix64(seed.wrapping_add(i as u64)) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

/// Move `strength` random items, each to a random other list, to escape a local
/// minimum. A larger strength is a bigger perturbation for a deeper basin.
pub(super) fn random_kick(per: &PerList, state: &mut State, seed: u64, strength: usize) {
    if state.lists.len() < 2 {
        return;
    }
    for step in 0..strength.max(1) {
        let s = seed ^ mix64(step as u64);
        let total: usize = state.lists.iter().map(Vec::len).sum();
        if total == 0 {
            return;
        }
        let mut pick = (mix64(s) % total as u64) as usize;
        let mut src = 0;
        let mut src_pos = 0;
        for (r, l) in state.lists.iter().enumerate() {
            if pick < l.len() {
                src = r;
                src_pos = pick;
                break;
            }
            pick -= l.len();
        }
        let dst = (mix64(s ^ 0x9E37) % state.lists.len() as u64) as usize;
        if dst == src {
            continue;
        }
        let item = state.lists[src].remove(src_pos);
        state.lists[dst].push(item);
        state.rescore(per, src);
        state.rescore(per, dst);
        state.set_item_list(per, item, dst);
    }
    state.global_viol = per.globals.total(&state.item_list);
}
