use std::sync::atomic::{AtomicBool, Ordering};

use super::eval::{eval_expr, violation_of, INFEASIBLE};
use super::incremental::{EvalScratch, ListView};
use super::local_search::{
    active_list_indices, full_score, full_score_exact_lists, full_score_raw, list_score_exact, score_with_replacements, tier_values,
    ConstraintViolations, ListScore, ObjectiveReductionValues, PerList, ReductionDeltaKind, ReductionValues, Score, State, TrialList,
};
use crate::mix64;
use crate::model::list::{CollectionModel, Iterable, Reduction};

pub(super) fn snapshot(per: &PerList, state: &State) -> (Vec<Vec<i32>>, Score, bool) {
    let score = full_score_raw(per, state);
    let feasible = score.violation == 0;
    (state.lists.clone(), score, feasible)
}

/// Prefer feasible over infeasible, then better lexicographic score.
pub(super) fn better(feasible: bool, score: &Score, best_feasible: bool, best_score: &Score) -> bool {
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
fn score_one(
    per: &PerList,
    state: &State,
    list: usize,
    score: &ListScore,
    contents: &dyn ListView,
    global_delta: i64,
    scratch: &mut EvalScratch,
) -> Score {
    per.metrics.record_candidate();
    score_with_replacements(per, state, &[TrialList { list, score, contents }], global_delta, scratch)
}

/// Score of the state if two lists were replaced by candidate contents.
fn score_two<'a>(
    per: &PerList,
    state: &'a State,
    left: TrialList<'a>,
    right: TrialList<'a>,
    global_delta: i64,
    scratch: &mut EvalScratch,
) -> Score {
    per.metrics.record_candidate();
    score_with_replacements(per, state, &[left, right], global_delta, scratch)
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
pub(super) enum Edit {
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

struct EditView<'a> {
    base: &'a [i32],
    edit: Edit,
}

impl<'a> EditView<'a> {
    fn new(base: &'a [i32], edit: Edit) -> Self {
        Self { base, edit }
    }
}

impl ListView for EditView<'_> {
    fn len(&self) -> usize {
        self.edit.new_len(self.base.len())
    }

    fn at(&self, index: usize) -> i32 {
        match self.edit {
            Edit::Remove { pos } => self.base[if index < pos { index } else { index + 1 }],
            Edit::Insert { pos, item } => {
                let pos = pos.min(self.base.len());
                if index < pos {
                    self.base[index]
                } else if index == pos {
                    item
                } else {
                    self.base[index - 1]
                }
            }
            Edit::MoveWithin { from, to } => {
                let to = to.min(self.base.len() - 1);
                if from < to {
                    if index < from || index > to {
                        self.base[index]
                    } else if index == to {
                        self.base[from]
                    } else {
                        self.base[index + 1]
                    }
                } else if from > to {
                    if index < to || index > from {
                        self.base[index]
                    } else if index == to {
                        self.base[from]
                    } else {
                        self.base[index - 1]
                    }
                } else {
                    self.base[index]
                }
            }
            Edit::Replace { pos, item } => {
                if index == pos {
                    item
                } else {
                    self.base[index]
                }
            }
            Edit::Reverse { i, j } => {
                if (i..=j).contains(&index) {
                    self.base[j - (index - i)]
                } else {
                    self.base[index]
                }
            }
            Edit::SegmentRemove { start, len } => self.base[if index < start { index } else { index + len }],
            Edit::SegmentInsert { pos, items, len } => {
                let pos = pos.min(self.base.len());
                if index < pos {
                    self.base[index]
                } else if index < pos + len {
                    items[index - pos]
                } else {
                    self.base[index - len]
                }
            }
            Edit::SegmentMoveWithin { start, len, to } => {
                let post_len = self.base.len() - len;
                let to = to.min(post_len);
                if (to..to + len).contains(&index) {
                    self.base[start + index - to]
                } else {
                    let removed_index = if index < to { index } else { index - len };
                    self.base[if removed_index < start { removed_index } else { removed_index + len }]
                }
            }
        }
    }

    fn common_prefix_len(&self, _old: &[i32]) -> usize {
        match self.edit {
            Edit::Remove { pos } | Edit::Insert { pos, .. } | Edit::Replace { pos, .. } => pos,
            Edit::MoveWithin { from, to } => from.min(to),
            Edit::Reverse { i, .. } => i,
            Edit::SegmentRemove { start, .. } => start,
            Edit::SegmentInsert { pos, .. } => pos,
            Edit::SegmentMoveWithin { start, to, .. } => start.min(to),
        }
    }

    fn common_suffix_len(&self, _old: &[i32], _prefix: usize) -> usize {
        let old_len = self.base.len();
        match self.edit {
            Edit::Remove { pos } => old_len - pos - 1,
            Edit::Insert { pos, .. } => old_len - pos.min(old_len),
            Edit::MoveWithin { from, to } => old_len - from.max(to) - 1,
            Edit::Replace { pos, .. } => old_len - pos - 1,
            Edit::Reverse { j, .. } => old_len - j - 1,
            Edit::SegmentRemove { start, len } => old_len - start - len,
            Edit::SegmentInsert { pos, .. } => old_len - pos.min(old_len),
            Edit::SegmentMoveWithin { start, len, to } => old_len - (start + len).max(to + len),
        }
    }
}

struct ChunkView<'a> {
    chunks: [&'a [i32]; 3],
    count: usize,
    len: usize,
    common_prefix: usize,
    common_suffix: usize,
}

impl<'a> ChunkView<'a> {
    fn two(first: &'a [i32], second: &'a [i32]) -> Self {
        Self { chunks: [first, second, &[]], count: 2, len: first.len() + second.len(), common_prefix: first.len(), common_suffix: 0 }
    }

    fn three(first: &'a [i32], second: &'a [i32], third: &'a [i32]) -> Self {
        Self {
            chunks: [first, second, third],
            count: 3,
            len: first.len() + second.len() + third.len(),
            common_prefix: first.len(),
            common_suffix: third.len(),
        }
    }
}

impl ListView for ChunkView<'_> {
    fn len(&self) -> usize {
        self.len
    }

    fn at(&self, mut index: usize) -> i32 {
        for chunk in &self.chunks[..self.count] {
            if index < chunk.len() {
                return chunk[index];
            }
            index -= chunk.len();
        }
        unreachable!("candidate chunk index out of range")
    }

    fn common_prefix_len(&self, _old: &[i32]) -> usize {
        self.common_prefix
    }

    fn common_suffix_len(&self, _old: &[i32], _prefix: usize) -> usize {
        self.common_suffix
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

/// Delta of one supported reduction's raw value under `edit`.
fn reduction_delta(r: &Reduction, kind: ReductionDeltaKind, list: &[i32], edit: &Edit) -> i64 {
    match kind {
        ReductionDeltaKind::ItemsCount => {
            items_value_delta(list, edit, |item| i64::from(eval_expr(&r.arena.exprs, r.body, &[i64::from(item)]) != 0))
        }
        ReductionDeltaKind::Used => {
            let old = list.len();
            i64::from(edit.new_len(old) > 0) - i64::from(old > 0)
        }
        ReductionDeltaKind::Unsupported => unreachable!("reduction_delta on an unsupported reduction"),
    }
}

#[allow(clippy::too_many_arguments)]
fn candidate_reduction_value(
    per: &PerList,
    reduction: &Reduction,
    kind: ReductionDeltaKind,
    cache: &super::incremental::ReductionCache,
    old: &[i32],
    candidate: &dyn ListView,
    edit: Option<&Edit>,
    scratch: &mut EvalScratch,
) -> Option<i64> {
    // Count and Used are the only algebraic deltas that remain exact across
    // every i64 saturation boundary. Sum reductions use the ordered prefix and
    // suffix transforms in ReductionCache instead: adding a delta to an already
    // saturated total is not equivalent to replaying saturating_add.
    let exact_raw_delta = matches!(kind, ReductionDeltaKind::ItemsCount | ReductionDeltaKind::Used);
    let value = match (exact_raw_delta, edit) {
        (true, Some(edit)) => per.metrics.measure_delta(reduction, || {
            cache.raw_value().map(|raw| raw.saturating_add(reduction_delta(reduction, kind, old, edit)).saturating_mul(reduction.coeff))
        }),
        _ => per.metrics.measure_delta(reduction, || cache.candidate_value(reduction, old, candidate, scratch)),
    };
    if matches!(reduction.iterable, Iterable::Scan { .. }) {
        per.metrics.record_incremental_scan(scratch.recomputed_scan_steps());
    }
    value
}

/// Trial score from accepted-state reduction caches. Local edits retain the
/// existing O(edit) formulas for Count/Used; every other reduction recomputes
/// only its affected output span through `candidate`.
pub(super) fn trial_list_score_view(
    per: &PerList,
    state: &State,
    idx: usize,
    candidate: &dyn ListView,
    edit: Option<&Edit>,
    scratch: &mut EvalScratch,
) -> ListScore {
    per.metrics.record_incremental_trial();
    let old = &state.lists[idx];
    let caches = &state.caches[idx];
    let mut violation = 0i64;
    let mut undefined_violation = 0i64;
    let mut objectives = tier_values(per.tiers, 0);
    let mut objective_reductions = ObjectiveReductionValues::with_capacity(per.tiers);
    for (tier, slot) in objectives.iter_mut().enumerate() {
        let mut values = ReductionValues::with_capacity(per.objective[idx][tier].len());
        for ((reduction, kind), cache) in per.objective[idx][tier].iter().zip(&per.objective_delta[idx][tier]).zip(&caches.objective[tier])
        {
            match candidate_reduction_value(per, reduction, *kind, cache, old, candidate, edit, scratch) {
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
    for ((constraint, kind), cache) in per.constraints[idx].iter().zip(&per.constraint_delta[idx]).zip(&caches.constraints) {
        match candidate_reduction_value(per, &constraint.reduction, *kind, cache, old, candidate, edit, scratch) {
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
    let edge_penalty = per.edge_penalty(idx, candidate);
    ListScore { violation, objectives, constraint_violations, objective_reductions, undefined_violation, edge_penalty }
}

fn rotated_routes(mut routes: Vec<usize>, seed: u64) -> Vec<usize> {
    if seed != 0 && !routes.is_empty() {
        let offset = (mix64(seed) % routes.len() as u64) as usize;
        routes.rotate_left(offset);
    }
    routes
}

fn trial_list_score(per: &PerList, state: &State, idx: usize, edit: Edit, scratch: &mut EvalScratch) -> ListScore {
    let candidate = EditView::new(&state.lists[idx], edit);
    trial_list_score_view(per, state, idx, &candidate, Some(&edit), scratch)
}

/// Test oracle: for `model` with the given list contents, assert the incremental
/// trial score of every single-list edit equals a full recompute of the edited
/// list. Panics on the first mismatch; returns the number of edits checked.
/// Exposed for the integration oracle, which keeps the actual test out of `src/`.
#[doc(hidden)]
pub fn audit_incremental(model: &CollectionModel, lists: &[Vec<i32>]) -> usize {
    let per = PerList::build(model);
    let state = State::from_lists(model, &per, lists.to_vec());

    let mut scratch = EvalScratch::default();
    let mut right_scratch = EvalScratch::default();
    let mut score_scratch = EvalScratch::default();
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
            let view = EditView::new(list, edit);
            let inc = trial_list_score(&per, &state, idx, edit, &mut scratch);
            let incremental_total =
                score_with_replacements(&per, &state, &[TrialList { list: idx, score: &inc, contents: &view }], 0, &mut score_scratch);
            edit.apply(&mut full_buf, list);
            let full = list_score_exact(&per, idx, &full_buf);
            assert!(
                same(&inc, &full),
                "incremental != full for list {idx} edit {edit:?}: inc=({},{:?}) full=({},{:?})",
                inc.violation,
                inc.objectives,
                full.violation,
                full.objectives
            );
            let mut full_lists = lists.to_vec();
            full_lists[idx] = full_buf.clone();
            let full_total = full_score_exact_lists(&per, &full_lists, state.global_viol);
            assert!(
                incremental_total == full_total,
                "incremental total != full total for list {idx} edit {edit:?}: inc=({},{:?}) full=({},{:?})",
                incremental_total.violation,
                incremental_total.tiers,
                full_total.violation,
                full_total.tiers
            );
            checked += 1;
        }
    }

    for left in 0..lists.len() {
        for right in (left + 1)..lists.len() {
            for cut_left in 0..=lists[left].len() {
                for cut_right in 0..=lists[right].len() {
                    if cut_left == lists[left].len() && cut_right == lists[right].len() {
                        continue;
                    }
                    let left_view = ChunkView::two(&lists[left][..cut_left], &lists[right][cut_right..]);
                    let right_view = ChunkView::two(&lists[right][..cut_right], &lists[left][cut_left..]);
                    let left_score = trial_list_score_view(&per, &state, left, &left_view, None, &mut scratch);
                    let right_score = trial_list_score_view(&per, &state, right, &right_view, None, &mut right_scratch);
                    let incremental_total = score_with_replacements(
                        &per,
                        &state,
                        &[
                            TrialList { list: left, score: &left_score, contents: &left_view },
                            TrialList { list: right, score: &right_score, contents: &right_view },
                        ],
                        0,
                        &mut score_scratch,
                    );
                    let mut materialized_left = Vec::new();
                    let mut materialized_right = Vec::new();
                    build_two_opt_star(&mut materialized_left, &mut materialized_right, &lists[left], cut_left, &lists[right], cut_right);
                    let mut full_lists = lists.to_vec();
                    full_lists[left] = materialized_left;
                    full_lists[right] = materialized_right;
                    let full_total = full_score_exact_lists(&per, &full_lists, state.global_viol);
                    assert!(incremental_total == full_total, "2-opt* incremental total differs from full recomputation");
                    checked += 1;
                }
            }

            for len_left in 1..=MAX_OR_OPT.min(lists[left].len()) {
                for start_left in 0..=lists[left].len() - len_left {
                    for len_right in 1..=MAX_OR_OPT.min(lists[right].len()) {
                        for start_right in 0..=lists[right].len() - len_right {
                            let left_view = ChunkView::three(
                                &lists[left][..start_left],
                                &lists[right][start_right..start_right + len_right],
                                &lists[left][start_left + len_left..],
                            );
                            let right_view = ChunkView::three(
                                &lists[right][..start_right],
                                &lists[left][start_left..start_left + len_left],
                                &lists[right][start_right + len_right..],
                            );
                            let left_score = trial_list_score_view(&per, &state, left, &left_view, None, &mut scratch);
                            let right_score = trial_list_score_view(&per, &state, right, &right_view, None, &mut right_scratch);
                            let incremental_total = score_with_replacements(
                                &per,
                                &state,
                                &[
                                    TrialList { list: left, score: &left_score, contents: &left_view },
                                    TrialList { list: right, score: &right_score, contents: &right_view },
                                ],
                                0,
                                &mut score_scratch,
                            );
                            let mut materialized_left = Vec::new();
                            let mut materialized_right = Vec::new();
                            build_cross_exchange(
                                &mut materialized_left,
                                &mut materialized_right,
                                (&lists[left], start_left, len_left),
                                (&lists[right], start_right, len_right),
                            );
                            let mut full_lists = lists.to_vec();
                            full_lists[left] = materialized_left;
                            full_lists[right] = materialized_right;
                            let full_total = full_score_exact_lists(&per, &full_lists, state.global_viol);
                            assert!(incremental_total == full_total, "cross-exchange incremental total differs from full recomputation");
                            checked += 1;
                        }
                    }
                }
            }
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

fn candidate_swap(per: &PerList, state: &State, a: usize, a_pos: usize, b: usize, b_pos: usize) -> bool {
    let a_item = state.lists[a][a_pos];
    let b_item = state.lists[b][b_pos];
    match (route_before_after(per, state, a, a_pos, 1), route_before_after(per, state, b, b_pos, 1)) {
        (Some((a_before, a_after)), Some((b_before, b_after))) => {
            candidate_edge(per, a_before, b_item)
                || candidate_edge(per, b_item, a_after)
                || candidate_edge(per, b_before, a_item)
                || candidate_edge(per, a_item, b_after)
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
pub(super) fn best_improving_move(
    per: &PerList,
    state: &State,
    stop: &AtomicBool,
    memory: &mut SearchMemory,
    scan_seed: u64,
) -> Option<Move> {
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
    let interchangeable_lists = per.interchangeable_lists.get();
    let active_routes = active_list_indices(&state.lists, interchangeable_lists);
    let sources = rotated_routes(
        active_routes.iter().copied().filter(|&list| !interchangeable_lists || !state.lists[list].is_empty()).collect(),
        scan_seed,
    );
    let mut scratch_a = EvalScratch::default();
    let mut scratch_b = EvalScratch::default();
    let mut score_scratch = EvalScratch::default();
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
    for src in sources {
        if memory.skip(src) {
            continue;
        }
        let destination_seed = if scan_seed == 0 { 0 } else { scan_seed ^ mix64(src as u64) };
        let destinations = rotated_routes(active_routes.clone(), destination_seed);
        // --- Relocate a single item out of `src`. ---
        for src_pos in 0..state.lists[src].len() {
            let item = state.lists[src][src_pos];
            for &dst in &destinations {
                if dst == src {
                    let len = state.lists[src].len();
                    for dst_pos in 0..len {
                        if dst_pos == src_pos {
                            continue;
                        }
                        if stopped() {
                            return None;
                        }
                        let edit = Edit::MoveWithin { from: src_pos, to: dst_pos };
                        let nl = trial_list_score(per, state, src, edit, &mut scratch_a);
                        let view = EditView::new(&state.lists[src], edit);
                        // Within-list move: no item changes list, so gdelta = 0.
                        if score_one(per, state, src, &nl, &view, 0, &mut score_scratch) < current {
                            return Some(Move::Relocate { src, src_pos, dst, dst_pos });
                        }
                    }
                } else {
                    let source_edit = Edit::Remove { pos: src_pos };
                    let na = trial_list_score(per, state, src, source_edit, &mut scratch_a);
                    let source_view = EditView::new(&state.lists[src], source_edit);
                    let gd = per.globals.delta(&state.item_list, &[(item, dst)]);
                    for dst_pos in 0..=state.lists[dst].len() {
                        if stopped() {
                            return None;
                        }
                        if use_candidates && !candidate_segment_insert(per, state, dst, dst_pos, item, item) {
                            continue;
                        }
                        let destination_edit = Edit::Insert { pos: dst_pos, item };
                        let nb = trial_list_score(per, state, dst, destination_edit, &mut scratch_b);
                        let destination_view = EditView::new(&state.lists[dst], destination_edit);
                        if score_two(
                            per,
                            state,
                            TrialList { list: src, score: &na, contents: &source_view },
                            TrialList { list: dst, score: &nb, contents: &destination_view },
                            gd,
                            &mut score_scratch,
                        ) < current
                        {
                            return Some(Move::Relocate { src, src_pos, dst, dst_pos });
                        }
                    }
                }
            }
        }
        // --- Swap one item of `src` with one item of another route. ---
        for &y in &destinations {
            if y == src {
                continue;
            }
            for xp in 0..state.lists[src].len() {
                for yp in 0..state.lists[y].len() {
                    if stopped() {
                        return None;
                    }
                    let (vx, vy) = (state.lists[src][xp], state.lists[y][yp]);
                    // A swap removes the two items from their current
                    // neighbourhoods. Its promising edges are therefore the four
                    // adjacencies it creates, not the edge between the swapped
                    // items, which does not appear in the resulting routes.
                    // Candidate pruning is disabled while infeasible so the full
                    // membership neighbourhood remains available for load repair.
                    if use_candidates && !candidate_swap(per, state, src, xp, y, yp) {
                        continue;
                    }
                    let left_edit = Edit::Replace { pos: xp, item: vy };
                    let right_edit = Edit::Replace { pos: yp, item: vx };
                    let na = trial_list_score(per, state, src, left_edit, &mut scratch_a);
                    let nb = trial_list_score(per, state, y, right_edit, &mut scratch_b);
                    let left_view = EditView::new(&state.lists[src], left_edit);
                    let right_view = EditView::new(&state.lists[y], right_edit);
                    let gd = per.globals.delta(&state.item_list, &[(vx, y), (vy, src)]);
                    if score_two(
                        per,
                        state,
                        TrialList { list: src, score: &na, contents: &left_view },
                        TrialList { list: y, score: &nb, contents: &right_view },
                        gd,
                        &mut score_scratch,
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
                    for &dst in &destinations {
                        if dst == src {
                            let post_len = src_len - len;
                            for dst_pos in 0..=post_len {
                                if dst_pos == start {
                                    continue;
                                }
                                if stopped() {
                                    return None;
                                }
                                let edit = Edit::SegmentMoveWithin { start, len, to: dst_pos };
                                let nl = trial_list_score(per, state, src, edit, &mut scratch_a);
                                let view = EditView::new(&state.lists[src], edit);
                                if score_one(per, state, src, &nl, &view, 0, &mut score_scratch) < current {
                                    return Some(Move::OrOpt { src, start, len, dst, dst_pos });
                                }
                            }
                        } else {
                            let source_edit = Edit::SegmentRemove { start, len };
                            let na = trial_list_score(per, state, src, source_edit, &mut scratch_a);
                            let source_view = EditView::new(&state.lists[src], source_edit);
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
                                let destination_edit = Edit::SegmentInsert { pos: dst_pos, items, len };
                                let nb = trial_list_score(per, state, dst, destination_edit, &mut scratch_b);
                                let destination_view = EditView::new(&state.lists[dst], destination_edit);
                                if score_two(
                                    per,
                                    state,
                                    TrialList { list: src, score: &na, contents: &source_view },
                                    TrialList { list: dst, score: &nb, contents: &destination_view },
                                    gd,
                                    &mut score_scratch,
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
            for &y in &destinations {
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
                        let left_view = ChunkView::two(&state.lists[src][..cut_x], &state.lists[y][cut_y..]);
                        let right_view = ChunkView::two(&state.lists[y][..cut_y], &state.lists[src][cut_x..]);
                        let na = trial_list_score_view(per, state, src, &left_view, None, &mut scratch_a);
                        let nb = trial_list_score_view(per, state, y, &right_view, None, &mut scratch_b);
                        overrides.clear();
                        overrides.extend(state.lists[src][cut_x..].iter().map(|&item| (item, y)));
                        overrides.extend(state.lists[y][cut_y..].iter().map(|&item| (item, src)));
                        let gd = per.globals.delta(&state.item_list, &overrides);
                        if score_two(
                            per,
                            state,
                            TrialList { list: src, score: &na, contents: &left_view },
                            TrialList { list: y, score: &nb, contents: &right_view },
                            gd,
                            &mut score_scratch,
                        ) < current
                        {
                            return Some(Move::TwoOptStar { a: src, cut_a: cut_x, b: y, cut_b: cut_y });
                        }
                    }
                }
            }
            // Cross-exchange: swap a segment of `src` with a segment of another route.
            for &y in &destinations {
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
                                let left_view = ChunkView::three(
                                    &state.lists[src][..start_x],
                                    &state.lists[y][start_y..start_y + len_y],
                                    &state.lists[src][start_x + len_x..],
                                );
                                let right_view = ChunkView::three(
                                    &state.lists[y][..start_y],
                                    &state.lists[src][start_x..start_x + len_x],
                                    &state.lists[y][start_y + len_y..],
                                );
                                let na = trial_list_score_view(per, state, src, &left_view, None, &mut scratch_a);
                                let nb = trial_list_score_view(per, state, y, &right_view, None, &mut scratch_b);
                                overrides.clear();
                                overrides.extend(state.lists[src][start_x..start_x + len_x].iter().map(|&item| (item, y)));
                                overrides.extend(state.lists[y][start_y..start_y + len_y].iter().map(|&item| (item, src)));
                                let gd = per.globals.delta(&state.item_list, &overrides);
                                if score_two(
                                    per,
                                    state,
                                    TrialList { list: src, score: &na, contents: &left_view },
                                    TrialList { list: y, score: &nb, contents: &right_view },
                                    gd,
                                    &mut score_scratch,
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
                    let edit = Edit::Reverse { i, j };
                    let nl = trial_list_score(per, state, src, edit, &mut scratch_a);
                    let view = EditView::new(&state.lists[src], edit);
                    if score_one(per, state, src, &nl, &view, 0, &mut score_scratch) < current {
                        return Some(Move::Reverse { list: src, i, j });
                    }
                }
            }
        }
        memory.mark_inactive(src);
    }
    None
}

pub(super) fn apply_move(per: &PerList, state: &mut State, mv: Move, stop: &AtomicBool) -> bool {
    match mv {
        Move::Relocate { src, src_pos, dst, dst_pos } => {
            let item = state.lists[src].remove(src_pos);
            let pos = dst_pos.min(state.lists[dst].len());
            state.lists[dst].insert(pos, item);
            if !state.rescore_interruptible(per, src, stop) || !state.rescore_interruptible(per, dst, stop) {
                return false;
            }
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
            if !state.rescore_interruptible(per, src, stop) {
                return false;
            }
            if src != dst {
                if !state.rescore_interruptible(per, dst, stop) {
                    return false;
                }
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
            if !state.rescore_interruptible(per, a, stop) || !state.rescore_interruptible(per, b, stop) {
                return false;
            }
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
            if !state.rescore_interruptible(per, a, stop) || !state.rescore_interruptible(per, b, stop) {
                return false;
            }
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
            if !state.rescore_interruptible(per, a, stop) || !state.rescore_interruptible(per, b, stop) {
                return false;
            }
            // After the swap, lists[a][a_pos] holds the item that was in b.
            state.set_item_list(per, state.lists[a][a_pos], a);
            state.set_item_list(per, state.lists[b][b_pos], b);
            state.global_viol = per.globals.total(&state.item_list);
        }
        Move::Reverse { list, i, j } => {
            state.lists[list][i..=j].reverse();
            if !state.rescore_interruptible(per, list, stop) {
                return false;
            }
            // Reversal keeps every item in the same list, so globals are unchanged.
        }
    }
    true
}

pub(super) fn shuffle(order: &mut [usize], seed: u64) {
    for i in (1..order.len()).rev() {
        let j = (mix64(seed.wrapping_add(i as u64)) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}
