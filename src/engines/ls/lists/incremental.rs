use super::eval::eval_expr;
use crate::model::list::{Iterable, ReduceOp, Reduction};

/// Read-only speculative list. Implementations compose existing slices or a
/// local edit, so candidate evaluation never needs to copy a complete list.
pub(super) trait ListView {
    fn len(&self) -> usize;
    fn at(&self, index: usize) -> i32;

    fn common_prefix_len(&self, old: &[i32]) -> usize {
        let limit = self.len().min(old.len());
        let mut prefix = 0;
        while prefix < limit && self.at(prefix) == old[prefix] {
            prefix += 1;
        }
        prefix
    }

    fn common_suffix_len(&self, old: &[i32], prefix: usize) -> usize {
        let limit = self.len().min(old.len()).saturating_sub(prefix);
        let mut suffix = 0;
        while suffix < limit && self.at(self.len() - 1 - suffix) == old[old.len() - 1 - suffix] {
            suffix += 1;
        }
        suffix
    }
}

impl ListView for [i32] {
    fn len(&self) -> usize {
        <[i32]>::len(self)
    }

    fn at(&self, index: usize) -> i32 {
        self[index]
    }
}

impl ListView for Vec<i32> {
    fn len(&self) -> usize {
        self.len()
    }

    fn at(&self, index: usize) -> i32 {
        self[index]
    }
}

/// Speculative view with one position removed from an accepted list.
pub(super) struct RemoveView<'a> {
    base: &'a [i32],
    pos: usize,
}

impl<'a> RemoveView<'a> {
    pub(super) fn new(base: &'a [i32], pos: usize) -> Self {
        debug_assert!(pos < base.len());
        Self { base, pos }
    }
}

impl ListView for RemoveView<'_> {
    fn len(&self) -> usize {
        self.base.len() - 1
    }

    fn at(&self, index: usize) -> i32 {
        if index < self.pos {
            self.base[index]
        } else {
            self.base[index + 1]
        }
    }

    fn common_prefix_len(&self, _old: &[i32]) -> usize {
        self.pos
    }

    fn common_suffix_len(&self, _old: &[i32], _prefix: usize) -> usize {
        self.base.len() - self.pos - 1
    }
}

pub(super) struct InsertView<'a> {
    base: &'a [i32],
    pos: usize,
    item: i32,
}

impl<'a> InsertView<'a> {
    pub(super) fn new(base: &'a [i32], pos: usize, item: i32) -> Self {
        Self { base, pos: pos.min(base.len()), item }
    }
}

impl ListView for InsertView<'_> {
    fn len(&self) -> usize {
        self.base.len() + 1
    }

    fn at(&self, index: usize) -> i32 {
        if index < self.pos {
            self.base[index]
        } else if index == self.pos {
            self.item
        } else {
            self.base[index - 1]
        }
    }

    fn common_prefix_len(&self, _old: &[i32]) -> usize {
        self.pos
    }

    fn common_suffix_len(&self, _old: &[i32], _prefix: usize) -> usize {
        self.base.len() - self.pos
    }
}

/// Reusable value buffers for one speculative reduction. Their capacity grows
/// to the largest affected span seen by a neighbourhood pass, not per candidate.
#[derive(Default)]
pub(super) struct EvalScratch {
    removed: Vec<i64>,
    added: Vec<i64>,
    members: Vec<i32>,
    recomputed_scan_steps: u64,
    sum_value: Option<i64>,
}

impl EvalScratch {
    fn clear(&mut self) {
        self.removed.clear();
        self.added.clear();
        self.members.clear();
        self.recomputed_scan_steps = 0;
        self.sum_value = None;
    }

    pub(super) fn recomputed_scan_steps(&self) -> u64 {
        self.recomputed_scan_steps
    }
}

/// Accepted-state cache for one reduction. `outputs` is kept in iterable order,
/// `sorted` supports order statistics after removing/adding only affected
/// outputs, and `scan_acc` stores the accumulator before every scan position.
pub(super) struct ReductionCache {
    raw: Option<i64>,
    value: Option<i64>,
    outputs: Vec<i64>,
    sorted: Vec<i64>,
    scan_acc: Vec<i64>,
    sum_prefix: Vec<i64>,
    sum_suffix: Vec<SatTransform>,
    sum_ranges: Option<SumRangeCache>,
}

impl ReductionCache {
    pub(super) fn build(reduction: &Reduction, contents: &[i32]) -> Self {
        let mut outputs = Vec::new();
        let mut scan_acc = Vec::new();
        emit_slice(reduction, contents, &mut outputs, &mut scan_acc);
        let raw = aggregate(reduction.op, &outputs);
        let value = raw.map(|value| value.saturating_mul(reduction.coeff));
        let mut sorted =
            if matches!(reduction.op, ReduceOp::Min | ReduceOp::Max | ReduceOp::SelectKth(_)) { outputs.clone() } else { Vec::new() };
        sorted.sort_unstable();
        let mut sum_prefix = Vec::with_capacity(outputs.len() + 1);
        sum_prefix.push(0);
        for &output in &outputs {
            let next = sum_prefix.last().copied().unwrap_or(0i64).saturating_add(output);
            sum_prefix.push(next);
        }
        let mut sum_suffix = vec![SatTransform::identity(); outputs.len() + 1];
        for index in (0..outputs.len()).rev() {
            sum_suffix[index] = SatTransform::value(outputs[index]).then(sum_suffix[index + 1]);
        }
        let sum_ranges = (matches!(reduction.op, ReduceOp::Sum) && matches!(reduction.iterable, Iterable::Pairs(_)))
            .then(|| SumRangeCache::build(&outputs));
        Self { raw, value, outputs, sorted, scan_acc, sum_prefix, sum_suffix, sum_ranges }
    }

    pub(super) fn value(&self) -> Option<i64> {
        self.value
    }

    pub(super) fn raw_value(&self) -> Option<i64> {
        self.raw
    }

    /// Evaluate `reduction` on `candidate` by retaining all unaffected emitted
    /// values from this cache and recomputing only the changed span.
    pub(super) fn candidate_value(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &dyn ListView,
        scratch: &mut EvalScratch,
    ) -> Option<i64> {
        scratch.clear();
        let prefix = candidate.common_prefix_len(old);
        if old.len() == candidate.len() && prefix == old.len() {
            return self.value;
        }
        let suffix = candidate.common_suffix_len(old, prefix);
        match &reduction.iterable {
            Iterable::Items(_) => self.items_delta(reduction, old, candidate, prefix, suffix, scratch),
            Iterable::SetItems(_) => self.set_items_delta(reduction, candidate, scratch),
            Iterable::Edges { start, end, .. } => self.edges_delta(reduction, old, candidate, prefix, suffix, *start, *end, scratch),
            Iterable::Pairs(_) => self.pairs_delta(reduction, old, candidate, scratch),
            Iterable::Scan { init: _, boundary, step, end, .. } => {
                self.scan_delta(reduction, old, candidate, prefix, *boundary, *step, *end, scratch)
            }
            Iterable::Windows { size, inner, .. } => self.windows_delta(reduction, old, candidate, prefix, suffix, *size, *inner, scratch),
        }
        candidate_aggregate(self, reduction, scratch).map(|value| value.saturating_mul(reduction.coeff))
    }

    fn items_delta(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &dyn ListView,
        prefix: usize,
        suffix: usize,
        scratch: &mut EvalScratch,
    ) {
        let old_end = old.len().saturating_sub(suffix);
        scratch.removed.extend_from_slice(&self.outputs[prefix..old_end]);
        let new_end = candidate.len().saturating_sub(suffix);
        for pos in prefix..new_end {
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(candidate.at(pos))]));
        }
        self.finish_contiguous_sum(prefix, old_end, scratch);
    }

    fn set_items_delta(&self, reduction: &Reduction, candidate: &dyn ListView, scratch: &mut EvalScratch) {
        scratch.removed.extend_from_slice(&self.outputs);
        scratch.members.extend((0..candidate.len()).map(|index| candidate.at(index)));
        scratch.members.sort_unstable();
        for &item in &scratch.members {
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(item)]));
        }
        if matches!(reduction.op, ReduceOp::Sum) {
            scratch.sum_value = Some(scratch.added.iter().fold(0i64, |sum, &value| sum.saturating_add(value)));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn edges_delta(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &dyn ListView,
        prefix: usize,
        suffix: usize,
        start: i32,
        end: i32,
        scratch: &mut EvalScratch,
    ) {
        let old_affected_end = self.outputs.len().saturating_sub(suffix);
        scratch.removed.extend_from_slice(&self.outputs[prefix..old_affected_end]);
        let new_outputs = candidate.len().saturating_add(1);
        let new_affected_end = new_outputs.saturating_sub(suffix);
        for edge in prefix..new_affected_end {
            let (from, to) = edge_nodes(candidate, edge, start, end);
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(from), i64::from(to)]));
        }
        self.finish_contiguous_sum(prefix, old_affected_end, scratch);
        let _ = old;
    }

    fn pairs_delta(&self, reduction: &Reduction, old: &[i32], candidate: &dyn ListView, scratch: &mut EvalScratch) {
        let old_len = old.len();
        let new_len = candidate.len();
        let unchanged = |pos: usize| pos < old_len && pos < new_len && old[pos] == candidate.at(pos);
        if matches!(reduction.op, ReduceOp::Sum) {
            let ranges = self.sum_ranges.as_ref().expect("pairs/sum reduction has a range cache");
            let mut sequence = SatTransform::identity();
            for i in 0..new_len {
                if unchanged(i) {
                    let row_start = i * old_len;
                    let mut retained_start = 0usize;
                    for j in 0..new_len {
                        if unchanged(j) {
                            continue;
                        }
                        if retained_start < j {
                            sequence = sequence.then(ranges.range(&self.outputs, row_start + retained_start, row_start + j));
                        }
                        let args = [i64::from(candidate.at(i)), i64::from(candidate.at(j)), i as i64, j as i64];
                        sequence = sequence.then(SatTransform::value(eval_expr(&reduction.arena.exprs, reduction.body, &args)));
                        retained_start = j + 1;
                    }
                    if retained_start < new_len {
                        sequence = sequence.then(ranges.range(&self.outputs, row_start + retained_start, row_start + new_len));
                    }
                } else {
                    for j in 0..new_len {
                        let args = [i64::from(candidate.at(i)), i64::from(candidate.at(j)), i as i64, j as i64];
                        sequence = sequence.then(SatTransform::value(eval_expr(&reduction.arena.exprs, reduction.body, &args)));
                    }
                }
            }
            scratch.sum_value = Some(sequence.apply(0));
            return;
        }
        let old_changed: Vec<usize> = (0..old_len).filter(|&pos| !unchanged(pos)).collect();
        for i in 0..old_len {
            if !unchanged(i) {
                scratch.removed.extend_from_slice(&self.outputs[i * old_len..(i + 1) * old_len]);
            } else {
                scratch.removed.extend(old_changed.iter().map(|&j| self.outputs[i * old_len + j]));
            }
        }
        let new_changed: Vec<usize> = (0..new_len).filter(|&pos| !unchanged(pos)).collect();
        for i in 0..new_len {
            if !unchanged(i) {
                for j in 0..new_len {
                    let args = [i64::from(candidate.at(i)), i64::from(candidate.at(j)), i as i64, j as i64];
                    scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &args));
                }
            } else {
                for &j in &new_changed {
                    let args = [i64::from(candidate.at(i)), i64::from(candidate.at(j)), i as i64, j as i64];
                    scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &args));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_delta(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &dyn ListView,
        prefix: usize,
        boundary: i32,
        step: crate::model::list::ExprId,
        end: Option<i32>,
        scratch: &mut EvalScratch,
    ) {
        scratch.removed.extend_from_slice(&self.outputs[prefix..]);
        let mut acc = self.scan_acc[prefix];
        let mut prev = if prefix == 0 { boundary } else { old[prefix - 1] };
        for pos in prefix..candidate.len() {
            let current = candidate.at(pos);
            let next = eval_expr(&reduction.arena.exprs, step, &[i64::from(current), acc, i64::from(prev)]);
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(current), next, i64::from(prev)]));
            acc = next;
            prev = current;
        }
        if let Some(end) = end {
            let next = eval_expr(&reduction.arena.exprs, step, &[i64::from(end), acc, i64::from(prev)]);
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(end), next, i64::from(prev)]));
        }
        scratch.recomputed_scan_steps = u64::try_from(scratch.added.len()).unwrap_or(u64::MAX);
        self.finish_contiguous_sum(prefix, self.outputs.len(), scratch);
    }

    #[allow(clippy::too_many_arguments)]
    fn windows_delta(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &dyn ListView,
        prefix: usize,
        suffix: usize,
        size: usize,
        inner: crate::model::list::ExprId,
        scratch: &mut EvalScratch,
    ) {
        let old_windows = window_count(old.len(), size);
        let new_windows = window_count(candidate.len(), size);
        let prefix_windows = window_count(prefix, size);
        let old_suffix_windows = window_count(suffix, size);
        let new_suffix_windows = window_count(suffix, size);
        let old_end = old_windows.saturating_sub(old_suffix_windows);
        scratch.removed.extend_from_slice(&self.outputs[prefix_windows..old_end]);
        let new_end = new_windows.saturating_sub(new_suffix_windows);
        for start in prefix_windows..new_end {
            scratch.added.push(window_value(reduction, candidate, start, size, inner));
        }
        self.finish_contiguous_sum(prefix_windows, old_end, scratch);
    }

    fn finish_contiguous_sum(&self, prefix_end: usize, suffix_start: usize, scratch: &mut EvalScratch) {
        let mut sum = self.sum_prefix[prefix_end];
        for &value in &scratch.added {
            sum = sum.saturating_add(value);
        }
        scratch.sum_value = Some(self.sum_suffix[suffix_start].apply(sum));
    }
}

fn emit_slice(reduction: &Reduction, contents: &[i32], outputs: &mut Vec<i64>, scan_acc: &mut Vec<i64>) {
    let arena = &reduction.arena.exprs;
    match reduction.iterable {
        Iterable::Items(_) => {
            outputs.extend(contents.iter().map(|&item| eval_expr(arena, reduction.body, &[i64::from(item)])));
        }
        Iterable::SetItems(_) => {
            let mut members = contents.to_vec();
            members.sort_unstable();
            outputs.extend(members.into_iter().map(|item| eval_expr(arena, reduction.body, &[i64::from(item)])));
        }
        Iterable::Edges { start, end, .. } => {
            for edge in 0..=contents.len() {
                let from = if edge == 0 { start } else { contents[edge - 1] };
                let to = if edge == contents.len() { end } else { contents[edge] };
                outputs.push(eval_expr(arena, reduction.body, &[i64::from(from), i64::from(to)]));
            }
        }
        Iterable::Pairs(_) => {
            for (i, &left) in contents.iter().enumerate() {
                for (j, &right) in contents.iter().enumerate() {
                    outputs.push(eval_expr(arena, reduction.body, &[i64::from(left), i64::from(right), i as i64, j as i64]));
                }
            }
        }
        Iterable::Scan { init, boundary, step, end, .. } => {
            let mut acc = init;
            let mut prev = boundary;
            scan_acc.push(acc);
            for &current in contents {
                let next = eval_expr(arena, step, &[i64::from(current), acc, i64::from(prev)]);
                outputs.push(eval_expr(arena, reduction.body, &[i64::from(current), next, i64::from(prev)]));
                acc = next;
                prev = current;
                scan_acc.push(acc);
            }
            if let Some(end) = end {
                let next = eval_expr(arena, step, &[i64::from(end), acc, i64::from(prev)]);
                outputs.push(eval_expr(arena, reduction.body, &[i64::from(end), next, i64::from(prev)]));
            }
        }
        Iterable::Windows { size, inner, .. } => {
            for start in 0..window_count(contents.len(), size) {
                outputs.push(window_value(reduction, contents, start, size, inner));
            }
        }
    }
}

fn candidate_aggregate(cache: &ReductionCache, reduction: &Reduction, scratch: &mut EvalScratch) -> Option<i64> {
    let new_len = cache.outputs.len().saturating_sub(scratch.removed.len()).saturating_add(scratch.added.len());
    match reduction.op {
        ReduceOp::Sum => {
            if let Some(sum) = scratch.sum_value {
                return Some(sum);
            }
            let removed = scratch.removed.iter().fold(0i64, |sum, &value| sum.saturating_add(value));
            let added = scratch.added.iter().fold(0i64, |sum, &value| sum.saturating_add(value));
            Some(cache.raw?.saturating_sub(removed).saturating_add(added))
        }
        ReduceOp::Count => {
            let removed = scratch.removed.iter().filter(|&&value| value != 0).count();
            let added = scratch.added.iter().filter(|&&value| value != 0).count();
            let raw = cache.raw?;
            Some(raw.saturating_sub(removed as i64).saturating_add(added as i64))
        }
        ReduceOp::Used => Some(i64::from(new_len > 0)),
        ReduceOp::Min => select_after_changes(&cache.sorted, &mut scratch.removed, &mut scratch.added, 0),
        ReduceOp::Max => {
            if new_len == 0 {
                None
            } else {
                select_after_changes(&cache.sorted, &mut scratch.removed, &mut scratch.added, new_len - 1)
            }
        }
        ReduceOp::SelectKth(k) => {
            if k >= new_len {
                None
            } else {
                select_after_changes(&cache.sorted, &mut scratch.removed, &mut scratch.added, k)
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SatTransform {
    offset: i128,
    low: i64,
    high: i64,
}

impl SatTransform {
    const fn identity() -> Self {
        Self { offset: 0, low: i64::MIN, high: i64::MAX }
    }

    fn value(value: i64) -> Self {
        Self { offset: i128::from(value), low: i64::MIN.saturating_add(value), high: i64::MAX.saturating_add(value) }
    }

    fn then(self, next: Self) -> Self {
        Self { offset: self.offset.saturating_add(next.offset), low: next.apply(self.low), high: next.apply(self.high) }
    }

    fn apply(self, value: i64) -> i64 {
        let shifted = i128::from(value).saturating_add(self.offset);
        shifted.clamp(i128::from(self.low), i128::from(self.high)) as i64
    }
}

/// Compact block cache over ordered saturating additions. It lets Pairs/Sum
/// splice unchanged row fragments without visiting every retained pair, while
/// adding only one transform per block instead of a full segment tree.
struct SumRangeCache {
    blocks: Vec<SatTransform>,
}

impl SumRangeCache {
    const BLOCK_SIZE: usize = 32;

    fn build(values: &[i64]) -> Self {
        let blocks = values
            .chunks(Self::BLOCK_SIZE)
            .map(|chunk| chunk.iter().fold(SatTransform::identity(), |transform, &value| transform.then(SatTransform::value(value))))
            .collect();
        Self { blocks }
    }

    fn range(&self, values: &[i64], mut start: usize, end: usize) -> SatTransform {
        debug_assert!(start <= end);
        debug_assert!(end <= values.len());
        let mut transform = SatTransform::identity();
        while start < end && !start.is_multiple_of(Self::BLOCK_SIZE) {
            transform = transform.then(SatTransform::value(values[start]));
            start += 1;
        }
        while start.saturating_add(Self::BLOCK_SIZE) <= end {
            transform = transform.then(self.blocks[start / Self::BLOCK_SIZE]);
            start += Self::BLOCK_SIZE;
        }
        while start < end {
            transform = transform.then(SatTransform::value(values[start]));
            start += 1;
        }
        transform
    }
}

fn aggregate(op: ReduceOp, outputs: &[i64]) -> Option<i64> {
    match op {
        ReduceOp::Sum => Some(outputs.iter().fold(0i64, |sum, &value| sum.saturating_add(value))),
        ReduceOp::Count => Some(outputs.iter().filter(|&&value| value != 0).count() as i64),
        ReduceOp::Used => Some(i64::from(!outputs.is_empty())),
        ReduceOp::Min => outputs.iter().copied().min(),
        ReduceOp::Max => outputs.iter().copied().max(),
        ReduceOp::SelectKth(k) => {
            if k >= outputs.len() {
                None
            } else {
                let mut values = outputs.to_vec();
                values.select_nth_unstable(k);
                Some(values[k])
            }
        }
    }
}

fn select_after_changes(base: &[i64], removed: &mut [i64], added: &mut [i64], target: usize) -> Option<i64> {
    removed.sort_unstable();
    added.sort_unstable();
    let new_len = base.len().saturating_sub(removed.len()).saturating_add(added.len());
    if target >= new_len {
        return None;
    }

    // Min and Max are the hot cases. Walk only duplicates removed at the
    // corresponding edge of the accepted sorted cache.
    if target == 0 {
        let (mut base_pos, mut removed_pos) = (0usize, 0usize);
        while base_pos < base.len() && removed_pos < removed.len() && base[base_pos] == removed[removed_pos] {
            base_pos += 1;
            removed_pos += 1;
        }
        return match (base.get(base_pos), added.first()) {
            (Some(&left), Some(&right)) => Some(left.min(right)),
            (Some(&value), None) | (None, Some(&value)) => Some(value),
            (None, None) => None,
        };
    }
    if target + 1 == new_len {
        let (mut base_pos, mut removed_pos) = (base.len(), removed.len());
        while base_pos > 0 && removed_pos > 0 && base[base_pos - 1] == removed[removed_pos - 1] {
            base_pos -= 1;
            removed_pos -= 1;
        }
        return match (base_pos.checked_sub(1).and_then(|pos| base.get(pos)), added.last()) {
            (Some(&left), Some(&right)) => Some(left.max(right)),
            (Some(&value), None) | (None, Some(&value)) => Some(value),
            (None, None) => None,
        };
    }

    // An order statistic is found by value-domain binary search. Rank queries
    // use the three sorted multisets, so the cost is logarithmic in list size
    // instead of scanning up to k retained values.
    let first = match (base.first(), added.first()) {
        (Some(&left), Some(&right)) => left.min(right),
        (Some(&value), None) | (None, Some(&value)) => value,
        (None, None) => return None,
    };
    let last = match (base.last(), added.last()) {
        (Some(&left), Some(&right)) => left.max(right),
        (Some(&value), None) | (None, Some(&value)) => value,
        (None, None) => return None,
    };
    let mut low = i128::from(first);
    let mut high = i128::from(last);
    while low < high {
        let middle = low + (high - low) / 2;
        let middle = middle as i64;
        let base_count = base.partition_point(|&value| value <= middle);
        let removed_count = removed.partition_point(|&value| value <= middle);
        let added_count = added.partition_point(|&value| value <= middle);
        if base_count.saturating_sub(removed_count).saturating_add(added_count) > target {
            high = i128::from(middle);
        } else {
            low = i128::from(middle) + 1;
        }
    }
    Some(low as i64)
}

fn edge_nodes(view: &dyn ListView, edge: usize, start: i32, end: i32) -> (i32, i32) {
    let from = if edge == 0 { start } else { view.at(edge - 1) };
    let to = if edge == view.len() { end } else { view.at(edge) };
    (from, to)
}

fn window_count(len: usize, size: usize) -> usize {
    if size > 0 && len >= size {
        len - size + 1
    } else {
        0
    }
}

fn window_value(
    reduction: &Reduction,
    view: &(impl ListView + ?Sized),
    start: usize,
    size: usize,
    inner: crate::model::list::ExprId,
) -> i64 {
    let mut total = 0i64;
    for pos in start..start + size {
        total = total.saturating_add(eval_expr(&reduction.arena.exprs, inner, &[i64::from(view.at(pos))]));
    }
    eval_expr(&reduction.arena.exprs, reduction.body, &[0, total])
}
