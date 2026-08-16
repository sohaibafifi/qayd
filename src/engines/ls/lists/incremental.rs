use std::sync::atomic::{AtomicBool, Ordering};

use super::eval::eval_expr;
use crate::model::list::{Iterable, ReduceOp, Reduction};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct EvaluationInterrupted;

type EvaluationResult<T> = Result<T, EvaluationInterrupted>;

const STOP_POLL_INTERVAL: usize = 64;

fn poll_stop(stop: &AtomicBool, work: usize) -> EvaluationResult<()> {
    if work.is_multiple_of(STOP_POLL_INTERVAL) && stop.load(Ordering::Relaxed) {
        Err(EvaluationInterrupted)
    } else {
        Ok(())
    }
}

fn extend_interruptible<T: Copy>(destination: &mut Vec<T>, source: &[T], stop: &AtomicBool) -> EvaluationResult<()> {
    for (chunk_at, chunk) in source.chunks(1024).enumerate() {
        poll_stop(stop, chunk_at.saturating_mul(1024))?;
        destination.extend_from_slice(chunk);
    }
    poll_stop(stop, 0)
}

fn sort_interruptible<T: Copy + Default + Ord>(values: &mut [T], stop: &AtomicBool) -> EvaluationResult<()> {
    if values.len() < 2 {
        return poll_stop(stop, 0);
    }
    let mut source = values.to_vec();
    let mut target = vec![T::default(); values.len()];
    let mut width = 1usize;
    let mut work = 0usize;
    while width < source.len() {
        for start in (0..source.len()).step_by(width.saturating_mul(2)) {
            poll_stop(stop, work)?;
            let middle = start.saturating_add(width).min(source.len());
            let end = middle.saturating_add(width).min(source.len());
            let (mut left, mut right, mut output) = (start, middle, start);
            while left < middle && right < end {
                poll_stop(stop, work)?;
                if source[left] <= source[right] {
                    target[output] = source[left];
                    left += 1;
                } else {
                    target[output] = source[right];
                    right += 1;
                }
                output += 1;
                work = work.saturating_add(1);
            }
            while left < middle {
                poll_stop(stop, work)?;
                target[output] = source[left];
                left += 1;
                output += 1;
                work = work.saturating_add(1);
            }
            while right < end {
                poll_stop(stop, work)?;
                target[output] = source[right];
                right += 1;
                output += 1;
                work = work.saturating_add(1);
            }
        }
        std::mem::swap(&mut source, &mut target);
        width = width.saturating_mul(2);
    }
    for (chunk_at, (destination, sorted)) in values.chunks_mut(1024).zip(source.chunks(1024)).enumerate() {
        poll_stop(stop, chunk_at.saturating_mul(1024))?;
        destination.copy_from_slice(sorted);
    }
    poll_stop(stop, 0)
}

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

    fn common_prefix_len_interruptible(&self, old: &[i32], stop: &AtomicBool) -> EvaluationResult<usize> {
        let limit = self.len().min(old.len());
        let mut prefix = 0;
        while prefix < limit && self.at(prefix) == old[prefix] {
            poll_stop(stop, prefix)?;
            prefix += 1;
        }
        poll_stop(stop, prefix)?;
        Ok(prefix)
    }

    fn common_suffix_len_interruptible(&self, old: &[i32], prefix: usize, stop: &AtomicBool) -> EvaluationResult<usize> {
        let limit = self.len().min(old.len()).saturating_sub(prefix);
        let mut suffix = 0;
        while suffix < limit && self.at(self.len() - 1 - suffix) == old[old.len() - 1 - suffix] {
            poll_stop(stop, suffix)?;
            suffix += 1;
        }
        poll_stop(stop, suffix)?;
        Ok(suffix)
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

    fn common_prefix_len_interruptible(&self, _old: &[i32], stop: &AtomicBool) -> EvaluationResult<usize> {
        poll_stop(stop, 0)?;
        Ok(self.pos)
    }

    fn common_suffix_len_interruptible(&self, _old: &[i32], _prefix: usize, stop: &AtomicBool) -> EvaluationResult<usize> {
        poll_stop(stop, 0)?;
        Ok(self.base.len() - self.pos - 1)
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

    fn common_prefix_len_interruptible(&self, _old: &[i32], stop: &AtomicBool) -> EvaluationResult<usize> {
        poll_stop(stop, 0)?;
        Ok(self.pos)
    }

    fn common_suffix_len_interruptible(&self, _old: &[i32], _prefix: usize, stop: &AtomicBool) -> EvaluationResult<usize> {
        poll_stop(stop, 0)?;
        Ok(self.base.len() - self.pos)
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
        let stop = AtomicBool::new(false);
        Self::build_interruptible(reduction, contents, &stop).expect("an uninterrupted cache build must complete")
    }

    pub(super) fn build_interruptible(reduction: &Reduction, contents: &[i32], stop: &AtomicBool) -> Option<Self> {
        let mut outputs = Vec::new();
        let mut scan_acc = Vec::new();
        if !emit_slice(reduction, contents, &mut outputs, &mut scan_acc, stop) {
            return None;
        }
        let raw = aggregate(reduction.op, &outputs);
        let value = raw.map(|value| value.saturating_mul(reduction.coeff));
        let mut sorted =
            if matches!(reduction.op, ReduceOp::Min | ReduceOp::Max | ReduceOp::SelectKth(_)) { outputs.clone() } else { Vec::new() };
        sorted.sort_unstable();
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let mut sum_prefix = Vec::with_capacity(outputs.len() + 1);
        sum_prefix.push(0);
        for (index, &output) in outputs.iter().enumerate() {
            if index.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                return None;
            }
            let next = sum_prefix.last().copied().unwrap_or(0i64).saturating_add(output);
            sum_prefix.push(next);
        }
        let mut sum_suffix = vec![SatTransform::identity(); outputs.len() + 1];
        for index in (0..outputs.len()).rev() {
            if index.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                return None;
            }
            sum_suffix[index] = SatTransform::value(outputs[index]).then(sum_suffix[index + 1]);
        }
        let sum_ranges = if matches!(reduction.op, ReduceOp::Sum) && matches!(reduction.iterable, Iterable::Pairs(_)) {
            Some(SumRangeCache::build_interruptible(&outputs, stop)?)
        } else {
            None
        };
        Some(Self { raw, value, outputs, sorted, scan_acc, sum_prefix, sum_suffix, sum_ranges })
    }

    pub(super) fn value(&self) -> Option<i64> {
        self.value
    }

    pub(super) fn raw_value(&self) -> Option<i64> {
        self.raw
    }

    /// Evaluate `reduction` on `candidate` by retaining all unaffected emitted
    /// values from this cache and recomputing only the changed span.
    pub(super) fn candidate_value_interruptible(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &(impl ListView + ?Sized),
        scratch: &mut EvalScratch,
        stop: &AtomicBool,
    ) -> EvaluationResult<Option<i64>> {
        scratch.clear();
        poll_stop(stop, 0)?;
        let prefix = candidate.common_prefix_len_interruptible(old, stop)?;
        if old.len() == candidate.len() && prefix == old.len() {
            return Ok(self.value);
        }
        let suffix = candidate.common_suffix_len_interruptible(old, prefix, stop)?;
        match &reduction.iterable {
            Iterable::Items(_) => self.items_delta(reduction, old, candidate, prefix, suffix, scratch, stop)?,
            Iterable::SetItems(_) => self.set_items_delta(reduction, candidate, scratch, stop)?,
            Iterable::Edges { start, end, .. } => {
                self.edges_delta(reduction, old, candidate, prefix, suffix, *start, *end, scratch, stop)?;
            }
            Iterable::Pairs(_) => self.pairs_delta(reduction, old, candidate, scratch, stop)?,
            Iterable::Scan { init: _, boundary, step, end, .. } => {
                self.scan_delta(reduction, old, candidate, prefix, suffix, *boundary, *step, *end, scratch, stop)?;
            }
            Iterable::Windows { size, inner, .. } => {
                self.windows_delta(reduction, old, candidate, prefix, suffix, *size, *inner, scratch, stop)?;
            }
        }
        candidate_aggregate_interruptible(self, reduction, scratch, stop)
            .map(|value| value.map(|value| value.saturating_mul(reduction.coeff)))
    }

    #[allow(clippy::too_many_arguments)]
    fn items_delta(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &(impl ListView + ?Sized),
        prefix: usize,
        suffix: usize,
        scratch: &mut EvalScratch,
        stop: &AtomicBool,
    ) -> EvaluationResult<()> {
        let old_end = old.len().saturating_sub(suffix);
        extend_interruptible(&mut scratch.removed, &self.outputs[prefix..old_end], stop)?;
        let new_end = candidate.len().saturating_sub(suffix);
        for (work, pos) in (prefix..new_end).enumerate() {
            poll_stop(stop, work)?;
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(candidate.at(pos))]));
        }
        self.finish_contiguous_sum(prefix, old_end, scratch, stop)
    }

    fn set_items_delta(
        &self,
        reduction: &Reduction,
        candidate: &(impl ListView + ?Sized),
        scratch: &mut EvalScratch,
        stop: &AtomicBool,
    ) -> EvaluationResult<()> {
        extend_interruptible(&mut scratch.removed, &self.outputs, stop)?;
        for index in 0..candidate.len() {
            poll_stop(stop, index)?;
            scratch.members.push(candidate.at(index));
        }
        sort_interruptible(&mut scratch.members, stop)?;
        for (index, &item) in scratch.members.iter().enumerate() {
            poll_stop(stop, index)?;
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(item)]));
        }
        if matches!(reduction.op, ReduceOp::Sum) {
            let mut sum = 0i64;
            for (index, &value) in scratch.added.iter().enumerate() {
                poll_stop(stop, index)?;
                sum = sum.saturating_add(value);
            }
            scratch.sum_value = Some(sum);
        }
        poll_stop(stop, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn edges_delta(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &(impl ListView + ?Sized),
        prefix: usize,
        suffix: usize,
        start: i32,
        end: i32,
        scratch: &mut EvalScratch,
        stop: &AtomicBool,
    ) -> EvaluationResult<()> {
        let old_affected_end = self.outputs.len().saturating_sub(suffix);
        extend_interruptible(&mut scratch.removed, &self.outputs[prefix..old_affected_end], stop)?;
        let new_outputs = candidate.len().saturating_add(1);
        let new_affected_end = new_outputs.saturating_sub(suffix);
        for (work, edge) in (prefix..new_affected_end).enumerate() {
            poll_stop(stop, work)?;
            let (from, to) = edge_nodes(candidate, edge, start, end);
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(from), i64::from(to)]));
        }
        self.finish_contiguous_sum(prefix, old_affected_end, scratch, stop)?;
        let _ = old;
        Ok(())
    }

    fn pairs_delta(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &(impl ListView + ?Sized),
        scratch: &mut EvalScratch,
        stop: &AtomicBool,
    ) -> EvaluationResult<()> {
        let old_len = old.len();
        let new_len = candidate.len();
        let unchanged = |pos: usize| pos < old_len && pos < new_len && old[pos] == candidate.at(pos);
        let mut work = 0usize;
        if matches!(reduction.op, ReduceOp::Sum) {
            let ranges = self.sum_ranges.as_ref().expect("pairs/sum reduction has a range cache");
            let mut sequence = SatTransform::identity();
            for i in 0..new_len {
                poll_stop(stop, work)?;
                if unchanged(i) {
                    let row_start = i * old_len;
                    let mut retained_start = 0usize;
                    for j in 0..new_len {
                        poll_stop(stop, work)?;
                        if unchanged(j) {
                            continue;
                        }
                        if retained_start < j {
                            sequence = sequence.then(ranges.range_interruptible(
                                &self.outputs,
                                row_start + retained_start,
                                row_start + j,
                                stop,
                                &mut work,
                            )?);
                        }
                        let args = [i64::from(candidate.at(i)), i64::from(candidate.at(j)), i as i64, j as i64];
                        sequence = sequence.then(SatTransform::value(eval_expr(&reduction.arena.exprs, reduction.body, &args)));
                        work = work.saturating_add(1);
                        retained_start = j + 1;
                    }
                    if retained_start < new_len {
                        sequence = sequence.then(ranges.range_interruptible(
                            &self.outputs,
                            row_start + retained_start,
                            row_start + new_len,
                            stop,
                            &mut work,
                        )?);
                    }
                } else {
                    for j in 0..new_len {
                        poll_stop(stop, work)?;
                        let args = [i64::from(candidate.at(i)), i64::from(candidate.at(j)), i as i64, j as i64];
                        sequence = sequence.then(SatTransform::value(eval_expr(&reduction.arena.exprs, reduction.body, &args)));
                        work = work.saturating_add(1);
                    }
                }
            }
            scratch.sum_value = Some(sequence.apply(0));
            return poll_stop(stop, work);
        }
        let mut old_changed = Vec::new();
        for pos in 0..old_len {
            poll_stop(stop, work)?;
            if !unchanged(pos) {
                old_changed.push(pos);
            }
            work = work.saturating_add(1);
        }
        for i in 0..old_len {
            poll_stop(stop, work)?;
            if !unchanged(i) {
                extend_interruptible(&mut scratch.removed, &self.outputs[i * old_len..(i + 1) * old_len], stop)?;
                work = work.saturating_add(old_len);
            } else {
                for &j in &old_changed {
                    poll_stop(stop, work)?;
                    scratch.removed.push(self.outputs[i * old_len + j]);
                    work = work.saturating_add(1);
                }
            }
        }
        let mut new_changed = Vec::new();
        for pos in 0..new_len {
            poll_stop(stop, work)?;
            if !unchanged(pos) {
                new_changed.push(pos);
            }
            work = work.saturating_add(1);
        }
        for i in 0..new_len {
            poll_stop(stop, work)?;
            if !unchanged(i) {
                for j in 0..new_len {
                    poll_stop(stop, work)?;
                    let args = [i64::from(candidate.at(i)), i64::from(candidate.at(j)), i as i64, j as i64];
                    scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &args));
                    work = work.saturating_add(1);
                }
            } else {
                for &j in &new_changed {
                    poll_stop(stop, work)?;
                    let args = [i64::from(candidate.at(i)), i64::from(candidate.at(j)), i as i64, j as i64];
                    scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &args));
                    work = work.saturating_add(1);
                }
            }
        }
        poll_stop(stop, work)
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_delta(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &(impl ListView + ?Sized),
        prefix: usize,
        suffix: usize,
        boundary: i32,
        step: crate::model::list::ExprId,
        end: Option<i32>,
        scratch: &mut EvalScratch,
        stop: &AtomicBool,
    ) -> EvaluationResult<()> {
        let mut acc = self.scan_acc[prefix];
        let mut prev = if prefix == 0 { boundary } else { old[prefix - 1] };
        let old_suffix_start = old.len().saturating_sub(suffix);
        let new_suffix_start = candidate.len().saturating_sub(suffix);
        let mut work = 0usize;

        for pos in prefix..new_suffix_start {
            poll_stop(stop, work)?;
            let current = candidate.at(pos);
            let next = eval_expr(&reduction.arena.exprs, step, &[i64::from(current), acc, i64::from(prev)]);
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(current), next, i64::from(prev)]));
            acc = next;
            prev = current;
            work = work.saturating_add(1);
        }

        // A deterministic scan can reuse an unchanged suffix only when both
        // pieces of threaded state, the accumulator and predecessor, rejoin the
        // accepted cache. If they do not match at the edit boundary, advance
        // through the common suffix until they do. This is exact for arbitrary
        // scan expressions and turns the usual feasible routing insertion into
        // O(changed span) work without imposing VRPTW-specific semantics.
        let old_predecessor = if old_suffix_start == 0 { boundary } else { old[old_suffix_start.saturating_sub(1)] };
        let mut reuse_from = (acc == self.scan_acc[old_suffix_start] && prev == old_predecessor).then_some(old_suffix_start);
        if reuse_from.is_none() {
            for offset in 0..suffix {
                poll_stop(stop, work)?;
                let new_pos = new_suffix_start + offset;
                let old_pos = old_suffix_start + offset;
                let current = candidate.at(new_pos);
                debug_assert_eq!(current, old[old_pos]);
                let next = eval_expr(&reduction.arena.exprs, step, &[i64::from(current), acc, i64::from(prev)]);
                scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(current), next, i64::from(prev)]));
                acc = next;
                prev = current;
                work = work.saturating_add(1);
                let old_boundary = old_pos + 1;
                if acc == self.scan_acc[old_boundary] {
                    reuse_from = Some(old_boundary);
                    break;
                }
            }
        }

        let old_reuse_start = if let Some(reuse_from) = reuse_from {
            reuse_from
        } else if let Some(end) = end {
            poll_stop(stop, work)?;
            let next = eval_expr(&reduction.arena.exprs, step, &[i64::from(end), acc, i64::from(prev)]);
            scratch.added.push(eval_expr(&reduction.arena.exprs, reduction.body, &[i64::from(end), next, i64::from(prev)]));
            self.outputs.len()
        } else {
            old.len()
        };

        extend_interruptible(&mut scratch.removed, &self.outputs[prefix..old_reuse_start], stop)?;
        scratch.recomputed_scan_steps = u64::try_from(scratch.added.len()).unwrap_or(u64::MAX);
        self.finish_contiguous_sum(prefix, old_reuse_start, scratch, stop)
    }

    #[allow(clippy::too_many_arguments)]
    fn windows_delta(
        &self,
        reduction: &Reduction,
        old: &[i32],
        candidate: &(impl ListView + ?Sized),
        prefix: usize,
        suffix: usize,
        size: usize,
        inner: crate::model::list::ExprId,
        scratch: &mut EvalScratch,
        stop: &AtomicBool,
    ) -> EvaluationResult<()> {
        let old_windows = window_count(old.len(), size);
        let new_windows = window_count(candidate.len(), size);
        let prefix_windows = window_count(prefix, size);
        let old_suffix_windows = window_count(suffix, size);
        let new_suffix_windows = window_count(suffix, size);
        let old_end = old_windows.saturating_sub(old_suffix_windows);
        extend_interruptible(&mut scratch.removed, &self.outputs[prefix_windows..old_end], stop)?;
        let new_end = new_windows.saturating_sub(new_suffix_windows);
        let mut work = 0usize;
        for start in prefix_windows..new_end {
            poll_stop(stop, work)?;
            scratch.added.push(window_value_interruptible(reduction, candidate, start, size, inner, stop, &mut work)?);
        }
        self.finish_contiguous_sum(prefix_windows, old_end, scratch, stop)
    }

    fn finish_contiguous_sum(
        &self,
        prefix_end: usize,
        suffix_start: usize,
        scratch: &mut EvalScratch,
        stop: &AtomicBool,
    ) -> EvaluationResult<()> {
        let mut sum = self.sum_prefix[prefix_end];
        for (index, &value) in scratch.added.iter().enumerate() {
            poll_stop(stop, index)?;
            sum = sum.saturating_add(value);
        }
        scratch.sum_value = Some(self.sum_suffix[suffix_start].apply(sum));
        poll_stop(stop, 0)
    }
}

fn emit_slice(reduction: &Reduction, contents: &[i32], outputs: &mut Vec<i64>, scan_acc: &mut Vec<i64>, stop: &AtomicBool) -> bool {
    if stop.load(Ordering::Relaxed) {
        return false;
    }
    let arena = &reduction.arena.exprs;
    match reduction.iterable {
        Iterable::Items(_) => {
            for (index, &item) in contents.iter().enumerate() {
                if index.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                    return false;
                }
                outputs.push(eval_expr(arena, reduction.body, &[i64::from(item)]));
            }
        }
        Iterable::SetItems(_) => {
            let mut members = contents.to_vec();
            members.sort_unstable();
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            for (index, item) in members.into_iter().enumerate() {
                if index.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                    return false;
                }
                outputs.push(eval_expr(arena, reduction.body, &[i64::from(item)]));
            }
        }
        Iterable::Edges { start, end, .. } => {
            for edge in 0..=contents.len() {
                if edge.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                    return false;
                }
                let from = if edge == 0 { start } else { contents[edge - 1] };
                let to = if edge == contents.len() { end } else { contents[edge] };
                outputs.push(eval_expr(arena, reduction.body, &[i64::from(from), i64::from(to)]));
            }
        }
        Iterable::Pairs(_) => {
            for (i, &left) in contents.iter().enumerate() {
                for (j, &right) in contents.iter().enumerate() {
                    if (i.saturating_mul(contents.len()).saturating_add(j)).is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                        return false;
                    }
                    outputs.push(eval_expr(arena, reduction.body, &[i64::from(left), i64::from(right), i as i64, j as i64]));
                }
            }
        }
        Iterable::Scan { init, boundary, step, end, .. } => {
            let mut acc = init;
            let mut prev = boundary;
            scan_acc.push(acc);
            for (index, &current) in contents.iter().enumerate() {
                if index.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                    return false;
                }
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
                if start.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                    return false;
                }
                outputs.push(window_value(reduction, contents, start, size, inner));
            }
        }
    }
    !stop.load(Ordering::Relaxed)
}

fn candidate_aggregate_interruptible(
    cache: &ReductionCache,
    reduction: &Reduction,
    scratch: &mut EvalScratch,
    stop: &AtomicBool,
) -> EvaluationResult<Option<i64>> {
    poll_stop(stop, 0)?;
    let new_len = cache.outputs.len().saturating_sub(scratch.removed.len()).saturating_add(scratch.added.len());
    let value = match reduction.op {
        ReduceOp::Sum => {
            if let Some(sum) = scratch.sum_value {
                return Ok(Some(sum));
            }
            let mut removed = 0i64;
            for (index, &value) in scratch.removed.iter().enumerate() {
                poll_stop(stop, index)?;
                removed = removed.saturating_add(value);
            }
            let mut added = 0i64;
            for (index, &value) in scratch.added.iter().enumerate() {
                poll_stop(stop, index)?;
                added = added.saturating_add(value);
            }
            cache.raw.map(|raw| raw.saturating_sub(removed).saturating_add(added))
        }
        ReduceOp::Count => {
            let mut removed = 0usize;
            for (index, &value) in scratch.removed.iter().enumerate() {
                poll_stop(stop, index)?;
                removed = removed.saturating_add(usize::from(value != 0));
            }
            let mut added = 0usize;
            for (index, &value) in scratch.added.iter().enumerate() {
                poll_stop(stop, index)?;
                added = added.saturating_add(usize::from(value != 0));
            }
            cache.raw.map(|raw| raw.saturating_sub(removed as i64).saturating_add(added as i64))
        }
        ReduceOp::Used => Some(i64::from(new_len > 0)),
        ReduceOp::Min => select_after_changes_interruptible(&cache.sorted, &mut scratch.removed, &mut scratch.added, 0, stop)?,
        ReduceOp::Max => {
            if new_len == 0 {
                None
            } else {
                select_after_changes_interruptible(&cache.sorted, &mut scratch.removed, &mut scratch.added, new_len - 1, stop)?
            }
        }
        ReduceOp::SelectKth(k) => {
            if k >= new_len {
                None
            } else {
                select_after_changes_interruptible(&cache.sorted, &mut scratch.removed, &mut scratch.added, k, stop)?
            }
        }
    };
    poll_stop(stop, 0)?;
    Ok(value)
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

    fn build_interruptible(values: &[i64], stop: &AtomicBool) -> Option<Self> {
        let mut blocks = Vec::with_capacity(values.len().div_ceil(Self::BLOCK_SIZE));
        for (index, chunk) in values.chunks(Self::BLOCK_SIZE).enumerate() {
            if index.is_multiple_of(32) && stop.load(Ordering::Relaxed) {
                return None;
            }
            blocks.push(chunk.iter().fold(SatTransform::identity(), |transform, &value| transform.then(SatTransform::value(value))));
        }
        Some(Self { blocks })
    }

    fn range_interruptible(
        &self,
        values: &[i64],
        mut start: usize,
        end: usize,
        stop: &AtomicBool,
        work: &mut usize,
    ) -> EvaluationResult<SatTransform> {
        debug_assert!(start <= end);
        debug_assert!(end <= values.len());
        let mut transform = SatTransform::identity();
        while start < end && !start.is_multiple_of(Self::BLOCK_SIZE) {
            poll_stop(stop, *work)?;
            transform = transform.then(SatTransform::value(values[start]));
            start += 1;
            *work = work.saturating_add(1);
        }
        while start.saturating_add(Self::BLOCK_SIZE) <= end {
            poll_stop(stop, *work)?;
            transform = transform.then(self.blocks[start / Self::BLOCK_SIZE]);
            start += Self::BLOCK_SIZE;
            *work = work.saturating_add(Self::BLOCK_SIZE);
        }
        while start < end {
            poll_stop(stop, *work)?;
            transform = transform.then(SatTransform::value(values[start]));
            start += 1;
            *work = work.saturating_add(1);
        }
        Ok(transform)
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

fn select_after_changes_interruptible(
    base: &[i64],
    removed: &mut [i64],
    added: &mut [i64],
    target: usize,
    stop: &AtomicBool,
) -> EvaluationResult<Option<i64>> {
    sort_interruptible(removed, stop)?;
    sort_interruptible(added, stop)?;
    let new_len = base.len().saturating_sub(removed.len()).saturating_add(added.len());
    if target >= new_len {
        return Ok(None);
    }

    // Min and Max are the hot cases. Walk only duplicates removed at the
    // corresponding edge of the accepted sorted cache.
    if target == 0 {
        let (mut base_pos, mut removed_pos) = (0usize, 0usize);
        while base_pos < base.len() && removed_pos < removed.len() && base[base_pos] == removed[removed_pos] {
            poll_stop(stop, base_pos)?;
            base_pos += 1;
            removed_pos += 1;
        }
        return Ok(match (base.get(base_pos), added.first()) {
            (Some(&left), Some(&right)) => Some(left.min(right)),
            (Some(&value), None) | (None, Some(&value)) => Some(value),
            (None, None) => None,
        });
    }
    if target + 1 == new_len {
        let (mut base_pos, mut removed_pos) = (base.len(), removed.len());
        let mut work = 0usize;
        while base_pos > 0 && removed_pos > 0 && base[base_pos - 1] == removed[removed_pos - 1] {
            poll_stop(stop, work)?;
            base_pos -= 1;
            removed_pos -= 1;
            work = work.saturating_add(1);
        }
        return Ok(match (base_pos.checked_sub(1).and_then(|pos| base.get(pos)), added.last()) {
            (Some(&left), Some(&right)) => Some(left.max(right)),
            (Some(&value), None) | (None, Some(&value)) => Some(value),
            (None, None) => None,
        });
    }

    // An order statistic is found by value-domain binary search. Rank queries
    // use the three sorted multisets, so the cost is logarithmic in list size
    // instead of scanning up to k retained values.
    let first = match (base.first(), added.first()) {
        (Some(&left), Some(&right)) => left.min(right),
        (Some(&value), None) | (None, Some(&value)) => value,
        (None, None) => return Ok(None),
    };
    let last = match (base.last(), added.last()) {
        (Some(&left), Some(&right)) => left.max(right),
        (Some(&value), None) | (None, Some(&value)) => value,
        (None, None) => return Ok(None),
    };
    let mut low = i128::from(first);
    let mut high = i128::from(last);
    let mut iteration = 0usize;
    while low < high {
        poll_stop(stop, iteration.saturating_mul(STOP_POLL_INTERVAL))?;
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
        iteration += 1;
    }
    Ok(Some(low as i64))
}

fn edge_nodes(view: &(impl ListView + ?Sized), edge: usize, start: i32, end: i32) -> (i32, i32) {
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

fn window_value_interruptible(
    reduction: &Reduction,
    view: &(impl ListView + ?Sized),
    start: usize,
    size: usize,
    inner: crate::model::list::ExprId,
    stop: &AtomicBool,
    work: &mut usize,
) -> EvaluationResult<i64> {
    let mut total = 0i64;
    for pos in start..start + size {
        poll_stop(stop, *work)?;
        total = total.saturating_add(eval_expr(&reduction.arena.exprs, inner, &[i64::from(view.at(pos))]));
        *work = work.saturating_add(1);
    }
    poll_stop(stop, *work)?;
    Ok(eval_expr(&reduction.arena.exprs, reduction.body, &[0, total]))
}
