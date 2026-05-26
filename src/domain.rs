//! Trailed sparse-set integer domain.
//!
//! Membership is "position in `dense` < size". The `dense`/`sparse` permutation
//! is never undone; only `size`/`min`/`max` are trailed, and restoring them
//! restores membership.

use std::sync::Arc;

use crate::propagator::Inconsistency;
use crate::trail::{ReversibleInt, Trail};

const MAX_DENSE_VALUES: usize = 10_000_000;
const DENSE_SET_FACTOR: usize = 4;

#[derive(Clone)]
enum ValueMap {
    /// Internal index `i` represents `offset + i`.
    Dense { offset: i32 },
    /// Internal index `i` represents `values[i]`.
    Sparse { values: Arc<[i32]> },
}

/// A finite-integer domain backed by a trailed sparse set.
#[derive(Clone)]
pub struct Domain {
    values: ValueMap,
    /// Internal index at each position; present iff position `< size`.
    dense: Vec<u32>,
    /// Position of each internal index in `dense`.
    sparse: Vec<u32>,
    size: ReversibleInt,
    min: ReversibleInt,
    max: ReversibleInt,
}

impl Domain {
    /// Domain over the inclusive range `[lo, hi]`. Requires `lo <= hi`.
    pub fn new_range(lo: i32, hi: i32, trail: &mut Trail) -> Self {
        assert!(lo <= hi, "empty domain range [{lo}, {hi}]");
        let size = span_len(lo, hi);
        assert!(
            size <= MAX_DENSE_VALUES,
            "dense domain range [{lo}, {hi}] has {size} values; maximum supported is {MAX_DENSE_VALUES}"
        );
        Self::from_dense_range(lo, hi, size, trail)
    }

    /// Domain over an explicit non-empty value set; duplicates ignored.
    pub fn new_set(values: &[i32], trail: &mut Trail) -> Self {
        assert!(!values.is_empty(), "empty domain set");
        let mut vals: Vec<i32> = values.to_vec();
        vals.sort_unstable();
        vals.dedup();
        let lo = vals[0];
        let hi = *vals.last().unwrap();
        let span = span_len(lo, hi);
        if span <= MAX_DENSE_VALUES && span <= vals.len().saturating_mul(DENSE_SET_FACTOR) {
            Self::from_dense_set(vals, lo, hi, span, trail)
        } else {
            Self::from_sparse_set(vals, trail)
        }
    }

    fn from_dense_range(lo: i32, hi: i32, size: usize, trail: &mut Trail) -> Self {
        let dense: Vec<u32> = (0..size as u32).collect();
        let sparse = dense.clone();
        Self {
            values: ValueMap::Dense { offset: lo },
            dense,
            sparse,
            size: trail.new_int(size as i32),
            min: trail.new_int(lo),
            max: trail.new_int(hi),
        }
    }

    fn from_dense_set(vals: Vec<i32>, lo: i32, hi: i32, span: usize, trail: &mut Trail) -> Self {
        let size = vals.len();
        let mut present = vec![false; span];
        for &v in &vals {
            present[(i64::from(v) - i64::from(lo)) as usize] = true;
        }

        let mut dense = vec![0u32; span];
        let mut front = 0usize;
        let mut back = size;
        for (idx, &is_present) in present.iter().enumerate() {
            if is_present {
                dense[front] = idx as u32;
                front += 1;
            } else {
                dense[back] = idx as u32;
                back += 1;
            }
        }
        let mut sparse = vec![0u32; span];
        for (pos, &idx) in dense.iter().enumerate() {
            sparse[idx as usize] = pos as u32;
        }

        Self {
            values: ValueMap::Dense { offset: lo },
            dense,
            sparse,
            size: trail.new_int(size as i32),
            min: trail.new_int(lo),
            max: trail.new_int(hi),
        }
    }

    fn from_sparse_set(vals: Vec<i32>, trail: &mut Trail) -> Self {
        let lo = vals[0];
        let hi = *vals.last().unwrap();
        let size = vals.len();
        assert!(
            size <= i32::MAX as usize,
            "sparse domain has {size} values; maximum supported is {}",
            i32::MAX
        );
        let dense: Vec<u32> = (0..size as u32).collect();
        let sparse = dense.clone();
        Self {
            values: ValueMap::Sparse {
                values: Arc::from(vals),
            },
            dense,
            sparse,
            size: trail.new_int(size as i32),
            min: trail.new_int(lo),
            max: trail.new_int(hi),
        }
    }

    /// Number of present values.
    #[inline]
    pub fn size(&self, trail: &Trail) -> usize {
        trail.get(self.size) as usize
    }

    /// Current minimum present value. Meaningful only while `size > 0`.
    #[inline]
    pub fn min(&self, trail: &Trail) -> i32 {
        trail.get(self.min)
    }

    /// Current maximum present value. Meaningful only while `size > 0`.
    #[inline]
    pub fn max(&self, trail: &Trail) -> i32 {
        trail.get(self.max)
    }

    /// Whether exactly one value remains.
    #[inline]
    pub fn is_fixed(&self, trail: &Trail) -> bool {
        self.size(trail) == 1
    }

    /// Whether `val` is currently present.
    #[inline]
    pub fn contains(&self, val: i32, trail: &Trail) -> bool {
        self.index_of(val)
            .is_some_and(|i| (self.sparse[i] as usize) < self.size(trail))
    }

    /// Iterate present values in arbitrary order.
    pub fn values<'a>(&'a self, trail: &Trail) -> impl Iterator<Item = i32> + 'a {
        self.dense[..self.size(trail)]
            .iter()
            .map(|&i| self.value_of(i as usize))
    }

    /// Iterate values in the immutable root representation.
    pub(crate) fn root_values(&self) -> impl Iterator<Item = i32> + '_ {
        (0..self.dense.len()).map(|i| self.value_of(i))
    }

    /// Root support when this domain uses cardinality-sized storage.
    pub(crate) fn sparse_values(&self) -> Option<Arc<[i32]>> {
        match &self.values {
            ValueMap::Sparse { values } => Some(Arc::clone(values)),
            ValueMap::Dense { .. } => None,
        }
    }

    pub(crate) fn is_sparse(&self) -> bool {
        matches!(&self.values, ValueMap::Sparse { .. })
    }

    fn index_of(&self, val: i32) -> Option<usize> {
        match &self.values {
            ValueMap::Dense { offset } => {
                let shifted = i64::from(val) - i64::from(*offset);
                (shifted >= 0 && shifted < self.dense.len() as i64).then_some(shifted as usize)
            }
            ValueMap::Sparse { values } => values.binary_search(&val).ok(),
        }
    }

    fn value_of(&self, index: usize) -> i32 {
        match &self.values {
            ValueMap::Dense { offset } => (i64::from(*offset) + index as i64) as i32,
            ValueMap::Sparse { values } => values[index],
        }
    }

    /// Swap the values at positions `a` and `b`, keeping `sparse` consistent.
    fn swap(&mut self, a: usize, b: usize) {
        let ia = self.dense[a];
        let ib = self.dense[b];
        self.dense[a] = ib;
        self.dense[b] = ia;
        self.sparse[ia as usize] = b as u32;
        self.sparse[ib as usize] = a as u32;
    }

    /// Remove `val`; returns whether the domain changed. Absent value is a
    /// no-op. Caller detects wipeout via [`Domain::size`].
    pub fn remove(&mut self, val: i32, trail: &mut Trail) -> bool {
        let Some(i) = self.index_of(val) else {
            return false;
        };
        let pos = self.sparse[i] as usize;
        let size = self.size(trail);
        if pos >= size {
            return false;
        }
        let last = size - 1;
        self.swap(pos, last);
        trail.set(self.size, last as i32);
        if last > 0 {
            if val == self.min(trail) {
                self.recompute_min(trail);
            } else if val == self.max(trail) {
                self.recompute_max(trail);
            }
        }
        true
    }

    /// Fix the domain to `val`. Returns `Ok(changed)`, or `Err(Inconsistency)`
    /// if `val` is absent.
    pub fn fix(&mut self, val: i32, trail: &mut Trail) -> Result<bool, Inconsistency> {
        let Some(i) = self.index_of(val) else {
            return Err(Inconsistency);
        };
        let pos = self.sparse[i] as usize;
        if pos >= self.size(trail) {
            return Err(Inconsistency);
        }
        let changed = self.size(trail) > 1;
        self.swap(pos, 0);
        trail.set(self.size, 1);
        trail.set(self.min, val);
        trail.set(self.max, val);
        Ok(changed)
    }

    /// Remove every value `< bound`. Returns whether the domain changed.
    pub fn remove_below(&mut self, bound: i32, trail: &mut Trail) -> bool {
        if self.size(trail) == 0 || bound <= self.min(trail) {
            return false;
        }
        match &self.values {
            ValueMap::Dense { .. } => {
                let mut changed = false;
                let mut v = i64::from(self.min(trail));
                while v < i64::from(bound) {
                    changed |= self.remove(v as i32, trail);
                    if self.size(trail) == 0 {
                        break;
                    }
                    v += 1;
                }
                changed
            }
            ValueMap::Sparse { .. } => self.remove_if(trail, |v| v < bound),
        }
    }

    /// Remove every value `> bound`. Returns whether the domain changed.
    pub fn remove_above(&mut self, bound: i32, trail: &mut Trail) -> bool {
        if self.size(trail) == 0 || bound >= self.max(trail) {
            return false;
        }
        match &self.values {
            ValueMap::Dense { .. } => {
                let mut changed = false;
                let mut v = i64::from(self.max(trail));
                while v > i64::from(bound) {
                    changed |= self.remove(v as i32, trail);
                    if self.size(trail) == 0 {
                        break;
                    }
                    v -= 1;
                }
                changed
            }
            ValueMap::Sparse { .. } => self.remove_if(trail, |v| v > bound),
        }
    }

    fn remove_if(&mut self, trail: &mut Trail, remove: impl Fn(i32) -> bool) -> bool {
        let old_size = self.size(trail);
        let mut size = old_size;
        let mut pos = 0;
        while pos < size {
            if remove(self.value_of(self.dense[pos] as usize)) {
                self.swap(pos, size - 1);
                size -= 1;
            } else {
                pos += 1;
            }
        }
        if size == old_size {
            return false;
        }
        trail.set(self.size, size as i32);
        if size > 0 {
            self.recompute_bounds(trail);
        }
        true
    }

    fn recompute_min(&mut self, trail: &mut Trail) {
        let min = match &self.values {
            ValueMap::Dense { .. } => {
                let mut v = i64::from(self.min(trail)) + 1;
                let hi = i64::from(self.max(trail));
                while !self.contains(v as i32, trail) {
                    debug_assert!(v < hi);
                    v += 1;
                }
                v as i32
            }
            ValueMap::Sparse { .. } => self.values(trail).min().unwrap(),
        };
        trail.set(self.min, min);
    }

    fn recompute_max(&mut self, trail: &mut Trail) {
        let max = match &self.values {
            ValueMap::Dense { .. } => {
                let mut v = i64::from(self.max(trail)) - 1;
                let lo = i64::from(self.min(trail));
                while !self.contains(v as i32, trail) {
                    debug_assert!(v > lo);
                    v -= 1;
                }
                v as i32
            }
            ValueMap::Sparse { .. } => self.values(trail).max().unwrap(),
        };
        trail.set(self.max, max);
    }

    fn recompute_bounds(&mut self, trail: &mut Trail) {
        let (min, max) = self
            .values(trail)
            .fold((i32::MAX, i32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        trail.set(self.min, min);
        trail.set(self.max, max);
    }
}

fn span_len(lo: i32, hi: i32) -> usize {
    usize::try_from(i64::from(hi) - i64::from(lo) + 1).expect("domain span does not fit usize")
}
