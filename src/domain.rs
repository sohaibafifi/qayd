//! Trailed sparse-set integer domain (MiniCP design).
//!
//! Values are offset-shifted into `[0, capacity)`. Membership is "the value's
//! position in `dense` is `< size`". The elegant part: the `dense`/`sparse`
//! permutation is **never undone** — restoring the trailed `size` (plus `min`
//! and `max`) on backtrack restores membership regardless of how elements were
//! swapped during removals. Only `size`, `min`, `max` live on the trail.

use crate::propagator::Inconsistency;
use crate::trail::{ReversibleInt, Trail};

/// A finite-integer domain backed by a trailed sparse set.
pub struct Domain {
    /// Value `v` maps to internal index `(v - offset) as u32`.
    offset: i32,
    /// `dense[p]` is the internal index of the value at position `p`. Present
    /// iff `p < size`.
    dense: Vec<u32>,
    /// `sparse[i]` is the position of internal index `i` in `dense`.
    sparse: Vec<u32>,
    /// Number of present values.
    size: ReversibleInt,
    /// Current minimum present value.
    min: ReversibleInt,
    /// Current maximum present value.
    max: ReversibleInt,
}

impl Domain {
    /// Domain over the inclusive range `[lo, hi]`. Requires `lo <= hi`.
    pub fn new_range(lo: i32, hi: i32, trail: &mut Trail) -> Self {
        assert!(lo <= hi, "empty domain range [{lo}, {hi}]");
        let n = (hi - lo + 1) as usize;
        let dense: Vec<u32> = (0..n as u32).collect();
        let sparse: Vec<u32> = (0..n as u32).collect();
        Self {
            offset: lo,
            dense,
            sparse,
            size: trail.new_int(n as i32),
            min: trail.new_int(lo),
            max: trail.new_int(hi),
        }
    }

    /// Domain over an explicit set of values. Duplicates are ignored; the set
    /// must be non-empty. Holes between `min` and `max` are simply absent.
    pub fn new_set(values: &[i32], trail: &mut Trail) -> Self {
        assert!(!values.is_empty(), "empty domain set");
        let mut vals: Vec<i32> = values.to_vec();
        vals.sort_unstable();
        vals.dedup();
        let lo = vals[0];
        let hi = *vals.last().unwrap();
        let cap = (hi - lo + 1) as usize;
        let size = vals.len();

        let mut present = vec![false; cap];
        for &v in &vals {
            present[(v - lo) as usize] = true;
        }

        // Present internal indices fill positions `[0, size)`, absent ones the rest.
        let mut dense = vec![0u32; cap];
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
        let mut sparse = vec![0u32; cap];
        for (pos, &idx) in dense.iter().enumerate() {
            sparse[idx as usize] = pos as u32;
        }

        Self {
            offset: lo,
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

    /// Whether `val` is currently present. O(1).
    #[inline]
    pub fn contains(&self, val: i32, trail: &Trail) -> bool {
        let shifted = val - self.offset;
        if shifted < 0 {
            return false;
        }
        let i = shifted as usize;
        i < self.dense.len() && (self.sparse[i] as usize) < self.size(trail)
    }

    /// Iterate the present values (in arbitrary order). Borrows only the
    /// domain; `size`/`offset` are copied out, so it does not hold the trail.
    pub fn values<'a>(&'a self, trail: &Trail) -> impl Iterator<Item = i32> + 'a {
        let size = self.size(trail);
        let offset = self.offset;
        self.dense[..size].iter().map(move |&i| i as i32 + offset)
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

    /// Remove `val`. Returns whether the domain changed. Removing an absent
    /// value is a no-op. The caller checks for wipeout via [`Domain::size`].
    pub fn remove(&mut self, val: i32, trail: &mut Trail) -> bool {
        if !self.contains(val, trail) {
            return false;
        }
        let i = (val - self.offset) as usize;
        let pos = self.sparse[i] as usize;
        let last = self.size(trail) - 1;
        // Move `val` to the last present slot and shrink: it leaves the set,
        // but the permutation itself is never undone.
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
        if !self.contains(val, trail) {
            return Err(Inconsistency);
        }
        let changed = self.size(trail) > 1;
        let i = (val - self.offset) as usize;
        let pos = self.sparse[i] as usize;
        self.swap(pos, 0);
        trail.set(self.size, 1);
        trail.set(self.min, val);
        trail.set(self.max, val);
        Ok(changed)
    }

    /// Remove every value `< bound`. Returns whether the domain changed.
    pub fn remove_below(&mut self, bound: i32, trail: &mut Trail) -> bool {
        let mut changed = false;
        // Walk the value range; `remove` is a no-op for absent values.
        let mut v = self.min(trail);
        while v < bound {
            if self.remove(v, trail) {
                changed = true;
            }
            if self.size(trail) == 0 {
                break;
            }
            v += 1;
        }
        changed
    }

    /// Remove every value `> bound`. Returns whether the domain changed.
    pub fn remove_above(&mut self, bound: i32, trail: &mut Trail) -> bool {
        let mut changed = false;
        let mut v = self.max(trail);
        while v > bound {
            if self.remove(v, trail) {
                changed = true;
            }
            if self.size(trail) == 0 {
                break;
            }
            v -= 1;
        }
        changed
    }

    /// Find the smallest present value `> old min` and store it as the new min.
    fn recompute_min(&mut self, trail: &mut Trail) {
        let hi = self.max(trail);
        let mut v = self.min(trail) + 1;
        while v <= hi {
            if self.contains(v, trail) {
                trail.set(self.min, v);
                return;
            }
            v += 1;
        }
    }

    /// Find the largest present value `< old max` and store it as the new max.
    fn recompute_max(&mut self, trail: &mut Trail) {
        let lo = self.min(trail);
        let mut v = self.max(trail) - 1;
        while v >= lo {
            if self.contains(v, trail) {
                trail.set(self.max, v);
                return;
            }
            v -= 1;
        }
    }
}
