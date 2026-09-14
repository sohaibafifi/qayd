//! Exact integer convex bounds for a fixed-sum sum of squares.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Solver, Store};

/// Enforces `sum(variables) = total` and `objective = sum(variables^2)`.
/// Domain holes are relaxed to their interval hull for filtering, but fixed
/// assignments satisfy both equalities exactly. All arithmetic stays in i128.
#[derive(Clone)]
struct FixedSumSquares {
    variables: Vec<VarId>,
    objective: VarId,
    total: i128,
    bounds: Vec<(i128, i128)>,
}

impl FixedSumSquares {
    fn propagate_with_stop(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        loop {
            if should_stop() {
                return Ok(());
            }
            self.bounds.clear();
            let mut sum_lo = 0i128;
            let mut sum_hi = 0i128;
            let mut square_upper = 0i128;
            let mut lowest = i128::MAX;
            let mut highest = i128::MIN;
            for &variable in &self.variables {
                if should_stop() {
                    return Ok(());
                }
                let lo = i128::from(store.min(variable));
                let hi = i128::from(store.max(variable));
                self.bounds.push((lo, hi));
                sum_lo += lo;
                sum_hi += hi;
                square_upper += (lo * lo).max(hi * hi);
                lowest = lowest.min(lo);
                highest = highest.max(hi);
            }
            if self.total < sum_lo || self.total > sum_hi {
                return Err(Inconsistency);
            }

            // Bound consistency for the fixed sum. Repeat when sparse domains
            // jump beyond the requested bound, before computing the water level.
            let mut changed = false;
            for (&variable, &(lo, hi)) in self.variables.iter().zip(&self.bounds) {
                if should_stop() {
                    return Ok(());
                }
                let new_lo = lo.max(self.total - (sum_hi - hi));
                let new_hi = hi.min(self.total - (sum_lo - lo));
                if new_lo > new_hi {
                    return Err(Inconsistency);
                }
                if new_lo > lo {
                    store.remove_below(variable, new_lo as i32)?;
                    changed = true;
                }
                if new_hi < hi {
                    store.remove_above(variable, new_hi as i32)?;
                    changed = true;
                }
            }
            if changed {
                continue;
            }

            // At the optimum, every unfrozen variable is q or q+1. Find the
            // largest integer q for which sum(clamp(q, lo_i, hi_i)) <= total.
            let mut left = if self.bounds.is_empty() { 0 } else { lowest };
            let mut right = if self.bounds.is_empty() { 0 } else { highest };
            while left < right {
                if should_stop() {
                    return Ok(());
                }
                let mid = left + (right - left + 1) / 2;
                let mut sum = 0i128;
                for &(lo, hi) in &self.bounds {
                    if should_stop() {
                        return Ok(());
                    }
                    sum += mid.clamp(lo, hi);
                }
                if sum <= self.total {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            }
            let q = left;
            let mut lower = 0i128;
            let mut remainder = self.total;
            for &(lo, hi) in &self.bounds {
                if should_stop() {
                    return Ok(());
                }
                let value = q.clamp(lo, hi);
                remainder -= value;
                lower += value * value;
            }
            lower += remainder * (2 * q + 1);
            if lower > i128::from(store.max(self.objective)) || square_upper < i128::from(store.min(self.objective)) {
                return Err(Inconsistency);
            }
            // lower is nonnegative and no greater than the current i32 upper
            // bound; square_upper is converted only when it tightens that bound.
            store.remove_below(self.objective, lower as i32)?;
            if square_upper < i128::from(store.max(self.objective)) {
                store.remove_above(self.objective, square_upper as i32)?;
            }

            // lambda=2q+1 supports the integer convex envelope of x^2.
            // lower = lambda*total + sum min(x^2-lambda*x). Fixing variable i
            // to v therefore costs at least penalty(v)-penalty(clamp(q)), with
            // penalty(v)=(v-q)(v-q-1). This certificate remains valid after any
            // filtering in this pass and avoids a water-fill solve per value.
            let slack = i128::from(store.max(self.objective)) - lower;
            for (&variable, &(lo, hi)) in self.variables.iter().zip(&self.bounds) {
                if should_stop() {
                    return Ok(());
                }
                let penalty = |value: i128| (value - q) * (value - q - 1);
                let base = q.clamp(lo, hi);
                let allowance = slack + penalty(base);
                if penalty(lo) > allowance {
                    let mut left = lo;
                    let mut right = base;
                    while left < right {
                        if should_stop() {
                            return Ok(());
                        }
                        let mid = left + (right - left) / 2;
                        if penalty(mid) > allowance {
                            left = mid + 1;
                        } else {
                            right = mid;
                        }
                    }
                    store.remove_below(variable, left as i32)?;
                    changed = true;
                }
                if penalty(hi) > allowance {
                    let mut left = (q + 1).clamp(lo, hi);
                    let mut right = hi;
                    while left < right {
                        if should_stop() {
                            return Ok(());
                        }
                        let mid = left + (right - left + 1) / 2;
                        if penalty(mid) > allowance {
                            right = mid - 1;
                        } else {
                            left = mid;
                        }
                    }
                    store.remove_above(variable, left as i32)?;
                    changed = true;
                }
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

impl Propagator for FixedSumSquares {
    fn register(&mut self, store: &mut Store, me: PropId) {
        self.register_until(store, me, &|| false);
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        // Whole-scope fallback reasons must include the objective cutoff as
        // well as every input bound. No reversible state is cached here.
        store.subscribe(self.objective, me, Event::BoundChange);
        for &variable in &self.variables {
            if should_stop() {
                return false;
            }
            store.subscribe(variable, me, Event::BoundChange);
        }
        !should_stop()
    }

    fn priority(&self) -> Priority {
        Priority::Expensive
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        self.propagate_with_stop(store, &|| false)
    }

    fn propagate_until(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        self.propagate_with_stop(store, should_stop)
    }
}

/// Post a square-sum equality with a separately established unconditional
/// fixed sum. Scope entries must be distinct and exclude the objective.
pub(crate) fn fixed_sum_squares_interruptible(
    solver: &mut Solver,
    variables: &[VarId],
    objective: VarId,
    total: i128,
    stop: &AtomicBool,
) -> bool {
    let should_stop = || stop.load(Ordering::Acquire);
    let mut copied = Vec::with_capacity(variables.len());
    let mut seen = BTreeSet::new();
    for &variable in variables {
        if should_stop() {
            return false;
        }
        assert!(variable != objective && seen.insert(variable), "fixed sum of squares requires distinct input and objective variables");
        copied.push(variable);
    }
    solver.post_until(Box::new(FixedSumSquares { variables: copied, objective, total, bounds: Vec::new() }), &should_stop).is_some()
}
