//! Scheduling constraints (Phase 4, weak-but-correct): `noOverlap`,
//! `cumulative` (time-tabling), `binPacking` (Shaw load pruning), and
//! `knapsack` (a pair of linear constraints).
//!
//! Durations, heights, sizes, and capacities are fixed integers here; variable
//! durations and the stronger filters (edge-finding, Shaw's full bound) are
//! Phase 6 upgrades.

use crate::constraints::linear::{linear, Relation};
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Solver, Store};

fn clamp_i32(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

// ===========================================================================
// noOverlap (disjunctive, pairwise)
// ===========================================================================

/// Tasks `[start_i, start_i + duration_i)` must be pairwise non-overlapping.
struct NoOverlap {
    starts: Vec<VarId>,
    durations: Vec<i64>,
}

impl Propagator for NoOverlap {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &s in &self.starts {
            store.subscribe(s, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.starts.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let (di, dj) = (self.durations[i], self.durations[j]);
                let si_min = store.min(self.starts[i]) as i64;
                let si_max = store.max(self.starts[i]) as i64;
                let sj_min = store.min(self.starts[j]) as i64;
                let sj_max = store.max(self.starts[j]) as i64;

                let i_before_j = si_min + di <= sj_max;
                let j_before_i = sj_min + dj <= si_max;

                match (i_before_j, j_before_i) {
                    (false, false) => return Err(Inconsistency),
                    (true, false) => {
                        // i must precede j.
                        store.remove_below(self.starts[j], clamp_i32(si_min + di))?;
                        store.remove_above(self.starts[i], clamp_i32(sj_max - di))?;
                    }
                    (false, true) => {
                        // j must precede i.
                        store.remove_below(self.starts[i], clamp_i32(sj_min + dj))?;
                        store.remove_above(self.starts[j], clamp_i32(si_max - dj))?;
                    }
                    (true, true) => {}
                }
            }
        }
        Ok(())
    }
}

/// Post `noOverlap`: the tasks must not overlap in time.
pub fn no_overlap(solver: &mut Solver, starts: &[VarId], durations: &[i64]) {
    assert_eq!(
        starts.len(),
        durations.len(),
        "noOverlap: starts/durations mismatch"
    );
    solver.post(Box::new(NoOverlap {
        starts: starts.to_vec(),
        durations: durations.to_vec(),
    }));
}

// ===========================================================================
// cumulative (time-tabling)
// ===========================================================================

/// At every time point the total height of running tasks must not exceed
/// `capacity`. Time-tabling: build the profile of mandatory parts and forbid
/// starts that would push any covered time over capacity.
struct Cumulative {
    starts: Vec<VarId>,
    dur: Vec<i64>,
    height: Vec<i64>,
    capacity: i64,
    profile: Vec<i64>,
    buf: Vec<i32>,
}

impl Propagator for Cumulative {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &s in &self.starts {
            store.subscribe(s, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.starts.len();
        if n == 0 {
            return Ok(());
        }
        let hmin = (0..n)
            .map(|i| store.min(self.starts[i]) as i64)
            .min()
            .unwrap();
        let hmax = (0..n)
            .map(|i| store.max(self.starts[i]) as i64 + self.dur[i])
            .max()
            .unwrap();
        if hmax <= hmin {
            return Ok(());
        }
        let horizon = (hmax - hmin) as usize;

        // Mandatory-part profile: task i mandatorily runs on [max start, min end).
        self.profile.clear();
        self.profile.resize(horizon, 0);
        for i in 0..n {
            let mand_start = store.max(self.starts[i]) as i64;
            let mand_end = store.min(self.starts[i]) as i64 + self.dur[i];
            for t in mand_start..mand_end {
                self.profile[(t - hmin) as usize] += self.height[i];
            }
        }
        for &p in &self.profile {
            if p > self.capacity {
                return Err(Inconsistency);
            }
        }

        // For each task, forbid starts whose execution window would exceed
        // capacity against the other tasks' mandatory parts.
        for i in 0..n {
            let hi = self.height[i];
            let mand_start = store.max(self.starts[i]) as i64;
            let mand_end = store.min(self.starts[i]) as i64 + self.dur[i];
            self.buf.clear();
            self.buf.extend(store.values(self.starts[i]));
            for &s in &self.buf {
                let s = s as i64;
                let conflict = (s..s + self.dur[i]).any(|t| {
                    let idx = (t - hmin) as usize;
                    let own = if t >= mand_start && t < mand_end {
                        hi
                    } else {
                        0
                    };
                    self.profile[idx] - own + hi > self.capacity
                });
                if conflict {
                    store.remove(self.starts[i], clamp_i32(s))?;
                }
            }
        }
        Ok(())
    }
}

/// Post `cumulative`: resource usage never exceeds `capacity`.
pub fn cumulative(
    solver: &mut Solver,
    starts: &[VarId],
    durations: &[i64],
    heights: &[i64],
    capacity: i64,
) {
    assert_eq!(starts.len(), durations.len(), "cumulative: length mismatch");
    assert_eq!(starts.len(), heights.len(), "cumulative: length mismatch");
    solver.post(Box::new(Cumulative {
        starts: starts.to_vec(),
        dur: durations.to_vec(),
        height: heights.to_vec(),
        capacity,
        profile: Vec::new(),
        buf: Vec::new(),
    }));
}

// ===========================================================================
// binPacking (Shaw, load-based)
// ===========================================================================

/// Each item is assigned to a bin (`items[i]` = bin index); the total size in
/// each bin must not exceed its capacity.
struct BinPacking {
    items: Vec<VarId>,
    sizes: Vec<i64>,
    capacities: Vec<i64>,
    load: Vec<i64>,
    buf: Vec<i32>,
}

impl Propagator for BinPacking {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &it in &self.items {
            store.subscribe(it, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let nbins = self.capacities.len();
        for &it in &self.items {
            store.remove_below(it, 0)?;
            store.remove_above(it, (nbins - 1) as i32)?;
        }

        // Committed load = sizes of items already fixed to each bin.
        self.load.clear();
        self.load.resize(nbins, 0);
        for (i, &it) in self.items.iter().enumerate() {
            if store.is_fixed(it) {
                self.load[store.value(it) as usize] += self.sizes[i];
            }
        }
        for b in 0..nbins {
            if self.load[b] > self.capacities[b] {
                return Err(Inconsistency);
            }
        }

        // An unfixed item cannot go in a bin that can no longer hold it.
        for i in 0..self.items.len() {
            if store.is_fixed(self.items[i]) {
                continue;
            }
            self.buf.clear();
            self.buf.extend(store.values(self.items[i]));
            for &b in &self.buf {
                if self.load[b as usize] + self.sizes[i] > self.capacities[b as usize] {
                    store.remove(self.items[i], b)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `binPacking`: `items[i]` is the bin holding item `i`, of `sizes[i]`;
/// each bin `b` holds at most `capacities[b]`.
pub fn bin_packing(solver: &mut Solver, items: &[VarId], sizes: &[i64], capacities: &[i64]) {
    assert_eq!(items.len(), sizes.len(), "binPacking: items/sizes mismatch");
    solver.post(Box::new(BinPacking {
        items: items.to_vec(),
        sizes: sizes.to_vec(),
        capacities: capacities.to_vec(),
        load: Vec::new(),
        buf: Vec::new(),
    }));
}

// ===========================================================================
// knapsack (two linear constraints)
// ===========================================================================

/// Post `knapsack`: `Σ weights·vars  weight_rel  weight_limit` and
/// `Σ profits·vars  profit_rel  profit_limit`. Reuses linear bounds propagation.
#[allow(clippy::too_many_arguments)]
pub fn knapsack(
    solver: &mut Solver,
    vars: &[VarId],
    weights: &[i64],
    profits: &[i64],
    weight_rel: Relation,
    weight_limit: i64,
    profit_rel: Relation,
    profit_limit: i64,
) {
    assert_eq!(vars.len(), weights.len(), "knapsack: vars/weights mismatch");
    assert_eq!(vars.len(), profits.len(), "knapsack: vars/profits mismatch");
    linear(solver, weights, vars, weight_rel, weight_limit);
    linear(solver, profits, vars, profit_rel, profit_limit);
}
