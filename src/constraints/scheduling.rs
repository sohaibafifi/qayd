//! Scheduling constraints: `noOverlap`, `cumulative` (time-tabling),
//! `binPacking` (Shaw load pruning), `knapsack` (two linear constraints).

use crate::constraints::linear::{linear, Relation};
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Solver, Store};

fn clamp_i32(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Ceiling of `a / b` for `a >= 0`, `b > 0`.
fn ceil_div_pos(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
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

/// At every time point the total height of running tasks must not exceed `capacity`.
struct Cumulative {
    starts: Vec<VarId>,
    dur: Vec<i64>,
    height: Vec<i64>,
    capacity: i64,
    profile: Vec<i64>,
    buf: Vec<i32>,
    /// Reused scratch for energetic overload checking.
    est: Vec<i64>,
    lct: Vec<i64>,
    energy: Vec<i64>,
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

        // Local fixpoint: self-changes don't re-enqueue, so one pass could miss
        // overloads involving a just-fixed task.
        loop {
            let before: usize = (0..n).map(|i| store.size(self.starts[i])).sum();

            // Snapshot est, lct, energy.
            self.est.clear();
            self.lct.clear();
            self.energy.clear();
            for i in 0..n {
                self.est.push(store.min(self.starts[i]) as i64);
                self.lct
                    .push(store.max(self.starts[i]) as i64 + self.dur[i]);
                self.energy.push(self.height[i] * self.dur[i]);
            }

            // Energetic overload check: window [L, U) energy > capacity*(U-L) fails.
            let mut by_est: Vec<usize> = (0..n).collect();
            by_est.sort_unstable_by(|&a, &b| self.est[b].cmp(&self.est[a]));
            for u in 0..n {
                let ub = self.lct[u];
                let mut e = 0i64;
                for &k in &by_est {
                    if self.lct[k] <= ub {
                        e += self.energy[k];
                        if e > self.capacity * (ub - self.est[k]) {
                            return Err(Inconsistency);
                        }
                    }
                }
            }

            // Edge-finding: if Omega ∪ {i} can't fit in [est, U], i ends after Omega,
            // so it can't start until Omega's non-parallelisable rest energy is done.
            let mut by_lct: Vec<usize> = (0..n).collect();
            by_lct.sort_unstable_by(|&a, &b| self.lct[a].cmp(&self.lct[b]));
            let mut lb = self.est.clone();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let hi = self.height[i];
                if hi == 0 {
                    continue;
                }
                let mut e_omega = 0i64;
                let mut est_omega = i64::MAX;
                for &j in &by_lct {
                    if j == i {
                        continue;
                    }
                    e_omega += self.energy[j];
                    est_omega = est_omega.min(self.est[j]);
                    let u = self.lct[j];
                    if e_omega + self.energy[i] > self.capacity * (u - est_omega.min(self.est[i])) {
                        let rest = e_omega - (self.capacity - hi) * (u - est_omega);
                        if rest > 0 {
                            lb[i] = lb[i].max(est_omega + ceil_div_pos(rest, hi));
                        }
                    }
                }
            }
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                if lb[i] > self.est[i] {
                    store.remove_below(self.starts[i], clamp_i32(lb[i]))?;
                }
            }

            // Time-tabling over the mandatory-part profile.
            let hmin = self.est.iter().copied().min().unwrap();
            let hmax = self.lct.iter().copied().max().unwrap();
            if hmax > hmin {
                let horizon = (hmax - hmin) as usize;
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
                // Forbid starts whose window exceeds capacity vs others' mandatory parts.
                for i in 0..n {
                    let hi = self.height[i];
                    let mand_start = store.max(self.starts[i]) as i64;
                    let mand_end = store.min(self.starts[i]) as i64 + self.dur[i];
                    self.buf.clear();
                    self.buf.extend(store.values(self.starts[i]));
                    for &start in &self.buf {
                        let s = start as i64;
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
                            store.remove(self.starts[i], start)?;
                        }
                    }
                }
            }

            let after: usize = (0..n).map(|i| store.size(self.starts[i])).sum();
            if after == before {
                break;
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
        est: Vec::new(),
        lct: Vec::new(),
        energy: Vec::new(),
    }));
}

// cumulative with variable durations, heights, and/or capacity.
// Time-tabling: profile sums min heights over mandatory parts
// [latest_start, earliest_start + min_dur); pruning uses cap.max and min_dur,
// peak min-usage raises cap.min. Weaker than the fixed-height edge-finder above.
// TODO(strong): variable-resource edge-finding / energetic reasoning.

struct CumulativeVar {
    starts: Vec<VarId>,
    dur: Vec<VarId>,
    height: Vec<VarId>,
    capacity: VarId,
    profile: Vec<i64>,
    buf: Vec<i32>,
}

impl CumulativeVar {
    fn state(&self, store: &Store) -> usize {
        let n = self.starts.len();
        let mut s = store.size(self.capacity);
        for i in 0..n {
            s += store.size(self.starts[i]) + store.size(self.dur[i]) + store.size(self.height[i]);
        }
        s
    }
}

impl Propagator for CumulativeVar {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &s in &self.starts {
            store.subscribe(s, me, Event::BoundChange);
        }
        for &d in &self.dur {
            store.subscribe(d, me, Event::BoundChange);
        }
        for &h in &self.height {
            store.subscribe(h, me, Event::BoundChange);
        }
        store.subscribe(self.capacity, me, Event::BoundChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.starts.len();
        if n == 0 {
            return Ok(());
        }
        loop {
            let before = self.state(store);
            let cap_max = store.max(self.capacity) as i64;

            // Time horizon over all tasks (longest possible durations).
            let mut hmin = i64::MAX;
            let mut hmax = i64::MIN;
            for i in 0..n {
                hmin = hmin.min(store.min(self.starts[i]) as i64);
                hmax = hmax.max(store.max(self.starts[i]) as i64 + store.max(self.dur[i]) as i64);
            }
            if hmax <= hmin {
                break;
            }
            let horizon = (hmax - hmin) as usize;
            self.profile.clear();
            self.profile.resize(horizon, 0);

            // Profile of min heights over mandatory parts.
            for i in 0..n {
                let h_lo = store.min(self.height[i]) as i64;
                if h_lo == 0 {
                    continue;
                }
                let mand_start = store.max(self.starts[i]) as i64;
                let mand_end = store.min(self.starts[i]) as i64 + store.min(self.dur[i]) as i64;
                for t in mand_start..mand_end {
                    self.profile[(t - hmin) as usize] += h_lo;
                }
            }

            // Overload check; tighten cap.min up to the peak.
            let peak = self.profile.iter().copied().max().unwrap_or(0);
            if peak > cap_max {
                return Err(Inconsistency);
            }
            if peak > store.min(self.capacity) as i64 {
                store.remove_below(self.capacity, clamp_i32(peak))?;
            }

            for i in 0..n {
                let h_lo = store.min(self.height[i]) as i64;
                let d_lo = store.min(self.dur[i]) as i64;
                let mand_start = store.max(self.starts[i]) as i64;
                let mand_end = store.min(self.starts[i]) as i64 + d_lo;

                // During mandatory part, h_i ≤ cap_max − others' min usage.
                if mand_end > mand_start {
                    let mut slack = i64::MAX;
                    for t in mand_start..mand_end {
                        let others = self.profile[(t - hmin) as usize] - h_lo;
                        slack = slack.min(cap_max - others);
                    }
                    if slack < store.max(self.height[i]) as i64 {
                        store.remove_above(self.height[i], clamp_i32(slack))?;
                    }
                }

                // Forbid starts whose least occupation [s, s+min_dur) exceeds cap_max.
                if h_lo > 0 && d_lo > 0 {
                    self.buf.clear();
                    self.buf.extend(store.values(self.starts[i]));
                    for &start in &self.buf {
                        let s = start as i64;
                        let conflict = (s..s + d_lo).any(|t| {
                            let idx = (t - hmin) as usize;
                            let own = if t >= mand_start && t < mand_end {
                                h_lo
                            } else {
                                0
                            };
                            self.profile[idx] - own + h_lo > cap_max
                        });
                        if conflict {
                            store.remove(self.starts[i], start)?;
                        }
                    }
                }
            }

            if self.state(store) == before {
                break;
            }
        }
        Ok(())
    }
}

/// Post `cumulative` with variable task durations and heights and a (possibly
/// variable) `capacity`.
pub fn cumulative_var(
    solver: &mut Solver,
    starts: &[VarId],
    durations: &[VarId],
    heights: &[VarId],
    capacity: VarId,
) {
    assert_eq!(starts.len(), durations.len(), "cumulative: length mismatch");
    assert_eq!(starts.len(), heights.len(), "cumulative: length mismatch");
    solver.post(Box::new(CumulativeVar {
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

/// `items[i]` = bin index; total size per bin must not exceed its capacity.
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

        // Committed load = sizes of items fixed to each bin.
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

/// Post `knapsack`: \( \sum_i \texttt{weights}[i] \cdot \texttt{vars}[i] \;\texttt{weight\_rel}\; \texttt{weight\_limit} \)
/// and \( \sum_i \texttt{profits}[i] \cdot \texttt{vars}[i] \;\texttt{profit\_rel}\; \texttt{profit\_limit} \).
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
