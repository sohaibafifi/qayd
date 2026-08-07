//! Scheduling constraints: `noOverlap`, `cumulative` (time-tabling),
//! `binPacking` (Shaw load pruning), `knapsack` (two linear constraints).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::constraints::linear::{clamp_i32, linear, Relation};
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Premise, Solver, Store};

/// Ceiling of `a / b` for `a >= 0`, `b > 0`.
fn ceil_div_pos(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// Run `$body` to a local fixpoint. `$measure` is a progress quantity that only
/// shrinks as domains shrink (typically summed domain size); the loop repeats
/// until a pass leaves it unchanged, then falls through. A propagator's own
/// filtering does not re-enqueue it, so without this a single pass can miss
/// inferences unlocked by an earlier removal in the same call. `$body` may
/// `?`/`return Err` to report a wipeout.
macro_rules! local_fixpoint {
    ($measure:expr, $body:block) => {
        loop {
            let before = $measure;
            $body
            if $measure == before {
                break;
            }
        }
    };
}

fn copy_slice_interruptible<T: Copy>(values: &[T], stop: &AtomicBool) -> Option<Vec<T>> {
    let mut copied = Vec::with_capacity(values.len());
    for &value in values {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        copied.push(value);
    }
    Some(copied)
}

fn register_variables_until(store: &mut Store, me: PropId, variables: &[VarId], event: Event, should_stop: &dyn Fn() -> bool) -> bool {
    for &variable in variables {
        if should_stop() {
            return false;
        }
        store.subscribe(variable, me, event);
    }
    !should_stop()
}

// ===========================================================================
// noOverlap (disjunctive, pairwise)
// ===========================================================================

/// Tasks `[start_i, start_i + duration_i)` must be pairwise non-overlapping.
#[derive(Clone)]
struct NoOverlap {
    starts: Vec<VarId>,
    durations: Vec<i64>,
}

impl Propagator for NoOverlap {
    fn priority(&self) -> Priority {
        Priority::Expensive
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &s in &self.starts {
            store.subscribe(s, me, Event::BoundChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        register_variables_until(store, me, &self.starts, Event::BoundChange, should_stop)
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.starts.len();
        local_fixpoint!(self.starts.iter().map(|&s| store.size(s)).sum::<usize>(), {
            for i in 0..n {
                for j in (i + 1)..n {
                    let (di, dj) = (self.durations[i], self.durations[j]);
                    let si_min = store.min(self.starts[i]) as i64;
                    let si_max = store.max(self.starts[i]) as i64;
                    let sj_min = store.min(self.starts[j]) as i64;
                    let sj_max = store.max(self.starts[j]) as i64;

                    let i_before_j = si_min + di <= sj_max;
                    let j_before_i = sj_min + dj <= si_max;

                    // Bound premises pinning each start's live window; every emitted
                    // reason is a subset of these four literals, so it can never
                    // exceed the whole-scope fallback width (no guard needed).
                    let ge_i = Premise::Ge { var: self.starts[i], bound: si_min as i32 };
                    let le_i = Premise::Le { var: self.starts[i], bound: si_max as i32 };
                    let ge_j = Premise::Ge { var: self.starts[j], bound: sj_min as i32 };
                    let le_j = Premise::Le { var: self.starts[j], bound: sj_max as i32 };
                    match (i_before_j, j_before_i) {
                        (false, false) => {
                            // Neither ordering fits: `si_min+di > sj_max` (ge_i+le_j
                            // kill i-before-j) and `sj_min+dj > si_max` (ge_j+le_i
                            // kill j-before-i); together no ordering survives.
                            if store.explaining() {
                                return Err(store.fail_because(vec![ge_i, le_i, ge_j, le_j]));
                            }
                            return Err(Inconsistency);
                        }
                        (true, false) => {
                            // i must precede j (j-before-i infeasible: sj_min+dj > si_max).
                            if store.explaining() {
                                store.remove_below_because(self.starts[j], clamp_i32(si_min + di), vec![ge_i, ge_j, le_i])?;
                                store.remove_above_because(self.starts[i], clamp_i32(sj_max - di), vec![le_j, ge_j, le_i])?;
                            } else {
                                store.remove_below(self.starts[j], clamp_i32(si_min + di))?;
                                store.remove_above(self.starts[i], clamp_i32(sj_max - di))?;
                            }
                        }
                        (false, true) => {
                            // j must precede i (i-before-j infeasible: si_min+di > sj_max).
                            if store.explaining() {
                                store.remove_below_because(self.starts[i], clamp_i32(sj_min + dj), vec![ge_j, ge_i, le_j])?;
                                store.remove_above_because(self.starts[j], clamp_i32(si_max - dj), vec![le_i, ge_i, le_j])?;
                            } else {
                                store.remove_below(self.starts[i], clamp_i32(sj_min + dj))?;
                                store.remove_above(self.starts[j], clamp_i32(si_max - dj))?;
                            }
                        }
                        (true, true) => {}
                    }
                }
            }
        });
        Ok(())
    }
}

/// Post `noOverlap`: the tasks must not overlap in time.
pub fn no_overlap(solver: &mut Solver, starts: &[VarId], durations: &[i64]) {
    assert_eq!(starts.len(), durations.len(), "noOverlap: starts/durations mismatch");
    solver.post(Box::new(NoOverlap { starts: starts.to_vec(), durations: durations.to_vec() }));
}

pub(crate) fn no_overlap_interruptible(solver: &mut Solver, starts: &[VarId], durations: &[i64], stop: &AtomicBool) -> bool {
    assert_eq!(starts.len(), durations.len(), "noOverlap: starts/durations mismatch");
    let Some(starts) = copy_slice_interruptible(starts, stop) else {
        return false;
    };
    let Some(durations) = copy_slice_interruptible(durations, stop) else {
        return false;
    };
    let should_stop = || stop.load(Ordering::Acquire);
    solver.post_until(Box::new(NoOverlap { starts, durations }), &should_stop).is_some()
}

// ===========================================================================
// cumulative (time-tabling)
// ===========================================================================

/// At every time point the total height of running tasks must not exceed `capacity`.
#[derive(Clone)]
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
    by_est: Vec<usize>,
    by_lct: Vec<usize>,
    lb: Vec<i64>,
}

impl Cumulative {
    /// Premises pinning a subset of tasks whose mandatory parts cover instant `t`
    /// and whose heights (plus `extra`) cross `capacity`. Each cited task `k` gets
    /// `Ge{min}`+`Le{max}`, forcing `[max, min+dur) ∋ t`, so `k` runs at `t`.
    /// Stops once the running sum exceeds `capacity` — any covering subset with
    /// height-sum > capacity is an unsatisfiable core, and the index-order stop
    /// keeps it narrower than the full covering set. `except` (the task being
    /// pushed) is skipped and its height passed as `extra`. Time-table pruning
    /// only raises mins (and `remove` only lowers maxes), so mandatory parts only
    /// grow within a pass; reading live bounds is thus a sound superset of the
    /// stale-profile cause (reason_before_step demotes any bound this call moved).
    fn overload_premises(&self, store: &Store, t: i64, except: Option<usize>, extra: i64) -> Vec<Premise> {
        let mut why = Vec::new();
        let mut sum = extra;
        for k in 0..self.starts.len() {
            // Zero-height tasks cover instants without consuming capacity:
            // citing them widens the reason for nothing.
            if Some(k) == except || self.height[k] == 0 {
                continue;
            }
            let kmin = store.min(self.starts[k]) as i64;
            let kmax = store.max(self.starts[k]) as i64;
            if kmax <= t && t < kmin + self.dur[k] {
                why.push(Premise::Ge { var: self.starts[k], bound: kmin as i32 });
                why.push(Premise::Le { var: self.starts[k], bound: kmax as i32 });
                sum += self.height[k];
                if sum > self.capacity {
                    break;
                }
            }
        }
        why
    }
}

impl Propagator for Cumulative {
    fn priority(&self) -> Priority {
        Priority::Expensive
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &s in &self.starts {
            store.subscribe(s, me, Event::BoundChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        register_variables_until(store, me, &self.starts, Event::BoundChange, should_stop)
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.starts.len();
        if n == 0 {
            return Ok(());
        }

        // Self-changes don't re-enqueue, so one pass could miss overloads
        // involving a just-fixed task.
        local_fixpoint!((0..n).map(|i| store.size(self.starts[i])).sum::<usize>(), {
            // Snapshot est, lct, energy.
            for i in 0..n {
                self.est[i] = store.min(self.starts[i]) as i64;
                self.lct[i] = store.max(self.starts[i]) as i64 + self.dur[i];
                self.energy[i] = self.height[i] * self.dur[i];
            }

            // Energetic overload check: window [L, U) energy > capacity*(U-L) fails.
            self.by_est.sort_unstable_by(|&a, &b| self.est[b].cmp(&self.est[a]));
            for u in 0..n {
                let ub = self.lct[u];
                let mut e = 0i64;
                for &k in &self.by_est {
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
            self.by_lct.sort_unstable_by(|&a, &b| self.lct[a].cmp(&self.lct[b]));
            self.lb.copy_from_slice(&self.est);
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let hi = self.height[i];
                if hi == 0 {
                    continue;
                }
                let mut e_omega = 0i64;
                let mut est_omega = i64::MAX;
                for &j in &self.by_lct {
                    if j == i {
                        continue;
                    }
                    e_omega += self.energy[j];
                    est_omega = est_omega.min(self.est[j]);
                    let u = self.lct[j];
                    if e_omega + self.energy[i] > self.capacity * (u - est_omega.min(self.est[i])) {
                        let rest = e_omega - (self.capacity - hi) * (u - est_omega);
                        if rest > 0 {
                            self.lb[i] = self.lb[i].max(est_omega + ceil_div_pos(rest, hi));
                        }
                    }
                }
            }
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                if self.lb[i] > self.est[i] {
                    store.remove_below(self.starts[i], clamp_i32(self.lb[i]))?;
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
                for (offset, &p) in self.profile.iter().enumerate() {
                    if p > self.capacity {
                        // Overload at instant t: cite a covering subset whose
                        // mandatory parts force Σ height > capacity there.
                        if store.explaining() {
                            let t = hmin + offset as i64;
                            return Err(store.fail_because(self.overload_premises(store, t, None, 0)));
                        }
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
                        let conflict = (s..s + self.dur[i]).find(|&t| {
                            let idx = (t - hmin) as usize;
                            let own = if t >= mand_start && t < mand_end { hi } else { 0 };
                            self.profile[idx] - own + hi > self.capacity
                        });
                        if let Some(t_c) = conflict {
                            // At t_c the other tasks' mandatory parts already leave
                            // < hi free, so i cannot occupy `start`. Cite that
                            // covering subset (i's own bound is not needed: the
                            // subset alone fills t_c beyond capacity - hi).
                            if store.explaining() {
                                let why = self.overload_premises(store, t_c, Some(i), hi);
                                // Empty premises only happen when `hi` alone busts the
                                // capacity (degenerate height > capacity): root-implied,
                                // but keep the scope fallback rather than an
                                // antecedent-free reason at depth.
                                if why.is_empty() {
                                    store.remove(self.starts[i], start)?;
                                } else {
                                    store.remove_because(self.starts[i], start, why)?;
                                }
                            } else {
                                store.remove(self.starts[i], start)?;
                            }
                        }
                    }
                }
            }
        });
        Ok(())
    }
}

/// Post `cumulative`: resource usage never exceeds `capacity`.
pub fn cumulative(solver: &mut Solver, starts: &[VarId], durations: &[i64], heights: &[i64], capacity: i64) {
    assert_eq!(starts.len(), durations.len(), "cumulative: length mismatch");
    assert_eq!(starts.len(), heights.len(), "cumulative: length mismatch");
    let n = starts.len();
    solver.post(Box::new(Cumulative {
        starts: starts.to_vec(),
        dur: durations.to_vec(),
        height: heights.to_vec(),
        capacity,
        profile: Vec::new(),
        buf: Vec::new(),
        est: vec![0; n],
        lct: vec![0; n],
        energy: vec![0; n],
        by_est: (0..n).collect(),
        by_lct: (0..n).collect(),
        lb: vec![0; n],
    }));
}

pub(crate) fn cumulative_interruptible(
    solver: &mut Solver,
    starts: &[VarId],
    durations: &[i64],
    heights: &[i64],
    capacity: i64,
    stop: &AtomicBool,
) -> bool {
    assert_eq!(starts.len(), durations.len(), "cumulative: length mismatch");
    assert_eq!(starts.len(), heights.len(), "cumulative: length mismatch");
    let Some(starts) = copy_slice_interruptible(starts, stop) else {
        return false;
    };
    let Some(dur) = copy_slice_interruptible(durations, stop) else {
        return false;
    };
    let Some(height) = copy_slice_interruptible(heights, stop) else {
        return false;
    };
    let n = starts.len();
    if stop.load(Ordering::Acquire) {
        return false;
    }
    let propagator = Cumulative {
        starts,
        dur,
        height,
        capacity,
        profile: Vec::new(),
        buf: Vec::new(),
        est: vec![0; n],
        lct: vec![0; n],
        energy: vec![0; n],
        by_est: (0..n).collect(),
        by_lct: (0..n).collect(),
        lb: vec![0; n],
    };
    let should_stop = || stop.load(Ordering::Acquire);
    solver.post_until(Box::new(propagator), &should_stop).is_some()
}

// cumulative with variable durations, heights, and/or capacity.
// Time-tabling: profile sums min heights over mandatory parts
// [latest_start, earliest_start + min_dur); pruning uses cap.max and min_dur,
// peak min-usage raises cap.min. Weaker than the fixed-height edge-finder above.
// TODO(strong): variable-resource edge-finding / energetic reasoning.

#[derive(Clone)]
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
    fn priority(&self) -> Priority {
        Priority::Expensive
    }

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

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        register_variables_until(store, me, &self.starts, Event::BoundChange, should_stop)
            && register_variables_until(store, me, &self.dur, Event::BoundChange, should_stop)
            && register_variables_until(store, me, &self.height, Event::BoundChange, should_stop)
            && {
                if should_stop() {
                    false
                } else {
                    store.subscribe(self.capacity, me, Event::BoundChange);
                    !should_stop()
                }
            }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.starts.len();
        if n == 0 {
            return Ok(());
        }
        local_fixpoint!(self.state(store), {
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
                            let own = if t >= mand_start && t < mand_end { h_lo } else { 0 };
                            self.profile[idx] - own + h_lo > cap_max
                        });
                        if conflict {
                            store.remove(self.starts[i], start)?;
                        }
                    }
                }
            }
        });
        Ok(())
    }
}

/// Post `cumulative` with variable task durations and heights and a (possibly
/// variable) `capacity`.
pub fn cumulative_var(solver: &mut Solver, starts: &[VarId], durations: &[VarId], heights: &[VarId], capacity: VarId) {
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

pub(crate) fn cumulative_var_interruptible(
    solver: &mut Solver,
    starts: &[VarId],
    durations: &[VarId],
    heights: &[VarId],
    capacity: VarId,
    stop: &AtomicBool,
) -> bool {
    assert_eq!(starts.len(), durations.len(), "cumulative: length mismatch");
    assert_eq!(starts.len(), heights.len(), "cumulative: length mismatch");
    let Some(starts) = copy_slice_interruptible(starts, stop) else {
        return false;
    };
    let Some(dur) = copy_slice_interruptible(durations, stop) else {
        return false;
    };
    let Some(height) = copy_slice_interruptible(heights, stop) else {
        return false;
    };
    let should_stop = || stop.load(Ordering::Acquire);
    solver
        .post_until(Box::new(CumulativeVar { starts, dur, height, capacity, profile: Vec::new(), buf: Vec::new() }), &should_stop)
        .is_some()
}

// ===========================================================================
// binPacking (Shaw, load-based)
// ===========================================================================

/// `items[i]` = bin index; total size per bin must not exceed its capacity.
#[derive(Clone)]
struct BinPacking {
    items: Vec<VarId>,
    sizes: Vec<i64>,
    capacities: Vec<i64>,
    load: Vec<i64>,
    buf: Vec<i32>,
}

impl Propagator for BinPacking {
    fn priority(&self) -> Priority {
        Priority::Expensive
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &it in &self.items {
            store.subscribe(it, me, Event::DomainChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        register_variables_until(store, me, &self.items, Event::DomainChange, should_stop)
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let nbins = self.capacities.len();
        local_fixpoint!(self.items.iter().map(|&it| store.size(it)).sum::<usize>(), {
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
        });
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

pub(crate) fn bin_packing_interruptible(
    solver: &mut Solver,
    items: &[VarId],
    sizes: &[i64],
    capacities: &[i64],
    stop: &AtomicBool,
) -> bool {
    assert_eq!(items.len(), sizes.len(), "binPacking: items/sizes mismatch");
    let Some(items) = copy_slice_interruptible(items, stop) else {
        return false;
    };
    let Some(sizes) = copy_slice_interruptible(sizes, stop) else {
        return false;
    };
    let Some(capacities) = copy_slice_interruptible(capacities, stop) else {
        return false;
    };
    let should_stop = || stop.load(Ordering::Acquire);
    solver.post_until(Box::new(BinPacking { items, sizes, capacities, load: Vec::new(), buf: Vec::new() }), &should_stop).is_some()
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

pub(crate) struct KnapsackInput<'a> {
    pub(crate) variables: &'a [VarId],
    pub(crate) weights: &'a [i64],
    pub(crate) profits: &'a [i64],
    pub(crate) weight_relation: Relation,
    pub(crate) weight_limit: i64,
    pub(crate) profit_relation: Relation,
    pub(crate) profit_limit: i64,
}

pub(crate) fn knapsack_interruptible(solver: &mut Solver, input: KnapsackInput<'_>, stop: &AtomicBool) -> bool {
    assert_eq!(input.variables.len(), input.weights.len(), "knapsack: vars/weights mismatch");
    assert_eq!(input.variables.len(), input.profits.len(), "knapsack: vars/profits mismatch");
    let Some(weight_variables) = copy_slice_interruptible(input.variables, stop) else {
        return false;
    };
    let Some(profit_variables) = copy_slice_interruptible(input.variables, stop) else {
        return false;
    };
    let Some(weights) = copy_slice_interruptible(input.weights, stop) else {
        return false;
    };
    let Some(profits) = copy_slice_interruptible(input.profits, stop) else {
        return false;
    };
    crate::constraints::linear::linear_interruptible(solver, weights, weight_variables, input.weight_relation, input.weight_limit, stop)
        && crate::constraints::linear::linear_interruptible(
            solver,
            profits,
            profit_variables,
            input.profit_relation,
            input.profit_limit,
            stop,
        )
}
