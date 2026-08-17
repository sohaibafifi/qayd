//! Scheduling constraints: `noOverlap`, `cumulative` (time-tabling),
//! `binPacking` (Shaw load pruning), `knapsack` (two linear constraints).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::constraints::interval as interval_constraints;
use crate::constraints::linear::{linear, Relation};
use crate::constraints::resource_profile::{
    build_profile, earliest_feasible_start, first_overload, latest_feasible_start, mandatory_height_limit, peak_usage, ProfileSegment,
};
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Premise, Solver, Store};

/// Ceiling of `a / b` for `a >= 0`, `b > 0`.
fn ceil_div_pos(a: i128, b: i128) -> i128 {
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

fn clamp_i128_i32(value: i128) -> i32 {
    value.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
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

fn propagate_no_overlap_bounds(
    store: &mut Store,
    first: VarId,
    first_duration: i64,
    second: VarId,
    second_duration: i64,
) -> Result<(), Inconsistency> {
    let first_min_value = store.min(first);
    let first_max_value = store.max(first);
    let second_min_value = store.min(second);
    let second_max_value = store.max(second);
    let first_min = i128::from(first_min_value);
    let first_max = i128::from(first_max_value);
    let second_min = i128::from(second_min_value);
    let second_max = i128::from(second_max_value);
    let first_duration = i128::from(first_duration);
    let second_duration = i128::from(second_duration);

    let first_before_second = first_min + first_duration <= second_max;
    let second_before_first = second_min + second_duration <= first_max;

    // Bound premises pin each start's live window. Every emitted reason is a
    // subset of these four literals.
    let ge_first = Premise::Ge { var: first, bound: first_min_value };
    let le_first = Premise::Le { var: first, bound: first_max_value };
    let ge_second = Premise::Ge { var: second, bound: second_min_value };
    let le_second = Premise::Le { var: second, bound: second_max_value };
    match (first_before_second, second_before_first) {
        (false, false) => {
            if store.explaining() {
                return Err(store.fail_because(vec![ge_first, le_first, ge_second, le_second]));
            }
            return Err(Inconsistency);
        }
        (true, false) => {
            // The second task cannot precede the first, so the first must end
            // before the second starts.
            if store.explaining() {
                store.remove_below_because(second, clamp_i128_i32(first_min + first_duration), vec![ge_first, ge_second, le_first])?;
                store.remove_above_because(first, clamp_i128_i32(second_max - first_duration), vec![le_second, ge_second, le_first])?;
            } else {
                store.remove_below(second, clamp_i128_i32(first_min + first_duration))?;
                store.remove_above(first, clamp_i128_i32(second_max - first_duration))?;
            }
        }
        (false, true) => {
            // The first task cannot precede the second, so the second must end
            // before the first starts.
            if store.explaining() {
                store.remove_below_because(first, clamp_i128_i32(second_min + second_duration), vec![ge_second, ge_first, le_second])?;
                store.remove_above_because(second, clamp_i128_i32(first_max - second_duration), vec![le_first, ge_first, le_second])?;
            } else {
                store.remove_below(first, clamp_i128_i32(second_min + second_duration))?;
                store.remove_above(second, clamp_i128_i32(first_max - second_duration))?;
            }
        }
        (true, true) => {}
    }
    Ok(())
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
        self.propagate_core(store, None)
    }

    fn propagate_until(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        self.propagate_core(store, Some(should_stop))
    }
}

impl NoOverlap {
    fn propagate_core(&mut self, store: &mut Store, should_stop: Option<&dyn Fn() -> bool>) -> Result<(), Inconsistency> {
        let n = self.starts.len();
        loop {
            if should_stop.is_some_and(|stop| stop()) {
                return Ok(());
            }
            let before = self.starts.iter().map(|&start| store.size(start)).sum::<usize>();
            for i in 0..n {
                if should_stop.is_some_and(|stop| stop()) {
                    return Ok(());
                }
                for j in (i + 1)..n {
                    if should_stop.is_some_and(|stop| stop()) {
                        return Ok(());
                    }
                    propagate_no_overlap_bounds(store, self.starts[i], self.durations[i], self.starts[j], self.durations[j])?;
                }
            }
            let after = self.starts.iter().map(|&start| store.size(start)).sum::<usize>();
            if after == before {
                return Ok(());
            }
        }
    }
}

/// Post `noOverlap`: the tasks must not overlap in time.
pub fn no_overlap(solver: &mut Solver, starts: &[VarId], durations: &[i64]) {
    assert_eq!(starts.len(), durations.len(), "noOverlap: starts/durations mismatch");
    if let Some(durations) = durations.iter().map(|&duration| i32::try_from(duration).ok()).collect::<Option<Vec<_>>>() {
        let intervals = starts
            .iter()
            .zip(durations)
            .map(|(&start, duration)| solver.store.register_interval(start, duration, None))
            .collect::<Vec<_>>();
        interval_constraints::mandatory_no_overlap(solver, &intervals);
    } else {
        solver.post(Box::new(NoOverlap { starts: starts.to_vec(), durations: durations.to_vec() }));
    }
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
    if let Some(durations) = durations.iter().map(|&duration| i32::try_from(duration).ok()).collect::<Option<Vec<_>>>() {
        let mut intervals = Vec::with_capacity(starts.len());
        for (&start, duration) in starts.iter().zip(durations) {
            if should_stop() {
                return false;
            }
            intervals.push(solver.store.register_interval(start, duration, None));
        }
        interval_constraints::mandatory_no_overlap_until(solver, intervals, &should_stop).is_some()
    } else {
        solver.post_until(Box::new(NoOverlap { starts, durations }), &should_stop).is_some()
    }
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
    events: Vec<(i128, i128)>,
    profile: Vec<ProfileSegment>,
    buf: Vec<i32>,
    /// Reused scratch for energetic overload checking.
    est: Vec<i128>,
    lct: Vec<i128>,
    energy: Vec<i128>,
    by_est: Vec<usize>,
    by_lct: Vec<usize>,
    lb: Vec<i128>,
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
    fn overload_premises(&self, store: &Store, t: i128, except: Option<usize>, extra: i128) -> Vec<Premise> {
        let mut why = Vec::new();
        let mut sum = extra;
        for k in 0..self.starts.len() {
            // Zero-height tasks cover instants without consuming capacity:
            // citing them widens the reason for nothing.
            if Some(k) == except || self.height[k] == 0 {
                continue;
            }
            let kmin = i128::from(store.min(self.starts[k]));
            let kmax = i128::from(store.max(self.starts[k]));
            if kmax <= t && t < kmin + i128::from(self.dur[k]) {
                why.push(Premise::Ge { var: self.starts[k], bound: store.min(self.starts[k]) });
                why.push(Premise::Le { var: self.starts[k], bound: store.max(self.starts[k]) });
                sum += i128::from(self.height[k]);
                if sum > i128::from(self.capacity) {
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
                self.est[i] = i128::from(store.min(self.starts[i]));
                self.lct[i] = i128::from(store.max(self.starts[i])) + i128::from(self.dur[i]);
                self.energy[i] = i128::from(self.height[i]) * i128::from(self.dur[i]);
            }

            // Energetic overload check: window [L, U) energy > capacity*(U-L) fails.
            self.by_est.sort_unstable_by(|&a, &b| self.est[b].cmp(&self.est[a]));
            for u in 0..n {
                let ub = self.lct[u];
                let mut e = 0i128;
                for &k in &self.by_est {
                    if self.lct[k] <= ub {
                        e += self.energy[k];
                        if e > i128::from(self.capacity) * (ub - self.est[k]) {
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
                let hi = i128::from(self.height[i]);
                if hi == 0 {
                    continue;
                }
                let mut e_omega = 0i128;
                let mut est_omega = i128::MAX;
                for &j in &self.by_lct {
                    if j == i {
                        continue;
                    }
                    e_omega += self.energy[j];
                    est_omega = est_omega.min(self.est[j]);
                    let u = self.lct[j];
                    if e_omega + self.energy[i] > i128::from(self.capacity) * (u - est_omega.min(self.est[i])) {
                        let rest = e_omega - (i128::from(self.capacity) - hi) * (u - est_omega);
                        if rest > 0 {
                            self.lb[i] = self.lb[i].max(est_omega + ceil_div_pos(rest, hi));
                        }
                    }
                }
            }
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                if self.lb[i] > self.est[i] {
                    store.remove_below(self.starts[i], clamp_i128_i32(self.lb[i]))?;
                }
            }

            // Sparse time-tabling over mandatory-part events. Memory is O(n),
            // independent of the numeric horizon.
            build_profile(
                (0..n).map(|i| {
                    (
                        i128::from(store.max(self.starts[i])),
                        i128::from(store.min(self.starts[i])) + i128::from(self.dur[i]),
                        i128::from(self.height[i]),
                    )
                }),
                &mut self.events,
                &mut self.profile,
            );
            if let Some(segment) = self.profile.iter().find(|segment| segment.usage > i128::from(self.capacity)) {
                if store.explaining() {
                    return Err(store.fail_because(self.overload_premises(store, segment.start, None, 0)));
                }
                return Err(Inconsistency);
            }
            for i in 0..n {
                let hi = i128::from(self.height[i]);
                let mand_start = i128::from(store.max(self.starts[i]));
                let mand_end = i128::from(store.min(self.starts[i])) + i128::from(self.dur[i]);
                if !store.explaining() {
                    let earliest = i128::from(store.min(self.starts[i]));
                    let latest = i128::from(store.max(self.starts[i]));
                    let duration = i128::from(self.dur[i]);
                    let own_part = Some((mand_start, mand_end, hi));
                    let capacity = i128::from(self.capacity);
                    let Some(new_min) = earliest_feasible_start(&self.profile, earliest, latest, duration, hi, own_part, capacity) else {
                        return Err(Inconsistency);
                    };
                    let Some(new_max) = latest_feasible_start(&self.profile, new_min, latest, duration, hi, own_part, capacity) else {
                        return Err(Inconsistency);
                    };
                    store.remove_below(self.starts[i], clamp_i128_i32(new_min))?;
                    store.remove_above(self.starts[i], clamp_i128_i32(new_max))?;
                    continue;
                }
                self.buf.clear();
                self.buf.extend(store.values(self.starts[i]));
                for &start in &self.buf {
                    let start_time = i128::from(start);
                    let conflict = first_overload(
                        &self.profile,
                        start_time,
                        start_time + i128::from(self.dur[i]),
                        hi,
                        Some((mand_start, mand_end, hi)),
                        i128::from(self.capacity),
                    );
                    if let Some((t_c, _)) = conflict {
                        if store.explaining() {
                            let why = self.overload_premises(store, t_c, Some(i), hi);
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
        events: Vec::new(),
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
        events: Vec::new(),
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
    events: Vec<(i128, i128)>,
    profile: Vec<ProfileSegment>,
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
            let cap_max = i128::from(store.max(self.capacity));
            build_profile(
                (0..n).map(|i| {
                    (
                        i128::from(store.max(self.starts[i])),
                        i128::from(store.min(self.starts[i])) + i128::from(store.min(self.dur[i])),
                        i128::from(store.min(self.height[i])),
                    )
                }),
                &mut self.events,
                &mut self.profile,
            );

            // Overload check; tighten cap.min up to the peak.
            let peak = peak_usage(&self.profile);
            if peak > cap_max {
                return Err(Inconsistency);
            }
            if peak > i128::from(store.min(self.capacity)) {
                store.remove_below(self.capacity, clamp_i128_i32(peak))?;
            }

            for i in 0..n {
                let h_lo = i128::from(store.min(self.height[i]));
                let d_lo = i128::from(store.min(self.dur[i]));
                let mand_start = i128::from(store.max(self.starts[i]));
                let mand_end = i128::from(store.min(self.starts[i])) + d_lo;

                // During mandatory part, h_i ≤ cap_max − others' min usage.
                if mand_end > mand_start {
                    let slack = mandatory_height_limit(&self.profile, mand_start, mand_end, h_lo, cap_max);
                    if slack < i128::from(store.max(self.height[i])) {
                        store.remove_above(self.height[i], clamp_i128_i32(slack))?;
                    }
                }

                // Forbid starts whose least occupation [s, s+min_dur) exceeds cap_max.
                if h_lo > 0 && d_lo > 0 {
                    let earliest = i128::from(store.min(self.starts[i]));
                    let latest = i128::from(store.max(self.starts[i]));
                    let own_part = Some((mand_start, mand_end, h_lo));
                    let Some(new_min) = earliest_feasible_start(&self.profile, earliest, latest, d_lo, h_lo, own_part, cap_max) else {
                        return Err(Inconsistency);
                    };
                    let Some(new_max) = latest_feasible_start(&self.profile, new_min, latest, d_lo, h_lo, own_part, cap_max) else {
                        return Err(Inconsistency);
                    };
                    store.remove_below(self.starts[i], clamp_i128_i32(new_min))?;
                    store.remove_above(self.starts[i], clamp_i128_i32(new_max))?;
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
        events: Vec::new(),
        profile: Vec::new(),
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
        .post_until(Box::new(CumulativeVar { starts, dur, height, capacity, events: Vec::new(), profile: Vec::new() }), &should_stop)
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
    load: Vec<i128>,
    buf: Vec<i32>,
    domains: Vec<Vec<usize>>,
    subsets: Vec<Vec<usize>>,
    subset_mask: Vec<bool>,
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
        if nbins == 0 {
            return if self.items.is_empty() { Ok(()) } else { Err(Inconsistency) };
        }
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
                    self.load[store.value(it) as usize] += i128::from(self.sizes[i]);
                }
            }
            for b in 0..nbins {
                if self.load[b] > i128::from(self.capacities[b]) {
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
                    if self.load[b as usize] + i128::from(self.sizes[i]) > i128::from(self.capacities[b as usize]) {
                        store.remove(self.items[i], b)?;
                    }
                }
            }

            // Every item must be assigned once, so aggregate capacity is a
            // necessary global bound even before any item is fixed.
            let total_size = self.sizes.iter().fold(0i128, |sum, &size| sum + i128::from(size));
            let total_capacity = self.capacities.iter().fold(0i128, |sum, &capacity| sum + i128::from(capacity));
            if total_size > total_capacity {
                return Err(Inconsistency);
            }

            // Shaw-style load subsets derived from live item domains. For a
            // bin set B, every item whose domain is contained in B contributes
            // mandatory energy to B. Overload fails; exact saturation removes
            // B from every other positive-size item.
            if self.domains.len() < self.items.len() {
                self.domains.resize_with(self.items.len(), Vec::new);
            }
            self.subsets.clear();
            for (i, &item) in self.items.iter().enumerate() {
                self.domains[i].clear();
                self.domains[i].extend(store.values(item).map(|bin| bin as usize));
            }
            self.subsets.extend(self.domains.iter().take(self.items.len()).cloned());
            self.subsets.sort_unstable();
            self.subsets.dedup();
            self.subset_mask.resize(nbins, false);
            for subset in &self.subsets {
                self.subset_mask.fill(false);
                for &bin in subset {
                    self.subset_mask[bin] = true;
                }
                let capacity = subset.iter().fold(0i128, |sum, &bin| sum + i128::from(self.capacities[bin]));
                let mut mandatory = 0i128;
                for (i, domain) in self.domains.iter().take(self.items.len()).enumerate() {
                    if domain.iter().all(|&bin| self.subset_mask[bin]) {
                        mandatory += i128::from(self.sizes[i]);
                    }
                }
                if mandatory > capacity {
                    return Err(Inconsistency);
                }
                if mandatory == capacity {
                    for i in 0..self.items.len() {
                        if self.sizes[i] == 0 || self.domains[i].iter().all(|&bin| self.subset_mask[bin]) {
                            continue;
                        }
                        self.buf.clear();
                        self.buf.extend(store.values(self.items[i]).filter(|&bin| self.subset_mask[bin as usize]));
                        for bin in self.buf.drain(..) {
                            store.remove(self.items[i], bin)?;
                        }
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
        domains: Vec::new(),
        subsets: Vec::new(),
        subset_mask: Vec::new(),
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
    solver
        .post_until(
            Box::new(BinPacking {
                items,
                sizes,
                capacities,
                load: Vec::new(),
                buf: Vec::new(),
                domains: Vec::new(),
                subsets: Vec::new(),
                subset_mask: Vec::new(),
            }),
            &should_stop,
        )
        .is_some()
}

/// Native bin loads: `loads[b]` is exactly the total size of items assigned to
/// bin `b`. Bounds and item domains are propagated in both directions without
/// one-hot indicator variables.
#[derive(Clone)]
struct BinLoads {
    items: Vec<VarId>,
    sizes: Vec<i64>,
    loads: Vec<VarId>,
    committed: Vec<i128>,
    possible: Vec<i128>,
    buf: Vec<i32>,
}

impl BinLoads {
    fn state(&self, store: &Store) -> usize {
        self.items.iter().chain(&self.loads).fold(0usize, |state, &variable| state.saturating_add(store.size(variable)))
    }
}

impl Propagator for BinLoads {
    fn priority(&self) -> Priority {
        Priority::Expensive
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &item in &self.items {
            store.subscribe(item, me, Event::DomainChange);
        }
        for &load in &self.loads {
            store.subscribe(load, me, Event::BoundChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        register_variables_until(store, me, &self.items, Event::DomainChange, should_stop)
            && register_variables_until(store, me, &self.loads, Event::BoundChange, should_stop)
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let bins = self.loads.len();
        if bins == 0 {
            return if self.items.is_empty() { Ok(()) } else { Err(Inconsistency) };
        }
        local_fixpoint!(self.state(store), {
            for &item in &self.items {
                store.remove_below(item, 0)?;
                store.remove_above(item, bins as i32 - 1)?;
            }

            self.committed.fill(0);
            self.possible.fill(0);
            for (index, &item) in self.items.iter().enumerate() {
                let size = i128::from(self.sizes[index]);
                if store.is_fixed(item) {
                    self.committed[store.value(item) as usize] += size;
                }
                for bin in store.values(item) {
                    self.possible[bin as usize] += size;
                }
            }
            for bin in 0..bins {
                if self.committed[bin] > i128::from(store.max(self.loads[bin]))
                    || self.possible[bin] < i128::from(store.min(self.loads[bin]))
                {
                    return Err(Inconsistency);
                }
                store.remove_below(self.loads[bin], clamp_i128_i32(self.committed[bin]))?;
                store.remove_above(self.loads[bin], clamp_i128_i32(self.possible[bin]))?;
            }

            // Loads partition the total item size exactly.
            let total = self.sizes.iter().fold(0i128, |sum, &size| sum + i128::from(size));
            let sum_min = self.loads.iter().fold(0i128, |sum, &load| sum + i128::from(store.min(load)));
            let sum_max = self.loads.iter().fold(0i128, |sum, &load| sum + i128::from(store.max(load)));
            if total < sum_min || total > sum_max {
                return Err(Inconsistency);
            }
            for &load in &self.loads {
                let min = i128::from(store.min(load));
                let max = i128::from(store.max(load));
                store.remove_below(load, clamp_i128_i32(total - (sum_max - max)))?;
                store.remove_above(load, clamp_i128_i32(total - (sum_min - min)))?;
            }

            for index in 0..self.items.len() {
                if store.is_fixed(self.items[index]) || self.sizes[index] == 0 {
                    continue;
                }
                let size = i128::from(self.sizes[index]);
                self.buf.clear();
                self.buf.extend(store.values(self.items[index]));
                let mut required = None;
                for &bin in &self.buf {
                    let bin = bin as usize;
                    if self.committed[bin] + size > i128::from(store.max(self.loads[bin])) {
                        store.remove(self.items[index], bin as i32)?;
                    } else if self.possible[bin] - size < i128::from(store.min(self.loads[bin]))
                        && required.replace(bin as i32).is_some_and(|other| other != bin as i32)
                    {
                        return Err(Inconsistency);
                    }
                }
                if let Some(bin) = required {
                    store.fix(self.items[index], bin)?;
                }
            }
        });
        Ok(())
    }
}

pub fn bin_loads(solver: &mut Solver, items: &[VarId], sizes: &[i64], loads: &[VarId]) {
    assert_eq!(items.len(), sizes.len(), "binLoads: items/sizes mismatch");
    solver.post(Box::new(BinLoads {
        items: items.to_vec(),
        sizes: sizes.to_vec(),
        loads: loads.to_vec(),
        committed: vec![0; loads.len()],
        possible: vec![0; loads.len()],
        buf: Vec::new(),
    }));
}

pub(crate) fn bin_loads_interruptible(solver: &mut Solver, items: &[VarId], sizes: &[i64], loads: &[VarId], stop: &AtomicBool) -> bool {
    assert_eq!(items.len(), sizes.len(), "binLoads: items/sizes mismatch");
    let Some(items) = copy_slice_interruptible(items, stop) else {
        return false;
    };
    let Some(sizes) = copy_slice_interruptible(sizes, stop) else {
        return false;
    };
    let Some(loads) = copy_slice_interruptible(loads, stop) else {
        return false;
    };
    let bins = loads.len();
    let should_stop = || stop.load(Ordering::Acquire);
    solver
        .post_until(
            Box::new(BinLoads { items, sizes, loads, committed: vec![0; bins], possible: vec![0; bins], buf: Vec::new() }),
            &should_stop,
        )
        .is_some()
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
