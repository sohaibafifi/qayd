//! Propagators over interval domains (scheduling).

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use crate::constraints::cumulative_profile::{CumulativeStats, IncrementalProfile, ProfileUpdate};
use crate::constraints::resource_profile::{
    build_profile, earliest_feasible_start, EnergeticAnalysis, EnergeticWorkspace, FixedCumulativeTask, ProfileSegment,
};
use crate::domains::interval::{IntervalEvent, IntervalPresence};
use crate::ids::{IntervalId, PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Premise, Solver, Store};

/// Structured fixed-duration interval precedence.
///
/// If both intervals are present, `before` must end no later than `after`
/// starts. If either interval is absent, the constraint is inactive.
#[derive(Clone)]
pub struct IntervalPrecedence {
    before: IntervalId,
    after: IntervalId,
}

impl Propagator for IntervalPrecedence {
    fn priority(&self) -> Priority {
        Priority::Cheap
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe_interval(self.before, me, IntervalEvent::EndBoundChange);
        store.subscribe_interval(self.before, me, IntervalEvent::PresenceChange);
        store.subscribe_interval(self.after, me, IntervalEvent::StartBoundChange);
        store.subscribe_interval(self.after, me, IntervalEvent::PresenceChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let before_presence = store.interval_presence(self.before);
        let after_presence = store.interval_presence(self.after);
        if before_presence == IntervalPresence::Absent || after_presence == IntervalPresence::Absent {
            return Ok(());
        }

        let before_start = store.interval_start_var(self.before);
        let after_start = store.interval_start_var(self.after);
        let before_lb = store.interval_start_min(self.before);
        let after_ub = store.interval_start_max(self.after);
        let duration = store.interval_duration(self.before);
        // `before` ends after `after`'s latest start: the precedence cannot hold.
        let before_too_late = before_lb.saturating_add(duration) > after_ub;
        let present = IntervalPresence::Present;
        let optional = IntervalPresence::Optional;

        if before_too_late {
            // Reason: `before.start >= before_lb` and `after.start <= after_ub`,
            // plus the presence of whichever intervals are optional.
            let mut why = vec![Premise::Ge { var: before_start, bound: before_lb }, Premise::Le { var: after_start, bound: after_ub }];
            match (before_presence, after_presence) {
                (p, q) if p == present && q == present => {
                    why.extend(present_premise(store, self.before));
                    why.extend(present_premise(store, self.after));
                    return Err(store.fail_because(why));
                }
                (p, q) if p == present && q == optional => {
                    why.extend(present_premise(store, self.before));
                    store.forbid_interval_presence_because(self.after, why)?;
                    return Ok(());
                }
                (p, q) if p == optional && q == present => {
                    why.extend(present_premise(store, self.after));
                    store.forbid_interval_presence_because(self.before, why)?;
                    return Ok(());
                }
                _ => return Ok(()),
            }
        }

        if before_presence == present && after_presence == present {
            // after.start >= before.start_min + duration(before).
            let mut why_after = vec![Premise::Ge { var: before_start, bound: before_lb }];
            why_after.extend(present_premise(store, self.before));
            why_after.extend(present_premise(store, self.after));
            store.set_interval_start_min_because(self.after, before_lb.saturating_add(duration), why_after)?;

            // before.start <= after.start_max - duration(before).
            let mut why_before = vec![Premise::Le { var: after_start, bound: after_ub }];
            why_before.extend(present_premise(store, self.before));
            why_before.extend(present_premise(store, self.after));
            store.set_interval_start_max_because(self.before, after_ub.saturating_sub(duration), why_before)?;
        }

        Ok(())
    }
}

/// `[present(interval)]` as a premise: an optional interval cites its presence
/// variable; a mandatory interval is present at the root and cites nothing.
fn present_premise(store: &Store, interval: IntervalId) -> Option<Premise> {
    store.interval_presence_var(interval).map(|var| Premise::Eq { var, val: 1 })
}

/// Premises fixing the cumulative mandatory-part profile over `[lo, hi)`: every
/// present interval (except `except`) whose compulsory region intersects the
/// window, citing its presence and start bounds. A superset of the minimal cause
/// is still sound (it only weakens the learned clause), and pinning each start's
/// bounds pins its compulsory region and hence its contribution to the profile.
fn cumulative_profile_premises(store: &Store, intervals: &[IntervalId], lo: i128, hi: i128, except: IntervalId) -> Vec<Premise> {
    let mut why = Vec::new();
    for &iv in intervals {
        if iv == except || store.interval_presence(iv) != IntervalPresence::Present {
            continue;
        }
        let cp_lo = i128::from(store.interval_start_max(iv));
        let cp_hi = i128::from(store.interval_start_min(iv)) + i128::from(store.interval_duration(iv));
        if cp_lo < hi && lo < cp_hi {
            why.extend(present_premise(store, iv));
            why.push(Premise::Ge { var: store.interval_start_var(iv), bound: store.interval_start_min(iv) });
            why.push(Premise::Le { var: store.interval_start_var(iv), bound: store.interval_start_max(iv) });
        }
    }
    why
}

/// A native timetable deduction's complete integer-start coverage obligation.
#[derive(Clone, Copy)]
pub(crate) enum TimetableCertificate {
    Overload { time: i128 },
    LowerBound { occurrence: usize, bound: i128 },
    NoFit { occurrence: usize },
}

pub(crate) struct LiftedCumulativeReason {
    pub(crate) premises: Vec<Premise>,
    pub(crate) lifted_bounds: u64,
    pub(crate) omitted_tautologies: u64,
}

/// This is an explanation constructor, not an additional scheduling filter.
/// `None` preserves the original reason when a coverage certificate cannot be
/// reconstructed. `Err(())` is cancellation and must never become a conflict.
pub(crate) struct CumulativeReasonContext<'a> {
    pub(crate) store: &'a Store,
    pub(crate) intervals: &'a [IntervalId],
    pub(crate) demands: &'a [i32],
    pub(crate) capacity: i32,
    pub(crate) profile: &'a [ProfileSegment],
}

impl CumulativeReasonContext<'_> {
    fn cover_window(
        &self,
        window: (i128, i128),
        target: Option<usize>,
        target_demand: i128,
        covered: &mut [Option<(i128, i128)>],
        stop: Option<&dyn Fn() -> bool>,
    ) -> Result<bool, ()> {
        let mut usage = target_demand;
        if usage > i128::from(self.capacity) {
            return Ok(true);
        }
        for (occurrence, &interval) in self.intervals.iter().enumerate() {
            if interruption_requested(stop) {
                return Err(());
            }
            // Exclude one occurrence, rather than every equal IntervalId.
            // All other occurrences retain their demand and presence semantics.
            if Some(occurrence) == target || self.store.interval_presence(interval) != IntervalPresence::Present {
                continue;
            }
            let demand = i128::from(self.demands[occurrence]);
            let duration = i128::from(self.store.interval_duration(interval));
            if demand <= 0 || duration <= 0 {
                continue;
            }
            let start = i128::from(self.store.interval_start_max(interval));
            let end = i128::from(self.store.interval_start_min(interval)) + duration;
            if start <= window.0 && window.1 <= end {
                covered[occurrence] =
                    Some(covered[occurrence].map_or(window, |previous| (previous.0.min(window.0), previous.1.max(window.1))));
                usage += demand;
                if usage > i128::from(self.capacity) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    pub(crate) fn explain(
        &self,
        certificate: TimetableCertificate,
        stop: Option<&dyn Fn() -> bool>,
    ) -> Result<Option<LiftedCumulativeReason>, ()> {
        if interruption_requested(stop) {
            return Err(());
        }
        if self.intervals.len() != self.demands.len() || self.capacity < 0 {
            return Ok(None);
        }
        let mut covered = vec![None; self.intervals.len()];
        let mut target_bounds = None;
        match certificate {
            TimetableCertificate::Overload { time } => {
                // Integer starts and durations make a single unit window an
                // exact overload witness, even beyond the i32 time horizon.
                if !self.cover_window((time, time + 1), None, 0, &mut covered, stop)? {
                    return Ok(None);
                }
            }
            TimetableCertificate::LowerBound { occurrence, .. } | TimetableCertificate::NoFit { occurrence } => {
                let Some(&interval) = self.intervals.get(occurrence) else { return Ok(None) };
                if self.store.interval_presence(interval) != IntervalPresence::Present {
                    return Ok(None);
                }
                let duration = i128::from(self.store.interval_duration(interval));
                let demand = i128::from(self.demands[occurrence]);
                if duration <= 0 || demand <= 0 {
                    return Ok(None);
                }
                let lower = i128::from(self.store.interval_start_min(interval));
                let upper = i128::from(self.store.interval_start_max(interval));
                let end = match certificate {
                    TimetableCertificate::LowerBound { bound, .. } if lower < bound && bound <= upper => bound,
                    TimetableCertificate::NoFit { .. } => upper + 1,
                    _ => return Ok(None),
                };
                let mut candidate = lower;
                let mut lifted_lower = lower;
                while candidate < end {
                    if interruption_requested(stop) {
                        return Err(());
                    }
                    // Replay the same first-overload jumps as the unchanged
                    // timetable filter. A certificate cannot skip a removed
                    // integer start, including between separated segments.
                    let own_start = upper;
                    let own_end = lower + duration;
                    let mut conflict_end = None;
                    if demand > i128::from(self.capacity) {
                        conflict_end = Some(end);
                    } else {
                        for segment in self.profile {
                            if interruption_requested(stop) {
                                return Err(());
                            }
                            if candidate.max(segment.start) >= (candidate + duration).min(segment.end) {
                                continue;
                            }
                            let own = if own_start <= segment.start && segment.end <= own_end { demand } else { 0 };
                            if segment.usage - own + demand > i128::from(self.capacity) {
                                conflict_end = Some(segment.end);
                                break;
                            }
                        }
                    }
                    let Some(conflict_end) = conflict_end else { return Ok(None) };
                    let window_end = conflict_end.min(end);
                    if window_end <= candidate {
                        return Ok(None);
                    }
                    let window_start = (candidate + duration - 1).min(window_end - 1);
                    if !self.cover_window((window_start, window_end), Some(occurrence), demand, &mut covered, stop)? {
                        return Ok(None);
                    }
                    if candidate == lower {
                        // Every start at least this bound intersects the first
                        // certified window; this may weaken the old lower bound.
                        lifted_lower = window_start - duration + 1;
                    }
                    candidate = window_end;
                }
                target_bounds = Some((interval, lifted_lower, matches!(certificate, TimetableCertificate::NoFit { .. })));
            }
        }
        let mut reason = LiftedCumulativeReason { premises: Vec::new(), lifted_bounds: 0, omitted_tautologies: 0 };
        for (occurrence, window) in covered.into_iter().enumerate() {
            if interruption_requested(stop) {
                return Err(());
            }
            let Some((start, end)) = window else { continue };
            let interval = self.intervals[occurrence];
            reason.premises.extend(present_premise(self.store, interval));
            let duration = i128::from(self.store.interval_duration(interval));
            if !reason.bound(self.store, interval, end - duration, true) || !reason.bound(self.store, interval, start, false) {
                return Ok(None);
            }
        }
        if let Some((interval, lower, conflict)) = target_bounds {
            reason.premises.extend(present_premise(self.store, interval));
            if !reason.bound(self.store, interval, lower, true) {
                return Ok(None);
            }
            if conflict && !reason.bound(self.store, interval, i128::from(self.store.interval_start_max(interval)), false) {
                return Ok(None);
            }
        }
        if interruption_requested(stop) {
            return Err(());
        }
        Ok(Some(reason))
    }
}

impl LiftedCumulativeReason {
    fn bound(&mut self, store: &Store, interval: IntervalId, bound: i128, lower: bool) -> bool {
        let original = if lower { store.interval_start_min(interval) } else { store.interval_start_max(interval) };
        // A certificate may weaken a current premise, never strengthen it.
        if (lower && bound > i128::from(original)) || (!lower && bound < i128::from(original)) {
            return false;
        }
        if (lower && original > i32::MIN && bound < i128::from(original)) || (!lower && original < i32::MAX && bound > i128::from(original))
        {
            self.lifted_bounds += 1;
        }
        if (lower && bound <= i128::from(i32::MIN)) || (!lower && bound >= i128::from(i32::MAX)) {
            self.omitted_tautologies += 1;
            return true;
        }
        let Ok(bound) = i32::try_from(bound) else { return false };
        let var = store.interval_start_var(interval);
        self.premises.push(if lower { Premise::Ge { var, bound } } else { Premise::Le { var, bound } });
        true
    }
}

/// Post interval precedence.
pub fn interval_precedence(solver: &mut Solver, before: IntervalId, after: IntervalId) -> PropId {
    solver.post(Box::new(IntervalPrecedence { before, after }))
}

/// Structured unary-resource no-overlap over fixed-duration intervals.
///
/// Present intervals sharing the resource may not overlap: for any two present
/// intervals one must end before the other starts. Weak pairwise (disjunctive)
/// propagation: when only one ordering of a present pair is still feasible it is
/// enforced on the start bounds; when neither ordering fits, a present pair is
/// inconsistent and an optional partner is forbidden. Absent intervals are
/// ignored. The propagator iterates to its own fixpoint, so it is idempotent.
#[derive(Clone)]
pub struct NoOverlap {
    intervals: Vec<IntervalId>,
    branch_orders: bool,
    /// Frozen semantics, also marks the supported mandatory positive-duration scope.
    edge_finding_tasks: Option<Arc<[(VarId, i64)]>>,
    /// `(a, b, order index)` for each unordered pair; the order index addresses
    /// the trailed order decision in the store.
    pairs: Vec<(usize, usize, usize)>,
    /// `pair_index[a][b]` (a < b) is the order index of that pair, for O(1) lookup
    /// of a decided order; `usize::MAX` off the upper triangle.
    pair_index: Vec<Vec<usize>>,
    /// Reused scratch for detectable precedences (no per-call allocation).
    present: Vec<usize>,
    prec: Vec<usize>,
    values: Vec<i32>,
}

#[inline]
fn interruption_polled(should_stop: Option<&dyn Fn() -> bool>, _work: usize) -> bool {
    should_stop.is_some_and(|stop| stop())
}

#[inline]
fn interruption_requested(should_stop: Option<&dyn Fn() -> bool>) -> bool {
    should_stop.is_some_and(|stop| stop())
}

/// Whether group interval `i` is decided to run before group interval `j`.
fn decided_before(store: &Store, pair_index: &[Vec<usize>], i: usize, j: usize) -> bool {
    let (lo, hi, want) = if i < j { (i, j, 1) } else { (j, i, 2) };
    let order = pair_index.get(lo).and_then(|row| row.get(hi)).copied().unwrap_or(usize::MAX);
    order != usize::MAX && store.disjunctive_order(order) == want
}

/// `[before(i, j)]` as a premise when that order is decided: cites the pair's
/// boolean order variable at its decided value. `None` if the order is not
/// decided in the `i`-before-`j` direction.
fn decided_before_premise(store: &Store, pair_index: &[Vec<usize>], i: usize, j: usize) -> Option<Premise> {
    let (lo, hi, want) = if i < j { (i, j, 1) } else { (j, i, 2) };
    let order = pair_index.get(lo).and_then(|row| row.get(hi)).copied().unwrap_or(usize::MAX);
    if order == usize::MAX || store.disjunctive_order(order) != want {
        return None;
    }
    // order() == 1 means the var is 1 (first-before-second); == 2 means var 0.
    Some(Premise::Eq { var: store.disjunctive_order_var(order), val: if want == 1 { 1 } else { 0 } })
}

/// Enforce `before` ends no later than `after` starts on the start bounds (both
/// present), citing the decided order `order_var = order_value` and the bounds it
/// rests on. Returns whether a bound changed.
fn enforce_before_because(
    store: &mut Store,
    before: IntervalId,
    after: IntervalId,
    order: Option<(VarId, i32)>,
    extra: &[Premise],
) -> Result<bool, Inconsistency> {
    let duration = store.interval_duration(before);
    let before_start = store.interval_start_var(before);
    let after_start = store.interval_start_var(after);
    let before_lb = store.interval_start_min(before);
    let after_ub = store.interval_start_max(after);

    // after.start >= before.start_min + duration(before); reason: the order, plus
    // before's lower bound and both presences.
    let mut why_after = Vec::with_capacity(4);
    if let Some((order_var, order_value)) = order {
        why_after.push(Premise::Eq { var: order_var, val: order_value });
    }
    why_after.extend_from_slice(extra);
    why_after.push(Premise::Ge { var: before_start, bound: before_lb });
    why_after.extend(present_premise(store, before));
    why_after.extend(present_premise(store, after));
    let mut changed = store.set_interval_start_min_because(after, before_lb.saturating_add(duration), why_after)?;

    // before.start <= after.start_max - duration(before); reason: the order, plus
    // after's upper bound and both presences.
    let mut why_before = Vec::with_capacity(4);
    if let Some((order_var, order_value)) = order {
        why_before.push(Premise::Eq { var: order_var, val: order_value });
    }
    why_before.extend_from_slice(extra);
    why_before.push(Premise::Le { var: after_start, bound: after_ub });
    why_before.extend(present_premise(store, before));
    why_before.extend(present_premise(store, after));
    changed |= store.set_interval_start_max_because(before, after_ub.saturating_sub(duration), why_before)?;
    Ok(changed)
}

/// Premises stating neither ordering of `i` and `j` fits: both end-before-start
/// directions are already violated by the current start bounds.
fn both_orders_infeasible(store: &Store, i: IntervalId, j: IntervalId) -> Vec<Premise> {
    vec![
        Premise::Ge { var: store.interval_start_var(i), bound: store.interval_start_min(i) },
        Premise::Le { var: store.interval_start_var(j), bound: store.interval_start_max(j) },
        Premise::Ge { var: store.interval_start_var(j), bound: store.interval_start_min(j) },
        Premise::Le { var: store.interval_start_var(i), bound: store.interval_start_max(i) },
    ]
}

/// Remove start values that have no support in either ordering of a mandatory
/// interval pair. This preserves the domain consistency of the old binary
/// decomposition while keeping one resource propagator for the whole scope.
fn remove_pairwise_unsupported_values(
    store: &mut Store,
    first: IntervalId,
    second: IntervalId,
    values: &mut Vec<i32>,
) -> Result<(), Inconsistency> {
    let first_var = store.interval_start_var(first);
    let second_var = store.interval_start_var(second);
    let first_duration = i128::from(store.interval_duration(first));
    let second_duration = i128::from(store.interval_duration(second));
    let second_min = store.min(second_var);
    let second_max = store.max(second_var);
    let unsupported_low = i128::from(second_max) - first_duration + 1;
    let unsupported_high = i128::from(second_min) + second_duration - 1;
    values.clear();
    values.extend(store.values(first_var).filter(|&value| unsupported_low <= i128::from(value) && i128::from(value) <= unsupported_high));
    let why = store.explaining().then(|| {
        let mut why = vec![Premise::Ge { var: second_var, bound: second_min }, Premise::Le { var: second_var, bound: second_max }];
        why.extend(present_premise(store, first));
        why.extend(present_premise(store, second));
        why
    });
    for value in values.drain(..) {
        store.remove_because(first_var, value, why.clone().unwrap_or_default())?;
    }

    let first_min = store.min(first_var);
    let first_max = store.max(first_var);
    let unsupported_low = i128::from(first_max) - second_duration + 1;
    let unsupported_high = i128::from(first_min) + first_duration - 1;
    values.clear();
    values.extend(store.values(second_var).filter(|&value| unsupported_low <= i128::from(value) && i128::from(value) <= unsupported_high));
    let why = store.explaining().then(|| {
        let mut why = vec![Premise::Ge { var: first_var, bound: first_min }, Premise::Le { var: first_var, bound: first_max }];
        why.extend(present_premise(store, first));
        why.extend(present_premise(store, second));
        why
    });
    for value in values.drain(..) {
        store.remove_because(second_var, value, why.clone().unwrap_or_default())?;
    }
    Ok(())
}

impl Propagator for NoOverlap {
    fn inference_constraint(&self) -> Option<crate::inference::InferenceConstraint> {
        self.edge_finding_tasks.as_ref().map(|tasks| crate::inference::InferenceConstraint::NoOverlap { tasks: tasks.to_vec() })
    }

    fn inference_description_work(&self) -> usize {
        self.edge_finding_tasks.as_ref().map_or(0, |tasks| tasks.len())
    }

    fn priority(&self) -> Priority {
        Priority::Expensive
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        let completed = self.register_core(store, me, None);
        debug_assert!(completed);
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        self.register_core(store, me, Some(should_stop))
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        self.propagate_core(store, None)
    }

    fn propagate_until(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        self.propagate_core(store, Some(should_stop))
    }
}

impl NoOverlap {
    fn register_core(&mut self, store: &mut Store, me: PropId, should_stop: Option<&dyn Fn() -> bool>) -> bool {
        if interruption_requested(should_stop) {
            return false;
        }
        for (index, &interval) in self.intervals.iter().enumerate() {
            if interruption_polled(should_stop, index) {
                return false;
            }
            store.subscribe_interval(interval, me, IntervalEvent::StartBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::EndBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::PresenceChange);
            store.subscribe(store.interval_start_var(interval), me, Event::DomainChange);
        }
        self.edge_finding_tasks = None;
        if !self.branch_orders {
            let mut tasks = Vec::with_capacity(self.intervals.len());
            let mut eligible = true;
            for &interval in &self.intervals {
                if interruption_requested(should_stop) {
                    return false;
                }
                if store.interval_presence_var(interval).is_some() || store.interval_duration(interval) <= 0 {
                    eligible = false;
                    break;
                }
                tasks.push((store.interval_start_var(interval), i64::from(store.interval_duration(interval))));
            }
            if eligible {
                self.edge_finding_tasks = Some(tasks.into());
            }
        }
        self.pairs.clear();
        let n = self.intervals.len();
        self.pair_index.clear();
        if self.branch_orders {
            self.pair_index.reserve(n);
            for row in 0..n {
                if interruption_polled(should_stop, row) {
                    return false;
                }
                self.pair_index.push(vec![usize::MAX; n]);
            }
        }
        let mut pair_count = 0usize;
        for a in 0..n {
            for b in (a + 1)..n {
                if interruption_polled(should_stop, pair_count) {
                    return false;
                }
                let order =
                    if self.branch_orders { store.register_disjunctive_pair(self.intervals[a], self.intervals[b], me) } else { usize::MAX };
                self.pairs.push((a, b, order));
                if self.branch_orders {
                    self.pair_index[a][b] = order;
                }
                pair_count = pair_count.saturating_add(1);
            }
        }
        !interruption_requested(should_stop)
    }

    fn propagate_core(&mut self, store: &mut Store, should_stop: Option<&dyn Fn() -> bool>) -> Result<(), Inconsistency> {
        store.record_unary_edge_finding(crate::constraints::unary_edge_finding::UnaryEdgeFindingStats {
            propagation_calls: 1,
            unsupported_calls: u64::from(store.unary_edge_finding_enabled() && self.edge_finding_tasks.is_none()),
            ..Default::default()
        });
        loop {
            if interruption_requested(should_stop) {
                return Ok(());
            }
            let mut changed = false;
            for (pair_count, &(a, b, order_index)) in self.pairs.iter().enumerate() {
                if interruption_polled(should_stop, pair_count) {
                    return Ok(());
                }
                let i = self.intervals[a];
                let j = self.intervals[b];
                let pi = store.interval_presence(i);
                let pj = store.interval_presence(j);
                if pi == IntervalPresence::Absent || pj == IntervalPresence::Absent {
                    continue;
                }
                // A zero-duration interval occupies no instant, so it never
                // overlaps and imposes no ordering on anyone.
                if store.interval_duration(i) == 0 || store.interval_duration(j) == 0 {
                    continue;
                }
                let both_present = pi == IntervalPresence::Present && pj == IntervalPresence::Present;
                if both_present && store.interval_duration(i) > 0 && store.interval_duration(j) > 0 {
                    remove_pairwise_unsupported_values(store, i, j, &mut self.values)?;
                }
                let order_var = (order_index != usize::MAX).then(|| store.disjunctive_order_var(order_index));
                let order = if order_index == usize::MAX { 0 } else { store.disjunctive_order(order_index) };
                match order {
                    // Order already decided (by the brancher, a deduction, or a
                    // learning-engine branch): durably enforce that precedence.
                    1 if both_present => changed |= enforce_before_because(store, i, j, order_var.map(|var| (var, 1)), &[])?,
                    2 if both_present => changed |= enforce_before_because(store, j, i, order_var.map(|var| (var, 0)), &[])?,
                    1 | 2 => {}
                    // Undecided: weak pairwise feasibility; deduce a forced order
                    // (detectable precedence) or forbid an unschedulable optional.
                    _ => {
                        let i_before_j = store.interval_end_min(i) <= store.interval_start_max(j);
                        let j_before_i = store.interval_end_min(j) <= store.interval_start_max(i);
                        match (i_before_j, j_before_i) {
                            (false, false) => match (pi, pj) {
                                // Both present yet neither order fits: conflict.
                                (IntervalPresence::Present, IntervalPresence::Present) => {
                                    let mut why = both_orders_infeasible(store, i, j);
                                    why.extend(present_premise(store, i));
                                    why.extend(present_premise(store, j));
                                    return Err(store.fail_because(why));
                                }
                                // A present partner leaves no room for the optional.
                                (IntervalPresence::Present, IntervalPresence::Optional) => {
                                    let mut why = both_orders_infeasible(store, i, j);
                                    why.extend(present_premise(store, i));
                                    changed |= store.forbid_interval_presence_because(j, why)?;
                                }
                                (IntervalPresence::Optional, IntervalPresence::Present) => {
                                    let mut why = both_orders_infeasible(store, i, j);
                                    why.extend(present_premise(store, j));
                                    changed |= store.forbid_interval_presence_because(i, why)?;
                                }
                                _ => {}
                            },
                            (true, false) if both_present => {
                                // j cannot precede i (its earliest end is past i's
                                // latest start), so i must run first.
                                let mut why = vec![
                                    Premise::Ge { var: store.interval_start_var(j), bound: store.interval_start_min(j) },
                                    Premise::Le { var: store.interval_start_var(i), bound: store.interval_start_max(i) },
                                ];
                                why.extend(present_premise(store, i));
                                why.extend(present_premise(store, j));
                                if order_index != usize::MAX {
                                    changed |= store.set_disjunctive_order_because(order_index, 1, why)?;
                                    changed |= enforce_before_because(store, i, j, order_var.map(|var| (var, 1)), &[])?;
                                } else {
                                    changed |= enforce_before_because(store, i, j, None, &why)?;
                                }
                            }
                            (false, true) if both_present => {
                                // i cannot precede j, so j must run first.
                                let mut why = vec![
                                    Premise::Ge { var: store.interval_start_var(i), bound: store.interval_start_min(i) },
                                    Premise::Le { var: store.interval_start_var(j), bound: store.interval_start_max(j) },
                                ];
                                why.extend(present_premise(store, i));
                                why.extend(present_premise(store, j));
                                if order_index != usize::MAX {
                                    changed |= store.set_disjunctive_order_because(order_index, 2, why)?;
                                    changed |= enforce_before_because(store, j, i, order_var.map(|var| (var, 0)), &[])?;
                                } else {
                                    changed |= enforce_before_because(store, j, i, None, &why)?;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Global detectable precedences: for each present interval, the set
            // that must run before it (decided, or because they cannot fit after
            // it) must all complete first, so its start is at least that set's
            // earliest completion time -- stronger than any single predecessor.
            let Some(detectable_changed) = self.detectable_precedences(store, should_stop)? else {
                return Ok(());
            };
            changed |= detectable_changed;

            if store.unary_edge_finding_enabled() && self.edge_finding_tasks.is_some() {
                let Some(edge_changed) = crate::constraints::unary_edge_finding::propagate(store, &self.intervals, should_stop)? else {
                    return Ok(());
                };
                changed |= edge_changed;
            }

            if !changed {
                return Ok(());
            }
        }
    }

    fn detectable_precedences(&mut self, store: &mut Store, should_stop: Option<&dyn Fn() -> bool>) -> Result<Option<bool>, Inconsistency> {
        self.present.clear();
        for k in 0..self.intervals.len() {
            if interruption_polled(should_stop, k) {
                return Ok(None);
            }
            if store.interval_presence(self.intervals[k]) == IntervalPresence::Present && store.interval_duration(self.intervals[k]) > 0 {
                self.present.push(k);
            }
        }
        let mut changed = false;
        for index in 0..self.present.len() {
            if interruption_polled(should_stop, index) {
                return Ok(None);
            }
            let qj = self.present[index];
            let j = self.intervals[qj];
            // Predecessors that must run before j: decided so, or unable to fit
            // after j (their latest start is before j's earliest end).
            self.prec.clear();
            for (candidate, &pi) in self.present.iter().enumerate() {
                if interruption_polled(should_stop, candidate) {
                    return Ok(None);
                }
                if pi == qj {
                    continue;
                }
                let i = self.intervals[pi];
                if decided_before(store, &self.pair_index, pi, qj) || store.interval_end_min(j) > store.interval_start_max(i) {
                    self.prec.push(pi);
                }
            }
            if self.prec.is_empty() {
                continue;
            }
            // Earliest completion of the whole predecessor set on the unary
            // resource: schedule them in est order, each after the previous ends.
            self.prec.sort_by_key(|&pi| store.interval_start_min(self.intervals[pi]));
            if interruption_requested(should_stop) {
                return Ok(None);
            }
            let mut ect = i32::MIN;
            for (predecessor, &pi) in self.prec.iter().enumerate() {
                if interruption_polled(should_stop, predecessor) {
                    return Ok(None);
                }
                let i = self.intervals[pi];
                ect = ect.max(store.interval_start_min(i)).saturating_add(store.interval_duration(i));
            }
            // j must start no earlier than the set's earliest completion. Cite j
            // present; for each predecessor, its presence, why it precedes j (a
            // decided order, or it cannot fit after j -- its latest start is before
            // j's earliest end), and its start lower bound, which feeds the ECT.
            let mut why: Vec<Premise> = present_premise(store, j).into_iter().collect();
            let mut uses_j_end_min = false;
            for (predecessor, &pi) in self.prec.iter().enumerate() {
                if interruption_polled(should_stop, predecessor) {
                    return Ok(None);
                }
                let i = self.intervals[pi];
                why.extend(present_premise(store, i));
                if let Some(order) = decided_before_premise(store, &self.pair_index, pi, qj) {
                    why.push(order);
                } else {
                    why.push(Premise::Le { var: store.interval_start_var(i), bound: store.interval_start_max(i) });
                    uses_j_end_min = true;
                }
                why.push(Premise::Ge { var: store.interval_start_var(i), bound: store.interval_start_min(i) });
            }
            if uses_j_end_min {
                why.push(Premise::Ge { var: store.interval_start_var(j), bound: store.interval_start_min(j) });
            }
            changed |= store.set_interval_start_min_because(j, ect, why)?;
        }
        Ok(Some(changed))
    }
}

/// Post a unary-resource no-overlap over the given intervals.
pub fn no_overlap(solver: &mut Solver, intervals: &[IntervalId]) -> PropId {
    solver.post(Box::new(NoOverlap {
        intervals: intervals.to_vec(),
        branch_orders: true,
        edge_finding_tasks: None,
        pairs: Vec::new(),
        pair_index: Vec::new(),
        present: Vec::new(),
        prec: Vec::new(),
        values: Vec::new(),
    }))
}

/// Interruptible owned-input variant used by scheduling physical builders.
/// `None` requires discarding `solver` because posting may have stopped after
/// partially registering the propagator.
pub(crate) fn no_overlap_until(solver: &mut Solver, intervals: Vec<IntervalId>, should_stop: &dyn Fn() -> bool) -> Option<PropId> {
    solver.post_until(
        Box::new(NoOverlap {
            intervals,
            branch_orders: true,
            edge_finding_tasks: None,
            pairs: Vec::new(),
            pair_index: Vec::new(),
            present: Vec::new(),
            prec: Vec::new(),
            values: Vec::new(),
        }),
        should_stop,
    )
}

/// Mandatory unary resource without auxiliary pair-order variables. Used when
/// the semantic model already exposes every start as a primary decision.
pub(crate) fn mandatory_no_overlap(solver: &mut Solver, intervals: &[IntervalId]) -> PropId {
    solver.post(Box::new(NoOverlap {
        intervals: intervals.to_vec(),
        branch_orders: false,
        edge_finding_tasks: None,
        pairs: Vec::new(),
        pair_index: Vec::new(),
        present: Vec::new(),
        prec: Vec::new(),
        values: Vec::new(),
    }))
}

pub(crate) fn mandatory_no_overlap_until(
    solver: &mut Solver,
    intervals: Vec<IntervalId>,
    should_stop: &dyn Fn() -> bool,
) -> Option<PropId> {
    solver.post_until(
        Box::new(NoOverlap {
            intervals,
            branch_orders: false,
            edge_finding_tasks: None,
            pairs: Vec::new(),
            pair_index: Vec::new(),
            present: Vec::new(),
            prec: Vec::new(),
            values: Vec::new(),
        }),
        should_stop,
    )
}

/// Makespan upper bound for branch-and-bound: every interval must end no later
/// than the shared `upper_bound`, i.e. `start + duration <= ub`. Lowering the
/// bound on each improving solution (the search writes `ub`) prunes any subtree
/// whose makespan cannot beat the incumbent. The bound is a monotone global
/// incumbent, not trailed.
#[derive(Clone)]
pub struct MakespanBound {
    intervals: Vec<IntervalId>,
    durations: Vec<i32>,
    upper_bound: Arc<AtomicI32>,
}

impl Propagator for MakespanBound {
    fn priority(&self) -> Priority {
        Priority::Cheap
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &interval in &self.intervals {
            store.subscribe_interval(interval, me, IntervalEvent::StartBoundChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        for (index, &interval) in self.intervals.iter().enumerate() {
            if interruption_polled(Some(should_stop), index) {
                return false;
            }
            store.subscribe_interval(interval, me, IntervalEvent::StartBoundChange);
        }
        !should_stop()
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let ub = self.upper_bound.load(Ordering::Relaxed);
        for (&interval, &duration) in self.intervals.iter().zip(&self.durations) {
            // Only present intervals contribute to the makespan; an optional
            // interval might be absent, so bounding its start would be unsound.
            if store.interval_presence(interval) != IntervalPresence::Present {
                continue;
            }
            // end = start + duration <= ub  =>  start <= ub - duration
            store.set_interval_start_max(interval, ub.saturating_sub(duration))?;
        }
        Ok(())
    }
}

/// Post a [`MakespanBound`]; the caller keeps a clone of `upper_bound` to lower
/// it as better solutions are found.
pub fn makespan_bound(solver: &mut Solver, intervals: &[IntervalId], durations: &[i32], upper_bound: Arc<AtomicI32>) -> PropId {
    solver.post(Box::new(MakespanBound { intervals: intervals.to_vec(), durations: durations.to_vec(), upper_bound }))
}

/// Interruptible owned-input variant used during physical schedule construction.
pub(crate) fn makespan_bound_until(
    solver: &mut Solver,
    intervals: Vec<IntervalId>,
    durations: Vec<i32>,
    upper_bound: Arc<AtomicI32>,
    should_stop: &dyn Fn() -> bool,
) -> Option<PropId> {
    solver.post_until(Box::new(MakespanBound { intervals, durations, upper_bound }), should_stop)
}

/// Structured cumulative resource with energetic reasoning and sparse
/// mandatory-part time-tabling.
///
/// Each present interval consumes `demand` units of a resource of `capacity`
/// while it runs; the total at any instant must not exceed the capacity. Build
/// the mandatory-part profile (each interval's compulsory
/// region `[start_max, end_min)`), fail on overload, and push each interval's
/// start past instants where it could not fit beside the others' mandatory
/// parts. Event-delimited segments keep memory independent of the numeric time
/// horizon. Mandatory tasks additionally receive energetic overload checks and
/// edge-finding lower bounds.
#[derive(Clone)]
pub struct Cumulative {
    intervals: Vec<IntervalId>,
    demands: Vec<i32>,
    capacity: i32,
    events: Vec<(i128, i128)>,
    profile: Vec<ProfileSegment>,
    present: Vec<usize>,
    tasks: Vec<FixedCumulativeTask>,
    energetic: EnergeticWorkspace,
    profile_cache: Option<IncrementalProfile>,
    root_registered: bool,
    inference_tasks: Option<Arc<[crate::inference::InferenceCumulativeTask]>>,
}

impl Cumulative {
    fn timetable_reason(
        &self,
        store: &Store,
        original: Vec<Premise>,
        certificate: TimetableCertificate,
        stats: &mut CumulativeStats,
        stop: Option<&dyn Fn() -> bool>,
    ) -> Option<Vec<Premise>> {
        let original_width = original.len();
        let reason = if store.lifted_cumulative_explanations_enabled() && self.inference_tasks.is_some() {
            let context = CumulativeReasonContext {
                store,
                intervals: &self.intervals,
                demands: &self.demands,
                capacity: self.capacity,
                profile: &self.profile,
            };
            match context.explain(certificate, stop) {
                Ok(Some(lifted)) => {
                    stats.lifted_bounds = stats.lifted_bounds.saturating_add(lifted.lifted_bounds);
                    stats.omitted_tautologies = stats.omitted_tautologies.saturating_add(lifted.omitted_tautologies);
                    lifted.premises
                }
                Ok(None) => original,
                Err(()) => {
                    stats.interrupted = stats.interrupted.saturating_add(1);
                    return None;
                }
            }
        } else {
            original
        };
        stats.reason_calls = stats.reason_calls.saturating_add(1);
        stats.original_premises = stats.original_premises.saturating_add(original_width as u64);
        stats.emitted_premises = stats.emitted_premises.saturating_add(reason.len() as u64);
        Some(reason)
    }

    fn register_core(&mut self, store: &mut Store, me: PropId, should_stop: Option<&dyn Fn() -> bool>) -> bool {
        self.root_registered = store.level() == 0;
        let mut supported = self.intervals.len() == self.demands.len() && self.capacity >= 0;
        let mut tasks = Vec::with_capacity(self.intervals.len());
        for (index, &interval) in self.intervals.iter().enumerate() {
            if interruption_polled(should_stop, index) {
                return false;
            }
            store.subscribe_interval(interval, me, IntervalEvent::StartBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::EndBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::PresenceChange);
            let duration = store.interval_duration(interval);
            let demand = self.demands.get(index).copied().unwrap_or(-1);
            supported &= duration >= 0 && demand >= 0;
            tasks.push(crate::inference::InferenceCumulativeTask {
                start: store.interval_start_var(interval),
                duration: i64::from(duration),
                demand: i64::from(demand),
                presence: store.interval_presence_var(interval),
            });
        }
        if interruption_requested(should_stop) {
            return false;
        }
        self.inference_tasks = supported.then(|| Arc::from(tasks));
        true
    }

    fn propagate_core(&mut self, store: &mut Store, should_stop: Option<&dyn Fn() -> bool>) -> Result<(), Inconsistency> {
        let mut stats = CumulativeStats { propagation_calls: 1, ..CumulativeStats::default() };
        let result = self.propagate_with_profile(store, should_stop, &mut stats);
        store.record_cumulative(stats);
        result
    }

    fn propagate_with_profile(
        &mut self,
        store: &mut Store,
        should_stop: Option<&dyn Fn() -> bool>,
        stats: &mut CumulativeStats,
    ) -> Result<(), Inconsistency> {
        let n = self.intervals.len();
        loop {
            if interruption_requested(should_stop) {
                stats.interrupted = stats.interrupted.saturating_add(1);
                return Ok(());
            }
            self.present.clear();
            self.tasks.clear();
            for idx in 0..n {
                if store.interval_presence(self.intervals[idx]) == IntervalPresence::Present
                    && store.interval_duration(self.intervals[idx]) > 0
                    && self.demands[idx] > 0
                {
                    self.present.push(idx);
                    self.tasks.push(FixedCumulativeTask::new(
                        store.interval_start_min(self.intervals[idx]),
                        store.interval_start_max(self.intervals[idx]),
                        i64::from(store.interval_duration(self.intervals[idx])),
                        i64::from(self.demands[idx]),
                    ));
                }
            }

            // Only present, positive-duration tasks enter the pure fixed-task
            // kernel. Presence and Store explanations stay in this adapter. No
            // bound or conflict is consumed from an interrupted analysis.
            let energetic = match should_stop {
                Some(stop) => self.energetic.analyse_until(&self.tasks, i128::from(self.capacity), stop),
                None => self.energetic.analyse(&self.tasks, i128::from(self.capacity)).map(|()| EnergeticAnalysis::Complete),
            };
            match energetic {
                Ok(EnergeticAnalysis::Complete) => {}
                Ok(EnergeticAnalysis::Interrupted) => {
                    stats.interrupted = stats.interrupted.saturating_add(1);
                    return Ok(());
                }
                Err(_) if interruption_requested(should_stop) => {
                    stats.interrupted = stats.interrupted.saturating_add(1);
                    return Ok(());
                }
                Err(_) => return Err(Inconsistency),
            }
            let mut changed = false;
            for (position, &task) in self.present.iter().enumerate() {
                if interruption_requested(should_stop) {
                    stats.interrupted = stats.interrupted.saturating_add(1);
                    return Ok(());
                }
                let lower_bound = self.energetic.lower_bounds()[position];
                if lower_bound > self.tasks[position].earliest_start() {
                    let lower_bound = i32::try_from(lower_bound).map_err(|_| Inconsistency)?;
                    changed |= store.set_interval_start_min(self.intervals[task], lower_bound)?;
                }
            }

            for (position, &task) in self.present.iter().enumerate() {
                if interruption_requested(should_stop) {
                    stats.interrupted = stats.interrupted.saturating_add(1);
                    return Ok(());
                }
                let interval = self.intervals[task];
                self.tasks[position] = FixedCumulativeTask::new(
                    store.interval_start_min(interval),
                    store.interval_start_max(interval),
                    i64::from(store.interval_duration(interval)),
                    i64::from(self.demands[task]),
                );
            }

            // Mandatory-part profile from the compulsory region of every
            // *present* interval. Optional (undecided) intervals do not yet
            // consume the resource, so they are excluded from the profile.
            let mut cached = false;
            if store.incremental_cumulative_enabled() {
                if self.root_registered && self.inference_tasks.is_some() {
                    if self.profile_cache.is_none() && store.level() == 0 {
                        self.profile_cache = IncrementalProfile::new(store, n);
                    }
                    if let Some(cache) = &mut self.profile_cache {
                        // Existing energetic snapshots are already refreshed.
                        // Map them back to stable original occurrence indices,
                        // including absent and undecided optional occurrences.
                        let mut position = 0;
                        let parts = (0..n).map(|occurrence| {
                            if self.present.get(position) == Some(&occurrence) {
                                let part = self.tasks[position].compulsory_part();
                                position += 1;
                                Some(part)
                            } else {
                                None
                            }
                        });
                        match cache.update(store, parts, stats, should_stop) {
                            ProfileUpdate::Ready => match cache.materialize(store, &mut self.profile, stats, should_stop) {
                                ProfileUpdate::Ready => cached = true,
                                ProfileUpdate::Interrupted => return Ok(()),
                                ProfileUpdate::Fallback => unreachable!("updated cumulative cache must be readable"),
                            },
                            ProfileUpdate::Interrupted => return Ok(()),
                            ProfileUpdate::Fallback => {}
                        }
                    } else {
                        stats.fallback_calls = stats.fallback_calls.saturating_add(1);
                    }
                } else {
                    stats.fallback_calls = stats.fallback_calls.saturating_add(1);
                }
            }
            if !cached {
                build_profile(self.tasks.iter().copied().map(FixedCumulativeTask::compulsory_part), &mut self.events, &mut self.profile);
                if store.cumulative_metrics_enabled() {
                    stats.profile_rebuilds = stats.profile_rebuilds.saturating_add(1);
                    stats.endpoint_updates = stats.endpoint_updates.saturating_add(self.events.len() as u64);
                    stats.peak_event_points = stats.peak_event_points.max(
                        self.events
                            .iter()
                            .map(|(time, _)| time)
                            .fold((None, 0u64), |(previous, count), time| (Some(time), count + u64::from(previous != Some(time))))
                            .1,
                    );
                    stats.segments_emitted = stats.segments_emitted.saturating_add(self.profile.len() as u64);
                }
            }
            for segment in &self.profile {
                if interruption_requested(should_stop) {
                    stats.interrupted = stats.interrupted.saturating_add(1);
                    return Ok(());
                }
                if segment.usage > i128::from(self.capacity) {
                    // Overload at instant `t`: cite every present interval whose
                    // compulsory region must cover `t` (it is present, its start is
                    // at most `start_max`, and at least `start_min`, which together
                    // force it to run at `t`). Their demands exceed the capacity.
                    let t = segment.start;
                    let mut why = Vec::new();
                    for &interval in &self.intervals {
                        if interruption_requested(should_stop) {
                            stats.interrupted = stats.interrupted.saturating_add(1);
                            return Ok(());
                        }
                        if store.interval_presence(interval) != IntervalPresence::Present {
                            continue;
                        }
                        let mandatory_start = i128::from(store.interval_start_max(interval));
                        let mandatory_end = i128::from(store.interval_start_min(interval)) + i128::from(store.interval_duration(interval));
                        if mandatory_start <= t && t < mandatory_end {
                            why.extend(present_premise(store, interval));
                            why.push(Premise::Ge { var: store.interval_start_var(interval), bound: store.interval_start_min(interval) });
                            why.push(Premise::Le { var: store.interval_start_var(interval), bound: store.interval_start_max(interval) });
                        }
                    }
                    if interruption_requested(should_stop) {
                        return Ok(());
                    }
                    let Some(why) = self.timetable_reason(store, why, TimetableCertificate::Overload { time: t }, stats, should_stop)
                    else {
                        return Ok(());
                    };
                    stats.timetable_conflicts = stats.timetable_conflicts.saturating_add(1);
                    return Err(store.fail_because(why));
                }
            }

            // Push each present interval's start past instants it cannot cover.
            for idx in 0..n {
                if interruption_requested(should_stop) {
                    stats.interrupted = stats.interrupted.saturating_add(1);
                    return Ok(());
                }
                let interval = self.intervals[idx];
                if store.interval_presence(interval) == IntervalPresence::Absent {
                    continue;
                }
                let demand = i128::from(self.demands[idx]);
                let duration = i128::from(store.interval_duration(interval));
                if demand == 0 || duration == 0 {
                    continue;
                }
                let smin = i128::from(store.interval_start_min(interval));
                let smax = i128::from(store.interval_start_max(interval));
                // Subtract the interval's own compulsory region only if it is
                // present (so already in the profile); an optional interval does
                // not contribute, so nothing to subtract.
                let in_profile = store.interval_presence(interval) == IntervalPresence::Present;
                let own_part = in_profile.then_some((smax, smin + duration, demand));

                let feasible = earliest_feasible_start(&self.profile, smin, smax, duration, demand, own_part, i128::from(self.capacity));
                if interruption_requested(should_stop) {
                    stats.interrupted = stats.interrupted.saturating_add(1);
                    return Ok(());
                }

                let present = store.interval_presence(interval) == IntervalPresence::Present;
                match feasible {
                    Some(start) if start > smin && present => {
                        // Positions `[smin, start)` overload, so a present
                        // interval must start no earlier than `start`. Cite
                        // its presence, its current lower bound, and the
                        // present intervals fixing the profile there.
                        let mut why = cumulative_profile_premises(store, &self.intervals, smin, start + duration, interval);
                        if interruption_requested(should_stop) {
                            stats.interrupted = stats.interrupted.saturating_add(1);
                            return Ok(());
                        }
                        why.extend(present_premise(store, interval));
                        why.push(Premise::Ge { var: store.interval_start_var(interval), bound: smin as i32 });
                        let start = i32::try_from(start).map_err(|_| Inconsistency)?;
                        let Some(why) = self.timetable_reason(
                            store,
                            why,
                            TimetableCertificate::LowerBound { occurrence: idx, bound: i128::from(start) },
                            stats,
                            should_stop,
                        ) else {
                            return Ok(());
                        };
                        let updated = store.set_interval_start_min_because(interval, start, why)?;
                        stats.timetable_updates = stats.timetable_updates.saturating_add(u64::from(updated));
                        changed |= updated;
                    }
                    // For an undecided optional interval, a start bound is
                    // conditional on presence. Keep its backing domain unchanged
                    // until presence is asserted.
                    Some(_) => {}
                    None => match store.interval_presence(interval) {
                        IntervalPresence::Present => {
                            // A present interval fits nowhere in its window.
                            let mut why = cumulative_profile_premises(store, &self.intervals, smin, smax + duration, interval);
                            if interruption_requested(should_stop) {
                                stats.interrupted = stats.interrupted.saturating_add(1);
                                return Ok(());
                            }
                            why.extend(present_premise(store, interval));
                            why.push(Premise::Ge { var: store.interval_start_var(interval), bound: smin as i32 });
                            why.push(Premise::Le { var: store.interval_start_var(interval), bound: smax as i32 });
                            let Some(why) =
                                self.timetable_reason(store, why, TimetableCertificate::NoFit { occurrence: idx }, stats, should_stop)
                            else {
                                return Ok(());
                            };
                            stats.timetable_conflicts = stats.timetable_conflicts.saturating_add(1);
                            return Err(store.fail_because(why));
                        }
                        IntervalPresence::Optional => {
                            if interruption_requested(should_stop) {
                                stats.interrupted = stats.interrupted.saturating_add(1);
                                return Ok(());
                            }
                            let updated = store.forbid_interval_presence(interval)?;
                            stats.timetable_updates = stats.timetable_updates.saturating_add(u64::from(updated));
                            changed |= updated;
                        }
                        IntervalPresence::Absent => {}
                    },
                }
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

impl Propagator for Cumulative {
    fn inference_constraint(&self) -> Option<crate::inference::InferenceConstraint> {
        self.inference_tasks
            .as_ref()
            .map(|tasks| crate::inference::InferenceConstraint::Cumulative { tasks: tasks.to_vec(), capacity: i64::from(self.capacity) })
    }

    fn inference_description_work(&self) -> usize {
        self.inference_tasks.as_ref().map_or(0, |tasks| tasks.len())
    }

    fn priority(&self) -> Priority {
        Priority::Expensive
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        let completed = self.register_core(store, me, None);
        debug_assert!(completed);
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        self.register_core(store, me, Some(should_stop))
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        self.propagate_core(store, None)
    }

    fn propagate_until(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        self.propagate_core(store, Some(should_stop))
    }
}

/// Post a cumulative resource: `intervals[k]` uses `demands[k]` units
/// of a resource of `capacity` while running.
pub fn cumulative(solver: &mut Solver, intervals: &[IntervalId], demands: &[i32], capacity: i32) -> PropId {
    let n = intervals.len();
    solver.post(Box::new(Cumulative {
        intervals: intervals.to_vec(),
        demands: demands.to_vec(),
        capacity,
        events: Vec::new(),
        profile: Vec::new(),
        present: Vec::new(),
        tasks: Vec::with_capacity(n),
        energetic: EnergeticWorkspace::default(),
        profile_cache: None,
        root_registered: false,
        inference_tasks: None,
    }))
}

/// Interruptible owned-input variant used during physical schedule construction.
pub(crate) fn cumulative_until(
    solver: &mut Solver,
    intervals: Vec<IntervalId>,
    demands: Vec<i32>,
    capacity: i32,
    should_stop: &dyn Fn() -> bool,
) -> Option<PropId> {
    let n = intervals.len();
    solver.post_until(
        Box::new(Cumulative {
            intervals,
            demands,
            capacity,
            events: Vec::new(),
            profile: Vec::new(),
            present: Vec::new(),
            tasks: Vec::with_capacity(n),
            energetic: EnergeticWorkspace::default(),
            profile_cache: None,
            root_registered: false,
            inference_tasks: None,
        }),
        should_stop,
    )
}

/// Exactly one of a set of optional intervals is present (an `alternative`):
/// the building block for machine/mode choice (flexible job shop). Each operation
/// becomes one optional fixed-duration interval per eligible machine; this keeps
/// exactly one present. Reuses the optional-aware `no_overlap` (per machine over
/// the mode intervals), `interval_precedence` (posted over every mode pair of two
/// ops), and `makespan_bound` (over the present mode), with no master interval and
/// no variable duration.
#[derive(Clone)]
pub struct ExactlyOneMode {
    modes: Vec<IntervalId>,
}

impl Propagator for ExactlyOneMode {
    fn priority(&self) -> Priority {
        Priority::Cheap
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &mode in &self.modes {
            store.subscribe_interval(mode, me, IntervalEvent::PresenceChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        for (index, &mode) in self.modes.iter().enumerate() {
            if interruption_polled(Some(should_stop), index) {
                return false;
            }
            store.subscribe_interval(mode, me, IntervalEvent::PresenceChange);
        }
        !should_stop()
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        loop {
            let mut present = 0;
            let mut non_absent = 0;
            let mut last_non_absent = None;
            for (k, &mode) in self.modes.iter().enumerate() {
                match store.interval_presence(mode) {
                    IntervalPresence::Present => {
                        present += 1;
                        non_absent += 1;
                        last_non_absent = Some(k);
                    }
                    IntervalPresence::Optional => {
                        non_absent += 1;
                        last_non_absent = Some(k);
                    }
                    IntervalPresence::Absent => {}
                }
            }
            if present > 1 {
                // Two modes present at once: cite each present mode's presence.
                let why = self
                    .modes
                    .iter()
                    .filter(|&&m| store.interval_presence(m) == IntervalPresence::Present)
                    .filter_map(|&m| store.interval_presence_var(m).map(|var| Premise::Eq { var, val: 1 }))
                    .collect();
                return Err(store.fail_because(why));
            }
            if non_absent == 0 {
                // Every mode forbidden: cite each mode's absence.
                let why =
                    self.modes.iter().filter_map(|&m| store.interval_presence_var(m).map(|var| Premise::Eq { var, val: 0 })).collect();
                return Err(store.fail_because(why));
            }
            let mut changed = false;
            if present == 1 {
                // A mode is chosen: forbid every other (still optional) mode,
                // because that mode is present.
                let chosen = self.modes.iter().copied().find(|&m| store.interval_presence(m) == IntervalPresence::Present).unwrap();
                let chosen_present: Vec<Premise> =
                    store.interval_presence_var(chosen).map(|var| Premise::Eq { var, val: 1 }).into_iter().collect();
                for &mode in &self.modes {
                    if store.interval_presence(mode) == IntervalPresence::Optional {
                        changed |= store.forbid_interval_presence_because(mode, chosen_present.clone())?;
                    }
                }
            } else if non_absent == 1 {
                // Only one candidate remains: it must be present, because every
                // other mode is absent.
                let last = self.modes[last_non_absent.unwrap()];
                let why = self
                    .modes
                    .iter()
                    .filter(|&&m| m != last)
                    .filter_map(|&m| store.interval_presence_var(m).map(|var| Premise::Eq { var, val: 0 }))
                    .collect();
                changed |= store.require_interval_presence_because(last, why)?;
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

/// Post an `alternative`: exactly one of `modes` (optional intervals) is present.
pub fn exactly_one_mode(solver: &mut Solver, modes: &[IntervalId]) -> PropId {
    solver.post(Box::new(ExactlyOneMode { modes: modes.to_vec() }))
}

/// Interruptible owned-input variant used during physical schedule construction.
pub(crate) fn exactly_one_mode_until(solver: &mut Solver, modes: Vec<IntervalId>, should_stop: &dyn Fn() -> bool) -> Option<PropId> {
    solver.post_until(Box::new(ExactlyOneMode { modes }), should_stop)
}

/// Raise a plain variable's lower bound, explained, reporting whether it moved.
fn set_var_min_because(store: &mut Store, var: VarId, min: i32, why: Vec<Premise>) -> Result<bool, Inconsistency> {
    let before = store.min(var);
    store.remove_below_because(var, min, why)?;
    Ok(store.min(var) != before)
}

/// Lower a plain variable's upper bound, explained, reporting whether it moved.
fn set_var_max_because(store: &mut Store, var: VarId, max: i32, why: Vec<Premise>) -> Result<bool, Inconsistency> {
    let before = store.max(var);
    store.remove_above_because(var, max, why)?;
    Ok(store.max(var) != before)
}

/// Bounds channel for an `alternative` master: one shared start `S`, and one optional
/// per-member interval (own start `s_m`, presence `p_m`) per `(machine, duration)`
/// mode, with the co-posted [`ExactlyOneMode`] keeping exactly one present. `S`
/// is the chosen mode's start. A mode is *capable* while `p_m` is not fixed to
/// `0` (it can still host `S`).
///
/// The naive encoding (`p_m == 1 ⇒ s_m == S` as generic implications) prunes
/// nothing until `p_m` is decided, so it never confines `S` and never rules a
/// mode out early. This does both on bounds:
/// - (a) `S` lies in the union of the capable modes' windows (it equals the
///   chosen capable mode's start);
/// - (b) a capable mode whose window is disjoint from `S`'s can no longer host
///   `S`, so it is forced absent;
/// - (c) the chosen (`p_m == 1`) mode's start equals `S`, propagated both ways.
///
/// An unchosen mode's start is left alone beyond (b): while `p_m` is undecided
/// that start is meaningless, so tying it to `S` would be the very over-
/// constraint the guarded channel exists to avoid.
#[derive(Clone)]
pub struct AlternativeChannel {
    shared_start: VarId,
    modes: Vec<IntervalId>,
}

impl AlternativeChannel {
    /// Reason for rule (a)'s union bound on `S`: every mode that is not fixed-
    /// absent must be covered, because `S` could still equal any of their starts.
    /// A capable mode cites the bound feeding the union (`Ge`/`Le` on its start);
    /// a fixed-absent mode cites `p_m == 0`, which is what lets us drop it.
    fn union_reason(&self, store: &Store, lower: bool) -> Vec<Premise> {
        let mut why = Vec::with_capacity(self.modes.len());
        for &m in &self.modes {
            if store.interval_presence(m) == IntervalPresence::Absent {
                if let Some(var) = store.interval_presence_var(m) {
                    why.push(Premise::Eq { var, val: 0 });
                }
            } else {
                let start = store.interval_start_var(m);
                why.push(if lower {
                    Premise::Ge { var: start, bound: store.interval_start_min(m) }
                } else {
                    Premise::Le { var: start, bound: store.interval_start_max(m) }
                });
            }
        }
        why
    }
}

impl Propagator for AlternativeChannel {
    fn priority(&self) -> Priority {
        Priority::Cheap
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.shared_start, me, Event::BoundChange);
        for &mode in &self.modes {
            store.subscribe_interval(mode, me, IntervalEvent::StartBoundChange);
            store.subscribe_interval(mode, me, IntervalEvent::PresenceChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        if should_stop() {
            return false;
        }
        store.subscribe(self.shared_start, me, Event::BoundChange);
        for &mode in &self.modes {
            if should_stop() {
                return false;
            }
            store.subscribe_interval(mode, me, IntervalEvent::StartBoundChange);
            store.subscribe_interval(mode, me, IntervalEvent::PresenceChange);
        }
        !should_stop()
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        // This propagator mutates its own scope (forcing `p_m := 0`, shifting `S`
        // and `s_m`), which never re-wakes it, so it loops to its own fixpoint.
        // Every rule only tightens domains, and each reason is built from the
        // reads current at its mutating call, so a bound another rule moved
        // earlier in the pass only means one more iteration, never an unsound push.
        loop {
            let mut changed = false;

            // Rule (a): confine `S` to the union of the capable modes' windows.
            let mut union_lo = i32::MAX;
            let mut union_hi = i32::MIN;
            for &m in &self.modes {
                if store.interval_presence(m) != IntervalPresence::Absent {
                    union_lo = union_lo.min(store.interval_start_min(m));
                    union_hi = union_hi.max(store.interval_start_max(m));
                }
            }
            if union_lo > union_hi {
                // No capable mode remains; `ExactlyOneMode` owns that conflict.
                return Ok(());
            }
            if union_lo > store.min(self.shared_start) {
                let why = if store.explaining() { self.union_reason(store, true) } else { Vec::new() };
                changed |= set_var_min_because(store, self.shared_start, union_lo, why)?;
            }
            if union_hi < store.max(self.shared_start) {
                let why = if store.explaining() { self.union_reason(store, false) } else { Vec::new() };
                changed |= set_var_max_because(store, self.shared_start, union_hi, why)?;
            }

            let s_lo = store.min(self.shared_start);
            let s_hi = store.max(self.shared_start);

            // Rule (b): a capable mode disjoint from `S`'s window cannot host the
            // chosen start, so it becomes absent (`p_m == 1 ⇒ s_m == S` is then
            // unsatisfiable). Reason: the two bounds that make the windows disjoint.
            for &m in &self.modes {
                if store.interval_presence(m) == IntervalPresence::Absent {
                    continue;
                }
                let start = store.interval_start_var(m);
                let m_lo = store.interval_start_min(m);
                let m_hi = store.interval_start_max(m);
                if m_hi < s_lo {
                    let why = if store.explaining() {
                        vec![Premise::Le { var: start, bound: m_hi }, Premise::Ge { var: self.shared_start, bound: s_lo }]
                    } else {
                        Vec::new()
                    };
                    changed |= store.forbid_interval_presence_because(m, why)?;
                } else if m_lo > s_hi {
                    let why = if store.explaining() {
                        vec![Premise::Ge { var: start, bound: m_lo }, Premise::Le { var: self.shared_start, bound: s_hi }]
                    } else {
                        Vec::new()
                    };
                    changed |= store.forbid_interval_presence_because(m, why)?;
                }
            }

            // Rule (c): the chosen mode's start equals `S`; propagate both ways.
            for &m in &self.modes {
                if store.interval_presence(m) != IntervalPresence::Present {
                    continue;
                }
                let start = store.interval_start_var(m);
                // `S >= s_m.lb` and `S <= s_m.ub`.
                let m_lo = store.interval_start_min(m);
                if m_lo > store.min(self.shared_start) {
                    let why = if store.explaining() {
                        let mut w = vec![Premise::Ge { var: start, bound: m_lo }];
                        w.extend(present_premise(store, m));
                        w
                    } else {
                        Vec::new()
                    };
                    changed |= set_var_min_because(store, self.shared_start, m_lo, why)?;
                }
                let m_hi = store.interval_start_max(m);
                if m_hi < store.max(self.shared_start) {
                    let why = if store.explaining() {
                        let mut w = vec![Premise::Le { var: start, bound: m_hi }];
                        w.extend(present_premise(store, m));
                        w
                    } else {
                        Vec::new()
                    };
                    changed |= set_var_max_because(store, self.shared_start, m_hi, why)?;
                }
                // `s_m >= S.lb` and `s_m <= S.ub` (re-read `S` after the pushes above).
                let s_lo = store.min(self.shared_start);
                if s_lo > store.interval_start_min(m) {
                    let why = if store.explaining() {
                        let mut w = vec![Premise::Ge { var: self.shared_start, bound: s_lo }];
                        w.extend(present_premise(store, m));
                        w
                    } else {
                        Vec::new()
                    };
                    changed |= store.set_interval_start_min_because(m, s_lo, why)?;
                }
                let s_hi = store.max(self.shared_start);
                if s_hi < store.interval_start_max(m) {
                    let why = if store.explaining() {
                        let mut w = vec![Premise::Le { var: self.shared_start, bound: s_hi }];
                        w.extend(present_premise(store, m));
                        w
                    } else {
                        Vec::new()
                    };
                    changed |= store.set_interval_start_max_because(m, s_hi, why)?;
                }
            }

            if !changed {
                return Ok(());
            }
        }
    }
}

/// Post the bounds channel of a moded interval: `shared_start` is the chosen
/// mode's start, over the per-mode optional `modes`. Co-post [`exactly_one_mode`]
/// so exactly one mode is present.
pub fn alternative_channel(solver: &mut Solver, shared_start: VarId, modes: &[IntervalId]) -> PropId {
    solver.post(Box::new(AlternativeChannel { shared_start, modes: modes.to_vec() }))
}

pub(crate) fn alternative_channel_until(
    solver: &mut Solver,
    shared_start: VarId,
    modes: Vec<IntervalId>,
    should_stop: &dyn Fn() -> bool,
) -> Option<PropId> {
    solver.post_until(Box::new(AlternativeChannel { shared_start, modes }), should_stop)
}
