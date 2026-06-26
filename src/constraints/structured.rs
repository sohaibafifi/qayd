//! Propagators over structured list and interval domains.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use crate::ids::{IntervalId, ListId, PropId};
use crate::propagator::{Inconsistency, Propagator};
use crate::store::{Solver, Store};
use crate::structured::{IntervalEvent, IntervalPresence, ListEvent};

/// Exact partition over structured list membership.
///
/// Every item must be required by exactly one list. This first structured-list
/// propagator reasons only about membership, not order, arcs, or positions.
#[derive(Clone)]
pub struct Partition {
    lists: Vec<ListId>,
    items: Vec<i32>,
}

impl Propagator for Partition {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &list in &self.lists {
            store.subscribe_list(list, me, ListEvent::PossibleChange);
            store.subscribe_list(list, me, ListEvent::RequiredChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        for &item in &self.items {
            let mut owner = None;
            let mut possible_count = 0usize;
            let mut last_possible = None;

            for &list in &self.lists {
                if store.list_required(list, item) && owner.replace(list).is_some() {
                    return Err(Inconsistency);
                }
                if store.list_possible(list, item) {
                    possible_count += 1;
                    last_possible = Some(list);
                }
            }

            if let Some(owner) = owner {
                for &list in &self.lists {
                    if list != owner {
                        store.forbid_list_item(list, item)?;
                    }
                }
            } else if let Some(list) = last_possible {
                if possible_count == 1 {
                    store.require_list_item(list, item)?;
                }
            } else {
                return Err(Inconsistency);
            }
        }
        Ok(())
    }
}

/// Post a structured list partition.
pub fn partition(solver: &mut Solver, lists: &[ListId], items: &[i32]) -> PropId {
    solver.post(Box::new(Partition { lists: lists.to_vec(), items: items.to_vec() }))
}

/// Keep two assigned items on the same list.
#[derive(Clone)]
pub struct SameList {
    lists: Vec<ListId>,
    a: i32,
    b: i32,
}

impl Propagator for SameList {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &list in &self.lists {
            store.subscribe_list(list, me, ListEvent::PossibleChange);
            store.subscribe_list(list, me, ListEvent::RequiredChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if self.a == self.b {
            return Ok(());
        }

        loop {
            let mut changed = false;
            let owner_a = required_owner(store, &self.lists, self.a)?;
            let owner_b = required_owner(store, &self.lists, self.b)?;

            if let (Some(a), Some(b)) = (owner_a, owner_b) {
                if a != b {
                    return Err(Inconsistency);
                }
            }

            if let Some(i) = owner_a {
                changed |= store.require_list_item(self.lists[i], self.b)?;
            }
            if let Some(i) = owner_b {
                changed |= store.require_list_item(self.lists[i], self.a)?;
            }

            let mut has_common_owner = false;
            for &list in &self.lists {
                let a_possible = store.list_possible(list, self.a);
                let b_possible = store.list_possible(list, self.b);
                has_common_owner |= a_possible && b_possible;

                if !a_possible {
                    changed |= store.forbid_list_item(list, self.b)?;
                }
                if !b_possible {
                    changed |= store.forbid_list_item(list, self.a)?;
                }
            }

            if !has_common_owner {
                return Err(Inconsistency);
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

/// Post a same-list constraint over assigned items.
pub fn same_list(solver: &mut Solver, lists: &[ListId], a: i32, b: i32) -> PropId {
    solver.post(Box::new(SameList { lists: lists.to_vec(), a, b }))
}

/// Keep the assigned list index of `before` no greater than that of `after`.
#[derive(Clone)]
pub struct ItemPrecedence {
    lists: Vec<ListId>,
    before: i32,
    after: i32,
}

impl Propagator for ItemPrecedence {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &list in &self.lists {
            store.subscribe_list(list, me, ListEvent::PossibleChange);
            store.subscribe_list(list, me, ListEvent::RequiredChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        loop {
            let mut changed = false;
            let before_owner = required_owner(store, &self.lists, self.before)?;
            let after_owner = required_owner(store, &self.lists, self.after)?;

            if let (Some(before), Some(after)) = (before_owner, after_owner) {
                if before > after {
                    return Err(Inconsistency);
                }
            }

            if let Some(before) = before_owner {
                for i in 0..before {
                    changed |= store.forbid_list_item(self.lists[i], self.after)?;
                }
            }
            if let Some(after) = after_owner {
                for i in (after + 1)..self.lists.len() {
                    changed |= store.forbid_list_item(self.lists[i], self.before)?;
                }
            }

            for i in 0..self.lists.len() {
                if store.list_possible(self.lists[i], self.before) && !self.has_after_support(store, i) {
                    changed |= store.forbid_list_item(self.lists[i], self.before)?;
                }
                if store.list_possible(self.lists[i], self.after) && !self.has_before_support(store, i) {
                    changed |= store.forbid_list_item(self.lists[i], self.after)?;
                }
            }

            if !has_possible_owner(store, &self.lists, self.before) || !has_possible_owner(store, &self.lists, self.after) {
                return Err(Inconsistency);
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

impl ItemPrecedence {
    fn has_after_support(&self, store: &Store, before_index: usize) -> bool {
        (before_index..self.lists.len()).any(|i| store.list_possible(self.lists[i], self.after))
    }

    fn has_before_support(&self, store: &Store, after_index: usize) -> bool {
        (0..=after_index).any(|i| store.list_possible(self.lists[i], self.before))
    }
}

/// Post item precedence over assigned list indices.
pub fn item_precedence(solver: &mut Solver, lists: &[ListId], before: i32, after: i32) -> PropId {
    solver.post(Box::new(ItemPrecedence { lists: lists.to_vec(), before, after }))
}

/// Alias for item precedence, matching list-model terminology.
pub fn list_precedence(solver: &mut Solver, lists: &[ListId], before: i32, after: i32) -> PropId {
    item_precedence(solver, lists, before, after)
}

/// Bounds the number of required items in a list.
#[derive(Clone)]
pub struct ListLength {
    list: ListId,
    min: usize,
    max: usize,
}

impl Propagator for ListLength {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe_list(self.list, me, ListEvent::PossibleChange);
        store.subscribe_list(self.list, me, ListEvent::RequiredChange);
        store.subscribe_list(self.list, me, ListEvent::LengthChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if self.min > self.max {
            return Err(Inconsistency);
        }
        store.set_list_len_min(self.list, self.min)?;
        store.set_list_len_max(self.list, self.max)?;
        Ok(())
    }
}

/// Post list length bounds.
pub fn list_len(solver: &mut Solver, list: ListId, min: usize, max: usize) -> PropId {
    solver.post(Box::new(ListLength { list, min, max }))
}

/// Post list length bounds.
pub fn list_length(solver: &mut Solver, list: ListId, min: usize, max: usize) -> PropId {
    list_len(solver, list, min, max)
}

/// Bounds a weighted item sum over one list.
#[derive(Clone)]
pub struct ListItemSum {
    list: ListId,
    weights: Vec<(i32, i64)>,
    min: i64,
    max: i64,
}

impl Propagator for ListItemSum {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe_list(self.list, me, ListEvent::PossibleChange);
        store.subscribe_list(self.list, me, ListEvent::RequiredChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if self.min > self.max {
            return Err(Inconsistency);
        }

        loop {
            let (required, negative_optional, positive_optional) = self.sum_parts(store);
            let lower = required.saturating_add(negative_optional);
            let upper = required.saturating_add(positive_optional);
            if lower > self.max || upper < self.min {
                return Err(Inconsistency);
            }

            let mut changed = false;
            for &(item, weight) in &self.weights {
                if !store.list_possible(self.list, item) || store.list_required(self.list, item) {
                    continue;
                }

                let lower_with_item = required.saturating_add(negative_optional).saturating_add(weight.max(0));
                if lower_with_item > self.max {
                    changed |= store.forbid_list_item(self.list, item)?;
                    continue;
                }

                let upper_without_item = required.saturating_add(positive_optional).saturating_sub(if weight > 0 { weight } else { 0 });
                if upper_without_item < self.min {
                    changed |= store.require_list_item(self.list, item)?;
                }
            }

            if !changed {
                return Ok(());
            }
        }
    }
}

impl ListItemSum {
    fn sum_parts(&self, store: &Store) -> (i64, i64, i64) {
        let mut required = 0i64;
        let mut negative_optional = 0i64;
        let mut positive_optional = 0i64;
        for &(item, weight) in &self.weights {
            if !store.list_possible(self.list, item) {
                continue;
            }
            if store.list_required(self.list, item) {
                required = required.saturating_add(weight);
            } else if weight < 0 {
                negative_optional = negative_optional.saturating_add(weight);
            } else {
                positive_optional = positive_optional.saturating_add(weight);
            }
        }
        (required, negative_optional, positive_optional)
    }
}

/// Post weighted item-sum bounds.
pub fn list_item_sum(solver: &mut Solver, list: ListId, weights: Vec<(i32, i64)>, min: i64, max: i64) -> PropId {
    solver.post(Box::new(ListItemSum { list, weights, min, max }))
}

/// Post a capacity-style upper bound over weighted list items.
pub fn list_item_sum_le(solver: &mut Solver, list: ListId, weights: Vec<(i32, i64)>, max: i64) -> PropId {
    list_item_sum(solver, list, weights, i64::MIN / 4, max)
}

fn required_owner(store: &Store, lists: &[ListId], item: i32) -> Result<Option<usize>, Inconsistency> {
    let mut owner = None;
    for (i, &list) in lists.iter().enumerate() {
        if store.list_required(list, item) && owner.replace(i).is_some() {
            return Err(Inconsistency);
        }
    }
    Ok(owner)
}

fn has_possible_owner(store: &Store, lists: &[ListId], item: i32) -> bool {
    lists.iter().any(|&list| store.list_possible(list, item))
}

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

        let feasible = store.interval_end_min(self.before) <= store.interval_start_max(self.after);
        if !feasible {
            match (before_presence, after_presence) {
                (IntervalPresence::Present, IntervalPresence::Present) => return Err(Inconsistency),
                (IntervalPresence::Present, IntervalPresence::Optional) => {
                    store.forbid_interval_presence(self.after)?;
                    return Ok(());
                }
                (IntervalPresence::Optional, IntervalPresence::Present) => {
                    store.forbid_interval_presence(self.before)?;
                    return Ok(());
                }
                _ => return Ok(()),
            }
        }

        if before_presence == IntervalPresence::Present && after_presence == IntervalPresence::Present {
            let before_duration = store.interval_duration(self.before);
            let before_max = store.interval_start_max(self.before);
            let after_min = store.interval_start_min(self.after);
            let after_max = store.interval_start_max(self.after);

            store.set_interval_start_max(self.before, before_max.min(after_max.saturating_sub(before_duration)))?;
            store
                .set_interval_start_min(self.after, after_min.max(store.interval_start_min(self.before).saturating_add(before_duration)))?;
        }

        Ok(())
    }
}

/// Post structured interval precedence.
pub fn interval_precedence(solver: &mut Solver, before: IntervalId, after: IntervalId) -> PropId {
    solver.post(Box::new(IntervalPrecedence { before, after }))
}

/// Post structured interval precedence.
pub fn precedence(solver: &mut Solver, before: IntervalId, after: IntervalId) -> PropId {
    interval_precedence(solver, before, after)
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
}

impl Propagator for NoOverlap {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &interval in &self.intervals {
            store.subscribe_interval(interval, me, IntervalEvent::StartBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::EndBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::PresenceChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.intervals.len();
        loop {
            let mut changed = false;
            for a in 0..n {
                let i = self.intervals[a];
                if store.interval_presence(i) == IntervalPresence::Absent {
                    continue;
                }
                for b in (a + 1)..n {
                    let j = self.intervals[b];
                    let pi = store.interval_presence(i);
                    if pi == IntervalPresence::Absent {
                        break; // `i` was just forbidden; nothing more to pair it with
                    }
                    let pj = store.interval_presence(j);
                    if pj == IntervalPresence::Absent {
                        continue;
                    }
                    // An ordering is feasible iff the earlier interval can end no
                    // later than the latter can start.
                    let i_before_j = store.interval_end_min(i) <= store.interval_start_max(j);
                    let j_before_i = store.interval_end_min(j) <= store.interval_start_max(i);
                    let both_present = pi == IntervalPresence::Present && pj == IntervalPresence::Present;
                    match (i_before_j, j_before_i) {
                        (false, false) => match (pi, pj) {
                            (IntervalPresence::Present, IntervalPresence::Present) => return Err(Inconsistency),
                            (IntervalPresence::Present, IntervalPresence::Optional) => changed |= store.forbid_interval_presence(j)?,
                            (IntervalPresence::Optional, IntervalPresence::Present) => changed |= store.forbid_interval_presence(i)?,
                            _ => {}
                        },
                        (true, false) if both_present => {
                            // Only i-before-j fits: i ends before j starts.
                            let di = store.interval_duration(i);
                            changed |= store.set_interval_start_min(j, store.interval_end_min(i))?;
                            changed |= store.set_interval_start_max(i, store.interval_start_max(j).saturating_sub(di))?;
                        }
                        (false, true) if both_present => {
                            // Only j-before-i fits.
                            let dj = store.interval_duration(j);
                            changed |= store.set_interval_start_min(i, store.interval_end_min(j))?;
                            changed |= store.set_interval_start_max(j, store.interval_start_max(i).saturating_sub(dj))?;
                        }
                        _ => {}
                    }
                }
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

/// Post a structured unary-resource no-overlap over the given intervals.
pub fn no_overlap(solver: &mut Solver, intervals: &[IntervalId]) -> PropId {
    solver.post(Box::new(NoOverlap { intervals: intervals.to_vec() }))
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
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &interval in &self.intervals {
            store.subscribe_interval(interval, me, IntervalEvent::StartBoundChange);
        }
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

/// Structured cumulative resource by time-tabling.
///
/// Each present interval consumes `demand` units of a resource of `capacity`
/// while it runs; the total at any instant must not exceed the capacity. Weak
/// time-tabling: build the mandatory-part profile (each interval's compulsory
/// region `[start_max, end_min)`), fail on overload, and push each interval's
/// start past instants where it could not fit beside the others' mandatory
/// parts. The profile is rebuilt each pass and the propagator iterates to a
/// fixpoint, so it is idempotent. Pushing only ever uses a lower bound on usage,
/// so it never over-prunes.
#[derive(Clone)]
pub struct Cumulative {
    intervals: Vec<IntervalId>,
    demands: Vec<i32>,
    capacity: i32,
    profile: Vec<i32>,
}

impl Propagator for Cumulative {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &interval in &self.intervals {
            store.subscribe_interval(interval, me, IntervalEvent::StartBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::EndBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::PresenceChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.intervals.len();
        loop {
            // Time window covering every present interval.
            let mut hmin = i32::MAX;
            let mut hmax = i32::MIN;
            for &interval in &self.intervals {
                if store.interval_presence(interval) == IntervalPresence::Absent {
                    continue;
                }
                hmin = hmin.min(store.interval_start_min(interval));
                hmax = hmax.max(store.interval_end_max(interval));
            }
            if hmin >= hmax {
                return Ok(()); // nothing present
            }
            let span = (hmax - hmin) as usize;

            // Mandatory-part profile from the compulsory region of every
            // *present* interval. Optional (undecided) intervals do not yet
            // consume the resource, so they are excluded from the profile.
            self.profile.clear();
            self.profile.resize(span, 0);
            for (idx, &interval) in self.intervals.iter().enumerate() {
                if store.interval_presence(interval) != IntervalPresence::Present {
                    continue;
                }
                let (cp_lo, cp_hi) = (store.interval_start_max(interval), store.interval_end_min(interval));
                for t in cp_lo..cp_hi {
                    self.profile[(t - hmin) as usize] += self.demands[idx];
                }
            }
            for &usage in &self.profile {
                if usage > self.capacity {
                    return Err(Inconsistency);
                }
            }

            // Push each present interval's start past instants it cannot cover.
            let mut changed = false;
            for idx in 0..n {
                let interval = self.intervals[idx];
                if store.interval_presence(interval) == IntervalPresence::Absent {
                    continue;
                }
                let demand = self.demands[idx];
                let duration = store.interval_duration(interval);
                if demand == 0 || duration == 0 {
                    continue;
                }
                let smin = store.interval_start_min(interval);
                let smax = store.interval_start_max(interval);
                // Subtract the interval's own compulsory region only if it is
                // present (so already in the profile); an optional interval does
                // not contribute, so nothing to subtract.
                let in_profile = store.interval_presence(interval) == IntervalPresence::Present;
                let (own_lo, own_hi) = (smax, smin + duration);

                let mut start = smin;
                let mut feasible = None;
                'scan: while start <= smax {
                    let mut t = start;
                    while t < start + duration {
                        let own = if in_profile && t >= own_lo && t < own_hi { demand } else { 0 };
                        if self.profile[(t - hmin) as usize] - own + demand > self.capacity {
                            start = t + 1; // `interval` cannot cover instant `t`
                            continue 'scan;
                        }
                        t += 1;
                    }
                    feasible = Some(start);
                    break;
                }

                match feasible {
                    Some(start) => {
                        if start > smin {
                            changed |= store.set_interval_start_min(interval, start)?;
                        }
                    }
                    None => match store.interval_presence(interval) {
                        IntervalPresence::Present => return Err(Inconsistency),
                        IntervalPresence::Optional => changed |= store.forbid_interval_presence(interval)?,
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

/// Post a structured cumulative resource: `intervals[k]` uses `demands[k]` units
/// of a resource of `capacity` while running.
pub fn cumulative(solver: &mut Solver, intervals: &[IntervalId], demands: &[i32], capacity: i32) -> PropId {
    solver.post(Box::new(Cumulative { intervals: intervals.to_vec(), demands: demands.to_vec(), capacity, profile: Vec::new() }))
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
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &mode in &self.modes {
            store.subscribe_interval(mode, me, IntervalEvent::PresenceChange);
        }
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
            if present > 1 || non_absent == 0 {
                return Err(Inconsistency);
            }
            let mut changed = false;
            if present == 1 {
                // A mode is chosen: forbid every other (still optional) mode.
                for &mode in &self.modes {
                    if store.interval_presence(mode) == IntervalPresence::Optional {
                        changed |= store.forbid_interval_presence(mode)?;
                    }
                }
            } else if non_absent == 1 {
                // Only one candidate remains: it must be present.
                changed |= store.require_interval_presence(self.modes[last_non_absent.unwrap()])?;
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
