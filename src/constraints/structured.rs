//! Propagators over structured list and interval domains.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use crate::ids::{IntervalId, ListId, PropId, VarId};
use crate::propagator::{Inconsistency, Propagator};
use crate::store::{Premise, Solver, Store};
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
    /// `(a, b, order index)` for each unordered pair; the order index addresses
    /// the trailed order decision in the store.
    pairs: Vec<(usize, usize, usize)>,
    /// `pair_index[a][b]` (a < b) is the order index of that pair, for O(1) lookup
    /// of a decided order; `usize::MAX` off the upper triangle.
    pair_index: Vec<Vec<usize>>,
    /// Reused scratch for detectable precedences (no per-call allocation).
    present: Vec<usize>,
    prec: Vec<usize>,
}

/// Whether group interval `i` is decided to run before group interval `j`.
fn decided_before(store: &Store, pair_index: &[Vec<usize>], i: usize, j: usize) -> bool {
    let (lo, hi, want) = if i < j { (i, j, 1) } else { (j, i, 2) };
    let order = pair_index[lo][hi];
    order != usize::MAX && store.disjunctive_order(order) == want
}

/// Enforce `before` ends no later than `after` starts on the start bounds (both
/// present), citing the decided order `order_var = order_value` and the bounds it
/// rests on. Returns whether a bound changed.
fn enforce_before_because(store: &mut Store, before: IntervalId, after: IntervalId, order_var: VarId, order_value: i32) -> Result<bool, Inconsistency> {
    let duration = store.interval_duration(before);
    let before_start = store.interval_start_var(before);
    let after_start = store.interval_start_var(after);
    let before_lb = store.interval_start_min(before);
    let after_ub = store.interval_start_max(after);

    // after.start >= before.start_min + duration(before); reason: the order, plus
    // before's lower bound and both presences.
    let mut why_after = vec![Premise::Eq { var: order_var, val: order_value }, Premise::Ge { var: before_start, bound: before_lb }];
    why_after.extend(present_premise(store, before));
    why_after.extend(present_premise(store, after));
    let mut changed = store.set_interval_start_min_because(after, before_lb.saturating_add(duration), why_after)?;

    // before.start <= after.start_max - duration(before); reason: the order, plus
    // after's upper bound and both presences.
    let mut why_before = vec![Premise::Eq { var: order_var, val: order_value }, Premise::Le { var: after_start, bound: after_ub }];
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

impl Propagator for NoOverlap {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &interval in &self.intervals {
            store.subscribe_interval(interval, me, IntervalEvent::StartBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::EndBoundChange);
            store.subscribe_interval(interval, me, IntervalEvent::PresenceChange);
        }
        self.pairs.clear();
        let n = self.intervals.len();
        self.pair_index = vec![vec![usize::MAX; n]; n];
        for a in 0..n {
            for b in (a + 1)..n {
                let order = store.register_disjunctive_pair(self.intervals[a], self.intervals[b], me);
                self.pairs.push((a, b, order));
                self.pair_index[a][b] = order;
            }
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        loop {
            let mut changed = false;
            for &(a, b, order_index) in &self.pairs {
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
                let order_var = store.disjunctive_order_var(order_index);
                match store.disjunctive_order(order_index) {
                    // Order already decided (by the brancher, a deduction, or a
                    // learning-engine branch): durably enforce that precedence.
                    1 if both_present => changed |= enforce_before_because(store, i, j, order_var, 1)?,
                    2 if both_present => changed |= enforce_before_because(store, j, i, order_var, 0)?,
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
                                changed |= store.set_disjunctive_order_because(order_index, 1, why)?;
                                changed |= enforce_before_because(store, i, j, order_var, 1)?;
                            }
                            (false, true) if both_present => {
                                // i cannot precede j, so j must run first.
                                let mut why = vec![
                                    Premise::Ge { var: store.interval_start_var(i), bound: store.interval_start_min(i) },
                                    Premise::Le { var: store.interval_start_var(j), bound: store.interval_start_max(j) },
                                ];
                                why.extend(present_premise(store, i));
                                why.extend(present_premise(store, j));
                                changed |= store.set_disjunctive_order_because(order_index, 2, why)?;
                                changed |= enforce_before_because(store, j, i, order_var, 0)?;
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
            changed |= self.detectable_precedences(store)?;

            if !changed {
                return Ok(());
            }
        }
    }
}

impl NoOverlap {
    fn detectable_precedences(&mut self, store: &mut Store) -> Result<bool, Inconsistency> {
        self.present.clear();
        for k in 0..self.intervals.len() {
            if store.interval_presence(self.intervals[k]) == IntervalPresence::Present && store.interval_duration(self.intervals[k]) > 0 {
                self.present.push(k);
            }
        }
        let mut changed = false;
        for index in 0..self.present.len() {
            let qj = self.present[index];
            let j = self.intervals[qj];
            // Predecessors that must run before j: decided so, or unable to fit
            // after j (their latest start is before j's earliest end).
            self.prec.clear();
            for &pi in &self.present {
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
            let mut ect = i32::MIN;
            for &pi in &self.prec {
                let i = self.intervals[pi];
                ect = ect.max(store.interval_start_min(i)).saturating_add(store.interval_duration(i));
            }
            changed |= store.set_interval_start_min(j, ect)?;
        }
        Ok(changed)
    }
}

/// Post a structured unary-resource no-overlap over the given intervals.
pub fn no_overlap(solver: &mut Solver, intervals: &[IntervalId]) -> PropId {
    solver.post(Box::new(NoOverlap { intervals: intervals.to_vec(), pairs: Vec::new(), pair_index: Vec::new(), present: Vec::new(), prec: Vec::new() }))
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
