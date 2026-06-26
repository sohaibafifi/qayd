//! Propagators over structured list and interval domains.

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
