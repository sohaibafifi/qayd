//! Propagators over list domains.

use crate::domains::list::ListEvent;
use crate::ids::{ListId, PropId};
use crate::propagator::{Inconsistency, Propagator};
use crate::store::{Solver, Store};

/// Exact partition over list membership.
///
/// Every item must be required by exactly one list. This first list
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

/// Post a list partition.
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
