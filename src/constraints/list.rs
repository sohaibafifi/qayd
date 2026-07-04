//! Propagators over list domains.

use crate::domains::list::ListEvent;
use crate::ids::{ListId, PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Premise, Solver, Store};

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
    fn priority(&self) -> Priority { Priority::Expensive }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &list in &self.lists {
            store.subscribe_list(list, me, ListEvent::PossibleChange);
            store.subscribe_list(list, me, ListEvent::RequiredChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        for &item in &self.items {
            let mut required: Vec<ListId> = Vec::new();
            let mut possible: Vec<ListId> = Vec::new();
            for &list in &self.lists {
                if store.list_required(list, item) {
                    required.push(list);
                }
                if store.list_possible(list, item) {
                    possible.push(list);
                }
            }

            if required.len() > 1 {
                // Two lists require the item: conflict, cite both memberships = 1.
                let why = required.iter().filter_map(|&l| store.list_member_var(l, item).map(|var| Premise::Eq { var, val: 1 })).collect();
                return Err(store.fail_because(why));
            }
            if possible.is_empty() {
                // No list can hold the item: conflict, cite every membership = 0.
                let why =
                    self.lists.iter().filter_map(|&l| store.list_member_var(l, item).map(|var| Premise::Eq { var, val: 0 })).collect();
                return Err(store.fail_because(why));
            }

            if let Some(&owner) = required.first() {
                // The item is required by `owner`: forbid it from every other list,
                // because `owner` already holds it.
                let why: Vec<Premise> = store.list_member_var(owner, item).map(|var| Premise::Eq { var, val: 1 }).into_iter().collect();
                for &list in &self.lists {
                    if list != owner {
                        store.forbid_list_item_because(list, item, why.clone())?;
                    }
                }
            } else if possible.len() == 1 {
                // Only one list still admits the item: require it there, because
                // every other list has excluded it.
                let last = possible[0];
                let why = self
                    .lists
                    .iter()
                    .filter(|&&l| l != last)
                    .filter_map(|&l| store.list_member_var(l, item).map(|var| Premise::Eq { var, val: 0 }))
                    .collect();
                store.require_list_item_because(last, item, why)?;
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
    fn priority(&self) -> Priority { Priority::Linear }

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

            if let (Some(i), Some(j)) = (owner_a, owner_b) {
                if i != j {
                    // a and b are pinned to different lists: same_list is violated.
                    let mut why = member_eq(store, self.lists[i], self.a, 1);
                    why.extend(member_eq(store, self.lists[j], self.b, 1));
                    return Err(store.fail_because(why));
                }
            }

            if let Some(i) = owner_a {
                // a is in lists[i], so b must be too.
                let why = member_eq(store, self.lists[i], self.a, 1);
                changed |= store.require_list_item_because(self.lists[i], self.b, why)?;
            }
            if let Some(i) = owner_b {
                let why = member_eq(store, self.lists[i], self.b, 1);
                changed |= store.require_list_item_because(self.lists[i], self.a, why)?;
            }

            let mut has_common_owner = false;
            let mut no_common_reason: Vec<Premise> = Vec::new();
            for &list in &self.lists {
                let a_possible = store.list_possible(list, self.a);
                let b_possible = store.list_possible(list, self.b);
                has_common_owner |= a_possible && b_possible;

                if !a_possible {
                    // a cannot be in `list`, so neither can b.
                    let why = member_eq(store, list, self.a, 0);
                    no_common_reason.extend(why.iter().cloned());
                    changed |= store.forbid_list_item_because(list, self.b, why)?;
                }
                if !b_possible {
                    let why = member_eq(store, list, self.b, 0);
                    no_common_reason.extend(why.iter().cloned());
                    changed |= store.forbid_list_item_because(list, self.a, why)?;
                }
            }

            if !has_common_owner {
                // No list still admits both items: cite the exclusion behind every
                // list (one of the two items is barred from each).
                return Err(store.fail_because(no_common_reason));
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
    fn priority(&self) -> Priority { Priority::Cheap }

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

            if let (Some(b), Some(a)) = (before_owner, after_owner) {
                if b > a {
                    // `before` sits in a later list than `after`: precedence violated.
                    let mut why = member_eq(store, self.lists[b], self.before, 1);
                    why.extend(member_eq(store, self.lists[a], self.after, 1));
                    return Err(store.fail_because(why));
                }
            }

            if let Some(b) = before_owner {
                // `before` is in list b, so `after` cannot be in any earlier list.
                let why = member_eq(store, self.lists[b], self.before, 1);
                for i in 0..b {
                    changed |= store.forbid_list_item_because(self.lists[i], self.after, why.clone())?;
                }
            }
            if let Some(a) = after_owner {
                // `after` is in list a, so `before` cannot be in any later list.
                let why = member_eq(store, self.lists[a], self.after, 1);
                for i in (a + 1)..self.lists.len() {
                    changed |= store.forbid_list_item_because(self.lists[i], self.before, why.clone())?;
                }
            }

            for i in 0..self.lists.len() {
                if store.list_possible(self.lists[i], self.before) && !self.has_after_support(store, i) {
                    // No list at or after i admits `after`, so `before` cannot sit at i.
                    let why = (i..self.lists.len()).flat_map(|j| member_eq(store, self.lists[j], self.after, 0)).collect();
                    changed |= store.forbid_list_item_because(self.lists[i], self.before, why)?;
                }
                if store.list_possible(self.lists[i], self.after) && !self.has_before_support(store, i) {
                    // No list at or before i admits `before`, so `after` cannot sit at i.
                    let why = (0..=i).flat_map(|j| member_eq(store, self.lists[j], self.before, 0)).collect();
                    changed |= store.forbid_list_item_because(self.lists[i], self.after, why)?;
                }
            }

            if !has_possible_owner(store, &self.lists, self.before) {
                return Err(store.fail_because(all_forbidden_reason(store, &self.lists, self.before)));
            }
            if !has_possible_owner(store, &self.lists, self.after) {
                return Err(store.fail_because(all_forbidden_reason(store, &self.lists, self.after)));
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

/// The coupling `sum(members) == length` for one list: keeps the length
/// variable within `[required, possible]` and, when the length is pinned to
/// either bound, fixes the still-undecided memberships. This is the only place
/// that relates membership and length, so a learning decision on any membership
/// or on the length routes back through this propagator's explanations rather
/// than through hidden `Store` logic.
#[derive(Clone)]
pub struct ListCardinality {
    list: ListId,
    items: Vec<i32>,
    length: VarId,
}

impl Propagator for ListCardinality {
    fn priority(&self) -> Priority { Priority::Linear }

    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe_list(self.list, me, ListEvent::PossibleChange);
        store.subscribe_list(self.list, me, ListEvent::RequiredChange);
        store.subscribe_list(self.list, me, ListEvent::LengthChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        loop {
            let required: Vec<i32> = self.items.iter().copied().filter(|&it| store.list_required(self.list, it)).collect();
            let forbidden: Vec<i32> = self.items.iter().copied().filter(|&it| !store.list_possible(self.list, it)).collect();
            let req_n = required.len() as i32;
            let poss_n = (self.items.len() - forbidden.len()) as i32;

            let before_min = store.min(self.length);
            let before_max = store.max(self.length);

            // length >= required: cite the required memberships.
            if before_min < req_n {
                let why: Vec<Premise> = required.iter().flat_map(|&it| member_eq(store, self.list, it, 1)).collect();
                store.remove_below_because(self.length, req_n, why)?;
            }
            // length <= possible: cite the excluded memberships.
            if before_max > poss_n {
                let why: Vec<Premise> = forbidden.iter().flat_map(|&it| member_eq(store, self.list, it, 0)).collect();
                store.remove_above_because(self.length, poss_n, why)?;
            }

            let mut changed = store.min(self.length) != before_min || store.max(self.length) != before_max;
            let lmin = store.min(self.length);
            let lmax = store.max(self.length);

            // length pinned at `required`: no room for more, forbid the undecided.
            if lmax == req_n && poss_n > req_n {
                let mut why = vec![Premise::Le { var: self.length, bound: req_n }];
                why.extend(required.iter().flat_map(|&it| member_eq(store, self.list, it, 1)));
                for &it in &self.items {
                    if store.list_possible(self.list, it) && !store.list_required(self.list, it) {
                        changed |= store.forbid_list_item_because(self.list, it, why.clone())?;
                    }
                }
            }
            // length pinned at `possible`: every remaining item is needed, require them.
            if lmin == poss_n && poss_n > req_n {
                let mut why = vec![Premise::Ge { var: self.length, bound: poss_n }];
                why.extend(forbidden.iter().flat_map(|&it| member_eq(store, self.list, it, 0)));
                for &it in &self.items {
                    if store.list_possible(self.list, it) && !store.list_required(self.list, it) {
                        changed |= store.require_list_item_because(self.list, it, why.clone())?;
                    }
                }
            }

            if !changed {
                return Ok(());
            }
        }
    }
}

/// Post the `sum(members) == length` coupling for `list`.
pub fn list_cardinality(solver: &mut Solver, list: ListId) -> PropId {
    let items = solver.store.list_universe(list).to_vec();
    let length = solver.store.list_length_var(list);
    solver.post(Box::new(ListCardinality { list, items, length }))
}

/// `used = (length(list) >= 1)`: the boolean indicator that a list is non-empty,
/// used to count open bins or used vehicles in an integer-backed objective.
#[derive(Clone)]
pub struct ListUsed {
    list: ListId,
    length: VarId,
    used: VarId,
}

impl Propagator for ListUsed {
    fn priority(&self) -> Priority { Priority::Linear }

    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe_list(self.list, me, ListEvent::LengthChange);
        store.subscribe(self.used, me, Event::Fix);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        // The length forces the indicator.
        if store.min(self.length) >= 1 {
            store.fix_because(self.used, 1, vec![Premise::Ge { var: self.length, bound: 1 }])?;
        } else if store.max(self.length) <= 0 {
            store.fix_because(self.used, 0, vec![Premise::Le { var: self.length, bound: 0 }])?;
        }
        // The indicator forces the length.
        if store.is_fixed(self.used) {
            if store.value(self.used) == 1 {
                store.remove_below_because(self.length, 1, vec![Premise::Eq { var: self.used, val: 1 }])?;
            } else {
                store.remove_above_because(self.length, 0, vec![Premise::Eq { var: self.used, val: 0 }])?;
            }
        }
        Ok(())
    }
}

/// Post `used = (length(list) >= 1)`.
pub fn list_used(solver: &mut Solver, list: ListId, used: VarId) -> PropId {
    let length = solver.store.list_length_var(list);
    solver.post(Box::new(ListUsed { list, length, used }))
}

/// Bounds the number of required items in a list.
#[derive(Clone)]
pub struct ListLength {
    list: ListId,
    min: usize,
    max: usize,
}

impl Propagator for ListLength {
    fn priority(&self) -> Priority { Priority::Cheap }

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

/// Bounds a weighted item sum over one list.
#[derive(Clone)]
pub struct ListItemSum {
    list: ListId,
    weights: Vec<(i32, i64)>,
    min: i64,
    max: i64,
}

impl Propagator for ListItemSum {
    fn priority(&self) -> Priority { Priority::Linear }

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
            // Floor: required plus every still-takeable negative item (taking them
            // lowers the sum). Ceiling: required plus every still-takeable positive.
            let lower = required.saturating_add(negative_optional);
            let upper = required.saturating_add(positive_optional);
            if lower > self.max {
                let why = self.lower_reason(store);
                return Err(store.fail_because(why));
            }
            if upper < self.min {
                let why = self.upper_reason(store);
                return Err(store.fail_because(why));
            }

            // Within one pass the floor only rises and the ceiling only falls, and
            // both reasons stay valid (excluding/requiring an item only adds facts).
            // So compute each reason once against the top-of-pass bounds and reuse
            // it: every deduction below is sound against the current state, and the
            // outer loop re-tightens with the fresh bounds.
            let lower_reason = self.lower_reason(store);
            let upper_reason = self.upper_reason(store);
            let mut changed = false;
            for &(item, weight) in &self.weights {
                if !store.list_possible(self.list, item) || store.list_required(self.list, item) {
                    continue;
                }
                if weight > 0 {
                    if lower.saturating_add(weight) > self.max {
                        // Taking a positive item raises the floor past max: forbid it.
                        changed |= store.forbid_list_item_because(self.list, item, lower_reason.clone())?;
                    } else if upper.saturating_sub(weight) < self.min {
                        // Dropping a positive item lowers the ceiling below min: require it.
                        changed |= store.require_list_item_because(self.list, item, upper_reason.clone())?;
                    }
                } else if weight < 0 {
                    if upper.saturating_add(weight) < self.min {
                        // Taking a negative item lowers the ceiling below min: forbid it.
                        changed |= store.forbid_list_item_because(self.list, item, upper_reason.clone())?;
                    } else if lower.saturating_sub(weight) > self.max {
                        // Dropping a negative item raises the floor past max: require it.
                        changed |= store.require_list_item_because(self.list, item, lower_reason.clone())?;
                    }
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

    /// Why the sum cannot drop below `lower`: every required item contributes its
    /// weight, and every excluded negative item is no longer available to lower it.
    fn lower_reason(&self, store: &Store) -> Vec<Premise> {
        let mut why = Vec::new();
        for &(item, weight) in &self.weights {
            if store.list_required(self.list, item) {
                why.extend(member_eq(store, self.list, item, 1));
            } else if weight < 0 && !store.list_possible(self.list, item) {
                why.extend(member_eq(store, self.list, item, 0));
            }
        }
        why
    }

    /// Why the sum cannot rise above `upper`: every required item contributes its
    /// weight, and every excluded positive item is no longer available to raise it.
    fn upper_reason(&self, store: &Store) -> Vec<Premise> {
        let mut why = Vec::new();
        for &(item, weight) in &self.weights {
            if store.list_required(self.list, item) {
                why.extend(member_eq(store, self.list, item, 1));
            } else if weight > 0 && !store.list_possible(self.list, item) {
                why.extend(member_eq(store, self.list, item, 0));
            }
        }
        why
    }
}

/// Post weighted item-sum bounds.
pub fn list_item_sum(solver: &mut Solver, list: ListId, weights: Vec<(i32, i64)>, min: i64, max: i64) -> PropId {
    solver.post(Box::new(ListItemSum { list, weights, min, max }))
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

/// The premise `member(list, item) == val`, as a one-element reason (empty if
/// `item` is not in the list's universe).
fn member_eq(store: &Store, list: ListId, item: i32, val: i32) -> Vec<Premise> {
    store.list_member_var(list, item).map(|var| Premise::Eq { var, val }).into_iter().collect()
}

/// Why `item` can no longer be placed anywhere: it is excluded from every list.
fn all_forbidden_reason(store: &Store, lists: &[ListId], item: i32) -> Vec<Premise> {
    lists.iter().flat_map(|&list| member_eq(store, list, item, 0)).collect()
}
