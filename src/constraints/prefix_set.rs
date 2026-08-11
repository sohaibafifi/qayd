//! Compact implications linking an index prefix to forbidden slot values.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Premise, Solver, Store};

/// Enforces every disjunction `(index <= threshold) || (slot not in forbidden)`.
///
/// A decomposition may emit one such implication for every position of a
/// sequence. Keeping the common index and forbidden set in one propagator
/// avoids compiling the repeated Boolean expression trees independently.
#[derive(Clone)]
struct PrefixSetExclusion {
    index: VarId,
    entries: Vec<(i32, VarId)>,
    slots: Vec<VarId>,
    forbidden: Vec<i32>,
}

impl PrefixSetExclusion {
    fn propagate_with_stop(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        for &(threshold, slot) in &self.entries {
            if should_stop() {
                return Ok(());
            }

            // If every supported slot value is forbidden, the alternative on
            // the right-hand side is false and the index must remain inside
            // the prefix. Checking the whole current domain is necessary for
            // the same arc consistency as the decomposed binary expression:
            // a slot need not be fixed when several forbidden values remain.
            let mut slot_is_forced_forbidden = true;
            for value in store.values(slot) {
                if should_stop() {
                    return Ok(());
                }
                if self.forbidden.binary_search(&value).is_err() {
                    slot_is_forced_forbidden = false;
                    break;
                }
            }
            if slot_is_forced_forbidden {
                let mut slot_domain = Vec::new();
                if store.explaining() && !store.domain_premises_until(slot, &mut slot_domain, should_stop) {
                    return Ok(());
                }
                if store.min(self.index) > threshold {
                    slot_domain.push(Premise::Ge { var: self.index, bound: threshold + 1 });
                    return Err(store.fail_because(slot_domain));
                }
                store.remove_above_because(self.index, threshold, slot_domain)?;
            }

            if store.min(self.index) <= threshold {
                continue;
            }
            let index_past_prefix = Premise::Ge { var: self.index, bound: threshold + 1 };
            for &value in &self.forbidden {
                if should_stop() {
                    return Ok(());
                }
                if !store.contains(slot, value) {
                    continue;
                }
                if store.is_fixed(slot) {
                    return Err(store.fail_because(vec![index_past_prefix, Premise::Eq { var: slot, val: value }]));
                }
                let why = if store.explaining() { vec![index_past_prefix] } else { Vec::new() };
                store.remove_because(slot, value, why)?;
            }
        }
        Ok(())
    }
}

impl Propagator for PrefixSetExclusion {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.index, me, Event::BoundChange);
        for &slot in &self.slots {
            store.subscribe(slot, me, Event::DomainChange);
        }
    }

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        if should_stop() {
            return false;
        }
        store.subscribe(self.index, me, Event::BoundChange);
        for &slot in &self.slots {
            if should_stop() {
                return false;
            }
            store.subscribe(slot, me, Event::DomainChange);
        }
        !should_stop()
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        self.propagate_with_stop(store, &|| false)
    }

    fn propagate_until(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        if should_stop() {
            return Ok(());
        }
        self.propagate_with_stop(store, should_stop)
    }
}

/// Posts a normalized group of prefix exclusions. Duplicate entries and
/// forbidden values are semantically redundant and are collapsed here so
/// direct callers receive the same physical representation as the compiler.
pub(crate) fn prefix_set_exclusion_interruptible(
    solver: &mut Solver,
    index: VarId,
    entries: &[(i32, VarId)],
    forbidden: &[i32],
    stop: &AtomicBool,
) -> bool {
    if stop.load(Ordering::Acquire) || entries.is_empty() || forbidden.is_empty() {
        return false;
    }

    let mut normalized_entries = BTreeSet::new();
    for &entry in entries {
        if stop.load(Ordering::Acquire) || entry.0 == i32::MAX {
            return false;
        }
        normalized_entries.insert(entry);
    }
    let mut normalized_forbidden = BTreeSet::new();
    for &value in forbidden {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        normalized_forbidden.insert(value);
    }
    let mut entries = Vec::new();
    for entry in normalized_entries {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        entries.push(entry);
    }
    let mut forbidden = Vec::new();
    for value in normalized_forbidden {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        forbidden.push(value);
    }
    let mut normalized_slots = BTreeSet::new();
    for &(_, slot) in &entries {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        normalized_slots.insert(slot);
    }
    let mut slots = Vec::new();
    for slot in normalized_slots {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        slots.push(slot);
    }
    let should_stop = || stop.load(Ordering::Acquire);
    solver.post_until(Box::new(PrefixSetExclusion { index, entries, slots, forbidden }), &should_stop).is_some()
}
