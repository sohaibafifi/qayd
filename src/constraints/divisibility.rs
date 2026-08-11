//! Compact propagation for repeated modular exclusion predicates.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Premise, Solver, Store};

/// Enforces the conjunction, for every `d` in `divisors`, of
/// `x mod d != 0 || y mod d != 0`.
#[derive(Clone)]
struct SharedDivisibilityExclusion {
    x: VarId,
    y: VarId,
    divisors: Vec<i64>,
    active_divisors: Vec<i64>,
    values: Vec<i32>,
}

impl SharedDivisibilityExclusion {
    fn filter_from(
        &mut self,
        store: &mut Store,
        source: VarId,
        target: VarId,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<(), Inconsistency> {
        let source_value = store.value(source);
        self.active_divisors.clear();
        for &divisor in &self.divisors {
            if should_stop() {
                return Ok(());
            }
            if i64::from(source_value) % divisor == 0 {
                self.active_divisors.push(divisor);
            }
        }
        if self.active_divisors.is_empty() {
            return Ok(());
        }

        self.values.clear();
        for value in store.values(target) {
            if should_stop() {
                return Ok(());
            }
            self.values.push(value);
        }
        for &value in &self.values {
            if should_stop() {
                return Ok(());
            }
            if !store.contains(target, value) {
                continue;
            }
            let mut excluded = false;
            for &divisor in &self.active_divisors {
                if should_stop() {
                    return Ok(());
                }
                if i64::from(value) % divisor == 0 {
                    excluded = true;
                    break;
                }
            }
            if !excluded {
                continue;
            }

            if store.is_fixed(target) {
                let why = if store.explaining() {
                    let mut why = vec![Premise::Eq { var: source, val: source_value }];
                    if target != source {
                        why.push(Premise::Eq { var: target, val: value });
                    }
                    why
                } else {
                    Vec::new()
                };
                return Err(store.fail_because(why));
            }

            let why = if store.explaining() { vec![Premise::Eq { var: source, val: source_value }] } else { Vec::new() };
            store.remove_because(target, value, why)?;
        }
        Ok(())
    }

    fn propagate_with_stop(&mut self, store: &mut Store, should_stop: &dyn Fn() -> bool) -> Result<(), Inconsistency> {
        if store.is_fixed(self.x) {
            self.filter_from(store, self.x, self.y, should_stop)?;
        }
        if should_stop() {
            return Ok(());
        }
        if store.is_fixed(self.y) {
            self.filter_from(store, self.y, self.x, should_stop)?;
        }
        Ok(())
    }
}

impl Propagator for SharedDivisibilityExclusion {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.x, me, Event::Fix);
        store.subscribe(self.y, me, Event::Fix);
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

/// Post a conjunction of `x mod d != 0 || y mod d != 0` predicates while
/// polling during registration. Divisors must be positive and greater than
/// one.
pub(crate) fn shared_divisibility_exclusion_interruptible(
    solver: &mut Solver,
    x: VarId,
    y: VarId,
    divisors: &[i64],
    stop: &AtomicBool,
) -> bool {
    if stop.load(Ordering::Acquire) {
        return false;
    }
    let mut normalized = BTreeSet::new();
    for &divisor in divisors {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        assert!(divisor > 1, "shared divisibility exclusion requires divisors greater than one");
        normalized.insert(divisor);
    }
    let mut divisors = Vec::new();
    for divisor in normalized {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        divisors.push(divisor);
    }
    assert!(!divisors.is_empty(), "shared divisibility exclusion requires at least one divisor");
    let should_stop = || stop.load(Ordering::Acquire);
    solver
        .post_until(Box::new(SharedDivisibilityExclusion { x, y, divisors, active_divisors: Vec::new(), values: Vec::new() }), &should_stop)
        .is_some()
}
