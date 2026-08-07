//! Behavioural tests for the primitive propagators: forward checking and
//! idempotence (propagate twice, assert no further change).

use qayd::constraints::primitives::{not_equal, not_equal_offset};
use qayd::Solver;

#[test]
fn not_equal_forward_checks_and_is_idempotent() {
    let mut s = Solver::new();
    let x = s.new_var_range(0, 3);
    let y = s.new_var_range(0, 3);
    not_equal(&mut s, x, y);

    s.store.push_level();
    s.store.fix(x, 2).unwrap();
    s.propagate().unwrap();

    // Forward checking removed 2 from y.
    assert!(!s.store.contains(y, 2));
    let size_after = s.store.size(y);
    assert_eq!(size_after, 3);

    // Idempotence: a second fixpoint changes nothing.
    s.propagate().unwrap();
    assert_eq!(s.store.size(y), size_after);

    s.store.pop_level();
}

#[test]
fn not_equal_offset_prunes_shifted_value() {
    // x != y + 1. Fix y = 2  =>  x != 3.
    let mut s = Solver::new();
    let x = s.new_var_range(0, 5);
    let y = s.new_var_range(0, 5);
    not_equal_offset(&mut s, x, y, 1);

    s.store.push_level();
    s.store.fix(y, 2).unwrap();
    s.propagate().unwrap();
    assert!(!s.store.contains(x, 3));
    assert!(s.store.contains(x, 2));
    s.store.pop_level();
}

#[test]
fn propagation_failure_is_detected() {
    // Two variables, both pinned to the same value via != => inconsistency.
    let mut s = Solver::new();
    let x = s.new_var_set(&[4]);
    let y = s.new_var_set(&[4]);
    not_equal(&mut s, x, y);

    s.store.push_level();
    // x is already fixed to 4; propagation must wipe 4 from y and fail.
    let res = s.propagate();
    assert!(res.is_err());
    s.store.pop_level();
}
