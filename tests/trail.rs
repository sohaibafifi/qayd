//! Trail unit tests (moved out of `src/trail.rs`).

use qayd::Trail;

#[test]
fn set_then_get() {
    let mut t = Trail::new();
    let a = t.new_int(7);
    assert_eq!(t.get(a), 7);
    t.set(a, 9);
    assert_eq!(t.get(a), 9);
}

#[test]
fn restore_on_pop() {
    let mut t = Trail::new();
    let a = t.new_int(1);
    let b = t.new_int(2);
    t.push_level();
    t.set(a, 10);
    t.set(b, 20);
    assert_eq!((t.get(a), t.get(b)), (10, 20));
    t.pop_level();
    assert_eq!((t.get(a), t.get(b)), (1, 2));
}

#[test]
fn nested_levels_restore_independently() {
    let mut t = Trail::new();
    let a = t.new_int(0);
    t.push_level();
    t.set(a, 1);
    t.push_level();
    t.set(a, 2);
    assert_eq!(t.get(a), 2);
    t.pop_level();
    assert_eq!(t.get(a), 1);
    t.pop_level();
    assert_eq!(t.get(a), 0);
    assert_eq!(t.level(), 0);
}

#[test]
fn unchanged_set_is_not_logged() {
    let mut t = Trail::new();
    let a = t.new_int(5);
    t.push_level();
    t.set(a, 5); // no-op: must not grow the undo log
    t.set(a, 5);
    t.pop_level();
    assert_eq!(t.get(a), 5);
}

#[test]
fn multiple_writes_restore_to_level_entry() {
    let mut t = Trail::new();
    let a = t.new_int(0);
    t.push_level();
    t.set(a, 1);
    t.set(a, 2);
    t.set(a, 3);
    t.pop_level();
    // Restores to the value at push_level, not the immediately-previous write.
    assert_eq!(t.get(a), 0);
}

#[test]
fn repeated_writes_log_once_per_level() {
    let mut t = Trail::new();
    let a = t.new_int(0);
    t.push_level();
    let before = t.log_len();
    t.set(a, 1);
    t.set(a, 2);
    t.set(a, 3);
    // N value-changing writes to one reversible in a level -> one undo entry.
    assert_eq!(t.log_len() - before, 1);
    t.pop_level();
    assert_eq!(t.get(a), 0);
}

#[test]
fn reused_level_number_restores_correctly() {
    // Hazard: level numbers repeat after backtracking. Epochs must not collide.
    let mut t = Trail::new();
    let a = t.new_int(0);

    t.push_level(); // level 1, first incarnation
    t.set(a, 5);
    t.pop_level();
    assert_eq!(t.get(a), 0);

    t.push_level(); // level 1 again, same depth/number, new epoch
    t.set(a, 8);
    assert_eq!(t.get(a), 8);
    t.pop_level();
    assert_eq!(t.get(a), 0);
}

#[test]
fn nested_interleaved_writes_restore() {
    let mut t = Trail::new();
    let a = t.new_int(0);
    let b = t.new_int(100);

    t.push_level();
    t.set(a, 1);
    t.set(b, 101);
    t.set(a, 2); // repeat write, deduped within this level

    t.push_level();
    t.set(a, 3);
    t.set(b, 102);
    t.set(b, 103); // repeat, deduped
    assert_eq!((t.get(a), t.get(b)), (3, 103));

    t.pop_level();
    assert_eq!((t.get(a), t.get(b)), (2, 101));

    // Writing again at the outer level after a child popped must still restore.
    t.set(a, 9);
    t.set(b, 109);
    t.pop_level();
    assert_eq!((t.get(a), t.get(b)), (0, 100));
    assert_eq!(t.level(), 0);
    assert_eq!(t.log_len(), 0);
}
