//! Domain unit tests (moved out of `src/domain.rs`).

use qayd::domains::int::Domain;
use qayd::trail::Trail;
use qayd::Inconsistency;

fn present(dom: &Domain, trail: &Trail) -> Vec<i32> {
    let mut vs: Vec<i32> = dom.values(trail).collect();
    vs.sort_unstable();
    vs
}

#[test]
fn range_basics() {
    let mut t = Trail::new();
    let d = Domain::new_range(2, 5, &mut t);
    assert_eq!(d.size(&t), 4);
    assert_eq!(d.min(&t), 2);
    assert_eq!(d.max(&t), 5);
    assert!(d.contains(2, &t));
    assert!(d.contains(5, &t));
    assert!(!d.contains(1, &t));
    assert!(!d.contains(6, &t));
    assert_eq!(present(&d, &t), vec![2, 3, 4, 5]);
}

#[test]
fn remove_updates_bounds() {
    let mut t = Trail::new();
    let mut d = Domain::new_range(0, 4, &mut t);
    assert!(d.remove(0, &mut t));
    assert_eq!(d.min(&t), 1);
    assert!(d.remove(4, &mut t));
    assert_eq!(d.max(&t), 3);
    assert!(d.remove(2, &mut t)); // interior: bounds unchanged
    assert_eq!((d.min(&t), d.max(&t)), (1, 3));
    assert_eq!(present(&d, &t), vec![1, 3]);
    assert!(!d.remove(2, &mut t)); // already gone: no-op
}

#[test]
fn remove_restored_on_backtrack() {
    let mut t = Trail::new();
    let mut d = Domain::new_range(0, 4, &mut t);
    t.push_level();
    d.remove(1, &mut t);
    d.remove(2, &mut t);
    assert_eq!(present(&d, &t), vec![0, 3, 4]);
    t.pop_level();
    assert_eq!(present(&d, &t), vec![0, 1, 2, 3, 4]);
    assert_eq!((d.min(&t), d.max(&t), d.size(&t)), (0, 4, 5));
}

#[test]
fn fix_keeps_only_one() {
    let mut t = Trail::new();
    let mut d = Domain::new_range(0, 9, &mut t);
    assert_eq!(d.fix(4, &mut t), Ok(true));
    assert!(d.is_fixed(&t));
    assert_eq!((d.min(&t), d.max(&t)), (4, 4));
    assert!(d.contains(4, &t));
    assert!(!d.contains(3, &t));
    assert_eq!(d.fix(4, &mut t), Ok(false)); // already fixed: no change
    assert_eq!(d.fix(5, &mut t), Err(Inconsistency)); // absent
}

#[test]
fn bound_pruning() {
    let mut t = Trail::new();
    let mut d = Domain::new_range(0, 9, &mut t);
    assert!(d.remove_below(3, &mut t));
    assert_eq!(d.min(&t), 3);
    assert!(d.remove_above(6, &mut t));
    assert_eq!(d.max(&t), 6);
    assert_eq!(present(&d, &t), vec![3, 4, 5, 6]);
    assert!(!d.remove_below(3, &mut t)); // no-op
}

#[test]
fn set_with_holes() {
    let mut t = Trail::new();
    let d = Domain::new_set(&[1, 1, 4, 7, 4], &mut t);
    assert_eq!(d.size(&t), 3);
    assert_eq!((d.min(&t), d.max(&t)), (1, 7));
    assert!(d.contains(4, &t));
    assert!(!d.contains(2, &t));
    assert_eq!(present(&d, &t), vec![1, 4, 7]);
}

#[test]
fn sparse_set_handles_extreme_gaps() {
    let mut t = Trail::new();
    let mut d = Domain::new_set(&[i32::MIN, 0, i32::MAX], &mut t);
    assert_eq!(present(&d, &t), vec![i32::MIN, 0, i32::MAX]);

    t.push_level();
    assert!(d.remove_below(-1, &mut t));
    assert_eq!(present(&d, &t), vec![0, i32::MAX]);
    assert!(d.remove_above(0, &mut t));
    assert_eq!(present(&d, &t), vec![0]);
    t.pop_level();

    assert_eq!(present(&d, &t), vec![i32::MIN, 0, i32::MAX]);
}

#[test]
fn wide_range_uses_bounds_and_restores_holes() {
    let mut t = Trail::new();
    let mut d = Domain::new_range(i32::MIN, i32::MAX, &mut t);
    assert_eq!(d.size(&t), 1usize << 32);

    t.push_level();
    assert!(d.remove(0, &mut t));
    assert!(d.remove_below(-2, &mut t));
    assert!(d.remove_above(2, &mut t));
    assert_eq!(present(&d, &t), vec![-2, -1, 1, 2]);
    t.pop_level();

    assert_eq!((d.min(&t), d.max(&t)), (i32::MIN, i32::MAX));
    assert!(d.contains(0, &t));
    assert_eq!(d.size(&t), 1usize << 32);
}

#[test]
fn remove_down_to_one() {
    let mut t = Trail::new();
    let mut d = Domain::new_range(0, 2, &mut t);
    d.remove(0, &mut t);
    d.remove(2, &mut t);
    assert!(d.is_fixed(&t));
    assert_eq!(d.min(&t), 1);
    // Removing the last value wipes the domain (size 0).
    d.remove(1, &mut t);
    assert_eq!(d.size(&t), 0);
}

#[test]
fn threshold_range_uses_dense_storage() {
    // 4096 values: exactly at the dense-range threshold (MAX_DENSE_RANGE_VALUES).
    let mut t = Trail::new();
    let d = Domain::new_range(0, 4095, &mut t);
    assert!(!d.is_bounds() && !d.is_sparse());
    assert_eq!(d.size(&t), 4096);
}

#[test]
fn above_threshold_range_uses_bounds_storage() {
    // 4097 values: one past the dense-range threshold.
    let mut t = Trail::new();
    let mut d = Domain::new_range(0, 4096, &mut t);
    assert!(d.is_bounds());
    assert_eq!(d.size(&t), 4097);

    assert!(d.remove_below(4096, &mut t));
    assert_eq!(d.size(&t), 1);
    assert_eq!((d.min(&t), d.max(&t)), (4096, 4096));
}

#[test]
fn compact_explicit_sets_still_use_dense_storage() {
    let mut t = Trail::new();
    let d = Domain::new_set(&[0, 1, 2, 3, 4], &mut t);
    assert!(!d.is_bounds() && !d.is_sparse());
}

// `size()` must equal a naive recount of present values, for a bounds-backed
// domain, under interleaved interior removes and bound moves with backtracking.
fn recount(dom: &Domain, trail: &Trail) -> usize {
    dom.values(trail).count()
}

#[test]
fn bounds_size_matches_recount_under_interleaving() {
    let mut t = Trail::new();
    let mut d = Domain::new_range(0, 5000, &mut t); // > 4096 -> bounds storage
    assert!(d.is_bounds());
    assert_eq!(d.size(&t), recount(&d, &t));

    t.push_level();
    // Interior holes, then bound moves that step over some of them.
    for v in [10, 11, 12, 100, 2500, 4990, 4991] {
        d.remove(v, &mut t);
        assert_eq!(d.size(&t), recount(&d, &t));
        assert_eq!(d.is_fixed(&t), recount(&d, &t) == 1);
    }
    // remove_below jumps past the {10,11,12,100} holes; those must not be
    // double-counted against the present total.
    d.remove_below(200, &mut t);
    assert_eq!(d.size(&t), recount(&d, &t));
    d.remove_above(4980, &mut t); // steps back over nothing here
    assert_eq!(d.size(&t), recount(&d, &t));
    // Removing the current min/max (bound moves over adjacent holes).
    d.remove(200, &mut t);
    d.remove(4980, &mut t);
    assert_eq!(d.size(&t), recount(&d, &t));

    let mid = present(&d, &t)[recount(&d, &t) / 2];
    d.fix(mid, &mut t).unwrap();
    assert!(d.is_fixed(&t));
    assert_eq!(d.size(&t), 1);
    assert_eq!(recount(&d, &t), 1);

    t.pop_level();
    assert_eq!(d.size(&t), 5001);
    assert_eq!(d.size(&t), recount(&d, &t));
}

#[test]
fn bounds_remove_below_over_existing_holes_no_double_count() {
    let mut t = Trail::new();
    let mut d = Domain::new_range(0, 10_000, &mut t);
    // Punch holes at 1..=50, then remove_below(100).
    for v in 1..=50 {
        d.remove(v, &mut t);
    }
    assert_eq!(d.size(&t), 10_001 - 50);
    d.remove_below(100, &mut t); // drops 0 + (1..=50 already gone) + 51..=99
    assert_eq!(d.min(&t), 100);
    assert_eq!(d.size(&t), recount(&d, &t));
    assert_eq!(d.size(&t), 10_001 - 100); // present = 100..=10000
}

// Micro-bench for the hot path the audit flagged: size()/is_fixed() are queried
// per mutation (Fix events check size==1). With H interior holes, the old code
// rescanned holes[..H] on every query (O(H)); now it is O(1). Punch a wall of
// holes once, then hammer size()/is_fixed() and assert it stays cheap.
#[test]
fn bounds_size_query_is_o1_with_many_holes() {
    use std::time::Instant;
    let mut t = Trail::new();
    let mut d = Domain::new_range(0, 100_000, &mut t);
    assert!(d.is_bounds());

    // Punch 20_000 interior holes.
    let mut v = 2;
    while v <= 40_000 {
        d.remove(v, &mut t);
        v += 2;
    }
    assert_eq!(d.size(&t), recount(&d, &t));

    // Hammer the per-mutation queries; O(1) means this is flat regardless of H.
    let start = Instant::now();
    let mut sink = 0usize;
    for _ in 0..2_000_000 {
        sink += d.size(&t) + usize::from(d.is_fixed(&t));
    }
    let elapsed = start.elapsed();
    assert!(sink > 0);
    // 2M O(1) queries over a 20k-hole domain: trivial. Old O(H) code would do
    // ~4e10 comparisons here.
    assert!(elapsed.as_secs() < 2, "size() not O(1): {elapsed:?}");
    eprintln!("bounds_size_query_is_o1_with_many_holes: 2M size()+is_fixed() over 20k holes in {elapsed:?}");
}
