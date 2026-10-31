use std::collections::BTreeSet;
use std::sync::mpsc;
use std::time::Duration;

use qayd::constraints::interval::no_overlap;
use qayd::{solve_domains, SearchControl, Solver};

#[test]
fn wide_start_windows_propagate_without_enumerating_the_horizon() {
    // Bounds domains deliberately make the numeric horizon much larger than
    // the actual scheduling input. A resource pass must finish promptly when
    // both orders have support and no date can be removed.
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut solver = Solver::new();
        let intervals: Vec<_> = (1..=12).map(|duration| solver.store.new_interval(-1_000_000_000, 1_000_000_000, duration)).collect();
        no_overlap(&mut solver, &intervals);
        solver.propagate().expect("wide windows are feasible");
        let bounds: Vec<_> =
            intervals.iter().map(|&iv| (solver.store.interval_start_min(iv), solver.store.interval_start_max(iv))).collect();
        let _ = sender.send(bounds);
    });
    let bounds = receiver.recv_timeout(Duration::from_secs(5)).expect("NoOverlap must not enumerate billions of start dates");
    assert_eq!(bounds, vec![(-1_000_000_000, 1_000_000_000); 12]);
}

#[test]
fn a_narrow_forbidden_window_in_a_wide_domain_is_exact_and_trailed() {
    let mut solver = Solver::new();
    let a = solver.store.new_interval(0, 1_000_000_000, 4);
    let b = solver.store.new_interval(0, 1_000_000_000, 3);
    let a_start = solver.store.interval_start_var(a);
    let b_start = solver.store.interval_start_var(b);
    no_overlap(&mut solver, &[a, b]);
    solver.propagate().unwrap();
    solver.store.push_level();
    solver.store.remove_below(b_start, 500_000_000).unwrap();
    solver.store.remove_above(b_start, 500_000_001).unwrap();
    solver.propagate().unwrap();
    for value in 499_999_998..=500_000_002 {
        assert!(!solver.store.contains(a_start, value), "start {value} has no support in either order");
    }
    assert!(solver.store.contains(a_start, 499_999_997));
    assert!(solver.store.contains(a_start, 500_000_003));
    solver.store.pop_level();
    for value in 499_999_998..=500_000_002 {
        assert!(solver.store.contains(a_start, value), "backtracking must restore an interior removal");
    }
}

#[test]
fn sparse_supports_across_a_large_span_preserve_their_holes() {
    let mut solver = Solver::new();
    let a = solver.new_var_set(&[-1_000_000_000, 0, 1_000_000_000]);
    let b = solver.new_var_set(&[-2, 3]);
    qayd::constraints::scheduling::no_overlap(&mut solver, &[a, b], &[10, 3]);
    solver.propagate().unwrap();
    assert_eq!(solver.store.values(a).collect::<BTreeSet<_>>(), BTreeSet::from([-1_000_000_000, 1_000_000_000]));
    assert_eq!(solver.store.values(b).collect::<BTreeSet<_>>(), BTreeSet::from([-2, 3]));
}

#[test]
fn holey_start_domains_match_exhaustive_no_overlap_oracle() {
    let domains = [vec![-4, -1, 0, 2, 5], vec![-3, -1, 1, 4], vec![-4, 0, 1, 3, 5]];
    for durations in [[1, 2, 3], [3, 1, 2], [4, 3, 2]] {
        let mut solver = Solver::new();
        let intervals: Vec<_> = domains
            .iter()
            .zip(durations)
            .map(|(values, duration)| {
                let interval = solver.store.new_interval(*values.first().unwrap(), *values.last().unwrap(), duration);
                let start = solver.store.interval_start_var(interval);
                for value in *values.first().unwrap()..=*values.last().unwrap() {
                    if !values.contains(&value) {
                        solver.store.remove(start, value).unwrap();
                    }
                }
                interval
            })
            .collect();
        no_overlap(&mut solver, &intervals);
        let mut actual = BTreeSet::new();
        solve_domains(&mut solver, |_, solution| {
            actual.insert(intervals.iter().map(|iv| solution.interval_starts[iv.index()].unwrap()).collect::<Vec<_>>());
            SearchControl::Continue
        });
        let mut expected = BTreeSet::new();
        for &a in &domains[0] {
            for &b in &domains[1] {
                for &c in &domains[2] {
                    let starts = [a, b, c];
                    if (0..3).all(|i| ((i + 1)..3).all(|j| starts[i] + durations[i] <= starts[j] || starts[j] + durations[j] <= starts[i]))
                    {
                        expected.insert(starts.to_vec());
                    }
                }
            }
        }
        assert_eq!(actual, expected, "durations {durations:?}");
    }
}

#[test]
fn forbidden_windows_near_integer_endpoints_do_not_wrap() {
    for offset in [i32::MIN, i32::MAX - 8] {
        let mut solver = Solver::new();
        let a = solver.store.new_interval(offset, offset + 6, 2);
        let b = solver.store.new_interval(offset + 3, offset + 4, 2);
        let start = solver.store.interval_start_var(a);
        no_overlap(&mut solver, &[a, b]);
        solver.propagate().unwrap();
        let actual: BTreeSet<_> = solver.store.values(start).collect();
        let expected: BTreeSet<_> = (offset..=offset + 6)
            .filter(|&value| {
                (offset + 3..=offset + 4).any(|other| i64::from(value) + 2 <= i64::from(other) || i64::from(other) + 2 <= i64::from(value))
            })
            .collect();
        assert_eq!(actual, expected);
    }
}
