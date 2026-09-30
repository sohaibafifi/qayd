use std::cell::Cell;

use crate::constraints::interval::{mandatory_no_overlap, no_overlap};
use crate::constraints::unary_edge_finding::{self, Membership, Snapshot, ThetaLambdaTree, UnaryEdgeFindingStats};
use crate::ids::{IntervalId, VarId};
use crate::store::Solver;

fn snapshot(index: usize, est: i128, duration: i128) -> Snapshot {
    Snapshot { interval: IntervalId(index as u32), start: VarId(index as u32), lower: 0, upper: 100, duration, est, lct: 100 }
}

fn ect(tasks: &[Snapshot], members: impl Iterator<Item = usize>) -> Option<i128> {
    let mut members = members.collect::<Vec<_>>();
    members.sort_unstable_by_key(|&index| (tasks[index].est, index));
    members
        .into_iter()
        .fold(None, |completion, index| Some(completion.unwrap_or(tasks[index].est).max(tasks[index].est) + tasks[index].duration))
}

#[test]
fn tree_matches_every_small_membership_and_responsible_witness() {
    for tasks in [
        vec![snapshot(0, -9, 2), snapshot(1, 0, 3), snapshot(2, 0, 1), snapshot(3, 11, 4), snapshot(4, 13, 2)],
        vec![snapshot(0, i128::from(i32::MIN) - 20, 10), snapshot(1, 0, 1), snapshot(2, i128::from(i32::MAX), 20)],
    ] {
        for encoding in 0..3usize.pow(tasks.len() as u32) {
            let mut stats = UnaryEdgeFindingStats::default();
            let mut tree = ThetaLambdaTree::new(&tasks, &mut stats, None).unwrap();
            let mut digits = encoding;
            for index in 0..tasks.len() {
                let membership = match digits % 3 {
                    0 => Membership::Empty,
                    1 => Membership::Theta,
                    _ => Membership::Lambda,
                };
                digits /= 3;
                assert!(tree.set(&tasks, index, membership, &mut stats, None));
            }
            let theta = || tree.membership.iter().enumerate().filter_map(|(index, member)| (*member == Membership::Theta).then_some(index));
            let lambda = tree
                .membership
                .iter()
                .enumerate()
                .filter_map(|(index, member)| (*member == Membership::Lambda).then_some(index))
                .collect::<Vec<_>>();
            let node = tree.root();
            assert_eq!(node.theta_duration, theta().map(|index| tasks[index].duration).sum::<i128>());
            assert_eq!(node.theta_ect, ect(&tasks, theta()));
            assert_eq!(
                node.lambda_ect.map(|entry| entry.value),
                lambda.iter().filter_map(|&index| ect(&tasks, theta().chain(std::iter::once(index)))).max()
            );
            assert_eq!(
                node.lambda_duration.map(|entry| entry.value),
                lambda.iter().map(|&index| node.theta_duration + tasks[index].duration).max()
            );
            if let Some(responsible) = node.lambda_ect {
                assert_eq!(tree.membership[responsible.task], Membership::Lambda);
                assert_eq!(Some(responsible.value), ect(&tasks, theta().chain(std::iter::once(responsible.task))));
            }
            if let Some(responsible) = node.lambda_duration {
                assert_eq!(tree.membership[responsible.task], Membership::Lambda);
                assert_eq!(responsible.value, node.theta_duration + tasks[responsible.task].duration);
            }
        }
    }
}

fn model(enabled: bool, bounds: &[(i32, i32, i32)]) -> (Solver, Vec<IntervalId>) {
    let mut solver = Solver::new();
    solver.set_unary_edge_finding(enabled);
    solver.set_unary_edge_finding_metrics(true);
    let intervals = bounds.iter().map(|&(lo, hi, duration)| solver.store.new_interval(lo, hi, duration)).collect::<Vec<_>>();
    mandatory_no_overlap(&mut solver, &intervals);
    (solver, intervals)
}

#[test]
fn edge_finding_is_strictly_stronger_than_detectable_precedence() {
    let bounds = [(0, 3, 3), (0, 3, 3), (0, 8, 2)];
    let (mut baseline, baseline_intervals) = model(false, &bounds);
    baseline.propagate().unwrap();
    assert_eq!(baseline.store.interval_start_min(baseline_intervals[2]), 0);
    let (mut stronger, intervals) = model(true, &bounds);
    stronger.propagate().unwrap();
    assert_eq!(stronger.store.interval_start_min(intervals[2]), 6);
    let stats = stronger.unary_edge_finding_stats();
    assert!(stats.bound_updates > 0);
    // The existing pairwise filter already removes unsupported interior values.
    // Count actual remaining starts eliminated by the additional global pass.
    let baseline_size = baseline.store.size(baseline.store.interval_start_var(baseline_intervals[2]));
    let final_size = stronger.store.size(stronger.store.interval_start_var(intervals[2]));
    assert_eq!(stats.eliminated_starts, (baseline_size - final_size) as u64);
    assert!(stats.reason_premises >= 5);
    assert_eq!(baseline.unary_edge_finding_stats().directional_passes, 0);
}

#[test]
fn mirror_prunes_upper_bounds_and_offsets_preserve_results() {
    for offset in [i32::MIN + 20, -10, 0, i32::MAX - 20] {
        let bounds = [(offset + 4, offset + 7, 3), (offset + 4, offset + 7, 3), (offset, offset + 8, 2)];
        let (mut solver, intervals) = model(true, &bounds);
        solver.propagate().unwrap();
        assert_eq!(solver.store.interval_start_max(intervals[2]), offset + 2);
        assert!(solver.unary_edge_finding_stats().bound_updates > 0);
    }
}

#[test]
fn equal_deadline_overload_is_detected_and_backtracking_rebuilds() {
    let (mut baseline, _) = model(false, &[(0, 5, 3); 3]);
    baseline.propagate().unwrap();
    let (mut overloaded, _) = model(true, &[(0, 5, 3); 3]);
    assert!(overloaded.propagate().is_err());
    assert_eq!(overloaded.unary_edge_finding_stats().overloads, 1);

    let (mut solver, intervals) = model(true, &[(0, 3, 3), (0, 3, 3), (0, 8, 2)]);
    solver.propagate().unwrap();
    for chosen in [0, 3, 0] {
        solver.store.push_level();
        solver.store.fix(solver.store.interval_start_var(intervals[0]), chosen).unwrap();
        solver.propagate().unwrap();
        assert_eq!(solver.store.interval_start_min(intervals[1]), 3 - chosen);
        solver.store.pop_level();
        solver.propagate().unwrap();
        assert_eq!(solver.store.interval_start_min(intervals[0]), 0);
        assert_eq!(solver.store.interval_start_max(intervals[0]), 3);
        assert_eq!(solver.store.interval_start_min(intervals[2]), 6);
    }
}

#[test]
fn unsupported_zero_optional_and_order_scopes_keep_old_path() {
    for shape in 0..3 {
        let mut solver = Solver::new();
        solver.set_unary_edge_finding(true);
        solver.set_unary_edge_finding_metrics(true);
        let mut intervals = vec![solver.store.new_interval(0, 3, 3), solver.store.new_interval(0, 3, 3)];
        intervals.push(match shape {
            0 => solver.store.new_interval(0, 8, 0),
            1 => solver.store.new_optional_interval(0, 8, 2),
            _ => solver.store.new_interval(0, 8, 2),
        });
        if shape == 2 {
            no_overlap(&mut solver, &intervals);
        } else {
            mandatory_no_overlap(&mut solver, &intervals);
        }
        solver.propagate().unwrap();
        let stats = solver.unary_edge_finding_stats();
        assert!(stats.unsupported_calls > 0);
        assert_eq!(stats.directional_passes, 0);
    }
}

#[test]
fn interruption_discards_partial_trees_and_never_posts_partial_reasons() {
    for stop_at in 0..100 {
        let mut solver = Solver::new();
        let intervals =
            [(0, 3, 3), (0, 3, 3), (0, 8, 2)].into_iter().map(|(lo, hi, p)| solver.store.new_interval(lo, hi, p)).collect::<Vec<_>>();
        let visits = Cell::new(0usize);
        let stop = || {
            let before = visits.get();
            visits.set(before + 1);
            before >= stop_at
        };
        let result = unary_edge_finding::propagate(&mut solver.store, &intervals, Some(&stop)).unwrap();
        let lower = solver.store.interval_start_min(intervals[2]);
        assert!(lower == 0 || lower == 6);
        assert_eq!(solver.store.interval_start_max(intervals[2]), 8);
        if result.is_some() {
            assert_eq!(lower, 6);
        }
        // Resume with fresh snapshots, regardless of where tree construction or
        // reason generation was interrupted.
        unary_edge_finding::propagate(&mut solver.store, &intervals, None).unwrap().unwrap();
        assert_eq!(solver.store.interval_start_min(intervals[2]), 6);
    }
}

#[test]
fn mathematical_ends_beyond_i32_are_not_saturated_in_tree() {
    let mut solver = Solver::new();
    let intervals = vec![solver.store.new_interval(i32::MAX - 2, i32::MAX, 2), solver.store.new_interval(i32::MAX - 2, i32::MAX, 2)];
    assert_eq!(unary_edge_finding::propagate(&mut solver.store, &intervals, None).unwrap(), Some(false));
    let extra = solver.store.new_interval(i32::MAX - 2, i32::MAX, 2);
    let all = vec![intervals[0], intervals[1], extra];
    assert!(unary_edge_finding::propagate(&mut solver.store, &all, None).is_err());
}
