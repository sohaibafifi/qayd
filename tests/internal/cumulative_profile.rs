use std::cell::Cell;
use std::collections::BTreeSet;

use crate::constraints::cumulative_profile::{CompulsoryPart, CumulativeStats, IncrementalProfile, ProfileUpdate};
use crate::constraints::interval::cumulative;
use crate::constraints::resource_profile::{build_profile, ProfileSegment};
use crate::store::{Solver, Store};

fn oracle(parts: &[Option<CompulsoryPart>]) -> Vec<ProfileSegment> {
    let parts = parts.iter().flatten().copied().filter(|&(lo, hi, demand)| lo < hi && demand != 0).collect::<Vec<_>>();
    let endpoints = parts.iter().flat_map(|&(lo, hi, _)| [lo, hi]).collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
    endpoints
        .windows(2)
        .filter_map(|window| {
            let usage = parts.iter().filter(|&&(lo, hi, _)| lo <= window[0] && window[0] < hi).map(|&(_, _, demand)| demand).sum();
            (usage != 0).then_some(ProfileSegment { start: window[0], end: window[1], usage })
        })
        .collect()
}

fn check(store: &mut Store, cache: &mut IncrementalProfile, parts: &[Option<CompulsoryPart>], stats: &mut CumulativeStats) {
    assert_eq!(cache.update(store, parts.iter().copied(), stats, None), ProfileUpdate::Ready);
    let mut profile = Vec::new();
    assert_eq!(cache.materialize(store, &mut profile, stats, None), ProfileUpdate::Ready);
    assert_eq!(profile, oracle(parts));
    let mut fresh = Vec::new();
    build_profile(parts.iter().flatten().copied(), &mut Vec::new(), &mut fresh);
    assert_eq!(profile, fresh);
}

#[test]
fn counted_zero_net_endpoints_preserve_fresh_profile_segmentation() {
    let mut store = Store::new();
    let mut cache = IncrementalProfile::new(&mut store, 4).unwrap();
    let mut stats = CumulativeStats::default();
    let mut parts = [Some((0, 2, 3)), Some((2, 4, 3)), Some((0, 4, 1)), None];
    check(&mut store, &mut cache, &parts, &mut stats);
    assert_eq!(oracle(&parts), [ProfileSegment { start: 0, end: 2, usage: 4 }, ProfileSegment { start: 2, end: 4, usage: 4 }]);
    parts[0] = None;
    check(&mut store, &mut cache, &parts, &mut stats);
    parts[1] = None;
    check(&mut store, &mut cache, &parts, &mut stats);
    assert_eq!(oracle(&parts), [ProfileSegment { start: 0, end: 4, usage: 1 }]);
    assert_eq!(stats.undo_records, 0, "root writes must not accumulate undo payloads");
}

#[test]
fn exact_profiles_cover_empty_repeated_zero_and_extreme_parts() {
    let mut store = Store::new();
    let mut cache = IncrementalProfile::new(&mut store, 6).unwrap();
    let mut stats = CumulativeStats::default();
    for parts in [
        [None; 6],
        [Some((0, 0, 3)), Some((4, 2, 2)), Some((0, 5, 0)), None, None, None],
        [Some((-4, 7, 2)), Some((-4, 7, 2)), Some((7, 9, 4)), None, None, None],
        [
            Some((i128::MIN, i128::MIN + 1, i128::from(i64::MAX))),
            Some((i128::MIN, 0, 3)),
            Some((0, i128::MAX, 2)),
            Some((i128::MAX - 1, i128::MAX, 9)),
            None,
            Some((0, 1, 0)),
        ],
    ] {
        check(&mut store, &mut cache, &parts, &mut stats);
    }
}

#[test]
fn nested_and_sibling_backtracking_undo_occurrences_before_reading() {
    let mut store = Store::new();
    let mut cache = IncrementalProfile::new(&mut store, 3).unwrap();
    let mut stats = CumulativeStats::default();
    let root = [Some((0, 2, 2)), Some((2, 4, 2)), None];
    let left = [Some((0, 3, 2)), Some((1, 4, 2)), Some((9, 12, 1))];
    let child = [None, Some((-4, 5, 2)), Some((10, 13, 1))];
    let right = [Some((5, 8, 2)), None, Some((-9, -7, 1))];
    check(&mut store, &mut cache, &root, &mut stats);
    store.push_level();
    check(&mut store, &mut cache, &left, &mut stats);
    store.push_level();
    check(&mut store, &mut cache, &child, &mut stats);
    store.pop_level();
    let mut stale = vec![ProfileSegment { start: 0, end: 1, usage: 99 }];
    assert_eq!(cache.materialize(&store, &mut stale, &mut stats, None), ProfileUpdate::Fallback);
    assert!(stale.is_empty());
    check(&mut store, &mut cache, &left, &mut stats);
    store.pop_level();
    store.push_level();
    check(&mut store, &mut cache, &right, &mut stats);
    store.pop_level();
    check(&mut store, &mut cache, &root, &mut stats);
    assert!(stats.restored_tasks >= 9);
    assert!(stats.peak_undo_records >= 6);
}

#[test]
fn cloned_cache_and_store_restore_independently_from_nested_branches() {
    let mut store = Store::new();
    let mut cache = IncrementalProfile::new(&mut store, 2).unwrap();
    let mut stats = CumulativeStats::default();
    let root = [Some((0, 2, 1)), None];
    let left = [Some((0, 4, 1)), Some((2, 5, 2))];
    check(&mut store, &mut cache, &root, &mut stats);
    store.push_level();
    check(&mut store, &mut cache, &left, &mut stats);
    let mut other_store = store.clone();
    let mut other_cache = cache.clone();
    store.push_level();
    check(&mut store, &mut cache, &[None, Some((-5, 5, 2))], &mut stats);
    other_store.pop_level();
    check(&mut other_store, &mut other_cache, &root, &mut stats);
    other_store.push_level();
    check(&mut other_store, &mut other_cache, &[Some((20, 24, 1)), None], &mut stats);
    store.pop_level();
    check(&mut store, &mut cache, &left, &mut stats);
    store.pop_level();
    check(&mut store, &mut cache, &root, &mut stats);
}

#[test]
fn cap_freezes_without_losing_ancestor_undo_and_recovers_at_root() {
    let mut store = Store::new();
    let mut cache = IncrementalProfile::with_log_limit(&mut store, 2, 3).unwrap();
    let mut stats = CumulativeStats::default();
    let root = [Some((0, 2, 1)), None];
    let first = [Some((0, 4, 1)), Some((2, 5, 2))];
    let second = [Some((10, 14, 1)), None];
    check(&mut store, &mut cache, &root, &mut stats);
    store.push_level();
    check(&mut store, &mut cache, &first, &mut stats);
    assert_eq!(stats.undo_records, 2, "both occurrences changed in the first branch");
    store.push_level();
    assert_eq!(cache.update(&mut store, second.into_iter(), &mut stats, None), ProfileUpdate::Fallback);
    assert_eq!(stats.peak_undo_records, 3);
    assert_eq!(stats.undo_records, 3, "the cap prevents the second occurrence's child update");
    let mut profile = Vec::new();
    assert_eq!(cache.materialize(&store, &mut profile, &mut stats, None), ProfileUpdate::Fallback);
    store.pop_level();
    assert_eq!(cache.update(&mut store, first.into_iter(), &mut stats, None), ProfileUpdate::Fallback);
    assert_eq!(stats.restored_tasks, 1, "only the applied child update is restored");
    store.pop_level();
    check(&mut store, &mut cache, &root, &mut stats);
    assert_eq!(stats.restored_tasks, 3, "both preserved ancestor updates are restored at root");
    store.push_level();
    check(&mut store, &mut cache, &second, &mut stats);
    // Occurrence 1 is None in both root and second, so this fresh branch
    // changes and logs only occurrence 0.
    assert_eq!(stats.undo_records, 4);
    store.pop_level();
    check(&mut store, &mut cache, &root, &mut stats);
    assert_eq!(stats.restored_tasks, 4);
}

#[test]
fn interrupted_atomic_updates_can_resume_or_undo_to_an_ancestor() {
    let original = [Some((0, 2, 1)), Some((2, 4, 1)), None, Some((-8, -3, 2))];
    let changed = [Some((0, 4, 1)), None, Some((5, 8, 1)), Some((-9, -1, 2))];
    for after in 0..8 {
        for resume in [false, true] {
            let mut store = Store::new();
            let mut cache = IncrementalProfile::new(&mut store, 4).unwrap();
            let mut stats = CumulativeStats::default();
            check(&mut store, &mut cache, &original, &mut stats);
            store.push_level();
            let calls = Cell::new(0);
            let stop = || {
                let previous = calls.get();
                calls.set(previous + 1);
                previous >= after
            };
            let outcome = cache.update(&mut store, changed.into_iter(), &mut stats, Some(&stop));
            assert!(matches!(outcome, ProfileUpdate::Ready | ProfileUpdate::Interrupted));
            if resume {
                check(&mut store, &mut cache, &changed, &mut stats);
            }
            store.pop_level();
            check(&mut store, &mut cache, &original, &mut stats);
        }
    }
}

#[test]
fn interrupted_undo_keeps_remaining_ancestor_records() {
    let root = [Some((0, 2, 1)); 4];
    let first = [Some((1, 4, 1)); 4];
    let second = [Some((2, 6, 1)); 4];
    for after in 0..10 {
        let mut store = Store::new();
        let mut cache = IncrementalProfile::new(&mut store, 4).unwrap();
        let mut stats = CumulativeStats::default();
        check(&mut store, &mut cache, &root, &mut stats);
        store.push_level();
        check(&mut store, &mut cache, &first, &mut stats);
        store.push_level();
        check(&mut store, &mut cache, &second, &mut stats);
        store.pop_level();
        let calls = Cell::new(0);
        let stop = || {
            let previous = calls.get();
            calls.set(previous + 1);
            previous >= after
        };
        let result = cache.update(&mut store, first.into_iter(), &mut stats, Some(&stop));
        assert!(matches!(result, ProfileUpdate::Ready | ProfileUpdate::Interrupted));
        // A second pop can occur before interrupted restoration finishes.
        store.pop_level();
        check(&mut store, &mut cache, &root, &mut stats);
        assert_eq!(stats.restored_tasks, 8);
    }
}

#[test]
fn interrupted_materialization_clears_output_and_preserves_reusable_cache() {
    let parts = [Some((-5, 2, 1)), Some((0, 8, 3)), Some((2, 4, 2))];
    for after in 0..9 {
        let mut store = Store::new();
        let mut cache = IncrementalProfile::new(&mut store, parts.len()).unwrap();
        let mut stats = CumulativeStats::default();
        check(&mut store, &mut cache, &parts, &mut stats);
        let calls = Cell::new(0);
        let stop = || {
            let previous = calls.get();
            calls.set(previous + 1);
            previous >= after
        };
        let mut profile = Vec::new();
        let result = cache.materialize(&store, &mut profile, &mut stats, Some(&stop));
        if result == ProfileUpdate::Interrupted {
            assert!(profile.is_empty());
        } else {
            assert_eq!(result, ProfileUpdate::Ready);
            assert_eq!(profile, oracle(&parts));
        }
        assert_eq!(cache.materialize(&store, &mut profile, &mut stats, None), ProfileUpdate::Ready);
        assert_eq!(profile, oracle(&parts));
    }
}

#[test]
fn deterministic_nested_traces_match_fresh_builds_after_every_step() {
    for seed in [0u64, 1, 7] {
        let mut store = Store::new();
        let mut cache = IncrementalProfile::new(&mut store, 16).unwrap();
        let mut stats = CumulativeStats::default();
        let mut state = seed + 37;
        let mut parts = vec![None; 16];
        let mut ancestors = Vec::new();
        for step in 0..300 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            match step % 13 {
                1 | 5 | 8 => {
                    store.push_level();
                    ancestors.push(parts.clone());
                }
                3 | 9 if !ancestors.is_empty() => {
                    store.pop_level();
                    parts = ancestors.pop().unwrap();
                }
                _ => {
                    let occurrence = (state as usize) % parts.len();
                    let start = i128::from((state >> 32) as i32) * 1_000_000;
                    parts[occurrence] = if state & 8 == 0 { None } else { Some((start, start + 5, 1 + i128::from(state % 4))) };
                }
            }
            check(&mut store, &mut cache, &parts, &mut stats);
        }
        while let Some(parent) = ancestors.pop() {
            store.pop_level();
            parts = parent;
        }
        check(&mut store, &mut cache, &parts, &mut stats);
    }
}

#[test]
fn cursor_allocation_is_root_only_and_unchanged_tasks_do_not_append_undo() {
    let mut store = Store::new();
    store.push_level();
    assert!(IncrementalProfile::new(&mut store, 2).is_none());
    store.pop_level();
    let mut cache = IncrementalProfile::with_log_limit(&mut store, 2, 1).unwrap();
    let mut stats = CumulativeStats::default();
    let parts = [Some((0, 2, 1)), None];
    for _ in 0..100 {
        check(&mut store, &mut cache, &parts, &mut stats);
    }
    store.push_level();
    for _ in 0..100 {
        check(&mut store, &mut cache, &parts, &mut stats);
    }
    assert_eq!(stats.changed_tasks, 1);
    assert_eq!(stats.undo_records, 0);
    assert_eq!(stats.fallback_calls, 0);
}

#[test]
fn native_adapter_can_enable_at_root_and_falls_back_for_nonroot_first_use() {
    let mut solver = Solver::new();
    solver.set_cumulative_metrics(true);
    let intervals = [solver.store.new_interval(0, 2, 3), solver.store.new_optional_interval(-2, 2, 2)];
    cumulative(&mut solver, &intervals, &[1, 2], 4);
    solver.propagate().unwrap();
    assert!(solver.cumulative_stats().profile_rebuilds > 0);
    assert_eq!(solver.cumulative_stats().incremental_updates, 0);
    solver.set_incremental_cumulative(true);
    // Enabling is a root action, but no propagation has yet allocated a cache.
    solver.store.push_level();
    solver.propagate().unwrap();
    assert!(solver.cumulative_stats().fallback_calls > 0);
    assert_eq!(solver.cumulative_stats().incremental_updates, 0);
    solver.store.pop_level();
    solver.enqueue_all();
    solver.propagate().unwrap();
    assert!(solver.cumulative_stats().incremental_updates > 0);
    let before = solver.cumulative_stats().restored_tasks;
    solver.store.push_level();
    solver.store.set_interval_start_min(intervals[0], 1).unwrap();
    solver.propagate().unwrap();
    solver.store.pop_level();
    solver.set_incremental_cumulative(false);
    solver.propagate().unwrap();
    solver.set_incremental_cumulative(true);
    solver.propagate().unwrap();
    assert!(solver.cumulative_stats().restored_tasks > before);
}

#[test]
fn constraints_registered_nonroot_keep_the_fresh_profile_fallback() {
    let mut solver = Solver::new();
    solver.set_incremental_cumulative(true);
    solver.set_cumulative_metrics(true);
    let interval = solver.store.new_interval(0, 2, 3);
    solver.store.push_level();
    cumulative(&mut solver, &[interval], &[1], 2);
    solver.propagate().unwrap();
    assert_eq!(solver.cumulative_stats().incremental_updates, 0);
    assert!(solver.cumulative_stats().fallback_calls > 0);
    solver.store.pop_level();
    solver.enqueue_all();
    solver.propagate().unwrap();
    assert_eq!(solver.cumulative_stats().incremental_updates, 0);
}

fn inference_domains(store: &Store) -> Vec<crate::inference::InferenceDomain> {
    (0..store.num_vars())
        .map(|index| {
            let var = crate::ids::VarId(index as u32);
            crate::inference::InferenceDomain {
                var,
                min: store.min(var),
                max: store.max(var),
                values: Some(std::sync::Arc::from(store.values(var).collect::<Vec<_>>())),
                holes: Vec::new(),
            }
        })
        .collect()
}

fn check_lifted(
    store: &Store,
    roots: &[crate::inference::InferenceDomain],
    intervals: &[crate::ids::IntervalId],
    demands: &[i32],
    capacity: i32,
    premises: &[crate::store::Premise],
    conclusion: crate::inference::InferenceConclusion,
) -> crate::inference::InferenceCheckOutcome {
    let tasks = intervals
        .iter()
        .zip(demands)
        .map(|(&interval, &demand)| crate::inference::InferenceCumulativeTask {
            start: store.interval_start_var(interval),
            duration: i64::from(store.interval_duration(interval)),
            demand: i64::from(demand),
            presence: store.interval_presence_var(interval),
        })
        .collect();
    let constraint = crate::inference::InferenceConstraint::Cumulative { tasks, capacity: i64::from(capacity) };
    crate::inference::check_inference(
        &crate::inference::Inference {
            propagator: crate::ids::PropId(0),
            constraint: &constraint,
            root_domains: roots,
            premises,
            conclusion,
            snapshot: &inference_domains(store),
        },
        crate::inference::InferenceCheckConfig::default(),
    )
}

fn current_profile(store: &Store, intervals: &[crate::ids::IntervalId], demands: &[i32]) -> Vec<ProfileSegment> {
    let mut profile = Vec::new();
    build_profile(
        intervals.iter().zip(demands).filter_map(|(&interval, &demand)| {
            (store.interval_presence(interval) == crate::domains::interval::IntervalPresence::Present).then_some((
                i128::from(store.interval_start_max(interval)),
                i128::from(store.interval_start_min(interval)) + i128::from(store.interval_duration(interval)),
                i128::from(demand),
            ))
        }),
        &mut Vec::new(),
        &mut profile,
    );
    profile
}

#[test]
fn lifted_multisegment_certificate_weakens_both_contributor_bounds() {
    use crate::constraints::interval::{CumulativeReasonContext, TimetableCertificate};
    use crate::inference::{InferenceCheckOutcome, InferenceConclusion};
    use crate::store::Premise;
    let mut store = Store::new();
    // B precedes A in occurrence order so their overlapping segment is
    // certified by B. A then needs to cover only the first narrow window.
    let b = store.new_interval(-3, 12, 8);
    let a = store.new_interval(-3, 12, 4);
    let target = store.new_interval(-3, 12, 2);
    let roots = inference_domains(&store);
    store.set_interval_start_min(b, 2).unwrap();
    store.set_interval_start_max(b, 2).unwrap();
    store.set_interval_start_min(a, 0).unwrap();
    store.set_interval_start_max(a, 0).unwrap();
    store.set_interval_start_min(target, 0).unwrap();
    let intervals = [b, a, target];
    let demands = [2, 1, 3];
    let profile = current_profile(&store, &intervals, &demands);
    let context = CumulativeReasonContext { store: &store, intervals: &intervals, demands: &demands, capacity: 3, profile: &profile };
    let reason = context.explain(TimetableCertificate::LowerBound { occurrence: 2, bound: 10 }, None).unwrap().unwrap();
    let a_var = store.interval_start_var(a);
    assert!(reason.premises.contains(&Premise::Ge { var: a_var, bound: -2 }));
    assert!(reason.premises.contains(&Premise::Le { var: a_var, bound: 1 }));
    assert!(reason.premises.contains(&Premise::Le { var: store.interval_start_var(b), bound: 3 }));
    assert!(reason.lifted_bounds >= 3);
    assert!(matches!(
        check_lifted(
            &store,
            &roots,
            &intervals,
            &demands,
            3,
            &reason.premises,
            InferenceConclusion::Predicate(Premise::Ge { var: store.interval_start_var(target), bound: 10 })
        ),
        InferenceCheckOutcome::Checked { .. }
    ));
}

#[test]
fn lifted_native_timetable_uses_checked_reasons_and_keeps_same_pruning() {
    for incremental in [false, true] {
        for lifting in [false, true] {
            let mut solver = Solver::new();
            solver.set_incremental_cumulative(incremental);
            solver.set_lifted_cumulative_explanations(lifting);
            solver.set_cumulative_metrics(true);
            let b = solver.store.new_interval(2, 2, 8);
            let a = solver.store.new_interval(0, 0, 4);
            let target = solver.store.new_interval(0, 12, 2);
            solver.set_inference_checking(Some(crate::inference::InferenceCheckConfig::default()));
            solver.store.set_explain(true);
            cumulative(&mut solver, &[b, a, target], &[2, 1, 3], 3);
            solver.propagate().unwrap();
            assert_eq!(solver.store.interval_start_min(target), 10);
            let stats = solver.cumulative_stats();
            assert!(stats.timetable_updates > 0 && stats.reason_calls > 0);
            assert!(solver.inference_check_stats().checked_inferences > 0);
            assert_eq!(solver.inference_check_stats().uncovered_inferences, 0);
            if lifting {
                assert!(stats.lifted_bounds >= 3);
                assert!(stats.emitted_premises <= stats.original_premises);
                let a_var = solver.store.interval_start_var(a);
                assert!(solver.store.events.iter().any(|event| match &event.cause {
                    crate::store::Cause::Premises(premises) => premises.contains(&crate::store::Premise::Ge { var: a_var, bound: -2 }),
                    _ => false,
                }));
            } else {
                assert_eq!(stats.lifted_bounds, 0);
                assert_eq!(stats.omitted_tautologies, 0);
                assert_eq!(stats.original_premises, stats.emitted_premises);
            }
        }
    }
}

#[test]
fn lifted_conflicts_keep_optional_presence_and_all_repeated_occurrences() {
    use crate::constraints::interval::{CumulativeReasonContext, TimetableCertificate};
    use crate::inference::{InferenceCheckOutcome, InferenceConclusion};
    let mut store = Store::new();
    let repeated = store.new_optional_interval(-2, 2, 3);
    let roots = inference_domains(&store);
    store.require_interval_presence(repeated).unwrap();
    store.set_interval_start_min(repeated, 0).unwrap();
    store.set_interval_start_max(repeated, 1).unwrap();
    let intervals = [repeated, repeated];
    let demands = [2, 2];
    let profile = current_profile(&store, &intervals, &demands);
    let context = CumulativeReasonContext { store: &store, intervals: &intervals, demands: &demands, capacity: 3, profile: &profile };
    for certificate in [TimetableCertificate::Overload { time: 1 }, TimetableCertificate::NoFit { occurrence: 0 }] {
        let reason = context.explain(certificate, None).unwrap().unwrap();
        assert!(matches!(
            check_lifted(&store, &roots, &intervals, &demands, 3, &reason.premises, InferenceConclusion::Conflict),
            InferenceCheckOutcome::Checked { .. }
        ));
        let presence = store.interval_presence_var(repeated).unwrap();
        assert!(reason.premises.iter().any(|premise| matches!(premise, crate::store::Premise::Eq { var, val: 1 } if *var == presence)));
        let without_presence = reason
            .premises
            .iter()
            .copied()
            .filter(|premise| {
                !matches!(premise,
            crate::store::Premise::Eq { var, .. } if *var == presence)
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            check_lifted(&store, &roots, &intervals, &demands, 3, &without_presence, InferenceConclusion::Conflict),
            InferenceCheckOutcome::Rejected(_)
        ));
    }
}

#[test]
fn nofit_lifting_weakens_bounds_but_falls_back_when_clipped_window_lacks_coverage() {
    use crate::constraints::interval::{CumulativeReasonContext, TimetableCertificate};
    use crate::inference::{InferenceCheckOutcome, InferenceConclusion};
    use crate::store::Premise;
    let mut store = Store::new();
    let blocker = store.new_interval(-7, 5, 10);
    let target = store.new_interval(0, 2, 2);
    let roots = inference_domains(&store);
    store.set_interval_start_min(blocker, 0).unwrap();
    store.set_interval_start_max(blocker, 0).unwrap();
    let intervals = [blocker, target];
    let demands = [1, 1];
    let profile = current_profile(&store, &intervals, &demands);
    let context = CumulativeReasonContext { store: &store, intervals: &intervals, demands: &demands, capacity: 1, profile: &profile };
    let reason = context.explain(TimetableCertificate::NoFit { occurrence: 1 }, None).unwrap().unwrap();
    assert!(reason.premises.contains(&Premise::Ge { var: store.interval_start_var(blocker), bound: -7 }));
    assert!(reason.premises.contains(&Premise::Le { var: store.interval_start_var(blocker), bound: 1 }));
    assert!(matches!(
        check_lifted(&store, &roots, &intervals, &demands, 1, &reason.premises, InferenceConclusion::Conflict),
        InferenceCheckOutcome::Checked { .. }
    ));

    let mut store = Store::new();
    let target = store.new_interval(0, 2, 10);
    let blocker = store.new_interval(5, 5, 3);
    let intervals = [target, blocker];
    let profile = current_profile(&store, &intervals, &demands);
    let context = CumulativeReasonContext { store: &store, intervals: &intervals, demands: &demands, capacity: 1, profile: &profile };
    // The clipped [2,3) window precedes blocker [5,8). The builder must not
    // certify it merely because the unshortened occupation does overload.
    assert!(context.explain(TimetableCertificate::NoFit { occurrence: 0 }, None).unwrap().is_none());
}

#[test]
fn lifted_endpoint_tautologies_are_omitted_with_widened_arithmetic() {
    use crate::constraints::interval::{CumulativeReasonContext, TimetableCertificate};
    use crate::inference::{InferenceCheckOutcome, InferenceConclusion};
    for origin in [i32::MIN, i32::MAX - 2] {
        let mut store = Store::new();
        let first = store.new_interval(origin, origin + 2, 5);
        let second = store.new_interval(origin, origin + 2, 5);
        let roots = inference_domains(&store);
        let intervals = [first, second];
        let demands = [1, 1];
        let profile = current_profile(&store, &intervals, &demands);
        let time = if origin == i32::MIN { i128::from(origin) + 2 } else { i128::from(i32::MAX) + 1 };
        let context = CumulativeReasonContext { store: &store, intervals: &intervals, demands: &demands, capacity: 1, profile: &profile };
        let reason = context.explain(TimetableCertificate::Overload { time }, None).unwrap().unwrap();
        assert!(reason.omitted_tautologies >= 2);
        assert!(matches!(
            check_lifted(&store, &roots, &intervals, &demands, 1, &reason.premises, InferenceConclusion::Conflict),
            InferenceCheckOutcome::Checked { .. }
        ));
    }
}

#[test]
fn lifted_certificate_interruptions_and_gaps_never_emit_partial_reasons() {
    use crate::constraints::interval::{CumulativeReasonContext, TimetableCertificate};
    let mut store = Store::new();
    let a = store.new_interval(0, 0, 2);
    let b = store.new_interval(5, 5, 2);
    let target = store.new_interval(0, 10, 1);
    let intervals = [a, b, target];
    let demands = [1, 1, 1];
    let profile = current_profile(&store, &intervals, &demands);
    let context = CumulativeReasonContext { store: &store, intervals: &intervals, demands: &demands, capacity: 1, profile: &profile };
    // Starts 2,3,4 fit; a purported bound7 cannot be explained by joining the
    // two disjoint overload regions and ignoring the feasible gap.
    assert!(context.explain(TimetableCertificate::LowerBound { occurrence: 2, bound: 7 }, None).unwrap().is_none());
    for after in 0..20 {
        let calls = Cell::new(0);
        let stop = || {
            let previous = calls.get();
            calls.set(previous + 1);
            previous >= after
        };
        match context.explain(TimetableCertificate::LowerBound { occurrence: 2, bound: 2 }, Some(&stop)) {
            Err(()) => {}
            Ok(Some(reason)) => assert!(!reason.premises.is_empty()),
            Ok(None) => panic!("the complete bound2 certificate is valid"),
        }
    }
}
