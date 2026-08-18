use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use qayd::engines::ls::lists::move_acceptance::MinimizingMoveAcceptance;
use qayd::engines::ls::lists::resource_schedule::{
    audit_event_profile_fit, GenerationScheme, Justification, PriorityRule, PrioritySgs, ProfileFit, ResourceAlnsBudget,
    ResourceMoveOutcome, ResourceMoveRejection, ResourceScheduleMove, ResourceScheduleProblem, ResourceScheduleState,
};
use qayd::model::list::{verify_collection_solution, CollectionModel, IntervalVar, Mode, Resource, Schedule};

fn collection(schedule: Schedule) -> CollectionModel {
    CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule),
    }
}

fn rcpsp() -> Schedule {
    let durations = [2, 2, 1, 1];
    let demands = [1, 2, 1, 2];
    Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon: 8, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 2), (1, 3)],
        resources: vec![Resource::Cumulative { demands: demands.into_iter().enumerate().collect(), capacity: 2 }],
        minimize_makespan: true,
    }
}

fn assert_verified(schedule: Schedule, state: &ResourceScheduleState) {
    assert_eq!(verify_collection_solution(&collection(schedule), &state.to_solution()).unwrap(), vec![state.makespan()]);
}

fn dense_profile_fit(capacity: i128, bookings: &[(i64, i64, i128)], start: i64, end: i64, demand: i128) -> ProfileFit {
    if demand == 0 || start == end {
        return ProfileFit::Fits;
    }
    if demand > capacity {
        return ProfileFit::Impossible;
    }

    let mut boundaries = bookings.iter().flat_map(|&(booking_start, booking_end, _)| [booking_start, booking_end]).collect::<Vec<_>>();
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries.retain(|&boundary| {
        let delta = bookings.iter().fold(0i128, |delta, &(booking_start, booking_end, booking_demand)| {
            delta + i128::from(booking_start == boundary) * booking_demand - i128::from(booking_end == boundary) * booking_demand
        });
        delta != 0
    });

    for time in start..end {
        let Some(usage) = bookings
            .iter()
            .filter(|&&(booking_start, booking_end, _)| booking_start <= time && time < booking_end)
            .try_fold(0i128, |usage, &(_, _, booking_demand)| usage.checked_add(booking_demand))
        else {
            return ProfileFit::Impossible;
        };
        let Some(with_candidate) = usage.checked_add(demand) else {
            return ProfileFit::Impossible;
        };
        if with_candidate > capacity {
            return boundaries.iter().copied().find(|&boundary| boundary > time).map_or(ProfileFit::Impossible, ProfileFit::RetryAt);
        }
    }
    ProfileFit::Fits
}

fn assert_profile_fit(
    capacity: i128,
    bookings: &[(i64, i64, i128)],
    start: i64,
    end: i64,
    demand: i128,
    expected: ProfileFit,
) -> [(u64, u64); 2] {
    let stop = AtomicBool::new(false);
    let audit = audit_event_profile_fit(capacity, bookings, start, end, demand, &stop).expect("valid profile bookings");
    let mut metrics = [(0, 0); 2];
    for (index, audit) in audit.into_iter().enumerate() {
        assert_eq!(audit.result.unwrap(), expected, "profile backend {index}");
        metrics[index] = (audit.profile_checks, audit.event_visits);
    }
    assert_eq!(metrics[0], metrics[1]);
    metrics
}

#[test]
fn event_profile_fit_preserves_half_open_boundaries_and_retry_events() {
    assert_eq!(assert_profile_fit(2, &[(0, 2, 2)], 2, 4, 2, ProfileFit::Fits), [(1, 2), (1, 2)]);
    assert_eq!(assert_profile_fit(2, &[(2, 4, 2)], 0, 2, 2, ProfileFit::Fits), [(1, 1), (1, 1)]);
    assert_eq!(assert_profile_fit(2, &[(0, 2, 2)], 1, 3, 1, ProfileFit::RetryAt(2)), [(1, 1), (1, 1)]);
    let metrics = assert_profile_fit(2, &[(0, 2, 2), (2, 4, 2)], 1, 3, 1, ProfileFit::RetryAt(4));
    assert_eq!(metrics, [(1, 1), (1, 1)], "the zero net event at the touching boundary must remain absent");
    assert_eq!(assert_profile_fit(2, &[(0, 2, 2)], 0, 2, 0, ProfileFit::Fits), [(1, 0), (1, 0)]);
    assert_eq!(assert_profile_fit(2, &[(0, 2, 2)], 1, 1, i128::MAX, ProfileFit::Fits), [(1, 0), (1, 0)]);
}

#[test]
fn both_event_profiles_match_a_dense_randomized_fit_oracle() {
    let mut random = 0x92d6_8ca2_7b31_4f05u64;
    for case in 0..1_000 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let raw_capacity = random % 8;
        let capacity = i128::from(raw_capacity);
        let booking_count = usize::try_from((random >> 8) % 9).unwrap();
        let mut bookings = Vec::with_capacity(booking_count);
        for _ in 0..booking_count {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let booking_start = i64::try_from(random % 12).unwrap() - 2;
            let booking_end = booking_start + i64::try_from(1 + (random >> 8) % 5).unwrap();
            let booking_demand = i128::from((random >> 16) % (raw_capacity + 2));
            bookings.push((booking_start, booking_end, booking_demand));
        }
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let start = i64::try_from(random % 14).unwrap() - 2;
        let end = start + i64::try_from((random >> 8) % 6).unwrap();
        let demand = i128::from((random >> 16) % (raw_capacity + 2));
        let expected = dense_profile_fit(capacity, &bookings, start, end, demand);
        let metrics = assert_profile_fit(capacity, &bookings, start, end, demand, expected);
        assert_eq!(metrics[0].0, 1, "case {case}");
    }
}

#[test]
fn event_profile_fit_preserves_numeric_overflow_as_impossible() {
    let candidate_overflow = assert_profile_fit(i128::MAX, &[(0, 10, i128::MAX)], 1, 2, 1, ProfileFit::Impossible);
    assert_eq!(candidate_overflow, [(1, 1), (1, 1)]);

    let usage_overflow = assert_profile_fit(i128::MAX, &[(0, 10, i128::MAX), (1, 9, 1)], 2, 3, 1, ProfileFit::Impossible);
    assert_eq!(usage_overflow, [(1, 2), (1, 2)]);
}

#[test]
fn event_profile_fit_honors_a_prearmed_stop_before_metrics() {
    let stop = AtomicBool::new(true);
    let audit = audit_event_profile_fit(2, &[(0, 2, 1)], 0, 2, 1, &stop).expect("valid profile bookings");
    for audit in audit {
        assert!(audit.result.is_err());
        assert_eq!(audit.profile_checks, 0);
        assert_eq!(audit.event_visits, 0);
    }
}

#[test]
fn recognizes_only_mandatory_fixed_duration_cumulative_schedules() {
    let stop = AtomicBool::new(false);
    let problem = ResourceScheduleProblem::recognize(&rcpsp(), &stop).unwrap().expect("strict RCPSP");
    assert_eq!(problem.activity_count(), 4);
    assert_eq!(problem.resource_count(), 1);
    assert_eq!(problem.topological_order(), &[0, 1, 2, 3]);
    assert_eq!((0..4).map(|activity| problem.duration(activity)).collect::<Vec<_>>(), vec![2, 2, 1, 1]);
    assert_eq!((0..4).map(|activity| problem.latest_start(activity)).collect::<Vec<_>>(), vec![6, 6, 7, 7]);
    assert_eq!((0..4).map(|activity| problem.demand(0, activity)).collect::<Vec<_>>(), vec![1, 2, 1, 2]);
    assert_eq!(problem.capacity(0), 2);

    let mut optional = rcpsp();
    optional.intervals[0].optional = true;
    assert!(ResourceScheduleProblem::recognize(&optional, &stop).unwrap().is_none());

    let mut multi_mode = rcpsp();
    multi_mode.intervals[0].modes.extend([
        Mode { reference: Some(0), machine: 0, duration: 2, start_window: (0, 6) },
        Mode { reference: Some(1), machine: 1, duration: 1, start_window: (0, 7) },
    ]);
    assert!(ResourceScheduleProblem::recognize(&multi_mode, &stop).unwrap().is_none());

    let mut unary = rcpsp();
    unary.resources.push(Resource::NoOverlap(vec![0, 1]));
    assert!(ResourceScheduleProblem::recognize(&unary, &stop).unwrap().is_none());

    let mut cyclic = rcpsp();
    cyclic.precedences.extend([(2, 1), (3, 0)]);
    assert!(ResourceScheduleProblem::recognize(&cyclic, &stop).unwrap().is_none());

    let mut duplicate_precedence = rcpsp();
    duplicate_precedence.precedences.push((0, 2));
    let duplicate_problem =
        ResourceScheduleProblem::recognize(&duplicate_precedence, &stop).unwrap().expect("duplicate arcs are semantic no-ops");
    let duplicate_state = ResourceScheduleState::serial(&duplicate_problem, PrioritySgs::stable(&duplicate_problem), &stop)
        .unwrap()
        .expect("duplicate precedence schedule");
    assert_verified(duplicate_precedence, &duplicate_state);

    let mut no_objective = rcpsp();
    no_objective.minimize_makespan = false;
    assert!(ResourceScheduleProblem::recognize(&no_objective, &stop).unwrap().is_none());

    let prearmed = AtomicBool::new(true);
    assert!(ResourceScheduleProblem::recognize(&rcpsp(), &prearmed).is_err());
}

#[test]
fn compiled_resource_demands_are_sparse_and_problem_clones_share_storage() {
    let stop = AtomicBool::new(false);
    let activity_count = 4_096;
    let resource_count = 128;
    let schedule = Schedule {
        intervals: (0..activity_count).map(|_| IntervalVar { duration: 1, horizon: 2, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: (0..resource_count)
            .map(|resource| Resource::Cumulative { demands: vec![(resource * 7 % activity_count, 1)], capacity: 1 })
            .collect(),
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    assert_eq!(problem.demand_entry_count(), resource_count);
    assert!(problem.demand_entry_count() < problem.activity_count() * problem.resource_count());

    let clone = problem.clone();
    assert!(problem.shares_storage_with(&clone));
    drop(problem);
    assert_eq!(clone.activity_count(), activity_count);
    assert_eq!(clone.resource_count(), resource_count);
}

#[test]
fn single_fixed_modes_preserve_windows_machines_and_semantic_references() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: vec![
            IntervalVar {
                duration: 0,
                horizon: 12,
                modes: vec![Mode { reference: Some(41), machine: 3, duration: 2, start_window: (3, 5) }],
                optional: false,
            },
            IntervalVar {
                duration: 0,
                horizon: 12,
                modes: vec![Mode { reference: Some(73), machine: 7, duration: 3, start_window: (5, 8) }],
                optional: false,
            },
        ],
        precedences: vec![(0, 1)],
        resources: vec![Resource::Cumulative { demands: vec![(0, 1), (1, 1)], capacity: 1 }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().expect("single modes are fixed decisions");
    assert_eq!(problem.duration(0), 2);
    assert_eq!(problem.earliest_start(0), 3);
    assert_eq!(problem.latest_start(1), 8);

    for scheme in [GenerationScheme::Serial, GenerationScheme::Parallel] {
        let state = ResourceScheduleState::construct(&problem, PrioritySgs::stable(&problem), scheme, &stop).unwrap().unwrap();
        assert_eq!(state.starts(), &[3, 5]);
        let solution = state.to_solution();
        assert_eq!(solution.machines, vec![3, 7]);
        assert_eq!(solution.modes, vec![Some(41), Some(73)]);
        assert_verified(schedule.clone(), &state);
    }
}

#[test]
fn duplicate_demands_are_aggregated_without_i64_overflow() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: vec![IntervalVar { duration: 1, horizon: 1, modes: Vec::new(), optional: false }],
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: vec![(0, i64::MAX), (0, i64::MAX)], capacity: i64::MAX }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    assert_eq!(problem.demand(0, 0), i128::from(i64::MAX) * 2);
    assert!(ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().is_none());
}

#[test]
fn serial_and_parallel_sgs_materialize_complete_verified_schedules() {
    let schedule = rcpsp();
    let stop = AtomicBool::new(false);
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let priority = PrioritySgs::stable(&problem);
    let serial = ResourceScheduleState::serial(&problem, priority.clone(), &stop).unwrap().unwrap();
    let parallel = ResourceScheduleState::parallel(&problem, priority, &stop).unwrap().unwrap();

    assert_eq!(serial.starts(), &[0, 2, 4, 5]);
    assert_eq!(serial.makespan(), 6);
    assert_eq!(serial.scheme(), GenerationScheme::Serial);
    assert_eq!(parallel.starts(), &[0, 2, 4, 5]);
    assert_eq!(parallel.makespan(), 6);
    assert_eq!(parallel.scheme(), GenerationScheme::Parallel);
    assert_verified(schedule.clone(), &serial);
    assert_verified(schedule, &parallel);
    assert!(serial.metrics().construction_candidates > 0);
    assert!(serial.metrics().profile_checks > 0);
    assert_eq!(serial.metrics().serial_constructions, 1);
    assert_eq!(parallel.metrics().parallel_constructions, 1);
}

#[test]
fn generic_priority_dispatch_is_topological_diversified_and_reproducible() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: [4, 1, 3, 2, 1, 2, 3, 4]
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 20, modes: Vec::new(), optional: false })
            .collect(),
        precedences: vec![(0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 7)],
        resources: vec![Resource::Cumulative { demands: (0..8).map(|activity| (activity, 1)).collect(), capacity: 2 }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();

    for rule in PriorityRule::ALL {
        let first = PrioritySgs::dispatch(&problem, 91, rule, &stop).unwrap().unwrap();
        let second = PrioritySgs::dispatch(&problem, 91, rule, &stop).unwrap().unwrap();
        assert_eq!(first, second);
        assert!(PrioritySgs::compile(&problem, first.order().to_vec(), &stop).unwrap().is_some());
        let state = ResourceScheduleState::serial(&problem, first, &stop).unwrap().unwrap();
        assert_verified(schedule.clone(), &state);
    }

    let shortest = PrioritySgs::dispatch(&problem, 17, PriorityRule::ShortestDuration, &stop).unwrap().unwrap();
    let longest = PrioritySgs::dispatch(&problem, 17, PriorityRule::LongestDuration, &stop).unwrap().unwrap();
    assert_eq!(problem.duration(shortest.order()[0]), 1);
    assert_eq!(problem.duration(longest.order()[0]), 4);

    let random_a = PrioritySgs::dispatch(&problem, 1, PriorityRule::Randomized, &stop).unwrap().unwrap();
    let random_b = PrioritySgs::dispatch(&problem, 2, PriorityRule::Randomized, &stop).unwrap().unwrap();
    assert_ne!(random_a.order(), random_b.order());
}

fn permutations(values: &mut [usize], position: usize, output: &mut Vec<Vec<usize>>) {
    if position == values.len() {
        output.push(values.to_vec());
        return;
    }
    for index in position..values.len() {
        values.swap(position, index);
        permutations(values, position + 1, output);
        values.swap(position, index);
    }
}

fn naive_serial(schedule: &Schedule, order: &[usize]) -> Option<Vec<i64>> {
    let count = schedule.intervals.len();
    let mut starts = vec![0i64; count];
    let mut scheduled = vec![false; count];
    for &activity in order {
        let duration = schedule.intervals[activity].duration;
        let latest = schedule.intervals[activity].horizon - duration;
        let release = schedule
            .precedences
            .iter()
            .filter_map(|&(before, after)| {
                (after == activity).then(|| scheduled[before].then_some(starts[before] + schedule.intervals[before].duration)).flatten()
            })
            .max()
            .unwrap_or(0);
        let start = (release..=latest).find(|&candidate| {
            schedule.resources.iter().all(|resource| {
                let Resource::Cumulative { demands, capacity } = resource else {
                    return false;
                };
                (candidate..candidate + duration).all(|time| {
                    let existing: i128 = demands
                        .iter()
                        .filter(|&&(other, _)| {
                            scheduled[other] && starts[other] <= time && time < starts[other] + schedule.intervals[other].duration
                        })
                        .map(|&(_, demand)| i128::from(demand))
                        .sum();
                    let added: i128 = demands.iter().filter(|&&(other, _)| other == activity).map(|&(_, demand)| i128::from(demand)).sum();
                    existing + added <= i128::from(*capacity)
                })
            })
        })?;
        starts[activity] = start;
        scheduled[activity] = true;
    }
    Some(starts)
}

#[test]
fn sparse_event_serial_sgs_matches_a_naive_time_oracle_for_every_small_priority() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: [1, 2, 1, 2]
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 6, modes: Vec::new(), optional: false })
            .collect(),
        precedences: Vec::new(),
        resources: vec![
            Resource::Cumulative { demands: vec![(0, 1), (1, 1), (2, 1), (3, 1)], capacity: 2 },
            Resource::Cumulative { demands: vec![(0, 2), (1, 1), (2, 1), (3, 2)], capacity: 3 },
        ],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut values = vec![0, 1, 2, 3];
    let mut orders = Vec::new();
    permutations(&mut values, 0, &mut orders);

    for order in orders {
        let priority = PrioritySgs::compile(&problem, order.clone(), &stop).unwrap().unwrap();
        let state = ResourceScheduleState::serial(&problem, priority.clone(), &stop).unwrap().unwrap();
        assert_eq!(state.starts(), naive_serial(&schedule, &order).unwrap(), "priority {order:?}");
        assert_verified(schedule.clone(), &state);

        let parallel = ResourceScheduleState::parallel(&problem, priority, &stop).unwrap().unwrap();
        assert_verified(schedule.clone(), &parallel);
    }
}

#[test]
fn profiles_scale_with_events_instead_of_the_numeric_horizon() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: (0..8).map(|_| IntervalVar { duration: 1, horizon: 1_000_000_000_000, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: (0..8).map(|activity| (activity, 1)).collect(), capacity: 2 }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    assert_eq!(state.makespan(), 4);
    assert!(state.metrics().peak_profile_events <= 2 * problem.activity_count());
    assert_verified(schedule, &state);
}

#[test]
fn zero_duration_activities_do_not_consume_capacity() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: vec![
            IntervalVar { duration: 0, horizon: 0, modes: Vec::new(), optional: false },
            IntervalVar { duration: 1, horizon: 1, modes: Vec::new(), optional: false },
        ],
        precedences: vec![(0, 1)],
        resources: vec![Resource::Cumulative { demands: vec![(0, i64::MAX), (1, 1)], capacity: 1 }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    for scheme in [GenerationScheme::Serial, GenerationScheme::Parallel] {
        let state = ResourceScheduleState::construct(&problem, PrioritySgs::stable(&problem), scheme, &stop).unwrap().unwrap();
        assert_eq!(state.starts(), &[0, 0]);
        assert_verified(schedule.clone(), &state);
    }
}

#[test]
fn priorities_are_complete_topological_orders_with_exact_relocation_bounds() {
    let stop = AtomicBool::new(false);
    let problem = ResourceScheduleProblem::recognize(&rcpsp(), &stop).unwrap().unwrap();
    assert!(PrioritySgs::compile(&problem, vec![0, 1, 2], &stop).unwrap().is_none());
    assert!(PrioritySgs::compile(&problem, vec![0, 1, 1, 3], &stop).unwrap().is_none());
    assert!(PrioritySgs::compile(&problem, vec![2, 0, 1, 3], &stop).unwrap().is_none());

    let priority = PrioritySgs::stable(&problem);
    assert_eq!(priority.order(), &[0, 1, 2, 3]);
    assert_eq!(priority.relocation_bounds(&problem, 0), Some((0, 1)));
    assert_eq!(priority.relocation_bounds(&problem, 1), Some((0, 2)));
    assert_eq!(priority.relocation_bounds(&problem, 2), Some((1, 3)));
    assert_eq!(priority.relocation_bounds(&problem, 3), Some((2, 3)));
}

#[test]
fn bounded_moves_preserve_precedence_and_commit_complete_schedules() {
    let schedule = rcpsp();
    let stop = AtomicBool::new(false);
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let base = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    let movements = base.bounded_moves(100, &stop).unwrap();
    assert!(!movements.is_empty());
    assert!(movements.len() <= 100);

    for movement in movements {
        let mut state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
        assert!(matches!(
            state.consider_move(movement, MinimizingMoveAcceptance::Always, &stop).unwrap(),
            ResourceMoveOutcome::Accepted { .. }
        ));
        assert!(PrioritySgs::compile(&problem, state.priority().order().to_vec(), &stop).unwrap().is_some());
        assert_verified(schedule.clone(), &state);
    }
}

#[test]
fn cyclic_bounded_batches_do_not_starve_late_priority_positions() {
    let stop = AtomicBool::new(false);
    let activity_count = 20;
    let schedule = Schedule {
        intervals: (0..activity_count).map(|_| IntervalVar { duration: 1, horizon: 20, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative {
            demands: (0..activity_count).map(|activity| (activity, 1)).collect(),
            capacity: activity_count as i64,
        }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    let mut visited = vec![false; activity_count - 1];
    for offset in 0..activity_count - 1 {
        let movements = state.bounded_moves_from(offset, 1, &stop).unwrap();
        let [ResourceScheduleMove::AdjacentSwap { first_position }] = movements.as_slice() else {
            panic!("a precedence-free cyclic batch starts with its offset adjacent move");
        };
        visited[*first_position] = true;
    }
    assert!(visited.into_iter().all(|seen| seen));
}

fn neighborhood_schedule() -> Schedule {
    Schedule {
        intervals: [1, 2, 1, 3, 2, 1]
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 20, modes: Vec::new(), optional: false })
            .collect(),
        precedences: vec![(0, 3), (1, 4)],
        resources: vec![
            Resource::Cumulative { demands: vec![(0, 2), (1, 1), (2, 1), (3, 2), (4, 2), (5, 1)], capacity: 3 },
            Resource::Cumulative { demands: vec![(0, 1), (1, 2), (2, 1), (3, 1), (4, 2), (5, 2)], capacity: 3 },
        ],
        minimize_makespan: true,
    }
}

#[test]
fn delta_and_parallel_workspace_match_the_full_reconstruction_oracle() {
    let stop = AtomicBool::new(false);
    let schedule = neighborhood_schedule();
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();

    for scheme in [GenerationScheme::Serial, GenerationScheme::Parallel] {
        let mut generator = ResourceScheduleState::construct(&problem, PrioritySgs::stable(&problem), scheme, &stop).unwrap().unwrap();
        let mut movements = generator.bounded_moves(128, &stop).unwrap();
        let mut macro_moves = Vec::new();
        let generated = generator
            .fill_alns_segment_moves(
                0x51a7,
                ResourceAlnsBudget { attempts: 256, max_moves: 64, max_segment_len: 4 },
                &mut macro_moves,
                &stop,
            )
            .unwrap();
        assert_eq!(generated.generated as usize, macro_moves.len());
        movements.extend(macro_moves);
        movements.sort_unstable();
        movements.dedup();
        assert!(movements.iter().any(|movement| matches!(movement, ResourceScheduleMove::SegmentRelocate { len, .. } if *len > 1)));

        for movement in movements {
            let mut state = ResourceScheduleState::construct(&problem, PrioritySgs::stable(&problem), scheme, &stop).unwrap().unwrap();
            let capacities = state.workspace_capacities();
            assert!(matches!(
                state.consider_move(movement, MinimizingMoveAcceptance::Always, &stop).unwrap(),
                ResourceMoveOutcome::Accepted { .. }
            ));
            let candidate_priority = PrioritySgs::compile(&problem, state.priority().order().to_vec(), &stop).unwrap().unwrap();
            let oracle = ResourceScheduleState::construct(&problem, candidate_priority, scheme, &stop).unwrap().unwrap();
            assert_eq!(state.starts(), oracle.starts(), "scheme={scheme:?}, movement={movement:?}");
            assert_eq!(state.makespan(), oracle.makespan(), "scheme={scheme:?}, movement={movement:?}");
            assert_eq!(state.metrics().oracle_validations, 1);
            assert_eq!(state.metrics().oracle_mismatches, 0);
            assert_eq!(state.metrics().workspace_growths, 0);
            assert_eq!(state.workspace_capacities(), capacities);
            assert_verified(schedule.clone(), &state);
        }
    }
}

#[test]
fn rejected_candidates_reuse_fixed_workspace_capacity_without_oracle_rebuilds() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: (0..12).map(|_| IntervalVar { duration: 1, horizon: 12, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: (0..12).map(|activity| (activity, 1)).collect(), capacity: 1 }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    let capacities = state.workspace_capacities();
    let construction_candidates = state.metrics().construction_candidates;
    for iteration in 0..1_024 {
        let first_position = iteration % 11;
        assert!(matches!(
            state
                .consider_move(ResourceScheduleMove::AdjacentSwap { first_position }, MinimizingMoveAcceptance::Improving, &stop,)
                .unwrap(),
            ResourceMoveOutcome::Rejected(ResourceMoveRejection::NotAccepted { .. })
        ));
        assert_eq!(state.workspace_capacities(), capacities);
    }
    let metrics = state.metrics();
    assert_eq!(metrics.delta_evaluations, 1_024);
    assert_eq!(metrics.oracle_validations, 0, "objective rejections must not invoke the allocating oracle");
    assert_eq!(metrics.workspace_growths, 0);
    assert_eq!(metrics.workspace_rollbacks, 1_024);
    assert_eq!(metrics.construction_candidates, construction_candidates);
    assert!(metrics.candidate_scheduling_attempts > 0);
    assert_verified(schedule, &state);
}

#[test]
fn a_late_serial_exchange_reschedules_only_its_changed_suffix() {
    let stop = AtomicBool::new(false);
    let activity_count = 32;
    let schedule = Schedule {
        intervals: (0..activity_count)
            .map(|_| IntervalVar { duration: 1, horizon: activity_count as i64, modes: Vec::new(), optional: false })
            .collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: (0..activity_count).map(|activity| (activity, 1)).collect(), capacity: 1 }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    assert!(matches!(
        state
            .consider_move(
                ResourceScheduleMove::AdjacentSwap { first_position: activity_count - 2 },
                MinimizingMoveAcceptance::Improving,
                &stop,
            )
            .unwrap(),
        ResourceMoveOutcome::Rejected(ResourceMoveRejection::NotAccepted { .. })
    ));
    assert_eq!(state.metrics().delta_activities_rescheduled, 2);
    assert_eq!(state.metrics().full_workspace_evaluations, 0);
    assert_eq!(state.metrics().oracle_validations, 0);
    assert_verified(schedule, &state);
}

#[test]
fn alns_segment_generation_is_budgeted_reproducible_and_precedence_safe() {
    let stop = AtomicBool::new(false);
    let schedule = neighborhood_schedule();
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let budget = ResourceAlnsBudget { attempts: 100, max_moves: 20, max_segment_len: 4 };
    let mut first_state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    let mut second_state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    let mut first = Vec::new();
    let mut second = Vec::new();
    let first_stats = first_state.fill_alns_segment_moves(991, budget, &mut first, &stop).unwrap();
    let second_stats = second_state.fill_alns_segment_moves(991, budget, &mut second, &stop).unwrap();

    assert_eq!(first, second);
    assert_eq!(first_stats, second_stats);
    assert!(first_stats.attempts <= budget.attempts as u64);
    assert_eq!(first_stats.generated as usize, first.len());
    assert!(first.len() <= budget.max_moves);
    assert!(!first.is_empty());

    for movement in first {
        let mut state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
        assert!(matches!(
            state.consider_move(movement, MinimizingMoveAcceptance::Always, &stop).unwrap(),
            ResourceMoveOutcome::Accepted { .. }
        ));
        assert!(PrioritySgs::compile(&problem, state.priority().order().to_vec(), &stop).unwrap().is_some());
        assert_verified(schedule.clone(), &state);
    }
}

#[test]
fn caller_owned_move_buffers_reserve_the_full_requested_target() {
    let stop = AtomicBool::new(false);
    let schedule = neighborhood_schedule();
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    let mut moves = Vec::with_capacity(64);

    state.fill_bounded_moves(100, &mut moves, &stop).unwrap();
    assert!(moves.capacity() >= 100);

    let mut moves = Vec::with_capacity(64);
    state.fill_alns_segment_moves(991, ResourceAlnsBudget { attempts: 4, max_moves: 100, max_segment_len: 2 }, &mut moves, &stop).unwrap();
    assert!(moves.capacity() >= 100);
}

#[test]
fn precedence_and_objective_rejections_leave_state_unchanged() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: [2, 2, 2].into_iter().map(|duration| IntervalVar { duration, horizon: 8, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1)],
        resources: vec![Resource::Cumulative { demands: vec![(0, 1), (1, 1), (2, 1)], capacity: 1 }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    let old_order = state.priority().order().to_vec();
    let old_starts = state.starts().to_vec();

    let precedence =
        state.consider_move(ResourceScheduleMove::AdjacentSwap { first_position: 0 }, MinimizingMoveAcceptance::Always, &stop).unwrap();
    assert_eq!(precedence, ResourceMoveOutcome::Rejected(ResourceMoveRejection::Precedence));
    assert_eq!(state.priority().order(), old_order);
    assert_eq!(state.starts(), old_starts);

    let objective =
        state.consider_move(ResourceScheduleMove::AdjacentSwap { first_position: 1 }, MinimizingMoveAcceptance::Improving, &stop).unwrap();
    assert!(matches!(objective, ResourceMoveOutcome::Rejected(ResourceMoveRejection::NotAccepted { .. })));
    assert_eq!(state.priority().order(), old_order);
    assert_eq!(state.starts(), old_starts);
}

#[test]
fn left_right_and_double_justification_keep_verified_nonworsening_states() {
    let schedule = rcpsp();
    let stop = AtomicBool::new(false);
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = ResourceScheduleState::parallel(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    for kind in [Justification::Right, Justification::Left, Justification::Double] {
        let before = state.makespan();
        let outcome = state.justify(kind, &stop).unwrap().expect("justification remains feasible");
        assert_eq!(outcome.previous, before);
        assert!(outcome.current <= before);
        assert_verified(schedule.clone(), &state);
    }
    assert_eq!(state.metrics().right_justifications, 1);
    assert_eq!(state.metrics().left_justifications, 1);
    assert_eq!(state.metrics().double_justifications, 1);
}

#[test]
fn a_move_after_right_justification_uses_a_safe_full_suffix_and_matches_the_oracle() {
    let stop = AtomicBool::new(false);
    let schedule = neighborhood_schedule();
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    state.justify(Justification::Right, &stop).unwrap().unwrap();
    let movement = state.bounded_moves(1, &stop).unwrap()[0];
    assert!(matches!(
        state.consider_move(movement, MinimizingMoveAcceptance::Always, &stop).unwrap(),
        ResourceMoveOutcome::Accepted { .. }
    ));
    let priority = PrioritySgs::compile(&problem, state.priority().order().to_vec(), &stop).unwrap().unwrap();
    let oracle = ResourceScheduleState::serial(&problem, priority, &stop).unwrap().unwrap();
    assert_eq!(state.starts(), oracle.starts());
    assert_eq!(state.makespan(), oracle.makespan());
    assert_eq!(state.metrics().oracle_mismatches, 0);
    assert!(state.metrics().delta_activities_rescheduled >= problem.activity_count() as u64);
    assert_verified(schedule, &state);
}

#[test]
fn cumulative_usage_above_i64_max_is_never_accepted_by_the_event_profile() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: (0..2).map(|_| IntervalVar { duration: 1, horizon: 2, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: vec![(0, i64::MAX), (1, i64::MAX)], capacity: i64::MAX }],
        minimize_makespan: true,
    };
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    assert_eq!(state.makespan(), 2);
    assert_ne!(state.starts()[0], state.starts()[1]);
    assert_verified(schedule, &state);
}

#[test]
fn cancellation_during_move_reconstruction_preserves_the_incumbent() {
    let activity_count = 30_000;
    let schedule = Schedule {
        intervals: (0..activity_count).map(|_| IntervalVar { duration: 1, horizon: 2, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: (0..4)
            .map(|_| Resource::Cumulative {
                demands: (0..activity_count).map(|activity| (activity, 1)).collect(),
                capacity: activity_count as i64,
            })
            .collect(),
        minimize_makespan: true,
    };
    let stop = AtomicBool::new(false);
    let problem = ResourceScheduleProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &stop).unwrap().unwrap();
    let old_order = state.priority().order().to_vec();
    let old_starts = state.starts().to_vec();
    let old_makespan = state.makespan();

    let outcome = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(1));
            stop.store(true, Ordering::Release);
        });
        state.consider_move(ResourceScheduleMove::Relocate { from: 0, to: activity_count - 1 }, MinimizingMoveAcceptance::Always, &stop)
    });

    assert!(outcome.is_err());
    assert_eq!(state.priority().order(), old_order);
    assert_eq!(state.starts(), old_starts);
    assert_eq!(state.makespan(), old_makespan);
}

#[test]
fn workspace_rebuild_honors_prearmed_and_concurrent_cancellation_without_mutating_state() {
    let activity_count = 200_000;
    let schedule = Schedule {
        intervals: (0..activity_count).map(|_| IntervalVar { duration: 0, horizon: 0, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: Vec::new(), capacity: 0 }],
        minimize_makespan: true,
    };
    let build_stop = AtomicBool::new(false);
    let problem = ResourceScheduleProblem::recognize(&schedule, &build_stop).unwrap().unwrap();
    let mut state = ResourceScheduleState::serial(&problem, PrioritySgs::stable(&problem), &build_stop).unwrap().unwrap();
    let starts = state.starts().to_vec();

    let prearmed = AtomicBool::new(true);
    assert!(state.probe_workspace_rebuild(&prearmed).is_err());

    let concurrent = AtomicBool::new(false);
    let outcome = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(1));
            concurrent.store(true, Ordering::Release);
        });
        state.probe_workspace_rebuild(&concurrent)
    });
    assert!(outcome.is_err());
    assert_eq!(state.starts(), starts);

    concurrent.store(false, Ordering::Release);
    let reload_outcome = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(1));
            concurrent.store(true, Ordering::Release);
        });
        state.probe_workspace_reload(&concurrent)
    });
    assert!(reload_outcome.is_err());
    assert_eq!(state.starts(), starts);
    concurrent.store(false, Ordering::Release);
    assert!(matches!(
        state
            .consider_move(
                ResourceScheduleMove::AdjacentSwap { first_position: activity_count - 2 },
                MinimizingMoveAcceptance::Improving,
                &concurrent,
            )
            .unwrap(),
        ResourceMoveOutcome::Rejected(ResourceMoveRejection::NotAccepted { .. })
    ));
    assert_verified(schedule, &state);
}
