use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use qayd::engines::ls::lists::move_acceptance::MinimizingMoveAcceptance;
use qayd::engines::ls::lists::schedule_state::{
    CriticalNeighborhood, DispatchRule, JobShopProblem, JobShopState, MachineArc, MoveOutcome, MoveProbe, MoveRejection, ScheduleMove,
};
use qayd::engines::ls::lists::solve_schedule;
use qayd::model::list::{verify_collection_solution, CollectionModel, IntervalVar, Mode, Resource, Schedule};

#[test]
fn minimizing_move_acceptance_has_the_shared_makespan_semantics() {
    assert!(MinimizingMoveAcceptance::Improving.accepts(10, 9));
    assert!(!MinimizingMoveAcceptance::Improving.accepts(10, 10));
    assert!(MinimizingMoveAcceptance::NonWorsening.accepts(10, 10));
    assert!(!MinimizingMoveAcceptance::NonWorsening.accepts(10, 11));
    assert!(MinimizingMoveAcceptance::StrictlyBelow(11).accepts(100, 10));
    assert!(!MinimizingMoveAcceptance::StrictlyBelow(10).accepts(0, 10));
    assert!(MinimizingMoveAcceptance::Always.accepts(10, 11));
}

fn fixed_job_shop() -> Schedule {
    let durations = [3, 2, 2, 4];
    let horizon = durations.iter().sum();
    Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (2, 3)],
        resources: vec![Resource::NoOverlap(vec![0, 3]), Resource::NoOverlap(vec![1, 2])],
        minimize_makespan: true,
    }
}

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

fn fixed_flow_shop(jobs: usize, machines: usize) -> Schedule {
    let durations = (0..jobs * machines)
        .map(|operation| 1 + i64::try_from((operation * 5 + operation / machines * 3) % 7).unwrap())
        .collect::<Vec<_>>();
    let horizon = durations.iter().sum();
    let intervals = durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect();
    let precedences =
        (0..jobs).flat_map(|job| (0..machines - 1).map(move |machine| (job * machines + machine, job * machines + machine + 1))).collect();
    let resources = (0..machines).map(|machine| Resource::NoOverlap((0..jobs).map(|job| job * machines + machine).collect())).collect();
    Schedule { intervals, precedences, resources, minimize_makespan: true }
}

fn independent_fixed_oracle(schedule: &Schedule, machine_sequences: &[Vec<usize>]) -> Option<(Vec<i64>, i64)> {
    let operation_count = schedule.intervals.len();
    let mut edges = schedule.precedences.clone();
    let mut seen = vec![false; operation_count];
    for sequence in machine_sequences {
        for &operation in sequence {
            if operation >= operation_count || seen[operation] {
                return None;
            }
            seen[operation] = true;
        }
        edges.extend(sequence.windows(2).map(|pair| (pair[0], pair[1])));
    }
    if seen.iter().any(|present| !present) {
        return None;
    }

    let mut starts = vec![0i64; operation_count];
    for _ in 0..operation_count {
        let mut changed = false;
        for &(before, after) in &edges {
            let candidate = starts[before].checked_add(schedule.intervals[before].duration)?;
            if candidate > starts[after] {
                starts[after] = candidate;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if edges.iter().any(|&(before, after)| {
        starts[before].checked_add(schedule.intervals[before].duration).is_none_or(|candidate| candidate > starts[after])
    }) {
        return None;
    }
    let mut makespan = 0i64;
    for (operation, interval) in schedule.intervals.iter().enumerate() {
        let end = starts[operation].checked_add(interval.duration)?;
        if starts[operation] > interval.horizon.checked_sub(interval.duration)? || end > interval.horizon {
            return None;
        }
        makespan = makespan.max(end);
    }
    Some((starts, makespan))
}

fn permutations(values: &[usize]) -> Vec<Vec<usize>> {
    fn visit(values: &mut [usize], position: usize, result: &mut Vec<Vec<usize>>) {
        if position == values.len() {
            result.push(values.to_vec());
            return;
        }
        for index in position..values.len() {
            values.swap(position, index);
            visit(values, position + 1, result);
            values.swap(position, index);
        }
    }
    let mut working = values.to_vec();
    let mut result = Vec::new();
    visit(&mut working, 0, &mut result);
    result
}

fn machine_arcs(sequence: &[usize]) -> BTreeSet<MachineArc> {
    sequence.windows(2).map(|pair| MachineArc { machine: 0, before: pair[0], after: pair[1] }).collect()
}

fn apply_test_move(sequence: &mut Vec<usize>, movement: ScheduleMove) {
    match movement {
        ScheduleMove::AdjacentSwap { first_position, .. } => sequence.swap(first_position, first_position + 1),
        ScheduleMove::Insert { from, to, .. } => {
            let operation = sequence.remove(from);
            sequence.insert(to, operation);
        }
    }
}

fn single_machine_schedule(operation_count: usize) -> Schedule {
    Schedule {
        intervals: (0..operation_count)
            .map(|operation| IntervalVar {
                duration: 1 + i64::try_from(operation % 3).unwrap(),
                horizon: i64::try_from(operation_count * 3).unwrap(),
                modes: Vec::new(),
                optional: false,
            })
            .collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap((0..operation_count).collect())],
        minimize_makespan: true,
    }
}

#[test]
fn strict_job_shop_recognition_has_explicit_machine_assignments() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&fixed_job_shop(), &stop).unwrap().expect("strict fixed JSSP");

    assert_eq!(problem.operation_count(), 4);
    assert_eq!(problem.machine_count(), 2);
    assert_eq!((0..4).map(|operation| problem.duration(operation)).collect::<Vec<_>>(), vec![3, 2, 2, 4]);
    assert_eq!((0..4).map(|operation| problem.machine(operation)).collect::<Vec<_>>(), vec![0, 1, 1, 0]);
}

#[test]
fn redundant_precedences_are_canonicalized_without_changing_semantics() {
    let stop = AtomicBool::new(false);
    let mut schedule = fixed_job_shop();
    schedule.precedences.extend([(0, 1), (2, 3), (0, 1)]);

    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().expect("redundant arcs are semantically harmless");
    let state = JobShopState::giffler_thompson(&problem, 5, DispatchRule::EarliestStart, &stop).unwrap().unwrap();
    let solution = state.to_solution();

    assert_eq!(verify_collection_solution(&collection(schedule), &solution).unwrap(), solution.objectives);
}

#[test]
fn recognition_rejects_shapes_that_need_the_general_scheduler() {
    let stop = AtomicBool::new(false);

    let mut optional = fixed_job_shop();
    optional.intervals[0].optional = true;
    assert!(JobShopProblem::recognize(&optional, &stop).unwrap().is_none());

    let mut cumulative = fixed_job_shop();
    cumulative.resources.push(Resource::Cumulative { demands: vec![(0, 1)], capacity: 1 });
    assert!(JobShopProblem::recognize(&cumulative, &stop).unwrap().is_none());

    let mut multiply_assigned = fixed_job_shop();
    multiply_assigned.resources.push(Resource::NoOverlap(vec![0]));
    assert!(JobShopProblem::recognize(&multiply_assigned, &stop).unwrap().is_none());

    let mut cyclic = fixed_job_shop();
    cyclic.precedences.extend([(1, 2), (3, 0)]);
    assert!(JobShopProblem::recognize(&cyclic, &stop).unwrap().is_none());

    let prearmed = AtomicBool::new(true);
    assert!(JobShopProblem::recognize(&fixed_job_shop(), &prearmed).is_err());
}

#[test]
fn single_mode_jobs_preserve_machine_and_mode_identity() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: vec![
            IntervalVar {
                duration: 0,
                horizon: 8,
                modes: vec![Mode { reference: Some(17), machine: 4, duration: 3, start_window: (0, 5) }],
                optional: false,
            },
            IntervalVar {
                duration: 0,
                horizon: 8,
                modes: vec![Mode { reference: Some(23), machine: 4, duration: 2, start_window: (0, 6) }],
                optional: false,
            },
        ],
        precedences: Vec::new(),
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().expect("single fixed mode is not flexible");
    let state = JobShopState::giffler_thompson(&problem, 4, DispatchRule::ShortestProcessingTime, &stop).unwrap().unwrap();
    let solution = state.to_solution();

    assert_eq!(solution.machines, vec![4, 4]);
    assert_eq!(solution.modes, vec![Some(17), Some(23)]);
    assert_eq!(verify_collection_solution(&collection(schedule), &solution).unwrap(), vec![5]);
}

#[test]
fn giffler_thompson_is_reproducible_and_materializes_a_verified_schedule() {
    let schedule = fixed_job_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let first = JobShopState::giffler_thompson(&problem, 91, DispatchRule::MostWorkRemaining, &stop).unwrap().unwrap();
    let second = JobShopState::giffler_thompson(&problem, 91, DispatchRule::MostWorkRemaining, &stop).unwrap().unwrap();

    assert_eq!(first.machine_sequences(), second.machine_sequences());
    assert_eq!(first.starts(), second.starts());
    assert_eq!(first.makespan(), second.makespan());
    assert_eq!(first.topological_order().len(), problem.operation_count());
    assert!(first.latest_starts().iter().zip(first.starts()).all(|(latest, earliest)| latest >= earliest));
    assert_eq!(verify_collection_solution(&collection(schedule), &first.to_solution()).unwrap(), vec![first.makespan()]);
    assert!(first.metrics().construction_candidates > 0);
    assert_eq!(first.metrics().reconstructions, 1);
}

#[test]
fn all_dispatch_rules_construct_complete_states() {
    let schedule = fixed_job_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();

    for (seed, rule) in DispatchRule::ALL.into_iter().enumerate() {
        let state = JobShopState::giffler_thompson(&problem, seed as u64, rule, &stop).unwrap().expect("constructor must complete");
        assert_eq!(state.starts().len(), problem.operation_count());
        assert!(verify_collection_solution(&collection(schedule.clone()), &state.to_solution()).is_ok());
    }
}

#[test]
fn profiled_constructor_preserves_work_on_infeasibility_and_interruption() {
    let stop = AtomicBool::new(false);
    let impossible = Schedule {
        intervals: vec![
            IntervalVar { duration: 1, horizon: 1, modes: Vec::new(), optional: false },
            IntervalVar { duration: 1, horizon: 1, modes: Vec::new(), optional: false },
        ],
        precedences: vec![(0, 1)],
        resources: Vec::new(),
        minimize_makespan: true,
    };
    let impossible_problem = JobShopProblem::recognize(&impossible, &stop).unwrap().unwrap();
    let (result, metrics) = JobShopState::giffler_thompson_profiled(&impossible_problem, 0, DispatchRule::EarliestStart, &stop);
    assert!(result.unwrap().is_none());
    assert_eq!(metrics.construction_candidates, 2);

    let large = fixed_flow_shop(1_250, 8);
    let large_problem = JobShopProblem::recognize(&large, &stop).unwrap().unwrap();
    let (result, metrics) = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(5));
            stop.store(true, std::sync::atomic::Ordering::Release);
        });
        JobShopState::giffler_thompson_profiled(&large_problem, 7, DispatchRule::Randomized, &stop)
    });
    assert!(result.is_err());
    assert!(metrics.construction_candidates > 0, "completed candidate evaluations must survive cancellation");
}

#[test]
fn reconstruction_matches_an_independent_longest_path_oracle() {
    let schedule = fixed_job_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let durations = [3i64, 2, 2, 4];

    for machine_zero_reversed in [false, true] {
        for machine_one_reversed in [false, true] {
            let machine_zero = if machine_zero_reversed { vec![3, 0] } else { vec![0, 3] };
            let machine_one = if machine_one_reversed { vec![2, 1] } else { vec![1, 2] };
            let sequences = vec![machine_zero, machine_one];
            let mut edges = vec![(0usize, 1usize), (2, 3)];
            edges.extend(sequences.iter().flat_map(|sequence| sequence.windows(2).map(|pair| (pair[0], pair[1]))));

            let mut starts = [0i64; 4];
            for _ in 0..4 {
                for &(before, after) in &edges {
                    starts[after] = starts[after].max(starts[before] + durations[before]);
                }
            }
            let cyclic = edges.iter().any(|&(before, after)| starts[before] + durations[before] > starts[after]);
            let state = JobShopState::from_machine_sequences(&problem, sequences, &stop).unwrap();

            if cyclic {
                assert!(state.is_none(), "cyclic disjunctive orientation was accepted");
            } else {
                let state = state.expect("acyclic disjunctive orientation");
                let oracle_makespan = (0..4).map(|operation| starts[operation] + durations[operation]).max().unwrap();
                assert_eq!(state.makespan(), oracle_makespan);
                assert_eq!(verify_collection_solution(&collection(schedule.clone()), &state.to_solution()).unwrap(), vec![oracle_makespan]);
            }
        }
    }
}

#[test]
fn exhaustive_small_moves_report_exact_operation_identity_arc_differences() {
    let stop = AtomicBool::new(false);
    for operation_count in 2..=5 {
        let schedule = single_machine_schedule(operation_count);
        let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
        let operations = (0..operation_count).collect::<Vec<_>>();
        for sequence in permutations(&operations) {
            let state = JobShopState::from_machine_sequences(&problem, vec![sequence.clone()], &stop).unwrap().unwrap();
            let mut movements = (0..operation_count - 1)
                .map(|first_position| ScheduleMove::AdjacentSwap { machine: 0, first_position })
                .collect::<Vec<_>>();
            movements.extend((0..operation_count).flat_map(|from| {
                (0..operation_count).filter(move |&to| to != from).map(move |to| ScheduleMove::Insert { machine: 0, from, to })
            }));

            for movement in movements {
                let reported = state.move_arcs(movement).expect("enumerated move is valid");
                let old_arcs = machine_arcs(&sequence);
                let mut candidate = sequence.clone();
                apply_test_move(&mut candidate, movement);
                let new_arcs = machine_arcs(&candidate);
                let expected_removed = old_arcs.difference(&new_arcs).copied().collect::<BTreeSet<_>>();
                let expected_added = new_arcs.difference(&old_arcs).copied().collect::<BTreeSet<_>>();
                let actual_removed = reported.removed.into_iter().flatten().collect::<BTreeSet<_>>();
                let actual_added = reported.added.into_iter().flatten().collect::<BTreeSet<_>>();
                assert_eq!(actual_removed, expected_removed, "removed arcs for {movement:?} on {sequence:?}");
                assert_eq!(actual_added, expected_added, "added arcs for {movement:?} on {sequence:?}");
            }
            assert!(state.move_arcs(ScheduleMove::AdjacentSwap { machine: 0, first_position: operation_count - 1 }).is_none());
        }
    }
}

#[test]
fn inverse_insert_reintroduces_removed_arcs_after_positions_shift() {
    let stop = AtomicBool::new(false);
    let schedule = single_machine_schedule(5);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1, 2, 3, 4]], &stop).unwrap().unwrap();
    let forward = ScheduleMove::Insert { machine: 0, from: 1, to: 4 };
    let forward_arcs = state.move_arcs(forward).unwrap();
    assert!(matches!(state.consider_move(forward, MinimizingMoveAcceptance::Always, &stop), Ok(MoveOutcome::Accepted { .. })));
    let inverse = ScheduleMove::Insert { machine: 0, from: 4, to: 1 };
    let inverse_arcs = state.move_arcs(inverse).unwrap();

    let removed = forward_arcs.removed.into_iter().flatten().collect::<BTreeSet<_>>();
    let reintroduced = inverse_arcs.added.into_iter().flatten().collect::<BTreeSet<_>>();
    let added = forward_arcs.added.into_iter().flatten().collect::<BTreeSet<_>>();
    let removed_by_inverse = inverse_arcs.removed.into_iter().flatten().collect::<BTreeSet<_>>();
    assert_eq!(reintroduced, removed);
    assert_eq!(removed_by_inverse, added);
}

#[test]
fn move_probe_is_atomic_for_feasible_and_rejected_candidates() {
    let stop = AtomicBool::new(false);
    let schedule = fixed_job_shop();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut feasible_state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![2, 1]], &stop).unwrap().unwrap();
    let feasible_sequences = feasible_state.machine_sequences().to_vec();
    let feasible_starts = feasible_state.starts().to_vec();
    let feasible_latest = feasible_state.latest_starts().to_vec();
    let feasible_topological = feasible_state.topological_order().to_vec();
    let feasible_critical_path = feasible_state.critical_path().to_vec();
    let feasible_makespan = feasible_state.makespan();
    let feasible = feasible_state.probe_move(ScheduleMove::AdjacentSwap { machine: 1, first_position: 0 }, &stop).unwrap();
    assert!(matches!(feasible, MoveProbe::Feasible { .. }));
    assert_eq!(feasible_state.machine_sequences(), feasible_sequences);
    assert_eq!(feasible_state.starts(), feasible_starts);
    assert_eq!(feasible_state.latest_starts(), feasible_latest);
    assert_eq!(feasible_state.topological_order(), feasible_topological);
    assert_eq!(feasible_state.critical_path(), feasible_critical_path);
    assert_eq!(feasible_state.makespan(), feasible_makespan);

    let mut rejected_state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![1, 2]], &stop).unwrap().unwrap();
    let rejected_sequences = rejected_state.machine_sequences().to_vec();
    let rejected_starts = rejected_state.starts().to_vec();
    let rejected_latest = rejected_state.latest_starts().to_vec();
    let rejected_topological = rejected_state.topological_order().to_vec();
    let rejected_critical_path = rejected_state.critical_path().to_vec();
    let rejected_makespan = rejected_state.makespan();
    let rejected = rejected_state.probe_move(ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 }, &stop).unwrap();
    assert_eq!(rejected, MoveProbe::Rejected(MoveRejection::Cycle));
    assert_eq!(rejected_state.machine_sequences(), rejected_sequences);
    assert_eq!(rejected_state.starts(), rejected_starts);
    assert_eq!(rejected_state.latest_starts(), rejected_latest);
    assert_eq!(rejected_state.topological_order(), rejected_topological);
    assert_eq!(rejected_state.critical_path(), rejected_critical_path);
    assert_eq!(rejected_state.makespan(), rejected_makespan);
}

#[test]
fn committing_a_probed_move_does_not_count_the_candidate_twice() {
    let stop = AtomicBool::new(false);
    let schedule = single_machine_schedule(4);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1, 2, 3]], &stop).unwrap().unwrap();
    let _ = state.take_metrics();
    let movement = ScheduleMove::Insert { machine: 0, from: 0, to: 3 };

    let candidate = match state.probe_move(movement, &stop).unwrap() {
        MoveProbe::Feasible { candidate, .. } => candidate,
        rejected => panic!("expected feasible probe, got {rejected:?}"),
    };
    assert_eq!(state.metrics().moves_considered, 1);
    let committed = state.commit_probed_move(movement, MinimizingMoveAcceptance::Always, &stop).unwrap();
    assert_eq!(committed, MoveOutcome::Accepted { previous: candidate, current: candidate });
    assert_eq!(state.metrics().moves_considered, 1);
    assert_eq!(state.metrics().moves_accepted, 1);
}

#[test]
fn strictly_below_is_applied_to_the_committed_candidate() {
    let stop = AtomicBool::new(false);
    let schedule = single_machine_schedule(4);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1, 2, 3]], &stop).unwrap().unwrap();
    let makespan = state.makespan();

    let accepted = state
        .consider_move(ScheduleMove::Insert { machine: 0, from: 0, to: 3 }, MinimizingMoveAcceptance::StrictlyBelow(makespan + 1), &stop)
        .unwrap();
    assert!(matches!(accepted, MoveOutcome::Accepted { .. }));
    let retained_sequences = state.machine_sequences().to_vec();
    let rejected = state
        .consider_move(ScheduleMove::Insert { machine: 0, from: 3, to: 0 }, MinimizingMoveAcceptance::StrictlyBelow(makespan), &stop)
        .unwrap();
    assert!(matches!(rejected, MoveOutcome::Rejected(MoveRejection::NotAccepted { .. })));
    assert_eq!(state.machine_sequences(), retained_sequences);
}

#[test]
fn replacement_rebuild_uses_the_owned_problem_without_mutating_the_state() {
    let stop = AtomicBool::new(false);
    let schedule = fixed_job_shop();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![2, 1]], &stop).unwrap().unwrap();
    let old_sequences = state.machine_sequences().to_vec();
    let old_starts = state.starts().to_vec();
    let old_makespan = state.makespan();

    let rebuilt = state.rebuilt_from_machine_sequences(vec![vec![0, 3], vec![1, 2]], &stop).unwrap().unwrap();
    assert_eq!(state.machine_sequences(), old_sequences);
    assert_eq!(state.starts(), old_starts);
    assert_eq!(state.makespan(), old_makespan);
    assert_eq!(rebuilt.machine_sequences(), &[vec![0, 3], vec![1, 2]]);
    assert_eq!(verify_collection_solution(&collection(schedule), &rebuilt.to_solution()).unwrap(), vec![rebuilt.makespan()]);
}

#[test]
fn taking_metrics_returns_deltas_and_preserves_workspace_observations() {
    let stop = AtomicBool::new(false);
    let schedule = single_machine_schedule(8);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![(0..8).collect()], &stop).unwrap().unwrap();
    let capacities = state.workspace_capacities();

    let construction = state.take_metrics();
    assert_eq!(construction.reconstructions, 1);
    assert_eq!(state.metrics(), Default::default());
    assert!(matches!(state.probe_move(ScheduleMove::Insert { machine: 0, from: 0, to: 7 }, &stop), Ok(MoveProbe::Feasible { .. })));
    let probe = state.take_metrics();
    assert_eq!(probe.moves_considered, 1);
    assert_eq!(probe.delta_evaluations, 1);
    assert_eq!(probe.full_evaluations, 0);
    assert_eq!(probe.oracle_validations, 0);
    assert_eq!(state.workspace_capacities(), capacities);
    assert_eq!(state.take_metrics(), Default::default());
    assert_eq!(state.workspace_capacities(), capacities);
}

#[test]
fn cycle_rejection_rolls_back_the_machine_order_and_schedule() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&fixed_job_shop(), &stop).unwrap().unwrap();
    // Machine 0: 0 -> 3, machine 1: 1 -> 2. Swapping machine 0 would
    // create 3 -> 0 -> 1 -> 2 -> 3.
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![1, 2]], &stop).unwrap().unwrap();
    let old_sequences = state.machine_sequences().to_vec();
    let old_starts = state.starts().to_vec();
    let old_makespan = state.makespan();

    let outcome =
        state.consider_move(ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 }, MinimizingMoveAcceptance::Always, &stop).unwrap();

    assert_eq!(outcome, MoveOutcome::Rejected(MoveRejection::Cycle));
    assert_eq!(state.machine_sequences(), old_sequences);
    assert_eq!(state.starts(), old_starts);
    assert_eq!(state.makespan(), old_makespan);
    assert_eq!(state.metrics().cycle_rejections, 1);
}

#[test]
fn objective_rejection_rolls_back_and_acceptance_commits_atomically() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&fixed_job_shop(), &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![2, 1]], &stop).unwrap().unwrap();
    let old_sequences = state.machine_sequences().to_vec();
    let old_starts = state.starts().to_vec();

    let rejected = state
        .consider_move(ScheduleMove::AdjacentSwap { machine: 1, first_position: 0 }, MinimizingMoveAcceptance::Improving, &stop)
        .unwrap();
    assert!(matches!(rejected, MoveOutcome::Rejected(MoveRejection::NotAccepted { .. }) | MoveOutcome::Rejected(MoveRejection::Cycle)));
    assert_eq!(state.machine_sequences(), old_sequences);
    assert_eq!(state.starts(), old_starts);

    let accepted =
        state.consider_move(ScheduleMove::AdjacentSwap { machine: 1, first_position: 0 }, MinimizingMoveAcceptance::Always, &stop).unwrap();
    assert!(matches!(accepted, MoveOutcome::Accepted { .. }));
    assert_ne!(state.machine_sequences(), old_sequences);
    assert!(verify_collection_solution(&collection(fixed_job_shop()), &state.to_solution()).is_ok());
}

#[test]
fn prearmed_move_interruption_is_observationally_atomic() {
    let build_stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&fixed_job_shop(), &build_stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![2, 1]], &build_stop).unwrap().unwrap();
    let old_sequences = state.machine_sequences().to_vec();
    let old_starts = state.starts().to_vec();
    let old_metrics = state.metrics();
    let stop = AtomicBool::new(true);

    let outcome =
        state.consider_move(ScheduleMove::AdjacentSwap { machine: 1, first_position: 0 }, MinimizingMoveAcceptance::Always, &stop);

    assert!(outcome.is_err());
    assert_eq!(state.machine_sequences(), old_sequences);
    assert_eq!(state.starts(), old_starts);
    assert_eq!(state.metrics(), old_metrics);
    assert!(state.critical_moves(CriticalNeighborhood::N1, &stop).is_err());
}

#[test]
fn critical_blocks_generate_bounded_n1_n5_and_n6_moves() {
    let stop = AtomicBool::new(false);
    let durations = [3, 2, 4, 1];
    let horizon = durations.iter().sum();
    let schedule = Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1, 2, 3])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1, 2, 3]], &stop).unwrap().unwrap();

    assert_eq!(state.critical_path(), &[0, 1, 2, 3]);
    assert_eq!(state.critical_blocks().len(), 1);
    let block = state.critical_blocks()[0];
    assert_eq!((block.machine(), block.first_position(), block.last_position(), block.len()), (0, 0, 3, 4));

    let n1 = state.critical_moves(CriticalNeighborhood::N1, &stop).unwrap();
    let n5 = state.critical_moves(CriticalNeighborhood::N5, &stop).unwrap();
    let n6 = state.critical_moves(CriticalNeighborhood::N6, &stop).unwrap();
    assert_eq!(n1.len(), 3);
    assert_eq!(n5.len(), 2);
    assert!(n6.len() > n5.len());
    assert!(n6.iter().any(|movement| matches!(movement, ScheduleMove::Insert { from: 1, to: 3, .. })));
    assert!(n6.iter().any(|movement| matches!(movement, ScheduleMove::Insert { from: 2, to: 0, .. })));
}

#[test]
fn exhaustive_three_by_three_orders_match_an_independent_oracle() {
    let stop = AtomicBool::new(false);
    let durations = [2, 3, 1, 3, 1, 2, 1, 2, 3];
    let horizon = durations.iter().sum();
    let schedule = Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)],
        resources: vec![Resource::NoOverlap(vec![0, 4, 8]), Resource::NoOverlap(vec![1, 5, 6]), Resource::NoOverlap(vec![2, 3, 7])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let machine_zero = permutations(&[0, 4, 8]);
    let machine_one = permutations(&[1, 5, 6]);
    let machine_two = permutations(&[2, 3, 7]);

    for zero in &machine_zero {
        for one in &machine_one {
            for two in &machine_two {
                let sequences = vec![zero.clone(), one.clone(), two.clone()];
                let oracle = independent_fixed_oracle(&schedule, &sequences);
                let state = JobShopState::from_machine_sequences(&problem, sequences, &stop).unwrap();
                match (oracle, state) {
                    (None, None) => {}
                    (Some((oracle_starts, oracle_makespan)), Some(mut state)) => {
                        assert_eq!(state.starts(), oracle_starts);
                        assert_eq!(state.makespan(), oracle_makespan);
                        assert!(state.matches_full_oracle(&stop).unwrap());
                    }
                    (oracle, state) => panic!("oracle/state disagreement: oracle={}, state={}", oracle.is_some(), state.is_some()),
                }
            }
        }
    }
}

#[test]
fn randomized_move_sequence_matches_full_and_independent_oracles() {
    let stop = AtomicBool::new(false);
    let schedule = fixed_flow_shop(4, 3);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let initial_sequences = (0..3).map(|machine| (0..4).map(|job| job * 3 + machine).collect()).collect();
    let mut state = JobShopState::from_machine_sequences(&problem, initial_sequences, &stop).unwrap().unwrap();
    let mut random = 0x82a2_b175_229d_6a5bu64;

    for step in 0..300 {
        random = random.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let machine = (random as usize) % 3;
        let first = ((random >> 16) as usize) % 4;
        let second = ((random >> 32) as usize) % 4;
        let movement = if step % 3 == 0 || first == second {
            ScheduleMove::AdjacentSwap { machine, first_position: first.min(2) }
        } else {
            ScheduleMove::Insert { machine, from: first, to: second }
        };
        let old_sequences = state.machine_sequences().to_vec();
        let outcome = state.consider_move(movement, MinimizingMoveAcceptance::Always, &stop).unwrap();
        match outcome {
            MoveOutcome::Accepted { .. } => {
                let (oracle_starts, oracle_makespan) =
                    independent_fixed_oracle(&schedule, state.machine_sequences()).expect("accepted order must be acyclic");
                assert_eq!(state.starts(), oracle_starts);
                assert_eq!(state.makespan(), oracle_makespan);
                assert!(state.matches_full_oracle(&stop).unwrap());
            }
            MoveOutcome::Rejected(MoveRejection::Cycle) => assert_eq!(state.machine_sequences(), old_sequences),
            unexpected => panic!("always-accepted move had unexpected outcome: {unexpected:?}"),
        }
    }
    let metrics = state.metrics();
    assert!(metrics.delta_evaluations > 0);
    assert_eq!(metrics.topological_rebuilds, 0);
    assert!(metrics.dirty_cone_operations >= metrics.delta_evaluations);
    assert!(metrics.oracle_validations >= metrics.moves_accepted);
    assert_eq!(metrics.oracle_mismatches, 0);
}

#[test]
fn large_dirty_cone_deduplicates_coincident_arcs_and_rejects_a_real_cycle() {
    let stop = AtomicBool::new(false);
    let operation_count = 65;
    let schedule = Schedule {
        intervals: (0..operation_count)
            .map(|_| IntervalVar { duration: 1, horizon: operation_count as i64, modes: Vec::new(), optional: false })
            .collect(),
        precedences: (0..operation_count - 1).step_by(2).map(|operation| (operation, operation + 1)).collect(),
        resources: vec![Resource::NoOverlap((0..operation_count).collect())],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![(0..operation_count).collect()], &stop).unwrap().unwrap();

    let accepted = state
        .consider_move(ScheduleMove::Insert { machine: 0, from: operation_count - 1, to: 0 }, MinimizingMoveAcceptance::Always, &stop)
        .unwrap();
    assert!(matches!(accepted, MoveOutcome::Accepted { .. }));
    let (oracle_starts, oracle_makespan) = independent_fixed_oracle(&schedule, state.machine_sequences()).unwrap();
    assert_eq!(state.starts(), oracle_starts);
    assert_eq!(state.makespan(), oracle_makespan);
    assert!(state.matches_full_oracle(&stop).unwrap());
    assert_eq!(state.metrics().max_dirty_cone, operation_count as u64);
    assert_eq!(state.metrics().topological_rebuilds, 0);
    assert_eq!(state.metrics().oracle_mismatches, 0);

    let old_sequences = state.machine_sequences().to_vec();
    let old_starts = state.starts().to_vec();
    let cycle =
        state.consider_move(ScheduleMove::AdjacentSwap { machine: 0, first_position: 1 }, MinimizingMoveAcceptance::Always, &stop).unwrap();
    assert_eq!(cycle, MoveOutcome::Rejected(MoveRejection::Cycle));
    assert_eq!(state.machine_sequences(), old_sequences);
    assert_eq!(state.starts(), old_starts);
    assert!(state.matches_full_oracle(&stop).unwrap());
    assert_eq!(state.metrics().cycle_rejections, 1);
    assert_eq!(state.metrics().topological_rebuilds, 0);
}

#[test]
fn start_windows_reject_and_rollback_a_disjunctive_move() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: vec![
            IntervalVar {
                duration: 0,
                horizon: 8,
                modes: vec![Mode { reference: None, machine: 0, duration: 3, start_window: (0, 0) }],
                optional: false,
            },
            IntervalVar {
                duration: 0,
                horizon: 8,
                modes: vec![Mode { reference: None, machine: 0, duration: 2, start_window: (3, 3) }],
                optional: false,
            },
        ],
        precedences: Vec::new(),
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1]], &stop).unwrap().unwrap();
    let old_starts = state.starts().to_vec();

    let outcome =
        state.consider_move(ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 }, MinimizingMoveAcceptance::Always, &stop);

    assert_eq!(outcome.unwrap(), MoveOutcome::Rejected(MoveRejection::Window));
    assert_eq!(state.machine_sequences(), &[vec![0, 1]]);
    assert_eq!(state.starts(), old_starts);
    assert!(state.matches_full_oracle(&stop).unwrap());
}

#[test]
fn insert_moves_in_both_directions_commit_and_rollback_exactly() {
    let stop = AtomicBool::new(false);
    let durations = [1, 2, 3, 4];
    let horizon = durations.iter().sum();
    let schedule = Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1, 2, 3])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1, 2, 3]], &stop).unwrap().unwrap();

    let rejected =
        state.consider_move(ScheduleMove::Insert { machine: 0, from: 0, to: 3 }, MinimizingMoveAcceptance::Improving, &stop).unwrap();
    assert!(matches!(rejected, MoveOutcome::Rejected(MoveRejection::NotAccepted { .. })));
    assert_eq!(state.machine_sequences(), &[vec![0, 1, 2, 3]]);

    assert!(matches!(
        state.consider_move(ScheduleMove::Insert { machine: 0, from: 0, to: 3 }, MinimizingMoveAcceptance::Always, &stop),
        Ok(MoveOutcome::Accepted { .. })
    ));
    assert_eq!(state.machine_sequences(), &[vec![1, 2, 3, 0]]);
    assert!(matches!(
        state.consider_move(ScheduleMove::Insert { machine: 0, from: 3, to: 1 }, MinimizingMoveAcceptance::Always, &stop),
        Ok(MoveOutcome::Accepted { .. })
    ));
    assert_eq!(state.machine_sequences(), &[vec![1, 0, 2, 3]]);
    assert!(state.matches_full_oracle(&stop).unwrap());
}

#[test]
fn repeated_rejections_reuse_all_candidate_buffers() {
    let stop = AtomicBool::new(false);
    let operation_count = 64;
    let schedule = Schedule {
        intervals: (0..operation_count)
            .map(|operation| IntervalVar {
                duration: 1 + i64::try_from(operation % 5).unwrap(),
                horizon: 1_000,
                modes: Vec::new(),
                optional: false,
            })
            .collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap((0..operation_count).collect())],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let sequence = (0..operation_count).collect::<Vec<_>>();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![sequence.clone()], &stop).unwrap().unwrap();

    let warmup = state
        .consider_move(ScheduleMove::Insert { machine: 0, from: 0, to: operation_count - 1 }, MinimizingMoveAcceptance::Improving, &stop)
        .unwrap();
    assert!(matches!(warmup, MoveOutcome::Rejected(MoveRejection::NotAccepted { .. })));
    let capacities = state.workspace_capacities();
    let initial_metrics = state.metrics();
    assert_eq!(initial_metrics.workspace_growths, 0);
    for iteration in 0..1_000 {
        let from = iteration % operation_count;
        let mut to = (iteration * 17 + 11) % operation_count;
        if to == from {
            to = (to + 1) % operation_count;
        }
        let outcome =
            state.consider_move(ScheduleMove::Insert { machine: 0, from, to }, MinimizingMoveAcceptance::Improving, &stop).unwrap();
        assert!(matches!(outcome, MoveOutcome::Rejected(MoveRejection::NotAccepted { .. })));
        assert_eq!(state.workspace_capacities(), capacities);
    }
    assert_eq!(state.machine_sequences(), &[sequence]);
    assert_eq!(state.metrics().full_evaluations, initial_metrics.full_evaluations);
    assert_eq!(state.metrics().oracle_validations, initial_metrics.oracle_validations);
    assert_eq!(state.metrics().critical_path_updates, initial_metrics.critical_path_updates);
    assert_eq!(state.metrics().workspace_growths, 0);

    let before_local_delta = state.metrics();
    let local = state
        .consider_move(
            ScheduleMove::AdjacentSwap { machine: 0, first_position: operation_count - 2 },
            MinimizingMoveAcceptance::Improving,
            &stop,
        )
        .unwrap();
    assert!(matches!(local, MoveOutcome::Rejected(MoveRejection::NotAccepted { .. })));
    assert_eq!(state.metrics().topological_rebuilds, before_local_delta.topological_rebuilds);
    assert_eq!(state.metrics().delta_evaluations, before_local_delta.delta_evaluations + 1);
}

#[test]
fn critical_move_buffer_is_reused_and_duplicate_free() {
    let stop = AtomicBool::new(false);
    let durations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let horizon = durations.iter().sum();
    let schedule = Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap((0..durations.len()).collect())],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = JobShopState::from_machine_sequences(&problem, vec![(0..durations.len()).collect()], &stop).unwrap().unwrap();
    let mut movements = Vec::new();

    state.fill_critical_moves(CriticalNeighborhood::N6, &mut movements, &stop).unwrap();
    let capacity = movements.capacity();
    assert!(movements.windows(2).all(|pair| pair[0] < pair[1]));
    for neighborhood in [CriticalNeighborhood::N1, CriticalNeighborhood::N5, CriticalNeighborhood::N6] {
        state.fill_critical_moves(neighborhood, &mut movements, &stop).unwrap();
        assert_eq!(movements.capacity(), capacity);
        assert!(movements.windows(2).all(|pair| pair[0] < pair[1]));
    }
    state
        .fill_critical_move_union(&[CriticalNeighborhood::N5, CriticalNeighborhood::N1, CriticalNeighborhood::N6], &mut movements, &stop)
        .unwrap();
    let union_capacity = movements.capacity();
    assert!(movements.windows(2).all(|pair| pair[0] < pair[1]));
    state
        .fill_critical_move_union(&[CriticalNeighborhood::N5, CriticalNeighborhood::N1, CriticalNeighborhood::N6], &mut movements, &stop)
        .unwrap();
    assert_eq!(movements.capacity(), union_capacity);
}

#[test]
fn cancellation_during_move_probe_and_commit_preserves_the_incumbent() {
    let stop = AtomicBool::new(false);
    let operation_count = 20_000;
    let schedule = Schedule {
        intervals: (0..operation_count)
            .map(|_| IntervalVar { duration: 1, horizon: operation_count as i64, modes: Vec::new(), optional: false })
            .collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap((0..operation_count).collect())],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let sequence = (0..operation_count).collect::<Vec<_>>();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![sequence], &stop).unwrap().unwrap();
    let old_sequences = state.machine_sequences().to_vec();
    let old_starts = state.starts().to_vec();

    let probe = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(1));
            stop.store(true, std::sync::atomic::Ordering::Release);
        });
        state.probe_move(ScheduleMove::Insert { machine: 0, from: 0, to: operation_count - 1 }, &stop)
    });

    assert!(probe.is_err());
    assert_eq!(state.machine_sequences(), old_sequences);
    assert_eq!(state.starts(), old_starts);

    stop.store(false, std::sync::atomic::Ordering::Release);
    let commit = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(1));
            stop.store(true, std::sync::atomic::Ordering::Release);
        });
        state.consider_move(ScheduleMove::Insert { machine: 0, from: 0, to: operation_count - 1 }, MinimizingMoveAcceptance::Always, &stop)
    });
    assert!(commit.is_err());
    assert_eq!(state.machine_sequences(), old_sequences);
    assert_eq!(state.starts(), old_starts);
}

#[test]
fn large_job_shop_uses_structured_moves_deterministically_and_publishes_only_valid_schedules() {
    let schedule = fixed_flow_shop(7, 8);
    assert!(schedule.intervals.len() > 48);
    let model = collection(schedule.clone());

    let run = || {
        let stop = AtomicBool::new(false);
        let mut reports = Vec::new();
        let (solution, metrics) = solve_schedule(&schedule, 73, &stop, &mut |objective| reports.push(objective));
        (solution, metrics, reports)
    };
    let (first, first_metrics, first_reports) = run();
    let (second, second_metrics, second_reports) = run();

    assert!(first.feasible);
    assert_eq!(first_metrics.constructor, "giffler-thompson-critical-path");
    assert!(first_metrics.candidates > 0);
    assert!(first_metrics.moves_considered > 0, "critical neighborhoods were not entered");
    assert!(first_metrics.moves_accepted > 0, "bounded diversification never committed a complete move");
    assert!(first_metrics.reconstructions > first_metrics.moves_accepted);
    assert!(first_metrics.critical_path_updates > 0);
    assert_eq!(verify_collection_solution(&model, &first).unwrap(), first.objectives);
    assert!(!first_reports.is_empty());
    assert!(first_reports.windows(2).all(|pair| pair[1] < pair[0]));

    assert_eq!(second.objectives, first.objectives);
    assert_eq!(second.starts, first.starts);
    assert_eq!(second_reports, first_reports);
    assert_eq!(second_metrics.candidates, first_metrics.candidates);
    assert_eq!(second_metrics.moves_considered, first_metrics.moves_considered);
    assert_eq!(second_metrics.moves_accepted, first_metrics.moves_accepted);
    assert_eq!(second_metrics.cycle_rejections, first_metrics.cycle_rejections);
    assert_eq!(second_metrics.reconstructions, first_metrics.reconstructions);
    assert_eq!(second_metrics.critical_path_updates, first_metrics.critical_path_updates);
}
