use std::sync::atomic::AtomicBool;

use qayd::engines::ls::lists::schedule_elite::{
    test_arc_bottom_k, test_bottom_k_jaccard_distance, test_clear_interrupt, test_interrupt_after_checkpoints, ScheduleEliteArchive,
    ScheduleEliteCandidate, ScheduleEliteConsiderOutcome, ScheduleEliteError, ScheduleEliteSolveToken, SCHEDULE_ELITE_CAPACITY,
};
use qayd::engines::ls::lists::schedule_state::{JobShopProblem, JobShopState};
use qayd::model::list::{verify_collection_solution, CollectionModel, IntervalVar, Mode, Resource, Schedule};

fn flow_shop() -> Schedule {
    let durations = [3, 1, 1, 4, 2, 2];
    let horizon = durations.iter().sum();
    Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (2, 3), (4, 5)],
        resources: vec![Resource::NoOverlap(vec![0, 2, 4]), Resource::NoOverlap(vec![1, 3, 5])],
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

    let mut values = values.to_vec();
    let mut result = Vec::new();
    visit(&mut values, 0, &mut result);
    result
}

fn all_orders() -> Vec<Vec<Vec<usize>>> {
    let first = permutations(&[0, 2, 4]);
    let second = permutations(&[1, 3, 5]);
    first.iter().flat_map(|first| second.iter().map(move |second| vec![first.clone(), second.clone()])).collect()
}

fn state(problem: &JobShopProblem, orders: Vec<Vec<usize>>) -> JobShopState {
    JobShopState::from_machine_sequences(problem, orders, &AtomicBool::new(false)).unwrap().expect("flow-shop order is acyclic")
}

fn snapshot(archive: &ScheduleEliteArchive) -> Vec<(i64, u64, Vec<Vec<usize>>)> {
    let stop = AtomicBool::new(false);
    archive.entries().iter().map(|entry| (entry.objective(), entry.order_hash(), entry.decode_machine_sequences(&stop).unwrap())).collect()
}

#[test]
fn compressed_elites_round_trip_through_the_full_oracle_and_semantic_replay() {
    let schedule = flow_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut archive = ScheduleEliteArchive::new();

    for orders in all_orders() {
        archive.consider(&state(&problem, orders), &stop).unwrap();
    }

    assert_eq!(archive.len(), SCHEDULE_ELITE_CAPACITY);
    for entry in archive.entries() {
        assert_eq!(entry.machine_count(), 2);
        assert_eq!(entry.operation_count(), 6);
        let _: &[u32] = entry.machine_sequence(0).unwrap();
        let orders = entry.decode_machine_sequences(&stop).unwrap();
        let rebuilt = JobShopState::from_machine_sequences(&problem, orders, &stop).unwrap().expect("archived order reconstructs");
        let solution = rebuilt.to_solution();

        assert_eq!(rebuilt.makespan(), entry.objective());
        assert_eq!(verify_collection_solution(&collection(schedule.clone()), &solution).unwrap(), vec![entry.objective()]);
    }
}

#[test]
fn capacity_four_keeps_the_global_best_and_a_farthest_first_frontier() {
    let schedule = flow_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut archive = ScheduleEliteArchive::new();
    let mut global_best = i64::MAX;
    let mut observed_bounded_replacement = false;

    for orders in all_orders() {
        let state = state(&problem, orders);
        global_best = global_best.min(state.makespan());
        let outcome = archive.consider(&state, &stop).unwrap();
        observed_bounded_replacement |=
            matches!(outcome, ScheduleEliteConsiderOutcome::InsertedAndEvicted | ScheduleEliteConsiderOutcome::Dominated);
        assert!(archive.len() <= archive.capacity());
    }

    assert!(observed_bounded_replacement);
    assert_eq!(archive.len(), SCHEDULE_ELITE_CAPACITY);
    assert_eq!(archive.best().unwrap().objective(), global_best);

    // The retained order itself is a farthest-first traversal after the best.
    for position in 1..archive.len() {
        let selected = &archive.entries()[..position];
        let chosen = &archive.entries()[position];
        let chosen_distance = selected.iter().map(|entry| chosen.arc_distance(entry).unwrap()).min().unwrap();
        for remaining in &archive.entries()[position + 1..] {
            let remaining_distance = selected.iter().map(|entry| remaining.arc_distance(entry).unwrap()).min().unwrap();
            assert!(chosen_distance >= remaining_distance);
        }
    }
}

#[test]
fn duplicate_orders_do_not_grow_or_perturb_the_archive() {
    let schedule = flow_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = state(&problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    let mut archive = ScheduleEliteArchive::new();

    assert_eq!(archive.consider(&state, &stop).unwrap(), ScheduleEliteConsiderOutcome::Inserted);
    let before = snapshot(&archive);
    assert_eq!(archive.consider(&state, &stop).unwrap(), ScheduleEliteConsiderOutcome::Duplicate);

    assert_eq!(archive.len(), 1);
    assert_eq!(snapshot(&archive), before);
}

#[test]
fn bottom_k_distance_separates_distinct_machine_orders() {
    let schedule = flow_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let first = state(&problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    let second = state(&problem, vec![vec![4, 2, 0], vec![1, 3, 5]]);
    let mut archive = ScheduleEliteArchive::new();

    archive.consider(&first, &stop).unwrap();
    archive.consider(&second, &stop).unwrap();

    assert_eq!(archive.len(), 2);
    assert_eq!(archive.entries()[0].arc_distance(&archive.entries()[0]), Some(0));
    assert!(archive.entries()[0].arc_distance(&archive.entries()[1]).unwrap() > 0);
}

#[test]
fn fixed_candidate_stream_produces_a_bit_stable_frontier() {
    let schedule = flow_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let orders = all_orders();
    let mut first = ScheduleEliteArchive::new();
    let mut second = ScheduleEliteArchive::new();

    for order in &orders {
        first.consider(&state(&problem, order.clone()), &stop).unwrap();
    }
    for order in &orders {
        second.consider(&state(&problem, order.clone()), &stop).unwrap();
    }

    assert_eq!(snapshot(&first), snapshot(&second));
}

#[test]
fn canonical_batch_is_independent_of_every_five_candidate_permutation() {
    let schedule = flow_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let states = all_orders().into_iter().take(5).map(|orders| state(&problem, orders)).collect::<Vec<_>>();
    let index_permutations = permutations(&[0, 1, 2, 3, 4]);
    let mut expected = None;

    for permutation in index_permutations {
        let batch = permutation.iter().map(|&index| &states[index]).collect::<Vec<_>>();
        let mut archive = ScheduleEliteArchive::new();
        let outcome = archive.consider_batch(&batch, &stop).unwrap();

        assert_eq!(outcome.considered, 5);
        assert_eq!(outcome.duplicates, 0);
        assert_eq!(outcome.retained, SCHEDULE_ELITE_CAPACITY);
        assert_eq!(outcome.dominated, 1);
        assert_eq!(outcome.evicted, 0);
        let actual = snapshot(&archive);
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected);
        } else {
            expected = Some(actual);
        }
    }
}

#[test]
fn canonical_batch_deduplicates_before_selecting_the_frontier() {
    let schedule = flow_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let first = state(&problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    let second = state(&problem, vec![vec![2, 0, 4], vec![1, 3, 5]]);
    let mut archive = ScheduleEliteArchive::new();

    let outcome = archive.consider_batch(&[&first, &second, &first, &second], &stop).unwrap();

    assert_eq!(outcome.considered, 4);
    assert_eq!(outcome.duplicates, 2);
    assert_eq!(outcome.retained, 2);
    assert_eq!(archive.len(), 2);
}

#[test]
fn compact_candidate_batch_matches_the_full_state_batch_and_reports_pool_encoding() {
    let schedule = flow_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let states = all_orders().into_iter().take(6).map(|orders| state(&problem, orders)).collect::<Vec<_>>();
    let references = states.iter().collect::<Vec<_>>();
    let mut full = ScheduleEliteArchive::new();
    let full_outcome = full.consider_batch(&references, &stop).unwrap();

    let token = ScheduleEliteSolveToken::new();
    let candidates = states.iter().rev().map(|state| ScheduleEliteCandidate::capture(state, &token, &stop).unwrap()).collect::<Vec<_>>();
    assert!(candidates.iter().all(|candidate| candidate.heap_lower_bound_bytes() > 0));
    let mut compact = ScheduleEliteArchive::new();
    let compact_outcome = compact.consider_candidate_batch(&states[0], &token, candidates, &stop).unwrap();

    assert_eq!(compact_outcome, full_outcome);
    assert_eq!(snapshot(&compact), snapshot(&full));
    assert_eq!(
        compact.objectives_summary(),
        compact.entries().iter().map(|entry| entry.objective().to_string()).collect::<Vec<_>>().join(",")
    );
    assert!(compact.heap_lower_bound_bytes() > 0);

    let pairwise = compact
        .entries()
        .iter()
        .enumerate()
        .flat_map(|(left, entry)| compact.entries()[left + 1..].iter().map(move |other| entry.arc_distance(other).unwrap()))
        .collect::<Vec<_>>();
    let expected_min = *pairwise.iter().min().unwrap();
    let expected_max = *pairwise.iter().max().unwrap();
    let expected_mean = u32::try_from(pairwise.iter().map(|&distance| u64::from(distance)).sum::<u64>() / pairwise.len() as u64).unwrap();
    assert_eq!(compact.distance_stats_ppm(), Some((expected_min, expected_mean, expected_max)));
    let encoded_distances = compact.pairwise_distances_ppm_summary();
    assert_eq!(encoded_distances.split(',').count(), pairwise.len());
    assert!(encoded_distances.split(',').all(|pair| pair.contains('=') && pair.contains('-')));
}

#[test]
fn bottom_k_distance_remains_informative_at_one_million_arcs() {
    const ARCS: usize = 1_000_000;
    const SHARED: usize = 900_000;
    let left = test_arc_bottom_k((0..ARCS).map(|operation| (operation % 1_000, operation, operation + 1)));
    let right = test_arc_bottom_k((0..ARCS).map(|operation| {
        let after = if operation < SHARED { operation + 1 } else { operation.wrapping_add(17) };
        (operation % 1_000, operation, after)
    }));

    assert_eq!(left.len(), 256);
    assert_eq!(right.len(), 256);
    assert_eq!(test_bottom_k_jaccard_distance(&left, &left), 0);
    let distance = test_bottom_k_jaccard_distance(&left, &right);
    assert!((50_000..=350_000).contains(&distance), "ten-percent arc replacement should remain visible, got {distance}");
}

fn single_mode_schedule(first_window: (i64, i64)) -> Schedule {
    Schedule {
        intervals: vec![
            IntervalVar {
                duration: 0,
                horizon: 10,
                modes: vec![Mode { reference: Some(1), machine: 7, duration: 2, start_window: first_window }],
                optional: false,
            },
            IntervalVar {
                duration: 0,
                horizon: 10,
                modes: vec![Mode { reference: Some(2), machine: 7, duration: 2, start_window: (0, 8) }],
                optional: false,
            },
        ],
        precedences: Vec::new(),
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    }
}

#[test]
fn exact_identity_rejects_timing_precedence_and_machine_mode_identity_changes() {
    let stop = AtomicBool::new(false);
    let schedule = flow_shop();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let original = state(&problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    let mut archive = ScheduleEliteArchive::new();
    archive.consider(&original, &stop).unwrap();
    let before = snapshot(&archive);

    let mut changed_horizon = schedule;
    changed_horizon.intervals[0].horizon += 1;
    let changed_problem = JobShopProblem::recognize(&changed_horizon, &stop).unwrap().unwrap();
    let changed_state = state(&changed_problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    assert_eq!(archive.consider(&changed_state, &stop).unwrap(), ScheduleEliteConsiderOutcome::Incompatible);
    assert_eq!(snapshot(&archive), before);

    let mut changed_raw_machines = flow_shop();
    changed_raw_machines.resources.insert(0, Resource::NoOverlap(Vec::new()));
    let changed_problem = JobShopProblem::recognize(&changed_raw_machines, &stop).unwrap().unwrap();
    let changed_state = state(&changed_problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    assert_eq!(archive.consider(&changed_state, &stop).unwrap(), ScheduleEliteConsiderOutcome::Incompatible);
    assert_eq!(snapshot(&archive), before);

    let mut changed_precedence = flow_shop();
    changed_precedence.precedences[0] = (0, 3);
    let changed_problem = JobShopProblem::recognize(&changed_precedence, &stop).unwrap().unwrap();
    let changed_state = state(&changed_problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    assert_eq!(archive.consider(&changed_state, &stop).unwrap(), ScheduleEliteConsiderOutcome::Incompatible);
    assert_eq!(snapshot(&archive), before);

    let first_modes = single_mode_schedule((0, 8));
    let first_problem = JobShopProblem::recognize(&first_modes, &stop).unwrap().unwrap();
    let first_state = state(&first_problem, vec![vec![0, 1]]);
    let mut mode_archive = ScheduleEliteArchive::new();
    mode_archive.consider(&first_state, &stop).unwrap();
    let mode_before = snapshot(&mode_archive);
    let changed_modes = single_mode_schedule((1, 8));
    let changed_problem = JobShopProblem::recognize(&changed_modes, &stop).unwrap().unwrap();
    let changed_state = state(&changed_problem, vec![vec![0, 1]]);
    assert_eq!(mode_archive.consider(&changed_state, &stop).unwrap(), ScheduleEliteConsiderOutcome::Incompatible);
    assert_eq!(snapshot(&mode_archive), mode_before);

    let mut changed_output_machine = single_mode_schedule((0, 8));
    for interval in &mut changed_output_machine.intervals {
        interval.modes[0].machine = 70;
    }
    let changed_problem = JobShopProblem::recognize(&changed_output_machine, &stop).unwrap().unwrap();
    let changed_state = state(&changed_problem, vec![vec![0, 1]]);
    assert_eq!(mode_archive.consider(&changed_state, &stop).unwrap(), ScheduleEliteConsiderOutcome::Incompatible);
    assert_eq!(snapshot(&mode_archive), mode_before);

    let mut changed_mode_reference = single_mode_schedule((0, 8));
    changed_mode_reference.intervals[0].modes[0].reference = Some(99);
    let changed_problem = JobShopProblem::recognize(&changed_mode_reference, &stop).unwrap().unwrap();
    let changed_state = state(&changed_problem, vec![vec![0, 1]]);
    assert_eq!(mode_archive.consider(&changed_state, &stop).unwrap(), ScheduleEliteConsiderOutcome::Incompatible);
    assert_eq!(snapshot(&mode_archive), mode_before);
}

#[test]
fn compact_candidates_reject_cross_solve_tokens_and_corrupt_semantics_transactionally() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&flow_shop(), &stop).unwrap().unwrap();
    let original = state(&problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    let first_token = ScheduleEliteSolveToken::new();
    let second_token = ScheduleEliteSolveToken::new();
    let mut archive = ScheduleEliteArchive::new();
    let candidate = ScheduleEliteCandidate::capture(&original, &first_token, &stop).unwrap();
    archive.consider_candidate_batch(&original, &first_token, vec![candidate], &stop).unwrap();
    let before = snapshot(&archive);

    let cross_solve = ScheduleEliteCandidate::capture(&original, &second_token, &stop).unwrap();
    assert_eq!(
        archive.consider_candidate_batch(&original, &second_token, vec![cross_solve], &stop),
        Err(ScheduleEliteError::IncompatibleSolve)
    );
    assert_eq!(snapshot(&archive), before);

    let mut corrupt_order = ScheduleEliteCandidate::capture(&original, &first_token, &stop).unwrap();
    corrupt_order.test_corrupt_first_operation();
    assert_eq!(
        archive.consider_candidate_batch(&original, &first_token, vec![corrupt_order], &stop),
        Err(ScheduleEliteError::InvalidMachineOrder)
    );
    assert_eq!(snapshot(&archive), before);

    let mut corrupt_objective = ScheduleEliteCandidate::capture(&original, &first_token, &stop).unwrap();
    corrupt_objective.test_corrupt_objective();
    assert_eq!(
        archive.consider_candidate_batch(&original, &first_token, vec![corrupt_objective], &stop),
        Err(ScheduleEliteError::InvalidObjective)
    );
    assert_eq!(snapshot(&archive), before);
}

#[test]
fn compact_candidate_mid_batch_interruption_leaves_the_archive_unchanged() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&flow_shop(), &stop).unwrap().unwrap();
    let states = all_orders().into_iter().take(3).map(|orders| state(&problem, orders)).collect::<Vec<_>>();
    let token = ScheduleEliteSolveToken::new();
    let mut archive = ScheduleEliteArchive::new();
    archive
        .consider_candidate_batch(&states[0], &token, vec![ScheduleEliteCandidate::capture(&states[0], &token, &stop).unwrap()], &stop)
        .unwrap();
    let before = snapshot(&archive);
    let candidates = states[1..].iter().map(|state| ScheduleEliteCandidate::capture(state, &token, &stop).unwrap()).collect::<Vec<_>>();

    test_interrupt_after_checkpoints(10);
    assert_eq!(archive.consider_candidate_batch(&states[0], &token, candidates, &stop), Err(ScheduleEliteError::Interrupted));
    test_clear_interrupt();
    assert_eq!(snapshot(&archive), before);
}

#[test]
fn inconsistent_objective_for_an_exact_order_is_an_error() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&flow_shop(), &stop).unwrap().unwrap();
    let state = state(&problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    let mut archive = ScheduleEliteArchive::new();
    archive.consider(&state, &stop).unwrap();
    archive.test_corrupt_best_objective();
    let before = snapshot(&archive);

    assert_eq!(archive.consider(&state, &stop), Err(ScheduleEliteError::InvalidMachineOrder));
    assert_eq!(snapshot(&archive), before);
}

#[test]
fn injected_mid_encode_and_mid_identity_interruptions_are_transactional() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&flow_shop(), &stop).unwrap().unwrap();
    let state = state(&problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    let mut archive = ScheduleEliteArchive::new();

    test_interrupt_after_checkpoints(2);
    assert_eq!(archive.consider(&state, &stop), Err(ScheduleEliteError::Interrupted));
    test_clear_interrupt();
    assert!(archive.is_empty());

    archive.consider(&state, &stop).unwrap();
    let before = snapshot(&archive);
    test_interrupt_after_checkpoints(2);
    assert_eq!(archive.consider(&state, &stop), Err(ScheduleEliteError::Interrupted));
    test_clear_interrupt();
    assert_eq!(snapshot(&archive), before);
}

#[test]
fn incompatible_problem_and_prearmed_interruption_leave_the_archive_unchanged() {
    let schedule = flow_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let original = state(&problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    let mut archive = ScheduleEliteArchive::new();
    archive.consider(&original, &stop).unwrap();
    let before = snapshot(&archive);

    let mut changed = schedule;
    changed.intervals[0].duration += 1;
    let changed_problem = JobShopProblem::recognize(&changed, &stop).unwrap().unwrap();
    let changed_state = state(&changed_problem, vec![vec![0, 2, 4], vec![1, 3, 5]]);
    assert_eq!(archive.consider(&changed_state, &stop).unwrap(), ScheduleEliteConsiderOutcome::Incompatible);
    assert_eq!(snapshot(&archive), before);

    let stopped = AtomicBool::new(true);
    assert_eq!(archive.consider(&original, &stopped), Err(ScheduleEliteError::Interrupted));
    assert_eq!(snapshot(&archive), before);
}
