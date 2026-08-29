use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use qayd::engines::ls::lists::move_acceptance::MinimizingMoveAcceptance;
use qayd::engines::ls::lists::schedule_state::{
    audit_strict_n5_layout, CriticalNeighborhood, DispatchRule, GifflerThompsonWorkspace, JobShopProblem, JobShopState, MachineArc,
    MoveOutcome, MoveProbe, MoveRejection, ScheduleMove, ScheduleStateMetrics,
};
use qayd::engines::ls::lists::{schedule_constructor_multistart_supported, solve_schedule, solve_schedule_capped_persistent_hybrid};
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

fn fixed_permutation_job_shop(jobs: usize, machines: usize, salt: usize) -> Schedule {
    let operation_count = jobs * machines;
    let durations = (0..operation_count)
        .map(|operation| 1 + i64::try_from((operation * 11 + salt * 7 + operation / machines) % 13).unwrap())
        .collect::<Vec<_>>();
    let horizon = durations.iter().sum();
    let intervals = durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect();
    let precedences =
        (0..jobs).flat_map(|job| (0..machines - 1).map(move |step| (job * machines + step, job * machines + step + 1))).collect();
    let mut assigned = vec![Vec::new(); machines];
    for job in 0..jobs {
        for step in 0..machines {
            let shift = (job + salt) % machines;
            let machine = if (job + salt).is_multiple_of(2) { (shift + step) % machines } else { (shift + machines - step) % machines };
            assigned[machine].push(job * machines + step);
        }
    }
    Schedule { intervals, precedences, resources: assigned.into_iter().map(Resource::NoOverlap).collect(), minimize_makespan: true }
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

fn independent_schedule_oracle(schedule: &Schedule, machine_sequences: &[Vec<usize>]) -> Option<(Vec<i64>, i64)> {
    let operation_count = schedule.intervals.len();
    let mut durations = Vec::with_capacity(operation_count);
    let mut windows = Vec::with_capacity(operation_count);
    for interval in &schedule.intervals {
        match interval.modes.as_slice() {
            [] => {
                durations.push(interval.duration);
                windows.push((0, interval.horizon.checked_sub(interval.duration)?));
            }
            [mode] => {
                durations.push(mode.duration);
                windows.push(mode.start_window);
            }
            _ => return None,
        }
    }

    let mut successors = vec![BTreeSet::new(); operation_count];
    for &(before, after) in &schedule.precedences {
        if before >= operation_count || after >= operation_count {
            return None;
        }
        successors[before].insert(after);
    }
    let mut seen = vec![false; operation_count];
    for sequence in machine_sequences {
        for &operation in sequence {
            if operation >= operation_count || seen[operation] {
                return None;
            }
            seen[operation] = true;
        }
        for pair in sequence.windows(2) {
            successors[pair[0]].insert(pair[1]);
        }
    }
    if seen.iter().any(|&present| !present) {
        return None;
    }

    let mut indegrees = vec![0usize; operation_count];
    for targets in &successors {
        for &target in targets {
            indegrees[target] = indegrees[target].checked_add(1)?;
        }
    }
    let mut ready = (0..operation_count).filter(|&operation| indegrees[operation] == 0).collect::<BTreeSet<_>>();
    let mut starts = windows.iter().map(|&(earliest, _)| earliest).collect::<Vec<_>>();
    let mut visited = 0usize;
    while let Some(operation) = ready.pop_first() {
        visited += 1;
        let end = starts[operation].checked_add(durations[operation])?;
        for &successor in &successors[operation] {
            starts[successor] = starts[successor].max(end);
            indegrees[successor] -= 1;
            if indegrees[successor] == 0 {
                ready.insert(successor);
            }
        }
    }
    if visited != operation_count {
        return None;
    }

    let mut makespan = 0i64;
    for operation in 0..operation_count {
        let end = starts[operation].checked_add(durations[operation])?;
        let (earliest, latest) = windows[operation];
        if starts[operation] < earliest || starts[operation] > latest || end > schedule.intervals[operation].horizon {
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

fn generated_general_dag(seed: u64, operation_count: usize, machine_count: usize) -> Schedule {
    fn sample(seed: u64, index: usize) -> u64 {
        let mut value = seed ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    let durations = (0..operation_count).map(|operation| i64::try_from(sample(seed, operation) % 6).unwrap()).collect::<Vec<_>>();
    let horizon = durations.iter().sum::<i64>().saturating_add(32);
    let intervals = durations
        .iter()
        .enumerate()
        .map(|(operation, &duration)| {
            let earliest = i64::try_from(sample(seed ^ 0xa5a5_a5a5_a5a5_a5a5, operation) % 5).unwrap();
            IntervalVar {
                duration: 0,
                horizon,
                modes: vec![Mode {
                    reference: Some(operation),
                    machine: usize::try_from(sample(seed ^ 0x1234_5678_9abc_def0, operation)).unwrap() % machine_count,
                    duration,
                    start_window: (earliest, horizon - duration),
                }],
                optional: false,
            }
        })
        .collect();
    let mut precedences = Vec::new();
    for before in 0..operation_count {
        for after in before + 1..operation_count {
            if sample(seed ^ 0xfedc_ba98_7654_3210, before * operation_count + after).is_multiple_of(5) {
                precedences.push((before, after));
            }
        }
    }
    Schedule { intervals, precedences, resources: vec![Resource::MachineNoOverlap], minimize_makespan: true }
}

#[test]
fn strict_job_shop_recognition_has_explicit_machine_assignments() {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&fixed_job_shop(), &stop).unwrap().expect("strict fixed JSSP");

    assert_eq!(problem.operation_count(), 4);
    assert_eq!(problem.machine_count(), 2);
    assert_eq!((0..4).map(|operation| problem.duration(operation)).collect::<Vec<_>>(), vec![3, 2, 2, 4]);
    assert_eq!((0..4).map(|operation| problem.machine(operation)).collect::<Vec<_>>(), vec![0, 1, 1, 0]);
    assert_eq!(problem.job_predecessors(1), &[0]);
    assert_eq!(problem.job_successors(2), &[3]);
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
fn bucketed_giffler_thompson_matches_the_frozen_scan_oracle_exhaustively() {
    let stop = AtomicBool::new(false);
    for operation_count in 1..=10 {
        for machine_count in 1..=4.min(operation_count) {
            for seed in 0..16u64 {
                let schedule = generated_general_dag(seed, operation_count, machine_count);
                let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().expect("generated strict job shop");
                for rule in DispatchRule::ALL {
                    let fast = JobShopState::giffler_thompson(&problem, seed, rule, &stop).unwrap();
                    let reference = JobShopState::giffler_thompson_reference(&problem, seed, rule, &stop).unwrap();
                    match (fast, reference) {
                        (Some(fast), Some(reference)) => {
                            assert_eq!(fast.machine_sequences(), reference.machine_sequences(), "sequence mismatch: n={operation_count}, m={machine_count}, seed={seed}, rule={rule:?}");
                            assert_eq!(fast.starts(), reference.starts(), "start mismatch: n={operation_count}, m={machine_count}, seed={seed}, rule={rule:?}");
                            assert_eq!(fast.makespan(), reference.makespan(), "makespan mismatch: n={operation_count}, m={machine_count}, seed={seed}, rule={rule:?}");
                        }
                        (None, None) => {}
                        (fast, reference) => panic!(
                            "feasibility mismatch: n={operation_count}, m={machine_count}, seed={seed}, rule={rule:?}, fast={}, reference={}",
                            fast.is_some(),
                            reference.is_some()
                        ),
                    }
                }
            }
        }
    }
}

#[test]
fn compact_multistart_workspace_matches_complete_earliest_start_states() {
    let stop = AtomicBool::new(false);
    for operation_count in 1..=10 {
        for machine_count in 1..=4.min(operation_count) {
            for seed in 0..24u64 {
                let schedule = generated_general_dag(seed, operation_count, machine_count);
                let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().expect("generated strict job shop");
                let complete = JobShopState::giffler_thompson(&problem, seed, DispatchRule::EarliestStart, &stop).unwrap();
                let mut workspace = GifflerThompsonWorkspace::new(&problem);
                let mut metrics = ScheduleStateMetrics::default();
                let compact = workspace.construct(&problem, seed, &stop, &mut metrics).unwrap();
                match (complete, compact) {
                    (Some(complete), Some((objective, _))) => {
                        let solution = workspace.to_solution(&problem, objective);
                        assert_eq!(solution.starts, complete.starts());
                        assert_eq!(solution.objectives, vec![complete.makespan()]);
                        assert_eq!(verify_collection_solution(&collection(schedule), &solution).unwrap(), solution.objectives);
                    }
                    (None, None) => {}
                    (complete, compact) => panic!(
                        "compact feasibility mismatch: n={operation_count}, m={machine_count}, seed={seed}, complete={}, compact={}",
                        complete.is_some(),
                        compact.is_some()
                    ),
                }
            }
        }
    }
}

#[test]
fn bucketed_giffler_thompson_matches_infeasibility_overflow_and_cancellation() {
    let stop = AtomicBool::new(false);
    let infeasible = Schedule {
        intervals: vec![
            IntervalVar { duration: 2, horizon: 2, modes: Vec::new(), optional: false },
            IntervalVar { duration: 2, horizon: 2, modes: Vec::new(), optional: false },
        ],
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&infeasible, &stop).unwrap().unwrap();
    for rule in DispatchRule::ALL {
        assert!(JobShopState::giffler_thompson(&problem, 7, rule, &stop).unwrap().is_none());
        assert!(JobShopState::giffler_thompson_reference(&problem, 7, rule, &stop).unwrap().is_none());
    }

    let overflow = Schedule {
        intervals: vec![
            IntervalVar { duration: i64::MAX - 1, horizon: i64::MAX, modes: Vec::new(), optional: false },
            IntervalVar { duration: i64::MAX - 1, horizon: i64::MAX, modes: Vec::new(), optional: false },
        ],
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&overflow, &stop).unwrap().unwrap();
    assert!(JobShopState::giffler_thompson(&problem, 3, DispatchRule::Randomized, &stop).unwrap().is_none());
    assert!(JobShopState::giffler_thompson_reference(&problem, 3, DispatchRule::Randomized, &stop).unwrap().is_none());

    let cancelled = AtomicBool::new(true);
    let problem = JobShopProblem::recognize(&fixed_job_shop(), &stop).unwrap().unwrap();
    assert!(JobShopState::giffler_thompson(&problem, 0, DispatchRule::EarliestStart, &cancelled).is_err());
    assert!(JobShopState::giffler_thompson_reference(&problem, 0, DispatchRule::EarliestStart, &cancelled).is_err());
}

#[test]
fn bucketed_giffler_heap_discards_stale_entries_and_rebuilds_at_a_fixed_bound() {
    let operation_count = 160;
    let schedule = Schedule {
        intervals: (0..operation_count).map(|_| IntervalVar { duration: 0, horizon: 0, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap((0..operation_count).collect())],
        minimize_makespan: true,
    };
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut observed_stale = false;
    let mut observed_rebuild = false;
    for seed in 0..16 {
        let state = JobShopState::giffler_thompson(&problem, seed, DispatchRule::Randomized, &stop).unwrap().unwrap();
        let reference = JobShopState::giffler_thompson_reference(&problem, seed, DispatchRule::Randomized, &stop).unwrap().unwrap();
        assert_eq!(state.machine_sequences(), reference.machine_sequences());
        let metrics = state.metrics();
        observed_stale |= metrics.construction_stale_pops > 0;
        observed_rebuild |= metrics.construction_heap_rebuilds > 0;
        assert!(metrics.construction_heap_pushes >= operation_count as u64);
        assert!(metrics.construction_bucket_visits >= operation_count as u64);
        assert!(metrics.construction_heap_peak <= (operation_count * 5 + 1) as u64);
    }
    assert!(observed_stale, "the lazy heap never exercised its stale-entry path");
    assert!(observed_rebuild, "the lazy heap never exercised its bounded rebuild path");
}

#[test]
fn bucketed_giffler_refreshes_far_less_work_than_the_scan_oracle() {
    let schedule = fixed_flow_shop(80, 80);
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let fast = JobShopState::giffler_thompson(&problem, 41, DispatchRule::MostWorkRemaining, &stop).unwrap().unwrap();
    let reference = JobShopState::giffler_thompson_reference(&problem, 41, DispatchRule::MostWorkRemaining, &stop).unwrap().unwrap();

    assert_eq!(fast.machine_sequences(), reference.machine_sequences());
    assert_eq!(fast.starts(), reference.starts());
    assert_eq!(fast.makespan(), reference.makespan());
    assert!(
        fast.metrics().construction_candidates.saturating_mul(5) < reference.metrics().construction_candidates,
        "dirty-machine refresh did not remove enough full-ready rescans: fast={}, scan={}",
        fast.metrics().construction_candidates,
        reference.metrics().construction_candidates
    );
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
fn local_move_estimate_uses_the_audited_span_formula_without_mutation() {
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
    let old_sequences = state.machine_sequences().to_vec();
    let old_starts = state.starts().to_vec();
    let _ = state.take_metrics();

    let estimate = state
        .estimate_move_local(ScheduleMove::Insert { machine: 0, from: 0, to: 3 }, &stop)
        .unwrap()
        .expect("valid insert has an estimate");

    assert_eq!(estimate.estimated_makespan, 10);
    assert_eq!(estimate.span_operations, 4);
    assert!(!estimate.acyclicity_certified, "a backward topological arc must remain unknown");
    assert_eq!(state.machine_sequences(), old_sequences);
    assert_eq!(state.starts(), old_starts);
    assert_eq!(state.metrics().local_move_estimates, 1);
    assert_eq!(state.metrics().local_move_certified, 0);
    assert_eq!(state.metrics().local_move_unknown, 1);
    assert_eq!(state.metrics().delta_evaluations, 0);
}

#[test]
fn direct_full_oracle_matches_the_independent_oracle_exhaustively() {
    let stop = AtomicBool::new(false);
    let schedule = fixed_job_shop();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    for machine_zero in permutations(&[0, 3]) {
        for machine_one in permutations(&[1, 2]) {
            let sequences = vec![machine_zero.clone(), machine_one.clone()];
            if independent_fixed_oracle(&schedule, &sequences).is_none() {
                continue;
            }
            for machine in 0..2 {
                for movement in [
                    ScheduleMove::AdjacentSwap { machine, first_position: 0 },
                    ScheduleMove::Insert { machine, from: 0, to: 1 },
                    ScheduleMove::Insert { machine, from: 1, to: 0 },
                ] {
                    let mut candidate_sequences = sequences.clone();
                    apply_test_move(&mut candidate_sequences[machine], movement);
                    let oracle = independent_fixed_oracle(&schedule, &candidate_sequences);
                    let mut state = JobShopState::from_machine_sequences(&problem, sequences.clone(), &stop).unwrap().unwrap();
                    let _ = state.take_metrics();
                    let outcome = state.consider_move_full_oracle(movement, MinimizingMoveAcceptance::Always, &stop).unwrap();
                    match (oracle, outcome) {
                        (Some((starts, objective)), MoveOutcome::Accepted { current, .. }) => {
                            assert_eq!(current, objective);
                            assert_eq!(state.starts(), starts);
                            assert_eq!(state.machine_sequences(), candidate_sequences);
                            assert!(state.matches_full_oracle(&stop).unwrap());
                        }
                        (None, MoveOutcome::Rejected(MoveRejection::Cycle)) => {
                            assert_eq!(state.machine_sequences(), sequences);
                        }
                        (oracle, outcome) => panic!("direct oracle disagreement: oracle={oracle:?}, outcome={outcome:?}"),
                    }
                    let metrics = state.metrics();
                    assert_eq!(metrics.moves_considered, 1);
                    assert_eq!(metrics.direct_oracle_attempts, 1);
                    assert_eq!(metrics.delta_evaluations, 0);
                }
            }
        }
    }
}

#[test]
fn direct_full_oracle_rejections_are_atomic_for_objective_window_and_cycle() {
    let stop = AtomicBool::new(false);
    let single = single_machine_schedule(4);
    let problem = JobShopProblem::recognize(&single, &stop).unwrap().unwrap();
    let mut objective_state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1, 2, 3]], &stop).unwrap().unwrap();
    let objective_sequences = objective_state.machine_sequences().to_vec();
    let objective_starts = objective_state.starts().to_vec();
    let objective = objective_state
        .consider_move_full_oracle(ScheduleMove::Insert { machine: 0, from: 0, to: 3 }, MinimizingMoveAcceptance::Improving, &stop)
        .unwrap();
    assert!(matches!(objective, MoveOutcome::Rejected(MoveRejection::NotAccepted { .. })));
    assert_eq!(objective_state.machine_sequences(), objective_sequences);
    assert_eq!(objective_state.starts(), objective_starts);
    assert_eq!(objective_state.metrics().direct_oracle_objective_rejections, 1);

    let window_schedule = Schedule {
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
    let problem = JobShopProblem::recognize(&window_schedule, &stop).unwrap().unwrap();
    let mut window_state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1]], &stop).unwrap().unwrap();
    let window = window_state
        .consider_move_full_oracle(ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 }, MinimizingMoveAcceptance::Always, &stop)
        .unwrap();
    assert_eq!(window, MoveOutcome::Rejected(MoveRejection::Window));
    assert_eq!(window_state.machine_sequences(), &[vec![0, 1]]);
    assert_eq!(window_state.metrics().direct_oracle_windows, 1);

    let cycle_schedule = fixed_job_shop();
    let problem = JobShopProblem::recognize(&cycle_schedule, &stop).unwrap().unwrap();
    let mut cycle_state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![1, 2]], &stop).unwrap().unwrap();
    let cycle = cycle_state
        .consider_move_full_oracle(ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 }, MinimizingMoveAcceptance::Always, &stop)
        .unwrap();
    assert_eq!(cycle, MoveOutcome::Rejected(MoveRejection::Cycle));
    assert_eq!(cycle_state.machine_sequences(), &[vec![0, 3], vec![1, 2]]);
    assert_eq!(cycle_state.metrics().direct_oracle_cycles, 1);
}

#[test]
fn zero_duration_direct_oracle_is_exact_and_uses_no_delta_probe() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: (0..4).map(|_| IntervalVar { duration: 0, horizon: 0, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1)],
        resources: vec![Resource::NoOverlap(vec![0, 1, 2, 3])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1, 2, 3]], &stop).unwrap().unwrap();
    let outcome = state
        .consider_move_full_oracle(ScheduleMove::Insert { machine: 0, from: 3, to: 2 }, MinimizingMoveAcceptance::Always, &stop)
        .unwrap();
    assert_eq!(outcome, MoveOutcome::Accepted { previous: 0, current: 0 });
    assert_eq!(state.metrics().delta_evaluations, 0);
    assert_eq!(state.metrics().direct_oracle_accepts, 1);
    assert!(state.matches_full_oracle(&stop).unwrap());

    let retained = state.machine_sequences().to_vec();
    let coincident_cycle = state
        .consider_move_full_oracle(ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 }, MinimizingMoveAcceptance::Always, &stop)
        .unwrap();
    assert_eq!(coincident_cycle, MoveOutcome::Rejected(MoveRejection::Cycle));
    assert_eq!(state.machine_sequences(), retained);
    assert_eq!(state.metrics().direct_oracle_cycles, 1);
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
fn all_zero_slack_machine_blocks_cover_multiple_critical_paths() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: (0..4).map(|_| IntervalVar { duration: 1, horizon: 4, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1]), Resource::NoOverlap(vec![2, 3])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1], vec![2, 3]], &stop).unwrap().unwrap();

    assert_eq!(state.critical_path(), &[2, 3], "the canonical path remains deterministic");
    assert_eq!(
        state
            .canonical_critical_blocks()
            .iter()
            .map(|block| (block.machine(), block.first_position(), block.last_position()))
            .collect::<Vec<_>>(),
        vec![(1, 0, 1)],
        "the historical generator must retain only its deterministic path block",
    );
    assert_eq!(
        state.critical_blocks().iter().map(|block| (block.machine(), block.first_position(), block.last_position())).collect::<Vec<_>>(),
        vec![(0, 0, 1), (1, 0, 1)],
        "both equal-length critical paths must contribute their machine block",
    );

    let plain = state.critical_moves(CriticalNeighborhood::N5, &stop).unwrap();
    let mut canonical = Vec::new();
    state.fill_canonical_critical_moves(CriticalNeighborhood::N5, &mut canonical, &stop).unwrap();
    assert_eq!(canonical, vec![ScheduleMove::AdjacentSwap { machine: 1, first_position: 0 }]);
    let mut scored = Vec::new();
    state.fill_scored_critical_moves(&[CriticalNeighborhood::N5], &mut scored, &stop).unwrap();
    assert_eq!(plain.iter().copied().collect::<BTreeSet<_>>(), scored.iter().map(|candidate| candidate.movement).collect());
    assert_eq!(scored[0].score, scored[1].score, "symmetric paths must produce an exact score tie");
    assert!(scored[0].movement < scored[1].movement, "move identity is the deterministic final tie break");
    assert_eq!(state.makespan(), 2);

    let retained = canonical.clone();
    let interrupted = AtomicBool::new(true);
    assert!(state.fill_canonical_critical_moves(CriticalNeighborhood::N6, &mut canonical, &interrupted).is_err());
    assert_eq!(canonical, retained, "prearmed interruption must preserve the caller buffer");

    state.retain_canonical_critical_blocks_only();
    assert!(state.critical_blocks().is_empty());
    assert_eq!(state.canonical_critical_blocks().len(), 1);
    assert!(matches!(
        state.consider_move(retained[0], MinimizingMoveAcceptance::Always, &stop),
        Ok(MoveOutcome::Accepted { previous: 2, current: 2 })
    ));
    assert!(state.critical_blocks().is_empty(), "baseline commits must skip all-critical materialization");
    assert_eq!(state.canonical_critical_blocks().len(), 1);
    assert!(state.matches_full_oracle(&stop).unwrap());
}

#[test]
fn head_tail_scoring_is_advisory_deterministic_and_interruptible() {
    let build_stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&fixed_job_shop(), &build_stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![1, 2]], &build_stop).unwrap().unwrap();
    let cycle = ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 };
    let sequences = state.machine_sequences().to_vec();
    let starts = state.starts().to_vec();

    let first = state.score_move_head_tail(cycle, &build_stop).unwrap().expect("valid structured move has a guidance score");
    let second = state.score_move_head_tail(cycle, &build_stop).unwrap().unwrap();
    assert_eq!(first, second);
    assert_eq!(state.machine_sequences(), sequences);
    assert_eq!(state.starts(), starts);
    assert_eq!(state.probe_move(cycle, &build_stop).unwrap(), MoveProbe::Rejected(MoveRejection::Cycle));

    let mut scored = Vec::new();
    state.fill_scored_critical_moves(&[CriticalNeighborhood::N5, CriticalNeighborhood::N6], &mut scored, &build_stop).unwrap();
    let retained = scored.clone();
    let stop = AtomicBool::new(true);
    assert!(state.score_move_head_tail(cycle, &stop).is_err());
    assert!(state.fill_scored_critical_moves(&[CriticalNeighborhood::N6], &mut scored, &stop).is_err());
    assert_eq!(scored, retained, "a prearmed interruption is observed before the caller buffer is cleared");
    assert_eq!(state.machine_sequences(), sequences);
    assert_eq!(state.starts(), starts);
}

#[test]
fn zero_duration_job_shop_keeps_exact_heads_tails_and_scores() {
    let stop = AtomicBool::new(false);
    let schedule = Schedule {
        intervals: (0..4).map(|_| IntervalVar { duration: 0, horizon: 0, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1]), Resource::NoOverlap(vec![2, 3])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().expect("zero duration is semantically supported");
    let mut state = JobShopState::giffler_thompson(&problem, 19, DispatchRule::EarliestStart, &stop).unwrap().unwrap();

    assert_eq!(state.makespan(), 0);
    assert_eq!(state.ends(), &[0, 0, 0, 0]);
    assert_eq!(state.duration(0), 0);
    assert!(state.job_predecessors(0).is_empty());
    assert!(state.job_successors(0).is_empty());
    assert_eq!(state.critical_blocks().len(), 2);
    let movements = state.critical_moves(CriticalNeighborhood::N5, &stop).unwrap();
    assert_eq!(movements.len(), 2);
    for movement in movements {
        let score = state.score_move_head_tail(movement, &stop).unwrap().unwrap();
        assert_eq!(score.max_added_arc_path, 0);
        assert!(matches!(state.probe_move(movement, &stop), Ok(MoveProbe::Feasible { current: 0, candidate: 0 })));
    }
    assert!(state.matches_full_oracle(&stop).unwrap());
    assert_eq!(verify_collection_solution(&collection(schedule), &state.to_solution()).unwrap(), vec![0]);
}

#[test]
fn sink_union_delta_makespan_matches_exhaustive_oracle() {
    let stop = AtomicBool::new(false);
    let schedule = fixed_job_shop();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    for machine_zero in permutations(&[0, 3]) {
        for machine_one in permutations(&[1, 2]) {
            let sequences = vec![machine_zero.clone(), machine_one.clone()];
            let Some(mut state) = JobShopState::from_machine_sequences(&problem, sequences.clone(), &stop).unwrap() else {
                continue;
            };
            for machine in 0..2 {
                for movement in [
                    ScheduleMove::AdjacentSwap { machine, first_position: 0 },
                    ScheduleMove::Insert { machine, from: 0, to: 1 },
                    ScheduleMove::Insert { machine, from: 1, to: 0 },
                ] {
                    let mut candidate_sequences = sequences.clone();
                    apply_test_move(&mut candidate_sequences[machine], movement);
                    match (independent_fixed_oracle(&schedule, &candidate_sequences), state.probe_move(movement, &stop).unwrap()) {
                        (Some((_, objective)), MoveProbe::Feasible { candidate, .. }) => assert_eq!(candidate, objective),
                        (None, MoveProbe::Rejected(MoveRejection::Cycle)) => {}
                        (oracle, probe) => panic!("sink-union/oracle disagreement: oracle={oracle:?}, probe={probe:?}"),
                    }
                    assert_eq!(state.machine_sequences(), sequences);
                    assert!(state.matches_full_oracle(&stop).unwrap());
                }
            }
        }
    }

    let coincident = Schedule {
        intervals: [1, 0, 2].into_iter().map(|duration| IntervalVar { duration, horizon: 3, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1)],
        resources: vec![Resource::NoOverlap(vec![0, 1, 2])],
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&coincident, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1, 2]], &stop).unwrap().unwrap();
    assert!(matches!(
        state.probe_move(ScheduleMove::Insert { machine: 0, from: 2, to: 1 }, &stop),
        Ok(MoveProbe::Feasible { candidate: 3, .. })
    ));
    assert_eq!(
        state.probe_move(ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 }, &stop).unwrap(),
        MoveProbe::Rejected(MoveRejection::Cycle),
    );
}

#[test]
fn critical_move_buffers_do_not_reserve_for_unrelated_operations() {
    let stop = AtomicBool::new(false);
    let operation_count = 5_000;
    let schedule = Schedule {
        intervals: (0..operation_count).map(|_| IntervalVar { duration: 1, horizon: 1, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: Vec::new(),
        minimize_makespan: true,
    };
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let sequences = (0..operation_count).map(|operation| vec![operation]).collect();
    let state = JobShopState::from_machine_sequences(&problem, sequences, &stop).unwrap().unwrap();
    assert!(state.critical_blocks().is_empty());

    let mut plain = Vec::new();
    let mut scored = Vec::new();
    state.fill_critical_moves(CriticalNeighborhood::N6, &mut plain, &stop).unwrap();
    state.fill_scored_critical_moves(&[CriticalNeighborhood::N6], &mut scored, &stop).unwrap();
    assert!(plain.is_empty());
    assert!(scored.is_empty());
    assert!(plain.capacity() <= 4_096);
    assert!(scored.capacity() <= 4_096);
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

    stop.store(false, std::sync::atomic::Ordering::Release);
    let direct = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(1));
            stop.store(true, std::sync::atomic::Ordering::Release);
        });
        state.consider_move_full_oracle(
            ScheduleMove::Insert { machine: 0, from: 0, to: operation_count - 1 },
            MinimizingMoveAcceptance::Always,
            &stop,
        )
    });
    assert!(direct.is_err());
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

#[test]
fn constructor_multistart_ordinal_six_is_the_historical_worker_six_start() {
    let schedule = fixed_flow_shop(12, 9);
    let model = collection(schedule.clone());
    let stop = AtomicBool::new(false);
    for seed in [0, 1, 0x1234_5678_9abc_def0] {
        let (historical, _, _) = solve_schedule_capped_persistent_hybrid(
            &schedule,
            seed,
            seed,
            6,
            7,
            &stop,
            0,
            None,
            None,
            false,
            None,
            false,
            false,
            None,
            &mut |_| {},
        );
        let mut reports = Vec::new();
        let (sampled, metrics, _) = solve_schedule_capped_persistent_hybrid(
            &schedule,
            seed,
            seed,
            6,
            7,
            &stop,
            64,
            None,
            None,
            false,
            None,
            false,
            true,
            None,
            &mut |objective| reports.push(objective),
        );

        assert_eq!(sampled.starts, historical.starts);
        assert_eq!(sampled.objectives, historical.objectives);
        assert_eq!(verify_collection_solution(&model, &sampled).unwrap(), sampled.objectives);
        assert!(reports.is_empty(), "constructor candidates must cross the orchestrator replay before publication");
        assert_eq!(metrics.schedule_constructor_multistart_attempts, 1);
        assert_eq!(metrics.schedule_constructor_multistart_constructions, 1);
        assert_eq!(metrics.schedule_constructor_multistart_feasible, 1);
        assert_eq!(metrics.schedule_constructor_multistart_best_ordinal, Some(6));
        assert_eq!(metrics.schedule_constructor_multistart_best_seed, Some(seed));
        assert_eq!(metrics.schedule_constructor_multistart_next_ordinal, Some(7));
    }
}

#[test]
fn constructor_multistart_preserves_credit_across_slices_and_caps_each_boundary() {
    let schedule = fixed_flow_shop(10, 8);
    let stop = AtomicBool::new(false);
    let seed = 91;
    let (first, first_metrics, first_session) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        1,
        None,
        None,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    assert!(!first.feasible);
    assert_eq!(first_metrics.work_steps, 1);
    assert_eq!(first_metrics.schedule_constructor_multistart_attempts, 0);
    let (split, split_metrics, _) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        63,
        None,
        first_session,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    let (monolithic, monolithic_metrics, _) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        64,
        None,
        None,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    assert_eq!(split.starts, monolithic.starts);
    assert_eq!(split.objectives, monolithic.objectives);
    assert_eq!(split_metrics.work_steps, 63);
    assert_eq!(monolithic_metrics.work_steps, 64);

    let (four, four_metrics, four_session) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        512,
        None,
        None,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    assert!(four.feasible);
    assert_eq!(four_metrics.work_steps, 256);
    assert_eq!(four_metrics.schedule_constructor_multistart_attempts, 4);
    assert_eq!(four_metrics.schedule_constructor_multistart_attempts, four_metrics.schedule_constructor_multistart_constructions);
    assert_eq!(four_metrics.schedule_constructor_multistart_constructions, four_metrics.schedule_constructor_multistart_feasible);
    assert_eq!(
        four_metrics.schedule_constructor_multistart_feasible,
        four_metrics.schedule_constructor_multistart_distinct_fingerprints
            + four_metrics.schedule_constructor_multistart_other_fingerprint_observations
    );
    assert_eq!(four_metrics.schedule_constructor_multistart_next_ordinal, Some(10));
    assert_eq!(four_metrics.schedule_constructor_multistart_distinct_fingerprints, 2);
    let (eight, eight_metrics, _) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        256,
        None,
        four_session,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    assert!(eight.objectives[0] <= four.objectives[0]);
    assert_eq!(eight_metrics.schedule_constructor_multistart_next_ordinal, Some(14));
    assert!(eight_metrics.schedule_constructor_multistart_workspace_peak_bytes > 0);
}

#[test]
fn constructor_multistart_keeps_a_complete_best_when_a_resumed_slice_is_stopped() {
    let schedule = fixed_flow_shop(10, 8);
    let running = AtomicBool::new(false);
    let (best, _, session) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        97,
        97,
        6,
        7,
        &running,
        64,
        None,
        None,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    let stopped = AtomicBool::new(true);
    let (retained, metrics, _) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        97,
        97,
        6,
        7,
        &stopped,
        256,
        None,
        session,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    assert_eq!(retained.starts, best.starts);
    assert_eq!(retained.objectives, best.objectives);
    assert_eq!(metrics.work_steps, 0);
    assert_eq!(metrics.schedule_constructor_multistart_attempts, 0);
    assert_eq!(metrics.schedule_constructor_multistart_next_ordinal, Some(7));
}

#[test]
fn constructor_multistart_is_strictly_gated_and_prearmed_stop_does_no_work() {
    let schedule = fixed_flow_shop(9, 7);
    let running = AtomicBool::new(false);
    for worker in 0..6 {
        let (off, off_metrics, _) = solve_schedule_capped_persistent_hybrid(
            &schedule,
            47,
            47,
            worker,
            7,
            &running,
            96,
            None,
            None,
            false,
            None,
            false,
            false,
            None,
            &mut |_| {},
        );
        let (requested, requested_metrics, _) = solve_schedule_capped_persistent_hybrid(
            &schedule,
            47,
            47,
            worker,
            7,
            &running,
            96,
            None,
            None,
            false,
            None,
            false,
            true,
            None,
            &mut |_| {},
        );
        assert_eq!(requested.starts, off.starts, "worker {worker} trajectory changed");
        assert_eq!(requested.objectives, off.objectives, "worker {worker} objective changed");
        assert_eq!(requested_metrics.work_steps, off_metrics.work_steps);
        assert_eq!(requested_metrics.schedule_constructor_multistart_owner_worker_mask, 0);
    }

    let stopped = AtomicBool::new(true);
    let (solution, metrics, session) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        47,
        47,
        6,
        7,
        &stopped,
        256,
        None,
        None,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    assert!(!solution.feasible);
    assert!(session.is_none(), "recognition must fail closed when cancellation is already armed");
    assert_eq!(metrics.work_steps, 0);
    assert_eq!(metrics.schedule_constructor_multistart_attempts, 0);
}

#[test]
fn constructor_multistart_diversity_is_global_bounded_and_conservative() {
    let schedule = Schedule {
        intervals: (0..6).map(|_| IntervalVar { duration: 1, horizon: 6, modes: Vec::new(), optional: false }).collect(),
        precedences: (0..5).map(|operation| (operation, operation + 1)).collect(),
        resources: vec![Resource::NoOverlap((0..6).collect())],
        minimize_makespan: true,
    };
    let stop = AtomicBool::new(false);
    let (_, first, session) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        53,
        53,
        6,
        7,
        &stop,
        64,
        None,
        None,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    let (_, second, _) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        53,
        53,
        6,
        7,
        &stop,
        64,
        None,
        session,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    assert_eq!(first.schedule_constructor_multistart_distinct_fingerprints, 1);
    assert_eq!(first.schedule_constructor_multistart_other_fingerprint_observations, 0);
    assert_eq!(second.schedule_constructor_multistart_distinct_fingerprints, 0);
    assert_eq!(second.schedule_constructor_multistart_other_fingerprint_observations, 1);
}

#[test]
fn constructor_multistart_distinguishes_failure_from_interruption_and_advances() {
    let schedule = Schedule {
        intervals: vec![
            IntervalVar { duration: 2, horizon: 2, modes: Vec::new(), optional: false },
            IntervalVar { duration: 2, horizon: 2, modes: Vec::new(), optional: false },
        ],
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1])],
        minimize_makespan: true,
    };
    let stop = AtomicBool::new(false);
    let (solution, metrics, _) = solve_schedule_capped_persistent_hybrid(
        &schedule,
        59,
        59,
        6,
        7,
        &stop,
        64,
        None,
        None,
        false,
        None,
        false,
        true,
        None,
        &mut |_| {},
    );
    assert!(!solution.feasible);
    assert_eq!(metrics.schedule_constructor_multistart_attempts, 1);
    assert_eq!(metrics.schedule_constructor_multistart_failures, 1);
    assert_eq!(metrics.schedule_constructor_multistart_interruptions, 0);
    assert_eq!(metrics.schedule_constructor_multistart_next_ordinal, Some(7));
}

#[test]
fn constructor_multistart_support_sentinel_rejects_duplicate_machine_assignments_and_bad_dags() {
    let stop = AtomicBool::new(false);
    assert!(schedule_constructor_multistart_supported(&fixed_job_shop(), &stop));

    let mut duplicate = fixed_job_shop();
    duplicate.resources.push(Resource::NoOverlap(vec![0]));
    assert!(!schedule_constructor_multistart_supported(&duplicate, &stop));

    let mut invalid_index = fixed_job_shop();
    invalid_index.precedences.push((0, invalid_index.intervals.len()));
    assert!(!schedule_constructor_multistart_supported(&invalid_index, &stop));

    let mut cycle = fixed_job_shop();
    cycle.precedences.extend([(1, 2), (3, 0)]);
    assert!(!schedule_constructor_multistart_supported(&cycle, &stop));
}

#[test]
fn strict_tsab_n5_uses_head_internal_and_tail_exclusions() {
    assert!(audit_strict_n5_layout(&[4]).is_empty(), "a sole critical block contributes no strict N5 move");
    assert_eq!(
        audit_strict_n5_layout(&[4, 3, 5]),
        vec![
            ScheduleMove::AdjacentSwap { machine: 0, first_position: 2 },
            ScheduleMove::AdjacentSwap { machine: 1, first_position: 4 },
            ScheduleMove::AdjacentSwap { machine: 1, first_position: 5 },
            ScheduleMove::AdjacentSwap { machine: 2, first_position: 7 },
        ]
    );
    assert_eq!(
        audit_strict_n5_layout(&[2, 2, 2]),
        vec![
            ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 },
            ScheduleMove::AdjacentSwap { machine: 1, first_position: 2 },
            ScheduleMove::AdjacentSwap { machine: 2, first_position: 4 },
        ],
        "an internal length-two block is deduplicated"
    );
}

#[test]
fn strict_n5_fast_kernel_matches_independent_and_full_oracles_exhaustively() {
    let stop = AtomicBool::new(false);
    let mut exercised = 0usize;
    for jobs in 2..=4 {
        for machines in 2..=4 {
            let schedules = std::iter::once(fixed_flow_shop(jobs, machines))
                .chain((0..machines).map(|salt| fixed_permutation_job_shop(jobs, machines, salt)));
            for schedule in schedules {
                let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
                for seed in 0..24u64 {
                    let Some(mut reference) = JobShopState::giffler_thompson(&problem, seed, DispatchRule::Randomized, &stop).unwrap()
                    else {
                        continue;
                    };
                    reference.retain_canonical_critical_blocks_only();
                    let mut movements = Vec::new();
                    reference
                        .visit_strict_n5_canonical_moves(&stop, |movement, _| {
                            movements.push(movement);
                            Ok(())
                        })
                        .unwrap();
                    for movement in movements {
                        let sequences = reference.machine_sequences().to_vec();
                        let mut expected_sequences = sequences.clone();
                        let machine = match movement {
                            ScheduleMove::AdjacentSwap { machine, .. } | ScheduleMove::Insert { machine, .. } => machine,
                        };
                        apply_test_move(&mut expected_sequences[machine], movement);
                        let oracle = independent_fixed_oracle(&schedule, &expected_sequences).expect("strict N5 must preserve acyclicity");
                        let mut state = JobShopState::from_machine_sequences(&problem, sequences, &stop).unwrap().unwrap();
                        state.retain_canonical_critical_blocks_only();
                        let fast = state.consider_strict_n5_fast(movement, MinimizingMoveAcceptance::Always, &stop).unwrap();
                        assert!(fast.used_fast_path, "classical positive default-window JSSP must use the strict fast gate");
                        assert!(!fast.fell_back);
                        assert!(matches!(fast.outcome, MoveOutcome::Accepted { .. }));
                        assert_eq!(state.machine_sequences(), expected_sequences);
                        assert_eq!(state.starts(), oracle.0);
                        assert_eq!(state.makespan(), oracle.1);
                        assert!(
                            state.refresh_from_full_oracle(&stop).unwrap(),
                            "incremental dates, tails and canonical path must match full replay"
                        );
                        assert!(state.matches_full_oracle(&stop).unwrap());
                        exercised += 1;
                    }
                }
            }
        }
    }
    assert!(exercised >= 64, "fixture sweep did not exercise enough strict N5 transitions: {exercised}");
}

#[test]
fn strict_n5_fast_gate_falls_back_for_general_dag_zero_duration_and_windows() {
    let stop = AtomicBool::new(false);
    let mut schedules = Vec::new();

    let mut general_dag = fixed_flow_shop(3, 3);
    general_dag.precedences.push((0, 5));
    schedules.push(general_dag);

    let mut zero_duration = fixed_flow_shop(3, 3);
    zero_duration.intervals[0].duration = 0;
    schedules.push(zero_duration);

    schedules.push(Schedule {
        intervals: vec![
            IntervalVar {
                duration: 0,
                horizon: 20,
                modes: vec![Mode { reference: None, machine: 0, duration: 3, start_window: (1, 17) }],
                optional: false,
            },
            IntervalVar {
                duration: 0,
                horizon: 20,
                modes: vec![Mode { reference: None, machine: 0, duration: 2, start_window: (0, 18) }],
                optional: false,
            },
        ],
        precedences: Vec::new(),
        resources: vec![Resource::MachineNoOverlap],
        minimize_makespan: true,
    });

    for schedule in schedules {
        let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
        let mut state = JobShopState::giffler_thompson(&problem, 0, DispatchRule::EarliestStart, &stop).unwrap().unwrap();
        assert!(!state.supports_strict_n5_fast_path());
        let movement = ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 };
        let before = state.machine_sequences().to_vec();
        let outcome = state.consider_strict_n5_fast(movement, MinimizingMoveAcceptance::Always, &stop).unwrap();
        assert!(!outcome.used_fast_path);
        assert!(outcome.fell_back);
        if matches!(outcome.outcome, MoveOutcome::Rejected(_)) {
            assert_eq!(state.machine_sequences(), before, "fallback rejection must remain atomic");
        }
        assert!(state.matches_full_oracle(&stop).unwrap());
    }
}

#[test]
fn strict_n5_fast_fallback_matches_independent_oracle_across_general_dags_windows_and_zero_durations() {
    let stop = AtomicBool::new(false);
    let mut exercised = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut saw_zero_duration = false;
    let mut saw_nontrivial_window = false;

    for operation_count in 3..=8 {
        for machine_count in 1..=4.min(operation_count) {
            for seed in 0..32u64 {
                let schedule = generated_general_dag(seed, operation_count, machine_count);
                saw_zero_duration |= schedule.intervals.iter().any(|interval| interval.modes[0].duration == 0);
                saw_nontrivial_window |= schedule.intervals.iter().any(|interval| interval.modes[0].start_window.0 != 0);
                let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().expect("generated general job shop");
                let Some(reference) = JobShopState::giffler_thompson(&problem, seed, DispatchRule::Randomized, &stop).unwrap() else {
                    continue;
                };
                if reference.supports_strict_n5_fast_path() {
                    continue;
                }
                let initial_sequences = reference.machine_sequences().to_vec();
                for machine in 0..initial_sequences.len() {
                    for first_position in 0..initial_sequences[machine].len().saturating_sub(1) {
                        let movement = ScheduleMove::AdjacentSwap { machine, first_position };
                        let mut candidate_sequences = initial_sequences.clone();
                        apply_test_move(&mut candidate_sequences[machine], movement);
                        let oracle = independent_schedule_oracle(&schedule, &candidate_sequences);
                        let mut state = JobShopState::from_machine_sequences(&problem, initial_sequences.clone(), &stop)
                            .unwrap()
                            .expect("constructor-produced order remains reconstructible");
                        let outcome = state.consider_strict_n5_fast(movement, MinimizingMoveAcceptance::Always, &stop).unwrap();
                        assert!(!outcome.used_fast_path);
                        assert!(outcome.fell_back);
                        match (oracle, outcome.outcome) {
                            (Some((starts, makespan)), MoveOutcome::Accepted { current, .. }) => {
                                assert_eq!(current, makespan);
                                assert_eq!(state.machine_sequences(), candidate_sequences);
                                assert_eq!(state.starts(), starts);
                                accepted += 1;
                            }
                            (None, MoveOutcome::Rejected(MoveRejection::Cycle | MoveRejection::Window | MoveRejection::Numeric)) => {
                                assert_eq!(state.machine_sequences(), initial_sequences);
                                rejected += 1;
                            }
                            (oracle, outcome) => panic!(
                                "fast fallback/oracle disagreement: n={operation_count}, m={machine_count}, seed={seed}, move={movement:?}, oracle={oracle:?}, outcome={outcome:?}"
                            ),
                        }
                        assert!(state.matches_full_oracle(&stop).unwrap());
                        exercised += 1;
                    }
                }
            }
        }
    }

    assert!(saw_zero_duration && saw_nontrivial_window);
    assert!(accepted > 0 && rejected > 0);
    assert!(exercised >= 128, "general fallback sweep exercised only {exercised} moves");
}

#[test]
fn strict_n5_fast_rejection_restores_graph_dates_and_tails() {
    let stop = AtomicBool::new(false);
    let schedule = fixed_flow_shop(6, 5);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::giffler_thompson(&problem, 11, DispatchRule::Randomized, &stop).unwrap().unwrap();
    state.retain_canonical_critical_blocks_only();
    let mut movement = None;
    state
        .visit_strict_n5_canonical_moves(&stop, |candidate, _| {
            movement.get_or_insert(candidate);
            Ok(())
        })
        .unwrap();
    let movement = movement.expect("fixture must expose a strict N5 movement");
    let before_sequences = state.machine_sequences().to_vec();
    let before_starts = state.starts().to_vec();
    let before_makespan = state.makespan();
    let rejected = state.consider_strict_n5_fast(movement, MinimizingMoveAcceptance::StrictlyBelow(i64::MIN), &stop).unwrap();
    assert!(rejected.used_fast_path);
    assert!(matches!(rejected.outcome, MoveOutcome::Rejected(MoveRejection::NotAccepted { .. })));
    assert_eq!(state.machine_sequences(), before_sequences);
    assert_eq!(state.starts(), before_starts);
    assert_eq!(state.makespan(), before_makespan);
    assert!(state.matches_full_oracle(&stop).unwrap());
}
