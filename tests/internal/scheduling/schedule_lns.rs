use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use qayd::engines::ls::lists::schedule_lns::{
    repair_critical_window, repair_critical_window_with_workspace, test_selected_segment_shapes, ScheduleLnsConfig, ScheduleLnsMetrics,
    ScheduleLnsMode, ScheduleLnsTestPhase, ScheduleLnsWorkspace,
};
use qayd::engines::ls::lists::schedule_state::{JobShopProblem, JobShopState};
use qayd::model::list::{verify_collection_solution, CollectionModel, IntervalVar, Resource, Schedule};

fn repairable_job_shop() -> Schedule {
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

fn poor_state(schedule: &Schedule) -> JobShopState {
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(schedule, &stop).unwrap().unwrap();
    JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![1, 2]], &stop).unwrap().unwrap()
}

fn apply_config() -> ScheduleLnsConfig {
    ScheduleLnsConfig {
        target_operations: 2,
        max_operations: 2,
        local_budget: Duration::from_millis(250),
        verification_reserve: Duration::from_millis(50),
        seed: 7,
        mode: ScheduleLnsMode::Apply,
        test_delay_phase: None,
        test_phase_delay: Duration::ZERO,
    }
}

const TEST_PHASES: [ScheduleLnsTestPhase; 7] = [
    ScheduleLnsTestPhase::Preparation,
    ScheduleLnsTestPhase::Selection,
    ScheduleLnsTestPhase::Exact,
    ScheduleLnsTestPhase::Splice,
    ScheduleLnsTestPhase::Reconstruction,
    ScheduleLnsTestPhase::Oracle,
    ScheduleLnsTestPhase::Publication,
];

#[test]
fn bounded_repair_returns_only_a_strict_oracle_verified_improvement() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    assert_eq!(state.makespan(), 11);
    let stop = AtomicBool::new(false);
    let mut metrics = ScheduleLnsMetrics::default();

    let result = repair_critical_window(&state, apply_config(), &mut metrics, &stop).unwrap();
    let candidate = result.candidate.expect("the selected critical block has an improving order");

    assert_eq!(result.observed_makespan, Some(7));
    assert_eq!(candidate.makespan(), 7);
    assert!(candidate.makespan() < state.makespan());
    assert_eq!(candidate.machine_sequences()[0], state.machine_sequences()[0], "the exterior machine order stays frozen");
    assert_eq!(candidate.machine_sequences()[1], [2, 1]);
    assert_eq!(verify_collection_solution(&collection(schedule), &candidate.to_solution()).unwrap(), vec![7]);
    assert_eq!(metrics.attempts, 1);
    assert_eq!(metrics.feasible, 1);
    assert_eq!(metrics.reconstructed, 1);
    assert_eq!(metrics.improvements, 1);
    assert_eq!(metrics.oracle_rejections, 0);
}

#[test]
fn repair_is_deterministic_for_a_fixed_seed_and_budget() {
    let schedule = repairable_job_shop();
    let first = poor_state(&schedule);
    let second = poor_state(&schedule);
    let stop = AtomicBool::new(false);
    let mut first_metrics = ScheduleLnsMetrics::default();
    let mut second_metrics = ScheduleLnsMetrics::default();

    let first = repair_critical_window(&first, apply_config(), &mut first_metrics, &stop).unwrap().candidate.unwrap();
    let second = repair_critical_window(&second, apply_config(), &mut second_metrics, &stop).unwrap().candidate.unwrap();

    assert_eq!(first.machine_sequences(), second.machine_sequences());
    assert_eq!(first.starts(), second.starts());
    assert_eq!(first.makespan(), second.makespan());
}

#[test]
fn shadow_mode_audits_an_improvement_without_returning_it() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);
    let mut metrics = ScheduleLnsMetrics::default();
    let mut config = apply_config();
    config.mode = ScheduleLnsMode::Shadow;

    let result = repair_critical_window(&state, config, &mut metrics, &stop).unwrap();

    assert!(result.candidate.is_none());
    assert_eq!(result.observed_makespan, Some(7));
    assert_eq!(metrics.improvements, 1);
    assert_eq!(metrics.shadow_improvements, 1);
    let expected_gain = u64::try_from(state.makespan() - 7).unwrap();
    assert_eq!(metrics.shadow_improvement_sum, expected_gain);
    assert_eq!(metrics.shadow_best_improvement, expected_gain);
}

#[test]
fn aggregated_shadow_metrics_sum_work_and_keep_the_best_gain() {
    let mut aggregate = ScheduleLnsMetrics {
        attempts: 1,
        shadow_improvements: 1,
        shadow_improvement_sum: 3,
        shadow_best_improvement: 3,
        elapsed_micros: 5,
        ..ScheduleLnsMetrics::default()
    };
    aggregate.add(ScheduleLnsMetrics {
        attempts: 1,
        shadow_improvements: 1,
        shadow_improvement_sum: 7,
        shadow_best_improvement: 7,
        elapsed_micros: 11,
        ..ScheduleLnsMetrics::default()
    });

    assert_eq!(aggregate.attempts, 2);
    assert_eq!(aggregate.shadow_improvements, 2);
    assert_eq!(aggregate.shadow_improvement_sum, 10);
    assert_eq!(aggregate.shadow_best_improvement, 7);
    assert_eq!(aggregate.elapsed_micros, 16);
}

#[test]
fn prearmed_parent_interruption_never_starts_or_publishes_a_repair() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(true);
    let mut metrics = ScheduleLnsMetrics::default();

    assert!(repair_critical_window(&state, apply_config(), &mut metrics, &stop).is_err());
    assert_eq!(metrics.attempts, 0);
    assert_eq!(metrics.interruptions, 1);
    assert_eq!(metrics.improvements, 0);
}

#[test]
fn zero_local_budget_is_a_bounded_timeout_without_a_candidate() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);
    let mut metrics = ScheduleLnsMetrics::default();
    let mut config = apply_config();
    config.local_budget = Duration::ZERO;
    config.verification_reserve = Duration::ZERO;
    let started = Instant::now();

    let result = repair_critical_window(&state, config, &mut metrics, &stop).unwrap();

    assert!(result.candidate.is_none());
    assert!(result.timed_out);
    assert_eq!(metrics.attempts, 1);
    assert_eq!(metrics.timeouts, 1);
    assert_eq!(metrics.improvements, 0);
    assert!(started.elapsed() < Duration::from_millis(100));
}

#[test]
fn total_timeout_during_exact_never_publishes_an_apply_candidate() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);
    let mut metrics = ScheduleLnsMetrics::default();
    let mut config = apply_config();
    config.local_budget = Duration::from_millis(15);
    config.verification_reserve = Duration::ZERO;
    config.test_delay_phase = Some(ScheduleLnsTestPhase::Exact);
    config.test_phase_delay = Duration::from_millis(50);

    let result = repair_critical_window(&state, config, &mut metrics, &stop).unwrap();

    assert!(result.timed_out);
    assert!(result.candidate.is_none());
    assert_eq!(metrics.timeouts, 1);
}

#[test]
fn total_timeout_after_exact_never_enters_apply_publication() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);
    let mut metrics = ScheduleLnsMetrics::default();
    let mut config = apply_config();
    config.local_budget = Duration::from_millis(100);
    config.verification_reserve = Duration::from_millis(50);
    config.test_delay_phase = Some(ScheduleLnsTestPhase::Splice);
    config.test_phase_delay = Duration::from_millis(200);

    let result = repair_critical_window(&state, config, &mut metrics, &stop).unwrap();

    assert!(result.timed_out);
    assert!(result.candidate.is_none());
    assert_eq!(metrics.feasible, 1, "the exact outcome existed before the total timeout");
}

#[test]
fn total_timeout_after_full_oracle_suppresses_apply_candidate() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);
    let mut metrics = ScheduleLnsMetrics::default();
    let mut config = apply_config();
    config.local_budget = Duration::from_millis(100);
    config.verification_reserve = Duration::from_millis(50);
    config.test_delay_phase = Some(ScheduleLnsTestPhase::Publication);
    config.test_phase_delay = Duration::from_millis(200);

    let result = repair_critical_window(&state, config, &mut metrics, &stop).unwrap();

    assert!(result.timed_out);
    assert!(result.candidate.is_none());
    assert_eq!(metrics.reconstructed, 1);
    assert_eq!(metrics.improvements, 0, "an expired candidate is not accounted as publishable");
}

#[test]
fn concurrent_parent_cancellation_is_returned_as_an_error() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);
    let mut metrics = ScheduleLnsMetrics::default();
    let mut config = apply_config();
    config.test_delay_phase = Some(ScheduleLnsTestPhase::Exact);
    config.test_phase_delay = Duration::from_millis(100);

    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(10));
            stop.store(true, Ordering::Release);
        });
        repair_critical_window(&state, config, &mut metrics, &stop)
    });

    assert!(result.is_err());
    assert_eq!(metrics.interruptions, 1);
    assert_eq!(metrics.timeouts, 0);
}

#[test]
fn every_repair_phase_obeys_the_total_deadline_and_fails_closed() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);

    for phase in TEST_PHASES {
        let mut metrics = ScheduleLnsMetrics::default();
        let mut config = apply_config();
        config.local_budget = Duration::from_millis(20);
        config.verification_reserve = Duration::ZERO;
        config.test_delay_phase = Some(phase);
        config.test_phase_delay = Duration::from_millis(75);

        let result = repair_critical_window(&state, config, &mut metrics, &stop).unwrap();

        assert!(result.timed_out, "{phase:?} must observe the total deadline");
        assert!(result.candidate.is_none(), "{phase:?} must not publish after the total deadline");
        assert_eq!(metrics.timeouts, 1, "{phase:?} must be accounted once");
        assert_eq!(metrics.improvements, 0, "{phase:?} must not account an expired candidate as an improvement");
    }
}

#[test]
fn every_repair_phase_obeys_parent_cancellation() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);

    for phase in TEST_PHASES {
        let stop = AtomicBool::new(false);
        let mut metrics = ScheduleLnsMetrics::default();
        let mut config = apply_config();
        config.local_budget = Duration::from_secs(1);
        config.verification_reserve = Duration::ZERO;
        config.test_delay_phase = Some(phase);
        config.test_phase_delay = Duration::from_millis(100);

        let result = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(5));
                stop.store(true, Ordering::Release);
            });
            repair_critical_window(&state, config, &mut metrics, &stop)
        });

        assert!(result.is_err(), "{phase:?} must return parent cancellation");
        assert_eq!(metrics.interruptions, 1, "{phase:?} must be accounted once");
        assert_eq!(metrics.timeouts, 0, "{phase:?} parent cancellation is not a local timeout");
        assert_eq!(metrics.improvements, 0, "{phase:?} must not account a cancelled candidate as an improvement");
    }
}

fn two_critical_sinks_schedule() -> Schedule {
    Schedule {
        intervals: (0..4).map(|_| IntervalVar { duration: 2, horizon: 8, modes: Vec::new(), optional: false }).collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap(vec![0, 1]), Resource::NoOverlap(vec![2, 3])],
        minimize_makespan: true,
    }
}

#[test]
fn bounded_selection_covers_multiple_critical_blocks_and_sinks() {
    let schedule = two_critical_sinks_schedule();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 1], vec![2, 3]], &stop).unwrap().unwrap();
    assert_eq!(state.makespan(), 4);
    assert!(state.critical_blocks().len() >= 2);
    let mut config = apply_config();
    config.target_operations = 4;
    config.max_operations = 4;
    let mut metrics = ScheduleLnsMetrics::default();
    let mut workspace = ScheduleLnsWorkspace::default();

    let result = repair_critical_window_with_workspace(&state, config, &mut metrics, &mut workspace, &stop).unwrap();

    assert!(result.candidate.is_none());
    assert_eq!(result.selected_operations, 4);
    assert_eq!(workspace.selected_segment_count(), 2);
    assert_eq!(workspace.selected_machine_count(), 2);
}

#[test]
fn canonical_critical_blocks_keep_baseline_states_repairable() {
    let schedule = repairable_job_shop();
    let mut state = poor_state(&schedule);
    assert!(!state.canonical_critical_blocks().is_empty());
    state.retain_canonical_critical_blocks_only();
    assert!(state.critical_blocks().is_empty());
    let stop = AtomicBool::new(false);
    let mut metrics = ScheduleLnsMetrics::default();

    let result = repair_critical_window(&state, apply_config(), &mut metrics, &stop).unwrap();
    let candidate = result.candidate.expect("canonical critical blocks must seed the bounded repair");

    assert_eq!(candidate.makespan(), 7);
    assert_eq!(verify_collection_solution(&collection(schedule), &candidate.to_solution()).unwrap(), vec![7]);
    assert_eq!(metrics.feasible, 1);
    assert_eq!(metrics.reconstructed, 1);
}

#[test]
fn selected_segments_preserve_separate_machine_boundaries() {
    let sequences = vec![vec![0, 1, 2, 3, 4], vec![5, 6]];
    let shapes = test_selected_segment_shapes(&sequences, 7, &[1, 3, 5, 6]);

    assert_eq!(shapes, vec![(0, 1, 1), (0, 3, 1), (1, 0, 2)]);
}

#[test]
fn repair_respects_external_job_and_machine_boundaries() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);
    let mut metrics = ScheduleLnsMetrics::default();
    let mut workspace = ScheduleLnsWorkspace::default();

    let candidate =
        repair_critical_window_with_workspace(&state, apply_config(), &mut metrics, &mut workspace, &stop).unwrap().candidate.unwrap();

    assert_eq!(candidate.machine_sequences()[0], [0, 3], "external machine order must remain fixed");
    assert!(candidate.starts()[1] >= candidate.ends()[0], "external job predecessor bounds the repair release");
    assert!(candidate.starts()[3] >= candidate.ends()[2], "external job successor bounds the repaired segment");
    let before = state.machine_sequences()[0].iter().copied().filter(|&operation| !workspace.selected_mask()[operation]);
    let after = candidate.machine_sequences()[0].iter().copied().filter(|&operation| !workspace.selected_mask()[operation]);
    assert!(before.eq(after));
}

#[test]
fn zero_duration_operations_remain_sound_in_bounded_repair() {
    let durations = [3, 2, 2, 0];
    let horizon = durations.iter().sum();
    let schedule = Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (2, 3)],
        resources: vec![Resource::NoOverlap(vec![0, 3]), Resource::NoOverlap(vec![1, 2])],
        minimize_makespan: true,
    };
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = JobShopState::from_machine_sequences(&problem, vec![vec![0, 3], vec![1, 2]], &stop).unwrap().unwrap();
    let mut metrics = ScheduleLnsMetrics::default();

    let result = repair_critical_window(&state, apply_config(), &mut metrics, &stop).unwrap();
    let candidate = result
        .candidate
        .unwrap_or_else(|| panic!("zero-duration repair was rejected: observed={:?}, metrics={metrics:?}", result.observed_makespan));

    assert!(candidate.makespan() < state.makespan());
    assert_eq!(verify_collection_solution(&collection(schedule), &candidate.to_solution()).unwrap(), vec![candidate.makespan()]);
    assert_eq!(metrics.oracle_rejections, 0);
}

#[test]
fn invalid_and_exact_boundary_configs_are_non_publishing() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);

    let mut zero_target = apply_config();
    zero_target.target_operations = 0;
    let mut metrics = ScheduleLnsMetrics::default();
    let result = repair_critical_window(&state, zero_target, &mut metrics, &stop).unwrap();
    assert!(result.candidate.is_none());
    assert!(!result.timed_out);
    assert_eq!(metrics.attempts, 0);
    assert_eq!(metrics.exact_rejections, 1);

    let mut zero_max = apply_config();
    zero_max.max_operations = 0;
    let mut metrics = ScheduleLnsMetrics::default();
    let result = repair_critical_window(&state, zero_max, &mut metrics, &stop).unwrap();
    assert!(result.candidate.is_none());
    assert_eq!(metrics.attempts, 0);
    assert_eq!(metrics.exact_rejections, 1);

    let mut oversized_reserve = apply_config();
    oversized_reserve.local_budget = Duration::from_millis(10);
    oversized_reserve.verification_reserve = Duration::from_millis(11);
    let mut metrics = ScheduleLnsMetrics::default();
    let result = repair_critical_window(&state, oversized_reserve, &mut metrics, &stop).unwrap();
    assert!(result.candidate.is_none());
    assert_eq!(metrics.attempts, 0);
    assert_eq!(metrics.exact_rejections, 1);

    let mut no_search_time = apply_config();
    no_search_time.local_budget = Duration::from_millis(50);
    no_search_time.verification_reserve = no_search_time.local_budget;
    let mut metrics = ScheduleLnsMetrics::default();
    let result = repair_critical_window(&state, no_search_time, &mut metrics, &stop).unwrap();
    assert!(result.candidate.is_none());
    assert!(!result.timed_out, "an exhausted search slice is distinct from the total deadline");
    assert_eq!(metrics.attempts, 1);
    assert_eq!(metrics.timeouts, 0);

    let mut capped = apply_config();
    capped.target_operations = 4;
    capped.max_operations = 2;
    let mut metrics = ScheduleLnsMetrics::default();
    let result = repair_critical_window(&state, capped, &mut metrics, &stop).unwrap();
    assert_eq!(result.selected_operations, 2);
}

fn synthetic_single_machine(operation_count: usize) -> (Schedule, JobShopState) {
    let schedule = Schedule {
        intervals: (0..operation_count)
            .map(|_| IntervalVar { duration: 1, horizon: i64::try_from(operation_count).unwrap(), modes: Vec::new(), optional: false })
            .collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap((0..operation_count).collect())],
        minimize_makespan: true,
    };
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let state = JobShopState::from_machine_sequences(&problem, vec![(0..operation_count).collect()], &stop).unwrap().unwrap();
    (schedule, state)
}

#[test]
fn workspace_reuses_large_index_buffers_and_keeps_neighborhood_storage_bounded() {
    let (_schedule, state) = synthetic_single_machine(4_096);
    let stop = AtomicBool::new(false);
    let mut config = apply_config();
    config.target_operations = 8;
    config.max_operations = 8;
    config.local_budget = Duration::from_secs(1);
    config.verification_reserve = Duration::from_millis(200);
    let mut metrics = ScheduleLnsMetrics::default();
    let mut workspace = ScheduleLnsWorkspace::default();

    let first = repair_critical_window_with_workspace(&state, config, &mut metrics, &mut workspace, &stop).unwrap();
    let first_capacities = workspace.capacities();
    let second = repair_critical_window_with_workspace(&state, config, &mut metrics, &mut workspace, &stop).unwrap();
    let second_capacities = workspace.capacities();

    assert_eq!(first.selected_operations, 8);
    assert_eq!(second.selected_operations, 8);
    assert_eq!(first_capacities, second_capacities, "the second same-sized attempt must not grow scratch storage");
    assert!(first_capacities.positions >= 4_096);
    assert!(first_capacities.operations <= 16, "bounded neighborhood storage must not scale with the full instance");
    assert!(first_capacities.segments <= 8);
    assert!(first_capacities.segment_windows <= 8);
    assert!(first_capacities.ordered <= 8);
    assert!(first_capacities.frontier <= 64);
    let operation_index_bytes = first_capacities
        .positions
        .saturating_mul(size_of::<usize>())
        .saturating_add(first_capacities.selected.saturating_mul(size_of::<bool>()))
        .saturating_add(first_capacities.local_index.saturating_mul(size_of::<usize>()))
        .saturating_add(first_capacities.segment_of.saturating_mul(size_of::<usize>()));
    assert!(first_capacities.estimated_bytes >= operation_index_bytes);
}

#[test]
fn repeated_shadow_repairs_reuse_full_candidate_sequence_capacity_and_memory_estimate() {
    let schedule = repairable_job_shop();
    let state = poor_state(&schedule);
    let stop = AtomicBool::new(false);
    let mut config = apply_config();
    config.mode = ScheduleLnsMode::Shadow;
    config.local_budget = Duration::from_secs(1);
    config.verification_reserve = Duration::from_millis(100);
    let mut metrics = ScheduleLnsMetrics::default();
    let mut workspace = ScheduleLnsWorkspace::default();

    let first = repair_critical_window_with_workspace(&state, config, &mut metrics, &mut workspace, &stop).unwrap();
    let first_capacities = workspace.capacities();
    let second = repair_critical_window_with_workspace(&state, config, &mut metrics, &mut workspace, &stop).unwrap();
    let second_capacities = workspace.capacities();

    assert_eq!(first.observed_makespan, Some(7));
    assert_eq!(second.observed_makespan, Some(7));
    assert!(first.candidate.is_none());
    assert!(second.candidate.is_none());
    assert_eq!(first_capacities, second_capacities, "a same-sized shadow attempt must reuse every retained capacity");
    assert!(first_capacities.candidate_sequence_vectors >= state.machine_count());
    assert!(first_capacities.candidate_sequence_operations >= state.operation_count());
    assert!(first_capacities.estimated_bytes > 0);
}
