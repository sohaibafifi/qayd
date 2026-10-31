use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::orchestrator::{
    audit_schedule_path_relink_guide_choices, audit_schedule_search_elite_merge_error_classification,
    audit_schedule_search_elite_shadow_default_enabled, audit_schedule_transfer_attempts, audit_schedule_transfer_order,
    audit_with_schedule_search_elite_shadow, audit_with_schedule_search_elite_stop_before_merge,
    audit_with_schedule_tsab_stop_after_pending_report, audit_with_schedule_tsab_stop_after_pending_report_for_owner,
    schedule_local_search_finalization_reserve, schedule_restart_work, schedule_worker_allows_rebase,
};
use qayd::engines::ls::lists::schedule_elite::{
    test_clear_interrupt, test_interrupt_after_checkpoints, ScheduleEliteArchive, ScheduleEliteError, ScheduleEliteSolveToken,
};
use qayd::engines::ls::lists::schedule_relink::{
    ScheduleRelinkGuideKind, ScheduleRelinkMetrics, ScheduleRelinkRequest, ScheduleRelinkWorkspace,
};
use qayd::engines::ls::lists::{
    audit_direct_oracle_scan_layout, audit_direct_oracle_slot_split, audit_persistent_schedule_baseline_reference,
    audit_persistent_schedule_profile_split, audit_persistent_schedule_split, audit_schedule_island_profile,
    audit_schedule_lns_shadow_policy, audit_tsab_elite_restart_policy, audit_tsab_n6_restart_streaming_reference, audit_tsab_owner_mask,
    audit_tsab_phase_policy, audit_tsab_selection_policy, audit_tsab_shortlist_policy, audit_tsab_streaming_reference,
    audit_tsab_tail_corridor_selection, schedule_state::CanonicalPathPolicy, schedule_state::DispatchRule, schedule_state::JobShopProblem,
    schedule_state::JobShopState, schedule_state::ScheduleMove, schedule_state::ScoredScheduleMove, solve_schedule, solve_schedule_capped,
    solve_schedule_capped_persistent, solve_schedule_capped_persistent_relink, solve_schedule_capped_persistent_relink_with_lns_shadow,
    solve_schedule_capped_persistent_shadow, solve_schedule_capped_persistent_tsab, solve_schedule_capped_persistent_tsab_multi,
    solve_schedule_capped_persistent_tsab_multi_relink, solve_schedule_capped_persistent_tsab_with_finalization_stop,
    ScheduleConstructionMetrics, ScheduleJsspSearchStrategy, ScheduleSearchSession,
};
use qayd::model::list::{verify_collection_solution, CollectionModel, CollectionSolution, IntervalVar, Resource, Schedule};
use qayd::model::{Model, ModelPackage};
use qayd::orchestrator::{
    audit_interrupt_next_collection_final_replay, EventCallback, EventControl, IgnoreEvents, ScheduleJsspSearch, SolveEvent, SolveLimits,
    SolveMode, SolveRequest, SolveResult, SolveStatus,
};

static SCHEDULE_LNS_SHADOW_TIMING_LOCK: Mutex<()> = Mutex::new(());

fn job_shop_schedule(operation_count: usize) -> Schedule {
    Schedule {
        intervals: (0..operation_count)
            .map(|_| IntervalVar {
                duration: 1,
                horizon: i64::try_from(operation_count).expect("test size fits i64"),
                modes: Vec::new(),
                optional: false,
            })
            .collect(),
        precedences: Vec::new(),
        resources: vec![Resource::NoOverlap((0..operation_count).collect())],
        minimize_makespan: true,
    }
}

fn deterministic_job_shop(operation_count: usize) -> ModelPackage {
    ModelPackage::new(Model::from_collection(&CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(job_shop_schedule(operation_count)),
    }))
}

fn diversified_job_shop() -> Schedule {
    let durations = [3, 2, 2, 2, 1, 4, 4, 3, 1, 2, 3, 3];
    let horizon = durations.iter().sum();
    Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11)],
        resources: vec![
            Resource::NoOverlap(vec![0, 3, 7, 11]),
            Resource::NoOverlap(vec![1, 5, 6, 10]),
            Resource::NoOverlap(vec![2, 4, 8, 9]),
        ],
        minimize_makespan: true,
    }
}

fn diversified_job_shop_package() -> ModelPackage {
    ModelPackage::new(Model::from_collection(&CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(diversified_job_shop()),
    }))
}

fn tsab_candidate_fixture() -> (Schedule, u64) {
    for jobs in 4..=8usize {
        for machines in 3..=6usize {
            for salt in 0..16usize {
                let operation_count = jobs * machines;
                let durations: Vec<i64> = (0..operation_count)
                    .map(|operation| 1 + i64::try_from((operation * 7 + salt * 3) % 11).expect("small duration"))
                    .collect();
                let horizon = durations.iter().sum();
                let intervals =
                    durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect();
                let precedences = (0..jobs)
                    .flat_map(|job| (0..machines - 1).map(move |operation| (job * machines + operation, job * machines + operation + 1)))
                    .collect();
                let mut assigned = vec![Vec::new(); machines];
                for job in 0..jobs {
                    for operation in 0..machines {
                        let machine = (job + salt + machines - operation % machines) % machines;
                        assigned[machine].push(job * machines + operation);
                    }
                }
                let schedule = Schedule {
                    intervals,
                    precedences,
                    resources: assigned.into_iter().map(Resource::NoOverlap).collect(),
                    minimize_makespan: true,
                };
                for seed in 0..32u64 {
                    if audit_tsab_streaming_reference(&schedule, seed) {
                        return (schedule, seed);
                    }
                }
            }
        }
    }
    panic!("deterministic fixture search must find a strict N5 neighborhood with more than four moves")
}

fn deterministic_resource_project() -> ModelPackage {
    let durations = [2, 1, 3, 2, 4, 1, 2, 3];
    let schedule = Schedule {
        intervals: durations
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 18, modes: Vec::new(), optional: false })
            .collect(),
        precedences: vec![(0, 4), (1, 5), (2, 6), (3, 7)],
        resources: vec![Resource::Cumulative { demands: (0..durations.len()).map(|activity| (activity, 1)).collect(), capacity: 2 }],
        minimize_makespan: true,
    };
    ModelPackage::new(Model::from_collection(&CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule),
    }))
}

fn large_generic_schedule() -> ModelPackage {
    let schedule = Schedule {
        intervals: (0..64).map(|_| IntervalVar { duration: 1, horizon: 64, modes: Vec::new(), optional: true }).collect(),
        precedences: Vec::new(),
        resources: Vec::new(),
        minimize_makespan: true,
    };
    ModelPackage::new(Model::from_collection(&CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule),
    }))
}

fn metadata<'a>(result: &'a SolveResult, name: &str) -> &'a str {
    result
        .reports()
        .first()
        .expect("schedule report")
        .metadata
        .iter()
        .find_map(|(key, value)| (key == name).then_some(value.as_str()))
        .unwrap_or_else(|| panic!("missing schedule metadata {name}"))
}

fn metric(result: &SolveResult, name: &str) -> u64 {
    metadata(result, name).parse().unwrap_or_else(|_| panic!("schedule metric {name} is not an integer"))
}

fn assert_schedule_solution_eq(actual: &qayd::model::list::CollectionSolution, expected: &qayd::model::list::CollectionSolution) {
    assert_eq!(actual.feasible, expected.feasible);
    assert_eq!(actual.objectives, expected.objectives);
    assert_eq!(actual.starts, expected.starts);
    assert_eq!(actual.presences, expected.presences);
    assert_eq!(actual.machines, expected.machines);
    assert_eq!(actual.modes, expected.modes);
}

fn solve_seeded_portfolio_profiled(package: &ModelPackage, search_elite_shadow: bool, profile: bool) -> (SolveResult, Vec<Vec<i64>>) {
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 2,
        seed: 47,
        limits: SolveLimits { iterations: Some(8_192), ..SolveLimits::default() },
        profile,
        ..SolveRequest::default()
    };
    let mut progress = Vec::new();
    let result = audit_with_schedule_search_elite_shadow(search_elite_shadow, || {
        qayd::solve(
            package,
            &request,
            &mut EventCallback(|event| {
                if let SolveEvent::Progress { objectives, .. } = event {
                    progress.push(objectives);
                }
                Ok(EventControl::Continue)
            }),
        )
        .expect("deterministic scheduling portfolio succeeds")
    });
    (result, progress)
}

fn solve_seeded_portfolio_with_shadow(package: &ModelPackage, search_elite_shadow: bool) -> (SolveResult, Vec<Vec<i64>>) {
    solve_seeded_portfolio_profiled(package, search_elite_shadow, true)
}

fn solve_seeded_portfolio(package: &ModelPackage) -> (SolveResult, Vec<Vec<i64>>) {
    solve_seeded_portfolio_with_shadow(package, true)
}

#[test]
fn schedule_transfer_batch_orders_once_falls_back_only_on_rejection_and_stops_on_interruption() {
    let ordered = audit_schedule_transfer_order(vec![(3, 4, vec![10]), (2, 8, vec![9]), (1, 9, vec![9]), (1, 3, vec![9])]);
    assert_eq!(ordered, vec![(1, 3, vec![9]), (1, 9, vec![9]), (2, 8, vec![9]), (3, 4, vec![10])]);

    assert_eq!(audit_schedule_transfer_attempts(4, &[], None), (vec![0], Some(0), false));
    assert_eq!(audit_schedule_transfer_attempts(4, &[0, 1], None), (vec![0, 1, 2], Some(2), false));
    assert_eq!(audit_schedule_transfer_attempts(4, &[0], Some(1)), (vec![0, 1], None, true));
    assert_eq!(audit_schedule_transfer_attempts(3, &[0, 1, 2], None), (vec![0, 1, 2], None, false));
}

#[test]
fn schedule_finalization_reserve_scales_with_replay_size_and_caps_at_large_ta() {
    assert_eq!(schedule_local_search_finalization_reserve(0), Duration::from_millis(250));
    assert!(schedule_local_search_finalization_reserve(16) < Duration::from_millis(251));
    assert!(schedule_local_search_finalization_reserve(100_000) > schedule_local_search_finalization_reserve(16));
    assert_eq!(schedule_local_search_finalization_reserve(1_000_000), Duration::from_millis(2_500));
    assert_eq!(schedule_local_search_finalization_reserve(usize::MAX), Duration::from_millis(2_500));
}

#[test]
fn schedule_tsab_terminal_pass_recovers_a_pending_report_then_replays_transfer_once() {
    let (schedule, _) = tsab_candidate_fixture();
    let seed = 8;
    let package = ModelPackage::new(Model::from_collection(&CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule),
    }));
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        seed,
        threads: 7,
        limits: SolveLimits { time: Some(Duration::from_secs(30)), iterations: Some(7 * (256 + 256 + 128)), ..SolveLimits::default() },
        profile: true,
        schedule_jssp_search: ScheduleJsspSearch::TsabCandidate,
        ..SolveRequest::default()
    };

    let (result, terminal_replay_attempts, pending_objective, published_objective) =
        audit_with_schedule_tsab_stop_after_pending_report(|| {
            qayd::solve(&package, &request, &mut IgnoreEvents).expect("terminal TSAB validation remains inside the solve contract")
        });

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    eprintln!(
        "TSAB terminal orchestrator: objective={:?}, replay_attempts={}, validations={}, promotions={}, boundaries={}, work=[{},{}]",
        result.primal().map(|primal| primal.objectives()),
        terminal_replay_attempts,
        metric(&result, "schedule_tsab_fast_full_validations"),
        metric(&result, "schedule_tsab_fast_pending_promotions"),
        metric(&result, "schedule_restart_boundaries"),
        metric(&result, "schedule_worker_work_min"),
        metric(&result, "schedule_worker_work_max")
    );
    assert_eq!(terminal_replay_attempts, 1, "the recovered pending candidate must cross Transfer replay exactly once");
    let pending_objective = pending_objective.expect("the terminal kernel must expose its validated pending objective");
    let published_objective = published_objective.expect("the Transfer replay must install the terminal candidate");
    assert_eq!(pending_objective, published_objective);
    assert_eq!(result.primal().and_then(|primal| primal.objectives().first().copied()), Some(pending_objective));
    assert_eq!(metric(&result, "schedule_tsab_fast_full_validations"), 1);
    assert_eq!(metric(&result, "schedule_tsab_fast_pending_promotions"), 1);
    assert_eq!(metric(&result, "schedule_tsab_fast_oracle_mismatches"), 0);
    assert_eq!(metric(&result, "schedule_tsab_fast_pending_discards"), 0);
    assert_eq!(metric(&result, "schedule_restart_boundaries"), 3, "the quota-zero terminal pass is not a search boundary");
    assert!(metric(&result, "schedule_worker_work_min") <= 640);
    assert!(metric(&result, "schedule_worker_work_max") <= 640);
    assert_eq!(metric(&result, "schedule_work_budget_overruns"), 0);
    assert_eq!(metadata(&result, "schedule_incumbent_source_worker"), "6");
    assert_eq!(metadata(&result, "schedule_incumbent_verification_rejections"), "0");
    assert_eq!(metadata(&result, "schedule_incumbent_verification_interruptions"), "0");
}

#[test]
fn schedule_tsab_multi_terminal_pass_recovers_worker_three_pending_report() {
    let (schedule, _) = tsab_candidate_fixture();
    let seed = 55;
    let package = ModelPackage::new(Model::from_collection(&CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule),
    }));
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        seed,
        threads: 7,
        limits: SolveLimits { time: Some(Duration::from_secs(30)), iterations: Some(7 * (256 + 256 + 128)), ..SolveLimits::default() },
        profile: true,
        schedule_jssp_search: ScheduleJsspSearch::TsabMultiCandidate,
        ..SolveRequest::default()
    };

    let (result, terminal_replay_attempts, pending_objective, published_objective) =
        audit_with_schedule_tsab_stop_after_pending_report_for_owner(3, || {
            qayd::solve(&package, &request, &mut IgnoreEvents)
                .expect("terminal multi-owner TSAB validation remains inside the solve contract")
        });

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    eprintln!(
        "TSAB multi terminal orchestrator: objective={:?}, replay_attempts={}, pending={pending_objective:?}, published={published_objective:?}, source={}",
        result.primal().map(|primal| primal.objectives()),
        terminal_replay_attempts,
        metadata(&result, "schedule_incumbent_source_worker")
    );
    assert_eq!(terminal_replay_attempts, 1, "worker three pending candidate must cross Transfer replay exactly once");
    let pending_objective = pending_objective.expect("worker three terminal validation must expose its pending objective");
    let published_objective = published_objective.expect("worker three Transfer replay must install the terminal candidate");
    assert_eq!(pending_objective, published_objective);
    assert_eq!(result.primal().and_then(|primal| primal.objectives().first().copied()), Some(pending_objective));
    assert_eq!(metric(&result, "schedule_tsab_owner_worker_mask"), 0x7f);
    assert_eq!(metric(&result, "schedule_tsab_activations"), 7);
    assert!(metric(&result, "schedule_tsab_fast_full_validations") >= 1);
    assert!(metric(&result, "schedule_tsab_fast_pending_promotions") >= 1);
    assert_eq!(metric(&result, "schedule_tsab_fast_oracle_mismatches"), 0);
    assert_eq!(metric(&result, "schedule_tsab_fast_pending_discards"), 0);
    assert_eq!(metric(&result, "schedule_restart_boundaries"), 3, "quota-zero terminal finalization is not a search boundary");
    assert!(metric(&result, "schedule_worker_work_min") <= 640);
    assert!(metric(&result, "schedule_worker_work_max") <= 640);
    assert_eq!(metric(&result, "schedule_work_budget_overruns"), 0);
    assert_eq!(metadata(&result, "schedule_incumbent_source_worker"), "3");
    assert_eq!(metadata(&result, "schedule_incumbent_verification_rejections"), "0");
    assert_eq!(metadata(&result, "schedule_incumbent_verification_interruptions"), "0");
}

#[test]
fn one_boundary_verifies_and_publishes_only_the_best_profile_zero_candidate() {
    let package = deterministic_job_shop(8);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 2,
        seed: 47,
        limits: SolveLimits { iterations: Some(4_096), ..SolveLimits::default() },
        profile: true,
        ..SolveRequest::default()
    };

    let result = qayd::solve(&package, &request, &mut IgnoreEvents).expect("one scheduling boundary succeeds");

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(metadata(&result, "schedule_restart_boundaries"), "1");
    assert_eq!(metadata(&result, "schedule_incumbent_verifications"), "1");
    assert_eq!(metadata(&result, "schedule_incumbent_publications"), "1");
    assert_eq!(metadata(&result, "schedule_incumbent_source_worker"), "0");
    assert_eq!(metadata(&result, "schedule_incumbent_verification_rejections"), "0");
}

fn assert_no_search_elite_shadow(result: &SolveResult) {
    for name in [
        "schedule_search_elite_pool_size",
        "schedule_search_elite_batches",
        "schedule_search_elite_batches_skipped_after_stop",
        "schedule_search_elite_candidates",
        "schedule_search_elite_insertions",
        "schedule_search_elite_duplicates",
        "schedule_search_elite_dominated",
        "schedule_search_elite_evictions",
        "schedule_search_elite_interruptions",
        "schedule_search_elite_merge_errors",
        "schedule_search_elite_snapshot_captures",
        "schedule_search_elite_snapshot_interruptions",
        "schedule_search_elite_snapshot_errors",
        "schedule_search_elite_heap_lower_bound_bytes",
        "schedule_search_elite_peak_heap_lower_bound_bytes",
    ] {
        assert_eq!(metadata(result, name), "0", "non-JSSP shadow metric {name} must be zero");
    }
    for name in [
        "schedule_search_elite_objectives",
        "schedule_search_elite_pairwise_distances_ppm",
        "schedule_search_elite_min_distance_ppm",
        "schedule_search_elite_mean_distance_ppm",
        "schedule_search_elite_max_distance_ppm",
    ] {
        assert_eq!(metadata(result, name), "none", "non-JSSP shadow summary {name} must be empty");
    }
    for name in ["schedule_search_elite_capture_worker_seconds_sum", "schedule_search_elite_merge_wall_seconds"] {
        assert_eq!(metadata(result, name), "0", "non-JSSP shadow timing {name} must be zero");
    }
}

#[test]
fn scheduling_portfolio_exchanges_only_at_reproducible_restart_boundaries() {
    let package = deterministic_job_shop(8);
    let (first, first_progress) = solve_seeded_portfolio(&package);
    let (second, second_progress) = solve_seeded_portfolio(&package);

    assert_eq!(first.status(), SolveStatus::Satisfiable);
    assert_eq!(first.primal(), second.primal());
    assert_eq!(first_progress, second_progress);
    assert!(metric(&first, "schedule_restart_boundaries") > 1);
    assert!(metric(&first, "schedule_incumbent_injection_attempts") >= 2);
    assert!(metric(&first, "schedule_incumbent_injections") <= metric(&first, "schedule_incumbent_injection_attempts"));
    assert_eq!(metadata(&first, "schedule_incumbent_verification_rejections"), "0");
    assert!(metric(&first, "schedule_incumbent_verifications") <= metric(&first, "schedule_restart_boundaries"));
    let verification_seconds = metadata(&first, "schedule_incumbent_verification_seconds").parse::<f64>().unwrap();
    let max_verification_seconds = metadata(&first, "schedule_incumbent_verification_max_seconds").parse::<f64>().unwrap();
    assert!(verification_seconds >= 0.0);
    assert!((0.0..=verification_seconds).contains(&max_verification_seconds));
    assert_eq!(metadata(&first, "schedule_elite_pool_size"), "1");
    assert_eq!(metadata(&first, "elite_pool_size"), "1");
    assert_eq!(metadata(&first, "schedule_incumbent_source_worker"), "0");
    assert_eq!(metadata(&first, "schedule_incumbent_source_round"), "0");
    assert_eq!(metadata(&first, "schedule_work_steps"), "8192");
    assert_eq!(metadata(&first, "schedule_restart_work"), "2048");
    assert_eq!(metadata(&first, "schedule_worker_work_min"), "4096");
    assert_eq!(metadata(&first, "schedule_worker_work_max"), "4096");
    assert_eq!(metadata(&first, "schedule_unused_work_steps"), "0");
    assert_eq!(metadata(&first, "schedule_stalled_unused_work_steps"), "0");
    assert_eq!(metadata(&first, "schedule_work_budget_overruns"), "0");
    assert_eq!(metadata(&first, "schedule_peak_buffered_candidates"), "2");
    assert_eq!(metadata(&first, "schedule_session_initializations"), "2");
    assert!(metric(&first, "schedule_session_resumes") > 0);
    assert!(metric(&first, "schedule_tabu_steps") > 0);
    assert_eq!(metadata(&first, "schedule_island_profile_mask"), "3");
    assert_eq!(metadata(&first, "schedule_island_profile_count"), "2");
    assert!(metadata(&first, "schedule_profile_construction_seconds").contains("0="));
    assert!(metadata(&first, "schedule_profile_construction_seconds").contains("1="));
    assert_eq!(metadata(&first, "schedule_reactive_restarts"), "0");
    assert_eq!(metadata(&first, "schedule_reactive_restart_dispatches"), "0");
    assert_eq!(metadata(&first, "schedule_reactive_restart_rebuild_failures"), "0");
    assert_eq!(metadata(&first, "schedule_island_scored_candidates"), "0");
    assert!(metric(&first, "schedule_island_shortlisted_candidates") > 0);
    assert!(metric(&first, "schedule_moves_accepted") > metric(&first, "schedule_local_improvements").saturating_sub(2));

    for name in [
        "schedule_incumbent_publications",
        "schedule_incumbent_injection_attempts",
        "schedule_incumbent_injections",
        "schedule_incumbent_rejections",
        "schedule_restart_boundaries",
        "schedule_global_improvements",
        "schedule_tabu_steps",
        "schedule_tabu_hits",
        "schedule_tabu_aspirations",
        "schedule_tabu_forced_moves",
        "schedule_session_initializations",
        "schedule_session_resumes",
        "schedule_session_rebases",
        "schedule_island_profile_mask",
        "schedule_island_profile_count",
        "schedule_reactive_restarts",
        "schedule_reactive_restart_dispatches",
        "schedule_reactive_restart_perturbations",
        "schedule_reactive_restart_rebuild_failures",
        "schedule_island_scored_candidates",
        "schedule_island_shortlisted_candidates",
        "schedule_search_elite_pool_size",
        "schedule_search_elite_batches",
        "schedule_search_elite_batches_skipped_after_stop",
        "schedule_search_elite_candidates",
        "schedule_search_elite_insertions",
        "schedule_search_elite_duplicates",
        "schedule_search_elite_dominated",
        "schedule_search_elite_evictions",
        "schedule_search_elite_interruptions",
        "schedule_search_elite_merge_errors",
        "schedule_search_elite_snapshot_captures",
        "schedule_search_elite_snapshot_interruptions",
        "schedule_search_elite_snapshot_errors",
        "schedule_search_elite_objectives",
        "schedule_search_elite_pairwise_distances_ppm",
        "schedule_search_elite_min_distance_ppm",
        "schedule_search_elite_mean_distance_ppm",
        "schedule_search_elite_max_distance_ppm",
        "schedule_search_elite_heap_lower_bound_bytes",
        "schedule_search_elite_peak_heap_lower_bound_bytes",
        "schedule_work_steps",
    ] {
        assert_eq!(metadata(&first, name), metadata(&second, name), "metric {name} must be reproducible");
    }
}

#[test]
fn global_search_elite_archive_is_strictly_shadow_and_reports_a_canonical_pool() {
    assert!(audit_schedule_search_elite_shadow_default_enabled(), "the production-equivalent default must keep shadow measurement on");
    let package = diversified_job_shop_package();
    let (disabled, disabled_progress) = solve_seeded_portfolio_with_shadow(&package, false);
    let (enabled, enabled_progress) = solve_seeded_portfolio_with_shadow(&package, true);

    assert_eq!(disabled.status(), enabled.status());
    assert_eq!(disabled.primal(), enabled.primal());
    assert_eq!(disabled_progress, enabled_progress);
    for name in [
        "constructor",
        "schedule_work_steps",
        "schedule_moves_considered",
        "schedule_moves_accepted",
        "schedule_moves_rejected",
        "schedule_local_improvements",
        "schedule_global_improvements",
        "schedule_progress_publications",
        "schedule_incumbent_publication_attempts",
        "schedule_incumbent_publications",
        "schedule_incumbent_injection_attempts",
        "schedule_incumbent_injections",
        "schedule_incumbent_rejections",
        "schedule_incumbent_verifications",
        "schedule_incumbent_verification_rejections",
        "schedule_restart_boundaries",
        "schedule_peak_buffered_candidates",
        "schedule_worker_work_min",
        "schedule_worker_work_max",
        "schedule_work_budget_overruns",
        "schedule_tabu_steps",
        "schedule_tabu_hits",
        "schedule_tabu_aspirations",
        "schedule_tabu_forced_moves",
        "schedule_session_initializations",
        "schedule_session_resumes",
        "schedule_session_rebases",
        "schedule_reactive_restarts",
        "schedule_reactive_restart_dispatches",
        "schedule_reactive_restart_perturbations",
        "schedule_reactive_restart_rebuild_failures",
        "schedule_incumbent_source_worker",
        "schedule_incumbent_source_round",
        "schedule_elite_pool_size",
        "elite_pool_size",
    ] {
        assert_eq!(metadata(&disabled, name), metadata(&enabled, name), "shadow archive changed trajectory metric {name}");
    }

    assert_no_search_elite_shadow(&disabled);
    let pool_size = metric(&enabled, "schedule_search_elite_pool_size");
    let batches = metric(&enabled, "schedule_search_elite_batches");
    let candidates = metric(&enabled, "schedule_search_elite_candidates");
    let insertions = metric(&enabled, "schedule_search_elite_insertions");
    let duplicates = metric(&enabled, "schedule_search_elite_duplicates");
    let dominated = metric(&enabled, "schedule_search_elite_dominated");
    let evictions = metric(&enabled, "schedule_search_elite_evictions");
    let snapshot_captures = metric(&enabled, "schedule_search_elite_snapshot_captures");
    assert!((1..=4).contains(&pool_size));
    assert!(batches > 0);
    assert_eq!(metadata(&enabled, "schedule_search_elite_batches_skipped_after_stop"), "0");
    assert!(candidates >= pool_size);
    assert_eq!(candidates, insertions.saturating_add(duplicates).saturating_add(dominated));
    assert!(evictions <= insertions);
    assert_eq!(metadata(&enabled, "schedule_search_elite_interruptions"), "0");
    assert_eq!(metadata(&enabled, "schedule_search_elite_merge_errors"), "0");
    assert_eq!(metadata(&enabled, "schedule_search_elite_snapshot_interruptions"), "0");
    assert_eq!(metadata(&enabled, "schedule_search_elite_snapshot_errors"), "0");
    assert!(snapshot_captures >= candidates);
    assert!(snapshot_captures <= 2 * metric(&enabled, "schedule_restart_boundaries"));

    let objectives = metadata(&enabled, "schedule_search_elite_objectives")
        .split(',')
        .map(|objective| objective.parse::<i64>().expect("elite objective is encoded as i64"))
        .collect::<Vec<_>>();
    assert_eq!(u64::try_from(objectives.len()).unwrap(), pool_size);
    assert_eq!(objectives.first(), objectives.iter().min());

    if pool_size == 1 {
        assert_eq!(metadata(&enabled, "schedule_search_elite_pairwise_distances_ppm"), "none");
        for name in
            ["schedule_search_elite_min_distance_ppm", "schedule_search_elite_mean_distance_ppm", "schedule_search_elite_max_distance_ppm"]
        {
            assert_eq!(metadata(&enabled, name), "none");
        }
    } else {
        let distances = metadata(&enabled, "schedule_search_elite_pairwise_distances_ppm")
            .split(',')
            .map(|encoded| {
                let (pair, distance) = encoded.split_once('=').expect("pairwise distance has an index pair");
                assert!(pair.split_once('-').is_some());
                distance.parse::<u32>().expect("pairwise distance is encoded in ppm")
            })
            .collect::<Vec<_>>();
        assert_eq!(u64::try_from(distances.len()).unwrap(), pool_size.saturating_mul(pool_size.saturating_sub(1)) / 2);
        assert!(distances.iter().all(|&distance| distance <= 1_000_000));
        let minimum = *distances.iter().min().unwrap();
        let maximum = *distances.iter().max().unwrap();
        let mean = u32::try_from(distances.iter().map(|&distance| u64::from(distance)).sum::<u64>() / distances.len() as u64).unwrap();
        assert_eq!(metadata(&enabled, "schedule_search_elite_min_distance_ppm"), minimum.to_string());
        assert_eq!(metadata(&enabled, "schedule_search_elite_mean_distance_ppm"), mean.to_string());
        assert_eq!(metadata(&enabled, "schedule_search_elite_max_distance_ppm"), maximum.to_string());
    }

    let current_bytes = metric(&enabled, "schedule_search_elite_heap_lower_bound_bytes");
    let peak_bytes = metric(&enabled, "schedule_search_elite_peak_heap_lower_bound_bytes");
    assert!(current_bytes > 0);
    assert!(peak_bytes >= current_bytes);
    let capture_seconds = metadata(&enabled, "schedule_search_elite_capture_worker_seconds_sum").parse::<f64>().unwrap();
    let merge_seconds = metadata(&enabled, "schedule_search_elite_merge_wall_seconds").parse::<f64>().unwrap();
    assert!(capture_seconds >= 0.0 && merge_seconds >= 0.0);
}

#[test]
fn production_shadow_activation_requires_the_profile_gate() {
    assert!(audit_schedule_search_elite_shadow_default_enabled());
    let package = diversified_job_shop_package();
    let (profile_off, off_progress) = solve_seeded_portfolio_profiled(&package, true, false);
    let (profile_on, on_progress) = solve_seeded_portfolio_profiled(&package, true, true);

    assert_eq!(profile_off.status(), profile_on.status());
    assert_eq!(profile_off.primal(), profile_on.primal());
    assert_eq!(off_progress, on_progress);
    for name in [
        "schedule_work_steps",
        "schedule_moves_considered",
        "schedule_moves_accepted",
        "schedule_local_improvements",
        "schedule_global_improvements",
        "schedule_incumbent_publications",
        "schedule_incumbent_injection_attempts",
        "schedule_incumbent_injections",
        "schedule_session_rebases",
    ] {
        assert_eq!(metadata(&profile_off, name), metadata(&profile_on, name), "profile-gated shadow changed {name}");
    }
    assert_no_search_elite_shadow(&profile_off);
    assert!(metric(&profile_on, "schedule_search_elite_pool_size") > 0);
    assert!(metric(&profile_on, "schedule_search_elite_snapshot_captures") > 0);
    assert_eq!(metadata(&profile_on, "schedule_search_elite_snapshot_errors"), "0");
    assert_eq!(metadata(&profile_on, "schedule_search_elite_merge_errors"), "0");
}

fn solve_path_relink_gate(profile: bool, enabled: bool, threads: usize) -> (SolveResult, Vec<Vec<i64>>) {
    let package = diversified_job_shop_package();
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads,
        seed: 211,
        limits: SolveLimits { iterations: Some(86_016), ..SolveLimits::default() },
        profile,
        schedule_path_relink: enabled,
        ..SolveRequest::default()
    };
    let mut progress = Vec::new();
    let result = qayd::solve(
        &package,
        &request,
        &mut EventCallback(|event| {
            if let SolveEvent::Progress { objectives, .. } = event {
                progress.push(objectives);
            }
            Ok(EventControl::Continue)
        }),
    )
    .expect("path-relink gate solve succeeds");
    (result, progress)
}

#[test]
fn path_relink_requires_flag_profile_seven_workers_and_two_archived_orders() {
    let (default_off, _) = solve_path_relink_gate(true, false, 7);
    assert_eq!(metadata(&default_off, "schedule_path_relink_enabled"), "0");
    assert_eq!(metadata(&default_off, "schedule_path_relink_active"), "0");
    assert_eq!(metadata(&default_off, "schedule_path_relink_guide_requests"), "0");

    let (profile_off, _) = solve_path_relink_gate(false, true, 7);
    assert_eq!(metadata(&profile_off, "schedule_path_relink_enabled"), "1");
    assert_eq!(metadata(&profile_off, "schedule_path_relink_active"), "0");
    assert_eq!(metadata(&profile_off, "schedule_path_relink_guide_requests"), "0");

    let (six_workers, _) = solve_path_relink_gate(true, true, 6);
    assert_eq!(metadata(&six_workers, "schedule_path_relink_active"), "0");
    assert_eq!(metadata(&six_workers, "schedule_path_relink_guide_requests"), "0");

    let (active, active_progress) = solve_path_relink_gate(true, true, 7);
    let (repeated, repeated_progress) = solve_path_relink_gate(true, true, 7);
    assert_eq!(active.status(), SolveStatus::Satisfiable);
    assert_eq!(active.primal(), repeated.primal());
    assert_eq!(active_progress, repeated_progress);
    assert_eq!(metadata(&active, "schedule_path_relink_enabled"), "1");
    assert_eq!(metadata(&active, "schedule_path_relink_active"), "1");
    let requests = metric(&active, "schedule_path_relink_guide_requests");
    let shortlisted = metric(&active, "schedule_path_relink_candidates_shortlisted");
    let attempts = metric(&active, "schedule_path_relink_oracle_attempts");
    let accepts = metric(&active, "schedule_path_relink_oracle_accepts");
    let rollbacks = metric(&active, "schedule_path_relink_rollbacks");
    assert!(requests > 0);
    assert!(metric(&active, "schedule_path_relink_best_guides") > 0);
    assert!(metric(&active, "schedule_path_relink_diverse_guides") > 0);
    assert_eq!(requests, metric(&active, "schedule_path_relink_best_guides") + metric(&active, "schedule_path_relink_diverse_guides"));
    assert!(shortlisted <= requests, "at most one relink candidate may consume an oracle slot per scan");
    let positive_gain = metric(&active, "schedule_path_relink_candidates_positive_gain");
    let certified = metric(&active, "schedule_path_relink_acyclicity_certified");
    let unknown = metric(&active, "schedule_path_relink_acyclicity_unknown");
    assert_eq!(positive_gain, certified.saturating_add(unknown));
    assert_eq!(unknown, metric(&active, "schedule_path_relink_prefilter_rejections"));
    assert!(attempts <= shortlisted);
    assert_eq!(attempts, accepts.saturating_add(rollbacks));
    assert!(attempts <= metric(&active, "schedule_direct_oracle_attempts"));
    assert_eq!(metadata(&active, "schedule_incumbent_verification_rejections"), "0");
    for name in [
        "schedule_path_relink_enabled",
        "schedule_path_relink_active",
        "schedule_path_relink_guide_requests",
        "schedule_path_relink_best_guides",
        "schedule_path_relink_diverse_guides",
        "schedule_path_relink_guide_loads",
        "schedule_path_relink_guide_incompatible",
        "schedule_path_relink_guide_interruptions",
        "schedule_path_relink_critical_operations_scanned",
        "schedule_path_relink_candidates_generated",
        "schedule_path_relink_candidates_positive_gain",
        "schedule_path_relink_acyclicity_certified",
        "schedule_path_relink_acyclicity_unknown",
        "schedule_path_relink_prefilter_rejections",
        "schedule_path_relink_candidates_retained",
        "schedule_path_relink_candidates_refined",
        "schedule_path_relink_candidates_shortlisted",
        "schedule_path_relink_no_move",
        "schedule_path_relink_guide_arc_gain_shortlisted",
        "schedule_path_relink_oracle_attempts",
        "schedule_path_relink_oracle_accepts",
        "schedule_path_relink_cycle_rejections",
        "schedule_path_relink_window_rejections",
        "schedule_path_relink_other_rejections",
        "schedule_path_relink_rollbacks",
        "schedule_path_relink_elite_improvements",
        "schedule_path_relink_guide_arc_gain_accepted",
        "schedule_path_relink_workspace_peak_bytes",
        "schedule_work_steps",
        "schedule_incumbent_verifications",
        "schedule_incumbent_verification_rejections",
    ] {
        assert_eq!(metadata(&active, name), metadata(&repeated, name), "active path-relink metric {name} changed");
    }
}

#[test]
fn path_relink_guide_kind_strictly_alternates_and_diverse_slot_is_seeded_deterministically() {
    for seed in [0, 1, 47, 211, u64::MAX] {
        let first = audit_schedule_path_relink_guide_choices(seed, 12, 4);
        let second = audit_schedule_path_relink_guide_choices(seed, 12, 4);
        assert_eq!(first, second);
        assert_eq!(first.len(), 12);
        for pair in first.windows(2) {
            assert_ne!(pair[0].0, pair[1].0, "guide kind must alternate every round");
        }
        for &(best, index) in &first {
            if best {
                assert_eq!(index, 0);
            } else {
                assert!((1..4).contains(&index));
            }
        }
    }
    assert!(audit_schedule_path_relink_guide_choices(0, 4, 1).is_empty());
}

#[test]
fn path_relink_never_consumes_the_last_normal_direct_oracle_slot() {
    assert_eq!(audit_direct_oracle_slot_split(0, 1), (0, 0));
    assert_eq!(audit_direct_oracle_slot_split(1, 1), (0, 1));
    assert_eq!(audit_direct_oracle_slot_split(2, 0), (0, 2));
    assert_eq!(audit_direct_oracle_slot_split(2, 2), (1, 1));
    assert_eq!(audit_direct_oracle_slot_split(4, usize::MAX), (1, 3));
    assert_eq!(audit_direct_oracle_scan_layout(), (1, 2, true));
}

#[test]
fn path_relink_request_is_ignored_bit_for_bit_by_profiles_zero_through_five() {
    let schedule = diversified_job_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let guide = JobShopState::giffler_thompson(&problem, 223, DispatchRule::Randomized, &stop).unwrap().unwrap();
    let mut archive = ScheduleEliteArchive::new();
    archive.consider(&guide, &stop).unwrap();
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Diverse };

    for worker in 0..6 {
        let mut baseline_reports = Vec::new();
        let (baseline, baseline_metrics, _) = solve_schedule_capped_persistent_relink(
            &schedule,
            227,
            229,
            worker,
            7,
            &stop,
            512,
            None,
            None,
            false,
            None,
            None,
            &mut |objective| baseline_reports.push(objective),
        );
        let mut requested_reports = Vec::new();
        let (requested, requested_metrics, _) = solve_schedule_capped_persistent_relink(
            &schedule,
            227,
            229,
            worker,
            7,
            &stop,
            512,
            None,
            None,
            false,
            None,
            Some(request),
            &mut |objective| requested_reports.push(objective),
        );

        assert_schedule_solution_eq(&requested, &baseline);
        assert_eq!(requested_reports, baseline_reports, "profile {worker} reports changed");
        assert_eq!(requested_metrics.work_steps, baseline_metrics.work_steps);
        assert_eq!(requested_metrics.moves_considered, baseline_metrics.moves_considered);
        assert_eq!(requested_metrics.moves_accepted, baseline_metrics.moves_accepted);
        assert_eq!(requested_metrics.incumbent_improvements, baseline_metrics.incumbent_improvements);
        assert_eq!(requested_metrics.schedule_path_relink.requests, 0);
    }
}

#[test]
fn path_relink_uses_existing_four_unit_oracle_credit_across_a_one_plus_three_split() {
    let schedule = diversified_job_shop();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let (current_order, guide_order) = (0..64)
        .find_map(|current_seed| {
            let current_order = JobShopState::giffler_thompson(&problem, current_seed, DispatchRule::Randomized, &stop)
                .unwrap()
                .unwrap()
                .machine_sequences()
                .to_vec();
            (0..64).find_map(|guide_seed| {
                let guide = JobShopState::giffler_thompson(&problem, guide_seed, DispatchRule::Randomized, &stop).unwrap().unwrap();
                let mut candidate_archive = ScheduleEliteArchive::new();
                candidate_archive.consider(&guide, &stop).unwrap();
                let mut current = JobShopState::from_machine_sequences(&problem, current_order.clone(), &stop).unwrap().unwrap();
                let mut workspace = ScheduleRelinkWorkspace::default();
                let mut metrics = ScheduleRelinkMetrics::default();
                workspace
                    .prepare(
                        &mut current,
                        ScheduleRelinkRequest { guide: candidate_archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best },
                        &mut metrics,
                        &stop,
                    )
                    .unwrap();
                (!workspace.shortlist().is_empty()).then(|| (current_order.clone(), guide.machine_sequences().to_vec()))
            })
        })
        .expect("a deterministic current and guide offer profile 6 one path move");
    let initial = JobShopState::from_machine_sequences(&problem, current_order, &stop).unwrap().unwrap().to_solution();
    let guide = JobShopState::from_machine_sequences(&problem, guide_order, &stop).unwrap().unwrap();
    let mut archive = ScheduleEliteArchive::new();
    archive.consider(&guide, &stop).unwrap();
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Best };

    let (monolithic, monolithic_metrics, _) = solve_schedule_capped_persistent_relink(
        &schedule,
        239,
        241,
        6,
        7,
        &stop,
        4,
        Some(&initial),
        None,
        false,
        None,
        Some(request),
        &mut |_| {},
    );
    let (_, first_metrics, first_session) = solve_schedule_capped_persistent_relink(
        &schedule,
        239,
        241,
        6,
        7,
        &stop,
        1,
        Some(&initial),
        None,
        false,
        None,
        Some(request),
        &mut |_| {},
    );
    let (split, second_metrics, _) = solve_schedule_capped_persistent_relink(
        &schedule,
        239,
        241,
        6,
        7,
        &stop,
        3,
        None,
        first_session,
        false,
        None,
        Some(request),
        &mut |_| {},
    );

    assert_schedule_solution_eq(&split, &monolithic);
    assert_eq!(first_metrics.work_steps, 1);
    assert_eq!(second_metrics.work_steps, 3);
    assert_eq!(first_metrics.schedule_path_relink.requests, 0);
    assert_eq!(second_metrics.schedule_path_relink.requests, monolithic_metrics.schedule_path_relink.requests);
    assert_eq!(second_metrics.schedule_path_relink.oracle_attempts, monolithic_metrics.schedule_path_relink.oracle_attempts);
    assert_eq!(monolithic_metrics.direct_oracle_attempts, 1);
    assert_eq!(monolithic_metrics.schedule_path_relink.oracle_attempts, 1);
}

#[test]
fn shadow_merge_metrics_do_not_mask_integrity_errors_as_interruptions() {
    assert_eq!(audit_schedule_search_elite_merge_error_classification(ScheduleEliteError::Interrupted), (1, 0));
    for error in [
        ScheduleEliteError::EncodingOverflow,
        ScheduleEliteError::InvalidMachineOrder,
        ScheduleEliteError::InvalidObjective,
        ScheduleEliteError::IncompatibleProblem,
        ScheduleEliteError::IncompatibleSolve,
    ] {
        assert_eq!(audit_schedule_search_elite_merge_error_classification(error), (0, 1));
    }
}

#[test]
fn final_pending_shadow_batch_is_skipped_when_stop_is_already_armed() {
    let package = deterministic_job_shop(8);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 2,
        seed: 163,
        limits: SolveLimits { iterations: Some(8_192), ..SolveLimits::default() },
        profile: true,
        ..SolveRequest::default()
    };
    let result = audit_with_schedule_search_elite_stop_before_merge(|| {
        qayd::solve(&package, &request, &mut IgnoreEvents).expect("pre-merge stop keeps the verified scheduling result")
    });

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(result.primal().is_some());
    assert_eq!(metadata(&result, "schedule_search_elite_batches_skipped_after_stop"), "1");
    assert_eq!(metadata(&result, "schedule_search_elite_batches"), "0");
    assert_eq!(metadata(&result, "schedule_search_elite_candidates"), "0");
    assert_eq!(metadata(&result, "schedule_search_elite_pool_size"), "0");
    assert_eq!(metadata(&result, "schedule_search_elite_interruptions"), "0");
    assert_eq!(metadata(&result, "schedule_search_elite_merge_errors"), "0");
    assert_eq!(metadata(&result, "schedule_search_elite_snapshot_interruptions"), "0");
    assert_eq!(metadata(&result, "schedule_search_elite_snapshot_errors"), "0");
    assert_eq!(metadata(&result, "schedule_search_elite_heap_lower_bound_bytes"), "0");
    assert!(metric(&result, "schedule_search_elite_peak_heap_lower_bound_bytes") > 0);
    assert!(metric(&result, "schedule_search_elite_snapshot_captures") > 0);
}

#[test]
fn resource_portfolio_reconstructs_the_verified_shared_incumbent() {
    let package = deterministic_resource_project();
    let (first, first_progress) = solve_seeded_portfolio(&package);
    let (second, second_progress) = solve_seeded_portfolio(&package);

    assert_eq!(first.status(), SolveStatus::Satisfiable);
    assert_eq!(first.primal(), second.primal());
    assert_eq!(first_progress, second_progress);
    assert_eq!(metadata(&first, "constructor"), "resource-priority-sgs");
    assert!(metric(&first, "schedule_restart_boundaries") > 1);
    assert!(metric(&first, "schedule_incumbent_injections") >= 2);
    assert_eq!(metadata(&first, "schedule_incumbent_injections"), metadata(&first, "schedule_incumbent_injection_attempts"));
    assert_eq!(metadata(&first, "schedule_incumbent_verification_rejections"), "0");
    assert_eq!(metadata(&first, "schedule_elite_pool_size"), "1");
    assert_eq!(metadata(&first, "schedule_work_steps"), "8192");
    assert_eq!(metadata(&first, "schedule_restart_work"), "2048");
    assert_eq!(metadata(&first, "schedule_worker_work_min"), "4096");
    assert_eq!(metadata(&first, "schedule_worker_work_max"), "4096");
    assert_eq!(metadata(&first, "schedule_unused_work_steps"), "0");
    assert_eq!(metadata(&first, "schedule_stalled_unused_work_steps"), "0");
    assert_eq!(metadata(&first, "schedule_work_budget_overruns"), "0");
    assert_eq!(metadata(&first, "schedule_peak_buffered_candidates"), "2");
    assert_eq!(metadata(&first, "schedule_tabu_steps"), "0");
    assert_eq!(metadata(&first, "schedule_session_initializations"), "0");
    assert_eq!(metadata(&first, "schedule_session_resumes"), "0");
    assert_eq!(metadata(&first, "schedule_session_rebases"), "0");
    assert_no_search_elite_shadow(&first);

    for name in [
        "schedule_incumbent_publications",
        "schedule_incumbent_injection_attempts",
        "schedule_incumbent_injections",
        "schedule_incumbent_rejections",
        "schedule_incumbent_source_worker",
        "schedule_incumbent_source_round",
        "schedule_restart_boundaries",
        "schedule_global_improvements",
        "schedule_tabu_steps",
        "schedule_tabu_hits",
        "schedule_tabu_aspirations",
        "schedule_tabu_forced_moves",
        "schedule_session_initializations",
        "schedule_session_resumes",
        "schedule_session_rebases",
        "schedule_work_steps",
    ] {
        assert_eq!(metadata(&first, name), metadata(&second, name), "metric {name} must be reproducible");
    }
}

#[test]
fn soft_search_stop_preserves_the_last_verified_portfolio_incumbent() {
    let _timing_guard = SCHEDULE_LNS_SHADOW_TIMING_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let package = deterministic_job_shop(16);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 2,
        seed: 71,
        limits: SolveLimits { time: Some(Duration::from_millis(100)), ..SolveLimits::default() },
        profile: true,
        ..SolveRequest::default()
    };

    let result = qayd::solve(&package, &request, &mut IgnoreEvents).expect("soft scheduling stop returns a verified incumbent");

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().expect("verified incumbent survives the stop").objectives(), [16]);
    assert_eq!(metadata(&result, "schedule_elite_pool_size"), "1");
    assert_eq!(metadata(&result, "schedule_incumbent_verification_rejections"), "0");
    assert!(metric(&result, "schedule_peak_buffered_candidates") <= 2);
}

#[test]
fn zero_step_generic_fallback_retires_workers_without_spinning() {
    let package = large_generic_schedule();
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 2,
        seed: 83,
        limits: SolveLimits { iterations: Some(100), ..SolveLimits::default() },
        profile: true,
        ..SolveRequest::default()
    };

    let result = qayd::solve(&package, &request, &mut IgnoreEvents).expect("zero-step generic schedule terminates safely");

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(metadata(&result, "schedule_work_steps"), "0");
    assert_eq!(metadata(&result, "schedule_stalled_workers"), "2");
    assert_eq!(metadata(&result, "schedule_unused_work_steps"), "100");
    assert_eq!(metadata(&result, "schedule_stalled_unused_work_steps"), "100");
    assert_eq!(metadata(&result, "schedule_restart_boundaries"), "1");
    assert_eq!(metadata(&result, "schedule_elite_pool_size"), "1");
    assert_no_search_elite_shadow(&result);
    assert!(result.message().is_some_and(|message| message.contains("stalled worker")));
}

#[test]
fn structured_worker_counts_only_a_reconstructed_injected_incumbent() {
    let schedule = job_shop_schedule(8);
    let running = AtomicBool::new(false);
    let (incumbent, _) = solve_schedule(&schedule, 101, &running, &mut |_| {});
    assert!(incumbent.feasible);

    let (_, injected) = solve_schedule_capped(&schedule, 103, &running, 128, false, Some(&incumbent), &mut |_| {});
    assert_eq!(injected.incumbent_injections, 1);

    let prearmed = AtomicBool::new(true);
    let (_, interrupted) = solve_schedule_capped(&schedule, 107, &prearmed, 128, false, Some(&incumbent), &mut |_| {});
    assert_eq!(interrupted.incumbent_injections, 0);
}

#[test]
fn interrupted_shadow_snapshot_is_fail_open_and_does_not_change_the_local_trajectory() {
    let schedule = diversified_job_shop();
    let stop = AtomicBool::new(false);
    let mut disabled_reports = Vec::new();
    let (disabled, disabled_metrics, _) =
        solve_schedule_capped_persistent_shadow(&schedule, 137, 137, 0, 2, &stop, 512, None, None, false, None, &mut |objective| {
            disabled_reports.push(objective)
        });

    test_interrupt_after_checkpoints(0);
    let token = ScheduleEliteSolveToken::new();
    let mut enabled_reports = Vec::new();
    let (enabled, enabled_metrics, _) =
        solve_schedule_capped_persistent_shadow(&schedule, 137, 137, 0, 2, &stop, 512, None, None, false, Some(&token), &mut |objective| {
            enabled_reports.push(objective)
        });
    test_clear_interrupt();

    assert_eq!(enabled.feasible, disabled.feasible);
    assert_eq!(enabled.objectives, disabled.objectives);
    assert_eq!(enabled.starts, disabled.starts);
    assert_eq!(enabled.presences, disabled.presences);
    assert_eq!(enabled.machines, disabled.machines);
    assert_eq!(enabled.modes, disabled.modes);
    assert_eq!(enabled_reports, disabled_reports);
    assert_eq!(enabled_metrics.work_steps, disabled_metrics.work_steps);
    assert_eq!(enabled_metrics.moves_considered, disabled_metrics.moves_considered);
    assert_eq!(enabled_metrics.moves_accepted, disabled_metrics.moves_accepted);
    assert_eq!(enabled_metrics.incumbent_improvements, disabled_metrics.incumbent_improvements);
    assert_eq!(enabled_metrics.incumbent_injections, disabled_metrics.incumbent_injections);
    assert_eq!(enabled_metrics.cycle_rejections, disabled_metrics.cycle_rejections);
    assert_eq!(enabled_metrics.window_rejections, disabled_metrics.window_rejections);
    assert_eq!(enabled_metrics.objective_rejections, disabled_metrics.objective_rejections);
    assert_eq!(enabled_metrics.tabu_steps, disabled_metrics.tabu_steps);
    assert_eq!(enabled_metrics.tabu_hits, disabled_metrics.tabu_hits);
    assert_eq!(enabled_metrics.tabu_aspirations, disabled_metrics.tabu_aspirations);
    assert_eq!(enabled_metrics.tabu_forced_moves, disabled_metrics.tabu_forced_moves);
    assert_eq!(enabled_metrics.session_initializations, disabled_metrics.session_initializations);
    assert_eq!(enabled_metrics.session_resumes, disabled_metrics.session_resumes);
    assert_eq!(enabled_metrics.session_rebases, disabled_metrics.session_rebases);
    assert_eq!(disabled_metrics.search_elite_snapshot_captures, 0);
    assert_eq!(disabled_metrics.search_elite_snapshot_interruptions, 0);
    assert_eq!(disabled_metrics.search_elite_snapshot_errors, 0);
    assert_eq!(enabled_metrics.search_elite_snapshot_interruptions, 1);
    assert_eq!(enabled_metrics.search_elite_snapshot_errors, 0);
}

#[test]
fn mismatched_resume_token_is_a_snapshot_error_not_an_interruption() {
    let schedule = diversified_job_shop();
    let stop = AtomicBool::new(false);
    let correct_token = ScheduleEliteSolveToken::new();
    let mismatch_origin = ScheduleEliteSolveToken::new();
    let mismatch_resume = ScheduleEliteSolveToken::new();
    let (_, _, correct_session) =
        solve_schedule_capped_persistent_shadow(&schedule, 149, 149, 0, 2, &stop, 0, None, None, false, Some(&correct_token), &mut |_| {});
    let (_, _, mismatch_session) = solve_schedule_capped_persistent_shadow(
        &schedule,
        149,
        149,
        0,
        2,
        &stop,
        0,
        None,
        None,
        false,
        Some(&mismatch_origin),
        &mut |_| {},
    );

    let (correct, correct_metrics, _) = solve_schedule_capped_persistent_shadow(
        &schedule,
        149,
        149,
        0,
        2,
        &stop,
        512,
        None,
        correct_session,
        false,
        Some(&correct_token),
        &mut |_| {},
    );
    let (mismatch, mismatch_metrics, _) = solve_schedule_capped_persistent_shadow(
        &schedule,
        149,
        149,
        0,
        2,
        &stop,
        512,
        None,
        mismatch_session,
        false,
        Some(&mismatch_resume),
        &mut |_| {},
    );

    assert_eq!(mismatch.objectives, correct.objectives);
    assert_eq!(mismatch.starts, correct.starts);
    assert_eq!(mismatch_metrics.work_steps, correct_metrics.work_steps);
    assert_eq!(mismatch_metrics.moves_considered, correct_metrics.moves_considered);
    assert_eq!(mismatch_metrics.moves_accepted, correct_metrics.moves_accepted);
    assert_eq!(correct_metrics.search_elite_snapshot_errors, 0);
    assert_eq!(mismatch_metrics.search_elite_snapshot_interruptions, 0);
    assert_eq!(mismatch_metrics.search_elite_snapshot_errors, 1);
}

#[test]
fn one_worker_slice_keeps_only_its_first_pending_snapshot() {
    let schedule = diversified_job_shop();
    let stop = AtomicBool::new(false);
    let token = ScheduleEliteSolveToken::new();
    let (_, metrics, _) =
        solve_schedule_capped_persistent_shadow(&schedule, 157, 157, 0, 2, &stop, 8_192, None, None, false, Some(&token), &mut |_| {});

    assert_eq!(metrics.search_elite_snapshot_captures, 1);
    assert_eq!(metrics.search_elite_snapshot_interruptions, 0);
    assert_eq!(metrics.search_elite_snapshot_errors, 0);
}

#[test]
fn lns_shadow_production_cadence_and_budget_are_explicitly_audited() {
    let (_, stagnation, first, repeat, budget_ms, reserve_ms) = audit_schedule_lns_shadow_policy(0, 0, 0, 0);
    assert_eq!((stagnation, first, repeat), (512, 512, 2_048));
    assert_eq!((budget_ms, reserve_ms), (500, 100));
    assert!(reserve_ms < budget_ms);

    assert!(!audit_schedule_lns_shadow_policy(511, 0, 0, 0).0);
    assert!(audit_schedule_lns_shadow_policy(512, 0, 0, 0).0);
    assert!(audit_schedule_lns_shadow_policy(512, 512, 0, 0).0, "the first diagnostic observation does not require stagnation");

    assert!(!audit_schedule_lns_shadow_policy(2_559, 0, 1, 512).0);
    assert!(audit_schedule_lns_shadow_policy(2_560, 0, 1, 512).0);
    assert!(!audit_schedule_lns_shadow_policy(2_560, 2_049, 1, 512).0);
}

#[test]
fn explicit_lns_shadow_is_worker_zero_only_persistent_and_trajectory_inert() {
    let _timing_guard = SCHEDULE_LNS_SHADOW_TIMING_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let schedule = diversified_job_shop();
    let stop = AtomicBool::new(false);
    let token = ScheduleEliteSolveToken::new();
    let (_, _, off_session) =
        solve_schedule_capped_persistent_relink(&schedule, 167, 167, 0, 2, &stop, 0, None, None, false, Some(&token), None, &mut |_| {});
    let (_, initialization_metrics, on_session) = solve_schedule_capped_persistent_relink_with_lns_shadow(
        &schedule,
        167,
        167,
        0,
        2,
        &stop,
        0,
        None,
        None,
        false,
        Some(&token),
        true,
        None,
        &mut |_| {},
    );
    assert_eq!(initialization_metrics.schedule_lns.attempts, 0, "shadow LNS requires a resumed session");
    let mut off_session = off_session.unwrap();
    let mut on_session = on_session.unwrap();
    off_session.test_force_schedule_lns_shadow_due();
    on_session.test_force_schedule_lns_shadow_due();

    let (off_solution, off_metrics, off_session) = solve_schedule_capped_persistent_relink(
        &schedule,
        167,
        167,
        0,
        2,
        &stop,
        1,
        None,
        Some(off_session),
        false,
        Some(&token),
        None,
        &mut |_| {},
    );
    let (on_solution, on_metrics, on_session) = solve_schedule_capped_persistent_relink_with_lns_shadow(
        &schedule,
        167,
        167,
        0,
        2,
        &stop,
        1,
        None,
        Some(on_session),
        false,
        Some(&token),
        true,
        None,
        &mut |_| {},
    );
    let off_session = off_session.unwrap();
    let mut on_session = on_session.unwrap();

    assert_schedule_solution_eq(&on_solution, &off_solution);
    assert_eq!(off_metrics.schedule_lns.attempts, 0);
    assert_eq!(off_metrics.schedule_lns_shadow_owner_worker_mask, 0);
    assert_eq!(on_metrics.schedule_lns.attempts, 1);
    assert_eq!(on_metrics.schedule_lns_shadow_owner_worker_mask, 1);
    assert!(on_metrics.schedule_lns_workspace_peak_bytes > 0);
    assert_eq!(on_metrics.work_steps, off_metrics.work_steps);
    assert_eq!(on_metrics.moves_considered, off_metrics.moves_considered);
    assert_eq!(on_metrics.moves_accepted, off_metrics.moves_accepted);
    assert!(on_session.test_same_tabu_trajectory(&off_session));

    let first_capacities = on_session.test_schedule_lns_workspace_capacities().unwrap();
    on_session.test_force_schedule_lns_shadow_due();
    let (_, repeat_metrics, on_session) = solve_schedule_capped_persistent_relink_with_lns_shadow(
        &schedule,
        167,
        167,
        0,
        2,
        &stop,
        1,
        None,
        Some(on_session),
        false,
        Some(&token),
        true,
        None,
        &mut |_| {},
    );
    let second_capacities = on_session.unwrap().test_schedule_lns_workspace_capacities().unwrap();
    assert_eq!(repeat_metrics.schedule_lns.attempts, 1);
    assert_eq!(second_capacities, first_capacities);
}

#[test]
fn explicit_lns_shadow_never_runs_on_a_non_owner_worker() {
    let _timing_guard = SCHEDULE_LNS_SHADOW_TIMING_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let schedule = diversified_job_shop();
    let stop = AtomicBool::new(false);
    let token = ScheduleEliteSolveToken::new();
    let (_, _, session) =
        solve_schedule_capped_persistent_relink(&schedule, 173, 173, 1, 2, &stop, 0, None, None, false, Some(&token), None, &mut |_| {});
    let mut session = session.unwrap();
    session.test_force_schedule_lns_shadow_due();

    let (_, metrics, _) = solve_schedule_capped_persistent_relink_with_lns_shadow(
        &schedule,
        173,
        173,
        1,
        2,
        &stop,
        1,
        None,
        Some(session),
        false,
        Some(&token),
        true,
        None,
        &mut |_| {},
    );

    assert_eq!(metrics.schedule_lns_shadow_owner_worker_mask, 0);
    assert_eq!(metrics.schedule_lns.attempts, 0);
    assert_eq!(metrics.schedule_lns_workspace_peak_bytes, 0);
}

#[test]
fn adaptive_schedule_restart_work_is_pure_and_caps_large_batches() {
    assert_eq!(schedule_restart_work(8, 100_000), 2_048);
    assert_eq!(schedule_restart_work(64, 100_000), 16_384);
    assert_eq!(schedule_restart_work(2_560, 100_000), 100_000);
    assert_eq!(schedule_restart_work(1_000_000, 100_000), 256);
    assert_eq!(schedule_restart_work(2_000_000, 100_000), 256);
    assert_eq!(schedule_restart_work(2_000_000, 200), 200);
}

#[test]
fn persistent_tabu_trajectory_is_invariant_to_a_mid_scan_boundary() {
    assert!(audit_persistent_schedule_split(&job_shop_schedule(8), 109, 37, 63));
}

#[test]
fn baseline_profile_matches_history_and_preserves_scan_epoch_on_rebase() {
    assert!(audit_persistent_schedule_baseline_reference(&job_shop_schedule(8), 113, 4_096));
}

#[test]
fn direct_oracle_profile_is_split_invariant_and_respects_its_work_quota() {
    assert!(audit_persistent_schedule_profile_split(&job_shop_schedule(8), 113, 6, 7, 128, 128));
    assert!(audit_persistent_schedule_profile_split(&job_shop_schedule(8), 113, 6, 7, 1, 3));
}

#[test]
fn seven_worker_adjusted_work_profiles_are_split_invariant() {
    let schedule = diversified_job_shop();
    for worker in 0..=4 {
        assert!(audit_persistent_schedule_profile_split(&schedule, 127, worker, 7, 37, 63), "profile {worker} changed across slices");
    }
}

#[test]
fn adjusted_work_dispatch_portfolio_is_bounded_to_seven_workers_and_deterministic() {
    let previous_dispatch = [
        "earliest-start",
        "earliest-start",
        "earliest-start",
        "earliest-start",
        "earliest-start-then-most-work-remaining",
        "earliest-start",
        "earliest-start",
        "earliest-start",
    ];
    let seven_worker_dispatch = [
        "earliest-start-then-most-adjusted-work-c24",
        "earliest-start-then-most-adjusted-work-c51",
        "earliest-start-then-most-adjusted-work-c48",
        "earliest-start-then-most-adjusted-work-c16",
        "earliest-start-then-most-adjusted-work-c41",
        "earliest-start",
        "earliest-start-then-most-work-remaining",
        "earliest-start",
    ];
    let expected_widths = [32, 32, 32, 32, 32, 32, 96, 128];

    for workers in [1, 2, 6, 7, 8] {
        let expected_dispatch = if workers == 7 { seven_worker_dispatch } else { previous_dispatch };
        for worker in 0..workers {
            let first = audit_schedule_island_profile(worker, workers);
            let second = audit_schedule_island_profile(worker, workers);
            assert_eq!(first, second);
            assert_eq!(first.0, worker);
            assert_eq!(first.1, expected_dispatch[worker]);
            assert_eq!(first.2, "earliest-start", "reactive dispatch remains separate from initial construction");
            assert_eq!(first.3, expected_widths[worker]);
            assert!(first.4 >= 3);
            if worker < 6 {
                assert_eq!(first.4, 32, "baseline tenure must equal the historical base");
            }
            assert_eq!(first.5, u64::MAX, "inactive restart metadata must never advertise a restart interval");
        }
    }

    assert_eq!(audit_schedule_island_profile(0, 1).3, 32, "the single-worker default retains the legacy scan width");
}

#[test]
fn profile_constructor_mapping_is_executed_deterministically_and_reports_only_new_sessions() {
    let schedule = diversified_job_shop();
    let running = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &running).unwrap().unwrap();
    let previous_rules = [
        DispatchRule::EarliestStart,
        DispatchRule::EarliestStart,
        DispatchRule::EarliestStart,
        DispatchRule::EarliestStart,
        DispatchRule::EarliestStartThenMostWorkRemaining,
        DispatchRule::EarliestStart,
        DispatchRule::EarliestStart,
        DispatchRule::EarliestStart,
    ];
    let seven_worker_rules = [
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[0],
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[1],
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[2],
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[3],
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[4],
        DispatchRule::EarliestStart,
        DispatchRule::EarliestStartThenMostWorkRemaining,
        DispatchRule::EarliestStart,
    ];
    let previous_names = [
        "earliest-start",
        "earliest-start",
        "earliest-start",
        "earliest-start",
        "earliest-start-then-most-work-remaining",
        "earliest-start",
        "earliest-start",
        "earliest-start",
    ];
    let seven_worker_names = [
        "earliest-start-then-most-adjusted-work-c24",
        "earliest-start-then-most-adjusted-work-c51",
        "earliest-start-then-most-adjusted-work-c48",
        "earliest-start-then-most-adjusted-work-c16",
        "earliest-start-then-most-adjusted-work-c41",
        "earliest-start",
        "earliest-start-then-most-work-remaining",
        "earliest-start",
    ];

    for workers in [1, 2, 6, 7, 8] {
        let expected_rules = if workers == 7 { seven_worker_rules } else { previous_rules };
        let expected_names = if workers == 7 { seven_worker_names } else { previous_names };
        for worker in 0..workers {
            let expected = JobShopState::giffler_thompson(&problem, 127, expected_rules[worker], &running).unwrap().unwrap();
            let expected_metrics = expected.metrics();
            let mut first_reports = Vec::new();
            let (first, first_metrics, first_session) =
                solve_schedule_capped_persistent(&schedule, 127, 127, worker, workers, &running, 0, None, None, false, &mut |objective| {
                    first_reports.push(objective)
                });
            let mut second_reports = Vec::new();
            let (second, second_metrics, _) =
                solve_schedule_capped_persistent(&schedule, 127, 127, worker, workers, &running, 0, None, None, false, &mut |objective| {
                    second_reports.push(objective)
                });

            assert_eq!(first.starts, expected.starts(), "worker {worker}/{workers} must execute its declared constructor");
            assert_eq!(first.objectives, vec![expected.makespan()]);
            assert_eq!(first.starts, second.starts);
            assert_eq!(first.objectives, second.objectives);
            assert_eq!(first_reports, second_reports);
            assert_eq!(first_metrics.initial_objective, Some(expected.makespan()));
            assert_eq!(first_metrics.initial_dispatch_rule, Some(expected_names[worker]));
            assert_eq!(second_metrics.initial_objective, first_metrics.initial_objective);
            assert_eq!(second_metrics.initial_dispatch_rule, first_metrics.initial_dispatch_rule);
            assert_eq!(first_metrics.session_initializations, 1);
            assert_eq!(first_metrics.session_resumes, 0);
            assert_eq!(first_metrics.candidates, expected_metrics.construction_candidates);
            assert_eq!(first_metrics.construction_bucket_visits, expected_metrics.construction_bucket_visits);
            assert_eq!(first_metrics.construction_heap_pushes, expected_metrics.construction_heap_pushes);
            assert_eq!(first_metrics.construction_stale_pops, expected_metrics.construction_stale_pops);
            assert_eq!(first_metrics.construction_heap_rebuilds, expected_metrics.construction_heap_rebuilds);
            assert_eq!(first_metrics.construction_heap_peak, expected_metrics.construction_heap_peak);

            let (_, resumed_metrics, _) = solve_schedule_capped_persistent(
                &schedule,
                127,
                127,
                worker,
                workers,
                &running,
                0,
                None,
                first_session,
                false,
                &mut |_| {},
            );
            assert_eq!(resumed_metrics.session_initializations, 0);
            assert_eq!(resumed_metrics.session_resumes, 1);
            assert_eq!(resumed_metrics.initial_objective, None);
            assert_eq!(resumed_metrics.initial_dispatch_rule, None);
            assert_eq!(resumed_metrics.candidates, 0);
            assert_eq!(resumed_metrics.construction_bucket_visits, 0);
            assert_eq!(resumed_metrics.construction_heap_pushes, 0);
            assert_eq!(resumed_metrics.construction_stale_pops, 0);
            assert_eq!(resumed_metrics.construction_heap_rebuilds, 0);
            assert_eq!(resumed_metrics.construction_heap_peak, 0);
        }
    }
}

#[test]
fn schedule_soft_deadline_after_a_verified_boundary_promotes_without_replay() {
    let package = deterministic_job_shop(16);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 2,
        seed: 71,
        limits: SolveLimits { iterations: Some(8_192), ..SolveLimits::default() },
        profile: true,
        ..SolveRequest::default()
    };
    audit_interrupt_next_collection_final_replay(false);

    let result = qayd::solve(&package, &request, &mut IgnoreEvents).expect("soft final publication deadline keeps the verified incumbent");

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().expect("verified schedule survives final publication").objectives(), [16]);
    assert_eq!(metadata(&result, "schedule_incumbent_verification_rejections"), "0");
    assert!(metric(&result, "schedule_incumbent_verifications") > 0);
}

#[test]
fn schedule_hard_cancel_after_a_verified_boundary_still_blocks_publication() {
    let package = deterministic_job_shop(16);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 2,
        seed: 71,
        limits: SolveLimits { iterations: Some(8_192), ..SolveLimits::default() },
        profile: true,
        ..SolveRequest::default()
    };
    audit_interrupt_next_collection_final_replay(true);

    let result = qayd::solve(&package, &request, &mut IgnoreEvents).expect("hard cancellation still returns a normal solve result");

    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none());
    assert!(result.message().is_some_and(|message| message.contains("ExternalCancellation")));
}

fn run_tsab_slices(
    schedule: &Schedule,
    seed: u64,
    slices: &[u64],
) -> (CollectionSolution, Vec<ScheduleConstructionMetrics>, ScheduleSearchSession) {
    let stop = AtomicBool::new(false);
    let mut solution = None::<CollectionSolution>;
    let mut session = None::<ScheduleSearchSession>;
    let mut metrics = Vec::new();
    for &work in slices {
        let (next_solution, next_metrics, next_session) =
            solve_schedule_capped_persistent_tsab(schedule, seed, seed, 6, 7, &stop, work, solution.as_ref(), session, &mut |_| {});
        solution = Some(next_solution);
        session = next_session;
        metrics.push(next_metrics);
    }
    (solution.expect("at least one TSAB slice"), metrics, session.expect("JSSP owns a persistent TSAB session"))
}

fn run_tsab_multi_slices(
    schedule: &Schedule,
    seed: u64,
    worker: usize,
    slices: &[u64],
) -> (CollectionSolution, Vec<ScheduleConstructionMetrics>, ScheduleSearchSession) {
    let stop = AtomicBool::new(false);
    let mut solution = None::<CollectionSolution>;
    let mut session = None::<ScheduleSearchSession>;
    let mut metrics = Vec::new();
    for &work in slices {
        let (next_solution, next_metrics, next_session) = solve_schedule_capped_persistent_tsab_multi(
            schedule,
            seed,
            seed,
            worker,
            7,
            &stop,
            work,
            solution.as_ref(),
            session,
            &mut |_| {},
        );
        solution = Some(next_solution);
        session = next_session;
        metrics.push(next_metrics);
    }
    (solution.expect("at least one multi-worker TSAB slice"), metrics, session.expect("JSSP owns a persistent multi-worker TSAB session"))
}

fn latched_tsab_pending_elite() -> (Schedule, u64, CollectionSolution, i64, ScheduleSearchSession) {
    let (schedule, _) = tsab_candidate_fixture();
    let seed = 0;
    let (warm_solution, _, warm_session) = run_tsab_slices(&schedule, seed, &[256, 256]);
    let stop = AtomicBool::new(false);
    let (active_solution, _, active_session) =
        solve_schedule_capped_persistent_tsab(&schedule, seed, seed, 6, 7, &stop, 1, Some(&warm_solution), Some(warm_session), &mut |_| {});

    // Normalize the activation slice so the test hook starts from a fully
    // validated fast state, whether its first four transitions improved or not.
    stop.store(true, Ordering::Release);
    let (latched_solution, _, latched_session) =
        solve_schedule_capped_persistent_tsab(&schedule, seed, seed, 6, 7, &stop, 0, Some(&active_solution), active_session, &mut |_| {});
    stop.store(false, Ordering::Release);
    let (clean_solution, _, clean_session) =
        solve_schedule_capped_persistent_tsab(&schedule, seed, seed, 6, 7, &stop, 0, Some(&latched_solution), latched_session, &mut |_| {});
    let mut clean_session = clean_session.expect("the normalized TSAB session remains resumable");
    assert!(!clean_session.test_tsab_fast_validation_due_on_resume());
    assert_eq!(clean_session.test_tsab_pending_elite_objective(), None);
    clean_session.test_stop_after_fast_pending_elite();

    let (verified_solution, metrics, pending_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        128,
        Some(&clean_solution),
        Some(clean_session),
        &mut |_| {},
    );
    assert!(stop.load(Ordering::Acquire), "the test hook must stop on the first pending elite");
    assert_eq!(metrics.schedule_tsab.fast_pending_promotions, 0);
    let pending_session = pending_session.expect("a stopped exact fast trajectory remains resumable");
    assert!(pending_session.test_tsab_fast_validation_due_on_resume());
    let pending_objective = pending_session.test_tsab_pending_elite_objective().expect("the stop hook retains a pending elite");
    assert_eq!(pending_session.tsab_fast_pending_objective(), Some(pending_objective));
    assert!(pending_objective < verified_solution.objectives[0], "the pending objective must strictly improve the last verified elite");
    (schedule, seed, verified_solution, pending_objective, pending_session)
}

fn exact_two_job_restart_fixture() -> (Schedule, CollectionSolution, CollectionSolution) {
    let schedule = Schedule {
        intervals: [10, 1, 10, 1]
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 22, modes: Vec::new(), optional: false })
            .collect(),
        precedences: vec![(0, 1), (2, 3)],
        resources: vec![Resource::NoOverlap(vec![0, 3]), Resource::NoOverlap(vec![1, 2])],
        minimize_makespan: true,
    };
    let solution = |starts: Vec<i64>, objective| CollectionSolution {
        lists: Vec::new(),
        objectives: vec![objective],
        feasible: true,
        starts,
        presences: vec![true; 4],
        machines: vec![0, 1, 1, 0],
        modes: vec![None; 4],
        bound: None,
    };
    (schedule, solution(vec![0, 10, 11, 21], 22), solution(vec![0, 10, 0, 10], 11))
}

fn tail_corridor_restart_fixture() -> (Schedule, CollectionSolution) {
    let schedule = Schedule {
        intervals: (0..8).map(|_| IntervalVar { duration: 1, horizon: 8, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(3, 4)],
        resources: vec![Resource::NoOverlap(vec![0, 1, 2, 3]), Resource::NoOverlap(vec![4, 5, 6, 7])],
        minimize_makespan: true,
    };
    let incumbent = CollectionSolution {
        lists: Vec::new(),
        objectives: vec![8],
        feasible: true,
        starts: (0..8).map(i64::from).collect(),
        presences: vec![true; 8],
        machines: vec![0, 0, 0, 0, 1, 1, 1, 1],
        modes: vec![None; 8],
        bound: None,
    };
    (schedule, incumbent)
}

fn coordinated_macro_restart_fixture() -> (Schedule, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let durations = [9, 5, 3, 2, 8, 5, 2, 6, 8, 9, 4, 5, 2, 9, 2, 3];
    let horizon = durations.iter().sum();
    let schedule = Schedule {
        intervals: durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect(),
        precedences: vec![(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (12, 13), (13, 14), (14, 15)],
        resources: vec![
            Resource::NoOverlap(vec![2, 5, 9, 13]),
            Resource::NoOverlap(vec![0, 6, 10, 14]),
            Resource::NoOverlap(vec![3, 7, 11, 12]),
            Resource::NoOverlap(vec![1, 4, 8, 15]),
        ],
        minimize_makespan: true,
    };
    // Compact machine ids follow first occurrence in operation order, which
    // maps raw machines [1, 3, 0, 2] to compact ids [0, 1, 2, 3].
    let current = vec![vec![10, 0, 6, 14], vec![8, 4, 15, 1], vec![5, 9, 13, 2], vec![11, 12, 3, 7]];
    let guide = vec![vec![10, 0, 6, 14], vec![8, 1, 4, 15], vec![9, 13, 2, 5], vec![11, 12, 3, 7]];
    (schedule, current, guide)
}

fn rejected_tail_corridor_restart_schedule() -> Schedule {
    const JOBS: usize = 7;
    const MACHINES: usize = 4;
    const SALT: usize = 15;
    let operation_count = JOBS * MACHINES;
    let durations: Vec<i64> =
        (0..operation_count).map(|operation| 1 + i64::try_from((operation * 7 + SALT * 3) % 17).expect("small duration")).collect();
    let horizon = durations.iter().sum();
    let intervals = durations.into_iter().map(|duration| IntervalVar { duration, horizon, modes: Vec::new(), optional: false }).collect();
    let precedences = (0..JOBS)
        .flat_map(|job| (0..MACHINES - 1).map(move |operation| (job * MACHINES + operation, job * MACHINES + operation + 1)))
        .collect();
    let mut assigned = vec![Vec::new(); MACHINES];
    for job in 0..JOBS {
        for operation in 0..MACHINES {
            let machine = (job * 3 + SALT + MACHINES - operation % MACHINES) % MACHINES;
            assigned[machine].push(job * MACHINES + operation);
        }
    }
    Schedule { intervals, precedences, resources: assigned.into_iter().map(Resource::NoOverlap).collect(), minimize_makespan: true }
}

fn forced_tsab_restart_session(
    schedule: &Schedule,
    seed: u64,
    incumbent: &CollectionSolution,
) -> (CollectionSolution, ScheduleSearchSession) {
    let stop = AtomicBool::new(false);
    let (solution, metrics, session) =
        solve_schedule_capped_persistent_tsab(schedule, seed, seed, 6, 7, &stop, 0, Some(incumbent), None, &mut |_| {});
    assert_eq!(metrics.work_steps, 0);
    let mut session = session.expect("a JSSP fixture owns a persistent session");
    session.test_force_tsab_restart_due();
    (solution, session)
}

fn naturally_due_tsab_restart(schedule: &Schedule, seed: u64) -> (CollectionSolution, ScheduleSearchSession) {
    let (mut solution, _, mut session) = run_tsab_slices(schedule, seed, &[256, 256, 16]);
    for _ in 0..=3 {
        let (n5_work, _, _, _, _, scan_empty) = session.test_tsab_restart_state();
        if n5_work >= 16 && scan_empty {
            return (solution, session);
        }
        let stop = AtomicBool::new(false);
        let (next_solution, _, next_session) =
            solve_schedule_capped_persistent_tsab(schedule, seed, seed, 6, 7, &stop, 1, Some(&solution), Some(session), &mut |_| {});
        solution = next_solution;
        session = next_session.expect("the active TSAB session persists");
    }
    panic!("a K4 shortlist must be empty no later than three probes after the restart becomes due")
}

#[test]
fn tsab_candidate_owner_phase_shortlist_and_admissibility_policies_are_bounded() {
    assert_eq!(audit_tsab_owner_mask(ScheduleJsspSearchStrategy::TsabCandidate, 7), 0x40);
    assert_eq!(audit_tsab_owner_mask(ScheduleJsspSearchStrategy::TsabMultiCandidate, 7), 0x7f);
    assert_eq!(audit_tsab_owner_mask(ScheduleJsspSearchStrategy::Legacy, 7), 0);
    assert_eq!(audit_tsab_owner_mask(ScheduleJsspSearchStrategy::TsabCandidate, 6), 0);
    assert_eq!(audit_tsab_owner_mask(ScheduleJsspSearchStrategy::TsabMultiCandidate, 8), 0);
    assert_eq!(audit_tsab_phase_policy(), (512, 128, 8, 4), "warmup, burst, tenure, and K4 are fixed");
    assert_eq!(audit_tsab_elite_restart_policy(), (16, 8, 4, 64), "elite restart cadence, credit, K4, and ppm bound are fixed");
    assert_eq!(audit_tsab_shortlist_policy(), (4, 1, true), "streaming top-k retains one tabu candidate and no duplicates");
    assert!(audit_tsab_selection_policy(), "aspiration is strict and all-tabu search resets instead of forcing a commit");
    for worker in 0..7 {
        assert_eq!(
            schedule_worker_allows_rebase(ScheduleJsspSearchStrategy::TsabMultiCandidate, worker),
            matches!(worker, 0 | 6),
            "the multi-owner candidate preserves the historical rebase policy"
        );
        assert_eq!(
            schedule_worker_allows_rebase(ScheduleJsspSearchStrategy::TsabCandidate, worker),
            matches!(worker, 0 | 6),
            "the single-owner candidate preserves the same historical rebase policy"
        );
    }
    let (schedule, seed) = tsab_candidate_fixture();
    assert!(audit_tsab_streaming_reference(&schedule, seed));
    let (matches_reference, generated, shortlisted, heap_bytes) = audit_tsab_n6_restart_streaming_reference(&job_shop_schedule(64), seed);
    assert!(matches_reference, "streaming N6 inserts match the materialized deterministic reference");
    assert_eq!((generated, shortlisted), (122, 4));
    assert_eq!(heap_bytes, 4 * std::mem::size_of::<(bool, u64, ScoredScheduleMove)>());
}

#[test]
fn tsab_multi_candidate_workers_zero_three_and_six_activate_after_the_exact_warmup() {
    let (schedule, seed) = tsab_candidate_fixture();
    let stop = AtomicBool::new(false);
    for worker in [0, 3, 6] {
        let (warm_solution, warm_metrics, warm_session) = run_tsab_multi_slices(&schedule, seed, worker, &[256, 256]);
        assert_eq!(warm_metrics.iter().map(|metrics| metrics.schedule_tsab.legacy_warmup_work_steps).sum::<u64>(), 512);
        assert!(warm_metrics.iter().all(|metrics| metrics.schedule_tsab.owner_worker_mask == 0));
        assert_eq!(warm_session.test_tsab_phase_state().0, ScheduleJsspSearchStrategy::Legacy);
        assert_eq!(warm_session.test_tsab_phase_state().2, 512);
        assert_eq!(warm_session.test_tsab_requested_strategy(), ScheduleJsspSearchStrategy::TsabMultiCandidate);

        let (_, active_metrics, active_session) = solve_schedule_capped_persistent_tsab_multi(
            &schedule,
            seed,
            seed,
            worker,
            7,
            &stop,
            1,
            Some(&warm_solution),
            Some(warm_session),
            &mut |_| {},
        );
        assert_eq!(active_metrics.schedule_tsab.owner_worker_mask, 1 << worker);
        assert_eq!(active_metrics.schedule_tsab.activations, 1);
        assert_eq!(active_metrics.schedule_tsab.legacy_warmup_work_steps, 0);
        let active_session = active_session.expect("the activated multi-worker TSAB session remains persistent");
        assert_eq!(active_session.test_tsab_phase_state().0, ScheduleJsspSearchStrategy::TsabCandidate);
        assert_eq!(active_session.test_tsab_requested_strategy(), ScheduleJsspSearchStrategy::TsabMultiCandidate);
        assert_eq!(active_session.test_tsab_canonical_path().0, CanonicalPathPolicy::Historical);
    }
}

#[test]
fn tsab_multi_candidate_arbitrary_splits_preserve_worker_three_trajectory() {
    let (schedule, seed) = tsab_candidate_fixture();
    let (monolithic_solution, monolithic_metrics, monolithic_session) = run_tsab_multi_slices(&schedule, seed, 3, &[256, 256, 40]);
    let (split_solution, split_metrics, split_session) =
        run_tsab_multi_slices(&schedule, seed, 3, &[1, 3, 17, 29, 61, 113, 256, 32, 15, 1, 1, 3, 4, 16]);

    assert_schedule_solution_eq(&split_solution, &monolithic_solution);
    assert!(split_session.test_same_tabu_trajectory(&monolithic_session));
    assert_eq!(split_session.test_tsab_requested_strategy(), ScheduleJsspSearchStrategy::TsabMultiCandidate);
    assert_eq!(split_session.test_tsab_phase_state().0, ScheduleJsspSearchStrategy::TsabCandidate);

    let sum =
        |metrics: &[ScheduleConstructionMetrics], field: fn(&ScheduleConstructionMetrics) -> u64| metrics.iter().map(field).sum::<u64>();
    assert_eq!(sum(&split_metrics, |m| m.work_steps), sum(&monolithic_metrics, |m| m.work_steps));
    assert_eq!(
        sum(&split_metrics, |m| m.schedule_tsab.legacy_warmup_work_steps),
        sum(&monolithic_metrics, |m| m.schedule_tsab.legacy_warmup_work_steps)
    );
    assert_eq!(sum(&split_metrics, |m| m.schedule_tsab.burst_work_units), sum(&monolithic_metrics, |m| m.schedule_tsab.burst_work_units));
    assert_eq!(sum(&split_metrics, |m| m.schedule_tsab.fast_work_units), sum(&monolithic_metrics, |m| m.schedule_tsab.fast_work_units));
    assert_eq!(
        sum(&split_metrics, |m| m.schedule_tsab.restart_work_units),
        sum(&monolithic_metrics, |m| m.schedule_tsab.restart_work_units)
    );
    assert_eq!(sum(&split_metrics, |m| m.schedule_tsab.fast_transitions), sum(&monolithic_metrics, |m| m.schedule_tsab.fast_transitions));
    assert_eq!(
        sum(&split_metrics, |m| m.schedule_tsab.fast_full_validations),
        sum(&monolithic_metrics, |m| m.schedule_tsab.fast_full_validations)
    );
    assert_eq!(sum(&split_metrics, |m| m.schedule_tsab.activations), 1);
    assert_eq!(split_metrics.iter().fold(0, |mask, metrics| mask | metrics.schedule_tsab.owner_worker_mask), 1 << 3);
}

#[test]
fn tsab_multi_worker_six_preserves_the_single_owner_historical_anchor() {
    let (schedule, seed) = tsab_candidate_fixture();
    let slices = [256, 256, 128, 64];
    let (single_solution, single_metrics, single_session) = run_tsab_slices(&schedule, seed, &slices);
    let (multi_solution, multi_metrics, multi_session) = run_tsab_multi_slices(&schedule, seed, 6, &slices);

    assert_schedule_solution_eq(&multi_solution, &single_solution);
    assert_eq!(multi_session.test_tsab_canonical_path(), single_session.test_tsab_canonical_path());
    assert_eq!(multi_session.test_tsab_canonical_path().0, CanonicalPathPolicy::Historical);
    assert_eq!(multi_session.test_tsab_phase_state(), single_session.test_tsab_phase_state());
    for (multi, single) in multi_metrics.iter().zip(&single_metrics) {
        assert_eq!(multi.work_steps, single.work_steps);
        assert_eq!(multi.moves_considered, single.moves_considered);
        assert_eq!(multi.moves_accepted, single.moves_accepted);
        let mut multi_tsab = multi.schedule_tsab;
        let mut single_tsab = single.schedule_tsab;
        multi_tsab.fast_elapsed = Duration::ZERO;
        single_tsab.fast_elapsed = Duration::ZERO;
        assert_eq!(multi_tsab, single_tsab);
    }
}

#[test]
fn tsab_candidate_workers_zero_through_five_are_bit_identical_to_legacy() {
    let schedule = diversified_job_shop();
    let stop = AtomicBool::new(false);
    for worker in 0..=5 {
        let mut legacy_reports = Vec::new();
        let (legacy_solution, legacy_metrics, legacy_session) =
            solve_schedule_capped_persistent(&schedule, 127, 127, worker, 7, &stop, 16, None, None, false, &mut |objective| {
                legacy_reports.push(objective)
            });
        let mut strategy_reports = Vec::new();
        let (strategy_solution, strategy_metrics, strategy_session) =
            solve_schedule_capped_persistent_tsab(&schedule, 127, 127, worker, 7, &stop, 16, None, None, &mut |objective| {
                strategy_reports.push(objective)
            });
        assert_schedule_solution_eq(&strategy_solution, &legacy_solution);
        assert_eq!(strategy_reports, legacy_reports);
        assert_eq!(strategy_metrics.moves_considered, legacy_metrics.moves_considered);
        assert_eq!(strategy_metrics.moves_accepted, legacy_metrics.moves_accepted);
        assert_eq!(strategy_metrics.tabu_steps, legacy_metrics.tabu_steps);
        assert_eq!(strategy_metrics.schedule_tsab, Default::default());
        let (n5_work, restart_credit, restart_epoch, _, _, _) =
            strategy_session.as_ref().expect("strategy session").test_tsab_restart_state();
        assert_eq!((n5_work, restart_credit, restart_epoch), (0, 0, 0));
        assert!(strategy_session
            .as_ref()
            .zip(legacy_session.as_ref())
            .is_some_and(|(strategy, legacy)| strategy.test_same_tabu_trajectory(legacy)));
    }
}

#[test]
fn tsab_candidate_warms_for_exactly_two_256_unit_boundaries_then_activates_once() {
    let (schedule, seed) = tsab_candidate_fixture();
    let (warm_solution, warm_metrics, warm_session) = run_tsab_slices(&schedule, seed, &[256, 256]);
    for metrics in &warm_metrics {
        assert_eq!(metrics.work_steps, 256);
        assert_eq!(metrics.schedule_tsab.legacy_warmup_work_steps, 256);
        assert_eq!(metrics.schedule_tsab.owner_worker_mask, 0);
        assert_eq!(metrics.schedule_tsab.activations, 0);
        assert_eq!(metrics.schedule_tsab.burst_work_units, 0);
        assert_eq!(metrics.scored_island_profile_mask, 1 << 6);
    }
    let (phase, boundaries, warmup, activation_boundary, warm_current, warm_elite, _, _, _) = warm_session.test_tsab_phase_state();
    assert_eq!(phase, ScheduleJsspSearchStrategy::Legacy);
    assert_eq!(boundaries, 2);
    assert_eq!(warmup, 512);
    assert_eq!(activation_boundary, None);

    let stop = AtomicBool::new(false);
    let (active_solution, active_metrics, active_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        128,
        Some(&warm_solution),
        Some(warm_session),
        &mut |_| {},
    );
    let tsab = active_metrics.schedule_tsab;
    assert_eq!(active_metrics.work_steps, 128);
    assert_eq!(tsab.owner_worker_mask, 1 << 6);
    assert_eq!(tsab.activations, 1);
    assert_eq!(tsab.activation_boundary, Some(2));
    assert_eq!(tsab.legacy_warmup_work_steps, 0);
    assert_eq!(tsab.activation_rebases, 0);
    assert_eq!(tsab.activation_objective, Some(warm_current));
    assert!(
        active_solution.objectives.first().is_some_and(|&objective| objective <= warm_elite),
        "active search keeps the verified elite monotone"
    );
    assert_eq!(tsab.active_boundaries, 1);
    assert_eq!(tsab.burst_work_limit, 128);
    assert_eq!(tsab.burst_work_units, 128);
    assert_eq!(tsab.ranked, tsab.n5_generated);
    assert!(tsab.delta_probes <= tsab.shortlists.saturating_mul(4));
    assert!(tsab.full_oracle_commits <= tsab.selections);
    assert_eq!(tsab.best_committed_objective.is_some(), tsab.improving_commits != 0);
    assert_eq!((tsab.fast_eligible, tsab.fast_enabled, tsab.fast_disabled), (1, 1, 0));
    assert!(tsab.restart_attempts > 0);
    assert_eq!(tsab.fast_work_units.saturating_add(tsab.restart_work_units), tsab.burst_work_units);
    assert!(tsab.fast_attempts > 0);
    assert_eq!(tsab.fast_transitions, tsab.fast_commits);
    assert!(tsab.fast_commits <= tsab.fast_attempts);
    assert!(tsab.fast_full_validations > 0);
    assert_eq!(tsab.fast_oracle_mismatches, 0);
    assert_eq!(tsab.fast_pending_discards, 0);
    assert!(tsab.fast_date_changes > 0);
    assert!(tsab.fast_queue_pops >= tsab.fast_date_changes);
    assert!(tsab.fast_elapsed > Duration::ZERO);
    assert!(tsab.fast_workspace_peak_bytes > 0);
    let active_session = active_session.expect("active session");
    let (phase, boundaries, warmup, activation_boundary, _, _, _, direct_credit, tenure) = active_session.test_tsab_phase_state();
    assert_eq!(phase, ScheduleJsspSearchStrategy::TsabCandidate);
    assert_eq!((boundaries, warmup, activation_boundary, direct_credit, tenure), (3, 512, Some(2), 0, 8));
    assert_eq!(active_session.test_tsab_canonical_path().0, CanonicalPathPolicy::Historical);

    let (_, next_metrics, next_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        1_000,
        Some(&active_solution),
        Some(active_session),
        &mut |_| {},
    );
    assert_eq!(next_metrics.work_steps, 128, "each active invocation is capped independently");
    assert_eq!(next_metrics.schedule_tsab.activations, 0, "the phase transition is unique");
    assert_eq!(next_metrics.schedule_tsab.active_boundaries, 1);
    assert_eq!(next_metrics.schedule_tsab.burst_work_units, 128);
    assert_eq!(next_session.expect("active session").test_tsab_phase_state().0, ScheduleJsspSearchStrategy::TsabCandidate);
}

#[test]
fn tsab_fast_batch_executes_at_least_ten_times_more_transitions_per_equal_work_budget() {
    let (schedule, seed) = tsab_candidate_fixture();
    let (fast_warm_solution, _, fast_warm_session) = run_tsab_slices(&schedule, seed, &[256, 256]);
    let (reference_warm_solution, _, reference_warm_session) = run_tsab_slices(&schedule, seed, &[256, 256]);
    assert_schedule_solution_eq(&fast_warm_solution, &reference_warm_solution);

    let stop = AtomicBool::new(false);
    let (fast_base, _, fast_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        1,
        Some(&fast_warm_solution),
        Some(fast_warm_session),
        &mut |_| {},
    );
    let (reference_base, _, reference_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        1,
        Some(&reference_warm_solution),
        Some(reference_warm_session),
        &mut |_| {},
    );
    assert_schedule_solution_eq(&fast_base, &reference_base);
    let fast_session = fast_session.expect("fast TSAB session activates");
    let mut reference_session = reference_session.expect("reference TSAB session activates");
    reference_session.test_disable_tsab_fast_path();

    let work_units = audit_tsab_elite_restart_policy().0.saturating_sub(1);
    let fast_started = Instant::now();
    let (fast_solution, fast_metrics, fast_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        work_units,
        Some(&fast_base),
        Some(fast_session),
        &mut |_| {},
    );
    let fast_wall = fast_started.elapsed();
    let reference_started = Instant::now();
    let (reference_solution, reference_metrics, reference_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        work_units,
        Some(&reference_base),
        Some(reference_session),
        &mut |_| {},
    );
    let reference_wall = reference_started.elapsed();

    assert!(fast_solution.feasible && reference_solution.feasible);
    let replay_model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule.clone()),
    };
    assert_eq!(verify_collection_solution(&replay_model, &fast_solution).unwrap(), fast_solution.objectives);
    assert_eq!(verify_collection_solution(&replay_model, &reference_solution).unwrap(), reference_solution.objectives);
    assert!(fast_session.is_some() && reference_session.is_some());
    assert_eq!(fast_metrics.work_steps, work_units);
    assert_eq!(reference_metrics.work_steps, work_units);
    let fast_transitions = fast_metrics.schedule_tsab.fast_transitions;
    let reference_transitions = reference_metrics.schedule_tsab.full_oracle_commits;
    assert!(reference_transitions > 0, "reference K4 kernel must commit at least one transition");
    assert!(
        fast_transitions >= reference_transitions.saturating_mul(10),
        "fast strict N5 must deliver at least 10x transitions under equal work: fast={fast_transitions}, reference={reference_transitions}"
    );
    assert!(fast_metrics.schedule_tsab.fast_full_validations <= fast_transitions / 16 + 1);
    eprintln!(
        "strict-N5 synthetic batch: fast_transitions={fast_transitions}, reference_transitions={reference_transitions}, ratio={:.2}, fast_full_validations={}, fast_wall_ms={}, reference_wall_ms={}",
        fast_transitions as f64 / reference_transitions as f64,
        fast_metrics.schedule_tsab.fast_full_validations,
        fast_wall.as_millis(),
        reference_wall.as_millis()
    );
}

#[test]
fn tsab_fast_work_cap_recovery_is_counted_without_a_transactional_fallback() {
    let (schedule, seed) = tsab_candidate_fixture();
    let (warm_solution, _, warm_session) = run_tsab_slices(&schedule, seed, &[256, 256]);
    let stop = AtomicBool::new(false);
    let (active_solution, _, active_session) =
        solve_schedule_capped_persistent_tsab(&schedule, seed, seed, 6, 7, &stop, 1, Some(&warm_solution), Some(warm_session), &mut |_| {});
    let mut active_session = active_session.expect("TSAB fast path activates");
    active_session.test_force_tsab_fast_work_cap(0);

    let (solution, metrics, session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        16,
        Some(&active_solution),
        Some(active_session),
        &mut |_| {},
    );
    let fast = metrics.schedule_tsab;
    assert!(fast.fast_attempts > 0);
    assert_eq!(fast.fast_work_cap_recoveries, fast.fast_attempts);
    assert_eq!(fast.fast_fallbacks, 0, "a recovered artificial WorkCap is not a transactional fallback");
    assert_eq!(fast.fast_oracle_mismatches, 0);
    assert!(fast.fast_full_validations > 0);
    assert!(session.is_some(), "validated recovery keeps the persistent session");

    let replay_model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule),
    };
    assert_eq!(verify_collection_solution(&replay_model, &solution).unwrap(), solution.objectives);
}

#[test]
fn tsab_candidate_arbitrary_warmup_and_active_splits_preserve_the_trajectory() {
    let (schedule, seed) = tsab_candidate_fixture();
    let (mono_warm_solution, _, mono_warm_session) = run_tsab_slices(&schedule, seed, &[256, 256]);
    let (split_warm_solution, split_warm_metrics, split_warm_session) = run_tsab_slices(&schedule, seed, &[1, 3, 17, 29, 61, 113, 288, 32]);
    assert_schedule_solution_eq(&split_warm_solution, &mono_warm_solution);
    assert!(split_warm_session.test_same_tabu_trajectory(&mono_warm_session));
    assert_eq!(split_warm_metrics.iter().map(|metrics| metrics.schedule_tsab.legacy_warmup_work_steps).sum::<u64>(), 512);

    let stop = AtomicBool::new(false);
    let (mono_active_solution, mono_metrics, mono_active_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        40,
        Some(&mono_warm_solution),
        Some(mono_warm_session),
        &mut |_| {},
    );
    assert_eq!(mono_metrics.schedule_tsab.fast_work_units, 32);
    assert_eq!(mono_metrics.schedule_tsab.restart_work_units, 8);
    assert_eq!(mono_metrics.schedule_tsab.restart_attempts, 1);
    assert_eq!(mono_metrics.schedule_tsab.elite_restarts, 1);
    assert_eq!(mono_metrics.schedule_tsab.burst_work_units, 40);
    assert_eq!(
        mono_metrics.schedule_tsab.fast_work_units.saturating_add(mono_metrics.schedule_tsab.restart_work_units),
        mono_metrics.schedule_tsab.burst_work_units
    );
    let mut split_solution = split_warm_solution;
    let mut split_session = Some(split_warm_session);
    let mut split_delta_probes = 0u64;
    let mut split_commits = 0u64;
    let mut split_fast_attempts = 0u64;
    let mut split_fast_transitions = 0u64;
    let mut split_fast_validations = 0u64;
    let mut split_fast_promotions = 0u64;
    let mut split_fast_work = 0u64;
    let mut split_restart_work = 0u64;
    let mut split_restart_attempts = 0u64;
    let mut split_restarts = 0u64;
    for work in [15, 1, 1, 3, 4, 16] {
        let (solution, metrics, session) = solve_schedule_capped_persistent_tsab(
            &schedule,
            seed,
            seed,
            6,
            7,
            &stop,
            work,
            Some(&split_solution),
            split_session,
            &mut |_| {},
        );
        split_solution = solution;
        split_session = session;
        split_delta_probes = split_delta_probes.saturating_add(metrics.schedule_tsab.delta_probes);
        split_commits = split_commits.saturating_add(metrics.schedule_tsab.full_oracle_commits);
        split_fast_attempts = split_fast_attempts.saturating_add(metrics.schedule_tsab.fast_attempts);
        split_fast_transitions = split_fast_transitions.saturating_add(metrics.schedule_tsab.fast_transitions);
        split_fast_validations = split_fast_validations.saturating_add(metrics.schedule_tsab.fast_full_validations);
        split_fast_promotions = split_fast_promotions.saturating_add(metrics.schedule_tsab.fast_pending_promotions);
        split_fast_work = split_fast_work.saturating_add(metrics.schedule_tsab.fast_work_units);
        split_restart_work = split_restart_work.saturating_add(metrics.schedule_tsab.restart_work_units);
        split_restart_attempts = split_restart_attempts.saturating_add(metrics.schedule_tsab.restart_attempts);
        split_restarts = split_restarts.saturating_add(metrics.schedule_tsab.elite_restarts);
    }
    assert_schedule_solution_eq(&split_solution, &mono_active_solution);
    assert!(split_session
        .as_ref()
        .zip(mono_active_session.as_ref())
        .is_some_and(|(split, monolithic)| split.test_same_tabu_trajectory(monolithic)));
    assert_eq!(split_delta_probes, mono_metrics.schedule_tsab.delta_probes);
    assert_eq!(split_commits, mono_metrics.schedule_tsab.full_oracle_commits);
    assert_eq!(split_fast_attempts, mono_metrics.schedule_tsab.fast_attempts);
    assert_eq!(split_fast_transitions, mono_metrics.schedule_tsab.fast_transitions);
    assert_eq!(split_fast_validations, mono_metrics.schedule_tsab.fast_full_validations);
    assert_eq!(split_fast_promotions, mono_metrics.schedule_tsab.fast_pending_promotions);
    assert_eq!(split_fast_work, mono_metrics.schedule_tsab.fast_work_units);
    assert_eq!(split_restart_work, mono_metrics.schedule_tsab.restart_work_units);
    assert_eq!(split_restart_attempts, mono_metrics.schedule_tsab.restart_attempts);
    assert_eq!(split_restarts, mono_metrics.schedule_tsab.elite_restarts);
    assert_eq!(split_fast_work.saturating_add(split_restart_work), 40);
}

#[test]
fn tsab_candidate_stop_at_activation_is_atomic_and_resumable() {
    let (schedule, seed) = tsab_candidate_fixture();
    let (warm_solution, _, warm_session) = run_tsab_slices(&schedule, seed, &[256, 256]);
    let before = warm_session.test_tsab_phase_state();
    let stop = AtomicBool::new(true);
    let (stopped_solution, stopped_metrics, stopped_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        128,
        Some(&warm_solution),
        Some(warm_session),
        &mut |_| {},
    );
    assert_schedule_solution_eq(&stopped_solution, &warm_solution);
    assert_eq!(stopped_metrics.work_steps, 0);
    assert_eq!(stopped_metrics.schedule_tsab.activations, 0);
    assert_eq!(stopped_session.as_ref().expect("stopped session is retained").test_tsab_phase_state(), before);

    stop.store(false, Ordering::Release);
    let (_, resumed_metrics, resumed_session) =
        solve_schedule_capped_persistent_tsab(&schedule, seed, seed, 6, 7, &stop, 1, Some(&stopped_solution), stopped_session, &mut |_| {});
    assert_eq!(resumed_metrics.schedule_tsab.activations, 1);
    assert_eq!(resumed_metrics.schedule_tsab.activation_boundary, Some(2));
    assert_eq!(resumed_metrics.schedule_tsab.burst_work_units, 1);
    assert_eq!(resumed_session.expect("resumed session").test_tsab_phase_state().0, ScheduleJsspSearchStrategy::TsabCandidate);
}

#[test]
fn tsab_fast_stop_latches_full_validation_before_any_resumed_attempt() {
    let (schedule, seed) = tsab_candidate_fixture();
    let (warm_solution, _, warm_session) = run_tsab_slices(&schedule, seed, &[256, 256]);
    let stop = AtomicBool::new(false);
    let (active_solution, _, active_session) =
        solve_schedule_capped_persistent_tsab(&schedule, seed, seed, 6, 7, &stop, 1, Some(&warm_solution), Some(warm_session), &mut |_| {});
    let mut active_session = active_session.expect("TSAB fast path activates");
    active_session.test_stop_after_fast_attempts(1);

    let (stopped_solution, stopped_metrics, stopped_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        8,
        Some(&active_solution),
        Some(active_session),
        &mut |_| {},
    );
    assert!(stop.load(Ordering::Acquire), "test hook must stop after one committed fast attempt");
    assert_eq!(stopped_metrics.schedule_tsab.fast_transitions, 1);
    assert_eq!(stopped_metrics.schedule_tsab.fast_full_validations, 0);
    let stopped_session = stopped_session.expect("an exact transactional fast state remains resumable");
    assert!(stopped_session.test_tsab_fast_validation_due_on_resume());

    stop.store(false, Ordering::Release);
    let (validated_solution, validation_metrics, validated_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        0,
        Some(&stopped_solution),
        Some(stopped_session),
        &mut |_| {},
    );
    assert_eq!(validation_metrics.work_steps, 0);
    assert_eq!(validation_metrics.schedule_tsab.fast_attempts, 0, "resume validation must precede every new candidate attempt");
    assert_eq!(validation_metrics.schedule_tsab.fast_full_validations, 1);
    assert_eq!(validation_metrics.schedule_tsab.fast_oracle_mismatches, 0);
    let validated_session = validated_session.expect("successful oracle validation keeps the session");
    assert!(!validated_session.test_tsab_fast_validation_due_on_resume());

    let replay_model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule.clone()),
    };
    assert_eq!(verify_collection_solution(&replay_model, &stopped_solution).unwrap(), stopped_solution.objectives);
    assert_eq!(verify_collection_solution(&replay_model, &validated_solution).unwrap(), validated_solution.objectives);

    let (_, resumed_metrics, resumed_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        1,
        Some(&validated_solution),
        Some(validated_session),
        &mut |_| {},
    );
    assert!(resumed_metrics.schedule_tsab.fast_attempts > 0, "new work resumes only after the latched oracle boundary");
    assert!(resumed_session.is_some());
}

#[test]
fn tsab_fast_finalization_stop_promotes_an_exact_pending_elite() {
    let (schedule, seed, verified_solution, pending_objective, pending_session) = latched_tsab_pending_elite();
    let search_stop = AtomicBool::new(true);
    let finalization_stop = AtomicBool::new(false);
    let mut reports = Vec::new();

    let (solution, metrics, session) = solve_schedule_capped_persistent_tsab_with_finalization_stop(
        &schedule,
        seed,
        seed,
        6,
        7,
        &search_stop,
        &finalization_stop,
        0,
        Some(&verified_solution),
        Some(pending_session),
        &mut |objective| reports.push(objective),
    );

    assert_eq!(solution.objectives, vec![pending_objective]);
    assert_eq!(metrics.work_steps, 0);
    assert_eq!(metrics.schedule_tsab.fast_attempts, 0);
    assert_eq!(metrics.schedule_tsab.fast_full_validations, 1);
    assert_eq!(metrics.schedule_tsab.fast_pending_promotions, 1);
    assert_eq!(metrics.schedule_tsab.fast_oracle_mismatches, 0);
    assert_eq!(metrics.schedule_tsab.fast_pending_discards, 0);
    assert_eq!(reports, vec![pending_objective]);
    eprintln!(
        "TSAB terminal validation: verified={}, pending={}, returned={}, validations={}, promotions={}",
        verified_solution.objectives[0],
        pending_objective,
        solution.objectives[0],
        metrics.schedule_tsab.fast_full_validations,
        metrics.schedule_tsab.fast_pending_promotions
    );
    let session = session.expect("terminal validation keeps the exact TSAB session");
    assert!(!session.test_tsab_fast_validation_due_on_resume());
    assert_eq!(session.test_tsab_pending_elite_objective(), None);

    let replay_model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule),
    };
    assert_eq!(verify_collection_solution(&replay_model, &solution).unwrap(), vec![pending_objective]);
}

#[test]
fn tsab_fast_finalization_already_stopped_keeps_the_verified_elite_and_pending_latch() {
    let (schedule, seed, verified_solution, pending_objective, pending_session) = latched_tsab_pending_elite();
    let search_stop = AtomicBool::new(true);
    let finalization_stop = AtomicBool::new(true);
    let mut reports = Vec::new();

    let (solution, metrics, session) = solve_schedule_capped_persistent_tsab_with_finalization_stop(
        &schedule,
        seed,
        seed,
        6,
        7,
        &search_stop,
        &finalization_stop,
        0,
        Some(&verified_solution),
        Some(pending_session),
        &mut |objective| reports.push(objective),
    );

    assert_schedule_solution_eq(&solution, &verified_solution);
    assert_eq!(metrics.schedule_tsab.fast_full_validations, 0);
    assert_eq!(metrics.schedule_tsab.fast_pending_promotions, 0);
    assert!(reports.is_empty());
    let session = session.expect("the untouched pending trajectory remains resumable");
    assert!(session.test_tsab_fast_validation_due_on_resume());
    assert_eq!(session.test_tsab_pending_elite_objective(), Some(pending_objective));
}

#[test]
fn tsab_fast_parent_stop_during_terminal_oracle_preserves_then_resumes_the_verified_elite() {
    let (schedule, seed, verified_solution, pending_objective, mut pending_session) = latched_tsab_pending_elite();
    pending_session.test_stop_during_tsab_fast_validation();
    let search_stop = AtomicBool::new(true);
    let finalization_stop = AtomicBool::new(false);

    let (interrupted_solution, interrupted_metrics, interrupted_session) = solve_schedule_capped_persistent_tsab_with_finalization_stop(
        &schedule,
        seed,
        seed,
        6,
        7,
        &search_stop,
        &finalization_stop,
        0,
        Some(&verified_solution),
        Some(pending_session),
        &mut |_| {},
    );
    assert!(finalization_stop.load(Ordering::Acquire), "the oracle hook must arm the parent stop");
    assert_schedule_solution_eq(&interrupted_solution, &verified_solution);
    assert_eq!(interrupted_metrics.schedule_tsab.fast_full_validations, 0);
    assert_eq!(interrupted_metrics.schedule_tsab.fast_pending_promotions, 0);
    let interrupted_session = interrupted_session.expect("an interrupted side oracle keeps the exact live trajectory");
    assert!(interrupted_session.test_tsab_fast_validation_due_on_resume());
    assert_eq!(interrupted_session.test_tsab_pending_elite_objective(), Some(pending_objective));

    finalization_stop.store(false, Ordering::Release);
    let (resumed_solution, resumed_metrics, resumed_session) = solve_schedule_capped_persistent_tsab_with_finalization_stop(
        &schedule,
        seed,
        seed,
        6,
        7,
        &search_stop,
        &finalization_stop,
        0,
        Some(&interrupted_solution),
        Some(interrupted_session),
        &mut |_| {},
    );
    assert_eq!(resumed_solution.objectives, vec![pending_objective]);
    assert_eq!(resumed_metrics.schedule_tsab.fast_full_validations, 1);
    assert_eq!(resumed_metrics.schedule_tsab.fast_pending_promotions, 1);
    let resumed_session = resumed_session.expect("a later reserve can finish the exact validation");
    assert!(!resumed_session.test_tsab_fast_validation_due_on_resume());
    assert_eq!(resumed_session.test_tsab_pending_elite_objective(), None);
}

#[test]
fn separate_finalization_stop_is_bit_identical_for_legacy_workers() {
    let schedule = diversified_job_shop();
    let baseline_stop = AtomicBool::new(false);
    let mut baseline_reports = Vec::new();
    let (baseline_solution, baseline_metrics, baseline_session) =
        solve_schedule_capped_persistent_tsab(&schedule, 127, 127, 0, 7, &baseline_stop, 16, None, None, &mut |objective| {
            baseline_reports.push(objective)
        });
    let search_stop = AtomicBool::new(false);
    let finalization_stop = AtomicBool::new(true);
    let mut separate_reports = Vec::new();
    let (separate_solution, separate_metrics, separate_session) = solve_schedule_capped_persistent_tsab_with_finalization_stop(
        &schedule,
        127,
        127,
        0,
        7,
        &search_stop,
        &finalization_stop,
        16,
        None,
        None,
        &mut |objective| separate_reports.push(objective),
    );

    assert_schedule_solution_eq(&separate_solution, &baseline_solution);
    assert_eq!(separate_reports, baseline_reports);
    assert_eq!(separate_metrics.moves_considered, baseline_metrics.moves_considered);
    assert_eq!(separate_metrics.moves_accepted, baseline_metrics.moves_accepted);
    assert_eq!(separate_metrics.tabu_steps, baseline_metrics.tabu_steps);
    assert_eq!(separate_metrics.schedule_tsab, Default::default());
    assert!(separate_session
        .as_ref()
        .zip(baseline_session.as_ref())
        .is_some_and(|(separate, baseline)| separate.test_same_tabu_trajectory(baseline)));
}

#[test]
fn tsab_candidate_requested_on_non_jssp_fails_closed_for_worker_six() {
    let durations = [2, 1, 3, 2, 4, 1, 2, 3];
    let schedule = Schedule {
        intervals: durations
            .into_iter()
            .map(|duration| IntervalVar { duration, horizon: 18, modes: Vec::new(), optional: false })
            .collect(),
        precedences: vec![(0, 4), (1, 5), (2, 6), (3, 7)],
        resources: vec![Resource::Cumulative { demands: (0..durations.len()).map(|activity| (activity, 1)).collect(), capacity: 2 }],
        minimize_makespan: true,
    };
    let stop = AtomicBool::new(false);
    let (solution, metrics, session) = solve_schedule_capped_persistent_tsab(&schedule, 13, 13, 6, 7, &stop, 8, None, None, &mut |_| {});
    assert!(!solution.feasible);
    assert!(session.is_none());
    assert_eq!(metrics.schedule_tsab, Default::default());
}

#[test]
fn tsab_candidate_activation_rebase_uses_the_fresh_third_boundary_incumbent() {
    let (schedule, seed) = tsab_candidate_fixture();
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).expect("recognition completes").expect("fixture is JSSP");
    let mut constructor_solutions = Vec::new();
    for candidate_seed in 0..128u64 {
        for rule in [
            DispatchRule::EarliestStart,
            DispatchRule::ShortestProcessingTime,
            DispatchRule::LongestProcessingTime,
            DispatchRule::MostWorkRemaining,
            DispatchRule::Randomized,
        ] {
            if let Some(state) =
                JobShopState::giffler_thompson(&problem, candidate_seed, rule, &stop).expect("constructor is not interrupted")
            {
                constructor_solutions.push(state.to_solution());
            }
        }
    }
    constructor_solutions.sort_by_key(|solution| solution.objectives[0]);
    constructor_solutions.dedup_by_key(|solution| solution.objectives[0]);

    let (first_solution, _, first_session) = run_tsab_slices(&schedule, seed, &[256]);
    let stale = constructor_solutions.last().expect("at least one constructor solution").clone();
    let (_, second_metrics, second_session) =
        solve_schedule_capped_persistent_tsab(&schedule, seed, seed, 6, 7, &stop, 256, Some(&stale), Some(first_session), &mut |_| {});
    assert_eq!(second_metrics.schedule_tsab.activations, 0);
    let second_session = second_session.expect("second warmup boundary retains the session");
    let elite_before = second_session.test_tsab_phase_state().5;
    let fresh = constructor_solutions
        .iter()
        .find(|solution| solution.objectives[0] < elite_before)
        .unwrap_or_else(|| panic!("fixture needs a constructor better than warmup elite {elite_before}"))
        .clone();
    assert_ne!(fresh.starts, stale.starts, "fresh and stale snapshots must be distinguishable");

    let mut activation_reports = Vec::new();
    let (_, activation_metrics, activation_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        1,
        Some(&fresh),
        Some(second_session),
        &mut |objective| activation_reports.push(objective),
    );
    assert_eq!(activation_metrics.schedule_tsab.activations, 1);
    assert_eq!(activation_metrics.schedule_tsab.activation_rebases, 1);
    assert_eq!(activation_metrics.schedule_tsab.activation_boundary, Some(2));
    assert_eq!(activation_metrics.schedule_tsab.activation_objective, fresh.objectives.first().copied());
    assert!(activation_reports.is_empty(), "activation rebase is not a post-commit improvement publication");
    let activation_session = activation_session.expect("activation retains the session");
    assert_eq!(activation_session.test_tsab_phase_state().5, fresh.objectives[0]);
    assert!(first_solution.feasible);
}

#[test]
fn tsab_elite_restart_becomes_due_after_exactly_sixteen_fast_work_units_and_two_validations() {
    let (schedule, seed) = tsab_candidate_fixture();
    let (warm_solution, _, warm_session) = run_tsab_slices(&schedule, seed, &[256, 256]);
    let stop = AtomicBool::new(false);

    let (before_due_solution, before_due_metrics, before_due_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        15,
        Some(&warm_solution),
        Some(warm_session),
        &mut |_| {},
    );
    assert_eq!(before_due_metrics.schedule_tsab.fast_work_units, 15);
    assert_eq!(before_due_metrics.schedule_tsab.restart_work_units, 0);
    assert_eq!(before_due_metrics.schedule_tsab.restart_attempts, 0);
    assert_eq!(before_due_metrics.schedule_tsab.fast_transitions, 60);
    assert_eq!(before_due_metrics.schedule_tsab.fast_full_validations, 1);
    let before_due_session = before_due_session.expect("fifteen N5 work units retain the session");
    let (n5_work, credit, epoch, _, _, scan_empty) = before_due_session.test_tsab_restart_state();
    assert_eq!((n5_work, credit, epoch, scan_empty), (15, 0, 0, true));

    let (due_solution, due_metrics, due_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        1,
        Some(&before_due_solution),
        Some(before_due_session),
        &mut |_| {},
    );
    assert_eq!(due_metrics.schedule_tsab.fast_work_units, 1);
    assert_eq!(due_metrics.schedule_tsab.restart_work_units, 0);
    assert_eq!(due_metrics.schedule_tsab.restart_attempts, 0);
    assert_eq!(due_metrics.schedule_tsab.fast_transitions, 4);
    assert_eq!(due_metrics.schedule_tsab.fast_full_validations, 1);
    let mut due_session = due_session.expect("the exact restart boundary retains the session");
    let (n5_work, credit, epoch, _, _, scan_empty) = due_session.test_tsab_restart_state();
    assert_eq!((n5_work, credit, epoch, scan_empty), (16, 0, 0, true));
    assert!(!due_session.test_tsab_fast_validation_due_on_resume());
    assert!(due_session.test_tsab_state_matches_full_oracle());

    let (_, restart_metrics, restarted_session) =
        solve_schedule_capped_persistent_tsab(&schedule, seed, seed, 6, 7, &stop, 8, Some(&due_solution), Some(due_session), &mut |_| {});
    assert_eq!(restart_metrics.schedule_tsab.fast_work_units, 0);
    assert_eq!(restart_metrics.schedule_tsab.restart_work_units, 8);
    assert_eq!(restart_metrics.schedule_tsab.restart_attempts, 1);
    assert_eq!(restart_metrics.schedule_tsab.elite_restarts, 1);
    let mut restarted_session = restarted_session.expect("the atomic N6 restart retains the session");
    let (n5_work, credit, epoch, _, _, scan_empty) = restarted_session.test_tsab_restart_state();
    assert_eq!((n5_work, credit, epoch, scan_empty), (0, 0, 1, true));
    assert!(restarted_session.test_tsab_state_matches_full_oracle());
}

#[test]
fn tsab_elite_restart_credit_is_split_invariant_and_local_fallback_is_monotone() {
    let (schedule, seed) = tsab_candidate_fixture();
    let (mono_before, mono_session) = naturally_due_tsab_restart(&schedule, seed);
    let (split_before, split_session) = naturally_due_tsab_restart(&schedule, seed);
    assert_schedule_solution_eq(&split_before, &mono_before);
    assert!(split_session.test_same_tabu_trajectory(&mono_session));
    let base = mono_session.test_tsab_restart_state().4;
    let objective_limit = base.saturating_add(base.saturating_mul(64) / 1_000_000);
    let stop = AtomicBool::new(false);

    let mut mono_reports = Vec::new();
    let (mono_solution, mono_metrics, mono_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        seed,
        seed,
        6,
        7,
        &stop,
        8,
        Some(&mono_before),
        Some(mono_session),
        &mut |objective| mono_reports.push(objective),
    );
    assert_eq!(mono_metrics.schedule_tsab.restart_work_units, 8);
    assert_eq!(mono_metrics.schedule_tsab.restart_attempts, 1);
    assert_eq!(mono_metrics.schedule_tsab.elite_restarts, 1);
    assert_eq!(mono_metrics.schedule_tsab.restart_global_rebases, 0);
    assert!(mono_metrics.schedule_tsab.restart_delta_probes <= 4);
    assert!(mono_metrics.schedule_tsab.restart_oracle_commits <= 1);
    let mut mono_session = mono_session.expect("monolithic restart retains the session");
    let (n5_work, credit, epoch, current, elite, scan_empty) = mono_session.test_tsab_restart_state();
    assert_eq!((n5_work, credit, epoch, scan_empty), (0, 0, 1, true));
    assert!(current <= objective_limit, "the N6 kick stays within 64 ppm of its local elite base");
    assert!(elite <= base, "the published elite is monotone");
    assert!(mono_session.test_tsab_state_matches_full_oracle());

    let mut split_solution = split_before;
    let mut split_session = Some(split_session);
    let mut split_reports = Vec::new();
    let mut split_restart_work = 0u64;
    let mut split_attempts = 0u64;
    for (index, work) in [1, 3, 4].into_iter().enumerate() {
        let (solution, metrics, session) = solve_schedule_capped_persistent_tsab(
            &schedule,
            seed,
            seed,
            6,
            7,
            &stop,
            work,
            Some(&split_solution),
            split_session,
            &mut |objective| split_reports.push(objective),
        );
        split_solution = solution;
        split_session = session;
        split_restart_work = split_restart_work.saturating_add(metrics.schedule_tsab.restart_work_units);
        split_attempts = split_attempts.saturating_add(metrics.schedule_tsab.restart_attempts);
        let expected_credit = [1, 4, 0][index];
        assert_eq!(split_session.as_ref().expect("split restart session").test_tsab_restart_state().1, expected_credit);
    }
    let mut split_session = split_session.expect("split restart retains the session");
    assert_eq!((split_restart_work, split_attempts), (8, 1));
    assert_schedule_solution_eq(&split_solution, &mono_solution);
    assert_eq!(split_reports, mono_reports);
    assert!(split_session.test_same_tabu_trajectory(&mono_session));
    assert!(split_session.test_tsab_state_matches_full_oracle());
}

#[test]
fn tsab_elite_restart_stop_preserves_partial_credit_and_resumes_exactly() {
    let schedule = job_shop_schedule(64);
    let incumbent = CollectionSolution {
        lists: Vec::new(),
        objectives: vec![64],
        feasible: true,
        starts: (0..64).map(i64::from).collect(),
        presences: vec![true; 64],
        machines: vec![0; 64],
        modes: vec![None; 64],
        bound: None,
    };
    let (test_solution, test_session) = forced_tsab_restart_session(&schedule, 19, &incumbent);
    let (control_solution, control_session) = forced_tsab_restart_session(&schedule, 19, &incumbent);
    let stop = AtomicBool::new(false);
    let (test_solution, charged_metrics, test_session) =
        solve_schedule_capped_persistent_tsab(&schedule, 19, 19, 6, 7, &stop, 3, Some(&test_solution), Some(test_session), &mut |_| {});
    let (control_solution, _, control_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        19,
        19,
        6,
        7,
        &stop,
        3,
        Some(&control_solution),
        Some(control_session),
        &mut |_| {},
    );
    assert_eq!(charged_metrics.schedule_tsab.restart_work_units, 3);
    assert_eq!(test_session.as_ref().expect("charged session").test_tsab_restart_state().1, 3);

    stop.store(true, Ordering::Release);
    let before = test_session.as_ref().expect("charged session").test_tsab_restart_state();
    let (stopped_solution, stopped_metrics, stopped_session) =
        solve_schedule_capped_persistent_tsab(&schedule, 19, 19, 6, 7, &stop, 8, Some(&test_solution), test_session, &mut |_| {});
    assert_schedule_solution_eq(&stopped_solution, &test_solution);
    assert_eq!(stopped_metrics.work_steps, 0);
    assert_eq!(stopped_metrics.schedule_tsab.restart_work_units, 0);
    assert_eq!(stopped_session.as_ref().expect("stopped session").test_tsab_restart_state(), before);
    assert!(stopped_session
        .as_ref()
        .zip(control_session.as_ref())
        .is_some_and(|(stopped, control)| stopped.test_same_tabu_trajectory(control)));

    stop.store(false, Ordering::Release);
    let (resumed_solution, resumed_metrics, resumed_session) =
        solve_schedule_capped_persistent_tsab(&schedule, 19, 19, 6, 7, &stop, 5, Some(&stopped_solution), stopped_session, &mut |_| {});
    let (control_solution, control_metrics, control_session) =
        solve_schedule_capped_persistent_tsab(&schedule, 19, 19, 6, 7, &stop, 5, Some(&control_solution), control_session, &mut |_| {});
    assert_eq!(resumed_metrics.schedule_tsab.restart_work_units, 5);
    assert_eq!(resumed_metrics.schedule_tsab.restart_attempts, 1);
    assert_eq!(resumed_metrics.schedule_tsab.restart_delta_probes, control_metrics.schedule_tsab.restart_delta_probes);
    assert_schedule_solution_eq(&resumed_solution, &control_solution);
    assert!(resumed_session
        .as_ref()
        .zip(control_session.as_ref())
        .is_some_and(|(resumed, control)| resumed.test_same_tabu_trajectory(control)));
}

#[test]
fn tsab_elite_restart_uses_fresh_exact_base_without_requiring_a_kick() {
    let (schedule, stale, fresh) = exact_two_job_restart_fixture();
    let (stale_solution, session) = forced_tsab_restart_session(&schedule, 31, &stale);
    assert_eq!(stale_solution.objectives, vec![22]);
    let stop = AtomicBool::new(false);
    let mut reports = Vec::new();
    let (solution, metrics, session) =
        solve_schedule_capped_persistent_tsab(&schedule, 31, 31, 6, 7, &stop, 8, Some(&fresh), Some(session), &mut |objective| {
            reports.push(objective)
        });
    let mut session = session.expect("fresh restart retains the session");
    assert_eq!(metrics.schedule_tsab.restart_attempts, 1);
    assert_eq!(metrics.schedule_tsab.restart_global_rebases, 1);
    assert_eq!(metrics.schedule_tsab.restart_best_base_objective, Some(11));
    assert_eq!(metrics.schedule_tsab.restart_n6_generated, 0);
    assert_eq!(metrics.schedule_tsab.restart_delta_probes, 0);
    assert_eq!(metrics.schedule_tsab.restart_oracle_commits, 0);
    assert_eq!(metrics.schedule_tsab.elite_restarts, 1);
    assert_eq!(solution.objectives, vec![11]);
    let (_, _, _, current, elite, _) = session.test_tsab_restart_state();
    assert_eq!((current, elite), (11, 11));
    assert!(reports.is_empty(), "a shared-incumbent rebase is not a new post-commit publication");
    assert!(session.test_tsab_state_matches_full_oracle());
}

#[test]
fn tsab_elite_restart_rejects_a_false_fresh_objective_and_falls_back_locally() {
    let (schedule, stale, _) = exact_two_job_restart_fixture();
    let (stale_solution, session) = forced_tsab_restart_session(&schedule, 37, &stale);
    let mut false_fresh = stale.clone();
    false_fresh.objectives[0] = 21;
    let stop = AtomicBool::new(false);
    let mut reports = Vec::new();
    let (solution, metrics, session) =
        solve_schedule_capped_persistent_tsab(&schedule, 37, 37, 6, 7, &stop, 8, Some(&false_fresh), Some(session), &mut |objective| {
            reports.push(objective)
        });
    let mut session = session.expect("local fallback retains the session");
    assert_schedule_solution_eq(&solution, &stale_solution);
    assert_eq!(metrics.schedule_tsab.restart_global_rebases, 0);
    assert_eq!(metrics.schedule_tsab.restart_best_base_objective, Some(22));
    assert_eq!(session.test_tsab_restart_state().3, 22);
    assert!(reports.is_empty());
    assert!(session.test_tsab_state_matches_full_oracle());
}

#[test]
fn tsab_elite_restart_n6_kick_is_deterministic_bounded_and_oracle_sound() {
    let schedule = job_shop_schedule(64);
    let incumbent = CollectionSolution {
        lists: Vec::new(),
        objectives: vec![64],
        feasible: true,
        starts: (0..64).map(i64::from).collect(),
        presences: vec![true; 64],
        machines: vec![0; 64],
        modes: vec![None; 64],
        bound: None,
    };
    let (first_solution, first_session) = forced_tsab_restart_session(&schedule, 43, &incumbent);
    let (second_solution, second_session) = forced_tsab_restart_session(&schedule, 43, &incumbent);
    let stop = AtomicBool::new(false);
    let mut first_reports = Vec::new();
    let (first_solution, first_metrics, first_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        43,
        43,
        6,
        7,
        &stop,
        8,
        Some(&first_solution),
        Some(first_session),
        &mut |objective| first_reports.push(objective),
    );
    let mut second_reports = Vec::new();
    let (second_solution, second_metrics, second_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        43,
        43,
        6,
        7,
        &stop,
        8,
        Some(&second_solution),
        Some(second_session),
        &mut |objective| second_reports.push(objective),
    );
    let mut first_session = first_session.expect("first N6 restart retains the session");
    let second_session = second_session.expect("second N6 restart retains the session");
    let tsab = first_metrics.schedule_tsab;
    assert_eq!(tsab.restart_n6_generated, 122);
    assert_eq!(tsab.restart_delta_probes, 4);
    assert_eq!(tsab.restart_oracle_commits, 1);
    assert_eq!((tsab.n6_kicks, tsab.kick_moves), (1, 1));
    assert_eq!(tsab.restart_best_base_objective, Some(64));
    assert_eq!(tsab.restart_best_kicked_objective, Some(64));
    assert_eq!(tsab.restart_shortlist_peak_bytes, 4 * std::mem::size_of::<(u64, ScoredScheduleMove)>());
    assert_eq!(first_solution.objectives, vec![64]);
    assert!(first_reports.is_empty(), "a non-improving kick is never published");
    assert_schedule_solution_eq(&second_solution, &first_solution);
    assert_eq!(second_reports, first_reports);
    assert_eq!(second_metrics.schedule_tsab.restart_best_kicked_objective, tsab.restart_best_kicked_objective);
    assert!(second_session.test_same_tabu_trajectory(&first_session));
    assert!(first_session.test_tsab_state_matches_full_oracle());
}

#[test]
fn tsab_tail_corridor_restart_is_deterministic_atomic_and_split_invariant() {
    let (schedule, incumbent) = tail_corridor_restart_fixture();
    let (first_solution, first_session) = forced_tsab_restart_session(&schedule, 47, &incumbent);
    let (second_solution, second_session) = forced_tsab_restart_session(&schedule, 47, &incumbent);
    let (split_solution, split_session) = forced_tsab_restart_session(&schedule, 47, &incumbent);
    let stop = AtomicBool::new(false);

    let mut first_reports = Vec::new();
    let (first_solution, first_metrics, first_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        47,
        47,
        6,
        7,
        &stop,
        8,
        Some(&first_solution),
        Some(first_session),
        &mut |objective| first_reports.push(objective),
    );
    let mut second_reports = Vec::new();
    let (second_solution, second_metrics, second_session) = solve_schedule_capped_persistent_tsab(
        &schedule,
        47,
        47,
        6,
        7,
        &stop,
        8,
        Some(&second_solution),
        Some(second_session),
        &mut |objective| second_reports.push(objective),
    );

    let (split_solution, first_split_metrics, split_session) =
        solve_schedule_capped_persistent_tsab(&schedule, 47, 47, 6, 7, &stop, 3, Some(&split_solution), Some(split_session), &mut |_| {});
    assert_eq!(first_split_metrics.schedule_tsab.restart_work_units, 3);
    assert_eq!(first_split_metrics.schedule_tsab.restart_attempts, 0);
    let mut split_reports = Vec::new();
    let (split_solution, second_split_metrics, split_session) =
        solve_schedule_capped_persistent_tsab(&schedule, 47, 47, 6, 7, &stop, 5, Some(&split_solution), split_session, &mut |objective| {
            split_reports.push(objective)
        });

    let tsab = first_metrics.schedule_tsab;
    assert_eq!(tsab.restart_attempts, 1);
    assert_eq!(tsab.elite_restarts, 1);
    assert_eq!(tsab.restart_n6_generated, 2);
    assert_eq!(tsab.restart_delta_probes, 0, "the compound restart uses one full oracle and no single-move probe");
    assert_eq!(tsab.restart_oracle_commits, 1);
    assert_eq!(tsab.n6_kicks, 1);
    assert_eq!(tsab.kick_moves, 2);
    assert!(tsab.kick_moves > tsab.n6_kicks, "accepted compound restarts expose their component count");
    assert_eq!(tsab.restart_best_base_objective, Some(8));
    assert_eq!(tsab.restart_best_kicked_objective, Some(7));
    assert_eq!(first_solution.objectives, vec![7]);
    assert_eq!(first_reports, vec![7]);
    let replay_model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(schedule.clone()),
    };
    assert_eq!(verify_collection_solution(&replay_model, &first_solution).unwrap(), vec![7]);

    assert_schedule_solution_eq(&second_solution, &first_solution);
    assert_eq!(second_reports, first_reports);
    assert_eq!(second_metrics.schedule_tsab, tsab);
    assert_schedule_solution_eq(&split_solution, &first_solution);
    assert_eq!(split_reports, first_reports);
    assert_eq!(second_split_metrics.schedule_tsab.restart_work_units, 5);
    assert_eq!(second_split_metrics.schedule_tsab.restart_attempts, 1);
    let mut first_session = first_session.expect("first tail-corridor restart retains the session");
    let second_session = second_session.expect("second tail-corridor restart retains the session");
    let split_session = split_session.expect("split tail-corridor restart retains the session");
    assert!(second_session.test_same_tabu_trajectory(&first_session));
    assert!(split_session.test_same_tabu_trajectory(&first_session));
    assert!(first_session.test_tsab_state_matches_full_oracle());
    assert!(audit_persistent_schedule_baseline_reference(&schedule, 47, 64), "the Legacy kernel remains on its reference trajectory");
}

#[test]
fn tsab_multi_restart_commits_one_coordinated_farthest_guide_macro() {
    let stop = AtomicBool::new(false);
    let (schedule, current_order, guide_order) = coordinated_macro_restart_fixture();
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let current_state = JobShopState::from_machine_sequences(&problem, current_order, &stop).unwrap().unwrap();
    let guide_state = JobShopState::from_machine_sequences(&problem, guide_order, &stop).unwrap().unwrap();
    let current_objective = current_state.makespan();
    let guide_objective = guide_state.makespan();
    let incumbent = current_state.to_solution();
    let mut archive = ScheduleEliteArchive::new();
    archive.consider(&guide_state, &stop).unwrap();

    let (base_solution, base_metrics, session) =
        solve_schedule_capped_persistent_tsab_multi(&schedule, 53, 53, 6, 7, &stop, 0, Some(&incumbent), None, &mut |_| {});
    assert_eq!(base_metrics.work_steps, 0);
    assert_eq!(base_solution.objectives, vec![current_objective]);
    let mut session = session.expect("the macro fixture owns a multi-worker TSAB session");
    session.test_force_tsab_restart_due();
    let request = ScheduleRelinkRequest { guide: archive.best().unwrap(), kind: ScheduleRelinkGuideKind::Diverse };
    let mut reports = Vec::new();
    let (solution, metrics, session) = solve_schedule_capped_persistent_tsab_multi_relink(
        &schedule,
        53,
        53,
        6,
        7,
        &stop,
        8,
        Some(&base_solution),
        Some(session),
        request,
        &mut |objective| reports.push(objective),
    );

    let tsab = metrics.schedule_tsab;
    assert_eq!(tsab.restart_attempts, 1);
    assert_eq!(tsab.restart_relink_attempts, 1);
    assert_eq!(tsab.restart_relink_commits, 1);
    assert_eq!(tsab.restart_relink_components_shortlisted, 2);
    assert_eq!(tsab.restart_relink_components_committed, 2);
    assert_eq!(tsab.restart_relink_guide_arc_gain_shortlisted, 3);
    assert_eq!(tsab.restart_relink_guide_arc_gain_accepted, 3);
    assert_eq!(tsab.restart_oracle_commits, tsab.n6_kicks + tsab.restart_relink_commits);
    assert_eq!((tsab.n6_kicks, tsab.kick_moves, tsab.restart_n6_generated, tsab.restart_delta_probes), (0, 0, 0, 0));
    assert_eq!(tsab.restart_rejections, 0);
    assert_eq!(tsab.restart_best_base_objective, Some(current_objective));
    assert_eq!(tsab.restart_best_kicked_objective, Some(guide_objective));
    assert_eq!(metrics.schedule_path_relink.oracle_attempts, 1);
    assert_eq!(metrics.schedule_path_relink.oracle_accepts, 1);
    assert_eq!(metrics.schedule_path_relink.rollbacks, 0);
    assert_eq!(metrics.schedule_path_relink.guide_arc_gain_accepted, 3);
    assert_eq!(solution.objectives, vec![current_objective.min(guide_objective)]);
    assert_eq!(reports, (guide_objective < current_objective).then_some(vec![guide_objective]).unwrap_or_default());
    let mut session = session.expect("the accepted macro restart remains resumable");
    assert_eq!(session.test_tsab_restart_state().3, guide_objective);
    assert!(session.test_tsab_state_matches_full_oracle());
}

#[test]
fn tsab_tail_corridor_uses_the_shortest_non_adjacent_tailward_insert() {
    let schedule = job_shop_schedule(6);
    let stop = AtomicBool::new(false);
    let problem = JobShopProblem::recognize(&schedule, &stop).unwrap().unwrap();
    let mut state = JobShopState::from_machine_sequences(&problem, vec![(0..6).collect()], &stop).unwrap().unwrap();
    state.retain_canonical_critical_blocks_only();

    assert_eq!(audit_tsab_tail_corridor_selection(&state), vec![ScheduleMove::Insert { machine: 0, from: 3, to: 5 }]);
}

#[test]
fn tsab_rejected_tail_corridor_does_not_fall_back_to_a_second_oracle() {
    let schedule = rejected_tail_corridor_restart_schedule();
    let stop = AtomicBool::new(false);
    let (base_solution, initial_metrics, session) =
        solve_schedule_capped_persistent_tsab_multi(&schedule, 18, 18, 5, 7, &stop, 0, None, None, &mut |_| {});
    assert_eq!(initial_metrics.work_steps, 0);
    assert_eq!(base_solution.objectives, vec![113]);
    let mut session = session.expect("the rejected-corridor fixture owns a persistent session");
    session.test_force_tsab_restart_due();
    let mut reports = Vec::new();
    let (solution, metrics, session) = solve_schedule_capped_persistent_tsab_multi(
        &schedule,
        18,
        18,
        5,
        7,
        &stop,
        8,
        Some(&base_solution),
        Some(session),
        &mut |objective| reports.push(objective),
    );

    let tsab = metrics.schedule_tsab;
    assert_eq!(tsab.restart_attempts, 1);
    assert_eq!(tsab.elite_restarts, 1);
    assert_eq!(tsab.restart_n6_generated, 2);
    assert_eq!(tsab.restart_delta_probes, 0, "a rejected batch must not probe a single fallback");
    assert_eq!(tsab.restart_oracle_commits, 0);
    assert_eq!((tsab.n6_kicks, tsab.kick_moves), (0, 0));
    assert_eq!(tsab.restart_rejections, 1);
    assert_eq!(tsab.restart_best_base_objective, Some(113));
    assert_eq!(tsab.restart_best_kicked_objective, None);
    assert_schedule_solution_eq(&solution, &base_solution);
    assert!(reports.is_empty());
    let mut session = session.expect("a rejected tail-corridor restart retains the exact base session");
    assert_eq!(session.test_tsab_restart_state().3, 113);
    assert!(session.test_tsab_state_matches_full_oracle());
}
