use std::sync::atomic::AtomicBool;
use std::time::Duration;

use qayd::engines::ls::lists::{solve_schedule, solve_schedule_capped};
use qayd::model::list::{CollectionModel, IntervalVar, Resource, Schedule};
use qayd::model::{Model, ModelPackage};
use qayd::orchestrator::{
    EventCallback, EventControl, IgnoreEvents, SolveEvent, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus,
};

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

fn solve_seeded_portfolio(package: &ModelPackage) -> (SolveResult, Vec<Vec<i64>>) {
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 2,
        seed: 47,
        limits: SolveLimits { iterations: Some(8_192), ..SolveLimits::default() },
        profile: true,
        ..SolveRequest::default()
    };
    let mut progress = Vec::new();
    let result = qayd::solve(
        package,
        &request,
        &mut EventCallback(|event| {
            if let SolveEvent::Progress { objectives, .. } = event {
                progress.push(objectives);
            }
            Ok(EventControl::Continue)
        }),
    )
    .expect("deterministic scheduling portfolio succeeds");
    (result, progress)
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
    assert!(metric(&first, "schedule_incumbent_injections") >= 2);
    assert_eq!(metadata(&first, "schedule_incumbent_injections"), metadata(&first, "schedule_incumbent_injection_attempts"));
    assert_eq!(metadata(&first, "schedule_incumbent_verification_rejections"), "0");
    assert_eq!(metadata(&first, "schedule_elite_pool_size"), "1");
    assert_eq!(metadata(&first, "elite_pool_size"), "1");
    assert_eq!(metadata(&first, "schedule_incumbent_source_worker"), "0");
    assert_eq!(metadata(&first, "schedule_incumbent_source_round"), "0");
    assert_eq!(metadata(&first, "schedule_work_steps"), "8192");
    assert_eq!(metadata(&first, "schedule_worker_work_min"), "4096");
    assert_eq!(metadata(&first, "schedule_worker_work_max"), "4096");
    assert_eq!(metadata(&first, "schedule_unused_work_steps"), "0");
    assert_eq!(metadata(&first, "schedule_stalled_unused_work_steps"), "0");
    assert_eq!(metadata(&first, "schedule_work_budget_overruns"), "0");
    assert_eq!(metadata(&first, "schedule_peak_buffered_candidates"), "2");

    for name in [
        "schedule_incumbent_publications",
        "schedule_incumbent_injection_attempts",
        "schedule_incumbent_injections",
        "schedule_incumbent_rejections",
        "schedule_restart_boundaries",
        "schedule_global_improvements",
        "schedule_work_steps",
    ] {
        assert_eq!(metadata(&first, name), metadata(&second, name), "metric {name} must be reproducible");
    }
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
    assert_eq!(metadata(&first, "schedule_worker_work_min"), "4096");
    assert_eq!(metadata(&first, "schedule_worker_work_max"), "4096");
    assert_eq!(metadata(&first, "schedule_unused_work_steps"), "0");
    assert_eq!(metadata(&first, "schedule_stalled_unused_work_steps"), "0");
    assert_eq!(metadata(&first, "schedule_work_budget_overruns"), "0");
    assert_eq!(metadata(&first, "schedule_peak_buffered_candidates"), "2");

    for name in [
        "schedule_incumbent_publications",
        "schedule_incumbent_injection_attempts",
        "schedule_incumbent_injections",
        "schedule_incumbent_rejections",
        "schedule_incumbent_source_worker",
        "schedule_incumbent_source_round",
        "schedule_restart_boundaries",
        "schedule_global_improvements",
        "schedule_work_steps",
    ] {
        assert_eq!(metadata(&first, name), metadata(&second, name), "metric {name} must be reproducible");
    }
}

#[test]
fn soft_search_stop_preserves_the_last_verified_portfolio_incumbent() {
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
