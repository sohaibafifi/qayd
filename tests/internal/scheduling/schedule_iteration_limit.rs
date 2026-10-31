use std::time::Duration;

use qayd::model::list::{CollectionModel, IntervalVar, Resource, Schedule};
use qayd::model::{Model, ModelPackage};
use qayd::orchestrator::{IgnoreEvents, SolveLimits, SolveMode, SolveRequest, SolveStatus};

fn long_single_machine_job_shop(operation_count: usize) -> ModelPackage {
    let schedule = Schedule {
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

fn flat_resource_project(activity_count: usize) -> ModelPackage {
    let schedule = Schedule {
        intervals: (0..activity_count)
            .map(|_| IntervalVar {
                duration: 1,
                horizon: i64::try_from(activity_count).expect("test size fits i64"),
                modes: Vec::new(),
                optional: false,
            })
            .collect(),
        precedences: Vec::new(),
        resources: vec![Resource::Cumulative { demands: (0..activity_count).map(|activity| (activity, 1)).collect(), capacity: 1 }],
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

fn metadata<'a>(result: &'a qayd::orchestrator::SolveResult, name: &str) -> &'a str {
    result
        .reports()
        .first()
        .expect("schedule engine report")
        .metadata
        .iter()
        .find_map(|(key, value)| (key == name).then_some(value.as_str()))
        .unwrap_or_else(|| panic!("missing schedule metadata {name}"))
}

#[test]
fn schedule_portfolio_consumes_one_shared_iteration_quota() {
    let package = long_single_machine_job_shop(56);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 4,
        limits: SolveLimits { time: Some(Duration::from_secs(2)), iterations: Some(7), ..SolveLimits::default() },
        profile: true,
        ..SolveRequest::default()
    };

    let result = qayd::solve(&package, &request, &mut IgnoreEvents).expect("bounded schedule local search succeeds");
    let repeated = qayd::solve(&package, &request, &mut IgnoreEvents).expect("bounded portfolio repeats");

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.aggregate_search_stats().nodes, 7);
    assert!(result.message().is_some_and(|message| message.contains("iteration limit")));
    let report = result.reports().first().expect("schedule engine report");
    assert!(report.metadata.iter().any(|(name, value)| name == "schedule_moves_considered" && value == "7"));
    assert_eq!(metadata(&result, "schedule_global_improvements"), metadata(&repeated, "schedule_global_improvements"));
    assert_eq!(report.improvements, repeated.reports().first().expect("repeated schedule report").improvements);
}

#[test]
fn resource_search_consumes_the_exact_work_budget_and_reaches_alns() {
    let package = flat_resource_project(12);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: SolveLimits { time: Some(Duration::from_secs(2)), iterations: Some(512), ..SolveLimits::default() },
        profile: true,
        seed: 19,
        ..SolveRequest::default()
    };

    let first = qayd::solve(&package, &request, &mut IgnoreEvents).expect("bounded resource search succeeds");
    let second = qayd::solve(&package, &request, &mut IgnoreEvents).expect("seeded resource search repeats");

    for result in [&first, &second] {
        assert_eq!(result.status(), SolveStatus::Satisfiable);
        assert_eq!(result.primal().expect("complete verified schedule").objectives(), [12]);
        assert_eq!(result.aggregate_search_stats().nodes, 512);
        assert_eq!(metadata(result, "constructor"), "resource-priority-sgs");
        assert_eq!(metadata(result, "schedule_work_steps"), "512");
        assert!(metadata(result, "schedule_delta_evaluations").parse::<u64>().unwrap() > 0);
        assert!(metadata(result, "schedule_alns_generation_attempts").parse::<u64>().unwrap() > 0);
        assert!(metadata(result, "schedule_alns_moves_generated").parse::<u64>().unwrap() > 0);
        assert_eq!(metadata(result, "schedule_oracle_mismatches"), "0");
        assert_eq!(metadata(result, "schedule_workspace_growths"), "0");
        assert!(result.message().is_some_and(|message| message.contains("iteration limit")));
    }
    for name in [
        "schedule_moves_considered",
        "schedule_moves_accepted",
        "schedule_delta_evaluations",
        "schedule_alns_generation_attempts",
        "schedule_alns_moves_generated",
        "schedule_oracle_validations",
    ] {
        assert_eq!(metadata(&first, name), metadata(&second, name), "metric {name} must be reproducible");
    }
}
