use std::cell::Cell;
use std::time::Duration;

use qayd::engines::schedule;
use qayd::model::list::{CollectionModel, IntervalVar, Mode, Resource, Schedule};
use qayd::model::{Model, ModelPackage};
use qayd::orchestrator::{
    audit_memory_stop_next_collection_finalization, result_from_schedule, solve_model_with_budget, verify_semantic_assignment, Assignment,
    CandidateSolution, EngineKind, EventCallback, EventControl, IntervalValue, SolveBudget, SolveEvent, SolveMode, SolveRequest,
    SolveStatus, TerminationReason, VerificationLevel,
};

fn package(native: bool) -> ModelPackage {
    let intervals = (0..8)
        .map(|index| IntervalVar {
            duration: if native { 0 } else { 1 },
            horizon: 12,
            modes: if native { vec![Mode { reference: Some(index), machine: 0, duration: 1, start_window: (0, 11) }] } else { Vec::new() },
            optional: false,
        })
        .collect();
    ModelPackage::new(Model::from_collection(&CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(Schedule {
            intervals,
            precedences: vec![(0, 1), (2, 3)],
            resources: vec![if native { Resource::MachineNoOverlap } else { Resource::NoOverlap((0..8).collect()) }],
            minimize_makespan: true,
        }),
    }))
}

fn assert_verified_schedule_survives_memory_stop(native: bool) {
    let package = package(native);
    let budget = SolveBudget::new(None);
    let request = SolveRequest { mode: SolveMode::Exact, schedule_cdcl: true, ..SolveRequest::default() };
    let at_stop = Cell::new(None);
    let result = solve_model_with_budget(
        &package,
        &request,
        &budget,
        &mut EventCallback(|event| {
            if matches!(event, SolveEvent::Progress { .. }) {
                at_stop.set(Some(()));
                budget.cancel_with(TerminationReason::MemoryLimit);
            }
            Ok(EventControl::Continue)
        }),
    )
    .expect("memory exhaustion after canonical verification is a feasible interrupted result");
    assert!(at_stop.get().is_some(), "the test must reach a complete constructive incumbent");
    assert_eq!(result.status(), SolveStatus::Satisfiable, "an interrupted search cannot claim an optimum");
    assert!(result.proof().is_none());
    let primal = result.primal().expect("keep the complete canonically replayed incumbent");
    assert_eq!(primal.verification(), VerificationLevel::Final);
    assert_eq!(primal.objectives(), &[8]);
    assert!(!result.reports().is_empty(), "keep the observed engine report");
    assert!(result.message().is_some_and(|message| message.contains("MemoryLimit")));
    assert!(budget.expired(), "retention must not clear the original cancellation flag");
    verify_semantic_assignment(&package.model, primal.assignment(), primal.objectives()).unwrap();
}

#[test]
fn verified_native_schedule_survives_memory_stop() {
    assert_verified_schedule_survives_memory_stop(true);
}

#[test]
fn verified_fixed_schedule_survives_memory_stop() {
    assert_verified_schedule_survives_memory_stop(false);
}

#[test]
fn memory_stop_before_any_candidate_stays_unknown() {
    let package = package(true);
    let budget = SolveBudget::new(None);
    budget.cancel_with(TerminationReason::MemoryLimit);
    let request = SolveRequest { mode: SolveMode::Exact, schedule_cdcl: true, ..SolveRequest::default() };
    let result = solve_model_with_budget(&package, &request, &budget, &mut qayd::orchestrator::IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none());
    assert!(result.proof().is_none());
}

#[test]
fn external_or_event_cancellation_after_memory_still_blocks_publication() {
    for reason in [TerminationReason::ExternalCancellation, TerminationReason::EventSink] {
        let package = package(true);
        let budget = SolveBudget::new(None);
        let request = SolveRequest { mode: SolveMode::Exact, schedule_cdcl: true, ..SolveRequest::default() };
        let result = solve_model_with_budget(
            &package,
            &request,
            &budget,
            &mut EventCallback(|event| {
                if matches!(event, SolveEvent::Progress { .. }) {
                    budget.cancel_with(TerminationReason::MemoryLimit);
                    budget.cancel_with(reason);
                }
                Ok(EventControl::Continue)
            }),
        )
        .unwrap();
        assert_eq!(result.status(), SolveStatus::Unknown);
        assert!(result.primal().is_none());
        assert!(result.proof().is_none());
    }
}

fn assert_memory_during_finalization(previous_start: i64, expected_objective: i64) {
    let model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(Schedule {
            intervals: (0..2).map(|_| IntervalVar { duration: 1, horizon: 5, modes: Vec::new(), optional: false }).collect(),
            precedences: Vec::new(),
            resources: vec![Resource::NoOverlap(vec![0, 1])],
            minimize_makespan: true,
        }),
    };
    let semantic = Model::from_collection(&model);
    let candidate = CandidateSolution::replayed(
        &semantic,
        Assignment {
            intervals: [0, previous_start]
                .into_iter()
                .map(|start| IntervalValue { start: Some(start), present: true, machine: None, mode: None })
                .collect(),
            ..Assignment::default()
        },
        vec![expected_objective],
        EngineKind::ScheduleExact,
        VerificationLevel::Transfer,
    )
    .unwrap();
    let budget = SolveBudget::new(None);
    audit_memory_stop_next_collection_finalization();
    let result = result_from_schedule(
        &semantic,
        &model,
        schedule::Outcome {
            status: schedule::Status::Optimal,
            objective: Some(2),
            starts: vec![0, 1],
            presences: vec![true, true],
            machines: vec![-1, -1],
            modes: vec![None, None],
            stats: Default::default(),
        },
        Some(candidate),
        schedule::Execution::default(),
        None,
        Duration::ZERO,
        &budget,
    )
    .unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(result.proof().is_none());
    let primal = result.primal().unwrap();
    assert_eq!(primal.objectives(), &[expected_objective]);
    assert_eq!(primal.assignment().intervals[1].start, Some(previous_start));
    assert!(result.bounds().iter().all(|bound| bound.value <= expected_objective));
    assert!(result.message().is_some_and(|message| message.contains("MemoryLimit")));
    verify_semantic_assignment(&semantic, primal.assignment(), primal.objectives()).unwrap();
}

#[test]
fn interrupted_replay_of_a_newer_best_retains_the_older_verified_candidate() {
    assert_memory_during_finalization(4, 5);
}

#[test]
fn memory_arriving_during_promotion_downgrades_the_proof_and_keeps_the_diagnostic() {
    assert_memory_during_finalization(1, 2);
}
