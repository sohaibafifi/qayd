use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use qayd::constraints::table::STAR;
use qayd::model::list::{register_external_function, ExprArena, Iterable, ReduceOp, Reduction};
use qayd::model::{Constraint, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{
    audit_interrupt_next_collection_final_replay, verify_semantic_assignment, verify_semantic_assignment_validated_interruptible,
    Assignment, EventCallback, EventControl, SolveError, SolveEvent, SolveLimits, SolveMode, SolveRequest, SolveStatus,
};

fn list_assignment_model(items: usize) -> ModelPackage {
    let universe = (1..=items).map(|item| item as i32).collect::<Vec<_>>();
    let mut model = Model::new();
    let lists = (0..4).map(|_| model.list(universe.clone())).collect::<Vec<_>>();
    model.add_constraint(Constraint::ListPartition { lists: lists.clone(), items: universe });
    let mut arena = ExprArena::default();
    let one = arena.constant(1);
    model.add_objective(Objective::ListTerms {
        minimize: true,
        terms: vec![Reduction { op: ReduceOp::Count, iterable: Iterable::Items(lists[0].0), arena, body: one, coeff: 1 }],
        max_terms: None,
    });
    ModelPackage::new(model)
}

#[test]
fn local_search_does_not_replay_the_global_model_in_its_hot_loop() {
    let package = list_assignment_model(32);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        seed: 7,
        limits: SolveLimits { iterations: Some(200), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let before = qayd::model::list::audit_verification_calls();
    let result = qayd::solve(&package, &request, &mut qayd::orchestrator::IgnoreEvents).unwrap();
    let replay_count = qayd::model::list::audit_verification_calls() - before;

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(result.primal().is_some());
    assert!(replay_count <= 2, "global collection replay ran {replay_count} times for 200 search iterations");
}

#[test]
fn expiration_before_compilation_is_a_normal_unknown_result() {
    let package = list_assignment_model(10_000);
    let request = SolveRequest {
        mode: SolveMode::Exact,
        limits: SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let result = qayd::solve(&package, &request, &mut qayd::orchestrator::IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none());
}

#[test]
fn cancellation_before_final_replay_cannot_publish_an_unverified_candidate() {
    let package = list_assignment_model(120);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        seed: 19,
        limits: SolveLimits { iterations: Some(100_000), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let mut stopped = false;
    let mut sink = EventCallback(|event| {
        if matches!(event, SolveEvent::Progress { .. }) {
            stopped = true;
            Ok(EventControl::Stop)
        } else {
            Ok(EventControl::Continue)
        }
    });
    let result = qayd::solve(&package, &request, &mut sink).unwrap();

    assert!(stopped);
    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none());
}

#[test]
fn a_soft_deadline_during_final_replay_keeps_the_verified_incumbent() {
    let package = list_assignment_model(16);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        seed: 3,
        limits: SolveLimits { iterations: Some(0), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    audit_interrupt_next_collection_final_replay(false);

    let result = qayd::solve(&package, &request, &mut qayd::orchestrator::IgnoreEvents).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(result.primal().is_some());
}

#[test]
fn a_hard_cancel_after_soft_deadline_prevents_final_publication() {
    let package = list_assignment_model(16);
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        seed: 3,
        limits: SolveLimits { iterations: Some(0), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    audit_interrupt_next_collection_final_replay(true);

    let result = qayd::solve(&package, &request, &mut qayd::orchestrator::IgnoreEvents).unwrap();

    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none());
    assert!(result.message().is_some_and(|message| message.contains("ExternalCancellation")));
}

#[test]
fn public_verification_boundary_still_validates_the_model() {
    let mut model = Model::new();
    model.int_set(vec![0, 0]);
    let assignment = Assignment { integers: vec![Some(0)], ..Assignment::default() };

    let error = verify_semantic_assignment(&model, &assignment, &[]).unwrap_err();

    assert!(matches!(error, SolveError::Compile(message) if message.contains("duplicate")));
}

#[test]
fn validated_verification_still_replays_constraints_and_honors_interruption() {
    let mut model = Model::new();
    let value = model.int_range(0, 1);
    model.add_constraint(Constraint::Linear { terms: vec![(1, value)], relation: Relation::Eq, rhs: 1 });
    model.validate().unwrap();
    let assignment = Assignment { integers: vec![Some(0)], ..Assignment::default() };

    let error = verify_semantic_assignment_validated_interruptible(&model, &assignment, &[], &AtomicBool::new(false)).unwrap_err();
    assert!(matches!(error, SolveError::InvalidResult(message) if message.contains("constraint 0")));

    let interrupted = verify_semantic_assignment_validated_interruptible(&model, &assignment, &[], &AtomicBool::new(true)).unwrap_err();
    assert!(matches!(interrupted, SolveError::Interrupted(_)));
}

fn table_model(tuples: Vec<Vec<i32>>, positive: bool) -> Model {
    let mut model = Model::new();
    let variables = (0..3).map(|_| model.int_range(i32::MIN, i32::MAX)).collect();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, tuples, positive }));
    model
}

fn integer_assignment(values: [i32; 3]) -> Assignment {
    Assignment { integers: values.into_iter().map(|value| Some(i64::from(value))).collect(), ..Assignment::default() }
}

#[test]
fn canonical_replay_matches_starred_supports_in_every_column() {
    let model = table_model(vec![vec![STAR, 1, 2], vec![3, STAR, 4], vec![5, 6, STAR]], true);

    for values in [[99, 1, 2], [3, -7, 4], [5, 6, 42]] {
        verify_semantic_assignment(&model, &integer_assignment(values), &[])
            .unwrap_or_else(|error| panic!("starred support rejected {values:?}: {error}"));
    }

    let error = verify_semantic_assignment(&model, &integer_assignment([99, 0, 2]), &[]).unwrap_err();
    assert!(matches!(error, SolveError::InvalidResult(message) if message.contains("constraint 0")));
}

#[test]
fn canonical_replay_matches_starred_conflicts_in_every_column() {
    let model = table_model(vec![vec![STAR, 1, 2], vec![3, STAR, 4], vec![5, 6, STAR]], false);

    for values in [[99, 1, 2], [3, -7, 4], [5, 6, 42]] {
        let error = verify_semantic_assignment(&model, &integer_assignment(values), &[]).unwrap_err();
        assert!(
            matches!(error, SolveError::InvalidResult(ref message) if message.contains("constraint 0")),
            "starred conflict accepted {values:?}: {error}"
        );
    }

    verify_semantic_assignment(&model, &integer_assignment([99, 0, 2]), &[]).expect("assignment outside every conflict must be accepted");
}

#[test]
fn wildcard_semantics_include_the_minimum_integer_domain_value() {
    let support = table_model(vec![vec![7, STAR, 9]], true);
    let conflict = table_model(vec![vec![7, STAR, 9]], false);
    let assignment = integer_assignment([7, i32::MIN, 9]);

    verify_semantic_assignment(&support, &assignment, &[]).expect("wildcard must support an actual i32::MIN assignment");
    let error = verify_semantic_assignment(&conflict, &assignment, &[]).unwrap_err();
    assert!(matches!(error, SolveError::InvalidResult(message) if message.contains("constraint 0")));
}

#[test]
fn non_cooperative_external_expression_cannot_hold_canonical_replay() {
    const NAME: &str = "phase10_5_blocking_canonical_replay";
    register_external_function(NAME, |_| {
        std::thread::sleep(Duration::from_millis(600));
        Ok(1)
    })
    .unwrap();

    let mut model = Model::new();
    let list = model.list(vec![1]);
    model.add_constraint(Constraint::ListPartition { lists: vec![list], items: vec![1] });
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let body = arena.external(NAME, vec![item]);
    model.add_objective(Objective::ListTerms {
        minimize: true,
        terms: vec![Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list.0), arena, body, coeff: 1 }],
        max_terms: None,
    });
    let assignment = Assignment { lists: vec![vec![1]], ..Assignment::default() };
    let stop = Arc::new(AtomicBool::new(false));
    let trigger = Arc::clone(&stop);
    let stopper = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        trigger.store(true, Ordering::Release);
    });
    let started = Instant::now();

    let result = verify_semantic_assignment_validated_interruptible(&model, &assignment, &[1], &stop);

    stopper.join().unwrap();
    assert!(matches!(result, Err(SolveError::Interrupted(_))));
    assert!(started.elapsed() < Duration::from_millis(300), "external callback escaped the replay cancellation boundary");
}
