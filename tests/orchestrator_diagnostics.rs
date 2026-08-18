use std::sync::atomic::AtomicBool;
use std::time::Duration;

use qayd::model::{Model, ModelPackage};
use qayd::orchestrator::{
    count_model_solutions_with_external_stop, extract_model_mus_with_external_stop, IgnoreEvents, ModelMusResult, SemanticSolveSession,
    SolveError, SolveLimits, SolveRequest, SolveStatus,
};

fn binary_model() -> ModelPackage {
    let mut model = Model::new();
    model.bool_var();
    ModelPackage::new(model)
}

#[test]
fn interrupted_count_is_not_reported_as_zero() {
    let error =
        count_model_solutions_with_external_stop(&binary_model(), &[0], &SolveRequest::default(), &AtomicBool::new(true)).unwrap_err();
    assert!(matches!(error, SolveError::Interrupted(_)), "{error}");
}

#[test]
fn zero_budget_count_is_not_reported_as_complete() {
    let request = SolveRequest { limits: SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() }, ..SolveRequest::default() };
    let error = count_model_solutions_with_external_stop(&binary_model(), &[0], &request, &AtomicBool::new(false)).unwrap_err();
    assert!(matches!(error, SolveError::Interrupted(_)), "{error}");
}

#[test]
fn satisfiable_mus_diagnostic_does_not_expose_a_physical_assignment() {
    let result =
        extract_model_mus_with_external_stop(&binary_model(), &[0], &[], &SolveRequest::default(), &AtomicBool::new(false)).unwrap();

    assert_eq!(result, ModelMusResult::Satisfiable);
}

#[test]
fn session_observes_external_stop_before_mapping_large_guidance() {
    let mut session = SemanticSolveSession::new(binary_model()).unwrap();
    let request = SolveRequest { branch_order: vec![usize::MAX; 100_000], ..SolveRequest::default() };
    let result = session.solve_with_external_stop(&request, &AtomicBool::new(true), &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Unknown);
}
