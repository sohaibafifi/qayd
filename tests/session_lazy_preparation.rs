use std::sync::atomic::AtomicBool;
use std::time::Duration;

use qayd::model::{IntExpr, Model, ModelPackage, Objective};
use qayd::orchestrator::{IgnoreEvents, SemanticSolveSession, SolveError, SolveLimits, SolveRequest, SolveStatus};

fn solve(session: &mut SemanticSolveSession, request: &SolveRequest) -> Result<qayd::orchestrator::SolveResult, SolveError> {
    session.solve_with_external_stop(request, &AtomicBool::new(false), &mut IgnoreEvents)
}

#[test]
fn construction_and_zero_time_limit_do_not_validate_the_model() {
    let mut model = Model::new();
    model.int_range(1, 0);
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).expect("session construction is lazy");

    assert_eq!(session.learned_nogoods(), 0);
    assert!(session.raw_nogoods(None).is_empty());
    assert!(session.nogoods(None).unwrap().is_empty());

    let zero_time =
        SolveRequest { limits: SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() }, ..SolveRequest::default() };
    assert_eq!(solve(&mut session, &zero_time).unwrap().status(), SolveStatus::Unknown);

    let error = solve(&mut session, &SolveRequest::default()).unwrap_err();
    assert!(matches!(error, SolveError::Compile(ref message) if message.contains("lower bound exceeds")), "{error}");
}

#[test]
fn interrupted_or_failed_objective_materialization_does_not_commit_partial_state() {
    let mut model = Model::new();
    let value = model.bool_var();
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(value), IntExpr::Constant(0)]) });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Constant(i64::from(i32::MAX) + 1) });
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).expect("objectives are not materialized at construction");
    let zero_time =
        SolveRequest { limits: SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() }, ..SolveRequest::default() };

    assert_eq!(solve(&mut session, &zero_time).unwrap().status(), SolveStatus::Unknown);
    for _ in 0..2 {
        let error = solve(&mut session, &SolveRequest::default()).unwrap_err();
        assert!(
            matches!(error, SolveError::Compile(ref message) if message.contains("objective") && message.contains("outside i32")),
            "{error}"
        );
    }
}

#[test]
fn memory_preflight_applies_before_first_compile_and_to_reused_plans() {
    let mut model = Model::new();
    model.bool_var();
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let below_estimate =
        SolveRequest { limits: SolveLimits { memory_bytes: Some(1), ..SolveLimits::default() }, ..SolveRequest::default() };

    let first_error = solve(&mut session, &below_estimate).unwrap_err();
    assert!(
        matches!(&first_error, SolveError::Compile(message) if message.contains("estimated session CP footprint") && message.contains("memory limit")),
        "{first_error}"
    );
    assert_eq!(solve(&mut session, &SolveRequest::default()).unwrap().status(), SolveStatus::Satisfiable);

    let reused_error = solve(&mut session, &below_estimate).unwrap_err();
    assert!(
        matches!(reused_error, SolveError::Compile(ref message) if message.contains("estimated session CP footprint") && message.contains("memory limit")),
        "{reused_error}"
    );
}
