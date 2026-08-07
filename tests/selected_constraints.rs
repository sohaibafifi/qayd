use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{solve_model_silent, SolveLimits, SolveMode, SolveRequest, SolveStatus};

fn selected_linear(selector_value: i64) -> ModelPackage {
    let mut model = Model::new();
    let selector = model.bool_var();
    let value = model.bool_var();
    model.add_constraint(Constraint::Linear { terms: vec![(1, selector)], relation: Relation::Eq, rhs: selector_value });
    model.add_constraint(Constraint::Selected {
        selector,
        constraint: Box::new(Constraint::Linear { terms: vec![(1, value)], relation: Relation::Eq, rhs: 0 }),
    });
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(value) });
    ModelPackage::new(model)
}

#[test]
fn disabled_selector_does_not_apply_linear_constraint() {
    let disabled = solve_model_silent(&selected_linear(0), &SolveRequest::default()).unwrap();
    let enabled = solve_model_silent(&selected_linear(1), &SolveRequest::default()).unwrap();

    assert_eq!(disabled.status(), SolveStatus::Optimal);
    assert_eq!(disabled.primal().unwrap().objectives(), [1]);
    assert_eq!(enabled.status(), SolveStatus::Optimal);
    assert_eq!(enabled.primal().unwrap().objectives(), [0]);
}

fn selected_element(selector_value: i64) -> ModelPackage {
    let mut model = Model::new();
    let selector = model.bool_var();
    let array_value = model.int_range(0, 0);
    let index = model.int_range(0, 0);
    let target = model.int_range(1, 1);
    model.add_constraint(Constraint::Linear { terms: vec![(1, selector)], relation: Relation::Eq, rhs: selector_value });
    model.add_constraint(Constraint::Selected {
        selector,
        constraint: Box::new(Constraint::IntegerGlobal(IntGlobalConstraint::Element { array: vec![array_value], index, value: target })),
    });
    ModelPackage::new(model)
}

#[test]
fn disabled_selector_does_not_apply_functional_global() {
    let disabled = solve_model_silent(&selected_element(0), &SolveRequest::default()).unwrap();
    let enabled = solve_model_silent(&selected_element(1), &SolveRequest::default()).unwrap();

    assert_eq!(disabled.status(), SolveStatus::Satisfiable);
    assert_eq!(enabled.status(), SolveStatus::Unsatisfiable);
}

#[test]
fn selected_functional_is_explicitly_unsupported_by_local_search() {
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: SolveLimits { iterations: Some(100), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let result = solve_model_silent(&selected_element(0), &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Unsupported);
    assert!(result.message().is_some_and(|message| message.contains("compilation rejected")));
    assert!(result.reports().is_empty());
}

#[test]
fn multi_worker_local_search_handles_empty_objective_vectors() {
    let mut model = Model::new();
    let value = model.bool_var();
    model.add_constraint(Constraint::Linear { terms: vec![(1, value)], relation: Relation::Eq, rhs: 1 });
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 2,
        limits: SolveLimits { iterations: Some(100), ..SolveLimits::default() },
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(result.primal().unwrap().objectives().is_empty());
}
