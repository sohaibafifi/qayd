use std::time::Duration;

use qayd::model::{Constraint, IntExpr, Model, ModelPackage, Objective, Relation};
#[cfg(not(feature = "lp-relaxation"))]
use qayd::orchestrator::SolveError;
#[cfg(feature = "lp-relaxation")]
use qayd::orchestrator::SolveStatus;
use qayd::orchestrator::{solve_model, IgnoreEvents, LinearBackendMode, LinearControls, SolveRequest};

fn request(backend: LinearBackendMode) -> SolveRequest {
    SolveRequest {
        linear: LinearControls { backend, root_time: Duration::from_secs(1), ..LinearControls::default() },
        ..SolveRequest::default()
    }
}

fn two_variable_model(relation: Relation, rhs: i64, minimize: bool) -> ModelPackage {
    let mut model = Model::new();
    let x = model.int_range(0, 10);
    let y = model.int_range(0, 10);
    model.add_constraint(Constraint::Linear { terms: vec![(2, x), (2, y)], relation, rhs });
    model.add_objective(Objective::IntExpr { minimize, expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Variable(y)]) });
    ModelPackage::new(model)
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn amthal_minimization_bound_is_recertified_and_rounded_for_integer_search() {
    let result = solve_model(&two_variable_model(Relation::Ge, 5, true), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [3]);
    let stats = result.aggregate_search_stats();
    assert!(stats.lp_rows >= 1);
    assert_eq!(stats.lp_solves, 1);
    assert_eq!(stats.lp_certified, 1);
    assert_eq!(stats.lp_root_bound, Some(3));
    assert!(result.reports().iter().flat_map(|report| &report.metadata).any(|(key, value)| key == "linear_backends" && value == "amthal"));
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn amthal_maximization_certificate_has_the_public_upper_bound_direction() {
    let result = solve_model(&two_variable_model(Relation::Le, 5, false), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [2]);
    assert_eq!(result.aggregate_search_stats().lp_root_bound, Some(2));
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn native_mode_does_not_construct_the_optional_relaxation() {
    let result = solve_model(&two_variable_model(Relation::Ge, 5, true), &request(LinearBackendMode::Native), &mut IgnoreEvents).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.aggregate_search_stats().lp_solves, 0);
    assert_eq!(result.aggregate_search_stats().lp_root_bound, None);
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn affine_intension_comparisons_feed_the_same_relaxation_ir() {
    let mut model = Model::new();
    let x = model.int_range(0, 10);
    let y = model.int_range(0, 10);
    model.add_constraint(Constraint::Intension(IntExpr::Ge(
        Box::new(IntExpr::Add(vec![
            IntExpr::Mul(vec![IntExpr::Constant(2), IntExpr::Variable(x)]),
            IntExpr::Mul(vec![IntExpr::Constant(2), IntExpr::Variable(y)]),
        ])),
        Box::new(IntExpr::Constant(5)),
    )));
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Variable(y)]) });

    let result = solve_model(&ModelPackage::new(model), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.aggregate_search_stats().lp_root_bound, Some(3));
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn selected_constraints_are_not_misrepresented_as_unconditional_rows() {
    let mut model = Model::new();
    let selector = model.bool_var();
    let value = model.int_range(0, 10);
    model.add_constraint(Constraint::Selected {
        selector,
        constraint: Box::new(Constraint::Linear { terms: vec![(1, value)], relation: Relation::Ge, rhs: 5 }),
    });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });

    let result = solve_model(&ModelPackage::new(model), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.aggregate_search_stats().lp_rows, 0);
}

#[cfg(not(feature = "lp-relaxation"))]
#[test]
fn requesting_an_unlinked_amthal_backend_is_an_explicit_error() {
    let error =
        solve_model(&two_variable_model(Relation::Ge, 5, true), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap_err();
    assert!(matches!(error, SolveError::InvalidRequest(_)), "{error}");
    assert!(error.to_string().contains("lp-relaxation"), "{error}");
}
