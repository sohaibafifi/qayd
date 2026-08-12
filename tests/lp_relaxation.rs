use std::time::Duration;

#[cfg(feature = "lp-relaxation")]
use qayd::model::IntGlobalConstraint;
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
fn clique_vertex_cover(size: usize) -> ModelPackage {
    let mut model = Model::new();
    let vertices = (0..size).map(|_| model.bool_var()).collect::<Vec<_>>();
    for left in 0..size {
        for right in (left + 1)..size {
            model.add_constraint(Constraint::Linear {
                terms: vec![(1, vertices[left]), (1, vertices[right])],
                relation: Relation::Ge,
                rhs: 1,
            });
        }
    }
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vertices.into_iter().map(IntExpr::Variable).collect()) });
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
fn selected_constraints_use_a_selector_guarded_big_m_row() {
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
    assert!(result.aggregate_search_stats().lp_rows > 0);
    assert_eq!(result.aggregate_search_stats().lp_root_bound, Some(0));
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn persistent_node_relaxation_prunes_only_after_exact_recertification() {
    let request = SolveRequest {
        linear: LinearControls {
            backend: LinearBackendMode::Amthal,
            root_time: Duration::from_secs(1),
            node_time: Duration::from_millis(20),
            node_depth_interval: 1,
            phase_max_variables: 0,
            ..LinearControls::default()
        },
        ..SolveRequest::default()
    };
    let result = solve_model(&clique_vertex_cover(6), &request, &mut IgnoreEvents).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [5]);
    let stats = result.aggregate_search_stats();
    assert_eq!(stats.lp_root_bound, Some(3));
    assert!(stats.lp_solves > 1, "the persistent node session never re-optimized");
    assert!(stats.lp_node_prunes > 0, "no exactly certified node bound reached the incumbent");
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn reified_affine_comparison_uses_a_valid_big_m_relaxation() {
    let mut model = Model::new();
    let x = model.int_range(0, 10);
    let selected = model.bool_var();
    model.add_constraint(Constraint::Intension(IntExpr::Iff(
        Box::new(IntExpr::Variable(selected)),
        Box::new(IntExpr::Ge(Box::new(IntExpr::Variable(x)), Box::new(IntExpr::Constant(5)))),
    )));
    model.add_objective(Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Mul(vec![IntExpr::Constant(-100), IntExpr::Variable(selected)])]),
    });

    let result = solve_model(&ModelPackage::new(model), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [-95]);
    assert_eq!(result.aggregate_search_stats().lp_root_bound, Some(-95));
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn all_different_adds_exact_aggregate_bounds_for_a_permutation() {
    let mut model = Model::new();
    let variables = (0..4).map(|_| model.int_range(0, 3)).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: variables.clone(), except: Vec::new() }));
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(variables.into_iter().map(IntExpr::Variable).collect()) });

    let result = solve_model(&ModelPackage::new(model), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [6]);
    assert_eq!(result.aggregate_search_stats().lp_root_bound, Some(6));
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn boolean_count_is_reused_as_a_linear_relaxation() {
    let mut model = Model::new();
    let variables = (0..4).map(|_| model.bool_var()).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Count {
        variables: variables.clone(),
        value: 1,
        relation: Relation::Ge,
        count: 3,
    }));
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(variables.into_iter().map(IntExpr::Variable).collect()) });

    let result = solve_model(&ModelPackage::new(model), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [3]);
    assert_eq!(result.aggregate_search_stats().lp_root_bound, Some(3));
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn boolean_clause_and_reified_or_feed_the_relaxation() {
    let mut model = Model::new();
    let left = model.bool_var();
    let right = model.bool_var();
    let disjunction = model.bool_var();
    model.add_constraint(Constraint::Intension(IntExpr::Or(vec![
        IntExpr::Eq(Box::new(IntExpr::Variable(left)), Box::new(IntExpr::Constant(1))),
        IntExpr::Eq(Box::new(IntExpr::Variable(right)), Box::new(IntExpr::Constant(1))),
    ])));
    model.add_constraint(Constraint::Intension(IntExpr::Iff(
        Box::new(IntExpr::Variable(disjunction)),
        Box::new(IntExpr::Or(vec![IntExpr::Variable(left), IntExpr::Variable(right)])),
    )));
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(disjunction) });

    let result = solve_model(&ModelPackage::new(model), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [1]);
    assert_eq!(result.aggregate_search_stats().lp_root_bound, Some(1));
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn model_build_rejection_exposes_its_exact_reason() {
    let mut model = Model::new();
    let x = model.int_range(0, 10);
    model.add_constraint(Constraint::Intension(IntExpr::Ne(Box::new(IntExpr::Variable(x)), Box::new(IntExpr::Constant(4)))));
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(x) });

    let result = solve_model(&ModelPackage::new(model), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap();
    let stats = result.aggregate_search_stats();
    assert_eq!(stats.lp_model_status, qayd::search::LinearModelStatus::NoRows);
    assert_eq!(stats.lp_variables, 1);
    assert_eq!(stats.lp_source_rows, 0);
}

#[cfg(not(feature = "lp-relaxation"))]
#[test]
fn requesting_an_unlinked_amthal_backend_is_an_explicit_error() {
    let error =
        solve_model(&two_variable_model(Relation::Ge, 5, true), &request(LinearBackendMode::Amthal), &mut IgnoreEvents).unwrap_err();
    assert!(matches!(error, SolveError::InvalidRequest(_)), "{error}");
    assert!(error.to_string().contains("lp-relaxation"), "{error}");
}
