use std::sync::atomic::AtomicBool;

use crate::engines::ls::exact_cover::ExactCoverPlan;
use crate::model::{Constraint, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use crate::orchestrator::{solve_model_silent, EngineKind, SolveRequest};

fn exact_one(model: &mut Model, variables: &[crate::model::IntVarRef]) {
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Count {
        variables: variables.to_vec(),
        value: 1,
        relation: Relation::Eq,
        count: 1,
    }));
}

#[test]
fn exact_cover_constructor_backtracks_to_a_valid_assignment() {
    let mut model = Model::new();
    let a = model.bool_var();
    let b = model.bool_var();
    let c = model.bool_var();
    exact_one(&mut model, &[a, b]);
    exact_one(&mut model, &[a, c]);
    exact_one(&mut model, &[b]);
    model.add_objective(Objective::IntExpr { minimize: true, expr: crate::model::IntExpr::Add(vec![a.into(), b.into(), c.into()]) });

    let stop = AtomicBool::new(false);
    let plan = ExactCoverPlan::compile(&model, &stop, u64::MAX).unwrap();
    let solution = plan.construct(0, &stop).unwrap();
    assert_eq!(solution.values, vec![Some(0), Some(1), Some(1)]);
    assert_eq!(solution.selected, 2);
}

#[test]
fn exact_cover_constructor_leaves_a_materialized_affine_objective_to_cp_replay() {
    let mut model = Model::new();
    let x = model.int_range(0, 1);
    let y = model.int_range(0, 1);
    let objective = model.int_range(0, 2);
    exact_one(&mut model, &[x, y]);
    model.add_constraint(Constraint::Linear { terms: vec![(1, x), (1, y), (-1, objective)], relation: Relation::Eq, rhs: 0 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: objective.into() });

    let stop = AtomicBool::new(false);
    let plan = ExactCoverPlan::compile(&model, &stop, u64::MAX).unwrap();
    let solution = plan.construct(0, &stop).unwrap();
    assert_eq!(solution.values.iter().filter(|value| **value == Some(1)).count(), 1);
    assert_eq!(solution.values[objective.0], None);
}

#[test]
fn verified_exact_cover_warm_start_crosses_the_orchestrator_boundary() {
    let mut model = Model::new();
    let a = model.int_range(0, 1);
    let b = model.int_range(0, 1);
    let c = model.int_range(0, 1);
    let objective = model.int_range(0, 3);
    exact_one(&mut model, &[a, b]);
    exact_one(&mut model, &[a, c]);
    exact_one(&mut model, &[b]);
    model.add_constraint(Constraint::Linear { terms: vec![(1, a), (1, b), (1, c), (-1, objective)], relation: Relation::Eq, rhs: 0 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: objective.into() });

    let result = solve_model_silent(&ModelPackage::new(model), &SolveRequest::default()).unwrap();
    assert_eq!(result.primal().unwrap().objectives(), [2]);
    assert!(result.reports().iter().any(|report| {
        report.engine == Some(EngineKind::IntegerLocalSearch)
            && report.metadata.iter().any(|(key, value)| key == "ls_role" && value == "exact_cover_warm_start")
    }));
    assert!(result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerExact)));
}

#[test]
fn exact_cover_constructor_rejects_mixed_models_and_obeys_cancellation() {
    let mut model = Model::new();
    let x = model.bool_var();
    exact_one(&mut model, &[x]);
    model.add_constraint(Constraint::Clause(vec![]));
    model.add_objective(Objective::IntExpr { minimize: true, expr: x.into() });
    assert!(ExactCoverPlan::compile(&model, &AtomicBool::new(false), u64::MAX).is_none());

    let mut model = Model::new();
    let x = model.bool_var();
    exact_one(&mut model, &[x]);
    model.add_objective(Objective::IntExpr { minimize: true, expr: x.into() });
    assert!(ExactCoverPlan::compile(&model, &AtomicBool::new(false), 1).is_none());
    let plan = ExactCoverPlan::compile(&model, &AtomicBool::new(false), u64::MAX).unwrap();
    assert!(plan.construct(0, &AtomicBool::new(true)).is_none());
}
