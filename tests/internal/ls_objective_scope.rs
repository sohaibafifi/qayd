use std::sync::atomic::AtomicBool;

use crate::constraints::linear::Relation;
use crate::engines::ls::cop::LocalSearchSpec;
use crate::expr::Expr;
use crate::problem::{Objective, Problem};
use crate::{Solver, VarId};

fn spec_for(decisions: &[VarId]) -> LocalSearchSpec {
    let mut spec = LocalSearchSpec::default();
    for &variable in decisions {
        spec.add_var(variable);
    }
    spec
}

fn scope(problem: &Problem, spec: &mut LocalSearchSpec) -> Vec<VarId> {
    assert!(spec.normalize_functionals());
    spec.objective_decision_scope(problem, &AtomicBool::new(false), 20_000).expect("expected bounded objective decision scope")
}

#[test]
fn objective_scope_expands_linear_and_expression_closure_in_sorted_order() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 3);
    let y = solver.new_var_range(0, 3);
    let unused = solver.new_var_range(0, 3);
    let sum = solver.new_var_range(0, 6);
    let doubled = solver.new_var_range(0, 12);
    let mut spec = spec_for(&[x, y, unused]);
    // Post consumers before producers, then rely on normalized reverse order.
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(doubled)), Box::new(Expr::Mul(vec![Expr::Const(2), Expr::Var(sum)]))));
    spec.add_linear(vec![1, 1, -1], vec![y, x, sum], Relation::Eq, 0);
    let problem = Problem {
        solver,
        search: vec![x, y, unused, sum, doubled],
        objective: Some(Objective::Expr(true, Expr::Add(vec![Expr::Var(doubled), Expr::Var(x), Expr::Var(doubled)]))),
    };
    assert_eq!(scope(&problem, &mut spec), vec![x, y]);
}

#[test]
fn objective_scope_follows_materialized_index_shifts_and_literal_element_offsets() {
    let mut solver = Solver::new();
    let period = solver.new_var_range(1, 4);
    let unrelated = solver.new_var_range(0, 3);
    let shifted = solver.new_var_range(0, 3);
    let first_cost = solver.new_var_range(0, 2);
    let second_cost = solver.new_var_range(0, 2);
    let mut spec = spec_for(&[period, unrelated, first_cost, second_cost]);
    spec.add_element_const(vec![0, 2, 0, 2], shifted, first_cost, 0);
    spec.add_element_const(vec![2, 0, 2, 0], period, second_cost, 1);
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(shifted)), Box::new(Expr::Sub(Box::new(Expr::Var(period)), Box::new(Expr::Const(1))))));
    let problem = Problem {
        solver,
        search: vec![period, unrelated, shifted, first_cost, second_cost],
        objective: Some(Objective::Linear(true, vec![1, 3], vec![first_cost, second_cost])),
    };
    assert_eq!(scope(&problem, &mut spec), vec![period]);
}

#[test]
fn objective_scope_covers_boolean_tables_and_variable_element_inputs() {
    let mut solver = Solver::new();
    let left = solver.new_var_range(0, 1);
    let right = solver.new_var_range(0, 1);
    let a = solver.new_var_range(0, 9);
    let b = solver.new_var_range(0, 9);
    let index = solver.new_var_range(0, 1);
    let target = solver.new_var_range(0, 9);
    let mut spec = spec_for(&[left, right, a, b]);
    spec.add_element(vec![a, b], index, target, 0);
    spec.add_extension(vec![left, right, index], vec![vec![0, 0, 0], vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]], true);
    let problem = Problem { solver, search: vec![left, right, a, b, index, target], objective: Some(Objective::Var(true, target)) };
    assert_eq!(scope(&problem, &mut spec), vec![left, right, a, b]);
}

#[test]
fn objective_scope_keeps_demoted_cycle_and_alias_targets_as_decisions() {
    for kind in 0..3 {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 1);
        let y = solver.new_var_range(0, 1);
        let mut spec = spec_for(&[x, y]);
        spec.add_element_const(vec![0, 1], x, y, 0);
        match kind {
            0 => spec.add_element_const(vec![0, 1], y, x, 0),
            1 => spec.add_expr(Expr::Eq(Box::new(Expr::Var(x)), Box::new(Expr::Var(y)))),
            _ => spec.add_linear(vec![-1, 1], vec![x, y], Relation::Eq, 0),
        }
        let problem = Problem { solver, search: vec![x, y], objective: Some(Objective::Var(true, x)) };
        assert_eq!(scope(&problem, &mut spec), vec![y]);
        assert!(!spec.is_derived(y));
    }

    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let mut spec = spec_for(&[x]);
    spec.add_element_const(vec![0, 1], x, x, 0);
    let problem = Problem { solver, search: vec![x], objective: Some(Objective::Var(true, x)) };
    assert_eq!(scope(&problem, &mut spec), vec![x]);
}

#[test]
fn objective_scope_uses_retained_duplicate_producer_and_actual_affine_objective_variable() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let guide_only = solver.new_var_range(0, 1);
    let target = solver.new_var_range(0, 1);
    let mut spec = spec_for(&[x, y, guide_only]);
    spec.add_element_const(vec![0, 1], x, target, 0);
    spec.add_expr(Expr::Eq(Box::new(Expr::Var(target)), Box::new(Expr::Add(vec![Expr::Var(y), Expr::Const(0)]))));
    let problem = Problem {
        solver,
        search: vec![x, y, guide_only, target],
        objective: Some(Objective::VarWithAffine(true, target, vec![1], vec![guide_only])),
    };
    assert_eq!(scope(&problem, &mut spec), vec![y]);
}

#[test]
fn objective_scope_ignores_zero_linear_terms_and_fixed_helpers() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let zero_term = solver.new_var_range(0, 1);
    let constant = solver.new_var_range(3, 3);
    let mut spec = spec_for(&[x, zero_term]);
    let problem = Problem {
        solver,
        search: vec![x, zero_term, constant],
        objective: Some(Objective::Linear(true, vec![2, 0, 1], vec![x, zero_term, constant])),
    };
    assert_eq!(scope(&problem, &mut spec), vec![x]);
}

#[test]
fn objective_scope_declines_unnormalized_or_unresolved_physical_dependencies() {
    let mut solver = Solver::new();
    let helper = solver.new_var_range(0, 1);
    let problem = Problem { solver, search: vec![helper], objective: Some(Objective::Var(true, helper)) };
    let mut spec = LocalSearchSpec::default();
    assert!(spec.objective_decision_scope(&problem, &AtomicBool::new(false), 100).is_none());
    assert!(spec.normalize_functionals());
    assert!(spec.objective_decision_scope(&problem, &AtomicBool::new(false), 100).is_none());
}

#[test]
fn objective_scope_honors_stop_and_exact_work_boundary_without_mutating_the_spec() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let mut spec = spec_for(&[x]);
    spec.add_element_const(vec![0, 1], x, y, 0);
    assert!(spec.normalize_functionals());
    let problem = Problem { solver, search: vec![x, y], objective: Some(Objective::Var(true, y)) };
    assert!(spec.objective_decision_scope(&problem, &AtomicBool::new(true), u64::MAX).is_none());
    let minimum = (0..100).find(|&work| spec.objective_decision_scope(&problem, &AtomicBool::new(false), work).is_some()).unwrap();
    assert!(minimum > 0);
    assert!(spec.objective_decision_scope(&problem, &AtomicBool::new(false), minimum - 1).is_none());
    assert_eq!(spec.objective_decision_scope(&problem, &AtomicBool::new(false), minimum).unwrap(), vec![x]);
    assert!(spec.is_derived(y));
    assert!(!spec.is_derived(x));
}

#[test]
fn objective_scope_bounds_expression_work_and_depth_without_fanout_storage() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let mut spec = spec_for(&[x]);
    assert!(spec.normalize_functionals());
    let mut problem = Problem { solver, search: vec![x], objective: Some(Objective::Expr(true, Expr::Add(vec![Expr::Var(x); 4_097]))) };
    assert!(spec.objective_decision_scope(&problem, &AtomicBool::new(false), 100).is_none());
    assert_eq!(spec.objective_decision_scope(&problem, &AtomicBool::new(false), 20_000).unwrap(), vec![x]);
    let mut expression = Expr::Var(x);
    for _ in 0..300 {
        expression = Expr::Neg(Box::new(expression));
    }
    problem.objective = Some(Objective::Expr(true, expression));
    assert!(spec.objective_decision_scope(&problem, &AtomicBool::new(false), 20_000).is_none());
}

#[test]
fn objective_scope_rejects_invalid_references_without_panicking() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let mut spec = spec_for(&[x]);
    assert!(spec.normalize_functionals());
    let problem = Problem { solver, search: vec![x], objective: Some(Objective::Var(true, VarId(99))) };
    assert!(spec.objective_decision_scope(&problem, &AtomicBool::new(false), 100).is_none());
}
