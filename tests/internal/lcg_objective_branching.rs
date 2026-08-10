use std::sync::atomic::AtomicBool;

use crate::constraints::linear::{linear, Relation};
use crate::expr::Expr;
use crate::search::{optimize_seeded, optimize_seeded_with_scope, Objective};
use crate::Solver;

fn first_linear_incumbent(coeffs: &[i64], minimizing: bool) -> (i64, Vec<i32>) {
    first_linear_incumbent_seeded(coeffs, minimizing, 0)
}

fn first_linear_incumbent_seeded(coeffs: &[i64], minimizing: bool, seed: u64) -> (i64, Vec<i32>) {
    let mut solver = Solver::new();
    let vars = (0..coeffs.len()).map(|_| solver.new_var_range(0, 4)).collect::<Vec<_>>();
    let mut incumbents = Vec::new();
    let _ = optimize_seeded(
        &mut solver,
        &vars,
        Objective::Linear { coeffs, vars: &vars },
        minimizing,
        &AtomicBool::new(false),
        seed,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        |value, assignment| incumbents.push((value, assignment.to_vec())),
    );
    incumbents.into_iter().next().expect("optimizer must find an incumbent")
}

#[test]
fn objective_guidance_preserves_seeded_portfolio_diversity() {
    let incumbents = (0..4).map(|seed| first_linear_incumbent_seeded(&[3, -2], false, seed)).collect::<std::collections::BTreeSet<_>>();
    assert!(incumbents.len() >= 2, "all objective workers followed the same first dive: {incumbents:?}");
    assert!(incumbents.contains(&(12, vec![4, 0])));
}

#[test]
fn explicit_scope_observably_branches_before_completion_variables() {
    let first = |primary_scope: Option<&[crate::ids::VarId]>| {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 1);
        let y = solver.new_var_range(0, 1);
        linear(&mut solver, &[1, 1], &[x, y], Relation::Eq, 1);
        let search = [x, y];
        let objective_vars = [y];
        let mut incumbent = None;
        let _ = optimize_seeded_with_scope(
            &mut solver,
            &search,
            primary_scope,
            Objective::Linear { coeffs: &[1], vars: &objective_vars },
            true,
            &AtomicBool::new(false),
            0,
            None,
            None,
            &[],
            None,
            Vec::new(),
            Vec::new(),
            |value, _| {
                incumbent.get_or_insert(value);
            },
        );
        incumbent.unwrap()
    };

    assert_eq!(first(None), 0);
    let primary = [crate::ids::VarId(0)];
    assert_eq!(first(Some(&primary)), 1);
}

#[test]
fn first_incumbent_uses_the_improving_endpoint_for_linear_terms() {
    assert_eq!(first_linear_incumbent(&[3, -2], false), (12, vec![4, 0]));
    assert_eq!(first_linear_incumbent(&[3, -2], true), (-8, vec![0, 4]));
}

#[test]
fn variable_objective_uses_its_improving_endpoint() {
    for (minimizing, expected) in [(true, 0), (false, 4)] {
        let mut solver = Solver::new();
        let unrelated = solver.new_var_range(0, 4);
        let objective = solver.new_var_range(0, 4);
        let vars = [unrelated, objective];
        let mut first = None;
        let _ = optimize_seeded(
            &mut solver,
            &vars,
            Objective::Var(objective),
            minimizing,
            &AtomicBool::new(false),
            0,
            None,
            None,
            &[],
            None,
            Vec::new(),
            Vec::new(),
            |value, _| {
                first.get_or_insert(value);
            },
        );
        assert_eq!(first, Some(expected));
    }
}

#[test]
fn affine_expression_objective_guides_values_without_changing_its_representation() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 4);
    let y = solver.new_var_range(0, 4);
    let vars = [x, y];
    let expression =
        Expr::Add(vec![Expr::Mul(vec![Expr::Const(3), Expr::Var(x)]), Expr::Neg(Box::new(Expr::Mul(vec![Expr::Const(2), Expr::Var(y)])))]);
    let mut first = None;
    let _ = optimize_seeded(
        &mut solver,
        &vars,
        Objective::Expr(&expression),
        false,
        &AtomicBool::new(false),
        0,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        |value, assignment| {
            first.get_or_insert((value, assignment.to_vec()));
        },
    );
    assert_eq!(first, Some((12, vec![4, 0])));
}

#[test]
fn repeated_extreme_linear_terms_keep_their_mathematical_direction() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 4);
    let search = [x];
    let objective_vars = [x, x, x];
    let coeffs = [i64::MAX, i64::MAX, i64::MIN];
    let mut first = None;
    let _ = optimize_seeded(
        &mut solver,
        &search,
        Objective::Linear { coeffs: &coeffs, vars: &objective_vars },
        true,
        &AtomicBool::new(false),
        0,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        |value, assignment| {
            first.get_or_insert((value, assignment.to_vec()));
        },
    );
    assert_eq!(first, Some((0, vec![0])));
}

#[test]
fn nonlinear_objective_guidance_keeps_a_fast_feasibility_dive() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    linear(&mut solver, &[1, 1], &[x, y], Relation::Eq, 1);
    let search = [x, y];
    let objective = Expr::And(vec![
        Expr::Eq(Box::new(Expr::Var(x)), Box::new(Expr::Const(1))),
        Expr::Eq(Box::new(Expr::Var(y)), Box::new(Expr::Const(0))),
    ]);
    let mut incumbents = Vec::new();
    let _ = optimize_seeded(
        &mut solver,
        &search,
        Objective::Expr(&objective),
        false,
        &AtomicBool::new(false),
        0,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        |value, assignment| incumbents.push((value, assignment.to_vec())),
    );

    assert_eq!(incumbents.first(), Some(&(0, vec![0, 1])));
    assert_eq!(incumbents.last(), Some(&(1, vec![1, 0])));
}

#[test]
fn all_none_caller_phase_still_allows_the_guarded_objective_dive() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    let search = [x, y];
    let objective = Expr::Add(vec![Expr::And(vec![Expr::Not(Box::new(Expr::Var(x)))]), Expr::And(vec![Expr::Not(Box::new(Expr::Var(y)))])]);
    let mut incumbents = Vec::new();
    let _ = optimize_seeded(
        &mut solver,
        &search,
        Objective::Expr(&objective),
        true,
        &AtomicBool::new(false),
        0,
        None,
        None,
        &[],
        None,
        vec![None; 2],
        Vec::new(),
        |value, assignment| incumbents.push((value, assignment.to_vec())),
    );

    assert_eq!(incumbents, vec![(2, vec![0, 0]), (0, vec![1, 1])]);
}

#[test]
fn symbolic_objective_bound_does_not_enumerate_a_wide_domain() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, i32::MAX);
    let objective = Expr::Eq(Box::new(Expr::Var(x)), Box::new(Expr::Const(i64::from(i32::MAX))));
    let stop = AtomicBool::new(false);
    let mut first = None;
    let (_, _, complete) = optimize_seeded(
        &mut solver,
        &[x],
        Objective::Expr(&objective),
        false,
        &stop,
        0,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        |value, _| {
            first = Some(value);
            stop.store(true, std::sync::atomic::Ordering::Release);
        },
    );

    assert_eq!(first, Some(0));
    assert!(!complete);
}
