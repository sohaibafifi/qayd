use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::constraints::linear::{linear, Relation};
use crate::expr::Expr;
use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::lcg::engine::audit_affine_expr_directions;
use crate::problem::Objective as ProblemObjective;
use crate::search::{optimize_seeded, optimize_seeded_with_scope, Objective, SharedObjectiveBound};
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
fn shared_objective_bound_reserves_no_i64_value() {
    let empty = SharedObjectiveBound::new(None);
    assert_eq!(empty.load(), None);
    empty.publish(i64::MAX);
    assert_eq!(empty.load(), Some(i64::MAX));

    let minimum = SharedObjectiveBound::new(Some(i64::MIN));
    assert_eq!(minimum.load(), Some(i64::MIN));
}

#[test]
fn failed_affine_analysis_does_not_leak_partial_value_directions() {
    let x = crate::ids::VarId(0);
    let y = crate::ids::VarId(1);
    let expression = Expr::Add(vec![Expr::Var(x), Expr::Mul(vec![Expr::Var(x), Expr::Var(y)])]);
    let (affine, directions) = audit_affine_expr_directions(&expression, 2);
    assert!(!affine);
    assert_eq!(directions, [0, 0]);
}

#[test]
fn exact_objective_directions_do_not_depend_on_seed_bits() {
    for seed in 0..8 {
        assert_eq!(first_linear_incumbent_seeded(&[3, -2], false, seed), (12, vec![4, 0]));
    }
}

#[test]
fn clause_sharing_portfolio_keeps_an_opposite_objective_seed_class() {
    let first = |seed| {
        let mut solver = Solver::new();
        let variable = solver.new_var_range(0, 4);
        let pool = Arc::new(SharedClausePool::with_capacity(4));
        let mut incumbent = None;
        let _ = optimize_seeded(
            &mut solver,
            &[variable],
            Objective::Var(variable),
            false,
            &AtomicBool::new(false),
            seed,
            None,
            Some(ClauseSharing::new(pool, seed as usize)),
            &[],
            None,
            Vec::new(),
            Vec::new(),
            |_, assignment| {
                incumbent.get_or_insert(assignment[variable.index()]);
            },
        );
        incumbent.expect("optimizer must find an incumbent")
    };

    assert_eq!(first(0), 4);
    assert_eq!(first(1), 0);
    assert_eq!(first(2), 4);
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
fn wide_affine_search_separates_feasibility_and_bounded_dive_policies() {
    let run = |bounded_dive: bool| {
        let mut solver = Solver::new();
        let variables = (0..64).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
        let coefficients = vec![1; variables.len()];
        let objective = if bounded_dive {
            Objective::BoundedDiveLinear { coeffs: &coefficients, vars: &variables }
        } else {
            Objective::Linear { coeffs: &coefficients, vars: &variables }
        };
        let mut incumbents = Vec::new();
        let _ = optimize_seeded(
            &mut solver,
            &variables,
            objective,
            false,
            &AtomicBool::new(false),
            0,
            None,
            None,
            &[],
            None,
            Vec::new(),
            Vec::new(),
            |value, _| {
                incumbents.push(value);
            },
        );
        incumbents
    };

    let complete = run(false);
    let bounded_dive = run(true);
    assert_eq!(&complete[..2], &[0, 1]);
    assert_eq!(complete.last(), Some(&64));
    assert_eq!(bounded_dive.first(), Some(&64));
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
fn seed_zero_hashes_variables_without_an_exact_objective_direction() {
    let mut solver = Solver::new();
    let objective = solver.new_var_range(0, 4);
    let unrelated = (0..8).map(|_| solver.new_var_range(0, 4)).collect::<Vec<_>>();
    let mut search = vec![objective];
    search.extend(&unrelated);
    let mut first = None;
    let _ = optimize_seeded(
        &mut solver,
        &search,
        Objective::Var(objective),
        true,
        &AtomicBool::new(false),
        0,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        |_, assignment| {
            first.get_or_insert(assignment.to_vec());
        },
    );

    let assignment = first.expect("optimizer must find an incumbent");
    assert_eq!(assignment[objective.index()], 0, "the exact minimizing direction must win over the hash");
    for variable in unrelated {
        let expected = if crate::mix64(variable.0 as u64) & 1 == 0 { 0 } else { 4 };
        assert_eq!(assignment[variable.index()], expected, "seed-zero polarity mismatch for {variable:?}");
    }
}

#[test]
fn compact_materialized_affine_view_guides_the_complete_pass() {
    let mut root = Solver::new();
    let x = root.new_var_range(0, 4);
    let y = root.new_var_range(0, 4);
    let objective = root.new_var_range(0, 8);
    linear(&mut root, &[1, 1, -1], &[x, y, objective], Relation::Eq, 0);
    let search = [x, y, objective];
    let physical_objective = ProblemObjective::VarWithAffine(false, objective, vec![1, 1], vec![x, y]);

    let run = |mut solver: Solver, search_objective: Objective<'_>| {
        let mut incumbents = Vec::new();
        let _ = optimize_seeded(
            &mut solver,
            &search,
            search_objective,
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
        incumbents
    };

    let baseline = run(root.clone(), Objective::Var(objective));
    let ordinary = run(root.clone(), physical_objective.search());
    let bounded_dive = run(root, physical_objective.bounded_dive_search());
    assert_ne!(ordinary, baseline);
    assert_eq!(ordinary.first().map(|incumbent| incumbent.0), Some(8));
    assert_eq!(bounded_dive.first().map(|incumbent| incumbent.0), Some(8));
    assert_eq!(ordinary.last().map(|incumbent| incumbent.0), Some(8));
    assert_eq!(bounded_dive.last().map(|incumbent| incumbent.0), Some(8));
}

#[test]
fn materialized_search_views_import_the_shared_cutoff() {
    let mut root = Solver::new();
    let x = root.new_var_range(0, 1);
    let y = root.new_var_range(0, 1);
    let objective = root.new_var_range(0, 2);
    linear(&mut root, &[1, 1, -1], &[x, y, objective], Relation::Eq, 0);
    let search = [x, y, objective];
    let physical_objective = ProblemObjective::VarWithAffine(true, objective, vec![1, 1], vec![x, y]);
    let shared_optimum = SharedObjectiveBound::new(Some(0));
    let run = |mut solver: Solver, search_objective: Objective<'_>| {
        let mut incumbents = Vec::new();
        let _ = optimize_seeded(
            &mut solver,
            &search,
            search_objective,
            true,
            &AtomicBool::new(false),
            0,
            Some(&shared_optimum),
            None,
            &[],
            None,
            Vec::new(),
            Vec::new(),
            |value, _| incumbents.push(value),
        );
        incumbents
    };

    assert!(run(root.clone(), physical_objective.search()).is_empty());
    assert!(run(root, physical_objective.bounded_dive_search()).is_empty());
}

#[test]
fn materialized_affine_constant_never_changes_the_authoritative_cutoff() {
    let mut root = Solver::new();
    let x = root.new_var_range(0, 1);
    let objective = root.new_var_range(5, 6);
    linear(&mut root, &[1, -1], &[x, objective], Relation::Eq, -5);
    let search = [x, objective];
    let physical_objective = ProblemObjective::VarWithAffine(true, objective, vec![1], vec![x]);
    let shared_optimum = SharedObjectiveBound::new(Some(5));

    for search_objective in [physical_objective.search(), physical_objective.bounded_dive_search()] {
        let mut solver = root.clone();
        let mut publications = 0;
        let (best, _, complete) = optimize_seeded(
            &mut solver,
            &search,
            search_objective,
            true,
            &AtomicBool::new(false),
            0,
            Some(&shared_optimum),
            None,
            &[],
            None,
            Vec::new(),
            Vec::new(),
            |_, _| publications += 1,
        );
        assert!(complete);
        assert!(best.is_none());
        assert_eq!(publications, 0);
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
fn monotone_extremum_guides_every_compatible_value_direction() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 4);
    let y = solver.new_var_range(0, 4);
    let search = [x, y];
    let objective = Expr::Max(vec![Expr::Var(x), Expr::Neg(Box::new(Expr::Var(y)))]);
    let mut first = None;
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
        Vec::new(),
        Vec::new(),
        |value, assignment| {
            first.get_or_insert((value, assignment.to_vec()));
        },
    );

    assert_eq!(first, Some((0, vec![0, 4])));
}

#[test]
fn positive_weighted_equalities_reward_their_targets_on_the_first_incumbent() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 15);
    let y = solver.new_var_range(0, 15);
    let search = [x, y];
    let objective = Expr::Add(vec![
        Expr::Mul(vec![Expr::Const(7), Expr::Eq(Box::new(Expr::Var(x)), Box::new(Expr::Const(15)))]),
        Expr::Mul(vec![Expr::Const(3), Expr::Eq(Box::new(Expr::Const(0)), Box::new(Expr::Var(y)))]),
    ]);
    let mut first = None;
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
        |value, assignment| {
            first.get_or_insert((value, assignment.to_vec()));
        },
    );

    assert_eq!(first, Some((10, vec![15, 0])));
}

#[test]
fn normalized_negative_equality_rewards_guide_minimization() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 5);
    let objective = Expr::Add(vec![
        Expr::Const(10),
        Expr::Neg(Box::new(Expr::Mul(vec![Expr::Const(7), Expr::Eq(Box::new(Expr::Var(x)), Box::new(Expr::Const(5)))]))),
        Expr::Add(vec![Expr::Const(2), Expr::Const(-2)]),
    ]);
    let mut first = None;
    let _ = optimize_seeded(
        &mut solver,
        &[x],
        Objective::Expr(&objective),
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
    assert_eq!(first, Some((3, vec![5])));
}

#[test]
fn any_constrained_model_gets_a_feasibility_incumbent_before_exact_targets() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 5);
    let y = solver.new_var_range(0, 1);
    linear(&mut solver, &[1], &[y], Relation::Eq, 0);
    let objective = Expr::Eq(Box::new(Expr::Var(x)), Box::new(Expr::Const(5)));
    let mut incumbents = Vec::new();
    let _ = optimize_seeded(
        &mut solver,
        &[x, y],
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
    assert_eq!(incumbents.first(), Some(&(0, vec![0, 0])));
    assert_eq!(incumbents.last(), Some(&(1, vec![5, 0])));
}

#[test]
fn constrained_equality_rewards_keep_feasibility_variable_order() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    linear(&mut solver, &[1, 1], &[x, y], Relation::Le, 1);
    let objective = Expr::Add(vec![
        Expr::Mul(vec![Expr::Const(7), Expr::Eq(Box::new(Expr::Var(x)), Box::new(Expr::Const(1)))]),
        Expr::Mul(vec![Expr::Const(3), Expr::Eq(Box::new(Expr::Var(y)), Box::new(Expr::Const(1)))]),
    ]);
    let mut incumbents = Vec::new();
    let _ = optimize_seeded(
        &mut solver,
        &[x, y],
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

    assert_eq!(incumbents.first(), Some(&(0, vec![0, 0])));
    assert_eq!(incumbents.last(), Some(&(7, vec![1, 0])));
}

#[test]
fn equality_objective_hint_rejects_partial_or_ambiguous_structures() {
    let first = |objective: Expr, minimizing: bool| {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 5);
        let y = solver.new_var_range(0, 5);
        let search = [x, y];
        let mut incumbent = None;
        let _ = optimize_seeded(
            &mut solver,
            &search,
            Objective::Expr(&objective),
            minimizing,
            &AtomicBool::new(false),
            0,
            None,
            None,
            &[],
            None,
            Vec::new(),
            Vec::new(),
            |value, assignment| {
                incumbent.get_or_insert((value, assignment.to_vec()));
            },
        );
        incumbent.unwrap()
    };
    let x = crate::ids::VarId(0);
    let y = crate::ids::VarId(1);
    let equality = |variable, value| Expr::Eq(Box::new(Expr::Var(variable)), Box::new(Expr::Const(value)));

    let mixed = Expr::Add(vec![Expr::Mul(vec![Expr::Const(7), equality(x, 5)]), Expr::Var(y)]);
    assert_eq!(first(mixed, false), (0, vec![0, 0]));

    let conflicting = Expr::Add(vec![equality(x, 5), equality(x, 4)]);
    assert_eq!(first(conflicting, false), (0, vec![0, 0]));

    let minimized = Expr::Add(vec![Expr::Mul(vec![Expr::Const(7), equality(x, 5)])]);
    assert_eq!(first(minimized, true), (0, vec![0, 0]));

    let unsupported_target = equality(x, 9);
    assert_eq!(first(unsupported_target, false), (0, vec![0, 0]));
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
fn equality_objective_target_does_not_enumerate_a_wide_domain() {
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

    assert_eq!(first, Some(1));
    assert!(!complete);
}
