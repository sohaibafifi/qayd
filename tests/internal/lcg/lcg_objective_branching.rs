use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::constraints::linear::{linear, Relation};
use crate::expr::Expr;
use crate::lcg::clause::{ClauseSharing, SharedClausePool};
use crate::lcg::engine::audit_affine_expr_directions;
use crate::problem::Objective as ProblemObjective;
use crate::search::{
    optimize_seeded, optimize_seeded_with_scope, optimize_seeded_with_scope_and_relaxation, Objective, SharedObjectiveBound,
};
use crate::search_policy::{CompiledSearchPhase, ValueSelector, VariableSelector};
use crate::Solver;

fn first_linear_incumbent(coeffs: &[i64], minimizing: bool) -> (i64, Vec<i32>) {
    first_linear_incumbent_seeded(coeffs, minimizing, 0)
}

fn first_linear_incumbent_seeded(coeffs: &[i64], minimizing: bool, seed: u64) -> (i64, Vec<i32>) {
    first_linear_incumbent_with_domain(coeffs, minimizing, seed, 4)
}

fn first_linear_incumbent_with_domain(coeffs: &[i64], minimizing: bool, seed: u64, domain_max: i32) -> (i64, Vec<i32>) {
    let mut solver = Solver::new();
    let vars = (0..coeffs.len()).map(|_| solver.new_var_range(0, domain_max)).collect::<Vec<_>>();
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
fn wide_boolean_affine_search_uses_exact_directions_on_the_first_dive() {
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
    assert_eq!(complete, [64]);
    assert_eq!(bounded_dive, [64]);
}

#[test]
fn constrained_wide_boolean_objective_uses_exact_directions_immediately() {
    let mut solver = Solver::new();
    let variables = (0..9).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let coefficients = vec![1; variables.len()];
    linear(&mut solver, &coefficients, &variables, Relation::Le, 1);
    let mut incumbents = Vec::new();

    let _ = optimize_seeded(
        &mut solver,
        &variables,
        Objective::Linear { coeffs: &coefficients, vars: &variables },
        false,
        &AtomicBool::new(false),
        0,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        |value, _| incumbents.push(value),
    );

    assert_eq!(incumbents, [1]);
}

#[test]
fn wide_boolean_affine_mixed_coefficients_use_improving_values() {
    let coefficients = [3, -2, 5, -7, 1, 4, -3, 2, -1];
    for minimizing in [false, true] {
        let expected = coefficients.iter().map(|&coefficient| i32::from((coefficient < 0) == minimizing)).collect::<Vec<_>>();
        let optimum = coefficients.iter().zip(&expected).map(|(coefficient, &value)| coefficient * i64::from(value)).sum();
        for seed in 0..4 {
            assert_eq!(first_linear_incumbent_with_domain(&coefficients, minimizing, seed, 1), (optimum, expected.clone()));
        }
    }
}

#[test]
fn wide_affine_objective_with_an_active_non_boolean_term_keeps_the_feasibility_policy() {
    for non_boolean_terms in [1, 9] {
        let mut solver = Solver::new();
        let variables = (0..9).map(|index| solver.new_var_range(0, if index < non_boolean_terms { 4 } else { 1 })).collect::<Vec<_>>();
        let coefficients = vec![1; variables.len()];
        let mut incumbents = Vec::new();
        let (best, _, complete) = optimize_seeded(
            &mut solver,
            &variables,
            Objective::Linear { coeffs: &coefficients, vars: &variables },
            false,
            &AtomicBool::new(false),
            0,
            None,
            None,
            &[],
            None,
            Vec::new(),
            Vec::new(),
            |value, _| incumbents.push(value),
        );
        assert!(complete);
        assert_eq!(incumbents.first(), Some(&0));
        assert_eq!(best.map(|(_, value)| value), Some(9 + 3 * non_boolean_terms));
    }
}

#[test]
fn wide_boolean_affine_incumbent_phases_do_not_limit_improvement_to_one() {
    let mut solver = Solver::new();
    let selector = solver.new_var_range(0, 2);
    let terms = (0..9).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    for &term in &terms {
        linear(&mut solver, &[1, -1], &[term, selector], Relation::Le, 0);
    }
    let coefficients = vec![1; terms.len()];
    let mut search = vec![selector];
    search.extend(&terms);
    let phases = vec![CompiledSearchPhase {
        scope: vec![selector],
        semantic_salts: vec![0],
        variable: VariableSelector::InputOrder,
        value: ValueSelector::Min,
    }];
    let mut incumbents = Vec::new();
    let (best, _, complete) = optimize_seeded_with_scope_and_relaxation(
        &mut solver,
        &search,
        None,
        Objective::Linear { coeffs: &coefficients, vars: &terms },
        false,
        &AtomicBool::new(false),
        0,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        phases,
        None,
        |value, assignment| {
            assert!(assignment[1..].iter().all(|&term| term <= assignment[0]));
            assert_eq!(value, assignment[1..].iter().map(|&term| i64::from(term)).sum::<i64>());
            incumbents.push(value);
        },
    );
    assert!(complete);
    assert_eq!(best.map(|(_, value)| value), Some(9));
    assert_eq!(incumbents, [0, 9]);
}

#[test]
fn wide_boolean_affine_initial_hints_and_explicit_value_selectors_keep_precedence() {
    for (selector, hinted, minimizing, expected_first) in [
        (None, 9, false, 0),
        (None, 1, false, 8),
        (Some(ValueSelector::Min), 0, false, 0),
        (Some(ValueSelector::Max), 0, true, 9),
        (Some(ValueSelector::Hint), 9, false, 0),
        (Some(ValueSelector::Auto), 0, false, 9),
    ] {
        let mut solver = Solver::new();
        let variables = (0..9).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
        let coefficients = vec![1; variables.len()];
        let initial_phase = (0..variables.len()).map(|index| (index < hinted).then_some(0)).collect();
        let phases = selector
            .map(|value| {
                vec![CompiledSearchPhase {
                    scope: variables.clone(),
                    semantic_salts: (0..variables.len() as u64).collect(),
                    variable: VariableSelector::Auto,
                    value,
                }]
            })
            .unwrap_or_default();
        let mut first = None;
        let (best, _, complete) = optimize_seeded_with_scope_and_relaxation(
            &mut solver,
            &variables,
            None,
            Objective::Linear { coeffs: &coefficients, vars: &variables },
            minimizing,
            &AtomicBool::new(false),
            0,
            None,
            None,
            &[],
            None,
            initial_phase,
            Vec::new(),
            phases,
            None,
            |value, _| {
                first.get_or_insert(value);
            },
        );
        assert_eq!(first, Some(expected_first), "selector={selector:?}, hinted={hinted}");
        assert!(complete);
        assert_eq!(best.map(|(_, value)| value), Some(if minimizing { 0 } else { 9 }));
    }
}

#[test]
fn wide_boolean_affine_directions_ignore_zero_and_cancelled_non_boolean_terms() {
    let mut solver = Solver::new();
    let terms = (0..9).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let cancelled = solver.new_var_range(0, 4);
    let zero = solver.new_var_range(0, 4);
    let mut objective_vars = terms.clone();
    objective_vars.extend([cancelled, cancelled, zero]);
    let mut coefficients = vec![1; terms.len()];
    coefficients.extend([7, -7, 0]);
    let mut search = terms;
    search.extend([cancelled, zero]);
    let mut first = None;
    let (best, _, complete) = optimize_seeded(
        &mut solver,
        &search,
        Objective::Linear { coeffs: &coefficients, vars: &objective_vars },
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
            if first.replace(value).is_none() {
                assert_eq!(&assignment[9..], &[0, 0], "early objective guidance must preserve neutral terms' initial minimum values");
            }
        },
    );
    assert!(complete);
    assert_eq!(first, Some(9));
    assert_eq!(best.map(|(_, value)| value), Some(9));
}

#[test]
fn constrained_wide_boolean_affine_objectives_with_mixed_signs_match_brute_force() {
    let coefficients = [3, -2, 5, -7, 1, 4, -3, 2, -1, 6];
    let feasible_values = (0u16..1 << coefficients.len())
        .filter(|bits| bits.count_ones() <= 4 && (bits & 1) + ((bits >> 1) & 1) == 1)
        .map(|bits| coefficients.iter().enumerate().map(|(index, coefficient)| coefficient * i64::from((bits >> index) & 1)).sum::<i64>())
        .collect::<Vec<_>>();
    for minimizing in [false, true] {
        let optimum = if minimizing { *feasible_values.iter().min().unwrap() } else { *feasible_values.iter().max().unwrap() };
        for representation in 0..3 {
            let mut solver = Solver::new();
            let terms = (0..coefficients.len()).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
            linear(&mut solver, &vec![1; terms.len()], &terms, Relation::Le, 4);
            linear(&mut solver, &[1, 1], &terms[..2], Relation::Eq, 1);
            let offset = if representation == 0 { 0 } else { 7 };
            let mut search = terms.clone();
            let materialized = (representation == 2).then(|| solver.new_var_range(-50, 50));
            if let Some(materialized) = materialized {
                let mut equality_coefficients = coefficients.to_vec();
                equality_coefficients.push(-1);
                let mut equality_terms = terms.clone();
                equality_terms.push(materialized);
                linear(&mut solver, &equality_coefficients, &equality_terms, Relation::Eq, -offset);
                search.push(materialized);
            }
            let expression = Expr::Add(
                std::iter::once(Expr::Const(offset))
                    .chain(
                        coefficients
                            .iter()
                            .zip(&terms)
                            .map(|(&coefficient, &term)| Expr::Mul(vec![Expr::Const(coefficient), Expr::Var(term)])),
                    )
                    .collect(),
            );
            let objective = match representation {
                0 => Objective::Linear { coeffs: &coefficients, vars: &terms },
                1 => Objective::Expr(&expression),
                _ => Objective::VarWithAffine { objective: materialized.unwrap(), coeffs: &coefficients, vars: &terms },
            };
            let mut previous = None;
            let (best, _, complete) = optimize_seeded(
                &mut solver,
                &search,
                objective,
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
                    let terms = &assignment[..coefficients.len()];
                    assert!(terms.iter().all(|&term| (0..=1).contains(&term)));
                    assert!(terms.iter().sum::<i32>() <= 4);
                    assert_eq!(terms[0] + terms[1], 1);
                    assert_eq!(
                        value,
                        offset + terms.iter().zip(&coefficients).map(|(&term, coefficient)| i64::from(term) * coefficient).sum::<i64>()
                    );
                    if let Some(previous) = previous.replace(value) {
                        assert!(if minimizing { value < previous } else { value > previous });
                    }
                },
            );
            assert!(complete, "representation={representation}, minimizing={minimizing}");
            assert_eq!(best.map(|(_, value)| value), Some(optimum + offset));
        }
    }
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

#[test]
fn makespan_completion_scope_preserves_feasible_order_hints_before_objective_branching() {
    let run = |use_primary: bool| {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 1);
        let y = solver.new_var_range(0, 1);
        let makespan = solver.new_var_range(1, 2);
        for variable in [x, y] {
            linear(&mut solver, &[1, -1], &[makespan, variable], Relation::Ge, 1);
        }
        let search = [x, y, makespan];
        let mut phase = vec![None; solver.store.num_vars()];
        phase[x.index()] = Some(1);
        phase[y.index()] = Some(1);
        let mut incumbents = Vec::new();
        let (best, _, complete) = optimize_seeded_with_scope(
            &mut solver,
            &search,
            use_primary.then_some(&search[..2]),
            Objective::Var(makespan),
            true,
            &AtomicBool::new(false),
            0,
            None,
            None,
            &[],
            None,
            phase,
            Vec::new(),
            |value, assignment| incumbents.push((value, assignment.to_vec())),
        );
        assert!(complete);
        assert_eq!(best, Some((vec![0, 0, 1], 1)));
        incumbents
    };

    assert_eq!(run(false), vec![(1, vec![0, 0, 1])]);
    assert_eq!(run(true), vec![(2, vec![1, 1, 2]), (1, vec![0, 0, 1])]);
}

#[test]
fn makespan_is_still_completed_when_the_primary_scope_is_empty_or_already_fixed() {
    for fixed_primary in [false, true] {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 1);
        let y = solver.new_var_range(0, 1);
        let makespan = solver.new_var_range(1, 2);
        for variable in [x, y] {
            linear(&mut solver, &[1, -1], &[makespan, variable], Relation::Ge, 1);
            if fixed_primary {
                solver.store.fix(variable, 0).unwrap();
            }
        }
        let search = [x, y, makespan];
        let primary = if fixed_primary { &search[..2] } else { &search[..0] };
        let mut incumbents = Vec::new();
        let (best, _, complete) = optimize_seeded_with_scope(
            &mut solver,
            &search,
            Some(primary),
            Objective::Var(makespan),
            true,
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
        assert!(complete);
        assert_eq!(best, Some((vec![0, 0, 1], 1)));
        assert_eq!(incumbents, vec![(1, vec![0, 0, 1])]);
    }
}
