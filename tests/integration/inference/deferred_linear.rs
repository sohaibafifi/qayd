//! Independent Boolean truth-table gates for the first deferred producer.

use std::collections::BTreeSet;

use qayd::constraints::linear::{linear, Relation};
use qayd::inference::InferenceCheckConfig;
use qayd::{solve_search, SearchControl, SolveStats, Solver};

fn holds(lhs: i128, relation: Relation, rhs: i64) -> bool {
    let rhs = i128::from(rhs);
    match relation {
        Relation::Eq => lhs == rhs,
        Relation::Ne => lhs != rhs,
        Relation::Le => lhs <= rhs,
        Relation::Lt => lhs < rhs,
        Relation::Ge => lhs >= rhs,
        Relation::Gt => lhs > rhs,
    }
}

fn oracle(terms: &[(i64, usize)], relation: Relation, rhs: i64) -> BTreeSet<Vec<i32>> {
    (0..16u8)
        .map(|mask| (0..4).map(|variable| i32::from(mask & (1 << variable) != 0)).collect::<Vec<_>>())
        .filter(|values| {
            holds(terms.iter().map(|&(coefficient, variable)| i128::from(coefficient) * i128::from(values[variable])).sum(), relation, rhs)
        })
        .collect()
}

fn solutions(terms: &[(i64, usize)], relation: Relation, rhs: i64, deferred: bool, checking: bool) -> (BTreeSet<Vec<i32>>, SolveStats) {
    let mut solver = Solver::new();
    solver.set_deferred_explanations(deferred);
    let variables = (0..4).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let coefficients = terms.iter().map(|&(coefficient, _)| coefficient).collect::<Vec<_>>();
    let scope = terms.iter().map(|&(_, variable)| variables[variable]).collect::<Vec<_>>();
    linear(&mut solver, &coefficients, &scope, relation, rhs);
    if checking {
        solver.set_inference_checking(Some(InferenceCheckConfig::default()));
    }
    let mut actual = BTreeSet::new();
    let stats = solve_search(&mut solver, &variables, |state| {
        actual.insert(variables.iter().map(|&var| state.store.value(var)).collect());
        SearchControl::Continue
    });
    assert_eq!(solver.inference_check_stats().resource_limited, 0);
    (actual, stats)
}

#[test]
fn eager_and_deferred_normalized_boolean_rows_match_every_truth_table() {
    let families = [
        vec![(5, 0), (-2, 0), (-4, 1), (3, 2), (1, 3)],
        vec![(4, 0), (-4, 0), (-3, 1), (2, 2), (-2, 3)],
        vec![(1, 0), (1, 1), (1, 2), (1, 3)],
    ];
    for terms in families {
        for relation in [Relation::Le, Relation::Lt, Relation::Ge, Relation::Gt] {
            for rhs in -6..=6 {
                let expected = oracle(&terms, relation, rhs);
                let (eager, eager_stats) = solutions(&terms, relation, rhs, false, false);
                assert_eq!(eager, expected);
                for checking in [false, true] {
                    let (deferred, stats) = solutions(&terms, relation, rhs, true, checking);
                    assert_eq!(deferred, expected, "terms={terms:?}, relation={relation:?}, rhs={rhs}, checking={checking}");
                    assert_eq!(
                        (stats.nodes, stats.failures, stats.learned_lits),
                        (eager_stats.nodes, eager_stats.failures, eager_stats.learned_lits),
                        "deferred reasons must preserve search and learned clauses"
                    );
                }
            }
        }
    }
}

#[test]
fn deferred_strict_relations_and_duplicate_coefficients_handle_i64_extremes() {
    for (terms, relation, rhs) in [
        (vec![(i64::MAX, 0)], Relation::Lt, i64::MIN),
        (vec![(i64::MAX, 0)], Relation::Gt, i64::MAX),
        (vec![(i64::MIN, 0)], Relation::Lt, i64::MIN),
        (vec![(i64::MIN, 0)], Relation::Le, i64::MIN),
        (vec![(i64::MAX, 0), (i64::MAX, 0), (-1, 1)], Relation::Ge, i64::MAX),
        (vec![(i64::MIN, 0), (i64::MAX, 0), (1, 1)], Relation::Le, 0),
    ] {
        let expected = oracle(&terms, relation, rhs);
        for deferred in [false, true] {
            for checking in [false, true] {
                assert_eq!(solutions(&terms, relation, rhs, deferred, checking).0, expected);
            }
        }
    }
}

#[test]
fn empty_singleton_and_cancelled_reasons_preserve_solution_sets() {
    for terms in [vec![(2, 0)], vec![(2, 0), (1, 1)], vec![(2, 0), (-2, 0)], Vec::new()] {
        for rhs in [-1, 0, 1, 2] {
            let expected = oracle(&terms, Relation::Le, rhs);
            for deferred in [false, true] {
                assert_eq!(solutions(&terms, Relation::Le, rhs, deferred, true).0, expected);
            }
        }
    }
}

#[test]
fn multiple_deductions_share_history_across_complemented_values() {
    for deferred in [false, true] {
        for checking in [false, true] {
            let mut solver = Solver::new();
            solver.set_deferred_explanations(deferred);
            solver.set_explanation_metrics(true);
            let variables = (0..8).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
            // In normalized form, the first three literals are active when
            // x0=1, x1=0 and x2=1. Every remaining literal must be inactive.
            let coefficients = [1, -1, 1, -1, 1, -1, 1, -1];
            linear(&mut solver, &coefficients, &variables, Relation::Le, -1);
            solver.store.fix(variables[0], 1).unwrap();
            solver.store.fix(variables[1], 0).unwrap();
            solver.store.fix(variables[2], 1).unwrap();
            if checking {
                solver.set_inference_checking(Some(InferenceCheckConfig::default()));
            }
            let mut actual = Vec::new();
            solve_search(&mut solver, &variables, |state| {
                actual.push(variables.iter().map(|&var| state.store.value(var)).collect::<Vec<_>>());
                SearchControl::Continue
            });
            assert_eq!(actual, vec![vec![1, 0, 1, 1, 0, 1, 0, 1]]);
            let explanations = solver.explanation_stats();
            assert_eq!(explanations.created_handles, if deferred { 5 } else { 0 });
            assert_eq!(explanations.lcg_first_expansions, 0, "unused root reasons should stay deferred");
            assert_eq!(explanations.checker_expansions, if deferred && checking { 5 } else { 0 });
            if checking {
                assert!(solver.inference_check_stats().checked_inferences >= 5);
                assert_eq!(solver.inference_check_stats().resource_limited, 0);
            }
        }
    }
}

#[test]
fn equalities_and_disequalities_keep_the_eager_fallback() {
    let terms = [(2, 0), (-1, 1), (1, 2), (-2, 3)];
    for relation in [Relation::Eq, Relation::Ne] {
        for rhs in -3..=3 {
            assert_eq!(solutions(&terms, relation, rhs, true, true).0, oracle(&terms, relation, rhs));
        }
    }
}

#[test]
fn first_slice_cap_falls_back_to_eager_for_larger_rows() {
    for n in [256, 257] {
        let mut solver = Solver::new();
        solver.set_deferred_explanations(true);
        solver.set_explanation_metrics(true);
        let variables = (0..n).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
        linear(&mut solver, &vec![1; n], &variables, Relation::Le, 1);
        solver.store.fix(variables[0], 1).unwrap();
        let stats = solve_search(&mut solver, &variables, |_| SearchControl::Continue);
        assert_eq!(stats.solutions, 1);
        let explanations = solver.explanation_stats();
        assert_eq!(explanations.created_handles, if n == 256 { 255 } else { 0 });
        assert_eq!(explanations.lcg_first_expansions, 0);
        if n == 257 {
            assert!(explanations.eager_premise_buffers >= 256);
        }
    }
}
