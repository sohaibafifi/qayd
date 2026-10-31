use std::collections::BTreeSet;

use qayd::constraints::linear::{linear, Relation};
use qayd::{solve_search, SearchControl, Solver};

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

fn solver_solutions(terms: &[(i64, usize)], relation: Relation, rhs: i64) -> BTreeSet<Vec<i32>> {
    let mut solver = Solver::new();
    let variables = (0..3).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    let coefficients = terms.iter().map(|&(coefficient, _)| coefficient).collect::<Vec<_>>();
    let scope = terms.iter().map(|&(_, variable)| variables[variable]).collect::<Vec<_>>();
    linear(&mut solver, &coefficients, &scope, relation, rhs);

    let mut solutions = BTreeSet::new();
    solve_search(&mut solver, &variables, |store| {
        solutions.insert(variables.iter().map(|&variable| store.store.value(variable)).collect());
        SearchControl::Continue
    });
    solutions
}

fn oracle_solutions(terms: &[(i64, usize)], relation: Relation, rhs: i64) -> BTreeSet<Vec<i32>> {
    let mut solutions = BTreeSet::new();
    for mask in 0..8u8 {
        let values = (0..3).map(|variable| i32::from(mask & (1 << variable) != 0)).collect::<Vec<_>>();
        let lhs = terms.iter().map(|&(coefficient, variable)| i128::from(coefficient) * i128::from(values[variable])).sum();
        if holds(lhs, relation, rhs) {
            solutions.insert(values);
        }
    }
    solutions
}

#[test]
fn mixed_signs_duplicates_and_all_relations_match_the_oracle() {
    let terms = [(5, 0), (-2, 0), (-4, 1), (3, 2)];
    for relation in [Relation::Eq, Relation::Ne, Relation::Le, Relation::Lt, Relation::Ge, Relation::Gt] {
        for rhs in -8..=8 {
            assert_eq!(solver_solutions(&terms, relation, rhs), oracle_solutions(&terms, relation, rhs), "relation={relation:?} rhs={rhs}");
        }
    }
}

#[test]
fn complemented_literals_propagate_after_backtracking() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    linear(&mut solver, &[-7, 5], &[x, y], Relation::Le, 0);

    solver.store.push_level();
    solver.store.fix(y, 1).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(x), 1);
    solver.store.pop_level();

    solver.store.push_level();
    solver.store.fix(x, 0).unwrap();
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(y), 0);
    solver.store.pop_level();
}

#[test]
fn boolean_equality_reaches_a_two_sided_fixpoint() {
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 1);
    linear(&mut solver, &[2, -3], &[x, y], Relation::Eq, 0);
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(x), 0);
    assert_eq!(solver.store.value(y), 0);
}

#[test]
fn strict_relations_are_safe_at_i64_boundaries() {
    let cases = [
        (i64::MAX, Relation::Lt, i64::MIN),
        (i64::MAX, Relation::Gt, i64::MAX),
        (i64::MIN, Relation::Lt, i64::MIN),
        (i64::MIN, Relation::Le, i64::MIN),
    ];
    for (coefficient, relation, rhs) in cases {
        let terms = [(coefficient, 0)];
        assert_eq!(solver_solutions(&terms, relation, rhs), oracle_solutions(&terms, relation, rhs));
    }
}

#[test]
fn root_fixed_boolean_domains_are_normalized_without_losing_constants() {
    let mut solver = Solver::new();
    let fixed_true = solver.new_var_range(1, 1);
    let fixed_false = solver.new_var_range(0, 0);
    let decision = solver.new_var_range(0, 1);
    linear(&mut solver, &[-5, 7, 3], &[fixed_true, fixed_false, decision], Relation::Le, -3);
    solver.propagate().unwrap();
    assert_eq!(solver.store.value(decision), 0);
}
