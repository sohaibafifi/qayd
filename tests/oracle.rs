//! Cross-check the solver's enumeration against the brute-force oracle. This is
//! the pattern every future constraint test follows.

use std::collections::BTreeSet;

use qayd::constraints::primitives::not_equal;
mod common;

use common::oracle;
use qayd::{solve_search, SearchControl, Solver, VarId};

/// Solve a pairwise-distinct CSP and collect every solution as a sorted set.
fn solver_solutions(n_vars: usize, lo: i32, hi: i32) -> BTreeSet<Vec<i32>> {
    let mut solver = Solver::new();
    let vars: Vec<VarId> = (0..n_vars).map(|_| solver.new_var_range(lo, hi)).collect();
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            not_equal(&mut solver, vars[i], vars[j]);
        }
    }
    let mut sols = BTreeSet::new();
    solve_search(&mut solver, &vars, |s| {
        sols.insert(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    sols
}

fn all_distinct(a: &[i32]) -> bool {
    (0..a.len()).all(|i| ((i + 1)..a.len()).all(|j| a[i] != a[j]))
}

#[test]
fn all_distinct_matches_oracle_square() {
    // 4 vars over {0..3}: solver set must equal the oracle set exactly (4! = 24).
    let solver_set = solver_solutions(4, 0, 3);
    let domains = vec![(0..=3).collect::<Vec<i32>>(); 4];
    let oracle_set: BTreeSet<Vec<i32>> = oracle::solutions(&domains, all_distinct).into_iter().collect();

    assert_eq!(solver_set, oracle_set);
    assert_eq!(solver_set.len(), 24);
}

#[test]
fn all_distinct_matches_oracle_rectangular() {
    // 3 vars over {0..4}: more values than vars. 5 * 4 * 3 = 60 injections.
    let solver_set = solver_solutions(3, 0, 4);
    let domains = vec![(0..=4).collect::<Vec<i32>>(); 3];
    let oracle_set: BTreeSet<Vec<i32>> = oracle::solutions(&domains, all_distinct).into_iter().collect();

    assert_eq!(solver_set, oracle_set);
    assert_eq!(solver_set.len(), 60);
}

#[test]
fn overconstrained_is_unsat() {
    // 4 vars but only 3 values: pigeonhole => no all-different assignment.
    let solver_set = solver_solutions(4, 0, 2);
    assert!(solver_set.is_empty());
}

// --- oracle module unit tests (moved from src/oracle.rs) ---

#[test]
fn oracle_counts_all_distinct() {
    let domains = vec![vec![0, 1, 2]; 3];
    let n = oracle::count(&domains, |a| a[0] != a[1] && a[0] != a[2] && a[1] != a[2]);
    assert_eq!(n, 6);
}

#[test]
fn oracle_enumerates_full_product_when_trivially_true() {
    let domains = vec![vec![0, 1], vec![5, 6, 7]];
    assert_eq!(oracle::solutions(&domains, |_| true).len(), 6);
}
