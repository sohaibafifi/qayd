//! Linear/sum constraints cross-checked against the brute-force oracle.

use std::collections::BTreeSet;

use qayd::constraints::linear::{linear, sum, Relation};
mod common;

use common::oracle;
use qayd::{solve_search, SearchControl, Solver, VarId};

/// Enumerate the solver's solutions for `n` vars over `[lo, hi]` after posting
/// a single linear constraint, as a sorted set.
fn solver_set(n: usize, lo: i32, hi: i32, coeffs: &[i64], rel: Relation, rhs: i64) -> BTreeSet<Vec<i32>> {
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..n).map(|_| s.new_var_range(lo, hi)).collect();
    linear(&mut s, coeffs, &vars, rel, rhs);
    let mut out = BTreeSet::new();
    solve_search(&mut s, &vars, |st| {
        out.insert(vars.iter().map(|&v| st.store.value(v)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    out
}

fn oracle_set(n: usize, lo: i32, hi: i32, coeffs: &[i64], rel: Relation, rhs: i64) -> BTreeSet<Vec<i32>> {
    let domains = vec![(lo..=hi).collect::<Vec<i32>>(); n];
    let pred = |a: &[i32]| {
        let lhs: i64 = a.iter().zip(coeffs).map(|(&x, &c)| c * x as i64).sum();
        match rel {
            Relation::Eq => lhs == rhs,
            Relation::Ne => lhs != rhs,
            Relation::Le => lhs <= rhs,
            Relation::Lt => lhs < rhs,
            Relation::Ge => lhs >= rhs,
            Relation::Gt => lhs > rhs,
        }
    };
    oracle::solutions(&domains, pred).into_iter().collect()
}

fn check(n: usize, lo: i32, hi: i32, coeffs: &[i64], rel: Relation, rhs: i64) {
    let got = solver_set(n, lo, hi, coeffs, rel, rhs);
    let want = oracle_set(n, lo, hi, coeffs, rel, rhs);
    assert_eq!(got, want, "rel={rel:?} rhs={rhs} coeffs={coeffs:?}");
}

#[test]
fn sum_all_relations_match_oracle() {
    let ones = [1i64; 3];
    for rhs in -1..=10 {
        check(3, 0, 3, &ones, Relation::Eq, rhs);
        check(3, 0, 3, &ones, Relation::Ne, rhs);
        check(3, 0, 3, &ones, Relation::Le, rhs);
        check(3, 0, 3, &ones, Relation::Lt, rhs);
        check(3, 0, 3, &ones, Relation::Ge, rhs);
        check(3, 0, 3, &ones, Relation::Gt, rhs);
    }
}

#[test]
fn weighted_linear_matches_oracle() {
    // Mixed signs and magnitudes.
    let coeffs = [2i64, -3, 1];
    for rhs in -6..=6 {
        check(3, -2, 3, &coeffs, Relation::Eq, rhs);
        check(3, -2, 3, &coeffs, Relation::Le, rhs);
        check(3, -2, 3, &coeffs, Relation::Ge, rhs);
        check(3, -2, 3, &coeffs, Relation::Ne, rhs);
    }
}

#[test]
fn unsatisfiable_sum() {
    // 3 vars in 0..3 cannot sum to 10.
    assert!(solver_set(3, 0, 3, &[1, 1, 1], Relation::Eq, 10).is_empty());
}

#[test]
fn sum_helper_matches_explicit_coeffs() {
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 5)).collect();
    sum(&mut s, &vars, Relation::Eq, 7);
    let mut count = 0u64;
    solve_search(&mut s, &vars, |_| {
        count += 1;
        SearchControl::Continue
    });
    // Number of (a,b,c) in 0..5 with a+b+c = 7.
    let want = oracle::count(&vec![(0..=5).collect::<Vec<i32>>(); 3], |a| a.iter().sum::<i32>() == 7);
    assert_eq!(count as usize, want);
}

// --- linear propagation unit tests (moved from src/constraints/linear.rs) ---

#[test]
fn leq_prunes_bounds_at_root() {
    let mut s = Solver::new();
    let x = s.new_var_range(0, 10);
    let y = s.new_var_range(0, 10);
    sum(&mut s, &[x, y], Relation::Le, 5);
    s.store.push_level();
    s.enqueue_all();
    s.propagate().unwrap();
    assert_eq!(s.store.max(x), 5);
    assert_eq!(s.store.max(y), 5);
    s.store.pop_level();
}

#[test]
fn eq_propagates_both_directions() {
    let mut s = Solver::new();
    let x = s.new_var_range(0, 10);
    let y = s.new_var_range(0, 10);
    sum(&mut s, &[x, y], Relation::Eq, 12);
    s.store.push_level();
    s.enqueue_all();
    s.propagate().unwrap();
    assert_eq!(s.store.min(x), 2);
    assert_eq!(s.store.min(y), 2);
    s.store.pop_level();
}
