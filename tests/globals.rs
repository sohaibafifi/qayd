//! Oracle cross-checks for the Phase 1 globals: ordered, allEqual, minimum,
//! maximum, instantiation.

use std::collections::BTreeSet;

use qayd::constraints::linear::Relation;
use qayd::constraints::primitives::{all_equal, instantiation, maximum, minimum, ordered};
mod common;

use common::oracle;
use qayd::{solve_search, SearchControl, Solver, VarId};

fn enumerate(solver: &mut Solver, vars: &[VarId]) -> BTreeSet<Vec<i32>> {
    let mut out = BTreeSet::new();
    solve_search(solver, vars, |s| {
        out.insert(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    out
}

fn oracle_set(n: usize, lo: i32, hi: i32, pred: impl FnMut(&[i32]) -> bool) -> BTreeSet<Vec<i32>> {
    let domains = vec![(lo..=hi).collect::<Vec<i32>>(); n];
    oracle::solutions(&domains, pred).into_iter().collect()
}

#[test]
fn ordered_nondecreasing_matches_oracle() {
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 3)).collect();
    ordered(&mut s, &vars, Relation::Le);
    let got = enumerate(&mut s, &vars);
    let want = oracle_set(4, 0, 3, |a| a.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(got, want);
}

#[test]
fn ordered_strictly_increasing_matches_oracle() {
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 4)).collect();
    ordered(&mut s, &vars, Relation::Lt);
    let got = enumerate(&mut s, &vars);
    let want = oracle_set(3, 0, 4, |a| a.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(got, want);
}

#[test]
fn all_equal_matches_oracle() {
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 3)).collect();
    all_equal(&mut s, &vars);
    let got = enumerate(&mut s, &vars);
    let want = oracle_set(3, 0, 3, |a| a.iter().all(|&x| x == a[0]));
    assert_eq!(got, want);
    assert_eq!(got.len(), 4); // one per shared value
}

#[test]
fn minimum_matches_oracle() {
    // vars = [y, x0, x1, x2]; y = min(x0, x1, x2).
    let mut s = Solver::new();
    let y = s.new_var_range(0, 3);
    let xs: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 3)).collect();
    minimum(&mut s, y, &xs);
    let mut all = vec![y];
    all.extend(&xs);
    let got = enumerate(&mut s, &all);
    let want = oracle_set(4, 0, 3, |a| a[0] == *a[1..].iter().min().unwrap());
    assert_eq!(got, want);
}

#[test]
fn maximum_matches_oracle() {
    let mut s = Solver::new();
    let y = s.new_var_range(0, 3);
    let xs: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 3)).collect();
    maximum(&mut s, y, &xs);
    let mut all = vec![y];
    all.extend(&xs);
    let got = enumerate(&mut s, &all);
    let want = oracle_set(4, 0, 3, |a| a[0] == *a[1..].iter().max().unwrap());
    assert_eq!(got, want);
}

#[test]
fn instantiation_pins_values() {
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 9)).collect();
    instantiation(&mut s, &vars, &[3, 7, 1]);
    let got = enumerate(&mut s, &vars);
    let mut want = BTreeSet::new();
    want.insert(vec![3, 7, 1]);
    assert_eq!(got, want);
}

#[test]
fn instantiation_with_impossible_value_is_unsat() {
    let mut s = Solver::new();
    let x = s.new_var_range(0, 2);
    instantiation(&mut s, &[x], &[5]); // 5 not in domain
    let got = enumerate(&mut s, &[x]);
    assert!(got.is_empty());
}
