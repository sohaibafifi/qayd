//! Composed small models: several constraint families working together, each
//! checked against the brute-force oracle. This is the Phase 1 milestone.

use std::collections::BTreeSet;

use qayd::constraints::linear::{sum, Relation};
use qayd::constraints::primitives::{all_equal, maximum, not_equal, ordered};
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

#[test]
fn distinct_increasing_with_fixed_sum() {
    // a < b < c, all in 0..6, a + b + c = 6.
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 6)).collect();
    ordered(&mut s, &v, Relation::Lt);
    sum(&mut s, &v, Relation::Eq, 6);
    let got = enumerate(&mut s, &v);

    let want: BTreeSet<Vec<i32>> =
        oracle::solutions(&vec![(0..=6).collect::<Vec<i32>>(); 3], |a| a[0] < a[1] && a[1] < a[2] && a.iter().sum::<i32>() == 6)
            .into_iter()
            .collect();

    assert_eq!(got, want);
    assert_eq!(got, BTreeSet::from([vec![0, 1, 5], vec![0, 2, 4], vec![1, 2, 3]]));
}

#[test]
fn alldifferent_plus_sum_window() {
    // 4 distinct values in 0..4 with sum bounded into [4, 8].
    let n = 4;
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..n).map(|_| s.new_var_range(0, 4)).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            not_equal(&mut s, v[i], v[j]);
        }
    }
    sum(&mut s, &v, Relation::Ge, 4);
    sum(&mut s, &v, Relation::Le, 8);
    let got = enumerate(&mut s, &v);

    let want: BTreeSet<Vec<i32>> = oracle::solutions(&vec![(0..=4).collect::<Vec<i32>>(); n], |a| {
        let distinct = (0..a.len()).all(|i| ((i + 1)..a.len()).all(|j| a[i] != a[j]));
        let total: i32 = a.iter().sum();
        distinct && (4..=8).contains(&total)
    })
    .into_iter()
    .collect();

    assert_eq!(got, want);
}

#[test]
fn maximum_with_all_equal_inputs() {
    // y = max(xs) and all xs equal => y equals the shared value.
    let mut s = Solver::new();
    let y = s.new_var_range(0, 5);
    let xs: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 5)).collect();
    all_equal(&mut s, &xs);
    maximum(&mut s, y, &xs);

    let mut all = vec![y];
    all.extend(&xs);
    let got = enumerate(&mut s, &all);

    let want: BTreeSet<Vec<i32>> = oracle::solutions(&vec![(0..=5).collect::<Vec<i32>>(); 4], |a| {
        let (y, xs) = (a[0], &a[1..]);
        xs.iter().all(|&x| x == xs[0]) && y == *xs.iter().max().unwrap()
    })
    .into_iter()
    .collect();

    assert_eq!(got, want);
    assert_eq!(got.len(), 6); // shared value 0..5, y matches
}
