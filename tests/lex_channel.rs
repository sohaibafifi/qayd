//! Oracle cross-checks for lex, channel, and slide.

use std::collections::BTreeSet;

use qayd::constraints::lex::{channel, lex, slide};
use qayd::constraints::primitives::not_equal;
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

fn lex_le(a: &[i32], b: &[i32]) -> bool {
    a <= b
}

#[test]
fn lex_le_matches_oracle() {
    // [x0,x1] <=lex [y0,y1] over 0..2.
    let mut s = Solver::new();
    let x: Vec<VarId> = (0..2).map(|_| s.new_var_range(0, 2)).collect();
    let y: Vec<VarId> = (0..2).map(|_| s.new_var_range(0, 2)).collect();
    lex(&mut s, &x, &y, false);

    let mut all = x.clone();
    all.extend(&y);
    let got = enumerate(&mut s, &all);

    let want: BTreeSet<Vec<i32>> =
        oracle::solutions(&vec![(0..=2).collect::<Vec<i32>>(); 4], |a| lex_le(&a[0..2], &a[2..4])).into_iter().collect();
    assert_eq!(got, want);
}

#[test]
fn lex_strict_matches_oracle() {
    let mut s = Solver::new();
    let x: Vec<VarId> = (0..2).map(|_| s.new_var_range(0, 1)).collect();
    let y: Vec<VarId> = (0..2).map(|_| s.new_var_range(0, 1)).collect();
    lex(&mut s, &x, &y, true);

    let mut all = x.clone();
    all.extend(&y);
    let got = enumerate(&mut s, &all);

    let want: BTreeSet<Vec<i32>> = oracle::solutions(&vec![(0..=1).collect::<Vec<i32>>(); 4], |a| a[0..2] < a[2..4]).into_iter().collect();
    assert_eq!(got, want);
}

#[test]
fn channel_is_inverse_permutation() {
    let n = 4;
    let mut s = Solver::new();
    let x: Vec<VarId> = (0..n).map(|_| s.new_var_range(0, (n - 1) as i32)).collect();
    let y: Vec<VarId> = (0..n).map(|_| s.new_var_range(0, (n - 1) as i32)).collect();
    channel(&mut s, &x, &y);

    let mut all = x.clone();
    all.extend(&y);
    let got = enumerate(&mut s, &all);

    let want: BTreeSet<Vec<i32>> = oracle::solutions(&vec![(0..=(n as i32 - 1)).collect::<Vec<i32>>(); 2 * n], |a| {
        let (xv, yv) = a.split_at(n);
        (0..n).all(|i| (0..n).all(|j| (xv[i] == j as i32) == (yv[j] == i as i32)))
    })
    .into_iter()
    .collect();

    assert_eq!(got, want);
    // x must be a permutation; there are n! of them.
    assert_eq!(got.len(), (1..=n).product::<usize>());
}

#[test]
fn slide_distinct_neighbours_matches_oracle() {
    // slide "a != b" across windows of width 2 => no two adjacent equal.
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 2)).collect();
    slide(&mut s, &v, 2, |solver, w| {
        not_equal(solver, w[0], w[1]);
    });
    let got = enumerate(&mut s, &v);

    let want: BTreeSet<Vec<i32>> =
        oracle::solutions(&vec![(0..=2).collect::<Vec<i32>>(); 4], |a| a.windows(2).all(|w| w[0] != w[1])).into_iter().collect();
    assert_eq!(got, want);
}
