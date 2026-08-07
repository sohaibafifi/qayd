//! Oracle cross-checks for `element` and `count`.

use std::collections::BTreeSet;

use qayd::constraints::count::count;
use qayd::constraints::linear::Relation;
use qayd::constraints::primitives::element;
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
fn element_matches_oracle() {
    // layout: [idx, value, a0, a1, a2]
    let mut s = Solver::new();
    let idx = s.new_var_range(0, 2);
    let value = s.new_var_range(0, 3);
    let a: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 3)).collect();
    element(&mut s, &a, idx, value);

    let mut all = vec![idx, value];
    all.extend(&a);
    let got = enumerate(&mut s, &all);

    let domains = vec![
        (0..=2).collect::<Vec<i32>>(), // idx
        (0..=3).collect::<Vec<i32>>(), // value
        (0..=3).collect::<Vec<i32>>(), // a0
        (0..=3).collect::<Vec<i32>>(), // a1
        (0..=3).collect::<Vec<i32>>(), // a2
    ];
    let want: BTreeSet<Vec<i32>> = oracle::solutions(&domains, |x| {
        let (i, v) = (x[0] as usize, x[1]);
        x[2 + i] == v
    })
    .into_iter()
    .collect();

    assert_eq!(got, want);
}

#[test]
fn element_with_index_offset_into_smaller_array() {
    // value = a[idx] where idx in 0..1, a has 2 entries; tighten value's domain.
    let mut s = Solver::new();
    let idx = s.new_var_range(0, 1);
    let value = s.new_var_range(0, 9);
    let a0 = s.new_var_set(&[5]);
    let a1 = s.new_var_set(&[8]);
    element(&mut s, &[a0, a1], idx, value);

    let got = enumerate(&mut s, &[idx, value, a0, a1]);
    let mut want = BTreeSet::new();
    want.insert(vec![0, 5, 5, 8]);
    want.insert(vec![1, 8, 5, 8]);
    assert_eq!(got, want);
}

fn count_check(n: usize, lo: i32, hi: i32, value: i32, rel: Relation, k: i64) {
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..n).map(|_| s.new_var_range(lo, hi)).collect();
    count(&mut s, &vars, value, rel, k);
    let got = enumerate(&mut s, &vars);

    let domains = vec![(lo..=hi).collect::<Vec<i32>>(); n];
    let want: BTreeSet<Vec<i32>> = oracle::solutions(&domains, |a| {
        let c = a.iter().filter(|&&x| x == value).count() as i64;
        match rel {
            Relation::Eq => c == k,
            Relation::Ne => c != k,
            Relation::Le => c <= k,
            Relation::Lt => c < k,
            Relation::Ge => c >= k,
            Relation::Gt => c > k,
        }
    })
    .into_iter()
    .collect();

    assert_eq!(got, want, "count rel={rel:?} k={k}");
}

#[test]
fn count_all_relations_match_oracle() {
    for k in 0..=4 {
        count_check(4, 0, 2, 1, Relation::Eq, k);
        count_check(4, 0, 2, 1, Relation::Ne, k);
        count_check(4, 0, 2, 1, Relation::Le, k);
        count_check(4, 0, 2, 1, Relation::Lt, k);
        count_check(4, 0, 2, 1, Relation::Ge, k);
        count_check(4, 0, 2, 1, Relation::Gt, k);
    }
}
