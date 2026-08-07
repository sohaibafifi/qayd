//! `precedence` (value precedence) cross-checked against the oracle.

use std::collections::BTreeSet;

use qayd::constraints::primitives::{precedence, precedence_with_covered};
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

/// First occurrence of each value, in order, must be non-decreasing in position.
fn prec_ok(a: &[i32], values: &[i32]) -> bool {
    let first = |val: i32| a.iter().position(|&x| x == val);
    for w in values.windows(2) {
        match (first(w[0]), first(w[1])) {
            (_, None) => {}                  // later value absent: fine
            (None, Some(_)) => return false, // later present, earlier absent
            (Some(fs), Some(ft)) => {
                if fs >= ft {
                    return false;
                }
            }
        }
    }
    true
}

fn prec_covered_ok(a: &[i32], values: &[i32]) -> bool {
    prec_ok(a, values) && values.iter().all(|&value| a.contains(&value))
}

#[test]
fn precedence_matches_oracle() {
    let n = 4;
    let values = [0, 1, 2];

    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..n).map(|_| s.new_var_range(0, 2)).collect();
    precedence(&mut s, &vars, &values);
    let got = enumerate(&mut s, &vars);

    let domains = vec![(0..=2).collect::<Vec<i32>>(); n];
    let want: BTreeSet<Vec<i32>> = oracle::solutions(&domains, |a| prec_ok(a, &values)).into_iter().collect();

    assert_eq!(got, want);
    // The first slot must be value 0 (nothing precedes it for 1 or 2).
    assert!(got.iter().all(|sol| sol[0] == 0));
}

#[test]
fn covered_precedence_requires_each_value() {
    let n = 4;
    let values = [0, 1, 2];

    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..n).map(|_| s.new_var_range(0, 2)).collect();
    precedence_with_covered(&mut s, &vars, &values, true);
    let got = enumerate(&mut s, &vars);

    let domains = vec![(0..=2).collect::<Vec<i32>>(); n];
    let want: BTreeSet<Vec<i32>> = oracle::solutions(&domains, |a| prec_covered_ok(a, &values)).into_iter().collect();

    assert_eq!(got, want);
    assert!(got.iter().all(|sol| values.iter().all(|value| sol.contains(value))));
}
