//! `intension` cross-checked against the oracle across arithmetic, relational,
//! logical, and division/modulo expressions.

use std::collections::BTreeSet;

use qayd::constraints::intension::intension;
use qayd::expr::{abs, add, and, eq, ge, imp, lt, ne, or, rem, sub, var, Expr};
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

/// Build a 3-variable solver over `[lo, hi]`, post `expr`, enumerate.
fn solve_expr(lo: i32, hi: i32, build: impl Fn(VarId, VarId, VarId) -> Expr) -> BTreeSet<Vec<i32>> {
    let mut s = Solver::new();
    let a = s.new_var_range(lo, hi);
    let b = s.new_var_range(lo, hi);
    let c = s.new_var_range(lo, hi);
    intension(&mut s, build(a, b, c));
    enumerate(&mut s, &[a, b, c])
}

#[test]
fn all_different_via_intension() {
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 2)).collect();
    for i in 0..3 {
        for j in (i + 1)..3 {
            intension(&mut s, ne(var(vars[i]), var(vars[j])));
        }
    }
    let got = enumerate(&mut s, &vars);
    let want = oracle_set(3, 0, 2, |x| x[0] != x[1] && x[0] != x[2] && x[1] != x[2]);
    assert_eq!(got, want);
    assert_eq!(got.len(), 6);
}

#[test]
fn sum_equation_via_intension() {
    // a + b = c
    let got = solve_expr(0, 4, |a, b, c| eq(add(vec![var(a), var(b)]), var(c)));
    let want = oracle_set(3, 0, 4, |x| x[0] + x[1] == x[2]);
    assert_eq!(got, want);
}

#[test]
fn absolute_difference_constraint() {
    // |a - b| >= 2
    let got = solve_expr(0, 4, |a, b, _c| ge(abs(sub(var(a), var(b))), Expr::Const(2)));
    // c is unconstrained, so it ranges freely; mirror that in the oracle.
    let want = oracle_set(3, 0, 4, |x| (x[0] - x[1]).abs() >= 2);
    assert_eq!(got, want);
}

#[test]
fn logical_or_and_implication() {
    // (a < b) OR (b < c)
    let got_or = solve_expr(0, 3, |a, b, c| or(vec![lt(var(a), var(b)), lt(var(b), var(c))]));
    let want_or = oracle_set(3, 0, 3, |x| x[0] < x[1] || x[1] < x[2]);
    assert_eq!(got_or, want_or);

    // (a == 0) -> (b == c)
    let got_imp = solve_expr(0, 2, |a, b, c| imp(eq(var(a), Expr::Const(0)), eq(var(b), var(c))));
    let want_imp = oracle_set(3, 0, 2, |x| (x[0] != 0) || (x[1] == x[2]));
    assert_eq!(got_imp, want_imp);
}

#[test]
fn conjunction_matches_oracle() {
    // (a < b) AND (b < c)  -> strictly increasing
    let got = solve_expr(0, 4, |a, b, c| and(vec![lt(var(a), var(b)), lt(var(b), var(c))]));
    let want = oracle_set(3, 0, 4, |x| x[0] < x[1] && x[1] < x[2]);
    assert_eq!(got, want);
}

#[test]
fn modulo_constraint_uses_exact_check() {
    // a % 3 == 1  (bounds for `%` are conservative; the exact leaf check decides)
    let mut s = Solver::new();
    let a = s.new_var_range(0, 9);
    intension(&mut s, eq(rem(var(a), Expr::Const(3)), Expr::Const(1)));
    let got = enumerate(&mut s, &[a]);
    let want: BTreeSet<Vec<i32>> = [1, 4, 7].iter().map(|&x| vec![x]).collect();
    assert_eq!(got, want);
}

#[test]
fn unsatisfiable_intension() {
    // a > a is never true.
    let mut s = Solver::new();
    let a = s.new_var_range(0, 5);
    intension(&mut s, qayd::expr::gt(var(a), var(a)));
    assert!(enumerate(&mut s, &[a]).is_empty());
}
