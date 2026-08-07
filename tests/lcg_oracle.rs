//! Oracle-equivalence tests for the LCG engine: the solver's full solution set
//! must match brute-force enumeration on small models. This is the project's
//! primary correctness check — it exercises decisions, channeling, generic
//! propagator explanations (including holes), 1-UIP learning, and backjumping.

use std::collections::BTreeSet;

use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::primitives::{all_different, not_equal_offset};
mod common;

use common::oracle;
use qayd::{count_solutions, maximize, minimize, solve_search, SearchControl, Solver, VarId};

/// Every solution the engine finds over `vars`, as a set of value tuples.
fn engine_solutions(solver: &mut Solver, vars: &[VarId]) -> BTreeSet<Vec<i32>> {
    let mut set = BTreeSet::new();
    solve_search(solver, vars, |s| {
        set.insert(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    set
}

/// Brute-force solutions over explicit `domains`.
fn brute(domains: &[Vec<i32>], pred: impl FnMut(&[i32]) -> bool) -> BTreeSet<Vec<i32>> {
    oracle::solutions(domains, pred).into_iter().collect()
}

fn queens_ok(a: &[i32]) -> bool {
    let n = a.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if a[i] == a[j] || (a[i] - a[j]).abs() == (i as i32 - j as i32).abs() {
                return false;
            }
        }
    }
    true
}

fn build_queens(n: i32) -> (Solver, Vec<VarId>) {
    let mut solver = Solver::new();
    let q: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
    for i in 0..n as usize {
        for j in (i + 1)..n as usize {
            let (di, dj) = (i as i32, j as i32);
            not_equal_offset(&mut solver, q[i], q[j], 0);
            not_equal_offset(&mut solver, q[i], q[j], di - dj);
            not_equal_offset(&mut solver, q[i], q[j], dj - di);
        }
    }
    (solver, q)
}

#[test]
fn n_queens_counts_match_known_values() {
    // Classic counts: 4→2, 5→10, 6→4, 7→40.
    for (n, expected) in [(4, 2), (5, 10), (6, 4), (7, 40)] {
        let (mut solver, q) = build_queens(n);
        assert_eq!(count_solutions(&mut solver, &q), expected, "n={n}");
    }
}

#[test]
fn n_queens_solution_set_matches_oracle() {
    for n in 4..=6i32 {
        let (mut solver, q) = build_queens(n);
        let got = engine_solutions(&mut solver, &q);
        let domains: Vec<Vec<i32>> = (0..n).map(|_| (0..n).collect()).collect();
        let want = brute(&domains, queens_ok);
        assert_eq!(got, want, "n={n}");
    }
}

#[test]
fn holey_all_different_matches_oracle() {
    // Variables with explicit-set domains (holes), so equality atoms for absent
    // values are seeded as root facts and the generic explanation must carry
    // holes — the critical hole-soundness case.
    let sets = [vec![0, 2, 3], vec![1, 2], vec![0, 3, 4], vec![2, 3], vec![0, 1, 4]];
    let mut solver = Solver::new();
    let vars: Vec<VarId> = sets.iter().map(|s| solver.new_var_set(s)).collect();
    all_different(&mut solver, &vars);
    let got = engine_solutions(&mut solver, &vars);

    let domains: Vec<Vec<i32>> = sets.to_vec();
    let want = brute(&domains, |a| {
        let mut seen = BTreeSet::new();
        a.iter().all(|&x| seen.insert(x))
    });
    assert_eq!(got, want);
    assert!(!got.is_empty(), "model should be satisfiable");
}

#[test]
fn graph_coloring_matches_oracle() {
    // A 5-cycle with 3 colors.
    let n = 5usize;
    let k = 3i32;
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
    let mut solver = Solver::new();
    let v: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, k - 1)).collect();
    for &(a, b) in &edges {
        not_equal_offset(&mut solver, v[a], v[b], 0);
    }
    let got = engine_solutions(&mut solver, &v);

    let domains: Vec<Vec<i32>> = (0..n).map(|_| (0..k).collect()).collect();
    let want = brute(&domains, |c| edges.iter().all(|&(a, b)| c[a] != c[b]));
    assert_eq!(got, want);
}

#[test]
fn minimize_matches_brute_force_optimum() {
    // Minimise x+y+z subject to x+y+z >= 7, each in [0,5].
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 5);
    let y = solver.new_var_range(0, 5);
    let z = solver.new_var_range(0, 5);
    let obj = solver.new_var_range(0, 15);
    // obj = x + y + z
    linear(&mut solver, &[1, 1, 1, -1], &[x, y, z, obj], Relation::Eq, 0);
    linear(&mut solver, &[1, 1, 1], &[x, y, z], Relation::Ge, 7);
    let (sol, value) = minimize(&mut solver, &[x, y, z, obj], obj).expect("feasible");
    assert_eq!(value, 7);
    assert_eq!(sol[0] + sol[1] + sol[2], 7); // witness is consistent
    assert_eq!(sol[3], 7);
}

#[test]
fn maximize_knapsack_matches_brute_force() {
    let weights = [3i64, 4, 5, 2];
    let profits = [4i64, 5, 6, 3];
    let capacity = 9i64;

    let mut solver = Solver::new();
    let items: Vec<VarId> = (0..4).map(|_| solver.new_var_range(0, 1)).collect();
    let total: i64 = profits.iter().sum();
    let profit = solver.new_var_range(0, total as i32);
    linear(&mut solver, &weights, &items, Relation::Le, capacity);
    // profit = Σ profits[i] * items[i]
    let mut coeffs = profits.to_vec();
    coeffs.push(-1);
    let mut pvars = items.clone();
    pvars.push(profit);
    linear(&mut solver, &coeffs, &pvars, Relation::Eq, 0);

    let mut all = items.clone();
    all.push(profit);
    let (_, value) = maximize(&mut solver, &all, profit).expect("feasible");

    // Brute force over the four 0/1 items.
    let domains: Vec<Vec<i32>> = (0..4).map(|_| vec![0, 1]).collect();
    let best = oracle::solutions(&domains, |a| (0..4).map(|i| a[i] as i64 * weights[i]).sum::<i64>() <= capacity)
        .into_iter()
        .map(|a| (0..4).map(|i| a[i] as i64 * profits[i]).sum::<i64>())
        .max()
        .unwrap();
    assert_eq!(value as i64, best);
}

#[test]
fn unsatisfiable_models_have_no_solutions() {
    // 3 pairwise-distinct variables over {0,1}: pigeonhole UNSAT.
    let mut solver = Solver::new();
    let v: Vec<VarId> = (0..3).map(|_| solver.new_var_range(0, 1)).collect();
    for i in 0..3 {
        for j in (i + 1)..3 {
            not_equal_offset(&mut solver, v[i], v[j], 0);
        }
    }
    assert_eq!(count_solutions(&mut solver, &v), 0);
}
