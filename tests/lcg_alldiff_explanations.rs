//! Tight-explanation checks for the Régin allDifferent propagator: Hall-set
//! conflict budgets under CDCL, randomized holey oracle equivalence (both the
//! chronological enumerator and the learning branch-and-bound driver), and an
//! ignored metrics sweep comparing Hall-set reasons against the whole-scope
//! fallback.

use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::primitives::all_different;
mod common;

use common::oracle;
use qayd::{count_solutions, optimize_with, solve_search, SearchControl, SolveStats, Solver, VarId};

static NEVER_STOP: AtomicBool = AtomicBool::new(false);

/// Permutation of `0..n-1` whose sum must exceed the invariant n(n-1)/2:
/// UNSAT, but only provable at depth, where allDifferent failures (and hence
/// their explanations) drive the learning. Solved with the CDCL
/// branch-and-bound driver so learned clauses actually bite.
fn perm_sum_stats(n: i32) -> SolveStats {
    perm_sum_stats_with_scope(n, false)
}

fn perm_sum_stats_with_scope(n: i32, force_scope_reasons: bool) -> SolveStats {
    let mut solver = Solver::new();
    solver.set_force_scope_reasons(force_scope_reasons);
    let vars: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
    all_different(&mut solver, &vars);
    let coeffs = vec![1i64; n as usize];
    linear(&mut solver, &coeffs, &vars, Relation::Ge, i64::from(n * (n - 1) / 2) + 1);
    let (best, stats) = optimize_with(&mut solver, &vars, vars[0], true, &NEVER_STOP, |_| {});
    assert!(best.is_none(), "perm-sum({n}) must be UNSAT");
    stats
}

#[test]
fn php_is_unsat() {
    // PHP(n): n+1 pigeons into n holes. The single global allDifferent fails
    // at the root (pigeonhole check), so this is a soundness assert, not a
    // learning benchmark.
    for n in [6, 9, 12] {
        let mut solver = Solver::new();
        let vars: Vec<VarId> = (0..=n).map(|_| solver.new_var_range(0, n - 1)).collect();
        all_different(&mut solver, &vars);
        assert_eq!(count_solutions(&mut solver, &vars), 0, "PHP({n}) must be UNSAT");
    }
}

#[test]
fn perm_sum_learned_width_budget() {
    // Hall-set explanations shrink the average learned clause: perm-sum(8)
    // measures 13.0 lits/clause with them and 18.2 with the whole-scope
    // whole-scope fallback. The budget sits between the two, so a
    // regression to scope-width reasons busts it. Conflict count is roughly
    // par on this family (the win is width and time), hence only a loose
    // sanity ceiling on failures. Both budgets are calibrated on the current
    // search heuristics (seed 0); if a heuristic change moves the measured
    // width, recalibrate against a fresh whole-scope run rather than
    // deleting the check.
    let stats = perm_sum_stats(8);
    let width = stats.learned_lits as f64 / stats.failures.max(1) as f64;
    assert!(width <= 16.0, "perm-sum(8) learned width {width:.2}; Hall-set explanations should keep this under 16");
    assert!(stats.failures <= 60_000, "perm-sum(8) took {} conflicts", stats.failures);
}

/// Tiny deterministic PRNG (xorshift64*) so the holey instances are stable.
fn next(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Random holey subset of `0..universe` with at least `min_size` values.
fn random_set(state: &mut u64, universe: i32, min_size: usize) -> Vec<i32> {
    loop {
        let set: Vec<i32> = (0..universe).filter(|_| next(state).is_multiple_of(2)).collect();
        if set.len() >= min_size {
            return set;
        }
    }
}

#[test]
fn randomized_holey_all_different_matches_oracle() {
    // Holey explicit-set domains seed absent-value eq atoms as root facts, so
    // an explanation that forgets a hole shows up as a wrong solution set
    // (chronological enumeration) or a wrong optimum / spurious UNSAT
    // (learning branch-and-bound, which resolves over the explanations).
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for round in 0..20 {
        let n = 4 + (next(&mut state) % 4) as usize; // 4..=7 vars
        let universe = n as i32 + 1; // tight universe: UNSAT rounds included
        let sets: Vec<Vec<i32>> = (0..n).map(|_| random_set(&mut state, universe, 2)).collect();

        let want: BTreeSet<Vec<i32>> = oracle::solutions(&sets, |a| {
            let mut seen = BTreeSet::new();
            a.iter().all(|&x| seen.insert(x))
        })
        .into_iter()
        .collect();

        let mut solver = Solver::new();
        let vars: Vec<VarId> = sets.iter().map(|s| solver.new_var_set(s)).collect();
        all_different(&mut solver, &vars);
        let mut got = BTreeSet::new();
        solve_search(&mut solver, &vars, |s| {
            got.insert(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
            SearchControl::Continue
        });
        assert_eq!(got, want, "round {round}, sets {sets:?}");

        let mut solver = Solver::new();
        let vars: Vec<VarId> = sets.iter().map(|s| solver.new_var_set(s)).collect();
        all_different(&mut solver, &vars);
        let (best, _) = optimize_with(&mut solver, &vars, vars[0], true, &NEVER_STOP, |_| {});
        let brute_min = want.iter().map(|a| a[0]).min();
        assert_eq!(best.map(|(_, v)| v), brute_min, "round {round}, sets {sets:?}");
    }
}

/// Metrics sweep. Run with:
/// `cargo test --release --test lcg_alldiff_explanations -- --ignored --nocapture`
#[test]
#[ignore]
fn alldiff_metrics_sweep() {
    for scope in [false, true] {
        eprintln!("family,n,failures,learned_lits_per_failure,nodes,ms scope_reasons={scope}");
        for n in [7, 8, 9] {
            let t = std::time::Instant::now();
            let stats = perm_sum_stats_with_scope(n, scope);
            let width = stats.learned_lits as f64 / stats.failures.max(1) as f64;
            eprintln!("permsum,{n},{},{width:.2},{},{:.1}", stats.failures, stats.nodes, t.elapsed().as_secs_f64() * 1e3);
        }
    }
}
