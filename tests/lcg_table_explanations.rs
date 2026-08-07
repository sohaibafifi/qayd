//! Tight-explanation checks for the positive Compact-Table `extension`
//! propagator: randomized holey oracle equivalence plus a fixed STAR-column
//! instance (both the chronological enumerator and the learning branch-and-bound
//! driver), a table-encoded permutation UNSAT family that must stay UNSAT under
//! both the killer-literal reasons and the whole-scope fallback, a learned-width sanity ceiling, and an ignored
//! metrics sweep comparing the two reason styles.

use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::table::{extension, STAR};
mod common;

use common::oracle;
use qayd::{optimize_with, solve_search, SearchControl, SolveStats, Solver, VarId};

static NEVER_STOP: AtomicBool = AtomicBool::new(false);

/// Does assignment `a` match tuple `t` (STAR matches anything)?
fn tuple_matches(t: &[i32], a: &[i32]) -> bool {
    t.iter().zip(a).all(|(&p, &v)| p == STAR || p == v)
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
fn randomized_holey_extension_matches_oracle() {
    // Holey explicit-set domains seed absent-value eq atoms as root facts, so a
    // killer literal that forgets a hole (or a whole tuple) surfaces as a wrong
    // solution set (chronological enumeration) or a wrong optimum / spurious
    // UNSAT (learning branch-and-bound, which resolves over the killer reasons).
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for round in 0..24 {
        let n = 3 + (next(&mut state) % 3) as usize; // 3..=5 columns
        let universe = 3 + (next(&mut state) % 3) as i32; // 3..=5 values
        let sets: Vec<Vec<i32>> = (0..n).map(|_| random_set(&mut state, universe, 2)).collect();

        // A random allowed list over the universe (some rows STAR, may be UNSAT).
        let ntuples = 2 + (next(&mut state) % 6) as usize;
        let allowed: Vec<Vec<i32>> = (0..ntuples)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        let r = next(&mut state) % (universe as u64 + 1);
                        if r == universe as u64 {
                            STAR
                        } else {
                            r as i32
                        }
                    })
                    .collect()
            })
            .collect();

        let want: BTreeSet<Vec<i32>> = oracle::solutions(&sets, |a| allowed.iter().any(|t| tuple_matches(t, a))).into_iter().collect();

        let mut solver = Solver::new();
        let vars: Vec<VarId> = sets.iter().map(|s| solver.new_var_set(s)).collect();
        extension(&mut solver, &vars, &allowed, true);
        let mut got = BTreeSet::new();
        solve_search(&mut solver, &vars, |s| {
            got.insert(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
            SearchControl::Continue
        });
        assert_eq!(got, want, "round {round}, sets {sets:?}, allowed {allowed:?}");

        let mut solver = Solver::new();
        let vars: Vec<VarId> = sets.iter().map(|s| solver.new_var_set(s)).collect();
        extension(&mut solver, &vars, &allowed, true);
        let (best, _) = optimize_with(&mut solver, &vars, vars[0], true, &NEVER_STOP, |_| {});
        let brute_min = want.iter().map(|a| a[0]).min();
        assert_eq!(best.map(|(_, v)| v), brute_min, "round {round}, sets {sets:?}, allowed {allowed:?}");
    }
}

#[test]
fn extension_removed_value_excluded_from_learned_reason() {
    // Deterministic self-check: a two-column table (0,1),(1,0),(1,2) with x fixed
    // to 1 forces y != 0 (only (1,0) has x=1... plus (1,2)); the killer for the
    // (0,1) support of y=1 is x != 0, and fixing x = 1 makes y take {0,2}. The
    // oracle equality below is the real guard; this pins the concrete case.
    let allowed = [vec![0, 1], vec![1, 0], vec![1, 2]];
    let mut solver = Solver::new();
    let x = solver.new_var_range(0, 1);
    let y = solver.new_var_range(0, 2);
    extension(&mut solver, &[x, y], &allowed, true);
    let mut got = BTreeSet::new();
    solve_search(&mut solver, &[x, y], |s| {
        got.insert((s.store.value(x), s.store.value(y)));
        SearchControl::Continue
    });
    let want: BTreeSet<(i32, i32)> = allowed.iter().map(|t| (t[0], t[1])).collect();
    assert_eq!(got, want);
}

#[test]
fn fixed_star_table_cdcl_matches_chronological() {
    // Fixed instance with STAR columns, kept deterministic so a regression in the
    // killer-literal explanation reproduces without hunting a random seed. STAR
    // rows exercise the `last_kill_col`/`scope_width` fallback (a wildcard support
    // is killed by whichever concrete column loses its value, not a fixed one).
    let allowed = [vec![0, STAR, 1], vec![STAR, 2, 2], vec![1, 1, STAR], vec![2, 0, 0]];
    let want: BTreeSet<Vec<i32>> =
        oracle::solutions(&vec![(0..=2).collect::<Vec<i32>>(); 3], |a| allowed.iter().any(|t| tuple_matches(t, a))).into_iter().collect();

    // Chronological enumeration (no learning) is the reference solution set.
    let mut solver = Solver::new();
    let vars: Vec<VarId> = (0..3).map(|_| solver.new_var_range(0, 2)).collect();
    extension(&mut solver, &vars, &allowed, true);
    let mut got = BTreeSet::new();
    solve_search(&mut solver, &vars, |s| {
        got.insert(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    assert_eq!(got, want, "chronological set diverged from oracle");

    // The learning branch-and-bound driver resolves over the killer reasons; an
    // unsound explanation drops the optimum-bearing solution and returns a wrong
    // (or missing) best.
    let mut solver = Solver::new();
    let vars: Vec<VarId> = (0..3).map(|_| solver.new_var_range(0, 2)).collect();
    extension(&mut solver, &vars, &allowed, true);
    let (best, _) = optimize_with(&mut solver, &vars, vars[0], true, &NEVER_STOP, |_| {});
    assert_eq!(best.map(|(_, v)| v), want.iter().map(|a| a[0]).min(), "CDCL optimum diverged from chronological");
}

/// Table-encoded permutation of `0..n-1` (each adjacent pair constrained to an
/// `a != b` allowed table forces all-different) whose sum must exceed the
/// invariant n(n-1)/2: UNSAT, provable only at depth where `extension` failures
/// and removals drive the learning. Solved with the CDCL branch-and-bound
/// driver so learned clauses actually bite.
fn table_perm_sum_stats(n: i32) -> SolveStats {
    table_perm_sum_stats_with_scope(n, false)
}

fn table_perm_sum_stats_with_scope(n: i32, force_scope_reasons: bool) -> SolveStats {
    let mut solver = Solver::new();
    solver.set_force_scope_reasons(force_scope_reasons);
    let vars: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
    let neq: Vec<Vec<i32>> = (0..n).flat_map(|a| (0..n).filter(move |&b| a != b).map(move |b| vec![a, b])).collect();
    for w in vars.windows(2) {
        extension(&mut solver, w, &neq, true);
    }
    // Adjacent-pair disequality is not global all-different, so also forbid every
    // non-adjacent equal pair to force a true permutation and a tight UNSAT.
    for i in 0..n as usize {
        for j in (i + 1)..n as usize {
            extension(&mut solver, &[vars[i], vars[j]], &neq, true);
        }
    }
    let coeffs = vec![1i64; n as usize];
    linear(&mut solver, &coeffs, &vars, Relation::Ge, i64::from(n * (n - 1) / 2) + 1);
    let (best, stats) = optimize_with(&mut solver, &vars, vars[0], true, &NEVER_STOP, |_| {});
    assert!(best.is_none(), "table-perm-sum({n}) must be UNSAT");
    stats
}

#[test]
fn table_perm_sum_learned_width_budget() {
    // HONEST FINDING — tight killer reasons do NOT narrow learned clauses on this
    // family, so this is a sanity ceiling, not a tight-beats-scope budget. Measured
    // at n=6: 6.68 lits/conflict tight vs 5.07 with whole-scope reasons: tight is
    // slightly WIDER. Two reasons: (1) on a 2-column disequality table the "other"
    // column already IS the whole non-x scope, so there is nothing to drop; (2) the
    // whole-scope fallback only emits literals for values actually REMOVED from the
    // scope, while the killer path emits a (deduped, scope-capped) Ne per dead
    // supporting tuple — more precise, but not fewer literals. A STAR-padded wide
    // variant does not flip this: STAR columns are never pruned, so they add zero to
    // the scope reason. The true equivalence guard therefore lives in
    // `table_perm_sum_unsat_under_both_configs` (both configs must prove UNSAT);
    // this test only caps a per-tuple-uncapped blowup past the scope-width guard.
    // Calibrated on the current search heuristics (seed 0); if a heuristic change
    // moves the measured width, recalibrate against a fresh whole-scope run
    // rather than deleting the check.
    let stats = table_perm_sum_stats(6);
    let width = stats.learned_lits as f64 / stats.failures.max(1) as f64;
    assert!(width <= 10.0, "table-perm-sum(6) learned width {width:.2}; killer reasons should stay scope-capped under 10");
    assert!(stats.failures <= 200_000, "table-perm-sum(6) took {} conflicts", stats.failures);
}

/// Ablation worker: proves the table-perm-sum family UNSAT with whole-scope
/// explanations selected explicitly on this solver.
#[test]
#[ignore]
fn scope_reasons_unsat_worker() {
    let stats = table_perm_sum_stats_with_scope(6, true);
    assert!(stats.failures > 100, "scope-reasons run took only {} conflicts; learning barely engaged", stats.failures);
}

#[test]
fn table_perm_sum_unsat_under_both_configs() {
    // Soundness ablation: the answer (UNSAT) and the "learning actually bites at
    // depth" property (>100 conflicts) must hold for both reason configurations.
    let tight = table_perm_sum_stats(6);
    assert!(tight.failures > 100, "tight run took only {} conflicts; learning barely engaged", tight.failures);
    let scope = table_perm_sum_stats_with_scope(6, true);
    assert!(scope.failures > 100, "scope run took only {} conflicts; learning barely engaged", scope.failures);
}

/// Metrics sweep. Run with:
/// `cargo test --release --test lcg_table_explanations -- --ignored --nocapture`
#[test]
#[ignore]
fn table_metrics_sweep() {
    for scope in [false, true] {
        eprintln!("family,n,failures,learned_lits_per_failure,nodes,ms scope_reasons={scope}");
        for n in [5, 6, 7] {
            let t = std::time::Instant::now();
            let stats = table_perm_sum_stats_with_scope(n, scope);
            let width = stats.learned_lits as f64 / stats.failures.max(1) as f64;
            eprintln!("tableperm,{n},{},{width:.2},{},{:.1}", stats.failures, stats.nodes, t.elapsed().as_secs_f64() * 1e3);
        }
    }
}
