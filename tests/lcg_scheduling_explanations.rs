//! Tight-explanation checks for the integer scheduling stack (`scheduling.rs`):
//! the time-table `cumulative` overload/push explanations and the pairwise
//! `noOverlap` bound explanations. Correctness gates (holey-domain oracle
//! equivalence for both the chronological enumerator and the learning
//! branch-and-bound driver, plus a root-overload soundness check) guard against
//! a dropped premise; a calibrated learned-width budget and an ignored metrics
//! sweep compare tight reasons against the whole-scope fallback
//! selected explicitly on each solver.

use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::scheduling::{cumulative, no_overlap};
mod common;

use common::oracle;
use qayd::{count_solutions, optimize_with, solve_search, SearchControl, SolveStats, Solver, VarId};

static NEVER_STOP: AtomicBool = AtomicBool::new(false);

/// Minimum sum-of-starts of a feasible schedule of `n` (even) unit-height,
/// duration-2 tasks on a capacity-2 resource: two lanes, each holding `n/2`
/// back-to-back tasks at `0, 2, .., n-2`.
fn min_start_sum(n: i32) -> i64 {
    i64::from((n / 2) * (n - 2))
}

/// RCPSP-like learning family: `n` (even) tasks, capacity 2, unit heights,
/// duration 2. Forcing the total start time below the feasible minimum clusters
/// the tasks and overloads the resource, but only after the brancher commits
/// enough starts that the time-table profile peaks — the analogue of perm-sum
/// (infeasible by a global count the propagator sees only at depth). The
/// energetic area bound holds at the root (energy `2n` fits in `2 * horizon`),
/// so the deep failures are driven by time-tabling. Minimised with the CDCL
/// branch-and-bound driver so learned clauses bite.
fn rcpsp_tight_stats(n: i32) -> SolveStats {
    rcpsp_tight_stats_with_scope(n, false)
}

fn rcpsp_tight_stats_with_scope(n: i32, force_scope_reasons: bool) -> SolveStats {
    assert!(n % 2 == 0, "family expects even n");
    let mut solver = Solver::new();
    solver.set_force_scope_reasons(force_scope_reasons);
    let starts: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
    let dur = vec![2i64; n as usize];
    let h = vec![1i64; n as usize];
    cumulative(&mut solver, &starts, &dur, &h, 2);
    let coeffs = vec![1i64; n as usize];
    linear(&mut solver, &coeffs, &starts, Relation::Le, min_start_sum(n) - 1);
    let (best, stats) = optimize_with(&mut solver, &starts, starts[0], true, &NEVER_STOP, |_| {});
    assert!(best.is_none(), "rcpsp-tight({n}) must be UNSAT");
    stats
}

#[test]
fn cumulative_root_overload_unsat() {
    // Three unit-duration tasks all fixed to t=0 with total height 3 > capacity 2:
    // the time-table profile overloads at the root, so this is a soundness assert.
    let mut s = Solver::new();
    let starts = [s.new_var_set(&[0]), s.new_var_set(&[0]), s.new_var_set(&[0])];
    cumulative(&mut s, &starts, &[1, 1, 1], &[1, 1, 1], 2);
    assert_eq!(count_solutions(&mut s, &starts), 0, "root overload (3 > 2) must be UNSAT");
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
fn randomized_holey_cumulative_matches_oracle() {
    // Holey start domains seed absent-value eq atoms as root facts, so an
    // explanation that forgets a covering task shows up as a wrong solution set
    // (chronological enumeration) or a wrong optimum / spurious UNSAT (the
    // learning branch-and-bound, which resolves over the explanations).
    let mut state = 0xC0FF_EE12_3456_789Au64;
    for round in 0..40 {
        let n = 3 + (next(&mut state) % 3) as usize; // 3..=5 tasks
        let cap = 2 + (next(&mut state) % 2) as i64; // 2..=3
        let durs: Vec<i64> = (0..n).map(|_| 1 + (next(&mut state) % 3) as i64).collect();
        let heights: Vec<i64> = (0..n).map(|_| 1 + (next(&mut state) % cap as u64) as i64).collect();
        let sets: Vec<Vec<i32>> = (0..n).map(|_| random_set(&mut state, 4, 2)).collect();

        let feasible = |a: &[i32]| {
            let end = (0..n).map(|i| a[i] as i64 + durs[i]).max().unwrap();
            (0..end).all(|t| {
                let used: i64 = (0..n).filter(|&i| a[i] as i64 <= t && t < a[i] as i64 + durs[i]).map(|i| heights[i]).sum();
                used <= cap
            })
        };
        let want: BTreeSet<Vec<i32>> = oracle::solutions(&sets, feasible).into_iter().collect();

        let mut solver = Solver::new();
        let vars: Vec<VarId> = sets.iter().map(|s| solver.new_var_set(s)).collect();
        cumulative(&mut solver, &vars, &durs, &heights, cap);
        let mut got = BTreeSet::new();
        solve_search(&mut solver, &vars, |s| {
            got.insert(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
            SearchControl::Continue
        });
        assert_eq!(got, want, "enum round {round}, sets {sets:?} dur {durs:?} h {heights:?} cap {cap}");

        let mut solver = Solver::new();
        let vars: Vec<VarId> = sets.iter().map(|s| solver.new_var_set(s)).collect();
        cumulative(&mut solver, &vars, &durs, &heights, cap);
        let (best, _) = optimize_with(&mut solver, &vars, vars[0], true, &NEVER_STOP, |_| {});
        let brute_min = want.iter().map(|a| a[0]).min();
        assert_eq!(best.map(|(_, v)| v), brute_min, "opt round {round}, sets {sets:?} dur {durs:?} h {heights:?} cap {cap}");
    }
}

#[test]
fn randomized_holey_no_overlap_matches_oracle() {
    // Same tripwire for the pairwise disjunctive noOverlap: its four-bound
    // explanations must not drop a feasible ordering.
    let mut state = 0x1357_9BDF_2468_ACE0u64;
    for round in 0..40 {
        let n = 3 + (next(&mut state) % 3) as usize; // 3..=5 tasks
        let durs: Vec<i64> = (0..n).map(|_| 1 + (next(&mut state) % 3) as i64).collect();
        let sets: Vec<Vec<i32>> = (0..n).map(|_| random_set(&mut state, 5, 2)).collect();

        let feasible = |a: &[i32]| {
            (0..n).all(|i| {
                ((i + 1)..n).all(|j| {
                    let (si, di) = (a[i] as i64, durs[i]);
                    let (sj, dj) = (a[j] as i64, durs[j]);
                    si + di <= sj || sj + dj <= si
                })
            })
        };
        let want: BTreeSet<Vec<i32>> = oracle::solutions(&sets, feasible).into_iter().collect();

        let mut solver = Solver::new();
        let vars: Vec<VarId> = sets.iter().map(|s| solver.new_var_set(s)).collect();
        no_overlap(&mut solver, &vars, &durs);
        let mut got = BTreeSet::new();
        solve_search(&mut solver, &vars, |s| {
            got.insert(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
            SearchControl::Continue
        });
        assert_eq!(got, want, "enum round {round}, sets {sets:?} dur {durs:?}");

        let mut solver = Solver::new();
        let vars: Vec<VarId> = sets.iter().map(|s| solver.new_var_set(s)).collect();
        no_overlap(&mut solver, &vars, &durs);
        let (best, _) = optimize_with(&mut solver, &vars, vars[0], true, &NEVER_STOP, |_| {});
        let brute_min = want.iter().map(|a| a[0]).min();
        assert_eq!(best.map(|(_, v)| v), brute_min, "opt round {round}, sets {sets:?} dur {durs:?}");
    }
}

#[test]
fn rcpsp_tight_unsat_under_tight_reasons() {
    // Soundness under the tight time-table reasons: this family is UNSAT and the
    // learning branch-and-bound must prove it (a dropped premise would resolve to
    // a spurious solution / wrong optimum and break `rcpsp_tight_stats`'s
    // `best.is_none()`). On THIS family tight ≈ scope width — the covering set at
    // a time-table failure is nearly all tasks, and edge-finding (which stays on
    // scope) fires before time-tabling, so most reasons demote. Measured at n=8:
    // 7.48 lits/failure tight vs 7.36 with whole-scope reasons (no drop); the
    // real width win only emerges at depth (n=10: 15.98 tight vs 19.96 scope,
    // but ~48s — too slow to gate). Per the CT-wave precedent we therefore assert
    // only equivalence (the randomized oracle tests) plus this soundness + loose
    // sanity ceiling, not a tight-vs-scope width drop. Recalibrate the ceiling
    // against a fresh sweep rather than deleting it.
    let stats = rcpsp_tight_stats(8);
    let width = stats.learned_lits as f64 / stats.failures.max(1) as f64;
    assert!(width <= 12.0, "rcpsp-tight(8) learned width {width:.2} blew past the sanity ceiling (measured ~7.4)");
    assert!(stats.failures <= 40_000, "rcpsp-tight(8) took {} conflicts (measured ~12.9k)", stats.failures);
}

/// Metrics sweep. Run with:
/// `cargo test --release --test lcg_scheduling_explanations -- --ignored --nocapture`
#[test]
#[ignore]
fn scheduling_metrics_sweep() {
    for scope in [false, true] {
        eprintln!("family,n,failures,learned_lits_per_failure,nodes,ms scope_reasons={scope}");
        for n in [6, 8, 10] {
            let t = std::time::Instant::now();
            let stats = rcpsp_tight_stats_with_scope(n, scope);
            let width = stats.learned_lits as f64 / stats.failures.max(1) as f64;
            eprintln!("rcpsp_tight,{n},{},{width:.2},{},{:.1}", stats.failures, stats.nodes, t.elapsed().as_secs_f64() * 1e3);
        }
    }
}
