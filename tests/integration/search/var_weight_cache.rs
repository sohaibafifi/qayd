//! The incremental dom/wdeg cache read by the brancher must stay bit-identical
//! to the naive per-subscription rescan across a full search (many weight
//! bumps), so variable selection is unchanged.

use qayd::constraints::primitives::not_equal_offset;
use qayd::{count_solutions, Solver, VarId};
use std::time::Instant;

fn build_queens(solver: &mut Solver, n: i32) -> Vec<VarId> {
    let q: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
    for i in 0..n as usize {
        for j in (i + 1)..n as usize {
            let (di, dj) = (i as i32, j as i32);
            not_equal_offset(solver, q[i], q[j], 0);
            not_equal_offset(solver, q[i], q[j], di - dj);
            not_equal_offset(solver, q[i], q[j], dj - di);
        }
    }
    q
}

#[test]
fn cached_weight_matches_naive_after_search() {
    let mut solver = Solver::new();
    let q = build_queens(&mut solver, 8);
    // Drive a real search so propagators fail and bump dom/wdeg weights.
    let _ = count_solutions(&mut solver, &q);
    let weights = solver.weights().to_vec();
    // The test is only meaningful if bumps actually happened.
    assert!(weights.iter().any(|&w| w > 1), "expected dom/wdeg bumps during search");
    for i in 0..solver.store.num_vars() {
        let v = VarId(i as u32);
        assert_eq!(
            solver.store.var_weight_cached(v),
            solver.store.var_weight(v, &weights),
            "cached weighted degree diverged from naive rescan for var {i}",
        );
    }
}

/// Micro-bench: per-node cost of the branching weighted-degree lookup over all
/// vars, naive rescan vs the O(1) cache, at growing sizes. Run with
/// `cargo test --release --test var_weight_cache bench -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_select_var_weight_cost() {
    for n in [50i32, 100, 200, 400] {
        let mut solver = Solver::new();
        let q = build_queens(&mut solver, n);
        let weights = solver.weights().to_vec();
        let iters = 2000u32;
        // Naive: what select_var used to do every node (sum over subscriptions).
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            for &v in &q {
                acc += solver.store.var_weight(v, &weights);
            }
        }
        let naive = t0.elapsed();
        // Cached: the O(1) read select_var does now.
        let t1 = Instant::now();
        let mut acc2 = 0u64;
        for _ in 0..iters {
            for &v in &q {
                acc2 += solver.store.var_weight_cached(v);
            }
        }
        let cached = t1.elapsed();
        assert_eq!(acc, acc2);
        let incidence = solver.store.var_weight(q[0], &weights);
        println!(
            "n={n:3} vars={:4} incidence~{incidence:4} | naive {naive:>10.2?}  cached {cached:>10.2?}  speedup {:.1}x",
            q.len(),
            naive.as_secs_f64() / cached.as_secs_f64().max(1e-9),
        );
    }
}
