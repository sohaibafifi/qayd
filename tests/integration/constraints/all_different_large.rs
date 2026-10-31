//! Stack-safety + scale for the Régin allDifferent propagator. Its matching and
//! SCC passes are iterative, so a large instance whose SCC graph is one deep
//! chain completes where a recursive DFS would overflow the test thread's stack.

use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::primitives::all_different;
use qayd::{count_solutions, Solver, VarId};

#[test]
fn large_all_different_completes() {
    // Sparse cyclic domains: var i in {i, (i+1) mod n}. allDifferent is
    // satisfiable (both the identity and the cyclic shift are perfect matchings),
    // so nothing is pruned. One root propagation builds the matching and runs a
    // single Tarjan SCC pass over ~2n chained nodes — deep enough to overflow a
    // recursive DFS on a test thread's stack; the iterative version completes.
    let n = 20_000i32;
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..n)
        .map(|i| {
            let b = (i + 1) % n;
            s.new_var_set(&[i.min(b), i.max(b)])
        })
        .collect();
    all_different(&mut s, &vars);

    s.store.push_level();
    s.enqueue_all();
    s.propagate().expect("cyclic allDifferent is satisfiable");
    // Both domain values remain supported for every variable.
    for &v in &vars {
        assert_eq!(s.store.size(v), 2);
    }
    s.store.pop_level();
}

/// Timing harness (ignored by default): enumerate all solutions of N-Queens
/// modelled with three allDifferent constraints. Exercises many propagate()
/// calls under backtracking, where warm-start + scratch reuse pays off.
/// Run with: cargo test --test all_different_large -- --ignored --nocapture
#[test]
#[ignore]
fn bench_queens_alldiff_enumeration() {
    let n = 11i32;
    let mut solver = Solver::new();
    let q: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
    let up: Vec<VarId> = (0..n).map(|i| solver.new_var_range(i, i + n - 1)).collect();
    let down: Vec<VarId> = (0..n).map(|i| solver.new_var_range(-i, n - 1 - i)).collect();
    // up[i] = q[i] + i, down[i] = q[i] - i.
    for i in 0..n {
        let iu = i as usize;
        linear(&mut solver, &[1, -1], &[q[iu], up[iu]], Relation::Eq, -(i as i64));
        linear(&mut solver, &[1, -1], &[q[iu], down[iu]], Relation::Eq, i as i64);
    }
    all_different(&mut solver, &q);
    all_different(&mut solver, &up);
    all_different(&mut solver, &down);

    let t = std::time::Instant::now();
    let count = count_solutions(&mut solver, &q);
    eprintln!("bench_queens_alldiff n={n}: {count} solutions in {:?}", t.elapsed());
}
