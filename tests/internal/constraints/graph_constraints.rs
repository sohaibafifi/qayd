use crate::constraints::graph::{circuit, circuit_with};
use crate::search::{solve, SearchControl};
use crate::store::Solver;
use std::collections::BTreeSet;

/// Whether `succ` (as a function `i -> succ[i]`) is one Hamiltonian cycle.
fn is_single_cycle(succ: &[usize]) -> bool {
    let n = succ.len();
    let mut seen = vec![false; n];
    let mut cur = 0usize;
    for _ in 0..n {
        if cur >= n || seen[cur] {
            return false;
        }
        seen[cur] = true;
        cur = succ[cur];
    }
    cur == 0 && seen.iter().all(|&s| s)
}

fn permute(perm: &mut [usize], k: usize, out: &mut impl FnMut(&[usize])) {
    if k == perm.len() {
        out(perm);
        return;
    }
    for i in k..perm.len() {
        perm.swap(k, i);
        permute(perm, k + 1, out);
        perm.swap(k, i);
    }
}

/// All `succ` assignments over `0..n` that form a single Hamiltonian cycle.
fn brute_circuits(n: usize) -> BTreeSet<Vec<i32>> {
    let mut out = BTreeSet::new();
    let mut perm: Vec<usize> = (0..n).collect();
    permute(&mut perm, 0, &mut |p| {
        if is_single_cycle(p) {
            out.insert(p.iter().map(|&x| x as i32).collect());
        }
    });
    out
}

fn enumerate(n: usize, cutset: bool) -> BTreeSet<Vec<i32>> {
    let mut solver = Solver::new();
    let succ: Vec<_> = (0..n).map(|_| solver.store.new_var_range(0, n as i32 - 1)).collect();
    circuit_with(&mut solver, &succ, cutset);
    let mut set = BTreeSet::new();
    solve(&mut solver, &succ, |s: &Solver| {
        set.insert(succ.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    set
}

#[test]
fn circuit_enumerates_single_hamiltonian_cycles() {
    for n in 2..=6usize {
        let set = enumerate(n, false);
        let brute = brute_circuits(n);
        assert_eq!(set, brute, "circuit enumerates exactly the single Hamiltonian cycles (n={n})");
        assert_eq!(set.len(), (1..n).product::<usize>(), "there are (n-1)! directed Hamiltonian cycles (n={n})");
    }
}

/// The cut-set variant only adds sound failures, so it must enumerate the
/// exact same solution set as the base propagator. This catches any unsound
/// cut-set explanation (which would prune a real Hamiltonian cycle).
#[test]
fn cutset_circuit_enumerates_same_hamiltonian_cycles() {
    for n in 2..=7usize {
        let base = enumerate(n, false);
        let cut = enumerate(n, true);
        assert_eq!(cut, base, "cut-set circuit enumerates exactly the base solution set (n={n})");
    }
}

/// Enumerate every Hamiltonian circuit of a sparse graph given by adjacency
/// (`adj[i]` = successors allowed for node `i`), under the given mode.
fn enumerate_sparse(adj: &[Vec<i32>], cutset: bool) -> BTreeSet<Vec<i32>> {
    let mut solver = Solver::new();
    let succ: Vec<_> = adj.iter().map(|ns| solver.store.new_var_set(ns)).collect();
    circuit_with(&mut solver, &succ, cutset);
    let mut set = BTreeSet::new();
    solve(&mut solver, &succ, |s: &Solver| {
        set.insert(succ.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    set
}

/// On random sparse directed graphs the cut-set filtering fires often (many
/// subsets become closed), so this is the sharp soundness check: the cut-set
/// propagator must enumerate exactly the same Hamiltonian circuits as the
/// base one — never dropping a real solution via an unsound cut explanation.
#[test]
fn cutset_matches_base_on_random_sparse_graphs() {
    // Deterministic xorshift PRNG: reproducible, no external crates.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for n in 3..=7usize {
        for _ in 0..40 {
            // Random simple digraph; guarantee out/in coverage by seeding a
            // random Hamiltonian cycle so non-trivial instances appear too.
            let mut adj_set: Vec<std::collections::BTreeSet<i32>> = vec![Default::default(); n];
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (next() as usize) % (i + 1);
                perm.swap(i, j);
            }
            for k in 0..n {
                adj_set[perm[k]].insert(perm[(k + 1) % n] as i32);
            }
            for (i, set) in adj_set.iter_mut().enumerate() {
                for j in 0..n {
                    if i != j && next() % 3 == 0 {
                        set.insert(j as i32);
                    }
                }
            }
            let adj: Vec<Vec<i32>> = adj_set.iter().map(|s| s.iter().copied().collect()).collect();
            let base = enumerate_sparse(&adj, false);
            let cut = enumerate_sparse(&adj, true);
            assert_eq!(cut, base, "cut-set matches base on random sparse graph n={n} adj={adj:?}");
            assert!(!base.is_empty(), "seeded cycle guarantees a solution n={n}");
        }
    }
}

/// `circuit()` auto-enables cut-set on sparse graphs; that dispatch must not
/// change the solution set versus the base propagator.
#[test]
fn auto_gated_circuit_is_sound_on_sparse_graph() {
    // Sparse ring-plus-chords over 0..8: average degree well below n/4? no —
    // pick a genuinely sparse graph so the density gate turns cut-set on.
    let n = 12usize;
    let adj: Vec<Vec<i32>> = (0..n)
        .map(|i| vec![((i + 1) % n) as i32, ((i + n - 1) % n) as i32, ((i + 5) % n) as i32])
        .map(|mut v| {
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();

    let mut solver = Solver::new();
    let succ: Vec<_> = adj.iter().map(|ns| solver.store.new_var_set(ns)).collect();
    circuit(&mut solver, &succ); // auto-gated entry
    let mut auto = BTreeSet::new();
    solve(&mut solver, &succ, |s: &Solver| {
        auto.insert(succ.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    assert_eq!(auto, enumerate_sparse(&adj, false), "auto-gated circuit matches base solution set");
}
