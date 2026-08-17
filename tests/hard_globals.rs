//! Oracle cross-checks for the Phase 4 hard globals (weak-but-correct):
//! allDifferent, circuit, noOverlap, cumulative, binPacking, knapsack.

use std::collections::BTreeSet;

use qayd::constraints::graph::circuit;
use qayd::constraints::linear::Relation;
use qayd::constraints::primitives::all_different;
use qayd::constraints::scheduling::{bin_packing, cumulative, cumulative_var, knapsack, no_overlap};
use qayd::model::{Constraint, IntGlobalConstraint, Model};
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

#[test]
fn all_different_is_permutations() {
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 3)).collect();
    all_different(&mut s, &v);
    let got = enumerate(&mut s, &v);
    let want = oracle_set(4, 0, 3, |a| (0..a.len()).all(|i| ((i + 1)..a.len()).all(|j| a[i] != a[j])));
    assert_eq!(got, want);
    assert_eq!(got.len(), 24);
}

#[test]
fn all_different_is_domain_consistent() {
    // Hall set {x1, x2} = {1, 2} forces x3 = 3 at the root — Régin prunes this
    // immediately, where forward-checking would wait for x1/x2 to be fixed.
    let mut s = Solver::new();
    let x1 = s.new_var_range(1, 2);
    let x2 = s.new_var_range(1, 2);
    let x3 = s.new_var_range(1, 3);
    all_different(&mut s, &[x1, x2, x3]);
    s.store.push_level();
    s.enqueue_all();
    s.propagate().unwrap();
    assert_eq!(s.store.value(x3), 3);
    s.store.pop_level();
}

fn is_circuit(a: &[i32]) -> bool {
    let n = a.len();
    let mut seen = vec![false; n];
    for &x in a {
        if x < 0 || x as usize >= n || seen[x as usize] {
            return false;
        }
        seen[x as usize] = true;
    }
    if n > 1 && (0..n).any(|i| a[i] == i as i32) {
        return false;
    }
    let mut cur = 0usize;
    let mut steps = 0;
    loop {
        cur = a[cur] as usize;
        steps += 1;
        if cur == 0 || steps > n {
            break;
        }
    }
    steps == n
}

#[test]
fn circuit_counts_hamiltonian_cycles() {
    let n = 4;
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..n).map(|_| s.new_var_range(0, (n - 1) as i32)).collect();
    circuit(&mut s, &v);
    let got = enumerate(&mut s, &v);
    let want = oracle_set(n, 0, (n - 1) as i32, is_circuit);
    assert_eq!(got, want);
    assert_eq!(got.len(), 6); // (n-1)! Hamiltonian circuits
}

#[test]
fn circuit_rejects_duplicate_incoming_arc_fixed_during_propagation() {
    // Fixed arcs 1->2 and 3->0 prune 0->1 and 2->1. The duplicate incoming
    // arc to 1 is created during the same propagator call and must be rejected.
    let mut s = Solver::new();
    let succ = [s.new_var_set(&[1, 2]), s.new_var_set(&[2]), s.new_var_set(&[0, 1]), s.new_var_set(&[0])];
    circuit(&mut s, &succ);
    assert!(enumerate(&mut s, &succ).is_empty());
}

#[test]
fn no_overlap_matches_oracle() {
    let durations = [2i64, 1, 2];
    let mut s = Solver::new();
    let starts: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 4)).collect();
    no_overlap(&mut s, &starts, &durations);
    let got = enumerate(&mut s, &starts);

    let want = oracle_set(3, 0, 4, |a| {
        (0..3).all(|i| {
            ((i + 1)..3).all(|j| {
                let (si, di) = (a[i] as i64, durations[i]);
                let (sj, dj) = (a[j] as i64, durations[j]);
                si + di <= sj || sj + dj <= si
            })
        })
    });
    assert_eq!(got, want);
}

#[test]
fn no_overlap_rejects_overlap_created_during_propagation() {
    // Task 2 fixed at 1 forces tasks 0 and 1 to start at 0, making them overlap.
    let mut s = Solver::new();
    let starts = [s.new_var_set(&[0, 1]), s.new_var_set(&[0, 1]), s.new_var_set(&[1])];
    no_overlap(&mut s, &starts, &[1, 1, 1]);
    assert!(enumerate(&mut s, &starts).is_empty());
}

#[test]
fn no_overlap_removes_pairwise_unsupported_interior_starts() {
    let mut solver = Solver::new();
    // Decomposition work uses cardinality, not the million-wide numeric span.
    let first = solver.new_var_set(&[0, 2, 1_000_000]);
    let second = solver.new_var_set(&[1, 3]);
    no_overlap(&mut solver, &[first, second], &[2, 2]);
    solver.enqueue_all();
    solver.propagate().unwrap();

    assert!(solver.store.contains(first, 0));
    assert!(!solver.store.contains(first, 2));
    assert!(solver.store.contains(first, 1_000_000));
}

#[test]
fn no_overlap_posts_one_global_resource_independent_of_domain_size() {
    let mut small = Solver::new();
    let small_starts = (0..4).map(|_| small.new_var_range(0, 1)).collect::<Vec<_>>();
    no_overlap(&mut small, &small_starts, &[1; 4]);
    assert_eq!(small.num_propagators(), 1);
    assert_eq!(small.store.disjunctive_pair_count(), 0);

    let mut medium = Solver::new();
    let medium_starts = (0..4).map(|_| medium.new_var_range(0, 4_000)).collect::<Vec<_>>();
    no_overlap(&mut medium, &medium_starts, &[1; 4]);
    assert_eq!(medium.num_propagators(), 1);
    assert_eq!(medium.store.disjunctive_pair_count(), 0);

    let mut large = Solver::new();
    let large_starts = (0..33).map(|_| large.new_var_range(0, 64)).collect::<Vec<_>>();
    no_overlap(&mut large, &large_starts, &[1; 33]);
    assert_eq!(large.num_propagators(), 1);
    assert_eq!(large.store.disjunctive_pair_count(), 0);
}

#[test]
fn no_overlap_validation_rejects_negative_and_accepts_i64_durations() {
    let mut negative = Model::new();
    let starts = vec![negative.int_range(0, 10), negative.int_range(0, 10)];
    negative.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap { starts, durations: vec![-1, 1] }));
    let errors = negative.validate().expect_err("negative no-overlap durations must be rejected").join("\n");
    assert!(errors.contains("interval duration must be non-negative"), "missing negative-duration diagnostic: {errors}");

    let mut wide = Model::new();
    let starts = vec![wide.int_range(i32::MIN, i32::MIN), wide.int_range(i32::MAX, i32::MAX)];
    wide.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap { starts, durations: vec![3_000_000_000, 1] }));
    wide.validate().expect("integer no-overlap uses i64 durations");

    let mut solver = Solver::new();
    let first = solver.new_var_set(&[i32::MIN]);
    let second = solver.new_var_set(&[i32::MAX]);
    no_overlap(&mut solver, &[first, second], &[3_000_000_000, 1]);
    assert_eq!(enumerate(&mut solver, &[first, second]), BTreeSet::from([vec![i32::MIN, i32::MAX]]));
}

#[test]
fn cumulative_matches_oracle() {
    let dur = [2i64, 2, 2];
    let h = [1i64, 1, 1];
    let cap = 2;
    let mut s = Solver::new();
    let starts: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 3)).collect();
    cumulative(&mut s, &starts, &dur, &h, cap);
    let got = enumerate(&mut s, &starts);

    let want = oracle_set(3, 0, 3, |a| {
        // Check every integer time point in the span.
        (0..8).all(|t| {
            let used: i64 = (0..3).filter(|&i| a[i] as i64 <= t && t < a[i] as i64 + dur[i]).map(|i| h[i]).sum();
            used <= cap
        })
    });
    assert_eq!(got, want);
}

#[test]
fn cumulative_search_finds_no_infeasible() {
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 3)).collect();
    cumulative(&mut s, &v, &[2, 1, 2, 1], &[4, 4, 3, 2], 4);
    let got = enumerate(&mut s, &v);
    assert!(got.is_empty(), "search returned infeasible: {got:?}");
}

#[test]
fn cumulative_rejects_overloaded_leaf() {
    // [3,2,0,0]: at t=0, tasks 2 (h3) and 3 (h2) overlap => 5 > capacity 4.
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 3)).collect();
    cumulative(&mut s, &v, &[2, 1, 2, 1], &[4, 4, 3, 2], 4);
    s.store.push_level();
    s.store.fix(v[0], 3).unwrap();
    s.store.fix(v[1], 2).unwrap();
    s.store.fix(v[2], 0).unwrap();
    s.store.fix(v[3], 0).unwrap();
    assert!(s.propagate().is_err(), "t=0 overload (3+2 > 4) must fail");
    s.store.pop_level();
}

#[test]
fn cumulative_edge_finding_matches_oracle_randomized() {
    // Soundness net for edge-finding: across many random small cumulative
    // instances, the solver's solution set must EQUAL the oracle's. Any unsound
    // earliest-start bump would drop a feasible start and fail this.
    let mut seed: u64 = 0x1234_5678_9abc_def1;
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed >> 33
    };
    for _ in 0..300 {
        let n = 2 + (rng() % 3) as usize; // 2..4 tasks
        let cap = 2 + (rng() % 3) as i64; // 2..4
        let durations: Vec<i64> = (0..n).map(|_| 1 + (rng() % 3) as i64).collect();
        let heights: Vec<i64> = (0..n).map(|_| 1 + (rng() % cap as u64) as i64).collect(); // 1..cap
        let (lo, hi) = (0, 3);

        let mut s = Solver::new();
        let starts: Vec<VarId> = (0..n).map(|_| s.new_var_range(lo, hi)).collect();
        cumulative(&mut s, &starts, &durations, &heights, cap);
        let got = enumerate(&mut s, &starts);

        let want = oracle_set(n, lo, hi, |a| {
            let end = (0..n).map(|i| a[i] as i64 + durations[i]).max().unwrap();
            (0..end).all(|t| {
                let used: i64 = (0..n).filter(|&i| a[i] as i64 <= t && t < a[i] as i64 + durations[i]).map(|i| heights[i]).sum();
                used <= cap
            })
        });
        assert_eq!(got, want, "n={n} cap={cap} dur={durations:?} h={heights:?}");
    }
}

#[test]
fn cumulative_var_height_matches_oracle_randomized() {
    // Variable heights AND a variable capacity: the solver's solution set must
    // EQUAL the oracle's. Any unsound prune (a removed feasible start/height/cap)
    // breaks equality; the weak propagator may only refuse the infeasible.
    let mut seed: u64 = 0xfeed_face_dead_beef;
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed >> 33
    };
    for _ in 0..150 {
        let n = 2 + (rng() % 2) as usize; // 2..3 tasks
        let durations: Vec<i64> = (0..n).map(|_| 1 + (rng() % 3) as i64).collect();
        let (slo, shi) = (0i32, 3i32);
        let (hlo, hhi) = (0i32, 3i32);
        let (clo, chi) = (2i32, 4i32);

        let mut s = Solver::new();
        let starts: Vec<VarId> = (0..n).map(|_| s.new_var_range(slo, shi)).collect();
        let heights: Vec<VarId> = (0..n).map(|_| s.new_var_range(hlo, hhi)).collect();
        let durs: Vec<VarId> = durations.iter().map(|&d| s.new_var_range(d as i32, d as i32)).collect();
        let capv = s.new_var_range(clo, chi);
        cumulative_var(&mut s, &starts, &durs, &heights, capv);
        let all: Vec<VarId> = starts.iter().chain(&heights).chain(std::iter::once(&capv)).copied().collect();
        let got = enumerate(&mut s, &all);

        let mut domains: Vec<Vec<i32>> = Vec::new();
        domains.extend((0..n).map(|_| (slo..=shi).collect::<Vec<i32>>()));
        domains.extend((0..n).map(|_| (hlo..=hhi).collect::<Vec<i32>>()));
        domains.push((clo..=chi).collect());
        let want: BTreeSet<Vec<i32>> = oracle::solutions(&domains, |a| {
            let (st, rest) = a.split_at(n);
            let (ht, capslice) = rest.split_at(n);
            let cap = capslice[0] as i64;
            let end = (0..n).map(|i| st[i] as i64 + durations[i]).max().unwrap();
            (0..end).all(|t| {
                let used: i64 = (0..n).filter(|&i| st[i] as i64 <= t && t < st[i] as i64 + durations[i]).map(|i| ht[i] as i64).sum();
                used <= cap
            })
        })
        .into_iter()
        .collect();
        assert_eq!(got, want, "n={n} dur={durations:?}");
    }
}

#[test]
fn cumulative_var_duration_matches_oracle_randomized() {
    // Variable durations as well: the solver's solution set must EQUAL the
    // oracle's, exercising the mandatory-part `[lst, est+min_dur)` reasoning and
    // the `min_dur` start-forbidding.
    let mut seed: u64 = 0x0bad_c0de_1357_9bdf;
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed >> 33
    };
    for _ in 0..120 {
        let n = 2 + (rng() % 2) as usize; // 2..3 tasks
        let (slo, shi) = (0i32, 2i32);
        let (dlo, dhi) = (1i32, 2i32);
        let (hlo, hhi) = (0i32, 2i32);
        let cap = 2 + (rng() % 2) as i64; // fixed 2..3

        let mut s = Solver::new();
        let starts: Vec<VarId> = (0..n).map(|_| s.new_var_range(slo, shi)).collect();
        let durs: Vec<VarId> = (0..n).map(|_| s.new_var_range(dlo, dhi)).collect();
        let heights: Vec<VarId> = (0..n).map(|_| s.new_var_range(hlo, hhi)).collect();
        let capv = s.new_var_range(cap as i32, cap as i32);
        cumulative_var(&mut s, &starts, &durs, &heights, capv);
        let all: Vec<VarId> = starts.iter().chain(&durs).chain(&heights).copied().collect();
        let got = enumerate(&mut s, &all);

        let mut domains: Vec<Vec<i32>> = Vec::new();
        domains.extend((0..n).map(|_| (slo..=shi).collect::<Vec<i32>>()));
        domains.extend((0..n).map(|_| (dlo..=dhi).collect::<Vec<i32>>()));
        domains.extend((0..n).map(|_| (hlo..=hhi).collect::<Vec<i32>>()));
        let want: BTreeSet<Vec<i32>> = oracle::solutions(&domains, |a| {
            let (st, rest) = a.split_at(n);
            let (du, ht) = rest.split_at(n);
            let end = (0..n).map(|i| st[i] as i64 + du[i] as i64).max().unwrap();
            (0..end).all(|t| {
                let used: i64 = (0..n).filter(|&i| st[i] as i64 <= t && t < st[i] as i64 + du[i] as i64).map(|i| ht[i] as i64).sum();
                used <= cap
            })
        })
        .into_iter()
        .collect();
        assert_eq!(got, want, "n={n} cap={cap}");
    }
}

#[test]
fn bin_packing_matches_oracle() {
    let sizes = [2i64, 3, 2];
    let caps = [4i64, 4];
    let mut s = Solver::new();
    let items: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 1)).collect();
    bin_packing(&mut s, &items, &sizes, &caps);
    let got = enumerate(&mut s, &items);

    let want = oracle_set(3, 0, 1, |a| {
        (0..2).all(|b| {
            let load: i64 = (0..3).filter(|&i| a[i] == b as i32).map(|i| sizes[i]).sum();
            load <= caps[b]
        })
    });
    assert_eq!(got, want);
}

#[test]
fn bin_packing_rejects_overload_created_during_propagation() {
    // Item 0 fills bin 0. Pruning then fixes items 1 and 2 to bin 1, which
    // overloads it and must be detected before the propagator returns.
    let mut s = Solver::new();
    let items = [s.new_var_set(&[0]), s.new_var_set(&[0, 1]), s.new_var_set(&[0, 1])];
    bin_packing(&mut s, &items, &[1, 1, 1], &[1, 1]);
    assert!(enumerate(&mut s, &items).is_empty());
}

#[test]
fn knapsack_matches_oracle() {
    let weights = [2i64, 3, 4];
    let profits = [3i64, 4, 5];
    let mut s = Solver::new();
    let x: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 1)).collect();
    knapsack(&mut s, &x, &weights, &profits, Relation::Le, 5, Relation::Ge, 5);
    let got = enumerate(&mut s, &x);

    let want = oracle_set(3, 0, 1, |a| {
        let w: i64 = (0..3).map(|i| weights[i] * a[i] as i64).sum();
        let p: i64 = (0..3).map(|i| profits[i] * a[i] as i64).sum();
        w <= 5 && p >= 5
    });
    assert_eq!(got, want);
}
