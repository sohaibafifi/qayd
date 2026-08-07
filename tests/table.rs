//! Oracle cross-checks for the table family: extension (positive/negative),
//! regular, and mdd.

use std::collections::BTreeSet;

use std::sync::Arc;

use qayd::constraints::table::{extension, extension_from_template, extension_template, mdd, regular, Dfa, Mdd, MddArc, STAR};
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
fn positive_extension_matches_oracle() {
    let allowed = [vec![0, 0, 0], vec![1, 2, 0], vec![2, 1, 2], vec![0, 1, 2]];
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 2)).collect();
    extension(&mut s, &v, &allowed, true);
    let got = enumerate(&mut s, &v);

    let allowed_set: BTreeSet<Vec<i32>> = allowed.iter().cloned().collect();
    let want = oracle_set(3, 0, 2, |a| allowed_set.contains(a));
    assert_eq!(got, want);
    assert_eq!(got, allowed_set);
}

#[test]
fn negative_extension_matches_oracle() {
    let forbidden = [vec![0, 0], vec![1, 1], vec![2, 2]];
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..2).map(|_| s.new_var_range(0, 2)).collect();
    extension(&mut s, &v, &forbidden, false);
    let got = enumerate(&mut s, &v);

    let forbidden_set: BTreeSet<Vec<i32>> = forbidden.iter().cloned().collect();
    let want = oracle_set(2, 0, 2, |a| !forbidden_set.contains(a));
    assert_eq!(got, want);
    // "not equal" — the off-diagonal of a 3x3 grid.
    assert_eq!(got.len(), 6);
}

#[test]
fn negative_extension_handles_overlapping_wildcards() {
    let forbidden = [vec![STAR, 0], vec![0, 0]];
    let mut s = Solver::new();
    let x = s.new_var_set(&[0]);
    let y = s.new_var_set(&[0, 1]);
    extension(&mut s, &[x, y], &forbidden, false);

    assert_eq!(enumerate(&mut s, &[x, y]), BTreeSet::from([vec![0, 1]]));
}

#[test]
fn negative_extension_wildcard_subsets_match_oracle() {
    let patterns =
        [vec![0, 0], vec![0, 1], vec![0, STAR], vec![1, 0], vec![1, 1], vec![1, STAR], vec![STAR, 0], vec![STAR, 1], vec![STAR, STAR]];

    for selected in 0..1usize << patterns.len() {
        let forbidden: Vec<Vec<i32>> =
            patterns.iter().enumerate().filter(|(i, _)| selected & (1 << i) != 0).map(|(_, tuple)| tuple.clone()).collect();
        let mut s = Solver::new();
        let v: Vec<VarId> = (0..2).map(|_| s.new_var_range(0, 1)).collect();
        extension(&mut s, &v, &forbidden, false);
        let got = enumerate(&mut s, &v);

        let want = oracle_set(2, 0, 1, |a| {
            !forbidden.iter().any(|tuple| tuple.iter().zip(a).all(|(&pattern, &val)| pattern == STAR || pattern == val))
        });
        assert_eq!(got, want, "selected patterns: {selected:#011b}");
    }
}

#[test]
fn regular_no_two_consecutive_ones() {
    // States: 0 = last bit 0 (or start), 1 = last bit 1. No (1,1) transition.
    let dfa = Dfa { n_states: 2, start: 0, accept: vec![0, 1], transitions: vec![(0, 0, 0), (0, 1, 1), (1, 0, 0)] };
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 1)).collect();
    regular(&mut s, &v, dfa);
    let got = enumerate(&mut s, &v);

    let want = oracle_set(4, 0, 1, |a| a.windows(2).all(|w| !(w[0] == 1 && w[1] == 1)));
    assert_eq!(got, want);
}

#[test]
fn mdd_at_most_one_set_bit() {
    // Layered MDD over two binary vars enforcing a0 + a1 <= 1.
    let m = Mdd {
        nodes_per_layer: vec![1, 2, 1],
        layers: vec![
            // var0: root -> {sum0, sum1}
            vec![MddArc { from: 0, value: 0, to: 0 }, MddArc { from: 0, value: 1, to: 1 }],
            // var1: sum0 -> term on 0/1; sum1 -> term only on 0
            vec![MddArc { from: 0, value: 0, to: 0 }, MddArc { from: 0, value: 1, to: 0 }, MddArc { from: 1, value: 0, to: 0 }],
        ],
    };
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..2).map(|_| s.new_var_range(0, 1)).collect();
    mdd(&mut s, &v, m);
    let got = enumerate(&mut s, &v);

    let want = oracle_set(2, 0, 1, |a| a[0] + a[1] <= 1);
    assert_eq!(got, want);
    assert_eq!(got.len(), 3);
}

#[test]
fn positive_extension_prunes_at_root() {
    // Only one tuple => everything is forced.
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 5)).collect();
    extension(&mut s, &v, &[vec![3, 1, 4]], true);
    s.store.push_level();
    s.enqueue_all();
    s.propagate().unwrap();
    assert_eq!(s.store.value(v[0]), 3);
    assert_eq!(s.store.value(v[1]), 1);
    assert_eq!(s.store.value(v[2]), 4);
    s.store.pop_level();
}

#[test]
fn shared_extension_template_matches_oracle() {
    let allowed = [vec![0, 1], vec![1, 0]];
    let template = extension_template(2, &allowed);
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 1)).collect();
    extension_from_template(&mut s, &v[0..2], Arc::clone(&template), true);
    extension_from_template(&mut s, &v[2..4], template, true);

    let got = enumerate(&mut s, &v);
    let want = oracle_set(4, 0, 1, |a| a[0] != a[1] && a[2] != a[3]);
    assert_eq!(got, want);
}

/// Micro-bench: a wide, dense positive table woken by
/// single-value removals with periodic backtracking, so each wake dirties ~one
/// column. Exercises the reversible `currTable` load/update/restore path.
/// Run with `cargo test --release --test table -- --ignored --nocapture bench_wide`.
#[test]
#[ignore]
fn bench_wide_table_single_column_wakes() {
    let arity = 16;
    let (lo, hi) = (0, 9);
    let ntuples = 3000;
    let mut seed: u64 = 0xdead_beef_cafe_1234;
    let mut rnd = |m: i32| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        lo + ((seed >> 33) as i32 % m)
    };
    let span = hi - lo + 1;
    let tuples: Vec<Vec<i32>> = (0..ntuples).map(|_| (0..arity).map(|_| rnd(span)).collect()).collect();

    let mut s = Solver::new();
    let v: Vec<VarId> = (0..arity).map(|_| s.new_var_range(lo, hi)).collect();
    extension(&mut s, &v, &tuples, true);
    s.store.push_level();
    s.enqueue_all();
    s.propagate().unwrap();

    let start = std::time::Instant::now();
    let mut depth = 0usize;
    let mut propagates = 0u64;
    for iter in 0..300_000u64 {
        if depth >= 5 {
            for _ in 0..depth {
                s.store.pop_level();
            }
            depth = 0;
            continue;
        }
        let i = (iter as usize) % arity;
        let val = s.store.values(v[i]).find(|&x| s.store.size(v[i]) > 1 && x >= lo);
        let Some(val) = val else {
            if depth > 0 {
                s.store.pop_level();
                depth -= 1;
            }
            continue;
        };
        s.store.push_level();
        let _ = s.store.remove(v[i], val);
        s.enqueue_all();
        propagates += 1;
        if s.propagate().is_err() {
            s.store.pop_level();
        } else {
            depth += 1;
        }
    }
    for _ in 0..depth {
        s.store.pop_level();
    }
    s.store.pop_level();
    eprintln!("bench_wide_table: {propagates} propagates in {:?}", start.elapsed());
}

/// Independent domain-consistent fixpoint for a positive table (the reference
/// the reversible incremental `currTable` must reproduce). Returns false on
/// wipeout.
fn ac_fixpoint(tuples: &[Vec<i32>], arity: usize, dom: &mut [BTreeSet<i32>]) -> bool {
    loop {
        let mut changed = false;
        for i in 0..arity {
            let vals: Vec<i32> = dom[i].iter().copied().collect();
            for val in vals {
                let supported = tuples
                    .iter()
                    .any(|t| (t[i] == STAR || t[i] == val) && (0..arity).all(|j| j == i || t[j] == STAR || dom[j].contains(&t[j])));
                if !supported {
                    dom[i].remove(&val);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    dom.iter().all(|d| !d.is_empty())
}

/// Randomized remove/propagate/backtrack script: the reversible incremental
/// filtering must equal a from-scratch fixpoint at every stable point, across
/// push/pop so the reversible `currTable` restore is exercised.
#[test]
fn positive_extension_incremental_matches_scratch_under_backtracking() {
    let arity = 4;
    let (lo, hi) = (0, 3);
    let tuples: Vec<Vec<i32>> = vec![
        vec![0, 0, 0, 0],
        vec![1, 2, STAR, 0],
        vec![2, 1, 2, 3],
        vec![STAR, 3, 1, 1],
        vec![3, 3, 3, 3],
        vec![0, 1, 2, STAR],
        vec![2, 2, 2, 0],
        vec![1, 0, 0, 1],
        vec![STAR, STAR, 3, 2],
    ];

    let mut s = Solver::new();
    let v: Vec<VarId> = (0..arity).map(|_| s.new_var_range(lo, hi)).collect();
    extension(&mut s, &v, &tuples, true);
    s.store.push_level(); // base level; initial propagate folds all columns
    s.enqueue_all();
    s.propagate().unwrap();

    let full: BTreeSet<i32> = (lo..=hi).collect();
    let mut dom: Vec<BTreeSet<i32>> = vec![full.clone(); arity];
    ac_fixpoint(&tuples, arity, &mut dom);
    for i in 0..arity {
        assert_eq!(s.store.values(v[i]).collect::<BTreeSet<_>>(), dom[i], "initial column {i}");
    }

    // Deterministic LCG so failures reproduce without an rng dependency.
    let mut seed: u64 = 0x9e3779b97f4a7c15;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };

    let mut snapshots: Vec<Vec<BTreeSet<i32>>> = Vec::new();
    for _ in 0..4000 {
        let action = next() % 3;
        if action == 0 && !snapshots.is_empty() {
            s.store.pop_level();
            dom = snapshots.pop().unwrap();
            continue;
        }
        // Pick a column with a still-removable value.
        let start = (next() as usize) % arity;
        let mut picked = None;
        for k in 0..arity {
            let i = (start + k) % arity;
            if dom[i].len() > 1 {
                let vals: Vec<i32> = dom[i].iter().copied().collect();
                picked = Some((i, vals[(next() as usize) % vals.len()]));
                break;
            }
        }
        let Some((i, val)) = picked else { continue };

        snapshots.push(dom.clone());
        s.store.push_level();
        if s.store.contains(v[i], val) {
            let _ = s.store.remove(v[i], val);
        }
        dom[i].remove(&val);
        s.enqueue_all();
        let mut reference = dom.clone();
        let feasible = ac_fixpoint(&tuples, arity, &mut reference);
        match s.propagate() {
            Ok(()) => {
                assert!(feasible, "solver kept a wiped-out domain feasible");
                for c in 0..arity {
                    assert_eq!(s.store.values(v[c]).collect::<BTreeSet<_>>(), reference[c], "column {c} after remove {val}@{i}");
                }
                dom = reference;
            }
            Err(_) => {
                assert!(!feasible, "solver failed but reference is satisfiable");
                s.store.pop_level();
                dom = snapshots.pop().unwrap();
            }
        }
    }
    s.store.pop_level();
}

#[test]
fn cloned_solver_with_shared_extension_template_matches_oracle() {
    let allowed = [vec![0, 1], vec![1, 0]];
    let template = extension_template(2, &allowed);
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 1)).collect();
    extension_from_template(&mut s, &v[0..2], Arc::clone(&template), true);
    extension_from_template(&mut s, &v[2..4], template, true);
    let mut clone = s.clone();

    let want = oracle_set(4, 0, 1, |a| a[0] != a[1] && a[2] != a[3]);
    assert_eq!(enumerate(&mut s, &v), want);
    assert_eq!(enumerate(&mut clone, &v), want);
}
