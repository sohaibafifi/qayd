//! Oracle cross-checks for cardinality (GCC bounds) and nValues.

use std::collections::BTreeSet;

use qayd::constraints::count::{cardinality, count, n_values};
use qayd::constraints::linear::Relation;
mod common;

use common::oracle;
use qayd::{solve_search, SearchControl, Solver, VarId};

/// Deterministic LCG so the randomized oracle check is reproducible.
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state >> 33
}

fn enumerate(solver: &mut Solver, vars: &[VarId]) -> BTreeSet<Vec<i32>> {
    let mut out = BTreeSet::new();
    solve_search(solver, vars, |s| {
        out.insert(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
        SearchControl::Continue
    });
    out
}

fn count_val(a: &[i32], v: i32) -> i64 {
    a.iter().filter(|&&x| x == v).count() as i64
}

#[test]
fn cardinality_open_matches_oracle() {
    // 4 vars in 0..2: 1<=#0<=2 and 0<=#1<=1 (value 2 unconstrained).
    let values = [0, 1];
    let low = [1, 0];
    let high = [2, 1];
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 2)).collect();
    cardinality(&mut s, &v, &values, &low, &high, false);
    let got = enumerate(&mut s, &v);

    let want: BTreeSet<Vec<i32>> =
        oracle::solutions(&vec![(0..=2).collect::<Vec<i32>>(); 4], |a| (1..=2).contains(&count_val(a, 0)) && count_val(a, 1) <= 1)
            .into_iter()
            .collect();
    assert_eq!(got, want);
}

#[test]
fn cardinality_closed_restricts_domain() {
    // Domains 0..3 but closed to {0,1}; exactly two 0s, rest 1s among 3 vars.
    let values = [0, 1];
    let low = [2, 0];
    let high = [2, 3];
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 3)).collect();
    cardinality(&mut s, &v, &values, &low, &high, true);
    let got = enumerate(&mut s, &v);

    let want: BTreeSet<Vec<i32>> =
        oracle::solutions(&vec![(0..=3).collect::<Vec<i32>>(); 3], |a| a.iter().all(|&x| x == 0 || x == 1) && count_val(a, 0) == 2)
            .into_iter()
            .collect();
    assert_eq!(got, want);
    assert_eq!(got.len(), 3); // choose which position is the lone 1
}

fn distinct(a: &[i32]) -> i64 {
    let mut s: Vec<i32> = a.to_vec();
    s.sort_unstable();
    s.dedup();
    s.len() as i64
}

#[test]
fn nvalues_relations_match_oracle() {
    for (rel, k) in [(Relation::Eq, 2), (Relation::Le, 2), (Relation::Ge, 2), (Relation::Lt, 3), (Relation::Gt, 1), (Relation::Ne, 1)] {
        let mut s = Solver::new();
        let v: Vec<VarId> = (0..3).map(|_| s.new_var_range(0, 2)).collect();
        n_values(&mut s, &v, rel, k);
        let got = enumerate(&mut s, &v);

        let want: BTreeSet<Vec<i32>> = oracle::solutions(&vec![(0..=2).collect::<Vec<i32>>(); 3], |a| {
            let d = distinct(a);
            match rel {
                Relation::Eq => d == k,
                Relation::Le => d <= k,
                Relation::Ge => d >= k,
                Relation::Lt => d < k,
                Relation::Gt => d > k,
                Relation::Ne => d != k,
            }
        })
        .into_iter()
        .collect();
        assert_eq!(got, want, "nValues rel={rel:?} k={k}");
    }
}

// ===========================================================================
// Régin flow global: randomized equivalence, joint-infeasibility root
// detection, node-count evidence.
// ===========================================================================

/// GCC predicate for the brute-force oracle.
fn gcc_ok(a: &[i32], values: &[i32], low: &[i64], high: &[i64], closed: bool) -> bool {
    for (j, &val) in values.iter().enumerate() {
        let c = count_val(a, val);
        if c < low[j] || c > high[j] {
            return false;
        }
    }
    if closed && !a.iter().all(|&x| values.contains(&x)) {
        return false;
    }
    true
}

#[test]
fn cardinality_flow_matches_oracle_randomized() {
    let mut rng = 0x1234_5678_9abc_def0u64;
    for _ in 0..400 {
        let n = 2 + (lcg(&mut rng) % 3) as usize; // 2..=4
        let dmax = 1 + (lcg(&mut rng) % 3) as i32; // universe 0..=dmax, dmax 1..=3
        let closed = lcg(&mut rng) & 1 == 0;

        // Listed values: a nonempty subset of 0..=dmax with random [low, high].
        let mut values = Vec::new();
        let mut low = Vec::new();
        let mut high = Vec::new();
        for v in 0..=dmax {
            if lcg(&mut rng) & 1 == 0 {
                let lo = (lcg(&mut rng) % 3) as i64; // 0..=2
                let hi = lo + (lcg(&mut rng) % ((n as u64) + 1)) as i64;
                values.push(v);
                low.push(lo);
                high.push(hi);
            }
        }
        if values.is_empty() {
            values.push(0);
            low.push(0);
            high.push(n as i64);
        }

        let mut s = Solver::new();
        let vars: Vec<VarId> = (0..n).map(|_| s.new_var_range(0, dmax)).collect();
        cardinality(&mut s, &vars, &values, &low, &high, closed);
        let got = enumerate(&mut s, &vars);

        let domains = vec![(0..=dmax).collect::<Vec<i32>>(); n];
        let want: BTreeSet<Vec<i32>> = oracle::solutions(&domains, |a| gcc_ok(a, &values, &low, &high, closed)).into_iter().collect();

        assert_eq!(got, want, "n={n} dmax={dmax} closed={closed} values={values:?} low={low:?} high={high:?}");
    }
}

/// Post the old per-value decomposition for A/B comparison.
fn post_decomposition(s: &mut Solver, vars: &[VarId], values: &[i32], low: &[i64], high: &[i64]) {
    for (j, &val) in values.iter().enumerate() {
        count(s, vars, val, Relation::Ge, low[j]);
        count(s, vars, val, Relation::Le, high[j]);
    }
}

#[test]
fn cardinality_joint_infeasible_detected_at_root() {
    // 4 vars all in {1,2}; at most one of each value -> jointly infeasible,
    // yet each per-value Count is silent at the root. Measured by pure root
    // propagation (search-level node counts are masked by the engine's root
    // probing, which resolves both models without branching).
    let values = [1, 2];
    let low = [0, 0];
    let high = [1, 1];

    let mut sg = Solver::new();
    let vg: Vec<VarId> = (0..4).map(|_| sg.new_var_range(1, 2)).collect();
    cardinality(&mut sg, &vg, &values, &low, &high, false);
    assert!(sg.propagate().is_err(), "flow detects joint infeasibility at the root");

    let mut sd = Solver::new();
    let vd: Vec<VarId> = (0..4).map(|_| sd.new_var_range(1, 2)).collect();
    post_decomposition(&mut sd, &vd, &values, &low, &high);
    assert!(sd.propagate().is_ok(), "decomposition is silent at the root");

    // Both models are ultimately unsatisfiable.
    let mut sg2 = Solver::new();
    let vg2: Vec<VarId> = (0..4).map(|_| sg2.new_var_range(1, 2)).collect();
    cardinality(&mut sg2, &vg2, &values, &low, &high, false);
    assert!(enumerate(&mut sg2, &vg2).is_empty());
}

#[test]
fn cardinality_lower_bound_infeasible_at_root() {
    // 5 vars, but only 2 of them can take value 0; require three 0s -> fail.
    let values = [0];
    let low = [3];
    let high = [5];
    let mut s = Solver::new();
    let vars: Vec<VarId> = (0..5).enumerate().map(|(i, _)| if i < 2 { s.new_var_range(0, 1) } else { s.new_var_range(1, 2) }).collect();
    cardinality(&mut s, &vars, &values, &low, &high, false);
    let mut nsol = 0;
    let stats = solve_search(&mut s, &vars, |_| {
        nsol += 1;
        SearchControl::Continue
    });
    assert_eq!(nsol, 0);
    assert_eq!(stats.nodes, 0, "lower-bound infeasibility caught at the root");
}

#[test]
fn cardinality_flow_prunes_hall_set_at_root() {
    // Permutation of {0..3}: x0,x1 in {0,1} form a Hall set that must occupy
    // {0,1}, so DC removes 0 and 1 from x2,x3 at the root. The per-value bounds
    // decomposition stays silent (every value still has >=1 candidate).
    let values = [0, 1, 2, 3];
    let low = [1, 1, 1, 1];
    let high = [1, 1, 1, 1];

    let mut sg = Solver::new();
    let x0 = sg.new_var_range(0, 1);
    let x1 = sg.new_var_range(0, 1);
    let x2 = sg.new_var_range(0, 3);
    let x3 = sg.new_var_range(0, 3);
    let vg = [x0, x1, x2, x3];
    cardinality(&mut sg, &vg, &values, &low, &high, false);
    assert!(sg.propagate().is_ok());
    // DC pruned the Hall values from the outside variables.
    for &v in &[x2, x3] {
        assert!(!sg.store.contains(v, 0), "flow removes 0 from outside var");
        assert!(!sg.store.contains(v, 1), "flow removes 1 from outside var");
        assert!(sg.store.contains(v, 2) && sg.store.contains(v, 3));
    }

    let mut sd = Solver::new();
    let y0 = sd.new_var_range(0, 1);
    let y1 = sd.new_var_range(0, 1);
    let y2 = sd.new_var_range(0, 3);
    let y3 = sd.new_var_range(0, 3);
    let vd = [y0, y1, y2, y3];
    post_decomposition(&mut sd, &vd, &values, &low, &high);
    assert!(sd.propagate().is_ok());
    // The bounds decomposition leaves the Hall values in place (weaker).
    assert!(sd.store.contains(y2, 0), "decomposition keeps 0 (no DC)");
}

// ===========================================================================
// nValues: endpoint pruning.
// ===========================================================================

#[test]
fn nvalues_endpoint_a_forces_all_equal() {
    // Eq k=1 -> all variables equal -> prune to the common intersection {1,2}.
    let mut s = Solver::new();
    let a = s.new_var_range(0, 2);
    let b = s.new_var_set(&[1, 2, 3]);
    let c = s.new_var_set(&[1, 2, 4]);
    let vars = [a, b, c];
    n_values(&mut s, &vars, Relation::Eq, 1);
    let got = enumerate(&mut s, &vars);

    let domains = vec![vec![0, 1, 2], vec![1, 2, 3], vec![1, 2, 4]];
    let want: BTreeSet<Vec<i32>> = oracle::solutions(&domains, |x| distinct(x) == 1).into_iter().collect();
    assert_eq!(got, want);
    // Only 1 and 2 survive as common values -> two solutions.
    assert_eq!(got.len(), 2);
}

#[test]
fn nvalues_endpoint_b_fixes_unique_provider() {
    // Ge k=|union| forces every union value used; value 5 has a single provider.
    let mut s = Solver::new();
    let a = s.new_var_set(&[5]); // only provider of 5 once fixed... use domain forcing
    let b = s.new_var_set(&[6, 7]);
    let c = s.new_var_set(&[6, 7]);
    let vars = [a, b, c];
    // union = {5,6,7}, |union| = 3 = n; require >= 3 distinct.
    n_values(&mut s, &vars, Relation::Ge, 3);
    let got = enumerate(&mut s, &vars);

    let domains = vec![vec![5], vec![6, 7], vec![6, 7]];
    let want: BTreeSet<Vec<i32>> = oracle::solutions(&domains, |x| distinct(x) == 3).into_iter().collect();
    assert_eq!(got, want);
    assert!(!got.is_empty());
}

#[test]
fn nvalues_endpoint_b_provider_pruned_when_needed() {
    // 3 vars, union {0,1,2} size 3 = n, require all distinct. Value 2's only
    // provider is c, so c must be 2 (b would otherwise be free to take 2).
    let mut s = Solver::new();
    let a = s.new_var_set(&[0, 1]);
    let b = s.new_var_set(&[0, 1]);
    let c = s.new_var_set(&[0, 1, 2]);
    let vars = [a, b, c];
    n_values(&mut s, &vars, Relation::Ge, 3);

    // Endpoint B must have fixed c to 2 at the root.
    solve_search(&mut s, &vars, |_| SearchControl::Stop);
    assert_eq!(s.store.value(c), 2, "unique provider of value 2 is fixed");
}

#[test]
fn nvalues_self_mutation_leaves_no_invalid_leaf() {
    // Regression: forced_min used to fix variables and return without
    // re-checking, and a propagator is never re-woken by its own mutations,
    // so search accepted leaves violating the constraint. The fixpoint loop
    // re-derives the bounds after every mutation round.
    let doms: Vec<Vec<i32>> = vec![vec![0, 3], vec![0, 1, 3, 4], vec![0, 2, 3, 4], vec![2, 3], vec![1, 3, 4]];
    let mut s = Solver::new();
    let v: Vec<VarId> = doms.iter().map(|d| s.new_var_set(d)).collect();
    n_values(&mut s, &v, Relation::Eq, 4);
    let got = enumerate(&mut s, &v);
    let want: BTreeSet<Vec<i32>> = oracle::solutions(&doms, |a| distinct(a) == 4).into_iter().collect();
    assert_eq!(got, want);
    assert!(got.iter().all(|a| distinct(a) == 4));
}

#[test]
fn nvalues_forced_min_cascade_is_rechecked() {
    // Regression: a forced fix could strip the last provider of an
    // earlier-visited union value without that value being rechecked.
    let doms: Vec<Vec<i32>> = vec![vec![0, 2], vec![0, 2, 3], vec![0, 1, 2, 3], vec![0, 1, 3]];
    let mut s = Solver::new();
    let v: Vec<VarId> = doms.iter().map(|d| s.new_var_set(d)).collect();
    n_values(&mut s, &v, Relation::Eq, 4);
    let got = enumerate(&mut s, &v);
    let want: BTreeSet<Vec<i32>> = oracle::solutions(&doms, |a| distinct(a) == 4).into_iter().collect();
    assert_eq!(got, want);
}
