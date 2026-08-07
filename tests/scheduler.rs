//! Priority-banded scheduling must reach the same fixpoint as the old FIFO
//! queue: a model mixing a cheap monotone chain (Cheap/Linear bands) with an
//! expensive global (Expensive band) enumerates exactly the oracle solutions.

use std::collections::BTreeSet;

use qayd::constraints::count::{cardinality, n_values};
use qayd::constraints::linear::{sum, Relation};
use qayd::constraints::primitives::ordered;
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

fn distinct(a: &[i32]) -> i64 {
    a.iter().collect::<BTreeSet<_>>().len() as i64
}

fn build_bench_model() -> (Solver, Vec<VarId>) {
    // Long cheap monotone chain (band Linear) around an expensive cardinality
    // global (band Expensive, flow-based): under FIFO the global is re-woken
    // between chain steps; under banding the chain drains first, so the global
    // runs far less.
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..8).map(|_| s.new_var_range(0, 7)).collect();
    ordered(&mut s, &v, Relation::Le);
    cardinality(&mut s, &v, &[0, 1, 2, 3], &[0; 4], &[3; 4], false);
    sum(&mut s, &v, Relation::Le, 24);
    (s, v)
}

#[test]
fn banding_reduces_propagator_invocations() {
    // Same search tree either way (fixpoint is confluent); banding just runs the
    // expensive global fewer times. Assert identical solutions and fewer calls.
    let (mut fifo, fv) = build_bench_model();
    fifo.set_single_band(true);
    qayd::store::reset_prop_calls();
    let fifo_sols = enumerate(&mut fifo, &fv);
    let (fifo_calls, fifo_exp) = (qayd::store::prop_calls(), qayd::store::expensive_calls());

    let (mut banded, bv) = build_bench_model();
    banded.set_single_band(false);
    qayd::store::reset_prop_calls();
    let banded_sols = enumerate(&mut banded, &bv);
    let (banded_calls, banded_exp) = (qayd::store::prop_calls(), qayd::store::expensive_calls());

    eprintln!("total invocations:     fifo={fifo_calls} banded={banded_calls}");
    eprintln!("expensive invocations: fifo={fifo_exp} banded={banded_exp}");
    assert_eq!(fifo_sols, banded_sols, "banding must preserve solutions");
    assert!(banded_exp < fifo_exp, "banding should cut expensive-global invocations");
}

#[test]
fn cheap_chain_plus_expensive_global_matches_oracle() {
    // x1 <= x2 <= x3 <= x4 (cheap binary chain), exactly 3 distinct values
    // (expensive nValues global), sum == 6.
    let mut s = Solver::new();
    let v: Vec<VarId> = (0..4).map(|_| s.new_var_range(0, 4)).collect();
    ordered(&mut s, &v, Relation::Le);
    n_values(&mut s, &v, Relation::Eq, 3);
    sum(&mut s, &v, Relation::Eq, 6);
    let got = enumerate(&mut s, &v);

    let want: BTreeSet<Vec<i32>> = oracle::solutions(&vec![(0..=4).collect::<Vec<i32>>(); 4], |a| {
        a.windows(2).all(|w| w[0] <= w[1]) && distinct(a) == 3 && a.iter().sum::<i32>() == 6
    })
    .into_iter()
    .collect();

    assert_eq!(got, want);
    assert!(!got.is_empty());
}
