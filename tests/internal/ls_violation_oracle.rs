//! Violation oracle for the `--ls` COP engine.
//!
//! `src/engines/ls/cop.rs` hand-reimplements ~24 constraint semantics as
//! violation functions. Any drift from the kernel propagators means a silently
//! wrong incumbent under `--ls`. For every `LocalConstraint` variant this test
//! draws randomized COMPLETE assignments and asserts:
//!
//!   LS-feasible(assignment)  ⇔  kernel-propagator-accepts(assignment)
//!
//! Boundary trick: both sides are built at the public API. Kernel side posts the
//! real propagator over singleton-domain variables and reads `propagate().is_ok()`.
//! LS side posts the same constraint through `LocalSearchSpec::add_*`, also over
//! singleton domains, so `mutable` is empty and `solve_ls` reports an incumbent
//! iff the assignment scores violation-free.

use std::sync::atomic::AtomicBool;

use qayd::constraints::count::{cardinality, count, n_values};
use qayd::constraints::flatten::{
    count_values_to_var, post_all_different_except, post_bin_loads, post_channel_onehot_index, post_diffn, post_tuple_not_equal,
};
use qayd::constraints::graph::circuit;
use qayd::constraints::intension::intension;
use qayd::constraints::lex::{channel, lex_chain};
use qayd::constraints::linear::{linear, Relation};
use qayd::constraints::primitives::{all_different, all_equal, element, maximum, minimum, precedence_with_covered};
use qayd::constraints::scheduling::{bin_packing, cumulative, no_overlap};
use qayd::constraints::table::{extension, mdd, regular, Dfa, Mdd, MddArc, STAR};
use qayd::engines::ls::cop::{solve_ls, LocalRhs, LocalSearchSpec, LsConfig};
use qayd::expr::{self, Expr};
use qayd::problem::{Objective, Problem};
use qayd::{Solver, VarId};

const ROUNDS: usize = 60;

/// xorshift64* (matches tests/lcg_alldiff_explanations.rs) for stable instances.
fn next(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Uniform integer in `[lo, hi]` inclusive.
fn rand_in(state: &mut u64, lo: i32, hi: i32) -> i32 {
    let span = (hi - lo + 1) as u64;
    lo + (next(state) % span) as i32
}

fn rand_bool(state: &mut u64) -> bool {
    next(state) & 1 == 0
}

/// Fisher-Yates permutation of `0..n`.
fn perm(state: &mut u64, n: usize) -> Vec<i32> {
    let mut p: Vec<i32> = (0..n as i32).collect();
    for i in (1..n).rev() {
        let j = (next(state) % (i as u64 + 1)) as usize;
        p.swap(i, j);
    }
    p
}

/// Kernel reference: post the propagator over singleton domains and report
/// whether root propagation accepts the assignment.
fn kernel_accepts(assignment: &[i32], post: impl FnOnce(&mut Solver, &[VarId])) -> bool {
    let mut solver = Solver::new();
    let vars: Vec<VarId> = assignment.iter().map(|&v| solver.new_var_set(&[v])).collect();
    post(&mut solver, &vars);
    solver.propagate().is_ok()
}

/// LS side: post through `add_*` over singleton domains, then run `solve_ls`.
/// With every domain a singleton, `mutable` is empty, so `solve_ls` takes the
/// early exit and fires `on_improve` iff the assignment is violation-free.
fn ls_feasible(assignment: &[i32], post: impl FnOnce(&mut Solver, &mut LocalSearchSpec, &[VarId])) -> bool {
    let mut solver = Solver::new();
    let vars: Vec<VarId> = assignment.iter().map(|&v| solver.new_var_set(&[v])).collect();
    let mut spec = LocalSearchSpec::default();
    for &v in &vars {
        spec.add_var(v);
    }
    post(&mut solver, &mut spec, &vars);
    let problem = Problem { solver, search: vars.clone(), objective: Some(Objective::Var(true, vars[0])) };
    // Already-stopped flag: the mutable-empty early path runs before the search
    // loop is entered, so no iteration ever happens - this only guards against a
    // hang if that invariant ever regressed.
    let stop = AtomicBool::new(true);
    let mut fired = false;
    solve_ls(problem, spec, &stop, 1, LsConfig::default(), |_v, _sol, _src| fired = true);
    fired
}

/// Assert kernel/LS agree on every round and that both outcomes were exercised.
struct Mix {
    name: &'static str,
    feas: usize,
    infeas: usize,
}

impl Mix {
    fn new(name: &'static str) -> Self {
        Self { name, feas: 0, infeas: 0 }
    }

    fn record(&mut self, kernel: bool, ls: bool, detail: impl std::fmt::Debug) {
        assert_eq!(kernel, ls, "{}: LS/kernel divergence on {:?} (kernel={kernel}, ls={ls})", self.name, detail);
        if kernel {
            self.feas += 1;
        } else {
            self.infeas += 1;
        }
    }

    fn finish(self) {
        assert!(self.feas > 0, "{}: never saw a feasible assignment; loosen the generator", self.name);
        assert!(self.infeas > 0, "{}: never saw an infeasible assignment; loosen the generator", self.name);
    }
}

#[test]
fn oracle_expr() {
    let mut st = 0x1234_5678_9abc_def0u64;
    let mut mix = Mix::new("expr");
    for _ in 0..ROUNDS {
        let asg = vec![rand_in(&mut st, -2, 3), rand_in(&mut st, -2, 3), rand_in(&mut st, -2, 3)];
        // (x0 != x1) and (x1 <= x2)
        let make = |v: &[VarId]| expr::and(vec![expr::ne(expr::var(v[0]), expr::var(v[1])), expr::le(expr::var(v[1]), expr::var(v[2]))]);
        let k = kernel_accepts(&asg, |s, v| intension(s, make(v)));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_expr(make(v)));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

#[test]
fn oracle_linear() {
    let mut st = 0x2222_3333_4444_5555u64;
    let mut mix = Mix::new("linear");
    let rels = [Relation::Eq, Relation::Ne, Relation::Le, Relation::Lt, Relation::Ge, Relation::Gt];
    for r in 0..ROUNDS {
        let asg = vec![rand_in(&mut st, -2, 3), rand_in(&mut st, -2, 3), rand_in(&mut st, -2, 3)];
        let coeffs = vec![rand_in(&mut st, -2, 2) as i64, rand_in(&mut st, -2, 2) as i64, rand_in(&mut st, -2, 2) as i64];
        let rel = rels[r % rels.len()];
        let rhs = rand_in(&mut st, -3, 3) as i64;
        let k = kernel_accepts(&asg, |s, v| linear(s, &coeffs, v, rel, rhs));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_linear(coeffs.clone(), v.to_vec(), rel, rhs));
        mix.record(k, l, (&asg, &coeffs, rel, rhs));
    }
    mix.finish();
}

#[test]
fn oracle_all_different() {
    let mut st = 0x3333_4444_5555_6666u64;
    let mut mix = Mix::new("all_different");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let asg: Vec<i32> = if rand_bool(&mut st) { perm(&mut st, n) } else { (0..n).map(|_| rand_in(&mut st, 0, 2)).collect() };
        let k = kernel_accepts(&asg, all_different);
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_all_different(v.to_vec()));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

#[test]
fn oracle_all_different_rows() {
    let mut st = 0x4444_5555_6666_7777u64;
    let mut mix = Mix::new("all_different_rows");
    for _ in 0..ROUNDS {
        let nrows = rand_in(&mut st, 2, 4) as usize;
        let width = 2usize;
        let asg: Vec<i32> = (0..nrows * width).map(|_| rand_in(&mut st, 0, 1)).collect();
        let rows_of = |v: &[VarId]| (0..nrows).map(|r| (0..width).map(|c| v[r * width + c]).collect::<Vec<_>>()).collect::<Vec<_>>();
        let k = kernel_accepts(&asg, |s, v| {
            let rows = rows_of(v);
            for i in 0..rows.len() {
                for j in (i + 1)..rows.len() {
                    let _ = post_tuple_not_equal(s, &rows[i], &rows[j]);
                }
            }
        });
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_all_different_rows(rows_of(v)));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

#[test]
fn oracle_all_different_except() {
    let mut st = 0x5555_6666_7777_8888u64;
    let mut mix = Mix::new("all_different_except");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let except = vec![0i32];
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 2)).collect();
        let k = kernel_accepts(&asg, |s, v| post_all_different_except(s, v, &except));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_all_different_except(v.to_vec(), except.clone()));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

#[test]
fn oracle_all_equal() {
    let mut st = 0x6666_7777_8888_9999u64;
    let mut mix = Mix::new("all_equal");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let asg: Vec<i32> = if rand_bool(&mut st) {
            let v = rand_in(&mut st, 0, 3);
            vec![v; n]
        } else {
            (0..n).map(|_| rand_in(&mut st, 0, 2)).collect()
        };
        let k = kernel_accepts(&asg, all_equal);
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_all_equal(v.to_vec()));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

#[test]
fn oracle_extension() {
    let mut st = 0x7777_8888_9999_aaaau64;
    let mut mix = Mix::new("extension");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 2, 4) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 2)).collect();
        // A pool of random tuples, some with STAR wildcards; sometimes seed the
        // exact assignment so feasible rounds occur.
        let mut tuples: Vec<Vec<i32>> = Vec::new();
        if rand_bool(&mut st) {
            tuples.push(asg.clone());
        }
        for _ in 0..rand_in(&mut st, 1, 4) {
            tuples.push((0..n).map(|_| if rand_bool(&mut st) && rand_bool(&mut st) { STAR } else { rand_in(&mut st, 0, 2) }).collect());
        }
        let k = kernel_accepts(&asg, |s, v| extension(s, v, &tuples, true));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_extension(v.to_vec(), tuples.clone(), true));
        mix.record(k, l, (&asg, &tuples));
    }
    mix.finish();
}

#[test]
fn oracle_neg_extension() {
    let mut st = 0x8888_9999_aaaa_bbbbu64;
    let mut mix = Mix::new("neg_extension");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 2, 4) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 2)).collect();
        let mut tuples: Vec<Vec<i32>> = Vec::new();
        if rand_bool(&mut st) {
            tuples.push(asg.clone());
        }
        for _ in 0..rand_in(&mut st, 1, 4) {
            tuples.push((0..n).map(|_| if rand_bool(&mut st) && rand_bool(&mut st) { STAR } else { rand_in(&mut st, 0, 2) }).collect());
        }
        let k = kernel_accepts(&asg, |s, v| extension(s, v, &tuples, false));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_extension(v.to_vec(), tuples.clone(), false));
        mix.record(k, l, (&asg, &tuples));
    }
    mix.finish();
}

#[test]
fn oracle_lex() {
    let mut st = 0x9999_aaaa_bbbb_ccccu64;
    let mut mix = Mix::new("lex");
    for _ in 0..ROUNDS {
        let nrows = rand_in(&mut st, 2, 3) as usize;
        let width = 2usize;
        let strict = rand_bool(&mut st);
        let asg: Vec<i32> = (0..nrows * width).map(|_| rand_in(&mut st, 0, 1)).collect();
        let rows_of = |v: &[VarId]| (0..nrows).map(|r| (0..width).map(|c| v[r * width + c]).collect::<Vec<_>>()).collect::<Vec<_>>();
        let k = kernel_accepts(&asg, |s, v| lex_chain(s, &rows_of(v), strict));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_lex_chain(rows_of(v), strict));
        mix.record(k, l, (&asg, strict));
    }
    mix.finish();
}

#[test]
fn oracle_count() {
    let mut st = 0xaaaa_bbbb_cccc_ddddu64;
    let mut mix = Mix::new("count");
    let rels = [Relation::Eq, Relation::Ne, Relation::Le, Relation::Lt, Relation::Ge, Relation::Gt];
    for r in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 2)).collect();
        let value = 1i32;
        let rel = rels[r % rels.len()];
        let k = rand_in(&mut st, 0, n as i32) as i64;
        let kern = kernel_accepts(&asg, |s, v| count(s, v, value, rel, k));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_count(v.to_vec(), vec![value], rel, LocalRhs::Const(k)));
        mix.record(kern, l, (&asg, rel, k));
    }
    mix.finish();
}

#[test]
fn oracle_count_allowed() {
    let mut st = 0xbbbb_cccc_dddd_eeeeu64;
    let mut mix = Mix::new("count_allowed");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 2)).collect();
        let value = 1i32;
        // Allowed occurrence counts: a random subset of 0..=n.
        let allowed: Vec<i32> = (0..=n as i32).filter(|_| rand_bool(&mut st)).collect();
        let allowed = if allowed.is_empty() { vec![1] } else { allowed };
        let k = kernel_accepts(&asg, |s, v| {
            let y = s.new_var_set(&allowed);
            count_values_to_var(s, v, &[value], Relation::Eq, y);
        });
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_count_allowed(v.to_vec(), vec![value], allowed.clone()));
        mix.record(k, l, (&asg, &allowed));
    }
    mix.finish();
}

#[test]
fn oracle_n_values() {
    let mut st = 0xcccc_dddd_eeee_ffffu64;
    let mut mix = Mix::new("n_values");
    let rels = [Relation::Eq, Relation::Ne, Relation::Le, Relation::Lt, Relation::Ge, Relation::Gt];
    for r in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 2)).collect();
        let rel = rels[r % rels.len()];
        let k = rand_in(&mut st, 1, n as i32) as i64;
        let kern = kernel_accepts(&asg, |s, v| n_values(s, v, rel, k));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_n_values(v.to_vec(), rel, LocalRhs::Const(k)));
        mix.record(kern, l, (&asg, rel, k));
    }
    mix.finish();
}

#[test]
fn oracle_cardinality() {
    let mut st = 0xdddd_eeee_ffff_0000u64;
    let mut mix = Mix::new("cardinality");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 3)).collect();
        let values = vec![0i32, 1, 2];
        let closed = rand_bool(&mut st);
        let low: Vec<i64> = values.iter().map(|_| rand_in(&mut st, 0, 1) as i64).collect();
        let high: Vec<i64> = low.iter().map(|&lo| lo + rand_in(&mut st, 0, 2) as i64).collect();
        let k = kernel_accepts(&asg, |s, v| cardinality(s, v, &values, &low, &high, closed));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_cardinality(v.to_vec(), values.clone(), low.clone(), high.clone(), closed));
        mix.record(k, l, (&asg, &low, &high, closed));
    }
    mix.finish();
}

#[test]
fn oracle_extremum() {
    let mut st = 0xeeee_ffff_0000_1111u64;
    let mut mix = Mix::new("extremum");
    let rels = [Relation::Eq, Relation::Le, Relation::Ge];
    for r in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 4)).collect();
        let is_min = rand_bool(&mut st);
        let ext = if is_min { *asg.iter().min().unwrap() } else { *asg.iter().max().unwrap() };
        let k = (ext + rand_in(&mut st, -1, 1)) as i64;
        let rel = rels[r % rels.len()];
        let kern = kernel_accepts(&asg, |s, v| {
            let y = s.new_var_set(&[k as i32]);
            if matches!(rel, Relation::Eq) {
                if is_min {
                    minimum(s, y, v);
                } else {
                    maximum(s, y, v);
                }
            } else {
                let lo = v.iter().map(|&x| s.store.min(x)).min().unwrap();
                let hi = v.iter().map(|&x| s.store.max(x)).max().unwrap();
                let aux = s.new_var_range(lo, hi);
                if is_min {
                    minimum(s, aux, v);
                } else {
                    maximum(s, aux, v);
                }
                linear(s, &[1, -1], &[aux, y], rel, 0);
            }
        });
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_extremum(v.to_vec(), is_min, rel, LocalRhs::Const(k)));
        mix.record(kern, l, (&asg, is_min, rel, k));
    }
    mix.finish();
}

#[test]
fn oracle_element_member() {
    let mut st = 0xffff_0000_1111_2222u64;
    let mut mix = Mix::new("element_member");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 2)).collect();
        let value = rand_in(&mut st, 0, 2);
        let k = kernel_accepts(&asg, |s, v| {
            let idx = s.new_var_range(0, n as i32 - 1);
            let val = s.new_var_set(&[value]);
            element(s, v, idx, val);
        });
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_element_member(v.to_vec(), value));
        mix.record(k, l, (&asg, value));
    }
    mix.finish();
}

#[test]
fn oracle_cumulative() {
    let mut st = 0x0000_1111_2222_3333u64;
    let mut mix = Mix::new("cumulative");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 2, 4) as usize;
        // starts are the search variables; durations/heights/cap are constants
        // (as posted by cumulative_v1). Include zero-duration/zero-height tasks.
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 4)).collect();
        let durs: Vec<i64> = (0..n).map(|_| rand_in(&mut st, 0, 3) as i64).collect();
        let hts: Vec<i64> = (0..n).map(|_| rand_in(&mut st, 0, 3) as i64).collect();
        let cap = rand_in(&mut st, 1, 4) as i64;
        let k = kernel_accepts(&asg, |s, v| cumulative(s, v, &durs, &hts, cap));
        let l = ls_feasible(&asg, |s, spec, v| {
            let dv: Vec<VarId> = durs.iter().map(|&d| s.new_var_set(&[d as i32])).collect();
            let hv: Vec<VarId> = hts.iter().map(|&h| s.new_var_set(&[h as i32])).collect();
            spec.add_cumulative(v.to_vec(), dv, hv, LocalRhs::Const(cap));
        });
        mix.record(k, l, (&asg, &durs, &hts, cap));
    }
    mix.finish();
}

#[test]
fn oracle_channel_inverse() {
    let mut st = 0x1111_2222_3333_4444u64;
    let mut mix = Mix::new("channel_inverse");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 4) as usize;
        // Single-list channel (self-inverse) with a start index: feasible iff
        // the assignment is an involution permutation of start..start+n.
        let start = rand_in(&mut st, 0, 2);
        let asg: Vec<i32> = if rand_bool(&mut st) {
            // random involution: pair up or fix points
            let mut p: Vec<i32> = (start..start + n as i32).collect();
            let order = perm(&mut st, n);
            let mut i = 0;
            while i + 1 < n {
                if rand_bool(&mut st) {
                    let (a, b) = (order[i] as usize, order[i + 1] as usize);
                    p[a] = start + b as i32;
                    p[b] = start + a as i32;
                }
                i += 2;
            }
            p
        } else {
            // Includes out-of-range values on both sides of start..start+n so
            // the out-of-range violation branch faces the kernel encoding.
            (0..n).map(|_| rand_in(&mut st, start - 1, start + n as i32)).collect()
        };
        // Production pairing: the offset-aware decomposition posted by the XCSP
        // callback (value bounds + pairwise iff), valid for start == 0 too.
        let k = kernel_accepts(&asg, |s, v| {
            let m = v.len();
            for &x in v {
                linear(s, &[1], &[x], Relation::Ge, start as i64);
                linear(s, &[1], &[x], Relation::Le, start as i64 + m as i64 - 1);
            }
            for (i, &x) in v.iter().enumerate() {
                for (j, &y) in v.iter().enumerate() {
                    let xv = expr::eq(expr::var(x), expr::int(start as i64 + j as i64));
                    let yv = expr::eq(expr::var(y), expr::int(start as i64 + i as i64));
                    intension(s, expr::iff(xv, yv));
                }
            }
        });
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_channel_inverse(v.to_vec(), start, v.to_vec(), start));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

#[test]
fn oracle_channel_inverse_plain_kernel_in_range() {
    // The 0-based fast path (`channel` propagator, what the callback posts for
    // startIndex == 0) against the LS scorer, values kept in range: the plain
    // propagator's out-of-range behavior is not part of the production pairing
    // because the callback guards offsets through the decomposition above.
    let mut st = 0x5151_6262_7373_8484u64;
    let mut mix = Mix::new("channel_inverse_plain");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 4) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, n as i32 - 1)).collect();
        let k = kernel_accepts(&asg, |s, v| channel(s, v, v));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_channel_inverse(v.to_vec(), 0, v.to_vec(), 0));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

#[test]
fn oracle_channel_onehot() {
    let mut st = 0x2222_3333_4444_5555u64;
    let mut mix = Mix::new("channel_onehot");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 4) as usize;
        // xs are 0/1 vars, then a value var giving the index of the single 1.
        let mut asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 1)).collect();
        if rand_bool(&mut st) {
            // seed a valid one-hot so feasible rounds occur
            let idx = rand_in(&mut st, 0, n as i32 - 1) as usize;
            for (i, x) in asg.iter_mut().enumerate() {
                *x = i32::from(i == idx);
            }
            asg.push(idx as i32);
        } else {
            asg.push(rand_in(&mut st, 0, n as i32 - 1));
        }
        let k = kernel_accepts(&asg, |s, v| post_channel_onehot_index(s, &v[..n], v[n], 0));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_channel_onehot(v[..n].to_vec(), v[n], 0));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

#[test]
fn oracle_precedence() {
    let mut st = 0x3333_4444_5555_6666u64;
    let mut mix = Mix::new("precedence");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let values = vec![0i32, 1, 2];
        let covered = rand_bool(&mut st);
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 2)).collect();
        let k = kernel_accepts(&asg, |s, v| precedence_with_covered(s, v, &values, covered));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_precedence(v.to_vec(), values.clone(), covered));
        mix.record(k, l, (&asg, covered));
    }
    mix.finish();
}

#[test]
fn oracle_circuit() {
    let mut st = 0x4444_5555_6666_7777u64;
    let mut mix = Mix::new("circuit");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let asg: Vec<i32> = if rand_bool(&mut st) {
            // a single Hamiltonian cycle: succ[order[i]] = order[i+1]
            let order = perm(&mut st, n);
            let mut succ = vec![0i32; n];
            for i in 0..n {
                succ[order[i] as usize] = order[(i + 1) % n];
            }
            succ
        } else {
            (0..n).map(|_| rand_in(&mut st, 0, n as i32 - 1)).collect()
        };
        let k = kernel_accepts(&asg, circuit);
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_circuit(v.to_vec()));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

#[test]
fn oracle_bin_packing() {
    let mut st = 0x5555_6666_7777_8888u64;
    let mut mix = Mix::new("bin_packing");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let nbins = rand_in(&mut st, 2, 3) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, nbins as i32 - 1)).collect();
        let sizes: Vec<i64> = (0..n).map(|_| rand_in(&mut st, 1, 3) as i64).collect();
        let caps: Vec<i64> = (0..nbins).map(|_| rand_in(&mut st, 2, 6) as i64).collect();
        let k = kernel_accepts(&asg, |s, v| bin_packing(s, v, &sizes, &caps));
        let l = ls_feasible(&asg, |_s, spec, v| {
            spec.add_bin_packing(v.to_vec(), sizes.clone(), caps.iter().copied().map(LocalRhs::Const).collect(), false)
        });
        mix.record(k, l, (&asg, &sizes, &caps));
    }
    mix.finish();
}

#[test]
fn oracle_bin_packing_exact() {
    let mut st = 0x6666_7777_8888_9999u64;
    let mut mix = Mix::new("bin_packing_exact");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let nbins = rand_in(&mut st, 2, 3) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, nbins as i32 - 1)).collect();
        let sizes: Vec<i64> = (0..n).map(|_| rand_in(&mut st, 1, 3) as i64).collect();
        // Actual per-bin loads under asg; perturb some to force infeasibility.
        let mut loads = vec![0i64; nbins];
        for (i, &b) in asg.iter().enumerate() {
            loads[b as usize] += sizes[i];
        }
        for load in &mut loads {
            if rand_bool(&mut st) && rand_bool(&mut st) {
                *load += rand_in(&mut st, -1, 1) as i64;
            }
        }
        let k = kernel_accepts(&asg, |s, v| {
            let loadv: Vec<VarId> = loads.iter().map(|&x| s.new_var_range(x.max(0) as i32, x.max(0) as i32)).collect();
            post_bin_loads(s, v, &sizes, &loadv);
        });
        let l = ls_feasible(&asg, |_s, spec, v| {
            spec.add_bin_packing(v.to_vec(), sizes.clone(), loads.iter().map(|&x| LocalRhs::Const(x.max(0))).collect(), true)
        });
        mix.record(k, l, (&asg, &sizes, &loads));
    }
    mix.finish();
}

#[test]
fn oracle_no_overlap_1d() {
    let mut st = 0x7777_8888_9999_aaaau64;
    let mut mix = Mix::new("no_overlap_1d");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 2, 4) as usize;
        let asg: Vec<i32> = (0..n).map(|_| rand_in(&mut st, 0, 5)).collect();
        let durs: Vec<i64> = (0..n).map(|_| rand_in(&mut st, 1, 3) as i64).collect();
        let origins_of = |v: &[VarId]| v.iter().map(|&x| vec![x]).collect::<Vec<_>>();
        let lengths: Vec<Vec<Expr>> = durs.iter().map(|&d| vec![expr::int(d)]).collect();
        let k = kernel_accepts(&asg, |s, v| no_overlap(s, v, &durs));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_no_overlap(origins_of(v), lengths.clone(), false));
        mix.record(k, l, (&asg, &durs));
    }
    mix.finish();
}

#[test]
fn oracle_no_overlap_2d() {
    let mut st = 0x8888_9999_aaaa_bbbbu64;
    let mut mix = Mix::new("no_overlap_2d");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 2, 3) as usize;
        // Each box has an x and y origin; sizes are constants.
        let asg: Vec<i32> = (0..2 * n).map(|_| rand_in(&mut st, 0, 3)).collect();
        let sizes: Vec<(i64, i64)> = (0..n).map(|_| (rand_in(&mut st, 1, 2) as i64, rand_in(&mut st, 1, 2) as i64)).collect();
        let origins_of = |v: &[VarId]| (0..n).map(|i| vec![v[2 * i], v[2 * i + 1]]).collect::<Vec<_>>();
        let lengths: Vec<Vec<Expr>> = sizes.iter().map(|&(w, h)| vec![expr::int(w), expr::int(h)]).collect();
        let k = kernel_accepts(&asg, |s, v| {
            let _ = post_diffn(s, &origins_of(v), &lengths, false);
        });
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_no_overlap(origins_of(v), lengths.clone(), false));
        mix.record(k, l, (&asg, &sizes));
    }
    mix.finish();
}

#[test]
fn oracle_regular() {
    let mut st = 0x9999_aaaa_bbbb_ccccu64;
    let mut mix = Mix::new("regular");
    // DFA accepting binary strings with an even number of 1s.
    let dfa = Dfa { n_states: 2, start: 0, accept: vec![0], transitions: vec![(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)] };
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        // occasionally emit an out-of-alphabet value (2) to test the no-transition path
        let asg: Vec<i32> = (0..n)
            .map(|_| if rand_bool(&mut st) && rand_bool(&mut st) && rand_bool(&mut st) { 2 } else { rand_in(&mut st, 0, 1) })
            .collect();
        let k = kernel_accepts(&asg, |s, v| regular(s, v, dfa.clone()));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_regular(v.to_vec(), dfa.clone()));
        mix.record(k, l, &asg);
    }
    mix.finish();
}

/// Reduced layered MDD accepting binary strings of length `n` whose sum is
/// exactly `target`; every surviving final-layer node accepts (per the Mdd
/// contract), so dead states are pruned.
fn build_sum_mdd(n: usize, target: usize) -> Mdd {
    let feasible = |l: usize, s: usize| s <= target && target - s <= n - l;
    let states: Vec<Vec<usize>> = (0..=n).map(|l| (0..=l).filter(|&s| feasible(l, s)).collect()).collect();
    let index = |l: usize, s: usize| states[l].iter().position(|&x| x == s);
    let nodes_per_layer: Vec<usize> = states.iter().map(Vec::len).collect();
    let mut layers = Vec::new();
    for (l, layer_states) in states.iter().take(n).enumerate() {
        let mut arcs = Vec::new();
        for (fi, &s) in layer_states.iter().enumerate() {
            for (val, ns) in [(0i32, s), (1, s + 1)] {
                if let Some(ti) = index(l + 1, ns) {
                    arcs.push(MddArc { from: fi, value: val, to: ti });
                }
            }
        }
        layers.push(arcs);
    }
    Mdd { layers, nodes_per_layer }
}

#[test]
fn oracle_mdd() {
    let mut st = 0xaaaa_bbbb_cccc_ddddu64;
    let mut mix = Mix::new("mdd");
    for _ in 0..ROUNDS {
        let n = rand_in(&mut st, 3, 5) as usize;
        let target = rand_in(&mut st, 1, n as i32 - 1) as usize;
        let m = build_sum_mdd(n, target);
        let asg: Vec<i32> = (0..n)
            .map(|_| if rand_bool(&mut st) && rand_bool(&mut st) && rand_bool(&mut st) { 2 } else { rand_in(&mut st, 0, 1) })
            .collect();
        let k = kernel_accepts(&asg, |s, v| mdd(s, v, m.clone()));
        let l = ls_feasible(&asg, |_s, spec, v| spec.add_mdd(v.to_vec(), m.clone()));
        mix.record(k, l, (&asg, target));
    }
    mix.finish();
}
