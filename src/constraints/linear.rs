//! Linear (weighted-sum) constraints. All arithmetic in `i64` so sums of `i32`
//! terms cannot overflow.

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Premise, Solver, Store};

/// Sentinel for "exclude no term" (conflict reason: every term contributes).
const NO_SKIP: usize = usize::MAX;

/// Comparison operator for a linear constraint.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Relation {
    /// \( \sum_i a_i x_i = \texttt{rhs} \)
    Eq,
    /// \( \sum_i a_i x_i \ne \texttt{rhs} \)
    Ne,
    /// \( \sum_i a_i x_i \le \texttt{rhs} \)
    Le,
    /// \( \sum_i a_i x_i < \texttt{rhs} \)
    Lt,
    /// \( \sum_i a_i x_i \ge \texttt{rhs} \)
    Ge,
    /// \( \sum_i a_i x_i > \texttt{rhs} \)
    Gt,
}

/// \( \sum_i a_i x_i \le c \), bounds propagation to a local fixpoint per call.
#[derive(Clone)]
struct LinearLeq {
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    c: i64,
    /// Reused scratch for per-term minimum contributions.
    term_min: Vec<i64>,
}

impl LinearLeq {
    fn new(coeffs: &[i64], vars: &[VarId], c: i64) -> Self {
        debug_assert_eq!(coeffs.len(), vars.len());
        Self { coeffs: coeffs.to_vec(), vars: vars.to_vec(), c, term_min: vec![0; vars.len()] }
    }

    /// Near bound of every term but `skip` (`x_j ≥ min` if `a_j > 0`, else
    /// `x_j ≤ max`); entails `∑_{j≠skip} a_j x_j ≥ sum_min − term_min[skip]`.
    fn min_side(&self, store: &Store, skip: usize) -> Vec<Premise> {
        let mut why = Vec::new();
        for (j, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
            if j == skip || a == 0 {
                continue;
            }
            why.push(if a > 0 { Premise::Ge { var: v, bound: store.min(v) } } else { Premise::Le { var: v, bound: store.max(v) } });
        }
        why
    }
}

impl Propagator for LinearLeq {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        loop {
            let mut sum_min: i64 = 0;
            let mut sum_max: i64 = 0;
            for (slot, (&a, &v)) in self.term_min.iter_mut().zip(self.coeffs.iter().zip(&self.vars)) {
                let lo = store.min(v) as i64;
                let hi = store.max(v) as i64;
                let (tmin, tmax) = if a >= 0 { (a * lo, a * hi) } else { (a * hi, a * lo) };
                *slot = tmin;
                sum_min += tmin;
                sum_max += tmax;
            }

            if sum_min > self.c {
                return Err(store.fail_because(self.min_side(store, NO_SKIP)));
            }
            if sum_max <= self.c {
                return Ok(()); // entailed
            }

            let mut changed = false;
            for (idx, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
                if a == 0 {
                    continue;
                }
                let allowed = self.c - (sum_min - self.term_min[idx]);
                if a > 0 {
                    let bound = clamp_i32(floor_div(allowed, a));
                    let before = store.max(v);
                    if bound < before {
                        store.remove_above_because(v, bound, self.min_side(store, idx))?;
                        changed = true;
                    }
                } else {
                    let bound = clamp_i32(ceil_div(allowed, a));
                    let before = store.min(v);
                    if bound > before {
                        store.remove_below_because(v, bound, self.min_side(store, idx))?;
                        changed = true;
                    }
                }
            }

            if !changed {
                return Ok(());
            }
        }
    }
}

/// \( \sum_i a_i x_i = c \), both-sided bounds propagation to a local fixpoint.
#[derive(Clone)]
struct LinearEq {
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    c: i64,
    term_min: Vec<i64>,
    term_max: Vec<i64>,
}

impl LinearEq {
    fn new(coeffs: &[i64], vars: &[VarId], c: i64) -> Self {
        debug_assert_eq!(coeffs.len(), vars.len());
        Self { coeffs: coeffs.to_vec(), vars: vars.to_vec(), c, term_min: vec![0; vars.len()], term_max: vec![0; vars.len()] }
    }

    /// Near bound of every term but `skip`; entails
    /// `∑_{j≠skip} a_j x_j ≥ sum_min − term_min[skip]`.
    fn min_side(&self, store: &Store, skip: usize) -> Vec<Premise> {
        let mut why = Vec::new();
        for (j, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
            if j == skip || a == 0 {
                continue;
            }
            why.push(if a > 0 { Premise::Ge { var: v, bound: store.min(v) } } else { Premise::Le { var: v, bound: store.max(v) } });
        }
        why
    }

    /// Far bound of every term but `skip`; entails
    /// `∑_{j≠skip} a_j x_j ≤ sum_max − term_max[skip]`.
    fn max_side(&self, store: &Store, skip: usize) -> Vec<Premise> {
        let mut why = Vec::new();
        for (j, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
            if j == skip || a == 0 {
                continue;
            }
            why.push(if a > 0 { Premise::Le { var: v, bound: store.max(v) } } else { Premise::Ge { var: v, bound: store.min(v) } });
        }
        why
    }
}

impl Propagator for LinearEq {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        loop {
            let mut sum_min: i64 = 0;
            let mut sum_max: i64 = 0;
            for (i, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
                let lo = store.min(v) as i64;
                let hi = store.max(v) as i64;
                let (tmin, tmax) = if a >= 0 { (a * lo, a * hi) } else { (a * hi, a * lo) };
                self.term_min[i] = tmin;
                self.term_max[i] = tmax;
                sum_min += tmin;
                sum_max += tmax;
            }
            if sum_min > self.c {
                return Err(store.fail_because(self.min_side(store, NO_SKIP)));
            }
            if sum_max < self.c {
                return Err(store.fail_because(self.max_side(store, NO_SKIP)));
            }

            let mut changed = false;
            for (i, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
                if a == 0 {
                    continue;
                }
                // a_i*x_i in [tlo, thi]; sign of a_i decides x_i's lo vs hi bound.
                let tlo = self.c - (sum_max - self.term_max[i]);
                let thi = self.c - (sum_min - self.term_min[i]);
                let (lo, hi) = if a > 0 { (ceil_div(tlo, a), floor_div(thi, a)) } else { (ceil_div(thi, a), floor_div(tlo, a)) };
                let before_min = store.min(v);
                let before_max = store.max(v);
                let lo = clamp_i32(lo);
                let hi = clamp_i32(hi);
                if lo > before_min {
                    let why = if a > 0 { self.max_side(store, i) } else { self.min_side(store, i) };
                    store.remove_below_because(v, lo, why)?;
                    changed = true;
                }
                if hi < before_max {
                    let why = if a > 0 { self.min_side(store, i) } else { self.max_side(store, i) };
                    store.remove_above_because(v, hi, why)?;
                    changed = true;
                }
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

/// \( \sum_i a_i x_i \ne c \). Weak: only filters when at most one variable is still free.
#[derive(Clone)]
struct LinearNeq {
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    c: i64,
}

impl LinearNeq {
    fn new(coeffs: &[i64], vars: &[VarId], c: i64) -> Self {
        debug_assert_eq!(coeffs.len(), vars.len());
        Self { coeffs: coeffs.to_vec(), vars: vars.to_vec(), c }
    }
}

impl Propagator for LinearNeq {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::Fix);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let mut fixed_sum: i64 = 0;
        let mut free: Option<usize> = None;
        let mut free_count = 0usize;
        for (idx, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
            if store.is_fixed(v) {
                fixed_sum += a * store.value(v) as i64;
            } else {
                free = Some(idx);
                free_count += 1;
            }
        }

        match free_count {
            0 => {
                if fixed_sum == self.c {
                    return Err(Inconsistency);
                }
            }
            1 => {
                let idx = free.unwrap();
                let a = self.coeffs[idx];
                if a != 0 {
                    // a*x = c - fixed_sum: forbid x only if the quotient is integral.
                    let rem = self.c - fixed_sum;
                    if rem % a == 0 {
                        let forbidden = rem / a;
                        if (i32::MIN as i64..=i32::MAX as i64).contains(&forbidden) {
                            store.remove(self.vars[idx], forbidden as i32)?;
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Floor of `a / b` for integers (Rust `/` truncates toward zero).
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Ceiling of `a / b` for integers.
fn ceil_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) == (b < 0)) {
        q + 1
    } else {
        q
    }
}

/// Saturate an `i64` bound into the `i32` domain range.
pub(crate) fn clamp_i32(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn neg(coeffs: &[i64]) -> Vec<i64> {
    coeffs.iter().map(|&a| -a).collect()
}

fn post_leq(solver: &mut Solver, coeffs: &[i64], vars: &[VarId], c: i64) {
    solver.post(Box::new(LinearLeq::new(coeffs, vars, c)));
}

/// Post \( \sum_i \texttt{coeffs}[i] \cdot \texttt{vars}[i] \;\texttt{rel}\; \texttt{rhs} \).
pub fn linear(solver: &mut Solver, coeffs: &[i64], vars: &[VarId], rel: Relation, rhs: i64) {
    assert_eq!(coeffs.len(), vars.len(), "linear: coeffs/vars length mismatch");
    match rel {
        Relation::Le => post_leq(solver, coeffs, vars, rhs),
        Relation::Lt => post_leq(solver, coeffs, vars, rhs - 1),
        Relation::Ge => post_leq(solver, &neg(coeffs), vars, -rhs),
        Relation::Gt => post_leq(solver, &neg(coeffs), vars, -(rhs + 1)),
        Relation::Eq => {
            solver.post(Box::new(LinearEq::new(coeffs, vars, rhs)));
        }
        Relation::Ne => {
            solver.post(Box::new(LinearNeq::new(coeffs, vars, rhs)));
        }
    }
}

/// Post \( \sum_i \texttt{vars}[i] \;\texttt{rel}\; \texttt{rhs} \) (all coefficients 1).
pub fn sum(solver: &mut Solver, vars: &[VarId], rel: Relation, rhs: i64) {
    let coeffs = vec![1i64; vars.len()];
    linear(solver, &coeffs, vars, rel, rhs);
}
