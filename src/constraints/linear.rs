//! Linear (weighted-sum) constraints — the workhorse, reused throughout the
//! catalogue.
//!
//! The core propagator is `Σ aᵢ·xᵢ ≤ c` with bounds propagation. Everything
//! else is built from it:
//!
//! - `≥` / `>` / `<` are sign/offset rewrites of `≤`.
//! - `=` posts the constraint and its negation (`≤ c` and `≥ c`).
//! - `≠` gets its own (weak) propagator that only fires near a full assignment.
//!
//! All arithmetic is done in `i64` so sums of `i32` terms cannot overflow.

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Solver, Store};

/// Comparison operator for a linear constraint.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Relation {
    /// `Σ aᵢ·xᵢ = rhs`
    Eq,
    /// `Σ aᵢ·xᵢ ≠ rhs`
    Ne,
    /// `Σ aᵢ·xᵢ ≤ rhs`
    Le,
    /// `Σ aᵢ·xᵢ < rhs`
    Lt,
    /// `Σ aᵢ·xᵢ ≥ rhs`
    Ge,
    /// `Σ aᵢ·xᵢ > rhs`
    Gt,
}

/// `Σ aᵢ·xᵢ ≤ c`, with bounds propagation run to a local fixpoint per call.
struct LinearLeq {
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    c: i64,
    /// Reused scratch for per-term minimum contributions (no per-call alloc).
    term_min: Vec<i64>,
}

impl LinearLeq {
    fn new(coeffs: &[i64], vars: &[VarId], c: i64) -> Self {
        debug_assert_eq!(coeffs.len(), vars.len());
        Self {
            coeffs: coeffs.to_vec(),
            vars: vars.to_vec(),
            c,
            term_min: vec![0; vars.len()],
        }
    }
}

impl Propagator for LinearLeq {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        // Iterate to a local fixpoint: pruning one variable tightens the slack
        // for the others, and self-changes do not re-enqueue this propagator.
        loop {
            let mut sum_min: i64 = 0;
            let mut sum_max: i64 = 0;
            for (slot, (&a, &v)) in self
                .term_min
                .iter_mut()
                .zip(self.coeffs.iter().zip(&self.vars))
            {
                let lo = store.min(v) as i64;
                let hi = store.max(v) as i64;
                let (tmin, tmax) = if a >= 0 {
                    (a * lo, a * hi)
                } else {
                    (a * hi, a * lo)
                };
                *slot = tmin;
                sum_min += tmin;
                sum_max += tmax;
            }

            if sum_min > self.c {
                return Err(Inconsistency);
            }
            if sum_max <= self.c {
                return Ok(()); // entailed: no value can violate the bound
            }

            let mut changed = false;
            for (idx, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
                if a == 0 {
                    continue;
                }
                // aᵢ·xᵢ ≤ c − (sum_min − term_minᵢ)
                let allowed = self.c - (sum_min - self.term_min[idx]);
                if a > 0 {
                    let bound = clamp_i32(floor_div(allowed, a));
                    let before = store.max(v);
                    store.remove_above(v, bound)?;
                    changed |= store.max(v) != before;
                } else {
                    let bound = clamp_i32(ceil_div(allowed, a));
                    let before = store.min(v);
                    store.remove_below(v, bound)?;
                    changed |= store.min(v) != before;
                }
            }

            if !changed {
                return Ok(());
            }
        }
    }
}

/// `Σ aᵢ·xᵢ ≠ c`. Weak: only filters when at most one variable is still free.
struct LinearNeq {
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    c: i64,
}

impl LinearNeq {
    fn new(coeffs: &[i64], vars: &[VarId], c: i64) -> Self {
        debug_assert_eq!(coeffs.len(), vars.len());
        Self {
            coeffs: coeffs.to_vec(),
            vars: vars.to_vec(),
            c,
        }
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
                    // a·x = c − fixed_sum  =>  x forbidden only if it is integral.
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
fn clamp_i32(x: i64) -> i32 {
    x.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn neg(coeffs: &[i64]) -> Vec<i64> {
    coeffs.iter().map(|&a| -a).collect()
}

fn post_leq(solver: &mut Solver, coeffs: &[i64], vars: &[VarId], c: i64) {
    solver.post(Box::new(LinearLeq::new(coeffs, vars, c)));
}

/// Post `Σ coeffs[i]·vars[i]  rel  rhs`.
pub fn linear(solver: &mut Solver, coeffs: &[i64], vars: &[VarId], rel: Relation, rhs: i64) {
    assert_eq!(
        coeffs.len(),
        vars.len(),
        "linear: coeffs/vars length mismatch"
    );
    match rel {
        Relation::Le => post_leq(solver, coeffs, vars, rhs),
        Relation::Lt => post_leq(solver, coeffs, vars, rhs - 1),
        Relation::Ge => post_leq(solver, &neg(coeffs), vars, -rhs),
        Relation::Gt => post_leq(solver, &neg(coeffs), vars, -(rhs + 1)),
        Relation::Eq => {
            post_leq(solver, coeffs, vars, rhs);
            post_leq(solver, &neg(coeffs), vars, -rhs);
        }
        Relation::Ne => {
            solver.post(Box::new(LinearNeq::new(coeffs, vars, rhs)));
        }
    }
}

/// Post `Σ vars[i]  rel  rhs` (all coefficients 1).
pub fn sum(solver: &mut Solver, vars: &[VarId], rel: Relation, rhs: i64) {
    let coeffs = vec![1i64; vars.len()];
    linear(solver, &coeffs, vars, rel, rhs);
}
