//! Counting constraints.
//!
//! Phase 1 ships `count`: the number of variables equal to a target value,
//! compared against `k`. The reasoning is the bounds form of a single-value
//! cardinality: `lb` variables are already pinned to the value, `ub` could
//! still take it. `nValues` and the full Régin-flow `cardinality` (GCC) land in
//! Phase 3.

use crate::constraints::linear::Relation;
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Solver, Store};

/// `#{ i : vars[i] = value }  rel  k`.
struct Count {
    vars: Vec<VarId>,
    value: i32,
    rel: Relation,
    k: i64,
}

impl Count {
    /// `(lb, ub)`: variables definitely equal to `value`, and variables that
    /// could still equal it.
    fn counts(&self, store: &Store) -> (i64, i64) {
        let mut lb = 0;
        let mut ub = 0;
        for &v in &self.vars {
            if store.contains(v, self.value) {
                ub += 1;
                if store.is_fixed(v) {
                    lb += 1;
                }
            }
        }
        (lb, ub)
    }

    /// Enforce \( \texttt{count} \le k \): if the floor already reaches `k`, no other variable
    /// may take the value.
    fn enforce_le(&self, store: &mut Store, k: i64) -> Result<(), Inconsistency> {
        let (lb, _) = self.counts(store);
        if lb > k {
            return Err(Inconsistency);
        }
        if lb == k {
            for &v in &self.vars {
                if !store.is_fixed(v) && store.contains(v, self.value) {
                    store.remove(v, self.value)?;
                }
            }
        }
        Ok(())
    }

    /// Enforce \( \texttt{count} \ge k \): if the ceiling only just reaches `k`, every variable
    /// that can take the value must.
    fn enforce_ge(&self, store: &mut Store, k: i64) -> Result<(), Inconsistency> {
        let (_, ub) = self.counts(store);
        if ub < k {
            return Err(Inconsistency);
        }
        if ub == k {
            for &v in &self.vars {
                if store.contains(v, self.value) && !store.is_fixed(v) {
                    store.fix(v, self.value)?;
                }
            }
        }
        Ok(())
    }
}

impl Propagator for Count {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        match self.rel {
            Relation::Le => self.enforce_le(store, self.k)?,
            Relation::Lt => self.enforce_le(store, self.k - 1)?,
            Relation::Ge => self.enforce_ge(store, self.k)?,
            Relation::Gt => self.enforce_ge(store, self.k + 1)?,
            Relation::Eq => {
                self.enforce_le(store, self.k)?;
                self.enforce_ge(store, self.k)?;
            }
            Relation::Ne => {
                let (lb, ub) = self.counts(store);
                if lb == ub && lb == self.k {
                    return Err(Inconsistency);
                }
            }
        }
        Ok(())
    }
}

/// Post `#{ i : vars[i] = value }  rel  k`.
pub fn count(solver: &mut Solver, vars: &[VarId], value: i32, rel: Relation, k: i64) {
    solver.post(Box::new(Count {
        vars: vars.to_vec(),
        value,
        rel,
        k,
    }));
}

// ===========================================================================
// cardinality (GCC, bounds — decomposed into per-value count ranges)
// ===========================================================================

/// Restrict every variable's domain to a fixed set of values (for closed GCC).
struct InSet {
    vars: Vec<VarId>,
    set: Vec<i32>,
    buf: Vec<i32>,
}

impl Propagator for InSet {
    fn register(&mut self, _store: &mut Store, _me: PropId) {}

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        for &v in &self.vars {
            self.buf.clear();
            self.buf.extend(store.values(v));
            for &val in &self.buf {
                if !self.set.contains(&val) {
                    store.remove(v, val)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `cardinality`: for each `values[j]`, the number of variables equal to it
/// lies in `[low[j], high[j]]`. When `closed`, every variable must take a value
/// from `values`.
///
/// This is the bounds form, decomposed into independent `count` ranges (correct;
/// the Régin-flow GCC is a Phase 6 upgrade).
pub fn cardinality(
    solver: &mut Solver,
    vars: &[VarId],
    values: &[i32],
    low: &[i64],
    high: &[i64],
    closed: bool,
) {
    assert_eq!(values.len(), low.len(), "cardinality: values/low mismatch");
    assert_eq!(
        values.len(),
        high.len(),
        "cardinality: values/high mismatch"
    );
    for (j, &val) in values.iter().enumerate() {
        count(solver, vars, val, Relation::Ge, low[j]);
        count(solver, vars, val, Relation::Le, high[j]);
    }
    if closed {
        solver.post(Box::new(InSet {
            vars: vars.to_vec(),
            set: values.to_vec(),
            buf: Vec::new(),
        }));
    }
}

// ===========================================================================
// nValues
// ===========================================================================

/// `#{ distinct values among vars }  rel  k`.
///
/// Sound bounds only (full filtering is genuinely hard): `lb` = distinct values
/// among the fixed variables, `ub` = `min(n, |union of domains|)`. At a full
/// assignment `lb = ub` = the exact count, so the disentailment check is also
/// the leaf check.
struct NValues {
    vars: Vec<VarId>,
    rel: Relation,
    k: i64,
    fixed: Vec<i32>,
    union: Vec<i32>,
}

impl Propagator for NValues {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        self.fixed.clear();
        self.union.clear();
        for &v in &self.vars {
            if store.is_fixed(v) {
                self.fixed.push(store.value(v));
            }
            self.union.extend(store.values(v));
        }
        self.fixed.sort_unstable();
        self.fixed.dedup();
        self.union.sort_unstable();
        self.union.dedup();

        let n = self.vars.len();
        let lb = self.fixed.len() as i64; // >= this many distinct values for sure
        let ub = (n.min(self.union.len())) as i64; // at most this many

        let infeasible = match self.rel {
            Relation::Ge => ub < self.k,
            Relation::Gt => ub <= self.k,
            Relation::Le => lb > self.k,
            Relation::Lt => lb >= self.k,
            Relation::Eq => ub < self.k || lb > self.k,
            Relation::Ne => lb == ub && lb == self.k,
        };
        if infeasible {
            return Err(Inconsistency);
        }
        Ok(())
    }
}

/// Post `#{ distinct values among vars }  rel  k`.
pub fn n_values(solver: &mut Solver, vars: &[VarId], rel: Relation, k: i64) {
    solver.post(Box::new(NValues {
        vars: vars.to_vec(),
        rel,
        k,
        fixed: Vec::new(),
        union: Vec::new(),
    }));
}
