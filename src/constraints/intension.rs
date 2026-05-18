//! `intension`: universal fallback for any relational/logical [`Expr`].
//! Filtering: disentailment by bounds, probing each unfixed value, then an exact
//! check once all vars are fixed (the correctness safety net for weak bounds).

use crate::expr::Expr;
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Solver, Store};

#[derive(Clone)]
struct Intension {
    expr: Expr,
    vars: Vec<VarId>,
    /// Reused scratch for value snapshots during probing.
    scratch: Vec<i32>,
}

impl Propagator for Intension {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let (lo, hi) = {
            let dom = |x: VarId| (store.min(x) as i64, store.max(x) as i64);
            self.expr.bounds(&dom)
        };
        if hi < 1 {
            return Err(Inconsistency); // root can never be true
        }
        if lo >= 1 {
            return Ok(()); // entailed
        }

        // Probe each unfixed variable: remove values that force the root false.
        for &v in &self.vars {
            if store.is_fixed(v) {
                continue;
            }
            self.scratch.clear();
            self.scratch.extend(store.values(v));
            for &val in &self.scratch {
                let dead = {
                    let dom = |x: VarId| {
                        if x == v {
                            (val as i64, val as i64)
                        } else {
                            (store.min(x) as i64, store.max(x) as i64)
                        }
                    };
                    self.expr.bounds(&dom).1 < 1
                };
                if dead {
                    store.remove(v, val)?;
                }
            }
        }

        // Exact check once everything is fixed.
        if self.vars.iter().all(|&x| store.is_fixed(x)) {
            let assigned = self.expr.eval(&|x| store.value(x) as i64);
            if !matches!(assigned, Some(n) if n != 0) {
                return Err(Inconsistency);
            }
        }
        Ok(())
    }
}

/// Post the constraint "`expr` holds" (evaluates non-zero/true).
pub fn intension(solver: &mut Solver, expr: Expr) {
    let mut vars = Vec::new();
    expr.collect_vars(&mut vars);
    vars.sort_unstable();
    vars.dedup();
    solver.post(Box::new(Intension {
        expr,
        vars,
        scratch: Vec::new(),
    }));
}
