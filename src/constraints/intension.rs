//! `intension`: universal fallback for any relational/logical [`Expr`].
//! Filtering: disentailment by bounds, probing each unfixed value, then an exact
//! check once all vars are fixed (the correctness safety net for weak bounds).

use std::sync::atomic::{AtomicBool, Ordering};

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

    fn register_until(&mut self, store: &mut Store, me: PropId, should_stop: &dyn Fn() -> bool) -> bool {
        for &variable in &self.vars {
            if should_stop() {
                return false;
            }
            store.subscribe(variable, me, Event::DomainChange);
        }
        !should_stop()
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
    let stop = AtomicBool::new(false);
    assert!(intension_interruptible(solver, expr, &stop));
}

/// Post an intension expression while polling during variable collection and
/// propagator registration. A `false` result requires discarding `solver`.
pub(crate) fn intension_interruptible(solver: &mut Solver, expr: Expr, stop: &AtomicBool) -> bool {
    if stop.load(Ordering::Acquire) {
        return false;
    }
    #[cfg(feature = "lp-relaxation")]
    if let Some(row) = expr.linear_relation_interruptible(stop) {
        solver.record_linear_relaxation(&row.coefficients, &row.variables, row.lower, row.upper);
    }
    let mut vars = Vec::new();
    if !collect_vars_interruptible(&expr, &mut vars, stop) {
        return false;
    }
    vars.sort_unstable();
    if stop.load(Ordering::Acquire) {
        return false;
    }
    vars.dedup();
    let should_stop = || stop.load(Ordering::Acquire);
    solver.post_until(Box::new(Intension { expr, vars, scratch: Vec::new() }), &should_stop).is_some()
}

fn collect_vars_interruptible(expr: &Expr, output: &mut Vec<VarId>, stop: &AtomicBool) -> bool {
    if stop.load(Ordering::Acquire) {
        return false;
    }
    match expr {
        Expr::Const(_) => {}
        Expr::Var(variable) => output.push(*variable),
        Expr::Neg(value) | Expr::Abs(value) | Expr::Not(value) => {
            if !collect_vars_interruptible(value, output, stop) {
                return false;
            }
        }
        Expr::Add(values) | Expr::Mul(values) | Expr::Min(values) | Expr::Max(values) | Expr::And(values) | Expr::Or(values) => {
            for value in values {
                if !collect_vars_interruptible(value, output, stop) {
                    return false;
                }
            }
        }
        Expr::Sub(left, right)
        | Expr::Div(left, right)
        | Expr::Mod(left, right)
        | Expr::Eq(left, right)
        | Expr::Ne(left, right)
        | Expr::Lt(left, right)
        | Expr::Le(left, right)
        | Expr::Gt(left, right)
        | Expr::Ge(left, right)
        | Expr::Imp(left, right)
        | Expr::Iff(left, right) => {
            if !collect_vars_interruptible(left, output, stop) || !collect_vars_interruptible(right, output, stop) {
                return false;
            }
        }
        Expr::IfThenElse(condition, then_value, else_value) => {
            if !collect_vars_interruptible(condition, output, stop)
                || !collect_vars_interruptible(then_value, output, stop)
                || !collect_vars_interruptible(else_value, output, stop)
            {
                return false;
            }
        }
    }
    true
}
