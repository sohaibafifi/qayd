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
    {
        record_convex_relaxation(solver, &expr, stop);
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

#[cfg(feature = "lp-relaxation")]
fn record_convex_relaxation(solver: &mut Solver, expression: &Expr, stop: &AtomicBool) {
    if stop.load(Ordering::Acquire) {
        return;
    }
    if let Some(row) = expression.linear_relation_interruptible(stop) {
        solver.record_linear_relaxation(&row.coefficients, &row.variables, row.lower, row.upper);
        return;
    }
    match expression {
        Expr::And(terms) => {
            for term in terms {
                record_convex_relaxation(solver, term, stop);
            }
        }
        Expr::Or(terms) => {
            let literals = terms.iter().map(|term| boolean_literal(&solver.store, term)).collect::<Option<Vec<_>>>();
            if let Some(literals) = literals {
                record_clause(solver, &literals);
            }
        }
        Expr::Imp(antecedent, consequent) => {
            if let (Some(antecedent), Some(consequent)) =
                (boolean_literal(&solver.store, antecedent), boolean_literal(&solver.store, consequent))
            {
                record_clause(solver, &[(antecedent.0, !antecedent.1), consequent]);
            }
        }
        Expr::Iff(left, right) => {
            if !record_reified_boolean(solver, left, right)
                && !record_reified_boolean(solver, right, left)
                && !record_reified_comparison(solver, left, right, stop)
                && !record_reified_comparison(solver, right, left, stop)
            {
                record_literal_equivalence(solver, left, right);
            }
        }
        Expr::Eq(left, right) => {
            if !record_reified_boolean(solver, left, right)
                && !record_reified_boolean(solver, right, left)
                && !record_reified_comparison(solver, left, right, stop)
            {
                record_reified_comparison(solver, right, left, stop);
            }
        }
        Expr::Const(_)
        | Expr::Var(_)
        | Expr::Neg(_)
        | Expr::Abs(_)
        | Expr::Add(_)
        | Expr::Sub(_, _)
        | Expr::Mul(_)
        | Expr::Div(_, _)
        | Expr::Mod(_, _)
        | Expr::Min(_)
        | Expr::Max(_)
        | Expr::Ne(_, _)
        | Expr::Lt(_, _)
        | Expr::Le(_, _)
        | Expr::Gt(_, _)
        | Expr::Ge(_, _)
        | Expr::Not(_)
        | Expr::IfThenElse(_, _, _) => {}
    }
}

#[cfg(feature = "lp-relaxation")]
fn is_boolean(store: &Store, variable: VarId) -> bool {
    store.min(variable) >= 0 && store.max(variable) <= 1
}

#[cfg(feature = "lp-relaxation")]
fn boolean_literal(store: &Store, expression: &Expr) -> Option<(VarId, bool)> {
    match expression {
        Expr::Var(variable) if is_boolean(store, *variable) => Some((*variable, true)),
        Expr::Not(value) => boolean_literal(store, value).map(|(variable, positive)| (variable, !positive)),
        Expr::Eq(left, right) => boolean_equality_literal(store, left, right, false),
        Expr::Ne(left, right) => boolean_equality_literal(store, left, right, true),
        Expr::Ge(left, right) => boolean_threshold_literal(store, left, right, 1, true),
        Expr::Gt(left, right) => boolean_threshold_literal(store, left, right, 0, true),
        Expr::Le(left, right) => boolean_threshold_literal(store, left, right, 0, false),
        Expr::Lt(left, right) => boolean_threshold_literal(store, left, right, 1, false),
        _ => None,
    }
}

#[cfg(feature = "lp-relaxation")]
fn boolean_equality_literal(store: &Store, left: &Expr, right: &Expr, negated: bool) -> Option<(VarId, bool)> {
    let (variable, value) = match (left, right) {
        (Expr::Var(variable), Expr::Const(value)) | (Expr::Const(value), Expr::Var(variable)) => (*variable, *value),
        _ => return None,
    };
    if !is_boolean(store, variable) || !matches!(value, 0 | 1) {
        return None;
    }
    Some((variable, (value == 1) != negated))
}

#[cfg(feature = "lp-relaxation")]
fn boolean_threshold_literal(store: &Store, left: &Expr, right: &Expr, threshold: i64, positive: bool) -> Option<(VarId, bool)> {
    let (Expr::Var(variable), Expr::Const(value)) = (left, right) else {
        return None;
    };
    if !is_boolean(store, *variable) {
        return None;
    }
    (*value == threshold).then_some((*variable, positive))
}

#[cfg(feature = "lp-relaxation")]
fn record_clause(solver: &mut Solver, literals: &[(VarId, bool)]) {
    let mut coefficients = Vec::with_capacity(literals.len());
    let mut variables = Vec::with_capacity(literals.len());
    let mut negative = 0i64;
    for &(variable, positive) in literals {
        coefficients.push(if positive { 1 } else { -1 });
        variables.push(variable);
        negative += i64::from(!positive);
    }
    solver.record_linear_relaxation(&coefficients, &variables, Some(1 - negative), None);
}

#[cfg(feature = "lp-relaxation")]
fn record_literal_equivalence(solver: &mut Solver, left: &Expr, right: &Expr) {
    let (Some(left), Some(right)) = (boolean_literal(&solver.store, left), boolean_literal(&solver.store, right)) else {
        return;
    };
    record_clause(solver, &[(left.0, !left.1), right]);
    record_clause(solver, &[left, (right.0, !right.1)]);
}

#[cfg(feature = "lp-relaxation")]
fn record_reified_boolean(solver: &mut Solver, selector: &Expr, expression: &Expr) -> bool {
    let Expr::Var(selector) = selector else {
        return false;
    };
    if !is_boolean(&solver.store, *selector) {
        return false;
    }
    if let Some(literal) = boolean_literal(&solver.store, expression) {
        record_clause(solver, &[(*selector, false), literal]);
        record_clause(solver, &[(*selector, true), (literal.0, !literal.1)]);
        return true;
    }
    match expression {
        Expr::Or(terms) => {
            let Some(literals) = terms.iter().map(|term| boolean_literal(&solver.store, term)).collect::<Option<Vec<_>>>() else {
                return false;
            };
            for &literal in &literals {
                record_clause(solver, &[(literal.0, !literal.1), (*selector, true)]);
            }
            let mut clause = Vec::with_capacity(literals.len() + 1);
            clause.push((*selector, false));
            clause.extend(literals);
            record_clause(solver, &clause);
            true
        }
        Expr::And(terms) => {
            let Some(literals) = terms.iter().map(|term| boolean_literal(&solver.store, term)).collect::<Option<Vec<_>>>() else {
                return false;
            };
            for &literal in &literals {
                record_clause(solver, &[(*selector, false), literal]);
            }
            let mut clause = Vec::with_capacity(literals.len() + 1);
            clause.push((*selector, true));
            clause.extend(literals.into_iter().map(|(variable, positive)| (variable, !positive)));
            record_clause(solver, &clause);
            true
        }
        Expr::Imp(antecedent, consequent) => {
            let (Some(antecedent), Some(consequent)) =
                (boolean_literal(&solver.store, antecedent), boolean_literal(&solver.store, consequent))
            else {
                return false;
            };
            let disjunction = [(antecedent.0, !antecedent.1), consequent];
            for &literal in &disjunction {
                record_clause(solver, &[(literal.0, !literal.1), (*selector, true)]);
            }
            record_clause(solver, &[(*selector, false), disjunction[0], disjunction[1]]);
            true
        }
        Expr::Iff(left, right) => {
            let (Some(left), Some(right)) = (boolean_literal(&solver.store, left), boolean_literal(&solver.store, right)) else {
                return false;
            };
            let left_false = (left.0, !left.1);
            let right_false = (right.0, !right.1);
            record_clause(solver, &[(*selector, false), left_false, right]);
            record_clause(solver, &[(*selector, false), left, right_false]);
            record_clause(solver, &[(*selector, true), left, right]);
            record_clause(solver, &[(*selector, true), left_false, right_false]);
            true
        }
        _ => false,
    }
}

#[cfg(feature = "lp-relaxation")]
fn record_reified_comparison(solver: &mut Solver, selector: &Expr, predicate: &Expr, stop: &AtomicBool) -> bool {
    let Expr::Var(selector) = selector else {
        return false;
    };
    if !is_boolean(&solver.store, *selector) {
        return false;
    }
    let Some(row) = predicate.linear_relation_interruptible(stop) else {
        return false;
    };
    let Some((activity_lower, activity_upper)) = row_activity_bounds(&solver.store, &row.coefficients, &row.variables) else {
        return false;
    };
    if let Some(lower) = row.lower {
        record_lower_implication(solver, &row.coefficients, &row.variables, *selector, lower, activity_lower);
    }
    if let Some(upper) = row.upper {
        record_upper_implication(solver, &row.coefficients, &row.variables, *selector, upper, activity_upper);
    }
    match (row.lower, row.upper) {
        (None, Some(upper)) => {
            if let Some(complement) = upper.checked_add(1) {
                record_false_lower_implication(solver, &row.coefficients, &row.variables, *selector, complement, activity_lower);
            }
        }
        (Some(lower), None) => {
            if let Some(complement) = lower.checked_sub(1) {
                record_false_upper_implication(solver, &row.coefficients, &row.variables, *selector, complement, activity_upper);
            }
        }
        (Some(_), Some(_)) | (None, None) => {}
    }
    true
}

#[cfg(feature = "lp-relaxation")]
fn row_activity_bounds(store: &Store, coefficients: &[i64], variables: &[VarId]) -> Option<(i64, i64)> {
    let mut lower = 0i128;
    let mut upper = 0i128;
    for (&coefficient, &variable) in coefficients.iter().zip(variables) {
        let coefficient = i128::from(coefficient);
        let endpoints = (i128::from(store.min(variable)), i128::from(store.max(variable)));
        let (term_lower, term_upper) = if coefficient >= 0 {
            (coefficient.checked_mul(endpoints.0)?, coefficient.checked_mul(endpoints.1)?)
        } else {
            (coefficient.checked_mul(endpoints.1)?, coefficient.checked_mul(endpoints.0)?)
        };
        lower = lower.checked_add(term_lower)?;
        upper = upper.checked_add(term_upper)?;
    }
    Some((i64::try_from(lower).ok()?, i64::try_from(upper).ok()?))
}

#[cfg(feature = "lp-relaxation")]
fn append_selector(coefficients: &[i64], variables: &[VarId], selector: VarId, coefficient: i128) -> Option<(Vec<i64>, Vec<VarId>)> {
    let coefficient = i64::try_from(coefficient).ok()?;
    let mut output_coefficients = coefficients.to_vec();
    let mut output_variables = variables.to_vec();
    output_coefficients.push(coefficient);
    output_variables.push(selector);
    Some((output_coefficients, output_variables))
}

#[cfg(feature = "lp-relaxation")]
fn record_lower_implication(solver: &mut Solver, coefficients: &[i64], variables: &[VarId], selector: VarId, bound: i64, minimum: i64) {
    let gap = i128::from(bound) - i128::from(minimum);
    if gap <= 0 {
        return;
    }
    if let Some((coefficients, variables)) = append_selector(coefficients, variables, selector, -gap) {
        solver.record_linear_relaxation(&coefficients, &variables, Some(minimum), None);
    }
}

#[cfg(feature = "lp-relaxation")]
fn record_upper_implication(solver: &mut Solver, coefficients: &[i64], variables: &[VarId], selector: VarId, bound: i64, maximum: i64) {
    let gap = i128::from(maximum) - i128::from(bound);
    if gap <= 0 {
        return;
    }
    if let Some((coefficients, variables)) = append_selector(coefficients, variables, selector, gap) {
        solver.record_linear_relaxation(&coefficients, &variables, None, Some(maximum));
    }
}

#[cfg(feature = "lp-relaxation")]
fn record_false_lower_implication(
    solver: &mut Solver,
    coefficients: &[i64],
    variables: &[VarId],
    selector: VarId,
    bound: i64,
    minimum: i64,
) {
    let gap = i128::from(bound) - i128::from(minimum);
    if gap <= 0 {
        return;
    }
    if let Some((coefficients, variables)) = append_selector(coefficients, variables, selector, gap) {
        solver.record_linear_relaxation(&coefficients, &variables, Some(bound), None);
    }
}

#[cfg(feature = "lp-relaxation")]
fn record_false_upper_implication(
    solver: &mut Solver,
    coefficients: &[i64],
    variables: &[VarId],
    selector: VarId,
    bound: i64,
    maximum: i64,
) {
    let gap = i128::from(maximum) - i128::from(bound);
    if gap <= 0 {
        return;
    }
    if let Some((coefficients, variables)) = append_selector(coefficients, variables, selector, -gap) {
        solver.record_linear_relaxation(&coefficients, &variables, None, Some(bound));
    }
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
