//! Read-only semantic-model analysis shared by backend compilers.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{Constraint, IntExpr, IntVarRef};

/// Canonical affine form `constant + sum(coefficients[var] * var)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AffineForm {
    pub(crate) coefficients: BTreeMap<IntVarRef, i64>,
    pub(crate) constant: i64,
}

impl AffineForm {
    pub(crate) fn expression(expression: &IntExpr, stop: &AtomicBool) -> Option<Self> {
        let mut form = Self { coefficients: BTreeMap::new(), constant: 0 };
        form.add_expression(expression, 1, stop)?;
        Some(form)
    }

    pub(crate) fn difference(left: &IntExpr, right: &IntExpr, stop: &AtomicBool) -> Option<Self> {
        let mut form = Self { coefficients: BTreeMap::new(), constant: 0 };
        form.add_expression(left, 1, stop)?;
        form.add_expression(right, -1, stop)?;
        Some(form)
    }

    pub(crate) fn linear(terms: &[(i64, IntVarRef)], rhs: i64) -> Option<Self> {
        let mut form = Self { coefficients: BTreeMap::new(), constant: rhs.checked_neg()? };
        for &(coefficient, variable) in terms {
            form.add_coefficient(variable, coefficient)?;
        }
        Some(form)
    }

    pub(crate) fn scale(mut self, factor: i64) -> Option<Self> {
        self.constant = self.constant.checked_mul(factor)?;
        for coefficient in self.coefficients.values_mut() {
            *coefficient = coefficient.checked_mul(factor)?;
        }
        Some(self)
    }

    pub(crate) fn add_coefficient(&mut self, variable: IntVarRef, value: i64) -> Option<()> {
        let updated = self.coefficients.get(&variable).copied().unwrap_or(0).checked_add(value)?;
        if updated == 0 {
            self.coefficients.remove(&variable);
        } else {
            self.coefficients.insert(variable, updated);
        }
        Some(())
    }

    fn add_expression(&mut self, expression: &IntExpr, scale: i64, stop: &AtomicBool) -> Option<()> {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        match expression {
            IntExpr::Constant(value) => self.constant = self.constant.checked_add(scale.checked_mul(*value)?)?,
            IntExpr::Variable(variable) => self.add_coefficient(*variable, scale)?,
            IntExpr::Neg(value) => self.add_expression(value, scale.checked_neg()?, stop)?,
            IntExpr::Add(values) => {
                for value in values {
                    self.add_expression(value, scale, stop)?;
                }
            }
            IntExpr::Sub(left, right) => {
                self.add_expression(left, scale, stop)?;
                self.add_expression(right, scale.checked_neg()?, stop)?;
            }
            IntExpr::Mul(values) => {
                let mut coefficient = scale;
                let mut symbolic = None;
                for value in values {
                    if let IntExpr::Constant(value) = value {
                        coefficient = coefficient.checked_mul(*value)?;
                    } else if symbolic.replace(value).is_some() {
                        return None;
                    }
                }
                self.add_expression(symbolic?, coefficient, stop)?;
            }
            _ => return None,
        }
        Some(())
    }
}

impl Constraint {
    /// Sorted, duplicate-free integer-variable scope of this constraint.
    pub(crate) fn int_scope(&self) -> Vec<IntVarRef> {
        let mut variables = Vec::new();
        self.visit_int_variables(&mut |variable| variables.push(variable));
        variables.sort_unstable();
        variables.dedup();
        variables
    }

    /// Visit integer variables without allocating a temporary scope.
    pub(crate) fn visit_int_variables(&self, visitor: &mut impl FnMut(IntVarRef)) {
        match self {
            Constraint::Intension(expression) => {
                let mut variables = Vec::new();
                expression.variables(&mut variables);
                variables.into_iter().for_each(visitor);
            }
            Constraint::Linear { terms, .. } => terms.iter().for_each(|(_, variable)| visitor(*variable)),
            Constraint::Clause(literals) => literals.iter().for_each(|literal| visitor(literal.variable)),
            Constraint::IntegerGlobal(global) => {
                let mut variables = Vec::new();
                global.variables(&mut variables);
                variables.into_iter().for_each(visitor);
            }
            Constraint::Selected { selector, constraint } => {
                visitor(*selector);
                constraint.visit_int_variables(visitor);
            }
            Constraint::ListPartition { .. }
            | Constraint::ListPartitionWithCoverage { .. }
            | Constraint::SameList { .. }
            | Constraint::ItemPrecedence { .. }
            | Constraint::CollectionGlobal(_)
            | Constraint::ListLength { .. }
            | Constraint::ListItemSum { .. }
            | Constraint::ListReduction(_)
            | Constraint::IntervalPrecedence { .. }
            | Constraint::IntervalAlternative { .. }
            | Constraint::IntervalEndpointRelation { .. }
            | Constraint::IntervalResource(_)
            | Constraint::SetSubset { .. }
            | Constraint::SetDisjoint { .. }
            | Constraint::SetCardinality { .. } => {}
        }
    }
}
