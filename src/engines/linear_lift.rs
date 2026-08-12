//! Generic extended formulations for nonlinear integer objectives.
//!
//! Every lifted expression is an outer approximation of its exact graph. For
//! each integral CP assignment, assigning every auxiliary column its exact
//! expression value satisfies all emitted rows. Unsupported or over-budget
//! subexpressions are replaced by a sound root interval bound.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::expr::Expr;
use crate::ids::VarId;
use crate::store::Store;

const MAX_EXACT_F64_INTEGER: u128 = 1u128 << 53;

#[derive(Clone)]
pub(super) struct LiftedRow {
    pub(super) terms: Vec<(usize, i128)>,
    pub(super) lower: Option<i128>,
    pub(super) upper: Option<i128>,
}

pub(super) struct LiftedObjective {
    pub(super) constant: i128,
    pub(super) terms: BTreeMap<usize, i128>,
    pub(super) rows: Vec<LiftedRow>,
    pub(super) auxiliary_bounds: Vec<(i128, i128)>,
    pub(super) fallbacks: usize,
}

#[derive(Clone, Copy)]
pub(super) struct LiftLimits {
    pub(super) max_columns: usize,
    pub(super) max_rows: usize,
    pub(super) max_nonzeros: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LiftError {
    Interrupted,
    Invalid,
}

#[derive(Clone, Default, PartialEq, Eq)]
struct AffineValue {
    constant: i128,
    terms: BTreeMap<usize, i128>,
    lower: i128,
    upper: i128,
}

impl AffineValue {
    fn constant(value: i128) -> Self {
        Self { constant: value, terms: BTreeMap::new(), lower: value, upper: value }
    }

    fn is_boolean(&self) -> bool {
        self.lower >= 0 && self.upper <= 1
    }
}

#[derive(Clone)]
struct Checkpoint {
    auxiliary_count: usize,
    row_count: usize,
    nonzeros: usize,
    active: BTreeSet<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttemptError {
    Unsupported,
    Limit,
    Invalid,
    Interrupted,
}

impl From<AttemptError> for LiftError {
    fn from(value: AttemptError) -> Self {
        match value {
            AttemptError::Interrupted => Self::Interrupted,
            AttemptError::Unsupported | AttemptError::Limit | AttemptError::Invalid => Self::Invalid,
        }
    }
}

struct Lifter<'a> {
    store: &'a Store,
    stop: &'a AtomicBool,
    physical_columns: usize,
    limits: LiftLimits,
    auxiliary_bounds: Vec<(i128, i128)>,
    rows: Vec<LiftedRow>,
    active: BTreeSet<usize>,
    nonzeros: usize,
    fallbacks: usize,
}

pub(super) fn lift_expression_objective(
    expression: &Expr,
    scale: i128,
    store: &Store,
    limits: LiftLimits,
    stop: &AtomicBool,
) -> Result<LiftedObjective, LiftError> {
    let mut lifter = Lifter {
        store,
        stop,
        physical_columns: store.num_vars(),
        limits,
        auxiliary_bounds: Vec::new(),
        rows: Vec::new(),
        active: BTreeSet::new(),
        nonzeros: 0,
        fallbacks: 0,
    };
    let value = lifter.lower_relaxation(expression, scale).map_err(LiftError::from)?;
    if !exact(value.constant) || value.terms.values().any(|&coefficient| !exact(coefficient)) {
        return Err(LiftError::Invalid);
    }
    Ok(LiftedObjective {
        constant: value.constant,
        terms: value.terms,
        rows: lifter.rows,
        auxiliary_bounds: lifter.auxiliary_bounds,
        fallbacks: lifter.fallbacks,
    })
}

impl Lifter<'_> {
    fn lower_relaxation(&mut self, expression: &Expr, scale: i128) -> Result<AffineValue, AttemptError> {
        self.check_interrupt()?;
        if scale == 0 {
            return Ok(AffineValue::constant(0));
        }
        match expression {
            Expr::Add(values) => {
                let mut sum = AffineValue::constant(0);
                for value in values {
                    let term = self.lower_relaxation(value, scale)?;
                    sum = self.add(&sum, &term)?;
                }
                return Ok(sum);
            }
            Expr::Sub(left, right) => {
                let left = self.lower_relaxation(left, scale)?;
                let right = self.lower_relaxation(right, scale.checked_neg().ok_or(AttemptError::Invalid)?)?;
                return self.add(&left, &right);
            }
            Expr::Neg(value) => {
                return self.lower_relaxation(value, scale.checked_neg().ok_or(AttemptError::Invalid)?);
            }
            Expr::Mul(values) => {
                if let Some((constant, other)) = single_nonconstant_factor(values) {
                    return self.lower_relaxation(other, scale.checked_mul(i128::from(constant)).ok_or(AttemptError::Invalid)?);
                }
            }
            Expr::Const(_)
            | Expr::Var(_)
            | Expr::Abs(_)
            | Expr::Div(_, _)
            | Expr::Mod(_, _)
            | Expr::Min(_)
            | Expr::Max(_)
            | Expr::Eq(_, _)
            | Expr::Ne(_, _)
            | Expr::Lt(_, _)
            | Expr::Le(_, _)
            | Expr::Gt(_, _)
            | Expr::Ge(_, _)
            | Expr::Not(_)
            | Expr::And(_)
            | Expr::Or(_)
            | Expr::Imp(_, _)
            | Expr::Iff(_, _)
            | Expr::IfThenElse(_, _, _) => {}
        }

        let checkpoint = self.checkpoint();
        match self.lift_graph(expression).and_then(|value| self.scale(&value, scale)) {
            Ok(value) => Ok(value),
            Err(AttemptError::Interrupted) => Err(AttemptError::Interrupted),
            Err(AttemptError::Invalid) => Err(AttemptError::Invalid),
            Err(AttemptError::Unsupported | AttemptError::Limit) => {
                self.rollback(checkpoint);
                self.fallbacks = self.fallbacks.saturating_add(1);
                self.interval_fallback(expression, scale)
            }
        }
    }

    fn lift_graph(&mut self, expression: &Expr) -> Result<AffineValue, AttemptError> {
        self.check_interrupt()?;
        match expression {
            Expr::Const(value) => Ok(AffineValue::constant(i128::from(*value))),
            Expr::Var(variable) => self.variable(*variable),
            Expr::Neg(value) => {
                let value = self.lift_graph(value)?;
                self.scale(&value, -1)
            }
            Expr::Add(values) => {
                let mut sum = AffineValue::constant(0);
                for value in values {
                    let value = self.lift_graph(value)?;
                    sum = self.add(&sum, &value)?;
                }
                Ok(sum)
            }
            Expr::Sub(left, right) => {
                let left = self.lift_graph(left)?;
                let right = self.lift_graph(right)?;
                let right = self.scale(&right, -1)?;
                self.add(&left, &right)
            }
            Expr::Mul(values) => {
                let mut product = AffineValue::constant(1);
                for value in values {
                    let value = self.lift_graph(value)?;
                    product = self.product(&product, &value)?;
                }
                Ok(product)
            }
            Expr::Abs(value) => {
                let value = self.lift_graph(value)?;
                self.absolute(&value)
            }
            Expr::Min(values) => self.extremum(values, true),
            Expr::Max(values) => self.extremum(values, false),
            Expr::Eq(left, right) => self.comparison(left, right, Comparison::Eq),
            Expr::Ne(left, right) => {
                let equal = self.comparison(left, right, Comparison::Eq)?;
                self.boolean_not(&equal)
            }
            Expr::Lt(left, right) => self.comparison(left, right, Comparison::Le(-1)),
            Expr::Le(left, right) => self.comparison(left, right, Comparison::Le(0)),
            Expr::Gt(left, right) => self.comparison(left, right, Comparison::Ge(1)),
            Expr::Ge(left, right) => self.comparison(left, right, Comparison::Ge(0)),
            Expr::Not(value) => {
                let value = self.lift_boolean(value)?;
                self.boolean_not(&value)
            }
            Expr::And(values) => self.boolean_and(values),
            Expr::Or(values) => self.boolean_or(values),
            Expr::Imp(left, right) => {
                let left = self.lift_boolean(left)?;
                let left = self.boolean_not(&left)?;
                let right = self.lift_boolean(right)?;
                self.boolean_or_values(&[left, right])
            }
            Expr::Iff(left, right) => {
                let left = self.lift_boolean(left)?;
                let right = self.lift_boolean(right)?;
                self.boolean_iff(&left, &right)
            }
            Expr::IfThenElse(condition, then_value, else_value) => {
                let condition = self.lift_boolean(condition)?;
                let then_value = self.lift_graph(then_value)?;
                let else_value = self.lift_graph(else_value)?;
                self.if_then_else(&condition, &then_value, &else_value)
            }
            Expr::Div(_, _) | Expr::Mod(_, _) => Err(AttemptError::Unsupported),
        }
    }

    fn lift_boolean(&mut self, expression: &Expr) -> Result<AffineValue, AttemptError> {
        let value = self.lift_graph(expression)?;
        value.is_boolean().then_some(value).ok_or(AttemptError::Unsupported)
    }

    fn variable(&mut self, variable: VarId) -> Result<AffineValue, AttemptError> {
        if variable.index() >= self.physical_columns {
            return Err(AttemptError::Invalid);
        }
        self.activate(variable.index())?;
        Ok(AffineValue {
            constant: 0,
            terms: BTreeMap::from([(variable.index(), 1)]),
            lower: i128::from(self.store.min(variable)),
            upper: i128::from(self.store.max(variable)),
        })
    }

    fn add(&self, left: &AffineValue, right: &AffineValue) -> Result<AffineValue, AttemptError> {
        let mut terms = left.terms.clone();
        add_scaled_terms(&mut terms, &right.terms, 1)?;
        Ok(AffineValue {
            constant: left.constant.checked_add(right.constant).ok_or(AttemptError::Invalid)?,
            terms,
            lower: left.lower.checked_add(right.lower).ok_or(AttemptError::Invalid)?,
            upper: left.upper.checked_add(right.upper).ok_or(AttemptError::Invalid)?,
        })
    }

    fn scale(&self, value: &AffineValue, scale: i128) -> Result<AffineValue, AttemptError> {
        let terms = value
            .terms
            .iter()
            .map(|(&column, &coefficient)| coefficient.checked_mul(scale).map(|coefficient| (column, coefficient)))
            .collect::<Option<BTreeMap<_, _>>>()
            .ok_or(AttemptError::Invalid)?;
        let first = value.lower.checked_mul(scale).ok_or(AttemptError::Invalid)?;
        let second = value.upper.checked_mul(scale).ok_or(AttemptError::Invalid)?;
        Ok(AffineValue {
            constant: value.constant.checked_mul(scale).ok_or(AttemptError::Invalid)?,
            terms,
            lower: first.min(second),
            upper: first.max(second),
        })
    }

    fn product(&mut self, left: &AffineValue, right: &AffineValue) -> Result<AffineValue, AttemptError> {
        if left.terms.is_empty() {
            return self.scale(right, left.constant);
        }
        if right.terms.is_empty() {
            return self.scale(left, right.constant);
        }
        if left == right {
            return self.square(left);
        }
        let corners = [
            left.lower.checked_mul(right.lower),
            left.lower.checked_mul(right.upper),
            left.upper.checked_mul(right.lower),
            left.upper.checked_mul(right.upper),
        ];
        let corners = corners.into_iter().collect::<Option<Vec<_>>>().ok_or(AttemptError::Invalid)?;
        let lower = *corners.iter().min().ok_or(AttemptError::Invalid)?;
        let upper = *corners.iter().max().ok_or(AttemptError::Invalid)?;
        let result = self.auxiliary(lower, upper)?;

        self.mccormick_lower(&result, left, right, left.lower, right.lower)?;
        self.mccormick_lower(&result, left, right, left.upper, right.upper)?;
        self.mccormick_upper(&result, left, right, left.upper, right.lower)?;
        self.mccormick_upper(&result, left, right, left.lower, right.upper)?;
        Ok(result)
    }

    fn square(&mut self, value: &AffineValue) -> Result<AffineValue, AttemptError> {
        let lower_square = value.lower.checked_mul(value.lower).ok_or(AttemptError::Invalid)?;
        let upper_square = value.upper.checked_mul(value.upper).ok_or(AttemptError::Invalid)?;
        let lower = if value.lower <= 0 && value.upper >= 0 { 0 } else { lower_square.min(upper_square) };
        let upper = lower_square.max(upper_square);
        let result = self.auxiliary(lower, upper)?;
        if value.lower == value.upper {
            return Ok(result);
        }

        // Consecutive-integer secants are global underestimators at integer
        // points. Boundary cuts plus the two cuts around zero give a compact
        // envelope for wide domains without expanding one row per value.
        let last = value.upper.checked_sub(1).ok_or(AttemptError::Invalid)?;
        let mut points = BTreeSet::from([value.lower, last]);
        if value.lower <= -1 && -1 <= last {
            points.insert(-1);
        }
        if value.lower <= 0 && 0 <= last {
            points.insert(0);
        }
        for point in points {
            let slope = point.checked_mul(2).and_then(|value| value.checked_add(1)).ok_or(AttemptError::Invalid)?;
            let intercept = point.checked_mul(point.checked_add(1).ok_or(AttemptError::Invalid)?).ok_or(AttemptError::Invalid)?;
            let mut row = result.clone();
            row = self.add(&row, &self.scale(value, slope.checked_neg().ok_or(AttemptError::Invalid)?)?)?;
            self.add_value_row(&row, Some(intercept.checked_neg().ok_or(AttemptError::Invalid)?), None)?;
        }

        let slope = value.lower.checked_add(value.upper).ok_or(AttemptError::Invalid)?;
        let intercept = value.lower.checked_mul(value.upper).ok_or(AttemptError::Invalid)?;
        let mut row = result.clone();
        row = self.add(&row, &self.scale(value, slope.checked_neg().ok_or(AttemptError::Invalid)?)?)?;
        self.add_value_row(&row, None, Some(intercept.checked_neg().ok_or(AttemptError::Invalid)?))?;
        Ok(result)
    }

    fn mccormick_lower(
        &mut self,
        result: &AffineValue,
        left: &AffineValue,
        right: &AffineValue,
        left_bound: i128,
        right_bound: i128,
    ) -> Result<(), AttemptError> {
        let mut row = result.clone();
        row = self.add(&row, &self.scale(right, -left_bound)?)?;
        row = self.add(&row, &self.scale(left, -right_bound)?)?;
        let rhs = left_bound.checked_mul(right_bound).and_then(i128::checked_neg).ok_or(AttemptError::Invalid)?;
        self.add_value_row(&row, Some(rhs), None)
    }

    fn mccormick_upper(
        &mut self,
        result: &AffineValue,
        left: &AffineValue,
        right: &AffineValue,
        left_bound: i128,
        right_bound: i128,
    ) -> Result<(), AttemptError> {
        let mut row = result.clone();
        row = self.add(&row, &self.scale(right, -left_bound)?)?;
        row = self.add(&row, &self.scale(left, -right_bound)?)?;
        let rhs = left_bound.checked_mul(right_bound).and_then(i128::checked_neg).ok_or(AttemptError::Invalid)?;
        self.add_value_row(&row, None, Some(rhs))
    }

    fn absolute(&mut self, value: &AffineValue) -> Result<AffineValue, AttemptError> {
        if value.lower >= 0 {
            return Ok(value.clone());
        }
        if value.upper <= 0 {
            return self.scale(value, -1);
        }
        let upper = value.lower.checked_neg().ok_or(AttemptError::Invalid)?.max(value.upper);
        let result = self.auxiliary(0, upper)?;
        let minus_value = self.scale(value, -1)?;
        let row = self.add(&result, &minus_value)?;
        self.add_value_row(&row, Some(0), None)?;
        let row = self.add(&result, value)?;
        self.add_value_row(&row, Some(0), None)?;
        Ok(result)
    }

    fn extremum(&mut self, values: &[Expr], minimum: bool) -> Result<AffineValue, AttemptError> {
        let values = values.iter().map(|value| self.lift_graph(value)).collect::<Result<Vec<_>, _>>()?;
        if values.is_empty() {
            return Err(AttemptError::Unsupported);
        }
        let lower = if minimum { values.iter().map(|value| value.lower).min() } else { values.iter().map(|value| value.lower).max() }
            .ok_or(AttemptError::Invalid)?;
        let upper = if minimum { values.iter().map(|value| value.upper).min() } else { values.iter().map(|value| value.upper).max() }
            .ok_or(AttemptError::Invalid)?;
        let result = self.auxiliary(lower, upper)?;
        for value in values {
            let opposite = self.scale(&value, -1)?;
            let row = self.add(&result, &opposite)?;
            if minimum {
                self.add_value_row(&row, None, Some(0))?;
            } else {
                self.add_value_row(&row, Some(0), None)?;
            }
        }
        Ok(result)
    }

    fn comparison(&mut self, left: &Expr, right: &Expr, comparison: Comparison) -> Result<AffineValue, AttemptError> {
        let left = self.lift_graph(left)?;
        let right = self.lift_graph(right)?;
        if matches!(comparison, Comparison::Eq) && left.is_boolean() && right.is_boolean() {
            return self.boolean_iff(&left, &right);
        }
        let opposite = self.scale(&right, -1)?;
        let difference = self.add(&left, &opposite)?;
        match comparison {
            Comparison::Le(offset) => self.upper_indicator(&difference, i128::from(offset)),
            Comparison::Ge(offset) => self.lower_indicator(&difference, i128::from(offset)),
            Comparison::Eq => self.equality_indicator(&difference),
        }
    }

    fn upper_indicator(&mut self, value: &AffineValue, bound: i128) -> Result<AffineValue, AttemptError> {
        if value.upper <= bound {
            return Ok(AffineValue::constant(1));
        }
        if value.lower > bound {
            return Ok(AffineValue::constant(0));
        }
        let result = self.auxiliary(0, 1)?;
        self.imply_upper(value, &result, bound, true)?;
        let false_bound = bound.checked_add(1).ok_or(AttemptError::Invalid)?;
        self.imply_lower(value, &result, false_bound, false)?;
        Ok(result)
    }

    fn lower_indicator(&mut self, value: &AffineValue, bound: i128) -> Result<AffineValue, AttemptError> {
        if value.lower >= bound {
            return Ok(AffineValue::constant(1));
        }
        if value.upper < bound {
            return Ok(AffineValue::constant(0));
        }
        let result = self.auxiliary(0, 1)?;
        self.imply_lower(value, &result, bound, true)?;
        let false_bound = bound.checked_sub(1).ok_or(AttemptError::Invalid)?;
        self.imply_upper(value, &result, false_bound, false)?;
        Ok(result)
    }

    fn equality_indicator(&mut self, value: &AffineValue) -> Result<AffineValue, AttemptError> {
        if value.lower == 0 && value.upper == 0 {
            return Ok(AffineValue::constant(1));
        }
        if value.lower > 0 || value.upper < 0 {
            return Ok(AffineValue::constant(0));
        }
        if value.lower == 0 {
            return self.upper_indicator(value, 0);
        }
        if value.upper == 0 {
            return self.lower_indicator(value, 0);
        }
        let result = self.auxiliary(0, 1)?;
        self.imply_lower(value, &result, 0, true)?;
        self.imply_upper(value, &result, 0, true)?;
        Ok(result)
    }

    fn imply_upper(&mut self, value: &AffineValue, selector: &AffineValue, bound: i128, when_true: bool) -> Result<(), AttemptError> {
        let gap = value.upper.checked_sub(bound).ok_or(AttemptError::Invalid)?;
        if gap <= 0 {
            return Ok(());
        }
        let selector_scale = if when_true { gap } else { gap.checked_neg().ok_or(AttemptError::Invalid)? };
        let row = self.add(value, &self.scale(selector, selector_scale)?)?;
        let upper = if when_true { value.upper } else { bound };
        self.add_value_row(&row, None, Some(upper))
    }

    fn imply_lower(&mut self, value: &AffineValue, selector: &AffineValue, bound: i128, when_true: bool) -> Result<(), AttemptError> {
        let gap = bound.checked_sub(value.lower).ok_or(AttemptError::Invalid)?;
        if gap <= 0 {
            return Ok(());
        }
        let selector_scale = if when_true { gap.checked_neg().ok_or(AttemptError::Invalid)? } else { gap };
        let row = self.add(value, &self.scale(selector, selector_scale)?)?;
        let lower = if when_true { value.lower } else { bound };
        self.add_value_row(&row, Some(lower), None)
    }

    fn boolean_not(&self, value: &AffineValue) -> Result<AffineValue, AttemptError> {
        if !value.is_boolean() {
            return Err(AttemptError::Unsupported);
        }
        let negative = self.scale(value, -1)?;
        self.add(&AffineValue::constant(1), &negative)
    }

    fn boolean_and(&mut self, values: &[Expr]) -> Result<AffineValue, AttemptError> {
        let values = values.iter().map(|value| self.lift_boolean(value)).collect::<Result<Vec<_>, _>>()?;
        self.boolean_and_values(&values)
    }

    fn boolean_and_values(&mut self, values: &[AffineValue]) -> Result<AffineValue, AttemptError> {
        if values.is_empty() {
            return Ok(AffineValue::constant(1));
        }
        if values.len() == 1 {
            return Ok(values[0].clone());
        }
        let result = self.auxiliary(0, 1)?;
        for value in values {
            let opposite = self.scale(value, -1)?;
            let row = self.add(&result, &opposite)?;
            self.add_value_row(&row, None, Some(0))?;
        }
        let mut row = result.clone();
        for value in values {
            row = self.add(&row, &self.scale(value, -1)?)?;
        }
        let lower = 1i128.checked_sub(i128::try_from(values.len()).map_err(|_| AttemptError::Invalid)?).ok_or(AttemptError::Invalid)?;
        self.add_value_row(&row, Some(lower), None)?;
        Ok(result)
    }

    fn boolean_or(&mut self, values: &[Expr]) -> Result<AffineValue, AttemptError> {
        let values = values.iter().map(|value| self.lift_boolean(value)).collect::<Result<Vec<_>, _>>()?;
        self.boolean_or_values(&values)
    }

    fn boolean_or_values(&mut self, values: &[AffineValue]) -> Result<AffineValue, AttemptError> {
        if values.is_empty() {
            return Ok(AffineValue::constant(0));
        }
        if values.len() == 1 {
            return Ok(values[0].clone());
        }
        let result = self.auxiliary(0, 1)?;
        for value in values {
            let opposite = self.scale(value, -1)?;
            let row = self.add(&result, &opposite)?;
            self.add_value_row(&row, Some(0), None)?;
        }
        let mut row = result.clone();
        for value in values {
            row = self.add(&row, &self.scale(value, -1)?)?;
        }
        self.add_value_row(&row, None, Some(0))?;
        Ok(result)
    }

    fn boolean_iff(&mut self, left: &AffineValue, right: &AffineValue) -> Result<AffineValue, AttemptError> {
        if !left.is_boolean() || !right.is_boolean() {
            return Err(AttemptError::Unsupported);
        }
        let product = self.product(left, right)?;
        let mut formula = AffineValue::constant(1);
        formula = self.add(&formula, &self.scale(left, -1)?)?;
        formula = self.add(&formula, &self.scale(right, -1)?)?;
        formula = self.add(&formula, &self.scale(&product, 2)?)?;
        let result = self.auxiliary(0, 1)?;
        let row = self.add(&result, &self.scale(&formula, -1)?)?;
        self.add_value_row(&row, Some(0), Some(0))?;
        Ok(result)
    }

    fn if_then_else(
        &mut self,
        condition: &AffineValue,
        then_value: &AffineValue,
        else_value: &AffineValue,
    ) -> Result<AffineValue, AttemptError> {
        if !condition.is_boolean() {
            return Err(AttemptError::Unsupported);
        }
        let lower = then_value.lower.min(else_value.lower);
        let upper = then_value.upper.max(else_value.upper);
        let result = self.auxiliary(lower, upper)?;
        let then_difference = self.add(&result, &self.scale(then_value, -1)?)?;
        self.imply_lower(&then_difference, condition, 0, true)?;
        self.imply_upper(&then_difference, condition, 0, true)?;
        let else_difference = self.add(&result, &self.scale(else_value, -1)?)?;
        self.imply_lower(&else_difference, condition, 0, false)?;
        self.imply_upper(&else_difference, condition, 0, false)?;
        Ok(result)
    }

    fn auxiliary(&mut self, lower: i128, upper: i128) -> Result<AffineValue, AttemptError> {
        if lower > upper || !exact(lower) || !exact(upper) {
            return Err(AttemptError::Unsupported);
        }
        let column = self.physical_columns.checked_add(self.auxiliary_bounds.len()).ok_or(AttemptError::Invalid)?;
        self.activate(column)?;
        self.auxiliary_bounds.push((lower, upper));
        Ok(AffineValue { constant: 0, terms: BTreeMap::from([(column, 1)]), lower, upper })
    }

    fn add_value_row(&mut self, value: &AffineValue, lower: Option<i128>, upper: Option<i128>) -> Result<(), AttemptError> {
        let lower = lower.map(|bound| bound.checked_sub(value.constant).ok_or(AttemptError::Invalid)).transpose()?;
        let upper = upper.map(|bound| bound.checked_sub(value.constant).ok_or(AttemptError::Invalid)).transpose()?;
        self.add_row(value.terms.clone(), lower, upper)
    }

    fn add_row(&mut self, mut terms: BTreeMap<usize, i128>, lower: Option<i128>, upper: Option<i128>) -> Result<(), AttemptError> {
        terms.retain(|_, coefficient| *coefficient != 0);
        if terms.values().any(|&coefficient| !exact(coefficient))
            || lower.is_some_and(|bound| !exact(bound))
            || upper.is_some_and(|bound| !exact(bound))
        {
            return Err(AttemptError::Unsupported);
        }
        if self.rows.len() >= self.limits.max_rows {
            return Err(AttemptError::Limit);
        }
        let nonzeros = self.nonzeros.checked_add(terms.len()).ok_or(AttemptError::Invalid)?;
        if nonzeros > self.limits.max_nonzeros {
            return Err(AttemptError::Limit);
        }
        for &column in terms.keys() {
            self.activate(column)?;
        }
        self.nonzeros = nonzeros;
        self.rows.push(LiftedRow { terms: terms.into_iter().collect(), lower, upper });
        Ok(())
    }

    fn activate(&mut self, column: usize) -> Result<(), AttemptError> {
        if !self.active.contains(&column) && self.active.len() >= self.limits.max_columns {
            return Err(AttemptError::Limit);
        }
        self.active.insert(column);
        Ok(())
    }

    fn interval_fallback(&self, expression: &Expr, scale: i128) -> Result<AffineValue, AttemptError> {
        let (lower, upper) = expression.bounds(&|variable| (i64::from(self.store.min(variable)), i64::from(self.store.max(variable))));
        let lower = i128::from(lower).checked_mul(scale).ok_or(AttemptError::Invalid)?;
        let upper = i128::from(upper).checked_mul(scale).ok_or(AttemptError::Invalid)?;
        let bound = lower.min(upper);
        exact(bound).then_some(AffineValue::constant(bound)).ok_or(AttemptError::Invalid)
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            auxiliary_count: self.auxiliary_bounds.len(),
            row_count: self.rows.len(),
            nonzeros: self.nonzeros,
            active: self.active.clone(),
        }
    }

    fn rollback(&mut self, checkpoint: Checkpoint) {
        self.auxiliary_bounds.truncate(checkpoint.auxiliary_count);
        self.rows.truncate(checkpoint.row_count);
        self.nonzeros = checkpoint.nonzeros;
        self.active = checkpoint.active;
    }

    fn check_interrupt(&self) -> Result<(), AttemptError> {
        if self.stop.load(Ordering::Acquire) {
            Err(AttemptError::Interrupted)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
enum Comparison {
    Le(i64),
    Ge(i64),
    Eq,
}

fn add_scaled_terms(target: &mut BTreeMap<usize, i128>, source: &BTreeMap<usize, i128>, scale: i128) -> Result<(), AttemptError> {
    for (&column, &coefficient) in source {
        let scaled = coefficient.checked_mul(scale).ok_or(AttemptError::Invalid)?;
        let combined = target.get(&column).copied().unwrap_or(0).checked_add(scaled).ok_or(AttemptError::Invalid)?;
        if combined == 0 {
            target.remove(&column);
        } else {
            target.insert(column, combined);
        }
    }
    Ok(())
}

fn single_nonconstant_factor(values: &[Expr]) -> Option<(i64, &Expr)> {
    let mut constant = 1i64;
    let mut nonconstant = None;
    for value in values {
        match value {
            Expr::Const(value) => constant = constant.checked_mul(*value)?,
            _ if nonconstant.is_none() => nonconstant = Some(value),
            _ => return None,
        }
    }
    nonconstant.map(|value| (constant, value))
}

fn exact(value: i128) -> bool {
    value.unsigned_abs() <= MAX_EXACT_F64_INTEGER
}
