//! Semantic integer and Boolean expressions.
//!
//! These expressions reference model variables, never variables allocated in
//! a solver store. Physical expressions are produced only by a compiler.

use super::IntVarRef;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntExpr {
    Constant(i64),
    Variable(IntVarRef),
    Neg(Box<Self>),
    Abs(Box<Self>),
    Add(Vec<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Vec<Self>),
    Div(Box<Self>, Box<Self>),
    Mod(Box<Self>, Box<Self>),
    Min(Vec<Self>),
    Max(Vec<Self>),
    Eq(Box<Self>, Box<Self>),
    Ne(Box<Self>, Box<Self>),
    Lt(Box<Self>, Box<Self>),
    Le(Box<Self>, Box<Self>),
    Gt(Box<Self>, Box<Self>),
    Ge(Box<Self>, Box<Self>),
    Not(Box<Self>),
    And(Vec<Self>),
    Or(Vec<Self>),
    Imp(Box<Self>, Box<Self>),
    Iff(Box<Self>, Box<Self>),
    IfThenElse(Box<Self>, Box<Self>, Box<Self>),
}

impl IntExpr {
    pub fn variable(variable: IntVarRef) -> Self {
        Self::Variable(variable)
    }

    pub fn constant(value: i64) -> Self {
        Self::Constant(value)
    }

    pub fn variables(&self, output: &mut Vec<IntVarRef>) {
        match self {
            Self::Constant(_) => {}
            Self::Variable(variable) => output.push(*variable),
            Self::Neg(value) | Self::Abs(value) | Self::Not(value) => value.variables(output),
            Self::Add(values) | Self::Mul(values) | Self::Min(values) | Self::Max(values) | Self::And(values) | Self::Or(values) => {
                values.iter().for_each(|value| value.variables(output))
            }
            Self::Sub(left, right)
            | Self::Div(left, right)
            | Self::Mod(left, right)
            | Self::Eq(left, right)
            | Self::Ne(left, right)
            | Self::Lt(left, right)
            | Self::Le(left, right)
            | Self::Gt(left, right)
            | Self::Ge(left, right)
            | Self::Imp(left, right)
            | Self::Iff(left, right) => {
                left.variables(output);
                right.variables(output);
            }
            Self::IfThenElse(condition, then_value, else_value) => {
                condition.variables(output);
                then_value.variables(output);
                else_value.variables(output);
            }
        }
    }

    pub fn evaluate(&self, value: &impl Fn(IntVarRef) -> Option<i64>) -> Option<i64> {
        let truth = |number: i64| i64::from(number != 0);
        Some(match self {
            Self::Constant(number) => *number,
            Self::Variable(variable) => value(*variable)?,
            Self::Neg(inner) => inner.evaluate(value)?.checked_neg()?,
            Self::Abs(inner) => inner.evaluate(value)?.checked_abs()?,
            Self::Add(values) => values.iter().try_fold(0i64, |sum, item| sum.checked_add(item.evaluate(value)?))?,
            Self::Sub(left, right) => left.evaluate(value)?.checked_sub(right.evaluate(value)?)?,
            Self::Mul(values) => values.iter().try_fold(1i64, |product, item| product.checked_mul(item.evaluate(value)?))?,
            Self::Div(left, right) => left.evaluate(value)?.checked_div(right.evaluate(value)?)?,
            Self::Mod(left, right) => left.evaluate(value)?.checked_rem(right.evaluate(value)?)?,
            Self::Min(values) => values.iter().map(|item| item.evaluate(value)).collect::<Option<Vec<_>>>()?.into_iter().min()?,
            Self::Max(values) => values.iter().map(|item| item.evaluate(value)).collect::<Option<Vec<_>>>()?.into_iter().max()?,
            Self::Eq(left, right) => i64::from(left.evaluate(value)? == right.evaluate(value)?),
            Self::Ne(left, right) => i64::from(left.evaluate(value)? != right.evaluate(value)?),
            Self::Lt(left, right) => i64::from(left.evaluate(value)? < right.evaluate(value)?),
            Self::Le(left, right) => i64::from(left.evaluate(value)? <= right.evaluate(value)?),
            Self::Gt(left, right) => i64::from(left.evaluate(value)? > right.evaluate(value)?),
            Self::Ge(left, right) => i64::from(left.evaluate(value)? >= right.evaluate(value)?),
            Self::Not(inner) => i64::from(truth(inner.evaluate(value)?) == 0),
            Self::And(values) => i64::from(values.iter().all(|item| item.evaluate(value).is_some_and(|number| number != 0))),
            Self::Or(values) => i64::from(values.iter().any(|item| item.evaluate(value).is_some_and(|number| number != 0))),
            Self::Imp(left, right) => i64::from(truth(left.evaluate(value)?) == 0 || truth(right.evaluate(value)?) != 0),
            Self::Iff(left, right) => i64::from(truth(left.evaluate(value)?) == truth(right.evaluate(value)?)),
            Self::IfThenElse(condition, then_value, else_value) => {
                if condition.evaluate(value)? != 0 {
                    then_value.evaluate(value)?
                } else {
                    else_value.evaluate(value)?
                }
            }
        })
    }
}

impl From<IntVarRef> for IntExpr {
    fn from(value: IntVarRef) -> Self {
        Self::Variable(value)
    }
}

impl From<i64> for IntExpr {
    fn from(value: i64) -> Self {
        Self::Constant(value)
    }
}
