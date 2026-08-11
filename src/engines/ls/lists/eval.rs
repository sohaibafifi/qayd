//! List-search scoring primitives backed by the canonical runtime evaluator.

use crate::model::list::{Op, Reduction};

pub(crate) use crate::model::list::eval_expr_runtime as eval_expr;

/// Violation weight for an undefined reduction or operand.
pub(super) const INFEASIBLE: i64 = 1_000_000_000;

/// Evaluate a fully materialized list with the same semantics as canonical
/// replay. Incremental caches are checked against this path in the LS oracles.
pub(super) fn eval_reduction(reduction: &Reduction, contents: &[i32]) -> Option<i64> {
    crate::model::list::eval_reduction_on_contents(reduction, contents)
}

pub(super) fn violation_of(value: i64, op: Op, rhs: i64) -> i64 {
    match op {
        Op::Le => value.saturating_sub(rhs).max(0),
        Op::Ge => rhs.saturating_sub(value).max(0),
        Op::Eq => value.saturating_sub(rhs).saturating_abs(),
    }
}
