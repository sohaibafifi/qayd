//! Model objectives.

use super::{IntervalVarRef, ListReduction};
use crate::expr::Expr;

/// Objective declarations owned by the shared model.
#[derive(Clone)]
pub enum Objective {
    /// Integer expression objective.
    IntExpr { minimize: bool, expr: Expr },
    /// One tier over list reductions.
    ListTerms { minimize: bool, terms: Vec<ListReduction> },
    /// Minimize or maximize the latest interval end.
    Makespan { minimize: bool, intervals: Vec<IntervalVarRef> },
}
