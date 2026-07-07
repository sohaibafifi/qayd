//! Model objectives.

use super::{IntervalVarRef, ListMaxTerm, ListReduction};
use crate::expr::Expr;

/// Objective declarations owned by the shared model.
#[derive(Clone)]
pub enum Objective {
    /// Integer expression objective.
    IntExpr { minimize: bool, expr: Expr },
    /// One tier over list reductions.
    ListTerms { minimize: bool, terms: Vec<ListReduction>, max_terms: Option<Vec<ListMaxTerm>> },
    /// Minimize or maximize the latest interval end.
    Makespan { minimize: bool, intervals: Vec<IntervalVarRef> },
}
