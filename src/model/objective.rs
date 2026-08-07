//! Model objectives.

use super::{IntExpr, IntervalVarRef, ListMaxTerm, ListReduction};

/// Objective declarations owned by the shared model.
#[derive(Clone)]
pub enum Objective {
    /// Integer expression objective.
    IntExpr { minimize: bool, expr: IntExpr },
    /// One tier over list reductions.
    ListTerms { minimize: bool, terms: Vec<ListReduction>, max_terms: Option<Vec<ListMaxTerm>> },
    /// Minimize or maximize the latest interval end.
    Makespan { minimize: bool, intervals: Vec<IntervalVarRef> },
}

impl Objective {
    pub fn is_minimize(&self) -> bool {
        match self {
            Self::IntExpr { minimize, .. } | Self::ListTerms { minimize, .. } | Self::Makespan { minimize, .. } => *minimize,
        }
    }
}
