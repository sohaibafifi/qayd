//! Generic solver problem representation independent of any input format.

use crate::expr::Expr;
use crate::ids::VarId;
use crate::search::Objective as SearchObjective;
use crate::store::Solver;

/// Immutable root model cloned by portfolio workers.
#[derive(Clone)]
pub struct Problem {
    pub solver: Solver,
    pub search: Vec<VarId>,
    pub objective: Option<Objective>,
}

/// Objective form consumed by the search layer.
#[derive(Clone)]
pub enum Objective {
    Var(bool, VarId),
    Linear(bool, Vec<i64>, Vec<VarId>),
    Expr(bool, Expr),
}

impl Problem {
    pub(crate) fn var_objective(&self) -> Option<(bool, VarId)> {
        self.objective.as_ref().and_then(|objective| objective.var().map(|obj| (objective.minimizing(), obj)))
    }

    pub(crate) fn objective_dir(&self) -> Option<bool> {
        self.objective.as_ref().map(Objective::minimizing)
    }
}

impl Objective {
    pub(crate) fn minimizing(&self) -> bool {
        match self {
            Self::Var(minimizing, _) | Self::Linear(minimizing, _, _) | Self::Expr(minimizing, _) => *minimizing,
        }
    }

    pub(crate) fn var(&self) -> Option<VarId> {
        match self {
            Self::Var(_, var) => Some(*var),
            Self::Linear(_, _, _) | Self::Expr(_, _) => None,
        }
    }

    pub(crate) fn search(&self) -> SearchObjective<'_> {
        match self {
            Self::Var(_, var) => SearchObjective::Var(*var),
            Self::Linear(_, coeffs, vars) => SearchObjective::Linear { coeffs, vars },
            Self::Expr(_, expr) => SearchObjective::Expr(expr),
        }
    }
}
