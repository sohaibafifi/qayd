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
    /// A materialized objective variable with an affine direction guide
    /// inferred from an exact equality. Bounds and values use `objective`;
    /// `coeffs`/`vars` are used only by explicitly guided search policies.
    VarWithAffine(bool, VarId, Vec<i64>, Vec<VarId>),
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
            Self::Var(minimizing, _)
            | Self::VarWithAffine(minimizing, _, _, _)
            | Self::Linear(minimizing, _, _)
            | Self::Expr(minimizing, _) => *minimizing,
        }
    }

    pub(crate) fn var(&self) -> Option<VarId> {
        match self {
            Self::Var(_, var) | Self::VarWithAffine(_, var, _, _) => Some(*var),
            Self::Linear(_, _, _) | Self::Expr(_, _) => None,
        }
    }

    pub(crate) fn search(&self) -> SearchObjective<'_> {
        match self {
            Self::Var(_, var) => SearchObjective::Var(*var),
            Self::VarWithAffine(_, objective, coeffs, vars) => SearchObjective::VarWithAffine { objective: *objective, coeffs, vars },
            Self::Linear(_, coeffs, vars) => SearchObjective::Linear { coeffs, vars },
            Self::Expr(_, expr) => SearchObjective::Expr(expr),
        }
    }

    /// Search view for a bounded objective-oriented dive. The objective value
    /// and bound representation stay unchanged; affine and structural guidance
    /// may start before the first incumbent under the dive's work budget.
    pub(crate) fn bounded_dive_search(&self) -> SearchObjective<'_> {
        match self {
            Self::VarWithAffine(_, objective, coeffs, vars) => {
                SearchObjective::BoundedDiveVarWithAffine { objective: *objective, coeffs, vars }
            }
            Self::Linear(_, coeffs, vars) => SearchObjective::BoundedDiveLinear { coeffs, vars },
            Self::Var(_, var) => SearchObjective::Var(*var),
            Self::Expr(_, expr) => SearchObjective::BoundedDiveExpr(expr),
        }
    }
}
