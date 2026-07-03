//! Generic solver problem representation independent of any input format.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::expr::Expr;
use crate::ids::VarId;
use crate::search::Objective as SearchObjective;
use crate::store::Solver;

/// Root-level simplification summary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PresolveStats {
    pub(crate) fixed: usize,
    pub(crate) search_before: usize,
    pub(crate) search_after: usize,
    pub(crate) failed: bool,
    pub(crate) stopped: bool,
}

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
    pub(crate) fn presolve(&mut self, stop: &AtomicBool) -> PresolveStats {
        let search_before = self.search.len();
        self.solver.enqueue_all();
        let result = self.solver.propagate_until(|| stop.load(Ordering::Relaxed));
        self.solver.store.events.clear();
        let stopped = stop.load(Ordering::Relaxed);
        if result.is_err() || stopped {
            return PresolveStats { fixed: 0, search_before, search_after: self.search.len(), failed: result.is_err(), stopped };
        }

        let mut keep_fixed = vec![false; self.solver.store.num_vars()];
        self.mark_objective_vars(&mut keep_fixed);
        let fixed = self.search.iter().filter(|&&var| self.solver.store.is_fixed(var)).count();
        self.search.retain(|&var| !self.solver.store.is_fixed(var) || keep_fixed[var.index()]);
        PresolveStats { fixed, search_before, search_after: self.search.len(), failed: false, stopped: false }
    }

    pub(crate) fn var_objective(&self) -> Option<(bool, VarId)> {
        self.objective.as_ref().and_then(|objective| objective.var().map(|obj| (objective.minimizing(), obj)))
    }

    pub(crate) fn objective_dir(&self) -> Option<bool> {
        self.objective.as_ref().map(Objective::minimizing)
    }

    fn mark_objective_vars(&self, keep: &mut [bool]) {
        match &self.objective {
            Some(Objective::Var(_, var)) => keep[var.index()] = true,
            Some(Objective::Linear(_, _, vars)) => {
                for &var in vars {
                    keep[var.index()] = true;
                }
            }
            Some(Objective::Expr(_, expr)) => {
                let mut vars = Vec::new();
                expr.collect_vars(&mut vars);
                for var in vars {
                    keep[var.index()] = true;
                }
            }
            None => {}
        }
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
