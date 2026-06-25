//! General collection modeling and local search.
//!
//! A model partitions a universe of item ids among `lists` ordered list
//! variables. Objective and constraints are reductions over a list: an
//! aggregate of a lambda body evaluated at each item, or at each edge of a
//! route. The same engine covers routing, sequencing, assignment, and packing
//! models by changing only the reductions and constraints.

mod eval;
mod list;
mod model;
mod moves;
mod schedule;
mod validate;

pub use list::solve_collection;
pub use model::{
    CollectionModel, CollectionSolution, Constraint, Expr, ExprArena, ExprId, GlobalConstraint, IntervalVar, Iterable, Mode, ObjectiveTier,
    Op, ReduceOp, Reduction, Resource, Schedule, MAX_TIERS,
};
pub use moves::audit_incremental;
