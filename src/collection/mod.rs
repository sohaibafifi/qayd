//! Transitional collection IR and list local search.
//!
//! A model partitions a universe of item ids among `lists` ordered list
//! variables. Objective and constraints are reductions over a list: an
//! aggregate of a lambda body evaluated at each item, or at each edge of a
//! route. The same engine covers routing, sequencing, assignment, and packing
//! models by changing only the reductions and constraints.
//!
//! This module is not the final semantic owner for list or interval models. It
//! currently carries three responsibilities that will be split incrementally:
//! shared list-model data, validation, and heuristic local search. New model
//! semantics should be representable in the Rust core model and, where exact
//! solving is intended, in the structured list / interval kernel. The local
//! search code remains useful as a fallback and incumbent generator.

mod eval;
mod local_search;
mod model;
mod moves;
mod schedule_ls;
mod validate;

pub use local_search::solve_collection;
pub use model::{
    CollectionModel, CollectionSolution, Constraint, Expr, ExprArena, ExprId, GlobalConstraint, IntervalVar, Iterable, Mode, ObjectiveTier,
    Op, ReduceOp, Reduction, Resource, Schedule, MAX_TIERS,
};
pub use moves::audit_incremental;
