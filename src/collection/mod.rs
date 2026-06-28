//! Transitional collection IR and list local search.
//!
//! A model partitions a universe of item ids among `lists` ordered list
//! variables. Objective and constraints are reductions over a list: an
//! aggregate of a lambda body evaluated at each item, or at each edge of a
//! route. The same engine covers routing, sequencing, assignment, and packing
//! models by changing only the reductions and constraints.
//!
//! This module is not the final semantic owner for list or interval models. It
//! now carries the shared list-model data and its validation; the heuristic
//! local search moved to [`crate::engines::ls::lists`]. New model semantics
//! should be representable in the Rust core model and, where exact solving is
//! intended, in the structured list / interval kernel.

mod model;
mod validate;

pub use model::{
    CollectionModel, CollectionSolution, Constraint, Expr, ExprArena, ExprId, GlobalConstraint, IntervalVar, Iterable, Mode, ObjectiveTier,
    Op, ReduceOp, Reduction, Resource, Schedule, MAX_TIERS,
};
