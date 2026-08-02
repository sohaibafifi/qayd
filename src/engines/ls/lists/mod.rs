//! Move-based local search over list models.
//!
//! Incumbent and fallback engine for list-shaped models. Model semantics live in
//! `model` / `collection`; this layer only searches an already-defined model.

mod eval;
mod incremental;
mod local_search;
mod metrics;
mod moves;
mod schedule_ls;

pub use local_search::{
    solve_collection, solve_collection_capped, solve_collection_capped_profiled, solve_collection_hinted, solve_collection_profiled,
};
pub use metrics::{ListIterableKind, ListReduceOpKind, ListSearchMetrics, ReductionSearchMetrics};
pub use moves::audit_incremental;
