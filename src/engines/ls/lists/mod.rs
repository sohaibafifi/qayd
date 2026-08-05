//! Move-based local search over list models.
//!
//! Incumbent and fallback engine for list-shaped models. Model semantics live in
//! `model` / `collection`; this layer only searches an already-defined model.

mod alns;
pub(crate) mod eval;
mod incremental;
mod local_search;
mod metrics;
mod moves;
mod portfolio;
mod schedule_ls;

#[doc(hidden)]
pub use alns::{audit_annealing_acceptance, audit_operator_learning};
#[cfg(feature = "python")]
pub(crate) use local_search::solve_collection_validated;
pub use local_search::{
    solve_collection, solve_collection_capped, solve_collection_capped_profiled, solve_collection_hinted, solve_collection_profiled,
};
pub use metrics::{
    AdaptiveOperatorMetrics, AlnsSearchMetrics, ListIterableKind, ListReduceOpKind, ListSearchMetrics, ReductionSearchMetrics,
};
pub use moves::audit_incremental;
#[cfg(feature = "python")]
pub(crate) use portfolio::solve_collection_parallel_validated;
#[doc(hidden)]
pub use portfolio::{audit_incumbent_exchange, audit_portfolio_merge};
pub use portfolio::{
    solve_collection_parallel, solve_collection_parallel_capped_profiled, ListPortfolioMetrics, ListPortfolioWorkerMetrics,
};
