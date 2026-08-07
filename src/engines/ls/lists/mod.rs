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
#[cfg(test)]
pub(crate) use alns::{audit_annealing_acceptance, audit_operator_learning};
pub(crate) use local_search::solve_collection_validated;
#[cfg(test)]
pub(crate) use local_search::{solve_collection, solve_collection_capped, solve_collection_capped_profiled};
pub(crate) use metrics::ListSearchMetrics;
#[cfg(test)]
pub(crate) use metrics::{ListIterableKind, ListReduceOpKind};
#[cfg(test)]
pub(crate) use moves::audit_incremental;
#[cfg(test)]
pub(crate) use portfolio::solve_collection_parallel_capped_profiled;
pub(crate) use portfolio::solve_collection_parallel_validated;
#[doc(hidden)]
#[cfg(test)]
pub(crate) use portfolio::{audit_incumbent_exchange, audit_portfolio_merge};
pub(crate) use schedule_ls::{solve_schedule, ScheduleConstructionMetrics};
