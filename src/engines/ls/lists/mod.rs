//! Move-based local search over list models.
//!
//! Incumbent and fallback engine for list-shaped models. Model semantics live in
//! `model` / `collection`; this layer only searches an already-defined model.

mod alns;
mod elite;
pub(crate) mod eval;
mod incremental;
mod local_search;
mod metrics;
pub(crate) mod move_acceptance;
mod moves;
mod portfolio;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod resource_schedule;
mod routing_search;
mod schedule_ls;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod schedule_state;

#[doc(hidden)]
#[cfg(test)]
pub(crate) use alns::{
    audit_annealing_acceptance, audit_bounded_alns, audit_bounded_structural_work, audit_bounded_worst_removal,
    audit_completed_canonical_rebuild_accounting, audit_direct_guided_exchange_enumeration, audit_incremental_macro_accounting,
    audit_interrupted_attempt_accounting, audit_interruptible_routing_budget_admission, audit_macro_operators, audit_operator_learning,
    audit_route_elimination_budget_1000, audit_routing_scale_operator_exploration, audit_skipped_operator_adaptation,
    audit_timing_independent_operator_learning, audit_unbounded_alns_compatibility, audit_unproductive_deterministic_cost_balance,
};
#[cfg(test)]
pub(crate) use elite::{
    audit_elite_archive, audit_path_relink, audit_path_relink_interruption_accounting, audit_path_relink_large_partition,
    audit_relink_bound,
};
#[cfg(test)]
pub(crate) use local_search::{
    audit_lexicographic_regret, audit_size_safe_routing_compound_budget, solve_collection, solve_collection_capped,
    solve_collection_capped_profiled,
};
pub(crate) use local_search::{routing_search_supported, solve_collection_validated};
pub(crate) use metrics::ListSearchMetrics;
#[cfg(test)]
pub(crate) use metrics::{
    AnytimeCheckpointMetrics, ListIterableKind, ListReduceOpKind, NeighborhoodSearchMetrics, RoutingAuxiliaryMetrics, RoutingSearchMetrics,
};
#[cfg(test)]
pub(crate) use moves::{audit_incremental, NeighborhoodKind, RoutingAuditOutcome, RoutingNeighborhoodAudit, ScanMode, WorkBudget};
#[cfg(test)]
pub(crate) use portfolio::solve_collection_parallel_capped_profiled;
pub(crate) use portfolio::solve_collection_parallel_validated;
#[doc(hidden)]
#[cfg(test)]
pub(crate) use portfolio::{audit_incumbent_exchange, audit_portfolio_merge};
#[cfg(test)]
pub(crate) use routing_search::{
    audit_checkpoint_history, audit_exhausted_neighborhood_learning, audit_guaranteed_exploration, audit_neighborhood_learning,
    audit_routing_activity_counters, audit_scheduler_prefix, audit_timing_independent_cost_learning, audit_unproductive_cost_balance,
    SliceKind,
};
#[cfg(test)]
pub(crate) use schedule_ls::solve_schedule_capped;
#[cfg(test)]
pub(crate) use schedule_ls::{audit_persistent_schedule_split, solve_schedule};
pub(crate) use schedule_ls::{solve_schedule_capped_persistent, ScheduleConstructionMetrics, ScheduleSearchSession};
