//! `qayd`: a constraint-programming solver.
//!
//! ## Example
//!
//! ```
//! use qayd::model::{Constraint, Model, ModelPackage, Relation};
//! use qayd::orchestrator::{IgnoreEvents, SolveRequest, SolveStatus};
//!
//! let mut model = Model::new();
//! let value = model.int_range(0, 2);
//! model.add_constraint(Constraint::Linear {
//!     terms: vec![(1, value)],
//!     relation: Relation::Eq,
//!     rhs: 1,
//! });
//!
//! let package = ModelPackage::new(model);
//! let result = qayd::solve(
//!     &package,
//!     &SolveRequest::default(),
//!     &mut IgnoreEvents,
//! ).expect("the model is valid and supported");
//!
//! assert_eq!(result.status(), SolveStatus::Satisfiable);
//! assert_eq!(result.primal().unwrap().assignment().integers[value.0], Some(1));
//! ```

pub mod constraints;
pub mod domains;
mod engines;
pub mod expr;
pub mod frontends;
pub mod ids;
pub mod lcg;
mod lns;
pub mod mem;
pub mod model;
pub mod mus;
pub mod orchestrator;
#[allow(dead_code)]
mod problem;
pub mod propagator;
pub mod search;
mod search_policy;
/// Parser-facing SAT data types. Search implementation remains under the
/// engine layer and is selected only by the orchestrator.
pub mod sat {
    pub use crate::engines::sat::{assignment_satisfies, parse_dimacs, Cnf};
}
pub mod store;
pub mod trail;

pub use domains::interval::{IntervalEvent, IntervalPresence};
pub use domains::list::ListEvent;
pub use expr::Expr;
pub use ids::{IntervalId, ListId, PropId, VarId};
// The list-reduction (list/lambda) IR lives under `qayd::model::list::*`; its
// `Expr` is intentionally not surfaced at the crate root, where it would clash
// with the intension `Expr`.
/// Canonical frontend-neutral solve entry point.
pub use orchestrator::solve_model as solve;
pub use orchestrator::solve_model_with_stop as solve_with_stop;
pub use propagator::{Event, Inconsistency, Propagator};
pub use search::{
    count_solutions, first_domain_solution, first_solution, first_solution_assuming, maximize, minimize, optimize_var_assuming,
    optimize_with, solve as solve_search, solve_bool_cnf_interruptible, solve_bool_cnf_seeded, solve_bool_cnf_seeded_with_proof,
    solve_domains, solve_domains_interruptible, solve_interruptible, solve_under_assumptions, Assumption, AssumptionOp, AssumptionResult,
    BoolCnfError, BoolLit, DomainSolution, SearchControl, SolveStats,
};
pub use store::{Solver, Store};
pub use trail::{ReversibleInt, Trail};

#[cfg(test)]
pub(crate) use engines::ls::lists::audit_exhausted_neighborhood_learning;

/// SplitMix64 finalizer: scrambles a counter/seed into a well-distributed
/// `u64`. Shared so every seed-derived stream stays bit-identical across
/// modules - determinism depends on it.
pub(crate) fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

// Engine and physical-IR tests live under `tests/internal` so their helpers can
// remain crate-private without becoming part of the downstream Rust API.
#[cfg(test)]
extern crate self as qayd;
#[cfg(test)]
#[path = "../tests/internal/architecture_consolidation.rs"]
mod architecture_consolidation;
#[cfg(test)]
#[path = "../tests/internal/backend_phase3.rs"]
mod backend_phase3;
#[cfg(test)]
#[path = "../tests/internal/collection.rs"]
mod collection;
#[cfg(test)]
#[path = "../tests/internal/cp_compile_interruption.rs"]
mod cp_compile_interruption;
#[cfg(test)]
#[path = "../tests/internal/disjunctive_schedule.rs"]
mod disjunctive_schedule;
#[cfg(test)]
#[path = "../tests/internal/divisibility_compaction.rs"]
mod divisibility_compaction;
#[cfg(test)]
#[path = "../tests/internal/exact_cover_warm_start.rs"]
mod exact_cover_warm_start;
#[cfg(test)]
#[path = "../tests/internal/lcg_atom_layout.rs"]
mod lcg_atom_layout;
#[cfg(test)]
#[path = "../tests/internal/lcg_conflict_budget.rs"]
mod lcg_conflict_budget;
#[cfg(test)]
#[path = "../tests/internal/lcg_guarded_sum.rs"]
mod lcg_guarded_sum;
#[cfg(test)]
#[path = "../tests/internal/lcg_objective_branching.rs"]
mod lcg_objective_branching;
#[cfg(test)]
#[path = "../tests/internal/lcg_root_probe_budget.rs"]
mod lcg_root_probe_budget;
#[cfg(test)]
#[path = "../tests/internal/list_alns_phase4.rs"]
mod list_alns_phase4;
#[cfg(test)]
#[path = "../tests/internal/list_exact_interruption.rs"]
mod list_exact_interruption;
#[cfg(test)]
#[path = "../tests/internal/list_incremental_phase2.rs"]
mod list_incremental_phase2;
#[cfg(test)]
#[path = "../tests/internal/list_portfolio_phase5.rs"]
mod list_portfolio_phase5;
#[cfg(test)]
#[path = "../tests/internal/list_search_metrics.rs"]
mod list_search_metrics;
#[cfg(test)]
#[path = "../tests/internal/ls_cop.rs"]
mod ls_cop;
#[cfg(test)]
#[path = "../tests/internal/ls_violation_oracle.rs"]
mod ls_violation_oracle;
#[cfg(test)]
#[path = "../tests/internal/mem_limit.rs"]
mod mem_limit;
#[cfg(test)]
#[path = "../tests/internal/model_architecture.rs"]
mod model_architecture;
#[cfg(test)]
#[path = "../tests/internal/orchestrator_architecture.rs"]
mod orchestrator_architecture;
#[cfg(test)]
#[path = "../tests/internal/orchestrator_budget.rs"]
mod orchestrator_budget;
#[cfg(test)]
#[path = "../tests/internal/phase4_quality_bench.rs"]
mod phase4_quality_bench;
#[cfg(test)]
#[path = "../tests/internal/phase5_portfolio_bench.rs"]
mod phase5_portfolio_bench;
#[cfg(test)]
#[path = "../tests/internal/phase6_dual.rs"]
mod phase6_dual;
#[cfg(test)]
#[path = "../tests/internal/phase7.rs"]
mod phase7;
#[cfg(test)]
#[path = "../tests/internal/phase9_phase10.rs"]
mod phase9_phase10;
#[cfg(test)]
#[path = "../tests/internal/prefix_set_compaction.rs"]
mod prefix_set_compaction;
#[cfg(test)]
#[path = "../tests/internal/resource_schedule.rs"]
mod resource_schedule;
#[cfg(test)]
#[path = "../tests/internal/routing_alns_elite.rs"]
mod routing_alns_elite;
#[cfg(test)]
#[path = "../tests/internal/routing_neighborhoods.rs"]
mod routing_neighborhoods;
#[cfg(test)]
#[path = "../tests/internal/routing_search.rs"]
mod routing_search;
#[cfg(test)]
#[path = "../tests/internal/sat_preprocess_interrupt.rs"]
mod sat_preprocess_interrupt;
#[cfg(test)]
#[path = "../tests/internal/scenario_schedule.rs"]
mod scenario_schedule;
#[cfg(test)]
#[path = "../tests/internal/schedule_compile_contract.rs"]
mod schedule_compile_contract;
#[cfg(test)]
#[path = "../tests/internal/schedule_elite.rs"]
mod schedule_elite;
#[cfg(test)]
#[path = "../tests/internal/schedule_interruption.rs"]
mod schedule_interruption;
#[cfg(test)]
#[path = "../tests/internal/schedule_iteration_limit.rs"]
mod schedule_iteration_limit;
#[cfg(test)]
#[path = "../tests/internal/schedule_lns.rs"]
mod schedule_lns;
#[cfg(test)]
#[path = "../tests/internal/schedule_ls_interruption.rs"]
mod schedule_ls_interruption;
#[cfg(test)]
#[path = "../tests/internal/schedule_mode_identity.rs"]
mod schedule_mode_identity;
#[cfg(test)]
#[path = "../tests/internal/schedule_portfolio_exchange.rs"]
mod schedule_portfolio_exchange;
#[cfg(test)]
#[path = "../tests/internal/schedule_relink.rs"]
mod schedule_relink;
#[cfg(test)]
#[path = "../tests/internal/schedule_state.rs"]
mod schedule_state;
#[cfg(test)]
#[path = "../tests/internal/scheduling_kernel.rs"]
mod scheduling_kernel;
#[cfg(test)]
#[path = "../tests/internal/semantic_preparation_interruption.rs"]
mod semantic_preparation_interruption;
#[cfg(test)]
#[path = "../tests/internal/verification_boundaries.rs"]
mod verification_boundaries;
#[cfg(test)]
#[path = "../tests/internal/vrptw_search.rs"]
mod vrptw_search;
