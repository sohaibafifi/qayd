#![cfg_attr(test, allow(dead_code, unused_imports))]

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

pub mod certificate;
pub mod constraints;
pub mod domains;
pub(crate) mod engines;
pub mod explanation;
pub mod expr;
pub mod frontends;
pub mod ids;
pub mod inference;
pub mod lcg;
pub(crate) mod lns;
pub mod mem;
pub mod model;
pub mod mus;
pub mod orchestrator;
#[allow(dead_code)]
pub(crate) mod problem;
pub mod propagator;
pub mod search;
pub(crate) mod search_policy;
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
