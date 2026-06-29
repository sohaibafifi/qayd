//! Move-based local search over list models.
//!
//! Incumbent and fallback engine for list-shaped models. Model semantics live in
//! `model` / `collection`; this layer only searches an already-defined model.

mod eval;
mod local_search;
mod moves;
mod schedule_ls;

pub use local_search::{solve_collection, solve_collection_capped};
pub use moves::audit_incremental;
