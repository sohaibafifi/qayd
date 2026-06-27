//! Solver backend orchestration layers.
//!
//! Frontends should build a model and call an engine here instead of owning
//! backend-specific lowering, branching, replay, or bound-management logic.

pub mod schedule;

pub(crate) mod routing;
