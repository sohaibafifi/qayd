//! The constraint catalogue.
//!
//! Each constraint is a [`Propagator`](crate::Propagator) plus a posting helper
//! that builds it and registers it with the [`Solver`](crate::Solver). The
//! catalogue grows phase by phase; Phase 0
//! ships only the binary disequality needed for the N-Queens milestone.

pub mod count;
pub mod graph;
pub mod intension;
pub mod lex;
pub mod linear;
pub mod primitives;
pub mod scheduling;
pub mod table;
