//! Variable domains: the mutable per-variable state the solver searches over.
//!
//! `int` is the finite-domain integer set; `list` is list membership;
//! `interval` is a fixed-duration interval handle. Nothing here knows about model
//! semantics or search strategy.

pub mod int;
pub mod interval;
pub mod list;
