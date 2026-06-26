//! List local-search engine facade.
//!
//! This module gives the heuristic list engine an explicit home in the public
//! architecture while the implementation is still being split out of
//! `collection`. New semantic model concepts should not be added here. This
//! layer is for incumbent search, fallback solving, and routing-quality
//! heuristics over an already-defined model.

pub use crate::collection::{audit_incremental, solve_collection};
