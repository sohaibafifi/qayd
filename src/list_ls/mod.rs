//! List local-search engine facade.
//!
//! This module gives the heuristic list engine an explicit home in the public
//! architecture while the implementation is still being split out of
//! `collection`. New semantic model concepts should not be added here. This
//! layer is for incumbent search, fallback solving, and routing-quality
//! heuristics over an already-defined model.

use std::sync::atomic::AtomicBool;

use crate::collection::{CollectionModel, CollectionSolution};

pub use crate::collection::audit_incremental;

/// Run the list local-search heuristic over an already-defined collection
/// model. This is a fallback and incumbent engine; model semantics must be
/// represented in `model::Model` before they reach this layer.
pub fn solve_collection(model: &CollectionModel, seed: u64, stop: &AtomicBool, report: &mut dyn FnMut(i64)) -> CollectionSolution {
    crate::collection::solve_collection(model, seed, stop, report)
}
