//! List local-search engine facade.
//!
//! Thin compatibility facade over [`crate::engines::ls::lists`], where the list
//! local-search engine now lives. Kept until callers migrate to the engine
//! directly. New model semantics must not be added here.

use std::sync::atomic::AtomicBool;

use crate::collection::{CollectionModel, CollectionSolution};

pub use crate::engines::ls::lists::audit_incremental;

/// Run the list local-search heuristic over an already-defined collection
/// model. This is a fallback and incumbent engine; model semantics must be
/// represented in `model::Model` before they reach this layer.
pub fn solve_collection(model: &CollectionModel, seed: u64, stop: &AtomicBool, report: &mut dyn FnMut(i64)) -> CollectionSolution {
    crate::engines::ls::lists::solve_collection(model, seed, stop, report)
}
