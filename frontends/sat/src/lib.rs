//! DIMACS parsing and SAT compatibility exports.
//!
//! This crate owns only the command-line parser and protocol renderer. SAT
//! preprocessing, search, proof production, and finalization live in the core.

pub use qayd::sat::*;
