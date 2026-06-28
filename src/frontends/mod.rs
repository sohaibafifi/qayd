//! Frontends: build a `model::Model` (or the transitional collection IR) from an
//! API or file format, call an engine, then format the result. A frontend owns
//! neither model semantics nor backend strategy.

#[cfg(feature = "python")]
mod python;

pub mod xcsp;
