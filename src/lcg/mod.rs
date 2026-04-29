//! Lazy Clause Generation: a CDCL engine layered over the FD store.
//!
//! The FD [`Store`](crate::store::Store) stays the propagation engine; this layer
//! adds a Boolean view and conflict-driven clause learning with non-chronological
//! backjumping. Submodules, bottom-up: [`lit`] (atom numbering + [`Lit`](lit::Lit)
//! packing), [`view`] (atom truth as a domain view), [`clause`] (clause database),
//! [`trail`] (the [`Cdcl`](trail::Cdcl) engine: SAT trail, channeling, unit
//! propagation, 1-UIP analysis), [`engine`] (the CSP/COP search drivers).
//!
//! Atom truth is a view over the trailed domain, so backtracking restores it for
//! free. Each FD propagator's inference is explained generically from its scope,
//! making the whole catalogue learning-aware without per-propagator changes.

pub mod clause;
pub mod engine;
pub mod lit;
pub mod trail;
pub mod view;
