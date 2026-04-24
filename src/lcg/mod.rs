//! Lazy Clause Generation: a CDCL engine layered over the FD store.
//!
//! The finite-domain [`Store`](crate::store::Store) stays the propagation
//! engine — every existing propagator runs unchanged on integer domains. On top
//! of it this module adds a Boolean view (the *literal layer*) and conflict-
//! driven clause learning, so the search backjumps non-chronologically and
//! reuses what it learns.
//!
//! The pieces, bottom-up:
//!
//! - [`lit`] — the atom numbering and [`Lit`](lit::Lit) packing (order atoms
//!   `[x≥k]` and equality atoms `[x=v]`, eagerly allocated per variable).
//! - [`view`] — atom truth as a pure view over the domain, and
//!   [`apply`](view::apply) to push a literal back into the domain.
//! - [`clause`] — the clause database (channeling axioms, learned, and
//!   constraint clauses).
//! - [`trail`] — the [`Cdcl`](trail::Cdcl) engine: the SAT trail, the channeling
//!   hook, watched-literal unit propagation interleaved with the FD fixpoint,
//!   generic propagator explanations, and 1-UIP analysis with backjumping.
//! - [`engine`] — the CSP ([`enumerate`](trail::Cdcl::enumerate)) and COP
//!   ([`optimize`](trail::Cdcl::optimize)) drivers the public search entry
//!   points are built on.
//!
//! Atom truth is a view over the trailed domain, so backtracking restores it for
//! free; the SAT trail records only assignment order and reasons. Channeling
//! between a variable's order and equality atoms is handled by posted
//! order-encoding axiom clauses, and every FD propagator's inference is
//! explained generically from its scope's domains — so the whole constraint
//! catalogue becomes learning-aware without any per-propagator changes.

pub mod clause;
pub mod engine;
pub mod lit;
pub mod trail;
pub mod view;
