//! The [`Propagator`] trait, the filtering error type, and event kinds.
//!
//! Filtering returns `Result<(), Inconsistency>` — a wiped-out domain is
//! `Err(Inconsistency)`, never a panic. Propagators must be idempotent and must
//! not allocate in the hot loop.

use crate::ids::PropId;
use crate::store::Store;

/// A domain became empty: the current node is infeasible. Carries no data; the
/// search reacts by backtracking.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Inconsistency;

/// The kind of domain change that should wake a propagator.
///
/// Phase 0 schedules on *any* change to a subscribed variable; this enum is the
/// vocabulary for the event-driven refinement landing in Phase 1, so the
/// `register` API can already speak in these terms.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Event {
    /// The variable became fixed (`size == 1`).
    Fix,
    /// A bound (`min` or `max`) moved.
    BoundChange,
    /// Some interior value was removed (the most general change).
    DomainChange,
}

/// A filtering algorithm attached to a set of variables.
pub trait Propagator {
    /// Subscribe to the variables (and, later, events) this propagator reacts
    /// to. Called once when the propagator is posted; `me` is its own id.
    fn register(&mut self, store: &mut Store, me: PropId);

    /// Filter domains toward consistency. Must be idempotent: a second call
    /// with no intervening change makes no further change. Returns
    /// `Err(Inconsistency)` if it wipes out a domain.
    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency>;
}
