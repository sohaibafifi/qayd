//! The [`Propagator`] trait, the filtering error type, and event kinds.

use crate::ids::PropId;
use crate::store::Store;

/// A domain became empty: the current node is infeasible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Inconsistency;

/// The kind of domain change that should wake a propagator.
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
    /// Subscribe to the variables/events this propagator reacts to. `me` is its own id.
    fn register(&mut self, store: &mut Store, me: PropId);

    /// Filter domains toward consistency. Must be idempotent; `Err(Inconsistency)` on wipeout.
    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency>;
}
