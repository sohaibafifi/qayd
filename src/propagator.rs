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

/// Cost band that orders scheduling: cheaper bands drain to fixpoint before
/// pricier ones re-run, so an expensive global sees fewer, tighter domains and
/// runs less often. Ordering only — the fixpoint reached is independent of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Priority {
    /// Unary/binary and bound arithmetic: constant or near-constant filtering.
    Cheap,
    /// Sums and element-like propagators: work linear in the scope.
    Linear,
    /// Globals (cumulative, circuit, no-overlap…): superlinear filtering.
    Expensive,
}

impl Priority {
    /// Band index; lowest runs first.
    pub(crate) fn index(self) -> usize {
        match self {
            Priority::Cheap => 0,
            Priority::Linear => 1,
            Priority::Expensive => 2,
        }
    }
}

/// Number of scheduling bands.
pub(crate) const NBANDS: usize = 3;

/// Clone support for boxed propagators in immutable solver templates.
pub trait PropagatorClone {
    fn clone_box(&self) -> Box<dyn Propagator>;
}

impl<T> PropagatorClone for T
where
    T: 'static + Propagator + Clone,
{
    fn clone_box(&self) -> Box<dyn Propagator> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Propagator> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// A filtering algorithm attached to a set of variables.
pub trait Propagator: PropagatorClone + Send {
    /// Subscribe to the variables/events this propagator reacts to. `me` is its own id.
    fn register(&mut self, store: &mut Store, me: PropId);

    /// Filter domains toward consistency. Must be idempotent; `Err(Inconsistency)` on wipeout.
    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency>;

    /// Scheduling cost band. Defaults to the neutral middle; override to
    /// [`Priority::Cheap`] for bound/arith primitives or [`Priority::Expensive`]
    /// for globals.
    fn priority(&self) -> Priority {
        Priority::Linear
    }
}
