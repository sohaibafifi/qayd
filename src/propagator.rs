//! The [`Propagator`] trait, the filtering error type, and event kinds.

use crate::ids::{PropId, VarId};
use crate::store::{Premise, Store};

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

/// A half-reified guard around another propagator: the inner constraint is
/// enforced only while the selector variable `sel` is fixed to `1`, and every
/// reason the inner propagator produces is attributed to `sel = 1` (via the
/// store's injected premise), so a refutation traces back to this selector.
///
/// The building block for constraint-level unsat cores: post each candidate
/// constraint under its own selector, assume all selectors true, and a
/// [`solve_under_assumptions`](crate::search::solve_under_assumptions) core is
/// exactly the offending constraints. Posted transparently by
/// [`Solver::post`](crate::store::Solver::post) under
/// [`set_selector`](crate::store::Solver::set_selector).
#[derive(Clone)]
pub struct Selected {
    sel: VarId,
    inner: Box<dyn Propagator>,
}

impl Selected {
    /// Guard `inner` with selector `sel` (a `{0,1}` variable).
    pub fn new(sel: VarId, inner: Box<dyn Propagator>) -> Self {
        Self { sel, inner }
    }
}

impl Propagator for Selected {
    fn register(&mut self, store: &mut Store, me: PropId) {
        // Wake when the selector is fixed (0 disables, 1 activates), and adopt
        // the inner propagator's own subscriptions under this same id so the
        // whole-scope reason includes `sel`.
        store.subscribe(self.sel, me, Event::Fix);
        self.inner.register(store, me);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if !store.is_fixed(self.sel) {
            return Ok(()); // selector undecided: constraint not yet required
        }
        if store.value(self.sel) != 1 {
            return Ok(()); // selector off: constraint deactivated (half-reified)
        }
        // Active: attribute every reason the inner produces to `sel = 1`, so the
        // selector appears in each learned nogood. Tight inference reasons are
        // stamped through the store's injected premise; a `fail_because` conflict
        // is stamped explicitly; a bare wipeout falls back to the whole scope,
        // which already carries `sel` via the subscription above.
        let premise = Premise::Eq { var: self.sel, val: 1 };
        store.set_injected_premise(premise);
        let result = self.inner.propagate(store);
        if result.is_err() {
            store.push_conflict_premise(premise);
        }
        store.clear_injected_premise();
        result
    }

    fn priority(&self) -> Priority {
        self.inner.priority()
    }
}
