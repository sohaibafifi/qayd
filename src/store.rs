//! The solver core.
//!
//! [`Store`] owns every domain and the [`Trail`], plus the variable→propagator
//! subscription lists and the propagation queue. Every domain mutation that
//! actually changes a domain wakes the subscribed propagators. [`Solver`] adds
//! the propagator *objects* on top of the store and drives the fixpoint.
//!
//! Why the split: a propagator needs `&mut Store` while it runs, but the store
//! cannot also lend out the boxed propagator it owns. So [`Solver`] keeps the
//! `Box<dyn Propagator>` objects, takes one out to run it, and hands it the
//! store — clean borrows, no `RefCell`.
//!
//! Scheduling is event-driven. A propagator subscribes to each variable at one
//! granularity:
//!
//! - [`Event::Fix`] — wake only when the variable becomes fixed.
//! - [`Event::BoundChange`] — wake when `min`/`max` moves (fixing counts, since
//!   it collapses the bounds).
//! - [`Event::DomainChange`] — wake on any value removal.
//!
//! A domain change fires the set of events it satisfies, and every subscriber
//! whose chosen granularity is in that set is enqueued.

use std::collections::VecDeque;

use crate::domain::Domain;
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::trail::{ReversibleInt, Trail};

/// Per-variable subscription lists, one per event granularity.
#[derive(Default)]
struct VarSubs {
    /// Woken on any value removal.
    on_dom: Vec<PropId>,
    /// Woken when a bound moves (or the variable is fixed).
    on_bnd: Vec<PropId>,
    /// Woken only when the variable becomes fixed.
    on_fix: Vec<PropId>,
}

/// Owns all mutable solver state: domains, trail, subscriptions, and the
/// propagation queue.
#[derive(Default)]
pub struct Store {
    domains: Vec<Domain>,
    trail: Trail,
    subs: Vec<VarSubs>,
    /// Propagators waiting to run.
    queue: VecDeque<PropId>,
    /// `enqueued[prop]` guards against queuing a propagator twice.
    enqueued: Vec<bool>,
    /// The propagator currently running, if any (so it is not re-woken by its
    /// own filtering).
    pub(crate) current: Option<PropId>,
}

impl Store {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    // --- variable creation ---

    /// Create a variable with domain `[lo, hi]`.
    pub fn new_var_range(&mut self, lo: i32, hi: i32) -> VarId {
        let id = VarId(self.domains.len() as u32);
        let dom = Domain::new_range(lo, hi, &mut self.trail);
        self.domains.push(dom);
        self.subs.push(VarSubs::default());
        id
    }

    /// Create a variable with an explicit set of domain values.
    pub fn new_var_set(&mut self, values: &[i32]) -> VarId {
        let id = VarId(self.domains.len() as u32);
        let dom = Domain::new_set(values, &mut self.trail);
        self.domains.push(dom);
        self.subs.push(VarSubs::default());
        id
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.domains.len()
    }

    // --- domain reads ---

    /// Minimum value in `var`'s domain.
    pub fn min(&self, var: VarId) -> i32 {
        self.domains[var.index()].min(&self.trail)
    }

    /// Maximum value in `var`'s domain.
    pub fn max(&self, var: VarId) -> i32 {
        self.domains[var.index()].max(&self.trail)
    }

    /// Number of values in `var`'s domain.
    pub fn size(&self, var: VarId) -> usize {
        self.domains[var.index()].size(&self.trail)
    }

    /// Whether `var` is fixed to a single value.
    pub fn is_fixed(&self, var: VarId) -> bool {
        self.domains[var.index()].is_fixed(&self.trail)
    }

    /// Whether `val` is in `var`'s domain.
    pub fn contains(&self, var: VarId, val: i32) -> bool {
        self.domains[var.index()].contains(val, &self.trail)
    }

    /// The single value of a fixed variable. Debug-asserts `var` is fixed.
    pub fn value(&self, var: VarId) -> i32 {
        debug_assert!(self.is_fixed(var), "value() called on unfixed {var:?}");
        self.domains[var.index()].min(&self.trail)
    }

    /// Iterate the present values of `var` (arbitrary order).
    pub fn values(&self, var: VarId) -> impl Iterator<Item = i32> + '_ {
        self.domains[var.index()].values(&self.trail)
    }

    // --- domain mutations (each wakes subscribers on a real change) ---

    /// Remove `val` from `var`. `Err` on wipeout.
    pub fn remove(&mut self, var: VarId, val: i32) -> Result<(), Inconsistency> {
        let i = var.index();
        let (old_min, old_max) = self.bounds(i);
        let changed = {
            let dom = &mut self.domains[i];
            dom.remove(val, &mut self.trail)
        };
        if changed {
            self.after_change(var, old_min, old_max)?;
        }
        Ok(())
    }

    /// Fix `var` to `val`. `Err` if `val` is absent.
    pub fn fix(&mut self, var: VarId, val: i32) -> Result<(), Inconsistency> {
        let i = var.index();
        let (old_min, old_max) = self.bounds(i);
        let changed = {
            let dom = &mut self.domains[i];
            dom.fix(val, &mut self.trail)?
        };
        if changed {
            self.after_change(var, old_min, old_max)?;
        }
        Ok(())
    }

    /// Remove all values `< bound` from `var`. `Err` on wipeout.
    pub fn remove_below(&mut self, var: VarId, bound: i32) -> Result<(), Inconsistency> {
        let i = var.index();
        let (old_min, old_max) = self.bounds(i);
        let changed = {
            let dom = &mut self.domains[i];
            dom.remove_below(bound, &mut self.trail)
        };
        if changed {
            self.after_change(var, old_min, old_max)?;
        }
        Ok(())
    }

    /// Remove all values `> bound` from `var`. `Err` on wipeout.
    pub fn remove_above(&mut self, var: VarId, bound: i32) -> Result<(), Inconsistency> {
        let i = var.index();
        let (old_min, old_max) = self.bounds(i);
        let changed = {
            let dom = &mut self.domains[i];
            dom.remove_above(bound, &mut self.trail)
        };
        if changed {
            self.after_change(var, old_min, old_max)?;
        }
        Ok(())
    }

    /// `(min, max)` of variable index `i`.
    fn bounds(&self, i: usize) -> (i32, i32) {
        let dom = &self.domains[i];
        (dom.min(&self.trail), dom.max(&self.trail))
    }

    /// Check for wipeout and wake the right subscribers after a real change.
    fn after_change(
        &mut self,
        var: VarId,
        old_min: i32,
        old_max: i32,
    ) -> Result<(), Inconsistency> {
        let i = var.index();
        let size = self.domains[i].size(&self.trail);
        if size == 0 {
            return Err(Inconsistency);
        }
        let (new_min, new_max) = self.bounds(i);
        let bounds_moved = new_min != old_min || new_max != old_max;
        let fixed = size == 1;
        self.notify(var, bounds_moved, fixed);
        Ok(())
    }

    // --- propagator-owned reversible state ---
    //
    // Some propagators (e.g. circuit) keep incremental state that must be
    // restored on backtrack. Allocate it as reversible ints at registration
    // time (the root level), then read/write it during propagation.

    /// Allocate a reversible int (at the current/root level) for a propagator.
    pub fn new_rev_int(&mut self, v: i32) -> ReversibleInt {
        self.trail.new_int(v)
    }

    /// Read a reversible int.
    pub fn rev_get(&self, r: ReversibleInt) -> i32 {
        self.trail.get(r)
    }

    /// Write a reversible int (logged for backtracking).
    pub fn rev_set(&mut self, r: ReversibleInt, v: i32) {
        self.trail.set(r, v);
    }

    // --- propagator scheduling ---

    /// Subscribe `prop` to changes on `var` at the given granularity.
    /// Idempotent per (var, prop, event).
    pub fn subscribe(&mut self, var: VarId, prop: PropId, event: Event) {
        let s = &mut self.subs[var.index()];
        let list = match event {
            Event::DomainChange => &mut s.on_dom,
            Event::BoundChange => &mut s.on_bnd,
            Event::Fix => &mut s.on_fix,
        };
        if !list.contains(&prop) {
            list.push(prop);
        }
    }

    /// Reserve a scheduling slot for a new propagator and return its id. The
    /// returned id equals the number of propagators allocated so far, so the
    /// [`Solver`] can keep its `Box<dyn Propagator>` vector in lockstep.
    pub(crate) fn alloc_prop(&mut self) -> PropId {
        let id = PropId(self.enqueued.len() as u32);
        self.enqueued.push(false);
        id
    }

    /// Put `prop` on the queue unless it is already there.
    pub(crate) fn enqueue(&mut self, prop: PropId) {
        if !self.enqueued[prop.index()] {
            self.enqueued[prop.index()] = true;
            self.queue.push_back(prop);
        }
    }

    /// Pop the next propagator to run, clearing its enqueued flag.
    pub(crate) fn dequeue(&mut self) -> Option<PropId> {
        let p = self.queue.pop_front()?;
        self.enqueued[p.index()] = false;
        Some(p)
    }

    /// Drop every queued propagator (used after a failure so stale entries do
    /// not leak into the sibling search branch).
    pub(crate) fn clear_queue(&mut self) {
        while let Some(p) = self.queue.pop_front() {
            self.enqueued[p.index()] = false;
        }
    }

    /// Wake the appropriate subscribers of `var` (except the running one).
    fn notify(&mut self, var: VarId, bounds_moved: bool, fixed: bool) {
        // Split the borrow so subscription lists can be read while the queue and
        // flags are mutated. Subscription lists never change during propagation.
        let Store {
            subs,
            queue,
            enqueued,
            current,
            ..
        } = self;
        let s = &subs[var.index()];
        for &p in &s.on_dom {
            Self::wake(queue, enqueued, *current, p);
        }
        if bounds_moved {
            for &p in &s.on_bnd {
                Self::wake(queue, enqueued, *current, p);
            }
        }
        if fixed {
            for &p in &s.on_fix {
                Self::wake(queue, enqueued, *current, p);
            }
        }
    }

    /// Enqueue `p` unless it is the running propagator or already queued.
    fn wake(
        queue: &mut VecDeque<PropId>,
        enqueued: &mut [bool],
        current: Option<PropId>,
        p: PropId,
    ) {
        if current == Some(p) {
            return;
        }
        if !enqueued[p.index()] {
            enqueued[p.index()] = true;
            queue.push_back(p);
        }
    }

    /// Sum of `weights` over the propagators constraining `var` (its weighted
    /// degree, for `dom/wdeg`). At least 1 so it is safe as a divisor.
    pub fn var_weight(&self, var: VarId, weights: &[u64]) -> u64 {
        let s = &self.subs[var.index()];
        let mut w = 0u64;
        for &p in s.on_dom.iter().chain(&s.on_bnd).chain(&s.on_fix) {
            w += weights[p.index()];
        }
        w.max(1)
    }

    // --- search levels (delegated to the trail) ---

    /// Open a search level.
    pub fn push_level(&mut self) {
        self.trail.push_level();
    }

    /// Close the current search level, undoing its domain changes.
    pub fn pop_level(&mut self) {
        self.trail.pop_level();
    }

    /// Current search depth.
    pub fn level(&self) -> usize {
        self.trail.level()
    }
}

/// The solver: a [`Store`] plus the propagator objects, with the propagation
/// fixpoint.
#[derive(Default)]
pub struct Solver {
    /// All mutable state. Public so search and tests can read/branch on domains.
    pub store: Store,
    /// Propagator objects, indexed by `PropId`. `Option` so one can be taken
    /// out to run while still owning its slot.
    propagators: Vec<Option<Box<dyn Propagator>>>,
    /// `dom/wdeg` weights, indexed by `PropId`; bumped when a propagator fails.
    weights: Vec<u64>,
}

impl Solver {
    /// An empty solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a variable with domain `[lo, hi]`.
    pub fn new_var_range(&mut self, lo: i32, hi: i32) -> VarId {
        self.store.new_var_range(lo, hi)
    }

    /// Create a variable with an explicit set of domain values.
    pub fn new_var_set(&mut self, values: &[i32]) -> VarId {
        self.store.new_var_set(values)
    }

    /// Post a propagator: register it (subscribes to its variables) and enqueue
    /// it for an initial propagation.
    pub fn post(&mut self, mut prop: Box<dyn Propagator>) -> PropId {
        let id = self.store.alloc_prop();
        debug_assert_eq!(id.index(), self.propagators.len());
        prop.register(&mut self.store, id);
        self.propagators.push(Some(prop));
        self.weights.push(1);
        self.store.enqueue(id);
        id
    }

    /// Number of posted propagators.
    pub fn num_propagators(&self) -> usize {
        self.propagators.len()
    }

    /// The `dom/wdeg` weights, indexed by `PropId`.
    pub fn weights(&self) -> &[u64] {
        &self.weights
    }

    /// Enqueue every propagator. Called at the start of each search so a solve
    /// is self-contained even when the solver is reused.
    pub fn enqueue_all(&mut self) {
        for i in 0..self.propagators.len() {
            self.store.enqueue(PropId(i as u32));
        }
    }

    /// Run the propagation fixpoint until the queue empties (consistent) or a
    /// propagator reports `Inconsistency`.
    pub fn propagate(&mut self) -> Result<(), Inconsistency> {
        while let Some(id) = self.store.dequeue() {
            // Take the propagator out to satisfy the borrow checker, run it,
            // then put it straight back.
            let mut prop = self.propagators[id.index()]
                .take()
                .expect("running a propagator that is not present");
            self.store.current = Some(id);
            let result = prop.propagate(&mut self.store);
            self.store.current = None;
            self.propagators[id.index()] = Some(prop);

            if let Err(e) = result {
                // dom/wdeg: blame the propagator that wiped out a domain.
                self.weights[id.index()] += 1;
                self.store.clear_queue();
                return Err(e);
            }
        }
        Ok(())
    }
}
