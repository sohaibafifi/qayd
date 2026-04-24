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

/// A *primary* domain change, in domain terms, for the LCG layer to pick up.
///
/// The four [`Store`] mutators each emit one of these on a real change. They
/// name only the literal the op directly asserts (`remove_below(b)` asserts
/// `[x ≥ b]`, and so on); the engine translates it to a [`Lit`](crate::lcg::lit::Lit)
/// and records it. Secondary atom flips are left to channeling unit propagation,
/// so no per-value enumeration happens here. The buffer is inert unless an LCG
/// engine drains it — the plain FD solver never reads it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DomEvent {
    /// `remove_below(var, bound)` — asserts `[var ≥ bound]`.
    GeTrue { var: VarId, bound: i32 },
    /// `remove_above(var, bound)` — asserts `var ≤ bound`, i.e. `¬[var ≥ bound+1]`.
    LeTrue { var: VarId, bound: i32 },
    /// `remove(var, val)` — asserts `¬[var = val]`.
    NeTrue { var: VarId, val: i32 },
    /// `fix(var, val)` — asserts `[var = val]`.
    EqTrue { var: VarId, val: i32 },
}

/// A scope variable's domain, snapshotted *just before* a propagator's mutation.
///
/// This is the raw material for that propagator's explanation: a correct
/// propagator's inference is entailed by its scope domains at the time, so
/// recording them (per op, pre-op) gives a sound, hole-safe reason. `holes` are
/// the absent values strictly inside `(min, max)` — empty for a contiguous
/// range, so bounds-only propagators pay nothing extra.
#[derive(Clone, Debug)]
pub struct ScopeVar {
    pub var: VarId,
    pub min: i32,
    pub max: i32,
    pub holes: Vec<i32>,
}

/// A primary domain change plus, when a propagator caused it, the pre-op
/// snapshot of that propagator's scope. `scope` is empty for engine-driven
/// changes (decisions, unit propagation) — those carry their own reason.
#[derive(Clone, Debug)]
pub struct EngineEvent {
    pub primary: DomEvent,
    pub scope: Vec<ScopeVar>,
}

/// Outcome of running a single propagator via [`Solver::propagate_step`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepOutcome {
    /// The propagation queue was empty.
    Empty,
    /// This propagator ran successfully (its events are in the buffer).
    Ran(PropId),
    /// This propagator wiped out a domain.
    Failed(PropId),
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
    /// own filtering). The LCG layer also reads it to attribute a domain change
    /// to the propagator that caused it (for explanations).
    pub(crate) current: Option<PropId>,
    /// Variables each propagator constrains, indexed by `PropId` — the
    /// propagator's *scope*, gathered from its subscriptions. Used by the LCG
    /// layer to build a propagator's explanation; unused by plain FD search.
    scope: Vec<Vec<VarId>>,
    /// Primary domain-change events since the last drain (LCG layer only).
    pub(crate) events: Vec<EngineEvent>,
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
        // Snapshot the explanation scope only when the op will actually change
        // the domain — idempotent re-derivations (very common) skip the alloc.
        let cause = self.snapshot_if(self.domains[i].contains(val, &self.trail));
        let (old_min, old_max) = self.bounds(i);
        let changed = self.domains[i].remove(val, &mut self.trail);
        if changed {
            self.after_change(var, old_min, old_max)?;
            self.emit(DomEvent::NeTrue { var, val }, cause);
        }
        Ok(())
    }

    /// Fix `var` to `val`. `Err` if `val` is absent.
    pub fn fix(&mut self, var: VarId, val: i32) -> Result<(), Inconsistency> {
        let i = var.index();
        let will_change = {
            let dom = &self.domains[i];
            dom.contains(val, &self.trail) && !dom.is_fixed(&self.trail)
        };
        let cause = self.snapshot_if(will_change);
        let (old_min, old_max) = self.bounds(i);
        let changed = self.domains[i].fix(val, &mut self.trail)?;
        if changed {
            self.after_change(var, old_min, old_max)?;
            self.emit(DomEvent::EqTrue { var, val }, cause);
        }
        Ok(())
    }

    /// Remove all values `< bound` from `var`. `Err` on wipeout.
    pub fn remove_below(&mut self, var: VarId, bound: i32) -> Result<(), Inconsistency> {
        let i = var.index();
        let cause = self.snapshot_if(bound > self.domains[i].min(&self.trail));
        let (old_min, old_max) = self.bounds(i);
        let changed = self.domains[i].remove_below(bound, &mut self.trail);
        if changed {
            self.after_change(var, old_min, old_max)?;
            self.emit(DomEvent::GeTrue { var, bound }, cause);
        }
        Ok(())
    }

    /// Remove all values `> bound` from `var`. `Err` on wipeout.
    pub fn remove_above(&mut self, var: VarId, bound: i32) -> Result<(), Inconsistency> {
        let i = var.index();
        let cause = self.snapshot_if(bound < self.domains[i].max(&self.trail));
        let (old_min, old_max) = self.bounds(i);
        let changed = self.domains[i].remove_above(bound, &mut self.trail);
        if changed {
            self.after_change(var, old_min, old_max)?;
            self.emit(DomEvent::LeTrue { var, bound }, cause);
        }
        Ok(())
    }

    /// Snapshot the running propagator's scope only when `will_change` — so a
    /// no-op mutation pays nothing, and engine-driven changes (no propagator)
    /// stay snapshot-free.
    fn snapshot_if(&self, will_change: bool) -> Vec<ScopeVar> {
        if will_change && self.current.is_some() {
            self.snapshot_cause()
        } else {
            Vec::new()
        }
    }

    /// Snapshot the running propagator's scope domains (pre-op), or an empty
    /// vector when no propagator is running (engine-driven change).
    fn snapshot_cause(&self) -> Vec<ScopeVar> {
        match self.current {
            None => Vec::new(),
            Some(p) => self.scope[p.index()]
                .iter()
                .map(|&v| self.scope_var(v))
                .collect(),
        }
    }

    /// Snapshot one variable's current domain as a [`ScopeVar`].
    fn scope_var(&self, var: VarId) -> ScopeVar {
        let dom = &self.domains[var.index()];
        let min = dom.min(&self.trail);
        let max = dom.max(&self.trail);
        let size = dom.size(&self.trail) as i64;
        // Only enumerate holes when the domain is not a contiguous range.
        let holes = if size == max as i64 - min as i64 + 1 {
            Vec::new()
        } else {
            (min + 1..max)
                .filter(|&v| !dom.contains(v, &self.trail))
                .collect()
        };
        ScopeVar {
            var,
            min,
            max,
            holes,
        }
    }

    /// Append an engine event for a real domain change.
    fn emit(&mut self, primary: DomEvent, scope: Vec<ScopeVar>) {
        self.events.push(EngineEvent { primary, scope });
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
        let scope = &mut self.scope[prop.index()];
        if !scope.contains(&var) {
            scope.push(var);
        }
    }

    /// The variables `prop` constrains (its scope), gathered from subscriptions.
    pub fn scope(&self, prop: PropId) -> &[VarId] {
        &self.scope[prop.index()]
    }

    /// Reserve a scheduling slot for a new propagator and return its id. The
    /// returned id equals the number of propagators allocated so far, so the
    /// [`Solver`] can keep its `Box<dyn Propagator>` vector in lockstep.
    pub(crate) fn alloc_prop(&mut self) -> PropId {
        let id = PropId(self.enqueued.len() as u32);
        self.enqueued.push(false);
        self.scope.push(Vec::new());
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

    /// The next propagator that would run, without dequeuing it.
    pub(crate) fn peek_queue(&self) -> Option<PropId> {
        self.queue.front().copied()
    }

    /// Whether the propagation queue is empty.
    pub(crate) fn queue_is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Snapshot `prop`'s scope domains as they stand now into `buf` (reusing its
    /// capacity) — for building its explanation, e.g. a conflict reason captured
    /// before the propagator runs. Reused every step, so this allocates nothing
    /// for range domains.
    pub fn snapshot_scope_into(&self, prop: PropId, buf: &mut Vec<ScopeVar>) {
        buf.clear();
        for &v in &self.scope[prop.index()] {
            buf.push(self.scope_var(v));
        }
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
        self.propagate_until(|| false)
    }

    /// Run the propagation fixpoint, periodically polling `should_stop`. When it
    /// fires (e.g. a time limit during a very large root propagation), the queue
    /// is cleared and `Ok` is returned — the caller must then observe the stop
    /// and abandon the search (never treat this as a consistent fixpoint).
    pub fn propagate_until<F: Fn() -> bool>(
        &mut self,
        should_stop: F,
    ) -> Result<(), Inconsistency> {
        while let Some(id) = self.store.dequeue() {
            // Poll on every dequeue: a single propagator call can be slow on huge
            // instances, so a coarser interval would blow well past the limit.
            if should_stop() {
                self.store.clear_queue();
                return Ok(());
            }
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

    /// Run the next queued propagator, if any, leaving its primary events in the
    /// store buffer for the LCG layer to drain. Unlike [`propagate`](Self::propagate)
    /// this returns after a single propagator so the engine can attribute the
    /// events (and a conflict) to it.
    pub fn propagate_step(&mut self) -> StepOutcome {
        let Some(id) = self.store.dequeue() else {
            return StepOutcome::Empty;
        };
        let mut prop = self.propagators[id.index()]
            .take()
            .expect("running a propagator that is not present");
        self.store.current = Some(id);
        let result = prop.propagate(&mut self.store);
        self.store.current = None;
        self.propagators[id.index()] = Some(prop);
        match result {
            Ok(()) => StepOutcome::Ran(id),
            Err(_) => {
                self.weights[id.index()] += 1;
                self.store.clear_queue();
                StepOutcome::Failed(id)
            }
        }
    }

    /// The next propagator that would run (without dequeuing it).
    pub fn peek_prop(&self) -> Option<PropId> {
        self.store.peek_queue()
    }

    /// Whether the propagation queue is drained.
    pub fn fd_at_fixpoint(&self) -> bool {
        self.store.queue_is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::primitives::less_or_equal;

    #[test]
    fn mutators_emit_primary_events() {
        let mut store = Store::new();
        let x = store.new_var_range(0, 9);

        store.remove_below(x, 3).unwrap();
        store.remove_above(x, 7).unwrap();
        store.remove(x, 5).unwrap();
        let primaries: Vec<DomEvent> = store.events.iter().map(|e| e.primary).collect();
        assert_eq!(
            primaries,
            vec![
                DomEvent::GeTrue { var: x, bound: 3 },
                DomEvent::LeTrue { var: x, bound: 7 },
                DomEvent::NeTrue { var: x, val: 5 },
            ]
        );
        // No propagator running: events carry no scope snapshot.
        assert!(store.events.iter().all(|e| e.scope.is_empty()));

        store.events.clear();
        store.fix(x, 4).unwrap();
        assert_eq!(store.events.len(), 1);
        assert_eq!(store.events[0].primary, DomEvent::EqTrue { var: x, val: 4 });
    }

    #[test]
    fn no_op_mutations_emit_nothing() {
        let mut store = Store::new();
        let x = store.new_var_range(0, 9);
        store.remove_below(x, 0).unwrap(); // already the min
        store.remove_above(x, 9).unwrap(); // already the max
        store.remove(x, 100).unwrap(); // absent
        assert!(store.events.is_empty());
    }

    #[test]
    fn subscribe_records_scope() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 5);
        let y = solver.new_var_range(0, 5);
        let p = less_or_equal(&mut solver, x, y); // subscribes to both x and y
        let mut scope = solver.store.scope(p).to_vec();
        scope.sort();
        assert_eq!(scope, vec![x, y]);
    }
}
