//! Solver core. [`Store`] owns domains, trail, subscriptions, and the
//! propagation queue; [`Solver`] holds the propagator objects and drives the
//! event-driven fixpoint.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::domain::Domain;
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::trail::{ReversibleInt, Trail};

/// Per-variable subscriptions and their event granularity.
#[derive(Clone, Default)]
struct VarSubs {
    entries: Vec<(Event, PropId)>,
}

/// A primary domain change, in domain terms, for the LCG layer to pick up.
/// Inert unless an LCG engine drains it; plain FD search never reads it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DomEvent {
    /// `remove_below(var, bound)` asserts `[var ≥ bound]`.
    GeTrue { var: VarId, bound: i32 },
    /// `remove_above(var, bound)` asserts `var ≤ bound`, i.e. `¬[var ≥ bound+1]`.
    LeTrue { var: VarId, bound: i32 },
    /// `remove(var, val)` asserts `¬[var = val]`.
    NeTrue { var: VarId, val: i32 },
    /// `fix(var, val)` asserts `[var = val]`.
    EqTrue { var: VarId, val: i32 },
}

/// An atomic domain fact a propagator can cite as a premise in its tight
/// explanation: the domain-layer mirror of an LCG literal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Premise {
    /// `[var ≥ bound]`.
    Ge { var: VarId, bound: i32 },
    /// `[var ≤ bound]`.
    Le { var: VarId, bound: i32 },
    /// `[var = val]`.
    Eq { var: VarId, val: i32 },
    /// `[var ≠ val]`.
    Ne { var: VarId, val: i32 },
}

/// A scope variable's domain, snapshotted just before a propagator's mutation.
/// `holes` are absent root-supported values strictly inside `(min, max)`.
#[derive(Clone, Debug)]
pub struct ScopeVar {
    pub var: VarId,
    pub min: i32,
    pub max: i32,
    pub holes: Vec<i32>,
}

/// Why a domain change happened: the raw material for its LCG explanation.
#[derive(Clone, Debug)]
pub enum Cause {
    /// Generic fallback: use the LCG engine's pre-propagator scope snapshot.
    Scope,
    /// Tight propagator-supplied premises that entail the change; set via the
    /// `*_because` mutators.
    Premises(Vec<Premise>),
}

/// A primary domain change plus why it happened ([`Cause`]).
#[derive(Clone, Debug)]
pub struct EngineEvent {
    pub primary: DomEvent,
    pub cause: Cause,
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
#[derive(Clone, Default)]
pub struct Store {
    domains: Vec<Domain>,
    trail: Trail,
    subs: Vec<VarSubs>,
    /// Variables used by at least one propagator, including root-only filters.
    relevant: Vec<bool>,
    /// Propagators waiting to run.
    queue: VecDeque<PropId>,
    /// `enqueued[prop]` guards against queuing a propagator twice.
    enqueued: Vec<bool>,
    /// The propagator currently running, if any, so it is not re-woken by its
    /// own filtering. The LCG layer reads it to attribute changes.
    pub(crate) current: Option<PropId>,
    /// Each propagator's scope (constrained variables), gathered from
    /// subscriptions; indexed by `PropId`. LCG only.
    scope: Vec<Vec<VarId>>,
    /// Primary domain-change events since the last drain (LCG layer only).
    pub(crate) events: Vec<EngineEvent>,
    /// Tight reason staged by a `*_because` mutator; bound to its next real
    /// change, else falls back to a scope snapshot.
    pending_premises: Option<Vec<Premise>>,
    /// Tight reason staged by [`fail_because`](Store::fail_because); drained by
    /// the LCG layer on a conflict.
    pending_conflict: Option<Vec<Premise>>,
    /// Ablation switch: always explain with the whole-scope snapshot, ignoring
    /// propagator-supplied premises. Off in normal use.
    force_scope_reasons: bool,
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
        self.relevant.push(false);
        id
    }

    /// Create a variable with an explicit set of domain values.
    pub fn new_var_set(&mut self, values: &[i32]) -> VarId {
        let id = VarId(self.domains.len() as u32);
        let dom = Domain::new_set(values, &mut self.trail);
        self.domains.push(dom);
        self.subs.push(VarSubs::default());
        self.relevant.push(false);
        id
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.domains.len()
    }

    /// Number of domains using cardinality-sized sparse storage.
    pub fn num_sparse_domains(&self) -> usize {
        self.domains.iter().filter(|dom| dom.is_sparse()).count()
    }

    /// Number of wide contiguous domains using bounds-only storage.
    pub fn num_bounds_domains(&self) -> usize {
        self.domains.iter().filter(|dom| dom.is_bounds()).count()
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

    /// The single value of a fixed variable. Panics if `var` is not fixed.
    pub fn value(&self, var: VarId) -> i32 {
        assert!(self.is_fixed(var), "value() called on unfixed {var:?}");
        self.domains[var.index()].min(&self.trail)
    }

    /// Iterate the present values of `var` (arbitrary order).
    pub fn values(&self, var: VarId) -> impl Iterator<Item = i32> + '_ {
        self.domains[var.index()].values(&self.trail)
    }

    /// Root support when `var` uses cardinality-sized sparse storage.
    pub(crate) fn sparse_values(&self, var: VarId) -> Option<Arc<[i32]>> {
        self.domains[var.index()].sparse_values()
    }

    // --- domain mutations (each wakes subscribers on a real change) ---

    /// Remove `val` from `var`. `Err` on wipeout.
    pub fn remove(&mut self, var: VarId, val: i32) -> Result<(), Inconsistency> {
        let i = var.index();
        // Pick the reason only when the op will actually change the domain.
        let cause = self.cause_for(self.domains[i].contains(val, &self.trail));
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
        let cause = self.cause_for(will_change);
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
        let cause = self.cause_for(bound > self.domains[i].min(&self.trail));
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
        let cause = self.cause_for(bound < self.domains[i].max(&self.trail));
        let (old_min, old_max) = self.bounds(i);
        let changed = self.domains[i].remove_above(bound, &mut self.trail);
        if changed {
            self.after_change(var, old_min, old_max)?;
            self.emit(DomEvent::LeTrue { var, bound }, cause);
        }
        Ok(())
    }

    /// Reason for an imminent change: staged premises if pending, else the
    /// pre-propagator scope fallback. Always clears the pending premises.
    fn cause_for(&mut self, will_change: bool) -> Cause {
        let pending = self.pending_premises.take();
        match pending {
            Some(p) if !self.force_scope_reasons => {
                Cause::Premises(if will_change { p } else { Vec::new() })
            }
            _ => Cause::Scope,
        }
    }

    /// Set the [`force_scope_reasons`](Store::force_scope_reasons) ablation switch.
    pub(crate) fn set_force_scope_reasons(&mut self, on: bool) {
        self.force_scope_reasons = on;
    }

    // --- domain mutations with a tight, propagator-supplied reason ---
    //
    // Each stages `why` (premises true now that entail the change) for the next
    // real change, then calls the plain mutator.

    /// [`remove`](Store::remove), explained by `why`.
    pub fn remove_because(
        &mut self,
        var: VarId,
        val: i32,
        why: Vec<Premise>,
    ) -> Result<(), Inconsistency> {
        self.pending_premises = Some(why);
        self.remove(var, val)
    }

    /// [`fix`](Store::fix), explained by `why`.
    pub fn fix_because(
        &mut self,
        var: VarId,
        val: i32,
        why: Vec<Premise>,
    ) -> Result<(), Inconsistency> {
        self.pending_premises = Some(why);
        self.fix(var, val)
    }

    /// [`remove_below`](Store::remove_below), explained by `why`.
    pub fn remove_below_because(
        &mut self,
        var: VarId,
        bound: i32,
        why: Vec<Premise>,
    ) -> Result<(), Inconsistency> {
        self.pending_premises = Some(why);
        self.remove_below(var, bound)
    }

    /// [`remove_above`](Store::remove_above), explained by `why`.
    pub fn remove_above_because(
        &mut self,
        var: VarId,
        bound: i32,
        why: Vec<Premise>,
    ) -> Result<(), Inconsistency> {
        self.pending_premises = Some(why);
        self.remove_above(var, bound)
    }

    /// Report inconsistency with a tight conflict reason (`why`: premises true
    /// now that are jointly infeasible). Use as `return Err(store.fail_because(..))`.
    pub fn fail_because(&mut self, why: Vec<Premise>) -> Inconsistency {
        self.pending_conflict = Some(why);
        Inconsistency
    }

    /// Take any conflict reason staged by [`fail_because`](Store::fail_because).
    /// `None` under the `force_scope_reasons` ablation.
    pub(crate) fn take_pending_conflict(&mut self) -> Option<Vec<Premise>> {
        let p = self.pending_conflict.take();
        if self.force_scope_reasons {
            None
        } else {
            p
        }
    }

    /// Clear any staged-but-unconsumed reason so it never leaks across runs.
    pub(crate) fn clear_pending(&mut self) {
        self.pending_premises = None;
        self.pending_conflict = None;
    }

    /// Snapshot one variable's current domain as a [`ScopeVar`].
    fn scope_var(&self, var: VarId) -> ScopeVar {
        let dom = &self.domains[var.index()];
        let min = dom.min(&self.trail);
        let max = dom.max(&self.trail);
        let size = dom.size(&self.trail) as i64;
        // Enumerate only root-supported holes so sparse numeric gaps stay compact.
        let mut holes = Vec::new();
        if size != max as i64 - min as i64 + 1 {
            dom.append_removed_values(&self.trail, &mut holes);
        }
        ScopeVar {
            var,
            min,
            max,
            holes,
        }
    }

    /// Append an engine event for a real domain change.
    fn emit(&mut self, primary: DomEvent, cause: Cause) {
        self.events.push(EngineEvent { primary, cause });
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
    // Incremental propagator state that must be restored on backtrack.

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
        self.mark_relevant(var);
        let s = &mut self.subs[var.index()];
        if !s.entries.contains(&(event, prop)) {
            s.entries.push((event, prop));
        }
        let scope = &mut self.scope[prop.index()];
        if !scope.contains(&var) {
            scope.push(var);
        }
    }

    /// Mark a variable as used by a propagator that does not need wakeups.
    pub fn mark_relevant(&mut self, var: VarId) {
        self.relevant[var.index()] = true;
    }

    /// Whether any posted propagator uses `var`.
    pub fn is_relevant(&self, var: VarId) -> bool {
        self.relevant[var.index()]
    }

    /// The variables `prop` constrains (its scope), gathered from subscriptions.
    pub fn scope(&self, prop: PropId) -> &[VarId] {
        &self.scope[prop.index()]
    }

    /// Reserve a scheduling slot for a new propagator and return its id.
    /// The id equals the count of propagators allocated so far.
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

    /// Snapshot `prop`'s current scope domains into `buf` (reusing its capacity;
    /// no allocation for range domains).
    pub fn snapshot_scope_into(&self, prop: PropId, buf: &mut Vec<ScopeVar>) {
        buf.clear();
        for &v in &self.scope[prop.index()] {
            buf.push(self.scope_var(v));
        }
    }

    /// Drop every queued propagator (after a failure, so stale entries do not
    /// leak into the sibling branch).
    pub(crate) fn clear_queue(&mut self) {
        while let Some(p) = self.queue.pop_front() {
            self.enqueued[p.index()] = false;
        }
    }

    /// Wake the appropriate subscribers of `var` (except the running one).
    fn notify(&mut self, var: VarId, bounds_moved: bool, fixed: bool) {
        // Split the borrow: read subscription lists while mutating queue/flags.
        // Subscription lists never change during propagation.
        let Store {
            subs,
            queue,
            enqueued,
            current,
            ..
        } = self;
        let s = &subs[var.index()];
        for &(event, p) in &s.entries {
            if event == Event::DomainChange
                || (bounds_moved && event == Event::BoundChange)
                || (fixed && event == Event::Fix)
            {
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
        for &(_, p) in &s.entries {
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

/// A [`Store`] plus the propagator objects, with the propagation fixpoint.
#[derive(Clone, Default)]
pub struct Solver {
    /// All mutable state. Public so search and tests can read/branch on domains.
    pub store: Store,
    /// Propagator objects, indexed by `PropId`. `Option` so one can be taken
    /// out to run while still owning its slot.
    propagators: Vec<Option<Box<dyn Propagator>>>,
    /// `dom/wdeg` weights, indexed by `PropId`; bumped when a propagator fails.
    weights: Vec<u64>,
}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<Store>();
    assert_send::<Solver>();
};

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

    /// Enqueue every propagator (at the start of each search, so a reused
    /// solver still produces a self-contained solve).
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

    /// Run the propagation fixpoint, polling `should_stop`. On stop, clears the
    /// queue and returns `Ok`; the caller must abandon the search, never treat
    /// this as a consistent fixpoint.
    pub fn propagate_until<F: Fn() -> bool>(
        &mut self,
        should_stop: F,
    ) -> Result<(), Inconsistency> {
        while let Some(id) = self.store.dequeue() {
            // Poll every dequeue: one propagator call can be slow on huge instances.
            if should_stop() {
                self.store.clear_queue();
                return Ok(());
            }
            self.run_prop(id)?;
        }
        Ok(())
    }

    /// Run the next queued propagator (if any), leaving its primary events in
    /// the store buffer. Returns after one propagator so the LCG engine can
    /// attribute the events and any conflict to it.
    pub fn propagate_step(&mut self) -> StepOutcome {
        let Some(id) = self.store.dequeue() else {
            return StepOutcome::Empty;
        };
        match self.run_prop(id) {
            Ok(()) => StepOutcome::Ran(id),
            Err(_) => StepOutcome::Failed(id),
        }
    }

    fn run_prop(&mut self, id: PropId) -> Result<(), Inconsistency> {
        let mut prop = self.propagators[id.index()]
            .take()
            .expect("running a propagator that is not present");
        self.store.clear_pending();
        self.store.current = Some(id);
        let result = prop.propagate(&mut self.store);
        self.store.current = None;
        self.propagators[id.index()] = Some(prop);
        if let Err(e) = result {
            self.weights[id.index()] += 1;
            self.store.clear_queue();
            return Err(e);
        }
        Ok(())
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
