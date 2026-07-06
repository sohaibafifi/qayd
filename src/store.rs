//! Solver core. [`Store`] owns domains, trail, subscriptions, and the
//! propagation queue; [`Solver`] holds the propagator objects and drives the
//! event-driven fixpoint.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::domains::int::Domain;
use crate::domains::interval::{IntervalDomain, IntervalEvent, IntervalPresence};
use crate::domains::list::{ListDomain, ListEvent};
use crate::ids::{IntervalId, ListId, PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator, Selected, NBANDS};
use crate::trail::{ReversibleInt, Trail};

thread_local! {
    /// Per-thread count of propagator invocations (a scheduling diagnostic;
    /// no effect on results). Read via [`prop_calls`].
    static PROP_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Per-thread count of `Expensive`-band invocations — the calls whose cost
    /// band ordering is meant to reduce. Read via [`expensive_calls`].
    static EXPENSIVE_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Propagator invocations on this thread since the last [`reset_prop_calls`].
pub fn prop_calls() -> u64 {
    PROP_CALLS.with(std::cell::Cell::get)
}

/// `Expensive`-band invocations on this thread since the last [`reset_prop_calls`].
pub fn expensive_calls() -> u64 {
    EXPENSIVE_CALLS.with(std::cell::Cell::get)
}

/// Reset this thread's propagator-invocation counters.
pub fn reset_prop_calls() {
    PROP_CALLS.with(|c| c.set(0));
    EXPENSIVE_CALLS.with(|c| c.set(0));
}

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

impl Premise {
    /// The variable this premise constrains.
    fn var(&self) -> VarId {
        match *self {
            Premise::Ge { var, .. } | Premise::Le { var, .. } | Premise::Eq { var, .. } | Premise::Ne { var, .. } => var,
        }
    }
}

/// Number of distinct variables mentioned by a tight reason: its witness size.
fn distinct_premise_vars(why: &[Premise]) -> u32 {
    let mut vars: Vec<u32> = why.iter().map(|p| p.var().0).collect();
    vars.sort_unstable();
    vars.dedup();
    vars.len() as u32
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
    list_domains: Vec<ListDomain>,
    interval_domains: Vec<IntervalDomain>,
    trail: Trail,
    subs: Vec<VarSubs>,
    /// Cached weighted degree per variable: the sum of `dom/wdeg` weights over
    /// each of its subscription entries (mirrors [`var_weight`](Store::var_weight)).
    /// Grown per entry in [`subscribe`](Store::subscribe) and kept in sync with
    /// weight bumps by [`bump_var_weight`](Store::bump_var_weight), so the
    /// brancher reads it in O(1) instead of rescanning every subscription.
    var_weight_cache: Vec<u64>,
    /// Variables used by at least one propagator, including root-only filters.
    relevant: Vec<bool>,
    /// Propagators waiting to run, bucketed by cost band; the lowest non-empty
    /// band drains first so cheap propagators reach fixpoint before pricier ones.
    queues: [VecDeque<PropId>; NBANDS],
    /// Each propagator's cost band, indexed by `PropId`; picks its queue.
    prio: Vec<Priority>,
    /// Diagnostic: collapse all bands into one FIFO (the pre-banding scheduler),
    /// so a benchmark can compare band ordering in-process. Off in normal use.
    single_band: bool,
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
    /// A premise the [`Selected`](crate::propagator::Selected) wrapper stamps
    /// onto every tight reason its inner propagator emits, so the selector
    /// literal appears in the nogood (letting an unsat core trace to it). `None`
    /// outside a selected propagator's `propagate`.
    injected_premise: Option<Premise>,
    /// Research instrumentation (off by default): when `Some`, every tight
    /// propagator reason records `(distinct witness variables, propagator scope
    /// size, kind)` with `kind` = 1 for a failure, 0 for an inference. Used by the
    /// MUS "finesse" study to measure how small a global's Hall-set / energy-window
    /// witness is versus its full scope.
    finesse: Option<Vec<(u32, u32, u8)>>,
    /// Whether an LCG trail consumes reasons. Off outside learning, so
    /// propagators with costly explanation construction can skip it entirely.
    explain: bool,
    /// Unary-resource interval pairs registered by `no_overlap`. Each pair's order
    /// is a boolean variable (`1` = first-before-second, `0` = second-before-first,
    /// unfixed = undecided), so the order decision is a first-class atom the
    /// learning engine can branch on and learn over. `no_overlap` subscribes to it
    /// and enforces the decided precedence.
    disjunctive_pairs: Vec<(IntervalId, IntervalId)>,
    disjunctive_order_vars: Vec<VarId>,
}

impl Store {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a unary-resource pair owned by propagator `prop`, returning its
    /// index. The order is a boolean variable, undecided until fixed; `prop` is
    /// woken whenever it is decided (including by a learning-engine branch).
    pub fn register_disjunctive_pair(&mut self, a: IntervalId, b: IntervalId, prop: PropId) -> usize {
        let order = self.new_var_range(0, 1);
        self.subscribe(order, prop, Event::Fix);
        let index = self.disjunctive_pairs.len();
        self.disjunctive_pairs.push((a, b));
        self.disjunctive_order_vars.push(order);
        index
    }

    /// Number of registered unary-resource pairs.
    pub fn disjunctive_pair_count(&self) -> usize {
        self.disjunctive_pairs.len()
    }

    /// The intervals of registered pair `index`.
    pub fn disjunctive_pair(&self, index: usize) -> (IntervalId, IntervalId) {
        self.disjunctive_pairs[index]
    }

    /// The boolean order variable of pair `index` (`1` a->b, `0` b->a).
    pub fn disjunctive_order_var(&self, index: usize) -> VarId {
        self.disjunctive_order_vars[index]
    }

    /// Current order decision of pair `index` (0 undecided, 1 a->b, 2 b->a).
    pub fn disjunctive_order(&self, index: usize) -> i32 {
        let var = self.disjunctive_order_vars[index];
        if !self.is_fixed(var) {
            0
        } else if self.value(var) == 1 {
            1
        } else {
            2
        }
    }

    /// Decide the order of pair `index`: `1` (a->b) fixes the variable to 1, `2`
    /// (b->a) to 0. Fixing it wakes the owning propagator (a subscriber). The
    /// opposite of an already-decided order is inconsistent. Returns whether it
    /// changed.
    pub fn set_disjunctive_order(&mut self, index: usize, order: i32) -> Result<bool, Inconsistency> {
        debug_assert!(order == 1 || order == 2, "order decision must be 1 or 2");
        let var = self.disjunctive_order_vars[index];
        let target = if order == 1 { 1 } else { 0 };
        if self.is_fixed(var) {
            if self.value(var) == target {
                Ok(false)
            } else {
                Err(Inconsistency) // opposite order already decided
            }
        } else {
            self.fix(var, target)?;
            Ok(true)
        }
    }

    /// [`set_disjunctive_order`](Store::set_disjunctive_order), explained by `why`
    /// (the reason the order is forced). The opposite-of-decided conflict cites
    /// the current order, like the explained presence mutator.
    pub fn set_disjunctive_order_because(&mut self, index: usize, order: i32, why: Vec<Premise>) -> Result<bool, Inconsistency> {
        debug_assert!(order == 1 || order == 2, "order decision must be 1 or 2");
        let var = self.disjunctive_order_vars[index];
        let target = if order == 1 { 1 } else { 0 };
        if self.is_fixed(var) {
            if self.value(var) == target {
                Ok(false)
            } else {
                let opposite = self.value(var);
                let mut conflict = why;
                conflict.push(Premise::Eq { var, val: opposite });
                Err(self.fail_because(conflict))
            }
        } else {
            self.fix_because(var, target, why)?;
            Ok(true)
        }
    }

    // --- variable creation ---

    /// Create a variable with domain `[lo, hi]`.
    pub fn new_var_range(&mut self, lo: i32, hi: i32) -> VarId {
        let id = VarId(self.domains.len() as u32);
        let dom = Domain::new_range(lo, hi, &mut self.trail);
        self.domains.push(dom);
        self.subs.push(VarSubs::default());
        self.var_weight_cache.push(0);
        self.relevant.push(false);
        id
    }

    /// Create a variable with an explicit set of domain values.
    pub fn new_var_set(&mut self, values: &[i32]) -> VarId {
        let id = VarId(self.domains.len() as u32);
        let dom = Domain::new_set(values, &mut self.trail);
        self.domains.push(dom);
        self.subs.push(VarSubs::default());
        self.var_weight_cache.push(0);
        self.relevant.push(false);
        id
    }

    /// Create a list domain over a fixed item universe.
    pub fn new_list(&mut self, universe: Vec<i32>) -> ListId {
        assert!(i32::try_from(universe.len()).is_ok(), "list universe is too large");
        let mut sorted = universe.clone();
        sorted.sort_unstable();
        assert!(sorted.windows(2).all(|w| w[0] != w[1]), "list universe contains duplicate items");
        let n = universe.len();
        // One boolean membership variable per item: 1 = in the list, 0 = excluded.
        let members: Vec<VarId> = (0..n).map(|_| self.new_var_range(0, 1)).collect();
        // The length variable; `sum(members) == length` is the ListCardinality
        // propagator's job, posted per list at build time.
        let length = self.new_var_range(0, n as i32);
        let id = ListId(self.list_domains.len() as u32);
        self.list_domains.push(ListDomain { universe, members, length });
        id
    }

    /// Create a mandatory fixed-duration interval.
    pub fn new_interval(&mut self, start_min: i32, start_max: i32, duration: i32) -> IntervalId {
        self.new_interval_with_presence(start_min, start_max, duration, IntervalPresence::Present)
    }

    /// Create an optional fixed-duration interval.
    pub fn new_optional_interval(&mut self, start_min: i32, start_max: i32, duration: i32) -> IntervalId {
        self.new_interval_with_presence(start_min, start_max, duration, IntervalPresence::Optional)
    }

    fn new_interval_with_presence(&mut self, start_min: i32, start_max: i32, duration: i32, presence: IntervalPresence) -> IntervalId {
        assert!(start_min <= start_max, "interval start_min must be <= start_max");
        assert!(duration >= 0, "interval duration must be non-negative");
        // The start is a real integer variable; an optional interval gets a bool
        // presence variable. This makes start bounds and presence first-class
        // atoms for the learning engine.
        let start = self.new_var_range(start_min, start_max);
        let presence = match presence {
            IntervalPresence::Present => None,
            IntervalPresence::Optional => Some(self.new_var_range(0, 1)),
            IntervalPresence::Absent => {
                let var = self.new_var_range(0, 0);
                Some(var)
            }
        };
        let id = IntervalId(self.interval_domains.len() as u32);
        self.interval_domains.push(IntervalDomain { start, duration, presence });
        id
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.domains.len()
    }

    /// Number of list domains.
    pub fn num_lists(&self) -> usize {
        self.list_domains.len()
    }

    /// Number of interval domains.
    pub fn num_intervals(&self) -> usize {
        self.interval_domains.len()
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

    /// Immutable universe of a list.
    pub fn list_universe(&self, list: ListId) -> &[i32] {
        self.list_domains[list.index()].universe()
    }

    /// Structured list length lower bound.
    pub fn list_len_min(&self, list: ListId) -> usize {
        self.min(self.list_domains[list.index()].length).max(0) as usize
    }

    /// Structured list length upper bound.
    pub fn list_len_max(&self, list: ListId) -> usize {
        self.max(self.list_domains[list.index()].length).max(0) as usize
    }

    /// The integer length variable backing `list`.
    pub fn list_length_var(&self, list: ListId) -> VarId {
        self.list_domains[list.index()].length
    }

    /// Whether `item` can still appear in `list` (its membership variable still
    /// admits `1`).
    pub fn list_possible(&self, list: ListId, item: i32) -> bool {
        self.list_domains[list.index()].index_of(item).is_some_and(|i| self.contains(self.list_domains[list.index()].members[i], 1))
    }

    /// Whether `item` must appear in `list` (its membership variable is fixed to `1`).
    pub fn list_required(&self, list: ListId, item: i32) -> bool {
        self.list_domains[list.index()].index_of(item).is_some_and(|i| {
            let var = self.list_domains[list.index()].members[i];
            self.is_fixed(var) && self.value(var) == 1
        })
    }

    /// Current count of possible items in `list`.
    pub fn list_possible_count(&self, list: ListId) -> usize {
        self.list_domains[list.index()].members.iter().filter(|&&m| self.contains(m, 1)).count()
    }

    /// Current count of required items in `list`.
    pub fn list_required_count(&self, list: ListId) -> usize {
        self.list_domains[list.index()].members.iter().filter(|&&m| self.is_fixed(m) && self.value(m) == 1).count()
    }

    /// Interval start lower bound.
    pub fn interval_start_min(&self, interval: IntervalId) -> i32 {
        self.min(self.interval_domains[interval.index()].start)
    }

    /// Interval start upper bound.
    pub fn interval_start_max(&self, interval: IntervalId) -> i32 {
        self.max(self.interval_domains[interval.index()].start)
    }

    /// Interval end lower bound.
    pub fn interval_end_min(&self, interval: IntervalId) -> i32 {
        let dom = self.interval_domains[interval.index()];
        self.min(dom.start).saturating_add(dom.duration)
    }

    /// Interval end upper bound.
    pub fn interval_end_max(&self, interval: IntervalId) -> i32 {
        let dom = self.interval_domains[interval.index()];
        self.max(dom.start).saturating_add(dom.duration)
    }

    /// Fixed interval duration.
    pub fn interval_duration(&self, interval: IntervalId) -> i32 {
        self.interval_domains[interval.index()].duration
    }

    /// Interval presence state.
    pub fn interval_presence(&self, interval: IntervalId) -> IntervalPresence {
        match self.interval_domains[interval.index()].presence {
            None => IntervalPresence::Present,
            Some(var) if !self.is_fixed(var) => IntervalPresence::Optional,
            Some(var) if self.value(var) == 1 => IntervalPresence::Present,
            Some(_) => IntervalPresence::Absent,
        }
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

    /// Require an item in a list (fix its membership variable to `1`). The
    /// length/membership coupling is enforced separately by `ListCardinality`.
    pub fn require_list_item(&mut self, list: ListId, item: i32) -> Result<bool, Inconsistency> {
        let Some(i) = self.list_domains[list.index()].index_of(item) else {
            return Err(Inconsistency);
        };
        let member = self.list_domains[list.index()].members[i];
        if self.is_fixed(member) {
            return if self.value(member) == 1 { Ok(false) } else { Err(Inconsistency) };
        }
        self.fix(member, 1)?;
        Ok(true)
    }

    /// Forbid an item from a list (fix its membership variable to `0`).
    pub fn forbid_list_item(&mut self, list: ListId, item: i32) -> Result<bool, Inconsistency> {
        let Some(i) = self.list_domains[list.index()].index_of(item) else {
            return Ok(false);
        };
        let member = self.list_domains[list.index()].members[i];
        if self.is_fixed(member) {
            return if self.value(member) == 0 { Ok(false) } else { Err(Inconsistency) };
        }
        self.fix(member, 0)?;
        Ok(true)
    }

    /// Raise a list length lower bound (on the length variable).
    pub fn set_list_len_min(&mut self, list: ListId, min: usize) -> Result<bool, Inconsistency> {
        let length = self.list_domains[list.index()].length;
        let before = self.min(length);
        self.remove_below(length, min as i32)?;
        Ok(self.min(length) != before)
    }

    /// Lower a list length upper bound (on the length variable).
    pub fn set_list_len_max(&mut self, list: ListId, max: usize) -> Result<bool, Inconsistency> {
        let length = self.list_domains[list.index()].length;
        let before = self.max(length);
        self.remove_above(length, max as i32)?;
        Ok(self.max(length) != before)
    }

    /// The boolean membership variable of `item` in `list` (`None` if `item` is
    /// not in the list's universe).
    pub fn list_member_var(&self, list: ListId, item: i32) -> Option<VarId> {
        let dom = &self.list_domains[list.index()];
        dom.index_of(item).map(|i| dom.members[i])
    }

    /// [`require_list_item`](Store::require_list_item), explained by `why`. The
    /// opposite-of-decided conflict cites the current membership, like the
    /// explained interval presence mutator.
    pub fn require_list_item_because(&mut self, list: ListId, item: i32, why: Vec<Premise>) -> Result<bool, Inconsistency> {
        let Some(i) = self.list_domains[list.index()].index_of(item) else {
            return Err(self.fail_because(why));
        };
        let member = self.list_domains[list.index()].members[i];
        if self.is_fixed(member) {
            if self.value(member) == 1 {
                return Ok(false);
            }
            let mut conflict = why;
            conflict.push(Premise::Eq { var: member, val: 0 });
            return Err(self.fail_because(conflict));
        }
        self.fix_because(member, 1, why)?;
        Ok(true)
    }

    /// [`forbid_list_item`](Store::forbid_list_item), explained by `why`.
    pub fn forbid_list_item_because(&mut self, list: ListId, item: i32, why: Vec<Premise>) -> Result<bool, Inconsistency> {
        let Some(i) = self.list_domains[list.index()].index_of(item) else {
            return Ok(false);
        };
        let member = self.list_domains[list.index()].members[i];
        if self.is_fixed(member) {
            if self.value(member) == 0 {
                return Ok(false);
            }
            let mut conflict = why;
            conflict.push(Premise::Eq { var: member, val: 1 });
            return Err(self.fail_because(conflict));
        }
        self.fix_because(member, 0, why)?;
        Ok(true)
    }

    /// Raise an interval start lower bound.
    pub fn set_interval_start_min(&mut self, interval: IntervalId, min: i32) -> Result<bool, Inconsistency> {
        let start = self.interval_domains[interval.index()].start;
        let before = self.min(start);
        self.remove_below(start, min)?;
        Ok(self.min(start) != before)
    }

    /// Lower an interval start upper bound.
    pub fn set_interval_start_max(&mut self, interval: IntervalId, max: i32) -> Result<bool, Inconsistency> {
        let start = self.interval_domains[interval.index()].start;
        let before = self.max(start);
        self.remove_above(start, max)?;
        Ok(self.max(start) != before)
    }

    /// Require an optional interval to be present.
    pub fn require_interval_presence(&mut self, interval: IntervalId) -> Result<bool, Inconsistency> {
        self.fix_interval_presence(interval, 1)
    }

    /// Mark an optional interval absent.
    pub fn forbid_interval_presence(&mut self, interval: IntervalId) -> Result<bool, Inconsistency> {
        self.fix_interval_presence(interval, 0)
    }

    fn fix_interval_presence(&mut self, interval: IntervalId, value: i32) -> Result<bool, Inconsistency> {
        match self.interval_domains[interval.index()].presence {
            // A mandatory interval is permanently present: requiring is a no-op,
            // forbidding is inconsistent.
            None => {
                if value == 1 {
                    Ok(false)
                } else {
                    Err(Inconsistency)
                }
            }
            Some(var) => {
                if self.is_fixed(var) {
                    if self.value(var) == value {
                        Ok(false)
                    } else {
                        Err(Inconsistency)
                    }
                } else {
                    self.fix(var, value)?;
                    Ok(true)
                }
            }
        }
    }

    // --- integer-backing accessors and explained interval mutators (LCG) ---

    /// The integer start variable backing an interval.
    pub fn interval_start_var(&self, interval: IntervalId) -> VarId {
        self.interval_domains[interval.index()].start
    }

    /// The boolean presence variable backing an optional interval (`None` if the
    /// interval is mandatory).
    pub fn interval_presence_var(&self, interval: IntervalId) -> Option<VarId> {
        self.interval_domains[interval.index()].presence
    }

    /// [`set_interval_start_min`](Store::set_interval_start_min), explained by `why`.
    pub fn set_interval_start_min_because(&mut self, interval: IntervalId, min: i32, why: Vec<Premise>) -> Result<bool, Inconsistency> {
        let start = self.interval_domains[interval.index()].start;
        let before = self.min(start);
        self.remove_below_because(start, min, why)?;
        Ok(self.min(start) != before)
    }

    /// [`set_interval_start_max`](Store::set_interval_start_max), explained by `why`.
    pub fn set_interval_start_max_because(&mut self, interval: IntervalId, max: i32, why: Vec<Premise>) -> Result<bool, Inconsistency> {
        let start = self.interval_domains[interval.index()].start;
        let before = self.max(start);
        self.remove_above_because(start, max, why)?;
        Ok(self.max(start) != before)
    }

    /// [`require_interval_presence`](Store::require_interval_presence), explained by `why`.
    pub fn require_interval_presence_because(&mut self, interval: IntervalId, why: Vec<Premise>) -> Result<bool, Inconsistency> {
        self.fix_interval_presence_because(interval, 1, why)
    }

    /// [`forbid_interval_presence`](Store::forbid_interval_presence), explained by `why`.
    pub fn forbid_interval_presence_because(&mut self, interval: IntervalId, why: Vec<Premise>) -> Result<bool, Inconsistency> {
        self.fix_interval_presence_because(interval, 0, why)
    }

    fn fix_interval_presence_because(&mut self, interval: IntervalId, value: i32, why: Vec<Premise>) -> Result<bool, Inconsistency> {
        match self.interval_domains[interval.index()].presence {
            None => {
                if value == 1 {
                    Ok(false)
                } else {
                    Err(self.fail_because(why))
                }
            }
            Some(var) => {
                if self.is_fixed(var) {
                    if self.value(var) == value {
                        Ok(false)
                    } else {
                        // The conflict is `why AND presence = opposite`, not `why`
                        // alone: cite the current (opposite) presence so the
                        // engine cannot learn `not why` and prune valid branches.
                        let opposite = self.value(var);
                        let mut conflict = why;
                        conflict.push(Premise::Eq { var, val: opposite });
                        Err(self.fail_because(conflict))
                    }
                } else {
                    self.fix_because(var, value, why)?;
                    Ok(true)
                }
            }
        }
    }

    /// Reason for an imminent change: staged premises if pending, else the
    /// pre-propagator scope fallback. Always clears the pending premises.
    fn cause_for(&mut self, will_change: bool) -> Cause {
        let pending = self.pending_premises.take();
        match pending {
            Some(p) if !self.force_scope_reasons => {
                let mut p = if will_change { p } else { Vec::new() };
                // A real change under an active selector is entailed by the inner
                // premises *and* `sel = 1`; stamp the latter so the nogood carries it.
                if will_change {
                    self.record_finesse(&p, false);
                    if let Some(prem) = self.injected_premise {
                        p.push(prem);
                    }
                }
                Cause::Premises(p)
            }
            _ => Cause::Scope,
        }
    }

    /// Set the [`force_scope_reasons`](Store::force_scope_reasons) ablation switch.
    pub(crate) fn set_force_scope_reasons(&mut self, on: bool) {
        self.force_scope_reasons = on;
    }

    /// Stage the premise a [`Selected`](crate::propagator::Selected) wrapper
    /// attributes to every tight reason its inner propagator emits while active.
    pub(crate) fn set_injected_premise(&mut self, premise: Premise) {
        self.injected_premise = Some(premise);
    }

    /// Stop stamping the injected premise.
    pub(crate) fn clear_injected_premise(&mut self) {
        self.injected_premise = None;
    }

    /// Append `premise` to any staged [`fail_because`](Store::fail_because)
    /// conflict. A no-op when the inner propagator failed without a tight
    /// conflict (the whole-scope fallback already carries the selector).
    pub(crate) fn push_conflict_premise(&mut self, premise: Premise) {
        if let Some(conflict) = &mut self.pending_conflict {
            conflict.push(premise);
        }
    }

    /// Mark that an LCG trail consumes reasons (set by the learning engine).
    pub(crate) fn set_explain(&mut self, on: bool) {
        self.explain = on;
    }

    /// Whether reasons are consumed: propagators may skip building costly
    /// explanations when this is off.
    #[inline]
    pub fn explaining(&self) -> bool {
        self.explain
    }

    // --- domain mutations with a tight, propagator-supplied reason ---
    //
    // Each stages `why` (premises true now that entail the change) for the next
    // real change, then calls the plain mutator.

    /// [`remove`](Store::remove), explained by `why`.
    pub fn remove_because(&mut self, var: VarId, val: i32, why: Vec<Premise>) -> Result<(), Inconsistency> {
        self.pending_premises = Some(why);
        self.remove(var, val)
    }

    /// [`fix`](Store::fix), explained by `why`.
    pub fn fix_because(&mut self, var: VarId, val: i32, why: Vec<Premise>) -> Result<(), Inconsistency> {
        self.pending_premises = Some(why);
        self.fix(var, val)
    }

    /// [`remove_below`](Store::remove_below), explained by `why`.
    pub fn remove_below_because(&mut self, var: VarId, bound: i32, why: Vec<Premise>) -> Result<(), Inconsistency> {
        self.pending_premises = Some(why);
        self.remove_below(var, bound)
    }

    /// [`remove_above`](Store::remove_above), explained by `why`.
    pub fn remove_above_because(&mut self, var: VarId, bound: i32, why: Vec<Premise>) -> Result<(), Inconsistency> {
        self.pending_premises = Some(why);
        self.remove_above(var, bound)
    }

    /// Report inconsistency with a tight conflict reason (`why`: premises true
    /// now that are jointly infeasible). Use as `return Err(store.fail_because(..))`.
    pub fn fail_because(&mut self, why: Vec<Premise>) -> Inconsistency {
        self.record_finesse(&why, true);
        self.pending_conflict = Some(why);
        Inconsistency
    }

    /// Sample the current tight reason's `(witness vars, scope, kind)` for the
    /// finesse kill-test, where `kind` is 1 for a failure and 0 for an inference.
    /// No-op unless [`enable_finesse`](Store::enable_finesse) is on.
    fn record_finesse(&mut self, why: &[Premise], failure: bool) {
        if self.finesse.is_none() {
            return;
        }
        let Some(pid) = self.current else { return };
        let witness = distinct_premise_vars(why);
        let scope = self.scope[pid.index()].len() as u32;
        self.finesse.as_mut().unwrap().push((witness, scope, u8::from(failure)));
    }

    /// Start recording finesse samples (research instrumentation).
    pub fn enable_finesse(&mut self) {
        self.finesse = Some(Vec::new());
    }

    /// Drain the finesse samples: `(distinct witness vars, scope size, priority index)`.
    pub fn take_finesse(&mut self) -> Vec<(u32, u32, u8)> {
        self.finesse.take().unwrap_or_default()
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
        self.injected_premise = None;
    }

    /// Append premises pinning `var`'s current domain exactly: `Ge min`,
    /// `Le max`, one `Ne` per hole — the same shape as the whole-scope
    /// fallback snapshot, so a subset of variables can never explain wider
    /// than the fallback does.
    pub fn domain_premises(&self, var: VarId, out: &mut Vec<Premise>) {
        let sv = self.scope_var(var);
        out.push(Premise::Ge { var, bound: sv.min });
        out.push(Premise::Le { var, bound: sv.max });
        out.extend(sv.holes.into_iter().map(|h| Premise::Ne { var, val: h }));
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
        ScopeVar { var, min, max, holes }
    }

    /// Append an engine event for a real domain change. Only the LCG trail
    /// drains `events`, and it runs with `explain` on; under plain search the
    /// Vec has no consumer, so skip the push (and its `Cause` payload) to avoid
    /// leaking one heap allocation per propagation for the whole enumeration.
    fn emit(&mut self, primary: DomEvent, cause: Cause) {
        if self.explain {
            self.events.push(EngineEvent { primary, cause });
        }
    }

    /// `(min, max)` of variable index `i`.
    fn bounds(&self, i: usize) -> (i32, i32) {
        let dom = &self.domains[i];
        (dom.min(&self.trail), dom.max(&self.trail))
    }

    /// Check for wipeout and wake the right subscribers after a real change.
    fn after_change(&mut self, var: VarId, old_min: i32, old_max: i32) -> Result<(), Inconsistency> {
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
            // A fresh entry adds one term (weight 1 at registration time) to this
            // variable's weighted degree; bumps later adjust it via `bump_var_weight`.
            self.var_weight_cache[var.index()] += 1;
        }
        // Duplicates are collapsed once by `finalize_scope` after registration,
        // keeping n-ary registration near-linear instead of O(n^2) contains scans.
        self.scope[prop.index()].push(var);
    }

    /// Subscribe `prop` to list changes at the given granularity.
    /// Idempotent per (list, prop, event).
    pub fn subscribe_list(&mut self, list: ListId, prop: PropId, event: ListEvent) {
        match event {
            // Membership facts are backed by the per-item variables, so wake on
            // their `Fix` events (set by the brancher or the learning engine).
            ListEvent::PossibleChange | ListEvent::RequiredChange => {
                let members = self.list_domains[list.index()].members.clone();
                for member in members {
                    self.subscribe(member, prop, Event::Fix);
                }
            }
            // The length is backed by a variable, so wake on its bound changes.
            ListEvent::LengthChange => {
                let length = self.list_domains[list.index()].length;
                self.subscribe(length, prop, Event::BoundChange);
            }
            // Order events are not wired to any backing variable yet (no order
            // propagator subscribes to them); they will map onto position/successor
            // variables when list order lands.
            ListEvent::ArcChange | ListEvent::PositionChange => {}
        }
    }

    /// Subscribe `prop` to interval changes. Interval facts are backed
    /// by integer variables, so this maps onto the variable event system: start/
    /// end bound changes are the start variable's bound changes, presence changes
    /// are the presence variable being fixed.
    pub fn subscribe_interval(&mut self, interval: IntervalId, prop: PropId, event: IntervalEvent) {
        let dom = self.interval_domains[interval.index()];
        match event {
            IntervalEvent::StartBoundChange | IntervalEvent::EndBoundChange => self.subscribe(dom.start, prop, Event::BoundChange),
            IntervalEvent::PresenceChange => {
                if let Some(var) = dom.presence {
                    self.subscribe(var, prop, Event::Fix);
                }
            }
            IntervalEvent::ModeChange => {}
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
        self.prio.push(Priority::Linear);
        self.scope.push(Vec::new());
        id
    }

    /// Set `prop`'s scheduling band (called once, at post time).
    pub(crate) fn set_priority(&mut self, prop: PropId, priority: Priority) {
        self.prio[prop.index()] = priority;
    }

    /// Collapse duplicate variables in `prop`'s scope, preserving first-seen
    /// order. Run once after registration so `subscribe` stays scan-free.
    pub(crate) fn finalize_scope(&mut self, prop: PropId) {
        let scope = &mut self.scope[prop.index()];
        let mut seen = std::collections::HashSet::new();
        scope.retain(|v| seen.insert(*v));
    }

    /// The queue index for `prop`: its band, or 0 when collapsed to one FIFO.
    fn band_of(&self, prop: PropId) -> usize {
        if self.single_band {
            0
        } else {
            self.prio[prop.index()].index()
        }
    }

    /// Enable single-band (FIFO) scheduling for benchmarking. Off in normal use.
    pub(crate) fn set_single_band(&mut self, on: bool) {
        self.single_band = on;
    }

    /// Put `prop` on its band's queue unless it is already queued.
    pub(crate) fn enqueue(&mut self, prop: PropId) {
        if !self.enqueued[prop.index()] {
            self.enqueued[prop.index()] = true;
            let band = self.band_of(prop);
            self.queues[band].push_back(prop);
        }
    }

    /// Pop the next propagator to run from the lowest non-empty band, clearing
    /// its enqueued flag.
    pub(crate) fn dequeue(&mut self) -> Option<PropId> {
        for q in &mut self.queues {
            if let Some(p) = q.pop_front() {
                self.enqueued[p.index()] = false;
                return Some(p);
            }
        }
        None
    }

    /// The next propagator that would run, without dequeuing it.
    pub(crate) fn peek_queue(&self) -> Option<PropId> {
        self.queues.iter().find_map(|q| q.front().copied())
    }

    /// Whether every band's queue is empty.
    pub(crate) fn queue_is_empty(&self) -> bool {
        self.queues.iter().all(VecDeque::is_empty)
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
        for q in &mut self.queues {
            while let Some(p) = q.pop_front() {
                self.enqueued[p.index()] = false;
            }
        }
    }

    /// Wake the appropriate subscribers of `var` (except the running one).
    fn notify(&mut self, var: VarId, bounds_moved: bool, fixed: bool) {
        // Split the borrow: read subscription lists while mutating queues/flags.
        // Subscription lists never change during propagation.
        let Store { subs, queues, enqueued, prio, single_band, current, .. } = self;
        let s = &subs[var.index()];
        for &(event, p) in &s.entries {
            if event == Event::DomainChange || (bounds_moved && event == Event::BoundChange) || (fixed && event == Event::Fix) {
                Self::wake(queues, enqueued, prio, *single_band, *current, p);
            }
        }
    }

    /// Enqueue `p` on its band unless it is the running propagator or already queued.
    fn wake(queues: &mut [VecDeque<PropId>; NBANDS], enqueued: &mut [bool], prio: &[Priority], single_band: bool, current: Option<PropId>, p: PropId) {
        if current == Some(p) {
            return;
        }
        if !enqueued[p.index()] {
            enqueued[p.index()] = true;
            let band = if single_band { 0 } else { prio[p.index()].index() };
            queues[band].push_back(p);
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

    /// Weighted degree of `var` from the incremental cache: identical to
    /// [`var_weight`](Store::var_weight) but O(1), for the per-node brancher.
    pub fn var_weight_cached(&self, var: VarId) -> u64 {
        self.var_weight_cache[var.index()].max(1)
    }

    /// Mirror a `dom/wdeg` weight bump on `prop` into the per-variable cache.
    /// `scope` is deduplicated per variable, but `var_weight` sums one term per
    /// subscription entry, so a variable subscribed to `prop` at several
    /// granularities gains one per matching entry.
    pub(crate) fn bump_var_weight(&mut self, prop: PropId) {
        let p = prop.index();
        for i in 0..self.scope[p].len() {
            let v = self.scope[p][i].index();
            let mult = self.subs[v].entries.iter().filter(|(_, pp)| pp.index() == p).count() as u64;
            self.var_weight_cache[v] += mult;
        }
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
    /// When set, [`post`](Solver::post) wraps each propagator in a
    /// [`Selected`] guard on this selector, used to build constraint-level
    /// unsat cores. `None` in normal solving (no wrapping, no overhead).
    current_selector: Option<VarId>,
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
    /// it for an initial propagation. Under [`set_selector`](Solver::set_selector)
    /// the propagator is first wrapped in a [`Selected`] guard.
    pub fn post(&mut self, prop: Box<dyn Propagator>) -> PropId {
        let mut prop: Box<dyn Propagator> = match self.current_selector {
            Some(sel) => Box::new(Selected::new(sel, prop)),
            None => prop,
        };
        let id = self.store.alloc_prop();
        debug_assert_eq!(id.index(), self.propagators.len());
        prop.register(&mut self.store, id);
        self.store.set_priority(id, prop.priority());
        self.store.finalize_scope(id);
        self.propagators.push(Some(prop));
        self.weights.push(1);
        self.store.enqueue(id);
        id
    }

    /// Route subsequently [`post`](Solver::post)ed propagators through a
    /// [`Selected`] guard on `sel` (a `{0,1}` selector): the constraint is active
    /// only when `sel = 1`, and its inferences carry `sel` in their reason. Pass
    /// `None` to stop guarding. The basis for constraint-level unsat cores; see
    /// [`crate::mus`].
    pub fn set_selector(&mut self, sel: Option<VarId>) {
        self.current_selector = sel;
    }

    /// Collapse scheduling into a single FIFO band (the pre-banding order), for
    /// benchmarks comparing band ordering in-process. Off in normal use.
    pub fn set_single_band(&mut self, on: bool) {
        self.store.set_single_band(on);
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
    pub fn propagate_until<F: Fn() -> bool>(&mut self, should_stop: F) -> Result<(), Inconsistency> {
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
        PROP_CALLS.with(|c| c.set(c.get() + 1));
        if self.store.prio[id.index()] == Priority::Expensive {
            EXPENSIVE_CALLS.with(|c| c.set(c.get() + 1));
        }
        let mut prop = self.propagators[id.index()].take().expect("running a propagator that is not present");
        self.store.clear_pending();
        self.store.current = Some(id);
        let result = prop.propagate(&mut self.store);
        self.store.current = None;
        self.propagators[id.index()] = Some(prop);
        if let Err(e) = result {
            self.weights[id.index()] += 1;
            self.store.bump_var_weight(id);
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
