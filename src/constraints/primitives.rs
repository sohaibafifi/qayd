//! Primitive constraints.
//!
//! Phase 0 ships the binary disequality `x != y + c` with forward checking,
//! which is enough to model N-Queens (column, and the two diagonal offsets).
//! The remaining primitives — `instantiation`, `allEqual`, `ordered`,
//! `precedence`, `minimum`, `maximum`, `element`, `count` — land in Phase 1
//! alongside `intension` and `sum`.

use crate::constraints::intension::intension;
use crate::constraints::linear::Relation;
use crate::expr::{eq, imp, int, or, var};
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Solver, Store};

/// The constraint `x != y + c`.
///
/// Forward checking: when one side is fixed, the single forbidden value is
/// pruned from the other. For binary disequality this is value-consistent, and
/// it is trivially idempotent — once the forbidden value is gone, re-running
/// removes nothing.
pub struct NotEqualOffset {
    x: VarId,
    y: VarId,
    c: i32,
}

impl Propagator for NotEqualOffset {
    fn register(&mut self, store: &mut Store, me: PropId) {
        // Forward checking only fires once a side is fixed.
        store.subscribe(self.x, me, Event::Fix);
        store.subscribe(self.y, me, Event::Fix);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        // x = v  =>  y != v - c
        if store.is_fixed(self.x) {
            let forbidden = store.value(self.x) - self.c;
            store.remove(self.y, forbidden)?;
        }
        // y = w  =>  x != w + c
        if store.is_fixed(self.y) {
            let forbidden = store.value(self.y) + self.c;
            store.remove(self.x, forbidden)?;
        }
        Ok(())
    }
}

/// Post `x != y + c`.
pub fn not_equal_offset(solver: &mut Solver, x: VarId, y: VarId, c: i32) -> PropId {
    solver.post(Box::new(NotEqualOffset { x, y, c }))
}

/// Post `x != y`.
pub fn not_equal(solver: &mut Solver, x: VarId, y: VarId) -> PropId {
    not_equal_offset(solver, x, y, 0)
}

// ---------------------------------------------------------------------------
// Ordering: x + k <= y
// ---------------------------------------------------------------------------

/// `x + k ≤ y`. Bounds propagation; one pass is a fixpoint for two variables.
struct LessOrEqual {
    x: VarId,
    y: VarId,
    k: i32,
}

impl Propagator for LessOrEqual {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.x, me, Event::BoundChange);
        store.subscribe(self.y, me, Event::BoundChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        // x ≤ max(y) − k
        store.remove_above(self.x, store.max(self.y).saturating_sub(self.k))?;
        // y ≥ min(x) + k
        store.remove_below(self.y, store.min(self.x).saturating_add(self.k))?;
        Ok(())
    }
}

/// Post `x ≤ y`.
pub fn less_or_equal(solver: &mut Solver, x: VarId, y: VarId) -> PropId {
    solver.post(Box::new(LessOrEqual { x, y, k: 0 }))
}

/// Post `x < y`.
pub fn less_than(solver: &mut Solver, x: VarId, y: VarId) -> PropId {
    solver.post(Box::new(LessOrEqual { x, y, k: 1 }))
}

/// Post an ordering along the chain `vars[0] rel vars[1] rel ...`. `rel` must be
/// a comparison (`Le`, `Lt`, `Ge`, `Gt`).
pub fn ordered(solver: &mut Solver, vars: &[VarId], rel: Relation) {
    for w in vars.windows(2) {
        let (a, b) = (w[0], w[1]);
        match rel {
            Relation::Le => {
                less_or_equal(solver, a, b);
            }
            Relation::Lt => {
                less_than(solver, a, b);
            }
            Relation::Ge => {
                less_or_equal(solver, b, a);
            }
            Relation::Gt => {
                less_than(solver, b, a);
            }
            _ => panic!("ordered: relation must be a comparison, got {rel:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// instantiation
// ---------------------------------------------------------------------------

/// Fixes each variable to a prescribed value.
struct Instantiation {
    vars: Vec<VarId>,
    vals: Vec<i32>,
}

impl Propagator for Instantiation {
    fn register(&mut self, _store: &mut Store, _me: PropId) {
        // Nothing to subscribe: it fixes everything on its single run.
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        for (&v, &val) in self.vars.iter().zip(&self.vals) {
            store.fix(v, val)?;
        }
        Ok(())
    }
}

/// Post `vars[i] = vals[i]` for all `i`.
pub fn instantiation(solver: &mut Solver, vars: &[VarId], vals: &[i32]) {
    assert_eq!(vars.len(), vals.len(), "instantiation: length mismatch");
    solver.post(Box::new(Instantiation {
        vars: vars.to_vec(),
        vals: vals.to_vec(),
    }));
}

// ---------------------------------------------------------------------------
// allEqual (domain-consistent via intersection)
// ---------------------------------------------------------------------------

/// All variables take the same value. Filters every variable to the common
/// domain (the intersection), which is domain-consistent for equality.
struct AllEqual {
    vars: Vec<VarId>,
    /// Reused scratch: the running intersection.
    common: Vec<i32>,
    /// Reused scratch: a snapshot of one variable's values.
    buf: Vec<i32>,
}

impl Propagator for AllEqual {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if self.vars.len() < 2 {
            return Ok(());
        }
        let AllEqual { vars, common, buf } = self;

        // common = values present in every variable.
        common.clear();
        common.extend(store.values(vars[0]));
        common.retain(|&v| vars[1..].iter().all(|&u| store.contains(u, v)));

        // Prune each variable down to `common` (empty intersection => wipeout).
        for &v in vars.iter() {
            buf.clear();
            buf.extend(store.values(v));
            for &val in buf.iter() {
                if !common.contains(&val) {
                    store.remove(v, val)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `vars[0] = vars[1] = ...`.
pub fn all_equal(solver: &mut Solver, vars: &[VarId]) {
    solver.post(Box::new(AllEqual {
        vars: vars.to_vec(),
        common: Vec::new(),
        buf: Vec::new(),
    }));
}

// ---------------------------------------------------------------------------
// minimum / maximum
// ---------------------------------------------------------------------------

/// `y = min(xs)`. Sound bounds reasoning, exact at a full assignment.
struct Minimum {
    y: VarId,
    xs: Vec<VarId>,
}

impl Propagator for Minimum {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.y, me, Event::BoundChange);
        for &x in &self.xs {
            store.subscribe(x, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        // min(xs) lies in [min of the lower bounds, min of the upper bounds].
        let mut lb = i32::MAX;
        let mut ub = i32::MAX;
        for &x in &self.xs {
            lb = lb.min(store.min(x));
            ub = ub.min(store.max(x));
        }
        store.remove_below(self.y, lb)?;
        store.remove_above(self.y, ub)?;
        // Every x ≥ y.
        let ymin = store.min(self.y);
        for &x in &self.xs {
            store.remove_below(x, ymin)?;
        }
        Ok(())
    }
}

/// `y = max(xs)`.
struct Maximum {
    y: VarId,
    xs: Vec<VarId>,
}

impl Propagator for Maximum {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.y, me, Event::BoundChange);
        for &x in &self.xs {
            store.subscribe(x, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let mut lb = i32::MIN;
        let mut ub = i32::MIN;
        for &x in &self.xs {
            lb = lb.max(store.min(x));
            ub = ub.max(store.max(x));
        }
        store.remove_below(self.y, lb)?;
        store.remove_above(self.y, ub)?;
        // Every x ≤ y.
        let ymax = store.max(self.y);
        for &x in &self.xs {
            store.remove_above(x, ymax)?;
        }
        Ok(())
    }
}

/// Post `y = min(xs)`. `xs` must be non-empty.
pub fn minimum(solver: &mut Solver, y: VarId, xs: &[VarId]) {
    assert!(!xs.is_empty(), "minimum: empty list");
    solver.post(Box::new(Minimum { y, xs: xs.to_vec() }));
}

/// Post `y = max(xs)`. `xs` must be non-empty.
pub fn maximum(solver: &mut Solver, y: VarId, xs: &[VarId]) {
    assert!(!xs.is_empty(), "maximum: empty list");
    solver.post(Box::new(Maximum { y, xs: xs.to_vec() }));
}

// ---------------------------------------------------------------------------
// element: value = array[idx]   (idx is 0-based)
// ---------------------------------------------------------------------------

/// `value = array[idx]`, 0-based. Domain-consistent filtering: prunes unusable
/// indices, unsupported values, and (when the index is fixed) equates `value`
/// with the selected entry.
struct Element {
    array: Vec<VarId>,
    idx: VarId,
    value: VarId,
    /// Reused scratch for domain snapshots.
    buf: Vec<i32>,
}

impl Element {
    /// Force `a` and `b` to share the same domain.
    fn equate(&mut self, store: &mut Store, a: VarId, b: VarId) -> Result<(), Inconsistency> {
        self.buf.clear();
        self.buf.extend(store.values(a));
        for &v in &self.buf {
            if !store.contains(b, v) {
                store.remove(a, v)?;
            }
        }
        self.buf.clear();
        self.buf.extend(store.values(b));
        for &v in &self.buf {
            if !store.contains(a, v) {
                store.remove(b, v)?;
            }
        }
        Ok(())
    }
}

impl Propagator for Element {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.idx, me, Event::DomainChange);
        store.subscribe(self.value, me, Event::DomainChange);
        for &a in &self.array {
            store.subscribe(a, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.array.len() as i32;
        store.remove_below(self.idx, 0)?;
        store.remove_above(self.idx, n - 1)?;

        // Prune the index: i is impossible if array[i] and value share no value.
        self.buf.clear();
        self.buf.extend(store.values(self.idx));
        for &i in &self.buf {
            let ai = self.array[i as usize];
            let shares = store.values(ai).any(|v| store.contains(self.value, v));
            if !shares {
                store.remove(self.idx, i)?;
            }
        }

        if store.is_fixed(self.idx) {
            // value = array[i]
            let i = store.value(self.idx) as usize;
            let ai = self.array[i];
            self.equate(store, self.value, ai)?;
        } else {
            // Prune value: v is impossible if no live index selects an entry
            // whose domain contains v.
            self.buf.clear();
            self.buf.extend(store.values(self.value));
            for &v in &self.buf {
                let supported = store
                    .values(self.idx)
                    .any(|i| store.contains(self.array[i as usize], v));
                if !supported {
                    store.remove(self.value, v)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `value = array[idx]` (0-based index). `array` must be non-empty.
pub fn element(solver: &mut Solver, array: &[VarId], idx: VarId, value: VarId) {
    assert!(!array.is_empty(), "element: empty array");
    solver.post(Box::new(Element {
        array: array.to_vec(),
        idx,
        value,
        buf: Vec::new(),
    }));
}

// ---------------------------------------------------------------------------
// allDifferent (forward checking)
// ---------------------------------------------------------------------------

/// All variables take distinct values. Forward-checking: when a variable is
/// fixed, its value is pruned from the others. (Régin's matching/SCC algorithm
/// for full domain consistency is a later upgrade.)
struct AllDifferent {
    vars: Vec<VarId>,
}

impl Propagator for AllDifferent {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::Fix);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        for i in 0..self.vars.len() {
            if store.is_fixed(self.vars[i]) {
                let val = store.value(self.vars[i]);
                for (j, &other) in self.vars.iter().enumerate() {
                    if j != i {
                        store.remove(other, val)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Post `allDifferent(vars)`.
pub fn all_different(solver: &mut Solver, vars: &[VarId]) {
    solver.post(Box::new(AllDifferent {
        vars: vars.to_vec(),
    }));
}

// ---------------------------------------------------------------------------
// precedence (value precedence)
// ---------------------------------------------------------------------------

/// Value precedence over `list` for the sequence `values`: for each consecutive
/// pair `(s, t)`, the first occurrence of `s` must precede the first occurrence
/// of `t` (and `t` may not appear unless `s` already has).
///
/// Decomposed through `intension`: for every position `i`, `list[i] = t` implies
/// some earlier position holds `s`. At `i = 0` the disjunction is empty, so the
/// later value cannot occupy the first slot.
pub fn precedence(solver: &mut Solver, list: &[VarId], values: &[i32]) {
    for w in values.windows(2) {
        let (s, t) = (w[0], w[1]);
        for i in 0..list.len() {
            let earlier_s: Vec<_> = (0..i).map(|p| eq(var(list[p]), int(s as i64))).collect();
            intension(solver, imp(eq(var(list[i]), int(t as i64)), or(earlier_s)));
        }
    }
}
