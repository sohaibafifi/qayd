//! `lex` ordering, inverse `channel`, and the `slide` meta-constraint.

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Solver, Store};

// ===========================================================================
// lex
// ===========================================================================

/// `x <=lex y` (or `<lex y` when `strict`). Sound, leaf-correct: skip the
/// fixed-equal prefix, enforce `x[i] <= y[i]` at the first undecided position.
#[derive(Clone)]
struct Lex {
    x: Vec<VarId>,
    y: Vec<VarId>,
    strict: bool,
}

impl Propagator for Lex {
    fn priority(&self) -> Priority {
        Priority::Linear
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for (&a, &b) in self.x.iter().zip(&self.y) {
            store.subscribe(a, me, Event::BoundChange);
            store.subscribe(b, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.x.len();
        let fixed_equal = |store: &Store, i: usize| {
            store.is_fixed(self.x[i]) && store.is_fixed(self.y[i]) && store.value(self.x[i]) == store.value(self.y[i])
        };
        // Local fixpoint: forcing a pair equal can extend the settled prefix.
        let mut i = 0;
        loop {
            while i < n && fixed_equal(store, i) {
                i += 1;
            }
            if i == n {
                // Fully equal: only `<=` accepts it.
                return if self.strict { Err(Inconsistency) } else { Ok(()) };
            }
            // First undecided position: x[i] <= y[i] is necessary.
            store.remove_above(self.x[i], store.max(self.y[i]))?;
            store.remove_below(self.y[i], store.min(self.x[i]))?;

            if fixed_equal(store, i) {
                i += 1; // forced equal; prefix grew
            } else {
                return Ok(());
            }
        }
    }
}

/// Post `x <=lex y`, or `x <lex y` when `strict`.
pub fn lex(solver: &mut Solver, x: &[VarId], y: &[VarId], strict: bool) {
    assert_eq!(x.len(), y.len(), "lex: vectors must have equal length");
    solver.post(Box::new(Lex { x: x.to_vec(), y: y.to_vec(), strict }));
}

/// Post `rows[0] <=lex rows[1] <=lex ...` (strict between each consecutive pair
/// when `strict`).
pub fn lex_chain(solver: &mut Solver, rows: &[Vec<VarId>], strict: bool) {
    for w in rows.windows(2) {
        lex(solver, &w[0], &w[1], strict);
    }
}

// ===========================================================================
// channel (inverse)
// ===========================================================================

/// `x` and `y` are inverse permutations of `0..n`: `x[i] = j  <=>  y[j] = i`.
#[derive(Clone)]
struct Channel {
    x: Vec<VarId>,
    y: Vec<VarId>,
    buf: Vec<i32>,
}

impl Propagator for Channel {
    fn priority(&self) -> Priority {
        Priority::Linear
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for (&a, &b) in self.x.iter().zip(&self.y) {
            store.subscribe(a, me, Event::DomainChange);
            store.subscribe(b, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.x.len();
        let hi = (n - 1) as i32;
        for &v in self.x.iter().chain(self.y.iter()) {
            store.remove_below(v, 0)?;
            store.remove_above(v, hi)?;
        }

        // x[i] = j requires i in dom(y[j]), and symmetrically.
        for i in 0..n {
            self.buf.clear();
            self.buf.extend(store.values(self.x[i]));
            for &j in &self.buf {
                if !store.contains(self.y[j as usize], i as i32) {
                    store.remove(self.x[i], j)?;
                }
            }
        }
        for j in 0..n {
            self.buf.clear();
            self.buf.extend(store.values(self.y[j]));
            for &i in &self.buf {
                if !store.contains(self.x[i as usize], j as i32) {
                    store.remove(self.y[j], i)?;
                }
            }
        }
        Ok(())
    }
}

/// Post the inverse channel `x[i] = j  <=>  y[j] = i`. Both arrays have length
/// `n` and values are taken from `0..n`.
pub fn channel(solver: &mut Solver, x: &[VarId], y: &[VarId]) {
    assert_eq!(x.len(), y.len(), "channel: arrays must have equal length");
    solver.post(Box::new(Channel { x: x.to_vec(), y: y.to_vec(), buf: Vec::new() }));
}

// ===========================================================================
// slide (meta-constraint)
// ===========================================================================

/// Post a sub-constraint over every sliding window of `width` consecutive
/// variables. `post_window` is called with each window slice and posts whatever
/// constraint should hold there.
pub fn slide(solver: &mut Solver, vars: &[VarId], width: usize, mut post_window: impl FnMut(&mut Solver, &[VarId])) {
    assert!(width > 0, "slide: window width must be positive");
    if width > vars.len() {
        return;
    }
    for window in vars.windows(width) {
        post_window(solver, window);
    }
}
