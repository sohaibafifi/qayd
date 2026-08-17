//! `lex` ordering, inverse `channel`, and the `slide` meta-constraint.

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Solver, Store};

// ===========================================================================
// lex
// ===========================================================================

/// `x <=lex y` (or `<lex y` when `strict`). A two-state automaton tracks whether
/// the prefix is still equal or already smaller. Forward/backward reachability
/// gives support filtering for domain holes at every position.
#[derive(Clone)]
struct Lex {
    x: Vec<VarId>,
    y: Vec<VarId>,
    strict: bool,
    forward: Vec<[bool; 2]>,
    backward: Vec<[bool; 2]>,
    x_values: Vec<i32>,
    y_values: Vec<i32>,
}

impl Propagator for Lex {
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
        loop {
            let before = self.x.iter().chain(&self.y).map(|&variable| store.size(variable)).sum::<usize>();
            self.forward.clear();
            self.forward.resize(n + 1, [false; 2]);
            self.backward.clear();
            self.backward.resize(n + 1, [false; 2]);
            self.forward[0][0] = true;

            for i in 0..n {
                let same_variable = self.x[i] == self.y[i];
                let has_equal = if same_variable {
                    store.size(self.x[i]) > 0
                } else {
                    store.values(self.x[i]).any(|value| store.contains(self.y[i], value))
                };
                let has_less = !same_variable && i64::from(store.min(self.x[i])) < i64::from(store.max(self.y[i]));
                self.forward[i + 1][0] = self.forward[i][0] && has_equal;
                self.forward[i + 1][1] = self.forward[i][1] || (self.forward[i][0] && has_less);
            }

            self.backward[n] = [!self.strict, true];
            for i in (0..n).rev() {
                let same_variable = self.x[i] == self.y[i];
                let has_equal = if same_variable {
                    store.size(self.x[i]) > 0
                } else {
                    store.values(self.x[i]).any(|value| store.contains(self.y[i], value))
                };
                let has_less = !same_variable && i64::from(store.min(self.x[i])) < i64::from(store.max(self.y[i]));
                self.backward[i][0] = (has_equal && self.backward[i + 1][0]) || (has_less && self.backward[i + 1][1]);
                self.backward[i][1] = self.backward[i + 1][1];
            }
            if !self.backward[0][0] {
                return Err(Inconsistency);
            }

            for i in 0..n {
                self.x_values.clear();
                self.x_values.extend(store.values(self.x[i]));
                self.y_values.clear();
                self.y_values.extend(store.values(self.y[i]));
                let same_variable = self.x[i] == self.y[i];

                let prefix_less_supports_all = self.forward[i][1] && self.backward[i + 1][1];
                let equal_suffix = self.forward[i][0] && self.backward[i + 1][0];
                let less_suffix = self.forward[i][0] && self.backward[i + 1][1];
                let x_min = store.min(self.x[i]);
                let y_max = store.max(self.y[i]);

                for &x_value in &self.x_values {
                    let supported = prefix_less_supports_all
                        || (equal_suffix && store.contains(self.y[i], x_value))
                        || (!same_variable && less_suffix && x_value < y_max);
                    if !supported {
                        store.remove(self.x[i], x_value)?;
                    }
                }
                for &y_value in &self.y_values {
                    let supported = prefix_less_supports_all
                        || (equal_suffix && store.contains(self.x[i], y_value))
                        || (!same_variable && less_suffix && x_min < y_value);
                    if !supported {
                        store.remove(self.y[i], y_value)?;
                    }
                }
            }
            let after = self.x.iter().chain(&self.y).map(|&variable| store.size(variable)).sum::<usize>();
            if after == before {
                return Ok(());
            }
        }
    }
}

/// Post `x <=lex y`, or `x <lex y` when `strict`.
pub fn lex(solver: &mut Solver, x: &[VarId], y: &[VarId], strict: bool) {
    assert_eq!(x.len(), y.len(), "lex: vectors must have equal length");
    solver.post(Box::new(Lex {
        x: x.to_vec(),
        y: y.to_vec(),
        strict,
        forward: Vec::new(),
        backward: Vec::new(),
        x_values: Vec::new(),
        y_values: Vec::new(),
    }));
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
