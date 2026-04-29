//! Graph constraints: `circuit`.

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Solver, Store};
use crate::trail::ReversibleInt;

/// `circuit(succ)`: `succ[i]` is node `i`'s successor; the relation must form a
/// single Hamiltonian circuit over `0..n`. Weak-but-correct: forward-checking
/// distinctness plus subtour elimination via trailed path endpoints.
struct Circuit {
    succ: Vec<VarId>,
    /// End node of the path that currently starts at node `i`.
    dest: Vec<ReversibleInt>,
    /// Start node of the path that currently ends at node `i`.
    orig: Vec<ReversibleInt>,
    /// Number of nodes in the path that starts at node `i`.
    len: Vec<ReversibleInt>,
    /// Whether arc `i -> succ[i]` has already been folded into the chains.
    linked: Vec<ReversibleInt>,
}

impl Circuit {
    fn new(store: &mut Store, succ: &[VarId]) -> Self {
        let n = succ.len();
        let mut dest = Vec::with_capacity(n);
        let mut orig = Vec::with_capacity(n);
        let mut len = Vec::with_capacity(n);
        let mut linked = Vec::with_capacity(n);
        for i in 0..n {
            dest.push(store.new_rev_int(i as i32));
            orig.push(store.new_rev_int(i as i32));
            len.push(store.new_rev_int(1));
            linked.push(store.new_rev_int(0));
        }
        Self {
            succ: succ.to_vec(),
            dest,
            orig,
            len,
            linked,
        }
    }
}

impl Propagator for Circuit {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.succ {
            store.subscribe(v, me, Event::Fix);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.succ.len();
        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            return store.fix(self.succ[0], 0);
        }

        // Domain in range, no self-loops.
        for i in 0..n {
            store.remove_below(self.succ[i], 0)?;
            store.remove_above(self.succ[i], (n - 1) as i32)?;
            store.remove(self.succ[i], i as i32)?;
        }

        // Forward-checking distinctness.
        for i in 0..n {
            if store.is_fixed(self.succ[i]) {
                let val = store.value(self.succ[i]);
                for k in 0..n {
                    if k != i {
                        store.remove(self.succ[k], val)?;
                    }
                }
            }
        }

        // Subtour elimination: fold each newly-fixed arc into the chains.
        for i in 0..n {
            if store.is_fixed(self.succ[i]) && store.rev_get(self.linked[i]) == 0 {
                let j = store.value(self.succ[i]) as usize;
                let o = store.rev_get(self.orig[i]) as usize;
                let d = store.rev_get(self.dest[j]) as usize;

                store.rev_set(self.orig[d], o as i32);
                store.rev_set(self.dest[o], d as i32);
                let combined = store.rev_get(self.len[o]) + store.rev_get(self.len[j]);
                store.rev_set(self.len[o], combined);
                store.rev_set(self.linked[i], 1);

                if combined < n as i32 {
                    // Closing d -> o now would make a subtour.
                    store.remove(self.succ[d], o as i32)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `circuit(succ)`.
pub fn circuit(solver: &mut Solver, succ: &[VarId]) {
    let prop = Circuit::new(&mut solver.store, succ);
    solver.post(Box::new(prop));
}
