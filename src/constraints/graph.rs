//! Graph constraints: `circuit`.

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Premise, Solver, Store};
use crate::trail::ReversibleInt;

/// `circuit(succ)`: `succ[i]` is node `i`'s successor; the relation must form a
/// single Hamiltonian circuit over `0..n`. Weak-but-correct: forward-checking
/// distinctness plus subtour elimination via trailed path endpoints.
#[derive(Clone)]
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
        Self { succ: succ.to_vec(), dest, orig, len, linked }
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

        // Domain in range, no self-loops. Idempotent, so apply once up front.
        for i in 0..n {
            store.remove_below(self.succ[i], 0)?;
            store.remove_above(self.succ[i], (n - 1) as i32)?;
            store.remove(self.succ[i], i as i32)?;
        }

        loop {
            let before_size: usize = self.succ.iter().map(|&v| store.size(v)).sum();
            let before_linked: i32 = self.linked.iter().map(|&r| store.rev_get(r)).sum();

            // Forward-checking distinctness: `succ[i] = val` means no other node may
            // also point to `val`. Reason: that single fixed arc.
            for i in 0..n {
                if store.is_fixed(self.succ[i]) {
                    let val = store.value(self.succ[i]);
                    for k in 0..n {
                        if k != i {
                            store.remove_because(self.succ[k], val, vec![Premise::Eq { var: self.succ[i], val }])?;
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
                        // Closing d -> o now would make a subtour: the fixed path
                        // o -> ... -> d already covers `combined < n` nodes. Reason:
                        // the fixed arcs forming that path.
                        let why = path_arcs(store, &self.succ, o, d);
                        store.remove_because(self.succ[d], o as i32, why)?;
                    }
                }
            }

            let after_size: usize = self.succ.iter().map(|&v| store.size(v)).sum();
            let after_linked: i32 = self.linked.iter().map(|&r| store.rev_get(r)).sum();
            if after_size == before_size && after_linked == before_linked {
                return Ok(());
            }
        }
    }
}

/// The fixed arcs along the path `from -> ... -> to`, as `succ[x] = next`
/// premises. Every arc on a folded chain is fixed, so this terminates within `n`
/// steps; the loop bound is a safety net.
fn path_arcs(store: &Store, succ: &[VarId], from: usize, to: usize) -> Vec<Premise> {
    let mut why = Vec::with_capacity(succ.len());
    let mut x = from;
    for _ in 0..succ.len() {
        let next = store.value(succ[x]);
        why.push(Premise::Eq { var: succ[x], val: next });
        if next as usize == to {
            break;
        }
        x = next as usize;
    }
    why
}

/// Post `circuit(succ)`.
pub fn circuit(solver: &mut Solver, succ: &[VarId]) {
    let prop = Circuit::new(&mut solver.store, succ);
    solver.post(Box::new(prop));
}

#[cfg(test)]
mod tests {
    use super::circuit;
    use crate::search::{solve, SearchControl};
    use crate::store::Solver;
    use std::collections::BTreeSet;

    /// Whether `succ` (as a function `i -> succ[i]`) is one Hamiltonian cycle.
    fn is_single_cycle(succ: &[usize]) -> bool {
        let n = succ.len();
        let mut seen = vec![false; n];
        let mut cur = 0usize;
        for _ in 0..n {
            if cur >= n || seen[cur] {
                return false;
            }
            seen[cur] = true;
            cur = succ[cur];
        }
        cur == 0 && seen.iter().all(|&s| s)
    }

    fn permute(perm: &mut [usize], k: usize, out: &mut impl FnMut(&[usize])) {
        if k == perm.len() {
            out(perm);
            return;
        }
        for i in k..perm.len() {
            perm.swap(k, i);
            permute(perm, k + 1, out);
            perm.swap(k, i);
        }
    }

    /// All `succ` assignments over `0..n` that form a single Hamiltonian cycle.
    fn brute_circuits(n: usize) -> BTreeSet<Vec<i32>> {
        let mut out = BTreeSet::new();
        let mut perm: Vec<usize> = (0..n).collect();
        permute(&mut perm, 0, &mut |p| {
            if is_single_cycle(p) {
                out.insert(p.iter().map(|&x| x as i32).collect());
            }
        });
        out
    }

    #[test]
    fn circuit_enumerates_single_hamiltonian_cycles() {
        for n in 2..=6usize {
            let mut solver = Solver::new();
            let succ: Vec<_> = (0..n).map(|_| solver.store.new_var_range(0, n as i32 - 1)).collect();
            circuit(&mut solver, &succ);
            let mut set = BTreeSet::new();
            solve(&mut solver, &succ, |s: &Solver| {
                set.insert(succ.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
                SearchControl::Continue
            });
            let brute = brute_circuits(n);
            assert_eq!(set, brute, "circuit enumerates exactly the single Hamiltonian cycles (n={n})");
            assert_eq!(set.len(), (1..n).product::<usize>(), "there are (n-1)! directed Hamiltonian cycles (n={n})");
        }
    }
}
