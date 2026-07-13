//! Graph constraints: `circuit`.

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Premise, Solver, Store};
use crate::trail::ReversibleInt;

/// `circuit(succ)`: `succ[i]` is node `i`'s successor; the relation must form a
/// single Hamiltonian circuit over `0..n`. Weak-but-correct: forward-checking
/// distinctness plus subtour elimination via trailed path endpoints.
///
/// With `cutset` enabled it additionally enforces **cut-set / subtour-elimination
/// constraints** lazily: a Hamiltonian circuit's residual arc graph must be
/// strongly connected, so any proper node subset `S` that has become closed
/// under the still-possible arcs (`Δout(S) = ∅`) is infeasible. This mirrors the
/// cut-set forcing of Ohashi et al., *SAT-Based CEGAR for HCP by Cut-Set
/// Constraints* (SAT 2025), transferred from their SAT/CEGAR loop to a CP
/// propagator: instead of only forbidding the single arc that would close the
/// current chain, it fails as soon as any subset can no longer reach the rest of
/// the graph, with the crossing arcs as the (tight) reason.
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
    /// Enforce cut-set (strong-connectivity) filtering on top of the base rules.
    cutset: bool,
    /// Original neighbours of each node (its successor domain at post time), used
    /// to build tight cut-set explanations without citing structural non-edges.
    nbr: Vec<Vec<i32>>,
    /// Reverse of `nbr`: `rnbr[u]` lists nodes that may have `u` as a successor,
    /// so reverse reachability stays `O(n + m)` instead of scanning all nodes.
    rnbr: Vec<Vec<i32>>,
}

impl Circuit {
    fn new(store: &mut Store, succ: &[VarId], cutset: bool) -> Self {
        let n = succ.len();
        let mut dest = Vec::with_capacity(n);
        let mut orig = Vec::with_capacity(n);
        let mut len = Vec::with_capacity(n);
        let mut linked = Vec::with_capacity(n);
        let mut nbr = Vec::with_capacity(n);
        for (i, &var) in succ.iter().enumerate() {
            dest.push(store.new_rev_int(i as i32));
            orig.push(store.new_rev_int(i as i32));
            len.push(store.new_rev_int(1));
            linked.push(store.new_rev_int(0));
            nbr.push(store.values(var).collect::<Vec<i32>>());
        }
        let mut rnbr = vec![Vec::new(); n];
        for (i, ns) in nbr.iter().enumerate() {
            for &j in ns {
                rnbr[j as usize].push(i as i32);
            }
        }
        Self { succ: succ.to_vec(), dest, orig, len, linked, cutset, nbr, rnbr }
    }

    /// Cut-set filtering: if the residual arc graph (arc `i -> j` iff
    /// `j ∈ dom(succ[i])`) is not strongly connected, some proper subset `S` is
    /// closed under outgoing arcs (`Δout(S) = ∅`) and no Hamiltonian circuit can
    /// exist. Fail, citing the removed crossing arcs `succ[i] ≠ j` for `i ∈ S`,
    /// `j` an original neighbour outside `S` — a subtour-elimination cut.
    ///
    /// A Hamiltonian circuit is strongly connected and uses only residual arcs,
    /// so "residual graph not strongly connected" soundly implies infeasibility.
    /// Strong connectivity here reduces to: node 0 reaches every node, and every
    /// node reaches node 0 (any `u -> v` path then routes through 0). Each failing
    /// direction yields an out-closed proper subset for the explanation.
    fn cut_set_check(&self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.succ.len();
        if n < 3 {
            return Ok(());
        }

        // Forward reachability from node 0. The reached set is closed under
        // outgoing residual arcs by construction.
        let forward = self.reach(store, /*reverse=*/ false);
        if forward.iter().filter(|&&r| r).count() != n {
            return Err(self.cut_fail(store, &forward, true));
        }
        // Reverse reachability to node 0. Nodes that cannot reach 0 form a set
        // whose outgoing arcs all stay inside it (else they would reach 0).
        let backward = self.reach(store, /*reverse=*/ true);
        if backward.iter().filter(|&&r| r).count() != n {
            let closed: Vec<bool> = backward.iter().map(|&r| !r).collect();
            return Err(self.cut_fail(store, &closed, false));
        }
        Ok(())
    }

    /// BFS over the residual arc graph from node 0. `reverse` walks arcs backwards
    /// (which nodes can reach 0). Returns the reached-node bitset.
    fn reach(&self, store: &Store, reverse: bool) -> Vec<bool> {
        let n = self.succ.len();
        let mut seen = vec![false; n];
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(u) = stack.pop() {
            if reverse {
                // predecessors of u: original neighbours i that still keep u as a
                // possible successor (u ∈ dom(succ[i])).
                for &i in &self.rnbr[u] {
                    let i = i as usize;
                    if !seen[i] && store.contains(self.succ[i], u as i32) {
                        seen[i] = true;
                        stack.push(i);
                    }
                }
            } else {
                for v in store.values(self.succ[u]) {
                    let v = v as usize;
                    if !seen[v] {
                        seen[v] = true;
                        stack.push(v);
                    }
                }
            }
        }
        seen
    }

    /// Build the failure for a closed proper subset `closed` (`Δout = ∅`): cite
    /// every original neighbour arc that leaves the set and has been removed.
    fn cut_fail(&self, store: &mut Store, closed: &[bool], _forward: bool) -> Inconsistency {
        let mut why = Vec::new();
        for i in 0..self.succ.len() {
            if !closed[i] {
                continue;
            }
            for &j in &self.nbr[i] {
                if !closed[j as usize] {
                    // Arc i -> j leaves the closed set; it must be absent.
                    why.push(Premise::Ne { var: self.succ[i], val: j });
                }
            }
        }
        store.fail_because(why)
    }
}

impl Propagator for Circuit {
    fn priority(&self) -> Priority {
        Priority::Expensive
    }

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
                break;
            }
        }

        // Cut-set / strong-connectivity filtering on the fixpoint of the base
        // rules. Fail-only, so it needs no extra iteration.
        if self.cutset {
            self.cut_set_check(store)?;
        }
        Ok(())
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

/// Post `circuit(succ)`, automatically enabling cut-set (strong-connectivity)
/// filtering on **sparse** graphs and leaving it off on (near-)complete ones.
///
/// The cut-set check is an `O(n + m)` connectivity pass; it pays off when arcs
/// are restricted so subsets become closed (HCP, road networks, time-window /
/// precedence-pruned routing), but on a complete graph the residual graph stays
/// strongly connected until late and it would only add overhead. We gate on the
/// posted successor-domain density (average out-degree ≤ n/4).
pub fn circuit(solver: &mut Solver, succ: &[VarId]) {
    let n = succ.len();
    let arcs: usize = succ.iter().map(|&v| solver.store.size(v)).sum();
    let sparse = n >= 8 && arcs.saturating_mul(4) <= n.saturating_mul(n);
    circuit_with(solver, succ, sparse);
}

/// Post `circuit(succ)`, optionally enabling cut-set (strong-connectivity)
/// filtering on top of the base rules. See [`Circuit`].
pub fn circuit_with(solver: &mut Solver, succ: &[VarId], cutset: bool) {
    let prop = Circuit::new(&mut solver.store, succ, cutset);
    solver.post(Box::new(prop));
}