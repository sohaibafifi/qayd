//! Table-family constraints: `extension` (Compact-Table-style), `regular`
//! (layered DFA), `mdd` (layered DAG). A value is kept only if it participates
//! in some consistent tuple/path. All buffers recomputed each call.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Solver, Store};

// ===========================================================================
// extension (Compact-Table style)
// ===========================================================================

fn ts_and(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d &= *s;
    }
}
fn ts_or(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d |= *s;
    }
}
fn ts_is_zero(bs: &[u64]) -> bool {
    bs.iter().all(|&w| w == 0)
}
fn ts_intersects(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).any(|(x, y)| x & y != 0)
}
fn ts_has_match(current: &[u64], star: &[u64], exact: Option<&Vec<u64>>, residue: &mut usize) -> bool {
    if current.is_empty() {
        return false;
    }
    for offset in 0..current.len() {
        let word = (*residue + offset) % current.len();
        let exact = exact.map_or(0, |bits| bits[word]);
        if current[word] & (star[word] | exact) != 0 {
            *residue = word;
            return true;
        }
    }
    false
}
fn ts_and_match(dst: &mut [u64], star: &[u64], exact: Option<&Vec<u64>>) {
    match exact {
        Some(exact) => {
            for ((d, s), e) in dst.iter_mut().zip(star).zip(exact) {
                *d &= s | e;
            }
        }
        None => ts_and(dst, star),
    }
}

struct NegativeSearch<'a> {
    store: &'a Store,
    vars: &'a [VarId],
    supports: &'a [HashMap<i32, Vec<u64>>],
    star: &'a [Vec<u64>],
    cover: &'a [Vec<u64>],
    fixed_col: usize,
}

fn negative_has_support(ctx: &NegativeSearch<'_>, search: &mut [Vec<u64>], col: usize) -> bool {
    if ts_is_zero(&search[col]) {
        return true;
    }
    if ts_intersects(&search[col], &ctx.cover[col]) {
        return false;
    }

    let next = col + 1;
    if col == ctx.fixed_col {
        let (prefix, suffix) = search.split_at_mut(next);
        suffix[0].copy_from_slice(&prefix[col]);
        return negative_has_support(ctx, search, next);
    }

    for val in ctx.store.values(ctx.vars[col]) {
        {
            let (prefix, suffix) = search.split_at_mut(next);
            suffix[0].copy_from_slice(&prefix[col]);
            ts_and_match(&mut suffix[0], &ctx.star[col], ctx.supports[col].get(&val));
        }
        if negative_has_support(ctx, search, next) {
            return true;
        }
    }
    false
}

/// Table wildcard `*`: matches any value in its column.
pub const STAR: i32 = i32::MIN;

/// Immutable bitsets compiled from an extension table's tuple body.
#[derive(Clone)]
pub struct ExtensionTemplate {
    arity: usize,
    /// `supports[c][value]` = bitset of tuples whose column `c` equals `value`.
    supports: Vec<HashMap<i32, Vec<u64>>>,
    /// `star[c]` = bitset of tuples with a wildcard in column `c`.
    star: Vec<Vec<u64>>,
    /// Bitset with exactly the `ntuples` low bits set.
    full: Vec<u64>,
}

/// `extension`: assignment must (positive) or must not (negative) match a listed
/// tuple. A `STAR` entry matches any value.
#[derive(Clone)]
struct Extension {
    vars: Vec<VarId>,
    positive: bool,
    template: Arc<ExtensionTemplate>,
    current: Vec<u64>,
    union: Vec<u64>,
    /// Last supporting word per `(column, value)` for positive tables.
    residues: Vec<HashMap<i32, usize>>,
    /// Scratch stack for negative-table completion search.
    search: Vec<Vec<u64>>,
    /// Scratch: patterns wildcarding every remaining unfixed column.
    cover: Vec<Vec<u64>>,
    buf: Vec<i32>,
}

impl Extension {
    fn new(store: &Store, vars: &[VarId], template: Arc<ExtensionTemplate>, positive: bool) -> Self {
        assert_eq!(template.arity, vars.len(), "extension: arity mismatch");
        let nwords = template.full.len();
        let residues =
            if positive { vars.iter().map(|&var| store.values(var).map(|value| (value, 0)).collect()).collect() } else { Vec::new() };
        Self {
            vars: vars.to_vec(),
            positive,
            current: template.full.clone(),
            union: vec![0u64; nwords],
            residues,
            search: vec![vec![0u64; nwords]; vars.len() + 1],
            cover: vec![vec![0u64; nwords]; vars.len() + 1],
            template,
            buf: Vec::new(),
        }
    }
}

impl ExtensionTemplate {
    fn new(arity: usize, tuples: &[Vec<i32>]) -> Self {
        assert!(tuples.iter().all(|t| t.len() == arity), "extension: tuple arity mismatch");
        let ntuples = tuples.len();
        let nwords = ntuples.div_ceil(64);
        let mut supports: Vec<HashMap<i32, Vec<u64>>> = vec![HashMap::new(); arity];
        let mut star: Vec<Vec<u64>> = vec![vec![0u64; nwords]; arity];
        for (t, tuple) in tuples.iter().enumerate() {
            for (c, &val) in tuple.iter().enumerate() {
                if val == STAR {
                    star[c][t / 64] |= 1u64 << (t % 64);
                } else {
                    let bs = supports[c].entry(val).or_insert_with(|| vec![0u64; nwords]);
                    bs[t / 64] |= 1u64 << (t % 64);
                }
            }
        }
        let mut full = vec![0u64; nwords];
        for t in 0..ntuples {
            full[t / 64] |= 1u64 << (t % 64);
        }
        Self { arity, supports, star, full }
    }
}

impl Propagator for Extension {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let Extension { vars, positive, template, current, union, residues, search, cover, buf } = self;
        let ExtensionTemplate { supports, star, full, .. } = &**template;

        if *positive {
            // current = AND over columns of (wildcards OR (OR over present values)).
            current.copy_from_slice(full);
            for (c, &v) in vars.iter().enumerate() {
                union.copy_from_slice(&star[c]);
                for val in store.values(v) {
                    if let Some(bs) = supports[c].get(&val) {
                        ts_or(union, bs);
                    }
                }
                ts_and(current, union);
            }

            if ts_is_zero(current) {
                return Err(Inconsistency);
            }

            // Keep a residue word for each value and scan only when it went stale.
            for (c, &v) in vars.iter().enumerate() {
                buf.clear();
                buf.extend(store.values(v));
                for &val in buf.iter() {
                    let residue = residues[c].get_mut(&val).expect("extension residue missing root-domain value");
                    if !ts_has_match(current, &star[c], supports[c].get(&val), residue) {
                        store.remove(v, val)?;
                    }
                }
            }
        } else {
            // Remove a value only when no completion avoids every forbidden pattern.
            for (c, &v) in vars.iter().enumerate() {
                cover[vars.len()].copy_from_slice(full);
                for d in (0..vars.len()).rev() {
                    let (prefix, suffix) = cover.split_at_mut(d + 1);
                    prefix[d].copy_from_slice(&suffix[0]);
                    if d != c {
                        ts_and(&mut prefix[d], &star[d]);
                    }
                }

                buf.clear();
                buf.extend(store.values(v));
                for &val in buf.iter() {
                    search[0].copy_from_slice(full);
                    ts_and_match(&mut search[0], &star[c], supports[c].get(&val));
                    let ctx = NegativeSearch { store, vars, supports, star, cover, fixed_col: c };
                    if !negative_has_support(&ctx, search, 0) {
                        store.remove(v, val)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Post a table constraint. `positive`: tuple must be one of `tuples`; else none.
/// Use [`STAR`] for `*` wildcards.
pub fn extension(solver: &mut Solver, vars: &[VarId], tuples: &[Vec<i32>], positive: bool) {
    extension_from_template(solver, vars, extension_template(vars.len(), tuples), positive);
}

/// Compile an extension table's immutable tuple bitsets.
pub fn extension_template(arity: usize, tuples: &[Vec<i32>]) -> Arc<ExtensionTemplate> {
    Arc::new(ExtensionTemplate::new(arity, tuples))
}

/// Post an extension constraint backed by a shared immutable template.
pub fn extension_from_template(solver: &mut Solver, vars: &[VarId], template: Arc<ExtensionTemplate>, positive: bool) {
    let prop = Extension::new(&solver.store, vars, template, positive);
    solver.post(Box::new(prop));
}

// ===========================================================================
// regular (layered DFA)
// ===========================================================================

/// A deterministic finite automaton for `regular`.
#[derive(Clone)]
pub struct Dfa {
    /// Number of states (ids `0..n_states`).
    pub n_states: usize,
    /// Start state.
    pub start: usize,
    /// Accepting states.
    pub accept: Vec<usize>,
    /// Transitions `(src, value, dst)`.
    pub transitions: Vec<(usize, i32, usize)>,
}

#[derive(Clone)]
struct Regular {
    vars: Vec<VarId>,
    n_states: usize,
    start: usize,
    accept: Vec<usize>,
    delta: Vec<HashMap<i32, usize>>,
    fwd: Vec<bool>,
    bwd: Vec<bool>,
    buf: Vec<i32>,
}

impl Regular {
    fn new(vars: &[VarId], dfa: Dfa) -> Self {
        let mut delta = vec![HashMap::new(); dfa.n_states];
        for (s, val, d) in dfa.transitions {
            delta[s].insert(val, d);
        }
        let layers = vars.len() + 1;
        Self {
            vars: vars.to_vec(),
            n_states: dfa.n_states,
            start: dfa.start,
            accept: dfa.accept,
            delta,
            fwd: vec![false; layers * dfa.n_states],
            bwd: vec![false; layers * dfa.n_states],
            buf: Vec::new(),
        }
    }
}

impl Propagator for Regular {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.vars.len();
        let s = self.n_states;
        if n == 0 {
            return if self.accept.contains(&self.start) { Ok(()) } else { Err(Inconsistency) };
        }

        self.fwd.fill(false);
        self.bwd.fill(false);
        self.fwd[self.start] = true; // layer 0
        for i in 0..n {
            for q in 0..s {
                if self.fwd[i * s + q] {
                    for v in store.values(self.vars[i]) {
                        if let Some(&q2) = self.delta[q].get(&v) {
                            self.fwd[(i + 1) * s + q2] = true;
                        }
                    }
                }
            }
        }
        for &a in &self.accept {
            self.bwd[n * s + a] = true;
        }
        for i in (0..n).rev() {
            for q in 0..s {
                for v in store.values(self.vars[i]) {
                    if let Some(&q2) = self.delta[q].get(&v) {
                        if self.bwd[(i + 1) * s + q2] {
                            self.bwd[i * s + q] = true;
                            break;
                        }
                    }
                }
            }
        }

        // Keep values backed by a forward-and-backward arc.
        for i in 0..n {
            self.buf.clear();
            self.buf.extend(store.values(self.vars[i]));
            for &val in &self.buf {
                let supported =
                    (0..s).any(|q| self.fwd[i * s + q] && self.delta[q].get(&val).is_some_and(|&q2| self.bwd[(i + 1) * s + q2]));
                if !supported {
                    store.remove(self.vars[i], val)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `regular`: the sequence `vars` must be accepted by `dfa`.
pub fn regular(solver: &mut Solver, vars: &[VarId], dfa: Dfa) {
    assert!(dfa.start < dfa.n_states, "regular: start state out of range");
    solver.post(Box::new(Regular::new(vars, dfa)));
}

// ===========================================================================
// mdd (layered DAG)
// ===========================================================================

/// An arc in a layered MDD, from a node in one layer to a node in the next.
#[derive(Clone)]
pub struct MddArc {
    /// Source node index within its layer.
    pub from: usize,
    /// Arc label (the variable's value).
    pub value: i32,
    /// Destination node index within the next layer.
    pub to: usize,
}

/// A layered MDD: `layers[i]` = arcs from layer `i` to `i+1`; `nodes_per_layer`
/// has `n+1` entries. Layer 0 is the single root; every final-layer node accepts.
#[derive(Clone)]
pub struct Mdd {
    /// Arc lists, one per variable layer.
    pub layers: Vec<Vec<MddArc>>,
    /// Node counts for layers `0..=n`.
    pub nodes_per_layer: Vec<usize>,
}

#[derive(Clone)]
struct MddProp {
    vars: Vec<VarId>,
    layers: Vec<Vec<MddArc>>,
    offsets: Vec<usize>,
    fwd: Vec<bool>,
    bwd: Vec<bool>,
    buf: Vec<i32>,
}

impl MddProp {
    fn new(vars: &[VarId], mdd: Mdd) -> Self {
        let mut offsets = vec![0usize; mdd.nodes_per_layer.len()];
        let mut acc = 0;
        for (i, &c) in mdd.nodes_per_layer.iter().enumerate() {
            offsets[i] = acc;
            acc += c;
        }
        Self { vars: vars.to_vec(), layers: mdd.layers, offsets, fwd: vec![false; acc], bwd: vec![false; acc], buf: Vec::new() }
    }
}

impl Propagator for MddProp {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.vars.len();
        self.fwd.fill(false);
        self.bwd.fill(false);
        self.fwd[self.offsets[0]] = true; // root node 0 in layer 0

        for i in 0..n {
            for arc in &self.layers[i] {
                if self.fwd[self.offsets[i] + arc.from] && store.contains(self.vars[i], arc.value) {
                    self.fwd[self.offsets[i + 1] + arc.to] = true;
                }
            }
        }
        // Every node in the final layer is accepting.
        for node in self.offsets[n]..self.fwd.len() {
            self.bwd[node] = true;
        }
        for i in (0..n).rev() {
            for arc in &self.layers[i] {
                if store.contains(self.vars[i], arc.value) && self.bwd[self.offsets[i + 1] + arc.to] {
                    self.bwd[self.offsets[i] + arc.from] = true;
                }
            }
        }

        for i in 0..n {
            self.buf.clear();
            self.buf.extend(store.values(self.vars[i]));
            for &val in &self.buf {
                let supported = self.layers[i]
                    .iter()
                    .any(|arc| arc.value == val && self.fwd[self.offsets[i] + arc.from] && self.bwd[self.offsets[i + 1] + arc.to]);
                if !supported {
                    store.remove(self.vars[i], val)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `mdd`: the assignment must trace a root-to-final path of `mdd`.
pub fn mdd(solver: &mut Solver, vars: &[VarId], mdd: Mdd) {
    assert_eq!(mdd.layers.len(), vars.len(), "mdd: one arc layer per variable");
    assert_eq!(mdd.nodes_per_layer.len(), vars.len() + 1, "mdd: nodes_per_layer must have n+1 entries");
    solver.post(Box::new(MddProp::new(vars, mdd)));
}
