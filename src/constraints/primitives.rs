//! Primitive constraints: disequality, ordering, instantiation, allEqual,
//! min/max, element, allDifferent, precedence.

use crate::constraints::intension::intension;
use crate::constraints::linear::Relation;
use crate::expr::{eq, imp, int, or, var};
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::store::{Premise, Solver, Store};

/// The constraint `x != y + c` (forward checking).
#[derive(Clone)]
pub struct NotEqualOffset {
    x: VarId,
    y: VarId,
    c: i32,
}

impl Propagator for NotEqualOffset {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.x, me, Event::Fix);
        store.subscribe(self.y, me, Event::Fix);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        // x = v  =>  y != v - c.
        if store.is_fixed(self.x) {
            let v = store.value(self.x);
            let why = vec![Premise::Eq { var: self.x, val: v }];
            store.remove_because(self.y, v - self.c, why)?;
        }
        // y = w  =>  x != w + c.
        if store.is_fixed(self.y) {
            let w = store.value(self.y);
            let why = vec![Premise::Eq { var: self.y, val: w }];
            store.remove_because(self.x, w + self.c, why)?;
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
// sign products: y = x * z for {-1, 1} variables
// ---------------------------------------------------------------------------

const SIGN_PRODUCT_BATCH: usize = 64;

/// A small batch of sign-product equalities. With `{-1, 1}` domains, fixing
/// any two variables fixes the third.
#[derive(Clone)]
struct SignProducts {
    terms: Vec<[VarId; 3]>,
}

impl Propagator for SignProducts {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &vars in &self.terms {
            for var in vars {
                store.subscribe(var, me, Event::Fix);
            }
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        loop {
            let mut changed = false;
            for &[y, x, z] in &self.terms {
                let fixed = [y, x, z].map(|var| store.is_fixed(var));
                match fixed {
                    [true, true, true] => {
                        let (yv, xv, zv) = (store.value(y), store.value(x), store.value(z));
                        if yv != xv * zv {
                            return Err(store.fail_because(vec![
                                Premise::Eq { var: y, val: yv },
                                Premise::Eq { var: x, val: xv },
                                Premise::Eq { var: z, val: zv },
                            ]));
                        }
                    }
                    [false, true, true] => {
                        changed |= fix_sign_product(store, y, store.value(x) * store.value(z), x, z)?;
                    }
                    [true, false, true] => {
                        changed |= fix_sign_product(store, x, store.value(y) * store.value(z), y, z)?;
                    }
                    [true, true, false] => {
                        changed |= fix_sign_product(store, z, store.value(y) * store.value(x), y, x)?;
                    }
                    _ => {}
                }
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

fn fix_sign_product(store: &mut Store, target: VarId, value: i32, a: VarId, b: VarId) -> Result<bool, Inconsistency> {
    let why = vec![Premise::Eq { var: a, val: store.value(a) }, Premise::Eq { var: b, val: store.value(b) }];
    if !store.contains(target, value) {
        return Err(store.fail_because(why));
    }
    let changed = !store.is_fixed(target);
    store.fix_because(target, value, why)?;
    Ok(changed)
}

/// Post `y = x * z` equalities for distinct variables over `{-1, 1}`.
pub fn sign_products(solver: &mut Solver, terms: &[[VarId; 3]]) {
    for terms in terms.chunks(SIGN_PRODUCT_BATCH) {
        solver.post(Box::new(SignProducts { terms: terms.to_vec() }));
    }
}

// ---------------------------------------------------------------------------
// Ordering: x + k <= y
// ---------------------------------------------------------------------------

/// \( x + k \le y \) (bounds propagation).
#[derive(Clone)]
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
        // x <= max(y) - k.
        let my = store.max(self.y);
        let why = vec![Premise::Le { var: self.y, bound: my }];
        store.remove_above_because(self.x, my.saturating_sub(self.k), why)?;
        // y >= min(x) + k.
        let mx = store.min(self.x);
        let why = vec![Premise::Ge { var: self.x, bound: mx }];
        store.remove_below_because(self.y, mx.saturating_add(self.k), why)?;
        Ok(())
    }
}

/// Post \( x \le y \).
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
            Relation::Le => less_or_equal(solver, a, b),
            Relation::Lt => less_than(solver, a, b),
            Relation::Ge => less_or_equal(solver, b, a),
            Relation::Gt => less_than(solver, b, a),
            _ => panic!("ordered: relation must be a comparison, got {rel:?}"),
        };
    }
}

// ---------------------------------------------------------------------------
// instantiation
// ---------------------------------------------------------------------------

/// Fixes each variable to a prescribed value.
#[derive(Clone)]
struct Instantiation {
    vars: Vec<VarId>,
    vals: Vec<i32>,
}

impl Propagator for Instantiation {
    fn register(&mut self, store: &mut Store, _me: PropId) {
        for &var in &self.vars {
            store.mark_relevant(var);
        }
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
    solver.post(Box::new(Instantiation { vars: vars.to_vec(), vals: vals.to_vec() }));
}

// ---------------------------------------------------------------------------
// allEqual (domain-consistent via intersection)
// ---------------------------------------------------------------------------

/// All variables take the same value; prunes each to the common domain.
#[derive(Clone)]
struct AllEqual {
    vars: Vec<VarId>,
    common: Vec<i32>,
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
    solver.post(Box::new(AllEqual { vars: vars.to_vec(), common: Vec::new(), buf: Vec::new() }));
}

// ---------------------------------------------------------------------------
// minimum / maximum
// ---------------------------------------------------------------------------

/// `y = min(xs)` or `y = max(xs)` (bounds reasoning).
#[derive(Clone)]
struct Extremum {
    y: VarId,
    xs: Vec<VarId>,
    is_min: bool,
}

impl Propagator for Extremum {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.y, me, Event::BoundChange);
        for &x in &self.xs {
            store.subscribe(x, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let mut lb = if self.is_min { i32::MAX } else { i32::MIN };
        let mut ub = lb;
        for &x in &self.xs {
            if self.is_min {
                lb = lb.min(store.min(x));
                ub = ub.min(store.max(x));
            } else {
                lb = lb.max(store.min(x));
                ub = ub.max(store.max(x));
            }
        }
        store.remove_below(self.y, lb)?;
        store.remove_above(self.y, ub)?;
        let y_bound = if self.is_min { store.min(self.y) } else { store.max(self.y) };
        for &x in &self.xs {
            if self.is_min {
                store.remove_below(x, y_bound)?;
            } else {
                store.remove_above(x, y_bound)?;
            }
        }
        Ok(())
    }
}

/// Post `y = min(xs)`. `xs` must be non-empty.
pub fn minimum(solver: &mut Solver, y: VarId, xs: &[VarId]) {
    assert!(!xs.is_empty(), "minimum: empty list");
    solver.post(Box::new(Extremum { y, xs: xs.to_vec(), is_min: true }));
}

/// Post `y = max(xs)`. `xs` must be non-empty.
pub fn maximum(solver: &mut Solver, y: VarId, xs: &[VarId]) {
    assert!(!xs.is_empty(), "maximum: empty list");
    solver.post(Box::new(Extremum { y, xs: xs.to_vec(), is_min: false }));
}

// ---------------------------------------------------------------------------
// element: value = array[idx]   (idx is 0-based)
// ---------------------------------------------------------------------------

/// `value = array[idx]`, 0-based; domain-consistent filtering.
#[derive(Clone)]
struct Element {
    array: Vec<VarId>,
    idx: VarId,
    value: VarId,
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

        // Prune index: i impossible if array[i] and value share no value.
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
            // Prune value: v impossible if no live index selects an entry containing v.
            self.buf.clear();
            self.buf.extend(store.values(self.value));
            for &v in &self.buf {
                let supported = store.values(self.idx).any(|i| store.contains(self.array[i as usize], v));
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
    solver.post(Box::new(Element { array: array.to_vec(), idx, value, buf: Vec::new() }));
}

/// `value = array[idx]`, 0-based, where `array` is constant.
#[derive(Clone)]
struct ElementConst {
    array: Vec<i32>,
    idx: VarId,
    value: VarId,
    buf: Vec<i32>,
}

impl Propagator for ElementConst {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.idx, me, Event::DomainChange);
        store.subscribe(self.value, me, Event::DomainChange);
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.array.len() as i32;
        store.remove_below(self.idx, 0)?;
        store.remove_above(self.idx, n - 1)?;

        self.buf.clear();
        self.buf.extend(store.values(self.idx));
        for &i in &self.buf {
            if !store.contains(self.value, self.array[i as usize]) {
                store.remove(self.idx, i)?;
            }
        }

        if store.is_fixed(self.idx) {
            store.fix(self.value, self.array[store.value(self.idx) as usize])?;
        } else {
            self.buf.clear();
            self.buf.extend(store.values(self.value));
            for &val in &self.buf {
                let supported = store.values(self.idx).any(|i| self.array[i as usize] == val);
                if !supported {
                    store.remove(self.value, val)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `value = array[idx]` for a constant array.
pub fn element_const(solver: &mut Solver, array: &[i32], idx: VarId, value: VarId) {
    assert!(!array.is_empty(), "elementConst: empty array");
    solver.post(Box::new(ElementConst { array: array.to_vec(), idx, value, buf: Vec::new() }));
}

// ---------------------------------------------------------------------------
// allDifferent (Régin, domain-consistent)
// ---------------------------------------------------------------------------

/// All variables take distinct values (Régin, domain-consistent).
///
/// Orientation: matching edges `var -> value`, non-matching `value -> var`,
/// so a directed path is an alternating path.
///
/// All scratch buffers persist across `propagate()` calls (cleared and reused,
/// never reallocated per call). `matching` warm-starts the maximum matching:
/// the previous assignment is kept as a hint and validated against the current
/// domains on each wake. It is deliberately not trailed — after a backtrack a
/// stale pair is still a valid hint because domains only grow, and any pair no
/// longer in the domain is simply re-augmented.
#[derive(Clone, Default)]
struct AllDifferent {
    vars: Vec<VarId>,
    /// Persistent matching hint: `matching[i]` = value assigned to `vars[i]`
    /// (`i32::MIN` = unmatched / no hint yet).
    matching: Vec<i32>,
    values: Vec<i32>,
    adj: Vec<Vec<usize>>,
    var_to_val: Vec<i32>,
    val_to_var: Vec<i32>,
    seen: Vec<bool>,
    g: Vec<Vec<usize>>,
    reachable: Vec<bool>,
    dfs_stack: Vec<usize>,
    comp: Vec<usize>,
    index: Vec<i64>,
    low: Vec<i64>,
    on_stack: Vec<bool>,
    aug_stack: Vec<(usize, usize, usize)>,
    scc_work: Vec<(usize, usize)>,
    tj_stack: Vec<usize>,
    /// Pending edge removals `(var index, value index)`, collected before any
    /// mutation so explanations can cite the pre-removal domains.
    removals: Vec<(usize, usize)>,
    /// Per-value-SCC explanation memo, shared by every edge removed against
    /// that component; cleared each call (stale domains would be unsound).
    scc_why: Vec<Option<Vec<Premise>>>,
    /// Reverse of `g`, built only when there are removals to explain.
    rg: Vec<Vec<usize>>,
    visited: Vec<bool>,
}

fn index_of(values: &[i32], val: i32) -> usize {
    values.binary_search(&val).unwrap()
}

/// Iterative Kuhn augmenting step. `seen` must be reset by the caller; `stack`
/// is scratch (var, next-adj-index, chosen-value) frames.
fn augment(
    start: usize,
    adj: &[Vec<usize>],
    val_to_var: &mut [i32],
    var_to_val: &mut [i32],
    seen: &mut [bool],
    stack: &mut Vec<(usize, usize, usize)>,
) -> bool {
    stack.clear();
    stack.push((start, 0, 0));
    while let Some(&(u, idx, _)) = stack.last() {
        if idx >= adj[u].len() {
            stack.pop();
            continue;
        }
        let v = adj[u][idx];
        stack.last_mut().unwrap().1 = idx + 1;
        if seen[v] {
            continue;
        }
        seen[v] = true;
        stack.last_mut().unwrap().2 = v;
        let w = val_to_var[v];
        if w < 0 {
            // Augmenting path found: assign every (var, chosen value) on the path.
            for &(fu, _, fv) in stack.iter() {
                var_to_val[fu] = fv as i32;
                val_to_var[fv] = fu as i32;
            }
            return true;
        }
        stack.push((w as usize, 0, 0));
    }
    false
}

/// Iterative Tarjan strongly-connected-components over the oriented graph.
/// `work` is scratch (node, next-edge-index) frames; `tj_stack` is the Tarjan
/// component stack.
#[allow(clippy::too_many_arguments)]
fn scc_visit(
    start: usize,
    g: &[Vec<usize>],
    index: &mut [i64],
    low: &mut [i64],
    on_stack: &mut [bool],
    tj_stack: &mut Vec<usize>,
    comp: &mut [usize],
    counter: &mut i64,
    ncomp: &mut usize,
    work: &mut Vec<(usize, usize)>,
) {
    work.clear();
    work.push((start, 0));
    while let Some(&(u, i)) = work.last() {
        if i == 0 {
            index[u] = *counter;
            low[u] = *counter;
            *counter += 1;
            tj_stack.push(u);
            on_stack[u] = true;
        }
        if i < g[u].len() {
            work.last_mut().unwrap().1 = i + 1;
            let v = g[u][i];
            if index[v] < 0 {
                work.push((v, 0));
            } else if on_stack[v] {
                low[u] = low[u].min(index[v]);
            }
        } else {
            if low[u] == index[u] {
                loop {
                    let w = tj_stack.pop().unwrap();
                    on_stack[w] = false;
                    comp[w] = *ncomp;
                    if w == u {
                        break;
                    }
                }
                *ncomp += 1;
            }
            work.pop();
            if let Some(&(p, _)) = work.last() {
                low[p] = low[p].min(low[u]);
            }
        }
    }
}

impl Propagator for AllDifferent {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.vars.len();
        if n == 0 {
            return Ok(());
        }
        let AllDifferent {
            vars, matching, values, adj, var_to_val, val_to_var, seen, g, reachable,
            dfs_stack, comp, index, low, on_stack, aug_stack, scc_work, tj_stack,
            removals, scc_why, rg, visited,
        } = self;

        // Value universe (sorted, deduped); value index = position.
        values.clear();
        for &v in vars.iter() {
            values.extend(store.values(v));
        }
        values.sort_unstable();
        values.dedup();
        let m = values.len();
        if m < n {
            return Err(Inconsistency); // pigeonhole
        }

        // Variable -> value-index adjacency.
        if adj.len() < n {
            adj.resize(n, Vec::new());
        }
        for (i, &v) in vars.iter().enumerate() {
            adj[i].clear();
            adj[i].extend(store.values(v).map(|val| index_of(values, val)));
        }

        // Warm-start the matching from the previous call: keep each hinted pair
        // that is still in the domain (validation makes the untrailed hint safe).
        var_to_val.clear();
        var_to_val.resize(n, -1);
        val_to_var.clear();
        val_to_var.resize(m, -1);
        if matching.len() != n {
            matching.clear();
            matching.resize(n, i32::MIN);
        }
        for i in 0..n {
            let mv = matching[i];
            if mv != i32::MIN && store.contains(vars[i], mv) {
                let vidx = index_of(values, mv);
                if val_to_var[vidx] < 0 {
                    var_to_val[i] = vidx as i32;
                    val_to_var[vidx] = i as i32;
                }
            }
        }

        // Re-augment only the broken variables (Kuhn); the kept pairs stay.
        seen.clear();
        seen.resize(m, false);
        for i in 0..n {
            if var_to_val[i] < 0 {
                seen.fill(false);
                if !augment(i, adj, val_to_var, var_to_val, seen, aug_stack) {
                    if !store.explaining() {
                        return Err(Inconsistency);
                    }
                    // Variable i is uncoverable, and the failed DFS already
                    // holds the Hall violator: `seen` marks the value set A it
                    // exhausted, each such value matched to a distinct
                    // variable, so those variables plus i are |A|+1 variables
                    // confined to A. Pin their (still unmutated) domains.
                    let mut why = Vec::new();
                    store.domain_premises(vars[i], &mut why);
                    for v in 0..m {
                        if seen[v] {
                            debug_assert!(val_to_var[v] >= 0, "seen value must be matched");
                            store.domain_premises(vars[val_to_var[v] as usize], &mut why);
                        }
                    }
                    return Err(store.fail_because(why));
                }
            }
        }

        // Persist the matching (as concrete values) for the next call.
        for i in 0..n {
            matching[i] = values[var_to_val[i] as usize];
        }

        // Oriented graph: nodes 0..n vars, n..n+m values.
        let total = n + m;
        while g.len() < total {
            g.push(Vec::new());
        }
        for cell in g.iter_mut().take(total) {
            cell.clear();
        }
        for i in 0..n {
            let mv = var_to_val[i] as usize;
            g[i].push(n + mv); // matching: var -> value
            for &vidx in &adj[i] {
                if vidx != mv {
                    g[n + vidx].push(i); // non-matching: value -> var
                }
            }
        }

        // Reachable from free (unmatched) values.
        reachable.clear();
        reachable.resize(total, false);
        dfs_stack.clear();
        for vidx in 0..m {
            if val_to_var[vidx] < 0 {
                reachable[n + vidx] = true;
                dfs_stack.push(n + vidx);
            }
        }
        while let Some(u) = dfs_stack.pop() {
            for &w in &g[u] {
                if !reachable[w] {
                    reachable[w] = true;
                    dfs_stack.push(w);
                }
            }
        }

        // Strongly-connected components.
        comp.clear();
        comp.resize(total, usize::MAX);
        index.clear();
        index.resize(total, -1);
        low.clear();
        low.resize(total, 0);
        on_stack.clear();
        on_stack.resize(total, false);
        tj_stack.clear();
        let mut counter = 0i64;
        let mut ncomp = 0usize;
        for s in 0..total {
            if index[s] < 0 {
                scc_visit(s, g, index, low, on_stack, tj_stack, comp, &mut counter, &mut ncomp, scc_work);
            }
        }

        // Remove every non-matching edge that lies in no maximum matching.
        // Collect first, explain, then remove: an explanation must cite
        // domains as they stood before this call's own removals, or the LCG
        // trail rejects it and falls back to the whole scope.
        removals.clear();
        for (i, &var) in vars.iter().enumerate() {
            let matched = var_to_val[i] as usize;
            for val in store.values(var) {
                let vidx = index_of(values, val);
                if vidx != matched && comp[i] != comp[n + vidx] && !reachable[n + vidx] {
                    removals.push((i, vidx));
                }
            }
        }
        if removals.is_empty() {
            return Ok(());
        }
        if !store.explaining() {
            for &(i, vidx) in removals.iter() {
                store.remove(vars[i], values[vidx])?;
            }
            return Ok(());
        }

        // One explanation per value SCC, shared by every edge removed against
        // it: the variables that can reach the SCC (reverse DFS) are matched
        // one-to-one with the values that can reach it and their domains lie
        // inside that value set — a Hall set T saturating a set A containing
        // the removed value, so any variable outside T loses it. Pinning
        // T's domains is a sound reason for x != v (T may exceed the minimal
        // Hall set; a superset only weakens the learned clause).
        // ponytail: reverse graph + one DFS per referenced SCC is O(edges)
        // each — bounded by the Régin pass itself.
        while rg.len() < total {
            rg.push(Vec::new());
        }
        for cell in rg.iter_mut().take(total) {
            cell.clear();
        }
        for (u, out) in g.iter().enumerate().take(total) {
            for &w in out {
                rg[w].push(u);
            }
        }
        scc_why.clear();
        scc_why.resize(ncomp, None);
        visited.clear();
        visited.resize(total, false);
        let mut scc_uses = vec![0u32; ncomp];
        for &(_, vidx) in removals.iter() {
            let cid = comp[n + vidx];
            scc_uses[cid] += 1;
            if scc_why[cid].is_some() {
                continue;
            }
            visited.fill(false);
            visited[n + vidx] = true;
            dfs_stack.clear();
            dfs_stack.push(n + vidx);
            while let Some(u) = dfs_stack.pop() {
                for &w in &rg[u] {
                    if !visited[w] {
                        visited[w] = true;
                        dfs_stack.push(w);
                    }
                }
            }
            let mut why = Vec::new();
            for (j, &var) in vars.iter().enumerate() {
                if visited[j] {
                    store.domain_premises(var, &mut why);
                }
            }
            scc_why[cid] = Some(why);
        }
        for &(i, vidx) in removals.iter() {
            let cid = comp[n + vidx];
            scc_uses[cid] -= 1;
            // The last edge of an SCC takes the premise vector; earlier ones clone.
            let why = if scc_uses[cid] == 0 {
                scc_why[cid].take().expect("explained above")
            } else {
                scc_why[cid].as_ref().expect("explained above").clone()
            };
            store.remove_because(vars[i], values[vidx], why)?;
        }
        Ok(())
    }
}

/// Post `allDifferent(vars)` with Régin domain-consistent filtering.
pub fn all_different(solver: &mut Solver, vars: &[VarId]) {
    solver.post(Box::new(AllDifferent { vars: vars.to_vec(), ..Default::default() }));
}

// ---------------------------------------------------------------------------
// precedence (value precedence)
// ---------------------------------------------------------------------------

/// Value precedence: for each consecutive pair `(s, t)`, the first `s` must
/// precede the first `t`. Decomposed through `intension`.
pub fn precedence(solver: &mut Solver, list: &[VarId], values: &[i32]) {
    precedence_with_covered(solver, list, values, false);
}

/// Value precedence with optional coverage: when `covered` is true, every value
/// in `values` must occur at least once in `list`.
pub fn precedence_with_covered(solver: &mut Solver, list: &[VarId], values: &[i32], covered: bool) {
    for w in values.windows(2) {
        let (s, t) = (w[0], w[1]);
        for i in 0..list.len() {
            let earlier_s: Vec<_> = (0..i).map(|p| eq(var(list[p]), int(s as i64))).collect();
            intension(solver, imp(eq(var(list[i]), int(t as i64)), or(earlier_s)));
        }
    }
    if covered {
        for &value in values {
            let occurs = list.iter().map(|&v| eq(var(v), int(value as i64))).collect();
            intension(solver, or(occurs));
        }
    }
}
