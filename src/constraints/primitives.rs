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
            let why = vec![Premise::Eq {
                var: self.x,
                val: v,
            }];
            store.remove_because(self.y, v - self.c, why)?;
        }
        // y = w  =>  x != w + c.
        if store.is_fixed(self.y) {
            let w = store.value(self.y);
            let why = vec![Premise::Eq {
                var: self.y,
                val: w,
            }];
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
                        changed |=
                            fix_sign_product(store, y, store.value(x) * store.value(z), x, z)?;
                    }
                    [true, false, true] => {
                        changed |=
                            fix_sign_product(store, x, store.value(y) * store.value(z), y, z)?;
                    }
                    [true, true, false] => {
                        changed |=
                            fix_sign_product(store, z, store.value(y) * store.value(x), y, x)?;
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

fn fix_sign_product(
    store: &mut Store,
    target: VarId,
    value: i32,
    a: VarId,
    b: VarId,
) -> Result<bool, Inconsistency> {
    let why = vec![
        Premise::Eq {
            var: a,
            val: store.value(a),
        },
        Premise::Eq {
            var: b,
            val: store.value(b),
        },
    ];
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
        solver.post(Box::new(SignProducts {
            terms: terms.to_vec(),
        }));
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
        let why = vec![Premise::Le {
            var: self.y,
            bound: my,
        }];
        store.remove_above_because(self.x, my.saturating_sub(self.k), why)?;
        // y >= min(x) + k.
        let mx = store.min(self.x);
        let why = vec![Premise::Ge {
            var: self.x,
            bound: mx,
        }];
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
    solver.post(Box::new(Instantiation {
        vars: vars.to_vec(),
        vals: vals.to_vec(),
    }));
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
    solver.post(Box::new(AllEqual {
        vars: vars.to_vec(),
        common: Vec::new(),
        buf: Vec::new(),
    }));
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
        let y_bound = if self.is_min {
            store.min(self.y)
        } else {
            store.max(self.y)
        };
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
    solver.post(Box::new(Extremum {
        y,
        xs: xs.to_vec(),
        is_min: true,
    }));
}

/// Post `y = max(xs)`. `xs` must be non-empty.
pub fn maximum(solver: &mut Solver, y: VarId, xs: &[VarId]) {
    assert!(!xs.is_empty(), "maximum: empty list");
    solver.post(Box::new(Extremum {
        y,
        xs: xs.to_vec(),
        is_min: false,
    }));
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
// allDifferent (Régin, domain-consistent)
// ---------------------------------------------------------------------------

/// All variables take distinct values (Régin, domain-consistent).
///
/// Orientation: matching edges `var -> value`, non-matching `value -> var`,
/// so a directed path is an alternating path.
#[derive(Clone)]
struct AllDifferent {
    vars: Vec<VarId>,
    buf: Vec<i32>,
}

/// Kuhn augmenting step for the bipartite matching.
fn augment(
    u: usize,
    adj: &[Vec<usize>],
    val_to_var: &mut [i32],
    var_to_val: &mut [i32],
    seen: &mut [bool],
) -> bool {
    for &v in &adj[u] {
        if !seen[v] {
            seen[v] = true;
            let w = val_to_var[v];
            if w < 0 || augment(w as usize, adj, val_to_var, var_to_val, seen) {
                val_to_var[v] = u as i32;
                var_to_val[u] = v as i32;
                return true;
            }
        }
    }
    false
}

/// Tarjan strongly-connected-components DFS over the oriented graph.
#[allow(clippy::too_many_arguments)]
fn scc_dfs(
    u: usize,
    g: &[Vec<usize>],
    index: &mut [i64],
    low: &mut [i64],
    on_stack: &mut [bool],
    stack: &mut Vec<usize>,
    comp: &mut [usize],
    counter: &mut i64,
    ncomp: &mut usize,
) {
    index[u] = *counter;
    low[u] = *counter;
    *counter += 1;
    stack.push(u);
    on_stack[u] = true;
    for &v in &g[u] {
        if index[v] < 0 {
            scc_dfs(v, g, index, low, on_stack, stack, comp, counter, ncomp);
            low[u] = low[u].min(low[v]);
        } else if on_stack[v] {
            low[u] = low[u].min(index[v]);
        }
    }
    if low[u] == index[u] {
        loop {
            let w = stack.pop().unwrap();
            on_stack[w] = false;
            comp[w] = *ncomp;
            if w == u {
                break;
            }
        }
        *ncomp += 1;
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

        // Value universe (sorted, deduped); value index = position.
        let mut values: Vec<i32> = Vec::new();
        for &v in &self.vars {
            values.extend(store.values(v));
        }
        values.sort_unstable();
        values.dedup();
        let m = values.len();
        if m < n {
            return Err(Inconsistency); // pigeonhole
        }
        let index_of = |val: i32| values.binary_search(&val).unwrap();

        // Variable -> value-index adjacency.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &v) in self.vars.iter().enumerate() {
            adj[i] = store.values(v).map(&index_of).collect();
        }

        // Maximum matching (Kuhn).
        let mut var_to_val = vec![-1i32; n];
        let mut val_to_var = vec![-1i32; m];
        for i in 0..n {
            let mut seen = vec![false; m];
            if !augment(i, &adj, &mut val_to_var, &mut var_to_val, &mut seen) {
                return Err(Inconsistency); // variable i uncoverable
            }
        }

        // Oriented graph: nodes 0..n vars, n..n+m values.
        let total = n + m;
        let mut g: Vec<Vec<usize>> = vec![Vec::new(); total];
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
        let mut reachable = vec![false; total];
        let mut stack: Vec<usize> = Vec::new();
        for vidx in 0..m {
            if val_to_var[vidx] < 0 {
                reachable[n + vidx] = true;
                stack.push(n + vidx);
            }
        }
        while let Some(u) = stack.pop() {
            for &w in &g[u] {
                if !reachable[w] {
                    reachable[w] = true;
                    stack.push(w);
                }
            }
        }

        // Strongly-connected components.
        let mut comp = vec![usize::MAX; total];
        let mut index = vec![-1i64; total];
        let mut low = vec![0i64; total];
        let mut on_stack = vec![false; total];
        let mut tj_stack = Vec::new();
        let mut counter = 0i64;
        let mut ncomp = 0usize;
        for s in 0..total {
            if index[s] < 0 {
                scc_dfs(
                    s,
                    &g,
                    &mut index,
                    &mut low,
                    &mut on_stack,
                    &mut tj_stack,
                    &mut comp,
                    &mut counter,
                    &mut ncomp,
                );
            }
        }

        // Remove every non-matching edge that lies in no maximum matching.
        for (i, &var) in self.vars.iter().enumerate() {
            let matched = var_to_val[i] as usize;
            self.buf.clear();
            self.buf.extend(store.values(var));
            for &val in &self.buf {
                let vidx = index_of(val);
                if vidx == matched {
                    continue; // matching edge is always supported
                }
                if comp[i] != comp[n + vidx] && !reachable[n + vidx] {
                    store.remove(var, val)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `allDifferent(vars)` with Régin domain-consistent filtering.
pub fn all_different(solver: &mut Solver, vars: &[VarId]) {
    solver.post(Box::new(AllDifferent {
        vars: vars.to_vec(),
        buf: Vec::new(),
    }));
}

// ---------------------------------------------------------------------------
// precedence (value precedence)
// ---------------------------------------------------------------------------

/// Value precedence: for each consecutive pair `(s, t)`, the first `s` must
/// precede the first `t`. Decomposed through `intension`.
pub fn precedence(solver: &mut Solver, list: &[VarId], values: &[i32]) {
    for w in values.windows(2) {
        let (s, t) = (w[0], w[1]);
        for i in 0..list.len() {
            let earlier_s: Vec<_> = (0..i).map(|p| eq(var(list[p]), int(s as i64))).collect();
            intension(solver, imp(eq(var(list[i]), int(t as i64)), or(earlier_s)));
        }
    }
}
