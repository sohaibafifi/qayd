//! Counting constraints: `count`, `cardinality` (GCC bounds), `nValues`.

use crate::constraints::linear::Relation;
use crate::constraints::primitives::{index_of, scc_visit};
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Priority, Propagator};
use crate::store::{Solver, Store};

/// `#{ i : vars[i] = value }  rel  k`.
#[derive(Clone)]
struct Count {
    vars: Vec<VarId>,
    value: i32,
    rel: Relation,
    k: i64,
}

impl Count {
    /// `(lb, ub)`: variables definitely equal to `value`, and those that could be.
    fn counts(&self, store: &Store) -> (i64, i64) {
        let mut lb = 0;
        let mut ub = 0;
        for &v in &self.vars {
            if store.contains(v, self.value) {
                ub += 1;
                if store.is_fixed(v) {
                    lb += 1;
                }
            }
        }
        (lb, ub)
    }

    /// Enforce \( \texttt{count} \le k \): at `lb == k`, no other variable may take the value.
    fn enforce_le(&self, store: &mut Store, k: i64) -> Result<(), Inconsistency> {
        let (lb, _) = self.counts(store);
        if lb > k {
            return Err(Inconsistency);
        }
        if lb == k {
            for &v in &self.vars {
                if !store.is_fixed(v) && store.contains(v, self.value) {
                    store.remove(v, self.value)?;
                }
            }
        }
        Ok(())
    }

    /// Enforce \( \texttt{count} \ge k \): at `ub == k`, every variable that can take the value must.
    fn enforce_ge(&self, store: &mut Store, k: i64) -> Result<(), Inconsistency> {
        let (_, ub) = self.counts(store);
        if ub < k {
            return Err(Inconsistency);
        }
        if ub == k {
            for &v in &self.vars {
                if store.contains(v, self.value) && !store.is_fixed(v) {
                    store.fix(v, self.value)?;
                }
            }
        }
        Ok(())
    }
}

impl Propagator for Count {
    fn priority(&self) -> Priority {
        Priority::Linear
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        match self.rel {
            Relation::Le => self.enforce_le(store, self.k)?,
            Relation::Lt => self.enforce_le(store, self.k.saturating_sub(1))?,
            Relation::Ge => self.enforce_ge(store, self.k)?,
            Relation::Gt => self.enforce_ge(store, self.k.saturating_add(1))?,
            Relation::Eq => {
                self.enforce_le(store, self.k)?;
                self.enforce_ge(store, self.k)?;
            }
            Relation::Ne => {
                let (lb, ub) = self.counts(store);
                if lb == ub && lb == self.k {
                    return Err(Inconsistency);
                }
            }
        }
        Ok(())
    }
}

/// Post `#{ i : vars[i] = value }  rel  k`.
pub fn count(solver: &mut Solver, vars: &[VarId], value: i32, rel: Relation, k: i64) {
    record_boolean_count_relaxation(solver, vars, value, rel, k);
    solver.post(Box::new(Count { vars: vars.to_vec(), value, rel, k }));
}

fn record_boolean_count_relaxation(solver: &mut Solver, vars: &[VarId], value: i32, rel: Relation, k: i64) {
    if !matches!(value, 0 | 1) || vars.iter().any(|&var| solver.store.min(var) < 0 || solver.store.max(var) > 1) {
        return;
    }
    let mut coefficients = vec![1; vars.len()];
    let rhs = if value == 1 {
        k
    } else {
        coefficients.fill(-1);
        let Ok(count) = i64::try_from(vars.len()) else {
            return;
        };
        let Some(rhs) = k.checked_sub(count) else {
            return;
        };
        rhs
    };
    crate::constraints::linear::record_relaxation(solver, &coefficients, vars, rel, rhs);
}

// ===========================================================================
// cardinality (GCC bounds, Régin flow-based domain consistency)
// ===========================================================================

/// Generalized cardinality with constant bounds, filtered by Régin's flow
/// network (AAAI-96). A feasible integral flow of value `n` in the value
/// network — source `s`, value nodes, variable nodes (sink omitted since each
/// variable takes exactly one value) — is exactly a GCC solution; an edge
/// `x = v` is arc-consistent iff it lies on a residual cycle, i.e. `x` and `v`
/// share a strongly-connected component. See the module tests for the
/// alldiff (`low=0,high=1`) reduction that pins the arc directions.
///
/// All scratch persists across calls. `hint` warm-starts the flow: last call's
/// assignment, validated against the current domains (domains only grow on
/// backtrack, so a stale pair is a safe hint — an invalid one is simply
/// re-augmented). Explanations use the `Cause::Scope` fallback throughout
/// (bare `remove`/`Err`); generalized-Hall reasons from flow cuts are follow-up.
#[derive(Clone, Default)]
struct Cardinality {
    vars: Vec<VarId>,
    listed_values: Vec<i32>,
    listed_low: Vec<i64>,
    listed_high: Vec<i64>,
    /// Persistent flow hint: `hint[i]` = value last assigned to `vars[i]`
    /// (`i32::MIN` = none). Untrailed; re-validated every call.
    hint: Vec<i32>,
    values: Vec<i32>,
    low: Vec<i32>,
    high: Vec<i32>,
    assign: Vec<i32>,
    count: Vec<i32>,
    adj: Vec<Vec<usize>>,
    occ: Vec<Vec<usize>>,
    queue: Vec<usize>,
    visited_var: Vec<bool>,
    visited_val: Vec<bool>,
    par_var_of_val: Vec<i32>,
    par_val_of_var: Vec<i32>,
    edge_var: Vec<i32>,
    prev_val: Vec<i32>,
    g: Vec<Vec<usize>>,
    comp: Vec<usize>,
    index: Vec<i64>,
    scc_low: Vec<i64>,
    on_stack: Vec<bool>,
    tj_stack: Vec<usize>,
    scc_work: Vec<(usize, usize)>,
    removals: Vec<(usize, usize)>,
    domain_buf: Vec<i32>,
}

/// Phase 1 (upper bounds): route unassigned variable `start` to a value with
/// spare capacity, displacing occupants along an augmenting path (iterative
/// BFS b-matching). Returns `false` if no placement respects the `high` caps.
#[allow(clippy::too_many_arguments)]
fn place_var(
    start: usize,
    adj: &[Vec<usize>],
    high: &[i32],
    assign: &mut [i32],
    count: &mut [i32],
    occ: &mut [Vec<usize>],
    queue: &mut Vec<usize>,
    visited_var: &mut [bool],
    visited_val: &mut [bool],
    par_var_of_val: &mut [i32],
    par_val_of_var: &mut [i32],
) -> bool {
    visited_var.iter_mut().for_each(|b| *b = false);
    visited_val.iter_mut().for_each(|b| *b = false);
    queue.clear();
    queue.push(start);
    visited_var[start] = true;
    let mut head = 0;
    let mut found: i32 = -1;
    'outer: while head < queue.len() {
        let u = queue[head];
        head += 1;
        for &v in &adj[u] {
            if visited_val[v] {
                continue;
            }
            visited_val[v] = true;
            par_var_of_val[v] = u as i32;
            if count[v] < high[v] {
                found = v as i32;
                break 'outer;
            }
            for &w in &occ[v] {
                if !visited_var[w] {
                    visited_var[w] = true;
                    par_val_of_var[w] = v as i32;
                    queue.push(w);
                }
            }
        }
    }
    if found < 0 {
        return false;
    }
    let mut v = found as usize;
    loop {
        let u = par_var_of_val[v] as usize;
        let old = assign[u];
        assign[u] = v as i32;
        count[v] += 1;
        occ[v].push(u);
        if old < 0 {
            break;
        }
        let old = old as usize;
        count[old] -= 1;
        if let Some(pos) = occ[old].iter().position(|&z| z == u) {
            occ[old].swap_remove(pos);
        }
        v = old;
    }
    true
}

/// Phase 2 (lower bounds): raise `count[def]` by one by finding an augmenting
/// path from a surplus value (`count > low`) to `def` and rerouting one
/// variable along each edge. Returns `false` if no surplus can reach `def`.
#[allow(clippy::too_many_arguments)]
fn reroute_to(
    def: usize,
    low: &[i32],
    adj: &[Vec<usize>],
    assign: &mut [i32],
    count: &mut [i32],
    occ: &mut [Vec<usize>],
    queue: &mut Vec<usize>,
    visited_val: &mut [bool],
    edge_var: &mut [i32],
    prev_val: &mut [i32],
) -> bool {
    visited_val.iter_mut().for_each(|b| *b = false);
    queue.clear();
    for v in 0..count.len() {
        if count[v] > low[v] {
            visited_val[v] = true;
            prev_val[v] = -1;
            edge_var[v] = -1;
            queue.push(v);
        }
    }
    let mut head = 0;
    let mut reached = false;
    'outer: while head < queue.len() {
        let a = queue[head];
        head += 1;
        for &w in &occ[a] {
            for &b in &adj[w] {
                if b == a || visited_val[b] {
                    continue;
                }
                visited_val[b] = true;
                prev_val[b] = a as i32;
                edge_var[b] = w as i32;
                if b == def {
                    reached = true;
                    break 'outer;
                }
                queue.push(b);
            }
        }
    }
    if !reached {
        return false;
    }
    let mut b = def;
    while prev_val[b] >= 0 {
        let a = prev_val[b] as usize;
        let w = edge_var[b] as usize;
        if let Some(pos) = occ[a].iter().position(|&z| z == w) {
            occ[a].swap_remove(pos);
        }
        occ[b].push(w);
        assign[w] = b as i32;
        count[a] -= 1;
        count[b] += 1;
        b = a;
    }
    true
}

impl Propagator for Cardinality {
    fn priority(&self) -> Priority {
        // Flow repair + Tarjan SCC per call: costs like allDifferent's Régin
        // pass and more, so run after the cheap and linear bands.
        Priority::Expensive
    }

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
        let Cardinality {
            vars,
            listed_values,
            listed_low,
            listed_high,
            hint,
            values,
            low,
            high,
            assign,
            count,
            adj,
            occ,
            queue,
            visited_var,
            visited_val,
            par_var_of_val,
            par_val_of_var,
            edge_var,
            prev_val,
            g,
            comp,
            index,
            scc_low,
            on_stack,
            tj_stack,
            scc_work,
            removals,
            domain_buf,
        } = self;

        // Value universe = union of current domains plus every listed value.
        values.clear();
        for &v in vars.iter() {
            values.extend(store.values(v));
        }
        values.extend(listed_values.iter().copied());
        values.sort_unstable();
        values.dedup();
        let m = values.len();

        // Bounds per value index: listed values take their posted [low, high],
        // every other (unlisted) value is unconstrained [0, n].
        low.clear();
        low.resize(m, 0);
        high.clear();
        high.resize(m, n as i32);
        for j in 0..listed_values.len() {
            let vidx = index_of(values, listed_values[j]);
            let lo = listed_low[j];
            let hi = listed_high[j];
            if lo > hi || lo > n as i64 {
                return Err(Inconsistency);
            }
            low[vidx] = lo.max(0) as i32;
            high[vidx] = hi.clamp(0, n as i64) as i32;
        }

        // Variable -> value-index adjacency.
        if adj.len() < n {
            adj.resize(n, Vec::new());
        }
        for (i, &v) in vars.iter().enumerate() {
            adj[i].clear();
            adj[i].extend(store.values(v).map(|val| index_of(values, val)));
        }

        // Warm-start: keep each hinted pair still in domain and within its high.
        assign.clear();
        assign.resize(n, -1);
        count.clear();
        count.resize(m, 0);
        if occ.len() < m {
            occ.resize(m, Vec::new());
        }
        for cell in occ.iter_mut().take(m) {
            cell.clear();
        }
        if hint.len() != n {
            hint.clear();
            hint.resize(n, i32::MIN);
        }
        for i in 0..n {
            let hv = hint[i];
            if hv != i32::MIN && store.contains(vars[i], hv) {
                let vidx = index_of(values, hv);
                if count[vidx] < high[vidx] {
                    assign[i] = vidx as i32;
                    count[vidx] += 1;
                    occ[vidx].push(i);
                }
            }
        }

        // Phase 1: place every still-unassigned variable respecting highs.
        visited_var.clear();
        visited_var.resize(n, false);
        visited_val.clear();
        visited_val.resize(m, false);
        par_var_of_val.clear();
        par_var_of_val.resize(m, -1);
        par_val_of_var.clear();
        par_val_of_var.resize(n, -1);
        for i in 0..n {
            if assign[i] < 0
                && !place_var(i, adj, high, assign, count, occ, queue, visited_var, visited_val, par_var_of_val, par_val_of_var)
            {
                return Err(Inconsistency); // upper-infeasible
            }
        }

        // Phase 2: satisfy every lower bound by rerouting from surplus values.
        edge_var.clear();
        edge_var.resize(m, -1);
        prev_val.clear();
        prev_val.resize(m, -1);
        for v in 0..m {
            while count[v] < low[v] {
                if !reroute_to(v, low, adj, assign, count, occ, queue, visited_val, edge_var, prev_val) {
                    return Err(Inconsistency); // lower-infeasible
                }
            }
        }

        // Persist the flow (as concrete values) for the next call.
        for i in 0..n {
            hint[i] = values[assign[i] as usize];
        }

        // Residual graph: n var nodes, m value nodes, one source s = n + m
        // (sink omitted, saturated at [1,1]). Slack arcs live on the s side.
        let src = n + m;
        let total = src + 1;
        while g.len() < total {
            g.push(Vec::new());
        }
        for cell in g.iter_mut().take(total) {
            cell.clear();
        }
        for vidx in 0..m {
            if count[vidx] < high[vidx] {
                g[src].push(n + vidx); // s -> v: value can absorb more
            }
            if count[vidx] > low[vidx] {
                g[n + vidx].push(src); // v -> s: value can release one (lower-bound arc)
            }
        }
        for i in 0..n {
            let a = assign[i];
            for &vidx in &adj[i] {
                if vidx as i32 == a {
                    g[i].push(n + vidx); // x -> v: x could leave v
                } else {
                    g[n + vidx].push(i); // v -> x: x could take v
                }
            }
        }

        // Strongly-connected components.
        comp.clear();
        comp.resize(total, usize::MAX);
        index.clear();
        index.resize(total, -1);
        scc_low.clear();
        scc_low.resize(total, 0);
        on_stack.clear();
        on_stack.resize(total, false);
        tj_stack.clear();
        let mut counter = 0i64;
        let mut ncomp = 0usize;
        for s in 0..total {
            if index[s] < 0 {
                scc_visit(s, g, index, scc_low, on_stack, tj_stack, comp, &mut counter, &mut ncomp, scc_work);
            }
        }

        // An edge x = v not on the current flow survives iff x and v share an
        // SCC; otherwise it lies in no max flow and is removed.
        removals.clear();
        for i in 0..n {
            let a = assign[i];
            domain_buf.clear();
            domain_buf.extend(store.values(vars[i]));
            for &val in domain_buf.iter() {
                let vidx = index_of(values, val);
                if vidx as i32 != a && comp[i] != comp[n + vidx] {
                    removals.push((i, vidx));
                }
            }
        }
        for &(i, vidx) in removals.iter() {
            store.remove(vars[i], values[vidx])?;
        }
        Ok(())
    }
}

/// Restrict every variable's domain to a fixed set of values (for closed GCC).
#[derive(Clone)]
struct InSet {
    vars: Vec<VarId>,
    set: Vec<i32>,
    buf: Vec<i32>,
}

impl Propagator for InSet {
    fn priority(&self) -> Priority {
        Priority::Cheap
    }

    fn register(&mut self, store: &mut Store, _me: PropId) {
        for &var in &self.vars {
            store.mark_relevant(var);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        for &v in &self.vars {
            self.buf.clear();
            self.buf.extend(store.values(v));
            for &val in &self.buf {
                if !self.set.contains(&val) {
                    store.remove(v, val)?;
                }
            }
        }
        Ok(())
    }
}

/// Post `cardinality`: count of variables equal to `values[j]` lies in
/// `[low[j], high[j]]`. When `closed`, every variable must take a value from
/// `values`. Bounds form, decomposed into independent `count` ranges.
pub fn cardinality(solver: &mut Solver, vars: &[VarId], values: &[i32], low: &[i64], high: &[i64], closed: bool) {
    assert_eq!(values.len(), low.len(), "cardinality: values/low mismatch");
    assert_eq!(values.len(), high.len(), "cardinality: values/high mismatch");
    if vars.iter().all(|&var| solver.store.min(var) >= 0 && solver.store.max(var) <= 1) {
        for ((&value, &lower), &upper) in values.iter().zip(low).zip(high) {
            record_boolean_count_relaxation(solver, vars, value, Relation::Ge, lower);
            record_boolean_count_relaxation(solver, vars, value, Relation::Le, upper);
        }
        if closed && !values.contains(&0) {
            record_boolean_count_relaxation(solver, vars, 0, Relation::Eq, 0);
        }
        if closed && !values.contains(&1) {
            record_boolean_count_relaxation(solver, vars, 1, Relation::Eq, 0);
        }
    }
    solver.post(Box::new(Cardinality {
        vars: vars.to_vec(),
        listed_values: values.to_vec(),
        listed_low: low.to_vec(),
        listed_high: high.to_vec(),
        ..Default::default()
    }));
    if closed {
        solver.post(Box::new(InSet { vars: vars.to_vec(), set: values.to_vec(), buf: Vec::new() }));
    }
}

// ===========================================================================
// nValues
// ===========================================================================

/// `#{ distinct values among vars }  rel  k`. Bounds reasoning: `lb` is the max
/// of the distinct-fixed count and a greedy disjoint-domains count (Bessiere et
/// al.); `ub = min(n, |union of domains|)`. Two forced endpoints prune: when the
/// distinct count is forced to 1 all variables must be equal (intersect); when
/// it is forced to `|union|` every union value with a single provider is fixed.
#[derive(Clone)]
struct NValues {
    vars: Vec<VarId>,
    rel: Relation,
    k: i64,
    fixed: Vec<i32>,
    union: Vec<i32>,
    common: Vec<i32>,
    buf: Vec<i32>,
    order: Vec<usize>,
}

/// Greedy lower bound on the number of distinct values: the largest set of
/// variables with pairwise-disjoint `[min, max]` intervals (disjoint intervals
/// ⇒ disjoint domains ⇒ distinct values). Exact for interval domains, a valid
/// bound otherwise. Sorts variables by interval max and sweeps.
fn disjoint_lb(store: &Store, vars: &[VarId], order: &mut Vec<usize>) -> i64 {
    order.clear();
    order.extend(0..vars.len());
    order.sort_by_key(|&i| store.max(vars[i]));
    let mut last_max = i32::MIN;
    let mut picked = 0i64;
    for &i in order.iter() {
        if store.min(vars[i]) > last_max {
            picked += 1;
            last_max = store.max(vars[i]);
        }
    }
    picked
}

impl Propagator for NValues {
    fn priority(&self) -> Priority {
        Priority::Linear
    }

    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        // Own mutations never re-wake this propagator (`wake` skips `current`),
        // so iterate to an internal fixpoint: every round re-derives the bounds
        // and re-checks feasibility on the mutated state — the final round is
        // an exact check when the pruning below completed the assignment.
        loop {
            self.fixed.clear();
            self.union.clear();
            for &v in &self.vars {
                if store.is_fixed(v) {
                    self.fixed.push(store.value(v));
                }
                self.union.extend(store.values(v));
            }
            self.fixed.sort_unstable();
            self.fixed.dedup();
            self.union.sort_unstable();
            self.union.dedup();

            let n = self.vars.len();
            let lb = (self.fixed.len() as i64).max(disjoint_lb(store, &self.vars, &mut self.order));
            let ub = (n.min(self.union.len())) as i64;

            let infeasible = match self.rel {
                Relation::Ge => ub < self.k,
                Relation::Gt => ub <= self.k,
                Relation::Le => lb > self.k,
                Relation::Lt => lb >= self.k,
                Relation::Eq => ub < self.k || lb > self.k,
                Relation::Ne => lb == ub && lb == self.k,
            };
            if infeasible {
                return Err(Inconsistency);
            }

            if n == 0 {
                return Ok(());
            }

            let mut changed = false;

            // Forced max distinct == 1  =>  every variable equal: intersect domains.
            let forced_one = matches!((self.rel, self.k), (Relation::Le, 1) | (Relation::Lt, 2) | (Relation::Eq, 1));
            if forced_one {
                self.common.clear();
                self.common.extend(store.values(self.vars[0]));
                self.common.retain(|&v| self.vars[1..].iter().all(|&u| store.contains(u, v)));
                if self.common.is_empty() {
                    return Err(Inconsistency);
                }
                for &v in &self.vars {
                    self.buf.clear();
                    self.buf.extend(store.values(v));
                    for &val in &self.buf {
                        if !self.common.contains(&val) {
                            store.remove(v, val)?;
                            changed = true;
                        }
                    }
                }
            }

            // Forced min distinct == |union| (<= n)  =>  every union value must be
            // used; a value with a single in-domain provider is fixed to it.
            let forced_min = match self.rel {
                Relation::Ge => self.k,
                Relation::Gt => self.k + 1,
                Relation::Eq => self.k,
                _ => i64::MIN,
            };
            if forced_min == self.union.len() as i64 && (self.union.len() as i64) <= n as i64 {
                // Every union value must be used. A value reachable from a single
                // variable forces that variable; a value with no provider left (its
                // sole provider was just fixed to another required value) makes the
                // instance infeasible. Fixes cascade; earlier-visited values are
                // rechecked by the next fixpoint round.
                for &val in &self.union {
                    let mut provider = None;
                    let mut providers = 0u32;
                    for (i, &v) in self.vars.iter().enumerate() {
                        if store.contains(v, val) {
                            providers += 1;
                            provider = Some(i);
                            if providers > 1 {
                                break;
                            }
                        }
                    }
                    match providers {
                        0 => return Err(Inconsistency),
                        1 => {
                            let var = self.vars[provider.unwrap()];
                            if !store.is_fixed(var) {
                                changed = true;
                            }
                            store.fix(var, val)?;
                        }
                        _ => {}
                    }
                }
            }

            if !changed {
                return Ok(());
            }
        }
    }
}

/// Post `#{ distinct values among vars }  rel  k`.
pub fn n_values(solver: &mut Solver, vars: &[VarId], rel: Relation, k: i64) {
    solver.post(Box::new(NValues {
        vars: vars.to_vec(),
        rel,
        k,
        fixed: Vec::new(),
        union: Vec::new(),
        common: Vec::new(),
        buf: Vec::new(),
        order: Vec::new(),
    }));
}
