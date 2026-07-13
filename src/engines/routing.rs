//! Exact integer lowering for the small routing core.
//!
//! This handles closed routes with direct matrix edge-sum objectives:
//! `sum_edges(route, (i, j) => dist[i][j], start=d, end=d)`. Multi-route models
//! use replicated depot nodes and one Hamiltonian `circuit`. Optional identical
//! per-route capacities are lowered with MTZ-style load bounds plus a final
//! fixed-successor checker. Time windows and scan-style state remain with the
//! list local-search fallback until they have sound integer views.

#![cfg_attr(not(feature = "python"), allow(dead_code))]

use std::sync::atomic::{AtomicBool, AtomicI64};
use std::sync::Arc;

use crate::constraints::graph;
use crate::constraints::intension;
use crate::constraints::list as list_constraints;
use crate::constraints::primitives;
use crate::engines::ls::lists as ls_lists;
use crate::expr;
use crate::ids::{ListId, PropId, VarId};
use crate::model::list::{
    scan::{lower_scan, scan_routing_signature, ScanRoutingSpec},
    CollectionModel, CollectionSolution, Constraint, Expr, ExprId, GlobalConstraint, Iterable, Op, ReduceOp, Reduction,
};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::search::{self, Objective as SearchObjective, SolveStats};
use crate::store::{Premise, Solver, Store};

const MAX_INTEGER_ROUTING_NODES: usize = 32;

/// Local-search iteration cap for the exact backend's warm start: enough for a
/// good incumbent on small models, bounded so it terminates without a stop flag.
const WARM_START_ITERS: u64 = 2000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatrixOrientation {
    Direct,
    Reversed,
}

struct DirectEdgeMatrix {
    values: Arc<Vec<Vec<i64>>>,
    orientation: MatrixOrientation,
}

#[derive(Clone, PartialEq)]
struct CapacitySpec {
    demands: Vec<i32>,
    capacity: i32,
}

/// A homogeneous per-route weighted item sum: `(weights, min, max)`, where each
/// weight is `(item, coefficient)` and the route's weighted membership sum must
/// lie in `[min, max]`.
type ItemSumSpec = (Vec<(i32, i64)>, i64, i64);

struct RoutingSpec {
    depot_count: usize,
    depot: i32,
    items: Vec<i32>,
    edge: DirectEdgeMatrix,
    capacity: Option<CapacitySpec>,
    /// Label-free same-route requirements: each pair of items must share a route.
    same_list: Vec<(i32, i32)>,
    /// Homogeneous per-route length bounds (every route has `[min, max]` customers),
    /// or `None`. Route-specific lengths are unsafe under depot symmetry, so only the
    /// all-routes-identical form is accepted.
    length: Option<(usize, usize)>,
    /// Homogeneous per-route weighted item sum `(weights, min, max)`, or `None`. The
    /// non-capacity `Sum` shape (`>=`, `==`, or negative weights); a `Sum(nonneg) <=
    /// cap` is parsed as capacity instead. Route-specific bounds are rejected.
    item_sum: Option<ItemSumSpec>,
    /// Homogeneous per-route `Sum` scans (each a cumulative resource extension).
    /// A route may carry several (e.g. one time window per customer); every route
    /// carries the identical set, since route-specific scans are unsafe under
    /// depot-copy symmetry. Empty means no scan.
    scan: Vec<ScanRoutingSpec>,
    /// Optional-list model: `Some(pool_list)` marks it optional and carries the
    /// MODEL list index of the free pool (its LS partition), `None` for a
    /// non-optional model. `depot_count` includes one extra "pool depot" (the last
    /// depot index) whose zero-cost route absorbs dropped items; every cost channel
    /// is gated by `in_pool` so a pooled item contributes 0 everywhere.
    /// `depot_count == k + 1` real routes when set; `depot_count == k` otherwise.
    /// The pool depot is always the last depot index; `pool_list` is the distinct
    /// LS list index (which need not be last among the model's lists).
    pool_list: Option<usize>,
    /// Homogeneous per-route reified `Sum` scan folded into the objective, or
    /// `None`. Only set for pool models; a pooled customer contributes 0 (gated
    /// by `in_pool`). The `op`/`rhs` fields are unused (it is scored, not bounded).
    scan_objective: Option<ScanRoutingSpec>,
}

/// The model's homogeneous per-route side constraints, classified.
struct RouteSideConstraints {
    capacity: Option<CapacitySpec>,
    length: Option<(usize, usize)>,
    item_sum: Option<ItemSumSpec>,
    scan: Vec<ScanRoutingSpec>,
}

pub(crate) struct RoutingIntegerOutcome {
    pub solution: CollectionSolution,
    pub stats: SolveStats,
    pub complete: bool,
    pub improvements: u64,
}

struct LoweredProblem {
    solver: Solver,
    search_vars: Vec<VarId>,
    cost_vars: Vec<VarId>,
    nodes: Vec<i32>,
    depot_count: usize,
    /// Nearest-neighbour value-ordering seed for the successor variables (indexed
    /// by `VarId`); biases the first dive toward cheap arcs without changing the
    /// search space.
    initial_phase: Vec<Option<i32>>,
}

impl DirectEdgeMatrix {
    fn cost(&self, from: i32, to: i32) -> Option<i32> {
        let from = usize::try_from(from).ok()?;
        let to = usize::try_from(to).ok()?;
        let value = match self.orientation {
            MatrixOrientation::Direct => *self.values.get(from)?.get(to)?,
            MatrixOrientation::Reversed => *self.values.get(to)?.get(from)?,
        };
        i32::try_from(value).ok()
    }

    fn covers_i32_costs(&self, nodes: &[i32]) -> bool {
        nodes.iter().all(|&from| nodes.iter().all(|&to| self.cost(from, to).is_some()))
    }
}

impl RoutingSpec {
    fn from_model(model: &CollectionModel) -> Option<Self> {
        if model.schedule.is_some() || model.lists == 0 || model.objectives.len() != 1 {
            return None;
        }
        let node_count = model.lists.checked_add(model.items.len())?;
        if node_count > MAX_INTEGER_ROUTING_NODES {
            return None;
        }
        let tier = &model.objectives[0];
        if !tier.minimize || tier.has_max_terms() || !items_are_unique(&model.items) {
            return None;
        }

        // The objective tier is a mix of closed-edge terms (one per real route) and
        // reified `Sum`-scan terms (one per real route). Anything else is unsupported.
        let mut edge_terms = Vec::new();
        let mut scan_terms = Vec::new();
        for term in &tier.terms {
            match &term.iterable {
                Iterable::Edges { .. } => edge_terms.push(term),
                // A reified `Sum` scan, or a per-item `Sum` reward (`cp.sum(x, |i| ...)`)
                // that normalises to an identity-accumulator scan (see
                // `scan_routing_signature`), so its optionals get the dominance bound.
                Iterable::Scan { .. } => scan_terms.push(term),
                Iterable::Items(_) if matches!(term.op, ReduceOp::Sum) => scan_terms.push(term),
                _ => return None,
            }
        }

        // A list referenced by no objective term and no per-list constraint is a
        // free "pool". Exactly one pool => optional model with `k = lists - 1` real
        // routes; zero pools => today's exact non-optional path (every list served).
        let n_lists = model.lists;
        let mut referenced = vec![false; n_lists];
        for term in &tier.terms {
            let list = term.iterable.list();
            if list >= n_lists {
                return None;
            }
            referenced[list] = true;
        }
        for constraint in &model.constraints {
            let list = constraint.reduction.iterable.list();
            if list < n_lists {
                referenced[list] = true;
            }
        }
        let free: Vec<usize> = (0..n_lists).filter(|&list| !referenced[list]).collect();
        let pool = match free.len() {
            0 => false,
            1 => true,
            _ => return None,
        };
        let k = if pool { n_lists - 1 } else { n_lists };
        if k == 0 {
            return None;
        }

        // Homogeneous closed-edge matrix over the `k` real routes.
        let (depot, edge) = homogeneous_edge_objective(&edge_terms, k)?;
        if model.items.contains(&depot) {
            return None;
        }

        // Reified scan objective: `k` homogeneous copies, only for pool models (a
        // non-pool scan objective keeps its established enumeration path).
        let scan_objective = if scan_terms.is_empty() {
            None
        } else {
            if !pool || scan_terms.len() != k {
                return None;
            }
            Some(homogeneous_scan_terms(&scan_terms, &model.items)?)
        };

        // Side constraints. A pool is only lowerable alongside homogeneous scan
        // constraints (a capacity/length/item-sum/same-list route would wrongly cap
        // or count the free pool route), so those stay on enumeration.
        let (capacity, same_list, length, item_sum, scan) = if pool {
            if !model.globals.is_empty() {
                return None;
            }
            let pool_list = free[0];
            (None, Vec::new(), None, None, parse_pool_scan_constraints(model, pool_list)?)
        } else {
            let mut same_list = Vec::new();
            for global in &model.globals {
                match *global {
                    GlobalConstraint::SameList { a, b } => same_list.push((a, b)),
                    _ => return None,
                }
            }
            if same_list.iter().any(|&(a, b)| !model.items.contains(&a) || !model.items.contains(&b)) {
                return None;
            }
            let sides = parse_route_side_constraints(model)?;
            (sides.capacity, same_list, sides.length, sides.item_sum, sides.scan)
        };

        let depot_count = if pool { k + 1 } else { k };
        let mut nodes = Vec::with_capacity(depot_count + model.items.len());
        nodes.extend((0..depot_count).map(|_| depot));
        nodes.extend_from_slice(&model.items);
        edge.covers_i32_costs(&nodes).then_some(Self {
            depot_count,
            depot,
            items: model.items.clone(),
            edge,
            capacity,
            same_list,
            length,
            item_sum,
            scan,
            pool_list: pool.then(|| free[0]),
            scan_objective,
        })
    }

    fn pool(&self) -> bool {
        self.pool_list.is_some()
    }

    fn lower(&self) -> LoweredProblem {
        let mut solver = Solver::new();
        let mut nodes = Vec::with_capacity(self.depot_count + self.items.len());
        nodes.extend((0..self.depot_count).map(|_| self.depot));
        nodes.extend_from_slice(&self.items);
        let n = nodes.len();

        let succ = (0..n).map(|_| solver.new_var_range(0, n as i32 - 1)).collect::<Vec<_>>();
        // `circuit` auto-enables cut-set filtering only on sparse graphs; this
        // lowering builds the complete graph, so it stays on the base rules.
        graph::circuit(&mut solver, &succ);

        // Pool-membership channel. `in_pool[node]` is fixed 1 at the pool depot
        // (the last depot index), 0 at every real depot, and threaded to each
        // customer from its unique predecessor arc. `circuit` gives each customer
        // one predecessor, so the implication pins it exactly at a leaf.
        let in_pool = self.pool().then(|| self.build_pool_membership(&mut solver, &succ, n));

        let mut cost_vars = Vec::with_capacity(n);
        for (from_idx, &from_node) in nodes.iter().enumerate() {
            let row = nodes.iter().map(|&to_node| self.edge.cost(from_node, to_node).expect("validated edge cost")).collect::<Vec<_>>();
            match &in_pool {
                // Gate the edge: a pooled tail (customer or the pool depot itself)
                // contributes 0 on every one of its arcs, so the whole pool route
                // -- depot-to-item, item-to-item and item-to-depot -- is zero-cost.
                Some(in_pool) => {
                    let raw = solver.new_var_set(&row);
                    primitives::element_const(&mut solver, &row, succ[from_idx], raw);
                    let mut domain = row.clone();
                    if !domain.contains(&0) {
                        domain.push(0);
                    }
                    let cost = solver.new_var_set(&domain);
                    let gated = expr::ite(expr::eq(expr::var(in_pool[from_idx]), expr::int(1)), expr::int(0), expr::var(raw));
                    intension::intension(&mut solver, expr::eq(expr::var(cost), gated));
                    cost_vars.push(cost);
                }
                None => {
                    let cost = solver.new_var_set(&row);
                    primitives::element_const(&mut solver, &row, succ[from_idx], cost);
                    cost_vars.push(cost);
                }
            }
        }

        // Pool dominance: force-pool every optional that provably cannot beat pooling
        // (its minimum reward plus a lower bound on its insertion travel is >= 0), so
        // B&B never enumerates whether to serve it. Sound: serving such a customer can
        // only weakly worsen the objective (see `dominated_optionals`), so pooling it
        // preserves the optimum. The residual search is then the served-only routing.
        if let Some(in_pool) = &in_pool {
            for c in self.dominated_optionals(&nodes) {
                intension::intension(&mut solver, expr::eq(expr::var(in_pool[c]), expr::int(1)));
            }
            // Pool-route symmetry: its order is cost- and feasibility-irrelevant (every
            // pooled arc is gated to 0, the pool route is unbounded), so fix it to
            // increasing customer index -- a pooled customer never points back to a
            // lower-indexed customer. This removes the |pool|! equivalent orderings the
            // circuit would otherwise walk once the optionals are pooled.
            for c in self.depot_count..n {
                for d in self.depot_count..c {
                    let pooled = expr::eq(expr::var(in_pool[c]), expr::int(1));
                    intension::intension(&mut solver, expr::imp(pooled, expr::ne(expr::var(succ[c]), expr::int(d as i64))));
                }
            }
        }

        if let Some(capacity) = &self.capacity {
            let load = self.post_capacity_loads(&mut solver, &succ, capacity);
            post_route_capacity_check(&mut solver, &succ, self.depot_count, self.node_demands(capacity), capacity.capacity);
            debug_assert_eq!(load.len(), n);
        }

        for scan in &self.scan {
            self.post_scan_constraint(&mut solver, &succ, &nodes, scan, in_pool.as_deref());
        }

        // Reified scan objective: one gated per-customer contribution, summed into
        // the objective. A pooled customer contributes 0 (gated by `in_pool`), so
        // summing the ungated real customers equals the real routes' scan totals.
        if let Some(scan) = &self.scan_objective {
            let in_pool = in_pool.as_deref().expect("scan objective is only set for pool models");
            cost_vars.extend(self.post_scan_objective(&mut solver, &succ, &nodes, scan, in_pool));
        }

        // Depot symmetry orders only the `k` real depots; the pool depot (last
        // index) keeps its fixed, distinct, unconstrained role and is excluded, so
        // relabelling real depots never touches it. Non-pool orders all depots.
        let real_depots = if self.pool() { self.depot_count - 1 } else { self.depot_count };
        post_depot_symmetry(&mut solver, &succ, real_depots, self.depot_count, n);

        // Membership <-> route-order channel. Built only for same-list requirements
        // over >= 2 routes (one route makes `same_list` trivially true). The list
        // domains live in THIS solver so membership and successors share one trail
        // (R1). Order stays the source of truth: `segment_of`/membership are derived
        // views, never branched (search stays on succ + cost), and the decoded
        // solution is read from `succ`, not from the membership vars.
        if !self.same_list.is_empty() || self.length.is_some() || self.item_sum.is_some() {
            let lists: Vec<ListId> = (0..self.depot_count).map(|_| solver.store.new_list(self.items.clone())).collect();
            for &list in &lists {
                list_constraints::list_cardinality(&mut solver, list);
            }
            list_constraints::partition(&mut solver, &lists, &self.items);
            for &(a, b) in &self.same_list {
                list_constraints::same_list(&mut solver, &lists, a, b);
            }
            // Homogeneous per-route length applies the same bound to every route.
            if let Some((min, max)) = self.length {
                for &list in &lists {
                    list_constraints::list_len(&mut solver, list, min, max);
                }
            }
            // Homogeneous per-route weighted item sum applies to every route.
            if let Some((weights, min, max)) = &self.item_sum {
                for &list in &lists {
                    list_constraints::list_item_sum(&mut solver, list, weights.clone(), *min, *max);
                }
            }
            let segment_of: Vec<VarId> = (0..self.items.len()).map(|_| solver.new_var_range(0, self.depot_count as i32 - 1)).collect();
            solver.post(Box::new(RouteSegmentChannel {
                succ: succ.clone(),
                segment_of,
                lists,
                items: self.items.clone(),
                depot_count: self.depot_count,
                two_way: std::env::var("QAYD_ROUTING_TWOWAY").as_deref() != Ok("0"),
            }));
        }

        // Nearest-neighbour seed: each node's successor defaults to its cheapest
        // outgoing arc. A value-ordering hint only -- circuit/symmetry refine it,
        // and the saved phase takes over after the first dive. `QAYD_ROUTING_NN=0`
        // disables it (ablation).
        let mut initial_phase = vec![None; solver.store.num_vars()];
        if std::env::var("QAYD_ROUTING_NN").as_deref() != Ok("0") {
            for from in 0..n {
                if let Some(to) =
                    (0..n).filter(|&to| to != from).min_by_key(|&to| self.edge.cost(nodes[from], nodes[to]).expect("validated edge cost"))
                {
                    initial_phase[succ[from].index()] = Some(to as i32);
                }
            }
        }

        // Branch only the successor variables; per-node costs are functionally
        // determined from them by `element_const`, and branching them (they carry
        // the objective impact) would bypass the nearest-neighbour value ordering.
        // in_pool is a derived channel (dominated optionals are fixed above, the rest
        // follow from the fixed predecessor arc), never branched.
        let search_vars = succ;
        LoweredProblem { solver, search_vars, cost_vars, nodes, depot_count: self.depot_count, initial_phase }
    }

    /// Successor values for a local-search incumbent: chain each LS route
    /// `depot_i -> route_i -> depot_{i+1}` into one Hamiltonian cycle. Routes are
    /// sorted by their first customer so the tour matches the depot symmetry
    /// breaking (heads non-decreasing). Returns `None` if the incumbent does not
    /// decode to a full tour over `n` nodes.
    fn ls_tour_succ(&self, n: usize, ls_lists: &[Vec<i32>]) -> Option<Vec<usize>> {
        if ls_lists.len() != self.depot_count {
            return None;
        }
        let item_node = |item: i32| self.items.iter().position(|&x| x == item).map(|i| self.depot_count + i);
        let sort_head = |routes: &mut Vec<&Vec<i32>>| {
            routes.sort_by_key(|route| route.first().and_then(|&item| item_node(item)).unwrap_or(usize::MAX));
        };
        let routes: Vec<&Vec<i32>> = match self.pool_list {
            // Real routes fill depots `0..k-1` sorted by first customer (matching
            // post_depot_symmetry, which orders only the real depots); the pool list
            // fills the pool depot `k` (last), which that ordering excludes. The
            // `(depot + 1) % depot_count` closing then chains `k -> pool -> 0`.
            Some(pl) => {
                let mut real: Vec<&Vec<i32>> = ls_lists.iter().enumerate().filter(|&(i, _)| i != pl).map(|(_, r)| r).collect();
                sort_head(&mut real);
                real.push(&ls_lists[pl]);
                real
            }
            None => {
                let mut routes: Vec<&Vec<i32>> = ls_lists.iter().collect();
                sort_head(&mut routes);
                routes
            }
        };

        let mut succ_val: Vec<Option<usize>> = vec![None; n];
        for (depot, route) in routes.iter().enumerate() {
            let mut cur = depot;
            for &item in route.iter() {
                let node = item_node(item)?;
                succ_val[cur] = Some(node);
                cur = node;
            }
            succ_val[cur] = Some((depot + 1) % self.depot_count);
        }
        succ_val.into_iter().collect()
    }

    fn node_demands(&self, capacity: &CapacitySpec) -> Vec<i32> {
        let mut demands = vec![0; self.depot_count];
        demands.extend_from_slice(&capacity.demands);
        demands
    }

    fn post_capacity_loads(&self, solver: &mut Solver, succ: &[VarId], capacity: &CapacitySpec) -> Vec<VarId> {
        let mut load = Vec::with_capacity(self.depot_count + self.items.len());
        for _ in 0..self.depot_count {
            load.push(solver.new_var_set(&[0]));
        }
        for &demand in &capacity.demands {
            load.push(solver.new_var_range(demand, capacity.capacity));
        }

        for from in 0..succ.len() {
            for (item_offset, &demand) in capacity.demands.iter().enumerate() {
                let to = self.depot_count + item_offset;
                if from == to {
                    continue;
                }
                let arc_selected = expr::eq(expr::var(succ[from]), expr::int(to as i64));
                let required = expr::add(vec![expr::var(load[from]), expr::int(i64::from(demand))]);
                let enough_load = expr::ge(expr::var(load[to]), required);
                intension::intension(solver, expr::imp(arc_selected, enough_load));
            }
        }
        load
    }

    /// Resource-extension encoding for a homogeneous `Sum` scan. Threads a
    /// per-customer accumulator and a per-route cumulative sum along the chosen
    /// arcs, then bounds each route's total against the constraint's rhs.
    ///
    /// Soundness rests on `circuit` making each customer's predecessor unique, so
    /// the `==` step/emit implications pin every accumulator exactly (unlike the
    /// capacity `>=` relaxation). The cumulative sum is kept per route -- reset to
    /// `0` at every depot -- because the list-local-search spec applies the scan
    /// bound to each route independently, not to the fleet total.
    fn post_scan_constraint(&self, solver: &mut Solver, succ: &[VarId], nodes: &[i32], scan: &ScanRoutingSpec, in_pool: Option<&[VarId]>) {
        let n = nodes.len();
        let depot_count = self.depot_count;
        let arena = &scan.arena.exprs;
        // `cv[node]` = the accumulator's view of a node: `boundary` at a depot, else
        // the item value. `A(from)` = the accumulator entering an arc's tail.
        let cv = |node: usize| if node < depot_count { i64::from(scan.boundary) } else { i64::from(nodes[node]) };

        let mut acc = vec![None; n];
        let mut route_sum = vec![None; n];
        for (node, slot) in acc.iter_mut().enumerate().take(n).skip(depot_count) {
            *slot = Some(solver.new_var_range(scan.acc_dom.0 as i32, scan.acc_dom.1 as i32));
            route_sum[node] = Some(solver.new_var_range(scan.total_dom.0 as i32, scan.total_dom.1 as i32));
        }
        let a_of = |from: usize| if from < depot_count { expr::int(scan.init) } else { expr::var(acc[from].expect("customer accumulator")) };
        let sum_in = |from: usize| if from < depot_count { expr::int(0) } else { expr::var(route_sum[from].expect("customer route sum")) };

        for (from, &from_var) in succ.iter().enumerate() {
            for to in depot_count..n {
                if from == to {
                    continue;
                }
                let cur = i64::from(nodes[to]);
                let prev = cv(from);
                let arc = expr::eq(expr::var(from_var), expr::int(to as i64));
                let acc_to = acc[to].expect("customer accumulator");
                // acc[to] == step(cur, A(from), prev).
                let step_rhs = lower_scan(arena, scan.step, cur, prev, &a_of(from));
                intension::intension(solver, expr::imp(arc.clone(), expr::eq(expr::var(acc_to), step_rhs)));
                // route_sum[to] == sum_in(from) + coeff * emit(cur, acc[to], prev).
                let emit_rhs = lower_scan(arena, scan.emit, cur, prev, &expr::var(acc_to));
                let contribution = expr::mul(vec![expr::int(scan.coeff), emit_rhs]);
                let extended = expr::add(vec![sum_in(from), contribution]);
                intension::intension(solver, expr::imp(arc, expr::eq(expr::var(route_sum[to].expect("customer route sum")), extended)));
            }
        }

        // Per-route bound: when an arc closes a route (points back at a depot), the
        // route total is `sum_in(from)` -- `0` for an empty route -- and must obey
        // the constraint's op against rhs.
        for (from, &from_var) in succ.iter().enumerate() {
            for to in 0..depot_count {
                if from == to {
                    continue;
                }
                let arc = expr::eq(expr::var(from_var), expr::int(to as i64));
                let total = sum_in(from);
                let bounded = match scan.op {
                    Op::Le => expr::le(total, expr::int(scan.rhs)),
                    Op::Ge => expr::ge(total, expr::int(scan.rhs)),
                    Op::Eq => expr::eq(total, expr::int(scan.rhs)),
                };
                // The pool route is unbounded: a closing arc whose tail is pooled
                // (`in_pool[from] == 1`) closes the pool route, so its total is not
                // held against rhs. Real routes (tail not pooled) are bounded.
                let guarded = match in_pool {
                    Some(in_pool) => expr::imp(expr::eq(expr::var(in_pool[from]), expr::int(0)), bounded),
                    None => bounded,
                };
                intension::intension(solver, expr::imp(arc, guarded));
            }
        }
    }

    /// Build the pool-membership vars: `1` fixed at the pool depot (last depot
    /// index), `0` fixed at every real depot, `[0, 1]` at customers, threaded from
    /// each customer's unique predecessor arc `succ[from] == to => in_pool[to] ==
    /// in_pool[from]`. Pinned exactly at a leaf because `circuit` makes the
    /// predecessor unique; partial states leave a customer's membership free.
    fn build_pool_membership(&self, solver: &mut Solver, succ: &[VarId], n: usize) -> Vec<VarId> {
        let pool_depot = self.depot_count - 1;
        let in_pool: Vec<VarId> = (0..n)
            .map(|node| {
                if node == pool_depot {
                    solver.new_var_set(&[1])
                } else if node < self.depot_count {
                    solver.new_var_set(&[0])
                } else {
                    solver.new_var_range(0, 1)
                }
            })
            .collect();
        for from in 0..n {
            for to in self.depot_count..n {
                if from == to {
                    continue;
                }
                let arc = expr::eq(expr::var(succ[from]), expr::int(to as i64));
                intension::intension(solver, expr::imp(arc, expr::eq(expr::var(in_pool[to]), expr::var(in_pool[from]))));
            }
        }
        in_pool
    }

    /// Static `[min, max]` of customer `c`'s reified objective contribution
    /// (`coeff * emit`): `emit` at this fixed `cur`, bounded over every predecessor
    /// value and the accumulator's domain. A sentinel var stands in for the
    /// accumulator so it can be bounded over `acc_dom`.
    fn served_reward_bounds(&self, nodes: &[i32], scan: &ScanRoutingSpec, c: usize) -> (i64, i64) {
        const ACC_SENTINEL: VarId = VarId(u32::MAX);
        let arena = &scan.arena.exprs;
        let cur = i64::from(nodes[c]);
        let cv = |node: usize| if node < self.depot_count { i64::from(scan.boundary) } else { i64::from(nodes[node]) };
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for from in 0..nodes.len() {
            if from == c {
                continue;
            }
            let e = lower_scan(arena, scan.emit, cur, cv(from), &expr::var(ACC_SENTINEL));
            let (l, h) = e.bounds(&|v| if v == ACC_SENTINEL { scan.acc_dom } else { (0, 0) });
            let (cl, ch) = (scan.coeff.saturating_mul(l), scan.coeff.saturating_mul(h));
            lo = lo.min(cl.min(ch));
            hi = hi.max(cl.max(ch));
        }
        if lo > hi {
            (0, 0)
        } else {
            (lo, hi)
        }
    }

    /// Whether the objective scan emits a pure per-item value (depends only on the
    /// current node, `Arg(0)`, not the accumulator `Arg(1)` or predecessor `Arg(2)`).
    /// Only then is a served customer's contribution independent of the others, so
    /// serving/pooling it cannot ripple through the route -- the precondition for the
    /// pool dominance. `None` (edges only) is trivially per-item (zero contribution).
    fn scan_objective_is_per_item(&self) -> bool {
        let Some(scan) = &self.scan_objective else {
            return true;
        };
        let arena = &scan.arena.exprs;
        let mut stack = vec![scan.emit];
        while let Some(id) = stack.pop() {
            match &arena[id.0 as usize] {
                Expr::Arg(k) if *k >= 1 => return false,
                Expr::Arg(_) | Expr::Const(_) => {}
                Expr::Array(_, i) => stack.push(*i),
                Expr::Matrix(_, i, j) => {
                    stack.push(*i);
                    stack.push(*j);
                }
                Expr::Add(a, b)
                | Expr::Sub(a, b)
                | Expr::Mul(a, b)
                | Expr::Min(a, b)
                | Expr::Max(a, b)
                | Expr::Div(a, b)
                | Expr::Lt(a, b)
                | Expr::Le(a, b)
                | Expr::Eq(a, b)
                | Expr::Ne(a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                Expr::Abs(a) => stack.push(*a),
                Expr::IfThenElse(c, a, b) => {
                    stack.push(*c);
                    stack.push(*a);
                    stack.push(*b);
                }
            }
        }
        true
    }

    /// Optionals that cannot beat pooling: serving customer `c` adds a lower bound
    /// `marginal` on its insertion travel (min over predecessor/successor pairs of
    /// `edge(p,c) + edge(c,s) - edge(p,s)`, a true lower bound on the extra travel)
    /// and at best its minimum reward. When `reward_min + marginal >= 0`, serving `c`
    /// never strictly improves the objective, so pooling dominates and `c` is fixed
    /// pooled. Sound only with no per-route scan constraint (which could REQUIRE `c`)
    /// and a per-item reward (so serving `c` does not ripple onto other customers);
    /// on a non-metric matrix `marginal` may be negative and the rule simply does not
    /// fire. Returns the customer node indices to force `in_pool = 1`.
    fn dominated_optionals(&self, nodes: &[i32]) -> Vec<usize> {
        if !self.pool() || !self.scan.is_empty() || !self.scan_objective_is_per_item() {
            return Vec::new();
        }
        let n = nodes.len();
        let pool_depot = self.depot_count - 1;
        let servable: Vec<usize> = (0..n).filter(|&i| i != pool_depot).collect();
        let mut pooled = Vec::new();
        for c in self.depot_count..n {
            let reward_min = self.scan_objective.as_ref().map_or(0, |scan| self.served_reward_bounds(nodes, scan, c).0);
            let mut marginal = i64::MAX;
            for &p in &servable {
                if p == c {
                    continue;
                }
                for &s in &servable {
                    if s == c {
                        continue;
                    }
                    let edge = |a: i32, b: i32| i64::from(self.edge.cost(a, b).expect("validated edge cost"));
                    let m = edge(nodes[p], nodes[c]) + edge(nodes[c], nodes[s]) - edge(nodes[p], nodes[s]);
                    marginal = marginal.min(m);
                }
            }
            if reward_min.saturating_add(marginal) >= 0 {
                pooled.push(c);
            }
        }
        pooled
    }

    /// Lower a reified `Sum`-scan objective into per-customer contribution vars.
    /// Threads its own accumulator (same pinning discipline as
    /// [`Self::post_scan_constraint`]), pins each customer's `coeff * emit` by its
    /// incoming arc, then gates it by `in_pool` so a pooled customer contributes 0.
    /// Returns the contribution vars to append to the objective's cost vars.
    fn post_scan_objective(&self, solver: &mut Solver, succ: &[VarId], nodes: &[i32], scan: &ScanRoutingSpec, in_pool: &[VarId]) -> Vec<VarId> {
        let n = nodes.len();
        let depot_count = self.depot_count;
        let arena = &scan.arena.exprs;
        let cv = |node: usize| if node < depot_count { i64::from(scan.boundary) } else { i64::from(nodes[node]) };

        let mut acc = vec![None; n];
        for slot in acc.iter_mut().take(n).skip(depot_count) {
            *slot = Some(solver.new_var_range(scan.acc_dom.0 as i32, scan.acc_dom.1 as i32));
        }
        let a_of = |from: usize| if from < depot_count { expr::int(scan.init) } else { expr::var(acc[from].expect("customer accumulator")) };

        for (from, &from_var) in succ.iter().enumerate() {
            for to in depot_count..n {
                if from == to {
                    continue;
                }
                let cur = i64::from(nodes[to]);
                let prev = cv(from);
                let arc = expr::eq(expr::var(from_var), expr::int(to as i64));
                let step_rhs = lower_scan(arena, scan.step, cur, prev, &a_of(from));
                intension::intension(solver, expr::imp(arc, expr::eq(expr::var(acc[to].expect("customer accumulator")), step_rhs)));
            }
        }

        // Per-customer `coeff * emit` at the chosen predecessor, gated to 0 when pooled.
        let mut contrib = Vec::with_capacity(n - depot_count);
        for c in depot_count..n {
            let cur = i64::from(nodes[c]);
            // Tight per-customer contribution bounds. For an order-independent per-item
            // reward these collapse to a single constant, so the contribution var stays
            // 2-valued ({0, reward}) rather than a fleet-wide range -- keeping the
            // objective-bound propagation cost independent of the reward magnitude (a
            // wide range makes it O(reward)).
            let (lo, hi) = self.served_reward_bounds(nodes, scan, c);
            let emit_var = if lo == hi { solver.new_var_set(&[lo as i32]) } else { solver.new_var_range(lo as i32, hi as i32) };
            let acc_c = expr::var(acc[c].expect("customer accumulator"));
            for (from, &from_var) in succ.iter().enumerate() {
                if from == c {
                    continue;
                }
                let prev = cv(from);
                let arc = expr::eq(expr::var(from_var), expr::int(c as i64));
                let emit_rhs = lower_scan(arena, scan.emit, cur, prev, &acc_c);
                let scaled = expr::mul(vec![expr::int(scan.coeff), emit_rhs]);
                intension::intension(solver, expr::imp(arc, expr::eq(expr::var(emit_var), scaled)));
            }
            // The gated contribution is `emit_var` (served) or 0 (pooled). Keep it a
            // small set when the reward is a per-item constant, else the interval.
            let obj = if lo == hi {
                let vals: Vec<i32> = if lo == 0 { vec![0] } else { vec![0, lo as i32] };
                solver.new_var_set(&vals)
            } else {
                solver.new_var_range(lo.min(0) as i32, hi.max(0) as i32)
            };
            let gated = expr::ite(expr::eq(expr::var(in_pool[c]), expr::int(1)), expr::int(0), expr::var(emit_var));
            intension::intension(solver, expr::eq(expr::var(obj), gated));
            contrib.push(obj);
        }
        contrib
    }
}

/// Solve a supported collection routing model by lowering it to integer CP.
/// Returns `None` when the model shape is outside this exact lowering.
pub(crate) fn solve_collection(
    model: &CollectionModel,
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> Option<RoutingIntegerOutcome> {
    let spec = RoutingSpec::from_model(model)?;
    // Warm start: a quick local-search incumbent seeds the first dive and, after
    // replay verification, supplies an initial exact objective bound.
    // `QAYD_ROUTING_WARMSTART=0` disables it (ablation). Pool models are included:
    // `ls_tour_succ` chains the pool list through the fixed pool depot, and replay
    // verifies the decode -- an infeasible one degrades gracefully to bound-less.
    let warm = if std::env::var("QAYD_ROUTING_WARMSTART").as_deref() != Ok("0") {
        // A bounded local-search pass: enough to find a good incumbent, capped so
        // it terminates even without a stop flag and leaves the exact search the
        // bulk of the time.
        let mut ignore_warm_start_progress = |_| {};
        Some(ls_lists::solve_collection_capped(model, seed, stop, WARM_START_ITERS, None, &mut ignore_warm_start_progress))
    } else {
        None
    };
    // Verify/replay the LS incumbent on a FRESH lowering (so the exact search's
    // solver is untouched): reconstruct its tour, confirm it is feasible, and read
    // its exact cost. This yields a verified fallback solution AND an initial
    // objective bound -- so the exact prunes anything no better than `cost` from
    // the start, and if it proves nothing strictly better, we still return a real
    // (verified) solution at that cost.
    let verified: Option<(Vec<Vec<i32>>, i64)> = warm.as_ref().filter(|ls| ls.feasible).and_then(|ls| {
        let n = spec.depot_count + spec.items.len();
        let tour = spec.ls_tour_succ(n, &ls.lists)?;
        let mut check = spec.lower();
        let cost = replay_tour(&mut check.solver, &check.search_vars[..n], &check.cost_vars, &tour)?;
        let tour_i32: Vec<i32> = tour.iter().map(|&node| node as i32).collect();
        let routes = routes_from_successors(&check.nodes, check.depot_count, &tour_i32)?;
        Some((routes, cost))
    });
    let bound = verified.as_ref().map(|&(_, cost)| AtomicI64::new(cost));

    let lowered = spec.lower();
    let LoweredProblem { mut solver, search_vars, cost_vars, nodes, depot_count, initial_phase } = lowered;

    let coeffs = vec![1i64; cost_vars.len()];
    let mut improvements = 0u64;
    let (best, stats, complete) = search::optimize_seeded(
        &mut solver,
        &search_vars,
        SearchObjective::Linear { coeffs: &coeffs, vars: &cost_vars },
        true,
        stop,
        seed,
        bound.as_ref(),
        None,
        &[],
        None,
        initial_phase,
        Vec::new(),
        |value, _assignment| {
            improvements += 1;
            report(value);
        },
    );

    // The exact found a solution strictly better than the bound -> use it. Else fall
    // back to the verified LS incumbent (proven optimal when the search completed).
    let (lists, objective, feasible) = match best {
        Some((assignment, value)) => {
            // The circuit + capacity propagators guarantee a single Hamiltonian
            // cycle decoding into exactly `depot_count` routes; a failure here is a
            // logic bug, not an unsupported model -- panic rather than fall back.
            // A pool model decodes the pool depot's segment as the last list.
            let routes = if spec.pool() {
                decode_pool_routes(&nodes, depot_count, &assignment[..nodes.len()])
            } else {
                routes_from_successors(&nodes, depot_count, &assignment[..nodes.len()]).expect("solved circuit decodes into k routes")
            };
            (routes, Some(value), true)
        }
        None => match verified {
            Some((routes, cost)) => (routes, Some(cost), true),
            None => (vec![Vec::new(); depot_count], None, false),
        },
    };
    Some(RoutingIntegerOutcome {
        solution: CollectionSolution {
            lists,
            objectives: objective.map(|value| vec![value]).unwrap_or_default(),
            feasible,
            starts: Vec::new(),
            machines: Vec::new(),
        },
        stats,
        complete,
        improvements,
    })
}

/// Replay a successor tour in the solver to verify feasibility and read its exact
/// objective. Pushes a temporary level, fixes every successor, propagates, then
/// pops back to the root. Returns the verified cost, or `None` if infeasible.
fn replay_tour(solver: &mut Solver, succ_vars: &[VarId], cost_vars: &[VarId], tour: &[usize]) -> Option<i64> {
    solver.store.push_level();
    let mut feasible = true;
    for (&var, &value) in succ_vars.iter().zip(tour) {
        if solver.store.fix(var, value as i32).is_err() {
            feasible = false;
            break;
        }
    }
    let cost = (feasible && solver.propagate().is_ok()).then(|| cost_vars.iter().map(|&c| i64::from(solver.store.value(c))).sum());
    solver.store.pop_level();
    cost
}

#[derive(Clone)]
struct RouteCapacityCheck {
    succ: Vec<VarId>,
    depot_count: usize,
    demands: Vec<i32>,
    capacity: i32,
}

impl Propagator for RouteCapacityCheck {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &var in &self.succ {
            store.subscribe(var, me, Event::Fix);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        if self.succ.iter().any(|&var| !store.is_fixed(var)) {
            return Ok(());
        }
        let succ = self.succ.iter().map(|&var| store.value(var)).collect::<Vec<_>>();
        match route_capacity_check(self.depot_count, &self.demands, self.capacity, &succ) {
            CapacityCheck::Ok => Ok(()),
            CapacityCheck::Overflow(path) => {
                // The fixed arcs of the overflowing segment force these customers
                // into one route whose total demand exceeds capacity.
                let why = path.windows(2).map(|w| Premise::Eq { var: self.succ[w[0]], val: w[1] as i32 }).collect();
                Err(store.fail_because(why))
            }
            // Structural failure: `circuit` will (or did) reject it; plain conflict.
            CapacityCheck::Malformed => Err(Inconsistency),
        }
    }
}

fn post_route_capacity_check(solver: &mut Solver, succ: &[VarId], depot_count: usize, demands: Vec<i32>, capacity: i32) {
    solver.post(Box::new(RouteCapacityCheck { succ: succ.to_vec(), depot_count, demands, capacity }));
}

/// Channels between the successor circuit and list membership. Fixed successor
/// prefixes derive, for each customer, which depot segment it sits in
/// (`segment_of`), then require the matching list membership so the explained
/// list propagators (`partition`, `same_list`) apply. Conversely, when a list
/// propagator forbids a customer from a segment, the channel removes that segment
/// view and forbids arcs from nodes already known to be in that segment to the
/// forbidden customer.
///
/// `succ` remains the source of truth and the only search surface here.
/// `segment_of`/membership are propagated views, never branched. When every
/// successor is fixed, the leaf backstop must have placed every customer on
/// exactly one segment; a clash with `same_list` surfaces as the
/// `require`/`partition` conflict.
#[derive(Clone)]
struct RouteSegmentChannel {
    succ: Vec<VarId>,
    /// One length-`items.len()` view: `segment_of[j]` is the depot segment of `items[j]`.
    segment_of: Vec<VarId>,
    /// `lists[r]` is the list domain for depot copy `r`.
    lists: Vec<ListId>,
    items: Vec<i32>,
    depot_count: usize,
    /// Whether the reverse `member -> segment_of -> succ` pruning is enabled
    /// (`QAYD_ROUTING_TWOWAY=0` disables it, for ablation/benchmarking).
    two_way: bool,
}

impl RouteSegmentChannel {
    /// Customer `j` is proven to sit in depot segment `r`: fix its derived view and
    /// require the matching membership (partition forbids the other lists). The
    /// reason is the fixed arcs of the path `depot r -> ... -> customer`, so a clash
    /// with `same_list`/`list_len`/`item_sum` (via the membership) learns a clause
    /// over those arcs rather than a whole-scope snapshot.
    fn assign_segment(&self, store: &mut Store, j: usize, r: usize, reason: &[Premise]) -> Result<(), Inconsistency> {
        store.fix_because(self.segment_of[j], r as i32, reason.to_vec())?;
        store.require_list_item_because(self.lists[r], self.items[j], reason.to_vec())?;
        Ok(())
    }
}

impl Propagator for RouteSegmentChannel {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &var in &self.succ {
            store.subscribe(var, me, Event::Fix);
        }
        // Subscribe to the derived views too: this marks them relevant (so the LCG
        // gives them literals when we fix/cite them), without putting them in the
        // branched `search_vars`. They are derived from `succ`, never branched.
        for &var in &self.segment_of {
            store.subscribe(var, me, Event::Fix);
        }
        // Reverse channel: wake when membership is decided (e.g. same_list/list_len/
        // item_sum forbids a customer from a route), so we can prune `succ`.
        if self.two_way {
            for r in 0..self.depot_count {
                for &item in &self.items {
                    if let Some(member) = store.list_member_var(self.lists[r], item) {
                        store.subscribe(member, me, Event::Fix);
                    }
                }
            }
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let n = self.succ.len();
        let all_fixed = self.succ.iter().all(|&var| store.is_fixed(var));
        let mut assigned = vec![false; self.items.len()];
        // Walk each depot copy's fixed successor prefix; customers on it are proven
        // to be in that depot's segment.
        for r in 0..self.depot_count {
            let mut cur = r;
            let mut steps = 0usize;
            // Accumulate the fixed arcs walked from depot `r`; they are the reason
            // every customer reached on this prefix sits in segment `r`.
            let mut reason: Vec<Premise> = Vec::new();
            while store.is_fixed(self.succ[cur]) {
                steps += 1;
                if steps > n {
                    break; // malformed partial state; let `circuit` narrow first
                }
                let next = store.value(self.succ[cur]) as usize;
                reason.push(Premise::Eq { var: self.succ[cur], val: next as i32 });
                if next >= n || next < self.depot_count {
                    break; // segment ends at the next depot copy (or out of range)
                }
                let j = next - self.depot_count;
                self.assign_segment(store, j, r, &reason)?;
                assigned[j] = true;
                cur = next;
            }
        }
        // Reverse channel: membership -> segment_of -> succ. When a customer is
        // forbidden from route `r`, it cannot be in `r`'s segment, and no node known
        // to be in route `r` may point to it.
        for j in 0..self.items.len() {
            if !self.two_way {
                break;
            }
            let nc = (self.depot_count + j) as i32;
            for r in 0..self.depot_count {
                if store.list_possible(self.lists[r], self.items[j]) {
                    continue;
                }
                // member[r][customer] = 0.
                let member0: Vec<Premise> =
                    store.list_member_var(self.lists[r], self.items[j]).map(|var| Premise::Eq { var, val: 0 }).into_iter().collect();
                // membership -> segment_of: the customer is not in segment `r`.
                store.remove_because(self.segment_of[j], r as i32, member0.clone())?;
                // segment_of -> succ: depot `r` (definitely route `r`) cannot point to it.
                store.remove_because(self.succ[r], nc, member0.clone())?;
                // ...nor any customer already proven to be in route `r`.
                for x in self.depot_count..n {
                    let xj = x - self.depot_count;
                    if store.is_fixed(self.segment_of[xj]) && store.value(self.segment_of[xj]) == r as i32 {
                        let mut why = member0.clone();
                        why.push(Premise::Eq { var: self.segment_of[xj], val: r as i32 });
                        store.remove_because(self.succ[x], nc, why)?;
                    }
                }
            }
        }

        // Leaf backstop: with all successors fixed, every customer must lie on
        // exactly one segment, so all `segment_of`/membership are now determined.
        if all_fixed && assigned.iter().any(|&seen| !seen) {
            return Err(Inconsistency);
        }
        Ok(())
    }
}

/// Break the `k!` symmetry between the interchangeable depot copies. Order routes
/// by their first customer: `head[i]` is the (customer) node that depot copy `i`
/// points to, or the sentinel `n` when the route is empty, and the heads are
/// non-decreasing. Sound: relabelling depot copies to sort their heads always
/// yields an equivalent solution, and empty routes tie at the sentinel rather
/// than at a depot index (so the degenerate all-empty cycle is not over-pruned).
fn post_depot_symmetry(solver: &mut Solver, succ: &[VarId], order_count: usize, depot_threshold: usize, n: usize) {
    if order_count < 2 {
        return;
    }
    let sentinel = n as i64;
    let head: Vec<VarId> = succ[..order_count]
        .iter()
        .map(|&s| {
            let h = solver.new_var_range(depot_threshold as i32, n as i32);
            // head = (succ < depot_threshold) ? sentinel : succ   (empty route -> sentinel).
            // `depot_threshold` is the internal depot count, so a real depot pointing at
            // any depot (incl. the pool depot) still reads as an empty real route.
            let def = expr::ite(expr::lt(expr::var(s), expr::int(depot_threshold as i64)), expr::int(sentinel), expr::var(s));
            intension::intension(solver, expr::eq(expr::var(h), def));
            h
        })
        .collect();
    for pair in head.windows(2) {
        intension::intension(solver, expr::le(expr::var(pair[0]), expr::var(pair[1])));
    }
}

/// Outcome of the leaf capacity check.
enum CapacityCheck {
    /// Every route is within capacity.
    Ok,
    /// A route's demand exceeds capacity; the node path `depot -> ... -> the
    /// customer at which the load first overflowed`.
    Overflow(Vec<usize>),
    /// Structural problem (not a circuit at a valid leaf); `circuit` owns these.
    Malformed,
}

fn route_capacity_check(depot_count: usize, demands: &[i32], capacity: i32, succ: &[i32]) -> CapacityCheck {
    if depot_count == 0 || demands.len() != succ.len() {
        return CapacityCheck::Malformed;
    }
    let cap = i64::from(capacity);
    let mut seen = vec![false; succ.len()];
    seen[0] = true;
    let mut routes = 0usize;
    let mut load = 0i64;
    let mut cur = 0usize;
    // Node path of the route currently being walked, starting at its depot copy.
    let mut path = vec![0usize];
    loop {
        let Ok(next) = usize::try_from(succ[cur]) else {
            return CapacityCheck::Malformed;
        };
        if next >= succ.len() {
            return CapacityCheck::Malformed;
        }
        if next == 0 {
            // Closing the cycle: the last route's load is already within capacity
            // (overflow is caught at the customer below), so this is structural.
            routes += 1;
            return if load <= cap && routes == depot_count && seen.iter().all(|&x| x) {
                CapacityCheck::Ok
            } else {
                CapacityCheck::Malformed
            };
        }
        if seen[next] {
            return CapacityCheck::Malformed;
        }
        if next < depot_count {
            // Route boundary: the finished route is within capacity; start a fresh one.
            routes += 1;
            seen[next] = true;
            load = 0;
            path = vec![next];
        } else {
            load += i64::from(demands[next]);
            path.push(next);
            if load > cap {
                return CapacityCheck::Overflow(path);
            }
            seen[next] = true;
        }
        cur = next;
    }
}

fn routes_from_successors(nodes: &[i32], depot_count: usize, succ: &[i32]) -> Option<Vec<Vec<i32>>> {
    if depot_count == 0 || nodes.len() != succ.len() || depot_count > nodes.len() {
        return None;
    }
    let mut seen = vec![false; nodes.len()];
    seen[0] = true;
    let mut routes = Vec::with_capacity(depot_count);
    let mut route = Vec::new();
    let mut cur = 0usize;
    loop {
        let next = usize::try_from(succ[cur]).ok()?;
        if next >= nodes.len() {
            return None;
        }
        if next == 0 {
            routes.push(route);
            break;
        }
        if seen[next] {
            return None;
        }
        seen[next] = true;
        if next < depot_count {
            routes.push(route);
            route = Vec::new();
        } else {
            route.push(nodes[next]);
        }
        cur = next;
    }
    (routes.len() == depot_count && seen.iter().all(|&x| x)).then_some(routes)
}

/// A homogeneous closed-edge objective over exactly `k` real routes: every term
/// is a distinct list carrying the identical depot and distance matrix. Unlike
/// [`parse_route_edge_objective`] the lists need not be `0..k` (a pool model's
/// real lists are the objective-referenced subset), so distinctness is checked
/// against the seen set rather than an index array.
fn homogeneous_edge_objective(edge_terms: &[&Reduction], k: usize) -> Option<(i32, DirectEdgeMatrix)> {
    if edge_terms.len() != k {
        return None;
    }
    let mut seen: Vec<usize> = Vec::with_capacity(k);
    let mut base: Option<(i32, DirectEdgeMatrix)> = None;
    for term in edge_terms {
        let (list, depot, edge) = direct_closed_edge_term(term)?;
        if seen.contains(&list) {
            return None;
        }
        seen.push(list);
        match &base {
            None => base = Some((depot, edge)),
            Some((base_depot, base_edge)) => {
                if depot != *base_depot || !base_edge.same_matrix_as(&edge) {
                    return None;
                }
            }
        }
    }
    base
}

/// A homogeneous reified `Sum`-scan objective over the real routes: every term is
/// a distinct list with the identical lowered signature. The `op`/`rhs` are
/// synthesised (unused when scored) so the shared signature parser can validate
/// the accumulator domains and reject out-of-band or accumulator-in-divisor scans.
fn homogeneous_scan_terms(scan_terms: &[&Reduction], items: &[i32]) -> Option<ScanRoutingSpec> {
    let mut seen: Vec<usize> = Vec::with_capacity(scan_terms.len());
    let mut base: Option<ScanRoutingSpec> = None;
    for term in scan_terms {
        let list = term.iterable.list();
        if seen.contains(&list) {
            return None;
        }
        seen.push(list);
        let synth = Constraint { reduction: (*term).clone(), op: Op::Le, rhs: 0 };
        // The integer lowering does not thread a closing edge, so an `end`-scan is
        // not integer-lowerable (it falls to exact enumeration instead).
        let spec = scan_routing_signature(&synth, items).filter(|s| s.end.is_none())?;
        match &base {
            None => base = Some(spec),
            Some(base_spec) => {
                if &spec != base_spec {
                    return None;
                }
            }
        }
    }
    base
}

/// The pool model's side constraints: only homogeneous `Sum`-scan constraints
/// over the real routes are lowerable (the pool list carries none). Anything else
/// (capacity, length, item-sum) rejects to enumeration. Homogeneity is checked
/// across the real lists only -- the pool list is excluded, since it is unbounded.
fn parse_pool_scan_constraints(model: &CollectionModel, pool_list: usize) -> Option<Vec<ScanRoutingSpec>> {
    let mut by_list: Vec<Vec<ScanRoutingSpec>> = (0..model.lists).map(|_| Vec::new()).collect();
    for constraint in &model.constraints {
        let Iterable::Scan { list, .. } = constraint.reduction.iterable else {
            return None;
        };
        if list >= model.lists || list == pool_list {
            return None;
        }
        by_list[list].push(scan_routing_signature(constraint, &model.items).filter(|s| s.end.is_none())?);
    }
    let mut real = (0..model.lists).filter(|&list| list != pool_list).map(|list| &by_list[list]);
    let first = real.next()?;
    real.all(|scans| scans == first).then(|| first.clone())
}

/// Decode a solved pool tour into `depot_count` lists: the segment starting at
/// each real depot `0..k` (in depot-index order, arbitrary since the real depots
/// are interchangeable) followed by the pool depot's segment as the LAST list,
/// matching the enumeration convention where the free pool list contributes 0.
fn decode_pool_routes(nodes: &[i32], depot_count: usize, succ: &[i32]) -> Vec<Vec<i32>> {
    let segment = |start: usize| -> Vec<i32> {
        let mut route = Vec::new();
        let mut cur = succ[start] as usize;
        let mut steps = 0usize;
        while cur >= depot_count && cur < nodes.len() && steps <= nodes.len() {
            route.push(nodes[cur]);
            cur = succ[cur] as usize;
            steps += 1;
        }
        route
    };
    let pool_depot = depot_count - 1;
    let mut lists: Vec<Vec<i32>> = (0..pool_depot).map(segment).collect();
    lists.push(segment(pool_depot));
    lists
}

fn direct_closed_edge_term(reduction: &Reduction) -> Option<(usize, i32, DirectEdgeMatrix)> {
    if !matches!(reduction.op, ReduceOp::Sum) || reduction.coeff != 1 {
        return None;
    }
    let Iterable::Edges { list: reduction_list, start, end } = &reduction.iterable else {
        return None;
    };
    if start != end {
        return None;
    }
    direct_edge_matrix(reduction).map(|edge| (*reduction_list, *start, edge))
}

fn direct_edge_matrix(reduction: &Reduction) -> Option<DirectEdgeMatrix> {
    match reduction.arena.exprs.get(reduction.body.0 as usize) {
        Some(Expr::Matrix(matrix, row, col))
            if expr_is_arg(&reduction.arena.exprs, *row, 0) && expr_is_arg(&reduction.arena.exprs, *col, 1) =>
        {
            Some(DirectEdgeMatrix { values: Arc::clone(matrix), orientation: MatrixOrientation::Direct })
        }
        Some(Expr::Matrix(matrix, row, col))
            if expr_is_arg(&reduction.arena.exprs, *row, 1) && expr_is_arg(&reduction.arena.exprs, *col, 0) =>
        {
            Some(DirectEdgeMatrix { values: Arc::clone(matrix), orientation: MatrixOrientation::Reversed })
        }
        _ => None,
    }
}

impl DirectEdgeMatrix {
    fn same_matrix_as(&self, other: &Self) -> bool {
        self.orientation == other.orientation && (Arc::ptr_eq(&self.values, &other.values) || self.values.as_ref() == other.values.as_ref())
    }
}

/// Classify the model's side constraints into homogeneous per-route capacity
/// (`Sum(demands) <= cap`), length (`Count == k`), and general weighted item sum
/// (any other `Sum`). Each must be either absent or identical across every route --
/// route-specific bounds are unsafe under depot-copy symmetry, so they cause a
/// `None` (fall to LS). Precedence: `Count` -> length; `Sum(nonneg) <= cap` ->
/// capacity (MTZ load); any other `Sum` -> item sum (membership channel).
fn parse_route_side_constraints(model: &CollectionModel) -> Option<RouteSideConstraints> {
    let mut cap_by_list: Vec<Option<CapacitySpec>> = (0..model.lists).map(|_| None).collect();
    let mut len_by_list: Vec<Option<(usize, usize)>> = (0..model.lists).map(|_| None).collect();
    let mut sum_by_list: Vec<Option<ItemSumSpec>> = (0..model.lists).map(|_| None).collect();
    let mut scan_by_list: Vec<Vec<ScanRoutingSpec>> = (0..model.lists).map(|_| Vec::new()).collect();
    for constraint in &model.constraints {
        if let Some((list, min, max)) = parse_length_constraint(constraint, model.items.len()) {
            if list >= model.lists || len_by_list[list].is_some() {
                return None;
            }
            len_by_list[list] = Some((min, max));
        } else if let Some((list, capacity)) = parse_capacity_constraint(constraint, &model.items) {
            if list >= model.lists || cap_by_list[list].is_some() {
                return None;
            }
            cap_by_list[list] = Some(capacity);
        } else if let Some((list, spec)) = parse_item_sum_constraint(constraint, &model.items) {
            if list >= model.lists || sum_by_list[list].is_some() {
                return None;
            }
            sum_by_list[list] = Some(spec);
        } else if let Iterable::Scan { list, .. } = constraint.reduction.iterable {
            // Mirrors the classifier gate: a `Sum` scan lowers, anything else falls
            // to LS. Keeping this the same predicate keeps the two gates in lockstep.
            let spec = scan_routing_signature(constraint, &model.items).filter(|s| s.end.is_none())?;
            if list >= model.lists {
                return None;
            }
            scan_by_list[list].push(spec);
        } else {
            return None;
        }
    }
    Some(RouteSideConstraints {
        capacity: homogeneous(&cap_by_list)?,
        length: homogeneous(&len_by_list)?,
        item_sum: homogeneous(&sum_by_list)?,
        scan: homogeneous_scans(&scan_by_list)?,
    })
}

/// Every route must carry the identical (ordered) set of scans, or none.
/// Mirrors the classifier's `scan_lists_homogeneous`.
fn homogeneous_scans(by_list: &[Vec<ScanRoutingSpec>]) -> Option<Vec<ScanRoutingSpec>> {
    let first = by_list.first()?;
    by_list.iter().all(|scans| scans == first).then(|| first.clone())
}

/// `Some(None)` = absent from every route, `Some(Some(v))` = the shared value,
/// `None` = mixed/route-named (reject). A spec folds into one homogeneous rule
/// only if it is identical across every route.
fn homogeneous<T: Clone + PartialEq>(by_list: &[Option<T>]) -> Option<Option<T>> {
    let present = by_list.iter().filter(|x| x.is_some()).count();
    if present == 0 {
        return Some(None);
    }
    if present != by_list.len() {
        return None;
    }
    let first = by_list[0].as_ref().expect("all present");
    by_list.iter().all(|x| x.as_ref().expect("all present") == first).then(|| Some(first.clone()))
}

/// A general weighted item sum `Sum(weight(item)) <op> rhs` over one route's items.
/// Returns `(list, weights, min, max)`. Distinct from capacity (which is the
/// `<=`-with-nonnegative-demands special case handled by the MTZ load).
fn parse_item_sum_constraint(constraint: &Constraint, items: &[i32]) -> Option<(usize, ItemSumSpec)> {
    if !matches!(constraint.reduction.op, ReduceOp::Sum) {
        return None;
    }
    let Iterable::Items(list) = &constraint.reduction.iterable else {
        return None;
    };
    let mut weights = Vec::with_capacity(items.len());
    for &item in items {
        let weight = eval_collection_expr_one(&constraint.reduction.arena.exprs, constraint.reduction.body, i64::from(item))?
            .saturating_mul(constraint.reduction.coeff);
        weights.push((item, weight));
    }
    let (min, max) = match constraint.op {
        Op::Le => (i64::MIN / 4, constraint.rhs),
        Op::Ge => (constraint.rhs, i64::MAX / 4),
        Op::Eq => (constraint.rhs, constraint.rhs),
    };
    Some((*list, (weights, min, max)))
}

/// A homogeneous-capable length constraint `Count(items in list) <op> k` with a
/// non-zero constant body (so it counts membership). Returns `(list, min, max)`.
fn parse_length_constraint(constraint: &Constraint, item_count: usize) -> Option<(usize, usize, usize)> {
    if !matches!(constraint.reduction.op, ReduceOp::Count) {
        return None;
    }
    if constraint.reduction.coeff != 1 {
        return None;
    }
    let Iterable::Items(list) = &constraint.reduction.iterable else {
        return None;
    };
    let body = constraint.reduction.arena.exprs.get(constraint.reduction.body.0 as usize)?;
    let Expr::Const(value) = body else {
        return None;
    };
    if *value == 0 {
        return None;
    }
    let n = item_count as i64;
    let (min, max) = match constraint.op {
        Op::Le => (0, constraint.rhs),
        Op::Ge => (constraint.rhs, n),
        Op::Eq => (constraint.rhs, constraint.rhs),
    };
    if max < 0 || min > n || min > max {
        return None;
    }
    Some((*list, min.max(0) as usize, max.min(n) as usize))
}

fn parse_capacity_constraint(constraint: &Constraint, items: &[i32]) -> Option<(usize, CapacitySpec)> {
    if !matches!(constraint.op, Op::Le) || !matches!(constraint.reduction.op, ReduceOp::Sum) {
        return None;
    }
    let Iterable::Items(list) = &constraint.reduction.iterable else {
        return None;
    };
    let capacity = i32::try_from(constraint.rhs).ok()?;
    if capacity < 0 {
        return None;
    }
    let mut demands = Vec::with_capacity(items.len());
    for &item in items {
        let demand = i32::try_from(
            eval_collection_expr_one(&constraint.reduction.arena.exprs, constraint.reduction.body, i64::from(item))?
                .saturating_mul(constraint.reduction.coeff),
        )
        .ok()?;
        if demand < 0 || demand > capacity {
            return None;
        }
        demands.push(demand);
    }
    Some((*list, CapacitySpec { demands, capacity }))
}

fn expr_is_arg(exprs: &[Expr], id: ExprId, arg: u8) -> bool {
    matches!(exprs.get(id.0 as usize), Some(Expr::Arg(found)) if *found == arg)
}

fn items_are_unique(items: &[i32]) -> bool {
    for (i, &item) in items.iter().enumerate() {
        if items[i + 1..].contains(&item) {
            return false;
        }
    }
    true
}

fn eval_collection_expr_one(arena: &[Expr], id: ExprId, arg0: i64) -> Option<i64> {
    let node = arena.get(id.0 as usize)?;
    Some(match node {
        Expr::Const(value) => *value,
        Expr::Arg(0) => arg0,
        Expr::Arg(_) | Expr::Matrix(_, _, _) => return None,
        Expr::Array(values, index) => {
            let index = eval_collection_expr_one(arena, *index, arg0)?;
            *values.get(usize::try_from(index).ok()?)?
        }
        Expr::Add(a, b) => eval_collection_expr_one(arena, *a, arg0)?.checked_add(eval_collection_expr_one(arena, *b, arg0)?)?,
        Expr::Sub(a, b) => eval_collection_expr_one(arena, *a, arg0)?.checked_sub(eval_collection_expr_one(arena, *b, arg0)?)?,
        Expr::Mul(a, b) => eval_collection_expr_one(arena, *a, arg0)?.checked_mul(eval_collection_expr_one(arena, *b, arg0)?)?,
        Expr::Min(a, b) => eval_collection_expr_one(arena, *a, arg0)?.min(eval_collection_expr_one(arena, *b, arg0)?),
        Expr::Max(a, b) => eval_collection_expr_one(arena, *a, arg0)?.max(eval_collection_expr_one(arena, *b, arg0)?),
        Expr::Div(a, b) => {
            let numerator = eval_collection_expr_one(arena, *a, arg0)?;
            let denominator = eval_collection_expr_one(arena, *b, arg0)?;
            if denominator == 0 {
                return None;
            }
            numerator / denominator
        }
        Expr::Abs(a) => eval_collection_expr_one(arena, *a, arg0)?.saturating_abs(),
        Expr::Lt(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? < eval_collection_expr_one(arena, *b, arg0)?),
        Expr::Le(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? <= eval_collection_expr_one(arena, *b, arg0)?),
        Expr::Eq(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? == eval_collection_expr_one(arena, *b, arg0)?),
        Expr::Ne(a, b) => i64::from(eval_collection_expr_one(arena, *a, arg0)? != eval_collection_expr_one(arena, *b, arg0)?),
        Expr::IfThenElse(c, a, b) => {
            if eval_collection_expr_one(arena, *c, arg0)? != 0 {
                eval_collection_expr_one(arena, *a, arg0)?
            } else {
                eval_collection_expr_one(arena, *b, arg0)?
            }
        }
    })
}