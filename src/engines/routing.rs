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
        let spec = scan_routing_signature(&synth, items)?;
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
        by_list[list].push(scan_routing_signature(constraint, &model.items)?);
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
            let spec = scan_routing_signature(constraint, &model.items)?;
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::{atomic::AtomicBool, Arc};

    use crate::model::list::{CollectionModel, Constraint, ExprArena, GlobalConstraint, Iterable, ObjectiveTier, Op, ReduceOp, Reduction};
    use crate::search::{self, SearchControl};

    use super::{routes_from_successors, scan_routing_signature, solve_collection, CapacityCheck, RoutingSpec};

    #[test]
    fn lowers_closed_tsp_to_integer_circuit() {
        let dist = vec![vec![0, 10, 15, 20], vec![10, 0, 35, 25], vec![15, 35, 0, 30], vec![20, 25, 30, 0]];
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(std::sync::Arc::new(dist), i, j);
        let reduction = Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: 0, end: 0 }, arena, body, coeff: 1 };
        let model = CollectionModel {
            items: vec![1, 2, 3],
            lists: 1,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![reduction], max_terms: None }],
            constraints: Vec::new(),
            globals: Vec::new(),
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let mut reports = Vec::new();
        let outcome = solve_collection(&model, 0, &stop, &mut |value| reports.push(value)).expect("supported route lowering");

        assert!(outcome.complete);
        assert!(outcome.solution.feasible);
        assert_eq!(outcome.solution.objectives, vec![80]);
        let mut route = outcome.solution.lists[0].clone();
        route.sort_unstable();
        assert_eq!(route, vec![1, 2, 3]);
        // With the LS warm start the verified incumbent can already be optimal, so
        // the exact search may prove optimality without itself reporting/finding a
        // new solution; only the outcome (optimum + route) is the contract here.
        let _ = &reports;
    }

    #[test]
    fn lowers_small_cvrp_to_depot_replicated_circuit() {
        let dist = std::sync::Arc::new(vec![
            vec![0, 1, 1, 1, 1],
            vec![1, 0, 1, 50, 50],
            vec![1, 1, 0, 50, 50],
            vec![1, 50, 50, 0, 1],
            vec![1, 50, 50, 1, 0],
        ]);
        let demand = std::sync::Arc::new(vec![0, 1, 1, 1, 1]);

        let edge = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let j = arena.arg(1);
            let body = arena.matrix(Arc::clone(&dist), i, j);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        };
        let load = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(&demand), i);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 2,
            }
        };

        let model = CollectionModel {
            items: vec![1, 2, 3, 4],
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
            constraints: vec![load(0), load(1)],
            globals: Vec::new(),
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported cvrp lowering");

        assert!(outcome.complete);
        assert!(outcome.solution.feasible);
        assert_eq!(outcome.solution.objectives, vec![6]);
        assert_eq!(outcome.solution.lists.len(), 2);
        let mut served = outcome.solution.lists.iter().flatten().copied().collect::<Vec<_>>();
        served.sort_unstable();
        assert_eq!(served, vec![1, 2, 3, 4]);
        for route in &outcome.solution.lists {
            assert!(route.iter().map(|&node| demand[node as usize]).sum::<i64>() <= 2);
        }
    }

    // --- brute-force oracles ----------------------------------------------------
    //
    // These compare the engine's optimum against an independent exhaustive search,
    // a stronger guard than the hand-computed-value tests above: they prove the
    // engine finds the *true* optimum for the lowering, on arbitrary small matrices.

    /// All permutations of `items` (each a distinct visiting order).
    fn permutations(items: &[i32]) -> Vec<Vec<i32>> {
        if items.is_empty() {
            return vec![Vec::new()];
        }
        let mut out = Vec::new();
        for i in 0..items.len() {
            let mut rest = items.to_vec();
            let head = rest.remove(i);
            for mut tail in permutations(&rest) {
                tail.insert(0, head);
                out.push(tail);
            }
        }
        out
    }

    /// Minimum closed-tour cost `depot -> ... -> depot` over all orderings of
    /// `items` (directed, so asymmetric matrices are handled). Empty route = 0.
    fn min_closed_tour(depot: i32, items: &[i32], dist: &[Vec<i64>]) -> i64 {
        let mut best = i64::MAX;
        for perm in permutations(items) {
            let mut cost = 0i64;
            let mut prev = depot;
            for &node in &perm {
                cost += dist[prev as usize][node as usize];
                prev = node;
            }
            cost += dist[prev as usize][depot as usize];
            best = best.min(cost);
        }
        best
    }

    #[test]
    #[ignore = "benchmark, run with --ignored --nocapture; ablate QAYD_ROUTING_TWOWAY / QAYD_ROUTING_WARMSTART"]
    fn measure_two_way() {
        // Membership-routing instance: 2 routes, 8 customers, capacity 4 per route,
        // two same_list pairs, and each route exactly 4 customers. Exercises the
        // reverse member -> succ channel.
        let coords: [(i64, i64); 9] = [(0, 0), (1, 1), (2, 5), (5, 2), (6, 6), (2, 2), (7, 1), (3, 7), (8, 5)];
        let n = coords.len();
        let dist: Vec<Vec<i64>> =
            (0..n).map(|i| (0..n).map(|j| (coords[i].0 - coords[j].0).abs() + (coords[i].1 - coords[j].1).abs()).collect()).collect();
        let items: Vec<i32> = (1..n as i32).collect();
        let dist_arc = Arc::new(dist);
        let demand_arc = Arc::new(vec![1i64; n]);
        let edge = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let j = arena.arg(1);
            let body = arena.matrix(Arc::clone(&dist_arc), i, j);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        };
        let load = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(&demand_arc), i);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 4,
            }
        };
        let model = CollectionModel {
            items,
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
            constraints: vec![load(0), load(1), count_eq(0, 4), count_eq(1, 4)],
            globals: vec![GlobalConstraint::SameList { a: 1, b: 5 }, GlobalConstraint::SameList { a: 2, b: 6 }],
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported membership routing model");
        eprintln!(
            "TWOWAY={:?} WARM={:?} nodes={} failures={} learned_lits={} obj={:?} complete={}",
            std::env::var("QAYD_ROUTING_TWOWAY").ok(),
            std::env::var("QAYD_ROUTING_WARMSTART").ok(),
            outcome.stats.nodes,
            outcome.stats.failures,
            outcome.stats.learned_lits,
            outcome.solution.objectives,
            outcome.complete,
        );
    }

    #[test]
    #[ignore = "measurement, run explicitly with --ignored --nocapture"]
    fn measure_nn_seed() {
        let coords: [(i64, i64); 10] = [(0, 0), (1, 5), (2, 3), (5, 1), (8, 8), (3, 7), (7, 2), (6, 6), (4, 4), (9, 5)];
        let n = coords.len();
        let dist: Vec<Vec<i64>> =
            (0..n).map(|i| (0..n).map(|j| (coords[i].0 - coords[j].0).abs() + (coords[i].1 - coords[j].1).abs()).collect()).collect();
        let items: Vec<i32> = (1..n as i32).collect();
        let model = closed_tsp_model(items, 0, &Arc::new(dist));
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported");
        eprintln!(
            "NN={:?} nodes={} failures={} solutions={} obj={:?} complete={}",
            std::env::var("QAYD_ROUTING_NN").ok(),
            outcome.stats.nodes,
            outcome.stats.failures,
            outcome.stats.solutions,
            outcome.solution.objectives,
            outcome.complete,
        );
        assert!(outcome.complete);
    }

    fn closed_tsp_model(items: Vec<i32>, depot: i32, dist: &Arc<Vec<Vec<i64>>>) -> CollectionModel {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(dist), i, j);
        let reduction =
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: depot, end: depot }, arena, body, coeff: 1 };
        CollectionModel {
            items,
            lists: 1,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![reduction], max_terms: None }],
            constraints: Vec::new(),
            globals: Vec::new(),
            schedule: None,
        }
    }

    #[test]
    fn closed_tsp_matches_brute_force() {
        // depot = node 0, items = nodes 1..=3. Several matrices, including an
        // asymmetric one, each checked against the exhaustive closed-tour minimum.
        let cases: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![0, 10, 15, 20], vec![10, 0, 35, 25], vec![15, 35, 0, 30], vec![20, 25, 30, 0]],
            vec![vec![0, 3, 7, 5], vec![5, 0, 2, 9], vec![6, 4, 0, 1], vec![8, 7, 3, 0]],
            vec![vec![0, 2, 9, 1], vec![1, 0, 6, 4], vec![3, 8, 0, 2], vec![7, 5, 2, 0]],
        ];
        let items = vec![1, 2, 3];
        for dist in cases {
            let brute = min_closed_tour(0, &items, &dist);
            let model = closed_tsp_model(items.clone(), 0, &Arc::new(dist));
            let stop = AtomicBool::new(false);
            let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported route lowering");

            assert!(outcome.complete, "exhausted the search");
            assert_eq!(outcome.solution.objectives, vec![brute], "engine optimum equals the exhaustive closed-tour minimum");
            let mut route = outcome.solution.lists[0].clone();
            route.sort_unstable();
            assert_eq!(route, items, "route visits every item exactly once");
        }
    }

    /// Minimum total cost of partitioning `items` across `n_routes` capacitated
    /// closed tours: every assignment of items to routes, each route ordered
    /// optimally, capacity respected. `None` if no assignment is feasible.
    fn min_cvrp(depot: i32, items: &[i32], dist: &[Vec<i64>], demand: &[i64], n_routes: usize, capacity: i64) -> Option<i64> {
        let n = items.len();
        let combos = (n_routes as u64).pow(n as u32);
        let mut best: Option<i64> = None;
        for combo in 0..combos {
            let mut routes: Vec<Vec<i32>> = vec![Vec::new(); n_routes];
            let mut code = combo;
            for &item in items {
                let route = (code % n_routes as u64) as usize;
                code /= n_routes as u64;
                routes[route].push(item);
            }
            if routes.iter().any(|r| r.iter().map(|&node| demand[node as usize]).sum::<i64>() > capacity) {
                continue;
            }
            let cost: i64 = routes.iter().map(|r| min_closed_tour(depot, r, dist)).sum();
            best = Some(best.map_or(cost, |b| b.min(cost)));
        }
        best
    }

    #[test]
    fn small_cvrp_matches_brute_force() {
        // 2 routes, 4 customers, identical per-route capacity 2 (each customer
        // demands 1). depot = node 0. Engine optimum must equal the exhaustive
        // capacitated multi-route minimum.
        let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 50, 50], vec![1, 1, 0, 50, 50], vec![1, 50, 50, 0, 1], vec![1, 50, 50, 1, 0]];
        let demand = vec![0i64, 1, 1, 1, 1];
        let items = vec![1, 2, 3, 4];
        let brute = min_cvrp(0, &items, &dist, &demand, 2, 2).expect("a feasible packing exists");

        let dist_arc = Arc::new(dist);
        let demand_arc = Arc::new(demand.clone());
        let edge = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let j = arena.arg(1);
            let body = arena.matrix(Arc::clone(&dist_arc), i, j);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        };
        let load = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(&demand_arc), i);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 2,
            }
        };
        let model = CollectionModel {
            items: items.clone(),
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
            constraints: vec![load(0), load(1)],
            globals: Vec::new(),
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported cvrp lowering");

        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives, vec![brute], "engine optimum equals the exhaustive CVRP minimum");
        let mut served = outcome.solution.lists.iter().flatten().copied().collect::<Vec<_>>();
        served.sort_unstable();
        assert_eq!(served, items, "every customer served exactly once");
    }

    /// Like `min_cvrp`, but discard any assignment that splits a `same_list` pair.
    fn min_cvrp_same_list(
        depot: i32,
        items: &[i32],
        dist: &[Vec<i64>],
        demand: &[i64],
        n_routes: usize,
        capacity: i64,
        same: &[(i32, i32)],
    ) -> Option<i64> {
        let n = items.len();
        let combos = (n_routes as u64).pow(n as u32);
        let mut best: Option<i64> = None;
        for combo in 0..combos {
            let mut routes: Vec<Vec<i32>> = vec![Vec::new(); n_routes];
            let mut segment_by_item: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
            let mut code = combo;
            for &item in items {
                let route = (code % n_routes as u64) as usize;
                code /= n_routes as u64;
                routes[route].push(item);
                segment_by_item.insert(item, route);
            }
            if same.iter().any(|&(a, b)| segment_by_item[&a] != segment_by_item[&b]) {
                continue;
            }
            if routes.iter().any(|r| r.iter().map(|&node| demand[node as usize]).sum::<i64>() > capacity) {
                continue;
            }
            let cost: i64 = routes.iter().map(|r| min_closed_tour(depot, r, dist)).sum();
            best = Some(best.map_or(cost, |b| b.min(cost)));
        }
        best
    }

    /// Build a 2-route CVRP-shaped model with capacity 2 and optional `same_list`
    /// pairs, over the shared `dist`/`demand` tables.
    fn cvrp_same_list_model(
        items: Vec<i32>,
        dist: &Arc<Vec<Vec<i64>>>,
        demand: &Arc<Vec<i64>>,
        capacity: i64,
        same: &[(i32, i32)],
    ) -> CollectionModel {
        let edge = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let j = arena.arg(1);
            let body = arena.matrix(Arc::clone(dist), i, j);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        };
        let load = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(demand), i);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Le,
                rhs: capacity,
            }
        };
        CollectionModel {
            items,
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
            constraints: vec![load(0), load(1)],
            globals: same.iter().map(|&(a, b)| GlobalConstraint::SameList { a, b }).collect(),
            schedule: None,
        }
    }

    #[test]
    fn same_list_changes_cvrp_optimum_and_matches_brute_force() {
        // Unconstrained optimum pairs {1,2} and {3,4} (cost 6). same_list(1, 3)
        // forces 1 and 3 together, which is far more expensive. The engine optimum
        // must equal the same-list-filtered exhaustive minimum, and must exceed the
        // unconstrained optimum (proving the channel actually bites).
        let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 50, 50], vec![1, 1, 0, 50, 50], vec![1, 50, 50, 0, 1], vec![1, 50, 50, 1, 0]];
        let demand = vec![0i64, 1, 1, 1, 1];
        let items = vec![1, 2, 3, 4];
        let unconstrained = min_cvrp(0, &items, &dist, &demand, 2, 2).expect("feasible");
        let same = [(1, 3)];
        let brute = min_cvrp_same_list(0, &items, &dist, &demand, 2, 2, &same).expect("feasible under same_list");
        assert!(brute > unconstrained, "same_list(1,3) is more expensive than the free optimum");

        let model = cvrp_same_list_model(items.clone(), &Arc::new(dist), &Arc::new(demand), 2, &same);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported same_list cvrp");

        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives, vec![brute], "engine optimum equals the same-list-filtered minimum");
        let segment_of_item = |item: i32| outcome.solution.lists.iter().position(|r| r.contains(&item)).expect("served");
        assert_eq!(segment_of_item(1), segment_of_item(3), "same_list keeps 1 and 3 in the same route");
    }

    #[test]
    fn same_list_with_capacity_can_be_unsat() {
        // Customer 1 demands the whole capacity (2). same_list(1, 2) forces 1 and 2
        // into one route, whose combined demand 3 exceeds capacity 2: infeasible.
        let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 1, 1], vec![1, 1, 0, 1, 1], vec![1, 1, 1, 0, 1], vec![1, 1, 1, 1, 0]];
        let demand = vec![0i64, 2, 1, 1, 1];
        let items = vec![1, 2, 3, 4];
        let same = [(1, 2)];
        assert!(min_cvrp_same_list(0, &items, &dist, &demand, 2, 2, &same).is_none(), "no feasible same_list packing exists");

        let model = cvrp_same_list_model(items, &Arc::new(dist), &Arc::new(demand), 2, &same);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported same_list cvrp");

        assert!(outcome.complete, "search exhausted");
        assert!(!outcome.solution.feasible, "same_list(1,2) over-capacity is UNSAT");
    }

    /// General exhaustive multi-route minimum under optional capacity, same-list,
    /// and per-route length filters. `None` if no assignment is feasible.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn min_routes(
        depot: i32,
        items: &[i32],
        dist: &[Vec<i64>],
        demand: &[i64],
        n_routes: usize,
        capacity: Option<i64>,
        same: &[(i32, i32)],
        length: Option<(usize, usize)>,
        item_sum: Option<(&[(i32, i64)], i64, i64)>,
    ) -> Option<i64> {
        let n = items.len();
        let combos = (n_routes as u64).pow(n as u32);
        let mut best: Option<i64> = None;
        for combo in 0..combos {
            let mut routes: Vec<Vec<i32>> = vec![Vec::new(); n_routes];
            let mut segment_by_item: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
            let mut code = combo;
            for &item in items {
                let route = (code % n_routes as u64) as usize;
                code /= n_routes as u64;
                routes[route].push(item);
                segment_by_item.insert(item, route);
            }
            if same.iter().any(|&(a, b)| segment_by_item[&a] != segment_by_item[&b]) {
                continue;
            }
            if let Some(cap) = capacity {
                if routes.iter().any(|r| r.iter().map(|&node| demand[node as usize]).sum::<i64>() > cap) {
                    continue;
                }
            }
            if let Some((lo, hi)) = length {
                if routes.iter().any(|r| r.len() < lo || r.len() > hi) {
                    continue;
                }
            }
            if let Some((weights, lo, hi)) = item_sum {
                let wmap: std::collections::HashMap<i32, i64> = weights.iter().copied().collect();
                if routes.iter().any(|r| {
                    let sum: i64 = r.iter().map(|&node| wmap.get(&node).copied().unwrap_or(0)).sum();
                    sum < lo || sum > hi
                }) {
                    continue;
                }
            }
            let cost: i64 = routes.iter().map(|r| min_closed_tour(depot, r, dist)).sum();
            best = Some(best.map_or(cost, |b| b.min(cost)));
        }
        best
    }

    fn successor_cycle_from_order(n: usize, order: &[usize]) -> Vec<i32> {
        let mut succ = vec![0i32; n];
        let mut prev = 0usize;
        for &node in order {
            succ[prev] = node as i32;
            prev = node;
        }
        succ[prev] = 0;
        succ
    }

    fn depot_symmetry_holds(succ: &[i32], depot_count: usize) -> bool {
        let n = succ.len();
        succ[..depot_count]
            .iter()
            .map(|&head| {
                let head = head as usize;
                if head < depot_count {
                    n
                } else {
                    head
                }
            })
            .collect::<Vec<_>>()
            .windows(2)
            .all(|w| w[0] <= w[1])
    }

    fn routes_satisfy_spec(spec: &RoutingSpec, routes: &[Vec<i32>], succ: &[i32]) -> bool {
        if let Some(capacity) = &spec.capacity {
            let demands = spec.node_demands(capacity);
            if !matches!(super::route_capacity_check(spec.depot_count, &demands, capacity.capacity, succ), CapacityCheck::Ok) {
                return false;
            }
        }
        let segment_of = |item: i32| routes.iter().position(|route| route.contains(&item));
        if spec.same_list.iter().any(|&(a, b)| segment_of(a) != segment_of(b)) {
            return false;
        }
        if let Some((min, max)) = spec.length {
            if routes.iter().any(|route| route.len() < min || route.len() > max) {
                return false;
            }
        }
        if let Some((weights, min, max)) = &spec.item_sum {
            if routes.iter().any(|route| {
                let sum = route
                    .iter()
                    .map(|item| weights.iter().find_map(|&(x, weight)| (x == *item).then_some(weight)).unwrap_or(0))
                    .sum::<i64>();
                sum < *min || sum > *max
            }) {
                return false;
            }
        }
        true
    }

    fn chronological_successor_set(spec: &RoutingSpec) -> BTreeSet<Vec<i32>> {
        let mut nodes = Vec::with_capacity(spec.depot_count + spec.items.len());
        nodes.extend((0..spec.depot_count).map(|_| spec.depot));
        nodes.extend_from_slice(&spec.items);
        let n = nodes.len();
        let mut order: Vec<usize> = (1..n).collect();
        let mut out = BTreeSet::new();
        fn rec(pos: usize, order: &mut [usize], nodes: &[i32], spec: &RoutingSpec, out: &mut BTreeSet<Vec<i32>>) {
            if pos == order.len() {
                let succ = successor_cycle_from_order(nodes.len(), order);
                if !depot_symmetry_holds(&succ, spec.depot_count) {
                    return;
                }
                let Some(routes) = routes_from_successors(nodes, spec.depot_count, &succ) else {
                    return;
                };
                if routes_satisfy_spec(spec, &routes, &succ) {
                    out.insert(succ);
                }
                return;
            }
            for i in pos..order.len() {
                order.swap(pos, i);
                rec(pos + 1, order, nodes, spec, out);
                order.swap(pos, i);
            }
        }
        rec(0, &mut order, &nodes, spec, &mut out);
        out
    }

    fn cdcl_successor_set(model: &CollectionModel) -> BTreeSet<Vec<i32>> {
        let spec = RoutingSpec::from_model(model).expect("supported route lowering");
        let mut lowered = spec.lower();
        let n = lowered.nodes.len();
        let mut out = BTreeSet::new();
        search::solve(&mut lowered.solver, &lowered.search_vars, |solver| {
            out.insert(lowered.search_vars[..n].iter().map(|&var| solver.store.value(var)).collect::<Vec<_>>());
            SearchControl::Continue
        });
        out
    }

    fn count_eq(list: usize, k: i64) -> Constraint {
        let mut arena = ExprArena::default();
        let body = arena.constant(1); // count of items present = route length
        Constraint {
            reduction: Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body, coeff: 1 },
            op: Op::Eq,
            rhs: k,
        }
    }

    #[test]
    fn list_len_exact_route_size_matches_brute_force() {
        // Free optimum is unbalanced (one customer alone), but `list_len` forces
        // exactly 2 customers per route. The engine optimum must equal the
        // size-filtered exhaustive minimum and exceed the free optimum.
        let dist =
            vec![vec![0, 1, 10, 10, 10], vec![1, 0, 10, 10, 10], vec![10, 10, 0, 1, 1], vec![10, 10, 1, 0, 1], vec![10, 10, 1, 1, 0]];
        let zero = vec![0i64; 5];
        let items = vec![1, 2, 3, 4];
        let free = min_routes(0, &items, &dist, &zero, 2, None, &[], None, None).expect("feasible");
        let balanced = min_routes(0, &items, &dist, &zero, 2, None, &[], Some((2, 2)), None).expect("feasible balanced");
        assert!(balanced > free, "forcing two-per-route is costlier than the free optimum");

        let dist_arc = Arc::new(dist);
        let edge = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let j = arena.arg(1);
            let body = arena.matrix(Arc::clone(&dist_arc), i, j);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        };
        let model = CollectionModel {
            items: items.clone(),
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
            constraints: vec![count_eq(0, 2), count_eq(1, 2)],
            globals: Vec::new(),
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported route length model");

        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives, vec![balanced], "engine optimum equals the size-filtered minimum");
        for route in &outcome.solution.lists {
            assert_eq!(route.len(), 2, "every route holds exactly two customers");
        }
    }

    #[test]
    fn same_list_length_and_capacity_compose() {
        // same_list(1,3) + each route exactly 2 + capacity 2: the three constraints
        // coexist; the engine optimum equals the jointly-filtered exhaustive minimum.
        let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 50, 50], vec![1, 1, 0, 50, 50], vec![1, 50, 50, 0, 1], vec![1, 50, 50, 1, 0]];
        let demand = vec![0i64, 1, 1, 1, 1];
        let items = vec![1, 2, 3, 4];
        let same = [(1, 3)];
        let brute = min_routes(0, &items, &dist, &demand, 2, Some(2), &same, Some((2, 2)), None).expect("feasible");

        let dist_arc = Arc::new(dist);
        let demand_arc = Arc::new(demand);
        let edge = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let j = arena.arg(1);
            let body = arena.matrix(Arc::clone(&dist_arc), i, j);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        };
        let load = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(&demand_arc), i);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 2,
            }
        };
        let model = CollectionModel {
            items: items.clone(),
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
            constraints: vec![load(0), load(1), count_eq(0, 2), count_eq(1, 2)],
            globals: vec![GlobalConstraint::SameList { a: 1, b: 3 }],
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported combined model");

        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives, vec![brute], "engine optimum equals the jointly-filtered minimum");
        let segment_of_item = |item: i32| outcome.solution.lists.iter().position(|r| r.contains(&item)).expect("served");
        assert_eq!(segment_of_item(1), segment_of_item(3), "same_list keeps 1 and 3 together");
        for route in &outcome.solution.lists {
            assert_eq!(route.len(), 2, "every route holds exactly two customers");
        }
    }

    #[test]
    fn route_item_sum_ge_matches_brute_force() {
        // Each route must reach a value sum >= 11 (a non-capacity `Sum`, so it routes
        // through the membership channel, not the MTZ load). The cheap free optimum
        // groups the two low-value customers together, which the bound forbids, so
        // the engine optimum must equal the value-filtered exhaustive minimum and
        // exceed the free optimum.
        let dist = vec![vec![0, 1, 1, 1, 1], vec![1, 0, 1, 50, 50], vec![1, 1, 0, 50, 50], vec![1, 50, 50, 0, 1], vec![1, 50, 50, 1, 0]];
        let value = vec![0i64, 1, 1, 10, 10]; // indexed by node; customers 3,4 are high value
        let weights = vec![(1i32, 1i64), (2, 1), (3, 10), (4, 10)];
        let items = vec![1, 2, 3, 4];
        let zero = vec![0i64; 5];
        let free = min_routes(0, &items, &dist, &zero, 2, None, &[], None, None).expect("feasible");
        let bound = (weights.as_slice(), 11i64, i64::MAX / 4);
        let constrained = min_routes(0, &items, &dist, &zero, 2, None, &[], None, Some(bound)).expect("feasible under value bound");
        assert!(constrained > free, "the per-route value floor is costlier than the free optimum");

        let dist_arc = Arc::new(dist);
        let value_arc = Arc::new(value);
        let edge = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let j = arena.arg(1);
            let body = arena.matrix(Arc::clone(&dist_arc), i, j);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        };
        let value_sum = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(&value_arc), i);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Ge,
                rhs: 11,
            }
        };
        let model = CollectionModel {
            items: items.clone(),
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
            constraints: vec![value_sum(0), value_sum(1)],
            globals: Vec::new(),
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported route item-sum model");

        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives, vec![constrained], "engine optimum equals the value-filtered minimum");
        for route in &outcome.solution.lists {
            let sum: i64 = route.iter().map(|&node| value_arc[node as usize]).sum();
            assert!(sum >= 11, "every route reaches the value floor");
        }
    }

    #[test]
    fn routing_cdcl_enumerates_chronological_successor_set() {
        // A small mixed routing model: circuit + depot symmetry + capacity + channel
        // constraints (`same_list`, homogeneous length, homogeneous item_sum). CDCL
        // enumeration uses all routing explanations; the chronological set is an
        // independent permutation scan over successor cycles with the same filters.
        let dist = vec![vec![0, 1, 2, 3, 4], vec![1, 0, 1, 5, 5], vec![2, 1, 0, 5, 1], vec![3, 5, 5, 0, 1], vec![4, 5, 1, 1, 0]];
        let demand = vec![0i64, 1, 1, 1, 1];
        let value = vec![0i64, 1, 10, 10, 1];
        let items = vec![1, 2, 3, 4];

        let dist_arc = Arc::new(dist);
        let demand_arc = Arc::new(demand);
        let value_arc = Arc::new(value);
        let edge = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let j = arena.arg(1);
            let body = arena.matrix(Arc::clone(&dist_arc), i, j);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        };
        let load = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(&demand_arc), i);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 2,
            }
        };
        let value_sum = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(&value_arc), i);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Ge,
                rhs: 11,
            }
        };
        let model = CollectionModel {
            items,
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
            constraints: vec![load(0), load(1), count_eq(0, 2), count_eq(1, 2), value_sum(0), value_sum(1)],
            globals: vec![GlobalConstraint::SameList { a: 1, b: 3 }],
            schedule: None,
        };
        let spec = RoutingSpec::from_model(&model).expect("supported mixed routing model");
        let chronological = chronological_successor_set(&spec);
        assert!(!chronological.is_empty(), "mixed routing model has feasible successor tours");

        let cdcl = cdcl_successor_set(&model);
        assert_eq!(cdcl, chronological, "CDCL routing enumeration matches chronological successor enumeration");
    }

    #[test]
    fn route_capacity_overflow_is_unsat() {
        // 3 customers each demand 2, capacity 3, 2 routes: any route with two or more
        // customers overflows (>= 4 > 3), but three customers across two routes force
        // one such route. The explained capacity check must make this infeasible,
        // matching the exhaustive (capacity-filtered) result.
        let dist = vec![vec![0, 1, 1, 1], vec![1, 0, 1, 1], vec![1, 1, 0, 1], vec![1, 1, 1, 0]];
        let demand = vec![0i64, 2, 2, 2];
        let items = vec![1, 2, 3];
        assert!(min_routes(0, &items, &dist, &demand, 2, Some(3), &[], None, None).is_none(), "no capacity-feasible split exists");

        let dist_arc = Arc::new(dist);
        let demand_arc = Arc::new(demand);
        let edge = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let j = arena.arg(1);
            let body = arena.matrix(Arc::clone(&dist_arc), i, j);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        };
        let load = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(&demand_arc), i);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 3,
            }
        };
        let model = CollectionModel {
            items,
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
            constraints: vec![load(0), load(1)],
            globals: Vec::new(),
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported cvrp lowering");

        assert!(outcome.complete, "exhausted the search");
        assert!(!outcome.solution.feasible, "capacity overflow makes the model infeasible");
    }

    // --- scan (resource-extension) oracles --------------------------------------

    /// Time-window lateness scan: `step = max(earliest, acc + travel) + service`,
    /// emitting `max(0, end - latest)`. Mirrors `tests/collection.rs::lateness_scan`.
    fn lateness_scan(list: usize, dist: &[Vec<i64>], earliest: &[i64], latest: &[i64], service: &[i64], depot: i32) -> Reduction {
        let mut arena = ExprArena::default();
        let cur = arena.arg(0);
        let acc = arena.arg(1);
        let prev = arena.arg(2);
        let travel = arena.matrix(Arc::new(dist.to_vec()), prev, cur);
        let arrive = arena.add(acc, travel);
        let earl = arena.array(Arc::new(earliest.to_vec()), cur);
        let start = arena.max(earl, arrive);
        let svc = arena.array(Arc::new(service.to_vec()), cur);
        let step = arena.add(start, svc);
        let cur_e = arena.arg(0);
        let end = arena.arg(1);
        let lat = arena.array(Arc::new(latest.to_vec()), cur_e);
        let over = arena.sub(end, lat);
        let zero = arena.constant(0);
        let emit = arena.max(zero, over);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list, init: 0, boundary: depot, step }, arena, body: emit, coeff: 1 }
    }

    /// Oracle replaying the lateness scan over an ordered route.
    fn lateness_sum(order: &[i32], dist: &[Vec<i64>], earliest: &[i64], latest: &[i64], service: &[i64], depot: i32) -> i64 {
        let mut acc = 0i64;
        let mut prev = depot;
        let mut sum = 0i64;
        for &c in order {
            let end = earliest[c as usize].max(acc + dist[prev as usize][c as usize]) + service[c as usize];
            sum += (end - latest[c as usize]).max(0);
            acc = end;
            prev = c;
        }
        sum
    }

    fn ordered_closed_cost(depot: i32, order: &[i32], dist: &[Vec<i64>]) -> i64 {
        let mut cost = 0i64;
        let mut prev = depot;
        for &c in order {
            cost += dist[prev as usize][c as usize];
            prev = c;
        }
        cost + dist[prev as usize][depot as usize]
    }

    fn op_ok(v: i64, op: Op, rhs: i64) -> bool {
        match op {
            Op::Le => v <= rhs,
            Op::Ge => v >= rhs,
            Op::Eq => v == rhs,
        }
    }

    /// Exhaustive minimum total distance over two-route partitions where each
    /// route has a distance-optimal ordering meeting its per-route scan bound (and
    /// optional capacity). The scan is filtered per route, matching the LS spec.
    #[allow(clippy::too_many_arguments)]
    fn min_two_routes_scan(
        items: &[i32],
        dist: &[Vec<i64>],
        demand: &[i64],
        capacity: Option<i64>,
        earliest: &[i64],
        latest: &[i64],
        service: &[i64],
        depot: i32,
        op: Op,
        rhs: i64,
    ) -> Option<i64> {
        let best_route = |route: &[i32]| -> Option<i64> {
            if let Some(cap) = capacity {
                if route.iter().map(|&c| demand[c as usize]).sum::<i64>() > cap {
                    return None;
                }
            }
            permutations(route)
                .iter()
                .filter(|p| op_ok(lateness_sum(p, dist, earliest, latest, service, depot), op, rhs))
                .map(|p| ordered_closed_cost(depot, p, dist))
                .min()
        };
        let n = items.len();
        let mut best: Option<i64> = None;
        for code in 0..(1u32 << n) {
            let mut routes = [Vec::new(), Vec::new()];
            for (i, &it) in items.iter().enumerate() {
                routes[((code >> i) & 1) as usize].push(it);
            }
            if let (Some(a), Some(b)) = (best_route(&routes[0]), best_route(&routes[1])) {
                best = Some(best.map_or(a + b, |x| x.min(a + b)));
            }
        }
        best
    }

    fn line_dist(positions: &[i64]) -> Vec<Vec<i64>> {
        positions.iter().map(|&i| positions.iter().map(|&j| (i - j).abs()).collect()).collect()
    }

    fn scan_edge(dist: &Arc<Vec<Vec<i64>>>, list: usize) -> Reduction {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(Arc::clone(dist), i, j);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    }

    fn scan_load(demand: &Arc<Vec<i64>>, list: usize, rhs: i64) -> Constraint {
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let body = arena.array(Arc::clone(demand), i);
        Constraint { reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }, op: Op::Le, rhs }
    }

    #[test]
    fn scan_cvrptw_matches_brute_force() {
        // P2 shape: cumulative service+travel time with hard time windows (lateness
        // sum <= 0) plus capacity 2. Customers 1,2 share a deadline (5) they cannot
        // both meet on one route, so a feasible plan must split them. Proven OPTIMAL.
        let dist = line_dist(&[0, 2, 4, 6, 8]);
        let demand = vec![0i64, 1, 1, 1, 1];
        let service = vec![0i64, 1, 1, 1, 1];
        let earliest = vec![0i64, 0, 0, 0, 0];
        let latest = vec![0i64, 5, 5, 20, 20];
        let items = vec![1, 2, 3, 4];
        let depot = 0;
        let brute =
            min_two_routes_scan(&items, &dist, &demand, Some(2), &earliest, &latest, &service, depot, Op::Le, 0).expect("feasible plan exists");

        let dist_arc = Arc::new(dist.clone());
        let demand_arc = Arc::new(demand.clone());
        let mut constraints = vec![scan_load(&demand_arc, 0, 2), scan_load(&demand_arc, 1, 2)];
        for l in 0..2 {
            constraints.push(Constraint { reduction: lateness_scan(l, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs: 0 });
        }
        let model = CollectionModel {
            items: items.clone(),
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0), scan_edge(&dist_arc, 1)], max_terms: None }],
            constraints,
            globals: Vec::new(),
            schedule: None,
        };

        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported cvrptw scan lowering");
        assert!(outcome.complete, "exhausted the search");
        assert!(outcome.solution.feasible, "a time-window-feasible plan exists");
        assert_eq!(outcome.solution.objectives, vec![brute], "exact optimum equals the tw-filtered exhaustive minimum");
        for route in &outcome.solution.lists {
            assert!(!(route.contains(&1) && route.contains(&2)), "1 and 2 cannot share a route");
            assert!(lateness_sum(route, &dist, &earliest, &latest, &service, depot) <= 0, "each route meets its hard windows");
        }

        // LS-vs-exact: local search reaches the same optimum on this enumerable model.
        let ls = crate::engines::ls::lists::solve_collection_capped(&model, 0, &stop, 50_000, None, &mut |_| {});
        assert!(ls.feasible, "LS finds a feasible plan");
        assert_eq!(ls.objectives, vec![brute], "LS agrees with the exact optimum");
    }

    #[test]
    fn scan_bound_is_per_route_not_fleet_total() {
        // Every arc costs 1; each customer needs service 1 and has deadline 1, so a
        // route's k-th stop finishes at time 2k and is (2k-1) late. Any 2-2 split
        // gives each route lateness 1+3 = 4 <= 5 (feasible per route) but a fleet
        // total of 8 > 5. A global-sum lowering would wrongly reject every split and
        // return UNSAT; a correct per-route lowering keeps it feasible.
        let dist = vec![
            vec![0, 1, 1, 1, 1],
            vec![1, 0, 1, 1, 1],
            vec![1, 1, 0, 1, 1],
            vec![1, 1, 1, 0, 1],
            vec![1, 1, 1, 1, 0],
        ];
        let service = vec![0i64, 1, 1, 1, 1];
        let earliest = vec![0i64, 0, 0, 0, 0];
        let latest = vec![0i64, 1, 1, 1, 1];
        let demand = vec![0i64; 5];
        let items = vec![1, 2, 3, 4];
        let depot = 0;
        let rhs = 5;
        let brute = min_two_routes_scan(&items, &dist, &demand, None, &earliest, &latest, &service, depot, Op::Le, rhs).expect("feasible");

        let dist_arc = Arc::new(dist.clone());
        let mut constraints = Vec::new();
        for l in 0..2 {
            constraints.push(Constraint { reduction: lateness_scan(l, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs });
        }
        let model = CollectionModel {
            items: items.clone(),
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0), scan_edge(&dist_arc, 1)], max_terms: None }],
            constraints,
            globals: Vec::new(),
            schedule: None,
        };

        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported per-route scan lowering");
        assert!(outcome.complete, "exhausted the search");
        assert!(outcome.solution.feasible, "a per-route-feasible plan exists (a global-sum lowering would be UNSAT here)");
        assert_eq!(outcome.solution.objectives, vec![brute], "exact optimum equals the per-route-filtered minimum");
        let sums: Vec<i64> = outcome.solution.lists.iter().map(|r| lateness_sum(r, &dist, &earliest, &latest, &service, depot)).collect();
        assert!(sums.iter().all(|&s| s <= rhs), "every route respects the per-route bound");
        assert!(sums.iter().sum::<i64>() > rhs, "the fleet total exceeds rhs: the scan is enforced per route, not globally");
    }

    #[test]
    fn scan_single_route_unsat() {
        // One vehicle, two customers, each with deadline 1 and service 1 over unit
        // arcs: whichever is served first already finishes at time 2 (1 late), so no
        // ordering meets the hard windows. The exact search must prove UNSAT.
        let dist = vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]];
        let service = vec![0i64, 1, 1];
        let earliest = vec![0i64, 0, 0];
        let latest = vec![0i64, 1, 1];
        let items = vec![1, 2];
        let depot = 0;

        let dist_arc = Arc::new(dist.clone());
        let model = CollectionModel {
            items: items.clone(),
            lists: 1,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0)], max_terms: None }],
            constraints: vec![Constraint { reduction: lateness_scan(0, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs: 0 }],
            globals: Vec::new(),
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported single-route scan lowering");
        assert!(outcome.complete, "exhausted the search");
        assert!(!outcome.solution.feasible, "no ordering meets both hard windows");
    }

    #[test]
    fn scan_multiple_per_route_compose() {
        // Two identical lateness scans per route (the Python API posts one scan per
        // customer time window, so a route commonly carries several). The exact
        // optimum must still equal the single-scan brute-force minimum -- the extra
        // resource thread must not corrupt the result.
        let dist = line_dist(&[0, 2, 4, 6, 8]);
        let demand = vec![0i64, 1, 1, 1, 1];
        let service = vec![0i64, 1, 1, 1, 1];
        let earliest = vec![0i64, 0, 0, 0, 0];
        let latest = vec![0i64, 5, 5, 20, 20];
        let items = vec![1, 2, 3, 4];
        let depot = 0;
        let brute =
            min_two_routes_scan(&items, &dist, &demand, Some(2), &earliest, &latest, &service, depot, Op::Le, 0).expect("feasible plan exists");

        let dist_arc = Arc::new(dist.clone());
        let demand_arc = Arc::new(demand.clone());
        let mut constraints = vec![scan_load(&demand_arc, 0, 2), scan_load(&demand_arc, 1, 2)];
        for l in 0..2 {
            for _ in 0..2 {
                constraints.push(Constraint { reduction: lateness_scan(l, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs: 0 });
            }
        }
        let model = CollectionModel {
            items: items.clone(),
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0), scan_edge(&dist_arc, 1)], max_terms: None }],
            constraints,
            globals: Vec::new(),
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported multi-scan lowering");
        assert!(outcome.complete, "exhausted the search");
        assert!(outcome.solution.feasible, "a feasible plan exists");
        assert_eq!(outcome.solution.objectives, vec![brute], "duplicate scans are idempotent");
    }

    // --- dispatch (classifier agrees with the engine) ---------------------------

    use crate::model::classify::{Backend, BackendSelection, ModelClass};

    /// A two-route edge-objective model carrying one scan constraint builder per
    /// route, for classifier dispatch checks.
    fn scan_dispatch_model(scan: impl Fn(usize) -> Option<Constraint>) -> CollectionModel {
        let dist_arc = Arc::new(line_dist(&[0, 2, 4, 6, 8]));
        let mut constraints = Vec::new();
        for l in 0..2 {
            if let Some(c) = scan(l) {
                constraints.push(c);
            }
        }
        CollectionModel {
            items: vec![1, 2, 3, 4],
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![scan_edge(&dist_arc, 0), scan_edge(&dist_arc, 1)], max_terms: None }],
            constraints,
            globals: Vec::new(),
            schedule: None,
        }
    }

    fn tw_scan_constraint(l: usize) -> Constraint {
        let dist = line_dist(&[0, 2, 4, 6, 8]);
        let service = vec![0i64, 1, 1, 1, 1];
        let earliest = vec![0i64, 0, 0, 0, 0];
        let latest = vec![0i64, 5, 5, 20, 20];
        Constraint { reduction: lateness_scan(l, &dist, &earliest, &latest, &service, 0), op: Op::Le, rhs: 0 }
    }

    #[test]
    fn dispatch_supported_scan_routes_to_integer_exact() {
        let model = scan_dispatch_model(|l| Some(tw_scan_constraint(l)));
        let selection = BackendSelection::for_collection(&model);
        assert_eq!(selection.class, ModelClass::Routing);
        assert_eq!(selection.backend, Backend::IntegerExact, "a homogeneous lowerable scan routes to the exact engine");
    }

    #[test]
    fn dispatch_multiple_homogeneous_scans_route_to_integer_exact() {
        // Each route carries two identical scans: still homogeneous, still exact.
        let model = scan_dispatch_model(|l| Some(tw_scan_constraint(l)));
        let mut model = model;
        for l in 0..2 {
            model.constraints.push(tw_scan_constraint(l));
        }
        assert_eq!(BackendSelection::for_collection(&model).backend, Backend::IntegerExact);
    }

    #[test]
    fn dispatch_non_homogeneous_scan_falls_back_to_ls() {
        // Scan on route 0 only: route-specific, so unsafe under depot symmetry.
        let model = scan_dispatch_model(|l| (l == 0).then(|| tw_scan_constraint(l)));
        assert_eq!(BackendSelection::for_collection(&model).backend, Backend::ListLocalSearch);
    }

    #[test]
    fn dispatch_acc_in_divisor_scan_falls_back_to_ls() {
        let divisor_scan = |l: usize| {
            let mut arena = ExprArena::default();
            let cur = arena.arg(0);
            let acc = arena.arg(1);
            let step = arena.div(cur, acc); // divides by the accumulator: rejected
            let body = arena.constant(0);
            Some(Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list: l, init: 1, boundary: 0, step }, arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 0,
            })
        };
        assert_eq!(BackendSelection::for_collection(&scan_dispatch_model(divisor_scan)).backend, Backend::ListLocalSearch);
    }

    #[test]
    fn dispatch_zero_capable_divisor_scan_falls_back_to_ls() {
        // `emit = cur / rate[cur]` with `rate[1] == 0`: LS reads the zero divisor
        // as the value 0, but the kernel `Div` makes that stop infeasible, so the
        // two engines would disagree. The scan must fall back to LS, not lower.
        let zero_div = |l: usize| {
            let mut arena = ExprArena::default();
            let cur = arena.arg(0);
            let acc = arena.arg(1);
            let step = arena.add(acc, cur);
            let cur_e = arena.arg(0);
            let rate = arena.array(Arc::new(vec![9, 0, 2, 2, 2]), cur_e); // rate[item 1] == 0
            let body = arena.div(cur_e, rate);
            Some(Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list: l, init: 0, boundary: 0, step }, arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 100,
            })
        };
        assert_eq!(BackendSelection::for_collection(&scan_dispatch_model(zero_div)).backend, Backend::ListLocalSearch);
    }

    #[test]
    fn dispatch_nonzero_const_divisor_scan_stays_exact() {
        // The div guard must not over-reject: dividing by a nonzero constant lowers
        // cleanly (div-by-const with the accumulator in the numerator is supported).
        let const_div = |l: usize| {
            let mut arena = ExprArena::default();
            let cur = arena.arg(0);
            let acc = arena.arg(1);
            let step = arena.add(acc, cur);
            let acc_e = arena.arg(1);
            let two = arena.constant(2);
            let body = arena.div(acc_e, two);
            Some(Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list: l, init: 0, boundary: 0, step }, arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 100,
            })
        };
        assert_eq!(BackendSelection::for_collection(&scan_dispatch_model(const_div)).backend, Backend::IntegerExact);
    }

    #[test]
    fn dispatch_oversized_table_scan_falls_back_to_ls() {
        // A table value beyond the i32-safe band could overflow the kernel's plain
        // i64 `Expr::eval` on an intermediate (LS saturates), so the scan falls back.
        let big_table = |l: usize| {
            let mut arena = ExprArena::default();
            let acc = arena.arg(1);
            let cur_s = arena.arg(0);
            let huge = arena.array(Arc::new(vec![0, i64::MAX, 0, 0, 0]), cur_s);
            let step = arena.add(acc, huge);
            let body = arena.arg(1);
            Some(Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list: l, init: 0, boundary: 0, step }, arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 100,
            })
        };
        assert_eq!(BackendSelection::for_collection(&scan_dispatch_model(big_table)).backend, Backend::ListLocalSearch);
    }

    #[test]
    fn signature_rejects_zero_capable_and_oversized_leaves() {
        // Direct predicate checks: `scan_routing_signature` returns `None` for a
        // zero-capable divisor and for an out-of-band table leaf.
        let items = vec![1, 2, 3, 4];
        let zero_div = {
            let mut arena = ExprArena::default();
            let cur = arena.arg(0);
            let acc = arena.arg(1);
            let step = arena.add(acc, cur);
            let cur_e = arena.arg(0);
            let rate = arena.array(Arc::new(vec![9, 0, 2, 2, 2]), cur_e);
            let body = arena.div(cur_e, rate);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list: 0, init: 0, boundary: 0, step }, arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 100,
            }
        };
        assert!(scan_routing_signature(&zero_div, &items).is_none(), "zero-capable divisor is rejected");
    }

    #[test]
    fn dispatch_select_kth_scan_falls_back_to_ls() {
        let kth_scan = |l: usize| {
            let mut arena = ExprArena::default();
            let cur = arena.arg(0);
            let acc = arena.arg(1);
            let step = arena.add(acc, cur);
            let body = arena.arg(1);
            Some(Constraint {
                reduction: Reduction {
                    op: ReduceOp::SelectKth(0),
                    iterable: Iterable::Scan { list: l, init: 0, boundary: 0, step },
                    arena,
                    body,
                    coeff: 1,
                },
                op: Op::Le,
                rhs: 0,
            })
        };
        assert_eq!(BackendSelection::for_collection(&scan_dispatch_model(kth_scan)).backend, Backend::ListLocalSearch);
    }

    #[test]
    fn dispatch_out_of_i32_range_scan_falls_back_to_ls() {
        // acc multiplies by 100000 each step, leaving the i32-safe band immediately.
        let huge_scan = |l: usize| {
            let mut arena = ExprArena::default();
            let acc = arena.arg(1);
            let big = arena.constant(100_000);
            let step = arena.mul(acc, big);
            let body = arena.constant(0);
            Some(Constraint {
                reduction: Reduction {
                    op: ReduceOp::Sum,
                    iterable: Iterable::Scan { list: l, init: 100_000, boundary: 0, step },
                    arena,
                    body,
                    coeff: 1,
                },
                op: Op::Le,
                rhs: 0,
            })
        };
        assert_eq!(BackendSelection::for_collection(&scan_dispatch_model(huge_scan)).backend, Backend::ListLocalSearch);
    }

    // --- optional-list pool oracles (G9) ----------------------------------------
    //
    // The pool depot forms a zero-cost route that absorbs dropped items. Every test
    // asserts a three-way equality: engine integer optimum == exact list enumeration
    // optimum == an independent hand brute-force where each item goes to one of the
    // k real routes OR the pool (cost 0). This is the primary defence against a
    // silently mis-gated pooled item.

    /// Best closed-route value for one real route under a pool: the minimum
    /// closed-tour cost over orderings passing `filter`, minus a constant reward per
    /// served stop. `None` if no ordering passes the filter; empty route scores 0.
    fn best_pool_route(depot: i32, route: &[i32], dist: &[Vec<i64>], reward: i64, filter: &dyn Fn(&[i32]) -> bool) -> Option<i64> {
        if route.is_empty() {
            return Some(0);
        }
        permutations(route)
            .iter()
            .filter(|p| filter(p))
            .map(|p| ordered_closed_cost(depot, p, dist))
            .min()
            .map(|cost| cost - reward * route.len() as i64)
    }

    /// Exhaustive optimum for a `k`-real-route optional model: each item goes to one
    /// of `k` real routes OR the pool (contributing 0). Mirrors the in_pool gating.
    fn min_pool(depot: i32, items: &[i32], dist: &[Vec<i64>], k: usize, reward: i64, filter: &dyn Fn(&[i32]) -> bool) -> i64 {
        let choices = k + 1;
        let combos = (choices as u64).pow(items.len() as u32);
        let mut best = i64::MAX;
        for combo in 0..combos {
            let mut routes = vec![Vec::new(); k];
            let mut code = combo;
            for &item in items {
                let choice = (code % choices as u64) as usize;
                code /= choices as u64;
                if choice < k {
                    routes[choice].push(item);
                }
            }
            let total: Option<i64> = routes.iter().map(|r| best_pool_route(depot, r, dist, reward, filter)).sum();
            if let Some(total) = total {
                best = best.min(total);
            }
        }
        best
    }

    /// The exact list-enumeration optimum for a pool model (the G8 oracle). Asserts
    /// the instance is enumerable (proven OPTIMAL), scoring only the objective-
    /// referenced real lists so the free pool list contributes 0.
    fn enum_pool_optimum(model: &CollectionModel) -> i64 {
        let tiers = crate::model::list_objective_tiers(&model.objectives, &model.items).expect("objective parses for enumeration");
        let stop = AtomicBool::new(false);
        let outcome = crate::engines::list_exact::solve(model, &tiers, &stop, |_| {}).expect("enumeration runs").expect("enumeration supports the model");
        assert_eq!(outcome.status, crate::engines::list_exact::Status::Optimal, "enumeration proves optimality on the small instance");
        outcome.objectives[0]
    }

    /// A `k`-real-route pool model whose objective is edges only (list `k` is the
    /// free pool, referenced by nothing).
    fn pool_edge_model(items: Vec<i32>, dist: &Arc<Vec<Vec<i64>>>, k: usize) -> CollectionModel {
        let terms = (0..k).map(|list| scan_edge(dist, list)).collect();
        CollectionModel { items, lists: k + 1, objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }], constraints: Vec::new(), globals: Vec::new(), schedule: None }
    }

    /// A reified scan objective term paying a constant `reward` per served stop
    /// (`emit = -reward`, order-independent). Its `Sum` over a real route is
    /// `-reward * length`, so summing the served customers rewards serving them.
    fn reward_scan(list: usize, reward: i64) -> Reduction {
        let mut arena = ExprArena::default();
        let step = arena.arg(1); // accumulator threaded unchanged (unused)
        let emit = arena.constant(-reward);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list, init: 0, boundary: 0, step }, arena, body: emit, coeff: 1 }
    }

    /// A `k`-real-route pool model: edges plus a reified reward scan per real route.
    fn pool_reward_model(items: Vec<i32>, dist: &Arc<Vec<Vec<i64>>>, k: usize, reward: i64) -> CollectionModel {
        let mut terms: Vec<Reduction> = (0..k).map(|list| scan_edge(dist, list)).collect();
        terms.extend((0..k).map(|list| reward_scan(list, reward)));
        CollectionModel { items, lists: k + 1, objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }], constraints: Vec::new(), globals: Vec::new(), schedule: None }
    }

    fn allow_all(_: &[i32]) -> bool {
        true
    }

    #[test]
    fn pool_all_items_pooled_is_zero() {
        // No reward, every served edge strictly positive: dropping everything is
        // optimal. The whole pool route must be zero-cost (optimum 0).
        let dist = vec![vec![0, 7, 9, 8], vec![7, 0, 5, 6], vec![9, 5, 0, 4], vec![8, 6, 4, 0]];
        let items = vec![1, 2, 3];
        let brute = min_pool(0, &items, &dist, 1, 0, &allow_all);
        assert_eq!(brute, 0, "with no reward the best plan pools every item");
        let model = pool_edge_model(items.clone(), &Arc::new(dist), 1);
        let enumerated = enum_pool_optimum(&model);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported pool lowering");
        assert!(outcome.complete, "exhausted the search");
        assert!(outcome.solution.feasible);
        assert_eq!(outcome.solution.objectives, vec![0], "engine optimum is zero");
        assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
        assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
        assert!(outcome.solution.lists[0].is_empty(), "the real route is empty");
        let mut pooled = outcome.solution.lists[1].clone();
        pooled.sort_unstable();
        assert_eq!(pooled, items, "every item lands in the pool list");
    }

    #[test]
    fn pool_single_node_serve_or_drop() {
        // One customer, round trip 20. Reward 25 > 20 -> serve (net -5); reward 5 <
        // 20 -> drop (optimum 0). The gate must flip with the reward.
        let dist = vec![vec![0, 10], vec![10, 0]];
        let items = vec![1];

        let served_brute = min_pool(0, &items, &dist, 1, 25, &allow_all);
        assert_eq!(served_brute, -5);
        let served = pool_reward_model(items.clone(), &Arc::new(dist.clone()), 1, 25);
        let enumerated = enum_pool_optimum(&served);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&served, 0, &stop, &mut |_| {}).expect("supported");
        assert!(outcome.complete);
        assert_eq!(outcome.solution.objectives[0], served_brute, "serve: engine == brute");
        assert_eq!(outcome.solution.objectives[0], enumerated, "serve: engine == enumeration");
        assert_eq!(outcome.solution.lists[0], vec![1], "the customer is served");

        let dropped_brute = min_pool(0, &items, &dist, 1, 5, &allow_all);
        assert_eq!(dropped_brute, 0);
        let dropped = pool_reward_model(items.clone(), &Arc::new(dist), 1, 5);
        let enumerated = enum_pool_optimum(&dropped);
        let outcome = solve_collection(&dropped, 0, &stop, &mut |_| {}).expect("supported");
        assert!(outcome.complete);
        assert_eq!(outcome.solution.objectives[0], dropped_brute, "drop: engine == brute");
        assert_eq!(outcome.solution.objectives[0], enumerated, "drop: engine == enumeration");
        assert!(outcome.solution.lists[0].is_empty(), "the customer is pooled");
        assert_eq!(outcome.solution.lists[1], vec![1]);
    }

    #[test]
    fn pool_reward_scan_changes_optimum_and_matches_oracles() {
        // Customers on a line at positions 1, 2, 10 (depot 0). Reward 6/stop makes
        // the near cluster {1,2} profitable but the far one a loss, so the optimum
        // serves some and pools some -- and beats the edges-only optimum of 0,
        // proving the reified scan term actually bites (exercises the gating).
        let dist = line_dist(&[0, 1, 2, 10]);
        let items = vec![1, 2, 3];
        let reward = 6;
        let brute = min_pool(0, &items, &dist, 1, reward, &allow_all);
        assert!(brute < 0, "the reward makes serving profitable, changing the optimum from the edges-only 0");

        let model = pool_reward_model(items, &Arc::new(dist), 1, reward);
        let enumerated = enum_pool_optimum(&model);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported pool + reward lowering");
        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
        assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
        assert!(!outcome.solution.lists[0].is_empty(), "some customers are served");
        assert!(!outcome.solution.lists[1].is_empty(), "some customers are pooled");
    }

    #[test]
    fn pool_multi_route_matches_oracles() {
        // Two real routes + pool. Two near clusters go to the two routes; a far
        // customer is pooled. Reward 8/stop.
        let dist = line_dist(&[0, 1, 2, 20, 21, 50]);
        let items = vec![1, 2, 3, 4, 5];
        let reward = 8;
        let brute = min_pool(0, &items, &dist, 2, reward, &allow_all);
        let model = pool_reward_model(items.clone(), &Arc::new(dist), 2, reward);
        let enumerated = enum_pool_optimum(&model);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported multi-route pool");
        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
        assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
        let mut served: Vec<i32> = outcome.solution.lists[..2].iter().flatten().copied().collect();
        served.sort_unstable();
        let mut pooled = outcome.solution.lists[2].clone();
        pooled.sort_unstable();
        let mut all: Vec<i32> = served.iter().chain(pooled.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, items, "every item is either served or pooled exactly once");
    }

    #[test]
    fn pool_empty_real_route_matches_oracles() {
        // Two real routes but only one near cluster is profitable, so one real route
        // stays empty and the far customer is pooled.
        let dist = line_dist(&[0, 1, 2, 50]);
        let items = vec![1, 2, 3];
        let reward = 6;
        let brute = min_pool(0, &items, &dist, 2, reward, &allow_all);
        let model = pool_reward_model(items, &Arc::new(dist), 2, reward);
        let enumerated = enum_pool_optimum(&model);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported pool with empty route");
        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
        assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
        let empty_real_routes = outcome.solution.lists[..2].iter().filter(|route| route.is_empty()).count();
        assert!(empty_real_routes >= 1, "at least one real route is empty");
    }

    #[test]
    fn pool_scan_constraint_and_reward_objective_matches_oracles() {
        // The combined decisive case: optional + edges + a scan CONSTRAINT (hard
        // time windows) + a reified scan OBJECTIVE (reward). Customers 1,2 share a
        // deadline they cannot both meet in one route, so the window forbids serving
        // both; the reward still makes serving the reachable ones profitable, and the
        // far ones (3,4) are cheap to reach late. Three-way equality throughout.
        let dist = line_dist(&[0, 2, 4, 6, 8]);
        let service = vec![0i64, 1, 1, 1, 1];
        let earliest = vec![0i64, 0, 0, 0, 0];
        let latest = vec![0i64, 5, 5, 20, 20];
        let items = vec![1, 2, 3, 4];
        let depot = 0;
        let reward = 20;
        let windows = |route: &[i32]| lateness_sum(route, &dist, &earliest, &latest, &service, depot) <= 0;
        let brute = min_pool(depot, &items, &dist, 1, reward, &windows);
        assert!(brute < 0, "serving the reachable customers is profitable under the windows");

        let dist_arc = Arc::new(dist.clone());
        let mut terms = vec![scan_edge(&dist_arc, 0)];
        terms.push(reward_scan(0, reward));
        let model = CollectionModel {
            items: items.clone(),
            lists: 2,
            objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
            constraints: vec![Constraint { reduction: lateness_scan(0, &dist, &earliest, &latest, &service, depot), op: Op::Le, rhs: 0 }],
            globals: Vec::new(),
            schedule: None,
        };
        let enumerated = enum_pool_optimum(&model);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported combined pool lowering");
        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
        assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
        assert!(
            !(outcome.solution.lists[0].contains(&1) && outcome.solution.lists[0].contains(&2)),
            "the window forbids serving both 1 and 2 in the single real route"
        );
        assert!(windows(&outcome.solution.lists[0]), "the served route meets its hard windows");
    }

    #[test]
    fn pool_scale_proves_optimal_where_enumeration_times_out() {
        // 12 customers + a pool, every served edge strictly positive. Enumeration
        // cannot exhaust the ordered partitions within a short budget (SATISFIABLE,
        // best-found still positive); the integer path proves the optimum is 0 (pool
        // everything) from the non-negative edge bound. This is the scale win: an
        // instance enumeration cannot close but the integer path proves OPTIMAL.
        let count = 12usize;
        let positions: Vec<i64> = (0..=count as i64).collect();
        let dist = line_dist(&positions);
        let items: Vec<i32> = (1..=count as i32).collect();
        let model = pool_edge_model(items, &Arc::new(dist), 1);

        // Enumeration under a short wall-clock budget: it finds a plan but cannot
        // prove optimality (10! ordered partitions).
        let tiers = crate::model::list_objective_tiers(&model.objectives, &model.items).expect("objective parses");
        let enum_stop = Arc::new(AtomicBool::new(false));
        {
            let flag = Arc::clone(&enum_stop);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            });
        }
        let enum_start = std::time::Instant::now();
        let enum_out = crate::engines::list_exact::solve(&model, &tiers, &enum_stop, |_| {}).expect("runs").expect("supported");
        let enum_elapsed = enum_start.elapsed();
        assert_eq!(enum_out.status, crate::engines::list_exact::Status::Satisfiable, "enumeration cannot prove optimality in the budget");

        // Integer path: proves OPTIMAL. A safety timeout turns a hang into a clear
        // failure rather than blocking CI.
        let int_stop = Arc::new(AtomicBool::new(false));
        {
            let flag = Arc::clone(&int_stop);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(30));
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            });
        }
        let int_start = std::time::Instant::now();
        let outcome = solve_collection(&model, 0, &int_stop, &mut |_| {}).expect("supported pool lowering");
        let int_elapsed = int_start.elapsed();
        assert!(outcome.complete, "the integer path proves optimality");
        assert_eq!(outcome.solution.objectives[0], 0, "pooling every customer is optimal");
        assert!(
            outcome.solution.objectives[0] <= enum_out.objectives[0],
            "the proven optimum is no worse than enumeration's un-proven incumbent"
        );
        eprintln!(
            "SCALE pool: enumeration SATISFIABLE in {:?}, integer OPTIMAL ({}) in {:?}",
            enum_elapsed, outcome.solution.objectives[0], int_elapsed
        );
    }

    #[test]
    fn pool_classify_and_from_model_agree() {
        // Lockstep: classify-IntegerExact <=> RoutingSpec::from_model.is_some(), over
        // a battery covering the pool shapes and the must-enumerate exclusions. The
        // shared parse plus this test plus the enumeration fall-through mean a gap
        // can never yield a wrong optimum -- only the exact enumeration optimum.
        let dist4 = Arc::new(vec![vec![0, 7, 9, 8], vec![7, 0, 5, 6], vec![9, 5, 0, 4], vec![8, 6, 4, 0]]);
        let dist_line = Arc::new(line_dist(&[0, 2, 4, 6, 8]));
        let demand = Arc::new(vec![0i64, 1, 1, 1, 1]);

        // A pool + capacity model (must route to enumeration, not integer).
        let mut cap_pool = pool_reward_model(vec![1, 2, 3, 4], &dist_line, 1, 10);
        cap_pool.constraints.push(scan_load(&demand, 0, 2));

        // A pool + route-specific scan (non-homogeneous over the two real routes).
        let mut route_specific = pool_reward_model(vec![1, 2, 3, 4], &dist_line, 2, 10);
        route_specific.constraints.push(Constraint {
            reduction: lateness_scan(0, &line_dist(&[0, 2, 4, 6, 8]), &[0, 0, 0, 0, 0], &[0, 5, 5, 20, 20], &[0, 1, 1, 1, 1], 0),
            op: Op::Le,
            rhs: 0,
        });

        let battery: Vec<CollectionModel> = vec![
            pool_edge_model(vec![1, 2, 3], &dist4, 1),                                  // pool, single route
            pool_edge_model(vec![1], &dist4, 1),                                        // pool, single node
            pool_edge_model(vec![1, 2, 3], &dist4, 2),                                  // pool, multi route
            pool_reward_model(vec![1, 2, 3], &Arc::new(line_dist(&[0, 1, 2, 10])), 1, 6), // pool + reward objective
            pool_reward_model(vec![1, 2, 3, 4, 5], &Arc::new(line_dist(&[0, 1, 2, 20, 21, 50])), 2, 8), // pool + reward, multi
            cap_pool,                                                                   // pool + capacity -> enumeration
            route_specific,                                                             // pool + route-specific scan -> enumeration
            closed_tsp_model(vec![1, 2, 3], 0, &dist4),                                 // non-pool TSP
        ];

        for model in &battery {
            let classify_integer = BackendSelection::for_collection(model).backend == Backend::IntegerExact;
            let engine_lowers = RoutingSpec::from_model(model).is_some();
            assert_eq!(classify_integer, engine_lowers, "classifier and engine must agree on lowerability");
        }
        // The two exclusions really are excluded, and really do route to enumeration.
        assert!(RoutingSpec::from_model(&battery[5]).is_none(), "pool + capacity is not integer-lowerable");
        assert_eq!(BackendSelection::for_collection(&battery[5]).backend, Backend::DomainExact, "pool + capacity routes to enumeration");
        assert!(RoutingSpec::from_model(&battery[6]).is_none(), "route-specific scan is not integer-lowerable");
    }

    /// Per-route cumulative-load scan `Σ prefix_load`, as a constraint reduction.
    /// `emit = acc` (running load after each stop); `step = acc + item_value`.
    fn prefix_load_scan(list: usize) -> Reduction {
        let mut arena = ExprArena::default();
        let cur = arena.arg(0);
        let acc = arena.arg(1);
        let step = arena.add(acc, cur);
        let emit = arena.arg(1); // the new accumulator = load after this stop
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list, init: 0, boundary: 0, step }, arena, body: emit, coeff: 1 }
    }

    #[test]
    fn pool_scan_constraint_ge_matches_oracle() {
        // A `Ge` scan constraint on a pool model (only `Le` was covered): each real
        // route's cumulative load must reach at least `rhs`, so an empty (all-pooled)
        // route is forbidden when rhs > 0. A reward objective incentivises serving,
        // and three-way equality must hold including this feasibility coupling.
        let dist = line_dist(&[0, 1, 2, 3]);
        let items = vec![1, 2, 3];
        let reward = 12;
        let rhs = 3;
        let dist_arc = Arc::new(dist.clone());
        let mut model = pool_reward_model(items.clone(), &dist_arc, 1, reward);
        model.constraints.push(Constraint { reduction: prefix_load_scan(0), op: Op::Ge, rhs });

        // Hand brute-force: the real route's Σ prefix-load must be >= rhs.
        let prefix_ok = |route: &[i32]| {
            let (mut acc, mut total) = (0i64, 0i64);
            for &c in route {
                acc += i64::from(c);
                total += acc;
            }
            route.is_empty() || total >= rhs
        };
        let brute = min_pool(0, &items, &dist, 1, reward, &prefix_ok);

        let enumerated = enum_pool_optimum(&model);
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported Ge-scan pool lowering");
        assert!(outcome.complete, "exhausted the search");
        assert_eq!(outcome.solution.objectives[0], brute, "engine == hand brute-force");
        assert_eq!(outcome.solution.objectives[0], enumerated, "engine == enumeration");
        assert!(prefix_ok(&outcome.solution.lists[0]), "the served route meets the Ge load floor");
    }

    #[test]
    fn pool_random_instances_match_enumeration() {
        // Broad differential defense for the pool gating: on random small pool
        // instances (varied topology, item count, route count, reward), the integer
        // optimum must equal the enumeration optimum. Catches gating bugs in
        // combinations the hand-written cases do not enumerate. Fixed seed, trimmed
        // to stay CI-cheap while covering the pool cost channels.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for _ in 0..50 {
            let n = 2 + (next() % 3) as usize; // 2..=4 customers
            let k = 1 + (next() % 2) as usize; // 1..=2 real routes
            let m = n + 1; // node universe: depot + customers
            let mut dist = vec![vec![0i64; m]; m];
            #[allow(clippy::needless_range_loop)] // fills dist[a][b] over the full grid
            for a in 0..m {
                for b in 0..m {
                    if a != b {
                        dist[a][b] = 1 + (next() % 9) as i64; // strictly positive
                    }
                }
            }
            let items: Vec<i32> = (1..=n as i32).collect();
            let reward = (next() % 12) as i64; // 0..=11; 0 => pool everything
            let model = pool_reward_model(items, &Arc::new(dist), k, reward);

            let enumerated = enum_pool_optimum(&model);
            let stop = AtomicBool::new(false);
            let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported random pool lowering");
            assert!(outcome.complete, "integer path proves optimality on the small instance");
            assert_eq!(outcome.solution.objectives[0], enumerated, "integer optimum == enumeration optimum (n={n}, k={k}, reward={reward})");
        }
    }

    /// A reified scan objective paying a per-item reward: `emit = table[cur]`, so a
    /// route's `Sum` is the total reward of its served customers (each pooled one is
    /// gated to 0). Distinct from the order-independent constant `reward_scan`.
    fn item_reward_scan(list: usize, reward_table: &Arc<Vec<i64>>) -> Reduction {
        let mut arena = ExprArena::default();
        let step = arena.arg(1); // accumulator threaded unchanged (unused)
        let cur = arena.arg(0);
        let emit = arena.array(Arc::clone(reward_table), cur);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list, init: 0, boundary: 0, step }, arena, body: emit, coeff: 1 }
    }

    /// The same per-item reward as `item_reward_scan` (`emit = table[cur]`), but
    /// expressed as a `Sum` over `Items` (`cp.sum(x, |i| table[i])`) -- the CIBLE 1
    /// objective shape. Must normalise to the identical identity-accumulator scan.
    fn item_reward_items(list: usize, reward_table: &Arc<Vec<i64>>) -> Reduction {
        let mut arena = ExprArena::default();
        let cur = arena.arg(0);
        let emit = arena.array(Arc::clone(reward_table), cur);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body: emit, coeff: 1 }
    }

    /// The G10 repro as a model: one real route + pool, required customers `1..=n`
    /// (each paying a huge serve reward `-reward`) and disposable optionals
    /// `n+1..=n+m` (zero reward, pure positive detour on the line). Objective =
    /// travel - reward per served required.
    fn disposable_model(n: usize, m: usize, reward: i64) -> CollectionModel {
        let total = n + m;
        let positions: Vec<i64> = (0..=total as i64).collect();
        let dist = Arc::new(line_dist(&positions));
        let mut reward_table = vec![0i64; total + 1];
        for slot in reward_table.iter_mut().take(n + 1).skip(1) {
            *slot = -reward;
        }
        let reward_table = Arc::new(reward_table);
        let items: Vec<i32> = (1..=total as i32).collect();
        let terms = vec![scan_edge(&dist, 0), item_reward_scan(0, &reward_table)];
        CollectionModel { items, lists: 2, objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }], constraints: Vec::new(), globals: Vec::new(), schedule: None }
    }

    /// `disposable_model` with the per-served reward expressed as an `Items` sum
    /// (`cp.sum(x, |i| table[i])`) rather than a scan -- the CIBLE 1 objective shape.
    fn disposable_items_model(n: usize, m: usize, reward: i64) -> CollectionModel {
        let total = n + m;
        let positions: Vec<i64> = (0..=total as i64).collect();
        let dist = Arc::new(line_dist(&positions));
        let mut reward_table = vec![0i64; total + 1];
        for slot in reward_table.iter_mut().take(n + 1).skip(1) {
            *slot = -reward;
        }
        let reward_table = Arc::new(reward_table);
        let items: Vec<i32> = (1..=total as i32).collect();
        let terms = vec![scan_edge(&dist, 0), item_reward_items(0, &reward_table)];
        CollectionModel { items, lists: 2, objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }], constraints: Vec::new(), globals: Vec::new(), schedule: None }
    }

    #[test]
    fn cible1_items_reward_disposables_prove_optimal() {
        // CIBLE 1: disposable optionals under a per-item `Items` reward
        // (`cp.sum(x, |i| REQ[i]) * -reward`) must prove OPTIMAL, not just
        // SATISFIABLE -- the normalisation feeds the same dominance bound the scan
        // reward gets. The M=n optimum equals the M=0 optimum (disposables dropped).
        let reward = 100_000i64;
        for n in [6usize, 8] {
            let stop = AtomicBool::new(false);
            let v0 = solve_collection(&disposable_items_model(n, 0, reward), 0, &stop, &mut |_| {}).expect("M=0 lowering");
            assert!(v0.complete, "M=0 proves OPTIMAL (n={n})");
            let vn = solve_collection(&disposable_items_model(n, n, reward), 0, &stop, &mut |_| {}).expect("M=n lowering");
            assert!(vn.complete, "M=n proves OPTIMAL via the items-reward dominance bound (n={n})");
            assert_eq!(vn.solution.objectives[0], v0.solution.objectives[0], "disposables dropped: M=n optimum == M=0 optimum (n={n})");
        }
    }

    #[test]
    fn pool_items_reward_matches_scan_and_enumeration() {
        // Soundness of the `Items`-reward normalisation: on random small pool
        // instances the per-item `Items` reward must lower to the SAME optimum as
        // the equivalent per-item scan reward AND as enumeration, and prove OPTIMAL.
        // Signed rewards (some serve-profitable, some not) exercise the dominance
        // gate both ways. Fixed seed, trimmed for CI.
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for _ in 0..50 {
            let n = 2 + (next() % 3) as usize; // 2..=4 customers
            let k = 1 + (next() % 2) as usize; // 1..=2 real routes
            let m = n + 1;
            let mut dist = vec![vec![0i64; m]; m];
            #[allow(clippy::needless_range_loop)] // fills dist[a][b] over the full grid
            for a in 0..m {
                for b in 0..m {
                    if a != b {
                        dist[a][b] = 1 + (next() % 9) as i64;
                    }
                }
            }
            let dist = Arc::new(dist);
            let items: Vec<i32> = (1..=n as i32).collect();
            let mut reward_table = vec![0i64; m];
            for slot in reward_table.iter_mut().skip(1) {
                *slot = (next() % 13) as i64 - 6; // -6..=6
            }
            let reward_table = Arc::new(reward_table);
            let mk = |scan: bool| {
                let mut terms: Vec<Reduction> = (0..k).map(|l| scan_edge(&dist, l)).collect();
                terms.extend((0..k).map(|l| if scan { item_reward_scan(l, &reward_table) } else { item_reward_items(l, &reward_table) }));
                CollectionModel { items: items.clone(), lists: k + 1, objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }], constraints: Vec::new(), globals: Vec::new(), schedule: None }
            };
            let items_model = mk(false);
            let enumerated = enum_pool_optimum(&items_model);
            let stop = AtomicBool::new(false);
            let items_out = solve_collection(&items_model, 0, &stop, &mut |_| {}).expect("items-reward lowering");
            let scan_out = solve_collection(&mk(true), 0, &stop, &mut |_| {}).expect("scan-reward lowering");
            assert!(items_out.complete, "items-reward integer path proves optimality (n={n}, k={k})");
            assert_eq!(items_out.solution.objectives[0], enumerated, "items optimum == enumeration (n={n}, k={k})");
            assert_eq!(items_out.solution.objectives[0], scan_out.solution.objectives[0], "items reward == scan reward optimum (n={n}, k={k})");
        }
    }

    #[test]
    fn pool_disposable_optionals_prove_optimal_and_primal_never_worse() {
        // G10 acceptance: DISPOSABLE optionals (zero reward, pure detour) must not
        // collapse the optimality proof. (1) bench(8,8) proves OPTIMAL in seconds;
        // (2) for every (n, m) the incumbent is <= the same model's M=0 optimum
        // (serve required only) -- the primal is never worse than dropping all
        // optionals. Disposables must land in the pool, requireds in the real route.
        let reward = 100_000i64;
        // A safety timeout turns a hang into a clear failure rather than blocking CI.
        let stop = Arc::new(AtomicBool::new(false));
        {
            let flag = Arc::clone(&stop);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(120));
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            });
        }
        let solve = |model: &CollectionModel| -> (i64, bool, std::time::Duration) {
            let start = std::time::Instant::now();
            let out = solve_collection(model, 0, &stop, &mut |_| {}).expect("supported disposable-optional lowering");
            (if out.complete { out.solution.objectives[0] } else { i64::MAX }, out.complete, start.elapsed())
        };
        for n in 5..=8usize {
            let (v_star, m0_complete, _) = solve(&disposable_model(n, 0, reward));
            assert!(m0_complete, "M=0 optimum proves OPTIMAL for n={n}");
            for m in 0..=n {
                let model = disposable_model(n, m, reward);
                let stop_local = AtomicBool::new(false);
                let start = std::time::Instant::now();
                let out = solve_collection(&model, 0, &stop_local, &mut |_| {}).expect("supported");
                let elapsed = start.elapsed();
                eprintln!("G10 n={n} m={m} status={} obj={} time={:?}", if out.complete { "OPTIMAL" } else { "SATISFIABLE" }, out.solution.objectives[0], elapsed);
                assert!(out.complete, "OPTIMAL proven for n={n} m={m} in {elapsed:?}");
                assert!(
                    out.solution.objectives[0] <= v_star,
                    "primal never worse than M=0 optimum: n={n} m={m} got {} vs {v_star}",
                    out.solution.objectives[0]
                );
                for &c in &out.solution.lists[0] {
                    assert!((c as usize) <= n, "only required customers are served: n={n} m={m} served {c}");
                }
                let pooled = &out.solution.lists[1];
                for d in (n + 1)..=(n + m) {
                    assert!(pooled.contains(&(d as i32)), "disposable {d} lands in the pool: n={n} m={m}");
                }
            }
        }
    }
}
