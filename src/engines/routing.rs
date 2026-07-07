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
}

/// The model's homogeneous per-route side constraints, classified.
struct RouteSideConstraints {
    capacity: Option<CapacitySpec>,
    length: Option<(usize, usize)>,
    item_sum: Option<ItemSumSpec>,
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
        // Only label-free `SameList` globals are channeled (others -- e.g. ListLe --
        // are route-named and unsafe under depot-copy symmetry, so leave them to LS).
        let mut same_list = Vec::new();
        for global in &model.globals {
            match *global {
                GlobalConstraint::SameList { a, b } => same_list.push((a, b)),
                _ => return None,
            }
        }
        let node_count = model.lists.checked_add(model.items.len())?;
        if node_count > MAX_INTEGER_ROUTING_NODES {
            return None;
        }
        let tier = &model.objectives[0];
        if !tier.minimize || tier.terms.len() != model.lists {
            return None;
        }
        let (depot, edge) = parse_route_edge_objective(&tier.terms, model.lists)?;
        if model.items.contains(&depot) || !items_are_unique(&model.items) {
            return None;
        }
        // SameList references items that must be in the universe.
        if same_list.iter().any(|&(a, b)| !model.items.contains(&a) || !model.items.contains(&b)) {
            return None;
        }
        let mut nodes = Vec::with_capacity(model.items.len() + 1);
        nodes.push(depot);
        nodes.extend_from_slice(&model.items);
        let sides = parse_route_side_constraints(model)?;
        edge.covers_i32_costs(&nodes).then_some(Self {
            depot_count: model.lists,
            depot,
            items: model.items.clone(),
            edge,
            capacity: sides.capacity,
            same_list,
            length: sides.length,
            item_sum: sides.item_sum,
        })
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

        let mut cost_vars = Vec::with_capacity(n);
        for (from_idx, &from_node) in nodes.iter().enumerate() {
            let row = nodes.iter().map(|&to_node| self.edge.cost(from_node, to_node).expect("validated edge cost")).collect::<Vec<_>>();
            let cost = solver.new_var_set(&row);
            primitives::element_const(&mut solver, &row, succ[from_idx], cost);
            cost_vars.push(cost);
        }

        if let Some(capacity) = &self.capacity {
            let load = self.post_capacity_loads(&mut solver, &succ, capacity);
            post_route_capacity_check(&mut solver, &succ, self.depot_count, self.node_demands(capacity), capacity.capacity);
            debug_assert_eq!(load.len(), n);
        }

        post_depot_symmetry(&mut solver, &succ, self.depot_count, n);

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
        let mut routes: Vec<&Vec<i32>> = ls_lists.iter().collect();
        routes.sort_by_key(|route| route.first().and_then(|&item| item_node(item)).unwrap_or(usize::MAX));

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
    // `QAYD_ROUTING_WARMSTART=0` disables it (ablation).
    let warm = if std::env::var("QAYD_ROUTING_WARMSTART").as_deref() != Ok("0") {
        // A bounded local-search pass: enough to find a good incumbent, capped so
        // it terminates even without a stop flag and leaves the exact search the
        // bulk of the time.
        let mut ignore_warm_start_progress = |_| {};
        Some(ls_lists::solve_collection_capped(model, seed, stop, WARM_START_ITERS, &mut ignore_warm_start_progress))
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
            let routes =
                routes_from_successors(&nodes, depot_count, &assignment[..nodes.len()]).expect("solved circuit decodes into k routes");
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
fn post_depot_symmetry(solver: &mut Solver, succ: &[VarId], depot_count: usize, n: usize) {
    if depot_count < 2 {
        return;
    }
    let sentinel = n as i64;
    let head: Vec<VarId> = succ[..depot_count]
        .iter()
        .map(|&s| {
            let h = solver.new_var_range(depot_count as i32, n as i32);
            // head = (succ < depot_count) ? sentinel : succ   (empty route -> sentinel)
            let def = expr::ite(expr::lt(expr::var(s), expr::int(depot_count as i64)), expr::int(sentinel), expr::var(s));
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

fn parse_route_edge_objective(terms: &[Reduction], list_count: usize) -> Option<(i32, DirectEdgeMatrix)> {
    let mut by_list: Vec<Option<(i32, DirectEdgeMatrix)>> = (0..list_count).map(|_| None).collect();
    for term in terms {
        let (list, depot, edge) = direct_closed_edge_term(term)?;
        if list >= list_count || by_list[list].is_some() {
            return None;
        }
        by_list[list] = Some((depot, edge));
    }

    let mut parsed = by_list.into_iter();
    let (depot, edge) = parsed.next()??;
    for item in parsed {
        let (other_depot, other_edge) = item?;
        if other_depot != depot || !edge.same_matrix_as(&other_edge) {
            return None;
        }
    }
    Some((depot, edge))
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
        } else {
            return None;
        }
    }
    Some(RouteSideConstraints {
        capacity: homogeneous(&cap_by_list)?,
        length: homogeneous(&len_by_list)?,
        item_sum: homogeneous(&sum_by_list)?,
    })
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

    use super::{routes_from_successors, solve_collection, CapacityCheck, RoutingSpec};

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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![reduction] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![reduction] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)] }],
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
            objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)] }],
            constraints: vec![load(0), load(1)],
            globals: Vec::new(),
            schedule: None,
        };
        let stop = AtomicBool::new(false);
        let outcome = solve_collection(&model, 0, &stop, &mut |_| {}).expect("supported cvrp lowering");

        assert!(outcome.complete, "exhausted the search");
        assert!(!outcome.solution.feasible, "capacity overflow makes the model infeasible");
    }
}
