//! Exact integer lowering for the small routing core.
//!
//! This handles closed routes with direct matrix edge-sum objectives:
//! `sum_edges(route, (i, j) => dist[i][j], start=d, end=d)`. Multi-route models
//! use replicated depot nodes and one Hamiltonian `circuit`. Optional identical
//! per-route capacities are lowered with MTZ-style load bounds plus a final
//! fixed-successor checker. Time windows and scan-style state remain with the
//! list local-search fallback until they have sound integer views.

#![cfg_attr(not(feature = "python"), allow(dead_code))]

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::model::list::{CollectionModel, CollectionSolution, Constraint, Expr, ExprId, Iterable, Op, ReduceOp, Reduction};
use crate::constraints::graph;
use crate::constraints::intension;
use crate::constraints::primitives;
use crate::expr;
use crate::ids::{PropId, VarId};
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::search::{self, Objective as SearchObjective, SolveStats};
use crate::store::{Solver, Store};

const MAX_INTEGER_ROUTING_NODES: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatrixOrientation {
    Direct,
    Reversed,
}

struct DirectEdgeMatrix {
    values: Arc<Vec<Vec<i64>>>,
    orientation: MatrixOrientation,
}

struct CapacitySpec {
    demands: Vec<i32>,
    capacity: i32,
}

struct RoutingSpec {
    depot_count: usize,
    depot: i32,
    items: Vec<i32>,
    edge: DirectEdgeMatrix,
    capacity: Option<CapacitySpec>,
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
        if model.schedule.is_some() || model.lists == 0 || !model.globals.is_empty() || model.objectives.len() != 1 {
            return None;
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
        let mut nodes = Vec::with_capacity(model.items.len() + 1);
        nodes.push(depot);
        nodes.extend_from_slice(&model.items);
        let capacity = parse_capacity_constraints(model)?;
        edge.covers_i32_costs(&nodes).then_some(Self { depot_count: model.lists, depot, items: model.items.clone(), edge, capacity })
    }

    fn lower(&self) -> LoweredProblem {
        let mut solver = Solver::new();
        let mut nodes = Vec::with_capacity(self.depot_count + self.items.len());
        nodes.extend((0..self.depot_count).map(|_| self.depot));
        nodes.extend_from_slice(&self.items);
        let n = nodes.len();

        let succ = (0..n).map(|_| solver.new_var_range(0, n as i32 - 1)).collect::<Vec<_>>();
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

        let mut search_vars = succ;
        search_vars.extend(cost_vars.iter().copied());
        LoweredProblem { solver, search_vars, cost_vars, nodes, depot_count: self.depot_count }
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
    let lowered = spec.lower();
    let LoweredProblem { mut solver, search_vars, cost_vars, nodes, depot_count } = lowered;
    let coeffs = vec![1i64; cost_vars.len()];
    let mut improvements = 0u64;
    let (best, stats, complete) = search::optimize_seeded(
        &mut solver,
        &search_vars,
        SearchObjective::Linear { coeffs: &coeffs, vars: &cost_vars },
        true,
        stop,
        seed,
        None,
        None,
        &[],
        None,
        |value, _assignment| {
            improvements += 1;
            report(value);
        },
    );

    let Some((assignment, value)) = best else {
        return Some(RoutingIntegerOutcome {
            solution: CollectionSolution {
                lists: vec![Vec::new(); depot_count],
                objectives: Vec::new(),
                feasible: false,
                starts: Vec::new(),
                machines: Vec::new(),
            },
            stats,
            complete,
            improvements,
        });
    };

    let succ = &assignment[..nodes.len()];
    // The circuit + capacity propagators guarantee a single Hamiltonian cycle that
    // decodes into exactly `depot_count` routes, so a failure here is a logic bug,
    // not an unsupported model -- panic rather than silently fall back to LS.
    let routes = routes_from_successors(&nodes, depot_count, succ).expect("solved circuit decodes into k routes");
    Some(RoutingIntegerOutcome {
        solution: CollectionSolution { lists: routes, objectives: vec![value], feasible: true, starts: Vec::new(), machines: Vec::new() },
        stats,
        complete,
        improvements,
    })
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
        if route_capacity_ok(self.depot_count, &self.demands, self.capacity, &succ) {
            Ok(())
        } else {
            Err(Inconsistency)
        }
    }
}

fn post_route_capacity_check(solver: &mut Solver, succ: &[VarId], depot_count: usize, demands: Vec<i32>, capacity: i32) {
    solver.post(Box::new(RouteCapacityCheck { succ: succ.to_vec(), depot_count, demands, capacity }));
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

fn route_capacity_ok(depot_count: usize, demands: &[i32], capacity: i32, succ: &[i32]) -> bool {
    if depot_count == 0 || demands.len() != succ.len() {
        return false;
    }
    let mut seen = vec![false; succ.len()];
    seen[0] = true;
    let mut routes = 0usize;
    let mut load = 0i64;
    let mut cur = 0usize;
    loop {
        let Ok(next) = usize::try_from(succ[cur]) else {
            return false;
        };
        if next >= succ.len() {
            return false;
        }
        if next == 0 {
            routes += 1;
            return load <= i64::from(capacity) && routes == depot_count && seen.iter().all(|&x| x);
        }
        if seen[next] {
            return false;
        }
        if next < depot_count {
            routes += 1;
            if load > i64::from(capacity) {
                return false;
            }
            load = 0;
            seen[next] = true;
        } else {
            load += i64::from(demands[next]);
            if load > i64::from(capacity) {
                return false;
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
    if !matches!(reduction.op, ReduceOp::Sum) {
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

fn parse_capacity_constraints(model: &CollectionModel) -> Option<Option<CapacitySpec>> {
    if model.constraints.is_empty() {
        return Some(None);
    }
    let mut by_list: Vec<Option<CapacitySpec>> = (0..model.lists).map(|_| None).collect();
    for constraint in &model.constraints {
        let (list, capacity) = parse_capacity_constraint(constraint, &model.items)?;
        if list >= model.lists || by_list[list].is_some() {
            return None;
        }
        by_list[list] = Some(capacity);
    }
    let mut parsed = by_list.into_iter();
    let first = parsed.next()??;
    for item in parsed {
        let other = item?;
        if other.capacity != first.capacity || other.demands != first.demands {
            return None;
        }
    }
    Some(Some(first))
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
        let demand =
            i32::try_from(eval_collection_expr_one(&constraint.reduction.arena.exprs, constraint.reduction.body, i64::from(item))?).ok()?;
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
    use std::sync::{atomic::AtomicBool, Arc};

    use crate::model::list::{CollectionModel, Constraint, ExprArena, Iterable, ObjectiveTier, Op, ReduceOp, Reduction};

    use super::solve_collection;

    #[test]
    fn lowers_closed_tsp_to_integer_circuit() {
        let dist = vec![vec![0, 10, 15, 20], vec![10, 0, 35, 25], vec![15, 35, 0, 30], vec![20, 25, 30, 0]];
        let mut arena = ExprArena::default();
        let i = arena.arg(0);
        let j = arena.arg(1);
        let body = arena.matrix(std::sync::Arc::new(dist), i, j);
        let reduction = Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: 0, end: 0 }, arena, body };
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
        assert!(outcome.stats.solutions > 0);
        assert!(outcome.improvements > 0);
        assert_eq!(outcome.solution.objectives, vec![80]);
        let mut route = outcome.solution.lists[0].clone();
        route.sort_unstable();
        assert_eq!(route, vec![1, 2, 3]);
        assert_eq!(reports.last(), Some(&80));
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
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body }
        };
        let load = |list| {
            let mut arena = ExprArena::default();
            let i = arena.arg(0);
            let body = arena.array(Arc::clone(&demand), i);
            Constraint { reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body }, op: Op::Le, rhs: 2 }
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
}
