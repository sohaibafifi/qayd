//! Exact integer lowering for the smallest routing core.
//!
//! This handles one closed route with a direct matrix edge-sum objective:
//! `sum_edges(route, (i, j) => dist[i][j], start=d, end=d)`. It lowers to
//! successor variables plus the existing `circuit` propagator, one
//! `element_const` cost view per source node, and a symbolic linear objective.
//! Multi-route, capacity, time windows, and scan-style state remain with the
//! list local-search fallback until they have sound integer views.

#![cfg_attr(not(feature = "python"), allow(dead_code))]

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::collection::{CollectionModel, CollectionSolution, Expr, ExprId, Iterable, ReduceOp, Reduction};
use crate::constraints::graph;
use crate::constraints::primitives;
use crate::ids::VarId;
use crate::search::{self, Objective as SearchObjective, SolveStats};
use crate::Solver;

#[derive(Clone, Copy)]
enum MatrixOrientation {
    Direct,
    Reversed,
}

struct DirectEdgeMatrix {
    values: Arc<Vec<Vec<i64>>>,
    orientation: MatrixOrientation,
}

struct ClosedTourSpec {
    depot: i32,
    items: Vec<i32>,
    edge: DirectEdgeMatrix,
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

impl ClosedTourSpec {
    fn from_model(model: &CollectionModel) -> Option<Self> {
        if model.schedule.is_some()
            || model.lists != 1
            || !model.constraints.is_empty()
            || !model.globals.is_empty()
            || model.objectives.len() != 1
        {
            return None;
        }
        let tier = &model.objectives[0];
        if !tier.minimize || tier.terms.len() != 1 {
            return None;
        }
        let (depot, edge) = direct_closed_edge_term(&tier.terms[0], 0)?;
        if model.items.contains(&depot) || !items_are_unique(&model.items) {
            return None;
        }
        let mut nodes = Vec::with_capacity(model.items.len() + 1);
        nodes.push(depot);
        nodes.extend_from_slice(&model.items);
        edge.covers_i32_costs(&nodes).then_some(Self { depot, items: model.items.clone(), edge })
    }

    fn lower(&self) -> LoweredProblem {
        let mut solver = Solver::new();
        let mut nodes = Vec::with_capacity(self.items.len() + 1);
        nodes.push(self.depot);
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

        let mut search_vars = succ;
        search_vars.extend(cost_vars.iter().copied());
        LoweredProblem { solver, search_vars, cost_vars, nodes }
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
    let spec = ClosedTourSpec::from_model(model)?;
    let lowered = spec.lower();
    let LoweredProblem { mut solver, search_vars, cost_vars, nodes } = lowered;
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
                lists: vec![Vec::new()],
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
    let route = route_from_successors(&nodes, succ)?;
    Some(RoutingIntegerOutcome {
        solution: CollectionSolution {
            lists: vec![route],
            objectives: vec![value],
            feasible: true,
            starts: Vec::new(),
            machines: Vec::new(),
        },
        stats,
        complete,
        improvements,
    })
}

fn route_from_successors(nodes: &[i32], succ: &[i32]) -> Option<Vec<i32>> {
    if nodes.is_empty() || succ.len() != nodes.len() {
        return None;
    }
    let mut seen = vec![false; nodes.len()];
    seen[0] = true;
    let mut route = Vec::with_capacity(nodes.len().saturating_sub(1));
    let mut cur = usize::try_from(succ[0]).ok()?;
    while cur != 0 {
        if cur >= nodes.len() || seen[cur] {
            return None;
        }
        seen[cur] = true;
        route.push(nodes[cur]);
        cur = usize::try_from(succ[cur]).ok()?;
    }
    (route.len() + 1 == nodes.len() && seen.iter().all(|&x| x)).then_some(route)
}

fn direct_closed_edge_term(reduction: &Reduction, list: usize) -> Option<(i32, DirectEdgeMatrix)> {
    if !matches!(reduction.op, ReduceOp::Sum) {
        return None;
    }
    let Iterable::Edges { list: reduction_list, start, end } = &reduction.iterable else {
        return None;
    };
    if *reduction_list != list || start != end {
        return None;
    }
    direct_edge_matrix(reduction).map(|edge| (*start, edge))
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use crate::collection::{CollectionModel, ExprArena, Iterable, ObjectiveTier, ReduceOp, Reduction};

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
}
