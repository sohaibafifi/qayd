use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use qayd::engines::ls::lists::solve_collection_parallel_capped_profiled;
use qayd::model::list::{CollectionModel, Constraint, ExprArena, Iterable, ObjectiveTier, Op, ReduceOp, Reduction};

fn route_cost(list: usize, distances: Arc<Vec<Vec<i64>>>) -> Reduction {
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(distances, from, to);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

fn route_load(list: usize, demands: Arc<Vec<i64>>) -> Reduction {
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let body = arena.array(demands, item);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn benchmark_model() -> CollectionModel {
    let customers = 30usize;
    let vehicles = 5usize;
    let coordinates: Vec<(i64, i64)> =
        (0..=customers).map(|index| (((index * 37 + 11) % 101) as i64, ((index * 61 + 7) % 103) as i64)).collect();
    let distances: Vec<Vec<i64>> =
        coordinates.iter().map(|&(x1, y1)| coordinates.iter().map(|&(x2, y2)| (x1 - x2).abs() + (y1 - y2).abs()).collect()).collect();
    let demands: Vec<i64> = std::iter::once(0).chain((1..=customers).map(|item| 1 + (item * 7 % 9) as i64)).collect();
    let distances = Arc::new(distances);
    let demands = Arc::new(demands);
    CollectionModel {
        items: (1..=customers as i32).collect(),
        lists: vehicles,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..vehicles).map(|list| route_cost(list, Arc::clone(&distances))).collect(),
            max_terms: None,
        }],
        constraints: (0..vehicles)
            .map(|list| Constraint { reduction: route_load(list, Arc::clone(&demands)), op: Op::Le, rhs: 34 })
            .collect(),
        globals: Vec::new(),
        schedule: None,
    }
}

#[test]
#[ignore = "deterministic phase 5 portfolio benchmark"]
fn phase5_portfolio_benchmark() {
    let model = benchmark_model();
    let stop = AtomicBool::new(false);
    let iterations = 50;

    for workers in [1, 4] {
        let (solution, metrics) = solve_collection_parallel_capped_profiled(&model, 41, &stop, workers, iterations, None, &mut |_| {});
        let elapsed_nanos = metrics.worker_metrics.iter().map(|worker| worker.search.elapsed_nanos).max().unwrap_or(0);
        let candidates = metrics.worker_metrics.iter().map(|worker| worker.search.candidates).sum::<u64>();
        eprintln!(
            "phase5-bench workers={workers} feasible={} objective={} candidates={} elapsed_ms={:.3} publications={} injections={} best_worker={:?}",
            solution.feasible,
            solution.objectives.first().copied().unwrap_or(i64::MAX),
            candidates,
            elapsed_nanos as f64 / 1_000_000.0,
            metrics.publications,
            metrics.injections,
            metrics.best_worker,
        );
        assert!(solution.feasible);
    }
}
