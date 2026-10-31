use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use qayd::engines::ls::lists::{solve_collection_capped_profiled, solve_schedule};
use qayd::model::list::{
    CollectionModel, Constraint, ExprArena, IntervalVar, Iterable, ObjectiveTier, Op, ReduceOp, Reduction, Resource, Schedule,
};

fn edge_reduction(list: usize, matrix: Arc<Vec<Vec<i64>>>) -> Reduction {
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(matrix, from, to);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

fn demand_reduction(list: usize, demands: Arc<Vec<i64>>) -> Reduction {
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let body = arena.array(demands, item);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn cvrp(customers: usize, capacity: i64) -> CollectionModel {
    cvrp_with_lists(customers, capacity, customers)
}

fn cvrp_with_lists(customers: usize, capacity: i64, lists: usize) -> CollectionModel {
    let matrix =
        Arc::new((0..=customers).map(|from| (0..=customers).map(|to| from.abs_diff(to) as i64).collect::<Vec<_>>()).collect::<Vec<_>>());
    let demands = Arc::new(vec![1; customers + 1]);
    CollectionModel {
        items: (1..=customers as i32).collect(),
        lists,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..lists).map(|list| edge_reduction(list, Arc::clone(&matrix))).collect(),
            max_terms: None,
        }],
        constraints: (0..lists)
            .map(|list| Constraint { reduction: demand_reduction(list, Arc::clone(&demands)), op: Op::Le, rhs: capacity })
            .collect(),
        globals: Vec::new(),
        schedule: None,
    }
}

#[test]
fn routing_constructor_builds_directly_into_a_fixed_fleet() {
    let model = cvrp_with_lists(100, 5, 20);
    let stop = AtomicBool::new(false);
    let (solution, metrics) = solve_collection_capped_profiled(&model, 11, &stop, 0, None, &mut |_| {});

    assert!(solution.feasible);
    assert_eq!(solution.lists.iter().filter(|route| !route.is_empty()).count(), 20);
    assert!(matches!(metrics.constructor.as_deref(), Some("parallel-savings" | "cheapest-insertion" | "regret-insertion")));
    assert!(metrics.time_to_first_feasible_nanos.is_some());
    assert!(metrics.construction_candidates > 0);
}

#[test]
fn routing_constructor_publishes_a_feasible_fallback_and_merges_routes() {
    let model = cvrp(200, 10);
    let stop = AtomicBool::new(false);
    let mut progress = Vec::new();
    let (solution, metrics) = solve_collection_capped_profiled(&model, 7, &stop, 0, None, &mut |objective| progress.push(objective));

    assert!(solution.feasible);
    assert!(!progress.is_empty(), "the singleton fallback must be published before improvement");
    assert_eq!(metrics.constructor.as_deref(), Some("parallel-savings"));
    assert!(metrics.time_to_first_feasible_nanos.is_some());
    assert!(metrics.construction_candidates > 0);
    assert!(solution.lists.iter().filter(|route| !route.is_empty()).count() <= 20);
    assert!(solution.objectives[0] < 2 * (1..=200).sum::<i64>(), "savings must improve on singleton out-and-back routes");
}

#[test]
fn large_fixed_schedule_gets_a_serial_feasible_incumbent() {
    let jobs = 30;
    let machines = 10;
    let count = jobs * machines;
    let intervals =
        (0..count).map(|_| IntervalVar { duration: 1, horizon: count as i64, modes: Vec::new(), optional: false }).collect::<Vec<_>>();
    let precedences = (0..jobs)
        .flat_map(|job| (0..machines - 1).map(move |operation| (job * machines + operation, job * machines + operation + 1)))
        .collect::<Vec<_>>();
    let resources = (0..machines).map(|machine| Resource::NoOverlap((0..jobs).map(|job| job * machines + machine).collect())).collect();
    let model = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(Schedule { intervals, precedences: precedences.clone(), resources, minimize_makespan: true }),
    };
    let stop = AtomicBool::new(false);
    let (solution, metrics) = solve_schedule(model.schedule.as_ref().unwrap(), 0, &stop, &mut |_| {});

    assert!(solution.feasible);
    assert_eq!(solution.starts.len(), count);
    assert!(metrics.first_feasible.is_some());
    assert!(metrics.candidates > 0);
    for (before, after) in precedences {
        assert!(solution.starts[before] < solution.starts[after]);
    }
    for machine in 0..machines {
        let mut starts = (0..jobs).map(|job| solution.starts[job * machines + machine]).collect::<Vec<_>>();
        starts.sort_unstable();
        assert!(starts.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
