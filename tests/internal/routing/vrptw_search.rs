use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use qayd::engines::ls::lists::{audit_incremental, routing_search_supported, solve_collection_capped};
use qayd::model::list::{CollectionModel, Constraint, ExprArena, Iterable, ObjectiveTier, Op, ReduceOp, Reduction};

fn time_window_model() -> CollectionModel {
    let travel = Arc::new(vec![vec![0, 10, 12, 100], vec![10, 0, 1, 80], vec![12, 1, 0, 75], vec![100, 80, 75, 0]]);
    let demands = Arc::new(vec![0, 5, 5, 9]);
    let earliest = Arc::new(vec![0, 10, 12, 100]);
    let latest = Arc::new(vec![1_000, 20, 24, 105]);
    let service = Arc::new(vec![0, 2, 2, 1]);
    let mut fleet_terms = Vec::new();
    let mut distance_terms = Vec::new();
    let mut constraints = Vec::new();

    for list in 0..2 {
        let mut used_arena = ExprArena::default();
        let used_body = used_arena.arg(0);
        fleet_terms.push(Reduction { op: ReduceOp::Used, iterable: Iterable::Items(list), arena: used_arena, body: used_body, coeff: 1 });

        let mut distance_arena = ExprArena::default();
        let from = distance_arena.arg(0);
        let to = distance_arena.arg(1);
        let distance_body = distance_arena.matrix(Arc::clone(&travel), from, to);
        distance_terms.push(Reduction {
            op: ReduceOp::Sum,
            iterable: Iterable::Edges { list, start: 0, end: 0 },
            arena: distance_arena,
            body: distance_body,
            coeff: 1,
        });

        let mut capacity_arena = ExprArena::default();
        let item = capacity_arena.arg(0);
        let demand_body = capacity_arena.array(Arc::clone(&demands), item);
        constraints.push(Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena: capacity_arena, body: demand_body, coeff: 1 },
            op: Op::Le,
            rhs: 20,
        });

        let mut window_arena = ExprArena::default();
        let current = window_arena.arg(0);
        let accumulator = window_arena.arg(1);
        let previous = window_arena.arg(2);
        let arc = window_arena.matrix(Arc::clone(&travel), previous, current);
        let arrival = window_arena.add(accumulator, arc);
        let release = window_arena.array(Arc::clone(&earliest), current);
        let start = window_arena.max(release, arrival);
        let duration = window_arena.array(Arc::clone(&service), current);
        let departure = window_arena.add(start, duration);
        let emitted_current = window_arena.arg(0);
        let emitted_departure = window_arena.arg(1);
        let emitted_duration = window_arena.array(Arc::clone(&service), emitted_current);
        let emitted_start = window_arena.sub(emitted_departure, emitted_duration);
        let deadline = window_arena.array(Arc::clone(&latest), emitted_current);
        let late = window_arena.sub(emitted_start, deadline);
        let zero = window_arena.constant(0);
        let lateness = window_arena.max(zero, late);
        constraints.push(Constraint {
            reduction: Reduction {
                op: ReduceOp::Sum,
                iterable: Iterable::Scan { list, init: 0, boundary: 0, step: departure, end: Some(0) },
                arena: window_arena,
                body: lateness,
                coeff: 1,
            },
            op: Op::Le,
            rhs: 0,
        });
    }

    CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![
            ObjectiveTier { minimize: true, terms: fleet_terms, max_terms: None },
            ObjectiveTier { minimize: true, terms: distance_terms, max_terms: None },
        ],
        constraints,
        globals: Vec::new(),
        schedule: None,
    }
}

#[test]
fn canonical_time_window_routing_uses_the_specialized_search() {
    assert!(routing_search_supported(&time_window_model()));
}

#[test]
fn time_window_scan_deltas_match_full_recomputation() {
    let checked = audit_incremental(&time_window_model(), &[vec![1, 2], vec![3]]);
    assert!(checked > 20);
}

#[test]
fn strict_fleet_objective_reaches_the_feasible_single_route() {
    let model = time_window_model();
    let stop = AtomicBool::new(false);
    for seed in 0..5 {
        let solution = solve_collection_capped(&model, seed, &stop, 64, None, &mut |_| {});
        assert!(solution.feasible);
        assert_eq!(solution.objectives.first(), Some(&1));
        assert_eq!(solution.lists.iter().filter(|route| !route.is_empty()).count(), 1);
    }
}
