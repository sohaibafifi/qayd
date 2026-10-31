use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use qayd::audit_exhausted_neighborhood_learning;
use qayd::engines::ls::lists::{
    audit_checkpoint_history, audit_guaranteed_exploration, audit_lexicographic_regret, audit_neighborhood_learning,
    audit_routing_activity_counters, audit_scheduler_prefix, audit_timing_independent_cost_learning, audit_unproductive_cost_balance,
    solve_collection_capped_profiled, SliceKind,
};
use qayd::model::list::{CollectionModel, Constraint, ExprArena, Iterable, ObjectiveTier, Op, ReduceOp, Reduction};

#[test]
fn bounded_alns_starts_before_a_complete_local_optimum() {
    let slices = audit_scheduler_prefix(0, false, 9);
    assert_eq!(slices[0], SliceKind::Descent);
    assert_eq!(slices[1], SliceKind::Alns);
    assert_eq!(slices[7], SliceKind::Macro);
}

#[test]
fn stagnation_schedules_relinking_and_a_rare_global_scan() {
    let slices = audit_scheduler_prefix(16, true, 33);
    assert_eq!(slices[15], SliceKind::Relink);
    assert_eq!(slices[31], SliceKind::Global);
    assert_eq!(slices.iter().filter(|&&kind| kind == SliceKind::Global).count(), 1);
}

#[test]
fn checkpoints_keep_only_feasible_incumbent_data_from_before_the_threshold() {
    assert!(audit_checkpoint_history());
}

#[test]
fn productive_neighborhood_keeps_higher_probability_without_timing_lock_in() {
    assert!(audit_neighborhood_learning());
}

#[test]
fn exhausted_neighborhood_loses_weight_without_becoming_unreachable() {
    assert!(audit_exhausted_neighborhood_learning());
}

#[test]
fn every_neighborhood_has_a_deterministic_forced_exploration_turn() {
    assert!(audit_guaranteed_exploration());
}

#[test]
fn neighborhood_adaptation_uses_deterministic_cost_not_timing() {
    assert!(audit_timing_independent_cost_learning());
}

#[test]
fn unproductive_expensive_operator_cannot_dominate_projected_cost() {
    assert!(audit_unproductive_cost_balance());
}

#[test]
fn routing_activity_counters_distinguish_slices_attempts_and_completed_work() {
    assert!(audit_routing_activity_counters());
}

#[test]
fn regret_preserves_feasibility_and_every_lexicographic_tier() {
    assert!(audit_lexicographic_regret());
}

#[test]
fn routing_models_use_the_sliced_search_path() {
    let matrix = Arc::new(vec![vec![0, 2, 3], vec![2, 0, 1], vec![3, 1, 0]]);
    let demands = Arc::new(vec![0, 1, 1]);
    let edge = |list| {
        let mut arena = ExprArena::default();
        let from = arena.arg(0);
        let to = arena.arg(1);
        let body = arena.matrix(Arc::clone(&matrix), from, to);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
    };
    let demand = |list| {
        let mut arena = ExprArena::default();
        let item = arena.arg(0);
        let body = arena.array(Arc::clone(&demands), item);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }
    };
    let model = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge(0), edge(1)], max_terms: None }],
        constraints: (0..2).map(|list| Constraint { reduction: demand(list), op: Op::Le, rhs: 2 }).collect(),
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let (_, metrics) = solve_collection_capped_profiled(&model, 0, &stop, 2, None, &mut |_| {});

    assert_eq!(metrics.routing.slices, 2);
    assert_eq!(metrics.routing.descent_slices, 1);
    assert_eq!(metrics.routing.alns_slices, 1);
    let archive = metrics
        .routing
        .auxiliary
        .iter()
        .find(|entry| entry.name == "elite-archive")
        .expect("routing profile must include elite archive overhead");
    let selection = metrics
        .routing
        .auxiliary
        .iter()
        .find(|entry| entry.name == "elite-selection")
        .expect("routing profile must include elite selection overhead");
    assert!(archive.uses > 0);
    assert!(archive.work_units > 0);
    assert_eq!(selection.uses, metrics.routing.relink_slices);
}

#[test]
fn partial_edge_models_stay_on_the_generic_search_path() {
    let matrix = Arc::new(vec![vec![0, 2, 3], vec![2, 0, 1], vec![3, 1, 0]]);
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(matrix, from, to);
    let model = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: vec![Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: 0, end: 0 }, arena, body, coeff: 1 }],
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let (_, metrics) = solve_collection_capped_profiled(&model, 0, &stop, 2, None, &mut |_| {});

    assert_eq!(metrics.routing.slices, 0);
}

#[test]
fn maximization_and_heterogeneous_vehicles_stay_on_the_generic_path() {
    let matrix = Arc::new(vec![vec![0, 2, 3], vec![2, 0, 1], vec![3, 1, 0]]);
    let edge_terms = || {
        (0..2)
            .map(|list| {
                let mut arena = ExprArena::default();
                let from = arena.arg(0);
                let to = arena.arg(1);
                let body = arena.matrix(Arc::clone(&matrix), from, to);
                Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
            })
            .collect()
    };
    let make_model = |minimize, constraints| CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize, terms: edge_terms(), max_terms: None }],
        constraints,
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let (_, maximize_metrics) = solve_collection_capped_profiled(&make_model(false, Vec::new()), 0, &stop, 2, None, &mut |_| {});

    let demands = Arc::new(vec![0, 1, 1]);
    let capacities = (0..2)
        .map(|list| {
            let mut arena = ExprArena::default();
            let item = arena.arg(0);
            let body = arena.array(Arc::clone(&demands), item);
            Constraint {
                reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 },
                op: Op::Le,
                rhs: 1 + list as i64,
            }
        })
        .collect();
    let (_, heterogeneous_metrics) = solve_collection_capped_profiled(&make_model(true, capacities), 0, &stop, 2, None, &mut |_| {});

    assert_eq!(maximize_metrics.routing.slices, 0);
    assert_eq!(heterogeneous_metrics.routing.slices, 0);
}

#[test]
fn large_savings_construction_keeps_its_best_feasible_merge() {
    let customers = 257usize;
    let vehicles = 256usize;
    let mut costs = vec![vec![100; customers + 1]; customers + 1];
    for (node, row) in costs.iter_mut().enumerate() {
        row[node] = 0;
        row[0] = 1;
    }
    costs[0].fill(1);
    let costs = Arc::new(costs);
    let terms = (0..vehicles)
        .map(|list| {
            let mut arena = ExprArena::default();
            let from = arena.arg(0);
            let to = arena.arg(1);
            let body = arena.matrix(Arc::clone(&costs), from, to);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        })
        .collect();
    let model = CollectionModel {
        items: (1..=customers as i32).collect(),
        lists: vehicles,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let mut reports = Vec::new();
    let (solution, metrics) = solve_collection_capped_profiled(&model, 0, &stop, 0, None, &mut |objective| reports.push(objective));

    assert!(solution.feasible);
    assert_eq!(solution.objectives, vec![612]);
    assert_eq!(reports, vec![612]);
    assert_eq!(solution.lists.iter().filter(|route| !route.is_empty()).count(), vehicles);
    assert_eq!(metrics.constructor_cost, Some(612));
}

#[test]
fn temporary_savings_does_not_publish_an_infeasible_list_assignment() {
    let costs = Arc::new(vec![vec![0, 1, 1, 1], vec![1, 0, 2, 2], vec![1, 2, 0, 2], vec![1, 2, 2, 0]]);
    let terms = (0..2)
        .map(|list| {
            let mut arena = ExprArena::default();
            let from = arena.arg(0);
            let to = arena.arg(1);
            let body = arena.matrix(Arc::clone(&costs), from, to);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        })
        .collect();
    let mut arena = ExprArena::default();
    let one = arena.constant(1);
    let impossible_second_list = Constraint {
        reduction: Reduction { op: ReduceOp::Count, iterable: Iterable::Items(1), arena, body: one, coeff: 1 },
        op: Op::Le,
        rhs: -1,
    };
    let model = CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: vec![impossible_second_list],
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let (solution, metrics) = solve_collection_capped_profiled(&model, 0, &stop, 0, None, &mut |_| {});

    assert!(!solution.feasible);
    assert_eq!(metrics.constructor.as_deref(), Some("generic-greedy"));
}

#[test]
fn stopping_from_the_first_incumbent_callback_keeps_that_incumbent() {
    use std::sync::atomic::Ordering;

    let matrix = Arc::new(vec![vec![0, 10, 10], vec![10, 0, 1], vec![10, 1, 0]]);
    let terms = (0..2)
        .map(|list| {
            let mut arena = ExprArena::default();
            let from = arena.arg(0);
            let to = arena.arg(1);
            let body = arena.matrix(Arc::clone(&matrix), from, to);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        })
        .collect();
    let model = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let mut reports = Vec::new();
    let (solution, _) = solve_collection_capped_profiled(&model, 0, &stop, u64::MAX, None, &mut |objective| {
        reports.push(objective);
        if reports.len() == 2 {
            stop.store(true, Ordering::Relaxed);
        }
    });

    assert!(solution.feasible);
    assert_eq!(solution.objectives, vec![21]);
    assert_eq!(reports, vec![40, 21]);
}
