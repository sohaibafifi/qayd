use std::sync::atomic::AtomicBool;

use qayd::engines::ls::lists::{solve_collection_capped_profiled, ListIterableKind, ListReduceOpKind};
use qayd::model::list::{CollectionModel, ExprArena, Iterable, ObjectiveTier, ReduceOp, Reduction};

fn model_with_objective(items: Vec<i32>, reduction: Reduction) -> CollectionModel {
    CollectionModel {
        items,
        lists: 1,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![reduction], max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

fn item_sum() -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.arg(0);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(0), arena, body, coeff: 1 }
}

fn prefix_sum_scan() -> Reduction {
    let mut arena = ExprArena::default();
    let current = arena.arg(0);
    let accumulator = arena.arg(1);
    let step = arena.add(current, accumulator);
    let body = arena.arg(1);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Scan { list: 0, init: 0, boundary: 0, step, end: None }, arena, body, coeff: 1 }
}

#[test]
fn profiled_incremental_list_search_reports_candidates_and_delta_costs() {
    let model = model_with_objective(vec![1, 2, 3, 4], item_sum());
    let stop = AtomicBool::new(false);
    let (_solution, metrics) = solve_collection_capped_profiled(&model, 0, &stop, 1, None, &mut |_| {});

    assert!(metrics.candidates > 0);
    assert!(metrics.candidates_per_second().is_finite());
    assert_eq!(metrics.full_recompute_trial_list_evaluations, 0);
    assert_eq!(metrics.incremental_trial_list_evaluations, metrics.trial_list_evaluations);
    assert_eq!(metrics.full_recompute_percentage(), 0.0);

    let reduction = metrics.reduction(ListIterableKind::Items, ListReduceOpKind::Sum).expect("the item/sum reduction must be reported");
    assert!(reduction.full_calls > 0, "initial-state scoring is a full evaluation");
    assert!(reduction.delta_calls > 0, "candidate scoring must use the item/sum delta");
}

#[test]
fn profiled_scan_reports_incremental_suffix_recomputation() {
    let model = model_with_objective(vec![1, 2, 3, 4], prefix_sum_scan());
    let stop = AtomicBool::new(false);
    let (_solution, metrics) = solve_collection_capped_profiled(&model, 0, &stop, 1, None, &mut |_| {});

    assert!(metrics.candidates > 0);
    assert_eq!(metrics.full_recompute_trial_list_evaluations, 0);
    assert_eq!(metrics.incremental_trial_list_evaluations, metrics.trial_list_evaluations);
    assert_eq!(metrics.full_recompute_percentage(), 0.0);
    assert!(metrics.scan_recomputations > 0);
    assert!(metrics.scan_recomputed_steps > 0);
    assert!(metrics.average_recomputed_scan_steps() > 0.0);

    let reduction = metrics.reduction(ListIterableKind::Scan, ListReduceOpKind::Sum).expect("the scan/sum reduction must be reported");
    assert!(reduction.full_calls > 0);
    assert!(reduction.delta_calls > 0);

    let rendered = metrics.to_string();
    assert!(rendered.contains("candidates/s="));
    assert!(rendered.contains("full-fallback%="));
    assert!(rendered.contains("average-recomputed-scan-steps="));
    assert!(rendered.contains("reduction=scan/sum"));
}
