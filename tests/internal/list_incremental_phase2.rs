use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use qayd::engines::ls::lists::{audit_incremental, solve_collection_capped_profiled};
use qayd::model::list::{CollectionModel, ExprArena, Iterable, MaxTerm, ObjectiveTier, ReduceOp, Reduction};

#[derive(Clone, Copy)]
enum IterableShape {
    Items,
    Edges,
    Pairs,
    Scan,
    Windows,
}

fn reduction(list: usize, shape: IterableShape, op: ReduceOp, coeff: i64) -> Reduction {
    let mut arena = ExprArena::default();
    let (iterable, body) = match shape {
        IterableShape::Items => {
            let item = arena.arg(0);
            let two = arena.constant(2);
            (Iterable::Items(list), arena.sub(item, two))
        }
        IterableShape::Edges => {
            let from = arena.arg(0);
            let to = arena.arg(1);
            let seven = arena.constant(7);
            let weighted = arena.mul(from, seven);
            (Iterable::Edges { list, start: 0, end: 6 }, arena.add(weighted, to))
        }
        IterableShape::Pairs => {
            let left = arena.arg(0);
            let right = arena.arg(1);
            let i = arena.arg(2);
            let j = arena.arg(3);
            let product = arena.mul(left, right);
            let position = arena.sub(i, j);
            (Iterable::Pairs(list), arena.add(product, position))
        }
        IterableShape::Scan => {
            let current = arena.arg(0);
            let accumulator = arena.arg(1);
            let previous = arena.arg(2);
            let with_current = arena.add(accumulator, current);
            let step = arena.sub(with_current, previous);
            let emitted_current = arena.arg(0);
            let emitted_accumulator = arena.arg(1);
            let body = arena.add(emitted_current, emitted_accumulator);
            (Iterable::Scan { list, init: 3, boundary: 0, step, end: Some(6) }, body)
        }
        IterableShape::Windows => {
            let inner = arena.arg(0);
            let total = arena.arg(1);
            let body = arena.abs(total);
            (Iterable::Windows { list, size: 2, inner }, body)
        }
    };
    Reduction { op, iterable, arena, body, coeff }
}

fn all_reductions(list: usize) -> Vec<Reduction> {
    let shapes = [IterableShape::Items, IterableShape::Edges, IterableShape::Pairs, IterableShape::Scan, IterableShape::Windows];
    let mut reductions = Vec::new();
    for (shape_idx, shape) in shapes.into_iter().enumerate() {
        reductions.push(reduction(list, shape, ReduceOp::Sum, if shape_idx % 2 == 0 { 1 } else { -2 }));
        reductions.push(reduction(list, shape, ReduceOp::Count, 1));
        reductions.push(reduction(list, shape, ReduceOp::Used, 1));
        reductions.push(reduction(list, shape, ReduceOp::Min, -1));
        reductions.push(reduction(list, shape, ReduceOp::Max, 2));
        reductions.push(reduction(list, shape, ReduceOp::SelectKth(1), 1));
    }
    reductions
}

fn phase2_model() -> CollectionModel {
    let mut terms = all_reductions(0);
    terms.extend(all_reductions(1));
    CollectionModel {
        items: vec![1, 2, 3, 4, 5],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms, max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

#[test]
fn every_iterable_and_operator_matches_full_recomputation() {
    let model = phase2_model();
    let checked = audit_incremental(&model, &[vec![1, 3, 5], vec![2, 4]]);
    assert!(checked > 20);
}

#[test]
fn every_candidate_reduction_uses_the_incremental_path() {
    let stop = AtomicBool::new(false);
    let (_solution, metrics) = solve_collection_capped_profiled(&phase2_model(), 7, &stop, 1, None, &mut |_| {});
    assert!(metrics.candidates > 0);
    assert_eq!(metrics.full_recompute_trial_list_evaluations, 0);
    assert_eq!(metrics.incremental_trial_list_evaluations, metrics.trial_list_evaluations);
    assert_eq!(metrics.full_recompute_percentage(), 0.0);
    assert_eq!(metrics.reductions.len(), 30, "all iterable/operator buckets are present");
    assert!(metrics.reductions.iter().all(|reduction| reduction.delta_calls > 0));
    assert!(metrics.average_recomputed_scan_steps() > 0.0);
}

#[test]
fn max_of_reductions_can_share_the_same_incremental_model() {
    let max_terms = vec![MaxTerm {
        groups: vec![
            vec![reduction(0, IterableShape::Scan, ReduceOp::Max, 1), reduction(1, IterableShape::Items, ReduceOp::Min, 1)],
            vec![reduction(0, IterableShape::Pairs, ReduceOp::SelectKth(1), 1)],
        ],
        coeff: 3,
    }];
    let model = CollectionModel {
        items: vec![1, 2, 3, 4, 5],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: all_reductions(0), max_terms: Some(max_terms) }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let checked = audit_incremental(&model, &[vec![1, 3, 5], vec![2, 4]]);
    assert!(checked > 20);
}

#[test]
fn deltas_preserve_ordered_i64_saturation() {
    let mut sum_arena = ExprArena::default();
    let sum_item = sum_arena.arg(0);
    let sum_body = sum_arena.array(Arc::new(vec![0, i64::MAX, 10, -20, i64::MIN, -10]), sum_item);
    let sum = Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(0), arena: sum_arena, body: sum_body, coeff: 1 };

    let mut count_arena = ExprArena::default();
    let count_body = count_arena.constant(1);
    let count =
        Reduction { op: ReduceOp::Count, iterable: Iterable::Items(0), arena: count_arena, body: count_body, coeff: i64::MAX / 2 + 1 };
    let mut other_sum = sum.clone();
    other_sum.iterable = Iterable::Items(1);
    let mut other_count = count.clone();
    other_count.iterable = Iterable::Items(1);
    let model = CollectionModel {
        items: vec![1, 2, 3, 4, 5],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![sum, count, other_sum, other_count], max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };

    let checked = audit_incremental(&model, &[vec![1, 2, 3], vec![4, 5]]);
    assert!(checked > 20);
}
