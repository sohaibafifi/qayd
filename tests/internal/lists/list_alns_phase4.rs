use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use qayd::engines::ls::lists::{audit_annealing_acceptance, audit_operator_learning, solve_collection_capped_profiled};
use qayd::model::list::{CollectionModel, Constraint, ExprArena, Iterable, ObjectiveTier, Op, ReduceOp, Reduction};

fn count_items(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.constant(1);
    Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn edge_cost(list: usize, costs: Arc<Vec<Vec<i64>>>, coeff: i64) -> Reduction {
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(costs, from, to);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff }
}

fn item_cost(list: usize, costs: Arc<Vec<i64>>) -> Reduction {
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let body = arena.array(costs, item);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn granular_candidate(costs: &[Vec<i64>], from: i32, to: i32, limit: usize) -> bool {
    let nearest = |node: i32| {
        let mut neighbors: Vec<(i64, i32)> =
            (0..costs.len() as i32).filter(|&other| other != node).map(|other| (costs[node as usize][other as usize], other)).collect();
        neighbors.sort_unstable();
        neighbors.truncate(limit);
        neighbors.into_iter().any(|(_, neighbor)| neighbor == if node == from { to } else { from })
    };
    nearest(from) || nearest(to)
}

fn neutral_model() -> CollectionModel {
    CollectionModel {
        items: (1..=12).collect(),
        lists: 3,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

#[test]
fn adaptive_destroy_and_repair_portfolio_is_deterministic_and_exercised() {
    let model = neutral_model();
    let stop = AtomicBool::new(false);
    let (first_solution, first) = solve_collection_capped_profiled(&model, 19, &stop, 128, None, &mut |_| {});
    let (second_solution, second) = solve_collection_capped_profiled(&model, 19, &stop, 128, None, &mut |_| {});

    assert_eq!(first_solution.lists, second_solution.lists);
    assert_eq!(first.alns, second.alns);
    assert_eq!(first.alns.iterations, 128);
    assert_eq!(first.alns.destroy.len(), 4);
    assert_eq!(first.alns.repair.len(), 4);
    assert!(first.alns.destroy.iter().any(|operator| operator.name == "route"));
    assert!(first.alns.repair.iter().any(|operator| operator.name == "blink-regret-2"));
    assert!(first.alns.destroy.iter().all(|operator| operator.uses > 0));
    assert!(first.alns.repair.iter().all(|operator| operator.uses > 0));
    assert!(first.alns.destroy.iter().any(|operator| operator.weight_milli != 1_000));
    assert!(first.alns.repair.iter().any(|operator| operator.weight_milli != 1_000));
    assert!(first.alns.late_acceptance_accepts > 0);
}

#[test]
fn gls_penalizes_violated_list_reductions_without_faking_feasibility() {
    let model = CollectionModel {
        items: vec![1, 2, 3, 4],
        lists: 1,
        objectives: Vec::new(),
        constraints: vec![Constraint { reduction: count_items(0), op: Op::Le, rhs: 0 }],
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let (solution, metrics) = solve_collection_capped_profiled(&model, 7, &stop, 32, None, &mut |_| {});

    assert!(!solution.feasible, "GLS weights must not turn a violated reduction into a feasible solution");
    assert!(metrics.alns.gls_updates > 0);
    assert!(metrics.alns.gls_penalties >= metrics.alns.gls_updates);
}

#[test]
fn annealing_acceptance_is_used_for_a_controlled_worsening() {
    assert!(audit_annealing_acceptance(31));
}

#[test]
fn operator_weights_react_to_observed_rewards() {
    assert!(audit_operator_learning());
}

#[test]
fn granular_swap_is_selected_from_a_new_arc_not_the_swapped_item_pair() {
    const ITEMS: usize = 30;
    const CANDIDATES: usize = 24;
    let mut guide = vec![vec![100; ITEMS + 1]; ITEMS + 1];
    for (node, row) in guide.iter_mut().enumerate() {
        row[node] = 0;
    }
    let mut make_non_candidate = |a: usize, b: usize| {
        guide[a][b] = 10_000;
        guide[b][a] = 10_000;
    };

    // Swap 1 and 2 in the paths [..., 3, 1, 4, ...] and
    // [..., 5, 2, 6, ...]. The swapped pair itself is deliberately absent
    // from both granular neighborhoods. Of the four new arcs, only (3, 2) is
    // retained, so the move must be admitted through its actual adjacencies.
    make_non_candidate(1, 2);
    make_non_candidate(2, 4);
    make_non_candidate(5, 1);
    make_non_candidate(1, 6);
    guide[3][2] = 0;
    guide[2][3] = 0;

    assert!(!granular_candidate(&guide, 1, 2, CANDIDATES));
    let new_arcs = [(3, 2), (2, 4), (5, 1), (1, 6)];
    assert_eq!(
        new_arcs.into_iter().filter(|&(from, to)| granular_candidate(&guide, from, to, CANDIDATES)).collect::<Vec<_>>(),
        vec![(3, 2)]
    );

    let initial =
        vec![vec![3, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], vec![5, 2, 6, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]];
    let expected =
        vec![vec![3, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], vec![5, 1, 6, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]];

    // Exact list sizes rule out cross-route relocates. These assignment costs
    // make 1 <-> 2 the only improving swap: moving 1 alone costs +10, moving 2
    // alone saves 20, and every other item in route 0 costs at least +30 when
    // moved to route 1.
    let mut assignment = vec![0; ITEMS + 1];
    assignment[1] = 10;
    assignment[2] = 20;
    for &item in &initial[0] {
        if item != 1 {
            assignment[item as usize] = 30;
        }
    }
    let guide = Arc::new(guide);
    let model = CollectionModel {
        items: (1..=ITEMS as i32).collect(),
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            // The zero-coefficient edge term supplies only the controlled
            // granular matrix. The item term supplies the independent score.
            terms: vec![edge_cost(0, Arc::clone(&guide), 0), edge_cost(1, Arc::clone(&guide), 0), item_cost(1, Arc::new(assignment))],
            max_terms: None,
        }],
        constraints: (0..2).map(|list| Constraint { reduction: count_items(list), op: Op::Eq, rhs: 15 }).collect(),
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let (solution, _) = solve_collection_capped_profiled(&model, 0, &stop, 1, Some(&initial), &mut |_| {});

    assert!(solution.feasible);
    assert_eq!(solution.objectives, vec![10]);
    assert_eq!(solution.lists, expected, "the first descent pass must apply the uniquely improving granular swap");
}
