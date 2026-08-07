use std::sync::atomic::AtomicBool;

use qayd::engines::ls::lists::{audit_annealing_acceptance, audit_operator_learning, solve_collection_capped_profiled};
use qayd::model::list::{CollectionModel, Constraint, ExprArena, Iterable, Op, ReduceOp, Reduction};

fn count_items(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.constant(1);
    Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body, coeff: 1 }
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
