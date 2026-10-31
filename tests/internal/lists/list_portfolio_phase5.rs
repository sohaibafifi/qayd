use std::collections::HashSet;
use std::sync::atomic::AtomicBool;

use qayd::engines::ls::lists::{
    audit_incumbent_exchange, audit_portfolio_merge, solve_collection_capped_profiled, solve_collection_parallel_capped_profiled,
};
use qayd::model::list::{CollectionModel, ExprArena, Iterable, ObjectiveTier, ReduceOp, Reduction};

fn count_items(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.constant(1);
    Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn model() -> CollectionModel {
    CollectionModel {
        items: (1..=16).collect(),
        lists: 4,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![count_items(0)], max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

#[test]
fn one_worker_is_exactly_the_sequential_trajectory() {
    let model = model();
    let stop = AtomicBool::new(false);
    let (sequential, sequential_metrics) = solve_collection_capped_profiled(&model, 29, &stop, 64, None, &mut |_| {});
    let (portfolio, portfolio_metrics) = solve_collection_parallel_capped_profiled(&model, 29, &stop, 1, 64, None, &mut |_| {});

    assert_eq!(portfolio.lists, sequential.lists);
    assert_eq!(portfolio.objectives, sequential.objectives);
    assert_eq!(portfolio.feasible, sequential.feasible);
    assert_eq!(portfolio_metrics.worker_metrics[0].search.alns, sequential_metrics.alns);
    assert_eq!(portfolio_metrics.publications, 0);
    assert_eq!(portfolio_metrics.injections, 0);
}

#[test]
fn portfolio_keeps_worker_zero_and_returns_a_no_worse_incumbent() {
    let model = model();
    let stop = AtomicBool::new(false);
    let (sequential, _) = solve_collection_capped_profiled(&model, 41, &stop, 48, None, &mut |_| {});
    let (portfolio, metrics) = solve_collection_parallel_capped_profiled(&model, 41, &stop, 4, 48, None, &mut |_| {});

    assert!(portfolio.feasible);
    assert!(portfolio.objectives <= sequential.objectives);
    assert_eq!(metrics.workers, 4);
    assert_eq!(metrics.worker_metrics[0].seed, 41);
    assert_eq!(metrics.worker_metrics[0].profile, "sequential");
    assert!(metrics.publications > 0);
}

#[test]
fn portfolio_uses_distinct_seeds_and_neighborhood_profiles() {
    let model = model();
    let stop = AtomicBool::new(false);
    // The cap is shared by the whole portfolio. Give every worker enough of its
    // partition to exercise the full adaptive operator repertoire while keeping
    // this test deterministic.
    let (_, metrics) = solve_collection_parallel_capped_profiled(&model, 7, &stop, 4, 64, None, &mut |_| {});

    assert_eq!(metrics.worker_metrics.iter().map(|worker| worker.worker).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    let seeds: HashSet<_> = metrics.worker_metrics.iter().map(|worker| worker.seed).collect();
    let profiles: HashSet<_> = metrics.worker_metrics.iter().map(|worker| worker.profile.as_str()).collect();
    assert_eq!(seeds.len(), 4);
    assert_eq!(profiles, HashSet::from(["sequential", "intensify", "diversify", "balanced"]));
    assert!(metrics.worker_metrics[1..]
        .iter()
        .any(|worker| { worker.search.alns.destroy.iter().any(|operator| operator.name == "route" && operator.uses > 0) }));
    assert!(metrics.worker_metrics[1..]
        .iter()
        .any(|worker| { worker.search.alns.repair.iter().any(|operator| operator.name == "blink-regret-2" && operator.uses > 0) }));
}

#[test]
fn shared_incumbent_exchange_is_strict_and_injectable() {
    assert!(audit_incumbent_exchange());
}

#[test]
fn final_merge_respects_feasibility_and_every_objective_sense() {
    assert!(audit_portfolio_merge());
}
