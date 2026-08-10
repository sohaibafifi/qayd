use std::sync::Arc;

use qayd::model::list::{ExprArena, Iterable, ReduceOp, Reduction};
use qayd::model::{Constraint, Model, ModelPackage, Objective};
use qayd::orchestrator::{solve_model_silent, EngineKind, SolveLimits, SolveMode, SolveRequest};

fn routing_model(minimize: bool) -> ModelPackage {
    let mut model = Model::new();
    let items = vec![1, 2, 3];
    let lists = vec![model.list(items.clone()), model.list(items.clone())];
    model.add_constraint(Constraint::ListPartition { lists, items });
    let matrix = Arc::new(vec![vec![0, 4, 7, 6], vec![4, 0, 2, 5], vec![7, 2, 0, 3], vec![6, 5, 3, 0]]);
    let terms = (0..2)
        .map(|list| {
            let mut arena = ExprArena::default();
            let from = arena.arg(0);
            let to = arena.arg(1);
            let body = arena.matrix(Arc::clone(&matrix), from, to);
            Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
        })
        .collect();
    model.add_objective(Objective::ListTerms { minimize, terms, max_terms: None });
    ModelPackage::new(model)
}

#[test]
fn routing_iteration_limit_counts_slices_and_reports_the_limit() {
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: SolveLimits { iterations: Some(1), ..SolveLimits::default() },
        profile: true,
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&routing_model(true), &request).unwrap();

    assert_eq!(result.reports()[0].engine, Some(EngineKind::RoutingLocalSearch));
    assert_eq!(result.reports()[0].search.nodes, 1);
    assert!(result.message().is_some_and(|message| message.contains("shared iteration limit")));
}

#[test]
fn maximized_edge_objective_is_not_mislabeled_as_specialized_routing() {
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: SolveLimits { iterations: Some(1), ..SolveLimits::default() },
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&routing_model(false), &request).unwrap();

    assert_eq!(result.reports()[0].engine, Some(EngineKind::ListLocalSearch));
}
