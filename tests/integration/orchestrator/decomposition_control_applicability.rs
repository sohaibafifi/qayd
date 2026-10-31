use std::sync::Arc;
use std::time::Duration;

use qayd::model::list::{ExprArena, Iterable, ReduceOp, Reduction};
use qayd::model::{Constraint, IntExpr, Model, ModelPackage, Objective};
use qayd::orchestrator::{
    compile_model_plan, CpControls, ExecutablePlan, LinearControls, RoutingControls, SolveBudget, SolveError, SolveMode, SolveRequest,
};

fn integer_satisfaction(variables: usize) -> ModelPackage {
    let mut model = Model::new();
    for _ in 0..variables {
        model.bool_var();
    }
    ModelPackage::new(model)
}

fn integer_optimization(variables: usize) -> ModelPackage {
    let mut model = Model::new();
    for _ in 0..variables {
        let variable = model.int_range(0, 2);
        model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(variable) });
    }
    ModelPackage::new(model)
}

fn invalid_request_message(package: &ModelPackage, request: &SolveRequest) -> String {
    match compile_model_plan(package, request, &SolveBudget::new(None)) {
        Err(SolveError::InvalidRequest(message)) => message,
        Err(error) => panic!("expected InvalidRequest, got {error}"),
        Ok(_) => panic!("invalid control was accepted"),
    }
}

#[test]
fn integer_control_errors_do_not_depend_on_whether_the_model_decomposes() {
    let requests = [
        SolveRequest {
            mode: SolveMode::Exact,
            linear: LinearControls { root_time: Duration::from_millis(7), ..LinearControls::default() },
            ..SolveRequest::default()
        },
        SolveRequest {
            mode: SolveMode::Exact,
            routing: RoutingControls { two_way: false, ..RoutingControls::default() },
            ..SolveRequest::default()
        },
        SolveRequest { mode: SolveMode::Exact, schedule_cdcl: true, ..SolveRequest::default() },
        SolveRequest {
            mode: SolveMode::Exact,
            threads: 2,
            cp: CpControls { split: true, ..CpControls::default() },
            ..SolveRequest::default()
        },
        SolveRequest {
            mode: SolveMode::Exact,
            threads: 2,
            cp: CpControls { probes: 1, ..CpControls::default() },
            ..SolveRequest::default()
        },
        SolveRequest { mode: SolveMode::Exact, threads: 2, cp: CpControls { lns: 1, ..CpControls::default() }, ..SolveRequest::default() },
    ];

    for request in requests {
        let direct = invalid_request_message(&integer_satisfaction(1), &request);
        let decomposed = invalid_request_message(&integer_satisfaction(2), &request);
        assert_eq!(decomposed, direct, "decomposition changed control validation for {request:?}");
    }
}

#[test]
fn objective_controls_remain_valid_on_decomposed_integer_optimization() {
    let requests = [
        SolveRequest {
            mode: SolveMode::Exact,
            linear: LinearControls { root_time: Duration::from_millis(7), ..LinearControls::default() },
            ..SolveRequest::default()
        },
        SolveRequest {
            mode: SolveMode::Exact,
            threads: 2,
            cp: CpControls { split: true, ..CpControls::default() },
            ..SolveRequest::default()
        },
        SolveRequest {
            mode: SolveMode::Exact,
            threads: 2,
            cp: CpControls { probes: 1, ..CpControls::default() },
            ..SolveRequest::default()
        },
        SolveRequest { mode: SolveMode::Exact, threads: 2, cp: CpControls { lns: 1, ..CpControls::default() }, ..SolveRequest::default() },
    ];

    for request in requests {
        compile_model_plan(&integer_optimization(1), &request, &SolveBudget::new(None))
            .unwrap_or_else(|error| panic!("direct objective request failed for {request:?}: {error}"));
        let decomposed = compile_model_plan(&integer_optimization(2), &request, &SolveBudget::new(None))
            .unwrap_or_else(|error| panic!("decomposed objective request failed for {request:?}: {error}"));
        assert!(matches!(decomposed.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 2));
    }
}

fn mixed_routing_and_integer_package() -> ModelPackage {
    let mut model = Model::new();
    model.bool_var();
    let items = vec![1, 2, 3];
    let routes = vec![model.list(items.clone()), model.list(items.clone())];
    model.add_constraint(Constraint::ListPartition { lists: routes, items });
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
    model.add_objective(Objective::ListTerms { minimize: true, terms, max_terms: None });
    ModelPackage::new(model)
}

#[test]
fn mixed_routing_component_keeps_its_routing_and_root_linear_controls() {
    let request = SolveRequest {
        mode: SolveMode::Exact,
        routing: RoutingControls { two_way: false, ..RoutingControls::default() },
        linear: LinearControls { root_time: Duration::from_millis(7), route_ng_size: 3, ..LinearControls::default() },
        ..SolveRequest::default()
    };

    let plan = compile_model_plan(&mixed_routing_and_integer_package(), &request, &SolveBudget::new(None)).unwrap();

    assert!(matches!(plan.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 2));
}
