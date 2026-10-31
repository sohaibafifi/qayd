use std::sync::Arc;

use qayd::model::list::{
    CollectionModel, Constraint as ListConstraint, ExprArena, GlobalConstraint, IntervalVar, Iterable, ObjectiveTier, Op, ReduceOp,
    Reduction, Schedule,
};
use qayd::model::{Constraint, Model, ModelPackage, Objective};
use qayd::orchestrator::{compile_model_plan, EngineKind, ExecutablePlan, SolveBudget, SolveMode, SolveRequest};

fn compiled_engine(collection: &CollectionModel, mode: SolveMode) -> EngineKind {
    let package = ModelPackage::new(Model::from_collection(collection));
    let request = SolveRequest { mode, ..SolveRequest::default() };
    let plan = compile_model_plan(&package, &request, &SolveBudget::new(None)).expect("model compiles to an executable plan");
    match plan.description() {
        ExecutablePlan::Single(plan) => plan.engine(),
        ExecutablePlan::Sequential(plans) => match plans.last() {
            Some(ExecutablePlan::Single(plan)) => plan.engine(),
            _ => panic!("unexpected nested sequential plan"),
        },
        ExecutablePlan::Portfolio(_) | ExecutablePlan::Decomposed { .. } => panic!("unexpected composite plan"),
    }
}

fn length_constraint(list: usize, op: Op, rhs: i64) -> ListConstraint {
    let mut arena = ExprArena::default();
    let body = arena.constant(1);
    ListConstraint { reduction: Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body, coeff: 1 }, op, rhs }
}

fn edge_cost_reduction(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let j = arena.arg(1);
    let body = arena.matrix(Arc::new(vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]]), i, j);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

fn pair_reduction(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let a = arena.arg(0);
    let b = arena.arg(1);
    let body = arena.add(a, b);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Pairs(list), arena, body, coeff: 1 }
}

fn pair_constraint(list: usize, rhs: i64) -> ListConstraint {
    ListConstraint { reduction: pair_reduction(list), op: Op::Le, rhs }
}

fn item_sum_constraint(list: usize, rhs: i64) -> ListConstraint {
    ListConstraint { reduction: item_sum_reduction(list), op: Op::Le, rhs }
}

fn item_sum_reduction(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let i = arena.arg(0);
    let body = arena.array(Arc::new(vec![0, 5, 7]), i);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn used_reduction(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let body = arena.constant(0);
    Reduction { op: ReduceOp::Used, iterable: Iterable::Items(list), arena, body, coeff: 1 }
}

fn used_constraint(list: usize, op: Op, rhs: i64) -> ListConstraint {
    ListConstraint { reduction: used_reduction(list), op, rhs }
}

#[test]
fn shared_model_mirrors_basic_list_collection_model() {
    let collection = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: Vec::new(),
        constraints: vec![length_constraint(0, Op::Le, 1)],
        globals: vec![GlobalConstraint::SameList { a: 1, b: 2 }],
        schedule: None,
    };

    let model = Model::from_collection(&collection);
    assert_eq!(model.lists().len(), 2);
    assert_eq!(model.constraints().len(), 3);
    assert!(matches!(model.constraints()[0], Constraint::ListPartition { .. }));
    assert!(matches!(model.constraints()[1], Constraint::ListLength { min: 0, max: 1, .. }));
    assert!(matches!(model.constraints()[2], Constraint::SameList { a: 1, b: 2, .. }));
    assert_eq!(compiled_engine(&collection, SolveMode::Auto), EngineKind::ListExact);
}

#[test]
fn engine_roles_make_local_search_proof_limits_explicit() {
    assert!(EngineKind::IntegerExact.can_prove_complete());
    assert!(EngineKind::ListExact.can_prove_complete());
    assert!(!EngineKind::ListLocalSearch.can_prove_complete());
    assert!(!EngineKind::ScheduleLocalSearch.can_prove_complete());
}

#[test]
fn shared_model_lifts_item_sum_bounds_for_domain_exact() {
    let collection = CollectionModel {
        items: vec![1, 2],
        lists: 1,
        objectives: Vec::new(),
        constraints: vec![item_sum_constraint(0, 10)],
        globals: Vec::new(),
        schedule: None,
    };

    let model = Model::from_collection(&collection);
    assert!(matches!(model.constraints()[1], Constraint::ListItemSum { min, max: 10, .. } if min < 0));
    assert_eq!(compiled_engine(&collection, SolveMode::Auto), EngineKind::ListExact);
}

#[test]
fn unsupported_list_reductions_are_mirrored_before_fallback() {
    let collection = CollectionModel {
        items: vec![1, 2],
        lists: 1,
        objectives: Vec::new(),
        constraints: vec![pair_constraint(0, 10)],
        globals: Vec::new(),
        schedule: None,
    };

    let model = Model::from_collection(&collection);
    assert!(matches!(model.constraints()[1], Constraint::ListReduction(_)));
    assert_eq!(compiled_engine(&collection, SolveMode::Auto), EngineKind::ListLocalSearch);
}

#[test]
fn routing_compiler_lowers_small_closed_tsp_to_integer_exact() {
    let collection = CollectionModel {
        items: vec![1, 2],
        lists: 1,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge_cost_reduction(0)], max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };

    assert_eq!(compiled_engine(&collection, SolveMode::Exact), EngineKind::RoutingExact);
}

#[test]
fn routing_compiler_lowers_small_homogeneous_cvrp_to_integer_exact() {
    let collection = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge_cost_reduction(0), edge_cost_reduction(1)], max_terms: None }],
        constraints: vec![item_sum_constraint(0, 10), item_sum_constraint(1, 10)],
        globals: Vec::new(),
        schedule: None,
    };

    assert_eq!(compiled_engine(&collection, SolveMode::Exact), EngineKind::RoutingExact);
}

#[test]
fn rejected_routing_shape_falls_through_to_compiled_list_exact() {
    let collection = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![edge_cost_reduction(0), edge_cost_reduction(1)], max_terms: None }],
        constraints: vec![item_sum_constraint(0, 10)],
        globals: Vec::new(),
        schedule: None,
    };

    assert_eq!(compiled_engine(&collection, SolveMode::Auto), EngineKind::ListExact);
}

#[test]
fn simple_list_objectives_use_domain_exact() {
    let collection = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: vec![used_reduction(0), used_reduction(1), item_sum_reduction(0)],
            max_terms: None,
        }],
        constraints: vec![length_constraint(0, Op::Le, 2)],
        globals: Vec::new(),
        schedule: None,
    };

    assert_eq!(compiled_engine(&collection, SolveMode::Auto), EngineKind::ListExact);
}

#[test]
fn used_constraints_lift_to_length_bounds() {
    let collection = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: Vec::new(),
        constraints: vec![used_constraint(0, Op::Eq, 0), used_constraint(1, Op::Ge, 1)],
        globals: Vec::new(),
        schedule: None,
    };

    let model = Model::from_collection(&collection);
    assert!(matches!(model.constraints()[1], Constraint::ListLength { min: 0, max: 0, .. }));
    assert!(matches!(model.constraints()[2], Constraint::ListLength { min: 1, max: 2, .. }));
    assert_eq!(compiled_engine(&collection, SolveMode::Auto), EngineKind::ListExact);
}
#[test]
fn shared_model_mirrors_fixed_interval_schedule() {
    let collection = CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(Schedule {
            intervals: vec![
                IntervalVar { duration: 2, horizon: 10, modes: Vec::new(), optional: false },
                IntervalVar { duration: 3, horizon: 10, modes: Vec::new(), optional: false },
            ],
            precedences: vec![(0, 1)],
            resources: Vec::new(),
            minimize_makespan: true,
        }),
    };

    let model = Model::from_collection(&collection);
    assert_eq!(model.intervals().len(), 2);
    assert_eq!(model.constraints().len(), 1);
    assert!(matches!(model.constraints()[0], Constraint::IntervalPrecedence { .. }));
    assert!(matches!(model.objectives()[0], Objective::Makespan { minimize: true, .. }));
    assert_eq!(compiled_engine(&collection, SolveMode::Auto), EngineKind::ScheduleExact);
}
