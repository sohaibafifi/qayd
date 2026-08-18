use std::sync::Arc;
use std::time::Duration;

use qayd::engines::{list_exact, routing, schedule, CollectionCompileContext, CompileFailure};
use qayd::model::list::{
    CollectionModel, CollectionSolution, Constraint as ListConstraint, ExprArena, Iterable, ObjectiveTier, Op, ReduceOp, Reduction,
};
use qayd::model::{
    CompiledCollection, Constraint, IntExpr, IntGlobalConstraint, IntVarRef, ListOrdering, Model, ModelObject, ModelPackage, Objective,
    Relation,
};
use qayd::orchestrator::{
    audit_cp_repair_completion, audit_interrupt_next_transfer_replay, audit_partial_cp_repair_search_rejection,
    audit_prearmed_cp_repair_interruption, compile_collection_plan, compile_model_plan, solve_collection_plan, solve_model_silent,
    solve_model_with_budget, EngineKind, EventControl, EventSink, ExecutablePlan, RoutingControls, SolveBudget, SolveError, SolveEvent,
    SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus, VerificationLevel,
};
fn count(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let one = arena.constant(1);
    Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body: one, coeff: 1 }
}

fn item_sum(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body: item, coeff: 1 }
}

fn route_cost(list: usize, matrix: Arc<Vec<Vec<i64>>>) -> Reduction {
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(matrix, from, to);
    Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 1 }
}

#[derive(Default)]
struct StopOnTransfer {
    events: Vec<SolveEvent>,
}

impl EventSink for StopOnTransfer {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        let stop = matches!(&event, SolveEvent::Candidate(candidate) if candidate.verification() == VerificationLevel::Transfer);
        self.events.push(event);
        Ok(if stop { EventControl::Stop } else { EventControl::Continue })
    }
}

#[derive(Default)]
struct StopOnCandidate {
    events: Vec<SolveEvent>,
}

#[derive(Default)]
struct RecordEvents {
    events: Vec<SolveEvent>,
}

impl EventSink for RecordEvents {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        self.events.push(event);
        Ok(EventControl::Continue)
    }
}

struct CancelThenFail {
    budget: SolveBudget,
}

struct FailOnProgress {
    budget: SolveBudget,
}

impl EventSink for FailOnProgress {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        if matches!(event, SolveEvent::Progress { .. }) {
            self.budget.cancel();
            Err(SolveError::Engine("progress callback failed".to_string()))
        } else {
            Ok(EventControl::Continue)
        }
    }
}

struct FailSessionProgress;

impl EventSink for FailSessionProgress {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        if matches!(event, SolveEvent::Progress { .. }) {
            Err(SolveError::InvalidResult("session progress callback failed".to_string()))
        } else {
            Ok(EventControl::Continue)
        }
    }
}

impl EventSink for CancelThenFail {
    fn emit(&mut self, _event: SolveEvent) -> Result<EventControl, SolveError> {
        self.budget.cancel();
        Err(SolveError::InvalidResult("event callback failed".to_string()))
    }
}

impl EventSink for StopOnCandidate {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        let stop = matches!(event, SolveEvent::Candidate(_));
        self.events.push(event);
        Ok(if stop { EventControl::Stop } else { EventControl::Continue })
    }
}

#[test]
fn collection_round_trip_uses_semantic_model_as_the_compiler_input() {
    let legacy = CollectionModel {
        items: vec![-3, 7, 11],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![count(0), count(1)], max_terms: None }],
        constraints: vec![ListConstraint { reduction: count(0), op: Op::Ge, rhs: 1 }],
        globals: Vec::new(),
        schedule: None,
    };
    let semantic = Model::from_collection(&legacy);
    let compiled = CompiledCollection::compile(&semantic).expect("semantic collection compiles");
    let physical = compiled.as_model();
    assert_eq!(physical.items, legacy.items);
    assert_eq!(physical.lists, legacy.lists);
    assert_eq!(physical.objectives.len(), 1);
    assert!(!physical.constraints.is_empty());
    physical.validate().unwrap();
}

#[test]
fn collection_round_trip_preserves_one_sided_item_sum_bounds() {
    let legacy = CollectionModel {
        items: vec![-3, 7, 11],
        lists: 1,
        objectives: Vec::new(),
        constraints: vec![
            ListConstraint { reduction: item_sum(0), op: Op::Ge, rhs: -4 },
            ListConstraint { reduction: item_sum(0), op: Op::Le, rhs: 12 },
        ],
        globals: Vec::new(),
        schedule: None,
    };

    let semantic = Model::from_collection(&legacy);
    let compiled = CompiledCollection::compile(&semantic).expect("semantic collection compiles");
    let sum_bounds = compiled
        .as_model()
        .constraints
        .iter()
        .filter(|constraint| matches!(constraint.reduction.op, ReduceOp::Sum))
        .map(|constraint| (constraint.op, constraint.rhs))
        .collect::<Vec<_>>();

    assert_eq!(sum_bounds.len(), 2);
    assert!(sum_bounds[0].0 == Op::Ge && sum_bounds[0].1 == -4);
    assert!(sum_bounds[1].0 == Op::Le && sum_bounds[1].1 == 12);
}

#[test]
fn semantic_length_range_lowers_to_both_relations() {
    let mut model = Model::new();
    let lists = vec![model.list(vec![1, 2, 3]), model.list(vec![1, 2, 3])];
    model.add_constraint(Constraint::ListPartition { lists: lists.clone(), items: vec![1, 2, 3] });
    model.add_constraint(Constraint::ListLength { list: lists[0], min: 1, max: 2 });
    model.add_objective(Objective::ListTerms { minimize: true, terms: vec![count(0)], max_terms: None });
    let compiled = CompiledCollection::compile(&model).unwrap();
    let bounds = &compiled.as_model().constraints;
    assert_eq!(bounds.len(), 2);
    assert!(bounds.iter().any(|constraint| constraint.op == Op::Ge && constraint.rhs == 1));
    assert!(bounds.iter().any(|constraint| constraint.op == Op::Le && constraint.rhs == 2));
}

#[test]
fn canonical_collection_verifier_rejects_duplicate_items_and_wrong_objective() {
    let model = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![count(0)], max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let duplicate = CollectionSolution {
        lists: vec![vec![1], vec![1, 2]],
        objectives: vec![1],
        feasible: true,
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        modes: Vec::new(),
        bound: None,
    };
    assert!(qayd::model::list::verify_collection_solution(&model, &duplicate).is_err());

    let wrong_objective = CollectionSolution { lists: vec![vec![1], vec![2]], objectives: vec![99], ..duplicate };
    assert!(qayd::model::list::verify_collection_solution(&model, &wrong_objective).is_err());
}

#[test]
fn final_status_construction_is_sealed_inside_the_orchestrator() {
    let source = include_str!("../../src/orchestrator/types.rs");
    let declaration =
        source.split("pub struct SolveResult {").nth(1).and_then(|source| source.split('}').next()).expect("SolveResult declaration");
    for field in ["status", "primal", "bounds", "proof", "reports", "message"] {
        assert!(declaration.contains(&format!("pub(super) {field}:")), "{field} must stay sealed inside the orchestrator");
    }
    assert!(SolveResult::unknown().validate_contract().is_ok());
}

#[test]
fn heuristic_engines_cannot_issue_complete_search_proofs() {
    assert!(!EngineKind::ListLocalSearch.can_prove_complete());
    assert!(!EngineKind::RoutingLocalSearch.can_prove_complete());
    assert!(!EngineKind::ScheduleLocalSearch.can_prove_complete());
    assert!(EngineKind::IntegerExact.can_prove_complete());
}

#[test]
fn candidate_protocol_distinguishes_local_transfer_and_final_boundaries() {
    assert_ne!(VerificationLevel::Local, VerificationLevel::Transfer);
    assert_ne!(VerificationLevel::Transfer, VerificationLevel::Final);
}

#[test]
fn package_preserves_frontend_metadata_without_affecting_the_model() {
    let mut package = ModelPackage::new(Model::new());
    package.metadata.names.insert(ModelObject::IntVar(IntVarRef(0)), "x".to_string());
    assert_eq!(package.metadata.names[&ModelObject::IntVar(IntVarRef(0))], "x");
}

#[test]
fn zero_budget_and_empty_composites_are_rejected_early() {
    let budget = SolveBudget::new(Some(Duration::ZERO));
    assert!(budget.expired());
    assert!(ExecutablePlan::Sequential(Vec::new()).validate().is_err());
    assert!(ExecutablePlan::Portfolio(Vec::new()).validate().is_err());
}

#[test]
fn portfolio_memory_estimate_accounts_for_every_concurrent_worker() {
    let mut model = Model::new();
    model.int_range(0, 10);
    let package = ModelPackage::new(model);
    let single_request = SolveRequest { mode: SolveMode::Exact, threads: 1, ..SolveRequest::default() };
    let portfolio_request = SolveRequest { mode: SolveMode::Exact, threads: 4, ..SolveRequest::default() };
    let single = compile_model_plan(&package, &single_request, &SolveBudget::new(None)).unwrap();
    let portfolio = compile_model_plan(&package, &portfolio_request, &SolveBudget::new(None)).unwrap();

    assert_eq!(portfolio.estimated_backend_bytes(), single.estimated_backend_bytes().saturating_mul(4));
}

#[test]
fn collection_orchestrator_compiles_executes_and_verifies_exact_search() {
    let legacy = CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![count(0)], max_terms: None }],
        constraints: vec![ListConstraint { reduction: count(0), op: Op::Ge, rhs: 1 }],
        globals: Vec::new(),
        schedule: None,
    };
    let semantic = Model::from_collection(&legacy);
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let budget = SolveBudget::new(None);
    let plan = compile_collection_plan(&semantic, &request, &budget).unwrap();
    assert_eq!(plan.engine(), EngineKind::ListExact);
    let result = solve_model_silent(&ModelPackage::new(semantic), &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [1]);
    assert!(result.proof().is_some());
    result.validate_contract().unwrap();
}

#[test]
fn exact_collection_compares_lightweight_routing_and_list_blueprints() {
    let matrix = Arc::new(vec![vec![0, 4, 7, 6], vec![4, 0, 2, 5], vec![7, 2, 0, 3], vec![6, 5, 3, 0]]);
    let legacy = CollectionModel {
        items: vec![1, 2, 3],
        lists: 1,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![route_cost(0, Arc::clone(&matrix))], max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let semantic = Model::from_collection(&legacy);
    let request = SolveRequest {
        mode: SolveMode::Exact,
        routing: RoutingControls { warm_start: false, ..RoutingControls::default() },
        ..SolveRequest::default()
    };
    let budget = SolveBudget::new(None);

    routing::reset_compile_calls_for_test();
    list_exact::reset_compile_calls_for_test();
    let plan = compile_collection_plan(&semantic, &request, &budget).expect("both exact compilers accept the small TSP");

    assert_eq!(routing::compile_calls_for_test(), 1);
    assert_eq!(list_exact::compile_calls_for_test(), 1);
    assert_eq!(routing::materialization_calls_for_test(), 0, "compilation must keep a routing blueprint, not a Solver");
    assert_eq!(plan.engine(), EngineKind::RoutingExact, "the deterministic cost policy prefers the routing blueprint");

    let mut sink = RecordEvents::default();
    let result = solve_collection_plan(&semantic, &plan, &request, &budget, None, None, &mut sink).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(routing::materialization_calls_for_test(), 1, "the selected routing Solver is materialized exactly once");

    let limited = SolveRequest {
        routing: RoutingControls::default(),
        limits: SolveLimits { memory_bytes: Some(plan.estimated_backend_bytes().saturating_sub(1)), ..SolveLimits::default() },
        ..request.clone()
    };
    let memory_selected = compile_collection_plan(&semantic, &limited, &SolveBudget::new(None)).unwrap();
    assert_eq!(memory_selected.engine(), EngineKind::ListExact, "candidate-local memory costs must permit the smaller exact plan");
    assert!(memory_selected.estimated_backend_bytes() < plan.estimated_backend_bytes());
}

#[test]
fn exact_collection_reports_each_typed_compiler_decline() {
    let mut arena = ExprArena::default();
    let body = arena.constant(0);
    let legacy = CollectionModel {
        items: vec![1, 2, 3],
        lists: 1,
        objectives: Vec::new(),
        constraints: vec![ListConstraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Pairs(0), arena, body, coeff: 1 },
            op: Op::Le,
            rhs: 0,
        }],
        globals: Vec::new(),
        schedule: None,
    };
    let semantic = Model::from_collection(&legacy);
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };

    let error = match compile_collection_plan(&semantic, &request, &SolveBudget::new(None)) {
        Ok(_) => panic!("both exact backends must decline"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains("[routing-exact:routing-shape]"), "{message}");
    assert!(message.contains("[list-exact:list-constraint]"), "{message}");
}

#[test]
fn physical_compilers_never_report_a_prearmed_stop_as_unsupported() {
    let assignment =
        CollectionModel { items: vec![1], lists: 1, objectives: Vec::new(), constraints: Vec::new(), globals: Vec::new(), schedule: None };
    let assignment_semantic = Model::from_collection(&assignment);
    let assignment = CompiledCollection::compile(&assignment_semantic).unwrap();
    let mut schedule_model = Model::new();
    schedule_model.interval(0, 3, 1);
    let schedule_compiled = CompiledCollection::compile(&schedule_model).unwrap();
    let request = SolveRequest::default();
    let assignment_context = CollectionCompileContext::new(&assignment_semantic, &request, &assignment);
    let schedule_context = CollectionCompileContext::new(&schedule_model, &request, &schedule_compiled);
    let stopped = std::sync::atomic::AtomicBool::new(true);

    assert!(matches!(routing::compile(&assignment_context, &stopped), Err(CompileFailure::Interrupted { .. })));
    assert!(matches!(list_exact::compile(&assignment_context, &stopped), Err(CompileFailure::Interrupted { .. })));
    assert!(matches!(schedule::compile(&schedule_context, &stopped), Err(CompileFailure::Interrupted { .. })));
}

#[test]
fn guarded_mismatch_constructor_is_not_selected_as_an_automatic_warm_start() {
    let mut model = Model::new();
    let cell = model.int_range(0, 0);
    let guard = model.bool_var();
    let index = model.int_range(0, 0);
    let symbol = model.bool_var();
    let mismatch = model.bool_var();
    let mismatch_count = model.bool_var();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Element { array: vec![cell], index, value: symbol }));
    model.add_constraint(Constraint::Intension(IntExpr::Eq(
        Box::new(IntExpr::Variable(mismatch)),
        Box::new(IntExpr::Ne(Box::new(IntExpr::Variable(symbol)), Box::new(IntExpr::Constant(1)))),
    )));
    model.add_constraint(Constraint::Linear { terms: vec![(1, mismatch), (-1, mismatch_count)], relation: Relation::Eq, rhs: 0 });
    model.add_constraint(Constraint::Intension(IntExpr::Or(vec![
        IntExpr::Eq(Box::new(IntExpr::Variable(guard)), Box::new(IntExpr::Constant(0))),
        IntExpr::Le(Box::new(IntExpr::Variable(mismatch_count)), Box::new(IntExpr::Constant(1))),
    ])));
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
        variables: vec![index],
        tuples: vec![vec![0]].into(),
        positive: true,
    }));
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Mul(vec![IntExpr::Constant(7), IntExpr::Variable(guard)]) });

    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [7]);
    assert!(!result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerLocalSearch)));
    assert!(result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerExact)));
}

fn guarded_element_objective_model() -> Model {
    let mut model = Model::new();
    let cell = model.int_range(1, 1);
    let guard = model.bool_var();
    let index = model.int_range(0, 0);
    let symbol = model.bool_var();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Element { array: vec![cell], index, value: symbol }));
    model.add_constraint(Constraint::Intension(IntExpr::Or(vec![
        IntExpr::Eq(Box::new(IntExpr::Variable(guard)), Box::new(IntExpr::Constant(0))),
        IntExpr::Eq(Box::new(IntExpr::Variable(symbol)), Box::new(IntExpr::Constant(1))),
    ])));
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
        variables: vec![index],
        tuples: vec![vec![0]].into(),
        positive: true,
    }));
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Mul(vec![IntExpr::Constant(7), IntExpr::Variable(guard)]) });
    model
}

#[test]
fn direct_guarded_element_constructor_is_not_selected_as_an_automatic_warm_start() {
    let request = SolveRequest { mode: SolveMode::Auto, ..SolveRequest::default() };
    let result = solve_model_silent(&ModelPackage::new(guarded_element_objective_model()), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [7]);
    assert!(!result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerLocalSearch)));
}

#[test]
fn explicit_integer_local_search_keeps_the_guarded_element_constructor() {
    let package = ModelPackage::new(guarded_element_objective_model());
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: SolveLimits { iterations: Some(1), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let plan = compile_model_plan(&package, &request, &SolveBudget::new(None)).unwrap();
    assert!(matches!(plan.description(), ExecutablePlan::Single(stage) if stage.engine() == EngineKind::IntegerLocalSearch));

    let result = solve_model_silent(&package, &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().objectives(), [7]);
    assert!(result.reports().iter().any(|report| report.engine == Some(EngineKind::IntegerLocalSearch)));
}

#[test]
fn incomplete_cp_repair_rejects_the_optional_candidate() {
    assert!(!audit_cp_repair_completion(false, true).unwrap());
    assert!(matches!(audit_cp_repair_completion(true, true), Err(SolveError::InvalidResult(_))));
    assert!(!audit_cp_repair_completion(true, false).unwrap());
    assert!(audit_partial_cp_repair_search_rejection().unwrap());
    assert!(audit_prearmed_cp_repair_interruption().unwrap());
}

#[test]
fn session_callback_failure_stops_search_without_masking_the_error() {
    use std::sync::atomic::AtomicBool;

    use crate::orchestrator::SemanticSolveSession;

    let mut model = Model::new();
    let value = model.int_range(0, 10);
    model.add_objective(Objective::IntExpr { minimize: true, expr: qayd::model::IntExpr::Variable(value) });
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();

    let error = match session.solve_with_external_stop(&SolveRequest::default(), &AtomicBool::new(false), &mut FailSessionProgress) {
        Ok(_) => panic!("the failing session callback was ignored"),
        Err(error) => error,
    };

    assert!(matches!(error, SolveError::InvalidResult(message) if message == "session progress callback failed"));
}

#[test]
fn routing_warm_start_is_verified_and_transferred_before_exact_execution() {
    let matrix = Arc::new(vec![vec![0, 4, 7, 6], vec![4, 0, 2, 5], vec![7, 2, 0, 3], vec![6, 5, 3, 0]]);
    let legacy = CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..2).map(|list| route_cost(list, Arc::clone(&matrix))).collect(),
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let semantic = Model::from_collection(&legacy);
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let budget = SolveBudget::new(Some(Duration::from_secs(5)));
    let package = ModelPackage::new(semantic);
    let plan = compile_model_plan(&package, &request, &budget).unwrap();
    let ExecutablePlan::Sequential(stages) = plan.description() else {
        panic!("routing warm start must compile to a sequential plan");
    };
    assert_eq!(stages.len(), 2);
    assert!(matches!(&stages[0], ExecutablePlan::Single(stage) if stage.engine() == EngineKind::RoutingLocalSearch));
    assert!(matches!(&stages[1], ExecutablePlan::Single(stage) if stage.engine() == EngineKind::RoutingExact));

    let mut sink = StopOnTransfer::default();
    let result = solve_model_with_budget(&package, &request, &budget, &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(result.primal().is_some(), "a verified warm-start incumbent must survive a stop between sequential stages");
    assert!(matches!(sink.events.first(), Some(SolveEvent::StageStarted { engine: EngineKind::RoutingLocalSearch, warm_start: true })));
    assert!(sink
        .events
        .iter()
        .any(|event| matches!(event, SolveEvent::Candidate(candidate) if candidate.verification() == VerificationLevel::Transfer)));
    assert!(!sink.events.iter().any(|event| {
        matches!(event, SolveEvent::Progress { engine: EngineKind::RoutingExact, .. })
            || matches!(event, SolveEvent::Finished(report) if report.engine == Some(EngineKind::RoutingExact))
    }));
    let transfer = sink
        .events
        .iter()
        .position(|event| matches!(event, SolveEvent::Candidate(candidate) if candidate.verification() == VerificationLevel::Transfer))
        .unwrap();
    assert_eq!(transfer + 1, sink.events.len(), "consumer stop must end the routing sequence immediately");
    assert!(matches!(sink.events.last(), Some(SolveEvent::Candidate(_))), "a consumer stop on a transfer candidate must be terminal");
    assert!(result.reports().iter().all(|report| report.engine != Some(EngineKind::RoutingExact)));
}

#[test]
fn interrupted_transfer_replay_retains_the_verified_warm_incumbent() {
    let matrix = Arc::new(vec![vec![0, 4, 7, 6], vec![4, 0, 2, 5], vec![7, 2, 0, 3], vec![6, 5, 3, 0]]);
    let legacy = CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..2).map(|list| route_cost(list, Arc::clone(&matrix))).collect(),
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let package = ModelPackage::new(Model::from_collection(&legacy));
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let budget = SolveBudget::new(Some(Duration::from_secs(5)));
    let mut sink = RecordEvents::default();

    audit_interrupt_next_transfer_replay();
    let result = solve_model_with_budget(&package, &request, &budget, &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().verification(), VerificationLevel::Final);
    assert!(result.reports().iter().all(|report| report.engine != Some(EngineKind::RoutingExact)));
    assert!(!sink
        .events
        .iter()
        .any(|event| matches!(event, SolveEvent::Candidate(candidate) if candidate.verification() == VerificationLevel::Transfer)));
}

#[test]
fn routing_sequence_memory_preflight_includes_every_resident_stage() {
    let matrix = Arc::new(vec![vec![0, 4, 7, 6], vec![4, 0, 2, 5], vec![7, 2, 0, 3], vec![6, 5, 3, 0]]);
    let legacy = CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..2).map(|list| route_cost(list, Arc::clone(&matrix))).collect(),
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let package = ModelPackage::new(Model::from_collection(&legacy));
    let unlimited = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let estimate = compile_model_plan(&package, &unlimited, &SolveBudget::new(None)).unwrap().estimated_backend_bytes();
    let limited = SolveRequest {
        mode: SolveMode::Exact,
        limits: SolveLimits { memory_bytes: Some(estimate.saturating_sub(1)), ..SolveLimits::default() },
        ..SolveRequest::default()
    };

    assert!(matches!(
        compile_model_plan(&package, &limited, &SolveBudget::new(None)),
        Err(SolveError::Compile(message)) if message.contains("memory limit")
    ));
}

#[test]
fn routing_exact_consumes_the_verified_stage_candidate_and_preserves_stage_reports() {
    let matrix = Arc::new(vec![vec![0, 4, 7, 6], vec![4, 0, 2, 5], vec![7, 2, 0, 3], vec![6, 5, 3, 0]]);
    let legacy = CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..2).map(|list| route_cost(list, Arc::clone(&matrix))).collect(),
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let package = ModelPackage::new(Model::from_collection(&legacy));
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let budget = SolveBudget::new(Some(Duration::from_secs(5)));
    let mut sink = RecordEvents::default();

    let result = solve_model_with_budget(&package, &request, &budget, &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    let engines = result.reports().iter().filter_map(|report| report.engine).collect::<Vec<_>>();
    assert_eq!(engines, [EngineKind::RoutingLocalSearch, EngineKind::RoutingExact]);
    assert!(matches!(sink.events.first(), Some(SolveEvent::StageStarted { engine: EngineKind::RoutingLocalSearch, warm_start: true })));
    let transfer = sink
        .events
        .iter()
        .position(|event| matches!(event, SolveEvent::Candidate(candidate) if candidate.verification() == VerificationLevel::Transfer))
        .expect("the LS candidate crosses a transfer boundary");
    let exact_stage = sink
        .events
        .iter()
        .position(|event| matches!(event, SolveEvent::StageStarted { engine: EngineKind::RoutingExact, warm_start: false }))
        .expect("the exact routing stage is announced");
    assert_eq!(transfer + 1, exact_stage, "the exact stage marker must immediately precede exact-engine execution");
    let final_candidate = sink
        .events
        .iter()
        .position(|event| matches!(event, SolveEvent::Candidate(candidate) if candidate.verification() == VerificationLevel::Final))
        .expect("the exact result is finally verified");
    let exact_finished = sink
        .events
        .iter()
        .position(|event| matches!(event, SolveEvent::Finished(report) if report.engine == Some(EngineKind::RoutingExact)))
        .expect("the exact routing stage publishes its report");
    assert!(exact_stage < final_candidate, "the exact stage marker must precede its verified candidate");
    assert!(exact_stage < exact_finished, "the exact stage marker must precede its engine report");
    assert!(transfer < final_candidate);
}

#[test]
fn collection_options_are_rejected_when_the_selected_stage_cannot_consume_them() {
    let assignment = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let exact_hint = SolveRequest { mode: SolveMode::Exact, list_hint: Some(vec![vec![1], vec![2]]), ..SolveRequest::default() };
    assert!(matches!(
        compile_model_plan(
            &ModelPackage::new(Model::from_collection(&assignment)),
            &exact_hint,
            &SolveBudget::new(None)
        ),
        Err(SolveError::InvalidRequest(message)) if message.contains("list_hint")
    ));
    for (hint, expected) in
        [(vec![vec![1]], "sequences"), (vec![vec![1, 99], Vec::new()], "universe"), (vec![vec![1], vec![1]], "more than once")]
    {
        let local_search_hint = SolveRequest { mode: SolveMode::LocalSearch, list_hint: Some(hint), ..SolveRequest::default() };
        assert!(matches!(
            compile_model_plan(
                &ModelPackage::new(Model::from_collection(&assignment)),
                &local_search_hint,
                &SolveBudget::new(None)
            ),
            Err(SolveError::InvalidRequest(message)) if message.contains(expected)
        ));
    }
    let valid_partial_hint =
        SolveRequest { mode: SolveMode::LocalSearch, list_hint: Some(vec![vec![1], Vec::new()]), ..SolveRequest::default() };
    assert!(
        compile_model_plan(&ModelPackage::new(Model::from_collection(&assignment)), &valid_partial_hint, &SolveBudget::new(None)).is_ok()
    );

    let mut optional_assignment = Model::new();
    let items = vec![1, 2, 3, 4];
    let left = optional_assignment.list(items.clone());
    let middle = optional_assignment.list(items.clone());
    let right = optional_assignment.list(items.clone());
    let remainder = optional_assignment.remainder_list(items.clone(), ListOrdering::Ordered);
    optional_assignment.add_constraint(Constraint::ListPartition { lists: vec![left, middle, right, remainder], items });
    let implicit_pool_hint =
        SolveRequest { mode: SolveMode::LocalSearch, list_hint: Some(vec![vec![1], vec![2]]), ..SolveRequest::default() };
    assert!(compile_model_plan(&ModelPackage::new(optional_assignment.clone()), &implicit_pool_hint, &SolveBudget::new(None)).is_ok());
    let explicit_pool_hint =
        SolveRequest { mode: SolveMode::LocalSearch, list_hint: Some(vec![vec![1], vec![2], vec![3], vec![4]]), ..SolveRequest::default() };
    assert!(matches!(
        compile_model_plan(&ModelPackage::new(optional_assignment), &explicit_pool_hint, &SolveBudget::new(None)),
        Err(SolveError::InvalidRequest(message)) if message.contains("hidden remainder pool is implicit")
    ));

    let mut schedule = Model::new();
    schedule.interval(0, 10, 1);
    let schedule_iterations = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: SolveLimits { iterations: Some(10), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    assert!(compile_model_plan(&ModelPackage::new(schedule.clone()), &schedule_iterations, &SolveBudget::new(None)).is_ok());
    let schedule_cdcl_iterations = SolveRequest {
        mode: SolveMode::Auto,
        schedule_cdcl: true,
        limits: SolveLimits { iterations: Some(10), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    assert!(matches!(
        compile_model_plan(&ModelPackage::new(schedule), &schedule_cdcl_iterations, &SolveBudget::new(None)),
        Err(SolveError::InvalidRequest(message)) if message.contains("iteration")
    ));
    let mut large_schedule = Model::new();
    for _ in 0..60 {
        large_schedule.interval(0, 60, 1);
    }
    let auto_schedule_cdcl = SolveRequest { mode: SolveMode::Auto, schedule_cdcl: true, ..SolveRequest::default() };
    assert!(matches!(
        compile_model_plan(
            &ModelPackage::new(large_schedule),
            &auto_schedule_cdcl,
            &SolveBudget::new(None)
        ),
        Err(SolveError::InvalidRequest(message)) if message.contains("exact scheduling backend")
    ));

    let matrix = Arc::new(vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]]);
    let routing = CollectionModel {
        items: vec![1, 2],
        lists: 1,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![route_cost(0, Arc::clone(&matrix))], max_terms: None }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let ignored_routing_control = SolveRequest {
        mode: SolveMode::LocalSearch,
        routing: RoutingControls { two_way: false, ..RoutingControls::default() },
        ..SolveRequest::default()
    };
    assert!(matches!(
        compile_model_plan(
            &ModelPackage::new(Model::from_collection(&routing)),
            &ignored_routing_control,
            &SolveBudget::new(None)
        ),
        Err(SolveError::InvalidRequest(message)) if message.contains("routing controls")
    ));

    let fallback = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: vec![route_cost(0, Arc::clone(&matrix)), route_cost(1, matrix)],
            max_terms: None,
        }],
        constraints: vec![ListConstraint { reduction: item_sum(0), op: Op::Le, rhs: 10 }],
        globals: Vec::new(),
        schedule: None,
    };
    let fallback_control = SolveRequest {
        mode: SolveMode::Auto,
        routing: RoutingControls { two_way: false, ..RoutingControls::default() },
        ..SolveRequest::default()
    };
    assert!(matches!(
        compile_model_plan(
            &ModelPackage::new(Model::from_collection(&fallback)),
            &fallback_control,
            &SolveBudget::new(None)
        ),
        Err(SolveError::InvalidRequest(message)) if message.contains("selected exact routing backend")
    ));
}

#[test]
fn exact_schedule_memory_preflight_runs_before_numeric_lowering() {
    let mut model = Model::new();
    let oversized_duration = i64::from(i32::MAX) + 1;
    for _ in 0..200 {
        model.interval(0, 0, oversized_duration);
    }
    let request = SolveRequest {
        mode: SolveMode::Exact,
        limits: SolveLimits { memory_bytes: Some(1024 * 1024), ..SolveLimits::default() },
        ..SolveRequest::default()
    };

    let error = match compile_model_plan(&ModelPackage::new(model), &request, &SolveBudget::new(None)) {
        Ok(_) => panic!("the over-budget exact schedule was compiled"),
        Err(error) => error,
    };

    assert!(matches!(error, SolveError::Compile(message) if message.contains("memory limit") && !message.contains("duration")));
}

#[test]
fn interruption_is_explicit_and_does_not_mask_callback_failures() {
    let legacy = CollectionModel {
        items: vec![1, 2],
        lists: 2,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let package = ModelPackage::new(Model::from_collection(&legacy));
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    assert!(matches!(compile_model_plan(&package, &request, &SolveBudget::new(Some(Duration::ZERO))), Err(SolveError::Interrupted(_))));

    let budget = SolveBudget::new(None);
    let mut sink = CancelThenFail { budget: budget.clone() };
    let error = solve_model_with_budget(&package, &request, &budget, &mut sink).unwrap_err();
    assert!(matches!(error, SolveError::InvalidResult(message) if message == "event callback failed"));
    assert!(budget.expired());

    let improving = CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![count(0)], max_terms: None }],
        constraints: vec![ListConstraint { reduction: count(0), op: Op::Ge, rhs: 1 }],
        globals: Vec::new(),
        schedule: None,
    };
    let local_request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: SolveLimits { iterations: Some(100), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let local_budget = SolveBudget::new(None);
    let mut progress_sink = FailOnProgress { budget: local_budget.clone() };
    let error =
        solve_model_with_budget(&ModelPackage::new(Model::from_collection(&improving)), &local_request, &local_budget, &mut progress_sink)
            .unwrap_err();
    assert!(matches!(error, SolveError::Engine(message) if message == "progress callback failed"));
}

#[test]
fn event_stop_suppresses_the_rest_of_collection_finalization() {
    let legacy = CollectionModel {
        items: vec![1, 2, 3],
        lists: 2,
        objectives: vec![ObjectiveTier { minimize: true, terms: vec![count(0)], max_terms: None }],
        constraints: vec![ListConstraint { reduction: count(0), op: Op::Ge, rhs: 1 }],
        globals: Vec::new(),
        schedule: None,
    };
    let package = ModelPackage::new(Model::from_collection(&legacy));
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let budget = SolveBudget::new(None);
    let mut sink = StopOnCandidate::default();

    let result = solve_model_with_budget(&package, &request, &budget, &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    let candidate =
        sink.events.iter().position(|event| matches!(event, SolveEvent::Candidate(_))).expect("the final candidate is published");
    assert_eq!(candidate + 1, sink.events.len(), "no final event may follow a consumer stop");
    assert!(matches!(sink.events.last(), Some(SolveEvent::Candidate(_))), "a consumer stop on a candidate must be terminal");
}
