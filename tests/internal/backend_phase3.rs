use qayd::model::list::{CollectionModel, Constraint, ExprArena, IntervalVar, Iterable, Op, ReduceOp, Reduction, Schedule};
use qayd::model::{estimated_exact_backend_bytes, estimated_local_search_backend_bytes, Model, ModelPackage};
use qayd::orchestrator::{compile_model_plan, EngineKind, ExecutablePlan, SolveBudget, SolveLimits, SolveMode, SolveRequest};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn assignment(items: usize, lists: usize) -> CollectionModel {
    CollectionModel {
        items: (0..items as i32).collect(),
        lists,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

fn schedule(intervals: usize) -> CollectionModel {
    CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(Schedule {
            intervals: (0..intervals)
                .map(|_| IntervalVar { duration: 1, horizon: intervals as i64, modes: Vec::new(), optional: false })
                .collect(),
            precedences: Vec::new(),
            resources: Vec::new(),
            minimize_makespan: true,
        }),
    }
}

fn scan_model(items: usize) -> CollectionModel {
    let mut arena = ExprArena::default();
    let step = arena.arg(1);
    let body = arena.constant(0);
    CollectionModel {
        items: (1..=items as i32).collect(),
        lists: 1,
        objectives: Vec::new(),
        constraints: vec![Constraint {
            reduction: Reduction {
                op: ReduceOp::Sum,
                iterable: Iterable::Scan { list: 0, init: 0, boundary: 0, step, end: None },
                arena,
                body,
                coeff: 1,
            },
            op: Op::Le,
            rhs: 0,
        }],
        globals: Vec::new(),
        schedule: None,
    }
}

fn routing(items: usize) -> CollectionModel {
    let nodes = items + 1;
    let matrix = Arc::new(
        (0..nodes).map(|from| (0..nodes).map(|to| i64::try_from(from.abs_diff(to)).unwrap()).collect::<Vec<_>>()).collect::<Vec<_>>(),
    );
    let mut arena = ExprArena::default();
    let from = arena.arg(0);
    let to = arena.arg(1);
    let body = arena.matrix(matrix, from, to);
    CollectionModel {
        items: (1..=items as i32).collect(),
        lists: 1,
        objectives: vec![qayd::model::list::ObjectiveTier {
            minimize: true,
            terms: vec![Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list: 0, start: 0, end: 0 }, arena, body, coeff: 1 }],
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    }
}

fn expensive_validation_model(items: usize) -> CollectionModel {
    let mut arena = ExprArena::default();
    let item_a = arena.arg(0);
    let item_b = arena.arg(1);
    let position_a = arena.arg(2);
    let position_b = arena.arg(3);
    let item_sum = arena.add(item_a, item_b);
    let position_sum = arena.add(position_a, position_b);
    let condition = ExprArena::eq(&mut arena, item_sum, position_sum);
    let one = arena.constant(1);
    let zero = arena.constant(0);
    let body = arena.if_then_else(condition, one, zero);
    CollectionModel {
        items: (0..items as i32).collect(),
        lists: 1,
        objectives: Vec::new(),
        constraints: vec![Constraint {
            reduction: Reduction { op: ReduceOp::Sum, iterable: Iterable::Pairs(0), arena, body, coeff: 1 },
            op: Op::Le,
            rhs: i64::MAX,
        }],
        globals: Vec::new(),
        schedule: None,
    }
}

fn compiled_engine(model: &CollectionModel, mode: SolveMode, memory_bytes: Option<u64>) -> EngineKind {
    let package = ModelPackage::new(Model::from_collection(model));
    let request = SolveRequest { mode, limits: SolveLimits { memory_bytes, ..SolveLimits::default() }, ..SolveRequest::default() };
    let plan = compile_model_plan(&package, &request, &SolveBudget::new(None)).unwrap();
    match plan.description() {
        ExecutablePlan::Single(plan) => plan.engine(),
        ExecutablePlan::Sequential(plans) => match plans.last() {
            Some(ExecutablePlan::Single(plan)) => plan.engine(),
            _ => panic!("unexpected nested sequential plan"),
        },
        ExecutablePlan::Portfolio(_) | ExecutablePlan::Decomposed { .. } => panic!("unexpected composite plan"),
    }
}

#[test]
fn auto_keeps_compact_assignment_exact_and_routes_large_assignment_to_ls() {
    assert_eq!(compiled_engine(&assignment(8, 3), SolveMode::Auto, None), EngineKind::ListExact);
    assert_eq!(compiled_engine(&assignment(30, 3), SolveMode::Auto, None), EngineKind::ListLocalSearch);
    assert_eq!(compiled_engine(&assignment(30, 3), SolveMode::Exact, None), EngineKind::ListExact);
}

#[test]
fn auto_routes_large_schedule_to_schedule_ls_but_exact_remains_available() {
    assert_eq!(compiled_engine(&schedule(12), SolveMode::Auto, None), EngineKind::ScheduleExact);
    assert_eq!(compiled_engine(&schedule(60), SolveMode::Auto, None), EngineKind::ScheduleLocalSearch);
    assert_eq!(compiled_engine(&schedule(60), SolveMode::Exact, None), EngineKind::ScheduleExact);
}

#[test]
fn memory_guard_is_applied_before_exact_schedule_lowering() {
    let compact = schedule(12);
    let estimate = estimated_exact_backend_bytes(&compact);
    assert!(estimate > 0);
    assert_eq!(compiled_engine(&compact, SolveMode::Auto, Some(estimate)), EngineKind::ScheduleExact);
    assert_eq!(compiled_engine(&compact, SolveMode::Auto, Some(estimate - 1)), EngineKind::ScheduleLocalSearch);
}

#[test]
fn local_search_memory_cost_is_separate_from_exact_lowering_cost() {
    let large_assignment = assignment(100, 10);
    assert!(estimated_local_search_backend_bytes(&large_assignment) < estimated_exact_backend_bytes(&large_assignment));

    let large_schedule = schedule(60);
    assert!(estimated_local_search_backend_bytes(&large_schedule) < estimated_exact_backend_bytes(&large_schedule));
}

#[test]
fn portfolio_memory_preflight_rejects_the_aggregate_routing_footprint() {
    let package = ModelPackage::new(Model::from_collection(&routing(100)));
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 1_000,
        limits: SolveLimits { memory_bytes: Some(64 * 1024 * 1024), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let error = match compile_model_plan(&package, &request, &SolveBudget::new(None)) {
        Ok(_) => panic!("the aggregate portfolio estimate exceeded its memory budget"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("across concurrent workers"), "{error}");
}

#[test]
fn exact_scan_capability_is_decided_by_compilation() {
    assert_eq!(compiled_engine(&scan_model(300), SolveMode::Auto, None), EngineKind::ListLocalSearch);
    assert_eq!(compiled_engine(&scan_model(300), SolveMode::Exact, None), EngineKind::ListExact);
}

#[test]
fn validation_cooperatively_stops_on_the_shared_budget_flag() {
    let model = expensive_validation_model(100);
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(5));
            stop.store(true, Ordering::Relaxed);
        });
        assert!(!model.validate_interruptible(&stop).expect("well-formed model"));
    });
}

#[test]
fn accepted_state_pair_caches_cooperatively_stop() {
    let model = expensive_validation_model(2_000);
    let stop = AtomicBool::new(false);
    let started = Instant::now();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(5));
            stop.store(true, Ordering::Relaxed);
        });
        let _solution = qayd::engines::ls::lists::solve_collection_capped(&model, 1, &stop, 0, None, &mut |_| {});
    });
    assert!(started.elapsed() < Duration::from_secs(1));
}
