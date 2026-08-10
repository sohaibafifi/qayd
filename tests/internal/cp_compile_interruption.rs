use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use crate::constraints::table;
use crate::model::{Automaton, CompiledCp, Constraint, IntExpr, IntGlobalConstraint, Model, Objective};
use crate::orchestrator::{
    audit_cp_root_problem_clones, compile_cp_plan, solve_cp_plan, IgnoreEvents, SolveBudget, SolveError, SolveLimits, SolveRequest,
    SolveStatus,
};
use crate::Solver;

#[test]
fn cp_compilation_honors_a_prearmed_stop() {
    let mut model = Model::new();
    model.int_range(0, 10);
    let stop = AtomicBool::new(true);

    assert!(CompiledCp::compile_interruptible(&model, &stop).unwrap().is_none());
}

#[test]
fn validated_cp_compilation_honors_a_prearmed_stop() {
    let mut model = Model::new();
    model.int_range(0, 10);
    let stop = AtomicBool::new(true);

    assert!(CompiledCp::compile_validated_with_estimate_interruptible(&model, 0, &stop).unwrap().is_none());
}

#[test]
fn validating_cp_compiler_entry_still_rejects_an_invalid_model() {
    let mut model = Model::new();
    model.int_set(vec![0, 0]);
    let stop = AtomicBool::new(false);
    let estimated = CompiledCp::estimate_semantic_bytes_interruptible(&model, &stop).unwrap();

    let error = match CompiledCp::compile_with_estimate_interruptible(&model, estimated, &stop) {
        Ok(_) => panic!("the validating compiler accepted a duplicate explicit domain"),
        Err(error) => error,
    };

    assert!(error.reason.contains("duplicate"));
}

#[test]
fn mono_tier_cp_consumes_its_only_root_problem_copy() {
    let mut model = Model::new();
    let value = model.int_range(0, 4);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });
    let request = SolveRequest::default();
    let budget = SolveBudget::new(None);
    let plan = compile_cp_plan(&model, &request, &budget).unwrap();
    let before = audit_cp_root_problem_clones();

    let result = solve_cp_plan(&model, &plan, &request, &budget, &mut IgnoreEvents).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(audit_cp_root_problem_clones() - before, 1);
}

#[test]
fn lexicographic_cp_keeps_one_root_copy_per_tier() {
    let mut model = Model::new();
    let first = model.int_range(0, 2);
    let second = model.int_range(0, 2);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(first) });
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(second) });
    let request = SolveRequest::default();
    let budget = SolveBudget::new(None);
    let plan = compile_cp_plan(&model, &request, &budget).unwrap();
    let before = audit_cp_root_problem_clones();

    let result = solve_cp_plan(&model, &plan, &request, &budget, &mut IgnoreEvents).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(audit_cp_root_problem_clones() - before, 2);
}

#[test]
fn absent_search_guidance_stays_allocation_free() {
    let mut model = Model::new();
    for _ in 0..1_000 {
        model.bool_var();
    }
    let stop = AtomicBool::new(false);
    let compiled = CompiledCp::compile_interruptible(&model, &stop).unwrap().unwrap();

    let (phase, branch_order, primary_branch_scope) = compiled.search_guidance_interruptible(&[], &[], None, &stop).unwrap().unwrap();

    assert!(phase.is_empty());
    assert!(branch_order.is_empty());
    assert!(primary_branch_scope.is_none());
}

#[test]
fn table_memory_estimate_counts_distinct_supports_not_tuple_square() {
    const ARITY: usize = 14;
    let mut model = Model::new();
    let variables = (0..ARITY).map(|_| model.bool_var()).collect::<Vec<_>>();
    let tuples =
        (0..(1usize << ARITY)).map(|bits| (0..ARITY).map(|column| ((bits >> column) & 1) as i32).collect::<Vec<_>>()).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, tuples, positive: true }));

    let estimate = CompiledCp::estimate_semantic_bytes_interruptible(&model, &AtomicBool::new(false)).unwrap();

    assert!(estimate < 10 * 1024 * 1024, "repetitive table estimate is still quadratic in tuple count: {estimate}");
}

#[test]
fn flat_expression_lowering_stops_while_collecting_operands() {
    let mut model = Model::new();
    let variable = model.int_range(0, 1);
    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();
    let expression = IntExpr::Add((0..1_000_000).map(|_| IntExpr::Variable(variable)).collect());
    let stop = AtomicBool::new(false);
    let barrier = Arc::new(Barrier::new(2));

    let result = std::thread::scope(|scope| {
        let trigger = Arc::clone(&barrier);
        let stop_ref = &stop;
        scope.spawn(move || {
            trigger.wait();
            std::thread::sleep(Duration::from_millis(1));
            stop_ref.store(true, Ordering::Release);
        });
        barrier.wait();
        compiled.compile_expression_interruptible(&expression, &stop)
    });

    assert!(result.unwrap().is_none());
}

#[test]
fn extension_template_build_stops_between_bitset_allocations() {
    const ARITY: usize = 6;
    const TUPLES: usize = 8_192;
    let tuples = (0..TUPLES).map(|row| (0..ARITY).map(|column| ((row + column) % TUPLES) as i32).collect::<Vec<_>>()).collect::<Vec<_>>();
    let mut solver = Solver::new();
    let variables = (0..ARITY).map(|_| solver.new_var_range(0, TUPLES as i32 - 1)).collect::<Vec<_>>();
    let stop = AtomicBool::new(false);
    let barrier = Arc::new(Barrier::new(2));

    let completed = std::thread::scope(|scope| {
        let trigger = Arc::clone(&barrier);
        let stop_ref = &stop;
        scope.spawn(move || {
            trigger.wait();
            std::thread::sleep(Duration::from_millis(1));
            stop_ref.store(true, Ordering::Release);
        });
        barrier.wait();
        table::extension_interruptible(&mut solver, &variables, &tuples, true, &stop)
    });

    assert!(!completed);
}

#[test]
fn regular_memory_preflight_rejects_before_layer_allocation() {
    let mut model = Model::new();
    let variables = (0..64).map(|_| model.bool_var()).collect();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Regular {
        variables,
        automaton: Automaton { states: 200_000, start: 0, accepting: vec![0], transitions: Vec::new() },
    }));
    let request =
        SolveRequest { limits: SolveLimits { memory_bytes: Some(1024 * 1024), ..SolveLimits::default() }, ..SolveRequest::default() };

    let error = match compile_cp_plan(&model, &request, &SolveBudget::new(None)) {
        Ok(_) => panic!("the over-budget regular plan was compiled"),
        Err(error) => error,
    };
    assert!(matches!(error, SolveError::Compile(message) if message.contains("memory limit") && message.contains("estimated CP backend")));
}
