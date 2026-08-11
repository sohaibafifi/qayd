use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use crate::constraints::table;
use crate::model::{Automaton, CompiledCp, Constraint, IntExpr, IntGlobalConstraint, Mdd, MddArc, Model, Objective, Relation};
use crate::orchestrator::{
    audit_cp_root_problem_clones, audit_integer_warm_start_compile, audit_integer_warm_start_preflight, compile_cp_plan, solve_cp_plan,
    IgnoreEvents, SolveBudget, SolveError, SolveLimits, SolveRequest, SolveStatus,
};
use crate::problem::Objective as PhysicalObjective;
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
fn integer_warm_start_preflight_counts_table_and_mdd_payloads() {
    const TABLE_ARITY: usize = 8;
    const TABLE_ROWS: usize = 4_096;
    const MDD_LAYERS: usize = 4;
    const MDD_ARCS_PER_LAYER: usize = 4_096;
    let mut model = Model::new();
    let table_variables = (0..TABLE_ARITY).map(|_| model.bool_var()).collect::<Vec<_>>();
    let tuples = (0..TABLE_ROWS).map(|row| (0..TABLE_ARITY).map(|column| ((row >> column) & 1) as i32).collect::<Vec<_>>()).collect();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables: table_variables, tuples, positive: true }));
    let mdd_variables = (0..MDD_LAYERS).map(|_| model.bool_var()).collect::<Vec<_>>();
    let layers = (0..MDD_LAYERS)
        .map(|_| (0..MDD_ARCS_PER_LAYER).map(|arc| MddArc { from: arc, value: (arc & 1) as i32, to: arc }).collect::<Vec<_>>())
        .collect();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Mdd {
        variables: mdd_variables,
        mdd: Mdd { layers, nodes_per_layer: vec![MDD_ARCS_PER_LAYER; MDD_LAYERS + 1] },
    }));

    let (local_bytes, structural_bytes, inspection_work) =
        audit_integer_warm_start_preflight(&model, &AtomicBool::new(false)).expect("preflight completed");
    let table_cells = TABLE_ARITY * TABLE_ROWS;
    let mdd_arcs = MDD_LAYERS * MDD_ARCS_PER_LAYER;
    assert!(local_bytes >= u64::try_from(table_cells * 4 + mdd_arcs * std::mem::size_of::<MddArc>()).unwrap());
    assert!(inspection_work >= u64::try_from(table_cells + mdd_arcs).unwrap());
    assert!(structural_bytes > 0);

    assert!(audit_integer_warm_start_preflight(&model, &AtomicBool::new(true)).is_none());
}

#[test]
fn optional_integer_warm_start_obeys_its_memory_allowance_and_prearmed_stop() {
    let mut model = Model::new();
    let cell = model.int_range(1, 1);
    let guard = model.bool_var();
    let index = model.int_range(0, 0);
    let letter = model.bool_var();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Element { array: vec![cell], index, value: letter }));
    model.add_constraint(Constraint::Intension(IntExpr::Or(vec![
        IntExpr::Eq(Box::new(IntExpr::Variable(guard)), Box::new(IntExpr::Constant(0))),
        IntExpr::Eq(Box::new(IntExpr::Variable(letter)), Box::new(IntExpr::Constant(1))),
    ])));
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
        variables: vec![index],
        tuples: vec![vec![0]],
        positive: true,
    }));
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Mul(vec![IntExpr::Constant(7), IntExpr::Variable(guard)]) });
    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();

    assert_eq!(audit_integer_warm_start_compile(&model, &compiled, 0, &AtomicBool::new(false)), (false, 0));
    let (local_plan_bytes, _, _) =
        audit_integer_warm_start_preflight(&model, &AtomicBool::new(false)).expect("warm-start preflight completed");
    assert_eq!(
        audit_integer_warm_start_compile(&model, &compiled, local_plan_bytes, &AtomicBool::new(false)),
        (false, 0),
        "the memory gate must include the transient normalized LS copy"
    );
    let (compiled_plan, estimated_bytes) = audit_integer_warm_start_compile(&model, &compiled, u64::MAX, &AtomicBool::new(false));
    assert!(compiled_plan);
    assert!(estimated_bytes > 0);
    assert_eq!(audit_integer_warm_start_compile(&model, &compiled, u64::MAX, &AtomicBool::new(true)), (false, 0));

    let cp_bytes = CompiledCp::estimate_semantic_bytes_interruptible(&model, &AtomicBool::new(false)).unwrap();
    let unlimited = compile_cp_plan(&model, &SolveRequest::default(), &SolveBudget::new(None)).unwrap();
    assert!(unlimited.estimated_backend_bytes() > cp_bytes, "the resident warm-start plan was not added to the CP estimate");
    let limited_request =
        SolveRequest { limits: SolveLimits { memory_bytes: Some(cp_bytes), ..SolveLimits::default() }, ..SolveRequest::default() };
    let limited = compile_cp_plan(&model, &limited_request, &SolveBudget::new(None)).unwrap();
    assert_eq!(limited.estimated_backend_bytes(), cp_bytes, "an optional warm start must be omitted when no memory remains");
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
fn affine_cp_objective_is_normalized_and_duplicate_terms_are_combined() {
    let mut model = Model::new();
    let x = model.bool_var();
    let y = model.bool_var();
    let expression = IntExpr::Sub(
        Box::new(IntExpr::Add(vec![
            IntExpr::Constant(7),
            IntExpr::Mul(vec![IntExpr::Constant(2), IntExpr::Variable(x)]),
            IntExpr::Neg(Box::new(IntExpr::Variable(x))),
            IntExpr::Variable(y),
            IntExpr::Variable(y),
        ])),
        Box::new(IntExpr::Add(vec![
            IntExpr::Constant(7),
            IntExpr::Neg(Box::new(IntExpr::Mul(vec![IntExpr::Constant(2), IntExpr::Variable(x)]))),
            IntExpr::Mul(vec![IntExpr::Constant(2), IntExpr::Constant(3), IntExpr::Variable(y)]),
        ])),
    );
    model.add_objective(Objective::IntExpr { minimize: true, expr: expression });

    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();

    let PhysicalObjective::Linear(minimize, coefficients, variables) = &compiled.objectives()[0] else {
        panic!("zero-constant affine objective was not normalized");
    };
    assert!(*minimize);
    assert_eq!(coefficients, &[3, -4]);
    assert_eq!(variables, compiled.int_variables());
}

#[test]
fn large_affine_cp_objective_stays_linear() {
    let mut model = Model::new();
    let variables = (0..64).map(|_| model.bool_var()).collect::<Vec<_>>();
    model.add_objective(Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(variables.iter().copied().map(IntExpr::Variable).collect()),
    });

    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();

    let PhysicalObjective::Linear(minimize, coefficients, physical_variables) = &compiled.objectives()[0] else {
        panic!("large affine objective was left as a symbolic expression");
    };
    assert!(*minimize);
    assert_eq!(coefficients, &vec![1; variables.len()]);
    assert_eq!(physical_variables, compiled.int_variables());
}

#[test]
fn materialized_objective_recovers_an_exact_affine_search_view() {
    let mut model = Model::new();
    let x = model.int_range(0, 4);
    let y = model.int_range(-2, 3);
    let objective = model.int_range(-20, 20);
    model.add_constraint(Constraint::Linear { terms: vec![(3, x), (-1, x), (-3, y), (-1, objective)], relation: Relation::Eq, rhs: 5 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(objective) });

    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();

    let PhysicalObjective::VarWithAffine(minimize, physical_objective, coefficients, variables) = &compiled.objectives()[0] else {
        panic!("unit defining equality did not produce an affine objective view");
    };
    assert!(*minimize);
    assert_eq!(*physical_objective, compiled.int_variables()[objective.0]);
    assert_eq!(coefficients, &[2, -3]);
    assert_eq!(variables, &compiled.int_variables()[..2]);
}

#[test]
fn positive_unit_objective_coefficient_recovers_the_same_affine_view() {
    let mut model = Model::new();
    let x = model.int_range(0, 4);
    let objective = model.int_range(9, 25);
    model.add_constraint(Constraint::Linear { terms: vec![(-4, x), (1, objective)], relation: Relation::Eq, rhs: 9 });
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(objective) });

    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();

    let PhysicalObjective::VarWithAffine(minimize, physical_objective, coefficients, variables) = &compiled.objectives()[0] else {
        panic!("positive unit defining equality did not produce an affine objective view");
    };
    assert!(!minimize);
    assert_eq!(*physical_objective, compiled.int_variables()[objective.0]);
    assert_eq!(coefficients, &[4]);
    assert_eq!(variables, &[compiled.int_variables()[x.0]]);
}

#[test]
fn nonunit_materialized_objective_equality_is_not_used_as_an_affine_view() {
    let mut model = Model::new();
    let x = model.int_range(0, 4);
    let objective = model.int_range(0, 4);
    model.add_constraint(Constraint::Linear { terms: vec![(1, x), (-2, objective)], relation: Relation::Eq, rhs: 0 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(objective) });

    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();

    assert!(matches!(compiled.objectives()[0], PhysicalObjective::Var(true, _)));
}

#[test]
fn cp_objective_linearization_safely_preserves_unsupported_expressions() {
    let mut model = Model::new();
    let x = model.bool_var();
    let y = model.bool_var();
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Constant(1)]) });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Mul(vec![IntExpr::Variable(x), IntExpr::Variable(y)]) });
    model.add_objective(Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(vec![
            IntExpr::Mul(vec![IntExpr::Constant(i64::MAX), IntExpr::Variable(x)]),
            IntExpr::Mul(vec![IntExpr::Constant(2), IntExpr::Variable(x)]),
        ]),
    });

    let compiled = CompiledCp::compile_interruptible(&model, &AtomicBool::new(false)).unwrap().unwrap();

    assert!(matches!(compiled.objectives()[0], PhysicalObjective::Expr(true, _)));
    assert!(matches!(compiled.objectives()[1], PhysicalObjective::Expr(true, _)));
    assert!(matches!(compiled.objectives()[2], PhysicalObjective::Expr(true, _)));
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
