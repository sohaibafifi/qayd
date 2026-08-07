use std::time::Duration;

use qayd::model::list::{ExprArena, Iterable, ReduceOp, Reduction, Resource};
use qayd::model::{
    BoolLiteral, Constraint, IntExpr, IntGlobalConstraint, IntVarRef, Model, ModelObject, ModelPackage, Objective, Relation, SourceRange,
};
use qayd::orchestrator::{
    compile_model_plan, solve_model, solve_model_silent, CpControls, EngineKind, EventCallback, EventControl, ExecutablePlan, ProofKind,
    SemanticAssumption, SemanticAssumptionOp, SolveBudget, SolveError, SolveEvent, SolveLimits, SolveMode, SolveRequest, SolveStatus,
    VerificationLevel,
};

fn list_count(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let one = arena.constant(1);
    Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body: one, coeff: 1 }
}

fn independent_package() -> ModelPackage {
    let mut model = Model::new();
    let left = model.int_range(0, 5);
    let right = model.int_range(5, 9);
    model.add_constraint(Constraint::Linear { terms: vec![(1, left)], relation: Relation::Ge, rhs: 2 });
    model.add_constraint(Constraint::Linear { terms: vec![(1, right)], relation: Relation::Le, rhs: 7 });
    let mut package = ModelPackage::new(model);
    package.metadata.names.insert(ModelObject::IntVar(left), "left".to_string());
    package.metadata.names.insert(ModelObject::IntVar(right), "right".to_string());
    package.metadata.outputs.extend([ModelObject::IntVar(left), ModelObject::IntVar(right)]);
    package.metadata.sources.insert(ModelObject::IntVar(left), vec![SourceRange { source: "memory".to_string(), start: 3, end: 7 }]);
    package.metadata.annotations.insert("frontend".to_string(), "test".to_string());
    package
}

#[test]
fn same_family_dependency_graph_compiles_to_two_components() {
    let package = independent_package();
    let request = SolveRequest::default();
    let budget = SolveBudget::new(Some(Duration::from_secs(5)));
    let plan = compile_model_plan(&package, &request, &budget).unwrap();
    let ExecutablePlan::Decomposed { components, .. } = plan.description() else {
        panic!("independent integer variables were not decomposed");
    };
    assert_eq!(components.len(), 2);

    let metadata = plan.audit_component_metadata();
    assert_eq!(metadata.len(), 2);
    assert!(metadata.iter().all(|metadata| metadata.annotations.get("frontend").map(String::as_str) == Some("test")));
    assert_eq!(metadata.iter().map(|metadata| metadata.outputs.len()).sum::<usize>(), 2);
    assert_eq!(metadata.iter().map(|metadata| metadata.names.len()).sum::<usize>(), 2);
    assert_eq!(metadata.iter().map(|metadata| metadata.sources.len()).sum::<usize>(), 1);
}

#[test]
fn independent_lexicographic_objectives_are_optimized_and_merged_in_semantic_order() {
    let mut model = Model::new();
    let left = model.int_range(0, 5);
    let right = model.int_range(0, 6);
    model.add_constraint(Constraint::Linear { terms: vec![(1, left)], relation: Relation::Ge, rhs: 2 });
    model.add_constraint(Constraint::Linear { terms: vec![(1, right)], relation: Relation::Le, rhs: 4 });
    let first = model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(left) });
    let second = model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(right) });
    let third = model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(left) });
    let mut package = ModelPackage::new(model);
    package.metadata.names.insert(ModelObject::Objective(first), "first".to_string());
    package.metadata.names.insert(ModelObject::Objective(second), "second".to_string());
    package.metadata.names.insert(ModelObject::Objective(third), "third".to_string());

    let plan = compile_model_plan(&package, &SolveRequest::default(), &SolveBudget::new(None)).unwrap();
    let ExecutablePlan::Decomposed { components, .. } = plan.description() else {
        panic!("separable objective tiers were not decomposed");
    };
    assert_eq!(components.len(), 2);
    assert_eq!(
        plan.audit_component_metadata()
            .iter()
            .flat_map(|metadata| metadata.names.keys())
            .filter(|object| matches!(object, ModelObject::Objective(_)))
            .count(),
        3
    );

    let result = solve_model_silent(&package, &SolveRequest::default()).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    let candidate = result.primal().unwrap();
    assert_eq!(candidate.assignment().integers[left.0], Some(2));
    assert_eq!(candidate.assignment().integers[right.0], Some(4));
    assert_eq!(candidate.objectives(), &[2, 4, 2]);
    assert_eq!(result.bounds().iter().map(|bound| (bound.tier, bound.value)).collect::<Vec<_>>(), vec![(0, 2), (1, 4), (2, 2)]);
    let proof = result.proof().expect("the independent exact optima must produce an aggregate proof");
    assert_eq!(proof.objective_tiers(), Some(3));
    assert!(matches!(proof.kind(), ProofKind::Decomposed { components, objective_tiers: 3 } if components.len() == 2));
}

#[test]
fn objective_only_cp_controls_are_not_forwarded_to_satisfaction_components() {
    let mut model = Model::new();
    let optimized = model.int_range(0, 3);
    let satisfaction_only = model.int_range(0, 1);
    model.add_constraint(Constraint::Linear { terms: vec![(1, optimized)], relation: Relation::Ge, rhs: 1 });
    model.add_constraint(Constraint::Linear { terms: vec![(1, satisfaction_only)], relation: Relation::Ge, rhs: 0 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(optimized) });
    let package = ModelPackage::new(model);
    let request = SolveRequest {
        mode: SolveMode::Exact,
        threads: 2,
        cp: CpControls { split: true, ..CpControls::default() },
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&package, &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), &[1]);
}

#[test]
fn an_objective_spanning_components_stays_on_the_single_engine_path() {
    let mut model = Model::new();
    let left = model.int_range(0, 3);
    let right = model.int_range(0, 3);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(left), IntExpr::Variable(right)]) });
    let package = ModelPackage::new(model);

    let plan = compile_model_plan(&package, &SolveRequest::default(), &SolveBudget::new(None)).unwrap();
    assert!(matches!(plan.description(), ExecutablePlan::Single(_)), "a coupled objective was incorrectly decomposed");

    let result = solve_model_silent(&package, &SolveRequest::default()).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), &[0]);
}

#[test]
fn a_constant_objective_keeps_independent_integer_variables_on_one_engine() {
    let mut model = Model::new();
    model.bool_var();
    model.bool_var();
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Constant(7) });

    let plan = compile_model_plan(&ModelPackage::new(model), &SolveRequest::default(), &SolveBudget::new(None)).unwrap();

    assert!(matches!(plan.description(), ExecutablePlan::Single(_)));
}

#[test]
fn disjoint_list_partitions_are_projected_solved_and_merged_in_tier_order() {
    let mut model = Model::new();
    let first = model.list(vec![1, 2]);
    let second = model.list(vec![10, 11, 12]);
    let first_partition = model.add_constraint(Constraint::ListPartition { lists: vec![first], items: vec![1, 2] });
    let second_partition = model.add_constraint(Constraint::ListPartition { lists: vec![second], items: vec![10, 11, 12] });
    let second_first = model.add_objective(Objective::ListTerms { minimize: true, terms: vec![list_count(second.0)], max_terms: None });
    let first_second = model.add_objective(Objective::ListTerms { minimize: true, terms: vec![list_count(first.0)], max_terms: None });
    let second_third = model.add_objective(Objective::ListTerms { minimize: false, terms: vec![list_count(second.0)], max_terms: None });
    let mut package = ModelPackage::new(model);
    package.metadata.names.insert(ModelObject::ListVar(first), "first".to_string());
    package.metadata.names.insert(ModelObject::ListVar(second), "second".to_string());
    package.metadata.names.insert(ModelObject::Constraint(first_partition), "partition-a".to_string());
    package.metadata.names.insert(ModelObject::Constraint(second_partition), "partition-b".to_string());
    package.metadata.names.insert(ModelObject::Objective(second_first), "tier-0".to_string());
    package.metadata.names.insert(ModelObject::Objective(first_second), "tier-1".to_string());
    package.metadata.names.insert(ModelObject::Objective(second_third), "tier-2".to_string());

    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };
    let plan = compile_model_plan(&package, &request, &SolveBudget::new(None)).unwrap();
    assert!(matches!(plan.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 2));
    let metadata = plan.audit_component_metadata();
    assert_eq!(metadata.iter().map(|metadata| metadata.names.len()).sum::<usize>(), 7);

    let result = solve_model_silent(&package, &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    let candidate = result.primal().unwrap();
    assert_eq!(candidate.assignment().lists[first.0], vec![1, 2]);
    assert_eq!(candidate.assignment().lists[second.0], vec![10, 11, 12]);
    assert_eq!(candidate.objectives(), &[3, 2, 3]);
    assert!(matches!(
        result.proof().map(|proof| proof.kind()),
        Some(ProofKind::Decomposed { components, objective_tiers: 3 }) if components.len() == 2
    ));
}

#[test]
fn list_hint_is_projected_to_each_independent_partition() {
    let mut model = Model::new();
    let first = model.list(vec![1, 2]);
    let second = model.list(vec![10, 11]);
    model.add_constraint(Constraint::ListPartition { lists: vec![first], items: vec![1, 2] });
    model.add_constraint(Constraint::ListPartition { lists: vec![second], items: vec![10, 11] });
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        list_hint: Some(vec![vec![2, 1], vec![11, 10]]),
        limits: SolveLimits { iterations: Some(0), ..SolveLimits::default() },
        ..SolveRequest::default()
    };

    let plan = compile_model_plan(&ModelPackage::new(model.clone()), &request, &SolveBudget::new(None)).unwrap();
    assert!(matches!(plan.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 2));
    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().assignment().lists, vec![vec![2, 1], vec![11, 10]]);
}

#[test]
fn a_list_objective_coupling_two_partitions_prevents_decomposition() {
    let mut model = Model::new();
    let first = model.list(vec![1, 2]);
    let second = model.list(vec![10, 11]);
    model.add_constraint(Constraint::ListPartition { lists: vec![first], items: vec![1, 2] });
    model.add_constraint(Constraint::ListPartition { lists: vec![second], items: vec![10, 11] });
    model.add_objective(Objective::ListTerms { minimize: true, terms: vec![list_count(first.0), list_count(second.0)], max_terms: None });

    let error = match compile_model_plan(&ModelPackage::new(model), &SolveRequest::default(), &SolveBudget::new(None)) {
        Ok(_) => panic!("a coupled list tier was incorrectly decomposed"),
        Err(error) => error,
    };
    assert!(matches!(error, SolveError::Compile(message) if message.contains("share one universe")));
}

#[test]
fn two_independent_schedules_are_projected_and_solved() {
    let mut model = Model::new();
    let a0 = model.interval(0, 5, 2);
    let a1 = model.interval(0, 5, 3);
    let b0 = model.interval(0, 5, 1);
    let b1 = model.interval(0, 5, 4);
    model.add_constraint(Constraint::IntervalResource(Resource::NoOverlap(vec![a0.0, a1.0])));
    model.add_constraint(Constraint::IntervalResource(Resource::NoOverlap(vec![b0.0, b1.0])));
    model.add_objective(Objective::Makespan { minimize: true, intervals: vec![b0, b1] });
    model.add_objective(Objective::Makespan { minimize: true, intervals: vec![a0, a1] });
    let package = ModelPackage::new(model);
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };

    let plan = compile_model_plan(&package, &request, &SolveBudget::new(None)).unwrap();
    assert!(matches!(plan.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 2));
    let result = solve_model_silent(&package, &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), &[5, 5]);
    for pair in [[a0.0, a1.0], [b0.0, b1.0]] {
        let intervals = &result.primal().unwrap().assignment().intervals;
        let left = intervals[pair[0]];
        let right = intervals[pair[1]];
        let left_start = left.start.unwrap();
        let right_start = right.start.unwrap();
        let left_duration = if pair[0] == a0.0 { 2 } else { 1 };
        let right_duration = if pair[1] == a1.0 { 3 } else { 4 };
        assert!(left_start + left_duration <= right_start || right_start + right_duration <= left_start);
    }
}

#[test]
fn interval_mode_identity_survives_compaction_and_merge() {
    let mut model = Model::new();
    let first = model.interval(0, 10, 0);
    let second = model.interval(0, 10, 0);
    let _first_slow = model.add_interval_mode(first, 3, 8, Some((0, 0))).unwrap();
    let second_mode = model.add_interval_mode(second, 4, 1, Some((0, 0))).unwrap();
    let first_fast = model.add_interval_mode(first, 3, 1, Some((4, 4))).unwrap();
    model.add_objective(Objective::Makespan { minimize: true, intervals: vec![first] });
    model.add_objective(Objective::Makespan { minimize: true, intervals: vec![second] });
    let mut package = ModelPackage::new(model);
    package.metadata.names.insert(ModelObject::IntervalMode(first_fast), "first-fast".to_string());
    package.metadata.names.insert(ModelObject::IntervalMode(second_mode), "second-mode".to_string());
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };

    let plan = compile_model_plan(&package, &request, &SolveBudget::new(None)).unwrap();
    assert!(matches!(plan.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 2));
    assert_eq!(
        plan.audit_component_metadata()
            .iter()
            .flat_map(|metadata| metadata.names.values())
            .filter(|name| matches!(name.as_str(), "first-fast" | "second-mode"))
            .count(),
        2
    );

    let result = solve_model_silent(&package, &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    let intervals = &result.primal().unwrap().assignment().intervals;
    assert_eq!(intervals[first.0].mode, Some(first_fast.0));
    assert_eq!(intervals[second.0].mode, Some(second_mode.0));
    assert_eq!(result.primal().unwrap().objectives(), &[5, 1]);
}

#[test]
fn a_makespan_spanning_independent_intervals_stays_on_one_engine() {
    let mut model = Model::new();
    let first = model.interval(0, 2, 1);
    let second = model.interval(0, 2, 2);
    model.add_objective(Objective::Makespan { minimize: true, intervals: vec![first, second] });

    let plan = compile_model_plan(&ModelPackage::new(model), &SolveRequest::default(), &SolveBudget::new(None)).unwrap();

    assert!(matches!(plan.description(), ExecutablePlan::Single(_)));
}

#[test]
fn integer_component_controls_are_remapped_to_local_indices() {
    let mut model = Model::new();
    let isolated = model.bool_var();
    let right = model.bool_var();
    let right_peer = model.bool_var();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent {
        variables: vec![right, right_peer],
        except: Vec::new(),
    }));
    let package = ModelPackage::new(model);
    let request = SolveRequest {
        mode: SolveMode::Exact,
        assumptions: vec![SemanticAssumption { variable: right.0, operation: SemanticAssumptionOp::Eq, value: 1 }],
        hints: vec![(right_peer.0, 0), (isolated.0, 0)],
        branch_order: vec![right.0, right_peer.0, isolated.0],
        ..SolveRequest::default()
    };

    let plan = compile_model_plan(&package, &request, &SolveBudget::new(None)).unwrap();
    assert!(matches!(plan.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 2));
    let result = solve_model_silent(&package, &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    let integers = &result.primal().unwrap().assignment().integers;
    assert_eq!(integers[right.0], Some(1));
    assert_eq!(integers[right_peer.0], Some(0));
}

#[test]
fn compact_integer_components_pass_an_aggregate_memory_preflight() {
    let mut model = Model::new();
    for _ in 0..64 {
        model.bool_var();
    }
    let request = SolveRequest {
        mode: SolveMode::Exact,
        limits: SolveLimits { memory_bytes: Some(5 * 1024 * 1024), ..SolveLimits::default() },
        ..SolveRequest::default()
    };

    let plan = compile_model_plan(&ModelPackage::new(model), &request, &SolveBudget::new(None)).unwrap();

    assert!(matches!(plan.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 64));
}

#[test]
fn exact_objective_components_from_distinct_families_have_a_composite_proof() {
    let mut model = Model::new();
    let value = model.int_range(2, 5);
    let task = model.interval(0, 3, 2);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });
    model.add_objective(Objective::Makespan { minimize: true, intervals: vec![task] });
    let package = ModelPackage::new(model);
    let request = SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() };

    let result = solve_model_silent(&package, &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), &[2, 2]);
    assert_eq!(result.bounds().iter().map(|bound| (bound.tier, bound.value)).collect::<Vec<_>>(), vec![(0, 2), (1, 2)]);
    let engines = result.reports().iter().filter_map(|report| report.engine).collect::<Vec<_>>();
    assert!(engines.contains(&EngineKind::IntegerExact));
    assert!(engines.contains(&EngineKind::ScheduleExact));
    assert!(matches!(
        result.proof().map(|proof| proof.kind()),
        Some(ProofKind::Decomposed { components, objective_tiers: 2 }) if components.len() == 2
    ));
}

#[test]
fn an_unsatisfiable_component_preserves_prior_bounds_and_proves_the_whole_model_unsatisfiable() {
    let mut model = Model::new();
    let optimized = model.int_range(2, 5);
    let impossible = model.int_range(0, 0);
    model.add_constraint(Constraint::Linear { terms: vec![(1, optimized)], relation: Relation::Ge, rhs: 2 });
    model.add_constraint(Constraint::Linear { terms: vec![(1, impossible)], relation: Relation::Ge, rhs: 1 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(optimized) });

    let result = solve_model_silent(&ModelPackage::new(model), &SolveRequest::default()).unwrap();

    assert_eq!(result.status(), SolveStatus::Unsatisfiable);
    assert!(result.primal().is_none());
    assert_eq!(result.bounds().iter().map(|bound| (bound.tier, bound.value)).collect::<Vec<_>>(), vec![(0, 2)]);
    let proof = result.proof().expect("the infeasible component must prove aggregate infeasibility");
    assert_eq!(proof.objective_tiers(), Some(1));
    assert!(matches!(proof.kind(), ProofKind::Decomposed { components, objective_tiers: 1 } if components.len() == 1));
}

#[test]
fn merged_component_assignment_is_replayed_on_the_original_model() {
    let result = solve_model_silent(&independent_package(), &SolveRequest::default()).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    let candidate = result.primal().unwrap();
    assert_eq!(candidate.verification(), VerificationLevel::Final);
    let left = candidate.assignment().integers[IntVarRef(0).0].unwrap();
    let right = candidate.assignment().integers[IntVarRef(1).0].unwrap();
    assert!((2..=5).contains(&left));
    assert!((5..=7).contains(&right));

    let expected_nodes = result.reports().iter().map(|report| report.search.nodes).sum::<u64>();
    let expected_improvements = result.reports().iter().map(|report| report.improvements).sum::<u64>();
    let expected_elapsed = result.reports().iter().fold(std::time::Duration::ZERO, |total, report| total + report.elapsed);
    assert_eq!(result.aggregate_search_stats().nodes, expected_nodes);
    assert_eq!(result.improvements(), expected_improvements);
    assert_eq!(result.elapsed(), expected_elapsed);
}

#[test]
fn malformed_source_ranges_are_rejected_before_compilation() {
    let mut model = Model::new();
    let value = model.bool_var();
    let mut package = ModelPackage::new(model);
    package.metadata.sources.insert(ModelObject::IntVar(value), vec![SourceRange { source: " ".to_string(), start: 9, end: 2 }]);

    let errors = package.validate().unwrap_err();
    assert!(errors.iter().any(|error| error.contains("empty source identifier")));
    assert!(errors.iter().any(|error| error.contains("after its end")));
}

#[test]
fn decomposition_rejects_limits_that_a_component_family_cannot_consume() {
    let mut mixed_list = Model::new();
    mixed_list.int_range(0, 1);
    let list = mixed_list.list(vec![1]);
    mixed_list.add_constraint(Constraint::ListPartition { lists: vec![list], items: vec![1] });
    let conflict_request = SolveRequest { limits: SolveLimits { conflicts: Some(1), ..SolveLimits::default() }, ..SolveRequest::default() };
    let conflict_error = match compile_model_plan(&ModelPackage::new(mixed_list), &conflict_request, &SolveBudget::new(None)) {
        Ok(_) => panic!("a list component silently discarded the conflict limit"),
        Err(error) => error,
    };
    assert!(matches!(conflict_error, SolveError::InvalidRequest(message) if message.contains("do not consume conflicts")));

    let mut mixed_schedule = Model::new();
    mixed_schedule.int_range(0, 1);
    mixed_schedule.interval(0, 0, 1);
    let iteration_request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: SolveLimits { iterations: Some(1), ..SolveLimits::default() },
        ..SolveRequest::default()
    };
    let iteration_error = match compile_model_plan(&ModelPackage::new(mixed_schedule), &iteration_request, &SolveBudget::new(None)) {
        Ok(_) => panic!("an interval component silently discarded the iteration limit"),
        Err(error) => error,
    };
    assert!(matches!(iteration_error, SolveError::InvalidRequest(message) if message.contains("do not consume iterations")));
}

#[test]
fn decomposition_rejects_incumbent_assignment_publication_instead_of_dropping_component_candidates() {
    let package = independent_package();
    let baseline = compile_model_plan(&package, &SolveRequest::default(), &SolveBudget::new(None)).unwrap();
    assert!(matches!(baseline.description(), ExecutablePlan::Decomposed { components, .. } if components.len() == 2));

    let request = SolveRequest { publish_incumbent_assignments: true, ..SolveRequest::default() };
    let error = match compile_model_plan(&package, &request, &SolveBudget::new(None)) {
        Ok(_) => panic!("a decomposed plan silently accepted component-local incumbent assignment publication"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        SolveError::InvalidRequest(message)
            if message.contains("publish_incumbent_assignments") && message.contains("not complete assignments")
    ));
}

#[test]
fn component_reports_reach_the_parent_sink_once() {
    let package = independent_package();
    let mut events = Vec::new();
    let mut sink = EventCallback(|event| {
        events.push(event);
        Ok(EventControl::Continue)
    });

    let result = solve_model(&package, &SolveRequest::default(), &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(events.iter().filter(|event| matches!(event, SolveEvent::Finished(_))).count(), 2);
    assert_eq!(events.iter().filter(|event| matches!(event, SolveEvent::Candidate(_))).count(), 1);
}

#[test]
fn stopping_on_a_component_report_stops_before_the_next_component() {
    let package = independent_package();
    let mut events = Vec::new();
    let mut sink = EventCallback(|event| {
        let control = if matches!(event, SolveEvent::Finished(_)) { EventControl::Stop } else { EventControl::Continue };
        events.push(event);
        Ok(control)
    });

    let result = solve_model(&package, &SolveRequest::default(), &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.message().is_some_and(|message| message.contains("EventSink")));
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], SolveEvent::Finished(_)));
}

#[test]
fn incomplete_component_diagnostic_is_preserved() {
    let request = SolveRequest {
        mode: SolveMode::Exact,
        limits: SolveLimits { conflicts: Some(0), ..SolveLimits::default() },
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&independent_package(), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Unknown);
    assert_eq!(result.message(), Some("search stopped: ConflictLimit"));
}

fn easy_component_plus_pigeonhole(pigeons: usize, holes: usize) -> ModelPackage {
    let mut model = Model::new();
    let easy = model.int_range(0, 0);
    model.add_constraint(Constraint::Linear { terms: vec![(1, easy)], relation: Relation::Eq, rhs: 0 });
    let variables = (0..pigeons * holes).map(|_| model.bool_var()).collect::<Vec<_>>();
    for pigeon in 0..pigeons {
        model.add_constraint(Constraint::Clause(
            (0..holes).map(|hole| BoolLiteral { variable: variables[pigeon * holes + hole], positive: true }).collect(),
        ));
    }
    for hole in 0..holes {
        for left in 0..pigeons {
            for right in left + 1..pigeons {
                model.add_constraint(Constraint::Clause(vec![
                    BoolLiteral { variable: variables[left * holes + hole], positive: false },
                    BoolLiteral { variable: variables[right * holes + hole], positive: false },
                ]));
            }
        }
    }
    ModelPackage::new(model)
}

#[test]
fn unused_component_conflicts_remain_available_to_later_components() {
    let package = easy_component_plus_pigeonhole(5, 4);
    let complete = solve_model_silent(&package, &SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() }).unwrap();
    assert_eq!(complete.status(), SolveStatus::Unsatisfiable);
    let required = complete.aggregate_search_stats().failures;
    assert!(required > 2, "the causal fixture unexpectedly needed only {required} conflicts");
    let total = required.saturating_add(1);
    let request = SolveRequest {
        mode: SolveMode::Exact,
        limits: SolveLimits { conflicts: Some(total), ..SolveLimits::default() },
        ..SolveRequest::default()
    };

    let limited = solve_model_silent(&package, &request).unwrap();

    assert_eq!(limited.status(), SolveStatus::Unsatisfiable);
    assert!(limited.aggregate_search_stats().failures <= total);
}
