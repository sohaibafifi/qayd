use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use qayd::model::list::{ExprArena, Iterable, ReduceOp, Reduction};
use qayd::model::{Constraint, IntExpr, Model, ModelPackage, Objective, Relation};
#[cfg(feature = "python")]
use qayd::orchestrator::SemanticSolveSession;
use qayd::orchestrator::{
    compile_model_plan, solve_model, solve_model_silent, solve_model_with_external_stop, solve_model_with_stop, CpControls, EngineKind,
    EventCallback, EventControl, ExecutablePlan, IgnoreEvents, ProofRequest, SolveBudget, SolveEvent, SolveMode, SolveRequest, SolveStatus,
    VerificationLevel,
};

fn count(list: usize) -> Reduction {
    let mut arena = ExprArena::default();
    let one = arena.constant(1);
    Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body: one, coeff: 1 }
}

fn semantic_list_model(items: usize) -> Model {
    let mut model = Model::new();
    let universe = (0..items as i32).collect::<Vec<_>>();
    let lists = vec![model.list(universe.clone()), model.list(universe.clone())];
    model.add_constraint(Constraint::ListPartition { lists, items: universe });
    model.add_objective(Objective::ListTerms { minimize: true, terms: vec![count(0), count(1)], max_terms: None });
    model
}

#[test]
fn semantic_cp_model_is_compiled_solved_and_replayed() {
    let mut model = Model::new();
    let x = model.int_range(0, 9);
    let y = model.int_range(0, 9);
    model.add_constraint(Constraint::Linear { terms: vec![(1, x), (1, y)], relation: Relation::Ge, rhs: 5 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Variable(y)]) });

    let request = SolveRequest { mode: SolveMode::Exact, threads: 2, ..SolveRequest::default() };
    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [5]);
    assert_eq!(result.primal().unwrap().verification(), VerificationLevel::Final);
    assert!(result.proof().is_some());
    result.validate_contract().unwrap();
}

#[test]
fn symbolic_cp_objective_portfolio_never_crosses_a_semantic_constraint() {
    for threads in [1, 2] {
        for seed in 0..16 {
            let mut model = Model::new();
            let x = model.int_range(0, 9);
            let y = model.int_range(0, 9);
            model.add_constraint(Constraint::Linear { terms: vec![(1, x), (1, y)], relation: Relation::Ge, rhs: 5 });
            model
                .add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Variable(y)]) });

            let request = SolveRequest { mode: SolveMode::Exact, threads, seed, ..SolveRequest::default() };
            let result = solve_model_silent(&ModelPackage::new(model), &request)
                .unwrap_or_else(|error| panic!("symbolic objective failed with {threads} threads for seed {seed}: {error}"));
            let integers = &result.primal().unwrap().assignment().integers;
            assert!(
                integers[0].unwrap() + integers[1].unwrap() >= 5,
                "invalid assignment with {threads} threads for seed {seed}: {integers:?}"
            );
            assert_eq!(result.primal().unwrap().objectives(), [5]);
        }
    }
}

#[test]
fn borrowed_external_stop_reaches_the_canonical_budget() {
    let mut model = Model::new();
    model.int_range(0, 1);
    let stop = AtomicBool::new(true);
    let result = solve_model_with_external_stop(&ModelPackage::new(model), &SolveRequest::default(), &stop, &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.message().is_some_and(|message| message.contains("ExternalCancellation")));
}

#[test]
fn invalid_requests_win_over_expired_or_prearmed_budgets_at_every_entrypoint() {
    let package = ModelPackage::new(Model::new());
    let request = SolveRequest {
        threads: 0,
        limits: qayd::orchestrator::SolveLimits { time: Some(Duration::ZERO), ..Default::default() },
        ..SolveRequest::default()
    };

    let direct = solve_model(&package, &request, &mut IgnoreEvents).unwrap_err();
    assert!(direct.to_string().contains("threads"), "{direct}");

    let compile = match compile_model_plan(&package, &request, &SolveBudget::new(Some(Duration::ZERO))) {
        Ok(_) => panic!("compilation accepted an invalid request under an expired budget"),
        Err(error) => error,
    };
    assert!(compile.to_string().contains("threads"), "{compile}");

    let shared = solve_model_with_stop(&package, &request, Arc::new(AtomicBool::new(true)), &mut IgnoreEvents).unwrap_err();
    assert!(shared.to_string().contains("threads"), "{shared}");

    let external = AtomicBool::new(true);
    let borrowed = solve_model_with_external_stop(&package, &request, &external, &mut IgnoreEvents).unwrap_err();
    assert!(borrowed.to_string().contains("threads"), "{borrowed}");
}

#[test]
fn cp_root_propagation_is_shared_before_portfolio_cloning() {
    let mut model = Model::new();
    let value = model.int_range(0, 0);
    model.add_constraint(Constraint::Linear { terms: vec![(1, value)], relation: Relation::Ge, rhs: 1 });
    let request = SolveRequest { mode: SolveMode::Exact, threads: 4, ..SolveRequest::default() };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Unsatisfiable);
    assert_eq!(result.aggregate_search_stats().failures, 1);
    assert!(result.proof().is_some());
}

#[test]
fn cp_root_propagation_preserves_the_complete_semantic_assignment() {
    let mut model = Model::new();
    let value = model.int_range(0, 1);
    model.add_constraint(Constraint::Linear { terms: vec![(1, value)], relation: Relation::Ge, rhs: 1 });
    let request = SolveRequest { mode: SolveMode::Exact, threads: 4, ..SolveRequest::default() };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().assignment().integers, [Some(1)]);
    assert_eq!(result.primal().unwrap().verification(), VerificationLevel::Final);
}

#[test]
fn semantic_integer_local_search_uses_the_canonical_plan_and_replay() {
    let mut model = Model::new();
    let value = model.int_range(7, 7);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });
    let request = SolveRequest { mode: SolveMode::LocalSearch, threads: 2, ..SolveRequest::default() };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().objectives(), [7]);
    assert_eq!(result.primal().unwrap().verification(), VerificationLevel::Final);
    assert_eq!(result.primal().unwrap().source(), EngineKind::IntegerLocalSearch);
    assert!(result.proof().is_none());
    assert!(result.reports().iter().all(|report| report.engine == Some(qayd::orchestrator::EngineKind::IntegerLocalSearch)));
    result.validate_contract().unwrap();
}

#[test]
fn required_proofs_reject_heuristic_plans_and_accept_exact_completion() {
    let mut heuristic_model = Model::new();
    let heuristic_value = heuristic_model.int_range(7, 7);
    heuristic_model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(heuristic_value) });
    let heuristic_request = SolveRequest { mode: SolveMode::LocalSearch, proof: ProofRequest::Require, ..SolveRequest::default() };
    let error = match compile_model_plan(&ModelPackage::new(heuristic_model), &heuristic_request, &SolveBudget::new(None)) {
        Ok(_) => panic!("the heuristic plan accepted a required completion proof"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("proof=Require"), "{error}");

    let mut exact_model = Model::new();
    let exact_value = exact_model.int_range(0, 3);
    exact_model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(exact_value) });
    let exact_request = SolveRequest { mode: SolveMode::Exact, proof: ProofRequest::Require, ..SolveRequest::default() };
    let result = solve_model_silent(&ModelPackage::new(exact_model), &exact_request).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert!(result.proof().is_some());

    let mut interrupted_model = Model::new();
    let interrupted_value = interrupted_model.int_range(0, 3);
    interrupted_model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(interrupted_value) });
    let interrupted_request = SolveRequest {
        mode: SolveMode::Exact,
        proof: ProofRequest::Require,
        limits: qayd::orchestrator::SolveLimits { time: Some(Duration::ZERO), ..Default::default() },
        ..SolveRequest::default()
    };
    let error = solve_model_silent(&ModelPackage::new(interrupted_model), &interrupted_request).unwrap_err();
    assert!(error.to_string().contains("required proof"), "{error}");
}

#[test]
#[cfg(feature = "python")]
fn session_specific_validation_and_required_proofs_win_over_cancellation() {
    let mut invalid_session = SemanticSolveSession::new(ModelPackage::new(Model::new())).unwrap();
    let invalid_request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: qayd::orchestrator::SolveLimits { time: Some(Duration::ZERO), ..Default::default() },
        ..SolveRequest::default()
    };
    let error = invalid_session.solve_with_external_stop(&invalid_request, &AtomicBool::new(true), &mut IgnoreEvents).unwrap_err();
    assert!(error.to_string().contains("local-search-only") || error.to_string().contains("exact mode"), "{error}");

    let mut proof_model = Model::new();
    let value = proof_model.int_range(0, 3);
    proof_model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });
    let mut proof_session = SemanticSolveSession::new(ModelPackage::new(proof_model)).unwrap();
    let proof_request = SolveRequest {
        mode: SolveMode::Exact,
        proof: ProofRequest::Require,
        limits: qayd::orchestrator::SolveLimits { time: Some(Duration::ZERO), ..Default::default() },
        ..SolveRequest::default()
    };
    let error = proof_session.solve_with_external_stop(&proof_request, &AtomicBool::new(true), &mut IgnoreEvents).unwrap_err();
    assert!(error.to_string().contains("required session proof"), "{error}");
}

#[test]
fn collection_portfolio_policy_belongs_to_the_canonical_compiler() {
    let package = ModelPackage::new(semantic_list_model(12));
    let request = SolveRequest {
        mode: SolveMode::Auto,
        threads: 2,
        limits: qayd::orchestrator::SolveLimits { iterations: Some(4), ..Default::default() },
        ..SolveRequest::default()
    };
    let budget = SolveBudget::new(None);
    let plan = compile_model_plan(&package, &request, &budget).unwrap();
    assert!(matches!(plan.description(), ExecutablePlan::Portfolio(workers)
        if workers.len() == 2
            && workers.iter().all(|worker| matches!(worker, ExecutablePlan::Single(engine) if engine.engine() == EngineKind::ListLocalSearch))));

    let result = solve_model_silent(&package, &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.reports()[0].engine, Some(EngineKind::ListLocalSearch));
}

#[test]
fn exact_collection_plan_rejects_an_ignored_thread_count() {
    let package = ModelPackage::new(semantic_list_model(3));
    let request = SolveRequest { mode: SolveMode::Exact, threads: 2, ..SolveRequest::default() };
    let error = solve_model_silent(&package, &request).unwrap_err();
    assert!(error.to_string().contains("threads=1"));
}

#[test]
fn cp_compiler_rejects_probe_workers_for_a_symbolic_objective() {
    let mut model = Model::new();
    let x = model.int_range(0, 3);
    let y = model.int_range(0, 3);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Variable(y)]) });
    let request = SolveRequest {
        mode: SolveMode::Exact,
        threads: 2,
        cp: CpControls { probes: 1, ..CpControls::default() },
        ..SolveRequest::default()
    };

    let error = solve_model_silent(&ModelPackage::new(model), &request).unwrap_err();
    assert!(error.to_string().contains("materialized variable"), "{error}");
}

#[test]
fn cp_compiler_rejects_auxiliary_roles_without_a_complete_worker() {
    let mut model = Model::new();
    let x = model.int_range(0, 3);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(x) });
    let request = SolveRequest {
        mode: SolveMode::Exact,
        threads: 3,
        cp: CpControls { probes: 2, lns: 1, ..CpControls::default() },
        ..SolveRequest::default()
    };

    let error = solve_model_silent(&ModelPackage::new(model), &request).unwrap_err();
    assert!(error.to_string().contains("leave no complete worker"), "{error}");
}

#[test]
fn cp_compiler_rejects_optimization_controls_on_a_satisfaction_model() {
    let mut model = Model::new();
    model.bool_var();
    let request =
        SolveRequest { mode: SolveMode::Exact, threads: 2, cp: CpControls { lns: 1, ..CpControls::default() }, ..SolveRequest::default() };

    let error = solve_model_silent(&ModelPackage::new(model), &request).unwrap_err();
    assert!(error.to_string().contains("require an optimization objective"), "{error}");
}

#[test]
fn semantic_integer_local_search_honors_the_shared_iteration_limit() {
    let mut model = Model::new();
    let value = model.int_range(0, 20);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: qayd::orchestrator::SolveLimits { iterations: Some(0), time: Some(Duration::from_secs(10)), ..Default::default() },
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none());
    assert_eq!(result.reports()[0].search.nodes, 0);
    assert!(result.message().is_some_and(|message| message.contains("IterationLimit")));
}

#[test]
fn semantic_integer_local_search_accepts_satisfaction_models() {
    let mut model = Model::new();
    let value = model.bool_var();
    model.add_constraint(Constraint::Linear { terms: vec![(1, value)], relation: Relation::Eq, rhs: 1 });
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        limits: qayd::orchestrator::SolveLimits { iterations: Some(100), ..Default::default() },
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.primal().unwrap().assignment().integers, [Some(1)]);
    assert!(result.primal().unwrap().objectives().is_empty());
    assert!(result.proof().is_none());
}

#[test]
fn local_search_portfolios_share_one_total_iteration_quota() {
    let mut integer = Model::new();
    let value = integer.int_range(0, 100);
    integer.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });
    let integer_request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 4,
        limits: qayd::orchestrator::SolveLimits { iterations: Some(7), ..Default::default() },
        ..SolveRequest::default()
    };
    let integer_result = solve_model_silent(&ModelPackage::new(integer), &integer_request).unwrap();
    assert!(integer_result.aggregate_search_stats().nodes <= 7);

    let list_request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 4,
        limits: qayd::orchestrator::SolveLimits { iterations: Some(7), ..Default::default() },
        ..SolveRequest::default()
    };
    let list_result = solve_model_silent(&ModelPackage::new(semantic_list_model(16)), &list_request).unwrap();
    assert!(list_result.aggregate_search_stats().nodes <= 7);
}

#[test]
fn an_early_integer_ls_fixpoint_is_not_reported_as_iteration_exhaustion() {
    let mut model = Model::new();
    let value = model.int_range(9, 9);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });
    let request = SolveRequest {
        mode: SolveMode::LocalSearch,
        threads: 4,
        limits: qayd::orchestrator::SolveLimits { iterations: Some(10_000), ..Default::default() },
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();
    assert!(result.message().is_none_or(|message| !message.contains("IterationLimit")));
}

#[test]
fn cp_portfolio_conflicts_are_one_aggregate_limit() {
    let pigeons = 5usize;
    let holes = 4usize;
    let mut model = Model::new();
    let variables = (0..pigeons * holes).map(|_| model.bool_var()).collect::<Vec<_>>();
    for pigeon in 0..pigeons {
        model.add_constraint(Constraint::Clause(
            (0..holes).map(|hole| qayd::model::BoolLiteral { variable: variables[pigeon * holes + hole], positive: true }).collect(),
        ));
    }
    for hole in 0..holes {
        for left in 0..pigeons {
            for right in left + 1..pigeons {
                model.add_constraint(Constraint::Clause(vec![
                    qayd::model::BoolLiteral { variable: variables[left * holes + hole], positive: false },
                    qayd::model::BoolLiteral { variable: variables[right * holes + hole], positive: false },
                ]));
            }
        }
    }
    let request = SolveRequest {
        mode: SolveMode::Exact,
        threads: 4,
        limits: qayd::orchestrator::SolveLimits { conflicts: Some(1), ..Default::default() },
        ..SolveRequest::default()
    };

    let result = solve_model_silent(&ModelPackage::new(model), &request).unwrap();
    assert!(result.aggregate_search_stats().failures <= 1, "portfolio reported {:?}", result.aggregate_search_stats());
    if result.status() == SolveStatus::Unknown {
        assert!(result.message().is_some_and(|message| message.contains("ConflictLimit")));
    }
}

#[test]
#[cfg(feature = "python")]
fn physical_session_respects_zero_conflicts_for_decision_and_lexicographic_search() {
    for optimize in [false, true] {
        let mut model = Model::new();
        let value = model.bool_var();
        if optimize {
            model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });
        }
        let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
        let request = SolveRequest {
            mode: SolveMode::Exact,
            limits: qayd::orchestrator::SolveLimits { conflicts: Some(0), ..Default::default() },
            ..SolveRequest::default()
        };
        let result = session.solve_with_external_stop(&request, &AtomicBool::new(false), &mut IgnoreEvents).unwrap();
        assert_eq!(result.status(), SolveStatus::Unknown);
        assert_eq!(result.aggregate_search_stats().failures, 0);
        assert!(result.message().is_some_and(|message| message.contains("ConflictLimit")));
    }
}

#[test]
fn crate_root_solve_is_the_canonical_model_entry_point() {
    let mut model = Model::new();
    let value = model.int_range(4, 4);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });

    let result = qayd::solve(&ModelPackage::new(model), &SolveRequest::default(), &mut IgnoreEvents).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [4]);
}

#[test]
fn semantic_set_lowering_preserves_membership_constraints() {
    let mut model = Model::new();
    let left = model.set_bounds([1], [1, 2]).unwrap();
    let right = model.set([1, 2, 3]);
    model.add_constraint(Constraint::SetSubset { subset: left, superset: right });
    model.add_constraint(Constraint::SetCardinality { set: right, min: 1, max: 2 });

    let result = solve_model_silent(&ModelPackage::new(model), &SolveRequest::default()).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    let primal = result.primal().unwrap();
    let sets = &primal.assignment().sets;
    assert!(sets[0].contains(&1));
    assert!(sets[1].contains(&1));
}

#[test]
fn final_candidate_event_is_verified_before_publication() {
    let mut model = Model::new();
    let value = model.int_range(3, 3);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(value) });
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let mut sink = EventCallback(move |event| {
        captured.lock().unwrap().push(event);
        Ok(EventControl::Continue)
    });

    let result = solve_model(&ModelPackage::new(model), &SolveRequest::default(), &mut sink).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|event| { matches!(event, SolveEvent::Candidate(candidate) if candidate.verification() == VerificationLevel::Final) }));
}

#[test]
fn independent_mixed_model_is_decomposed_and_replayed_as_one_assignment() {
    let mut model = Model::new();
    model.int_range(0, 1);
    let list = model.list(vec![1]);
    model.add_constraint(Constraint::ListPartition { lists: vec![list], items: vec![1] });
    let result = solve_model_silent(&ModelPackage::new(model), &SolveRequest::default()).unwrap();
    assert_eq!(result.status(), SolveStatus::Satisfiable);
    let primal = result.primal().unwrap();
    let assignment = primal.assignment();
    assert_eq!(assignment.integers.len(), 1);
    assert_eq!(assignment.lists, vec![vec![1]]);
}
