use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(feature = "lp-relaxation")]
use std::time::Duration;
use std::time::Instant;

use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{
    execute_workers_silent, solve_model_with_external_stop, CpControls, EventControl, EventSink, IgnoreEvents, SemanticAssumption,
    SemanticAssumptionOp, SemanticSolveSession, SolveError, SolveEvent, SolveLimits, SolveMode, SolveRequest, SolveResult, SolveStatus,
};
#[cfg(feature = "lp-relaxation")]
use qayd::orchestrator::{LinearBackendMode, LinearControls};
#[cfg(feature = "lp-relaxation")]
use qayd::search::LinearModelStatus;
use qayd::{AssumptionResult, BoolLit, Event, Inconsistency, PropId, Propagator, Solver, Store, VarId};

fn solve(session: &mut SemanticSolveSession, request: &SolveRequest) -> Result<SolveResult, SolveError> {
    session.solve_with_external_stop(request, &AtomicBool::new(false), &mut IgnoreEvents)
}

fn metadata<'a>(result: &'a SolveResult, key: &str) -> &'a str {
    result
        .reports()
        .iter()
        .flat_map(|report| &report.metadata)
        .find_map(|(name, value)| (name == key).then_some(value.as_str()))
        .unwrap_or_else(|| panic!("missing session metadata {key}"))
}

fn run_ignored_subprocess_case(test: &str) {
    let mut child = Command::new(std::env::current_exe().unwrap()).args(["--ignored", "--exact", test, "--nocapture"]).spawn().unwrap();
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "subprocess {test} failed with {status}");
            return;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let _ = child.wait();
            panic!("subprocess {test} deadlocked");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[derive(Clone)]
struct StopAfterRootPropagation {
    variable: VarId,
    stop: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
}

impl Propagator for StopAfterRootPropagation {
    fn register(&mut self, store: &mut Store, me: PropId) {
        store.subscribe(self.variable, me, Event::Fix);
    }

    fn propagate(&mut self, _: &mut Store) -> Result<(), Inconsistency> {
        if self.calls.fetch_add(1, Ordering::AcqRel) > 0 {
            self.stop.store(true, Ordering::Release);
        }
        Ok(())
    }
}

#[test]
fn assumption_search_reports_interruption_after_assumption_or_search_propagation() {
    for assumption_driven in [true, false] {
        let stop = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut solver = Solver::new();
        let variable = solver.new_var_range(0, 1);
        solver.post(Box::new(StopAfterRootPropagation { variable, stop: Arc::clone(&stop), calls: Arc::clone(&calls) }));
        let cube = assumption_driven.then_some(BoolLit { var: variable, value: true });
        let outcome = qayd::solve_under_assumptions(&mut solver, &[variable], cube.as_slice(), stop.as_ref()).unwrap();
        assert_eq!(outcome, AssumptionResult::Interrupted);
        assert!(calls.load(Ordering::Acquire) >= 2);
    }
}

fn lexicographic_model() -> (Model, usize, usize) {
    let mut model = Model::new();
    let x = model.int_range(0, 3);
    let y = model.int_range(0, 3);
    model.add_constraint(Constraint::Linear { terms: vec![(1, x), (1, y)], relation: Relation::Ge, rhs: 3 });
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Variable(y)]) });
    model.add_objective(Objective::IntExpr { minimize: false, expr: IntExpr::Variable(x) });
    (model, x.0, y.0)
}

#[test]
fn multiworker_lexicographic_epochs_scope_the_proved_prefix() {
    let (model, x, y) = lexicographic_model();
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let request = SolveRequest { mode: SolveMode::Exact, threads: 4, seed: 17, ..SolveRequest::default() };

    let free = solve(&mut session, &request).unwrap();
    assert_eq!(free.status(), SolveStatus::Optimal);
    assert_eq!(free.primal().unwrap().objectives(), [3, 3]);
    assert_eq!(free.primal().unwrap().assignment().integers.len(), 2);
    assert_eq!(metadata(&free, "clause_session_first_worker"), "0");
    assert_eq!(metadata(&free, "clause_session_next_worker"), "8");

    let pinned_request = SolveRequest {
        assumptions: vec![SemanticAssumption { variable: x, operation: SemanticAssumptionOp::Le, value: 1 }],
        ..request.clone()
    };
    let pinned = solve(&mut session, &pinned_request).unwrap();
    assert_eq!(pinned.status(), SolveStatus::Optimal);
    assert_eq!(pinned.primal().unwrap().objectives(), [3, 1]);
    assert_eq!(pinned.primal().unwrap().assignment().integers[x], Some(1));
    assert_eq!(pinned.primal().unwrap().assignment().integers[y], Some(2));
    assert_eq!(metadata(&pinned, "clause_session_first_worker"), "8");
    assert_eq!(metadata(&pinned, "clause_session_next_worker"), "16");

    let forced_request = SolveRequest {
        assumptions: vec![
            SemanticAssumption { variable: x, operation: SemanticAssumptionOp::Eq, value: 3 },
            SemanticAssumption { variable: y, operation: SemanticAssumptionOp::Eq, value: 3 },
        ],
        ..request.clone()
    };
    let forced = solve(&mut session, &forced_request).unwrap();
    assert_eq!(forced.status(), SolveStatus::Optimal);
    assert_eq!(forced.primal().unwrap().objectives(), [6, 3]);
    assert_eq!(metadata(&forced, "clause_session_first_worker"), "16");
    assert_eq!(metadata(&forced, "clause_session_next_worker"), "24");

    let free_again = solve(&mut session, &request).unwrap();
    assert_eq!(free_again.status(), SolveStatus::Optimal);
    assert_eq!(free_again.primal().unwrap().objectives(), [3, 3]);
    assert_eq!(metadata(&free_again, "clause_session_first_worker"), "24");
    assert_eq!(metadata(&free_again, "clause_session_next_worker"), "32");
}

#[test]
fn repeated_multiworker_lexicographic_cancellation_never_publishes_partial_propagation() {
    let (model, _, _) = lexicographic_model();
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let request = SolveRequest { mode: SolveMode::Exact, threads: 4, seed: 17, ..SolveRequest::default() };
    for epoch in 0..40 {
        let result = solve(&mut session, &request).unwrap_or_else(|error| panic!("epoch {epoch}: {error}"));
        assert_eq!(result.status(), SolveStatus::Optimal, "epoch {epoch}: {result:?}");
        assert_eq!(result.primal().unwrap().objectives(), [3, 3], "epoch {epoch}: {result:?}");
    }
    for (_, literals) in session.nogoods(None).unwrap() {
        assert!(literals.iter().all(|literal| literal.variable < 2));
    }
}

#[test]
fn session_controls_and_assignments_never_cross_the_user_variable_boundary() {
    let (model, _, _) = lexicographic_model();
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let auxiliary_index = 2;
    let requests = [
        SolveRequest {
            mode: SolveMode::Exact,
            assumptions: vec![SemanticAssumption { variable: auxiliary_index, operation: SemanticAssumptionOp::Eq, value: 0 }],
            ..SolveRequest::default()
        },
        SolveRequest { mode: SolveMode::Exact, hints: vec![(auxiliary_index, 0)], ..SolveRequest::default() },
        SolveRequest { mode: SolveMode::Exact, branch_order: vec![auxiliary_index], ..SolveRequest::default() },
        SolveRequest { mode: SolveMode::Exact, primary_branch_scope: Some(vec![auxiliary_index]), ..SolveRequest::default() },
    ];
    for request in requests {
        let error = solve(&mut session, &request).unwrap_err();
        assert!(matches!(error, SolveError::InvalidRequest(ref message) if message.contains("2 user integer variables")), "{error}");
    }

    let result = solve(&mut session, &SolveRequest { mode: SolveMode::Exact, threads: 2, seed: 9, ..SolveRequest::default() }).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().assignment().integers.len(), 2);
    assert_eq!(metadata(&result, "clause_session_first_worker"), "0");
    for (_, literals) in session.nogoods(Some(16)).unwrap() {
        assert!(literals.iter().all(|literal| literal.variable < 2));
    }
}

#[test]
fn single_worker_lexicographic_prefix_keeps_the_original_atom_layout() {
    let mut model = Model::new();
    let variables = (0..4).map(|_| model.int_range(0, 3)).collect::<Vec<_>>();
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: variables.clone(), except: Vec::new() }));
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(variables[0]) });
    model.add_objective(Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(
            variables
                .iter()
                .enumerate()
                .map(|(index, &variable)| {
                    IntExpr::Mul(vec![IntExpr::Constant(i64::try_from(index + 1).unwrap()), IntExpr::Variable(variable)])
                })
                .collect(),
        ),
    });
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    for _ in 0..3 {
        let result = solve(&mut session, &SolveRequest { mode: SolveMode::Exact, threads: 1, seed: 5, ..SolveRequest::default() }).unwrap();
        assert_eq!(result.status(), SolveStatus::Optimal);
        assert_eq!(result.primal().unwrap().objectives(), [0, 16]);
    }
}

#[test]
fn clear_nogoods_resets_worker_ids_after_mixed_portfolio_widths() {
    let (model, _, _) = lexicographic_model();
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();

    let one = solve(&mut session, &SolveRequest { mode: SolveMode::Exact, threads: 1, ..SolveRequest::default() }).unwrap();
    assert_eq!(metadata(&one, "clause_session_first_worker"), "0");
    assert_eq!(metadata(&one, "clause_session_next_worker"), "2");

    let four = solve(&mut session, &SolveRequest { mode: SolveMode::Exact, threads: 4, ..SolveRequest::default() }).unwrap();
    assert_eq!(metadata(&four, "clause_session_first_worker"), "2");
    assert_eq!(metadata(&four, "clause_session_next_worker"), "10");

    session.clear_nogoods();
    assert_eq!(session.learned_nogoods(), 0);
    let two = solve(&mut session, &SolveRequest { mode: SolveMode::Exact, threads: 2, ..SolveRequest::default() }).unwrap();
    assert_eq!(two.status(), SolveStatus::Optimal);
    assert_eq!(metadata(&two, "clause_session_first_worker"), "0");
    assert_eq!(metadata(&two, "clause_session_next_worker"), "4");
}

#[test]
fn portfolio_assumptions_match_a_small_exhaustive_oracle() {
    let mut model = Model::new();
    let x = model.int_range(0, 2);
    let y = model.int_range(0, 2);
    model.add_constraint(Constraint::Linear { terms: vec![(1, x), (1, y)], relation: Relation::Eq, rhs: 2 });
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();

    for value in -1..=3 {
        let request = SolveRequest {
            mode: SolveMode::Exact,
            threads: 3,
            assumptions: vec![SemanticAssumption { variable: x.0, operation: SemanticAssumptionOp::Eq, value }],
            ..SolveRequest::default()
        };
        let result = solve(&mut session, &request).unwrap();
        let expected = (0..=2).any(|candidate_y| value + candidate_y == 2);
        assert_eq!(result.status() == SolveStatus::Satisfiable, expected, "x={value}: {result:?}");
        if expected {
            let assignment = &result.primal().unwrap().assignment().integers;
            assert_eq!(assignment[x.0], Some(i64::from(value)));
            assert_eq!(assignment[x.0].unwrap() + assignment[y.0].unwrap(), 2);
        } else {
            assert_eq!(result.status(), SolveStatus::Unsatisfiable);
        }
    }
}

#[test]
fn persistent_worker_ids_do_not_perturb_seeded_trajectories() {
    let mut model = Model::new();
    for _ in 0..8 {
        model.int_range(0, 20);
    }
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let request = SolveRequest { mode: SolveMode::Exact, seed: 91, threads: 1, ..SolveRequest::default() };

    let first = solve(&mut session, &request).unwrap();
    let second = solve(&mut session, &request).unwrap();
    assert_eq!(first.status(), SolveStatus::Satisfiable);
    assert_eq!(second.status(), SolveStatus::Satisfiable);
    assert_eq!(first.primal().unwrap().assignment(), second.primal().unwrap().assignment());
    assert_eq!(metadata(&first, "clause_session_first_worker"), "0");
    assert_eq!(metadata(&second, "clause_session_first_worker"), "1");
}

#[test]
fn prearmed_preparation_stop_does_not_consume_worker_ids() {
    let mut model = Model::new();
    model.int_range(0, 10);
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let stopped = session
        .solve_with_external_stop(
            &SolveRequest { mode: SolveMode::Exact, threads: 4, ..SolveRequest::default() },
            &AtomicBool::new(true),
            &mut IgnoreEvents,
        )
        .unwrap();
    assert_eq!(stopped.status(), SolveStatus::Unknown);

    let next = solve(&mut session, &SolveRequest { mode: SolveMode::Exact, threads: 4, ..SolveRequest::default() }).unwrap();
    assert_eq!(next.status(), SolveStatus::Satisfiable);
    assert_eq!(metadata(&next, "clause_session_first_worker"), "0");
    assert_eq!(metadata(&next, "clause_session_next_worker"), "4");
}

#[test]
fn portfolio_conflict_quota_is_global_and_exact() {
    let pigeons = 5;
    let holes = 4;
    let mut model = Model::new();
    let vars = (0..pigeons * holes).map(|_| model.bool_var()).collect::<Vec<_>>();
    for pigeon in 0..pigeons {
        model.add_constraint(Constraint::Linear {
            terms: (0..holes).map(|hole| (1, vars[pigeon * holes + hole])).collect(),
            relation: Relation::Eq,
            rhs: 1,
        });
    }
    for hole in 0..holes {
        model.add_constraint(Constraint::Linear {
            terms: (0..pigeons).map(|pigeon| (1, vars[pigeon * holes + hole])).collect(),
            relation: Relation::Le,
            rhs: 1,
        });
    }
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let result = solve(
        &mut session,
        &SolveRequest {
            mode: SolveMode::Exact,
            threads: 4,
            limits: SolveLimits { conflicts: Some(7), ..SolveLimits::default() },
            ..SolveRequest::default()
        },
    )
    .unwrap();
    assert!(result.aggregate_search_stats().failures <= 7, "portfolio exceeded its global quota: {result:?}");
    if result.status() == SolveStatus::Unknown {
        assert!(result.message().is_some_and(|message| message.contains("ConflictLimit")));
    }
}

#[test]
fn memory_preflight_accounts_for_every_exact_worker() {
    let memory_model = || {
        let mut model = Model::new();
        for _ in 0..32 {
            model.int_range(0, 100);
        }
        model
    };
    let mut single = SemanticSolveSession::new(ModelPackage::new(memory_model())).unwrap();
    let single_error = solve(
        &mut single,
        &SolveRequest {
            mode: SolveMode::Exact,
            threads: 1,
            limits: SolveLimits { memory_bytes: Some(1), ..SolveLimits::default() },
            ..SolveRequest::default()
        },
    )
    .unwrap_err();
    let single_message = single_error.to_string();
    let single_required = single_message
        .split_once("requires ")
        .and_then(|(_, suffix)| suffix.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("missing memory estimate in {single_message}"));
    assert_eq!(single_required % 2, 0);
    let estimate = single_required / 2;

    let mut portfolio = SemanticSolveSession::new(ModelPackage::new(memory_model())).unwrap();
    let error = solve(
        &mut portfolio,
        &SolveRequest {
            mode: SolveMode::Exact,
            threads: 4,
            limits: SolveLimits { memory_bytes: Some(estimate.saturating_mul(4)), ..SolveLimits::default() },
            ..SolveRequest::default()
        },
    )
    .unwrap_err();
    assert!(
        matches!(error, SolveError::Compile(ref message)
            if message.contains(&format!("requires {} bytes", estimate.saturating_mul(5)))
                && message.contains("across 4 exact workers and the persistent template")
                && message.contains("memory limit")),
        "{error}"
    );
}

struct FailOnProgress;

impl EventSink for FailOnProgress {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        if matches!(event, SolveEvent::Progress { .. }) {
            return Err(SolveError::InvalidResult("session portfolio callback failed".to_string()));
        }
        Ok(EventControl::Continue)
    }
}

#[test]
fn callback_failure_cancels_every_portfolio_worker_and_is_not_masked() {
    let (model, _, _) = lexicographic_model();
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let error = session
        .solve_with_external_stop(
            &SolveRequest { mode: SolveMode::Exact, threads: 4, ..SolveRequest::default() },
            &AtomicBool::new(false),
            &mut FailOnProgress,
        )
        .unwrap_err();
    assert!(matches!(error, SolveError::InvalidResult(ref message) if message == "session portfolio callback failed"), "{error}");

    let recovered = solve(&mut session, &SolveRequest { mode: SolveMode::Exact, threads: 4, seed: 7, ..SolveRequest::default() }).unwrap();
    assert_eq!(recovered.status(), SolveStatus::Optimal);
    assert_eq!(metadata(&recovered, "clause_session_first_worker"), "4");
}

struct PanicOnProgress;

impl EventSink for PanicOnProgress {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        if matches!(event, SolveEvent::Progress { .. }) {
            panic!("intentional in-portfolio event sink panic");
        }
        Ok(EventControl::Continue)
    }
}

#[test]
fn multiworker_cop_handler_panic_cancels_workers_before_scoped_join() {
    run_ignored_subprocess_case("multiworker_cop_handler_panic_child");
}

#[test]
#[ignore = "subprocess helper"]
fn multiworker_cop_handler_panic_child() {
    let (model, _, _) = lexicographic_model();
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let unwind = catch_unwind(AssertUnwindSafe(|| {
        let _ = session.solve_with_external_stop(
            &SolveRequest { mode: SolveMode::Exact, threads: 4, ..SolveRequest::default() },
            &AtomicBool::new(false),
            &mut PanicOnProgress,
        );
    }));
    assert!(unwind.is_err());

    let recovered = solve(&mut session, &SolveRequest { mode: SolveMode::Exact, threads: 4, seed: 23, ..SolveRequest::default() }).unwrap();
    assert_eq!(recovered.status(), SolveStatus::Optimal);
    let first_worker = metadata(&recovered, "clause_session_first_worker").parse::<usize>().unwrap();
    assert!(first_worker >= 4, "panicking epoch reused worker ids: {recovered:?}");
}

#[test]
fn executor_worker_panic_cancels_sibling_before_scoped_join() {
    run_ignored_subprocess_case("executor_worker_panic_child");
}

#[test]
#[ignore = "subprocess helper"]
fn executor_worker_panic_child() {
    let unwind = catch_unwind(AssertUnwindSafe(|| {
        let stop = AtomicBool::new(false);
        let cancel = Arc::new(AtomicBool::new(false));
        let _ = execute_workers_silent(vec![0_u8, 1], &stop, cancel, 0, |context, worker| {
            if worker == 0 {
                panic!("intentional worker panic");
            }
            while !context.is_cancelled() {
                std::thread::yield_now();
            }
            worker
        });
    }));
    assert!(unwind.is_err());
}

struct PanicOnFinalEvidence;

impl EventSink for PanicOnFinalEvidence {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        if matches!(event, SolveEvent::Candidate(_) | SolveEvent::Proof(_)) {
            panic!("intentional event sink panic");
        }
        Ok(EventControl::Continue)
    }
}

#[test]
fn external_stop_monitors_release_during_event_sink_unwinding() {
    let mut model = Model::new();
    model.int_range(0, 1);
    let package = ModelPackage::new(model.clone());
    let external = AtomicBool::new(false);
    let ordinary = catch_unwind(AssertUnwindSafe(|| {
        let _ = solve_model_with_external_stop(&package, &SolveRequest::default(), &external, &mut PanicOnFinalEvidence);
    }));
    assert!(ordinary.is_err());

    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let session_unwind = catch_unwind(AssertUnwindSafe(|| {
        let _ = session.solve_with_external_stop(&SolveRequest::default(), &external, &mut PanicOnFinalEvidence);
    }));
    assert!(session_unwind.is_err());
}

#[test]
fn unsupported_cooperative_roles_are_rejected_explicitly() {
    let mut model = Model::new();
    let x = model.int_range(0, 1);
    model.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Variable(x) });
    let mut session = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let request =
        SolveRequest { mode: SolveMode::Exact, threads: 2, cp: CpControls { lns: 1, ..CpControls::default() }, ..SolveRequest::default() };
    let error = solve(&mut session, &request).unwrap_err();
    assert!(matches!(error, SolveError::InvalidRequest(ref message) if message.contains("CP role")), "{error}");
}

#[cfg(feature = "lp-relaxation")]
#[test]
fn session_portfolio_preserves_certified_node_relaxations() {
    let size = 6;
    let mut model = Model::new();
    let vertices = (0..size).map(|_| model.bool_var()).collect::<Vec<_>>();
    for left in 0..size {
        for right in (left + 1)..size {
            model.add_constraint(Constraint::Linear {
                terms: vec![(1, vertices[left]), (1, vertices[right])],
                relation: Relation::Ge,
                rhs: 1,
            });
        }
    }
    model.add_objective(Objective::IntExpr {
        minimize: true,
        expr: IntExpr::Add(vertices.iter().copied().map(IntExpr::Variable).collect()),
    });
    let mut session = SemanticSolveSession::new(ModelPackage::new(model.clone())).unwrap();
    let request = SolveRequest {
        mode: SolveMode::Exact,
        threads: 1,
        hints: (0..size).map(|variable| (variable, 1)).collect(),
        primary_branch_scope: Some((0..size).collect()),
        branch_order: (0..size).collect(),
        linear: LinearControls {
            backend: LinearBackendMode::Amthal,
            root_time: Duration::from_secs(1),
            node_time: Duration::from_millis(20),
            node_depth_interval: 1,
            phase_max_variables: 0,
            ..LinearControls::default()
        },
        ..SolveRequest::default()
    };
    let result = solve(&mut session, &request).unwrap();
    assert_eq!(result.status(), SolveStatus::Optimal);
    assert_eq!(result.primal().unwrap().objectives(), [5]);
    let stats = result.aggregate_search_stats();
    assert!(stats.lp_solves > 1, "session worker dropped the node relaxation: {result:?}");
    assert!(stats.lp_certified > 1, "session worker did not recertify its node relaxations: {result:?}");

    let mut portfolio = SemanticSolveSession::new(ModelPackage::new(model)).unwrap();
    let portfolio_result = solve(&mut portfolio, &SolveRequest { threads: 2, ..request }).unwrap();
    assert_eq!(portfolio_result.status(), SolveStatus::Optimal);
    let portfolio_stats = portfolio_result.aggregate_search_stats();
    assert_eq!(portfolio_stats.lp_model_status, LinearModelStatus::Ready);
    assert!(portfolio_stats.lp_rows > 0);
    assert!(portfolio_stats.lp_solves > 0);
}
