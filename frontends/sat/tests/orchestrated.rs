use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use qayd::model::{BoolLiteral, Constraint, Model, ModelPackage};
use qayd::orchestrator::{
    EngineKind, EventControl, EventSink, SatBackendMode, SatControls, SatPreprocess, SolveError, SolveEvent, SolveMode, SolveRequest,
    SolveResult, SolveStatus, VerificationLevel,
};
use qayd::sat::{assignment_satisfies, parse_dimacs, Cnf};

#[derive(Default)]
struct RecordingSink {
    events: Vec<SolveEvent>,
    stop_on_candidate: bool,
}

impl EventSink for RecordingSink {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        let stop = self.stop_on_candidate && matches!(event, SolveEvent::Candidate(_));
        self.events.push(event);
        Ok(if stop { EventControl::Stop } else { EventControl::Continue })
    }
}

fn exact_request() -> SolveRequest {
    SolveRequest { mode: SolveMode::Exact, ..SolveRequest::default() }
}

fn semantic_package(cnf: &Cnf) -> ModelPackage {
    let mut model = Model::new();
    let variables = (0..cnf.vars).map(|_| model.bool_var()).collect::<Vec<_>>();
    for clause in &cnf.clauses {
        model.add_constraint(Constraint::Clause(
            clause
                .iter()
                .map(|literal| BoolLiteral { variable: variables[literal.unsigned_abs() as usize - 1], positive: *literal > 0 })
                .collect(),
        ));
    }
    ModelPackage::new(model)
}

fn solve_request(
    cnf: &Cnf,
    backend: SatBackendMode,
    preprocess: SatPreprocess,
    request: &SolveRequest,
    stop: Arc<AtomicBool>,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    let mut request = request.clone();
    request.sat = SatControls { backend: Some(backend), preprocess, proof_path: request.sat.proof_path.clone() };
    qayd::solve_with_stop(&semantic_package(cnf), &request, stop, sink)
}

#[test]
fn cli_builds_a_semantic_package_and_calls_the_canonical_entry_point() {
    let source = include_str!("../src/main.rs");
    assert!(source.contains("semantic_package"));
    assert!(source.contains("ModelPackage"));
    assert!(source.contains("qayd::solve_with_stop"));
    assert!(source.contains("SolveRequest"));
    assert!(source.contains("SolveResult"));
    for forbidden in [
        "SolveBudget",
        "qayd::engines",
        "solve_cnf_request",
        "solve_cnf_with_backend_seeded_options",
        "solve_cnf_native_seeded_with_proof_options",
        "thread::spawn",
        "std::thread",
    ] {
        assert!(!source.contains(forbidden), "SAT CLI bypasses its orchestrated adapter through {forbidden}");
    }
}

#[test]
fn frontend_library_exports_parser_data_without_importing_an_engine() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub use qayd::sat::*"));
    for forbidden in ["qayd::engines", "SolveBudget", "execute_specialized", "solve_cnf_request"] {
        assert!(!source.contains(forbidden), "SAT parser crate owns forbidden solve logic through {forbidden}");
    }
}

fn candidate_assignment(result: &qayd::orchestrator::SolveResult) -> Vec<bool> {
    result
        .primal()
        .expect("SAT result has a candidate")
        .assignment()
        .integers
        .iter()
        .map(|value| match value {
            Some(0) => false,
            Some(1) => true,
            other => panic!("unexpected Boolean assignment value: {other:?}"),
        })
        .collect()
}

#[test]
fn native_sat_is_verified_and_emits_candidate_then_finished() {
    let cnf = parse_dimacs("p cnf 3 3\n1 2 0\n-1 3 0\n-3 2 0\n").unwrap();
    let mut sink = RecordingSink::default();

    let result =
        solve_request(&cnf, SatBackendMode::Native, SatPreprocess::Basic, &exact_request(), Arc::new(AtomicBool::new(false)), &mut sink)
            .unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    let candidate = result.primal().unwrap();
    assert_eq!(candidate.source(), EngineKind::IntegerExact);
    assert_eq!(candidate.verification(), VerificationLevel::Final);
    assert!(assignment_satisfies(&cnf, &candidate_assignment(&result)));
    assert!(result.proof().is_none());
    assert!(matches!(sink.events.as_slice(), [SolveEvent::Candidate(_), SolveEvent::Finished(_)]));
}

#[test]
fn native_full_preprocessing_reconstructs_a_verified_candidate() {
    let cnf = parse_dimacs("p cnf 3 3\n1 2 0\n-1 3 0\n-2 -3 0\n").unwrap();
    let mut sink = RecordingSink::default();

    let result =
        solve_request(&cnf, SatBackendMode::Native, SatPreprocess::Full, &exact_request(), Arc::new(AtomicBool::new(false)), &mut sink)
            .unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(assignment_satisfies(&cnf, &candidate_assignment(&result)));
    let eliminated = result.reports()[0]
        .metadata
        .iter()
        .find_map(|(key, value)| if key == "pre_bve_vars" { Some(value.parse::<u64>().unwrap()) } else { None })
        .unwrap();
    assert!(eliminated > 0);
}

#[test]
fn native_unsat_gets_complete_search_proof_and_ordered_events() {
    let cnf = parse_dimacs("p cnf 1 2\n1 0\n-1 0\n").unwrap();
    let mut sink = RecordingSink::default();

    let result =
        solve_request(&cnf, SatBackendMode::Native, SatPreprocess::Basic, &exact_request(), Arc::new(AtomicBool::new(false)), &mut sink)
            .unwrap();

    assert_eq!(result.status(), SolveStatus::Unsatisfiable);
    assert!(result.primal().is_none());
    let proof = result.proof().expect("UNSAT result has a proof");
    assert_eq!(proof.engine(), Some(EngineKind::IntegerExact));
    assert_eq!(proof.objective_tiers(), Some(0));
    assert!(matches!(sink.events.as_slice(), [SolveEvent::Proof(_), SolveEvent::Finished(_)]));
}

#[test]
fn zero_budget_returns_unknown_before_any_engine_event() {
    let cnf = parse_dimacs("p cnf 1 1\n1 0\n").unwrap();
    let mut request = exact_request();
    request.limits.time = Some(Duration::ZERO);
    let stop = Arc::new(AtomicBool::new(false));
    let mut sink = RecordingSink::default();

    let result = solve_request(&cnf, SatBackendMode::Native, SatPreprocess::Basic, &request, Arc::clone(&stop), &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(result.primal().is_none());
    assert!(result.proof().is_none());
    assert!(stop.load(Ordering::Acquire));
    assert!(result.message().is_some_and(|message| message.contains("Deadline")));
    assert!(sink.events.is_empty());
}

#[test]
fn external_stop_reaches_the_orchestrator_before_any_engine_event() {
    let cnf = parse_dimacs("p cnf 1 1\n1 0\n").unwrap();
    let stop = Arc::new(AtomicBool::new(true));
    let mut sink = RecordingSink::default();

    let result = solve_request(&cnf, SatBackendMode::Native, SatPreprocess::Basic, &exact_request(), Arc::clone(&stop), &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Unknown);
    assert!(stop.load(Ordering::Acquire));
    assert!(result.message().is_some_and(|message| message.contains("ExternalCancellation")));
    assert!(sink.events.is_empty());
}

#[test]
fn stopping_on_candidate_cancels_budget_and_suppresses_finished_event() {
    let cnf = parse_dimacs("p cnf 1 1\n1 0\n").unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let mut sink = RecordingSink { events: Vec::new(), stop_on_candidate: true };

    let result = solve_request(&cnf, SatBackendMode::Native, SatPreprocess::Off, &exact_request(), Arc::clone(&stop), &mut sink).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert!(stop.load(Ordering::Acquire));
    assert!(matches!(sink.events.as_slice(), [SolveEvent::Candidate(_)]));
}

#[test]
fn linear_backend_is_finalized_by_the_same_contract() {
    let cnf = parse_dimacs("p cnf 2 2\n1 2 0\n-1 2 0\n").unwrap();
    let mut sink = RecordingSink::default();

    let result =
        solve_request(&cnf, SatBackendMode::Linear, SatPreprocess::Full, &exact_request(), Arc::new(AtomicBool::new(false)), &mut sink)
            .unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.reports()[0].engine, Some(EngineKind::Linear));
    assert!(assignment_satisfies(&cnf, &candidate_assignment(&result)));
}

#[test]
fn orchestrated_drat_keeps_preprocessing_proof_output() {
    let cnf = parse_dimacs("p cnf 1 2\n1 0\n-1 0\n").unwrap();
    let mut sink = RecordingSink::default();
    let path = std::env::temp_dir().join(format!("qayd-sat-orchestrated-{}.drat", std::process::id()));
    let mut request = exact_request();
    request.sat.proof_path = Some(path.to_string_lossy().into_owned());

    let result =
        solve_request(&cnf, SatBackendMode::Native, SatPreprocess::Basic, &request, Arc::new(AtomicBool::new(false)), &mut sink).unwrap();
    let proof = fs::read_to_string(&path).unwrap();
    fs::remove_file(path).unwrap();

    assert_eq!(result.status(), SolveStatus::Unsatisfiable);
    assert!(proof.lines().any(|line| line == "0"));
    assert!(result.reports()[0].metadata.contains(&("proof_format".to_string(), "drat".to_string())));
}

#[test]
fn unsupported_local_search_control_is_rejected() {
    let cnf = parse_dimacs("p cnf 1 1\n1 0\n").unwrap();
    let request = SolveRequest { mode: SolveMode::LocalSearch, ..SolveRequest::default() };
    let mut sink = RecordingSink::default();

    let error =
        solve_request(&cnf, SatBackendMode::Native, SatPreprocess::Off, &request, Arc::new(AtomicBool::new(false)), &mut sink).unwrap_err();

    assert!(matches!(error, SolveError::InvalidRequest(_)));
    assert!(sink.events.is_empty());
}
