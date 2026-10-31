//! Public lifecycle and independent oracle gates for the opt-in I7 wrapper.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use qayd::certificate::{
    solve_with_certificate, CertificateClaim, CertificateError, CertificateOptions, CertificateRun, CheckedCpCertificate,
};
use qayd::model::{Constraint, IntExpr, IntGlobalConstraint, Model, ModelPackage, Objective, Relation};
use qayd::orchestrator::{
    solve_model_silent, ProofKind, ProofRequest, SatBackendMode, SearchPhase, SearchPolicy, SemanticAssumption, SemanticAssumptionOp,
    SolveLimits, SolveMode, SolveRequest, SolveStatus, ValueSelector, VariableSelector,
};

#[path = "../../common/certificate_models.rs"]
mod certificate_models;
use certificate_models::{all_cases, Case};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let index = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!("qayd-i7-{}-{nonce}-{index}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        Self(directory)
    }

    fn options(&self, name: &str) -> CertificateOptions {
        CertificateOptions::new(self.0.join(name))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn request() -> SolveRequest {
    SolveRequest { mode: SolveMode::Exact, threads: 1, seed: 7, ..SolveRequest::default() }
}

fn run(package: &ModelPackage, request: &SolveRequest, options: &CertificateOptions) -> CertificateRun {
    solve_with_certificate(package, request, options, Arc::new(AtomicBool::new(false))).unwrap()
}

fn expected_claim(status: SolveStatus) -> CertificateClaim {
    match status {
        SolveStatus::Satisfiable => CertificateClaim::Satisfiable,
        SolveStatus::Unsatisfiable => CertificateClaim::Unsatisfiable,
        SolveStatus::Optimal => CertificateClaim::Optimal,
        status => panic!("no complete certificate claim for {status:?}"),
    }
}

fn assert_owned_files(checked: &CheckedCpCertificate, directory: &Path) {
    let directory = directory.canonicalize().unwrap();
    for path in [checked.model_path(), checked.proof_path()] {
        assert!(path.is_file());
        assert!(path.canonicalize().unwrap().starts_with(&directory));
    }
    assert_ne!(checked.model_path(), checked.proof_path());
    for hash in [checked.model_sha256(), checked.proof_sha256(), checked.checker_sha256()] {
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    }
    let proof = fs::read_to_string(checked.proof_path()).unwrap();
    assert_eq!(proof.lines().next().unwrap(), format!("qayd-cp-proof 1 {} {}", checked.model_sha256(), checked.claim().token()));
    assert_eq!(checked.stats().model_bytes, fs::metadata(checked.model_path()).unwrap().len());
    assert_eq!(checked.stats().proof_bytes, fs::metadata(checked.proof_path()).unwrap().len());
}

fn failure(outcome: Result<CertificateRun, CertificateError>) -> CertificateError {
    match outcome {
        Err(error) => error,
        Ok(run) => {
            run.result.validate_contract().unwrap();
            match run.certificate {
                Err(error) => error,
                Ok(_) => panic!("an interrupted or failed operation produced a checked certificate"),
            }
        }
    }
}

#[test]
fn frozen_pilot_roundtrips_match_exhaustive_oracles_without_changing_native_proofs() {
    let scratch = Scratch::new();
    let cases = all_cases();
    assert_eq!(cases.len(), 15);
    for case in cases {
        let package = case.package();
        let request = request();
        let baseline = solve_model_silent(&package, &request).unwrap();
        case.validate(&baseline).unwrap();
        let options = scratch.options(&format!("{}-{}", case.family, case.size));
        let run = run(&package, &request, &options);
        case.validate(&run.result).unwrap();
        assert_eq!(run.result.proof().map(|proof| proof.kind()), baseline.proof().map(|proof| proof.kind()));
        assert!(!matches!(run.result.proof().map(|proof| proof.kind()), Some(ProofKind::VerifiedArtifact { .. })));
        let checked = run.certificate.unwrap_or_else(|error| panic!("{case:?}: {error}"));
        assert_eq!(checked.claim(), expected_claim(case.expected_status()));
        assert!(checked.assumptions().is_empty());
        assert_owned_files(&checked, &options.directory);
    }
}

#[test]
fn certificate_handles_own_exact_assumptions_and_do_not_leak_between_runs() {
    let scratch = Scratch::new();
    let case = Case { family: "table_path", size: 4 };
    let package = case.package();
    let mut hashes = Vec::new();
    let mut retained = Vec::new();
    for value in [0, 1] {
        let mut request = request();
        request.assumptions = vec![SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value }];
        let options = scratch.options(&format!("scoped-{value}"));
        let run = run(&package, &request, &options);
        case.validate(&run.result).unwrap();
        assert_eq!(run.result.primal().unwrap().assignment().integers[0], Some(i64::from(value)));
        let checked = run.certificate.unwrap();
        assert_eq!(checked.assumptions(), request.assumptions);
        assert_owned_files(&checked, &options.directory);
        hashes.push(checked.model_sha256().to_owned());
        retained.push(checked);
    }
    assert_ne!(hashes[0], hashes[1]);
    let mut contradictory = request();
    contradictory.assumptions = vec![
        SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value: 0 },
        SemanticAssumption { variable: 0, operation: SemanticAssumptionOp::Eq, value: 1 },
    ];
    let contradiction = run(&package, &contradictory, &scratch.options("contradictory"));
    assert_eq!(contradiction.result.status(), SolveStatus::Unsatisfiable);
    let contradiction = contradiction.certificate.unwrap();
    assert_eq!(contradiction.claim(), CertificateClaim::Unsatisfiable);
    assert_eq!(contradiction.assumptions(), contradictory.assumptions);
    drop(contradictory);
    let fresh = run(&package, &request(), &scratch.options("fresh"));
    case.validate(&fresh.result).unwrap();
    assert!(fresh.certificate.unwrap().assumptions().is_empty());
    for (index, checked) in retained.iter().enumerate() {
        assert_eq!(checked.model_sha256(), hashes[index]);
        assert!(checked.model_path().is_file());
        assert_eq!(checked.assumptions()[0].value, index as i32);
    }
}

#[test]
fn all_linear_relations_normalize_repeated_ids_and_zero_coefficients() {
    let scratch = Scratch::new();
    for (index, relation) in [Relation::Eq, Relation::Ne, Relation::Le, Relation::Lt, Relation::Ge, Relation::Gt].into_iter().enumerate() {
        let mut model = Model::new();
        let x = model.int_set(vec![-2, 0, 3]);
        model.add_constraint(Constraint::Linear { terms: vec![(2, x), (-1, x), (0, x)], relation, rhs: 0 });
        let run = run(&ModelPackage::new(model), &request(), &scratch.options(&format!("relation-{index}")));
        assert_eq!(run.result.status(), SolveStatus::Satisfiable);
        let value = run.result.primal().unwrap().assignment().integers[0].unwrap();
        assert!([-2, 0, 3].contains(&value));
        assert!(match relation {
            Relation::Eq => value == 0,
            Relation::Ne => value != 0,
            Relation::Le => value <= 0,
            Relation::Lt => value < 0,
            Relation::Ge => value >= 0,
            Relation::Gt => value > 0,
        });
        assert_eq!(run.certificate.unwrap().claim(), CertificateClaim::Satisfiable);
    }
}

#[test]
fn wildcard_tables_keep_sparse_endpoints_and_repeated_columns_semantics() {
    let scratch = Scratch::new();
    let mut model = Model::new();
    let x = model.int_set(vec![i32::MIN, -2, i32::MAX]);
    let y = model.int_range(0, 1);
    model.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
        variables: vec![x, x, y],
        tuples: vec![vec![i32::MIN, i32::MIN, 1], vec![i32::MAX, i32::MAX, 0], vec![-2, i32::MAX, 0]].into(),
        positive: true,
    }));
    let package = ModelPackage::new(model);
    for (index, value) in [i32::MIN, -2, i32::MAX].into_iter().enumerate() {
        for presence in [0, 1] {
            let mut request = request();
            request.assumptions = vec![
                SemanticAssumption { variable: x.0, operation: SemanticAssumptionOp::Eq, value },
                SemanticAssumption { variable: y.0, operation: SemanticAssumptionOp::Eq, value: presence },
            ];
            let run = run(&package, &request, &scratch.options(&format!("sparse-{index}-{presence}")));
            let feasible = presence == 1 || value == i32::MAX;
            assert_eq!(run.result.status(), if feasible { SolveStatus::Satisfiable } else { SolveStatus::Unsatisfiable });
            let checked = run.certificate.unwrap();
            assert_eq!(checked.claim(), expected_claim(run.result.status()));
            let exported = fs::read_to_string(checked.model_path()).unwrap();
            assert!(exported.contains("row * * 1"));
            assert!(exported.contains("-2147483648"));
            assert!(exported.contains("2147483647"));
        }
    }
}

#[test]
fn endpoint_constant_objectives_and_native_proof_requirements_remain_distinct() {
    let scratch = Scratch::new();
    for (index, value) in [i64::MIN, i64::MAX].into_iter().enumerate() {
        for minimizing in [false, true] {
            let mut model = Model::new();
            let x = model.int_range(0, 0);
            model.add_constraint(Constraint::Linear { terms: vec![(1, x)], relation: Relation::Eq, rhs: 0 });
            model.add_objective(Objective::IntExpr { minimize: minimizing, expr: IntExpr::Constant(value) });
            let package = ModelPackage::new(model);
            let mut request = request();
            request.proof = ProofRequest::Require;
            let baseline = solve_model_silent(&package, &request).unwrap();
            let run = run(&package, &request, &scratch.options(&format!("constant-{index}-{minimizing}")));
            assert_eq!(run.result.status(), SolveStatus::Optimal);
            assert_eq!(run.result.primal().unwrap().objectives(), &[value]);
            assert_eq!(run.result.proof(), baseline.proof());
            assert!(!matches!(run.result.proof().unwrap().kind(), ProofKind::VerifiedArtifact { .. }));
            let checked = run.certificate.unwrap();
            assert_eq!(checked.claim(), CertificateClaim::Optimal);
            let proof = fs::read_to_string(checked.proof_path()).unwrap();
            assert!(proof.contains(&format!("bound {} {value}", if minimizing { "min" } else { "max" })));
        }
    }
}

#[test]
fn unsupported_models_fail_before_creating_artifact_directories() {
    let scratch = Scratch::new();
    let mut models = Vec::new();
    let mut expression = Model::new();
    let x = expression.int_range(0, 1);
    expression.add_constraint(Constraint::Intension(IntExpr::Variable(x)));
    models.push(("intension", expression));
    let mut negative = Model::new();
    let x = negative.int_range(0, 1);
    negative.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::Table {
        variables: vec![x],
        tuples: vec![vec![0]].into(),
        positive: false,
    }));
    models.push(("negative-table", negative));
    let mut global = Model::new();
    let x = global.int_range(0, 1);
    let y = global.int_range(0, 1);
    global.add_constraint(Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: vec![x, y], except: vec![] }));
    models.push(("all-different", global));
    let mut interval = Model::new();
    interval.interval(0, 1, 1);
    models.push(("interval", interval));
    let mut objective = Model::new();
    let x = objective.int_range(0, 1);
    objective.add_objective(Objective::IntExpr { minimize: true, expr: IntExpr::Add(vec![IntExpr::Variable(x), IntExpr::Constant(1)]) });
    models.push(("expression-objective", objective));
    let mut tiers = Model::new();
    let x = tiers.int_range(0, 1);
    for minimizing in [true, false] {
        tiers.add_objective(Objective::IntExpr { minimize: minimizing, expr: IntExpr::Variable(x) });
    }
    models.push(("multiple-objectives", tiers));
    let mut unsafe_activity = Model::new();
    let x = unsafe_activity.int_range(1, 2);
    unsafe_activity.add_constraint(Constraint::Linear { terms: vec![(i64::MAX, x)], relation: Relation::Ge, rhs: 0 });
    models.push(("unsafe-native-activity", unsafe_activity));
    for (name, model) in models {
        let options = scratch.options(name);
        let outcome = solve_with_certificate(&ModelPackage::new(model), &request(), &options, Arc::new(AtomicBool::new(false)));
        assert!(matches!(outcome, Err(CertificateError::Unsupported(_))), "{name} must fail supported-slice preflight");
        assert!(!options.directory.exists(), "{name} left an artifact directory");
    }
}

#[test]
fn stop_zero_deadline_and_generation_caps_never_produce_checked_handles() {
    let scratch = Scratch::new();
    let package = Case { family: "parity", size: 4 }.package();
    let stopped = solve_with_certificate(&package, &request(), &scratch.options("stopped"), Arc::new(AtomicBool::new(true)));
    assert!(matches!(failure(stopped), CertificateError::Interrupted));
    let mut zero = request();
    zero.limits = SolveLimits { time: Some(Duration::ZERO), ..SolveLimits::default() };
    let interrupted = solve_with_certificate(&package, &zero, &scratch.options("zero-time"), Arc::new(AtomicBool::new(false)));
    assert!(matches!(failure(interrupted), CertificateError::Interrupted | CertificateError::Incomplete(_) | CertificateError::Limit(_)));
    zero.limits.time = Some(Duration::from_nanos(1));
    let interrupted = solve_with_certificate(&package, &zero, &scratch.options("tiny-time"), Arc::new(AtomicBool::new(false)));
    assert!(matches!(failure(interrupted), CertificateError::Interrupted | CertificateError::Incomplete(_) | CertificateError::Limit(_)));
    for name in ["nodes", "steps", "work", "model-bytes", "proof-bytes", "depth", "variables", "constraints", "input-entries"] {
        let mut options = scratch.options(name);
        match name {
            "nodes" => options.limits.max_nodes = 0,
            "steps" => options.limits.max_steps = 0,
            "work" => options.limits.max_work = 0,
            "model-bytes" => options.limits.max_model_bytes = 1,
            "proof-bytes" => options.limits.max_proof_bytes = 1,
            "depth" => options.limits.max_depth = 0,
            "variables" => options.limits.max_variables = 0,
            "constraints" => options.limits.max_constraints = 0,
            "input-entries" => options.limits.max_input_entries = 0,
            _ => unreachable!(),
        }
        let outcome = solve_with_certificate(&package, &request(), &options, Arc::new(AtomicBool::new(false)));
        assert!(matches!(failure(outcome), CertificateError::Limit(_)), "cap {name}");
    }
}

#[test]
fn unsupported_request_combinations_are_explicit_and_leave_no_artifacts() {
    let scratch = Scratch::new();
    let package = Case { family: "table_path", size: 4 }.package();
    for name in [
        "parallel",
        "memory",
        "sat-proof",
        "local-search",
        "iterations",
        "list-hint",
        "depth-ceiling",
        "variable-ceiling",
        "constraint-ceiling",
        "entry-ceiling",
    ] {
        let mut request = request();
        let mut options = scratch.options(name);
        match name {
            "parallel" => request.threads = 2,
            "memory" => request.limits.memory_bytes = Some(1_000_000_000),
            "local-search" => request.mode = SolveMode::LocalSearch,
            "iterations" => request.limits.iterations = Some(1),
            "list-hint" => request.list_hint = Some(vec![]),
            "depth-ceiling" => options.limits.max_depth = 129,
            "variable-ceiling" => options.limits.max_variables = 65,
            "constraint-ceiling" => options.limits.max_constraints = 513,
            "entry-ceiling" => options.limits.max_input_entries = 100_001,
            "sat-proof" => {
                request.sat.backend = Some(SatBackendMode::Native);
                request.sat.proof_path = Some(scratch.0.join("not-a-sat-proof").to_string_lossy().into_owned());
            }
            _ => unreachable!(),
        }
        let outcome = solve_with_certificate(&package, &request, &options, Arc::new(AtomicBool::new(false)));
        assert!(matches!(outcome, Err(CertificateError::Unsupported(_))), "{name}");
        assert!(!options.directory.exists());
    }
    assert!(!scratch.0.join("not-a-sat-proof").exists());
}

#[test]
fn request_metadata_is_bounded_before_canonical_validation_or_artifact_creation() {
    let scratch = Scratch::new();
    let package = Case { family: "table_path", size: 4 }.package();
    // Duplicate indices also make several requests canonically invalid. The
    // certificate Limit outcome confirms that its small size gate runs first.
    for name in ["primary-scope", "hints", "branch-order", "phase-count", "phase-scope"] {
        let mut request = request();
        match name {
            "primary-scope" => request.primary_branch_scope = Some(vec![0; 65]),
            "hints" => request.hints = vec![(0, 0); 65],
            "branch-order" => request.branch_order = vec![0; 65],
            "phase-count" => {
                request.search_policy =
                    SearchPolicy::new(vec![SearchPhase::new(vec![0], VariableSelector::InputOrder, ValueSelector::Min); 65]);
            }
            "phase-scope" => {
                request.search_policy =
                    SearchPolicy::new(vec![SearchPhase::new(vec![0; 65], VariableSelector::InputOrder, ValueSelector::Min)]);
            }
            _ => unreachable!(),
        }
        let mut options = scratch.options(name);
        // Raising the generic variable cap cannot lift the wrapper's fixed
        // bound on non-interruptible canonical validation work.
        options.limits.max_variables = 1_000;
        let outcome = solve_with_certificate(&package, &request, &options, Arc::new(AtomicBool::new(false)));
        assert!(matches!(outcome, Err(CertificateError::Limit(ref reason)) if reason.starts_with("request search")), "{name}");
        assert!(!options.directory.exists());
    }
}

#[test]
fn native_incomplete_results_are_preserved_without_external_completion_authority() {
    let scratch = Scratch::new();
    let package = Case { family: "table_path", size: 4 }.package();
    let mut request = request();
    request.limits.conflicts = Some(0);
    let baseline = solve_model_silent(&package, &request).unwrap();
    assert_eq!(baseline.status(), SolveStatus::Unknown);
    let options = scratch.options("incomplete");
    let run = run(&package, &request, &options);
    run.result.validate_contract().unwrap();
    assert_eq!(run.result.status(), baseline.status());
    assert_eq!(run.result.proof(), baseline.proof());
    assert_eq!(run.result.message(), baseline.message());
    assert!(matches!(run.certificate, Err(CertificateError::Incomplete(_))));
    assert!(!options.directory.exists());
}

#[test]
fn existing_paths_and_io_failures_do_not_overwrite_user_files() {
    let scratch = Scratch::new();
    let package = Case { family: "table_path", size: 4 }.package();
    let options = scratch.options("existing-directory");
    fs::create_dir(&options.directory).unwrap();
    let sentinel = options.directory.join("sentinel.txt");
    fs::write(&sentinel, b"preserve this content\n").unwrap();
    let run = solve_with_certificate(&package, &request(), &options, Arc::new(AtomicBool::new(false))).unwrap();
    assert_eq!(run.result.status(), SolveStatus::Satisfiable);
    run.result.validate_contract().unwrap();
    assert!(matches!(run.certificate, Err(CertificateError::Io(_))));
    assert_eq!(fs::read(&sentinel).unwrap(), b"preserve this content\n");
    assert_eq!(fs::read_dir(&options.directory).unwrap().count(), 1);
    let file = scratch.0.join("ordinary-file");
    fs::write(&file, b"not a directory").unwrap();
    for directory in [file.clone(), file.join("child")] {
        let outcome = solve_with_certificate(&package, &request(), &CertificateOptions::new(directory), Arc::new(AtomicBool::new(false)));
        assert!(matches!(failure(outcome), CertificateError::Io(_)));
    }
    assert_eq!(fs::read(file).unwrap(), b"not a directory");
}
