use std::fs;

use qayd::model::{BoolLiteral, Constraint, Model, ModelPackage};
use qayd::orchestrator::{
    solve_model_silent, EngineKind, SatBackendMode, SatControls, SatPreprocess, SolveLimits, SolveMode, SolveRequest, SolveStatus,
};

fn package(clauses: &[&[i32]], variable_count: usize) -> ModelPackage {
    let mut model = Model::new();
    let variables = (0..variable_count).map(|_| model.bool_var()).collect::<Vec<_>>();
    for clause in clauses {
        model.add_constraint(Constraint::Clause(
            clause
                .iter()
                .map(|literal| BoolLiteral { variable: variables[literal.unsigned_abs() as usize - 1], positive: *literal > 0 })
                .collect(),
        ));
    }
    ModelPackage::new(model)
}

#[test]
fn sat_memory_preflight_rejects_before_physical_clause_allocation() {
    let package = package(&[&[1, 2], &[-1, 2]], 2);
    let mut request = request(SatBackendMode::Native);
    request.limits = SolveLimits { memory_bytes: Some(1), ..SolveLimits::default() };

    let error = solve_model_silent(&package, &request).unwrap_err();
    assert!(error.to_string().contains("estimated SAT backend requires"), "{error}");
}

fn request(backend: SatBackendMode) -> SolveRequest {
    SolveRequest {
        mode: SolveMode::Exact,
        sat: SatControls { backend: Some(backend), preprocess: SatPreprocess::Full, proof_path: None },
        ..SolveRequest::default()
    }
}

#[test]
fn semantic_clause_model_uses_the_native_sat_plan() {
    let package = package(&[&[1, 2], &[-1, 2]], 2);
    let result = solve_model_silent(&package, &request(SatBackendMode::Native)).unwrap();

    assert_eq!(result.status(), SolveStatus::Satisfiable);
    assert_eq!(result.reports()[0].engine, Some(EngineKind::IntegerExact));
    assert_eq!(result.primal().unwrap().assignment().integers[1], Some(1));
}

#[test]
fn semantic_clause_model_can_select_the_linear_sat_plan() {
    let package = package(&[&[1], &[-1]], 1);
    let result = solve_model_silent(&package, &request(SatBackendMode::Linear)).unwrap();

    assert_eq!(result.status(), SolveStatus::Unsatisfiable);
    assert_eq!(result.reports()[0].engine, Some(EngineKind::Linear));
    assert!(result.proof().is_some());
}

#[test]
fn native_sat_plan_owns_requested_drat_output() {
    let package = package(&[&[1], &[-1]], 1);
    let path = std::env::temp_dir().join(format!("qayd-phase10-5-{}.drat", std::process::id()));
    let mut request = request(SatBackendMode::Native);
    request.sat.proof_path = Some(path.to_string_lossy().into_owned());

    let result = solve_model_silent(&package, &request).unwrap();
    let proof = fs::read_to_string(&path).unwrap();
    fs::remove_file(&path).unwrap();

    assert_eq!(result.status(), SolveStatus::Unsatisfiable);
    assert!(proof.lines().any(|line| line == "0"));
    assert!(result.reports()[0].metadata.contains(&("proof_format".to_string(), "drat".to_string())));
}
