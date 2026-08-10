//! Compiled plan for a semantic Boolean clause model.

use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use crate::engines::sat::proof::{ProofFormat, ProofWriter};
use crate::engines::sat::{
    solve_cnf_native_seeded_with_proof_options, solve_cnf_with_backend_seeded_options, Cnf, PreprocessOptions, PreprocessStats, SatBackend,
    SatResult, Status,
};
use crate::model::{Constraint, IntDomain, Model};

use super::{
    finalize_decision_result, Assignment, CandidateSolution, DecisionOutcome, EngineKind, EventSink, SatBackendMode, SatPreprocess,
    SolveBudget, SolveError, SolveRequest, SolveResult, VerificationLevel,
};

#[derive(Clone)]
pub(crate) struct SatSolvePlan {
    cnf: Cnf,
    backend: SatBackend,
    preprocess: PreprocessOptions,
    proof_path: Option<String>,
    estimated_bytes: u64,
}

impl SatSolvePlan {
    pub(crate) fn engine(&self) -> EngineKind {
        match self.backend {
            SatBackend::Native => EngineKind::IntegerExact,
            SatBackend::Linear => EngineKind::Linear,
        }
    }

    pub(crate) fn estimated_backend_bytes(&self) -> u64 {
        self.estimated_bytes
    }
}

pub(crate) fn compile_sat_plan(model: &Model, request: &SolveRequest, budget: &SolveBudget) -> Result<SatSolvePlan, SolveError> {
    request.validate()?;
    if budget.expired() {
        return Err(SolveError::Interrupted("solve budget expired before SAT compilation".to_string()));
    }
    let backend = match request.sat.backend {
        Some(SatBackendMode::Native) => SatBackend::Native,
        Some(SatBackendMode::Linear) => SatBackend::Linear,
        None => return Err(SolveError::Compile("the specialized SAT compiler was not requested".to_string())),
    };
    if request.mode == super::SolveMode::LocalSearch {
        return Err(SolveError::InvalidRequest("the specialized SAT plan does not support local-search mode".to_string()));
    }
    if request.threads != 1 {
        return Err(SolveError::InvalidRequest("the specialized SAT plan currently supports exactly one worker".to_string()));
    }
    if request.limits.conflicts.is_some() || request.limits.iterations.is_some() {
        return Err(SolveError::InvalidRequest("the specialized SAT plan does not support conflict or iteration limits".to_string()));
    }
    if request.schedule_cdcl || request.routing != super::RoutingControls::default() || request.cp != super::CpControls::default() {
        return Err(SolveError::InvalidRequest(
            "the specialized SAT plan does not accept scheduling, routing, or CP portfolio controls".to_string(),
        ));
    }
    if !request.assumptions.is_empty()
        || !request.hints.is_empty()
        || request.primary_branch_scope.is_some()
        || !request.branch_order.is_empty()
        || request.list_hint.is_some()
        || request.publish_incumbent_assignments
    {
        return Err(SolveError::InvalidRequest(
            "the specialized SAT plan does not accept CP assumptions, hints, primary branch scope, branch order, list hints, or incumbent callbacks"
                .to_string(),
        ));
    }
    if !model.sets().is_empty() || !model.lists().is_empty() || !model.intervals().is_empty() || !model.objectives().is_empty() {
        return Err(SolveError::Unsupported("the specialized SAT compiler accepts only Boolean variables and clauses".to_string()));
    }
    if model.int_vars().iter().any(|domain| *domain != IntDomain::Bool) {
        return Err(SolveError::Unsupported("the specialized SAT compiler accepts only Boolean variable domains".to_string()));
    }
    let mut literal_count = 0u128;
    for constraint in model.constraints() {
        if budget.expired() {
            return Err(SolveError::Interrupted("solve budget expired during SAT memory preflight".to_string()));
        }
        let Constraint::Clause(literals) = constraint else {
            return Err(SolveError::Unsupported("the specialized SAT compiler accepts only clause constraints".to_string()));
        };
        literal_count = literal_count.saturating_add(literals.len() as u128);
        for literal in literals {
            if budget.expired() {
                return Err(SolveError::Interrupted("solve budget expired during SAT memory preflight".to_string()));
            }
            i32::try_from(literal.variable.0.saturating_add(1))
                .map_err(|_| SolveError::Compile("SAT variable index exceeds DIMACS i32 range".to_string()))?;
        }
    }
    let estimated_bytes = (model.int_vars().len() as u128)
        .saturating_mul(32)
        .saturating_add((model.constraints().len() as u128).saturating_mul(32))
        .saturating_add(literal_count.saturating_mul(8));
    let estimated_bytes = u64::try_from(estimated_bytes).unwrap_or(u64::MAX);
    if request.limits.memory_bytes.is_some_and(|limit| estimated_bytes > limit) {
        return Err(SolveError::Compile(format!("estimated SAT backend requires {estimated_bytes} bytes, above the memory limit")));
    }
    let mut clauses = Vec::with_capacity(model.constraints().len());
    for constraint in model.constraints() {
        if budget.expired() {
            return Err(SolveError::Interrupted("solve budget expired during SAT compilation".to_string()));
        }
        let Constraint::Clause(literals) = constraint else {
            unreachable!("SAT memory preflight accepted only clauses");
        };
        let mut clause = Vec::with_capacity(literals.len());
        for literal in literals {
            if budget.expired() {
                return Err(SolveError::Interrupted("solve budget expired during SAT compilation".to_string()));
            }
            let variable = i32::try_from(literal.variable.0.saturating_add(1))
                .map_err(|_| SolveError::Compile("SAT variable index exceeds DIMACS i32 range".to_string()))?;
            clause.push(if literal.positive { variable } else { -variable });
        }
        clauses.push(clause);
    }
    let preprocess = match request.sat.preprocess {
        SatPreprocess::Off => PreprocessOptions::off(),
        SatPreprocess::Basic => PreprocessOptions::basic(),
        SatPreprocess::Full => PreprocessOptions::full(),
    };
    Ok(SatSolvePlan {
        cnf: Cnf { vars: model.int_vars().len(), clauses },
        backend,
        preprocess,
        proof_path: request.sat.proof_path.clone(),
        estimated_bytes,
    })
}

pub(crate) fn solve_sat_plan(
    model: &Model,
    plan: &SatSolvePlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    _sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    if request.sat.proof_path != plan.proof_path {
        return Err(SolveError::InvalidRequest("SAT proof output must match the request used to compile the plan".to_string()));
    }
    let search_stop = budget.search_stop();
    let engine_stop = search_stop.flag();
    let started = Instant::now();
    let (raw, drat) = match plan.proof_path.as_deref() {
        None => (
            if budget.expired() {
                stopped_sat_result()
            } else {
                solve_cnf_with_backend_seeded_options(&plan.cnf, engine_stop, plan.backend, request.seed, plan.preprocess)
            },
            false,
        ),
        Some(path) => {
            let file = File::create(path).map_err(|error| SolveError::Engine(format!("cannot create SAT proof file {path}: {error}")))?;
            let mut proof = ProofWriter::new(BufWriter::new(file), ProofFormat::Drat);
            let raw = if budget.expired() {
                proof.flush().map_err(|error| SolveError::Engine(format!("cannot write SAT proof file {path}: {error}")))?;
                stopped_sat_result()
            } else {
                solve_cnf_native_seeded_with_proof_options(&plan.cnf, engine_stop, request.seed, plan.preprocess, &mut proof)
                    .map_err(|error| SolveError::Engine(format!("cannot write SAT proof file {path}: {error}")))?
            };
            (raw, true)
        }
    };
    finalize_sat_result(model, &plan.cnf, raw, plan.backend, drat, started, budget)
}

fn stopped_sat_result() -> SatResult {
    SatResult { status: Status::Unknown, assignment: None, stats: Default::default(), preprocess: PreprocessStats::default() }
}

#[allow(clippy::too_many_arguments)]
fn finalize_sat_result(
    model: &Model,
    cnf: &Cnf,
    raw: SatResult,
    backend: SatBackend,
    drat: bool,
    started: Instant,
    budget: &SolveBudget,
) -> Result<SolveResult, SolveError> {
    let engine = match backend {
        SatBackend::Native => EngineKind::IntegerExact,
        SatBackend::Linear => EngineKind::Linear,
    };
    let SatResult { status, assignment, stats, preprocess } = raw;
    let complete = status != Status::Unknown;
    let candidate = match (status, assignment) {
        (Status::Satisfiable, Some(assignment)) => {
            Some(super::verify_final_with_budget(budget, |stop| verified_sat_candidate(model, cnf, &assignment, engine, stop))?)
        }
        (Status::Satisfiable, None) => {
            return Err(SolveError::InvalidResult("SAT backend omitted its satisfying assignment".to_string()));
        }
        (Status::Unsatisfiable | Status::Unknown, None) => None,
        (Status::Unsatisfiable | Status::Unknown, Some(_)) => {
            return Err(SolveError::InvalidResult("non-SAT backend result unexpectedly carries an assignment".to_string()));
        }
    };
    let mut metadata = preprocessing_metadata(preprocess);
    metadata.push((
        "sat_backend".to_string(),
        match backend {
            SatBackend::Native => "native",
            SatBackend::Linear => "linear",
        }
        .to_string(),
    ));
    if drat {
        metadata.push(("proof_format".to_string(), "drat".to_string()));
    }
    let outcome = DecisionOutcome::exact(engine, candidate, complete, stats, started.elapsed(), metadata)?;
    finalize_decision_result(outcome, budget)
}

fn verified_sat_candidate(
    model: &Model,
    cnf: &Cnf,
    values: &[bool],
    engine: EngineKind,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<CandidateSolution, SolveError> {
    if values.len() != cnf.vars {
        return Err(SolveError::InvalidResult(format!("SAT backend returned {} values for {} variables", values.len(), cnf.vars)));
    }
    for (clause_index, clause) in cnf.clauses.iter().enumerate() {
        if clause_index & 0x3f == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SolveError::Interrupted("SAT assignment replay was interrupted".to_string()));
        }
        let mut satisfied = false;
        for (literal_index, &literal) in clause.iter().enumerate() {
            if literal_index & 0xff == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
                return Err(SolveError::Interrupted("SAT assignment replay was interrupted".to_string()));
            }
            let value = values[literal.unsigned_abs() as usize - 1];
            if if literal > 0 { value } else { !value } {
                satisfied = true;
                break;
            }
        }
        if !satisfied {
            return Err(SolveError::InvalidResult("SAT backend returned an invalid assignment".to_string()));
        }
    }
    let mut integers = Vec::with_capacity(values.len());
    for (index, &value) in values.iter().enumerate() {
        if index & 0xff == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SolveError::Interrupted("SAT assignment decoding was interrupted".to_string()));
        }
        integers.push(Some(i64::from(value)));
    }
    let assignment = Assignment { integers, sets: Vec::new(), lists: Vec::new(), intervals: Vec::new() };
    let objectives = super::verify_semantic_assignment_validated_interruptible(model, &assignment, &[], stop)?;
    Ok(CandidateSolution::verified(assignment, objectives, engine, VerificationLevel::Final))
}

fn preprocessing_metadata(stats: PreprocessStats) -> Vec<(String, String)> {
    [
        ("pre_rounds", stats.rounds),
        ("pre_in", stats.input_clauses),
        ("pre_out", stats.output_clauses),
        ("pre_dup_lits", stats.duplicate_literals),
        ("pre_taut", stats.tautological_clauses),
        ("pre_units", stats.unit_assignments),
        ("pre_pure", stats.pure_assignments),
        ("pre_subsumed", stats.subsumed_clauses),
        ("pre_ssr_lits", stats.self_subsumed_literals),
        ("pre_bve_vars", stats.bve_variables),
        ("pre_bve_resolvents", stats.bve_resolvents),
        ("pre_blocked", stats.blocked_clauses),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect()
}
