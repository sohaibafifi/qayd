//! Opt-in external certification without changing canonical result authority.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::certificate::{hash, model, produce, CertificateClaim, CertificateError, CertificateOptions, CertificateStats};
use crate::model::ModelPackage;

use super::{IgnoreEvents, SemanticAssumption, SolveBudget, SolveMode, SolveRequest, SolveResult, SolveStatus};

const CHECKER: &str = include_str!("../../tools/check_cp_certificate.py");

/// Acceptance of specific bytes by the bundled independent checker at the time
/// of this call. Files can subsequently change; retain and compare their hashes.
/// The claim includes every returned assumption, even when the caller omits it
/// from a later solve. This type does not confer native ProofClaim authority.
#[derive(Clone, Debug)]
pub struct CheckedCpCertificate {
    claim: CertificateClaim,
    model_path: PathBuf,
    proof_path: PathBuf,
    model_sha256: String,
    proof_sha256: String,
    checker_sha256: String,
    assumptions: Vec<SemanticAssumption>,
    stats: CertificateStats,
}

impl CheckedCpCertificate {
    pub fn claim(&self) -> CertificateClaim {
        self.claim
    }
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
    pub fn proof_path(&self) -> &Path {
        &self.proof_path
    }
    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }
    pub fn proof_sha256(&self) -> &str {
        &self.proof_sha256
    }
    pub fn checker_sha256(&self) -> &str {
        &self.checker_sha256
    }
    pub fn assumptions(&self) -> &[SemanticAssumption] {
        &self.assumptions
    }
    pub fn stats(&self) -> &CertificateStats {
        &self.stats
    }
}

#[derive(Debug)]
pub struct CertificateRun {
    pub result: SolveResult,
    pub certificate: Result<CheckedCpCertificate, CertificateError>,
    pub stats: CertificateStats,
}

/// Solve normally, then generate and independently check a bounded certificate.
///
/// One wall-clock/cancellation budget covers all phases. Unsupported preflight
/// and native solve errors are returned directly. After a native result exists,
/// certificate failure preserves it in `CertificateRun.result`. A native result
/// is not upgraded, downgraded or relabeled when certification fails.
///
/// This first slice accepts single-threaded Auto/Exact integer requests, without
/// SAT proof output or a process memory limit. The subprocess is independently
/// bounded by work, step, depth and input byte caps; the existing parent-process
/// memory monitor cannot enforce a combined parent/child memory-byte limit.
/// Python 3 must be available as `python3`. An existing output directory fails.
pub fn solve_with_certificate(
    package: &ModelPackage,
    request: &SolveRequest,
    options: &CertificateOptions,
    stop: Arc<AtomicBool>,
) -> Result<CertificateRun, CertificateError> {
    let budget = SolveBudget::with_stop(request.limits.time, stop);
    let stopped = || budget_stopped(&budget);
    check_budget(&budget)?;
    // Bound the canonical validator's non-interruptible metadata loops before
    // entering them. Semantic export and proof work poll the shared budget.
    let metadata_cap = options.limits.max_variables.min(64);
    if request.primary_branch_scope.as_ref().is_some_and(|scope| scope.len() > metadata_cap)
        || request.branch_order.len() > metadata_cap
        || request.hints.len() > metadata_cap
        || request.search_policy.phases().len() > metadata_cap
    {
        return Err(CertificateError::Limit("request search metadata".into()));
    }
    for phase in request.search_policy.phases() {
        check_budget(&budget)?;
        if phase.scope().len() > metadata_cap {
            return Err(CertificateError::Limit("request search phase scope".into()));
        }
    }
    if options.limits.max_depth > 128
        || options.limits.max_variables > 64
        || options.limits.max_constraints > 512
        || options.limits.max_input_entries > 100_000
    {
        return Err(CertificateError::Unsupported(
            "certificate v1 ceilings: 128 depth, 64 variables, 512 constraints, 100000 input entries".into(),
        ));
    }
    request.validate().map_err(|error| CertificateError::InvalidInput(error.to_string()))?;
    if !matches!(request.mode, SolveMode::Auto | SolveMode::Exact)
        || request.threads != 1
        || request.sat.proof_path.is_some()
        || request.limits.memory_bytes.is_some()
        || request.limits.iterations.is_some()
        || request.list_hint.is_some()
    {
        return Err(CertificateError::Unsupported(
            "certificate v1 needs Auto/Exact, one thread, no SAT proof path, process memory limit or LS iteration limit".into(),
        ));
    }
    let proof_model = model::from_package(package, request, &options.limits, &stopped)?;
    let model_text = proof_model.encode(&options.limits, &stopped)?;
    let model_sha256 = hash::sha256(model_text.as_bytes(), &stopped)?;
    check_budget(&budget)?;
    let solve_started = Instant::now();
    let result = super::solve::solve_model_with_budget(package, request, &budget, &mut IgnoreEvents)
        .map_err(|error| CertificateError::Solver(error.to_string()))?;
    let mut stats = CertificateStats { solve_seconds: solve_started.elapsed().as_secs_f64(), ..CertificateStats::default() };
    let certificate = (|| {
        check_budget(&budget)?;
        let claim = match result.status() {
            SolveStatus::Satisfiable => CertificateClaim::Satisfiable,
            SolveStatus::Unsatisfiable => CertificateClaim::Unsatisfiable,
            SolveStatus::Optimal => CertificateClaim::Optimal,
            status => return Err(CertificateError::Incomplete(format!("native status {}", status.as_str()))),
        };
        let witness = if claim == CertificateClaim::Unsatisfiable {
            None
        } else {
            let primal = result.primal().ok_or_else(|| CertificateError::ClaimDisagreement("missing witness".into()))?;
            Some(
                primal
                    .assignment()
                    .integers
                    .iter()
                    .map(|value| {
                        value
                            .and_then(|value| i32::try_from(value).ok())
                            .ok_or_else(|| CertificateError::ClaimDisagreement("incomplete or non-i32 witness".into()))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };
        let generated = produce::generate(&proof_model, claim, witness.as_deref(), &model_sha256, &options.limits, &stopped)?;
        let solve_seconds = stats.solve_seconds;
        stats = generated.stats;
        stats.solve_seconds = solve_seconds;
        stats.model_bytes = model_text.len() as u64;
        stats.proof_bytes = generated.text.len() as u64;
        let proof_sha256 = hash::sha256(generated.text.as_bytes(), &stopped)?;
        let checker_sha256 = hash::sha256(CHECKER.as_bytes(), &stopped)?;
        check_budget(&budget)?;
        fs::create_dir(&options.directory).map_err(io_error)?;
        let directory = fs::canonicalize(&options.directory).map_err(io_error)?;
        let model_path = directory.join("model.qcp");
        let proof_path = directory.join("proof.qcp");
        write_new(&model_path, model_text.as_bytes(), &budget)?;
        write_new(&proof_path, generated.text.as_bytes(), &budget)?;
        stats.generation_seconds = budget.elapsed().as_secs_f64() - stats.solve_seconds;
        let expected_ack =
            format!("VERIFIED qayd-cp-v1 {} {} {} {}\n", model_sha256, claim.token(), stats.steps, request.assumptions.len());
        let checker_started = Instant::now();
        let checked =
            run_checker("python3", &model_path, &proof_path, &model_sha256, &proof_sha256, claim, options, &budget, &expected_ack);
        stats.checker_seconds = checker_started.elapsed().as_secs_f64();
        checked?;
        // Bind the handle to the actual bytes checked, including the proof.
        // A changed file can never produce a handle for our generated bytes.
        verify_saved(&model_path, model_text.as_bytes(), &budget)?;
        verify_saved(&proof_path, generated.text.as_bytes(), &budget)?;
        check_budget(&budget)?;
        Ok(CheckedCpCertificate {
            claim,
            model_path,
            proof_path,
            model_sha256,
            proof_sha256,
            checker_sha256,
            assumptions: request.assumptions.clone(),
            stats: CertificateStats::default(),
        })
    })();
    stats.elapsed_seconds = budget.elapsed().as_secs_f64();
    if stats.generation_seconds == 0.0 {
        stats.generation_seconds = (stats.elapsed_seconds - stats.solve_seconds - stats.checker_seconds).max(0.0);
    }
    let certificate = certificate.map(|mut checked| {
        checked.stats = stats.clone();
        checked
    });
    Ok(CertificateRun { result, certificate, stats })
}

fn budget_stopped(budget: &SolveBudget) -> bool {
    budget.expired() || budget.remaining().is_some_and(|remaining| remaining.is_zero())
}

fn check_budget(budget: &SolveBudget) -> Result<(), CertificateError> {
    if budget_stopped(budget) {
        Err(CertificateError::Interrupted)
    } else {
        Ok(())
    }
}

fn io_error(error: std::io::Error) -> CertificateError {
    CertificateError::Io(error.to_string())
}

fn write_new(path: &Path, bytes: &[u8], budget: &SolveBudget) -> Result<(), CertificateError> {
    check_budget(budget)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path).map_err(io_error)?;
    for chunk in bytes.chunks(16 * 1024) {
        check_budget(budget)?;
        file.write_all(chunk).map_err(io_error)?;
    }
    file.flush().map_err(io_error)?;
    check_budget(budget)
}

fn verify_saved(path: &Path, expected: &[u8], budget: &SolveBudget) -> Result<(), CertificateError> {
    let mut file = File::open(path).map_err(io_error)?;
    let mut buffer = [0_u8; 16 * 1024];
    for chunk in expected.chunks(buffer.len()) {
        check_budget(budget)?;
        file.read_exact(&mut buffer[..chunk.len()]).map_err(io_error)?;
        if &buffer[..chunk.len()] != chunk {
            return Err(CertificateError::CheckerRejected("artifact changed during verification".into()));
        }
    }
    if file.read(&mut buffer[..1]).map_err(io_error)? != 0 {
        return Err(CertificateError::CheckerRejected("artifact grew during verification".into()));
    }
    check_budget(budget)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_checker(
    interpreter: &str,
    model_path: &Path,
    proof_path: &Path,
    model_sha256: &str,
    proof_sha256: &str,
    claim: CertificateClaim,
    options: &CertificateOptions,
    budget: &SolveBudget,
    expected_ack: &str,
) -> Result<(), CertificateError> {
    check_budget(budget)?;
    let stdout_path = proof_path.with_extension("checker.stdout");
    let stderr_path = proof_path.with_extension("checker.stderr");
    let stdout = OpenOptions::new().write(true).create_new(true).open(&stdout_path).map_err(io_error)?;
    let stderr = OpenOptions::new().write(true).create_new(true).open(&stderr_path).map_err(io_error)?;
    let mut command = Command::new(interpreter);
    command
        .args(["-I", "-c", CHECKER, "--model"])
        .arg(model_path)
        .arg("--proof")
        .arg(proof_path)
        .args(["--expected-model-sha256", model_sha256, "--expected-claim", claim.token()])
        .args(["--expected-proof-sha256", proof_sha256])
        .arg("--max-work")
        .arg(options.limits.max_work.to_string())
        .arg("--max-steps")
        .arg(options.limits.max_steps.to_string())
        .arg("--max-depth")
        .arg(options.limits.max_depth.to_string())
        .arg("--max-model-bytes")
        .arg(options.limits.max_model_bytes.to_string())
        .arg("--max-proof-bytes")
        .arg(options.limits.max_proof_bytes.to_string())
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr);
    if let Some(remaining) = budget.remaining() {
        command.arg("--time-limit").arg(remaining.as_secs_f64().to_string());
    }
    let mut child = command.spawn().map_err(|error| CertificateError::CheckerUnavailable(error.to_string()))?;
    let status = loop {
        if budget_stopped(budget) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(CertificateError::Interrupted);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(1)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(io_error(error));
            }
        }
    };
    check_budget(budget)?;
    let read_bounded = |path: &Path| -> Result<String, CertificateError> {
        let mut text = String::new();
        File::open(path).map_err(io_error)?.take(4097).read_to_string(&mut text).map_err(io_error)?;
        if text.len() > 4096 {
            return Err(CertificateError::CheckerRejected("oversized checker response".into()));
        }
        Ok(text)
    };
    let stdout = read_bounded(&stdout_path)?;
    if !status.success() || stdout != expected_ack {
        return Err(CertificateError::CheckerRejected(format!("checker status {status}: {}", read_bounded(&stderr_path)?)));
    }
    check_budget(budget)
}
