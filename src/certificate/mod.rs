//! Bounded external CP certificates over an explicit semantic model slice.
//! Generation is a separate proof search, not a native LCG event log.

pub(crate) mod hash;
pub(crate) mod model;
pub(crate) mod produce;

use std::path::PathBuf;

pub use crate::orchestrator::certificate::{solve_with_certificate, CertificateRun, CheckedCpCertificate};

#[derive(Clone, Debug)]
pub struct CertificateLimits {
    pub max_variables: usize,
    pub max_constraints: usize,
    pub max_input_entries: usize,
    pub max_nodes: u64,
    pub max_steps: u64,
    pub max_work: u64,
    pub max_depth: usize,
    pub max_model_bytes: usize,
    pub max_proof_bytes: usize,
}

impl Default for CertificateLimits {
    fn default() -> Self {
        Self {
            max_variables: 64,
            max_constraints: 512,
            max_input_entries: 100_000,
            max_nodes: 100_000,
            max_steps: 200_000,
            max_work: 20_000_000,
            max_depth: 128,
            max_model_bytes: 8 * 1024 * 1024,
            max_proof_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CertificateOptions {
    /// A new directory; an existing directory is never overwritten.
    pub directory: PathBuf,
    pub limits: CertificateLimits,
}

impl CertificateOptions {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self { directory: directory.into(), limits: CertificateLimits::default() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertificateClaim {
    Satisfiable,
    Unsatisfiable,
    Optimal,
}

impl CertificateClaim {
    pub fn token(self) -> &'static str {
        match self {
            Self::Satisfiable => "sat",
            Self::Unsatisfiable => "unsat",
            Self::Optimal => "optimal",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CertificateError {
    Unsupported(String),
    InvalidInput(String),
    Interrupted,
    Limit(String),
    Io(String),
    ClaimDisagreement(String),
    CheckerRejected(String),
    CheckerUnavailable(String),
    Solver(String),
    Incomplete(String),
}

impl std::fmt::Display for CertificateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CertificateError {}

#[derive(Clone, Debug, Default)]
pub struct CertificateStats {
    pub nodes: u64,
    pub steps: u64,
    pub work: u64,
    pub deductions: u64,
    pub conflicts: u64,
    pub learned_clauses: u64,
    pub splits: u64,
    pub model_bytes: u64,
    pub proof_bytes: u64,
    pub solve_seconds: f64,
    pub generation_seconds: f64,
    pub checker_seconds: f64,
    pub elapsed_seconds: f64,
}
