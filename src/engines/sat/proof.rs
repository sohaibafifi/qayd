//! SAT proof output helpers.
//!
//! This module owns only proof-file syntax. Solver integration and preprocessing
//! proof steps live in the core SAT engine and the LCG trail.

use std::io::{self, Write};

/// Supported SAT proof output formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofFormat {
    /// DRAT text proof format.
    Drat,
}

/// One proof step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofStep {
    /// Add a clause.
    Add(Vec<i32>),
    /// Delete a clause.
    Delete(Vec<i32>),
}

/// Text DRAT writer.
pub struct ProofWriter<W> {
    out: W,
    format: ProofFormat,
}

impl<W: Write> ProofWriter<W> {
    /// Create a proof writer for the selected format.
    pub fn new(out: W, format: ProofFormat) -> Self {
        Self { out, format }
    }

    /// Write one proof step.
    pub fn write_step(&mut self, step: &ProofStep) -> io::Result<()> {
        match self.format {
            ProofFormat::Drat => self.write_drat_step(step),
        }
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }

    /// Return the wrapped writer.
    pub fn into_inner(self) -> W {
        self.out
    }

    fn write_drat_step(&mut self, step: &ProofStep) -> io::Result<()> {
        match step {
            ProofStep::Add(clause) => write_clause(&mut self.out, "", clause),
            ProofStep::Delete(clause) => write_clause(&mut self.out, "d ", clause),
        }
    }
}

fn write_clause(out: &mut impl Write, prefix: &str, clause: &[i32]) -> io::Result<()> {
    out.write_all(prefix.as_bytes())?;
    for &lit in clause {
        write!(out, "{lit} ")?;
    }
    out.write_all(b"0\n")
}
