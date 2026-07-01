//! SAT proof output helpers.
//!
//! This module owns only proof-file syntax. Solver integration and preprocessing
//! proof steps live in the SAT front-end and the LCG trail.

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

#[cfg(test)]
mod tests {
    use super::{ProofFormat, ProofStep, ProofWriter};

    #[test]
    fn drat_writer_serializes_add_delete_and_empty_clause() {
        let mut writer = ProofWriter::new(Vec::new(), ProofFormat::Drat);
        writer.write_step(&ProofStep::Add(vec![1, -2, 3])).unwrap();
        writer.write_step(&ProofStep::Delete(vec![-4])).unwrap();
        writer.write_step(&ProofStep::Add(Vec::new())).unwrap();
        let bytes = writer.into_inner();
        assert_eq!(String::from_utf8(bytes).unwrap(), "1 -2 3 0\nd -4 0\n0\n");
    }
}
