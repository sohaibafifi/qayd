//! DIMACS CNF front-end for the `qayd` solver.
//!
//! This package intentionally lives outside the core `src/` tree. It lowers CNF
//! clauses to Boolean finite-domain variables and linear clause constraints:
//! `sum(positive x) + sum(negative (1 - x)) >= 1`.

use std::sync::atomic::{AtomicBool, Ordering};

use qayd::constraints::linear::{linear, Relation};
use qayd::{solve_interruptible, SearchControl, SolveStats, Solver, VarId};

/// Parsed DIMACS CNF instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cnf {
    /// Number of variables declared by `p cnf`.
    pub vars: usize,
    /// Clauses as signed DIMACS literals.
    pub clauses: Vec<Vec<i32>>,
}

/// SAT solving status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Satisfiable,
    Unsatisfiable,
    Unknown,
}

/// SAT solve result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SatResult {
    pub status: Status,
    /// DIMACS-style assignment: index 1..=n, value `true` means variable is true.
    pub assignment: Option<Vec<bool>>,
    pub stats: SolveStats,
}

/// Parse a DIMACS CNF string.
///
/// Supports comments (`c ...`), one `p cnf <vars> <clauses>` line, arbitrary
/// whitespace, clauses spanning multiple lines, and the SATLIB `%` end marker.
/// Every clause must end with `0`.
pub fn parse_dimacs(input: &str) -> Result<Cnf, String> {
    let mut declared: Option<(usize, usize)> = None;
    let mut tokens = Vec::new();

    for (line_no, raw) in input.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        if line.starts_with('%') {
            break;
        }
        if line.starts_with('p') {
            if declared.is_some() {
                return Err(format!("line {}: duplicate problem line", line_no + 1));
            }
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 4 || parts[0] != "p" || parts[1] != "cnf" {
                return Err(format!("line {}: expected `p cnf <vars> <clauses>`", line_no + 1));
            }
            let vars = parts[2].parse::<usize>().map_err(|_| format!("line {}: invalid variable count", line_no + 1))?;
            let clauses = parts[3].parse::<usize>().map_err(|_| format!("line {}: invalid clause count", line_no + 1))?;
            declared = Some((vars, clauses));
            continue;
        }
        tokens.extend(line.split_whitespace().map(str::to_string));
    }

    let (vars, expected_clauses) = declared.ok_or_else(|| "missing `p cnf` line".to_string())?;
    let mut clauses = Vec::new();
    let mut current = Vec::new();
    for token in tokens {
        let lit = token.parse::<i32>().map_err(|_| format!("invalid literal `{token}`"))?;
        if lit == 0 {
            clauses.push(std::mem::take(&mut current));
            continue;
        }
        let var = lit.unsigned_abs() as usize;
        if var == 0 || var > vars {
            return Err(format!("literal `{lit}` references variable outside 1..={vars}"));
        }
        current.push(lit);
    }
    if !current.is_empty() {
        return Err("last clause is missing terminating 0".to_string());
    }
    if clauses.len() != expected_clauses {
        return Err(format!("expected {expected_clauses} clauses, found {}", clauses.len()));
    }
    Ok(Cnf { vars, clauses })
}

/// Solve a CNF instance with the public finite-domain API.
pub fn solve_cnf(cnf: &Cnf, stop: &AtomicBool) -> SatResult {
    if cnf.clauses.iter().any(Vec::is_empty) {
        return SatResult { status: Status::Unsatisfiable, assignment: None, stats: SolveStats::default() };
    }

    let mut solver = Solver::new();
    let vars = (0..cnf.vars).map(|_| solver.new_var_range(0, 1)).collect::<Vec<_>>();
    for clause in &cnf.clauses {
        post_clause(&mut solver, &vars, clause);
    }

    let mut assignment = None;
    let stats = solve_interruptible(
        &mut solver,
        &vars,
        |s| {
            assignment = Some(vars.iter().map(|&v| s.store.value(v) != 0).collect::<Vec<_>>());
            SearchControl::Stop
        },
        stop,
    );

    match assignment {
        Some(values) => SatResult { status: Status::Satisfiable, assignment: Some(values), stats },
        None if stop.load(Ordering::Relaxed) => SatResult { status: Status::Unknown, assignment: None, stats },
        None => SatResult { status: Status::Unsatisfiable, assignment: None, stats },
    }
}

fn post_clause(solver: &mut Solver, vars: &[VarId], clause: &[i32]) {
    let mut coeffs = Vec::with_capacity(clause.len());
    let mut clause_vars = Vec::with_capacity(clause.len());
    let mut neg = 0i64;
    for &lit in clause {
        let var = vars[lit.unsigned_abs() as usize - 1];
        clause_vars.push(var);
        if lit > 0 {
            coeffs.push(1);
        } else {
            coeffs.push(-1);
            neg += 1;
        }
    }
    linear(solver, &coeffs, &clause_vars, Relation::Ge, 1 - neg);
}

/// Validate that `assignment` satisfies `cnf`.
pub fn assignment_satisfies(cnf: &Cnf, assignment: &[bool]) -> bool {
    assignment.len() == cnf.vars
        && cnf.clauses.iter().all(|clause| {
            clause.iter().any(|&lit| {
                let value = assignment[lit.unsigned_abs() as usize - 1];
                if lit > 0 {
                    value
                } else {
                    !value
                }
            })
        })
}
