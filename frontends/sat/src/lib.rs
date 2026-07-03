//! DIMACS CNF front-end for the `qayd` solver.
//!
//! This package intentionally lives outside the core `src/` tree. The default
//! solve path injects clauses directly into the LCG watched-clause database.
//! The previous linear lowering remains available for regression comparisons.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

use qayd::constraints::linear::{linear, Relation};
use qayd::{
    solve_bool_cnf_interruptible, solve_bool_cnf_seeded, solve_bool_cnf_seeded_with_proof, solve_interruptible, BoolLit, SearchControl,
    SolveStats, Solver, VarId,
};

pub mod proof;

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
    pub preprocess: PreprocessStats,
}

/// Backend used by the SAT front-end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SatBackend {
    /// Direct clauses in the LCG watched-clause database.
    Native,
    /// Clause linearization through the public finite-domain API.
    Linear,
}

/// Optional CNF preprocessing passes used by the native SAT backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreprocessOptions {
    /// Remove duplicate literals and tautological clauses.
    pub basic_simplification: bool,
    /// Propagate unit clauses at load time.
    pub unit_propagation: bool,
    /// Assign pure literals and keep them as root unit clauses.
    pub pure_literals: bool,
    /// Remove clauses subsumed by another clause.
    pub subsumption: bool,
    /// Remove a literal by self-subsuming resolution.
    pub self_subsuming_resolution: bool,
    /// Eliminate variables when the number of resolvents does not grow.
    pub bounded_variable_elimination: bool,
    /// Remove blocked clauses with model reconstruction.
    pub blocked_clause_elimination: bool,
}

impl PreprocessOptions {
    /// Disable all preprocessing passes.
    pub const fn off() -> Self {
        Self {
            basic_simplification: false,
            unit_propagation: false,
            pure_literals: false,
            subsumption: false,
            self_subsuming_resolution: false,
            bounded_variable_elimination: false,
            blocked_clause_elimination: false,
        }
    }

    /// Enable the cheap model-preserving preprocessing baseline.
    pub const fn basic() -> Self {
        Self {
            basic_simplification: true,
            unit_propagation: true,
            pure_literals: true,
            subsumption: false,
            self_subsuming_resolution: false,
            bounded_variable_elimination: false,
            blocked_clause_elimination: false,
        }
    }

    /// Enable all currently implemented model-preserving preprocessing passes.
    pub const fn full() -> Self {
        Self {
            basic_simplification: true,
            unit_propagation: true,
            pure_literals: true,
            subsumption: true,
            self_subsuming_resolution: true,
            bounded_variable_elimination: true,
            blocked_clause_elimination: true,
        }
    }
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self::basic()
    }
}

/// Counters for CNF preprocessing transformations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreprocessStats {
    pub rounds: u64,
    pub duplicate_literals: u64,
    pub tautological_clauses: u64,
    pub unit_assignments: u64,
    pub pure_assignments: u64,
    pub subsumed_clauses: u64,
    pub self_subsumed_literals: u64,
    pub bve_variables: u64,
    pub bve_resolvents: u64,
    pub blocked_clauses: u64,
    pub input_clauses: u64,
    pub output_clauses: u64,
}

/// Parse a DIMACS CNF string.
///
/// Supports comments (`c ...`), one `p cnf <vars> <clauses>` line, arbitrary
/// whitespace, clauses spanning multiple lines, and the SATLIB `%` end marker.
/// Standard clauses must end with `0`. As a compatibility fallback for old
/// SATLIB files, when the strict clause count does not match but the number of
/// non-comment data lines exactly matches the declared clause count, each data
/// line is accepted as one clause even if its trailing `0` is missing.
pub fn parse_dimacs(input: &str) -> Result<Cnf, String> {
    let mut declared: Option<(usize, usize)> = None;
    let mut tokens = Vec::new();
    let mut data_lines: Vec<(usize, Vec<String>)> = Vec::new();

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
        let line_tokens = line.split_whitespace().map(str::to_string).collect::<Vec<_>>();
        tokens.extend(line_tokens.iter().cloned());
        data_lines.push((line_no + 1, line_tokens));
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
        if let Some(clauses) = parse_line_terminated_clauses(vars, expected_clauses, &data_lines)? {
            return Ok(Cnf { vars, clauses });
        }
        return Err("last clause is missing terminating 0".to_string());
    }
    if clauses.len() != expected_clauses {
        if let Some(clauses) = parse_line_terminated_clauses(vars, expected_clauses, &data_lines)? {
            return Ok(Cnf { vars, clauses });
        }
        return Err(format!("expected {expected_clauses} clauses, found {}", clauses.len()));
    }
    Ok(Cnf { vars, clauses })
}

fn parse_line_terminated_clauses(
    vars: usize,
    expected_clauses: usize,
    data_lines: &[(usize, Vec<String>)],
) -> Result<Option<Vec<Vec<i32>>>, String> {
    if data_lines.len() != expected_clauses {
        return Ok(None);
    }
    let mut clauses = Vec::with_capacity(data_lines.len());
    for (line_no, tokens) in data_lines {
        if tokens.is_empty() {
            return Ok(None);
        }
        let mut clause = Vec::new();
        for (idx, token) in tokens.iter().enumerate() {
            let lit = token.parse::<i32>().map_err(|_| format!("line {line_no}: invalid literal `{token}`"))?;
            if lit == 0 {
                if idx + 1 != tokens.len() {
                    return Err(format!("line {line_no}: literal `0` must terminate the line"));
                }
                break;
            }
            let var = lit.unsigned_abs() as usize;
            if var == 0 || var > vars {
                return Err(format!("line {line_no}: literal `{lit}` references variable outside 1..={vars}"));
            }
            clause.push(lit);
        }
        if clause.is_empty() {
            return Err(format!("line {line_no}: empty line-terminated clause"));
        }
        clauses.push(clause);
    }
    Ok(Some(clauses))
}

/// Solve a CNF instance with explicit preprocessing options.
pub fn solve_cnf_with_backend_seeded_options(
    cnf: &Cnf,
    stop: &AtomicBool,
    backend: SatBackend,
    seed: u64,
    preprocess: PreprocessOptions,
) -> SatResult {
    match backend {
        SatBackend::Native => solve_cnf_native_seeded_options(cnf, stop, seed, preprocess),
        SatBackend::Linear => solve_cnf_linear(cnf, stop),
    }
}

/// Solve a CNF instance through the native backend with explicit preprocessing
/// options.
pub fn solve_cnf_native_seeded_options(cnf: &Cnf, stop: &AtomicBool, seed: u64, options: PreprocessOptions) -> SatResult {
    let (simplified, preprocess, reconstruction) = match preprocess_cnf(cnf, options) {
        PreprocessOutcome::Unsatisfiable(stats) => {
            return SatResult { status: Status::Unsatisfiable, assignment: None, stats: SolveStats::default(), preprocess: stats };
        }
        PreprocessOutcome::Simplified(cnf, stats, reconstruction) => (cnf, stats, reconstruction),
    };

    let mut solver = Solver::new();
    let vars = (0..simplified.vars).map(|_| solver.new_var_set(&[-1, 1])).collect::<Vec<_>>();
    let clauses = simplified
        .clauses
        .iter()
        .map(|clause| {
            clause
                .iter()
                .map(|&lit| {
                    let var = vars[lit.unsigned_abs() as usize - 1];
                    if lit > 0 {
                        BoolLit::positive(var)
                    } else {
                        BoolLit::negative(var)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let (assignment, stats, complete) = if seed == 0 {
        solve_bool_cnf_interruptible(&mut solver, &vars, &clauses, stop)
    } else {
        solve_bool_cnf_seeded(&mut solver, &vars, &clauses, stop, seed)
    }
    .expect("SAT frontend creates Boolean vars");
    match assignment {
        Some(values) => {
            let mut assignment = values.into_iter().map(|v| v > 0).collect::<Vec<_>>();
            reconstruct_assignment(&mut assignment, &reconstruction);
            debug_assert!(assignment_satisfies(cnf, &assignment));
            SatResult { status: Status::Satisfiable, assignment: Some(assignment), stats, preprocess }
        }
        None if !complete || stop.load(Ordering::Relaxed) => SatResult { status: Status::Unknown, assignment: None, stats, preprocess },
        None => SatResult { status: Status::Unsatisfiable, assignment: None, stats, preprocess },
    }
}

/// Solve a CNF instance with the native backend and stream a DRAT proof.
pub fn solve_cnf_native_seeded_with_proof<W: io::Write>(
    cnf: &Cnf,
    stop: &AtomicBool,
    seed: u64,
    writer: &mut proof::ProofWriter<W>,
) -> io::Result<SatResult> {
    solve_cnf_native_seeded_with_proof_options(cnf, stop, seed, PreprocessOptions::off(), writer)
}

/// Solve a CNF instance with preprocessing and stream a DRAT proof.
///
/// The proof is checked against the original CNF. Preprocessing steps are logged
/// as DRAT additions and deletions before CDCL learned clauses are streamed.
pub fn solve_cnf_native_seeded_with_proof_options<W: io::Write>(
    cnf: &Cnf,
    stop: &AtomicBool,
    seed: u64,
    options: PreprocessOptions,
    writer: &mut proof::ProofWriter<W>,
) -> io::Result<SatResult> {
    let mut write_error = None;
    let (simplified, preprocess, reconstruction) = match preprocess_cnf_with_proof(cnf, options, &mut |step| {
        if write_error.is_none() {
            if let Err(err) = writer.write_step(&step) {
                write_error = Some(err);
            }
        }
    }) {
        PreprocessOutcome::Unsatisfiable(stats) => {
            if let Some(err) = write_error {
                return Err(err);
            }
            writer.flush()?;
            return Ok(SatResult { status: Status::Unsatisfiable, assignment: None, stats: SolveStats::default(), preprocess: stats });
        }
        PreprocessOutcome::Simplified(cnf, stats, reconstruction) => (cnf, stats, reconstruction),
    };
    if let Some(err) = write_error {
        return Err(err);
    }

    let mut solver = Solver::new();
    let vars = (0..simplified.vars).map(|_| solver.new_var_set(&[-1, 1])).collect::<Vec<_>>();
    let clauses = cnf_to_bool_clauses(&simplified, &vars);
    let dimacs_by_var = dimacs_var_map(&vars);

    let mut write_error = None;
    let (assignment, stats, complete) = {
        let mut emit = |clause: &[BoolLit]| {
            if write_error.is_some() {
                return;
            }
            let dimacs = clause
                .iter()
                .map(|lit| {
                    let var = dimacs_by_var[lit.var.index()];
                    debug_assert_ne!(var, 0, "proof clause references a non-DIMACS variable");
                    if lit.value {
                        var
                    } else {
                        -var
                    }
                })
                .collect::<Vec<_>>();
            if let Err(err) = writer.write_step(&proof::ProofStep::Add(dimacs)) {
                write_error = Some(err);
            }
        };
        solve_bool_cnf_seeded_with_proof(&mut solver, &vars, &clauses, stop, seed, &mut emit).expect("SAT frontend creates Boolean vars")
    };
    if let Some(err) = write_error {
        return Err(err);
    }
    writer.flush()?;

    Ok(match assignment {
        Some(values) => {
            let mut assignment = values.into_iter().map(|v| v > 0).collect::<Vec<_>>();
            reconstruct_assignment(&mut assignment, &reconstruction);
            debug_assert!(assignment_satisfies(cnf, &assignment));
            SatResult { status: Status::Satisfiable, assignment: Some(assignment), stats, preprocess }
        }
        None if !complete || stop.load(Ordering::Relaxed) => SatResult { status: Status::Unknown, assignment: None, stats, preprocess },
        None => SatResult { status: Status::Unsatisfiable, assignment: None, stats, preprocess },
    })
}

fn cnf_to_bool_clauses(cnf: &Cnf, vars: &[VarId]) -> Vec<Vec<BoolLit>> {
    cnf.clauses
        .iter()
        .map(|clause| {
            clause
                .iter()
                .map(|&lit| {
                    let var = vars[lit.unsigned_abs() as usize - 1];
                    if lit > 0 {
                        BoolLit::positive(var)
                    } else {
                        BoolLit::negative(var)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn dimacs_var_map(vars: &[VarId]) -> Vec<i32> {
    let len = vars.iter().map(|var| var.index()).max().map_or(0, |idx| idx + 1);
    let mut dimacs_by_var = vec![0; len];
    for (idx, &var) in vars.iter().enumerate() {
        dimacs_by_var[var.index()] = idx as i32 + 1;
    }
    dimacs_by_var
}

fn canonical_clause(clause: &[i32]) -> Vec<i32> {
    let mut out = clause.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

struct PreprocessProof<'a> {
    log: Option<&'a mut dyn FnMut(proof::ProofStep)>,
}

impl<'a> PreprocessProof<'a> {
    fn none() -> Self {
        Self { log: None }
    }

    fn new(log: &'a mut dyn FnMut(proof::ProofStep)) -> Self {
        Self { log: Some(log) }
    }

    fn is_enabled(&self) -> bool {
        self.log.is_some()
    }

    fn add(&mut self, clause: &[i32]) {
        if let Some(log) = self.log.as_mut() {
            log(proof::ProofStep::Add(canonical_clause(clause)));
        }
    }

    fn delete(&mut self, clause: &[i32]) {
        if let Some(log) = self.log.as_mut() {
            log(proof::ProofStep::Delete(canonical_clause(clause)));
        }
    }
}

enum PreprocessOutcome {
    Simplified(Cnf, PreprocessStats, Vec<ReconstructionStep>),
    Unsatisfiable(PreprocessStats),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ReconstructionStep {
    EliminatedVariable { var: i32, positive: Vec<Vec<i32>>, negative: Vec<Vec<i32>> },
    BlockedClause { clause: Vec<i32>, blocking_lit: i32 },
}

fn preprocess_cnf(cnf: &Cnf, options: PreprocessOptions) -> PreprocessOutcome {
    let mut proof = PreprocessProof::none();
    preprocess_cnf_inner(cnf, options, &mut proof)
}

fn preprocess_cnf_with_proof(cnf: &Cnf, options: PreprocessOptions, proof: &mut dyn FnMut(proof::ProofStep)) -> PreprocessOutcome {
    let mut proof = PreprocessProof::new(proof);
    preprocess_cnf_inner(cnf, options, &mut proof)
}

fn preprocess_cnf_inner(cnf: &Cnf, options: PreprocessOptions, proof: &mut PreprocessProof<'_>) -> PreprocessOutcome {
    let mut stats = PreprocessStats { input_clauses: cnf.clauses.len() as u64, ..PreprocessStats::default() };
    let mut assigned = vec![None; cnf.vars];
    let mut clauses = cnf.clauses.clone();
    let mut reconstruction = Vec::new();

    loop {
        stats.rounds += 1;
        let mut changed = false;
        let mut simplified = Vec::with_capacity(clauses.len());
        let mut units = Vec::new();
        let mut polarity = vec![(false, false); cnf.vars];

        for clause in &clauses {
            let original = canonical_clause(clause);
            let mut out: Vec<i32> = Vec::with_capacity(clause.len());
            let mut satisfied = false;
            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                if var == 0 || var > cnf.vars {
                    return PreprocessOutcome::Unsatisfiable(stats);
                }
                let idx = var - 1;
                let value = lit > 0;
                match assigned[idx] {
                    Some(fixed) if fixed == value => {
                        satisfied = true;
                        proof.delete(&original);
                        break;
                    }
                    Some(_) => {
                        changed = true;
                    }
                    None => {
                        if options.basic_simplification && out.contains(&lit) {
                            stats.duplicate_literals += 1;
                            changed = true;
                        } else if options.basic_simplification && out.contains(&-lit) {
                            stats.tautological_clauses += 1;
                            satisfied = true;
                            changed = true;
                            proof.delete(&original);
                            break;
                        } else {
                            out.push(lit);
                        }
                    }
                }
            }
            if satisfied {
                changed = true;
                continue;
            }
            out.sort_unstable();
            if out.is_empty() {
                proof.add(&[]);
                return PreprocessOutcome::Unsatisfiable(stats);
            }
            if proof.is_enabled() && canonical_clause(&out) != original {
                proof.add(&out);
                proof.delete(&original);
            }
            if options.unit_propagation && out.len() == 1 {
                units.push(out[0]);
            }
            for &lit in &out {
                let p = &mut polarity[lit.unsigned_abs() as usize - 1];
                if lit > 0 {
                    p.0 = true;
                } else {
                    p.1 = true;
                }
            }
            simplified.push(out);
        }

        for lit in units {
            let idx = lit.unsigned_abs() as usize - 1;
            let value = lit > 0;
            match assigned[idx] {
                Some(fixed) if fixed != value => {
                    proof.add(&[]);
                    return PreprocessOutcome::Unsatisfiable(stats);
                }
                Some(_) => {}
                None => {
                    proof.add(&[lit]);
                    assigned[idx] = Some(value);
                    stats.unit_assignments += 1;
                    changed = true;
                }
            }
        }

        if options.pure_literals && !changed {
            for (idx, &(pos, neg)) in polarity.iter().enumerate() {
                if assigned[idx].is_none() && (pos ^ neg) {
                    let lit = if pos { idx as i32 + 1 } else { -(idx as i32 + 1) };
                    proof.add(&[lit]);
                    assigned[idx] = Some(pos);
                    stats.pure_assignments += 1;
                    changed = true;
                }
            }
        }

        if !changed && options.subsumption && remove_subsumed_clauses(&mut simplified, &mut stats, proof) {
            changed = true;
        }

        if !changed && options.self_subsuming_resolution && self_subsume_clauses(&mut simplified, &mut stats, proof) {
            changed = true;
        }

        if !changed && options.bounded_variable_elimination {
            match eliminate_bounded_variable(&mut simplified, cnf.vars, &assigned, &mut stats, &mut reconstruction, proof) {
                EliminationResult::Changed => changed = true,
                EliminationResult::Unsatisfiable => return PreprocessOutcome::Unsatisfiable(stats),
                EliminationResult::Unchanged => {}
            }
        }

        if !changed && options.blocked_clause_elimination && remove_blocked_clause(&mut simplified, &mut stats, &mut reconstruction, proof)
        {
            changed = true;
        }

        if !changed {
            let mut final_clauses = Vec::new();
            for (idx, value) in assigned.into_iter().enumerate() {
                if let Some(value) = value {
                    let lit = if value { idx as i32 + 1 } else { -(idx as i32 + 1) };
                    final_clauses.push(vec![lit]);
                }
            }
            final_clauses.extend(simplified);
            stats.output_clauses = final_clauses.len() as u64;
            return PreprocessOutcome::Simplified(Cnf { vars: cnf.vars, clauses: final_clauses }, stats, reconstruction);
        }

        clauses = simplified;
    }
}

fn remove_subsumed_clauses(clauses: &mut Vec<Vec<i32>>, stats: &mut PreprocessStats, proof: &mut PreprocessProof<'_>) -> bool {
    let mut removed = vec![false; clauses.len()];
    let mut changed = false;

    for i in 0..clauses.len() {
        if removed[i] {
            continue;
        }
        for j in 0..clauses.len() {
            if i == j || removed[j] {
                continue;
            }
            if clauses[i].len() > clauses[j].len() {
                continue;
            }
            if clauses[i].len() == clauses[j].len() && i > j {
                continue;
            }
            if sorted_subset(&clauses[i], &clauses[j]) {
                removed[j] = true;
                proof.delete(&clauses[j]);
                stats.subsumed_clauses += 1;
                changed = true;
            }
        }
    }

    if changed {
        let mut keep = Vec::with_capacity(clauses.len() - removed.iter().filter(|&&r| r).count());
        for (idx, clause) in clauses.drain(..).enumerate() {
            if !removed[idx] {
                keep.push(clause);
            }
        }
        *clauses = keep;
    }
    changed
}

fn self_subsume_clauses(clauses: &mut [Vec<i32>], stats: &mut PreprocessStats, proof: &mut PreprocessProof<'_>) -> bool {
    let mut changed = false;

    loop {
        let mut changed_this_round = false;
        'scan: for i in 0..clauses.len() {
            for j in 0..clauses.len() {
                if i == j {
                    continue;
                }
                let source = clauses[i].clone();
                for &lit in &source {
                    let neg = -lit;
                    if !contains_lit(&clauses[j], neg) {
                        continue;
                    }
                    if source.iter().filter(|&&other| other != lit).all(|&other| contains_lit(&clauses[j], other)) {
                        let old = clauses[j].clone();
                        clauses[j].retain(|&other| other != neg);
                        proof.add(&clauses[j]);
                        proof.delete(&old);
                        stats.self_subsumed_literals += 1;
                        changed = true;
                        changed_this_round = true;
                        break 'scan;
                    }
                }
            }
        }
        if !changed_this_round {
            break;
        }
    }

    changed
}

enum EliminationResult {
    Unchanged,
    Changed,
    Unsatisfiable,
}

fn eliminate_bounded_variable(
    clauses: &mut Vec<Vec<i32>>,
    vars: usize,
    assigned: &[Option<bool>],
    stats: &mut PreprocessStats,
    reconstruction: &mut Vec<ReconstructionStep>,
    proof: &mut PreprocessProof<'_>,
) -> EliminationResult {
    const BVE_PAIR_LIMIT: usize = 4096;

    for var in 1..=vars as i32 {
        if assigned[var as usize - 1].is_some() {
            continue;
        }

        let pos_lit = var;
        let neg_lit = -var;
        let mut positive_indices = Vec::new();
        let mut negative_indices = Vec::new();
        for (idx, clause) in clauses.iter().enumerate() {
            if contains_lit(clause, pos_lit) {
                positive_indices.push(idx);
            }
            if contains_lit(clause, neg_lit) {
                negative_indices.push(idx);
            }
        }

        if positive_indices.is_empty() || negative_indices.is_empty() {
            continue;
        }
        if positive_indices.len() * negative_indices.len() > BVE_PAIR_LIMIT {
            continue;
        }

        let old_count = positive_indices.len() + negative_indices.len();
        let mut resolvents: Vec<Vec<i32>> = Vec::new();
        for &p_idx in &positive_indices {
            for &n_idx in &negative_indices {
                let Some(resolvent) = resolve_on_var(&clauses[p_idx], pos_lit, &clauses[n_idx]) else {
                    continue;
                };
                if resolvent.is_empty() {
                    proof.add(&[]);
                    return EliminationResult::Unsatisfiable;
                }
                if !resolvents.contains(&resolvent) {
                    resolvents.push(resolvent);
                    if resolvents.len() > old_count {
                        break;
                    }
                }
            }
            if resolvents.len() > old_count {
                break;
            }
        }
        if resolvents.len() > old_count {
            continue;
        }

        let positive = positive_indices.iter().map(|&idx| clauses[idx].clone()).collect::<Vec<_>>();
        let negative = negative_indices.iter().map(|&idx| clauses[idx].clone()).collect::<Vec<_>>();
        for resolvent in &resolvents {
            proof.add(resolvent);
        }
        for clause in positive.iter().chain(&negative) {
            proof.delete(clause);
        }
        let mut remove = vec![false; clauses.len()];
        for idx in positive_indices.into_iter().chain(negative_indices) {
            remove[idx] = true;
        }

        let mut next = Vec::with_capacity(clauses.len() - old_count + resolvents.len());
        for (idx, clause) in clauses.drain(..).enumerate() {
            if !remove[idx] {
                next.push(clause);
            }
        }
        next.extend(resolvents.iter().cloned());
        *clauses = next;

        reconstruction.push(ReconstructionStep::EliminatedVariable { var, positive, negative });
        stats.bve_variables += 1;
        stats.bve_resolvents += resolvents.len() as u64;
        return EliminationResult::Changed;
    }

    EliminationResult::Unchanged
}

fn resolve_on_var(positive: &[i32], pos_lit: i32, negative: &[i32]) -> Option<Vec<i32>> {
    let mut out = Vec::with_capacity(positive.len() + negative.len() - 2);
    for &lit in positive {
        if lit != pos_lit && !push_resolvent_lit(&mut out, lit) {
            return None;
        }
    }
    for &lit in negative {
        if lit != -pos_lit && !push_resolvent_lit(&mut out, lit) {
            return None;
        }
    }
    out.sort_unstable();
    Some(out)
}

fn push_resolvent_lit(out: &mut Vec<i32>, lit: i32) -> bool {
    if out.contains(&-lit) {
        return false;
    }
    if !out.contains(&lit) {
        out.push(lit);
    }
    true
}

fn remove_blocked_clause(
    clauses: &mut Vec<Vec<i32>>,
    stats: &mut PreprocessStats,
    reconstruction: &mut Vec<ReconstructionStep>,
    proof: &mut PreprocessProof<'_>,
) -> bool {
    for idx in 0..clauses.len() {
        let clause = clauses[idx].clone();
        for &lit in &clause {
            if clause_is_blocked(clauses, idx, lit) {
                proof.delete(&clause);
                clauses.remove(idx);
                reconstruction.push(ReconstructionStep::BlockedClause { clause, blocking_lit: lit });
                stats.blocked_clauses += 1;
                return true;
            }
        }
    }
    false
}

fn clause_is_blocked(clauses: &[Vec<i32>], clause_idx: usize, blocking_lit: i32) -> bool {
    clauses.iter().enumerate().all(|(idx, other)| {
        idx == clause_idx || !contains_lit(other, -blocking_lit) || resolvent_is_tautological(&clauses[clause_idx], blocking_lit, other)
    })
}

fn resolvent_is_tautological(clause: &[i32], blocking_lit: i32, other: &[i32]) -> bool {
    clause.iter().any(|&lit| lit != blocking_lit && contains_lit(other, -lit))
}

fn reconstruct_assignment(assignment: &mut [bool], reconstruction: &[ReconstructionStep]) {
    for step in reconstruction.iter().rev() {
        match step {
            ReconstructionStep::EliminatedVariable { var, positive, negative } => {
                let idx = *var as usize - 1;
                let need_true = positive.iter().any(|clause| !clause_satisfied_without_var(clause, *var, assignment));
                let need_false = negative.iter().any(|clause| !clause_satisfied_without_var(clause, *var, assignment));
                if need_true {
                    assignment[idx] = true;
                } else if need_false {
                    assignment[idx] = false;
                }
                debug_assert!(positive.iter().chain(negative).all(|clause| clause_satisfied(clause, assignment)));
            }
            ReconstructionStep::BlockedClause { clause, blocking_lit } => {
                if !clause_satisfied(clause, assignment) {
                    assignment[blocking_lit.unsigned_abs() as usize - 1] = *blocking_lit > 0;
                }
                debug_assert!(clause_satisfied(clause, assignment));
            }
        }
    }
}

fn clause_satisfied_without_var(clause: &[i32], var: i32, assignment: &[bool]) -> bool {
    clause.iter().any(|&lit| lit.unsigned_abs() as i32 != var && lit_satisfied(lit, assignment))
}

fn clause_satisfied(clause: &[i32], assignment: &[bool]) -> bool {
    clause.iter().any(|&lit| lit_satisfied(lit, assignment))
}

fn lit_satisfied(lit: i32, assignment: &[bool]) -> bool {
    let value = assignment[lit.unsigned_abs() as usize - 1];
    if lit > 0 {
        value
    } else {
        !value
    }
}

fn sorted_subset(a: &[i32], b: &[i32]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Less => return false,
        }
    }
    i == a.len()
}

fn contains_lit(clause: &[i32], lit: i32) -> bool {
    clause.binary_search(&lit).is_ok()
}

/// Solve a CNF instance with the public finite-domain linear API.
pub fn solve_cnf_linear(cnf: &Cnf, stop: &AtomicBool) -> SatResult {
    if cnf.clauses.iter().any(Vec::is_empty) {
        return SatResult {
            status: Status::Unsatisfiable,
            assignment: None,
            stats: SolveStats::default(),
            preprocess: PreprocessStats::default(),
        };
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
        Some(values) => SatResult { status: Status::Satisfiable, assignment: Some(values), stats, preprocess: PreprocessStats::default() },
        None if stop.load(Ordering::Relaxed) => {
            SatResult { status: Status::Unknown, assignment: None, stats, preprocess: PreprocessStats::default() }
        }
        None => SatResult { status: Status::Unsatisfiable, assignment: None, stats, preprocess: PreprocessStats::default() },
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
