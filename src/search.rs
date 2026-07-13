//! Search entry points over the Lazy Clause Generation engine ([`crate::lcg`]).
//!
//! Enumeration uses chronological backtracking; optimization asserts a tighter
//! objective bound after each incumbent. Each entry point has an interruptible
//! variant that halts when a shared stop flag is set.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use crate::domains::interval::IntervalPresence;
use crate::expr::Expr;
use crate::ids::{IntervalId, ListId, VarId};
use crate::lcg::clause::ClauseSharing;
use crate::lcg::engine::AssumptionOutcome;
use crate::lcg::lit::{AtomKind, LazyAtomRegistry, Lit, LitOrConst};
use crate::lcg::trail::Cdcl;
use crate::lcg::view::Tri;
use crate::propagator::Inconsistency;
use crate::store::{Solver, Store};

/// A stop flag that is never set; used by the non-interruptible entry points.
static NEVER_STOP: AtomicBool = AtomicBool::new(false);

/// Whether to keep enumerating after a solution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SearchControl {
    /// Keep searching for more solutions.
    Continue,
    /// Stop the search now.
    Stop,
}

/// Counters gathered during a search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SolveStats {
    /// Number of solutions visited.
    pub solutions: u64,
    /// Number of internal (branching) nodes explored.
    pub nodes: u64,
    /// Number of dead ends (propagation failures).
    pub failures: u64,
    /// Total literals across all learned clauses (CDCL search only).
    pub learned_lits: u64,
    /// Learned clauses shortened by CP-aware vivification.
    pub vivified_clauses: u64,
    /// Literals removed by CP-aware vivification.
    pub vivified_lits: u64,
    /// Binary clauses visited during CDCL unit propagation.
    pub binary_clause_visits: u64,
    /// Non-binary watched clauses visited during CDCL unit propagation.
    pub watched_clause_visits: u64,
    /// Non-watch literals scanned while looking for a replacement watch.
    pub watched_literal_scans: u64,
    /// Literals implied by the binary-clause fast path.
    pub binary_implications: u64,
    /// Linear rows retained by the incremental LP relaxation.
    pub lp_rows: u64,
    /// Root and node LP solves attempted.
    pub lp_solves: u64,
    /// Floating dual candidates converted to exact rational bounds.
    pub lp_certified: u64,
    /// Search nodes cut by an exact LP certificate.
    pub lp_prunes: u64,
    /// Non-root subtrees cut by exact LP certificates.
    pub lp_node_prunes: u64,
    /// Whole optimization searches completed from a certified global bound.
    pub lp_global_prunes: u64,
    /// LP solves stopped by their local wall-clock budget.
    pub lp_timeouts: u64,
    /// Complete LP basis refactorizations.
    pub lp_refactorizations: u64,
    /// Wall-clock time spent inside LP solves.
    pub lp_micros: u64,
    /// Certified root bound in the user's objective direction.
    pub lp_root_bound: Option<i64>,
}

/// A Boolean literal over an integer variable encoded as `{0, 1}` or `{-1, 1}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoolLit {
    /// The Boolean finite-domain variable.
    pub var: VarId,
    /// The logical value that satisfies the literal.
    pub value: bool,
}

/// Relation used by a temporary solve assumption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssumptionOp {
    /// `var == value`.
    Eq,
    /// `var != value`.
    Ne,
    /// `var <= value`.
    Le,
    /// `var < value`.
    Lt,
    /// `var >= value`.
    Ge,
    /// `var > value`.
    Gt,
}

/// A temporary integer-domain assumption for one solve call.
///
/// Assumptions are translated to LCG literals after the atom table is built. They
/// are asserted for the current solve only, and learned clauses exported from
/// that solve are guarded by the negated assumption literals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assumption {
    /// Variable constrained by the assumption.
    pub var: VarId,
    /// Relation applied to [`value`](Self::value).
    pub op: AssumptionOp,
    /// Right-hand side value.
    pub value: i32,
}

impl BoolLit {
    /// Literal `var = 1`.
    #[inline]
    pub fn positive(var: VarId) -> Self {
        Self { var, value: true }
    }

    /// Literal `var = 0`.
    #[inline]
    pub fn negative(var: VarId) -> Self {
        Self { var, value: false }
    }
}

impl Assumption {
    /// Construct `var == value`.
    #[inline]
    pub fn eq(var: VarId, value: i32) -> Self {
        Self { var, op: AssumptionOp::Eq, value }
    }
}

/// Errors returned by the native Boolean CNF entry point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoolCnfError {
    /// A clause references a variable whose root domain is not Boolean.
    NonBooleanVar { var: VarId, min: i32, max: i32 },
}

struct BoolCnfHints {
    activity: Vec<f64>,
    phase: Vec<Option<i32>>,
}

/// Complete assignment for list and interval domains.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DomainSolution {
    /// Required item contents of each list.
    pub lists: Vec<Vec<i32>>,
    /// Fixed start of each present interval, or `None` if absent.
    pub interval_starts: Vec<Option<i32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DomainDecision {
    ListMembership {
        list: ListId,
        item: i32,
    },
    IntervalPresence {
        interval: IntervalId,
    },
    /// Durably order a unary-resource pair: first-before-second (left) or
    /// second-before-first (right). `no_overlap` enforces the chosen precedence.
    Order {
        pair: usize,
    },
    IntervalStartSplit {
        interval: IntervalId,
        mid: i32,
    },
}

/// Materialized or symbolic objective evaluated by the CDCL optimizer.
#[derive(Clone, Copy)]
pub(crate) enum Objective<'a> {
    Var(VarId),
    Linear { coeffs: &'a [i64], vars: &'a [VarId] },
    Expr(&'a Expr),
}

#[inline]
fn seeded_cdcl<'a>(
    solver: &'a mut Solver,
    vars: &[VarId],
    stop: &'a AtomicBool,
    seed: u64,
    lazy_atoms: Option<Arc<LazyAtomRegistry>>,
) -> Option<Cdcl<'a>> {
    if stop.load(Ordering::Relaxed) {
        return None;
    }
    let mut cdcl = match lazy_atoms {
        Some(registry) => Cdcl::new_with_registry(solver, vars, registry),
        None => Cdcl::new(solver, vars),
    };
    cdcl.set_stop(stop);
    cdcl.set_seed(seed);
    Some(cdcl)
}

fn assumption_lit(cdcl: &Cdcl<'_>, assumption: Assumption) -> Option<Option<Lit>> {
    let loc = match assumption.op {
        AssumptionOp::Eq => cdcl.atoms.eq(assumption.var, assumption.value),
        AssumptionOp::Ne => match cdcl.atoms.eq(assumption.var, assumption.value) {
            LitOrConst::Lit(lit) => return Some(Some(lit.negate())),
            LitOrConst::True => LitOrConst::False,
            LitOrConst::False => LitOrConst::True,
        },
        AssumptionOp::Ge => cdcl.atoms.ge(assumption.var, assumption.value),
        AssumptionOp::Gt => cdcl.atoms.ge_i64(assumption.var, i64::from(assumption.value) + 1),
        AssumptionOp::Le => match cdcl.atoms.ge_i64(assumption.var, i64::from(assumption.value) + 1) {
            LitOrConst::Lit(lit) => return Some(Some(lit.negate())),
            LitOrConst::True => LitOrConst::False,
            LitOrConst::False => LitOrConst::True,
        },
        AssumptionOp::Lt => match cdcl.atoms.ge(assumption.var, assumption.value) {
            LitOrConst::Lit(lit) => return Some(Some(lit.negate())),
            LitOrConst::True => LitOrConst::False,
            LitOrConst::False => LitOrConst::True,
        },
    };
    match loc {
        LitOrConst::Lit(lit) => Some(Some(lit)),
        LitOrConst::True => Some(None),
        LitOrConst::False => None,
    }
}

fn assumption_cube(cdcl: &Cdcl<'_>, assumptions: &[Assumption]) -> Option<Vec<Lit>> {
    let mut cube = Vec::with_capacity(assumptions.len());
    for &assumption in assumptions {
        if let Some(lit) = assumption_lit(cdcl, assumption)? {
            if !cube.contains(&lit) {
                if cube.contains(&lit.negate()) {
                    return None;
                }
                cube.push(lit);
            }
        }
    }
    Some(cube)
}

/// Enumerate solutions over `vars`; `on_solution` returns [`SearchControl::Stop`]
/// to halt early. Mutates `solver` (posts learnt clauses, leaves end state).
pub fn solve<F>(solver: &mut Solver, vars: &[VarId], on_solution: F) -> SolveStats
where
    F: FnMut(&Solver) -> SearchControl,
{
    solve_interruptible(solver, vars, on_solution, &NEVER_STOP)
}

/// Like [`solve`], but halts as soon as `stop` is set (time limit / Ctrl+C).
pub fn solve_interruptible<F>(solver: &mut Solver, vars: &[VarId], on_solution: F, stop: &AtomicBool) -> SolveStats
where
    F: FnMut(&Solver) -> SearchControl,
{
    solve_interruptible_seeded(solver, vars, on_solution, stop, 0)
}

/// Like [`solve_interruptible`], with reproducible search diversification.
pub(crate) fn solve_interruptible_seeded<F>(solver: &mut Solver, vars: &[VarId], on_solution: F, stop: &AtomicBool, seed: u64) -> SolveStats
where
    F: FnMut(&Solver) -> SearchControl,
{
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, None) else {
        return SolveStats::default();
    };
    cdcl.enumerate(vars, on_solution, stop)
}

/// Find one solution under temporary assumptions, with optional learned-clause
/// exchange, value hints, and a conflict budget.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decide_sat_assuming_seeded(
    solver: &mut Solver,
    vars: &[VarId],
    assumptions: &[Assumption],
    stop: &AtomicBool,
    seed: u64,
    clause_sharing: Option<ClauseSharing>,
    conflict_budget: Option<u64>,
    initial_phase: Vec<Option<i32>>,
    branch_order: Vec<VarId>,
) -> (Option<Vec<i32>>, SolveStats, bool) {
    let lazy_atoms = clause_sharing.as_ref().map(ClauseSharing::lazy_atoms);
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, lazy_atoms) else {
        return (None, SolveStats::default(), false);
    };
    let Some(cube) = assumption_cube(&cdcl, assumptions) else {
        return (None, SolveStats::default(), true);
    };
    if let Some(sharing) = clause_sharing {
        cdcl.set_clause_sharing(sharing);
    }
    cdcl.initial_phase = initial_phase;
    cdcl.branch_order = branch_order;
    cdcl.decide_sat_assuming(vars, &cube, conflict_budget, stop)
}

/// Find one solution under temporary assumptions, or prove that the assumed
/// subproblem is unsatisfiable.
pub fn first_solution_assuming(solver: &mut Solver, vars: &[VarId], assumptions: &[Assumption]) -> (Option<Vec<i32>>, SolveStats, bool) {
    decide_sat_assuming_seeded(solver, vars, assumptions, &NEVER_STOP, 0, None, None, Vec::new(), Vec::new())
}

/// Find one solution over `vars`, or prove UNSAT, with CDCL learning and restarts.
/// This is not an enumeration driver: it does not count or block all solutions.
pub(crate) fn decide_sat_seeded(solver: &mut Solver, vars: &[VarId], stop: &AtomicBool, seed: u64) -> (Option<Vec<i32>>, SolveStats, bool) {
    decide_sat_shared_seeded(solver, vars, stop, seed, None, false)
}

/// Like [`decide_sat_seeded`], cooperating in a CSP portfolio: learned clauses
/// (sound model consequences, never solution-blocking) are exchanged through
/// `clause_sharing`. `fast` picks the shorter restart schedule to diversify
/// workers. Find-one/UNSAT only - never used for enumeration.
pub(crate) fn decide_sat_shared_seeded(
    solver: &mut Solver,
    vars: &[VarId],
    stop: &AtomicBool,
    seed: u64,
    clause_sharing: Option<ClauseSharing>,
    fast: bool,
) -> (Option<Vec<i32>>, SolveStats, bool) {
    let lazy_atoms = clause_sharing.as_ref().map(ClauseSharing::lazy_atoms);
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, lazy_atoms) else {
        return (None, SolveStats::default(), false);
    };
    if let Some(sharing) = clause_sharing {
        cdcl.set_clause_sharing(sharing);
    }
    if fast {
        cdcl.use_fast_restarts();
    }
    cdcl.decide_sat(vars, stop)
}

/// Status of [`solve_under_assumptions`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssumptionResult {
    /// A full assignment over `vars` satisfying every assumption.
    Sat(Vec<i32>),
    /// A subset of `cube` inconsistent with the constraints: an unsat core.
    /// Empty means the constraints are unsatisfiable regardless of the cube.
    Unsat(Vec<BoolLit>),
    /// The search was interrupted by the stop flag before a status was decided.
    Interrupted,
}

/// Solve `solver` over `vars` under the Boolean assumption `cube`, returning a
/// model, an unsat core (a subset of `cube` inconsistent with the constraints),
/// or interruption.
///
/// Each assumption is a [`BoolLit`] over a `{0,1}`/`{-1,1}` variable. An
/// assumption already entailed by the constraints is dropped (it can never be
/// part of a core); one already contradicted forms a singleton core on its own.
/// The returned core is *a* core, not necessarily minimal; minimisation is a
/// separate step.
pub fn solve_under_assumptions(
    solver: &mut Solver,
    vars: &[VarId],
    cube: &[BoolLit],
    stop: &AtomicBool,
) -> Result<AssumptionResult, BoolCnfError> {
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, 0, None) else {
        return Ok(AssumptionResult::Interrupted);
    };
    // Translate the Boolean cube to internal literals, remembering the reverse map.
    let mut cube_lits: Vec<Lit> = Vec::with_capacity(cube.len());
    let mut reverse: Vec<(Lit, BoolLit)> = Vec::with_capacity(cube.len());
    for &bl in cube {
        validate_bool_var(&cdcl, bl.var)?;
        match cdcl.atoms.eq(bl.var, bool_value(&cdcl, bl.var, bl.value)) {
            LitOrConst::True => {} // trivially satisfied: never in a core
            LitOrConst::False => return Ok(AssumptionResult::Unsat(vec![bl])), // impossible alone
            LitOrConst::Lit(l) => {
                if !cube_lits.contains(&l) {
                    cube_lits.push(l);
                    reverse.push((l, bl));
                }
            }
        }
    }
    Ok(match cdcl.solve_under_assumptions(vars, &cube_lits, stop) {
        AssumptionOutcome::Sat(model) => AssumptionResult::Sat(model),
        AssumptionOutcome::Interrupted => AssumptionResult::Interrupted,
        AssumptionOutcome::Unsat(core) => AssumptionResult::Unsat(
            core.into_iter()
                .map(|l| reverse.iter().find(|(cl, _)| *cl == l).map(|&(_, bl)| bl).expect("core literal came from the cube"))
                .collect(),
        ),
    })
}

/// Find one solution or prove UNSAT for native Boolean CNF clauses.
///
/// This path injects clauses directly into the CDCL watched-clause database.
/// It does not post linear clause constraints. `vars` is the set of Boolean
/// variables to branch on, and every literal in `clauses` must reference one
/// of those `{0, 1}` variables.
pub fn solve_bool_cnf_interruptible(
    solver: &mut Solver,
    vars: &[VarId],
    clauses: &[Vec<BoolLit>],
    stop: &AtomicBool,
) -> Result<(Option<Vec<i32>>, SolveStats, bool), BoolCnfError> {
    solve_bool_cnf_seeded(solver, vars, clauses, stop, 0)
}

/// Like [`solve_bool_cnf_interruptible`], with reproducible search
/// diversification.
pub fn solve_bool_cnf_seeded(
    solver: &mut Solver,
    vars: &[VarId],
    clauses: &[Vec<BoolLit>],
    stop: &AtomicBool,
    seed: u64,
) -> Result<(Option<Vec<i32>>, SolveStats, bool), BoolCnfError> {
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, None) else {
        return Ok((None, SolveStats::default(), false));
    };
    let Some(hints) = post_native_bool_clauses(&mut cdcl, vars, clauses)? else {
        let mut stats = SolveStats { failures: cdcl.conflicts, ..SolveStats::default() };
        cdcl.copy_inprocessing_stats(&mut stats);
        return Ok((None, stats, true));
    };
    for (idx, activity) in hints.activity.into_iter().enumerate() {
        cdcl.activity[idx] += activity;
    }
    cdcl.initial_phase = hints.phase;
    cdcl.set_learned_clause_budget(clauses.len().saturating_mul(16).clamp(20_000, 200_000));
    cdcl.use_fast_restarts();
    Ok(cdcl.decide_sat(vars, stop))
}

/// Like [`solve_bool_cnf_seeded`], streaming learned clauses to `proof`.
///
/// The callback receives clauses in the public [`BoolLit`] vocabulary. It is
/// intended for proof-producing CNF frontends; preprocessing proof logging is a
/// separate responsibility.
pub fn solve_bool_cnf_seeded_with_proof<F>(
    solver: &mut Solver,
    vars: &[VarId],
    clauses: &[Vec<BoolLit>],
    stop: &AtomicBool,
    seed: u64,
    mut proof: F,
) -> Result<(Option<Vec<i32>>, SolveStats, bool), BoolCnfError>
where
    F: FnMut(&[BoolLit]),
{
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, None) else {
        return Ok((None, SolveStats::default(), false));
    };
    let lit_map = build_bool_lit_map(&cdcl, vars);
    let mut proof_clause = Vec::new();
    let mut logger = |lits: &[Lit]| {
        proof_clause.clear();
        proof_clause.extend(
            lits.iter()
                .map(|lit| lit_map.get(lit.code() as usize).and_then(|lit| *lit).expect("native SAT proof saw a non-Boolean literal")),
        );
        proof(&proof_clause);
    };
    cdcl.set_proof_logger(&mut logger);

    let Some(hints) = post_native_bool_clauses(&mut cdcl, vars, clauses)? else {
        let mut stats = SolveStats { failures: cdcl.conflicts, ..SolveStats::default() };
        cdcl.copy_inprocessing_stats(&mut stats);
        return Ok((None, stats, true));
    };
    for (idx, activity) in hints.activity.into_iter().enumerate() {
        cdcl.activity[idx] += activity;
    }
    cdcl.initial_phase = hints.phase;
    cdcl.set_learned_clause_budget(clauses.len().saturating_mul(16).clamp(20_000, 200_000));
    cdcl.use_fast_restarts();
    Ok(cdcl.decide_sat(vars, stop))
}

fn post_native_bool_clauses(cdcl: &mut Cdcl<'_>, vars: &[VarId], clauses: &[Vec<BoolLit>]) -> Result<Option<BoolCnfHints>, BoolCnfError> {
    for &var in vars {
        validate_bool_var(cdcl, var)?;
    }

    let mut activity = vec![0.0; cdcl.solver.store.num_vars()];
    let mut positive_score = vec![0.0; cdcl.solver.store.num_vars()];
    let mut negative_score = vec![0.0; cdcl.solver.store.num_vars()];

    for clause in clauses {
        match normalize_bool_clause(cdcl, clause)? {
            None => {}
            Some(lits) if lits.is_empty() => {
                cdcl.log_proof_clause(&[]);
                return Ok(None);
            }
            Some(lits) if lits.len() == 1 => {
                record_bool_clause_hints(cdcl, &lits, &mut activity, &mut positive_score, &mut negative_score);
                if !cdcl.assert_root_lit(lits[0]) {
                    cdcl.log_proof_clause(&[]);
                    return Ok(None);
                }
            }
            Some(lits) => {
                record_bool_clause_hints(cdcl, &lits, &mut activity, &mut positive_score, &mut negative_score);
                cdcl.add_clause(lits, false, 0);
            }
        }
    }
    let mut phase = vec![None; cdcl.solver.store.num_vars()];
    for &var in vars {
        let idx = var.index();
        if positive_score[idx] != 0.0 || negative_score[idx] != 0.0 {
            phase[idx] = Some(bool_value(cdcl, var, positive_score[idx] >= negative_score[idx]));
        }
    }
    Ok(Some(BoolCnfHints { activity, phase }))
}

fn normalize_bool_clause(cdcl: &Cdcl<'_>, clause: &[BoolLit]) -> Result<Option<Vec<Lit>>, BoolCnfError> {
    let mut out = Vec::with_capacity(clause.len());
    for &lit in clause {
        validate_bool_var(cdcl, lit.var)?;
        let l = match cdcl.atoms.eq(lit.var, bool_value(cdcl, lit.var, lit.value)) {
            LitOrConst::True => return Ok(None),
            LitOrConst::False => continue,
            LitOrConst::Lit(l) => l,
        };
        match cdcl.value(l) {
            Tri::True => return Ok(None),
            Tri::False => continue,
            Tri::Unknown => {}
        }
        if out.contains(&l) {
            continue;
        }
        if out.contains(&l.negate()) {
            return Ok(None);
        }
        out.push(l);
    }
    Ok(Some(out))
}

fn build_bool_lit_map(cdcl: &Cdcl<'_>, vars: &[VarId]) -> Vec<Option<BoolLit>> {
    let mut selected = vec![false; cdcl.solver.store.num_vars()];
    for &var in vars {
        selected[var.index()] = true;
    }

    let mut map = vec![None; cdcl.atoms.num_atoms() * 2];
    for atom in 0..cdcl.atoms.num_atoms() as u32 {
        let (var, positive_is_true) = match cdcl.atoms.decode(atom) {
            AtomKind::Ge { var, k } => {
                debug_assert_eq!(k, 1, "Boolean order atoms should be the true endpoint");
                (var, true)
            }
            AtomKind::Eq { var, v } => (var, v == 1),
        };
        if !selected.get(var.index()).copied().unwrap_or(false) {
            continue;
        }
        let positive = Lit::positive(atom);
        map[positive.code() as usize] = Some(BoolLit { var, value: positive_is_true });
        map[positive.negate().code() as usize] = Some(BoolLit { var, value: !positive_is_true });
    }
    map
}

fn validate_bool_var(cdcl: &Cdcl<'_>, var: VarId) -> Result<(), BoolCnfError> {
    let (min, max) = cdcl.atoms.var_span(var);
    let is_zero_one = min == 0 && max == 1;
    let is_signed = cdcl.atoms.is_sign(var);
    if !is_zero_one && !is_signed {
        return Err(BoolCnfError::NonBooleanVar { var, min, max });
    }
    Ok(())
}

fn bool_value(cdcl: &Cdcl<'_>, var: VarId, logical_true: bool) -> i32 {
    if logical_true {
        1
    } else if cdcl.solver.store.contains(var, 0) {
        0
    } else {
        -1
    }
}

fn record_bool_clause_hints(cdcl: &Cdcl<'_>, lits: &[Lit], activity: &mut [f64], positive_score: &mut [f64], negative_score: &mut [f64]) {
    let weight = 2.0_f64.powi(-(lits.len().min(32) as i32));
    for &lit in lits {
        let var = cdcl.atoms.var_of(lit.atom());
        let idx = var.index();
        activity[idx] += weight;
        if lit_is_logically_positive(cdcl, lit) {
            positive_score[idx] += weight;
        } else {
            negative_score[idx] += weight;
        }
    }
}

fn lit_is_logically_positive(cdcl: &Cdcl<'_>, lit: Lit) -> bool {
    match cdcl.atoms.decode(lit.atom()) {
        AtomKind::Ge { .. } => lit.is_positive(),
        AtomKind::Eq { v, .. } => (v == 1) == lit.is_positive(),
    }
}

/// Find one solution by non-learning chronological DFS, or report exhaustion.
/// Returns `(solution, stats, complete)`, where `complete` is `true` unless the
/// search was interrupted. Run as a diversified portfolio worker alongside the
/// learning [`decide_sat_shared_seeded`]: plain DFS wins on instances (e.g.
/// highly symmetric ones, such as graph colouring) where clause learning thrashes.
pub(crate) fn find_one_seeded(solver: &mut Solver, vars: &[VarId], stop: &AtomicBool, seed: u64) -> (Option<Vec<i32>>, SolveStats, bool) {
    let mut found = None;
    let stats = solve_interruptible_seeded(
        solver,
        vars,
        |s| {
            found = Some(vars.iter().map(|&v| s.store.value(v)).collect::<Vec<_>>());
            SearchControl::Stop
        },
        stop,
        seed,
    );
    let complete = !stop.load(Ordering::Relaxed);
    (found, stats, complete)
}

/// Find the first solution, returning the values of `vars`.
pub fn first_solution(solver: &mut Solver, vars: &[VarId]) -> Option<Vec<i32>> {
    let mut found = None;
    solve(solver, vars, |s| {
        found = Some(vars.iter().map(|&v| s.store.value(v)).collect());
        SearchControl::Stop
    });
    found
}

/// Count all solutions over `vars`.
pub fn count_solutions(solver: &mut Solver, vars: &[VarId]) -> u64 {
    solve(solver, vars, |_| SearchControl::Continue).solutions
}

/// Enumerate complete assignments over all list and interval domains.
///
/// This is a chronological DFS with propagation. It is intentionally separate
/// from the LCG drivers until domain facts have explanations.
pub fn solve_domains<F>(solver: &mut Solver, mut on_solution: F) -> SolveStats
where
    F: FnMut(&Solver, &DomainSolution) -> SearchControl,
{
    solve_domains_interruptible(solver, &mut on_solution, &NEVER_STOP).0
}

/// Like [`solve_domains`], but halts when `stop` is set. The Boolean is `true`
/// only when the domain search space was fully exhausted.
pub fn solve_domains_interruptible<F>(solver: &mut Solver, mut on_solution: F, stop: &AtomicBool) -> (SolveStats, bool)
where
    F: FnMut(&Solver, &DomainSolution) -> SearchControl,
{
    solver.enqueue_all();
    let mut stats = SolveStats::default();
    let complete = domain_dfs(solver, &mut stats, &mut on_solution, stop) == DomainExit::Exhausted && !stop.load(Ordering::Relaxed);
    (stats, complete)
}

/// Find one complete domain assignment.
pub fn first_domain_solution(solver: &mut Solver) -> Option<DomainSolution> {
    let mut found = None;
    solve_domains(solver, |_, solution| {
        found = Some(solution.clone());
        SearchControl::Stop
    });
    found
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DomainExit {
    Exhausted,
    Stopped,
}

fn domain_dfs<F>(solver: &mut Solver, stats: &mut SolveStats, on_solution: &mut F, stop: &AtomicBool) -> DomainExit
where
    F: FnMut(&Solver, &DomainSolution) -> SearchControl,
{
    if stop.load(Ordering::Relaxed) {
        return DomainExit::Stopped;
    }
    if solver.propagate().is_err() {
        stats.failures += 1;
        return DomainExit::Exhausted;
    }

    let Some(decision) = choose_domain_decision(solver) else {
        stats.solutions += 1;
        let solution = collect_domain_solution(solver);
        return if matches!(on_solution(solver, &solution), SearchControl::Stop) { DomainExit::Stopped } else { DomainExit::Exhausted };
    };

    stats.nodes += 1;
    for left in [true, false] {
        solver.store.push_level();
        let branch = apply_domain_decision(solver, decision, left);
        let exit = if branch.is_ok() {
            domain_dfs(solver, stats, on_solution, stop)
        } else {
            stats.failures += 1;
            DomainExit::Exhausted
        };
        solver.store.pop_level();
        if exit == DomainExit::Stopped {
            return DomainExit::Stopped;
        }
    }
    DomainExit::Exhausted
}

/// Whether two intervals can still strictly overlap (share an instant). Requires
/// positive durations (a zero-duration interval never strictly overlaps) and
/// start windows that admit a common covered instant.
fn can_strictly_overlap(store: &Store, a: IntervalId, b: IntervalId) -> bool {
    if store.interval_duration(a) == 0 || store.interval_duration(b) == 0 {
        return false;
    }
    let lo = store.interval_start_min(a).max(store.interval_start_min(b));
    // `interval_end_max` saturates, unlike a hand-rolled start_max + duration.
    let hi = store.interval_end_max(a).min(store.interval_end_max(b));
    lo < hi
}

fn choose_domain_decision(solver: &Solver) -> Option<DomainDecision> {
    let store = &solver.store;
    for i in 0..store.num_intervals() {
        let interval = IntervalId(i as u32);
        if store.interval_presence(interval) == IntervalPresence::Optional {
            return Some(DomainDecision::IntervalPresence { interval });
        }
    }

    // Disjunctive scheduling: durably order an undecided unary-resource pair only
    // when the two present intervals can still *strictly* overlap (a real
    // conflict). Skipping non-overlapping pairs (including zero-duration ones,
    // which never strictly overlap) keeps enumeration duplicate-free.
    for pair in 0..store.disjunctive_pair_count() {
        if store.disjunctive_order(pair) != 0 {
            continue;
        }
        let (a, b) = store.disjunctive_pair(pair);
        if store.interval_presence(a) == IntervalPresence::Present
            && store.interval_presence(b) == IntervalPresence::Present
            && can_strictly_overlap(store, a, b)
        {
            return Some(DomainDecision::Order { pair });
        }
    }

    for i in 0..store.num_intervals() {
        let interval = IntervalId(i as u32);
        if store.interval_presence(interval) != IntervalPresence::Absent {
            let lo = store.interval_start_min(interval);
            let hi = store.interval_start_max(interval);
            if lo < hi {
                return Some(DomainDecision::IntervalStartSplit { interval, mid: lo + (hi - lo) / 2 });
            }
        }
    }

    for i in 0..store.num_lists() {
        let list = ListId(i as u32);
        for &item in store.list_universe(list) {
            if store.list_possible(list, item) && !store.list_required(list, item) {
                return Some(DomainDecision::ListMembership { list, item });
            }
        }
    }

    None
}

fn apply_domain_decision(solver: &mut Solver, decision: DomainDecision, left: bool) -> Result<(), Inconsistency> {
    match decision {
        DomainDecision::ListMembership { list, item } => {
            if left {
                solver.store.require_list_item(list, item)?;
            } else {
                solver.store.forbid_list_item(list, item)?;
            }
        }
        DomainDecision::IntervalPresence { interval } => {
            if left {
                solver.store.require_interval_presence(interval)?;
            } else {
                solver.store.forbid_interval_presence(interval)?;
            }
        }
        DomainDecision::Order { pair } => {
            // Durable, trailed order decision; no_overlap enforces the precedence
            // (and is woken by the store on this change).
            solver.store.set_disjunctive_order(pair, if left { 1 } else { 2 })?;
        }
        DomainDecision::IntervalStartSplit { interval, mid } => {
            if left {
                solver.store.set_interval_start_max(interval, mid)?;
            } else {
                solver.store.set_interval_start_min(interval, mid.saturating_add(1))?;
            }
        }
    }
    Ok(())
}

fn collect_domain_solution(solver: &Solver) -> DomainSolution {
    let store = &solver.store;
    let mut lists = Vec::with_capacity(store.num_lists());
    for i in 0..store.num_lists() {
        let list = ListId(i as u32);
        lists.push(store.list_universe(list).iter().copied().filter(|&item| store.list_required(list, item)).collect());
    }

    let mut interval_starts = Vec::with_capacity(store.num_intervals());
    for i in 0..store.num_intervals() {
        let interval = IntervalId(i as u32);
        let start = match store.interval_presence(interval) {
            IntervalPresence::Absent => None,
            IntervalPresence::Optional | IntervalPresence::Present => Some(store.interval_start_min(interval)),
        };
        interval_starts.push(start);
    }

    DomainSolution { lists, interval_starts }
}

/// Optimise `obj` by CDCL branch-and-bound; `on_improve` fires per improving
/// bound. Returns best `(assignment, objective value)` and stats. `obj` must be
/// among `vars`.
pub fn optimize_with(
    solver: &mut Solver,
    vars: &[VarId],
    obj: VarId,
    minimizing: bool,
    stop: &AtomicBool,
    on_improve: impl FnMut(i32),
) -> (Option<(Vec<i32>, i32)>, SolveStats) {
    let mut on_improve = on_improve;
    let (best, stats, _) = optimize_seeded(
        solver,
        vars,
        Objective::Var(obj),
        minimizing,
        stop,
        0,
        None,
        None,
        &[],
        None,
        Vec::new(),
        Vec::new(),
        |value, _| on_improve(value as i32),
    );
    (best.map(|(solution, value)| (solution, value as i32)), stats)
}

/// Optimise with a reproducible seed and an optional shared incumbent.
/// The final Boolean is `true` only when search completed rather than stopping.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_seeded(
    solver: &mut Solver,
    vars: &[VarId],
    objective: Objective<'_>,
    minimizing: bool,
    stop: &AtomicBool,
    seed: u64,
    shared_bound: Option<&AtomicI64>,
    clause_sharing: Option<ClauseSharing>,
    cube: &[Lit],
    conflict_budget: Option<u64>,
    initial_phase: Vec<Option<i32>>,
    branch_order: Vec<VarId>,
    on_improve: impl FnMut(i64, &[i32]),
) -> (Option<(Vec<i32>, i64)>, SolveStats, bool) {
    let lazy_atoms = clause_sharing.as_ref().map(ClauseSharing::lazy_atoms);
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, lazy_atoms) else {
        return (None, SolveStats::default(), false);
    };
    if let Some(sharing) = clause_sharing {
        cdcl.set_clause_sharing(sharing);
    }
    cdcl.initial_phase = initial_phase;
    cdcl.branch_order = branch_order;
    cdcl.optimize(vars, objective, minimizing, stop, shared_bound, cube, conflict_budget, on_improve)
}

/// Optimise with temporary assumptions. This keeps the public callsite in terms
/// of integer variables rather than LCG literals.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_assuming_seeded(
    solver: &mut Solver,
    vars: &[VarId],
    assumptions: &[Assumption],
    objective: Objective<'_>,
    minimizing: bool,
    stop: &AtomicBool,
    seed: u64,
    shared_bound: Option<&AtomicI64>,
    clause_sharing: Option<ClauseSharing>,
    conflict_budget: Option<u64>,
    initial_phase: Vec<Option<i32>>,
    branch_order: Vec<VarId>,
    on_improve: impl FnMut(i64, &[i32]),
) -> (Option<(Vec<i32>, i64)>, SolveStats, bool) {
    let lazy_atoms = clause_sharing.as_ref().map(ClauseSharing::lazy_atoms);
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, lazy_atoms) else {
        return (None, SolveStats::default(), false);
    };
    let Some(cube) = assumption_cube(&cdcl, assumptions) else {
        return (None, SolveStats::default(), true);
    };
    if let Some(sharing) = clause_sharing {
        cdcl.set_clause_sharing(sharing);
    }
    cdcl.initial_phase = initial_phase;
    cdcl.branch_order = branch_order;
    cdcl.optimize(vars, objective, minimizing, stop, shared_bound, &cube, conflict_budget, on_improve)
}

/// Optimise an objective variable under temporary assumptions.
pub fn optimize_var_assuming(
    solver: &mut Solver,
    vars: &[VarId],
    assumptions: &[Assumption],
    obj: VarId,
    minimizing: bool,
    stop: &AtomicBool,
    on_improve: impl FnMut(i64, &[i32]),
) -> (Option<(Vec<i32>, i64)>, SolveStats, bool) {
    optimize_assuming_seeded(
        solver,
        vars,
        assumptions,
        Objective::Var(obj),
        minimizing,
        stop,
        0,
        None,
        None,
        None,
        Vec::new(),
        Vec::new(),
        on_improve,
    )
}

/// Pick a binary split for one root cube.
pub(crate) fn split_cube_seeded(
    solver: &mut Solver,
    vars: &[VarId],
    cube: &[Lit],
    stop: &AtomicBool,
    seed: u64,
    lazy_atoms: Option<Arc<LazyAtomRegistry>>,
) -> Option<Lit> {
    let mut cdcl = seeded_cdcl(solver, vars, stop, seed, lazy_atoms)?;
    cdcl.split_cube(vars, cube)
}

/// Find one solution under an optimistic objective bound.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probe_seeded(
    solver: &mut Solver,
    vars: &[VarId],
    obj: VarId,
    minimizing: bool,
    target: i32,
    stop: &AtomicBool,
    seed: u64,
    clause_sharing: Option<ClauseSharing>,
) -> (Option<(Vec<i32>, i32)>, SolveStats, bool) {
    let lazy_atoms = clause_sharing.as_ref().map(ClauseSharing::lazy_atoms);
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, lazy_atoms) else {
        return (None, SolveStats::default(), false);
    };
    if let Some(sharing) = clause_sharing {
        cdcl.set_clause_sharing(sharing);
    }
    cdcl.probe(vars, obj, minimizing, target, stop)
}

/// Minimise `obj`. Returns the best `(assignment of vars, obj value)`.
pub fn minimize(solver: &mut Solver, vars: &[VarId], obj: VarId) -> Option<(Vec<i32>, i32)> {
    optimize_with(solver, vars, obj, true, &NEVER_STOP, |_| {}).0
}

/// Maximise `obj`. Returns the best `(assignment of vars, obj value)`.
pub fn maximize(solver: &mut Solver, vars: &[VarId], obj: VarId) -> Option<(Vec<i32>, i32)> {
    optimize_with(solver, vars, obj, false, &NEVER_STOP, |_| {}).0
}
