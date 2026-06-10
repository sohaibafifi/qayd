//! Search entry points over the Lazy Clause Generation engine ([`crate::lcg`]).
//!
//! Enumeration uses chronological backtracking; optimization asserts a tighter
//! objective bound after each incumbent. Each entry point has an interruptible
//! variant that halts when a shared stop flag is set.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use crate::expr::Expr;
use crate::ids::{PropId, VarId};
use crate::lcg::clause::ClauseSharing;
use crate::lcg::lit::{LazyAtomRegistry, Lit};
use crate::lcg::trail::Cdcl;
use crate::store::Solver;

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

/// Find one solution over `vars`, or prove UNSAT, with CDCL learning and restarts.
/// This is not an enumeration driver: it does not count or block all solutions.
pub(crate) fn decide_sat_seeded(solver: &mut Solver, vars: &[VarId], stop: &AtomicBool, seed: u64) -> (Option<Vec<i32>>, SolveStats, bool) {
    decide_sat_shared_seeded(solver, vars, stop, seed, None, false)
}

/// Like [`decide_sat_seeded`], cooperating in a CSP portfolio: learned clauses
/// (sound model consequences, never solution-blocking) are exchanged through
/// `clause_sharing`. `fast` picks the shorter restart schedule to diversify
/// workers. Find-one/UNSAT only — never used for enumeration.
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

/// Extract a root-refutation unsat core: the propagators (constraints) whose
/// inferences jointly refute the model by root propagation alone. `None` when
/// the root is propagation-consistent (an unsat instance that needs search to
/// refute is out of scope for this single-pass extractor). The returned
/// [`PropId`]s are in first-seen order; map them to source constraints via the
/// frontend's propagator→constraint grouping.
pub(crate) fn root_unsat_core(solver: &mut Solver, vars: &[VarId]) -> Option<Vec<PropId>> {
    let mut cdcl = Cdcl::new(solver, vars);
    cdcl.root_core()
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
    let (best, stats, _) =
        optimize_seeded(solver, vars, Objective::Var(obj), minimizing, stop, 0, None, None, &[], None, |value, _| on_improve(value as i32));
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
    on_improve: impl FnMut(i64, &[i32]),
) -> (Option<(Vec<i32>, i64)>, SolveStats, bool) {
    let lazy_atoms = clause_sharing.as_ref().map(ClauseSharing::lazy_atoms);
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, lazy_atoms) else {
        return (None, SolveStats::default(), false);
    };
    if let Some(sharing) = clause_sharing {
        cdcl.set_clause_sharing(sharing);
    }
    cdcl.optimize(vars, objective, minimizing, stop, shared_bound, cube, conflict_budget, on_improve)
}

/// Incumbent-oriented optimization for fast COP mode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_fast_seeded(
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
    on_improve: impl FnMut(i64, &[i32]),
) -> (Option<(Vec<i32>, i64)>, SolveStats, bool) {
    let lazy_atoms = clause_sharing.as_ref().map(ClauseSharing::lazy_atoms);
    let Some(mut cdcl) = seeded_cdcl(solver, vars, stop, seed, lazy_atoms) else {
        return (None, SolveStats::default(), false);
    };
    if let Some(sharing) = clause_sharing {
        cdcl.set_clause_sharing(sharing);
    }
    cdcl.use_fast_restarts();
    cdcl.optimize_fast(vars, objective, minimizing, stop, shared_bound, cube, conflict_budget, on_improve)
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
