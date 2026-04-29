//! Search entry points over the Lazy Clause Generation engine ([`crate::lcg`]).
//!
//! Enumeration adds a blocking clause per solution; optimization asserts a
//! tighter objective bound after each incumbent. Each entry point has an
//! interruptible variant that halts when a shared stop flag is set.

use std::sync::atomic::AtomicBool;

use crate::ids::VarId;
use crate::lcg::trail::Cdcl;
use crate::store::Solver;

/// A stop flag that is never set — used by the non-interruptible entry points.
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
pub fn solve_interruptible<F>(
    solver: &mut Solver,
    vars: &[VarId],
    on_solution: F,
    stop: &AtomicBool,
) -> SolveStats
where
    F: FnMut(&Solver) -> SearchControl,
{
    let mut cdcl = Cdcl::new(solver);
    cdcl.set_stop(stop);
    cdcl.enumerate(vars, on_solution, stop)
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
    let mut cdcl = Cdcl::new(solver);
    cdcl.set_stop(stop);
    cdcl.optimize(vars, obj, minimizing, stop, on_improve)
}

/// Minimise `obj`. Returns the best `(assignment of vars, obj value)`.
pub fn minimize(solver: &mut Solver, vars: &[VarId], obj: VarId) -> Option<(Vec<i32>, i32)> {
    optimize_with(solver, vars, obj, true, &NEVER_STOP, |_| {}).0
}

/// Maximise `obj`. Returns the best `(assignment of vars, obj value)`.
pub fn maximize(solver: &mut Solver, vars: &[VarId], obj: VarId) -> Option<(Vec<i32>, i32)> {
    optimize_with(solver, vars, obj, false, &NEVER_STOP, |_| {}).0
}
