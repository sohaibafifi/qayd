//! Depth-first search with backtracking.
//!
//! Each node tries `var = val` (left) then `var != val` (right), propagating to
//! a fixpoint after each and backtracking on failure. Variable selection is
//! `dom/wdeg`; value selection is min-value. The COP optimiser ([`optimize_with`])
//! is in-search branch-and-bound with geometric restarts. All entry points have
//! an interruptible variant that halts when a shared stop flag is set (used for
//! the CLI time limit and Ctrl+C).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::VarId;
use crate::propagator::Inconsistency;
use crate::store::Solver;

/// A stop flag that is never set — used by the non-interruptible entry points.
static NEVER_STOP: AtomicBool = AtomicBool::new(false);

#[inline]
fn stopped(flag: &AtomicBool) -> bool {
    flag.load(Ordering::Relaxed)
}

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
}

/// Enumerate solutions over `vars` by DFS. `on_solution` is invoked for every
/// full assignment; return [`SearchControl::Stop`] to halt early. The solver is
/// restored to its pre-call state on return.
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
    mut on_solution: F,
    stop: &AtomicBool,
) -> SolveStats
where
    F: FnMut(&Solver) -> SearchControl,
{
    let mut stats = SolveStats::default();
    solver.store.push_level();
    solver.enqueue_all();
    match solver.propagate() {
        Ok(()) => {
            dfs(solver, vars, &mut stats, &mut on_solution, stop);
        }
        Err(Inconsistency) => {
            stats.failures += 1;
        }
    }
    solver.store.pop_level();
    stats
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

/// In-search branch-and-bound: a single DFS that keeps the best objective value
/// found so far as a (non-trailed) incumbent and, at every node, prunes the
/// objective to demand a strictly better value. Unlike restart B&B this never
/// re-searches an explored prefix. `on_improve` fires for each improving bound.
/// Returns the best `(assignment, objective value)` and search stats. `obj`
/// must be among `vars` (so search fixes it).
pub fn optimize_with(
    solver: &mut Solver,
    vars: &[VarId],
    obj: VarId,
    minimizing: bool,
    stop: &AtomicBool,
    mut on_improve: impl FnMut(i32),
) -> (Option<(Vec<i32>, i32)>, SolveStats) {
    let mut stats = SolveStats::default();
    let mut incumbent: Option<i32> = None;
    let mut best: Option<(Vec<i32>, i32)> = None;

    // Geometric restarts. dom/wdeg weights accumulate across runs (failures bump
    // them), so each restart searches differently and escapes early bad choices.
    let mut limit: u64 = 128;
    solver.store.push_level();
    loop {
        solver.store.clear_queue();
        solver.enqueue_all();
        let mut ctx = Bnb {
            run_failures: 0,
            limit,
            aborted: false,
        };
        bnb(
            solver,
            vars,
            obj,
            minimizing,
            &mut incumbent,
            &mut best,
            &mut stats,
            &mut ctx,
            stop,
            &mut on_improve,
        );
        if stopped(stop) {
            break; // interrupted / timed out: return the best found so far
        }
        if !ctx.aborted {
            break; // a full traversal completed: best is optimal
        }
        limit = limit.saturating_mul(2);
    }
    solver.store.pop_level();

    (best, stats)
}

/// Per-run restart state for [`bnb`].
struct Bnb {
    run_failures: u64,
    limit: u64,
    aborted: bool,
}

#[allow(clippy::too_many_arguments)]
fn bnb<F: FnMut(i32)>(
    solver: &mut Solver,
    vars: &[VarId],
    obj: VarId,
    minimizing: bool,
    incumbent: &mut Option<i32>,
    best: &mut Option<(Vec<i32>, i32)>,
    stats: &mut SolveStats,
    ctx: &mut Bnb,
    stop: &AtomicBool,
    on_improve: &mut F,
) {
    if ctx.aborted {
        return;
    }
    if stopped(stop) {
        ctx.aborted = true; // interrupted / timed out
        return;
    }
    if ctx.run_failures >= ctx.limit {
        ctx.aborted = true; // restart budget exhausted
        return;
    }

    // Reach the node's fixpoint (drains the decision that led here, or the
    // initial enqueue at the root).
    if solver.propagate().is_err() {
        stats.failures += 1;
        ctx.run_failures += 1;
        return;
    }
    // Enforce the incumbent bound here, then re-propagate. The incumbent is not
    // trailed, so re-applying it at each node makes the bound global.
    if let Some(inc) = *incumbent {
        let bounded = if minimizing {
            solver.store.remove_above(obj, inc - 1)
        } else {
            solver.store.remove_below(obj, inc + 1)
        };
        if bounded.is_err() || solver.propagate().is_err() {
            stats.failures += 1;
            ctx.run_failures += 1;
            return;
        }
    }

    let var = match select(solver, vars) {
        None => {
            // Full assignment: a new (strictly better) incumbent.
            let value = solver.store.value(obj);
            *incumbent = Some(value);
            *best = Some((vars.iter().map(|&v| solver.store.value(v)).collect(), value));
            stats.solutions += 1;
            on_improve(value);
            return;
        }
        Some(v) => v,
    };
    let val = solver.store.min(var);
    stats.nodes += 1;

    for apply in [Decision::Eq, Decision::Ne] {
        let level = solver.store.level();
        solver.store.push_level();
        let ok = match apply {
            Decision::Eq => solver.store.fix(var, val),
            Decision::Ne => solver.store.remove(var, val),
        };
        if ok.is_ok() {
            bnb(
                solver, vars, obj, minimizing, incumbent, best, stats, ctx, stop, on_improve,
            );
        } else {
            stats.failures += 1;
            ctx.run_failures += 1;
        }
        solver.store.pop_level();
        debug_assert_eq!(solver.store.level(), level, "trail level mismatch in bnb");
        if ctx.aborted {
            return;
        }
    }
}

#[derive(Clone, Copy)]
enum Decision {
    Eq,
    Ne,
}

/// Minimise `obj`. Returns the best `(assignment of vars, obj value)`.
pub fn minimize(solver: &mut Solver, vars: &[VarId], obj: VarId) -> Option<(Vec<i32>, i32)> {
    optimize_with(solver, vars, obj, true, &NEVER_STOP, |_| {}).0
}

/// Maximise `obj`. Returns the best `(assignment of vars, obj value)`.
pub fn maximize(solver: &mut Solver, vars: &[VarId], obj: VarId) -> Option<(Vec<i32>, i32)> {
    optimize_with(solver, vars, obj, false, &NEVER_STOP, |_| {}).0
}

/// `dom/wdeg`: pick the unfixed variable minimising domain-size / weighted
/// degree. Falls back to first-fail before any constraint has failed (all
/// weights 1, so the ratio is just the domain size).
fn select(solver: &Solver, vars: &[VarId]) -> Option<VarId> {
    let weights = solver.weights();
    let mut best: Option<VarId> = None;
    let mut best_score = f64::INFINITY;
    for &v in vars {
        let size = solver.store.size(v);
        if size > 1 {
            let wdeg = solver.store.var_weight(v, weights);
            let score = size as f64 / wdeg as f64;
            if score < best_score {
                best_score = score;
                best = Some(v);
            }
        }
    }
    best
}

fn dfs<F>(
    solver: &mut Solver,
    vars: &[VarId],
    stats: &mut SolveStats,
    on_solution: &mut F,
    stop: &AtomicBool,
) -> SearchControl
where
    F: FnMut(&Solver) -> SearchControl,
{
    if stopped(stop) {
        return SearchControl::Stop;
    }
    let var = match select(solver, vars) {
        None => {
            // No unfixed variable left: a full assignment.
            stats.solutions += 1;
            return on_solution(solver);
        }
        Some(v) => v,
    };
    let val = solver.store.min(var);
    stats.nodes += 1;

    // Left branch: var = val.
    if let SearchControl::Stop = branch(solver, vars, stats, on_solution, stop, |s| {
        s.store.fix(var, val)
    }) {
        return SearchControl::Stop;
    }
    // Right branch: var != val.
    if let SearchControl::Stop = branch(solver, vars, stats, on_solution, stop, |s| {
        s.store.remove(var, val)
    }) {
        return SearchControl::Stop;
    }

    SearchControl::Continue
}

/// Run one branch: open a level, apply `decision`, propagate, recurse, then
/// restore the level. Trail depth is asserted to round-trip in debug builds.
fn branch<F, D>(
    solver: &mut Solver,
    vars: &[VarId],
    stats: &mut SolveStats,
    on_solution: &mut F,
    stop: &AtomicBool,
    decision: D,
) -> SearchControl
where
    F: FnMut(&Solver) -> SearchControl,
    D: FnOnce(&mut Solver) -> Result<(), Inconsistency>,
{
    let level = solver.store.level();
    solver.store.push_level();

    let outcome = match decision(solver) {
        Ok(()) => solver.propagate(),
        Err(e) => Err(e),
    };
    let ctrl = match outcome {
        Ok(()) => dfs(solver, vars, stats, on_solution, stop),
        Err(Inconsistency) => {
            stats.failures += 1;
            SearchControl::Continue
        }
    };

    solver.store.pop_level();
    debug_assert_eq!(
        solver.store.level(),
        level,
        "trail level mismatch after branch"
    );
    ctrl
}
