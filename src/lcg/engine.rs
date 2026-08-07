//! CDCL search drivers.
//!
//! [`Cdcl::enumerate`] is the CSP driver: chronological-backtracking DFS, no
//! clause learning (which is unsound for enumerating every solution).
//! [`Cdcl::optimize`] is the COP driver: CDCL branch-and-bound with restarts.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use crate::constraints::linear::clamp_i32;
use crate::expr;
use crate::expr::Expr;
use crate::ids::{PropId, VarId};
use crate::lcg::lit::{Lit, LitOrConst};
use crate::lcg::trail::{Cdcl, Reason};
use crate::lcg::view::Tri;
use crate::propagator::{Event, Inconsistency, Propagator};
use crate::search::{Objective, SearchControl, SolveStats};
use crate::store::{Premise, Solver, Store};

/// One in every `REPHASE_PERIOD` restart segments rephases (ignores saved phases
/// and dives with a fresh polarity). The other segments keep saved phases, so
/// convergence is preserved and diversification stays a minority of the effort.
const REPHASE_PERIOD: u64 = 4;

use crate::mix64;

fn strict_bound(incumbent: i64, minimizing: bool) -> Option<i64> {
    if minimizing {
        incumbent.checked_sub(1)
    } else {
        incumbent.checked_add(1)
    }
}

impl Objective<'_> {
    fn value(self, solver: &Solver) -> i64 {
        match self {
            Self::Var(obj) => solver.store.value(obj) as i64,
            // Accumulate in i128 so large coeffs/many terms can't overflow, then
            // clamp: a wrapped objective would post a wrong bound (lost optimum).
            Self::Linear { coeffs, vars } => coeffs
                .iter()
                .zip(vars)
                .map(|(&coeff, &var)| coeff as i128 * solver.store.value(var) as i128)
                .sum::<i128>()
                .clamp(i64::MIN as i128, i64::MAX as i128) as i64,
            Self::Expr(expr) => expr.eval(&|var| solver.store.value(var) as i64).expect("objective expression is undefined at a solution"),
        }
    }
}

struct ObjectiveImpact {
    coeffs: Vec<u64>,
}

impl ObjectiveImpact {
    fn new(objective: Objective<'_>, nvars: usize, minimizing: bool) -> Option<Self> {
        let mut impact = vec![0u64; nvars];
        let mut signed = vec![0i64; nvars];
        match objective {
            Objective::Var(var) => {
                impact[var.index()] = 0;
                signed[var.index()] = if minimizing { 1 } else { -1 };
            }
            Objective::Linear { coeffs, vars } => {
                for (&coeff, &var) in coeffs.iter().zip(vars) {
                    impact[var.index()] = impact[var.index()].saturating_add(coeff.unsigned_abs());
                    let directed = if minimizing { coeff } else { coeff.saturating_neg() };
                    signed[var.index()] = signed[var.index()].saturating_add(directed);
                }
            }
            Objective::Expr(_) => return None,
        }
        (impact.iter().any(|&coeff| coeff > 0) || signed.iter().any(|&coeff| coeff != 0)).then_some(Self { coeffs: impact })
    }

    fn score(&self, solver: &Solver, var: VarId) -> u128 {
        let coeff = self.coeffs[var.index()] as u128;
        if coeff == 0 {
            return 0;
        }
        let width = i64::from(solver.store.max(var)) - i64::from(solver.store.min(var));
        coeff.saturating_mul(width.max(0) as u128)
    }
}

/// Outcome of [`Cdcl::solve_under_assumptions`].
pub(crate) enum AssumptionOutcome {
    /// A full assignment over `vars` satisfying the root and the whole cube.
    Sat(Vec<i32>),
    /// A subset of the cube inconsistent with the root (an unsat core). Empty
    /// means the root alone is unsatisfiable, independent of the cube.
    Unsat(Vec<Lit>),
    /// The stop flag fired before a status was decided.
    Interrupted,
}

impl Cdcl<'_> {
    /// Branching heuristic: unfixed variable minimising `size / (wdeg + activity)`
    /// (dom/wdeg combined with VSIDS activity), or `None` if all fixed.
    fn select_var(&self, vars: &[VarId], objective: Option<&ObjectiveImpact>) -> Option<VarId> {
        for &v in &self.branch_order {
            if vars.contains(&v) && self.solver.store.size(v) > 1 {
                return Some(v);
            }
        }
        let mut best: Option<VarId> = None;
        let mut best_score = f64::INFINITY;
        let mut best_objective: Option<VarId> = None;
        let mut best_objective_impact = 0u128;
        let mut best_objective_score = f64::INFINITY;
        for &v in vars {
            let size = self.solver.store.size(v);
            if size > 1 {
                let wdeg = self.solver.store.var_weight_cached(v) as f64;
                let score = size as f64 / (wdeg + self.activity[v.index()]);
                if let Some(objective) = objective {
                    let impact = objective.score(self.solver, v);
                    if impact > best_objective_impact || (impact == best_objective_impact && impact > 0 && score < best_objective_score) {
                        best_objective_impact = impact;
                        best_objective_score = score;
                        best_objective = Some(v);
                    }
                }
                if score < best_score {
                    best_score = score;
                    best = Some(v);
                }
            }
        }
        best_objective.or(best)
    }

    /// The decision literal for `v`: `[v = p]` toward the saved phase `p` when it
    /// is still in the domain (skipped on a rephasing segment), else a polarity
    /// endpoint chosen by [`rephase_value`](Self::rephase_value).
    fn decision_lit(&self, v: VarId, phase: &[Option<i32>]) -> Lit {
        let val = match phase[v.index()] {
            Some(p) if self.rephase_mode == 0 && self.solver.store.contains(v, p) => p,
            _ => self.rephase_value(v),
        };
        match self.atoms.eq(v, val) {
            LitOrConst::Lit(l) => l,
            _ => {
                unreachable!("an unfixed variable has a real equality atom for an in-domain value")
            }
        }
    }

    /// The domain endpoint to branch to when no saved phase applies: the
    /// seed-dependent endpoint by default, inverted on a rephasing segment.
    fn rephase_value(&self, v: VarId) -> i32 {
        let default_max = self.seed != 0 && mix64(self.seed ^ v.0 as u64) & 1 != 0;
        let want_max = default_max ^ (self.rephase_mode != 0);
        if want_max {
            self.solver.store.max(v)
        } else {
            self.solver.store.min(v)
        }
    }

    /// Restart to the root when the restart policy fires. Learned clauses and
    /// saved phases survive. Every `REPHASE_PERIOD` restarts the next segment
    /// *rephases*: it ignores saved phases and dives with an inverted or random
    /// polarity, escaping a region the saved phase keeps pulling search back to.
    fn maybe_restart(&mut self) -> bool {
        if self.should_restart() {
            self.backjump_to(0);
            self.restarts_done += 1;
            // Rephase only in a lone sequential search. Every `REPHASE_PERIOD`
            // restarts the next segment inverts the default polarity (not random:
            // an opposite dive diversifies the feasibility search without
            // shredding structured objectives like LABS). Portfolio workers
            // already diversify across workers, and per-worker rephasing
            // *synchronizes* their polarity flips and hurts - so it is disabled
            // whenever clause sharing (a portfolio) is active.
            let lone = self.clause_sharing.is_none();
            self.rephase_mode = u8::from(lone && self.restarts_done.is_multiple_of(REPHASE_PERIOD));
            self.maybe_reduce_db();
            return self.sync_shared_clauses();
        }
        true
    }

    /// Enumerate solutions over `vars`, invoking `on_solution` per full
    /// assignment and stopping early on [`SearchControl::Stop`] or when `stop` is set.
    ///
    /// Plain chronological-backtracking DFS. Clause learning is deliberately not
    /// used: resolving against the blocking clauses (not entailed by the
    /// constraints) can drop sibling solutions when enumerating.
    // TODO(strong): a sound CDCL enumeration (restart-per-solution with blocking
    // clauses kept out of analysis, or dual reasoning) would prune harder.
    pub fn enumerate<F>(&mut self, vars: &[VarId], mut on_solution: F, stop: &AtomicBool) -> SolveStats
    where
        F: FnMut(&Solver) -> SearchControl,
    {
        let mut stats = SolveStats::default();
        if !self.init() || !self.root_probe(vars) {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return stats; // root unsatisfiable
        }
        let mut phase: Vec<Option<i32>> = vec![None; self.solver.store.num_vars()];
        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            match self.propagate() {
                Ok(()) => match self.select_var(vars, None) {
                    None => {
                        // Full assignment.
                        for &v in vars {
                            phase[v.index()] = Some(self.solver.store.value(v));
                        }
                        stats.solutions += 1;
                        if matches!(on_solution(&*self.solver), SearchControl::Stop) {
                            break;
                        }
                        if !self.backtrack_and_forbid() {
                            break; // whole search space explored
                        }
                    }
                    Some(v) => {
                        stats.nodes += 1;
                        let lit = self.decision_lit(v, &phase);
                        self.decide(lit).expect("in-domain decision cannot fail");
                    }
                },
                Err(_) => {
                    stats.failures += 1;
                    if !self.backtrack_and_forbid() {
                        break; // dead end at the root: search exhausted
                    }
                }
            }
        }
        self.copy_inprocessing_stats(&mut stats);
        stats
    }

    /// Undo the deepest decision `x = v` and forbid `v` for `x` (DFS "try the next
    /// value"); climb if that empties `x`. Returns `false` once the space is exhausted.
    fn backtrack_and_forbid(&mut self) -> bool {
        loop {
            let d = self.decision_level();
            if d == 0 {
                return false;
            }
            let dec = self.deepest_decision();
            self.backjump_to(d - 1);
            // Assert the decision's negation via `assign` (not a direct domain
            // edit) so it is trailed at this level and its event is drained.
            match self.assign(dec.negate(), Reason::Decision) {
                Ok(()) => return true,
                Err(_) => continue, // emptied the variable: climb
            }
        }
    }

    /// The literal that demands a strictly better objective than `value`:
    /// `¬[obj ≥ value]` when minimizing, `[obj ≥ value+1]` when maximizing.
    /// `None` means no better value is representable; `value` is optimal.
    fn improvement_lit(&self, obj: VarId, value: i32, minimizing: bool) -> Option<Lit> {
        if minimizing {
            match self.atoms.ge(obj, value) {
                LitOrConst::Lit(l) => Some(l.negate()), // obj < value
                LitOrConst::True => None,               // obj ≥ value always: optimal
                LitOrConst::False => unreachable!("found objective value below its own minimum"),
            }
        } else {
            let next = value.checked_add(1)?;
            match self.atoms.ge(obj, next) {
                LitOrConst::Lit(l) => Some(l), // obj > value
                LitOrConst::False => None,     // obj ≤ value always: optimal
                LitOrConst::True => unreachable!("found objective value above its own maximum"),
            }
        }
    }

    /// Assert that the objective must improve on `incumbent`. Returns `false`
    /// when no better value exists under the imported bound.
    fn tighten_bound(&mut self, obj: VarId, minimizing: bool, incumbent: i32) -> bool {
        let Some(bound) = self.improvement_lit(obj, incumbent, minimizing) else {
            return false;
        };
        self.backjump_to(0);
        self.set_bound_scope(bound);
        self.assert_root_lit(bound)
    }

    /// Assert a strict improvement on a materialized or symbolic objective.
    ///
    /// The `Var` path asserts a root literal on the existing objective atom, so
    /// it never accumulates. `Linear`/`Expr` have no objective atom, so the bound
    /// is enforced by ONE persistent propagator whose rhs lives in `cell`; each
    /// improving incumbent updates the cell (bounds only ever tighten, keeping it
    /// sound) and re-enqueues that propagator instead of posting a new one.
    fn tighten_objective(&mut self, objective: Objective<'_>, minimizing: bool, incumbent: i64, cell: &mut ObjBoundCell) -> bool {
        match objective {
            Objective::Var(obj) => self.tighten_bound(obj, minimizing, incumbent as i32),
            Objective::Linear { coeffs, vars } => {
                let Some(bound) = strict_bound(incumbent, minimizing) else {
                    return false;
                };
                self.backjump_to(0);
                // Minimizing: ∑ coeffs·vars ≤ bound. Maximizing: negate both so it
                // is still a `≤ c` constraint whose c falls as incumbents improve.
                let c = if minimizing { bound } else { -bound };
                match &cell.handle {
                    Some((id, atom)) => {
                        atom.store(c, Ordering::Relaxed);
                        self.solver.store.enqueue(*id);
                    }
                    None => {
                        let atom = Arc::new(AtomicI64::new(c));
                        let coeffs = if minimizing { coeffs.to_vec() } else { coeffs.iter().map(|&a| -a).collect() };
                        let id = self.solver.post(Box::new(ObjLinearLeq::new(coeffs, vars, Arc::clone(&atom))));
                        cell.handle = Some((id, atom));
                    }
                }
                self.propagate_and_learn()
            }
            Objective::Expr(e) => {
                let Some(bound) = strict_bound(incumbent, minimizing) else {
                    return false;
                };
                self.backjump_to(0);
                let c = if minimizing { bound } else { -bound };
                match &cell.handle {
                    Some((id, atom)) => {
                        atom.store(c, Ordering::Relaxed);
                        self.solver.store.enqueue(*id);
                    }
                    None => {
                        let atom = Arc::new(AtomicI64::new(c));
                        // Maximizing (expr ≥ bound) as neg(expr) ≤ -bound.
                        let e = if minimizing { e.clone() } else { expr::neg(e.clone()) };
                        let id = self.solver.post(Box::new(ObjExprLeq::new(e, Arc::clone(&atom))));
                        cell.handle = Some((id, atom));
                    }
                }
                self.propagate_and_learn()
            }
        }
    }

    /// Assert the root units defining one disjoint search cube.
    fn assume_cube(&mut self, cube: &[Lit]) -> bool {
        self.set_cube_scope(cube);
        cube.iter().copied().all(|lit| self.assert_root_lit(lit))
    }

    /// Assert `obj <= target` when minimizing or `obj >= target` when maximizing.
    fn assume_objective_bound(&mut self, obj: VarId, target: i32, minimizing: bool) -> bool {
        let bound = if minimizing {
            let Some(next) = target.checked_add(1) else {
                return true;
            };
            match self.atoms.ge(obj, next) {
                LitOrConst::True => return false,
                LitOrConst::False => return true,
                LitOrConst::Lit(l) => l.negate(),
            }
        } else {
            match self.atoms.ge(obj, target) {
                LitOrConst::True => return true,
                LitOrConst::False => return false,
                LitOrConst::Lit(l) => l,
            }
        };
        self.set_bound_scope(bound);
        self.assert_root_lit(bound)
    }

    /// Pick one binary split for a cube, or `None` when it is already terminal.
    pub(crate) fn split_cube(&mut self, vars: &[VarId], cube: &[Lit]) -> Option<Lit> {
        if self.stopped() || !self.init() || !self.assume_cube(cube) || self.stopped() {
            return None;
        }
        let phase = vec![None; self.solver.store.num_vars()];
        self.select_var(vars, None).map(|v| self.decision_lit(v, &phase))
    }

    /// Find one solution under an optimistic objective bound.
    pub(crate) fn probe(
        &mut self,
        vars: &[VarId],
        obj: VarId,
        minimizing: bool,
        target: i32,
        stop: &AtomicBool,
    ) -> (Option<(Vec<i32>, i32)>, SolveStats, bool) {
        let mut stats = SolveStats::default();
        if !self.init() || !self.assume_objective_bound(obj, target, minimizing) || !self.sync_shared_clauses() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (None, stats, true);
        }
        let phase = vec![None; self.solver.store.num_vars()];
        let mut complete = true;
        let found = loop {
            if stop.load(Ordering::Relaxed) {
                complete = false;
                break None;
            }
            if !self.maybe_restart() {
                break None;
            }
            match self.select_var(vars, None) {
                None => {
                    stats.solutions += 1;
                    let value = self.solver.store.value(obj);
                    let assignment = vars.iter().map(|&v| self.solver.store.value(v)).collect();
                    break Some((assignment, value));
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit(v, &phase);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        break None;
                    }
                }
            }
        };
        stats.failures = self.conflicts;
        self.copy_inprocessing_stats(&mut stats);
        (found, stats, complete)
    }

    /// Record a new incumbent, report it, and assert a strictly-better objective
    /// bound at level 0. Returns `false` when the incumbent is proven optimal.
    #[allow(clippy::too_many_arguments)]
    fn accept_solution<F: FnMut(i64, &[i32])>(
        &mut self,
        vars: &[VarId],
        objective: Objective<'_>,
        minimizing: bool,
        best: &mut Option<(Vec<i32>, i64)>,
        enforced: &mut Option<i64>,
        phase: &mut [Option<i32>],
        stats: &mut SolveStats,
        cell: &mut ObjBoundCell,
        on_improve: &mut F,
    ) -> bool {
        let value = objective.value(self.solver);
        let assignment: Vec<i32> = vars.iter().map(|&v| self.solver.store.value(v)).collect();
        for &v in vars {
            phase[v.index()] = Some(self.solver.store.value(v));
        }
        stats.solutions += 1;
        on_improve(value, &assignment);
        *best = Some((assignment, value));
        *enforced = Some(value);
        self.tighten_objective(objective, minimizing, value, cell)
    }

    /// CDCL branch-and-bound with restarts. Returns the best `(assignment,
    /// value)`, proven optimal unless `stop` was set.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn optimize<F: FnMut(i64, &[i32])>(
        &mut self,
        vars: &[VarId],
        objective: Objective<'_>,
        minimizing: bool,
        stop: &AtomicBool,
        shared_bound: Option<&AtomicI64>,
        cube: &[Lit],
        conflict_budget: Option<u64>,
        mut on_improve: F,
    ) -> (Option<(Vec<i32>, i64)>, SolveStats, bool) {
        self.optimize_with_mode(vars, objective, minimizing, stop, shared_bound, cube, conflict_budget, &mut on_improve)
    }

    #[allow(clippy::too_many_arguments)]
    fn optimize_with_mode<F: FnMut(i64, &[i32])>(
        &mut self,
        vars: &[VarId],
        objective: Objective<'_>,
        minimizing: bool,
        stop: &AtomicBool,
        shared_bound: Option<&AtomicI64>,
        cube: &[Lit],
        conflict_budget: Option<u64>,
        on_improve: &mut F,
    ) -> (Option<(Vec<i32>, i64)>, SolveStats, bool) {
        if let Objective::Linear { coeffs, vars } = objective {
            assert_eq!(coeffs.len(), vars.len(), "linear objective: coeffs/terms length mismatch");
        }
        let mut stats = SolveStats::default();
        let mut best: Option<(Vec<i32>, i64)> = None;
        self.set_conflict_budget(conflict_budget);
        if !self.init() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (best, stats, !self.conflict_budget_exhausted());
        }
        if cube.is_empty() && conflict_budget.is_none() && !self.root_probe(vars) {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (best, stats, true);
        }
        if !self.assume_cube(cube) {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (best, stats, !self.conflict_budget_exhausted());
        }
        // Seed the saved-phase array with the caller's value-ordering hint (e.g.
        // nearest-neighbour successors) when supplied, else start blank.
        let mut phase: Vec<Option<i32>> = if self.initial_phase.len() == self.solver.store.num_vars() {
            self.initial_phase.clone()
        } else {
            vec![None; self.solver.store.num_vars()]
        };
        let objective_impact = ObjectiveImpact::new(objective, self.solver.store.num_vars(), minimizing);
        let mut enforced = None;
        let mut obj_bound = ObjBoundCell::default();
        let mut complete = true;
        if !self.sync_shared_clauses() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (best, stats, !self.conflict_budget_exhausted());
        }

        loop {
            if stop.load(Ordering::Relaxed) || self.conflict_budget_reached() {
                complete = false;
                break;
            }
            if let Some(shared) = shared_bound {
                let value = shared.load(Ordering::Relaxed);
                if value != i64::MAX && value != i64::MIN {
                    let stronger = enforced.is_none_or(|old| if minimizing { value < old } else { value > old });
                    if stronger {
                        enforced = Some(value);
                        if !self.tighten_objective(objective, minimizing, value, &mut obj_bound) {
                            if self.conflict_budget_exhausted() {
                                complete = false;
                            }
                            break;
                        }
                    }
                }
            }
            if !self.maybe_restart() {
                if self.conflict_budget_exhausted() {
                    complete = false;
                }
                break;
            }
            match self.select_var(vars, objective_impact.as_ref()) {
                None => {
                    let keep_searching = self.accept_solution(
                        vars,
                        objective,
                        minimizing,
                        &mut best,
                        &mut enforced,
                        &mut phase,
                        &mut stats,
                        &mut obj_bound,
                        on_improve,
                    );
                    if !keep_searching {
                        if self.conflict_budget_exhausted() {
                            complete = false;
                        }
                        break; // optimal
                    }
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit(v, &phase);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        if self.conflict_budget_exhausted() {
                            complete = false;
                        }
                        break; // tree exhausted under the bound: optimal
                    }
                }
            }
        }
        stats.failures = self.conflicts;
        self.copy_inprocessing_stats(&mut stats);
        (best, stats, complete)
    }

    /// CDCL decision driver for CSP: find one solution or prove UNSAT.
    /// It never posts solution-blocking clauses, so learned clauses remain
    /// consequences of the model. Use enumeration for counting/all-solutions.
    pub(crate) fn decide_sat(&mut self, vars: &[VarId], stop: &AtomicBool) -> (Option<Vec<i32>>, SolveStats, bool) {
        let mut stats = SolveStats::default();
        if !self.init() || !self.root_probe(vars) || !self.sync_shared_clauses() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (None, stats, true);
        }
        if self.initial_phase.len() == self.solver.store.num_vars() {
            self.saved_phase = self.initial_phase.clone();
        }
        let mut complete = true;
        let solution = loop {
            if stop.load(Ordering::Relaxed) {
                complete = false;
                break None;
            }
            if !self.maybe_restart() {
                break None;
            }
            match self.select_var(vars, None) {
                None => {
                    stats.solutions += 1;
                    break Some(vars.iter().map(|&v| self.solver.store.value(v)).collect());
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit(v, &self.saved_phase);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        break None;
                    }
                }
            }
        };
        stats.failures = self.conflicts;
        self.copy_inprocessing_stats(&mut stats);
        (solution, stats, complete)
    }

    /// CDCL decision driver under a temporary assumption cube. Learned clauses
    /// exported from this run are scoped by the negated cube literals.
    pub(crate) fn decide_sat_assuming(
        &mut self,
        vars: &[VarId],
        cube: &[Lit],
        conflict_budget: Option<u64>,
        stop: &AtomicBool,
    ) -> (Option<Vec<i32>>, SolveStats, bool) {
        let mut stats = SolveStats::default();
        self.set_conflict_budget(conflict_budget);
        if !self.init() || !self.assume_cube(cube) || !self.sync_shared_clauses() {
            stats.failures = self.conflicts;
            self.copy_inprocessing_stats(&mut stats);
            return (None, stats, !self.conflict_budget_exhausted());
        }
        if self.initial_phase.len() == self.solver.store.num_vars() {
            self.saved_phase = self.initial_phase.clone();
        }
        let mut complete = true;
        let solution = loop {
            if stop.load(Ordering::Relaxed) || self.conflict_budget_reached() {
                complete = false;
                break None;
            }
            if !self.maybe_restart() {
                complete = !self.conflict_budget_exhausted();
                break None;
            }
            match self.select_var(vars, None) {
                None => {
                    stats.solutions += 1;
                    break Some(vars.iter().map(|&v| self.solver.store.value(v)).collect());
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit(v, &self.saved_phase);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        complete = !self.conflict_budget_exhausted();
                        break None;
                    }
                }
            }
        };
        stats.failures = self.conflicts;
        self.copy_inprocessing_stats(&mut stats);
        (solution, stats, complete)
    }

    /// Solve the root constraints under `cube`, a set of assumption literals,
    /// returning a model, an unsat core (subset of `cube`), or interruption.
    ///
    /// MiniSat-style assumption solving: each assumption is placed as its own
    /// decision (not a root unit), so learned clauses stay entailed by the root
    /// alone and a refutation traces back through [`analyze_final`] to exactly
    /// the assumptions that participated. `decision_level()` doubles as the index
    /// of the next assumption to place; already-implied assumptions get a dummy
    /// level to keep that alignment (and to be replayed identically after a
    /// restart, which backjumps to the root).
    pub(crate) fn solve_under_assumptions(&mut self, vars: &[VarId], cube: &[Lit], stop: &AtomicBool) -> AssumptionOutcome {
        if !self.init() {
            return AssumptionOutcome::Unsat(Vec::new()); // root already unsatisfiable
        }
        self.resolve_assumptions(vars, cube, stop)
    }

    /// Re-solve under `cube` from the current root, reusing the engine's learned
    /// clauses (all entailed by the root, hence valid under any cube). The caller
    /// must have run [`init`](Cdcl::init) once. This is the per-query primitive
    /// for MUS minimisation, which fires many cubes at one engine.
    pub(crate) fn resolve_assumptions(&mut self, vars: &[VarId], cube: &[Lit], stop: &AtomicBool) -> AssumptionOutcome {
        self.backjump_to(0);
        loop {
            if stop.load(Ordering::Relaxed) {
                return AssumptionOutcome::Interrupted;
            }
            if !self.maybe_restart() {
                return AssumptionOutcome::Unsat(Vec::new());
            }
            let d = self.decision_level();
            if d < cube.len() {
                // Phase 1: place the next assumption as a decision.
                let p = cube[d];
                match self.value(p) {
                    Tri::True => self.open_level(), // already implied: dummy level
                    Tri::False => {
                        let core = self.analyze_final(p);
                        self.backjump_to(0);
                        return AssumptionOutcome::Unsat(core);
                    }
                    Tri::Unknown => {
                        self.decide(p).expect("in-domain assumption decision cannot fail");
                        if !self.propagate_and_learn() {
                            // Level-0 refutation: every learnt clause is root-entailed,
                            // so this means the root alone is unsatisfiable.
                            self.backjump_to(0);
                            return AssumptionOutcome::Unsat(Vec::new());
                        }
                    }
                }
                continue;
            }
            // Phase 2: assumptions all satisfied, branch for a model. A conflict
            // that refutes the cube backjumps below `cube.len()`, re-entering
            // Phase 1 where `analyze_final` catches the now-false assumption.
            match self.select_var(vars, None) {
                None => return AssumptionOutcome::Sat(vars.iter().map(|&v| self.solver.store.value(v)).collect()),
                Some(v) => {
                    let lit = self.decision_lit(v, &self.saved_phase);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        self.backjump_to(0);
                        return AssumptionOutcome::Unsat(Vec::new()); // root alone is unsat
                    }
                }
            }
        }
    }
}

/// Handle to the single persistent objective-bound propagator (Linear/Expr).
/// Owned by one `optimize` run; the propagator is posted lazily on the first
/// improving incumbent, so a solver cloned from the immutable model template
/// (parallel/cube workers, see `parallel.rs`) never aliases another worker's
/// bound cell, each posts its own on its own first incumbent.
// ponytail: per-run cell; no cross-run reuse needed since a solver runs one optimize.
#[derive(Default)]
struct ObjBoundCell {
    handle: Option<(PropId, Arc<AtomicI64>)>,
}

/// `∑ coeffs·vars ≤ bound`, with `bound` read from a shared cell each run so it
/// can tighten monotonically in place. Filtering mirrors `constraints::linear`'s
/// `LinearLeq`; the only difference is the mutable rhs.
#[derive(Clone)]
struct ObjLinearLeq {
    coeffs: Vec<i64>,
    vars: Vec<VarId>,
    bound: Arc<AtomicI64>,
    term_min: Vec<i64>,
}

impl ObjLinearLeq {
    fn new(coeffs: Vec<i64>, vars: &[VarId], bound: Arc<AtomicI64>) -> Self {
        let n = vars.len();
        Self { coeffs, vars: vars.to_vec(), bound, term_min: vec![0; n] }
    }

    fn min_side(&self, store: &Store, skip: usize) -> Vec<Premise> {
        let mut why = Vec::new();
        for (j, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
            if j == skip || a == 0 {
                continue;
            }
            why.push(if a > 0 { Premise::Ge { var: v, bound: store.min(v) } } else { Premise::Le { var: v, bound: store.max(v) } });
        }
        why
    }
}

impl Propagator for ObjLinearLeq {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::BoundChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let c = self.bound.load(Ordering::Relaxed);
        loop {
            let mut sum_min: i64 = 0;
            let mut sum_max: i64 = 0;
            for (slot, (&a, &v)) in self.term_min.iter_mut().zip(self.coeffs.iter().zip(&self.vars)) {
                let lo = store.min(v) as i64;
                let hi = store.max(v) as i64;
                let (tmin, tmax) = if a >= 0 { (a * lo, a * hi) } else { (a * hi, a * lo) };
                *slot = tmin;
                sum_min += tmin;
                sum_max += tmax;
            }

            if sum_min > c {
                return Err(store.fail_because(self.min_side(store, usize::MAX)));
            }
            if sum_max <= c {
                return Ok(()); // entailed
            }

            let mut changed = false;
            for (idx, (&a, &v)) in self.coeffs.iter().zip(&self.vars).enumerate() {
                if a == 0 {
                    continue;
                }
                let allowed = c - (sum_min - self.term_min[idx]);
                if a > 0 {
                    let bound = clamp_i32(floor_div(allowed, a));
                    if bound < store.max(v) {
                        store.remove_above_because(v, bound, self.min_side(store, idx))?;
                        changed = true;
                    }
                } else {
                    let bound = clamp_i32(ceil_div(allowed, a));
                    if bound > store.min(v) {
                        store.remove_below_because(v, bound, self.min_side(store, idx))?;
                        changed = true;
                    }
                }
            }

            if !changed {
                return Ok(());
            }
        }
    }
}

/// `expr ≤ bound`, with `bound` from a shared cell. Filtering mirrors
/// `constraints::intension`'s `Intension` specialized to a `≤ c` root so the rhs
/// can tighten in place (a maximizing objective is posted as `neg(expr) ≤ -c`).
#[derive(Clone)]
struct ObjExprLeq {
    expr: Expr,
    vars: Vec<VarId>,
    bound: Arc<AtomicI64>,
    scratch: Vec<i32>,
}

impl ObjExprLeq {
    fn new(expr: Expr, bound: Arc<AtomicI64>) -> Self {
        let mut vars = Vec::new();
        expr.collect_vars(&mut vars);
        vars.sort_unstable();
        vars.dedup();
        Self { expr, vars, bound, scratch: Vec::new() }
    }
}

impl Propagator for ObjExprLeq {
    fn register(&mut self, store: &mut Store, me: PropId) {
        for &v in &self.vars {
            store.subscribe(v, me, Event::DomainChange);
        }
    }

    fn propagate(&mut self, store: &mut Store) -> Result<(), Inconsistency> {
        let c = self.bound.load(Ordering::Relaxed);
        let (lo, hi) = {
            let dom = |x: VarId| (store.min(x) as i64, store.max(x) as i64);
            self.expr.bounds(&dom)
        };
        if lo > c {
            return Err(Inconsistency); // can never hold
        }
        if hi <= c {
            return Ok(()); // entailed
        }

        for &v in &self.vars {
            if store.is_fixed(v) {
                continue;
            }
            self.scratch.clear();
            self.scratch.extend(store.values(v));
            for &val in &self.scratch {
                let dead = {
                    let dom = |x: VarId| if x == v { (val as i64, val as i64) } else { (store.min(x) as i64, store.max(x) as i64) };
                    self.expr.bounds(&dom).0 > c
                };
                if dead {
                    store.remove(v, val)?;
                }
            }
        }

        if self.vars.iter().all(|&x| store.is_fixed(x)) {
            match self.expr.eval(&|x| store.value(x) as i64) {
                Some(n) if n <= c => {}
                _ => return Err(Inconsistency),
            }
        }
        Ok(())
    }
}

/// Floor of `a / b` for integers (Rust `/` truncates toward zero).
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Ceiling of `a / b` for integers.
fn ceil_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) == (b < 0)) {
        q + 1
    } else {
        q
    }
}
