//! CDCL search drivers.
//!
//! [`Cdcl::enumerate`] is the CSP driver: chronological-backtracking DFS, no
//! clause learning (which is unsound for enumerating every solution).
//! [`Cdcl::optimize`] is the COP driver: CDCL branch-and-bound with restarts.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

use crate::constraints::intension::intension;
use crate::constraints::linear::{linear, Relation};
use crate::expr;
use crate::ids::VarId;
use crate::lcg::lit::{Lit, LitOrConst};
use crate::lcg::trail::{Cdcl, Reason};
use crate::search::{Objective, SearchControl, SolveStats};
use crate::store::Solver;

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

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
            Self::Linear { coeffs, vars } => coeffs.iter().zip(vars).map(|(&coeff, &var)| coeff * solver.store.value(var) as i64).sum(),
            Self::Expr(expr) => expr.eval(&|var| solver.store.value(var) as i64).expect("objective expression is undefined at a solution"),
        }
    }
}

struct ObjectiveImpact {
    coeffs: Vec<u64>,
}

impl ObjectiveImpact {
    fn new(objective: Objective<'_>, nvars: usize) -> Option<Self> {
        let Objective::Linear { coeffs, vars } = objective else {
            return None;
        };
        let mut impact = vec![0u64; nvars];
        for (&coeff, &var) in coeffs.iter().zip(vars) {
            impact[var.index()] = impact[var.index()].saturating_add(coeff.unsigned_abs());
        }
        impact.iter().any(|&coeff| coeff > 0).then_some(Self { coeffs: impact })
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

impl Cdcl<'_> {
    /// Branching heuristic: unfixed variable minimising `size / (wdeg + activity)`
    /// (dom/wdeg combined with VSIDS activity), or `None` if all fixed.
    fn select_var(&self, vars: &[VarId], objective: Option<&ObjectiveImpact>) -> Option<VarId> {
        let weights = self.solver.weights();
        let mut best: Option<VarId> = None;
        let mut best_score = f64::INFINITY;
        let mut best_objective: Option<VarId> = None;
        let mut best_objective_impact = 0u128;
        let mut best_objective_score = f64::INFINITY;
        for &v in vars {
            let size = self.solver.store.size(v);
            if size > 1 {
                let wdeg = self.solver.store.var_weight(v, weights) as f64;
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

    /// The decision literal for `v`: `[v = p]` toward the saved phase `p` when
    /// it is still in the domain, else a seed-dependent endpoint.
    fn decision_lit(&self, v: VarId, phase: &[Option<i32>]) -> Lit {
        let val = match phase[v.index()] {
            Some(p) if self.solver.store.contains(v, p) => p,
            _ if self.seed != 0 && mix64(self.seed ^ v.0 as u64) & 1 != 0 => self.solver.store.max(v),
            _ => self.solver.store.min(v),
        };
        match self.atoms.eq(v, val) {
            LitOrConst::Lit(l) => l,
            _ => {
                unreachable!("an unfixed variable has a real equality atom for an in-domain value")
            }
        }
    }

    /// Restart to the root when the restart policy fires. Learned clauses and
    /// saved phases survive.
    fn maybe_restart(&mut self) -> bool {
        if self.should_restart() {
            self.backjump_to(0);
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
            stats.learned_lits = self.learned_lits;
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
        stats.learned_lits = self.learned_lits;
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
        self.assert_root(bound)
    }

    /// Assert a strict improvement on a materialized or symbolic objective.
    fn tighten_objective(&mut self, objective: Objective<'_>, minimizing: bool, incumbent: i64) -> bool {
        match objective {
            Objective::Var(obj) => self.tighten_bound(obj, minimizing, incumbent as i32),
            Objective::Linear { coeffs, vars } => {
                let Some(bound) = strict_bound(incumbent, minimizing) else {
                    return false;
                };
                self.backjump_to(0);
                linear(self.solver, coeffs, vars, if minimizing { Relation::Le } else { Relation::Ge }, bound);
                self.propagate_and_learn()
            }
            Objective::Expr(expr) => {
                let Some(bound) = strict_bound(incumbent, minimizing) else {
                    return false;
                };
                self.backjump_to(0);
                intension(
                    self.solver,
                    if minimizing { expr::le(expr.clone(), expr::int(bound)) } else { expr::ge(expr.clone(), expr::int(bound)) },
                );
                self.propagate_and_learn()
            }
        }
    }

    /// Assert a non-deletable root unit and propagate it.
    fn assert_root(&mut self, lit: Lit) -> bool {
        self.assert_root_lit(lit)
    }

    /// Assert the root units defining one disjoint search cube.
    fn assume_cube(&mut self, cube: &[Lit]) -> bool {
        self.set_cube_scope(cube);
        cube.iter().copied().all(|lit| self.assert_root(lit))
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
        self.assert_root(bound)
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
            stats.learned_lits = self.learned_lits;
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
        stats.learned_lits = self.learned_lits;
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
        self.tighten_objective(objective, minimizing, value)
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
        if let Objective::Linear { coeffs, vars } = objective {
            assert_eq!(coeffs.len(), vars.len(), "linear objective: coeffs/terms length mismatch");
        }
        let mut stats = SolveStats::default();
        let mut best: Option<(Vec<i32>, i64)> = None;
        if !self.init() {
            stats.failures = self.conflicts;
            stats.learned_lits = self.learned_lits;
            return (best, stats, true);
        }
        if cube.is_empty() && conflict_budget.is_none() && !self.root_probe(vars) {
            stats.failures = self.conflicts;
            stats.learned_lits = self.learned_lits;
            return (best, stats, true);
        }
        if !self.assume_cube(cube) {
            stats.failures = self.conflicts;
            stats.learned_lits = self.learned_lits;
            return (best, stats, true);
        }
        let mut phase: Vec<Option<i32>> = vec![None; self.solver.store.num_vars()];
        let objective_impact = ObjectiveImpact::new(objective, self.solver.store.num_vars());
        let mut enforced = None;
        let mut complete = true;
        let conflict_limit = conflict_budget.map(|n| self.conflicts.saturating_add(n));
        if !self.sync_shared_clauses() {
            stats.failures = self.conflicts;
            stats.learned_lits = self.learned_lits;
            return (best, stats, true);
        }

        loop {
            if stop.load(Ordering::Relaxed) || conflict_limit.is_some_and(|limit| self.conflicts >= limit) {
                complete = false;
                break;
            }
            if let Some(shared) = shared_bound {
                let value = shared.load(Ordering::Relaxed);
                if value != i64::MAX && value != i64::MIN {
                    let stronger = enforced.is_none_or(|old| if minimizing { value < old } else { value > old });
                    if stronger {
                        enforced = Some(value);
                        if !self.tighten_objective(objective, minimizing, value) {
                            break;
                        }
                    }
                }
            }
            if !self.maybe_restart() {
                break;
            }
            match self.select_var(vars, objective_impact.as_ref()) {
                None => {
                    if !self.accept_solution(vars, objective, minimizing, &mut best, &mut enforced, &mut phase, &mut stats, &mut on_improve)
                    {
                        break; // optimal
                    }
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit(v, &phase);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        break; // tree exhausted under the bound: optimal
                    }
                }
            }
        }
        stats.failures = self.conflicts;
        stats.learned_lits = self.learned_lits;
        (best, stats, complete)
    }
}
