//! CDCL search drivers.
//!
//! [`Cdcl::enumerate`] is the CSP driver: chronological-backtracking DFS, no
//! clause learning (which is unsound for enumerating every solution).
//! [`Cdcl::optimize`] is the COP driver: complete proving epochs interleaved
//! with large-neighbourhood-search bursts.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::VarId;
use crate::lcg::lit::{Lit, LitOrConst};
use crate::lcg::trail::{Cdcl, Reason};
use crate::lcg::view::Tri;
use crate::search::{SearchControl, SolveStats};
use crate::store::Solver;

/// LNS moves attempted between successive complete (proving) epochs.
const LNS_BURST: usize = 8;
/// Per-move conflict budget for an LNS neighbourhood search.
const LNS_BUDGET: u64 = 500;

/// Deterministic xorshift64 PRNG.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

impl Cdcl<'_> {
    /// Branching heuristic: unfixed variable minimising `size / (wdeg + activity)`
    /// (dom/wdeg combined with VSIDS activity), or `None` if all fixed.
    fn select_var(&self, vars: &[VarId]) -> Option<VarId> {
        let weights = self.solver.weights();
        let mut best: Option<VarId> = None;
        let mut best_score = f64::INFINITY;
        for &v in vars {
            let size = self.solver.store.size(v);
            if size > 1 {
                let wdeg = self.solver.store.var_weight(v, weights) as f64;
                let score = size as f64 / (wdeg + self.activity[v.index()]);
                if score < best_score {
                    best_score = score;
                    best = Some(v);
                }
            }
        }
        best
    }

    /// The decision literal for `v`: `[v = p]` toward the saved phase `p` when
    /// it is still in the domain, else `[v = min(v)]`.
    fn decision_lit(&self, v: VarId, phase: &[Option<i32>]) -> Lit {
        let val = match phase[v.index()] {
            Some(p) if self.solver.store.contains(v, p) => p,
            _ => self.solver.store.min(v),
        };
        match self.atoms.eq(v, val) {
            LitOrConst::Lit(l) => l,
            _ => {
                unreachable!("an unfixed variable has a real equality atom for an in-domain value")
            }
        }
    }

    /// Restart to the root once conflicts since the last restart reach `limit`,
    /// then grow `limit` by ×1.5. Learned clauses and saved phases survive.
    fn maybe_restart(&mut self, last: &mut u64, limit: &mut u64) {
        if self.conflicts - *last >= *limit {
            self.backjump_to(0);
            self.maybe_reduce_db();
            *last = self.conflicts;
            *limit += *limit / 2;
        }
    }

    /// Enumerate solutions over `vars`, invoking `on_solution` per full
    /// assignment and stopping early on [`SearchControl::Stop`] or when `stop` is set.
    ///
    /// Plain chronological-backtracking DFS. Clause learning is deliberately not
    /// used: resolving against the blocking clauses (not entailed by the
    /// constraints) can drop sibling solutions when enumerating.
    // TODO(strong): a sound CDCL enumeration (restart-per-solution with blocking
    // clauses kept out of analysis, or dual reasoning) would prune harder.
    pub fn enumerate<F>(
        &mut self,
        vars: &[VarId],
        mut on_solution: F,
        stop: &AtomicBool,
    ) -> SolveStats
    where
        F: FnMut(&Solver) -> SearchControl,
    {
        let mut stats = SolveStats::default();
        if !self.init() {
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
                Ok(()) => match self.select_var(vars) {
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
    /// `None` means no better value is representable — `value` is optimal.
    fn improvement_lit(&self, obj: VarId, value: i32, minimizing: bool) -> Option<Lit> {
        if minimizing {
            match self.atoms.ge(obj, value) {
                LitOrConst::Lit(l) => Some(l.negate()), // obj < value
                LitOrConst::True => None,               // obj ≥ value always: optimal
                LitOrConst::False => unreachable!("found objective value below its own minimum"),
            }
        } else {
            match self.atoms.ge(obj, value + 1) {
                LitOrConst::Lit(l) => Some(l), // obj > value
                LitOrConst::False => None,     // obj ≤ value always: optimal
                LitOrConst::True => unreachable!("found objective value above its own maximum"),
            }
        }
    }

    /// Record a new incumbent, report it, and assert a strictly-better objective
    /// bound at level 0. Returns `false` when the incumbent is proven optimal.
    #[allow(clippy::too_many_arguments)]
    fn accept_solution<F: FnMut(i32)>(
        &mut self,
        vars: &[VarId],
        obj: VarId,
        minimizing: bool,
        best: &mut Option<(Vec<i32>, i32)>,
        phase: &mut [Option<i32>],
        stats: &mut SolveStats,
        on_improve: &mut F,
    ) -> bool {
        let value = self.solver.store.value(obj);
        *best = Some((
            vars.iter().map(|&v| self.solver.store.value(v)).collect(),
            value,
        ));
        for &v in vars {
            phase[v.index()] = Some(self.solver.store.value(v));
        }
        stats.solutions += 1;
        on_improve(value);

        let Some(bound) = self.improvement_lit(obj, value, minimizing) else {
            return false; // optimal
        };
        self.backjump_to(0);
        // Non-deletable so the bound stays enforced.
        let cref = self.add_clause(vec![bound], false, 0);
        self.assign(bound, Reason::Clause(cref)).is_ok() && self.propagate_and_learn()
    }

    /// CDCL branch-and-bound: complete proving epochs (bounded by a conflict
    /// budget) interleaved with LNS bursts. `obj` must be among `vars`. Returns the
    /// best `(assignment, value)`, proven optimal unless `stop` was set.
    pub fn optimize<F: FnMut(i32)>(
        &mut self,
        vars: &[VarId],
        obj: VarId,
        minimizing: bool,
        stop: &AtomicBool,
        mut on_improve: F,
    ) -> (Option<(Vec<i32>, i32)>, SolveStats) {
        let mut stats = SolveStats::default();
        let mut best: Option<(Vec<i32>, i32)> = None;
        if !self.init() {
            stats.failures = self.conflicts;
            stats.learned_lits = self.learned_lits;
            return (best, stats);
        }
        let mut phase: Vec<Option<i32>> = vec![None; self.solver.store.num_vars()];
        let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15 ^ vars.len() as u64);
        let base = (vars.len() / 10).max(1);
        let mut relax = base;
        let mut epoch_budget: u64 = 1000;

        'search: loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            // --- proving epoch ---
            let epoch_start = self.conflicts;
            let (mut last_restart, mut restart_limit) = (self.conflicts, 100u64);
            loop {
                if stop.load(Ordering::Relaxed) {
                    break 'search;
                }
                self.maybe_restart(&mut last_restart, &mut restart_limit);
                match self.select_var(vars) {
                    None => {
                        if !self.accept_solution(
                            vars,
                            obj,
                            minimizing,
                            &mut best,
                            &mut phase,
                            &mut stats,
                            &mut on_improve,
                        ) {
                            break 'search; // optimal
                        }
                    }
                    Some(v) => {
                        stats.nodes += 1;
                        let lit = self.decision_lit(v, &phase);
                        self.decide(lit).expect("in-domain decision cannot fail");
                        if !self.propagate_and_learn() {
                            break 'search; // tree exhausted under the bound: optimal
                        }
                    }
                }
                if self.conflicts - epoch_start >= epoch_budget {
                    break; // yield to LNS
                }
            }
            self.backjump_to(0); // discard the partial proving tree
                                 // Grow the budget, capped so LNS keeps getting turns.
            epoch_budget = (epoch_budget + epoch_budget / 2).min(8000);

            // --- LNS bursts ---
            if best.is_some() {
                for _ in 0..LNS_BURST {
                    if stop.load(Ordering::Relaxed) {
                        break 'search;
                    }
                    let (improved, optimal) = self.lns_move(
                        vars,
                        obj,
                        minimizing,
                        relax,
                        &mut rng,
                        &mut best,
                        &mut phase,
                        &mut stats,
                        &mut on_improve,
                    );
                    if optimal {
                        break 'search;
                    }
                    relax = if improved {
                        base // intensify around the new incumbent
                    } else {
                        (relax + base).min(vars.len().saturating_sub(1)).max(base)
                        // diversify
                    };
                }
            }
        }
        stats.failures = self.conflicts;
        stats.learned_lits = self.learned_lits;
        (best, stats)
    }

    /// One LNS move: fix all but a random `relax`-sized subset of `vars` (a
    /// scattered set or contiguous window) to the incumbent, leaving `obj` free,
    /// then re-optimise the freed part within a budget. Returns `(improved,
    /// optimal)`. Trail-neutral: backjumps to the root before returning.
    #[allow(clippy::too_many_arguments)]
    fn lns_move<F: FnMut(i32)>(
        &mut self,
        vars: &[VarId],
        obj: VarId,
        minimizing: bool,
        relax: usize,
        rng: &mut Rng,
        best: &mut Option<(Vec<i32>, i32)>,
        phase: &mut [Option<i32>],
        stats: &mut SolveStats,
        on_improve: &mut F,
    ) -> (bool, bool) {
        let assignment = match best {
            Some((a, _)) => a.clone(),
            None => return (false, false),
        };
        debug_assert_eq!(self.decision_level(), 0);
        let n = vars.len();
        let start = self.conflicts;
        // Half scattered, half a contiguous window.
        let contiguous = rng.below(2) == 0;
        let window = if contiguous { rng.below(n) } else { 0 };

        // Fix the complement of the relaxed set to the incumbent.
        let mut feasible = true;
        for (k, &v) in vars.iter().enumerate() {
            if v == obj {
                continue; // keep the objective free
            }
            let relaxed = if contiguous {
                (k + n - window) % n < relax
            } else {
                rng.below(n) < relax
            };
            if relaxed {
                continue;
            }
            match self.atoms.eq(v, assignment[k]) {
                LitOrConst::Lit(l) => match self.tvalue(l) {
                    Tri::True => {}
                    Tri::False => {
                        feasible = false;
                        break;
                    }
                    Tri::Unknown => {
                        if self.decide(l).is_err() || !self.propagate_and_learn() {
                            feasible = false;
                            break;
                        }
                    }
                },
                LitOrConst::True => {}
                LitOrConst::False => {
                    feasible = false;
                    break;
                }
            }
            if self.stopped() || self.conflicts - start > LNS_BUDGET {
                feasible = false;
                break;
            }
        }

        let (mut improved, mut optimal) = (false, false);
        if feasible {
            loop {
                if self.stopped() || self.conflicts - start > LNS_BUDGET {
                    break; // out of time/budget
                }
                match self.select_var(vars) {
                    None => {
                        improved = true;
                        optimal = !self
                            .accept_solution(vars, obj, minimizing, best, phase, stats, on_improve);
                        break;
                    }
                    Some(v) => {
                        stats.nodes += 1;
                        let lit = self.decision_lit(v, phase);
                        if self.decide(lit).is_err() || !self.propagate_and_learn() {
                            break; // freed part is infeasible under the bound
                        }
                    }
                }
            }
        }
        if self.decision_level() != 0 {
            self.backjump_to(0); // discard the LNS fixings
        }
        (improved, optimal)
    }
}
