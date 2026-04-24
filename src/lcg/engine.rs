//! CDCL search drivers — the loops the public search entry points are built on.
//!
//! [`Cdcl::enumerate`] is the CSP driver: decide a variable (dom/wdeg, with
//! solution-guided phase saving), propagate-and-learn, and on a full assignment
//! report it and continue by adding a **blocking clause** (the negation of the
//! assignment) until the search space is exhausted. [`Cdcl::optimize`] is the COP
//! driver: it interleaves complete proving epochs (which assert a tighter
//! objective bound after each incumbent and ultimately prove optimality) with
//! large-neighbourhood-search bursts (which drive the incumbent down fast).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::VarId;
use crate::lcg::lit::{Lit, LitOrConst};
use crate::lcg::trail::{Cdcl, Conflict, Reason};
use crate::lcg::view::Tri;
use crate::search::{SearchControl, SolveStats};
use crate::store::Solver;

/// LNS moves attempted between successive complete (proving) epochs.
const LNS_BURST: usize = 8;
/// Per-move conflict budget for an LNS neighbourhood search.
const LNS_BUDGET: u64 = 500;

/// A tiny deterministic xorshift64 PRNG (reproducible runs, no dependency).
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
    /// Branching heuristic: the unfixed variable in `vars` minimising
    /// `size / (wdeg + activity)`, or `None` if all are fixed. Combines dom/wdeg
    /// (FD-failure signal) with VSIDS activity (clause-conflict signal), so
    /// learned-clause conflicts — the bulk under CDCL — steer branching too.
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

    /// Restart (backjump to the root) once the conflicts since the last restart
    /// reach `limit`, then grow `limit` geometrically. Learned clauses and saved
    /// phases survive, so the re-search reuses everything learned while escaping
    /// an unproductive region. The level-0 objective bound (if any) also
    /// survives, since it lives at the root.
    fn maybe_restart(&mut self, last: &mut u64, limit: &mut u64) {
        if self.conflicts - *last >= *limit {
            self.backjump_to(0);
            // At the root: a safe point to forget the worst learned clauses.
            self.maybe_reduce_db();
            *last = self.conflicts;
            *limit += *limit / 2; // geometric ×1.5
        }
    }

    /// The blocking clause for the current full assignment of `vars`: at least
    /// one variable must differ from its value (`⋁ ¬[v = value(v)]`).
    fn blocking_clause(&self, vars: &[VarId]) -> Vec<Lit> {
        vars.iter()
            .filter_map(|&v| match self.atoms.eq(v, self.solver.store.value(v)) {
                LitOrConst::Lit(l) => Some(l.negate()),
                _ => None,
            })
            .collect()
    }

    /// Enumerate solutions over `vars` by CDCL, invoking `on_solution` for each
    /// full assignment and stopping early on [`SearchControl::Stop`] or when
    /// `stop` is set. Returns the search statistics.
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
            return stats; // root is unsatisfiable
        }
        let mut phase: Vec<Option<i32>> = vec![None; self.solver.store.num_vars()];
        let (mut last_restart, mut restart_limit) = (0u64, 100u64);

        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            self.maybe_restart(&mut last_restart, &mut restart_limit);
            match self.select_var(vars) {
                None => {
                    // Full assignment: record the phase, report it, then block it.
                    for &v in vars {
                        phase[v.index()] = Some(self.solver.store.value(v));
                    }
                    stats.solutions += 1;
                    if matches!(on_solution(&*self.solver), SearchControl::Stop) {
                        break;
                    }
                    let blocking = self.blocking_clause(vars);
                    if blocking.is_empty() {
                        break; // no variables to constrain: the lone solution
                    }
                    // The blocking clause is falsified by the current solution;
                    // analyse it as a conflict to backjump, then continue. It must
                    // persist (never deleted) or solutions could be re-enumerated.
                    let cref = self.add_clause(blocking, false, 0);
                    if !self.resolve_conflict(Conflict::Clause(cref)) || !self.propagate_and_learn()
                    {
                        break; // search space exhausted
                    }
                }
                Some(v) => {
                    stats.nodes += 1;
                    let lit = self.decision_lit(v, &phase);
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        break;
                    }
                }
            }
        }
        stats.failures = self.conflicts;
        stats
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

    /// Record a full assignment as the new incumbent, save its phase, report it,
    /// and assert a strictly-better objective bound globally (a level-0 clause).
    /// Returns `false` when no better value is possible — the incumbent is then
    /// proven optimal and search should stop. Backjumps to the root.
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
            return false; // no better value possible: optimal
        };
        self.backjump_to(0);
        // The objective bound must persist (never deleted) to stay enforced.
        let cref = self.add_clause(vec![bound], false, 0);
        // A wipeout or root conflict from the tighter bound means it is optimal.
        self.assign(bound, Reason::Clause(cref)).is_ok() && self.propagate_and_learn()
    }

    /// Branch-and-bound by CDCL, interleaving a complete *prover* with large
    /// neighbourhood search (the *improver*). Each complete epoch searches under
    /// the current objective bound (asserted as a level-0 clause) until it finds
    /// a better incumbent, proves optimality (the bounded problem is
    /// unsatisfiable), or burns its conflict budget; between epochs, LNS bursts
    /// fix most variables to the incumbent and re-optimise a relaxed subset to
    /// drive the incumbent down fast. `obj` must be among `vars`. Returns the
    /// best `(assignment, value)`; the result is proven optimal unless `stop`
    /// was set.
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
            return (best, stats); // root unsatisfiable
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
            // --- complete proving epoch, bounded by a conflict budget ---
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
            // Grow the proving budget, but cap it so LNS keeps getting turns on
            // instances the prover cannot close.
            epoch_budget = (epoch_budget + epoch_budget / 2).min(8000);

            // --- LNS bursts to improve the incumbent before the next epoch ---
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
                        (relax + base).min(vars.len().saturating_sub(1)).max(base) // diversify
                    };
                }
            }
        }
        stats.failures = self.conflicts;
        (best, stats)
    }

    /// One LNS move: fix all but a random `relax`-sized subset of `vars` to the
    /// incumbent (the objective stays free), then re-optimise the freed part
    /// within a conflict budget. A scattered subset or a contiguous window is
    /// chosen at random. Returns `(improved, optimal)`. Trail-neutral: it
    /// backjumps to the root before returning.
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
        // Half the time a scattered subset, half a contiguous window (rows,
        // time-adjacent tasks and route segments tend to be consecutive).
        let contiguous = rng.below(2) == 0;
        let window = if contiguous { rng.below(n) } else { 0 };

        // Fix the complement of the relaxed set to the incumbent.
        let mut feasible = true;
        for (k, &v) in vars.iter().enumerate() {
            if v == obj {
                continue; // the objective must stay free to be re-derived
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
                    break; // time limit, or budget exhausted: abandon this move
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
