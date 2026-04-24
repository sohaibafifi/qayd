//! The SAT trail and the [`Cdcl`] engine state.
//!
//! [`Cdcl`] borrows the [`Solver`] for the duration of a solve and layers the
//! CDCL bookkeeping on top of the FD store:
//!
//! - the **SAT trail** — the literals assigned so far, in order, partitioned
//!   into decision levels by `level_starts`;
//! - per-atom side tables `atom_level` / `atom_reason` that the domain view
//!   cannot reconstruct (which level an atom was set at, and *why*).
//!
//! Atom *truth* is never stored here — it is read from the domain via
//! [`view`](crate::lcg::view). The trail only records the order and the reasons,
//! so on a backjump we truncate the trail and clear those side tables, then pop
//! the matching number of [`Trail`](crate::trail::Trail) levels; the domain
//! (hence every atom's truth) rolls back with it.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::VarId;
use crate::lcg::clause::{ClauseDb, ClauseRef};
use crate::lcg::lit::{Atom, AtomTable, Lit, LitOrConst};
use crate::lcg::view::{self, Tri};
use crate::propagator::Inconsistency;
use crate::store::{DomEvent, ScopeVar, Solver, StepOutcome};

/// A stop flag that is never set — the default until [`Cdcl::set_stop`] supplies
/// the search's real one. Lets the engine poll a flag without an `Option`.
pub(crate) static NEVER_STOP: AtomicBool = AtomicBool::new(false);

/// Sentinel `atom_level` for an unassigned atom.
pub const UNSET: u32 = u32::MAX;

/// Why an atom became assigned. Drives 1-UIP conflict analysis: each kind
/// expands to the clause that forced the literal (the [`Reason::Generic`]
/// propagator explanation lands in a later milestone).
#[derive(Clone)]
pub enum Reason {
    /// The atom is not currently assigned.
    Unset,
    /// True at the root (level 0) with no antecedent; dropped during analysis.
    Fact,
    /// A branching decision — a root of the implication graph; analysis stops.
    Decision,
    /// Forced as the last open literal of a clause (channeling axiom, learned,
    /// or constraint clause).
    Clause(ClauseRef),
    /// Inferred by a propagator. The literals are `D_S`: the (true) literals
    /// describing the propagator's scope domains at the time, which entail the
    /// inferred literal. The reason clause is `(consequent ∨ ⋁ ¬D_S)`.
    Generic(Vec<Lit>),
}

/// A detected conflict, before analysis turns it into a learned clause.
#[derive(Clone, Debug)]
pub enum Conflict {
    /// A clause is fully falsified (every literal false on the trail).
    Clause(ClauseRef),
    /// A propagator failed: these scope literals (`D_S`, all currently true)
    /// are jointly infeasible. The conflict clause is `⋁ ¬D_S`.
    Generic(Vec<Lit>),
}

/// Build `D_S`: the true literals describing a snapshotted scope's domains
/// (`[x≥min]`, `¬[x≥max+1]`, and `¬[x=v]` per hole). Their conjunction entails
/// whatever the propagator inferred from this scope.
fn build_ds(atoms: &AtomTable, scope: &[ScopeVar]) -> Vec<Lit> {
    let mut lits = Vec::new();
    for sv in scope {
        if let LitOrConst::Lit(l) = atoms.ge(sv.var, sv.min) {
            lits.push(l); // [x ≥ min]
        }
        if let LitOrConst::Lit(l) = atoms.ge(sv.var, sv.max + 1) {
            lits.push(l.negate()); // ¬[x ≥ max+1]
        }
        for &h in &sv.holes {
            if let LitOrConst::Lit(l) = atoms.eq(sv.var, h) {
                lits.push(l.negate()); // ¬[x = h]
            }
        }
    }
    lits
}

/// Negate a possibly-constant literal.
fn neg(loc: LitOrConst) -> LitOrConst {
    match loc {
        LitOrConst::True => LitOrConst::False,
        LitOrConst::False => LitOrConst::True,
        LitOrConst::Lit(l) => LitOrConst::Lit(l.negate()),
    }
}

/// Translate a primary [`DomEvent`] to the literal it asserts, or `None` if it
/// carries no information (folds to a constant — e.g. `remove_above` at the
/// current upper bound).
fn translate(atoms: &AtomTable, ev: &DomEvent) -> Option<Lit> {
    match *ev {
        DomEvent::GeTrue { var, bound } => match atoms.ge(var, bound) {
            LitOrConst::Lit(l) => Some(l),
            _ => None,
        },
        DomEvent::LeTrue { var, bound } => match atoms.ge(var, bound + 1) {
            LitOrConst::Lit(l) => Some(l.negate()),
            _ => None,
        },
        DomEvent::NeTrue { var, val } => match atoms.eq(var, val) {
            LitOrConst::Lit(l) => Some(l.negate()),
            _ => None,
        },
        DomEvent::EqTrue { var, val } => match atoms.eq(var, val) {
            LitOrConst::Lit(l) => Some(l),
            _ => None,
        },
    }
}

/// The CDCL engine: the FD solver plus the SAT trail and learning state.
pub struct Cdcl<'s> {
    /// The finite-domain solver (domains, trail, propagators). Ground truth.
    pub(crate) solver: &'s mut Solver,
    /// The Boolean atom numbering over the solver's variables.
    pub(crate) atoms: AtomTable,
    /// Assigned literals in chronological order (each literal is *true*).
    trail: Vec<Lit>,
    /// `level_starts[d]` is the `trail` index where decision level `d` begins.
    /// Always non-empty: `level_starts[0] == 0` is the root.
    level_starts: Vec<usize>,
    /// Decision level at which each atom was assigned, or [`UNSET`].
    atom_level: Vec<u32>,
    /// Why each atom was assigned (meaningful only while assigned).
    atom_reason: Vec<Reason>,
    /// Per-atom **trail** truth: what the recorded assignment says, which lags
    /// the domain view during propagation. Unit propagation reads this (not the
    /// view) so it still *records* atoms the domain already determined en masse.
    tval: Vec<Tri>,
    /// Scratch "seen" marks for 1-UIP analysis, indexed by atom; always cleared
    /// back to all-false at the end of each analysis.
    seen: Vec<bool>,
    /// Clause database: channeling axioms, learned, and constraint clauses.
    pub(crate) clauses: ClauseDb,
    /// Two-watched-literal occurrence lists, indexed by literal code. A clause
    /// appears under each of its two watched literals; it is re-examined only
    /// when one of them becomes false.
    watches: Vec<Vec<ClauseRef>>,
    /// Reusable scratch for the pre-step scope snapshot used to build a
    /// propagator's conflict reason (kept across steps to avoid per-step allocs).
    conflict_scope: Vec<ScopeVar>,
    /// Index into `trail` of the next literal whose watches need processing.
    qhead: usize,
    /// VSIDS-style per-variable activity (indexed by `VarId`), bumped for every
    /// variable involved in a conflict so branching is conflict-driven (not just
    /// FD-failure-driven like plain dom/wdeg). Read by the branching heuristic.
    pub(crate) activity: Vec<f64>,
    /// Current activity bump increment (grows each conflict to emulate decay).
    var_inc: f64,
    /// Stop flag polled during propagation so a time limit / Ctrl+C bounds even
    /// a huge root fixpoint (defaults to [`NEVER_STOP`]; set by the driver).
    stop: &'s AtomicBool,
    /// Number of live (non-tombstone) deletable learned clauses.
    num_learned: usize,
    /// Soft cap on live learned clauses; reduction triggers above it, then it
    /// grows so the database is allowed to expand over time.
    max_learned: usize,
    /// Running count of conflicts analysed (for search statistics and restarts).
    pub(crate) conflicts: u64,
}

impl<'s> Cdcl<'s> {
    /// Build the engine over `solver`, allocating atoms from the current (root)
    /// domains. Call before any decision.
    pub fn new(solver: &'s mut Solver) -> Self {
        let nvars = solver.store.num_vars();
        let atoms = AtomTable::build(nvars, |v: VarId| (solver.store.min(v), solver.store.max(v)));
        let n = atoms.num_atoms();
        Cdcl {
            solver,
            atoms,
            trail: Vec::new(),
            level_starts: vec![0],
            atom_level: vec![UNSET; n],
            atom_reason: vec![Reason::Unset; n],
            tval: vec![Tri::Unknown; n],
            seen: vec![false; n],
            clauses: ClauseDb::new(),
            watches: vec![Vec::new(); 2 * n],
            conflict_scope: Vec::new(),
            qhead: 0,
            activity: vec![0.0; nvars],
            var_inc: 1.0,
            stop: &NEVER_STOP,
            num_learned: 0,
            max_learned: 2000,
            conflicts: 0,
        }
    }

    /// Set the stop flag polled during propagation (the search time limit /
    /// Ctrl+C). Call before solving.
    pub(crate) fn set_stop(&mut self, stop: &'s AtomicBool) {
        self.stop = stop;
    }

    /// Whether the stop flag has fired.
    #[inline]
    pub(crate) fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Drop the worst (highest-LBD) deletable learned clauses once they exceed
    /// the soft cap, keeping "glue" clauses (LBD ≤ 2). Must be called at the root
    /// (decision level 0): there, the only assigned atoms are level-0 facts whose
    /// reasons conflict analysis never consults, so a deleted clause can never be
    /// a needed reason. Deletion is by tombstone (clear `lits`, drop from watch
    /// lists), keeping every [`ClauseRef`] valid.
    pub(crate) fn maybe_reduce_db(&mut self) {
        if self.num_learned <= self.max_learned {
            return;
        }
        debug_assert_eq!(self.decision_level(), 0, "reduction off the root");
        let mut learned: Vec<ClauseRef> = (0..self.clauses.len() as u32)
            .map(ClauseRef)
            .filter(|&c| {
                let cl = self.clauses.get(c);
                cl.deletable && cl.lits.len() >= 2
            })
            .collect();
        // Worst (highest LBD) first.
        learned.sort_unstable_by_key(|&c| std::cmp::Reverse(self.clauses.get(c).lbd));
        let target = learned.len() / 2;
        let mut removed = 0;
        for &c in &learned {
            if removed >= target {
                break;
            }
            if self.clauses.get(c).lbd <= 2 {
                continue; // keep glue clauses
            }
            self.delete_clause(c);
            removed += 1;
        }
        self.num_learned -= removed;
        self.max_learned += 500; // let the database grow over time
    }

    /// Tombstone a clause: unwatch it and free its literals. Its slot remains so
    /// outstanding [`ClauseRef`]s stay valid (the dead clause is in no watch list,
    /// so propagation never reaches it).
    fn delete_clause(&mut self, c: ClauseRef) {
        let (w0, w1) = {
            let lits = &self.clauses.get(c).lits;
            (lits[0], lits[1])
        };
        self.watches[w0.code() as usize].retain(|&x| x != c);
        self.watches[w1.code() as usize].retain(|&x| x != c);
        self.clauses.get_mut(c).lits.clear();
    }

    /// The Literal Block Distance of a clause: the number of distinct decision
    /// levels among its assigned literals.
    fn lbd_of(&self, lits: &[Lit]) -> u32 {
        let mut levels: Vec<u32> = lits
            .iter()
            .map(|l| self.level_of(l.atom()))
            .filter(|&lv| lv != UNSET)
            .collect();
        levels.sort_unstable();
        levels.dedup();
        levels.len() as u32
    }

    /// Add a clause to the database and set up its two watched literals (the
    /// first two literals). Clauses with fewer than two literals are not watched
    /// — callers assert their single literal directly. `deletable` marks a 1-UIP
    /// learned clause (eligible for reduction); `lbd` is its Literal Block
    /// Distance (only meaningful when deletable).
    pub(crate) fn add_clause(&mut self, lits: Vec<Lit>, deletable: bool, lbd: u32) -> ClauseRef {
        let (w0, w1) = if lits.len() >= 2 {
            (Some(lits[0]), Some(lits[1]))
        } else {
            (None, None)
        };
        let cref = self.clauses.add(lits, deletable, lbd);
        if let (Some(a), Some(b)) = (w0, w1) {
            self.watches[a.code() as usize].push(cref);
            self.watches[b.code() as usize].push(cref);
        }
        if deletable {
            self.num_learned += 1;
        }
        cref
    }

    /// Post the order-encoding consistency (channeling) clauses for every
    /// variable, and seed root facts. Run once, at the root, before search.
    ///
    /// For each value `v` of `x` over its initial span `[lo, hi]`:
    /// `[x=v] → [x≥v]`, `[x=v] → ¬[x≥v+1]`, `([x≥v] ∧ ¬[x≥v+1]) → [x=v]`, and
    /// the monotonic `[x≥v+1] → [x≥v]`. Constant atoms fold away, so a range
    /// variable gets only the genuinely informative clauses.
    pub fn post_channeling(&mut self) {
        for i in 0..self.solver.store.num_vars() {
            let x = VarId(i as u32);
            let lo = self.solver.store.min(x);
            let hi = self.solver.store.max(x);
            // Bounds-only variables (the objective) get only the monotonic order
            // chain — their equality atoms are never used, so the eq-channel is
            // dead weight (and, for a wide domain, the bulk of the clauses).
            for v in lo..=hi {
                let eqv = self.atoms.eq(x, v);
                let gev = self.atoms.ge(x, v);
                let gev1 = self.atoms.ge(x, v + 1);
                self.post_clause(&[neg(eqv), gev]); // [x=v] → [x≥v]
                self.post_clause(&[neg(eqv), neg(gev1)]); // [x=v] → ¬[x≥v+1]
                self.post_clause(&[neg(gev), gev1, eqv]); // [x≥v] ∧ ¬[x≥v+1] → [x=v]
                self.post_clause(&[neg(gev1), gev]); // [x≥v+1] → [x≥v]
            }
        }
        self.seed_facts();
    }

    /// Post a clause from possibly-constant terms: skip if it is a tautology
    /// (contains `⊤`), drop `⊥` terms, and add the rest. Returns the handle, or
    /// `None` if the clause folded away.
    fn post_clause(&mut self, terms: &[LitOrConst]) -> Option<ClauseRef> {
        let mut lits = Vec::with_capacity(terms.len());
        for &t in terms {
            match t {
                LitOrConst::True => return None, // tautology
                LitOrConst::False => {}          // drop
                LitOrConst::Lit(l) => lits.push(l),
            }
        }
        if lits.is_empty() {
            return None;
        }
        Some(self.add_clause(lits, false, 0)) // channeling axiom: not deletable
    }

    /// Record every atom whose truth is already fixed at the root as a level-0
    /// [`Fact`](Reason::Fact) (e.g. holes in an explicit-set domain). Keeps the
    /// trail a faithful mirror of the domain from the very start.
    fn seed_facts(&mut self) {
        debug_assert_eq!(self.decision_level(), 0);
        for a in 0..self.atoms.num_atoms() as u32 {
            let lit = match view::atom_value(&self.solver.store, self.atoms.decode(a)) {
                Tri::True => Lit::positive(a),
                Tri::False => Lit::negative(a),
                Tri::Unknown => continue,
            };
            self.record(lit, Reason::Fact);
        }
    }

    // --- reads ---

    /// The current decision level (0 = root, before any decision).
    #[inline]
    pub fn decision_level(&self) -> usize {
        self.level_starts.len() - 1
    }

    /// Number of literals currently on the trail.
    #[inline]
    pub fn trail_len(&self) -> usize {
        self.trail.len()
    }

    /// Whether `atom` is currently assigned.
    #[inline]
    pub fn is_assigned(&self, atom: Atom) -> bool {
        self.atom_level[atom as usize] != UNSET
    }

    /// The level `atom` was assigned at (only meaningful while assigned).
    #[inline]
    pub fn level_of(&self, atom: Atom) -> u32 {
        self.atom_level[atom as usize]
    }

    /// The reason `atom` was assigned (only meaningful while assigned).
    #[inline]
    pub fn reason_of(&self, atom: Atom) -> &Reason {
        &self.atom_reason[atom as usize]
    }

    /// A blocking clause over the current decisions: `⋁ ¬d` for each decision
    /// literal `d` on the trail. The decisions plus deterministic propagation
    /// reach a unique assignment, so this excludes exactly the current solution.
    /// Because every literal is a decision (an implication-graph root with no
    /// antecedents), 1-UIP analysis cannot generalize it through propagator
    /// reasons — which is what keeps all-solutions search complete (generalizing
    /// would drop feasible solutions that merely share a propagated consequence).
    /// Empty when the solution was forced at the root (the lone solution).
    pub(crate) fn decision_blocking(&self) -> Vec<Lit> {
        self.trail
            .iter()
            .filter(|l| matches!(self.atom_reason[l.atom() as usize], Reason::Decision))
            .map(|l| l.negate())
            .collect()
    }

    /// The current truth of `lit` in the **domain view**.
    #[inline]
    pub fn value(&self, lit: Lit) -> Tri {
        view::lit_value(&self.solver.store, &self.atoms, lit)
    }

    /// The current truth of `lit` on the **trail** (recorded assignment). Lags
    /// the view during propagation; equal to it at a fixpoint (invariant T2).
    #[inline]
    pub fn tvalue(&self, lit: Lit) -> Tri {
        let t = self.tval[lit.atom() as usize];
        if lit.is_positive() {
            t
        } else {
            t.negate()
        }
    }

    // --- assignment ---

    /// Open a fresh decision level.
    pub fn open_level(&mut self) {
        self.solver.store.push_level();
        self.level_starts.push(self.trail.len());
    }

    /// Record a literal that just became true in the domain, with `reason`, at
    /// the current decision level. No-op if its atom is already assigned.
    fn record(&mut self, lit: Lit, reason: Reason) {
        let a = lit.atom() as usize;
        if self.atom_level[a] != UNSET {
            return;
        }
        debug_assert_eq!(
            view::lit_value(&self.solver.store, &self.atoms, lit),
            Tri::True,
            "recorded literal is not true in the domain"
        );
        self.atom_level[a] = self.decision_level() as u32;
        self.atom_reason[a] = reason;
        self.tval[a] = if lit.is_positive() {
            Tri::True
        } else {
            Tri::False
        };
        self.trail.push(lit);
    }

    /// The currently-recorded atoms of `var`, each as the literal matching its
    /// trail truth. This conjunction is the full *recorded* knowledge of `var`;
    /// since the recorded atoms are the primaries of every op applied to it, they
    /// entail every value the domain determines — so they make a sound antecedent
    /// for any of `var`'s determined-but-unrecorded atoms, and (being recorded)
    /// they bottom out 1-UIP analysis.
    fn ds_recorded(&self, var: VarId) -> Vec<Lit> {
        let (lo, hi) = self.atoms.var_span(var);
        let mut lits = Vec::new();
        let push_if_recorded = |loc: LitOrConst, out: &mut Vec<Lit>| {
            if let LitOrConst::Lit(l) = loc {
                match self.tval[l.atom() as usize] {
                    Tri::True => out.push(Lit::positive(l.atom())),
                    Tri::False => out.push(Lit::negative(l.atom())),
                    Tri::Unknown => {}
                }
            }
        };
        for k in (lo + 1)..=hi {
            push_if_recorded(self.atoms.ge(var, k), &mut lits);
        }
        for v in lo..=hi {
            push_if_recorded(self.atoms.eq(var, v), &mut lits);
        }
        lits
    }

    /// Record every atom of `var` the domain now determines but the trail has
    /// not — keeping the SAT trail a faithful mirror of the domain (invariant
    /// T2) instead of letting it lag. Called eagerly, right after each op changes
    /// `var`, so each atom is recorded at the level it is actually determined
    /// (recording it later, at a deeper level, would let 1-UIP assert its
    /// negation at too low a backjump level and prune feasible solutions).
    ///
    /// The reason is the variable's already-recorded atoms ([`ds_recorded`]),
    /// which entail every determination and are themselves recorded, so analysis
    /// resolves through them soundly.
    fn sync_var(&mut self, var: VarId) {
        let (lo, hi) = self.atoms.var_span(var);
        let mut todo: Vec<Lit> = Vec::new();
        for k in (lo + 1)..=hi {
            if let LitOrConst::Lit(l) = self.atoms.ge(var, k) {
                if !self.is_assigned(l.atom()) {
                    match view::lit_value(&self.solver.store, &self.atoms, l) {
                        Tri::True => todo.push(l),
                        Tri::False => todo.push(l.negate()),
                        Tri::Unknown => {}
                    }
                }
            }
        }
        for v in lo..=hi {
            if let LitOrConst::Lit(l) = self.atoms.eq(var, v) {
                if !self.is_assigned(l.atom()) {
                    match view::lit_value(&self.solver.store, &self.atoms, l) {
                        Tri::True => todo.push(l),
                        Tri::False => todo.push(l.negate()),
                        Tri::Unknown => {}
                    }
                }
            }
        }
        if todo.is_empty() {
            return;
        }
        let ds = self.ds_recorded(var);
        for l in todo {
            self.record(l, Reason::Generic(ds.clone()));
        }
    }

    /// Push `lit` into the domain and record the primary it asserts with
    /// `reason`. Returns `Err` on wipeout (nothing recorded). Secondary atoms
    /// that also flip are recorded later by channeling unit propagation, not
    /// here — an op emits only its primary event.
    pub fn apply_lit(&mut self, lit: Lit, reason: Reason) -> Result<(), Inconsistency> {
        view::apply(&mut self.solver.store, &self.atoms, lit)?;
        let events = std::mem::take(&mut self.solver.store.events);
        for ev in &events {
            if let Some(el) = translate(&self.atoms, &ev.primary) {
                debug_assert_eq!(
                    el.atom(),
                    lit.atom(),
                    "an op emitted an unexpected non-primary event"
                );
                self.record(el, reason.clone());
            }
        }
        let var = self.atoms.var_of(lit.atom());
        self.sync_var(var);
        Ok(())
    }

    /// Make `lit` a branching decision: open a level and assign it.
    pub fn decide(&mut self, lit: Lit) -> Result<(), Inconsistency> {
        self.open_level();
        self.apply_lit(lit, Reason::Decision)
    }

    /// Force `lit` true (clause/unit propagation): enforce it on the domain and
    /// record it with `reason`. `Err` if it conflicts with the domain.
    pub fn assign(&mut self, lit: Lit, reason: Reason) -> Result<(), Inconsistency> {
        if self.tvalue(lit) == Tri::True {
            return Ok(());
        }
        view::apply(&mut self.solver.store, &self.atoms, lit)?;
        self.record(lit, reason);
        // The op's own primary event names `lit`, already recorded.
        self.solver.store.events.clear();
        let var = self.atoms.var_of(lit.atom());
        self.sync_var(var);
        Ok(())
    }

    /// Unit-propagate the clause database with the two-watched-literal scheme,
    /// processing newly-assigned trail literals from `qhead`. Forces the last
    /// open literal of every clause that becomes unit (recording channeled and
    /// learned consequences), and returns the conflicting clause if one is fully
    /// falsified.
    pub fn propagate_clauses(&mut self) -> Result<(), ClauseRef> {
        while self.qhead < self.trail.len() {
            // Bound a huge root channeling pass against the time limit.
            if self.stopped() {
                return Ok(());
            }
            let q = self.trail[self.qhead];
            self.qhead += 1;
            // Clauses watching `neg` (now false) may need a new watch or to fire.
            let neg = q.negate();
            let mut ws = std::mem::take(&mut self.watches[neg.code() as usize]);
            let mut keep = 0usize; // clauses that still watch `neg`
            let mut read = 0usize;
            let mut conflict: Option<ClauseRef> = None;

            while read < ws.len() {
                let cref = ws[read];
                read += 1;
                // Keep the watched literal we are updating in slot 1.
                {
                    let lits = &mut self.clauses.get_mut(cref).lits;
                    if lits[0] == neg {
                        lits.swap(0, 1);
                    }
                }
                let other = self.clauses.get(cref).lits[0];
                if self.tvalue(other) == Tri::True {
                    ws[keep] = cref; // clause already satisfied; keep watching `neg`
                    keep += 1;
                    continue;
                }
                // Hunt for a non-false literal to watch instead of `neg`.
                let mut new_watch: Option<(usize, Lit)> = None;
                {
                    let lits = &self.clauses.get(cref).lits;
                    #[allow(clippy::needless_range_loop)] // need the index `k` for the swap below
                    for k in 2..lits.len() {
                        if self.tvalue(lits[k]) != Tri::False {
                            new_watch = Some((k, lits[k]));
                            break;
                        }
                    }
                }
                if let Some((k, wl)) = new_watch {
                    self.clauses.get_mut(cref).lits.swap(1, k);
                    self.watches[wl.code() as usize].push(cref);
                    continue; // no longer watching `neg`
                }
                // No replacement: the clause is unit on `other` (or a conflict).
                ws[keep] = cref; // still watching `neg`
                keep += 1;
                if self.tvalue(other) == Tri::False
                    || self.assign(other, Reason::Clause(cref)).is_err()
                {
                    // Falsified clause (or asserting `other` wiped a domain).
                    while read < ws.len() {
                        ws[keep] = ws[read];
                        keep += 1;
                        read += 1;
                    }
                    conflict = Some(cref);
                    break;
                }
            }

            ws.truncate(keep);
            self.watches[neg.code() as usize] = ws;
            if let Some(c) = conflict {
                return Err(c);
            }
        }
        Ok(())
    }

    /// Propagate to a joint fixpoint of channeling/clause unit propagation and
    /// the FD propagators, in lockstep: clause propagation records channeled and
    /// learned consequences, FD propagation records each propagator's inferences
    /// (with [`Generic`](Reason::Generic) reasons), and forced domain changes on
    /// either side feed the other until nothing new is derived. Returns the
    /// first [`Conflict`] found.
    pub fn propagate(&mut self) -> Result<(), Conflict> {
        loop {
            // Honour the stop flag even mid-fixpoint: a huge root propagation
            // must not blow past the time limit. Abandon as a (non-)fixpoint —
            // the driver's own stop check then ends the search at once.
            if self.stopped() {
                self.solver.store.clear_queue();
                return Ok(());
            }
            // Channel + clause unit propagation to a fixpoint *first*, so every
            // earlier change (including the previous propagator's) is fully
            // recorded on the trail before the next propagator is snapshotted —
            // this keeps a propagator's `D_S` antecedents all trail-recorded.
            if let Err(c) = self.propagate_clauses() {
                return Err(Conflict::Clause(c));
            }
            // Then run a single FD propagator and record its inferences.
            let Some(pid) = self.solver.peek_prop() else {
                return Ok(()); // FD queue drained and clauses at fixpoint
            };
            // Snapshot its scope first (into the reused buffer), so a failure
            // has a sound (trail-consistent) conflict reason.
            {
                let Cdcl {
                    solver,
                    conflict_scope,
                    ..
                } = self;
                solver.store.snapshot_scope_into(pid, conflict_scope);
            }
            match self.solver.propagate_step() {
                StepOutcome::Empty => return Ok(()),
                StepOutcome::Ran(_) => self.record_prop_events()?,
                StepOutcome::Failed(_) => {
                    self.solver.store.events.clear();
                    return Err(Conflict::Generic(build_ds(
                        &self.atoms,
                        &self.conflict_scope,
                    )));
                }
            }
        }
    }

    /// Drain a just-run propagator's primary events, recording each inferred
    /// literal with a [`Generic`](Reason::Generic) reason built from the pre-op
    /// scope snapshot the store captured.
    ///
    /// Channeling is run after *each* event so that an op's secondary atom flips
    /// (e.g. a bound shift) are recorded before the next op's reason is built —
    /// keeping every `D_S` antecedent strictly earlier on the trail than the
    /// literal it explains (the invariant 1-UIP analysis relies on).
    fn record_prop_events(&mut self) -> Result<(), Conflict> {
        let events = std::mem::take(&mut self.solver.store.events);
        for ev in &events {
            let var = match ev.primary {
                DomEvent::GeTrue { var, .. }
                | DomEvent::LeTrue { var, .. }
                | DomEvent::NeTrue { var, .. }
                | DomEvent::EqTrue { var, .. } => var,
            };
            if let Some(lit) = translate(&self.atoms, &ev.primary) {
                if !self.is_assigned(lit.atom()) {
                    let ds = build_ds(&self.atoms, &ev.scope);
                    self.record(lit, Reason::Generic(ds));
                }
            }
            // Mirror the rest of this variable's bulk domain change onto the
            // trail at this (its determination) level before moving on.
            self.sync_var(var);
            self.propagate_clauses().map_err(Conflict::Clause)?;
        }
        Ok(())
    }

    /// Propagate; on each conflict, learn a 1-UIP clause, backjump, and assert
    /// it. Repeats until a clean fixpoint (`true`) or a conflict at the root
    /// proving the problem unsatisfiable (`false`).
    pub fn propagate_and_learn(&mut self) -> bool {
        loop {
            match self.propagate() {
                Ok(()) => return true,
                Err(conflict) => {
                    if !self.resolve_conflict(conflict) {
                        return false;
                    }
                }
            }
        }
    }

    /// Analyse `conflict`, backjump, learn the 1-UIP clause and assert it.
    /// Returns `false` if the conflict proves unsatisfiability. Used both by the
    /// propagation loop and to inject an externally-detected conflict (a fully
    /// falsified blocking clause after enumerating a solution).
    pub(crate) fn resolve_conflict(&mut self, conflict: Conflict) -> bool {
        self.conflicts += 1;
        let Some((learnt, btlevel)) = self.analyze(conflict) else {
            return false; // conflict independent of all decisions: UNSAT
        };
        // LBD is measured at learn time, while every literal is still assigned.
        let lbd = self.lbd_of(&learnt);
        let deletable = learnt.len() >= 2;
        self.backjump_to(btlevel);
        let asserting = learnt[0];
        let cref = self.add_clause(learnt, deletable, lbd);
        // After backjumping, the asserting literal is the clause's only open
        // literal; asserting it cannot wipe out a domain.
        !(self.assign(asserting, Reason::Clause(cref)).is_err() && btlevel == 0)
    }

    /// First-UIP conflict analysis. Returns the learned clause (its asserting
    /// literal first) and the level to backjump to (the second-highest level in
    /// the clause, or 0 for a unit clause).
    fn analyze(&mut self, conflict: Conflict) -> Option<(Vec<Lit>, usize)> {
        let mut clause_lits = self.conflict_lits(&conflict);
        // Analyse at the conflict's deepest decision level. This is usually the
        // current level, but a propagator can fail on literals all below it (a
        // late-detected conflict) — resolving at the deepest level still yields
        // a valid 1-UIP clause and a sound (possibly far) backjump. If every
        // literal is at the root (level 0), the conflict holds unconditionally:
        // the problem is unsatisfiable.
        let level = clause_lits
            .iter()
            .map(|l| self.level_of(l.atom()))
            .filter(|&lv| lv != UNSET)
            .max()
            .unwrap_or(0);
        if level == 0 {
            return None;
        }
        let mut learnt: Vec<Lit> = vec![Lit::positive(0)]; // slot 0 = asserting lit
        let mut touched: Vec<u32> = Vec::new();
        let mut counter = 0usize; // level-`level` literals not yet resolved
        let mut index = self.trail.len();
        let mut uip;

        loop {
            for q in clause_lits {
                let a = q.atom();
                let lv = self.level_of(a);
                if !self.seen[a as usize] && lv != UNSET && lv > 0 {
                    self.seen[a as usize] = true;
                    touched.push(a);
                    self.bump_var(self.atoms.var_of(a)); // VSIDS: this var is conflict-active
                    if lv == level {
                        counter += 1;
                    } else {
                        learnt.push(q); // a lower-level literal of the learned clause
                    }
                }
            }
            // The latest still-seen trail literal at the current level is next.
            loop {
                index -= 1;
                if self.seen[self.trail[index].atom() as usize] {
                    break;
                }
            }
            uip = self.trail[index];
            self.seen[uip.atom() as usize] = false;
            counter -= 1;
            if counter == 0 {
                break; // single remaining current-level literal: the UIP
            }
            clause_lits = self.reason_lits(uip);
        }

        learnt[0] = uip.negate(); // the asserting literal
        let btlevel = learnt[1..]
            .iter()
            .map(|l| self.level_of(l.atom()) as usize)
            .max()
            .unwrap_or(0);
        for a in touched {
            self.seen[a as usize] = false;
        }
        self.decay_activity();
        Some((learnt, btlevel))
    }

    /// Bump a variable's VSIDS activity, rescaling all activities if it overflows.
    fn bump_var(&mut self, var: VarId) {
        self.activity[var.index()] += self.var_inc;
        if self.activity[var.index()] > 1e100 {
            for a in &mut self.activity {
                *a *= 1e-100;
            }
            self.var_inc *= 1e-100;
        }
    }

    /// Grow the bump increment so later conflicts weigh more (geometric decay).
    fn decay_activity(&mut self) {
        self.var_inc *= 1.0 / 0.95;
    }

    /// Set up the root: post channeling clauses, enqueue every propagator, and
    /// propagate to the root fixpoint. Returns `false` if the root is already
    /// unsatisfiable.
    pub fn init(&mut self) -> bool {
        self.post_channeling();
        self.solver.enqueue_all();
        self.propagate_and_learn()
    }

    /// A decision literal `[x = min(x)]` for the first unfixed variable in
    /// `vars`, or `None` if all are fixed. (A simple heuristic; the real driver
    /// uses dom/wdeg.)
    fn pick_decision(&self, vars: &[VarId]) -> Option<Lit> {
        for &v in vars {
            if !self.solver.store.is_fixed(v) {
                let val = self.solver.store.min(v);
                if let LitOrConst::Lit(l) = self.atoms.eq(v, val) {
                    return Some(l);
                }
            }
        }
        None
    }

    /// Find one solution over `vars` by CDCL, or `None` if unsatisfiable.
    /// Assumes [`init`](Self::init) already succeeded.
    pub fn solve_first(&mut self, vars: &[VarId]) -> Option<Vec<i32>> {
        loop {
            match self.pick_decision(vars) {
                None => {
                    return Some(vars.iter().map(|&v| self.solver.store.value(v)).collect());
                }
                Some(lit) => {
                    self.decide(lit).expect("in-domain decision cannot fail");
                    if !self.propagate_and_learn() {
                        return None;
                    }
                }
            }
        }
    }

    /// The literals of the initial conflict clause (all currently false).
    fn conflict_lits(&self, conflict: &Conflict) -> Vec<Lit> {
        match conflict {
            Conflict::Clause(c) => self.clauses.get(*c).lits.clone(),
            Conflict::Generic(ds) => ds.iter().map(|l| l.negate()).collect(),
        }
    }

    /// The literals of `p`'s reason clause other than `p` (all currently false),
    /// i.e. the antecedents to resolve against in analysis.
    fn reason_lits(&self, p: Lit) -> Vec<Lit> {
        match self.reason_of(p.atom()) {
            Reason::Clause(c) => {
                let c = *c;
                self.clauses
                    .get(c)
                    .lits
                    .iter()
                    .copied()
                    .filter(|&l| l != p)
                    .collect()
            }
            // Reason clause is (p ∨ ⋁ ¬D_S); the antecedents are ¬D_S.
            Reason::Generic(ds) => ds.iter().map(|l| l.negate()).collect(),
            // Decisions/facts have no antecedents (and are never resolved here).
            Reason::Decision | Reason::Fact | Reason::Unset => Vec::new(),
        }
    }

    /// Backjump to decision level `d`, undoing every assignment above it. Clears
    /// the side tables for the popped atoms and pops the matching number of
    /// domain-trail levels, so the domain (and thus every atom's truth) is
    /// restored to its level-`d` state.
    pub fn backjump_to(&mut self, d: usize) {
        let cur = self.decision_level();
        if d >= cur {
            return;
        }
        let keep = self.level_starts[d + 1];
        for lit in self.trail.drain(keep..) {
            let a = lit.atom() as usize;
            self.atom_level[a] = UNSET;
            self.atom_reason[a] = Reason::Unset;
            self.tval[a] = Tri::Unknown;
        }
        self.level_starts.truncate(d + 1);
        for _ in 0..(cur - d) {
            self.solver.store.pop_level();
        }
        // Everything still on the trail was propagated before the backjump.
        self.qhead = self.trail.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcg::lit::LitOrConst;
    use crate::store::Solver;

    fn lit(loc: LitOrConst) -> Lit {
        match loc {
            LitOrConst::Lit(l) => l,
            _ => panic!("expected a real literal"),
        }
    }

    #[test]
    fn levels_track_decisions() {
        let mut solver = Solver::new();
        let _x = solver.new_var_range(0, 9);
        let mut cdcl = Cdcl::new(&mut solver);
        assert_eq!(cdcl.decision_level(), 0);
        cdcl.open_level();
        assert_eq!(cdcl.decision_level(), 1);
        cdcl.open_level();
        assert_eq!(cdcl.decision_level(), 2);
        cdcl.backjump_to(0);
        assert_eq!(cdcl.decision_level(), 0);
    }

    #[test]
    fn backjump_restores_domain_and_clears_atoms() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 9);
        let y = solver.new_var_range(0, 9);
        let base = solver.store.level();
        let mut cdcl = Cdcl::new(&mut solver);
        let ge5 = lit(cdcl.atoms.ge(x, 5));
        let eq3y = lit(cdcl.atoms.eq(y, 3));

        // Decide x >= 5 at level 1.
        cdcl.decide(ge5).unwrap();
        assert_eq!(cdcl.decision_level(), 1);
        assert!(cdcl.is_assigned(ge5.atom()));
        assert_eq!(cdcl.level_of(ge5.atom()), 1);
        assert_eq!(cdcl.solver.store.min(x), 5);

        // Decide y = 3 at level 2.
        cdcl.decide(eq3y).unwrap();
        assert_eq!(cdcl.decision_level(), 2);
        assert!(cdcl.is_assigned(eq3y.atom()));
        assert!(cdcl.solver.store.is_fixed(y));

        // Backjump to level 1: undoes the y decision, keeps the x decision.
        cdcl.backjump_to(1);
        assert_eq!(cdcl.decision_level(), 1);
        assert!(cdcl.is_assigned(ge5.atom())); // kept
        assert!(!cdcl.is_assigned(eq3y.atom())); // cleared
        assert_eq!(cdcl.solver.store.min(x), 5); // x still bounded
        assert_eq!(cdcl.solver.store.size(y), 10); // y fully restored
        assert_eq!(cdcl.value(eq3y), Tri::Unknown); // view agrees it's open again

        // Backjump to root: everything restored, store level back to base.
        cdcl.backjump_to(0);
        assert_eq!(cdcl.decision_level(), 0);
        assert!(!cdcl.is_assigned(ge5.atom()));
        assert_eq!(cdcl.solver.store.size(x), 10);
        assert_eq!(cdcl.solver.store.level(), base);
        assert_eq!(cdcl.trail_len(), 0);
    }

    #[test]
    fn idempotent_assignment_is_not_double_logged() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 9);
        let mut cdcl = Cdcl::new(&mut solver);
        let ge5 = lit(cdcl.atoms.ge(x, 5));
        cdcl.open_level();
        cdcl.apply_lit(ge5, Reason::Decision).unwrap();
        let len = cdcl.trail_len();
        // Re-applying the same bound changes nothing and must not re-log.
        cdcl.apply_lit(ge5, Reason::Decision).unwrap();
        assert_eq!(cdcl.trail_len(), len);
    }

    #[test]
    fn apply_lit_records_translated_primary() {
        // Each op kind round-trips on its own variable: the literal applied is
        // exactly the one recorded (from draining the store's primary event).
        let mut solver = Solver::new();
        let a = solver.new_var_range(0, 9);
        let b = solver.new_var_range(0, 9);
        let c = solver.new_var_range(0, 9);
        let mut cdcl = Cdcl::new(&mut solver);
        let cases = [
            lit(cdcl.atoms.ge(a, 4)),          // remove_below: [a>=4]
            lit(cdcl.atoms.ge(b, 7)).negate(), // remove_above: ¬[b>=7]  (b<=6)
            lit(cdcl.atoms.eq(c, 2)).negate(), // remove:       ¬[c=2]
        ];
        for (i, &l) in cases.iter().enumerate() {
            cdcl.open_level();
            cdcl.apply_lit(l, Reason::Decision).unwrap();
            assert!(cdcl.is_assigned(l.atom()), "case {i} not recorded");
            assert_eq!(cdcl.value(l), Tri::True, "case {i} not true in view");
        }
        // The store's event buffer is fully drained by apply_lit.
        assert!(cdcl.solver.store.events.is_empty());
    }

    #[test]
    fn fix_records_equality_primary() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 5);
        let mut cdcl = Cdcl::new(&mut solver);
        let eq3 = lit(cdcl.atoms.eq(x, 3));
        cdcl.open_level();
        cdcl.apply_lit(eq3, Reason::Decision).unwrap();
        assert!(cdcl.is_assigned(eq3.atom()));
        assert!(cdcl.solver.store.is_fixed(x));
    }

    #[test]
    fn wipeout_is_reported() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(5, 5); // singleton
        let mut cdcl = Cdcl::new(&mut solver);
        let eq5 = lit(cdcl.atoms.eq(x, 5));
        cdcl.open_level();
        // Removing the only value wipes the domain.
        assert!(cdcl.apply_lit(eq5.negate(), Reason::Decision).is_err());
    }

    /// T2: the SAT trail mirrors the domain — an atom is recorded iff its view
    /// truth is determined.
    fn assert_t2(cdcl: &Cdcl) {
        for a in 0..cdcl.atoms.num_atoms() as u32 {
            let determined =
                view::atom_value(&cdcl.solver.store, cdcl.atoms.decode(a)) != Tri::Unknown;
            assert_eq!(
                determined,
                cdcl.is_assigned(a),
                "atom {a} ({:?}): view-determined={determined} but is_assigned={}",
                cdcl.atoms.decode(a),
                cdcl.is_assigned(a)
            );
        }
    }

    #[test]
    fn channeling_records_bound_secondaries() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 5);
        let mut cdcl = Cdcl::new(&mut solver);
        cdcl.post_channeling();
        assert_t2(&cdcl); // nothing determined yet for a clean range var

        cdcl.decide(lit(cdcl.atoms.ge(x, 3))).unwrap(); // x >= 3
        cdcl.propagate_clauses().unwrap();

        // Lower equalities and weaker order atoms are now recorded as secondaries.
        for v in 0..3 {
            assert!(
                cdcl.is_assigned(lit(cdcl.atoms.eq(x, v)).atom()),
                "[x={v}] unset"
            );
        }
        for k in 1..=3 {
            assert!(
                cdcl.is_assigned(lit(cdcl.atoms.ge(x, k)).atom()),
                "[x>={k}] unset"
            );
        }
        assert_t2(&cdcl);
    }

    #[test]
    fn channeling_collapses_on_fix() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 5);
        let mut cdcl = Cdcl::new(&mut solver);
        cdcl.post_channeling();

        cdcl.decide(lit(cdcl.atoms.eq(x, 3))).unwrap(); // fix x = 3
        cdcl.propagate_clauses().unwrap();
        // Fixing forces every order atom and every other equality atom.
        assert_t2(&cdcl);
        assert_eq!(cdcl.solver.store.value(x), 3);
    }

    #[test]
    fn seed_facts_marks_initial_holes() {
        let mut solver = Solver::new();
        let x = solver.new_var_set(&[0, 2, 5]); // holes at 1, 3, 4
        let mut cdcl = Cdcl::new(&mut solver);
        cdcl.post_channeling();
        for v in [1, 3, 4] {
            assert!(
                cdcl.is_assigned(lit(cdcl.atoms.eq(x, v)).atom()),
                "hole [x={v}] not seeded"
            );
            assert_eq!(cdcl.value(lit(cdcl.atoms.eq(x, v))), Tri::False);
        }
        assert_t2(&cdcl);
    }

    #[test]
    fn constraint_clause_can_conflict() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 5);
        let mut cdcl = Cdcl::new(&mut solver);
        cdcl.post_channeling();
        // A constraint clause ([x=0] ∨ [x=1]) that fixing x=3 will falsify.
        let c0 = lit(cdcl.atoms.eq(x, 0));
        let c1 = lit(cdcl.atoms.eq(x, 1));
        cdcl.add_clause(vec![c0, c1], false, 0);

        cdcl.decide(lit(cdcl.atoms.eq(x, 3))).unwrap();
        assert!(cdcl.propagate_clauses().is_err());
    }

    use crate::constraints::primitives::{less_or_equal, less_than};

    /// Run an engine over `solver` to its root fixpoint with channeling posted.
    fn engine_at_root(solver: &mut Solver) -> Cdcl<'_> {
        solver.enqueue_all();
        let mut cdcl = Cdcl::new(solver);
        cdcl.post_channeling();
        cdcl.propagate().expect("root should be consistent");
        cdcl
    }

    #[test]
    fn unified_propagation_runs_fd_propagators() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 9);
        let y = solver.new_var_range(0, 9);
        less_or_equal(&mut solver, x, y); // x <= y
        let mut cdcl = engine_at_root(&mut solver);

        // Decide y <= 4; the FD propagator must drive x <= 4, and the trail must
        // mirror the domain afterwards.
        let le4 = lit(cdcl.atoms.ge(y, 5)).negate();
        cdcl.decide(le4).unwrap();
        cdcl.propagate().unwrap();
        assert_eq!(cdcl.solver.store.max(x), 4);
        assert_t2(&cdcl);
    }

    #[test]
    fn unified_propagation_matches_plain_fd() {
        // Root pruning of x < y over [0,9] must equal the plain FD fixpoint.
        let mut plain = Solver::new();
        let px = plain.new_var_range(0, 9);
        let py = plain.new_var_range(0, 9);
        less_than(&mut plain, px, py);
        plain.store.push_level();
        plain.enqueue_all();
        plain.propagate().unwrap();
        let (pxmin, pxmax, pymin, pymax) = (
            plain.store.min(px),
            plain.store.max(px),
            plain.store.min(py),
            plain.store.max(py),
        );

        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 9);
        let y = solver.new_var_range(0, 9);
        less_than(&mut solver, x, y);
        let cdcl = engine_at_root(&mut solver);
        assert_eq!(
            (
                cdcl.solver.store.min(x),
                cdcl.solver.store.max(x),
                cdcl.solver.store.min(y),
                cdcl.solver.store.max(y)
            ),
            (pxmin, pxmax, pymin, pymax)
        );
        assert_t2(&cdcl);
    }

    #[test]
    fn unified_propagation_detects_fd_conflict() {
        let mut solver = Solver::new();
        let x = solver.new_var_range(0, 9);
        let y = solver.new_var_range(0, 9);
        less_than(&mut solver, x, y); // x < y
        less_than(&mut solver, y, x); // y < x  — jointly infeasible
        solver.enqueue_all();
        let mut cdcl = Cdcl::new(&mut solver);
        cdcl.post_channeling();
        assert!(cdcl.propagate().is_err());
    }

    use crate::constraints::primitives::not_equal_offset;

    #[test]
    fn cdcl_solves_4_queens() {
        let n = 4;
        let mut solver = Solver::new();
        let q: Vec<VarId> = (0..n).map(|_| solver.new_var_range(0, n - 1)).collect();
        for i in 0..n as usize {
            for j in (i + 1)..n as usize {
                let (di, dj) = (i as i32, j as i32);
                not_equal_offset(&mut solver, q[i], q[j], 0);
                not_equal_offset(&mut solver, q[i], q[j], di - dj);
                not_equal_offset(&mut solver, q[i], q[j], dj - di);
            }
        }
        let mut cdcl = Cdcl::new(&mut solver);
        assert!(cdcl.init());
        let sol = cdcl.solve_first(&q).expect("4-queens is satisfiable");

        // Validate: a permutation with no shared diagonal.
        for i in 0..q.len() {
            for j in (i + 1)..q.len() {
                assert_ne!(sol[i], sol[j], "same column");
                assert_ne!(
                    (sol[i] - sol[j]).abs(),
                    (i as i32 - j as i32).abs(),
                    "same diagonal"
                );
            }
        }
    }

    #[test]
    fn cdcl_detects_pigeonhole_unsat() {
        // Three pairwise-distinct variables over {0,1} cannot be placed.
        let mut solver = Solver::new();
        let v: Vec<VarId> = (0..3).map(|_| solver.new_var_range(0, 1)).collect();
        for i in 0..3 {
            for j in (i + 1)..3 {
                not_equal_offset(&mut solver, v[i], v[j], 0);
            }
        }
        let mut cdcl = Cdcl::new(&mut solver);
        if !cdcl.init() {
            return; // proven UNSAT already at the root
        }
        assert!(cdcl.solve_first(&v).is_none(), "pigeonhole must be UNSAT");
    }
}
