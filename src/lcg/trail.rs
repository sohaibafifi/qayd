//! The SAT trail and [`Cdcl`] engine state layered on the FD store.
//!
//! Atom truth is read from the domain via [`view`](crate::lcg::view); the trail
//! records only assignment order and reasons. Backjump truncates the trail and
//! pops the matching domain-trail levels, rolling truth back with it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::ids::VarId;
use crate::lcg::clause::{ClauseDb, ClauseRef, ClauseSharing, SharedClause};
use crate::lcg::lit::{Atom, AtomTable, Lit, LitOrConst};
use crate::lcg::view::{self, Tri};
use crate::propagator::Inconsistency;
use crate::store::{Cause, DomEvent, Premise, ScopeVar, Solver, StepOutcome};

/// Stop flag that is never set; default until [`Cdcl::set_stop`] supplies the real one.
pub(crate) static NEVER_STOP: AtomicBool = AtomicBool::new(false);

/// Sentinel `atom_level` for an unassigned atom.
pub const UNSET: u32 = u32::MAX;

/// Why an atom became assigned. Drives 1-UIP conflict analysis.
#[derive(Clone)]
pub enum Reason {
    /// Not currently assigned.
    Unset,
    /// Level-0 fact, no antecedent; dropped during analysis.
    Fact,
    /// A branching decision; analysis stops here.
    Decision,
    /// Forced as the last open literal of a clause.
    Clause(ClauseRef),
    /// Propagator inference. Literals are `D_S` (true scope literals entailing
    /// the inference); reason clause is `(consequent ∨ ⋁ ¬D_S)`.
    Generic(Vec<Lit>),
}

/// A detected conflict, before analysis turns it into a learned clause.
#[derive(Clone, Debug)]
pub enum Conflict {
    /// A fully falsified clause.
    Clause(ClauseRef),
    /// Propagator failure: these `D_S` literals (all true) are jointly
    /// infeasible; conflict clause is `⋁ ¬D_S`.
    Generic(Vec<Lit>),
}

/// Build `D_S`: the true literals describing a scope's domains
/// (`[x≥min]`, `¬[x≥max+1]`, `¬[x=v]` per hole).
fn build_ds(atoms: &AtomTable, scope: &[ScopeVar]) -> Vec<Lit> {
    let mut lits = Vec::new();
    for sv in scope {
        if let LitOrConst::Lit(l) = atoms.ge(sv.var, sv.min) {
            lits.push(l);
        }
        if let LitOrConst::Lit(l) = atoms.ge(sv.var, sv.max + 1) {
            lits.push(l.negate());
        }
        for &h in &sv.holes {
            if let LitOrConst::Lit(l) = atoms.eq(sv.var, h) {
                lits.push(l.negate());
            }
        }
    }
    lits
}

/// Translate a tight [`Premise`] conjunction into the true literals that entail
/// the inference (constants fold away).
fn translate_premises(atoms: &AtomTable, premises: &[Premise]) -> Vec<Lit> {
    let mut lits = Vec::with_capacity(premises.len());
    let push = |loc: LitOrConst, out: &mut Vec<Lit>| {
        if let LitOrConst::Lit(l) = loc {
            out.push(l);
        }
    };
    for &p in premises {
        match p {
            Premise::Ge { var, bound } => push(atoms.ge(var, bound), &mut lits),
            Premise::Le { var, bound } => push(neg(atoms.ge(var, bound + 1)), &mut lits),
            // Express `[x = v]` as order atoms `[x ≥ v] ∧ [x ≤ v]`, never the
            // positive equality atom: the eq atom may carry a deeper level, which
            // would make 1-UIP backjump too far and prune feasible solutions.
            Premise::Eq { var, val } => {
                push(atoms.ge(var, val), &mut lits);
                push(neg(atoms.ge(var, val + 1)), &mut lits);
            }
            Premise::Ne { var, val } => push(neg(atoms.eq(var, val)), &mut lits),
        }
    }
    lits
}

/// Antecedent literals for an event's [`Cause`]: tight premises if supplied,
/// else the pre-propagator whole-scope snapshot.
fn cause_lits(atoms: &AtomTable, cause: &Cause, fallback: &[ScopeVar]) -> Vec<Lit> {
    match cause {
        Cause::Premises(p) => translate_premises(atoms, p),
        Cause::Scope => build_ds(atoms, fallback),
    }
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
/// folds to a constant.
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
    /// The FD solver (domains, trail, propagators); ground truth.
    pub(crate) solver: &'s mut Solver,
    /// Boolean atom numbering over the solver's variables.
    pub(crate) atoms: AtomTable,
    /// Assigned literals in chronological order (each is true).
    trail: Vec<Lit>,
    /// `level_starts[d]` = `trail` index where level `d` begins. Always non-empty;
    /// `level_starts[0] == 0` is the root.
    level_starts: Vec<usize>,
    /// Assignment level per atom, or [`UNSET`].
    atom_level: Vec<u32>,
    /// Trail position per assigned atom, or `usize::MAX`. Used to reject an
    /// explanation that depends on an intermediate mutation from the same
    /// propagator call.
    atom_trail_pos: Vec<usize>,
    /// Assignment reason per atom (meaningful only while assigned).
    atom_reason: Vec<Reason>,
    /// Per-atom trail truth (the recorded assignment), which lags the domain view
    /// during propagation; unit propagation reads this, not the view.
    tval: Vec<Tri>,
    /// Scratch "seen" marks for 1-UIP analysis; always cleared after each analysis.
    seen: Vec<bool>,
    /// Clause database: channeling axioms, learned, and constraint clauses.
    pub(crate) clauses: ClauseDb,
    /// Two-watched-literal occurrence lists, indexed by literal code.
    watches: Vec<Vec<ClauseRef>>,
    /// Reusable scratch for the pre-step scope snapshot of a conflict reason.
    conflict_scope: Vec<ScopeVar>,
    /// Reusable scratch for variables touched by one buffered FD step.
    changed_vars: Vec<VarId>,
    /// Index into `trail` of the next literal whose watches need processing.
    qhead: usize,
    /// VSIDS per-variable activity (indexed by `VarId`); read by the branching heuristic.
    pub(crate) activity: Vec<f64>,
    /// Activity bump increment (grows each conflict to emulate decay).
    var_inc: f64,
    /// Stop flag polled during propagation (defaults to [`NEVER_STOP`]).
    stop: &'s AtomicBool,
    /// Reproducible search diversification seed.
    pub(crate) seed: u64,
    /// Live (non-tombstone) deletable learned clauses.
    num_learned: usize,
    /// Soft cap on live learned clauses; grows after each reduction.
    max_learned: usize,
    /// Conflicts analysed (statistics and restarts).
    pub(crate) conflicts: u64,
    /// Total literals across all learned clauses (numerator of avg learned size).
    pub(crate) learned_lits: u64,
    /// Optional portfolio clause exchange.
    clause_sharing: Option<ClauseSharing>,
    /// Reused buffer for newly published clauses.
    shared_scratch: Vec<SharedClause>,
    /// Negated job assumptions prepended to exported learned clauses.
    cube_scope: Vec<Lit>,
    /// Negated objective bound prepended to exported learned clauses.
    bound_scope: Option<Lit>,
}

impl<'s> Cdcl<'s> {
    /// Build the engine over `solver`, allocating atoms from the root domains.
    pub fn new(solver: &'s mut Solver) -> Self {
        // Ablation hook: `QAYD_SCOPE_REASONS` forces whole-scope explanations.
        solver
            .store
            .set_force_scope_reasons(std::env::var_os("QAYD_SCOPE_REASONS").is_some());
        let nvars = solver.store.num_vars();
        let atoms = AtomTable::build(nvars, |v: VarId| (solver.store.min(v), solver.store.max(v)));
        let n = atoms.num_atoms();
        Cdcl {
            solver,
            atoms,
            trail: Vec::new(),
            level_starts: vec![0],
            atom_level: vec![UNSET; n],
            atom_trail_pos: vec![usize::MAX; n],
            atom_reason: vec![Reason::Unset; n],
            tval: vec![Tri::Unknown; n],
            seen: vec![false; n],
            clauses: ClauseDb::new(),
            watches: vec![Vec::new(); 2 * n],
            conflict_scope: Vec::new(),
            changed_vars: Vec::new(),
            qhead: 0,
            activity: vec![0.0; nvars],
            var_inc: 1.0,
            stop: &NEVER_STOP,
            seed: 0,
            num_learned: 0,
            max_learned: 2000,
            conflicts: 0,
            learned_lits: 0,
            clause_sharing: None,
            shared_scratch: Vec::new(),
            cube_scope: Vec::new(),
            bound_scope: None,
        }
    }

    /// Set the stop flag polled during propagation. Call before solving.
    pub(crate) fn set_stop(&mut self, stop: &'s AtomicBool) {
        self.stop = stop;
    }

    /// Set the reproducible search diversification seed.
    pub(crate) fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Enable learned-clause exchange for one portfolio worker.
    pub(crate) fn set_clause_sharing(&mut self, sharing: ClauseSharing) {
        self.clause_sharing = Some(sharing);
    }

    /// Scope exported clauses to the current search cube.
    pub(crate) fn set_cube_scope(&mut self, cube: &[Lit]) {
        self.cube_scope.clear();
        self.cube_scope
            .extend(cube.iter().map(|assumption| assumption.negate()));
    }

    /// Scope exported clauses to the current objective bound.
    pub(crate) fn set_bound_scope(&mut self, bound: Lit) {
        self.bound_scope = Some(bound.negate());
    }

    /// Whether the stop flag has fired.
    #[inline]
    pub(crate) fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Drop the worst (highest-LBD) deletable learned clauses above the soft cap,
    /// keeping glue clauses (LBD ≤ 2). Must run at the root (level 0), where no
    /// deleted clause can be a needed reason. Deletion is by tombstone.
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
                continue;
            }
            self.delete_clause(c);
            removed += 1;
        }
        self.num_learned -= removed;
        self.max_learned += 500;
    }

    /// Tombstone a clause: unwatch it and free its literals; its slot remains so
    /// outstanding [`ClauseRef`]s stay valid.
    fn delete_clause(&mut self, c: ClauseRef) {
        let (w0, w1) = {
            let clause = self.clauses.get(c);
            (clause.lits[clause.watch[0]], clause.lits[clause.watch[1]])
        };
        self.watches[w0.code() as usize].retain(|&x| x != c);
        self.watches[w1.code() as usize].retain(|&x| x != c);
        self.clauses.get_mut(c).lits = Arc::from([]);
    }

    /// Literal Block Distance: distinct decision levels among assigned literals.
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

    /// Add a clause and watch two non-false literals when possible. Clauses with
    /// fewer than two literals are not watched; callers assert units directly.
    /// `deletable` marks a reduction-eligible learned clause; `lbd` meaningful only then.
    pub(crate) fn add_clause(&mut self, lits: Vec<Lit>, deletable: bool, lbd: u32) -> ClauseRef {
        self.add_shared_clause(Arc::from(lits), deletable, lbd)
    }

    fn add_shared_clause(&mut self, lits: Arc<[Lit]>, deletable: bool, lbd: u32) -> ClauseRef {
        let watch = self.initial_watches(&lits);
        let (w0, w1) = if lits.len() >= 2 {
            (Some(lits[watch[0]]), Some(lits[watch[1]]))
        } else {
            (None, None)
        };
        let cref = self.clauses.add(lits, watch, deletable, lbd);
        if let (Some(a), Some(b)) = (w0, w1) {
            self.watches[a.code() as usize].push(cref);
            self.watches[b.code() as usize].push(cref);
        }
        if deletable {
            self.num_learned += 1;
        }
        cref
    }

    fn initial_watches(&self, lits: &[Lit]) -> [usize; 2] {
        let mut watch = [0, 0];
        let mut found = 0;
        for (i, &lit) in lits.iter().enumerate() {
            if self.tvalue(lit) != Tri::False {
                watch[found] = i;
                found += 1;
                if found == 2 {
                    return watch;
                }
            }
        }
        for i in 0..lits.len() {
            if found == 0 || i != watch[0] {
                watch[found] = i;
                found += 1;
                if found == 2 {
                    break;
                }
            }
        }
        watch
    }

    /// Post order-encoding channeling clauses for every variable and seed root
    /// facts. Run once, at the root, before search.
    ///
    /// Per value `v` over span `[lo, hi]`: `[x=v] → [x≥v]`, `[x=v] → ¬[x≥v+1]`,
    /// `([x≥v] ∧ ¬[x≥v+1]) → [x=v]`, and monotonic `[x≥v+1] → [x≥v]`. Constant
    /// atoms fold away.
    pub fn post_channeling(&mut self) {
        for i in 0..self.solver.store.num_vars() {
            let x = VarId(i as u32);
            let lo = self.solver.store.min(x);
            let hi = self.solver.store.max(x);
            for v in lo..=hi {
                let eqv = self.atoms.eq(x, v);
                let gev = self.atoms.ge(x, v);
                let gev1 = self.atoms.ge(x, v + 1);
                self.post_clause(&[neg(eqv), gev]);
                self.post_clause(&[neg(eqv), neg(gev1)]);
                self.post_clause(&[neg(gev), gev1, eqv]);
                self.post_clause(&[neg(gev1), gev]);
            }
        }
        self.seed_facts();
    }

    /// Post a clause from possibly-constant terms: skip tautologies (`⊤`), drop
    /// `⊥`, add the rest. `None` if it folds away.
    fn post_clause(&mut self, terms: &[LitOrConst]) -> Option<ClauseRef> {
        let mut lits = Vec::with_capacity(terms.len());
        for &t in terms {
            match t {
                LitOrConst::True => return None,
                LitOrConst::False => {}
                LitOrConst::Lit(l) => lits.push(l),
            }
        }
        if lits.is_empty() {
            return None;
        }
        Some(self.add_clause(lits, false, 0))
    }

    /// Record every atom already fixed at the root as a level-0 [`Fact`](Reason::Fact).
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

    /// The literal decided at the deepest open level. Panics at the root.
    pub(crate) fn deepest_decision(&self) -> Lit {
        let d = self.decision_level();
        self.trail[self.level_starts[d]]
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

    /// The current truth of `lit` in the domain view.
    #[inline]
    pub fn value(&self, lit: Lit) -> Tri {
        view::lit_value(&self.solver.store, &self.atoms, lit)
    }

    /// The current truth of `lit` on the trail. Lags the view during propagation;
    /// equal at a fixpoint (invariant T2).
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

    /// Record a literal now true in the domain, with `reason`, at the current
    /// level. No-op if its atom is already assigned.
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
        self.atom_trail_pos[a] = self.trail.len();
        self.atom_reason[a] = reason;
        self.tval[a] = if lit.is_positive() {
            Tri::True
        } else {
            Tri::False
        };
        self.trail.push(lit);
    }

    /// The currently-recorded atoms of `var`, each as the literal matching its
    /// trail truth. This conjunction is a sound, recorded antecedent for any of
    /// `var`'s determined-but-unrecorded atoms, and bottoms out 1-UIP analysis.
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
    /// not (invariant T2). Called eagerly after each op so each atom is recorded
    /// at its determination level; recording later, at a deeper level, would let
    /// 1-UIP backjump too far and prune feasible solutions. The reason is
    /// [`ds_recorded`], which entails every determination and is itself recorded.
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
    /// `reason`. `Err` on wipeout (nothing recorded). Secondary flips are
    /// mirrored onto the trail immediately.
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
        self.sync_var(self.atoms.var_of(lit.atom()));
        Ok(())
    }

    /// Make `lit` a branching decision: open a level and assign it.
    pub fn decide(&mut self, lit: Lit) -> Result<(), Inconsistency> {
        self.open_level();
        self.apply_lit(lit, Reason::Decision)
    }

    /// Force `lit` true (unit propagation): enforce on the domain and record it.
    /// `Err` if it conflicts with the domain.
    pub fn assign(&mut self, lit: Lit, reason: Reason) -> Result<(), Inconsistency> {
        if self.tvalue(lit) == Tri::True {
            return Ok(());
        }
        view::apply(&mut self.solver.store, &self.atoms, lit)?;
        self.record(lit, reason);
        // Primary event names `lit` (already recorded).
        self.solver.store.events.clear();
        self.sync_var(self.atoms.var_of(lit.atom()));
        Ok(())
    }

    /// Unit-propagate the clause database (two-watched-literal scheme) over trail
    /// literals from `qhead`. Forces the last open literal of each unit clause and
    /// returns the conflicting clause if one is fully falsified.
    pub fn propagate_clauses(&mut self) -> Result<(), ClauseRef> {
        while self.qhead < self.trail.len() {
            if self.stopped() {
                return Ok(());
            }
            let q = self.trail[self.qhead];
            self.qhead += 1;
            // Clauses watching `neg` (now false) may need a new watch or to fire.
            let neg = q.negate();
            let mut ws = std::mem::take(&mut self.watches[neg.code() as usize]);
            let mut keep = 0usize;
            let mut read = 0usize;
            let mut conflict: Option<ClauseRef> = None;

            while read < ws.len() {
                let cref = ws[read];
                read += 1;
                let (slot, other) = {
                    let clause = self.clauses.get(cref);
                    if clause.lits[clause.watch[0]] == neg {
                        (0, clause.lits[clause.watch[1]])
                    } else {
                        debug_assert_eq!(clause.lits[clause.watch[1]], neg);
                        (1, clause.lits[clause.watch[0]])
                    }
                };
                if self.tvalue(other) == Tri::True {
                    ws[keep] = cref;
                    keep += 1;
                    continue;
                }
                // Hunt for a non-false literal to watch instead of `neg`.
                let mut new_watch: Option<(usize, Lit)> = None;
                {
                    let clause = self.clauses.get(cref);
                    for (k, &lit) in clause.lits.iter().enumerate() {
                        if k != clause.watch[0]
                            && k != clause.watch[1]
                            && self.tvalue(lit) != Tri::False
                        {
                            new_watch = Some((k, lit));
                            break;
                        }
                    }
                }
                if let Some((k, wl)) = new_watch {
                    self.clauses.get_mut(cref).watch[slot] = k;
                    self.watches[wl.code() as usize].push(cref);
                    continue;
                }
                // No replacement: the clause is unit on `other` (or a conflict).
                ws[keep] = cref;
                keep += 1;
                if self.tvalue(other) == Tri::False
                    || self.assign(other, Reason::Clause(cref)).is_err()
                {
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

    /// Propagate to a joint fixpoint of clause unit propagation and the FD
    /// propagators, in lockstep, recording each side's consequences. Returns the
    /// first [`Conflict`] found.
    pub fn propagate(&mut self) -> Result<(), Conflict> {
        loop {
            if self.stopped() {
                self.solver.store.clear_queue();
                return Ok(());
            }
            // Clause propagation to a fixpoint *first*, so every earlier change is
            // recorded on the trail before the next propagator is snapshotted;
            // keeping its `D_S` antecedents all trail-recorded.
            if let Err(c) = self.propagate_clauses() {
                return Err(Conflict::Clause(c));
            }
            // Then run a single FD propagator and record its inferences.
            let Some(pid) = self.solver.peek_prop() else {
                return Ok(());
            };
            // Snapshot its scope first so every event has a trail-consistent
            // fallback reason. A propagator can make several FD mutations before
            // returning, but the SAT trail only records them after the call.
            let step_trail_len = self.trail.len();
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
                StepOutcome::Ran(_) => self.record_prop_events(step_trail_len)?,
                StepOutcome::Failed(_) => {
                    self.solver.store.events.clear();
                    // Prefer a tight conflict reason only when it existed on the
                    // trail before this propagator call. Otherwise use the
                    // pre-call scope: intermediate FD mutations are not yet SAT
                    // assignments and cannot participate in conflict analysis.
                    let preferred = match self.solver.store.take_pending_conflict() {
                        Some(p) => translate_premises(&self.atoms, &p),
                        None => build_ds(&self.atoms, &self.conflict_scope),
                    };
                    let ds = self.reason_before_step(preferred, step_trail_len);
                    return Err(Conflict::Generic(ds));
                }
            }
        }
    }

    /// Drain a just-run propagator's primary events, recording each inferred
    /// literal with a [`Generic`](Reason::Generic) reason from the store's scope
    /// snapshot. Record every primary first: the FD domain already reflects the
    /// full propagator call, so synchronising a variable after only its first
    /// buffered event could mirror facts caused by later, unrecorded events.
    fn record_prop_events(&mut self, step_trail_len: usize) -> Result<(), Conflict> {
        let events = std::mem::take(&mut self.solver.store.events);
        self.changed_vars.clear();
        for ev in &events {
            let var = match ev.primary {
                DomEvent::GeTrue { var, .. }
                | DomEvent::LeTrue { var, .. }
                | DomEvent::NeTrue { var, .. }
                | DomEvent::EqTrue { var, .. } => var,
            };
            if let Some(lit) = translate(&self.atoms, &ev.primary) {
                if !self.is_assigned(lit.atom()) {
                    let preferred = cause_lits(&self.atoms, &ev.cause, &self.conflict_scope);
                    let ds = self.reason_before_step(preferred, step_trail_len);
                    self.record(lit, Reason::Generic(ds));
                }
            }
            if !self.changed_vars.contains(&var) {
                self.changed_vars.push(var);
            }
        }
        for i in 0..self.changed_vars.len() {
            let var = self.changed_vars[i];
            // Mirror the rest of this variable's bulk change only after all
            // primaries from the FD call are available as antecedents.
            self.sync_var(var);
        }
        self.propagate_clauses().map_err(Conflict::Clause)
    }

    /// Keep a compact reason only when all of its premises were SAT-trail facts
    /// before the current propagator call. Otherwise fall back to the pre-call
    /// whole-scope snapshot, which is always trail-consistent.
    fn reason_before_step(&self, preferred: Vec<Lit>, step_trail_len: usize) -> Vec<Lit> {
        if preferred.iter().all(|&l| {
            self.tvalue(l) == Tri::True && self.atom_trail_pos[l.atom() as usize] < step_trail_len
        }) {
            return preferred;
        }
        let fallback = build_ds(&self.atoms, &self.conflict_scope);
        debug_assert!(fallback.iter().all(|&l| {
            self.tvalue(l) == Tri::True && self.atom_trail_pos[l.atom() as usize] < step_trail_len
        }));
        fallback
    }

    /// Propagate; on each conflict, learn a 1-UIP clause, backjump, assert it.
    /// `true` at a clean fixpoint, `false` if a root conflict proves UNSAT.
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

    /// Import clauses published by other workers and propagate their effects.
    pub(crate) fn sync_shared_clauses(&mut self) -> bool {
        let Some(sharing) = &mut self.clause_sharing else {
            return true;
        };
        sharing.copy_new_into(&mut self.shared_scratch);
        let mut imported = false;
        while let Some(clause) = self.shared_scratch.pop() {
            let sharing = self.clause_sharing.as_ref().unwrap();
            if sharing.is_own(&clause) {
                continue;
            }
            sharing.record_import();
            imported = true;
            if !self.import_shared_clause(clause) {
                return false;
            }
        }
        !imported || self.propagate_and_learn()
    }

    fn import_shared_clause(&mut self, clause: SharedClause) -> bool {
        let mut open = None;
        let mut num_open = 0;
        let mut satisfied = false;
        for &lit in clause.lits.iter() {
            match self.tvalue(lit) {
                Tri::True => satisfied = true,
                Tri::Unknown => {
                    open.get_or_insert(lit);
                    num_open += 1;
                }
                Tri::False => {}
            }
        }
        let cref = self.add_shared_clause(clause.lits, true, clause.lbd);
        if satisfied || num_open >= 2 {
            return true;
        }
        if let Some(lit) = open {
            return self.assign(lit, Reason::Clause(cref)).is_ok()
                || self.resolve_conflict(Conflict::Clause(cref));
        }
        self.resolve_conflict(Conflict::Clause(cref))
    }

    /// Analyse `conflict`, backjump, learn the 1-UIP clause and assert it. `false`
    /// if the conflict proves UNSAT. Also used to inject an externally-detected
    /// conflict (a falsified blocking clause after enumerating a solution).
    pub(crate) fn resolve_conflict(&mut self, conflict: Conflict) -> bool {
        self.conflicts += 1;
        let Some((learnt, btlevel)) = self.analyze(conflict) else {
            return false;
        };
        self.learned_lits += learnt.len() as u64;
        // LBD measured at learn time, while every literal is still assigned.
        let lbd = self.lbd_of(&learnt);
        let deletable = learnt.len() >= 2;
        let learnt: Arc<[Lit]> = Arc::from(learnt);
        if let Some(sharing) = &self.clause_sharing {
            if self.cube_scope.is_empty() && self.bound_scope.is_none() {
                sharing.publish(Arc::clone(&learnt), lbd);
            } else {
                let mut exported = Vec::with_capacity(
                    self.cube_scope.len() + usize::from(self.bound_scope.is_some()) + learnt.len(),
                );
                exported.extend_from_slice(&self.cube_scope);
                exported.extend(self.bound_scope);
                exported.extend_from_slice(&learnt);
                sharing.publish(Arc::from(exported), lbd);
            }
        }
        self.backjump_to(btlevel);
        let asserting = learnt[0];
        let cref = self.add_shared_clause(learnt, deletable, lbd);
        // Post-backjump, the asserting literal is the only open one; it cannot wipe a domain.
        match self.assign(asserting, Reason::Clause(cref)) {
            Ok(()) => true,
            Err(_) if btlevel == 0 => false,
            Err(_) => panic!("learned asserting literal wiped a domain above the root"),
        }
    }

    /// First-UIP conflict analysis. Returns the learned clause (asserting literal
    /// first) and the backjump level (second-highest in the clause, 0 if unit).
    fn analyze(&mut self, conflict: Conflict) -> Option<(Vec<Lit>, usize)> {
        let mut clause_lits = self.conflict_lits(&conflict);
        // Analyse at the conflict's deepest level (a propagator may fail on
        // literals all below the current level). All at level 0 means UNSAT.
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
        let mut counter = 0usize; // unresolved level-`level` literals
        let mut index = self.trail.len();
        let mut uip;

        loop {
            for q in clause_lits {
                debug_assert_eq!(
                    self.tvalue(q),
                    Tri::False,
                    "analysis saw non-false literal {:?} {:?}",
                    q,
                    self.atoms.decode(q.atom())
                );
                let a = q.atom();
                let lv = self.level_of(a);
                if !self.seen[a as usize] && lv != UNSET && lv > 0 {
                    self.seen[a as usize] = true;
                    touched.push(a);
                    self.bump_var(self.atoms.var_of(a)); // VSIDS: conflict-active
                    if lv == level {
                        counter += 1;
                    } else {
                        learnt.push(q); // lower-level literal of the learned clause
                    }
                }
            }
            // Next: the latest still-seen trail literal at the current level.
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
                break; // sole remaining current-level literal: the UIP
            }
            clause_lits = self.reason_lits(uip);
        }

        learnt[0] = uip.negate(); // asserting literal
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

    /// Bump a variable's VSIDS activity, rescaling all if it overflows.
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

    /// Set up the root: post channeling, enqueue every propagator, propagate to
    /// fixpoint. `false` if the root is already unsatisfiable.
    pub fn init(&mut self) -> bool {
        self.post_channeling();
        self.solver.enqueue_all();
        self.propagate_and_learn()
    }

    /// Decision literal `[x = min(x)]` for the first unfixed var in `vars`, or
    /// `None` if all fixed. (Simple heuristic; the real driver uses dom/wdeg.)
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

    /// One solution over `vars` by CDCL, or `None` if UNSAT. Assumes
    /// [`init`](Self::init) succeeded.
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

    /// Literals of the initial conflict clause (all currently false).
    fn conflict_lits(&self, conflict: &Conflict) -> Vec<Lit> {
        match conflict {
            Conflict::Clause(c) => self.clauses.get(*c).lits.to_vec(),
            Conflict::Generic(ds) => ds.iter().map(|l| l.negate()).collect(),
        }
    }

    /// The antecedents to resolve against: `p`'s reason-clause literals other
    /// than `p` (all currently false).
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
            // Reason clause (p ∨ ⋁ ¬D_S); antecedents are ¬D_S.
            Reason::Generic(ds) => ds.iter().map(|l| l.negate()).collect(),
            // No antecedents (never resolved here).
            Reason::Decision | Reason::Fact | Reason::Unset => Vec::new(),
        }
    }

    /// Backjump to level `d`, undoing every assignment above it: clear side
    /// tables for popped atoms and pop the matching domain-trail levels.
    pub fn backjump_to(&mut self, d: usize) {
        let cur = self.decision_level();
        if d >= cur {
            return;
        }
        let keep = self.level_starts[d + 1];
        for lit in self.trail.drain(keep..) {
            let a = lit.atom() as usize;
            self.atom_level[a] = UNSET;
            self.atom_trail_pos[a] = usize::MAX;
            self.atom_reason[a] = Reason::Unset;
            self.tval[a] = Tri::Unknown;
        }
        self.level_starts.truncate(d + 1);
        for _ in 0..(cur - d) {
            self.solver.store.pop_level();
        }
        // Everything still on the trail was already propagated.
        self.qhead = self.trail.len();
    }
}
