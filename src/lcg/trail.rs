//! The SAT trail and [`Cdcl`] engine state layered on the FD store.
//!
//! Atom truth is read from the domain via [`view`](crate::lcg::view); the trail
//! records only assignment order and reasons. Backjump truncates the trail and
//! pops the matching domain-trail levels, rolling truth back with it.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::ids::VarId;
use crate::lcg::clause::{ClauseDb, ClauseRef, ClauseSharing, SharedClause};
use crate::lcg::lit::{Atom, AtomKind, AtomTable, LazyAtomRegistry, Lit, LitOrConst};
use crate::lcg::view::{self, Tri};
use crate::propagator::Inconsistency;
use crate::store::{Cause, DomEvent, Premise, ScopeVar, Solver, StepOutcome};

/// Stop flag that is never set; default until [`Cdcl::set_stop`] supplies the real one.
pub(crate) static NEVER_STOP: AtomicBool = AtomicBool::new(false);

/// Sentinel `atom_level` for an unassigned atom.
pub const UNSET: u32 = u32::MAX;

const ROOT_PROBE_MAX_CANDIDATES: usize = 32;
const ROOT_PROBE_MAX_DOMAIN_SIZE: usize = 16;
const ROOT_PROBE_MAX_BRANCH_LITS: usize = 512;
const RESTART_BASE_CONFLICTS: u64 = 256;
const FAST_RESTART_BASE_CONFLICTS: u64 = 64;
const RESTART_LBD_WINDOW: usize = 128;
const RESTART_LBD_RATIO: f64 = 1.5;
const VIVIFY_MIN_LEN: usize = 3;
const VIVIFY_MAX_LEN: usize = 8;
const VIVIFY_MAX_LBD: u32 = 4;
const VIVIFY_MAX_REMOVED: usize = 2;
const VIVIFY_MAX_VARS: usize = 50_000;
const MINIMIZE_LEARNT_MAX_LEN: usize = 128;

type ProofLogger<'s> = dyn FnMut(&[Lit]) + 's;

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
    Generic(Arc<[Lit]>),
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

struct ProbeDomain {
    var: VarId,
    min: i32,
    max: i32,
}

#[derive(Default)]
struct ProbeBranch {
    lits: Vec<Lit>,
    domains: Vec<ProbeDomain>,
}

enum ProbeOutcome {
    Consistent(ProbeBranch),
    Failed,
    RootUnsat,
}

struct RestartPolicy {
    base_conflicts: u64,
    due: bool,
    luby_index: u64,
    next_luby_conflict: u64,
    lbd_window: [u32; RESTART_LBD_WINDOW],
    lbd_window_len: usize,
    lbd_window_pos: usize,
    lbd_window_sum: u64,
    lbd_global_sum: u64,
    lbd_global_count: u64,
}

impl RestartPolicy {
    fn new() -> Self {
        Self::with_base(RESTART_BASE_CONFLICTS)
    }

    fn fast() -> Self {
        Self::with_base(FAST_RESTART_BASE_CONFLICTS)
    }

    fn with_base(base_conflicts: u64) -> Self {
        Self {
            base_conflicts,
            due: false,
            luby_index: 1,
            next_luby_conflict: base_conflicts,
            lbd_window: [0; RESTART_LBD_WINDOW],
            lbd_window_len: 0,
            lbd_window_pos: 0,
            lbd_window_sum: 0,
            lbd_global_sum: 0,
            lbd_global_count: 0,
        }
    }

    fn on_conflict(&mut self, conflicts: u64, lbd: u32) {
        self.lbd_global_sum += lbd as u64;
        self.lbd_global_count += 1;
        if self.lbd_window_len < RESTART_LBD_WINDOW {
            self.lbd_window[self.lbd_window_len] = lbd;
            self.lbd_window_len += 1;
            self.lbd_window_sum += lbd as u64;
        } else {
            self.lbd_window_sum -= self.lbd_window[self.lbd_window_pos] as u64;
            self.lbd_window[self.lbd_window_pos] = lbd;
            self.lbd_window_sum += lbd as u64;
            self.lbd_window_pos = (self.lbd_window_pos + 1) % RESTART_LBD_WINDOW;
        }
        self.due |= conflicts >= self.next_luby_conflict || self.lbd_window_is_degraded();
    }

    fn should_restart(&mut self, conflicts: u64) -> bool {
        if !self.due {
            return false;
        }
        self.due = false;
        self.luby_index += 1;
        self.next_luby_conflict = conflicts.saturating_add(self.base_conflicts.saturating_mul(luby(self.luby_index)));
        self.clear_lbd_window();
        true
    }

    fn lbd_window_is_degraded(&self) -> bool {
        if self.lbd_window_len < RESTART_LBD_WINDOW || self.lbd_global_count <= RESTART_LBD_WINDOW as u64 {
            return false;
        }
        let global = self.lbd_global_sum as f64 / self.lbd_global_count as f64;
        let window = self.lbd_window_sum as f64 / self.lbd_window_len as f64;
        window > global * RESTART_LBD_RATIO
    }

    fn clear_lbd_window(&mut self) {
        self.lbd_window_len = 0;
        self.lbd_window_pos = 0;
        self.lbd_window_sum = 0;
    }
}

fn luby(mut i: u64) -> u64 {
    debug_assert!(i > 0);
    while i > 2 {
        let msb = 63 - (i + 1).leading_zeros();
        if (1u64 << msb) == i + 1 {
            return 1u64 << (msb - 1);
        }
        i -= (1u64 << msb) - 1;
    }
    1
}

/// Build `D_S`: the true literals describing a scope's domains
/// (`[x≥min]`, `¬[x≥max+1]`, `¬[x=v]` per hole).
fn build_ds(atoms: &AtomTable, scope: &[ScopeVar]) -> Vec<Lit> {
    let mut lits = Vec::new();
    for sv in scope {
        if let LitOrConst::Lit(l) = atoms.ge(sv.var, sv.min) {
            lits.push(l);
        }
        if let LitOrConst::Lit(l) = atoms.ge_i64(sv.var, i64::from(sv.max) + 1) {
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
fn translate_premises(atoms: &AtomTable, premises: &[Premise]) -> Option<Vec<Lit>> {
    let mut lits = Vec::with_capacity(premises.len());
    let push = |loc: LitOrConst, out: &mut Vec<Lit>| {
        if let LitOrConst::Lit(l) = loc {
            out.push(l);
        }
    };
    for &p in premises {
        let var = match p {
            Premise::Ge { var, .. } | Premise::Le { var, .. } | Premise::Eq { var, .. } | Premise::Ne { var, .. } => var,
        };
        if atoms.is_lazy(var) {
            return None;
        }
        match p {
            Premise::Ge { var, bound } => push(atoms.ge(var, bound), &mut lits),
            Premise::Le { var, bound } => push(neg(atoms.ge_i64(var, i64::from(bound) + 1)), &mut lits),
            // Express `[x = v]` as order atoms `[x ≥ v] ∧ [x ≤ v]`, never the
            // positive equality atom: the eq atom may carry a deeper level, which
            // would make 1-UIP backjump too far and prune feasible solutions.
            Premise::Eq { var, val } => {
                push(atoms.ge(var, val), &mut lits);
                push(neg(atoms.ge_i64(var, i64::from(val) + 1)), &mut lits);
            }
            Premise::Ne { var, val } => push(neg(atoms.eq(var, val)), &mut lits),
        }
    }
    Some(lits)
}

/// Antecedent literals for an event's [`Cause`]: tight premises if supplied,
/// else the pre-propagator whole-scope snapshot.
fn cause_lits(atoms: &AtomTable, cause: &Cause, fallback: &[ScopeVar]) -> Vec<Lit> {
    match cause {
        Cause::Premises(p) => translate_premises(atoms, p).unwrap_or_else(|| build_ds(atoms, fallback)),
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

#[inline]
fn lit(loc: LitOrConst) -> Option<Lit> {
    match loc {
        LitOrConst::Lit(l) => Some(l),
        _ => None,
    }
}

/// Translate a primary [`DomEvent`] to the literal it asserts, or `None` if it
/// folds to a constant.
fn translate(atoms: &AtomTable, ev: &DomEvent) -> Option<Lit> {
    match *ev {
        DomEvent::GeTrue { var, bound } => lit(atoms.ge(var, bound)),
        DomEvent::LeTrue { var, bound } => lit(atoms.ge_i64(var, i64::from(bound) + 1)).map(Lit::negate),
        DomEvent::NeTrue { var, val } => lit(atoms.eq(var, val)).map(Lit::negate),
        DomEvent::EqTrue { var, val } => lit(atoms.eq(var, val)),
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
    /// Clause database: learned and imported constraint clauses.
    pub(crate) clauses: ClauseDb,
    /// Two-watched-literal occurrence lists, indexed by literal code.
    watches: Vec<Vec<ClauseRef>>,
    /// Binary implication lists, indexed by a true literal code. Each entry
    /// `(b, c)` means this literal makes binary clause `c` unit on `b`.
    binary_implications: Vec<Vec<(Lit, ClauseRef)>>,
    /// Reusable scratch for the pre-step scope snapshot of a conflict reason.
    conflict_scope: Vec<ScopeVar>,
    /// Reusable scratch for variables touched by one buffered FD step.
    changed_vars: Vec<VarId>,
    /// Reusable scratch for atoms allocated for one variable.
    atom_scratch: Vec<Atom>,
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
    /// Optional initial branching phase per variable (value-ordering hint, e.g.
    /// nearest-neighbour successors for routing). Empty = none; otherwise indexed
    /// by `VarId` and used to seed the saved-phase array at the start of `optimize`.
    pub(crate) initial_phase: Vec<Option<i32>>,
    /// Optional user-supplied variable priority. When non-empty, branching picks
    /// the first unfixed variable in this list before using the default heuristic.
    pub(crate) branch_order: Vec<VarId>,
    /// Live (non-tombstone) deletable learned clauses.
    num_learned: usize,
    /// Soft cap on live learned clauses; grows after each reduction.
    max_learned: usize,
    /// Conflicts analysed (statistics and restarts).
    pub(crate) conflicts: u64,
    /// Total literals across all learned clauses (numerator of avg learned size).
    pub(crate) learned_lits: u64,
    /// Learned clauses shortened by CP-aware vivification.
    pub(crate) vivified_clauses: u64,
    /// Literals removed by CP-aware vivification.
    pub(crate) vivified_lits: u64,
    /// Binary clauses visited during unit propagation.
    pub(crate) binary_clause_visits: u64,
    /// Non-binary watched clauses visited during unit propagation.
    pub(crate) watched_clause_visits: u64,
    /// Non-watch literals scanned while looking for a replacement watch.
    pub(crate) watched_literal_scans: u64,
    /// Literals implied by the binary-clause fast path.
    pub(crate) binary_implications_count: u64,
    /// Restart schedule and LBD moving-average trigger.
    restart: RestartPolicy,
    /// Optional portfolio clause exchange.
    pub(crate) clause_sharing: Option<ClauseSharing>,
    /// Reused buffer for newly published clauses.
    shared_scratch: Vec<SharedClause>,
    /// Negated job assumptions prepended to exported learned clauses.
    cube_scope: Vec<Lit>,
    /// Negated objective bound prepended to exported learned clauses.
    bound_scope: Option<Lit>,
    /// Last value recorded when a variable is unassigned by backjumping.
    pub(crate) saved_phase: Vec<Option<i32>>,
    /// Restarts performed so far; drives periodic rephasing.
    pub(crate) restarts_done: u64,
    /// Phase policy for the current restart segment: 0 = saved phase (default),
    /// 1 = inverted default, 2 = pseudo-random. Non-zero only on periodic
    /// rephasing segments, which diversify like a portfolio without a portfolio.
    pub(crate) rephase_mode: u8,
    /// Optional proof sink for learned clauses.
    proof_logger: Option<&'s mut ProofLogger<'s>>,
    /// Whether the final empty clause was already sent to the proof sink.
    proof_empty_logged: bool,
    /// Root-state checker reused across a restart segment for CP vivification.
    /// Cloning the FD store is the dominant per-conflict cost, so it is cloned
    /// once per segment (lazily) and reset by push/pop between clauses; rebuilt
    /// after a restart or objective-bound change (root state can only tighten
    /// mid-segment, so a slightly stale checker under-removes but stays sound).
    vivify_checker: Option<Solver>,
    /// Reusable to-clear stack for allocation-free learned-clause minimization.
    min_stack: Vec<u32>,
}

impl<'s> Cdcl<'s> {
    /// Build the engine over `solver`, allocating atoms from the root domains.
    pub fn new(solver: &'s mut Solver, vars: &[VarId]) -> Self {
        Self::new_with_registry(solver, vars, LazyAtomRegistry::new())
    }

    pub(crate) fn new_with_registry(solver: &'s mut Solver, vars: &[VarId], lazy_atoms: Arc<LazyAtomRegistry>) -> Self {
        // Ablation hook: `QAYD_SCOPE_REASONS` forces whole-scope explanations.
        solver.store.set_force_scope_reasons(std::env::var_os("QAYD_SCOPE_REASONS").is_some());
        solver.store.set_explain(true);
        let nvars = solver.store.num_vars();
        let mut active = (0..nvars).map(|i| solver.store.is_relevant(VarId(i as u32))).collect::<Vec<_>>();
        for &var in vars {
            active[var.index()] = true;
        }
        let atoms = AtomTable::build_active_sparse_with_registry(
            nvars,
            |v: VarId| active[v.index()],
            |v: VarId| solver.store.size(v) == 2 && solver.store.contains(v, -1) && solver.store.contains(v, 1),
            |v: VarId| solver.store.sparse_values(v),
            |v: VarId| (solver.store.min(v), solver.store.max(v)),
            lazy_atoms,
        );
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
            watches: vec![Vec::new(); n * 2],
            binary_implications: vec![Vec::new(); n * 2],
            conflict_scope: Vec::new(),
            changed_vars: Vec::new(),
            atom_scratch: Vec::new(),
            qhead: 0,
            activity: vec![0.0; nvars],
            var_inc: 1.0,
            stop: &NEVER_STOP,
            seed: 0,
            initial_phase: Vec::new(),
            branch_order: Vec::new(),
            num_learned: 0,
            max_learned: 2000,
            conflicts: 0,
            learned_lits: 0,
            vivified_clauses: 0,
            vivified_lits: 0,
            binary_clause_visits: 0,
            watched_clause_visits: 0,
            watched_literal_scans: 0,
            binary_implications_count: 0,
            restart: RestartPolicy::new(),
            clause_sharing: None,
            shared_scratch: Vec::new(),
            cube_scope: Vec::new(),
            bound_scope: None,
            saved_phase: vec![None; nvars],
            restarts_done: 0,
            rephase_mode: 0,
            proof_logger: None,
            proof_empty_logged: false,
            vivify_checker: None,
            min_stack: Vec::new(),
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

    /// Use a shorter restart schedule to diversify a CSP portfolio worker.
    pub(crate) fn use_fast_restarts(&mut self) {
        self.restart = RestartPolicy::fast();
    }

    /// Raise the learned-clause database budget for pure SAT-style searches.
    pub(crate) fn set_learned_clause_budget(&mut self, max_learned: usize) {
        self.max_learned = self.max_learned.max(max_learned);
    }

    /// Stream learned clauses to an external proof writer.
    pub(crate) fn set_proof_logger(&mut self, logger: &'s mut ProofLogger<'s>) {
        self.proof_logger = Some(logger);
    }

    /// Send one proof clause to the optional proof sink.
    pub(crate) fn log_proof_clause(&mut self, clause: &[Lit]) {
        if clause.is_empty() {
            if self.proof_empty_logged {
                return;
            }
            self.proof_empty_logged = true;
        }
        if let Some(logger) = &mut self.proof_logger {
            logger(clause);
        }
    }

    /// Whether a proof sink is attached.
    pub(crate) fn proof_logging_enabled(&self) -> bool {
        self.proof_logger.is_some()
    }

    /// Scope exported clauses to the current search cube.
    pub(crate) fn set_cube_scope(&mut self, cube: &[Lit]) {
        self.cube_scope.clear();
        self.cube_scope.extend(cube.iter().map(|assumption| assumption.negate()));
    }

    /// Scope exported clauses to the current objective bound.
    pub(crate) fn set_bound_scope(&mut self, bound: Lit) {
        self.bound_scope = Some(bound.negate());
        // The tightened objective bound changes root state; drop the stale checker.
        self.vivify_checker = None;
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
        // Under memory pressure, force a reduction even below the soft cap so the
        // learned DB shrinks before the hard limit stops search.
        let soft = crate::mem::pressure() != crate::mem::Pressure::None;
        if self.num_learned <= self.max_learned && !soft {
            return;
        }
        if soft {
            crate::mem::note_soft();
        }
        debug_assert_eq!(self.decision_level(), 0, "reduction off the root");
        let mut learned: Vec<ClauseRef> = (0..self.clauses.len() as u32)
            .map(ClauseRef)
            .filter(|&c| {
                let cl = self.clauses.get(c);
                cl.deletable && cl.lits.len() >= 2
            })
            .collect();
        // Worst clauses first. Learned binary clauses are protected below, so
        // length is mostly a tie-break among longer clauses.
        learned.sort_unstable_by_key(|&c| {
            let clause = self.clauses.get(c);
            (std::cmp::Reverse(clause.lbd), std::cmp::Reverse(clause.lits.len()))
        });
        let target = learned.len() / 2;
        let mut removed = 0;
        for &c in &learned {
            if removed >= target {
                break;
            }
            let clause = self.clauses.get(c);
            if clause.lbd <= 2 || clause.lits.len() == 2 {
                continue;
            }
            self.delete_clause(c);
            removed += 1;
        }
        self.num_learned -= removed;
        if soft {
            self.max_learned = crate::mem::degrade_budget(self.max_learned);
        } else {
            self.max_learned += 500;
        }
    }

    /// Tombstone a clause: unwatch it and free its literals; its slot remains so
    /// outstanding [`ClauseRef`]s stay valid.
    fn delete_clause(&mut self, c: ClauseRef) {
        let lits = Arc::clone(&self.clauses.get(c).lits);
        if lits.len() == 2 {
            self.remove_binary_implication(lits[0].negate(), lits[1], c);
            self.remove_binary_implication(lits[1].negate(), lits[0], c);
        } else if lits.len() >= 2 {
            let (w0, w1) = {
                let clause = self.clauses.get(c);
                (clause.lits[clause.watch[0]], clause.lits[clause.watch[1]])
            };
            self.unwatch(w0, c);
            self.unwatch(w1, c);
        }
        self.clauses.remove(c);
    }

    fn unwatch(&mut self, lit: Lit, clause: ClauseRef) {
        if let Some(list) = self.watches.get_mut(lit.code() as usize) {
            list.retain(|&c| c != clause);
        }
    }

    fn add_watch(&mut self, lit: Lit, clause: ClauseRef) {
        let idx = lit.code() as usize;
        if idx >= self.watches.len() {
            self.watches.resize_with(idx + 1, Vec::new);
        }
        self.watches[idx].push(clause);
    }

    fn remove_binary_implication(&mut self, from: Lit, implied: Lit, clause: ClauseRef) {
        if let Some(list) = self.binary_implications.get_mut(from.code() as usize) {
            list.retain(|&(other, cref)| other != implied || cref != clause);
        }
    }

    fn add_binary_implication(&mut self, from: Lit, implied: Lit, clause: ClauseRef) {
        let idx = from.code() as usize;
        if idx >= self.binary_implications.len() {
            self.binary_implications.resize_with(idx + 1, Vec::new);
        }
        self.binary_implications[idx].push((implied, clause));
    }

    /// Literal Block Distance: distinct decision levels among assigned literals.
    fn lbd_of(&self, lits: &[Lit]) -> u32 {
        let mut levels: Vec<u32> = lits.iter().map(|l| self.level_of(l.atom())).filter(|&lv| lv != UNSET).collect();
        levels.sort_unstable();
        levels.dedup();
        levels.len() as u32
    }

    pub(crate) fn copy_inprocessing_stats(&self, stats: &mut crate::search::SolveStats) {
        stats.learned_lits = self.learned_lits;
        stats.vivified_clauses = self.vivified_clauses;
        stats.vivified_lits = self.vivified_lits;
        stats.binary_clause_visits = self.binary_clause_visits;
        stats.watched_clause_visits = self.watched_clause_visits;
        stats.watched_literal_scans = self.watched_literal_scans;
        stats.binary_implications = self.binary_implications_count;
    }

    /// Add a clause and watch two non-false literals when possible. Clauses with
    /// fewer than two literals are not watched; callers assert units directly.
    /// `deletable` marks a reduction-eligible learned clause; `lbd` meaningful only then.
    pub(crate) fn add_clause(&mut self, lits: Vec<Lit>, deletable: bool, lbd: u32) -> ClauseRef {
        self.add_shared_clause(Arc::from(lits), deletable, lbd)
    }

    /// Add a derived clause that must be visible to proof logging.
    pub(crate) fn add_derived_clause(&mut self, lits: Vec<Lit>, deletable: bool, lbd: u32) -> ClauseRef {
        self.log_proof_clause(&lits);
        self.add_clause(lits, deletable, lbd)
    }

    /// Assert a non-deletable root unit and propagate it.
    pub(crate) fn assert_root_lit(&mut self, lit: Lit) -> bool {
        debug_assert_eq!(self.decision_level(), 0);
        let cref = self.add_clause(vec![lit], false, 0);
        self.assign(lit, Reason::Clause(cref)).is_ok() && self.propagate_and_learn()
    }

    /// Assert a derived root unit and expose it to proof logging.
    pub(crate) fn assert_derived_root_lit(&mut self, lit: Lit) -> bool {
        debug_assert_eq!(self.decision_level(), 0);
        let cref = self.add_derived_clause(vec![lit], false, 0);
        self.assign(lit, Reason::Clause(cref)).is_ok() && self.propagate_and_learn()
    }

    fn add_shared_clause(&mut self, lits: Arc<[Lit]>, deletable: bool, lbd: u32) -> ClauseRef {
        let watch = self.initial_watches(&lits);
        let (w0, w1) = if lits.len() >= 3 { (Some(lits[watch[0]]), Some(lits[watch[1]])) } else { (None, None) };
        let cref = self.clauses.add(lits, watch, deletable, lbd);
        let clause = self.clauses.get(cref);
        if clause.lits.len() == 2 {
            let a = clause.lits[0];
            let b = clause.lits[1];
            self.add_binary_implication(a.negate(), b, cref);
            self.add_binary_implication(b.negate(), a, cref);
        } else if let (Some(a), Some(b)) = (w0, w1) {
            self.add_watch(a, cref);
            self.add_watch(b, cref);
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
        self.atom_level.get(atom as usize).is_some_and(|&level| level != UNSET)
    }

    /// The level `atom` was assigned at (only meaningful while assigned).
    #[inline]
    pub fn level_of(&self, atom: Atom) -> u32 {
        self.atom_level.get(atom as usize).copied().unwrap_or(UNSET)
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
        let t = self.tval.get(lit.atom() as usize).copied().unwrap_or(Tri::Unknown);
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
        self.ensure_atom(a);
        if self.atom_level[a] != UNSET {
            return;
        }
        debug_assert_eq!(view::lit_value(&self.solver.store, &self.atoms, lit), Tri::True, "recorded literal is not true in the domain");
        self.atom_level[a] = self.decision_level() as u32;
        self.atom_trail_pos[a] = self.trail.len();
        self.atom_reason[a] = reason;
        self.tval[a] = if lit.is_positive() { Tri::True } else { Tri::False };
        self.trail.push(lit);
    }

    fn ensure_atom(&mut self, atom: usize) {
        if atom < self.atom_level.len() {
            return;
        }
        let len = atom + 1;
        self.atom_level.resize(len, UNSET);
        self.atom_trail_pos.resize(len, usize::MAX);
        self.atom_reason.resize(len, Reason::Unset);
        self.tval.resize(len, Tri::Unknown);
        self.seen.resize(len, false);
    }

    /// The currently-recorded atoms of `var`, each as the literal matching its
    /// trail truth. This conjunction is a sound, recorded antecedent for any of
    /// `var`'s determined-but-unrecorded atoms, and bottoms out 1-UIP analysis.
    fn ds_recorded(&self, var: VarId) -> Vec<Lit> {
        let mut lits = Vec::new();
        let push_if_recorded = |loc: LitOrConst, out: &mut Vec<Lit>| {
            if let LitOrConst::Lit(l) = loc {
                match self.tval.get(l.atom() as usize).copied().unwrap_or(Tri::Unknown) {
                    Tri::True => out.push(Lit::positive(l.atom())),
                    Tri::False => out.push(Lit::negative(l.atom())),
                    Tri::Unknown => {}
                }
            }
        };
        if self.atoms.is_sign(var) {
            push_if_recorded(self.atoms.ge(var, 1), &mut lits);
            return lits;
        }
        let mut atoms = Vec::new();
        self.atoms.append_atoms(var, &mut atoms);
        for atom in atoms {
            push_if_recorded(LitOrConst::Lit(Lit::positive(atom)), &mut lits);
        }
        lits
    }

    /// Record every atom of `var` the domain now determines but the trail has
    /// not (invariant T2). Called eagerly after each op so each atom is recorded
    /// at its determination level; recording later, at a deeper level, would let
    /// 1-UIP backjump too far and prune feasible solutions. The reason is
    /// [`ds_recorded`], which entails every determination and is itself recorded.
    pub(crate) fn sync_var(&mut self, var: VarId) {
        if self.atoms.is_sign(var) {
            let LitOrConst::Lit(l) = self.atoms.ge(var, 1) else { unreachable!("a sign variable has one real atom") };
            if !self.is_assigned(l.atom()) {
                debug_assert_eq!(
                    view::lit_value(&self.solver.store, &self.atoms, l),
                    Tri::Unknown,
                    "a determined sign atom must be recorded by its primary event"
                );
            }
            return;
        }
        if self.atoms.is_lazy(var) {
            let _ = self.atoms.ge(var, self.solver.store.min(var));
            let _ = self.atoms.ge_i64(var, i64::from(self.solver.store.max(var)) + 1);
        }
        let mut todo: Vec<Lit> = Vec::new();
        self.atom_scratch.clear();
        self.atoms.append_atoms(var, &mut self.atom_scratch);
        for &atom in &self.atom_scratch {
            let l = Lit::positive(atom);
            if !self.is_assigned(atom) {
                match view::lit_value(&self.solver.store, &self.atoms, l) {
                    Tri::True => todo.push(l),
                    Tri::False => todo.push(l.negate()),
                    Tri::Unknown => {}
                }
            }
        }
        if todo.is_empty() {
            return;
        }
        let ds: Arc<[Lit]> = Arc::from(self.ds_recorded(var));
        for l in todo {
            self.record(l, Reason::Generic(Arc::clone(&ds)));
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
                debug_assert_eq!(el.atom(), lit.atom(), "an op emitted an unexpected non-primary event");
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
            if let Some(c) = self.propagate_binary_implications(q) {
                return Err(c);
            }
            // Clauses watching `neg` (now false) may need a new watch or to fire.
            let neg = q.negate();
            let watch_idx = neg.code() as usize;
            let mut ws = self.watches.get_mut(watch_idx).map(std::mem::take).unwrap_or_default();
            let mut keep = 0usize;
            let mut read = 0usize;
            let mut conflict: Option<ClauseRef> = None;

            while read < ws.len() {
                let cref = ws[read];
                read += 1;
                self.watched_clause_visits += 1;
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
                        self.watched_literal_scans += 1;
                        if k != clause.watch[0] && k != clause.watch[1] && self.tvalue(lit) != Tri::False {
                            new_watch = Some((k, lit));
                            break;
                        }
                    }
                }
                if let Some((k, wl)) = new_watch {
                    self.clauses.get_mut(cref).watch[slot] = k;
                    self.add_watch(wl, cref);
                    continue;
                }
                // No replacement: the clause is unit on `other` (or a conflict).
                ws[keep] = cref;
                keep += 1;
                if self.tvalue(other) == Tri::False || self.assign(other, Reason::Clause(cref)).is_err() {
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
            if watch_idx >= self.watches.len() {
                self.watches.resize_with(watch_idx + 1, Vec::new);
            }
            self.watches[watch_idx] = ws;
            if let Some(c) = conflict {
                return Err(c);
            }
        }
        Ok(())
    }

    fn propagate_binary_implications(&mut self, lit: Lit) -> Option<ClauseRef> {
        let implications = self.binary_implications.get(lit.code() as usize).cloned().unwrap_or_default();
        for (implied, cref) in implications {
            self.binary_clause_visits += 1;
            match self.tvalue(implied) {
                Tri::True => {}
                Tri::False => return Some(cref),
                Tri::Unknown => {
                    if self.assign(implied, Reason::Clause(cref)).is_err() {
                        return Some(cref);
                    }
                    self.binary_implications_count += 1;
                }
            }
        }
        None
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
                let Cdcl { solver, conflict_scope, .. } = self;
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
                        Some(p) => translate_premises(&self.atoms, &p).unwrap_or_else(|| build_ds(&self.atoms, &self.conflict_scope)),
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
                    self.record(lit, Reason::Generic(Arc::from(ds)));
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
        if preferred.iter().all(|&lit| self.was_recorded_before(lit, step_trail_len)) {
            return preferred;
        }
        let fallback = build_ds(&self.atoms, &self.conflict_scope);
        debug_assert!(fallback.iter().all(|&lit| self.was_recorded_before(lit, step_trail_len)));
        fallback
    }

    fn was_recorded_before(&self, lit: Lit, trail_len: usize) -> bool {
        self.atom_trail_pos.get(lit.atom() as usize).is_some_and(|&pos| self.tvalue(lit) == Tri::True && pos < trail_len)
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

    pub(crate) fn should_restart(&mut self) -> bool {
        let fire = self.restart.should_restart(self.conflicts);
        if fire {
            // New segment: rebuild the vivification checker from fresh root state.
            self.vivify_checker = None;
        }
        fire
    }

    /// One bounded root probing pass. For each selected decision literal, both
    /// branches are propagated once. Failed branches become root units; branch
    /// implications become binary clauses; implications common to both branches
    /// are asserted at the root.
    pub(crate) fn root_probe(&mut self, vars: &[VarId]) -> bool {
        debug_assert_eq!(self.decision_level(), 0);
        if self.stopped() {
            return true;
        }
        let candidates = self.root_probe_candidates(vars);
        let mut added = HashSet::new();
        for decision in candidates {
            if self.stopped() || self.tvalue(decision) != Tri::Unknown {
                continue;
            }
            let left = match self.probe_branch(decision, &mut added) {
                ProbeOutcome::Consistent(branch) => branch,
                ProbeOutcome::Failed => {
                    if !self.assert_derived_root_lit(decision.negate()) {
                        return false;
                    }
                    continue;
                }
                ProbeOutcome::RootUnsat => return false,
            };
            if self.tvalue(decision) != Tri::Unknown {
                continue;
            }
            let right = match self.probe_branch(decision.negate(), &mut added) {
                ProbeOutcome::Consistent(branch) => branch,
                ProbeOutcome::Failed => {
                    if !self.assert_derived_root_lit(decision) {
                        return false;
                    }
                    continue;
                }
                ProbeOutcome::RootUnsat => return false,
            };
            let bounds_ok = self.proof_logging_enabled() || self.assert_common_probe_bounds(&left.domains, &right.domains);
            if !self.assert_common_probe_lits(&left.lits, &right.lits) || !bounds_ok {
                return false;
            }
        }
        true
    }

    fn root_probe_candidates(&self, vars: &[VarId]) -> Vec<Lit> {
        let weights = self.solver.weights();
        let mut scored = Vec::new();
        for &var in vars {
            let size = self.solver.store.size(var);
            if !(2..=ROOT_PROBE_MAX_DOMAIN_SIZE).contains(&size) {
                continue;
            }
            let val = self.solver.store.min(var);
            let LitOrConst::Lit(lit) = self.atoms.eq(var, val) else {
                continue;
            };
            if self.tvalue(lit) == Tri::Unknown {
                scored.push((size, std::cmp::Reverse(self.solver.store.var_weight(var, weights)), var, lit));
            }
        }
        scored.sort_unstable_by_key(|&(size, weight, var, _)| (size, weight, var));
        scored.into_iter().take(ROOT_PROBE_MAX_CANDIDATES).map(|(_, _, _, lit)| lit).collect()
    }

    fn probe_branch(&mut self, decision: Lit, added: &mut HashSet<(u32, u32)>) -> ProbeOutcome {
        debug_assert_eq!(self.decision_level(), 0);
        let root_len = self.trail.len();
        match self.decide(decision) {
            Ok(()) => {}
            Err(_) => {
                self.backjump_to(0);
                return ProbeOutcome::Failed;
            }
        }
        if !self.propagate_and_learn() {
            self.backjump_to(0);
            return ProbeOutcome::RootUnsat;
        }
        if self.decision_level() == 0 {
            return if self.tvalue(decision) == Tri::False {
                ProbeOutcome::Failed
            } else {
                ProbeOutcome::Consistent(ProbeBranch::default())
            };
        }
        let mut lits: Vec<Lit> =
            self.trail[root_len..].iter().copied().filter(|&lit| lit != decision).take(ROOT_PROBE_MAX_BRANCH_LITS).collect();
        lits.sort_unstable_by_key(|lit| lit.code());
        lits.dedup_by_key(|lit| lit.code());
        let domains = self.probe_domains(&lits);
        self.backjump_to(0);
        if !self.add_probe_implications(decision, &lits, added) {
            return ProbeOutcome::RootUnsat;
        }
        ProbeOutcome::Consistent(ProbeBranch { lits, domains })
    }

    fn probe_domains(&self, lits: &[Lit]) -> Vec<ProbeDomain> {
        let mut vars: Vec<VarId> = lits.iter().map(|lit| self.atoms.var_of(lit.atom())).collect();
        vars.sort_unstable();
        vars.dedup();
        vars.into_iter().map(|var| ProbeDomain { var, min: self.solver.store.min(var), max: self.solver.store.max(var) }).collect()
    }

    fn add_probe_implications(&mut self, decision: Lit, implied: &[Lit], added: &mut HashSet<(u32, u32)>) -> bool {
        let guard = decision.negate();
        self.sync_var(self.atoms.var_of(guard.atom()));
        for &lit in implied {
            self.sync_var(self.atoms.var_of(lit.atom()));
            if !self.add_probe_clause(guard, lit, added) {
                return false;
            }
        }
        true
    }

    fn add_probe_clause(&mut self, guard: Lit, implied: Lit, added: &mut HashSet<(u32, u32)>) -> bool {
        if guard == implied || self.tvalue(guard) == Tri::True || self.tvalue(implied) == Tri::True {
            return true;
        }
        match (self.tvalue(guard), self.tvalue(implied)) {
            (Tri::False, Tri::False) => false,
            (Tri::False, Tri::Unknown) => {
                let cref = self.add_derived_clause(vec![guard, implied], true, 2);
                self.assign(implied, Reason::Clause(cref)).is_ok() && self.propagate_and_learn()
            }
            (Tri::Unknown, Tri::False) => {
                let cref = self.add_derived_clause(vec![guard, implied], true, 2);
                self.assign(guard, Reason::Clause(cref)).is_ok() && self.propagate_and_learn()
            }
            (Tri::Unknown, Tri::Unknown) => {
                if added.insert((guard.code(), implied.code())) {
                    self.add_derived_clause(vec![guard, implied], true, 2);
                }
                true
            }
            _ => true,
        }
    }

    fn assert_common_probe_lits(&mut self, left: &[Lit], right: &[Lit]) -> bool {
        let mut i = 0;
        let mut j = 0;
        while i < left.len() && j < right.len() {
            match left[i].code().cmp(&right[j].code()) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    let lit = left[i];
                    self.sync_var(self.atoms.var_of(lit.atom()));
                    match self.tvalue(lit) {
                        Tri::True => {}
                        Tri::Unknown if self.assert_derived_root_lit(lit) => {}
                        Tri::Unknown | Tri::False => return false,
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        true
    }

    fn assert_common_probe_bounds(&mut self, left: &[ProbeDomain], right: &[ProbeDomain]) -> bool {
        let mut i = 0;
        let mut j = 0;
        while i < left.len() && j < right.len() {
            match left[i].var.cmp(&right[j].var) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    let var = left[i].var;
                    let min = left[i].min.min(right[j].min);
                    let max = left[i].max.max(right[j].max);
                    if min > self.solver.store.min(var) {
                        if let LitOrConst::Lit(lit) = self.atoms.ge(var, min) {
                            if !self.assert_root_lit(lit) {
                                return false;
                            }
                        }
                    }
                    if max < self.solver.store.max(var) {
                        if let LitOrConst::Lit(lit) = self.atoms.ge_i64(var, i64::from(max) + 1) {
                            if !self.assert_root_lit(lit.negate()) {
                                return false;
                            }
                        }
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        true
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
        for &lit in clause.lits.iter() {
            self.sync_var(self.atoms.var_of(lit.atom()));
        }
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
            return self.assign(lit, Reason::Clause(cref)).is_ok() || self.resolve_conflict(Conflict::Clause(cref));
        }
        self.resolve_conflict(Conflict::Clause(cref))
    }

    fn vivify_learned(&mut self, learnt: &mut Vec<Lit>) {
        if learnt.len() < VIVIFY_MIN_LEN
            || learnt.len() > VIVIFY_MAX_LEN
            || self.lbd_of(learnt) > VIVIFY_MAX_LBD
            || self.solver.store.num_vars() > VIVIFY_MAX_VARS
        {
            return;
        }

        let mut checker = match self.vivify_checker.take() {
            Some(c) => c,
            None => self.root_solver_clone(),
        };
        let mut removed = 0usize;
        let mut i = 1; // keep the asserting literal
        while i < learnt.len() && removed < VIVIFY_MAX_REMOVED {
            if self.cp_refutes_without(&mut checker, learnt, i) {
                learnt.remove(i);
                removed += 1;
            } else {
                i += 1;
            }
        }
        self.vivify_checker = Some(checker);
        if removed > 0 {
            self.vivified_clauses += 1;
            self.vivified_lits += removed as u64;
        }
    }

    fn root_solver_clone(&self) -> Solver {
        let mut solver = self.solver.clone();
        while solver.store.level() > 0 {
            solver.store.pop_level();
        }
        solver.store.clear_queue();
        solver.store.events.clear();
        solver.store.clear_pending();
        solver
    }

    fn cp_refutes_without(&self, solver: &mut Solver, clause: &[Lit], skip: usize) -> bool {
        solver.store.push_level();
        solver.store.clear_queue();
        solver.store.events.clear();
        solver.store.clear_pending();

        let mut refuted = false;
        for (i, &lit) in clause.iter().enumerate() {
            if i == skip {
                continue;
            }
            match view::lit_value(&solver.store, &self.atoms, lit) {
                Tri::False => {}
                Tri::True => {
                    refuted = true;
                    break;
                }
                Tri::Unknown => {
                    if view::apply(&mut solver.store, &self.atoms, lit.negate()).is_err() {
                        refuted = true;
                        break;
                    }
                }
            }
        }
        if !refuted {
            refuted = solver.propagate().is_err();
        }

        solver.store.clear_queue();
        solver.store.events.clear();
        solver.store.clear_pending();
        solver.store.pop_level();
        refuted
    }

    /// Analyse `conflict`, backjump, learn the 1-UIP clause and assert it. `false`
    /// if the conflict proves UNSAT. Also used to inject an externally-detected
    /// conflict (a falsified blocking clause after enumerating a solution).
    pub(crate) fn resolve_conflict(&mut self, conflict: Conflict) -> bool {
        self.conflicts += 1;
        let Some(mut learnt) = self.analyze(conflict) else {
            self.log_proof_clause(&[]);
            return false;
        };
        self.minimize_learnt(&mut learnt);
        self.vivify_learned(&mut learnt);
        // Backjump level recomputed here: vivification may have shortened the clause.
        let btlevel = learnt[1..].iter().map(|l| self.level_of(l.atom()) as usize).max().unwrap_or(0);
        // Two-watched-literal invariant: slot 1 must hold a btlevel literal so the
        // two watched literals are the last two to be unassigned on backjump. Post
        // backjump every learnt[1..] is still false, so `initial_watches` picks
        // index 1 as the second watch; keep the max-level literal there.
        if let Some(k) = (1..learnt.len()).find(|&i| self.level_of(learnt[i].atom()) as usize == btlevel) {
            learnt.swap(1, k);
        }
        self.learned_lits += learnt.len() as u64;
        // LBD measured at learn time, while every literal is still assigned.
        let lbd = self.lbd_of(&learnt);
        self.restart.on_conflict(self.conflicts, lbd);
        let deletable = learnt.len() >= 2;
        self.log_proof_clause(&learnt);
        let learnt: Arc<[Lit]> = Arc::from(learnt);
        if let Some(sharing) = &self.clause_sharing {
            if self.cube_scope.is_empty() && self.bound_scope.is_none() {
                sharing.publish(Arc::clone(&learnt), lbd);
            } else {
                let mut exported = Vec::with_capacity(self.cube_scope.len() + usize::from(self.bound_scope.is_some()) + learnt.len());
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

    /// First-UIP conflict analysis. Returns the learned clause, asserting literal
    /// first. The caller derives the backjump level after vivification.
    fn analyze(&mut self, conflict: Conflict) -> Option<Vec<Lit>> {
        let mut clause_lits = self.conflict_lits(&conflict);
        // Analyse at the conflict's deepest level (a propagator may fail on
        // literals all below the current level). All at level 0 means UNSAT.
        let level = clause_lits.iter().map(|l| self.level_of(l.atom())).filter(|&lv| lv != UNSET).max().unwrap_or(0);
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
                debug_assert_eq!(self.tvalue(q), Tri::False, "analysis saw non-false literal {:?} {:?}", q, self.atoms.decode(q.atom()));
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
        for a in touched {
            self.seen[a as usize] = false;
        }
        self.decay_activity();
        Some(learnt)
    }

    /// MiniSat `analyzeFinal`, the sibling of [`analyze`](Self::analyze) for
    /// assumption solving: `p` is a cube assumption literal that is currently
    /// *false* (so `¬p` sits on the trail). Instead of stopping at a 1-UIP, walk
    /// the implication graph back to the assumption *decisions* that forced `¬p`
    /// and return that subset of the cube (including `p` itself): a set of
    /// assumptions jointly inconsistent with the root (an unsat core).
    ///
    /// Only assumption decisions are on the trail when this is called (the driver
    /// invokes it before branching, and any conflict backjumps branch levels
    /// away first), so every `Reason::Decision` literal reached is a cube
    /// assumption. Uses the shared `seen` buffer, restored on exit.
    pub(crate) fn analyze_final(&mut self, p: Lit) -> Vec<Lit> {
        let mut core = vec![p];
        if self.decision_level() == 0 {
            // ¬p is a root fact or a learned root unit: p alone refutes the root.
            return core;
        }
        let start = p.negate(); // true on the trail
        let root_end = self.level_starts[1];
        let mut touched = vec![start.atom()];
        self.seen[start.atom() as usize] = true;
        // Newest first, down to the first decision level; root literals (below
        // `root_end`) have no assumption antecedents and are skipped.
        for i in (root_end..self.trail.len()).rev() {
            let lit = self.trail[i];
            let a = lit.atom();
            if !self.seen[a as usize] {
                continue;
            }
            self.seen[a as usize] = false;
            if matches!(self.reason_of(a), Reason::Decision) {
                core.push(lit); // a participating assumption decision
            } else {
                for q in self.reason_lits(lit) {
                    let qa = q.atom() as usize;
                    let lv = self.level_of(q.atom());
                    if lv != 0 && lv != UNSET && !self.seen[qa] {
                        self.seen[qa] = true;
                        touched.push(q.atom());
                    }
                }
            }
        }
        for a in touched {
            self.seen[a as usize] = false;
        }
        core
    }

    /// The *semantic footprint* of an unsatisfiable set of selectors that refutes
    /// by propagation alone. Asserts every selector in `on` at the root,
    /// propagates once, and, if that already fails, walks the refutation's reason
    /// DAG bucketing the non-selector literals of each propagator explanation by
    /// the selector it carries. The result is, per constraint, the *specific*
    /// atoms it reasoned about (the Hall set of an `allDifferent`, the energy
    /// window of a `cumulative`), not the whole global. This is the differentiator
    /// from CNF-level proof trimming: the DAG leaves are the propagators'
    /// structured [`Reason::Generic`] explanations.
    ///
    /// `None` when asserting the selectors does not fail by propagation (the core
    /// needs search to refute; a sub-constraint explanation for that case is
    /// future work). The caller must have run [`init`](Cdcl::init) and pass a set
    /// known to be unsatisfiable. Leaves the trail dirty; use a fresh engine.
    pub(crate) fn explain_by_propagation(&mut self, on: &[Lit]) -> Option<Vec<(VarId, Vec<Lit>)>> {
        debug_assert_eq!(self.decision_level(), 0);
        for &lit in on {
            let cref = self.add_clause(vec![lit], false, 0);
            if self.assign(lit, Reason::Clause(cref)).is_err() {
                return Some(Vec::new()); // a selector wiped a domain on assertion: one-atom refutation
            }
        }
        let conflict = match self.propagate() {
            Ok(()) => return None, // consistent under propagation: the core needs search
            Err(conflict) => conflict,
        };
        // Recognise selectors by *variable*: `= 1` may surface as either the `Eq`
        // or the `Ge` atom of a `{0,1}` selector, so atom-matching would miss it.
        let sel_vars: HashSet<VarId> = on.iter().map(|l| self.atoms.var_of(l.atom())).collect();
        Some(self.footprint_of_cone(&conflict, &sel_vars))
    }

    /// Walk the reason DAG behind `conflict` (read-only, over the live trail),
    /// bucketing every propagator explanation's non-selector literals under the
    /// selector variable it carries. `sel_vars` are the selector variables.
    fn footprint_of_cone(&self, conflict: &Conflict, sel_vars: &HashSet<VarId>) -> Vec<(VarId, Vec<Lit>)> {
        let mut buckets: Vec<(VarId, Vec<Lit>)> = Vec::new();
        let mut seen: HashSet<Atom> = HashSet::new();

        // The failing propagator's own premises (all true literals).
        let conflict_ds: Vec<Lit> = match conflict {
            Conflict::Generic(ds) => ds.clone(),
            Conflict::Clause(c) => self.clauses.get(*c).lits.iter().map(|l| l.negate()).collect(),
        };
        self.bucket_footprint(&mut buckets, sel_vars, &conflict_ds);
        let mut stack: Vec<Lit> = conflict_ds;

        while let Some(t) = stack.pop() {
            if !seen.insert(t.atom()) {
                continue;
            }
            match self.reason_of(t.atom()) {
                // A propagator inference: `ds` are its true premises, tagged with
                // the constraint's selector by the `Selected` wrapper.
                Reason::Generic(ds) => {
                    let ds = ds.to_vec();
                    self.bucket_footprint(&mut buckets, sel_vars, &ds);
                    stack.extend_from_slice(&ds);
                }
                // A clause reason carries no selector here (no learning on this
                // path); just keep walking its antecedents.
                Reason::Clause(c) => {
                    let ante: Vec<Lit> = self.clauses.get(*c).lits.iter().copied().filter(|&l| l != t).map(|l| l.negate()).collect();
                    stack.extend_from_slice(&ante);
                }
                Reason::Decision | Reason::Fact | Reason::Unset => {} // leaves
            }
        }
        buckets
    }

    /// Add the non-selector literals of one propagator explanation `ds` to the
    /// footprint bucket of each selector variable `ds` carries.
    fn bucket_footprint(&self, buckets: &mut Vec<(VarId, Vec<Lit>)>, sel_vars: &HashSet<VarId>, ds: &[Lit]) {
        let is_sel = |l: &Lit| sel_vars.contains(&self.atoms.var_of(l.atom()));
        for sel in ds.iter().copied().filter(is_sel) {
            let sel_var = self.atoms.var_of(sel.atom());
            let entry = match buckets.iter_mut().find(|(v, _)| *v == sel_var) {
                Some(entry) => entry,
                None => {
                    buckets.push((sel_var, Vec::new()));
                    buckets.last_mut().unwrap()
                }
            };
            for &lit in ds {
                if !is_sel(&lit) && !entry.1.contains(&lit) {
                    entry.1.push(lit);
                }
            }
        }
    }

    /// Recursively minimize a learned clause using the implication graph.
    ///
    /// A non-asserting literal `q` is redundant when the reason for the assigned
    /// literal `not q` only depends on level-0 facts, literals already present in
    /// the learned clause, or other recursively redundant literals.
    fn minimize_learnt(&mut self, learnt: &mut Vec<Lit>) {
        if learnt.len() < 3 || learnt.len() > MINIMIZE_LEARNT_MAX_LEN {
            return;
        }
        // Reuse the persistent `seen` array (all-false after `analyze`) plus a
        // to-clear stack, so steady state allocates nothing. Both are borrowed
        // out to keep `self` free for the `&self` redundancy recursion, then
        // restored. `seen` marks exactly the atoms currently in `learnt` (the
        // base); the recursion's temporary marks are unwound after each literal
        // so the next sees only the base, matching the old per-literal copy.
        let mut seen = std::mem::take(&mut self.seen);
        let mut stack = std::mem::take(&mut self.min_stack);
        for &lit in learnt.iter() {
            seen[lit.atom() as usize] = true;
        }

        let mut i = 1;
        while i < learnt.len() {
            let mark = stack.len();
            let redundant = self.learnt_lit_redundant(learnt[i], &mut seen, &mut stack);
            while stack.len() > mark {
                seen[stack.pop().unwrap() as usize] = false;
            }
            if redundant {
                seen[learnt[i].atom() as usize] = false;
                learnt.remove(i);
            } else {
                i += 1;
            }
        }

        for &lit in learnt.iter() {
            seen[lit.atom() as usize] = false;
        }
        stack.clear();
        self.seen = seen;
        self.min_stack = stack;
    }

    fn learnt_lit_redundant(&self, lit: Lit, seen: &mut [bool], stack: &mut Vec<u32>) -> bool {
        let assigned = lit.negate();
        if matches!(self.reason_of(assigned.atom()), Reason::Decision | Reason::Fact | Reason::Unset) {
            return false;
        }
        let antecedents = self.reason_lits(assigned);
        if antecedents.is_empty() {
            return false;
        }
        for antecedent in antecedents {
            let atom = antecedent.atom() as usize;
            match self.level_of(antecedent.atom()) {
                0 => continue,
                UNSET => return false,
                _ if seen.get(atom).copied().unwrap_or(false) => continue,
                _ => {
                    seen[atom] = true;
                    stack.push(antecedent.atom());
                    if !self.learnt_lit_redundant(antecedent, seen, stack) {
                        return false;
                    }
                }
            }
        }
        true
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

    /// Set up the root: seed atom facts, enqueue every propagator, propagate to
    /// fixpoint. `false` if the root is already unsatisfiable.
    pub fn init(&mut self) -> bool {
        self.seed_facts();
        self.solver.enqueue_all();
        self.propagate_and_learn()
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
                self.clauses.get(c).lits.iter().copied().filter(|&l| l != p).collect()
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
            if lit.is_positive() {
                if let AtomKind::Eq { var, v } = self.atoms.decode(lit.atom()) {
                    self.saved_phase[var.index()] = Some(v);
                }
            }
        }
        self.level_starts.truncate(d + 1);
        for _ in 0..(cur - d) {
            self.solver.store.pop_level();
        }
        // Everything still on the trail was already propagated.
        self.qhead = self.trail.len();
    }
}
