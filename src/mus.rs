//! Constraint-level Minimal Unsatisfiable Subset (MUS) extraction.
//!
//! Post each candidate constraint under its own Boolean *selector* with
//! [`selected`] (`s → constraint`, half-reified). Assuming every selector true
//! and solving yields, on UNSAT, the subset of selectors that participated: an
//! unsat core at the constraint level (an over-approximation, cheap).
//! [`extract_mus`] then minimises that core with QuickXplain (Junker
//! 2004), reusing **one** learning engine across all its consistency checks:
//! every learned clause is entailed by the root constraints, so it stays valid
//! under any selector subset, making the minimisation genuinely incremental.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::ids::VarId;
use crate::lcg::engine::AssumptionOutcome;
use crate::lcg::lit::{AtomKind, Lit, LitOrConst};
use crate::lcg::trail::Cdcl;
use crate::search::{solve_bool_cnf_interruptible, BoolLit};
use crate::store::Solver;

/// Number of solver (re-)solves performed by MUS extraction - the dominant cost.
/// Instrumentation for tuning; call [`reset_solve_calls`] before a run and read
/// it after. Process-global and not thread-safe across concurrent extractions.
static SOLVE_CALLS: AtomicU64 = AtomicU64::new(0);

/// Read the MUS solver-call counter (see [`SOLVE_CALLS`]).
pub fn solve_calls() -> u64 {
    SOLVE_CALLS.load(Ordering::Relaxed)
}

/// Reset the MUS solver-call counter to zero.
pub fn reset_solve_calls() {
    SOLVE_CALLS.store(0, Ordering::Relaxed);
}

/// Post the constraints created by `post` under a fresh `{0,1}` selector, and
/// return that selector. The constraints are enforced only while the selector is
/// `1`; pass the selector to [`extract_mus`] to make it a MUS candidate.
///
/// ```
/// use qayd::constraints::linear::{sum, Relation};
/// use qayd::mus::selected;
/// use qayd::Solver;
///
/// let mut solver = Solver::new();
/// let a = solver.new_var_range(0, 1);
/// let s = selected(&mut solver, |solver| sum(solver, &[a], Relation::Ge, 1));
/// # let _ = s;
/// ```
pub fn selected<F: FnOnce(&mut Solver)>(solver: &mut Solver, post: F) -> VarId {
    let sel = solver.new_var_range(0, 1);
    solver.set_selector(Some(sel));
    post(solver);
    solver.set_selector(None);
    sel
}

/// Outcome of [`extract_mus`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MusResult {
    /// The selected constraints are jointly satisfiable - no MUS. Carries a
    /// satisfying assignment over `vars`.
    Sat(Vec<i32>),
    /// A minimal unsatisfiable subset of `selectors`. Empty means the root
    /// (non-selected) constraints are already unsatisfiable.
    Mus(Vec<VarId>),
    /// Interrupted by the stop flag before a status was decided.
    Interrupted,
}

/// Extract a minimal unsatisfiable subset of `selectors` (each a `{0,1}`
/// selector from [`selected`]) with respect to `solver`'s constraints over the
/// decision variables `vars`.
pub fn extract_mus<'a>(solver: &'a mut Solver, vars: &[VarId], selectors: &[VarId], stop: &'a AtomicBool) -> MusResult {
    // Work on a clone: the search mutates root domains (sound learned units), so
    // never touch the caller's model - keeps `extract_mus`/`explain_mus` pure and
    // composable on one solver.
    let mut work = solver.clone();
    let mut cdcl = Cdcl::new(&mut work, vars);
    cdcl.set_stop(stop);
    if !cdcl.init() {
        return MusResult::Mus(Vec::new()); // root already unsatisfiable
    }
    // The "on" literal (`sel = 1`) of each selector, and the selector it maps
    // back to. Selectors already root-fixed cannot be toggled, so they are
    // dropped: a fixed-0 constraint is inert, a fixed-1 one is unconditional.
    let mut on: Vec<Lit> = Vec::with_capacity(selectors.len());
    let mut sel_of: Vec<VarId> = Vec::with_capacity(selectors.len());
    for &s in selectors {
        if let LitOrConst::Lit(l) = cdcl.atoms.eq(s, 1) {
            on.push(l);
            sel_of.push(s);
        }
    }

    // assume every candidate on; a refutation core is the offending
    // constraints (an over-approximation of the MUS).
    SOLVE_CALLS.fetch_add(1, Ordering::Relaxed);
    let core = match cdcl.resolve_assumptions(vars, &on, stop) {
        AssumptionOutcome::Sat(model) => return MusResult::Sat(model),
        AssumptionOutcome::Interrupted => return MusResult::Interrupted,
        AssumptionOutcome::Unsat(core) => core,
    };
    let mut candidates: Vec<usize> = core.iter().filter_map(|l| on.iter().position(|x| x == l)).collect();
    candidates.sort_unstable();
    candidates.dedup();
    if candidates.is_empty() {
        return MusResult::Mus(Vec::new()); // level-0 refutation: root alone is unsat
    }

    match shrink_to_mus(&mut cdcl, vars, &on, candidates, stop) {
        None => MusResult::Interrupted,
        Some(minimal) => MusResult::Mus(minimal.into_iter().map(|i| sel_of[i]).collect()),
    }
}

/// Reduce a known-inconsistent candidate set to a single MUS: trim to a fixpoint
/// (cheap solves that report the participating subset), then QuickXplain-minimise
/// the residue. `candidates` must be inconsistent on its own; `None` on
/// interruption. Shared by [`extract_mus`] and MARCO enumeration.
fn shrink_to_mus(cdcl: &mut Cdcl<'_>, vars: &[VarId], on: &[Lit], mut candidates: Vec<usize>, stop: &AtomicBool) -> Option<Vec<usize>> {
    // Trimming: each refutation of the current set reports a (weakly) smaller
    // participating subset, so the super-linear QuickXplain runs on fewer items.
    loop {
        let refined = refine(cdcl, vars, on, &candidates, stop)?;
        let shrank = refined.len() < candidates.len();
        candidates = refined;
        if !shrank {
            break;
        }
    }
    quickxplain(cdcl, vars, on, stop, &[], false, &candidates)
}

/// Solve with exactly `active` selectors on (all others off) and return the
/// participating sub-core as sorted indices (⊆ `active`), or `None` if
/// interrupted. `active` must be unsatisfiable; a spurious SAT keeps it as-is.
fn refine(cdcl: &mut Cdcl<'_>, vars: &[VarId], on: &[Lit], active: &[usize], stop: &AtomicBool) -> Option<Vec<usize>> {
    let cube = active_cube(on, active);
    SOLVE_CALLS.fetch_add(1, Ordering::Relaxed);
    match cdcl.resolve_assumptions(vars, &cube, stop) {
        AssumptionOutcome::Unsat(core) => {
            let mut idx: Vec<usize> = core.iter().filter_map(|l| on.iter().position(|x| x == l)).collect();
            idx.sort_unstable();
            idx.dedup();
            Some(idx)
        }
        AssumptionOutcome::Sat(_) => Some(active.to_vec()),
        AssumptionOutcome::Interrupted => None,
    }
}

/// A relation in a [`MusAtom`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MusRel {
    /// `var = value`.
    Eq,
    /// `var ≠ value`.
    Ne,
    /// `var ≥ value`.
    Ge,
    /// `var ≤ value`.
    Le,
}

/// One semantic atom a constraint reasoned about in a refutation: `var rel value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MusAtom {
    pub var: VarId,
    pub rel: MusRel,
    pub value: i32,
}

/// A sub-constraint MUS explanation: for each MUS constraint, the
/// selector and the specific atoms it used in the refutation - a Hall set, an
/// energy window - rather than the whole global.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusExplanation {
    /// `(selector, atoms)` per constraint that participated in the refutation.
    pub constraints: Vec<(VarId, Vec<MusAtom>)>,
}

/// Explain *why* a MUS is unsatisfiable at sub-constraint granularity: the
/// specific semantic atoms each constraint contributed. `mus` is a set of
/// selectors (e.g. from [`extract_mus`]) whose constraints are jointly
/// unsatisfiable.
///
/// Returns `Some` when the MUS refutes by propagation alone (the common case
/// for a minimal global core: a pigeonhole `allDifferent`, an overloaded
/// `cumulative`); `None` when refuting it needs search, where a sub-constraint
/// explanation is out of scope for this prototype - the constraint-level MUS
/// from [`extract_mus`] still applies.
pub fn explain_mus<'a>(solver: &'a mut Solver, vars: &[VarId], mus: &[VarId], stop: &'a AtomicBool) -> Option<MusExplanation> {
    let mut work = solver.clone(); // pure w.r.t. the caller's model (see extract_mus)
    let mut cdcl = Cdcl::new(&mut work, vars);
    cdcl.set_stop(stop);
    if !cdcl.init() {
        return Some(MusExplanation { constraints: Vec::new() }); // root already unsatisfiable
    }
    let mut on: Vec<Lit> = Vec::with_capacity(mus.len());
    for &sel in mus {
        match cdcl.atoms.eq(sel, 1) {
            LitOrConst::Lit(lit) => on.push(lit),
            _ => return None, // a selector fixed at the root cannot be toggled
        }
    }
    let footprint = cdcl.explain_by_propagation(&on)?;
    let constraints = footprint
        .into_iter()
        .map(|(selector, atoms)| (selector, atoms.iter().map(|&lit| decode_atom(&cdcl, lit)).collect()))
        .collect();
    Some(MusExplanation { constraints })
}

/// Decode an LCG literal into its human-facing `var rel value` atom.
fn decode_atom(cdcl: &Cdcl<'_>, lit: Lit) -> MusAtom {
    let (var, rel, value) = match cdcl.atoms.decode(lit.atom()) {
        AtomKind::Eq { var, v } if lit.is_positive() => (var, MusRel::Eq, v),
        AtomKind::Eq { var, v } => (var, MusRel::Ne, v),
        AtomKind::Ge { var, k } if lit.is_positive() => (var, MusRel::Ge, k),
        AtomKind::Ge { var, k } => (var, MusRel::Le, k - 1),
    };
    MusAtom { var, rel, value }
}

/// The cube activating exactly `active`: assume those selectors on and leave
/// every other selector *free*. A free selector's [`Selected`] wrapper is
/// already inert (it only fires once fixed), so the constraint is off - same
/// effect as assuming it off, but with a cube of size `|active|` instead of
/// `|candidates|`, which is the dominant per-solve cost of QuickXplain.
fn active_cube(on: &[Lit], active: &[usize]) -> Vec<Lit> {
    active.iter().map(|&i| on[i]).collect()
}

/// `Some(true)` if activating exactly `active` is unsatisfiable, `Some(false)`
/// if satisfiable, `None` if interrupted.
fn inconsistent(cdcl: &mut Cdcl<'_>, vars: &[VarId], on: &[Lit], active: &[usize], stop: &AtomicBool) -> Option<bool> {
    let cube = active_cube(on, active);
    SOLVE_CALLS.fetch_add(1, Ordering::Relaxed);
    match cdcl.resolve_assumptions(vars, &cube, stop) {
        AssumptionOutcome::Unsat(_) => Some(true),
        AssumptionOutcome::Sat(_) => Some(false),
        AssumptionOutcome::Interrupted => None,
    }
}

/// QuickXplain (Junker 2004): the minimal subset of `candidates` that, together
/// with `background` active (all else off), is inconsistent. `background_grew`
/// records whether `background` just gained members, so we test it only when a
/// short-circuit is possible. Returns `None` on interruption.
fn quickxplain(
    cdcl: &mut Cdcl<'_>,
    vars: &[VarId],
    on: &[Lit],
    stop: &AtomicBool,
    background: &[usize],
    background_grew: bool,
    candidates: &[usize],
) -> Option<Vec<usize>> {
    if candidates.is_empty() {
        return Some(Vec::new()); // nothing to minimise (e.g. a search-only root refutation)
    }
    if background_grew && inconsistent(cdcl, vars, on, background, stop)? {
        return Some(Vec::new()); // background alone suffices; no candidate needed
    }
    if candidates.len() == 1 {
        return Some(candidates.to_vec()); // a lone survivor is essential
    }
    let mid = candidates.len() / 2; // ≥ 1 since len ≥ 2
    let (c1, c2) = candidates.split_at(mid);

    let mut b_c1: Vec<usize> = background.to_vec();
    b_c1.extend_from_slice(c1);
    let d2 = quickxplain(cdcl, vars, on, stop, &b_c1, !c1.is_empty(), c2)?;

    let mut b_d2: Vec<usize> = background.to_vec();
    b_d2.extend_from_slice(&d2);
    let d1 = quickxplain(cdcl, vars, on, stop, &b_d2, !d2.is_empty(), c1)?;

    let mut result = d1;
    result.extend(d2);
    Some(result)
}

// ---------------------------------------------------------------------------
// MARCO: enumerate ALL minimal unsatisfiable subsets and maximal satisfiable
// subsets (Liffiton, Previti, Malik, Marques-Silva, Constraints 2016).
// ---------------------------------------------------------------------------

/// The complete infeasibility landscape of a selected constraint set: every
/// minimal unsatisfiable subset (MUS) and every maximal satisfiable subset (MSS),
/// as selector sets. Each MSS's complement is a minimal correction set (MCS).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MusEnumeration {
    /// Every minimal unsatisfiable subset.
    pub muses: Vec<Vec<VarId>>,
    /// Every maximal satisfiable subset. `complement = selectors \ mss` is an MCS.
    pub msses: Vec<Vec<VarId>>,
    /// `true` if the search exhausted the space; `false` if it stopped early
    /// (interrupted or `limit` reached), so the lists may be partial.
    pub complete: bool,
}

/// Enumerate all MUSes and MSSes of `selectors` with respect to `solver`'s
/// constraints over `vars`, via the MARCO algorithm. A **map solver** (a native
/// SAT solver over one Boolean per selector) hands out unexplored subsets
/// ("seeds"); each seed is grown to an MSS (if satisfiable) or shrunk to a MUS
/// (if not), and its explored region is blocked out of the map. `limit` caps the
/// total number of MUS+MSS found (there can be exponentially many); `None` runs
/// to completion.
pub fn enumerate_mus<'a>(
    solver: &'a mut Solver,
    vars: &[VarId],
    selectors: &[VarId],
    stop: &'a AtomicBool,
    limit: Option<usize>,
) -> MusEnumeration {
    let mut out = MusEnumeration::default();
    // One oracle engine, cloned from the caller and reused across every
    // consistency check (learned clauses stay valid under any selector subset).
    let mut work = solver.clone();
    let mut cdcl = Cdcl::new(&mut work, vars);
    cdcl.set_stop(stop);
    if !cdcl.init() {
        out.muses.push(Vec::new()); // root alone is unsatisfiable: the empty MUS
        out.complete = true;
        return out;
    }
    let mut on: Vec<Lit> = Vec::new();
    let mut sel_of: Vec<VarId> = Vec::new();
    for &s in selectors {
        if let LitOrConst::Lit(l) = cdcl.atoms.eq(s, 1) {
            on.push(l);
            sel_of.push(s);
        }
    }
    let n = on.len();

    // Up-front root check on the FRESH engine: if the root constraints alone (no
    // selector active) are unsatisfiable, the empty set is the only MUS and there
    // are no MSSes. Must run before any other solve: a root proved unsat by
    // search leaves learned root units that empty a domain, which corrupts later
    // reuses of the engine (they would report a spurious model).
    match inconsistent(&mut cdcl, vars, &on, &[], stop) {
        None => return out, // interrupted (complete stays false)
        Some(true) => {
            out.muses.push(Vec::new());
            out.complete = true;
            return out;
        }
        Some(false) => {} // root satisfiable: enumerate normally
    }

    // The map solver is a plain SAT instance over `n` Booleans (m_i = "selector i
    // is in the seed"). Blocking clauses carve out explored super/sub-sets. We
    // rebuild a fresh map `Solver` each round (fast - the map is small) and read
    // its model; `VarId(i)` is always the i-th map variable, so clauses persist.
    let mut blocks: Vec<Vec<BoolLit>> = Vec::new();

    while limit.is_none_or(|k| out.muses.len() + out.msses.len() < k) {
        if stop.load(Ordering::Relaxed) {
            return out; // complete stays false
        }
        let seed = match map_seed(&blocks, n, stop) {
            MapOutcome::Seed(seed) => seed,
            MapOutcome::Exhausted => {
                out.complete = true; // map is UNSAT: every subset explored
                return out;
            }
            MapOutcome::Stopped => return out, // interrupted: complete stays false
        };
        match inconsistent(&mut cdcl, vars, &on, &seed, stop) {
            None => return out, // interrupted
            Some(false) => {
                // Seed is satisfiable → grow to a maximal satisfiable subset.
                let Some(mss) = grow(&mut cdcl, vars, &on, seed, n, stop) else {
                    return out;
                };
                // Block every subset of the MSS: a future seed must include at
                // least one selector *outside* it.
                blocks.push((0..n).filter(|i| !mss.contains(i)).map(bool_in).collect());
                out.msses.push(mss.into_iter().map(|i| sel_of[i]).collect());
            }
            Some(true) => {
                // Seed is unsatisfiable → shrink to a MUS.
                let Some(mus) = shrink_to_mus(&mut cdcl, vars, &on, seed, stop) else {
                    return out;
                };
                // Block every superset of the MUS: a future seed must *exclude*
                // at least one of its selectors.
                blocks.push(mus.iter().map(|&i| bool_out(i)).collect());
                out.muses.push(mus.into_iter().map(|i| sel_of[i]).collect());
            }
        }
    }
    out // limit reached: complete stays false
}

/// `m_i` positive (selector `i` is in the seed).
fn bool_in(i: usize) -> BoolLit {
    BoolLit::positive(VarId(i as u32))
}

/// `¬m_i` (selector `i` is out of the seed).
fn bool_out(i: usize) -> BoolLit {
    BoolLit::negative(VarId(i as u32))
}

/// Result of one map-solver query.
enum MapOutcome {
    /// An unexplored seed subset (selector indices with `m_i = 1`).
    Seed(Vec<usize>),
    /// The map is unsatisfiable: every subset has been explored.
    Exhausted,
    /// The solve was interrupted (or the map malformed) before a verdict.
    Stopped,
}

/// Solve the map for the next unexplored seed. Distinguishes a genuine
/// unsatisfiable map (search complete) from an interruption; conflating them
/// would report a partial enumeration as exhaustive.
fn map_seed(blocks: &[Vec<BoolLit>], n: usize, stop: &AtomicBool) -> MapOutcome {
    let mut map = Solver::new();
    let mvars: Vec<VarId> = (0..n).map(|_| map.new_var_range(0, 1)).collect();
    match solve_bool_cnf_interruptible(&mut map, &mvars, blocks, stop) {
        Ok((Some(model), _, _)) => MapOutcome::Seed((0..n).filter(|&i| model[i] == 1).collect()),
        Ok((None, _, true)) => MapOutcome::Exhausted, // `complete` true ⇒ genuine UNSAT
        _ => MapOutcome::Stopped,                      // interrupted, or a map error
    }
}

/// Grow a satisfiable subset to a maximal one: try to add each absent selector,
/// keeping it only while the set stays satisfiable. Reuses the oracle engine.
fn grow(cdcl: &mut Cdcl<'_>, vars: &[VarId], on: &[Lit], mut sat: Vec<usize>, n: usize, stop: &AtomicBool) -> Option<Vec<usize>> {
    let mut inside = vec![false; n];
    for &i in &sat {
        inside[i] = true;
    }
    #[allow(clippy::needless_range_loop)] // `i` is a selector index we push and test, not just an index into `inside`
    for i in 0..n {
        if inside[i] {
            continue;
        }
        sat.push(i);
        match inconsistent(cdcl, vars, on, &sat, stop)? {
            true => {
                sat.pop(); // adding `i` breaks satisfiability
            }
            false => inside[i] = true,
        }
    }
    sat.sort_unstable();
    Some(sat)
}
