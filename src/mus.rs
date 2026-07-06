//! Constraint-level Minimal Unsatisfiable Subset (MUS) extraction.
//!
//! Post each candidate constraint under its own Boolean *selector* with
//! [`selected`] (`s → constraint`, half-reified). Assuming every selector true
//! and solving yields, on UNSAT, the subset of selectors that participated — an
//! unsat core at the constraint level (Brique 2: an over-approximation, cheap).
//! [`extract_mus`] then minimises that core with QuickXplain (Brique 3, Junker
//! 2004), reusing **one** learning engine across all its consistency checks:
//! every learned clause is entailed by the root constraints, so it stays valid
//! under any selector subset, making the minimisation genuinely incremental.

use std::sync::atomic::AtomicBool;

use crate::ids::VarId;
use crate::lcg::engine::AssumptionOutcome;
use crate::lcg::lit::{AtomKind, Lit, LitOrConst};
use crate::lcg::trail::Cdcl;
use crate::store::Solver;

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
    /// The selected constraints are jointly satisfiable — no MUS. Carries a
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
    // never touch the caller's model — keeps `extract_mus`/`explain_mus` pure and
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

    // Brique 2: assume every candidate on; a refutation core is the offending
    // constraints (an over-approximation of the MUS).
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

    // Brique 3: QuickXplain minimises the over-approximated core. Precondition —
    // the candidate set is inconsistent on its own (Brique 1 soundness).
    match quickxplain(&mut cdcl, vars, &on, stop, &[], false, &candidates) {
        None => MusResult::Interrupted,
        Some(minimal) => MusResult::Mus(minimal.into_iter().map(|i| sel_of[i]).collect()),
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

/// A sub-constraint MUS explanation (Brique 4): for each MUS constraint, the
/// selector and the specific atoms it used in the refutation — a Hall set, an
/// energy window — rather than the whole global.
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
/// explanation is out of scope for this prototype — the constraint-level MUS
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

/// `Some(true)` if activating exactly `active` (every other selector off) is
/// unsatisfiable, `Some(false)` if satisfiable, `None` if interrupted.
fn inconsistent(cdcl: &mut Cdcl<'_>, vars: &[VarId], on: &[Lit], active: &[usize], stop: &AtomicBool) -> Option<bool> {
    let mut mask = vec![false; on.len()];
    for &i in active {
        mask[i] = true;
    }
    let cube: Vec<Lit> = on.iter().enumerate().map(|(i, &l)| if mask[i] { l } else { l.negate() }).collect();
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
