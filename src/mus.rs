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
use crate::lcg::lit::{Lit, LitOrConst};
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
    let mut cdcl = Cdcl::new(solver, vars);
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
