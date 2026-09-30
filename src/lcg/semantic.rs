//! Bounded, deletion-only semantic minimization of learned integer clauses.
//!
//! For a clause `q OR R`, removing `q` is sound exactly when `q => R`.
//! We prove a sufficient condition using only the other literals on `q`'s
//! variable: `q AND NOT R_same_var` has no integer solution in the immutable
//! root interval. The interval is a conservative superset for sparse domains.
//! No live domain, reason, new atom, or additional CP inference is consulted.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::lcg::lit::{AtomKind, AtomTable, Lit};

/// Hard bound on optional work per conflict. Domain width does not affect cost.
pub(crate) const MAX_CLAUSE_LEN: usize = 128;

/// Counters for the optional semantic learned-clause minimizer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticMinimizationStats {
    /// Enabled, bounded clauses considered for semantic deletion.
    pub considered_clauses: u64,
    /// Clauses with at least one semantic deletion.
    pub shortened_clauses: u64,
    /// Non-asserting literals removed by semantic implication.
    pub removed_literals: u64,
    /// Clauses skipped because their length exceeded the work bound.
    pub skipped_oversized: u64,
    /// Clauses skipped because a Boolean proof logger was attached.
    pub skipped_proof_logging: u64,
    /// Passes stopped early by the shared cancellation flag.
    pub interrupted: u64,
}

/// Minimize in place, preserving `clause[0]` and the order of every survivor.
/// Returns the number of deletions and whether cancellation interrupted work.
/// Each completed deletion is sound even if a later candidate is interrupted.
pub(crate) fn minimize(atoms: &AtomTable, clause: &mut Vec<Lit>, stop: &AtomicBool) -> (usize, bool) {
    if clause.len() < 2 || clause.len() > MAX_CLAUSE_LEN {
        return (0, false);
    }
    if stop.load(Ordering::Relaxed) {
        return (0, true);
    }

    // Decode once. Lazy atoms otherwise require a registry lock on every visit.
    let mut meanings: Vec<_> = clause.iter().map(|lit| atoms.decode(lit.atom())).collect();
    let original_len = clause.len();
    let mut excluded = Vec::with_capacity(clause.len());
    let mut index = 1;
    while index < clause.len() {
        if stop.load(Ordering::Relaxed) {
            return (original_len - clause.len(), true);
        }
        let variable = match meanings[index] {
            AtomKind::Ge { var, .. } | AtomKind::Eq { var, .. } => var,
        };
        let (root_lo, root_hi) = atoms.initial_bounds(variable);
        let mut lo = i64::from(root_lo);
        let mut hi = i64::from(root_hi);
        excluded.clear();
        for (other, (&literal, &meaning)) in clause.iter().zip(&meanings).enumerate() {
            let other_var = match meaning {
                AtomKind::Ge { var, .. } | AtomKind::Eq { var, .. } => var,
            };
            if variable != other_var {
                continue;
            }
            // Candidate is true; every surviving same-variable disjunct false.
            let positive = literal.is_positive() == (other == index);
            match (meaning, positive) {
                (AtomKind::Ge { k, .. }, true) => lo = lo.max(i64::from(k)),
                (AtomKind::Ge { k, .. }, false) => hi = hi.min(i64::from(k) - 1),
                (AtomKind::Eq { v, .. }, true) => {
                    lo = lo.max(i64::from(v));
                    hi = hi.min(i64::from(v));
                }
                (AtomKind::Eq { v, .. }, false) => excluded.push(i64::from(v)),
            }
            if lo > hi {
                break;
            }
        }
        let empty = if lo > hi {
            true
        } else {
            // There are at most MAX_CLAUSE_LEN exclusions, including on wide
            // domains. Counting distinct exclusions avoids enumerating values.
            excluded.sort_unstable();
            excluded.dedup();
            let count = excluded.iter().filter(|&&value| lo <= value && value <= hi).count();
            count as i64 == hi - lo + 1
        };
        if empty {
            clause.remove(index);
            meanings.remove(index);
        } else {
            index += 1;
        }
    }
    (original_len - clause.len(), false)
}
