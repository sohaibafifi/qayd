//! Bounded integer-semantic nogood propagation and CPIP experiment support.
//!
//! Clauses remain disjunctions in the SAT database. When their open literals
//! all concern one integer variable, remove the values making every literal
//! false. The engine applies this plan only at a synchronized trail boundary.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ids::VarId;
use crate::inference::InferenceLiteral;
use crate::lcg::lit::{AtomKind, AtomTable, Lit};
use crate::lcg::view::{self, Tri};
use crate::store::{Premise, Store};

pub const MAX_NOGOOD_CLAUSE_LEN: usize = 128;
pub const MAX_NOGOOD_DOMAIN_VALUES: usize = 4096;
pub(crate) const MAX_CPIP_RESOLUTION_STEPS: usize = 4096;
pub(crate) const MAX_CPIP_CHECK_LITERALS: usize = 16384;

/// Counters are engine-local. Peaks describe explicit temporary payloads only,
/// not total solver heap or process resident memory.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NogoodStats {
    pub extended_scans: u64,
    pub clause_visits: u64,
    pub predicate_visits: u64,
    pub value_visits: u64,
    pub propagations: u64,
    pub removed_values: u64,
    pub reasons: u64,
    pub reason_literals: u64,
    pub conflicts: u64,
    pub skipped_clause_limit: u64,
    pub skipped_domain_limit: u64,
    pub interrupted: u64,
    pub cpip_attempts: u64,
    pub cpip_resolution_steps: u64,
    pub cpip_learned: u64,
    pub cpip_multi_predicate: u64,
    pub cpip_asserting_predicates: u64,
    pub cpip_fallbacks: u64,
    pub peak_scratch_bytes: u64,
    pub skipped_proof_logging: u64,
}

impl NogoodStats {
    pub fn merge(&mut self, other: Self) {
        self.extended_scans = self.extended_scans.saturating_add(other.extended_scans);
        self.clause_visits = self.clause_visits.saturating_add(other.clause_visits);
        self.predicate_visits = self.predicate_visits.saturating_add(other.predicate_visits);
        self.value_visits = self.value_visits.saturating_add(other.value_visits);
        self.propagations = self.propagations.saturating_add(other.propagations);
        self.removed_values = self.removed_values.saturating_add(other.removed_values);
        self.reasons = self.reasons.saturating_add(other.reasons);
        self.reason_literals = self.reason_literals.saturating_add(other.reason_literals);
        self.conflicts = self.conflicts.saturating_add(other.conflicts);
        self.skipped_clause_limit = self.skipped_clause_limit.saturating_add(other.skipped_clause_limit);
        self.skipped_domain_limit = self.skipped_domain_limit.saturating_add(other.skipped_domain_limit);
        self.interrupted = self.interrupted.saturating_add(other.interrupted);
        self.cpip_attempts = self.cpip_attempts.saturating_add(other.cpip_attempts);
        self.cpip_resolution_steps = self.cpip_resolution_steps.saturating_add(other.cpip_resolution_steps);
        self.cpip_learned = self.cpip_learned.saturating_add(other.cpip_learned);
        self.cpip_multi_predicate = self.cpip_multi_predicate.saturating_add(other.cpip_multi_predicate);
        self.cpip_asserting_predicates = self.cpip_asserting_predicates.saturating_add(other.cpip_asserting_predicates);
        self.cpip_fallbacks = self.cpip_fallbacks.saturating_add(other.cpip_fallbacks);
        self.peak_scratch_bytes = self.peak_scratch_bytes.max(other.peak_scratch_bytes);
        self.skipped_proof_logging = self.skipped_proof_logging.saturating_add(other.skipped_proof_logging);
    }

    pub(crate) fn scratch(&mut self, bytes: usize) {
        self.peak_scratch_bytes = self.peak_scratch_bytes.max(bytes as u64);
    }
}

pub(crate) enum Plan {
    Inactive,
    Conflict,
    Interrupted,
    Remove { var: VarId, values: Vec<i32>, antecedents: Vec<Lit> },
}

/// Decode without normalizing endpoints, so even negated i32-bound predicates
/// retain their exact meaning in the independent checker.
pub(crate) fn inference_literal(atoms: &AtomTable, literal: Lit) -> InferenceLiteral {
    let predicate = match atoms.decode(literal.atom()) {
        AtomKind::Ge { var, k } => Premise::Ge { var, bound: k },
        AtomKind::Eq { var, v } => Premise::Eq { var, val: v },
    };
    InferenceLiteral { predicate, positive: literal.is_positive() }
}

pub(crate) fn premise(atoms: &AtomTable, literal: Lit) -> Premise {
    match (atoms.decode(literal.atom()), literal.is_positive()) {
        (AtomKind::Ge { var, k }, true) => Premise::Ge { var, bound: k },
        // Real order atoms are strictly above the root minimum, hence never
        // have i32::MIN as threshold. Constant endpoint predicates have no atom.
        (AtomKind::Ge { var, k }, false) => Premise::Le { var, bound: k.checked_sub(1).expect("nonconstant order atom threshold") },
        (AtomKind::Eq { var, v }, true) => Premise::Eq { var, val: v },
        (AtomKind::Eq { var, v }, false) => Premise::Ne { var, val: v },
    }
}

fn holds(atoms: &AtomTable, literal: Lit, value: i32) -> bool {
    let atom = match atoms.decode(literal.atom()) {
        AtomKind::Ge { k, .. } => value >= k,
        AtomKind::Eq { v, .. } => value == v,
    };
    atom == literal.is_positive()
}

pub(crate) fn plan(store: &Store, atoms: &AtomTable, clause: &[Lit], stop: &AtomicBool, stats: &mut NogoodStats) -> Plan {
    stats.clause_visits = stats.clause_visits.saturating_add(1);
    if clause.len() > MAX_NOGOOD_CLAUSE_LEN {
        stats.skipped_clause_limit = stats.skipped_clause_limit.saturating_add(1);
        return Plan::Inactive;
    }
    let mut target = None;
    for &literal in clause {
        if stop.load(Ordering::Relaxed) {
            stats.interrupted = stats.interrupted.saturating_add(1);
            return Plan::Interrupted;
        }
        stats.predicate_visits = stats.predicate_visits.saturating_add(1);
        match view::lit_value(store, atoms, literal) {
            Tri::True => return Plan::Inactive,
            Tri::False => {}
            Tri::Unknown => {
                let var = atoms.var_of(literal.atom());
                if target.is_some_and(|previous| previous != var) {
                    return Plan::Inactive;
                }
                target = Some(var);
            }
        }
    }
    let Some(var) = target else {
        return Plan::Conflict;
    };
    if store.size(var) > MAX_NOGOOD_DOMAIN_VALUES {
        stats.skipped_domain_limit = stats.skipped_domain_limit.saturating_add(1);
        return Plan::Inactive;
    }
    let antecedents =
        clause.iter().filter(|literal| atoms.var_of(literal.atom()) != var).map(|literal| literal.negate()).collect::<Vec<_>>();
    let mut values = Vec::new();
    for value in store.values(var) {
        if stop.load(Ordering::Relaxed) {
            stats.interrupted = stats.interrupted.saturating_add(1);
            return Plan::Interrupted;
        }
        stats.value_visits = stats.value_visits.saturating_add(1);
        let mut forbidden = true;
        for &literal in clause {
            stats.predicate_visits = stats.predicate_visits.saturating_add(1);
            if atoms.var_of(literal.atom()) == var && holds(atoms, literal, value) {
                forbidden = false;
                break;
            }
        }
        if forbidden {
            values.push(value);
        }
    }
    values.sort_unstable();
    stats.scratch(
        values.capacity().saturating_mul(std::mem::size_of::<i32>()) + antecedents.capacity().saturating_mul(std::mem::size_of::<Lit>()),
    );
    if values.is_empty() {
        Plan::Inactive
    } else {
        Plan::Remove { var, values, antecedents }
    }
}
