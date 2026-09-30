//! Independent, bounded chronological CP proof search.
//!
//! This is deliberately separate from Store, native LCG and its learned clauses.
//! Each exported inference is justified by the frozen v1 impossibility rules.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use crate::model::Relation;

use super::model::{holds, negate, token, Domain, ObjectiveTerm, Predicate, ProofConstraint, ProofModel, ProofObjective, Text, Work};
use super::{CertificateClaim, CertificateError, CertificateLimits, CertificateStats};

pub(crate) struct GeneratedCertificate {
    pub(crate) text: String,
    pub(crate) stats: CertificateStats,
}

#[derive(Clone, Copy)]
enum Cid {
    Domain,
    Constraint(usize),
    Cut,
}

impl Cid {
    fn token(self) -> String {
        match self {
            Self::Domain => "D".into(),
            Self::Constraint(id) => id.to_string(),
            Self::Cut => "C".into(),
        }
    }
}

#[derive(Clone, Copy)]
struct Cut {
    objective: ProofObjective,
    bound: i128,
}

fn overflow() -> CertificateError {
    CertificateError::InvalidInput("exact linear activity exceeds i128".into())
}

/// An override is temporary, so a negated-conclusion probe need only copy one
/// compact domain. All other domains are exactly the current active facts.
fn impossible(
    model: &ProofModel,
    cut: Option<Cut>,
    cid: Cid,
    domains: &[Domain],
    replacement: Option<(usize, &Domain)>,
    work: &mut Work<'_>,
) -> Result<bool, CertificateError> {
    let domain = |id: usize| -> &Domain {
        match replacement {
            Some((variable, replacement)) if variable == id => replacement,
            _ => &domains[id],
        }
    };
    for id in 0..domains.len() {
        work.charge(1)?;
        if domain(id).ranges.is_empty() {
            return Ok(true);
        }
    }
    match cid {
        Cid::Domain => Ok(false),
        Cid::Cut => {
            work.charge(1)?;
            let cut = cut.ok_or_else(|| CertificateError::InvalidInput("missing strict objective cut".into()))?;
            let (lo, hi) = match cut.objective.term {
                ObjectiveTerm::Constant(value) => (i128::from(value), i128::from(value)),
                ObjectiveTerm::Variable(variable) => {
                    let (lo, hi) = domain(variable).bounds().expect("nonempty working domain");
                    (i128::from(lo), i128::from(hi))
                }
            };
            Ok(if cut.objective.minimize { lo >= cut.bound } else { hi <= cut.bound })
        }
        Cid::Constraint(id) => match &model.constraints[id] {
            ProofConstraint::Linear { terms, relation, rhs } => {
                let (mut min, mut max) = (0_i128, 0_i128);
                for &(coefficient, variable) in terms {
                    work.charge(1)?;
                    let (lo, hi) = domain(variable).bounds().expect("nonempty working domain");
                    let a = coefficient.checked_mul(i128::from(lo)).ok_or_else(overflow)?;
                    let b = coefficient.checked_mul(i128::from(hi)).ok_or_else(overflow)?;
                    min = min.checked_add(a.min(b)).ok_or_else(overflow)?;
                    max = max.checked_add(a.max(b)).ok_or_else(overflow)?;
                }
                Ok(match relation {
                    Relation::Eq => *rhs < min || *rhs > max,
                    Relation::Ne => min == max && min == *rhs,
                    Relation::Le => min > *rhs,
                    Relation::Lt => min >= *rhs,
                    Relation::Ge => max < *rhs,
                    Relation::Gt => max <= *rhs,
                })
            }
            ProofConstraint::Table { variables, rows } => {
                for row in rows {
                    work.charge(1)?;
                    let mut assigned = BTreeMap::new();
                    let mut compatible = true;
                    for (&variable, &value) in variables.iter().zip(row) {
                        work.charge(1)?;
                        if let Some(value) = value {
                            if !domain(variable).contains(value) || assigned.insert(variable, value).is_some_and(|old| old != value) {
                                compatible = false;
                                break;
                            }
                        }
                    }
                    if compatible {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        },
    }
}

struct State {
    domains: Vec<Domain>,
    facts: Vec<u64>,
    decisions: Vec<u64>,
}

impl State {
    fn copy(&self, work: &mut Work<'_>) -> Result<Self, CertificateError> {
        let mut domains = Vec::new();
        for domain in &self.domains {
            work.charge(1)?;
            domains.push(domain.copy(work)?);
        }
        let mut facts = Vec::new();
        for &id in &self.facts {
            work.charge(1)?;
            facts.push(id);
        }
        let mut decisions = Vec::new();
        for &id in &self.decisions {
            work.charge(1)?;
            decisions.push(id);
        }
        Ok(Self { domains, facts, decisions })
    }
}

struct Producer<'a> {
    model: &'a ProofModel,
    cut: Option<Cut>,
    limits: &'a CertificateLimits,
    work: Work<'a>,
    output: Text,
    stats: CertificateStats,
}

impl Producer<'_> {
    fn step(&mut self) -> Result<u64, CertificateError> {
        self.work.charge(1)?;
        self.stats.steps = self.stats.steps.checked_add(1).ok_or_else(|| CertificateError::Limit("proof steps".into()))?;
        if self.stats.steps > self.limits.max_steps {
            return Err(CertificateError::Limit("proof steps".into()));
        }
        Ok(self.stats.steps)
    }

    fn premises(&mut self, ids: &[u64]) -> Result<(), CertificateError> {
        self.output.push(&format!(" {}", ids.len()), &mut self.work)?;
        for id in ids {
            self.output.push(&format!(" {id}"), &mut self.work)?;
        }
        self.output.push("\n", &mut self.work)
    }

    fn conflict(&mut self, cid: Cid, state: &State) -> Result<u64, CertificateError> {
        let conflict = self.step()?;
        self.output.push(&format!("conflict {conflict} {}", cid.token()), &mut self.work)?;
        self.premises(&state.facts)?;
        self.stats.conflicts += 1;
        let proof = self.step()?;
        self.output.push(&format!("learn {proof} {conflict}"), &mut self.work)?;
        self.premises(&state.decisions)?;
        self.stats.learned_clauses += 1;
        Ok(proof)
    }

    fn probe(&mut self, cid: Cid, state: &State, predicate: Predicate) -> Result<bool, CertificateError> {
        let mut domain = state.domains[predicate.variable].copy(&mut self.work)?;
        domain.apply(predicate.relation, predicate.value, &mut self.work)?;
        impossible(self.model, self.cut, cid, &state.domains, Some((predicate.variable, &domain)), &mut self.work)
    }

    fn infer(&mut self, cid: Cid, state: &mut State, predicate: Predicate) -> Result<bool, CertificateError> {
        let mut narrowed = state.domains[predicate.variable].copy(&mut self.work)?;
        narrowed.apply(predicate.relation, predicate.value, &mut self.work)?;
        self.work.charge(narrowed.ranges.len())?;
        if narrowed == state.domains[predicate.variable] {
            return Ok(false);
        }
        let negated = Predicate { relation: negate(predicate.relation), ..predicate };
        if !self.probe(cid, state, negated)? {
            return Err(CertificateError::Incomplete("candidate deduction failed its negated-conclusion check".into()));
        }
        let id = self.step()?;
        self.output.push(
            &format!("infer {id} {} {} {} {}", cid.token(), predicate.variable, token(predicate.relation), predicate.value),
            &mut self.work,
        )?;
        self.premises(&state.facts)?;
        state.domains[predicate.variable] = narrowed;
        state.facts.push(id);
        self.stats.deductions += 1;
        Ok(true)
    }

    fn variables(&mut self, cid: Cid) -> Result<Vec<usize>, CertificateError> {
        let mut variables = BTreeSet::new();
        match cid {
            Cid::Domain => {}
            Cid::Cut => {
                if let Some(Cut { objective: ProofObjective { term: ObjectiveTerm::Variable(variable), .. }, .. }) = self.cut {
                    variables.insert(variable);
                }
            }
            Cid::Constraint(id) => match &self.model.constraints[id] {
                ProofConstraint::Linear { terms, .. } => {
                    for &(_, variable) in terms {
                        self.work.charge(1)?;
                        variables.insert(variable);
                    }
                }
                ProofConstraint::Table { variables: positions, .. } => {
                    for &variable in positions {
                        self.work.charge(1)?;
                        variables.insert(variable);
                    }
                }
            },
        }
        let mut result = Vec::new();
        for variable in variables {
            self.work.charge(1)?;
            result.push(variable);
        }
        Ok(result)
    }

    /// Impossibility is monotone under restriction for every v1 rule. Binary
    /// search finds useful safe bounds, including over holes, without division
    /// rounding or enumeration of a wide domain. Each result is checked again
    /// by infer(), not accepted on the binary search's intermediate state.
    fn filter_bounds(&mut self, cid: Cid, state: &mut State, variable: usize) -> Result<bool, CertificateError> {
        let (lo, hi) = state.domains[variable].bounds().expect("conflict checked before filtering");
        if lo == hi {
            return Ok(false);
        }
        let (mut left, mut right) = (i64::from(lo), i64::from(hi));
        while left < right {
            self.work.charge(1)?;
            let middle = left + (right - left) / 2;
            if self.probe(cid, state, Predicate { variable, relation: Relation::Le, value: middle as i32 })? {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        let mut changed = false;
        if left > i64::from(lo) {
            changed |= self.infer(cid, state, Predicate { variable, relation: Relation::Ge, value: left as i32 })?;
        }
        if state.domains[variable].ranges.is_empty() {
            return Ok(changed);
        }
        let (lo, hi) = state.domains[variable].bounds().expect("nonempty filtered domain");
        let (mut left, mut right) = (i64::from(lo), i64::from(hi));
        while left < right {
            self.work.charge(1)?;
            let middle = left + (right - left + 1) / 2;
            if self.probe(cid, state, Predicate { variable, relation: Relation::Ge, value: middle as i32 })? {
                right = middle - 1;
            } else {
                left = middle;
            }
        }
        if left < i64::from(hi) {
            changed |= self.infer(cid, state, Predicate { variable, relation: Relation::Le, value: left as i32 })?;
        }
        Ok(changed)
    }

    fn propagate(&mut self, state: &mut State) -> Result<Option<u64>, CertificateError> {
        loop {
            self.work.charge(1)?;
            if impossible(self.model, self.cut, Cid::Domain, &state.domains, None, &mut self.work)? {
                return self.conflict(Cid::Domain, state).map(Some);
            }
            let mut changed = false;
            let count = self.model.constraints.len() + usize::from(self.cut.is_some());
            for index in 0..count {
                self.work.charge(1)?;
                let cid = if index < self.model.constraints.len() { Cid::Constraint(index) } else { Cid::Cut };
                if impossible(self.model, self.cut, cid, &state.domains, None, &mut self.work)? {
                    return self.conflict(cid, state).map(Some);
                }
                for variable in self.variables(cid)? {
                    self.work.charge(1)?;
                    changed |= self.filter_bounds(cid, state, variable)?;
                    if impossible(self.model, self.cut, cid, &state.domains, None, &mut self.work)? {
                        return self.conflict(cid, state).map(Some);
                    }
                    // Finite positive-table supports may leave internal holes.
                    // This bounded supplement never enumerates a wide domain.
                    if matches!(cid, Cid::Constraint(id) if matches!(&self.model.constraints[id], ProofConstraint::Table { .. })) {
                        if let Some(values) = state.domains[variable].small_values(256, &mut self.work)? {
                            for value in values {
                                self.work.charge(1)?;
                                if self.probe(cid, state, Predicate { variable, relation: Relation::Eq, value })? {
                                    changed |= self.infer(cid, state, Predicate { variable, relation: Relation::Ne, value })?;
                                }
                            }
                        }
                    }
                }
            }
            if !changed {
                return Ok(None);
            }
        }
    }

    fn search(&mut self, mut state: State, depth: usize) -> Result<u64, CertificateError> {
        self.work.charge(1)?;
        if depth > self.limits.max_depth {
            return Err(CertificateError::Limit("search depth".into()));
        }
        self.stats.nodes = self.stats.nodes.checked_add(1).ok_or_else(|| CertificateError::Limit("search nodes".into()))?;
        if self.stats.nodes > self.limits.max_nodes {
            return Err(CertificateError::Limit("search nodes".into()));
        }
        if let Some(proof) = self.propagate(&mut state)? {
            return Ok(proof);
        }
        let mut branch = None;
        for (variable, domain) in state.domains.iter().enumerate() {
            self.work.charge(1)?;
            let (lo, hi) = domain.bounds().expect("propagation removed empty domains");
            if lo < hi {
                branch = Some((variable, lo, hi));
                break;
            }
        }
        let Some((variable, lo, hi)) = branch else {
            // Every original constraint was just checked on singleton domains.
            // Such a leaf refutes the native UNSAT/OPTIMAL claim.
            return Err(CertificateError::ClaimDisagreement(
                "proof search found a satisfying assignment in the claimed refutation scope".into(),
            ));
        };
        if depth >= self.limits.max_depth {
            return Err(CertificateError::Limit("search depth".into()));
        }
        let pivot = (i64::from(lo) + (i64::from(hi) - i64::from(lo)) / 2) as i32;
        let split = self.step()?;
        self.output.push(&format!("split {split} {variable} {pivot}\n"), &mut self.work)?;
        self.stats.splits += 1;
        let mut left = state.copy(&mut self.work)?;
        left.domains[variable].apply(Relation::Le, pivot, &mut self.work)?;
        left.facts.push(split);
        left.decisions.push(split);
        let left_proof = self.search(left, depth + 1)?;
        let right_id = self.step()?;
        self.output.push(&format!("right {right_id} {split} {left_proof}\n"), &mut self.work)?;
        state.domains[variable].apply(Relation::Gt, pivot, &mut self.work)?;
        state.facts.push(right_id);
        state.decisions.push(right_id);
        let right_proof = self.search(state, depth + 1)?;
        let proof = self.step()?;
        self.output.push(&format!("resolve {proof} {split} {right_proof}\n"), &mut self.work)?;
        Ok(proof)
    }
}

fn validate_witness(model: &ProofModel, witness: &[i32], work: &mut Work<'_>) -> Result<(), CertificateError> {
    work.charge(1)?;
    if witness.len() != model.domains.len() {
        return Err(CertificateError::ClaimDisagreement("witness length differs from semantic variable count".into()));
    }
    let mut domains = Vec::new();
    for (variable, &value) in witness.iter().enumerate() {
        work.charge(1)?;
        if !model.domains[variable].contains(value) {
            return Err(CertificateError::ClaimDisagreement(format!("witness violates root domain {variable}")));
        }
        domains.push(Domain { ranges: vec![(value, value)] });
    }
    for predicate in &model.assumptions {
        work.charge(1)?;
        if !holds(i128::from(witness[predicate.variable]), predicate.relation, i128::from(predicate.value)) {
            return Err(CertificateError::ClaimDisagreement("witness violates a declared assumption".into()));
        }
    }
    for id in 0..model.constraints.len() {
        work.charge(1)?;
        if impossible(model, None, Cid::Constraint(id), &domains, None, work)? {
            return Err(CertificateError::ClaimDisagreement(format!("witness violates constraint {id}")));
        }
    }
    Ok(())
}

pub(crate) fn generate(
    model: &ProofModel,
    claim: CertificateClaim,
    witness: Option<&[i32]>,
    model_sha256: &str,
    limits: &CertificateLimits,
    stop: &dyn Fn() -> bool,
) -> Result<GeneratedCertificate, CertificateError> {
    let started = Instant::now();
    let mut work = Work::new(limits, stop);
    work.charge(1)?;
    if model_sha256.len() != 64 || !model_sha256.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return Err(CertificateError::InvalidInput("model SHA-256 must contain 64 lowercase hexadecimal digits".into()));
    }
    match (claim, witness) {
        (CertificateClaim::Unsatisfiable, None) => {}
        (CertificateClaim::Satisfiable | CertificateClaim::Optimal, Some(witness)) => validate_witness(model, witness, &mut work)?,
        _ => return Err(CertificateError::ClaimDisagreement("claim has an incompatible witness".into())),
    }
    let cut = if claim == CertificateClaim::Optimal {
        let objective = model.objective.ok_or_else(|| CertificateError::ClaimDisagreement("optimal claim has no objective".into()))?;
        let bound = match objective.term {
            ObjectiveTerm::Constant(value) => i128::from(value),
            ObjectiveTerm::Variable(variable) => i128::from(witness.expect("optimal witness validated")[variable]),
        };
        Some(Cut { objective, bound })
    } else {
        None
    };
    let mut producer = Producer { model, cut, limits, work, output: Text::new(limits.max_proof_bytes), stats: CertificateStats::default() };
    producer.output.push(&format!("qayd-cp-proof 1 {model_sha256} {}\n", claim.token()), &mut producer.work)?;
    if let Some(witness) = witness {
        producer.output.push(&format!("witness {}", witness.len()), &mut producer.work)?;
        for value in witness {
            producer.output.push(&format!(" {value}"), &mut producer.work)?;
        }
        producer.output.push("\n", &mut producer.work)?;
    } else {
        producer.output.push("witness none\n", &mut producer.work)?;
    }
    if let Some(cut) = cut {
        let sense = if cut.objective.minimize { "min" } else { "max" };
        producer.output.push(&format!("bound {sense} {}\n", cut.bound), &mut producer.work)?;
    } else {
        producer.output.push("bound none\n", &mut producer.work)?;
    }
    producer.output.push("begin\n", &mut producer.work)?;
    if claim == CertificateClaim::Satisfiable {
        producer.output.push("finish-sat\n", &mut producer.work)?;
    } else {
        let mut domains = Vec::new();
        for domain in &model.domains {
            producer.work.charge(1)?;
            domains.push(domain.copy(&mut producer.work)?);
        }
        // The root has only model domains and declared assumptions. The strict
        // objective cut is available exclusively through CID C, not implicit.
        for predicate in &model.assumptions {
            producer.work.charge(1)?;
            domains[predicate.variable].apply(predicate.relation, predicate.value, &mut producer.work)?;
        }
        let proof = producer.search(State { domains, facts: Vec::new(), decisions: Vec::new() }, 0)?;
        producer.output.push(&format!("finish {proof}\n"), &mut producer.work)?;
    }
    producer.work.charge(0)?;
    producer.stats.work = producer.work.used;
    producer.stats.proof_bytes = producer.output.text.len() as u64;
    producer.stats.generation_seconds = started.elapsed().as_secs_f64();
    Ok(GeneratedCertificate { text: producer.output.text, stats: producer.stats })
}
