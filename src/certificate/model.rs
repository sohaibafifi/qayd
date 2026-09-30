//! Bounded normalization of the explicitly supported semantic model slice.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::{Constraint, IntDomain, IntExpr, IntGlobalConstraint, ModelPackage, Objective, Relation};
use crate::orchestrator::{SemanticAssumptionOp, SolveRequest};

use super::{CertificateError, CertificateLimits};

pub(crate) struct Work<'a> {
    limits: &'a CertificateLimits,
    stop: &'a dyn Fn() -> bool,
    pub(crate) used: u64,
}

impl<'a> Work<'a> {
    pub(crate) fn new(limits: &'a CertificateLimits, stop: &'a dyn Fn() -> bool) -> Self {
        Self { limits, stop, used: 0 }
    }

    pub(crate) fn charge(&mut self, amount: usize) -> Result<(), CertificateError> {
        if (self.stop)() {
            return Err(CertificateError::Interrupted);
        }
        self.used = self
            .used
            .checked_add(u64::try_from(amount).map_err(|_| CertificateError::Limit("work".into()))?)
            .ok_or_else(|| CertificateError::Limit("work".into()))?;
        if self.used > self.limits.max_work {
            return Err(CertificateError::Limit("work".into()));
        }
        Ok(())
    }
}

pub(crate) struct Text {
    pub(crate) text: String,
    limit: usize,
}

impl Text {
    pub(crate) fn new(limit: usize) -> Self {
        Self { text: String::new(), limit }
    }

    pub(crate) fn push(&mut self, value: &str, work: &mut Work<'_>) -> Result<(), CertificateError> {
        work.charge(1)?;
        if self.text.len().checked_add(value.len()).is_none_or(|n| n > self.limit) {
            return Err(CertificateError::Limit("output bytes".into()));
        }
        self.text.push_str(value);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Domain {
    pub(crate) ranges: Vec<(i32, i32)>,
}

impl Domain {
    pub(crate) fn bounds(&self) -> Option<(i32, i32)> {
        Some((self.ranges.first()?.0, self.ranges.last()?.1))
    }

    pub(crate) fn contains(&self, value: i32) -> bool {
        let index = self.ranges.partition_point(|&(lo, _)| lo <= value);
        index > 0 && value <= self.ranges[index - 1].1
    }

    pub(crate) fn copy(&self, work: &mut Work<'_>) -> Result<Self, CertificateError> {
        let mut ranges = Vec::new();
        for &range in &self.ranges {
            work.charge(1)?;
            ranges.push(range);
        }
        Ok(Self { ranges })
    }

    /// Intersect compact ranges without enumerating a wide integer domain.
    pub(crate) fn apply(&mut self, relation: Relation, value: i32, work: &mut Work<'_>) -> Result<(), CertificateError> {
        let value = i64::from(value);
        let mut ranges = Vec::new();
        for &(lo, hi) in &self.ranges {
            work.charge(1)?;
            let (lo, hi) = (i64::from(lo), i64::from(hi));
            let mut add = |a: i64, b: i64| {
                if a <= b {
                    // Every added endpoint is clipped to an original i32 range.
                    ranges.push((a as i32, b as i32));
                }
            };
            match relation {
                Relation::Eq => add(lo.max(value), hi.min(value)),
                Relation::Ne => {
                    add(lo, hi.min(value - 1));
                    add(lo.max(value + 1), hi);
                }
                Relation::Le => add(lo, hi.min(value)),
                Relation::Lt => add(lo, hi.min(value - 1)),
                Relation::Ge => add(lo.max(value), hi),
                Relation::Gt => add(lo.max(value + 1), hi),
            }
        }
        self.ranges = ranges;
        Ok(())
    }

    pub(crate) fn small_values(&self, limit: usize, work: &mut Work<'_>) -> Result<Option<Vec<i32>>, CertificateError> {
        let mut count = 0_u64;
        for &(lo, hi) in &self.ranges {
            work.charge(1)?;
            count += (i64::from(hi) - i64::from(lo) + 1) as u64;
            if count > limit as u64 {
                return Ok(None);
            }
        }
        let mut values = Vec::new();
        for &(lo, hi) in &self.ranges {
            for value in i64::from(lo)..=i64::from(hi) {
                work.charge(1)?;
                values.push(value as i32);
            }
        }
        Ok(Some(values))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Predicate {
    pub(crate) variable: usize,
    pub(crate) relation: Relation,
    pub(crate) value: i32,
}

pub(crate) fn token(relation: Relation) -> &'static str {
    match relation {
        Relation::Eq => "eq",
        Relation::Ne => "ne",
        Relation::Le => "le",
        Relation::Lt => "lt",
        Relation::Ge => "ge",
        Relation::Gt => "gt",
    }
}

pub(crate) fn negate(relation: Relation) -> Relation {
    match relation {
        Relation::Eq => Relation::Ne,
        Relation::Ne => Relation::Eq,
        Relation::Le => Relation::Gt,
        Relation::Lt => Relation::Ge,
        Relation::Ge => Relation::Lt,
        Relation::Gt => Relation::Le,
    }
}

pub(crate) fn holds(left: i128, relation: Relation, right: i128) -> bool {
    match relation {
        Relation::Eq => left == right,
        Relation::Ne => left != right,
        Relation::Le => left <= right,
        Relation::Lt => left < right,
        Relation::Ge => left >= right,
        Relation::Gt => left > right,
    }
}

#[derive(Debug)]
pub(crate) enum ProofConstraint {
    Linear { terms: Vec<(i128, usize)>, relation: Relation, rhs: i128 },
    Table { variables: Vec<usize>, rows: Vec<Vec<Option<i32>>> },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ObjectiveTerm {
    Variable(usize),
    Constant(i64),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProofObjective {
    pub(crate) minimize: bool,
    pub(crate) term: ObjectiveTerm,
}

#[derive(Debug)]
pub(crate) struct ProofModel {
    pub(crate) domains: Vec<Domain>,
    pub(crate) constraints: Vec<ProofConstraint>,
    pub(crate) assumptions: Vec<Predicate>,
    pub(crate) objective: Option<ProofObjective>,
}

fn input_entries(total: &mut usize, extra: usize, limits: &CertificateLimits) -> Result<(), CertificateError> {
    *total = total.checked_add(extra).ok_or_else(|| CertificateError::Limit("input entries".into()))?;
    if *total > limits.max_input_entries {
        return Err(CertificateError::Limit("input entries".into()));
    }
    Ok(())
}

fn variable(id: usize, count: usize) -> Result<(), CertificateError> {
    if id >= count {
        return Err(CertificateError::InvalidInput(format!("unknown integer variable {id}")));
    }
    Ok(())
}

pub(crate) fn from_package(
    package: &ModelPackage,
    request: &SolveRequest,
    limits: &CertificateLimits,
    stop: &dyn Fn() -> bool,
) -> Result<ProofModel, CertificateError> {
    let mut work = Work::new(limits, stop);
    work.charge(1)?;
    let model = &package.model;
    if !model.sets.is_empty() || !model.lists.is_empty() || !model.intervals.is_empty() {
        return Err(CertificateError::Unsupported("certificate v1 supports integer variables only".into()));
    }
    if model.int_vars.len() > limits.max_variables || model.constraints.len() > limits.max_constraints {
        return Err(CertificateError::Limit("model variable or constraint count".into()));
    }
    if model.objectives.len() > 1 {
        return Err(CertificateError::Unsupported("multiple objectives".into()));
    }
    let mut entries = 0;
    let mut domains = Vec::new();
    for declaration in &model.int_vars {
        work.charge(1)?;
        let ranges = match declaration {
            IntDomain::Bool => {
                input_entries(&mut entries, 1, limits)?;
                vec![(0, 1)]
            }
            IntDomain::Range { lo, hi } => {
                input_entries(&mut entries, 1, limits)?;
                if lo > hi {
                    return Err(CertificateError::InvalidInput("empty integer range".into()));
                }
                vec![(*lo, *hi)]
            }
            IntDomain::Set(values) => {
                input_entries(&mut entries, values.len(), limits)?;
                if values.is_empty() {
                    return Err(CertificateError::InvalidInput("empty explicit domain".into()));
                }
                let mut sorted = BTreeSet::new();
                for &value in values {
                    work.charge(1)?;
                    sorted.insert(value);
                }
                let mut ranges: Vec<(i32, i32)> = Vec::new();
                for value in sorted {
                    work.charge(1)?;
                    if let Some(last) = ranges.last_mut() {
                        if i64::from(last.1) + 1 == i64::from(value) {
                            last.1 = value;
                            continue;
                        }
                    }
                    ranges.push((value, value));
                }
                ranges
            }
        };
        domains.push(Domain { ranges });
    }
    let mut constraints = Vec::new();
    for constraint in &model.constraints {
        work.charge(1)?;
        match constraint {
            Constraint::Linear { terms, relation, rhs } => {
                input_entries(&mut entries, terms.len(), limits)?;
                let mut aggregated = BTreeMap::<usize, i128>::new();
                let mut positive = 0_i128;
                let mut negative = 0_i128;
                for &(coefficient, id) in terms {
                    work.charge(1)?;
                    variable(id.0, domains.len())?;
                    if matches!(relation, Relation::Ge | Relation::Gt) && coefficient == i64::MIN {
                        return Err(CertificateError::Unsupported("native linear coefficient negation overflows i64".into()));
                    }
                    let (lo, hi) = domains[id.0].bounds().expect("validated root domain");
                    let a = i128::from(coefficient) * i128::from(lo);
                    let b = i128::from(coefficient) * i128::from(hi);
                    let min = a.min(b);
                    let max = a.max(b);
                    // Check ORIGINAL occurrences, before cancellation. This also
                    // bounds any ordering of native positive/negative partial sums.
                    if min < i128::from(i64::MIN) || max > i128::from(i64::MAX) {
                        return Err(CertificateError::Unsupported("native linear term activity can overflow i64".into()));
                    }
                    positive =
                        positive.checked_add(max.max(0)).ok_or_else(|| CertificateError::Unsupported("linear activity overflow".into()))?;
                    negative =
                        negative.checked_add(min.min(0)).ok_or_else(|| CertificateError::Unsupported("linear activity overflow".into()))?;
                    if positive > i128::from(i64::MAX) || negative < i128::from(i64::MIN) {
                        return Err(CertificateError::Unsupported("native linear partial activity can overflow i64".into()));
                    }
                    let entry = aggregated.entry(id.0).or_default();
                    *entry = entry
                        .checked_add(i128::from(coefficient))
                        .ok_or_else(|| CertificateError::Unsupported("aggregated coefficient overflow".into()))?;
                }
                // The native general-integer adapter also normalizes strict
                // relations and computes c - other_activity before division.
                // Check a conservative interval for every such intermediate,
                // even when a Boolean specialization could avoid the operation.
                let bound = match relation {
                    Relation::Lt => rhs.checked_sub(1),
                    Relation::Ge => rhs.checked_neg(),
                    Relation::Gt => rhs.checked_add(1).and_then(i64::checked_neg),
                    _ => Some(*rhs),
                }
                .ok_or_else(|| CertificateError::Unsupported("native linear relation normalization overflows i64".into()))?;
                let (native_negative, native_positive) =
                    if matches!(relation, Relation::Ge | Relation::Gt) { (-positive, -negative) } else { (negative, positive) };
                let smallest_numerator = i128::from(bound) - native_positive;
                let largest_numerator = i128::from(bound) - native_negative;
                if native_negative < i128::from(i64::MIN)
                    || native_positive > i128::from(i64::MAX)
                    || smallest_numerator <= i128::from(i64::MIN)
                    || largest_numerator > i128::from(i64::MAX)
                {
                    // Excluding MIN also rules out native MIN / -1 and MIN % -1.
                    return Err(CertificateError::Unsupported("native linear residual or division can overflow i64".into()));
                }
                let mut terms = Vec::new();
                for (id, coefficient) in aggregated {
                    work.charge(1)?;
                    if coefficient != 0 {
                        terms.push((coefficient, id));
                    }
                }
                constraints.push(ProofConstraint::Linear { terms, relation: *relation, rhs: i128::from(*rhs) });
            }
            Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, tuples, positive: true }) => {
                input_entries(&mut entries, variables.len(), limits)?;
                input_entries(&mut entries, tuples.len(), limits)?;
                let mut ids = Vec::new();
                for id in variables {
                    work.charge(1)?;
                    variable(id.0, domains.len())?;
                    ids.push(id.0);
                }
                let mut rows = Vec::new();
                for tuple in tuples.iter() {
                    work.charge(1)?;
                    if tuple.len() != variables.len() {
                        return Err(CertificateError::InvalidInput("ragged positive table".into()));
                    }
                    input_entries(&mut entries, tuple.len(), limits)?;
                    let mut row = Vec::new();
                    for &value in tuple {
                        work.charge(1)?;
                        row.push((value != i32::MIN).then_some(value));
                    }
                    rows.push(row);
                }
                constraints.push(ProofConstraint::Table { variables: ids, rows });
            }
            _ => return Err(CertificateError::Unsupported("certificate v1 supports Linear and positive Table constraints only".into())),
        }
    }
    input_entries(&mut entries, request.assumptions.len(), limits)?;
    let mut assumptions = Vec::new();
    for assumption in &request.assumptions {
        work.charge(1)?;
        variable(assumption.variable, domains.len())?;
        let relation = match assumption.operation {
            SemanticAssumptionOp::Eq => Relation::Eq,
            SemanticAssumptionOp::Ne => Relation::Ne,
            SemanticAssumptionOp::Le => Relation::Le,
            SemanticAssumptionOp::Lt => Relation::Lt,
            SemanticAssumptionOp::Ge => Relation::Ge,
            SemanticAssumptionOp::Gt => Relation::Gt,
        };
        assumptions.push(Predicate { variable: assumption.variable, relation, value: assumption.value });
    }
    let objective = match model.objectives.first() {
        None => None,
        Some(Objective::IntExpr { minimize, expr }) => {
            work.charge(1)?;
            let term = match expr {
                IntExpr::Variable(id) => {
                    variable(id.0, domains.len())?;
                    ObjectiveTerm::Variable(id.0)
                }
                IntExpr::Constant(value) => ObjectiveTerm::Constant(*value),
                _ => return Err(CertificateError::Unsupported("certificate objective must be a variable or constant".into())),
            };
            Some(ProofObjective { minimize: *minimize, term })
        }
        _ => return Err(CertificateError::Unsupported("certificate objective must be an integer variable or constant".into())),
    };
    work.charge(0)?;
    Ok(ProofModel { domains, constraints, assumptions, objective })
}

impl ProofModel {
    pub(crate) fn encode(&self, limits: &CertificateLimits, stop: &dyn Fn() -> bool) -> Result<String, CertificateError> {
        let mut work = Work::new(limits, stop);
        let mut output = Text::new(limits.max_model_bytes);
        output.push(&format!("qayd-cp-model 1\nvars {}\n", self.domains.len()), &mut work)?;
        for (id, domain) in self.domains.iter().enumerate() {
            output.push(&format!("var {id} ranges {}", domain.ranges.len()), &mut work)?;
            for &(lo, hi) in &domain.ranges {
                output.push(&format!(" {lo} {hi}"), &mut work)?;
            }
            output.push("\n", &mut work)?;
        }
        output.push(&format!("constraints {}\n", self.constraints.len()), &mut work)?;
        for (id, constraint) in self.constraints.iter().enumerate() {
            match constraint {
                ProofConstraint::Linear { terms, relation, rhs } => {
                    output.push(&format!("linear {id} {} {rhs} {}", token(*relation), terms.len()), &mut work)?;
                    for &(coefficient, variable) in terms {
                        output.push(&format!(" {coefficient} {variable}"), &mut work)?;
                    }
                    output.push("\n", &mut work)?;
                }
                ProofConstraint::Table { variables, rows } => {
                    output.push(&format!("table {id} {} {}", variables.len(), rows.len()), &mut work)?;
                    for variable in variables {
                        output.push(&format!(" {variable}"), &mut work)?;
                    }
                    output.push("\n", &mut work)?;
                    for row in rows {
                        output.push("row", &mut work)?;
                        for value in row {
                            match value {
                                Some(value) => output.push(&format!(" {value}"), &mut work)?,
                                None => output.push(" *", &mut work)?,
                            }
                        }
                        output.push("\n", &mut work)?;
                    }
                }
            }
        }
        output.push(&format!("assumptions {}\n", self.assumptions.len()), &mut work)?;
        for predicate in &self.assumptions {
            output.push(&format!("assume {} {} {}\n", predicate.variable, token(predicate.relation), predicate.value), &mut work)?;
        }
        match self.objective {
            None => output.push("objective none\n", &mut work)?,
            Some(objective) => {
                let sense = if objective.minimize { "min" } else { "max" };
                match objective.term {
                    ObjectiveTerm::Variable(id) => output.push(&format!("objective {sense} var {id}\n"), &mut work)?,
                    ObjectiveTerm::Constant(value) => output.push(&format!("objective {sense} const {value}\n"), &mut work)?,
                }
            }
        }
        output.push("end-model\n", &mut work)?;
        Ok(output.text)
    }
}
