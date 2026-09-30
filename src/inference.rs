//! Independent, bounded checks of individual propagation implications.
//!
//! The checker proves `constraint AND premises => conclusion` over immutable
//! root domains. The current snapshot is used only to verify that each cited
//! premise was true before the inference. No production propagator, support
//! cache, or domain filtering operation is called here.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::ids::{PropId, VarId};
use crate::store::Premise;

/// Absolute assignment ceiling, including when callers request a larger budget.
pub const MAX_INFERENCE_ASSIGNMENTS: usize = 1_000_000;
/// Absolute work ceiling, including when callers request a larger budget.
pub const MAX_INFERENCE_WORK: usize = 10_000_000;

/// Per-inference limits. Zero limits produce an uncovered inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InferenceCheckConfig {
    /// Maximum complete assignments (or raw positive-table rows) to inspect.
    pub max_assignments: usize,
    /// Maximum input visits and elementary semantic operations.
    pub max_work: usize,
}

impl Default for InferenceCheckConfig {
    fn default() -> Self {
        Self { max_assignments: 100_000, max_work: 1_000_000 }
    }
}

/// Counters for runtime checking. Assignment and work totals include only
/// completed checks; uncovered attempts have separate coverage counters.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct InferenceCheckStats {
    pub checked_inferences: u64,
    pub checked_conflicts: u64,
    pub uncovered_inferences: u64,
    pub uncovered_conflicts: u64,
    pub lcg_checked_inferences: u64,
    pub lcg_checked_conflicts: u64,
    /// Conditional entailment from the explicitly supplied source clauses.
    /// This is not certification of those clauses against the original model.
    pub lcg_checked_derivations: u64,
    pub lcg_uncovered_inferences: u64,
    pub lcg_uncovered_conflicts: u64,
    pub lcg_uncovered_derivations: u64,
    pub unsupported_calls: u64,
    pub resource_limited: u64,
    pub assignments: u64,
    pub work: u64,
    pub elapsed_nanos: u64,
    pub peak_snapshot_bytes: u64,
}

/// Relation of a linear expression to its right-hand side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceRelation {
    Le,
    Eq,
    Ne,
}

/// A signed integer predicate. Negation is logical, so endpoint predicates
/// never require an overflowing `bound + 1` or `bound - 1` conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InferenceLiteral {
    pub predicate: Premise,
    pub positive: bool,
}

/// One occurrence of a fixed-duration cumulative task. Distinct occurrences
/// consume resource independently, even when their start or presence variables
/// are shared. `None` is mandatory; `Some(var)` is present exactly at `var == 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InferenceCumulativeTask {
    pub start: VarId,
    pub duration: i64,
    pub demand: i64,
    pub presence: Option<VarId>,
}

/// Immutable semantics for the supported first slice of constraints.
#[derive(Clone, Debug)]
pub enum InferenceConstraint {
    /// Disjunction of signed integer predicates. A nogood's forbidden
    /// conjunction is represented by negating every condition here.
    Clause {
        literals: Vec<InferenceLiteral>,
    },
    Linear {
        terms: Vec<(VarId, i128)>,
        relation: InferenceRelation,
        rhs: i128,
    },
    /// Flat, row-major positive table. `i32::MIN` denotes a wildcard, matching
    /// the table representation used by the solver.
    PositiveTable {
        variables: Vec<VarId>,
        tuples: Arc<[i32]>,
        arity: usize,
        rows: usize,
    },
    /// Mandatory tasks with strictly positive, fixed durations. Every tuple is
    /// a distinct task, even when several tasks share the same start variable.
    /// Optional presences, order variables and zero-duration conventions are
    /// deliberately outside this descriptor's supported semantics.
    NoOverlap {
        tasks: Vec<(VarId, i64)>,
    },
    /// Half-open resource occupancy `[start, start + duration)`. Durations,
    /// demands and capacity must be nonnegative; zero-duration or zero-demand
    /// occurrences consume nothing. Presence variables add no implicit Boolean
    /// domain axiom: their frozen domains still determine allowed assignments.
    Cumulative {
        tasks: Vec<InferenceCumulativeTask>,
        capacity: i64,
    },
}

/// A domain described independently of mutable production domain machinery.
/// `values`, when present, is the immutable sparse root support. Bounds and
/// holes restrict it. Neither support nor holes need to be sorted.
#[derive(Clone, Debug)]
pub struct InferenceDomain {
    pub var: VarId,
    pub min: i32,
    pub max: i32,
    pub values: Option<Arc<[i32]>>,
    pub holes: Vec<i32>,
}

/// The asserted predicate, or a claim that the cited premises are infeasible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceConclusion {
    Predicate(Premise),
    Conflict,
}

/// An inference and its original constraint, roots, and pre-event snapshot.
#[derive(Clone, Copy, Debug)]
pub struct Inference<'a> {
    pub propagator: PropId,
    pub constraint: &'a InferenceConstraint,
    pub root_domains: &'a [InferenceDomain],
    pub premises: &'a [Premise],
    pub conclusion: InferenceConclusion,
    pub snapshot: &'a [InferenceDomain],
}

/// Why an implication could not be checked. Uncovered never means valid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InferenceUncoveredReason {
    AssignmentLimit,
    WorkLimit,
    ArithmeticOverflow,
    MissingRootDomain { var: VarId },
    MissingSnapshotDomain { var: VarId },
    InvalidDomain { var: VarId },
    InvalidConstraint(&'static str),
    Interrupted,
}

impl InferenceUncoveredReason {
    /// Whether coverage was limited by a numeric or resource bound.
    pub fn is_resource_limit(&self) -> bool {
        matches!(self, Self::AssignmentLimit | Self::WorkLimit | Self::ArithmeticOverflow)
    }
}

/// Owned diagnostic that survives backtracking or destruction of the solver.
#[derive(Clone, Debug)]
pub struct InferenceCheckFailure {
    pub propagator: PropId,
    pub constraint: String,
    pub premises: Vec<Premise>,
    pub conclusion: InferenceConclusion,
    pub snapshot: Vec<InferenceDomain>,
    pub counterexample: Option<Vec<(VarId, i32)>>,
    pub detail: String,
}

impl fmt::Display for InferenceCheckFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Keep panic/log formatting bounded even when sparse root supports
        // contain millions of values. The owned fields retain the full record.
        write!(
            f,
            "inference check failed for propagator {:?} ({}): {}; premises {:?} ({} total); conclusion {:?}; snapshot [",
            self.propagator,
            self.constraint,
            self.detail,
            &self.premises[..self.premises.len().min(32)],
            self.premises.len(),
            self.conclusion
        )?;
        for domain in self.snapshot.iter().take(16) {
            write!(
                f,
                "{:?}: {}..={}, support_size={:?}, holes={:?} ({} total); ",
                domain.var,
                domain.min,
                domain.max,
                domain.values.as_ref().map(|values| values.len()),
                &domain.holes[..domain.holes.len().min(16)],
                domain.holes.len()
            )?;
        }
        write!(
            f,
            "] ({} variables); counterexample {:?}",
            self.snapshot.len(),
            self.counterexample.as_ref().map(|values| &values[..values.len().min(32)])
        )
    }
}

impl std::error::Error for InferenceCheckFailure {}

/// A completed proof, a coverage gap, or a concrete invalid inference.
#[derive(Clone, Debug)]
pub enum InferenceCheckOutcome {
    Checked { assignments: u64, work: u64 },
    Uncovered(InferenceUncoveredReason),
    Rejected(InferenceCheckFailure),
}

/// Conditional clause entailment over frozen domains. Sources are hypotheses,
/// never implicitly certified original-model clauses. Current branch facts
/// must not be inserted as hypotheses to justify a globally learned clause.
#[derive(Clone, Copy, Debug)]
pub struct ClauseEntailment<'a> {
    pub root_domains: &'a [InferenceDomain],
    pub sources: &'a [&'a [InferenceLiteral]],
    pub conclusion: &'a [InferenceLiteral],
}

#[derive(Clone, Debug)]
pub struct ClauseCheckFailure {
    pub root_domains: Vec<InferenceDomain>,
    pub sources: Vec<Vec<InferenceLiteral>>,
    pub conclusion: Vec<InferenceLiteral>,
    pub counterexample: Vec<(VarId, i32)>,
    pub detail: String,
}

impl fmt::Display for ClauseCheckFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "conditional clause entailment failed: {}; {} source clauses; conclusion {:?} ({} literals); counterexample {:?}",
            self.detail,
            self.sources.len(),
            &self.conclusion[..self.conclusion.len().min(32)],
            self.conclusion.len(),
            &self.counterexample[..self.counterexample.len().min(32)]
        )?;
        for (index, source) in self.sources.iter().take(16).enumerate() {
            write!(f, "; source {index}: {:?} ({} literals)", &source[..source.len().min(32)], source.len())?;
        }
        for root in self.root_domains.iter().take(16) {
            write!(
                f,
                "; root {:?}: {}..={}, support_size={:?}, holes={:?} ({} total)",
                root.var,
                root.min,
                root.max,
                root.values.as_ref().map(|values| values.len()),
                &root.holes[..root.holes.len().min(16)],
                root.holes.len()
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ClauseCheckFailure {}

#[derive(Clone, Debug)]
pub enum ClauseCheckOutcome {
    Checked { assignments: u64, work: u64 },
    Uncovered(InferenceUncoveredReason),
    Rejected(ClauseCheckFailure),
}

struct Budget<'a> {
    work: usize,
    assignments: usize,
    max_work: usize,
    max_assignments: usize,
    stop: Option<&'a dyn Fn() -> bool>,
}

impl Budget<'_> {
    fn spend(&mut self, amount: usize) -> Result<(), InferenceUncoveredReason> {
        if self.stop.is_some_and(|stop| stop()) {
            return Err(InferenceUncoveredReason::Interrupted);
        }
        self.work = self.work.checked_add(amount).filter(|&v| v <= self.max_work).ok_or(InferenceUncoveredReason::WorkLimit)?;
        Ok(())
    }

    fn assignment(&mut self) -> Result<(), InferenceUncoveredReason> {
        if self.assignments >= self.max_assignments {
            return Err(InferenceUncoveredReason::AssignmentLimit);
        }
        self.spend(1)?;
        self.assignments += 1;
        Ok(())
    }
}

/// An interval, optional original support, and independently collected holes.
struct DomainSet<'a> {
    min: i64,
    max: i64,
    support: Option<&'a [i32]>,
    holes: BTreeSet<i32>,
}

impl<'a> DomainSet<'a> {
    fn new(domain: &'a InferenceDomain, budget: &mut Budget) -> Result<Self, InferenceUncoveredReason> {
        budget.spend(domain.holes.len())?;
        Ok(Self {
            min: i64::from(domain.min),
            max: i64::from(domain.max),
            support: domain.values.as_deref(),
            holes: domain.holes.iter().copied().collect(),
        })
    }

    fn restrict(&mut self, premise: Premise, negated: bool) {
        match (premise, negated) {
            (Premise::Ge { bound, .. }, false) => self.min = self.min.max(i64::from(bound)),
            (Premise::Ge { bound, .. }, true) => self.max = self.max.min(i64::from(bound) - 1),
            (Premise::Le { bound, .. }, false) => self.max = self.max.min(i64::from(bound)),
            (Premise::Le { bound, .. }, true) => self.min = self.min.max(i64::from(bound) + 1),
            (Premise::Eq { val, .. }, false) | (Premise::Ne { val, .. }, true) => {
                self.min = self.min.max(i64::from(val));
                self.max = self.max.min(i64::from(val));
            }
            (Premise::Ne { val, .. }, false) | (Premise::Eq { val, .. }, true) => {
                self.holes.insert(val);
            }
        }
    }

    fn in_bounds(&self, value: i32) -> bool {
        i64::from(value) >= self.min && i64::from(value) <= self.max && !self.holes.contains(&value)
    }

    fn contains(&self, value: i32, budget: &mut Budget) -> Result<bool, InferenceUncoveredReason> {
        budget.spend(1)?;
        if !self.in_bounds(value) {
            return Ok(false);
        }
        if let Some(support) = self.support {
            for &supported in support {
                budget.spend(1)?;
                if supported == value {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        Ok(true)
    }

    fn witness(&self, budget: &mut Budget) -> Result<Option<i32>, InferenceUncoveredReason> {
        self.witness_between(self.min, self.max, budget)
    }

    fn witness_between(&self, min: i64, max: i64, budget: &mut Budget) -> Result<Option<i32>, InferenceUncoveredReason> {
        budget.spend(1)?;
        let min = min.max(self.min);
        let max = max.min(self.max);
        if min > max {
            return Ok(None);
        }
        if let Some(support) = self.support {
            for &value in support {
                budget.spend(1)?;
                if i64::from(value) >= min && i64::from(value) <= max && !self.holes.contains(&value) {
                    return Ok(Some(value));
                }
            }
            return Ok(None);
        }
        let mut value = min;
        // Widened bounds make negation at both i32 endpoints exact.
        for &hole in self.holes.range((min as i32)..=(max as i32)) {
            budget.spend(1)?;
            if i64::from(hole) != value {
                break;
            }
            value += 1;
        }
        Ok((value <= max).then_some(value as i32))
    }

    fn enumerate(&self, cap: usize, budget: &mut Budget) -> Result<Vec<i32>, InferenceUncoveredReason> {
        if self.min > self.max {
            return Ok(Vec::new());
        }
        if let Some(support) = self.support {
            let mut values = BTreeSet::new();
            for &value in support {
                budget.spend(1)?;
                if self.in_bounds(value) {
                    values.insert(value);
                    if values.len() > cap {
                        return Err(InferenceUncoveredReason::AssignmentLimit);
                    }
                }
            }
            return Ok(values.into_iter().collect());
        }
        let mut size = (self.max - self.min + 1) as u64;
        for _ in self.holes.range((self.min as i32)..=(self.max as i32)) {
            budget.spend(1)?;
            size -= 1;
        }
        if size > cap as u64 {
            return Err(InferenceUncoveredReason::AssignmentLimit);
        }
        // Charge before allocating or iterating over a potentially wide range.
        budget.spend(size as usize)?;
        let mut values = Vec::with_capacity(size as usize);
        let mut next = self.min;
        for &hole in self.holes.range((self.min as i32)..=(self.max as i32)) {
            while next < i64::from(hole) {
                values.push(next as i32);
                next += 1;
            }
            next = i64::from(hole) + 1;
        }
        while next <= self.max {
            values.push(next as i32);
            next += 1;
        }
        Ok(values)
    }
}

fn premise_var(premise: Premise) -> VarId {
    match premise {
        Premise::Ge { var, .. } | Premise::Le { var, .. } | Premise::Eq { var, .. } | Premise::Ne { var, .. } => var,
    }
}

fn failure(inference: &Inference<'_>, detail: &str, counterexample: Option<Vec<(VarId, i32)>>) -> InferenceCheckOutcome {
    let constraint = match inference.constraint {
        InferenceConstraint::Clause { literals } => format!("signed integer clause, {} literals", literals.len()),
        InferenceConstraint::Linear { terms, relation, rhs } => format!("linear {:?}, {} terms, rhs {}", relation, terms.len(), rhs),
        InferenceConstraint::PositiveTable { arity, rows, .. } => format!("positive table, arity {}, {} rows", arity, rows),
        InferenceConstraint::NoOverlap { tasks } => format!("mandatory no-overlap, {} positive-duration tasks", tasks.len()),
        InferenceConstraint::Cumulative { tasks, capacity } => {
            format!("fixed cumulative, {} task occurrences, capacity {}", tasks.len(), capacity)
        }
    };
    InferenceCheckOutcome::Rejected(InferenceCheckFailure {
        propagator: inference.propagator,
        constraint,
        premises: inference.premises.to_vec(),
        conclusion: inference.conclusion,
        snapshot: inference.snapshot.to_vec(),
        counterexample,
        detail: detail.to_owned(),
    })
}

fn index_domains<'a>(
    domains: &'a [InferenceDomain],
    budget: &mut Budget,
) -> Result<BTreeMap<VarId, &'a InferenceDomain>, InferenceUncoveredReason> {
    budget.spend(domains.len())?;
    let mut indexed = BTreeMap::new();
    for domain in domains {
        // Bound any subsequently owned diagnostic before cloning it.
        budget.spend(domain.holes.len())?;
        if indexed.insert(domain.var, domain).is_some() {
            return Err(InferenceUncoveredReason::InvalidDomain { var: domain.var });
        }
    }
    Ok(indexed)
}

/// Independently prove one inference, or explicitly report incomplete coverage.
///
/// Work is bounded before allocations and interval enumeration. Configuration
/// values above the public hard ceilings are clamped. Arithmetic overflow is a
/// coverage gap, never an accepted implication. A rejected conclusion includes
/// an assignment satisfying the original constraint and premises while
/// falsifying the conclusion; a false premise includes its snapshot witness.
pub fn check_inference(inference: &Inference<'_>, config: InferenceCheckConfig) -> InferenceCheckOutcome {
    check_inference_with_stop(inference, config, None)
}

/// Interruptible counterpart of [`check_inference`]. Preparation and semantic
/// checks poll the supplied stop function while consuming the same bounded work
/// budget. An interrupted attempt is uncovered and cannot certify an inference.
pub fn check_inference_until(
    inference: &Inference<'_>,
    config: InferenceCheckConfig,
    should_stop: &dyn Fn() -> bool,
) -> InferenceCheckOutcome {
    check_inference_with_stop(inference, config, Some(should_stop))
}

fn check_inference_with_stop(
    inference: &Inference<'_>,
    config: InferenceCheckConfig,
    should_stop: Option<&dyn Fn() -> bool>,
) -> InferenceCheckOutcome {
    let mut budget = Budget {
        work: 0,
        assignments: 0,
        max_work: config.max_work.min(MAX_INFERENCE_WORK),
        max_assignments: config.max_assignments.min(MAX_INFERENCE_ASSIGNMENTS),
        stop: should_stop,
    };
    let result = check(inference, &mut budget);
    if should_stop.is_some_and(|stop| stop()) {
        return InferenceCheckOutcome::Uncovered(InferenceUncoveredReason::Interrupted);
    }
    match result {
        Ok(Some(failure)) => failure,
        Ok(None) => InferenceCheckOutcome::Checked { assignments: budget.assignments as u64, work: budget.work as u64 },
        Err(reason) => InferenceCheckOutcome::Uncovered(reason),
    }
}

/// Independently check conditional clause entailment using one representative
/// of each integer predicate truth cell. Numeric domain width never causes
/// interval enumeration. Sparse supports and initial holes remain exact.
pub fn check_clause_entailment(inference: &ClauseEntailment<'_>, config: InferenceCheckConfig) -> ClauseCheckOutcome {
    check_clause_entailment_until(inference, config, &|| false)
}

/// Interruptible counterpart. A capped or interrupted scan is uncovered, never
/// a proof. Preparation, sparse-support visits and truth-cell assignments all
/// consume the same work budget before allocations or potentially long scans.
pub fn check_clause_entailment_until(
    inference: &ClauseEntailment<'_>,
    config: InferenceCheckConfig,
    should_stop: &dyn Fn() -> bool,
) -> ClauseCheckOutcome {
    let mut budget = Budget {
        work: 0,
        assignments: 0,
        max_work: config.max_work.min(MAX_INFERENCE_WORK),
        max_assignments: config.max_assignments.min(MAX_INFERENCE_ASSIGNMENTS),
        stop: Some(should_stop),
    };
    match check_clauses(inference, &mut budget) {
        Ok(Some(counterexample)) => ClauseCheckOutcome::Rejected(ClauseCheckFailure {
            root_domains: inference.root_domains.to_vec(),
            sources: inference.sources.iter().map(|source| source.to_vec()).collect(),
            conclusion: inference.conclusion.to_vec(),
            counterexample,
            detail: "source clauses admit a counterexample to the target clause".to_owned(),
        }),
        Ok(None) => ClauseCheckOutcome::Checked { assignments: budget.assignments as u64, work: budget.work as u64 },
        Err(reason) => ClauseCheckOutcome::Uncovered(reason),
    }
}

fn literal_holds(literal: InferenceLiteral, value: i32) -> bool {
    let holds = match literal.predicate {
        Premise::Ge { bound, .. } => value >= bound,
        Premise::Le { bound, .. } => value <= bound,
        Premise::Eq { val, .. } => value == val,
        Premise::Ne { val, .. } => value != val,
    };
    holds == literal.positive
}

fn check_clauses(inference: &ClauseEntailment<'_>, budget: &mut Budget<'_>) -> Result<Option<Vec<(VarId, i32)>>, InferenceUncoveredReason> {
    if budget.max_assignments == 0 {
        return Err(InferenceUncoveredReason::AssignmentLimit);
    }
    budget.spend(1)?;
    let roots = index_domains(inference.root_domains, budget)?;
    budget.spend(inference.sources.len().saturating_add(inference.conclusion.len()))?;
    let mut vars = BTreeSet::new();
    for &source in inference.sources {
        budget.spend(source.len())?;
        for literal in source {
            budget.spend(1)?;
            vars.insert(premise_var(literal.predicate));
        }
    }
    for literal in inference.conclusion {
        budget.spend(1)?;
        vars.insert(premise_var(literal.predicate));
    }
    let variables: Vec<_> = vars.into_iter().collect();
    budget.spend(variables.len())?;
    let mut domains = Vec::with_capacity(variables.len());
    for &var in &variables {
        domains.push(DomainSet::new(roots.get(&var).ok_or(InferenceUncoveredReason::MissingRootDomain { var })?, budget)?);
    }
    // A target clause is false exactly when every target literal is false.
    // All restrictions on one CP variable act on its one shared domain.
    for &literal in inference.conclusion {
        budget.spend(1)?;
        let index = variables.binary_search(&premise_var(literal.predicate)).expect("target variable was collected");
        domains[index].restrict(literal.predicate, literal.positive);
    }
    let mut cuts = Vec::with_capacity(domains.len());
    for domain in &domains {
        if domain.witness(budget)?.is_none() {
            return Ok(None);
        }
        cuts.push(BTreeSet::from([domain.min, domain.max + 1]));
    }
    let mut sources = Vec::with_capacity(inference.sources.len());
    for &source in inference.sources {
        let mut indexed = Vec::with_capacity(source.len());
        for &literal in source {
            budget.spend(1)?;
            let index = variables.binary_search(&premise_var(literal.predicate)).expect("source variable was collected");
            indexed.push((index, literal));
            let (first, second) = match literal.predicate {
                Premise::Ge { bound, .. } => (i64::from(bound), None),
                Premise::Le { bound, .. } => (i64::from(bound) + 1, None),
                Premise::Eq { val, .. } | Premise::Ne { val, .. } => (i64::from(val), Some(i64::from(val) + 1)),
            };
            for cut in std::iter::once(first).chain(second) {
                if cut >= domains[index].min && cut <= domains[index].max + 1 {
                    cuts[index].insert(cut);
                }
            }
        }
        sources.push(indexed);
    }
    let mut values = Vec::with_capacity(domains.len());
    let mut product = 1usize;
    for (domain, cuts) in domains.iter().zip(cuts) {
        budget.spend(cuts.len())?;
        let cuts: Vec<_> = cuts.into_iter().collect();
        let mut candidates = Vec::new();
        for window in cuts.windows(2) {
            if let Some(value) = domain.witness_between(window[0], window[1] - 1, budget)? {
                candidates.push(value);
                if candidates.len() > budget.max_assignments / product {
                    return Err(InferenceUncoveredReason::AssignmentLimit);
                }
            }
        }
        if candidates.is_empty() {
            return Ok(None);
        }
        product = product.checked_mul(candidates.len()).ok_or(InferenceUncoveredReason::AssignmentLimit)?;
        values.push(candidates);
    }
    let mut digits = vec![0; values.len()];
    for _ in 0..product {
        budget.assignment()?;
        let mut satisfied = true;
        for source in &sources {
            budget.spend(1)?;
            let mut clause_true = false;
            for &(index, literal) in source {
                budget.spend(1)?;
                if literal_holds(literal, values[index][digits[index]]) {
                    clause_true = true;
                    break;
                }
            }
            if !clause_true {
                satisfied = false;
                break;
            }
        }
        if satisfied {
            return Ok(Some(variables.iter().enumerate().map(|(index, &var)| (var, values[index][digits[index]])).collect()));
        }
        for index in (0..digits.len()).rev() {
            budget.spend(1)?;
            digits[index] += 1;
            if digits[index] < values[index].len() {
                break;
            }
            digits[index] = 0;
        }
    }
    Ok(None)
}

fn check(inference: &Inference<'_>, budget: &mut Budget) -> Result<Option<InferenceCheckOutcome>, InferenceUncoveredReason> {
    if budget.max_assignments == 0 {
        return Err(InferenceUncoveredReason::AssignmentLimit);
    }
    budget.spend(1)?;
    let roots = index_domains(inference.root_domains, budget)?;
    let snapshot = index_domains(inference.snapshot, budget)?;
    budget.spend(inference.premises.len())?;
    let mut vars = BTreeSet::new();
    match inference.constraint {
        InferenceConstraint::Clause { literals } => {
            budget.spend(literals.len())?;
            vars.extend(literals.iter().map(|literal| premise_var(literal.predicate)));
        }
        InferenceConstraint::Linear { terms, .. } => {
            budget.spend(terms.len())?;
            vars.extend(terms.iter().map(|&(var, _)| var));
        }
        InferenceConstraint::PositiveTable { variables, tuples, arity, rows } => {
            if *arity != variables.len() || arity.checked_mul(*rows) != Some(tuples.len()) {
                return Err(InferenceUncoveredReason::InvalidConstraint("positive table shape mismatch"));
            }
            budget.spend(variables.len())?;
            vars.extend(variables.iter().copied());
        }
        InferenceConstraint::NoOverlap { tasks } => {
            budget.spend(tasks.len())?;
            for &(start, duration) in tasks {
                if duration <= 0 {
                    return Err(InferenceUncoveredReason::InvalidConstraint("no-overlap requires strictly positive durations"));
                }
                vars.insert(start);
            }
        }
        InferenceConstraint::Cumulative { tasks, capacity } => {
            if *capacity < 0 {
                return Err(InferenceUncoveredReason::InvalidConstraint("cumulative requires nonnegative capacity"));
            }
            budget.spend(tasks.len())?;
            for task in tasks {
                budget.spend(0)?;
                if task.duration < 0 || task.demand < 0 {
                    return Err(InferenceUncoveredReason::InvalidConstraint("cumulative requires nonnegative durations and demands"));
                }
                vars.insert(task.start);
                vars.extend(task.presence);
            }
        }
    }
    for &premise in inference.premises {
        let var = premise_var(premise);
        vars.insert(var);
        let current = snapshot.get(&var).ok_or(InferenceUncoveredReason::MissingSnapshotDomain { var })?;
        let mut actual = DomainSet::new(current, budget)?;
        if actual.witness(budget)?.is_none() {
            return Err(InferenceUncoveredReason::InvalidDomain { var });
        }
        actual.restrict(premise, true);
        if let Some(value) = actual.witness(budget)? {
            return Ok(Some(failure(inference, "cited premise is false in the pre-event snapshot", Some(vec![(var, value)]))));
        }
    }
    if let InferenceConclusion::Predicate(predicate) = inference.conclusion {
        vars.insert(premise_var(predicate));
    }
    let variables: Vec<_> = vars.into_iter().collect();
    budget.spend(variables.len())?;
    let mut domains = Vec::with_capacity(variables.len());
    for &var in &variables {
        let root = roots.get(&var).ok_or(InferenceUncoveredReason::MissingRootDomain { var })?;
        domains.push(DomainSet::new(root, budget)?);
    }
    for &premise in inference.premises {
        let index = variables.binary_search(&premise_var(premise)).expect("premise variable was collected");
        domains[index].restrict(premise, false);
    }
    if let InferenceConclusion::Predicate(predicate) = inference.conclusion {
        let index = variables.binary_search(&premise_var(predicate)).expect("conclusion variable was collected");
        domains[index].restrict(predicate, true);
    }
    let mut witnesses = Vec::with_capacity(variables.len());
    for (&var, domain) in variables.iter().zip(&domains) {
        let Some(value) = domain.witness(budget)? else {
            return Ok(None);
        };
        witnesses.push((var, value));
    }
    match inference.constraint {
        InferenceConstraint::Clause { literals } => {
            // The conjunction of premises and negated conclusion was already
            // applied. A disjunction has a model iff one disjunct has a model.
            // Respect shared CP variables by restricting the same DomainSet.
            for &literal in literals {
                budget.spend(1)?;
                let index = variables.binary_search(&premise_var(literal.predicate)).expect("clause variable was collected");
                budget.spend(domains[index].holes.len())?;
                let mut candidate = DomainSet {
                    min: domains[index].min,
                    max: domains[index].max,
                    support: domains[index].support,
                    holes: domains[index].holes.clone(),
                };
                candidate.restrict(literal.predicate, !literal.positive);
                if let Some(value) = candidate.witness(budget)? {
                    budget.assignment()?;
                    witnesses[index].1 = value;
                    return Ok(Some(failure(inference, "clause and premises admit a counterexample to the conclusion", Some(witnesses))));
                }
            }
            Ok(None)
        }
        InferenceConstraint::Linear { terms, relation, rhs } => {
            check_linear(inference, &variables, &domains, terms, *relation, *rhs, budget)
        }
        InferenceConstraint::NoOverlap { tasks } => check_no_overlap(inference, &variables, &domains, tasks, budget),
        InferenceConstraint::Cumulative { tasks, capacity } => check_cumulative(inference, &variables, &domains, tasks, *capacity, budget),
        InferenceConstraint::PositiveTable { variables: columns, tuples, arity, rows } => {
            let indices: Vec<_> = columns.iter().map(|var| variables.binary_search(var).expect("table variable was collected")).collect();
            let mut fixed = vec![None; variables.len()];
            for row in 0..*rows {
                budget.assignment()?;
                budget.spend(fixed.len())?;
                fixed.fill(None);
                let mut compatible = true;
                for (column, &index) in indices.iter().enumerate() {
                    budget.spend(1)?;
                    let value = tuples[row * arity + column];
                    if value == i32::MIN {
                        continue;
                    }
                    if fixed[index].is_some_and(|old| old != value) || !domains[index].contains(value, budget)? {
                        compatible = false;
                        break;
                    }
                    fixed[index] = Some(value);
                }
                if compatible {
                    for (witness, &value) in witnesses.iter_mut().zip(&fixed) {
                        if let Some(value) = value {
                            witness.1 = value;
                        }
                    }
                    return Ok(Some(failure(
                        inference,
                        "constraint and premises admit a counterexample to the conclusion",
                        Some(witnesses),
                    )));
                }
            }
            Ok(None)
        }
    }
}

fn check_linear(
    inference: &Inference<'_>,
    variables: &[VarId],
    domains: &[DomainSet<'_>],
    terms: &[(VarId, i128)],
    relation: InferenceRelation,
    rhs: i128,
    budget: &mut Budget,
) -> Result<Option<InferenceCheckOutcome>, InferenceUncoveredReason> {
    let mut values = Vec::with_capacity(domains.len());
    let mut product = 1usize;
    for domain in domains {
        let candidates = domain.enumerate(budget.max_assignments / product, budget)?;
        if candidates.is_empty() {
            return Ok(None);
        }
        product = product.checked_mul(candidates.len()).ok_or(InferenceUncoveredReason::AssignmentLimit)?;
        values.push(candidates);
    }
    budget.spend(terms.len())?;
    let indexed_terms: Vec<_> = terms
        .iter()
        .map(|&(var, coefficient)| (variables.binary_search(&var).expect("linear variable was collected"), coefficient))
        .collect();
    let mut digits = vec![0; values.len()];
    for _ in 0..product {
        budget.assignment()?;
        let mut sum = 0i128;
        for &(index, coefficient) in &indexed_terms {
            budget.spend(1)?;
            let term =
                coefficient.checked_mul(i128::from(values[index][digits[index]])).ok_or(InferenceUncoveredReason::ArithmeticOverflow)?;
            sum = sum.checked_add(term).ok_or(InferenceUncoveredReason::ArithmeticOverflow)?;
        }
        let satisfies = match relation {
            InferenceRelation::Le => sum <= rhs,
            InferenceRelation::Eq => sum == rhs,
            InferenceRelation::Ne => sum != rhs,
        };
        if satisfies {
            let counterexample = variables.iter().enumerate().map(|(index, &var)| (var, values[index][digits[index]])).collect();
            return Ok(Some(failure(inference, "constraint and premises admit a counterexample to the conclusion", Some(counterexample))));
        }
        for index in (0..digits.len()).rev() {
            budget.spend(1)?;
            digits[index] += 1;
            if digits[index] < values[index].len() {
                break;
            }
            digits[index] = 0;
        }
    }
    Ok(None)
}

/// Deliberately independent of scheduling propagation, energy bounds and
/// theta-lambda trees: test the mathematical pairwise disjunction on complete
/// assignments admitted by the frozen domains and explicit premises.
fn check_no_overlap(
    inference: &Inference<'_>,
    variables: &[VarId],
    domains: &[DomainSet<'_>],
    tasks: &[(VarId, i64)],
    budget: &mut Budget,
) -> Result<Option<InferenceCheckOutcome>, InferenceUncoveredReason> {
    let mut values = Vec::with_capacity(domains.len());
    let mut product = 1usize;
    for domain in domains {
        let candidates = domain.enumerate(budget.max_assignments / product, budget)?;
        if candidates.is_empty() {
            return Ok(None);
        }
        product = product.checked_mul(candidates.len()).ok_or(InferenceUncoveredReason::AssignmentLimit)?;
        values.push(candidates);
    }
    budget.spend(tasks.len())?;
    // Deduplicate assignments by variable, but preserve task multiplicity.
    // Two positive-duration tasks tied to one start always overlap.
    let indexed_tasks: Vec<_> = tasks
        .iter()
        .map(|&(start, duration)| (variables.binary_search(&start).expect("start variable was collected"), i128::from(duration)))
        .collect();
    let mut digits = vec![0; values.len()];
    for _ in 0..product {
        budget.assignment()?;
        let mut satisfies = true;
        'pairs: for (left, &(first, first_duration)) in indexed_tasks.iter().enumerate() {
            budget.spend(1)?;
            let first_start = i128::from(values[first][digits[first]]);
            let first_end = first_start.checked_add(first_duration).ok_or(InferenceUncoveredReason::ArithmeticOverflow)?;
            for &(second, second_duration) in &indexed_tasks[left + 1..] {
                budget.spend(1)?;
                let second_start = i128::from(values[second][digits[second]]);
                let second_end = second_start.checked_add(second_duration).ok_or(InferenceUncoveredReason::ArithmeticOverflow)?;
                if first_end > second_start && second_end > first_start {
                    satisfies = false;
                    break 'pairs;
                }
            }
        }
        if satisfies {
            let counterexample = variables.iter().enumerate().map(|(index, &var)| (var, values[index][digits[index]])).collect();
            return Ok(Some(failure(inference, "constraint and premises admit a counterexample to the conclusion", Some(counterexample))));
        }
        for index in (0..digits.len()).rev() {
            budget.spend(1)?;
            digits[index] += 1;
            if digits[index] < values[index].len() {
                break;
            }
            digits[index] = 0;
        }
    }
    Ok(None)
}

/// Check complete assignments without using the production event profile or
/// energetic reasoning. With nonnegative demands, every positive maximum load
/// occurs at an active task's start, so these instants suffice even when the
/// numeric horizon is large. Task occurrences remain distinct throughout.
fn check_cumulative(
    inference: &Inference<'_>,
    variables: &[VarId],
    domains: &[DomainSet<'_>],
    tasks: &[InferenceCumulativeTask],
    capacity: i64,
    budget: &mut Budget,
) -> Result<Option<InferenceCheckOutcome>, InferenceUncoveredReason> {
    let mut values = Vec::with_capacity(domains.len());
    let mut product = 1usize;
    for domain in domains {
        let candidates = domain.enumerate(budget.max_assignments / product, budget)?;
        if candidates.is_empty() {
            return Ok(None);
        }
        product = product.checked_mul(candidates.len()).ok_or(InferenceUncoveredReason::AssignmentLimit)?;
        values.push(candidates);
    }
    budget.spend(tasks.len())?;
    let mut indexed_tasks = Vec::with_capacity(tasks.len());
    for task in tasks {
        budget.spend(0)?;
        indexed_tasks.push((
            variables.binary_search(&task.start).expect("start variable was collected"),
            i128::from(task.duration),
            i128::from(task.demand),
            task.presence.map(|var| variables.binary_search(&var).expect("presence variable was collected")),
        ));
    }
    let mut digits = vec![0; values.len()];
    for _ in 0..product {
        budget.assignment()?;
        let assigned = |index: usize| values[index][digits[index]];
        let mut satisfies = true;
        'instants: for &(start, duration, demand, presence) in &indexed_tasks {
            budget.spend(1)?;
            if duration == 0 || demand == 0 || presence.is_some_and(|var| assigned(var) != 1) {
                continue;
            }
            let instant = i128::from(assigned(start));
            let mut load = 0i128;
            for &(other_start, other_duration, other_demand, other_presence) in &indexed_tasks {
                budget.spend(1)?;
                if other_duration == 0 || other_demand == 0 || other_presence.is_some_and(|var| assigned(var) != 1) {
                    continue;
                }
                let other_start = i128::from(assigned(other_start));
                let other_end = other_start.checked_add(other_duration).ok_or(InferenceUncoveredReason::ArithmeticOverflow)?;
                if other_start <= instant && instant < other_end {
                    load = load.checked_add(other_demand).ok_or(InferenceUncoveredReason::ArithmeticOverflow)?;
                    if load > i128::from(capacity) {
                        satisfies = false;
                        break 'instants;
                    }
                }
            }
        }
        if satisfies {
            let counterexample = variables.iter().enumerate().map(|(index, &var)| (var, assigned(index))).collect();
            return Ok(Some(failure(inference, "constraint and premises admit a counterexample to the conclusion", Some(counterexample))));
        }
        for index in (0..digits.len()).rev() {
            budget.spend(1)?;
            digits[index] += 1;
            if digits[index] < values[index].len() {
                break;
            }
            digits[index] = 0;
        }
    }
    Ok(None)
}
