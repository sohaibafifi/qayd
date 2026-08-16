//! Conservative warm starts for integer disjunctive schedules.
//!
//! The compiler recognizes fixed-duration operations owned by semantic
//! `NoOverlap` constraints. It may also prove the permutation encoding of an
//! additional unary resource. Construction only fixes schedule decisions. The
//! canonical CP root remains responsible for completing and verifying every
//! other model decision before an incumbent can cross the engine boundary.

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::constraints::table::STAR;
use crate::mix64;
use crate::model::{AffineForm, Constraint, IntDomain, IntExpr, IntGlobalConstraint, IntVarRef, Model, Objective, Relation};

use super::schedule_ir::PrecedenceDag;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConstructionBudget {
    pub(crate) work: u64,
}

impl ConstructionBudget {
    pub(crate) const fn new(work: u64) -> Self {
        Self { work }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DisjunctiveScheduleCandidates {
    pub(crate) assignments: Vec<Vec<Option<i32>>>,
    pub(crate) work: u64,
    pub(crate) checkpoints: u64,
    #[cfg(test)]
    pub(crate) materializations: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct DisjunctiveSchedulePlan {
    operations: Vec<Operation>,
    resources: Vec<Resource>,
    implicit_resources: Vec<ImplicitResource>,
    precedences: PrecedenceDag,
    objective_ranking: ObjectiveRanking,
    variable_count: usize,
}

#[derive(Clone, Debug)]
struct Operation {
    start: IntVarRef,
    duration: i64,
    earliest: i64,
    latest: i64,
    resources: Vec<usize>,
}

#[derive(Clone, Debug)]
struct Resource {
    operations: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct ImplicitSlot {
    position: IntVarRef,
    selected_start: IntVarRef,
    selected_duration: IntVarRef,
}

#[derive(Clone, Debug)]
struct ImplicitResource {
    resource: usize,
    array: Vec<usize>,
    slots: Vec<ImplicitSlot>,
    chain_order: Vec<usize>,
}

#[derive(Clone, Debug)]
enum ObjectiveRanking {
    /// The objective is the least value in `target`'s domain that bounds every
    /// listed completion. No other constraint may mention `target`.
    CompletionEnvelope { target: IntVarRef, terms: Vec<ScheduleTerm> },
    /// The objective is an exact maximum, either directly in the objective or
    /// through one proved `Maximum` definition.
    ExactMaximum { target: Option<IntVarRef>, terms: Vec<ScheduleTerm> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScheduleTerm {
    OperationCompletion(usize),
    ImplicitResourceCompletion(usize),
    Constant(i64),
}

impl ObjectiveRanking {
    fn evaluate(&self, model: &Model, plan: &DisjunctiveSchedulePlan, schedule: &ConstructedSchedule) -> Option<i64> {
        let (target, terms) = match self {
            Self::CompletionEnvelope { target, terms } => (Some(*target), terms),
            Self::ExactMaximum { target, terms } => (*target, terms),
        };
        let mut values = terms.iter();
        let mut lower_bound = values.next()?.evaluate(plan, schedule)?;
        for term in values {
            lower_bound = lower_bound.max(term.evaluate(plan, schedule)?);
        }
        match self {
            Self::CompletionEnvelope { target, .. } => model.int_vars().get(target.0)?.ceiling(lower_bound),
            Self::ExactMaximum { .. } => {
                if target.is_some_and(|variable| !model.int_vars().get(variable.0).is_some_and(|domain| domain.contains(lower_bound))) {
                    return None;
                }
                Some(lower_bound)
            }
        }
    }
}

impl ScheduleTerm {
    fn evaluate(self, plan: &DisjunctiveSchedulePlan, schedule: &ConstructedSchedule) -> Option<i64> {
        match self {
            Self::OperationCompletion(operation) => schedule.starts.get(operation)?.checked_add(plan.operations.get(operation)?.duration),
            Self::ImplicitResourceCompletion(resource) => {
                let implicit = plan.implicit_resources.get(resource)?;
                let operation = *schedule.resource_sequences.get(implicit.resource)?.last()?;
                schedule.starts.get(operation)?.checked_add(plan.operations.get(operation)?.duration)
            }
            Self::Constant(value) => Some(value),
        }
    }
}

#[derive(Clone, Copy)]
enum DispatchRule {
    EarliestStart,
    EarliestCompletion,
    LongestRemainingPath,
    ShortestProcessingTime,
    LongestProcessingTime,
    MinimumSlack,
    MostResources,
    Seeded,
}

const DETERMINISTIC_DISPATCH_RULES: [DispatchRule; 7] = [
    DispatchRule::EarliestStart,
    DispatchRule::EarliestCompletion,
    DispatchRule::LongestRemainingPath,
    DispatchRule::ShortestProcessingTime,
    DispatchRule::LongestProcessingTime,
    DispatchRule::MinimumSlack,
    DispatchRule::MostResources,
];
const SEEDED_ATTEMPTS_PER_OPERATION: u64 = 32;
const MAX_SEEDED_ATTEMPTS: u64 = 512;
pub(crate) const MAX_CONSTRUCTION_WORK: u64 = 2_097_152;
pub(crate) const MAX_REPAIR_CANDIDATES: usize = 8;
const ATTEMPT_SEED_STRIDE: u64 = 0x9e37_79b9_7f4a_7c15;

impl DisjunctiveSchedulePlan {
    pub(crate) fn compile(model: &Model, stop: &AtomicBool) -> Option<Self> {
        Compiler::new(model, stop)?.compile()
    }

    pub(crate) fn construction_budget(&self) -> Option<ConstructionBudget> {
        let work_per_attempt = self.work_per_attempt()?;
        let attempts = self.desired_attempts()?.min(MAX_CONSTRUCTION_WORK.checked_div(work_per_attempt)?);
        if attempts == 0 {
            return None;
        }
        Some(ConstructionBudget::new(work_per_attempt.checked_mul(attempts)?))
    }

    pub(crate) fn construct(
        &self,
        model: &Model,
        seed: u64,
        stop: &AtomicBool,
        budget: ConstructionBudget,
    ) -> Option<DisjunctiveScheduleCandidates> {
        if stop.load(Ordering::Acquire) || budget.work == 0 {
            return None;
        }
        let bottom = self.remaining_paths(stop)?;
        let mut counter = WorkCounter::new(budget);
        let attempts = budget.work.checked_div(self.work_per_attempt()?)?;
        if attempts == 0 {
            return None;
        }
        let deterministic_attempts = u64::try_from(DETERMINISTIC_DISPATCH_RULES.len()).ok()?;
        let mut retained = Vec::new();

        for attempt in 0..attempts {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            let (rule, rule_seed) = if attempt < deterministic_attempts {
                let index = usize::try_from(attempt).ok()?;
                (*DETERMINISTIC_DISPATCH_RULES.get(index)?, 0)
            } else {
                let diversified = attempt.checked_sub(deterministic_attempts)?.checked_add(1)?;
                (DispatchRule::Seeded, mix64(seed ^ diversified.wrapping_mul(ATTEMPT_SEED_STRIDE)))
            };
            match self.dispatch(model, rule, rule_seed, &bottom, stop, &mut counter) {
                Ok(schedule) => {
                    if let Some(score) = self.schedule_score(model, &schedule) {
                        retain_ranked_candidate(&mut retained, RankedSchedule { score, schedule });
                    }
                }
                Err(BuildFailure::Exhausted) => break,
                Err(BuildFailure::Infeasible) => {}
                Err(BuildFailure::Interrupted) => return None,
            }
        }

        if stop.load(Ordering::Acquire) || retained.is_empty() {
            return None;
        }
        let materializations = retained.len();
        let mut assignments = Vec::with_capacity(materializations);
        for candidate in retained {
            if let Some(assignment) = self.materialize(model, &candidate.schedule) {
                if !assignments.contains(&assignment) {
                    assignments.push(assignment);
                }
            }
        }
        if assignments.is_empty() {
            return None;
        }
        Some(DisjunctiveScheduleCandidates {
            assignments,
            work: counter.used,
            checkpoints: counter.checkpoints,
            #[cfg(test)]
            materializations,
        })
    }

    fn work_per_attempt(&self) -> Option<u64> {
        let operations = u64::try_from(self.operations.len()).ok()?;
        operations.checked_mul(operations)
    }

    fn desired_attempts(&self) -> Option<u64> {
        let operations = u64::try_from(self.operations.len()).ok()?;
        let deterministic = u64::try_from(DETERMINISTIC_DISPATCH_RULES.len()).ok()?;
        let diversified = operations.checked_mul(SEEDED_ATTEMPTS_PER_OPERATION)?.min(MAX_SEEDED_ATTEMPTS);
        deterministic.checked_add(diversified)
    }

    fn remaining_paths(&self, stop: &AtomicBool) -> Option<Vec<i64>> {
        let durations = self.operations.iter().map(|operation| operation.duration).collect::<Vec<_>>();
        self.precedences.remaining_paths(&durations, stop)
    }

    fn dispatch(
        &self,
        model: &Model,
        rule: DispatchRule,
        seed: u64,
        bottom: &[i64],
        stop: &AtomicBool,
        counter: &mut WorkCounter,
    ) -> Result<ConstructedSchedule, BuildFailure> {
        let count = self.operations.len();
        let mut remaining_predecessors = (0..count).map(|operation| self.precedences.predecessors(operation).len()).collect::<Vec<_>>();
        let mut predecessor_finish = vec![i64::MIN; count];
        let mut resource_available = vec![i64::MIN; self.resources.len()];
        let mut starts = vec![0i64; count];
        let mut finishes = vec![0i64; count];
        let mut scheduled = vec![false; count];
        let mut resource_sequences = vec![Vec::new(); self.resources.len()];

        for _ in 0..count {
            let mut choice: Option<(DispatchKey, usize, i64, i64)> = None;
            for operation in 0..count {
                counter.charge(stop)?;
                if scheduled[operation] || remaining_predecessors[operation] != 0 {
                    continue;
                }
                let record = &self.operations[operation];
                let resource_finish = record.resources.iter().map(|&resource| resource_available[resource]).max().unwrap_or(i64::MIN);
                let lower_bound = record.earliest.max(predecessor_finish[operation]).max(resource_finish);
                let start = model
                    .int_vars()
                    .get(record.start.0)
                    .ok_or(BuildFailure::Infeasible)?
                    .ceiling(lower_bound)
                    .ok_or(BuildFailure::Infeasible)?;
                if start > record.latest {
                    return Err(BuildFailure::Infeasible);
                }
                let finish = start.checked_add(record.duration).ok_or(BuildFailure::Infeasible)?;
                let key = DispatchKey::new(
                    rule,
                    operation,
                    start,
                    finish,
                    record.duration,
                    record.latest,
                    bottom[operation],
                    record.resources.len(),
                    seed,
                );
                if choice.as_ref().is_none_or(|current| key < current.0) {
                    choice = Some((key, operation, start, finish));
                }
            }

            let (_, operation, start, finish) = choice.ok_or(BuildFailure::Infeasible)?;
            starts[operation] = start;
            finishes[operation] = finish;
            scheduled[operation] = true;
            for &resource in &self.operations[operation].resources {
                resource_available[resource] = finish;
                resource_sequences[resource].push(operation);
            }
            for &successor in self.precedences.successors(operation) {
                remaining_predecessors[successor] = remaining_predecessors[successor].checked_sub(1).ok_or(BuildFailure::Infeasible)?;
                predecessor_finish[successor] = predecessor_finish[successor].max(finish);
            }
        }

        Ok(ConstructedSchedule { starts, resource_sequences })
    }

    fn schedule_score(&self, model: &Model, schedule: &ConstructedSchedule) -> Option<ScheduleScore> {
        if schedule.starts.len() != self.operations.len() {
            return None;
        }
        let mut makespan = i64::MIN;
        let mut total_completion = 0i128;
        let mut total_start = 0i128;
        for (&start, operation) in schedule.starts.iter().zip(&self.operations) {
            let completion = start.checked_add(operation.duration)?;
            makespan = makespan.max(completion);
            total_completion = total_completion.checked_add(i128::from(completion))?;
            total_start = total_start.checked_add(i128::from(start))?;
        }
        let objective = self.objective_ranking.evaluate(model, self, schedule)?;
        Some(ScheduleScore { objective, makespan, total_completion, total_start })
    }

    fn materialize(&self, model: &Model, schedule: &ConstructedSchedule) -> Option<Vec<Option<i32>>> {
        let mut assignment = vec![None; self.variable_count];
        for (operation, record) in self.operations.iter().enumerate() {
            let start = i32::try_from(schedule.starts[operation]).ok()?;
            assign(model, &mut assignment, record.start, start)?;
        }
        for implicit in &self.implicit_resources {
            let sequence = schedule.resource_sequences.get(implicit.resource)?;
            if sequence.len() != implicit.array.len() || implicit.chain_order.len() != implicit.array.len() {
                return None;
            }
            let array_positions =
                implicit.array.iter().enumerate().map(|(position, &operation)| (operation, position)).collect::<HashMap<_, _>>();
            if array_positions.len() != implicit.array.len() {
                return None;
            }
            for (rank, &operation) in sequence.iter().enumerate() {
                let slot = *implicit.slots.get(implicit.chain_order[rank])?;
                let position = i32::try_from(*array_positions.get(&operation)?).ok()?;
                let start = i32::try_from(schedule.starts[operation]).ok()?;
                let duration = i32::try_from(self.operations[operation].duration).ok()?;
                assign(model, &mut assignment, slot.position, position)?;
                assign(model, &mut assignment, slot.selected_start, start)?;
                assign(model, &mut assignment, slot.selected_duration, duration)?;
            }
        }
        Some(assignment)
    }

    #[cfg(test)]
    pub(crate) fn operation_count(&self) -> usize {
        self.operations.len()
    }

    #[cfg(test)]
    pub(crate) fn resource_count(&self) -> usize {
        self.resources.len()
    }

    #[cfg(test)]
    pub(crate) fn implicit_resource_count(&self) -> usize {
        self.implicit_resources.len()
    }

    #[cfg(test)]
    pub(crate) fn precedence_count(&self) -> usize {
        self.precedences.all_successors().iter().map(Vec::len).sum()
    }
}

fn assign(model: &Model, assignment: &mut [Option<i32>], variable: IntVarRef, value: i32) -> Option<()> {
    if !model.int_vars().get(variable.0)?.contains(i64::from(value)) {
        return None;
    }
    let slot = assignment.get_mut(variable.0)?;
    if slot.is_some_and(|current| current != value) {
        return None;
    }
    *slot = Some(value);
    Some(())
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ConstructedSchedule {
    starts: Vec<i64>,
    resource_sequences: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RankedSchedule {
    score: ScheduleScore,
    schedule: ConstructedSchedule,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ScheduleScore {
    objective: i64,
    makespan: i64,
    total_completion: i128,
    total_start: i128,
}

fn retain_ranked_candidate(retained: &mut Vec<RankedSchedule>, candidate: RankedSchedule) {
    if retained.iter().any(|current| current.schedule == candidate.schedule) {
        return;
    }
    retained.push(candidate);
    retained.sort_unstable_by(|left, right| left.score.cmp(&right.score).then_with(|| left.schedule.cmp(&right.schedule)));
    retained.truncate(MAX_REPAIR_CANDIDATES);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchKey {
    primary: i128,
    secondary: i128,
    tertiary: i128,
    tie: u64,
    operation: usize,
}

impl DispatchKey {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rule: DispatchRule,
        operation: usize,
        start: i64,
        finish: i64,
        duration: i64,
        latest: i64,
        bottom: i64,
        resource_count: usize,
        seed: u64,
    ) -> Self {
        let seeded = mix64(seed ^ u64::try_from(operation).unwrap_or(u64::MAX));
        let (primary, secondary, tertiary, tie) = match rule {
            DispatchRule::EarliestStart => (i128::from(start), i128::from(finish), -i128::from(bottom), 0),
            DispatchRule::EarliestCompletion => (i128::from(finish), i128::from(start), -i128::from(bottom), 0),
            DispatchRule::LongestRemainingPath => (-i128::from(bottom), i128::from(start), i128::from(finish), 0),
            DispatchRule::ShortestProcessingTime => (i128::from(duration), i128::from(start), -i128::from(bottom), 0),
            DispatchRule::LongestProcessingTime => (-i128::from(duration), i128::from(start), -i128::from(bottom), 0),
            DispatchRule::MinimumSlack => (i128::from(latest) - i128::from(start), -i128::from(bottom), i128::from(finish), 0),
            DispatchRule::MostResources => {
                (-i128::try_from(resource_count).unwrap_or(i128::MAX), -i128::from(bottom), i128::from(start), 0)
            }
            DispatchRule::Seeded => (i128::from(seeded), i128::from(start), i128::from(finish), 0),
        };
        Self { primary, secondary, tertiary, tie, operation }
    }
}

impl Ord for DispatchKey {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        (self.primary, self.secondary, self.tertiary, self.tie, self.operation).cmp(&(
            other.primary,
            other.secondary,
            other.tertiary,
            other.tie,
            other.operation,
        ))
    }
}

impl PartialOrd for DispatchKey {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuildFailure {
    Exhausted,
    Infeasible,
    Interrupted,
}

struct WorkCounter {
    limit: u64,
    used: u64,
    checkpoints: u64,
}

impl WorkCounter {
    fn new(budget: ConstructionBudget) -> Self {
        Self { limit: budget.work, used: 0, checkpoints: 0 }
    }

    fn charge(&mut self, stop: &AtomicBool) -> Result<(), BuildFailure> {
        self.checkpoints = self.checkpoints.checked_add(1).ok_or(BuildFailure::Exhausted)?;
        if stop.load(Ordering::Acquire) {
            return Err(BuildFailure::Interrupted);
        }
        if self.used >= self.limit {
            return Err(BuildFailure::Exhausted);
        }
        self.used = self.used.checked_add(1).ok_or(BuildFailure::Exhausted)?;
        Ok(())
    }
}

struct Compiler<'a> {
    model: &'a Model,
    stop: &'a AtomicBool,
    operations: Vec<Operation>,
    operation_by_start: HashMap<IntVarRef, usize>,
    resources: Vec<Resource>,
    resource_keys: BTreeSet<Vec<usize>>,
    implicit_resources: Vec<ImplicitResource>,
    recognized_encoding_constraints: HashSet<usize>,
}

impl<'a> Compiler<'a> {
    fn new(model: &'a Model, stop: &'a AtomicBool) -> Option<Self> {
        if stop.load(Ordering::Acquire)
            || !model.sets().is_empty()
            || !model.lists().is_empty()
            || !model.intervals().is_empty()
            || !model.interval_modes().is_empty()
            || !matches!(model.objectives(), [Objective::IntExpr { minimize: true, .. }])
        {
            return None;
        }
        Some(Self {
            model,
            stop,
            operations: Vec::new(),
            operation_by_start: HashMap::new(),
            resources: Vec::new(),
            resource_keys: BTreeSet::new(),
            implicit_resources: Vec::new(),
            recognized_encoding_constraints: HashSet::new(),
        })
    }

    fn compile(mut self) -> Option<DisjunctiveSchedulePlan> {
        self.explicit_resources()?;
        self.reject_unsupported_schedule_globals()?;
        self.implicit_resources()?;
        let precedences = self.precedences()?;
        let objective_ranking = self.prove_objective_ranking(&precedences)?;
        if self.stop.load(Ordering::Acquire) {
            return None;
        }
        Some(DisjunctiveSchedulePlan {
            operations: self.operations,
            resources: self.resources,
            implicit_resources: self.implicit_resources,
            precedences,
            objective_ranking,
            variable_count: self.model.int_vars().len(),
        })
    }

    fn explicit_resources(&mut self) -> Option<()> {
        for constraint in self.model.constraints() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap { starts, durations }) = constraint else {
                continue;
            };
            if starts.is_empty() || starts.len() != durations.len() {
                return None;
            }
            let mut resource_operations = Vec::with_capacity(starts.len());
            let mut local_starts = HashSet::new();
            for (&start, &duration) in starts.iter().zip(durations) {
                if !local_starts.insert(start) || duration <= 0 {
                    return None;
                }
                let (earliest, latest) = domain_bounds(self.model.int_vars().get(start.0)?)?;
                earliest.checked_add(duration)?;
                latest.checked_add(duration)?;
                let operation = if let Some(&operation) = self.operation_by_start.get(&start) {
                    if self.operations[operation].duration != duration {
                        return None;
                    }
                    operation
                } else {
                    let operation = self.operations.len();
                    self.operation_by_start.insert(start, operation);
                    self.operations.push(Operation { start, duration, earliest, latest, resources: Vec::new() });
                    operation
                };
                resource_operations.push(operation);
            }
            resource_operations.sort_unstable();
            if !self.resource_keys.insert(resource_operations.clone()) {
                return None;
            }
            let resource = self.resources.len();
            for &operation in &resource_operations {
                self.operations[operation].resources.push(resource);
            }
            self.resources.push(Resource { operations: resource_operations });
        }
        if self.operations.is_empty() || self.resources.is_empty() {
            return None;
        }
        Some(())
    }

    fn reject_unsupported_schedule_globals(&self) -> Option<()> {
        for constraint in self.model.constraints() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if matches!(
                constraint,
                Constraint::IntegerGlobal(
                    IntGlobalConstraint::OptionalNoOverlap { .. }
                        | IntGlobalConstraint::AlternativeChannel { .. }
                        | IntGlobalConstraint::Cumulative { .. }
                        | IntGlobalConstraint::CumulativeVar { .. }
                )
            ) {
                return None;
            }
        }
        Some(())
    }

    fn implicit_resources(&mut self) -> Option<()> {
        let objective_definitions = objective_maximum_definition_constraints(self.model, self.stop)?;
        let all_different = self
            .model
            .constraints()
            .iter()
            .enumerate()
            .filter_map(|(index, constraint)| {
                matches!(constraint, Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { .. })).then_some(index)
            })
            .collect::<Vec<_>>();
        let has_elements_or_tables = self.model.constraints().iter().any(|constraint| {
            matches!(
                constraint,
                Constraint::IntegerGlobal(
                    IntGlobalConstraint::Element { .. } | IntGlobalConstraint::ElementConst { .. } | IntGlobalConstraint::Table { .. }
                )
            )
        });
        if all_different.is_empty() {
            return (!has_elements_or_tables).then_some(());
        }

        let mut used_positions = HashSet::new();
        let mut used_auxiliary = HashSet::new();
        for all_different_index in all_different {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let Constraint::IntegerGlobal(IntGlobalConstraint::AllDifferent { variables: positions, except }) =
                &self.model.constraints()[all_different_index]
            else {
                return None;
            };
            let count = positions.len();
            if count < 2 || !except.is_empty() || positions.iter().any(|position| !used_positions.insert(*position)) {
                return None;
            }
            for &position in positions {
                if !domain_is_zero_based_permutation(self.model.int_vars().get(position.0)?, count) {
                    return None;
                }
            }

            let mut array: Option<Vec<usize>> = None;
            let mut slots = Vec::with_capacity(count);
            let mut local_constraints = vec![all_different_index];
            for &position in positions {
                if self.stop.load(Ordering::Acquire) {
                    return None;
                }
                let element_matches = self
                    .model
                    .constraints()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, constraint)| match constraint {
                        Constraint::IntegerGlobal(IntGlobalConstraint::Element { index: candidate, .. }) if *candidate == position => {
                            Some(index)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let [element_index] = element_matches.as_slice() else {
                    return None;
                };
                let Constraint::IntegerGlobal(IntGlobalConstraint::Element { array: semantic_array, index: _, value: selected_start }) =
                    &self.model.constraints()[*element_index]
                else {
                    return None;
                };
                if semantic_array.len() != count {
                    return None;
                }
                let operation_array =
                    semantic_array.iter().map(|start| self.operation_by_start.get(start).copied()).collect::<Option<Vec<_>>>()?;
                if operation_array.iter().copied().collect::<HashSet<_>>().len() != count
                    || array.as_ref().is_some_and(|current| current != &operation_array)
                {
                    return None;
                }
                array.get_or_insert(operation_array);

                let table_matches = self
                    .model
                    .constraints()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, constraint)| match constraint {
                        Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, .. }) if variables.contains(&position) => {
                            Some(index)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let [table_index] = table_matches.as_slice() else {
                    return None;
                };
                let Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, tuples, positive }) =
                    &self.model.constraints()[*table_index]
                else {
                    return None;
                };
                let [table_position, selected_duration] = variables.as_slice() else {
                    return None;
                };
                if !*positive || *table_position != position || tuples.len() != count {
                    return None;
                }
                let operation_array = array.as_ref()?;
                let mut duration_by_position = BTreeMap::new();
                for tuple in tuples.iter() {
                    if self.stop.load(Ordering::Acquire) {
                        return None;
                    }
                    let [tuple_position, tuple_duration] = tuple.as_slice() else {
                        return None;
                    };
                    if *tuple_position == STAR || *tuple_duration == STAR {
                        return None;
                    }
                    let tuple_position = usize::try_from(*tuple_position).ok()?;
                    if tuple_position >= count
                        || duration_by_position.insert(tuple_position, *tuple_duration).is_some()
                        || i64::from(*tuple_duration) != self.operations[*operation_array.get(tuple_position)?].duration
                        || !self.model.int_vars().get(selected_duration.0)?.contains(i64::from(*tuple_duration))
                    {
                        return None;
                    }
                }
                if duration_by_position.len() != count {
                    return None;
                }
                if position == *selected_start
                    || position == *selected_duration
                    || *selected_start == *selected_duration
                    || self.operation_by_start.contains_key(selected_start)
                    || self.operation_by_start.contains_key(selected_duration)
                    || !used_auxiliary.insert(*selected_start)
                    || !used_auxiliary.insert(*selected_duration)
                {
                    return None;
                }
                slots.push(ImplicitSlot { position, selected_start: *selected_start, selected_duration: *selected_duration });
                local_constraints.extend([*element_index, *table_index]);
            }
            if positions.iter().any(|position| used_auxiliary.contains(position)) {
                return None;
            }

            let (chain_order, chain_constraints) = self.implicit_chain(&slots, &local_constraints, &objective_definitions)?;
            local_constraints.extend(chain_constraints);
            for constraint in local_constraints {
                if !self.recognized_encoding_constraints.insert(constraint) {
                    return None;
                }
            }

            let mut resource_operations = array.clone()?;
            resource_operations.sort_unstable();
            if !self.resource_keys.insert(resource_operations.clone()) {
                return None;
            }
            let resource = self.resources.len();
            for &operation in &resource_operations {
                self.operations[operation].resources.push(resource);
            }
            self.resources.push(Resource { operations: resource_operations });
            self.implicit_resources.push(ImplicitResource { resource, array: array?, slots, chain_order });
        }

        for (index, constraint) in self.model.constraints().iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if matches!(
                constraint,
                Constraint::IntegerGlobal(
                    IntGlobalConstraint::AllDifferent { .. }
                        | IntGlobalConstraint::Element { .. }
                        | IntGlobalConstraint::ElementConst { .. }
                        | IntGlobalConstraint::Table { .. }
                )
            ) && !self.recognized_encoding_constraints.contains(&index)
            {
                return None;
            }
        }
        Some(())
    }

    fn implicit_chain(
        &self,
        slots: &[ImplicitSlot],
        local_constraints: &[usize],
        objective_definitions: &HashSet<usize>,
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let mut ignored = local_constraints.iter().copied().collect::<HashSet<_>>();
        ignored.extend(objective_definitions);
        let starts = slots.iter().enumerate().map(|(index, slot)| (slot.selected_start, index)).collect::<HashMap<_, _>>();
        let durations = slots.iter().enumerate().map(|(index, slot)| (slot.selected_duration, index)).collect::<HashMap<_, _>>();
        if starts.len() != slots.len() || durations.len() != slots.len() {
            return None;
        }
        let auxiliary = starts.keys().chain(durations.keys()).copied().collect::<HashSet<_>>();
        let mut successors = vec![Vec::new(); slots.len()];
        let mut constraints = Vec::new();
        let mut arcs = HashSet::new();

        for (index, constraint) in self.model.constraints().iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if ignored.contains(&index) {
                continue;
            }
            let scope = constraint.int_scope();
            if !scope.iter().any(|variable| auxiliary.contains(variable)) {
                continue;
            }
            let Some(Some(relation)) = normalized_le_constraint(constraint) else {
                continue;
            };
            let Some((from, to)) = chain_arc(&relation, &starts, &durations) else {
                continue;
            };
            if from == to || !arcs.insert((from, to)) {
                return None;
            }
            successors[from].push(to);
            constraints.push(index);
        }

        if arcs.len() != slots.len().checked_sub(1)? {
            return None;
        }
        for list in &mut successors {
            list.sort_unstable();
        }
        Some((PrecedenceDag::unique_order(&successors, self.stop)?, constraints))
    }

    fn precedences(&self) -> Option<PrecedenceDag> {
        let mut successors = vec![Vec::new(); self.operations.len()];
        let mut arcs = HashSet::new();
        let operation_starts = self.operation_by_start.keys().copied().collect::<HashSet<_>>();

        for (index, constraint) in self.model.constraints().iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if self.recognized_encoding_constraints.contains(&index)
                || matches!(constraint, Constraint::IntegerGlobal(IntGlobalConstraint::NoOverlap { .. }))
            {
                continue;
            }
            let scope = constraint.int_scope();
            let schedule_scope = scope.iter().filter(|variable| operation_starts.contains(variable)).copied().collect::<HashSet<_>>();
            if schedule_scope.len() < 2 {
                continue;
            }
            let relation = normalized_le_constraint(constraint)?;
            let relation = relation?;
            let (from_start, to_start) = precedence_arc(&relation, &self.operation_by_start, &self.operations)?;
            let from = *self.operation_by_start.get(&from_start)?;
            let to = *self.operation_by_start.get(&to_start)?;
            if from == to || !arcs.insert((from, to)) {
                return None;
            }
            successors[from].push(to);
        }
        PrecedenceDag::compile(successors, self.stop)
    }

    /// Compile an exact objective evaluator for constructed schedules and prove
    /// that every operation contributes directly or through fixed precedence.
    /// Surrogate-only rankings are deliberately declined.
    fn prove_objective_ranking(&self, precedences: &PrecedenceDag) -> Option<ObjectiveRanking> {
        let [Objective::IntExpr { minimize: true, expr }] = self.model.objectives() else {
            return None;
        };
        let ranking = match expr {
            IntExpr::Max(values) => {
                let terms = values.iter().map(|value| self.schedule_term(&affine_value(value, self.stop)?)).collect::<Option<Vec<_>>>()?;
                ObjectiveRanking::ExactMaximum { target: None, terms }
            }
            IntExpr::Variable(target) => self.materialized_maximum(*target)?.or_else(|| self.completion_envelope(*target))?,
            _ => return None,
        };
        self.prove_ranking_coverage(ranking.terms(), precedences)?;
        Some(ranking)
    }

    fn materialized_maximum(&self, target: IntVarRef) -> Option<Option<ObjectiveRanking>> {
        let matches = self
            .model
            .constraints()
            .iter()
            .enumerate()
            .filter_map(|(index, constraint)| match constraint {
                Constraint::IntegerGlobal(IntGlobalConstraint::Maximum { target: candidate, variables }) if *candidate == target => {
                    Some((index, variables))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let [] = matches.as_slice() else {
            let [(definition, variables)] = matches.as_slice() else {
                return None;
            };
            if variables.is_empty()
                || self
                    .model
                    .constraints()
                    .iter()
                    .enumerate()
                    .any(|(index, constraint)| index != *definition && constraint.int_scope().contains(&target))
            {
                return None;
            }
            let terms = variables
                .iter()
                .map(|&variable| self.schedule_term(&expand_affine_variable(self.model, variable, self.stop)?))
                .collect::<Option<Vec<_>>>()?;
            return Some(Some(ObjectiveRanking::ExactMaximum { target: Some(target), terms }));
        };
        Some(None)
    }

    fn completion_envelope(&self, target: IntVarRef) -> Option<ObjectiveRanking> {
        if self.operation_by_start.contains_key(&target) {
            return None;
        }
        let mut terms = Vec::new();
        for constraint in self.model.constraints() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if !constraint.int_scope().contains(&target) {
                continue;
            }
            let mut relation = normalized_le_constraint(constraint)??;
            if relation.coefficients.remove(&target) != Some(-1) {
                return None;
            }
            terms.push(self.schedule_term(&AffineValue { coefficients: relation.coefficients, constant: relation.constant })?);
        }
        (!terms.is_empty()).then_some(ObjectiveRanking::CompletionEnvelope { target, terms })
    }

    fn schedule_term(&self, value: &AffineValue) -> Option<ScheduleTerm> {
        if value.coefficients.is_empty() {
            return Some(ScheduleTerm::Constant(value.constant));
        }
        for (operation, record) in self.operations.iter().enumerate() {
            if value == &AffineValue::single(record.start, record.duration) {
                return Some(ScheduleTerm::OperationCompletion(operation));
            }
        }
        for (resource, implicit) in self.implicit_resources.iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let terminal = *implicit.chain_order.last()?;
            let slot = *implicit.slots.get(terminal)?;
            if value == &AffineValue::pair(slot.selected_start, slot.selected_duration) {
                return Some(ScheduleTerm::ImplicitResourceCompletion(resource));
            }
        }
        None
    }

    fn prove_ranking_coverage(&self, terms: &[ScheduleTerm], precedences: &PrecedenceDag) -> Option<()> {
        let mut covered = vec![false; self.operations.len()];
        for &term in terms {
            match term {
                ScheduleTerm::OperationCompletion(operation) => *covered.get_mut(operation)? = true,
                ScheduleTerm::ImplicitResourceCompletion(resource) => {
                    for &operation in &self.implicit_resources.get(resource)?.array {
                        *covered.get_mut(operation)? = true;
                    }
                }
                ScheduleTerm::Constant(_) => {}
            }
        }
        for &operation in precedences.topological().iter().rev() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            covered[operation] |= precedences.successors(operation).iter().any(|&successor| covered[successor]);
        }
        covered.into_iter().all(|operation| operation).then_some(())
    }
}

impl ObjectiveRanking {
    fn terms(&self) -> &[ScheduleTerm] {
        match self {
            Self::CompletionEnvelope { terms, .. } | Self::ExactMaximum { terms, .. } => terms,
        }
    }
}

type AffineLe = AffineForm;

#[derive(Clone, Debug, PartialEq, Eq)]
struct AffineValue {
    coefficients: BTreeMap<IntVarRef, i64>,
    constant: i64,
}

impl AffineValue {
    fn single(variable: IntVarRef, constant: i64) -> Self {
        Self { coefficients: BTreeMap::from([(variable, 1)]), constant }
    }

    fn pair(left: IntVarRef, right: IntVarRef) -> Self {
        let mut coefficients = BTreeMap::new();
        *coefficients.entry(left).or_insert(0) += 1;
        *coefficients.entry(right).or_insert(0) += 1;
        Self { coefficients, constant: 0 }
    }
}

fn objective_maximum_definition_constraints(model: &Model, stop: &AtomicBool) -> Option<HashSet<usize>> {
    let [Objective::IntExpr { minimize: true, expr: IntExpr::Variable(objective) }] = model.objectives() else {
        return Some(HashSet::new());
    };
    let mut maximum = None;
    for constraint in model.constraints() {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        let Constraint::IntegerGlobal(IntGlobalConstraint::Maximum { target, variables }) = constraint else {
            continue;
        };
        if target == objective && maximum.replace(variables).is_some() {
            return None;
        }
    }
    let Some(variables) = maximum else {
        return Some(HashSet::new());
    };
    let mut definitions = HashSet::new();
    for &variable in variables {
        let mut definition = None;
        for (index, constraint) in model.constraints().iter().enumerate() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            let Some(mut equality) = normalized_eq_constraint(constraint)? else {
                continue;
            };
            let Some(coefficient) = equality.coefficients.remove(&variable) else {
                continue;
            };
            if equality.coefficients.is_empty() || !matches!(coefficient, -1 | 1) {
                continue;
            }
            if definition.replace(index).is_some() {
                return None;
            }
        }
        if let Some(index) = definition {
            definitions.insert(index);
        }
    }
    Some(definitions)
}

fn affine_value(expression: &IntExpr, stop: &AtomicBool) -> Option<AffineValue> {
    let affine = AffineForm::expression(expression, stop)?;
    Some(AffineValue { coefficients: affine.coefficients, constant: affine.constant })
}

fn expand_affine_variable(model: &Model, target: IntVarRef, stop: &AtomicBool) -> Option<AffineValue> {
    let mut definition = None;
    for constraint in model.constraints() {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        let Some(mut equality) = normalized_eq_constraint(constraint)? else {
            continue;
        };
        let Some(target_coefficient) = equality.coefficients.remove(&target) else {
            continue;
        };
        if equality.coefficients.is_empty() || !matches!(target_coefficient, -1 | 1) {
            continue;
        }
        let orientation = target_coefficient.checked_neg()?;
        let mut coefficients = BTreeMap::new();
        for (variable, coefficient) in equality.coefficients {
            coefficients.insert(variable, coefficient.checked_mul(orientation)?);
        }
        let value = AffineValue { coefficients, constant: equality.constant.checked_mul(orientation)? };
        if definition.replace(value).is_some() {
            return None;
        }
    }
    definition.or_else(|| Some(AffineValue::single(target, 0)))
}

fn normalized_eq_constraint(constraint: &Constraint) -> Option<Option<AffineLe>> {
    match constraint {
        Constraint::Intension(IntExpr::Eq(left, right)) => Some(AffineForm::difference(left, right, &AtomicBool::new(false))),
        Constraint::Linear { terms, relation: Relation::Eq, rhs } => Some(AffineForm::linear(terms, *rhs)),
        _ => Some(None),
    }
}

fn normalized_le_constraint(constraint: &Constraint) -> Option<Option<AffineLe>> {
    match constraint {
        Constraint::Intension(expression) => normalized_le_expression(expression),
        Constraint::Linear { terms, relation: Relation::Le, rhs } => Some(AffineForm::linear(terms, *rhs)),
        Constraint::Linear { terms, relation: Relation::Ge, rhs } => Some(AffineForm::linear(terms, *rhs)?.scale(-1)),
        Constraint::Linear { relation: Relation::Eq | Relation::Ne | Relation::Lt | Relation::Gt, .. } => Some(None),
        _ => Some(None),
    }
}

fn normalized_le_expression(expression: &IntExpr) -> Option<Option<AffineLe>> {
    let (left, right, reverse) = match expression {
        IntExpr::Le(left, right) => (left.as_ref(), right.as_ref(), false),
        IntExpr::Ge(left, right) => (left.as_ref(), right.as_ref(), true),
        IntExpr::Eq(_, _) | IntExpr::Ne(_, _) | IntExpr::Lt(_, _) | IntExpr::Gt(_, _) => return Some(None),
        _ => return Some(None),
    };
    let affine = if reverse {
        AffineForm::difference(right, left, &AtomicBool::new(false))?
    } else {
        AffineForm::difference(left, right, &AtomicBool::new(false))?
    };
    Some(Some(affine))
}

fn precedence_arc(
    relation: &AffineLe,
    operation_by_start: &HashMap<IntVarRef, usize>,
    operations: &[Operation],
) -> Option<(IntVarRef, IntVarRef)> {
    if relation.coefficients.len() != 2 {
        return None;
    }
    let from = relation.coefficients.iter().find_map(|(&variable, &coefficient)| (coefficient == 1).then_some(variable))?;
    let to = relation.coefficients.iter().find_map(|(&variable, &coefficient)| (coefficient == -1).then_some(variable))?;
    if relation.coefficients.get(&from) != Some(&1)
        || relation.coefficients.get(&to) != Some(&-1)
        || from == to
        || !operation_by_start.contains_key(&from)
        || !operation_by_start.contains_key(&to)
        || relation.constant != operations[*operation_by_start.get(&from)?].duration
    {
        return None;
    }
    Some((from, to))
}

fn chain_arc(relation: &AffineLe, starts: &HashMap<IntVarRef, usize>, durations: &HashMap<IntVarRef, usize>) -> Option<(usize, usize)> {
    if relation.constant != 0 || relation.coefficients.len() != 3 {
        return None;
    }
    let end = relation.coefficients.iter().find_map(|(&variable, &coefficient)| (coefficient == -1).then_some(variable))?;
    let to = *starts.get(&end)?;
    let positive =
        relation.coefficients.iter().filter_map(|(&variable, &coefficient)| (coefficient == 1).then_some(variable)).collect::<Vec<_>>();
    let [left, right] = positive.as_slice() else {
        return None;
    };
    let from = match (starts.get(left), durations.get(right), starts.get(right), durations.get(left)) {
        (Some(&start), Some(&duration), _, _) if start == duration => start,
        (_, _, Some(&start), Some(&duration)) if start == duration => start,
        _ => return None,
    };
    Some((from, to))
}

fn domain_bounds(domain: &IntDomain) -> Option<(i64, i64)> {
    match domain {
        IntDomain::Bool => Some((0, 1)),
        IntDomain::Range { lo, hi } if lo <= hi => Some((i64::from(*lo), i64::from(*hi))),
        IntDomain::Range { .. } => None,
        IntDomain::Set(values) => {
            let ordered = values.iter().copied().collect::<BTreeSet<_>>();
            let first = *ordered.first()?;
            let last = *ordered.last()?;
            (ordered.len() == values.len()).then_some((i64::from(first), i64::from(last)))
        }
    }
}

fn domain_is_zero_based_permutation(domain: &IntDomain, count: usize) -> bool {
    let Ok(last) = i32::try_from(count.saturating_sub(1)) else {
        return false;
    };
    match domain {
        IntDomain::Bool => count == 2,
        IntDomain::Range { lo: 0, hi } => count > 0 && *hi == last,
        IntDomain::Set(values) => {
            count > 0 && values.len() == count && values.iter().copied().collect::<BTreeSet<_>>() == (0..=last).collect::<BTreeSet<_>>()
        }
        IntDomain::Range { .. } => false,
    }
}
