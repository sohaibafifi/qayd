//! Deterministic warm starts for repeated-scenario flexible schedules.
//!
//! Compilation is deliberately semantic and conservative. A model is accepted
//! only when every non-fixed variable and every constraint belongs to a proved
//! structure made of exact-one mode choices, scenario-dependent durations, a
//! common precedence DAG, unary resources, and a positive affine makespan
//! objective. Models outside that contract simply fall through to exact search.

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::model::{AffineForm, BoolLiteral, Constraint, IntDomain, IntExpr, IntGlobalConstraint, IntVarRef, Model, Objective, Relation};

use super::schedule_ir::PrecedenceDag;

/// Bounds the exhaustive Boolean proof used to establish exact-one semantics.
const MAX_TRUTH_TABLE_PROOF_VARIABLES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConstructionLimits {
    pub(crate) configurations: u64,
    pub(crate) candidate_visits: u64,
    pub(crate) passes: u32,
}

impl Default for ConstructionLimits {
    fn default() -> Self {
        Self { configurations: 1_024, candidate_visits: 1_000_000, passes: 32 }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScenarioScheduleSolution {
    pub(crate) values: Vec<i32>,
    pub(crate) objective: i64,
    pub(crate) configurations: u64,
    pub(crate) candidate_visits: u64,
    pub(crate) improvements: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ScenarioSchedulePlan {
    operations: Vec<Operation>,
    scenarios: Vec<Scenario>,
    resources: usize,
    fixed: Vec<Option<i32>>,
    precedences: PrecedenceDag,
    objective_value: Option<IntVarRef>,
}

#[derive(Clone, Debug)]
struct Operation {
    modes: Vec<Mode>,
}

#[derive(Clone, Debug)]
struct Mode {
    selector: IntVarRef,
    resource: usize,
    duration_by_scenario: Vec<i32>,
}

#[derive(Clone, Debug)]
struct Scenario {
    starts: Vec<IntVarRef>,
    durations: Vec<IntVarRef>,
    completion: IntVarRef,
    weight: i64,
}

#[derive(Clone, Debug)]
struct ModeImplication {
    selector: IntVarRef,
    duration: IntVarRef,
    value: i32,
    constraint: usize,
}

#[derive(Clone, Debug)]
struct OperationDraft {
    selectors: Vec<IntVarRef>,
    duration_targets: Vec<IntVarRef>,
    values: BTreeMap<IntVarRef, BTreeMap<IntVarRef, i32>>,
}

#[derive(Clone, Copy, Debug)]
struct PrecedenceArc {
    start: IntVarRef,
    duration: IntVarRef,
    end: IntVarRef,
}

#[derive(Clone, Debug)]
struct ScenarioDraft {
    completion: IntVarRef,
    starts: Vec<IntVarRef>,
    durations: Vec<IntVarRef>,
    successors: Vec<Vec<usize>>,
    terminal: Vec<bool>,
    weight: i64,
}

#[derive(Clone, Debug)]
struct ResourceRecord {
    scenario: usize,
    selectors: Vec<IntVarRef>,
}

#[derive(Clone, Debug)]
struct EvaluatedSchedule {
    objective: i64,
    starts: Vec<Vec<i32>>,
    completions: Vec<i32>,
}

struct DispatchWork<'a> {
    stop: &'a AtomicBool,
    limit: u64,
    used: &'a mut u64,
}

#[derive(Clone, Copy)]
enum DispatchRule {
    EarliestCompletion,
    EarliestCriticalStart,
    LongestRemainingPath,
    ShortestProcessingTime,
}

const DISPATCH_RULES: [DispatchRule; 4] = [
    DispatchRule::EarliestCompletion,
    DispatchRule::EarliestCriticalStart,
    DispatchRule::LongestRemainingPath,
    DispatchRule::ShortestProcessingTime,
];

impl ScenarioSchedulePlan {
    pub(crate) fn compile(model: &Model, stop: &AtomicBool) -> Option<Self> {
        Compiler::new(model, stop)?.compile()
    }

    pub(crate) fn construct(&self, model: &Model, stop: &AtomicBool, limits: ConstructionLimits) -> Option<ScenarioScheduleSolution> {
        let visits_per_configuration = self
            .operations
            .len()
            .checked_mul(self.operations.len())?
            .checked_mul(self.scenarios.len())?
            .checked_mul(DISPATCH_RULES.len())?;
        if stop.load(Ordering::Acquire)
            || limits.configurations == 0
            || limits.candidate_visits < u64::try_from(visits_per_configuration).ok()?
        {
            return None;
        }

        let mut configurations = 0u64;
        let mut candidate_visits = 0u64;
        let mut improvements = 0u64;
        let initialization_attempts = limits.configurations.min(u64::try_from(self.operations.len()).ok()?.saturating_add(1));
        let mut initial_best: Option<(Vec<usize>, EvaluatedSchedule)> = None;
        for attempt in 0..initialization_attempts {
            if stop.load(Ordering::Acquire) || candidate_visits >= limits.candidate_visits {
                break;
            }
            let selected = self.initial_configuration(usize::try_from(attempt).ok()?)?;
            let schedule = self.evaluate_configuration(model, &selected, stop, limits.candidate_visits, &mut candidate_visits);
            configurations = configurations.saturating_add(1);
            let Some(schedule) = schedule else {
                if stop.load(Ordering::Acquire) {
                    return None;
                }
                continue;
            };
            if initial_best.as_ref().is_none_or(|(_, current)| schedule.objective < current.objective) {
                initial_best = Some((selected, schedule));
            }
        }
        let (mut selected, mut best) = initial_best?;

        for _ in 0..limits.passes {
            if stop.load(Ordering::Acquire) || configurations >= limits.configurations || candidate_visits >= limits.candidate_visits {
                break;
            }
            let mut move_choice: Option<(i64, usize, IntVarRef, usize, EvaluatedSchedule)> = None;
            'operations: for (operation_index, operation) in self.operations.iter().enumerate() {
                for (mode_index, mode) in operation.modes.iter().enumerate() {
                    if mode_index == selected[operation_index] {
                        continue;
                    }
                    if stop.load(Ordering::Acquire)
                        || configurations >= limits.configurations
                        || candidate_visits >= limits.candidate_visits
                    {
                        break 'operations;
                    }
                    let mut trial = selected.clone();
                    trial[operation_index] = mode_index;
                    let schedule = self.evaluate_configuration(model, &trial, stop, limits.candidate_visits, &mut candidate_visits);
                    configurations += 1;
                    let Some(schedule) = schedule else {
                        if stop.load(Ordering::Acquire) {
                            return None;
                        }
                        continue;
                    };
                    if schedule.objective >= best.objective {
                        continue;
                    }
                    let key = (schedule.objective, operation_index, mode.selector);
                    if move_choice.as_ref().is_none_or(|current| key < (current.0, current.1, current.2)) {
                        move_choice = Some((key.0, key.1, key.2, mode_index, schedule));
                    }
                }
            }
            let Some((_, operation, _, mode, schedule)) = move_choice else {
                break;
            };
            selected[operation] = mode;
            best = schedule;
            improvements += 1;
        }

        if stop.load(Ordering::Acquire) {
            return None;
        }
        let mut values = self.fixed.clone();
        for (operation_index, operation) in self.operations.iter().enumerate() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            for (mode_index, mode) in operation.modes.iter().enumerate() {
                assign(&mut values, mode.selector, i32::from(mode_index == selected[operation_index]))?;
            }
        }
        for (scenario_index, scenario) in self.scenarios.iter().enumerate() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            for (operation_index, &mode_index) in selected.iter().enumerate() {
                let mode = &self.operations[operation_index].modes[mode_index];
                assign(&mut values, scenario.durations[operation_index], mode.duration_by_scenario[scenario_index])?;
                assign(&mut values, scenario.starts[operation_index], best.starts[scenario_index][operation_index])?;
            }
            assign(&mut values, scenario.completion, best.completions[scenario_index])?;
        }
        if let Some(objective_value) = self.objective_value {
            assign(&mut values, objective_value, i32::try_from(best.objective).ok()?)?;
        }
        let values = values.into_iter().collect::<Option<Vec<_>>>()?;
        if values.iter().zip(model.int_vars()).any(|(&value, domain)| !domain.contains(i64::from(value))) {
            return None;
        }
        if model.objectives()[0].as_int_value(&values)? != best.objective {
            return None;
        }
        Some(ScenarioScheduleSolution { values, objective: best.objective, configurations, candidate_visits, improvements })
    }

    fn initial_configuration(&self, attempt: usize) -> Option<Vec<usize>> {
        if attempt == 0 {
            let mut selected = Vec::with_capacity(self.operations.len());
            for operation in &self.operations {
                let mut choice = None;
                for (mode_index, mode) in operation.modes.iter().enumerate() {
                    let key = (weighted_duration(mode, &self.scenarios)?, mode.selector, mode_index);
                    if choice.is_none_or(|current| key < current) {
                        choice = Some(key);
                    }
                }
                selected.push(choice?.2);
            }
            return Some(selected);
        }

        let count = self.operations.len();
        let rotation = (attempt - 1) % count;
        let mut selected = vec![0usize; count];
        let mut resource_load = vec![0i128; self.resources];
        for position in 0..count {
            let operation_index = self.precedences.topological()[(position + rotation) % count];
            let operation = &self.operations[operation_index];
            let mut choice: Option<(i128, i128, IntVarRef, usize)> = None;
            for (mode_index, mode) in operation.modes.iter().enumerate() {
                let duration = weighted_duration(mode, &self.scenarios)?;
                let projected = resource_load.get(mode.resource)?.checked_add(duration)?;
                let key = (projected, duration, mode.selector, mode_index);
                if choice.is_none_or(|current| key < current) {
                    choice = Some(key);
                }
            }
            let (_, _, _, mode_index) = choice?;
            selected[operation_index] = mode_index;
            let mode = &operation.modes[mode_index];
            resource_load[mode.resource] = resource_load[mode.resource].checked_add(weighted_duration(mode, &self.scenarios)?)?;
        }
        Some(selected)
    }

    fn evaluate_configuration(
        &self,
        model: &Model,
        selected: &[usize],
        stop: &AtomicBool,
        dispatch_limit: u64,
        candidate_visits: &mut u64,
    ) -> Option<EvaluatedSchedule> {
        let mut work = DispatchWork { stop, limit: dispatch_limit, used: candidate_visits };
        let mut starts = Vec::with_capacity(self.scenarios.len());
        let mut completions = Vec::with_capacity(self.scenarios.len());
        let mut objective = 0i64;
        for scenario_index in 0..self.scenarios.len() {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            let mut best: Option<(i32, Vec<i32>, usize)> = None;
            for (rule_index, rule) in DISPATCH_RULES.into_iter().enumerate() {
                let schedule = self.dispatch_scenario(model, scenario_index, selected, rule, &mut work);
                let Some(schedule) = schedule else {
                    if work.stop.load(Ordering::Acquire) || *work.used >= work.limit {
                        return None;
                    }
                    continue;
                };
                if best.as_ref().is_none_or(|current| (schedule.0, rule_index) < (current.0, current.2)) {
                    best = Some((schedule.0, schedule.1, rule_index));
                }
            }
            let (completion, scenario_starts, _) = best?;
            objective = objective.checked_add(self.scenarios[scenario_index].weight.checked_mul(i64::from(completion))?)?;
            starts.push(scenario_starts);
            completions.push(completion);
        }
        Some(EvaluatedSchedule { objective, starts, completions })
    }

    fn dispatch_scenario(
        &self,
        model: &Model,
        scenario_index: usize,
        selected: &[usize],
        rule: DispatchRule,
        work: &mut DispatchWork<'_>,
    ) -> Option<(i32, Vec<i32>)> {
        let scenario = &self.scenarios[scenario_index];
        let count = self.operations.len();
        let durations = (0..count)
            .map(|operation| self.operations[operation].modes[selected[operation]].duration_by_scenario[scenario_index])
            .collect::<Vec<_>>();
        let path_durations = durations.iter().copied().map(i64::from).collect::<Vec<_>>();
        let bottom = self.precedences.remaining_paths(&path_durations, work.stop)?;

        let mut remaining_predecessors = (0..count).map(|operation| self.precedences.predecessors(operation).len()).collect::<Vec<_>>();
        let mut predecessor_finish = vec![i64::MIN; count];
        let mut resource_available = vec![i64::MIN; self.resources];
        let mut scheduled = vec![false; count];
        let mut starts = vec![0i32; count];
        let mut finishes = vec![0i64; count];

        for step in 0..count {
            if work.stop.load(Ordering::Acquire) {
                return None;
            }
            let mut candidate = None;
            for operation in 0..count {
                if work.stop.load(Ordering::Acquire) || *work.used >= work.limit {
                    return None;
                }
                *work.used += 1;
                if scheduled[operation] || remaining_predecessors[operation] != 0 {
                    continue;
                }
                let mode = &self.operations[operation].modes[selected[operation]];
                let lower_bound = predecessor_finish[operation].max(resource_available[mode.resource]);
                let earliest = model.int_vars()[scenario.starts[operation].0].ceiling(lower_bound)?;
                let finish = earliest.checked_add(i64::from(durations[operation]))?;
                let key = dispatch_key(rule, earliest, finish, bottom[operation], durations[operation], operation);
                if candidate.as_ref().is_none_or(|current: &(DispatchKey, usize, i64, i64)| key < current.0) {
                    candidate = Some((key, operation, earliest, finish));
                }
            }
            let (_, operation, start, finish) = candidate?;
            let mode = &self.operations[operation].modes[selected[operation]];
            if finish > i64::from(i32::MAX) {
                return None;
            }
            starts[operation] = i32::try_from(start).ok()?;
            finishes[operation] = finish;
            scheduled[operation] = true;
            resource_available[mode.resource] = finish;
            for &successor in self.precedences.successors(operation) {
                if work.stop.load(Ordering::Acquire) {
                    return None;
                }
                remaining_predecessors[successor] = remaining_predecessors[successor].checked_sub(1)?;
                predecessor_finish[successor] = predecessor_finish[successor].max(finish);
            }
            debug_assert_eq!(scheduled.iter().filter(|&&done| done).count(), step + 1);
        }

        let terminal_finish = self.precedences.terminals().map(|operation| finishes[operation]).max()?;
        let completion = model.int_vars()[scenario.completion.0].ceiling(terminal_finish)?;
        Some((i32::try_from(completion).ok()?, starts))
    }

    #[cfg(test)]
    pub(crate) fn operation_count(&self) -> usize {
        self.operations.len()
    }

    #[cfg(test)]
    pub(crate) fn scenario_count(&self) -> usize {
        self.scenarios.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchKey(i64, i64, i64, usize);

impl Ord for DispatchKey {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        (self.0, self.1, self.2, self.3).cmp(&(other.0, other.1, other.2, other.3))
    }
}

impl PartialOrd for DispatchKey {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

fn dispatch_key(rule: DispatchRule, earliest: i64, finish: i64, bottom: i64, duration: i32, operation: usize) -> DispatchKey {
    match rule {
        DispatchRule::EarliestCompletion => DispatchKey(finish, -bottom, earliest, operation),
        DispatchRule::EarliestCriticalStart => DispatchKey(earliest, -bottom, finish, operation),
        DispatchRule::LongestRemainingPath => DispatchKey(-bottom, earliest, finish, operation),
        DispatchRule::ShortestProcessingTime => DispatchKey(i64::from(duration), earliest, -bottom, operation),
    }
}

fn weighted_duration(mode: &Mode, scenarios: &[Scenario]) -> Option<i128> {
    mode.duration_by_scenario
        .iter()
        .zip(scenarios)
        .try_fold(0i128, |total, (&duration, scenario)| total.checked_add(i128::from(duration).checked_mul(i128::from(scenario.weight))?))
}

fn assign(values: &mut [Option<i32>], variable: IntVarRef, value: i32) -> Option<()> {
    let slot = values.get_mut(variable.0)?;
    if slot.is_some_and(|current| current != value) {
        return None;
    }
    *slot = Some(value);
    Some(())
}

trait ObjectiveValue {
    fn as_int_value(&self, values: &[i32]) -> Option<i64>;
}

impl ObjectiveValue for Objective {
    fn as_int_value(&self, values: &[i32]) -> Option<i64> {
        let Objective::IntExpr { expr, .. } = self else {
            return None;
        };
        expr.evaluate(&|variable| values.get(variable.0).copied().map(i64::from))
    }
}

struct Compiler<'a> {
    model: &'a Model,
    stop: &'a AtomicBool,
    classified: Vec<bool>,
    objective_terms: BTreeMap<IntVarRef, i64>,
    objective_value: Option<IntVarRef>,
}

impl<'a> Compiler<'a> {
    fn new(model: &'a Model, stop: &'a AtomicBool) -> Option<Self> {
        if stop.load(Ordering::Acquire) || !model.sets().is_empty() || !model.lists().is_empty() || !model.intervals().is_empty() {
            return None;
        }
        let [Objective::IntExpr { minimize: true, expr }] = model.objectives() else {
            return None;
        };
        let mut objective_terms = positive_affine_terms(expr, stop)?;
        let mut objective_value = None;
        let mut classified = vec![false; model.constraints().len()];
        if objective_terms.len() == 1 && objective_terms.values().next() == Some(&1) {
            let candidate = *objective_terms.keys().next()?;
            let mut channel = None;
            for (index, constraint) in model.constraints().iter().enumerate() {
                if stop.load(Ordering::Acquire) {
                    return None;
                }
                let Some(terms) = materialized_positive_sum(constraint, candidate, stop) else {
                    continue;
                };
                if channel.replace((index, terms)).is_some() {
                    return None;
                }
            }
            if let Some((index, terms)) = channel {
                classified[index] = true;
                objective_terms = terms;
                objective_value = Some(candidate);
            }
        }
        Some(Self { model, stop, classified, objective_terms, objective_value })
    }

    fn compile(mut self) -> Option<ScenarioSchedulePlan> {
        let implications = self.mode_implications()?;
        let drafts = self.operation_drafts(&implications)?;
        self.prove_exact_one(&drafts)?;
        let arcs = self.precedence_arcs()?;
        let scenarios = self.scenarios(&drafts, &arcs)?;
        let (resources, modes) = self.resources(&drafts, &scenarios)?;

        if self.stop.load(Ordering::Acquire) || self.classified.iter().any(|classified| !classified) {
            return None;
        }
        let mut fixed = Vec::with_capacity(self.model.int_vars().len());
        let mut explained = HashSet::new();
        for operation in &drafts {
            explained.extend(operation.selectors.iter().copied());
            explained.extend(operation.duration_targets.iter().copied());
        }
        for scenario in &scenarios {
            explained.extend(scenario.starts.iter().copied());
            explained.insert(scenario.completion);
        }
        if let Some(objective_value) = self.objective_value {
            explained.insert(objective_value);
        }
        for (index, domain) in self.model.int_vars().iter().enumerate() {
            let value = fixed_domain_value(domain);
            if value.is_none() && !explained.contains(&IntVarRef(index)) {
                return None;
            }
            fixed.push(value);
        }

        let successors = scenarios.first()?.successors.clone();
        let terminal = scenarios.first()?.terminal.clone();
        if scenarios.iter().any(|scenario| scenario.successors != successors || scenario.terminal != terminal) {
            return None;
        }
        let precedences = PrecedenceDag::compile(successors, self.stop)?;
        let operations = modes.into_iter().map(|modes| Operation { modes }).collect();
        let scenarios = scenarios
            .into_iter()
            .map(|scenario| Scenario {
                starts: scenario.starts,
                durations: scenario.durations,
                completion: scenario.completion,
                weight: scenario.weight,
            })
            .collect();
        Some(ScenarioSchedulePlan { operations, scenarios, resources, fixed, precedences, objective_value: self.objective_value })
    }

    fn mode_implications(&mut self) -> Option<Vec<ModeImplication>> {
        let mut result = Vec::new();
        for (index, constraint) in self.model.constraints().iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let Constraint::Intension(expression) = constraint else {
                continue;
            };
            let Some((selector, duration, value)) = mode_implication(expression) else {
                continue;
            };
            if selector == duration
                || !is_binary_domain(self.model.int_vars().get(selector.0)?)
                || !self.model.int_vars().get(duration.0)?.contains(i64::from(value))
            {
                return None;
            }
            self.classified[index] = true;
            result.push(ModeImplication { selector, duration, value, constraint: index });
        }
        (!result.is_empty()).then_some(result)
    }

    fn operation_drafts(&self, implications: &[ModeImplication]) -> Option<Vec<OperationDraft>> {
        let mut by_selector: BTreeMap<IntVarRef, BTreeMap<IntVarRef, i32>> = BTreeMap::new();
        let mut seen_constraints = HashSet::new();
        for implication in implications {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if !seen_constraints.insert(implication.constraint)
                || by_selector.entry(implication.selector).or_default().insert(implication.duration, implication.value).is_some()
            {
                return None;
            }
        }
        let mut grouped: BTreeMap<Vec<IntVarRef>, Vec<IntVarRef>> = BTreeMap::new();
        for (&selector, targets) in &by_selector {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            grouped.entry(targets.keys().copied().collect()).or_default().push(selector);
        }
        let mut used_targets = HashSet::new();
        let mut drafts = Vec::new();
        for (targets, mut selectors) in grouped {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if targets.is_empty() || selectors.is_empty() || targets.iter().any(|target| !used_targets.insert(*target)) {
                return None;
            }
            selectors.sort_unstable();
            let values = selectors.iter().map(|selector| (*selector, by_selector.get(selector).cloned().unwrap())).collect();
            drafts.push(OperationDraft { selectors, duration_targets: targets, values });
        }
        drafts.sort_by_key(|draft| draft.duration_targets[0]);
        (!drafts.is_empty()).then_some(drafts)
    }

    fn prove_exact_one(&mut self, operations: &[OperationDraft]) -> Option<()> {
        let implication_constraints = self.classified.clone();
        for operation in operations {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let selector_set = operation.selectors.iter().copied().collect::<HashSet<_>>();
            let mut local = Vec::new();
            for (index, constraint) in self.model.constraints().iter().enumerate() {
                if self.stop.load(Ordering::Acquire) {
                    return None;
                }
                if implication_constraints[index] {
                    continue;
                }
                let scope = constraint.int_scope();
                if !scope.is_empty() && scope.iter().all(|variable| selector_set.contains(variable)) {
                    local.push(index);
                }
            }
            if local.is_empty() {
                return None;
            }
            let direct = local.iter().try_fold(true, |direct, &index| {
                Some(direct && directly_proves_exact_one(&self.model.constraints()[index], &selector_set, self.stop)?)
            })?;
            if !direct {
                if operation.selectors.len() > MAX_TRUTH_TABLE_PROOF_VARIABLES {
                    return None;
                }
                let assignments = 1usize.checked_shl(u32::try_from(operation.selectors.len()).ok()?)?;
                for bits in 0..assignments {
                    if self.stop.load(Ordering::Acquire) {
                        return None;
                    }
                    let values = operation
                        .selectors
                        .iter()
                        .enumerate()
                        .map(|(position, &variable)| (variable, i64::from((bits >> position) & 1 != 0)))
                        .collect::<HashMap<_, _>>();
                    let mut allowed = true;
                    for &index in &local {
                        if self.stop.load(Ordering::Acquire) {
                            return None;
                        }
                        allowed &= evaluate_local_constraint(&self.model.constraints()[index], &values)?;
                        if !allowed {
                            break;
                        }
                    }
                    if allowed != (bits.count_ones() == 1) {
                        return None;
                    }
                }
            }
            for index in local {
                self.classified[index] = true;
            }
        }
        Some(())
    }

    fn precedence_arcs(&mut self) -> Option<Vec<PrecedenceArc>> {
        let mut duration_targets = HashSet::new();
        for constraint in self.model.constraints() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if let Constraint::Intension(expression) = constraint {
                if let Some((_, duration, _)) = mode_implication(expression) {
                    duration_targets.insert(duration);
                }
            }
        }
        let mut arcs = Vec::new();
        for (index, constraint) in self.model.constraints().iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if self.classified[index] {
                continue;
            }
            let Constraint::Intension(expression) = constraint else {
                continue;
            };
            let Some((left, right, end)) = precedence(expression) else {
                continue;
            };
            let (start, duration) = if duration_targets.contains(&left) && !duration_targets.contains(&right) {
                (right, left)
            } else if duration_targets.contains(&right) && !duration_targets.contains(&left) {
                (left, right)
            } else {
                return None;
            };
            if start == end || duration == end {
                return None;
            }
            self.classified[index] = true;
            arcs.push(PrecedenceArc { start, duration, end });
        }
        (!arcs.is_empty()).then_some(arcs)
    }

    fn scenarios(&self, operations: &[OperationDraft], arcs: &[PrecedenceArc]) -> Option<Vec<ScenarioDraft>> {
        let mut duration_operation = HashMap::new();
        for (operation, draft) in operations.iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            for &duration in &draft.duration_targets {
                if duration_operation.insert(duration, operation).is_some() {
                    return None;
                }
            }
        }
        let mut duration_start = HashMap::new();
        let mut start_duration = HashMap::new();
        for arc in arcs {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if duration_start.insert(arc.duration, arc.start).is_some_and(|start| start != arc.start)
                || start_duration.insert(arc.start, arc.duration).is_some_and(|duration| duration != arc.duration)
            {
                return None;
            }
        }
        if duration_start.len() != duration_operation.len() {
            return None;
        }

        let completion_set = self.objective_terms.keys().copied().collect::<HashSet<_>>();
        let mut nodes = duration_operation.keys().copied().collect::<Vec<_>>();
        nodes.extend(completion_set.iter().copied());
        nodes.sort_unstable();
        nodes.dedup();
        let indexes = nodes.iter().enumerate().map(|(index, &variable)| (variable, index)).collect::<HashMap<_, _>>();
        let mut components = DisjointSet::new(nodes.len());
        for arc in arcs {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let successor = if let Some(&duration) = start_duration.get(&arc.end) {
                duration
            } else if completion_set.contains(&arc.end) {
                arc.end
            } else {
                return None;
            };
            components.union(*indexes.get(&arc.duration)?, *indexes.get(&successor)?);
        }

        let mut by_component: BTreeMap<usize, Vec<IntVarRef>> = BTreeMap::new();
        for &node in &nodes {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            by_component.entry(components.find(*indexes.get(&node)?)).or_default().push(node);
        }
        let mut scenarios = Vec::new();
        for component_nodes in by_component.values() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let completions = component_nodes.iter().filter(|node| completion_set.contains(node)).copied().collect::<Vec<_>>();
            let [completion] = completions.as_slice() else {
                return None;
            };
            let component_durations =
                component_nodes.iter().filter(|node| duration_operation.contains_key(node)).copied().collect::<Vec<_>>();
            if component_durations.len() != operations.len() {
                return None;
            }
            let mut durations = vec![None; operations.len()];
            let mut starts = vec![None; operations.len()];
            for duration in component_durations {
                if self.stop.load(Ordering::Acquire) {
                    return None;
                }
                let operation = *duration_operation.get(&duration)?;
                if durations[operation].replace(duration).is_some() {
                    return None;
                }
                starts[operation] = Some(*duration_start.get(&duration)?);
            }
            let durations = durations.into_iter().collect::<Option<Vec<_>>>()?;
            let starts = starts.into_iter().collect::<Option<Vec<_>>>()?;
            let mut successors = vec![Vec::new(); operations.len()];
            let mut terminal = vec![false; operations.len()];
            let mut seen_arcs = HashSet::new();
            for arc in arcs.iter().filter(|arc| durations.contains(&arc.duration)) {
                if self.stop.load(Ordering::Acquire) {
                    return None;
                }
                let from = *duration_operation.get(&arc.duration)?;
                if arc.end == *completion {
                    if !seen_arcs.insert((from, None)) {
                        return None;
                    }
                    terminal[from] = true;
                } else {
                    let successor_duration = *start_duration.get(&arc.end)?;
                    let to = *duration_operation.get(&successor_duration)?;
                    if from == to || !seen_arcs.insert((from, Some(to))) {
                        return None;
                    }
                    successors[from].push(to);
                }
            }
            for list in &mut successors {
                list.sort_unstable();
            }
            if PrecedenceDag::compile(successors.clone(), self.stop).is_none()
                || (0..operations.len()).any(|operation| terminal[operation] != successors[operation].is_empty())
            {
                return None;
            }
            scenarios.push(ScenarioDraft {
                completion: *completion,
                starts,
                durations,
                successors,
                terminal,
                weight: *self.objective_terms.get(completion)?,
            });
        }
        scenarios.sort_by_key(|scenario| scenario.completion);
        if scenarios.len() != self.objective_terms.len()
            || operations.iter().any(|operation| operation.duration_targets.len() != scenarios.len())
        {
            return None;
        }
        Some(scenarios)
    }

    fn resources(&mut self, operations: &[OperationDraft], scenarios: &[ScenarioDraft]) -> Option<(usize, Vec<Vec<Mode>>)> {
        let selector_operation = operations
            .iter()
            .enumerate()
            .flat_map(|(operation, draft)| draft.selectors.iter().map(move |&selector| (selector, operation)))
            .collect::<HashMap<_, _>>();
        let start_location = scenarios
            .iter()
            .enumerate()
            .flat_map(|(scenario, draft)| draft.starts.iter().enumerate().map(move |(operation, &start)| (start, (scenario, operation))))
            .collect::<HashMap<_, _>>();
        let mut records = Vec::new();
        for (index, constraint) in self.model.constraints().iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if self.classified[index] {
                continue;
            }
            let Constraint::IntegerGlobal(IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity }) = constraint else {
                continue;
            };
            if starts.is_empty()
                || starts.len() != durations.len()
                || starts.len() != demands.len()
                || fixed_domain_value(self.model.int_vars().get(capacity.0)?) != Some(1)
            {
                return None;
            }
            let mut scenario = None;
            let mut selectors = Vec::with_capacity(starts.len());
            let mut seen_selectors = HashSet::new();
            for ((&start, &duration), &selector) in starts.iter().zip(durations).zip(demands) {
                if self.stop.load(Ordering::Acquire) {
                    return None;
                }
                let (tuple_scenario, operation) = *start_location.get(&start)?;
                if scenario.replace(tuple_scenario).is_some_and(|current| current != tuple_scenario)
                    || selector_operation.get(&selector) != Some(&operation)
                    || !seen_selectors.insert(selector)
                    || !is_binary_domain(self.model.int_vars().get(selector.0)?)
                {
                    return None;
                }
                let fixed_duration = fixed_domain_value(self.model.int_vars().get(duration.0)?)?;
                if fixed_duration <= 0 {
                    return None;
                }
                let target = scenarios[tuple_scenario].durations[operation];
                if operations[operation].values.get(&selector)?.get(&target).copied() != Some(fixed_duration) {
                    return None;
                }
                selectors.push(selector);
            }
            selectors.sort_unstable();
            self.classified[index] = true;
            records.push(ResourceRecord { scenario: scenario?, selectors });
        }
        if records.is_empty() {
            return None;
        }

        let mut resource_keys =
            records.iter().map(|record| record.selectors.clone()).collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
        resource_keys.sort();
        let all_selectors = selector_operation.keys().copied().collect::<HashSet<_>>();
        let mut resource_of_selector = HashMap::new();
        for (resource, selectors) in resource_keys.iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            for &selector in selectors {
                if !all_selectors.contains(&selector) || resource_of_selector.insert(selector, resource).is_some() {
                    return None;
                }
            }
        }
        if resource_of_selector.len() != all_selectors.len() {
            return None;
        }
        let mut seen = HashSet::new();
        for record in &records {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let resource = resource_keys.binary_search(&record.selectors).ok()?;
            if !seen.insert((record.scenario, resource)) {
                return None;
            }
        }
        if seen.len() != scenarios.len().checked_mul(resource_keys.len())?
            || (0..scenarios.len()).any(|scenario| (0..resource_keys.len()).any(|resource| !seen.contains(&(scenario, resource))))
        {
            return None;
        }

        let mut result = Vec::with_capacity(operations.len());
        for (operation, draft) in operations.iter().enumerate() {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let mut modes = Vec::with_capacity(draft.selectors.len());
            for &selector in &draft.selectors {
                if self.stop.load(Ordering::Acquire) {
                    return None;
                }
                let duration_by_scenario = scenarios
                    .iter()
                    .map(|scenario| draft.values.get(&selector)?.get(&scenario.durations[operation]).copied())
                    .collect::<Option<Vec<_>>>()?;
                modes.push(Mode { selector, resource: *resource_of_selector.get(&selector)?, duration_by_scenario });
            }
            modes.sort_by_key(|mode| mode.selector);
            result.push(modes);
        }
        Some((resource_keys.len(), result))
    }
}

fn directly_proves_exact_one(constraint: &Constraint, selectors: &HashSet<IntVarRef>, stop: &AtomicBool) -> Option<bool> {
    if stop.load(Ordering::Acquire) {
        return None;
    }
    match constraint {
        Constraint::IntegerGlobal(IntGlobalConstraint::Count { variables, value: 1, relation: Relation::Eq, count: 1 }) => {
            let distinct = variables.iter().copied().collect::<HashSet<_>>();
            Some(variables.len() == selectors.len() && distinct.len() == variables.len() && distinct == *selectors)
        }
        Constraint::Linear { terms, relation: Relation::Eq, rhs } => {
            let Some(constant) = rhs.checked_neg() else {
                return Some(false);
            };
            Some(affine_proves_exact_one(terms.iter().copied(), constant, selectors))
        }
        Constraint::Intension(IntExpr::Eq(left, right)) => {
            let normalized = AffineForm::difference(left, right, stop);
            if stop.load(Ordering::Acquire) {
                return None;
            }
            Some(normalized.is_some_and(|form| {
                affine_proves_exact_one(
                    form.coefficients.into_iter().map(|(variable, coefficient)| (coefficient, variable)),
                    form.constant,
                    selectors,
                )
            }))
        }
        _ => Some(false),
    }
}

fn affine_proves_exact_one(terms: impl IntoIterator<Item = (i64, IntVarRef)>, constant: i64, selectors: &HashSet<IntVarRef>) -> bool {
    let mut normalized = BTreeMap::new();
    for (coefficient, variable) in terms {
        let Some(updated) = normalized.get(&variable).copied().unwrap_or(0i64).checked_add(coefficient) else {
            return false;
        };
        if updated == 0 {
            normalized.remove(&variable);
        } else {
            normalized.insert(variable, updated);
        }
    }
    if normalized.len() != selectors.len() || !normalized.keys().all(|variable| selectors.contains(variable)) {
        return false;
    }
    (constant == -1 && normalized.values().all(|&coefficient| coefficient == 1))
        || (constant == 1 && normalized.values().all(|&coefficient| coefficient == -1))
}

fn positive_affine_terms(expression: &IntExpr, stop: &AtomicBool) -> Option<BTreeMap<IntVarRef, i64>> {
    let form = AffineForm::expression(expression, stop)?;
    if form.constant != 0 || form.coefficients.is_empty() {
        return None;
    }
    for &coefficient in form.coefficients.values() {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        if coefficient <= 0 {
            return None;
        }
    }
    Some(form.coefficients)
}

/// Recognize exactly `objective = sum(positive coefficient * variable)`.
/// This reverses objective materialization without depending on the frontend
/// that introduced the auxiliary value.
fn materialized_positive_sum(constraint: &Constraint, objective: IntVarRef, stop: &AtomicBool) -> Option<BTreeMap<IntVarRef, i64>> {
    let Constraint::Linear { terms, relation: Relation::Eq, rhs: 0 } = constraint else {
        return None;
    };
    let mut normalized = BTreeMap::new();
    for &(coefficient, variable) in terms {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        let updated = normalized.get(&variable).copied().unwrap_or(0i64).checked_add(coefficient)?;
        if updated == 0 {
            normalized.remove(&variable);
        } else {
            normalized.insert(variable, updated);
        }
    }
    let objective_coefficient = normalized.remove(&objective)?;
    if objective_coefficient != -1 && objective_coefficient != 1 {
        return None;
    }
    if normalized.is_empty() {
        return None;
    }
    let orientation = objective_coefficient.checked_neg()?;
    let mut result = BTreeMap::new();
    for (variable, coefficient) in normalized {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        let coefficient = coefficient.checked_mul(orientation)?;
        if coefficient <= 0 || result.insert(variable, coefficient).is_some() {
            return None;
        }
    }
    Some(result)
}

fn mode_implication(expression: &IntExpr) -> Option<(IntVarRef, IntVarRef, i32)> {
    let IntExpr::Imp(guard, body) = expression else {
        return None;
    };
    let IntExpr::Variable(selector) = guard.as_ref() else {
        return None;
    };
    let IntExpr::Eq(left, right) = body.as_ref() else {
        return None;
    };
    let (duration, value) = variable_constant(left, right).or_else(|| variable_constant(right, left))?;
    Some((*selector, duration, value))
}

fn variable_constant(variable: &IntExpr, constant: &IntExpr) -> Option<(IntVarRef, i32)> {
    let (IntExpr::Variable(variable), IntExpr::Constant(value)) = (variable, constant) else {
        return None;
    };
    Some((*variable, i32::try_from(*value).ok()?))
}

fn precedence(expression: &IntExpr) -> Option<(IntVarRef, IntVarRef, IntVarRef)> {
    match expression {
        IntExpr::Le(sum, end) => precedence_parts(sum, end),
        IntExpr::Ge(end, sum) => precedence_parts(sum, end),
        _ => None,
    }
}

fn precedence_parts(sum: &IntExpr, end: &IntExpr) -> Option<(IntVarRef, IntVarRef, IntVarRef)> {
    let (IntExpr::Add(parts), IntExpr::Variable(end)) = (sum, end) else {
        return None;
    };
    let [IntExpr::Variable(left), IntExpr::Variable(right)] = parts.as_slice() else {
        return None;
    };
    Some((*left, *right, *end))
}

fn evaluate_local_constraint(constraint: &Constraint, values: &HashMap<IntVarRef, i64>) -> Option<bool> {
    match constraint {
        Constraint::Intension(expression) => Some(expression.evaluate(&|variable| values.get(&variable).copied())? != 0),
        Constraint::Linear { terms, relation, rhs } => {
            let lhs = terms
                .iter()
                .try_fold(0i64, |sum, (coefficient, variable)| sum.checked_add(coefficient.checked_mul(*values.get(variable)?)?))?;
            Some(compare(lhs, *relation, *rhs))
        }
        Constraint::Clause(literals) => Some(literals.iter().any(|literal| literal_value(literal, values).unwrap_or(false))),
        Constraint::IntegerGlobal(IntGlobalConstraint::Count { variables, value, relation, count }) => {
            let occurrences = variables.iter().filter(|variable| values.get(variable) == Some(&i64::from(*value))).count();
            Some(compare(i64::try_from(occurrences).ok()?, *relation, *count))
        }
        _ => None,
    }
}

fn literal_value(literal: &BoolLiteral, values: &HashMap<IntVarRef, i64>) -> Option<bool> {
    let value = *values.get(&literal.variable)? != 0;
    Some(if literal.positive { value } else { !value })
}

fn compare(left: i64, relation: Relation, right: i64) -> bool {
    match relation {
        Relation::Eq => left == right,
        Relation::Ne => left != right,
        Relation::Le => left <= right,
        Relation::Lt => left < right,
        Relation::Ge => left >= right,
        Relation::Gt => left > right,
    }
}

fn is_binary_domain(domain: &IntDomain) -> bool {
    match domain {
        IntDomain::Bool | IntDomain::Range { lo: 0, hi: 1 } => true,
        IntDomain::Set(values) => values.len() == 2 && values.contains(&0) && values.contains(&1),
        IntDomain::Range { .. } => false,
    }
}

fn fixed_domain_value(domain: &IntDomain) -> Option<i32> {
    match domain {
        IntDomain::Range { lo, hi } if lo == hi => Some(*lo),
        IntDomain::Set(values) if values.len() == 1 => values.first().copied(),
        _ => None,
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self { parent: (0..size).collect(), rank: vec![0; size] }
    }

    fn find(&mut self, item: usize) -> usize {
        if self.parent[item] != item {
            self.parent[item] = self.find(self.parent[item]);
        }
        self.parent[item]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return;
        }
        if self.rank[left] < self.rank[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        if self.rank[left] == self.rank[right] {
            self.rank[left] = self.rank[left].saturating_add(1);
        }
    }
}
