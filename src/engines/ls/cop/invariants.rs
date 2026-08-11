//! Incremental invariant graph for integer local search.
//!
//! The graph is immutable after compilation. [`InvariantState`] owns one worker
//! trajectory and evaluates speculative assignments transactionally: update the
//! changed decision, propagate only reachable functional nodes, refresh only
//! dependent violations and the objective, then either commit or roll back the
//! journal.

use crate::ids::VarId;
use crate::problem::Objective;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::AtomicBool;

use super::{
    combine_violation, constraint_vars, functional_inputs, functional_target, weighted_sum, Functional, LocalConstraint, LocalModel, Score,
};

#[derive(Clone)]
pub(super) struct InvariantGraph {
    functional_inputs: Vec<Vec<VarId>>,
    /// Direct graph arcs. Transitive closures are deliberately not stored: a
    /// long functional chain must consume linear, not quadratic, memory.
    functionals_by_variable: Vec<Vec<usize>>,
    constraints_by_variable: Vec<Vec<usize>>,
    objective_by_variable: Vec<bool>,
    objective_update: ObjectiveUpdate,
}

#[derive(Clone)]
enum ObjectiveUpdate {
    Delta(Vec<i64>),
    Recompute,
}

impl InvariantGraph {
    pub(super) fn compile(
        variables: usize,
        constraints: &[LocalConstraint],
        functionals: &[Functional],
        objective: &Objective,
    ) -> Option<Self> {
        let mut producer = vec![None; variables];
        let mut inputs = Vec::with_capacity(functionals.len());
        for (index, functional) in functionals.iter().enumerate() {
            let target = functional_target(functional).index();
            let slot = producer.get_mut(target)?;
            if slot.replace(index).is_some() {
                return None;
            }
            let mut node_inputs = Vec::new();
            functional_inputs(functional, &mut node_inputs);
            if node_inputs.iter().any(|input| input.index() >= variables) {
                return None;
            }
            node_inputs.sort_unstable();
            node_inputs.dedup();
            inputs.push(node_inputs);
        }
        for (index, node_inputs) in inputs.iter().enumerate() {
            if node_inputs.iter().any(|input| producer[input.index()].is_some_and(|source| source >= index)) {
                return None;
            }
        }

        let mut functionals_by_variable = vec![Vec::new(); variables];
        for (index, node_inputs) in inputs.iter().enumerate() {
            for input in node_inputs {
                functionals_by_variable[input.index()].push(index);
            }
        }
        normalize_incidence(&mut functionals_by_variable);

        let mut constraints_by_variable = vec![Vec::new(); variables];
        let mut scope = Vec::new();
        for (index, constraint) in constraints.iter().enumerate() {
            scope.clear();
            constraint_vars(constraint, &mut scope);
            for variable in &scope {
                constraints_by_variable.get_mut(variable.index())?.push(index);
            }
        }
        normalize_incidence(&mut constraints_by_variable);

        let mut objective_by_variable = vec![false; variables];
        objective_variables(objective, &mut scope);
        for variable in &scope {
            *objective_by_variable.get_mut(variable.index())? = true;
        }
        let objective_update = ObjectiveUpdate::compile(objective, variables);
        Some(Self { functional_inputs: inputs, functionals_by_variable, constraints_by_variable, objective_by_variable, objective_update })
    }

    fn functionals(&self, variable: usize) -> &[usize] {
        &self.functionals_by_variable[variable]
    }

    pub(super) fn constraints(&self, variable: VarId) -> &[usize] {
        &self.constraints_by_variable[variable.index()]
    }

    fn objective_changes(&self, variable: usize) -> bool {
        self.objective_by_variable[variable]
    }

    #[cfg(test)]
    pub(super) fn functional_arcs(&self) -> usize {
        self.functionals_by_variable.iter().map(Vec::len).sum()
    }
}

impl ObjectiveUpdate {
    fn compile(objective: &Objective, variables: usize) -> Self {
        let minimizing = objective.minimizing();
        let mut coefficients = vec![0i64; variables];
        let terms = match objective {
            Objective::Var(_, variable) | Objective::VarWithAffine(_, variable, _, _) => vec![(1, *variable)],
            Objective::Linear(_, values, variables) => values.iter().copied().zip(variables.iter().copied()).collect(),
            Objective::Expr(_, expression) => {
                let stop = AtomicBool::new(false);
                let Some((_, values, variables)) = expression.affine_form_interruptible(&stop) else {
                    return Self::Recompute;
                };
                values.into_iter().zip(variables).collect()
            }
        };
        for (coefficient, variable) in terms {
            let Some(coefficient) = (if minimizing { Some(coefficient) } else { coefficient.checked_neg() }) else {
                return Self::Recompute;
            };
            let Some(slot) = coefficients.get_mut(variable.index()) else {
                return Self::Recompute;
            };
            let Some(value) = slot.checked_add(coefficient) else {
                return Self::Recompute;
            };
            *slot = value;
        }
        Self::Delta(coefficients)
    }
}

fn normalize_incidence(incidence: &mut [Vec<usize>]) {
    for nodes in incidence {
        nodes.sort_unstable();
        nodes.dedup();
    }
}

fn objective_variables(objective: &Objective, output: &mut Vec<VarId>) {
    output.clear();
    match objective {
        Objective::Var(_, variable) | Objective::VarWithAffine(_, variable, _, _) => output.push(*variable),
        Objective::Linear(_, _, variables) => output.extend(variables.iter().copied()),
        Objective::Expr(_, expression) => expression.collect_vars(output),
    }
    output.sort_unstable();
    output.dedup();
}

#[derive(Clone, Copy, Default)]
pub(super) struct InvariantMetrics {
    pub(super) trials: u64,
    pub(super) rollbacks: u64,
    pub(super) committed: u64,
    pub(super) functional_evaluations: u64,
    pub(super) constraint_evaluations: u64,
    pub(super) objective_evaluations: u64,
    pub(super) objective_full_evaluations: u64,
    pub(super) full_rebuilds: u64,
}

enum TrailEntry {
    Variable { index: usize, value: i32, valid: bool },
    Functional { index: usize, valid: bool },
    Constraint { index: usize, violation: i64 },
}

#[derive(Clone, Copy)]
struct Checkpoint {
    trail: usize,
    invalid_functionals: usize,
    weighted_violation: i128,
    objective: i64,
}

pub(super) struct InvariantState<'a> {
    model: &'a LocalModel,
    assignment: Vec<i32>,
    variable_valid: Vec<bool>,
    functional_valid: Vec<bool>,
    violations: Vec<i64>,
    weights: Vec<i64>,
    invalid_functionals: usize,
    weighted_violation: i128,
    objective: i64,
    trail: Vec<TrailEntry>,
    pending_functionals: BinaryHeap<Reverse<usize>>,
    touched_functionals: Vec<usize>,
    functional_pending: Vec<bool>,
    pending_constraints: Vec<usize>,
    constraint_pending: Vec<bool>,
    objective_pending: bool,
    metrics: InvariantMetrics,
}

impl<'a> InvariantState<'a> {
    pub(super) fn new(model: &'a LocalModel, assignment: Vec<i32>) -> Self {
        let mut state = Self {
            model,
            assignment,
            variable_valid: vec![true; model.domains.len()],
            functional_valid: vec![true; model.functionals.len()],
            violations: vec![0; model.constraints.len()],
            weights: vec![1; model.constraints.len()],
            invalid_functionals: 0,
            weighted_violation: 0,
            objective: 0,
            trail: Vec::new(),
            pending_functionals: BinaryHeap::new(),
            touched_functionals: Vec::new(),
            functional_pending: vec![false; model.functionals.len()],
            pending_constraints: Vec::new(),
            constraint_pending: vec![false; model.constraints.len()],
            objective_pending: false,
            metrics: InvariantMetrics::default(),
        };
        state.rebuild();
        state
    }

    pub(super) fn assignment(&self) -> &[i32] {
        &self.assignment
    }

    pub(super) fn score(&self) -> Score {
        Score {
            violation: combine_violation(i128::from(self.invalid_functionals != 0) * 1_000_000, self.weighted_violation),
            objective: self.objective,
        }
    }

    pub(super) fn complete(&self) -> bool {
        self.invalid_functionals == 0
    }

    pub(super) fn violations(&self) -> &[i64] {
        &self.violations
    }

    pub(super) fn metrics(&self) -> InvariantMetrics {
        self.metrics
    }

    pub(super) fn evaluate(&mut self, variable: VarId, value: i32) -> Score {
        let checkpoint = self.checkpoint();
        self.metrics.trials = self.metrics.trials.saturating_add(1);
        self.apply(variable, value, checkpoint.trail);
        let score = self.score();
        self.rollback(checkpoint);
        score
    }

    pub(super) fn commit(&mut self, variable: VarId, value: i32) -> Score {
        let checkpoint = self.checkpoint();
        self.apply(variable, value, checkpoint.trail);
        self.trail.truncate(checkpoint.trail);
        self.metrics.committed = self.metrics.committed.saturating_add(1);
        self.debug_validate();
        self.score()
    }

    pub(super) fn reset(&mut self, assignment: Vec<i32>) {
        self.assignment = assignment;
        self.rebuild();
    }

    pub(super) fn bump_gls_weights(&mut self) -> bool {
        if !self.complete() || self.score().violation <= 0 {
            return false;
        }
        let mut bumped = false;
        for (index, weight) in self.weights.iter_mut().enumerate() {
            if self.violations[index] > 0 {
                let next = weight.saturating_add(1);
                if next != *weight {
                    *weight = next;
                    self.weighted_violation = self.weighted_violation.saturating_add(i128::from(self.violations[index]));
                    bumped = true;
                }
            }
        }
        bumped
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            trail: self.trail.len(),
            invalid_functionals: self.invalid_functionals,
            weighted_violation: self.weighted_violation,
            objective: self.objective,
        }
    }

    fn apply(&mut self, variable: VarId, value: i32, trail_start: usize) {
        debug_assert!(self.pending_functionals.is_empty());
        debug_assert!(self.pending_constraints.is_empty());
        debug_assert!(!self.objective_pending);

        if self.set_variable(variable.index(), value, true) {
            self.schedule_variable(variable.index());
        }

        while let Some(Reverse(index)) = self.pending_functionals.pop() {
            let target = functional_target(&self.model.functionals[index]).index();
            if self.refresh_functional(index) {
                self.schedule_variable(target);
            }
        }
        for index in self.touched_functionals.drain(..) {
            self.functional_pending[index] = false;
        }

        self.pending_constraints.sort_unstable();
        for position in 0..self.pending_constraints.len() {
            let index = self.pending_constraints[position];
            self.refresh_constraint(index);
        }
        for index in self.pending_constraints.drain(..) {
            self.constraint_pending[index] = false;
        }

        if self.objective_pending {
            self.refresh_objective(trail_start);
            self.metrics.objective_evaluations = self.metrics.objective_evaluations.saturating_add(1);
            self.objective_pending = false;
        }
    }

    fn schedule_variable(&mut self, variable: usize) {
        let functional_count = self.model.invariants.functionals(variable).len();
        for position in 0..functional_count {
            let index = self.model.invariants.functionals(variable)[position];
            if !self.functional_pending[index] {
                self.functional_pending[index] = true;
                self.touched_functionals.push(index);
                self.pending_functionals.push(Reverse(index));
            }
        }
        let constraint_count = self.model.invariants.constraints_by_variable[variable].len();
        for position in 0..constraint_count {
            let index = self.model.invariants.constraints_by_variable[variable][position];
            if !self.constraint_pending[index] {
                self.constraint_pending[index] = true;
                self.pending_constraints.push(index);
            }
        }
        self.objective_pending |= self.model.invariants.objective_changes(variable);
    }

    fn refresh_objective(&mut self, trail_start: usize) {
        match &self.model.invariants.objective_update {
            ObjectiveUpdate::Delta(coefficients) => {
                let mut delta = 0i128;
                for entry in &self.trail[trail_start..] {
                    let TrailEntry::Variable { index, value, .. } = entry else {
                        continue;
                    };
                    delta = delta.saturating_add(
                        i128::from(coefficients[*index]).saturating_mul(i128::from(self.assignment[*index]) - i128::from(*value)),
                    );
                }
                let updated = i128::from(self.objective).saturating_add(delta);
                self.objective = i64::try_from(updated).unwrap_or(if updated.is_negative() { i64::MIN } else { i64::MAX });
            }
            ObjectiveUpdate::Recompute => {
                self.objective = normalized_objective(self.model, &self.assignment);
                self.metrics.objective_full_evaluations = self.metrics.objective_full_evaluations.saturating_add(1);
            }
        }
    }

    fn refresh_functional(&mut self, index: usize) -> bool {
        let functional = &self.model.functionals[index];
        let target = functional_target(functional).index();
        let inputs_valid = self.model.invariants.functional_inputs[index].iter().all(|input| self.variable_valid[input.index()]);
        let value = inputs_valid.then(|| self.model.functional_value(functional, &self.assignment)).flatten();
        let valid = value.is_some_and(|value| self.model.domains[target].contains(value));
        let changed = if let Some(value) = value.filter(|_| valid) {
            self.set_variable(target, value, true)
        } else {
            self.set_variable(target, self.assignment[target], false)
        };
        self.set_functional_valid(index, valid);
        self.metrics.functional_evaluations = self.metrics.functional_evaluations.saturating_add(1);
        changed
    }

    fn refresh_constraint(&mut self, index: usize) {
        let previous = self.violations[index];
        let next = self.model.violation(&self.model.constraints[index], &self.assignment);
        if previous != next {
            self.trail.push(TrailEntry::Constraint { index, violation: previous });
            self.violations[index] = next;
            let delta = i128::from(next) - i128::from(previous);
            self.weighted_violation = self.weighted_violation.saturating_add(delta.saturating_mul(i128::from(self.weights[index])));
        }
        self.metrics.constraint_evaluations = self.metrics.constraint_evaluations.saturating_add(1);
    }

    fn set_variable(&mut self, index: usize, value: i32, valid: bool) -> bool {
        if self.assignment[index] == value && self.variable_valid[index] == valid {
            return false;
        }
        self.trail.push(TrailEntry::Variable { index, value: self.assignment[index], valid: self.variable_valid[index] });
        self.assignment[index] = value;
        self.variable_valid[index] = valid;
        true
    }

    fn set_functional_valid(&mut self, index: usize, valid: bool) {
        let previous = self.functional_valid[index];
        if previous == valid {
            return;
        }
        self.trail.push(TrailEntry::Functional { index, valid: previous });
        self.functional_valid[index] = valid;
        if valid {
            self.invalid_functionals = self.invalid_functionals.saturating_sub(1);
        } else {
            self.invalid_functionals = self.invalid_functionals.saturating_add(1);
        }
    }

    fn rollback(&mut self, checkpoint: Checkpoint) {
        while self.trail.len() > checkpoint.trail {
            match self.trail.pop().expect("the rollback checkpoint bounds the invariant trail") {
                TrailEntry::Variable { index, value, valid } => {
                    self.assignment[index] = value;
                    self.variable_valid[index] = valid;
                }
                TrailEntry::Functional { index, valid } => self.functional_valid[index] = valid,
                TrailEntry::Constraint { index, violation } => self.violations[index] = violation,
            }
        }
        self.invalid_functionals = checkpoint.invalid_functionals;
        self.weighted_violation = checkpoint.weighted_violation;
        self.objective = checkpoint.objective;
        self.metrics.rollbacks = self.metrics.rollbacks.saturating_add(1);
    }

    fn rebuild(&mut self) {
        self.trail.clear();
        self.pending_functionals.clear();
        self.touched_functionals.clear();
        self.functional_pending.fill(false);
        self.pending_constraints.clear();
        self.constraint_pending.fill(false);
        self.objective_pending = false;
        self.variable_valid.fill(true);
        self.functional_valid.fill(true);
        self.invalid_functionals = 0;
        for index in 0..self.model.functionals.len() {
            let functional = &self.model.functionals[index];
            let target = functional_target(functional).index();
            let inputs_valid = self.model.invariants.functional_inputs[index].iter().all(|input| self.variable_valid[input.index()]);
            let value = inputs_valid.then(|| self.model.functional_value(functional, &self.assignment)).flatten();
            let valid = value.is_some_and(|value| self.model.domains[target].contains(value));
            if let Some(value) = value.filter(|_| valid) {
                self.assignment[target] = value;
            }
            self.variable_valid[target] = valid;
            self.functional_valid[index] = valid;
            self.invalid_functionals += usize::from(!valid);
        }
        for (index, constraint) in self.model.constraints.iter().enumerate() {
            self.violations[index] = self.model.violation(constraint, &self.assignment);
        }
        self.weighted_violation = weighted_sum(&self.violations, &self.weights);
        self.objective = normalized_objective(self.model, &self.assignment);
        self.metrics.objective_full_evaluations = self.metrics.objective_full_evaluations.saturating_add(1);
        self.metrics.full_rebuilds = self.metrics.full_rebuilds.saturating_add(1);
        self.debug_validate();
    }

    #[cfg(debug_assertions)]
    fn debug_validate(&self) {
        if !self.complete() {
            return;
        }
        let mut assignment = self.assignment.clone();
        let (score, violations, complete) = self.model.score_breakdown(&mut assignment, &self.weights);
        debug_assert!(complete);
        debug_assert_eq!(assignment, self.assignment, "invariant functional values drifted from full completion");
        debug_assert_eq!(violations, self.violations, "invariant violations drifted from full scoring");
        debug_assert_eq!(score, self.score(), "invariant score drifted from full scoring");
    }

    #[cfg(not(debug_assertions))]
    fn debug_validate(&self) {}
}

fn normalized_objective(model: &LocalModel, assignment: &[i32]) -> i64 {
    let value = model.objective_value(assignment).unwrap_or(i64::MAX / 4);
    if model.objective.minimizing() {
        value
    } else {
        value.saturating_neg()
    }
}
