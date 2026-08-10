//! Compiler from the semantic integer/set model to the CP physical plan.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::constraints::{count, flatten, graph, intension, interval as interval_constraints, lex, linear, primitives, scheduling, table};
use crate::expr::Expr;
use crate::ids::VarId;
use crate::problem::{Objective as PhysicalObjective, Problem};
use crate::Solver;

use super::{Constraint, IntDomain, IntExpr, IntGlobalConstraint, Model, Objective, Relation, SetVarRef};

type CompileStep<T> = Result<Option<T>, CpCompileError>;
type SearchGuidanceParts = (Vec<Option<i32>>, Vec<VarId>, Option<Vec<VarId>>);

const CP_ESTIMATE_BASE_BYTES: u128 = 64 * 1024;

#[inline]
fn interrupted(stop: &AtomicBool) -> bool {
    stop.load(Ordering::Acquire)
}

#[inline]
fn wide(value: usize) -> u128 {
    u128::try_from(value).unwrap_or(u128::MAX)
}

macro_rules! require_step {
    ($step:expr) => {
        match $step? {
            Some(value) => value,
            None => return Ok(None),
        }
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CpCompileError {
    pub(crate) reason: String,
}

impl CpCompileError {
    fn new(reason: impl Into<String>) -> Self {
        Self { reason: reason.into() }
    }
}

impl std::fmt::Display for CpCompileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl std::error::Error for CpCompileError {}

#[derive(Clone)]
pub(crate) struct CompiledSet {
    pub values: Vec<i32>,
    pub membership: Vec<VarId>,
}

/// Fully lowered CP plan with semantic decode maps.
#[derive(Clone)]
pub(crate) struct CompiledCp {
    problem: Problem,
    objectives: Vec<PhysicalObjective>,
    int_variables: Vec<VarId>,
    sets: Vec<CompiledSet>,
    estimated_bytes: u64,
}

pub(crate) struct DecodedCpAssignment {
    pub integers: Vec<Option<i64>>,
    pub sets: Vec<Vec<i32>>,
}

impl CompiledCp {
    pub(crate) fn estimate_semantic_bytes_interruptible(model: &Model, stop: &AtomicBool) -> Option<u64> {
        estimate_semantic_bytes(model, stop)
    }

    #[cfg(any(feature = "python", test))]
    pub(crate) fn compile_interruptible(model: &Model, stop: &AtomicBool) -> CompileStep<Self> {
        let Some(estimated_bytes) = Self::estimate_semantic_bytes_interruptible(model, stop) else {
            return Ok(None);
        };
        Self::compile_with_estimate_interruptible(model, estimated_bytes, stop)
    }

    pub(crate) fn compile_with_estimate_interruptible(model: &Model, estimated_bytes: u64, stop: &AtomicBool) -> CompileStep<Self> {
        match model.validate_interruptible(stop) {
            Ok(true) => {}
            Ok(false) => return Ok(None),
            Err(errors) => return Err(CpCompileError::new(errors.join("; "))),
        }
        Self::compile_validated_with_estimate_interruptible(model, estimated_bytes, stop)
    }

    /// Lower a model whose semantic invariants have already been checked by the
    /// orchestrator. Callers must not expose this entry point at a public model
    /// boundary; use `compile_with_estimate_interruptible` there instead.
    pub(crate) fn compile_validated_with_estimate_interruptible(
        model: &Model,
        estimated_bytes: u64,
        stop: &AtomicBool,
    ) -> CompileStep<Self> {
        if !model.lists.is_empty() || !model.intervals.is_empty() {
            return Err(CpCompileError::new("CP integer lowering does not accept list or interval variables"));
        }

        if interrupted(stop) {
            return Ok(None);
        }
        let mut solver = Solver::new();
        let mut int_variables = Vec::with_capacity(model.int_vars.len());
        for domain in &model.int_vars {
            if interrupted(stop) {
                return Ok(None);
            }
            int_variables.push(match domain {
                IntDomain::Bool => solver.new_var_range(0, 1),
                IntDomain::Range { lo, hi } => solver.new_var_range(*lo, *hi),
                IntDomain::Set(values) => {
                    let mut copied = Vec::with_capacity(values.len());
                    for &value in values {
                        if interrupted(stop) {
                            return Ok(None);
                        }
                        copied.push(value);
                    }
                    let mut values = copied;
                    values.sort_unstable();
                    if interrupted(stop) {
                        return Ok(None);
                    }
                    values.dedup();
                    solver.new_var_set(&values)
                }
            });
        }

        let mut sets = Vec::with_capacity(model.sets.len());
        for set in &model.sets {
            if interrupted(stop) {
                return Ok(None);
            }
            let mut values = Vec::with_capacity(set.possible.len());
            for &value in &set.possible {
                if interrupted(stop) {
                    return Ok(None);
                }
                values.push(value);
            }
            let mut membership = Vec::with_capacity(values.len());
            for value in &values {
                if interrupted(stop) {
                    return Ok(None);
                }
                let variable = solver.new_var_range(0, 1);
                if set.required.contains(value) {
                    solver.store.fix(variable, 1).expect("required set member has a Boolean domain");
                }
                membership.push(variable);
            }
            sets.push(CompiledSet { values, membership });
        }

        for constraint in &model.constraints {
            if interrupted(stop) {
                return Ok(None);
            }
            if post_constraint(&mut solver, &int_variables, &sets, constraint, stop)?.is_none() {
                return Ok(None);
            }
        }

        let mut objectives = Vec::with_capacity(model.objectives.len());
        for objective in &model.objectives {
            if interrupted(stop) {
                return Ok(None);
            }
            objectives.push(match objective {
                Objective::IntExpr { minimize, expr: IntExpr::Variable(variable) } => {
                    Ok(PhysicalObjective::Var(*minimize, int_variables[variable.0]))
                }
                Objective::IntExpr { minimize, expr } => {
                    let Some(expression) = compile_expr(expr, &int_variables, stop)? else {
                        return Ok(None);
                    };
                    Ok(PhysicalObjective::Expr(*minimize, expression))
                }
                Objective::ListTerms { .. } | Objective::Makespan { .. } => {
                    Err(CpCompileError::new("CP integer lowering received a non-integer objective"))
                }
            }?);
        }

        let mut search = Vec::with_capacity(
            int_variables.len().saturating_add(sets.iter().map(|set| set.membership.len()).fold(0usize, usize::saturating_add)),
        );
        let mut search_members = BTreeSet::new();
        for &variable in &int_variables {
            if interrupted(stop) {
                return Ok(None);
            }
            search.push(variable);
            search_members.insert(variable);
        }
        for set in &sets {
            for &variable in &set.membership {
                if interrupted(stop) {
                    return Ok(None);
                }
                search.push(variable);
                search_members.insert(variable);
            }
        }
        for index in 0..solver.store.disjunctive_pair_count() {
            if interrupted(stop) {
                return Ok(None);
            }
            let variable = solver.store.disjunctive_order_var(index);
            if search_members.insert(variable) {
                search.push(variable);
            }
        }
        if interrupted(stop) {
            return Ok(None);
        }
        Ok(Some(Self {
            problem: Problem { solver, search, objective: objectives.first().cloned() },
            objectives,
            int_variables,
            sets,
            estimated_bytes,
        }))
    }

    pub(crate) fn problem(&self) -> &Problem {
        &self.problem
    }

    pub(crate) fn objectives(&self) -> &[PhysicalObjective] {
        &self.objectives
    }

    pub(crate) fn int_variables(&self) -> &[VarId] {
        &self.int_variables
    }

    pub(crate) fn sets(&self) -> &[CompiledSet] {
        &self.sets
    }

    pub(crate) fn compile_expression(&self, expression: &IntExpr) -> Result<Expr, CpCompileError> {
        let stop = AtomicBool::new(false);
        self.compile_expression_interruptible(expression, &stop)?
            .ok_or_else(|| CpCompileError::new("CP expression compilation was interrupted unexpectedly"))
    }

    pub(crate) fn compile_expression_interruptible(&self, expression: &IntExpr, stop: &AtomicBool) -> CompileStep<Expr> {
        compile_expr(expression, &self.int_variables, stop)
    }

    pub(crate) fn fix_objective(&self, problem: &mut Problem, tier: usize, value: i64) -> Result<(), CpCompileError> {
        let objective = self.objectives.get(tier).ok_or_else(|| CpCompileError::new(format!("unknown CP objective tier {tier}")))?;
        let expression = match objective {
            PhysicalObjective::Var(_, variable) => Expr::Var(*variable),
            PhysicalObjective::Linear(_, coefficients, variables) => Expr::Add(
                coefficients
                    .iter()
                    .zip(variables)
                    .map(|(&coefficient, &variable)| Expr::Mul(vec![Expr::Const(coefficient), Expr::Var(variable)]))
                    .collect(),
            ),
            PhysicalObjective::Expr(_, expression) => expression.clone(),
        };
        intension::intension(&mut problem.solver, Expr::Eq(Box::new(expression), Box::new(Expr::Const(value))));
        Ok(())
    }

    pub(crate) fn estimated_bytes(&self) -> u64 {
        self.estimated_bytes
    }

    pub(crate) fn search_guidance_interruptible(
        &self,
        hints: &[(usize, i32)],
        branch_order: &[usize],
        primary_branch_scope: Option<&[usize]>,
        stop: &AtomicBool,
    ) -> CompileStep<SearchGuidanceParts> {
        if interrupted(stop) {
            return Ok(None);
        }
        if hints.is_empty() && branch_order.is_empty() && primary_branch_scope.is_none() {
            return Ok(Some((Vec::new(), Vec::new(), None)));
        }
        let mut phase = if hints.is_empty() { Vec::new() } else { vec![None; self.problem.solver.store.num_vars()] };
        let mut seen_hints = BTreeSet::new();
        for &(index, value) in hints {
            if interrupted(stop) {
                return Ok(None);
            }
            let variable = self
                .int_variables
                .get(index)
                .copied()
                .ok_or_else(|| CpCompileError::new(format!("hint references unknown integer variable {index}")))?;
            if !seen_hints.insert(index) {
                return Err(CpCompileError::new(format!("integer variable {index} has more than one hint")));
            }
            if !self.problem.solver.store.contains(variable, value) {
                return Err(CpCompileError::new(format!("hint value {value} is outside integer variable {index}'s domain")));
            }
            phase[variable.index()] = Some(value);
        }

        let mut semantic_scope = BTreeSet::new();
        let primary_scope =
            match primary_branch_scope {
                None => None,
                Some(scope) => {
                    let mut physical = Vec::with_capacity(scope.len());
                    for &index in scope {
                        if interrupted(stop) {
                            return Ok(None);
                        }
                        if !semantic_scope.insert(index) {
                            return Err(CpCompileError::new(format!("integer variable {index} appears twice in primary_branch_scope")));
                        }
                        physical.push(self.int_variables.get(index).copied().ok_or_else(|| {
                            CpCompileError::new(format!("primary_branch_scope references unknown integer variable {index}"))
                        })?);
                    }
                    Some(physical)
                }
            };

        let mut seen_order = BTreeSet::new();
        let mut order = Vec::with_capacity(branch_order.len());
        for &index in branch_order {
            if interrupted(stop) {
                return Ok(None);
            }
            if !seen_order.insert(index) {
                return Err(CpCompileError::new(format!("integer variable {index} appears twice in branch_order")));
            }
            if primary_branch_scope.is_some() && !semantic_scope.contains(&index) {
                return Err(CpCompileError::new(format!("branch_order variable {index} is outside primary_branch_scope")));
            }
            order.push(
                self.int_variables
                    .get(index)
                    .copied()
                    .ok_or_else(|| CpCompileError::new(format!("branch_order references unknown integer variable {index}")))?,
            );
        }
        Ok(Some((phase, order, primary_scope)))
    }

    pub(crate) fn decode(&self, values: &[i32]) -> Result<DecodedCpAssignment, String> {
        if values.len() != self.problem.search.len() {
            return Err(format!("CP assignment has {} values, expected {}", values.len(), self.problem.search.len()));
        }
        let by_variable = self.problem.search.iter().copied().zip(values.iter().copied()).collect::<BTreeMap<_, _>>();
        let integers = self.int_variables.iter().map(|variable| by_variable.get(variable).copied().map(i64::from)).collect::<Vec<_>>();
        let sets = self
            .sets
            .iter()
            .map(|set| {
                set.values
                    .iter()
                    .zip(&set.membership)
                    .filter_map(|(value, variable)| (by_variable.get(variable) == Some(&1)).then_some(*value))
                    .collect()
            })
            .collect();
        Ok(DecodedCpAssignment { integers, sets })
    }
}

fn estimate_semantic_bytes(model: &Model, stop: &AtomicBool) -> Option<u64> {
    if interrupted(stop) {
        return None;
    }
    let mut bytes = CP_ESTIMATE_BASE_BYTES;
    for domain in &model.int_vars {
        if interrupted(stop) {
            return None;
        }
        let cells = match domain {
            IntDomain::Bool => 2,
            IntDomain::Range { lo, hi } => {
                let width = u128::try_from(i64::from(*hi) - i64::from(*lo) + 1).unwrap_or(u128::MAX);
                if width <= 4_096 {
                    width
                } else {
                    1
                }
            }
            // A dense explicit set can use a value span up to four times its
            // cardinality. Count that worst case without allocating a set.
            IntDomain::Set(values) => wide(values.len()).saturating_mul(4),
        };
        bytes = bytes.saturating_add(1024).saturating_add(cells.saturating_mul(24));
    }
    for set in &model.sets {
        if interrupted(stop) {
            return None;
        }
        bytes = bytes.saturating_add(wide(set.possible.len()).saturating_mul(1536));
    }
    for constraint in &model.constraints {
        bytes = bytes.saturating_add(estimate_constraint_bytes(model, constraint, stop)?);
    }
    for objective in &model.objectives {
        if interrupted(stop) {
            return None;
        }
        if let Objective::IntExpr { expr, .. } = objective {
            bytes = bytes.saturating_add(estimate_expr_bytes(expr, stop)?);
        }
    }
    Some(u64::try_from(bytes).unwrap_or(u64::MAX))
}

fn estimate_constraint_bytes(model: &Model, constraint: &Constraint, stop: &AtomicBool) -> Option<u128> {
    if interrupted(stop) {
        return None;
    }
    Some(match constraint {
        Constraint::Intension(expr) => 1024u128.saturating_add(estimate_expr_bytes(expr, stop)?),
        Constraint::Selected { constraint, .. } => 512u128.saturating_add(estimate_constraint_bytes(model, constraint, stop)?),
        Constraint::Linear { terms, .. } => 1024u128.saturating_add(wide(terms.len()).saturating_mul(96)),
        Constraint::Clause(literals) => 1024u128.saturating_add(wide(literals.len()).saturating_mul(160)),
        Constraint::IntegerGlobal(global) => estimate_global_bytes(model, global, stop)?,
        Constraint::SetSubset { subset, superset } | Constraint::SetDisjoint { left: subset, right: superset } => {
            let cells = model
                .sets
                .get(subset.0)
                .map_or(0, |set| wide(set.possible.len()))
                .saturating_add(model.sets.get(superset.0).map_or(0, |set| wide(set.possible.len())));
            1024u128.saturating_add(cells.saturating_mul(1536))
        }
        Constraint::SetCardinality { set, .. } => {
            2048u128.saturating_add(model.sets.get(set.0).map_or(0, |declaration| wide(declaration.possible.len())).saturating_mul(192))
        }
        Constraint::ListPartition { .. }
        | Constraint::ListPartitionWithCoverage { .. }
        | Constraint::SameList { .. }
        | Constraint::ItemPrecedence { .. }
        | Constraint::CollectionGlobal(_)
        | Constraint::ListLength { .. }
        | Constraint::ListItemSum { .. }
        | Constraint::ListReduction(_)
        | Constraint::IntervalPrecedence { .. }
        | Constraint::IntervalAlternative { .. }
        | Constraint::IntervalEndpointRelation { .. }
        | Constraint::IntervalResource(_) => 1024,
    })
}

fn estimate_expr_bytes(expr: &IntExpr, stop: &AtomicBool) -> Option<u128> {
    if interrupted(stop) {
        return None;
    }
    let children = match expr {
        IntExpr::Constant(_) | IntExpr::Variable(_) => 0,
        IntExpr::Neg(value) | IntExpr::Abs(value) | IntExpr::Not(value) => estimate_expr_bytes(value, stop)?,
        IntExpr::Add(values)
        | IntExpr::Mul(values)
        | IntExpr::Min(values)
        | IntExpr::Max(values)
        | IntExpr::And(values)
        | IntExpr::Or(values) => {
            let mut bytes = wide(values.len()).saturating_mul(32);
            for value in values {
                bytes = bytes.saturating_add(estimate_expr_bytes(value, stop)?);
            }
            bytes
        }
        IntExpr::Sub(left, right)
        | IntExpr::Div(left, right)
        | IntExpr::Mod(left, right)
        | IntExpr::Eq(left, right)
        | IntExpr::Ne(left, right)
        | IntExpr::Lt(left, right)
        | IntExpr::Le(left, right)
        | IntExpr::Gt(left, right)
        | IntExpr::Ge(left, right)
        | IntExpr::Imp(left, right)
        | IntExpr::Iff(left, right) => estimate_expr_bytes(left, stop)?.saturating_add(estimate_expr_bytes(right, stop)?),
        IntExpr::IfThenElse(condition, then_value, else_value) => estimate_expr_bytes(condition, stop)?
            .saturating_add(estimate_expr_bytes(then_value, stop)?)
            .saturating_add(estimate_expr_bytes(else_value, stop)?),
    };
    Some(96u128.saturating_add(children))
}

fn domain_cardinality(model: &Model, variable: super::IntVarRef) -> u128 {
    match model.int_vars.get(variable.0) {
        Some(IntDomain::Bool) => 2,
        Some(IntDomain::Range { lo, hi }) => u128::try_from(i64::from(*hi) - i64::from(*lo) + 1).unwrap_or(u128::MAX),
        Some(IntDomain::Set(values)) => wide(values.len()),
        None => 0,
    }
}

fn domain_cells(model: &Model, variables: &[super::IntVarRef], stop: &AtomicBool) -> Option<u128> {
    let mut cells = 0u128;
    for &variable in variables {
        if interrupted(stop) {
            return None;
        }
        cells = cells.saturating_add(domain_cardinality(model, variable));
    }
    Some(cells)
}

fn domain_bounds(model: &Model, variable: super::IntVarRef, stop: &AtomicBool) -> Option<Option<(i64, i64)>> {
    if interrupted(stop) {
        return None;
    }
    Some(match model.int_vars.get(variable.0)? {
        IntDomain::Bool => Some((0, 1)),
        IntDomain::Range { lo, hi } => Some((i64::from(*lo), i64::from(*hi))),
        IntDomain::Set(values) => {
            let mut lo = i32::MAX;
            let mut hi = i32::MIN;
            for &value in values {
                if interrupted(stop) {
                    return None;
                }
                lo = lo.min(value);
                hi = hi.max(value);
            }
            (!values.is_empty()).then_some((i64::from(lo), i64::from(hi)))
        }
    })
}

fn fixed_cumulative_horizon(model: &Model, starts: &[super::IntVarRef], durations: &[i64], stop: &AtomicBool) -> Option<u128> {
    let mut first = true;
    let mut earliest = i64::MAX;
    let mut latest_end = i64::MIN;
    for (&start, &duration) in starts.iter().zip(durations) {
        if interrupted(stop) {
            return None;
        }
        let Some((lo, hi)) = domain_bounds(model, start, stop)? else {
            continue;
        };
        first = false;
        earliest = earliest.min(lo);
        latest_end = latest_end.max(hi.saturating_add(duration.max(0)));
    }
    Some(if first { 0 } else { u128::try_from(latest_end.saturating_sub(earliest).max(0)).unwrap_or(u128::MAX) })
}

fn variable_cumulative_horizon(
    model: &Model,
    starts: &[super::IntVarRef],
    durations: &[super::IntVarRef],
    stop: &AtomicBool,
) -> Option<u128> {
    let mut first = true;
    let mut earliest = i64::MAX;
    let mut latest_end = i64::MIN;
    for (&start, &duration) in starts.iter().zip(durations) {
        if interrupted(stop) {
            return None;
        }
        let Some((lo, hi)) = domain_bounds(model, start, stop)? else {
            continue;
        };
        let duration_hi = domain_bounds(model, duration, stop)?.map_or(0, |(_, hi)| hi.max(0));
        first = false;
        earliest = earliest.min(lo);
        latest_end = latest_end.max(hi.saturating_add(duration_hi));
    }
    Some(if first { 0 } else { u128::try_from(latest_end.saturating_sub(earliest).max(0)).unwrap_or(u128::MAX) })
}

fn table_support_count(tuples: &[Vec<i32>], arity: usize, stop: &AtomicBool) -> Option<u128> {
    let mut total = 0u128;
    // One set at a time keeps the estimator's peak temporary memory bounded by
    // a single column instead of duplicating the whole table body.
    let mut values = BTreeSet::new();
    for column in 0..arity {
        values.clear();
        for tuple in tuples {
            if interrupted(stop) {
                return None;
            }
            if let Some(&value) = tuple.get(column).filter(|&&value| value != table::STAR) {
                values.insert(value);
            }
        }
        total = total.saturating_add(wide(values.len()));
    }
    Some(total)
}

fn estimate_global_bytes(model: &Model, global: &IntGlobalConstraint, stop: &AtomicBool) -> Option<u128> {
    if interrupted(stop) {
        return None;
    }
    let bytes = match global {
        IntGlobalConstraint::AllDifferent { variables, except } => {
            let n = wide(variables.len());
            let domains = domain_cells(model, variables, stop)?;
            if except.is_empty() {
                4096u128.saturating_add(n.saturating_mul(512)).saturating_add(domains.saturating_mul(48))
            } else {
                let pairs = n.saturating_mul(n.saturating_sub(1)) / 2;
                4096u128.saturating_add(pairs.saturating_mul(wide(except.len()).saturating_add(1)).saturating_mul(256))
            }
        }
        IntGlobalConstraint::AllEqual(variables)
        | IntGlobalConstraint::Ordered { variables, .. }
        | IntGlobalConstraint::Instantiation { variables, .. }
        | IntGlobalConstraint::Count { variables, .. } => 2048u128.saturating_add(wide(variables.len()).saturating_mul(256)),
        IntGlobalConstraint::Minimum { variables, .. } | IntGlobalConstraint::Maximum { variables, .. } => {
            2048u128.saturating_add(wide(variables.len()).saturating_mul(256))
        }
        IntGlobalConstraint::Element { array, .. } => 2048u128.saturating_add(wide(array.len()).saturating_mul(256)),
        IntGlobalConstraint::ElementConst { array, .. } => 2048u128.saturating_add(wide(array.len()).saturating_mul(64)),
        IntGlobalConstraint::Cardinality { variables, values, closed, .. } => {
            let factor = if *closed { 2 } else { 1 };
            4096u128.saturating_add(wide(variables.len()).saturating_mul(wide(values.len())).saturating_mul(64)).saturating_mul(factor)
        }
        IntGlobalConstraint::NValues { variables, .. } => 4096u128.saturating_add(domain_cells(model, variables, stop)?.saturating_mul(32)),
        IntGlobalConstraint::Table { variables, tuples, positive } => {
            let arity = wide(variables.len());
            let tuple_count = wide(tuples.len());
            let words = tuple_count.saturating_add(63) / 64;
            let support_values = table_support_count(tuples, variables.len(), stop)?;
            let support_bitsets = support_values.saturating_mul(words).saturating_mul(8);
            let support_maps = support_values.saturating_mul(64);
            let table_body = arity.saturating_mul(tuple_count).saturating_mul(4);
            let layered_scratch = arity.saturating_add(1).saturating_mul(words).saturating_mul(if *positive { 48 } else { 32 });
            let residues = domain_cells(model, variables, stop)?.saturating_mul(if *positive { 160 } else { 32 });
            8192u128
                .saturating_add(support_bitsets)
                .saturating_add(support_maps)
                .saturating_add(table_body)
                .saturating_add(layered_scratch)
                .saturating_add(residues)
        }
        IntGlobalConstraint::Regular { variables, automaton } => {
            let layers = wide(variables.len()).saturating_add(1);
            4096u128
                .saturating_add(layers.saturating_mul(wide(automaton.states)).saturating_mul(4))
                .saturating_add(wide(automaton.states).saturating_mul(96))
                .saturating_add(wide(automaton.transitions.len()).saturating_mul(160))
        }
        IntGlobalConstraint::Mdd { variables, mdd } => {
            let mut arcs = 0u128;
            for layer in &mdd.layers {
                if interrupted(stop) {
                    return None;
                }
                arcs = arcs.saturating_add(wide(layer.len()));
            }
            let nodes = mdd.nodes_per_layer.iter().fold(0u128, |sum, &count| sum.saturating_add(wide(count)));
            4096u128
                .saturating_add(wide(variables.len()).saturating_mul(64))
                .saturating_add(arcs.saturating_mul(96))
                .saturating_add(nodes.saturating_mul(16))
        }
        IntGlobalConstraint::Lex { left, right, .. } | IntGlobalConstraint::Channel { left, right } => {
            2048u128.saturating_add(wide(left.len().saturating_add(right.len())).saturating_mul(192))
        }
        IntGlobalConstraint::LexChain { rows, .. } => {
            let cells = rows.iter().fold(0u128, |sum, row| sum.saturating_add(wide(row.len())));
            2048u128.saturating_add(cells.saturating_mul(384))
        }
        IntGlobalConstraint::Circuit { successors, .. } => {
            4096u128.saturating_add(domain_cells(model, successors, stop)?.saturating_mul(64))
        }
        IntGlobalConstraint::NoOverlap { starts, .. } => 4096u128.saturating_add(wide(starts.len()).saturating_mul(512)),
        IntGlobalConstraint::OptionalNoOverlap { starts, .. } => {
            let n = wide(starts.len());
            let pairs = n.saturating_mul(n.saturating_sub(1)) / 2;
            8192u128.saturating_add(n.saturating_mul(2048)).saturating_add(pairs.saturating_mul(8 * 1024))
        }
        IntGlobalConstraint::AlternativeChannel { starts, .. } => 8192u128.saturating_add(wide(starts.len()).saturating_mul(3 * 1024)),
        IntGlobalConstraint::Cumulative { starts, durations, .. } => 8192u128
            .saturating_add(wide(starts.len()).saturating_mul(1024))
            .saturating_add(fixed_cumulative_horizon(model, starts, durations, stop)?.saturating_mul(16)),
        IntGlobalConstraint::CumulativeVar { starts, durations, .. } => 8192u128
            .saturating_add(wide(starts.len()).saturating_mul(1024))
            .saturating_add(variable_cumulative_horizon(model, starts, durations, stop)?.saturating_mul(16)),
        IntGlobalConstraint::BinPacking { items, capacities, .. } => {
            4096u128.saturating_add(wide(items.len()).saturating_mul(512)).saturating_add(wide(capacities.len()).saturating_mul(128))
        }
        IntGlobalConstraint::BinLoads { items, loads, .. } => {
            4096u128.saturating_add(wide(items.len()).saturating_mul(wide(loads.len())).saturating_mul(3 * 1024))
        }
        IntGlobalConstraint::Knapsack { variables, .. } => 4096u128.saturating_add(wide(variables.len()).saturating_mul(384)),
        IntGlobalConstraint::ValuePrecedence { variables, values, covered } => {
            let n = wide(variables.len());
            let pairs = wide(values.len().saturating_sub(1));
            let coverage = if *covered { wide(values.len()).saturating_mul(n) } else { 0 };
            4096u128
                .saturating_add(pairs.saturating_mul(n).saturating_mul(n).saturating_mul(128))
                .saturating_add(coverage.saturating_mul(128))
        }
    };
    Some(bytes)
}

fn physical_relation(relation: Relation) -> linear::Relation {
    match relation {
        Relation::Eq => linear::Relation::Eq,
        Relation::Ne => linear::Relation::Ne,
        Relation::Le => linear::Relation::Le,
        Relation::Lt => linear::Relation::Lt,
        Relation::Ge => linear::Relation::Ge,
        Relation::Gt => linear::Relation::Gt,
    }
}

fn variables(ids: &[super::IntVarRef], map: &[VarId], stop: &AtomicBool) -> Option<Vec<VarId>> {
    let mut result = Vec::with_capacity(ids.len());
    for reference in ids {
        if interrupted(stop) {
            return None;
        }
        result.push(map[reference.0]);
    }
    Some(result)
}

fn post_constraint(
    solver: &mut Solver,
    int_variables: &[VarId],
    sets: &[CompiledSet],
    constraint: &Constraint,
    stop: &AtomicBool,
) -> CompileStep<()> {
    if interrupted(stop) {
        return Ok(None);
    }
    match constraint {
        Constraint::Intension(expr) => {
            let expression = require_step!(compile_expr(expr, int_variables, stop));
            if !intension::intension_interruptible(solver, expression, stop) {
                return Ok(None);
            }
        }
        Constraint::Selected { selector, constraint } => {
            solver.set_selector(Some(int_variables[selector.0]));
            let result = post_constraint(solver, int_variables, sets, constraint, stop);
            solver.set_selector(None);
            if result?.is_none() {
                return Ok(None);
            }
        }
        Constraint::Linear { terms, relation, rhs } => {
            let mut coefficients = Vec::with_capacity(terms.len());
            let mut variables = Vec::with_capacity(terms.len());
            for &(coefficient, variable) in terms {
                if interrupted(stop) {
                    return Ok(None);
                }
                coefficients.push(coefficient);
                variables.push(int_variables[variable.0]);
            }
            if !linear::linear_interruptible(solver, coefficients, variables, physical_relation(*relation), *rhs, stop) {
                return Ok(None);
            }
        }
        Constraint::Clause(literals) => {
            let mut terms = Vec::with_capacity(literals.len());
            for literal in literals {
                if interrupted(stop) {
                    return Ok(None);
                }
                let variable = Expr::Var(int_variables[literal.variable.0]);
                terms.push(if literal.positive {
                    Expr::Eq(Box::new(variable), Box::new(Expr::Const(1)))
                } else {
                    Expr::Eq(Box::new(variable), Box::new(Expr::Const(0)))
                });
            }
            if !intension::intension_interruptible(solver, Expr::Or(terms), stop) {
                return Ok(None);
            }
        }
        Constraint::IntegerGlobal(global) => {
            if post_global(solver, int_variables, global, stop)?.is_none() {
                return Ok(None);
            }
        }
        Constraint::SetSubset { subset, superset } => {
            let Some(values) = set_values(sets, [*subset, *superset], stop) else {
                return Ok(None);
            };
            for value in values {
                if interrupted(stop) {
                    return Ok(None);
                }
                let left = membership_for(&sets[subset.0], value);
                let right = membership_for(&sets[superset.0], value);
                match (left, right) {
                    (Some(left), Some(right)) => {
                        if !intension::intension_interruptible(
                            solver,
                            Expr::Imp(
                                Box::new(Expr::Eq(Box::new(Expr::Var(left)), Box::new(Expr::Const(1)))),
                                Box::new(Expr::Eq(Box::new(Expr::Var(right)), Box::new(Expr::Const(1)))),
                            ),
                            stop,
                        ) {
                            return Ok(None);
                        }
                    }
                    (Some(left), None) => {
                        solver.store.fix(left, 0).map_err(|_| CpCompileError::new("set subset bounds are inconsistent"))?;
                    }
                    (None, _) => {}
                }
            }
        }
        Constraint::SetDisjoint { left, right } => {
            let Some(values) = set_values(sets, [*left, *right], stop) else {
                return Ok(None);
            };
            for value in values {
                if interrupted(stop) {
                    return Ok(None);
                }
                if let (Some(left), Some(right)) = (membership_for(&sets[left.0], value), membership_for(&sets[right.0], value)) {
                    if !linear::linear_interruptible(solver, vec![1, 1], vec![left, right], linear::Relation::Le, 1, stop) {
                        return Ok(None);
                    }
                }
            }
        }
        Constraint::SetCardinality { set, min, max } => {
            let membership = &sets[set.0].membership;
            let minimum = i64::try_from(*min).map_err(|_| CpCompileError::new("set bound exceeds i64"))?;
            let maximum = i64::try_from(*max).map_err(|_| CpCompileError::new("set bound exceeds i64"))?;
            if !linear::sum_interruptible(solver, membership, linear::Relation::Ge, minimum, stop)
                || !linear::sum_interruptible(solver, membership, linear::Relation::Le, maximum, stop)
            {
                return Ok(None);
            }
        }
        Constraint::ListPartition { .. }
        | Constraint::ListPartitionWithCoverage { .. }
        | Constraint::SameList { .. }
        | Constraint::ItemPrecedence { .. }
        | Constraint::CollectionGlobal(_)
        | Constraint::ListLength { .. }
        | Constraint::ListItemSum { .. }
        | Constraint::ListReduction(_)
        | Constraint::IntervalPrecedence { .. }
        | Constraint::IntervalAlternative { .. }
        | Constraint::IntervalEndpointRelation { .. }
        | Constraint::IntervalResource(_) => {
            return Err(CpCompileError::new("CP integer lowering received a collection or interval constraint"));
        }
    }
    if interrupted(stop) {
        Ok(None)
    } else {
        Ok(Some(()))
    }
}

fn post_global(solver: &mut Solver, map: &[VarId], global: &IntGlobalConstraint, stop: &AtomicBool) -> CompileStep<()> {
    if interrupted(stop) {
        return Ok(None);
    }
    macro_rules! mapped {
        ($ids:expr) => {
            match variables($ids, map, stop) {
                Some(variables) => variables,
                None => return Ok(None),
            }
        };
    }
    match global {
        IntGlobalConstraint::AllDifferent { variables: ids, except } => {
            let variables = mapped!(ids);
            if except.is_empty() {
                primitives::all_different(solver, &variables);
            } else if !flatten::post_all_different_except_interruptible(solver, &variables, except, stop) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::AllEqual(ids) => primitives::all_equal(solver, &mapped!(ids)),
        IntGlobalConstraint::Ordered { variables: ids, relation } => {
            primitives::ordered(solver, &mapped!(ids), physical_relation(*relation));
        }
        IntGlobalConstraint::Instantiation { variables: ids, values } => {
            primitives::instantiation(solver, &mapped!(ids), values);
        }
        IntGlobalConstraint::Minimum { target, variables: ids } => {
            primitives::minimum(solver, map[target.0], &mapped!(ids));
        }
        IntGlobalConstraint::Maximum { target, variables: ids } => {
            primitives::maximum(solver, map[target.0], &mapped!(ids));
        }
        IntGlobalConstraint::Element { array, index, value } => {
            primitives::element(solver, &mapped!(array), map[index.0], map[value.0]);
        }
        IntGlobalConstraint::ElementConst { array, index, value } => {
            primitives::element_const(solver, array, map[index.0], map[value.0]);
        }
        IntGlobalConstraint::Count { variables: ids, value, relation, count: count_value } => {
            count::count(solver, &mapped!(ids), *value, physical_relation(*relation), *count_value);
        }
        IntGlobalConstraint::Cardinality { variables: ids, values, lower, upper, closed } => {
            count::cardinality(solver, &mapped!(ids), values, lower, upper, *closed);
        }
        IntGlobalConstraint::NValues { variables: ids, relation, count: count_value } => {
            count::n_values(solver, &mapped!(ids), physical_relation(*relation), *count_value);
        }
        IntGlobalConstraint::Table { variables: ids, tuples, positive } => {
            if !table::extension_interruptible(solver, &mapped!(ids), tuples, *positive, stop) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::Regular { variables: ids, automaton } => {
            let mut accept = Vec::with_capacity(automaton.accepting.len());
            for &state in &automaton.accepting {
                if interrupted(stop) {
                    return Ok(None);
                }
                accept.push(state);
            }
            let mut transitions = Vec::with_capacity(automaton.transitions.len());
            for &transition in &automaton.transitions {
                if interrupted(stop) {
                    return Ok(None);
                }
                transitions.push(transition);
            }
            if !table::regular_interruptible(
                solver,
                &mapped!(ids),
                table::Dfa { n_states: automaton.states, start: automaton.start, accept, transitions },
                stop,
            ) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::Mdd { variables: ids, mdd } => {
            let mut layers = Vec::with_capacity(mdd.layers.len());
            for layer in &mdd.layers {
                if interrupted(stop) {
                    return Ok(None);
                }
                let mut physical_layer = Vec::with_capacity(layer.len());
                for arc in layer {
                    if interrupted(stop) {
                        return Ok(None);
                    }
                    physical_layer.push(table::MddArc { from: arc.from, value: arc.value, to: arc.to });
                }
                layers.push(physical_layer);
            }
            let mut nodes_per_layer = Vec::with_capacity(mdd.nodes_per_layer.len());
            for &nodes in &mdd.nodes_per_layer {
                if interrupted(stop) {
                    return Ok(None);
                }
                nodes_per_layer.push(nodes);
            }
            if !table::mdd_interruptible(solver, &mapped!(ids), table::Mdd { layers, nodes_per_layer }, stop) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::Lex { left, right, strict } => {
            lex::lex(solver, &mapped!(left), &mapped!(right), *strict);
        }
        IntGlobalConstraint::LexChain { rows, strict } => {
            for pair in rows.windows(2) {
                if interrupted(stop) {
                    return Ok(None);
                }
                lex::lex(solver, &mapped!(&pair[0]), &mapped!(&pair[1]), *strict);
            }
        }
        IntGlobalConstraint::Channel { left, right } => {
            lex::channel(solver, &mapped!(left), &mapped!(right));
        }
        IntGlobalConstraint::Circuit { successors, cutset } => {
            graph::circuit_with(solver, &mapped!(successors), *cutset);
        }
        IntGlobalConstraint::NoOverlap { starts, durations } => {
            if !scheduling::no_overlap_interruptible(solver, &mapped!(starts), durations, stop) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::OptionalNoOverlap { starts, durations, presences } => {
            let mut intervals = Vec::with_capacity(starts.len());
            for ((&start, &duration), &presence) in starts.iter().zip(durations).zip(presences) {
                if interrupted(stop) {
                    return Ok(None);
                }
                let duration = i32::try_from(duration)
                    .map_err(|_| CpCompileError::new("optional no-overlap duration exceeds the CP representation"))?;
                intervals.push(solver.store.register_interval(map[start.0], duration, presence.map(|presence| map[presence.0])));
            }
            let should_stop = || interrupted(stop);
            if interval_constraints::no_overlap_until(solver, intervals, &should_stop).is_none() {
                return Ok(None);
            }
        }
        IntGlobalConstraint::AlternativeChannel { shared_start, starts, durations, presences } => {
            let mut intervals = Vec::with_capacity(starts.len());
            for ((&start, &duration), &presence) in starts.iter().zip(durations).zip(presences) {
                if interrupted(stop) {
                    return Ok(None);
                }
                let duration =
                    i32::try_from(duration).map_err(|_| CpCompileError::new("alternative duration exceeds the CP representation"))?;
                intervals.push(solver.store.register_interval(map[start.0], duration, Some(map[presence.0])));
            }
            let should_stop = || interrupted(stop);
            if interval_constraints::exactly_one_mode_until(solver, intervals.clone(), &should_stop).is_none()
                || interval_constraints::alternative_channel_until(solver, map[shared_start.0], intervals, &should_stop).is_none()
            {
                return Ok(None);
            }
        }
        IntGlobalConstraint::Cumulative { starts, durations, demands, capacity } => {
            if !scheduling::cumulative_interruptible(solver, &mapped!(starts), durations, demands, *capacity, stop) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity } => {
            if !scheduling::cumulative_var_interruptible(
                solver,
                &mapped!(starts),
                &mapped!(durations),
                &mapped!(demands),
                map[capacity.0],
                stop,
            ) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::BinPacking { items, sizes, capacities } => {
            if !scheduling::bin_packing_interruptible(solver, &mapped!(items), sizes, capacities, stop) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::BinLoads { items, sizes, loads } => {
            if !flatten::post_bin_loads_interruptible(solver, &mapped!(items), sizes, &mapped!(loads), stop) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::Knapsack {
            variables: ids,
            weights,
            profits,
            weight_relation,
            weight_limit,
            profit_relation,
            profit_limit,
        } => {
            if !scheduling::knapsack_interruptible(
                solver,
                scheduling::KnapsackInput {
                    variables: &mapped!(ids),
                    weights,
                    profits,
                    weight_relation: physical_relation(*weight_relation),
                    weight_limit: *weight_limit,
                    profit_relation: physical_relation(*profit_relation),
                    profit_limit: *profit_limit,
                },
                stop,
            ) {
                return Ok(None);
            }
        }
        IntGlobalConstraint::ValuePrecedence { variables: ids, values, covered } => {
            if !primitives::precedence_with_covered_interruptible(solver, &mapped!(ids), values, *covered, stop) {
                return Ok(None);
            }
        }
    }
    if interrupted(stop) {
        Ok(None)
    } else {
        Ok(Some(()))
    }
}

fn compile_expr(expr: &IntExpr, variables: &[VarId], stop: &AtomicBool) -> CompileStep<Expr> {
    if interrupted(stop) {
        return Ok(None);
    }
    let compiled = match expr {
        IntExpr::Constant(value) => Expr::Const(*value),
        IntExpr::Variable(variable) => Expr::Var(variables[variable.0]),
        IntExpr::Neg(value) => Expr::Neg(require_step!(compile_boxed_expr(value, variables, stop))),
        IntExpr::Abs(value) => Expr::Abs(require_step!(compile_boxed_expr(value, variables, stop))),
        IntExpr::Add(values) => Expr::Add(require_step!(compile_exprs(values, variables, stop))),
        IntExpr::Sub(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Sub(left, right)
        }
        IntExpr::Mul(values) => Expr::Mul(require_step!(compile_exprs(values, variables, stop))),
        IntExpr::Div(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Div(left, right)
        }
        IntExpr::Mod(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Mod(left, right)
        }
        IntExpr::Min(values) => Expr::Min(require_step!(compile_exprs(values, variables, stop))),
        IntExpr::Max(values) => Expr::Max(require_step!(compile_exprs(values, variables, stop))),
        IntExpr::Eq(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Eq(left, right)
        }
        IntExpr::Ne(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Ne(left, right)
        }
        IntExpr::Lt(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Lt(left, right)
        }
        IntExpr::Le(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Le(left, right)
        }
        IntExpr::Gt(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Gt(left, right)
        }
        IntExpr::Ge(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Ge(left, right)
        }
        IntExpr::Not(value) => Expr::Not(require_step!(compile_boxed_expr(value, variables, stop))),
        IntExpr::And(values) => Expr::And(require_step!(compile_exprs(values, variables, stop))),
        IntExpr::Or(values) => Expr::Or(require_step!(compile_exprs(values, variables, stop))),
        IntExpr::Imp(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Imp(left, right)
        }
        IntExpr::Iff(left, right) => {
            let (left, right) = require_step!(compile_binary_expr(left, right, variables, stop));
            Expr::Iff(left, right)
        }
        IntExpr::IfThenElse(condition, then_value, else_value) => Expr::IfThenElse(
            require_step!(compile_boxed_expr(condition, variables, stop)),
            require_step!(compile_boxed_expr(then_value, variables, stop)),
            require_step!(compile_boxed_expr(else_value, variables, stop)),
        ),
    };
    if interrupted(stop) {
        Ok(None)
    } else {
        Ok(Some(compiled))
    }
}

fn compile_boxed_expr(expr: &IntExpr, variables: &[VarId], stop: &AtomicBool) -> CompileStep<Box<Expr>> {
    Ok(compile_expr(expr, variables, stop)?.map(Box::new))
}

fn compile_binary_expr(left: &IntExpr, right: &IntExpr, variables: &[VarId], stop: &AtomicBool) -> CompileStep<(Box<Expr>, Box<Expr>)> {
    let left = require_step!(compile_boxed_expr(left, variables, stop));
    let right = require_step!(compile_boxed_expr(right, variables, stop));
    Ok(Some((left, right)))
}

fn compile_exprs(expressions: &[IntExpr], variables: &[VarId], stop: &AtomicBool) -> CompileStep<Vec<Expr>> {
    let mut compiled = Vec::with_capacity(expressions.len());
    for expression in expressions {
        if interrupted(stop) {
            return Ok(None);
        }
        compiled.push(require_step!(compile_expr(expression, variables, stop)));
    }
    Ok(Some(compiled))
}

fn membership_for(set: &CompiledSet, value: i32) -> Option<VarId> {
    set.values.binary_search(&value).ok().map(|index| set.membership[index])
}

fn set_values<const N: usize>(sets: &[CompiledSet], references: [SetVarRef; N], stop: &AtomicBool) -> Option<BTreeSet<i32>> {
    let mut values = BTreeSet::new();
    for reference in references {
        for &value in &sets[reference.0].values {
            if interrupted(stop) {
                return None;
            }
            values.insert(value);
        }
    }
    Some(values)
}
