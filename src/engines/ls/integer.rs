//! Semantic integer local-search compilation.
//!
//! The frontend-neutral model is lowered once to the same physical CP root as
//! exact search. This module owns only the local-search representation and its
//! lowering. The orchestrator owns budgets, candidate repair, verification,
//! events, and public solve results.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;

use crate::constraints::linear::Relation as PhysicalRelation;
use crate::constraints::table::{Dfa, Mdd, MddArc, STAR};
use crate::engines::ls::cop::{LocalRhs, LocalSearchSpec};
use crate::engines::ls::disjunctive_schedule::DisjunctiveSchedulePlan;
use crate::engines::ls::exact_cover::ExactCoverPlan;
use crate::engines::ls::scenario_schedule::ScenarioSchedulePlan;
use crate::expr::Expr;
use crate::ids::VarId;
use crate::model::{
    BoolLiteral, CompiledCp, Constraint, IntDomain, IntExpr as SemanticIntExpr, IntGlobalConstraint, IntVarRef, Model,
    Objective as SemanticObjective, Relation, SetVarRef,
};
use crate::orchestrator::SolveError;

#[derive(Clone)]
pub(crate) struct IntegerLocalSearchPlan {
    pub(crate) spec: LocalSearchSpec,
    pub(crate) warm_start: Option<IntegerWarmStartKind>,
    pub(crate) estimated_bytes: u64,
}

#[derive(Clone)]
pub(crate) enum IntegerWarmStartPlan {
    Local(IntegerLocalSearchPlan),
    ExactCover(ExactCoverPlan),
    ScenarioSchedule(ScenarioSchedulePlan),
    DisjunctiveSchedule(DisjunctiveSchedulePlan),
    Fallbacks(Vec<IntegerWarmStartPlan>),
}

#[derive(Clone, Copy)]
pub(crate) enum IntegerWarmStartKind {
    Structural,
    SignedProductSquares,
}

impl IntegerWarmStartKind {
    pub(crate) fn role(self) -> &'static str {
        match self {
            Self::Structural => "structural_local_search_warm_start",
            Self::SignedProductSquares => "signed_product_squares_warm_start",
        }
    }
}

const MAX_WARM_START_PLAN_BYTES: u64 = 64 * 1024 * 1024;
const MAX_WARM_START_INSPECTION_WORK: u64 = 2_000_000;

fn local_search_peak_bytes(plan_bytes: u64) -> u64 {
    // LocalModel owns its normalized constraint view while the reusable plan
    // remains resident, so execution temporarily needs a second plan-sized copy.
    plan_bytes.saturating_mul(2)
}

#[derive(Clone, Copy, Debug)]
struct WarmStartPreflight {
    local_plan_bytes: u64,
    structural_plan_bytes: u64,
    inspection_work: u64,
}

pub(crate) struct CompiledIntegerWarmStart {
    pub(crate) plan: Option<IntegerWarmStartPlan>,
    pub(crate) estimated_bytes: u64,
}

#[derive(Clone, Copy, Default)]
struct PlanCost {
    bytes: u128,
    work: u128,
    scope: u128,
}

impl PlanCost {
    fn add(self, other: Self) -> Self {
        Self {
            bytes: self.bytes.saturating_add(other.bytes),
            work: self.work.saturating_add(other.work),
            scope: self.scope.saturating_add(other.scope),
        }
    }
}

fn warm_start_preflight(model: &Model, stop: &AtomicBool) -> Option<WarmStartPreflight> {
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return None;
    }
    let physical_variables =
        model.int_vars().len().saturating_add(model.sets().iter().map(|set| set.possible.len()).fold(0usize, usize::saturating_add));
    let mut cost = PlanCost {
        bytes: 4_096u128.saturating_add((physical_variables as u128).saturating_mul(16)),
        work: physical_variables as u128,
        scope: physical_variables as u128,
    };
    for constraint in model.constraints() {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        cost = cost.add(constraint_plan_cost(model, constraint, stop)?);
    }
    let structural = 4_096u128
        .saturating_add((model.int_vars().len() as u128).saturating_mul(256))
        .saturating_add((model.constraints().len() as u128).saturating_mul(512))
        .saturating_add(cost.scope.saturating_mul(128));
    Some(WarmStartPreflight {
        local_plan_bytes: u64::try_from(cost.bytes).unwrap_or(u64::MAX),
        structural_plan_bytes: u64::try_from(structural).unwrap_or(u64::MAX),
        inspection_work: u64::try_from(cost.work).unwrap_or(u64::MAX),
    })
}

pub(crate) fn estimate_local_search_plan_bytes(model: &Model, stop: &AtomicBool) -> Option<u64> {
    warm_start_preflight(model, stop).map(|preflight| local_search_peak_bytes(preflight.local_plan_bytes))
}

#[cfg(test)]
pub(crate) fn audit_warm_start_preflight(model: &Model, stop: &AtomicBool) -> Option<(u64, u64, u64)> {
    let preflight = warm_start_preflight(model, stop)?;
    Some((preflight.local_plan_bytes, preflight.structural_plan_bytes, preflight.inspection_work))
}

#[cfg(test)]
pub(crate) fn audit_compile_warm_start(model: &Model, compiled: &CompiledCp, memory_allowance: u64, stop: &AtomicBool) -> (bool, u64) {
    let compiled = compile_warm_start(model, compiled, memory_allowance, stop);
    (compiled.plan.is_some(), compiled.estimated_bytes)
}

fn constraint_plan_cost(model: &Model, constraint: &Constraint, stop: &AtomicBool) -> Option<PlanCost> {
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return None;
    }
    let vector = |length: usize, element: usize| (length as u128).saturating_mul(element as u128);
    Some(match constraint {
        Constraint::Intension(expression) => expression_plan_cost(expression, stop)?,
        Constraint::Selected { constraint, .. } => {
            constraint_plan_cost(model, constraint, stop)?.add(PlanCost { bytes: 64, work: 1, scope: 1 })
        }
        Constraint::Linear { terms, .. } => {
            PlanCost { bytes: 128u128.saturating_add(vector(terms.len(), 16)), work: terms.len() as u128, scope: terms.len() as u128 }
        }
        Constraint::Clause(literals) => PlanCost {
            bytes: 128u128.saturating_add(vector(literals.len(), 192)),
            work: literals.len() as u128,
            scope: literals.len() as u128,
        },
        Constraint::IntegerGlobal(global) => global_plan_cost(global, stop)?,
        Constraint::SetSubset { subset, superset } | Constraint::SetDisjoint { left: subset, right: superset } => {
            let values = model
                .sets()
                .get(subset.0)
                .map_or(0usize, |set| set.possible.len())
                .saturating_add(model.sets().get(superset.0).map_or(0usize, |set| set.possible.len()));
            PlanCost { bytes: 256u128.saturating_add(vector(values, 192)), work: values as u128, scope: values as u128 }
        }
        Constraint::SetCardinality { set, .. } => {
            let values = model.sets().get(set.0).map_or(0usize, |set| set.possible.len());
            PlanCost { bytes: 256u128.saturating_add(vector(values, 32)), work: values as u128, scope: values as u128 }
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
        | Constraint::IntervalResource(_) => PlanCost { bytes: 256, work: 1, scope: 1 },
    })
}

fn expression_plan_cost(expression: &SemanticIntExpr, stop: &AtomicBool) -> Option<PlanCost> {
    let mut pending = vec![expression];
    let mut nodes = 0u128;
    while let Some(expression) = pending.pop() {
        nodes = nodes.saturating_add(1);
        if nodes.is_multiple_of(1_024) && stop.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        match expression {
            SemanticIntExpr::Constant(_) | SemanticIntExpr::Variable(_) => {}
            SemanticIntExpr::Neg(value) | SemanticIntExpr::Abs(value) | SemanticIntExpr::Not(value) => pending.push(value),
            SemanticIntExpr::Add(values)
            | SemanticIntExpr::Mul(values)
            | SemanticIntExpr::Min(values)
            | SemanticIntExpr::Max(values)
            | SemanticIntExpr::And(values)
            | SemanticIntExpr::Or(values) => pending.extend(values),
            SemanticIntExpr::Sub(left, right)
            | SemanticIntExpr::Div(left, right)
            | SemanticIntExpr::Mod(left, right)
            | SemanticIntExpr::Eq(left, right)
            | SemanticIntExpr::Ne(left, right)
            | SemanticIntExpr::Lt(left, right)
            | SemanticIntExpr::Le(left, right)
            | SemanticIntExpr::Gt(left, right)
            | SemanticIntExpr::Ge(left, right)
            | SemanticIntExpr::Imp(left, right)
            | SemanticIntExpr::Iff(left, right) => pending.extend([left.as_ref(), right.as_ref()]),
            SemanticIntExpr::IfThenElse(condition, then_value, else_value) => {
                pending.extend([condition.as_ref(), then_value.as_ref(), else_value.as_ref()]);
            }
        }
    }
    Some(PlanCost { bytes: nodes.saturating_mul(96), work: nodes, scope: nodes })
}

fn global_plan_cost(global: &IntGlobalConstraint, stop: &AtomicBool) -> Option<PlanCost> {
    let vector = |length: usize, element: usize| (length as u128).saturating_mul(element as u128);
    let simple = |scope: usize, payload: u128| PlanCost {
        // The semantic scope can expand into several physical vectors or one
        // constraint per adjacent/member variable. This intentionally counts
        // substantially more than the raw VarId payload.
        bytes: 256u128.saturating_add(vector(scope, 256)).saturating_add(payload),
        work: (scope as u128).saturating_add(payload / 8),
        scope: scope as u128,
    };
    Some(match global {
        IntGlobalConstraint::AllDifferent { variables, except } => simple(variables.len(), vector(except.len(), 4)),
        IntGlobalConstraint::AllEqual(variables)
        | IntGlobalConstraint::Ordered { variables, .. }
        | IntGlobalConstraint::Instantiation { variables, .. }
        | IntGlobalConstraint::Count { variables, .. }
        | IntGlobalConstraint::NValues { variables, .. }
        | IntGlobalConstraint::Circuit { successors: variables, .. } => simple(variables.len(), 0),
        IntGlobalConstraint::ValuePrecedence { variables, values, .. } => {
            simple(variables.len(), vector(values.len(), std::mem::size_of::<i32>()))
        }
        IntGlobalConstraint::Minimum { variables, .. } | IntGlobalConstraint::Maximum { variables, .. } => {
            simple(variables.len().saturating_add(1), 0)
        }
        IntGlobalConstraint::Element { array, .. } => simple(array.len().saturating_add(2), 0),
        IntGlobalConstraint::ElementConst { array, .. } => {
            let rows = array.len();
            simple(2, vector(rows, 24 + 8))
        }
        IntGlobalConstraint::Cardinality { variables, values, lower, upper, .. } => {
            simple(variables.len(), vector(values.len(), 4).saturating_add(vector(lower.len().saturating_add(upper.len()), 8)))
        }
        IntGlobalConstraint::Table { variables, tuples, .. } => {
            let cells = (variables.len() as u128).saturating_mul(tuples.len() as u128);
            PlanCost {
                bytes: 192u128
                    .saturating_add(vector(variables.len(), 16))
                    .saturating_add(vector(tuples.len(), std::mem::size_of::<Vec<i32>>()))
                    .saturating_add(cells.saturating_mul(4)),
                work: cells.saturating_add(tuples.len() as u128),
                scope: variables.len() as u128,
            }
        }
        IntGlobalConstraint::Regular { variables, automaton } => simple(
            variables.len(),
            vector(automaton.accepting.len(), std::mem::size_of::<usize>())
                .saturating_add(vector(automaton.transitions.len(), std::mem::size_of::<(usize, i32, usize)>())),
        ),
        IntGlobalConstraint::Mdd { variables, mdd } => {
            let mut arcs = 0u128;
            for layer in &mdd.layers {
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    return None;
                }
                arcs = arcs.saturating_add(layer.len() as u128);
            }
            PlanCost {
                bytes: 192u128
                    .saturating_add(vector(variables.len(), 16))
                    .saturating_add(vector(mdd.layers.len(), std::mem::size_of::<Vec<MddArc>>()))
                    .saturating_add(arcs.saturating_mul(std::mem::size_of::<MddArc>() as u128))
                    .saturating_add(vector(mdd.nodes_per_layer.len(), std::mem::size_of::<usize>())),
                work: arcs.saturating_add(mdd.layers.len() as u128),
                scope: variables.len() as u128,
            }
        }
        IntGlobalConstraint::Lex { left, right, .. } | IntGlobalConstraint::Channel { left, right } => {
            simple(left.len().saturating_add(right.len()), 0)
        }
        IntGlobalConstraint::LexChain { rows, .. } => {
            let cells = rows.iter().map(Vec::len).fold(0usize, usize::saturating_add);
            simple(cells, vector(rows.len(), std::mem::size_of::<Vec<VarId>>()))
        }
        IntGlobalConstraint::NoOverlap { starts, durations } => simple(starts.len(), vector(durations.len(), 24)),
        IntGlobalConstraint::OptionalNoOverlap { starts, durations, presences } => {
            simple(starts.len().saturating_add(presences.len()), vector(durations.len(), 40))
        }
        IntGlobalConstraint::AlternativeChannel { starts, durations, presences, .. } => {
            simple(starts.len().saturating_add(presences.len()).saturating_add(1), vector(durations.len(), 8))
        }
        IntGlobalConstraint::Cumulative { starts, durations, demands, .. } => {
            simple(starts.len(), vector(durations.len().saturating_add(demands.len()), 16))
        }
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, .. } => {
            simple(starts.len().saturating_add(durations.len()).saturating_add(demands.len()).saturating_add(1), 0)
        }
        IntGlobalConstraint::BinPacking { items, sizes, capacities } => {
            simple(items.len(), vector(sizes.len().saturating_add(capacities.len()), 16))
        }
        IntGlobalConstraint::BinLoads { items, sizes, loads } => simple(items.len().saturating_add(loads.len()), vector(sizes.len(), 16)),
        IntGlobalConstraint::Knapsack { variables, weights, profits, .. } => {
            simple(variables.len(), vector(weights.len().saturating_add(profits.len()), 16))
        }
    })
}

pub(crate) fn compile_warm_start(
    model: &Model,
    compiled: &CompiledCp,
    memory_allowance: u64,
    stop: &AtomicBool,
) -> CompiledIntegerWarmStart {
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return CompiledIntegerWarmStart { plan: None, estimated_bytes: 0 };
    }
    let Some(preflight) = warm_start_preflight(model, stop) else {
        return CompiledIntegerWarmStart { plan: None, estimated_bytes: 0 };
    };
    let allowance = memory_allowance.min(MAX_WARM_START_PLAN_BYTES);
    let mut plans = Vec::new();
    let mut estimated_bytes = 0u64;
    if let Some(plan) = ExactCoverPlan::compile(model, stop, allowance) {
        let plan_bytes = plan.estimated_bytes();
        if plan_bytes <= allowance {
            estimated_bytes = plan_bytes;
            plans.push(IntegerWarmStartPlan::ExactCover(plan));
        }
    }
    let scenario_allowance = allowance.saturating_sub(estimated_bytes);
    if preflight.structural_plan_bytes <= scenario_allowance {
        if let Some(plan) = ScenarioSchedulePlan::compile(model, stop) {
            estimated_bytes = estimated_bytes.saturating_add(preflight.structural_plan_bytes);
            plans.push(IntegerWarmStartPlan::ScenarioSchedule(plan));
        }
    }
    let local_possible = preflight.inspection_work <= MAX_WARM_START_INSPECTION_WORK
        && (semantic_sequence_prefilter(model) || LocalSearchSpec::may_have_signed_product_square_objective(compiled.problem(), stop));
    let local_allowance = allowance.saturating_sub(estimated_bytes);
    if local_possible && preflight.local_plan_bytes <= local_allowance {
        if let Ok(Some(plan)) = compile_interruptible(model, compiled, stop, local_allowance) {
            if plan.warm_start.is_some() {
                estimated_bytes = estimated_bytes.saturating_add(plan.estimated_bytes);
                plans.push(IntegerWarmStartPlan::Local(plan));
            }
        }
    }
    let structural_allowance = allowance.saturating_sub(estimated_bytes);
    if preflight.structural_plan_bytes <= structural_allowance {
        if let Some(plan) = DisjunctiveSchedulePlan::compile(model, stop) {
            estimated_bytes = estimated_bytes.saturating_add(preflight.structural_plan_bytes);
            plans.push(IntegerWarmStartPlan::DisjunctiveSchedule(plan));
        }
    }
    let plan = match plans.len() {
        0 => None,
        1 => plans.pop(),
        _ => Some(IntegerWarmStartPlan::Fallbacks(plans)),
    };
    if plan.is_none() {
        estimated_bytes = 0;
    }
    CompiledIntegerWarmStart { plan, estimated_bytes }
}

pub(crate) fn compile_interruptible(
    model: &Model,
    compiled: &CompiledCp,
    stop: &AtomicBool,
    memory_allowance: u64,
) -> Result<Option<IntegerLocalSearchPlan>, SolveError> {
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return Ok(None);
    }
    let Some(preflight) = warm_start_preflight(model, stop) else {
        return Ok(None);
    };
    let estimated_bytes = local_search_peak_bytes(preflight.local_plan_bytes);
    if estimated_bytes > memory_allowance {
        return Ok(None);
    }

    if compiled.objectives().len() > 1 {
        return Err(SolveError::InvalidRequest("integer local search currently supports at most one objective tier".to_string()));
    }

    let mut spec = LocalSearchSpec::default();
    for &variable in compiled.int_variables() {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }
        spec.add_var(variable);
    }
    for set in compiled.sets() {
        for &membership in &set.membership {
            if stop.load(std::sync::atomic::Ordering::Acquire) {
                return Ok(None);
            }
            spec.add_var(membership);
        }
    }
    for constraint in model.constraints() {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }
        if !compile_constraint(&mut spec, model, compiled, constraint, stop)? {
            return Ok(None);
        }
    }
    if spec.unsupported() > 0 {
        return Err(SolveError::Unsupported(format!(
            "integer local-search compilation rejected {} unsupported model construct(s)",
            spec.unsupported()
        )));
    }
    // This is deliberately a cheap prefilter. The LS model reconstructs exact
    // guarded sequences before using the constructor, otherwise it continues
    // as ordinary structural local search. Transfer candidates are CP-replayed.
    let warm_start = if semantic_sequence_prefilter(model) && spec.has_guarded_sequence_primitives() {
        Some(IntegerWarmStartKind::Structural)
    } else if spec.has_signed_product_square_structure_interruptible(compiled.problem(), stop) {
        Some(IntegerWarmStartKind::SignedProductSquares)
    } else {
        None
    };
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return Ok(None);
    }
    Ok(Some(IntegerLocalSearchPlan { spec, warm_start, estimated_bytes }))
}

// Cheap semantic screening only. Exact sequence reconstruction happens after
// compilation in LocalModel::guarded_sequences.
fn semantic_sequence_prefilter(model: &Model) -> bool {
    let [SemanticObjective::IntExpr { minimize, expr: objective }] = model.objectives() else {
        return false;
    };
    if semantic_direct_sequence_prefilter(model, objective, *minimize) {
        return true;
    }
    if !model.constraints().iter().any(|constraint| {
        matches!(
            constraint,
            Constraint::IntegerGlobal(IntGlobalConstraint::Table { positive: true, .. })
                | Constraint::IntegerGlobal(IntGlobalConstraint::ElementConst { .. })
        )
    }) {
        return false;
    }

    model.constraints().iter().any(|constraint| {
        let Constraint::Intension(expression) = constraint else {
            return false;
        };
        let Some((guard, counter)) = semantic_guarded_mismatch_bound(expression) else {
            return false;
        };
        model.int_vars().get(guard.0).is_some_and(semantic_binary_domain)
            && semantic_objective_rewards_guard(model, objective, *minimize, guard)
            && semantic_counter_uses_common_element_array(model, counter)
    })
}

fn semantic_direct_sequence_prefilter(model: &Model, objective: &SemanticIntExpr, minimizing: bool) -> bool {
    let mut requirements: HashMap<IntVarRef, Vec<(IntVarRef, i32)>> = HashMap::new();
    for constraint in model.constraints() {
        let guarded_value = match constraint {
            Constraint::Intension(expression) => semantic_direct_guarded_value(expression),
            Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, tuples, positive: true }) => {
                semantic_direct_guarded_table_value(variables, tuples)
            }
            _ => None,
        };
        if let Some((guard, target, value)) = guarded_value {
            requirements.entry(guard).or_default().push((target, value));
        }
    }

    requirements.into_iter().any(|(guard, mut values)| {
        values.sort_unstable();
        values.dedup();
        let has_conflicting_value = values.windows(2).any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1);
        !values.is_empty()
            && !has_conflicting_value
            && values.iter().all(|&(target, _)| target != guard)
            && model.int_vars().get(guard.0).is_some_and(semantic_binary_domain)
            && semantic_objective_rewards_guard(model, objective, minimizing, guard)
            && semantic_guarded_elements_have_positive_construction(model, &values)
    })
}

fn semantic_binary_domain(domain: &IntDomain) -> bool {
    match domain {
        IntDomain::Bool | IntDomain::Range { lo: 0, hi: 1 } => true,
        IntDomain::Set(values) => values.len() == 2 && values.contains(&0) && values.contains(&1),
        IntDomain::Range { .. } => false,
    }
}

fn semantic_direct_guarded_value(expression: &SemanticIntExpr) -> Option<(IntVarRef, IntVarRef, i32)> {
    let SemanticIntExpr::Or(parts) = expression else {
        return None;
    };
    let [left, right] = parts.as_slice() else {
        return None;
    };
    semantic_direct_guarded_parts(left, right).or_else(|| semantic_direct_guarded_parts(right, left))
}

fn semantic_direct_guarded_parts(guard: &SemanticIntExpr, target: &SemanticIntExpr) -> Option<(IntVarRef, IntVarRef, i32)> {
    let guard = semantic_zero_guard(guard)?;
    let (target, value) = semantic_eq_variable_constant(target)?;
    Some((guard, target, value))
}

fn semantic_eq_variable_constant(expression: &SemanticIntExpr) -> Option<(IntVarRef, i32)> {
    let SemanticIntExpr::Eq(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), SemanticIntExpr::Constant(value))
        | (SemanticIntExpr::Constant(value), SemanticIntExpr::Variable(variable)) => Some((*variable, i32::try_from(*value).ok()?)),
        _ => None,
    }
}

fn semantic_direct_guarded_table_value(variables: &[IntVarRef], tuples: &[Vec<i32>]) -> Option<(IntVarRef, IntVarRef, i32)> {
    let [guard, target] = variables else {
        return None;
    };
    let mut inactive_free = false;
    let mut active_value = None;
    for tuple in tuples {
        let [guard_value, target_value] = tuple.as_slice() else {
            return None;
        };
        match (*guard_value, *target_value) {
            (0, STAR) => inactive_free = true,
            (1, value) if value != STAR && active_value.is_none_or(|active| active == value) => active_value = Some(value),
            _ => return None,
        }
    }
    inactive_free.then_some((*guard, *target, active_value?))
}

fn semantic_guarded_elements_have_positive_construction(model: &Model, requirements: &[(IntVarRef, i32)]) -> bool {
    let first_target = requirements[0].0;
    let mut candidate_arrays: Vec<&[IntVarRef]> = Vec::new();
    for constraint in model.constraints() {
        let Constraint::IntegerGlobal(IntGlobalConstraint::Element { array, value, .. }) = constraint else {
            continue;
        };
        if *value == first_target && !candidate_arrays.contains(&array.as_slice()) {
            candidate_arrays.push(array);
        }
    }

    candidate_arrays.into_iter().any(|array| {
        let mut indexes = Vec::with_capacity(requirements.len());
        for &(target, _) in requirements {
            let mut matching_indexes = model
                .constraints()
                .iter()
                .filter_map(|constraint| {
                    let Constraint::IntegerGlobal(IntGlobalConstraint::Element { array: candidate_array, index, value }) = constraint
                    else {
                        return None;
                    };
                    (*value == target && candidate_array.as_slice() == array).then_some(*index)
                })
                .collect::<Vec<_>>();
            matching_indexes.sort_unstable();
            matching_indexes.dedup();
            let [index] = matching_indexes.as_slice() else {
                return false;
            };
            indexes.push(*index);
        }
        let distinct_indexes = indexes.iter().copied().collect::<HashSet<_>>();
        distinct_indexes.len() == indexes.len() && semantic_positive_tables_connect_indexes(model, &indexes)
    })
}

fn semantic_positive_tables_connect_indexes(model: &Model, indexes: &[IntVarRef]) -> bool {
    if indexes.is_empty() {
        return false;
    }
    let positions = indexes.iter().enumerate().map(|(position, &index)| (index, position)).collect::<HashMap<_, _>>();
    let mut adjacency = vec![Vec::new(); indexes.len()];
    let mut covered = vec![false; indexes.len()];
    for constraint in model.constraints() {
        let Constraint::IntegerGlobal(IntGlobalConstraint::Table { variables, tuples, positive: true }) = constraint else {
            continue;
        };
        if tuples.is_empty() {
            continue;
        }
        let mut table_positions = variables
            .iter()
            .enumerate()
            .filter_map(|(column, variable)| {
                positions
                    .get(variable)
                    .copied()
                    .filter(|_| tuples.iter().any(|tuple| tuple.get(column).is_some_and(|&value| value != STAR)))
            })
            .collect::<Vec<_>>();
        table_positions.sort_unstable();
        table_positions.dedup();
        for &position in &table_positions {
            covered[position] = true;
        }
        for &left in &table_positions {
            for &right in &table_positions {
                if left != right {
                    adjacency[left].push(right);
                }
            }
        }
    }
    if !covered.iter().all(|&value| value) {
        return false;
    }

    let mut reached = vec![false; indexes.len()];
    let mut pending = vec![0usize];
    reached[0] = true;
    while let Some(position) = pending.pop() {
        for &next in &adjacency[position] {
            if !reached[next] {
                reached[next] = true;
                pending.push(next);
            }
        }
    }
    reached.iter().all(|&value| value)
}

fn semantic_guarded_mismatch_bound(expression: &SemanticIntExpr) -> Option<(IntVarRef, IntVarRef)> {
    let SemanticIntExpr::Or(parts) = expression else {
        return None;
    };
    let [left, right] = parts.as_slice() else {
        return None;
    };
    semantic_zero_guard(left)
        .zip(semantic_nonnegative_bound(right))
        .or_else(|| semantic_zero_guard(right).zip(semantic_nonnegative_bound(left)))
}

fn semantic_zero_guard(expression: &SemanticIntExpr) -> Option<IntVarRef> {
    let SemanticIntExpr::Eq(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), SemanticIntExpr::Constant(0))
        | (SemanticIntExpr::Constant(0), SemanticIntExpr::Variable(variable)) => Some(*variable),
        _ => None,
    }
}

fn semantic_nonnegative_bound(expression: &SemanticIntExpr) -> Option<IntVarRef> {
    let SemanticIntExpr::Le(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), SemanticIntExpr::Constant(value)) if *value >= 0 => Some(*variable),
        _ => None,
    }
}

fn semantic_objective_rewards_guard(model: &Model, objective: &SemanticIntExpr, minimizing: bool, guard: IntVarRef) -> bool {
    let direct = semantic_affine_coefficient(objective, guard).filter(|&coefficient| coefficient != 0);
    let coefficient = direct.or_else(|| {
        let SemanticIntExpr::Variable(objective_variable) = objective else {
            return None;
        };
        semantic_materialized_objective_coefficient(model, *objective_variable, guard)
    });
    coefficient.is_some_and(|coefficient| if minimizing { coefficient < 0 } else { coefficient > 0 })
}

fn semantic_materialized_objective_coefficient(model: &Model, objective: IntVarRef, guard: IntVarRef) -> Option<i128> {
    let mut candidates = model.constraints().iter().filter_map(|constraint| {
        let Constraint::Linear { terms, relation: Relation::Eq, .. } = constraint else {
            return None;
        };
        let mut objective_coefficient = 0i128;
        let mut guard_coefficient = 0i128;
        for &(coefficient, variable) in terms {
            if variable == objective {
                objective_coefficient = objective_coefficient.checked_add(i128::from(coefficient))?;
            }
            if variable == guard {
                guard_coefficient = guard_coefficient.checked_add(i128::from(coefficient))?;
            }
        }
        (objective_coefficient.abs() == 1 && guard_coefficient != 0).then(|| -guard_coefficient / objective_coefficient)
    });
    let coefficient = candidates.next()?;
    candidates.all(|candidate| candidate == coefficient).then_some(coefficient)
}

fn semantic_affine_coefficient(expression: &SemanticIntExpr, target: IntVarRef) -> Option<i128> {
    match expression {
        SemanticIntExpr::Constant(_) => Some(0),
        SemanticIntExpr::Variable(variable) => Some(i128::from(*variable == target)),
        SemanticIntExpr::Neg(value) => semantic_affine_coefficient(value, target)?.checked_neg(),
        SemanticIntExpr::Add(values) => {
            values.iter().try_fold(0i128, |sum, value| sum.checked_add(semantic_affine_coefficient(value, target)?))
        }
        SemanticIntExpr::Sub(left, right) => {
            semantic_affine_coefficient(left, target)?.checked_sub(semantic_affine_coefficient(right, target)?)
        }
        SemanticIntExpr::Mul(values) => {
            let mut scale = 1i128;
            let mut nonconstant = None;
            for value in values {
                if let Some(constant) = semantic_constant_value(value) {
                    scale = scale.checked_mul(constant)?;
                } else if nonconstant.replace(value).is_some() {
                    return None;
                }
            }
            nonconstant.map_or(Some(0), |value| semantic_affine_coefficient(value, target)?.checked_mul(scale))
        }
        _ => None,
    }
}

fn semantic_constant_value(expression: &SemanticIntExpr) -> Option<i128> {
    match expression {
        SemanticIntExpr::Constant(value) => Some(i128::from(*value)),
        SemanticIntExpr::Neg(value) => semantic_constant_value(value)?.checked_neg(),
        SemanticIntExpr::Add(values) => values.iter().try_fold(0i128, |sum, value| sum.checked_add(semantic_constant_value(value)?)),
        SemanticIntExpr::Sub(left, right) => semantic_constant_value(left)?.checked_sub(semantic_constant_value(right)?),
        SemanticIntExpr::Mul(values) => {
            values.iter().try_fold(1i128, |product, value| product.checked_mul(semantic_constant_value(value)?))
        }
        _ => None,
    }
}

fn semantic_counter_uses_common_element_array(model: &Model, counter: IntVarRef) -> bool {
    model.constraints().iter().any(|constraint| {
        let Constraint::Linear { terms, relation: Relation::Eq, rhs: 0 } = constraint else {
            return false;
        };
        let mut saw_counter = false;
        let mut mismatches = Vec::new();
        for &(coefficient, variable) in terms {
            if variable == counter {
                if saw_counter || coefficient != -1 {
                    return false;
                }
                saw_counter = true;
            } else if coefficient == 1 {
                mismatches.push(variable);
            } else if coefficient != 0 {
                return false;
            }
        }
        saw_counter && !mismatches.is_empty() && semantic_mismatches_share_element_array(model, &mismatches)
    })
}

fn semantic_mismatches_share_element_array(model: &Model, mismatches: &[IntVarRef]) -> bool {
    let mut common_arrays: Option<Vec<Vec<IntVarRef>>> = None;
    for &mismatch in mismatches {
        let sources = model
            .constraints()
            .iter()
            .filter_map(|constraint| {
                let Constraint::Intension(expression) = constraint else {
                    return None;
                };
                semantic_mismatch_source(expression, mismatch)
            })
            .collect::<Vec<_>>();
        if sources.is_empty() {
            return false;
        }
        let mut arrays = model
            .constraints()
            .iter()
            .filter_map(|constraint| {
                let Constraint::IntegerGlobal(IntGlobalConstraint::Element { array, value, .. }) = constraint else {
                    return None;
                };
                sources.contains(value).then(|| array.clone())
            })
            .collect::<Vec<_>>();
        arrays.sort_unstable();
        arrays.dedup();
        if arrays.is_empty() {
            return false;
        }
        match &mut common_arrays {
            None => common_arrays = Some(arrays),
            Some(common) => common.retain(|array| arrays.binary_search(array).is_ok()),
        }
        if common_arrays.as_ref().is_none_or(Vec::is_empty) {
            return false;
        }
    }
    common_arrays.is_some_and(|arrays| !arrays.is_empty())
}

fn semantic_mismatch_source(expression: &SemanticIntExpr, mismatch: IntVarRef) -> Option<IntVarRef> {
    let SemanticIntExpr::Eq(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), value) if *variable == mismatch => semantic_ne_variable_constant(value),
        (value, SemanticIntExpr::Variable(variable)) if *variable == mismatch => semantic_ne_variable_constant(value),
        _ => None,
    }
}

fn semantic_ne_variable_constant(expression: &SemanticIntExpr) -> Option<IntVarRef> {
    let SemanticIntExpr::Ne(left, right) = expression else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (SemanticIntExpr::Variable(variable), SemanticIntExpr::Constant(_))
        | (SemanticIntExpr::Constant(_), SemanticIntExpr::Variable(variable)) => Some(*variable),
        _ => None,
    }
}

fn compile_constraint(
    spec: &mut LocalSearchSpec,
    model: &Model,
    compiled: &CompiledCp,
    constraint: &Constraint,
    stop: &AtomicBool,
) -> Result<bool, SolveError> {
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return Ok(false);
    }
    let map = compiled.int_variables();
    match constraint {
        Constraint::Intension(expression) => {
            let Some(expression) = expression_of(compiled, expression, stop)? else {
                return Ok(false);
            };
            spec.add_expr(expression);
        }
        Constraint::Selected { selector, constraint } => {
            let start = spec.begin_guarded_constraints();
            let result = compile_constraint(spec, model, compiled, constraint, stop);
            spec.finish_guarded_constraints(start, map[selector.0]);
            if !result? {
                return Ok(false);
            }
        }
        Constraint::Linear { terms, relation, rhs } => spec.add_linear(
            terms.iter().map(|(coefficient, _)| *coefficient).collect(),
            terms.iter().map(|(_, variable)| map[variable.0]).collect(),
            physical_relation(*relation),
            *rhs,
        ),
        Constraint::Clause(literals) => spec.add_expr(clause_expression(map, literals)),
        Constraint::IntegerGlobal(global) => {
            if !compile_global(spec, compiled, global, stop)? {
                return Ok(false);
            }
        }
        Constraint::SetSubset { subset, superset } => {
            for value in set_values(model, [*subset, *superset]) {
                match (membership(compiled, *subset, value), membership(compiled, *superset, value)) {
                    (Some(left), Some(right)) => spec.add_expr(Expr::Imp(
                        Box::new(Expr::Eq(Box::new(Expr::Var(left)), Box::new(Expr::Const(1)))),
                        Box::new(Expr::Eq(Box::new(Expr::Var(right)), Box::new(Expr::Const(1)))),
                    )),
                    (Some(left), None) => spec.add_linear(vec![1], vec![left], PhysicalRelation::Eq, 0),
                    (None, _) => {}
                }
            }
        }
        Constraint::SetDisjoint { left, right } => {
            for value in set_values(model, [*left, *right]) {
                if let (Some(left), Some(right)) = (membership(compiled, *left, value), membership(compiled, *right, value)) {
                    spec.add_linear(vec![1, 1], vec![left, right], PhysicalRelation::Le, 1);
                }
            }
        }
        Constraint::SetCardinality { set, min, max } => {
            let variables = compiled.sets()[set.0].membership.clone();
            spec.add_linear(vec![1; variables.len()], variables.clone(), PhysicalRelation::Ge, *min as i64);
            spec.add_linear(vec![1; variables.len()], variables, PhysicalRelation::Le, *max as i64);
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
            return Err(SolveError::Compile("integer local-search compiler received a collection or interval constraint".to_string()));
        }
    }
    Ok(!stop.load(std::sync::atomic::Ordering::Acquire))
}

fn compile_global(
    spec: &mut LocalSearchSpec,
    compiled: &CompiledCp,
    global: &IntGlobalConstraint,
    stop: &AtomicBool,
) -> Result<bool, SolveError> {
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return Ok(false);
    }
    let map = compiled.int_variables();
    let vars = |ids: &[crate::model::IntVarRef]| ids.iter().map(|variable| map[variable.0]).collect::<Vec<_>>();
    match global {
        IntGlobalConstraint::AllDifferent { variables, except } if except.is_empty() => spec.add_all_different(vars(variables)),
        IntGlobalConstraint::AllDifferent { variables, except } => spec.add_all_different_except(vars(variables), except.clone()),
        IntGlobalConstraint::AllEqual(variables) => spec.add_all_equal(vars(variables)),
        IntGlobalConstraint::Ordered { variables, relation } => {
            for pair in vars(variables).windows(2) {
                spec.add_linear(vec![1, -1], pair.to_vec(), physical_relation(*relation), 0);
            }
        }
        IntGlobalConstraint::Instantiation { variables, values } => {
            for (variable, value) in vars(variables).into_iter().zip(values) {
                spec.add_linear(vec![1], vec![variable], PhysicalRelation::Eq, i64::from(*value));
            }
        }
        IntGlobalConstraint::Minimum { target, variables } | IntGlobalConstraint::Maximum { target, variables } => {
            let values = vars(variables).into_iter().map(Expr::Var).collect();
            let extremum = if matches!(global, IntGlobalConstraint::Minimum { .. }) { Expr::Min(values) } else { Expr::Max(values) };
            spec.add_expr(Expr::Eq(Box::new(Expr::Var(map[target.0])), Box::new(extremum)));
        }
        IntGlobalConstraint::Element { array, index, value } => {
            spec.add_element(vars(array), map[index.0], map[value.0], 0);
        }
        IntGlobalConstraint::ElementConst { array, index, value } => {
            let mut tuples = Vec::with_capacity(array.len());
            for (position, value) in array.iter().enumerate() {
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    return Ok(false);
                }
                let Ok(position) = i32::try_from(position) else {
                    return Err(SolveError::Compile("element array index exceeds i32".to_string()));
                };
                tuples.push(vec![position, *value]);
            }
            spec.add_extension(vec![map[index.0], map[value.0]], tuples, true);
        }
        IntGlobalConstraint::Count { variables, value, relation, count } => {
            spec.add_count(vars(variables), vec![*value], physical_relation(*relation), LocalRhs::Const(*count));
        }
        IntGlobalConstraint::Cardinality { variables, values, lower, upper, closed } => {
            spec.add_cardinality(vars(variables), values.clone(), lower.clone(), upper.clone(), *closed);
        }
        IntGlobalConstraint::NValues { variables, relation, count } => {
            spec.add_n_values(vars(variables), physical_relation(*relation), LocalRhs::Const(*count));
        }
        IntGlobalConstraint::Table { variables, tuples, positive } => {
            let mut copied = Vec::with_capacity(tuples.len());
            let mut cells = 0usize;
            for tuple in tuples.iter() {
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    return Ok(false);
                }
                let mut row = Vec::with_capacity(tuple.len());
                for &value in tuple {
                    cells = cells.wrapping_add(1);
                    if cells.is_multiple_of(1_024) && stop.load(std::sync::atomic::Ordering::Acquire) {
                        return Ok(false);
                    }
                    row.push(value);
                }
                copied.push(row);
            }
            spec.add_extension(vars(variables), copied, *positive);
        }
        IntGlobalConstraint::Regular { variables, automaton } => {
            let mut accept = Vec::with_capacity(automaton.accepting.len());
            for &state in &automaton.accepting {
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    return Ok(false);
                }
                accept.push(state);
            }
            let mut transitions = Vec::with_capacity(automaton.transitions.len());
            for &transition in &automaton.transitions {
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    return Ok(false);
                }
                transitions.push(transition);
            }
            spec.add_regular(vars(variables), Dfa { n_states: automaton.states, start: automaton.start, accept, transitions });
        }
        IntGlobalConstraint::Mdd { variables, mdd } => {
            let mut layers = Vec::with_capacity(mdd.layers.len());
            let mut visits = 0usize;
            for layer in &mdd.layers {
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    return Ok(false);
                }
                let mut copied = Vec::with_capacity(layer.len());
                for arc in layer {
                    visits = visits.wrapping_add(1);
                    if visits.is_multiple_of(1_024) && stop.load(std::sync::atomic::Ordering::Acquire) {
                        return Ok(false);
                    }
                    copied.push(MddArc { from: arc.from, value: arc.value, to: arc.to });
                }
                layers.push(copied);
            }
            let mut nodes_per_layer = Vec::with_capacity(mdd.nodes_per_layer.len());
            for &nodes in &mdd.nodes_per_layer {
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    return Ok(false);
                }
                nodes_per_layer.push(nodes);
            }
            spec.add_mdd(vars(variables), Mdd { layers, nodes_per_layer });
        }
        IntGlobalConstraint::Lex { left, right, strict } => {
            spec.add_lex_chain(vec![vars(left), vars(right)], *strict);
        }
        IntGlobalConstraint::LexChain { rows, strict } => {
            spec.add_lex_chain(rows.iter().map(|row| vars(row)).collect(), *strict);
        }
        IntGlobalConstraint::Channel { left, right } => spec.add_channel_inverse(vars(left), 0, vars(right), 0),
        IntGlobalConstraint::Circuit { successors, .. } => spec.add_circuit(vars(successors)),
        IntGlobalConstraint::NoOverlap { starts, durations } => spec.add_no_overlap(
            vars(starts).into_iter().map(|start| vec![start]).collect(),
            durations.iter().copied().map(|duration| vec![Expr::Const(duration)]).collect(),
            false,
        ),
        IntGlobalConstraint::OptionalNoOverlap { starts, durations, presences } => spec.add_no_overlap(
            vars(starts).into_iter().map(|start| vec![start]).collect(),
            durations
                .iter()
                .zip(presences)
                .map(|(&duration, presence)| {
                    vec![presence
                        .map_or(Expr::Const(duration), |presence| Expr::Mul(vec![Expr::Const(duration), Expr::Var(map[presence.0])]))]
                })
                .collect(),
            true,
        ),
        IntGlobalConstraint::AlternativeChannel { shared_start, starts, presences, .. } => {
            let presences = vars(presences);
            spec.add_linear(vec![1; presences.len()], presences.clone(), PhysicalRelation::Eq, 1);
            for (&start, &presence) in starts.iter().zip(&presences) {
                spec.add_expr(Expr::Imp(
                    Box::new(Expr::Eq(Box::new(Expr::Var(presence)), Box::new(Expr::Const(1)))),
                    Box::new(Expr::Eq(Box::new(Expr::Var(map[shared_start.0])), Box::new(Expr::Var(map[start.0])))),
                ));
            }
        }
        IntGlobalConstraint::Cumulative { starts, durations, demands, capacity } => spec.add_cumulative_rhs(
            vars(starts),
            durations.iter().copied().map(LocalRhs::Const).collect(),
            demands.iter().copied().map(LocalRhs::Const).collect(),
            LocalRhs::Const(*capacity),
        ),
        IntGlobalConstraint::CumulativeVar { starts, durations, demands, capacity } => {
            spec.add_cumulative(vars(starts), vars(durations), vars(demands), LocalRhs::Var(map[capacity.0]))
        }
        IntGlobalConstraint::BinPacking { items, sizes, capacities } => {
            spec.add_bin_packing(vars(items), sizes.clone(), capacities.iter().copied().map(LocalRhs::Const).collect(), false)
        }
        IntGlobalConstraint::BinLoads { items, sizes, loads } => {
            spec.add_bin_packing(vars(items), sizes.clone(), vars(loads).into_iter().map(LocalRhs::Var).collect(), true)
        }
        IntGlobalConstraint::Knapsack { variables, weights, profits, weight_relation, weight_limit, profit_relation, profit_limit } => {
            let variables = vars(variables);
            spec.add_linear(weights.clone(), variables.clone(), physical_relation(*weight_relation), *weight_limit);
            spec.add_linear(profits.clone(), variables, physical_relation(*profit_relation), *profit_limit);
        }
        IntGlobalConstraint::ValuePrecedence { variables, values, covered } => {
            spec.add_precedence(vars(variables), values.clone(), *covered);
        }
    }
    Ok(!stop.load(std::sync::atomic::Ordering::Acquire))
}

fn expression_of(compiled: &CompiledCp, expression: &crate::model::IntExpr, stop: &AtomicBool) -> Result<Option<Expr>, SolveError> {
    compiled.compile_expression_interruptible(expression, stop).map_err(|error| SolveError::Compile(error.reason))
}

fn physical_relation(relation: Relation) -> PhysicalRelation {
    match relation {
        Relation::Eq => PhysicalRelation::Eq,
        Relation::Ne => PhysicalRelation::Ne,
        Relation::Le => PhysicalRelation::Le,
        Relation::Lt => PhysicalRelation::Lt,
        Relation::Ge => PhysicalRelation::Ge,
        Relation::Gt => PhysicalRelation::Gt,
    }
}

fn clause_expression(map: &[VarId], literals: &[BoolLiteral]) -> Expr {
    Expr::Or(
        literals
            .iter()
            .map(|literal| {
                let variable = Expr::Var(map[literal.variable.0]);
                if literal.positive {
                    variable
                } else {
                    Expr::Not(Box::new(variable))
                }
            })
            .collect(),
    )
}

fn membership(compiled: &CompiledCp, set: SetVarRef, value: i32) -> Option<VarId> {
    let set = &compiled.sets()[set.0];
    set.values.binary_search(&value).ok().map(|index| set.membership[index])
}

fn set_values<const N: usize>(model: &Model, sets: [SetVarRef; N]) -> std::collections::BTreeSet<i32> {
    sets.iter().flat_map(|set| model.sets()[set.0].possible.iter().copied()).collect()
}
