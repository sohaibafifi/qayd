//! Lower the canonical semantic model into domain-specific physical IRs.
//!
//! Capability recognition and lowering happen together. A successful return is
//! therefore executable input for the collection engines, rather than a promise
//! that a second parser might accept the model later.

use super::{list, Constraint, Model, Objective, PartitionCoverage};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
type ItemSumBound = (usize, Vec<(i32, i64)>, i64, i64);

/// Why a semantic model cannot be lowered to the collection physical IR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CollectionCompileError {
    pub(crate) reason: String,
}

impl CollectionCompileError {
    fn new(reason: impl Into<String>) -> Self {
        Self { reason: reason.into() }
    }
}

impl std::fmt::Display for CollectionCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for CollectionCompileError {}

/// Compiled collection-domain representation.
///
/// The newtype prevents frontends from depending on the physical layout.
#[derive(Clone)]
pub(crate) struct CompiledCollection {
    inner: list::CollectionModel,
}

impl CompiledCollection {
    /// Compile lists or intervals from the semantic model.
    #[cfg(test)]
    pub(crate) fn compile(model: &Model) -> Result<Self, CollectionCompileError> {
        let stop = AtomicBool::new(false);
        Self::compile_interruptible(model, &stop)?.ok_or_else(|| CollectionCompileError::new("collection compilation was interrupted"))
    }

    /// Compile and validate while sharing the caller's solve budget. `Ok(None)`
    /// means cancellation fired during physical validation.
    pub(crate) fn compile_interruptible(model: &Model, stop: &AtomicBool) -> Result<Option<Self>, CollectionCompileError> {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(None);
        }
        let inner = if !model.intervals.is_empty() { compile_schedule(model, stop)? } else { compile_lists(model, stop)? };
        let Some(inner) = inner else {
            return Ok(None);
        };
        if !inner.validate_interruptible(stop).map_err(CollectionCompileError::new)? {
            return Ok(None);
        }
        Ok(Some(Self { inner }))
    }

    pub(crate) fn as_model(&self) -> &list::CollectionModel {
        &self.inner
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CollectionMemoryEstimates {
    pub(crate) exact: u64,
    pub(crate) local_search: u64,
}

/// Estimate the resident physical state directly from the semantic model.
/// This deliberately runs before collection lowering so a request that cannot
/// fit never materializes a second copy of a large list or schedule model.
pub(crate) fn estimated_semantic_collection_bytes_interruptible(model: &Model, stop: &AtomicBool) -> Option<CollectionMemoryEstimates> {
    if stop.load(Ordering::Acquire) {
        return None;
    }
    let estimates = if !model.intervals.is_empty() {
        estimated_semantic_schedule_bytes(model, stop)?
    } else {
        estimated_semantic_list_bytes(model, stop)?
    };
    (!stop.load(Ordering::Acquire)).then_some(estimates)
}

fn estimated_semantic_schedule_bytes(model: &Model, stop: &AtomicBool) -> Option<CollectionMemoryEstimates> {
    let intervals = model.intervals.len() as u128;
    let mut modes = 0u128;
    for interval in &model.intervals {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        modes = modes.saturating_add(interval.modes.len().max(1) as u128);
    }
    let mut exact = (64u128 * 1024).saturating_add(intervals.saturating_mul(4 * 1024)).saturating_add(modes.saturating_mul(6 * 1024));
    let mut local_search = (64u128 * 1024).saturating_add(intervals.saturating_mul(1024)).saturating_add(modes.saturating_mul(512));
    let mut precedences = 0u128;
    let mut resources = 0u128;
    for constraint in &model.constraints {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        match constraint {
            Constraint::IntervalPrecedence { .. } => precedences = precedences.saturating_add(1),
            Constraint::IntervalResource(resource) => {
                resources = resources.saturating_add(1);
                match resource {
                    list::Resource::NoOverlap(group) => {
                        let size = group.len() as u128;
                        exact = exact.saturating_add(size.saturating_mul(size.saturating_sub(1)) / 2 * (8 * 1024));
                    }
                    list::Resource::MachineNoOverlap => {
                        let mut by_machine = std::collections::HashMap::<usize, u128>::new();
                        for interval in &model.intervals {
                            for reference in &interval.modes {
                                if stop.load(Ordering::Acquire) {
                                    return None;
                                }
                                let mode = model.interval_modes.get(reference.0)?;
                                *by_machine.entry(mode.machine).or_default() += 1;
                            }
                        }
                        for size in by_machine.into_values() {
                            exact = exact.saturating_add(size.saturating_mul(size.saturating_sub(1)) / 2 * (8 * 1024));
                        }
                    }
                    list::Resource::Cumulative { demands, .. } => {
                        exact = exact.saturating_add((demands.len() as u128).saturating_mul(4 * 1024));
                    }
                }
            }
            _ => {}
        }
    }
    exact = exact.saturating_add(precedences.saturating_mul(2 * 1024));
    local_search = local_search.saturating_add(precedences.saturating_mul(128)).saturating_add(resources.saturating_mul(512));
    Some(CollectionMemoryEstimates {
        exact: u64::try_from(exact).unwrap_or(u64::MAX),
        local_search: u64::try_from(local_search).unwrap_or(u64::MAX),
    })
}

fn estimated_semantic_list_bytes(model: &Model, stop: &AtomicBool) -> Option<CollectionMemoryEstimates> {
    let items = model.lists.first().map_or(0u128, |list| list.universe.len() as u128);
    let lists = model.lists.len().max(1) as u128;
    let mut reductions = 0u128;
    let mut synthetic_reductions = 0u128;
    for constraint in &model.constraints {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        if matches!(constraint, Constraint::ListLength { .. } | Constraint::ListItemSum { .. }) {
            synthetic_reductions = synthetic_reductions.saturating_add(1);
        }
    }
    let mut routing_nodes = None;
    let mut visit_reduction = |reduction: &list::Reduction| -> Option<()> {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        reductions = reductions.saturating_add(1);
        if routing_nodes.is_none() {
            routing_nodes = reduction.arena.exprs.iter().find_map(|expression| match expression {
                list::Expr::Matrix(matrix, _, _) => Some(matrix.len() as u128),
                _ => None,
            });
        }
        Some(())
    };
    for objective in &model.objectives {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        if let Objective::ListTerms { terms, max_terms, .. } = objective {
            for reduction in terms {
                visit_reduction(reduction)?;
            }
            for reduction in
                max_terms.iter().flat_map(|terms| terms.iter()).flat_map(|term| term.groups.iter()).flat_map(|group| group.iter())
            {
                visit_reduction(reduction)?;
            }
        }
    }
    for constraint in &model.constraints {
        if stop.load(Ordering::Acquire) {
            return None;
        }
        if let Constraint::ListReduction(constraint) = constraint {
            visit_reduction(&constraint.reduction)?;
        }
    }
    reductions = reductions.saturating_add(synthetic_reductions);
    let reductions = reductions.max(1);
    let exact = (64u128 * 1024)
        .saturating_add(items.saturating_mul(lists).saturating_mul(128))
        .saturating_add(items.saturating_mul(reductions).saturating_mul(256));
    let mut local_search = (64u128 * 1024)
        .saturating_add(items.saturating_mul(256))
        .saturating_add(lists.saturating_mul(4 * 1024))
        .saturating_add(items.saturating_mul(reductions).saturating_mul(64))
        .saturating_add(items.saturating_mul(items).saturating_mul(2));
    if let Some(nodes) = routing_nodes {
        let cells = nodes.saturating_mul(nodes);
        local_search = local_search.saturating_add(cells.saturating_mul(24)).saturating_add(nodes.saturating_mul(64).saturating_mul(16));
    }
    Some(CollectionMemoryEstimates {
        exact: u64::try_from(exact).unwrap_or(u64::MAX),
        local_search: u64::try_from(local_search).unwrap_or(u64::MAX),
    })
}

/// Conservative allocation estimate for an exact lowering. This is a plan
/// cost, not a backend selector or a live RSS prediction.
pub(crate) fn estimated_exact_backend_bytes(model: &list::CollectionModel) -> u64 {
    let mut bytes = 64u128 * 1024;
    if let Some(schedule) = &model.schedule {
        let intervals = schedule.intervals.len() as u128;
        let modes = schedule.intervals.iter().map(|interval| interval.modes.len().max(1) as u128).sum::<u128>();
        bytes = bytes
            .saturating_add(intervals.saturating_mul(4 * 1024))
            .saturating_add(modes.saturating_mul(6 * 1024))
            .saturating_add((schedule.precedences.len() as u128).saturating_mul(2 * 1024));
        for resource in &schedule.resources {
            match resource {
                list::Resource::NoOverlap(group) => {
                    let size = group.len() as u128;
                    bytes = bytes.saturating_add(size.saturating_mul(size.saturating_sub(1)) / 2 * (8 * 1024));
                }
                list::Resource::MachineNoOverlap => {
                    let mut by_machine = std::collections::HashMap::<usize, u128>::new();
                    for interval in &schedule.intervals {
                        for mode in &interval.modes {
                            *by_machine.entry(mode.machine).or_default() += 1;
                        }
                    }
                    for size in by_machine.into_values() {
                        bytes = bytes.saturating_add(size.saturating_mul(size.saturating_sub(1)) / 2 * (8 * 1024));
                    }
                }
                list::Resource::Cumulative { demands, .. } => {
                    bytes = bytes.saturating_add((demands.len() as u128).saturating_mul(4 * 1024));
                }
            }
        }
    } else {
        let cells = (model.items.len() as u128).saturating_mul(model.lists.max(1) as u128);
        let reductions =
            model.objectives.iter().flat_map(list::ObjectiveTier::reductions).count().saturating_add(model.constraints.len()).max(1)
                as u128;
        bytes = bytes
            .saturating_add(cells.saturating_mul(128))
            .saturating_add((model.items.len() as u128).saturating_mul(reductions).saturating_mul(256));
    }
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

/// Conservative resident-memory estimate for one collection local-search
/// trajectory, including its mutable solution and incremental evaluator state.
pub(crate) fn estimated_local_search_backend_bytes(model: &list::CollectionModel) -> u64 {
    let mut bytes = 64u128 * 1024;
    if let Some(schedule) = &model.schedule {
        let intervals = schedule.intervals.len() as u128;
        let modes = schedule.intervals.iter().map(|interval| interval.modes.len().max(1) as u128).sum::<u128>();
        bytes = bytes
            .saturating_add(intervals.saturating_mul(1024))
            .saturating_add(modes.saturating_mul(512))
            .saturating_add((schedule.precedences.len() as u128).saturating_mul(128))
            .saturating_add((schedule.resources.len() as u128).saturating_mul(512));
    } else {
        let items = model.items.len() as u128;
        let lists = model.lists.max(1) as u128;
        let reductions =
            model.objectives.iter().flat_map(list::ObjectiveTier::reductions).count().saturating_add(model.constraints.len()).max(1)
                as u128;
        bytes = bytes
            .saturating_add(items.saturating_mul(256))
            .saturating_add(lists.saturating_mul(4 * 1024))
            .saturating_add(items.saturating_mul(reductions).saturating_mul(64))
            .saturating_add(items.saturating_mul(items).saturating_mul(2));
        if let Some(nodes) = routing_matrix_size(model) {
            let cells = nodes.saturating_mul(nodes);
            // Per-worker GLS penalties, the routing relaxation cost matrix,
            // two conflict matrices used by certified bounds, and candidate
            // neighbour storage. The estimate intentionally rounds upward.
            bytes = bytes.saturating_add(cells.saturating_mul(24)).saturating_add(nodes.saturating_mul(64).saturating_mul(16));
        }
    }
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

fn routing_matrix_size(model: &list::CollectionModel) -> Option<u128> {
    model
        .objectives
        .iter()
        .flat_map(list::ObjectiveTier::reductions)
        .chain(model.constraints.iter().map(|constraint| &constraint.reduction))
        .find_map(|reduction| match reduction.arena.exprs.get(reduction.body.0 as usize) {
            Some(list::Expr::Matrix(matrix, _, _)) => Some(matrix.len() as u128),
            _ => None,
        })
}

fn compile_schedule(model: &Model, stop: &AtomicBool) -> Result<Option<list::CollectionModel>, CollectionCompileError> {
    if stopped(stop) {
        return Ok(None);
    }
    if !model.int_vars.is_empty() || !model.lists.is_empty() {
        return Err(CollectionCompileError::new("collection scheduling cannot mix integer, list, and interval variables"));
    }

    let mut precedences = Vec::new();
    let mut resources = Vec::new();
    for constraint in &model.constraints {
        if stopped(stop) {
            return Ok(None);
        }
        match constraint {
            Constraint::IntervalPrecedence { before, after } => {
                check_interval_ref(model, before.0)?;
                check_interval_ref(model, after.0)?;
                precedences.push((before.0, after.0));
            }
            Constraint::IntervalResource(resource) => resources.push(resource.clone()),
            _ => return Err(CollectionCompileError::new("non-scheduling constraint in an interval model")),
        }
    }

    if stopped(stop) {
        return Ok(None);
    }
    let minimize_makespan = match model.objectives.as_slice() {
        [] => false,
        [Objective::Makespan { minimize: true, intervals }] => {
            if intervals.len() != model.intervals.len() {
                return Err(CollectionCompileError::new(
                    "collection scheduling requires the makespan objective to cover every interval in declaration order",
                ));
            }
            for (index, interval) in intervals.iter().enumerate() {
                if stopped(stop) {
                    return Ok(None);
                }
                if interval.0 >= model.intervals.len() {
                    return Err(CollectionCompileError::new("makespan objective references an unknown interval"));
                }
                if interval.0 != index {
                    return Err(CollectionCompileError::new(
                        "collection scheduling requires the makespan objective to cover every interval in declaration order",
                    ));
                }
            }
            true
        }
        [_] => return Err(CollectionCompileError::new("collection scheduling supports only makespan minimization")),
        _ => return Err(CollectionCompileError::new("collection scheduling supports a single objective tier")),
    };

    let mut intervals = Vec::with_capacity(model.intervals.len());
    for interval in &model.intervals {
        if stopped(stop) {
            return Ok(None);
        }
        let mut modes = Vec::with_capacity(interval.modes.len());
        for reference in &interval.modes {
            if stopped(stop) {
                return Ok(None);
            }
            let mode = model
                .interval_modes
                .get(reference.0)
                .ok_or_else(|| CollectionCompileError::new("interval references an unknown execution mode"))?;
            let start_window = mode.start_window.unwrap_or((interval.start_min, interval.start_max));
            modes.push((*reference, mode, start_window));
        }
        let horizon = if modes.is_empty() {
            if interval.start_min != 0 {
                return Err(CollectionCompileError::new("fixed collection intervals currently require start_min = 0"));
            }
            interval
                .start_max
                .checked_add(interval.duration)
                .ok_or_else(|| CollectionCompileError::new("interval horizon overflows i64"))?
        } else {
            let mut horizon = None;
            for (_, mode, (_, start_max)) in &modes {
                if stopped(stop) {
                    return Ok(None);
                }
                let end = start_max
                    .checked_add(mode.duration)
                    .ok_or_else(|| CollectionCompileError::new("interval mode horizon overflows i64"))?;
                horizon = Some(horizon.map_or(end, |current: i64| current.max(end)));
            }
            horizon.ok_or_else(|| CollectionCompileError::new("mode schedule operation has no eligible mode"))?
        };
        let mut physical_modes = Vec::with_capacity(modes.len());
        for (reference, mode, start_window) in modes {
            if stopped(stop) {
                return Ok(None);
            }
            physical_modes.push(list::Mode { reference: Some(reference.0), machine: mode.machine, duration: mode.duration, start_window });
        }
        intervals.push(list::IntervalVar { duration: interval.duration, horizon, modes: physical_modes, optional: interval.optional });
    }

    Ok(Some(list::CollectionModel {
        items: Vec::new(),
        lists: 0,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: Some(list::Schedule { intervals, precedences, resources, minimize_makespan }),
    }))
}

fn check_interval_ref(model: &Model, index: usize) -> Result<(), CollectionCompileError> {
    if index < model.intervals.len() {
        Ok(())
    } else {
        Err(CollectionCompileError::new("constraint references an unknown interval"))
    }
}

fn compile_lists(model: &Model, stop: &AtomicBool) -> Result<Option<list::CollectionModel>, CollectionCompileError> {
    if stopped(stop) {
        return Ok(None);
    }
    if !model.int_vars.is_empty() {
        return Err(CollectionCompileError::new("collection lowering does not accept integer variables"));
    }
    if model.lists.is_empty() {
        return Err(CollectionCompileError::new("model has neither list nor interval variables"));
    }
    let items = model.lists[0].universe.clone();
    for list in &model.lists {
        if stopped(stop) {
            return Ok(None);
        }
        if list.universe != items {
            return Err(CollectionCompileError::new("collection lists must share one universe"));
        }
    }

    let mut constraints = Vec::new();
    let mut globals = Vec::new();
    let mut saw_partition = false;
    for constraint in &model.constraints {
        if stopped(stop) {
            return Ok(None);
        }
        match constraint {
            Constraint::ListPartition { lists, items: partition_items } => {
                let expected = (0..model.lists.len()).collect::<Vec<_>>();
                let actual = lists.iter().map(|list| list.0).collect::<Vec<_>>();
                if saw_partition || actual != expected || partition_items != &items {
                    return Err(CollectionCompileError::new("list partition must name every list once and use the shared universe"));
                }
                saw_partition = true;
            }
            Constraint::ListPartitionWithCoverage { lists, items: partition_items, coverage: PartitionCoverage::Exact } => {
                let expected = (0..model.lists.len()).collect::<Vec<_>>();
                let actual = lists.iter().map(|list| list.0).collect::<Vec<_>>();
                if saw_partition || actual != expected || partition_items != &items {
                    return Err(CollectionCompileError::new("list partition must name every list once and use the shared universe"));
                }
                saw_partition = true;
            }
            Constraint::ListPartitionWithCoverage { coverage: PartitionCoverage::Partial, .. } => {
                return Err(CollectionCompileError::new(
                    "partial list coverage requires an explicit hidden remainder list before collection lowering",
                ));
            }
            Constraint::SameList { lists, a, b } => {
                check_list_scope(model, lists)?;
                globals.push(list::GlobalConstraint::SameList { a: *a, b: *b });
            }
            Constraint::ItemPrecedence { lists, before, after } => {
                check_list_scope(model, lists)?;
                globals.push(list::GlobalConstraint::ListLe { before: *before, after: *after });
            }
            Constraint::CollectionGlobal(global) => globals.push(global.clone()),
            Constraint::ListLength { list: list_ref, min, max } => {
                check_list_ref(model, list_ref.0)?;
                constraints.extend(length_constraints(list_ref.0, *min, *max)?);
            }
            Constraint::ListItemSum { list: list_ref, weights, min, max } => {
                check_list_ref(model, list_ref.0)?;
                let Some(lowered) = item_sum_constraints(list_ref.0, weights, *min, *max, stop) else {
                    return Ok(None);
                };
                constraints.extend(lowered);
            }
            Constraint::ListReduction(constraint) => constraints.push(constraint.clone()),
            Constraint::IntervalPrecedence { .. }
            | Constraint::IntervalAlternative { .. }
            | Constraint::IntervalEndpointRelation { .. }
            | Constraint::IntervalResource(_)
            | Constraint::Selected { .. }
            | Constraint::Intension(_)
            | Constraint::Linear { .. }
            | Constraint::Clause(_)
            | Constraint::IntegerGlobal(_)
            | Constraint::SetSubset { .. }
            | Constraint::SetDisjoint { .. }
            | Constraint::SetCardinality { .. } => {
                return Err(CollectionCompileError::new("non-list constraint in a list collection model"));
            }
        }
    }
    if !saw_partition {
        return Err(CollectionCompileError::new("collection list model requires one explicit ListPartition"));
    }

    let mut objectives = Vec::with_capacity(model.objectives.len());
    for objective in &model.objectives {
        if stopped(stop) {
            return Ok(None);
        }
        objectives.push(match objective {
            Objective::ListTerms { minimize, terms, max_terms } => {
                Ok(list::ObjectiveTier { minimize: *minimize, terms: terms.clone(), max_terms: max_terms.clone() })
            }
            Objective::IntExpr { .. } | Objective::Makespan { .. } => {
                Err(CollectionCompileError::new("non-list objective in a list collection model"))
            }
        }?);
    }

    Ok(Some(list::CollectionModel { items, lists: model.lists.len(), objectives, constraints, globals, schedule: None }))
}

fn stopped(stop: &AtomicBool) -> bool {
    stop.load(std::sync::atomic::Ordering::Acquire)
}

fn check_list_scope(model: &Model, lists: &[super::ListVarRef]) -> Result<(), CollectionCompileError> {
    let expected = (0..model.lists.len()).collect::<Vec<_>>();
    if lists.iter().map(|list| list.0).eq(expected) {
        Ok(())
    } else {
        Err(CollectionCompileError::new("cross-list constraint must use the complete collection list scope"))
    }
}

fn check_list_ref(model: &Model, index: usize) -> Result<(), CollectionCompileError> {
    if index < model.lists.len() {
        Ok(())
    } else {
        Err(CollectionCompileError::new("constraint references an unknown list"))
    }
}

fn length_constraints(list_index: usize, min: usize, max: usize) -> Result<Vec<list::Constraint>, CollectionCompileError> {
    if min > max {
        return Err(CollectionCompileError::new("list length lower bound exceeds its upper bound"));
    }
    let make = |op, rhs| {
        let mut arena = list::ExprArena::default();
        let one = arena.constant(1);
        list::Constraint {
            reduction: list::Reduction {
                op: list::ReduceOp::Count,
                iterable: list::Iterable::Items(list_index),
                arena,
                body: one,
                coeff: 1,
            },
            op,
            rhs,
        }
    };
    let min = i64::try_from(min).map_err(|_| CollectionCompileError::new("list length bound exceeds i64"))?;
    let max = i64::try_from(max).map_err(|_| CollectionCompileError::new("list length bound exceeds i64"))?;
    if min == max {
        return Ok(vec![make(list::Op::Eq, min)]);
    }
    Ok(vec![make(list::Op::Ge, min), make(list::Op::Le, max)])
}

fn item_weight_body(weights: &[(i32, i64)], stop: &AtomicBool) -> Option<(list::ExprArena, list::ExprId)> {
    let mut arena = list::ExprArena::default();
    let item = arena.arg(0);
    let mut body = arena.constant(0);
    for &(value, weight) in weights.iter().rev() {
        if stopped(stop) {
            return None;
        }
        let key = arena.constant(i64::from(value));
        let condition = list::ExprArena::eq(&mut arena, item, key);
        let weight = arena.constant(weight);
        body = arena.if_then_else(condition, weight, body);
    }
    Some((arena, body))
}

fn item_sum_constraints(list_index: usize, weights: &[(i32, i64)], min: i64, max: i64, stop: &AtomicBool) -> Option<Vec<list::Constraint>> {
    let make = |op, rhs| {
        let (arena, body) = item_weight_body(weights, stop)?;
        Some(list::Constraint {
            reduction: list::Reduction { op: list::ReduceOp::Sum, iterable: list::Iterable::Items(list_index), arena, body, coeff: 1 },
            op,
            rhs,
        })
    };
    if min == max {
        Some(vec![make(list::Op::Eq, min)?])
    } else {
        let mut result = Vec::with_capacity(2);
        if min != i64::MIN {
            result.push(make(list::Op::Ge, min)?);
        }
        if max != i64::MAX {
            result.push(make(list::Op::Le, max)?);
        }
        Some(result)
    }
}

#[cfg(test)]
pub(crate) fn length_bound_from_collection_constraint(constraint: &list::Constraint, item_count: usize) -> Option<(usize, usize, usize)> {
    let list::Iterable::Items(list) = &constraint.reduction.iterable else {
        return None;
    };
    if matches!(constraint.reduction.op, list::ReduceOp::Used) {
        if constraint.reduction.coeff != 1 {
            return None;
        }
        return used_bound_as_length(*list, constraint.op, constraint.rhs, item_count);
    }
    if !matches!(constraint.reduction.op, list::ReduceOp::Count) || constraint.reduction.coeff != 1 {
        return None;
    }
    if !matches!(
        constraint.reduction.arena.exprs.get(constraint.reduction.body.0 as usize),
        Some(list::Expr::Const(value)) if *value != 0
    ) {
        return None;
    }

    let item_count_i64 = i64::try_from(item_count).unwrap_or(i64::MAX);
    let (min, max) = match constraint.op {
        list::Op::Le => (0, constraint.rhs),
        list::Op::Ge => (constraint.rhs, item_count_i64),
        list::Op::Eq => (constraint.rhs, constraint.rhs),
    };
    if max < 0 || min > item_count_i64 || min > max {
        return Some((*list, 1, 0));
    }
    Some((*list, min.max(0) as usize, max.min(item_count_i64) as usize))
}

#[cfg(test)]
fn used_bound_as_length(list: usize, op: list::Op, rhs: i64, item_count: usize) -> Option<(usize, usize, usize)> {
    let (min, max) = match op {
        list::Op::Le if rhs < 0 => return Some((list, 1, 0)),
        list::Op::Le if rhs == 0 => (0, 0),
        list::Op::Le => (0, item_count),
        list::Op::Ge if rhs <= 0 => (0, item_count),
        list::Op::Ge if rhs == 1 => (1, item_count),
        list::Op::Ge => return Some((list, 1, 0)),
        list::Op::Eq if rhs == 0 => (0, 0),
        list::Op::Eq if rhs == 1 => (1, item_count),
        list::Op::Eq => return Some((list, 1, 0)),
    };
    Some((list, min, max))
}

#[cfg(test)]
pub(crate) fn item_sum_bound_from_collection_constraint(constraint: &list::Constraint, items: &[i32]) -> Option<ItemSumBound> {
    if !matches!(constraint.reduction.op, list::ReduceOp::Sum) {
        return None;
    }
    let list::Iterable::Items(list) = &constraint.reduction.iterable else {
        return None;
    };
    let mut weights = Vec::with_capacity(items.len());
    for &item in items {
        let weight = list::eval_expr_checked(&constraint.reduction.arena.exprs, constraint.reduction.body, &[i64::from(item)])?
            .saturating_mul(constraint.reduction.coeff);
        weights.push((item, weight));
    }
    let (min, max) = match constraint.op {
        list::Op::Le => (i64::MIN, constraint.rhs),
        list::Op::Ge => (constraint.rhs, i64::MAX),
        list::Op::Eq => (constraint.rhs, constraint.rhs),
    };
    Some((*list, weights, min, max))
}
