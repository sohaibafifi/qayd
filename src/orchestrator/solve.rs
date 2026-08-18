//! Canonical frontend-neutral solve entry point.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use crate::model::{IndependentComponent, IndependentDecomposition, IndependentFamily, Model, ModelObject, ModelPackage};

use super::{
    compile_collection_plan, compile_cp_plan_validated, compile_sat_plan, preflight_collection_memory, solve_collection_plan_with_stop,
    solve_cp_plan_validated, solve_sat_plan, Assignment, Bound, CandidateSolution, CollectionSolvePlan, CpSolvePlan, DecompositionMerge,
    EngineKind, EnginePlan, EventControl, EventSink, ExecutablePlan, IgnoreEvents, ProofClaim, ProofRequest, ProvenConclusion,
    SatSolvePlan, SolveBudget, SolveError, SolveEvent, SolveRequest, SolveResult, SolveStatus, TerminationReason, VerificationLevel,
    WorkerAllocation,
};

#[cfg(test)]
thread_local! {
    static INTERRUPT_NEXT_TRANSFER_REPLAY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn audit_interrupt_next_transfer_replay() {
    INTERRUPT_NEXT_TRANSFER_REPLAY.set(true);
}

#[cfg(test)]
fn apply_transfer_replay_audit_hook(budget: &SolveBudget) {
    if INTERRUPT_NEXT_TRANSFER_REPLAY.replace(false) {
        budget.cancel();
    }
}

#[cfg(not(test))]
fn apply_transfer_replay_audit_hook(_budget: &SolveBudget) {}

#[derive(Clone)]
enum PreparedEngine {
    Cp(Box<CpSolvePlan>),
    Collection(Box<CollectionSolvePlan>),
    Sat(Box<SatSolvePlan>),
}

#[derive(Clone)]
struct PreparedComponent {
    family: IndependentFamily,
    package: ModelPackage,
    plan: Box<PreparedSolvePlan>,
    integers: Vec<(usize, usize)>,
    sets: Vec<(usize, usize)>,
    lists: Vec<(usize, usize)>,
    intervals: Vec<(usize, usize)>,
    /// Component-local mode index to original semantic mode index.
    interval_mode_originals: Vec<usize>,
    objective_tiers: Vec<usize>,
}

#[derive(Clone)]
struct PreparedStage {
    engine: PreparedEngine,
    allocation: WorkerAllocation,
}

impl PreparedStage {
    fn single(engine: PreparedEngine) -> Self {
        Self { engine, allocation: WorkerAllocation::single() }
    }

    fn allocated(engine: PreparedEngine, workers: usize) -> Self {
        Self { engine, allocation: WorkerAllocation::portfolio(workers) }
    }

    fn estimated_backend_bytes(&self) -> u64 {
        estimated_engine_bytes(&self.engine).saturating_mul(u64::try_from(self.allocation.workers()).unwrap_or(u64::MAX))
    }

    fn description(&self) -> ExecutablePlan {
        let engine = EnginePlan::new(engine_kind(&self.engine));
        if self.allocation.workers() == 1 {
            ExecutablePlan::Single(engine)
        } else {
            ExecutablePlan::Portfolio((0..self.allocation.workers()).map(|_| ExecutablePlan::Single(engine)).collect())
        }
    }
}

#[derive(Clone)]
enum PreparedNode {
    Single(Box<PreparedEngine>),
    /// Concrete engine stages executed in order by the orchestrator. Routing's
    /// first stage is LS and its second stage is the exact routing backend.
    Sequential(Vec<PreparedStage>),
    /// The prepared node owns the allocation. The physical engine receives it
    /// explicitly and only owns its algorithm-specific cooperation state.
    Portfolio {
        workers: usize,
        engine: Box<PreparedEngine>,
    },
    Decomposed(Vec<PreparedComponent>),
}

/// A plan is fully compiled before execution. Selection never promises a
/// capability that a second parser can later reject.
#[derive(Clone)]
pub struct PreparedSolvePlan {
    node: PreparedNode,
    request: SolveRequest,
}

impl PreparedSolvePlan {
    pub fn estimated_backend_bytes(&self) -> u64 {
        match &self.node {
            PreparedNode::Single(engine) => estimated_engine_bytes(engine),
            // Every compiled stage remains resident while the sequence runs.
            // Summing is conservative and, unlike `max`, includes the exact
            // backend retained during a warm-start stage.
            PreparedNode::Sequential(stages) => stages.iter().map(PreparedStage::estimated_backend_bytes).fold(0, u64::saturating_add),
            PreparedNode::Portfolio { workers, engine } => {
                estimated_engine_bytes(engine).saturating_mul(u64::try_from(*workers).unwrap_or(u64::MAX))
            }
            PreparedNode::Decomposed(components) => {
                components.iter().map(|component| component.plan.estimated_backend_bytes()).fold(0, u64::saturating_add)
            }
        }
    }

    pub fn description(&self) -> ExecutablePlan {
        match &self.node {
            PreparedNode::Single(engine) => ExecutablePlan::Single(EnginePlan::new(engine_kind(engine))),
            PreparedNode::Sequential(stages) => ExecutablePlan::Sequential(stages.iter().map(PreparedStage::description).collect()),
            PreparedNode::Portfolio { workers, engine } => {
                ExecutablePlan::Portfolio((0..*workers).map(|_| ExecutablePlan::Single(EnginePlan::new(engine_kind(engine)))).collect())
            }
            PreparedNode::Decomposed(components) => ExecutablePlan::Decomposed {
                components: components.iter().map(|component| component.plan.description()).collect(),
                merge: DecompositionMerge::Disjoint,
            },
        }
    }

    #[doc(hidden)]
    pub fn audit_component_metadata(&self) -> Vec<crate::model::ModelMetadata> {
        match &self.node {
            PreparedNode::Decomposed(components) => components.iter().map(|component| component.package.metadata.clone()).collect(),
            PreparedNode::Single(_) | PreparedNode::Sequential(_) | PreparedNode::Portfolio { .. } => Vec::new(),
        }
    }
}

fn engine_kind(engine: &PreparedEngine) -> EngineKind {
    match engine {
        PreparedEngine::Cp(plan) => plan.engine(),
        PreparedEngine::Collection(plan) => plan.engine(),
        PreparedEngine::Sat(plan) => plan.engine(),
    }
}

fn estimated_engine_bytes(engine: &PreparedEngine) -> u64 {
    match engine {
        PreparedEngine::Cp(plan) => plan.estimated_backend_bytes(),
        PreparedEngine::Collection(plan) => plan.estimated_backend_bytes(),
        PreparedEngine::Sat(plan) => plan.estimated_backend_bytes(),
    }
}

fn collection_node(plan: CollectionSolvePlan, threads: usize) -> PreparedNode {
    if let Some(warm_start) = plan.routing_warm_start_plan() {
        PreparedNode::Sequential(vec![
            PreparedStage::allocated(PreparedEngine::Collection(Box::new(warm_start)), threads),
            PreparedStage::single(PreparedEngine::Collection(Box::new(plan))),
        ])
    } else if threads > 1
        && matches!(plan.engine(), EngineKind::ListLocalSearch | EngineKind::RoutingLocalSearch | EngineKind::ScheduleLocalSearch)
    {
        PreparedNode::Portfolio { workers: threads, engine: Box::new(PreparedEngine::Collection(Box::new(plan))) }
    } else {
        PreparedNode::Single(Box::new(PreparedEngine::Collection(Box::new(plan))))
    }
}

fn cp_node(plan: CpSolvePlan, threads: usize) -> PreparedNode {
    if threads > 1 {
        PreparedNode::Portfolio { workers: threads, engine: Box::new(PreparedEngine::Cp(Box::new(plan))) }
    } else {
        PreparedNode::Single(Box::new(PreparedEngine::Cp(Box::new(plan))))
    }
}

fn sat_node(plan: SatSolvePlan) -> PreparedNode {
    PreparedNode::Single(Box::new(PreparedEngine::Sat(Box::new(plan))))
}

pub fn compile_model_plan(package: &ModelPackage, request: &SolveRequest, budget: &SolveBudget) -> Result<PreparedSolvePlan, SolveError> {
    compile_model_plan_inner(package, request, budget, true)
}

fn compile_model_plan_inner(
    package: &ModelPackage,
    request: &SolveRequest,
    budget: &SolveBudget,
    allow_decomposition: bool,
) -> Result<PreparedSolvePlan, SolveError> {
    request.validate()?;
    if budget.expired() {
        return Err(SolveError::Interrupted("solve budget expired before model preparation".to_string()));
    }
    if !package.validate_interruptible(budget.stop()).map_err(|errors| SolveError::Compile(errors.join("; ")))? {
        return Err(SolveError::Interrupted("solve budget expired during semantic validation".to_string()));
    }
    let model = &package.model;
    if request.sat.backend.is_some() {
        return compile_sat_plan(model, request, budget).and_then(|plan| prepared_plan(sat_node(plan), request));
    }
    let has_integer = !model.int_vars().is_empty() || !model.sets().is_empty();
    let has_collection = !model.lists().is_empty() || !model.intervals().is_empty();
    let collection_families = usize::from(!model.lists().is_empty()) + usize::from(!model.intervals().is_empty());
    if has_collection {
        preflight_collection_memory(model, request, budget)?;
    }
    if allow_decomposition {
        match model.independent_family_components_interruptible(budget.stop()) {
            IndependentDecomposition::Interrupted => {
                return Err(SolveError::Interrupted("solve budget expired during semantic decomposition".to_string()));
            }
            IndependentDecomposition::Components(components) => {
                // Many tiny homogeneous components can spend their shared
                // deadline in sequential setup while the corresponding
                // monolithic backend finds a complete incumbent immediately.
                // Prefer that route only after it has actually compiled. Mixed
                // families and disjoint list universes have no monolithic
                // representation and must retain their decomposed plan.
                if should_try_monolithic_deadline_plan(&components, request) {
                    if let Some(plan) =
                        try_compile_monolithic_plan(model, request, budget, has_integer, has_collection, collection_families)?
                    {
                        return Ok(plan);
                    }
                }
                validate_decomposition_coverage(&components, model)?;
                validate_decomposition_limits(&components, request)?;
                validate_decomposition_request_references(model, request)?;
                let mut prepared = Vec::with_capacity(components.len());
                let mut retained_backend_bytes = 0u64;
                for component in components {
                    if budget.expired() {
                        return Err(SolveError::Interrupted("solve budget expired while preparing a decomposition component".to_string()));
                    }
                    let interval_mode_originals = dense_local_to_original(&component.interval_modes, "interval mode")?;
                    let mut component_request = project_component_request(&component, !component.model.objectives().is_empty(), request)?;
                    if let Some(limit) = request.limits.memory_bytes {
                        let remaining = limit.checked_sub(retained_backend_bytes).ok_or_else(|| {
                            SolveError::Compile(format!(
                                "prepared decomposition components already require an estimated {retained_backend_bytes} bytes, above the memory limit"
                            ))
                        })?;
                        if remaining == 0 {
                            return Err(SolveError::Compile(
                                "prepared decomposition components exhaust the memory limit before the next component".to_string(),
                            ));
                        }
                        component_request.limits.memory_bytes = Some(remaining);
                    }
                    let package = package
                        .project_interruptible(component.model, &component.objects, budget.stop())
                        .ok_or_else(|| SolveError::Interrupted("solve budget expired during component metadata projection".to_string()))?;
                    let plan = compile_model_plan_inner(&package, &component_request, budget, false)?;
                    retained_backend_bytes = retained_backend_bytes.saturating_add(plan.estimated_backend_bytes());
                    prepared.push(PreparedComponent {
                        family: component.family,
                        package,
                        plan: Box::new(plan),
                        integers: component.integers,
                        sets: component.sets,
                        lists: component.lists,
                        intervals: component.intervals,
                        interval_mode_originals,
                        objective_tiers: component.objective_tiers,
                    });
                }
                if budget.expired() {
                    return Err(SolveError::Interrupted("solve budget expired while finalizing semantic decomposition".to_string()));
                }
                return prepared_plan(PreparedNode::Decomposed(prepared), request);
            }
            IndependentDecomposition::NotApplicable => {}
        }
    }
    if budget.expired() {
        return Err(SolveError::Interrupted("solve budget expired during model preparation".to_string()));
    }
    if (has_integer && has_collection) || collection_families > 1 {
        return Err(SolveError::Unsupported("mixed models with coupled objectives cannot be decomposed independently".to_string()));
    }
    if has_collection {
        return compile_collection_plan(model, request, budget)
            .and_then(|plan| prepared_plan(collection_node(plan, request.threads), request));
    }
    compile_cp_plan_validated(model, request, budget).and_then(|plan| prepared_plan(cp_node(plan, request.threads), request))
}

/// Above this size, try a compiled monolithic alternative under a deadline.
/// This is only a plan preference. Failure to compile that alternative always
/// preserves the complete decomposed representation.
const MAX_DEADLINE_DECOMPOSITION_COMPONENTS: usize = 16;

fn should_try_monolithic_deadline_plan(components: &[IndependentComponent], request: &SolveRequest) -> bool {
    request.limits.time.is_some() && components.len() > MAX_DEADLINE_DECOMPOSITION_COMPONENTS
}

fn try_compile_monolithic_plan(
    model: &Model,
    request: &SolveRequest,
    budget: &SolveBudget,
    has_integer: bool,
    has_collection: bool,
    collection_families: usize,
) -> Result<Option<PreparedSolvePlan>, SolveError> {
    if (has_integer && has_collection) || collection_families > 1 {
        return Ok(None);
    }
    let attempt = if has_collection {
        compile_collection_plan(model, request, budget).and_then(|plan| prepared_plan(collection_node(plan, request.threads), request))
    } else {
        compile_cp_plan_validated(model, request, budget).and_then(|plan| prepared_plan(cp_node(plan, request.threads), request))
    };
    match attempt {
        Ok(plan) => Ok(Some(plan)),
        Err(error @ SolveError::Interrupted(_)) => Err(error),
        // Compilation itself is the capability probe. Any semantic, physical,
        // memory, or request incompatibility leaves the proven decomposition in
        // place instead of changing a supported model into an error.
        Err(_) => Ok(None),
    }
}

fn validate_decomposition_coverage(components: &[IndependentComponent], model: &Model) -> Result<(), SolveError> {
    let mut integers = vec![false; model.int_vars().len()];
    let mut sets = vec![false; model.sets().len()];
    let mut lists = vec![false; model.lists().len()];
    let mut intervals = vec![false; model.intervals().len()];
    let mut modes = vec![false; model.interval_modes().len()];
    let mut constraints = vec![false; model.constraints().len()];
    let mut objectives = vec![false; model.objectives().len()];
    for component in components {
        mark_mapping_coverage(&component.integers, &mut integers, component.model.int_vars().len(), "integer")?;
        mark_mapping_coverage(&component.sets, &mut sets, component.model.sets().len(), "set")?;
        mark_mapping_coverage(&component.lists, &mut lists, component.model.lists().len(), "list")?;
        mark_mapping_coverage(&component.intervals, &mut intervals, component.model.intervals().len(), "interval")?;
        mark_mapping_coverage(&component.interval_modes, &mut modes, component.model.interval_modes().len(), "interval mode")?;
        for original in component.objects.keys().filter_map(|object| match object {
            ModelObject::Constraint(reference) => Some(reference.0),
            _ => None,
        }) {
            let Some(covered) = constraints.get_mut(original) else {
                return Err(SolveError::Compile(format!("decomposition maps unknown semantic constraint {original}")));
            };
            if std::mem::replace(covered, true) {
                return Err(SolveError::Compile(format!("semantic constraint {original} is owned by more than one component")));
            }
        }
        for &original in &component.objective_tiers {
            let Some(covered) = objectives.get_mut(original) else {
                return Err(SolveError::Compile(format!("decomposition maps unknown semantic objective tier {original}")));
            };
            if std::mem::replace(covered, true) {
                return Err(SolveError::Compile(format!("semantic objective tier {original} is owned by more than one component")));
            }
        }
    }
    for (kind, coverage) in [
        ("integer", integers.as_slice()),
        ("set", sets.as_slice()),
        ("list", lists.as_slice()),
        ("interval", intervals.as_slice()),
        ("interval mode", modes.as_slice()),
        ("constraint", constraints.as_slice()),
        ("objective tier", objectives.as_slice()),
    ] {
        if let Some(index) = coverage.iter().position(|covered| !covered) {
            return Err(SolveError::Compile(format!("semantic {kind} {index} has no decomposition owner")));
        }
    }
    Ok(())
}

fn mark_mapping_coverage(mappings: &[(usize, usize)], originals: &mut [bool], local_count: usize, kind: &str) -> Result<(), SolveError> {
    let mut locals = vec![false; local_count];
    for &(original, local) in mappings {
        let Some(original_covered) = originals.get_mut(original) else {
            return Err(SolveError::Compile(format!("decomposition maps unknown semantic {kind} {original}")));
        };
        if std::mem::replace(original_covered, true) {
            return Err(SolveError::Compile(format!("semantic {kind} {original} is owned by more than one component")));
        }
        let Some(local_covered) = locals.get_mut(local) else {
            return Err(SolveError::Compile(format!("{kind} decomposition maps out-of-range local index {local}")));
        };
        if std::mem::replace(local_covered, true) {
            return Err(SolveError::Compile(format!("{kind} decomposition maps local index {local} more than once")));
        }
    }
    if let Some(local) = locals.iter().position(|covered| !covered) {
        return Err(SolveError::Compile(format!("component-local {kind} {local} has no semantic owner")));
    }
    Ok(())
}

fn validate_decomposition_limits(components: &[IndependentComponent], request: &SolveRequest) -> Result<(), SolveError> {
    if request.publish_incumbent_assignments {
        return Err(SolveError::InvalidRequest(
            "publish_incumbent_assignments is not supported for decomposed plans because component incumbents are not complete assignments for the original model"
                .to_string(),
        ));
    }
    if request.limits.conflicts.is_some() && components.iter().any(|component| component.family != IndependentFamily::IntegerSet) {
        return Err(SolveError::InvalidRequest(
            "a conflict limit cannot be applied to decomposed list or interval components because those families do not consume conflicts"
                .to_string(),
        ));
    }
    if request.schedule_cdcl && components.iter().any(|component| component.family == IndependentFamily::Intervals) {
        return Err(SolveError::InvalidRequest("schedule_cdcl requires a selected exact scheduling backend".to_string()));
    }
    Ok(())
}

fn validate_decomposition_request_references(model: &Model, request: &SolveRequest) -> Result<(), SolveError> {
    for assumption in &request.assumptions {
        if assumption.variable >= model.int_vars().len() {
            return Err(SolveError::InvalidRequest(format!("assumption references unknown integer variable {}", assumption.variable)));
        }
    }
    for &(variable, _) in &request.hints {
        if variable >= model.int_vars().len() {
            return Err(SolveError::InvalidRequest(format!("hint references unknown integer variable {variable}")));
        }
    }
    for &variable in &request.branch_order {
        if variable >= model.int_vars().len() {
            return Err(SolveError::InvalidRequest(format!("branch_order references unknown integer variable {variable}")));
        }
    }
    if let Some(scope) = &request.primary_branch_scope {
        for &variable in scope {
            if variable >= model.int_vars().len() {
                return Err(SolveError::InvalidRequest(format!("primary_branch_scope references unknown integer variable {variable}")));
            }
        }
    }
    if let Some(hint) = &request.list_hint {
        if model.lists().is_empty() {
            return Err(SolveError::InvalidRequest("list_hint is only supported for list_vars models".to_string()));
        }
        let has_hidden_remainder = model.lists().last().is_some_and(|list| list.role == crate::model::ListRole::HiddenRemainder);
        let hintable_lists = model.lists().len().saturating_sub(usize::from(has_hidden_remainder));
        let wrong_sequence_count = if has_hidden_remainder { hint.len() > hintable_lists } else { hint.len() != hintable_lists };
        if wrong_sequence_count {
            return Err(SolveError::InvalidRequest(format!(
                "list_hint has {} sequences but the model has {} user-visible lists{}",
                hint.len(),
                hintable_lists,
                if has_hidden_remainder { "; the hidden remainder pool is implicit" } else { "" }
            )));
        }
    }
    Ok(())
}

fn project_component_request(
    component: &IndependentComponent,
    has_objectives: bool,
    request: &SolveRequest,
) -> Result<SolveRequest, SolveError> {
    let mut projected = request.clone();
    match component.family {
        IndependentFamily::IntegerSet => {
            projected.list_hint = None;
            projected.schedule_cdcl = false;
            projected.routing = super::RoutingControls::default();
            projected.assumptions = request
                .assumptions
                .iter()
                .filter_map(|assumption| {
                    component
                        .integers
                        .iter()
                        .find_map(|&(original, local)| (original == assumption.variable).then_some(local))
                        .map(|variable| super::SemanticAssumption { variable, ..*assumption })
                })
                .collect();
            projected.hints = request
                .hints
                .iter()
                .filter_map(|&(variable, value)| {
                    component
                        .integers
                        .iter()
                        .find_map(|&(original, local)| (original == variable).then_some(local))
                        .map(|variable| (variable, value))
                })
                .collect();
            projected.primary_branch_scope = request.primary_branch_scope.as_ref().map(|scope| {
                let local_by_original = component.integers.iter().copied().collect::<BTreeMap<_, _>>();
                scope.iter().filter_map(|variable| local_by_original.get(variable).copied()).collect()
            });
            projected.branch_order = request
                .branch_order
                .iter()
                .filter_map(|&variable| component.integers.iter().find_map(|&(original, local)| (original == variable).then_some(local)))
                .collect();
            if !has_objectives {
                projected.cp.split = false;
                projected.cp.probes = 0;
                projected.cp.lns = 0;
                projected.linear = super::LinearControls::default();
            }
        }
        IndependentFamily::Lists => {
            projected.assumptions.clear();
            projected.hints.clear();
            projected.primary_branch_scope = None;
            projected.branch_order.clear();
            projected.limits.conflicts = None;
            projected.schedule_cdcl = false;
            projected.cp = super::CpControls::default();
            projected.linear = super::LinearControls::default();
            projected.list_hint = match &request.list_hint {
                Some(hint) => {
                    let mut local = vec![Vec::new(); component.model.lists().len()];
                    for &(original, local_index) in &component.lists {
                        let Some(slot) = local.get_mut(local_index) else {
                            return Err(SolveError::Compile("list decomposition produced a sparse local index mapping".to_string()));
                        };
                        if let Some(sequence) = hint.get(original) {
                            *slot = sequence.clone();
                        }
                    }
                    if component.model.lists().last().is_some_and(|list| list.role == crate::model::ListRole::HiddenRemainder) {
                        local.pop();
                    }
                    Some(local)
                }
                None => None,
            };
        }
        IndependentFamily::Intervals => {
            projected.assumptions.clear();
            projected.hints.clear();
            projected.list_hint = None;
            projected.primary_branch_scope = None;
            projected.branch_order.clear();
            projected.limits.conflicts = None;
            projected.routing = super::RoutingControls::default();
            projected.cp = super::CpControls::default();
            projected.linear = super::LinearControls::default();
        }
    }
    Ok(projected)
}

fn dense_local_to_original(mappings: &[(usize, usize)], kind: &str) -> Result<Vec<usize>, SolveError> {
    let mut originals = vec![usize::MAX; mappings.len()];
    for &(original, local) in mappings {
        let Some(slot) = originals.get_mut(local) else {
            return Err(SolveError::Compile(format!("{kind} decomposition produced an out-of-range local index {local}")));
        };
        if std::mem::replace(slot, original) != usize::MAX {
            return Err(SolveError::Compile(format!("{kind} decomposition produced duplicate local index {local}")));
        }
    }
    if originals.contains(&usize::MAX) {
        return Err(SolveError::Compile(format!("{kind} decomposition produced a sparse local index mapping")));
    }
    Ok(originals)
}

fn prepared_plan(node: PreparedNode, request: &SolveRequest) -> Result<PreparedSolvePlan, SolveError> {
    let plan = PreparedSolvePlan { node, request: request.clone() };
    if request.proof == ProofRequest::Require && !prepared_node_can_prove_complete(&plan.node) {
        return Err(SolveError::InvalidRequest("proof=Require needs an exact plan authorized to produce a completion proof".to_string()));
    }
    if request.limits.memory_bytes.is_some_and(|limit| plan.estimated_backend_bytes() > limit) {
        return Err(SolveError::Compile(format!(
            "prepared plan requires an estimated {} bytes across concurrent workers, above the memory limit",
            plan.estimated_backend_bytes()
        )));
    }
    Ok(plan)
}

fn prepared_node_can_prove_complete(node: &PreparedNode) -> bool {
    match node {
        PreparedNode::Single(engine) | PreparedNode::Portfolio { engine, .. } => engine_kind(engine).can_prove_complete(),
        PreparedNode::Sequential(stages) => stages.last().is_some_and(|stage| engine_kind(&stage.engine).can_prove_complete()),
        PreparedNode::Decomposed(components) => {
            !components.is_empty() && components.iter().all(|component| prepared_node_can_prove_complete(&component.plan.node))
        }
    }
}

pub(crate) fn execute_model_plan(
    package: &ModelPackage,
    plan: &PreparedSolvePlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    if request != &plan.request {
        return Err(SolveError::InvalidRequest("execution request must match the request used to compile the solve plan".to_string()));
    }
    let decomposed = matches!(&plan.node, PreparedNode::Decomposed(_));
    let result = match &plan.node {
        PreparedNode::Single(engine) => {
            if emit_stage_started(sink, budget, engine_kind(engine), false)? {
                execute_engine(
                    EngineExecution {
                        model: &package.model,
                        engine,
                        allocation: WorkerAllocation::single(),
                        request,
                        budget,
                        transferred_incumbent: None,
                        warm_stops: None,
                    },
                    sink,
                )?
            } else {
                stopped_before_execution_result(budget)?
            }
        }
        PreparedNode::Sequential(stages) => execute_sequential(&package.model, stages, request, budget, sink)?,
        PreparedNode::Portfolio { workers, engine } => {
            if emit_stage_started(sink, budget, engine_kind(engine), false)? {
                execute_engine(
                    EngineExecution {
                        model: &package.model,
                        engine,
                        allocation: WorkerAllocation::portfolio(*workers),
                        request,
                        budget,
                        transferred_incumbent: None,
                        warm_stops: None,
                    },
                    sink,
                )?
            } else {
                stopped_before_execution_result(budget)?
            }
        }
        PreparedNode::Decomposed(components) => execute_decomposed(package, components, request, budget, sink)?,
    };
    if request.proof == ProofRequest::Require && result.proof.is_none() {
        return Err(SolveError::InvalidResult("a required proof was not produced".to_string()));
    }
    result.validate_model_contract(&package.model)?;
    if decomposed {
        let mut aggregate_events = AggregateEventSink { target: sink };
        super::publish_result_events(&result, budget, &mut aggregate_events)?;
    } else {
        super::publish_result_events(&result, budget, sink)?;
    }
    Ok(result)
}

/// Component candidates, bounds, and proofs are only valid against the
/// projected model. Progress and completed engine reports are safe to expose at
/// the parent boundary and let a consumer stop before the next component.
struct ComponentEventSink<'a> {
    target: &'a mut dyn EventSink,
    objective_tiers: &'a [usize],
    known_objectives: &'a [Option<i64>],
}

impl EventSink for ComponentEventSink<'_> {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        match event {
            SolveEvent::Progress { engine, objectives, elapsed } => {
                if objectives.len() > self.objective_tiers.len() {
                    return Err(SolveError::InvalidResult(format!(
                        "decomposition component published {} progress tiers for a {}-tier projection",
                        objectives.len(),
                        self.objective_tiers.len()
                    )));
                }
                if self.objective_tiers.is_empty() {
                    return Ok(EventControl::Continue);
                }
                let mut aggregate = self.known_objectives.to_vec();
                for (&value, &tier) in objectives.iter().zip(self.objective_tiers) {
                    let slot = aggregate.get_mut(tier).ok_or_else(|| {
                        SolveError::InvalidResult(format!("component progress maps to unknown semantic objective tier {tier}"))
                    })?;
                    if slot.is_some_and(|known| known != value) {
                        return Err(SolveError::InvalidResult(format!(
                            "component progress conflicts with the completed value of semantic objective tier {tier}"
                        )));
                    }
                    *slot = Some(value);
                }
                let Some(objectives) = aggregate.into_iter().collect::<Option<Vec<_>>>() else {
                    return Ok(EventControl::Continue);
                };
                self.target.emit(SolveEvent::Progress { engine, objectives, elapsed })
            }
            event @ SolveEvent::Finished(_) => self.target.emit(event),
            // A component's own stage markers are internal to the decomposition
            // (and would surface one per component); the parent solve is a single
            // logical run, so drop them here like the other component-local events.
            SolveEvent::Candidate(_) | SolveEvent::Bound(_) | SolveEvent::Proof(_) | SolveEvent::StageStarted { .. } => {
                Ok(EventControl::Continue)
            }
        }
    }
}

/// Decomposed engine reports are streamed at each component boundary. Keep the
/// aggregate finalization from publishing those same reports a second time.
struct AggregateEventSink<'a> {
    target: &'a mut dyn EventSink,
}

struct EngineExecution<'a> {
    model: &'a crate::model::Model,
    engine: &'a PreparedEngine,
    allocation: WorkerAllocation,
    request: &'a SolveRequest,
    budget: &'a SolveBudget,
    transferred_incumbent: Option<&'a CandidateSolution>,
    warm_stops: Option<&'a super::WarmStartStops>,
}

impl EventSink for AggregateEventSink<'_> {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        match event {
            SolveEvent::Finished(_) => Ok(EventControl::Continue),
            event => self.target.emit(event),
        }
    }
}

fn execute_engine(run: EngineExecution<'_>, sink: &mut dyn EventSink) -> Result<SolveResult, SolveError> {
    let EngineExecution { model, engine, allocation, request, budget, transferred_incumbent, warm_stops } = run;
    match engine {
        PreparedEngine::Cp(plan) => {
            reject_transferred_incumbent(transferred_incumbent, EngineKind::IntegerExact)?;
            solve_cp_plan_validated(model, plan, allocation, request, budget, sink)
        }
        PreparedEngine::Collection(plan) => {
            if let Some(warm_stops) = warm_stops {
                solve_collection_plan_with_stop(
                    model,
                    plan,
                    request,
                    budget,
                    request.list_hint.as_deref(),
                    transferred_incumbent,
                    warm_stops,
                    allocation,
                    sink,
                )
            } else {
                super::solve_collection_plan_allocated(
                    model,
                    plan,
                    request,
                    budget,
                    request.list_hint.as_deref(),
                    transferred_incumbent,
                    allocation,
                    sink,
                )
            }
        }
        PreparedEngine::Sat(plan) => {
            reject_transferred_incumbent(transferred_incumbent, EngineKind::IntegerExact)?;
            solve_sat_plan(model, plan, request, budget, sink)
        }
    }
}

fn reject_transferred_incumbent(candidate: Option<&CandidateSolution>, engine: EngineKind) -> Result<(), SolveError> {
    if candidate.is_some() {
        return Err(SolveError::InvalidResult(format!(
            "engine {} cannot consume an incumbent transferred by a preceding stage",
            engine.name()
        )));
    }
    Ok(())
}

fn execute_sequential(
    model: &crate::model::Model,
    stages: &[PreparedStage],
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    if stages.is_empty() {
        return Err(SolveError::Compile("sequential solve plan has no stages".to_string()));
    }
    let mut prior_reports = Vec::new();
    let mut transferred = None;
    let mut last_verified_primal = None;
    for (index, stage) in stages.iter().enumerate() {
        if budget.expired() {
            return stopped_sequence_result(budget, prior_reports, last_verified_primal);
        }
        let last = index + 1 == stages.len();
        if !emit_stage_started(sink, budget, engine_kind(&stage.engine), !last)? {
            return stopped_sequence_result(budget, prior_reports, last_verified_primal);
        }
        let warm_stops = (!last).then(|| budget.warm_start_stops()).flatten();
        if !last && warm_stops.is_none() {
            continue;
        }
        let warm_started = Instant::now();
        let mut result = match execute_engine(
            EngineExecution {
                model,
                engine: &stage.engine,
                allocation: stage.allocation,
                request,
                budget,
                transferred_incumbent: transferred.as_ref(),
                warm_stops: warm_stops.as_ref(),
            },
            sink,
        ) {
            Ok(result) => result,
            Err(SolveError::Interrupted(_)) if last_verified_primal.is_some() => {
                return stopped_sequence_result(budget, prior_reports, last_verified_primal);
            }
            Err(error) if !last && !budget.hard_cancelled() => {
                rejected_warm_start_result(engine_kind(&stage.engine), warm_started.elapsed(), error)
            }
            Err(error) => return Err(error),
        };
        if last {
            if !prior_reports.is_empty() {
                prior_reports.append(&mut result.reports);
                result.reports = prior_reports;
            }
            if transferred.is_some() && result.status == SolveStatus::Unsatisfiable {
                return Err(SolveError::InvalidResult(
                    "a sequential exact stage declared UNSAT despite a verified transferred incumbent".to_string(),
                ));
            }
            if let (Some(warm), Some(final_candidate)) = (transferred.as_ref(), result.primal.as_ref()) {
                if candidate_strictly_better(model, warm, final_candidate)? {
                    if result.status == SolveStatus::Optimal {
                        return Err(SolveError::InvalidResult(
                            "a sequential exact stage proved an objective worse than its verified warm-start incumbent".to_string(),
                        ));
                    }
                    result.status = SolveStatus::Satisfiable;
                    result.primal = Some(promote_transfer_candidate(warm.clone())?);
                    result.proof = None;
                    result.message = Some("retained a verified warm-start incumbent better than the final stage candidate".to_string());
                }
            }
            if result.status == SolveStatus::Unknown && result.primal.is_none() {
                if let Some(candidate) = transferred {
                    result.status = SolveStatus::Satisfiable;
                    result.primal = Some(promote_transfer_candidate(candidate)?);
                    result.proof = None;
                    result.message = Some(format!(
                        "final sequential stage stopped; retained the verified warm-start incumbent: {:?}",
                        budget.termination_reason()
                    ));
                }
            }
            return Ok(result);
        }
        last_verified_primal = result.primal.clone();
        prior_reports.append(&mut result.reports);
        apply_transfer_replay_audit_hook(budget);
        transferred = match last_verified_primal.as_ref().map(|candidate| replay_transfer_candidate(model, candidate, budget)).transpose() {
            Ok(candidate) => candidate,
            Err(SolveError::Interrupted(_)) if last_verified_primal.is_some() => {
                return stopped_sequence_result(budget, prior_reports, last_verified_primal);
            }
            Err(error) => return Err(error),
        };
        if let Some(candidate) = transferred.clone() {
            emit_transfer(sink, budget, candidate)?;
        }
        if budget.expired() {
            return stopped_sequence_result(budget, prior_reports, last_verified_primal);
        }
    }
    unreachable!("a non-empty sequential plan returns from its final stage")
}

fn rejected_warm_start_result(engine: EngineKind, elapsed: Duration, error: SolveError) -> SolveResult {
    SolveResult {
        status: SolveStatus::Unknown,
        primal: None,
        bounds: Vec::new(),
        proof: None,
        reports: vec![super::EngineReport {
            engine: Some(engine),
            search: Default::default(),
            elapsed,
            improvements: 0,
            metadata: vec![
                ("warm_start_outcome".to_string(), "rejected".to_string()),
                ("warm_start_rejection".to_string(), error.to_string()),
            ],
        }],
        message: Some(format!("optional warm start was rejected: {error}")),
    }
}

fn candidate_strictly_better(
    model: &crate::model::Model,
    candidate: &CandidateSolution,
    incumbent: &CandidateSolution,
) -> Result<bool, SolveError> {
    if candidate.objectives().len() != model.objectives().len() || incumbent.objectives().len() != model.objectives().len() {
        return Err(SolveError::InvalidResult("sequential candidates do not cover every semantic objective tier".to_string()));
    }
    for ((candidate, incumbent), objective) in candidate.objectives().iter().zip(incumbent.objectives()).zip(model.objectives()) {
        if candidate == incumbent {
            continue;
        }
        return Ok(if objective.is_minimize() { candidate < incumbent } else { candidate > incumbent });
    }
    Ok(false)
}

fn promote_transfer_candidate(candidate: CandidateSolution) -> Result<CandidateSolution, SolveError> {
    if candidate.verification() != VerificationLevel::Transfer {
        return Err(SolveError::InvalidResult("sequential fallback candidate was not verified for transfer".to_string()));
    }
    Ok(CandidateSolution::verified(
        candidate.assignment().clone(),
        candidate.objectives().to_vec(),
        candidate.source(),
        VerificationLevel::Final,
    ))
}

fn replay_transfer_candidate(
    model: &crate::model::Model,
    candidate: &CandidateSolution,
    budget: &SolveBudget,
) -> Result<CandidateSolution, SolveError> {
    let objectives = super::verify_semantic_assignment_interruptible(model, candidate.assignment(), candidate.objectives(), budget.stop())?;
    Ok(CandidateSolution::verified(candidate.assignment().clone(), objectives, candidate.source(), VerificationLevel::Transfer))
}

fn emit_transfer(sink: &mut dyn EventSink, budget: &SolveBudget, candidate: CandidateSolution) -> Result<(), SolveError> {
    if candidate.verification() != VerificationLevel::Transfer {
        return Err(SolveError::InvalidResult(
            "sequential solve attempted to publish an incumbent without transfer verification".to_string(),
        ));
    }
    match sink.emit(SolveEvent::Candidate(candidate))? {
        EventControl::Continue => Ok(()),
        EventControl::Stop => {
            budget.cancel_with(TerminationReason::EventSink);
            Ok(())
        }
    }
}

fn emit_stage_started(sink: &mut dyn EventSink, budget: &SolveBudget, engine: EngineKind, warm_start: bool) -> Result<bool, SolveError> {
    match sink.emit(SolveEvent::StageStarted { engine, warm_start })? {
        EventControl::Continue => Ok(true),
        EventControl::Stop => {
            budget.cancel_with(TerminationReason::EventSink);
            Ok(false)
        }
    }
}

fn stopped_before_execution_result(budget: &SolveBudget) -> Result<SolveResult, SolveError> {
    let result = SolveResult {
        status: SolveStatus::Unknown,
        primal: None,
        bounds: Vec::new(),
        proof: None,
        reports: Vec::new(),
        message: Some(format!("solve stopped before engine execution: {:?}", budget.termination_reason())),
    };
    result.validate_contract()?;
    Ok(result)
}

fn stopped_sequence_result(
    budget: &SolveBudget,
    reports: Vec<super::EngineReport>,
    primal: Option<CandidateSolution>,
) -> Result<SolveResult, SolveError> {
    let status = if primal.is_some() { SolveStatus::Satisfiable } else { SolveStatus::Unknown };
    let result = SolveResult {
        status,
        primal,
        bounds: Vec::new(),
        proof: None,
        reports,
        message: Some(format!("sequential solve stopped between stages: {:?}", budget.termination_reason())),
    };
    result.validate_contract()?;
    Ok(result)
}

fn execute_decomposed(
    package: &ModelPackage,
    components: &[PreparedComponent],
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    let mut assignment = Assignment {
        integers: vec![None; package.model.int_vars().len()],
        sets: vec![Vec::new(); package.model.sets().len()],
        lists: vec![Vec::new(); package.model.lists().len()],
        intervals: vec![Default::default(); package.model.intervals().len()],
    };
    let mut reports = Vec::new();
    let mut unsatisfiable_proof = None;
    let mut optimality_proofs = Vec::new();
    let mut all_objective_components_optimal = true;
    let mut objective_values = vec![None; package.model.objectives().len()];
    let mut bounds = Vec::new();
    let mut incomplete = None;
    let mut remaining_iterations = request.limits.iterations;
    let mut remaining_conflicts = request.limits.conflicts;
    for (index, component) in components.iter().enumerate() {
        if budget.expired() {
            incomplete = Some((
                SolveStatus::Unknown,
                Some(format!("decomposition stopped before component {}: {:?}", index + 1, budget.termination_reason())),
            ));
            break;
        }

        let component_request = component_execution_request(component, remaining_iterations, remaining_conflicts);
        let mut execution_plan = component.plan.as_ref().clone();
        execution_plan.request = component_request.clone();
        let mut component_events =
            ComponentEventSink { target: sink, objective_tiers: &component.objective_tiers, known_objectives: &objective_values };
        let mut result = execute_model_plan(&component.package, &execution_plan, &component_request, budget, &mut component_events)?;
        debit_component_quotas(&component_request, &result, &mut remaining_iterations, &mut remaining_conflicts)?;
        reports.append(&mut result.reports);
        merge_component_bounds(component, &mut result.bounds, &mut bounds)?;
        match result.status {
            SolveStatus::Unsatisfiable => {
                unsatisfiable_proof = Some(
                    result
                        .proof
                        .ok_or_else(|| SolveError::InvalidResult("unsatisfiable decomposition component has no proof".to_string()))?,
                );
                break;
            }
            SolveStatus::Unknown | SolveStatus::Unsupported => {
                incomplete = Some((
                    result.status,
                    result.message.or_else(|| {
                        Some(format!("decomposition component {} returned {} without a diagnostic", index + 1, result.status.as_str()))
                    }),
                ));
                break;
            }
            SolveStatus::Optimal | SolveStatus::Satisfiable => {
                if !component.objective_tiers.is_empty() {
                    if result.status == SolveStatus::Optimal {
                        optimality_proofs.push(
                            result
                                .proof
                                .take()
                                .ok_or_else(|| SolveError::InvalidResult("optimal decomposition component has no proof".to_string()))?,
                        );
                    } else {
                        all_objective_components_optimal = false;
                    }
                }
                let candidate = result
                    .primal
                    .ok_or_else(|| SolveError::InvalidResult("feasible decomposition component has no candidate".to_string()))?;
                if candidate.objectives().len() != component.objective_tiers.len() {
                    return Err(SolveError::InvalidResult(format!(
                        "decomposition component returned {} objective tiers for a {}-tier projection",
                        candidate.objectives().len(),
                        component.objective_tiers.len()
                    )));
                }
                for (&local_value, &original_tier) in candidate.objectives().iter().zip(&component.objective_tiers) {
                    let slot = objective_values.get_mut(original_tier).ok_or_else(|| {
                        SolveError::InvalidResult(format!("component objective maps to unknown semantic tier {original_tier}"))
                    })?;
                    if slot.replace(local_value).is_some() {
                        return Err(SolveError::InvalidResult(format!(
                            "semantic objective tier {original_tier} is owned by more than one decomposition component"
                        )));
                    }
                }
                match component.family {
                    IndependentFamily::IntegerSet => {
                        for &(original, local) in &component.integers {
                            assignment.integers[original] = candidate.assignment().integers[local];
                        }
                        for &(original, local) in &component.sets {
                            assignment.sets[original] = candidate.assignment().sets[local].clone();
                        }
                    }
                    IndependentFamily::Lists => {
                        for &(original, local) in &component.lists {
                            assignment.lists[original] = candidate.assignment().lists[local].clone();
                        }
                    }
                    IndependentFamily::Intervals => {
                        for &(original, local) in &component.intervals {
                            let mut value = candidate.assignment().intervals[local];
                            if let Some(local_mode) = value.mode {
                                value.mode = Some(*component.interval_mode_originals.get(local_mode).ok_or_else(|| {
                                    SolveError::InvalidResult(format!(
                                        "decomposition component returned unknown local interval mode {local_mode}"
                                    ))
                                })?);
                            }
                            assignment.intervals[original] = value;
                        }
                    }
                }
                if budget.expired() {
                    incomplete = Some((
                        SolveStatus::Unknown,
                        Some(format!("decomposition stopped after component {}: {:?}", index + 1, budget.termination_reason())),
                    ));
                    break;
                }
            }
        }
    }

    bounds.sort_by_key(|bound| bound.tier);
    let result = if let Some(proof) = unsatisfiable_proof {
        let proof = ProofClaim::decomposed(vec![proof], ProvenConclusion::Unsatisfiable, package.model.objectives().len())?;
        SolveResult { status: SolveStatus::Unsatisfiable, primal: None, bounds, proof: Some(proof), reports, message: None }
    } else if let Some((status, message)) = incomplete {
        SolveResult { status, primal: None, bounds, proof: None, reports, message }
    } else {
        let objective_values = objective_values
            .into_iter()
            .enumerate()
            .map(|(tier, value)| {
                value.ok_or_else(|| SolveError::InvalidResult(format!("semantic objective tier {tier} has no decomposition owner")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let objectives = super::verify_final_with_budget(budget, |stop| {
            super::verify_semantic_assignment_validated_interruptible(&package.model, &assignment, &objective_values, stop)
        })?;
        let optimal = !objectives.is_empty() && all_objective_components_optimal;
        let proof = if optimal {
            Some(ProofClaim::decomposed(optimality_proofs, ProvenConclusion::Optimal, package.model.objectives().len())?)
        } else {
            None
        };
        SolveResult {
            status: if optimal { SolveStatus::Optimal } else { SolveStatus::Satisfiable },
            primal: Some(CandidateSolution::verified(assignment, objectives, EngineKind::Verifier, VerificationLevel::Final)),
            bounds,
            proof,
            reports,
            message: None,
        }
    };

    result.validate_contract()?;
    Ok(result)
}

fn merge_component_bounds(
    component: &PreparedComponent,
    component_bounds: &mut Vec<Bound>,
    aggregate: &mut Vec<Bound>,
) -> Result<(), SolveError> {
    for mut bound in component_bounds.drain(..) {
        let original_tier = *component.objective_tiers.get(bound.tier).ok_or_else(|| {
            SolveError::InvalidResult(format!(
                "component bound references local objective tier {} but the component has {} tiers",
                bound.tier,
                component.objective_tiers.len()
            ))
        })?;
        if aggregate.iter().any(|existing| existing.tier == original_tier) {
            return Err(SolveError::InvalidResult(format!(
                "semantic objective tier {original_tier} received bounds from more than one decomposition component"
            )));
        }
        bound.tier = original_tier;
        aggregate.push(bound);
    }
    Ok(())
}

fn component_execution_request(
    component: &PreparedComponent,
    remaining_iterations: Option<u64>,
    remaining_conflicts: Option<u64>,
) -> SolveRequest {
    let mut request = component.plan.request.clone();
    if request.limits.iterations.is_some() {
        request.limits.iterations = remaining_iterations;
    }
    if request.limits.conflicts.is_some() {
        request.limits.conflicts = remaining_conflicts;
    }
    request
}

fn debit_component_quotas(
    request: &SolveRequest,
    result: &SolveResult,
    remaining_iterations: &mut Option<u64>,
    remaining_conflicts: &mut Option<u64>,
) -> Result<(), SolveError> {
    let search = result.aggregate_search_stats();
    if request.limits.iterations.is_some() {
        debit_component_quota(remaining_iterations, search.nodes, "iteration")?;
    }
    if request.limits.conflicts.is_some() {
        debit_component_quota(remaining_conflicts, search.failures, "conflict")?;
    }
    Ok(())
}

fn debit_component_quota(remaining: &mut Option<u64>, consumed: u64, name: &str) -> Result<(), SolveError> {
    let Some(available) = remaining else {
        return Err(SolveError::InvalidResult(format!(
            "a decomposed component consumed a {name} quota that was not present in the parent request"
        )));
    };
    if consumed > *available {
        return Err(SolveError::InvalidResult(format!(
            "a decomposed component consumed {consumed} {name}s with only {available} remaining"
        )));
    }
    *available -= consumed;
    Ok(())
}

/// Arms a scoped monitor completion flag. The flag is released on ordinary
/// return and while unwinding, so a panic in model code or an event sink cannot
/// leave the monitor waiting forever during `thread::scope` teardown.
pub(super) struct MonitorCompletion<'a>(&'a AtomicBool);

impl<'a> MonitorCompletion<'a> {
    pub(super) fn new(done: &'a AtomicBool) -> Self {
        Self(done)
    }
}

impl Drop for MonitorCompletion<'_> {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Compile and solve with one budget spanning validation, compilation, search,
/// transfers, and final verification.
pub fn solve_model(package: &ModelPackage, request: &SolveRequest, sink: &mut dyn EventSink) -> Result<SolveResult, SolveError> {
    let budget = SolveBudget::new(request.limits.time);
    solve_model_with_budget(package, request, &budget, sink)
}

pub fn solve_model_with_stop(
    package: &ModelPackage,
    request: &SolveRequest,
    stop: std::sync::Arc<AtomicBool>,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    let (result, stopped) = solve_with_monitored_external_stop(package, request, stop.as_ref(), sink);
    // Preserve the historical two-way shared flag contract without letting the
    // deadline timer write into the external cancellation source while the
    // solve is running. That separation is what lets a later hard stop override
    // a soft deadline during final replay.
    if stopped {
        stop.store(true, std::sync::atomic::Ordering::Release);
    }
    result
}

/// Canonical solve entry point for callers that expose a borrowed cancellation
/// flag. The scoped monitor feeds the same budget used by compilation, every
/// engine stage, transfers, and final replay, and is always joined before
/// returning.
pub fn solve_model_with_external_stop(
    package: &ModelPackage,
    request: &SolveRequest,
    external_stop: &AtomicBool,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    solve_with_monitored_external_stop(package, request, external_stop, sink).0
}

fn solve_with_monitored_external_stop(
    package: &ModelPackage,
    request: &SolveRequest,
    external_stop: &AtomicBool,
    sink: &mut dyn EventSink,
) -> (Result<SolveResult, SolveError>, bool) {
    let validation = request.validate();
    if let Err(error) = validation {
        return (Err(error), false);
    }
    let budget = SolveBudget::new(request.limits.time);
    if external_stop.load(std::sync::atomic::Ordering::Acquire) {
        budget.cancel_with(TerminationReason::ExternalCancellation);
    }
    let monitor_done = AtomicBool::new(false);
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            while !monitor_done.load(std::sync::atomic::Ordering::Acquire) {
                if external_stop.load(std::sync::atomic::Ordering::Acquire) {
                    budget.cancel_with(TerminationReason::ExternalCancellation);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        });
        let _monitor_completion = MonitorCompletion::new(&monitor_done);
        let (mut result, final_evidence_published) = {
            let mut monitored_sink = super::ExternalStopEventSink::new(external_stop, &budget, sink);
            let result = solve_model_with_budget(package, request, &budget, &mut monitored_sink);
            (result, monitored_sink.final_evidence_published())
        };
        // A stop before final evidence wins. Once a verified candidate or proof
        // has synchronously reached the caller, completion is the linearization
        // point and cannot be retroactively replaced by UNKNOWN.
        if external_stop.load(std::sync::atomic::Ordering::Acquire) && !final_evidence_published {
            budget.cancel_with(TerminationReason::ExternalCancellation);
            result = Ok(stopped_result(&budget, "external cancellation during finalization"));
        }
        result
    });
    // Read shared state only after the scoped monitor has joined. Reading it in
    // the scope body can miss the monitor's last cancellation poll.
    (result, budget.expired())
}

/// Run a specialized frontend adapter under the same control-plane budget as
/// canonical model solves. The adapter supplies only the engine operation;
/// request validation, memory policy, deadline, and cancellation stay owned by
/// the orchestrator.
#[cfg(test)]
pub(crate) fn execute_specialized<T>(
    request: &SolveRequest,
    stop: std::sync::Arc<AtomicBool>,
    operation: impl FnOnce(&SolveBudget) -> Result<T, SolveError>,
) -> Result<T, SolveError> {
    request.validate()?;
    let budget = SolveBudget::with_stop(request.limits.time, stop);
    super::budget::apply_memory_limit(request.limits.memory_bytes, &budget);
    operation(&budget)
}

pub fn solve_model_silent(package: &ModelPackage, request: &SolveRequest) -> Result<SolveResult, SolveError> {
    solve_model(package, request, &mut IgnoreEvents)
}

pub(crate) fn solve_model_with_budget(
    package: &ModelPackage,
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    request.validate()?;
    let result = solve_model_with_budget_inner(package, request, budget, sink)?;
    if request.proof == ProofRequest::Require && result.proof.is_none() {
        return Err(SolveError::InvalidResult("a required proof was not produced".to_string()));
    }
    Ok(result)
}

fn solve_model_with_budget_inner(
    package: &ModelPackage,
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    if budget.expired() {
        return Ok(stopped_result(budget, "request setup"));
    }
    let compile_started = Instant::now();
    match compile_model_plan(package, request, budget) {
        Ok(plan) => {
            // Backend compilers perform their allocation estimates before
            // materializing physical state. Start live RSS enforcement only
            // after that deterministic preflight so an impossible request is a
            // compile diagnostic rather than a scheduler-dependent UNKNOWN.
            super::budget::apply_memory_limit(request.limits.memory_bytes, budget);
            if budget.expired() {
                return Ok(stopped_result(budget, "backend memory setup"));
            }
            let build_seconds = compile_started.elapsed().as_secs_f64();
            let estimated_bytes = plan.estimated_backend_bytes();
            let mut result = match execute_model_plan(package, &plan, request, budget, sink) {
                Ok(result) => result,
                Err(SolveError::Interrupted(_)) => return Ok(stopped_result(budget, "execution or final verification")),
                Err(error) => return Err(error),
            };
            if let Some(report) = result.reports.first_mut() {
                report.metadata.extend([
                    ("backend_build_seconds".to_string(), build_seconds.to_string()),
                    ("estimated_backend_bytes".to_string(), estimated_bytes.to_string()),
                ]);
            }
            Ok(result)
        }
        Err(SolveError::Interrupted(_)) => Ok(stopped_result(budget, "compilation")),
        Err(SolveError::Unsupported(message)) => Ok(SolveResult {
            status: SolveStatus::Unsupported,
            primal: None,
            bounds: Vec::new(),
            proof: None,
            reports: Vec::new(),
            message: Some(message),
        }),
        Err(error) => Err(error),
    }
}

fn stopped_result(budget: &SolveBudget, phase: &str) -> SolveResult {
    SolveResult {
        status: SolveStatus::Unknown,
        primal: None,
        bounds: Vec::new(),
        proof: None,
        reports: Vec::new(),
        message: Some(format!("solve stopped during {phase}: {:?}", budget.termination_reason())),
    }
}
