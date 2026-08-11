#[cfg(test)]
use std::cell::RefCell;
use std::collections::HashSet;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::engines::ls::lists;
use crate::engines::CollectionCompileContext;
use crate::engines::CompileFailure;
use crate::engines::{dual, list_exact, routing, schedule};
use crate::model::list::{self, CollectionSolution};
use crate::model::{self, CompiledCollection, ListRole, Model};

use super::{
    Assignment, Bound, CandidateSolution, EngineKind, EngineReport, EventControl, EventSink, IntervalValue, ModelFamily, ProofClaim,
    ProvenConclusion, RoutingControls, SolveBudget, SolveError, SolveEvent, SolveMode, SolveRequest, SolveResult, SolveStatus,
    TerminationReason, VerificationLevel,
};

const ROUTING_WARM_START_ITERATIONS: u64 = 2_000;
const AUTO_ORDERED_EXACT_ITEMS: usize = 10;
const AUTO_ASSIGNMENT_EXACT_ITEMS: usize = 24;
const AUTO_ASSIGNMENT_EXACT_CELLS: usize = 192;
const AUTO_SCHEDULE_EXACT_INTERVALS: usize = 48;
const AUTO_SCHEDULE_EXACT_MODES: usize = 96;

#[derive(Clone)]
enum CollectionBackend {
    RoutingExact(Box<routing::CompiledRouting>),
    ListExact(Box<list_exact::CompiledListExact>),
    ScheduleExact(Box<schedule::CompiledSchedule>),
    ListLocalSearch,
    RoutingLocalSearch,
    ScheduleLocalSearch,
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum FinalReplayAudit {
    None,
    SoftDeadline,
    SoftDeadlineThenExternal,
}

#[cfg(test)]
std::thread_local! {
    static FINAL_REPLAY_AUDIT: std::cell::Cell<FinalReplayAudit> = const { std::cell::Cell::new(FinalReplayAudit::None) };
}

#[cfg(test)]
pub(crate) fn audit_interrupt_next_collection_final_replay(hard_cancel_after_interrupt: bool) {
    FINAL_REPLAY_AUDIT.set(if hard_cancel_after_interrupt {
        FinalReplayAudit::SoftDeadlineThenExternal
    } else {
        FinalReplayAudit::SoftDeadline
    });
}

#[cfg(test)]
fn apply_final_replay_audit_before_first_pass(budget: &SolveBudget) {
    FINAL_REPLAY_AUDIT.with(|audit| {
        if !matches!(audit.get(), FinalReplayAudit::None) {
            budget.cancel_with(TerminationReason::Deadline);
        }
    });
}

#[cfg(not(test))]
fn apply_final_replay_audit_before_first_pass(_budget: &SolveBudget) {}

#[cfg(test)]
fn apply_final_replay_audit_after_interrupt(budget: &SolveBudget) {
    FINAL_REPLAY_AUDIT.with(|audit| {
        if matches!(audit.replace(FinalReplayAudit::None), FinalReplayAudit::SoftDeadlineThenExternal) {
            budget.cancel_with(TerminationReason::ExternalCancellation);
        }
    });
}

#[cfg(not(test))]
fn apply_final_replay_audit_after_interrupt(_budget: &SolveBudget) {}

impl CollectionBackend {
    fn engine(&self) -> EngineKind {
        match self {
            Self::RoutingExact(_) => EngineKind::RoutingExact,
            Self::ListExact(_) => EngineKind::ListExact,
            Self::ScheduleExact(_) => EngineKind::ScheduleExact,
            Self::ListLocalSearch => EngineKind::ListLocalSearch,
            Self::RoutingLocalSearch => EngineKind::RoutingLocalSearch,
            Self::ScheduleLocalSearch => EngineKind::ScheduleLocalSearch,
        }
    }
}

struct BackendCandidate {
    backend: CollectionBackend,
    family: ModelFamily,
    reason: &'static str,
    estimated_bytes: u64,
    estimated_build_work: u128,
    auto_eligible: bool,
    preference: u8,
}

struct CompilerDecline {
    engine: EngineKind,
    code: &'static str,
    detail: String,
}

struct ExactCompilation {
    candidates: Vec<BackendCandidate>,
    declines: Vec<CompilerDecline>,
}

fn decline_summary(declines: &[CompilerDecline]) -> String {
    declines.iter().map(|decline| format!("[{}:{}] {}", decline.engine.name(), decline.code, decline.detail)).collect::<Vec<_>>().join("; ")
}

/// Collection plan whose backend-specific capability parsing is already done.
#[derive(Clone)]
pub(crate) struct CollectionSolvePlan {
    model: Arc<CompiledCollection>,
    backend: CollectionBackend,
    family: ModelFamily,
    reason: String,
    estimated_backend_bytes: u64,
    routing_controls: RoutingControls,
    objective_tiers: usize,
    local_search_iterations: Option<u64>,
}

impl CollectionSolvePlan {
    pub(crate) fn engine(&self) -> EngineKind {
        self.backend.engine()
    }

    pub(crate) fn estimated_backend_bytes(&self) -> u64 {
        self.estimated_backend_bytes
    }

    pub(crate) fn has_routing_warm_start(&self) -> bool {
        matches!(self.backend, CollectionBackend::RoutingExact(_)) && self.routing_controls.warm_start
    }

    pub(crate) fn routing_warm_start_plan(&self) -> Option<Self> {
        self.has_routing_warm_start().then(|| Self {
            model: Arc::clone(&self.model),
            backend: local_search_backend(self.model.as_model()),
            family: self.family,
            reason: "routing warm-start local search".to_string(),
            estimated_backend_bytes: model::estimated_local_search_backend_bytes(self.model.as_model()),
            routing_controls: self.routing_controls,
            objective_tiers: self.objective_tiers,
            local_search_iterations: Some(ROUTING_WARM_START_ITERATIONS),
        })
    }
}

/// Compile a semantic list or interval model into one executable collection
/// backend. Backend parsing happens here and is never repeated during run.
pub(crate) fn compile_collection_plan(
    semantic: &Model,
    request: &SolveRequest,
    budget: &SolveBudget,
) -> Result<CollectionSolvePlan, SolveError> {
    request.validate()?;
    let auto_requests_local_search =
        request.mode == SolveMode::Auto && !request.schedule_cdcl && (request.threads > 1 || request.limits.iterations.is_some());
    preflight_collection_memory(semantic, request, budget)?;
    let compiled = Arc::new(
        CompiledCollection::compile_interruptible(semantic, budget.stop())
            .map_err(|error| SolveError::Compile(error.reason))?
            .ok_or_else(|| SolveError::Interrupted("solve budget expired during collection compilation".to_string()))?,
    );
    let physical = compiled.as_model();
    validate_collection_request(semantic, physical, request)?;
    let estimated_backend_bytes = model::estimated_exact_backend_bytes(physical);
    let estimated_local_search_bytes = model::estimated_local_search_backend_bytes(physical);
    let family = physical_family(physical);
    if request.mode == SolveMode::LocalSearch || auto_requests_local_search {
        let backend = local_search_backend(physical);
        let plan = CollectionSolvePlan {
            model: Arc::clone(&compiled),
            backend,
            family,
            reason: if request.mode == SolveMode::LocalSearch {
                "explicit local-search request"
            } else {
                "automatic request requires a parallel or iteration-bounded local-search plan"
            }
            .to_string(),
            estimated_backend_bytes: estimated_local_search_bytes,
            routing_controls: request.routing,
            objective_tiers: semantic.objectives().len(),
            local_search_iterations: None,
        };
        validate_selected_plan_options(&plan, request)?;
        return Ok(plan);
    }

    if let Some(limit) = request.limits.memory_bytes.filter(|&limit| estimated_backend_bytes > limit) {
        if request.mode == SolveMode::Exact {
            return Err(SolveError::Compile(format!(
                "exact collection plan requires an estimated {estimated_backend_bytes} bytes, exceeding the {limit} byte memory limit"
            )));
        }
        let backend = local_search_backend(physical);
        let plan = CollectionSolvePlan {
            model: Arc::clone(&compiled),
            backend,
            family,
            reason: "exact plan exceeds the memory limit; using local search".to_string(),
            estimated_backend_bytes: estimated_local_search_bytes,
            routing_controls: request.routing,
            objective_tiers: semantic.objectives().len(),
            local_search_iterations: None,
        };
        validate_selected_plan_options(&plan, request)?;
        return Ok(plan);
    }
    if request.mode == SolveMode::Auto && physical.schedule.as_ref().is_some_and(|schedule| !schedule_auto_exact_eligible(schedule)) {
        let plan = CollectionSolvePlan {
            model: Arc::clone(&compiled),
            backend: CollectionBackend::ScheduleLocalSearch,
            family: ModelFamily::Schedule,
            reason: "schedule exceeds the automatic exact-size policy; using local search".to_string(),
            estimated_backend_bytes: estimated_local_search_bytes,
            routing_controls: request.routing,
            objective_tiers: semantic.objectives().len(),
            local_search_iterations: None,
        };
        validate_selected_plan_options(&plan, request)?;
        return Ok(plan);
    }

    let ExactCompilation { mut candidates, declines } =
        compile_exact_candidates(semantic, compiled.as_ref(), request, estimated_backend_bytes, budget.stop())?;
    let accepted_exact = !candidates.is_empty();
    let smallest_estimate = candidates.iter().map(|candidate| candidate.estimated_bytes).min().unwrap_or(estimated_backend_bytes);
    let decline_summary = decline_summary(&declines);
    candidates.retain(|candidate| {
        request.limits.memory_bytes.is_none_or(|limit| candidate.estimated_bytes <= limit)
            && (request.mode == SolveMode::Exact || candidate.auto_eligible)
    });
    candidates.sort_by_key(|candidate| {
        (candidate.preference, candidate.estimated_build_work, candidate.estimated_bytes, candidate.backend.engine())
    });

    if let Some(candidate) = candidates.into_iter().next() {
        let plan = CollectionSolvePlan {
            model: Arc::clone(&compiled),
            backend: candidate.backend,
            family: candidate.family,
            reason: candidate.reason.to_string(),
            estimated_backend_bytes: candidate.estimated_bytes,
            routing_controls: request.routing,
            objective_tiers: semantic.objectives().len(),
            local_search_iterations: None,
        };
        validate_selected_plan_options(&plan, request)?;
        return Ok(plan);
    }
    if request.mode == SolveMode::Auto {
        let backend = local_search_backend(physical);
        let plan = CollectionSolvePlan {
            model: Arc::clone(&compiled),
            backend,
            family,
            reason: if accepted_exact {
                "exact plans exceeded automatic cost or memory policy; using local search".to_string()
            } else if decline_summary.is_empty() {
                "no exact compiler accepted the model; using local search".to_string()
            } else {
                format!("no exact compiler accepted the model; using local search: {decline_summary}")
            },
            estimated_backend_bytes: estimated_local_search_bytes,
            routing_controls: request.routing,
            objective_tiers: semantic.objectives().len(),
            local_search_iterations: None,
        };
        validate_selected_plan_options(&plan, request)?;
        return Ok(plan);
    }
    if accepted_exact {
        let limit = request.limits.memory_bytes.unwrap_or(0);
        return Err(SolveError::Compile(format!(
            "all exact collection plans exceed the {limit} byte memory limit; smallest estimate is {smallest_estimate} bytes"
        )));
    }
    let detail = if decline_summary.is_empty() { "no applicable exact collection compiler".to_string() } else { decline_summary };
    Err(SolveError::Unsupported(format!("all exact collection compilers declined: {detail}")))
}

pub(crate) fn preflight_collection_memory(semantic: &Model, request: &SolveRequest, budget: &SolveBudget) -> Result<u64, SolveError> {
    if budget.expired() {
        return Err(SolveError::Interrupted("solve budget expired before collection compilation".to_string()));
    }
    let semantic_estimates = model::estimated_semantic_collection_bytes_interruptible(semantic, budget.stop())
        .ok_or_else(|| SolveError::Interrupted("solve budget expired during collection memory preflight".to_string()))?;
    let auto_requests_local_search =
        request.mode == SolveMode::Auto && !request.schedule_cdcl && (request.threads > 1 || request.limits.iterations.is_some());
    let local_search_estimate = semantic_estimates.local_search.saturating_mul(u64::try_from(request.threads).unwrap_or(u64::MAX));
    let preflight_estimate = match request.mode {
        SolveMode::Exact => semantic_estimates.exact,
        SolveMode::LocalSearch => local_search_estimate,
        SolveMode::Auto if auto_requests_local_search => local_search_estimate,
        SolveMode::Auto => semantic_estimates.exact.min(local_search_estimate),
    };
    if request.limits.memory_bytes.is_some_and(|limit| preflight_estimate > limit) {
        let estimate_kind = if request.mode == SolveMode::Exact { "estimated exact backend" } else { "estimated collection backend" };
        return Err(SolveError::Compile(format!(
            "{estimate_kind} requires at least {preflight_estimate} bytes across concurrent workers, above the memory limit"
        )));
    }
    if budget.expired() {
        return Err(SolveError::Interrupted("solve budget expired after collection memory preflight".to_string()));
    }
    Ok(preflight_estimate)
}

fn validate_collection_request(semantic: &Model, model: &list::CollectionModel, request: &SolveRequest) -> Result<(), SolveError> {
    if request.linear != super::LinearControls::default() {
        return Err(SolveError::InvalidRequest("linear relaxation controls require an exact integer CP objective".to_string()));
    }
    if request.sat != super::SatControls::default() {
        return Err(SolveError::InvalidRequest("SAT controls require a semantic Boolean clause model".to_string()));
    }
    if request.cp != super::CpControls::default() {
        return Err(SolveError::InvalidRequest("CP portfolio controls require a semantic integer model".to_string()));
    }
    if !request.assumptions.is_empty()
        || !request.hints.is_empty()
        || request.primary_branch_scope.is_some()
        || !request.branch_order.is_empty()
    {
        return Err(SolveError::InvalidRequest(
            "collection plans do not accept integer assumptions, hints, primary branch scope, or branch order".to_string(),
        ));
    }
    if request.publish_incumbent_assignments {
        return Err(SolveError::InvalidRequest(
            "collection plans do not publish assignment snapshots for internal improvements".to_string(),
        ));
    }
    if request.limits.conflicts.is_some() {
        return Err(SolveError::InvalidRequest("conflict limits are only supported by integer and SAT plans".to_string()));
    }
    if let Some(hint) = &request.list_hint {
        let has_hidden_remainder = semantic.lists().last().is_some_and(|list| list.role == ListRole::HiddenRemainder);
        let hintable_lists = model.lists.saturating_sub(usize::from(has_hidden_remainder));
        let wrong_sequence_count = if has_hidden_remainder { hint.len() > hintable_lists } else { hint.len() != hintable_lists };
        if wrong_sequence_count {
            return Err(SolveError::InvalidRequest(format!(
                "list_hint has {} sequences but the model has {} user-visible lists{}",
                hint.len(),
                hintable_lists,
                if has_hidden_remainder { "; the hidden remainder pool is implicit" } else { "" }
            )));
        }
        let universe = model.items.iter().copied().collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        for &item in hint.iter().flatten() {
            if !universe.contains(&item) {
                return Err(SolveError::InvalidRequest(format!("list_hint item {item} is not in the list_vars universe")));
            }
            if !seen.insert(item) {
                return Err(SolveError::InvalidRequest(format!("list_hint assigns item {item} more than once, across more than one list")));
            }
        }
    }
    if request.mode == SolveMode::Exact && request.limits.iterations.is_some() {
        return Err(SolveError::InvalidRequest("iteration limits require a local-search collection plan".to_string()));
    }
    if request.mode == SolveMode::Exact && request.threads > 1 {
        return Err(SolveError::InvalidRequest(
            "exact collection plans currently require threads=1; use mode=auto or local-search for a portfolio".to_string(),
        ));
    }
    if request.schedule_cdcl && model.schedule.is_none() {
        return Err(SolveError::InvalidRequest("schedule_cdcl requires a semantic schedule model".to_string()));
    }
    if request.schedule_cdcl && request.mode == SolveMode::LocalSearch {
        return Err(SolveError::InvalidRequest("schedule_cdcl requires an exact scheduling plan".to_string()));
    }
    if request.schedule_cdcl && request.threads > 1 {
        return Err(SolveError::InvalidRequest("schedule_cdcl currently requires threads=1".to_string()));
    }
    if request.schedule_cdcl && request.limits.iterations.is_some() {
        return Err(SolveError::InvalidRequest("schedule_cdcl does not consume local-search iteration limits".to_string()));
    }
    Ok(())
}

fn validate_selected_plan_options(plan: &CollectionSolvePlan, request: &SolveRequest) -> Result<(), SolveError> {
    if request.routing != RoutingControls::default() && !matches!(plan.backend, CollectionBackend::RoutingExact(_)) {
        return Err(SolveError::InvalidRequest("non-default routing controls require a selected exact routing backend".to_string()));
    }
    if request.schedule_cdcl && !matches!(plan.backend, CollectionBackend::ScheduleExact(_)) {
        return Err(SolveError::InvalidRequest("schedule_cdcl requires a selected exact scheduling backend".to_string()));
    }
    if request.list_hint.is_some()
        && !matches!(plan.backend, CollectionBackend::ListLocalSearch | CollectionBackend::RoutingLocalSearch)
        && !plan.has_routing_warm_start()
    {
        return Err(SolveError::InvalidRequest(
            "list_hint requires list local search or a routing exact plan with warm start enabled".to_string(),
        ));
    }
    if request.limits.iterations.is_some() && matches!(plan.backend, CollectionBackend::ScheduleLocalSearch) {
        return Err(SolveError::InvalidRequest("schedule local search does not currently consume iteration limits".to_string()));
    }
    Ok(())
}

fn compile_exact_candidates(
    semantic: &Model,
    domain: &CompiledCollection,
    request: &SolveRequest,
    estimated_bytes: u64,
    stop: &AtomicBool,
) -> Result<ExactCompilation, SolveError> {
    let context = CollectionCompileContext::new(semantic, request, domain);
    let model = domain.as_model();
    debug_assert_eq!(semantic.intervals().is_empty(), model.schedule.is_none());
    let mut candidates = Vec::new();
    let mut declines = Vec::new();
    ensure_compilation_running(stop, "before routing lowering")?;
    match routing::compile(&context, stop) {
        Ok(compiled) => {
            let nodes = model.items.len().saturating_add(model.lists) as u128;
            candidates.push(BackendCandidate {
                backend: CollectionBackend::RoutingExact(Box::new(compiled)),
                family: ModelFamily::Routing,
                reason: "routing compiler accepted the model",
                estimated_bytes: routing_backend_bytes(model, estimated_bytes),
                estimated_build_work: nodes.saturating_mul(nodes),
                auto_eligible: true,
                preference: 0,
            });
        }
        Err(CompileFailure::Unsupported { code, detail }) => {
            declines.push(CompilerDecline { engine: EngineKind::RoutingExact, code, detail: detail.to_string() });
        }
        Err(CompileFailure::Interrupted { phase }) => {
            return Err(SolveError::Interrupted(format!("solve budget expired {phase}")));
        }
        Err(CompileFailure::Invalid { reason }) => {
            return Err(SolveError::Compile(format!("routing compiler rejected a recognized shape: {reason}")));
        }
    }
    if let Some(schedule) = &model.schedule {
        ensure_compilation_running(stop, "before schedule lowering")?;
        match schedule::compile(&context, stop) {
            Ok(compiled) => candidates.push(BackendCandidate {
                backend: CollectionBackend::ScheduleExact(Box::new(compiled)),
                family: ModelFamily::Schedule,
                reason: "scheduling compiler accepted the model",
                estimated_bytes,
                estimated_build_work: schedule_build_work(schedule),
                auto_eligible: schedule_auto_exact_eligible(schedule),
                preference: 0,
            }),
            Err(CompileFailure::Unsupported { code, detail }) => {
                declines.push(CompilerDecline { engine: EngineKind::ScheduleExact, code, detail: detail.to_string() });
            }
            Err(CompileFailure::Interrupted { phase }) => {
                return Err(SolveError::Interrupted(format!("solve budget expired {phase}")));
            }
            Err(CompileFailure::Invalid { reason }) if request.mode == SolveMode::Exact => {
                return Err(SolveError::Compile(format!("schedule compiler failed: {reason}")));
            }
            Err(CompileFailure::Invalid { reason }) => {
                declines.push(CompilerDecline { engine: EngineKind::ScheduleExact, code: "schedule-numeric-lowering", detail: reason });
            }
        }
    } else {
        ensure_compilation_running(stop, "before list exact lowering")?;
        match list_exact::compile(&context, stop) {
            Ok(compiled) => {
                let ordered = collection_is_order_dependent(model);
                let cells = model.items.len().saturating_mul(model.lists.max(1));
                candidates.push(BackendCandidate {
                    backend: CollectionBackend::ListExact(Box::new(compiled)),
                    family: physical_family(model),
                    reason: "list exact compiler accepted the model",
                    estimated_bytes,
                    estimated_build_work: list_build_work(model, ordered),
                    auto_eligible: if ordered {
                        model.items.len() <= AUTO_ORDERED_EXACT_ITEMS
                    } else {
                        model.items.len() <= AUTO_ASSIGNMENT_EXACT_ITEMS && cells <= AUTO_ASSIGNMENT_EXACT_CELLS
                    },
                    preference: u8::from(physical_family(model) == ModelFamily::Routing),
                });
            }
            Err(CompileFailure::Unsupported { code, detail }) => {
                declines.push(CompilerDecline { engine: EngineKind::ListExact, code, detail: detail.to_string() });
            }
            Err(CompileFailure::Interrupted { phase }) => {
                return Err(SolveError::Interrupted(format!("solve budget expired {phase}")));
            }
            Err(CompileFailure::Invalid { reason }) => {
                return Err(SolveError::Compile(format!("list exact compiler failed: {reason}")));
            }
        }
    }
    ensure_compilation_running(stop, "after exact collection lowering")?;
    Ok(ExactCompilation { candidates, declines })
}

fn routing_backend_bytes(model: &list::CollectionModel, shared_estimate: u64) -> u64 {
    let nodes = u64::try_from(model.items.len().saturating_add(model.lists)).unwrap_or(u64::MAX);
    shared_estimate.saturating_add(nodes.saturating_mul(nodes).saturating_mul(256))
}

fn ensure_compilation_running(stop: &AtomicBool, phase: &str) -> Result<(), SolveError> {
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        Err(SolveError::Interrupted(format!("solve budget expired {phase}")))
    } else {
        Ok(())
    }
}

fn local_search_backend(model: &list::CollectionModel) -> CollectionBackend {
    if model.schedule.is_some() {
        CollectionBackend::ScheduleLocalSearch
    } else if lists::routing_search_supported(model) {
        // Same list engine, but edge-cost objectives activate the routing
        // construction and neighbourhoods; report it as routing local search.
        CollectionBackend::RoutingLocalSearch
    } else {
        CollectionBackend::ListLocalSearch
    }
}

fn schedule_build_work(schedule: &list::Schedule) -> u128 {
    let modes = schedule.intervals.iter().map(|interval| interval.modes.len().max(1) as u128).sum::<u128>();
    modes.saturating_mul(modes).saturating_add(schedule.precedences.len() as u128).saturating_add(schedule.resources.len() as u128)
}

fn schedule_auto_exact_eligible(schedule: &list::Schedule) -> bool {
    schedule.intervals.len() <= AUTO_SCHEDULE_EXACT_INTERVALS
        && schedule
            .intervals
            .iter()
            .try_fold(0usize, |total, interval| total.checked_add(interval.modes.len().max(1)))
            .is_some_and(|modes| modes <= AUTO_SCHEDULE_EXACT_MODES)
}

fn list_build_work(model: &list::CollectionModel, ordered: bool) -> u128 {
    let cells = (model.items.len() as u128).saturating_mul(model.lists.max(1) as u128);
    if !ordered {
        return cells;
    }
    (2..=model.items.len()).fold(cells.max(1), |work, value| work.saturating_mul(value as u128))
}

fn collection_is_order_dependent(model: &list::CollectionModel) -> bool {
    model
        .objectives
        .iter()
        .flat_map(list::ObjectiveTier::reductions)
        .chain(model.constraints.iter().map(|constraint| &constraint.reduction))
        .any(|reduction| {
            matches!(
                reduction.iterable,
                list::Iterable::Edges { .. } | list::Iterable::Pairs(_) | list::Iterable::Scan { .. } | list::Iterable::Windows { .. }
            )
        })
}

fn physical_family(model: &list::CollectionModel) -> ModelFamily {
    if model.schedule.is_some() {
        return ModelFamily::Schedule;
    }
    let mut has_sequence = false;
    for reduction in model
        .objectives
        .iter()
        .flat_map(list::ObjectiveTier::reductions)
        .chain(model.constraints.iter().map(|constraint| &constraint.reduction))
    {
        match reduction.iterable {
            list::Iterable::Edges { .. } => return ModelFamily::Routing,
            list::Iterable::Pairs(_) | list::Iterable::Scan { .. } | list::Iterable::Windows { .. } => has_sequence = true,
            list::Iterable::Items(_) | list::Iterable::SetItems(_) => {}
        }
    }
    if has_sequence {
        ModelFamily::Sequencing
    } else {
        ModelFamily::ListAssignment
    }
}

/// Execute an already compiled collection plan. Only verified candidates cross
/// the event boundary or appear in the final result.
#[cfg(test)]
pub(crate) fn solve_collection_plan(
    semantic: &Model,
    plan: &CollectionSolvePlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    list_hint: Option<&[Vec<i32>]>,
    transferred_incumbent: Option<&CandidateSolution>,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    solve_collection_plan_inner(
        semantic,
        plan,
        request,
        budget,
        list_hint,
        transferred_incumbent,
        None,
        super::WorkerAllocation::portfolio(request.threads),
        sink,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_collection_plan_allocated(
    semantic: &Model,
    plan: &CollectionSolvePlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    list_hint: Option<&[Vec<i32>]>,
    transferred_incumbent: Option<&CandidateSolution>,
    allocation: super::WorkerAllocation,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    solve_collection_plan_inner(semantic, plan, request, budget, list_hint, transferred_incumbent, None, allocation, sink)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_collection_plan_with_stop(
    semantic: &Model,
    plan: &CollectionSolvePlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    list_hint: Option<&[Vec<i32>]>,
    transferred_incumbent: Option<&CandidateSolution>,
    warm_stops: &super::WarmStartStops,
    allocation: super::WorkerAllocation,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    solve_collection_plan_inner(semantic, plan, request, budget, list_hint, transferred_incumbent, Some(warm_stops), allocation, sink)
}

#[allow(clippy::too_many_arguments)]
fn solve_collection_plan_inner(
    semantic: &Model,
    plan: &CollectionSolvePlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    list_hint: Option<&[Vec<i32>]>,
    transferred_incumbent: Option<&CandidateSolution>,
    optional_stops: Option<&super::WarmStartStops>,
    allocation: super::WorkerAllocation,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    request.validate()?;
    if matches!(plan.backend, CollectionBackend::RoutingExact(_)) && request.routing != plan.routing_controls {
        return Err(SolveError::InvalidRequest("routing controls must match the request used to compile the routing plan".to_string()));
    }
    if budget.expired() {
        return Ok(SolveResult::unknown());
    }
    if transferred_incumbent.is_some() && !matches!(plan.backend, CollectionBackend::RoutingExact(_)) {
        return Err(SolveError::InvalidResult("only an exact routing stage can consume a transferred collection incumbent".to_string()));
    }
    let model = plan.model.as_model();
    let transfer_stop = optional_stops.map(super::WarmStartStops::transfer);
    let search_stop = optional_stops.is_none().then(|| budget.search_stop());
    let engine_stop = optional_stops
        .map(super::WarmStartStops::search)
        .unwrap_or_else(|| search_stop.as_ref().expect("ordinary collection search owns a stop token").flag());
    let started = Instant::now();
    let mut event_error = None;
    let outcome = (|| -> Result<SolveResult, SolveError> {
        Ok(match &plan.backend {
            CollectionBackend::RoutingExact(compiled) => {
                let warm = transferred_incumbent.map(warm_solution_from_verified).transpose()?;
                let (outcome, structural_bound) = std::thread::scope(|scope| {
                    let dual_task = scope.spawn(|| dual::compute(model, engine_stop));
                    let outcome = routing::solve_compiled(compiled, request.seed, engine_stop, warm.as_ref(), &mut |value| {
                        record_progress(sink, budget, &mut event_error, EngineKind::RoutingExact, value);
                    });
                    let structural_bound = dual_task.join().map_err(|_| SolveError::Engine("dual-bound worker panicked".to_string()))?;
                    Ok::<_, SolveError>((outcome, structural_bound))
                })?;
                result_from_routing(semantic, model, outcome, structural_bound, started.elapsed(), budget)?
            }
            CollectionBackend::ListExact(compiled) => {
                let structural_bound = dual::compute(model, engine_stop);
                let outcome = list_exact::solve_compiled(compiled, model, engine_stop, |values| {
                    if let Some(&value) = values.first() {
                        record_progress(sink, budget, &mut event_error, EngineKind::ListExact, value);
                    }
                });
                result_from_list_exact(semantic, model, outcome, structural_bound, started.elapsed(), budget)?
            }
            CollectionBackend::ScheduleExact(compiled) => {
                let structural_bound = dual::compute(model, engine_stop);
                let outcome = schedule::solve_compiled(
                    compiled,
                    engine_stop,
                    schedule::Options { seed: request.seed, optional_modes_cdcl: request.schedule_cdcl },
                    |value| record_progress(sink, budget, &mut event_error, EngineKind::ScheduleExact, value),
                )
                .map_err(SolveError::Engine)?;
                result_from_schedule(semantic, model, outcome, structural_bound, started.elapsed(), budget)?
            }
            // Routing local search is the same list engine specialised on edge
            // costs; it only reports under a distinct kind so the phase is visible.
            CollectionBackend::ListLocalSearch | CollectionBackend::RoutingLocalSearch => result_from_local_search(
                ListLocalSearchRun {
                    semantic,
                    model,
                    request,
                    budget,
                    engine_stop,
                    transfer_stop,
                    list_hint,
                    max_iterations: plan.local_search_iterations.or(request.limits.iterations).unwrap_or(u64::MAX),
                    allocation,
                    engine: plan.engine(),
                    started,
                },
                &mut |engine, value| record_progress(sink, budget, &mut event_error, engine, value),
            )?,
            CollectionBackend::ScheduleLocalSearch => result_from_schedule_local_search(
                ScheduleLocalSearchRun { semantic, model, request, budget, engine_stop, transfer_stop, allocation, started },
                &mut |engine, value| record_progress(sink, budget, &mut event_error, engine, value),
            )?,
        })
    })();
    let mut result = match outcome {
        Ok(result) => result,
        Err(error) => return Err(event_error.take().unwrap_or(error)),
    };

    if let Some(error) = event_error.take() {
        return Err(error);
    }
    if let Some(report) = result.reports.first_mut() {
        report
            .metadata
            .extend([("selection_reason".to_string(), plan.reason.clone()), ("model_family".to_string(), plan.family.name().to_string())]);
    }
    if result.proof.is_some() {
        result.proof = completion_proof(plan.engine(), result.status, plan.objective_tiers, true);
    }
    result.validate_contract()?;
    Ok(result)
}

fn result_from_routing(
    semantic: &Model,
    model: &list::CollectionModel,
    mut outcome: routing::RoutingIntegerOutcome,
    structural_bound: Option<dual::DualBound>,
    elapsed: Duration,
    budget: &SolveBudget,
) -> Result<SolveResult, SolveError> {
    let status = if outcome.solution.feasible {
        if outcome.complete {
            SolveStatus::Optimal
        } else {
            SolveStatus::Satisfiable
        }
    } else if outcome.complete {
        SolveStatus::Unsatisfiable
    } else {
        SolveStatus::Unknown
    };
    if status == SolveStatus::Optimal {
        dual::attach_exact(&mut outcome.solution, "exact routing proof");
    } else {
        dual::attach(model, &mut outcome.solution, structural_bound);
    }
    finish_collection_result(
        semantic,
        model,
        CollectionCompletion {
            solution: outcome.solution,
            status,
            source: EngineKind::RoutingExact,
            proof: completion_proof(EngineKind::RoutingExact, status, model.objectives.len(), outcome.complete),
            report: EngineReport {
                engine: Some(EngineKind::RoutingExact),
                search: outcome.stats,
                elapsed,
                improvements: outcome.improvements,
                metadata: Vec::new(),
            },
        },
        budget,
        None,
    )
}

fn result_from_list_exact(
    semantic: &Model,
    model: &list::CollectionModel,
    outcome: list_exact::Outcome,
    structural_bound: Option<dual::DualBound>,
    elapsed: Duration,
    budget: &SolveBudget,
) -> Result<SolveResult, SolveError> {
    let status = match outcome.status {
        list_exact::Status::Optimal => SolveStatus::Optimal,
        list_exact::Status::Satisfiable => SolveStatus::Satisfiable,
        list_exact::Status::Unsatisfiable => SolveStatus::Unsatisfiable,
        list_exact::Status::Unknown => SolveStatus::Unknown,
    };
    let mut solution = CollectionSolution {
        lists: outcome.solution.unwrap_or_default(),
        objectives: outcome.objectives,
        feasible: matches!(status, SolveStatus::Optimal | SolveStatus::Satisfiable),
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        modes: Vec::new(),
        bound: None,
    };
    if status == SolveStatus::Optimal {
        dual::attach_exact(&mut solution, "exact list proof");
    } else {
        dual::attach(model, &mut solution, structural_bound);
    }
    let proof = completion_proof(EngineKind::ListExact, status, model.objectives.len(), true);
    finish_collection_result(
        semantic,
        model,
        CollectionCompletion {
            solution,
            status,
            source: EngineKind::ListExact,
            proof,
            report: EngineReport {
                engine: Some(EngineKind::ListExact),
                search: outcome.stats,
                elapsed,
                improvements: 0,
                metadata: Vec::new(),
            },
        },
        budget,
        None,
    )
}

fn result_from_schedule(
    semantic: &Model,
    model: &list::CollectionModel,
    outcome: schedule::Outcome,
    structural_bound: Option<dual::DualBound>,
    elapsed: Duration,
    budget: &SolveBudget,
) -> Result<SolveResult, SolveError> {
    let status = match outcome.status {
        schedule::Status::Optimal => SolveStatus::Optimal,
        schedule::Status::Satisfiable => SolveStatus::Satisfiable,
        schedule::Status::Unsatisfiable => SolveStatus::Unsatisfiable,
        schedule::Status::Unknown => SolveStatus::Unknown,
    };
    let mut solution = CollectionSolution {
        lists: Vec::new(),
        objectives: outcome.objective.into_iter().collect(),
        feasible: matches!(status, SolveStatus::Optimal | SolveStatus::Satisfiable),
        starts: outcome.starts,
        presences: outcome.presences,
        machines: outcome.machines,
        modes: outcome.modes,
        bound: None,
    };
    if status == SolveStatus::Optimal {
        dual::attach_exact(&mut solution, "exact schedule proof");
    } else {
        dual::attach(model, &mut solution, structural_bound);
    }
    let proof = completion_proof(EngineKind::ScheduleExact, status, model.objectives.len(), true);
    finish_collection_result(
        semantic,
        model,
        CollectionCompletion {
            solution,
            status,
            source: EngineKind::ScheduleExact,
            proof,
            report: EngineReport {
                engine: Some(EngineKind::ScheduleExact),
                search: outcome.stats,
                elapsed,
                improvements: 0,
                metadata: Vec::new(),
            },
        },
        budget,
        None,
    )
}

struct ListLocalSearchRun<'a> {
    semantic: &'a Model,
    model: &'a list::CollectionModel,
    request: &'a SolveRequest,
    budget: &'a SolveBudget,
    engine_stop: &'a AtomicBool,
    transfer_stop: Option<&'a AtomicBool>,
    list_hint: Option<&'a [Vec<i32>]>,
    max_iterations: u64,
    allocation: super::WorkerAllocation,
    engine: EngineKind,
    started: Instant,
}

#[derive(Clone, Default)]
struct DeferredDualStart {
    state: Arc<(Mutex<DeferredDualState>, Condvar)>,
}

#[derive(Clone, Copy, Default)]
struct DeferredDualState {
    released: bool,
    cancelled: bool,
}

impl DeferredDualStart {
    fn release(&self) {
        let (lock, wake) = &*self.state;
        let mut state = lock.lock().expect("dual-start state lock poisoned");
        state.released = true;
        wake.notify_one();
    }

    fn cancel(&self) {
        let (lock, wake) = &*self.state;
        let mut state = lock.lock().expect("dual-start state lock poisoned");
        state.cancelled = true;
        wake.notify_one();
    }

    fn wait(&self) -> bool {
        let (lock, wake) = &*self.state;
        let mut state = lock.lock().expect("dual-start state lock poisoned");
        while !state.released && !state.cancelled {
            state = wake.wait(state).expect("dual-start state lock poisoned");
        }
        state.released
    }
}

struct DeferredDualGate {
    start: DeferredDualStart,
    settled: bool,
}

impl DeferredDualGate {
    fn new(start: DeferredDualStart) -> Self {
        Self { start, settled: false }
    }

    fn release(&mut self) {
        if !self.settled {
            self.start.release();
            self.settled = true;
        }
    }

    fn cancel(&mut self) {
        if !self.settled {
            self.start.cancel();
            self.settled = true;
        }
    }
}

impl Drop for DeferredDualGate {
    fn drop(&mut self) {
        if !self.settled {
            self.start.cancel();
        }
    }
}

#[cfg(test)]
thread_local! {
    static LOCAL_SEARCH_DUAL_AUDIT: RefCell<Option<Arc<LocalSearchDualAudit>>> = const { RefCell::new(None) };
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct LocalSearchDualAudit {
    progress_depth: AtomicUsize,
    pub(crate) dual_started: AtomicBool,
    pub(crate) dual_started_during_progress: AtomicBool,
}

#[cfg(test)]
pub(crate) struct LocalSearchDualAuditGuard;

#[cfg(test)]
pub(crate) fn audit_watch_local_search_dual(state: Arc<LocalSearchDualAudit>) -> LocalSearchDualAuditGuard {
    LOCAL_SEARCH_DUAL_AUDIT.with(|slot| {
        *slot.borrow_mut() = Some(state);
    });
    LocalSearchDualAuditGuard
}

#[cfg(test)]
impl Drop for LocalSearchDualAuditGuard {
    fn drop(&mut self) {
        LOCAL_SEARCH_DUAL_AUDIT.with(|slot| {
            *slot.borrow_mut() = None;
        });
    }
}

#[cfg(test)]
fn local_search_dual_audit_state() -> Option<Arc<LocalSearchDualAudit>> {
    LOCAL_SEARCH_DUAL_AUDIT.with(|slot| slot.borrow().clone())
}

#[cfg(test)]
struct LocalSearchProgressAudit {
    state: Option<Arc<LocalSearchDualAudit>>,
}

#[cfg(test)]
impl LocalSearchProgressAudit {
    fn new() -> Self {
        let state = local_search_dual_audit_state();
        if let Some(state) = &state {
            state.progress_depth.fetch_add(1, Ordering::AcqRel);
        }
        Self { state }
    }
}

#[cfg(test)]
impl Drop for LocalSearchProgressAudit {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            state.progress_depth.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

#[cfg(test)]
fn audit_local_search_dual_started(state: Option<&Arc<LocalSearchDualAudit>>) {
    if let Some(state) = state {
        state.dual_started.store(true, Ordering::Release);
        if state.progress_depth.load(Ordering::Acquire) > 0 {
            state.dual_started_during_progress.store(true, Ordering::Release);
        }
    }
}

fn result_from_local_search(run: ListLocalSearchRun<'_>, improvement: &mut impl FnMut(EngineKind, i64)) -> Result<SolveResult, SolveError> {
    let ListLocalSearchRun {
        semantic,
        model,
        request,
        budget,
        engine_stop,
        transfer_stop,
        list_hint,
        max_iterations,
        allocation,
        engine,
        started,
    } = run;
    let (mut solution, metadata, structural_bound, iterations) = std::thread::scope(|scope| {
        let dual_start = DeferredDualStart::default();
        let mut dual_gate = DeferredDualGate::new(dual_start.clone());
        let dual_task = {
            let dual_start = dual_start.clone();
            #[cfg(test)]
            let dual_audit = local_search_dual_audit_state();
            scope.spawn(move || {
                if !dual_start.wait() || engine_stop.load(Ordering::Relaxed) {
                    return None;
                }
                #[cfg(test)]
                audit_local_search_dual_started(dual_audit.as_ref());
                dual::compute(model, engine_stop)
            })
        };
        let mut dual_released = false;
        let mut publish_first_incumbent = |value| {
            #[cfg(test)]
            let _progress_audit = LocalSearchProgressAudit::new();
            improvement(engine, value);
            if !dual_released {
                dual_gate.release();
                dual_released = true;
            }
        };
        let workers = allocation.workers();
        let (solution, metadata, iterations) = if workers > 1 {
            let (solution, metrics) = lists::solve_collection_parallel_validated(
                model,
                request.seed,
                engine_stop,
                workers,
                max_iterations,
                list_hint,
                &mut publish_first_incumbent,
                request.profile,
            );
            let candidates = metrics.worker_metrics.iter().map(|worker| worker.search.candidates).sum::<u64>();
            let iterations = metrics.worker_metrics.iter().map(|worker| list_search_iterations(&worker.search)).sum::<u64>();
            let elapsed_nanos = metrics.worker_metrics.iter().map(|worker| worker.search.elapsed_nanos).max().unwrap_or(0);
            let trials = metrics.worker_metrics.iter().map(|worker| worker.search.trial_list_evaluations).sum::<u64>();
            let full = metrics.worker_metrics.iter().map(|worker| worker.search.full_recompute_trial_list_evaluations).sum::<u64>();
            let construction_candidates = metrics.worker_metrics.iter().map(|worker| worker.search.construction_candidates).sum::<u64>();
            let candidates_per_second = if elapsed_nanos == 0 { 0.0 } else { candidates as f64 * 1_000_000_000.0 / elapsed_nanos as f64 };
            let full_recompute_percentage = if trials == 0 { 0.0 } else { full as f64 * 100.0 / trials as f64 };
            let mut metadata = vec![
                ("workers".to_string(), metrics.workers.to_string()),
                ("publications".to_string(), metrics.publications.to_string()),
                ("injections".to_string(), metrics.injections.to_string()),
                ("candidates_evaluated".to_string(), candidates.to_string()),
                ("alns_iterations".to_string(), iterations.to_string()),
                ("candidates_per_second".to_string(), candidates_per_second.to_string()),
                ("full_recompute_percentage".to_string(), full_recompute_percentage.to_string()),
                ("construction_candidates".to_string(), construction_candidates.to_string()),
            ];
            if let Some(best) = metrics.best_worker.and_then(|best| metrics.worker_metrics.iter().find(|worker| worker.worker == best)) {
                append_search_metadata(&mut metadata, &best.search);
            }
            (solution, metadata, iterations)
        } else {
            let (solution, metrics) = lists::solve_collection_validated(
                model,
                request.seed,
                engine_stop,
                max_iterations,
                list_hint,
                &mut publish_first_incumbent,
                request.profile,
            );
            let mut metadata = Vec::new();
            append_search_metadata(&mut metadata, &metrics);
            let iterations = list_search_iterations(&metrics);
            (solution, metadata, iterations)
        };
        if solution.feasible && !solution.objectives.is_empty() && !dual_released {
            dual_gate.release();
            dual_released = true;
        }
        if !dual_released {
            dual_gate.cancel();
        }
        let structural_bound = dual_task.join().map_err(|_| SolveError::Engine("dual-bound worker panicked".to_string()))?;
        Ok::<_, SolveError>((solution, metadata, structural_bound, iterations))
    })?;
    dual::attach(model, &mut solution, structural_bound);
    let status = if solution.feasible { SolveStatus::Satisfiable } else { SolveStatus::Unknown };
    let mut result = finish_collection_result(
        semantic,
        model,
        CollectionCompletion {
            solution,
            status,
            source: engine,
            proof: None,
            report: EngineReport {
                engine: Some(engine),
                search: crate::search::SolveStats { nodes: iterations, ..Default::default() },
                elapsed: started.elapsed(),
                improvements: 0,
                metadata,
            },
        },
        budget,
        transfer_stop,
    )?;
    if request.limits.iterations.is_some_and(|limit| iterations >= limit) && !budget.expired() {
        result.message = Some("list local search reached the shared iteration limit".to_string());
    }
    Ok(result)
}

struct ScheduleLocalSearchRun<'a> {
    semantic: &'a Model,
    model: &'a list::CollectionModel,
    request: &'a SolveRequest,
    budget: &'a SolveBudget,
    engine_stop: &'a AtomicBool,
    transfer_stop: Option<&'a AtomicBool>,
    allocation: super::WorkerAllocation,
    started: Instant,
}

fn result_from_schedule_local_search(
    run: ScheduleLocalSearchRun<'_>,
    improvement: &mut impl FnMut(EngineKind, i64),
) -> Result<SolveResult, SolveError> {
    let ScheduleLocalSearchRun { semantic, model, request, budget, engine_stop, transfer_stop, allocation, started } = run;
    let schedule = model
        .schedule
        .as_ref()
        .ok_or_else(|| SolveError::InvalidResult("schedule local-search plan has no physical schedule".to_string()))?;
    let (summary, structural_bound) = std::thread::scope(|scope| {
        let dual_task = scope.spawn(|| dual::compute(model, engine_stop));
        let summary = super::execute_workers(
            vec![(); allocation.workers()],
            engine_stop,
            Arc::new(AtomicBool::new(false)),
            request.seed,
            |context, ()| {
                let seed = context.seed();
                lists::solve_schedule(schedule, seed, context.stop(), &mut |value| {
                    let _ = context.publish_latest(value);
                })
            },
            |event| {
                improvement(EngineKind::ScheduleLocalSearch, event.payload);
                Ok::<_, SolveError>(EventControl::Continue)
            },
        )?;
        let structural_bound = dual_task.join().map_err(|_| SolveError::Engine("dual-bound worker panicked".to_string()))?;
        Ok::<_, SolveError>((summary, structural_bound))
    })?;

    let mut best: Option<CollectionSolution> = None;
    let mut construction_candidates = 0u64;
    let mut construction_elapsed = Duration::ZERO;
    let mut first_feasible: Option<Duration> = None;
    for report in summary.reports {
        let (candidate, metrics): (CollectionSolution, lists::ScheduleConstructionMetrics) = report.result;
        construction_candidates = construction_candidates.saturating_add(metrics.candidates);
        construction_elapsed = construction_elapsed.max(metrics.elapsed);
        first_feasible = match (first_feasible, metrics.first_feasible) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (None, right) => right,
            (left, None) => left,
        };
        let replace =
            candidate.feasible && best.as_ref().is_none_or(|incumbent| !incumbent.feasible || candidate.objectives < incumbent.objectives);
        if replace {
            best = Some(candidate);
        }
    }
    let mut solution = best.unwrap_or(CollectionSolution {
        lists: Vec::new(),
        objectives: Vec::new(),
        feasible: false,
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        modes: Vec::new(),
        bound: None,
    });
    dual::attach(model, &mut solution, structural_bound);
    let mut metadata = vec![
        ("workers".to_string(), allocation.workers().to_string()),
        ("constructor".to_string(), "serial-sgs".to_string()),
        ("construction_seconds".to_string(), construction_elapsed.as_secs_f64().to_string()),
        ("construction_candidates".to_string(), construction_candidates.to_string()),
    ];
    if let Some(first) = first_feasible {
        metadata.push(("time_to_first_feasible".to_string(), first.as_secs_f64().to_string()));
    }
    let status = if solution.feasible { SolveStatus::Satisfiable } else { SolveStatus::Unknown };
    finish_collection_result(
        semantic,
        model,
        CollectionCompletion {
            solution,
            status,
            source: EngineKind::ScheduleLocalSearch,
            proof: None,
            report: EngineReport {
                engine: Some(EngineKind::ScheduleLocalSearch),
                search: Default::default(),
                elapsed: started.elapsed(),
                improvements: 0,
                metadata,
            },
        },
        budget,
        transfer_stop,
    )
}

fn append_search_metadata(metadata: &mut Vec<(String, String)>, metrics: &lists::ListSearchMetrics) {
    let routing = metrics.routing();
    metadata.extend([
        ("alns_iterations".to_string(), metrics.alns.iterations.to_string()),
        ("candidates_evaluated".to_string(), metrics.candidates.to_string()),
        ("candidates_per_second".to_string(), metrics.candidates_per_second().to_string()),
        ("full_recompute_percentage".to_string(), metrics.full_recompute_percentage().to_string()),
        ("construction_seconds".to_string(), (metrics.construction_nanos as f64 / 1_000_000_000.0).to_string()),
        ("construction_candidates".to_string(), metrics.construction_candidates.to_string()),
        ("routing_slices".to_string(), routing.slices.to_string()),
        ("routing_descent_slices".to_string(), routing.descent_slices.to_string()),
        ("routing_alns_slices".to_string(), routing.alns_slices.to_string()),
        ("routing_relink_slices".to_string(), routing.relink_slices.to_string()),
        ("routing_global_scan_slices".to_string(), routing.global_scan_slices.to_string()),
        ("routing_route_elimination_attempts".to_string(), routing.route_elimination_attempts.to_string()),
        ("routing_ejection_chain_attempts".to_string(), routing.ejection_chain_attempts.to_string()),
        ("routing_chain_relocate_attempts".to_string(), routing.chain_relocate_attempts.to_string()),
        ("routing_guided_segment_exchange_attempts".to_string(), routing.guided_segment_exchange_attempts.to_string()),
        ("routing_macro_candidates_built".to_string(), routing.macro_candidates_built.to_string()),
        ("routing_macro_budget_exhaustions".to_string(), routing.macro_budget_exhaustions.to_string()),
        ("routing_elite_insertions".to_string(), routing.elite_insertions.to_string()),
        ("routing_elite_rejections".to_string(), routing.elite_rejections.to_string()),
        ("routing_path_relink_attempts".to_string(), routing.path_relink_attempts.to_string()),
        ("routing_path_relink_steps".to_string(), routing.path_relink_steps.to_string()),
        ("routing_path_relink_budget_exhaustions".to_string(), routing.path_relink_budget_exhaustions.to_string()),
        ("anytime_checkpoints".to_string(), metrics.anytime_checkpoints_metadata()),
        ("neighborhood_profile".to_string(), metrics.neighborhoods_metadata()),
    ]);
    if let Some(value) = metrics.time_to_first_feasible_nanos {
        metadata.push(("time_to_first_feasible".to_string(), (value as f64 / 1_000_000_000.0).to_string()));
    }
    if let Some(value) = &metrics.constructor {
        metadata.push(("constructor".to_string(), value.clone()));
    }
    if let Some(value) = metrics.constructor_fleet {
        metadata.push(("constructor_fleet".to_string(), value.to_string()));
    }
    if let Some(value) = metrics.constructor_cost {
        metadata.push(("constructor_cost".to_string(), value.to_string()));
    }
}

fn list_search_iterations(metrics: &lists::ListSearchMetrics) -> u64 {
    let slices = metrics.routing().slices;
    if slices == 0 {
        metrics.alns.iterations
    } else {
        slices
    }
}

fn record_progress(sink: &mut dyn EventSink, budget: &SolveBudget, event_error: &mut Option<SolveError>, engine: EngineKind, value: i64) {
    if event_error.is_some() {
        return;
    }
    let event = SolveEvent::Progress { engine, objectives: vec![value], elapsed: budget.elapsed() };
    match sink.emit(event) {
        Ok(EventControl::Continue) => {}
        Ok(EventControl::Stop) => budget.cancel_with(TerminationReason::EventSink),
        Err(error) => {
            *event_error = Some(error);
            budget.cancel_with(TerminationReason::EventSink);
        }
    }
}

fn warm_solution_from_verified(candidate: &CandidateSolution) -> Result<CollectionSolution, SolveError> {
    if !candidate.transferable() {
        return Err(SolveError::InvalidResult("exact routing received an incumbent that was not verified for transfer".to_string()));
    }
    if !candidate.assignment().integers.is_empty()
        || !candidate.assignment().sets.is_empty()
        || !candidate.assignment().intervals.is_empty()
    {
        return Err(SolveError::InvalidResult("exact routing received a transferred incumbent from a different model family".to_string()));
    }
    Ok(CollectionSolution {
        lists: candidate.assignment().lists.clone(),
        objectives: candidate.objectives().to_vec(),
        feasible: true,
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        modes: Vec::new(),
        bound: None,
    })
}

struct CollectionCompletion {
    solution: CollectionSolution,
    status: SolveStatus,
    source: EngineKind,
    proof: Option<ProofClaim>,
    report: EngineReport,
}

fn finish_collection_result(
    semantic: &Model,
    model: &list::CollectionModel,
    completion: CollectionCompletion,
    budget: &SolveBudget,
    transfer_stop: Option<&AtomicBool>,
) -> Result<SolveResult, SolveError> {
    let CollectionCompletion { solution, status, source, proof, report } = completion;
    let primal = if solution.feasible {
        #[cfg(test)]
        list::audit_record_final_verification_boundary();
        apply_final_replay_audit_before_first_pass(budget);
        apply_final_replay_audit_after_interrupt(budget);
        Some(if let Some(stop) = transfer_stop {
            verified_candidate(semantic, model, &solution, source, VerificationLevel::Final, stop)?
        } else {
            super::verify_final_with_budget(budget, |stop| {
                verified_candidate(semantic, model, &solution, source, VerificationLevel::Final, stop)
            })?
        })
    } else {
        None
    };
    let bounds =
        solution.bound.as_ref().map_or_else(Vec::new, |bound| vec![Bound { tier: 0, value: bound.dual, method: bound.method.clone() }]);
    Ok(SolveResult { status, primal, bounds, proof, reports: vec![report], message: None })
}

fn verified_candidate(
    semantic: &Model,
    model: &list::CollectionModel,
    solution: &CollectionSolution,
    source: EngineKind,
    verification: VerificationLevel,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<CandidateSolution, SolveError> {
    let assignment = if model.schedule.is_some() {
        let mut intervals = Vec::with_capacity(solution.starts.len());
        for (index, (((&start, &present), &machine), &mode)) in
            solution.starts.iter().zip(&solution.presences).zip(&solution.machines).zip(&solution.modes).enumerate()
        {
            if index & 0xff == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
                return Err(SolveError::Interrupted("schedule assignment decoding was interrupted".to_string()));
            }
            intervals.push(IntervalValue { start: present.then_some(start), present, machine: usize::try_from(machine).ok(), mode });
        }
        Assignment { integers: Vec::new(), sets: Vec::new(), lists: Vec::new(), intervals }
    } else {
        let mut lists = Vec::with_capacity(solution.lists.len());
        for (list_index, source) in solution.lists.iter().enumerate() {
            if list_index & 0x3f == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
                return Err(SolveError::Interrupted("list assignment decoding was interrupted".to_string()));
            }
            let mut list = Vec::with_capacity(source.len());
            for (item_index, &item) in source.iter().enumerate() {
                if item_index & 0xff == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
                    return Err(SolveError::Interrupted("list assignment decoding was interrupted".to_string()));
                }
                list.push(item);
            }
            lists.push(list);
        }
        Assignment { integers: Vec::new(), sets: Vec::new(), lists, intervals: Vec::new() }
    };
    // The semantic replay recompiles and checks the collection representation,
    // assignment, constraints, and objective vector. A separate physical replay
    // here would perform the same O(n) to O(n²) work twice at the deadline.
    let objectives = super::verify_semantic_assignment_validated_interruptible(semantic, &assignment, &solution.objectives, stop)?;
    Ok(CandidateSolution::verified(assignment, objectives, source, verification))
}

fn completion_proof(engine: EngineKind, status: SolveStatus, objective_tiers: usize, complete: bool) -> Option<ProofClaim> {
    if !complete {
        return None;
    }
    let conclusion = match status {
        SolveStatus::Optimal => ProvenConclusion::Optimal,
        SolveStatus::Unsatisfiable => ProvenConclusion::Unsatisfiable,
        SolveStatus::Satisfiable | SolveStatus::Unknown | SolveStatus::Unsupported => return None,
    };
    Some(ProofClaim::complete_search(engine, conclusion, objective_tiers))
}
