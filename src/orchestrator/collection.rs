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
use crate::search::SolveStats;

use super::{
    worker_iteration_quota, Assignment, Bound, CandidateSolution, EngineKind, EngineReport, EventControl, EventSink, IntervalValue,
    ModelFamily, ProofClaim, ProvenConclusion, RoutingControls, ScheduleJsspSearch, SolveBudget, SolveError, SolveEvent, SolveMode,
    SolveRequest, SolveResult, SolveStatus, TerminationReason, VerificationLevel,
};

const ROUTING_WARM_START_ITERATIONS: u64 = 2_000;
const AUTO_ORDERED_EXACT_ITEMS: usize = 10;
const AUTO_ASSIGNMENT_EXACT_ITEMS: usize = 24;
const AUTO_ASSIGNMENT_EXACT_CELLS: usize = 192;
const AUTO_SCHEDULE_EXACT_INTERVALS: usize = 48;
const AUTO_SCHEDULE_EXACT_MODES: usize = 96;
const MIN_SCHEDULE_LS_FINALIZATION_RESERVE: Duration = Duration::from_millis(250);
const MAX_SCHEDULE_LS_FINALIZATION_RESERVE: Duration = Duration::from_millis(2_500);
const SCHEDULE_LS_FINALIZATION_NANOS_PER_INTERVAL: u64 = 2_250;

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
    static SCHEDULE_SEARCH_ELITE_SHADOW_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static SCHEDULE_SEARCH_ELITE_STOP_BEFORE_MERGE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn audit_with_schedule_search_elite_shadow<R>(enabled: bool, audit: impl FnOnce() -> R) -> R {
    struct ResetOverride(Option<bool>);

    impl Drop for ResetOverride {
        fn drop(&mut self) {
            SCHEDULE_SEARCH_ELITE_SHADOW_OVERRIDE.set(self.0);
        }
    }

    let previous = SCHEDULE_SEARCH_ELITE_SHADOW_OVERRIDE.replace(Some(enabled));
    let _reset = ResetOverride(previous);
    audit()
}

#[cfg(test)]
pub(crate) fn audit_schedule_search_elite_shadow_default_enabled() -> bool {
    SCHEDULE_SEARCH_ELITE_SHADOW_OVERRIDE.get().is_none() && schedule_search_elite_shadow_enabled()
}

#[cfg(test)]
pub(crate) fn audit_with_schedule_search_elite_stop_before_merge<R>(audit: impl FnOnce() -> R) -> R {
    struct ResetStopBeforeMerge(bool);

    impl Drop for ResetStopBeforeMerge {
        fn drop(&mut self) {
            SCHEDULE_SEARCH_ELITE_STOP_BEFORE_MERGE.set(self.0);
        }
    }

    let previous = SCHEDULE_SEARCH_ELITE_STOP_BEFORE_MERGE.replace(true);
    let _reset = ResetStopBeforeMerge(previous);
    audit()
}

#[cfg(test)]
fn apply_schedule_search_elite_stop_before_merge(stop: &AtomicBool) {
    if SCHEDULE_SEARCH_ELITE_STOP_BEFORE_MERGE.replace(false) {
        stop.store(true, Ordering::Release);
    }
}

#[cfg(test)]
fn schedule_search_elite_shadow_enabled() -> bool {
    SCHEDULE_SEARCH_ELITE_SHADOW_OVERRIDE.get().unwrap_or(true)
}

#[cfg(not(test))]
fn schedule_search_elite_shadow_enabled() -> bool {
    true
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
    if !request.linear.node_time.is_zero() {
        return Err(SolveError::InvalidRequest(
            "in-search linear relaxations require an exact integer CP objective; collection plans support root bounds only".to_string(),
        ));
    }
    if request.sat != super::SatControls::default() {
        return Err(SolveError::InvalidRequest("SAT controls require a semantic Boolean clause model".to_string()));
    }
    if request.cp != super::CpControls::default() {
        return Err(SolveError::InvalidRequest("CP portfolio controls require a semantic integer model".to_string()));
    }
    if !request.search_policy.is_auto() {
        return Err(SolveError::InvalidRequest(
            "collection plans do not support an integer search_policy; use an exact integer CP model".to_string(),
        ));
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
    if request.schedule_constructor_workers != 0 {
        if !matches!(plan.backend, CollectionBackend::ScheduleLocalSearch) {
            return Err(SolveError::InvalidRequest(
                "schedule_constructor_workers requires a selected scheduling local-search backend".to_string(),
            ));
        }
        if request.limits.time.is_none() || request.limits.iterations.is_some() {
            return Err(SolveError::InvalidRequest(
                "schedule_constructor_workers requires a wall-clock search without a deterministic iteration limit".to_string(),
            ));
        }
    }
    if request.schedule_jssp_search == ScheduleJsspSearch::TsabCandidate && !matches!(plan.backend, CollectionBackend::ScheduleLocalSearch)
    {
        return Err(SolveError::InvalidRequest(
            "schedule_jssp_search='tsab-candidate' requires a selected scheduling local-search backend".to_string(),
        ));
    }
    if request.list_hint.is_some()
        && !matches!(plan.backend, CollectionBackend::ListLocalSearch | CollectionBackend::RoutingLocalSearch)
        && !plan.has_routing_warm_start()
    {
        return Err(SolveError::InvalidRequest(
            "list_hint requires list local search or a routing exact plan with warm start enabled".to_string(),
        ));
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
    let search_stop = optional_stops.is_none().then(|| {
        if matches!(plan.backend, CollectionBackend::ScheduleLocalSearch) {
            let interval_count = model.schedule.as_ref().map_or(0, |schedule| schedule.intervals.len());
            budget.search_stop_with_finalization_reserve(schedule_local_search_finalization_reserve(interval_count))
        } else {
            budget.search_stop()
        }
    });
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
                    let dual_task = scope.spawn(|| dual::compute_with_linear(model, request.linear, engine_stop));
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
                ScheduleLocalSearchRun {
                    semantic,
                    model,
                    request,
                    budget,
                    engine_stop,
                    transfer_stop,
                    max_iterations: plan.local_search_iterations.or(request.limits.iterations).unwrap_or(u64::MAX),
                    repeat_until_stopped: request.limits.time.is_some()
                        && plan.local_search_iterations.is_none()
                        && request.limits.iterations.is_none(),
                    search_elite_token: (request.profile && schedule_search_elite_shadow_enabled())
                        .then(lists::schedule_elite::ScheduleEliteSolveToken::new),
                    allocation,
                    started,
                },
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

pub(crate) fn schedule_local_search_finalization_reserve(interval_count: usize) -> Duration {
    // Small schedules use a 250 ms tail. The extra linear term
    // reaches the 2.5 s cap at one million intervals, where assignment decoding
    // and semantic replay are large enough to need the full transfer reserve.
    let interval_count = u64::try_from(interval_count).unwrap_or(u64::MAX);
    let scaled = Duration::from_nanos(interval_count.saturating_mul(SCHEDULE_LS_FINALIZATION_NANOS_PER_INTERVAL));
    MIN_SCHEDULE_LS_FINALIZATION_RESERVE.saturating_add(scaled).min(MAX_SCHEDULE_LS_FINALIZATION_RESERVE)
}

fn result_from_routing(
    semantic: &Model,
    model: &list::CollectionModel,
    mut outcome: routing::RoutingIntegerOutcome,
    structural_bound: Option<dual::DualBound>,
    elapsed: Duration,
    budget: &SolveBudget,
) -> Result<SolveResult, SolveError> {
    if let Some(bound) = &structural_bound {
        super::merge_search_stats(&mut outcome.stats, bound.stats);
    }
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
            verified_primal: None,
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
            verified_primal: None,
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
            verified_primal: None,
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
                dual::compute_with_linear(model, request.linear, engine_stop)
            })
        };
        let mut dual_released = false;
        let mut publish_first_incumbent = |value| {
            #[cfg(test)]
            {
                let progress_audit = LocalSearchProgressAudit::new();
                improvement(engine, value);
                drop(progress_audit);
            }
            #[cfg(not(test))]
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
    let mut search = structural_bound.as_ref().map_or_else(SolveStats::default, |bound| bound.stats);
    search.nodes = iterations;
    dual::attach(model, &mut solution, structural_bound);
    let status = if solution.feasible { SolveStatus::Satisfiable } else { SolveStatus::Unknown };
    let mut result = finish_collection_result(
        semantic,
        model,
        CollectionCompletion {
            solution,
            verified_primal: None,
            status,
            source: engine,
            proof: None,
            report: EngineReport { engine: Some(engine), search, elapsed: started.elapsed(), improvements: 0, metadata },
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
    max_iterations: u64,
    repeat_until_stopped: bool,
    search_elite_token: Option<lists::schedule_elite::ScheduleEliteSolveToken>,
    allocation: super::WorkerAllocation,
    started: Instant,
}

struct SharedScheduleIncumbent {
    solution: CollectionSolution,
    candidate: CandidateSolution,
    source_worker: usize,
    source_round: usize,
    constructor: &'static str,
}

struct ScheduleBoundaryCandidate {
    source_worker: usize,
    solution: CollectionSolution,
    constructor: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScheduleTransferAttempt {
    Accepted,
    Rejected,
    Interrupted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScheduleTransferAttemptSummary {
    accepted: Option<usize>,
    attempts: usize,
    interrupted: bool,
}

#[derive(Default)]
struct ScheduleMetricsAggregate {
    construction_candidates: u64,
    construction_elapsed: Duration,
    construction_bucket_visits: u64,
    construction_heap_pushes: u64,
    construction_stale_pops: u64,
    construction_heap_rebuilds: u64,
    construction_heap_peak: u64,
    constructor: Option<&'static str>,
    work_steps: u64,
    moves_considered: u64,
    moves_accepted: u64,
    incumbent_improvements: u64,
    incumbent_injections: u64,
    cycle_rejections: u64,
    window_rejections: u64,
    objective_rejections: u64,
    reconstructions: u64,
    critical_path_updates: u64,
    delta_evaluations: u64,
    full_evaluations: u64,
    full_fallbacks: u64,
    topological_rebuilds: u64,
    oracle_validations: u64,
    oracle_mismatches: u64,
    dirty_cone_operations: u64,
    max_dirty_cone: u64,
    workspace_growths: u64,
    workspace_rollbacks: u64,
    alns_generation_attempts: u64,
    alns_moves_generated: u64,
    resource_profile_checks: u64,
    resource_candidate_scheduling_attempts: u64,
    resource_event_visits: u64,
    resource_peak_profile_events: usize,
    precedence_rejections: u64,
    infeasible_rejections: u64,
    justification_attempts: u64,
    tabu_steps: u64,
    tabu_hits: u64,
    tabu_aspirations: u64,
    tabu_forced_moves: u64,
    schedule_tsab_owner_worker_mask: u64,
    schedule_tsab_n5_generated: u64,
    schedule_tsab_ranked: u64,
    schedule_tsab_shortlists: u64,
    schedule_tsab_delta_probes: u64,
    schedule_tsab_additional_delta_probes: u64,
    schedule_tsab_full_oracle_commits: u64,
    schedule_tsab_selected_shortlist_rank_sum: u64,
    schedule_tsab_selections: u64,
    schedule_tsab_aspirations: u64,
    schedule_tsab_tabu_rejections: u64,
    schedule_tsab_tabu_resets: u64,
    schedule_tsab_fingerprint_repeats: u64,
    schedule_tsab_escape_signals: u64,
    schedule_tsab_n1_kicks: u64,
    schedule_tsab_kick_moves: u64,
    schedule_tsab_elite_restarts: u64,
    schedule_tsab_n6_kicks: u64,
    schedule_tsab_restart_attempts: u64,
    schedule_tsab_restart_global_rebases: u64,
    schedule_tsab_restart_n6_generated: u64,
    schedule_tsab_restart_delta_probes: u64,
    schedule_tsab_restart_oracle_commits: u64,
    schedule_tsab_restart_rejections: u64,
    schedule_tsab_restart_interruptions: u64,
    schedule_tsab_restart_work_units: u64,
    schedule_tsab_restart_best_base_objective: Option<i64>,
    schedule_tsab_restart_best_kicked_objective: Option<i64>,
    schedule_tsab_post_restart_improvements: u64,
    schedule_tsab_restart_shortlist_peak_bytes: usize,
    schedule_tsab_ranking_audits: u64,
    schedule_tsab_exact_best_matches: u64,
    schedule_tsab_regret_sum: u64,
    schedule_tsab_regret_max: u64,
    schedule_tsab_workspace_peak_bytes: usize,
    schedule_tsab_activations: u64,
    schedule_tsab_activation_boundary: Option<u64>,
    schedule_tsab_legacy_warmup_work_steps: u64,
    schedule_tsab_activation_rebases: u64,
    schedule_tsab_activation_objective: Option<i64>,
    schedule_tsab_active_boundaries: u64,
    schedule_tsab_burst_work_limit: u64,
    schedule_tsab_burst_work_units: u64,
    schedule_tsab_improving_commits: u64,
    schedule_tsab_best_committed_objective: Option<i64>,
    schedule_tsab_fast_enabled: u64,
    schedule_tsab_fast_eligible: u64,
    schedule_tsab_fast_disabled: u64,
    schedule_tsab_fast_attempts: u64,
    schedule_tsab_fast_commits: u64,
    schedule_tsab_fast_fallbacks: u64,
    schedule_tsab_fast_date_changes: u64,
    schedule_tsab_fast_queue_pops: u64,
    schedule_tsab_fast_full_validations: u64,
    schedule_tsab_fast_oracle_mismatches: u64,
    schedule_tsab_fast_pending_promotions: u64,
    schedule_tsab_fast_pending_discards: u64,
    schedule_tsab_fast_transitions: u64,
    schedule_tsab_fast_work_units: u64,
    schedule_tsab_fast_elapsed: Duration,
    schedule_tsab_fast_workspace_peak_bytes: usize,
    session_initializations: u64,
    session_resumes: u64,
    session_rebases: u64,
    island_profile_mask: u64,
    baseline_island_profile_mask: u64,
    scored_island_profile_mask: u64,
    profile_construction_elapsed: [Duration; 8],
    profile_initial_objectives: [Option<i64>; 8],
    profile_initial_dispatch_rules: [Option<&'static str>; 8],
    profile_best_objectives: [Option<i64>; 8],
    profile_work_steps: [u64; 8],
    reactive_restarts: u64,
    reactive_restart_dispatches: u64,
    reactive_restart_perturbations: u64,
    reactive_restart_rebuild_failures: u64,
    island_scored_candidates: u64,
    island_shortlisted_candidates: u64,
    approximate_candidates_generated: u64,
    approximate_candidates_refined: u64,
    approximate_candidates_certified: u64,
    approximate_candidates_unknown: u64,
    approximation_score_items: u64,
    approximation_sort_items: u64,
    approximation_local_span_items: u64,
    approximation_elapsed: Duration,
    approximation_work_units: u64,
    direct_oracle_attempts: u64,
    direct_oracle_accepts: u64,
    direct_oracle_cycles: u64,
    direct_oracle_windows: u64,
    direct_oracle_objective_rejections: u64,
    exact_probes_avoided: u64,
    search_elite_snapshot_captures: u64,
    search_elite_snapshot_interruptions: u64,
    search_elite_snapshot_errors: u64,
    search_elite_capture_worker_elapsed_sum: Duration,
    search_elite_snapshot_peak_heap_lower_bound_bytes: usize,
    schedule_lns_shadow_owner_worker_mask: u64,
    schedule_lns: lists::schedule_lns::ScheduleLnsMetrics,
    schedule_lns_workspace_peak_bytes: usize,
    schedule_path_relink: lists::schedule_relink::ScheduleRelinkMetrics,
    schedule_constructor_multistart_owner_worker_mask: u64,
    schedule_constructor_multistart_attempts: u64,
    schedule_constructor_multistart_constructions: u64,
    schedule_constructor_multistart_interruptions: u64,
    schedule_constructor_multistart_failures: u64,
    schedule_constructor_multistart_feasible: u64,
    schedule_constructor_multistart_distinct_fingerprints: u64,
    schedule_constructor_multistart_other_fingerprint_observations: u64,
    schedule_constructor_multistart_initial_objective: Option<i64>,
    schedule_constructor_multistart_best_objective: Option<i64>,
    schedule_constructor_multistart_improvements: u64,
    schedule_constructor_multistart_work_units: u64,
    schedule_constructor_multistart_elapsed: Duration,
    schedule_constructor_multistart_workspace_peak_bytes: usize,
    schedule_constructor_multistart_best_ordinal: Option<u64>,
    schedule_constructor_multistart_best_seed: Option<u64>,
    schedule_constructor_multistart_best_fingerprint: Option<u64>,
    schedule_constructor_multistart_next_ordinal: Option<u64>,
}

impl ScheduleMetricsAggregate {
    fn record(&mut self, metrics: lists::ScheduleConstructionMetrics, local_objective: Option<i64>) {
        if let Some(profile) = metrics.island_profile {
            if let Some(work) = self.profile_work_steps.get_mut(profile) {
                *work = work.saturating_add(metrics.work_steps);
            }
            if let Some(elapsed) = self.profile_construction_elapsed.get_mut(profile) {
                *elapsed = elapsed.saturating_add(metrics.elapsed);
            }
            if let (Some(best), Some(objective)) = (self.profile_best_objectives.get_mut(profile), local_objective) {
                *best = Some(best.map_or(objective, |current| current.min(objective)));
            }
            if let (Some(initial), Some(objective)) = (self.profile_initial_objectives.get_mut(profile), metrics.initial_objective) {
                *initial = Some(initial.map_or(objective, |current| current.min(objective)));
            }
            if let (Some(initial), Some(rule)) = (self.profile_initial_dispatch_rules.get_mut(profile), metrics.initial_dispatch_rule) {
                *initial = Some(initial.map_or(rule, |current| current.min(rule)));
            }
        }
        self.construction_candidates = self.construction_candidates.saturating_add(metrics.candidates);
        self.construction_elapsed = self.construction_elapsed.max(metrics.elapsed);
        self.construction_bucket_visits = self.construction_bucket_visits.saturating_add(metrics.construction_bucket_visits);
        self.construction_heap_pushes = self.construction_heap_pushes.saturating_add(metrics.construction_heap_pushes);
        self.construction_stale_pops = self.construction_stale_pops.saturating_add(metrics.construction_stale_pops);
        self.construction_heap_rebuilds = self.construction_heap_rebuilds.saturating_add(metrics.construction_heap_rebuilds);
        self.construction_heap_peak = self.construction_heap_peak.max(metrics.construction_heap_peak);
        self.work_steps = self.work_steps.saturating_add(metrics.work_steps);
        self.moves_considered = self.moves_considered.saturating_add(metrics.moves_considered);
        self.moves_accepted = self.moves_accepted.saturating_add(metrics.moves_accepted);
        self.incumbent_improvements = self.incumbent_improvements.saturating_add(metrics.incumbent_improvements);
        self.incumbent_injections = self.incumbent_injections.saturating_add(metrics.incumbent_injections);
        self.cycle_rejections = self.cycle_rejections.saturating_add(metrics.cycle_rejections);
        self.window_rejections = self.window_rejections.saturating_add(metrics.window_rejections);
        self.objective_rejections = self.objective_rejections.saturating_add(metrics.objective_rejections);
        self.reconstructions = self.reconstructions.saturating_add(metrics.reconstructions);
        self.critical_path_updates = self.critical_path_updates.saturating_add(metrics.critical_path_updates);
        self.delta_evaluations = self.delta_evaluations.saturating_add(metrics.delta_evaluations);
        self.full_evaluations = self.full_evaluations.saturating_add(metrics.full_evaluations);
        self.full_fallbacks = self.full_fallbacks.saturating_add(metrics.full_fallbacks);
        self.topological_rebuilds = self.topological_rebuilds.saturating_add(metrics.topological_rebuilds);
        self.oracle_validations = self.oracle_validations.saturating_add(metrics.oracle_validations);
        self.oracle_mismatches = self.oracle_mismatches.saturating_add(metrics.oracle_mismatches);
        self.dirty_cone_operations = self.dirty_cone_operations.saturating_add(metrics.dirty_cone_operations);
        self.max_dirty_cone = self.max_dirty_cone.max(metrics.max_dirty_cone);
        self.workspace_growths = self.workspace_growths.saturating_add(metrics.workspace_growths);
        self.workspace_rollbacks = self.workspace_rollbacks.saturating_add(metrics.workspace_rollbacks);
        self.alns_generation_attempts = self.alns_generation_attempts.saturating_add(metrics.alns_generation_attempts);
        self.alns_moves_generated = self.alns_moves_generated.saturating_add(metrics.alns_moves_generated);
        self.resource_profile_checks = self.resource_profile_checks.saturating_add(metrics.resource_profile_checks);
        self.resource_candidate_scheduling_attempts =
            self.resource_candidate_scheduling_attempts.saturating_add(metrics.resource_candidate_scheduling_attempts);
        self.resource_event_visits = self.resource_event_visits.saturating_add(metrics.resource_event_visits);
        self.resource_peak_profile_events = self.resource_peak_profile_events.max(metrics.resource_peak_profile_events);
        self.precedence_rejections = self.precedence_rejections.saturating_add(metrics.precedence_rejections);
        self.infeasible_rejections = self.infeasible_rejections.saturating_add(metrics.infeasible_rejections);
        self.justification_attempts = self.justification_attempts.saturating_add(metrics.justification_attempts);
        self.tabu_steps = self.tabu_steps.saturating_add(metrics.tabu_steps);
        self.tabu_hits = self.tabu_hits.saturating_add(metrics.tabu_hits);
        self.tabu_aspirations = self.tabu_aspirations.saturating_add(metrics.tabu_aspirations);
        self.tabu_forced_moves = self.tabu_forced_moves.saturating_add(metrics.tabu_forced_moves);
        self.schedule_tsab_owner_worker_mask |= metrics.schedule_tsab.owner_worker_mask;
        self.schedule_tsab_n5_generated = self.schedule_tsab_n5_generated.saturating_add(metrics.schedule_tsab.n5_generated);
        self.schedule_tsab_ranked = self.schedule_tsab_ranked.saturating_add(metrics.schedule_tsab.ranked);
        self.schedule_tsab_shortlists = self.schedule_tsab_shortlists.saturating_add(metrics.schedule_tsab.shortlists);
        self.schedule_tsab_delta_probes = self.schedule_tsab_delta_probes.saturating_add(metrics.schedule_tsab.delta_probes);
        self.schedule_tsab_additional_delta_probes =
            self.schedule_tsab_additional_delta_probes.saturating_add(metrics.schedule_tsab.additional_delta_probes);
        self.schedule_tsab_full_oracle_commits =
            self.schedule_tsab_full_oracle_commits.saturating_add(metrics.schedule_tsab.full_oracle_commits);
        self.schedule_tsab_selected_shortlist_rank_sum =
            self.schedule_tsab_selected_shortlist_rank_sum.saturating_add(metrics.schedule_tsab.selected_shortlist_rank_sum);
        self.schedule_tsab_selections = self.schedule_tsab_selections.saturating_add(metrics.schedule_tsab.selections);
        self.schedule_tsab_aspirations = self.schedule_tsab_aspirations.saturating_add(metrics.schedule_tsab.aspirations);
        self.schedule_tsab_tabu_rejections = self.schedule_tsab_tabu_rejections.saturating_add(metrics.schedule_tsab.tabu_rejections);
        self.schedule_tsab_tabu_resets = self.schedule_tsab_tabu_resets.saturating_add(metrics.schedule_tsab.tabu_resets);
        self.schedule_tsab_fingerprint_repeats =
            self.schedule_tsab_fingerprint_repeats.saturating_add(metrics.schedule_tsab.fingerprint_repeats);
        self.schedule_tsab_escape_signals = self.schedule_tsab_escape_signals.saturating_add(metrics.schedule_tsab.escape_signals);
        self.schedule_tsab_n1_kicks = self.schedule_tsab_n1_kicks.saturating_add(metrics.schedule_tsab.n1_kicks);
        self.schedule_tsab_kick_moves = self.schedule_tsab_kick_moves.saturating_add(metrics.schedule_tsab.kick_moves);
        self.schedule_tsab_elite_restarts = self.schedule_tsab_elite_restarts.saturating_add(metrics.schedule_tsab.elite_restarts);
        self.schedule_tsab_n6_kicks = self.schedule_tsab_n6_kicks.saturating_add(metrics.schedule_tsab.n6_kicks);
        self.schedule_tsab_restart_attempts = self.schedule_tsab_restart_attempts.saturating_add(metrics.schedule_tsab.restart_attempts);
        self.schedule_tsab_restart_global_rebases =
            self.schedule_tsab_restart_global_rebases.saturating_add(metrics.schedule_tsab.restart_global_rebases);
        self.schedule_tsab_restart_n6_generated =
            self.schedule_tsab_restart_n6_generated.saturating_add(metrics.schedule_tsab.restart_n6_generated);
        self.schedule_tsab_restart_delta_probes =
            self.schedule_tsab_restart_delta_probes.saturating_add(metrics.schedule_tsab.restart_delta_probes);
        self.schedule_tsab_restart_oracle_commits =
            self.schedule_tsab_restart_oracle_commits.saturating_add(metrics.schedule_tsab.restart_oracle_commits);
        self.schedule_tsab_restart_rejections =
            self.schedule_tsab_restart_rejections.saturating_add(metrics.schedule_tsab.restart_rejections);
        self.schedule_tsab_restart_interruptions =
            self.schedule_tsab_restart_interruptions.saturating_add(metrics.schedule_tsab.restart_interruptions);
        self.schedule_tsab_restart_work_units =
            self.schedule_tsab_restart_work_units.saturating_add(metrics.schedule_tsab.restart_work_units);
        if let Some(objective) = metrics.schedule_tsab.restart_best_base_objective {
            self.schedule_tsab_restart_best_base_objective =
                Some(self.schedule_tsab_restart_best_base_objective.map_or(objective, |current| current.min(objective)));
        }
        if let Some(objective) = metrics.schedule_tsab.restart_best_kicked_objective {
            self.schedule_tsab_restart_best_kicked_objective =
                Some(self.schedule_tsab_restart_best_kicked_objective.map_or(objective, |current| current.min(objective)));
        }
        self.schedule_tsab_post_restart_improvements =
            self.schedule_tsab_post_restart_improvements.saturating_add(metrics.schedule_tsab.post_restart_improvements);
        self.schedule_tsab_restart_shortlist_peak_bytes =
            self.schedule_tsab_restart_shortlist_peak_bytes.max(metrics.schedule_tsab.restart_shortlist_peak_bytes);
        self.schedule_tsab_ranking_audits = self.schedule_tsab_ranking_audits.saturating_add(metrics.schedule_tsab.ranking_audits);
        self.schedule_tsab_exact_best_matches =
            self.schedule_tsab_exact_best_matches.saturating_add(metrics.schedule_tsab.exact_best_matches);
        self.schedule_tsab_regret_sum = self.schedule_tsab_regret_sum.saturating_add(metrics.schedule_tsab.regret_sum);
        self.schedule_tsab_regret_max = self.schedule_tsab_regret_max.max(metrics.schedule_tsab.regret_max);
        self.schedule_tsab_workspace_peak_bytes = self.schedule_tsab_workspace_peak_bytes.max(metrics.schedule_tsab.workspace_peak_bytes);
        self.schedule_tsab_activations = self.schedule_tsab_activations.saturating_add(metrics.schedule_tsab.activations);
        if let Some(boundary) = metrics.schedule_tsab.activation_boundary {
            self.schedule_tsab_activation_boundary.get_or_insert(boundary);
        }
        self.schedule_tsab_legacy_warmup_work_steps =
            self.schedule_tsab_legacy_warmup_work_steps.saturating_add(metrics.schedule_tsab.legacy_warmup_work_steps);
        self.schedule_tsab_activation_rebases =
            self.schedule_tsab_activation_rebases.saturating_add(metrics.schedule_tsab.activation_rebases);
        if let Some(objective) = metrics.schedule_tsab.activation_objective {
            self.schedule_tsab_activation_objective.get_or_insert(objective);
        }
        self.schedule_tsab_active_boundaries = self.schedule_tsab_active_boundaries.saturating_add(metrics.schedule_tsab.active_boundaries);
        self.schedule_tsab_burst_work_limit = self.schedule_tsab_burst_work_limit.max(metrics.schedule_tsab.burst_work_limit);
        self.schedule_tsab_burst_work_units = self.schedule_tsab_burst_work_units.saturating_add(metrics.schedule_tsab.burst_work_units);
        self.schedule_tsab_improving_commits = self.schedule_tsab_improving_commits.saturating_add(metrics.schedule_tsab.improving_commits);
        if let Some(objective) = metrics.schedule_tsab.best_committed_objective {
            self.schedule_tsab_best_committed_objective =
                Some(self.schedule_tsab_best_committed_objective.map_or(objective, |current| current.min(objective)));
        }
        self.schedule_tsab_fast_enabled = self.schedule_tsab_fast_enabled.saturating_add(metrics.schedule_tsab.fast_enabled);
        self.schedule_tsab_fast_eligible = self.schedule_tsab_fast_eligible.saturating_add(metrics.schedule_tsab.fast_eligible);
        self.schedule_tsab_fast_disabled = self.schedule_tsab_fast_disabled.saturating_add(metrics.schedule_tsab.fast_disabled);
        self.schedule_tsab_fast_attempts = self.schedule_tsab_fast_attempts.saturating_add(metrics.schedule_tsab.fast_attempts);
        self.schedule_tsab_fast_commits = self.schedule_tsab_fast_commits.saturating_add(metrics.schedule_tsab.fast_commits);
        self.schedule_tsab_fast_fallbacks = self.schedule_tsab_fast_fallbacks.saturating_add(metrics.schedule_tsab.fast_fallbacks);
        self.schedule_tsab_fast_date_changes = self.schedule_tsab_fast_date_changes.saturating_add(metrics.schedule_tsab.fast_date_changes);
        self.schedule_tsab_fast_queue_pops = self.schedule_tsab_fast_queue_pops.saturating_add(metrics.schedule_tsab.fast_queue_pops);
        self.schedule_tsab_fast_full_validations =
            self.schedule_tsab_fast_full_validations.saturating_add(metrics.schedule_tsab.fast_full_validations);
        self.schedule_tsab_fast_oracle_mismatches =
            self.schedule_tsab_fast_oracle_mismatches.saturating_add(metrics.schedule_tsab.fast_oracle_mismatches);
        self.schedule_tsab_fast_pending_promotions =
            self.schedule_tsab_fast_pending_promotions.saturating_add(metrics.schedule_tsab.fast_pending_promotions);
        self.schedule_tsab_fast_pending_discards =
            self.schedule_tsab_fast_pending_discards.saturating_add(metrics.schedule_tsab.fast_pending_discards);
        self.schedule_tsab_fast_transitions = self.schedule_tsab_fast_transitions.saturating_add(metrics.schedule_tsab.fast_transitions);
        self.schedule_tsab_fast_work_units = self.schedule_tsab_fast_work_units.saturating_add(metrics.schedule_tsab.fast_work_units);
        self.schedule_tsab_fast_elapsed = self.schedule_tsab_fast_elapsed.saturating_add(metrics.schedule_tsab.fast_elapsed);
        self.schedule_tsab_fast_workspace_peak_bytes =
            self.schedule_tsab_fast_workspace_peak_bytes.max(metrics.schedule_tsab.fast_workspace_peak_bytes);
        self.session_initializations = self.session_initializations.saturating_add(metrics.session_initializations);
        self.session_resumes = self.session_resumes.saturating_add(metrics.session_resumes);
        self.session_rebases = self.session_rebases.saturating_add(metrics.session_rebases);
        self.island_profile_mask |= metrics.island_profile_mask;
        self.baseline_island_profile_mask |= metrics.baseline_island_profile_mask;
        self.scored_island_profile_mask |= metrics.scored_island_profile_mask;
        self.reactive_restarts = self.reactive_restarts.saturating_add(metrics.reactive_restarts);
        self.reactive_restart_dispatches = self.reactive_restart_dispatches.saturating_add(metrics.reactive_restart_dispatches);
        self.reactive_restart_perturbations = self.reactive_restart_perturbations.saturating_add(metrics.reactive_restart_perturbations);
        self.reactive_restart_rebuild_failures =
            self.reactive_restart_rebuild_failures.saturating_add(metrics.reactive_restart_rebuild_failures);
        self.island_scored_candidates = self.island_scored_candidates.saturating_add(metrics.island_scored_candidates);
        self.island_shortlisted_candidates = self.island_shortlisted_candidates.saturating_add(metrics.island_shortlisted_candidates);
        self.approximate_candidates_generated =
            self.approximate_candidates_generated.saturating_add(metrics.approximate_candidates_generated);
        self.approximate_candidates_refined = self.approximate_candidates_refined.saturating_add(metrics.approximate_candidates_refined);
        self.approximate_candidates_certified =
            self.approximate_candidates_certified.saturating_add(metrics.approximate_candidates_certified);
        self.approximate_candidates_unknown = self.approximate_candidates_unknown.saturating_add(metrics.approximate_candidates_unknown);
        self.approximation_score_items = self.approximation_score_items.saturating_add(metrics.approximation_score_items);
        self.approximation_sort_items = self.approximation_sort_items.saturating_add(metrics.approximation_sort_items);
        self.approximation_local_span_items = self.approximation_local_span_items.saturating_add(metrics.approximation_local_span_items);
        self.approximation_elapsed = self.approximation_elapsed.saturating_add(metrics.approximation_elapsed);
        self.approximation_work_units = self.approximation_work_units.saturating_add(metrics.approximation_work_units);
        self.direct_oracle_attempts = self.direct_oracle_attempts.saturating_add(metrics.direct_oracle_attempts);
        self.direct_oracle_accepts = self.direct_oracle_accepts.saturating_add(metrics.direct_oracle_accepts);
        self.direct_oracle_cycles = self.direct_oracle_cycles.saturating_add(metrics.direct_oracle_cycles);
        self.direct_oracle_windows = self.direct_oracle_windows.saturating_add(metrics.direct_oracle_windows);
        self.direct_oracle_objective_rejections =
            self.direct_oracle_objective_rejections.saturating_add(metrics.direct_oracle_objective_rejections);
        self.exact_probes_avoided = self.exact_probes_avoided.saturating_add(metrics.exact_probes_avoided);
        self.search_elite_snapshot_captures = self.search_elite_snapshot_captures.saturating_add(metrics.search_elite_snapshot_captures);
        self.search_elite_snapshot_interruptions =
            self.search_elite_snapshot_interruptions.saturating_add(metrics.search_elite_snapshot_interruptions);
        self.search_elite_snapshot_errors = self.search_elite_snapshot_errors.saturating_add(metrics.search_elite_snapshot_errors);
        self.search_elite_capture_worker_elapsed_sum =
            self.search_elite_capture_worker_elapsed_sum.saturating_add(metrics.search_elite_capture_worker_elapsed_sum);
        self.search_elite_snapshot_peak_heap_lower_bound_bytes =
            self.search_elite_snapshot_peak_heap_lower_bound_bytes.max(metrics.search_elite_snapshot_peak_heap_lower_bound_bytes);
        self.schedule_lns_shadow_owner_worker_mask |= metrics.schedule_lns_shadow_owner_worker_mask;
        self.schedule_lns.add(metrics.schedule_lns);
        self.schedule_lns_workspace_peak_bytes = self.schedule_lns_workspace_peak_bytes.max(metrics.schedule_lns_workspace_peak_bytes);
        self.schedule_path_relink.add(metrics.schedule_path_relink);
        self.schedule_constructor_multistart_owner_worker_mask |= metrics.schedule_constructor_multistart_owner_worker_mask;
        self.schedule_constructor_multistart_attempts =
            self.schedule_constructor_multistart_attempts.saturating_add(metrics.schedule_constructor_multistart_attempts);
        self.schedule_constructor_multistart_constructions =
            self.schedule_constructor_multistart_constructions.saturating_add(metrics.schedule_constructor_multistart_constructions);
        self.schedule_constructor_multistart_interruptions =
            self.schedule_constructor_multistart_interruptions.saturating_add(metrics.schedule_constructor_multistart_interruptions);
        self.schedule_constructor_multistart_failures =
            self.schedule_constructor_multistart_failures.saturating_add(metrics.schedule_constructor_multistart_failures);
        self.schedule_constructor_multistart_feasible =
            self.schedule_constructor_multistart_feasible.saturating_add(metrics.schedule_constructor_multistart_feasible);
        self.schedule_constructor_multistart_distinct_fingerprints = self
            .schedule_constructor_multistart_distinct_fingerprints
            .saturating_add(metrics.schedule_constructor_multistart_distinct_fingerprints);
        self.schedule_constructor_multistart_other_fingerprint_observations = self
            .schedule_constructor_multistart_other_fingerprint_observations
            .saturating_add(metrics.schedule_constructor_multistart_other_fingerprint_observations);
        if let Some(initial) = metrics.schedule_constructor_multistart_initial_objective {
            self.schedule_constructor_multistart_initial_objective.get_or_insert(initial);
        }
        if let Some(best) = metrics.schedule_constructor_multistart_best_objective {
            if self.schedule_constructor_multistart_best_objective.is_none_or(|current| best < current) {
                self.schedule_constructor_multistart_best_objective = Some(best);
                self.schedule_constructor_multistart_best_ordinal = metrics.schedule_constructor_multistart_best_ordinal;
                self.schedule_constructor_multistart_best_seed = metrics.schedule_constructor_multistart_best_seed;
                self.schedule_constructor_multistart_best_fingerprint = metrics.schedule_constructor_multistart_best_fingerprint;
            }
        }
        self.schedule_constructor_multistart_improvements =
            self.schedule_constructor_multistart_improvements.saturating_add(metrics.schedule_constructor_multistart_improvements);
        self.schedule_constructor_multistart_work_units =
            self.schedule_constructor_multistart_work_units.saturating_add(metrics.schedule_constructor_multistart_work_units);
        self.schedule_constructor_multistart_elapsed =
            self.schedule_constructor_multistart_elapsed.saturating_add(metrics.schedule_constructor_multistart_elapsed);
        self.schedule_constructor_multistart_workspace_peak_bytes =
            self.schedule_constructor_multistart_workspace_peak_bytes.max(metrics.schedule_constructor_multistart_workspace_peak_bytes);
        if let Some(next) = metrics.schedule_constructor_multistart_next_ordinal {
            self.schedule_constructor_multistart_next_ordinal =
                Some(self.schedule_constructor_multistart_next_ordinal.map_or(next, |current| current.max(next)));
        }
        self.constructor.get_or_insert(metrics.constructor);
    }
}

#[derive(Default)]
struct ScheduleSearchEliteShadow {
    archive: lists::schedule_elite::ScheduleEliteArchive,
    batches: u64,
    batches_skipped_after_stop: u64,
    candidates: u64,
    insertions: u64,
    duplicates: u64,
    dominated: u64,
    evictions: u64,
    interruptions: u64,
    merge_errors: u64,
    merge_wall_elapsed: Duration,
    peak_heap_lower_bound_bytes: usize,
}

impl ScheduleSearchEliteShadow {
    fn record_merge_failure(&mut self, error: lists::schedule_elite::ScheduleEliteError) {
        if error == lists::schedule_elite::ScheduleEliteError::Interrupted {
            self.interruptions = self.interruptions.saturating_add(1);
        } else {
            self.merge_errors = self.merge_errors.saturating_add(1);
        }
    }

    fn merge(
        &mut self,
        reference: Option<&lists::ScheduleSearchSession>,
        candidates: Vec<lists::schedule_elite::ScheduleEliteCandidate>,
        stop: &AtomicBool,
    ) {
        if candidates.is_empty() {
            return;
        }
        if stop.load(Ordering::Acquire) {
            self.batches_skipped_after_stop = self.batches_skipped_after_stop.saturating_add(1);
            self.peak_heap_lower_bound_bytes =
                self.peak_heap_lower_bound_bytes.max(self.archive.pending_candidates_heap_lower_bound(&candidates, candidates.capacity()));
            return;
        }
        self.batches = self.batches.saturating_add(1);
        self.candidates = self.candidates.saturating_add(u64::try_from(candidates.len()).unwrap_or(u64::MAX));
        let identity_bytes_before = self.archive.identity_heap_lower_bound_bytes();
        let batch_peak_heap_lower_bound = self.archive.candidate_batch_peak_heap_lower_bound(&candidates, candidates.capacity());
        self.peak_heap_lower_bound_bytes = self.peak_heap_lower_bound_bytes.max(batch_peak_heap_lower_bound);
        let started = Instant::now();
        if let Some(reference) = reference {
            match reference.merge_search_elite_batch(&mut self.archive, candidates, stop) {
                Ok(outcome) => {
                    self.insertions = self.insertions.saturating_add(u64::try_from(outcome.retained).unwrap_or(u64::MAX));
                    self.duplicates = self.duplicates.saturating_add(u64::try_from(outcome.duplicates).unwrap_or(u64::MAX));
                    self.dominated = self.dominated.saturating_add(u64::try_from(outcome.dominated).unwrap_or(u64::MAX));
                    self.evictions = self.evictions.saturating_add(u64::try_from(outcome.evicted).unwrap_or(u64::MAX));
                }
                Err(error) => self.record_merge_failure(error),
            }
        } else {
            self.merge_errors = self.merge_errors.saturating_add(1);
        }
        self.merge_wall_elapsed = self.merge_wall_elapsed.saturating_add(started.elapsed());
        let identity_growth = self.archive.identity_heap_lower_bound_bytes().saturating_sub(identity_bytes_before);
        self.peak_heap_lower_bound_bytes =
            self.peak_heap_lower_bound_bytes.max(batch_peak_heap_lower_bound.saturating_add(identity_growth));
        self.peak_heap_lower_bound_bytes = self.peak_heap_lower_bound_bytes.max(self.archive.heap_lower_bound_bytes());
    }
}

#[cfg(test)]
pub(crate) fn audit_schedule_search_elite_merge_error_classification(error: lists::schedule_elite::ScheduleEliteError) -> (u64, u64) {
    let mut shadow = ScheduleSearchEliteShadow::default();
    shadow.record_merge_failure(error);
    (shadow.interruptions, shadow.merge_errors)
}

pub(crate) fn schedule_restart_work(interval_count: usize, quota: u64) -> u64 {
    const TARGET_INTERVAL_WORK_PER_BOUNDARY: u128 = 256 * 1_000_000;

    let interval_count_u128 = u128::try_from(interval_count.max(1)).unwrap_or(u128::MAX);
    let legacy = u64::try_from(interval_count).unwrap_or(u64::MAX).saturating_mul(256).clamp(2_048, 100_000);
    let adaptive_cap = u64::try_from(TARGET_INTERVAL_WORK_PER_BOUNDARY / interval_count_u128).unwrap_or(u64::MAX).clamp(256, 100_000);
    quota.min(legacy.min(adaptive_cap))
}

fn schedule_round_seed(seed: u64, round: usize) -> u64 {
    if round == 0 {
        seed
    } else {
        crate::mix64(seed ^ u64::try_from(round).unwrap_or(u64::MAX).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }
}

fn schedule_candidate_precedes(
    candidate: &CollectionSolution,
    source_worker: usize,
    source_round: usize,
    incumbent: &SharedScheduleIncumbent,
) -> bool {
    schedule_candidate_order(candidate, source_worker, source_round, &incumbent.solution, incumbent.source_worker, incumbent.source_round)
        == std::cmp::Ordering::Less
}

fn schedule_candidate_order(
    left: &CollectionSolution,
    left_worker: usize,
    left_round: usize,
    right: &CollectionSolution,
    right_worker: usize,
    right_round: usize,
) -> std::cmp::Ordering {
    schedule_candidate_key_order(&left.objectives, left_worker, left_round, &right.objectives, right_worker, right_round)
}

fn schedule_candidate_key_order(
    left_objectives: &[i64],
    left_worker: usize,
    left_round: usize,
    right_objectives: &[i64],
    right_worker: usize,
    right_round: usize,
) -> std::cmp::Ordering {
    left_objectives.cmp(right_objectives).then_with(|| (left_worker, left_round).cmp(&(right_worker, right_round)))
}

fn attempt_schedule_transfer_batch(
    candidates: &[usize],
    mut attempt: impl FnMut(usize) -> ScheduleTransferAttempt,
) -> ScheduleTransferAttemptSummary {
    for (position, &candidate) in candidates.iter().enumerate() {
        match attempt(candidate) {
            ScheduleTransferAttempt::Accepted => {
                return ScheduleTransferAttemptSummary { accepted: Some(candidate), attempts: position + 1, interrupted: false };
            }
            ScheduleTransferAttempt::Rejected => {}
            ScheduleTransferAttempt::Interrupted => {
                return ScheduleTransferAttemptSummary { accepted: None, attempts: position + 1, interrupted: true };
            }
        }
    }
    ScheduleTransferAttemptSummary { accepted: None, attempts: candidates.len(), interrupted: false }
}

#[cfg(test)]
pub(crate) fn audit_schedule_transfer_attempts(
    candidate_count: usize,
    rejected: &[usize],
    interrupt_at: Option<usize>,
) -> (Vec<usize>, Option<usize>, bool) {
    let candidates = (0..candidate_count).collect::<Vec<_>>();
    let mut attempts = Vec::new();
    let summary = attempt_schedule_transfer_batch(&candidates, |candidate| {
        attempts.push(candidate);
        if interrupt_at == Some(candidate) {
            ScheduleTransferAttempt::Interrupted
        } else if rejected.contains(&candidate) {
            ScheduleTransferAttempt::Rejected
        } else {
            ScheduleTransferAttempt::Accepted
        }
    });
    (attempts, summary.accepted, summary.interrupted)
}

#[cfg(test)]
pub(crate) fn audit_schedule_transfer_order(mut candidates: Vec<(usize, usize, Vec<i64>)>) -> Vec<(usize, usize, Vec<i64>)> {
    candidates.sort_by(|left, right| schedule_candidate_key_order(&left.2, left.0, left.1, &right.2, right.0, right.1));
    candidates
}

fn schedule_path_relink_guide<'a>(
    archive: &'a lists::schedule_elite::ScheduleEliteArchive,
    seed: u64,
    round: usize,
) -> Option<lists::schedule_relink::ScheduleRelinkRequest<'a>> {
    let (kind, index) = schedule_path_relink_guide_choice(seed, round, archive.len())?;
    Some(lists::schedule_relink::ScheduleRelinkRequest { guide: &archive.entries()[index], kind })
}

fn schedule_path_relink_guide_choice(
    seed: u64,
    round: usize,
    entries: usize,
) -> Option<(lists::schedule_relink::ScheduleRelinkGuideKind, usize)> {
    if entries < 2 {
        return None;
    }
    let round = u64::try_from(round).unwrap_or(u64::MAX);
    let phase = crate::mix64(seed) & 1;
    if (round ^ phase) & 1 == 0 {
        return Some((lists::schedule_relink::ScheduleRelinkGuideKind::Best, 0));
    }
    let diverse = entries.saturating_sub(1);
    let selection = crate::mix64(seed ^ round.wrapping_mul(0xa076_1d64_78bd_642f) ^ 0xe703_7ed1_a0b4_28db);
    let index = 1 + usize::try_from(selection).unwrap_or(usize::MAX) % diverse;
    Some((lists::schedule_relink::ScheduleRelinkGuideKind::Diverse, index))
}

#[cfg(test)]
pub(crate) fn audit_schedule_path_relink_guide_choices(seed: u64, rounds: usize, entries: usize) -> Vec<(bool, usize)> {
    (0..rounds)
        .filter_map(|round| {
            schedule_path_relink_guide_choice(seed, round, entries)
                .map(|(kind, index)| (kind == lists::schedule_relink::ScheduleRelinkGuideKind::Best, index))
        })
        .collect()
}

fn result_from_schedule_local_search(
    run: ScheduleLocalSearchRun<'_>,
    improvement: &mut impl FnMut(EngineKind, i64),
) -> Result<SolveResult, SolveError> {
    let ScheduleLocalSearchRun {
        semantic,
        model,
        request,
        budget,
        engine_stop,
        transfer_stop,
        max_iterations,
        repeat_until_stopped,
        search_elite_token,
        allocation,
        started,
    } = run;
    let schedule = model
        .schedule
        .as_ref()
        .ok_or_else(|| SolveError::InvalidResult("schedule local-search plan has no physical schedule".to_string()))?;
    if request.schedule_constructor_workers != 0 && !lists::schedule_constructor_multistart_supported(schedule, engine_stop) {
        return Err(if engine_stop.load(Ordering::Acquire) {
            SolveError::Interrupted("solve budget expired while validating schedule constructor multi-start".to_string())
        } else {
            SolveError::InvalidRequest(
                "schedule_constructor_workers currently supports only mandatory fixed-assignment JSSP local search".to_string(),
            )
        });
    }
    if request.schedule_jssp_search == ScheduleJsspSearch::TsabCandidate
        && !lists::schedule_constructor_multistart_supported(schedule, engine_stop)
    {
        return Err(if engine_stop.load(Ordering::Acquire) {
            SolveError::Interrupted("solve budget expired while validating TsabCandidate JSSP search".to_string())
        } else {
            SolveError::InvalidRequest(
                "schedule_jssp_search='tsab-candidate' currently supports only mandatory fixed-assignment JSSP local search".to_string(),
            )
        });
    }
    let restart_work_cap = schedule_restart_work(schedule.intervals.len(), u64::MAX);
    let mut aggregate = ScheduleMetricsAggregate::default();
    let (
        shared_incumbent,
        structural_bound,
        progress_publications,
        incumbent_publications,
        incumbent_injection_attempts,
        incumbent_rejections,
        incumbent_verifications,
        verification_rejections,
        verification_interruptions,
        incumbent_verification_elapsed,
        incumbent_verification_max_elapsed,
        incomplete_rejections,
        restart_boundaries,
        peak_buffered_candidates,
        first_feasible,
        stalled_workers,
        stalled_unused_work_steps,
        worker_work_min,
        worker_work_max,
        work_budget_overruns,
        search_elite,
    ) = std::thread::scope(|scope| {
        let dual_task = scope.spawn(|| dual::compute(model, engine_stop));
        let mut shared_incumbent: Option<SharedScheduleIncumbent> = None;
        let mut published_objective = None;
        let mut progress_publications = 0u64;
        let mut incumbent_publications = 0u64;
        let mut incumbent_injection_attempts = 0u64;
        let mut incumbent_rejections = 0u64;
        let mut incumbent_verifications = 0u64;
        let mut verification_rejections = 0u64;
        let mut verification_interruptions = 0u64;
        let mut incumbent_verification_elapsed = Duration::ZERO;
        let mut incumbent_verification_max_elapsed = Duration::ZERO;
        let mut incomplete_rejections = 0u64;
        let mut restart_boundaries = 0u64;
        let mut peak_buffered_candidates = 0usize;
        let mut first_feasible = None;
        let mut stalled_workers = 0u64;
        let mut stalled_unused_work_steps = 0u64;
        let mut work_budget_overruns = 0u64;
        let mut search_elite = ScheduleSearchEliteShadow::default();
        let finite_work = max_iterations != u64::MAX;
        let schedule_lns_shadow = request.schedule_lns_shadow && request.limits.iterations.is_none() && request.limits.time.is_some();
        let schedule_constructor_multistart = request.schedule_constructor_workers == 1;
        let schedule_jssp_strategy = match request.schedule_jssp_search {
            ScheduleJsspSearch::Legacy => lists::ScheduleJsspSearchStrategy::Legacy,
            ScheduleJsspSearch::TsabCandidate => lists::ScheduleJsspSearchStrategy::TsabCandidate,
        };
        let mut remaining = (0..allocation.workers())
            .map(|worker| worker_iteration_quota(max_iterations, worker, allocation.workers()))
            .collect::<Vec<_>>();
        let mut retired = vec![false; allocation.workers()];
        let mut zero_progress_rounds = vec![0u8; allocation.workers()];
        let mut worker_work = vec![0u64; allocation.workers()];
        let mut worker_sessions = (0..allocation.workers()).map(|_| None::<lists::ScheduleSearchSession>).collect::<Vec<_>>();
        let mut round = 0usize;

        loop {
            if engine_stop.load(Ordering::Acquire) || (!finite_work && !repeat_until_stopped && round > 0) {
                break;
            }
            let inputs = remaining
                .iter()
                .enumerate()
                .filter_map(|(worker, &left)| {
                    let worker_quota = if finite_work { left } else { u64::MAX };
                    let quota = schedule_restart_work(schedule.intervals.len(), worker_quota);
                    (!retired[worker] && quota > 0).then_some((worker, quota))
                })
                .collect::<Vec<_>>();
            if inputs.is_empty() {
                break;
            }
            let injected = shared_incumbent.as_ref().map(|incumbent| incumbent.solution.clone());
            if injected.is_some() {
                incumbent_injection_attempts = incumbent_injection_attempts.saturating_add(u64::try_from(inputs.len()).unwrap_or(u64::MAX));
            }
            let mut prepared_inputs = Vec::with_capacity(inputs.len());
            for (source_worker, quota) in inputs {
                prepared_inputs.push((source_worker, quota, worker_sessions[source_worker].take()));
            }
            let round_seed = schedule_round_seed(request.seed, round);
            let relink_guide = (request.profile && request.schedule_path_relink && allocation.workers() >= 7)
                .then(|| schedule_path_relink_guide(&search_elite.archive, request.seed, round))
                .flatten();
            let summary = super::execute_workers_silent(
                prepared_inputs,
                engine_stop,
                Arc::new(AtomicBool::new(false)),
                round_seed,
                |context, (source_worker, quota, session)| {
                    let worker = u64::try_from(source_worker).unwrap_or(u64::MAX);
                    let stable_seed = crate::mix64(request.seed ^ worker.wrapping_mul(0x9e37_79b9_7f4a_7c15));
                    let fallback_seed = round_seed.wrapping_add(worker);
                    let worker_relink = (source_worker == 6).then_some(relink_guide).flatten();
                    let (candidate, metrics, session) = lists::solve_schedule_capped_persistent_hybrid_with_strategy(
                        schedule,
                        stable_seed,
                        fallback_seed,
                        source_worker,
                        allocation.workers(),
                        context.stop(),
                        quota,
                        injected.as_ref(),
                        session,
                        source_worker == 0
                            || (source_worker == 6 && schedule_jssp_strategy == lists::ScheduleJsspSearchStrategy::TsabCandidate),
                        search_elite_token.as_ref(),
                        schedule_lns_shadow,
                        schedule_constructor_multistart,
                        schedule_jssp_strategy,
                        worker_relink,
                        &mut |_| {},
                    );
                    (source_worker, quota, candidate, metrics, session)
                },
            );
            peak_buffered_candidates = peak_buffered_candidates.max(summary.reports.len());
            restart_boundaries = restart_boundaries.saturating_add(1);

            let mut boundary_candidates = Vec::with_capacity(summary.reports.len());
            let mut reported_workers = Vec::with_capacity(summary.reports.len());
            for report in summary.reports {
                let (source_worker, quota, candidate, metrics, returned_session): (
                    usize,
                    u64,
                    CollectionSolution,
                    lists::ScheduleConstructionMetrics,
                    Option<lists::ScheduleSearchSession>,
                ) = report.result;
                let local_objective =
                    if candidate.feasible && candidate.objectives.len() == 1 { candidate.objectives.first().copied() } else { None };
                let constructor = metrics.constructor;
                let used = metrics.work_steps;
                worker_work[source_worker] = worker_work[source_worker].saturating_add(used);
                if finite_work {
                    remaining[source_worker] = remaining[source_worker].saturating_sub(used.min(quota));
                    if used > quota {
                        work_budget_overruns = work_budget_overruns.saturating_add(1);
                        remaining[source_worker] = 0;
                    }
                }
                if used == 0 {
                    zero_progress_rounds[source_worker] = zero_progress_rounds[source_worker].saturating_add(1);
                    if candidate.feasible || zero_progress_rounds[source_worker] >= 2 {
                        retired[source_worker] = true;
                        stalled_workers = stalled_workers.saturating_add(1);
                        if finite_work {
                            stalled_unused_work_steps = stalled_unused_work_steps.saturating_add(remaining[source_worker]);
                            remaining[source_worker] = 0;
                        }
                    }
                } else {
                    zero_progress_rounds[source_worker] = 0;
                }
                if candidate.feasible {
                    boundary_candidates.push(ScheduleBoundaryCandidate { source_worker, solution: candidate, constructor });
                } else {
                    incumbent_rejections = incumbent_rejections.saturating_add(1);
                    incomplete_rejections = incomplete_rejections.saturating_add(1);
                }
                aggregate.record(metrics, local_objective);
                reported_workers.push(source_worker);
                worker_sessions[source_worker] = returned_session;
            }

            boundary_candidates.sort_by(|left, right| {
                schedule_candidate_order(&left.solution, left.source_worker, round, &right.solution, right.source_worker, round)
            });
            let mut eligible = Vec::with_capacity(boundary_candidates.len());
            for (index, candidate) in boundary_candidates.iter().enumerate() {
                if shared_incumbent
                    .as_ref()
                    .is_none_or(|incumbent| schedule_candidate_precedes(&candidate.solution, candidate.source_worker, round, incumbent))
                {
                    eligible.push(index);
                } else {
                    incumbent_rejections = incumbent_rejections.saturating_add(1);
                }
            }

            let mut transfer_error = None;
            let transfer_summary = attempt_schedule_transfer_batch(&eligible, |index| {
                let candidate = &mut boundary_candidates[index];
                incumbent_verifications = incumbent_verifications.saturating_add(1);
                let verification_started = Instant::now();
                let verification = verified_candidate(
                    semantic,
                    model,
                    &candidate.solution,
                    EngineKind::ScheduleLocalSearch,
                    VerificationLevel::Transfer,
                    budget.stop(),
                );
                let verification_elapsed = verification_started.elapsed();
                incumbent_verification_elapsed = incumbent_verification_elapsed.saturating_add(verification_elapsed);
                incumbent_verification_max_elapsed = incumbent_verification_max_elapsed.max(verification_elapsed);
                match verification {
                    Ok(verified) => {
                        let canonical = match schedule_solution_from_verified(&verified) {
                            Ok(canonical) => canonical,
                            Err(error) => {
                                transfer_error = Some(error);
                                return ScheduleTransferAttempt::Interrupted;
                            }
                        };
                        let precedes = shared_incumbent
                            .as_ref()
                            .is_none_or(|incumbent| schedule_candidate_precedes(&canonical, candidate.source_worker, round, incumbent));
                        if precedes {
                            let strict_improvement =
                                shared_incumbent.as_ref().is_none_or(|incumbent| canonical.objectives < incumbent.solution.objectives);
                            candidate.solution = canonical.clone();
                            shared_incumbent = Some(SharedScheduleIncumbent {
                                solution: canonical,
                                candidate: verified,
                                source_worker: candidate.source_worker,
                                source_round: round,
                                constructor: candidate.constructor,
                            });
                            incumbent_publications = incumbent_publications.saturating_add(1);
                            first_feasible.get_or_insert_with(|| started.elapsed());
                            if strict_improvement {
                                if let Some(objective) = candidate.solution.objectives.first().copied() {
                                    if published_objective.is_none_or(|current| objective < current) {
                                        published_objective = Some(objective);
                                        progress_publications = progress_publications.saturating_add(1);
                                        improvement(EngineKind::ScheduleLocalSearch, objective);
                                    }
                                }
                            }
                        } else {
                            incumbent_rejections = incumbent_rejections.saturating_add(1);
                        }
                        ScheduleTransferAttempt::Accepted
                    }
                    Err(error) => {
                        worker_sessions[candidate.source_worker] = None;
                        incumbent_rejections = incumbent_rejections.saturating_add(1);
                        if matches!(error, SolveError::Interrupted(_)) || budget.stop().load(Ordering::Acquire) {
                            verification_interruptions = verification_interruptions.saturating_add(1);
                            ScheduleTransferAttempt::Interrupted
                        } else {
                            verification_rejections = verification_rejections.saturating_add(1);
                            ScheduleTransferAttempt::Rejected
                        }
                    }
                }
            });
            if let Some(error) = transfer_error {
                return Err(error);
            }
            incumbent_rejections = incumbent_rejections
                .saturating_add(u64::try_from(eligible.len().saturating_sub(transfer_summary.attempts)).unwrap_or(u64::MAX));
            debug_assert!(!transfer_summary.interrupted || transfer_summary.accepted.is_none());
            debug_assert!(transfer_summary.attempts <= eligible.len());

            let mut round_search_elites = Vec::new();
            for source_worker in reported_workers {
                if let Some(session) = worker_sessions[source_worker].as_mut() {
                    if let Some(candidate) = session.take_search_elite_candidate() {
                        round_search_elites.push(candidate);
                    }
                }
            }
            let reference = worker_sessions.iter().flatten().next();
            #[cfg(test)]
            apply_schedule_search_elite_stop_before_merge(engine_stop);
            search_elite.merge(reference, round_search_elites, engine_stop);
            if transfer_summary.interrupted {
                break;
            }
            if finite_work && remaining.iter().all(|&left| left == 0) {
                break;
            }
            round = round.saturating_add(1);
        }

        let worker_work_min = worker_work.iter().copied().min().unwrap_or(0);
        let worker_work_max = worker_work.iter().copied().max().unwrap_or(0);
        let structural_bound = dual_task.join().map_err(|_| SolveError::Engine("dual-bound worker panicked".to_string()))?;
        Ok::<_, SolveError>((
            shared_incumbent,
            structural_bound,
            progress_publications,
            incumbent_publications,
            incumbent_injection_attempts,
            incumbent_rejections,
            incumbent_verifications,
            verification_rejections,
            verification_interruptions,
            incumbent_verification_elapsed,
            incumbent_verification_max_elapsed,
            incomplete_rejections,
            restart_boundaries,
            peak_buffered_candidates,
            first_feasible,
            stalled_workers,
            stalled_unused_work_steps,
            worker_work_min,
            worker_work_max,
            work_budget_overruns,
            search_elite,
        ))
    })?;
    if work_budget_overruns > 0 {
        return Err(SolveError::Engine(format!(
            "schedule local search exceeded a deterministic round work quota {work_budget_overruns} time(s)"
        )));
    }

    let ScheduleMetricsAggregate {
        construction_candidates,
        construction_elapsed,
        construction_bucket_visits,
        construction_heap_pushes,
        construction_stale_pops,
        construction_heap_rebuilds,
        construction_heap_peak,
        mut constructor,
        work_steps,
        moves_considered,
        moves_accepted,
        incumbent_improvements,
        incumbent_injections,
        cycle_rejections,
        window_rejections,
        objective_rejections,
        reconstructions,
        critical_path_updates,
        delta_evaluations,
        full_evaluations,
        full_fallbacks,
        topological_rebuilds,
        oracle_validations,
        oracle_mismatches,
        dirty_cone_operations,
        max_dirty_cone,
        workspace_growths,
        workspace_rollbacks,
        alns_generation_attempts,
        alns_moves_generated,
        resource_profile_checks,
        resource_candidate_scheduling_attempts,
        resource_event_visits,
        resource_peak_profile_events,
        precedence_rejections,
        infeasible_rejections,
        justification_attempts,
        tabu_steps,
        tabu_hits,
        tabu_aspirations,
        tabu_forced_moves,
        schedule_tsab_owner_worker_mask,
        schedule_tsab_n5_generated,
        schedule_tsab_ranked,
        schedule_tsab_shortlists,
        schedule_tsab_delta_probes,
        schedule_tsab_additional_delta_probes,
        schedule_tsab_full_oracle_commits,
        schedule_tsab_selected_shortlist_rank_sum,
        schedule_tsab_selections,
        schedule_tsab_aspirations,
        schedule_tsab_tabu_rejections,
        schedule_tsab_tabu_resets,
        schedule_tsab_fingerprint_repeats,
        schedule_tsab_escape_signals,
        schedule_tsab_n1_kicks,
        schedule_tsab_kick_moves,
        schedule_tsab_elite_restarts,
        schedule_tsab_n6_kicks,
        schedule_tsab_restart_attempts,
        schedule_tsab_restart_global_rebases,
        schedule_tsab_restart_n6_generated,
        schedule_tsab_restart_delta_probes,
        schedule_tsab_restart_oracle_commits,
        schedule_tsab_restart_rejections,
        schedule_tsab_restart_interruptions,
        schedule_tsab_restart_work_units,
        schedule_tsab_restart_best_base_objective,
        schedule_tsab_restart_best_kicked_objective,
        schedule_tsab_post_restart_improvements,
        schedule_tsab_restart_shortlist_peak_bytes,
        schedule_tsab_ranking_audits,
        schedule_tsab_exact_best_matches,
        schedule_tsab_regret_sum,
        schedule_tsab_regret_max,
        schedule_tsab_workspace_peak_bytes,
        schedule_tsab_activations,
        schedule_tsab_activation_boundary,
        schedule_tsab_legacy_warmup_work_steps,
        schedule_tsab_activation_rebases,
        schedule_tsab_activation_objective,
        schedule_tsab_active_boundaries,
        schedule_tsab_burst_work_limit,
        schedule_tsab_burst_work_units,
        schedule_tsab_improving_commits,
        schedule_tsab_best_committed_objective,
        schedule_tsab_fast_enabled,
        schedule_tsab_fast_eligible,
        schedule_tsab_fast_disabled,
        schedule_tsab_fast_attempts,
        schedule_tsab_fast_commits,
        schedule_tsab_fast_fallbacks,
        schedule_tsab_fast_date_changes,
        schedule_tsab_fast_queue_pops,
        schedule_tsab_fast_full_validations,
        schedule_tsab_fast_oracle_mismatches,
        schedule_tsab_fast_pending_promotions,
        schedule_tsab_fast_pending_discards,
        schedule_tsab_fast_transitions,
        schedule_tsab_fast_work_units,
        schedule_tsab_fast_elapsed,
        schedule_tsab_fast_workspace_peak_bytes,
        session_initializations,
        session_resumes,
        session_rebases,
        island_profile_mask,
        baseline_island_profile_mask,
        scored_island_profile_mask,
        profile_construction_elapsed,
        profile_initial_objectives,
        profile_initial_dispatch_rules,
        profile_best_objectives,
        profile_work_steps,
        reactive_restarts,
        reactive_restart_dispatches,
        reactive_restart_perturbations,
        reactive_restart_rebuild_failures,
        island_scored_candidates,
        island_shortlisted_candidates,
        approximate_candidates_generated,
        approximate_candidates_refined,
        approximate_candidates_certified,
        approximate_candidates_unknown,
        approximation_score_items,
        approximation_sort_items,
        approximation_local_span_items,
        approximation_elapsed,
        approximation_work_units,
        direct_oracle_attempts,
        direct_oracle_accepts,
        direct_oracle_cycles,
        direct_oracle_windows,
        direct_oracle_objective_rejections,
        exact_probes_avoided,
        search_elite_snapshot_captures,
        search_elite_snapshot_interruptions,
        search_elite_snapshot_errors,
        search_elite_capture_worker_elapsed_sum,
        search_elite_snapshot_peak_heap_lower_bound_bytes,
        schedule_lns_shadow_owner_worker_mask,
        schedule_lns,
        schedule_lns_workspace_peak_bytes,
        schedule_path_relink,
        schedule_constructor_multistart_owner_worker_mask,
        schedule_constructor_multistart_attempts,
        schedule_constructor_multistart_constructions,
        schedule_constructor_multistart_interruptions,
        schedule_constructor_multistart_failures,
        schedule_constructor_multistart_feasible,
        schedule_constructor_multistart_distinct_fingerprints,
        schedule_constructor_multistart_other_fingerprint_observations,
        schedule_constructor_multistart_initial_objective,
        schedule_constructor_multistart_best_objective,
        schedule_constructor_multistart_improvements,
        schedule_constructor_multistart_work_units,
        schedule_constructor_multistart_elapsed,
        schedule_constructor_multistart_workspace_peak_bytes,
        schedule_constructor_multistart_best_ordinal,
        schedule_constructor_multistart_best_seed,
        schedule_constructor_multistart_best_fingerprint,
        schedule_constructor_multistart_next_ordinal,
    } = aggregate;
    let global_improvements = progress_publications;
    let elite_pool_size = usize::from(shared_incumbent.is_some());
    let incumbent_source_worker = shared_incumbent.as_ref().map(|incumbent| incumbent.source_worker);
    let incumbent_source_round = shared_incumbent.as_ref().map(|incumbent| incumbent.source_round);
    let verified_primal = shared_incumbent.as_ref().map(|incumbent| incumbent.candidate.clone());
    let mut solution = if let Some(incumbent) = shared_incumbent {
        constructor = Some(incumbent.constructor);
        incumbent.solution
    } else {
        CollectionSolution {
            lists: Vec::new(),
            objectives: Vec::new(),
            feasible: false,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
            modes: Vec::new(),
            bound: None,
        }
    };
    let mut search = structural_bound.as_ref().map_or_else(SolveStats::default, |bound| bound.stats);
    search.nodes = search.nodes.saturating_add(work_steps);
    search.solutions = search.solutions.saturating_add(global_improvements);
    dual::attach(model, &mut solution, structural_bound);
    let elapsed = started.elapsed();
    let candidates_per_second = if elapsed.is_zero() { 0.0 } else { moves_considered as f64 / elapsed.as_secs_f64() };
    let moves_rejected = moves_considered.saturating_sub(moves_accepted);
    let full_recompute_percentage = if delta_evaluations == 0 { 0.0 } else { full_fallbacks as f64 * 100.0 / delta_evaluations as f64 };
    let unused_work_steps = (max_iterations != u64::MAX).then(|| max_iterations.saturating_sub(work_steps));
    let search_elite_size = search_elite.archive.len();
    let search_elite_objectives = search_elite.archive.objectives_summary();
    let search_elite_pairwise_distances_ppm = search_elite.archive.pairwise_distances_ppm_summary();
    let search_elite_distance_stats_ppm = search_elite.archive.distance_stats_ppm();
    let search_elite_min_distance_ppm = search_elite_distance_stats_ppm.map_or_else(|| "none".to_string(), |stats| stats.0.to_string());
    let search_elite_mean_distance_ppm = search_elite_distance_stats_ppm.map_or_else(|| "none".to_string(), |stats| stats.1.to_string());
    let search_elite_max_distance_ppm = search_elite_distance_stats_ppm.map_or_else(|| "none".to_string(), |stats| stats.2.to_string());
    let search_elite_heap_lower_bound_bytes = search_elite.archive.heap_lower_bound_bytes();
    let search_elite_peak_heap_lower_bound_bytes = search_elite
        .peak_heap_lower_bound_bytes
        .max(search_elite_snapshot_peak_heap_lower_bound_bytes)
        .max(search_elite_heap_lower_bound_bytes);
    let profile_construction_seconds = profile_construction_elapsed
        .iter()
        .enumerate()
        .filter(|(profile, _)| island_profile_mask & (1u64 << profile) != 0)
        .map(|(profile, elapsed)| format!("{profile}={}", elapsed.as_secs_f64()))
        .collect::<Vec<_>>()
        .join(",");
    let profile_best_objectives = profile_best_objectives
        .iter()
        .enumerate()
        .filter_map(|(profile, objective)| objective.map(|objective| format!("{profile}={objective}")))
        .collect::<Vec<_>>()
        .join(",");
    let profile_initial_objectives = profile_initial_objectives
        .iter()
        .enumerate()
        .filter_map(|(profile, objective)| objective.map(|objective| format!("{profile}={objective}")))
        .collect::<Vec<_>>()
        .join(",");
    let profile_initial_dispatch_rules = profile_initial_dispatch_rules
        .iter()
        .enumerate()
        .filter_map(|(profile, rule)| rule.map(|rule| format!("{profile}={rule}")))
        .collect::<Vec<_>>()
        .join(",");
    let profile_work_steps = profile_work_steps
        .iter()
        .enumerate()
        .filter(|(profile, _)| island_profile_mask & (1u64 << profile) != 0)
        .map(|(profile, work)| format!("{profile}={work}"))
        .collect::<Vec<_>>()
        .join(",");
    let mut metadata = vec![
        ("workers".to_string(), allocation.workers().to_string()),
        ("constructor".to_string(), constructor.unwrap_or("priority-sgs").to_string()),
        ("construction_seconds".to_string(), construction_elapsed.as_secs_f64().to_string()),
        ("construction_candidates".to_string(), construction_candidates.to_string()),
        ("schedule_construction_bucket_visits".to_string(), construction_bucket_visits.to_string()),
        ("schedule_construction_heap_pushes".to_string(), construction_heap_pushes.to_string()),
        ("schedule_construction_stale_pops".to_string(), construction_stale_pops.to_string()),
        ("schedule_construction_heap_rebuilds".to_string(), construction_heap_rebuilds.to_string()),
        ("schedule_construction_heap_peak".to_string(), construction_heap_peak.to_string()),
        ("candidates_evaluated".to_string(), moves_considered.to_string()),
        ("candidates_per_second".to_string(), candidates_per_second.to_string()),
        ("schedule_work_steps".to_string(), work_steps.to_string()),
        ("schedule_moves_considered".to_string(), moves_considered.to_string()),
        ("schedule_moves_accepted".to_string(), moves_accepted.to_string()),
        ("schedule_moves_rejected".to_string(), moves_rejected.to_string()),
        ("schedule_local_improvements".to_string(), incumbent_improvements.to_string()),
        ("schedule_global_improvements".to_string(), global_improvements.to_string()),
        ("schedule_progress_publications".to_string(), progress_publications.to_string()),
        ("schedule_incumbent_publication_attempts".to_string(), incumbent_publications.saturating_add(incumbent_rejections).to_string()),
        ("schedule_incumbent_publications".to_string(), incumbent_publications.to_string()),
        ("schedule_incumbent_injection_attempts".to_string(), incumbent_injection_attempts.to_string()),
        ("schedule_incumbent_injections".to_string(), incumbent_injections.to_string()),
        ("schedule_incumbent_rejections".to_string(), incumbent_rejections.to_string()),
        ("schedule_incumbent_verifications".to_string(), incumbent_verifications.to_string()),
        ("schedule_incumbent_verification_rejections".to_string(), verification_rejections.to_string()),
        ("schedule_incumbent_verification_interruptions".to_string(), verification_interruptions.to_string()),
        ("schedule_incumbent_verification_seconds".to_string(), incumbent_verification_elapsed.as_secs_f64().to_string()),
        ("schedule_incumbent_verification_max_seconds".to_string(), incumbent_verification_max_elapsed.as_secs_f64().to_string()),
        ("schedule_incumbent_incomplete_rejections".to_string(), incomplete_rejections.to_string()),
        ("schedule_restart_boundaries".to_string(), restart_boundaries.to_string()),
        ("schedule_peak_buffered_candidates".to_string(), peak_buffered_candidates.to_string()),
        ("schedule_restart_work".to_string(), restart_work_cap.to_string()),
        ("schedule_stalled_workers".to_string(), stalled_workers.to_string()),
        ("schedule_stalled_unused_work_steps".to_string(), stalled_unused_work_steps.to_string()),
        ("schedule_worker_work_min".to_string(), worker_work_min.to_string()),
        ("schedule_worker_work_max".to_string(), worker_work_max.to_string()),
        ("schedule_work_budget_overruns".to_string(), work_budget_overruns.to_string()),
        ("schedule_tabu_steps".to_string(), tabu_steps.to_string()),
        ("schedule_tabu_hits".to_string(), tabu_hits.to_string()),
        ("schedule_tabu_aspirations".to_string(), tabu_aspirations.to_string()),
        ("schedule_tabu_forced_moves".to_string(), tabu_forced_moves.to_string()),
        ("schedule_tsab_owner_worker_mask".to_string(), schedule_tsab_owner_worker_mask.to_string()),
        ("schedule_tsab_n5_generated".to_string(), schedule_tsab_n5_generated.to_string()),
        ("schedule_tsab_ranked".to_string(), schedule_tsab_ranked.to_string()),
        ("schedule_tsab_shortlists".to_string(), schedule_tsab_shortlists.to_string()),
        ("schedule_tsab_delta_probes".to_string(), schedule_tsab_delta_probes.to_string()),
        ("schedule_tsab_additional_delta_probes".to_string(), schedule_tsab_additional_delta_probes.to_string()),
        ("schedule_tsab_full_oracle_commits".to_string(), schedule_tsab_full_oracle_commits.to_string()),
        ("schedule_tsab_selected_shortlist_rank_sum".to_string(), schedule_tsab_selected_shortlist_rank_sum.to_string()),
        ("schedule_tsab_selections".to_string(), schedule_tsab_selections.to_string()),
        ("schedule_tsab_aspirations".to_string(), schedule_tsab_aspirations.to_string()),
        ("schedule_tsab_tabu_rejections".to_string(), schedule_tsab_tabu_rejections.to_string()),
        ("schedule_tsab_tabu_resets".to_string(), schedule_tsab_tabu_resets.to_string()),
        ("schedule_tsab_fingerprint_repeats".to_string(), schedule_tsab_fingerprint_repeats.to_string()),
        ("schedule_tsab_escape_signals".to_string(), schedule_tsab_escape_signals.to_string()),
        ("schedule_tsab_n1_kicks".to_string(), schedule_tsab_n1_kicks.to_string()),
        ("schedule_tsab_kick_moves".to_string(), schedule_tsab_kick_moves.to_string()),
        ("schedule_tsab_elite_restarts".to_string(), schedule_tsab_elite_restarts.to_string()),
        ("schedule_tsab_n6_kicks".to_string(), schedule_tsab_n6_kicks.to_string()),
        ("schedule_tsab_restart_attempts".to_string(), schedule_tsab_restart_attempts.to_string()),
        ("schedule_tsab_restart_global_rebases".to_string(), schedule_tsab_restart_global_rebases.to_string()),
        ("schedule_tsab_restart_n6_generated".to_string(), schedule_tsab_restart_n6_generated.to_string()),
        ("schedule_tsab_restart_delta_probes".to_string(), schedule_tsab_restart_delta_probes.to_string()),
        ("schedule_tsab_restart_oracle_commits".to_string(), schedule_tsab_restart_oracle_commits.to_string()),
        ("schedule_tsab_restart_rejections".to_string(), schedule_tsab_restart_rejections.to_string()),
        ("schedule_tsab_restart_interruptions".to_string(), schedule_tsab_restart_interruptions.to_string()),
        ("schedule_tsab_restart_work_units".to_string(), schedule_tsab_restart_work_units.to_string()),
        (
            "schedule_tsab_restart_best_base_objective".to_string(),
            schedule_tsab_restart_best_base_objective.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        (
            "schedule_tsab_restart_best_kicked_objective".to_string(),
            schedule_tsab_restart_best_kicked_objective.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        ("schedule_tsab_post_restart_improvements".to_string(), schedule_tsab_post_restart_improvements.to_string()),
        ("schedule_tsab_restart_shortlist_peak_bytes".to_string(), schedule_tsab_restart_shortlist_peak_bytes.to_string()),
        ("schedule_tsab_ranking_audits".to_string(), schedule_tsab_ranking_audits.to_string()),
        ("schedule_tsab_exact_best_matches".to_string(), schedule_tsab_exact_best_matches.to_string()),
        ("schedule_tsab_regret_sum".to_string(), schedule_tsab_regret_sum.to_string()),
        ("schedule_tsab_regret_max".to_string(), schedule_tsab_regret_max.to_string()),
        ("schedule_tsab_workspace_peak_bytes".to_string(), schedule_tsab_workspace_peak_bytes.to_string()),
        ("schedule_tsab_activations".to_string(), schedule_tsab_activations.to_string()),
        (
            "schedule_tsab_activation_boundary".to_string(),
            schedule_tsab_activation_boundary.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        ("schedule_tsab_legacy_warmup_work_steps".to_string(), schedule_tsab_legacy_warmup_work_steps.to_string()),
        ("schedule_tsab_activation_rebases".to_string(), schedule_tsab_activation_rebases.to_string()),
        (
            "schedule_tsab_activation_objective".to_string(),
            schedule_tsab_activation_objective.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        ("schedule_tsab_active_boundaries".to_string(), schedule_tsab_active_boundaries.to_string()),
        ("schedule_tsab_burst_work_limit".to_string(), schedule_tsab_burst_work_limit.to_string()),
        ("schedule_tsab_burst_work_units".to_string(), schedule_tsab_burst_work_units.to_string()),
        ("schedule_tsab_improving_commits".to_string(), schedule_tsab_improving_commits.to_string()),
        (
            "schedule_tsab_best_committed_objective".to_string(),
            schedule_tsab_best_committed_objective.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        ("schedule_tsab_fast_enabled".to_string(), schedule_tsab_fast_enabled.to_string()),
        ("schedule_tsab_fast_eligible".to_string(), schedule_tsab_fast_eligible.to_string()),
        ("schedule_tsab_fast_disabled".to_string(), schedule_tsab_fast_disabled.to_string()),
        ("schedule_tsab_fast_attempts".to_string(), schedule_tsab_fast_attempts.to_string()),
        ("schedule_tsab_fast_commits".to_string(), schedule_tsab_fast_commits.to_string()),
        ("schedule_tsab_fast_fallbacks".to_string(), schedule_tsab_fast_fallbacks.to_string()),
        ("schedule_tsab_fast_date_changes".to_string(), schedule_tsab_fast_date_changes.to_string()),
        ("schedule_tsab_fast_queue_pops".to_string(), schedule_tsab_fast_queue_pops.to_string()),
        ("schedule_tsab_fast_full_validations".to_string(), schedule_tsab_fast_full_validations.to_string()),
        ("schedule_tsab_fast_oracle_mismatches".to_string(), schedule_tsab_fast_oracle_mismatches.to_string()),
        ("schedule_tsab_fast_pending_promotions".to_string(), schedule_tsab_fast_pending_promotions.to_string()),
        ("schedule_tsab_fast_pending_discards".to_string(), schedule_tsab_fast_pending_discards.to_string()),
        ("schedule_tsab_fast_transitions".to_string(), schedule_tsab_fast_transitions.to_string()),
        ("schedule_tsab_fast_work_units".to_string(), schedule_tsab_fast_work_units.to_string()),
        ("schedule_tsab_fast_elapsed_seconds".to_string(), schedule_tsab_fast_elapsed.as_secs_f64().to_string()),
        ("schedule_tsab_fast_workspace_peak_bytes".to_string(), schedule_tsab_fast_workspace_peak_bytes.to_string()),
        ("schedule_session_initializations".to_string(), session_initializations.to_string()),
        ("schedule_session_resumes".to_string(), session_resumes.to_string()),
        ("schedule_session_rebases".to_string(), session_rebases.to_string()),
        ("schedule_island_profile_mask".to_string(), island_profile_mask.to_string()),
        ("schedule_baseline_island_profile_mask".to_string(), baseline_island_profile_mask.to_string()),
        ("schedule_scored_island_profile_mask".to_string(), scored_island_profile_mask.to_string()),
        ("schedule_island_profile_count".to_string(), island_profile_mask.count_ones().to_string()),
        ("schedule_profile_construction_seconds".to_string(), profile_construction_seconds),
        ("schedule_profile_initial_objectives".to_string(), profile_initial_objectives),
        ("schedule_profile_initial_dispatch_rules".to_string(), profile_initial_dispatch_rules),
        ("schedule_profile_best_objectives".to_string(), profile_best_objectives),
        ("schedule_profile_work_steps".to_string(), profile_work_steps),
        ("schedule_reactive_restarts".to_string(), reactive_restarts.to_string()),
        ("schedule_reactive_restart_dispatches".to_string(), reactive_restart_dispatches.to_string()),
        ("schedule_reactive_restart_perturbations".to_string(), reactive_restart_perturbations.to_string()),
        ("schedule_reactive_restart_rebuild_failures".to_string(), reactive_restart_rebuild_failures.to_string()),
        ("schedule_island_scored_candidates".to_string(), island_scored_candidates.to_string()),
        ("schedule_island_shortlisted_candidates".to_string(), island_shortlisted_candidates.to_string()),
        ("schedule_approximate_candidates_generated".to_string(), approximate_candidates_generated.to_string()),
        ("schedule_approximate_candidates_refined".to_string(), approximate_candidates_refined.to_string()),
        ("schedule_approximate_candidates_certified".to_string(), approximate_candidates_certified.to_string()),
        ("schedule_approximate_candidates_unknown".to_string(), approximate_candidates_unknown.to_string()),
        ("schedule_approximation_score_items".to_string(), approximation_score_items.to_string()),
        ("schedule_approximation_sort_items".to_string(), approximation_sort_items.to_string()),
        ("schedule_approximation_local_span_items".to_string(), approximation_local_span_items.to_string()),
        ("schedule_approximation_elapsed_seconds".to_string(), approximation_elapsed.as_secs_f64().to_string()),
        ("schedule_approximation_work_units".to_string(), approximation_work_units.to_string()),
        ("schedule_direct_oracle_attempts".to_string(), direct_oracle_attempts.to_string()),
        ("schedule_direct_oracle_accepts".to_string(), direct_oracle_accepts.to_string()),
        ("schedule_direct_oracle_cycles".to_string(), direct_oracle_cycles.to_string()),
        ("schedule_direct_oracle_windows".to_string(), direct_oracle_windows.to_string()),
        ("schedule_direct_oracle_objective_rejections".to_string(), direct_oracle_objective_rejections.to_string()),
        ("schedule_exact_probes_avoided".to_string(), exact_probes_avoided.to_string()),
        ("schedule_search_elite_pool_size".to_string(), search_elite_size.to_string()),
        ("schedule_search_elite_batches".to_string(), search_elite.batches.to_string()),
        ("schedule_search_elite_batches_skipped_after_stop".to_string(), search_elite.batches_skipped_after_stop.to_string()),
        ("schedule_search_elite_candidates".to_string(), search_elite.candidates.to_string()),
        ("schedule_search_elite_insertions".to_string(), search_elite.insertions.to_string()),
        ("schedule_search_elite_duplicates".to_string(), search_elite.duplicates.to_string()),
        ("schedule_search_elite_dominated".to_string(), search_elite.dominated.to_string()),
        ("schedule_search_elite_evictions".to_string(), search_elite.evictions.to_string()),
        ("schedule_search_elite_interruptions".to_string(), search_elite.interruptions.to_string()),
        ("schedule_search_elite_merge_errors".to_string(), search_elite.merge_errors.to_string()),
        ("schedule_search_elite_snapshot_captures".to_string(), search_elite_snapshot_captures.to_string()),
        ("schedule_search_elite_snapshot_interruptions".to_string(), search_elite_snapshot_interruptions.to_string()),
        ("schedule_search_elite_snapshot_errors".to_string(), search_elite_snapshot_errors.to_string()),
        ("schedule_search_elite_objectives".to_string(), search_elite_objectives),
        ("schedule_search_elite_pairwise_distances_ppm".to_string(), search_elite_pairwise_distances_ppm),
        ("schedule_search_elite_min_distance_ppm".to_string(), search_elite_min_distance_ppm),
        ("schedule_search_elite_mean_distance_ppm".to_string(), search_elite_mean_distance_ppm),
        ("schedule_search_elite_max_distance_ppm".to_string(), search_elite_max_distance_ppm),
        ("schedule_search_elite_capture_worker_seconds_sum".to_string(), search_elite_capture_worker_elapsed_sum.as_secs_f64().to_string()),
        ("schedule_search_elite_merge_wall_seconds".to_string(), search_elite.merge_wall_elapsed.as_secs_f64().to_string()),
        ("schedule_search_elite_heap_lower_bound_bytes".to_string(), search_elite_heap_lower_bound_bytes.to_string()),
        ("schedule_search_elite_peak_heap_lower_bound_bytes".to_string(), search_elite_peak_heap_lower_bound_bytes.to_string()),
        ("schedule_lns_shadow_enabled".to_string(), u64::from(request.schedule_lns_shadow).to_string()),
        ("schedule_lns_shadow_active".to_string(), u64::from(schedule_lns_shadow_owner_worker_mask != 0).to_string()),
        ("schedule_lns_shadow_owner_worker_mask".to_string(), schedule_lns_shadow_owner_worker_mask.to_string()),
        ("schedule_lns_attempts".to_string(), schedule_lns.attempts.to_string()),
        ("schedule_lns_selected_operations".to_string(), schedule_lns.selected_operations.to_string()),
        ("schedule_lns_feasible".to_string(), schedule_lns.feasible.to_string()),
        ("schedule_lns_reconstructed".to_string(), schedule_lns.reconstructed.to_string()),
        ("schedule_lns_improvements".to_string(), schedule_lns.improvements.to_string()),
        ("schedule_lns_shadow_improvements".to_string(), schedule_lns.shadow_improvements.to_string()),
        ("schedule_lns_shadow_improvement_sum".to_string(), schedule_lns.shadow_improvement_sum.to_string()),
        ("schedule_lns_shadow_best_improvement".to_string(), schedule_lns.shadow_best_improvement.to_string()),
        ("schedule_lns_timeouts".to_string(), schedule_lns.timeouts.to_string()),
        ("schedule_lns_interruptions".to_string(), schedule_lns.interruptions.to_string()),
        ("schedule_lns_infeasible".to_string(), schedule_lns.infeasible.to_string()),
        ("schedule_lns_non_improving".to_string(), schedule_lns.non_improving.to_string()),
        ("schedule_lns_reconstruction_rejections".to_string(), schedule_lns.reconstruction_rejections.to_string()),
        ("schedule_lns_oracle_rejections".to_string(), schedule_lns.oracle_rejections.to_string()),
        ("schedule_lns_exact_rejections".to_string(), schedule_lns.exact_rejections.to_string()),
        ("schedule_lns_worker_seconds_sum".to_string(), Duration::from_micros(schedule_lns.elapsed_micros).as_secs_f64().to_string()),
        ("schedule_lns_workspace_peak_bytes".to_string(), schedule_lns_workspace_peak_bytes.to_string()),
        ("schedule_constructor_workers_requested".to_string(), request.schedule_constructor_workers.to_string()),
        ("schedule_constructor_multistart_enabled".to_string(), u64::from(request.schedule_constructor_workers == 1).to_string()),
        (
            "schedule_constructor_multistart_active".to_string(),
            u64::from(schedule_constructor_multistart_owner_worker_mask != 0).to_string(),
        ),
        ("schedule_constructor_multistart_owner_worker_mask".to_string(), schedule_constructor_multistart_owner_worker_mask.to_string()),
        ("schedule_constructor_multistart_attempts".to_string(), schedule_constructor_multistart_attempts.to_string()),
        ("schedule_constructor_multistart_constructions".to_string(), schedule_constructor_multistart_constructions.to_string()),
        ("schedule_constructor_multistart_interruptions".to_string(), schedule_constructor_multistart_interruptions.to_string()),
        ("schedule_constructor_multistart_failures".to_string(), schedule_constructor_multistart_failures.to_string()),
        ("schedule_constructor_multistart_feasible".to_string(), schedule_constructor_multistart_feasible.to_string()),
        (
            "schedule_constructor_multistart_distinct_fingerprints".to_string(),
            schedule_constructor_multistart_distinct_fingerprints.to_string(),
        ),
        (
            "schedule_constructor_multistart_other_fingerprint_observations".to_string(),
            schedule_constructor_multistart_other_fingerprint_observations.to_string(),
        ),
        (
            "schedule_constructor_multistart_initial_objective".to_string(),
            schedule_constructor_multistart_initial_objective.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        (
            "schedule_constructor_multistart_best_objective".to_string(),
            schedule_constructor_multistart_best_objective.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        ("schedule_constructor_multistart_improvements".to_string(), schedule_constructor_multistart_improvements.to_string()),
        ("schedule_constructor_multistart_work_units".to_string(), schedule_constructor_multistart_work_units.to_string()),
        (
            "schedule_constructor_multistart_worker_seconds_sum".to_string(),
            schedule_constructor_multistart_elapsed.as_secs_f64().to_string(),
        ),
        (
            "schedule_constructor_multistart_workspace_peak_bytes".to_string(),
            schedule_constructor_multistart_workspace_peak_bytes.to_string(),
        ),
        (
            "schedule_constructor_multistart_best_ordinal".to_string(),
            schedule_constructor_multistart_best_ordinal.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        (
            "schedule_constructor_multistart_best_seed".to_string(),
            schedule_constructor_multistart_best_seed.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        (
            "schedule_constructor_multistart_best_fingerprint".to_string(),
            schedule_constructor_multistart_best_fingerprint.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        (
            "schedule_constructor_multistart_next_ordinal".to_string(),
            schedule_constructor_multistart_next_ordinal.map_or_else(|| "none".to_string(), |value| value.to_string()),
        ),
        ("schedule_path_relink_enabled".to_string(), u64::from(request.schedule_path_relink).to_string()),
        ("schedule_path_relink_active".to_string(), u64::from(schedule_path_relink.requests != 0).to_string()),
        ("schedule_path_relink_guide_requests".to_string(), schedule_path_relink.requests.to_string()),
        ("schedule_path_relink_best_guides".to_string(), schedule_path_relink.best_guides.to_string()),
        ("schedule_path_relink_diverse_guides".to_string(), schedule_path_relink.diverse_guides.to_string()),
        ("schedule_path_relink_guide_loads".to_string(), schedule_path_relink.guide_loads.to_string()),
        ("schedule_path_relink_guide_incompatible".to_string(), schedule_path_relink.guide_incompatible.to_string()),
        ("schedule_path_relink_guide_interruptions".to_string(), schedule_path_relink.guide_interruptions.to_string()),
        ("schedule_path_relink_critical_operations_scanned".to_string(), schedule_path_relink.critical_operations_scanned.to_string()),
        ("schedule_path_relink_candidates_generated".to_string(), schedule_path_relink.candidates_generated.to_string()),
        ("schedule_path_relink_candidates_positive_gain".to_string(), schedule_path_relink.candidates_positive_gain.to_string()),
        ("schedule_path_relink_acyclicity_certified".to_string(), schedule_path_relink.acyclicity_certified.to_string()),
        ("schedule_path_relink_acyclicity_unknown".to_string(), schedule_path_relink.acyclicity_unknown.to_string()),
        ("schedule_path_relink_prefilter_rejections".to_string(), schedule_path_relink.prefilter_rejections.to_string()),
        ("schedule_path_relink_candidates_retained".to_string(), schedule_path_relink.candidates_retained.to_string()),
        ("schedule_path_relink_candidates_refined".to_string(), schedule_path_relink.candidates_refined.to_string()),
        ("schedule_path_relink_candidates_shortlisted".to_string(), schedule_path_relink.candidates_shortlisted.to_string()),
        ("schedule_path_relink_no_move".to_string(), schedule_path_relink.no_move.to_string()),
        ("schedule_path_relink_guide_arc_gain_shortlisted".to_string(), schedule_path_relink.guide_arc_gain_shortlisted.to_string()),
        ("schedule_path_relink_oracle_attempts".to_string(), schedule_path_relink.oracle_attempts.to_string()),
        ("schedule_path_relink_oracle_accepts".to_string(), schedule_path_relink.oracle_accepts.to_string()),
        ("schedule_path_relink_cycle_rejections".to_string(), schedule_path_relink.cycle_rejections.to_string()),
        ("schedule_path_relink_window_rejections".to_string(), schedule_path_relink.window_rejections.to_string()),
        ("schedule_path_relink_other_rejections".to_string(), schedule_path_relink.other_rejections.to_string()),
        ("schedule_path_relink_rollbacks".to_string(), schedule_path_relink.rollbacks.to_string()),
        ("schedule_path_relink_elite_improvements".to_string(), schedule_path_relink.elite_improvements.to_string()),
        ("schedule_path_relink_guide_arc_gain_accepted".to_string(), schedule_path_relink.guide_arc_gain_accepted.to_string()),
        ("schedule_path_relink_worker_seconds_sum".to_string(), schedule_path_relink.elapsed.as_secs_f64().to_string()),
        ("schedule_path_relink_workspace_peak_bytes".to_string(), schedule_path_relink.workspace_peak_bytes.to_string()),
        ("schedule_elite_pool_size".to_string(), elite_pool_size.to_string()),
        ("elite_pool_size".to_string(), elite_pool_size.to_string()),
        (
            "schedule_incumbent_source_worker".to_string(),
            incumbent_source_worker.map_or_else(|| "none".to_string(), |worker| worker.to_string()),
        ),
        (
            "schedule_incumbent_source_round".to_string(),
            incumbent_source_round.map_or_else(|| "none".to_string(), |round| round.to_string()),
        ),
        ("schedule_cycle_rejections".to_string(), cycle_rejections.to_string()),
        ("schedule_window_rejections".to_string(), window_rejections.to_string()),
        ("schedule_objective_rejections".to_string(), objective_rejections.to_string()),
        ("schedule_reconstructions".to_string(), reconstructions.to_string()),
        ("critical_path_updates".to_string(), critical_path_updates.to_string()),
        ("schedule_delta_evaluations".to_string(), delta_evaluations.to_string()),
        ("schedule_full_evaluations".to_string(), full_evaluations.to_string()),
        ("schedule_full_fallbacks".to_string(), full_fallbacks.to_string()),
        ("schedule_topological_rebuilds".to_string(), topological_rebuilds.to_string()),
        ("schedule_oracle_validations".to_string(), oracle_validations.to_string()),
        ("schedule_oracle_mismatches".to_string(), oracle_mismatches.to_string()),
        ("schedule_dirty_cone_operations".to_string(), dirty_cone_operations.to_string()),
        ("schedule_max_dirty_cone".to_string(), max_dirty_cone.to_string()),
        ("schedule_workspace_growths".to_string(), workspace_growths.to_string()),
        ("schedule_workspace_rollbacks".to_string(), workspace_rollbacks.to_string()),
        ("schedule_alns_generation_attempts".to_string(), alns_generation_attempts.to_string()),
        ("schedule_alns_moves_generated".to_string(), alns_moves_generated.to_string()),
        ("full_recompute_percentage".to_string(), full_recompute_percentage.to_string()),
        ("resource_profile_checks".to_string(), resource_profile_checks.to_string()),
        ("resource_candidate_scheduling_attempts".to_string(), resource_candidate_scheduling_attempts.to_string()),
        ("resource_event_visits".to_string(), resource_event_visits.to_string()),
        ("resource_peak_profile_events".to_string(), resource_peak_profile_events.to_string()),
        ("schedule_precedence_rejections".to_string(), precedence_rejections.to_string()),
        ("schedule_infeasible_rejections".to_string(), infeasible_rejections.to_string()),
        ("schedule_justification_attempts".to_string(), justification_attempts.to_string()),
        ("schedule_candidates_per_second".to_string(), candidates_per_second.to_string()),
    ];
    if let Some(first) = first_feasible {
        metadata.push(("time_to_first_feasible".to_string(), first.as_secs_f64().to_string()));
    }
    if let Some(unused) = unused_work_steps {
        metadata.push(("schedule_unused_work_steps".to_string(), unused.to_string()));
    }
    let status = if solution.feasible { SolveStatus::Satisfiable } else { SolveStatus::Unknown };
    let mut result = finish_collection_result(
        semantic,
        model,
        CollectionCompletion {
            solution,
            verified_primal,
            status,
            source: EngineKind::ScheduleLocalSearch,
            proof: None,
            report: EngineReport {
                engine: Some(EngineKind::ScheduleLocalSearch),
                search,
                elapsed,
                improvements: global_improvements,
                metadata,
            },
        },
        budget,
        transfer_stop,
    )?;
    if request.limits.iterations.is_some_and(|limit| work_steps >= limit) && !budget.expired() {
        result.message = Some("schedule local search reached the shared iteration limit".to_string());
    } else if stalled_unused_work_steps > 0 && !budget.expired() {
        result.message = Some(format!(
            "schedule local search retired {stalled_workers} stalled worker(s) with {stalled_unused_work_steps} unused work step(s)"
        ));
    }
    Ok(result)
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
    verified_primal: Option<CandidateSolution>,
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
    let CollectionCompletion { solution, verified_primal, status, source, proof, report } = completion;
    let primal = if let Some(candidate) = verified_primal {
        #[cfg(test)]
        list::audit_record_final_verification_boundary();
        Some(promote_verified_collection_candidate(candidate, budget, transfer_stop)?)
    } else if solution.feasible {
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

fn schedule_solution_from_verified(candidate: &CandidateSolution) -> Result<CollectionSolution, SolveError> {
    if !candidate.transferable() {
        return Err(SolveError::InvalidResult(
            "schedule local search received an incumbent that was not verified for transfer".to_string(),
        ));
    }
    if !candidate.assignment().integers.is_empty() || !candidate.assignment().sets.is_empty() || !candidate.assignment().lists.is_empty() {
        return Err(SolveError::InvalidResult(
            "schedule local search received a transferred incumbent from a different model family".to_string(),
        ));
    }
    Ok(CollectionSolution {
        lists: Vec::new(),
        objectives: candidate.objectives().to_vec(),
        feasible: true,
        starts: candidate.assignment().intervals.iter().map(|interval| interval.start.unwrap_or_default()).collect(),
        presences: candidate.assignment().intervals.iter().map(|interval| interval.present).collect(),
        machines: candidate
            .assignment()
            .intervals
            .iter()
            .map(|interval| interval.machine.and_then(|value| i64::try_from(value).ok()).unwrap_or(-1))
            .collect(),
        modes: candidate.assignment().intervals.iter().map(|interval| interval.mode).collect(),
        bound: None,
    })
}

fn promote_verified_collection_candidate(
    candidate: CandidateSolution,
    budget: &SolveBudget,
    transfer_stop: Option<&AtomicBool>,
) -> Result<CandidateSolution, SolveError> {
    if candidate.verification() != VerificationLevel::Transfer {
        return Err(SolveError::InvalidResult("collection result candidate was not verified for transfer".to_string()));
    }
    apply_final_replay_audit_before_first_pass(budget);
    apply_final_replay_audit_after_interrupt(budget);
    if transfer_stop.is_some_and(|stop| stop.load(Ordering::Acquire)) || budget.hard_cancelled() {
        return Err(SolveError::Interrupted(format!(
            "canonical final replay exceeded its grace or was cancelled: {:?}",
            budget.termination_reason()
        )));
    }
    Ok(CandidateSolution::verified(
        candidate.assignment().clone(),
        candidate.objectives().to_vec(),
        candidate.source(),
        VerificationLevel::Final,
    ))
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
