//! Orchestration adapter for compiled integer and set models.

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use crate::engines::cp::incremental::IncrementalSearch;
use crate::engines::cp::portfolio::{self, RunOptions, SearchGuidance};
use crate::engines::linear;
use crate::engines::ls::integer;
use crate::lcg::clause::SharedClausePool;
use crate::model::{CompiledCp, Constraint, Model, Relation};
use crate::problem::Problem;
use crate::search::{Assumption, SolveStats};

use super::{
    merge_search_stats, Assignment, Bound, CandidateSolution, EngineKind, EngineReport, EventControl, EventSink, ProofClaim,
    ProvenConclusion, SolveBudget, SolveError, SolveEvent, SolveMode, SolveRequest, SolveResult, SolveStatus, TerminationReason,
    VerificationLevel,
};

#[cfg(test)]
std::thread_local! {
    static ROOT_PROBLEM_CLONES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[inline]
fn clone_problem(problem: &Problem) -> Problem {
    #[cfg(test)]
    ROOT_PROBLEM_CLONES.set(ROOT_PROBLEM_CLONES.get().saturating_add(1));
    problem.clone()
}

#[cfg(test)]
pub(crate) fn audit_cp_root_problem_clones() -> u64 {
    ROOT_PROBLEM_CLONES.get()
}

#[cfg(test)]
pub(crate) fn audit_cp_repair_completion(complete: bool, fully_specified: bool) -> Result<bool, SolveError> {
    super::integer_search::audit_cp_repair_completion(complete, fully_specified)
}

#[cfg(test)]
pub(crate) fn audit_partial_cp_repair_search_rejection() -> Result<bool, SolveError> {
    super::integer_search::audit_partial_cp_repair_search_rejection()
}

#[cfg(test)]
pub(crate) fn audit_prearmed_cp_repair_interruption() -> Result<bool, SolveError> {
    super::integer_search::audit_prearmed_cp_repair_interruption()
}

#[cfg(test)]
pub(crate) fn audit_integer_warm_start_preflight(model: &Model, stop: &std::sync::atomic::AtomicBool) -> Option<(u64, u64, u64)> {
    integer::audit_warm_start_preflight(model, stop)
}

#[cfg(test)]
pub(crate) fn audit_integer_warm_start_compile(
    model: &Model,
    compiled: &CompiledCp,
    memory_allowance: u64,
    stop: &std::sync::atomic::AtomicBool,
) -> (bool, u64) {
    integer::audit_compile_warm_start(model, compiled, memory_allowance, stop)
}

#[derive(Clone)]
pub(crate) struct CpSolvePlan {
    compiled: CompiledCp,
    local_search: Option<integer::IntegerLocalSearchPlan>,
    integer_warm_start: Option<integer::IntegerWarmStartPlan>,
    estimated_backend_bytes: u64,
    mode: SolveMode,
    guidance: SearchGuidance,
    assumptions: Vec<super::SemanticAssumption>,
    hints: Vec<(usize, i32)>,
    primary_branch_scope: Option<Vec<usize>>,
    branch_order: Vec<usize>,
}

impl CpSolvePlan {
    pub(crate) fn estimated_backend_bytes(&self) -> u64 {
        self.estimated_backend_bytes
    }

    pub(crate) fn engine(&self) -> EngineKind {
        if self.mode == SolveMode::LocalSearch {
            EngineKind::IntegerLocalSearch
        } else {
            EngineKind::IntegerExact
        }
    }
}

#[cfg(test)]
pub(crate) fn compile_cp_plan(model: &Model, request: &SolveRequest, budget: &SolveBudget) -> Result<CpSolvePlan, SolveError> {
    request.validate()?;
    compile_cp_plan_inner(model, request, budget, false)
}

/// Compile after the orchestrator has validated both the request and the
/// enclosing `ModelPackage`. Assumption constraints are constructed below from
/// checked integer references, so they preserve the validated model invariants.
pub(crate) fn compile_cp_plan_validated(model: &Model, request: &SolveRequest, budget: &SolveBudget) -> Result<CpSolvePlan, SolveError> {
    compile_cp_plan_inner(model, request, budget, true)
}

fn compile_cp_plan_inner(
    model: &Model,
    request: &SolveRequest,
    budget: &SolveBudget,
    model_is_validated: bool,
) -> Result<CpSolvePlan, SolveError> {
    if budget.expired() {
        return Err(SolveError::Interrupted("solve budget expired before CP compilation".to_string()));
    }
    if request.list_hint.is_some() {
        return Err(SolveError::InvalidRequest("list_hint is only supported for list_vars models".to_string()));
    }
    if request.schedule_cdcl {
        return Err(SolveError::InvalidRequest("schedule_cdcl is only valid for a semantic schedule model".to_string()));
    }
    if request.routing != super::RoutingControls::default() {
        return Err(SolveError::InvalidRequest("routing controls are only valid for a semantic routing model".to_string()));
    }
    if request.sat != super::SatControls::default() {
        return Err(SolveError::InvalidRequest("SAT controls require the specialized SAT plan".to_string()));
    }
    if request.mode != SolveMode::LocalSearch && request.limits.iterations.is_some() {
        return Err(SolveError::InvalidRequest("max_iterations requires engine='ls' for integer models".to_string()));
    }
    if request.mode == SolveMode::LocalSearch {
        if !request.hints.is_empty() || request.primary_branch_scope.is_some() || !request.branch_order.is_empty() {
            return Err(SolveError::InvalidRequest(
                "integer local search does not yet accept value hints, primary branch scope, or branch order".to_string(),
            ));
        }
        if request.limits.conflicts.is_some() {
            return Err(SolveError::InvalidRequest("conflict limits require integer exact mode".to_string()));
        }
        if request.cp != super::CpControls::default() {
            return Err(SolveError::InvalidRequest("CP portfolio controls require integer exact mode".to_string()));
        }
        if request.linear != super::LinearControls::default() {
            return Err(SolveError::InvalidRequest("linear relaxation controls require integer exact mode".to_string()));
        }
    } else if model.objectives().is_empty() && request.linear != super::LinearControls::default() {
        return Err(SolveError::InvalidRequest("linear relaxation controls require an integer optimization objective".to_string()));
    }
    for assumption in &request.assumptions {
        if assumption.variable >= model.int_vars().len() {
            return Err(SolveError::InvalidRequest(format!("assumption references unknown integer variable {}", assumption.variable)));
        }
    }
    let estimated_backend_bytes = CompiledCp::estimate_semantic_bytes_interruptible(model, budget.stop())
        .ok_or_else(|| SolveError::Interrupted("solve budget expired during CP memory preflight".to_string()))?
        .saturating_add(u64::try_from(request.assumptions.len()).unwrap_or(u64::MAX).saturating_mul(2048));
    let estimated_local_search_bytes = if request.mode == SolveMode::LocalSearch {
        integer::estimate_local_search_plan_bytes(model, budget.stop())
            .ok_or_else(|| SolveError::Interrupted("solve budget expired during integer local-search memory preflight".to_string()))?
            .saturating_add(u64::try_from(request.assumptions.len()).unwrap_or(u64::MAX).saturating_mul(256))
    } else {
        0
    };
    let estimated_per_worker_bytes = estimated_backend_bytes.saturating_add(estimated_local_search_bytes);
    let estimated_concurrent_bytes = estimated_per_worker_bytes.saturating_mul(u64::try_from(request.threads).unwrap_or(u64::MAX));
    if request.limits.memory_bytes.is_some_and(|memory| estimated_concurrent_bytes > memory) {
        return Err(SolveError::Compile(format!(
            "estimated CP backend and local-search plan require {estimated_concurrent_bytes} bytes across concurrent workers, above the memory limit"
        )));
    }
    if budget.expired() {
        return Err(SolveError::Interrupted("solve budget expired after CP memory preflight".to_string()));
    }
    let mut effective = (!request.assumptions.is_empty()).then(|| model.clone());
    if let Some(effective) = &mut effective {
        for assumption in &request.assumptions {
            let relation = match assumption.operation {
                super::SemanticAssumptionOp::Eq => Relation::Eq,
                super::SemanticAssumptionOp::Ne => Relation::Ne,
                super::SemanticAssumptionOp::Le => Relation::Le,
                super::SemanticAssumptionOp::Lt => Relation::Lt,
                super::SemanticAssumptionOp::Ge => Relation::Ge,
                super::SemanticAssumptionOp::Gt => Relation::Gt,
            };
            effective.add_constraint(Constraint::Linear {
                terms: vec![(1, crate::model::IntVarRef(assumption.variable))],
                relation,
                rhs: i64::from(assumption.value),
            });
        }
    }
    let effective = effective.as_ref().unwrap_or(model);
    let compiled = if model_is_validated {
        CompiledCp::compile_validated_with_estimate_interruptible(effective, estimated_backend_bytes, budget.stop())
    } else {
        CompiledCp::compile_with_estimate_interruptible(effective, estimated_backend_bytes, budget.stop())
    }
    .map_err(|error| SolveError::Compile(error.reason))?
    .ok_or_else(|| SolveError::Interrupted("solve budget expired during CP compilation".to_string()))?;
    validate_cp_controls(&compiled, request)?;
    let thread_count = u64::try_from(request.threads).unwrap_or(u64::MAX).max(1);
    let per_worker_memory = request.limits.memory_bytes.map_or(u64::MAX, |memory| memory / thread_count);
    let mut resident_bytes = compiled.estimated_bytes();
    let local_search = if request.mode == SolveMode::LocalSearch {
        let allowance = per_worker_memory.saturating_sub(resident_bytes);
        let Some(plan) = integer::compile_interruptible(effective, &compiled, budget.stop(), allowance)? else {
            if budget.expired() {
                return Err(SolveError::Interrupted("solve budget expired during integer local-search compilation".to_string()));
            }
            return Err(SolveError::Compile(
                "estimated integer local-search plan exceeds the remaining per-worker memory allowance".to_string(),
            ));
        };
        resident_bytes = resident_bytes.saturating_add(plan.estimated_bytes);
        Some(plan)
    } else {
        None
    };
    let warm_start_eligible = request.mode != SolveMode::LocalSearch
        && request.threads == 1
        && request.limits.conflicts.is_none()
        && request.cp == super::CpControls::default()
        && request.assumptions.is_empty()
        && request.hints.is_empty()
        && request.primary_branch_scope.is_none()
        && request.branch_order.is_empty();
    let integer_warm_start = if warm_start_eligible {
        let allowance = per_worker_memory.saturating_sub(resident_bytes);
        budget.warm_start_stop().and_then(|warm_stop| {
            let warm = integer::compile_warm_start(effective, &compiled, allowance, warm_stop.flag());
            resident_bytes = resident_bytes.saturating_add(warm.estimated_bytes);
            warm.plan
        })
    } else {
        None
    };
    let (initial_phase, branch_order, primary_branch_scope) = compiled
        .search_guidance_interruptible(&request.hints, &request.branch_order, request.primary_branch_scope.as_deref(), budget.stop())
        .map_err(|error| SolveError::InvalidRequest(error.reason))?
        .ok_or_else(|| SolveError::Interrupted("solve budget expired during CP search-guidance compilation".to_string()))?;
    Ok(CpSolvePlan {
        compiled,
        local_search,
        integer_warm_start,
        estimated_backend_bytes: resident_bytes,
        mode: request.mode,
        guidance: SearchGuidance { initial_phase, branch_order, primary_branch_scope, linear: None },
        assumptions: request.assumptions.clone(),
        hints: request.hints.clone(),
        primary_branch_scope: request.primary_branch_scope.clone(),
        branch_order: request.branch_order.clone(),
    })
}

fn validate_cp_controls(compiled: &CompiledCp, request: &SolveRequest) -> Result<(), SolveError> {
    let controls = request.cp;
    let cooperative_roles = controls.split || controls.probes > 0 || controls.lns > 0;
    if request.limits.conflicts.is_some() && cooperative_roles {
        return Err(SolveError::InvalidRequest("a total conflict limit cannot be combined with split, probe, or LNS workers".to_string()));
    }
    let objectives = compiled.objectives();
    if objectives.is_empty() {
        if cooperative_roles {
            return Err(SolveError::InvalidRequest("split, probes, and LNS require an optimization objective".to_string()));
        }
        return Ok(());
    }
    if controls.no_learn_csp {
        return Err(SolveError::InvalidRequest("no_learn_csp applies only to satisfaction models without an objective".to_string()));
    }
    if cooperative_roles && request.threads == 1 {
        return Err(SolveError::InvalidRequest("split, probes, and LNS require threads greater than one".to_string()));
    }
    if (controls.split || controls.probes > 0) && objectives.iter().any(|objective| objective.var().is_none()) {
        return Err(SolveError::InvalidRequest(
            "split and probe workers require every objective tier to be a materialized variable".to_string(),
        ));
    }
    let auxiliary_workers = controls.probes.saturating_add(controls.lns);
    if auxiliary_workers >= request.threads && auxiliary_workers > 0 {
        return Err(SolveError::InvalidRequest(format!(
            "{} probe and LNS workers leave no complete worker in a {}-thread portfolio",
            auxiliary_workers, request.threads
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn solve_cp_plan(
    model: &Model,
    plan: &CpSolvePlan,
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    request.validate()?;
    solve_cp_plan_inner(model, plan, super::WorkerAllocation::portfolio(request.threads), request, budget, sink, true)
}

/// Execute a CP plan against the same semantic model and request that the
/// orchestrator validated before compilation.
pub(crate) fn solve_cp_plan_validated(
    model: &Model,
    plan: &CpSolvePlan,
    allocation: super::WorkerAllocation,
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    solve_cp_plan_inner(model, plan, allocation, request, budget, sink, false)
}

fn solve_cp_plan_inner(
    model: &Model,
    plan: &CpSolvePlan,
    allocation: super::WorkerAllocation,
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
    validate_model: bool,
) -> Result<SolveResult, SolveError> {
    if request.assumptions != plan.assumptions
        || request.hints != plan.hints
        || request.primary_branch_scope != plan.primary_branch_scope
        || request.branch_order != plan.branch_order
    {
        return Err(SolveError::InvalidRequest(
            "assumptions, hints, primary_branch_scope, and branch_order must match the request used to compile the CP plan".to_string(),
        ));
    }
    if request.mode != plan.mode {
        return Err(SolveError::InvalidRequest("solve mode must match the request used to compile the CP plan".to_string()));
    }
    if budget.expired() {
        return Ok(SolveResult::unknown());
    }
    if validate_model {
        match model.validate_interruptible(budget.stop()) {
            Ok(true) => {}
            Ok(false) => return Ok(SolveResult::unknown()),
            Err(errors) => return Err(SolveError::Compile(errors.join("; "))),
        }
    }
    let search_stop = budget.search_stop();
    let engine_stop = search_stop.flag();
    let started = Instant::now();
    let mut result = if let Some(local_search) = &plan.local_search {
        super::integer_search::solve(super::integer_search::IntegerSearchRun {
            model,
            compiled: &plan.compiled,
            plan: local_search,
            allocation,
            request,
            budget,
            engine_stop,
            sink,
        })?
    } else if !plan.compiled.objectives().is_empty() {
        solve_lexicographic(LexicographicSolve {
            model,
            compiled: &plan.compiled,
            guidance: &plan.guidance,
            assumptions: &plan.assumptions,
            warm_start: plan.integer_warm_start.as_ref(),
            request,
            budget,
            engine_stop,
            sink,
            incremental: None,
            allocation,
        })?
    } else {
        solve_satisfaction(SatisfactionSolve {
            model,
            compiled: &plan.compiled,
            guidance: &plan.guidance,
            request,
            budget,
            engine_stop,
            started,
            incremental: None,
            allocation,
        })?
    };

    if let Some(report) = result.reports.first_mut() {
        append_compilation_metadata(&mut report.metadata, &plan.compiled);
    }

    if let Some(candidate) = &result.primal {
        verify_assumptions(candidate, &plan.assumptions)?;
    }
    if budget.expired() && result.status == SolveStatus::Unknown {
        result.message.get_or_insert_with(|| format!("search stopped: {:?}", budget.termination_reason()));
    }
    result.validate_contract()?;
    Ok(result)
}

/// Execute a reusable compiled CP model through the canonical exact control
/// plane. Only the engine adapter differs: session assumptions and learned
/// clauses are supplied to a complete assumption-aware CDCL portfolio per
/// tier. Split, probe, and LNS roles remain outside the persistent session.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cp_session_validated(
    model: &Model,
    compiled: &CompiledCp,
    physical_assumptions: &[Assumption],
    guidance: SearchGuidance,
    clauses: Arc<SharedClausePool>,
    next_worker: &mut usize,
    request: &SolveRequest,
    budget: &SolveBudget,
    sink: &mut dyn EventSink,
) -> Result<SolveResult, SolveError> {
    if budget.expired() {
        return Ok(SolveResult::unknown());
    }
    let search_stop = budget.search_stop();
    let engine_stop = search_stop.flag();
    let started = Instant::now();
    let allocation = super::WorkerAllocation::portfolio(request.threads);
    let mut incremental = IncrementalSearch::new(physical_assumptions, clauses, next_worker, allocation.workers(), guidance.clone());
    let mut result = if compiled.objectives().is_empty() {
        solve_satisfaction(SatisfactionSolve {
            model,
            compiled,
            guidance: &guidance,
            request,
            budget,
            engine_stop,
            started,
            incremental: Some(&mut incremental),
            allocation,
        })?
    } else {
        solve_lexicographic(LexicographicSolve {
            model,
            compiled,
            guidance: &guidance,
            assumptions: &request.assumptions,
            warm_start: None,
            request,
            budget,
            engine_stop,
            sink,
            incremental: Some(&mut incremental),
            allocation,
        })?
    };
    if let Some(candidate) = &result.primal {
        verify_assumptions(candidate, &request.assumptions)?;
    }
    if budget.expired() && result.status == SolveStatus::Unknown {
        result.message.get_or_insert_with(|| format!("search stopped: {:?}", budget.termination_reason()));
    }
    result.validate_contract()?;
    Ok(result)
}

struct SatisfactionSolve<'a, 'session, 'assumptions> {
    model: &'a Model,
    compiled: &'a CompiledCp,
    guidance: &'a SearchGuidance,
    request: &'a SolveRequest,
    budget: &'a SolveBudget,
    engine_stop: &'a std::sync::atomic::AtomicBool,
    started: Instant,
    incremental: Option<&'session mut IncrementalSearch<'assumptions>>,
    allocation: super::WorkerAllocation,
}

fn solve_satisfaction(context: SatisfactionSolve<'_, '_, '_>) -> Result<SolveResult, SolveError> {
    let SatisfactionSolve { model, compiled, guidance, request, budget, engine_stop, started, mut incremental, allocation } = context;
    let first_worker = incremental.as_ref().map(|search| search.next_worker());
    let problem = clone_problem(compiled.problem());
    let outcome = if let Some(search) = incremental.as_deref_mut() {
        search.solve_csp(problem, engine_stop, request.seed, request.limits.conflicts)
    } else {
        let options = portfolio::normalize_options(false, false, cp_options(request, allocation, request.seed, request.limits.conflicts));
        portfolio::solve_csp(problem, engine_stop, options, guidance.clone())
    };
    let conflict_limit_reached =
        request.limits.conflicts.is_some_and(|limit| outcome.stats.failures >= limit) && !outcome.decided && !budget.expired();
    let primal = outcome
        .solution
        .as_ref()
        .map(|values| candidate_if_running(model, compiled, values, budget, VerificationLevel::Final))
        .transpose()?
        .flatten();
    let status = if primal.is_some() {
        SolveStatus::Satisfiable
    } else if outcome.solution.is_some() {
        SolveStatus::Unknown
    } else if outcome.decided {
        SolveStatus::Unsatisfiable
    } else {
        SolveStatus::Unknown
    };
    let mut metadata = cp_metadata(outcome.shared_clauses, outcome.imported_clauses, None, None, None);
    metadata.push(("csp_search".to_string(), outcome.search_kind.to_string()));
    if let Some(first_worker) = first_worker {
        let workers = incremental.as_ref().map_or(0, |search| search.next_worker().saturating_sub(first_worker));
        metadata.push(("clause_session_workers".to_string(), workers.to_string()));
        metadata.push(("clause_session_first_worker".to_string(), first_worker.to_string()));
        metadata.push((
            "clause_session_next_worker".to_string(),
            incremental.as_ref().map_or(first_worker, |search| search.next_worker()).to_string(),
        ));
    }
    Ok(SolveResult {
        status,
        primal,
        bounds: Vec::new(),
        proof: completion_proof(status, 0, outcome.decided && outcome.solution.is_none()),
        reports: vec![EngineReport {
            engine: Some(EngineKind::IntegerExact),
            search: outcome.stats,
            elapsed: started.elapsed(),
            improvements: u64::from(outcome.solution.is_some()),
            metadata,
        }],
        message: conflict_limit_reached.then(|| "search stopped: ConflictLimit".to_string()),
    })
}

struct LexicographicSolve<'a, 'session, 'assumptions> {
    model: &'a Model,
    compiled: &'a CompiledCp,
    guidance: &'a SearchGuidance,
    assumptions: &'a [super::SemanticAssumption],
    warm_start: Option<&'a integer::IntegerWarmStartPlan>,
    request: &'a SolveRequest,
    budget: &'a SolveBudget,
    engine_stop: &'a std::sync::atomic::AtomicBool,
    sink: &'a mut dyn EventSink,
    incremental: Option<&'session mut IncrementalSearch<'assumptions>>,
    allocation: super::WorkerAllocation,
}

fn solve_lexicographic(context: LexicographicSolve<'_, '_, '_>) -> Result<SolveResult, SolveError> {
    let LexicographicSolve {
        model,
        compiled,
        guidance,
        assumptions,
        warm_start,
        request,
        budget,
        engine_stop,
        sink,
        mut incremental,
        allocation,
    } = context;
    let objective_count = compiled.objectives().len();
    let first_incremental_worker = incremental.as_ref().map(|search| search.next_worker());
    let mut reports = Vec::new();
    let mut initial_incumbent = None;
    let mut primal = None;
    if let (Some(warm_plan), Some(warm_stops)) = (warm_start.filter(|_| !budget.expired()), budget.warm_start_stops()) {
        let warm_started = Instant::now();
        match super::integer_search::warm_start(model, compiled, warm_plan, request, budget, warm_stops.search(), warm_stops.transfer()) {
            Ok(Some(warm)) => {
                let candidate_control = sink.emit(SolveEvent::Candidate(warm.candidate.clone()))?;
                let progress_control = if candidate_control == EventControl::Continue {
                    sink.emit(SolveEvent::Progress {
                        engine: EngineKind::IntegerLocalSearch,
                        objectives: warm.candidate.objectives().to_vec(),
                        elapsed: budget.elapsed(),
                    })?
                } else {
                    EventControl::Stop
                };
                if progress_control == EventControl::Stop {
                    budget.cancel_with(TerminationReason::EventSink);
                }
                primal = Some(promote_warm_candidate(&warm.candidate));
                initial_incumbent = Some(portfolio::InitialIncumbent { solution: warm.physical_solution, value: warm.physical_objective });
                reports.push(warm.report);
            }
            Ok(None) if warm_stops.transfer().load(std::sync::atomic::Ordering::Acquire) && !budget.expired() => {
                reports.push(optional_warm_start_report(warm_started.elapsed(), "time allowance exhausted"));
            }
            Ok(None) => {}
            Err(error) => {
                reports.push(optional_warm_start_report(warm_started.elapsed(), &error.to_string()));
            }
        }
    }
    let exact_started = Instant::now();
    let mut problem = None;
    let mut output = Vec::new();
    let mut total_stats = SolveStats::default();
    let mut shared_clauses = 0usize;
    let mut imported_clauses = 0u64;
    let mut split_jobs = None;
    let mut probe_stats = None;
    let mut lns_stats = None;
    let mut remaining_conflicts = request.limits.conflicts;
    let mut proven_prefix = Vec::with_capacity(objective_count);
    let mut scoped_physical_prefix = Vec::with_capacity(objective_count.saturating_sub(1));
    let mut bounds = Vec::with_capacity(objective_count);
    let mut complete = false;
    let mut unsatisfiable = false;
    let mut stopped_tier = None;
    let mut conflict_limit_reached = false;
    let mut linear_backends = std::collections::BTreeSet::new();

    for (tier, objective) in compiled.objectives().iter().enumerate() {
        if engine_stop.load(std::sync::atomic::Ordering::Acquire) || remaining_conflicts == Some(0) {
            conflict_limit_reached = remaining_conflicts == Some(0) && !budget.expired();
            stopped_tier = Some(tier);
            break;
        }

        let final_tier = tier + 1 == objective_count;
        let tier_problem = if incremental.is_some() {
            // Session prefixes are assumptions, so no mutable root needs to
            // survive between tiers. Keep only the persistent compiled template
            // plus the exact worker copies that run this tier.
            let mut tier_problem = clone_problem(compiled.problem());
            tier_problem.objective = Some(objective.clone());
            tier_problem
        } else {
            let problem_root = problem.get_or_insert_with(|| clone_problem(compiled.problem()));
            problem_root.objective = Some(objective.clone());
            // Ordinary lexicographic solving posts each proved prefix into this
            // retained root. The final tier consumes it directly.
            if final_tier {
                problem.take().expect("the active lexicographic tier has a CP problem")
            } else {
                clone_problem(problem.as_ref().expect("the active lexicographic tier has a CP problem"))
            }
        };
        let var_objective = tier_problem.var_objective().is_some();
        let mut linear_controls = request.linear;
        if let Some(remaining) = budget.remaining() {
            linear_controls.root_time = linear_controls.root_time.min(remaining.saturating_sub(Duration::from_millis(10)));
        }
        let relaxation = linear::solve_root(
            &tier_problem.solver,
            &tier_problem.search,
            objective.search(),
            objective.minimizing(),
            linear_controls,
            engine_stop,
        );
        if let Some(backend) = relaxation.backend {
            linear_backends.insert(backend);
        }
        let mut tier_guidance = guidance.clone();
        tier_guidance.linear = relaxation.search.clone();
        if tier_guidance.initial_phase.len() != tier_problem.solver.store.num_vars() {
            tier_guidance.initial_phase.resize(tier_problem.solver.store.num_vars(), None);
        }
        if relaxation.phase.len() == tier_guidance.initial_phase.len() {
            for (current, &proposed) in tier_guidance.initial_phase.iter_mut().zip(&relaxation.phase) {
                if current.is_none() {
                    *current = proposed;
                }
            }
        }
        if let Some(value) = relaxation.bound {
            upsert_bound(
                &mut bounds,
                Bound {
                    tier,
                    value,
                    method: format!("{} LP relaxation with exact rational recertification", relaxation.backend.unwrap_or("linear")),
                },
            );
        }
        let progress_prefix = proven_prefix.clone();
        let mut event_error = None;
        let mut outcome = {
            let mut progress = |objective, assignment: Option<&[i32]>| {
                if event_error.is_some() {
                    return;
                }
                let mut objectives = progress_prefix.clone();
                objectives.push(objective);
                if let Some(values) = assignment {
                    let candidate =
                        candidate_if_running(model, compiled, values, budget, VerificationLevel::Transfer).and_then(|candidate| {
                            if let Some(candidate) = &candidate {
                                verify_assumptions(candidate, assumptions)?;
                            }
                            Ok(candidate)
                        });
                    match candidate {
                        Ok(Some(candidate)) => match sink.emit(SolveEvent::Candidate(candidate)) {
                            Ok(EventControl::Continue) => {}
                            Ok(EventControl::Stop) => {
                                budget.cancel_with(TerminationReason::EventSink);
                                return;
                            }
                            Err(error) => {
                                event_error = Some(error);
                                budget.cancel_with(TerminationReason::EventSink);
                                return;
                            }
                        },
                        Ok(None) => return,
                        Err(error) => {
                            event_error = Some(error);
                            budget.cancel_with(TerminationReason::EventSink);
                            return;
                        }
                    }
                }
                match sink.emit(SolveEvent::Progress { engine: EngineKind::IntegerExact, objectives, elapsed: budget.elapsed() }) {
                    Ok(EventControl::Continue) => {}
                    Ok(EventControl::Stop) => budget.cancel_with(TerminationReason::EventSink),
                    Err(error) => {
                        event_error = Some(error);
                        budget.cancel_with(TerminationReason::EventSink);
                    }
                }
            };
            if let Some(search) = incremental.as_deref_mut() {
                search.solve_cop(
                    tier_problem,
                    &scoped_physical_prefix,
                    engine_stop,
                    request.seed.wrapping_add(tier as u64),
                    remaining_conflicts,
                    &tier_guidance,
                    request.publish_incumbent_assignments,
                    &mut progress,
                )
            } else {
                let options = portfolio::normalize_options(
                    true,
                    var_objective,
                    cp_options(request, allocation, request.seed.wrapping_add(tier as u64), remaining_conflicts),
                );
                portfolio::solve_cop_with_progress(
                    tier_problem,
                    false,
                    engine_stop,
                    &mut output,
                    options,
                    tier_guidance,
                    initial_incumbent.take(),
                    relaxation.bound,
                    request.publish_incumbent_assignments,
                    &mut progress,
                )
                .map_err(SolveError::Engine)?
            }
        };
        if let Some(error) = event_error {
            return Err(error);
        }
        if let (Some((_, value)), Some(bound)) = (&outcome.best, relaxation.bound) {
            let reaches_bound = if objective.minimizing() { *value <= bound } else { *value >= bound };
            outcome.proved |= reaches_bound;
        }

        merge_search_stats(&mut total_stats, relaxation.stats);
        merge_search_stats(&mut total_stats, outcome.stats);
        shared_clauses = shared_clauses.saturating_add(outcome.shared_clauses);
        imported_clauses = imported_clauses.saturating_add(outcome.imported_clauses);
        add_optional_pair(&mut split_jobs, outcome.split_jobs);
        add_optional_pair(&mut probe_stats, outcome.probe_stats);
        add_optional_pair(&mut lns_stats, outcome.lns_stats);
        if let Some(remaining) = &mut remaining_conflicts {
            *remaining = remaining.saturating_sub(outcome.stats.failures);
        }

        let tier_candidate = outcome
            .best
            .as_ref()
            .map(|(values, _)| candidate_if_running(model, compiled, values, budget, VerificationLevel::Final))
            .transpose()?
            .flatten();
        if let (Some(candidate), Some((_, physical_value))) = (&tier_candidate, &outcome.best) {
            if candidate.objectives().get(..tier) != Some(proven_prefix.as_slice()) {
                return Err(SolveError::InvalidResult(format!("CP tier {tier} candidate violates the proved lexicographic prefix")));
            }
            if candidate.objectives().get(tier) != Some(physical_value) {
                return Err(SolveError::InvalidResult(format!(
                    "CP tier {tier} reported objective {physical_value}, canonical replay produced {:?}",
                    candidate.objectives().get(tier)
                )));
            }
            primal = Some(candidate.clone());
        }

        match (&outcome.best, outcome.proved) {
            (None, true) if primal.is_some() => {
                return Err(SolveError::InvalidResult(
                    "CP root is inconsistent with a canonically verified warm-start incumbent".to_string(),
                ));
            }
            (None, true) if tier == 0 => {
                unsatisfiable = true;
                complete = true;
                bounds.clear();
                break;
            }
            (None, true) => {
                return Err(SolveError::InvalidResult(format!("CP proved the fixed lexicographic subproblem infeasible at tier {tier}")));
            }
            (Some((_, value)), true) if tier_candidate.is_some() => {
                proven_prefix.push(*value);
                upsert_bound(
                    &mut bounds,
                    Bound { tier, value: *value, method: format!("complete CP search for lexicographic tier {tier}") },
                );
                if final_tier {
                    complete = true;
                    break;
                }
                if incremental.is_some() {
                    let variable = objective.var().ok_or_else(|| {
                        SolveError::Unsupported("persistent clause sharing requires materialized lexicographic objectives".to_string())
                    })?;
                    let value = i32::try_from(*value)
                        .map_err(|_| SolveError::InvalidResult(format!("materialized objective tier {tier} escaped its i32 domain")))?;
                    scoped_physical_prefix.push(Assumption { var: variable, op: crate::search::AssumptionOp::Eq, value });
                } else {
                    compiled
                        .fix_objective(problem.as_mut().expect("a non-final lexicographic tier retains its CP problem"), tier, *value)
                        .map_err(|error| SolveError::Engine(error.reason))?;
                }
                if remaining_conflicts == Some(0) {
                    conflict_limit_reached = !budget.expired();
                    stopped_tier = Some(tier + 1);
                    break;
                }
            }
            _ => {
                stopped_tier = Some(tier);
                conflict_limit_reached = remaining_conflicts == Some(0) && !budget.expired();
                break;
            }
        }
    }

    let status = if unsatisfiable {
        SolveStatus::Unsatisfiable
    } else if complete {
        SolveStatus::Optimal
    } else if primal.is_some() {
        SolveStatus::Satisfiable
    } else {
        SolveStatus::Unknown
    };
    let proof = completion_proof(status, objective_count, complete);
    let message = stopped_tier.map(|tier| {
        let reason = if conflict_limit_reached { TerminationReason::ConflictLimit } else { budget.termination_reason() };
        format!("CP lexicographic search stopped before proving tier {tier}: {reason:?}")
    });

    let mut metadata = cp_metadata(shared_clauses, imported_clauses, split_jobs, probe_stats, lns_stats);
    if !linear_backends.is_empty() {
        metadata.push(("linear_backends".to_string(), linear_backends.into_iter().collect::<Vec<_>>().join(",")));
    }
    if let (Some(first_worker), Some(search)) = (first_incremental_worker, incremental.as_ref()) {
        metadata.push(("clause_session_workers".to_string(), search.next_worker().saturating_sub(first_worker).to_string()));
        metadata.push(("clause_session_first_worker".to_string(), first_worker.to_string()));
        metadata.push(("clause_session_next_worker".to_string(), search.next_worker().to_string()));
    }
    reports.push(EngineReport {
        engine: Some(EngineKind::IntegerExact),
        search: total_stats,
        elapsed: exact_started.elapsed(),
        improvements: total_stats.solutions,
        metadata,
    });

    Ok(SolveResult { status, primal, bounds, proof, reports, message })
}

fn upsert_bound(bounds: &mut Vec<Bound>, replacement: Bound) {
    if let Some(bound) = bounds.iter_mut().find(|bound| bound.tier == replacement.tier) {
        *bound = replacement;
    } else {
        bounds.push(replacement);
    }
}

fn optional_warm_start_report(elapsed: Duration, rejection: &str) -> EngineReport {
    EngineReport {
        engine: Some(EngineKind::IntegerLocalSearch),
        search: SolveStats::default(),
        elapsed,
        improvements: 0,
        metadata: vec![
            ("ls_role".to_string(), "optional_warm_start".to_string()),
            ("ls_outcome".to_string(), "rejected".to_string()),
            ("ls_rejection".to_string(), rejection.to_string()),
        ],
    }
}

fn promote_warm_candidate(candidate: &CandidateSolution) -> CandidateSolution {
    CandidateSolution::verified(
        candidate.assignment().clone(),
        candidate.objectives().to_vec(),
        candidate.source(),
        VerificationLevel::Final,
    )
}

fn cp_options(request: &SolveRequest, allocation: super::WorkerAllocation, seed: u64, conflict_limit: Option<u64>) -> RunOptions {
    RunOptions {
        seed,
        workers: allocation.workers(),
        split: request.cp.split,
        probes: request.cp.probes,
        lns: request.cp.lns,
        no_learn_csp: request.cp.no_learn_csp,
        force_scope_reasons: request.cp.force_scope_reasons,
        shared_pool_capacity: request.cp.shared_pool_capacity,
        conflict_limit,
    }
}

fn add_optional_pair(total: &mut Option<(u64, u64)>, part: Option<(u64, u64)>) {
    let Some((left, right)) = part else {
        return;
    };
    let (total_left, total_right) = total.get_or_insert((0, 0));
    *total_left = total_left.saturating_add(left);
    *total_right = total_right.saturating_add(right);
}

pub(super) fn verify_assumptions(candidate: &CandidateSolution, assumptions: &[super::SemanticAssumption]) -> Result<(), SolveError> {
    verify_assumptions_interruptible(candidate, assumptions, &std::sync::atomic::AtomicBool::new(false))
}

pub(super) fn verify_assumptions_interruptible(
    candidate: &CandidateSolution,
    assumptions: &[super::SemanticAssumption],
    stop: &std::sync::atomic::AtomicBool,
) -> Result<(), SolveError> {
    for (index, assumption) in assumptions.iter().enumerate() {
        if index & 0xff == 0 && stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err(SolveError::Interrupted("assumption verification was interrupted".to_string()));
        }
        let value = candidate
            .assignment()
            .integers
            .get(assumption.variable)
            .copied()
            .flatten()
            .ok_or_else(|| SolveError::InvalidResult(format!("assumed integer variable {} is unassigned", assumption.variable)))?;
        let expected = i64::from(assumption.value);
        let holds = match assumption.operation {
            super::SemanticAssumptionOp::Eq => value == expected,
            super::SemanticAssumptionOp::Ne => value != expected,
            super::SemanticAssumptionOp::Le => value <= expected,
            super::SemanticAssumptionOp::Lt => value < expected,
            super::SemanticAssumptionOp::Ge => value >= expected,
            super::SemanticAssumptionOp::Gt => value > expected,
        };
        if !holds {
            return Err(SolveError::InvalidResult(format!("candidate violates assumption on integer variable {}", assumption.variable)));
        }
    }
    Ok(())
}

fn candidate(
    model: &Model,
    compiled: &CompiledCp,
    values: &[i32],
    stop: &std::sync::atomic::AtomicBool,
    verification: VerificationLevel,
) -> Result<CandidateSolution, SolveError> {
    let (assignment, objectives) = decode_candidate(model, compiled, values, stop)?;
    let objectives = verify_cp_assignment(model, &assignment, &objectives, stop)?;
    Ok(CandidateSolution::verified(assignment, objectives, EngineKind::IntegerExact, verification))
}

fn decode_candidate(
    model: &Model,
    compiled: &CompiledCp,
    values: &[i32],
    stop: &std::sync::atomic::AtomicBool,
) -> Result<(Assignment, Vec<i64>), SolveError> {
    let mut decoded = compiled.decode(values).map_err(SolveError::InvalidResult)?;
    let user_integer_count = model.int_vars().len();
    if decoded.integers.len() < user_integer_count {
        return Err(SolveError::InvalidResult(format!(
            "CP assignment decoded {} integer values for {user_integer_count} user variables",
            decoded.integers.len()
        )));
    }
    decoded.integers.truncate(user_integer_count);
    let assignment = Assignment { integers: decoded.integers, sets: decoded.sets, lists: Vec::new(), intervals: Vec::new() };
    let objectives = model
        .objectives()
        .iter()
        .map(|objective| match objective {
            crate::model::Objective::IntExpr { expr, .. } => {
                super::evaluate_int_expr_interruptible(expr, &|variable| assignment.integers.get(variable.0).copied().flatten(), stop)?
                    .ok_or_else(|| SolveError::InvalidResult("integer objective is undefined or overflows".to_string()))
            }
            crate::model::Objective::ListTerms { .. } | crate::model::Objective::Makespan { .. } => {
                Err(SolveError::InvalidResult("CP candidate contains a non-integer objective".to_string()))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((assignment, objectives))
}

fn verify_cp_assignment(
    model: &Model,
    assignment: &Assignment,
    objectives: &[i64],
    stop: &std::sync::atomic::AtomicBool,
) -> Result<Vec<i64>, SolveError> {
    match super::verify_semantic_assignment_validated_interruptible(model, assignment, objectives, stop) {
        Ok(objectives) => Ok(objectives),
        Err(error) => {
            let prefix = assignment.integers.iter().take(16).copied().collect::<Vec<_>>();
            let reason = match error {
                SolveError::InvalidResult(reason) => reason,
                other => return Err(other),
            };
            Err(SolveError::InvalidResult(format!("{reason}; CP assignment prefix {prefix:?}")))
        }
    }
}

pub(super) fn candidate_if_running(
    model: &Model,
    compiled: &CompiledCp,
    values: &[i32],
    budget: &SolveBudget,
    verification: VerificationLevel,
) -> Result<Option<CandidateSolution>, SolveError> {
    candidate_if_running_with_stop(model, compiled, values, budget, budget.stop(), verification)
}

pub(super) fn candidate_if_running_with_stop(
    model: &Model,
    compiled: &CompiledCp,
    values: &[i32],
    budget: &SolveBudget,
    stop: &std::sync::atomic::AtomicBool,
    verification: VerificationLevel,
) -> Result<Option<CandidateSolution>, SolveError> {
    let replay = if verification == VerificationLevel::Final {
        // Decode, objective evaluation, and semantic replay all live inside
        // the same interruptible finalization boundary. The first pass borrows
        // the compiled state; only the work is repeated during deadline grace.
        super::verify_final_with_budget(budget, |stop| candidate(model, compiled, values, stop, verification))
    } else {
        candidate(model, compiled, values, stop, verification)
    };
    match replay {
        Ok(candidate) => Ok(Some(candidate)),
        Err(SolveError::Interrupted(_)) if stop.load(std::sync::atomic::Ordering::Acquire) || budget.expired() => Ok(None),
        Err(error) => Err(error),
    }
}

fn completion_proof(status: SolveStatus, objective_tiers: usize, complete: bool) -> Option<ProofClaim> {
    if !complete {
        return None;
    }
    let conclusion = match status {
        SolveStatus::Optimal => ProvenConclusion::Optimal,
        SolveStatus::Unsatisfiable => ProvenConclusion::Unsatisfiable,
        SolveStatus::Satisfiable | SolveStatus::Unknown | SolveStatus::Unsupported => return None,
    };
    Some(ProofClaim::complete_search(EngineKind::IntegerExact, conclusion, objective_tiers))
}

fn cp_metadata(
    shared_clauses: usize,
    imported_clauses: u64,
    split_jobs: Option<(u64, u64)>,
    probe_stats: Option<(u64, u64)>,
    lns_stats: Option<(u64, u64)>,
) -> Vec<(String, String)> {
    let mut metadata =
        vec![("shared_clauses".to_string(), shared_clauses.to_string()), ("imported_clauses".to_string(), imported_clauses.to_string())];
    if let Some((split, completed)) = split_jobs {
        metadata.extend([("split_jobs".to_string(), split.to_string()), ("completed_jobs".to_string(), completed.to_string())]);
    }
    if let Some((attempts, unsatisfiable)) = probe_stats {
        metadata.extend([("probe_attempts".to_string(), attempts.to_string()), ("probe_unsat".to_string(), unsatisfiable.to_string())]);
    }
    if let Some((attempts, improved)) = lns_stats {
        metadata.extend([("lns_attempts".to_string(), attempts.to_string()), ("lns_improved".to_string(), improved.to_string())]);
    }
    metadata
}

fn append_compilation_metadata(metadata: &mut Vec<(String, String)>, compiled: &CompiledCp) {
    let stats = compiled.compilation_stats();
    metadata.extend([
        ("table_instances".to_string(), stats.table_instances.to_string()),
        ("table_templates".to_string(), stats.table_templates.to_string()),
        ("native_intensions".to_string(), stats.native_intensions.to_string()),
        ("fallback_intensions".to_string(), stats.fallback_intensions.to_string()),
        ("native_all_different_except".to_string(), stats.native_all_different_except.to_string()),
        ("native_no_overlap".to_string(), stats.native_no_overlap.to_string()),
        ("no_overlap_fallback".to_string(), stats.no_overlap_fallback.to_string()),
        ("native_cumulative".to_string(), stats.native_cumulative.to_string()),
        ("cumulative_fallback".to_string(), stats.cumulative_fallback.to_string()),
        ("native_lex".to_string(), stats.native_lex.to_string()),
        ("native_bin_packing".to_string(), stats.native_bin_packing.to_string()),
        ("native_bin_loads".to_string(), stats.native_bin_loads.to_string()),
        ("physical_propagators".to_string(), stats.physical_propagators.to_string()),
    ]);
}
