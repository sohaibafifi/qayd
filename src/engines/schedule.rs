//! Structured interval scheduling backend.
//!
//! This module owns the reusable scheduling-engine orchestration: lowering the
//! schedule IR to intervals, choosing the exact CDCL or chronological
//! backend, maintaining makespan bounds, and replaying optional-mode incumbents.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;

use crate::constraints::intension;
use crate::constraints::interval as interval_constraints;
use crate::expr;
use crate::ids::{IntervalId, VarId};
use crate::model::list::{Resource, Schedule};
use crate::search::{self, Objective as SearchObjective, SearchControl, SolveStats};
use crate::Solver;

use super::{CollectionCompileContext, CompileFailure};

/// Runtime options for the domain scheduling engine.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Seed for CDCL-backed searches.
    pub seed: u64,
    /// Use the CDCL backend for optional-mode schedules. It is correct but not
    /// yet the strongest default on this model shape.
    pub optional_modes_cdcl: bool,
}

/// Solver status returned by the scheduling engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    Satisfiable,
    Unsatisfiable,
    Unknown,
}

impl Status {
    /// Stable status text for frontends.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Optimal => "OPTIMAL",
            Self::Satisfiable => "SATISFIABLE",
            Self::Unsatisfiable => "UNSATISFIABLE",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// Structured interval schedule result.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub status: Status,
    pub objective: Option<i64>,
    pub starts: Vec<i64>,
    /// Presence of each operation. Moded operations are mandatory.
    pub presences: Vec<bool>,
    /// Chosen machine per operation, or `-1` for fixed intervals.
    pub machines: Vec<i64>,
    /// Stable semantic execution mode per operation, when available.
    pub modes: Vec<Option<usize>>,
    pub stats: SolveStats,
}

/// Observed exact-scheduling dispatch, independent of the requested options.
/// A kernel is recorded only at its search call, after backend construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Execution {
    pub encoding: Option<&'static str>,
    pub search_kernel: Option<&'static str>,
}

/// A borrowed, complete physical incumbent available before its progress event.
/// Canonical semantic verification remains the responsibility of the caller.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScheduleIncumbent<'a> {
    pub starts: &'a [i64],
    pub presences: &'a [bool],
    pub machines: &'a [i64],
    pub modes: &'a [Option<usize>],
}

fn interrupted_outcome(stats: SolveStats) -> Outcome {
    Outcome {
        status: Status::Unknown,
        objective: None,
        starts: Vec::new(),
        presences: Vec::new(),
        machines: Vec::new(),
        modes: Vec::new(),
        stats,
    }
}

/// Schedule shape already admitted by the exact backend, including every
/// numeric conversion required by the i32 CP representation.
#[derive(Clone)]
pub(crate) struct CompiledSchedule {
    kind: CompiledScheduleKind,
}

#[derive(Clone)]
enum CompiledScheduleKind {
    Fixed(CompiledFixedSchedule),
    Moded(CompiledModedSchedule),
}

#[derive(Clone)]
struct CompiledFixedSchedule {
    intervals: Vec<CompiledFixedInterval>,
    precedences: Vec<(usize, usize)>,
    resources: Vec<CompiledFixedResource>,
    minimize_makespan: bool,
    max_horizon: i32,
}

#[derive(Clone, Copy)]
struct CompiledFixedInterval {
    duration: i32,
    start_max: i32,
    optional: bool,
}

#[derive(Clone)]
enum CompiledFixedResource {
    NoOverlap(Vec<usize>),
    Cumulative { demands: Vec<(usize, i32)>, capacity: i32 },
}

#[derive(Clone)]
struct CompiledModedSchedule {
    operations: Vec<Vec<CompiledMode>>,
    precedences: Vec<(usize, usize)>,
    minimize_makespan: bool,
    max_horizon: i32,
}

#[derive(Clone, Copy)]
struct CompiledMode {
    reference: Option<usize>,
    machine: usize,
    machine_value: i64,
    duration: i32,
    start_min: i32,
    start_max: i32,
}

pub(crate) fn lower(context: &CollectionCompileContext<'_>, stop: &AtomicBool) -> Result<Option<CompiledSchedule>, String> {
    let Some(schedule) = context.physical().schedule.as_ref() else {
        return Ok(None);
    };
    compile_schedule(schedule, stop)
}

pub(crate) fn compile(context: &CollectionCompileContext<'_>, stop: &AtomicBool) -> Result<CompiledSchedule, CompileFailure> {
    debug_assert!(!context.semantic().intervals().is_empty());
    match lower(context, stop) {
        Ok(Some(compiled)) => Ok(compiled),
        Ok(None) if schedule_stopped(stop) => Err(CompileFailure::Interrupted { phase: "during exact schedule lowering" }),
        Ok(None) => {
            Err(CompileFailure::Unsupported { code: "schedule-shape", detail: "semantic model is outside the exact scheduling shape" })
        }
        Err(reason) => Err(CompileFailure::Invalid { reason }),
    }
}

pub(crate) fn solve_compiled<F>(
    compiled: &CompiledSchedule,
    stop: &AtomicBool,
    options: Options,
    execution: &mut Execution,
    on_improve: F,
) -> Result<Outcome, String>
where
    F: FnMut(i64),
{
    solve_compiled_with_lower_bound(compiled, stop, options, None, execution, on_improve)
}

/// Solve with a certified structural makespan lower bound. Bounds relying on
/// nonnegative dates are admitted only when that property holds for the shape.
pub(crate) fn solve_compiled_with_lower_bound<F>(
    compiled: &CompiledSchedule,
    stop: &AtomicBool,
    options: Options,
    objective_lb: Option<i64>,
    execution: &mut Execution,
    mut on_improve: F,
) -> Result<Outcome, String>
where
    F: FnMut(i64),
{
    solve_compiled_with_observer(compiled, stop, options, objective_lb, execution, |objective, _| on_improve(objective))
}

/// Observe complete cached incumbents while the search budget is still live.
/// Progress without a reconstructed physical snapshot is reported as `None`.
pub(crate) fn solve_compiled_with_observer<F>(
    compiled: &CompiledSchedule,
    stop: &AtomicBool,
    options: Options,
    objective_lb: Option<i64>,
    execution: &mut Execution,
    on_improve: F,
) -> Result<Outcome, String>
where
    F: FnMut(i64, Option<ScheduleIncumbent<'_>>),
{
    solve_prepared(compiled, stop, options, objective_lb, execution, on_improve)
}

/// Mode-interval layout of an optional-mode schedule. Built deterministically,
/// so VarId / IntervalId / disjunctive-pair indices match across fresh solver
/// instances created from the same schedule. That property is required to replay
/// the incumbent assignment on a clean solver after optimization.
struct ModedBuild {
    /// Physical modes grouped by semantic operation.
    op_modes: Vec<Vec<BuiltMode>>,
    all_modes: Vec<IntervalId>,
    all_durations: Vec<i32>,
}

#[derive(Clone, Copy)]
struct BuiltMode {
    reference: Option<usize>,
    machine: usize,
    machine_value: i64,
    interval: IntervalId,
    duration: i32,
}

type ModeScheduleIncumbent = (i64, Vec<i64>, Vec<i64>, Vec<Option<usize>>);

/// A complete incumbent reconstructed while its search budget is still live.
/// Returning this owned value after interruption performs no further solving.
struct CachedSchedule {
    objective: i64,
    starts: Vec<i64>,
    presences: Vec<bool>,
    machines: Vec<i64>,
    modes: Vec<Option<usize>>,
}

impl CachedSchedule {
    fn snapshot(&self) -> ScheduleIncumbent<'_> {
        ScheduleIncumbent { starts: &self.starts, presences: &self.presences, machines: &self.machines, modes: &self.modes }
    }
}

fn finish_cached_schedule(cache: Option<CachedSchedule>, search_best: Option<i64>, complete: bool, stats: SolveStats) -> Outcome {
    let Some(cache) = cache else {
        let mut outcome = interrupted_outcome(stats);
        if complete && search_best.is_none() {
            outcome.status = Status::Unsatisfiable;
        }
        return outcome;
    };
    Outcome {
        status: if complete && search_best == Some(cache.objective) { Status::Optimal } else { Status::Satisfiable },
        objective: Some(cache.objective),
        starts: cache.starts,
        presences: cache.presences,
        machines: cache.machines,
        modes: cache.modes,
        stats,
    }
}

/// Earliest feasible dates for positive-duration unary schedules with fixed
/// presence and pair orders. Every selected resource pair becomes a precedence,
/// so a topological longest-path pass is a complete feasibility check for this
/// shape. Cumulative resources and zero-duration cycles use the CP replay path.
#[allow(clippy::too_many_arguments)]
fn earliest_unary_dates(
    durations: &[i64],
    releases: &[i64],
    latest: &[i64],
    present: &[bool],
    precedences: &[(usize, usize)],
    ordered_pairs: impl Iterator<Item = (usize, usize)>,
    stop: &AtomicBool,
) -> Result<Option<Vec<i64>>, String> {
    let count = durations.len();
    let mut outgoing = vec![Vec::new(); count];
    let mut indegrees = vec![0usize; count];
    for (before, after) in precedences.iter().copied().chain(ordered_pairs) {
        if schedule_stopped(stop) {
            return Ok(None);
        }
        if !present[before] || !present[after] {
            continue;
        }
        outgoing[before].push(after);
        indegrees[after] = indegrees[after].checked_add(1).ok_or("schedule predecessor count overflow")?;
    }
    let mut starts = vec![0; count];
    let mut ready = VecDeque::new();
    let mut expected = 0usize;
    for operation in 0..count {
        if schedule_stopped(stop) {
            return Ok(None);
        }
        if present[operation] {
            expected += 1;
            starts[operation] = releases[operation];
            if indegrees[operation] == 0 {
                ready.push_back(operation);
            }
        }
    }
    let mut visited = 0usize;
    while let Some(operation) = ready.pop_front() {
        if schedule_stopped(stop) {
            return Ok(None);
        }
        if starts[operation] > latest[operation] {
            return Err("internal error: reconstructed unary schedule exceeds a start window".to_string());
        }
        visited += 1;
        let end = starts[operation].checked_add(durations[operation]).ok_or("schedule completion overflows i64")?;
        for &successor in &outgoing[operation] {
            if schedule_stopped(stop) {
                return Ok(None);
            }
            starts[successor] = starts[successor].max(end);
            indegrees[successor] -= 1;
            if indegrees[successor] == 0 {
                ready.push_back(successor);
            }
        }
    }
    if visited != expected {
        return Err("internal error: selected positive-duration schedule contains a precedence cycle".to_string());
    }
    Ok(Some(starts))
}

/// A bounded serial construction for mandatory, positive unary operations.
/// Ready operations are appended to every resource they use, choosing the
/// earliest completion first. Failure is only a failed heuristic attempt.
fn construct_unary_dates(
    durations: &[i64],
    releases: &[i64],
    latest: &[i64],
    precedences: &[(usize, usize)],
    groups: &[Vec<usize>],
    stop: &AtomicBool,
) -> Option<Vec<i64>> {
    let count = durations.len();
    let mut outgoing = vec![Vec::new(); count];
    let mut indegrees = vec![0usize; count];
    for &(before, after) in precedences {
        if schedule_stopped(stop) || before == after {
            return None;
        }
        outgoing[before].push(after);
        indegrees[after] = indegrees[after].checked_add(1)?;
    }
    let mut memberships = vec![Vec::new(); count];
    for (resource, group) in groups.iter().enumerate() {
        for &operation in group {
            if schedule_stopped(stop) || memberships[operation].last() == Some(&resource) {
                return None;
            }
            memberships[operation].push(resource);
        }
    }
    let mut available = vec![i64::MIN; groups.len()];
    let mut earliest = releases.to_vec();
    let mut starts = vec![0; count];
    let mut scheduled = vec![false; count];
    for _ in 0..count {
        let mut selected = None;
        for operation in 0..count {
            if schedule_stopped(stop) {
                return None;
            }
            if scheduled[operation] || indegrees[operation] != 0 {
                continue;
            }
            let mut start = earliest[operation];
            for &resource in &memberships[operation] {
                if schedule_stopped(stop) {
                    return None;
                }
                start = start.max(available[resource]);
            }
            if start > latest[operation] {
                return None;
            }
            let completion = start.checked_add(durations[operation])?;
            let candidate = (completion, start, operation);
            if selected.is_none_or(|old| candidate < old) {
                selected = Some(candidate);
            }
        }
        let (completion, start, operation) = selected?;
        starts[operation] = start;
        scheduled[operation] = true;
        for &resource in &memberships[operation] {
            if schedule_stopped(stop) {
                return None;
            }
            available[resource] = completion;
        }
        for &after in &outgoing[operation] {
            if schedule_stopped(stop) {
                return None;
            }
            earliest[after] = earliest[after].max(completion);
            indegrees[after] -= 1;
        }
    }
    (!schedule_stopped(stop)).then_some(starts)
}

/// Translate a feasible construction into the exact search's pair assignment.
/// The objective is an upper bound only; its variable receives no phase hint.
fn constructed_unary_assignment(
    starts: &[i64],
    durations: &[i64],
    pairs: &[(usize, usize)],
    order_base: usize,
    scope_len: usize,
    stop: &AtomicBool,
) -> Option<(Vec<i32>, i64)> {
    let mut assignment = vec![1; scope_len];
    let mut objective = 0;
    for (&start, &duration) in starts.iter().zip(durations) {
        if schedule_stopped(stop) {
            return None;
        }
        objective = objective.max(start.checked_add(duration)?);
    }
    for (index, &(a, b)) in pairs.iter().enumerate() {
        if schedule_stopped(stop) || a == b {
            return None;
        }
        assignment[order_base + index] = if starts[a].checked_add(durations[a])? <= starts[b] {
            1
        } else if starts[b].checked_add(durations[b])? <= starts[a] {
            0
        } else {
            return None;
        };
    }
    Some((assignment, objective))
}

fn initial_order_phases(
    solver: &Solver,
    assignment: &[i32],
    order_base: usize,
    count: usize,
    stop: &AtomicBool,
) -> Option<Vec<Option<i32>>> {
    let mut phase = vec![None; solver.store.num_vars()];
    for index in 0..count {
        if schedule_stopped(stop) {
            return None;
        }
        phase[solver.store.disjunctive_order_var(index).index()] = Some(assignment[order_base + index]);
    }
    Some(phase)
}

/// Install a certified root bound before the LCG atom table is constructed.
/// A concrete incumbent that contradicts a supplied bound takes precedence.
fn install_makespan_lower_bound(solver: &mut Solver, makespan: VarId, lower: Option<i64>, cache: Option<&CachedSchedule>) -> bool {
    let Some(lower) = lower else { return true };
    if lower <= i64::from(solver.store.min(makespan)) || cache.is_some_and(|candidate| lower > candidate.objective) {
        return true;
    }
    if lower > i64::from(solver.store.max(makespan)) {
        return false;
    }
    solver.store.remove_below(makespan, lower as i32).is_ok()
}

/// Solve a supported interval schedule. Returns `Ok(None)` when the
/// schedule shape is not supported by this exact backend.
pub fn solve<F>(schedule: &Schedule, stop: &AtomicBool, options: Options, on_improve: F) -> Result<Option<Outcome>, String>
where
    F: FnMut(i64),
{
    let Some(compiled) = compile_schedule(schedule, stop)? else {
        return Ok(schedule_stopped(stop).then(|| interrupted_outcome(SolveStats::default())));
    };
    solve_compiled(&compiled, stop, options, &mut Execution::default(), on_improve).map(Some)
}

fn solve_prepared<F>(
    compiled: &CompiledSchedule,
    stop: &AtomicBool,
    options: Options,
    objective_lb: Option<i64>,
    execution: &mut Execution,
    on_improve: F,
) -> Result<Outcome, String>
where
    F: FnMut(i64, Option<ScheduleIncumbent<'_>>),
{
    *execution = Execution {
        encoding: Some(match &compiled.kind {
            CompiledScheduleKind::Fixed(_) => "fixed",
            CompiledScheduleKind::Moded(_) => "moded",
        }),
        search_kernel: None,
    };
    if schedule_stopped(stop) {
        return Ok(interrupted_outcome(SolveStats::default()));
    }
    match &compiled.kind {
        CompiledScheduleKind::Fixed(schedule) => {
            let objective_lb = objective_lb.filter(|_| schedule.minimize_makespan);
            solve_fixed_intervals(schedule, stop, options, objective_lb, execution, on_improve)
        }
        CompiledScheduleKind::Moded(schedule) => {
            let mut nonnegative = true;
            if objective_lb.is_some() && schedule.minimize_makespan {
                for modes in &schedule.operations {
                    for mode in modes {
                        if schedule_stopped(stop) {
                            return Ok(interrupted_outcome(SolveStats::default()));
                        }
                        nonnegative &= mode.start_min >= 0;
                    }
                }
            }
            let objective_lb = objective_lb.filter(|_| schedule.minimize_makespan && nonnegative);
            solve_optional_modes(schedule, stop, options, objective_lb, execution, on_improve)
        }
    }
}

fn solve_fixed_intervals<F>(
    schedule: &CompiledFixedSchedule,
    stop: &AtomicBool,
    options: Options,
    objective_lb: Option<i64>,
    execution: &mut Execution,
    on_improve: F,
) -> Result<Outcome, String>
where
    F: FnMut(i64, Option<ScheduleIncumbent<'_>>),
{
    let mut reconstruction_error = None;
    let outcome = solve_fixed_intervals_inner(schedule, stop, options, objective_lb, execution, &mut reconstruction_error, on_improve);
    reconstruction_error.map_or(Ok(outcome), Err)
}

fn solve_fixed_intervals_inner<F>(
    schedule: &CompiledFixedSchedule,
    stop: &AtomicBool,
    options: Options,
    objective_lb: Option<i64>,
    execution: &mut Execution,
    reconstruction_error: &mut Option<String>,
    mut on_improve: F,
) -> Outcome
where
    F: FnMut(i64, Option<ScheduleIncumbent<'_>>),
{
    let should_stop = || schedule_stopped(stop);
    let mut solver = Solver::new();
    let mut intervals: Vec<IntervalId> = Vec::with_capacity(schedule.intervals.len());
    let mut durations_i32 = Vec::with_capacity(schedule.intervals.len());
    for interval in &schedule.intervals {
        if should_stop() {
            return interrupted_outcome(SolveStats::default());
        }
        intervals.push(if interval.optional {
            solver.store.new_optional_interval(0, interval.start_max, interval.duration)
        } else {
            solver.store.new_interval(0, interval.start_max, interval.duration)
        });
        durations_i32.push(interval.duration);
    }
    for &(before, after) in &schedule.precedences {
        if should_stop() {
            return interrupted_outcome(SolveStats::default());
        }
        interval_constraints::interval_precedence(&mut solver, intervals[before], intervals[after]);
    }
    for resource in &schedule.resources {
        if should_stop() {
            return interrupted_outcome(SolveStats::default());
        }
        match resource {
            CompiledFixedResource::NoOverlap(group) => {
                let mut group_ids = Vec::with_capacity(group.len());
                for &index in group {
                    if should_stop() {
                        return interrupted_outcome(SolveStats::default());
                    }
                    group_ids.push(intervals[index]);
                }
                if interval_constraints::no_overlap_until(&mut solver, group_ids, &should_stop).is_none() {
                    return interrupted_outcome(SolveStats::default());
                }
            }
            CompiledFixedResource::Cumulative { demands, capacity } => {
                let mut group_ids = Vec::with_capacity(demands.len());
                let mut group_demands = Vec::with_capacity(demands.len());
                for &(index, demand) in demands {
                    if should_stop() {
                        return interrupted_outcome(SolveStats::default());
                    }
                    group_ids.push(intervals[index]);
                    group_demands.push(demand);
                }
                if interval_constraints::cumulative_until(&mut solver, group_ids, group_demands, *capacity, &should_stop).is_none() {
                    return interrupted_outcome(SolveStats::default());
                }
            }
        }
    }

    if schedule.minimize_makespan {
        if should_stop() {
            return interrupted_outcome(SolveStats::default());
        }
        let makespan = solver.store.new_var_range(0, schedule.max_horizon);
        for (&iv, &dur) in intervals.iter().zip(&durations_i32) {
            if should_stop() {
                return interrupted_outcome(SolveStats::default());
            }
            let start = solver.store.interval_start_var(iv);
            let bound = expr::ge(expr::var(makespan), expr::add(vec![expr::var(start), expr::int(i64::from(dur))]));
            let bound = solver
                .store
                .interval_presence_var(iv)
                .map_or(bound.clone(), |presence| expr::imp(expr::eq(expr::var(presence), expr::int(1)), bound));
            intension::intension(&mut solver, bound);
        }
        // Positive unary schedules are completely described by presence and
        // machine orders. Explicit start decisions would enumerate avoidable
        // waiting time. Other shapes retain the general date search below.
        let mut orders_only = true;
        for interval in &schedule.intervals {
            if should_stop() {
                return interrupted_outcome(SolveStats::default());
            }
            orders_only &= interval.duration > 0;
        }
        for resource in &schedule.resources {
            if should_stop() {
                return interrupted_outcome(SolveStats::default());
            }
            orders_only &= matches!(resource, CompiledFixedResource::NoOverlap(_));
        }
        let mut search_vars = Vec::with_capacity(intervals.len());
        for &iv in &intervals {
            if should_stop() {
                return interrupted_outcome(SolveStats::default());
            }
            if !orders_only {
                search_vars.push(solver.store.interval_start_var(iv));
            }
        }
        let n_starts = search_vars.len();
        let mut presence_positions = Vec::with_capacity(intervals.len());
        for &iv in &intervals {
            if should_stop() {
                return interrupted_outcome(SolveStats::default());
            }
            let position = solver.store.interval_presence_var(iv).map(|presence| {
                let position = search_vars.len();
                search_vars.push(presence);
                position
            });
            presence_positions.push(position);
        }
        let order_base = search_vars.len();
        let mut order_pairs = Vec::new();
        for index in 0..solver.store.disjunctive_pair_count() {
            if index.is_multiple_of(256) && should_stop() {
                return interrupted_outcome(SolveStats::default());
            }
            search_vars.push(solver.store.disjunctive_order_var(index));
            if orders_only {
                let (before, after) = solver.store.disjunctive_pair(index);
                order_pairs.push((before.index(), after.index()));
            }
        }
        search_vars.push(makespan);

        let mut cache = None::<CachedSchedule>;
        let mut initial_phase = Vec::new();
        if orders_only {
            match construct_fixed_incumbent(schedule, &presence_positions, &order_pairs, order_base, search_vars.len(), stop) {
                Ok(Some((candidate, assignment))) => {
                    let objective = candidate.objective;
                    cache = Some(candidate);
                    on_improve(objective, cache.as_ref().map(CachedSchedule::snapshot));
                    if should_stop() {
                        return finish_cached_schedule(cache, None, false, SolveStats::default());
                    }
                    let Some(phase) = initial_order_phases(&solver, &assignment, order_base, order_pairs.len(), stop) else {
                        return finish_cached_schedule(cache, None, false, SolveStats::default());
                    };
                    initial_phase = phase;
                    if solver.store.remove_above(makespan, objective as i32).is_err() {
                        reconstruction_error
                            .get_or_insert("internal error: constructed fixed schedule violates its objective bound".into());
                        return finish_cached_schedule(cache, None, false, SolveStats::default());
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    reconstruction_error.get_or_insert(error);
                    return interrupted_outcome(SolveStats::default());
                }
            }
        }
        if should_stop() {
            return finish_cached_schedule(cache, None, false, SolveStats::default());
        }
        if !install_makespan_lower_bound(&mut solver, makespan, objective_lb, cache.as_ref()) {
            return finish_cached_schedule(cache, None, !should_stop(), SolveStats::default());
        }
        execution.search_kernel = Some("cdcl");
        let primary_scope = orders_only.then_some(&search_vars[..search_vars.len() - 1]);
        let (best, stats, complete) = search::optimize_seeded_with_scope(
            &mut solver,
            &search_vars,
            primary_scope,
            SearchObjective::Var(makespan),
            true,
            stop,
            options.seed,
            None,
            None,
            &[],
            None,
            initial_phase,
            Vec::new(),
            |value, assignment| {
                if !orders_only {
                    on_improve(value, None);
                    return;
                }
                match reconstruct_fixed_incumbent(schedule, &presence_positions, &order_pairs, order_base, assignment, value, stop) {
                    Ok(Some(candidate)) if cache.as_ref().is_none_or(|old| candidate.objective < old.objective) => {
                        let objective = candidate.objective;
                        cache = Some(candidate);
                        on_improve(objective, cache.as_ref().map(CachedSchedule::snapshot));
                    }
                    Ok(_) => {}
                    Err(error) => {
                        reconstruction_error.get_or_insert(error);
                    }
                }
            },
        );
        let complete = complete && !should_stop();
        if orders_only {
            // The outer builder propagates reconstruction errors; they cannot
            // become proofs or partially published assignments.
            if reconstruction_error.is_some() {
                return finish_cached_schedule(cache, best.map(|(_, value)| value), false, stats);
            }
            return finish_cached_schedule(cache, best.map(|(_, value)| value), complete, stats);
        }
        return match (best, complete) {
            (Some((assignment, value)), complete) => Outcome {
                status: if complete { Status::Optimal } else { Status::Satisfiable },
                objective: Some(value),
                starts: assignment[..n_starts].iter().map(|&s| i64::from(s)).collect(),
                presences: presence_positions.iter().map(|position| position.is_none_or(|position| assignment[position] != 0)).collect(),
                machines: vec![-1; schedule.intervals.len()],
                modes: vec![None; schedule.intervals.len()],
                stats,
            },
            (None, true) => Outcome {
                status: Status::Unsatisfiable,
                objective: None,
                starts: Vec::new(),
                presences: Vec::new(),
                machines: Vec::new(),
                modes: Vec::new(),
                stats,
            },
            (None, false) => Outcome {
                status: Status::Unknown,
                objective: None,
                starts: Vec::new(),
                presences: Vec::new(),
                machines: Vec::new(),
                modes: Vec::new(),
                stats,
            },
        };
    }

    let mut assignment: Option<(Vec<i64>, Vec<bool>)> = None;
    execution.search_kernel = Some("chronological-dfs");
    let (stats, complete) = search::solve_domains_interruptible(
        &mut solver,
        |_, domain| {
            assignment = Some((
                domain.interval_starts.iter().map(|start| i64::from(start.unwrap_or(0))).collect(),
                domain.interval_starts.iter().map(Option::is_some).collect(),
            ));
            SearchControl::Stop
        },
        stop,
    );
    let complete = complete && !should_stop();
    match (assignment, complete) {
        (Some((starts, presences)), _) => Outcome {
            status: Status::Satisfiable,
            objective: None,
            starts,
            presences,
            machines: vec![-1; schedule.intervals.len()],
            modes: vec![None; schedule.intervals.len()],
            stats,
        },
        (None, true) => Outcome {
            status: Status::Unsatisfiable,
            objective: None,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
            modes: Vec::new(),
            stats,
        },
        (None, false) => Outcome {
            status: Status::Unknown,
            objective: None,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
            modes: Vec::new(),
            stats,
        },
    }
}

fn reconstruct_fixed_incumbent(
    schedule: &CompiledFixedSchedule,
    presence_positions: &[Option<usize>],
    order_pairs: &[(usize, usize)],
    order_base: usize,
    assignment: &[i32],
    value: i64,
    stop: &AtomicBool,
) -> Result<Option<CachedSchedule>, String> {
    let count = schedule.intervals.len();
    let mut durations = Vec::with_capacity(count);
    let mut latest = Vec::with_capacity(count);
    let mut presences = Vec::with_capacity(count);
    for (interval, position) in schedule.intervals.iter().zip(presence_positions) {
        if schedule_stopped(stop) {
            return Ok(None);
        }
        durations.push(i64::from(interval.duration));
        latest.push(i64::from(interval.start_max));
        presences.push(position.is_none_or(|position| assignment[position] == 1));
    }
    let releases = vec![0; count];
    let pairs = order_pairs.iter().enumerate().take_while(|_| !schedule_stopped(stop)).map(|(index, &(a, b))| {
        if assignment[order_base + index] == 1 {
            (a, b)
        } else {
            (b, a)
        }
    });
    let Some(starts) = earliest_unary_dates(&durations, &releases, &latest, &presences, &schedule.precedences, pairs, stop)? else {
        return Ok(None);
    };
    let mut objective = 0;
    for (operation, &start) in starts.iter().enumerate() {
        if schedule_stopped(stop) {
            return Ok(None);
        }
        if presences[operation] {
            objective = objective.max(start.checked_add(durations[operation]).ok_or("schedule completion overflows i64")?);
        }
    }
    if objective > value {
        return Err("internal error: reconstructed fixed schedule exceeds its objective".to_string());
    }
    Ok(Some(CachedSchedule { objective, starts, presences, machines: vec![-1; count], modes: vec![None; count] }))
}

fn construct_fixed_incumbent(
    schedule: &CompiledFixedSchedule,
    presence_positions: &[Option<usize>],
    pairs: &[(usize, usize)],
    order_base: usize,
    scope_len: usize,
    stop: &AtomicBool,
) -> Result<Option<(CachedSchedule, Vec<i32>)>, String> {
    let mut durations = Vec::with_capacity(schedule.intervals.len());
    let mut latest = Vec::with_capacity(schedule.intervals.len());
    for interval in &schedule.intervals {
        if schedule_stopped(stop) || interval.optional || interval.duration <= 0 {
            return Ok(None);
        }
        durations.push(i64::from(interval.duration));
        latest.push(i64::from(interval.start_max));
    }
    let mut groups = Vec::with_capacity(schedule.resources.len());
    for resource in &schedule.resources {
        if schedule_stopped(stop) {
            return Ok(None);
        }
        let CompiledFixedResource::NoOverlap(group) = resource else { return Ok(None) };
        let mut members = Vec::with_capacity(group.len());
        for &operation in group {
            if schedule_stopped(stop) {
                return Ok(None);
            }
            members.push(operation);
        }
        groups.push(members);
    }
    let releases = vec![0; durations.len()];
    let Some(starts) = construct_unary_dates(&durations, &releases, &latest, &schedule.precedences, &groups, stop) else {
        return Ok(None);
    };
    let Some((assignment, value)) = constructed_unary_assignment(&starts, &durations, pairs, order_base, scope_len, stop) else {
        return Ok(None);
    };
    Ok(reconstruct_fixed_incumbent(schedule, presence_positions, pairs, order_base, &assignment, value, stop)?
        .map(|candidate| (candidate, assignment)))
}

fn solve_optional_modes<F>(
    schedule: &CompiledModedSchedule,
    stop: &AtomicBool,
    options: Options,
    objective_lb: Option<i64>,
    execution: &mut Execution,
    mut on_improve: F,
) -> Result<Outcome, String>
where
    F: FnMut(i64, Option<ScheduleIncumbent<'_>>),
{
    let should_stop = || schedule_stopped(stop);
    let mut solver = Solver::new();
    let Some(ModedBuild { op_modes, all_modes, all_durations }) = build_moded_schedule(schedule, &mut solver, stop) else {
        return Ok(interrupted_outcome(SolveStats::default()));
    };
    let op_count = op_modes.len();

    if schedule.minimize_makespan && options.optional_modes_cdcl {
        return solve_optional_modes_cdcl(
            schedule,
            solver,
            op_modes,
            all_modes,
            all_durations,
            stop,
            options.seed,
            objective_lb,
            execution,
            on_improve,
        );
    }

    let minimize = schedule.minimize_makespan;
    let makespan_ub = Arc::new(AtomicI32::new(i32::MAX));
    if minimize
        && interval_constraints::makespan_bound_until(&mut solver, all_modes, all_durations, Arc::clone(&makespan_ub), &should_stop)
            .is_none()
    {
        return Ok(interrupted_outcome(SolveStats::default()));
    }
    let bound_on_improve = Arc::clone(&makespan_ub);
    let mut best: Option<ModeScheduleIncumbent> = None;
    execution.search_kernel = Some("chronological-dfs");
    let (stats, complete) = search::solve_domains_interruptible(
        &mut solver,
        |_, domain| {
            let mut starts = vec![0i64; op_count];
            let mut machines = vec![-1i64; op_count];
            let mut chosen_modes = vec![None; op_count];
            let mut makespan = 0i64;
            for (op, modes) in op_modes.iter().enumerate() {
                if should_stop() {
                    return SearchControl::Stop;
                }
                for mode in modes {
                    if should_stop() {
                        return SearchControl::Stop;
                    }
                    if let Some(start) = domain.interval_starts[mode.interval.index()] {
                        starts[op] = i64::from(start);
                        machines[op] = mode.machine_value;
                        chosen_modes[op] = mode.reference;
                        makespan = makespan.max(i64::from(start) + i64::from(mode.duration));
                    }
                }
            }
            if best.as_ref().is_none_or(|(value, _, _, _)| makespan < *value) {
                if minimize {
                    on_improve(makespan, None);
                    bound_on_improve.store(i32::try_from(makespan.saturating_sub(1)).unwrap_or(i32::MAX), Ordering::Relaxed);
                }
                best = Some((makespan, starts, machines, chosen_modes));
            }
            if minimize {
                SearchControl::Continue
            } else {
                SearchControl::Stop
            }
        },
        stop,
    );
    let complete = complete && !should_stop();
    let outcome = match (best, complete) {
        (Some((objective, starts, machines, modes)), _) if minimize => Outcome {
            status: if complete { Status::Optimal } else { Status::Satisfiable },
            objective: Some(objective),
            starts,
            presences: vec![true; schedule.operations.len()],
            machines,
            modes,
            stats,
        },
        (Some((_, starts, machines, modes)), _) => Outcome {
            status: Status::Satisfiable,
            objective: None,
            starts,
            presences: vec![true; schedule.operations.len()],
            machines,
            modes,
            stats,
        },
        (None, true) => Outcome {
            status: Status::Unsatisfiable,
            objective: None,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
            modes: Vec::new(),
            stats,
        },
        (None, false) => Outcome {
            status: Status::Unknown,
            objective: None,
            starts: Vec::new(),
            presences: Vec::new(),
            machines: Vec::new(),
            modes: Vec::new(),
            stats,
        },
    };
    Ok(outcome)
}

#[allow(clippy::too_many_arguments)]
fn solve_optional_modes_cdcl<F>(
    schedule: &CompiledModedSchedule,
    mut solver: Solver,
    op_modes: Vec<Vec<BuiltMode>>,
    all_modes: Vec<IntervalId>,
    all_durations: Vec<i32>,
    stop: &AtomicBool,
    seed: u64,
    objective_lb: Option<i64>,
    execution: &mut Execution,
    mut on_improve: F,
) -> Result<Outcome, String>
where
    F: FnMut(i64, Option<ScheduleIncumbent<'_>>),
{
    let should_stop = || schedule_stopped(stop);
    if should_stop() {
        return Ok(interrupted_outcome(SolveStats::default()));
    }
    let op_count = op_modes.len();
    let makespan = solver.store.new_var_range(0, schedule.max_horizon);
    let mut start_vars = Vec::with_capacity(all_modes.len());
    let mut presence_vars = Vec::with_capacity(all_modes.len());
    for (index, &id) in all_modes.iter().enumerate() {
        if index.is_multiple_of(256) && should_stop() {
            return Ok(interrupted_outcome(SolveStats::default()));
        }
        start_vars.push(solver.store.interval_start_var(id));
        presence_vars.push(solver.store.interval_presence_var(id).expect("mode interval is optional"));
    }
    for ((&start, &present), &dur) in start_vars.iter().zip(&presence_vars).zip(&all_durations) {
        if should_stop() {
            return Ok(interrupted_outcome(SolveStats::default()));
        }
        intension::intension(
            &mut solver,
            expr::imp(
                expr::eq(expr::var(present), expr::int(1)),
                expr::ge(expr::var(makespan), expr::add(vec![expr::var(start), expr::int(i64::from(dur))])),
            ),
        );
    }

    let mut op_modes_flat: Vec<Vec<(BuiltMode, usize)>> = Vec::with_capacity(op_count);
    let mut flat = 0usize;
    for modes in &op_modes {
        if should_stop() {
            return Ok(interrupted_outcome(SolveStats::default()));
        }
        let mut row = Vec::with_capacity(modes.len());
        for &mode in modes {
            if should_stop() {
                return Ok(interrupted_outcome(SolveStats::default()));
            }
            row.push((mode, flat));
            flat += 1;
        }
        op_modes_flat.push(row);
    }
    let n_orders = solver.store.disjunctive_pair_count();
    let mut search_vars: Vec<VarId> = presence_vars;
    for index in 0..n_orders {
        if index.is_multiple_of(256) && should_stop() {
            return Ok(interrupted_outcome(SolveStats::default()));
        }
        search_vars.push(solver.store.disjunctive_order_var(index));
    }
    search_vars.push(makespan);

    let mut order_pairs = Vec::with_capacity(n_orders);
    for index in 0..n_orders {
        if should_stop() {
            return Ok(interrupted_outcome(SolveStats::default()));
        }
        let (before, after) = solver.store.disjunctive_pair(index);
        order_pairs.push((before.index(), after.index()));
    }
    let mut cache = None::<CachedSchedule>;
    let mut reconstruction_error = None;
    let mut initial_phase = Vec::new();
    if let Some((candidate, assignment)) =
        construct_singleton_moded_incumbent(schedule, &op_modes_flat, &all_durations, &order_pairs, search_vars.len(), stop)?
    {
        let objective = candidate.objective;
        cache = Some(candidate);
        on_improve(objective, cache.as_ref().map(CachedSchedule::snapshot));
        if should_stop() {
            return Ok(finish_cached_schedule(cache, None, false, SolveStats::default()));
        }
        let Some(phase) = initial_order_phases(&solver, &assignment, all_durations.len(), n_orders, stop) else {
            return Ok(finish_cached_schedule(cache, None, false, SolveStats::default()));
        };
        initial_phase = phase;
        solver
            .store
            .remove_above(makespan, i32::try_from(objective).map_err(|_| "constructed mode objective exceeds i32")?)
            .map_err(|_| "internal error: constructed mode schedule violates its objective bound")?;
    }
    if should_stop() {
        return Ok(finish_cached_schedule(cache, None, false, SolveStats::default()));
    }
    if !install_makespan_lower_bound(&mut solver, makespan, objective_lb, cache.as_ref()) {
        return Ok(finish_cached_schedule(cache, None, !should_stop(), SolveStats::default()));
    }
    execution.search_kernel = Some("cdcl");
    let (best, stats, complete) = search::optimize_seeded_with_scope(
        &mut solver,
        &search_vars,
        Some(&search_vars[..search_vars.len() - 1]),
        SearchObjective::Var(makespan),
        true,
        stop,
        seed,
        None,
        None,
        &[],
        None,
        initial_phase,
        Vec::new(),
        |value, assignment| {
            let reconstructed =
                reconstruct_moded_incumbent(schedule, &op_modes_flat, &all_durations, &order_pairs, assignment, value, stop);
            match reconstructed {
                Ok(Some(candidate)) if cache.as_ref().is_none_or(|old| candidate.objective < old.objective) => {
                    let objective = candidate.objective;
                    cache = Some(candidate);
                    on_improve(objective, cache.as_ref().map(CachedSchedule::snapshot));
                }
                Ok(_) => {}
                Err(error) => {
                    reconstruction_error.get_or_insert(error);
                }
            }
        },
    );
    if let Some(error) = reconstruction_error {
        return Err(error);
    }
    Ok(finish_cached_schedule(cache, best.map(|(_, value)| value), complete && !should_stop(), stats))
}

fn construct_singleton_moded_incumbent(
    schedule: &CompiledModedSchedule,
    op_modes_flat: &[Vec<(BuiltMode, usize)>],
    all_durations: &[i32],
    pairs: &[(usize, usize)],
    scope_len: usize,
    stop: &AtomicBool,
) -> Result<Option<(CachedSchedule, Vec<i32>)>, String> {
    let count = schedule.operations.len();
    let mut durations = Vec::with_capacity(count);
    let mut releases = Vec::with_capacity(count);
    let mut latest = Vec::with_capacity(count);
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (operation, modes) in schedule.operations.iter().enumerate() {
        if schedule_stopped(stop) || modes.len() != 1 || modes[0].duration <= 0 {
            return Ok(None);
        }
        let mode = &modes[0];
        durations.push(i64::from(mode.duration));
        releases.push(i64::from(mode.start_min));
        latest.push(i64::from(mode.start_max));
        groups.entry(mode.machine).or_default().push(operation);
    }
    let groups = groups.into_values().collect::<Vec<_>>();
    let Some(starts) = construct_unary_dates(&durations, &releases, &latest, &schedule.precedences, &groups, stop) else {
        return Ok(None);
    };
    let Some((assignment, value)) = constructed_unary_assignment(&starts, &durations, pairs, count, scope_len, stop) else {
        return Ok(None);
    };
    Ok(reconstruct_moded_incumbent(schedule, op_modes_flat, all_durations, pairs, &assignment, value, stop)?
        .map(|candidate| (candidate, assignment)))
}

fn reconstruct_moded_incumbent(
    schedule: &CompiledModedSchedule,
    op_modes_flat: &[Vec<(BuiltMode, usize)>],
    all_durations: &[i32],
    order_pairs: &[(usize, usize)],
    assignment: &[i32],
    value: i64,
    stop: &AtomicBool,
) -> Result<Option<CachedSchedule>, String> {
    let should_stop = || schedule_stopped(stop);
    let n_modes = all_durations.len();
    let n_orders = order_pairs.len();
    let op_count = op_modes_flat.len();
    if should_stop() {
        return Ok(None);
    }
    let mut all_positive = true;
    for &duration in all_durations {
        if should_stop() {
            return Ok(None);
        }
        all_positive &= duration > 0;
    }
    if all_positive {
        let mut durations = Vec::with_capacity(op_count);
        let mut releases = Vec::with_capacity(op_count);
        let mut latest = Vec::with_capacity(op_count);
        let mut machines = Vec::with_capacity(op_count);
        let mut modes = Vec::with_capacity(op_count);
        let mut operation_of_mode = vec![0; n_modes];
        for (operation, choices) in op_modes_flat.iter().enumerate() {
            if should_stop() {
                return Ok(None);
            }
            let mut chosen = None;
            for (choice, &(mode, index)) in choices.iter().enumerate() {
                if should_stop() {
                    return Ok(None);
                }
                operation_of_mode[index] = operation;
                if assignment[index] == 1 && chosen.replace((mode, choice)).is_some() {
                    return Err("internal error: a reconstructed operation has multiple modes".to_string());
                }
            }
            let (chosen, choice) = chosen.ok_or("internal error: a reconstructed operation has no mode")?;
            let compiled = &schedule.operations[operation][choice];
            durations.push(i64::from(chosen.duration));
            releases.push(i64::from(compiled.start_min));
            latest.push(i64::from(compiled.start_max));
            machines.push(chosen.machine_value);
            modes.push(chosen.reference);
        }
        let presences = vec![true; op_count];
        let pairs = order_pairs.iter().enumerate().take_while(|_| !should_stop()).filter_map(|(index, &(a, b))| {
            if assignment[a] != 1 || assignment[b] != 1 {
                return None;
            }
            let (a, b) = (operation_of_mode[a], operation_of_mode[b]);
            Some(if assignment[n_modes + index] == 1 { (a, b) } else { (b, a) })
        });
        let Some(starts) = earliest_unary_dates(&durations, &releases, &latest, &presences, &schedule.precedences, pairs, stop)? else {
            return Ok(None);
        };
        let mut objective = 0;
        for (&start, &duration) in starts.iter().zip(&durations) {
            if should_stop() {
                return Ok(None);
            }
            objective = objective.max(start.checked_add(duration).ok_or("schedule completion overflows i64")?);
        }
        if objective > value {
            return Err("internal error: reconstructed mode schedule exceeds its objective".to_string());
        }
        return Ok(Some(CachedSchedule { objective, starts, presences, machines, modes }));
    }

    // General modes, including zero-duration precedence cycles, retain the
    // existing CP reconstruction. It now runs before caching and reporting an
    // incumbent, never after the search deadline has already fired.
    let mut replay = Solver::new();
    let Some(replay_build) = build_moded_schedule(schedule, &mut replay, stop) else {
        return Ok(None);
    };
    let mut replay_present = Vec::with_capacity(replay_build.all_modes.len());
    for (index, &id) in replay_build.all_modes.iter().enumerate() {
        if index.is_multiple_of(256) && should_stop() {
            return Ok(None);
        }
        replay_present.push(replay.store.interval_presence_var(id).expect("mode interval is optional"));
    }
    for (i, &present) in replay_present.iter().enumerate() {
        if i.is_multiple_of(256) && should_stop() {
            return Ok(None);
        }
        replay.store.fix(present, assignment[i]).map_err(|_| "internal error: replaying the optimal mode schedule was inconsistent")?;
    }
    for k in 0..n_orders {
        if k.is_multiple_of(256) && should_stop() {
            return Ok(None);
        }
        let order_var = replay.store.disjunctive_order_var(k);
        replay
            .store
            .fix(order_var, assignment[n_modes + k])
            .map_err(|_| "internal error: replaying the optimal mode schedule was inconsistent")?;
    }
    let replay_result = replay.propagate_until(should_stop);
    if should_stop() {
        return Ok(None);
    }
    replay_result.map_err(|_| "internal error: replaying the optimal mode schedule was inconsistent")?;

    let mut starts = vec![0i64; op_count];
    let mut machines = vec![-1i64; op_count];
    let mut chosen_modes = vec![None; op_count];
    let mut makespan_value = 0i64;
    for (op, modes) in op_modes_flat.iter().enumerate() {
        if should_stop() {
            return Ok(None);
        }
        for &(mode, flat_idx) in modes {
            if should_stop() {
                return Ok(None);
            }
            if assignment[flat_idx] == 1 {
                let start = i64::from(replay.store.interval_start_min(replay_build.all_modes[flat_idx]));
                starts[op] = start;
                machines[op] = mode.machine_value;
                chosen_modes[op] = mode.reference;
                makespan_value = makespan_value.max(start + i64::from(all_durations[flat_idx]));
            }
        }
    }
    if makespan_value > value {
        return Err("internal error: replayed mode schedule exceeds its objective".to_string());
    }
    if should_stop() {
        return Ok(None);
    }
    Ok(Some(CachedSchedule {
        objective: makespan_value,
        starts,
        presences: vec![true; schedule.operations.len()],
        machines,
        modes: chosen_modes,
    }))
}

fn compile_schedule(schedule: &Schedule, stop: &AtomicBool) -> Result<Option<CompiledSchedule>, String> {
    if schedule_stopped(stop) {
        return Ok(None);
    }
    let fixed = schedule.intervals.iter().all(|interval| interval.modes.is_empty())
        && schedule.resources.iter().all(|resource| matches!(resource, Resource::NoOverlap(_) | Resource::Cumulative { .. }));
    if fixed {
        return Ok(compile_fixed_schedule(schedule, stop)?.map(|schedule| CompiledSchedule { kind: CompiledScheduleKind::Fixed(schedule) }));
    }

    let moded = schedule.intervals.iter().all(|interval| !interval.modes.is_empty())
        && schedule.resources.iter().all(|resource| matches!(resource, Resource::MachineNoOverlap));
    if moded {
        return Ok(compile_moded_schedule(schedule, stop)?.map(|schedule| CompiledSchedule { kind: CompiledScheduleKind::Moded(schedule) }));
    }

    Ok(None)
}

fn compile_fixed_schedule(schedule: &Schedule, stop: &AtomicBool) -> Result<Option<CompiledFixedSchedule>, String> {
    let interval_count = schedule.intervals.len();
    if !validate_precedences(&schedule.precedences, interval_count, stop)? {
        return Ok(None);
    }

    let mut max_horizon = 0;
    let mut intervals = Vec::with_capacity(interval_count);
    for interval in &schedule.intervals {
        if schedule_stopped(stop) {
            return Ok(None);
        }
        let duration = checked_nonnegative_i32(interval.duration, "interval duration")?;
        let horizon = checked_i32(interval.horizon, "interval horizon")?;
        let start_max = checked_start_max(interval.horizon, interval.duration, "interval")?;
        max_horizon = max_horizon.max(horizon);
        intervals.push(CompiledFixedInterval { duration, start_max, optional: interval.optional });
    }

    let mut resources = Vec::with_capacity(schedule.resources.len());
    for resource in &schedule.resources {
        if schedule_stopped(stop) {
            return Ok(None);
        }
        match resource {
            Resource::NoOverlap(group) => {
                if group.iter().any(|&index| index >= interval_count) {
                    return Err("no-overlap references a missing interval".to_string());
                }
                resources.push(CompiledFixedResource::NoOverlap(group.clone()));
            }
            Resource::Cumulative { demands, capacity } => {
                let capacity = checked_nonnegative_i32(*capacity, "cumulative capacity")?;
                let mut compiled_demands = Vec::with_capacity(demands.len());
                for &(index, demand) in demands {
                    if schedule_stopped(stop) {
                        return Ok(None);
                    }
                    if index >= interval_count {
                        return Err("cumulative references a missing interval".to_string());
                    }
                    compiled_demands.push((index, checked_nonnegative_i32(demand, "cumulative demand")?));
                }
                resources.push(CompiledFixedResource::Cumulative { demands: compiled_demands, capacity });
            }
            Resource::MachineNoOverlap => unreachable!("fixed schedule capability was checked before numeric compilation"),
        }
    }

    Ok(Some(CompiledFixedSchedule {
        intervals,
        precedences: schedule.precedences.clone(),
        resources,
        minimize_makespan: schedule.minimize_makespan,
        max_horizon,
    }))
}

fn compile_moded_schedule(schedule: &Schedule, stop: &AtomicBool) -> Result<Option<CompiledModedSchedule>, String> {
    let operation_count = schedule.intervals.len();
    if !validate_precedences(&schedule.precedences, operation_count, stop)? {
        return Ok(None);
    }

    let mut max_horizon = i32::MIN;
    let mut operations = Vec::with_capacity(operation_count);
    for interval in &schedule.intervals {
        if schedule_stopped(stop) {
            return Ok(None);
        }
        let mut modes = Vec::with_capacity(interval.modes.len());
        for mode in &interval.modes {
            if schedule_stopped(stop) {
                return Ok(None);
            }
            let duration = checked_nonnegative_i32(mode.duration, "mode duration")?;
            let (start_min, start_max) = checked_mode_window(mode.start_window, interval.horizon, mode.duration)?;
            let machine_value = i64::try_from(mode.machine).map_err(|_| "mode machine is outside the i64 range")?;
            modes.push(CompiledMode { reference: mode.reference, machine: mode.machine, machine_value, duration, start_min, start_max });
        }
        if modes.is_empty() {
            return Err("mode schedule operation has no eligible mode".to_string());
        }
        max_horizon = max_horizon.max(checked_i32(interval.horizon, "interval horizon")?);
        operations.push(modes);
    }

    let max_horizon = if operation_count == 0 { 0 } else { max_horizon };
    if schedule.minimize_makespan && max_horizon < 0 {
        return Err("schedule horizon must be non-negative for makespan minimization".to_string());
    }
    Ok(Some(CompiledModedSchedule {
        operations,
        precedences: schedule.precedences.clone(),
        minimize_makespan: schedule.minimize_makespan,
        max_horizon,
    }))
}

fn validate_precedences(precedences: &[(usize, usize)], interval_count: usize, stop: &AtomicBool) -> Result<bool, String> {
    for &(before, after) in precedences {
        if schedule_stopped(stop) {
            return Ok(false);
        }
        if before >= interval_count || after >= interval_count {
            return Err("schedule precedence references a missing interval".to_string());
        }
    }
    Ok(true)
}

fn schedule_stopped(stop: &AtomicBool) -> bool {
    stop.load(std::sync::atomic::Ordering::Acquire)
}

fn build_moded_schedule(schedule: &CompiledModedSchedule, solver: &mut Solver, stop: &AtomicBool) -> Option<ModedBuild> {
    let should_stop = || schedule_stopped(stop);
    let mut op_modes: Vec<Vec<BuiltMode>> = Vec::with_capacity(schedule.operations.len());
    let mut all_modes: Vec<IntervalId> = Vec::new();
    let mut all_durations: Vec<i32> = Vec::new();
    let mut machine_groups: BTreeMap<usize, Vec<IntervalId>> = BTreeMap::new();
    for operation in &schedule.operations {
        if should_stop() {
            return None;
        }
        let mut modes = Vec::with_capacity(operation.len());
        let mut ids = Vec::with_capacity(operation.len());
        for mode in operation {
            if should_stop() {
                return None;
            }
            let id = solver.store.new_optional_interval(mode.start_min, mode.start_max, mode.duration);
            modes.push(BuiltMode {
                reference: mode.reference,
                machine: mode.machine,
                machine_value: mode.machine_value,
                interval: id,
                duration: mode.duration,
            });
            ids.push(id);
            all_modes.push(id);
            all_durations.push(mode.duration);
            machine_groups.entry(mode.machine).or_default().push(id);
        }
        interval_constraints::exactly_one_mode_until(solver, ids, &should_stop)?;
        op_modes.push(modes);
    }
    for (_, group) in machine_groups {
        interval_constraints::no_overlap_until(solver, group, &should_stop)?;
    }
    for &(before, after) in &schedule.precedences {
        if should_stop() {
            return None;
        }
        for before_mode in &op_modes[before] {
            for after_mode in &op_modes[after] {
                if should_stop() {
                    return None;
                }
                interval_constraints::interval_precedence(solver, before_mode.interval, after_mode.interval);
            }
        }
    }
    (!should_stop()).then_some(ModedBuild { op_modes, all_modes, all_durations })
}

fn checked_i32(value: i64, name: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{name} is outside the i32 domain range"))
}

fn checked_nonnegative_i32(value: i64, name: &str) -> Result<i32, String> {
    if value < 0 {
        return Err(format!("{name} must be non-negative"));
    }
    checked_i32(value, name)
}

fn checked_start_max(horizon: i64, duration: i64, name: &str) -> Result<i32, String> {
    if duration < 0 {
        return Err(format!("{name} duration must be non-negative"));
    }
    let start_max = horizon.checked_sub(duration).ok_or_else(|| format!("{name} horizon minus duration overflows"))?;
    if start_max < 0 {
        return Err(format!("{name} duration exceeds its horizon"));
    }
    checked_i32(start_max, "interval start upper bound")
}

fn checked_mode_window(window: (i64, i64), horizon: i64, duration: i64) -> Result<(i32, i32), String> {
    let (start_min, start_max) = window;
    if duration < 0 {
        return Err("mode interval duration must be non-negative".to_string());
    }
    if start_min > start_max {
        return Err("mode interval start window is empty".to_string());
    }
    let end_max = start_max.checked_add(duration).ok_or_else(|| "mode interval end overflows i64".to_string())?;
    if end_max > horizon {
        return Err("mode interval start window extends beyond its horizon".to_string());
    }
    Ok((checked_i32(start_min, "mode start lower bound")?, checked_i32(start_max, "mode start upper bound")?))
}
