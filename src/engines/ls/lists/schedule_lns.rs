//! Bounded exact repair for strict job-shop local search.
//!
//! The repair is deliberately independent from the persistent search session.
//! It selects a deterministic, bounded critical neighborhood, freezes every
//! exterior machine order, and gives each selected machine segment to the
//! exact scheduling engine as a separate pseudo-machine. A repaired order is
//! returned only after a complete disjunctive reconstruction, a strict
//! makespan improvement, and a second full-oracle comparison.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::engines::schedule::{self, Options as ExactScheduleOptions};
use crate::mix64;
use crate::model::list::{IntervalVar, Mode, Resource, Schedule};

use super::schedule_state::{JobShopState, ScheduleStateInterrupted};

const NO_OPERATION: usize = usize::MAX;
const INTERRUPT_CHUNK: usize = 4_096;
const WATCHDOG_POLL: Duration = Duration::from_millis(1);
const WATCHDOG_RUNNING: u8 = 0;
const WATCHDOG_TOTAL_TIMEOUT: u8 = 1;
const WATCHDOG_PARENT_STOP: u8 = 2;

/// Whether a verified repair may be returned to the caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ScheduleLnsMode {
    /// Execute and audit the complete repair path, but suppress injection.
    #[default]
    Shadow,
    /// Return a strictly improving, fully reconstructed repair.
    Apply,
}

/// Deterministic phase hook used only by cancellation and timeout tests.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScheduleLnsTestPhase {
    Preparation,
    Selection,
    Exact,
    Splice,
    Reconstruction,
    Oracle,
    Publication,
}

/// One bounded repair configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScheduleLnsConfig {
    /// Desired number of operations after deterministic neighborhood growth.
    pub(crate) target_operations: usize,
    /// Hard selection cap, including the initial critical block.
    pub(crate) max_operations: usize,
    /// Complete wall-clock allowance, including reconstruction and audit.
    pub(crate) local_budget: Duration,
    /// Tail reserved for splice, reconstruction, and the full oracle.
    pub(crate) verification_reserve: Duration,
    /// Stable stream selector. Callers can mix in worker and attempt ids.
    pub(crate) seed: u64,
    pub(crate) mode: ScheduleLnsMode,
    #[cfg(test)]
    pub(crate) test_delay_phase: Option<ScheduleLnsTestPhase>,
    #[cfg(test)]
    pub(crate) test_phase_delay: Duration,
}

impl Default for ScheduleLnsConfig {
    fn default() -> Self {
        Self {
            target_operations: 128,
            max_operations: 512,
            local_budget: Duration::from_millis(200),
            verification_reserve: Duration::from_millis(40),
            seed: 0,
            mode: ScheduleLnsMode::Shadow,
            #[cfg(test)]
            test_delay_phase: None,
            #[cfg(test)]
            test_phase_delay: Duration::ZERO,
        }
    }
}

impl ScheduleLnsConfig {
    pub(crate) fn shadow(
        target_operations: usize,
        max_operations: usize,
        local_budget: Duration,
        verification_reserve: Duration,
        seed: u64,
    ) -> Self {
        Self {
            target_operations,
            max_operations,
            local_budget,
            verification_reserve,
            seed,
            mode: ScheduleLnsMode::Shadow,
            #[cfg(test)]
            test_delay_phase: None,
            #[cfg(test)]
            test_phase_delay: Duration::ZERO,
        }
    }
}

/// Aggregated accounting for optional repairs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScheduleLnsMetrics {
    pub(crate) attempts: u64,
    pub(crate) selected_operations: u64,
    /// Complete incumbents produced by the exact subproblem.
    pub(crate) feasible: u64,
    /// Exact incumbents which survived full disjunctive reconstruction.
    pub(crate) reconstructed: u64,
    pub(crate) improvements: u64,
    pub(crate) shadow_improvements: u64,
    /// Sum of verified makespan gains observed in shadow mode.
    pub(crate) shadow_improvement_sum: u64,
    /// Largest verified makespan gain observed in shadow mode.
    pub(crate) shadow_best_improvement: u64,
    pub(crate) timeouts: u64,
    pub(crate) interruptions: u64,
    pub(crate) infeasible: u64,
    pub(crate) non_improving: u64,
    pub(crate) reconstruction_rejections: u64,
    pub(crate) oracle_rejections: u64,
    pub(crate) exact_rejections: u64,
    pub(crate) elapsed_micros: u64,
}

impl ScheduleLnsMetrics {
    pub(crate) fn add(&mut self, other: Self) {
        self.attempts = self.attempts.saturating_add(other.attempts);
        self.selected_operations = self.selected_operations.saturating_add(other.selected_operations);
        self.feasible = self.feasible.saturating_add(other.feasible);
        self.reconstructed = self.reconstructed.saturating_add(other.reconstructed);
        self.improvements = self.improvements.saturating_add(other.improvements);
        self.shadow_improvements = self.shadow_improvements.saturating_add(other.shadow_improvements);
        self.shadow_improvement_sum = self.shadow_improvement_sum.saturating_add(other.shadow_improvement_sum);
        self.shadow_best_improvement = self.shadow_best_improvement.max(other.shadow_best_improvement);
        self.timeouts = self.timeouts.saturating_add(other.timeouts);
        self.interruptions = self.interruptions.saturating_add(other.interruptions);
        self.infeasible = self.infeasible.saturating_add(other.infeasible);
        self.non_improving = self.non_improving.saturating_add(other.non_improving);
        self.reconstruction_rejections = self.reconstruction_rejections.saturating_add(other.reconstruction_rejections);
        self.oracle_rejections = self.oracle_rejections.saturating_add(other.oracle_rejections);
        self.exact_rejections = self.exact_rejections.saturating_add(other.exact_rejections);
        self.elapsed_micros = self.elapsed_micros.saturating_add(other.elapsed_micros);
    }
}

/// Result of one repair attempt. Shadow mode reports the observed value but
/// never carries a candidate.
pub(crate) struct ScheduleLnsResult {
    pub(crate) candidate: Option<JobShopState>,
    pub(crate) observed_makespan: Option<i64>,
    pub(crate) selected_operations: usize,
    pub(crate) timed_out: bool,
}

impl ScheduleLnsResult {
    fn empty(timed_out: bool) -> Self {
        Self { candidate: None, observed_makespan: None, selected_operations: 0, timed_out }
    }
}

#[derive(Clone, Debug)]
struct MachineSegment {
    machine: usize,
    first_position: usize,
    len: usize,
}

/// Reusable scratch storage for repeated repairs on the same-sized problem.
///
/// The four operation-indexed vectors are the important part on Large-TA:
/// keeping them here avoids allocating and initializing several million
/// entries on every optional repair attempt.
#[derive(Debug, Default)]
pub(crate) struct ScheduleLnsWorkspace {
    positions: Vec<usize>,
    selected: Vec<bool>,
    local_index: Vec<usize>,
    segment_of: Vec<usize>,
    frontier: BinaryHeap<Reverse<(i64, u64, u64, usize)>>,
    operations: Vec<usize>,
    segments: Vec<MachineSegment>,
    segment_windows: Vec<BoundaryWindow>,
    ordered: Vec<(i64, i64, usize)>,
    candidate_sequences: Vec<Vec<usize>>,
    capacity_growths: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScheduleLnsWorkspaceCapacities {
    pub(crate) positions: usize,
    pub(crate) selected: usize,
    pub(crate) local_index: usize,
    pub(crate) segment_of: usize,
    pub(crate) operations: usize,
    pub(crate) segments: usize,
    pub(crate) segment_windows: usize,
    pub(crate) ordered: usize,
    pub(crate) frontier: usize,
    pub(crate) candidate_sequence_vectors: usize,
    pub(crate) candidate_sequence_operations: usize,
    pub(crate) estimated_bytes: usize,
    pub(crate) capacity_growths: u64,
}

impl ScheduleLnsWorkspace {
    pub(crate) fn capacities(&self) -> ScheduleLnsWorkspaceCapacities {
        let candidate_sequence_operations = self.candidate_sequences.iter().map(Vec::capacity).fold(0usize, usize::saturating_add);
        let estimated_bytes = size_of::<Self>()
            .saturating_add(self.positions.capacity().saturating_mul(size_of::<usize>()))
            .saturating_add(self.selected.capacity().saturating_mul(size_of::<bool>()))
            .saturating_add(self.local_index.capacity().saturating_mul(size_of::<usize>()))
            .saturating_add(self.segment_of.capacity().saturating_mul(size_of::<usize>()))
            .saturating_add(self.operations.capacity().saturating_mul(size_of::<usize>()))
            .saturating_add(self.segments.capacity().saturating_mul(size_of::<MachineSegment>()))
            .saturating_add(self.segment_windows.capacity().saturating_mul(size_of::<BoundaryWindow>()))
            .saturating_add(self.ordered.capacity().saturating_mul(size_of::<(i64, i64, usize)>()))
            .saturating_add(self.frontier.capacity().saturating_mul(size_of::<Reverse<(i64, u64, u64, usize)>>()))
            .saturating_add(self.candidate_sequences.capacity().saturating_mul(size_of::<Vec<usize>>()))
            .saturating_add(candidate_sequence_operations.saturating_mul(size_of::<usize>()));
        ScheduleLnsWorkspaceCapacities {
            positions: self.positions.capacity(),
            selected: self.selected.capacity(),
            local_index: self.local_index.capacity(),
            segment_of: self.segment_of.capacity(),
            operations: self.operations.capacity(),
            segments: self.segments.capacity(),
            segment_windows: self.segment_windows.capacity(),
            ordered: self.ordered.capacity(),
            frontier: self.frontier.capacity(),
            candidate_sequence_vectors: self.candidate_sequences.capacity(),
            candidate_sequence_operations,
            estimated_bytes,
            capacity_growths: self.capacity_growths,
        }
    }

    #[cfg(test)]
    pub(crate) fn selected_segment_count(&self) -> usize {
        self.segments.len()
    }

    #[cfg(test)]
    pub(crate) fn selected_machine_count(&self) -> usize {
        self.segments
            .iter()
            .enumerate()
            .filter(|(index, segment)| *index == 0 || self.segments[*index - 1].machine != segment.machine)
            .count()
    }

    #[cfg(test)]
    pub(crate) fn selected_mask(&self) -> &[bool] {
        &self.selected
    }

    fn prepare_operation_buffers(&mut self, operation_count: usize, control: PhaseControl<'_>) -> Result<(), ScheduleStateInterrupted> {
        self.frontier.clear();
        self.operations.clear();
        self.segments.clear();
        self.segment_windows.clear();
        self.ordered.clear();

        let before = self.positions.capacity();
        let reset = reset_buffer_chunked(&mut self.positions, operation_count, NO_OPERATION, control);
        self.capacity_growths = self.capacity_growths.saturating_add(u64::from(before != self.positions.capacity()));
        reset?;

        let before = self.selected.capacity();
        let reset = reset_buffer_chunked(&mut self.selected, operation_count, false, control);
        self.capacity_growths = self.capacity_growths.saturating_add(u64::from(before != self.selected.capacity()));
        reset?;

        let before = self.local_index.capacity();
        let reset = reset_buffer_chunked(&mut self.local_index, operation_count, NO_OPERATION, control);
        self.capacity_growths = self.capacity_growths.saturating_add(u64::from(before != self.local_index.capacity()));
        reset?;

        let before = self.segment_of.capacity();
        let reset = reset_buffer_chunked(&mut self.segment_of, operation_count, NO_OPERATION, control);
        self.capacity_growths = self.capacity_growths.saturating_add(u64::from(before != self.segment_of.capacity()));
        reset?;
        Ok(())
    }

    fn clone_candidate_sequences(&mut self, source: &[Vec<usize>], control: PhaseControl<'_>) -> Result<(), ScheduleStateInterrupted> {
        control.checkpoint()?;
        let before_outer = self.candidate_sequences.capacity();
        self.candidate_sequences.truncate(source.len());
        while self.candidate_sequences.len() < source.len() {
            control.checkpoint()?;
            self.candidate_sequences.push(Vec::new());
        }
        self.capacity_growths = self.capacity_growths.saturating_add(u64::from(before_outer != self.candidate_sequences.capacity()));

        for (machine, sequence) in source.iter().enumerate() {
            control.checkpoint()?;
            let target = &mut self.candidate_sequences[machine];
            let before = target.capacity();
            target.clear();
            if target.capacity() < sequence.len() {
                target.reserve_exact(sequence.len());
            }
            control.checkpoint()?;
            let copied = (|| {
                for chunk in sequence.chunks(INTERRUPT_CHUNK) {
                    control.checkpoint()?;
                    target.extend_from_slice(chunk);
                }
                Ok(())
            })();
            self.capacity_growths = self.capacity_growths.saturating_add(u64::from(before != target.capacity()));
            copied?;
        }
        Ok(())
    }
}

fn reset_buffer_chunked<T: Copy>(
    buffer: &mut Vec<T>,
    target_len: usize,
    value: T,
    control: PhaseControl<'_>,
) -> Result<(), ScheduleStateInterrupted> {
    control.checkpoint()?;
    if buffer.capacity() < target_len {
        *buffer = Vec::with_capacity(target_len);
    }
    buffer.clear();
    while buffer.len() < target_len {
        control.checkpoint()?;
        let next = buffer.len().saturating_add(INTERRUPT_CHUNK).min(target_len);
        buffer.resize(next, value);
    }
    control.checkpoint()
}

#[derive(Clone, Copy, Debug)]
struct BoundaryWindow {
    release: i64,
    deadline: i64,
}

struct NeighborhoodGrowth<'a> {
    state: &'a JobShopState,
    positions: &'a [usize],
    center: i64,
    seed: u64,
    control: PhaseControl<'a>,
}

#[derive(Clone, Copy)]
struct PhaseControl<'a> {
    stop: &'a AtomicBool,
    parent_stop: &'a AtomicBool,
    started: Instant,
    budget: Duration,
}

impl PhaseControl<'_> {
    fn checkpoint(self) -> Result<(), ScheduleStateInterrupted> {
        if self.parent_stop.load(Ordering::Acquire) {
            self.stop.store(true, Ordering::Release);
            return Err(ScheduleStateInterrupted);
        }
        if self.stop.load(Ordering::Acquire) || self.started.elapsed() >= self.budget {
            self.stop.store(true, Ordering::Release);
            return Err(ScheduleStateInterrupted);
        }
        Ok(())
    }
}

struct RepairControls<'a> {
    search: PhaseControl<'a>,
    verification: PhaseControl<'a>,
    search_done: &'a AtomicBool,
}

impl NeighborhoodGrowth<'_> {
    fn enqueue(
        &self,
        operation: usize,
        selected: &[bool],
        frontier: &mut BinaryHeap<Reverse<(i64, u64, u64, usize)>>,
    ) -> Result<(), ScheduleStateInterrupted> {
        for &neighbor in self.state.job_predecessors(operation).iter().chain(self.state.job_successors(operation)) {
            self.control.checkpoint()?;
            self.push(neighbor, selected, frontier);
        }
        let machine = self.state.machine(operation);
        let position = self.positions[operation];
        let sequence = &self.state.machine_sequences()[machine];
        if position > 0 {
            self.push(sequence[position - 1], selected, frontier);
        }
        if position + 1 < sequence.len() {
            self.push(sequence[position + 1], selected, frontier);
        }
        Ok(())
    }

    fn push(&self, operation: usize, selected: &[bool], frontier: &mut BinaryHeap<Reverse<(i64, u64, u64, usize)>>) {
        if selected[operation] {
            return;
        }
        let slack = self.state.latest_starts()[operation].saturating_sub(self.state.starts()[operation]);
        let midpoint = self.state.starts()[operation].saturating_add(self.state.duration(operation) / 2);
        let distance = midpoint.abs_diff(self.center);
        frontier.push(Reverse((slack, distance, mix64(self.seed ^ operation as u64), operation)));
    }
}

/// Attempt one deterministic bounded repair around a critical block.
///
/// Parent cancellation is returned as [`ScheduleStateInterrupted`]. A local
/// timeout is an ordinary, non-publishing heuristic outcome.
pub(crate) fn repair_critical_window(
    state: &JobShopState,
    config: ScheduleLnsConfig,
    metrics: &mut ScheduleLnsMetrics,
    parent_stop: &AtomicBool,
) -> Result<ScheduleLnsResult, ScheduleStateInterrupted> {
    repair_critical_window_with_workspace(state, config, metrics, &mut ScheduleLnsWorkspace::default(), parent_stop)
}

/// Workspace-backed form for persistent search sessions.
pub(crate) fn repair_critical_window_with_workspace(
    state: &JobShopState,
    config: ScheduleLnsConfig,
    metrics: &mut ScheduleLnsMetrics,
    workspace: &mut ScheduleLnsWorkspace,
    parent_stop: &AtomicBool,
) -> Result<ScheduleLnsResult, ScheduleStateInterrupted> {
    if parent_stop.load(Ordering::Acquire) {
        metrics.interruptions = metrics.interruptions.saturating_add(1);
        return Err(ScheduleStateInterrupted);
    }
    if config.max_operations == 0 || config.target_operations == 0 || config.verification_reserve > config.local_budget {
        metrics.exact_rejections = metrics.exact_rejections.saturating_add(1);
        return Ok(ScheduleLnsResult::empty(false));
    }

    metrics.attempts = metrics.attempts.saturating_add(1);
    let started = Instant::now();
    let search_duration = config.local_budget.saturating_sub(config.verification_reserve);
    let search_stop = Arc::new(AtomicBool::new(search_duration.is_zero()));
    let verification_stop = Arc::new(AtomicBool::new(config.local_budget.is_zero()));
    let search_done = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let reason = Arc::new(AtomicU8::new(if config.local_budget.is_zero() { WATCHDOG_TOTAL_TIMEOUT } else { WATCHDOG_RUNNING }));

    let result = std::thread::scope(|scope| {
        let worker_search_stop = Arc::clone(&search_stop);
        let worker_verification_stop = Arc::clone(&verification_stop);
        let worker_search_done = Arc::clone(&search_done);
        let worker_done = Arc::clone(&done);
        let worker_reason = Arc::clone(&reason);
        let watchdog = scope.spawn(move || loop {
            if worker_done.load(Ordering::Acquire) {
                return;
            }
            if parent_stop.load(Ordering::Acquire) {
                worker_reason.store(WATCHDOG_PARENT_STOP, Ordering::Release);
                worker_search_stop.store(true, Ordering::Release);
                worker_verification_stop.store(true, Ordering::Release);
                return;
            }
            let elapsed = started.elapsed();
            if elapsed >= search_duration && !worker_search_done.load(Ordering::Acquire) {
                worker_search_stop.store(true, Ordering::Release);
            }
            if elapsed >= config.local_budget {
                let _ = worker_reason.compare_exchange(WATCHDOG_RUNNING, WATCHDOG_TOTAL_TIMEOUT, Ordering::AcqRel, Ordering::Acquire);
                worker_verification_stop.store(true, Ordering::Release);
                return;
            }
            let until_total = config.local_budget.saturating_sub(elapsed);
            let until_event = if elapsed < search_duration && !worker_search_done.load(Ordering::Acquire) {
                search_duration.saturating_sub(elapsed).min(until_total)
            } else {
                until_total
            };
            std::thread::park_timeout(WATCHDOG_POLL.min(until_event));
        });

        let search = PhaseControl { stop: &search_stop, parent_stop, started, budget: search_duration };
        let verification = PhaseControl { stop: &verification_stop, parent_stop, started, budget: config.local_budget };
        let inner =
            repair_with_stops(state, config, metrics, workspace, RepairControls { search, verification, search_done: &search_done });
        done.store(true, Ordering::Release);
        watchdog.thread().unpark();
        let _ = watchdog.join();
        inner
    });

    metrics.elapsed_micros = metrics.elapsed_micros.saturating_add(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
    let mut result = result.ok();
    if parent_stop.load(Ordering::Acquire) || reason.load(Ordering::Acquire) == WATCHDOG_PARENT_STOP {
        if let Some(result) = &mut result {
            recycle_result_candidate(workspace, result);
        }
        metrics.interruptions = metrics.interruptions.saturating_add(1);
        return Err(ScheduleStateInterrupted);
    }
    let timed_out = reason.load(Ordering::Acquire) == WATCHDOG_TOTAL_TIMEOUT || started.elapsed() >= config.local_budget;
    if timed_out {
        metrics.timeouts = metrics.timeouts.saturating_add(1);
    }
    match result {
        Some(mut result) => {
            result.timed_out = timed_out;
            if timed_out {
                recycle_result_candidate(workspace, &mut result);
            } else if result.candidate.is_some() {
                metrics.improvements = metrics.improvements.saturating_add(1);
                if config.mode == ScheduleLnsMode::Shadow {
                    metrics.shadow_improvements = metrics.shadow_improvements.saturating_add(1);
                    let gain = result
                        .observed_makespan
                        .and_then(|candidate| state.makespan().checked_sub(candidate))
                        .and_then(|gain| u64::try_from(gain).ok())
                        .unwrap_or(0);
                    metrics.shadow_improvement_sum = metrics.shadow_improvement_sum.saturating_add(gain);
                    metrics.shadow_best_improvement = metrics.shadow_best_improvement.max(gain);
                    recycle_result_candidate(workspace, &mut result);
                }
            }
            Ok(result)
        }
        None => Ok(ScheduleLnsResult::empty(timed_out)),
    }
}

fn recycle_result_candidate(workspace: &mut ScheduleLnsWorkspace, result: &mut ScheduleLnsResult) {
    if let Some(candidate) = result.candidate.take() {
        workspace.candidate_sequences = candidate.into_machine_sequences();
    }
}

fn repair_with_stops(
    state: &JobShopState,
    config: ScheduleLnsConfig,
    metrics: &mut ScheduleLnsMetrics,
    workspace: &mut ScheduleLnsWorkspace,
    controls: RepairControls<'_>,
) -> Result<ScheduleLnsResult, ScheduleStateInterrupted> {
    #[cfg(test)]
    interruptible_test_phase_delay(config, ScheduleLnsTestPhase::Preparation, controls.search)?;
    workspace.prepare_operation_buffers(state.operation_count(), controls.search)?;
    #[cfg(test)]
    interruptible_test_phase_delay(config, ScheduleLnsTestPhase::Selection, controls.search)?;
    if !select_critical_neighborhood(state, config, workspace, controls.search)? {
        return Ok(ScheduleLnsResult::empty(false));
    }
    let selected_count = workspace.operations.len();
    metrics.selected_operations = metrics.selected_operations.saturating_add(u64::try_from(selected_count).unwrap_or(u64::MAX));
    let Some(repair_schedule) = build_repair_schedule(state, workspace, controls.search)? else {
        metrics.infeasible = metrics.infeasible.saturating_add(1);
        return Ok(ScheduleLnsResult { candidate: None, observed_makespan: None, selected_operations: selected_count, timed_out: false });
    };

    #[cfg(test)]
    interruptible_test_phase_delay(config, ScheduleLnsTestPhase::Exact, controls.search)?;
    let exact = match schedule::solve(
        &repair_schedule,
        controls.search.stop,
        ExactScheduleOptions { seed: config.seed, optional_modes_cdcl: false },
        |_| {},
    ) {
        Ok(Some(outcome)) => outcome,
        Ok(None) | Err(_) => {
            metrics.exact_rejections = metrics.exact_rejections.saturating_add(1);
            return Ok(ScheduleLnsResult {
                candidate: None,
                observed_makespan: None,
                selected_operations: selected_count,
                timed_out: false,
            });
        }
    };
    controls.search_done.store(true, Ordering::Release);
    if outcome_is_incomplete(&exact, selected_count) {
        if !controls.search.stop.load(Ordering::Acquire) {
            metrics.infeasible = metrics.infeasible.saturating_add(1);
        }
        return Ok(ScheduleLnsResult { candidate: None, observed_makespan: None, selected_operations: selected_count, timed_out: false });
    }
    metrics.feasible = metrics.feasible.saturating_add(1);
    controls.verification.checkpoint()?;
    if exact.objective.is_some_and(|objective| objective >= state.makespan()) {
        metrics.non_improving = metrics.non_improving.saturating_add(1);
        return Ok(ScheduleLnsResult {
            candidate: None,
            observed_makespan: exact.objective,
            selected_operations: selected_count,
            timed_out: false,
        });
    }
    #[cfg(test)]
    interruptible_test_phase_delay(config, ScheduleLnsTestPhase::Splice, controls.verification)?;
    splice_segments(state, workspace, &exact.starts, controls.verification)?;
    if !outside_orders_are_frozen(state.machine_sequences(), &workspace.candidate_sequences, &workspace.selected, controls.verification)? {
        metrics.reconstruction_rejections = metrics.reconstruction_rejections.saturating_add(1);
        return Ok(ScheduleLnsResult { candidate: None, observed_makespan: None, selected_operations: selected_count, timed_out: false });
    }
    #[cfg(test)]
    interruptible_test_phase_delay(config, ScheduleLnsTestPhase::Reconstruction, controls.verification)?;
    controls.verification.checkpoint()?;
    let candidate_sequences = std::mem::take(&mut workspace.candidate_sequences);
    let Some(mut candidate) = state.rebuilt_from_machine_sequences(candidate_sequences, controls.verification.stop)? else {
        metrics.reconstruction_rejections = metrics.reconstruction_rejections.saturating_add(1);
        return Ok(ScheduleLnsResult { candidate: None, observed_makespan: None, selected_operations: selected_count, timed_out: false });
    };
    if let Err(interrupted) = controls.verification.checkpoint() {
        workspace.candidate_sequences = candidate.into_machine_sequences();
        return Err(interrupted);
    }
    metrics.reconstructed = metrics.reconstructed.saturating_add(1);
    if candidate.makespan() >= state.makespan() {
        metrics.non_improving = metrics.non_improving.saturating_add(1);
        let observed_makespan = candidate.makespan();
        workspace.candidate_sequences = candidate.into_machine_sequences();
        return Ok(ScheduleLnsResult {
            candidate: None,
            observed_makespan: Some(observed_makespan),
            selected_operations: selected_count,
            timed_out: false,
        });
    }
    #[cfg(test)]
    if let Err(interrupted) = interruptible_test_phase_delay(config, ScheduleLnsTestPhase::Oracle, controls.verification) {
        workspace.candidate_sequences = candidate.into_machine_sequences();
        return Err(interrupted);
    }
    if let Err(interrupted) = controls.verification.checkpoint() {
        workspace.candidate_sequences = candidate.into_machine_sequences();
        return Err(interrupted);
    }
    let matches_oracle = match candidate.matches_full_oracle(controls.verification.stop) {
        Ok(matches) => matches,
        Err(interrupted) => {
            workspace.candidate_sequences = candidate.into_machine_sequences();
            return Err(interrupted);
        }
    };
    if !matches_oracle {
        metrics.oracle_rejections = metrics.oracle_rejections.saturating_add(1);
        workspace.candidate_sequences = candidate.into_machine_sequences();
        return Ok(ScheduleLnsResult { candidate: None, observed_makespan: None, selected_operations: selected_count, timed_out: false });
    }
    if let Err(interrupted) = controls.verification.checkpoint() {
        workspace.candidate_sequences = candidate.into_machine_sequences();
        return Err(interrupted);
    }
    #[cfg(test)]
    if let Err(interrupted) = interruptible_test_phase_delay(config, ScheduleLnsTestPhase::Publication, controls.verification) {
        workspace.candidate_sequences = candidate.into_machine_sequences();
        return Err(interrupted);
    }
    if let Err(interrupted) = controls.verification.checkpoint() {
        workspace.candidate_sequences = candidate.into_machine_sequences();
        return Err(interrupted);
    }

    let observed_makespan = candidate.makespan();
    Ok(ScheduleLnsResult {
        candidate: Some(candidate),
        observed_makespan: Some(observed_makespan),
        selected_operations: selected_count,
        timed_out: false,
    })
}

fn outcome_is_incomplete(outcome: &schedule::Outcome, selected_count: usize) -> bool {
    outcome.objective.is_none()
        || outcome.starts.len() != selected_count
        || outcome.presences.len() != selected_count
        || outcome.presences.iter().any(|present| !present)
}

fn select_critical_neighborhood(
    state: &JobShopState,
    config: ScheduleLnsConfig,
    workspace: &mut ScheduleLnsWorkspace,
    control: PhaseControl<'_>,
) -> Result<bool, ScheduleStateInterrupted> {
    control.checkpoint()?;
    let blocks = if state.critical_blocks().is_empty() { state.canonical_critical_blocks() } else { state.critical_blocks() };
    if blocks.is_empty() {
        return Ok(false);
    }
    let mut visited = 0usize;
    for sequence in state.machine_sequences() {
        control.checkpoint()?;
        for (position, &operation) in sequence.iter().enumerate() {
            periodic_checkpoint(control, visited)?;
            visited = visited.saturating_add(1);
            if operation >= workspace.positions.len() || workspace.positions[operation] != NO_OPERATION {
                return Ok(false);
            }
            workspace.positions[operation] = position;
        }
    }
    for chunk in workspace.positions.chunks(INTERRUPT_CHUNK) {
        control.checkpoint()?;
        if chunk.contains(&NO_OPERATION) {
            return Ok(false);
        }
    }

    let target = config.target_operations.min(config.max_operations);
    let block_start = mix64(config.seed ^ 0xa076_1d64_78bd_642f) as usize % blocks.len();
    let seed_blocks = blocks.len().min(target.div_ceil(2)).max(1);
    for rank in 0..seed_blocks {
        control.checkpoint()?;
        if workspace.operations.len() >= target {
            break;
        }
        let block_index = (block_start + rank) % blocks.len();
        let block = blocks[block_index];
        let sequence = &state.machine_sequences()[block.machine()];
        let take = block.len().min(2).min(target - workspace.operations.len());
        let offset = if block.len() == take {
            0
        } else {
            mix64(config.seed ^ (block_index as u64).wrapping_mul(0xe703_7ed1_a0b4_28db)) as usize % (block.len() - take + 1)
        };
        let first = block.first_position() + offset;
        for &operation in &sequence[first..first + take] {
            control.checkpoint()?;
            if !workspace.selected[operation] {
                workspace.selected[operation] = true;
                workspace.operations.push(operation);
            }
        }
    }
    if workspace.operations.is_empty() {
        return Ok(false);
    }

    let center = block_center_time(state, &workspace.operations, control)?;
    let growth = NeighborhoodGrowth { state, positions: &workspace.positions, center, seed: config.seed, control };
    let initial_len = workspace.operations.len();
    for index in 0..initial_len {
        let operation = workspace.operations[index];
        growth.enqueue(operation, &workspace.selected, &mut workspace.frontier)?;
    }
    while workspace.operations.len() < target {
        control.checkpoint()?;
        let Some(Reverse(key)) = workspace.frontier.pop() else {
            break;
        };
        let operation = key.3;
        if workspace.selected[operation] {
            continue;
        }
        workspace.selected[operation] = true;
        workspace.operations.push(operation);
        growth.enqueue(operation, &workspace.selected, &mut workspace.frontier)?;
    }
    workspace.operations.sort_unstable();
    selected_segments(state.machine_sequences(), &workspace.selected, &mut workspace.segments, control)?;
    Ok(true)
}

fn block_center_time(state: &JobShopState, operations: &[usize], control: PhaseControl<'_>) -> Result<i64, ScheduleStateInterrupted> {
    let mut first = None;
    let mut last = None;
    for (index, &operation) in operations.iter().enumerate() {
        periodic_checkpoint(control, index)?;
        first = Some(first.map_or(state.starts()[operation], |current: i64| current.min(state.starts()[operation])));
        last = Some(last.map_or(state.ends()[operation], |current: i64| current.max(state.ends()[operation])));
    }
    control.checkpoint()?;
    let first = first.unwrap_or(0);
    Ok(first.saturating_add(last.unwrap_or(first).saturating_sub(first) / 2))
}

fn selected_segments(
    machine_sequences: &[Vec<usize>],
    selected: &[bool],
    segments: &mut Vec<MachineSegment>,
    control: PhaseControl<'_>,
) -> Result<(), ScheduleStateInterrupted> {
    segments.clear();
    let mut visited = 0usize;
    for (machine, sequence) in machine_sequences.iter().enumerate() {
        control.checkpoint()?;
        let mut position = 0usize;
        while position < sequence.len() {
            periodic_checkpoint(control, visited)?;
            visited = visited.saturating_add(1);
            if !selected[sequence[position]] {
                position += 1;
                continue;
            }
            let first_position = position;
            while position < sequence.len() && selected[sequence[position]] {
                periodic_checkpoint(control, visited)?;
                visited = visited.saturating_add(1);
                position += 1;
            }
            segments.push(MachineSegment { machine, first_position, len: position - first_position });
        }
    }
    control.checkpoint()?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_selected_segment_shapes(
    machine_sequences: &[Vec<usize>],
    operation_count: usize,
    selected_operations: &[usize],
) -> Vec<(usize, usize, usize)> {
    let mut selected = vec![false; operation_count];
    for &operation in selected_operations {
        selected[operation] = true;
    }
    let mut segments = Vec::new();
    let stop = AtomicBool::new(false);
    let control = PhaseControl { stop: &stop, parent_stop: &stop, started: Instant::now(), budget: Duration::MAX };
    selected_segments(machine_sequences, &selected, &mut segments, control).expect("uninterrupted test selection");
    segments.into_iter().map(|segment| (segment.machine, segment.first_position, segment.len)).collect()
}

fn build_repair_schedule(
    state: &JobShopState,
    workspace: &mut ScheduleLnsWorkspace,
    control: PhaseControl<'_>,
) -> Result<Option<Schedule>, ScheduleStateInterrupted> {
    control.checkpoint()?;
    let Some(improving_horizon) = state.makespan().checked_sub(1) else {
        return Ok(None);
    };
    for (index, &operation) in workspace.operations.iter().enumerate() {
        control.checkpoint()?;
        workspace.local_index[operation] = index;
    }
    for (segment_index, segment) in workspace.segments.iter().enumerate() {
        control.checkpoint()?;
        let sequence = &state.machine_sequences()[segment.machine];
        let release = segment.first_position.checked_sub(1).map_or(0, |position| state.ends()[sequence[position]]);
        let after = segment.first_position + segment.len;
        let deadline = sequence.get(after).map_or(improving_horizon, |&operation| state.starts()[operation]).min(improving_horizon);
        workspace.segment_windows.push(BoundaryWindow { release, deadline });
        for &operation in &sequence[segment.first_position..after] {
            control.checkpoint()?;
            workspace.segment_of[operation] = segment_index;
        }
    }

    let mut intervals = Vec::with_capacity(workspace.operations.len());
    for &operation in &workspace.operations {
        control.checkpoint()?;
        let segment_index = workspace.segment_of[operation];
        if segment_index == NO_OPERATION {
            return Ok(None);
        }
        let mut window = workspace.segment_windows[segment_index];
        for &predecessor in state.job_predecessors(operation) {
            control.checkpoint()?;
            if !workspace.selected[predecessor] {
                window.release = window.release.max(state.ends()[predecessor]);
            }
        }
        for &successor in state.job_successors(operation) {
            control.checkpoint()?;
            if !workspace.selected[successor] {
                window.deadline = window.deadline.min(state.starts()[successor]);
            }
        }
        let duration = state.duration(operation);
        let Some(latest_start) = window.deadline.checked_sub(duration) else {
            return Ok(None);
        };
        if duration < 0 || window.release < 0 || window.release > latest_start {
            return Ok(None);
        }
        intervals.push(IntervalVar {
            duration,
            horizon: window.deadline,
            modes: vec![Mode { reference: None, machine: segment_index, duration, start_window: (window.release, latest_start) }],
            optional: false,
        });
    }

    let mut precedences = Vec::new();
    for &operation in &workspace.operations {
        control.checkpoint()?;
        let before = workspace.local_index[operation];
        for &successor in state.job_successors(operation) {
            control.checkpoint()?;
            let after = workspace.local_index[successor];
            if after != NO_OPERATION {
                precedences.push((before, after));
            }
        }
    }
    Ok(Some(Schedule { intervals, precedences, resources: vec![Resource::MachineNoOverlap], minimize_makespan: true }))
}

fn splice_segments(
    state: &JobShopState,
    workspace: &mut ScheduleLnsWorkspace,
    repaired_starts: &[i64],
    control: PhaseControl<'_>,
) -> Result<(), ScheduleStateInterrupted> {
    workspace.clone_candidate_sequences(state.machine_sequences(), control)?;
    for segment in &workspace.segments {
        control.checkpoint()?;
        workspace.ordered.clear();
        let end = segment.first_position + segment.len;
        for &operation in &state.machine_sequences()[segment.machine][segment.first_position..end] {
            control.checkpoint()?;
            let index = workspace.local_index[operation];
            if index == NO_OPERATION || index >= repaired_starts.len() {
                return Err(ScheduleStateInterrupted);
            }
            let end = repaired_starts[index].saturating_add(state.duration(operation));
            workspace.ordered.push((repaired_starts[index], end, operation));
        }
        workspace.ordered.sort_unstable();
        for (offset, &(_, _, operation)) in workspace.ordered.iter().enumerate() {
            workspace.candidate_sequences[segment.machine][segment.first_position + offset] = operation;
        }
    }
    control.checkpoint()
}

fn outside_orders_are_frozen(
    incumbent: &[Vec<usize>],
    candidate: &[Vec<usize>],
    selected: &[bool],
    control: PhaseControl<'_>,
) -> Result<bool, ScheduleStateInterrupted> {
    if incumbent.len() != candidate.len() {
        return Ok(false);
    }
    for (before, after) in incumbent.iter().zip(candidate) {
        control.checkpoint()?;
        let mut before_position = 0usize;
        let mut after_position = 0usize;
        let mut visited = 0usize;
        loop {
            while before_position < before.len() && selected[before[before_position]] {
                periodic_checkpoint(control, visited)?;
                visited = visited.saturating_add(1);
                before_position += 1;
            }
            while after_position < after.len() && selected[after[after_position]] {
                periodic_checkpoint(control, visited)?;
                visited = visited.saturating_add(1);
                after_position += 1;
            }
            match (before.get(before_position), after.get(after_position)) {
                (None, None) => break,
                (Some(&left), Some(&right)) if left == right => {
                    periodic_checkpoint(control, visited)?;
                    visited = visited.saturating_add(1);
                    before_position += 1;
                    after_position += 1;
                }
                _ => return Ok(false),
            }
        }
    }
    control.checkpoint()?;
    Ok(true)
}

fn periodic_checkpoint(control: PhaseControl<'_>, index: usize) -> Result<(), ScheduleStateInterrupted> {
    if index.is_multiple_of(INTERRUPT_CHUNK) {
        control.checkpoint()?;
    }
    Ok(())
}

#[cfg(test)]
fn interruptible_test_phase_delay(
    config: ScheduleLnsConfig,
    phase: ScheduleLnsTestPhase,
    control: PhaseControl<'_>,
) -> Result<(), ScheduleStateInterrupted> {
    if config.test_delay_phase != Some(phase) {
        return Ok(());
    }
    let started = Instant::now();
    while started.elapsed() < config.test_phase_delay {
        control.checkpoint()?;
        std::thread::park_timeout(WATCHDOG_POLL.min(config.test_phase_delay.saturating_sub(started.elapsed())));
    }
    control.checkpoint()
}
