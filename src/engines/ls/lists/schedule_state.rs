//! Compact disjunctive state for strict job-shop schedules.
//!
//! The semantic schedule remains the source of truth. This module recognizes a
//! deliberately narrow, safe subset, represents each unary machine by an
//! explicit operation order, and reconstructs a complete schedule by longest
//! path. Unsupported schedules stay on the general scheduling fallback.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::engines::ls::schedule_ir::PrecedenceDag;
use crate::mix64;
use crate::model::list::{CollectionSolution, Resource, Schedule};

use super::move_acceptance::MinimizingMoveAcceptance;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScheduleStateInterrupted;

const STRICT_N5_MIN_STREAMING_POP_CAP: usize = 4_096;

fn strict_n5_streaming_pop_cap(operation_count: usize) -> usize {
    operation_count.max(STRICT_N5_MIN_STREAMING_POP_CAP)
}

fn checkpoint(stop: &AtomicBool) -> Result<(), ScheduleStateInterrupted> {
    if stop.load(Ordering::Acquire) {
        Err(ScheduleStateInterrupted)
    } else {
        Ok(())
    }
}

/// Duration coefficients and provenance labels for the bounded adjusted-work
/// Giffler-Thompson portfolio.
///
/// Each lane maximizes `remaining_work - coefficient * duration` after first
/// minimizing earliest start. Keep the table as the single retuning and
/// provenance point for the experimental seven-worker portfolio.
pub(crate) const ADJUSTED_WORK_DISPATCH_SPECS: [(i128, &str); 5] = [
    (24, "earliest-start-then-most-adjusted-work-c24"),
    (51, "earliest-start-then-most-adjusted-work-c51"),
    (48, "earliest-start-then-most-adjusted-work-c48"),
    (16, "earliest-start-then-most-adjusted-work-c16"),
    (41, "earliest-start-then-most-adjusted-work-c41"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdjustedWorkDispatchLane {
    Lane0,
    Lane1,
    Lane2,
    Lane3,
    Lane4,
}

impl AdjustedWorkDispatchLane {
    pub(crate) const ALL: [Self; 5] = [Self::Lane0, Self::Lane1, Self::Lane2, Self::Lane3, Self::Lane4];

    fn index(self) -> usize {
        match self {
            Self::Lane0 => 0,
            Self::Lane1 => 1,
            Self::Lane2 => 2,
            Self::Lane3 => 3,
            Self::Lane4 => 4,
        }
    }

    pub(crate) fn duration_coefficient(self) -> i128 {
        ADJUSTED_WORK_DISPATCH_SPECS[self.index()].0
    }

    pub(crate) fn dispatch_name(self) -> &'static str {
        ADJUSTED_WORK_DISPATCH_SPECS[self.index()].1
    }
}

/// Generic dispatch rules for the Giffler-Thompson constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DispatchRule {
    EarliestStart,
    EarliestStartThenMostWorkRemaining,
    EarliestStartThenMostAdjustedWork(AdjustedWorkDispatchLane),
    ShortestProcessingTime,
    LongestProcessingTime,
    MostWorkRemaining,
    Randomized,
}

impl DispatchRule {
    pub(crate) const ADJUSTED_WORK_PORTFOLIO: [Self; 5] = [
        Self::EarliestStartThenMostAdjustedWork(AdjustedWorkDispatchLane::Lane0),
        Self::EarliestStartThenMostAdjustedWork(AdjustedWorkDispatchLane::Lane1),
        Self::EarliestStartThenMostAdjustedWork(AdjustedWorkDispatchLane::Lane2),
        Self::EarliestStartThenMostAdjustedWork(AdjustedWorkDispatchLane::Lane3),
        Self::EarliestStartThenMostAdjustedWork(AdjustedWorkDispatchLane::Lane4),
    ];

    pub(crate) const ALL: [Self; 11] = [
        Self::EarliestStart,
        Self::EarliestStartThenMostWorkRemaining,
        Self::ADJUSTED_WORK_PORTFOLIO[0],
        Self::ADJUSTED_WORK_PORTFOLIO[1],
        Self::ADJUSTED_WORK_PORTFOLIO[2],
        Self::ADJUSTED_WORK_PORTFOLIO[3],
        Self::ADJUSTED_WORK_PORTFOLIO[4],
        Self::ShortestProcessingTime,
        Self::LongestProcessingTime,
        Self::MostWorkRemaining,
        Self::Randomized,
    ];

    /// Preserve the historical generic multistart sequence. The profiled
    /// constructor ablation is selected explicitly by its owning worker and
    /// must not perturb any other construction path.
    pub(crate) const LEGACY_MULTISTART: [Self; 5] =
        [Self::EarliestStart, Self::ShortestProcessingTime, Self::LongestProcessingTime, Self::MostWorkRemaining, Self::Randomized];
}

/// Validated strict job-shop data shared by constructors and states.
#[derive(Clone)]
pub(crate) struct JobShopProblem {
    durations: Vec<i64>,
    all_durations_positive: bool,
    n8_classical_chain_model: bool,
    horizons: Vec<i64>,
    start_windows: Vec<(i64, i64)>,
    raw_machines: Vec<usize>,
    operation_machines: Vec<usize>,
    machine_count: usize,
    precedences: PrecedenceDag,
    precedence_sinks: Vec<usize>,
    remaining_work: Vec<i64>,
    solution_machines: Vec<i64>,
    solution_modes: Vec<Option<usize>>,
}

impl JobShopProblem {
    /// Recognize a mandatory, fixed-assignment job shop.
    ///
    /// Fixed intervals obtain their machine from at most one `NoOverlap`
    /// resource. An interval omitted from all unary groups is assigned a private
    /// internal machine, which adds no semantic restriction. Alternatively,
    /// every interval may carry exactly one mode and a single
    /// `MachineNoOverlap` resource. Cumulative resources, optional operations,
    /// flexible modes, and mixed encodings are intentionally left to the
    /// general scheduling engine.
    pub(crate) fn recognize(schedule: &Schedule, stop: &AtomicBool) -> Result<Option<Self>, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let operation_count = schedule.intervals.len();
        if operation_count == 0 || !schedule.minimize_makespan {
            return Ok(None);
        }

        let mut all_fixed = true;
        let mut all_single_mode = true;
        for interval in &schedule.intervals {
            checkpoint(stop)?;
            if interval.optional {
                return Ok(None);
            }
            all_fixed &= interval.modes.is_empty();
            all_single_mode &= interval.modes.len() == 1;
        }
        if !all_fixed && !all_single_mode {
            return Ok(None);
        }

        let (raw_machines, durations, start_windows, solution_machines, solution_modes) = if all_fixed {
            let Some(raw_machines) = recognize_fixed_machines(schedule, stop)? else {
                return Ok(None);
            };
            let mut durations = Vec::with_capacity(operation_count);
            let mut start_windows = Vec::with_capacity(operation_count);
            for interval in &schedule.intervals {
                checkpoint(stop)?;
                if interval.duration < 0 || interval.horizon < interval.duration {
                    return Ok(None);
                }
                durations.push(interval.duration);
                start_windows.push((0, interval.horizon - interval.duration));
            }
            (raw_machines, durations, start_windows, vec![-1; operation_count], vec![None; operation_count])
        } else {
            let mut machine_resources = 0usize;
            for resource in &schedule.resources {
                checkpoint(stop)?;
                machine_resources += usize::from(matches!(resource, Resource::MachineNoOverlap));
            }
            if machine_resources != 1 || schedule.resources.len() != 1 {
                return Ok(None);
            }
            let mut raw_machines = Vec::with_capacity(operation_count);
            let mut durations = Vec::with_capacity(operation_count);
            let mut start_windows = Vec::with_capacity(operation_count);
            let mut solution_machines = Vec::with_capacity(operation_count);
            let mut solution_modes = Vec::with_capacity(operation_count);
            for interval in &schedule.intervals {
                checkpoint(stop)?;
                let mode = &interval.modes[0];
                let (start_min, start_max) = mode.start_window;
                let Some(last_start) = interval.horizon.checked_sub(mode.duration) else {
                    return Ok(None);
                };
                let Ok(machine) = i64::try_from(mode.machine) else {
                    return Ok(None);
                };
                if mode.duration < 0 || start_min < 0 || start_min > start_max || start_max > last_start {
                    return Ok(None);
                }
                raw_machines.push(mode.machine);
                durations.push(mode.duration);
                start_windows.push(mode.start_window);
                solution_machines.push(machine);
                solution_modes.push(mode.reference);
            }
            (raw_machines, durations, start_windows, solution_machines, solution_modes)
        };

        let (operation_machines, machine_count) = compact_machines(&raw_machines, stop)?;
        let mut successors = vec![Vec::new(); operation_count];
        for &(before, after) in &schedule.precedences {
            checkpoint(stop)?;
            if before >= operation_count || after >= operation_count {
                return Ok(None);
            }
            successors[before].push(after);
        }
        let mut precedence_sinks = Vec::new();
        for (operation, list) in successors.iter_mut().enumerate() {
            checkpoint(stop)?;
            list.sort_unstable();
            checkpoint(stop)?;
            list.dedup();
            if list.is_empty() {
                precedence_sinks.push(operation);
            }
        }
        let Some(precedences) = PrecedenceDag::compile(successors, stop) else {
            checkpoint(stop)?;
            return Ok(None);
        };
        let n8_classical_chain_model = recognizes_classical_n8_chains(&precedences, &operation_machines, machine_count, stop)?;
        let Some(remaining_work) = precedences.remaining_paths(&durations, stop) else {
            checkpoint(stop)?;
            return Ok(None);
        };

        let mut horizons = Vec::with_capacity(operation_count);
        for interval in &schedule.intervals {
            checkpoint(stop)?;
            horizons.push(interval.horizon);
        }
        let all_durations_positive = durations.iter().all(|&duration| duration > 0);
        Ok(Some(Self {
            durations,
            all_durations_positive,
            n8_classical_chain_model,
            horizons,
            start_windows,
            raw_machines,
            operation_machines,
            machine_count,
            precedences,
            precedence_sinks,
            remaining_work,
            solution_machines,
            solution_modes,
        }))
    }

    pub(crate) fn operation_count(&self) -> usize {
        self.durations.len()
    }

    pub(crate) fn machine_count(&self) -> usize {
        self.machine_count
    }

    pub(crate) fn duration(&self, operation: usize) -> i64 {
        self.durations[operation]
    }

    pub(crate) fn start_window(&self, operation: usize) -> (i64, i64) {
        self.start_windows[operation]
    }

    pub(crate) fn horizon(&self, operation: usize) -> i64 {
        self.horizons[operation]
    }

    pub(crate) fn job_predecessors(&self, operation: usize) -> &[usize] {
        self.precedences.predecessors(operation)
    }

    pub(crate) fn job_successors(&self, operation: usize) -> &[usize] {
        self.precedences.successors(operation)
    }

    pub(crate) fn machine(&self, operation: usize) -> usize {
        self.operation_machines[operation]
    }

    pub(crate) fn raw_machine(&self, operation: usize) -> Option<usize> {
        self.raw_machines.get(operation).copied()
    }

    pub(crate) fn solution_machine(&self, operation: usize) -> Option<i64> {
        self.solution_machines.get(operation).copied()
    }

    pub(crate) fn solution_mode(&self, operation: usize) -> Option<Option<usize>> {
        self.solution_modes.get(operation).copied()
    }

    fn solution_from_starts(&self, starts: Vec<i64>, objective: i64) -> CollectionSolution {
        CollectionSolution {
            lists: Vec::new(),
            objectives: vec![objective],
            feasible: true,
            starts,
            presences: vec![true; self.operation_count()],
            machines: self.solution_machines.clone(),
            modes: self.solution_modes.clone(),
            bound: None,
        }
    }

    fn supports_strict_n5_fast_path(&self) -> bool {
        self.n8_classical_chain_model
            && self.all_durations_positive
            && self.start_windows.iter().enumerate().all(|(operation, &(start_min, start_max))| {
                start_min == 0 && self.horizons[operation].checked_sub(self.durations[operation]) == Some(start_max)
            })
    }
}

fn recognizes_classical_n8_chains(
    precedences: &PrecedenceDag,
    operation_machines: &[usize],
    machine_count: usize,
    stop: &AtomicBool,
) -> Result<bool, ScheduleStateInterrupted> {
    for operation in 0..operation_machines.len() {
        checkpoint(stop)?;
        if precedences.predecessors(operation).len() > 1 || precedences.successors(operation).len() > 1 {
            return Ok(false);
        }
    }

    let mut machine_chain_owner = vec![NO_OPERATION; machine_count];
    let mut visited = 0usize;
    for source in 0..operation_machines.len() {
        checkpoint(stop)?;
        if !precedences.predecessors(source).is_empty() {
            continue;
        }
        let mut operation = source;
        loop {
            checkpoint(stop)?;
            let machine = operation_machines[operation];
            if machine_chain_owner[machine] == source {
                return Ok(false);
            }
            machine_chain_owner[machine] = source;
            visited = visited.saturating_add(1);
            let Some(&successor) = precedences.successors(operation).first() else {
                break;
            };
            operation = successor;
        }
    }
    Ok(visited == operation_machines.len())
}

fn recognize_fixed_machines(schedule: &Schedule, stop: &AtomicBool) -> Result<Option<Vec<usize>>, ScheduleStateInterrupted> {
    let operation_count = schedule.intervals.len();
    let mut assignments = vec![None; operation_count];
    let mut unary_machine_count = 0usize;
    for resource in &schedule.resources {
        checkpoint(stop)?;
        let Resource::NoOverlap(operations) = resource else {
            return Ok(None);
        };
        for &operation in operations {
            checkpoint(stop)?;
            let Some(assignment) = assignments.get_mut(operation) else {
                return Ok(None);
            };
            if assignment.replace(unary_machine_count).is_some() {
                return Ok(None);
            }
        }
        unary_machine_count = unary_machine_count.saturating_add(1);
    }

    let mut next_private_machine = unary_machine_count;
    let mut raw_machines = Vec::with_capacity(operation_count);
    for assignment in assignments {
        checkpoint(stop)?;
        raw_machines.push(assignment.unwrap_or_else(|| {
            let machine = next_private_machine;
            next_private_machine = next_private_machine.saturating_add(1);
            machine
        }));
    }
    Ok(Some(raw_machines))
}

fn compact_machines(raw_machines: &[usize], stop: &AtomicBool) -> Result<(Vec<usize>, usize), ScheduleStateInterrupted> {
    let mut compact = BTreeMap::new();
    for &machine in raw_machines {
        checkpoint(stop)?;
        if !compact.contains_key(&machine) {
            let index = compact.len();
            compact.insert(machine, index);
        }
    }
    let mut machines = Vec::with_capacity(raw_machines.len());
    for machine in raw_machines {
        checkpoint(stop)?;
        machines.push(compact[machine]);
    }
    Ok((machines, compact.len()))
}

/// A maximal contiguous machine segment belonging to at least one critical
/// path.
///
/// Blocks are not limited to the deterministic canonical path exposed by
/// [`JobShopState::critical_path`]. Every operation in a block has zero
/// head-tail slack and every internal machine arc is tight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CriticalBlock {
    machine: usize,
    first_position: usize,
    last_position: usize,
}

impl CriticalBlock {
    pub(crate) fn machine(self) -> usize {
        self.machine
    }

    pub(crate) fn first_position(self) -> usize {
        self.first_position
    }

    pub(crate) fn last_position(self) -> usize {
        self.last_position
    }

    pub(crate) fn len(self) -> usize {
        self.last_position - self.first_position + 1
    }
}

/// Deterministic tie policy for the single critical path consumed by strict N5.
///
/// `Historical` preserves the original machine-first, greatest-operation
/// ordering exactly. The other variants provide reproducible portfolio lanes
/// without changing dates, feasibility, or the complete zero-slack block set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CanonicalPathPolicy {
    Historical,
    MachineFirstLowest,
    JobFirstHighest,
    JobFirstLowest,
    OperationHighest,
    OperationLowest,
    Mixed,
}

#[derive(Clone, Copy)]
struct CanonicalChoice {
    machine_arc: bool,
    operation: usize,
}

fn canonical_choice_better(policy: CanonicalPathPolicy, anchor: usize, candidate: CanonicalChoice, incumbent: CanonicalChoice) -> bool {
    match policy {
        CanonicalPathPolicy::Historical => (candidate.machine_arc, candidate.operation) > (incumbent.machine_arc, incumbent.operation),
        CanonicalPathPolicy::MachineFirstLowest => {
            (candidate.machine_arc && !incumbent.machine_arc)
                || (candidate.machine_arc == incumbent.machine_arc && candidate.operation < incumbent.operation)
        }
        CanonicalPathPolicy::JobFirstHighest => {
            (!candidate.machine_arc && incumbent.machine_arc)
                || (candidate.machine_arc == incumbent.machine_arc && candidate.operation > incumbent.operation)
        }
        CanonicalPathPolicy::JobFirstLowest => {
            (!candidate.machine_arc && incumbent.machine_arc)
                || (candidate.machine_arc == incumbent.machine_arc && candidate.operation < incumbent.operation)
        }
        CanonicalPathPolicy::OperationHighest => {
            candidate.operation > incumbent.operation
                || (candidate.operation == incumbent.operation && candidate.machine_arc && !incumbent.machine_arc)
        }
        CanonicalPathPolicy::OperationLowest => {
            candidate.operation < incumbent.operation
                || (candidate.operation == incumbent.operation && candidate.machine_arc && !incumbent.machine_arc)
        }
        CanonicalPathPolicy::Mixed => {
            let mixed_key = |choice: CanonicalChoice| {
                let operation = u64::try_from(choice.operation).unwrap_or(u64::MAX);
                let anchor = u64::try_from(anchor).unwrap_or(u64::MAX);
                mix64(operation ^ anchor.rotate_left(23) ^ u64::from(choice.machine_arc).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            };
            (mixed_key(candidate), candidate.machine_arc, candidate.operation)
                > (mixed_key(incumbent), incumbent.machine_arc, incumbent.operation)
        }
    }
}

fn consider_canonical_choice(
    policy: CanonicalPathPolicy,
    anchor: usize,
    selected: &mut Option<CanonicalChoice>,
    candidate: CanonicalChoice,
) {
    if selected.is_none_or(|incumbent| canonical_choice_better(policy, anchor, candidate, incumbent)) {
        *selected = Some(candidate);
    }
}

fn select_canonical_terminal(
    candidates: impl Iterator<Item = usize>,
    machine_predecessors: &[usize],
    starts: &[i64],
    ends: &[i64],
    makespan: i64,
    policy: CanonicalPathPolicy,
    stop: &AtomicBool,
) -> Result<Option<CanonicalChoice>, ReconstructionFailure> {
    let mut terminal = None;
    for operation in candidates {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        if ends[operation] != makespan {
            continue;
        }
        let machine_predecessor = machine_predecessors[operation];
        let has_tight_machine_predecessor = machine_predecessor != NO_OPERATION && ends[machine_predecessor] == starts[operation];
        consider_canonical_choice(
            policy,
            NO_OPERATION,
            &mut terminal,
            CanonicalChoice { machine_arc: has_tight_machine_predecessor, operation },
        );
    }
    Ok(terminal)
}

#[allow(clippy::too_many_arguments)]
fn build_canonical_path_from_terminal(
    critical_path: &mut Vec<usize>,
    canonical_critical_blocks: &mut Vec<CriticalBlock>,
    problem: &JobShopProblem,
    machine_predecessors: &[usize],
    positions: &[usize],
    starts: &[i64],
    ends: &[i64],
    policy: CanonicalPathPolicy,
    terminal: Option<CanonicalChoice>,
    fine_grained_checkpoints: bool,
    stop_after_path_operations: &mut Option<usize>,
    stop: &AtomicBool,
) -> Result<(), ReconstructionFailure> {
    critical_path.clear();
    if let Some(CanonicalChoice { mut operation, .. }) = terminal {
        loop {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            critical_path.push(operation);
            if *stop_after_path_operations == Some(critical_path.len()) {
                *stop_after_path_operations = None;
                stop.store(true, Ordering::Release);
                return Err(ReconstructionFailure::Interrupted);
            }
            let machine_predecessor = machine_predecessors[operation];
            let mut predecessor_choice = None;
            for &predecessor in problem.precedences.predecessors(operation) {
                if fine_grained_checkpoints {
                    checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
                }
                if ends[predecessor] == starts[operation] {
                    consider_canonical_choice(
                        policy,
                        operation,
                        &mut predecessor_choice,
                        CanonicalChoice { machine_arc: predecessor == machine_predecessor, operation: predecessor },
                    );
                }
            }
            if machine_predecessor != NO_OPERATION
                && !problem.precedences.predecessors(operation).contains(&machine_predecessor)
                && ends[machine_predecessor] == starts[operation]
            {
                consider_canonical_choice(
                    policy,
                    operation,
                    &mut predecessor_choice,
                    CanonicalChoice { machine_arc: true, operation: machine_predecessor },
                );
            }
            let Some(CanonicalChoice { operation: predecessor, .. }) = predecessor_choice else {
                break;
            };
            operation = predecessor;
            if critical_path.len() > problem.operation_count() {
                return Err(ReconstructionFailure::Cycle);
            }
        }
        critical_path.reverse();
    }

    canonical_critical_blocks.clear();
    let mut path_start = 0usize;
    while path_start < critical_path.len() {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        let machine = problem.machine(critical_path[path_start]);
        let first_position = positions[critical_path[path_start]];
        let mut path_end = path_start;
        while path_end + 1 < critical_path.len()
            && problem.machine(critical_path[path_end + 1]) == machine
            && positions[critical_path[path_end + 1]] == positions[critical_path[path_end]] + 1
        {
            if fine_grained_checkpoints {
                checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            }
            path_end += 1;
        }
        if path_end > path_start {
            canonical_critical_blocks.push(CriticalBlock { machine, first_position, last_position: positions[critical_path[path_end]] });
        }
        path_start = path_end + 1;
    }
    Ok(())
}

const NO_OPERATION: usize = usize::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReconstructionFailure {
    Interrupted,
    Cycle,
    Window,
    Numeric,
}

#[derive(Clone, Debug)]
struct Reconstruction {
    starts: Vec<i64>,
    ends: Vec<i64>,
    latest_starts: Vec<i64>,
    tails: Vec<i64>,
    topological: Vec<usize>,
    topological_rank: Vec<usize>,
    makespan: i64,
    critical_path: Vec<usize>,
    canonical_critical_blocks: Vec<CriticalBlock>,
    critical_blocks: Vec<CriticalBlock>,
}

impl Reconstruction {
    fn empty(operation_count: usize) -> Self {
        Self {
            starts: vec![0; operation_count],
            ends: vec![0; operation_count],
            latest_starts: vec![0; operation_count],
            tails: vec![0; operation_count],
            topological: Vec::with_capacity(operation_count),
            topological_rank: vec![0; operation_count],
            makespan: 0,
            critical_path: Vec::with_capacity(operation_count),
            canonical_critical_blocks: Vec::with_capacity(operation_count),
            critical_blocks: Vec::with_capacity(operation_count),
        }
    }

    fn commit(&mut self, workspace: &EvaluationWorkspace) {
        self.starts.copy_from_slice(&workspace.trial_starts);
        self.ends.copy_from_slice(&workspace.trial_ends);
        self.latest_starts.copy_from_slice(&workspace.trial_latest_starts);
        self.tails.copy_from_slice(&workspace.trial_tails);
        self.topological.clear();
        self.topological.extend_from_slice(&workspace.trial_topological);
        self.topological_rank.copy_from_slice(&workspace.trial_topological_rank);
        self.makespan = workspace.trial_makespan;
        self.critical_path.clear();
        self.critical_path.extend_from_slice(&workspace.trial_critical_path);
        self.canonical_critical_blocks.clear();
        self.canonical_critical_blocks.extend_from_slice(&workspace.trial_canonical_critical_blocks);
        self.critical_blocks.clear();
        self.critical_blocks.extend_from_slice(&workspace.trial_critical_blocks);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScheduleWorkspaceCapacities {
    pub(crate) ready: usize,
    pub(crate) changed_roots: usize,
    pub(crate) dirty_queue: usize,
    pub(crate) dirty_topological: usize,
    pub(crate) date_patch: usize,
    pub(crate) topological: usize,
    pub(crate) critical_path: usize,
    pub(crate) canonical_critical_blocks: usize,
    pub(crate) critical_blocks: usize,
}

struct EvaluationWorkspace {
    indegrees: Vec<usize>,
    ready: Vec<usize>,
    trial_starts: Vec<i64>,
    trial_ends: Vec<i64>,
    trial_latest_starts: Vec<i64>,
    trial_tails: Vec<i64>,
    trial_topological: Vec<usize>,
    trial_topological_rank: Vec<usize>,
    trial_makespan: i64,
    trial_critical_path: Vec<usize>,
    trial_canonical_critical_blocks: Vec<CriticalBlock>,
    trial_critical_blocks: Vec<CriticalBlock>,
    changed_roots: Vec<usize>,
    dirty_marks: Vec<u32>,
    dirty_epoch: u32,
    dirty_queue: Vec<usize>,
    dirty_topological: Vec<usize>,
    patched_operations: Vec<usize>,
    patched_starts: Vec<i64>,
    patched_ends: Vec<i64>,
    fast_patch_marks: Vec<u32>,
    fast_patch_epoch: u32,
    fast_patched_operations: Vec<usize>,
    fast_patched_starts: Vec<i64>,
    fast_patched_ends: Vec<i64>,
    fast_patched_tails: Vec<i64>,
    fast_queue_marks: Vec<u32>,
    local_span_operations: Vec<usize>,
    local_heads: Vec<i128>,
    local_tails: Vec<i128>,
    observed_capacities: [usize; 14],
    growths: u64,
    #[cfg(test)]
    fast_forward_value_change_pop_cap: Option<usize>,
    #[cfg(test)]
    fast_reverse_value_change_pop_cap: Option<usize>,
    #[cfg(test)]
    fast_stop_on_work_cap: bool,
    #[cfg(test)]
    fast_stop_during_reverse_recovery: bool,
    #[cfg(test)]
    fast_last_work_cap_phase: Option<FastN5PropagationPhase>,
    #[cfg(test)]
    batch_stop_after_applied_moves: Option<usize>,
    analysis_stop_after_path_operations: Option<usize>,
}

impl EvaluationWorkspace {
    fn new(operation_count: usize) -> Self {
        let mut workspace = Self {
            indegrees: vec![0; operation_count],
            ready: Vec::with_capacity(operation_count),
            trial_starts: vec![0; operation_count],
            trial_ends: vec![0; operation_count],
            trial_latest_starts: vec![0; operation_count],
            trial_tails: vec![0; operation_count],
            trial_topological: Vec::with_capacity(operation_count),
            trial_topological_rank: vec![0; operation_count],
            trial_makespan: 0,
            trial_critical_path: Vec::with_capacity(operation_count),
            trial_canonical_critical_blocks: Vec::with_capacity(operation_count),
            trial_critical_blocks: Vec::with_capacity(operation_count),
            changed_roots: Vec::with_capacity(operation_count),
            dirty_marks: vec![0; operation_count],
            dirty_epoch: 0,
            dirty_queue: Vec::with_capacity(operation_count),
            dirty_topological: Vec::with_capacity(operation_count),
            patched_operations: Vec::with_capacity(operation_count),
            patched_starts: Vec::with_capacity(operation_count),
            patched_ends: Vec::with_capacity(operation_count),
            fast_patch_marks: vec![0; operation_count],
            fast_patch_epoch: 0,
            fast_patched_operations: Vec::with_capacity(operation_count.min(65_536)),
            fast_patched_starts: Vec::with_capacity(operation_count.min(65_536)),
            fast_patched_ends: Vec::with_capacity(operation_count.min(65_536)),
            fast_patched_tails: Vec::with_capacity(operation_count.min(65_536)),
            fast_queue_marks: vec![0; operation_count],
            local_span_operations: Vec::with_capacity(operation_count.min(4_096)),
            local_heads: Vec::with_capacity(operation_count.min(4_096)),
            local_tails: Vec::with_capacity(operation_count.min(4_096)),
            observed_capacities: [0; 14],
            growths: 0,
            #[cfg(test)]
            fast_forward_value_change_pop_cap: None,
            #[cfg(test)]
            fast_reverse_value_change_pop_cap: None,
            #[cfg(test)]
            fast_stop_on_work_cap: false,
            #[cfg(test)]
            fast_stop_during_reverse_recovery: false,
            #[cfg(test)]
            fast_last_work_cap_phase: None,
            #[cfg(test)]
            batch_stop_after_applied_moves: None,
            analysis_stop_after_path_operations: None,
        };
        workspace.observed_capacities = workspace.capacity_snapshot();
        workspace
    }

    fn capacities(&self) -> ScheduleWorkspaceCapacities {
        ScheduleWorkspaceCapacities {
            ready: self.ready.capacity(),
            changed_roots: self.changed_roots.capacity(),
            dirty_queue: self.dirty_queue.capacity(),
            dirty_topological: self.dirty_topological.capacity(),
            date_patch: self.patched_operations.capacity().min(self.patched_starts.capacity()).min(self.patched_ends.capacity()),
            topological: self.trial_topological.capacity(),
            critical_path: self.trial_critical_path.capacity(),
            canonical_critical_blocks: self.trial_canonical_critical_blocks.capacity(),
            critical_blocks: self.trial_critical_blocks.capacity(),
        }
    }

    fn capacity_snapshot(&self) -> [usize; 14] {
        [
            self.ready.capacity(),
            self.trial_topological.capacity(),
            self.trial_critical_path.capacity(),
            self.trial_canonical_critical_blocks.capacity(),
            self.trial_critical_blocks.capacity(),
            self.changed_roots.capacity(),
            self.dirty_queue.capacity(),
            self.dirty_topological.capacity(),
            self.patched_operations.capacity(),
            self.patched_starts.capacity(),
            self.patched_ends.capacity(),
            self.local_span_operations.capacity(),
            self.local_heads.capacity(),
            self.local_tails.capacity(),
        ]
    }

    fn observe_growths(&mut self) {
        let current = self.capacity_snapshot();
        for (observed, current) in self.observed_capacities.into_iter().zip(current) {
            if current > observed {
                self.growths = self.growths.saturating_add(1);
            }
        }
        self.observed_capacities = current;
    }

    fn next_dirty_epoch(&mut self) -> u32 {
        self.dirty_epoch = self.dirty_epoch.wrapping_add(1);
        if self.dirty_epoch == 0 {
            self.dirty_marks.fill(0);
            self.fast_queue_marks.fill(0);
            self.dirty_epoch = 1;
        }
        self.dirty_epoch
    }

    fn begin_fast_patch(&mut self) -> u32 {
        self.fast_patch_epoch = self.fast_patch_epoch.wrapping_add(1);
        if self.fast_patch_epoch == 0 {
            self.fast_patch_marks.fill(0);
            self.fast_patch_epoch = 1;
        }
        self.fast_patched_operations.clear();
        self.fast_patched_starts.clear();
        self.fast_patched_ends.clear();
        self.fast_patched_tails.clear();
        self.fast_patch_epoch
    }

    fn full_rebuild(
        &mut self,
        problem: &JobShopProblem,
        machine_predecessors: &[usize],
        machine_successors: &[usize],
        stop: &AtomicBool,
    ) -> Result<i64, ReconstructionFailure> {
        self.rebuild_topology(problem, machine_predecessors, machine_successors, stop)?;
        for &operation in &self.trial_topological {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let (start, end) = earliest_dates(problem, machine_predecessors, &self.trial_ends, operation)?;
            self.trial_starts[operation] = start;
            self.trial_ends[operation] = end;
        }
        self.trial_makespan = self.trial_ends.iter().copied().max().unwrap_or(0);
        Ok(self.trial_makespan)
    }

    fn rebuild_topology(
        &mut self,
        problem: &JobShopProblem,
        machine_predecessors: &[usize],
        machine_successors: &[usize],
        stop: &AtomicBool,
    ) -> Result<(), ReconstructionFailure> {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        let operation_count = problem.operation_count();
        self.ready.clear();
        self.trial_topological.clear();
        self.indegrees.fill(0);

        for (operation, degree) in self.indegrees.iter_mut().enumerate() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let mut indegree = problem.precedences.predecessors(operation).len();
            let machine_predecessor = machine_predecessors[operation];
            if machine_predecessor != NO_OPERATION && !problem.precedences.predecessors(operation).contains(&machine_predecessor) {
                indegree = indegree.checked_add(1).ok_or(ReconstructionFailure::Numeric)?;
            }
            *degree = indegree;
            if indegree == 0 {
                heap_push(&mut self.ready, operation);
            }
        }

        while let Some(operation) = heap_pop(&mut self.ready) {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            self.trial_topological.push(operation);
            for &successor in problem.precedences.successors(operation) {
                release_topological_successor(&mut self.indegrees, &mut self.ready, successor)?;
            }
            let machine_successor = machine_successors[operation];
            if machine_successor != NO_OPERATION && !problem.precedences.successors(operation).contains(&machine_successor) {
                release_topological_successor(&mut self.indegrees, &mut self.ready, machine_successor)?;
            }
        }
        if self.trial_topological.len() != operation_count {
            return Err(ReconstructionFailure::Cycle);
        }

        Ok(())
    }

    fn rebuild_dirty_topology(
        &mut self,
        problem: &JobShopProblem,
        machine_successors: &[usize],
        epoch: u32,
        stop: &AtomicBool,
    ) -> Result<(), ReconstructionFailure> {
        // `evaluate_delta` populated the induced indegrees while building the
        // forward closure. A FIFO Kahn traversal is sufficient because dates
        // depend on precedence, not on the tie order among ready operations.
        self.ready.clear();
        self.dirty_topological.clear();
        for &operation in &self.dirty_queue {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            if self.indegrees[operation] == 0 {
                self.ready.push(operation);
            }
        }

        let mut cursor = 0usize;
        while cursor < self.ready.len() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let operation = self.ready[cursor];
            cursor += 1;
            self.dirty_topological.push(operation);
            for &successor in problem.precedences.successors(operation) {
                if self.dirty_marks[successor] == epoch {
                    release_fifo_successor(&mut self.indegrees, &mut self.ready, successor)?;
                }
            }
            let machine_successor = machine_successors[operation];
            if machine_successor != NO_OPERATION
                && self.dirty_marks[machine_successor] == epoch
                && !problem.precedences.successors(operation).contains(&machine_successor)
            {
                release_fifo_successor(&mut self.indegrees, &mut self.ready, machine_successor)?;
            }
        }
        if self.dirty_topological.len() != self.dirty_queue.len() {
            return Err(ReconstructionFailure::Cycle);
        }
        Ok(())
    }

    fn rebuild_reverse_dirty_topology(
        &mut self,
        problem: &JobShopProblem,
        machine_predecessors: &[usize],
        epoch: u32,
        stop: &AtomicBool,
    ) -> Result<(), ReconstructionFailure> {
        // The reverse closure stores induced outdegrees in `indegrees`.
        // Starting from its sinks and releasing predecessors yields the exact
        // order needed to recompute tails once per affected operation.
        self.ready.clear();
        self.dirty_topological.clear();
        for &operation in &self.dirty_queue {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            if self.indegrees[operation] == 0 {
                self.ready.push(operation);
            }
        }

        let mut cursor = 0usize;
        while cursor < self.ready.len() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let operation = self.ready[cursor];
            cursor += 1;
            self.dirty_topological.push(operation);
            for &predecessor in problem.precedences.predecessors(operation) {
                if self.dirty_marks[predecessor] == epoch {
                    release_fifo_successor(&mut self.indegrees, &mut self.ready, predecessor)?;
                }
            }
            let machine_predecessor = machine_predecessors[operation];
            if machine_predecessor != NO_OPERATION
                && self.dirty_marks[machine_predecessor] == epoch
                && !problem.precedences.predecessors(operation).contains(&machine_predecessor)
            {
                release_fifo_successor(&mut self.indegrees, &mut self.ready, machine_predecessor)?;
            }
        }
        if self.dirty_topological.len() != self.dirty_queue.len() {
            return Err(ReconstructionFailure::Cycle);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_analysis(
        &mut self,
        problem: &JobShopProblem,
        machine_predecessors: &[usize],
        machine_successors: &[usize],
        positions: &[usize],
        machine_sequences: &[Vec<usize>],
        canonical_path_policy: CanonicalPathPolicy,
        collect_all_critical_blocks: bool,
        stop: &AtomicBool,
    ) -> Result<(), ReconstructionFailure> {
        for (rank, &operation) in self.trial_topological.iter().enumerate() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            self.trial_topological_rank[operation] = rank;
        }
        for &operation in self.trial_topological.iter().rev() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let Some(makespan_latest) = self.trial_makespan.checked_sub(problem.duration(operation)) else {
                return Err(ReconstructionFailure::Numeric);
            };
            let mut latest = problem.start_windows[operation].1.min(makespan_latest);
            let duration = problem.duration(operation);
            let mut tail = duration;
            for &successor in problem.precedences.successors(operation) {
                latest = latest_before_successor(latest, problem.duration(operation), self.trial_latest_starts[successor])?;
                tail = tail.max(duration.checked_add(self.trial_tails[successor]).ok_or(ReconstructionFailure::Numeric)?);
            }
            let machine_successor = machine_successors[operation];
            if machine_successor != NO_OPERATION && !problem.precedences.successors(operation).contains(&machine_successor) {
                latest = latest_before_successor(latest, problem.duration(operation), self.trial_latest_starts[machine_successor])?;
                tail = tail.max(duration.checked_add(self.trial_tails[machine_successor]).ok_or(ReconstructionFailure::Numeric)?);
            }
            if latest < self.trial_starts[operation] {
                return Err(ReconstructionFailure::Window);
            }
            self.trial_latest_starts[operation] = latest;
            self.trial_tails[operation] = tail;
        }

        let terminal = select_canonical_terminal(
            self.trial_topological.iter().copied(),
            machine_predecessors,
            &self.trial_starts,
            &self.trial_ends,
            self.trial_makespan,
            canonical_path_policy,
            stop,
        )?;
        build_canonical_path_from_terminal(
            &mut self.trial_critical_path,
            &mut self.trial_canonical_critical_blocks,
            problem,
            machine_predecessors,
            positions,
            &self.trial_starts,
            &self.trial_ends,
            canonical_path_policy,
            terminal,
            true,
            &mut self.analysis_stop_after_path_operations,
            stop,
        )?;

        self.trial_critical_blocks.clear();
        if !collect_all_critical_blocks {
            return Ok(());
        }
        for (machine, sequence) in machine_sequences.iter().enumerate() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let mut block_start = None;
            for position in 0..sequence.len() {
                checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
                let operation = sequence[position];
                let critical = self.trial_starts[operation]
                    .checked_add(self.trial_tails[operation])
                    .is_some_and(|critical_end| critical_end == self.trial_makespan);
                let extends_block = position > 0
                    && critical
                    && block_start.is_some()
                    && self.trial_ends[sequence[position - 1]] == self.trial_starts[operation];
                if extends_block {
                    continue;
                }
                if let Some(first_position) = block_start.take() {
                    if position - first_position >= 2 {
                        self.trial_critical_blocks.push(CriticalBlock { machine, first_position, last_position: position - 1 });
                    }
                }
                if critical {
                    block_start = Some(position);
                }
            }
            if let Some(first_position) = block_start {
                if sequence.len() - first_position >= 2 {
                    self.trial_critical_blocks.push(CriticalBlock { machine, first_position, last_position: sequence.len() - 1 });
                }
            }
        }
        Ok(())
    }

    /// Return the exact maximum end over a deterministic superset of all sinks
    /// in the combined precedence and machine DAG.
    ///
    /// With non-negative durations, every non-sink has a successor whose end
    /// is no smaller. Therefore some combined sink attains the global maximum.
    /// Every combined sink is both a precedence sink and a machine tail, so the
    /// union scanned here is a safe superset. It is much smaller than all
    /// operations on a conventional job shop.
    fn makespan_from_sink_union(
        &mut self,
        problem: &JobShopProblem,
        machine_sequences: &[Vec<usize>],
        ends: &[i64],
        stop: &AtomicBool,
    ) -> Result<i64, ReconstructionFailure> {
        let epoch = self.next_dirty_epoch();
        let mut makespan = None;
        for &operation in &problem.precedence_sinks {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            if self.dirty_marks[operation] != epoch {
                self.dirty_marks[operation] = epoch;
                makespan = Some(makespan.map_or(ends[operation], |current: i64| current.max(ends[operation])));
            }
        }
        for sequence in machine_sequences {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let Some(&operation) = sequence.last() else {
                continue;
            };
            if self.dirty_marks[operation] != epoch {
                self.dirty_marks[operation] = epoch;
                makespan = Some(makespan.map_or(ends[operation], |current: i64| current.max(ends[operation])));
            }
        }
        Ok(makespan.unwrap_or(0))
    }
}

fn earliest_dates(
    problem: &JobShopProblem,
    machine_predecessors: &[usize],
    ends: &[i64],
    operation: usize,
) -> Result<(i64, i64), ReconstructionFailure> {
    let mut start = problem.start_windows[operation].0;
    for &predecessor in problem.precedences.predecessors(operation) {
        start = start.max(ends[predecessor]);
    }
    let machine_predecessor = machine_predecessors[operation];
    if machine_predecessor != NO_OPERATION {
        start = start.max(ends[machine_predecessor]);
    }
    if start > problem.start_windows[operation].1 {
        return Err(ReconstructionFailure::Window);
    }
    let end = start.checked_add(problem.duration(operation)).ok_or(ReconstructionFailure::Numeric)?;
    if end > problem.horizons[operation] {
        return Err(ReconstructionFailure::Window);
    }
    Ok((start, end))
}

fn latest_before_successor(current: i64, duration: i64, successor_latest: i64) -> Result<i64, ReconstructionFailure> {
    let latest = successor_latest.checked_sub(duration).ok_or(ReconstructionFailure::Numeric)?;
    Ok(current.min(latest))
}

fn release_topological_successor(indegrees: &mut [usize], ready: &mut Vec<usize>, successor: usize) -> Result<(), ReconstructionFailure> {
    let degree = indegrees.get_mut(successor).ok_or(ReconstructionFailure::Cycle)?;
    *degree = degree.checked_sub(1).ok_or(ReconstructionFailure::Cycle)?;
    if *degree == 0 {
        heap_push(ready, successor);
    }
    Ok(())
}

fn release_fifo_successor(indegrees: &mut [usize], ready: &mut Vec<usize>, successor: usize) -> Result<(), ReconstructionFailure> {
    let degree = indegrees.get_mut(successor).ok_or(ReconstructionFailure::Cycle)?;
    *degree = degree.checked_sub(1).ok_or(ReconstructionFailure::Cycle)?;
    if *degree == 0 {
        ready.push(successor);
    }
    Ok(())
}

fn heap_push(heap: &mut Vec<usize>, value: usize) {
    heap.push(value);
    let mut index = heap.len() - 1;
    while index > 0 {
        let parent = (index - 1) / 2;
        if heap[parent] <= heap[index] {
            break;
        }
        heap.swap(parent, index);
        index = parent;
    }
}

fn heap_pop(heap: &mut Vec<usize>) -> Option<usize> {
    let result = *heap.first()?;
    let last = heap.pop()?;
    if !heap.is_empty() {
        heap[0] = last;
        let mut index = 0;
        loop {
            let left = index * 2 + 1;
            if left >= heap.len() {
                break;
            }
            let right = left + 1;
            let child = if right < heap.len() && heap[right] < heap[left] { right } else { left };
            if heap[index] <= heap[child] {
                break;
            }
            heap.swap(index, child);
            index = child;
        }
    }
    Some(result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ScheduleMove {
    AdjacentSwap {
        machine: usize,
        first_position: usize,
    },
    /// Move one operation to the final index `to` in the same machine order.
    Insert {
        machine: usize,
        from: usize,
        to: usize,
    },
}

impl ScheduleMove {
    fn machine(self) -> usize {
        match self {
            Self::AdjacentSwap { machine, .. } | Self::Insert { machine, .. } => machine,
        }
    }

    fn position_bounds(self) -> (usize, usize) {
        match self {
            Self::AdjacentSwap { first_position, .. } => (first_position, first_position + 1),
            Self::Insert { from, to, .. } => (from.min(to), from.max(to)),
        }
    }
}

/// An immediate precedence induced by one unary-machine sequence.
///
/// Operation identities make tabu attributes stable when an insertion changes
/// the positions of every operation between its source and destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MachineArc {
    pub(crate) machine: usize,
    pub(crate) before: usize,
    pub(crate) after: usize,
}

/// Machine arcs removed and added by one valid structured move.
///
/// Adjacent swaps and insertions change at most three immediate machine arcs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScheduleMoveArcs {
    pub(crate) removed: [Option<MachineArc>; 3],
    pub(crate) added: [Option<MachineArc>; 3],
}

/// Deterministic, constant-work guidance computed from the accepted state's
/// heads and tails.
///
/// These values are deliberately heuristic. In particular,
/// `max_added_arc_path` is not a certified makespan bound because a move can
/// remove arcs used by an endpoint's current head or tail. The score may rank
/// cyclic or otherwise infeasible moves. Only [`JobShopState::probe_move`] may
/// decide feasibility and only the full oracle in
/// [`JobShopState::commit_probed_move`] may commit a selected move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadTailMoveScore {
    /// Maximum old-head plus old-tail path induced by one newly added arc.
    pub(crate) max_added_arc_path: i128,
    /// Sum used only as a deterministic secondary ranking signal.
    pub(crate) total_added_arc_path: i128,
    /// Tight zero-slack machine arcs removed from the accepted state.
    pub(crate) critical_arcs_removed: u8,
    /// Added arcs whose endpoints happen to be tight in the accepted dates.
    pub(crate) tight_arcs_added: u8,
}

impl HeadTailMoveScore {
    pub(crate) fn ranking_key(self) -> (i128, i128, u8, Reverse<u8>) {
        (self.max_added_arc_path, self.total_added_arc_path, self.tight_arcs_added, Reverse(self.critical_arcs_removed))
    }
}

/// Taillard's constant-work evaluation of one adjacent strict N5 swap.
///
/// The estimate uses the accepted heads and tails outside the four affected
/// machine positions. It is advisory only: the exact fast kernel remains the
/// sole authority for feasibility and the candidate makespan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StrictN5TaillardScore {
    pub(crate) estimated_makespan: i128,
    pub(crate) secondary: HeadTailMoveScore,
}

impl StrictN5TaillardScore {
    pub(crate) fn ranking_key(self) -> (i128, (i128, i128, u8, Reverse<u8>)) {
        (self.estimated_makespan, self.secondary.ranking_key())
    }
}

/// One structured move paired with advisory head-tail guidance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScoredScheduleMove {
    pub(crate) movement: ScheduleMove,
    pub(crate) score: HeadTailMoveScore,
}

/// Advisory local head-tail propagation over the machine span changed by a
/// structured move.
///
/// `estimated_makespan` is only a ranking signal. It deliberately keeps job
/// heads and tails outside the changed span at their accepted-state values, so
/// it is neither a feasibility test nor an objective bound. The acyclicity bit
/// is different: it is a sufficient certificate obtained when every added
/// machine arc goes forward in one accepted topological order. `false` means
/// unknown, never cyclic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocalMoveEstimate {
    pub(crate) estimated_makespan: i128,
    pub(crate) acyclicity_certified: bool,
    pub(crate) span_operations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CriticalNeighborhood {
    /// Every adjacent exchange inside a critical block.
    N1,
    /// The first and last adjacent exchanges of each critical block.
    N5,
    /// N5 plus non-adjacent insertions of internal operations at a block end.
    N6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoveRejection {
    Invalid,
    Cycle,
    Window,
    Numeric,
    NotAccepted { current: i64, candidate: i64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoveOutcome {
    Accepted { previous: i64, current: i64 },
    Rejected(MoveRejection),
}

/// Transactional evaluation of a move without changing the accepted state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoveProbe {
    Feasible { current: i64, candidate: i64 },
    Rejected(MoveRejection),
}

/// Result of the default-off strict N5 micro-kernel.
///
/// `used_fast_path` is false when the structural gate failed or the bounded
/// value-change propagation fell back to the complete transactional kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FastN5FallbackReason {
    Unsupported,
    WorkCap,
    Cycle,
    Window,
    Numeric,
    Analysis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FastN5Outcome {
    pub(crate) outcome: MoveOutcome,
    pub(crate) used_fast_path: bool,
    pub(crate) fell_back: bool,
    pub(crate) fallback_reason: Option<FastN5FallbackReason>,
    /// A bounded streaming phase exhausted its work cap and an exact Kahn
    /// recovery replaced the artificial fallback in one or both directions.
    pub(crate) used_topological_recovery: bool,
    /// Total value mutations performed, including streaming mutations rolled
    /// back before a topological recovery.
    pub(crate) forward_date_changes: u64,
    pub(crate) reverse_tail_changes: u64,
    /// Total queue work spent across both streaming phases, including any
    /// abandoned prefix and the closure/Kahn traversals used for recovery.
    pub(crate) queue_pops: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastN5PropagationPhase {
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastN5PropagationFailureKind {
    Interrupted,
    WorkCap(FastN5PropagationPhase),
    Reconstruction(ReconstructionFailure),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FastN5PropagationFailure {
    kind: FastN5PropagationFailureKind,
    forward_date_changes: u64,
    reverse_tail_changes: u64,
    queue_pops: u64,
}

impl FastN5PropagationFailure {
    fn new(kind: FastN5PropagationFailureKind, forward_date_changes: u64, reverse_tail_changes: u64, queue_pops: usize) -> Self {
        Self { kind, forward_date_changes, reverse_tail_changes, queue_pops: u64::try_from(queue_pops).unwrap_or(u64::MAX) }
    }

    fn from_reconstruction(
        failure: ReconstructionFailure,
        forward_date_changes: u64,
        reverse_tail_changes: u64,
        queue_pops: usize,
    ) -> Self {
        let kind = match failure {
            ReconstructionFailure::Interrupted => FastN5PropagationFailureKind::Interrupted,
            failure => FastN5PropagationFailureKind::Reconstruction(failure),
        };
        Self::new(kind, forward_date_changes, reverse_tail_changes, queue_pops)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScheduleStateMetrics {
    pub(crate) construction_candidates: u64,
    pub(crate) construction_bucket_visits: u64,
    pub(crate) construction_heap_pushes: u64,
    pub(crate) construction_stale_pops: u64,
    pub(crate) construction_heap_rebuilds: u64,
    pub(crate) construction_heap_peak: u64,
    pub(crate) reconstructions: u64,
    pub(crate) delta_evaluations: u64,
    pub(crate) full_evaluations: u64,
    /// Full candidate-date fallbacks, excluding full topological cycle checks.
    pub(crate) full_fallbacks: u64,
    pub(crate) topological_rebuilds: u64,
    pub(crate) oracle_validations: u64,
    pub(crate) oracle_mismatches: u64,
    pub(crate) dirty_cone_operations: u64,
    pub(crate) max_dirty_cone: u64,
    pub(crate) workspace_growths: u64,
    pub(crate) moves_considered: u64,
    pub(crate) moves_accepted: u64,
    pub(crate) cycle_rejections: u64,
    pub(crate) window_rejections: u64,
    pub(crate) objective_rejections: u64,
    pub(crate) critical_path_updates: u64,
    pub(crate) local_move_estimates: u64,
    pub(crate) local_move_certified: u64,
    pub(crate) local_move_unknown: u64,
    pub(crate) direct_oracle_attempts: u64,
    pub(crate) direct_oracle_accepts: u64,
    pub(crate) direct_oracle_cycles: u64,
    pub(crate) direct_oracle_windows: u64,
    pub(crate) direct_oracle_objective_rejections: u64,
}

/// Complete compact state. Machine orders are authoritative. The default path
/// fully reconstructs every accepted order; the gated strict-N5 path keeps
/// exact dates, tails and canonical critical analysis incrementally, then
/// refreshes auxiliary topological/latest-date buffers at full checkpoints.
pub(crate) struct JobShopState {
    problem: JobShopProblem,
    machine_sequences: Vec<Vec<usize>>,
    machine_predecessors: Vec<usize>,
    machine_successors: Vec<usize>,
    positions: Vec<usize>,
    canonical_path_policy: CanonicalPathPolicy,
    collect_all_critical_blocks: bool,
    reconstruction: Reconstruction,
    workspace: EvaluationWorkspace,
    metrics: ScheduleStateMetrics,
}

impl JobShopState {
    pub(crate) fn giffler_thompson(
        problem: &JobShopProblem,
        seed: u64,
        rule: DispatchRule,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ScheduleStateInterrupted> {
        Self::giffler_thompson_profiled(problem, seed, rule, stop).0
    }

    /// Profiled constructor preserving completed work even when construction is
    /// infeasible or interrupted. A successful state's counters are identical
    /// to the returned snapshot.
    pub(crate) fn giffler_thompson_profiled(
        problem: &JobShopProblem,
        seed: u64,
        rule: DispatchRule,
        stop: &AtomicBool,
    ) -> (Result<Option<Self>, ScheduleStateInterrupted>, ScheduleStateMetrics) {
        let mut metrics = ScheduleStateMetrics::default();
        let result = Self::giffler_thompson_inner(problem, seed, rule, stop, &mut metrics);
        (result, metrics)
    }

    fn giffler_thompson_inner(
        problem: &JobShopProblem,
        seed: u64,
        rule: DispatchRule,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<Option<Self>, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let operation_count = problem.operation_count();
        let mut machine_sequences = vec![Vec::new(); problem.machine_count()];
        let mut completion = vec![0i64; operation_count];
        let mut indegrees = (0..operation_count).map(|operation| problem.precedences.predecessors(operation).len()).collect::<Vec<_>>();
        let mut ready = GifflerReadySet::new(problem);
        for (operation, &indegree) in indegrees.iter().enumerate() {
            checkpoint(stop)?;
            if indegree == 0 && !ready.activate(problem, operation, 0, stop, metrics)? {
                return Ok(None);
            }
        }

        for step in 0..operation_count {
            checkpoint(stop)?;
            ready.maybe_rebuild(problem, stop, metrics)?;
            let Some((pivot, pivot_machine, cutoff)) = ready.pop_pivot(stop, metrics)? else {
                return Ok(None);
            };
            let Some(selected) = ready.select(problem, pivot, pivot_machine, cutoff, rule, seed, step, stop, metrics)? else {
                return Ok(None);
            };
            let operation = selected.operation;
            if !ready.remove(problem, operation) {
                return Ok(None);
            }
            let machine = problem.machine(operation);
            machine_sequences[machine].push(operation);
            completion[operation] = selected.end;
            if !ready.refresh_machine(problem, machine, selected.end, stop, metrics)? {
                return Ok(None);
            }
            for &successor in problem.precedences.successors(operation) {
                checkpoint(stop)?;
                let Some(value) = indegrees[successor].checked_sub(1) else {
                    return Ok(None);
                };
                indegrees[successor] = value;
                if value == 0 {
                    let release =
                        problem.precedences.predecessors(successor).iter().map(|&predecessor| completion[predecessor]).max().unwrap_or(0);
                    if !ready.activate(problem, successor, release, stop, metrics)? {
                        return Ok(None);
                    }
                }
            }
        }

        Self::initialize(problem, machine_sequences, stop, metrics)
    }

    #[cfg(test)]
    pub(crate) fn giffler_thompson_reference(
        problem: &JobShopProblem,
        seed: u64,
        rule: DispatchRule,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ScheduleStateInterrupted> {
        let mut metrics = ScheduleStateMetrics::default();
        Self::giffler_thompson_reference_inner(problem, seed, rule, stop, &mut metrics)
    }

    /// Frozen scan-based constructor used only as an equivalence oracle for the
    /// bucketed implementation above.
    #[cfg(test)]
    fn giffler_thompson_reference_inner(
        problem: &JobShopProblem,
        seed: u64,
        rule: DispatchRule,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<Option<Self>, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let operation_count = problem.operation_count();
        let mut machine_sequences = vec![Vec::new(); problem.machine_count()];
        let mut machine_ready = vec![0i64; problem.machine_count()];
        let mut completion = vec![0i64; operation_count];
        let mut indegrees = (0..operation_count).map(|operation| problem.precedences.predecessors(operation).len()).collect::<Vec<_>>();
        let mut ready = Vec::with_capacity(operation_count);
        for (operation, &indegree) in indegrees.iter().enumerate() {
            checkpoint(stop)?;
            if indegree == 0 {
                ready.push(operation);
            }
        }
        let mut eligible = Vec::with_capacity(operation_count);

        for step in 0..operation_count {
            checkpoint(stop)?;
            eligible.clear();
            for &operation in &ready {
                checkpoint(stop)?;
                metrics.construction_candidates = metrics.construction_candidates.saturating_add(1);
                let release =
                    problem.precedences.predecessors(operation).iter().map(|&predecessor| completion[predecessor]).max().unwrap_or(0);
                let machine = problem.machine(operation);
                let start = release.max(machine_ready[machine]).max(problem.start_windows[operation].0);
                let Some(end) = start.checked_add(problem.duration(operation)) else {
                    return Ok(None);
                };
                if start > problem.start_windows[operation].1 || end > problem.horizons[operation] {
                    return Ok(None);
                }
                eligible.push((operation, start, end));
            }
            let Some(&(pivot, _, cutoff)) = eligible.iter().min_by_key(|&&(operation, _, end)| (end, operation)) else {
                return Ok(None);
            };
            let pivot_machine = problem.machine(pivot);
            let mut selected = None;
            for &(operation, start, end) in &eligible {
                checkpoint(stop)?;
                if problem.machine(operation) != pivot_machine || (operation != pivot && start >= cutoff) {
                    continue;
                }
                let candidate = DispatchCandidate { operation, start, end };
                if selected.is_none_or(|current| dispatch_better(problem, rule, seed, step, candidate, current)) {
                    selected = Some(candidate);
                }
            }
            let Some(selected) = selected else {
                return Ok(None);
            };
            let operation = selected.operation;
            let Some(ready_position) = ready.iter().position(|&candidate| candidate == operation) else {
                return Ok(None);
            };
            ready.swap_remove(ready_position);
            let machine = problem.machine(operation);
            machine_sequences[machine].push(operation);
            machine_ready[machine] = selected.end;
            completion[operation] = selected.end;
            for &successor in problem.precedences.successors(operation) {
                let Some(value) = indegrees[successor].checked_sub(1) else {
                    return Ok(None);
                };
                indegrees[successor] = value;
                if value == 0 {
                    ready.push(successor);
                }
            }
        }

        Self::initialize(problem, machine_sequences, stop, metrics)
    }

    pub(crate) fn from_machine_sequences(
        problem: &JobShopProblem,
        machine_sequences: Vec<Vec<usize>>,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ScheduleStateInterrupted> {
        let mut metrics = ScheduleStateMetrics::default();
        Self::initialize(problem, machine_sequences, stop, &mut metrics)
    }

    fn initialize(
        problem: &JobShopProblem,
        machine_sequences: Vec<Vec<usize>>,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<Option<Self>, ScheduleStateInterrupted> {
        Self::initialize_configured(problem, machine_sequences, CanonicalPathPolicy::Historical, true, stop, metrics)
    }

    fn initialize_configured(
        problem: &JobShopProblem,
        machine_sequences: Vec<Vec<usize>>,
        canonical_path_policy: CanonicalPathPolicy,
        collect_all_critical_blocks: bool,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<Option<Self>, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        if machine_sequences.len() != problem.machine_count() {
            return Ok(None);
        }
        let operation_count = problem.operation_count();
        let mut machine_predecessors = vec![NO_OPERATION; operation_count];
        let mut machine_successors = vec![NO_OPERATION; operation_count];
        let mut positions = vec![NO_OPERATION; operation_count];
        if !initialize_machine_graph(problem, &machine_sequences, &mut machine_predecessors, &mut machine_successors, &mut positions, stop)?
        {
            return Ok(None);
        }

        let mut workspace = EvaluationWorkspace::new(operation_count);
        metrics.reconstructions = metrics.reconstructions.saturating_add(1);
        metrics.full_evaluations = metrics.full_evaluations.saturating_add(1);
        match workspace.full_rebuild(problem, &machine_predecessors, &machine_successors, stop) {
            Ok(_) => {}
            Err(ReconstructionFailure::Interrupted) => return Err(ScheduleStateInterrupted),
            Err(_) => return Ok(None),
        }
        match workspace.build_analysis(
            problem,
            &machine_predecessors,
            &machine_successors,
            &positions,
            &machine_sequences,
            canonical_path_policy,
            collect_all_critical_blocks,
            stop,
        ) {
            Ok(()) => {}
            Err(ReconstructionFailure::Interrupted) => return Err(ScheduleStateInterrupted),
            Err(_) => return Ok(None),
        }
        workspace.observe_growths();
        metrics.workspace_growths = workspace.growths;
        let mut reconstruction = Reconstruction::empty(operation_count);
        reconstruction.commit(&workspace);
        metrics.critical_path_updates = metrics.critical_path_updates.saturating_add(1);
        Ok(Some(Self {
            problem: problem.clone(),
            machine_sequences,
            machine_predecessors,
            machine_successors,
            positions,
            canonical_path_policy,
            collect_all_critical_blocks,
            reconstruction,
            workspace,
            metrics: *metrics,
        }))
    }

    pub(crate) fn makespan(&self) -> i64 {
        self.reconstruction.makespan
    }

    pub(crate) fn operation_count(&self) -> usize {
        self.problem.operation_count()
    }

    pub(crate) fn start_window(&self, operation: usize) -> (i64, i64) {
        self.problem.start_window(operation)
    }

    pub(crate) fn horizon(&self, operation: usize) -> i64 {
        self.problem.horizon(operation)
    }

    pub(crate) fn machine_count(&self) -> usize {
        self.problem.machine_count()
    }

    pub(crate) fn problem(&self) -> &JobShopProblem {
        &self.problem
    }

    pub(crate) fn machine(&self, operation: usize) -> usize {
        self.problem.machine(operation)
    }

    pub(crate) fn raw_machine(&self, operation: usize) -> Option<usize> {
        self.problem.raw_machine(operation)
    }

    pub(crate) fn solution_machine(&self, operation: usize) -> Option<i64> {
        self.problem.solution_machine(operation)
    }

    pub(crate) fn solution_mode(&self, operation: usize) -> Option<Option<usize>> {
        self.problem.solution_mode(operation)
    }

    pub(crate) fn starts(&self) -> &[i64] {
        &self.reconstruction.starts
    }

    pub(crate) fn ends(&self) -> &[i64] {
        &self.reconstruction.ends
    }

    #[cfg(test)]
    pub(crate) fn test_tails(&self) -> &[i64] {
        &self.reconstruction.tails
    }

    pub(crate) fn duration(&self, operation: usize) -> i64 {
        self.problem.duration(operation)
    }

    pub(crate) fn job_predecessors(&self, operation: usize) -> &[usize] {
        self.problem.job_predecessors(operation)
    }

    pub(crate) fn job_successors(&self, operation: usize) -> &[usize] {
        self.problem.job_successors(operation)
    }

    pub(crate) fn latest_starts(&self) -> &[i64] {
        &self.reconstruction.latest_starts
    }

    pub(crate) fn machine_sequences(&self) -> &[Vec<usize>] {
        &self.machine_sequences
    }

    pub(crate) fn position(&self, operation: usize) -> Option<usize> {
        self.positions.get(operation).copied()
    }

    /// Recover owned machine-order buffers from a state that is no longer
    /// needed, so bounded repair can reuse their capacity on the next attempt.
    pub(crate) fn into_machine_sequences(self) -> Vec<Vec<usize>> {
        self.machine_sequences
    }

    pub(crate) fn topological_order(&self) -> &[usize] {
        &self.reconstruction.topological
    }

    pub(crate) fn critical_path(&self) -> &[usize] {
        &self.reconstruction.critical_path
    }

    pub(crate) fn canonical_path_policy(&self) -> CanonicalPathPolicy {
        self.canonical_path_policy
    }

    /// Stable identity of the currently published canonical path.
    pub(crate) fn canonical_path_fingerprint(&self) -> u64 {
        let mut fingerprint = mix64(u64::try_from(self.reconstruction.critical_path.len()).unwrap_or(u64::MAX));
        for (index, &operation) in self.reconstruction.critical_path.iter().enumerate() {
            fingerprint = mix64(
                fingerprint ^ u64::try_from(index).unwrap_or(u64::MAX).rotate_left(17) ^ u64::try_from(operation).unwrap_or(u64::MAX),
            );
        }
        fingerprint
    }

    /// Re-select the canonical critical path under a deterministic portfolio
    /// policy. Trial analysis remains private until every checkpoint succeeds.
    pub(crate) fn set_canonical_path_policy(
        &mut self,
        policy: CanonicalPathPolicy,
        stop: &AtomicBool,
    ) -> Result<bool, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        if policy == self.canonical_path_policy {
            return Ok(true);
        }
        let terminal = match select_canonical_terminal(
            self.reconstruction.topological.iter().copied(),
            &self.machine_predecessors,
            &self.reconstruction.starts,
            &self.reconstruction.ends,
            self.reconstruction.makespan,
            policy,
            stop,
        ) {
            Ok(terminal) => terminal,
            Err(ReconstructionFailure::Interrupted) => return Err(ScheduleStateInterrupted),
            Err(_) => return Ok(false),
        };
        if terminal.is_none() && self.problem.operation_count() != 0 {
            return Ok(false);
        }
        match build_canonical_path_from_terminal(
            &mut self.workspace.trial_critical_path,
            &mut self.workspace.trial_canonical_critical_blocks,
            &self.problem,
            &self.machine_predecessors,
            &self.positions,
            &self.reconstruction.starts,
            &self.reconstruction.ends,
            policy,
            terminal,
            true,
            &mut self.workspace.analysis_stop_after_path_operations,
            stop,
        ) {
            Ok(()) => {}
            Err(ReconstructionFailure::Interrupted) => return Err(ScheduleStateInterrupted),
            Err(_) => return Ok(false),
        }

        self.canonical_path_policy = policy;
        self.reconstruction.critical_path.clear();
        self.reconstruction.critical_path.extend_from_slice(&self.workspace.trial_critical_path);
        self.reconstruction.canonical_critical_blocks.clear();
        self.reconstruction.canonical_critical_blocks.extend_from_slice(&self.workspace.trial_canonical_critical_blocks);
        self.metrics.critical_path_updates = self.metrics.critical_path_updates.saturating_add(1);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        Ok(true)
    }

    pub(crate) fn critical_blocks(&self) -> &[CriticalBlock] {
        &self.reconstruction.critical_blocks
    }

    /// Critical blocks from the single deterministic path used by the
    /// historical persistent JSSP search.
    pub(crate) fn canonical_critical_blocks(&self) -> &[CriticalBlock] {
        &self.reconstruction.canonical_critical_blocks
    }

    pub(crate) fn metrics(&self) -> ScheduleStateMetrics {
        let mut metrics = self.metrics;
        metrics.workspace_growths = self.workspace.growths;
        metrics
    }

    /// Return the counters accumulated since the previous take and reset them.
    /// Workspace capacity observations remain anchored at their current values,
    /// so a later take reports only genuine subsequent allocation growth.
    pub(crate) fn take_metrics(&mut self) -> ScheduleStateMetrics {
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        let metrics = self.metrics;
        self.metrics = ScheduleStateMetrics::default();
        self.workspace.growths = 0;
        metrics
    }

    pub(crate) fn workspace_capacities(&self) -> ScheduleWorkspaceCapacities {
        self.workspace.capacities()
    }

    pub(crate) fn fast_n5_workspace_lower_bound_bytes(&self) -> usize {
        self.workspace
            .fast_patched_operations
            .capacity()
            .saturating_mul(size_of::<usize>())
            .saturating_add(self.workspace.fast_patched_starts.capacity().saturating_mul(size_of::<i64>()))
            .saturating_add(self.workspace.fast_patched_ends.capacity().saturating_mul(size_of::<i64>()))
            .saturating_add(self.workspace.fast_patched_tails.capacity().saturating_mul(size_of::<i64>()))
            .saturating_add(self.workspace.fast_patch_marks.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.workspace.fast_queue_marks.len().saturating_mul(size_of::<u32>()))
    }

    #[cfg(test)]
    pub(crate) fn test_configure_strict_n5_work_cap(&mut self, pop_cap: usize, stop_on_work_cap: bool) {
        self.workspace.fast_forward_value_change_pop_cap = Some(pop_cap);
        self.workspace.fast_reverse_value_change_pop_cap = Some(pop_cap);
        self.workspace.fast_stop_on_work_cap = stop_on_work_cap;
        self.workspace.fast_stop_during_reverse_recovery = false;
    }

    #[cfg(test)]
    pub(crate) fn test_configure_strict_n5_phase_work_caps(
        &mut self,
        forward_pop_cap: usize,
        reverse_pop_cap: usize,
        stop_on_work_cap: bool,
        stop_during_reverse_recovery: bool,
    ) {
        self.workspace.fast_forward_value_change_pop_cap = Some(forward_pop_cap);
        self.workspace.fast_reverse_value_change_pop_cap = Some(reverse_pop_cap);
        self.workspace.fast_stop_on_work_cap = stop_on_work_cap;
        self.workspace.fast_stop_during_reverse_recovery = stop_during_reverse_recovery;
    }

    #[cfg(test)]
    pub(crate) fn test_strict_n5_streaming_pop_cap(operation_count: usize) -> usize {
        strict_n5_streaming_pop_cap(operation_count)
    }

    #[cfg(test)]
    pub(crate) fn test_last_strict_n5_work_cap_was_reverse(&self) -> bool {
        matches!(self.workspace.fast_last_work_cap_phase, Some(FastN5PropagationPhase::Reverse))
    }

    #[cfg(test)]
    pub(crate) fn test_stop_batch_after_applied_moves(&mut self, applied_moves: usize) {
        self.workspace.batch_stop_after_applied_moves = Some(applied_moves);
    }

    #[cfg(test)]
    pub(crate) fn test_stop_analysis_after_path_operations(&mut self, operations: usize) {
        self.workspace.analysis_stop_after_path_operations = Some(operations);
    }

    #[cfg(test)]
    pub(crate) fn test_wrap_fast_dirty_epoch(&mut self) -> bool {
        self.workspace.dirty_epoch = u32::MAX;
        self.workspace.dirty_marks.fill(u32::MAX);
        self.workspace.fast_queue_marks.fill(u32::MAX);
        let epoch = self.workspace.next_dirty_epoch();
        epoch == 1
            && self.workspace.dirty_marks.iter().all(|&mark| mark == 0)
            && self.workspace.fast_queue_marks.iter().all(|&mark| mark == 0)
    }

    /// Rebuild a replacement state against this state's validated problem.
    /// The accepted state is never mutated; callers replace it only after a
    /// successful reconstruction.
    pub(crate) fn rebuilt_from_machine_sequences(
        &self,
        machine_sequences: Vec<Vec<usize>>,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ScheduleStateInterrupted> {
        self.rebuilt_from_machine_sequences_with_policy(machine_sequences, self.canonical_path_policy, stop)
    }

    /// Rebuild directly under a requested canonical lane while inheriting the
    /// current all-critical block materialization policy.
    pub(crate) fn rebuilt_from_machine_sequences_with_policy(
        &self,
        machine_sequences: Vec<Vec<usize>>,
        policy: CanonicalPathPolicy,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ScheduleStateInterrupted> {
        let mut metrics = ScheduleStateMetrics::default();
        Self::initialize_configured(&self.problem, machine_sequences, policy, self.collect_all_critical_blocks, stop, &mut metrics)
    }

    /// Stop materializing the broader zero-slack block set on future commits.
    /// Baseline islands retain only the canonical historical path blocks.
    pub(crate) fn retain_canonical_critical_blocks_only(&mut self) {
        self.collect_all_critical_blocks = false;
        self.reconstruction.critical_blocks.clear();
        self.workspace.trial_critical_blocks.clear();
    }

    /// Return the operation-identity machine arcs changed by a valid move.
    pub(crate) fn move_arcs(&self, movement: ScheduleMove) -> Option<ScheduleMoveArcs> {
        if !move_is_valid(&self.machine_sequences, movement) {
            return None;
        }
        let machine = movement.machine();
        let sequence = &self.machine_sequences[machine];
        let (old_indices, new_indices) = changed_arc_indices(movement);
        let mut old_arcs = [None; 3];
        let mut new_arcs = [None; 3];
        for index in old_indices.into_iter().flatten() {
            if let Some(arc) = sequence_arc(sequence, machine, index) {
                push_unique_arc(&mut old_arcs, arc);
            }
        }
        for index in new_indices.into_iter().flatten() {
            if let Some(arc) = moved_sequence_arc(sequence, movement, machine, index) {
                push_unique_arc(&mut new_arcs, arc);
            }
        }
        Some(ScheduleMoveArcs { removed: arc_difference(old_arcs, &new_arcs), added: arc_difference(new_arcs, &old_arcs) })
    }

    /// Compute advisory move guidance without patching the graph or changing
    /// any accepted date. Work is bounded by the six changed arc slots.
    pub(crate) fn score_move_head_tail(
        &self,
        movement: ScheduleMove,
        stop: &AtomicBool,
    ) -> Result<Option<HeadTailMoveScore>, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let Some(arcs) = self.move_arcs(movement) else {
            return Ok(None);
        };
        Ok(Some(self.score_move_head_tail_from_arcs(arcs, stop)?))
    }

    fn score_move_head_tail_from_arcs(
        &self,
        arcs: ScheduleMoveArcs,
        stop: &AtomicBool,
    ) -> Result<HeadTailMoveScore, ScheduleStateInterrupted> {
        let mut max_added_arc_path = i128::MIN;
        let mut total_added_arc_path = 0i128;
        let mut critical_arcs_removed = 0u8;
        let mut tight_arcs_added = 0u8;
        for arc in arcs.removed.into_iter().flatten() {
            checkpoint(stop)?;
            critical_arcs_removed = critical_arcs_removed.saturating_add(u8::from(self.is_critical_machine_arc(arc)));
        }
        for arc in arcs.added.into_iter().flatten() {
            checkpoint(stop)?;
            let path = i128::from(self.reconstruction.ends[arc.before]) + i128::from(self.reconstruction.tails[arc.after]);
            max_added_arc_path = max_added_arc_path.max(path);
            total_added_arc_path += path;
            tight_arcs_added =
                tight_arcs_added.saturating_add(u8::from(self.reconstruction.ends[arc.before] == self.reconstruction.starts[arc.after]));
        }
        if max_added_arc_path == i128::MIN {
            max_added_arc_path = i128::from(self.makespan());
        }
        Ok(HeadTailMoveScore { max_added_arc_path, total_added_arc_path, critical_arcs_removed, tight_arcs_added })
    }

    /// Evaluate `... left, a, b, right ... -> ... left, b, a, right ...`
    /// with Taillard's adjacent-swap head-tail equations. This specialized
    /// score is available only for the strict classical N5 fast-path shape.
    pub(crate) fn score_strict_n5_taillard(
        &self,
        movement: ScheduleMove,
        stop: &AtomicBool,
    ) -> Result<Option<StrictN5TaillardScore>, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        if !self.supports_strict_n5_fast_path() || !self.is_current_strict_n5_move(movement) {
            return Ok(None);
        }
        let Some(arcs) = self.move_arcs(movement) else {
            return Ok(None);
        };
        Ok(Some(self.score_strict_n5_taillard_unchecked(movement, arcs, stop)?))
    }

    fn score_strict_n5_taillard_unchecked(
        &self,
        movement: ScheduleMove,
        arcs: ScheduleMoveArcs,
        stop: &AtomicBool,
    ) -> Result<StrictN5TaillardScore, ScheduleStateInterrupted> {
        let ScheduleMove::AdjacentSwap { machine, first_position } = movement else {
            unreachable!("the strict N5 visitor emits only adjacent swaps");
        };
        let sequence = &self.machine_sequences[machine];
        let a = sequence[first_position];
        let b = sequence[first_position + 1];

        let job_release = |operation: usize| {
            self.problem
                .job_predecessors(operation)
                .iter()
                .map(|&predecessor| i128::from(self.reconstruction.ends[predecessor]))
                .max()
                .unwrap_or(0)
        };
        let job_tail = |operation: usize| {
            self.problem
                .job_successors(operation)
                .iter()
                .map(|&successor| i128::from(self.reconstruction.tails[successor]))
                .max()
                .unwrap_or(0)
        };
        let left_completion = first_position
            .checked_sub(1)
            .and_then(|position| sequence.get(position))
            .map_or(0, |&left| i128::from(self.reconstruction.ends[left]));
        let right_tail = sequence.get(first_position.saturating_add(2)).map_or(0, |&right| i128::from(self.reconstruction.tails[right]));
        let duration_a = i128::from(self.problem.duration(a));
        let duration_b = i128::from(self.problem.duration(b));

        let release_b = job_release(b).max(left_completion);
        let release_a = job_release(a).max(release_b.saturating_add(duration_b));
        let tail_a = duration_a.saturating_add(job_tail(a).max(right_tail));
        let tail_b = duration_b.saturating_add(job_tail(b).max(tail_a));
        let estimated_makespan = release_b.saturating_add(tail_b).max(release_a.saturating_add(tail_a));
        let secondary = self.score_move_head_tail_from_arcs(arcs, stop)?;
        Ok(StrictN5TaillardScore { estimated_makespan, secondary })
    }

    /// Generate the union of critical neighborhoods and rank it by advisory
    /// head-tail guidance. Equal scores use [`ScheduleMove`] identity, making
    /// the output deterministic. The returned score never replaces an exact
    /// probe.
    pub(crate) fn fill_scored_critical_moves(
        &self,
        neighborhoods: &[CriticalNeighborhood],
        movements: &mut Vec<ScoredScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleStateInterrupted> {
        checkpoint(stop)?;
        movements.clear();
        let capacity_hint = Self::critical_move_capacity_hint(neighborhoods, &self.reconstruction.critical_blocks).min(4_096);
        if movements.capacity() < capacity_hint {
            movements.reserve(capacity_hint);
        }
        Self::visit_critical_move_union(neighborhoods, &self.reconstruction.critical_blocks, stop, |movement| {
            let score = self.score_move_head_tail(movement, stop)?.expect("critical-block movements are structurally valid");
            movements.push(ScoredScheduleMove { movement, score });
            Ok(())
        })?;
        movements.sort_unstable_by_key(|candidate| (candidate.score.ranking_key(), candidate.movement));
        movements.dedup_by_key(|candidate| candidate.movement);
        Ok(())
    }

    /// Rank moves from only the deterministic critical path used by the
    /// historical schedule-search kernel. Scores remain advisory: every
    /// shortlisted move must still pass an exact probe and oracle commit.
    pub(crate) fn fill_scored_canonical_critical_moves(
        &self,
        neighborhood: CriticalNeighborhood,
        movements: &mut Vec<ScoredScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleStateInterrupted> {
        movements.clear();
        self.visit_scored_canonical_critical_moves(neighborhood, stop, |candidate, _| {
            movements.push(candidate);
            Ok(())
        })?;
        movements.sort_unstable_by_key(|candidate| (candidate.score.ranking_key(), candidate.movement));
        movements.dedup_by_key(|candidate| candidate.movement);
        Ok(())
    }

    /// Stream every scored move from the canonical critical path without
    /// materializing the neighborhood. This lets large-instance callers keep
    /// a fixed-size top-k shortlist while still accounting for all generated
    /// N5 moves.
    pub(crate) fn visit_scored_canonical_critical_moves(
        &self,
        neighborhood: CriticalNeighborhood,
        stop: &AtomicBool,
        mut visit: impl FnMut(ScoredScheduleMove, ScheduleMoveArcs) -> Result<(), ScheduleStateInterrupted>,
    ) -> Result<usize, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let mut generated = 0usize;
        let emit = |movement| {
            let score = self.score_move_head_tail(movement, stop)?.expect("canonical critical-block movements are structurally valid");
            let arcs = self.move_arcs(movement).expect("canonical critical-block movements have changed arcs");
            generated = generated.saturating_add(1);
            visit(ScoredScheduleMove { movement, score }, arcs)
        };
        if neighborhood != CriticalNeighborhood::N5 {
            Self::visit_critical_move_union(&[neighborhood], &self.reconstruction.canonical_critical_blocks, stop, emit)?;
            return Ok(generated);
        }

        Self::visit_strict_n5_moves(&self.reconstruction.canonical_critical_blocks, stop, emit)?;
        Ok(generated)
    }

    /// Stream the strict canonical N5 neighborhood without scores or heap
    /// materialization. This is the only neighborhood accepted by the
    /// default-off fast micro-kernel.
    pub(crate) fn visit_strict_n5_canonical_moves(
        &self,
        stop: &AtomicBool,
        mut visit: impl FnMut(ScheduleMove, ScheduleMoveArcs) -> Result<(), ScheduleStateInterrupted>,
    ) -> Result<usize, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let mut generated = 0usize;
        Self::visit_strict_n5_moves(&self.reconstruction.canonical_critical_blocks, stop, |movement| {
            let arcs = self.move_arcs(movement).expect("strict canonical N5 movements are structurally valid");
            generated = generated.saturating_add(1);
            visit(movement, arcs)
        })?;
        Ok(generated)
    }

    /// Stream strict N5 moves with Taillard's adjacent-swap estimate. No
    /// neighborhood or shortlist is materialized.
    pub(crate) fn visit_taillard_scored_strict_n5_canonical_moves(
        &self,
        stop: &AtomicBool,
        mut visit: impl FnMut(ScheduleMove, ScheduleMoveArcs, StrictN5TaillardScore) -> Result<(), ScheduleStateInterrupted>,
    ) -> Result<usize, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        if !self.supports_strict_n5_fast_path() {
            return Ok(0);
        }
        let mut generated = 0usize;
        Self::visit_strict_n5_moves(&self.reconstruction.canonical_critical_blocks, stop, |movement| {
            let arcs = self.move_arcs(movement).expect("strict canonical N5 movements are structurally valid");
            let score = self.score_strict_n5_taillard_unchecked(movement, arcs, stop)?;
            generated = generated.saturating_add(1);
            visit(movement, arcs, score)
        })?;
        Ok(generated)
    }

    pub(crate) fn supports_strict_n5_fast_path(&self) -> bool {
        self.problem.supports_strict_n5_fast_path()
    }

    fn is_current_strict_n5_move(&self, movement: ScheduleMove) -> bool {
        let ScheduleMove::AdjacentSwap { machine, first_position } = movement else {
            return false;
        };
        self.reconstruction.canonical_critical_blocks.iter().enumerate().any(|(block_index, block)| {
            (block_index > 0 && block.machine == machine && block.first_position == first_position)
                || (block_index + 1 < self.reconstruction.canonical_critical_blocks.len()
                    && block.machine == machine
                    && block.last_position > block.first_position
                    && block.last_position - 1 == first_position)
        })
    }

    fn visit_strict_n5_moves(
        blocks: &[CriticalBlock],
        stop: &AtomicBool,
        mut visit: impl FnMut(ScheduleMove) -> Result<(), ScheduleStateInterrupted>,
    ) -> Result<(), ScheduleStateInterrupted> {
        for (block_index, block) in blocks.iter().enumerate() {
            checkpoint(stop)?;
            let mut first_move = None;
            if block_index > 0 {
                let movement = ScheduleMove::AdjacentSwap { machine: block.machine, first_position: block.first_position };
                visit(movement)?;
                first_move = Some(movement);
            }
            if block_index + 1 < blocks.len() && block.last_position > block.first_position {
                let movement = ScheduleMove::AdjacentSwap { machine: block.machine, first_position: block.last_position - 1 };
                if first_move != Some(movement) {
                    visit(movement)?;
                }
            }
        }
        Ok(())
    }

    /// Certify that every machine arc added by a move follows the accepted
    /// topological order. Components sharing this certificate can be composed
    /// into one batch without creating a cycle. `false` means unknown.
    pub(crate) fn certifies_move_topological_acyclicity(&self, movement: ScheduleMove) -> bool {
        self.move_arcs(movement).is_some_and(|arcs| {
            arcs.added
                .into_iter()
                .flatten()
                .all(|arc| self.reconstruction.topological_rank[arc.before] < self.reconstruction.topological_rank[arc.after])
        })
    }

    /// Certify that one insertion preserves acyclicity without mutating the
    /// accepted graph.
    ///
    /// The first certificate keeps the existing accepted topological order.
    /// The second applies Propositions 1 and 2 of the N8 neighborhood to the
    /// operation immediately left or right of the moved operation. These N8
    /// propositions are sound for one move but are not composable across a
    /// batch. Batch callers must use
    /// [`Self::certifies_move_topological_acyclicity`] for every component.
    /// The paper's non-strict inequalities apply only when the precedence
    /// graph is a union of job chains, no chain revisits a machine, and every
    /// duration is positive. With zero duration, equality is insufficient.
    /// Non-classical precedence graphs use only the topological certificate.
    /// `false` means unknown, never cyclic.
    pub(crate) fn certifies_insert_acyclicity(&self, movement: ScheduleMove, stop: &AtomicBool) -> Result<bool, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let ScheduleMove::Insert { machine, from, to } = movement else {
            return Ok(false);
        };
        if !move_is_valid(&self.machine_sequences, movement) {
            return Ok(false);
        }
        if self.certifies_move_topological_acyclicity(movement) {
            return Ok(true);
        }
        if !self.problem.n8_classical_chain_model {
            return Ok(false);
        }

        let sequence = &self.machine_sequences[machine];
        let moved = sequence[from];
        let anchor = sequence[to];
        if from < to {
            let anchor_tail = self.reconstruction.tails[anchor];
            for &successor in self.problem.precedences.successors(moved) {
                checkpoint(stop)?;
                let tail_after_successor = self.reconstruction.tails[successor]
                    .checked_sub(self.problem.duration(successor))
                    .expect("non-negative durations make an operation tail at least its duration");
                if anchor_tail < tail_after_successor || (anchor_tail == tail_after_successor && !self.problem.all_durations_positive) {
                    return Ok(false);
                }
            }
        } else {
            let anchor_end = self.reconstruction.ends[anchor];
            for &predecessor in self.problem.precedences.predecessors(moved) {
                checkpoint(stop)?;
                let predecessor_head = self.reconstruction.starts[predecessor];
                if anchor_end < predecessor_head || (anchor_end == predecessor_head && !self.problem.all_durations_positive) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Refine one structured move with local head-tail propagation on its
    /// changed machine span. This routine never patches the graph and its
    /// result is advisory even when acyclicity is certified.
    pub(crate) fn estimate_move_local(
        &mut self,
        movement: ScheduleMove,
        stop: &AtomicBool,
    ) -> Result<Option<LocalMoveEstimate>, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        if !move_is_valid(&self.machine_sequences, movement) {
            return Ok(None);
        }
        let arcs = self.move_arcs(movement).expect("a validated move has changed arcs");
        let acyclicity_certified = arcs
            .added
            .into_iter()
            .flatten()
            .all(|arc| self.reconstruction.topological_rank[arc.before] < self.reconstruction.topological_rank[arc.after]);
        let machine = movement.machine();
        let sequence = &self.machine_sequences[machine];
        let (first_position, last_position) = movement.position_bounds();
        let external_predecessor = first_position.checked_sub(1).and_then(|position| moved_operation_at(sequence, movement, position));
        let external_successor = last_position.checked_add(1).and_then(|position| moved_operation_at(sequence, movement, position));

        self.workspace.local_span_operations.clear();
        self.workspace.local_heads.clear();
        self.workspace.local_tails.clear();
        for position in first_position..=last_position {
            checkpoint(stop)?;
            self.workspace
                .local_span_operations
                .push(moved_operation_at(sequence, movement, position).expect("a valid move preserves the sequence length"));
        }

        for index in 0..self.workspace.local_span_operations.len() {
            checkpoint(stop)?;
            let operation = self.workspace.local_span_operations[index];
            let mut head = i128::from(self.problem.start_windows[operation].0);
            for &predecessor in self.problem.precedences.predecessors(operation) {
                head = head.max(i128::from(self.reconstruction.ends[predecessor]));
            }
            let machine_head = if index == 0 {
                external_predecessor.map_or(0, |predecessor| i128::from(self.reconstruction.ends[predecessor]))
            } else {
                let predecessor = self.workspace.local_span_operations[index - 1];
                self.workspace.local_heads[index - 1] + i128::from(self.problem.duration(predecessor))
            };
            self.workspace.local_heads.push(head.max(machine_head));
        }

        self.workspace.local_tails.resize(self.workspace.local_span_operations.len(), 0);
        for index in (0..self.workspace.local_span_operations.len()).rev() {
            checkpoint(stop)?;
            let operation = self.workspace.local_span_operations[index];
            let mut job_tail = 0i128;
            for &successor in self.problem.precedences.successors(operation) {
                job_tail = job_tail.max(i128::from(self.reconstruction.tails[successor]));
            }
            let machine_tail = if index + 1 == self.workspace.local_span_operations.len() {
                external_successor.map_or(0, |successor| i128::from(self.reconstruction.tails[successor]))
            } else {
                self.workspace.local_tails[index + 1]
            };
            self.workspace.local_tails[index] = i128::from(self.problem.duration(operation)) + job_tail.max(machine_tail);
        }
        let estimated_makespan = self
            .workspace
            .local_heads
            .iter()
            .zip(&self.workspace.local_tails)
            .map(|(head, tail)| head + tail)
            .max()
            .unwrap_or(i128::from(self.makespan()));
        self.metrics.local_move_estimates = self.metrics.local_move_estimates.saturating_add(1);
        if acyclicity_certified {
            self.metrics.local_move_certified = self.metrics.local_move_certified.saturating_add(1);
        } else {
            self.metrics.local_move_unknown = self.metrics.local_move_unknown.saturating_add(1);
        }
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        Ok(Some(LocalMoveEstimate {
            estimated_makespan,
            acyclicity_certified,
            span_operations: self.workspace.local_span_operations.len(),
        }))
    }

    fn is_critical_machine_arc(&self, arc: MachineArc) -> bool {
        self.machine_successors[arc.before] == arc.after
            && self.reconstruction.ends[arc.before] == self.reconstruction.starts[arc.after]
            && self.reconstruction.starts[arc.before]
                .checked_add(self.reconstruction.tails[arc.before])
                .is_some_and(|critical_end| critical_end == self.reconstruction.makespan)
            && self.reconstruction.starts[arc.after]
                .checked_add(self.reconstruction.tails[arc.after])
                .is_some_and(|critical_end| critical_end == self.reconstruction.makespan)
    }

    pub(crate) fn critical_moves(
        &self,
        neighborhood: CriticalNeighborhood,
        stop: &AtomicBool,
    ) -> Result<Vec<ScheduleMove>, ScheduleStateInterrupted> {
        let mut movements = Vec::new();
        self.fill_critical_moves(neighborhood, &mut movements, stop)?;
        Ok(movements)
    }

    /// Fill a caller-owned buffer so neighborhood scans reuse their allocation.
    /// The result is sorted and duplicate-free even when critical blocks touch.
    pub(crate) fn fill_critical_moves(
        &self,
        neighborhood: CriticalNeighborhood,
        movements: &mut Vec<ScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleStateInterrupted> {
        self.fill_critical_move_union(&[neighborhood], movements, stop)
    }

    /// Fill moves from only the historical deterministic critical path.
    ///
    /// This intentionally preserves the old sort/dedup order. Baseline
    /// portfolio islands rotate this exact vector before probing it.
    pub(crate) fn fill_canonical_critical_moves(
        &self,
        neighborhood: CriticalNeighborhood,
        movements: &mut Vec<ScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleStateInterrupted> {
        self.fill_critical_moves_from_blocks(&[neighborhood], &self.reconstruction.canonical_critical_blocks, movements, stop)
    }

    /// Build a duplicate-free union when several nested neighborhoods are
    /// scanned at the same incumbent.
    pub(crate) fn fill_critical_move_union(
        &self,
        neighborhoods: &[CriticalNeighborhood],
        movements: &mut Vec<ScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleStateInterrupted> {
        self.fill_critical_moves_from_blocks(neighborhoods, &self.reconstruction.critical_blocks, movements, stop)
    }

    fn fill_critical_moves_from_blocks(
        &self,
        neighborhoods: &[CriticalNeighborhood],
        blocks: &[CriticalBlock],
        movements: &mut Vec<ScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleStateInterrupted> {
        checkpoint(stop)?;
        movements.clear();
        let capacity_hint = Self::critical_move_capacity_hint(neighborhoods, blocks).min(4_096);
        if movements.capacity() < capacity_hint {
            movements.reserve(capacity_hint);
        }
        Self::visit_critical_move_union(neighborhoods, blocks, stop, |movement| {
            movements.push(movement);
            Ok(())
        })?;
        movements.sort_unstable();
        movements.dedup();
        Ok(())
    }

    fn critical_move_capacity_hint(neighborhoods: &[CriticalNeighborhood], blocks: &[CriticalBlock]) -> usize {
        neighborhoods.iter().fold(0usize, |total, neighborhood| {
            blocks.iter().fold(total, |total, block| {
                let block_moves = match neighborhood {
                    CriticalNeighborhood::N1 => block.len().saturating_sub(1),
                    CriticalNeighborhood::N5 => usize::from(block.len() >= 2) + usize::from(block.len() >= 3),
                    CriticalNeighborhood::N6 => {
                        let n5 = usize::from(block.len() >= 2) + usize::from(block.len() >= 3);
                        n5.saturating_add(block.len().saturating_sub(3).saturating_mul(2))
                    }
                };
                total.saturating_add(block_moves)
            })
        })
    }

    fn visit_critical_move_union(
        neighborhoods: &[CriticalNeighborhood],
        blocks: &[CriticalBlock],
        stop: &AtomicBool,
        mut visit: impl FnMut(ScheduleMove) -> Result<(), ScheduleStateInterrupted>,
    ) -> Result<(), ScheduleStateInterrupted> {
        for &neighborhood in neighborhoods {
            for &block in blocks {
                checkpoint(stop)?;
                match neighborhood {
                    CriticalNeighborhood::N1 => {
                        for first_position in block.first_position..block.last_position {
                            checkpoint(stop)?;
                            visit(ScheduleMove::AdjacentSwap { machine: block.machine, first_position })?;
                        }
                    }
                    CriticalNeighborhood::N5 | CriticalNeighborhood::N6 => {
                        visit(ScheduleMove::AdjacentSwap { machine: block.machine, first_position: block.first_position })?;
                        if block.last_position > block.first_position + 1 {
                            visit(ScheduleMove::AdjacentSwap { machine: block.machine, first_position: block.last_position - 1 })?;
                        }
                        if neighborhood == CriticalNeighborhood::N6 && block.len() >= 3 {
                            for from in (block.first_position + 1)..block.last_position {
                                checkpoint(stop)?;
                                if from > block.first_position + 1 {
                                    visit(ScheduleMove::Insert { machine: block.machine, from, to: block.first_position })?;
                                }
                                if from + 1 < block.last_position {
                                    visit(ScheduleMove::Insert { machine: block.machine, from, to: block.last_position })?;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply one strict canonical N5 swap with bounded value-change
    /// propagation. A wide value-change cone switches to one exact forward and
    /// reverse Kahn pass. Unsupported or semantically invalid cases fall back
    /// to the existing complete transactional path after exact restoration.
    pub(crate) fn consider_strict_n5_fast(
        &mut self,
        movement: ScheduleMove,
        acceptance: MinimizingMoveAcceptance,
        stop: &AtomicBool,
    ) -> Result<FastN5Outcome, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        if !self.supports_strict_n5_fast_path() || !self.is_current_strict_n5_move(movement) {
            let outcome = self.consider_move(movement, acceptance, stop)?;
            return Ok(FastN5Outcome {
                outcome,
                used_fast_path: false,
                fell_back: true,
                fallback_reason: Some(FastN5FallbackReason::Unsupported),
                used_topological_recovery: false,
                forward_date_changes: 0,
                reverse_tail_changes: 0,
                queue_pops: 0,
            });
        }

        let arcs = self.move_arcs(movement).expect("a strict N5 movement has changed arcs");
        self.metrics.moves_considered = self.metrics.moves_considered.saturating_add(1);
        self.metrics.delta_evaluations = self.metrics.delta_evaluations.saturating_add(1);
        #[cfg(test)]
        {
            self.workspace.fast_last_work_cap_phase = None;
        }
        self.workspace.begin_fast_patch();
        self.apply_graph_patch(movement);
        if stop.load(Ordering::Acquire) {
            self.rollback_fast_dates();
            self.rollback_graph_patch(movement);
            return Err(ScheduleStateInterrupted);
        }

        let previous = self.makespan();
        let propagated = self.evaluate_strict_n5_value_changes(arcs, stop);
        let mut used_topological_recovery = false;
        let (candidate, forward_date_changes, reverse_tail_changes, queue_pops) = match propagated {
            Ok(result) => result,
            Err(FastN5PropagationFailure { kind: FastN5PropagationFailureKind::Interrupted, .. }) => {
                self.rollback_fast_dates();
                self.rollback_graph_patch(movement);
                return Err(ScheduleStateInterrupted);
            }
            Err(
                streaming @ FastN5PropagationFailure {
                    kind: FastN5PropagationFailureKind::WorkCap(FastN5PropagationPhase::Forward), ..
                },
            ) => {
                #[cfg(test)]
                {
                    self.workspace.fast_last_work_cap_phase = Some(FastN5PropagationPhase::Forward);
                }
                self.rollback_fast_dates();
                self.workspace.begin_fast_patch();
                used_topological_recovery = true;
                #[cfg(test)]
                if self.workspace.fast_stop_on_work_cap {
                    stop.store(true, Ordering::Release);
                }
                match self.evaluate_strict_n5_topologically(arcs, streaming, stop) {
                    Ok(result) => result,
                    Err(FastN5PropagationFailure { kind: FastN5PropagationFailureKind::Interrupted, .. }) => {
                        self.rollback_fast_dates();
                        self.rollback_graph_patch(movement);
                        return Err(ScheduleStateInterrupted);
                    }
                    Err(failure) => {
                        self.rollback_fast_dates();
                        self.rollback_graph_patch(movement);
                        let outcome = self.consider_move_inner(movement, acceptance, stop, false)?;
                        let fallback_reason = match failure.kind {
                            FastN5PropagationFailureKind::WorkCap(_) => FastN5FallbackReason::WorkCap,
                            FastN5PropagationFailureKind::Reconstruction(failure) => Self::fast_n5_fallback_reason(failure),
                            FastN5PropagationFailureKind::Interrupted => unreachable!("interruption handled above"),
                        };
                        return Ok(FastN5Outcome {
                            outcome,
                            used_fast_path: false,
                            fell_back: true,
                            fallback_reason: Some(fallback_reason),
                            used_topological_recovery: true,
                            forward_date_changes: failure.forward_date_changes,
                            reverse_tail_changes: failure.reverse_tail_changes,
                            queue_pops: failure.queue_pops,
                        });
                    }
                }
            }
            Err(
                streaming @ FastN5PropagationFailure {
                    kind: FastN5PropagationFailureKind::WorkCap(FastN5PropagationPhase::Reverse), ..
                },
            ) => {
                #[cfg(test)]
                {
                    self.workspace.fast_last_work_cap_phase = Some(FastN5PropagationPhase::Reverse);
                }
                self.rollback_fast_tails();
                used_topological_recovery = true;
                #[cfg(test)]
                if self.workspace.fast_stop_on_work_cap {
                    stop.store(true, Ordering::Release);
                }
                match self.evaluate_strict_n5_reverse_topologically(
                    arcs,
                    streaming.forward_date_changes,
                    streaming.reverse_tail_changes,
                    usize::try_from(streaming.queue_pops).unwrap_or(usize::MAX),
                    stop,
                ) {
                    Ok(result) => result,
                    Err(FastN5PropagationFailure { kind: FastN5PropagationFailureKind::Interrupted, .. }) => {
                        self.rollback_fast_dates();
                        self.rollback_graph_patch(movement);
                        return Err(ScheduleStateInterrupted);
                    }
                    Err(failure) => {
                        self.rollback_fast_dates();
                        self.rollback_graph_patch(movement);
                        let outcome = self.consider_move_inner(movement, acceptance, stop, false)?;
                        let fallback_reason = match failure.kind {
                            FastN5PropagationFailureKind::WorkCap(_) => FastN5FallbackReason::WorkCap,
                            FastN5PropagationFailureKind::Reconstruction(failure) => Self::fast_n5_fallback_reason(failure),
                            FastN5PropagationFailureKind::Interrupted => unreachable!("interruption handled above"),
                        };
                        return Ok(FastN5Outcome {
                            outcome,
                            used_fast_path: false,
                            fell_back: true,
                            fallback_reason: Some(fallback_reason),
                            used_topological_recovery: true,
                            forward_date_changes: failure.forward_date_changes,
                            reverse_tail_changes: failure.reverse_tail_changes,
                            queue_pops: failure.queue_pops,
                        });
                    }
                }
            }
            Err(FastN5PropagationFailure {
                kind: FastN5PropagationFailureKind::Reconstruction(failure),
                forward_date_changes,
                reverse_tail_changes,
                queue_pops,
            }) => {
                self.rollback_fast_dates();
                self.rollback_graph_patch(movement);
                let outcome = self.consider_move_inner(movement, acceptance, stop, false)?;
                return Ok(FastN5Outcome {
                    outcome,
                    used_fast_path: false,
                    fell_back: true,
                    fallback_reason: Some(Self::fast_n5_fallback_reason(failure)),
                    used_topological_recovery: false,
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                });
            }
        };

        if !acceptance.accepts(previous, candidate) {
            self.rollback_fast_dates();
            self.rollback_graph_patch(movement);
            self.metrics.objective_rejections = self.metrics.objective_rejections.saturating_add(1);
            return Ok(FastN5Outcome {
                outcome: MoveOutcome::Rejected(MoveRejection::NotAccepted { current: previous, candidate }),
                used_fast_path: true,
                fell_back: false,
                fallback_reason: None,
                used_topological_recovery,
                forward_date_changes,
                reverse_tail_changes,
                queue_pops,
            });
        }

        match self.build_fast_canonical_analysis(candidate, stop) {
            Ok(()) => {}
            Err(ReconstructionFailure::Interrupted) => {
                self.rollback_fast_dates();
                self.rollback_graph_patch(movement);
                return Err(ScheduleStateInterrupted);
            }
            Err(_) => {
                self.rollback_fast_dates();
                self.rollback_graph_patch(movement);
                let outcome = self.consider_move_inner(movement, acceptance, stop, false)?;
                return Ok(FastN5Outcome {
                    outcome,
                    used_fast_path: false,
                    fell_back: true,
                    fallback_reason: Some(FastN5FallbackReason::Analysis),
                    used_topological_recovery,
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                });
            }
        }

        self.reconstruction.makespan = candidate;
        self.reconstruction.critical_path.clear();
        self.reconstruction.critical_path.extend_from_slice(&self.workspace.trial_critical_path);
        self.reconstruction.canonical_critical_blocks.clear();
        self.reconstruction.canonical_critical_blocks.extend_from_slice(&self.workspace.trial_canonical_critical_blocks);
        self.reconstruction.critical_blocks.clear();
        self.clear_fast_patch();
        self.metrics.moves_accepted = self.metrics.moves_accepted.saturating_add(1);
        self.metrics.critical_path_updates = self.metrics.critical_path_updates.saturating_add(1);
        self.metrics.dirty_cone_operations = self.metrics.dirty_cone_operations.saturating_add(queue_pops);
        self.metrics.max_dirty_cone = self.metrics.max_dirty_cone.max(queue_pops);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        Ok(FastN5Outcome {
            outcome: MoveOutcome::Accepted { previous, current: candidate },
            used_fast_path: true,
            fell_back: false,
            fallback_reason: None,
            used_topological_recovery,
            forward_date_changes,
            reverse_tail_changes,
            queue_pops,
        })
    }

    fn evaluate_strict_n5_value_changes(
        &mut self,
        arcs: ScheduleMoveArcs,
        stop: &AtomicBool,
    ) -> Result<(i64, u64, u64, u64), FastN5PropagationFailure> {
        let operation_count = self.problem.operation_count();
        // Each direction receives an independent streaming budget C=max(n,
        // 4_096). Direct convergence costs at most 2C. A forward WorkCap costs
        // C plus at most 4n for exact forward/reverse recovery. A reverse
        // WorkCap costs at most 2C plus 2n for reverse-only recovery. For large
        // instances these bounds are respectively 2n, 5n and 4n.
        let default_pop_cap = strict_n5_streaming_pop_cap(operation_count);
        #[cfg(test)]
        let forward_pop_cap = self.workspace.fast_forward_value_change_pop_cap.unwrap_or(default_pop_cap);
        #[cfg(not(test))]
        let forward_pop_cap = default_pop_cap;
        #[cfg(test)]
        let reverse_pop_cap = self.workspace.fast_reverse_value_change_pop_cap.unwrap_or(default_pop_cap);
        #[cfg(not(test))]
        let reverse_pop_cap = default_pop_cap;
        let mut queue_pops = 0usize;
        let mut forward_queue_pops = 0usize;
        let mut reverse_queue_pops = 0usize;
        let mut forward_date_changes = 0u64;
        let mut reverse_tail_changes = 0u64;

        let epoch = self.workspace.next_dirty_epoch();
        self.workspace.dirty_queue.clear();
        for index in 0..self.workspace.changed_roots.len() {
            let operation = self.workspace.changed_roots[index];
            self.enqueue_fast_value_change(epoch, operation);
        }
        let mut cursor = 0usize;
        while cursor < self.workspace.dirty_queue.len() {
            if stop.load(Ordering::Acquire) {
                return Err(FastN5PropagationFailure::new(
                    FastN5PropagationFailureKind::Interrupted,
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                ));
            }
            if forward_queue_pops >= forward_pop_cap {
                return Err(FastN5PropagationFailure::new(
                    FastN5PropagationFailureKind::WorkCap(FastN5PropagationPhase::Forward),
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                ));
            }
            let operation = self.workspace.dirty_queue[cursor];
            cursor += 1;
            queue_pops = queue_pops.saturating_add(1);
            forward_queue_pops = forward_queue_pops.saturating_add(1);
            self.workspace.fast_queue_marks[operation] = 0;
            let (start, end) =
                earliest_dates(&self.problem, &self.machine_predecessors, &self.reconstruction.ends, operation).map_err(|failure| {
                    FastN5PropagationFailure::from_reconstruction(failure, forward_date_changes, reverse_tail_changes, queue_pops)
                })?;
            if start == self.reconstruction.starts[operation] && end == self.reconstruction.ends[operation] {
                continue;
            }
            self.record_fast_patch(operation);
            self.reconstruction.starts[operation] = start;
            self.reconstruction.ends[operation] = end;
            forward_date_changes = forward_date_changes.saturating_add(1);
            for successor_index in 0..self.problem.precedences.successors(operation).len() {
                let successor = self.problem.precedences.successors(operation)[successor_index];
                self.enqueue_fast_value_change(epoch, successor);
            }
            let machine_successor = self.machine_successors[operation];
            if machine_successor != NO_OPERATION {
                self.enqueue_fast_value_change(epoch, machine_successor);
            }
        }

        // The strict gate guarantees positive durations, and changed_roots
        // contains the head of every added machine arc. Any newly introduced
        // cycle therefore contains a changed root and cannot reach a forward
        // value-change fixed point: dates keep increasing until WorkCap (or a
        // numeric error), where the exact forward Kahn pass detects the cycle.
        // Reaching the reverse phase proves that starts/ends are exact on a DAG.
        // More specifically, for a canonical block fragment p->a->b->s, the
        // replacement arcs p->b and a->s only shortcut existing paths. A cycle
        // through b->a would require another a=>b path after removing a->b;
        // with positive durations that would force start[b] > end[a], contrary
        // to a and b being consecutive on the critical block.

        let epoch = self.workspace.next_dirty_epoch();
        self.workspace.dirty_queue.clear();
        for arc in arcs.removed.into_iter().flatten().chain(arcs.added.into_iter().flatten()) {
            self.enqueue_fast_value_change(epoch, arc.before);
        }
        cursor = 0;
        while cursor < self.workspace.dirty_queue.len() {
            if stop.load(Ordering::Acquire) {
                return Err(FastN5PropagationFailure::new(
                    FastN5PropagationFailureKind::Interrupted,
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                ));
            }
            if reverse_queue_pops >= reverse_pop_cap {
                return Err(FastN5PropagationFailure::new(
                    FastN5PropagationFailureKind::WorkCap(FastN5PropagationPhase::Reverse),
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                ));
            }
            let operation = self.workspace.dirty_queue[cursor];
            cursor += 1;
            queue_pops = queue_pops.saturating_add(1);
            reverse_queue_pops = reverse_queue_pops.saturating_add(1);
            self.workspace.fast_queue_marks[operation] = 0;
            let duration = self.problem.duration(operation);
            let mut tail = duration;
            for &successor in self.problem.precedences.successors(operation) {
                let Some(candidate) = duration.checked_add(self.reconstruction.tails[successor]) else {
                    return Err(FastN5PropagationFailure::from_reconstruction(
                        ReconstructionFailure::Numeric,
                        forward_date_changes,
                        reverse_tail_changes,
                        queue_pops,
                    ));
                };
                tail = tail.max(candidate);
            }
            let machine_successor = self.machine_successors[operation];
            if machine_successor != NO_OPERATION && !self.problem.precedences.successors(operation).contains(&machine_successor) {
                let Some(candidate) = duration.checked_add(self.reconstruction.tails[machine_successor]) else {
                    return Err(FastN5PropagationFailure::from_reconstruction(
                        ReconstructionFailure::Numeric,
                        forward_date_changes,
                        reverse_tail_changes,
                        queue_pops,
                    ));
                };
                tail = tail.max(candidate);
            }
            if tail == self.reconstruction.tails[operation] {
                continue;
            }
            self.record_fast_patch(operation);
            self.reconstruction.tails[operation] = tail;
            reverse_tail_changes = reverse_tail_changes.saturating_add(1);
            for predecessor_index in 0..self.problem.precedences.predecessors(operation).len() {
                let predecessor = self.problem.precedences.predecessors(operation)[predecessor_index];
                self.enqueue_fast_value_change(epoch, predecessor);
            }
            let machine_predecessor = self.machine_predecessors[operation];
            if machine_predecessor != NO_OPERATION {
                self.enqueue_fast_value_change(epoch, machine_predecessor);
            }
        }

        let makespan =
            self.workspace.makespan_from_sink_union(&self.problem, &self.machine_sequences, &self.reconstruction.ends, stop).map_err(
                |failure| FastN5PropagationFailure::from_reconstruction(failure, forward_date_changes, reverse_tail_changes, queue_pops),
            )?;
        Ok((makespan, forward_date_changes, reverse_tail_changes, u64::try_from(queue_pops).unwrap_or(u64::MAX)))
    }

    /// Recover a wide strict-N5 value-change cone with one exact Kahn pass in
    /// each direction. The graph is already patched and all streaming date
    /// mutations have been rolled back before entry.
    fn evaluate_strict_n5_topologically(
        &mut self,
        arcs: ScheduleMoveArcs,
        partial: FastN5PropagationFailure,
        stop: &AtomicBool,
    ) -> Result<(i64, u64, u64, u64), FastN5PropagationFailure> {
        let mut forward_date_changes = partial.forward_date_changes;
        let reverse_tail_changes = partial.reverse_tail_changes;
        let mut queue_pops = usize::try_from(partial.queue_pops).unwrap_or(usize::MAX);

        let epoch = self.workspace.next_dirty_epoch();
        self.workspace.dirty_queue.clear();
        for &operation in &self.workspace.changed_roots {
            mark_dirty(&mut self.workspace.dirty_marks, &mut self.workspace.dirty_queue, &mut self.workspace.indegrees, epoch, operation);
        }
        let mut cursor = 0usize;
        while cursor < self.workspace.dirty_queue.len() {
            if stop.load(Ordering::Acquire) {
                return Err(FastN5PropagationFailure::new(
                    FastN5PropagationFailureKind::Interrupted,
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                ));
            }
            let operation = self.workspace.dirty_queue[cursor];
            cursor += 1;
            queue_pops = queue_pops.saturating_add(1);
            for &successor in self.problem.precedences.successors(operation) {
                mark_dirty(
                    &mut self.workspace.dirty_marks,
                    &mut self.workspace.dirty_queue,
                    &mut self.workspace.indegrees,
                    epoch,
                    successor,
                );
                let Some(degree) = self.workspace.indegrees[successor].checked_add(1) else {
                    return Err(FastN5PropagationFailure::from_reconstruction(
                        ReconstructionFailure::Numeric,
                        forward_date_changes,
                        reverse_tail_changes,
                        queue_pops,
                    ));
                };
                self.workspace.indegrees[successor] = degree;
            }
            let machine_successor = self.machine_successors[operation];
            if machine_successor != NO_OPERATION && !self.problem.precedences.successors(operation).contains(&machine_successor) {
                mark_dirty(
                    &mut self.workspace.dirty_marks,
                    &mut self.workspace.dirty_queue,
                    &mut self.workspace.indegrees,
                    epoch,
                    machine_successor,
                );
                let Some(degree) = self.workspace.indegrees[machine_successor].checked_add(1) else {
                    return Err(FastN5PropagationFailure::from_reconstruction(
                        ReconstructionFailure::Numeric,
                        forward_date_changes,
                        reverse_tail_changes,
                        queue_pops,
                    ));
                };
                self.workspace.indegrees[machine_successor] = degree;
            }
        }
        if let Err(failure) = self.workspace.rebuild_dirty_topology(&self.problem, &self.machine_successors, epoch, stop) {
            queue_pops = queue_pops.saturating_add(self.workspace.dirty_topological.len());
            return Err(FastN5PropagationFailure::from_reconstruction(failure, forward_date_changes, reverse_tail_changes, queue_pops));
        }
        queue_pops = queue_pops.saturating_add(self.workspace.dirty_topological.len());
        for index in 0..self.workspace.dirty_topological.len() {
            if stop.load(Ordering::Acquire) {
                return Err(FastN5PropagationFailure::new(
                    FastN5PropagationFailureKind::Interrupted,
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                ));
            }
            let operation = self.workspace.dirty_topological[index];
            let (start, end) =
                earliest_dates(&self.problem, &self.machine_predecessors, &self.reconstruction.ends, operation).map_err(|failure| {
                    FastN5PropagationFailure::from_reconstruction(failure, forward_date_changes, reverse_tail_changes, queue_pops)
                })?;
            if start == self.reconstruction.starts[operation] && end == self.reconstruction.ends[operation] {
                continue;
            }
            self.record_fast_patch(operation);
            self.reconstruction.starts[operation] = start;
            self.reconstruction.ends[operation] = end;
            forward_date_changes = forward_date_changes.saturating_add(1);
        }

        self.evaluate_strict_n5_reverse_topologically(arcs, forward_date_changes, reverse_tail_changes, queue_pops, stop)
    }

    /// Rebuild only the exact reverse closure. Starts and ends are already
    /// exact for the patched graph, while every tail has been restored to its
    /// original value before this method is entered.
    fn evaluate_strict_n5_reverse_topologically(
        &mut self,
        arcs: ScheduleMoveArcs,
        forward_date_changes: u64,
        mut reverse_tail_changes: u64,
        mut queue_pops: usize,
        stop: &AtomicBool,
    ) -> Result<(i64, u64, u64, u64), FastN5PropagationFailure> {
        let epoch = self.workspace.next_dirty_epoch();
        self.workspace.dirty_queue.clear();
        for arc in arcs.removed.into_iter().flatten().chain(arcs.added.into_iter().flatten()) {
            mark_dirty(&mut self.workspace.dirty_marks, &mut self.workspace.dirty_queue, &mut self.workspace.indegrees, epoch, arc.before);
        }
        let mut cursor = 0usize;
        while cursor < self.workspace.dirty_queue.len() {
            if stop.load(Ordering::Acquire) {
                return Err(FastN5PropagationFailure::new(
                    FastN5PropagationFailureKind::Interrupted,
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                ));
            }
            let operation = self.workspace.dirty_queue[cursor];
            cursor += 1;
            queue_pops = queue_pops.saturating_add(1);
            #[cfg(test)]
            if self.workspace.fast_stop_during_reverse_recovery && cursor == 1 {
                stop.store(true, Ordering::Release);
            }
            for &predecessor in self.problem.precedences.predecessors(operation) {
                mark_dirty(
                    &mut self.workspace.dirty_marks,
                    &mut self.workspace.dirty_queue,
                    &mut self.workspace.indegrees,
                    epoch,
                    predecessor,
                );
                let Some(degree) = self.workspace.indegrees[predecessor].checked_add(1) else {
                    return Err(FastN5PropagationFailure::from_reconstruction(
                        ReconstructionFailure::Numeric,
                        forward_date_changes,
                        reverse_tail_changes,
                        queue_pops,
                    ));
                };
                self.workspace.indegrees[predecessor] = degree;
            }
            let machine_predecessor = self.machine_predecessors[operation];
            if machine_predecessor != NO_OPERATION && !self.problem.precedences.predecessors(operation).contains(&machine_predecessor) {
                mark_dirty(
                    &mut self.workspace.dirty_marks,
                    &mut self.workspace.dirty_queue,
                    &mut self.workspace.indegrees,
                    epoch,
                    machine_predecessor,
                );
                let Some(degree) = self.workspace.indegrees[machine_predecessor].checked_add(1) else {
                    return Err(FastN5PropagationFailure::from_reconstruction(
                        ReconstructionFailure::Numeric,
                        forward_date_changes,
                        reverse_tail_changes,
                        queue_pops,
                    ));
                };
                self.workspace.indegrees[machine_predecessor] = degree;
            }
        }
        if let Err(failure) = self.workspace.rebuild_reverse_dirty_topology(&self.problem, &self.machine_predecessors, epoch, stop) {
            queue_pops = queue_pops.saturating_add(self.workspace.dirty_topological.len());
            return Err(FastN5PropagationFailure::from_reconstruction(failure, forward_date_changes, reverse_tail_changes, queue_pops));
        }
        queue_pops = queue_pops.saturating_add(self.workspace.dirty_topological.len());
        for index in 0..self.workspace.dirty_topological.len() {
            if stop.load(Ordering::Acquire) {
                return Err(FastN5PropagationFailure::new(
                    FastN5PropagationFailureKind::Interrupted,
                    forward_date_changes,
                    reverse_tail_changes,
                    queue_pops,
                ));
            }
            let operation = self.workspace.dirty_topological[index];
            let duration = self.problem.duration(operation);
            let mut tail = duration;
            for &successor in self.problem.precedences.successors(operation) {
                let Some(candidate) = duration.checked_add(self.reconstruction.tails[successor]) else {
                    return Err(FastN5PropagationFailure::from_reconstruction(
                        ReconstructionFailure::Numeric,
                        forward_date_changes,
                        reverse_tail_changes,
                        queue_pops,
                    ));
                };
                tail = tail.max(candidate);
            }
            let machine_successor = self.machine_successors[operation];
            if machine_successor != NO_OPERATION && !self.problem.precedences.successors(operation).contains(&machine_successor) {
                let Some(candidate) = duration.checked_add(self.reconstruction.tails[machine_successor]) else {
                    return Err(FastN5PropagationFailure::from_reconstruction(
                        ReconstructionFailure::Numeric,
                        forward_date_changes,
                        reverse_tail_changes,
                        queue_pops,
                    ));
                };
                tail = tail.max(candidate);
            }
            if tail == self.reconstruction.tails[operation] {
                continue;
            }
            self.record_fast_patch(operation);
            self.reconstruction.tails[operation] = tail;
            reverse_tail_changes = reverse_tail_changes.saturating_add(1);
        }

        let makespan =
            self.workspace.makespan_from_sink_union(&self.problem, &self.machine_sequences, &self.reconstruction.ends, stop).map_err(
                |failure| FastN5PropagationFailure::from_reconstruction(failure, forward_date_changes, reverse_tail_changes, queue_pops),
            )?;
        Ok((makespan, forward_date_changes, reverse_tail_changes, u64::try_from(queue_pops).unwrap_or(u64::MAX)))
    }

    fn fast_n5_fallback_reason(failure: ReconstructionFailure) -> FastN5FallbackReason {
        match failure {
            ReconstructionFailure::Interrupted => unreachable!("interruption is returned to the caller"),
            ReconstructionFailure::Cycle => FastN5FallbackReason::Cycle,
            ReconstructionFailure::Window => FastN5FallbackReason::Window,
            ReconstructionFailure::Numeric => FastN5FallbackReason::Numeric,
        }
    }

    fn enqueue_fast_value_change(&mut self, epoch: u32, operation: usize) {
        if self.workspace.fast_queue_marks[operation] != epoch {
            self.workspace.fast_queue_marks[operation] = epoch;
            self.workspace.dirty_queue.push(operation);
        }
    }

    fn record_fast_patch(&mut self, operation: usize) {
        let epoch = self.workspace.fast_patch_epoch;
        if self.workspace.fast_patch_marks[operation] == epoch {
            return;
        }
        self.workspace.fast_patch_marks[operation] = epoch;
        self.workspace.fast_patched_operations.push(operation);
        self.workspace.fast_patched_starts.push(self.reconstruction.starts[operation]);
        self.workspace.fast_patched_ends.push(self.reconstruction.ends[operation]);
        self.workspace.fast_patched_tails.push(self.reconstruction.tails[operation]);
    }

    fn rollback_fast_dates(&mut self) {
        for index in (0..self.workspace.fast_patched_operations.len()).rev() {
            let operation = self.workspace.fast_patched_operations[index];
            self.reconstruction.starts[operation] = self.workspace.fast_patched_starts[index];
            self.reconstruction.ends[operation] = self.workspace.fast_patched_ends[index];
            self.reconstruction.tails[operation] = self.workspace.fast_patched_tails[index];
        }
        self.clear_fast_patch();
    }

    /// Restore the original tails while retaining the forward start/end patch
    /// and its journal. A reverse-only recovery can then replace the discarded
    /// streaming tails, and any later failure still has the original snapshot
    /// needed for a complete rollback.
    fn rollback_fast_tails(&mut self) {
        for index in (0..self.workspace.fast_patched_operations.len()).rev() {
            let operation = self.workspace.fast_patched_operations[index];
            self.reconstruction.tails[operation] = self.workspace.fast_patched_tails[index];
        }
    }

    fn clear_fast_patch(&mut self) {
        self.workspace.fast_patched_operations.clear();
        self.workspace.fast_patched_starts.clear();
        self.workspace.fast_patched_ends.clear();
        self.workspace.fast_patched_tails.clear();
    }

    fn build_fast_canonical_analysis(&mut self, makespan: i64, stop: &AtomicBool) -> Result<(), ReconstructionFailure> {
        let candidates = self
            .problem
            .precedence_sinks
            .iter()
            .copied()
            .chain(self.machine_sequences.iter().filter_map(|sequence| sequence.last().copied()));
        let terminal = select_canonical_terminal(
            candidates,
            &self.machine_predecessors,
            &self.reconstruction.starts,
            &self.reconstruction.ends,
            makespan,
            self.canonical_path_policy,
            stop,
        )?
        .ok_or(ReconstructionFailure::Cycle)?;
        build_canonical_path_from_terminal(
            &mut self.workspace.trial_critical_path,
            &mut self.workspace.trial_canonical_critical_blocks,
            &self.problem,
            &self.machine_predecessors,
            &self.positions,
            &self.reconstruction.starts,
            &self.reconstruction.ends,
            self.canonical_path_policy,
            Some(terminal),
            false,
            &mut self.workspace.analysis_stop_after_path_operations,
            stop,
        )
    }

    /// Reconstruct a move into temporary buffers and commit it only when it is
    /// feasible and accepted. Every other outcome restores the exact machine
    /// order before returning, including cancellation.
    pub(crate) fn consider_move(
        &mut self,
        movement: ScheduleMove,
        acceptance: MinimizingMoveAcceptance,
        stop: &AtomicBool,
    ) -> Result<MoveOutcome, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let result = self.consider_move_inner(movement, acceptance, stop, true);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        result
    }

    /// Evaluate a candidate with the incremental kernel and restore the
    /// accepted state before returning. A selected candidate is independently
    /// oracle-validated by `commit_probed_move` before it can be committed.
    pub(crate) fn probe_move(&mut self, movement: ScheduleMove, stop: &AtomicBool) -> Result<MoveProbe, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let result = self.probe_move_inner(movement, stop);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        result
    }

    /// Commit a previously selected move without counting the same candidate a
    /// second time. Validation is deliberately repeated against current state.
    pub(crate) fn commit_probed_move(
        &mut self,
        movement: ScheduleMove,
        acceptance: MinimizingMoveAcceptance,
        stop: &AtomicBool,
    ) -> Result<MoveOutcome, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let result = self.consider_move_inner(movement, acceptance, stop, false);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        result
    }

    /// Apply a move and send it directly through the complete reconstruction
    /// oracle. The accepted state changes only after feasibility, objective
    /// acceptance, and analysis all succeed. No incremental delta evaluation
    /// is performed.
    pub(crate) fn consider_move_full_oracle(
        &mut self,
        movement: ScheduleMove,
        acceptance: MinimizingMoveAcceptance,
        stop: &AtomicBool,
    ) -> Result<MoveOutcome, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        self.metrics.moves_considered = self.metrics.moves_considered.saturating_add(1);
        self.metrics.direct_oracle_attempts = self.metrics.direct_oracle_attempts.saturating_add(1);
        if !move_is_valid(&self.machine_sequences, movement) {
            return Ok(MoveOutcome::Rejected(MoveRejection::Invalid));
        }
        self.apply_graph_patch(movement);
        if stop.load(Ordering::Acquire) {
            self.rollback_graph_patch(movement);
            return Err(ScheduleStateInterrupted);
        }

        let previous = self.makespan();
        let candidate = match self.run_full_evaluation(stop) {
            Ok(candidate) => candidate,
            Err(failure) => {
                self.rollback_graph_patch(movement);
                return self.direct_oracle_failure(failure);
            }
        };
        self.metrics.oracle_validations = self.metrics.oracle_validations.saturating_add(1);
        if !acceptance.accepts(previous, candidate) {
            self.rollback_graph_patch(movement);
            self.metrics.objective_rejections = self.metrics.objective_rejections.saturating_add(1);
            self.metrics.direct_oracle_objective_rejections = self.metrics.direct_oracle_objective_rejections.saturating_add(1);
            return Ok(MoveOutcome::Rejected(MoveRejection::NotAccepted { current: previous, candidate }));
        }
        if let Err(failure) = self.workspace.build_analysis(
            &self.problem,
            &self.machine_predecessors,
            &self.machine_successors,
            &self.positions,
            &self.machine_sequences,
            self.canonical_path_policy,
            self.collect_all_critical_blocks,
            stop,
        ) {
            self.rollback_graph_patch(movement);
            return self.direct_oracle_failure(failure);
        }
        self.reconstruction.commit(&self.workspace);
        self.clear_date_patch();
        self.metrics.moves_accepted = self.metrics.moves_accepted.saturating_add(1);
        self.metrics.direct_oracle_accepts = self.metrics.direct_oracle_accepts.saturating_add(1);
        self.metrics.critical_path_updates = self.metrics.critical_path_updates.saturating_add(1);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        Ok(MoveOutcome::Accepted { previous, current: candidate })
    }

    /// Apply a sequence of structured moves as one atomic candidate and run
    /// the complete reconstruction oracle exactly once. Every move is
    /// interpreted against the order produced by the preceding moves. No
    /// single-move feasibility certificate can authorize the combined graph.
    ///
    /// The batch counts as one considered candidate and one direct-oracle
    /// attempt. Invalid or rejected batches, oracle failures, and cancellation
    /// restore all applied sequence changes in reverse order before returning.
    pub(crate) fn consider_move_batch_full_oracle(
        &mut self,
        movements: &[ScheduleMove],
        acceptance: MinimizingMoveAcceptance,
        stop: &AtomicBool,
    ) -> Result<MoveOutcome, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        self.metrics.moves_considered = self.metrics.moves_considered.saturating_add(1);
        self.metrics.direct_oracle_attempts = self.metrics.direct_oracle_attempts.saturating_add(1);
        if movements.is_empty() {
            return Ok(MoveOutcome::Rejected(MoveRejection::Invalid));
        }

        let previous = self.makespan();
        let mut applied = 0usize;
        for &movement in movements {
            if stop.load(Ordering::Acquire) {
                self.rollback_graph_patches(&movements[..applied]);
                return Err(ScheduleStateInterrupted);
            }
            if !move_is_valid(&self.machine_sequences, movement) {
                self.rollback_graph_patches(&movements[..applied]);
                return Ok(MoveOutcome::Rejected(MoveRejection::Invalid));
            }
            self.apply_graph_patch(movement);
            applied += 1;

            #[cfg(test)]
            if self.workspace.batch_stop_after_applied_moves == Some(applied) {
                self.workspace.batch_stop_after_applied_moves = None;
                stop.store(true, Ordering::Release);
            }
            if stop.load(Ordering::Acquire) {
                self.rollback_graph_patches(&movements[..applied]);
                return Err(ScheduleStateInterrupted);
            }
        }

        let candidate = match self.run_full_evaluation(stop) {
            Ok(candidate) => candidate,
            Err(failure) => {
                self.rollback_graph_patches(&movements[..applied]);
                return self.direct_oracle_failure(failure);
            }
        };
        self.metrics.oracle_validations = self.metrics.oracle_validations.saturating_add(1);
        if !acceptance.accepts(previous, candidate) {
            self.rollback_graph_patches(&movements[..applied]);
            self.metrics.objective_rejections = self.metrics.objective_rejections.saturating_add(1);
            self.metrics.direct_oracle_objective_rejections = self.metrics.direct_oracle_objective_rejections.saturating_add(1);
            return Ok(MoveOutcome::Rejected(MoveRejection::NotAccepted { current: previous, candidate }));
        }
        if let Err(failure) = self.workspace.build_analysis(
            &self.problem,
            &self.machine_predecessors,
            &self.machine_successors,
            &self.positions,
            &self.machine_sequences,
            self.canonical_path_policy,
            self.collect_all_critical_blocks,
            stop,
        ) {
            self.rollback_graph_patches(&movements[..applied]);
            return self.direct_oracle_failure(failure);
        }
        if stop.load(Ordering::Acquire) {
            self.rollback_graph_patches(&movements[..applied]);
            return Err(ScheduleStateInterrupted);
        }

        self.reconstruction.commit(&self.workspace);
        self.clear_date_patch();
        self.metrics.moves_accepted = self.metrics.moves_accepted.saturating_add(1);
        self.metrics.direct_oracle_accepts = self.metrics.direct_oracle_accepts.saturating_add(1);
        self.metrics.critical_path_updates = self.metrics.critical_path_updates.saturating_add(1);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        Ok(MoveOutcome::Accepted { previous, current: candidate })
    }

    fn probe_move_inner(&mut self, movement: ScheduleMove, stop: &AtomicBool) -> Result<MoveProbe, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        self.metrics.moves_considered = self.metrics.moves_considered.saturating_add(1);
        if !move_is_valid(&self.machine_sequences, movement) {
            return Ok(MoveProbe::Rejected(MoveRejection::Invalid));
        }
        self.apply_graph_patch(movement);
        if stop.load(Ordering::Acquire) {
            self.rollback_graph_patch(movement);
            return Err(ScheduleStateInterrupted);
        }

        let current = self.makespan();
        self.metrics.delta_evaluations = self.metrics.delta_evaluations.saturating_add(1);
        let candidate = match self.evaluate_delta(stop) {
            Ok(candidate) => candidate,
            Err(failure) => {
                self.rollback_dates();
                self.rollback_graph_patch(movement);
                return self.record_failure(failure).map(MoveProbe::Rejected);
            }
        };
        self.rollback_dates();
        self.rollback_graph_patch(movement);
        Ok(MoveProbe::Feasible { current, candidate })
    }

    fn consider_move_inner(
        &mut self,
        movement: ScheduleMove,
        acceptance: MinimizingMoveAcceptance,
        stop: &AtomicBool,
        count_considered: bool,
    ) -> Result<MoveOutcome, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        if count_considered {
            self.metrics.moves_considered = self.metrics.moves_considered.saturating_add(1);
        }
        if !move_is_valid(&self.machine_sequences, movement) {
            return Ok(MoveOutcome::Rejected(MoveRejection::Invalid));
        }
        self.apply_graph_patch(movement);
        if stop.load(Ordering::Acquire) {
            self.rollback_graph_patch(movement);
            return Err(ScheduleStateInterrupted);
        }

        let previous = self.makespan();
        self.metrics.delta_evaluations = self.metrics.delta_evaluations.saturating_add(1);
        let evaluation = self.evaluate_delta(stop);
        let mut candidate = match evaluation {
            Ok(candidate) => candidate,
            Err(failure) => {
                self.rollback_dates();
                self.rollback_graph_patch(movement);
                return self.rejected_failure(failure);
            }
        };

        if !acceptance.accepts(previous, candidate) {
            self.rollback_dates();
            self.rollback_graph_patch(movement);
            self.metrics.objective_rejections = self.metrics.objective_rejections.saturating_add(1);
            return Ok(MoveOutcome::Rejected(MoveRejection::NotAccepted { current: previous, candidate }));
        }

        let delta_candidate = candidate;
        candidate = match self.run_full_evaluation(stop) {
            Ok(candidate) => candidate,
            Err(failure) => {
                self.rollback_dates();
                self.rollback_graph_patch(movement);
                return self.rejected_failure(failure);
            }
        };
        self.metrics.oracle_validations = self.metrics.oracle_validations.saturating_add(1);
        if delta_candidate != candidate
            || self.reconstruction.starts != self.workspace.trial_starts
            || self.reconstruction.ends != self.workspace.trial_ends
        {
            self.metrics.oracle_mismatches = self.metrics.oracle_mismatches.saturating_add(1);
        }
        if !acceptance.accepts(previous, candidate) {
            self.rollback_dates();
            self.rollback_graph_patch(movement);
            self.metrics.objective_rejections = self.metrics.objective_rejections.saturating_add(1);
            return Ok(MoveOutcome::Rejected(MoveRejection::NotAccepted { current: previous, candidate }));
        }

        if let Err(failure) = self.workspace.build_analysis(
            &self.problem,
            &self.machine_predecessors,
            &self.machine_successors,
            &self.positions,
            &self.machine_sequences,
            self.canonical_path_policy,
            self.collect_all_critical_blocks,
            stop,
        ) {
            self.rollback_dates();
            self.rollback_graph_patch(movement);
            return self.rejected_failure(failure);
        }
        self.reconstruction.commit(&self.workspace);
        self.clear_date_patch();
        self.metrics.moves_accepted = self.metrics.moves_accepted.saturating_add(1);
        self.metrics.critical_path_updates = self.metrics.critical_path_updates.saturating_add(1);
        Ok(MoveOutcome::Accepted { previous, current: candidate })
    }

    fn apply_graph_patch(&mut self, movement: ScheduleMove) {
        let machine = movement.machine();
        let (first_position, last_position) = movement.position_bounds();
        let applied = apply_sequence_move(&mut self.machine_sequences, movement);
        debug_assert!(applied);
        self.workspace.changed_roots.clear();
        refresh_machine_links(
            &self.machine_sequences[machine],
            first_position,
            last_position,
            &mut self.machine_predecessors,
            &mut self.machine_successors,
            &mut self.positions,
            Some(&mut self.workspace.changed_roots),
        );
    }

    fn rollback_graph_patch(&mut self, movement: ScheduleMove) {
        undo_sequence_move(&mut self.machine_sequences, movement);
        let machine = movement.machine();
        let (first_position, last_position) = movement.position_bounds();
        refresh_machine_links(
            &self.machine_sequences[machine],
            first_position,
            last_position,
            &mut self.machine_predecessors,
            &mut self.machine_successors,
            &mut self.positions,
            None,
        );
    }

    fn rollback_graph_patches(&mut self, movements: &[ScheduleMove]) {
        for &movement in movements.iter().rev() {
            self.rollback_graph_patch(movement);
        }
    }

    fn evaluate_delta(&mut self, stop: &AtomicBool) -> Result<i64, ReconstructionFailure> {
        self.clear_date_patch();
        let epoch = self.workspace.next_dirty_epoch();
        self.workspace.dirty_queue.clear();
        // Build the forward dirty closure and its induced indegrees together.
        // Coincident job and machine arcs count once in both directions.
        for &operation in &self.workspace.changed_roots {
            mark_dirty(&mut self.workspace.dirty_marks, &mut self.workspace.dirty_queue, &mut self.workspace.indegrees, epoch, operation);
        }
        let mut cursor = 0usize;
        while cursor < self.workspace.dirty_queue.len() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let operation = self.workspace.dirty_queue[cursor];
            cursor += 1;
            for &successor in self.problem.precedences.successors(operation) {
                mark_dirty(
                    &mut self.workspace.dirty_marks,
                    &mut self.workspace.dirty_queue,
                    &mut self.workspace.indegrees,
                    epoch,
                    successor,
                );
                self.workspace.indegrees[successor] =
                    self.workspace.indegrees[successor].checked_add(1).ok_or(ReconstructionFailure::Numeric)?;
            }
            let machine_successor = self.machine_successors[operation];
            if machine_successor != NO_OPERATION && !self.problem.precedences.successors(operation).contains(&machine_successor) {
                mark_dirty(
                    &mut self.workspace.dirty_marks,
                    &mut self.workspace.dirty_queue,
                    &mut self.workspace.indegrees,
                    epoch,
                    machine_successor,
                );
                self.workspace.indegrees[machine_successor] =
                    self.workspace.indegrees[machine_successor].checked_add(1).ok_or(ReconstructionFailure::Numeric)?;
            }
        }
        let dirty = u64::try_from(self.workspace.dirty_queue.len()).unwrap_or(u64::MAX);
        self.metrics.dirty_cone_operations = self.metrics.dirty_cone_operations.saturating_add(dirty);
        self.metrics.max_dirty_cone = self.metrics.max_dirty_cone.max(dirty);

        if self.workspace.dirty_queue.len() == self.problem.operation_count() {
            self.metrics.full_fallbacks = self.metrics.full_fallbacks.saturating_add(1);
        }
        self.workspace.rebuild_dirty_topology(&self.problem, &self.machine_successors, epoch, stop)?;

        for index in 0..self.workspace.dirty_topological.len() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let operation = self.workspace.dirty_topological[index];
            let (start, end) = earliest_dates(&self.problem, &self.machine_predecessors, &self.reconstruction.ends, operation)?;
            if start != self.reconstruction.starts[operation] || end != self.reconstruction.ends[operation] {
                self.workspace.patched_operations.push(operation);
                self.workspace.patched_starts.push(self.reconstruction.starts[operation]);
                self.workspace.patched_ends.push(self.reconstruction.ends[operation]);
                self.reconstruction.starts[operation] = start;
                self.reconstruction.ends[operation] = end;
            }
        }
        self.workspace.makespan_from_sink_union(&self.problem, &self.machine_sequences, &self.reconstruction.ends, stop)
    }

    fn run_full_evaluation(&mut self, stop: &AtomicBool) -> Result<i64, ReconstructionFailure> {
        self.metrics.reconstructions = self.metrics.reconstructions.saturating_add(1);
        self.metrics.full_evaluations = self.metrics.full_evaluations.saturating_add(1);
        self.workspace.full_rebuild(&self.problem, &self.machine_predecessors, &self.machine_successors, stop)
    }

    fn rollback_dates(&mut self) {
        for index in (0..self.workspace.patched_operations.len()).rev() {
            let operation = self.workspace.patched_operations[index];
            self.reconstruction.starts[operation] = self.workspace.patched_starts[index];
            self.reconstruction.ends[operation] = self.workspace.patched_ends[index];
        }
        self.clear_date_patch();
    }

    fn clear_date_patch(&mut self) {
        self.workspace.patched_operations.clear();
        self.workspace.patched_starts.clear();
        self.workspace.patched_ends.clear();
    }

    fn rejected_failure(&mut self, failure: ReconstructionFailure) -> Result<MoveOutcome, ScheduleStateInterrupted> {
        self.record_failure(failure).map(MoveOutcome::Rejected)
    }

    fn direct_oracle_failure(&mut self, failure: ReconstructionFailure) -> Result<MoveOutcome, ScheduleStateInterrupted> {
        match failure {
            ReconstructionFailure::Cycle => {
                self.metrics.direct_oracle_cycles = self.metrics.direct_oracle_cycles.saturating_add(1);
            }
            ReconstructionFailure::Window => {
                self.metrics.direct_oracle_windows = self.metrics.direct_oracle_windows.saturating_add(1);
            }
            ReconstructionFailure::Interrupted | ReconstructionFailure::Numeric => {}
        }
        self.rejected_failure(failure)
    }

    fn record_failure(&mut self, failure: ReconstructionFailure) -> Result<MoveRejection, ScheduleStateInterrupted> {
        match failure {
            ReconstructionFailure::Interrupted => Err(ScheduleStateInterrupted),
            ReconstructionFailure::Cycle => {
                self.metrics.cycle_rejections = self.metrics.cycle_rejections.saturating_add(1);
                Ok(MoveRejection::Cycle)
            }
            ReconstructionFailure::Window => {
                self.metrics.window_rejections = self.metrics.window_rejections.saturating_add(1);
                Ok(MoveRejection::Window)
            }
            ReconstructionFailure::Numeric => Ok(MoveRejection::Numeric),
        }
    }

    /// Rebuild the accepted order from scratch into the reusable oracle buffers.
    pub(crate) fn matches_full_oracle(&mut self, stop: &AtomicBool) -> Result<bool, ScheduleStateInterrupted> {
        let result = self.matches_full_oracle_inner(stop);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        result
    }

    /// Differentially validate semantic dates produced by the strict N5
    /// micro-kernel, then refresh every derived analysis buffer from the full
    /// oracle. A mismatch is repaired before returning `false`.
    pub(crate) fn refresh_from_full_oracle(&mut self, stop: &AtomicBool) -> Result<bool, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let makespan = match self.run_full_evaluation(stop) {
            Ok(makespan) => makespan,
            Err(ReconstructionFailure::Interrupted) => return Err(ScheduleStateInterrupted),
            Err(_) => {
                self.metrics.oracle_validations = self.metrics.oracle_validations.saturating_add(1);
                self.metrics.oracle_mismatches = self.metrics.oracle_mismatches.saturating_add(1);
                return Ok(false);
            }
        };
        self.metrics.oracle_validations = self.metrics.oracle_validations.saturating_add(1);
        if let Err(failure) = self.workspace.build_analysis(
            &self.problem,
            &self.machine_predecessors,
            &self.machine_successors,
            &self.positions,
            &self.machine_sequences,
            self.canonical_path_policy,
            self.collect_all_critical_blocks,
            stop,
        ) {
            return match failure {
                ReconstructionFailure::Interrupted => Err(ScheduleStateInterrupted),
                _ => {
                    self.metrics.oracle_mismatches = self.metrics.oracle_mismatches.saturating_add(1);
                    Ok(false)
                }
            };
        }
        let exact = makespan == self.reconstruction.makespan
            && self.workspace.trial_starts == self.reconstruction.starts
            && self.workspace.trial_ends == self.reconstruction.ends
            && self.workspace.trial_tails == self.reconstruction.tails
            && self.workspace.trial_critical_path == self.reconstruction.critical_path
            && self.workspace.trial_canonical_critical_blocks == self.reconstruction.canonical_critical_blocks;
        if !exact {
            self.metrics.oracle_mismatches = self.metrics.oracle_mismatches.saturating_add(1);
        }
        self.reconstruction.commit(&self.workspace);
        self.clear_fast_patch();
        self.metrics.critical_path_updates = self.metrics.critical_path_updates.saturating_add(1);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        Ok(exact)
    }

    fn matches_full_oracle_inner(&mut self, stop: &AtomicBool) -> Result<bool, ScheduleStateInterrupted> {
        let result = self.run_full_evaluation(stop);
        let makespan = match result {
            Ok(makespan) => makespan,
            Err(ReconstructionFailure::Interrupted) => return Err(ScheduleStateInterrupted),
            Err(_) => return Ok(false),
        };
        self.metrics.oracle_validations = self.metrics.oracle_validations.saturating_add(1);
        match self.workspace.build_analysis(
            &self.problem,
            &self.machine_predecessors,
            &self.machine_successors,
            &self.positions,
            &self.machine_sequences,
            self.canonical_path_policy,
            self.collect_all_critical_blocks,
            stop,
        ) {
            Ok(()) => {}
            Err(ReconstructionFailure::Interrupted) => return Err(ScheduleStateInterrupted),
            Err(_) => return Ok(false),
        }
        Ok(makespan == self.reconstruction.makespan
            && self.workspace.trial_starts == self.reconstruction.starts
            && self.workspace.trial_ends == self.reconstruction.ends
            && self.workspace.trial_latest_starts == self.reconstruction.latest_starts
            && self.workspace.trial_tails == self.reconstruction.tails
            && self.workspace.trial_topological == self.reconstruction.topological
            && self.workspace.trial_topological_rank == self.reconstruction.topological_rank
            && self.workspace.trial_critical_path == self.reconstruction.critical_path
            && self.workspace.trial_canonical_critical_blocks == self.reconstruction.canonical_critical_blocks
            && self.workspace.trial_critical_blocks == self.reconstruction.critical_blocks)
    }

    pub(crate) fn to_solution(&self) -> CollectionSolution {
        CollectionSolution {
            lists: Vec::new(),
            objectives: vec![self.makespan()],
            feasible: true,
            starts: self.reconstruction.starts.clone(),
            presences: vec![true; self.problem.operation_count()],
            machines: self.problem.solution_machines.clone(),
            modes: self.problem.solution_modes.clone(),
            bound: None,
        }
    }
}

/// Reusable compact scratch for repeated Giffler-Thompson starts.
///
/// It deliberately retains neither reconstructed states nor past orders.  The
/// owner can materialize a complete semantic solution only after deciding that
/// the most recent construction improves its incumbent.
pub(crate) struct GifflerThompsonWorkspace {
    starts: Vec<i64>,
    completion: Vec<i64>,
    indegrees: Vec<usize>,
    machine_sequences: Vec<Vec<usize>>,
    ready: GifflerReadySet,
}

impl GifflerThompsonWorkspace {
    pub(crate) fn new(problem: &JobShopProblem) -> Self {
        Self {
            starts: vec![0; problem.operation_count()],
            completion: vec![0; problem.operation_count()],
            indegrees: vec![0; problem.operation_count()],
            machine_sequences: vec![Vec::new(); problem.machine_count()],
            ready: GifflerReadySet::new(problem),
        }
    }

    pub(crate) fn construct(
        &mut self,
        problem: &JobShopProblem,
        seed: u64,
        rule: DispatchRule,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<Option<(i64, u64)>, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        if self.starts.len() != problem.operation_count() || self.ready.positions.len() != problem.operation_count() {
            return Ok(None);
        }
        self.starts.fill(0);
        self.completion.fill(0);
        for sequence in &mut self.machine_sequences {
            sequence.clear();
        }
        self.ready.reset();
        for operation in 0..problem.operation_count() {
            checkpoint(stop)?;
            self.indegrees[operation] = problem.precedences.predecessors(operation).len();
            if self.indegrees[operation] == 0 && !self.ready.activate(problem, operation, 0, stop, metrics)? {
                return Ok(None);
            }
        }

        let mut objective = 0i64;
        for step in 0..problem.operation_count() {
            checkpoint(stop)?;
            self.ready.maybe_rebuild(problem, stop, metrics)?;
            let Some((pivot, pivot_machine, cutoff)) = self.ready.pop_pivot(stop, metrics)? else {
                return Ok(None);
            };
            let Some(selected) = self.ready.select(problem, pivot, pivot_machine, cutoff, rule, seed, step, stop, metrics)? else {
                return Ok(None);
            };
            let operation = selected.operation;
            if !self.ready.remove(problem, operation) {
                return Ok(None);
            }
            self.starts[operation] = selected.start;
            self.completion[operation] = selected.end;
            objective = objective.max(selected.end);
            let machine = problem.machine(operation);
            self.machine_sequences[machine].push(operation);
            if !self.ready.refresh_machine(problem, machine, selected.end, stop, metrics)? {
                return Ok(None);
            }
            for &successor in problem.precedences.successors(operation) {
                checkpoint(stop)?;
                let Some(value) = self.indegrees[successor].checked_sub(1) else {
                    return Ok(None);
                };
                self.indegrees[successor] = value;
                if value == 0 {
                    let release = problem
                        .precedences
                        .predecessors(successor)
                        .iter()
                        .map(|&predecessor| self.completion[predecessor])
                        .max()
                        .unwrap_or(0);
                    if !self.ready.activate(problem, successor, release, stop, metrics)? {
                        return Ok(None);
                    }
                }
            }
        }
        let fingerprint = machine_sequence_fingerprint(&self.machine_sequences);
        Ok(Some((objective, fingerprint)))
    }

    pub(crate) fn to_solution(&self, problem: &JobShopProblem, objective: i64) -> CollectionSolution {
        problem.solution_from_starts(self.starts.clone(), objective)
    }

    pub(crate) fn heap_lower_bound_bytes(&self) -> usize {
        self.starts
            .capacity()
            .saturating_mul(size_of::<i64>())
            .saturating_add(self.completion.capacity().saturating_mul(size_of::<i64>()))
            .saturating_add(self.indegrees.capacity().saturating_mul(size_of::<usize>()))
            .saturating_add(self.machine_sequences.capacity().saturating_mul(size_of::<Vec<usize>>()))
            .saturating_add(
                self.machine_sequences
                    .iter()
                    .fold(0usize, |total, sequence| total.saturating_add(sequence.capacity().saturating_mul(size_of::<usize>()))),
            )
            .saturating_add(self.ready.heap_lower_bound_bytes())
    }
}

#[cfg(test)]
pub(crate) fn audit_strict_n5_layout(block_lengths: &[usize]) -> Vec<ScheduleMove> {
    let stop = AtomicBool::new(false);
    let mut first_position = 0usize;
    let blocks: Vec<CriticalBlock> = block_lengths
        .iter()
        .enumerate()
        .map(|(machine, &length)| {
            let block = CriticalBlock { machine, first_position, last_position: first_position.saturating_add(length.saturating_sub(1)) };
            first_position = first_position.saturating_add(length);
            block
        })
        .collect();
    let mut movements = Vec::new();
    JobShopState::visit_strict_n5_moves(&blocks, &stop, |movement| {
        movements.push(movement);
        Ok(())
    })
    .expect("non-interrupted synthetic N5 layout");
    movements
}

fn machine_sequence_fingerprint(sequences: &[Vec<usize>]) -> u64 {
    let mut fingerprint = mix64(u64::try_from(sequences.len()).unwrap_or(u64::MAX));
    for (machine, sequence) in sequences.iter().enumerate() {
        fingerprint = mix64(
            fingerprint
                ^ u64::try_from(machine).unwrap_or(u64::MAX).wrapping_mul(0xa076_1d64_78bd_642f)
                ^ u64::try_from(sequence.len()).unwrap_or(u64::MAX),
        );
        for (position, &operation) in sequence.iter().enumerate() {
            fingerprint = mix64(
                fingerprint
                    ^ u64::try_from(operation).unwrap_or(u64::MAX)
                    ^ u64::try_from(position).unwrap_or(u64::MAX).wrapping_mul(0xe703_7ed1_a0b4_28db),
            );
        }
    }
    fingerprint
}

type GifflerHeapEntry = Reverse<(i64, usize, usize, u64)>;

/// Ready operations indexed by machine for the Giffler-Thompson constructor.
///
/// A machine readiness change invalidates only entries for that machine. Heap
/// generations make those invalidations lazy, while the position vector keeps
/// activation and removal constant time. Cached releases never change after an
/// operation becomes precedence-ready.
struct GifflerReadySet {
    buckets: Vec<Vec<usize>>,
    positions: Vec<usize>,
    machine_ready: Vec<i64>,
    releases: Vec<i64>,
    starts: Vec<i64>,
    ends: Vec<i64>,
    generations: Vec<u64>,
    heap: BinaryHeap<GifflerHeapEntry>,
    active: usize,
}

impl GifflerReadySet {
    fn new(problem: &JobShopProblem) -> Self {
        Self {
            buckets: vec![Vec::new(); problem.machine_count()],
            positions: vec![NO_OPERATION; problem.operation_count()],
            machine_ready: vec![0; problem.machine_count()],
            releases: vec![0; problem.operation_count()],
            starts: vec![0; problem.operation_count()],
            ends: vec![0; problem.operation_count()],
            generations: vec![0; problem.machine_count()],
            heap: BinaryHeap::new(),
            active: 0,
        }
    }

    fn reset(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.positions.fill(NO_OPERATION);
        self.machine_ready.fill(0);
        self.releases.fill(0);
        self.starts.fill(0);
        self.ends.fill(0);
        self.generations.fill(0);
        self.heap.clear();
        self.active = 0;
    }

    fn heap_lower_bound_bytes(&self) -> usize {
        let bucket_items =
            self.buckets.iter().fold(0usize, |total, bucket| total.saturating_add(bucket.capacity().saturating_mul(size_of::<usize>())));
        self.buckets
            .capacity()
            .saturating_mul(size_of::<Vec<usize>>())
            .saturating_add(bucket_items)
            .saturating_add(self.positions.capacity().saturating_mul(size_of::<usize>()))
            .saturating_add(self.machine_ready.capacity().saturating_mul(size_of::<i64>()))
            .saturating_add(self.releases.capacity().saturating_mul(size_of::<i64>()))
            .saturating_add(self.starts.capacity().saturating_mul(size_of::<i64>()))
            .saturating_add(self.ends.capacity().saturating_mul(size_of::<i64>()))
            .saturating_add(self.generations.capacity().saturating_mul(size_of::<u64>()))
            .saturating_add(self.heap.capacity().saturating_mul(size_of::<GifflerHeapEntry>()))
    }

    fn activate(
        &mut self,
        problem: &JobShopProblem,
        operation: usize,
        release: i64,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<bool, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        if operation >= self.positions.len() || self.positions[operation] != NO_OPERATION {
            return Ok(false);
        }
        let machine = problem.machine(operation);
        let position = self.buckets[machine].len();
        self.buckets[machine].push(operation);
        self.positions[operation] = position;
        self.releases[operation] = release;
        self.active = self.active.saturating_add(1);
        if !self.refresh_operation(problem, operation, metrics) {
            return Ok(false);
        }
        self.push(operation, machine, metrics);
        Ok(true)
    }

    fn refresh_machine(
        &mut self,
        problem: &JobShopProblem,
        machine: usize,
        ready_at: i64,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<bool, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        self.machine_ready[machine] = ready_at;
        let wrapped = self.generations[machine] == u64::MAX;
        self.generations[machine] = self.generations[machine].wrapping_add(1);
        for index in 0..self.buckets[machine].len() {
            checkpoint(stop)?;
            metrics.construction_bucket_visits = metrics.construction_bucket_visits.saturating_add(1);
            let operation = self.buckets[machine][index];
            if !self.refresh_operation(problem, operation, metrics) {
                return Ok(false);
            }
            if !wrapped {
                self.push(operation, machine, metrics);
            }
        }
        if wrapped {
            self.rebuild(stop, metrics)?;
        }
        Ok(true)
    }

    fn refresh_operation(&mut self, problem: &JobShopProblem, operation: usize, metrics: &mut ScheduleStateMetrics) -> bool {
        metrics.construction_candidates = metrics.construction_candidates.saturating_add(1);
        let machine = problem.machine(operation);
        let start = self.releases[operation].max(self.machine_ready[machine]).max(problem.start_windows[operation].0);
        let Some(end) = start.checked_add(problem.duration(operation)) else {
            return false;
        };
        if start > problem.start_windows[operation].1 || end > problem.horizons[operation] {
            return false;
        }
        self.starts[operation] = start;
        self.ends[operation] = end;
        true
    }

    fn push(&mut self, operation: usize, machine: usize, metrics: &mut ScheduleStateMetrics) {
        self.heap.push(Reverse((self.ends[operation], operation, machine, self.generations[machine])));
        metrics.construction_heap_pushes = metrics.construction_heap_pushes.saturating_add(1);
        metrics.construction_heap_peak = metrics.construction_heap_peak.max(u64::try_from(self.heap.len()).unwrap_or(u64::MAX));
    }

    fn pop_pivot(
        &mut self,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<Option<(usize, usize, i64)>, ScheduleStateInterrupted> {
        while let Some(Reverse((end, operation, machine, generation))) = self.heap.pop() {
            checkpoint(stop)?;
            let valid = operation < self.positions.len()
                && self.positions[operation] != NO_OPERATION
                && self.generations[machine] == generation
                && self.ends[operation] == end;
            if valid {
                return Ok(Some((operation, machine, end)));
            }
            metrics.construction_stale_pops = metrics.construction_stale_pops.saturating_add(1);
        }
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn select(
        &self,
        problem: &JobShopProblem,
        pivot: usize,
        machine: usize,
        cutoff: i64,
        rule: DispatchRule,
        seed: u64,
        step: usize,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<Option<DispatchCandidate>, ScheduleStateInterrupted> {
        let mut selected = None;
        for &operation in &self.buckets[machine] {
            checkpoint(stop)?;
            metrics.construction_bucket_visits = metrics.construction_bucket_visits.saturating_add(1);
            let start = self.starts[operation];
            if operation != pivot && start >= cutoff {
                continue;
            }
            let candidate = DispatchCandidate { operation, start, end: self.ends[operation] };
            if selected.is_none_or(|current| dispatch_better(problem, rule, seed, step, candidate, current)) {
                selected = Some(candidate);
            }
        }
        Ok(selected)
    }

    fn remove(&mut self, problem: &JobShopProblem, operation: usize) -> bool {
        let Some(position) = self.positions.get(operation).copied().filter(|&position| position != NO_OPERATION) else {
            return false;
        };
        let machine = problem.machine(operation);
        let bucket = &mut self.buckets[machine];
        if position >= bucket.len() || bucket[position] != operation {
            return false;
        }
        bucket.swap_remove(position);
        if let Some(&replacement) = bucket.get(position) {
            self.positions[replacement] = position;
        }
        self.positions[operation] = NO_OPERATION;
        self.active = self.active.saturating_sub(1);
        true
    }

    fn maybe_rebuild(
        &mut self,
        problem: &JobShopProblem,
        stop: &AtomicBool,
        metrics: &mut ScheduleStateMetrics,
    ) -> Result<(), ScheduleStateInterrupted> {
        let bound = self.active.saturating_mul(4).saturating_add(problem.machine_count()).max(64);
        if self.heap.len() > bound {
            self.rebuild(stop, metrics)?;
        }
        Ok(())
    }

    fn rebuild(&mut self, stop: &AtomicBool, metrics: &mut ScheduleStateMetrics) -> Result<(), ScheduleStateInterrupted> {
        self.heap.clear();
        metrics.construction_heap_rebuilds = metrics.construction_heap_rebuilds.saturating_add(1);
        for machine in 0..self.buckets.len() {
            checkpoint(stop)?;
            for index in 0..self.buckets[machine].len() {
                checkpoint(stop)?;
                metrics.construction_bucket_visits = metrics.construction_bucket_visits.saturating_add(1);
                let operation = self.buckets[machine][index];
                self.push(operation, machine, metrics);
            }
        }
        debug_assert_eq!(self.heap.len(), self.active);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct DispatchCandidate {
    operation: usize,
    start: i64,
    end: i64,
}

fn adjusted_remaining_work_score(remaining_work: i64, duration: i64, duration_coefficient: i128) -> i128 {
    i128::from(remaining_work).saturating_sub(i128::from(duration).saturating_mul(duration_coefficient))
}

#[cfg(test)]
pub(crate) fn audit_adjusted_remaining_work_score(remaining_work: i64, duration: i64, duration_coefficient: i128) -> i128 {
    adjusted_remaining_work_score(remaining_work, duration, duration_coefficient)
}

fn dispatch_better(
    problem: &JobShopProblem,
    rule: DispatchRule,
    seed: u64,
    step: usize,
    candidate: DispatchCandidate,
    incumbent: DispatchCandidate,
) -> bool {
    let random_key = |operation: usize| mix64(seed ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ operation as u64);
    let tie = |left: usize, right: usize| (random_key(left), left) < (random_key(right), right);
    match rule {
        DispatchRule::EarliestStart => {
            (candidate.start, random_key(candidate.operation), candidate.operation)
                < (incumbent.start, random_key(incumbent.operation), incumbent.operation)
        }
        DispatchRule::EarliestStartThenMostWorkRemaining => {
            (candidate.start, Reverse(problem.remaining_work[candidate.operation]), random_key(candidate.operation), candidate.operation)
                < (
                    incumbent.start,
                    Reverse(problem.remaining_work[incumbent.operation]),
                    random_key(incumbent.operation),
                    incumbent.operation,
                )
        }
        DispatchRule::EarliestStartThenMostAdjustedWork(lane) => {
            let adjusted_work = |operation: usize| {
                adjusted_remaining_work_score(problem.remaining_work[operation], problem.duration(operation), lane.duration_coefficient())
            };
            (candidate.start, Reverse(adjusted_work(candidate.operation)), random_key(candidate.operation), candidate.operation)
                < (incumbent.start, Reverse(adjusted_work(incumbent.operation)), random_key(incumbent.operation), incumbent.operation)
        }
        DispatchRule::ShortestProcessingTime => {
            (problem.duration(candidate.operation), random_key(candidate.operation), candidate.operation)
                < (problem.duration(incumbent.operation), random_key(incumbent.operation), incumbent.operation)
        }
        DispatchRule::LongestProcessingTime => {
            (
                problem.duration(candidate.operation),
                std::cmp::Reverse(random_key(candidate.operation)),
                std::cmp::Reverse(candidate.operation),
            ) > (
                problem.duration(incumbent.operation),
                std::cmp::Reverse(random_key(incumbent.operation)),
                std::cmp::Reverse(incumbent.operation),
            )
        }
        DispatchRule::MostWorkRemaining => {
            (
                problem.remaining_work[candidate.operation],
                std::cmp::Reverse(random_key(candidate.operation)),
                std::cmp::Reverse(candidate.operation),
            ) > (
                problem.remaining_work[incumbent.operation],
                std::cmp::Reverse(random_key(incumbent.operation)),
                std::cmp::Reverse(incumbent.operation),
            )
        }
        DispatchRule::Randomized => tie(candidate.operation, incumbent.operation),
    }
}

fn initialize_machine_graph(
    problem: &JobShopProblem,
    machine_sequences: &[Vec<usize>],
    machine_predecessors: &mut [usize],
    machine_successors: &mut [usize],
    positions: &mut [usize],
    stop: &AtomicBool,
) -> Result<bool, ScheduleStateInterrupted> {
    machine_predecessors.fill(NO_OPERATION);
    machine_successors.fill(NO_OPERATION);
    positions.fill(NO_OPERATION);
    for (machine, sequence) in machine_sequences.iter().enumerate() {
        checkpoint(stop)?;
        for (position, &operation) in sequence.iter().enumerate() {
            checkpoint(stop)?;
            if operation >= problem.operation_count() || problem.machine(operation) != machine || positions[operation] != NO_OPERATION {
                return Ok(false);
            }
            positions[operation] = position;
            if position > 0 {
                machine_predecessors[operation] = sequence[position - 1];
            }
            if position + 1 < sequence.len() {
                machine_successors[operation] = sequence[position + 1];
            }
        }
    }
    Ok(positions.iter().all(|&position| position != NO_OPERATION))
}

fn refresh_machine_links(
    sequence: &[usize],
    first_changed_position: usize,
    last_changed_position: usize,
    machine_predecessors: &mut [usize],
    machine_successors: &mut [usize],
    positions: &mut [usize],
    mut changed_roots: Option<&mut Vec<usize>>,
) {
    debug_assert!(!sequence.is_empty());
    debug_assert!(first_changed_position <= last_changed_position);
    debug_assert!(last_changed_position < sequence.len());
    let first = first_changed_position.saturating_sub(1);
    let last = last_changed_position.saturating_add(1).min(sequence.len() - 1);
    for (position, &operation) in sequence.iter().enumerate().take(last + 1).skip(first) {
        let predecessor = if position == 0 { NO_OPERATION } else { sequence[position - 1] };
        if machine_predecessors[operation] != predecessor {
            if let Some(roots) = changed_roots.as_mut() {
                roots.push(operation);
            }
            machine_predecessors[operation] = predecessor;
        }
        positions[operation] = position;
        machine_successors[operation] = if position + 1 == sequence.len() { NO_OPERATION } else { sequence[position + 1] };
    }
}

fn mark_dirty(marks: &mut [u32], queue: &mut Vec<usize>, indegrees: &mut [usize], epoch: u32, operation: usize) {
    if marks[operation] != epoch {
        marks[operation] = epoch;
        indegrees[operation] = 0;
        queue.push(operation);
    }
}

fn move_is_valid(machine_sequences: &[Vec<usize>], movement: ScheduleMove) -> bool {
    match movement {
        ScheduleMove::AdjacentSwap { machine, first_position } => machine_sequences
            .get(machine)
            .and_then(|sequence| first_position.checked_add(1).map(|second| second < sequence.len()))
            .unwrap_or(false),
        ScheduleMove::Insert { machine, from, to } => {
            machine_sequences.get(machine).is_some_and(|sequence| from < sequence.len() && to < sequence.len() && from != to)
        }
    }
}

fn changed_arc_indices(movement: ScheduleMove) -> ([Option<usize>; 3], [Option<usize>; 3]) {
    match movement {
        ScheduleMove::AdjacentSwap { first_position, .. } => {
            let indices = [first_position.checked_sub(1), Some(first_position), first_position.checked_add(1)];
            (indices, indices)
        }
        ScheduleMove::Insert { from, to, .. } if from < to => {
            ([from.checked_sub(1), Some(from), Some(to)], [from.checked_sub(1), to.checked_sub(1), Some(to)])
        }
        ScheduleMove::Insert { from, to, .. } => {
            ([to.checked_sub(1), from.checked_sub(1), Some(from)], [to.checked_sub(1), Some(to), Some(from)])
        }
    }
}

fn sequence_arc(sequence: &[usize], machine: usize, first_position: usize) -> Option<MachineArc> {
    let second_position = first_position.checked_add(1)?;
    Some(MachineArc { machine, before: *sequence.get(first_position)?, after: *sequence.get(second_position)? })
}

fn moved_sequence_arc(sequence: &[usize], movement: ScheduleMove, machine: usize, first_position: usize) -> Option<MachineArc> {
    let second_position = first_position.checked_add(1)?;
    Some(MachineArc {
        machine,
        before: moved_operation_at(sequence, movement, first_position)?,
        after: moved_operation_at(sequence, movement, second_position)?,
    })
}

fn moved_operation_at(sequence: &[usize], movement: ScheduleMove, position: usize) -> Option<usize> {
    if position >= sequence.len() {
        return None;
    }
    match movement {
        ScheduleMove::AdjacentSwap { first_position, .. } => {
            let second_position = first_position.checked_add(1)?;
            if position == first_position {
                sequence.get(second_position).copied()
            } else if position == second_position {
                sequence.get(first_position).copied()
            } else {
                sequence.get(position).copied()
            }
        }
        ScheduleMove::Insert { from, to, .. } if from < to => {
            if position < from || position > to {
                sequence.get(position).copied()
            } else if position == to {
                sequence.get(from).copied()
            } else {
                sequence.get(position + 1).copied()
            }
        }
        ScheduleMove::Insert { from, to, .. } => {
            if position < to || position > from {
                sequence.get(position).copied()
            } else if position == to {
                sequence.get(from).copied()
            } else {
                sequence.get(position - 1).copied()
            }
        }
    }
}

fn push_unique_arc(arcs: &mut [Option<MachineArc>; 3], arc: MachineArc) {
    if arcs.contains(&Some(arc)) {
        return;
    }
    if let Some(slot) = arcs.iter_mut().find(|slot| slot.is_none()) {
        *slot = Some(arc);
    } else {
        debug_assert!(false, "a structured schedule move changed more than three machine arcs");
    }
}

fn arc_difference(arcs: [Option<MachineArc>; 3], other: &[Option<MachineArc>; 3]) -> [Option<MachineArc>; 3] {
    let mut difference = [None; 3];
    for arc in arcs.into_iter().flatten() {
        if !other.contains(&Some(arc)) {
            push_unique_arc(&mut difference, arc);
        }
    }
    difference
}

fn apply_sequence_move(machine_sequences: &mut [Vec<usize>], movement: ScheduleMove) -> bool {
    match movement {
        ScheduleMove::AdjacentSwap { machine, first_position } => {
            let Some(sequence) = machine_sequences.get_mut(machine) else {
                return false;
            };
            let Some(second_position) = first_position.checked_add(1) else {
                return false;
            };
            if second_position >= sequence.len() {
                return false;
            }
            sequence.swap(first_position, second_position);
            true
        }
        ScheduleMove::Insert { machine, from, to } => {
            let Some(sequence) = machine_sequences.get_mut(machine) else {
                return false;
            };
            if from >= sequence.len() || to >= sequence.len() || from == to {
                return false;
            }
            let operation = sequence.remove(from);
            sequence.insert(to, operation);
            true
        }
    }
}

fn undo_sequence_move(machine_sequences: &mut [Vec<usize>], movement: ScheduleMove) {
    match movement {
        ScheduleMove::AdjacentSwap { machine, first_position } => {
            machine_sequences[machine].swap(first_position, first_position + 1);
        }
        ScheduleMove::Insert { machine, from, to } => {
            let operation = machine_sequences[machine].remove(to);
            machine_sequences[machine].insert(from, operation);
        }
    }
}
