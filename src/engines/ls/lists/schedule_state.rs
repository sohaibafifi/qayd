//! Compact disjunctive state for strict job-shop schedules.
//!
//! The semantic schedule remains the source of truth. This module recognizes a
//! deliberately narrow, safe subset, represents each unary machine by an
//! explicit operation order, and reconstructs a complete schedule by longest
//! path. Unsupported schedules stay on the general scheduling fallback.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::engines::ls::schedule_ir::PrecedenceDag;
use crate::mix64;
use crate::model::list::{CollectionSolution, Resource, Schedule};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScheduleStateInterrupted;

fn checkpoint(stop: &AtomicBool) -> Result<(), ScheduleStateInterrupted> {
    if stop.load(Ordering::Acquire) {
        Err(ScheduleStateInterrupted)
    } else {
        Ok(())
    }
}

/// Generic dispatch rules for the Giffler-Thompson constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DispatchRule {
    EarliestStart,
    ShortestProcessingTime,
    LongestProcessingTime,
    MostWorkRemaining,
    Randomized,
}

impl DispatchRule {
    pub(crate) const ALL: [Self; 5] =
        [Self::EarliestStart, Self::ShortestProcessingTime, Self::LongestProcessingTime, Self::MostWorkRemaining, Self::Randomized];
}

/// Validated strict job-shop data shared by constructors and states.
#[derive(Clone)]
pub(crate) struct JobShopProblem {
    durations: Vec<i64>,
    horizons: Vec<i64>,
    start_windows: Vec<(i64, i64)>,
    operation_machines: Vec<usize>,
    machine_count: usize,
    precedences: PrecedenceDag,
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
                if interval.duration <= 0 || interval.horizon < interval.duration {
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
                if mode.duration <= 0 || start_min < 0 || start_min > start_max || start_max > last_start {
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
        for list in &mut successors {
            checkpoint(stop)?;
            list.sort_unstable();
            checkpoint(stop)?;
            list.dedup();
        }
        let Some(precedences) = PrecedenceDag::compile(successors, stop) else {
            checkpoint(stop)?;
            return Ok(None);
        };
        let Some(remaining_work) = precedences.remaining_paths(&durations, stop) else {
            checkpoint(stop)?;
            return Ok(None);
        };

        let mut horizons = Vec::with_capacity(operation_count);
        for interval in &schedule.intervals {
            checkpoint(stop)?;
            horizons.push(interval.horizon);
        }
        Ok(Some(Self {
            durations,
            horizons,
            start_windows,
            operation_machines,
            machine_count,
            precedences,
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

    pub(crate) fn machine(&self, operation: usize) -> usize {
        self.operation_machines[operation]
    }
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

/// A maximal contiguous machine segment on the selected critical path.
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
    topological: Vec<usize>,
    makespan: i64,
    critical_path: Vec<usize>,
    critical_blocks: Vec<CriticalBlock>,
}

impl Reconstruction {
    fn empty(operation_count: usize) -> Self {
        Self {
            starts: vec![0; operation_count],
            ends: vec![0; operation_count],
            latest_starts: vec![0; operation_count],
            topological: Vec::with_capacity(operation_count),
            makespan: 0,
            critical_path: Vec::with_capacity(operation_count),
            critical_blocks: Vec::with_capacity(operation_count),
        }
    }

    fn commit(&mut self, workspace: &EvaluationWorkspace) {
        self.starts.copy_from_slice(&workspace.trial_starts);
        self.ends.copy_from_slice(&workspace.trial_ends);
        self.latest_starts.copy_from_slice(&workspace.trial_latest_starts);
        self.topological.clear();
        self.topological.extend_from_slice(&workspace.trial_topological);
        self.makespan = workspace.trial_makespan;
        self.critical_path.clear();
        self.critical_path.extend_from_slice(&workspace.trial_critical_path);
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
    pub(crate) critical_blocks: usize,
}

struct EvaluationWorkspace {
    indegrees: Vec<usize>,
    ready: Vec<usize>,
    trial_starts: Vec<i64>,
    trial_ends: Vec<i64>,
    trial_latest_starts: Vec<i64>,
    trial_topological: Vec<usize>,
    trial_makespan: i64,
    trial_critical_path: Vec<usize>,
    trial_critical_blocks: Vec<CriticalBlock>,
    changed_roots: Vec<usize>,
    dirty_marks: Vec<u32>,
    dirty_epoch: u32,
    dirty_queue: Vec<usize>,
    dirty_topological: Vec<usize>,
    patched_operations: Vec<usize>,
    patched_starts: Vec<i64>,
    patched_ends: Vec<i64>,
    observed_capacities: [usize; 10],
    growths: u64,
}

impl EvaluationWorkspace {
    fn new(operation_count: usize) -> Self {
        let mut workspace = Self {
            indegrees: vec![0; operation_count],
            ready: Vec::with_capacity(operation_count),
            trial_starts: vec![0; operation_count],
            trial_ends: vec![0; operation_count],
            trial_latest_starts: vec![0; operation_count],
            trial_topological: Vec::with_capacity(operation_count),
            trial_makespan: 0,
            trial_critical_path: Vec::with_capacity(operation_count),
            trial_critical_blocks: Vec::with_capacity(operation_count),
            changed_roots: Vec::with_capacity(operation_count),
            dirty_marks: vec![0; operation_count],
            dirty_epoch: 0,
            dirty_queue: Vec::with_capacity(operation_count),
            dirty_topological: Vec::with_capacity(operation_count),
            patched_operations: Vec::with_capacity(operation_count),
            patched_starts: Vec::with_capacity(operation_count),
            patched_ends: Vec::with_capacity(operation_count),
            observed_capacities: [0; 10],
            growths: 0,
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
            critical_blocks: self.trial_critical_blocks.capacity(),
        }
    }

    fn capacity_snapshot(&self) -> [usize; 10] {
        [
            self.ready.capacity(),
            self.trial_topological.capacity(),
            self.trial_critical_path.capacity(),
            self.trial_critical_blocks.capacity(),
            self.changed_roots.capacity(),
            self.dirty_queue.capacity(),
            self.dirty_topological.capacity(),
            self.patched_operations.capacity(),
            self.patched_starts.capacity(),
            self.patched_ends.capacity(),
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
            self.dirty_epoch = 1;
        }
        self.dirty_epoch
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
        machine_predecessors: &[usize],
        machine_successors: &[usize],
        epoch: u32,
        stop: &AtomicBool,
    ) -> Result<(), ReconstructionFailure> {
        self.ready.clear();
        self.dirty_topological.clear();
        for index in 0..self.dirty_queue.len() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let operation = self.dirty_queue[index];
            let mut indegree =
                problem.precedences.predecessors(operation).iter().filter(|&&predecessor| self.dirty_marks[predecessor] == epoch).count();
            let machine_predecessor = machine_predecessors[operation];
            if machine_predecessor != NO_OPERATION
                && self.dirty_marks[machine_predecessor] == epoch
                && !problem.precedences.predecessors(operation).contains(&machine_predecessor)
            {
                indegree = indegree.checked_add(1).ok_or(ReconstructionFailure::Numeric)?;
            }
            self.indegrees[operation] = indegree;
            if indegree == 0 {
                heap_push(&mut self.ready, operation);
            }
        }

        while let Some(operation) = heap_pop(&mut self.ready) {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            self.dirty_topological.push(operation);
            for &successor in problem.precedences.successors(operation) {
                if self.dirty_marks[successor] == epoch {
                    release_topological_successor(&mut self.indegrees, &mut self.ready, successor)?;
                }
            }
            let machine_successor = machine_successors[operation];
            if machine_successor != NO_OPERATION
                && self.dirty_marks[machine_successor] == epoch
                && !problem.precedences.successors(operation).contains(&machine_successor)
            {
                release_topological_successor(&mut self.indegrees, &mut self.ready, machine_successor)?;
            }
        }
        if self.dirty_topological.len() != self.dirty_queue.len() {
            return Err(ReconstructionFailure::Cycle);
        }
        Ok(())
    }

    fn build_analysis(
        &mut self,
        problem: &JobShopProblem,
        machine_predecessors: &[usize],
        machine_successors: &[usize],
        positions: &[usize],
        stop: &AtomicBool,
    ) -> Result<(), ReconstructionFailure> {
        for &operation in self.trial_topological.iter().rev() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let Some(makespan_latest) = self.trial_makespan.checked_sub(problem.duration(operation)) else {
                return Err(ReconstructionFailure::Numeric);
            };
            let mut latest = problem.start_windows[operation].1.min(makespan_latest);
            for &successor in problem.precedences.successors(operation) {
                latest = latest_before_successor(latest, problem.duration(operation), self.trial_latest_starts[successor])?;
            }
            let machine_successor = machine_successors[operation];
            if machine_successor != NO_OPERATION && !problem.precedences.successors(operation).contains(&machine_successor) {
                latest = latest_before_successor(latest, problem.duration(operation), self.trial_latest_starts[machine_successor])?;
            }
            if latest < self.trial_starts[operation] {
                return Err(ReconstructionFailure::Window);
            }
            self.trial_latest_starts[operation] = latest;
        }

        self.trial_critical_path.clear();
        let mut terminal = None;
        for &operation in &self.trial_topological {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            if self.trial_ends[operation] == self.trial_makespan {
                let machine_predecessor = machine_predecessors[operation];
                let has_tight_machine_predecessor =
                    machine_predecessor != NO_OPERATION && self.trial_ends[machine_predecessor] == self.trial_starts[operation];
                let key = (has_tight_machine_predecessor, operation);
                if terminal.is_none_or(|old_key| key > old_key) {
                    terminal = Some(key);
                }
            }
        }
        if let Some((_, mut operation)) = terminal {
            loop {
                checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
                self.trial_critical_path.push(operation);
                let machine_predecessor = machine_predecessors[operation];
                let mut predecessor_choice = None;
                for &predecessor in problem.precedences.predecessors(operation) {
                    checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
                    if self.trial_ends[predecessor] == self.trial_starts[operation] {
                        let key = (predecessor == machine_predecessor, predecessor);
                        if predecessor_choice.is_none_or(|old_key| key > old_key) {
                            predecessor_choice = Some(key);
                        }
                    }
                }
                if machine_predecessor != NO_OPERATION
                    && !problem.precedences.predecessors(operation).contains(&machine_predecessor)
                    && self.trial_ends[machine_predecessor] == self.trial_starts[operation]
                {
                    let key = (true, machine_predecessor);
                    if predecessor_choice.is_none_or(|old_key| key > old_key) {
                        predecessor_choice = Some(key);
                    }
                }
                let Some((_, predecessor)) = predecessor_choice else {
                    break;
                };
                operation = predecessor;
                if self.trial_critical_path.len() > problem.operation_count() {
                    return Err(ReconstructionFailure::Cycle);
                }
            }
            self.trial_critical_path.reverse();
        }

        self.trial_critical_blocks.clear();
        let mut path_start = 0usize;
        while path_start < self.trial_critical_path.len() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let machine = problem.machine(self.trial_critical_path[path_start]);
            let first_position = positions[self.trial_critical_path[path_start]];
            let mut path_end = path_start;
            while path_end + 1 < self.trial_critical_path.len()
                && problem.machine(self.trial_critical_path[path_end + 1]) == machine
                && positions[self.trial_critical_path[path_end + 1]] == positions[self.trial_critical_path[path_end]] + 1
            {
                checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
                path_end += 1;
            }
            if path_end > path_start {
                self.trial_critical_blocks.push(CriticalBlock {
                    machine,
                    first_position,
                    last_position: positions[self.trial_critical_path[path_end]],
                });
            }
            path_start = path_end + 1;
        }
        Ok(())
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
pub(crate) enum MoveAcceptance {
    Improving,
    NonWorsening,
    Always,
}

impl MoveAcceptance {
    fn accepts(self, current: i64, candidate: i64) -> bool {
        match self {
            Self::Improving => candidate < current,
            Self::NonWorsening => candidate <= current,
            Self::Always => true,
        }
    }
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScheduleStateMetrics {
    pub(crate) construction_candidates: u64,
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
}

/// Complete compact state. Machine orders are authoritative and every accepted
/// order has a matching, fully reconstructed schedule.
pub(crate) struct JobShopState {
    problem: JobShopProblem,
    machine_sequences: Vec<Vec<usize>>,
    machine_predecessors: Vec<usize>,
    machine_successors: Vec<usize>,
    positions: Vec<usize>,
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
                if problem.machine(operation) != pivot_machine || start >= cutoff {
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
        match workspace.build_analysis(problem, &machine_predecessors, &machine_successors, &positions, stop) {
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
            reconstruction,
            workspace,
            metrics: *metrics,
        }))
    }

    pub(crate) fn makespan(&self) -> i64 {
        self.reconstruction.makespan
    }

    pub(crate) fn starts(&self) -> &[i64] {
        &self.reconstruction.starts
    }

    pub(crate) fn latest_starts(&self) -> &[i64] {
        &self.reconstruction.latest_starts
    }

    pub(crate) fn machine_sequences(&self) -> &[Vec<usize>] {
        &self.machine_sequences
    }

    pub(crate) fn topological_order(&self) -> &[usize] {
        &self.reconstruction.topological
    }

    pub(crate) fn critical_path(&self) -> &[usize] {
        &self.reconstruction.critical_path
    }

    pub(crate) fn critical_blocks(&self) -> &[CriticalBlock] {
        &self.reconstruction.critical_blocks
    }

    pub(crate) fn metrics(&self) -> ScheduleStateMetrics {
        let mut metrics = self.metrics;
        metrics.workspace_growths = self.workspace.growths;
        metrics
    }

    pub(crate) fn workspace_capacities(&self) -> ScheduleWorkspaceCapacities {
        self.workspace.capacities()
    }

    pub(crate) fn critical_moves(
        &self,
        neighborhood: CriticalNeighborhood,
        stop: &AtomicBool,
    ) -> Result<Vec<ScheduleMove>, ScheduleStateInterrupted> {
        let mut movements = Vec::with_capacity(self.problem.operation_count().saturating_mul(2));
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

    /// Build a duplicate-free union when several nested neighborhoods are
    /// scanned at the same incumbent.
    pub(crate) fn fill_critical_move_union(
        &self,
        neighborhoods: &[CriticalNeighborhood],
        movements: &mut Vec<ScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleStateInterrupted> {
        checkpoint(stop)?;
        movements.clear();
        let capacity_factor = if neighborhoods.len() > 1 { 3 } else { 2 };
        let target_capacity = self.problem.operation_count().saturating_mul(capacity_factor);
        if movements.capacity() < target_capacity {
            movements.reserve(target_capacity);
        }
        for &neighborhood in neighborhoods {
            for &block in &self.reconstruction.critical_blocks {
                checkpoint(stop)?;
                match neighborhood {
                    CriticalNeighborhood::N1 => {
                        for first_position in block.first_position..block.last_position {
                            checkpoint(stop)?;
                            movements.push(ScheduleMove::AdjacentSwap { machine: block.machine, first_position });
                        }
                    }
                    CriticalNeighborhood::N5 | CriticalNeighborhood::N6 => {
                        movements.push(ScheduleMove::AdjacentSwap { machine: block.machine, first_position: block.first_position });
                        if block.last_position > block.first_position + 1 {
                            movements.push(ScheduleMove::AdjacentSwap { machine: block.machine, first_position: block.last_position - 1 });
                        }
                        if neighborhood == CriticalNeighborhood::N6 && block.len() >= 3 {
                            for from in (block.first_position + 1)..block.last_position {
                                checkpoint(stop)?;
                                if from > block.first_position + 1 {
                                    movements.push(ScheduleMove::Insert { machine: block.machine, from, to: block.first_position });
                                }
                                if from + 1 < block.last_position {
                                    movements.push(ScheduleMove::Insert { machine: block.machine, from, to: block.last_position });
                                }
                            }
                        }
                    }
                }
            }
        }
        movements.sort_unstable();
        movements.dedup();
        Ok(())
    }

    /// Reconstruct a move into temporary buffers and commit it only when it is
    /// feasible and accepted. Every other outcome restores the exact machine
    /// order before returning, including cancellation.
    pub(crate) fn consider_move(
        &mut self,
        movement: ScheduleMove,
        acceptance: MoveAcceptance,
        stop: &AtomicBool,
    ) -> Result<MoveOutcome, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        let result = self.consider_move_inner(movement, acceptance, stop);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        result
    }

    fn consider_move_inner(
        &mut self,
        movement: ScheduleMove,
        acceptance: MoveAcceptance,
        stop: &AtomicBool,
    ) -> Result<MoveOutcome, ScheduleStateInterrupted> {
        checkpoint(stop)?;
        self.metrics.moves_considered = self.metrics.moves_considered.saturating_add(1);
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

        if let Err(failure) =
            self.workspace.build_analysis(&self.problem, &self.machine_predecessors, &self.machine_successors, &self.positions, stop)
        {
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

    fn evaluate_delta(&mut self, stop: &AtomicBool) -> Result<i64, ReconstructionFailure> {
        self.clear_date_patch();
        let epoch = self.workspace.next_dirty_epoch();
        self.workspace.dirty_queue.clear();
        for &operation in &self.workspace.changed_roots {
            mark_dirty(&mut self.workspace.dirty_marks, &mut self.workspace.dirty_queue, epoch, operation);
        }
        let mut cursor = 0usize;
        while cursor < self.workspace.dirty_queue.len() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let operation = self.workspace.dirty_queue[cursor];
            cursor += 1;
            for &successor in self.problem.precedences.successors(operation) {
                mark_dirty(&mut self.workspace.dirty_marks, &mut self.workspace.dirty_queue, epoch, successor);
            }
            let machine_successor = self.machine_successors[operation];
            if machine_successor != NO_OPERATION {
                mark_dirty(&mut self.workspace.dirty_marks, &mut self.workspace.dirty_queue, epoch, machine_successor);
            }
        }
        let dirty = u64::try_from(self.workspace.dirty_queue.len()).unwrap_or(u64::MAX);
        self.metrics.dirty_cone_operations = self.metrics.dirty_cone_operations.saturating_add(dirty);
        self.metrics.max_dirty_cone = self.metrics.max_dirty_cone.max(dirty);

        let operation_count = self.problem.operation_count();
        if self.workspace.dirty_queue.len() == operation_count {
            self.metrics.full_fallbacks = self.metrics.full_fallbacks.saturating_add(1);
        }
        let rebuild_full_topology = self.workspace.dirty_queue.len().saturating_mul(4) >= operation_count.saturating_mul(3);
        if rebuild_full_topology {
            self.metrics.topological_rebuilds = self.metrics.topological_rebuilds.saturating_add(1);
            self.workspace.rebuild_topology(&self.problem, &self.machine_predecessors, &self.machine_successors, stop)?;
        } else {
            self.workspace.rebuild_dirty_topology(&self.problem, &self.machine_predecessors, &self.machine_successors, epoch, stop)?;
        }

        let topological_len =
            if rebuild_full_topology { self.workspace.trial_topological.len() } else { self.workspace.dirty_topological.len() };
        for index in 0..topological_len {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let operation =
                if rebuild_full_topology { self.workspace.trial_topological[index] } else { self.workspace.dirty_topological[index] };
            if self.workspace.dirty_marks[operation] != epoch {
                continue;
            }
            let (start, end) = earliest_dates(&self.problem, &self.machine_predecessors, &self.reconstruction.ends, operation)?;
            if start != self.reconstruction.starts[operation] || end != self.reconstruction.ends[operation] {
                self.workspace.patched_operations.push(operation);
                self.workspace.patched_starts.push(self.reconstruction.starts[operation]);
                self.workspace.patched_ends.push(self.reconstruction.ends[operation]);
                self.reconstruction.starts[operation] = start;
                self.reconstruction.ends[operation] = end;
            }
        }
        Ok(self.reconstruction.ends.iter().copied().max().unwrap_or(0))
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
        match failure {
            ReconstructionFailure::Interrupted => Err(ScheduleStateInterrupted),
            ReconstructionFailure::Cycle => {
                self.metrics.cycle_rejections = self.metrics.cycle_rejections.saturating_add(1);
                Ok(MoveOutcome::Rejected(MoveRejection::Cycle))
            }
            ReconstructionFailure::Window => {
                self.metrics.window_rejections = self.metrics.window_rejections.saturating_add(1);
                Ok(MoveOutcome::Rejected(MoveRejection::Window))
            }
            ReconstructionFailure::Numeric => Ok(MoveOutcome::Rejected(MoveRejection::Numeric)),
        }
    }

    /// Rebuild the accepted order from scratch into the reusable oracle buffers.
    pub(crate) fn matches_full_oracle(&mut self, stop: &AtomicBool) -> Result<bool, ScheduleStateInterrupted> {
        let result = self.matches_full_oracle_inner(stop);
        self.workspace.observe_growths();
        self.metrics.workspace_growths = self.workspace.growths;
        result
    }

    fn matches_full_oracle_inner(&mut self, stop: &AtomicBool) -> Result<bool, ScheduleStateInterrupted> {
        let result = self.run_full_evaluation(stop);
        let makespan = match result {
            Ok(makespan) => makespan,
            Err(ReconstructionFailure::Interrupted) => return Err(ScheduleStateInterrupted),
            Err(_) => return Ok(false),
        };
        self.metrics.oracle_validations = self.metrics.oracle_validations.saturating_add(1);
        match self.workspace.build_analysis(&self.problem, &self.machine_predecessors, &self.machine_successors, &self.positions, stop) {
            Ok(()) => {}
            Err(ReconstructionFailure::Interrupted) => return Err(ScheduleStateInterrupted),
            Err(_) => return Ok(false),
        }
        Ok(makespan == self.reconstruction.makespan
            && self.workspace.trial_starts == self.reconstruction.starts
            && self.workspace.trial_ends == self.reconstruction.ends
            && self.workspace.trial_latest_starts == self.reconstruction.latest_starts
            && self.workspace.trial_topological == self.reconstruction.topological
            && self.workspace.trial_critical_path == self.reconstruction.critical_path
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

#[derive(Clone, Copy)]
struct DispatchCandidate {
    operation: usize,
    start: i64,
    end: i64,
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

fn mark_dirty(marks: &mut [u32], queue: &mut Vec<usize>, epoch: u32, operation: usize) {
    if marks[operation] != epoch {
        marks[operation] = epoch;
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
