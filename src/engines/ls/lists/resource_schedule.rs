//! Compact resource-constrained scheduling state.
//!
//! This module deliberately recognizes only mandatory, fixed-duration RCPSP
//! schedules. A complete topological priority list is authoritative. Initial
//! serial and parallel schedule-generation uses sparse event profiles, so no
//! array proportional to the horizon is required. Serial moves update only the
//! changed priority suffix with event rollback; parallel moves conservatively
//! reuse a full preallocated workspace. Every provisional acceptance is checked
//! by an independent full reconstruction before it is committed.

use std::collections::BTreeMap;
use std::ops::Bound::{Excluded, Unbounded};
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::engines::ls::schedule_ir::PrecedenceDag;
use crate::mix64;
use crate::model::list::{CollectionSolution, Resource, Schedule};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResourceScheduleInterrupted;

fn checkpoint(stop: &AtomicBool) -> Result<(), ResourceScheduleInterrupted> {
    if stop.load(Ordering::Acquire) {
        Err(ResourceScheduleInterrupted)
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct RenewableResource {
    capacity: i128,
    event_capacity: usize,
}

#[derive(Debug)]
pub(crate) struct ResourceScheduleProblemData {
    durations: Vec<i64>,
    earliest_starts: Vec<i64>,
    latest_starts: Vec<i64>,
    machines: Vec<i64>,
    modes: Vec<Option<usize>>,
    precedences: PrecedenceDag,
    resources: Vec<RenewableResource>,
    activity_resources: Vec<Vec<(usize, i128)>>,
}

/// Validated mandatory, fixed-duration RCPSP data.
///
/// A search state owns only this small handle. The immutable compiled problem
/// is shared by all states derived from that handle, while mutable priorities,
/// dates, and event profiles remain trajectory-local.
#[derive(Clone, Debug)]
pub(crate) struct ResourceScheduleProblem {
    data: Arc<ResourceScheduleProblemData>,
}

impl Deref for ResourceScheduleProblem {
    type Target = ResourceScheduleProblemData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl ResourceScheduleProblem {
    /// Recognize the strict RCPSP subset handled by this compact state.
    ///
    /// Optional activities, alternative mode choices, unary resources, missing
    /// makespan objectives, malformed/cyclic precedences, and schedules without
    /// a cumulative resource remain on the general scheduling fallback. An
    /// interval with exactly one mode is a fixed decision and is normalized
    /// while preserving its window, machine, and semantic mode identity.
    pub(crate) fn recognize(schedule: &Schedule, stop: &AtomicBool) -> Result<Option<Self>, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        let activity_count = schedule.intervals.len();
        if activity_count == 0 || !schedule.minimize_makespan || schedule.resources.is_empty() {
            return Ok(None);
        }

        let mut durations = Vec::with_capacity(activity_count);
        let mut earliest_starts = Vec::with_capacity(activity_count);
        let mut latest_starts = Vec::with_capacity(activity_count);
        let mut machines = Vec::with_capacity(activity_count);
        let mut modes = Vec::with_capacity(activity_count);
        for interval in &schedule.intervals {
            checkpoint(stop)?;
            if interval.optional {
                return Ok(None);
            }
            match interval.modes.as_slice() {
                [] => {
                    if interval.duration < 0 || interval.duration > interval.horizon {
                        return Ok(None);
                    }
                    durations.push(interval.duration);
                    earliest_starts.push(0);
                    latest_starts.push(interval.horizon - interval.duration);
                    machines.push(-1);
                    modes.push(None);
                }
                [mode] => {
                    let Ok(machine) = i64::try_from(mode.machine) else {
                        return Ok(None);
                    };
                    if mode.duration < 0
                        || mode.start_window.0 < 0
                        || mode.start_window.0 > mode.start_window.1
                        || mode.start_window.1.checked_add(mode.duration).is_none_or(|end| end > interval.horizon)
                    {
                        return Ok(None);
                    }
                    durations.push(mode.duration);
                    earliest_starts.push(mode.start_window.0);
                    latest_starts.push(mode.start_window.1);
                    machines.push(machine);
                    modes.push(mode.reference);
                }
                _ => return Ok(None),
            }
        }

        let mut successors = vec![Vec::new(); activity_count];
        for &(before, after) in &schedule.precedences {
            checkpoint(stop)?;
            let Some(list) = successors.get_mut(before) else {
                return Ok(None);
            };
            if after >= activity_count {
                return Ok(None);
            }
            list.push(after);
        }
        // Repeated semantic precedences are idempotent. Canonicalize them
        // before compiling the physical DAG, whose input contract is unique.
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

        let mut resources = Vec::with_capacity(schedule.resources.len());
        let mut activity_resources = vec![Vec::new(); activity_count];
        for resource in &schedule.resources {
            checkpoint(stop)?;
            let Resource::Cumulative { demands, capacity } = resource else {
                return Ok(None);
            };
            if *capacity < 0 {
                return Ok(None);
            }
            let mut aggregated = vec![0i128; activity_count];
            for &(activity, demand) in demands {
                checkpoint(stop)?;
                let Some(value) = aggregated.get_mut(activity) else {
                    return Ok(None);
                };
                if demand < 0 {
                    return Ok(None);
                }
                let Some(sum) = value.checked_add(i128::from(demand)) else {
                    return Ok(None);
                };
                *value = sum;
            }
            let resource_index = resources.len();
            let mut event_capacity = 0usize;
            for (activity, &demand) in aggregated.iter().enumerate() {
                checkpoint(stop)?;
                if demand > 0 {
                    activity_resources[activity].push((resource_index, demand));
                    if durations[activity] > 0 {
                        event_capacity = event_capacity.saturating_add(2);
                    }
                }
            }
            resources.push(RenewableResource { capacity: i128::from(*capacity), event_capacity });
        }

        checkpoint(stop)?;
        Ok(Some(Self {
            data: Arc::new(ResourceScheduleProblemData {
                durations,
                earliest_starts,
                latest_starts,
                machines,
                modes,
                precedences,
                resources,
                activity_resources,
            }),
        }))
    }

    pub(crate) fn activity_count(&self) -> usize {
        self.durations.len()
    }

    pub(crate) fn resource_count(&self) -> usize {
        self.resources.len()
    }

    pub(crate) fn duration(&self, activity: usize) -> i64 {
        self.durations[activity]
    }

    pub(crate) fn latest_start(&self, activity: usize) -> i64 {
        self.latest_starts[activity]
    }

    pub(crate) fn earliest_start(&self, activity: usize) -> i64 {
        self.earliest_starts[activity]
    }

    pub(crate) fn demand(&self, resource: usize, activity: usize) -> i128 {
        self.activity_resources
            .get(activity)
            .and_then(|demands| demands.iter().find_map(|&(candidate, demand)| (candidate == resource).then_some(demand)))
            .unwrap_or(0)
    }

    pub(crate) fn capacity(&self, resource: usize) -> i128 {
        self.resources[resource].capacity
    }

    pub(crate) fn topological_order(&self) -> &[usize] {
        self.precedences.topological()
    }

    pub(crate) fn demand_entry_count(&self) -> usize {
        self.activity_resources.iter().map(Vec::len).sum()
    }

    pub(crate) fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

/// A complete precedence-feasible activity priority list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrioritySgs {
    order: Vec<usize>,
    positions: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PriorityRule {
    Stable,
    ShortestDuration,
    LongestDuration,
    MostSuccessors,
    Randomized,
}

impl PriorityRule {
    pub(crate) const ALL: [Self; 5] = [Self::Stable, Self::ShortestDuration, Self::LongestDuration, Self::MostSuccessors, Self::Randomized];
}

impl PrioritySgs {
    pub(crate) fn stable(problem: &ResourceScheduleProblem) -> Self {
        let order = problem.precedences.topological().to_vec();
        let positions = positions(&order);
        Self { order, positions }
    }

    /// Build a complete topological priority permutation using a generic
    /// dispatch rule. Every seed-derived tie is stable across runs.
    pub(crate) fn dispatch(
        problem: &ResourceScheduleProblem,
        seed: u64,
        rule: PriorityRule,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        let count = problem.activity_count();
        let mut indegrees = (0..count).map(|activity| problem.precedences.predecessors(activity).len()).collect::<Vec<_>>();
        let mut selected = vec![false; count];
        let mut order = Vec::with_capacity(count);
        while order.len() < count {
            checkpoint(stop)?;
            let step = order.len();
            let mut choice = None;
            for activity in 0..count {
                checkpoint(stop)?;
                if selected[activity] || indegrees[activity] != 0 {
                    continue;
                }
                if choice.is_none_or(|current| dispatch_better(problem, seed, step, rule, activity, current)) {
                    choice = Some(activity);
                }
            }
            let Some(activity) = choice else {
                return Ok(None);
            };
            selected[activity] = true;
            order.push(activity);
            for &successor in problem.precedences.successors(activity) {
                checkpoint(stop)?;
                let Some(updated) = indegrees[successor].checked_sub(1) else {
                    return Ok(None);
                };
                indegrees[successor] = updated;
            }
        }
        Ok(Some(Self { positions: positions(&order), order }))
    }

    pub(crate) fn compile(
        problem: &ResourceScheduleProblem,
        order: Vec<usize>,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        let Some(positions) = validate_priority(problem, &order, stop)? else {
            return Ok(None);
        };
        Ok(Some(Self { order, positions }))
    }

    pub(crate) fn order(&self) -> &[usize] {
        &self.order
    }

    pub(crate) fn position(&self, activity: usize) -> usize {
        self.positions[activity]
    }

    /// Inclusive final positions to which `activity` may be relocated while
    /// keeping the priority list topological.
    pub(crate) fn relocation_bounds(&self, problem: &ResourceScheduleProblem, activity: usize) -> Option<(usize, usize)> {
        let &current = self.positions.get(activity)?;
        let first = problem
            .precedences
            .predecessors(activity)
            .iter()
            .map(|&predecessor| self.positions[predecessor].saturating_add(1))
            .max()
            .unwrap_or(0);
        let last = problem
            .precedences
            .successors(activity)
            .iter()
            .map(|&successor| self.positions[successor].saturating_sub(1))
            .min()
            .unwrap_or_else(|| self.order.len().saturating_sub(1));
        (first <= current && current <= last).then_some((first, last))
    }
}

fn dispatch_better(
    problem: &ResourceScheduleProblem,
    seed: u64,
    step: usize,
    rule: PriorityRule,
    candidate: usize,
    incumbent: usize,
) -> bool {
    let key = |activity: usize| {
        mix64(
            seed ^ u64::try_from(step).unwrap_or(u64::MAX).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ u64::try_from(activity).unwrap_or(u64::MAX),
        )
    };
    match rule {
        PriorityRule::Stable => candidate < incumbent,
        PriorityRule::ShortestDuration => {
            (problem.durations[candidate], key(candidate), candidate) < (problem.durations[incumbent], key(incumbent), incumbent)
        }
        PriorityRule::LongestDuration => {
            (problem.durations[candidate], std::cmp::Reverse(key(candidate)), std::cmp::Reverse(candidate))
                > (problem.durations[incumbent], std::cmp::Reverse(key(incumbent)), std::cmp::Reverse(incumbent))
        }
        PriorityRule::MostSuccessors => {
            (problem.precedences.successors(candidate).len(), std::cmp::Reverse(key(candidate)), std::cmp::Reverse(candidate))
                > (problem.precedences.successors(incumbent).len(), std::cmp::Reverse(key(incumbent)), std::cmp::Reverse(incumbent))
        }
        PriorityRule::Randomized => (key(candidate), candidate) < (key(incumbent), incumbent),
    }
}

fn positions(order: &[usize]) -> Vec<usize> {
    let mut positions = vec![usize::MAX; order.len()];
    for (position, &activity) in order.iter().enumerate() {
        positions[activity] = position;
    }
    positions
}

fn validate_priority(
    problem: &ResourceScheduleProblem,
    order: &[usize],
    stop: &AtomicBool,
) -> Result<Option<Vec<usize>>, ResourceScheduleInterrupted> {
    if order.len() != problem.activity_count() {
        return Ok(None);
    }
    let mut positions = vec![usize::MAX; order.len()];
    for (position, &activity) in order.iter().enumerate() {
        checkpoint(stop)?;
        let Some(slot) = positions.get_mut(activity) else {
            return Ok(None);
        };
        if *slot != usize::MAX {
            return Ok(None);
        }
        *slot = position;
    }
    for activity in 0..order.len() {
        checkpoint(stop)?;
        for &successor in problem.precedences.successors(activity) {
            checkpoint(stop)?;
            if positions[activity] >= positions[successor] {
                return Ok(None);
            }
        }
    }
    Ok(Some(positions))
}

#[derive(Clone, Debug, Default)]
struct EventProfile {
    capacity: i128,
    /// Net usage changes. Starts add demand and ends subtract it, so equal
    /// timestamps are combined and half-open interval semantics are exact.
    events: BTreeMap<i64, i128>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProfileFit {
    Fits,
    RetryAt(i64),
    Impossible,
}

impl EventProfile {
    fn new(capacity: i128) -> Self {
        Self { capacity, events: BTreeMap::new() }
    }

    fn event_count(&self) -> usize {
        self.events.len()
    }

    fn add_event(&mut self, time: i64, delta: i128) -> Option<()> {
        let current = self.events.get(&time).copied().unwrap_or(0);
        let updated = current.checked_add(delta)?;
        if updated == 0 {
            self.events.remove(&time);
        } else {
            self.events.insert(time, updated);
        }
        Some(())
    }

    fn book(&mut self, start: i64, end: i64, demand: i128) -> Option<()> {
        if demand == 0 || start == end {
            return Some(());
        }
        self.add_event(start, demand)?;
        if self.add_event(end, -demand).is_none() {
            let _ = self.add_event(start, -demand);
            return None;
        }
        Some(())
    }

    /// Return the first event boundary after an overloaded segment. Jumping a
    /// candidate start directly to this boundary cannot skip a feasible start.
    fn fit(
        &self,
        start: i64,
        end: i64,
        demand: i128,
        stop: &AtomicBool,
        stats: &mut ReconstructionStats,
    ) -> Result<ProfileFit, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        stats.profile_checks = stats.profile_checks.saturating_add(1);
        if demand == 0 || start == end {
            return Ok(ProfileFit::Fits);
        }
        if demand > self.capacity {
            return Ok(ProfileFit::Impossible);
        }

        let mut usage = 0i128;
        for (_, &delta) in self.events.range(..=start) {
            checkpoint(stop)?;
            stats.event_visits = stats.event_visits.saturating_add(1);
            let Some(updated) = usage.checked_add(delta) else {
                return Ok(ProfileFit::Impossible);
            };
            usage = updated;
        }

        let mut cursor = start;
        let mut future = self.events.range((Excluded(start), Unbounded));
        loop {
            checkpoint(stop)?;
            if cursor >= end {
                return Ok(ProfileFit::Fits);
            }
            let next = future.next();
            let next_time = next.map(|(&time, _)| time);
            let Some(with_candidate) = usage.checked_add(demand) else {
                return Ok(ProfileFit::Impossible);
            };
            if with_candidate > self.capacity {
                return Ok(next_time.map_or(ProfileFit::Impossible, ProfileFit::RetryAt));
            }
            let Some((&time, &delta)) = next else {
                return Ok(ProfileFit::Fits);
            };
            stats.event_visits = stats.event_visits.saturating_add(1);
            if time >= end {
                return Ok(ProfileFit::Fits);
            }
            let Some(updated) = usage.checked_add(delta) else {
                return Ok(ProfileFit::Impossible);
            };
            usage = updated;
            cursor = time;
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReconstructionStats {
    candidates: u64,
    profile_checks: u64,
    event_visits: u64,
}

#[derive(Clone, Debug)]
struct Reconstruction {
    starts: Vec<i64>,
    makespan: i64,
    profile_events: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReconstructionFailure {
    Interrupted,
    Infeasible,
    Numeric,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GenerationScheme {
    Serial,
    Parallel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReusableEvent {
    time: i64,
    delta: i128,
}

/// Sorted event profile whose storage is fixed when the state is built. A
/// candidate only removes and reinserts events for existing activities, so two
/// events per positive-duration demand are a strict capacity bound.
#[derive(Debug)]
struct ReusableEventProfile {
    capacity: i128,
    events: Vec<ReusableEvent>,
}

impl ReusableEventProfile {
    fn with_event_capacity(capacity: i128, event_capacity: usize) -> Self {
        Self { capacity, events: Vec::with_capacity(event_capacity) }
    }

    fn add_event(&mut self, time: i64, delta: i128) -> Option<()> {
        match self.events.binary_search_by_key(&time, |event| event.time) {
            Ok(index) => {
                let updated = self.events[index].delta.checked_add(delta)?;
                if updated == 0 {
                    self.events.remove(index);
                } else {
                    self.events[index].delta = updated;
                }
            }
            Err(index) => {
                if delta != 0 {
                    // Refuse to grow here. The bound is structural, and a
                    // failure is safer than allocating inside a hot candidate.
                    if self.events.len() >= self.events.capacity() {
                        return None;
                    }
                    self.events.insert(index, ReusableEvent { time, delta });
                }
            }
        }
        Some(())
    }

    fn book(&mut self, start: i64, end: i64, demand: i128) -> Option<()> {
        if demand == 0 || start == end {
            return Some(());
        }
        self.add_event(start, demand)?;
        if self.add_event(end, -demand).is_none() {
            let restored = self.add_event(start, -demand);
            debug_assert!(restored.is_some());
            return None;
        }
        Some(())
    }

    fn fit(
        &self,
        start: i64,
        end: i64,
        demand: i128,
        stop: &AtomicBool,
        stats: &mut ReconstructionStats,
    ) -> Result<ProfileFit, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        stats.profile_checks = stats.profile_checks.saturating_add(1);
        if demand == 0 || start == end {
            return Ok(ProfileFit::Fits);
        }
        if demand > self.capacity {
            return Ok(ProfileFit::Impossible);
        }

        let first_future = self.events.partition_point(|event| event.time <= start);
        let mut usage = 0i128;
        for event in &self.events[..first_future] {
            checkpoint(stop)?;
            stats.event_visits = stats.event_visits.saturating_add(1);
            let Some(updated) = usage.checked_add(event.delta) else {
                return Ok(ProfileFit::Impossible);
            };
            usage = updated;
        }

        let mut cursor = start;
        let mut future = self.events[first_future..].iter();
        loop {
            checkpoint(stop)?;
            if cursor >= end {
                return Ok(ProfileFit::Fits);
            }
            let next = future.next();
            let Some(with_candidate) = usage.checked_add(demand) else {
                return Ok(ProfileFit::Impossible);
            };
            if with_candidate > self.capacity {
                return Ok(next.map_or(ProfileFit::Impossible, |event| ProfileFit::RetryAt(event.time)));
            }
            let Some(event) = next else {
                return Ok(ProfileFit::Fits);
            };
            stats.event_visits = stats.event_visits.saturating_add(1);
            if event.time >= end {
                return Ok(ProfileFit::Fits);
            }
            let Some(updated) = usage.checked_add(event.delta) else {
                return Ok(ProfileFit::Impossible);
            };
            usage = updated;
            cursor = event.time;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResourceWorkspaceCapacities {
    pub(crate) order: usize,
    pub(crate) positions: usize,
    pub(crate) starts: usize,
    pub(crate) ends: usize,
    pub(crate) scheduled: usize,
    pub(crate) touched: usize,
    pub(crate) profile_events: usize,
}

#[derive(Debug)]
struct ResourceEvaluationWorkspace {
    order: Vec<usize>,
    positions: Vec<usize>,
    starts: Vec<i64>,
    ends: Vec<i64>,
    scratch_starts: Vec<i64>,
    scratch_ends: Vec<i64>,
    scheduled: Vec<bool>,
    touched: Vec<usize>,
    profiles: Vec<ReusableEventProfile>,
    scratch_profiles: Vec<ReusableEventProfile>,
    stats: ReconstructionStats,
    rescheduled: usize,
    /// Capacity increases after construction. Hot-path profiles refuse to grow,
    /// so this remains zero unless a future operator explicitly reserves.
    growths: u64,
}

impl ResourceEvaluationWorkspace {
    fn new(
        problem: &ResourceScheduleProblem,
        priority: &PrioritySgs,
        reconstruction: &Reconstruction,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        let count = problem.activity_count();
        if priority.order.len() != count || priority.positions.len() != count || reconstruction.starts.len() != count {
            return Ok(None);
        }
        let profiles = reusable_profiles(problem, stop)?;
        let scratch_profiles = reusable_profiles(problem, stop)?;
        let mut order = Vec::with_capacity(count);
        let mut positions = Vec::with_capacity(count);
        let mut starts = Vec::with_capacity(count);
        let mut ends = Vec::with_capacity(count);
        let mut scratch_starts = Vec::with_capacity(count);
        let mut scratch_ends = Vec::with_capacity(count);
        let mut scheduled = Vec::with_capacity(count);
        for activity in 0..count {
            checkpoint(stop)?;
            let start = reconstruction.starts[activity];
            let Some(end) = start.checked_add(problem.durations[activity]) else {
                return Ok(None);
            };
            order.push(priority.order[activity]);
            positions.push(priority.positions[activity]);
            starts.push(start);
            ends.push(end);
            scratch_starts.push(0);
            scratch_ends.push(0);
            scheduled.push(false);
        }
        let mut workspace = Self {
            order,
            positions,
            starts,
            ends,
            scratch_starts,
            scratch_ends,
            scheduled,
            touched: Vec::with_capacity(count),
            profiles,
            scratch_profiles,
            stats: ReconstructionStats::default(),
            rescheduled: 0,
            growths: 0,
        };
        if workspace.reload_profiles(problem, reconstruction, stop)?.is_none() {
            return Ok(None);
        }
        Ok(Some(workspace))
    }

    fn capacities(&self) -> ResourceWorkspaceCapacities {
        ResourceWorkspaceCapacities {
            order: self.order.capacity(),
            positions: self.positions.capacity(),
            starts: self.starts.capacity(),
            ends: self.ends.capacity(),
            scheduled: self.scheduled.capacity(),
            touched: self.touched.capacity(),
            profile_events: self.profiles.iter().chain(&self.scratch_profiles).map(|profile| profile.events.capacity()).sum(),
        }
    }

    fn sync_priority(&mut self, priority: &PrioritySgs, stop: &AtomicBool) -> Result<(), ResourceScheduleInterrupted> {
        debug_assert_eq!(self.order.len(), priority.order.len());
        debug_assert_eq!(self.positions.len(), priority.positions.len());
        const CHUNK: usize = 4_096;
        for first in (0..self.order.len()).step_by(CHUNK) {
            checkpoint(stop)?;
            let end = first.saturating_add(CHUNK).min(self.order.len());
            self.order[first..end].copy_from_slice(&priority.order[first..end]);
            self.positions[first..end].copy_from_slice(&priority.positions[first..end]);
        }
        Ok(())
    }

    /// Rollback and final commit cannot abandon a half-copied priority after
    /// cancellation. Their caller has already polled the stop token.
    fn sync_priority_atomically(&mut self, priority: &PrioritySgs) {
        self.order.copy_from_slice(priority.order());
        self.positions.copy_from_slice(&priority.positions);
    }

    /// Build a complete profile in scratch storage, then swap all derived
    /// arrays in one non-failing commit. Interruption leaves the committed
    /// workspace byte-for-byte usable.
    fn reload_profiles(
        &mut self,
        problem: &ResourceScheduleProblem,
        reconstruction: &Reconstruction,
        stop: &AtomicBool,
    ) -> Result<Option<()>, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        if reconstruction.starts.len() != problem.activity_count() {
            return Ok(None);
        }
        for profile in &mut self.scratch_profiles {
            checkpoint(stop)?;
            profile.events.clear();
        }
        for activity in 0..problem.activity_count() {
            checkpoint(stop)?;
            let start = reconstruction.starts[activity];
            let Some(end) = start.checked_add(problem.durations[activity]) else {
                return Ok(None);
            };
            self.scratch_starts[activity] = start;
            self.scratch_ends[activity] = end;
            if book_reusable_activity_interruptible(problem, &mut self.scratch_profiles, activity, start, 1, stop)?.is_none() {
                return Ok(None);
            }
        }
        checkpoint(stop)?;
        std::mem::swap(&mut self.profiles, &mut self.scratch_profiles);
        std::mem::swap(&mut self.starts, &mut self.scratch_starts);
        std::mem::swap(&mut self.ends, &mut self.scratch_ends);
        Ok(Some(()))
    }
}

fn reusable_profiles(
    problem: &ResourceScheduleProblem,
    stop: &AtomicBool,
) -> Result<Vec<ReusableEventProfile>, ResourceScheduleInterrupted> {
    let mut profiles = Vec::with_capacity(problem.resources.len());
    for resource in &problem.resources {
        checkpoint(stop)?;
        profiles.push(ReusableEventProfile::with_event_capacity(resource.capacity, resource.event_capacity));
    }
    Ok(profiles)
}

fn book_reusable_activity(
    problem: &ResourceScheduleProblem,
    profiles: &mut [ReusableEventProfile],
    activity: usize,
    start: i64,
    direction: i128,
) -> Option<i64> {
    let end = start.checked_add(problem.durations[activity])?;
    for (updated, &(resource, demand)) in problem.activity_resources[activity].iter().enumerate() {
        let signed = demand.checked_mul(direction)?;
        if profiles[resource].book(start, end, signed).is_none() {
            for &(rollback_resource, rollback_demand) in problem.activity_resources[activity][..updated].iter().rev() {
                let signed = rollback_demand.checked_mul(-direction)?;
                let restored = profiles[rollback_resource].book(start, end, signed);
                debug_assert!(restored.is_some());
            }
            return None;
        }
    }
    Some(end)
}

fn book_reusable_activity_interruptible(
    problem: &ResourceScheduleProblem,
    profiles: &mut [ReusableEventProfile],
    activity: usize,
    start: i64,
    direction: i128,
    stop: &AtomicBool,
) -> Result<Option<i64>, ResourceScheduleInterrupted> {
    let Some(end) = start.checked_add(problem.durations[activity]) else {
        return Ok(None);
    };
    for (updated, &(resource, demand)) in problem.activity_resources[activity].iter().enumerate() {
        if checkpoint(stop).is_err() {
            for &(rollback_resource, rollback_demand) in problem.activity_resources[activity][..updated].iter().rev() {
                let Some(signed) = rollback_demand.checked_mul(-direction) else {
                    return Ok(None);
                };
                if profiles[rollback_resource].book(start, end, signed).is_none() {
                    return Ok(None);
                }
            }
            return Err(ResourceScheduleInterrupted);
        }
        let Some(signed) = demand.checked_mul(direction) else {
            return Ok(None);
        };
        if profiles[resource].book(start, end, signed).is_none() {
            for &(rollback_resource, rollback_demand) in problem.activity_resources[activity][..updated].iter().rev() {
                let Some(signed) = rollback_demand.checked_mul(-direction) else {
                    return Ok(None);
                };
                if profiles[rollback_resource].book(start, end, signed).is_none() {
                    return Ok(None);
                }
            }
            return Ok(None);
        }
    }
    Ok(Some(end))
}

fn earliest_reusable_start(
    problem: &ResourceScheduleProblem,
    profiles: &[ReusableEventProfile],
    activity: usize,
    release: i64,
    stop: &AtomicBool,
    stats: &mut ReconstructionStats,
) -> Result<Option<i64>, ReconstructionFailure> {
    let latest = problem.latest_starts[activity];
    let duration = problem.durations[activity];
    let mut candidate = release.max(problem.earliest_starts[activity]);
    while candidate <= latest {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        stats.candidates = stats.candidates.saturating_add(1);
        let end = candidate.checked_add(duration).ok_or(ReconstructionFailure::Numeric)?;
        let mut retry = None;
        for &(resource, demand) in &problem.activity_resources[activity] {
            match profiles[resource].fit(candidate, end, demand, stop, stats).map_err(|_| ReconstructionFailure::Interrupted)? {
                ProfileFit::Fits => {}
                ProfileFit::RetryAt(time) => retry = Some(retry.map_or(time, |old: i64| old.max(time))),
                ProfileFit::Impossible => return Ok(None),
            }
        }
        let Some(next) = retry else {
            return Ok(Some(candidate));
        };
        if next <= candidate {
            return Err(ReconstructionFailure::Numeric);
        }
        candidate = next;
    }
    Ok(None)
}

fn reusable_resources_fit_at(
    problem: &ResourceScheduleProblem,
    profiles: &[ReusableEventProfile],
    activity: usize,
    start: i64,
    stop: &AtomicBool,
    stats: &mut ReconstructionStats,
) -> Result<bool, ReconstructionFailure> {
    let end = start.checked_add(problem.durations[activity]).ok_or(ReconstructionFailure::Numeric)?;
    for &(resource, demand) in &problem.activity_resources[activity] {
        if profiles[resource].fit(start, end, demand, stop, stats).map_err(|_| ReconstructionFailure::Interrupted)? != ProfileFit::Fits {
            return Ok(false);
        }
    }
    Ok(true)
}

fn empty_profiles(problem: &ResourceScheduleProblem) -> Vec<EventProfile> {
    problem.resources.iter().map(|resource| EventProfile::new(resource.capacity)).collect()
}

fn event_profile_count(profiles: &[EventProfile], stop: &AtomicBool) -> Result<usize, ReconstructionFailure> {
    let mut count = 0usize;
    for profile in profiles {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        count = count.saturating_add(profile.event_count());
    }
    Ok(count)
}

fn reusable_profile_count(profiles: &[ReusableEventProfile], stop: &AtomicBool) -> Result<usize, ResourceScheduleInterrupted> {
    let mut count = 0usize;
    for profile in profiles {
        checkpoint(stop)?;
        count = count.saturating_add(profile.events.len());
    }
    Ok(count)
}

fn book_activity(
    problem: &ResourceScheduleProblem,
    profiles: &mut [EventProfile],
    activity: usize,
    start: i64,
) -> Result<i64, ReconstructionFailure> {
    let end = start.checked_add(problem.durations[activity]).ok_or(ReconstructionFailure::Numeric)?;
    for &(resource, demand) in &problem.activity_resources[activity] {
        profiles[resource].book(start, end, demand).ok_or(ReconstructionFailure::Numeric)?;
    }
    Ok(end)
}

fn earliest_resource_start(
    problem: &ResourceScheduleProblem,
    profiles: &[EventProfile],
    activity: usize,
    release: i64,
    stop: &AtomicBool,
    stats: &mut ReconstructionStats,
) -> Result<Option<i64>, ReconstructionFailure> {
    let latest = problem.latest_starts[activity];
    let duration = problem.durations[activity];
    let mut candidate = release.max(problem.earliest_starts[activity]);
    while candidate <= latest {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        stats.candidates = stats.candidates.saturating_add(1);
        let end = candidate.checked_add(duration).ok_or(ReconstructionFailure::Numeric)?;
        let mut retry = None;
        for &(resource, demand) in &problem.activity_resources[activity] {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            match profiles[resource].fit(candidate, end, demand, stop, stats).map_err(|_| ReconstructionFailure::Interrupted)? {
                ProfileFit::Fits => {}
                ProfileFit::RetryAt(time) => retry = Some(retry.map_or(time, |old: i64| old.max(time))),
                ProfileFit::Impossible => return Ok(None),
            }
        }
        let Some(next) = retry else {
            return Ok(Some(candidate));
        };
        if next <= candidate {
            return Err(ReconstructionFailure::Numeric);
        }
        candidate = next;
    }
    Ok(None)
}

fn resources_fit_at(
    problem: &ResourceScheduleProblem,
    profiles: &[EventProfile],
    activity: usize,
    start: i64,
    stop: &AtomicBool,
    stats: &mut ReconstructionStats,
) -> Result<bool, ReconstructionFailure> {
    let end = start.checked_add(problem.durations[activity]).ok_or(ReconstructionFailure::Numeric)?;
    for &(resource, demand) in &problem.activity_resources[activity] {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        if profiles[resource].fit(start, end, demand, stop, stats).map_err(|_| ReconstructionFailure::Interrupted)? != ProfileFit::Fits {
            return Ok(false);
        }
    }
    Ok(true)
}

fn serial_reconstruction(
    problem: &ResourceScheduleProblem,
    priority: &PrioritySgs,
    stop: &AtomicBool,
) -> Result<(Reconstruction, ReconstructionStats), ReconstructionFailure> {
    checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
    let mut stats = ReconstructionStats::default();
    let mut profiles = empty_profiles(problem);
    let mut starts = vec![0i64; problem.activity_count()];
    let mut ends = vec![0i64; problem.activity_count()];
    for &activity in priority.order() {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        let release = problem.precedences.predecessors(activity).iter().map(|&predecessor| ends[predecessor]).max().unwrap_or(0);
        let Some(start) = earliest_resource_start(problem, &profiles, activity, release, stop, &mut stats)? else {
            return Err(ReconstructionFailure::Infeasible);
        };
        let end = book_activity(problem, &mut profiles, activity, start)?;
        if start > problem.latest_starts[activity] {
            return Err(ReconstructionFailure::Infeasible);
        }
        starts[activity] = start;
        ends[activity] = end;
    }
    let makespan = ends.into_iter().max().unwrap_or(0);
    let profile_events = event_profile_count(&profiles, stop)?;
    Ok((Reconstruction { starts, makespan, profile_events }, stats))
}

fn parallel_reconstruction(
    problem: &ResourceScheduleProblem,
    priority: &PrioritySgs,
    stop: &AtomicBool,
) -> Result<(Reconstruction, ReconstructionStats), ReconstructionFailure> {
    checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
    let count = problem.activity_count();
    let mut stats = ReconstructionStats::default();
    let mut profiles = empty_profiles(problem);
    let mut starts = vec![0i64; count];
    let mut ends = vec![0i64; count];
    let mut scheduled = vec![false; count];
    let mut remaining = count;
    let mut time = 0i64;

    while remaining > 0 {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        let mut scheduled_at_time = false;
        loop {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let mut selected = None;
            for &activity in priority.order() {
                checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
                if scheduled[activity]
                    || problem
                        .precedences
                        .predecessors(activity)
                        .iter()
                        .any(|&predecessor| !scheduled[predecessor] || ends[predecessor] > time)
                {
                    continue;
                }
                if time < problem.earliest_starts[activity] {
                    continue;
                }
                if time > problem.latest_starts[activity] {
                    return Err(ReconstructionFailure::Infeasible);
                }
                stats.candidates = stats.candidates.saturating_add(1);
                if resources_fit_at(problem, &profiles, activity, time, stop, &mut stats)? {
                    selected = Some(activity);
                    break;
                }
            }
            let Some(activity) = selected else {
                break;
            };
            let end = book_activity(problem, &mut profiles, activity, time)?;
            starts[activity] = time;
            ends[activity] = end;
            scheduled[activity] = true;
            remaining -= 1;
            scheduled_at_time = true;
        }
        if remaining == 0 {
            break;
        }
        let next_completion =
            (0..count).filter(|&activity| scheduled[activity] && ends[activity] > time).map(|activity| ends[activity]).min();
        let next_release = (0..count)
            .filter(|&activity| {
                !scheduled[activity]
                    && problem.earliest_starts[activity] > time
                    && problem.precedences.predecessors(activity).iter().all(|&predecessor| scheduled[predecessor])
            })
            .map(|activity| problem.earliest_starts[activity])
            .min();
        let next = next_completion.into_iter().chain(next_release).min();
        let Some(next) = next else {
            return Err(ReconstructionFailure::Infeasible);
        };
        if next <= time || (!scheduled_at_time && next == time) {
            return Err(ReconstructionFailure::Numeric);
        }
        time = next;
    }

    let makespan = ends.into_iter().max().unwrap_or(0);
    let profile_events = event_profile_count(&profiles, stop)?;
    Ok((Reconstruction { starts, makespan, profile_events }, stats))
}

fn reconstruct(
    problem: &ResourceScheduleProblem,
    priority: &PrioritySgs,
    scheme: GenerationScheme,
    stop: &AtomicBool,
) -> Result<(Reconstruction, ReconstructionStats), ReconstructionFailure> {
    match scheme {
        GenerationScheme::Serial => serial_reconstruction(problem, priority, stop),
        GenerationScheme::Parallel => parallel_reconstruction(problem, priority, stop),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateEvaluation {
    makespan: i64,
    profile_events: usize,
}

fn validate_reusable_priority(
    problem: &ResourceScheduleProblem,
    order: &[usize],
    positions: &mut [usize],
    stop: &AtomicBool,
) -> Result<bool, ResourceScheduleInterrupted> {
    if order.len() != problem.activity_count() || positions.len() != order.len() {
        return Ok(false);
    }
    positions.fill(usize::MAX);
    for (position, &activity) in order.iter().enumerate() {
        checkpoint(stop)?;
        let Some(slot) = positions.get_mut(activity) else {
            return Ok(false);
        };
        if *slot != usize::MAX {
            return Ok(false);
        }
        *slot = position;
    }
    for activity in 0..order.len() {
        checkpoint(stop)?;
        for &successor in problem.precedences.successors(activity) {
            if positions[activity] >= positions[successor] {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn first_priority_difference(current: &[usize], candidate: &[usize]) -> usize {
    current.iter().zip(candidate).position(|(left, right)| left != right).unwrap_or(current.len())
}

fn rollback_serial_workspace(
    problem: &ResourceScheduleProblem,
    current: &PrioritySgs,
    reconstruction: &Reconstruction,
    workspace: &mut ResourceEvaluationWorkspace,
    cut: usize,
    removed_old: usize,
) -> Result<(), ReconstructionFailure> {
    for &activity in workspace.touched.iter().rev() {
        let start = workspace.starts[activity];
        book_reusable_activity(problem, &mut workspace.profiles, activity, start, -1).ok_or(ReconstructionFailure::Numeric)?;
    }
    workspace.touched.clear();
    for &activity in &current.order()[cut..cut.saturating_add(removed_old)] {
        let start = reconstruction.starts[activity];
        book_reusable_activity(problem, &mut workspace.profiles, activity, start, 1).ok_or(ReconstructionFailure::Numeric)?;
    }
    for &activity in &current.order()[cut..] {
        let start = reconstruction.starts[activity];
        workspace.starts[activity] = start;
        workspace.ends[activity] = start.checked_add(problem.durations[activity]).ok_or(ReconstructionFailure::Numeric)?;
    }
    Ok(())
}

/// Evaluate only the changed suffix of a serial SGS. The workspace profile is
/// initially the incumbent's full profile. Its old suffix is removed, the new
/// suffix is booked, and the caller either keeps it or rolls it back.
fn evaluate_serial_delta(
    problem: &ResourceScheduleProblem,
    current: &PrioritySgs,
    reconstruction: &Reconstruction,
    workspace: &mut ResourceEvaluationWorkspace,
    cut: usize,
    stop: &AtomicBool,
) -> Result<CandidateEvaluation, ReconstructionFailure> {
    workspace.stats = ReconstructionStats::default();
    workspace.rescheduled = 0;
    workspace.touched.clear();
    let mut removed_old = 0usize;

    for &activity in &current.order()[cut..] {
        if checkpoint(stop).is_err() {
            rollback_serial_workspace(problem, current, reconstruction, workspace, cut, removed_old)?;
            return Err(ReconstructionFailure::Interrupted);
        }
        let start = reconstruction.starts[activity];
        match book_reusable_activity_interruptible(problem, &mut workspace.profiles, activity, start, -1, stop) {
            Ok(Some(_)) => {}
            Ok(None) => {
                rollback_serial_workspace(problem, current, reconstruction, workspace, cut, removed_old)?;
                return Err(ReconstructionFailure::Numeric);
            }
            Err(ResourceScheduleInterrupted) => {
                rollback_serial_workspace(problem, current, reconstruction, workspace, cut, removed_old)?;
                return Err(ReconstructionFailure::Interrupted);
            }
        }
        removed_old += 1;
    }

    for position in cut..workspace.order.len() {
        if checkpoint(stop).is_err() {
            rollback_serial_workspace(problem, current, reconstruction, workspace, cut, removed_old)?;
            return Err(ReconstructionFailure::Interrupted);
        }
        let activity = workspace.order[position];
        let release = problem.precedences.predecessors(activity).iter().map(|&predecessor| workspace.ends[predecessor]).max().unwrap_or(0);
        let start = match earliest_reusable_start(problem, &workspace.profiles, activity, release, stop, &mut workspace.stats) {
            Ok(Some(start)) => start,
            Ok(None) => {
                rollback_serial_workspace(problem, current, reconstruction, workspace, cut, removed_old)?;
                return Err(ReconstructionFailure::Infeasible);
            }
            Err(failure) => {
                rollback_serial_workspace(problem, current, reconstruction, workspace, cut, removed_old)?;
                return Err(failure);
            }
        };
        let end = match book_reusable_activity_interruptible(problem, &mut workspace.profiles, activity, start, 1, stop) {
            Ok(Some(end)) => end,
            Ok(None) => {
                rollback_serial_workspace(problem, current, reconstruction, workspace, cut, removed_old)?;
                return Err(ReconstructionFailure::Numeric);
            }
            Err(ResourceScheduleInterrupted) => {
                rollback_serial_workspace(problem, current, reconstruction, workspace, cut, removed_old)?;
                return Err(ReconstructionFailure::Interrupted);
            }
        };
        workspace.starts[activity] = start;
        workspace.ends[activity] = end;
        workspace.touched.push(activity);
        workspace.rescheduled = workspace.rescheduled.saturating_add(1);
    }

    let makespan = workspace.ends.iter().copied().max().unwrap_or(0);
    let profile_events = match reusable_profile_count(&workspace.profiles, stop) {
        Ok(profile_events) => profile_events,
        Err(ResourceScheduleInterrupted) => {
            rollback_serial_workspace(problem, current, reconstruction, workspace, cut, problem.activity_count().saturating_sub(cut))?;
            return Err(ReconstructionFailure::Interrupted);
        }
    };
    Ok(CandidateEvaluation { makespan, profile_events })
}

fn evaluate_parallel_reusable(
    problem: &ResourceScheduleProblem,
    workspace: &mut ResourceEvaluationWorkspace,
    stop: &AtomicBool,
) -> Result<CandidateEvaluation, ReconstructionFailure> {
    workspace.stats = ReconstructionStats::default();
    workspace.rescheduled = 0;
    workspace.touched.clear();
    for profile in &mut workspace.scratch_profiles {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        profile.events.clear();
    }
    for activity in 0..problem.activity_count() {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        workspace.scratch_starts[activity] = 0;
        workspace.scratch_ends[activity] = 0;
        workspace.scheduled[activity] = false;
    }
    let count = problem.activity_count();
    let mut remaining = count;
    let mut time = 0i64;

    while remaining > 0 {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        let mut scheduled_at_time = false;
        loop {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            let mut selected = None;
            for &activity in &workspace.order {
                checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
                if workspace.scheduled[activity]
                    || problem
                        .precedences
                        .predecessors(activity)
                        .iter()
                        .any(|&predecessor| !workspace.scheduled[predecessor] || workspace.scratch_ends[predecessor] > time)
                {
                    continue;
                }
                if time < problem.earliest_starts[activity] {
                    continue;
                }
                if time > problem.latest_starts[activity] {
                    return Err(ReconstructionFailure::Infeasible);
                }
                workspace.stats.candidates = workspace.stats.candidates.saturating_add(1);
                if reusable_resources_fit_at(problem, &workspace.scratch_profiles, activity, time, stop, &mut workspace.stats)? {
                    selected = Some(activity);
                    break;
                }
            }
            let Some(activity) = selected else {
                break;
            };
            let end = book_reusable_activity_interruptible(problem, &mut workspace.scratch_profiles, activity, time, 1, stop)
                .map_err(|_| ReconstructionFailure::Interrupted)?
                .ok_or(ReconstructionFailure::Numeric)?;
            workspace.scratch_starts[activity] = time;
            workspace.scratch_ends[activity] = end;
            workspace.scheduled[activity] = true;
            workspace.touched.push(activity);
            workspace.rescheduled = workspace.rescheduled.saturating_add(1);
            remaining -= 1;
            scheduled_at_time = true;
        }
        if remaining == 0 {
            break;
        }
        let next_completion = (0..count)
            .filter(|&activity| workspace.scheduled[activity] && workspace.scratch_ends[activity] > time)
            .map(|activity| workspace.scratch_ends[activity])
            .min();
        let next_release = (0..count)
            .filter(|&activity| {
                !workspace.scheduled[activity]
                    && problem.earliest_starts[activity] > time
                    && problem.precedences.predecessors(activity).iter().all(|&predecessor| workspace.scheduled[predecessor])
            })
            .map(|activity| problem.earliest_starts[activity])
            .min();
        let next = next_completion.into_iter().chain(next_release).min();
        let Some(next) = next else {
            return Err(ReconstructionFailure::Infeasible);
        };
        if next <= time || (!scheduled_at_time && next == time) {
            return Err(ReconstructionFailure::Numeric);
        }
        time = next;
    }

    let makespan = workspace.scratch_ends.iter().copied().max().unwrap_or(0);
    let profile_events = reusable_profile_count(&workspace.scratch_profiles, stop).map_err(|_| ReconstructionFailure::Interrupted)?;
    Ok(CandidateEvaluation { makespan, profile_events })
}

fn latest_resource_start(
    problem: &ResourceScheduleProblem,
    profiles: &[EventProfile],
    activity: usize,
    latest: i64,
    stop: &AtomicBool,
    stats: &mut ReconstructionStats,
) -> Result<Option<i64>, ReconstructionFailure> {
    checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
    let duration = problem.durations[activity];
    let latest = latest.min(problem.latest_starts[activity]);
    let earliest = problem.earliest_starts[activity];
    if latest < earliest {
        return Ok(None);
    }
    let mut candidates = vec![latest];
    for &(resource, demand) in &problem.activity_resources[activity] {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        if duration > 0 && demand > problem.resources[resource].capacity {
            return Ok(None);
        }
        for &time in profiles[resource].events.keys() {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            if (earliest..=latest).contains(&time) {
                candidates.push(time);
            }
            if let Some(candidate) = time.checked_sub(duration) {
                if (earliest..=latest).contains(&candidate) {
                    candidates.push(candidate);
                }
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    for &candidate in candidates.iter().rev() {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        stats.candidates = stats.candidates.saturating_add(1);
        if resources_fit_at(problem, profiles, activity, candidate, stop, stats)? {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn right_reconstruction(
    problem: &ResourceScheduleProblem,
    priority: &PrioritySgs,
    target_makespan: i64,
    stop: &AtomicBool,
) -> Result<(Reconstruction, ReconstructionStats), ReconstructionFailure> {
    checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
    let count = problem.activity_count();
    let mut stats = ReconstructionStats::default();
    let mut profiles = empty_profiles(problem);
    let mut starts = vec![0i64; count];
    let mut ends = vec![0i64; count];
    let mut scheduled = vec![false; count];
    for &activity in priority.order().iter().rev() {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        let mut deadline = target_makespan
            .min(problem.latest_starts[activity].checked_add(problem.durations[activity]).ok_or(ReconstructionFailure::Numeric)?);
        for &successor in problem.precedences.successors(activity) {
            checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
            if !scheduled[successor] {
                return Err(ReconstructionFailure::Infeasible);
            }
            deadline = deadline.min(starts[successor]);
        }
        let latest = deadline.checked_sub(problem.durations[activity]).ok_or(ReconstructionFailure::Numeric)?;
        let Some(start) = latest_resource_start(problem, &profiles, activity, latest, stop, &mut stats)? else {
            return Err(ReconstructionFailure::Infeasible);
        };
        let end = book_activity(problem, &mut profiles, activity, start)?;
        starts[activity] = start;
        ends[activity] = end;
        scheduled[activity] = true;
    }
    let makespan = ends.into_iter().max().unwrap_or(0);
    if makespan > target_makespan {
        return Err(ReconstructionFailure::Infeasible);
    }
    let profile_events = event_profile_count(&profiles, stop)?;
    Ok((Reconstruction { starts, makespan, profile_events }, stats))
}

fn priority_from_schedule(
    problem: &ResourceScheduleProblem,
    starts: &[i64],
    previous: &PrioritySgs,
    stop: &AtomicBool,
) -> Result<PrioritySgs, ReconstructionFailure> {
    let count = problem.activity_count();
    let mut indegrees = (0..count).map(|activity| problem.precedences.predecessors(activity).len()).collect::<Vec<_>>();
    let mut selected = vec![false; count];
    let mut order = Vec::with_capacity(count);
    while order.len() < count {
        checkpoint(stop).map_err(|_| ReconstructionFailure::Interrupted)?;
        let next = (0..count)
            .filter(|&activity| !selected[activity] && indegrees[activity] == 0)
            .min_by_key(|&activity| (starts[activity], previous.position(activity), activity));
        let Some(activity) = next else {
            return Err(ReconstructionFailure::Infeasible);
        };
        selected[activity] = true;
        order.push(activity);
        for &successor in problem.precedences.successors(activity) {
            let Some(updated) = indegrees[successor].checked_sub(1) else {
                return Err(ReconstructionFailure::Numeric);
            };
            indegrees[successor] = updated;
        }
    }
    let positions = positions(&order);
    Ok(PrioritySgs { order, positions })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ResourceScheduleMove {
    AdjacentSwap {
        first_position: usize,
    },
    /// Move one activity to final index `to` after removing index `from`.
    Relocate {
        from: usize,
        to: usize,
    },
    /// Move the contiguous half-open segment `[first, first + len)` so its
    /// first activity ends at index `to` in the final priority order.
    SegmentRelocate {
        first: usize,
        len: usize,
        to: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResourceAlnsBudget {
    pub(crate) attempts: usize,
    pub(crate) max_moves: usize,
    pub(crate) max_segment_len: usize,
}

impl ResourceAlnsBudget {
    pub(crate) fn bounded(activity_count: usize) -> Self {
        Self {
            attempts: activity_count.saturating_mul(4).clamp(16, 512),
            max_moves: activity_count.saturating_mul(2).clamp(8, 128),
            max_segment_len: activity_count.saturating_div(8).clamp(2, 16),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ResourceAlnsGeneration {
    pub(crate) attempts: u64,
    pub(crate) generated: u64,
    pub(crate) precedence_rejections: u64,
    pub(crate) duplicate_rejections: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResourceMoveAcceptance {
    Improving,
    NonWorsening,
    Always,
}

impl ResourceMoveAcceptance {
    fn accepts(self, current: i64, candidate: i64) -> bool {
        match self {
            Self::Improving => candidate < current,
            Self::NonWorsening => candidate <= current,
            Self::Always => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResourceMoveRejection {
    Invalid,
    Precedence,
    Infeasible,
    Numeric,
    NotAccepted { current: i64, candidate: i64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResourceMoveOutcome {
    Accepted { previous: i64, current: i64 },
    Rejected(ResourceMoveRejection),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Justification {
    Left,
    Right,
    Double,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct JustificationOutcome {
    pub(crate) previous: i64,
    pub(crate) current: i64,
    pub(crate) changed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ResourceScheduleMetrics {
    pub(crate) serial_constructions: u64,
    pub(crate) parallel_constructions: u64,
    pub(crate) reconstructions: u64,
    pub(crate) construction_candidates: u64,
    pub(crate) candidate_scheduling_attempts: u64,
    pub(crate) profile_checks: u64,
    pub(crate) event_visits: u64,
    pub(crate) peak_profile_events: usize,
    pub(crate) moves_considered: u64,
    pub(crate) moves_accepted: u64,
    pub(crate) precedence_rejections: u64,
    pub(crate) infeasible_rejections: u64,
    pub(crate) objective_rejections: u64,
    pub(crate) left_justifications: u64,
    pub(crate) right_justifications: u64,
    pub(crate) double_justifications: u64,
    pub(crate) delta_evaluations: u64,
    pub(crate) full_workspace_evaluations: u64,
    pub(crate) delta_activities_rescheduled: u64,
    pub(crate) workspace_rollbacks: u64,
    pub(crate) oracle_validations: u64,
    pub(crate) oracle_mismatches: u64,
    pub(crate) alns_generation_attempts: u64,
    pub(crate) alns_moves_generated: u64,
    pub(crate) workspace_growths: u64,
}

impl ResourceScheduleMetrics {
    fn record_construction(&mut self, stats: ReconstructionStats, profile_events: usize) {
        self.reconstructions = self.reconstructions.saturating_add(1);
        self.construction_candidates = self.construction_candidates.saturating_add(stats.candidates);
        self.record_profile_work(stats, profile_events);
    }

    fn record_search_reconstruction(&mut self, stats: ReconstructionStats, profile_events: usize) {
        self.reconstructions = self.reconstructions.saturating_add(1);
        self.candidate_scheduling_attempts = self.candidate_scheduling_attempts.saturating_add(stats.candidates);
        self.record_profile_work(stats, profile_events);
    }

    fn record_profile_work(&mut self, stats: ReconstructionStats, profile_events: usize) {
        self.profile_checks = self.profile_checks.saturating_add(stats.profile_checks);
        self.event_visits = self.event_visits.saturating_add(stats.event_visits);
        self.peak_profile_events = self.peak_profile_events.max(profile_events);
    }

    fn record_candidate_work(&mut self, stats: ReconstructionStats, profile_events: usize, rescheduled: usize) {
        self.candidate_scheduling_attempts = self.candidate_scheduling_attempts.saturating_add(stats.candidates);
        self.record_profile_work(stats, profile_events);
        self.delta_activities_rescheduled =
            self.delta_activities_rescheduled.saturating_add(u64::try_from(rescheduled).unwrap_or(u64::MAX));
    }
}

/// Complete compact RCPSP state. The current priority and reconstruction are
/// always mutually consistent at public method boundaries.
pub(crate) struct ResourceScheduleState {
    problem: ResourceScheduleProblem,
    priority: PrioritySgs,
    reconstruction: Reconstruction,
    workspace: ResourceEvaluationWorkspace,
    scheme: GenerationScheme,
    serial_delta_valid: bool,
    metrics: ResourceScheduleMetrics,
}

impl ResourceScheduleState {
    pub(crate) fn construct(
        problem: &ResourceScheduleProblem,
        priority: PrioritySgs,
        scheme: GenerationScheme,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        if validate_priority(problem, priority.order(), stop)?.is_none() {
            return Ok(None);
        }
        let (reconstruction, stats) = match reconstruct(problem, &priority, scheme, stop) {
            Ok(value) => value,
            Err(ReconstructionFailure::Interrupted) => return Err(ResourceScheduleInterrupted),
            Err(_) => return Ok(None),
        };
        let mut metrics = ResourceScheduleMetrics::default();
        match scheme {
            GenerationScheme::Serial => metrics.serial_constructions = 1,
            GenerationScheme::Parallel => metrics.parallel_constructions = 1,
        }
        metrics.record_construction(stats, reconstruction.profile_events);
        let Some(workspace) = ResourceEvaluationWorkspace::new(problem, &priority, &reconstruction, stop)? else {
            return Ok(None);
        };
        Ok(Some(Self { problem: problem.clone(), priority, reconstruction, workspace, scheme, serial_delta_valid: true, metrics }))
    }

    pub(crate) fn serial(
        problem: &ResourceScheduleProblem,
        priority: PrioritySgs,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ResourceScheduleInterrupted> {
        Self::construct(problem, priority, GenerationScheme::Serial, stop)
    }

    pub(crate) fn parallel(
        problem: &ResourceScheduleProblem,
        priority: PrioritySgs,
        stop: &AtomicBool,
    ) -> Result<Option<Self>, ResourceScheduleInterrupted> {
        Self::construct(problem, priority, GenerationScheme::Parallel, stop)
    }

    pub(crate) fn priority(&self) -> &PrioritySgs {
        &self.priority
    }

    pub(crate) fn starts(&self) -> &[i64] {
        &self.reconstruction.starts
    }

    pub(crate) fn makespan(&self) -> i64 {
        self.reconstruction.makespan
    }

    pub(crate) fn scheme(&self) -> GenerationScheme {
        self.scheme
    }

    pub(crate) fn metrics(&self) -> ResourceScheduleMetrics {
        ResourceScheduleMetrics { workspace_growths: self.workspace.growths, ..self.metrics }
    }

    pub(crate) fn workspace_capacities(&self) -> ResourceWorkspaceCapacities {
        self.workspace.capacities()
    }

    #[cfg(test)]
    pub(crate) fn probe_workspace_rebuild(&self, stop: &AtomicBool) -> Result<bool, ResourceScheduleInterrupted> {
        ResourceEvaluationWorkspace::new(&self.problem, &self.priority, &self.reconstruction, stop).map(|workspace| workspace.is_some())
    }

    #[cfg(test)]
    pub(crate) fn probe_workspace_reload(&mut self, stop: &AtomicBool) -> Result<bool, ResourceScheduleInterrupted> {
        self.workspace.reload_profiles(&self.problem, &self.reconstruction, stop).map(|reloaded| reloaded.is_some())
    }

    /// Generate precedence-safe adjacent swaps and relocations up to `limit`.
    /// Relocations remain inside each activity's direct precedence bounds.
    pub(crate) fn bounded_moves(&self, limit: usize, stop: &AtomicBool) -> Result<Vec<ResourceScheduleMove>, ResourceScheduleInterrupted> {
        self.bounded_moves_from(0, limit, stop)
    }

    /// Cyclic variant used by search loops. Advancing `offset` between calls
    /// prevents a small batch limit from starving late priority positions.
    pub(crate) fn bounded_moves_from(
        &self,
        offset: usize,
        limit: usize,
        stop: &AtomicBool,
    ) -> Result<Vec<ResourceScheduleMove>, ResourceScheduleInterrupted> {
        let mut moves = Vec::new();
        self.fill_bounded_moves_from(offset, limit, &mut moves, stop)?;
        Ok(moves)
    }

    /// Fill a caller-owned neighborhood buffer. After its first reserve, the
    /// same buffer can be reused across every descent pass.
    pub(crate) fn fill_bounded_moves(
        &self,
        limit: usize,
        moves: &mut Vec<ResourceScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<(), ResourceScheduleInterrupted> {
        self.fill_bounded_moves_from(0, limit, moves, stop)
    }

    pub(crate) fn fill_bounded_moves_from(
        &self,
        offset: usize,
        limit: usize,
        moves: &mut Vec<ResourceScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<(), ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        moves.clear();
        if limit == 0 {
            return Ok(());
        }
        if moves.capacity() < limit {
            moves.reserve(limit);
        }
        let adjacent_count = self.priority.order.len().saturating_sub(1);
        for step in 0..adjacent_count {
            checkpoint(stop)?;
            let first_position = offset.wrapping_add(step) % adjacent_count;
            let left = self.priority.order[first_position];
            let right = self.priority.order[first_position + 1];
            if !self.problem.precedences.successors(left).contains(&right) {
                moves.push(ResourceScheduleMove::AdjacentSwap { first_position });
                if moves.len() == limit {
                    return Ok(());
                }
            }
        }
        let activity_count = self.priority.order.len();
        for activity_step in 0..activity_count {
            checkpoint(stop)?;
            let activity_position = offset.wrapping_add(activity_step) % activity_count;
            let activity = self.priority.order[activity_position];
            let current = self.priority.position(activity);
            let Some((first, last)) = self.priority.relocation_bounds(&self.problem, activity) else {
                continue;
            };
            let destination_count = last - first + 1;
            for destination_step in 0..destination_count {
                checkpoint(stop)?;
                let to = first + offset.wrapping_add(destination_step) % destination_count;
                if to == current || (current.abs_diff(to) == 1) {
                    continue;
                }
                moves.push(ResourceScheduleMove::Relocate { from: current, to });
                if moves.len() == limit {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Generate a deterministic, budgeted ALNS macro-neighborhood of
    /// precedence-feasible contiguous segment relocations. No candidate is
    /// reconstructed here; generation only patches the reusable priority
    /// buffer and checks the canonical DAG.
    pub(crate) fn fill_alns_segment_moves(
        &mut self,
        seed: u64,
        budget: ResourceAlnsBudget,
        moves: &mut Vec<ResourceScheduleMove>,
        stop: &AtomicBool,
    ) -> Result<ResourceAlnsGeneration, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        moves.clear();
        let count = self.problem.activity_count();
        if count < 2 || budget.attempts == 0 || budget.max_moves == 0 || budget.max_segment_len == 0 {
            return Ok(ResourceAlnsGeneration::default());
        }
        if moves.capacity() < budget.max_moves {
            moves.reserve(budget.max_moves);
        }
        let max_len = budget.max_segment_len.min(count.saturating_sub(1)).max(1);
        let mut generation = ResourceAlnsGeneration::default();
        for attempt in 0..budget.attempts {
            checkpoint(stop)?;
            generation.attempts = generation.attempts.saturating_add(1);
            self.metrics.alns_generation_attempts = self.metrics.alns_generation_attempts.saturating_add(1);
            let attempt_u64 = u64::try_from(attempt).unwrap_or(u64::MAX);
            let choice = mix64(seed ^ attempt_u64.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let len = 1 + usize::try_from(choice).unwrap_or(usize::MAX) % max_len;
            let placements = count - len + 1;
            let first = usize::try_from(mix64(choice ^ 0xd1b5_4a32_d192_ed03)).unwrap_or(usize::MAX) % placements;
            let to = usize::try_from(mix64(choice ^ 0x94d0_49bb_1331_11eb)).unwrap_or(usize::MAX) % placements;
            if first == to {
                continue;
            }
            let movement = ResourceScheduleMove::SegmentRelocate { first, len, to };
            self.workspace.sync_priority(&self.priority, stop)?;
            let applied = apply_move(&mut self.workspace.order, movement);
            debug_assert!(applied);
            if !validate_reusable_priority(&self.problem, &self.workspace.order, &mut self.workspace.positions, stop)? {
                generation.precedence_rejections = generation.precedence_rejections.saturating_add(1);
                continue;
            }
            if moves.contains(&movement) {
                generation.duplicate_rejections = generation.duplicate_rejections.saturating_add(1);
                continue;
            }
            moves.push(movement);
            generation.generated = generation.generated.saturating_add(1);
            self.metrics.alns_moves_generated = self.metrics.alns_moves_generated.saturating_add(1);
            if moves.len() == budget.max_moves {
                break;
            }
        }
        moves.sort_unstable();
        Ok(generation)
    }

    /// Evaluate a candidate in reusable buffers, then commit only a fully
    /// feasible schedule accepted by the requested objective rule. Serial SGS
    /// moves recalculate only the changed suffix. Parallel SGS moves reuse all
    /// storage but conservatively recalculate the full event schedule.
    pub(crate) fn consider_move(
        &mut self,
        movement: ResourceScheduleMove,
        acceptance: ResourceMoveAcceptance,
        stop: &AtomicBool,
    ) -> Result<ResourceMoveOutcome, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        self.metrics.moves_considered = self.metrics.moves_considered.saturating_add(1);
        self.workspace.sync_priority(&self.priority, stop)?;
        if !apply_move(&mut self.workspace.order, movement) {
            return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Invalid));
        }
        if !validate_reusable_priority(&self.problem, &self.workspace.order, &mut self.workspace.positions, stop)? {
            self.metrics.precedence_rejections = self.metrics.precedence_rejections.saturating_add(1);
            return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Precedence));
        }
        let changed_cut = first_priority_difference(self.priority.order(), &self.workspace.order);
        if changed_cut == self.problem.activity_count() {
            return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Invalid));
        }
        let cut = if self.scheme == GenerationScheme::Serial && !self.serial_delta_valid { 0 } else { changed_cut };

        self.metrics.delta_evaluations = self.metrics.delta_evaluations.saturating_add(1);
        if self.scheme == GenerationScheme::Parallel {
            self.metrics.full_workspace_evaluations = self.metrics.full_workspace_evaluations.saturating_add(1);
        }
        let evaluated = match self.scheme {
            GenerationScheme::Serial => {
                evaluate_serial_delta(&self.problem, &self.priority, &self.reconstruction, &mut self.workspace, cut, stop)
            }
            GenerationScheme::Parallel => evaluate_parallel_reusable(&self.problem, &mut self.workspace, stop),
        };
        let stats = self.workspace.stats;
        let rescheduled = self.workspace.rescheduled;
        let observed_profile_events = evaluated.as_ref().map_or(0, |candidate| candidate.profile_events);
        self.metrics.record_candidate_work(stats, observed_profile_events, rescheduled);
        let candidate = match evaluated {
            Ok(candidate) => {
                self.metrics.reconstructions = self.metrics.reconstructions.saturating_add(1);
                candidate
            }
            Err(ReconstructionFailure::Interrupted) => {
                self.metrics.workspace_rollbacks = self.metrics.workspace_rollbacks.saturating_add(1);
                return Err(ResourceScheduleInterrupted);
            }
            Err(ReconstructionFailure::Infeasible) => {
                self.metrics.workspace_rollbacks = self.metrics.workspace_rollbacks.saturating_add(1);
                self.metrics.infeasible_rejections = self.metrics.infeasible_rejections.saturating_add(1);
                return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Infeasible));
            }
            Err(ReconstructionFailure::Numeric) => {
                self.metrics.workspace_rollbacks = self.metrics.workspace_rollbacks.saturating_add(1);
                return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Numeric));
            }
        };
        let previous = self.makespan();
        let current = candidate.makespan;
        if !acceptance.accepts(previous, current) {
            if self.rollback_workspace_after_candidate(cut).is_err() {
                return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Numeric));
            }
            self.metrics.objective_rejections = self.metrics.objective_rejections.saturating_add(1);
            return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::NotAccepted { current: previous, candidate: current }));
        }

        // A complete, independent reconstruction is the acceptance oracle.
        // It may allocate because it runs only for provisionally accepted
        // moves, never for the rejected-candidate hot path.
        let candidate_priority = PrioritySgs { order: self.workspace.order.clone(), positions: self.workspace.positions.clone() };
        let (oracle, oracle_stats) = match reconstruct(&self.problem, &candidate_priority, self.scheme, stop) {
            Ok(value) => value,
            Err(ReconstructionFailure::Interrupted) => {
                self.rollback_workspace_after_candidate(cut).map_err(|_| ResourceScheduleInterrupted)?;
                return Err(ResourceScheduleInterrupted);
            }
            Err(ReconstructionFailure::Infeasible) => {
                if self.rollback_workspace_after_candidate(cut).is_err() {
                    return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Numeric));
                }
                self.metrics.infeasible_rejections = self.metrics.infeasible_rejections.saturating_add(1);
                return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Infeasible));
            }
            Err(ReconstructionFailure::Numeric) => {
                if self.rollback_workspace_after_candidate(cut).is_err() {
                    return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Numeric));
                }
                return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Numeric));
            }
        };
        self.metrics.oracle_validations = self.metrics.oracle_validations.saturating_add(1);
        self.metrics.record_search_reconstruction(oracle_stats, oracle.profile_events);
        let candidate_starts = match self.scheme {
            GenerationScheme::Serial => &self.workspace.starts,
            GenerationScheme::Parallel => &self.workspace.scratch_starts,
        };
        if candidate.makespan != oracle.makespan || *candidate_starts != oracle.starts {
            self.metrics.oracle_mismatches = self.metrics.oracle_mismatches.saturating_add(1);
        }
        if !acceptance.accepts(previous, oracle.makespan) {
            if self.rollback_workspace_after_candidate(cut).is_err() {
                return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Numeric));
            }
            self.metrics.objective_rejections = self.metrics.objective_rejections.saturating_add(1);
            return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::NotAccepted { current: previous, candidate: oracle.makespan }));
        }
        // Prepare the complete physical state before changing the incumbent.
        // If this structurally bounded reload ever fails, restore the old state
        // and reject rather than publishing an incumbent with a broken cache.
        match self.workspace.reload_profiles(&self.problem, &oracle, stop) {
            Ok(Some(())) => {}
            Ok(None) => {
                if self.rollback_workspace_after_candidate(cut).is_err() {
                    return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Numeric));
                }
                return Ok(ResourceMoveOutcome::Rejected(ResourceMoveRejection::Numeric));
            }
            Err(ResourceScheduleInterrupted) => {
                self.rollback_workspace_after_candidate(cut).map_err(|_| ResourceScheduleInterrupted)?;
                return Err(ResourceScheduleInterrupted);
            }
        }
        self.workspace.touched.clear();
        self.priority = candidate_priority;
        self.reconstruction = oracle;
        self.serial_delta_valid = true;
        self.metrics.moves_accepted = self.metrics.moves_accepted.saturating_add(1);
        Ok(ResourceMoveOutcome::Accepted { previous, current: self.makespan() })
    }

    fn restore_committed_workspace(&mut self) -> Result<(), ReconstructionFailure> {
        self.workspace.sync_priority_atomically(&self.priority);
        let restore_stop = AtomicBool::new(false);
        if matches!(self.workspace.reload_profiles(&self.problem, &self.reconstruction, &restore_stop), Ok(Some(()))) {
            self.workspace.touched.clear();
            return Ok(());
        }
        let growths = self.workspace.growths.saturating_add(1);
        let mut replacement = ResourceEvaluationWorkspace::new(&self.problem, &self.priority, &self.reconstruction, &restore_stop)
            .map_err(|_| ReconstructionFailure::Interrupted)?
            .ok_or(ReconstructionFailure::Numeric)?;
        replacement.growths = growths;
        self.workspace = replacement;
        Ok(())
    }

    fn rollback_workspace_after_candidate(&mut self, cut: usize) -> Result<(), ReconstructionFailure> {
        let rollback = match self.scheme {
            GenerationScheme::Serial => rollback_serial_workspace(
                &self.problem,
                &self.priority,
                &self.reconstruction,
                &mut self.workspace,
                cut,
                self.problem.activity_count().saturating_sub(cut),
            ),
            GenerationScheme::Parallel => {
                self.workspace.sync_priority_atomically(&self.priority);
                self.workspace.touched.clear();
                Ok(())
            }
        };
        if rollback.is_err() {
            self.restore_committed_workspace()?;
        }
        self.workspace.sync_priority_atomically(&self.priority);
        self.metrics.workspace_rollbacks = self.metrics.workspace_rollbacks.saturating_add(1);
        Ok(())
    }

    /// Apply left, right, or double justification atomically. A completed
    /// justification is committed only when it does not worsen the makespan.
    pub(crate) fn justify(
        &mut self,
        kind: Justification,
        stop: &AtomicBool,
    ) -> Result<Option<JustificationOutcome>, ResourceScheduleInterrupted> {
        checkpoint(stop)?;
        let previous = self.makespan();
        let (candidate_priority, candidate, stats, candidate_scheme) = match kind {
            Justification::Left => {
                let (candidate, stats) = match serial_reconstruction(&self.problem, &self.priority, stop) {
                    Ok(value) => value,
                    Err(ReconstructionFailure::Interrupted) => return Err(ResourceScheduleInterrupted),
                    Err(_) => return Ok(None),
                };
                (self.priority.clone(), candidate, stats, GenerationScheme::Serial)
            }
            Justification::Right => {
                let (candidate, stats) = match right_reconstruction(&self.problem, &self.priority, previous, stop) {
                    Ok(value) => value,
                    Err(ReconstructionFailure::Interrupted) => return Err(ResourceScheduleInterrupted),
                    Err(_) => return Ok(None),
                };
                (self.priority.clone(), candidate, stats, self.scheme)
            }
            Justification::Double => {
                let (right, mut stats) = match right_reconstruction(&self.problem, &self.priority, previous, stop) {
                    Ok(value) => value,
                    Err(ReconstructionFailure::Interrupted) => return Err(ResourceScheduleInterrupted),
                    Err(_) => return Ok(None),
                };
                let priority = match priority_from_schedule(&self.problem, &right.starts, &self.priority, stop) {
                    Ok(priority) => priority,
                    Err(ReconstructionFailure::Interrupted) => return Err(ResourceScheduleInterrupted),
                    Err(_) => return Ok(None),
                };
                let (candidate, left_stats) = match serial_reconstruction(&self.problem, &priority, stop) {
                    Ok(value) => value,
                    Err(ReconstructionFailure::Interrupted) => return Err(ResourceScheduleInterrupted),
                    Err(_) => return Ok(None),
                };
                stats.candidates = stats.candidates.saturating_add(left_stats.candidates);
                stats.profile_checks = stats.profile_checks.saturating_add(left_stats.profile_checks);
                stats.event_visits = stats.event_visits.saturating_add(left_stats.event_visits);
                (priority, candidate, stats, GenerationScheme::Serial)
            }
        };
        if candidate.makespan > previous {
            return Ok(None);
        }
        self.workspace.sync_priority(&candidate_priority, stop)?;
        match self.workspace.reload_profiles(&self.problem, &candidate, stop) {
            Ok(Some(())) => {}
            Ok(None) => return Ok(None),
            Err(ResourceScheduleInterrupted) => return Err(ResourceScheduleInterrupted),
        }
        self.workspace.touched.clear();
        self.metrics.record_search_reconstruction(stats, candidate.profile_events);
        if kind == Justification::Double {
            self.metrics.reconstructions = self.metrics.reconstructions.saturating_add(1);
        }
        match kind {
            Justification::Left => self.metrics.left_justifications = self.metrics.left_justifications.saturating_add(1),
            Justification::Right => self.metrics.right_justifications = self.metrics.right_justifications.saturating_add(1),
            Justification::Double => self.metrics.double_justifications = self.metrics.double_justifications.saturating_add(1),
        }
        let changed = self.reconstruction.starts != candidate.starts || self.priority != candidate_priority;
        self.priority = candidate_priority;
        self.reconstruction = candidate;
        self.scheme = candidate_scheme;
        self.serial_delta_valid = candidate_scheme != GenerationScheme::Serial || kind != Justification::Right;
        Ok(Some(JustificationOutcome { previous, current: self.makespan(), changed }))
    }

    pub(crate) fn to_solution(&self) -> CollectionSolution {
        CollectionSolution {
            lists: Vec::new(),
            objectives: vec![self.makespan()],
            feasible: true,
            starts: self.reconstruction.starts.clone(),
            presences: vec![true; self.problem.activity_count()],
            machines: self.problem.machines.clone(),
            modes: self.problem.modes.clone(),
            bound: None,
        }
    }
}

fn apply_move(order: &mut [usize], movement: ResourceScheduleMove) -> bool {
    match movement {
        ResourceScheduleMove::AdjacentSwap { first_position } => {
            let Some(second) = first_position.checked_add(1) else {
                return false;
            };
            if second >= order.len() {
                return false;
            }
            order.swap(first_position, second);
            true
        }
        ResourceScheduleMove::Relocate { from, to } => apply_segment_relocate(order, from, 1, to),
        ResourceScheduleMove::SegmentRelocate { first, len, to } => apply_segment_relocate(order, first, len, to),
    }
}

fn apply_segment_relocate(order: &mut [usize], first: usize, len: usize, to: usize) -> bool {
    if len == 0 || first >= order.len() || len > order.len() - first || to > order.len() - len || first == to {
        return false;
    }
    if to < first {
        order[to..first + len].rotate_right(len);
    } else {
        order[first..to + len].rotate_left(len);
    }
    true
}
