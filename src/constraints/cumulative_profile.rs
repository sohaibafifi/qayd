//! Opt-in compulsory-part event maintenance for native cumulative resources.
//!
//! Occurrences own distinct cache slots even when their interval IDs repeat.
//! The event map preserves referenced zero-net endpoints, so materialization
//! produces exactly the same event-delimited segments as a fresh sorted build.
//! Reading the cache first restores its owned undo log to a root-allocated
//! reversible cursor. Profile materialization remains a linear event scan.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::constraints::resource_profile::ProfileSegment;
use crate::store::Store;
use crate::trail::ReversibleInt;

pub(crate) type CompulsoryPart = (i128, i128, i128);
pub(crate) const MAX_PROFILE_UNDO_RECORDS: usize = 65_536;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CumulativeStats {
    pub propagation_calls: u64,
    pub profile_rebuilds: u64,
    pub incremental_updates: u64,
    pub task_comparisons: u64,
    pub changed_tasks: u64,
    pub endpoint_updates: u64,
    pub segments_emitted: u64,
    pub undo_records: u64,
    pub restored_tasks: u64,
    pub peak_undo_records: u64,
    pub peak_event_points: u64,
    pub fallback_calls: u64,
    pub interrupted: u64,
    pub reason_calls: u64,
    pub original_premises: u64,
    pub emitted_premises: u64,
    pub lifted_bounds: u64,
    pub omitted_tautologies: u64,
    pub timetable_updates: u64,
    pub timetable_conflicts: u64,
}

impl CumulativeStats {
    pub fn merge(&mut self, other: Self) {
        self.propagation_calls = self.propagation_calls.saturating_add(other.propagation_calls);
        self.profile_rebuilds = self.profile_rebuilds.saturating_add(other.profile_rebuilds);
        self.incremental_updates = self.incremental_updates.saturating_add(other.incremental_updates);
        self.task_comparisons = self.task_comparisons.saturating_add(other.task_comparisons);
        self.changed_tasks = self.changed_tasks.saturating_add(other.changed_tasks);
        self.endpoint_updates = self.endpoint_updates.saturating_add(other.endpoint_updates);
        self.segments_emitted = self.segments_emitted.saturating_add(other.segments_emitted);
        self.undo_records = self.undo_records.saturating_add(other.undo_records);
        self.restored_tasks = self.restored_tasks.saturating_add(other.restored_tasks);
        self.peak_undo_records = self.peak_undo_records.max(other.peak_undo_records);
        self.peak_event_points = self.peak_event_points.max(other.peak_event_points);
        self.fallback_calls = self.fallback_calls.saturating_add(other.fallback_calls);
        self.interrupted = self.interrupted.saturating_add(other.interrupted);
        self.reason_calls = self.reason_calls.saturating_add(other.reason_calls);
        self.original_premises = self.original_premises.saturating_add(other.original_premises);
        self.emitted_premises = self.emitted_premises.saturating_add(other.emitted_premises);
        self.lifted_bounds = self.lifted_bounds.saturating_add(other.lifted_bounds);
        self.omitted_tautologies = self.omitted_tautologies.saturating_add(other.omitted_tautologies);
        self.timetable_updates = self.timetable_updates.saturating_add(other.timetable_updates);
        self.timetable_conflicts = self.timetable_conflicts.saturating_add(other.timetable_conflicts);
    }
}

/// Shared optional counters survive solver cloning and worker ownership.
#[derive(Clone, Debug, Default)]
pub struct CumulativeMetricsHandle(Arc<Mutex<CumulativeStats>>);

impl CumulativeMetricsHandle {
    pub fn stats(&self) -> CumulativeStats {
        *self.0.lock().expect("cumulative metrics mutex poisoned")
    }

    pub(crate) fn record(&self, stats: CumulativeStats) {
        self.0.lock().expect("cumulative metrics mutex poisoned").merge(stats);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProfileUpdate {
    Ready,
    Fallback,
    Interrupted,
}

#[derive(Clone, Copy, Debug, Default)]
struct Endpoint {
    delta: i128,
    references: usize,
}

#[derive(Clone, Copy, Debug)]
struct Undo {
    occurrence: usize,
    previous: Option<CompulsoryPart>,
}

#[derive(Clone, Debug)]
pub(crate) struct IncrementalProfile {
    cursor: ReversibleInt,
    parts: Vec<Option<CompulsoryPart>>,
    endpoints: BTreeMap<i128, Endpoint>,
    undo: Vec<Undo>,
    log_limit: usize,
    frozen: bool,
    ready: bool,
}

fn stopped(stop: Option<&dyn Fn() -> bool>) -> bool {
    stop.is_some_and(|stop| stop())
}

fn active_part(part: Option<CompulsoryPart>) -> Option<CompulsoryPart> {
    part.filter(|&(start, end, demand)| start < end && demand != 0)
}

impl IncrementalProfile {
    /// The reversible handle is never allocated below root. Each complete
    /// workspace clone must travel with the matching Store clone.
    pub(crate) fn new(store: &mut Store, occurrences: usize) -> Option<Self> {
        Self::with_log_limit(store, occurrences, MAX_PROFILE_UNDO_RECORDS)
    }

    pub(crate) fn with_log_limit(store: &mut Store, occurrences: usize, limit: usize) -> Option<Self> {
        if store.level() != 0 {
            return None;
        }
        Some(Self {
            cursor: store.new_rev_int(0),
            parts: vec![None; occurrences],
            endpoints: BTreeMap::new(),
            undo: Vec::new(),
            log_limit: limit.min(i32::MAX as usize),
            frozen: false,
            ready: false,
        })
    }

    fn endpoint(&mut self, time: i128, delta: i128, adding: bool, stats: &mut CumulativeStats) {
        if adding {
            let entry = self.endpoints.entry(time).or_default();
            entry.delta += delta;
            entry.references += 1;
        } else {
            let entry = self.endpoints.get_mut(&time).expect("cached compulsory endpoint must exist");
            entry.delta -= delta;
            entry.references -= 1;
            if entry.references == 0 {
                assert_eq!(entry.delta, 0, "unreferenced endpoint must have zero net demand");
                self.endpoints.remove(&time);
            }
        }
        stats.endpoint_updates = stats.endpoint_updates.saturating_add(1);
        stats.peak_event_points = stats.peak_event_points.max(self.endpoints.len() as u64);
    }

    /// Remove/add all endpoints of one occurrence without cancellation between
    /// them. The operation is bounded by four ordered-map mutations.
    fn replace(&mut self, occurrence: usize, part: Option<CompulsoryPart>, stats: &mut CumulativeStats) {
        if let Some((start, end, demand)) = self.parts[occurrence] {
            self.endpoint(start, demand, false, stats);
            self.endpoint(end, -demand, false, stats);
        }
        if let Some((start, end, demand)) = part {
            self.endpoint(start, demand, true, stats);
            self.endpoint(end, -demand, true, stats);
        }
        self.parts[occurrence] = part;
    }

    /// Lazy undo also works across several pops without an intervening read,
    /// and across siblings at the same depth. A partially restored cache keeps
    /// every remaining ancestor undo entry until the next invocation.
    fn restore(&mut self, store: &Store, stats: &mut CumulativeStats, stop: Option<&dyn Fn() -> bool>) -> ProfileUpdate {
        let target = usize::try_from(store.rev_get(self.cursor)).expect("nonnegative cumulative undo cursor");
        assert!(target <= self.undo.len(), "cumulative cache and Store must be cloned together");
        while self.undo.len() > target {
            self.ready = false;
            if stopped(stop) {
                stats.interrupted = stats.interrupted.saturating_add(1);
                return ProfileUpdate::Interrupted;
            }
            let record = self.undo.pop().unwrap();
            self.replace(record.occurrence, record.previous, stats);
            stats.restored_tasks = stats.restored_tasks.saturating_add(1);
        }
        if store.level() == 0 {
            assert_eq!(target, 0, "root profile cursor must be compacted");
            self.frozen = false;
        }
        ProfileUpdate::Ready
    }

    /// Compare every occurrence, editing only changed compulsory parts.
    /// Demands must fit nonnegative i64, as in the native constant adapter;
    /// coordinates themselves are i128 and are never incremented here.
    pub(crate) fn update(
        &mut self,
        store: &mut Store,
        parts: impl ExactSizeIterator<Item = Option<CompulsoryPart>>,
        stats: &mut CumulativeStats,
        stop: Option<&dyn Fn() -> bool>,
    ) -> ProfileUpdate {
        self.ready = false;
        if self.restore(store, stats, stop) == ProfileUpdate::Interrupted {
            return ProfileUpdate::Interrupted;
        }
        if stopped(stop) {
            stats.interrupted = stats.interrupted.saturating_add(1);
            return ProfileUpdate::Interrupted;
        }
        if self.frozen || parts.len() != self.parts.len() {
            stats.fallback_calls = stats.fallback_calls.saturating_add(1);
            return ProfileUpdate::Fallback;
        }
        stats.incremental_updates = stats.incremental_updates.saturating_add(1);
        for (occurrence, part) in parts.enumerate() {
            if stopped(stop) {
                stats.interrupted = stats.interrupted.saturating_add(1);
                return ProfileUpdate::Interrupted;
            }
            let part = active_part(part);
            if part.is_some_and(|(_, _, demand)| !(0..=i128::from(i64::MAX)).contains(&demand)) {
                self.frozen = true;
                stats.fallback_calls = stats.fallback_calls.saturating_add(1);
                return ProfileUpdate::Fallback;
            }
            stats.task_comparisons = stats.task_comparisons.saturating_add(1);
            if self.parts[occurrence] == part {
                continue;
            }
            if store.level() != 0 {
                if self.undo.len() >= self.log_limit {
                    // Never discard an ancestor's inverse update to make room.
                    self.frozen = true;
                    stats.fallback_calls = stats.fallback_calls.saturating_add(1);
                    return ProfileUpdate::Fallback;
                }
                self.undo.push(Undo { occurrence, previous: self.parts[occurrence] });
                store.rev_set(self.cursor, self.undo.len() as i32);
                stats.undo_records = stats.undo_records.saturating_add(1);
                stats.peak_undo_records = stats.peak_undo_records.max(self.undo.len() as u64);
            }
            self.replace(occurrence, part, stats);
            stats.changed_tasks = stats.changed_tasks.saturating_add(1);
        }
        self.ready = true;
        ProfileUpdate::Ready
    }

    /// Emit a complete profile after a successful update. Interrupted output
    /// is cleared, and lazy undo prevents stale branch data from being read.
    /// A fresh update is required if backtracking changed the cached snapshot.
    pub(crate) fn materialize(
        &mut self,
        store: &Store,
        output: &mut Vec<ProfileSegment>,
        stats: &mut CumulativeStats,
        stop: Option<&dyn Fn() -> bool>,
    ) -> ProfileUpdate {
        output.clear();
        if self.restore(store, stats, stop) == ProfileUpdate::Interrupted {
            return ProfileUpdate::Interrupted;
        }
        if !self.ready || self.frozen {
            return ProfileUpdate::Fallback;
        }
        let mut usage = 0;
        let mut previous = None;
        for (&time, endpoint) in &self.endpoints {
            if stopped(stop) {
                output.clear();
                stats.interrupted = stats.interrupted.saturating_add(1);
                return ProfileUpdate::Interrupted;
            }
            if let Some(start) = previous {
                if start < time && usage != 0 {
                    output.push(ProfileSegment { start, end: time, usage });
                    stats.segments_emitted = stats.segments_emitted.saturating_add(1);
                }
            }
            usage += endpoint.delta;
            previous = Some(time);
        }
        assert_eq!(usage, 0, "compulsory endpoint demands must balance");
        if stopped(stop) {
            output.clear();
            stats.interrupted = stats.interrupted.saturating_add(1);
            return ProfileUpdate::Interrupted;
        }
        ProfileUpdate::Ready
    }
}
