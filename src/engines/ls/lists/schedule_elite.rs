//! Bounded deterministic elite archive for strict job-shop search.
//!
//! This is deliberately a search-memory component, not a publication path.
//! Entries contain only compressed machine orders and an objective observed on
//! a fully reconstructed [`JobShopState`]. A caller that wants to use an entry
//! must rebuild it through the schedule oracle before it can become an
//! incumbent.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;

#[cfg(test)]
use std::cell::Cell;

use crate::mix64;

use super::schedule_state::JobShopState;

/// The archive stays intentionally small on million-operation instances.
pub(crate) const SCHEDULE_ELITE_CAPACITY: usize = 4;
const ARC_BOTTOM_K: usize = 256;
const DISTANCE_SCALE: u32 = 1_000_000;
const STOP_POLL_MASK: usize = 4_095;
const ORDER_HASH_SALT: u64 = 0x37e2_17c1_5b09_4f63;
const ARC_HASH_SALT: u64 = 0x8c85_2f98_617a_d341;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScheduleEliteError {
    Interrupted,
    EncodingOverflow,
    InvalidMachineOrder,
    InvalidObjective,
    IncompatibleProblem,
    IncompatibleSolve,
}

/// Exact in-process identity of one schedule solve.
///
/// Pointer identity is collision-free while an archive retains the token. The
/// token is allocated once by the orchestrator and cloned into worker sessions
/// and compact snapshots, without copying the static problem identity.
#[derive(Clone, Debug)]
pub(crate) struct ScheduleEliteSolveToken(Arc<()>);

impl ScheduleEliteSolveToken {
    pub(crate) fn new() -> Self {
        Self(Arc::new(()))
    }

    pub(crate) fn same_solve(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScheduleEliteConsiderOutcome {
    /// The entry was added without removing an archive member.
    Inserted,
    /// The entry survived the objective/diversity frontier and evicted one
    /// member from the full archive.
    InsertedAndEvicted,
    /// The exact order was already stored with the same oracle objective.
    Duplicate,
    /// The candidate did not survive the capacity-four farthest-first frontier.
    Dominated,
    /// The candidate has a different exact problem identity.
    Incompatible,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScheduleEliteBatchOutcome {
    pub(crate) considered: usize,
    pub(crate) duplicates: usize,
    pub(crate) retained: usize,
    pub(crate) dominated: usize,
    pub(crate) evicted: usize,
}

/// Exact static identity shared by every entry of one archive.
///
/// Keeping this behind an `Arc` avoids repeating tens of megabytes of JSSP
/// data in each capacity-four entry. Compatibility is checked field by field,
/// never inferred from a probabilistic fingerprint.
#[derive(Debug)]
struct ProblemIdentity {
    operation_count: u32,
    machine_count: u32,
    machines: Box<[u32]>,
    raw_machines: Box<[u64]>,
    solution_machines: Box<[i64]>,
    solution_modes: Box<[Option<u64>]>,
    durations: Box<[i64]>,
    horizons: Box<[i64]>,
    start_windows: Box<[(i64, i64)]>,
    predecessor_offsets: Box<[u32]>,
    predecessors: Box<[u32]>,
}

impl ProblemIdentity {
    fn capture(state: &JobShopState, stop: &AtomicBool) -> Result<Self, ScheduleEliteError> {
        checkpoint(stop)?;
        let operation_count = u32::try_from(state.operation_count()).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
        let machine_count = u32::try_from(state.machine_count()).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
        let mut machines = Vec::with_capacity(state.operation_count());
        let mut raw_machines = Vec::with_capacity(state.operation_count());
        let mut solution_machines = Vec::with_capacity(state.operation_count());
        let mut solution_modes = Vec::with_capacity(state.operation_count());
        let mut durations = Vec::with_capacity(state.operation_count());
        let mut horizons = Vec::with_capacity(state.operation_count());
        let mut start_windows = Vec::with_capacity(state.operation_count());
        let mut predecessor_offsets = Vec::with_capacity(state.operation_count().saturating_add(1));
        let mut predecessors = Vec::with_capacity(state.operation_count());
        predecessor_offsets.push(0);
        let mut arc_count = 0usize;

        for operation in 0..state.operation_count() {
            poll(stop, operation)?;
            machines.push(u32::try_from(state.machine(operation)).map_err(|_| ScheduleEliteError::EncodingOverflow)?);
            raw_machines.push(
                u64::try_from(state.raw_machine(operation).ok_or(ScheduleEliteError::InvalidMachineOrder)?)
                    .map_err(|_| ScheduleEliteError::EncodingOverflow)?,
            );
            solution_machines.push(state.solution_machine(operation).ok_or(ScheduleEliteError::InvalidMachineOrder)?);
            solution_modes.push(
                state
                    .solution_mode(operation)
                    .ok_or(ScheduleEliteError::InvalidMachineOrder)?
                    .map(u64::try_from)
                    .transpose()
                    .map_err(|_| ScheduleEliteError::EncodingOverflow)?,
            );
            durations.push(state.duration(operation));
            horizons.push(state.horizon(operation));
            start_windows.push(state.start_window(operation));
            for &predecessor in state.job_predecessors(operation) {
                poll(stop, arc_count)?;
                predecessors.push(u32::try_from(predecessor).map_err(|_| ScheduleEliteError::EncodingOverflow)?);
                arc_count = arc_count.saturating_add(1);
            }
            predecessor_offsets.push(u32::try_from(arc_count).map_err(|_| ScheduleEliteError::EncodingOverflow)?);
        }

        Ok(Self {
            operation_count,
            machine_count,
            machines: machines.into_boxed_slice(),
            raw_machines: raw_machines.into_boxed_slice(),
            solution_machines: solution_machines.into_boxed_slice(),
            solution_modes: solution_modes.into_boxed_slice(),
            durations: durations.into_boxed_slice(),
            horizons: horizons.into_boxed_slice(),
            start_windows: start_windows.into_boxed_slice(),
            predecessor_offsets: predecessor_offsets.into_boxed_slice(),
            predecessors: predecessors.into_boxed_slice(),
        })
    }

    fn matches_state(&self, state: &JobShopState, stop: &AtomicBool) -> Result<bool, ScheduleEliteError> {
        checkpoint(stop)?;
        if usize::try_from(self.operation_count).ok() != Some(state.operation_count())
            || usize::try_from(self.machine_count).ok() != Some(state.machine_count())
        {
            return Ok(false);
        }

        let mut arc_count = 0usize;
        for operation in 0..state.operation_count() {
            poll(stop, operation)?;
            if usize::try_from(self.machines[operation]).ok() != Some(state.machine(operation))
                || state.raw_machine(operation).and_then(|machine| u64::try_from(machine).ok()) != Some(self.raw_machines[operation])
                || state.solution_machine(operation) != Some(self.solution_machines[operation])
                || state.solution_mode(operation).and_then(|mode| mode.map(u64::try_from).transpose().ok())
                    != Some(self.solution_modes[operation])
                || self.durations[operation] != state.duration(operation)
                || self.horizons[operation] != state.horizon(operation)
                || self.start_windows[operation] != state.start_window(operation)
            {
                return Ok(false);
            }
            let start = usize::try_from(self.predecessor_offsets[operation]).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
            let end = usize::try_from(self.predecessor_offsets[operation + 1]).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
            let expected = &self.predecessors[start..end];
            let actual = state.job_predecessors(operation);
            if expected.len() != actual.len() {
                return Ok(false);
            }
            for (&expected, &actual) in expected.iter().zip(actual) {
                poll(stop, arc_count)?;
                if usize::try_from(expected).ok() != Some(actual) {
                    return Ok(false);
                }
                arc_count = arc_count.saturating_add(1);
            }
        }
        Ok(true)
    }

    fn heap_lower_bound_bytes(&self) -> usize {
        size_of::<Self>()
            .saturating_add(self.machines.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.raw_machines.len().saturating_mul(size_of::<u64>()))
            .saturating_add(self.solution_machines.len().saturating_mul(size_of::<i64>()))
            .saturating_add(self.solution_modes.len().saturating_mul(size_of::<Option<u64>>()))
            .saturating_add(self.durations.len().saturating_mul(size_of::<i64>()))
            .saturating_add(self.horizons.len().saturating_mul(size_of::<i64>()))
            .saturating_add(self.start_windows.len().saturating_mul(size_of::<(i64, i64)>()))
            .saturating_add(self.predecessor_offsets.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.predecessors.len().saturating_mul(size_of::<u32>()))
    }
}

/// Compact owned worker snapshot waiting for the next global shadow merge.
///
/// It intentionally excludes the static problem identity. Every worker solves
/// the same recognized schedule, and the central archive captures that identity
/// exactly once from a returned session at the merge boundary.
#[derive(Clone, Debug)]
pub(crate) struct ScheduleEliteCandidate {
    solve_token: Option<ScheduleEliteSolveToken>,
    objective: i64,
    machine_offsets: Box<[u32]>,
    operations: Box<[u32]>,
    order_hash: u64,
    arc_bottom_k: Box<[u64]>,
}

impl ScheduleEliteCandidate {
    pub(crate) fn capture(
        state: &JobShopState,
        solve_token: &ScheduleEliteSolveToken,
        stop: &AtomicBool,
    ) -> Result<Self, ScheduleEliteError> {
        Self::capture_inner(state, Some(solve_token.clone()), stop)
    }

    fn capture_unscoped(state: &JobShopState, stop: &AtomicBool) -> Result<Self, ScheduleEliteError> {
        Self::capture_inner(state, None, stop)
    }

    fn capture_inner(
        state: &JobShopState,
        solve_token: Option<ScheduleEliteSolveToken>,
        stop: &AtomicBool,
    ) -> Result<Self, ScheduleEliteError> {
        checkpoint(stop)?;
        let operation_count = state.operation_count();
        u32::try_from(operation_count).map_err(|_| ScheduleEliteError::EncodingOverflow)?;

        let mut operations = Vec::with_capacity(operation_count);
        let mut machine_offsets = Vec::with_capacity(state.machine_count().saturating_add(1));
        machine_offsets.push(0);
        let mut order_hash = mix64(ORDER_HASH_SALT ^ u64::try_from(state.machine_count()).unwrap_or(u64::MAX));
        let mut arc_bottom_k = BTreeSet::new();
        let mut seen = 0usize;
        for (machine, sequence) in state.machine_sequences().iter().enumerate() {
            checkpoint(stop)?;
            order_hash = mix64(order_hash ^ mix64(u64::try_from(machine).unwrap_or(u64::MAX) ^ ORDER_HASH_SALT));
            for (position, &operation) in sequence.iter().enumerate() {
                poll(stop, seen)?;
                if operation >= operation_count || state.machine(operation) != machine {
                    return Err(ScheduleEliteError::InvalidMachineOrder);
                }
                let encoded = u32::try_from(operation).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
                operations.push(encoded);
                order_hash = mix64(order_hash ^ mix64(u64::from(encoded).wrapping_add(position as u64)));
                seen = seen.saturating_add(1);
            }
            for (position, pair) in sequence.windows(2).enumerate() {
                poll(stop, position)?;
                bottom_k_insert(&mut arc_bottom_k, machine_arc_hash(machine, pair[0], pair[1]));
            }
            machine_offsets.push(u32::try_from(seen).map_err(|_| ScheduleEliteError::EncodingOverflow)?);
        }
        if seen != operation_count {
            return Err(ScheduleEliteError::InvalidMachineOrder);
        }

        Ok(Self {
            solve_token,
            objective: state.makespan(),
            machine_offsets: machine_offsets.into_boxed_slice(),
            operations: operations.into_boxed_slice(),
            order_hash,
            arc_bottom_k: arc_bottom_k.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        })
    }

    pub(crate) fn heap_lower_bound_bytes(&self) -> usize {
        self.machine_offsets
            .len()
            .saturating_mul(size_of::<u32>())
            .saturating_add(self.operations.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.arc_bottom_k.len().saturating_mul(size_of::<u64>()))
    }

    fn validate_for_reference(
        &self,
        reference: &JobShopState,
        solve_token: &ScheduleEliteSolveToken,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleEliteError> {
        checkpoint(stop)?;
        if !self.solve_token.as_ref().is_some_and(|candidate_token| candidate_token.same_solve(solve_token)) {
            return Err(ScheduleEliteError::IncompatibleSolve);
        }
        let operation_count = reference.operation_count();
        if self.operations.len() != operation_count
            || self.machine_offsets.len() != reference.machine_count().saturating_add(1)
            || self.machine_offsets.first().copied() != Some(0)
            || self.machine_offsets.last().and_then(|&offset| usize::try_from(offset).ok()) != Some(operation_count)
        {
            return Err(ScheduleEliteError::InvalidMachineOrder);
        }

        let mut seen = vec![false; operation_count];
        let mut order_hash = mix64(ORDER_HASH_SALT ^ u64::try_from(reference.machine_count()).unwrap_or(u64::MAX));
        let mut arc_bottom_k = BTreeSet::new();
        let mut visited = 0usize;
        for machine in 0..reference.machine_count() {
            checkpoint(stop)?;
            let start = usize::try_from(self.machine_offsets[machine]).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
            let end = usize::try_from(self.machine_offsets[machine + 1]).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
            if start > end || end > operation_count {
                return Err(ScheduleEliteError::InvalidMachineOrder);
            }
            order_hash = mix64(order_hash ^ mix64(u64::try_from(machine).unwrap_or(u64::MAX) ^ ORDER_HASH_SALT));
            for (position, &encoded) in self.operations[start..end].iter().enumerate() {
                poll(stop, visited)?;
                let operation = usize::try_from(encoded).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
                if operation >= operation_count || seen[operation] || reference.machine(operation) != machine {
                    return Err(ScheduleEliteError::InvalidMachineOrder);
                }
                seen[operation] = true;
                order_hash = mix64(order_hash ^ mix64(u64::from(encoded).wrapping_add(position as u64)));
                visited = visited.saturating_add(1);
            }
            for (position, pair) in self.operations[start..end].windows(2).enumerate() {
                poll(stop, position)?;
                bottom_k_insert(
                    &mut arc_bottom_k,
                    machine_arc_hash(
                        machine,
                        usize::try_from(pair[0]).map_err(|_| ScheduleEliteError::EncodingOverflow)?,
                        usize::try_from(pair[1]).map_err(|_| ScheduleEliteError::EncodingOverflow)?,
                    ),
                );
            }
        }
        if visited != operation_count || seen.iter().any(|&present| !present) {
            return Err(ScheduleEliteError::InvalidMachineOrder);
        }
        let expected_sketch = arc_bottom_k.into_iter().collect::<Vec<_>>();
        if order_hash != self.order_hash || expected_sketch.as_slice() != self.arc_bottom_k.as_ref() {
            return Err(ScheduleEliteError::InvalidMachineOrder);
        }

        let mut objective_lower_bound = 0i64;
        let mut objective_upper_bound = 0i64;
        for operation in 0..operation_count {
            poll(stop, operation)?;
            objective_lower_bound = objective_lower_bound.max(
                reference
                    .start_window(operation)
                    .0
                    .checked_add(reference.duration(operation))
                    .ok_or(ScheduleEliteError::EncodingOverflow)?,
            );
            objective_upper_bound = objective_upper_bound.max(reference.horizon(operation));
        }
        if self.objective < objective_lower_bound || self.objective > objective_upper_bound {
            return Err(ScheduleEliteError::InvalidObjective);
        }
        Ok(())
    }

    fn into_entry(self, identity: Arc<ProblemIdentity>) -> ScheduleEliteEntry {
        ScheduleEliteEntry {
            objective: self.objective,
            machine_offsets: self.machine_offsets,
            operations: self.operations,
            identity,
            order_hash: self.order_hash,
            arc_bottom_k: self.arc_bottom_k,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_corrupt_first_operation(&mut self) {
        if let Some(operation) = self.operations.first_mut() {
            *operation = u32::MAX;
        }
    }

    #[cfg(test)]
    pub(crate) fn test_corrupt_objective(&mut self) {
        self.objective = i64::MAX;
    }
}

/// One compact search-only elite.
///
/// `machine_offsets` partitions `operations` into the authoritative order for
/// every compact machine. Both use `u32`, so a Large-TA entry needs about four
/// bytes per operation rather than a nested `usize` representation.
#[derive(Clone, Debug)]
pub(crate) struct ScheduleEliteEntry {
    objective: i64,
    machine_offsets: Box<[u32]>,
    operations: Box<[u32]>,
    identity: Arc<ProblemIdentity>,
    order_hash: u64,
    arc_bottom_k: Box<[u64]>,
}

impl ScheduleEliteEntry {
    pub(crate) const fn objective(&self) -> i64 {
        self.objective
    }

    pub(crate) fn machine_count(&self) -> usize {
        self.machine_offsets.len().saturating_sub(1)
    }

    pub(crate) fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub(crate) const fn order_hash(&self) -> u64 {
        self.order_hash
    }

    pub(crate) fn matches_state(&self, state: &JobShopState, stop: &AtomicBool) -> Result<bool, ScheduleEliteError> {
        self.identity.matches_state(state, stop)
    }

    pub(crate) fn machine_sequence(&self, machine: usize) -> Option<&[u32]> {
        let start = usize::try_from(*self.machine_offsets.get(machine)?).ok()?;
        let end = usize::try_from(*self.machine_offsets.get(machine.checked_add(1)?)?).ok()?;
        self.operations.get(start..end)
    }

    /// Estimated Jaccard distance in millionths, from `0` to `1_000_000`.
    ///
    /// The stable bottom-k sketch does not saturate as the operation count
    /// grows. For fewer than 256 machine arcs it stores every arc hash and the
    /// distance is exact. Entries are comparable only when they share the exact
    /// archive identity.
    pub(crate) fn arc_distance(&self, other: &Self) -> Option<u32> {
        if !Arc::ptr_eq(&self.identity, &other.identity) || self.machine_offsets != other.machine_offsets {
            return None;
        }
        Some(bottom_k_jaccard_distance(&self.arc_bottom_k, &other.arc_bottom_k))
    }

    /// Decode an order for a later full reconstruction. This does not make the
    /// entry publishable by itself.
    pub(crate) fn decode_machine_sequences(&self, stop: &AtomicBool) -> Result<Vec<Vec<usize>>, ScheduleEliteError> {
        checkpoint(stop)?;
        let mut sequences = Vec::with_capacity(self.machine_count());
        for machine in 0..self.machine_count() {
            checkpoint(stop)?;
            let encoded = self.machine_sequence(machine).ok_or(ScheduleEliteError::InvalidMachineOrder)?;
            let mut sequence = Vec::with_capacity(encoded.len());
            for (index, &operation) in encoded.iter().enumerate() {
                poll(stop, index)?;
                sequence.push(usize::try_from(operation).map_err(|_| ScheduleEliteError::EncodingOverflow)?);
            }
            sequences.push(sequence);
        }
        Ok(sequences)
    }

    fn heap_lower_bound_bytes(&self) -> usize {
        self.machine_offsets
            .len()
            .saturating_mul(size_of::<u32>())
            .saturating_add(self.operations.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.arc_bottom_k.len().saturating_mul(size_of::<u64>()))
    }
}

/// Worker-local capacity-four archive.
///
/// The first entry is always the best objective. Remaining entries are chosen
/// deterministically by repeatedly maximizing their minimum estimated Jaccard
/// distance from the already selected entries. Objective, stable order hash,
/// and compressed lexicographic order break ties in that order.
#[derive(Debug)]
pub(crate) struct ScheduleEliteArchive {
    solve_token: Option<ScheduleEliteSolveToken>,
    identity: Option<Arc<ProblemIdentity>>,
    entries: Vec<ScheduleEliteEntry>,
}

impl Default for ScheduleEliteArchive {
    fn default() -> Self {
        Self::new()
    }
}

impl ScheduleEliteArchive {
    pub(crate) fn new() -> Self {
        Self { solve_token: None, identity: None, entries: Vec::new() }
    }

    pub(crate) const fn capacity(&self) -> usize {
        SCHEDULE_ELITE_CAPACITY
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn best(&self) -> Option<&ScheduleEliteEntry> {
        self.entries.first()
    }

    pub(crate) fn entries(&self) -> &[ScheduleEliteEntry] {
        &self.entries
    }

    /// Known heap allocations owned by the archive.
    ///
    /// This deliberately remains a lower bound: allocator metadata, capacity
    /// rounding, and the `Arc` headers are not observable here.
    pub(crate) fn heap_lower_bound_bytes(&self) -> usize {
        let identity = self.identity_heap_lower_bound_bytes();
        self.entries
            .capacity()
            .saturating_mul(size_of::<ScheduleEliteEntry>())
            .saturating_add(identity)
            .saturating_add(self.entries.iter().map(ScheduleEliteEntry::heap_lower_bound_bytes).sum::<usize>())
    }

    pub(crate) fn identity_heap_lower_bound_bytes(&self) -> usize {
        self.identity.as_ref().map_or(0, |identity| identity.heap_lower_bound_bytes())
    }

    /// Lower bound at the busiest known batch stages, including simultaneous
    /// worker snapshots and old/new archive buffers during replacement.
    pub(crate) fn candidate_batch_peak_heap_lower_bound(&self, candidates: &[ScheduleEliteCandidate], candidate_capacity: usize) -> usize {
        let archive = self.heap_lower_bound_bytes();
        let pending = self.pending_candidates_heap_lower_bound(candidates, candidate_capacity);
        let snapshot_payload = candidates.iter().map(ScheduleEliteCandidate::heap_lower_bound_bytes).sum::<usize>();
        let encoded_buffer = candidates.len().saturating_mul(size_of::<ScheduleEliteEntry>());
        let validation_bitset = candidates.first().map_or(0, |candidate| candidate.operations.len().saturating_add(7) / 8);
        let incoming = pending.saturating_add(validation_bitset);
        let encoding = incoming.saturating_add(encoded_buffer);
        let source_count = self.entries.len().saturating_add(candidates.len());
        let frontier = archive
            .saturating_add(snapshot_payload)
            .saturating_add(encoded_buffer)
            .saturating_add(source_count.saturating_mul(size_of::<EntrySource>()))
            .saturating_add(SCHEDULE_ELITE_CAPACITY.saturating_mul(size_of::<usize>()));
        let replacement = frontier
            .saturating_add(self.entries.len().saturating_mul(size_of::<Option<ScheduleEliteEntry>>()))
            .saturating_add(candidates.len().saturating_mul(size_of::<Option<ScheduleEliteEntry>>()))
            .saturating_add(SCHEDULE_ELITE_CAPACITY.saturating_mul(size_of::<ScheduleEliteEntry>()));
        incoming.max(encoding).max(frontier).max(replacement)
    }

    pub(crate) fn pending_candidates_heap_lower_bound(&self, candidates: &[ScheduleEliteCandidate], candidate_capacity: usize) -> usize {
        self.heap_lower_bound_bytes()
            .saturating_add(candidates.iter().map(ScheduleEliteCandidate::heap_lower_bound_bytes).sum::<usize>())
            .saturating_add(candidate_capacity.saturating_mul(size_of::<ScheduleEliteCandidate>()))
    }

    pub(crate) fn objectives_summary(&self) -> String {
        if self.entries.is_empty() {
            return "none".to_string();
        }
        self.entries.iter().map(|entry| entry.objective.to_string()).collect::<Vec<_>>().join(",")
    }

    pub(crate) fn pairwise_distances_ppm_summary(&self) -> String {
        let mut distances = Vec::new();
        for left in 0..self.entries.len() {
            for right in left + 1..self.entries.len() {
                let distance = bottom_k_jaccard_distance(&self.entries[left].arc_bottom_k, &self.entries[right].arc_bottom_k);
                distances.push(format!("{left}-{right}={distance}"));
            }
        }
        if distances.is_empty() {
            "none".to_string()
        } else {
            distances.join(",")
        }
    }

    pub(crate) fn distance_stats_ppm(&self) -> Option<(u32, u32, u32)> {
        let mut minimum = u32::MAX;
        let mut maximum = 0u32;
        let mut total = 0u64;
        let mut count = 0u64;
        for left in 0..self.entries.len() {
            for right in left + 1..self.entries.len() {
                let distance = bottom_k_jaccard_distance(&self.entries[left].arc_bottom_k, &self.entries[right].arc_bottom_k);
                minimum = minimum.min(distance);
                maximum = maximum.max(distance);
                total = total.saturating_add(u64::from(distance));
                count = count.saturating_add(1);
            }
        }
        (count > 0).then(|| (minimum, u32::try_from(total / count).unwrap_or(u32::MAX), maximum))
    }

    /// Consider one completely reconstructed state.
    ///
    /// This streaming API is deterministic for a fixed arrival sequence. Use
    /// [`Self::consider_batch`] when candidates arrive concurrently and their
    /// arrival order must not affect the retained frontier.
    ///
    /// All interruptible identity, encoding, comparison, and frontier work
    /// completes before the archive is mutated.
    pub(crate) fn consider(&mut self, state: &JobShopState, stop: &AtomicBool) -> Result<ScheduleEliteConsiderOutcome, ScheduleEliteError> {
        let identity = match &self.identity {
            Some(identity) => {
                if !identity.matches_state(state, stop)? {
                    return Ok(ScheduleEliteConsiderOutcome::Incompatible);
                }
                Arc::clone(identity)
            }
            None => Arc::new(ProblemIdentity::capture(state, stop)?),
        };
        let candidate = encode_state(state, identity.clone(), stop)?;

        for entry in &self.entries {
            checkpoint(stop)?;
            if entry.order_hash == candidate.order_hash && exact_same_order(entry, &candidate, stop)? {
                if entry.objective != candidate.objective {
                    return Err(ScheduleEliteError::InvalidMachineOrder);
                }
                return Ok(ScheduleEliteConsiderOutcome::Duplicate);
            }
        }

        let candidates = [candidate];
        let mut sources = (0..self.entries.len()).map(EntrySource::Existing).collect::<Vec<_>>();
        sources.push(EntrySource::Candidate(0));
        let selected = farthest_first_sources(&self.entries, &candidates, &sources, stop)?;
        let candidate_survives = selected.iter().any(|&source_index| sources[source_index] == EntrySource::Candidate(0));
        if !candidate_survives {
            return Ok(ScheduleEliteConsiderOutcome::Dominated);
        }

        let outcome = if self.entries.len() == SCHEDULE_ELITE_CAPACITY {
            ScheduleEliteConsiderOutcome::InsertedAndEvicted
        } else {
            ScheduleEliteConsiderOutcome::Inserted
        };
        self.commit_sources(candidates.into(), &sources, selected, identity);
        Ok(outcome)
    }

    /// Canonically merge a set of concurrently produced states.
    ///
    /// Candidates are encoded and sorted by the complete deterministic entry
    /// order before a single capacity-four frontier is selected. The operation
    /// is transactional and therefore independent of the input slice order.
    pub(crate) fn consider_batch(
        &mut self,
        states: &[&JobShopState],
        stop: &AtomicBool,
    ) -> Result<ScheduleEliteBatchOutcome, ScheduleEliteError> {
        checkpoint(stop)?;
        let outcome = ScheduleEliteBatchOutcome { considered: states.len(), ..ScheduleEliteBatchOutcome::default() };
        let Some(first) = states.first() else {
            return Ok(outcome);
        };

        let identity = match &self.identity {
            Some(identity) => Arc::clone(identity),
            None => Arc::new(ProblemIdentity::capture(first, stop)?),
        };
        let mut encoded = Vec::with_capacity(states.len());
        for state in states {
            if !identity.matches_state(state, stop)? {
                return Err(ScheduleEliteError::IncompatibleProblem);
            }
            encoded.push(encode_state(state, identity.clone(), stop)?);
        }
        self.consider_encoded_batch(encoded, identity, states.len(), stop)
    }

    /// Merge worker-owned compact snapshots at one deterministic portfolio
    /// boundary. The reference state supplies the exact static identity once;
    /// candidates themselves contain only machine orders and objectives.
    pub(crate) fn consider_candidate_batch(
        &mut self,
        reference: &JobShopState,
        solve_token: &ScheduleEliteSolveToken,
        candidates: Vec<ScheduleEliteCandidate>,
        stop: &AtomicBool,
    ) -> Result<ScheduleEliteBatchOutcome, ScheduleEliteError> {
        checkpoint(stop)?;
        let considered = candidates.len();
        if candidates.is_empty() {
            return Ok(ScheduleEliteBatchOutcome::default());
        }
        if self.solve_token.as_ref().is_some_and(|archive_token| !archive_token.same_solve(solve_token)) {
            return Err(ScheduleEliteError::IncompatibleSolve);
        }
        for candidate in &candidates {
            candidate.validate_for_reference(reference, solve_token, stop)?;
        }
        let identity = match &self.identity {
            Some(identity) => {
                if !identity.matches_state(reference, stop)? {
                    return Err(ScheduleEliteError::IncompatibleProblem);
                }
                Arc::clone(identity)
            }
            None => Arc::new(ProblemIdentity::capture(reference, stop)?),
        };
        let encoded = candidates.into_iter().map(|candidate| candidate.into_entry(identity.clone())).collect();
        let outcome = self.consider_encoded_batch(encoded, identity, considered, stop)?;
        self.solve_token.get_or_insert_with(|| solve_token.clone());
        Ok(outcome)
    }

    fn consider_encoded_batch(
        &mut self,
        mut encoded: Vec<ScheduleEliteEntry>,
        identity: Arc<ProblemIdentity>,
        considered: usize,
        stop: &AtomicBool,
    ) -> Result<ScheduleEliteBatchOutcome, ScheduleEliteError> {
        let mut outcome = ScheduleEliteBatchOutcome { considered, ..ScheduleEliteBatchOutcome::default() };
        canonical_sort(&mut encoded, stop)?;

        let mut candidates = Vec::with_capacity(encoded.len());
        for candidate in encoded {
            let mut duplicate_objective = None;
            for entry in self.entries.iter().chain(&candidates) {
                checkpoint(stop)?;
                if entry.order_hash == candidate.order_hash && exact_same_order(entry, &candidate, stop)? {
                    duplicate_objective = Some(entry.objective);
                    break;
                }
            }
            if let Some(objective) = duplicate_objective {
                if objective != candidate.objective {
                    return Err(ScheduleEliteError::InvalidMachineOrder);
                }
                outcome.duplicates = outcome.duplicates.saturating_add(1);
            } else {
                candidates.push(candidate);
            }
        }
        if candidates.is_empty() {
            return Ok(outcome);
        }

        let mut sources = (0..self.entries.len()).map(EntrySource::Existing).collect::<Vec<_>>();
        sources.extend((0..candidates.len()).map(EntrySource::Candidate));
        let selected = farthest_first_sources(&self.entries, &candidates, &sources, stop)?;
        let retained_existing = selected.iter().filter(|&&source_index| matches!(sources[source_index], EntrySource::Existing(_))).count();
        outcome.retained = selected.len().saturating_sub(retained_existing);
        outcome.dominated = candidates.len().saturating_sub(outcome.retained);
        outcome.evicted = self.entries.len().saturating_sub(retained_existing);
        if outcome.retained == 0 {
            return Ok(outcome);
        }

        self.commit_sources(candidates, &sources, selected, identity);
        Ok(outcome)
    }

    fn commit_sources(
        &mut self,
        candidates: Vec<ScheduleEliteEntry>,
        sources: &[EntrySource],
        selected: Vec<usize>,
        identity: Arc<ProblemIdentity>,
    ) {
        let old_entries = std::mem::take(&mut self.entries);
        let mut old_slots = old_entries.into_iter().map(Some).collect::<Vec<_>>();
        let mut candidate_slots = candidates.into_iter().map(Some).collect::<Vec<_>>();
        let mut committed = Vec::with_capacity(SCHEDULE_ELITE_CAPACITY);
        for source_index in selected {
            match sources[source_index] {
                EntrySource::Existing(index) => committed.push(old_slots[index].take().expect("selected elite exists")),
                EntrySource::Candidate(index) => committed.push(candidate_slots[index].take().expect("selected candidate exists")),
            }
        }
        self.identity = Some(identity);
        self.entries = committed;
    }

    #[cfg(test)]
    pub(crate) fn test_corrupt_best_objective(&mut self) {
        if let Some(best) = self.entries.first_mut() {
            best.objective = best.objective.saturating_add(1);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntrySource {
    Existing(usize),
    Candidate(usize),
}

fn encode_state(state: &JobShopState, identity: Arc<ProblemIdentity>, stop: &AtomicBool) -> Result<ScheduleEliteEntry, ScheduleEliteError> {
    Ok(ScheduleEliteCandidate::capture_unscoped(state, stop)?.into_entry(identity))
}

fn machine_arc_hash(machine: usize, before: usize, after: usize) -> u64 {
    mix64(
        ARC_HASH_SALT
            ^ u64::try_from(machine).unwrap_or(u64::MAX)
            ^ u64::try_from(before).unwrap_or(u64::MAX).rotate_left(21)
            ^ u64::try_from(after).unwrap_or(u64::MAX).rotate_left(43),
    )
}

fn bottom_k_insert(bottom_k: &mut BTreeSet<u64>, hash: u64) {
    if bottom_k.len() < ARC_BOTTOM_K {
        bottom_k.insert(hash);
    } else if bottom_k.last().is_some_and(|&largest| hash < largest) && bottom_k.insert(hash) {
        bottom_k.pop_last();
    }
}

fn bottom_k_jaccard_distance(left: &[u64], right: &[u64]) -> u32 {
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    let mut sample = 0u32;
    let mut shared = 0u32;
    let sample_limit = if left.len() < ARC_BOTTOM_K && right.len() < ARC_BOTTOM_K { u32::MAX } else { ARC_BOTTOM_K as u32 };
    while sample < sample_limit && (left_index < left.len() || right_index < right.len()) {
        match (left.get(left_index), right.get(right_index)) {
            (Some(&left_hash), Some(&right_hash)) if left_hash == right_hash => {
                shared = shared.saturating_add(1);
                left_index += 1;
                right_index += 1;
            }
            (Some(&left_hash), Some(&right_hash)) if left_hash < right_hash => left_index += 1,
            (Some(_), Some(_)) => right_index += 1,
            (Some(_), None) => left_index += 1,
            (None, Some(_)) => right_index += 1,
            (None, None) => break,
        }
        sample = sample.saturating_add(1);
    }
    if sample == 0 {
        0
    } else {
        sample.saturating_sub(shared).saturating_mul(DISTANCE_SCALE) / sample
    }
}

fn exact_same_order(left: &ScheduleEliteEntry, right: &ScheduleEliteEntry, stop: &AtomicBool) -> Result<bool, ScheduleEliteError> {
    if !equal_u32_slices(&left.machine_offsets, &right.machine_offsets, stop)? {
        return Ok(false);
    }
    equal_u32_slices(&left.operations, &right.operations, stop)
}

fn equal_u32_slices(left: &[u32], right: &[u32], stop: &AtomicBool) -> Result<bool, ScheduleEliteError> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        poll(stop, index)?;
        if left != right {
            return Ok(false);
        }
    }
    Ok(true)
}

fn canonical_sort(entries: &mut [ScheduleEliteEntry], stop: &AtomicBool) -> Result<(), ScheduleEliteError> {
    for index in 1..entries.len() {
        checkpoint(stop)?;
        let mut position = index;
        while position > 0 && compare_entries(&entries[position], &entries[position - 1], stop)? == Ordering::Less {
            entries.swap(position, position - 1);
            position -= 1;
        }
    }
    Ok(())
}

fn farthest_first_sources(
    existing: &[ScheduleEliteEntry],
    candidates: &[ScheduleEliteEntry],
    sources: &[EntrySource],
    stop: &AtomicBool,
) -> Result<Vec<usize>, ScheduleEliteError> {
    checkpoint(stop)?;
    let target = sources.len().min(SCHEDULE_ELITE_CAPACITY);
    let mut selected = Vec::with_capacity(target);
    if target == 0 {
        return Ok(selected);
    }

    let mut best = 0usize;
    for index in 1..sources.len() {
        checkpoint(stop)?;
        if compare_entries(source_entry(existing, candidates, sources[index]), source_entry(existing, candidates, sources[best]), stop)?
            == Ordering::Less
        {
            best = index;
        }
    }
    selected.push(best);

    while selected.len() < target {
        checkpoint(stop)?;
        let mut farthest: Option<(usize, u32)> = None;
        for index in 0..sources.len() {
            if selected.contains(&index) {
                continue;
            }
            let entry = source_entry(existing, candidates, sources[index]);
            let mut minimum_distance = u32::MAX;
            for &selected_index in &selected {
                checkpoint(stop)?;
                let selected_entry = source_entry(existing, candidates, sources[selected_index]);
                minimum_distance = minimum_distance.min(bottom_k_jaccard_distance(&entry.arc_bottom_k, &selected_entry.arc_bottom_k));
            }
            let replace = match farthest {
                None => true,
                Some((_, incumbent_distance)) if minimum_distance > incumbent_distance => true,
                Some((incumbent_index, incumbent_distance)) if minimum_distance == incumbent_distance => {
                    compare_entries(entry, source_entry(existing, candidates, sources[incumbent_index]), stop)? == Ordering::Less
                }
                Some(_) => false,
            };
            if replace {
                farthest = Some((index, minimum_distance));
            }
        }
        selected.push(farthest.expect("an unselected elite remains").0);
    }
    Ok(selected)
}

fn source_entry<'a>(
    existing: &'a [ScheduleEliteEntry],
    candidates: &'a [ScheduleEliteEntry],
    source: EntrySource,
) -> &'a ScheduleEliteEntry {
    match source {
        EntrySource::Existing(index) => &existing[index],
        EntrySource::Candidate(index) => &candidates[index],
    }
}

fn compare_entries(left: &ScheduleEliteEntry, right: &ScheduleEliteEntry, stop: &AtomicBool) -> Result<Ordering, ScheduleEliteError> {
    let prefix = left.objective.cmp(&right.objective).then_with(|| left.order_hash.cmp(&right.order_hash));
    if prefix != Ordering::Equal {
        return Ok(prefix);
    }
    let offsets = compare_u32_slices(&left.machine_offsets, &right.machine_offsets, stop)?;
    if offsets != Ordering::Equal {
        return Ok(offsets);
    }
    compare_u32_slices(&left.operations, &right.operations, stop)
}

fn compare_u32_slices(left: &[u32], right: &[u32], stop: &AtomicBool) -> Result<Ordering, ScheduleEliteError> {
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        poll(stop, index)?;
        let ordering = left.cmp(&right);
        if ordering != Ordering::Equal {
            return Ok(ordering);
        }
    }
    Ok(left.len().cmp(&right.len()))
}

#[cfg(test)]
pub(crate) fn test_arc_bottom_k<I>(arcs: I) -> Vec<u64>
where
    I: IntoIterator<Item = (usize, usize, usize)>,
{
    let mut bottom_k = BTreeSet::new();
    for (machine, before, after) in arcs {
        bottom_k_insert(&mut bottom_k, machine_arc_hash(machine, before, after));
    }
    bottom_k.into_iter().collect()
}

#[cfg(test)]
pub(crate) fn test_bottom_k_jaccard_distance(left: &[u64], right: &[u64]) -> u32 {
    bottom_k_jaccard_distance(left, right)
}

#[cfg(test)]
thread_local! {
    static TEST_CHECKPOINTS_UNTIL_INTERRUPT: Cell<Option<usize>> = const { Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn test_interrupt_after_checkpoints(checkpoints: usize) {
    TEST_CHECKPOINTS_UNTIL_INTERRUPT.with(|remaining| remaining.set(Some(checkpoints)));
}

#[cfg(test)]
pub(crate) fn test_clear_interrupt() {
    TEST_CHECKPOINTS_UNTIL_INTERRUPT.with(|remaining| remaining.set(None));
}

fn checkpoint(stop: &AtomicBool) -> Result<(), ScheduleEliteError> {
    #[cfg(test)]
    let injected = TEST_CHECKPOINTS_UNTIL_INTERRUPT.with(|remaining| match remaining.get() {
        Some(0) => {
            remaining.set(None);
            true
        }
        Some(value) => {
            remaining.set(Some(value - 1));
            false
        }
        None => false,
    });
    #[cfg(not(test))]
    let injected = false;

    if injected || stop.load(AtomicOrdering::Acquire) {
        Err(ScheduleEliteError::Interrupted)
    } else {
        Ok(())
    }
}

fn poll(stop: &AtomicBool, index: usize) -> Result<(), ScheduleEliteError> {
    if index & STOP_POLL_MASK == 0 {
        checkpoint(stop)?;
    }
    Ok(())
}
