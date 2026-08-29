//! Bounded guide-directed path relinking for strict job-shop schedules.
//!
//! The archive remains search memory only. This module derives at most two
//! N8-style insertions from one borrowed guide, and the scheduling kernel sends
//! every selected insertion through the complete reconstruction oracle.

use std::cmp::Reverse;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::schedule_elite::{ScheduleEliteEntry, ScheduleEliteError};
use super::schedule_state::{HeadTailMoveScore, JobShopState, LocalMoveEstimate, MachineArc, ScheduleMove};

const STREAMING_CAPACITY: usize = 16;
pub(crate) const RELINK_ORACLE_CAPACITY: usize = 1;
const STOP_POLL_MASK: usize = 4_095;
const NO_POSITION: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScheduleRelinkGuideKind {
    Best,
    Diverse,
}

#[derive(Clone, Copy)]
pub(crate) struct ScheduleRelinkRequest<'a> {
    pub(crate) guide: &'a ScheduleEliteEntry,
    pub(crate) kind: ScheduleRelinkGuideKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScheduleRelinkCandidate {
    pub(crate) movement: ScheduleMove,
    pub(crate) guide_arc_gain: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScheduleRelinkMetrics {
    pub(crate) requests: u64,
    pub(crate) best_guides: u64,
    pub(crate) diverse_guides: u64,
    pub(crate) guide_loads: u64,
    pub(crate) guide_incompatible: u64,
    pub(crate) guide_interruptions: u64,
    pub(crate) critical_operations_scanned: u64,
    pub(crate) candidates_generated: u64,
    pub(crate) candidates_positive_gain: u64,
    pub(crate) acyclicity_certified: u64,
    pub(crate) acyclicity_unknown: u64,
    pub(crate) prefilter_rejections: u64,
    pub(crate) candidates_retained: u64,
    pub(crate) candidates_refined: u64,
    pub(crate) candidates_shortlisted: u64,
    pub(crate) no_move: u64,
    pub(crate) guide_arc_gain_shortlisted: u64,
    pub(crate) oracle_attempts: u64,
    pub(crate) oracle_accepts: u64,
    pub(crate) cycle_rejections: u64,
    pub(crate) window_rejections: u64,
    pub(crate) other_rejections: u64,
    pub(crate) rollbacks: u64,
    pub(crate) elite_improvements: u64,
    pub(crate) guide_arc_gain_accepted: u64,
    pub(crate) workspace_peak_bytes: usize,
    pub(crate) elapsed: Duration,
}

impl ScheduleRelinkMetrics {
    pub(crate) fn add(&mut self, other: Self) {
        self.requests = self.requests.saturating_add(other.requests);
        self.best_guides = self.best_guides.saturating_add(other.best_guides);
        self.diverse_guides = self.diverse_guides.saturating_add(other.diverse_guides);
        self.guide_loads = self.guide_loads.saturating_add(other.guide_loads);
        self.guide_incompatible = self.guide_incompatible.saturating_add(other.guide_incompatible);
        self.guide_interruptions = self.guide_interruptions.saturating_add(other.guide_interruptions);
        self.critical_operations_scanned = self.critical_operations_scanned.saturating_add(other.critical_operations_scanned);
        self.candidates_generated = self.candidates_generated.saturating_add(other.candidates_generated);
        self.candidates_positive_gain = self.candidates_positive_gain.saturating_add(other.candidates_positive_gain);
        self.acyclicity_certified = self.acyclicity_certified.saturating_add(other.acyclicity_certified);
        self.acyclicity_unknown = self.acyclicity_unknown.saturating_add(other.acyclicity_unknown);
        self.prefilter_rejections = self.prefilter_rejections.saturating_add(other.prefilter_rejections);
        self.candidates_retained = self.candidates_retained.saturating_add(other.candidates_retained);
        self.candidates_refined = self.candidates_refined.saturating_add(other.candidates_refined);
        self.candidates_shortlisted = self.candidates_shortlisted.saturating_add(other.candidates_shortlisted);
        self.no_move = self.no_move.saturating_add(other.no_move);
        self.guide_arc_gain_shortlisted = self.guide_arc_gain_shortlisted.saturating_add(other.guide_arc_gain_shortlisted);
        self.oracle_attempts = self.oracle_attempts.saturating_add(other.oracle_attempts);
        self.oracle_accepts = self.oracle_accepts.saturating_add(other.oracle_accepts);
        self.cycle_rejections = self.cycle_rejections.saturating_add(other.cycle_rejections);
        self.window_rejections = self.window_rejections.saturating_add(other.window_rejections);
        self.other_rejections = self.other_rejections.saturating_add(other.other_rejections);
        self.rollbacks = self.rollbacks.saturating_add(other.rollbacks);
        self.elite_improvements = self.elite_improvements.saturating_add(other.elite_improvements);
        self.guide_arc_gain_accepted = self.guide_arc_gain_accepted.saturating_add(other.guide_arc_gain_accepted);
        self.workspace_peak_bytes = self.workspace_peak_bytes.max(other.workspace_peak_bytes);
        self.elapsed = self.elapsed.saturating_add(other.elapsed);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StreamingCandidate {
    movement: ScheduleMove,
    guide_arc_gain: u8,
    displacement: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RefinedCandidate {
    candidate: StreamingCandidate,
    score: HeadTailMoveScore,
    estimate: LocalMoveEstimate,
}

#[derive(Default)]
pub(crate) struct ScheduleRelinkWorkspace {
    guide_order_hash: Option<u64>,
    guide_objective: Option<i64>,
    guide_positions: Vec<u32>,
    streaming: Vec<StreamingCandidate>,
    refined: Vec<RefinedCandidate>,
    shortlist: Vec<ScheduleRelinkCandidate>,
}

impl ScheduleRelinkWorkspace {
    pub(crate) fn shortlist(&self) -> &[ScheduleRelinkCandidate] {
        &self.shortlist
    }

    pub(crate) fn heap_lower_bound_bytes(&self) -> usize {
        self.guide_positions
            .capacity()
            .saturating_mul(size_of::<u32>())
            .saturating_add(self.streaming.capacity().saturating_mul(size_of::<StreamingCandidate>()))
            .saturating_add(self.refined.capacity().saturating_mul(size_of::<RefinedCandidate>()))
            .saturating_add(self.shortlist.capacity().saturating_mul(size_of::<ScheduleRelinkCandidate>()))
    }

    pub(crate) fn prepare(
        &mut self,
        state: &mut JobShopState,
        request: ScheduleRelinkRequest<'_>,
        metrics: &mut ScheduleRelinkMetrics,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleEliteError> {
        let started = Instant::now();
        *metrics = ScheduleRelinkMetrics { requests: 1, ..ScheduleRelinkMetrics::default() };
        match request.kind {
            ScheduleRelinkGuideKind::Best => metrics.best_guides = 1,
            ScheduleRelinkGuideKind::Diverse => metrics.diverse_guides = 1,
        }
        self.shortlist.clear();
        self.streaming.clear();
        self.refined.clear();

        let result = self.prepare_inner(state, request, metrics, stop);
        if let Err(error) = result {
            match error {
                ScheduleEliteError::Interrupted => metrics.guide_interruptions = 1,
                _ => metrics.guide_incompatible = 1,
            }
        }
        metrics.workspace_peak_bytes = self.heap_lower_bound_bytes();
        metrics.elapsed = started.elapsed();
        result
    }

    fn prepare_inner(
        &mut self,
        state: &mut JobShopState,
        request: ScheduleRelinkRequest<'_>,
        metrics: &mut ScheduleRelinkMetrics,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleEliteError> {
        if self.guide_order_hash != Some(request.guide.order_hash())
            || self.guide_objective != Some(request.guide.objective())
            || self.guide_positions.len() != state.operation_count()
        {
            metrics.guide_loads = 1;
            self.load_guide(state, request.guide, stop)?;
        }

        let blocks = state.critical_blocks().len();
        for block_index in 0..blocks {
            checkpoint(stop)?;
            let block = state.critical_blocks()[block_index];
            let machine = block.machine();
            for from in block.first_position()..=block.last_position() {
                poll(stop, from)?;
                metrics.critical_operations_scanned = metrics.critical_operations_scanned.saturating_add(1);
                let operation = state.machine_sequences()[machine][from];
                let guide_position = usize::try_from(self.guide_positions[operation]).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
                let guide_sequence = request.guide.machine_sequence(machine).ok_or(ScheduleEliteError::InvalidMachineOrder)?;
                if guide_sequence.get(guide_position).and_then(|&value| usize::try_from(value).ok()) != Some(operation) {
                    return Err(ScheduleEliteError::InvalidMachineOrder);
                }

                if let Some(&encoded_predecessor) = guide_position.checked_sub(1).and_then(|position| guide_sequence.get(position)) {
                    let predecessor = usize::try_from(encoded_predecessor).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
                    let predecessor_position = state.position(predecessor).ok_or(ScheduleEliteError::InvalidMachineOrder)?;
                    if !(block.first_position()..=block.last_position()).contains(&predecessor_position) {
                        let to = if predecessor_position < from { predecessor_position.saturating_add(1) } else { predecessor_position };
                        self.consider_candidate(state, machine, from, to, metrics, stop)?;
                    }
                }
                if let Some(&encoded_successor) = guide_sequence.get(guide_position.saturating_add(1)) {
                    let successor = usize::try_from(encoded_successor).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
                    let successor_position = state.position(successor).ok_or(ScheduleEliteError::InvalidMachineOrder)?;
                    if !(block.first_position()..=block.last_position()).contains(&successor_position) {
                        let to = if successor_position < from { successor_position } else { successor_position.saturating_sub(1) };
                        self.consider_candidate(state, machine, from, to, metrics, stop)?;
                    }
                }
            }
        }
        metrics.candidates_retained = u64::try_from(self.streaming.len()).unwrap_or(u64::MAX);

        for candidate in self.streaming.iter().copied() {
            checkpoint(stop)?;
            let Some(score) = state.score_move_head_tail(candidate.movement, stop).map_err(|_| ScheduleEliteError::Interrupted)? else {
                continue;
            };
            let Some(estimate) = state.estimate_move_local(candidate.movement, stop).map_err(|_| ScheduleEliteError::Interrupted)? else {
                continue;
            };
            self.refined.push(RefinedCandidate { candidate, score, estimate });
        }
        metrics.candidates_refined = u64::try_from(self.refined.len()).unwrap_or(u64::MAX);
        self.refined.sort_unstable_by_key(|candidate| {
            (
                Reverse(candidate.candidate.guide_arc_gain),
                !candidate.estimate.acyclicity_certified,
                candidate.estimate.estimated_makespan,
                candidate.score.max_added_arc_path,
                candidate.score.total_added_arc_path,
                candidate.score.tight_arcs_added,
                Reverse(candidate.score.critical_arcs_removed),
                candidate.candidate.displacement,
                candidate.candidate.movement,
            )
        });
        self.shortlist.extend(self.refined.iter().take(RELINK_ORACLE_CAPACITY).map(|candidate| ScheduleRelinkCandidate {
            movement: candidate.candidate.movement,
            guide_arc_gain: candidate.candidate.guide_arc_gain,
        }));
        metrics.candidates_shortlisted = u64::try_from(self.shortlist.len()).unwrap_or(u64::MAX);
        metrics.guide_arc_gain_shortlisted = self.shortlist.iter().map(|candidate| u64::from(candidate.guide_arc_gain)).sum();
        metrics.no_move = u64::from(self.shortlist.is_empty());
        Ok(())
    }

    fn load_guide(&mut self, state: &JobShopState, guide: &ScheduleEliteEntry, stop: &AtomicBool) -> Result<(), ScheduleEliteError> {
        // A failed replacement must never leave the previous cache key paired
        // with a partially overwritten position vector. Keeping the allocation
        // is safe, but the key is committed only after the full guide validates.
        self.guide_order_hash = None;
        self.guide_objective = None;
        checkpoint(stop)?;
        if !guide.matches_state(state, stop)?
            || guide.operation_count() != state.operation_count()
            || guide.machine_count() != state.machine_count()
        {
            return Err(ScheduleEliteError::IncompatibleProblem);
        }
        self.guide_positions.clear();
        self.guide_positions.resize(state.operation_count(), NO_POSITION);
        let mut visited = 0usize;
        for machine in 0..state.machine_count() {
            checkpoint(stop)?;
            let sequence = guide.machine_sequence(machine).ok_or(ScheduleEliteError::InvalidMachineOrder)?;
            if sequence.len() != state.machine_sequences()[machine].len() {
                return Err(ScheduleEliteError::InvalidMachineOrder);
            }
            for (position, &encoded) in sequence.iter().enumerate() {
                poll(stop, visited)?;
                let operation = usize::try_from(encoded).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
                if operation >= state.operation_count()
                    || state.machine(operation) != machine
                    || self.guide_positions[operation] != NO_POSITION
                {
                    return Err(ScheduleEliteError::InvalidMachineOrder);
                }
                self.guide_positions[operation] = u32::try_from(position).map_err(|_| ScheduleEliteError::EncodingOverflow)?;
                visited = visited.saturating_add(1);
            }
        }
        if visited != state.operation_count() || self.guide_positions.contains(&NO_POSITION) {
            return Err(ScheduleEliteError::InvalidMachineOrder);
        }
        self.guide_order_hash = Some(guide.order_hash());
        self.guide_objective = Some(guide.objective());
        Ok(())
    }

    fn consider_candidate(
        &mut self,
        state: &JobShopState,
        machine: usize,
        from: usize,
        to: usize,
        metrics: &mut ScheduleRelinkMetrics,
        stop: &AtomicBool,
    ) -> Result<(), ScheduleEliteError> {
        if from == to {
            return Ok(());
        }
        metrics.candidates_generated = metrics.candidates_generated.saturating_add(1);
        let movement = ScheduleMove::Insert { machine, from, to };
        let Some(arcs) = state.move_arcs(movement) else {
            return Ok(());
        };
        let added = arcs.added.into_iter().flatten().filter(|&arc| self.is_guide_arc(arc)).count();
        let removed = arcs.removed.into_iter().flatten().filter(|&arc| self.is_guide_arc(arc)).count();
        let Some(gain) = added.checked_sub(removed) else {
            return Ok(());
        };
        if gain == 0 {
            return Ok(());
        }
        metrics.candidates_positive_gain = metrics.candidates_positive_gain.saturating_add(1);
        if !state.certifies_insert_acyclicity(movement, stop).map_err(|_| ScheduleEliteError::Interrupted)? {
            metrics.acyclicity_unknown = metrics.acyclicity_unknown.saturating_add(1);
            metrics.prefilter_rejections = metrics.prefilter_rejections.saturating_add(1);
            return Ok(());
        }
        metrics.acyclicity_certified = metrics.acyclicity_certified.saturating_add(1);
        let candidate =
            StreamingCandidate { movement, guide_arc_gain: u8::try_from(gain).unwrap_or(u8::MAX), displacement: from.abs_diff(to) };
        if let Some(existing) = self.streaming.iter_mut().find(|existing| existing.movement == movement) {
            if streaming_key(candidate) < streaming_key(*existing) {
                *existing = candidate;
            }
        } else {
            self.streaming.push(candidate);
        }
        self.streaming.sort_unstable_by_key(|candidate| streaming_key(*candidate));
        self.streaming.truncate(STREAMING_CAPACITY);
        Ok(())
    }

    fn is_guide_arc(&self, arc: MachineArc) -> bool {
        let Some(&before) = self.guide_positions.get(arc.before) else {
            return false;
        };
        let Some(&after) = self.guide_positions.get(arc.after) else {
            return false;
        };
        before != NO_POSITION && after == before.saturating_add(1)
    }
}

fn streaming_key(candidate: StreamingCandidate) -> (Reverse<u8>, usize, ScheduleMove) {
    (Reverse(candidate.guide_arc_gain), candidate.displacement, candidate.movement)
}

fn checkpoint(stop: &AtomicBool) -> Result<(), ScheduleEliteError> {
    if stop.load(Ordering::Acquire) {
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
