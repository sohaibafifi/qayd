use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use smallvec::SmallVec;

use crate::engines::ls::schedule_ir::PrecedenceDag;
use crate::mix64;
use crate::model::list::{CollectionSolution, IntervalVar, Resource, Schedule};

use super::move_acceptance::MinimizingMoveAcceptance;
use super::resource_schedule::{
    GenerationScheme, Justification, PriorityRule, PrioritySgs, ResourceAlnsBudget, ResourceMoveOutcome, ResourceScheduleMetrics,
    ResourceScheduleProblem, ResourceScheduleState,
};
use super::schedule_elite::{ScheduleEliteArchive, ScheduleEliteCandidate, ScheduleEliteError, ScheduleEliteSolveToken};
#[cfg(test)]
use super::schedule_lns::ScheduleLnsWorkspaceCapacities;
use super::schedule_lns::{repair_critical_window_with_workspace, ScheduleLnsConfig, ScheduleLnsMetrics, ScheduleLnsWorkspace};
use super::schedule_relink::{
    ScheduleRelinkCandidate, ScheduleRelinkMetrics, ScheduleRelinkRequest, ScheduleRelinkWorkspace, RELINK_MACRO_CAPACITY,
    RELINK_ORACLE_CAPACITY,
};
use super::schedule_state::{
    CanonicalPathPolicy, CriticalNeighborhood, DispatchRule, GifflerThompsonWorkspace, HeadTailMoveScore, JobShopProblem, JobShopState,
    LocalMoveEstimate, MachineArc, MoveOutcome, MoveProbe, MoveRejection, ScheduleMove, ScheduleMoveArcs, ScheduleStateMetrics,
    ScoredScheduleMove,
};

/// Explicit JSSP local-search kernel selection. All historical entry points
/// pass [`ScheduleJsspSearchStrategy::Legacy`] so the new strategy is strictly
/// default-off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ScheduleJsspSearchStrategy {
    #[default]
    Legacy,
    TsabCandidate,
    TsabMultiCandidate,
}

impl ScheduleJsspSearchStrategy {
    pub(crate) fn requests_tsab(self) -> bool {
        matches!(self, Self::TsabCandidate | Self::TsabMultiCandidate)
    }

    pub(crate) fn owner_mask(self, workers: usize) -> u64 {
        if workers != 7 {
            return 0;
        }
        match self {
            Self::Legacy => 0,
            Self::TsabCandidate => 1 << 6,
            Self::TsabMultiCandidate => 0x7f,
        }
    }

    pub(crate) fn owns_worker(self, worker: usize, workers: usize) -> bool {
        let worker_bit = 1u64.checked_shl(u32::try_from(worker).unwrap_or(u32::MAX)).unwrap_or(0);
        self.owner_mask(workers) & worker_bit != 0
    }
}

fn tsab_canonical_path_policy(strategy: ScheduleJsspSearchStrategy, worker: usize) -> CanonicalPathPolicy {
    const DIVERSIFIED_CANONICAL_PATHS: bool = false;
    if strategy != ScheduleJsspSearchStrategy::TsabMultiCandidate || !DIVERSIFIED_CANONICAL_PATHS {
        return CanonicalPathPolicy::Historical;
    }
    match worker {
        0 => CanonicalPathPolicy::MachineFirstLowest,
        1 => CanonicalPathPolicy::JobFirstHighest,
        2 => CanonicalPathPolicy::JobFirstLowest,
        3 => CanonicalPathPolicy::OperationHighest,
        4 => CanonicalPathPolicy::OperationLowest,
        5 => CanonicalPathPolicy::Mixed,
        _ => CanonicalPathPolicy::Historical,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScheduleTsabMetrics {
    pub(crate) owner_worker_mask: u64,
    pub(crate) activations: u64,
    pub(crate) activation_boundary: Option<u64>,
    pub(crate) legacy_warmup_work_steps: u64,
    pub(crate) activation_rebases: u64,
    pub(crate) activation_objective: Option<i64>,
    pub(crate) active_boundaries: u64,
    pub(crate) burst_work_limit: u64,
    pub(crate) burst_work_units: u64,
    pub(crate) n5_generated: u64,
    pub(crate) ranked: u64,
    pub(crate) shortlists: u64,
    pub(crate) delta_probes: u64,
    /// Delta probes after the first probe of the same persistent shortlist.
    pub(crate) additional_delta_probes: u64,
    /// Selected moves accepted by `commit_probed_move` after its independent
    /// complete reconstruction oracle. Rejected or interrupted commit calls
    /// are not included.
    pub(crate) full_oracle_commits: u64,
    pub(crate) selected_shortlist_rank_sum: u64,
    /// Admissible candidates selected for a commit attempt. Therefore
    /// `full_oracle_commits <= selections`.
    pub(crate) selections: u64,
    pub(crate) aspirations: u64,
    pub(crate) tabu_rejections: u64,
    pub(crate) tabu_resets: u64,
    pub(crate) fingerprint_repeats: u64,
    pub(crate) escape_signals: u64,
    pub(crate) n1_kicks: u64,
    pub(crate) kick_moves: u64,
    pub(crate) elite_restarts: u64,
    pub(crate) n6_kicks: u64,
    pub(crate) restart_attempts: u64,
    pub(crate) restart_global_rebases: u64,
    pub(crate) restart_n6_generated: u64,
    pub(crate) restart_delta_probes: u64,
    pub(crate) restart_oracle_commits: u64,
    pub(crate) restart_relink_attempts: u64,
    pub(crate) restart_relink_commits: u64,
    pub(crate) restart_relink_components_shortlisted: u64,
    pub(crate) restart_relink_components_committed: u64,
    pub(crate) restart_relink_guide_arc_gain_shortlisted: u64,
    pub(crate) restart_relink_guide_arc_gain_accepted: u64,
    pub(crate) restart_rejections: u64,
    pub(crate) restart_interruptions: u64,
    pub(crate) restart_work_units: u64,
    pub(crate) restart_best_base_objective: Option<i64>,
    pub(crate) restart_best_kicked_objective: Option<i64>,
    pub(crate) post_restart_improvements: u64,
    /// Fixed-size N6 shortlist storage. The transient scratch state is not
    /// included because `JobShopState` owns its memory accounting.
    pub(crate) restart_shortlist_peak_bytes: usize,
    pub(crate) ranking_audits: u64,
    pub(crate) exact_best_matches: u64,
    pub(crate) regret_sum: u64,
    pub(crate) regret_max: u64,
    pub(crate) workspace_peak_bytes: usize,
    pub(crate) improving_commits: u64,
    pub(crate) best_committed_objective: Option<i64>,
    pub(crate) fast_enabled: u64,
    pub(crate) fast_eligible: u64,
    pub(crate) fast_disabled: u64,
    pub(crate) fast_attempts: u64,
    pub(crate) fast_commits: u64,
    pub(crate) fast_fallbacks: u64,
    pub(crate) fast_work_cap_recoveries: u64,
    pub(crate) fast_date_changes: u64,
    pub(crate) fast_queue_pops: u64,
    pub(crate) fast_full_validations: u64,
    pub(crate) fast_oracle_mismatches: u64,
    pub(crate) fast_pending_promotions: u64,
    pub(crate) fast_pending_discards: u64,
    pub(crate) fast_transitions: u64,
    pub(crate) fast_work_units: u64,
    pub(crate) fast_elapsed: Duration,
    pub(crate) fast_workspace_peak_bytes: usize,
}

/// Scheduling-search measurements shared with the collection profiler.
pub(crate) struct ScheduleConstructionMetrics {
    pub(crate) elapsed: Duration,
    pub(crate) first_feasible: Option<Duration>,
    /// Objective of the state that opened a new persistent search session.
    /// Resumed slices leave this empty so portfolio aggregation does not
    /// mistake the current elite for another construction result.
    pub(crate) initial_objective: Option<i64>,
    /// Initial constructive rule used by a new persistent search session.
    /// This is distinct from the generic constructor family and from any
    /// reactive-restart dispatch rule.
    pub(crate) initial_dispatch_rule: Option<&'static str>,
    pub(crate) candidates: u64,
    pub(crate) construction_bucket_visits: u64,
    pub(crate) construction_heap_pushes: u64,
    pub(crate) construction_stale_pops: u64,
    pub(crate) construction_heap_rebuilds: u64,
    pub(crate) construction_heap_peak: u64,
    pub(crate) work_steps: u64,
    pub(crate) constructor: &'static str,
    pub(crate) moves_considered: u64,
    pub(crate) moves_accepted: u64,
    pub(crate) incumbent_improvements: u64,
    pub(crate) incumbent_injections: u64,
    pub(crate) cycle_rejections: u64,
    pub(crate) window_rejections: u64,
    pub(crate) objective_rejections: u64,
    pub(crate) reconstructions: u64,
    pub(crate) critical_path_updates: u64,
    pub(crate) delta_evaluations: u64,
    pub(crate) full_evaluations: u64,
    pub(crate) full_fallbacks: u64,
    pub(crate) topological_rebuilds: u64,
    pub(crate) oracle_validations: u64,
    pub(crate) oracle_mismatches: u64,
    pub(crate) dirty_cone_operations: u64,
    pub(crate) max_dirty_cone: u64,
    pub(crate) workspace_growths: u64,
    pub(crate) workspace_rollbacks: u64,
    pub(crate) alns_generation_attempts: u64,
    pub(crate) alns_moves_generated: u64,
    pub(crate) resource_profile_checks: u64,
    pub(crate) resource_candidate_scheduling_attempts: u64,
    pub(crate) resource_event_visits: u64,
    pub(crate) resource_peak_profile_events: usize,
    pub(crate) precedence_rejections: u64,
    pub(crate) infeasible_rejections: u64,
    pub(crate) justification_attempts: u64,
    pub(crate) tabu_steps: u64,
    pub(crate) tabu_hits: u64,
    pub(crate) tabu_aspirations: u64,
    pub(crate) tabu_forced_moves: u64,
    pub(crate) session_initializations: u64,
    pub(crate) session_resumes: u64,
    pub(crate) session_rebases: u64,
    pub(crate) island_profile: Option<usize>,
    pub(crate) island_profile_mask: u64,
    pub(crate) baseline_island_profile_mask: u64,
    pub(crate) scored_island_profile_mask: u64,
    pub(crate) reactive_restarts: u64,
    pub(crate) reactive_restart_dispatches: u64,
    pub(crate) reactive_restart_perturbations: u64,
    pub(crate) reactive_restart_rebuild_failures: u64,
    pub(crate) island_scored_candidates: u64,
    pub(crate) island_shortlisted_candidates: u64,
    pub(crate) approximate_candidates_generated: u64,
    pub(crate) approximate_candidates_refined: u64,
    pub(crate) approximate_candidates_certified: u64,
    pub(crate) approximate_candidates_unknown: u64,
    pub(crate) approximation_score_items: u64,
    pub(crate) approximation_sort_items: u64,
    pub(crate) approximation_local_span_items: u64,
    pub(crate) approximation_elapsed: Duration,
    /// Deterministic structural work outside the search-work budget: generated
    /// candidates, local span propagation, and sorting estimates.
    pub(crate) approximation_work_units: u64,
    pub(crate) direct_oracle_attempts: u64,
    pub(crate) direct_oracle_accepts: u64,
    pub(crate) direct_oracle_cycles: u64,
    pub(crate) direct_oracle_windows: u64,
    pub(crate) direct_oracle_objective_rejections: u64,
    /// Locally refined candidates discarded without an exact delta probe.
    /// Candidates retained across a slice boundary are not counted yet.
    pub(crate) exact_probes_avoided: u64,
    pub(crate) search_elite_snapshot_captures: u64,
    pub(crate) search_elite_snapshot_interruptions: u64,
    pub(crate) search_elite_snapshot_errors: u64,
    pub(crate) search_elite_capture_worker_elapsed_sum: Duration,
    pub(crate) search_elite_snapshot_peak_heap_lower_bound_bytes: usize,
    /// Worker ownership and outcome counters for the measurement-only repair.
    pub(crate) schedule_lns_shadow_owner_worker_mask: u64,
    pub(crate) schedule_lns: ScheduleLnsMetrics,
    pub(crate) schedule_lns_workspace_peak_bytes: usize,
    pub(crate) schedule_path_relink: ScheduleRelinkMetrics,
    pub(crate) schedule_constructor_multistart_owner_worker_mask: u64,
    pub(crate) schedule_constructor_multistart_attempts: u64,
    pub(crate) schedule_constructor_multistart_constructions: u64,
    pub(crate) schedule_constructor_multistart_interruptions: u64,
    pub(crate) schedule_constructor_multistart_failures: u64,
    pub(crate) schedule_constructor_multistart_feasible: u64,
    pub(crate) schedule_constructor_multistart_distinct_fingerprints: u64,
    pub(crate) schedule_constructor_multistart_other_fingerprint_observations: u64,
    pub(crate) schedule_constructor_multistart_initial_objective: Option<i64>,
    pub(crate) schedule_constructor_multistart_best_objective: Option<i64>,
    pub(crate) schedule_constructor_multistart_improvements: u64,
    pub(crate) schedule_constructor_multistart_work_units: u64,
    pub(crate) schedule_constructor_multistart_elapsed: Duration,
    pub(crate) schedule_constructor_multistart_workspace_peak_bytes: usize,
    pub(crate) schedule_constructor_multistart_best_ordinal: Option<u64>,
    pub(crate) schedule_constructor_multistart_best_seed: Option<u64>,
    pub(crate) schedule_constructor_multistart_best_fingerprint: Option<u64>,
    pub(crate) schedule_constructor_multistart_next_ordinal: Option<u64>,
    pub(crate) schedule_tsab: ScheduleTsabMetrics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProbedScheduleCandidate {
    movement: ScheduleMove,
    arcs: ScheduleMoveArcs,
    objective: i64,
    aspiration: bool,
    rank: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EstimatedScheduleCandidate {
    movement: ScheduleMove,
    score: HeadTailMoveScore,
    estimate: LocalMoveEstimate,
    tabu: bool,
    tie: u64,
}

#[derive(Default)]
struct PersistentScheduleScan {
    movements: Vec<ScheduleMove>,
    scored_movements: Vec<ScoredScheduleMove>,
    estimated_movements: Vec<EstimatedScheduleCandidate>,
    cursor: usize,
    end: usize,
    best_admissible: Option<ProbedScheduleCandidate>,
    best_any: Option<ProbedScheduleCandidate>,
    relink_end: usize,
    relink_guide_arc_gain: [u8; RELINK_ORACLE_CAPACITY],
    tsab_ranks: Vec<usize>,
}

#[derive(Default)]
struct ScheduleLnsShadowSession {
    workspace: ScheduleLnsWorkspace,
    attempt_epoch: u64,
    last_attempt_probe: u64,
}

struct ScheduleConstructorMultistartSession {
    problem: JobShopProblem,
    workspace: GifflerThompsonWorkspace,
    work_credit: u64,
    initial_objective: Option<i64>,
    best_ordinal: Option<u64>,
    best_seed: Option<u64>,
    best_fingerprint: Option<u64>,
    first_fingerprint: Option<u64>,
    alternative_fingerprint_seen: bool,
    next_ordinal: u64,
    workspace_peak_bytes: usize,
}

const TSAB_EXACT_SHORTLIST_CAPACITY: usize = 4;
const TSAB_TENURE: u64 = 8;
const TSAB_LEGACY_WARMUP_WORK_UNITS: u64 = 512;
const TSAB_LEGACY_WARMUP_BOUNDARY_LIMIT: u64 = 256;
const TSAB_BURST_WORK_LIMIT: u64 = 128;
const TSAB_ELITE_RESTART_N5_WORK_UNITS: u64 = 16;
const TSAB_ELITE_RESTART_WORK_UNITS: u64 = 8;
const TSAB_KICK_MAX_WORSENING_PPM: i64 = 64;
const TSAB_FAST_TRANSITIONS_PER_WORK_UNIT: u64 = 4;
const TSAB_FAST_VALIDATION_TRANSITIONS: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TsabRankedCandidate {
    scored: ScoredScheduleMove,
    tabu: bool,
}

#[derive(Default)]
struct TsabShortlistBuilder {
    non_tabu: Vec<ScoredScheduleMove>,
    best_tabu: Option<ScoredScheduleMove>,
}

impl TsabShortlistBuilder {
    fn ranking_key(candidate: ScoredScheduleMove) -> ((i128, i128, u8, Reverse<u8>), ScheduleMove) {
        (candidate.score.ranking_key(), candidate.movement)
    }

    fn consider(&mut self, candidate: ScoredScheduleMove, tabu: bool) {
        if self.non_tabu.iter().any(|current| current.movement == candidate.movement)
            || self.best_tabu.is_some_and(|current| current.movement == candidate.movement)
        {
            return;
        }
        if tabu {
            if self.best_tabu.is_none_or(|current| Self::ranking_key(candidate) < Self::ranking_key(current)) {
                self.best_tabu = Some(candidate);
            }
            return;
        }
        if self.non_tabu.len() < TSAB_EXACT_SHORTLIST_CAPACITY {
            self.non_tabu.push(candidate);
            self.non_tabu.sort_unstable_by_key(|&candidate| Self::ranking_key(candidate));
        } else if self.non_tabu.last().is_some_and(|&worst| Self::ranking_key(candidate) < Self::ranking_key(worst)) {
            *self.non_tabu.last_mut().expect("a full shortlist has a worst candidate") = candidate;
            self.non_tabu.sort_unstable_by_key(|&candidate| Self::ranking_key(candidate));
        }
    }

    fn finish(mut self) -> Vec<TsabRankedCandidate> {
        if self.best_tabu.is_some() && self.non_tabu.len() == TSAB_EXACT_SHORTLIST_CAPACITY {
            self.non_tabu.pop();
        }
        let mut shortlisted = Vec::with_capacity(TSAB_EXACT_SHORTLIST_CAPACITY);
        shortlisted.extend(self.non_tabu.into_iter().map(|scored| TsabRankedCandidate { scored, tabu: false }));
        if let Some(scored) = self.best_tabu {
            shortlisted.push(TsabRankedCandidate { scored, tabu: true });
        }
        shortlisted.sort_unstable_by_key(|candidate| (Self::ranking_key(candidate.scored), candidate.tabu));
        shortlisted
    }
}

type TsabN6RestartCandidate = (bool, u64, ScoredScheduleMove);
type TsabN6RestartRankingKey = (Reverse<bool>, (i128, i128, u8, Reverse<u8>), u64, ScheduleMove);

struct TsabN6RestartShortlistBuilder {
    candidates: Vec<TsabN6RestartCandidate>,
}

impl Default for TsabN6RestartShortlistBuilder {
    fn default() -> Self {
        Self { candidates: Vec::with_capacity(TSAB_EXACT_SHORTLIST_CAPACITY) }
    }
}

impl TsabN6RestartShortlistBuilder {
    fn ranking_key(candidate: TsabN6RestartCandidate) -> TsabN6RestartRankingKey {
        (Reverse(candidate.0), candidate.2.score.ranking_key(), candidate.1, candidate.2.movement)
    }

    fn consider(&mut self, candidate: ScoredScheduleMove, acyclicity_certified: bool, seed: u64, epoch: u64) {
        if self.candidates.iter().any(|(_, _, current)| current.movement == candidate.movement) {
            return;
        }
        let diversity = mix64(seed ^ epoch.wrapping_mul(0xa076_1d64_78bd_642f) ^ schedule_move_identity(candidate.movement));
        let ranked = (acyclicity_certified, diversity, candidate);
        if self.candidates.len() < TSAB_EXACT_SHORTLIST_CAPACITY {
            self.candidates.push(ranked);
            self.candidates.sort_unstable_by_key(|&current| Self::ranking_key(current));
        } else if self.candidates.last().is_some_and(|&worst| Self::ranking_key(ranked) < Self::ranking_key(worst)) {
            *self.candidates.last_mut().expect("a full restart shortlist has a worst candidate") = ranked;
            self.candidates.sort_unstable_by_key(|&current| Self::ranking_key(current));
        }
    }

    fn finish(self) -> Vec<TsabN6RestartCandidate> {
        self.candidates
    }

    fn heap_lower_bound_bytes(&self) -> usize {
        self.candidates.capacity().saturating_mul(std::mem::size_of::<TsabN6RestartCandidate>())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScheduleIslandProfile {
    id: usize,
    initial_dispatch_rule: DispatchRule,
    /// Reserved for a reactive constructor restart. It remains earliest-start
    /// and inactive for both the historical and direct-oracle kernels.
    dispatch_rule: DispatchRule,
    scan_width: usize,
    tenure_numerator: u64,
    n6_period: u64,
    restart_multiplier: u64,
    tie_break_salt: u64,
}

impl ScheduleIslandProfile {
    const SCAN_WIDTHS: [usize; 8] = [4, 8, 16, 32, 48, 64, 96, 128];
    const TENURE_NUMERATORS: [u64; 8] = [6, 7, 8, 9, 10, 12, 14, 16];
    const N6_PERIODS: [u64; 8] = [8, 7, 6, 5, 4, 3, 2, 1];
    const RESTART_MULTIPLIERS: [u64; 8] = [16, 12, 10, 8, 8, 6, 5, 4];
    const SEVEN_WORKER_INITIAL_DISPATCH_RULES: [DispatchRule; 8] = [
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[0],
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[1],
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[2],
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[3],
        DispatchRule::ADJUSTED_WORK_PORTFOLIO[4],
        DispatchRule::EarliestStart,
        DispatchRule::EarliestStartThenMostWorkRemaining,
        DispatchRule::EarliestStart,
    ];

    fn initial_dispatch_rule(id: usize, workers: usize) -> DispatchRule {
        if workers == 7 {
            Self::SEVEN_WORKER_INITIAL_DISPATCH_RULES[id]
        } else if id == 4 {
            DispatchRule::EarliestStartThenMostWorkRemaining
        } else {
            DispatchRule::EarliestStart
        }
    }

    fn for_worker(worker: usize, workers: usize) -> Self {
        let id = worker % Self::SCAN_WIDTHS.len();
        if id < 6 {
            return Self {
                id,
                initial_dispatch_rule: Self::initial_dispatch_rule(id, workers),
                dispatch_rule: DispatchRule::EarliestStart,
                scan_width: 32,
                tenure_numerator: 8,
                n6_period: 4,
                restart_multiplier: 0,
                tie_break_salt: 0,
            };
        }
        let width_index = if workers >= 7 { id } else { (id + 3).min(Self::SCAN_WIDTHS.len() - 1) };
        Self {
            id,
            initial_dispatch_rule: Self::initial_dispatch_rule(id, workers),
            dispatch_rule: DispatchRule::EarliestStart,
            scan_width: Self::SCAN_WIDTHS[width_index],
            tenure_numerator: Self::TENURE_NUMERATORS[id],
            n6_period: Self::N6_PERIODS[id],
            restart_multiplier: Self::RESTART_MULTIPLIERS[id],
            tie_break_salt: mix64(u64::try_from(id).unwrap_or(u64::MAX).wrapping_mul(0xa076_1d64_78bd_642f)),
        }
    }

    fn uses_baseline_kernel(self) -> bool {
        self.id < 6
    }

    fn uses_direct_oracle(self) -> bool {
        self.id >= 6
    }

    fn initial_dispatch_name(self) -> &'static str {
        match self.initial_dispatch_rule {
            DispatchRule::EarliestStart => "earliest-start",
            DispatchRule::EarliestStartThenMostWorkRemaining => "earliest-start-then-most-work-remaining",
            DispatchRule::EarliestStartThenMostAdjustedWork(lane) => lane.dispatch_name(),
            DispatchRule::ShortestProcessingTime => "shortest-processing-time",
            DispatchRule::LongestProcessingTime => "longest-processing-time",
            DispatchRule::MostWorkRemaining => "most-work-remaining",
            DispatchRule::Randomized => "randomized",
        }
    }

    fn restart_dispatch_name(self) -> &'static str {
        match self.dispatch_rule {
            DispatchRule::EarliestStart => "earliest-start",
            DispatchRule::EarliestStartThenMostWorkRemaining => "earliest-start-then-most-work-remaining",
            DispatchRule::EarliestStartThenMostAdjustedWork(lane) => lane.dispatch_name(),
            DispatchRule::ShortestProcessingTime => "shortest-processing-time",
            DispatchRule::LongestProcessingTime => "longest-processing-time",
            DispatchRule::MostWorkRemaining => "most-work-remaining",
            DispatchRule::Randomized => "randomized",
        }
    }

    fn scaled_tenure(self, base: u64) -> u64 {
        (base.saturating_mul(self.tenure_numerator).saturating_add(7) / 8).clamp(3, 64)
    }

    fn restart_probe_interval(self, tenure_base: u64) -> u64 {
        if self.uses_baseline_kernel() || self.uses_direct_oracle() {
            return u64::MAX;
        }
        tenure_base
            .saturating_mul(u64::try_from(self.scan_width).unwrap_or(u64::MAX))
            .saturating_mul(self.restart_multiplier)
            .clamp(256, 65_536)
    }

    fn dispatch_probe_interval(self) -> u64 {
        8_192u64.saturating_mul(u64::try_from(1 + self.id % 4).unwrap_or(u64::MAX))
    }
}

/// Persistent tabu trajectory owned by one schedule-search worker.
///
/// The current state may move away from the best known schedule. Only the
/// state-validated elite is returned to the orchestrator.
pub(crate) struct ScheduleSearchSession {
    seed: u64,
    profile: ScheduleIslandProfile,
    requested_strategy: ScheduleJsspSearchStrategy,
    strategy: ScheduleJsspSearchStrategy,
    state: Option<JobShopState>,
    elite_solution: CollectionSolution,
    elite_objective: i64,
    tabu_until: BTreeMap<MachineArc, u64>,
    accepted_step: u64,
    last_improvement_step: u64,
    probe_step: u64,
    last_improvement_probe: u64,
    last_restart_probe: u64,
    dispatch_restart_done: bool,
    restart_epoch: u64,
    scan_epoch: u64,
    consecutive_direct_oracle_failures: u64,
    /// Search-work units already charged but not yet sufficient for the next
    /// four-unit complete-oracle attempt.
    direct_oracle_work_credit: u64,
    /// Exact N5 work completed since the last atomic TSAB elite restart.
    tsab_n5_work_since_restart: u64,
    /// Work charged toward the next eight-unit TSAB elite restart. It remains
    /// persistent so arbitrary slice boundaries cannot change the trajectory.
    tsab_restart_work_credit: u64,
    tsab_restart_epoch: u64,
    tsab_fast_enabled: bool,
    tsab_fast_disabled: bool,
    /// A failed full-oracle checkpoint may leave the private fast trajectory
    /// unsuitable for a later resume. The verified elite remains publishable,
    /// but the session must be discarded unless rebuilding that elite succeeds.
    tsab_fast_hard_failure: bool,
    /// A stopped slice may retain an exact local trajectory that has not yet
    /// crossed its periodic full oracle. A resumed slice must validate it
    /// before selecting or applying another move.
    tsab_fast_validation_due_on_resume: bool,
    tsab_fast_commits_since_validation: u64,
    tsab_pending_elite: Option<CollectionSolution>,
    #[cfg(test)]
    tsab_fast_test_stop_after_attempts: Option<u64>,
    #[cfg(test)]
    tsab_fast_test_stop_after_pending_elite: bool,
    #[cfg(test)]
    tsab_fast_test_return_after_pending_elite: bool,
    #[cfg(test)]
    tsab_fast_test_stop_during_validation: bool,
    tenure_base: u64,
    completed_boundaries: u64,
    tsab_legacy_warmup_work_steps: u64,
    tsab_activation_boundary: Option<u64>,
    scan: PersistentScheduleScan,
    search_elite_token: Option<ScheduleEliteSolveToken>,
    pending_search_elite: Option<ScheduleEliteCandidate>,
    schedule_lns_shadow: Option<ScheduleLnsShadowSession>,
    schedule_relink: ScheduleRelinkWorkspace,
    constructor_multistart: Option<ScheduleConstructorMultistartSession>,
}

impl ScheduleSearchSession {
    fn state(&self) -> &JobShopState {
        self.state.as_ref().expect("search sessions always own a job-shop state")
    }

    fn state_mut(&mut self) -> &mut JobShopState {
        self.state.as_mut().expect("search sessions always own a job-shop state")
    }

    pub(crate) fn take_search_elite_candidate(&mut self) -> Option<ScheduleEliteCandidate> {
        self.pending_search_elite.take()
    }

    pub(crate) fn has_tsab_fast_pending_elite(&self) -> bool {
        self.strategy == ScheduleJsspSearchStrategy::TsabCandidate && self.tsab_fast_enabled && self.tsab_pending_elite.is_some()
    }

    pub(crate) fn tsab_fast_pending_objective(&self) -> Option<i64> {
        if !self.has_tsab_fast_pending_elite() {
            return None;
        }
        self.tsab_pending_elite.as_ref()?.objectives.first().copied()
    }

    pub(crate) fn merge_search_elite_batch(
        &self,
        archive: &mut ScheduleEliteArchive,
        candidates: Vec<ScheduleEliteCandidate>,
        stop: &AtomicBool,
    ) -> Result<super::schedule_elite::ScheduleEliteBatchOutcome, ScheduleEliteError> {
        let token = self.search_elite_token.as_ref().ok_or(ScheduleEliteError::IncompatibleSolve)?;
        archive.consider_candidate_batch(self.state(), token, candidates, stop)
    }

    #[cfg(test)]
    pub(crate) fn test_force_schedule_lns_shadow_due(&mut self) {
        let shadow = self.schedule_lns_shadow.get_or_insert_with(ScheduleLnsShadowSession::default);
        let cadence = if shadow.attempt_epoch == 0 { SCHEDULE_LNS_FIRST_ATTEMPT_PROBES } else { SCHEDULE_LNS_REPEAT_ATTEMPT_PROBES };
        self.probe_step = self
            .probe_step
            .max(self.last_improvement_probe.saturating_add(SCHEDULE_LNS_STAGNATION_PROBES))
            .max(shadow.last_attempt_probe.saturating_add(cadence));
    }

    #[cfg(test)]
    pub(crate) fn test_force_tsab_restart_due(&mut self) {
        debug_assert!(self.requested_strategy.requests_tsab());
        self.strategy = ScheduleJsspSearchStrategy::TsabCandidate;
        self.tsab_legacy_warmup_work_steps = TSAB_LEGACY_WARMUP_WORK_UNITS;
        self.tsab_n5_work_since_restart = TSAB_ELITE_RESTART_N5_WORK_UNITS;
        self.tsab_restart_work_credit = 0;
        self.tsab_fast_enabled = self.state().supports_strict_n5_fast_path();
        self.tsab_fast_disabled = false;
        self.tsab_fast_hard_failure = false;
        self.tsab_fast_validation_due_on_resume = false;
        self.tsab_fast_commits_since_validation = 0;
        self.tsab_pending_elite = None;
        self.tenure_base = TSAB_TENURE;
        self.tabu_until.clear();
        clear_persistent_scan(self);
        self.state_mut().retain_canonical_critical_blocks_only();
    }

    #[cfg(test)]
    pub(crate) fn test_disable_tsab_fast_path(&mut self) {
        debug_assert_eq!(self.strategy, ScheduleJsspSearchStrategy::TsabCandidate);
        self.tsab_fast_enabled = false;
        self.tsab_fast_disabled = true;
        self.tsab_fast_hard_failure = false;
        self.tsab_fast_validation_due_on_resume = false;
        self.tsab_fast_commits_since_validation = 0;
        self.tsab_pending_elite = None;
    }

    #[cfg(test)]
    pub(crate) fn test_stop_after_fast_attempts(&mut self, attempts: u64) {
        debug_assert_eq!(self.strategy, ScheduleJsspSearchStrategy::TsabCandidate);
        self.tsab_fast_test_stop_after_attempts = Some(attempts.max(1));
    }

    #[cfg(test)]
    pub(crate) fn test_stop_after_fast_pending_elite(&mut self) {
        debug_assert!(self.requested_strategy.requests_tsab());
        self.tsab_fast_test_stop_after_pending_elite = true;
    }

    #[cfg(test)]
    pub(crate) fn test_return_after_fast_pending_elite(&mut self) {
        debug_assert!(self.requested_strategy.requests_tsab());
        self.tsab_fast_test_return_after_pending_elite = true;
    }

    #[cfg(test)]
    pub(crate) fn test_stop_during_tsab_fast_validation(&mut self) {
        debug_assert_eq!(self.strategy, ScheduleJsspSearchStrategy::TsabCandidate);
        self.tsab_fast_test_stop_during_validation = true;
    }

    #[cfg(test)]
    pub(crate) fn test_tsab_pending_elite_objective(&self) -> Option<i64> {
        self.tsab_pending_elite.as_ref().and_then(|solution| solution.objectives.first().copied())
    }

    #[cfg(test)]
    pub(crate) fn test_force_tsab_fast_work_cap(&mut self, pop_cap: usize) {
        debug_assert_eq!(self.strategy, ScheduleJsspSearchStrategy::TsabCandidate);
        self.state_mut().test_configure_strict_n5_work_cap(pop_cap, false);
    }

    #[cfg(test)]
    pub(crate) fn test_tsab_fast_validation_due_on_resume(&self) -> bool {
        self.tsab_fast_validation_due_on_resume
    }

    #[cfg(test)]
    pub(crate) fn test_schedule_lns_workspace_capacities(&self) -> Option<ScheduleLnsWorkspaceCapacities> {
        self.schedule_lns_shadow.as_ref().map(|shadow| shadow.workspace.capacities())
    }

    #[cfg(test)]
    pub(crate) fn test_same_tabu_trajectory(&self, other: &Self) -> bool {
        self.state().makespan() == other.state().makespan()
            && self.state().canonical_path_policy() == other.state().canonical_path_policy()
            && self.state().starts() == other.state().starts()
            && self.state().machine_sequences() == other.state().machine_sequences()
            && self.elite_solution.starts == other.elite_solution.starts
            && self.elite_solution.objectives == other.elite_solution.objectives
            && self.elite_objective == other.elite_objective
            && self.tabu_until == other.tabu_until
            && self.accepted_step == other.accepted_step
            && self.last_improvement_step == other.last_improvement_step
            && self.probe_step == other.probe_step
            && self.last_improvement_probe == other.last_improvement_probe
            && self.last_restart_probe == other.last_restart_probe
            && self.restart_epoch == other.restart_epoch
            && self.requested_strategy == other.requested_strategy
            && self.strategy == other.strategy
            && self.scan_epoch == other.scan_epoch
            && self.scan.movements == other.scan.movements
            && self.scan.scored_movements == other.scan.scored_movements
            && self.scan.estimated_movements == other.scan.estimated_movements
            && self.scan.cursor == other.scan.cursor
            && self.scan.end == other.scan.end
            && self.scan.best_admissible == other.scan.best_admissible
            && self.scan.best_any == other.scan.best_any
            && self.scan.tsab_ranks == other.scan.tsab_ranks
            && self.tsab_n5_work_since_restart == other.tsab_n5_work_since_restart
            && self.tsab_restart_work_credit == other.tsab_restart_work_credit
            && self.tsab_restart_epoch == other.tsab_restart_epoch
            && self.tsab_fast_enabled == other.tsab_fast_enabled
            && self.tsab_fast_disabled == other.tsab_fast_disabled
            && self.tsab_fast_hard_failure == other.tsab_fast_hard_failure
            && self.tsab_fast_validation_due_on_resume == other.tsab_fast_validation_due_on_resume
            && self.tsab_fast_commits_since_validation == other.tsab_fast_commits_since_validation
            && match (&self.tsab_pending_elite, &other.tsab_pending_elite) {
                (None, None) => true,
                (Some(left), Some(right)) => {
                    left.objectives == right.objectives
                        && left.starts == right.starts
                        && left.presences == right.presences
                        && left.machines == right.machines
                        && left.modes == right.modes
                }
                _ => false,
            }
    }

    #[cfg(test)]
    pub(crate) fn test_tsab_phase_state(&self) -> (ScheduleJsspSearchStrategy, u64, u64, Option<u64>, i64, i64, usize, u64, u64) {
        (
            self.strategy,
            self.completed_boundaries,
            self.tsab_legacy_warmup_work_steps,
            self.tsab_activation_boundary,
            self.state().makespan(),
            self.elite_objective,
            self.tabu_until.len(),
            self.direct_oracle_work_credit,
            self.tenure_base,
        )
    }

    #[cfg(test)]
    pub(crate) fn test_tsab_requested_strategy(&self) -> ScheduleJsspSearchStrategy {
        self.requested_strategy
    }

    #[cfg(test)]
    pub(crate) fn test_tsab_canonical_path(&self) -> (CanonicalPathPolicy, u64) {
        (self.state().canonical_path_policy(), self.state().canonical_path_fingerprint())
    }

    #[cfg(test)]
    pub(crate) fn test_tsab_restart_state(&self) -> (u64, u64, u64, i64, i64, bool) {
        (
            self.tsab_n5_work_since_restart,
            self.tsab_restart_work_credit,
            self.tsab_restart_epoch,
            self.state().makespan(),
            self.elite_objective,
            self.scan.end == 0,
        )
    }

    #[cfg(test)]
    pub(crate) fn test_tsab_state_matches_full_oracle(&mut self) -> bool {
        self.state_mut().matches_full_oracle(&AtomicBool::new(false)).unwrap_or(false)
    }
}

#[derive(Default)]
struct PersistentScheduleSliceMetrics {
    tabu_steps: u64,
    tabu_hits: u64,
    tabu_aspirations: u64,
    tabu_forced_moves: u64,
    session_initializations: u64,
    session_resumes: u64,
    session_rebases: u64,
    island_profile_mask: u64,
    baseline_island_profile_mask: u64,
    scored_island_profile_mask: u64,
    reactive_restarts: u64,
    reactive_restart_dispatches: u64,
    reactive_restart_perturbations: u64,
    reactive_restart_rebuild_failures: u64,
    island_scored_candidates: u64,
    island_shortlisted_candidates: u64,
    approximate_candidates_generated: u64,
    approximation_score_items: u64,
    approximation_sort_items: u64,
    approximation_local_span_items: u64,
    approximation_elapsed: Duration,
    approximation_work_units: u64,
    exact_probes_avoided: u64,
    search_elite_snapshot_captures: u64,
    search_elite_snapshot_interruptions: u64,
    search_elite_snapshot_errors: u64,
    search_elite_capture_worker_elapsed_sum: Duration,
    search_elite_snapshot_peak_heap_lower_bound_bytes: usize,
    schedule_lns_shadow_owner_worker_mask: u64,
    schedule_lns: ScheduleLnsMetrics,
    schedule_lns_workspace_peak_bytes: usize,
    schedule_path_relink: ScheduleRelinkMetrics,
    schedule_tsab: ScheduleTsabMetrics,
}

// A Large-TA worker currently evaluates roughly one thousand probes in a
// 60-second diagnostic run.  Keep the first shadow observation inside that
// window, then space later repairs far enough apart that exact subproblems do
// not dominate the local-search trajectory.
const SCHEDULE_LNS_STAGNATION_PROBES: u64 = 512;
const SCHEDULE_LNS_FIRST_ATTEMPT_PROBES: u64 = 512;
const SCHEDULE_LNS_REPEAT_ATTEMPT_PROBES: u64 = 2_048;
const SCHEDULE_LNS_LOCAL_BUDGET: Duration = Duration::from_millis(500);
const SCHEDULE_LNS_VERIFICATION_RESERVE: Duration = Duration::from_millis(100);

fn schedule_lns_shadow_due(probe_step: u64, last_improvement_probe: u64, attempt_epoch: u64, last_attempt_probe: u64) -> bool {
    let cadence = if attempt_epoch == 0 { SCHEDULE_LNS_FIRST_ATTEMPT_PROBES } else { SCHEDULE_LNS_REPEAT_ATTEMPT_PROBES };
    if probe_step.saturating_sub(last_attempt_probe) < cadence {
        return false;
    }
    attempt_epoch == 0 || probe_step.saturating_sub(last_improvement_probe) >= SCHEDULE_LNS_STAGNATION_PROBES
}

#[cfg(test)]
pub(crate) fn audit_schedule_lns_shadow_policy(
    probe_step: u64,
    last_improvement_probe: u64,
    attempt_epoch: u64,
    last_attempt_probe: u64,
) -> (bool, u64, u64, u64, u64, u64) {
    (
        schedule_lns_shadow_due(probe_step, last_improvement_probe, attempt_epoch, last_attempt_probe),
        SCHEDULE_LNS_STAGNATION_PROBES,
        SCHEDULE_LNS_FIRST_ATTEMPT_PROBES,
        SCHEDULE_LNS_REPEAT_ATTEMPT_PROBES,
        u64::try_from(SCHEDULE_LNS_LOCAL_BUDGET.as_millis()).unwrap_or(u64::MAX),
        u64::try_from(SCHEDULE_LNS_VERIFICATION_RESERVE.as_millis()).unwrap_or(u64::MAX),
    )
}

fn maybe_run_schedule_lns_shadow(
    session: &mut ScheduleSearchSession,
    slice: &mut PersistentScheduleSliceMetrics,
    worker: usize,
    resumed: bool,
    enabled: bool,
    slice_probes: u64,
    stop: &AtomicBool,
) {
    if worker != 0 || !enabled {
        return;
    }
    slice.schedule_lns_shadow_owner_worker_mask = 1;
    if !resumed || slice_probes == 0 || stop.load(Ordering::Acquire) {
        return;
    }

    let (attempt_epoch, last_attempt_probe) =
        session.schedule_lns_shadow.as_ref().map_or((0, 0), |shadow| (shadow.attempt_epoch, shadow.last_attempt_probe));
    if !schedule_lns_shadow_due(session.probe_step, session.last_improvement_probe, attempt_epoch, last_attempt_probe) {
        return;
    }

    let attempt_seed = mix64(session.seed ^ attempt_epoch.wrapping_mul(0xd1b5_4a32_d192_ed03));
    let ScheduleSearchSession { state, schedule_lns_shadow, .. } = session;
    let shadow = schedule_lns_shadow.get_or_insert_with(ScheduleLnsShadowSession::default);
    shadow.last_attempt_probe = session.probe_step;
    shadow.attempt_epoch = shadow.attempt_epoch.saturating_add(1);

    let config = ScheduleLnsConfig::shadow(64, 128, SCHEDULE_LNS_LOCAL_BUDGET, SCHEDULE_LNS_VERIFICATION_RESERVE, attempt_seed);
    let mut attempt_metrics = ScheduleLnsMetrics::default();
    let _ = repair_critical_window_with_workspace(
        state.as_ref().expect("search sessions always own a job-shop state"),
        config,
        &mut attempt_metrics,
        &mut shadow.workspace,
        stop,
    );
    slice.schedule_lns.add(attempt_metrics);
    slice.schedule_lns_workspace_peak_bytes = slice.schedule_lns_workspace_peak_bytes.max(shadow.workspace.capacities().estimated_bytes);
}

fn capture_search_elite_shadow(
    session: &mut ScheduleSearchSession,
    slice: &mut PersistentScheduleSliceMetrics,
    enabled: bool,
    stop: &AtomicBool,
) {
    if !enabled || session.pending_search_elite.is_some() {
        return;
    }
    let started = Instant::now();
    let Some(token) = session.search_elite_token.as_ref() else {
        slice.search_elite_snapshot_errors = slice.search_elite_snapshot_errors.saturating_add(1);
        slice.search_elite_capture_worker_elapsed_sum = slice.search_elite_capture_worker_elapsed_sum.saturating_add(started.elapsed());
        return;
    };
    match ScheduleEliteCandidate::capture(session.state(), token, stop) {
        Ok(candidate) => {
            slice.search_elite_snapshot_captures = slice.search_elite_snapshot_captures.saturating_add(1);
            let replacement_bytes = session
                .pending_search_elite
                .as_ref()
                .map_or(0, ScheduleEliteCandidate::heap_lower_bound_bytes)
                .saturating_add(candidate.heap_lower_bound_bytes());
            slice.search_elite_snapshot_peak_heap_lower_bound_bytes =
                slice.search_elite_snapshot_peak_heap_lower_bound_bytes.max(replacement_bytes);
            session.pending_search_elite = Some(candidate);
        }
        Err(ScheduleEliteError::Interrupted) => {
            slice.search_elite_snapshot_interruptions = slice.search_elite_snapshot_interruptions.saturating_add(1);
        }
        Err(_) => slice.search_elite_snapshot_errors = slice.search_elite_snapshot_errors.saturating_add(1),
    }
    slice.search_elite_capture_worker_elapsed_sum = slice.search_elite_capture_worker_elapsed_sum.saturating_add(started.elapsed());
}

struct ConstructedSchedule {
    starts: Vec<i64>,
    chosen: Vec<usize>,
    present: Vec<bool>,
}

struct ConstructedIncumbent {
    schedule: ConstructedSchedule,
    durations: Vec<i64>,
    machines: Vec<i64>,
    score: (i64, i64),
    solution: CollectionSolution,
}

#[derive(Clone, Copy)]
enum SgsRule {
    Stable,
    LongestRemainingPath,
    ShortestProcessingTime,
    LongestProcessingTime,
    MostSuccessors,
    Seeded,
}

const SGS_RULES: [SgsRule; 5] = [
    SgsRule::Stable,
    SgsRule::LongestRemainingPath,
    SgsRule::ShortestProcessingTime,
    SgsRule::LongestProcessingTime,
    SgsRule::MostSuccessors,
];

#[derive(Clone, Copy)]
struct Interrupted;

type Interruptible<T> = Result<T, Interrupted>;

fn checkpoint(stop: &AtomicBool) -> Interruptible<()> {
    if stop.load(Ordering::Acquire) {
        Err(Interrupted)
    } else {
        Ok(())
    }
}

fn collect_interruptible<T>(values: impl IntoIterator<Item = T>, stop: &AtomicBool) -> Interruptible<Vec<T>> {
    checkpoint(stop)?;
    let mut values = values.into_iter();
    let (lower, upper) = values.size_hint();
    let mut collected = Vec::with_capacity(upper.unwrap_or(lower));
    loop {
        checkpoint(stop)?;
        let Some(value) = values.next() else {
            break;
        };
        collected.push(value);
    }
    checkpoint(stop)?;
    Ok(collected)
}

fn sift_down(values: &mut [i64], mut root: usize, end: usize, stop: &AtomicBool) -> Interruptible<()> {
    loop {
        checkpoint(stop)?;
        let left = root.saturating_mul(2).saturating_add(1);
        if left >= end {
            return Ok(());
        }
        let right = left + 1;
        let larger = if right < end && values[right] > values[left] { right } else { left };
        if values[root] >= values[larger] {
            return Ok(());
        }
        values.swap(root, larger);
        root = larger;
    }
}

fn sort_dedup_interruptible(values: &mut Vec<i64>, stop: &AtomicBool) -> Interruptible<()> {
    checkpoint(stop)?;
    if values.len() < 2 {
        return Ok(());
    }
    let len = values.len();
    for root in (0..len / 2).rev() {
        checkpoint(stop)?;
        sift_down(values, root, len, stop)?;
    }
    for end in (1..len).rev() {
        checkpoint(stop)?;
        values.swap(0, end);
        sift_down(values, 0, end, stop)?;
    }

    let mut write = 1usize;
    for read in 1..values.len() {
        checkpoint(stop)?;
        if values[read] != values[write - 1] {
            values[write] = values[read];
            write += 1;
        }
    }
    values.truncate(write);
    checkpoint(stop)
}

/// Score of an interval schedule: total constraint violation (bound, precedence,
/// resource), then the makespan. Smaller is better, violation first.
/// Duration of an interval under the chosen mode (the fixed duration when it has
/// no modes).
fn iv_duration(iv: &IntervalVar, mode: usize) -> i64 {
    if iv.modes.is_empty() {
        iv.duration
    } else {
        iv.modes[mode].duration
    }
}

/// Chosen machine of an interval, or `-1` when it has no modes (no machine).
fn iv_machine(iv: &IntervalVar, mode: usize) -> i64 {
    if iv.modes.is_empty() {
        -1
    } else {
        i64::try_from(iv.modes[mode].machine).expect("validated schedule mode machine fits i64")
    }
}

fn machines_are_representable(sched: &Schedule, stop: &AtomicBool) -> Interruptible<bool> {
    for interval in &sched.intervals {
        checkpoint(stop)?;
        for mode in &interval.modes {
            checkpoint(stop)?;
            if i64::try_from(mode.machine).is_err() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Inclusive start bounds implied by the selected mode.
fn iv_start_window(iv: &IntervalVar, mode: usize) -> (i64, i64) {
    if iv.modes.is_empty() {
        (0, iv.horizon.saturating_sub(iv.duration))
    } else {
        iv.modes[mode].start_window
    }
}

/// Stable semantic identity implied by the selected mode.
fn iv_mode_reference(iv: &IntervalVar, mode: usize) -> Option<usize> {
    (!iv.modes.is_empty()).then(|| iv.modes[mode].reference).flatten()
}

fn random_start((start_min, start_max): (i64, i64), random: u64) -> i64 {
    debug_assert!(start_min <= start_max);
    let width = i128::from(start_max) - i128::from(start_min) + 1;
    let offset = i128::from(random) % width;
    i64::try_from(i128::from(start_min) + offset).expect("sampled start stays inside its i64 window")
}

/// Durations and machines implied by the chosen mode of each interval.
fn mode_view(sched: &Schedule, chosen: &[usize], stop: &AtomicBool) -> Interruptible<(Vec<i64>, Vec<i64>)> {
    checkpoint(stop)?;
    let mut durations = Vec::with_capacity(chosen.len());
    checkpoint(stop)?;
    let mut machines = Vec::with_capacity(chosen.len());
    for (interval, &mode) in sched.intervals.iter().zip(chosen) {
        checkpoint(stop)?;
        durations.push(iv_duration(interval, mode));
        machines.push(iv_machine(interval, mode));
    }
    checkpoint(stop)?;
    Ok((durations, machines))
}

fn mode_references(sched: &Schedule, chosen: &[usize], present: &[bool], stop: &AtomicBool) -> Interruptible<Vec<Option<usize>>> {
    collect_interruptible(
        sched
            .intervals
            .iter()
            .zip(chosen)
            .zip(present)
            .map(|((interval, &mode), &present)| present.then(|| iv_mode_reference(interval, mode)).flatten()),
        stop,
    )
}

fn overlaps(start: i64, duration: i64, other_start: i64, other_duration: i64) -> bool {
    start < other_start.saturating_add(other_duration) && other_start < start.saturating_add(duration)
}

/// Earliest start not conflicting with already scheduled operations. Every
/// rejection advances to an existing end event, so the loop is finite and its
/// candidate count is a useful construction-work metric.
#[allow(clippy::too_many_arguments)]
fn earliest_resource_start(
    sched: &Schedule,
    index: usize,
    duration: i64,
    machine: i64,
    earliest: i64,
    start_window: (i64, i64),
    durations: &[i64],
    machines: &[i64],
    starts: &[i64],
    present: &[bool],
    scheduled: &[bool],
    candidates: &mut u64,
    stop: &AtomicBool,
) -> Interruptible<Option<i64>> {
    checkpoint(stop)?;
    let (start_min, latest) = start_window;
    let mut start = earliest.max(start_min);
    while start <= latest {
        checkpoint(stop)?;
        *candidates = candidates.saturating_add(1);
        let end = start.saturating_add(duration);
        let mut advance: Option<i64> = None;
        for resource in &sched.resources {
            checkpoint(stop)?;
            match resource {
                Resource::NoOverlap(group) => {
                    let mut applies = false;
                    for &member in group {
                        checkpoint(stop)?;
                        if member == index {
                            applies = true;
                            break;
                        }
                    }
                    if !applies {
                        continue;
                    }
                    for &other in group {
                        checkpoint(stop)?;
                        if other != index
                            && present.get(other).copied().unwrap_or(false)
                            && scheduled.get(other).copied().unwrap_or(false)
                            && overlaps(start, duration, starts[other], durations[other])
                        {
                            advance =
                                Some(advance.map_or(starts[other] + durations[other], |old| old.max(starts[other] + durations[other])));
                        }
                    }
                }
                Resource::MachineNoOverlap if machine >= 0 => {
                    for other in 0..sched.intervals.len() {
                        checkpoint(stop)?;
                        if other != index
                            && present[other]
                            && scheduled[other]
                            && machines[other] == machine
                            && overlaps(start, duration, starts[other], durations[other])
                        {
                            advance =
                                Some(advance.map_or(starts[other] + durations[other], |old| old.max(starts[other] + durations[other])));
                        }
                    }
                }
                Resource::Cumulative { demands, capacity } => {
                    let mut demand = 0;
                    for &(task, task_demand) in demands {
                        checkpoint(stop)?;
                        if task == index {
                            demand = task_demand;
                            break;
                        }
                    }
                    if demand > *capacity {
                        return Ok(None);
                    }
                    if demand == 0 {
                        continue;
                    }
                    checkpoint(stop)?;
                    let mut events = vec![start];
                    for &(other, _) in demands {
                        checkpoint(stop)?;
                        if other != index && present[other] && scheduled[other] {
                            let other_start = starts[other];
                            let other_end = other_start.saturating_add(durations[other]);
                            if start < other_start && other_start < end {
                                events.push(other_start);
                            }
                            if start < other_end && other_end < end {
                                events.push(other_end);
                            }
                        }
                    }
                    sort_dedup_interruptible(&mut events, stop)?;
                    for event in events {
                        checkpoint(stop)?;
                        let mut usage = i128::from(demand);
                        let mut next_end: Option<i64> = None;
                        for &(other, other_demand) in demands {
                            checkpoint(stop)?;
                            if other != index
                                && present[other]
                                && scheduled[other]
                                && starts[other] <= event
                                && event < starts[other].saturating_add(durations[other])
                            {
                                usage += i128::from(other_demand);
                                let other_end = starts[other].saturating_add(durations[other]);
                                next_end = Some(next_end.map_or(other_end, |old| old.min(other_end)));
                            }
                        }
                        if usage > i128::from(*capacity) {
                            advance = Some(advance.map_or(next_end.unwrap_or(end), |old| old.max(next_end.unwrap_or(end))));
                            break;
                        }
                    }
                }
                Resource::MachineNoOverlap => {}
            }
        }
        let Some(next) = advance else {
            checkpoint(stop)?;
            return Ok(Some(start));
        };
        start = next.max(start.saturating_add(1));
    }
    checkpoint(stop)?;
    Ok(None)
}

fn ready_priority(
    rule: SgsRule,
    operation: usize,
    durations: &[i64],
    bottom: &[i64],
    successors: &[Vec<usize>],
    seed: u64,
) -> (i128, i128, u64, usize) {
    let duration = durations[operation];
    let successor_count = successors[operation].len();
    match rule {
        SgsRule::Stable => (0, 0, 0, operation),
        SgsRule::LongestRemainingPath => (-i128::from(bottom[operation]), i128::from(duration), 0, operation),
        SgsRule::ShortestProcessingTime => (i128::from(duration), -i128::from(bottom[operation]), 0, operation),
        SgsRule::LongestProcessingTime => (-i128::from(duration), -i128::from(bottom[operation]), 0, operation),
        SgsRule::MostSuccessors => (-i128::try_from(successor_count).unwrap_or(i128::MAX), -i128::from(bottom[operation]), 0, operation),
        SgsRule::Seeded => (0, 0, mix64(seed ^ u64::try_from(operation).unwrap_or(u64::MAX)), operation),
    }
}

fn select_ready(
    ready: &[usize],
    rule: SgsRule,
    durations: &[i64],
    bottom: &[i64],
    successors: &[Vec<usize>],
    seed: u64,
    stop: &AtomicBool,
) -> Interruptible<usize> {
    let mut selected = 0usize;
    for position in 1..ready.len() {
        checkpoint(stop)?;
        if ready_priority(rule, ready[position], durations, bottom, successors, seed)
            < ready_priority(rule, ready[selected], durations, bottom, successors, seed)
        {
            selected = position;
        }
    }
    Ok(selected)
}

/// Serial schedule generation scheme. Eligible operations are selected by one
/// generic priority rule and placed at their earliest resource-feasible event.
/// For flexible operations, the mode with the earliest completion is selected.
fn serial_schedule(sched: &Schedule, rule: SgsRule, seed: u64, stop: &AtomicBool) -> (Interruptible<Option<ConstructedSchedule>>, u64) {
    if let Err(interrupted) = checkpoint(stop) {
        return (Err(interrupted), 0);
    }
    let n = sched.intervals.len();
    let present = match collect_interruptible(sched.intervals.iter().map(|interval| !interval.optional), stop) {
        Ok(present) => present,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut indegree = match collect_interruptible(std::iter::repeat_n(0usize, n), stop) {
        Ok(indegree) => indegree,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut successors = match collect_interruptible((0..n).map(|_| Vec::new()), stop) {
        Ok(successors) => successors,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    for &(before, after) in &sched.precedences {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), 0);
        }
        if present[before] && present[after] {
            successors[before].push(after);
        }
    }
    for list in &mut successors {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), 0);
        }
        list.sort_unstable();
        list.dedup();
        for &successor in list.iter() {
            indegree[successor] = indegree[successor].saturating_add(1);
        }
    }
    let base_durations = match collect_interruptible(
        sched.intervals.iter().map(|interval| interval.modes.iter().map(|mode| mode.duration).min().unwrap_or(interval.duration)),
        stop,
    ) {
        Ok(durations) => durations,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let bottom = match PrecedenceDag::compile(successors.clone(), stop).and_then(|dag| dag.remaining_paths(&base_durations, stop)) {
        Some(bottom) => bottom,
        None if stop.load(Ordering::Acquire) => return (Err(Interrupted), 0),
        None => return (Ok(None), 0),
    };
    let mut ready = Vec::new();
    for index in 0..n {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), 0);
        }
        if present[index] && indegree[index] == 0 {
            ready.push(index);
        }
    }
    let mut chosen = match collect_interruptible(std::iter::repeat_n(0usize, n), stop) {
        Ok(chosen) => chosen,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let (mut durations, mut machines) = match mode_view(sched, &chosen, stop) {
        Ok(view) => view,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut starts = match collect_interruptible(std::iter::repeat_n(0i64, n), stop) {
        Ok(starts) => starts,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut scheduled = match collect_interruptible(present.iter().map(|value| !*value), stop) {
        Ok(scheduled) => scheduled,
        Err(interrupted) => return (Err(interrupted), 0),
    };
    let mut scheduled_count = 0usize;
    let mut candidates = 0u64;
    while !ready.is_empty() {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), candidates);
        }
        let ready_position = match select_ready(&ready, rule, &base_durations, &bottom, &successors, seed, stop) {
            Ok(position) => position,
            Err(interrupted) => return (Err(interrupted), candidates),
        };
        let index = ready.swap_remove(ready_position);
        let mut earliest = 0;
        for &(before, after) in &sched.precedences {
            if let Err(interrupted) = checkpoint(stop) {
                return (Err(interrupted), candidates);
            }
            if after == index && present[before] {
                earliest = earliest.max(starts[before].saturating_add(durations[before]));
            }
        }
        let mode_count = sched.intervals[index].modes.len().max(1);
        let mut best: Option<(i64, i64, usize, i64, i64)> = None;
        for mode in 0..mode_count {
            if let Err(interrupted) = checkpoint(stop) {
                return (Err(interrupted), candidates);
            }
            let duration = iv_duration(&sched.intervals[index], mode);
            let machine = iv_machine(&sched.intervals[index], mode);
            let start_window = iv_start_window(&sched.intervals[index], mode);
            let start = match earliest_resource_start(
                sched,
                index,
                duration,
                machine,
                earliest,
                start_window,
                &durations,
                &machines,
                &starts,
                &present,
                &scheduled,
                &mut candidates,
                stop,
            ) {
                Ok(start) => start,
                Err(interrupted) => return (Err(interrupted), candidates),
            };
            let Some(start) = start else {
                continue;
            };
            let candidate = (start.saturating_add(duration), start, mode, duration, machine);
            if best.is_none_or(|current| candidate < current) {
                best = Some(candidate);
            }
        }
        let Some((_, start, mode, duration, machine)) = best else {
            return (Ok(None), candidates);
        };
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), candidates);
        }
        chosen[index] = mode;
        durations[index] = duration;
        machines[index] = machine;
        starts[index] = start;
        scheduled[index] = true;
        scheduled_count += 1;
        for &successor in &successors[index] {
            if let Err(interrupted) = checkpoint(stop) {
                return (Err(interrupted), candidates);
            }
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                ready.push(successor);
            }
        }
    }
    let mut required_count = 0usize;
    for &is_present in &present {
        if let Err(interrupted) = checkpoint(stop) {
            return (Err(interrupted), candidates);
        }
        required_count += usize::from(is_present);
    }
    if scheduled_count != required_count {
        return (Ok(None), candidates);
    }
    if let Err(interrupted) = checkpoint(stop) {
        return (Err(interrupted), candidates);
    }
    (Ok(Some(ConstructedSchedule { starts, chosen, present })), candidates)
}

fn construct_schedule(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> (Interruptible<Option<ConstructedIncumbent>>, u64, Option<Duration>) {
    let started = Instant::now();
    let mut candidates = 0u64;
    let mut first_feasible = None;
    let mut best: Option<ConstructedIncumbent> = None;
    let attempts = SGS_RULES
        .iter()
        .copied()
        .map(|rule| (rule, 0))
        .chain((0..3u64).map(|lane| (SgsRule::Seeded, mix64(seed ^ lane.wrapping_mul(0x9e37_79b9_7f4a_7c15)))));

    let interrupted_result = |best: Option<ConstructedIncumbent>, interrupted, candidates, first_feasible| {
        if let Some(incumbent) = best {
            (Ok(Some(incumbent)), candidates, first_feasible)
        } else {
            (Err(interrupted), candidates, first_feasible)
        }
    };

    for (rule, rule_seed) in attempts {
        if let Err(interrupted) = checkpoint(stop) {
            return interrupted_result(best, interrupted, candidates, first_feasible);
        }
        let (constructed, evaluated) = serial_schedule(sched, rule, rule_seed, stop);
        candidates = candidates.saturating_add(evaluated);
        let schedule = match constructed {
            Ok(Some(schedule)) => schedule,
            Ok(None) => continue,
            Err(interrupted) => return interrupted_result(best, interrupted, candidates, first_feasible),
        };
        let (durations, machines) = match mode_view(sched, &schedule.chosen, stop) {
            Ok(view) => view,
            Err(interrupted) => return interrupted_result(best, interrupted, candidates, first_feasible),
        };
        let score = match schedule_score(sched, &schedule.chosen, &durations, &machines, &schedule.starts, &schedule.present, stop) {
            Ok(score) if score.0 == 0 => score,
            Ok(_) => continue,
            Err(interrupted) => return interrupted_result(best, interrupted, candidates, first_feasible),
        };
        let solution = match snapshot_solution(sched, &schedule.chosen, &schedule.starts, &schedule.present, score, stop) {
            Ok(solution) => solution,
            Err(interrupted) => return interrupted_result(best, interrupted, candidates, first_feasible),
        };
        first_feasible.get_or_insert_with(|| started.elapsed());
        if best.as_ref().is_none_or(|incumbent| score < incumbent.score) {
            best = Some(ConstructedIncumbent { schedule, durations, machines, score, solution });
            report(score.1);
        }
    }

    (Ok(best), candidates, first_feasible)
}

/// Overlap (>= 0) of two intervals' time spans.
fn pair_overlap(starts: &[i64], dur: &[i64], present: &[bool], i: usize, j: usize) -> i64 {
    if !present[i] || !present[j] {
        return 0;
    }
    ((starts[i] + dur[i]).min(starts[j] + dur[j]) - starts[i].max(starts[j])).max(0)
}

fn schedule_score(
    sched: &Schedule,
    chosen: &[usize],
    dur: &[i64],
    mach: &[i64],
    starts: &[i64],
    present: &[bool],
    stop: &AtomicBool,
) -> Interruptible<(i64, i64)> {
    checkpoint(stop)?;
    let mut viol = 0i64;
    let mut makespan = 0i64;
    for (i, iv) in sched.intervals.iter().enumerate() {
        checkpoint(stop)?;
        if !present[i] {
            continue;
        }
        let end = starts[i].saturating_add(dur[i]);
        let (start_min, start_max) = iv_start_window(iv, chosen[i]);
        if starts[i] < start_min {
            viol = viol.saturating_add(start_min.saturating_sub(starts[i]));
        }
        if starts[i] > start_max {
            viol = viol.saturating_add(starts[i].saturating_sub(start_max));
        }
        makespan = makespan.max(end);
    }
    for &(a, b) in &sched.precedences {
        checkpoint(stop)?;
        if present[a] && present[b] {
            viol = viol.saturating_add((starts[a].saturating_add(dur[a]) - starts[b]).max(0));
        }
    }
    for res in &sched.resources {
        checkpoint(stop)?;
        match res {
            Resource::NoOverlap(ivs) => {
                for x in 0..ivs.len() {
                    checkpoint(stop)?;
                    for y in (x + 1)..ivs.len() {
                        checkpoint(stop)?;
                        viol = viol.saturating_add(pair_overlap(starts, dur, present, ivs[x], ivs[y]));
                    }
                }
            }
            Resource::MachineNoOverlap => {
                // Group moded intervals by their chosen machine; intervals on the
                // same machine may not overlap.
                let n = sched.intervals.len();
                for i in 0..n {
                    checkpoint(stop)?;
                    if mach[i] < 0 {
                        continue;
                    }
                    for j in (i + 1)..n {
                        checkpoint(stop)?;
                        if mach[i] == mach[j] {
                            viol = viol.saturating_add(pair_overlap(starts, dur, present, i, j));
                        }
                    }
                }
            }
            Resource::Cumulative { demands, capacity } => {
                viol = viol.saturating_add(cumulative_overload(demands, *capacity, dur, starts, present, stop)?);
            }
        }
    }
    checkpoint(stop)?;
    Ok((viol, makespan))
}

/// Total resource-overload area of a cumulative resource (usage above capacity,
/// integrated over time). Usage changes only at interval boundaries.
fn cumulative_overload(
    demands: &[(usize, i64)],
    capacity: i64,
    dur: &[i64],
    starts: &[i64],
    present: &[bool],
    stop: &AtomicBool,
) -> Interruptible<i64> {
    checkpoint(stop)?;
    let mut times: Vec<i64> = Vec::with_capacity(demands.len() * 2);
    for &(i, _) in demands {
        checkpoint(stop)?;
        if !present[i] {
            continue;
        }
        times.push(starts[i]);
        times.push(starts[i].saturating_add(dur[i]));
    }
    sort_dedup_interruptible(&mut times, stop)?;
    let mut total = 0i128;
    for w in times.windows(2) {
        checkpoint(stop)?;
        let (t0, t1) = (w[0], w[1]);
        let mut usage = 0i128;
        for &(i, demand) in demands {
            checkpoint(stop)?;
            if present[i] && starts[i] <= t0 && t0 < starts[i].saturating_add(dur[i]) {
                usage += i128::from(demand);
            }
        }
        let over = (usage - i128::from(capacity)).max(0);
        let width = i128::from(t1) - i128::from(t0);
        total = total.saturating_add(over.saturating_mul(width)).min(i128::from(i64::MAX));
    }
    checkpoint(stop)?;
    Ok(i64::try_from(total).expect("non-negative cumulative overload is capped at i64::MAX"))
}

/// Earliest start of each interval respecting precedence only (a longest-path
/// forward pass); resources are left to the search to fix.
fn earliest_starts(sched: &Schedule, chosen: &[usize], dur: &[i64], present: &[bool], stop: &AtomicBool) -> Interruptible<Vec<i64>> {
    let mut starts =
        collect_interruptible(sched.intervals.iter().zip(chosen).map(|(interval, &mode)| iv_start_window(interval, mode).0), stop)?;
    for _ in 0..sched.intervals.len() {
        checkpoint(stop)?;
        let mut changed = false;
        for &(a, b) in &sched.precedences {
            checkpoint(stop)?;
            if !present[a] || !present[b] {
                continue;
            }
            let need = starts[a].saturating_add(dur[a]);
            if need > starts[b] {
                starts[b] = need;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    checkpoint(stop)?;
    Ok(starts)
}

/// Candidate start times for shifting interval `i`: its precedence-earliest, the
/// ends of other intervals (for resource packing), and 0; clamped to its window.
fn schedule_candidates(
    sched: &Schedule,
    chosen: &[usize],
    dur: &[i64],
    starts: &[i64],
    present: &[bool],
    i: usize,
    stop: &AtomicBool,
) -> Interruptible<Vec<i64>> {
    checkpoint(stop)?;
    let (lo, hi) = iv_start_window(&sched.intervals[i], chosen[i]);
    let mut est = 0i64;
    for &(before, after) in &sched.precedences {
        checkpoint(stop)?;
        if after == i && present[before] && present[after] {
            est = est.max(starts[before] + dur[before]);
        }
    }
    let mut cands = vec![lo, est];
    for (j, &sj) in starts.iter().enumerate() {
        checkpoint(stop)?;
        if j != i && present[j] {
            cands.push(sj + dur[j]);
            cands.push(sj.saturating_sub(dur[i]));
        }
    }
    for c in &mut cands {
        checkpoint(stop)?;
        *c = (*c).clamp(lo, hi);
    }
    sort_dedup_interruptible(&mut cands, stop)?;
    checkpoint(stop)?;
    Ok(cands)
}

fn unknown_solution() -> CollectionSolution {
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
}

fn snapshot_solution(
    sched: &Schedule,
    chosen: &[usize],
    starts: &[i64],
    present: &[bool],
    score: (i64, i64),
    stop: &AtomicBool,
) -> Interruptible<CollectionSolution> {
    checkpoint(stop)?;
    if score.0 != 0 {
        return Ok(unknown_solution());
    }

    let starts = collect_interruptible(starts.iter().copied(), stop)?;
    let presences = collect_interruptible(present.iter().copied(), stop)?;
    let machines = collect_interruptible(
        sched
            .intervals
            .iter()
            .zip(chosen)
            .zip(present)
            .map(|((interval, &mode), &is_present)| if is_present { iv_machine(interval, mode) } else { -1 }),
        stop,
    )?;
    let modes = mode_references(sched, chosen, present, stop)?;
    checkpoint(stop)?;
    let objectives = if sched.minimize_makespan { vec![score.1] } else { Vec::new() };
    checkpoint(stop)?;
    Ok(CollectionSolution { lists: Vec::new(), objectives, feasible: true, starts, presences, machines, modes, bound: None })
}

fn add_state_metrics(total: &mut ScheduleStateMetrics, metrics: ScheduleStateMetrics) {
    total.construction_candidates = total.construction_candidates.saturating_add(metrics.construction_candidates);
    total.construction_bucket_visits = total.construction_bucket_visits.saturating_add(metrics.construction_bucket_visits);
    total.construction_heap_pushes = total.construction_heap_pushes.saturating_add(metrics.construction_heap_pushes);
    total.construction_stale_pops = total.construction_stale_pops.saturating_add(metrics.construction_stale_pops);
    total.construction_heap_rebuilds = total.construction_heap_rebuilds.saturating_add(metrics.construction_heap_rebuilds);
    total.construction_heap_peak = total.construction_heap_peak.max(metrics.construction_heap_peak);
    total.reconstructions = total.reconstructions.saturating_add(metrics.reconstructions);
    total.moves_considered = total.moves_considered.saturating_add(metrics.moves_considered);
    total.moves_accepted = total.moves_accepted.saturating_add(metrics.moves_accepted);
    total.cycle_rejections = total.cycle_rejections.saturating_add(metrics.cycle_rejections);
    total.window_rejections = total.window_rejections.saturating_add(metrics.window_rejections);
    total.objective_rejections = total.objective_rejections.saturating_add(metrics.objective_rejections);
    total.critical_path_updates = total.critical_path_updates.saturating_add(metrics.critical_path_updates);
    total.delta_evaluations = total.delta_evaluations.saturating_add(metrics.delta_evaluations);
    total.full_evaluations = total.full_evaluations.saturating_add(metrics.full_evaluations);
    total.full_fallbacks = total.full_fallbacks.saturating_add(metrics.full_fallbacks);
    total.topological_rebuilds = total.topological_rebuilds.saturating_add(metrics.topological_rebuilds);
    total.oracle_validations = total.oracle_validations.saturating_add(metrics.oracle_validations);
    total.oracle_mismatches = total.oracle_mismatches.saturating_add(metrics.oracle_mismatches);
    total.dirty_cone_operations = total.dirty_cone_operations.saturating_add(metrics.dirty_cone_operations);
    total.max_dirty_cone = total.max_dirty_cone.max(metrics.max_dirty_cone);
    total.workspace_growths = total.workspace_growths.saturating_add(metrics.workspace_growths);
    total.local_move_estimates = total.local_move_estimates.saturating_add(metrics.local_move_estimates);
    total.local_move_certified = total.local_move_certified.saturating_add(metrics.local_move_certified);
    total.local_move_unknown = total.local_move_unknown.saturating_add(metrics.local_move_unknown);
    total.direct_oracle_attempts = total.direct_oracle_attempts.saturating_add(metrics.direct_oracle_attempts);
    total.direct_oracle_accepts = total.direct_oracle_accepts.saturating_add(metrics.direct_oracle_accepts);
    total.direct_oracle_cycles = total.direct_oracle_cycles.saturating_add(metrics.direct_oracle_cycles);
    total.direct_oracle_windows = total.direct_oracle_windows.saturating_add(metrics.direct_oracle_windows);
    total.direct_oracle_objective_rejections =
        total.direct_oracle_objective_rejections.saturating_add(metrics.direct_oracle_objective_rejections);
}

fn job_shop_result(
    best: Option<CollectionSolution>,
    construction_elapsed: Duration,
    first_feasible: Option<Duration>,
    metrics: ScheduleStateMetrics,
    incumbent_improvements: u64,
    incumbent_injections: u64,
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    (
        best.unwrap_or_else(unknown_solution),
        ScheduleConstructionMetrics {
            elapsed: construction_elapsed,
            first_feasible,
            initial_objective: None,
            initial_dispatch_rule: None,
            candidates: metrics.construction_candidates,
            construction_bucket_visits: metrics.construction_bucket_visits,
            construction_heap_pushes: metrics.construction_heap_pushes,
            construction_stale_pops: metrics.construction_stale_pops,
            construction_heap_rebuilds: metrics.construction_heap_rebuilds,
            construction_heap_peak: metrics.construction_heap_peak,
            work_steps: metrics.moves_considered,
            constructor: "giffler-thompson-critical-path",
            moves_considered: metrics.moves_considered,
            moves_accepted: metrics.moves_accepted,
            incumbent_improvements,
            incumbent_injections,
            cycle_rejections: metrics.cycle_rejections,
            window_rejections: metrics.window_rejections,
            objective_rejections: metrics.objective_rejections,
            reconstructions: metrics.reconstructions,
            critical_path_updates: metrics.critical_path_updates,
            delta_evaluations: metrics.delta_evaluations,
            full_evaluations: metrics.full_evaluations,
            full_fallbacks: metrics.full_fallbacks,
            topological_rebuilds: metrics.topological_rebuilds,
            oracle_validations: metrics.oracle_validations,
            oracle_mismatches: metrics.oracle_mismatches,
            dirty_cone_operations: metrics.dirty_cone_operations,
            max_dirty_cone: metrics.max_dirty_cone,
            workspace_growths: metrics.workspace_growths,
            workspace_rollbacks: 0,
            alns_generation_attempts: 0,
            alns_moves_generated: 0,
            resource_profile_checks: 0,
            resource_candidate_scheduling_attempts: 0,
            resource_event_visits: 0,
            resource_peak_profile_events: 0,
            precedence_rejections: 0,
            infeasible_rejections: 0,
            justification_attempts: 0,
            tabu_steps: 0,
            tabu_hits: 0,
            tabu_aspirations: 0,
            tabu_forced_moves: 0,
            session_initializations: 0,
            session_resumes: 0,
            session_rebases: 0,
            island_profile: None,
            island_profile_mask: 0,
            baseline_island_profile_mask: 0,
            scored_island_profile_mask: 0,
            reactive_restarts: 0,
            reactive_restart_dispatches: 0,
            reactive_restart_perturbations: 0,
            reactive_restart_rebuild_failures: 0,
            island_scored_candidates: 0,
            island_shortlisted_candidates: 0,
            approximate_candidates_generated: 0,
            approximate_candidates_refined: metrics.local_move_estimates,
            approximate_candidates_certified: metrics.local_move_certified,
            approximate_candidates_unknown: metrics.local_move_unknown,
            approximation_score_items: 0,
            approximation_sort_items: 0,
            approximation_local_span_items: 0,
            approximation_elapsed: Duration::ZERO,
            approximation_work_units: 0,
            direct_oracle_attempts: metrics.direct_oracle_attempts,
            direct_oracle_accepts: metrics.direct_oracle_accepts,
            direct_oracle_cycles: metrics.direct_oracle_cycles,
            direct_oracle_windows: metrics.direct_oracle_windows,
            direct_oracle_objective_rejections: metrics.direct_oracle_objective_rejections,
            exact_probes_avoided: 0,
            search_elite_snapshot_captures: 0,
            search_elite_snapshot_interruptions: 0,
            search_elite_snapshot_errors: 0,
            search_elite_capture_worker_elapsed_sum: Duration::ZERO,
            search_elite_snapshot_peak_heap_lower_bound_bytes: 0,
            schedule_lns_shadow_owner_worker_mask: 0,
            schedule_lns: ScheduleLnsMetrics::default(),
            schedule_lns_workspace_peak_bytes: 0,
            schedule_path_relink: ScheduleRelinkMetrics::default(),
            schedule_constructor_multistart_owner_worker_mask: 0,
            schedule_constructor_multistart_attempts: 0,
            schedule_constructor_multistart_constructions: 0,
            schedule_constructor_multistart_interruptions: 0,
            schedule_constructor_multistart_failures: 0,
            schedule_constructor_multistart_feasible: 0,
            schedule_constructor_multistart_distinct_fingerprints: 0,
            schedule_constructor_multistart_other_fingerprint_observations: 0,
            schedule_constructor_multistart_initial_objective: None,
            schedule_constructor_multistart_best_objective: None,
            schedule_constructor_multistart_improvements: 0,
            schedule_constructor_multistart_work_units: 0,
            schedule_constructor_multistart_elapsed: Duration::ZERO,
            schedule_constructor_multistart_workspace_peak_bytes: 0,
            schedule_constructor_multistart_best_ordinal: None,
            schedule_constructor_multistart_best_seed: None,
            schedule_constructor_multistart_best_fingerprint: None,
            schedule_constructor_multistart_next_ordinal: None,
            schedule_tsab: ScheduleTsabMetrics::default(),
        },
    )
}

fn generic_schedule_metrics(
    elapsed: Duration,
    first_feasible: Option<Duration>,
    construction_candidates: u64,
    moves_considered: u64,
    moves_accepted: u64,
    incumbent_improvements: u64,
) -> ScheduleConstructionMetrics {
    ScheduleConstructionMetrics {
        elapsed,
        first_feasible,
        initial_objective: None,
        initial_dispatch_rule: None,
        candidates: construction_candidates,
        construction_bucket_visits: 0,
        construction_heap_pushes: 0,
        construction_stale_pops: 0,
        construction_heap_rebuilds: 0,
        construction_heap_peak: 0,
        work_steps: moves_considered,
        constructor: "priority-sgs",
        moves_considered,
        moves_accepted,
        incumbent_improvements,
        incumbent_injections: 0,
        cycle_rejections: 0,
        window_rejections: 0,
        objective_rejections: 0,
        reconstructions: 0,
        critical_path_updates: 0,
        delta_evaluations: 0,
        full_evaluations: 0,
        full_fallbacks: 0,
        topological_rebuilds: 0,
        oracle_validations: 0,
        oracle_mismatches: 0,
        dirty_cone_operations: 0,
        max_dirty_cone: 0,
        workspace_growths: 0,
        workspace_rollbacks: 0,
        alns_generation_attempts: 0,
        alns_moves_generated: 0,
        resource_profile_checks: 0,
        resource_candidate_scheduling_attempts: 0,
        resource_event_visits: 0,
        resource_peak_profile_events: 0,
        precedence_rejections: 0,
        infeasible_rejections: 0,
        justification_attempts: 0,
        tabu_steps: 0,
        tabu_hits: 0,
        tabu_aspirations: 0,
        tabu_forced_moves: 0,
        session_initializations: 0,
        session_resumes: 0,
        session_rebases: 0,
        island_profile: None,
        island_profile_mask: 0,
        baseline_island_profile_mask: 0,
        scored_island_profile_mask: 0,
        reactive_restarts: 0,
        reactive_restart_dispatches: 0,
        reactive_restart_perturbations: 0,
        reactive_restart_rebuild_failures: 0,
        island_scored_candidates: 0,
        island_shortlisted_candidates: 0,
        approximate_candidates_generated: 0,
        approximate_candidates_refined: 0,
        approximate_candidates_certified: 0,
        approximate_candidates_unknown: 0,
        approximation_score_items: 0,
        approximation_sort_items: 0,
        approximation_local_span_items: 0,
        approximation_elapsed: Duration::ZERO,
        approximation_work_units: 0,
        direct_oracle_attempts: 0,
        direct_oracle_accepts: 0,
        direct_oracle_cycles: 0,
        direct_oracle_windows: 0,
        direct_oracle_objective_rejections: 0,
        exact_probes_avoided: 0,
        search_elite_snapshot_captures: 0,
        search_elite_snapshot_interruptions: 0,
        search_elite_snapshot_errors: 0,
        search_elite_capture_worker_elapsed_sum: Duration::ZERO,
        search_elite_snapshot_peak_heap_lower_bound_bytes: 0,
        schedule_lns_shadow_owner_worker_mask: 0,
        schedule_lns: ScheduleLnsMetrics::default(),
        schedule_lns_workspace_peak_bytes: 0,
        schedule_path_relink: ScheduleRelinkMetrics::default(),
        schedule_constructor_multistart_owner_worker_mask: 0,
        schedule_constructor_multistart_attempts: 0,
        schedule_constructor_multistart_constructions: 0,
        schedule_constructor_multistart_interruptions: 0,
        schedule_constructor_multistart_failures: 0,
        schedule_constructor_multistart_feasible: 0,
        schedule_constructor_multistart_distinct_fingerprints: 0,
        schedule_constructor_multistart_other_fingerprint_observations: 0,
        schedule_constructor_multistart_initial_objective: None,
        schedule_constructor_multistart_best_objective: None,
        schedule_constructor_multistart_improvements: 0,
        schedule_constructor_multistart_work_units: 0,
        schedule_constructor_multistart_elapsed: Duration::ZERO,
        schedule_constructor_multistart_workspace_peak_bytes: 0,
        schedule_constructor_multistart_best_ordinal: None,
        schedule_constructor_multistart_best_seed: None,
        schedule_constructor_multistart_best_fingerprint: None,
        schedule_constructor_multistart_next_ordinal: None,
        schedule_tsab: ScheduleTsabMetrics::default(),
    }
}

fn retain_job_shop_incumbent(
    state: &JobShopState,
    best_objective: &mut Option<i64>,
    best: &mut Option<CollectionSolution>,
    first_feasible: &mut Option<Duration>,
    started: Instant,
    report: &mut dyn FnMut(i64),
) -> bool {
    let objective = state.makespan();
    if best_objective.is_none_or(|current| objective < current) {
        *best_objective = Some(objective);
        *best = Some(state.to_solution());
        first_feasible.get_or_insert_with(|| started.elapsed());
        report(objective);
        true
    } else {
        false
    }
}

fn complete_schedule_incumbent(sched: &Schedule, incumbent: Option<&CollectionSolution>) -> Option<CollectionSolution> {
    incumbent
        .filter(|solution| {
            solution.feasible
                && solution.starts.len() == sched.intervals.len()
                && solution.presences.len() == sched.intervals.len()
                && solution.machines.len() == sched.intervals.len()
                && solution.modes.len() == sched.intervals.len()
                && (!sched.minimize_makespan || solution.objectives.len() == 1)
        })
        .cloned()
}

/// Structured search for mandatory fixed-assignment job shops. Recognition is
/// deliberately conservative; every unsupported schedule remains on the
/// generic SGS and interval-move fallback below.
fn solve_job_shop(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    max_iterations: u64,
    repeat_until_stopped: bool,
    initial_incumbent: Option<&CollectionSolution>,
    report: &mut dyn FnMut(i64),
) -> Option<(CollectionSolution, ScheduleConstructionMetrics)> {
    let started = Instant::now();
    let problem = match JobShopProblem::recognize(sched, stop) {
        Ok(Some(problem)) => problem,
        Ok(None) => return None,
        Err(_) => {
            return Some(job_shop_result(None, started.elapsed(), None, ScheduleStateMetrics::default(), 0, 0));
        }
    };
    let operation_count = problem.operation_count();
    let max_attempts = (operation_count / 8).clamp(8, 32);
    let default_moves = u64::try_from(operation_count).unwrap_or(u64::MAX).saturating_mul(256).clamp(2_048, 100_000);
    let max_moves = if max_iterations != u64::MAX {
        max_iterations
    } else if repeat_until_stopped {
        u64::MAX
    } else {
        default_moves
    };
    let mut construction_elapsed = Duration::ZERO;
    let mut total = ScheduleStateMetrics::default();
    let mut best = complete_schedule_incumbent(sched, initial_incumbent);
    let mut best_objective = best.as_ref().and_then(|solution| solution.objectives.first().copied());
    let mut first_feasible = None;
    let mut incumbent_improvements = 0u64;
    let mut incumbent_injections = 0u64;
    let mut moves_used = 0u64;
    let mut attempt = 0usize;
    let mut injected_machine_sequences = best.as_ref().map(|incumbent| {
        let mut sequences = vec![Vec::new(); problem.machine_count()];
        for operation in 0..problem.operation_count() {
            sequences[problem.machine(operation)].push(operation);
        }
        for sequence in &mut sequences {
            sequence.sort_by_key(|&operation| (incumbent.starts[operation], operation));
        }
        sequences
    });

    'attempts: loop {
        if checkpoint(stop).is_err() || moves_used >= max_moves || (attempt >= max_attempts && !repeat_until_stopped) {
            break;
        }
        let current_attempt = attempt;
        attempt = attempt.saturating_add(1);
        let rule = DispatchRule::LEGACY_MULTISTART.get(current_attempt).copied().unwrap_or(DispatchRule::Randomized);
        let attempt_seed = mix64(seed ^ u64::try_from(current_attempt).unwrap_or(u64::MAX).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let construction_started = Instant::now();
        let injection_attempt = injected_machine_sequences.is_some();
        let (constructed, partial_metrics) = if let Some(machine_sequences) = injected_machine_sequences.take() {
            (JobShopState::from_machine_sequences(&problem, machine_sequences, stop), ScheduleStateMetrics::default())
        } else {
            JobShopState::giffler_thompson_profiled(&problem, attempt_seed, rule, stop)
        };
        let mut state = match constructed {
            Ok(Some(state)) => {
                incumbent_injections = incumbent_injections.saturating_add(u64::from(injection_attempt));
                state
            }
            Ok(None) => {
                construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
                add_state_metrics(&mut total, partial_metrics);
                continue;
            }
            Err(_) => {
                construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
                add_state_metrics(&mut total, partial_metrics);
                break;
            }
        };
        construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_job_shop_incumbent(
            &state,
            &mut best_objective,
            &mut best,
            &mut first_feasible,
            started,
            report,
        )));
        if checkpoint(stop).is_err() {
            add_state_metrics(&mut total, state.metrics());
            break;
        }

        let neighborhoods = match current_attempt % 3 {
            0 => [CriticalNeighborhood::N5, CriticalNeighborhood::N1, CriticalNeighborhood::N6],
            1 => [CriticalNeighborhood::N6, CriticalNeighborhood::N5, CriticalNeighborhood::N1],
            _ => [CriticalNeighborhood::N1, CriticalNeighborhood::N6, CriticalNeighborhood::N5],
        };
        let mut kicks = 0u64;
        let move_capacity = operation_count.saturating_mul(3);
        let mut movements = Vec::with_capacity(move_capacity);
        loop {
            if checkpoint(stop).is_err() || moves_used >= max_moves {
                add_state_metrics(&mut total, state.metrics());
                break 'attempts;
            }
            let mut accepted = false;
            if state.fill_critical_move_union(&neighborhoods, &mut movements, stop).is_err() {
                add_state_metrics(&mut total, state.metrics());
                break 'attempts;
            }
            if !movements.is_empty() {
                let offset = usize::try_from(mix64(attempt_seed ^ moves_used)).unwrap_or(usize::MAX) % movements.len();
                movements.rotate_left(offset);
            }
            for &movement in &movements {
                if checkpoint(stop).is_err() || moves_used >= max_moves {
                    add_state_metrics(&mut total, state.metrics());
                    break 'attempts;
                }
                moves_used = moves_used.saturating_add(1);
                match state.consider_move(movement, MinimizingMoveAcceptance::Improving, stop) {
                    Ok(MoveOutcome::Accepted { .. }) => {
                        accepted = true;
                        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_job_shop_incumbent(
                            &state,
                            &mut best_objective,
                            &mut best,
                            &mut first_feasible,
                            started,
                            report,
                        )));
                        break;
                    }
                    Ok(MoveOutcome::Rejected(_)) => {}
                    Err(_) => {
                        add_state_metrics(&mut total, state.metrics());
                        break 'attempts;
                    }
                }
            }
            if accepted {
                continue;
            }
            if kicks >= 3 {
                break;
            }

            if state.fill_critical_moves(CriticalNeighborhood::N6, &mut movements, stop).is_err() {
                add_state_metrics(&mut total, state.metrics());
                break 'attempts;
            }
            if movements.is_empty() {
                break;
            }
            let offset =
                usize::try_from(mix64(attempt_seed ^ kicks.wrapping_mul(0xd1b5_4a32_d192_ed03))).unwrap_or(usize::MAX) % movements.len();
            movements.rotate_left(offset);
            let mut kicked = false;
            for &movement in &movements {
                if checkpoint(stop).is_err() || moves_used >= max_moves {
                    add_state_metrics(&mut total, state.metrics());
                    break 'attempts;
                }
                moves_used = moves_used.saturating_add(1);
                match state.consider_move(movement, MinimizingMoveAcceptance::Always, stop) {
                    Ok(MoveOutcome::Accepted { .. }) => {
                        kicked = true;
                        kicks = kicks.saturating_add(1);
                        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_job_shop_incumbent(
                            &state,
                            &mut best_objective,
                            &mut best,
                            &mut first_feasible,
                            started,
                            report,
                        )));
                        break;
                    }
                    Ok(MoveOutcome::Rejected(_)) => {}
                    Err(_) => {
                        add_state_metrics(&mut total, state.metrics());
                        break 'attempts;
                    }
                }
            }
            if !kicked {
                break;
            }
        }
        add_state_metrics(&mut total, state.metrics());
    }

    Some(job_shop_result(best, construction_elapsed, first_feasible, total, incumbent_improvements, incumbent_injections))
}

fn add_resource_metrics(total: &mut ResourceScheduleMetrics, metrics: ResourceScheduleMetrics) {
    total.serial_constructions = total.serial_constructions.saturating_add(metrics.serial_constructions);
    total.parallel_constructions = total.parallel_constructions.saturating_add(metrics.parallel_constructions);
    total.reconstructions = total.reconstructions.saturating_add(metrics.reconstructions);
    total.construction_candidates = total.construction_candidates.saturating_add(metrics.construction_candidates);
    total.candidate_scheduling_attempts = total.candidate_scheduling_attempts.saturating_add(metrics.candidate_scheduling_attempts);
    total.profile_checks = total.profile_checks.saturating_add(metrics.profile_checks);
    total.event_visits = total.event_visits.saturating_add(metrics.event_visits);
    total.peak_profile_events = total.peak_profile_events.max(metrics.peak_profile_events);
    total.moves_considered = total.moves_considered.saturating_add(metrics.moves_considered);
    total.moves_accepted = total.moves_accepted.saturating_add(metrics.moves_accepted);
    total.precedence_rejections = total.precedence_rejections.saturating_add(metrics.precedence_rejections);
    total.infeasible_rejections = total.infeasible_rejections.saturating_add(metrics.infeasible_rejections);
    total.objective_rejections = total.objective_rejections.saturating_add(metrics.objective_rejections);
    total.left_justifications = total.left_justifications.saturating_add(metrics.left_justifications);
    total.right_justifications = total.right_justifications.saturating_add(metrics.right_justifications);
    total.double_justifications = total.double_justifications.saturating_add(metrics.double_justifications);
    total.delta_evaluations = total.delta_evaluations.saturating_add(metrics.delta_evaluations);
    total.full_workspace_evaluations = total.full_workspace_evaluations.saturating_add(metrics.full_workspace_evaluations);
    total.delta_activities_rescheduled = total.delta_activities_rescheduled.saturating_add(metrics.delta_activities_rescheduled);
    total.workspace_rollbacks = total.workspace_rollbacks.saturating_add(metrics.workspace_rollbacks);
    total.oracle_validations = total.oracle_validations.saturating_add(metrics.oracle_validations);
    total.oracle_mismatches = total.oracle_mismatches.saturating_add(metrics.oracle_mismatches);
    total.alns_generation_attempts = total.alns_generation_attempts.saturating_add(metrics.alns_generation_attempts);
    total.alns_moves_generated = total.alns_moves_generated.saturating_add(metrics.alns_moves_generated);
    total.workspace_growths = total.workspace_growths.saturating_add(metrics.workspace_growths);
}

fn retain_resource_incumbent(
    state: &ResourceScheduleState,
    best_objective: &mut Option<i64>,
    best: &mut Option<CollectionSolution>,
    first_feasible: &mut Option<Duration>,
    started: Instant,
    report: &mut dyn FnMut(i64),
) -> bool {
    let objective = state.makespan();
    if best_objective.is_none_or(|current| objective < current) {
        *best_objective = Some(objective);
        *best = Some(state.to_solution());
        first_feasible.get_or_insert_with(|| started.elapsed());
        report(objective);
        true
    } else {
        false
    }
}

#[derive(Default)]
struct ResourceSearchSummary {
    metrics: ResourceScheduleMetrics,
    steps: u64,
    incumbent_improvements: u64,
    incumbent_injections: u64,
    justification_attempts: u64,
}

fn resource_schedule_result(
    best: Option<CollectionSolution>,
    construction_elapsed: Duration,
    first_feasible: Option<Duration>,
    summary: ResourceSearchSummary,
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    let ResourceSearchSummary { metrics, steps, incumbent_improvements, incumbent_injections, justification_attempts } = summary;
    (
        best.unwrap_or_else(unknown_solution),
        ScheduleConstructionMetrics {
            elapsed: construction_elapsed,
            first_feasible,
            initial_objective: None,
            initial_dispatch_rule: None,
            candidates: metrics.construction_candidates,
            construction_bucket_visits: 0,
            construction_heap_pushes: 0,
            construction_stale_pops: 0,
            construction_heap_rebuilds: 0,
            construction_heap_peak: 0,
            work_steps: steps,
            constructor: "resource-priority-sgs",
            moves_considered: metrics.moves_considered,
            moves_accepted: metrics.moves_accepted,
            incumbent_improvements,
            incumbent_injections,
            cycle_rejections: 0,
            window_rejections: 0,
            objective_rejections: metrics.objective_rejections,
            reconstructions: metrics.reconstructions,
            critical_path_updates: 0,
            delta_evaluations: metrics.delta_evaluations,
            full_evaluations: metrics.full_workspace_evaluations.saturating_add(metrics.oracle_validations),
            full_fallbacks: metrics.full_workspace_evaluations,
            topological_rebuilds: 0,
            oracle_validations: metrics.oracle_validations,
            oracle_mismatches: metrics.oracle_mismatches,
            dirty_cone_operations: metrics.delta_activities_rescheduled,
            max_dirty_cone: 0,
            workspace_growths: metrics.workspace_growths,
            workspace_rollbacks: metrics.workspace_rollbacks,
            alns_generation_attempts: metrics.alns_generation_attempts,
            alns_moves_generated: metrics.alns_moves_generated,
            resource_profile_checks: metrics.profile_checks,
            resource_candidate_scheduling_attempts: metrics.candidate_scheduling_attempts,
            resource_event_visits: metrics.event_visits,
            resource_peak_profile_events: metrics.peak_profile_events,
            precedence_rejections: metrics.precedence_rejections,
            infeasible_rejections: metrics.infeasible_rejections,
            justification_attempts,
            tabu_steps: 0,
            tabu_hits: 0,
            tabu_aspirations: 0,
            tabu_forced_moves: 0,
            session_initializations: 0,
            session_resumes: 0,
            session_rebases: 0,
            island_profile: None,
            island_profile_mask: 0,
            baseline_island_profile_mask: 0,
            scored_island_profile_mask: 0,
            reactive_restarts: 0,
            reactive_restart_dispatches: 0,
            reactive_restart_perturbations: 0,
            reactive_restart_rebuild_failures: 0,
            island_scored_candidates: 0,
            island_shortlisted_candidates: 0,
            approximate_candidates_generated: 0,
            approximate_candidates_refined: 0,
            approximate_candidates_certified: 0,
            approximate_candidates_unknown: 0,
            approximation_score_items: 0,
            approximation_sort_items: 0,
            approximation_local_span_items: 0,
            approximation_elapsed: Duration::ZERO,
            approximation_work_units: 0,
            direct_oracle_attempts: 0,
            direct_oracle_accepts: 0,
            direct_oracle_cycles: 0,
            direct_oracle_windows: 0,
            direct_oracle_objective_rejections: 0,
            exact_probes_avoided: 0,
            search_elite_snapshot_captures: 0,
            search_elite_snapshot_interruptions: 0,
            search_elite_snapshot_errors: 0,
            search_elite_capture_worker_elapsed_sum: Duration::ZERO,
            search_elite_snapshot_peak_heap_lower_bound_bytes: 0,
            schedule_lns_shadow_owner_worker_mask: 0,
            schedule_lns: ScheduleLnsMetrics::default(),
            schedule_lns_workspace_peak_bytes: 0,
            schedule_path_relink: ScheduleRelinkMetrics::default(),
            schedule_constructor_multistart_owner_worker_mask: 0,
            schedule_constructor_multistart_attempts: 0,
            schedule_constructor_multistart_constructions: 0,
            schedule_constructor_multistart_interruptions: 0,
            schedule_constructor_multistart_failures: 0,
            schedule_constructor_multistart_feasible: 0,
            schedule_constructor_multistart_distinct_fingerprints: 0,
            schedule_constructor_multistart_other_fingerprint_observations: 0,
            schedule_constructor_multistart_initial_objective: None,
            schedule_constructor_multistart_best_objective: None,
            schedule_constructor_multistart_improvements: 0,
            schedule_constructor_multistart_work_units: 0,
            schedule_constructor_multistart_elapsed: Duration::ZERO,
            schedule_constructor_multistart_workspace_peak_bytes: 0,
            schedule_constructor_multistart_best_ordinal: None,
            schedule_constructor_multistart_best_seed: None,
            schedule_constructor_multistart_best_fingerprint: None,
            schedule_constructor_multistart_next_ordinal: None,
            schedule_tsab: ScheduleTsabMetrics::default(),
        },
    )
}

/// Structured search for mandatory fixed-duration RCPSP schedules. Every
/// priority move is decoded into a complete schedule before it can replace the
/// incumbent. Unsupported interval shapes continue through the generic path.
fn solve_resource_schedule(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    max_iterations: u64,
    repeat_until_stopped: bool,
    initial_incumbent: Option<&CollectionSolution>,
    report: &mut dyn FnMut(i64),
) -> Option<(CollectionSolution, ScheduleConstructionMetrics)> {
    let started = Instant::now();
    let problem = match ResourceScheduleProblem::recognize(sched, stop) {
        Ok(Some(problem)) => problem,
        Ok(None) => return None,
        Err(_) => {
            return Some(resource_schedule_result(None, started.elapsed(), None, ResourceSearchSummary::default()));
        }
    };
    let activity_count = problem.activity_count();
    let default_steps = u64::try_from(activity_count).unwrap_or(u64::MAX).saturating_mul(256).clamp(2_048, 100_000);
    let internal_limit = if max_iterations != u64::MAX {
        max_iterations
    } else if repeat_until_stopped {
        u64::MAX
    } else {
        default_steps
    };
    let max_attempts = PriorityRule::ALL.len().saturating_mul(2).saturating_add(6);
    let movement_batch = activity_count.saturating_mul(8).clamp(32, 1_024);
    let mut construction_elapsed = Duration::ZERO;
    let mut total = ResourceScheduleMetrics::default();
    let mut best = complete_schedule_incumbent(sched, initial_incumbent);
    let mut best_objective = best.as_ref().and_then(|solution| solution.objectives.first().copied());
    let mut first_feasible = None;
    let mut search_steps = 0u64;
    let mut incumbent_improvements = 0u64;
    let mut incumbent_injections = 0u64;
    let mut justification_attempts = 0u64;
    let mut attempt = 0usize;
    let mut injected_priority = best.as_ref().map(|incumbent| {
        let mut order = (0..problem.activity_count()).collect::<Vec<_>>();
        order.sort_by_key(|&activity| (incumbent.starts[activity], activity));
        order
    });

    'attempts: loop {
        if checkpoint(stop).is_err() || search_steps >= internal_limit || (attempt >= max_attempts && !repeat_until_stopped) {
            break;
        }
        let current_attempt = attempt;
        attempt = attempt.saturating_add(1);
        let rule = PriorityRule::ALL.get(current_attempt % PriorityRule::ALL.len()).copied().unwrap_or(PriorityRule::Randomized);
        let attempt_seed = mix64(seed ^ u64::try_from(current_attempt).unwrap_or(u64::MAX).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let construction_started = Instant::now();
        let injection_attempt = injected_priority.is_some();
        let priority_result = if let Some(order) = injected_priority.take() {
            PrioritySgs::compile(&problem, order, stop)
        } else {
            PrioritySgs::dispatch(&problem, attempt_seed, rule, stop)
        };
        let priority = match priority_result {
            Ok(Some(priority)) => priority,
            Ok(None) => {
                construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
                continue;
            }
            Err(_) => break,
        };
        let scheme = if current_attempt.is_multiple_of(2) { GenerationScheme::Serial } else { GenerationScheme::Parallel };
        let mut state = match ResourceScheduleState::construct(&problem, priority, scheme, stop) {
            Ok(Some(state)) => {
                incumbent_injections = incumbent_injections.saturating_add(u64::from(injection_attempt));
                state
            }
            Ok(None) => {
                construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
                continue;
            }
            Err(_) => break,
        };
        construction_elapsed = construction_elapsed.saturating_add(construction_started.elapsed());
        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
            &state,
            &mut best_objective,
            &mut best,
            &mut first_feasible,
            started,
            report,
        )));

        let justifications = match current_attempt % 3 {
            0 => [Justification::Double, Justification::Left, Justification::Right],
            1 => [Justification::Right, Justification::Double, Justification::Left],
            _ => [Justification::Left, Justification::Right, Justification::Double],
        };
        let mut next_justification = 0usize;
        let mut kicks = 0u64;
        let alns_budget = ResourceAlnsBudget::bounded(activity_count);
        let mut movements = Vec::with_capacity(movement_batch);
        let mut alns_movements = Vec::with_capacity(alns_budget.max_moves);
        let mut stop_all = false;
        loop {
            if checkpoint(stop).is_err() || search_steps >= internal_limit {
                stop_all = true;
                break;
            }
            let neighborhood_offset = usize::try_from(mix64(attempt_seed ^ search_steps)).unwrap_or(usize::MAX);
            if state.fill_bounded_moves_from(neighborhood_offset, movement_batch, &mut movements, stop).is_err() {
                stop_all = true;
                break;
            }
            let mut accepted = false;
            for &movement in &movements {
                if checkpoint(stop).is_err() || search_steps >= internal_limit {
                    stop_all = true;
                    break;
                }
                search_steps = search_steps.saturating_add(1);
                match state.consider_move(movement, MinimizingMoveAcceptance::Improving, stop) {
                    Ok(ResourceMoveOutcome::Accepted { .. }) => {
                        accepted = true;
                        next_justification = 0;
                        incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                            &state,
                            &mut best_objective,
                            &mut best,
                            &mut first_feasible,
                            started,
                            report,
                        )));
                        break;
                    }
                    Ok(ResourceMoveOutcome::Rejected(_)) => {}
                    Err(_) => {
                        stop_all = true;
                        break;
                    }
                }
            }
            if stop_all {
                break;
            }
            if accepted {
                continue;
            }

            if next_justification < justifications.len() && search_steps < internal_limit {
                let kind = justifications[next_justification];
                next_justification += 1;
                search_steps = search_steps.saturating_add(1);
                justification_attempts = justification_attempts.saturating_add(1);
                match state.justify(kind, stop) {
                    Ok(Some(outcome)) => {
                        if outcome.changed {
                            incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                                &state,
                                &mut best_objective,
                                &mut best,
                                &mut first_feasible,
                                started,
                                report,
                            )));
                            continue;
                        }
                    }
                    Ok(None) => {}
                    Err(_) => {
                        stop_all = true;
                        break;
                    }
                }
            }

            if kicks < 2 && search_steps < internal_limit {
                let alns_seed = mix64(attempt_seed ^ search_steps ^ kicks.wrapping_mul(0xa076_1d64_78bd_642f));
                if state.fill_alns_segment_moves(alns_seed, alns_budget, &mut alns_movements, stop).is_err() {
                    stop_all = true;
                    break;
                }
                for &movement in &alns_movements {
                    if checkpoint(stop).is_err() || search_steps >= internal_limit {
                        stop_all = true;
                        break;
                    }
                    search_steps = search_steps.saturating_add(1);
                    match state.consider_move(movement, MinimizingMoveAcceptance::Improving, stop) {
                        Ok(ResourceMoveOutcome::Accepted { .. }) => {
                            accepted = true;
                            next_justification = 0;
                            incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                                &state,
                                &mut best_objective,
                                &mut best,
                                &mut first_feasible,
                                started,
                                report,
                            )));
                            break;
                        }
                        Ok(ResourceMoveOutcome::Rejected(_)) => {}
                        Err(_) => {
                            stop_all = true;
                            break;
                        }
                    }
                }
                if stop_all {
                    break;
                }
                if accepted {
                    continue;
                }
                if let Some(&movement) = alns_movements.first() {
                    if search_steps >= internal_limit {
                        break;
                    }
                    search_steps = search_steps.saturating_add(1);
                    match state.consider_move(movement, MinimizingMoveAcceptance::Always, stop) {
                        Ok(ResourceMoveOutcome::Accepted { .. }) => {
                            kicks = kicks.saturating_add(1);
                            next_justification = 0;
                            incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                                &state,
                                &mut best_objective,
                                &mut best,
                                &mut first_feasible,
                                started,
                                report,
                            )));
                            continue;
                        }
                        Ok(ResourceMoveOutcome::Rejected(_)) => {}
                        Err(_) => {
                            stop_all = true;
                            break;
                        }
                    }
                }
            }

            if kicks >= 2 || search_steps >= internal_limit {
                break;
            }
            let kick_offset = usize::try_from(mix64(attempt_seed ^ kicks.wrapping_mul(0xd1b5_4a32_d192_ed03))).unwrap_or(usize::MAX);
            if state.fill_bounded_moves_from(kick_offset, movement_batch, &mut movements, stop).is_err() {
                stop_all = true;
                break;
            }
            if movements.is_empty() {
                break;
            }
            search_steps = search_steps.saturating_add(1);
            match state.consider_move(movements[0], MinimizingMoveAcceptance::Always, stop) {
                Ok(ResourceMoveOutcome::Accepted { .. }) => {
                    kicks = kicks.saturating_add(1);
                    next_justification = 0;
                    incumbent_improvements = incumbent_improvements.saturating_add(u64::from(retain_resource_incumbent(
                        &state,
                        &mut best_objective,
                        &mut best,
                        &mut first_feasible,
                        started,
                        report,
                    )));
                }
                Ok(ResourceMoveOutcome::Rejected(_)) => break,
                Err(_) => {
                    stop_all = true;
                    break;
                }
            }
        }
        add_resource_metrics(&mut total, state.metrics());
        if stop_all {
            break 'attempts;
        }
    }

    Some(resource_schedule_result(
        best,
        construction_elapsed,
        first_feasible,
        ResourceSearchSummary { metrics: total, steps: search_steps, incumbent_improvements, incumbent_injections, justification_attempts },
    ))
}

fn complete_schedule_incumbent_ref<'a>(sched: &Schedule, incumbent: Option<&'a CollectionSolution>) -> Option<&'a CollectionSolution> {
    incumbent.filter(|solution| {
        solution.feasible
            && solution.starts.len() == sched.intervals.len()
            && solution.presences.len() == sched.intervals.len()
            && solution.machines.len() == sched.intervals.len()
            && solution.modes.len() == sched.intervals.len()
            && (!sched.minimize_makespan || solution.objectives.len() == 1)
    })
}

fn incumbent_machine_sequences(
    operation_count: usize,
    machine_count: usize,
    mut machine: impl FnMut(usize) -> usize,
    incumbent: &CollectionSolution,
    stop: &AtomicBool,
) -> Interruptible<Vec<Vec<usize>>> {
    checkpoint(stop)?;
    let mut sequences = vec![Vec::new(); machine_count];
    for operation in 0..operation_count {
        checkpoint(stop)?;
        sequences[machine(operation)].push(operation);
    }
    for sequence in &mut sequences {
        checkpoint(stop)?;
        sequence.sort_by_key(|&operation| (incumbent.starts[operation], operation));
    }
    checkpoint(stop)?;
    Ok(sequences)
}

fn schedule_move_identity(movement: ScheduleMove) -> u64 {
    match movement {
        ScheduleMove::AdjacentSwap { machine, first_position } => {
            u64::try_from(machine).unwrap_or(u64::MAX).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ u64::try_from(first_position).unwrap_or(u64::MAX).wrapping_mul(0xd1b5_4a32_d192_ed03)
        }
        ScheduleMove::Insert { machine, from, to } => {
            u64::try_from(machine).unwrap_or(u64::MAX).wrapping_mul(0x94d0_49bb_1331_11eb)
                ^ u64::try_from(from).unwrap_or(u64::MAX).wrapping_mul(0xbf58_476d_1ce4_e5b9)
                ^ u64::try_from(to).unwrap_or(u64::MAX).wrapping_mul(0x369d_ea0f_31a5_3f85)
        }
    }
}

fn deterministic_sort_work_units(len: usize) -> u64 {
    if len < 2 {
        return 0;
    }
    let levels = usize::BITS - (len - 1).leading_zeros();
    u64::try_from(len).unwrap_or(u64::MAX).saturating_mul(u64::from(levels))
}

fn candidate_tie_key(session: &ScheduleSearchSession, candidate: ProbedScheduleCandidate) -> (u64, ScheduleMove) {
    (
        mix64(
            session.seed
                ^ session.profile.tie_break_salt
                ^ session.scan_epoch.wrapping_mul(0xa076_1d64_78bd_642f)
                ^ schedule_move_identity(candidate.movement),
        ),
        candidate.movement,
    )
}

fn candidate_better(session: &ScheduleSearchSession, candidate: ProbedScheduleCandidate, incumbent: ProbedScheduleCandidate) -> bool {
    if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate {
        return (candidate.objective, candidate.rank, candidate.movement) < (incumbent.objective, incumbent.rank, incumbent.movement);
    }
    if session.profile.uses_baseline_kernel() {
        return (candidate.objective, candidate.movement) < (incumbent.objective, incumbent.movement);
    }
    candidate.objective < incumbent.objective
        || (candidate.objective == incumbent.objective && candidate_tie_key(session, candidate) < candidate_tie_key(session, incumbent))
}

fn tabu_aspiration(tabu: bool, candidate: i64, elite_objective: i64) -> bool {
    tabu && candidate < elite_objective
}

fn select_scanned_candidate(
    strategy: ScheduleJsspSearchStrategy,
    best_admissible: Option<ProbedScheduleCandidate>,
    best_any: Option<ProbedScheduleCandidate>,
) -> (Option<ProbedScheduleCandidate>, bool, bool) {
    if strategy == ScheduleJsspSearchStrategy::TsabCandidate {
        return (best_admissible, false, best_admissible.is_none() && best_any.is_some());
    }
    let forced = best_admissible.is_none() && best_any.is_some();
    (best_admissible.or(best_any), forced, false)
}

fn direct_oracle_slot_split(oracle_quota: usize, available_relink: usize) -> (usize, usize) {
    let relink = available_relink.min(RELINK_ORACLE_CAPACITY).min(oracle_quota.saturating_sub(1));
    (relink, oracle_quota.saturating_sub(relink))
}

#[cfg(test)]
pub(crate) fn audit_direct_oracle_slot_split(oracle_quota: usize, available_relink: usize) -> (usize, usize) {
    direct_oracle_slot_split(oracle_quota, available_relink)
}

fn fill_direct_oracle_slots(
    movements: &mut Vec<ScheduleMove>,
    relink_guide_arc_gain: &mut [u8; RELINK_ORACLE_CAPACITY],
    relink_candidates: [Option<ScheduleRelinkCandidate>; RELINK_ORACLE_CAPACITY],
    normal_movements: impl IntoIterator<Item = ScheduleMove>,
    oracle_quota: usize,
) -> (usize, usize) {
    movements.clear();
    relink_guide_arc_gain.fill(0);
    let available_relink = relink_candidates.iter().flatten().count();
    let (relink_slots, normal_slots) = direct_oracle_slot_split(oracle_quota, available_relink);
    for (index, candidate) in relink_candidates.into_iter().flatten().take(relink_slots).enumerate() {
        movements.push(candidate.movement);
        relink_guide_arc_gain[index] = candidate.guide_arc_gain;
    }
    let relink_end = movements.len();
    debug_assert_eq!(relink_end, relink_slots);
    let mut normal_selected = 0usize;
    for movement in normal_movements {
        if normal_selected >= normal_slots {
            break;
        }
        if movements.contains(&movement) {
            continue;
        }
        movements.push(movement);
        normal_selected = normal_selected.saturating_add(1);
    }
    (relink_end, normal_selected)
}

#[cfg(test)]
pub(crate) fn audit_direct_oracle_scan_layout() -> (usize, usize, bool) {
    let path = ScheduleRelinkCandidate { movement: ScheduleMove::Insert { machine: 0, from: 0, to: 1 }, guide_arc_gain: 1 };
    let normal = [path.movement, ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 }];
    let mut movements = Vec::new();
    let mut gains = [0; RELINK_ORACLE_CAPACITY];
    let (relink_end, normal_selected) = fill_direct_oracle_slots(&mut movements, &mut gains, [Some(path)], normal, 2);
    (relink_end, movements.len(), normal_selected == 1 && movements.get(1).is_some_and(|&movement| movement != path.movement))
}

#[cfg(test)]
pub(crate) fn audit_tsab_owner_mask(strategy: ScheduleJsspSearchStrategy, workers: usize) -> u64 {
    strategy.owner_mask(workers)
}

#[cfg(test)]
pub(crate) fn audit_tsab_phase_policy() -> (u64, u64, u64, usize) {
    (TSAB_LEGACY_WARMUP_WORK_UNITS, TSAB_BURST_WORK_LIMIT, TSAB_TENURE, TSAB_EXACT_SHORTLIST_CAPACITY)
}

#[cfg(test)]
pub(crate) fn audit_tsab_elite_restart_policy() -> (u64, u64, usize, i64) {
    (TSAB_ELITE_RESTART_N5_WORK_UNITS, TSAB_ELITE_RESTART_WORK_UNITS, TSAB_EXACT_SHORTLIST_CAPACITY, TSAB_KICK_MAX_WORSENING_PPM)
}

#[cfg(test)]
pub(crate) fn audit_tsab_n6_restart_streaming_reference(schedule: &Schedule, seed: u64) -> (bool, usize, usize, usize) {
    let stop = AtomicBool::new(false);
    let Ok(Some(problem)) = JobShopProblem::recognize(schedule, &stop) else {
        return (false, 0, 0, 0);
    };
    let Ok(Some(mut state)) = JobShopState::giffler_thompson(&problem, seed, DispatchRule::EarliestStart, &stop) else {
        return (false, 0, 0, 0);
    };
    state.retain_canonical_critical_blocks_only();
    let mut full = Vec::new();
    if state.fill_scored_canonical_critical_moves(CriticalNeighborhood::N6, &mut full, &stop).is_err() {
        return (false, 0, 0, 0);
    }
    full.retain(|candidate| matches!(candidate.movement, ScheduleMove::Insert { from, to, .. } if from.abs_diff(to) > 1));
    let epoch = 0u64;
    let mut expected: Vec<(bool, u64, ScoredScheduleMove)> = full
        .iter()
        .copied()
        .map(|candidate| {
            let certified = state.certifies_insert_acyclicity(candidate.movement, &stop).unwrap_or(false);
            (certified, mix64(seed ^ epoch.wrapping_mul(0xa076_1d64_78bd_642f) ^ schedule_move_identity(candidate.movement)), candidate)
        })
        .collect();
    expected.sort_unstable_by_key(|&candidate| TsabN6RestartShortlistBuilder::ranking_key(candidate));
    expected.dedup_by_key(|candidate| candidate.2.movement);
    expected.truncate(TSAB_EXACT_SHORTLIST_CAPACITY);

    let mut builder = TsabN6RestartShortlistBuilder::default();
    let mut generated = 0usize;
    let visited = state.visit_scored_canonical_critical_moves(CriticalNeighborhood::N6, &stop, |candidate, _| {
        if matches!(candidate.movement, ScheduleMove::Insert { from, to, .. } if from.abs_diff(to) > 1) {
            generated = generated.saturating_add(1);
            let certified = state.certifies_insert_acyclicity(candidate.movement, &stop)?;
            builder.consider(candidate, certified, seed, epoch);
        }
        Ok(())
    });
    let heap_bytes = builder.heap_lower_bound_bytes();
    let actual = builder.finish();
    (visited.is_ok() && generated == full.len() && actual == expected, generated, actual.len(), heap_bytes)
}

#[cfg(test)]
pub(crate) fn audit_tsab_shortlist_policy() -> (usize, usize, bool) {
    let mut builder = TsabShortlistBuilder::default();
    for index in 0..8usize {
        let candidate = ScoredScheduleMove {
            movement: ScheduleMove::AdjacentSwap { machine: 0, first_position: index },
            score: HeadTailMoveScore {
                max_added_arc_path: i128::try_from(index).unwrap_or(i128::MAX),
                total_added_arc_path: 0,
                critical_arcs_removed: 0,
                tight_arcs_added: 0,
            },
        };
        builder.consider(candidate, index == 6);
        if index == 0 {
            builder.consider(candidate, false);
        }
    }
    let shortlist = builder.finish();
    let unique = shortlist
        .iter()
        .enumerate()
        .all(|(index, candidate)| shortlist[..index].iter().all(|other| other.scored.movement != candidate.scored.movement));
    (shortlist.len(), shortlist.iter().filter(|candidate| candidate.tabu).count(), unique)
}

#[cfg(test)]
pub(crate) fn audit_tsab_selection_policy() -> bool {
    let candidate = ProbedScheduleCandidate {
        movement: ScheduleMove::AdjacentSwap { machine: 0, first_position: 0 },
        arcs: ScheduleMoveArcs::default(),
        objective: 9,
        aspiration: false,
        rank: 0,
    };
    let (tsab_selected, tsab_forced, tsab_reset) =
        select_scanned_candidate(ScheduleJsspSearchStrategy::TsabCandidate, None, Some(candidate));
    let (legacy_selected, legacy_forced, legacy_reset) =
        select_scanned_candidate(ScheduleJsspSearchStrategy::Legacy, None, Some(candidate));
    tsab_selected.is_none()
        && !tsab_forced
        && tsab_reset
        && legacy_selected == Some(candidate)
        && legacy_forced
        && !legacy_reset
        && tabu_aspiration(true, 9, 10)
        && !tabu_aspiration(true, 10, 10)
        && !tabu_aspiration(true, 11, 10)
        && !tabu_aspiration(false, 9, 10)
}

#[cfg(test)]
pub(crate) fn audit_tsab_streaming_reference(schedule: &Schedule, seed: u64) -> bool {
    let stop = AtomicBool::new(false);
    let Ok(Some(problem)) = JobShopProblem::recognize(schedule, &stop) else {
        return false;
    };
    let Ok(Some(mut state)) = JobShopState::giffler_thompson(&problem, seed, DispatchRule::EarliestStart, &stop) else {
        return false;
    };
    state.retain_canonical_critical_blocks_only();
    let mut full = Vec::new();
    if state.fill_scored_canonical_critical_moves(CriticalNeighborhood::N5, &mut full, &stop).is_err()
        || full.len() <= TSAB_EXACT_SHORTLIST_CAPACITY
    {
        return false;
    }
    let marked_arc = full.iter().rev().find_map(|candidate| state.move_arcs(candidate.movement)?.added.into_iter().flatten().next());
    let Some(marked_arc) = marked_arc else {
        return false;
    };
    let is_tabu = |candidate: ScoredScheduleMove| {
        state.move_arcs(candidate.movement).is_some_and(|arcs| arcs.added.into_iter().flatten().any(|arc| arc == marked_arc))
    };

    let mut expected: Vec<TsabRankedCandidate> = full
        .iter()
        .copied()
        .filter(|&candidate| !is_tabu(candidate))
        .take(TSAB_EXACT_SHORTLIST_CAPACITY)
        .map(|scored| TsabRankedCandidate { scored, tabu: false })
        .collect();
    if let Some(scored) = full.iter().copied().find(|&candidate| is_tabu(candidate)) {
        if expected.len() == TSAB_EXACT_SHORTLIST_CAPACITY {
            expected.pop();
        }
        expected.push(TsabRankedCandidate { scored, tabu: true });
    }
    expected.sort_unstable_by_key(|candidate| (TsabShortlistBuilder::ranking_key(candidate.scored), candidate.tabu));

    let mut builder = TsabShortlistBuilder::default();
    let visited = state.visit_scored_canonical_critical_moves(CriticalNeighborhood::N5, &stop, |candidate, _| {
        builder.consider(candidate, is_tabu(candidate));
        Ok(())
    });
    let actual = builder.finish();
    matches!(visited, Ok(count) if count == full.len())
        && actual.len() == TSAB_EXACT_SHORTLIST_CAPACITY
        && actual.iter().filter(|candidate| candidate.tabu).count() == 1
        && actual == expected
}

fn prepare_persistent_scan(
    session: &mut ScheduleSearchSession,
    slice: &mut PersistentScheduleSliceMetrics,
    relink_request: Option<ScheduleRelinkRequest<'_>>,
    stop: &AtomicBool,
) -> Interruptible<bool> {
    let stagnation = session.accepted_step.saturating_sub(session.last_improvement_step);
    if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate {
        let accepted_step = session.accepted_step;
        let tabu_until = &session.tabu_until;
        let mut builder = TsabShortlistBuilder::default();
        let generated = session
            .state()
            .visit_scored_canonical_critical_moves(CriticalNeighborhood::N5, stop, |candidate, arcs| {
                let tabu =
                    arcs.added.into_iter().flatten().any(|arc| tabu_until.get(&arc).is_some_and(|&expiration| accepted_step < expiration));
                builder.consider(candidate, tabu);
                Ok(())
            })
            .map_err(|_| Interrupted)?;
        slice.schedule_tsab.n5_generated = slice.schedule_tsab.n5_generated.saturating_add(u64::try_from(generated).unwrap_or(u64::MAX));
        slice.schedule_tsab.ranked = slice.schedule_tsab.ranked.saturating_add(u64::try_from(generated).unwrap_or(u64::MAX));
        slice.approximate_candidates_generated =
            slice.approximate_candidates_generated.saturating_add(u64::try_from(generated).unwrap_or(u64::MAX));
        slice.approximation_score_items = slice.approximation_score_items.saturating_add(u64::try_from(generated).unwrap_or(u64::MAX));
        // Streaming top-k performs no global sort. Charge one scoring unit and
        // an upper bound of four fixed-shortlist comparisons per generated
        // move, independently of allocator capacity or timing.
        slice.approximation_work_units =
            slice.approximation_work_units.saturating_add(u64::try_from(generated).unwrap_or(u64::MAX).saturating_mul(5));
        if generated == 0 {
            clear_persistent_scan(session);
            return Ok(false);
        }
        let shortlisted = builder.finish();
        session.scan.movements.clear();
        session.scan.tsab_ranks.clear();
        for (rank, candidate) in shortlisted.into_iter().enumerate() {
            session.scan.tsab_ranks.push(rank);
            session.scan.movements.push(candidate.scored.movement);
        }
        session.scan.scored_movements.clear();
        session.scan.cursor = 0;
        session.scan.end = session.scan.movements.len();
        session.scan.best_admissible = None;
        session.scan.best_any = None;
        session.scan_epoch = session.scan_epoch.saturating_add(1);
        let workspace_bytes = session
            .scan
            .movements
            .capacity()
            .saturating_mul(std::mem::size_of::<ScheduleMove>())
            .saturating_add(session.scan.tsab_ranks.capacity().saturating_mul(std::mem::size_of::<usize>()))
            .saturating_add(TSAB_EXACT_SHORTLIST_CAPACITY.saturating_mul(std::mem::size_of::<TsabRankedCandidate>()))
            .saturating_add(TSAB_EXACT_SHORTLIST_CAPACITY.saturating_mul(std::mem::size_of::<ScoredScheduleMove>()));
        slice.schedule_tsab.workspace_peak_bytes = slice.schedule_tsab.workspace_peak_bytes.max(workspace_bytes);
        slice.schedule_tsab.shortlists = slice.schedule_tsab.shortlists.saturating_add(u64::from(session.scan.end != 0));
        slice.island_shortlisted_candidates =
            slice.island_shortlisted_candidates.saturating_add(u64::try_from(session.scan.end).unwrap_or(u64::MAX));
        return Ok(session.scan.end != 0);
    }
    if session.profile.uses_baseline_kernel() {
        let neighborhood = if stagnation >= session.tenure_base.saturating_mul(2) || session.scan_epoch % 4 == 3 {
            CriticalNeighborhood::N6
        } else {
            CriticalNeighborhood::N5
        };
        let ScheduleSearchSession { state, scan, .. } = session;
        state
            .as_mut()
            .expect("search sessions always own a job-shop state")
            .fill_canonical_critical_moves(neighborhood, &mut scan.movements, stop)
            .map_err(|_| Interrupted)?;
        if session.scan.movements.is_empty() {
            session.scan.cursor = 0;
            session.scan.end = 0;
            return Ok(false);
        }
        let offset = usize::try_from(mix64(
            session.seed
                ^ session.accepted_step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ session.scan_epoch.wrapping_mul(0xd1b5_4a32_d192_ed03),
        ))
        .unwrap_or(usize::MAX)
            % session.scan.movements.len();
        session.scan.movements.rotate_left(offset);
        session.scan.cursor = 0;
        session.scan.end = session.scan.movements.len().min(32);
        session.scan.tsab_ranks.clear();
        slice.island_shortlisted_candidates =
            slice.island_shortlisted_candidates.saturating_add(u64::try_from(session.scan.end).unwrap_or(u64::MAX));
        session.scan.best_admissible = None;
        session.scan.best_any = None;
        session.scan_epoch = session.scan_epoch.saturating_add(1);
        return Ok(true);
    }
    debug_assert!(session.profile.uses_direct_oracle());
    let neighborhood = if stagnation >= session.tenure_base.saturating_mul(2)
        || session.scan_epoch % session.profile.n6_period == session.profile.n6_period - 1
    {
        CriticalNeighborhood::N6
    } else {
        CriticalNeighborhood::N5
    };
    let ScheduleSearchSession { state, scan, .. } = session;
    state
        .as_mut()
        .expect("search sessions always own a job-shop state")
        .fill_scored_critical_moves(&[neighborhood], &mut scan.scored_movements, stop)
        .map_err(|_| Interrupted)?;
    if session.scan.scored_movements.is_empty() && relink_request.is_none() {
        session.scan.cursor = 0;
        session.scan.end = 0;
        return Ok(false);
    }
    let scored = session.scan.scored_movements.len();
    slice.island_scored_candidates = slice.island_scored_candidates.saturating_add(u64::try_from(scored).unwrap_or(u64::MAX));
    slice.approximate_candidates_generated =
        slice.approximate_candidates_generated.saturating_add(u64::try_from(scored).unwrap_or(u64::MAX));
    slice.approximation_score_items = slice.approximation_score_items.saturating_add(u64::try_from(scored).unwrap_or(u64::MAX));
    slice.approximation_sort_items = slice.approximation_sort_items.saturating_add(u64::try_from(scored).unwrap_or(u64::MAX));
    slice.approximation_work_units = slice
        .approximation_work_units
        .saturating_add(u64::try_from(scored).unwrap_or(u64::MAX))
        .saturating_add(deterministic_sort_work_units(scored));
    let exploitation = scored.min(12);
    let exploration = scored.saturating_sub(exploitation).min(4);
    let mut shortlist = Vec::with_capacity(exploitation + exploration);
    shortlist.extend(0..exploitation);
    if exploration != 0 {
        let remainder = scored - exploitation;
        for bucket in 0..exploration {
            let bucket_start = bucket.saturating_mul(remainder) / exploration;
            let bucket_end = (bucket + 1).saturating_mul(remainder) / exploration;
            let bucket_width = bucket_end.saturating_sub(bucket_start).max(1);
            let bucket_offset = usize::try_from(mix64(
                session.seed
                    ^ session.profile.tie_break_salt
                    ^ session.accepted_step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ session.scan_epoch.wrapping_mul(0xd1b5_4a32_d192_ed03)
                    ^ session.restart_epoch.wrapping_mul(0x94d0_49bb_1331_11eb)
                    ^ u64::try_from(bucket).unwrap_or(u64::MAX).wrapping_mul(0xe703_7ed1_a0b4_28db),
            ))
            .unwrap_or(usize::MAX)
                % bucket_width;
            let index = exploitation + bucket_start + bucket_offset;
            shortlist.push(index);
        }
    }
    session.scan.estimated_movements.clear();
    for index in shortlist {
        checkpoint(stop)?;
        let scored_candidate = session.scan.scored_movements[index];
        let Some(arcs) = session.state().move_arcs(scored_candidate.movement) else {
            continue;
        };
        let Some(estimate) = session.state_mut().estimate_move_local(scored_candidate.movement, stop).map_err(|_| Interrupted)? else {
            continue;
        };
        slice.approximation_work_units =
            slice.approximation_work_units.saturating_add(u64::try_from(estimate.span_operations).unwrap_or(u64::MAX).saturating_mul(2));
        slice.approximation_local_span_items =
            slice.approximation_local_span_items.saturating_add(u64::try_from(estimate.span_operations).unwrap_or(u64::MAX));
        let tabu = arcs
            .added
            .into_iter()
            .flatten()
            .any(|arc| session.tabu_until.get(&arc).is_some_and(|&expiration| session.accepted_step < expiration));
        let tie = mix64(
            session.seed
                ^ session.profile.tie_break_salt
                ^ session.scan_epoch.wrapping_mul(0xa076_1d64_78bd_642f)
                ^ schedule_move_identity(scored_candidate.movement),
        );
        session.scan.estimated_movements.push(EstimatedScheduleCandidate {
            movement: scored_candidate.movement,
            score: scored_candidate.score,
            estimate,
            tabu,
            tie,
        });
    }
    slice.approximation_work_units =
        slice.approximation_work_units.saturating_add(deterministic_sort_work_units(session.scan.estimated_movements.len()));
    slice.approximation_sort_items =
        slice.approximation_sort_items.saturating_add(u64::try_from(session.scan.estimated_movements.len()).unwrap_or(u64::MAX));
    session.scan.estimated_movements.sort_unstable_by_key(|candidate| {
        (
            candidate.tabu,
            !candidate.estimate.acyclicity_certified,
            candidate.estimate.estimated_makespan,
            candidate.score.max_added_arc_path,
            candidate.score.total_added_arc_path,
            candidate.score.tight_arcs_added,
            Reverse(candidate.score.critical_arcs_removed),
            candidate.tie,
            candidate.movement,
        )
    });
    let oracle_quota = if session.consecutive_direct_oracle_failures == 0 { 2 } else { 4 };
    let mut relink_candidates = [None::<ScheduleRelinkCandidate>; RELINK_ORACLE_CAPACITY];
    if session.profile.id == 6 {
        if let Some(request) = relink_request {
            let mut relink_metrics = ScheduleRelinkMetrics::default();
            let ScheduleSearchSession { state, schedule_relink, .. } = session;
            let prepared = schedule_relink.prepare(
                state.as_mut().expect("search sessions always own a job-shop state"),
                request,
                &mut relink_metrics,
                stop,
            );
            if let Err(error) = prepared {
                match error {
                    ScheduleEliteError::Interrupted => {
                        if relink_metrics.guide_interruptions == 0 {
                            relink_metrics.guide_interruptions = 1;
                        }
                    }
                    _ => {
                        if relink_metrics.guide_incompatible == 0 {
                            relink_metrics.guide_incompatible = 1;
                        }
                    }
                }
            }
            for (slot, candidate) in relink_candidates.iter_mut().zip(session.schedule_relink.shortlist().iter().copied()) {
                *slot = Some(candidate);
            }
            slice.schedule_path_relink.add(relink_metrics);
            if matches!(prepared, Err(ScheduleEliteError::Interrupted)) {
                return Err(Interrupted);
            }
        }
    }
    let (relink_end, normal_selected) = fill_direct_oracle_slots(
        &mut session.scan.movements,
        &mut session.scan.relink_guide_arc_gain,
        relink_candidates,
        session.scan.estimated_movements.iter().map(|candidate| candidate.movement),
        oracle_quota,
    );
    session.scan.relink_end = relink_end;
    slice.exact_probes_avoided = slice
        .exact_probes_avoided
        .saturating_add(u64::try_from(session.scan.estimated_movements.len().saturating_sub(normal_selected)).unwrap_or(u64::MAX));
    session.scan.scored_movements.clear();
    session.scan.estimated_movements.clear();
    session.scan.cursor = 0;
    session.scan.end = session.scan.movements.len();
    session.scan.tsab_ranks.clear();
    slice.island_shortlisted_candidates =
        slice.island_shortlisted_candidates.saturating_add(u64::try_from(session.scan.end).unwrap_or(u64::MAX));
    session.scan.best_admissible = None;
    session.scan.best_any = None;
    session.scan_epoch = session.scan_epoch.saturating_add(1);
    Ok(relink_request.is_none() || session.scan.end != 0)
}

fn arc_tenure(session: &ScheduleSearchSession, arc: MachineArc) -> u64 {
    if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate {
        return TSAB_TENURE;
    }
    let identity = u64::try_from(arc.machine).unwrap_or(u64::MAX).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::try_from(arc.before).unwrap_or(u64::MAX).wrapping_mul(0xd1b5_4a32_d192_ed03)
        ^ u64::try_from(arc.after).unwrap_or(u64::MAX).wrapping_mul(0x94d0_49bb_1331_11eb);
    let width = session.tenure_base.saturating_add(1);
    if session.profile.uses_baseline_kernel() {
        return session.tenure_base.saturating_add(mix64(session.seed ^ session.accepted_step ^ identity) % width);
    }
    session.tenure_base.saturating_add(mix64(session.seed ^ session.profile.tie_break_salt ^ session.accepted_step ^ identity) % width)
}

fn select_tsab_fast_candidate(
    session: &mut ScheduleSearchSession,
    slice: &mut PersistentScheduleSliceMetrics,
    stop: &AtomicBool,
) -> Interruptible<Option<(ScheduleMove, ScheduleMoveArcs)>> {
    let seed = session.seed;
    let accepted_step = session.accepted_step;
    let tabu_until = &session.tabu_until;
    let state = session.state();
    type FastN5Rank = ((i128, (i128, i128, u8, Reverse<u8>)), u64, ScheduleMove);
    let mut best_admissible = None::<(FastN5Rank, ScheduleMoveArcs)>;
    let mut best_any = None::<(FastN5Rank, ScheduleMoveArcs)>;
    let generated = state
        .visit_taillard_scored_strict_n5_canonical_moves(stop, |movement, arcs, score| {
            let tie = mix64(seed ^ accepted_step.wrapping_mul(0xa076_1d64_78bd_642f) ^ schedule_move_identity(movement));
            let ranked = ((score.ranking_key(), tie, movement), arcs);
            if best_any.is_none_or(|current| ranked.0 < current.0) {
                best_any = Some(ranked);
            }
            let tabu =
                arcs.added.into_iter().flatten().any(|arc| tabu_until.get(&arc).is_some_and(|&expiration| accepted_step < expiration));
            if !tabu && best_admissible.is_none_or(|current| ranked.0 < current.0) {
                best_admissible = Some(ranked);
            }
            Ok(())
        })
        .map_err(|_| Interrupted)?;
    slice.schedule_tsab.n5_generated = slice.schedule_tsab.n5_generated.saturating_add(u64::try_from(generated).unwrap_or(u64::MAX));
    slice.schedule_tsab.ranked = slice.schedule_tsab.ranked.saturating_add(u64::try_from(generated).unwrap_or(u64::MAX));
    let selected = if let Some(candidate) = best_admissible {
        Some(candidate)
    } else if let Some(candidate) = best_any {
        session.tabu_until.clear();
        slice.schedule_tsab.tabu_resets = slice.schedule_tsab.tabu_resets.saturating_add(1);
        Some(candidate)
    } else {
        None
    };
    if selected.is_some() {
        slice.schedule_tsab.shortlists = slice.schedule_tsab.shortlists.saturating_add(1);
        slice.schedule_tsab.selections = slice.schedule_tsab.selections.saturating_add(1);
    }
    Ok(selected.map(|((_, _, movement), arcs)| (movement, arcs)))
}

fn collection_solution_heap_lower_bound_bytes(solution: &CollectionSolution) -> usize {
    let lists = solution.lists.capacity().saturating_mul(std::mem::size_of::<Vec<i32>>()).saturating_add(
        solution.lists.iter().map(|list| list.capacity().saturating_mul(std::mem::size_of::<i32>())).fold(0usize, usize::saturating_add),
    );
    lists
        .saturating_add(solution.objectives.capacity().saturating_mul(std::mem::size_of::<i64>()))
        .saturating_add(solution.starts.capacity().saturating_mul(std::mem::size_of::<i64>()))
        .saturating_add(solution.presences.capacity().saturating_add(7) / 8)
        .saturating_add(solution.machines.capacity().saturating_mul(std::mem::size_of::<i64>()))
        .saturating_add(solution.modes.capacity().saturating_mul(std::mem::size_of::<Option<usize>>()))
        .saturating_add(solution.bound.as_ref().map_or(0, |bound| bound.method.capacity()))
}

#[allow(clippy::too_many_arguments)]
fn validate_tsab_fast_checkpoint(
    session: &mut ScheduleSearchSession,
    total: &mut ScheduleStateMetrics,
    slice: &mut PersistentScheduleSliceMetrics,
    stop: &AtomicBool,
    incumbent_improvements: &mut u64,
    search_elite_shadow: bool,
    report: &mut dyn FnMut(i64),
) -> Interruptible<bool> {
    if !session.tsab_fast_enabled || (session.tsab_fast_commits_since_validation == 0 && session.tsab_pending_elite.is_none()) {
        session.tsab_fast_validation_due_on_resume = false;
        return Ok(true);
    }
    // Keep the latch armed across every fallible oracle step. Only a complete
    // validation or a successful restoration of the verified elite clears it.
    session.tsab_fast_validation_due_on_resume = true;
    checkpoint(stop)?;
    #[cfg(test)]
    if std::mem::take(&mut session.tsab_fast_test_stop_during_validation) {
        stop.store(true, Ordering::Release);
    }

    if let Some(pending) = session.tsab_pending_elite.take() {
        let declared = pending.objectives.first().copied().unwrap_or(i64::MAX);
        let rebuilt = match rebuild_session_from_incumbent(session, &pending, stop) {
            Ok(rebuilt) => rebuilt,
            Err(_) => {
                // The oracle rebuild is off to the side, so the live fast
                // trajectory remains atomic. Preserve its sole pending elite
                // for validation after a resumed boundary.
                session.tsab_pending_elite = Some(pending);
                return Err(Interrupted);
            }
        };
        slice.schedule_tsab.fast_full_validations = slice.schedule_tsab.fast_full_validations.saturating_add(1);
        let Some(mut rebuilt) = rebuilt.filter(|state| state.makespan() == declared && declared < session.elite_objective) else {
            slice.schedule_tsab.fast_pending_discards = slice.schedule_tsab.fast_pending_discards.saturating_add(1);
            slice.schedule_tsab.fast_oracle_mismatches = slice.schedule_tsab.fast_oracle_mismatches.saturating_add(1);
            slice.schedule_tsab.fast_disabled = slice.schedule_tsab.fast_disabled.saturating_add(u64::from(session.tsab_fast_enabled));
            session.tsab_fast_enabled = false;
            session.tsab_fast_disabled = true;
            session.tsab_fast_commits_since_validation = 0;
            restore_tsab_verified_elite(session, total, stop)?;
            return Ok(false);
        };
        rebuilt.retain_canonical_critical_blocks_only();
        add_state_metrics(total, session.state_mut().take_metrics());
        session.state = Some(rebuilt);
        session.elite_objective = declared;
        session.elite_solution = session.state().to_solution();
        session.last_improvement_step = session.accepted_step;
        session.last_improvement_probe = session.probe_step;
        session.tsab_fast_commits_since_validation = 0;
        session.tsab_fast_validation_due_on_resume = false;
        session.tabu_until.clear();
        clear_persistent_scan(session);
        slice.schedule_tsab.fast_pending_promotions = slice.schedule_tsab.fast_pending_promotions.saturating_add(1);
        *incumbent_improvements = incumbent_improvements.saturating_add(1);
        report(declared);
        capture_search_elite_shadow(session, slice, search_elite_shadow, stop);
        return Ok(true);
    }

    match session.state_mut().refresh_from_full_oracle(stop) {
        Ok(true) => {
            slice.schedule_tsab.fast_full_validations = slice.schedule_tsab.fast_full_validations.saturating_add(1);
            session.tsab_fast_commits_since_validation = 0;
            session.tsab_fast_validation_due_on_resume = false;
            Ok(true)
        }
        Ok(false) => {
            slice.schedule_tsab.fast_full_validations = slice.schedule_tsab.fast_full_validations.saturating_add(1);
            slice.schedule_tsab.fast_oracle_mismatches = slice.schedule_tsab.fast_oracle_mismatches.saturating_add(1);
            slice.schedule_tsab.fast_disabled = slice.schedule_tsab.fast_disabled.saturating_add(u64::from(session.tsab_fast_enabled));
            session.tsab_fast_enabled = false;
            session.tsab_fast_disabled = true;
            session.tsab_fast_commits_since_validation = 0;
            restore_tsab_verified_elite(session, total, stop)?;
            Ok(false)
        }
        Err(_) => Err(Interrupted),
    }
}

fn restore_tsab_verified_elite(
    session: &mut ScheduleSearchSession,
    total: &mut ScheduleStateMetrics,
    stop: &AtomicBool,
) -> Interruptible<bool> {
    // Set the latch before any fallible work. If rebuilding is interrupted or
    // the supposedly verified sequence cannot be reconstructed exactly, the
    // caller may still publish `elite_solution` but must not retain this
    // session for a future resume.
    session.tsab_fast_hard_failure = true;
    let verified_solution = session.elite_solution.clone();
    let Some(mut verified_state) = rebuild_session_from_incumbent(session, &verified_solution, stop)? else {
        return Ok(false);
    };
    if verified_state.makespan() != session.elite_objective {
        return Ok(false);
    }
    verified_state.retain_canonical_critical_blocks_only();
    add_state_metrics(total, session.state_mut().take_metrics());
    session.state = Some(verified_state);
    session.tsab_pending_elite = None;
    session.tabu_until.clear();
    clear_persistent_scan(session);
    session.tsab_fast_hard_failure = false;
    session.tsab_fast_validation_due_on_resume = false;
    Ok(true)
}

fn clear_persistent_scan(session: &mut ScheduleSearchSession) {
    session.scan.movements.clear();
    session.scan.scored_movements.clear();
    session.scan.estimated_movements.clear();
    session.scan.cursor = 0;
    session.scan.end = 0;
    session.scan.best_admissible = None;
    session.scan.best_any = None;
    session.scan.relink_end = 0;
    session.scan.relink_guide_arc_gain = [0; RELINK_ORACLE_CAPACITY];
    session.scan.tsab_ranks.clear();
}

fn rebuild_session_from_incumbent(
    session: &ScheduleSearchSession,
    incumbent: &CollectionSolution,
    stop: &AtomicBool,
) -> Interruptible<Option<JobShopState>> {
    let sequences = incumbent_machine_sequences(
        session.state().operation_count(),
        session.state().machine_count(),
        |operation| session.state().machine(operation),
        incumbent,
        stop,
    )?;
    session.state().rebuilt_from_machine_sequences(sequences, stop).map_err(|_| Interrupted)
}

fn rebuild_session_from_incumbent_with_policy(
    session: &ScheduleSearchSession,
    incumbent: &CollectionSolution,
    policy: CanonicalPathPolicy,
    stop: &AtomicBool,
) -> Interruptible<Option<JobShopState>> {
    let sequences = incumbent_machine_sequences(
        session.state().operation_count(),
        session.state().machine_count(),
        |operation| session.state().machine(operation),
        incumbent,
        stop,
    )?;
    session.state().rebuilt_from_machine_sequences_with_policy(sequences, policy, stop).map_err(|_| Interrupted)
}

#[allow(clippy::too_many_arguments)]
fn activate_tsab_candidate(
    session: &mut ScheduleSearchSession,
    worker: usize,
    incumbent: &CollectionSolution,
    allow_rebase: bool,
    stop: &AtomicBool,
    total: &mut ScheduleStateMetrics,
    slice: &mut PersistentScheduleSliceMetrics,
    construction_elapsed: &mut Duration,
    incumbent_injections: &mut u64,
    search_elite_shadow: bool,
) -> Interruptible<bool> {
    if !session.requested_strategy.requests_tsab()
        || session.strategy != ScheduleJsspSearchStrategy::Legacy
        || session.tsab_legacy_warmup_work_steps != TSAB_LEGACY_WARMUP_WORK_UNITS
    {
        return Ok(false);
    }
    checkpoint(stop)?;

    // Reconstruct off to the side so interruption cannot leave a half-applied
    // phase transition. Collection incumbents have already crossed semantic
    // replay; this second reconstruction checks the JSSP machine sequences.
    let rebase_started = Instant::now();
    let declared = incumbent.objectives.first().copied().unwrap_or(i64::MAX);
    let path_policy = tsab_canonical_path_policy(session.requested_strategy, worker);
    let rebuilt = if allow_rebase && declared < session.elite_objective {
        rebuild_session_from_incumbent_with_policy(session, incumbent, path_policy, stop)?
            .filter(|rebuilt| rebuilt.makespan() < session.elite_objective)
    } else {
        None
    };
    *construction_elapsed = construction_elapsed.saturating_add(rebase_started.elapsed());
    checkpoint(stop)?;

    if rebuilt.is_none() {
        match session.state_mut().set_canonical_path_policy(path_policy, stop) {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            Err(_) => return Err(Interrupted),
        }
    }

    if let Some(rebuilt) = rebuilt {
        add_state_metrics(total, session.state_mut().take_metrics());
        session.elite_objective = rebuilt.makespan();
        session.elite_solution = rebuilt.to_solution();
        session.state = Some(rebuilt);
        session.last_improvement_step = session.accepted_step;
        session.last_improvement_probe = session.probe_step;
        session.last_restart_probe = session.probe_step;
        slice.session_rebases = slice.session_rebases.saturating_add(1);
        slice.schedule_tsab.activation_rebases = slice.schedule_tsab.activation_rebases.saturating_add(1);
        *incumbent_injections = incumbent_injections.saturating_add(1);
        capture_search_elite_shadow(session, slice, search_elite_shadow, stop);
    }

    session.state_mut().retain_canonical_critical_blocks_only();
    session.tabu_until.clear();
    clear_persistent_scan(session);
    session.direct_oracle_work_credit = 0;
    session.tsab_n5_work_since_restart = 0;
    session.tsab_restart_work_credit = 0;
    session.tsab_restart_epoch = 0;
    session.tsab_fast_enabled = session.state().supports_strict_n5_fast_path();
    session.tsab_fast_disabled = false;
    session.tsab_fast_hard_failure = false;
    session.tsab_fast_validation_due_on_resume = false;
    session.tsab_fast_commits_since_validation = 0;
    session.tsab_pending_elite = None;
    #[cfg(test)]
    {
        session.tsab_fast_test_stop_after_attempts = None;
    }
    session.tenure_base = TSAB_TENURE;
    session.strategy = ScheduleJsspSearchStrategy::TsabCandidate;
    let activation_boundary = session.completed_boundaries;
    let activation_objective = session.state().makespan();
    session.tsab_activation_boundary = Some(activation_boundary);
    slice.schedule_tsab.owner_worker_mask = 1u64.checked_shl(u32::try_from(worker).unwrap_or(u32::MAX)).unwrap_or(0);
    slice.schedule_tsab.burst_work_limit = TSAB_BURST_WORK_LIMIT;
    slice.schedule_tsab.activations = slice.schedule_tsab.activations.saturating_add(1);
    slice.schedule_tsab.activation_boundary = Some(activation_boundary);
    slice.schedule_tsab.activation_objective = Some(activation_objective);
    slice.schedule_tsab.fast_eligible = u64::from(session.state().supports_strict_n5_fast_path());
    slice.schedule_tsab.fast_enabled = u64::from(session.tsab_fast_enabled);
    Ok(true)
}

fn tsab_kick_objective_limit(base: i64) -> i64 {
    if base <= 0 {
        return base;
    }
    let allowance = i128::from(base).saturating_mul(i128::from(TSAB_KICK_MAX_WORSENING_PPM)) / 1_000_000;
    i64::try_from(i128::from(base).saturating_add(allowance).min(i128::from(i64::MAX))).unwrap_or(i64::MAX)
}

fn tsab_kick_acceptance(limit: i64) -> MinimizingMoveAcceptance {
    limit.checked_add(1).map_or(MinimizingMoveAcceptance::Always, MinimizingMoveAcceptance::StrictlyBelow)
}

/// Select a bounded tail corridor from the accepted canonical critical path.
///
/// Canonical blocks are stored from the head of the path to its tail. Walking
/// them in reverse therefore gives a stable tail-to-head order. Each selected
/// N6 insertion is the shortest non-adjacent move toward its block tail. This
/// keeps the compound perturbation bounded before the complete objective gate,
/// and distinct machines make the resulting batch structurally unambiguous.
/// The complete batch oracle remains the sole feasibility authority because
/// individually feasible insertions can still form a cycle together.
fn select_tsab_tail_corridor_n6(state: &JobShopState, stop: &AtomicBool) -> Result<(Vec<ScheduleMove>, usize), Interrupted> {
    checkpoint(stop)?;
    let mut selected = Vec::with_capacity(TSAB_EXACT_SHORTLIST_CAPACITY);
    let mut generated = 0usize;
    for block in state.canonical_critical_blocks().iter().rev() {
        checkpoint(stop)?;
        if selected.len() == TSAB_EXACT_SHORTLIST_CAPACITY {
            break;
        }
        if block.len() < 4
            || selected.iter().any(|movement| matches!(movement, ScheduleMove::Insert { machine, .. } if *machine == block.machine()))
        {
            continue;
        }
        generated = generated.saturating_add(1);
        selected.push(ScheduleMove::Insert { machine: block.machine(), from: block.last_position() - 2, to: block.last_position() });
    }
    Ok((selected, generated))
}

#[cfg(test)]
pub(crate) fn audit_tsab_tail_corridor_selection(state: &JobShopState) -> Vec<ScheduleMove> {
    let stop = AtomicBool::new(false);
    match select_tsab_tail_corridor_n6(state, &stop) {
        Ok((selected, _)) => selected,
        Err(_) => panic!("an unarmed audit stop cannot interrupt"),
    }
}

#[allow(clippy::too_many_arguments)]
fn perform_tsab_elite_n6_restart(
    session: &mut ScheduleSearchSession,
    fresh_incumbent: Option<&CollectionSolution>,
    allow_rebase: bool,
    relink_request: Option<ScheduleRelinkRequest<'_>>,
    stop: &AtomicBool,
    total: &mut ScheduleStateMetrics,
    slice: &mut PersistentScheduleSliceMetrics,
    construction_elapsed: &mut Duration,
    incumbent_improvements: &mut u64,
    incumbent_injections: &mut u64,
    search_elite_shadow: bool,
    report: &mut dyn FnMut(i64),
) -> Interruptible<bool> {
    checkpoint(stop)?;
    debug_assert_eq!(session.strategy, ScheduleJsspSearchStrategy::TsabCandidate);
    debug_assert!(session.scan.end == 0);
    debug_assert!(session.tsab_n5_work_since_restart >= TSAB_ELITE_RESTART_N5_WORK_UNITS);
    debug_assert_eq!(session.tsab_restart_work_credit, TSAB_ELITE_RESTART_WORK_UNITS);
    slice.schedule_tsab.restart_attempts = slice.schedule_tsab.restart_attempts.saturating_add(1);
    slice.schedule_tsab.escape_signals = slice.schedule_tsab.escape_signals.saturating_add(1);

    let rebuild_started = Instant::now();
    let previous_elite_objective = session.elite_objective;
    let mut used_fresh = false;
    let mut scratch = None;
    if allow_rebase {
        if let Some(fresh) = fresh_incumbent {
            let declared = fresh.objectives.first().copied().unwrap_or(i64::MAX);
            if declared < previous_elite_objective {
                if let Some(rebuilt) = rebuild_session_from_incumbent(session, fresh, stop)? {
                    if rebuilt.makespan() < previous_elite_objective {
                        used_fresh = true;
                        scratch = Some(rebuilt);
                    }
                }
            }
        }
    }
    if scratch.is_none() {
        scratch = rebuild_session_from_incumbent(session, &session.elite_solution, stop)?
            .filter(|rebuilt| rebuilt.makespan() == previous_elite_objective);
    }
    *construction_elapsed = construction_elapsed.saturating_add(rebuild_started.elapsed());
    let Some(mut scratch) = scratch else {
        slice.schedule_tsab.restart_rejections = slice.schedule_tsab.restart_rejections.saturating_add(1);
        session.tsab_n5_work_since_restart = 0;
        session.tsab_restart_work_credit = 0;
        return Ok(false);
    };

    scratch.retain_canonical_critical_blocks_only();
    let base_objective = scratch.makespan();
    let base_solution = scratch.to_solution();
    slice.schedule_tsab.restart_best_base_objective =
        Some(slice.schedule_tsab.restart_best_base_objective.map_or(base_objective, |best| best.min(base_objective)));

    let objective_limit = tsab_kick_objective_limit(base_objective);
    let mut restart_probe_count = 0u64;
    let mut committed_kick_moves = 0u64;
    let mut committed_relink_moves = 0u64;
    let mut committed_relink_gain = 0u64;
    let mut relink_plan = SmallVec::<[ScheduleMove; RELINK_MACRO_CAPACITY]>::new();
    let mut relink_objective_limit = base_objective;
    if session.requested_strategy == ScheduleJsspSearchStrategy::TsabMultiCandidate {
        if let Some(request) = relink_request {
            let mut relink_metrics = ScheduleRelinkMetrics::default();
            let prepared = session.schedule_relink.prepare_macro(&mut scratch, request, &mut relink_metrics, stop);
            if prepared.is_ok() {
                relink_plan.extend(session.schedule_relink.shortlist().iter().map(|candidate| candidate.movement));
                if relink_plan.len() >= 2 {
                    relink_objective_limit = base_objective.max(request.guide.objective());
                    let components = u64::try_from(relink_plan.len()).unwrap_or(u64::MAX);
                    slice.schedule_tsab.restart_relink_components_shortlisted =
                        slice.schedule_tsab.restart_relink_components_shortlisted.saturating_add(components);
                    slice.schedule_tsab.restart_relink_guide_arc_gain_shortlisted = slice
                        .schedule_tsab
                        .restart_relink_guide_arc_gain_shortlisted
                        .saturating_add(session.schedule_relink.shortlist_guide_arc_gain());
                } else {
                    relink_plan.clear();
                }
            }
            slice.schedule_path_relink.add(relink_metrics);
            if matches!(prepared, Err(ScheduleEliteError::Interrupted)) {
                add_state_metrics(total, scratch.take_metrics());
                return Err(Interrupted);
            }
        }
    }

    let (tail_corridor, tail_corridor_generated) = if relink_plan.is_empty() {
        match select_tsab_tail_corridor_n6(&scratch, stop) {
            Ok(selection) => selection,
            Err(_) => {
                add_state_metrics(total, scratch.take_metrics());
                return Err(Interrupted);
            }
        }
    } else {
        (Vec::new(), 0)
    };
    let kicked_objective = if !relink_plan.is_empty() {
        slice.schedule_tsab.restart_relink_attempts = slice.schedule_tsab.restart_relink_attempts.saturating_add(1);
        slice.schedule_path_relink.oracle_attempts = slice.schedule_path_relink.oracle_attempts.saturating_add(1);
        match scratch.consider_move_batch_full_oracle(&relink_plan, tsab_kick_acceptance(relink_objective_limit), stop) {
            Ok(MoveOutcome::Accepted { current, .. }) => {
                committed_relink_moves = u64::try_from(relink_plan.len()).unwrap_or(u64::MAX);
                committed_relink_gain = session.schedule_relink.shortlist_guide_arc_gain();
                slice.schedule_path_relink.oracle_accepts = slice.schedule_path_relink.oracle_accepts.saturating_add(1);
                slice.schedule_path_relink.guide_arc_gain_accepted =
                    slice.schedule_path_relink.guide_arc_gain_accepted.saturating_add(committed_relink_gain);
                Some(current)
            }
            Ok(MoveOutcome::Rejected(rejection)) => {
                slice.schedule_path_relink.rollbacks = slice.schedule_path_relink.rollbacks.saturating_add(1);
                match rejection {
                    MoveRejection::Cycle => {
                        slice.schedule_path_relink.cycle_rejections = slice.schedule_path_relink.cycle_rejections.saturating_add(1);
                    }
                    MoveRejection::Window => {
                        slice.schedule_path_relink.window_rejections = slice.schedule_path_relink.window_rejections.saturating_add(1);
                    }
                    _ => {
                        slice.schedule_path_relink.other_rejections = slice.schedule_path_relink.other_rejections.saturating_add(1);
                    }
                }
                None
            }
            Err(_) => {
                slice.schedule_path_relink.rollbacks = slice.schedule_path_relink.rollbacks.saturating_add(1);
                add_state_metrics(total, scratch.take_metrics());
                return Err(Interrupted);
            }
        }
    } else if tail_corridor.len() >= 2 {
        slice.schedule_tsab.restart_n6_generated =
            slice.schedule_tsab.restart_n6_generated.saturating_add(u64::try_from(tail_corridor_generated).unwrap_or(u64::MAX));
        slice.schedule_tsab.restart_shortlist_peak_bytes = slice
            .schedule_tsab
            .restart_shortlist_peak_bytes
            .max(tail_corridor.capacity().saturating_mul(std::mem::size_of::<ScheduleMove>()));
        match scratch.consider_move_batch_full_oracle(&tail_corridor, tsab_kick_acceptance(objective_limit), stop) {
            Ok(MoveOutcome::Accepted { current, .. }) => {
                committed_kick_moves = u64::try_from(tail_corridor.len()).unwrap_or(u64::MAX);
                Some(current)
            }
            Ok(MoveOutcome::Rejected(_)) => None,
            Err(_) => {
                add_state_metrics(total, scratch.take_metrics());
                return Err(Interrupted);
            }
        }
    } else {
        let mut builder = TsabN6RestartShortlistBuilder::default();
        let mut n6_insert_generated = 0usize;
        let restart_epoch = session.tsab_restart_epoch;
        let seed = session.seed;
        if scratch
            .visit_scored_canonical_critical_moves(CriticalNeighborhood::N6, stop, |candidate, _| {
                if matches!(candidate.movement, ScheduleMove::Insert { from, to, .. } if from.abs_diff(to) > 1) {
                    n6_insert_generated = n6_insert_generated.saturating_add(1);
                    let certified = scratch.certifies_insert_acyclicity(candidate.movement, stop)?;
                    builder.consider(candidate, certified, seed, restart_epoch);
                }
                Ok(())
            })
            .is_err()
        {
            add_state_metrics(total, scratch.take_metrics());
            return Err(Interrupted);
        }
        slice.schedule_tsab.restart_n6_generated =
            slice.schedule_tsab.restart_n6_generated.saturating_add(u64::try_from(n6_insert_generated).unwrap_or(u64::MAX));
        slice.schedule_tsab.restart_shortlist_peak_bytes =
            slice.schedule_tsab.restart_shortlist_peak_bytes.max(builder.heap_lower_bound_bytes());
        let shortlist = builder.finish();
        debug_assert!(shortlist.len() <= TSAB_EXACT_SHORTLIST_CAPACITY);
        let mut best = None::<(i64, usize, ScheduleMove)>;
        for (rank, (_, _, candidate)) in shortlist.into_iter().enumerate() {
            slice.schedule_tsab.restart_delta_probes = slice.schedule_tsab.restart_delta_probes.saturating_add(1);
            restart_probe_count = restart_probe_count.saturating_add(1);
            match scratch.probe_move(candidate.movement, stop) {
                Ok(MoveProbe::Feasible { candidate: objective, .. }) if objective <= objective_limit => {
                    let probed = (objective, rank, candidate.movement);
                    if best.is_none_or(|incumbent| probed < incumbent) {
                        best = Some(probed);
                    }
                }
                Ok(MoveProbe::Feasible { .. } | MoveProbe::Rejected(_)) => {}
                Err(_) => {
                    add_state_metrics(total, scratch.take_metrics());
                    return Err(Interrupted);
                }
            }
        }
        if let Some((_, _, movement)) = best {
            match scratch.commit_probed_move(movement, tsab_kick_acceptance(objective_limit), stop) {
                Ok(MoveOutcome::Accepted { current, .. }) => {
                    committed_kick_moves = 1;
                    Some(current)
                }
                Ok(MoveOutcome::Rejected(_)) => None,
                Err(_) => {
                    add_state_metrics(total, scratch.take_metrics());
                    return Err(Interrupted);
                }
            }
        } else {
            None
        }
    };
    if checkpoint(stop).is_err() {
        if committed_relink_moves != 0 {
            slice.schedule_path_relink.rollbacks = slice.schedule_path_relink.rollbacks.saturating_add(1);
        }
        add_state_metrics(total, scratch.take_metrics());
        return Err(Interrupted);
    }

    // No fallible work remains after this point. Replace the current state and
    // all trajectory-local controls as one transaction.
    add_state_metrics(total, session.state_mut().take_metrics());
    session.state = Some(scratch);
    session.tabu_until.clear();
    clear_persistent_scan(session);
    session.direct_oracle_work_credit = 0;
    session.tsab_n5_work_since_restart = 0;
    session.tsab_restart_work_credit = 0;
    session.tsab_restart_epoch = session.tsab_restart_epoch.saturating_add(1);
    session.tsab_fast_enabled = session.state().supports_strict_n5_fast_path() && !session.tsab_fast_disabled;
    session.tsab_fast_hard_failure = false;
    session.tsab_fast_validation_due_on_resume = false;
    session.tsab_fast_commits_since_validation = 0;
    session.tsab_pending_elite = None;
    session.restart_epoch = session.restart_epoch.saturating_add(1);
    session.probe_step = session.probe_step.saturating_add(restart_probe_count);
    session.scan_epoch = 0;
    session.last_restart_probe = session.probe_step;
    slice.schedule_tsab.elite_restarts = slice.schedule_tsab.elite_restarts.saturating_add(1);

    if used_fresh {
        slice.schedule_tsab.restart_global_rebases = slice.schedule_tsab.restart_global_rebases.saturating_add(1);
        slice.session_rebases = slice.session_rebases.saturating_add(1);
        *incumbent_injections = incumbent_injections.saturating_add(1);
    }
    if let Some(current) = kicked_objective {
        session.accepted_step = session.accepted_step.saturating_add(1);
        if committed_relink_moves != 0 {
            slice.schedule_tsab.restart_relink_commits = slice.schedule_tsab.restart_relink_commits.saturating_add(1);
            slice.schedule_tsab.restart_relink_components_committed =
                slice.schedule_tsab.restart_relink_components_committed.saturating_add(committed_relink_moves);
            slice.schedule_tsab.restart_relink_guide_arc_gain_accepted =
                slice.schedule_tsab.restart_relink_guide_arc_gain_accepted.saturating_add(committed_relink_gain);
        } else {
            slice.schedule_tsab.n6_kicks = slice.schedule_tsab.n6_kicks.saturating_add(1);
            slice.schedule_tsab.kick_moves = slice.schedule_tsab.kick_moves.saturating_add(committed_kick_moves);
        }
        slice.schedule_tsab.restart_oracle_commits = slice.schedule_tsab.restart_oracle_commits.saturating_add(1);
        slice.schedule_tsab.restart_best_kicked_objective =
            Some(slice.schedule_tsab.restart_best_kicked_objective.map_or(current, |best| best.min(current)));
    } else {
        slice.schedule_tsab.restart_rejections = slice.schedule_tsab.restart_rejections.saturating_add(1);
    }

    let mut elite_changed = false;
    if base_objective < session.elite_objective {
        // A fresh incumbent was already published and verified by the shared
        // portfolio. Rebase the local elite monotonically, but reserve
        // `report` for a new improvement produced by the committed restart.
        session.elite_objective = base_objective;
        session.elite_solution = base_solution;
        elite_changed = true;
    }
    let post_commit_improvement = kicked_objective.is_some_and(|current| current < session.elite_objective);
    if post_commit_improvement {
        let current = session.state().makespan();
        if committed_relink_moves != 0 {
            slice.schedule_path_relink.elite_improvements = slice.schedule_path_relink.elite_improvements.saturating_add(1);
        }
        slice.schedule_tsab.improving_commits = slice.schedule_tsab.improving_commits.saturating_add(1);
        slice.schedule_tsab.best_committed_objective =
            Some(slice.schedule_tsab.best_committed_objective.map_or(current, |best| best.min(current)));
        slice.schedule_tsab.post_restart_improvements = slice.schedule_tsab.post_restart_improvements.saturating_add(1);
        session.elite_objective = current;
        session.elite_solution = session.state().to_solution();
        session.last_improvement_step = session.accepted_step;
        session.last_improvement_probe = session.probe_step;
        *incumbent_improvements = incumbent_improvements.saturating_add(1);
        report(current);
        elite_changed = true;
    } else if elite_changed {
        session.last_improvement_step = session.accepted_step;
        session.last_improvement_probe = session.probe_step;
    }
    if elite_changed {
        capture_search_elite_shadow(session, slice, search_elite_shadow, stop);
    }
    session.last_improvement_step = session.accepted_step;
    Ok(true)
}

fn reactive_restart_due(session: &ScheduleSearchSession) -> bool {
    if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate
        || session.profile.uses_baseline_kernel()
        || session.profile.uses_direct_oracle()
    {
        return false;
    }
    let progress_anchor = session.last_improvement_probe.max(session.last_restart_probe);
    session.probe_step.saturating_sub(progress_anchor) >= session.profile.restart_probe_interval(session.tenure_base)
}

#[allow(clippy::too_many_arguments)]
fn perform_reactive_restart(
    session: &mut ScheduleSearchSession,
    sched: &Schedule,
    stop: &AtomicBool,
    probes: &mut u64,
    probe_limit: u64,
    total: &mut ScheduleStateMetrics,
    slice: &mut PersistentScheduleSliceMetrics,
    construction_elapsed: &mut Duration,
    incumbent_improvements: &mut u64,
    search_elite_shadow: bool,
    report: &mut dyn FnMut(i64),
) -> Interruptible<()> {
    checkpoint(stop)?;
    let restart_started = Instant::now();
    let mut rebuilt = None;
    let dispatch_due = session.profile.dispatch_rule != DispatchRule::EarliestStart
        && !session.dispatch_restart_done
        && session.probe_step.saturating_sub(session.last_improvement_probe) >= session.profile.dispatch_probe_interval();
    if dispatch_due {
        session.dispatch_restart_done = true;
        slice.reactive_restart_dispatches = slice.reactive_restart_dispatches.saturating_add(1);
        if let Some(problem) = JobShopProblem::recognize(sched, stop).map_err(|_| Interrupted)? {
            let restart_seed =
                mix64(session.seed ^ session.profile.tie_break_salt ^ session.restart_epoch.wrapping_mul(0x8ebc_6af0_9c88_c6e3));
            let (constructed, partial_metrics) =
                JobShopState::giffler_thompson_profiled(&problem, restart_seed, session.profile.dispatch_rule, stop);
            match constructed {
                Ok(Some(state)) => rebuilt = Some(state),
                Ok(None) => add_state_metrics(total, partial_metrics),
                Err(_) => {
                    add_state_metrics(total, partial_metrics);
                    return Err(Interrupted);
                }
            }
        }
    }
    if rebuilt.is_none() {
        rebuilt = rebuild_session_from_incumbent(session, &session.elite_solution, stop)?;
    }
    *construction_elapsed = construction_elapsed.saturating_add(restart_started.elapsed());
    let Some(rebuilt) = rebuilt else {
        slice.reactive_restart_rebuild_failures = slice.reactive_restart_rebuild_failures.saturating_add(1);
        session.last_restart_probe = session.probe_step;
        return Ok(());
    };

    add_state_metrics(total, session.state_mut().take_metrics());
    session.state = Some(rebuilt);
    session.tabu_until.clear();
    clear_persistent_scan(session);
    session.scan_epoch = 0;
    session.last_improvement_step = session.accepted_step;
    session.restart_epoch = session.restart_epoch.saturating_add(1);
    session.last_restart_probe = session.probe_step;
    slice.reactive_restarts = slice.reactive_restarts.saturating_add(1);

    let restart_objective = session.state().makespan();
    if restart_objective < session.elite_objective {
        session.elite_objective = restart_objective;
        session.elite_solution = session.state().to_solution();
        session.last_improvement_step = session.accepted_step;
        session.last_improvement_probe = session.probe_step;
        *incumbent_improvements = incumbent_improvements.saturating_add(1);
        report(restart_objective);
        capture_search_elite_shadow(session, slice, search_elite_shadow, stop);
    }

    if *probes >= probe_limit {
        return Ok(());
    }
    let ScheduleSearchSession { state, scan, .. } = session;
    state
        .as_mut()
        .expect("search sessions always own a job-shop state")
        .fill_critical_moves(CriticalNeighborhood::N6, &mut scan.movements, stop)
        .map_err(|_| Interrupted)?;
    if session.scan.movements.is_empty() {
        clear_persistent_scan(session);
        return Ok(());
    }
    let offset =
        usize::try_from(mix64(session.seed ^ session.profile.tie_break_salt ^ session.restart_epoch.wrapping_mul(0xe703_7ed1_a0b4_28db)))
            .unwrap_or(usize::MAX)
            % session.scan.movements.len();
    let movement = session.scan.movements[offset];
    *probes = probes.saturating_add(1);
    session.probe_step = session.probe_step.saturating_add(1);
    let probe = session.state_mut().probe_move(movement, stop);
    let Ok(MoveProbe::Feasible { .. }) = probe else {
        clear_persistent_scan(session);
        session.last_restart_probe = session.probe_step;
        return probe.map(|_| ()).map_err(|_| Interrupted);
    };
    match session.state_mut().commit_probed_move(movement, MinimizingMoveAcceptance::Always, stop) {
        Ok(MoveOutcome::Accepted { current, .. }) => {
            session.accepted_step = session.accepted_step.saturating_add(1);
            slice.reactive_restart_perturbations = slice.reactive_restart_perturbations.saturating_add(1);
            if current < session.elite_objective {
                session.elite_objective = current;
                session.elite_solution = session.state().to_solution();
                session.last_improvement_step = session.accepted_step;
                session.last_improvement_probe = session.probe_step;
                *incumbent_improvements = incumbent_improvements.saturating_add(1);
                report(current);
                capture_search_elite_shadow(session, slice, search_elite_shadow, stop);
            }
        }
        Ok(MoveOutcome::Rejected(_)) => {}
        Err(_) => {
            clear_persistent_scan(session);
            session.last_restart_probe = session.probe_step;
            return Err(Interrupted);
        }
    }
    clear_persistent_scan(session);
    session.last_restart_probe = session.probe_step;
    Ok(())
}

/// Advance one persistent JSSP tabu trajectory by a bounded number of probes.
/// Unsupported schedules retain the existing stateless scheduling behavior.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    allow_rebase: bool,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    let search_elite_token =
        previous_session.as_ref().and_then(|session| session.search_elite_token.clone()).unwrap_or_else(ScheduleEliteSolveToken::new);
    solve_schedule_capped_persistent_shadow(
        sched,
        seed,
        fallback_seed,
        worker,
        workers,
        stop,
        max_iterations,
        initial_incumbent,
        previous_session,
        allow_rebase,
        Some(&search_elite_token),
        report,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent_tsab(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    solve_schedule_capped_persistent_tsab_with_finalization_stop(
        sched,
        seed,
        fallback_seed,
        worker,
        workers,
        stop,
        stop,
        max_iterations,
        initial_incumbent,
        previous_session,
        report,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent_tsab_with_finalization_stop(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    search_stop: &AtomicBool,
    finalization_stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    let search_elite_token =
        previous_session.as_ref().and_then(|session| session.search_elite_token.clone()).unwrap_or_else(ScheduleEliteSolveToken::new);
    solve_schedule_capped_persistent_hybrid_with_strategy(
        sched,
        seed,
        fallback_seed,
        worker,
        workers,
        search_stop,
        finalization_stop,
        max_iterations,
        initial_incumbent,
        previous_session,
        initial_incumbent.is_some(),
        Some(&search_elite_token),
        false,
        false,
        ScheduleJsspSearchStrategy::TsabCandidate,
        None,
        report,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent_tsab_multi(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    let search_elite_token =
        previous_session.as_ref().and_then(|session| session.search_elite_token.clone()).unwrap_or_else(ScheduleEliteSolveToken::new);
    solve_schedule_capped_persistent_hybrid_with_strategy(
        sched,
        seed,
        fallback_seed,
        worker,
        workers,
        stop,
        stop,
        max_iterations,
        initial_incumbent,
        previous_session,
        initial_incumbent.is_some(),
        Some(&search_elite_token),
        false,
        false,
        ScheduleJsspSearchStrategy::TsabMultiCandidate,
        None,
        report,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent_tsab_multi_relink(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    relink_request: ScheduleRelinkRequest<'_>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    let search_elite_token =
        previous_session.as_ref().and_then(|session| session.search_elite_token.clone()).unwrap_or_else(ScheduleEliteSolveToken::new);
    solve_schedule_capped_persistent_hybrid_with_strategy(
        sched,
        seed,
        fallback_seed,
        worker,
        workers,
        stop,
        stop,
        max_iterations,
        initial_incumbent,
        previous_session,
        initial_incumbent.is_some(),
        Some(&search_elite_token),
        false,
        false,
        ScheduleJsspSearchStrategy::TsabMultiCandidate,
        Some(relink_request),
        report,
    )
}

/// Persistent scheduling with an explicit measurement-only archive switch for
/// the portfolio and deterministic shadow audits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent_shadow(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    allow_rebase: bool,
    search_elite_token: Option<&ScheduleEliteSolveToken>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    solve_schedule_capped_persistent_relink(
        sched,
        seed,
        fallback_seed,
        worker,
        workers,
        stop,
        max_iterations,
        initial_incumbent,
        previous_session,
        allow_rebase,
        search_elite_token,
        None,
        report,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent_relink(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    allow_rebase: bool,
    search_elite_token: Option<&ScheduleEliteSolveToken>,
    relink_request: Option<ScheduleRelinkRequest<'_>>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    solve_schedule_capped_persistent_relink_with_lns_shadow(
        sched,
        seed,
        fallback_seed,
        worker,
        workers,
        stop,
        max_iterations,
        initial_incumbent,
        previous_session,
        allow_rebase,
        search_elite_token,
        false,
        relink_request,
        report,
    )
}

/// Persistent scheduling with an explicit, default-off measurement-only LNS
/// switch. Shadow repairs never replace either the current state or the elite.
/// The caller must keep the switch off for deterministic iteration budgets,
/// because exact-repair wall work is intentionally separate from tabu probes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent_relink_with_lns_shadow(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    allow_rebase: bool,
    search_elite_token: Option<&ScheduleEliteSolveToken>,
    schedule_lns_shadow_enabled: bool,
    relink_request: Option<ScheduleRelinkRequest<'_>>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    solve_schedule_capped_persistent_hybrid(
        sched,
        seed,
        fallback_seed,
        worker,
        workers,
        stop,
        max_iterations,
        initial_incumbent,
        previous_session,
        allow_rebase,
        search_elite_token,
        schedule_lns_shadow_enabled,
        false,
        relink_request,
        report,
    )
}

pub(crate) fn schedule_constructor_multistart_supported(schedule: &Schedule, stop: &AtomicBool) -> bool {
    if stop.load(Ordering::Acquire) || schedule.intervals.is_empty() || !schedule.minimize_makespan {
        return false;
    }
    let mut all_fixed = true;
    let mut all_single_mode = true;
    let mut durations = Vec::with_capacity(schedule.intervals.len());
    for interval in &schedule.intervals {
        if stop.load(Ordering::Acquire) || interval.optional {
            return false;
        }
        all_fixed &= interval.modes.is_empty();
        all_single_mode &= interval.modes.len() == 1;
        if interval.modes.is_empty() {
            if interval.duration < 0 || interval.horizon < interval.duration {
                return false;
            }
            durations.push(interval.duration);
        } else if interval.modes.len() == 1 {
            let mode = &interval.modes[0];
            let (start_min, start_max) = mode.start_window;
            let Some(last_start) = interval.horizon.checked_sub(mode.duration) else {
                return false;
            };
            if mode.duration < 0 || start_min < 0 || start_min > start_max || start_max > last_start || i64::try_from(mode.machine).is_err()
            {
                return false;
            }
            durations.push(mode.duration);
        }
    }
    if !all_fixed && !all_single_mode {
        return false;
    }
    if all_fixed {
        let mut assigned = vec![false; schedule.intervals.len()];
        for resource in &schedule.resources {
            let Resource::NoOverlap(operations) = resource else {
                return false;
            };
            for &operation in operations {
                if stop.load(Ordering::Acquire) || operation >= assigned.len() || assigned[operation] {
                    return false;
                }
                assigned[operation] = true;
            }
        }
        if assigned.iter().any(|assigned| !assigned) {
            return false;
        }
    } else if schedule.resources.len() != 1 || !matches!(schedule.resources.first(), Some(Resource::MachineNoOverlap)) {
        return false;
    }

    let mut successors = vec![Vec::new(); schedule.intervals.len()];
    for &(before, after) in &schedule.precedences {
        if stop.load(Ordering::Acquire) || before >= successors.len() || after >= successors.len() {
            return false;
        }
        successors[before].push(after);
    }
    for list in &mut successors {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        list.sort_unstable();
        list.dedup();
    }
    let Some(precedences) = PrecedenceDag::compile(successors, stop) else {
        return false;
    };
    precedences.remaining_paths(&durations, stop).is_some() && !stop.load(Ordering::Acquire)
}

const SCHEDULE_CONSTRUCTOR_MULTISTART_FIRST_ORDINAL: u64 = 6;
const SCHEDULE_CONSTRUCTOR_MULTISTART_WORK_UNITS: u64 = 64;
const SCHEDULE_CONSTRUCTOR_MULTISTART_MAX_PER_BOUNDARY: u64 = 4;

fn schedule_constructor_multistart_seed(base_seed: u64, ordinal: u64) -> u64 {
    if ordinal == SCHEDULE_CONSTRUCTOR_MULTISTART_FIRST_ORDINAL {
        base_seed
    } else {
        mix64(base_seed ^ ordinal.saturating_sub(SCHEDULE_CONSTRUCTOR_MULTISTART_FIRST_ORDINAL).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_schedule_constructor_multistart(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    let started = Instant::now();
    let profile = ScheduleIslandProfile::for_worker(worker, workers);
    let mut initialized = false;
    let mut resumed = false;
    let mut session = match previous_session {
        Some(session) if session.constructor_multistart.is_some() => {
            debug_assert!(session.state.is_none());
            resumed = true;
            session
        }
        Some(session) => {
            return solve_schedule_capped_persistent_relink_with_lns_shadow(
                sched,
                seed,
                fallback_seed,
                worker,
                workers,
                stop,
                max_iterations,
                initial_incumbent,
                Some(session),
                false,
                None,
                false,
                None,
                &mut |_| {},
            );
        }
        None => {
            let problem = match JobShopProblem::recognize(sched, stop) {
                Ok(Some(problem)) => problem,
                Ok(None) | Err(_) => {
                    return solve_schedule_capped_persistent_relink_with_lns_shadow(
                        sched,
                        seed,
                        fallback_seed,
                        worker,
                        workers,
                        stop,
                        max_iterations,
                        initial_incumbent,
                        None,
                        false,
                        None,
                        false,
                        None,
                        &mut |_| {},
                    );
                }
            };
            let workspace = GifflerThompsonWorkspace::new(&problem);
            let workspace_peak_bytes = workspace.heap_lower_bound_bytes();
            initialized = true;
            ScheduleSearchSession {
                seed,
                profile,
                requested_strategy: ScheduleJsspSearchStrategy::Legacy,
                strategy: ScheduleJsspSearchStrategy::Legacy,
                state: None,
                elite_solution: unknown_solution(),
                elite_objective: i64::MAX,
                tabu_until: BTreeMap::new(),
                accepted_step: 0,
                last_improvement_step: 0,
                probe_step: 0,
                last_improvement_probe: 0,
                last_restart_probe: 0,
                dispatch_restart_done: false,
                restart_epoch: 0,
                scan_epoch: 0,
                consecutive_direct_oracle_failures: 0,
                direct_oracle_work_credit: 0,
                tsab_n5_work_since_restart: 0,
                tsab_restart_work_credit: 0,
                tsab_restart_epoch: 0,
                tsab_fast_enabled: false,
                tsab_fast_disabled: false,
                tsab_fast_hard_failure: false,
                tsab_fast_validation_due_on_resume: false,
                tsab_fast_commits_since_validation: 0,
                tsab_pending_elite: None,
                #[cfg(test)]
                tsab_fast_test_stop_after_attempts: None,
                #[cfg(test)]
                tsab_fast_test_stop_after_pending_elite: false,
                #[cfg(test)]
                tsab_fast_test_return_after_pending_elite: false,
                #[cfg(test)]
                tsab_fast_test_stop_during_validation: false,
                tenure_base: 0,
                completed_boundaries: 0,
                tsab_legacy_warmup_work_steps: 0,
                tsab_activation_boundary: None,
                scan: PersistentScheduleScan::default(),
                search_elite_token: None,
                pending_search_elite: None,
                schedule_lns_shadow: None,
                schedule_relink: ScheduleRelinkWorkspace::default(),
                constructor_multistart: Some(ScheduleConstructorMultistartSession {
                    problem,
                    workspace,
                    work_credit: 0,
                    initial_objective: None,
                    best_ordinal: None,
                    best_seed: None,
                    best_fingerprint: None,
                    first_fingerprint: None,
                    alternative_fingerprint_seen: false,
                    next_ordinal: SCHEDULE_CONSTRUCTOR_MULTISTART_FIRST_ORDINAL,
                    workspace_peak_bytes,
                }),
            }
        }
    };
    debug_assert!(session.state.is_none() && session.constructor_multistart.is_some());
    let initial_dispatch_rule = session.profile.initial_dispatch_rule;
    let initial_dispatch_name = session.profile.initial_dispatch_name();

    let mut state_metrics = ScheduleStateMetrics::default();
    let mut attempts = 0u64;
    let mut constructions = 0u64;
    let mut interruptions = 0u64;
    let mut failures = 0u64;
    let mut feasible = 0u64;
    let mut distinct_fingerprints = 0u64;
    let mut other_fingerprint_observations = 0u64;
    let mut improvements = 0u64;
    let mut work_units = 0u64;
    let mut constructor_elapsed = Duration::ZERO;
    let mut first_feasible = None;
    let mut initial_objective = None;
    while attempts < SCHEDULE_CONSTRUCTOR_MULTISTART_MAX_PER_BOUNDARY && work_units < max_iterations {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let constructor = session.constructor_multistart.as_mut().expect("constructor session must be active");
        let needed = SCHEDULE_CONSTRUCTOR_MULTISTART_WORK_UNITS.saturating_sub(constructor.work_credit);
        let granted = needed.min(max_iterations.saturating_sub(work_units));
        constructor.work_credit = constructor.work_credit.saturating_add(granted);
        work_units = work_units.saturating_add(granted);
        if constructor.work_credit < SCHEDULE_CONSTRUCTOR_MULTISTART_WORK_UNITS {
            break;
        }
        constructor.work_credit = 0;
        attempts = attempts.saturating_add(1);
        let ordinal = constructor.next_ordinal;
        let attempt_seed = schedule_constructor_multistart_seed(seed, ordinal);
        let attempt_started = Instant::now();
        let outcome = constructor.workspace.construct(&constructor.problem, attempt_seed, initial_dispatch_rule, stop, &mut state_metrics);
        constructor_elapsed = constructor_elapsed.saturating_add(attempt_started.elapsed());
        constructor.workspace_peak_bytes = constructor.workspace_peak_bytes.max(constructor.workspace.heap_lower_bound_bytes());
        let (objective, fingerprint) = match outcome {
            Ok(Some(candidate)) => candidate,
            Ok(None) => {
                failures = failures.saturating_add(1);
                constructor.next_ordinal = constructor.next_ordinal.saturating_add(1);
                continue;
            }
            Err(_) => {
                interruptions = interruptions.saturating_add(1);
                break;
            }
        };
        constructions = constructions.saturating_add(1);
        feasible = feasible.saturating_add(1);
        if constructor.first_fingerprint.is_none() {
            constructor.first_fingerprint = Some(fingerprint);
            distinct_fingerprints = distinct_fingerprints.saturating_add(1);
        } else if constructor.first_fingerprint != Some(fingerprint) && !constructor.alternative_fingerprint_seen {
            constructor.alternative_fingerprint_seen = true;
            distinct_fingerprints = distinct_fingerprints.saturating_add(1);
        } else {
            other_fingerprint_observations = other_fingerprint_observations.saturating_add(1);
        }
        if constructor.initial_objective.is_none() {
            constructor.initial_objective = Some(objective);
            initial_objective = Some(objective);
            first_feasible = Some(started.elapsed());
        }
        if objective < session.elite_objective {
            if session.elite_solution.feasible {
                improvements = improvements.saturating_add(1);
            }
            session.elite_objective = objective;
            session.elite_solution = constructor.workspace.to_solution(&constructor.problem, objective);
            constructor.best_ordinal = Some(ordinal);
            constructor.best_seed = Some(attempt_seed);
            constructor.best_fingerprint = Some(fingerprint);
        }
        constructor.next_ordinal = constructor.next_ordinal.saturating_add(1);
    }

    let constructor = session.constructor_multistart.as_ref().expect("constructor session must be active");
    let solution = session.elite_solution.clone();
    let (_, mut metrics) =
        job_shop_result(solution.feasible.then_some(solution.clone()), constructor_elapsed, first_feasible, state_metrics, 0, 0);
    metrics.elapsed = started.elapsed();
    metrics.constructor = "giffler-thompson-multistart";
    metrics.work_steps = work_units;
    metrics.initial_objective = initial_objective;
    metrics.initial_dispatch_rule = initial_objective.map(|_| initial_dispatch_name);
    metrics.session_initializations = u64::from(initialized);
    metrics.session_resumes = u64::from(resumed);
    metrics.island_profile = None;
    metrics.island_profile_mask = 0;
    metrics.baseline_island_profile_mask = 0;
    metrics.scored_island_profile_mask = 0;
    metrics.schedule_constructor_multistart_owner_worker_mask = 1u64.checked_shl(u32::try_from(worker).unwrap_or(u32::MAX)).unwrap_or(0);
    metrics.schedule_constructor_multistart_attempts = attempts;
    metrics.schedule_constructor_multistart_constructions = constructions;
    metrics.schedule_constructor_multistart_interruptions = interruptions;
    metrics.schedule_constructor_multistart_failures = failures;
    metrics.schedule_constructor_multistart_feasible = feasible;
    metrics.schedule_constructor_multistart_distinct_fingerprints = distinct_fingerprints;
    metrics.schedule_constructor_multistart_other_fingerprint_observations = other_fingerprint_observations;
    metrics.schedule_constructor_multistart_initial_objective = initial_objective;
    metrics.schedule_constructor_multistart_best_objective = solution.objectives.first().copied();
    metrics.schedule_constructor_multistart_improvements = improvements;
    metrics.schedule_constructor_multistart_work_units = work_units;
    metrics.schedule_constructor_multistart_elapsed = constructor_elapsed;
    metrics.schedule_constructor_multistart_workspace_peak_bytes = constructor.workspace_peak_bytes;
    metrics.schedule_constructor_multistart_best_ordinal = constructor.best_ordinal;
    metrics.schedule_constructor_multistart_best_seed = constructor.best_seed;
    metrics.schedule_constructor_multistart_best_fingerprint = constructor.best_fingerprint;
    metrics.schedule_constructor_multistart_next_ordinal = Some(constructor.next_ordinal);
    (solution, metrics, Some(session))
}

/// Persistent scheduling with independent default-off shadow controls.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent_hybrid(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    allow_rebase: bool,
    search_elite_token: Option<&ScheduleEliteSolveToken>,
    schedule_lns_shadow_enabled: bool,
    schedule_constructor_multistart_enabled: bool,
    relink_request: Option<ScheduleRelinkRequest<'_>>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    solve_schedule_capped_persistent_hybrid_with_strategy(
        sched,
        seed,
        fallback_seed,
        worker,
        workers,
        stop,
        stop,
        max_iterations,
        initial_incumbent,
        previous_session,
        allow_rebase,
        search_elite_token,
        schedule_lns_shadow_enabled,
        schedule_constructor_multistart_enabled,
        ScheduleJsspSearchStrategy::Legacy,
        relink_request,
        report,
    )
}

/// Persistent scheduling with an explicit, default-off JSSP search kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_schedule_capped_persistent_hybrid_with_strategy(
    sched: &Schedule,
    seed: u64,
    fallback_seed: u64,
    worker: usize,
    workers: usize,
    search_stop: &AtomicBool,
    finalization_stop: &AtomicBool,
    max_iterations: u64,
    initial_incumbent: Option<&CollectionSolution>,
    previous_session: Option<ScheduleSearchSession>,
    allow_rebase: bool,
    search_elite_token: Option<&ScheduleEliteSolveToken>,
    schedule_lns_shadow_enabled: bool,
    schedule_constructor_multistart_enabled: bool,
    strategy: ScheduleJsspSearchStrategy,
    relink_request: Option<ScheduleRelinkRequest<'_>>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics, Option<ScheduleSearchSession>) {
    let stop = search_stop;
    if schedule_constructor_multistart_enabled && workers == 7 && worker == 6 {
        return solve_schedule_constructor_multistart(
            sched,
            seed,
            fallback_seed,
            worker,
            workers,
            stop,
            max_iterations,
            initial_incumbent,
            previous_session,
        );
    }
    let started = Instant::now();
    let mut slice = PersistentScheduleSliceMetrics::default();
    let mut total = ScheduleStateMetrics::default();
    let mut construction_elapsed = Duration::ZERO;
    let mut first_feasible = None;
    let mut initial_objective = None;
    let mut incumbent_improvements = 0u64;
    let mut incumbent_injections = 0u64;
    let requested_profile = ScheduleIslandProfile::for_worker(worker, workers);
    let requested_strategy = if strategy.owns_worker(worker, workers) { strategy } else { ScheduleJsspSearchStrategy::Legacy };

    let mut session = if let Some(session) = previous_session {
        debug_assert_eq!(session.profile, requested_profile);
        debug_assert_eq!(session.requested_strategy, requested_strategy);
        slice.session_resumes = 1;
        session
    } else {
        let problem = match JobShopProblem::recognize(sched, stop) {
            Ok(Some(problem)) => problem,
            Ok(None) | Err(_) if requested_strategy.requests_tsab() => {
                return (unknown_solution(), generic_schedule_metrics(Duration::ZERO, None, 0, 0, 0, 0), None);
            }
            Ok(None) | Err(_) => {
                let (solution, metrics) =
                    solve_schedule_capped(sched, fallback_seed, stop, max_iterations, false, initial_incumbent, report);
                return (solution, metrics, None);
            }
        };
        let complete_incumbent = complete_schedule_incumbent_ref(sched, initial_incumbent);
        let mut injected = false;
        let mut state = None;
        if let Some(incumbent) = complete_incumbent {
            if let Ok(sequences) = incumbent_machine_sequences(
                problem.operation_count(),
                problem.machine_count(),
                |operation| problem.machine(operation),
                incumbent,
                stop,
            ) {
                if let Ok(rebuilt) = JobShopState::from_machine_sequences(&problem, sequences, stop) {
                    state = rebuilt;
                    injected = state.is_some();
                }
            }
        }
        let mut state = if let Some(state) = state {
            state
        } else {
            let (constructed, _) = JobShopState::giffler_thompson_profiled(&problem, seed, requested_profile.initial_dispatch_rule, stop);
            let Ok(Some(state)) = constructed else {
                let (solution, metrics) =
                    solve_schedule_capped(sched, fallback_seed, stop, max_iterations, false, initial_incumbent, report);
                return (solution, metrics, None);
            };
            state
        };
        if requested_profile.uses_baseline_kernel() {
            state.retain_canonical_critical_blocks_only();
        }
        let objective = state.makespan();
        initial_objective = Some(objective);
        let elite_solution = state.to_solution();
        construction_elapsed = started.elapsed();
        first_feasible = Some(construction_elapsed);
        incumbent_injections = u64::from(injected);
        if !injected {
            incumbent_improvements = 1;
            report(objective);
        }
        let ratio = u64::try_from(problem.operation_count() / problem.machine_count().max(1)).unwrap_or(u64::MAX);
        slice.session_initializations = 1;
        ScheduleSearchSession {
            seed,
            profile: requested_profile,
            requested_strategy,
            strategy: ScheduleJsspSearchStrategy::Legacy,
            state: Some(state),
            elite_solution,
            elite_objective: objective,
            tabu_until: BTreeMap::new(),
            accepted_step: 0,
            last_improvement_step: 0,
            probe_step: 0,
            last_improvement_probe: 0,
            last_restart_probe: 0,
            dispatch_restart_done: false,
            restart_epoch: 0,
            scan_epoch: 0,
            consecutive_direct_oracle_failures: 0,
            direct_oracle_work_credit: 0,
            tsab_n5_work_since_restart: 0,
            tsab_restart_work_credit: 0,
            tsab_restart_epoch: 0,
            tsab_fast_enabled: false,
            tsab_fast_disabled: false,
            tsab_fast_hard_failure: false,
            tsab_fast_validation_due_on_resume: false,
            tsab_fast_commits_since_validation: 0,
            tsab_pending_elite: None,
            #[cfg(test)]
            tsab_fast_test_stop_after_attempts: None,
            #[cfg(test)]
            tsab_fast_test_stop_after_pending_elite: false,
            #[cfg(test)]
            tsab_fast_test_return_after_pending_elite: false,
            #[cfg(test)]
            tsab_fast_test_stop_during_validation: false,
            tenure_base: requested_profile.scaled_tenure(ratio.isqrt().clamp(5, 32)),
            completed_boundaries: 0,
            tsab_legacy_warmup_work_steps: 0,
            tsab_activation_boundary: None,
            scan: PersistentScheduleScan::default(),
            search_elite_token: search_elite_token.cloned(),
            pending_search_elite: None,
            schedule_lns_shadow: None,
            schedule_relink: ScheduleRelinkWorkspace::default(),
            constructor_multistart: None,
        }
    };

    slice.island_profile_mask = 1u64.checked_shl(u32::try_from(requested_profile.id).unwrap_or(u32::MAX)).unwrap_or(0);
    if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate {
        slice.schedule_tsab.owner_worker_mask = 1u64.checked_shl(u32::try_from(worker).unwrap_or(u32::MAX)).unwrap_or(0);
        slice.schedule_tsab.burst_work_limit = TSAB_BURST_WORK_LIMIT;
        slice.schedule_tsab.fast_eligible = u64::from(session.state().supports_strict_n5_fast_path());
        slice.schedule_tsab.fast_enabled = u64::from(session.tsab_fast_enabled);
        slice.schedule_tsab.fast_disabled = u64::from(session.tsab_fast_disabled);
    } else if requested_profile.uses_baseline_kernel() {
        slice.baseline_island_profile_mask = slice.island_profile_mask;
    } else {
        slice.scored_island_profile_mask = slice.island_profile_mask;
    }

    let search_elite_shadow = match (search_elite_token, session.search_elite_token.as_ref()) {
        (Some(expected), Some(actual)) if actual.same_solve(expected) => true,
        (Some(_), _) => {
            slice.search_elite_snapshot_errors = slice.search_elite_snapshot_errors.saturating_add(1);
            session.pending_search_elite = None;
            false
        }
        (None, _) => {
            session.pending_search_elite = None;
            false
        }
    };

    if slice.session_initializations != 0 {
        capture_search_elite_shadow(&mut session, &mut slice, search_elite_shadow, stop);
    }

    if allow_rebase && slice.session_resumes != 0 && session.requested_strategy == ScheduleJsspSearchStrategy::Legacy {
        if let Some(incumbent) = complete_schedule_incumbent_ref(sched, initial_incumbent) {
            let declared = incumbent.objectives.first().copied().unwrap_or(i64::MAX);
            if declared < session.elite_objective {
                let rebase_started = Instant::now();
                if let Ok(Some(rebuilt)) = rebuild_session_from_incumbent(&session, incumbent, stop) {
                    if rebuilt.makespan() < session.elite_objective {
                        add_state_metrics(&mut total, session.state_mut().take_metrics());
                        session.elite_objective = rebuilt.makespan();
                        session.elite_solution = rebuilt.to_solution();
                        session.state = Some(rebuilt);
                        session.tabu_until.clear();
                        clear_persistent_scan(&mut session);
                        session.last_improvement_step = session.accepted_step;
                        session.last_improvement_probe = session.probe_step;
                        session.last_restart_probe = session.probe_step;
                        if session.strategy == ScheduleJsspSearchStrategy::Legacy && session.profile.uses_direct_oracle() {
                            session.scan_epoch = 0;
                        }
                        slice.session_rebases = 1;
                        incumbent_injections = 1;
                        capture_search_elite_shadow(&mut session, &mut slice, search_elite_shadow, stop);
                    }
                }
                construction_elapsed = construction_elapsed.saturating_add(rebase_started.elapsed());
            }
        }
    }

    let default_moves = u64::try_from(session.state().operation_count()).unwrap_or(u64::MAX).saturating_mul(256).clamp(2_048, 100_000);
    let requested_probe_limit = if max_iterations == u64::MAX { default_moves } else { max_iterations };
    let complete_tsab_incumbent =
        session.requested_strategy.requests_tsab().then(|| complete_schedule_incumbent_ref(sched, initial_incumbent)).flatten();
    let warmup_start = session.tsab_legacy_warmup_work_steps;
    let mut active_start_probe = (session.strategy == ScheduleJsspSearchStrategy::TsabCandidate).then_some(0u64);
    let probe_limit = if session.requested_strategy.requests_tsab() {
        if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate {
            requested_probe_limit.min(TSAB_BURST_WORK_LIMIT)
        } else {
            let remaining_warmup = TSAB_LEGACY_WARMUP_WORK_UNITS.saturating_sub(warmup_start);
            if remaining_warmup == 0 && complete_tsab_incumbent.is_some() {
                requested_probe_limit.min(TSAB_BURST_WORK_LIMIT)
            } else {
                requested_probe_limit.min(remaining_warmup.min(TSAB_LEGACY_WARMUP_BOUNDARY_LIMIT))
            }
        }
    } else {
        requested_probe_limit
    };
    let mut probes = 0u64;
    let mut search_allowed = true;
    if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate
        && session.tsab_fast_validation_due_on_resume
        && !stop.load(Ordering::Acquire)
    {
        match validate_tsab_fast_checkpoint(
            &mut session,
            &mut total,
            &mut slice,
            stop,
            &mut incumbent_improvements,
            search_elite_shadow,
            report,
        ) {
            Ok(_) if !session.tsab_fast_hard_failure => {}
            Ok(_) | Err(_) => search_allowed = false,
        }
    }
    'search: while search_allowed && probes < probe_limit && checkpoint(stop).is_ok() {
        if session.requested_strategy.requests_tsab()
            && session.strategy == ScheduleJsspSearchStrategy::Legacy
            && warmup_start.saturating_add(probes) == TSAB_LEGACY_WARMUP_WORK_UNITS
        {
            session.tsab_legacy_warmup_work_steps = TSAB_LEGACY_WARMUP_WORK_UNITS;
            let Some(incumbent) = complete_tsab_incumbent else {
                break;
            };
            match activate_tsab_candidate(
                &mut session,
                worker,
                incumbent,
                allow_rebase,
                stop,
                &mut total,
                &mut slice,
                &mut construction_elapsed,
                &mut incumbent_injections,
                search_elite_shadow,
            ) {
                Ok(true) => active_start_probe = Some(probes),
                Ok(false) => break,
                Err(_) => break,
            }
        }
        if session.strategy == ScheduleJsspSearchStrategy::Legacy && session.profile.uses_direct_oracle() && session.scan.end == 0 {
            const DIRECT_ORACLE_WORK_UNITS: u64 = 4;
            let required = DIRECT_ORACLE_WORK_UNITS.saturating_sub(session.direct_oracle_work_credit);
            let remaining = probe_limit.saturating_sub(probes);
            if remaining < required {
                session.direct_oracle_work_credit = session.direct_oracle_work_credit.saturating_add(remaining);
                probes = probes.saturating_add(remaining);
                break;
            }
        }
        if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate
            && session.tsab_fast_enabled
            && session.scan.end == 0
            && session.tsab_n5_work_since_restart >= TSAB_ELITE_RESTART_N5_WORK_UNITS
            && (session.tsab_fast_commits_since_validation != 0 || session.tsab_pending_elite.is_some())
        {
            let validation = validate_tsab_fast_checkpoint(
                &mut session,
                &mut total,
                &mut slice,
                stop,
                &mut incumbent_improvements,
                search_elite_shadow,
                report,
            );
            if validation.is_err() || session.tsab_fast_hard_failure {
                break;
            }
        }
        if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate
            && session.scan.end == 0
            && session.tsab_n5_work_since_restart >= TSAB_ELITE_RESTART_N5_WORK_UNITS
        {
            let required = TSAB_ELITE_RESTART_WORK_UNITS.saturating_sub(session.tsab_restart_work_credit);
            let charged = required.min(probe_limit.saturating_sub(probes));
            session.tsab_restart_work_credit = session.tsab_restart_work_credit.saturating_add(charged);
            probes = probes.saturating_add(charged);
            slice.schedule_tsab.restart_work_units = slice.schedule_tsab.restart_work_units.saturating_add(charged);
            if session.tsab_restart_work_credit < TSAB_ELITE_RESTART_WORK_UNITS {
                break;
            }
            if perform_tsab_elite_n6_restart(
                &mut session,
                complete_tsab_incumbent,
                allow_rebase,
                relink_request,
                stop,
                &mut total,
                &mut slice,
                &mut construction_elapsed,
                &mut incumbent_improvements,
                &mut incumbent_injections,
                search_elite_shadow,
                report,
            )
            .is_err()
            {
                slice.schedule_tsab.restart_interruptions = slice.schedule_tsab.restart_interruptions.saturating_add(1);
                break;
            }
            if probes >= probe_limit {
                break;
            }
            continue;
        }
        if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate && session.tsab_fast_enabled {
            let fast_started = Instant::now();
            probes = probes.saturating_add(1);
            session.tsab_n5_work_since_restart = session.tsab_n5_work_since_restart.saturating_add(1);
            slice.schedule_tsab.fast_work_units = slice.schedule_tsab.fast_work_units.saturating_add(1);
            let mut interrupted = false;
            for _ in 0..TSAB_FAST_TRANSITIONS_PER_WORK_UNIT {
                if checkpoint(stop).is_err() {
                    interrupted = true;
                    break;
                }
                let selected = match select_tsab_fast_candidate(&mut session, &mut slice, stop) {
                    Ok(selected) => selected,
                    Err(_) => {
                        interrupted = true;
                        break;
                    }
                };
                let Some((movement, arcs)) = selected else {
                    break;
                };
                session.probe_step = session.probe_step.saturating_add(1);
                slice.schedule_tsab.fast_attempts = slice.schedule_tsab.fast_attempts.saturating_add(1);
                slice.schedule_tsab.delta_probes = slice.schedule_tsab.delta_probes.saturating_add(1);
                let fast = match session.state_mut().consider_strict_n5_fast(movement, MinimizingMoveAcceptance::Always, stop) {
                    Ok(fast) => fast,
                    Err(_) => {
                        interrupted = true;
                        break;
                    }
                };
                slice.schedule_tsab.fast_fallbacks = slice.schedule_tsab.fast_fallbacks.saturating_add(u64::from(fast.fell_back));
                slice.schedule_tsab.fast_work_cap_recoveries = slice
                    .schedule_tsab
                    .fast_work_cap_recoveries
                    .saturating_add(u64::from(fast.used_topological_recovery && !fast.fell_back));
                slice.schedule_tsab.fast_date_changes = slice
                    .schedule_tsab
                    .fast_date_changes
                    .saturating_add(fast.forward_date_changes.saturating_add(fast.reverse_tail_changes));
                slice.schedule_tsab.fast_queue_pops = slice.schedule_tsab.fast_queue_pops.saturating_add(fast.queue_pops);
                let MoveOutcome::Accepted { current, .. } = fast.outcome else {
                    continue;
                };
                slice.schedule_tsab.fast_transitions = slice.schedule_tsab.fast_transitions.saturating_add(1);
                slice.schedule_tsab.fast_commits = slice.schedule_tsab.fast_commits.saturating_add(1);
                session.tsab_fast_commits_since_validation =
                    session.tsab_fast_commits_since_validation.saturating_add(u64::from(fast.used_fast_path));
                session.accepted_step = session.accepted_step.saturating_add(1);
                slice.tabu_steps = slice.tabu_steps.saturating_add(1);
                for arc in arcs.removed.into_iter().flatten() {
                    let expiration = session.accepted_step.saturating_add(arc_tenure(&session, arc));
                    session.tabu_until.entry(arc).and_modify(|value| *value = (*value).max(expiration)).or_insert(expiration);
                }
                session.tabu_until.retain(|_, expiration| session.accepted_step < *expiration);

                let local_best = session
                    .tsab_pending_elite
                    .as_ref()
                    .and_then(|solution| solution.objectives.first().copied())
                    .unwrap_or(session.elite_objective);
                if current < local_best {
                    session.tsab_pending_elite = Some(session.state().to_solution());
                    let pending_bytes = session.tsab_pending_elite.as_ref().map_or(0, collection_solution_heap_lower_bound_bytes);
                    let workspace_bytes = session.state().fast_n5_workspace_lower_bound_bytes().saturating_add(pending_bytes);
                    slice.schedule_tsab.fast_workspace_peak_bytes = slice.schedule_tsab.fast_workspace_peak_bytes.max(workspace_bytes);
                    slice.schedule_tsab.improving_commits = slice.schedule_tsab.improving_commits.saturating_add(1);
                    if session.tsab_restart_epoch != 0 {
                        slice.schedule_tsab.post_restart_improvements = slice.schedule_tsab.post_restart_improvements.saturating_add(1);
                    }
                    slice.schedule_tsab.best_committed_objective =
                        Some(slice.schedule_tsab.best_committed_objective.map_or(current, |best| best.min(current)));
                }
                #[cfg(test)]
                if current < local_best && std::mem::take(&mut session.tsab_fast_test_return_after_pending_elite) {
                    interrupted = true;
                    break;
                }
                #[cfg(test)]
                if current < local_best && std::mem::take(&mut session.tsab_fast_test_stop_after_pending_elite) {
                    stop.store(true, Ordering::Release);
                    interrupted = true;
                    break;
                }
                #[cfg(test)]
                if session.tsab_fast_test_stop_after_attempts.is_some_and(|limit| slice.schedule_tsab.fast_attempts >= limit) {
                    session.tsab_fast_test_stop_after_attempts = None;
                    stop.store(true, Ordering::Release);
                    interrupted = true;
                    break;
                }
                if session.tsab_fast_commits_since_validation >= TSAB_FAST_VALIDATION_TRANSITIONS {
                    match validate_tsab_fast_checkpoint(
                        &mut session,
                        &mut total,
                        &mut slice,
                        stop,
                        &mut incumbent_improvements,
                        search_elite_shadow,
                        report,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            interrupted = session.tsab_fast_hard_failure;
                            break;
                        }
                        Err(_) => {
                            interrupted = true;
                            break;
                        }
                    }
                }
            }
            slice.schedule_tsab.fast_elapsed = slice.schedule_tsab.fast_elapsed.saturating_add(fast_started.elapsed());
            let pending_bytes = session.tsab_pending_elite.as_ref().map_or(0, collection_solution_heap_lower_bound_bytes);
            let fast_workspace_bytes = session.state().fast_n5_workspace_lower_bound_bytes().saturating_add(pending_bytes);
            slice.schedule_tsab.fast_workspace_peak_bytes = slice.schedule_tsab.fast_workspace_peak_bytes.max(fast_workspace_bytes);
            if interrupted {
                break;
            }
            continue 'search;
        }
        if session.scan.end == 0 && reactive_restart_due(&session) {
            if perform_reactive_restart(
                &mut session,
                sched,
                stop,
                &mut probes,
                probe_limit,
                &mut total,
                &mut slice,
                &mut construction_elapsed,
                &mut incumbent_improvements,
                search_elite_shadow,
                report,
            )
            .is_err()
            {
                break;
            }
            if probes >= probe_limit {
                break;
            }
        }
        if session.scan.end == 0 {
            let preparation_started = ((session.strategy == ScheduleJsspSearchStrategy::Legacy && session.profile.uses_direct_oracle())
                || session.strategy == ScheduleJsspSearchStrategy::TsabCandidate)
                .then(Instant::now);
            let scan_relink_request =
                (session.requested_strategy != ScheduleJsspSearchStrategy::TsabMultiCandidate).then_some(relink_request).flatten();
            let prepared = prepare_persistent_scan(&mut session, &mut slice, scan_relink_request, stop);
            if let Some(preparation_started) = preparation_started {
                slice.approximation_elapsed = slice.approximation_elapsed.saturating_add(preparation_started.elapsed());
            }
            if !matches!(prepared, Ok(true)) {
                break;
            }
        }
        if session.strategy == ScheduleJsspSearchStrategy::Legacy && session.profile.uses_direct_oracle() {
            const DIRECT_ORACLE_WORK_UNITS: u64 = 4;
            while session.scan.cursor < session.scan.end && probes < probe_limit {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                let required = DIRECT_ORACLE_WORK_UNITS.saturating_sub(session.direct_oracle_work_credit);
                let remaining = probe_limit.saturating_sub(probes);
                if remaining < required {
                    session.direct_oracle_work_credit = session.direct_oracle_work_credit.saturating_add(remaining);
                    probes = probes.saturating_add(remaining);
                    break 'search;
                }
                probes = probes.saturating_add(required);
                session.direct_oracle_work_credit = 0;
                let scan_index = session.scan.cursor;
                let movement = session.scan.movements[scan_index];
                let relink = scan_index < session.scan.relink_end;
                let relink_gain = if relink { session.scan.relink_guide_arc_gain[scan_index] } else { 0 };
                session.scan.cursor += 1;
                let Some(arcs) = session.state().move_arcs(movement) else {
                    if relink {
                        slice.schedule_path_relink.oracle_attempts = slice.schedule_path_relink.oracle_attempts.saturating_add(1);
                        slice.schedule_path_relink.other_rejections = slice.schedule_path_relink.other_rejections.saturating_add(1);
                    }
                    continue;
                };
                let tabu = !relink
                    && arcs
                        .added
                        .into_iter()
                        .flatten()
                        .any(|arc| session.tabu_until.get(&arc).is_some_and(|&expiration| session.accepted_step < expiration));
                slice.tabu_hits = slice.tabu_hits.saturating_add(u64::from(tabu));
                session.probe_step = session.probe_step.saturating_add(1);
                if relink {
                    slice.schedule_path_relink.oracle_attempts = slice.schedule_path_relink.oracle_attempts.saturating_add(1);
                }
                let acceptance =
                    if tabu { MinimizingMoveAcceptance::StrictlyBelow(session.elite_objective) } else { MinimizingMoveAcceptance::Always };
                match session.state_mut().consider_move_full_oracle(movement, acceptance, stop) {
                    Ok(MoveOutcome::Accepted { current, .. }) => {
                        if relink {
                            slice.schedule_path_relink.oracle_accepts = slice.schedule_path_relink.oracle_accepts.saturating_add(1);
                            slice.schedule_path_relink.guide_arc_gain_accepted =
                                slice.schedule_path_relink.guide_arc_gain_accepted.saturating_add(u64::from(relink_gain));
                        }
                        session.consecutive_direct_oracle_failures = 0;
                        session.accepted_step = session.accepted_step.saturating_add(1);
                        slice.tabu_steps = slice.tabu_steps.saturating_add(1);
                        slice.tabu_aspirations = slice.tabu_aspirations.saturating_add(u64::from(tabu));
                        for arc in arcs.removed.into_iter().flatten() {
                            let expiration = session.accepted_step.saturating_add(arc_tenure(&session, arc));
                            session.tabu_until.entry(arc).and_modify(|value| *value = (*value).max(expiration)).or_insert(expiration);
                        }
                        session.tabu_until.retain(|_, expiration| session.accepted_step < *expiration);
                        if current < session.elite_objective {
                            if relink {
                                slice.schedule_path_relink.elite_improvements =
                                    slice.schedule_path_relink.elite_improvements.saturating_add(1);
                            }
                            session.elite_objective = current;
                            session.elite_solution = session.state().to_solution();
                            session.last_improvement_step = session.accepted_step;
                            session.last_improvement_probe = session.probe_step;
                            incumbent_improvements = incumbent_improvements.saturating_add(1);
                            report(current);
                            capture_search_elite_shadow(&mut session, &mut slice, search_elite_shadow, stop);
                        }
                        slice.exact_probes_avoided = slice
                            .exact_probes_avoided
                            .saturating_add(u64::try_from(session.scan.end.saturating_sub(session.scan.cursor)).unwrap_or(u64::MAX));
                        clear_persistent_scan(&mut session);
                        continue 'search;
                    }
                    Ok(MoveOutcome::Rejected(rejection)) => {
                        if relink {
                            slice.schedule_path_relink.rollbacks = slice.schedule_path_relink.rollbacks.saturating_add(1);
                            match rejection {
                                MoveRejection::Cycle => {
                                    slice.schedule_path_relink.cycle_rejections =
                                        slice.schedule_path_relink.cycle_rejections.saturating_add(1);
                                }
                                MoveRejection::Window => {
                                    slice.schedule_path_relink.window_rejections =
                                        slice.schedule_path_relink.window_rejections.saturating_add(1);
                                }
                                _ => {
                                    slice.schedule_path_relink.other_rejections =
                                        slice.schedule_path_relink.other_rejections.saturating_add(1);
                                }
                            }
                        }
                        session.consecutive_direct_oracle_failures = session.consecutive_direct_oracle_failures.saturating_add(1);
                    }
                    Err(_) => {
                        if relink {
                            slice.schedule_path_relink.rollbacks = slice.schedule_path_relink.rollbacks.saturating_add(1);
                        }
                        break 'search;
                    }
                }
            }
            if probes >= probe_limit {
                break;
            }
            if session.scan.cursor >= session.scan.end {
                clear_persistent_scan(&mut session);
            }
            continue;
        }
        while session.scan.cursor < session.scan.end && probes < probe_limit {
            if checkpoint(stop).is_err() {
                break 'search;
            }
            let scan_index = session.scan.cursor;
            let movement = session.scan.movements[scan_index];
            let rank = session.scan.tsab_ranks.get(scan_index).copied().unwrap_or(scan_index);
            let Some(arcs) = session.state().move_arcs(movement) else {
                session.scan.cursor += 1;
                continue;
            };
            let tabu = arcs
                .added
                .into_iter()
                .flatten()
                .any(|arc| session.tabu_until.get(&arc).is_some_and(|&expiration| session.accepted_step < expiration));
            probes = probes.saturating_add(1);
            session.probe_step = session.probe_step.saturating_add(1);
            if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate {
                session.tsab_n5_work_since_restart = session.tsab_n5_work_since_restart.saturating_add(1);
                slice.schedule_tsab.delta_probes = slice.schedule_tsab.delta_probes.saturating_add(1);
                slice.schedule_tsab.additional_delta_probes =
                    slice.schedule_tsab.additional_delta_probes.saturating_add(u64::from(scan_index != 0));
            }
            let probe = session.state_mut().probe_move(movement, stop);
            let candidate = match probe {
                Ok(MoveProbe::Feasible { candidate, .. }) => candidate,
                Ok(MoveProbe::Rejected(_)) => {
                    session.scan.cursor += 1;
                    continue;
                }
                Err(_) => break 'search,
            };
            session.scan.cursor += 1;
            if tabu {
                slice.tabu_hits = slice.tabu_hits.saturating_add(1);
            }
            let aspiration = tabu_aspiration(tabu, candidate, session.elite_objective);
            if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate && tabu && !aspiration {
                slice.schedule_tsab.tabu_rejections = slice.schedule_tsab.tabu_rejections.saturating_add(1);
            }
            let candidate = ProbedScheduleCandidate { movement, arcs, objective: candidate, aspiration, rank };
            if session.scan.best_any.is_none_or(|incumbent| candidate_better(&session, candidate, incumbent)) {
                session.scan.best_any = Some(candidate);
            }
            if (!tabu || aspiration)
                && session.scan.best_admissible.is_none_or(|incumbent| candidate_better(&session, candidate, incumbent))
            {
                session.scan.best_admissible = Some(candidate);
            }
        }
        if session.scan.cursor < session.scan.end {
            break;
        }
        let (selected, forced, reset_tabu) =
            select_scanned_candidate(session.strategy, session.scan.best_admissible, session.scan.best_any);
        let Some(selected) = selected else {
            if reset_tabu {
                session.tabu_until.clear();
                slice.schedule_tsab.tabu_resets = slice.schedule_tsab.tabu_resets.saturating_add(1);
            }
            clear_persistent_scan(&mut session);
            continue;
        };
        if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate {
            slice.schedule_tsab.selections = slice.schedule_tsab.selections.saturating_add(1);
        }
        let acceptance = if selected.aspiration {
            MinimizingMoveAcceptance::StrictlyBelow(session.elite_objective)
        } else {
            MinimizingMoveAcceptance::Always
        };
        let outcome = session.state_mut().commit_probed_move(selected.movement, acceptance, stop);
        let Ok(MoveOutcome::Accepted { current, .. }) = outcome else {
            if outcome.is_err() {
                break;
            }
            clear_persistent_scan(&mut session);
            continue;
        };
        session.accepted_step = session.accepted_step.saturating_add(1);
        slice.tabu_steps = slice.tabu_steps.saturating_add(1);
        slice.tabu_forced_moves = slice.tabu_forced_moves.saturating_add(u64::from(forced));
        slice.tabu_aspirations = slice.tabu_aspirations.saturating_add(u64::from(selected.aspiration));
        if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate {
            slice.schedule_tsab.full_oracle_commits = slice.schedule_tsab.full_oracle_commits.saturating_add(1);
            slice.schedule_tsab.selected_shortlist_rank_sum =
                slice.schedule_tsab.selected_shortlist_rank_sum.saturating_add(u64::try_from(selected.rank).unwrap_or(u64::MAX));
            slice.schedule_tsab.aspirations = slice.schedule_tsab.aspirations.saturating_add(u64::from(selected.aspiration));
        }
        for arc in selected.arcs.removed.into_iter().flatten() {
            let expiration = session.accepted_step.saturating_add(arc_tenure(&session, arc));
            session.tabu_until.entry(arc).and_modify(|value| *value = (*value).max(expiration)).or_insert(expiration);
        }
        session.tabu_until.retain(|_, expiration| session.accepted_step < *expiration);
        if current < session.elite_objective {
            if session.strategy == ScheduleJsspSearchStrategy::TsabCandidate {
                slice.schedule_tsab.improving_commits = slice.schedule_tsab.improving_commits.saturating_add(1);
                if session.tsab_restart_epoch != 0 {
                    slice.schedule_tsab.post_restart_improvements = slice.schedule_tsab.post_restart_improvements.saturating_add(1);
                }
                slice.schedule_tsab.best_committed_objective =
                    Some(slice.schedule_tsab.best_committed_objective.map_or(current, |best| best.min(current)));
            }
            session.elite_objective = current;
            session.elite_solution = session.state().to_solution();
            session.last_improvement_step = session.accepted_step;
            session.last_improvement_probe = session.probe_step;
            incumbent_improvements = incumbent_improvements.saturating_add(1);
            report(current);
            capture_search_elite_shadow(&mut session, &mut slice, search_elite_shadow, stop);
        }
        clear_persistent_scan(&mut session);
    }

    let search_stopped = stop.load(Ordering::Acquire);
    if !session.tsab_fast_hard_failure && search_stopped {
        session.tsab_fast_validation_due_on_resume =
            session.tsab_fast_enabled && (session.tsab_fast_commits_since_validation != 0 || session.tsab_pending_elite.is_some());
    }
    if search_stopped
        && !finalization_stop.load(Ordering::Acquire)
        && session.strategy == ScheduleJsspSearchStrategy::TsabCandidate
        && session.tsab_fast_validation_due_on_resume
        && session.tsab_pending_elite.is_some()
    {
        let _ = validate_tsab_fast_checkpoint(
            &mut session,
            &mut total,
            &mut slice,
            finalization_stop,
            &mut incumbent_improvements,
            false,
            report,
        );
    }
    if session.requested_strategy.requests_tsab() {
        let warmup_work_units = active_start_probe.unwrap_or(probes).min(probes);
        session.tsab_legacy_warmup_work_steps = warmup_start.saturating_add(warmup_work_units).min(TSAB_LEGACY_WARMUP_WORK_UNITS);
        slice.schedule_tsab.legacy_warmup_work_steps = warmup_work_units;
        if let Some(active_start) = active_start_probe {
            let burst_work_units = probes.saturating_sub(active_start);
            slice.schedule_tsab.burst_work_units = burst_work_units;
            slice.schedule_tsab.active_boundaries = u64::from(burst_work_units != 0);
        }
    }
    if checkpoint(stop).is_ok() {
        session.completed_boundaries = session.completed_boundaries.saturating_add(1);
    }

    let resumed = slice.session_resumes != 0;
    if !session.tsab_fast_hard_failure {
        maybe_run_schedule_lns_shadow(&mut session, &mut slice, worker, resumed, schedule_lns_shadow_enabled, probes, stop);
    }
    add_state_metrics(&mut total, session.state_mut().take_metrics());
    let retain_session = !session.tsab_fast_hard_failure;
    let elite = session.elite_solution.clone();
    let (solution, mut metrics) =
        job_shop_result(Some(elite), construction_elapsed, first_feasible, total, incumbent_improvements, incumbent_injections);
    metrics.tabu_steps = slice.tabu_steps;
    metrics.tabu_hits = slice.tabu_hits;
    metrics.tabu_aspirations = slice.tabu_aspirations;
    metrics.tabu_forced_moves = slice.tabu_forced_moves;
    metrics.session_initializations = slice.session_initializations;
    metrics.session_resumes = slice.session_resumes;
    metrics.session_rebases = slice.session_rebases;
    metrics.island_profile = Some(session.profile.id);
    metrics.island_profile_mask = slice.island_profile_mask;
    metrics.baseline_island_profile_mask = slice.baseline_island_profile_mask;
    metrics.scored_island_profile_mask = slice.scored_island_profile_mask;
    metrics.reactive_restarts = slice.reactive_restarts;
    metrics.reactive_restart_dispatches = slice.reactive_restart_dispatches;
    metrics.reactive_restart_perturbations = slice.reactive_restart_perturbations;
    metrics.reactive_restart_rebuild_failures = slice.reactive_restart_rebuild_failures;
    metrics.island_scored_candidates = slice.island_scored_candidates;
    metrics.island_shortlisted_candidates = slice.island_shortlisted_candidates;
    metrics.approximate_candidates_generated = slice.approximate_candidates_generated;
    metrics.approximation_score_items = slice.approximation_score_items;
    metrics.approximation_sort_items = slice.approximation_sort_items;
    metrics.approximation_local_span_items = slice.approximation_local_span_items;
    metrics.approximation_elapsed = slice.approximation_elapsed;
    metrics.approximation_work_units = slice.approximation_work_units;
    metrics.exact_probes_avoided = slice.exact_probes_avoided;
    metrics.search_elite_snapshot_captures = slice.search_elite_snapshot_captures;
    metrics.search_elite_snapshot_interruptions = slice.search_elite_snapshot_interruptions;
    metrics.search_elite_snapshot_errors = slice.search_elite_snapshot_errors;
    metrics.search_elite_capture_worker_elapsed_sum = slice.search_elite_capture_worker_elapsed_sum;
    metrics.search_elite_snapshot_peak_heap_lower_bound_bytes = slice.search_elite_snapshot_peak_heap_lower_bound_bytes;
    metrics.schedule_lns_shadow_owner_worker_mask = slice.schedule_lns_shadow_owner_worker_mask;
    metrics.schedule_lns = slice.schedule_lns;
    metrics.schedule_lns_workspace_peak_bytes = slice.schedule_lns_workspace_peak_bytes;
    metrics.schedule_path_relink = slice.schedule_path_relink;
    metrics.schedule_tsab = slice.schedule_tsab;
    metrics.work_steps = probes;
    metrics.initial_objective = initial_objective;
    metrics.initial_dispatch_rule = initial_objective.map(|_| session.profile.initial_dispatch_name());
    (solution, metrics, retain_session.then_some(session))
}

#[cfg(test)]
pub(crate) fn audit_persistent_schedule_split(schedule: &Schedule, seed: u64, first: u64, second: u64) -> bool {
    audit_persistent_schedule_profile_split(schedule, seed, 0, 1, first, second)
}

/// Compare the protected historical profile 5 with an independent transcription of the historical
/// canonical-block tabu loop. This audit deliberately does not call
/// `prepare_persistent_scan`, `candidate_better`, `arc_tenure`, or the profile
/// restart code for its reference trajectory.
#[cfg(test)]
pub(crate) fn audit_persistent_schedule_baseline_reference(schedule: &Schedule, seed: u64, probe_limit: u64) -> bool {
    let stop = AtomicBool::new(false);
    let mut reports = Vec::new();
    let (solution, metrics, Some(mut session)) =
        solve_schedule_capped_persistent(schedule, seed, seed, 5, 7, &stop, probe_limit, None, None, false, &mut |objective| {
            reports.push(objective)
        })
    else {
        return false;
    };
    let Ok(Some(problem)) = JobShopProblem::recognize(schedule, &stop) else {
        return false;
    };
    let Ok(Some(mut reference)) = JobShopState::giffler_thompson(&problem, seed, DispatchRule::EarliestStart, &stop) else {
        return false;
    };
    reference.retain_canonical_critical_blocks_only();
    let ratio = u64::try_from(problem.operation_count() / problem.machine_count().max(1)).unwrap_or(u64::MAX);
    let tenure_base = ratio.isqrt().clamp(5, 32);
    let mut elite_objective = reference.makespan();
    let mut reference_reports = vec![elite_objective];
    let mut tabu_until = BTreeMap::new();
    let mut accepted_step = 0u64;
    let mut last_improvement_step = 0u64;
    let mut scan_epoch = 0u64;
    let mut probes = 0u64;
    let mut movements = Vec::new();
    let mut cursor = 0usize;
    let mut end = 0usize;
    let mut best_admissible = None;
    let mut best_any = None;

    while probes < probe_limit {
        if end == 0 {
            let stagnation = accepted_step.saturating_sub(last_improvement_step);
            let neighborhood = if stagnation >= tenure_base.saturating_mul(2) || scan_epoch % 4 == 3 {
                CriticalNeighborhood::N6
            } else {
                CriticalNeighborhood::N5
            };
            if reference.fill_canonical_critical_moves(neighborhood, &mut movements, &stop).is_err() || movements.is_empty() {
                break;
            }
            let offset = usize::try_from(mix64(
                seed ^ accepted_step.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ scan_epoch.wrapping_mul(0xd1b5_4a32_d192_ed03),
            ))
            .unwrap_or(usize::MAX)
                % movements.len();
            movements.rotate_left(offset);
            cursor = 0;
            end = movements.len().min(32);
            best_admissible = None;
            best_any = None;
            scan_epoch = scan_epoch.saturating_add(1);
        }
        while cursor < end && probes < probe_limit {
            let movement = movements[cursor];
            let Some(arcs) = reference.move_arcs(movement) else {
                cursor += 1;
                continue;
            };
            let tabu =
                arcs.added.into_iter().flatten().any(|arc| tabu_until.get(&arc).is_some_and(|&expiration| accepted_step < expiration));
            probes = probes.saturating_add(1);
            let Ok(MoveProbe::Feasible { candidate, .. }) = reference.probe_move(movement, &stop) else {
                cursor += 1;
                continue;
            };
            cursor += 1;
            let aspiration = tabu && candidate < elite_objective;
            let candidate = ProbedScheduleCandidate { movement, arcs, objective: candidate, aspiration, rank: cursor.saturating_sub(1) };
            if best_any.is_none_or(|incumbent: ProbedScheduleCandidate| {
                (candidate.objective, candidate.movement) < (incumbent.objective, incumbent.movement)
            }) {
                best_any = Some(candidate);
            }
            if (!tabu || aspiration)
                && best_admissible.is_none_or(|incumbent: ProbedScheduleCandidate| {
                    (candidate.objective, candidate.movement) < (incumbent.objective, incumbent.movement)
                })
            {
                best_admissible = Some(candidate);
            }
        }
        if cursor < end {
            break;
        }
        let Some(selected) = best_admissible.or(best_any) else {
            movements.clear();
            cursor = 0;
            end = 0;
            continue;
        };
        let acceptance =
            if selected.aspiration { MinimizingMoveAcceptance::StrictlyBelow(elite_objective) } else { MinimizingMoveAcceptance::Always };
        let Ok(MoveOutcome::Accepted { current, .. }) = reference.commit_probed_move(selected.movement, acceptance, &stop) else {
            movements.clear();
            cursor = 0;
            end = 0;
            continue;
        };
        accepted_step = accepted_step.saturating_add(1);
        for arc in selected.arcs.removed.into_iter().flatten() {
            let identity = u64::try_from(arc.machine).unwrap_or(u64::MAX).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ u64::try_from(arc.before).unwrap_or(u64::MAX).wrapping_mul(0xd1b5_4a32_d192_ed03)
                ^ u64::try_from(arc.after).unwrap_or(u64::MAX).wrapping_mul(0x94d0_49bb_1331_11eb);
            let width = tenure_base.saturating_add(1);
            let tenure = tenure_base.saturating_add(mix64(seed ^ accepted_step ^ identity) % width);
            let expiration = accepted_step.saturating_add(tenure);
            tabu_until.entry(arc).and_modify(|value| *value = (*value).max(expiration)).or_insert(expiration);
        }
        tabu_until.retain(|_, expiration| accepted_step < *expiration);
        if current < elite_objective {
            elite_objective = current;
            last_improvement_step = accepted_step;
            reference_reports.push(current);
        }
        movements.clear();
        cursor = 0;
        end = 0;
        best_admissible = None;
        best_any = None;
    }

    let historical_trajectory_matches = solution.objectives == vec![elite_objective]
        && reports == reference_reports
        && metrics.initial_objective == reference_reports.first().copied()
        && metrics.initial_dispatch_rule == Some("earliest-start")
        && metrics.baseline_island_profile_mask == 1 << 5
        && metrics.scored_island_profile_mask == 0
        && metrics.reactive_restarts == 0
        && metrics.island_scored_candidates == 0
        && session.state().machine_sequences() == reference.machine_sequences()
        && session.state().starts() == reference.starts()
        && session.elite_objective == elite_objective
        && session.accepted_step == accepted_step
        && session.last_improvement_step == last_improvement_step
        && session.scan_epoch == scan_epoch
        && session.tabu_until == tabu_until
        && session.scan.movements == movements
        && session.scan.cursor == cursor
        && session.scan.end == end
        && session.scan.best_admissible == best_admissible
        && session.scan.best_any == best_any;

    let epoch_before_rebase = session.scan_epoch;
    let injected = session.elite_solution.clone();
    session.elite_objective = i64::MAX;
    let (_, rebase_metrics, Some(rebased)) =
        solve_schedule_capped_persistent(schedule, seed, seed, 5, 7, &stop, 0, Some(&injected), Some(session), true, &mut |_| {})
    else {
        return false;
    };

    historical_trajectory_matches
        && epoch_before_rebase > 0
        && rebase_metrics.session_rebases == 1
        && rebased.scan_epoch == epoch_before_rebase
        && rebased.profile.uses_baseline_kernel()
}

#[cfg(test)]
pub(crate) fn audit_persistent_schedule_profile_split(
    schedule: &Schedule,
    seed: u64,
    worker: usize,
    workers: usize,
    first: u64,
    second: u64,
) -> bool {
    let stop = AtomicBool::new(false);
    let mut monolithic_reports = Vec::new();
    let (monolithic_solution, monolithic_metrics, monolithic_session) = solve_schedule_capped_persistent(
        schedule,
        seed,
        seed,
        worker,
        workers,
        &stop,
        first.saturating_add(second),
        None,
        None,
        true,
        &mut |objective| monolithic_reports.push(objective),
    );
    let mut split_reports = Vec::new();
    let (_, first_metrics, first_session) =
        solve_schedule_capped_persistent(schedule, seed, seed, worker, workers, &stop, first, None, None, true, &mut |objective| {
            split_reports.push(objective)
        });
    let (split_solution, second_metrics, split_session) = solve_schedule_capped_persistent(
        schedule,
        seed,
        seed,
        worker,
        workers,
        &stop,
        second,
        None,
        first_session,
        true,
        &mut |objective| split_reports.push(objective),
    );
    let (Some(monolithic_session), Some(split_session)) = (monolithic_session, split_session) else {
        return false;
    };

    let same_solution = monolithic_solution.objectives == split_solution.objectives
        && monolithic_solution.starts == split_solution.starts
        && monolithic_solution.presences == split_solution.presences
        && monolithic_solution.machines == split_solution.machines
        && monolithic_solution.modes == split_solution.modes;
    let same_session = monolithic_session.state().machine_sequences() == split_session.state().machine_sequences()
        && monolithic_session.elite_objective == split_session.elite_objective
        && monolithic_session.elite_solution.starts == split_session.elite_solution.starts
        && monolithic_session.tabu_until == split_session.tabu_until
        && monolithic_session.profile == split_session.profile
        && monolithic_session.accepted_step == split_session.accepted_step
        && monolithic_session.last_improvement_step == split_session.last_improvement_step
        && monolithic_session.probe_step == split_session.probe_step
        && monolithic_session.last_improvement_probe == split_session.last_improvement_probe
        && monolithic_session.last_restart_probe == split_session.last_restart_probe
        && monolithic_session.dispatch_restart_done == split_session.dispatch_restart_done
        && monolithic_session.restart_epoch == split_session.restart_epoch
        && monolithic_session.scan_epoch == split_session.scan_epoch
        && monolithic_session.consecutive_direct_oracle_failures == split_session.consecutive_direct_oracle_failures
        && monolithic_session.direct_oracle_work_credit == split_session.direct_oracle_work_credit
        && monolithic_session.scan.movements == split_session.scan.movements
        && monolithic_session.scan.scored_movements == split_session.scan.scored_movements
        && monolithic_session.scan.estimated_movements == split_session.scan.estimated_movements
        && monolithic_session.scan.cursor == split_session.scan.cursor
        && monolithic_session.scan.end == split_session.scan.end
        && monolithic_session.scan.best_admissible == split_session.scan.best_admissible
        && monolithic_session.scan.best_any == split_session.scan.best_any;
    let same_search_metrics = monolithic_metrics.moves_considered
        == first_metrics.moves_considered.saturating_add(second_metrics.moves_considered)
        && monolithic_metrics.candidates == first_metrics.candidates.saturating_add(second_metrics.candidates)
        && monolithic_metrics.construction_bucket_visits
            == first_metrics.construction_bucket_visits.saturating_add(second_metrics.construction_bucket_visits)
        && monolithic_metrics.construction_heap_pushes
            == first_metrics.construction_heap_pushes.saturating_add(second_metrics.construction_heap_pushes)
        && monolithic_metrics.construction_stale_pops
            == first_metrics.construction_stale_pops.saturating_add(second_metrics.construction_stale_pops)
        && monolithic_metrics.construction_heap_rebuilds
            == first_metrics.construction_heap_rebuilds.saturating_add(second_metrics.construction_heap_rebuilds)
        && monolithic_metrics.construction_heap_peak == first_metrics.construction_heap_peak.max(second_metrics.construction_heap_peak)
        && monolithic_metrics.moves_accepted == first_metrics.moves_accepted.saturating_add(second_metrics.moves_accepted)
        && monolithic_metrics.tabu_steps == first_metrics.tabu_steps.saturating_add(second_metrics.tabu_steps)
        && monolithic_metrics.tabu_hits == first_metrics.tabu_hits.saturating_add(second_metrics.tabu_hits)
        && monolithic_metrics.tabu_aspirations == first_metrics.tabu_aspirations.saturating_add(second_metrics.tabu_aspirations)
        && monolithic_metrics.tabu_forced_moves == first_metrics.tabu_forced_moves.saturating_add(second_metrics.tabu_forced_moves)
        && monolithic_metrics.reactive_restarts == first_metrics.reactive_restarts.saturating_add(second_metrics.reactive_restarts)
        && monolithic_metrics.reactive_restart_dispatches
            == first_metrics.reactive_restart_dispatches.saturating_add(second_metrics.reactive_restart_dispatches)
        && monolithic_metrics.reactive_restart_perturbations
            == first_metrics.reactive_restart_perturbations.saturating_add(second_metrics.reactive_restart_perturbations)
        && monolithic_metrics.reactive_restart_rebuild_failures
            == first_metrics.reactive_restart_rebuild_failures.saturating_add(second_metrics.reactive_restart_rebuild_failures)
        && monolithic_metrics.island_scored_candidates
            == first_metrics.island_scored_candidates.saturating_add(second_metrics.island_scored_candidates)
        && monolithic_metrics.island_shortlisted_candidates
            == first_metrics.island_shortlisted_candidates.saturating_add(second_metrics.island_shortlisted_candidates)
        && monolithic_metrics.approximate_candidates_generated
            == first_metrics.approximate_candidates_generated.saturating_add(second_metrics.approximate_candidates_generated)
        && monolithic_metrics.approximate_candidates_refined
            == first_metrics.approximate_candidates_refined.saturating_add(second_metrics.approximate_candidates_refined)
        && monolithic_metrics.approximate_candidates_certified
            == first_metrics.approximate_candidates_certified.saturating_add(second_metrics.approximate_candidates_certified)
        && monolithic_metrics.approximate_candidates_unknown
            == first_metrics.approximate_candidates_unknown.saturating_add(second_metrics.approximate_candidates_unknown)
        && monolithic_metrics.approximation_work_units
            == first_metrics.approximation_work_units.saturating_add(second_metrics.approximation_work_units)
        && monolithic_metrics.approximation_score_items
            == first_metrics.approximation_score_items.saturating_add(second_metrics.approximation_score_items)
        && monolithic_metrics.approximation_sort_items
            == first_metrics.approximation_sort_items.saturating_add(second_metrics.approximation_sort_items)
        && monolithic_metrics.approximation_local_span_items
            == first_metrics.approximation_local_span_items.saturating_add(second_metrics.approximation_local_span_items)
        && monolithic_metrics.direct_oracle_attempts
            == first_metrics.direct_oracle_attempts.saturating_add(second_metrics.direct_oracle_attempts)
        && monolithic_metrics.direct_oracle_accepts
            == first_metrics.direct_oracle_accepts.saturating_add(second_metrics.direct_oracle_accepts)
        && monolithic_metrics.direct_oracle_cycles
            == first_metrics.direct_oracle_cycles.saturating_add(second_metrics.direct_oracle_cycles)
        && monolithic_metrics.direct_oracle_windows
            == first_metrics.direct_oracle_windows.saturating_add(second_metrics.direct_oracle_windows)
        && monolithic_metrics.direct_oracle_objective_rejections
            == first_metrics.direct_oracle_objective_rejections.saturating_add(second_metrics.direct_oracle_objective_rejections)
        && monolithic_metrics.exact_probes_avoided
            == first_metrics.exact_probes_avoided.saturating_add(second_metrics.exact_probes_avoided)
        && monolithic_metrics.work_steps == first_metrics.work_steps.saturating_add(second_metrics.work_steps)
        && monolithic_metrics.island_profile == first_metrics.island_profile
        && first_metrics.island_profile == second_metrics.island_profile
        && monolithic_metrics.island_profile_mask == first_metrics.island_profile_mask
        && first_metrics.island_profile_mask == second_metrics.island_profile_mask
        && monolithic_metrics.session_initializations == 1
        && first_metrics.session_initializations == 1
        && second_metrics.session_initializations == 0
        && monolithic_metrics.session_resumes == 0
        && first_metrics.session_resumes == 0
        && second_metrics.session_resumes == 1
        && monolithic_metrics.initial_objective == first_metrics.initial_objective
        && monolithic_metrics.initial_objective.is_some()
        && second_metrics.initial_objective.is_none()
        && monolithic_metrics.initial_dispatch_rule == first_metrics.initial_dispatch_rule
        && monolithic_metrics.initial_dispatch_rule == Some(monolithic_session.profile.initial_dispatch_name())
        && second_metrics.initial_dispatch_rule.is_none();

    let direct_oracle_accounting = if monolithic_session.profile.uses_direct_oracle() {
        let budget = first.saturating_add(second);
        monolithic_metrics.work_steps <= budget
            && monolithic_metrics.work_steps
                == monolithic_metrics.direct_oracle_attempts.saturating_mul(4).saturating_add(monolithic_session.direct_oracle_work_credit)
            && monolithic_metrics.direct_oracle_attempts <= budget / 4
            && monolithic_metrics.moves_considered == monolithic_metrics.direct_oracle_attempts
            && monolithic_metrics.delta_evaluations == 0
            && monolithic_metrics.direct_oracle_attempts.saturating_add(monolithic_metrics.exact_probes_avoided)
                <= monolithic_metrics.approximate_candidates_refined
            && monolithic_metrics
                .approximate_candidates_refined
                .saturating_sub(monolithic_metrics.direct_oracle_attempts.saturating_add(monolithic_metrics.exact_probes_avoided))
                <= 3
            && monolithic_metrics.approximate_candidates_refined
                == monolithic_metrics.approximate_candidates_certified.saturating_add(monolithic_metrics.approximate_candidates_unknown)
            && monolithic_metrics.approximation_work_units
                >= monolithic_metrics
                    .approximate_candidates_generated
                    .saturating_add(monolithic_metrics.approximate_candidates_refined.saturating_mul(2))
            && monolithic_metrics.approximation_score_items == monolithic_metrics.approximate_candidates_generated
            && monolithic_metrics.approximation_sort_items
                >= monolithic_metrics.approximation_score_items.saturating_add(monolithic_metrics.approximate_candidates_refined)
            && monolithic_metrics.approximation_local_span_items >= monolithic_metrics.approximate_candidates_refined.saturating_mul(2)
            && monolithic_metrics.tabu_forced_moves == 0
    } else {
        true
    };
    let deferred_direct_preparation = !monolithic_session.profile.uses_direct_oracle()
        || first >= 4
        || (first_metrics.work_steps == first
            && first_metrics.direct_oracle_attempts == 0
            && first_metrics.approximate_candidates_generated == 0
            && first_metrics.approximate_candidates_refined == 0
            && first_metrics.approximation_score_items == 0
            && first_metrics.approximation_sort_items == 0
            && first_metrics.approximation_local_span_items == 0
            && first_metrics.approximation_work_units == 0);

    same_solution
        && same_session
        && same_search_metrics
        && direct_oracle_accounting
        && deferred_direct_preparation
        && monolithic_reports == split_reports
}

#[cfg(test)]
pub(crate) fn audit_schedule_island_profile(worker: usize, workers: usize) -> (usize, &'static str, &'static str, usize, u64, u64) {
    let profile = ScheduleIslandProfile::for_worker(worker, workers);
    (
        profile.id,
        profile.initial_dispatch_name(),
        profile.restart_dispatch_name(),
        profile.scan_width,
        profile.scaled_tenure(32),
        profile.restart_probe_interval(profile.scaled_tenure(32)),
    )
}

/// Solve an interval-scheduling subproblem by local search. The decisions are
/// each interval's start time and, for flexible (moded) intervals, its mode
/// (which sets the duration and machine). Moves shift a start or switch a mode.
pub(crate) fn solve_schedule(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    solve_schedule_capped(sched, seed, stop, u64::MAX, false, None, report)
}

pub(crate) fn solve_schedule_capped(
    sched: &Schedule,
    seed: u64,
    stop: &AtomicBool,
    max_iterations: u64,
    repeat_until_stopped: bool,
    initial_incumbent: Option<&CollectionSolution>,
    report: &mut dyn FnMut(i64),
) -> (CollectionSolution, ScheduleConstructionMetrics) {
    if !matches!(machines_are_representable(sched, stop), Ok(true)) {
        return (unknown_solution(), generic_schedule_metrics(Duration::ZERO, None, 0, 0, 0, 0));
    }
    if let Some(result) = solve_job_shop(sched, seed, stop, max_iterations, repeat_until_stopped, initial_incumbent, report) {
        return result;
    }
    if let Some(result) = solve_resource_schedule(sched, seed, stop, max_iterations, repeat_until_stopped, initial_incumbent, report) {
        return result;
    }

    let construction_started = Instant::now();
    let n = sched.intervals.len();
    let injected = complete_schedule_incumbent(sched, initial_incumbent);
    let (constructed, candidates, constructed_first_feasible) = construct_schedule(sched, seed, stop, report);
    let construction_elapsed = construction_started.elapsed();
    let Ok(Some(ConstructedIncumbent {
        schedule: ConstructedSchedule { mut starts, mut chosen, mut present },
        durations: mut dur,
        machines: mut mach,
        score: mut cur,
        solution: mut best_solution,
    })) = constructed
    else {
        return (injected.unwrap_or_else(unknown_solution), generic_schedule_metrics(construction_elapsed, None, candidates, 0, 0, 0));
    };
    let first_feasible = best_solution.feasible.then_some(constructed_first_feasible.unwrap_or(construction_elapsed));
    let mut moves_considered = 0u64;
    let mut moves_accepted = 0u64;
    let mut incumbent_improvements = u64::from(first_feasible.is_some());
    let mut incumbent_score = cur;
    if let Some(injected) = injected {
        let injected_score = (0, injected.objectives.first().copied().unwrap_or_default());
        if injected_score < incumbent_score {
            incumbent_score = injected_score;
            best_solution = injected;
        }
    }

    // Large schedules stay on the compact constructor. The pairwise shift
    // engine is retained only for small instances until the critical-path
    // neighbourhoods replace it.
    if n > 48 || checkpoint(stop).is_err() {
        return (
            best_solution,
            generic_schedule_metrics(
                construction_elapsed,
                first_feasible,
                candidates,
                moves_considered,
                moves_accepted,
                incumbent_improvements,
            ),
        );
    }

    const RESTART_AFTER: u64 = 25;
    let mut search_best = cur;
    let mut since_improve = 0u64;
    let mut iter = 0u64;

    'search: loop {
        if checkpoint(stop).is_err() || moves_considered >= max_iterations {
            break;
        }
        iter += 1;
        let mut moved = false;
        'scan: for i in 0..n {
            if checkpoint(stop).is_err() {
                break 'search;
            }

            // (a) Shift the start under the current mode.
            let candidates = match schedule_candidates(sched, &chosen, &dur, &starts, &present, i, stop) {
                Ok(candidates) => candidates,
                Err(_) => break 'search,
            };
            for t in candidates {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                if t == starts[i] {
                    continue;
                }
                if moves_considered >= max_iterations {
                    break 'search;
                }
                moves_considered = moves_considered.saturating_add(1);
                let old = starts[i];
                starts[i] = t;
                let trial = match schedule_score(sched, &chosen, &dur, &mach, &starts, &present, stop) {
                    Ok(trial) => trial,
                    Err(_) => {
                        starts[i] = old;
                        break 'search;
                    }
                };
                if trial < cur {
                    if trial < incumbent_score {
                        let snapshot = match snapshot_solution(sched, &chosen, &starts, &present, trial, stop) {
                            Ok(snapshot) => snapshot,
                            Err(_) => {
                                starts[i] = old;
                                break 'search;
                            }
                        };
                        incumbent_score = trial;
                        best_solution = snapshot;
                        incumbent_improvements = incumbent_improvements.saturating_add(1);
                    }
                    cur = trial;
                    moves_accepted = moves_accepted.saturating_add(1);
                    moved = true;
                    break 'scan;
                }
                starts[i] = old;
            }

            // (b) Switch this interval's mode (machine / duration).
            let modes = sched.intervals[i].modes.len();
            for m in 0..modes {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                if m == chosen[i] {
                    continue;
                }
                if moves_considered >= max_iterations {
                    break 'search;
                }
                moves_considered = moves_considered.saturating_add(1);
                let (old_mode, old_duration, old_machine, old_start) = (chosen[i], dur[i], mach[i], starts[i]);
                chosen[i] = m;
                dur[i] = iv_duration(&sched.intervals[i], m);
                mach[i] = iv_machine(&sched.intervals[i], m);
                let (start_min, start_max) = iv_start_window(&sched.intervals[i], m);
                starts[i] = old_start.clamp(start_min, start_max);
                let trial = match schedule_score(sched, &chosen, &dur, &mach, &starts, &present, stop) {
                    Ok(trial) => trial,
                    Err(_) => {
                        chosen[i] = old_mode;
                        dur[i] = old_duration;
                        mach[i] = old_machine;
                        starts[i] = old_start;
                        break 'search;
                    }
                };
                if trial < cur {
                    if trial < incumbent_score {
                        let snapshot = match snapshot_solution(sched, &chosen, &starts, &present, trial, stop) {
                            Ok(snapshot) => snapshot,
                            Err(_) => {
                                chosen[i] = old_mode;
                                dur[i] = old_duration;
                                mach[i] = old_machine;
                                starts[i] = old_start;
                                break 'search;
                            }
                        };
                        incumbent_score = trial;
                        best_solution = snapshot;
                        incumbent_improvements = incumbent_improvements.saturating_add(1);
                    }
                    cur = trial;
                    moves_accepted = moves_accepted.saturating_add(1);
                    moved = true;
                    break 'scan;
                }
                chosen[i] = old_mode;
                dur[i] = old_duration;
                mach[i] = old_machine;
                starts[i] = old_start;
            }

            if sched.intervals[i].optional {
                if checkpoint(stop).is_err() || moves_considered >= max_iterations {
                    break 'search;
                }
                moves_considered = moves_considered.saturating_add(1);
                present[i] = !present[i];
                let trial = match schedule_score(sched, &chosen, &dur, &mach, &starts, &present, stop) {
                    Ok(trial) => trial,
                    Err(_) => {
                        present[i] = !present[i];
                        break 'search;
                    }
                };
                if trial < cur {
                    if trial < incumbent_score {
                        let snapshot = match snapshot_solution(sched, &chosen, &starts, &present, trial, stop) {
                            Ok(snapshot) => snapshot,
                            Err(_) => {
                                present[i] = !present[i];
                                break 'search;
                            }
                        };
                        incumbent_score = trial;
                        best_solution = snapshot;
                        incumbent_improvements = incumbent_improvements.saturating_add(1);
                    }
                    cur = trial;
                    moves_accepted = moves_accepted.saturating_add(1);
                    moved = true;
                    break 'scan;
                }
                present[i] = !present[i];
            }
        }
        if moved {
            continue;
        }

        // Local optimum: record, then kick or restart.
        if cur < search_best {
            search_best = cur;
            if cur.0 == 0 && checkpoint(stop).is_ok() {
                report(cur.1);
            }
            since_improve = 0;
        } else {
            since_improve += 1;
        }
        if moves_considered >= max_iterations {
            break;
        }
        moves_considered = moves_considered.saturating_add(1);
        if since_improve >= RESTART_AFTER {
            // Random modes, then precedence-earliest starts with jitter.
            for (i, selected) in chosen.iter_mut().enumerate() {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                let modes = sched.intervals[i].modes.len().max(1);
                *selected = (mix64(seed ^ mix64(iter ^ 0x9e37).wrapping_add(i as u64)) % modes as u64) as usize;
            }
            (dur, mach) = match mode_view(sched, &chosen, stop) {
                Ok(view) => view,
                Err(_) => break 'search,
            };
            for (i, interval) in sched.intervals.iter().enumerate() {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                present[i] = !interval.optional || mix64(seed ^ iter.wrapping_add(i as u64)) & 1 == 1;
            }
            starts = match earliest_starts(sched, &chosen, &dur, &present, stop) {
                Ok(starts) => starts,
                Err(_) => break 'search,
            };
            for (i, start) in starts.iter_mut().enumerate() {
                if checkpoint(stop).is_err() {
                    break 'search;
                }
                let window = iv_start_window(&sched.intervals[i], chosen[i]);
                *start = random_start(window, mix64(seed ^ mix64(iter).wrapping_add(i as u64)));
            }
            since_improve = 0;
        } else if n > 0 {
            if checkpoint(stop).is_err() {
                break;
            }
            let i = (mix64(seed ^ mix64(iter)) % n as u64) as usize;
            starts[i] = random_start(iv_start_window(&sched.intervals[i], chosen[i]), mix64(seed ^ mix64(iter ^ 0x5151)));
        }
        cur = match schedule_score(sched, &chosen, &dur, &mach, &starts, &present, stop) {
            Ok(score) => score,
            Err(_) => break,
        };
        moves_accepted = moves_accepted.saturating_add(1);
        if cur < incumbent_score {
            let snapshot = match snapshot_solution(sched, &chosen, &starts, &present, cur, stop) {
                Ok(snapshot) => snapshot,
                Err(_) => break,
            };
            incumbent_score = cur;
            best_solution = snapshot;
            incumbent_improvements = incumbent_improvements.saturating_add(1);
        }
    }

    (
        best_solution,
        generic_schedule_metrics(
            construction_elapsed,
            first_feasible,
            candidates,
            moves_considered,
            moves_accepted,
            incumbent_improvements,
        ),
    )
}
