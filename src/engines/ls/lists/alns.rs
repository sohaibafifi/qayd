use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use smallvec::SmallVec;

use super::incremental::{EvalScratch, InsertView, RemoveView};
use super::local_search::{
    active_list_indices, full_score_raw, score_regret, score_with_replaced_list, score_with_replacements_raw, tier_values,
    CandidateNeighbors, ListScore, PerList, Score, ScoreRegret, State, TrialList,
};
use super::metrics::{AdaptiveOperatorMetrics, AlnsSearchMetrics};
use super::moves::{apply_move, best_improving_move, score_move_raw, trial_list_score_view, Move, MoveScoreWorkspace, SearchMemory};
use crate::mix64;
use crate::model::list::{CollectionModel, GlobalConstraint, Iterable, ReduceOp, Reduction};

const DESTROY_OPERATORS: usize = 4;
const REPAIR_OPERATORS: usize = 4;
const WEIGHT_SCALE: u64 = 1_000;
const MIN_WEIGHT: u64 = 100;
const COST_MIN_WEIGHT: u64 = 1;
const COST_RATE_SCALE: u64 = 1 << 14;
const COST_EXPLORATION_REWARD: u64 = WEIGHT_SCALE;
const MAX_OBSERVED_REWARD: u64 = 20_000;
const RELATIVE_EXPLORATION_DIVISOR: u64 = 32;
const FORCED_EXPLORATION_PERIOD: u64 = 32;
const MAX_REPAIR_PLACEMENTS: usize = 96;
const REGRET_ITEM_SAMPLE: usize = 8;

pub(super) struct CpuTimer {
    wall: Instant,
    cpu: Option<u64>,
}

impl CpuTimer {
    pub(super) fn start() -> Self {
        Self { wall: Instant::now(), cpu: thread_cpu_nanos() }
    }

    pub(super) fn elapsed_nanos(&self) -> u64 {
        let wall = || u64::try_from(self.wall.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.cpu.zip(thread_cpu_nanos()).map_or_else(wall, |(before, after)| after.saturating_sub(before))
    }
}

#[cfg(unix)]
fn thread_cpu_nanos() -> Option<u64> {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `time` is a valid writable timespec and the clock id does not
    // retain its pointer. Failure is represented as `None` and falls back to
    // monotonic wall time in `CpuTimer`.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) } != 0 {
        return None;
    }
    let seconds = u64::try_from(time.tv_sec).ok()?;
    let nanos = u64::try_from(time.tv_nsec).ok()?;
    Some(seconds.saturating_mul(1_000_000_000).saturating_add(nanos))
}

#[cfg(not(unix))]
fn thread_cpu_nanos() -> Option<u64> {
    None
}

#[derive(Clone, Copy)]
pub(super) enum SearchProfile {
    Sequential,
    Intensify,
    Balanced,
    Diversify,
}

impl SearchProfile {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Intensify => "intensify",
            Self::Balanced => "balanced",
            Self::Diversify => "diversify",
        }
    }

    pub(super) const fn candidate_limit(self) -> usize {
        match self {
            Self::Sequential => 24,
            Self::Intensify => 16,
            Self::Balanced => 32,
            Self::Diversify => 48,
        }
    }

    pub(super) const fn diversify_initial_descent(self) -> bool {
        !matches!(self, Self::Sequential)
    }

    pub(super) const fn use_arc_gls(self) -> bool {
        true
    }

    pub(super) const fn elite_restart_period(self) -> u64 {
        match self {
            Self::Sequential => 24,
            Self::Intensify => 8,
            Self::Balanced => 16,
            Self::Diversify => 32,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum DestroyOperator {
    Shaw = 0,
    Segments = 1,
    Worst = 2,
    Route = 3,
}

impl DestroyOperator {
    const ALL: [Self; DESTROY_OPERATORS] = [Self::Shaw, Self::Segments, Self::Worst, Self::Route];

    const fn name(self) -> &'static str {
        match self {
            Self::Shaw => "shaw",
            Self::Segments => "segments",
            Self::Worst => "worst",
            Self::Route => "route",
        }
    }

    const fn cost_model(self) -> CostModel {
        match self {
            Self::Shaw => CostModel::new(6, 2, 1, 2),
            Self::Segments => CostModel::new(3, 1, 1, 1),
            Self::Worst => CostModel::new(8, 2, 2, 5),
            Self::Route => CostModel::new(4, 1, 1, 2),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum RepairOperator {
    Greedy = 0,
    Regret2 = 1,
    Regret3 = 2,
    BlinkRegret2 = 3,
}

#[derive(Clone, Copy)]
pub(super) struct AlnsChoice {
    pub(super) destroy: DestroyOperator,
    pub(super) repair: RepairOperator,
    pub(super) target: usize,
}

impl RepairOperator {
    const ALL: [Self; REPAIR_OPERATORS] = [Self::Greedy, Self::Regret2, Self::Regret3, Self::BlinkRegret2];

    const fn name(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Regret2 => "regret-2",
            Self::Regret3 => "regret-3",
            Self::BlinkRegret2 => "blink-regret-2",
        }
    }

    const fn regret(self) -> Option<(usize, bool)> {
        match self {
            Self::Greedy => None,
            Self::Regret2 => Some((2, false)),
            Self::Regret3 => Some((3, false)),
            Self::BlinkRegret2 => Some((2, true)),
        }
    }

    const fn cost_model(self) -> CostModel {
        match self {
            Self::Greedy => CostModel::new(4, 1, 1, 4),
            Self::Regret2 => CostModel::new(6, 2, 2, 8),
            Self::Regret3 => CostModel::new(8, 2, 2, 11),
            Self::BlinkRegret2 => CostModel::new(7, 2, 2, 7),
        }
    }
}

#[derive(Clone, Copy)]
struct CostModel {
    fixed: u64,
    work: u64,
    generated: u64,
    evaluated: u64,
}

impl CostModel {
    const fn new(fixed: u64, work: u64, generated: u64, evaluated: u64) -> Self {
        Self { fixed, work, generated, evaluated }
    }

    fn estimate(self, work: u64, generated: u64, evaluated: u64) -> u64 {
        self.fixed
            .saturating_add(self.work.saturating_mul(work))
            .saturating_add(self.generated.saturating_mul(generated))
            .saturating_add(self.evaluated.saturating_mul(evaluated))
            .max(1)
    }
}

#[derive(Clone, Copy)]
struct OperatorState {
    weight: u64,
    pending_reward: u64,
    pending_uses: u64,
    pending_cost: u64,
    uses: u64,
    accepted: u64,
    improvements: u64,
    global_bests: u64,
    generated: u64,
    evaluated: u64,
    work_units: u64,
    cpu_nanos: u64,
    positive_rewards: u64,
}

impl Default for OperatorState {
    fn default() -> Self {
        Self {
            weight: WEIGHT_SCALE,
            pending_reward: 0,
            pending_uses: 0,
            pending_cost: 0,
            uses: 0,
            accepted: 0,
            improvements: 0,
            global_bests: 0,
            generated: 0,
            evaluated: 0,
            work_units: 0,
            cpu_nanos: 0,
            positive_rewards: 0,
        }
    }
}

#[derive(Clone, Copy)]
enum AdaptationMode {
    PerUse,
    DeterministicCostRate,
}

struct AlnsConfig {
    reaction_milli: u64,
    segment_length: u64,
    initial_temperature: f64,
    cooling: f64,
    late_window: usize,
    min_destroy_percent: usize,
    max_destroy_percent: usize,
}

impl AlnsConfig {
    fn for_profile(items: usize, profile: SearchProfile) -> Self {
        let root = integer_sqrt(items.max(1));
        let (temperature, cooling, late_window, destroy_min, destroy_max) = match profile {
            SearchProfile::Sequential | SearchProfile::Balanced => (0.12, 0.997, root.saturating_mul(2), 6, 45),
            SearchProfile::Intensify => (0.05, 0.995, root, 3, 28),
            SearchProfile::Diversify => (0.30, 0.999, root.saturating_mul(4), 15, 65),
        };
        Self {
            reaction_milli: 200,
            segment_length: root as u64,
            initial_temperature: temperature,
            cooling,
            late_window,
            min_destroy_percent: destroy_min,
            max_destroy_percent: destroy_max,
        }
    }
}

fn integer_sqrt(value: usize) -> usize {
    if value < 2 {
        return value;
    }
    let mut lo = 1usize;
    let mut hi = value.min(1usize << (usize::BITS / 2));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if mid <= value / mid {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

#[derive(Clone, Copy)]
pub(super) enum AcceptanceKind {
    Improving,
    Late,
    Annealing,
    Rejected,
}

pub(super) struct AlnsController {
    config: AlnsConfig,
    adaptation: AdaptationMode,
    destroy: [OperatorState; DESTROY_OPERATORS],
    repair: [OperatorState; REPAIR_OPERATORS],
    temperature: f64,
    late_scores: Vec<Score>,
    late_cursor: usize,
    segment_uses: u64,
    metrics: AlnsSearchMetrics,
}

impl AlnsController {
    pub(super) fn new(items: usize, initial: &Score) -> Self {
        Self::new_profile(items, initial, SearchProfile::Sequential)
    }

    pub(super) fn new_profile(items: usize, initial: &Score, profile: SearchProfile) -> Self {
        Self::with_adaptation(items, initial, profile, AdaptationMode::PerUse)
    }

    pub(super) fn new_routing_profile(items: usize, initial: &Score, profile: SearchProfile) -> Self {
        Self::with_adaptation(items, initial, profile, AdaptationMode::DeterministicCostRate)
    }

    fn with_adaptation(items: usize, initial: &Score, profile: SearchProfile, adaptation: AdaptationMode) -> Self {
        let config = AlnsConfig::for_profile(items, profile);
        Self {
            temperature: config.initial_temperature,
            late_scores: vec![initial.clone(); config.late_window],
            config,
            adaptation,
            destroy: [OperatorState::default(); DESTROY_OPERATORS],
            repair: [OperatorState::default(); REPAIR_OPERATORS],
            late_cursor: 0,
            segment_uses: 0,
            metrics: AlnsSearchMetrics::default(),
        }
    }

    pub(super) fn select(&self, seed: u64, iteration: u64) -> (DestroyOperator, RepairOperator) {
        if matches!(self.adaptation, AdaptationMode::DeterministicCostRate) {
            let attempt = self.metrics.iterations;
            let position = attempt % FORCED_EXPLORATION_PERIOD;
            if position < DESTROY_OPERATORS as u64 {
                let destroy = position as usize;
                let cycle = attempt / FORCED_EXPLORATION_PERIOD;
                let repair = (destroy.saturating_mul(3).saturating_add(cycle as usize)) % REPAIR_OPERATORS;
                return (DestroyOperator::ALL[destroy], RepairOperator::ALL[repair]);
            }
        }
        let relative_floor = matches!(self.adaptation, AdaptationMode::DeterministicCostRate).then_some(RELATIVE_EXPLORATION_DIVISOR);
        let destroy = roulette(&self.destroy, 4, seed ^ mix64(iteration), relative_floor);
        let repair = roulette(&self.repair, 4, seed ^ mix64(iteration ^ 0xA076_1D64_78BD_642F), relative_floor);
        (DestroyOperator::ALL[destroy], RepairOperator::ALL[repair])
    }

    pub(super) fn choose(&self, total: usize, stagnant: u64, seed: u64, iteration: u64) -> AlnsChoice {
        let (destroy, repair) = self.select(seed, iteration);
        AlnsChoice { destroy, repair, target: self.destroy_size(total, stagnant, seed) }
    }

    pub(super) fn destroy_size(&self, total: usize, stagnant: u64, seed: u64) -> usize {
        if total == 0 {
            return 0;
        }
        let horizon = self.config.late_window as u64;
        let span = self.config.max_destroy_percent - self.config.min_destroy_percent;
        let growth = u64::try_from(span).unwrap_or(u64::MAX).saturating_mul(stagnant) / stagnant.saturating_add(horizon).max(1);
        let jitter_span = (span / 8).max(1);
        let jitter = (mix64(seed) % jitter_span as u64) as usize;
        let percent =
            self.config.min_destroy_percent.saturating_add(growth as usize).saturating_add(jitter).min(self.config.max_destroy_percent);
        total.saturating_mul(percent).div_ceil(100).clamp(1, total)
    }

    pub(super) fn accept(
        &mut self,
        current_guided: &Score,
        candidate_guided: &Score,
        current_raw: &Score,
        candidate_raw: &Score,
        seed: u64,
        iteration: u64,
    ) -> AcceptanceKind {
        let kind = if candidate_guided < current_guided {
            AcceptanceKind::Improving
        } else if candidate_raw <= &self.late_scores[self.late_cursor] {
            AcceptanceKind::Late
        } else {
            let worsening = normalized_worsening(current_guided, candidate_guided);
            let probability = (-worsening / self.temperature.max(1.0e-9)).exp();
            let sample = (mix64(seed ^ mix64(iteration ^ 0xE703_7ED1_A0B4_28DB)) >> 11) as f64 / ((1u64 << 53) as f64);
            if sample < probability {
                AcceptanceKind::Annealing
            } else {
                AcceptanceKind::Rejected
            }
        };

        let accepted = !matches!(kind, AcceptanceKind::Rejected);
        if accepted {
            self.temperature = (self.temperature * self.config.cooling).max(self.config.initial_temperature * 0.01);
        } else {
            let reheat = 1.0 + 1.0 / self.config.late_window as f64;
            self.temperature = (self.temperature * reheat).min(self.config.initial_temperature);
        }
        self.late_scores[self.late_cursor] = if accepted { candidate_raw.clone() } else { current_raw.clone() };
        self.late_cursor = (self.late_cursor + 1) % self.late_scores.len();
        kind
    }

    /// Advance acceptance memory after a semantic prefix guard rejects a
    /// candidate before annealing. Routing uses this for a strict fleet-first
    /// objective, where temporary distance gains may never buy an extra route.
    pub(super) fn reject_semantic_worsening(&mut self, current_raw: &Score) -> AcceptanceKind {
        let reheat = 1.0 + 1.0 / self.config.late_window as f64;
        self.temperature = (self.temperature * reheat).min(self.config.initial_temperature);
        self.late_scores[self.late_cursor] = current_raw.clone();
        self.late_cursor = (self.late_cursor + 1) % self.late_scores.len();
        AcceptanceKind::Rejected
    }

    pub(super) fn record(
        &mut self,
        destroy: DestroyOperator,
        repair: RepairOperator,
        acceptance: AcceptanceKind,
        improved_current: bool,
        global_best: bool,
    ) {
        self.record_with_work(destroy, repair, acceptance, improved_current, global_best, 1, 1, 0);
    }

    /// Update adaptive weights from deterministic work cost. Thread CPU is
    /// retained only as profiler telemetry.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_with_work(
        &mut self,
        destroy: DestroyOperator,
        repair: RepairOperator,
        acceptance: AcceptanceKind,
        improved_current: bool,
        global_best: bool,
        generated: u64,
        evaluated: u64,
        cpu_nanos: u64,
    ) {
        self.record_with_split(
            destroy,
            repair,
            acceptance,
            improved_current,
            global_best,
            generated,
            evaluated,
            cpu_nanos,
            generated,
            evaluated,
            cpu_nanos,
            true,
            true,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn record_with_split(
        &mut self,
        destroy: DestroyOperator,
        repair: RepairOperator,
        acceptance: AcceptanceKind,
        improved_current: bool,
        global_best: bool,
        destroy_generated: u64,
        destroy_evaluated: u64,
        destroy_cpu_nanos: u64,
        repair_generated: u64,
        repair_evaluated: u64,
        repair_cpu_nanos: u64,
        destroy_executed: bool,
        repair_executed: bool,
    ) {
        let accepted = !matches!(acceptance, AcceptanceKind::Rejected);
        let reward = if global_best {
            20_000
        } else if improved_current {
            8_000
        } else if accepted {
            3_000
        } else {
            0
        };
        // Selection is the accounting boundary: every ALNS slice contributes
        // exactly one use to both selected operators, even when interruption
        // or exhaustion prevents one of them from executing. Selection-only
        // observations remain visible in telemetry, but must not enter the
        // adaptive sample. In particular, pricing skipped work at one unit
        // would make it look maximally productive.
        if destroy_executed {
            let destroy_work = destroy_generated.saturating_add(destroy_evaluated);
            let destroy_cost = destroy.cost_model().estimate(destroy_work, destroy_generated, destroy_evaluated);
            update_operator(
                &mut self.destroy[destroy as usize],
                reward,
                accepted,
                improved_current,
                global_best,
                destroy_generated,
                destroy_evaluated,
                destroy_work,
                destroy_cost,
                destroy_cpu_nanos,
            );
            if matches!(self.adaptation, AdaptationMode::DeterministicCostRate) && reward == 0 {
                apply_unproductive_cost_weight(&mut self.destroy[destroy as usize], destroy_cost);
            }
        } else {
            record_skipped_operator(&mut self.destroy[destroy as usize]);
        }

        if repair_executed {
            let repair_work = repair_generated.saturating_add(repair_evaluated);
            let repair_cost = repair.cost_model().estimate(repair_work, repair_generated, repair_evaluated);
            update_operator(
                &mut self.repair[repair as usize],
                reward,
                accepted,
                improved_current,
                global_best,
                repair_generated,
                repair_evaluated,
                repair_work,
                repair_cost,
                repair_cpu_nanos,
            );
            if matches!(self.adaptation, AdaptationMode::DeterministicCostRate) && reward == 0 {
                apply_unproductive_cost_weight(&mut self.repair[repair as usize], repair_cost);
            }
        } else {
            record_skipped_operator(&mut self.repair[repair as usize]);
        }
        self.segment_uses = self.segment_uses.saturating_add(1);
        self.metrics.iterations = self.metrics.iterations.saturating_add(1);
        if accepted {
            self.metrics.accepted = self.metrics.accepted.saturating_add(1);
        }
        if improved_current {
            self.metrics.improving = self.metrics.improving.saturating_add(1);
        }
        if global_best {
            self.metrics.global_bests = self.metrics.global_bests.saturating_add(1);
        }
        match acceptance {
            AcceptanceKind::Annealing => {
                self.metrics.simulated_annealing_accepts = self.metrics.simulated_annealing_accepts.saturating_add(1);
            }
            AcceptanceKind::Late => {
                self.metrics.late_acceptance_accepts = self.metrics.late_acceptance_accepts.saturating_add(1);
            }
            AcceptanceKind::Improving | AcceptanceKind::Rejected => {}
        }
        if self.segment_uses >= self.config.segment_length {
            react(&mut self.destroy, self.config.reaction_milli, self.adaptation);
            react(&mut self.repair, self.config.reaction_milli, self.adaptation);
            self.segment_uses = 0;
        }
    }

    pub(super) fn record_failed(&mut self, destroy: DestroyOperator, repair: RepairOperator) {
        self.record(destroy, repair, AcceptanceKind::Rejected, false, false);
    }

    pub(super) fn record_bounded(
        &mut self,
        choice: AlnsChoice,
        acceptance: AcceptanceKind,
        improved_current: bool,
        global_best: bool,
        run: &AlnsBuildRun,
    ) {
        self.record_with_split(
            choice.destroy,
            choice.repair,
            acceptance,
            improved_current,
            global_best,
            run.destroy_generated,
            run.destroy_evaluated,
            run.destroy_cpu_nanos,
            run.repair_generated,
            run.repair_evaluated,
            run.repair_cpu_nanos,
            run.destroy_executed,
            run.repair_executed,
        );
    }

    pub(super) fn record_failed_bounded(&mut self, choice: AlnsChoice, run: &AlnsBuildRun) {
        self.record_bounded(choice, AcceptanceKind::Rejected, false, false, run);
    }

    pub(super) fn record_gls(&mut self, penalties: usize) {
        if penalties > 0 {
            self.metrics.gls_updates = self.metrics.gls_updates.saturating_add(1);
            self.metrics.gls_penalties = self.metrics.gls_penalties.saturating_add(penalties as u64);
        }
    }

    pub(super) fn record_shared_publication(&mut self) {
        self.metrics.shared_publications = self.metrics.shared_publications.saturating_add(1);
    }

    pub(super) fn record_shared_injection(&mut self) {
        self.metrics.shared_injections = self.metrics.shared_injections.saturating_add(1);
    }

    pub(super) fn reset_after_injection(&mut self, score: &Score) {
        self.temperature = self.config.initial_temperature;
        self.late_scores.fill(score.clone());
        self.late_cursor = 0;
    }

    pub(super) fn metrics(mut self) -> AlnsSearchMetrics {
        react(&mut self.destroy, self.config.reaction_milli, self.adaptation);
        react(&mut self.repair, self.config.reaction_milli, self.adaptation);
        self.metrics.destroy =
            DestroyOperator::ALL.iter().enumerate().map(|(idx, operator)| operator_metrics(operator.name(), self.destroy[idx])).collect();
        self.metrics.repair =
            RepairOperator::ALL.iter().enumerate().map(|(idx, operator)| operator_metrics(operator.name(), self.repair[idx])).collect();
        self.metrics
    }
}

fn roulette<const N: usize>(operators: &[OperatorState; N], limit: usize, seed: u64, relative_floor: Option<u64>) -> usize {
    let limit = limit.min(N).max(1);
    let floor = relative_floor
        .map(|divisor| operators[..limit].iter().map(|operator| operator.weight).max().unwrap_or(1).div_ceil(divisor.max(1)).max(1))
        .unwrap_or(0);
    let total: u64 = operators[..limit].iter().map(|operator| operator.weight.max(floor)).sum();
    let mut pick = mix64(seed) % total.max(1);
    for (idx, operator) in operators[..limit].iter().enumerate() {
        let weight = operator.weight.max(floor);
        if pick < weight {
            return idx;
        }
        pick -= weight;
    }
    limit - 1
}

#[allow(clippy::too_many_arguments)]
fn update_operator(
    operator: &mut OperatorState,
    reward: u64,
    accepted: bool,
    improved: bool,
    global_best: bool,
    generated: u64,
    evaluated: u64,
    work: u64,
    deterministic_cost: u64,
    cpu_nanos: u64,
) {
    operator.uses = operator.uses.saturating_add(1);
    operator.pending_uses = operator.pending_uses.saturating_add(1);
    operator.pending_cost = operator.pending_cost.saturating_add(deterministic_cost);
    operator.pending_reward = operator.pending_reward.saturating_add(reward);
    operator.accepted = operator.accepted.saturating_add(u64::from(accepted));
    operator.improvements = operator.improvements.saturating_add(u64::from(improved));
    operator.global_bests = operator.global_bests.saturating_add(u64::from(global_best));
    operator.generated = operator.generated.saturating_add(generated);
    operator.evaluated = operator.evaluated.saturating_add(evaluated);
    operator.work_units = operator.work_units.saturating_add(work);
    operator.cpu_nanos = operator.cpu_nanos.saturating_add(cpu_nanos);
    operator.positive_rewards = operator.positive_rewards.saturating_add(u64::from(reward > 0));
}

fn record_skipped_operator(operator: &mut OperatorState) {
    // `uses` is deliberately a selection counter. All other fields describe
    // executed work and therefore stay unchanged for a skipped stage.
    operator.uses = operator.uses.saturating_add(1);
}

fn apply_unproductive_cost_weight(operator: &mut OperatorState, deterministic_cost: u64) {
    // Do not erase evidence from an operator that has already produced an
    // improvement. Before its first positive reward, however, one observation
    // is enough to price its exploration cost and prevent a slow operator from
    // retaining the same selection weight as a cheap one until segment end.
    if operator.positive_rewards != 0 {
        return;
    }
    operator.weight = scaled_cost_rate(COST_EXPLORATION_REWARD, deterministic_cost).clamp(COST_MIN_WEIGHT, MAX_OBSERVED_REWARD);
}

fn scaled_cost_rate(reward: u64, deterministic_cost: u64) -> u64 {
    let scaled = u128::from(reward).saturating_mul(u128::from(COST_RATE_SCALE)) / u128::from(deterministic_cost.max(1));
    u64::try_from(scaled.min(u128::from(MAX_OBSERVED_REWARD))).unwrap_or(MAX_OBSERVED_REWARD)
}

fn react<const N: usize>(operators: &mut [OperatorState; N], reaction_milli: u64, adaptation: AdaptationMode) {
    for operator in operators {
        if operator.pending_uses == 0 {
            continue;
        }
        let observed = match adaptation {
            AdaptationMode::PerUse => operator.pending_reward / operator.pending_uses,
            AdaptationMode::DeterministicCostRate => {
                // A small exploration credit keeps zero-reward observations
                // cost-sensitive. Without it every unproductive operator
                // converges to the same floor, so a very slow operator can
                // consume most of the projected cost despite providing no
                // reward.
                let credited_reward = operator.pending_reward.saturating_add(operator.pending_uses.saturating_mul(COST_EXPLORATION_REWARD));
                scaled_cost_rate(credited_reward, operator.pending_cost)
            }
        };
        operator.weight =
            ((WEIGHT_SCALE - reaction_milli).saturating_mul(operator.weight).saturating_add(reaction_milli.saturating_mul(observed)))
                / WEIGHT_SCALE;
        let floor = match adaptation {
            AdaptationMode::PerUse => MIN_WEIGHT,
            AdaptationMode::DeterministicCostRate => COST_MIN_WEIGHT,
        };
        operator.weight = operator.weight.max(floor);
        operator.pending_reward = 0;
        operator.pending_uses = 0;
        operator.pending_cost = 0;
    }
}

fn operator_metrics(name: &str, state: OperatorState) -> AdaptiveOperatorMetrics {
    AdaptiveOperatorMetrics {
        name: name.to_owned(),
        uses: state.uses,
        accepted: state.accepted,
        improvements: state.improvements,
        global_bests: state.global_bests,
        generated: state.generated,
        evaluated: state.evaluated,
        work_units: state.work_units,
        cpu_nanos: state.cpu_nanos,
        positive_rewards: state.positive_rewards,
        weight_milli: state.weight,
    }
}

fn normalized_worsening(current: &Score, candidate: &Score) -> f64 {
    if candidate <= current {
        return 0.0;
    }
    if candidate.violation != current.violation {
        return normalized_delta(current.violation, candidate.violation);
    }
    for (&before, &after) in current.tiers.iter().zip(&candidate.tiers) {
        if before != after {
            return normalized_delta(before, after);
        }
    }
    0.0
}

fn normalized_delta(before: i64, after: i64) -> f64 {
    let delta = i128::from(after).saturating_sub(i128::from(before)).max(0) as f64;
    let scale = before.unsigned_abs().max(after.unsigned_abs()).max(1) as f64;
    delta / scale
}

/// Deterministic work cap for one destroy/repair or macro-neighborhood slice.
/// Generation and exact evaluation are separated so filtered candidates still
/// consume budget and an operator cannot hide an unbounded generator behind a
/// small evaluation cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AlnsWorkBudget {
    pub(super) generated: u64,
    pub(super) evaluated: u64,
}

impl AlnsWorkBudget {
    pub(super) const fn new(generated: u64, evaluated: u64) -> Self {
        Self { generated, evaluated }
    }

    const fn unbounded() -> Self {
        Self { generated: u64::MAX, evaluated: u64::MAX }
    }

    const fn is_unbounded(self) -> bool {
        self.generated == u64::MAX && self.evaluated == u64::MAX
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AlnsBuildStatus {
    Built,
    Infeasible,
    BudgetExhausted,
    Interrupted,
}

pub(super) struct AlnsBuildRun {
    pub(super) status: AlnsBuildStatus,
    pub(super) candidate: Option<State>,
    pub(super) generated: u64,
    pub(super) evaluated: u64,
    pub(super) work_units: u64,
    /// Generation units spent on bounded structural work such as copying a
    /// route state, building or updating the repair index, and enumerating
    /// placement candidates. This is a subset of `generated`.
    pub(super) structural_work: u64,
    /// Structural units attributable specifically to the mutable repair index.
    pub(super) repair_index_work: u64,
    /// Full canonical `State::from_lists` rebuilds performed by this slice.
    /// Speculative move scoring never increments this counter.
    pub(super) canonical_rebuilds: u64,
    pub(super) cpu_nanos: u64,
    pub(super) destroy_generated: u64,
    pub(super) destroy_evaluated: u64,
    pub(super) destroy_cpu_nanos: u64,
    pub(super) repair_generated: u64,
    pub(super) repair_evaluated: u64,
    pub(super) repair_cpu_nanos: u64,
    pub(super) destroy_executed: bool,
    pub(super) repair_executed: bool,
    pub(super) removed: usize,
    pub(super) eliminated_route: bool,
}

#[derive(Default)]
struct BuildBreakdown {
    destroy_generated: u64,
    destroy_evaluated: u64,
    destroy_cpu_nanos: u64,
    repair_generated: u64,
    repair_evaluated: u64,
    repair_cpu_nanos: u64,
    destroy_executed: bool,
    repair_executed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Abort {
    Infeasible,
    BudgetExhausted,
    Interrupted,
}

impl Abort {
    const fn status(self) -> AlnsBuildStatus {
        match self {
            Self::Infeasible => AlnsBuildStatus::Infeasible,
            Self::BudgetExhausted => AlnsBuildStatus::BudgetExhausted,
            Self::Interrupted => AlnsBuildStatus::Interrupted,
        }
    }
}

struct WorkCounter {
    budget: AlnsWorkBudget,
    generated: u64,
    evaluated: u64,
    structural: u64,
    repair_index: u64,
    canonical_rebuilds: u64,
}

impl WorkCounter {
    const fn new(budget: AlnsWorkBudget) -> Self {
        Self { budget, generated: 0, evaluated: 0, structural: 0, repair_index: 0, canonical_rebuilds: 0 }
    }

    fn generated(&mut self, stop: &AtomicBool) -> Result<(), Abort> {
        if stop.load(Ordering::Relaxed) {
            return Err(Abort::Interrupted);
        }
        if self.generated >= self.budget.generated {
            return Err(Abort::BudgetExhausted);
        }
        self.generated += 1;
        Ok(())
    }

    fn evaluated(&mut self, stop: &AtomicBool) -> Result<(), Abort> {
        if stop.load(Ordering::Relaxed) {
            return Err(Abort::Interrupted);
        }
        if self.evaluated >= self.budget.evaluated {
            return Err(Abort::BudgetExhausted);
        }
        self.evaluated += 1;
        Ok(())
    }

    fn candidate(&mut self, stop: &AtomicBool) -> Result<(), Abort> {
        self.generated(stop)?;
        self.evaluated(stop)
    }

    /// Debit one unit of real structural work against the generation budget.
    /// The generic unbounded path keeps its historical generated/evaluated
    /// counters and only performs the interruption check.
    fn structural(&mut self, stop: &AtomicBool) -> Result<(), Abort> {
        self.structural_units(1, stop)
    }

    /// Reserve structural work in O(1) before performing it. A failed
    /// reservation leaves the counters unchanged, so callers never begin an
    /// allocation, clone, or cache pass that cannot fit in the slice.
    fn structural_many(&mut self, units: usize, stop: &AtomicBool) -> Result<(), Abort> {
        self.structural_units(u64::try_from(units).unwrap_or(u64::MAX), stop)
    }

    fn structural_units(&mut self, units: u64, stop: &AtomicBool) -> Result<(), Abort> {
        if stop.load(Ordering::Relaxed) {
            return Err(Abort::Interrupted);
        }
        if !self.is_bounded() {
            return Ok(());
        }
        if units > self.remaining_generated() {
            return Err(Abort::BudgetExhausted);
        }
        self.generated = self.generated.saturating_add(units);
        self.structural = self.structural.saturating_add(units);
        Ok(())
    }

    fn repair_index(&mut self, stop: &AtomicBool) -> Result<(), Abort> {
        self.structural(stop)?;
        if self.is_bounded() {
            self.repair_index = self.repair_index.saturating_add(1);
        }
        Ok(())
    }

    const fn is_bounded(&self) -> bool {
        !self.budget.is_unbounded()
    }

    fn remaining_generated(&self) -> u64 {
        self.budget.generated.saturating_sub(self.generated)
    }

    fn remaining_evaluated(&self) -> u64 {
        self.budget.evaluated.saturating_sub(self.evaluated)
    }
}

pub(super) fn build_candidate(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    choice: AlnsChoice,
    seed: u64,
    stop: &AtomicBool,
) -> Option<State> {
    let run = build_candidate_bounded(model, per, current, choice, seed, AlnsWorkBudget::unbounded(), stop);
    let mut candidate = run.candidate?;
    let polish_factor = choice.repair.regret().map_or(1, |(regret, _)| regret);
    let polish_steps = run.removed.saturating_mul(polish_factor);
    polish(per, &mut candidate, stop, polish_steps, seed ^ 0xD1B5_4A32_D192_ED03).then_some(candidate)
}

pub(super) fn build_candidate_bounded(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    choice: AlnsChoice,
    seed: u64,
    budget: AlnsWorkBudget,
    stop: &AtomicBool,
) -> AlnsBuildRun {
    let started = CpuTimer::start();
    let mut work = WorkCounter::new(budget);
    let mut breakdown = BuildBreakdown::default();
    let result = build_candidate_inner(model, per, current, choice, seed, stop, &mut work, &mut breakdown);
    let (status, candidate, removed, eliminated_route) = match result {
        Ok((candidate, removed, eliminated_route)) => (AlnsBuildStatus::Built, Some(candidate), removed, eliminated_route),
        Err(abort) => (abort.status(), None, 0, false),
    };
    AlnsBuildRun {
        status,
        candidate,
        generated: work.generated,
        evaluated: work.evaluated,
        work_units: work.generated.saturating_add(work.evaluated),
        structural_work: work.structural,
        repair_index_work: work.repair_index,
        canonical_rebuilds: work.canonical_rebuilds,
        cpu_nanos: started.elapsed_nanos(),
        destroy_generated: breakdown.destroy_generated,
        destroy_evaluated: breakdown.destroy_evaluated,
        destroy_cpu_nanos: breakdown.destroy_cpu_nanos,
        repair_generated: breakdown.repair_generated,
        repair_evaluated: breakdown.repair_evaluated,
        repair_cpu_nanos: breakdown.repair_cpu_nanos,
        destroy_executed: breakdown.destroy_executed,
        repair_executed: breakdown.repair_executed,
        removed,
        eliminated_route,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_candidate_inner(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    choice: AlnsChoice,
    seed: u64,
    stop: &AtomicBool,
    work: &mut WorkCounter,
    breakdown: &mut BuildBreakdown,
) -> Result<(State, usize, bool), Abort> {
    if stop.load(Ordering::Relaxed) {
        return Err(Abort::Interrupted);
    }
    if choice.target == 0 {
        return Err(Abort::Infeasible);
    }
    if work.budget.generated == 0 || work.budget.evaluated == 0 {
        return Err(Abort::BudgetExhausted);
    }

    let destroy_generated = work.generated;
    let destroy_evaluated = work.evaluated;
    let destroy_timer = CpuTimer::start();
    breakdown.destroy_executed = true;
    let destroyed = (|| {
        let mut lists = clone_state_lists_interruptible(model, current, work, stop)?;
        let routes_before = nonempty_routes(&lists);
        let target = if work.is_bounded() { bounded_destroy_target(model.items.len(), choice, work.budget) } else { choice.target };
        let (removed, allow_empty_route) = match choice.destroy {
            DestroyOperator::Shaw => (destroy_shaw(&mut lists, per, target, seed, work, stop)?, true),
            DestroyOperator::Segments => (destroy_segments(&mut lists, target, seed, work, stop)?, true),
            DestroyOperator::Worst => (destroy_worst(per, current, &mut lists, target, seed, work, stop)?, true),
            DestroyOperator::Route => (destroy_route(&mut lists, target, seed, work, stop)?, false),
        };
        if removed.is_empty() {
            return Err(Abort::Infeasible);
        }
        let state = canonical_state_from_lists(model, per, lists, work, stop)?;
        Ok((state, removed, allow_empty_route, routes_before))
    })();
    breakdown.destroy_generated = work.generated.saturating_sub(destroy_generated);
    breakdown.destroy_evaluated = work.evaluated.saturating_sub(destroy_evaluated);
    breakdown.destroy_cpu_nanos = destroy_timer.elapsed_nanos();
    let (mut state, removed, allow_empty_route, routes_before) = destroyed?;

    let repair_generated = work.generated;
    let repair_evaluated = work.evaluated;
    let repair_timer = CpuTimer::start();
    breakdown.repair_executed = true;
    let repaired = (|| {
        match choice.repair.regret() {
            Some((k, blink)) => repair_regret(
                per,
                &mut state,
                &removed,
                RegretRepair { k, blink, allow_empty: allow_empty_route, seed: seed ^ 0xA076_1D64_78BD_642F },
                work,
                stop,
            ),
            None => repair_greedy(per, &mut state, &removed, allow_empty_route, seed ^ 0xA076_1D64_78BD_642F, work, stop),
        }?;
        state = canonical_state_from_lists(model, per, state.lists, work, stop)?;
        let eliminated_route = matches!(choice.destroy, DestroyOperator::Route);
        if eliminated_route && nonempty_routes(&state.lists) >= routes_before {
            return Err(Abort::Infeasible);
        }
        Ok((state, removed.len(), eliminated_route))
    })();
    breakdown.repair_generated = work.generated.saturating_sub(repair_generated);
    breakdown.repair_evaluated = work.evaluated.saturating_sub(repair_evaluated);
    breakdown.repair_cpu_nanos = repair_timer.elapsed_nanos();
    repaired
}

fn bounded_destroy_target(items: usize, choice: AlnsChoice, budget: AlnsWorkBudget) -> usize {
    if matches!(choice.destroy, DestroyOperator::Route) {
        return choice.target;
    }
    let destroy_generated = if matches!(choice.destroy, DestroyOperator::Worst) { items } else { choice.target };
    let destroy_evaluated = if matches!(choice.destroy, DestroyOperator::Worst) { items } else { 0 };
    let repair_generated = budget.generated.saturating_sub(destroy_generated as u64);
    let repair_evaluated = budget.evaluated.saturating_sub(destroy_evaluated as u64);
    let repairable = usize::try_from(repair_generated.min(repair_evaluated)).unwrap_or(usize::MAX);
    choice.target.min(repairable.max(1))
}

fn state_from_lists(model: &CollectionModel, per: &PerList, lists: Vec<Vec<i32>>, stop: &AtomicBool) -> Result<State, Abort> {
    State::from_lists_interruptible(model, per, lists, stop).ok_or_else(|| {
        if stop.load(Ordering::Relaxed) {
            Abort::Interrupted
        } else {
            Abort::Infeasible
        }
    })
}

fn clone_state_lists_interruptible(
    model: &CollectionModel,
    state: &State,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Vec<Vec<i32>>, Abort> {
    work.structural_units(state_clone_work(model, state), stop)?;
    clone_lists_reserved(&state.lists, stop)
}

fn clone_route_reserved(route: &[i32], stop: &AtomicBool) -> Result<Vec<i32>, Abort> {
    let mut copied = Vec::with_capacity(route.len());
    for (index, &item) in route.iter().enumerate() {
        if index.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
            return Err(Abort::Interrupted);
        }
        copied.push(item);
    }
    Ok(copied)
}

fn clone_lists_reserved(lists: &[Vec<i32>], stop: &AtomicBool) -> Result<Vec<Vec<i32>>, Abort> {
    let mut copied = Vec::with_capacity(lists.len());
    for route in lists {
        if stop.load(Ordering::Relaxed) {
            return Err(Abort::Interrupted);
        }
        copied.push(clone_route_reserved(route, stop)?);
    }
    Ok(copied)
}

fn ceil_log2(value: u64) -> u64 {
    if value <= 1 {
        1
    } else {
        u64::from(u64::BITS - (value - 1).leading_zeros())
    }
}

/// Conservative deterministic work for building one complete reduction cache.
/// Pair reductions are quadratic, windows include every inner evaluation, and
/// set/order-statistic sorts include a comparison-sort upper envelope.
fn reduction_cache_work(reduction: &Reduction, len: usize) -> u64 {
    let len = u64::try_from(len).unwrap_or(u64::MAX);
    let (outputs, emit_work) = match &reduction.iterable {
        Iterable::Items(_) => (len, len),
        Iterable::SetItems(_) => (len, len.saturating_add(len.saturating_mul(ceil_log2(len)))),
        Iterable::Edges { .. } => {
            let outputs = len.saturating_add(1);
            (outputs, outputs)
        }
        Iterable::Pairs(_) => {
            let outputs = len.saturating_mul(len);
            (outputs, outputs)
        }
        Iterable::Scan { end, .. } => {
            let outputs = len.saturating_add(u64::from(end.is_some()));
            (outputs, outputs.saturating_mul(2))
        }
        Iterable::Windows { size, .. } => {
            let size = u64::try_from(*size).unwrap_or(u64::MAX);
            let outputs = if size == 0 || len < size { 0 } else { len.saturating_sub(size).saturating_add(1) };
            (outputs, outputs.saturating_mul(size.saturating_add(1)))
        }
    };
    let sorted_work = if matches!(reduction.op, ReduceOp::Min | ReduceOp::Max | ReduceOp::SelectKth(_)) {
        outputs.saturating_mul(ceil_log2(outputs))
    } else {
        0
    };
    let pair_ranges = if matches!(reduction.op, ReduceOp::Sum) && matches!(reduction.iterable, Iterable::Pairs(_)) { outputs } else { 0 };
    // aggregate, sum prefix, sum suffix, and score extraction each traverse at
    // most the emitted sequence once.
    1u64.saturating_add(emit_work).saturating_add(outputs.saturating_mul(4)).saturating_add(sorted_work).saturating_add(pair_ranges)
}

const COMPOUND_TOUCHED_ROUTES: usize = 4;
const COMPOUND_ROUTE_GROWTH: usize = 3;

pub(super) fn state_clone_work(model: &CollectionModel, state: &State) -> u64 {
    u64::try_from(state.lists.len()).unwrap_or(u64::MAX).saturating_add(u64::try_from(model.items.len()).unwrap_or(u64::MAX))
}

const FLOOR_STOP_POLL_INTERVAL: usize = 256;

struct FloorStopPoll<'a> {
    stop: &'a AtomicBool,
    until_check: usize,
}

impl<'a> FloorStopPoll<'a> {
    fn new(stop: &'a AtomicBool) -> Option<Self> {
        (!stop.load(Ordering::Relaxed)).then_some(Self { stop, until_check: FLOOR_STOP_POLL_INTERVAL })
    }

    fn tick(&mut self) -> Option<()> {
        self.until_check = self.until_check.saturating_sub(1);
        if self.until_check == 0 {
            if self.stop.load(Ordering::Relaxed) {
                return None;
            }
            self.until_check = FLOOR_STOP_POLL_INTERVAL;
        }
        Some(())
    }

    fn finish(&self) -> Option<()> {
        (!self.stop.load(Ordering::Relaxed)).then_some(())
    }
}

fn list_reduction_work_polled(per: &PerList, list: usize, len: usize, poll: &mut FloorStopPoll<'_>) -> Option<u64> {
    let mut total = 0u64;
    for tier in &per.objective[list] {
        poll.tick()?;
        for reduction in tier {
            poll.tick()?;
            total = total.saturating_add(reduction_cache_work(reduction, len));
        }
    }
    for constraint in &per.constraints[list] {
        poll.tick()?;
        total = total.saturating_add(reduction_cache_work(&constraint.reduction, len));
    }
    for tier in &per.max_objective {
        poll.tick()?;
        for term in tier {
            poll.tick()?;
            for group in &term.groups {
                poll.tick()?;
                for reduction in group {
                    poll.tick()?;
                    if reduction.iterable.list() == list {
                        total = total.saturating_add(reduction_cache_work(reduction, len));
                    }
                }
            }
        }
    }
    Some(total.saturating_add(1))
}

fn global_memberships_polled(model: &CollectionModel, poll: &mut FloorStopPoll<'_>) -> Option<u64> {
    let mut total = 0u64;
    for constraint in &model.globals {
        poll.tick()?;
        let members = match constraint {
            GlobalConstraint::ListLe { .. }
            | GlobalConstraint::SameList { .. }
            | GlobalConstraint::DifferentList { .. }
            | GlobalConstraint::ListDistance { .. } => 2usize,
            GlobalConstraint::AllSameList { items } | GlobalConstraint::AllDifferentLists { items } => items.len(),
        };
        total = total.saturating_add(u64::try_from(members).unwrap_or(u64::MAX));
    }
    Some(total)
}

fn state_rebuild_work_from_lengths_polled(
    model: &CollectionModel,
    per: &PerList,
    lengths: impl IntoIterator<Item = usize>,
    poll: &mut FloorStopPoll<'_>,
) -> Option<u64> {
    let mut total = u64::try_from(model.lists)
        .unwrap_or(u64::MAX)
        .saturating_add(global_memberships_polled(model, poll)?)
        .saturating_add(u64::try_from(model.globals.len()).unwrap_or(u64::MAX));
    for (list, len) in lengths.into_iter().enumerate() {
        poll.tick()?;
        total =
            total.saturating_add(u64::try_from(len).unwrap_or(u64::MAX)).saturating_add(list_reduction_work_polled(per, list, len, poll)?);
    }
    Some(total)
}

fn state_rebuild_work_interruptible(model: &CollectionModel, per: &PerList, lists: &[Vec<i32>], stop: &AtomicBool) -> Option<u64> {
    let mut poll = FloorStopPoll::new(stop)?;
    let work = state_rebuild_work_from_lengths_polled(model, per, lists.iter().map(Vec::len), &mut poll)?;
    poll.finish()?;
    Some(work)
}

fn score_fold_work_polled(per: &PerList, state: &State, poll: &mut FloorStopPoll<'_>) -> Option<u64> {
    let state_fold = u64::try_from(state.scores.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(per.tiers.saturating_add(1)).unwrap_or(u64::MAX));
    let mut max_fold = 0u64;
    for tier in &per.max_objective {
        poll.tick()?;
        for term in tier {
            poll.tick()?;
            max_fold = max_fold.saturating_add(1);
            for group in &term.groups {
                poll.tick()?;
                max_fold = max_fold.saturating_add(u64::try_from(group.len()).unwrap_or(u64::MAX));
            }
        }
    }
    Some(state_fold.saturating_add(max_fold))
}

fn retain_largest_four(largest: &mut [u64; COMPOUND_TOUCHED_ROUTES], value: u64) {
    if value > largest[0] {
        largest[0] = value;
        largest.sort_unstable();
    }
}

fn guided_exchange_discovery_floor_with_items(per: &PerList, state: &State, items: u64, poll: &mut FloorStopPoll<'_>) -> Option<u64> {
    let lists = u64::try_from(state.lists.len()).unwrap_or(u64::MAX);
    let Some(candidates) = &per.candidates else {
        // One pass collects non-empty routes, then the first segment pair is a
        // candidate because no geometric filter is active.
        return Some(lists.saturating_add(2));
    };
    let mut directed_edges = 0u64;
    for route in &state.lists {
        poll.tick()?;
        for &item in route {
            poll.tick()?;
            directed_edges = directed_edges.saturating_add(u64::try_from(candidates.neighbors(item).len()).unwrap_or(u64::MAX));
        }
    }
    // Build item locations and visit every directed candidate edge. Every edge
    // between distinct routes, unless both are singletons, immediately induces
    // at least one valid segment pair. The final two units derive and admit it.
    Some(lists.saturating_mul(2).saturating_add(items.saturating_mul(2)).saturating_add(directed_edges).saturating_add(2))
}

fn guided_exchange_discovery_floor(model: &CollectionModel, per: &PerList, state: &State, stop: &AtomicBool) -> Option<u64> {
    let mut poll = FloorStopPoll::new(stop)?;
    let items = u64::try_from(model.items.len()).unwrap_or(u64::MAX);
    let floor = guided_exchange_discovery_floor_with_items(per, state, items, &mut poll)?;
    poll.finish()?;
    Some(floor)
}

fn trial_score_work(
    model: &CollectionModel,
    per: &PerList,
    state: &State,
    touched: &[(usize, usize)],
    override_items: usize,
    stop: &AtomicBool,
) -> Option<u64> {
    let mut poll = FloorStopPoll::new(stop)?;
    let mut reduction_units = 0u64;
    for &(list, len) in touched {
        poll.tick()?;
        reduction_units = reduction_units.saturating_add(list_reduction_work_polled(per, list, len, &mut poll)?);
    }
    let memberships = global_memberships_polled(model, &mut poll)?;
    let overrides = u64::try_from(override_items).unwrap_or(u64::MAX);
    let globals = u64::try_from(model.globals.len()).unwrap_or(u64::MAX);
    let global_delta =
        memberships.saturating_mul(overrides.saturating_add(2)).saturating_add(overrides.saturating_mul(globals)).saturating_add(globals);
    let units = reduction_units.saturating_add(score_fold_work_polled(per, state, &mut poll)?).saturating_add(global_delta);
    poll.finish()?;
    Some(units)
}

/// Conservative generation budget that lets a compound routing neighborhood
/// score its first complete candidate and retain enough headroom to rebuild a
/// winner. The bound follows the actual cache shapes, so a fixed exploration
/// cap cannot silently disable the neighborhoods on models with many lists or
/// expensive reductions.
pub(super) fn routing_compound_structural_floor(model: &CollectionModel, per: &PerList, state: &State, stop: &AtomicBool) -> Option<u64> {
    let mut poll = FloorStopPoll::new(stop)?;
    let model_lists = u64::try_from(model.lists).unwrap_or(u64::MAX);
    let globals = u64::try_from(model.globals.len()).unwrap_or(u64::MAX);
    let memberships = global_memberships_polled(model, &mut poll)?;
    let total_items = model.items.len();
    let mut max_route = 0usize;
    let clone_work = state_clone_work(model, state);
    let mut rebuild = model_lists.saturating_add(memberships).saturating_add(globals);
    let mut rebuild_growths = [0u64; COMPOUND_TOUCHED_ROUTES];
    let mut touched_reductions = [0u64; COMPOUND_TOUCHED_ROUTES];

    // One interruptible route pass replaces the previous clone, rebuild,
    // maximum-length, and two reduction-envelope rescans.
    for (list, route) in state.lists.iter().enumerate() {
        poll.tick()?;
        let len = route.len();
        let len_work = u64::try_from(len).unwrap_or(u64::MAX);
        max_route = max_route.max(len);
        let before_reductions = list_reduction_work_polled(per, list, len, &mut poll)?;
        rebuild = rebuild.saturating_add(len_work).saturating_add(before_reductions);

        let grown_len = len.saturating_add(COMPOUND_ROUTE_GROWTH);
        let grown_len_work = u64::try_from(grown_len).unwrap_or(u64::MAX);
        let grown_reductions = list_reduction_work_polled(per, list, grown_len, &mut poll)?;
        let growth = grown_len_work.saturating_add(grown_reductions).saturating_sub(len_work.saturating_add(before_reductions));
        retain_largest_four(&mut rebuild_growths, growth);
        retain_largest_four(&mut touched_reductions, grown_reductions);
    }
    rebuild = rebuild.saturating_add(rebuild_growths.into_iter().fold(0u64, u64::saturating_add));

    // Or-opt has the largest materialization edit: one source suffix and up to
    // three destination suffix shifts. Cross-exchange is bounded by this too.
    let edit = u64::try_from(max_route).unwrap_or(u64::MAX).saturating_mul(4).saturating_add(6);
    let materialization = clone_work.saturating_add(edit).saturating_add(rebuild);

    let total_items_work = u64::try_from(total_items).unwrap_or(u64::MAX);
    let global_delta = memberships
        .saturating_mul(total_items_work.saturating_add(2))
        .saturating_add(total_items_work.saturating_mul(globals))
        .saturating_add(globals);
    let speculative_score = touched_reductions
        .into_iter()
        .fold(0u64, u64::saturating_add)
        .saturating_add(score_fold_work_polled(per, state, &mut poll)?)
        .saturating_add(global_delta);

    // Before the first ejection candidate is complete, the implementation can
    // remove one source suffix, clone that route twice across the overlay fork,
    // enumerate one destination, shift its insertion suffix, and clone it.
    let ejection_setup = u64::try_from(max_route).unwrap_or(u64::MAX).saturating_mul(6).saturating_add(8);
    let guided_setup = guided_exchange_discovery_floor_with_items(per, state, total_items_work, &mut poll)?;
    let floor = ejection_setup.max(guided_setup).saturating_add(speculative_score).saturating_add(materialization);
    poll.finish()?;
    Some(floor)
}

/// Conservative budget for the bounded route-elimination pipeline. It covers
/// the destroy scan, the destroyed-state rebuild, a one-placement regret-2
/// repair for every sampled item, repair-index maintenance, and the final
/// canonical rebuild. Placement work is bounded by the hard repair cap, so the
/// final rebuild remains affordable even when all 96 placements are scored.
pub(super) fn routing_route_elimination_floor(
    model: &CollectionModel,
    per: &PerList,
    state: &State,
    stop: &AtomicBool,
) -> Option<AlnsWorkBudget> {
    let mut poll = FloorStopPoll::new(stop)?;
    let lists = u64::try_from(state.lists.len()).unwrap_or(u64::MAX);
    let model_lists = u64::try_from(model.lists).unwrap_or(u64::MAX);
    let globals = u64::try_from(model.globals.len()).unwrap_or(u64::MAX);
    let memberships = global_memberships_polled(model, &mut poll)?;
    let items = model.items.len();
    let mut removed = None;
    let mut max_route = 0usize;
    let clone_work = state_clone_work(model, state);
    let mut rebuild = model_lists.saturating_add(memberships).saturating_add(globals);
    for (list, route) in state.lists.iter().enumerate() {
        poll.tick()?;
        let len = route.len();
        let len_work = u64::try_from(len).unwrap_or(u64::MAX);
        max_route = max_route.max(len);
        if len != 0 {
            removed = Some(removed.map_or(len, |smallest: usize| smallest.min(len)));
        }
        rebuild = rebuild.saturating_add(len_work).saturating_add(list_reduction_work_polled(per, list, len, &mut poll)?);
    }
    let items_work = u64::try_from(items).unwrap_or(u64::MAX);
    let Some(removed) = removed else {
        poll.finish()?;
        return Some(AlnsWorkBudget::new(0, 0));
    };
    let removed_work = u64::try_from(removed).unwrap_or(u64::MAX);
    let repair_calls = regret_repair_calls(removed);
    let mut max_neighbors = 0usize;
    if let Some(candidates) = &per.candidates {
        for &item in &model.items {
            poll.tick()?;
            max_neighbors = max_neighbors.max(candidates.neighbors(item).len().saturating_add(candidates.semantic_neighbors(item).len()));
        }
    }

    let growth_count = removed.min(state.lists.len());
    let mut largest_growths = BinaryHeap::<Reverse<u64>>::new();
    for (list, route) in state.lists.iter().enumerate() {
        poll.tick()?;
        let len = route.len();
        let before_reductions = list_reduction_work_polled(per, list, len, &mut poll)?;
        let grown_len = len.saturating_add(removed);
        let grown_reductions = list_reduction_work_polled(per, list, grown_len, &mut poll)?;
        let growth = u64::try_from(grown_len)
            .unwrap_or(u64::MAX)
            .saturating_add(grown_reductions)
            .saturating_sub(u64::try_from(len).unwrap_or(u64::MAX).saturating_add(before_reductions));
        if largest_growths.len() < growth_count {
            largest_growths.push(Reverse(growth));
        } else if largest_growths.peek().is_some_and(|&Reverse(smallest)| growth > smallest) {
            largest_growths.pop();
            largest_growths.push(Reverse(growth));
        }
    }
    let final_growth = largest_growths.into_iter().fold(0u64, |total, Reverse(value)| total.saturating_add(value));
    let final_rebuild = rebuild.saturating_add(final_growth);

    // Prefix through RepairWorkspace::build. The two removed-item terms cover
    // destroy generation/shuffle plus the pending repair copy.
    let prefix = clone_work
        .saturating_add(lists.saturating_mul(5))
        .saturating_add(items_work)
        .saturating_add(removed_work.saturating_mul(2))
        .saturating_add(rebuild);

    let placement_cap = u64::try_from(MAX_REPAIR_PLACEMENTS).unwrap_or(u64::MAX);
    // Candidate-graph repair performs one lookup and at most two pushes per
    // neighbor, then at most two endpoint pushes per route. Generic repair
    // scans every eligible route and pushes at most `placement_cap` gaps.
    let neighbor_work = u64::try_from(max_neighbors).unwrap_or(u64::MAX).saturating_mul(3).saturating_add(lists.saturating_mul(2));
    let generic_work = lists.saturating_add(placement_cap);
    let placement_call = neighbor_work.max(generic_work);
    let placement_work = repair_calls.saturating_mul(placement_call);
    let insertion_route = u64::try_from(max_route.saturating_add(removed)).unwrap_or(u64::MAX);
    let index_updates = removed_work.saturating_mul(insertion_route).saturating_mul(2);
    let suffix = placement_work.saturating_add(index_updates).saturating_add(final_rebuild);

    let budget = AlnsWorkBudget::new(prefix.saturating_add(suffix), repair_calls.saturating_mul(placement_cap));
    poll.finish()?;
    Some(budget)
}

/// Structural envelope for one relinking step: clone the mutable partition,
/// move one item, clone it for the candidate, and rebuild all canonical caches.
pub(super) fn routing_relink_structural_floor(model: &CollectionModel, per: &PerList, state: &State, stop: &AtomicBool) -> Option<u64> {
    let mut poll = FloorStopPoll::new(stop)?;
    let mut max_route = 0u64;
    let mut rebuild_growth = 0u64;
    let clone_work = state_clone_work(model, state);
    let mut rebuild = u64::try_from(model.lists)
        .unwrap_or(u64::MAX)
        .saturating_add(global_memberships_polled(model, &mut poll)?)
        .saturating_add(u64::try_from(model.globals.len()).unwrap_or(u64::MAX));
    for (list, route) in state.lists.iter().enumerate() {
        poll.tick()?;
        let len = route.len();
        let len_work = u64::try_from(len).unwrap_or(u64::MAX);
        max_route = max_route.max(len_work);
        let before_reductions = list_reduction_work_polled(per, list, len, &mut poll)?;
        rebuild = rebuild.saturating_add(len_work).saturating_add(before_reductions);
        let grown_len = len.saturating_add(1);
        let growth = u64::try_from(grown_len)
            .unwrap_or(u64::MAX)
            .saturating_add(list_reduction_work_polled(per, list, grown_len, &mut poll)?)
            .saturating_sub(len_work.saturating_add(before_reductions));
        rebuild_growth = rebuild_growth.max(growth);
    }
    let floor = clone_work
        .saturating_mul(5)
        .saturating_add(max_route.saturating_mul(2).saturating_add(2))
        .saturating_add(rebuild)
        .saturating_add(rebuild_growth)
        .saturating_add(1);
    poll.finish()?;
    Some(floor)
}

/// Exact charged work after a relinking move has been selected: both vector
/// suffix shifts, cloning the edited partition for canonical ownership, and
/// rebuilding caches for the resulting route lengths.
#[allow(clippy::too_many_arguments)]
pub(super) fn routing_relink_candidate_work(
    model: &CollectionModel,
    per: &PerList,
    lists: &[Vec<i32>],
    source: usize,
    source_position: usize,
    destination: usize,
    destination_position: usize,
    stop: &AtomicBool,
) -> Option<u64> {
    let mut poll = FloorStopPoll::new(stop)?;
    let source_len = lists[source].len();
    let source_shift = source_len.saturating_sub(source_position.saturating_add(1));
    let destination_len_after_removal = lists[destination].len().saturating_sub(usize::from(source == destination));
    let insertion = destination_position.min(destination_len_after_removal);
    let destination_shift = destination_len_after_removal.saturating_sub(insertion);
    let clone_work = u64::try_from(lists.len()).unwrap_or(u64::MAX).saturating_add(u64::try_from(model.items.len()).unwrap_or(u64::MAX));
    let mut rebuild = u64::try_from(model.lists)
        .unwrap_or(u64::MAX)
        .saturating_add(global_memberships_polled(model, &mut poll)?)
        .saturating_add(u64::try_from(model.globals.len()).unwrap_or(u64::MAX));
    for (list, route) in lists.iter().enumerate() {
        poll.tick()?;
        let rebuilt_len = if source == destination {
            route.len()
        } else if list == source {
            route.len().saturating_sub(1)
        } else if list == destination {
            route.len().saturating_add(1)
        } else {
            route.len()
        };
        rebuild = rebuild.saturating_add(u64::try_from(rebuilt_len).unwrap_or(u64::MAX)).saturating_add(list_reduction_work_polled(
            per,
            list,
            rebuilt_len,
            &mut poll,
        )?);
    }
    let work = u64::try_from(source_shift.saturating_add(destination_shift))
        .unwrap_or(u64::MAX)
        .saturating_add(clone_work)
        .saturating_add(rebuild);
    poll.finish()?;
    Some(work)
}

struct TrialScoreWork {
    touched: SmallVec<[(usize, usize); 4]>,
    override_items: usize,
    materialization_headroom: u64,
}

fn reserve_trial_scores(
    model: &CollectionModel,
    per: &PerList,
    state: &State,
    request: TrialScoreWork,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<(), Abort> {
    let units = trial_score_work(model, per, state, &request.touched, request.override_items, stop).ok_or(Abort::Interrupted)?;
    if work.evaluated >= work.budget.evaluated {
        return Err(Abort::BudgetExhausted);
    }
    if work.is_bounded() && units.saturating_add(request.materialization_headroom) > work.remaining_generated() {
        return Err(Abort::BudgetExhausted);
    }
    work.structural_units(units, stop)?;
    work.evaluated = work.evaluated.saturating_add(1);
    Ok(())
}

fn canonical_state_from_lists(
    model: &CollectionModel,
    per: &PerList,
    lists: Vec<Vec<i32>>,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<State, Abort> {
    let rebuild = state_rebuild_work_interruptible(model, per, &lists, stop).ok_or(Abort::Interrupted)?;
    work.structural_units(rebuild, stop)?;
    let state = state_from_lists(model, per, lists, stop)?;
    work.canonical_rebuilds = work.canonical_rebuilds.saturating_add(1);
    Ok(state)
}

fn nonempty_routes(lists: &[Vec<i32>]) -> usize {
    lists.iter().filter(|route| !route.is_empty()).count()
}

fn destroy_route(lists: &mut [Vec<i32>], target: usize, seed: u64, work: &mut WorkCounter, stop: &AtomicBool) -> Result<Vec<i32>, Abort> {
    let route = if work.is_bounded() {
        let mut min_len = usize::MAX;
        for list in lists.iter() {
            work.structural(stop)?;
            if !list.is_empty() {
                min_len = min_len.min(list.len());
            }
        }
        if min_len == usize::MAX {
            return Ok(Vec::new());
        }
        let mut selected: Option<(u64, usize)> = None;
        for (idx, list) in lists.iter().enumerate() {
            work.structural(stop)?;
            if list.len() == min_len {
                let tie = mix64(seed ^ mix64(idx as u64));
                if selected.is_none_or(|(best, _)| tie < best) {
                    selected = Some((tie, idx));
                }
            }
        }
        selected.expect("a non-empty route has the minimum length").1
    } else {
        let nonempty: Vec<usize> = lists.iter().enumerate().filter_map(|(idx, list)| (!list.is_empty()).then_some(idx)).collect();
        let Some(min_len) = nonempty.iter().map(|&route| lists[route].len()).min() else { return Ok(Vec::new()) };
        let smallest: Vec<usize> = nonempty.into_iter().filter(|&route| lists[route].len() == min_len).collect();
        smallest[(mix64(seed) % smallest.len() as u64) as usize]
    };
    for _ in 0..lists[route].len() {
        work.generated(stop)?;
    }
    let mut removed = std::mem::take(&mut lists[route]);
    if !work.is_bounded() && removed.len() < target {
        removed.extend(destroy_segments(lists, target - removed.len(), seed ^ 0xE703_7ED1_A0B4_28DB, work, stop)?);
    }
    shuffle_values_bounded(&mut removed, seed ^ 0xD1B5_4A32_D192_ED03, work, stop)?;
    Ok(removed)
}

fn destroy_segments(
    lists: &mut [Vec<i32>],
    target: usize,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Vec<i32>, Abort> {
    let mut removed = Vec::with_capacity(target);
    let mut step = 0u64;
    work.structural_many(lists.len(), stop)?;
    let mut total = lists.iter().map(Vec::len).sum::<usize>();
    while removed.len() < target {
        if total == 0 {
            break;
        }
        let mut pick = (mix64(seed ^ mix64(step)) % total as u64) as usize;
        let mut route = 0usize;
        let mut pos = 0usize;
        for (idx, list) in lists.iter().enumerate() {
            work.structural(stop)?;
            if pick < list.len() {
                route = idx;
                pos = pick;
                break;
            }
            pick -= list.len();
        }
        let max_len = lists[route].len().min(target - removed.len());
        let len = 1 + (mix64(seed.wrapping_add(step).wrapping_add(0x9E37_79B9_7F4A_7C15)) % max_len as u64) as usize;
        let start = pos.min(lists[route].len() - len);
        let shifted = lists[route].len().saturating_sub(start + len);
        work.structural_many(shifted, stop)?;
        for _ in 0..len {
            work.generated(stop)?;
        }
        removed.extend(lists[route].drain(start..start + len));
        total -= len;
        step = step.wrapping_add(1);
    }
    shuffle_values_bounded(&mut removed, seed ^ 0xD1B5_4A32_D192_ED03, work, stop)?;
    Ok(removed)
}

fn destroy_shaw(
    lists: &mut [Vec<i32>],
    per: &PerList,
    target: usize,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Vec<i32>, Abort> {
    let mut present = HashSet::new();
    let mut all = Vec::new();
    let mut owner = HashMap::new();
    for (list_index, list) in lists.iter().enumerate() {
        work.structural(stop)?;
        for (position, &item) in list.iter().enumerate() {
            work.structural(stop)?;
            present.insert(item);
            all.push(item);
            owner.insert(item, (list_index, position));
        }
    }
    if present.is_empty() {
        return Ok(Vec::new());
    }
    let mut removed = HashSet::with_capacity(target);
    let mut order = Vec::with_capacity(target);
    let first = all[(mix64(seed) % all.len() as u64) as usize];
    work.generated(stop)?;
    removed.insert(first);
    order.push(first);
    let mut step = 0u64;
    while removed.len() < target && removed.len() < present.len() {
        let pivot = order[(mix64(seed ^ mix64(step)) % order.len() as u64) as usize];
        let related = shaw_related_candidate(per, lists, &owner, &all, pivot, &removed, &present, seed ^ step, work, stop)?;
        let related = match related {
            Some(item) => Some(item),
            None => random_available(&all, &removed, seed ^ step, work, stop)?,
        };
        let Some(next) = related else { break };
        work.generated(stop)?;
        removed.insert(next);
        order.push(next);
        step = step.wrapping_add(1);
    }
    let mut result = Vec::with_capacity(removed.len());
    for list in lists {
        let contents = std::mem::take(list);
        list.reserve(contents.len().saturating_sub(removed.len()));
        for item in contents {
            work.structural(stop)?;
            if removed.contains(&item) {
                result.push(item);
            } else {
                list.push(item);
            }
        }
    }
    shuffle_values_bounded(&mut result, seed ^ 0xD1B5_4A32_D192_ED03, work, stop)?;
    Ok(result)
}

fn push_shaw_candidate(
    pool: &mut Vec<i32>,
    seen: &mut HashSet<i32>,
    item: i32,
    pivot: i32,
    removed: &HashSet<i32>,
    present: &HashSet<i32>,
    limit: usize,
) {
    if pool.len() < limit && item != pivot && present.contains(&item) && !removed.contains(&item) && seen.insert(item) {
        pool.push(item);
    }
}

#[allow(clippy::too_many_arguments)]
fn shaw_related_candidate(
    per: &PerList,
    lists: &[Vec<i32>],
    owner: &HashMap<i32, (usize, usize)>,
    all: &[i32],
    pivot: i32,
    removed: &HashSet<i32>,
    present: &HashSet<i32>,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Option<i32>, Abort> {
    const RELATED_POOL: usize = 32;
    let mut pool = Vec::with_capacity(RELATED_POOL);
    let mut seen = HashSet::with_capacity(RELATED_POOL);
    if let Some(candidates) = &per.candidates {
        for &item in candidates.neighbors(pivot) {
            if pool.len() >= RELATED_POOL {
                break;
            }
            work.structural(stop)?;
            push_shaw_candidate(&mut pool, &mut seen, item, pivot, removed, present, RELATED_POOL);
        }
    }
    if let Some(&(route, position)) = owner.get(&pivot) {
        work.structural(stop)?;
        if let Some(&before) = position.checked_sub(1).and_then(|index| lists[route].get(index)) {
            push_shaw_candidate(&mut pool, &mut seen, before, pivot, removed, present, RELATED_POOL);
        }
        if let Some(&after) = lists[route].get(position + 1) {
            push_shaw_candidate(&mut pool, &mut seen, after, pivot, removed, present, RELATED_POOL);
        }
    }
    if !all.is_empty() {
        let start = (mix64(seed) % all.len() as u64) as usize;
        for offset in 0..all.len() {
            if pool.len() >= RELATED_POOL {
                break;
            }
            work.structural(stop)?;
            push_shaw_candidate(&mut pool, &mut seen, all[(start + offset) % all.len()], pivot, removed, present, RELATED_POOL);
        }
    }
    let pivot_route = owner.get(&pivot).map(|&(route, _)| route);
    let mut best = None;
    for item in pool {
        work.structural(stop)?;
        let relatedness = per.shaw_relatedness(pivot, item, pivot_route == owner.get(&item).map(|&(route, _)| route));
        let tie = mix64(seed ^ mix64(item as i64 as u64));
        if best.is_none_or(|current: (u64, u64, i32)| (relatedness, tie, item) < current) {
            best = Some((relatedness, tie, item));
        }
    }
    Ok(best.map(|(_, _, item)| item))
}

fn random_available(
    all: &[i32],
    removed: &HashSet<i32>,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Option<i32>, Abort> {
    if all.len() == removed.len() {
        return Ok(None);
    }
    if !work.is_bounded() {
        let available: Vec<i32> = all.iter().copied().filter(|item| !removed.contains(item)).collect();
        return Ok(Some(available[(mix64(seed) % available.len() as u64) as usize]));
    }
    let start = (mix64(seed) % all.len() as u64) as usize;
    for offset in 0..all.len() {
        work.structural(stop)?;
        let item = all[(start + offset) % all.len()];
        if !removed.contains(&item) {
            return Ok(Some(item));
        }
    }
    Ok(None)
}

struct WorstRemovalCandidate {
    score: Score,
    tie: u64,
    ordinal: usize,
    item: i32,
}

fn worst_removal_cmp(left: &WorstRemovalCandidate, right: &WorstRemovalCandidate) -> std::cmp::Ordering {
    left.score.cmp(&right.score).then_with(|| left.tie.cmp(&right.tie)).then_with(|| left.ordinal.cmp(&right.ordinal))
}

fn worst_heap_sift_up(
    heap: &mut [WorstRemovalCandidate],
    mut index: usize,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<(), Abort> {
    while index != 0 {
        let parent = (index - 1) / 2;
        work.structural(stop)?;
        if !worst_removal_cmp(&heap[parent], &heap[index]).is_lt() {
            break;
        }
        work.structural(stop)?;
        heap.swap(parent, index);
        index = parent;
    }
    Ok(())
}

fn worst_heap_sift_down(
    heap: &mut [WorstRemovalCandidate],
    mut index: usize,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<(), Abort> {
    loop {
        let left = index.saturating_mul(2).saturating_add(1);
        if left >= heap.len() {
            return Ok(());
        }
        let right = left.saturating_add(1);
        let mut larger = left;
        if right < heap.len() {
            work.structural(stop)?;
            if worst_removal_cmp(&heap[right], &heap[left]).is_gt() {
                larger = right;
            }
        }
        work.structural(stop)?;
        if !worst_removal_cmp(&heap[larger], &heap[index]).is_gt() {
            return Ok(());
        }
        work.structural(stop)?;
        heap.swap(index, larger);
        index = larger;
    }
}

fn retain_worst_removal(
    heap: &mut Vec<WorstRemovalCandidate>,
    candidate: WorstRemovalCandidate,
    limit: usize,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<(), Abort> {
    if heap.len() < limit {
        work.structural(stop)?;
        heap.push(candidate);
        let index = heap.len() - 1;
        return worst_heap_sift_up(heap, index, work, stop);
    }

    work.structural(stop)?;
    if !worst_removal_cmp(&candidate, &heap[0]).is_lt() {
        return Ok(());
    }
    work.structural(stop)?;
    heap[0] = candidate;
    worst_heap_sift_down(heap, 0, work, stop)
}

fn destroy_worst(
    per: &PerList,
    state: &State,
    lists: &mut Vec<Vec<i32>>,
    target: usize,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Vec<i32>, Abort> {
    if target == 0 {
        if stop.load(Ordering::Relaxed) {
            return Err(Abort::Interrupted);
        }
        return Ok(Vec::new());
    }

    let limit = target;
    // Admit the bounded top-k storage before allocating it. Heap comparisons
    // and swaps are charged separately because their exact count depends on
    // the deterministic ranking encountered below.
    work.structural_many(limit, stop)?;
    let mut selected_heap = Vec::with_capacity(limit);
    let mut list_scratch = EvalScratch::default();
    let mut score_scratch = EvalScratch::default();
    let mut ordinal = 0usize;
    for (list, contents) in state.lists.iter().enumerate() {
        for pos in 0..contents.len() {
            work.candidate(stop)?;
            let view = RemoveView::new(contents, pos);
            per.metrics.record_candidate();
            let replacement = trial_list_score_view(per, state, list, &view, None, &mut list_scratch);
            let score = score_with_replaced_list(per, state, list, &replacement, &view, 0, &mut score_scratch);
            let item = contents[pos];
            let candidate = WorstRemovalCandidate { score, tie: mix64(seed ^ item as i64 as u64), ordinal, item };
            retain_worst_removal(&mut selected_heap, candidate, limit, work, stop)?;
            ordinal = ordinal.saturating_add(1);
        }
    }

    work.structural_many(selected_heap.len(), stop)?;
    let mut selected = HashSet::with_capacity(selected_heap.len());
    for candidate in selected_heap {
        if stop.load(Ordering::Relaxed) {
            return Err(Abort::Interrupted);
        }
        selected.insert(candidate.item);
    }

    let rewrite_work = u64::try_from(lists.len())
        .unwrap_or(u64::MAX)
        .saturating_add(u64::try_from(ordinal).unwrap_or(u64::MAX))
        .saturating_add(u64::try_from(selected.len()).unwrap_or(u64::MAX));
    work.structural_units(rewrite_work, stop)?;
    let mut rewritten = Vec::with_capacity(lists.len());
    let mut removed = Vec::with_capacity(selected.len());
    for list in lists.iter() {
        if stop.load(Ordering::Relaxed) {
            return Err(Abort::Interrupted);
        }
        let mut kept = Vec::with_capacity(list.len());
        for &item in list {
            if stop.load(Ordering::Relaxed) {
                return Err(Abort::Interrupted);
            }
            if selected.contains(&item) {
                removed.push(item);
            } else {
                kept.push(item);
            }
        }
        rewritten.push(kept);
    }

    // The mutable partition changes only after every selection and rewrite
    // step has completed. Exhaustion or interruption therefore leaves the
    // caller's state byte-for-byte unchanged.
    work.structural(stop)?;
    *lists = rewritten;
    Ok(removed)
}

fn repair_routes(state: &State, per: &PerList, allow_empty: bool) -> Vec<usize> {
    if allow_empty {
        return active_list_indices(&state.lists, per.interchangeable_lists.get());
    }
    let nonempty: Vec<usize> = (0..state.lists.len()).filter(|&list| !state.lists[list].is_empty()).collect();
    if nonempty.is_empty() {
        active_list_indices(&state.lists, per.interchangeable_lists.get())
    } else {
        nonempty
    }
}

struct RepairIndex {
    locations: HashMap<i32, (usize, usize)>,
}

struct RepairWorkspace {
    index: RepairIndex,
    routes: Vec<usize>,
    allowed: Vec<bool>,
    fleet_reduction: bool,
}

impl RepairWorkspace {
    fn build(state: &State, per: &PerList, allow_empty: bool, work: &mut WorkCounter, stop: &AtomicBool) -> Result<Self, Abort> {
        work.structural_many(state.lists.len(), stop)?;
        let routes = repair_routes(state, per, allow_empty);
        work.structural_many(state.lists.len(), stop)?;
        let mut allowed = vec![false; state.lists.len()];
        for &route in &routes {
            allowed[route] = true;
        }
        Ok(Self { index: RepairIndex::build(state, work, stop)?, routes, allowed, fleet_reduction: !allow_empty })
    }
}

impl RepairIndex {
    fn build(state: &State, work: &mut WorkCounter, stop: &AtomicBool) -> Result<Self, Abort> {
        // Grow only after each insertion has been debited. Reserving n slots up
        // front would itself be unbounded work before the first stop check.
        let mut locations = HashMap::new();
        for (list, route) in state.lists.iter().enumerate() {
            work.repair_index(stop)?;
            for (position, &item) in route.iter().enumerate() {
                work.repair_index(stop)?;
                locations.insert(item, (list, position));
            }
        }
        Ok(Self { locations })
    }

    /// Update only the suffix shifted by one insertion. The old implementation
    /// rebuilt all n locations after every repaired item, yielding O(nr) index
    /// work for r removed customers.
    fn record_insertion(
        &mut self,
        state: &State,
        list: usize,
        position: usize,
        work: &mut WorkCounter,
        stop: &AtomicBool,
    ) -> Result<(), Abort> {
        for (offset, &item) in state.lists[list].iter().enumerate().skip(position) {
            work.repair_index(stop)?;
            self.locations.insert(item, (list, offset));
        }
        Ok(())
    }
}

fn push_bounded_placement(
    placements: &mut Vec<(usize, usize)>,
    seen: &mut HashSet<(usize, usize)>,
    placement: (usize, usize),
    limit: usize,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<(), Abort> {
    if placements.len() >= limit {
        return Ok(());
    }
    work.structural(stop)?;
    if seen.insert(placement) {
        placements.push(placement);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_neighbor_placements(
    placements: &mut Vec<(usize, usize)>,
    seen: &mut HashSet<(usize, usize)>,
    workspace: &RepairWorkspace,
    neighbor: i32,
    limit: usize,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<(), Abort> {
    work.structural(stop)?;
    let Some(&(list, position)) = workspace.index.locations.get(&neighbor) else { return Ok(()) };
    if !workspace.allowed[list] {
        return Ok(());
    }
    push_bounded_placement(placements, seen, (list, position), limit, work, stop)?;
    push_bounded_placement(placements, seen, (list, position + 1), limit, work, stop)
}

fn repair_placements_bounded(
    per: &PerList,
    state: &State,
    item: i32,
    workspace: &RepairWorkspace,
    placement_limit: usize,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Vec<(usize, usize)>, Abort> {
    let mut placements = Vec::with_capacity(placement_limit.min(MAX_REPAIR_PLACEMENTS));
    let mut seen = HashSet::with_capacity(placement_limit.min(MAX_REPAIR_PLACEMENTS));
    if let Some(neighbors) = &per.candidates {
        if workspace.fleet_reduction {
            for &list in &workspace.routes {
                push_bounded_placement(&mut placements, &mut seen, (list, 0), placement_limit, work, stop)?;
                push_bounded_placement(&mut placements, &mut seen, (list, state.lists[list].len()), placement_limit, work, stop)?;
            }
        }
        let geometric = neighbors.neighbors(item);
        let semantic = if workspace.fleet_reduction { neighbors.semantic_neighbors(item) } else { &[] };
        for index in 0..geometric.len().max(semantic.len()) {
            if placements.len() >= placement_limit {
                break;
            }
            if let Some(&neighbor) = semantic.get(index) {
                push_neighbor_placements(&mut placements, &mut seen, workspace, neighbor, placement_limit, work, stop)?;
            }
            if let Some(&neighbor) = geometric.get(index) {
                push_neighbor_placements(&mut placements, &mut seen, workspace, neighbor, placement_limit, work, stop)?;
            }
        }
        if !workspace.fleet_reduction {
            for &list in &workspace.routes {
                push_bounded_placement(&mut placements, &mut seen, (list, 0), placement_limit, work, stop)?;
                push_bounded_placement(&mut placements, &mut seen, (list, state.lists[list].len()), placement_limit, work, stop)?;
            }
        }
        return Ok(placements);
    }

    // Without a candidate graph, sample at most `placement_limit` gaps evenly
    // over all eligible routes. This avoids collecting and sorting O(n) gaps
    // before applying the bound.
    let mut cumulative = Vec::with_capacity(workspace.routes.len());
    let mut total_gaps = 0usize;
    for &list in &workspace.routes {
        work.structural(stop)?;
        total_gaps = total_gaps.saturating_add(state.lists[list].len().saturating_add(1));
        cumulative.push((total_gaps, list));
    }
    let sample_count = total_gaps.min(placement_limit);
    if sample_count == 0 {
        return Ok(placements);
    }
    let offset = (mix64(item as i64 as u64) % total_gaps as u64) as usize;
    for sample in 0..sample_count {
        let rank = (offset + sample.saturating_mul(total_gaps) / sample_count) % total_gaps;
        let route_index = cumulative.partition_point(|&(end, _)| end <= rank);
        let previous_end = route_index.checked_sub(1).map_or(0, |index| cumulative[index].0);
        let list = cumulative[route_index].1;
        push_bounded_placement(&mut placements, &mut seen, (list, rank - previous_end), placement_limit, work, stop)?;
    }
    Ok(placements)
}

fn repair_placements_unbounded(
    per: &PerList,
    state: &State,
    allow_empty: bool,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Vec<(usize, usize)>, Abort> {
    let mut placements = Vec::new();
    for list in repair_routes(state, per, allow_empty) {
        for position in 0..=state.lists[list].len() {
            placements.push((list, position));
        }
    }
    for _ in &placements {
        work.generated(stop)?;
    }
    Ok(placements)
}

fn repair_placement_limit(work: &WorkCounter, calls: u64) -> Result<usize, Abort> {
    if !work.is_bounded() {
        return Ok(usize::MAX);
    }
    // Leave half of the generation budget for route rescoring, mutable-index
    // suffix updates, and the final canonical rebuild.
    // One accepted placement takes up to three bounded generation units: a
    // neighbor lookup plus the two adjacent insertion gaps. The extra factor
    // prevents the cap itself from promising more placements than the real
    // structural budget can enumerate.
    let generated_capacity = work.remaining_generated() / 2 / 3;
    let available = generated_capacity.min(work.remaining_evaluated());
    let limit = available.checked_div(calls.max(1)).unwrap_or(0).min(MAX_REPAIR_PLACEMENTS as u64);
    if limit == 0 {
        Err(Abort::BudgetExhausted)
    } else {
        Ok(usize::try_from(limit).unwrap_or(MAX_REPAIR_PLACEMENTS))
    }
}

fn regret_repair_calls(items: usize) -> u64 {
    let sampled = items.min(REGRET_ITEM_SAMPLE);
    let triangular = sampled.saturating_mul(sampled.saturating_add(1)) / 2;
    let remaining = items.saturating_sub(sampled).saturating_mul(sampled);
    u64::try_from(triangular.saturating_add(remaining)).unwrap_or(u64::MAX)
}

fn repair_positions(per: &PerList, route: &[i32], item: i32, work: &mut WorkCounter, stop: &AtomicBool) -> Result<Vec<usize>, Abort> {
    let mut all = Vec::with_capacity(route.len() + 1);
    let mut granular = Vec::with_capacity(route.len().min(64) + 1);
    for position in 0..=route.len() {
        work.generated(stop)?;
        all.push(position);
        if per.candidates.as_ref().is_none_or(|neighbors| {
            route.is_empty()
                || position.checked_sub(1).is_some_and(|before| neighbors.contains_semantic(item, route[before]))
                || route.get(position).is_some_and(|&after| neighbors.contains_semantic(item, after))
        }) {
            granular.push(position);
        }
    }
    Ok(if granular.is_empty() { all } else { granular })
}

#[allow(clippy::too_many_arguments)]
fn repair_greedy(
    per: &PerList,
    state: &mut State,
    removed: &[i32],
    allow_empty: bool,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<(), Abort> {
    let mut scratch = EvalScratch::default();
    let mut workspace = if work.is_bounded() { Some(RepairWorkspace::build(state, per, allow_empty, work, stop)?) } else { None };
    let placement_limit = repair_placement_limit(work, u64::try_from(removed.len()).unwrap_or(u64::MAX))?;
    for &item in removed {
        let mut best: Option<(Score, u64, usize, usize, ListScore)> = None;
        let placements = if let Some(workspace) = &workspace {
            repair_placements_bounded(per, state, item, workspace, placement_limit, work, stop)?
        } else {
            repair_placements_unbounded(per, state, allow_empty, work, stop)?
        };
        for (list, pos) in placements {
            work.evaluated(stop)?;
            let candidate = InsertView::new(&state.lists[list], pos, item);
            per.metrics.record_candidate();
            let next = trial_list_score_view(per, state, list, &candidate, None, &mut scratch);
            let global_delta = per.globals.delta(&state.item_list, &[(item, list)]);
            let score = score_with_replaced_list(per, state, list, &next, &candidate, global_delta, &mut scratch);
            let tie = mix64(seed ^ item as i64 as u64 ^ ((list as u64) << 32) ^ pos as u64);
            if best.as_ref().is_none_or(|(best_score, best_tie, _, _, _)| score < *best_score || (score == *best_score && tie < *best_tie))
            {
                best = Some((score, tie, list, pos, next));
            }
        }
        let Some((_, _, list, pos, expected)) = best else { return Err(Abort::Infeasible) };
        state.lists[list].insert(pos, item);
        work.structural_many(state.lists[list].len(), stop)?;
        if !state.rescore_interruptible(per, list, stop) {
            return Err(if stop.load(Ordering::Relaxed) { Abort::Interrupted } else { Abort::Infeasible });
        }
        debug_assert_eq!(state.scores[list].violation, expected.violation);
        state.set_item_list(per, item, list);
        state.global_viol = per.globals.total(&state.item_list);
        if let Some(workspace) = &mut workspace {
            workspace.index.record_insertion(state, list, pos, work, stop)?;
        }
    }
    Ok(())
}

struct RegretRepair {
    k: usize,
    blink: bool,
    allow_empty: bool,
    seed: u64,
}

fn repair_regret(
    per: &PerList,
    state: &mut State,
    removed: &[i32],
    settings: RegretRepair,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<(), Abort> {
    work.structural_many(removed.len(), stop)?;
    let mut pending = removed.to_vec();
    let mut scratch = EvalScratch::default();
    let mut workspace = if work.is_bounded() { Some(RepairWorkspace::build(state, per, settings.allow_empty, work, stop)?) } else { None };
    let placement_limit = repair_placement_limit(work, regret_repair_calls(removed.len()))?;
    while !pending.is_empty() {
        let mut choice: Option<(usize, bool, ScoreRegret, u64, usize, usize, ListScore)> = None;
        let sample = if work.is_bounded() { pending.len().min(REGRET_ITEM_SAMPLE) } else { pending.len() };
        let start =
            if work.is_bounded() { (mix64(settings.seed ^ mix64(pending.len() as u64)) % pending.len() as u64) as usize } else { 0 };
        for offset in 0..sample {
            let idx = (start + offset) % pending.len();
            let item = pending[idx];
            let mut topk: SmallVec<[Score; 4]> = SmallVec::with_capacity(settings.k);
            let mut best_place: Option<(Score, u64, usize, usize, ListScore)> = None;
            let placements = if let Some(workspace) = &workspace {
                repair_placements_bounded(per, state, item, workspace, placement_limit, work, stop)?
            } else {
                repair_placements_unbounded(per, state, settings.allow_empty, work, stop)?
            };
            for (list, pos) in placements {
                let blink_sample = mix64(settings.seed ^ mix64(item as i64 as u64) ^ mix64(list as u64) ^ mix64(pos as u64));
                if settings.blink && best_place.is_some() && blink_sample % 100 < 15 {
                    continue;
                }
                work.evaluated(stop)?;
                let candidate = InsertView::new(&state.lists[list], pos, item);
                per.metrics.record_candidate();
                let next = trial_list_score_view(per, state, list, &candidate, None, &mut scratch);
                let global_delta = per.globals.delta(&state.item_list, &[(item, list)]);
                let score = score_with_replaced_list(per, state, list, &next, &candidate, global_delta, &mut scratch);
                let tie = mix64(settings.seed ^ item as i64 as u64 ^ ((list as u64) << 32) ^ pos as u64);
                let insert = topk.partition_point(|candidate_score| candidate_score <= &score);
                if insert < settings.k {
                    topk.insert(insert, score.clone());
                    topk.truncate(settings.k);
                }
                if best_place
                    .as_ref()
                    .is_none_or(|(best_score, best_tie, _, _, _)| score < *best_score || (score == *best_score && tie < *best_tie))
                {
                    best_place = Some((score, tie, list, pos, next));
                }
            }
            let Some((best_score, best_tie, best_list, best_pos, best_list_score)) = best_place else { return Err(Abort::Infeasible) };
            let forced = topk.len() < settings.k;
            let alternative = topk.get(settings.k.saturating_sub(1)).unwrap_or_else(|| topk.last().expect("a best placement was recorded"));
            let regret = score_regret(alternative, &best_score);
            if choice.as_ref().is_none_or(|(_, old_forced, old_regret, old_tie, _, _, _)| {
                (forced, regret) > (*old_forced, *old_regret) || ((forced, regret) == (*old_forced, *old_regret) && best_tie < *old_tie)
            }) {
                choice = Some((idx, forced, regret, best_tie, best_list, best_pos, best_list_score));
            }
        }
        let (idx, _, _, _, list, pos, expected) = choice.expect("pending is non-empty");
        let item = pending.swap_remove(idx);
        state.lists[list].insert(pos, item);
        work.structural_many(state.lists[list].len(), stop)?;
        if !state.rescore_interruptible(per, list, stop) {
            return Err(if stop.load(Ordering::Relaxed) { Abort::Interrupted } else { Abort::Infeasible });
        }
        debug_assert_eq!(state.scores[list].violation, expected.violation);
        state.set_item_list(per, item, list);
        state.global_viol = per.globals.total(&state.item_list);
        if let Some(workspace) = &mut workspace {
            workspace.index.record_insertion(state, list, pos, work, stop)?;
        }
    }
    Ok(())
}

/// Preserve the legacy generic ALNS trajectory. Routing uses the bounded
/// scheduler and performs its local improvement in separate slices.
fn polish(per: &PerList, state: &mut State, stop: &AtomicBool, steps: usize, seed: u64) -> bool {
    let mut memory = SearchMemory::new(state.lists.len());
    for step in 0..steps {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        let Some(mv) = best_improving_move(per, state, stop, &mut memory, seed ^ mix64(step as u64)) else {
            return !stop.load(Ordering::Relaxed);
        };
        if !apply_move(per, state, mv, stop) {
            return false;
        }
        memory.reset_touched(mv);
    }
    true
}

/// Larger bounded neighborhoods scheduled independently from ordinary ALNS.
/// They all rebuild and rescore the final state through the canonical list
/// evaluator before returning it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MacroOperator {
    RouteElimination,
    EjectionChain,
    ChainRelocate,
    GuidedSegmentExchange,
}

impl MacroOperator {
    pub(super) const ALL: [Self; 4] = [Self::RouteElimination, Self::EjectionChain, Self::ChainRelocate, Self::GuidedSegmentExchange];

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::RouteElimination => "route-elimination",
            Self::EjectionChain => "ejection-chain",
            Self::ChainRelocate => "chain-relocate",
            Self::GuidedSegmentExchange => "guided-segment-exchange",
        }
    }

    pub(super) const fn index(self) -> usize {
        match self {
            Self::RouteElimination => 0,
            Self::EjectionChain => 1,
            Self::ChainRelocate => 2,
            Self::GuidedSegmentExchange => 3,
        }
    }
}

pub(super) fn build_macro_candidate_bounded(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    operator: MacroOperator,
    seed: u64,
    budget: AlnsWorkBudget,
    stop: &AtomicBool,
) -> AlnsBuildRun {
    if matches!(operator, MacroOperator::RouteElimination) {
        return build_route_elimination_candidate_bounded(model, per, current, seed, budget, stop);
    }

    let started = CpuTimer::start();
    let mut work = WorkCounter::new(budget);
    let result = if stop.load(Ordering::Relaxed) {
        Err(Abort::Interrupted)
    } else if budget.generated == 0 || budget.evaluated == 0 {
        Err(Abort::BudgetExhausted)
    } else {
        match operator {
            MacroOperator::RouteElimination => unreachable!(),
            MacroOperator::EjectionChain => ejection_chain(model, per, current, seed, &mut work, stop),
            MacroOperator::ChainRelocate => chain_relocate(model, per, current, seed, &mut work, stop),
            MacroOperator::GuidedSegmentExchange => guided_segment_exchange(model, per, current, seed, &mut work, stop),
        }
    };
    let (status, candidate) = match result {
        Ok(Some(candidate)) => (AlnsBuildStatus::Built, Some(candidate)),
        Ok(None) => (AlnsBuildStatus::Infeasible, None),
        Err(abort) => (abort.status(), None),
    };
    AlnsBuildRun {
        status,
        candidate,
        generated: work.generated,
        evaluated: work.evaluated,
        work_units: work.generated.saturating_add(work.evaluated),
        structural_work: work.structural,
        repair_index_work: work.repair_index,
        canonical_rebuilds: work.canonical_rebuilds,
        cpu_nanos: started.elapsed_nanos(),
        destroy_generated: 0,
        destroy_evaluated: 0,
        destroy_cpu_nanos: 0,
        repair_generated: 0,
        repair_evaluated: 0,
        repair_cpu_nanos: 0,
        destroy_executed: false,
        repair_executed: false,
        removed: 0,
        eliminated_route: false,
    }
}

fn build_route_elimination_candidate_bounded(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    seed: u64,
    budget: AlnsWorkBudget,
    stop: &AtomicBool,
) -> AlnsBuildRun {
    let started = CpuTimer::start();
    let mut work = WorkCounter::new(budget);
    let mut breakdown = BuildBreakdown::default();
    let choice = AlnsChoice { destroy: DestroyOperator::Route, repair: RepairOperator::Regret2, target: 1 };
    let greedy = build_candidate_inner(model, per, current, choice, seed, stop, &mut work, &mut breakdown);
    let mut removed = 0;
    let mut candidate = None;
    let mut terminal = None;

    match greedy {
        Ok((built, count, _)) => {
            removed = count;
            if route_is_eliminated(per, current, &built) {
                candidate = Some(built);
            }
        }
        Err(Abort::Interrupted) => terminal = Some(Abort::Interrupted),
        Err(Abort::BudgetExhausted) => {
            terminal = Some(Abort::BudgetExhausted);
        }
        Err(Abort::Infeasible) => {}
    }

    if candidate.is_none() && terminal.is_none() {
        match eliminate_route_with_ejections(model, per, current, seed ^ 0x8EBC_6AF0_9C88_C6E3, &mut work, stop) {
            Ok(Some(built)) => candidate = Some(built),
            Ok(None) => terminal = Some(Abort::Infeasible),
            Err(abort) => terminal = Some(abort),
        }
    }

    let status = if candidate.is_some() { AlnsBuildStatus::Built } else { terminal.unwrap_or(Abort::Infeasible).status() };
    AlnsBuildRun {
        status,
        eliminated_route: candidate.is_some(),
        candidate,
        generated: work.generated,
        evaluated: work.evaluated,
        work_units: work.generated.saturating_add(work.evaluated),
        structural_work: work.structural,
        repair_index_work: work.repair_index,
        canonical_rebuilds: work.canonical_rebuilds,
        cpu_nanos: started.elapsed_nanos(),
        destroy_generated: breakdown.destroy_generated,
        destroy_evaluated: breakdown.destroy_evaluated,
        destroy_cpu_nanos: breakdown.destroy_cpu_nanos,
        repair_generated: breakdown.repair_generated,
        repair_evaluated: breakdown.repair_evaluated,
        repair_cpu_nanos: breakdown.repair_cpu_nanos,
        destroy_executed: breakdown.destroy_executed,
        repair_executed: breakdown.repair_executed,
        removed,
    }
}

fn route_is_eliminated(per: &PerList, current: &State, candidate: &State) -> bool {
    nonempty_routes(&candidate.lists) < nonempty_routes(&current.lists) && full_score_raw(per, candidate).violation == 0
}

fn improving_raw(before: &Score, after: &Score) -> bool {
    after < before && (before.violation != 0 || after.violation == 0)
}

fn macro_move_materialization_work(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    mv: Move,
    stop: &AtomicBool,
) -> Result<u64, Abort> {
    let edit_work = match mv {
        Move::OrOpt { src, start, len, dst, dst_pos } if src != dst => current.lists[src]
            .len()
            .saturating_sub(start)
            .saturating_add(current.lists[dst].len().saturating_sub(dst_pos).saturating_mul(len))
            .saturating_add(len),
        Move::CrossExchange { a, start_a, len_a, b, start_b, len_b } => current.lists[a]
            .len()
            .saturating_sub(start_a)
            .saturating_add(current.lists[b].len().saturating_sub(start_b))
            .saturating_add(len_a)
            .saturating_add(len_b),
        _ => return Err(Abort::Infeasible),
    };
    let clone_work = state_clone_work(model, current);
    let mut poll = FloorStopPoll::new(stop).ok_or(Abort::Interrupted)?;
    let rebuild = state_rebuild_work_from_lengths_polled(
        model,
        per,
        (0..current.lists.len()).map(|list| match mv {
            Move::OrOpt { src, len, .. } if list == src => current.lists[list].len().saturating_sub(len),
            Move::OrOpt { len, dst, .. } if list == dst => current.lists[list].len().saturating_add(len),
            Move::CrossExchange { a, len_a, len_b, .. } if list == a => {
                current.lists[list].len().saturating_sub(len_a).saturating_add(len_b)
            }
            Move::CrossExchange { b, len_a, len_b, .. } if list == b => {
                current.lists[list].len().saturating_sub(len_b).saturating_add(len_a)
            }
            _ => current.lists[list].len(),
        }),
        &mut poll,
    )
    .ok_or(Abort::Interrupted)?;
    poll.finish().ok_or(Abort::Interrupted)?;
    Ok(clone_work.saturating_add(u64::try_from(edit_work).unwrap_or(u64::MAX)).saturating_add(rebuild))
}

fn materialize_macro_move(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    mv: Move,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<State, Abort> {
    let reserved = macro_move_materialization_work(model, per, current, mv, stop)?;
    work.structural_units(reserved, stop)?;

    let mut lists = clone_lists_reserved(&current.lists, stop)?;
    match mv {
        Move::OrOpt { src, start, len, dst, dst_pos } => {
            let segment = lists[src].drain(start..start + len).collect::<SmallVec<[_; 3]>>();
            let position = dst_pos.min(lists[dst].len());
            lists[dst].splice(position..position, segment);
        }
        Move::CrossExchange { a, start_a, len_a, b, start_b, len_b } => {
            let segment_a = lists[a][start_a..start_a + len_a].to_vec();
            let segment_b = lists[b][start_b..start_b + len_b].to_vec();
            lists[a].splice(start_a..start_a + len_a, segment_b);
            lists[b].splice(start_b..start_b + len_b, segment_a);
        }
        _ => unreachable!("macro materialization validates its move before cloning"),
    }
    let state = state_from_lists(model, per, lists, stop)?;
    work.canonical_rebuilds = work.canonical_rebuilds.saturating_add(1);
    Ok(state)
}

fn rotated_indices(len: usize, seed: u64) -> impl Iterator<Item = usize> {
    let start = if len == 0 { 0 } else { (mix64(seed) % len as u64) as usize };
    (0..len).map(move |offset| (start + offset) % len.max(1))
}

fn chain_relocate(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Option<State>, Abort> {
    let route_count = current.lists.len();
    if route_count < 2 {
        return Ok(None);
    }
    let before = full_score_raw(per, current);
    let mut workspace = MoveScoreWorkspace::default();
    for src in rotated_indices(route_count, seed) {
        let src_len = current.lists[src].len();
        for len in 2..=3.min(src_len) {
            for start in 0..=src_len - len {
                let segment = &current.lists[src][start..start + len];
                for dst in rotated_indices(route_count, seed ^ mix64(src as u64)) {
                    if src == dst {
                        continue;
                    }
                    for position in repair_positions(per, &current.lists[dst], segment[0], work, stop)? {
                        let mv = Move::OrOpt { src, start, len, dst, dst_pos: position };
                        let materialization = macro_move_materialization_work(model, per, current, mv, stop)?;
                        reserve_trial_scores(
                            model,
                            per,
                            current,
                            TrialScoreWork {
                                touched: [(src, src_len.saturating_sub(len)), (dst, current.lists[dst].len().saturating_add(len))]
                                    .into_iter()
                                    .collect(),
                                override_items: len,
                                materialization_headroom: materialization,
                            },
                            work,
                            stop,
                        )?;
                        let Some(after) = score_move_raw(per, current, mv, &mut workspace) else {
                            continue;
                        };
                        if improving_raw(&before, &after) {
                            let candidate = materialize_macro_move(model, per, current, mv, work, stop)?;
                            debug_assert!(full_score_raw(per, &candidate) == after);
                            return Ok(Some(candidate));
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

fn guided_boundary(per: &PerList, prefix: Option<i32>, segment: &[i32], suffix: Option<i32>) -> bool {
    let Some(neighbors) = &per.candidates else {
        return true;
    };
    prefix.zip(segment.first().copied()).is_some_and(|(from, to)| neighbors.contains(from, to))
        || segment.last().copied().zip(suffix).is_some_and(|(from, to)| neighbors.contains(from, to))
}

#[derive(Clone, Copy)]
struct GuidedSegment {
    list: usize,
    start: usize,
    len: usize,
}

fn segments_starting_at(list: usize, position: usize, route_len: usize) -> SmallVec<[GuidedSegment; 3]> {
    let mut segments = SmallVec::new();
    for len in 1..=COMPOUND_ROUTE_GROWTH.min(route_len.saturating_sub(position)) {
        segments.push(GuidedSegment { list, start: position, len });
    }
    segments
}

fn segments_ending_at(list: usize, position: usize) -> SmallVec<[GuidedSegment; 3]> {
    let mut segments = SmallVec::new();
    for len in 1..=COMPOUND_ROUTE_GROWTH.min(position.saturating_add(1)) {
        segments.push(GuidedSegment { list, start: position + 1 - len, len });
    }
    segments
}

fn segments_after(list: usize, position: usize, route_len: usize) -> SmallVec<[GuidedSegment; 3]> {
    segments_starting_at(list, position.saturating_add(1), route_len)
}

fn segments_before(list: usize, position: usize) -> SmallVec<[GuidedSegment; 3]> {
    let mut segments = SmallVec::new();
    for len in 1..=COMPOUND_ROUTE_GROWTH.min(position) {
        segments.push(GuidedSegment { list, start: position - len, len });
    }
    segments
}

fn cross_exchange_move(left: GuidedSegment, right: GuidedSegment) -> Option<Move> {
    if left.list == right.list {
        return None;
    }
    let (left, right) = if left.list < right.list { (left, right) } else { (right, left) };
    Some(Move::CrossExchange { a: left.list, start_a: left.start, len_a: left.len, b: right.list, start_b: right.start, len_b: right.len })
}

struct GuidedExchangeContext<'a> {
    model: &'a CollectionModel,
    per: &'a PerList,
    current: &'a State,
    before: Score,
    workspace: MoveScoreWorkspace,
    seen: HashSet<Move>,
    work: &'a mut WorkCounter,
    stop: &'a AtomicBool,
}

impl GuidedExchangeContext<'_> {
    fn try_pair(&mut self, left: GuidedSegment, right: GuidedSegment) -> Result<Option<State>, Abort> {
        self.work.structural(self.stop)?;
        let Some(mv) = cross_exchange_move(left, right) else {
            return Ok(None);
        };
        if !self.seen.insert(mv) {
            return Ok(None);
        }
        debug_assert!(match mv {
            Move::CrossExchange { a, start_a, len_a, b, start_b, len_b } => {
                let left = &self.current.lists[a];
                let right = &self.current.lists[b];
                guided_boundary(
                    self.per,
                    start_a.checked_sub(1).map(|index| left[index]),
                    &right[start_b..start_b + len_b],
                    left.get(start_a + len_a).copied(),
                ) || guided_boundary(
                    self.per,
                    start_b.checked_sub(1).map(|index| right[index]),
                    &left[start_a..start_a + len_a],
                    right.get(start_b + len_b).copied(),
                )
            }
            _ => false,
        });
        self.work.generated(self.stop)?;
        let materialization = macro_move_materialization_work(self.model, self.per, self.current, mv, self.stop)?;
        let Move::CrossExchange { a, len_a, b, len_b, .. } = mv else { unreachable!("guided exchange only builds cross-exchange moves") };
        reserve_trial_scores(
            self.model,
            self.per,
            self.current,
            TrialScoreWork {
                touched: [
                    (a, self.current.lists[a].len().saturating_sub(len_a).saturating_add(len_b)),
                    (b, self.current.lists[b].len().saturating_sub(len_b).saturating_add(len_a)),
                ]
                .into_iter()
                .collect(),
                override_items: len_a.saturating_add(len_b),
                materialization_headroom: materialization,
            },
            self.work,
            self.stop,
        )?;
        let Some(after) = score_move_raw(self.per, self.current, mv, &mut self.workspace) else {
            return Ok(None);
        };
        if !improving_raw(&self.before, &after) {
            return Ok(None);
        }
        let candidate = materialize_macro_move(self.model, self.per, self.current, mv, self.work, self.stop)?;
        debug_assert!(full_score_raw(self.per, &candidate) == after);
        Ok(Some(candidate))
    }

    fn try_pairs(&mut self, left: &[GuidedSegment], right: &[GuidedSegment]) -> Result<Option<State>, Abort> {
        for &left in left {
            for &right in right {
                if let Some(candidate) = self.try_pair(left, right)? {
                    return Ok(Some(candidate));
                }
            }
        }
        Ok(None)
    }
}

fn unfiltered_segment_exchange(context: &mut GuidedExchangeContext<'_>, seed: u64) -> Result<Option<State>, Abort> {
    let mut active = Vec::new();
    for (list, route) in context.current.lists.iter().enumerate() {
        context.work.structural(context.stop)?;
        if !route.is_empty() {
            active.push(list);
        }
    }
    if active.len() < 2 {
        return Ok(None);
    }
    for left_index in rotated_indices(active.len(), seed) {
        let left_list = active[left_index];
        let left = &context.current.lists[left_list];
        for right_index in rotated_indices(active.len(), seed ^ 0xA076_1D64_78BD_642F) {
            let right_list = active[right_index];
            if left_list == right_list {
                continue;
            }
            let right = &context.current.lists[right_list];
            for len_left in 1..=COMPOUND_ROUTE_GROWTH.min(left.len()) {
                for start_left in 0..=left.len() - len_left {
                    let left_segment = GuidedSegment { list: left_list, start: start_left, len: len_left };
                    for len_right in 1..=COMPOUND_ROUTE_GROWTH.min(right.len()) {
                        for start_right in 0..=right.len() - len_right {
                            let right_segment = GuidedSegment { list: right_list, start: start_right, len: len_right };
                            if let Some(candidate) = context.try_pair(left_segment, right_segment)? {
                                return Ok(Some(candidate));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

fn directly_guided_segment_exchange(
    context: &mut GuidedExchangeContext<'_>,
    candidates: &CandidateNeighbors,
    seed: u64,
) -> Result<Option<State>, Abort> {
    let locations = build_guided_location_index(context.model, context.current, context.work, context.stop)?;

    for list in rotated_indices(context.current.lists.len(), seed) {
        context.work.structural(context.stop)?;
        let route = &context.current.lists[list];
        for position in rotated_indices(route.len(), seed ^ mix64(list as u64)) {
            context.work.structural(context.stop)?;
            let item = route[position];
            let neighbors = candidates.neighbors(item);
            for neighbor_index in rotated_indices(neighbors.len(), seed ^ mix64(item as u64)) {
                context.work.structural(context.stop)?;
                let Some(&(other_list, other_position)) = locations.get(&neighbors[neighbor_index]) else {
                    continue;
                };
                let other_len = context.current.lists[other_list].len();
                if other_list == list || (route.len() == 1 && other_len == 1) {
                    continue;
                }
                let families = [
                    (segments_after(list, position, route.len()), segments_starting_at(other_list, other_position, other_len)),
                    (segments_after(other_list, other_position, other_len), segments_starting_at(list, position, route.len())),
                    (segments_before(list, position), segments_ending_at(other_list, other_position)),
                    (segments_before(other_list, other_position), segments_ending_at(list, position)),
                ];
                for (left, right) in families {
                    if let Some(candidate) = context.try_pairs(&left, &right)? {
                        return Ok(Some(candidate));
                    }
                }
            }
        }
    }
    Ok(None)
}

fn build_guided_location_index(
    model: &CollectionModel,
    current: &State,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<HashMap<i32, (usize, usize)>, Abort> {
    // A search state is a partition of the model items. Reserving the complete
    // index pass before allocating guarantees that a rejected slice cannot
    // allocate or partially populate an O(n) table. The reserved pass is then
    // polled, but not debited again, while the table is constructed.
    let item_count = model.items.len();
    let index_work = u64::try_from(current.lists.len()).unwrap_or(u64::MAX).saturating_add(u64::try_from(item_count).unwrap_or(u64::MAX));
    work.structural_units(index_work, stop)?;

    let mut locations = HashMap::with_capacity(item_count);
    for (list, route) in current.lists.iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            return Err(Abort::Interrupted);
        }
        for (position, &item) in route.iter().enumerate() {
            if position.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                return Err(Abort::Interrupted);
            }
            locations.insert(item, (list, position));
        }
    }
    if stop.load(Ordering::Relaxed) {
        return Err(Abort::Interrupted);
    }
    Ok(locations)
}

fn guided_segment_exchange(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Option<State>, Abort> {
    let route_count = current.lists.len();
    if route_count < 2 {
        return Ok(None);
    }
    let mut context = GuidedExchangeContext {
        model,
        per,
        current,
        before: full_score_raw(per, current),
        workspace: MoveScoreWorkspace::default(),
        seen: HashSet::new(),
        work,
        stop,
    };
    match &per.candidates {
        Some(candidates) => directly_guided_segment_exchange(&mut context, candidates, seed),
        None => unfiltered_segment_exchange(&mut context, seed),
    }
}

struct EjectionContext<'a> {
    model: &'a CollectionModel,
    per: &'a PerList,
    current: &'a State,
    before: Score,
    work: &'a mut WorkCounter,
    stop: &'a AtomicBool,
    seed: u64,
    list_scratch: EvalScratch,
    score_scratch: EvalScratch,
    overrides: Vec<(i32, usize)>,
}

struct EjectionOverlay<'a> {
    base: &'a [Vec<i32>],
    touched: SmallVec<[(usize, Vec<i32>); 8]>,
}

impl<'a> EjectionOverlay<'a> {
    fn new(base: &'a [Vec<i32>]) -> Self {
        Self { base, touched: SmallVec::new() }
    }

    fn route(&self, list: usize) -> &[i32] {
        self.touched.iter().find(|(owner, _)| *owner == list).map_or(self.base[list].as_slice(), |(_, route)| route.as_slice())
    }

    fn route_mut(&mut self, list: usize, work: &mut WorkCounter, stop: &AtomicBool) -> Result<&mut Vec<i32>, Abort> {
        if let Some(position) = self.touched.iter().position(|(owner, _)| *owner == list) {
            return Ok(&mut self.touched[position].1);
        }
        work.structural_units(u64::try_from(self.base[list].len().saturating_add(1)).unwrap_or(u64::MAX), stop)?;
        self.touched.push((list, clone_route_reserved(&self.base[list], stop)?));
        Ok(&mut self.touched.last_mut().expect("a touched route was just appended").1)
    }

    fn fork(&self, work: &mut WorkCounter, stop: &AtomicBool) -> Result<Self, Abort> {
        let units = self
            .touched
            .iter()
            .fold(0u64, |total, (_, route)| total.saturating_add(u64::try_from(route.len().saturating_add(1)).unwrap_or(u64::MAX)));
        work.structural_units(units, stop)?;
        let mut touched = SmallVec::new();
        for (list, route) in &self.touched {
            touched.push((*list, clone_route_reserved(route, stop)?));
        }
        Ok(Self { base: self.base, touched })
    }

    fn final_len(&self, list: usize) -> usize {
        self.route(list).len()
    }

    fn replace_route(&mut self, list: usize, route: Vec<i32>, work: &mut WorkCounter, stop: &AtomicBool) -> Result<(), Abort> {
        work.structural(stop)?;
        if let Some(position) = self.touched.iter().position(|(owner, _)| *owner == list) {
            self.touched[position].1 = route;
        } else {
            self.touched.push((list, route));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct PendingEjection {
    item: i32,
    origin: usize,
    depth: usize,
}

struct RouteEliminationOption {
    destination: usize,
    route: Vec<i32>,
    ejected: Option<PendingEjection>,
    rank: (u8, i64, i64, SmallVec<[i64; 4]>, u64),
}

struct InfeasibleInsertion {
    destination: usize,
    route: Vec<i32>,
    rank: (i64, i64, SmallVec<[i64; 4]>, u64),
}

struct RouteEliminationContext<'a> {
    model: &'a CollectionModel,
    per: &'a PerList,
    current: &'a State,
    before: Score,
    eliminated: usize,
    initial_removed: usize,
    seed: u64,
    work: &'a mut WorkCounter,
    stop: &'a AtomicBool,
    list_scratch: EvalScratch,
    score_scratch: EvalScratch,
    overrides: Vec<(i32, usize)>,
}

fn signed_list_objectives(per: &PerList, score: &ListScore) -> SmallVec<[i64; 4]> {
    score.objectives.iter().enumerate().map(|(tier, &value)| if per.senses[tier] { value } else { value.saturating_neg() }).collect()
}

fn retain_best<T, K: Ord>(entries: &mut Vec<T>, entry: T, limit: usize, key: impl Fn(&T) -> K) {
    entries.push(entry);
    entries.sort_unstable_by_key(&key);
    entries.truncate(limit);
}

fn route_elimination_positions(
    per: &PerList,
    route: &[i32],
    item: i32,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Vec<usize>, Abort> {
    let preferred = repair_positions(per, route, item, work, stop)?;
    if route.len() <= 32 {
        return Ok((0..=route.len()).collect());
    }
    Ok(preferred)
}

fn score_route_elimination_list(context: &mut RouteEliminationContext<'_>, list: usize, route: &[i32]) -> Result<ListScore, Abort> {
    context.work.evaluated(context.stop)?;
    context.per.metrics.record_candidate();
    Ok(trial_list_score_view(context.per, context.current, list, route, None, &mut context.list_scratch))
}

fn route_elimination_options(
    context: &mut RouteEliminationContext<'_>,
    overlay: &EjectionOverlay<'_>,
    pending: PendingEjection,
    forbidden: &HashSet<i32>,
) -> Result<Vec<RouteEliminationOption>, Abort> {
    const DIRECT_OPTIONS: usize = 8;
    const INFEASIBLE_INSERTIONS: usize = 12;
    const EJECTION_OPTIONS: usize = 8;
    const MAX_EJECTION_DEPTH: usize = 4;

    let mut direct = Vec::with_capacity(DIRECT_OPTIONS);
    let mut infeasible = Vec::with_capacity(INFEASIBLE_INSERTIONS);
    let destinations =
        rotated_indices(context.current.lists.len(), context.seed ^ mix64(pending.item as i64 as u64) ^ mix64(pending.depth as u64));
    for destination in destinations {
        if destination == context.eliminated || destination == pending.origin || overlay.route(destination).is_empty() {
            continue;
        }
        let positions = route_elimination_positions(context.per, overlay.route(destination), pending.item, context.work, context.stop)?;
        for position in positions {
            let base = overlay.route(destination);
            context.work.structural_many(base.len().saturating_add(1), context.stop)?;
            let mut route = clone_route_reserved(base, context.stop)?;
            route.insert(position.min(route.len()), pending.item);
            let score = score_route_elimination_list(context, destination, &route)?;
            let objectives = signed_list_objectives(context.per, &score);
            let headroom = context.per.routing_headroom(&route).saturating_neg();
            let tie = mix64(context.seed ^ mix64(pending.item as i64 as u64) ^ mix64(destination as u64) ^ mix64(position as u64));
            if score.violation == 0 {
                let option = RouteEliminationOption { destination, route, ejected: None, rank: (0, 0, headroom, objectives, tie) };
                retain_best(&mut direct, option, DIRECT_OPTIONS, |candidate| candidate.rank.clone());
            } else if pending.depth < MAX_EJECTION_DEPTH {
                let insertion = InfeasibleInsertion { destination, route, rank: (score.violation, headroom, objectives, tie) };
                retain_best(&mut infeasible, insertion, INFEASIBLE_INSERTIONS, |candidate| candidate.rank.clone());
            }
        }
    }

    if pending.depth >= MAX_EJECTION_DEPTH {
        return Ok(direct);
    }

    let mut ejections = Vec::with_capacity(EJECTION_OPTIONS);
    for insertion in infeasible {
        for position in
            rotated_indices(insertion.route.len(), context.seed ^ mix64(insertion.destination as u64) ^ mix64(pending.item as i64 as u64))
        {
            let ejected = insertion.route[position];
            if ejected == pending.item || forbidden.contains(&ejected) {
                continue;
            }
            context.work.structural_many(insertion.route.len(), context.stop)?;
            let mut route = clone_route_reserved(&insertion.route, context.stop)?;
            route.remove(position);
            let score = score_route_elimination_list(context, insertion.destination, &route)?;
            if score.violation != 0 {
                continue;
            }
            let objectives = signed_list_objectives(context.per, &score);
            let headroom = context.per.routing_headroom(&route).saturating_neg();
            let tie = mix64(context.seed ^ mix64(ejected as i64 as u64) ^ mix64(insertion.destination as u64) ^ mix64(position as u64));
            let option = RouteEliminationOption {
                destination: insertion.destination,
                route,
                ejected: Some(PendingEjection { item: ejected, origin: insertion.destination, depth: pending.depth + 1 }),
                rank: (1, 0, headroom, objectives, tie),
            };
            retain_best(&mut ejections, option, EJECTION_OPTIONS, |candidate| candidate.rank.clone());
        }
    }
    let mut options = Vec::with_capacity(direct.len().saturating_add(ejections.len()));
    let mut direct = direct.into_iter();
    let mut ejections = ejections.into_iter();
    loop {
        let before = options.len();
        options.extend(direct.by_ref().take(2));
        options.extend(ejections.by_ref().take(1));
        if options.len() == before {
            break;
        }
    }
    Ok(options)
}

fn search_route_elimination(
    context: &mut RouteEliminationContext<'_>,
    overlay: EjectionOverlay<'_>,
    pending: Vec<PendingEjection>,
    forbidden: HashSet<i32>,
) -> Result<Option<State>, Abort> {
    const BRANCHES: usize = 6;
    const MAX_EXTRA_EJECTIONS: usize = 6;

    if pending.is_empty() {
        let mut scoring = EjectionContext {
            model: context.model,
            per: context.per,
            current: context.current,
            before: context.before.clone(),
            work: context.work,
            stop: context.stop,
            seed: context.seed,
            list_scratch: std::mem::take(&mut context.list_scratch),
            score_scratch: std::mem::take(&mut context.score_scratch),
            overrides: std::mem::take(&mut context.overrides),
        };
        let score = score_ejection_overlay(&mut scoring, &overlay)?;
        let result = if score.violation == 0 && improving_raw(&scoring.before, &score) && overlay.final_len(context.eliminated) == 0 {
            Some(materialize_ejection_overlay(&mut scoring, overlay)?)
        } else {
            None
        };
        context.list_scratch = scoring.list_scratch;
        context.score_scratch = scoring.score_scratch;
        context.overrides = scoring.overrides;
        return Ok(result);
    }
    if forbidden.len() > context.initial_removed.saturating_add(MAX_EXTRA_EJECTIONS) {
        return Ok(None);
    }

    let mut selected: Option<(usize, Vec<RouteEliminationOption>)> = None;
    for index in rotated_indices(pending.len(), context.seed ^ mix64(forbidden.len() as u64)) {
        let options = route_elimination_options(context, &overlay, pending[index], &forbidden)?;
        if options.is_empty() {
            return Ok(None);
        }
        if selected.as_ref().is_none_or(|(_, best)| options.len() < best.len()) {
            selected = Some((index, options));
        }
    }
    let (selected_index, options) = selected.expect("pending route-elimination items are non-empty");

    for option in options.into_iter().take(BRANCHES) {
        context.work.structural_many(pending.len().saturating_add(forbidden.len()), context.stop)?;
        let mut next_pending = pending.clone();
        next_pending.swap_remove(selected_index);
        let mut next_forbidden = forbidden.clone();
        if let Some(ejected) = option.ejected {
            next_forbidden.insert(ejected.item);
            next_pending.push(ejected);
        }
        let mut next = overlay.fork(context.work, context.stop)?;
        next.replace_route(option.destination, option.route, context.work, context.stop)?;
        if let Some(candidate) = search_route_elimination(context, next, next_pending, next_forbidden)? {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn eliminate_route_with_ejections(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Option<State>, Abort> {
    if nonempty_routes(&current.lists) < 2 {
        return Ok(None);
    }
    let mut sources: Vec<usize> =
        current.lists.iter().enumerate().filter_map(|(list, route)| (!route.is_empty()).then_some(list)).collect();
    let minimum = sources.iter().map(|&list| current.lists[list].len()).min().unwrap_or(0);
    let maximum = minimum.saturating_add(integer_sqrt(minimum.max(1)).max(1));
    sources.retain(|&list| current.lists[list].len() <= maximum);
    let sort_work =
        u64::try_from(sources.len()).unwrap_or(u64::MAX).saturating_mul(ceil_log2(u64::try_from(sources.len()).unwrap_or(u64::MAX)));
    work.structural_units(sort_work, stop)?;
    sources.sort_by_key(|&list| (mix64(seed ^ mix64(list as u64)), current.lists[list].len()));

    for eliminated in sources {
        work.generated(stop)?;
        let removed = &current.lists[eliminated];
        let mut overlay = EjectionOverlay::new(&current.lists);
        work.structural_many(removed.len(), stop)?;
        overlay.route_mut(eliminated, work, stop)?.clear();
        work.structural_many(removed.len().saturating_mul(2), stop)?;
        let pending = removed.iter().copied().map(|item| PendingEjection { item, origin: eliminated, depth: 0 }).collect::<Vec<_>>();
        let forbidden = removed.iter().copied().collect::<HashSet<_>>();
        let mut context = RouteEliminationContext {
            model,
            per,
            current,
            before: full_score_raw(per, current),
            eliminated,
            initial_removed: removed.len(),
            seed: seed ^ mix64(eliminated as u64),
            work,
            stop,
            list_scratch: EvalScratch::default(),
            score_scratch: EvalScratch::default(),
            overrides: Vec::new(),
        };
        if let Some(candidate) = search_route_elimination(&mut context, overlay, pending, forbidden)? {
            debug_assert!(route_is_eliminated(per, current, &candidate));
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn ejection_chain(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Option<State>, Abort> {
    if current.lists.len() < 2 {
        return Ok(None);
    }
    let mut context = EjectionContext {
        model,
        per,
        current,
        before: full_score_raw(per, current),
        work,
        stop,
        seed,
        list_scratch: EvalScratch::default(),
        score_scratch: EvalScratch::default(),
        overrides: Vec::new(),
    };
    let mut sources: Vec<usize> = rotated_indices(current.lists.len(), seed).collect();
    if per.minimizes_fleet() {
        let sort_work =
            u64::try_from(sources.len()).unwrap_or(u64::MAX).saturating_mul(ceil_log2(u64::try_from(sources.len()).unwrap_or(u64::MAX)));
        context.work.structural_units(sort_work, context.stop)?;
        sources.sort_by_key(|&list| (current.lists[list].len(), mix64(seed ^ mix64(list as u64))));
    }
    for src in sources {
        for position in rotated_indices(current.lists[src].len(), seed ^ mix64(src as u64)) {
            context.work.generated(context.stop)?;
            let item = current.lists[src][position];
            let mut overlay = EjectionOverlay::new(&current.lists);
            let shifted = current.lists[src].len().saturating_sub(position);
            context.work.structural_many(shifted, context.stop)?;
            overlay.route_mut(src, context.work, context.stop)?.remove(position);
            let mut visited = SmallVec::<[i32; 4]>::new();
            visited.push(item);
            if let Some(candidate) = eject_insert(&mut context, overlay, item, src, 0, &mut visited)? {
                return Ok(Some(candidate));
            }
        }
    }
    Ok(None)
}

fn eject_insert(
    context: &mut EjectionContext<'_>,
    overlay: EjectionOverlay<'_>,
    item: i32,
    origin: usize,
    depth: usize,
    visited: &mut SmallVec<[i32; 4]>,
) -> Result<Option<State>, Abort> {
    const MAX_DEPTH: usize = 2;
    let route_count = context.current.lists.len();
    for dst in rotated_indices(route_count, context.seed ^ mix64(item as i64 as u64) ^ mix64(depth as u64)) {
        if dst == origin {
            continue;
        }
        for insert_at in repair_positions(context.per, overlay.route(dst), item, context.work, context.stop)? {
            let mut inserted = match overlay.fork(context.work, context.stop) {
                Ok(inserted) => inserted,
                Err(Abort::Infeasible) => continue,
                Err(abort) => return Err(abort),
            };
            let insert_at = insert_at.min(inserted.route(dst).len());
            context.work.structural_many(inserted.route(dst).len().saturating_sub(insert_at).saturating_add(1), context.stop)?;
            match inserted.route_mut(dst, context.work, context.stop) {
                Ok(route) => route.insert(insert_at, item),
                Err(Abort::Infeasible) => continue,
                Err(abort) => return Err(abort),
            }
            let candidate_score = score_ejection_overlay(context, &inserted)?;
            if candidate_score.violation == 0 && improving_raw(&context.before, &candidate_score) {
                let candidate = materialize_ejection_overlay(context, inserted)?;
                debug_assert!(full_score_raw(context.per, &candidate) == candidate_score);
                return Ok(Some(candidate));
            }
            if depth >= MAX_DEPTH {
                continue;
            }

            for eject_at in temporal_ejection_positions(
                context.per,
                inserted.route(dst),
                item,
                context.seed ^ mix64(dst as u64) ^ mix64(insert_at as u64),
                context.work,
                context.stop,
            )? {
                let ejected = inserted.route(dst)[eject_at];
                if ejected == item || visited.contains(&ejected) {
                    continue;
                }
                let mut next = inserted.fork(context.work, context.stop)?;
                context.work.structural_many(next.route(dst).len().saturating_sub(eject_at), context.stop)?;
                next.route_mut(dst, context.work, context.stop)?.remove(eject_at);
                visited.push(ejected);
                if let Some(result) = eject_insert(context, next, ejected, dst, depth + 1, visited)? {
                    return Ok(Some(result));
                }
                visited.pop();
            }
        }
    }
    Ok(None)
}

fn temporal_ejection_positions(
    per: &PerList,
    route: &[i32],
    inserted: i32,
    seed: u64,
    work: &mut WorkCounter,
    stop: &AtomicBool,
) -> Result<Vec<usize>, Abort> {
    const EJECTION_BRANCHES: usize = 4;
    let mut ranked = Vec::with_capacity(EJECTION_BRANCHES);
    for (position, &item) in route.iter().enumerate() {
        work.structural(stop)?;
        if item == inserted {
            continue;
        }
        let key = (per.shaw_relatedness(inserted, item, true), mix64(seed ^ mix64(item as i64 as u64)), position);
        let insertion = ranked.partition_point(|current| current <= &key);
        work.structural_many(ranked.len().saturating_add(1), stop)?;
        if insertion < EJECTION_BRANCHES {
            ranked.insert(insertion, key);
            ranked.truncate(EJECTION_BRANCHES);
        }
    }
    Ok(ranked.into_iter().map(|(_, _, position)| position).collect())
}

fn score_ejection_overlay(context: &mut EjectionContext<'_>, overlay: &EjectionOverlay<'_>) -> Result<Score, Abort> {
    let override_items = overlay.touched.iter().fold(0usize, |total, (_, route)| total.saturating_add(route.len()));
    let materialization = ejection_materialization_work(context, overlay)?;
    reserve_trial_scores(
        context.model,
        context.per,
        context.current,
        TrialScoreWork {
            touched: overlay.touched.iter().map(|(list, route)| (*list, route.len())).collect(),
            override_items,
            materialization_headroom: materialization,
        },
        context.work,
        context.stop,
    )?;
    context.per.metrics.record_candidate();

    let mut scores = SmallVec::<[ListScore; 4]>::new();
    for (list, route) in &overlay.touched {
        scores.push(trial_list_score_view(context.per, context.current, *list, route, None, &mut context.list_scratch));
    }
    context.overrides.clear();
    for (list, route) in &overlay.touched {
        context.overrides.extend(route.iter().copied().map(|item| (item, *list)));
    }
    let replacements = overlay
        .touched
        .iter()
        .zip(&scores)
        .map(|((list, route), score)| TrialList { list: *list, score, contents: route as &dyn super::incremental::ListView })
        .collect::<SmallVec<[_; 4]>>();
    let global_delta = context.per.globals.delta(&context.current.item_list, &context.overrides);
    Ok(score_with_replacements_raw(context.per, context.current, &replacements, global_delta, &mut context.score_scratch))
}

fn ejection_materialization_work(context: &EjectionContext<'_>, overlay: &EjectionOverlay<'_>) -> Result<u64, Abort> {
    let clone_work = state_clone_work(context.model, context.current);
    let mut poll = FloorStopPoll::new(context.stop).ok_or(Abort::Interrupted)?;
    let rebuild = state_rebuild_work_from_lengths_polled(
        context.model,
        context.per,
        (0..context.current.lists.len()).map(|list| overlay.final_len(list)),
        &mut poll,
    )
    .ok_or(Abort::Interrupted)?;
    poll.finish().ok_or(Abort::Interrupted)?;
    Ok(clone_work.saturating_add(rebuild))
}

fn materialize_ejection_overlay(context: &mut EjectionContext<'_>, overlay: EjectionOverlay<'_>) -> Result<State, Abort> {
    let reserved = ejection_materialization_work(context, &overlay)?;
    context.work.structural_units(reserved, context.stop)?;
    let mut lists = clone_lists_reserved(&context.current.lists, context.stop)?;
    for (list, route) in overlay.touched {
        lists[list] = route;
    }
    let state = state_from_lists(context.model, context.per, lists, context.stop)?;
    context.work.canonical_rebuilds = context.work.canonical_rebuilds.saturating_add(1);
    Ok(state)
}

fn shuffle_values(values: &mut [i32], seed: u64) {
    for index in (1..values.len()).rev() {
        let other = (mix64(seed.wrapping_add(index as u64)) % (index as u64 + 1)) as usize;
        values.swap(index, other);
    }
}

fn shuffle_values_bounded(values: &mut [i32], seed: u64, work: &mut WorkCounter, stop: &AtomicBool) -> Result<(), Abort> {
    for index in (1..values.len()).rev() {
        work.structural(stop)?;
        let other = (mix64(seed.wrapping_add(index as u64)) % (index as u64 + 1)) as usize;
        values.swap(index, other);
    }
    Ok(())
}

/// Deterministic integration oracle for the simulated-annealing branch.
#[doc(hidden)]
pub fn audit_annealing_acceptance(seed: u64) -> bool {
    let current = Score { violation: 0, tiers: tier_values(1, 1_000_000_000_000) };
    let candidate = Score { violation: 0, tiers: tier_values(1, 1_000_000_000_001) };
    let mut controller = AlnsController::new(100, &current);
    (0..64).any(|iteration| {
        matches!(controller.accept(&current, &candidate, &current, &candidate, seed, iteration), AcceptanceKind::Annealing)
    })
}

/// Deterministic integration oracle for segmented operator learning.
#[doc(hidden)]
pub fn audit_operator_learning() -> bool {
    let score = Score { violation: 0, tiers: tier_values(1, 0) };
    let mut controller = AlnsController::new(100, &score);
    for _ in 0..12 {
        controller.record(DestroyOperator::Shaw, RepairOperator::Greedy, AcceptanceKind::Improving, true, true);
        controller.record_failed(DestroyOperator::Segments, RepairOperator::Regret2);
        controller.record(DestroyOperator::Worst, RepairOperator::Regret3, AcceptanceKind::Late, false, false);
    }
    let metrics = controller.metrics();
    metrics.destroy[0].weight_milli > metrics.destroy[2].weight_milli
        && metrics.destroy[2].weight_milli > metrics.destroy[1].weight_milli
        && metrics.repair[0].weight_milli > metrics.repair[2].weight_milli
        && metrics.repair[2].weight_milli > metrics.repair[1].weight_milli
}

/// Every selected destroy and repair operator is counted once per ALNS slice,
/// including slices interrupted before either stage, or between the stages.
#[doc(hidden)]
pub fn audit_interrupted_attempt_accounting() -> bool {
    let score = Score { violation: 0, tiers: tier_values(1, 0) };
    let mut controller = AlnsController::new_routing_profile(16, &score, SearchProfile::Sequential);
    let choice = AlnsChoice { destroy: DestroyOperator::Worst, repair: RepairOperator::Regret3, target: 2 };
    let mut run = AlnsBuildRun {
        status: AlnsBuildStatus::Interrupted,
        candidate: None,
        generated: 0,
        evaluated: 0,
        work_units: 0,
        structural_work: 0,
        repair_index_work: 0,
        canonical_rebuilds: 0,
        cpu_nanos: 0,
        destroy_generated: 0,
        destroy_evaluated: 0,
        destroy_cpu_nanos: 0,
        repair_generated: 0,
        repair_evaluated: 0,
        repair_cpu_nanos: 0,
        destroy_executed: false,
        repair_executed: false,
        removed: 0,
        eliminated_route: false,
    };
    controller.record_failed_bounded(choice, &run);
    run.generated = 3;
    run.work_units = 3;
    run.destroy_generated = 3;
    run.destroy_cpu_nanos = 91;
    run.destroy_executed = true;
    controller.record_failed_bounded(choice, &run);

    let metrics = controller.metrics();
    metrics.iterations == 2
        && metrics.destroy.iter().map(|operator| operator.uses).sum::<u64>() == 2
        && metrics.repair.iter().map(|operator| operator.uses).sum::<u64>() == 2
        && metrics.destroy[DestroyOperator::Worst as usize].uses == 2
        && metrics.repair[RepairOperator::Regret3 as usize].uses == 2
        && metrics.destroy[DestroyOperator::Worst as usize].generated == 3
        && metrics.repair[RepairOperator::Regret3 as usize].generated == 0
}

/// A selected stage that never starts is telemetry, not an adaptive sample.
/// Its weight and pending learning state must remain unchanged.
#[doc(hidden)]
pub fn audit_skipped_operator_adaptation() -> bool {
    let score = Score { violation: 0, tiers: tier_values(1, 0) };
    let mut controller = AlnsController::new_routing_profile(16, &score, SearchProfile::Sequential);
    let choice = AlnsChoice { destroy: DestroyOperator::Worst, repair: RepairOperator::Regret3, target: 2 };
    let run = AlnsBuildRun {
        status: AlnsBuildStatus::Interrupted,
        candidate: None,
        generated: 0,
        evaluated: 0,
        work_units: 0,
        structural_work: 0,
        repair_index_work: 0,
        canonical_rebuilds: 0,
        cpu_nanos: 0,
        destroy_generated: 0,
        destroy_evaluated: 0,
        destroy_cpu_nanos: 0,
        repair_generated: 0,
        repair_evaluated: 0,
        repair_cpu_nanos: 0,
        destroy_executed: false,
        repair_executed: false,
        removed: 0,
        eliminated_route: false,
    };
    controller.record_failed_bounded(choice, &run);

    let destroy = controller.destroy[DestroyOperator::Worst as usize];
    let repair = controller.repair[RepairOperator::Regret3 as usize];
    let pending_is_neutral = destroy.weight == WEIGHT_SCALE
        && repair.weight == WEIGHT_SCALE
        && destroy.pending_uses == 0
        && repair.pending_uses == 0
        && destroy.pending_cost == 0
        && repair.pending_cost == 0;
    let metrics = controller.metrics();
    let destroy = &metrics.destroy[DestroyOperator::Worst as usize];
    let repair = &metrics.repair[RepairOperator::Regret3 as usize];
    pending_is_neutral
        && metrics.iterations == 1
        && destroy.uses == 1
        && repair.uses == 1
        && destroy.work_units == 0
        && repair.work_units == 0
        && destroy.weight_milli == WEIGHT_SCALE
        && repair.weight_milli == WEIGHT_SCALE
}

/// The bounded worst-removal selector has the same deterministic top-k result
/// as a full ranking. Its final partition rewrite commits atomically.
#[doc(hidden)]
pub fn audit_bounded_worst_removal() -> bool {
    let model = CollectionModel {
        items: (1..=32).collect(),
        lists: 4,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let per = PerList::build(&model);
    let original = vec![
        vec![1, 5, 9, 13, 17, 21, 25, 29],
        vec![2, 6, 10, 14, 18, 22, 26, 30],
        vec![3, 7, 11, 15, 19, 23, 27, 31],
        vec![4, 8, 12, 16, 20, 24, 28, 32],
    ];
    let state = State::from_lists(&model, &per, original.clone());
    let stop = AtomicBool::new(false);
    let seed = 0xA076_1D64_78BD_642F;
    let target = 9;

    let mut reference: Vec<(u64, usize, i32)> =
        original.iter().flatten().copied().enumerate().map(|(ordinal, item)| (mix64(seed ^ item as i64 as u64), ordinal, item)).collect();
    reference.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let selected: HashSet<i32> = reference.into_iter().take(target).map(|(_, _, item)| item).collect();
    let expected_removed: Vec<i32> = original.iter().flatten().copied().filter(|item| selected.contains(item)).collect();
    let expected_lists: Vec<Vec<i32>> =
        original.iter().map(|list| list.iter().copied().filter(|item| !selected.contains(item)).collect()).collect();

    let mut completed_lists = original.clone();
    let mut completed_work = WorkCounter::new(AlnsWorkBudget::new(10_000, 32));
    let completed = destroy_worst(&per, &state, &mut completed_lists, target, seed, &mut completed_work, &stop);
    let exact_generated = completed_work.generated;
    let exact_structural = completed_work.structural;

    let mut exhausted_lists = original.clone();
    let mut exhausted_work = WorkCounter::new(AlnsWorkBudget::new(exact_generated.saturating_sub(1), 32));
    let exhausted = destroy_worst(&per, &state, &mut exhausted_lists, target, seed, &mut exhausted_work, &stop);

    let interrupted_stop = AtomicBool::new(true);
    let mut interrupted_lists = original.clone();
    let mut interrupted_work = WorkCounter::new(AlnsWorkBudget::new(10_000, 32));
    let interrupted = destroy_worst(&per, &state, &mut interrupted_lists, target, seed, &mut interrupted_work, &interrupted_stop);

    completed.is_ok_and(|removed| removed == expected_removed)
        && completed_lists == expected_lists
        && completed_work.evaluated == 32
        && exact_structural > 0
        && exact_structural < exact_generated
        && matches!(exhausted, Err(Abort::BudgetExhausted))
        && exhausted_lists == original
        && exhausted_work.generated == exact_generated - 1
        && exhausted_work.structural == exact_structural - 1
        && matches!(interrupted, Err(Abort::Interrupted))
        && interrupted_lists == original
        && interrupted_work.generated == 0
        && interrupted_work.structural == 0
}

/// Causal oracle for bounded destroy/repair outcomes and exact route
/// elimination. Kept here so it can exercise private state caches directly.
#[doc(hidden)]
pub fn audit_bounded_alns() -> bool {
    let model = CollectionModel {
        items: (1..=6).collect(),
        lists: 3,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let per = PerList::build(&model);
    let state = State::from_lists(&model, &per, vec![vec![1], vec![2, 3], vec![4, 5, 6]]);
    let choice = AlnsChoice { destroy: DestroyOperator::Shaw, repair: RepairOperator::Greedy, target: 2 };
    let exhausted = build_candidate_bounded(&model, &per, &state, choice, 7, AlnsWorkBudget::new(0, 0), &stop);
    let built = build_candidate_bounded(&model, &per, &state, choice, 7, AlnsWorkBudget::new(256, 256), &stop);
    let eliminated =
        build_macro_candidate_bounded(&model, &per, &state, MacroOperator::RouteElimination, 11, AlnsWorkBudget::new(256, 256), &stop);
    let interrupted_stop = AtomicBool::new(true);
    let interrupted = build_candidate_bounded(&model, &per, &state, choice, 7, AlnsWorkBudget::new(256, 256), &interrupted_stop);
    exhausted.status == AlnsBuildStatus::BudgetExhausted
        && exhausted.generated == 0
        && built.status == AlnsBuildStatus::Built
        && built.canonical_rebuilds == 2
        && built.work_units == built.generated.saturating_add(built.evaluated)
        && built.structural_work <= built.generated
        && built.repair_index_work <= built.structural_work
        && eliminated.status == AlnsBuildStatus::Built
        && eliminated.canonical_rebuilds == 2
        && eliminated.eliminated_route
        && eliminated.candidate.as_ref().is_some_and(|candidate| nonempty_routes(&candidate.lists) == 2)
        && interrupted.status == AlnsBuildStatus::Interrupted
}

/// Fixed generation budgets cap copying, segment destruction, placement
/// enumeration, and repair-index maintenance independently of instance size.
/// The successful n=1,000 run also proves that the index is built once and then
/// updated by shifted route suffixes instead of being rebuilt once per item.
#[doc(hidden)]
pub fn audit_bounded_structural_work() -> bool {
    fn exhausted_run(items: usize) -> AlnsBuildRun {
        let model = CollectionModel {
            items: (1..=i32::try_from(items).expect("test size fits i32")).collect(),
            lists: 4,
            objectives: Vec::new(),
            constraints: Vec::new(),
            globals: Vec::new(),
            schedule: None,
        };
        let per = PerList::build(&model);
        let mut lists = vec![Vec::new(); 4];
        for (offset, item) in model.items.iter().copied().enumerate() {
            lists[offset % 4].push(item);
        }
        let state = State::from_lists(&model, &per, lists);
        let choice = AlnsChoice { destroy: DestroyOperator::Segments, repair: RepairOperator::Regret2, target: items / 2 };
        build_candidate_bounded(&model, &per, &state, choice, 31, AlnsWorkBudget::new(128, 128), &AtomicBool::new(false))
    }

    let small = exhausted_run(64);
    let large = exhausted_run(4_096);
    let interrupted_model = CollectionModel {
        items: (1..=4_096).collect(),
        lists: 4,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let interrupted_per = PerList::build(&interrupted_model);
    let interrupted_state = State::from_lists(&interrupted_model, &interrupted_per, vec![interrupted_model.items.clone()]);
    let interrupted = build_candidate_bounded(
        &interrupted_model,
        &interrupted_per,
        &interrupted_state,
        AlnsChoice { destroy: DestroyOperator::Segments, repair: RepairOperator::Greedy, target: 2_048 },
        37,
        AlnsWorkBudget::new(128, 128),
        &AtomicBool::new(true),
    );

    let (model, per, state) = routing_scale_fixture();
    let indexed = build_candidate_bounded(
        &model,
        &per,
        &state,
        AlnsChoice { destroy: DestroyOperator::Route, repair: RepairOperator::Regret2, target: 1 },
        29,
        AlnsWorkBudget::new(24_000, 12_000),
        &AtomicBool::new(false),
    );

    [small, large].into_iter().all(|run| {
        run.status == AlnsBuildStatus::BudgetExhausted
            && run.generated <= 128
            && run.structural_work <= run.generated
            && run.repair_index_work <= run.structural_work
    }) && interrupted.status == AlnsBuildStatus::Interrupted
        && interrupted.generated == 0
        && interrupted.structural_work == 0
        && indexed.status == AlnsBuildStatus::Built
        && indexed.repair_index_work > 0
        && indexed.repair_index_work < 3_000
        && indexed.structural_work <= indexed.generated
}

fn has_exact_partition(model: &CollectionModel, state: &State) -> bool {
    let mut expected = model.items.clone();
    let mut actual: Vec<i32> = state.lists.iter().flatten().copied().collect();
    expected.sort_unstable();
    actual.sort_unstable();
    actual == expected
}

/// Causal compatibility oracle for the generic, unbounded ALNS path.
#[doc(hidden)]
pub fn audit_unbounded_alns_compatibility() -> bool {
    let model = CollectionModel {
        items: (1..=12).collect(),
        lists: 2,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let stop = AtomicBool::new(false);
    let per = PerList::build(&model);
    let state = State::from_lists(&model, &per, vec![(1..=6).collect(), (7..=12).collect()]);
    let regret_choice = AlnsChoice { destroy: DestroyOperator::Shaw, repair: RepairOperator::Regret2, target: 10 };
    let regret = build_candidate_bounded(&model, &per, &state, regret_choice, 17, AlnsWorkBudget::unbounded(), &stop);
    let route_choice = AlnsChoice { destroy: DestroyOperator::Route, repair: RepairOperator::Greedy, target: 10 };
    let route = build_candidate_bounded(&model, &per, &state, route_choice, 23, AlnsWorkBudget::unbounded(), &stop);
    let polished = build_candidate(&model, &per, &state, regret_choice, 17, &stop);

    regret.status == AlnsBuildStatus::Built
        && regret.removed == 10
        && regret.repair_generated == 385
        && regret.repair_evaluated == 385
        && regret.candidate.as_ref().is_some_and(|candidate| has_exact_partition(&model, candidate))
        && route.status == AlnsBuildStatus::Built
        && route.removed == 10
        && route.candidate.as_ref().is_some_and(|candidate| has_exact_partition(&model, candidate))
        && polished.as_ref().is_some_and(|candidate| has_exact_partition(&model, candidate))
}

/// A 24-customer route must be repairable at n=1,000 under the production
/// macro slice. The dynamic placement cap is exercised by using no candidate
/// graph, which exposes more than 96 insertion positions per sampled item.
#[doc(hidden)]
fn routing_scale_fixture() -> (CollectionModel, PerList, State) {
    let model = CollectionModel {
        items: (1..=1_000).collect(),
        lists: 40,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let mut next = 1i32;
    let mut lists = Vec::with_capacity(40);
    for length in std::iter::once(24).chain(std::iter::repeat_n(25, 38)).chain(std::iter::once(26)) {
        lists.push((next..next + length).collect());
        next += length;
    }
    let per = PerList::build(&model);
    let state = State::from_lists(&model, &per, lists);
    (model, per, state)
}

pub fn audit_route_elimination_budget_1000() -> bool {
    let (model, per, state) = routing_scale_fixture();
    let stop = AtomicBool::new(false);
    [RepairOperator::Regret2, RepairOperator::Regret3, RepairOperator::BlinkRegret2].into_iter().all(|repair| {
        let choice = AlnsChoice { destroy: DestroyOperator::Route, repair, target: 1 };
        let run = build_candidate_bounded(&model, &per, &state, choice, 29, AlnsWorkBudget::new(24_000, 12_000), &stop);
        run.status == AlnsBuildStatus::Built
            && run.removed == 24
            && run.canonical_rebuilds == 2
            && run.generated <= 24_000
            && run.evaluated <= 12_000
            && run.structural_work <= run.generated
            && run.repair_index_work <= run.structural_work
            && run
                .candidate
                .as_ref()
                .is_some_and(|candidate| nonempty_routes(&candidate.lists) == 39 && has_exact_partition(&model, candidate))
    })
}

/// Adaptive choices and weights must be identical under different timing
/// observations when rewards and deterministic work are identical.
#[doc(hidden)]
pub fn audit_timing_independent_operator_learning() -> bool {
    let score = Score { violation: 0, tiers: tier_values(1, 0) };
    let mut first_timeline = AlnsController::new_routing_profile(16, &score, SearchProfile::Sequential);
    let mut second_timeline = AlnsController::new_routing_profile(16, &score, SearchProfile::Sequential);
    for _ in 0..8 {
        first_timeline.record_with_work(
            DestroyOperator::Shaw,
            RepairOperator::Greedy,
            AcceptanceKind::Improving,
            true,
            false,
            3_000,
            3_000,
            100_000,
        );
        first_timeline.record_with_work(
            DestroyOperator::Segments,
            RepairOperator::Regret2,
            AcceptanceKind::Improving,
            true,
            false,
            10_000,
            10_000,
            20_000_000,
        );
        second_timeline.record_with_work(
            DestroyOperator::Shaw,
            RepairOperator::Greedy,
            AcceptanceKind::Improving,
            true,
            false,
            3_000,
            3_000,
            50_000_000,
        );
        second_timeline.record_with_work(
            DestroyOperator::Segments,
            RepairOperator::Regret2,
            AcceptanceKind::Improving,
            true,
            false,
            10_000,
            10_000,
            1,
        );
    }
    let same_weights = first_timeline.destroy.iter().zip(second_timeline.destroy).all(|(left, right)| left.weight == right.weight)
        && first_timeline.repair.iter().zip(second_timeline.repair).all(|(left, right)| left.weight == right.weight);
    let same_choices = (0..4_096).all(|iteration| {
        let (left_destroy, left_repair) = first_timeline.select(41, iteration);
        let (right_destroy, right_repair) = second_timeline.select(41, iteration);
        left_destroy as usize == right_destroy as usize && left_repair as usize == right_repair as usize
    });
    let first = first_timeline.metrics();
    let second = second_timeline.metrics();
    same_weights
        && same_choices
        && first.destroy[DestroyOperator::Shaw as usize].weight_milli > first.destroy[DestroyOperator::Segments as usize].weight_milli
        && first.repair[RepairOperator::Greedy as usize].weight_milli > first.repair[RepairOperator::Regret2 as usize].weight_milli
        && first.destroy[DestroyOperator::Shaw as usize].weight_milli == second.destroy[DestroyOperator::Shaw as usize].weight_milli
        && first.destroy[DestroyOperator::Shaw as usize].cpu_nanos != second.destroy[DestroyOperator::Shaw as usize].cpu_nanos
}

/// n=101 and n=401 work profiles must retain useful fixed-point resolution and
/// deterministic exploration after a long run of accepted, non-improving
/// leader observations.
#[doc(hidden)]
pub fn audit_routing_scale_operator_exploration() -> bool {
    fn record_fixture(
        controller: &mut AlnsController,
        customers: u64,
        destroy: DestroyOperator,
        repair: RepairOperator,
        acceptance: AcceptanceKind,
        cpu_nanos: u64,
    ) {
        let (destroy_generated_factor, destroy_evaluated_factor) = match destroy {
            DestroyOperator::Shaw => (6, 2),
            DestroyOperator::Segments => (4, 1),
            DestroyOperator::Worst => (8, 6),
            DestroyOperator::Route => (3, 1),
        };
        let (repair_generated_factor, repair_evaluated_factor) = match repair {
            RepairOperator::Greedy => (8, 6),
            RepairOperator::Regret2 => (10, 10),
            RepairOperator::Regret3 => (12, 12),
            RepairOperator::BlinkRegret2 => (9, 8),
        };
        controller.record_with_split(
            destroy,
            repair,
            acceptance,
            false,
            false,
            customers.saturating_mul(destroy_generated_factor),
            customers.saturating_mul(destroy_evaluated_factor),
            cpu_nanos,
            customers.saturating_mul(repair_generated_factor),
            customers.saturating_mul(repair_evaluated_factor),
            cpu_nanos,
            true,
            true,
        );
    }

    [101u64, 401].into_iter().all(|customers| {
        let score = Score { violation: 0, tiers: tier_values(1, 0) };
        let mut controller = AlnsController::new_routing_profile(customers as usize, &score, SearchProfile::Sequential);
        for sample in 0..48 {
            record_fixture(&mut controller, customers, DestroyOperator::Shaw, RepairOperator::Regret3, AcceptanceKind::Late, sample + 1);
        }

        let mut destroy_draws = [0u64; DESTROY_OPERATORS];
        let mut repair_draws = [0u64; REPAIR_OPERATORS];
        for draw in 0..256 {
            let (destroy, repair) = controller.select(0xA076_1D64_78BD_642F, draw);
            destroy_draws[destroy as usize] = destroy_draws[destroy as usize].saturating_add(1);
            repair_draws[repair as usize] = repair_draws[repair as usize].saturating_add(1);
            let acceptance = if matches!(destroy, DestroyOperator::Shaw) && matches!(repair, RepairOperator::Regret3) {
                AcceptanceKind::Late
            } else {
                AcceptanceKind::Rejected
            };
            record_fixture(&mut controller, customers, destroy, repair, acceptance, draw.saturating_mul(1_000_003));
        }

        let metrics = controller.metrics();
        let destroy_weights = metrics.destroy.iter().map(|operator| operator.weight_milli).collect::<HashSet<_>>();
        let repair_weights = metrics.repair.iter().map(|operator| operator.weight_milli).collect::<HashSet<_>>();
        destroy_draws.into_iter().all(|uses| uses > 1)
            && repair_draws.into_iter().all(|uses| uses > 1)
            && destroy_weights.len() == DESTROY_OPERATORS
            && repair_weights.len() == REPAIR_OPERATORS
            && metrics.destroy.iter().all(|operator| operator.weight_milli > COST_MIN_WEIGHT)
            && metrics.repair.iter().all(|operator| operator.weight_milli > COST_MIN_WEIGHT)
            && metrics.destroy.iter().map(|operator| operator.uses).sum::<u64>() == 304
            && metrics.repair.iter().map(|operator| operator.uses).sum::<u64>() == 304
    })
}

/// A zero-reward cost-rate segment must favor cheap deterministic work. The
/// costly operator's projected share is capped below 70%, while generic
/// per-use learning intentionally ignores the routing cost model.
#[doc(hidden)]
pub fn audit_unproductive_deterministic_cost_balance() -> bool {
    let score = Score { violation: 0, tiers: tier_values(1, 0) };
    let mut causal = AlnsController::new_routing_profile(16, &score, SearchProfile::Sequential);
    causal.record_with_work(
        DestroyOperator::Segments,
        RepairOperator::Greedy,
        AcceptanceKind::Rejected,
        false,
        false,
        5_000,
        5_000,
        20_000_000,
    );
    causal.record_with_work(DestroyOperator::Worst, RepairOperator::Regret2, AcceptanceKind::Rejected, false, false, 5_000, 5_000, 1);
    let immediate_cheap = causal.destroy[DestroyOperator::Segments as usize].weight;
    let immediate_costly = causal.destroy[DestroyOperator::Worst as usize].weight;
    let cheap_cost = DestroyOperator::Segments.cost_model().estimate(10_000, 5_000, 5_000);
    let costly_cost = DestroyOperator::Worst.cost_model().estimate(10_000, 5_000, 5_000);
    let immediate_cheap_share = u128::from(immediate_cheap).saturating_mul(u128::from(cheap_cost));
    let immediate_costly_share = u128::from(immediate_costly).saturating_mul(u128::from(costly_cost));
    let immediate_total = immediate_cheap_share.saturating_add(immediate_costly_share);

    let mut routing = AlnsController::new_routing_profile(16, &score, SearchProfile::Sequential);
    let mut generic = AlnsController::new_profile(16, &score, SearchProfile::Sequential);
    for _ in 0..64 {
        for controller in [&mut routing, &mut generic] {
            controller.record_with_work(
                DestroyOperator::Segments,
                RepairOperator::Greedy,
                AcceptanceKind::Rejected,
                false,
                false,
                5_000,
                5_000,
                20_000_000,
            );
            controller.record_with_work(
                DestroyOperator::Worst,
                RepairOperator::Regret2,
                AcceptanceKind::Rejected,
                false,
                false,
                5_000,
                5_000,
                1,
            );
        }
    }
    let routing = routing.metrics();
    let generic = generic.metrics();
    let cheap = &routing.destroy[DestroyOperator::Segments as usize];
    let costly = &routing.destroy[DestroyOperator::Worst as usize];
    let cheap_share = u128::from(cheap.weight_milli).saturating_mul(u128::from(cheap_cost));
    let costly_share = u128::from(costly.weight_milli).saturating_mul(u128::from(costly_cost));
    let total = cheap_share.saturating_add(costly_share);

    immediate_cheap > immediate_costly
        && immediate_costly_share.saturating_mul(10) <= immediate_total.saturating_mul(7)
        && cheap.positive_rewards == 0
        && costly.positive_rewards == 0
        && cheap.weight_milli > costly.weight_milli
        && costly_share.saturating_mul(10) <= total.saturating_mul(7)
        && generic.destroy[DestroyOperator::Segments as usize].weight_milli == generic.destroy[DestroyOperator::Worst as usize].weight_milli
}

/// Causal accounting oracle for speculative macro scans. Large scans exhaust a
/// deterministic budget without rebuilding, flat objectives complete without
/// rebuilding, and a quadratic route-elimination cache is charged before the
/// canonical pass begins.
#[doc(hidden)]
pub fn audit_incremental_macro_accounting() -> bool {
    use crate::model::list::{ExprArena, ObjectiveTier};

    fn flat_fixture(items: usize, lists: usize) -> (CollectionModel, PerList, State) {
        let model = CollectionModel {
            items: (1..=i32::try_from(items).expect("audit size fits i32")).collect(),
            lists,
            objectives: Vec::new(),
            constraints: Vec::new(),
            globals: Vec::new(),
            schedule: None,
        };
        let per = PerList::build(&model);
        let mut contents = vec![Vec::new(); lists];
        for (offset, item) in model.items.iter().copied().enumerate() {
            contents[offset % lists].push(item);
        }
        let state = State::from_lists(&model, &per, contents);
        (model, per, state)
    }

    let stop = AtomicBool::new(false);
    let (large_model, large_per, large_state) = flat_fixture(1_000, 40);
    let large = [MacroOperator::ChainRelocate, MacroOperator::GuidedSegmentExchange, MacroOperator::EjectionChain].map(|operator| {
        build_macro_candidate_bounded(&large_model, &large_per, &large_state, operator, 17, AlnsWorkBudget::new(128, 128), &stop)
    });
    let production = [MacroOperator::ChainRelocate, MacroOperator::GuidedSegmentExchange, MacroOperator::EjectionChain].map(|operator| {
        build_macro_candidate_bounded(&large_model, &large_per, &large_state, operator, 19, AlnsWorkBudget::new(24_576, 12_288), &stop)
    });

    let (flat_model, flat_per, flat_state) = flat_fixture(6, 3);
    let flat = [MacroOperator::ChainRelocate, MacroOperator::GuidedSegmentExchange, MacroOperator::EjectionChain].map(|operator| {
        build_macro_candidate_bounded(&flat_model, &flat_per, &flat_state, operator, 23, AlnsWorkBudget::new(100_000, 100_000), &stop)
    });

    let mut arena = ExprArena::default();
    let body = arena.constant(1);
    let pair_model = CollectionModel {
        items: (1..=200).collect(),
        lists: 2,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..2)
                .map(|list| Reduction { op: ReduceOp::Sum, iterable: Iterable::Pairs(list), arena: arena.clone(), body, coeff: 1 })
                .collect(),
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let pair_per = PerList::build(&pair_model);
    let pair_state = State::from_lists(&pair_model, &pair_per, vec![(1..=100).collect(), (101..=200).collect()]);
    let pair_run = build_macro_candidate_bounded(
        &pair_model,
        &pair_per,
        &pair_state,
        MacroOperator::RouteElimination,
        29,
        AlnsWorkBudget::new(2_048, 2_048),
        &stop,
    );

    large.iter().all(|run| {
        run.status == AlnsBuildStatus::BudgetExhausted
            && run.canonical_rebuilds == 0
            && run.structural_work <= run.generated
            && run.generated <= 128
            && run.evaluated <= 128
    }) && production.iter().all(|run| {
        run.status == AlnsBuildStatus::BudgetExhausted
            && run.candidate.is_none()
            && run.canonical_rebuilds == 0
            && run.evaluated > 0
            && run.generated <= 24_576
            && run.evaluated <= 12_288
            && run.structural_work <= run.generated
    }) && flat.iter().all(|run| run.candidate.is_none() && run.canonical_rebuilds == 0 && run.structural_work <= run.generated)
        && pair_run.status == AlnsBuildStatus::BudgetExhausted
        && pair_run.canonical_rebuilds == 0
        && pair_run.structural_work <= pair_run.generated
}

/// Exercise every compound routing neighborhood on a tiny exact oracle. The
/// count-constrained cases force ejection and exchange instead of accepting a
/// one-way relocate.
#[doc(hidden)]
pub fn audit_macro_operators() -> bool {
    use std::sync::Arc;

    use crate::model::list::{Constraint, ExprArena, Iterable, ObjectiveTier, Op, ReduceOp, Reduction};

    fn count(list: usize) -> Reduction {
        let mut arena = ExprArena::default();
        let body = arena.constant(1);
        Reduction { op: ReduceOp::Count, iterable: Iterable::Items(list), arena, body, coeff: 1 }
    }
    fn item_cost(list: usize, costs: Arc<Vec<i64>>) -> Reduction {
        let mut arena = ExprArena::default();
        let item = arena.arg(0);
        let body = arena.array(costs, item);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(list), arena, body, coeff: 1 }
    }
    fn edge_guide(list: usize, matrix: Arc<Vec<Vec<i64>>>) -> Reduction {
        let mut arena = ExprArena::default();
        let from = arena.arg(0);
        let to = arena.arg(1);
        let body = arena.matrix(matrix, from, to);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 0 }
    }
    fn model(constrained: bool) -> CollectionModel {
        let matrix = Arc::new((0..5).map(|from| (0..5).map(|to| i64::from(from != to)).collect::<Vec<_>>()).collect::<Vec<_>>());
        let mut left = vec![0; 5];
        left[1] = 10;
        left[2] = 10;
        let mut right = vec![0; 5];
        right[3] = 10;
        right[4] = 10;
        CollectionModel {
            items: vec![1, 2, 3, 4],
            lists: 2,
            objectives: vec![ObjectiveTier {
                minimize: true,
                terms: vec![
                    edge_guide(0, Arc::clone(&matrix)),
                    edge_guide(1, matrix),
                    item_cost(0, Arc::new(left)),
                    item_cost(1, Arc::new(right)),
                ],
                max_terms: None,
            }],
            constraints: if constrained {
                (0..2).map(|list| Constraint { reduction: count(list), op: Op::Eq, rhs: 2 }).collect()
            } else {
                Vec::new()
            },
            globals: Vec::new(),
            schedule: None,
        }
    }

    let stop = AtomicBool::new(false);
    let initial = vec![vec![1, 2], vec![3, 4]];
    let relaxed = model(false);
    let relaxed_per = PerList::build(&relaxed);
    let relaxed_state = State::from_lists(&relaxed, &relaxed_per, initial.clone());
    let chain = build_macro_candidate_bounded(
        &relaxed,
        &relaxed_per,
        &relaxed_state,
        MacroOperator::ChainRelocate,
        3,
        AlnsWorkBudget::new(10_000, 10_000),
        &stop,
    );

    let constrained = model(true);
    let constrained_per = PerList::build(&constrained);
    let constrained_state = State::from_lists(&constrained, &constrained_per, initial);
    let ejection = build_macro_candidate_bounded(
        &constrained,
        &constrained_per,
        &constrained_state,
        MacroOperator::EjectionChain,
        5,
        AlnsWorkBudget::new(10_000, 10_000),
        &stop,
    );
    let exchange = build_macro_candidate_bounded(
        &constrained,
        &constrained_per,
        &constrained_state,
        MacroOperator::GuidedSegmentExchange,
        7,
        AlnsWorkBudget::new(10_000, 10_000),
        &stop,
    );
    let interrupted_stop = AtomicBool::new(true);
    let interrupted = build_macro_candidate_bounded(
        &constrained,
        &constrained_per,
        &constrained_state,
        MacroOperator::EjectionChain,
        9,
        AlnsWorkBudget::new(10_000, 10_000),
        &interrupted_stop,
    );
    let valid_run = |run: &AlnsBuildRun, model: &CollectionModel| {
        run.status == AlnsBuildStatus::Built
            && run.candidate.as_ref().is_some_and(|candidate| has_exact_partition(model, candidate))
            && run.canonical_rebuilds == 1
            && run.generated > 0
            && run.evaluated > 0
            && run.work_units == run.generated + run.evaluated
            && run.structural_work <= run.generated
    };
    valid_run(&chain, &relaxed)
        && valid_run(&ejection, &constrained)
        && valid_run(&exchange, &constrained)
        && interrupted.status == AlnsBuildStatus::Interrupted
        && interrupted.generated == 0
        && interrupted.evaluated == 0
}

/// A canonical rebuild is counted only after every interruptible cache pass has
/// completed and produced a usable state.
#[doc(hidden)]
pub fn audit_completed_canonical_rebuild_accounting() -> bool {
    use std::sync::Arc;

    use crate::model::list::{register_external_function, ExprArena, ObjectiveTier};

    const INTERRUPTER: &str = "qayd_audit_interrupt_canonical_rebuild";

    let stop = Arc::new(AtomicBool::new(false));
    let callback_stop = Arc::clone(&stop);
    if register_external_function(INTERRUPTER, move |_| {
        callback_stop.store(true, Ordering::Release);
        Ok(0)
    })
    .is_err()
    {
        return false;
    }

    let mut arena = ExprArena::default();
    let item = arena.arg(0);
    let body = arena.external(INTERRUPTER, vec![item]);
    let interrupted_model = CollectionModel {
        items: vec![1],
        lists: 1,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: vec![Reduction { op: ReduceOp::Sum, iterable: Iterable::Items(0), arena, body, coeff: 1 }],
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let interrupted_per = PerList::build(&interrupted_model);
    let mut interrupted_work = WorkCounter::new(AlnsWorkBudget::unbounded());
    let interrupted = canonical_state_from_lists(&interrupted_model, &interrupted_per, vec![vec![1]], &mut interrupted_work, stop.as_ref());

    let complete_model =
        CollectionModel { items: vec![1], lists: 1, objectives: Vec::new(), constraints: Vec::new(), globals: Vec::new(), schedule: None };
    let complete_per = PerList::build(&complete_model);
    let complete_stop = AtomicBool::new(false);
    let mut complete_work = WorkCounter::new(AlnsWorkBudget::unbounded());
    let complete = canonical_state_from_lists(&complete_model, &complete_per, vec![vec![1]], &mut complete_work, &complete_stop);

    matches!(interrupted, Err(Abort::Interrupted))
        && interrupted_work.canonical_rebuilds == 0
        && complete.is_ok()
        && complete_work.canonical_rebuilds == 1
}

/// A large non-neighbour route pair used to consume the entire 65,536-unit
/// exploration cap before the first guided pair. Direct edge enumeration jumps
/// to the causal route pair and scores one candidate within that same cap.
#[doc(hidden)]
pub fn audit_direct_guided_exchange_enumeration() -> bool {
    use std::sync::Arc;

    use crate::model::list::{ExprArena, ObjectiveTier};

    const ROUTES: usize = 2;
    const ITEMS: usize = 1_000;
    const SPLIT: usize = ITEMS / 2;
    const LEGACY_EXPLORATION_CAP: u64 = 65_536;

    fn edge_guide(list: usize, matrix: Arc<Vec<Vec<i64>>>) -> Reduction {
        let mut arena = ExprArena::default();
        let from = arena.arg(0);
        let to = arena.arg(1);
        let body = arena.matrix(matrix, from, to);
        Reduction { op: ReduceOp::Sum, iterable: Iterable::Edges { list, start: 0, end: 0 }, arena, body, coeff: 0 }
    }

    let mut matrix: Vec<Vec<i64>> = (0..=ITEMS)
        .map(|from| {
            (0..=ITEMS)
                .map(|to| {
                    if from == to {
                        0
                    } else if (from <= SPLIT) == (to <= SPLIT) {
                        1
                    } else {
                        1_000
                    }
                })
                .collect()
        })
        .collect();
    // The only candidate edge crossing the two routes is directed from 584 to
    // 499. In the legacy Cartesian order it first appears after more than
    // 700,000 rejected segment pairs.
    matrix[584][499] = -1;
    let matrix = Arc::new(matrix);
    let model = CollectionModel {
        items: (1..=ITEMS as i32).collect(),
        lists: ROUTES,
        objectives: vec![ObjectiveTier {
            minimize: true,
            terms: (0..ROUTES).map(|list| edge_guide(list, Arc::clone(&matrix))).collect(),
            max_terms: None,
        }],
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let per = PerList::build(&model);
    let graph_is_causal = per.candidates.as_ref().is_some_and(|candidates| {
        candidates.contains(499, 584) && candidates.neighbors(584).contains(&499) && !candidates.neighbors(499).contains(&584)
    });
    let lists = vec![(1..=SPLIT as i32).collect(), ((SPLIT + 1) as i32..=ITEMS as i32).collect()];
    let state = State::from_lists(&model, &per, lists);
    let stop = AtomicBool::new(false);
    let capped = build_macro_candidate_bounded(
        &model,
        &per,
        &state,
        MacroOperator::GuidedSegmentExchange,
        17,
        AlnsWorkBudget::new(LEGACY_EXPLORATION_CAP, 1),
        &stop,
    );
    let Some(floor) = routing_compound_structural_floor(&model, &per, &state, &stop) else {
        return false;
    };
    let floored =
        build_macro_candidate_bounded(&model, &per, &state, MacroOperator::GuidedSegmentExchange, 17, AlnsWorkBudget::new(floor, 1), &stop);

    graph_is_causal
        && capped.status == AlnsBuildStatus::BudgetExhausted
        && capped.evaluated == 1
        && capped.canonical_rebuilds == 0
        && capped.generated <= LEGACY_EXPLORATION_CAP
        && capped.structural_work <= capped.generated
        && guided_exchange_discovery_floor(&model, &per, &state, &stop).is_some_and(|guided| floor >= guided)
        && floor <= LEGACY_EXPLORATION_CAP
        && floored.evaluated == 1
        && floored.canonical_rebuilds == 0
        && floored.structural_work <= floored.generated
}

/// The guided location table is admitted as one atomic structural pass before
/// allocation, while every routing budget estimator observes a pre-armed stop.
/// The large partition guards against accidentally returning to an O(n) setup
/// scan before the first budget or interruption check.
#[doc(hidden)]
pub fn audit_interruptible_routing_budget_admission() -> bool {
    const ITEMS: usize = 50_000;
    const LISTS: usize = 64;

    let model = CollectionModel {
        items: (1..=i32::try_from(ITEMS).expect("audit size fits i32")).collect(),
        lists: LISTS,
        objectives: Vec::new(),
        constraints: Vec::new(),
        globals: Vec::new(),
        schedule: None,
    };
    let per = PerList::build(&model);
    let mut lists = vec![Vec::new(); LISTS];
    for (offset, item) in model.items.iter().copied().enumerate() {
        lists[offset % LISTS].push(item);
    }
    let state = State::from_lists(&model, &per, lists);
    let index_work = u64::try_from(ITEMS + LISTS).expect("audit work fits u64");
    let running = AtomicBool::new(false);

    let mut short_work = WorkCounter::new(AlnsWorkBudget::new(index_work - 1, 1));
    let short = build_guided_location_index(&model, &state, &mut short_work, &running);
    let mut exact_work = WorkCounter::new(AlnsWorkBudget::new(index_work, 1));
    let exact = build_guided_location_index(&model, &state, &mut exact_work, &running);

    let stopped = AtomicBool::new(true);
    let mut stopped_work = WorkCounter::new(AlnsWorkBudget::new(index_work, 1));
    let stopped_index = build_guided_location_index(&model, &state, &mut stopped_work, &stopped);
    let stopped_floors = routing_compound_structural_floor(&model, &per, &state, &stopped).is_none()
        && routing_route_elimination_floor(&model, &per, &state, &stopped).is_none()
        && routing_relink_structural_floor(&model, &per, &state, &stopped).is_none()
        && routing_relink_candidate_work(&model, &per, &state.lists, 0, 0, 1, state.lists[1].len(), &stopped).is_none();
    let running_floors = routing_compound_structural_floor(&model, &per, &state, &running).is_some()
        && routing_route_elimination_floor(&model, &per, &state, &running).is_some()
        && routing_relink_structural_floor(&model, &per, &state, &running).is_some()
        && routing_relink_candidate_work(&model, &per, &state.lists, 0, 0, 1, state.lists[1].len(), &running).is_some();

    matches!(short, Err(Abort::BudgetExhausted))
        && short_work.generated == 0
        && short_work.structural == 0
        && exact.is_ok_and(|locations| locations.len() == ITEMS)
        && exact_work.generated == index_work
        && exact_work.structural == index_work
        && matches!(stopped_index, Err(Abort::Interrupted))
        && stopped_work.generated == 0
        && stopped_work.structural == 0
        && stopped_floors
        && running_floors
}
