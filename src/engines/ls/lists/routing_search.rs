use std::time::Duration;

use super::alns::{AlnsBuildStatus, MacroOperator};
use super::elite::{EliteOperationStatus, PathRelinkStatus};
use super::metrics::{AnytimeCheckpointMetrics, NeighborhoodSearchMetrics, RoutingAuxiliaryMetrics, RoutingSearchMetrics};
use super::moves::{NeighborhoodKind, NeighborhoodRun};
use crate::mix64;

const WEIGHT_SCALE: u64 = 1_000;
const MIN_WEIGHT: u64 = 10;
const REACTION_MILLI: u64 = 200;
const REWARD_RATE_SCALE: u64 = 1 << 14;
const MAX_OBSERVED_WEIGHT: u64 = 20_000;
const COST_EXPLORATION_REWARD: u64 = 100;
const GUARANTEED_EXPLORATION_INTERVAL: u64 = 32;
const CHECKPOINTS_NANOS: [u64; 4] = [250_000_000, 1_000_000_000, 5_000_000_000, 10_000_000_000];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SliceKind {
    Descent,
    Alns,
    Macro,
    Relink,
    Global,
}

/// Work-sliced routing scheduler. Its cadence and adaptive selectors depend
/// only on deterministic work counters, never on elapsed time.
pub(super) struct SliceScheduler {
    slices: u64,
}

/// Deterministic proxy for the relative CPU cost of an evaluator path.
///
/// The search plan calls for rewards normalized by CPU cost, but feeding
/// measured timings back into a sequential worker would make its trajectory
/// depend on the machine and scheduler. These coefficients are calibrated to
/// the stable work performed by each generation and evaluation path instead.
/// Measured `cpu_nanos` remains telemetry and is never selection input.
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

// The coefficients encode stable relative costs of the evaluator paths. They
// deliberately use no samples from the machine running the search.
const NEIGHBORHOOD_COSTS: [CostModel; 6] = [
    CostModel::new(4, 1, 1, 4),
    CostModel::new(5, 1, 1, 6),
    CostModel::new(5, 1, 2, 6),
    CostModel::new(6, 1, 2, 7),
    CostModel::new(8, 1, 3, 10),
    CostModel::new(3, 1, 1, 3),
];
const MACRO_COSTS: [CostModel; 4] =
    [CostModel::new(12, 2, 1, 8), CostModel::new(10, 2, 1, 7), CostModel::new(8, 1, 1, 6), CostModel::new(12, 2, 2, 9)];
const RELINK_COSTS: [CostModel; 1] = [CostModel::new(8, 1, 1, 6)];

impl SliceScheduler {
    pub(super) const fn new() -> Self {
        Self { slices: 0 }
    }

    pub(super) fn next(&mut self, stagnant: u64, has_relink_target: bool) -> SliceKind {
        let slice = self.slices;
        self.slices = self.slices.saturating_add(1);
        if stagnant >= 16 && slice % 32 == 31 {
            SliceKind::Global
        } else if has_relink_target && stagnant >= 8 && slice % 16 == 15 {
            SliceKind::Relink
        } else if slice % 8 == 7 {
            SliceKind::Macro
        } else if slice.is_multiple_of(2) {
            SliceKind::Descent
        } else {
            SliceKind::Alns
        }
    }
}

#[derive(Clone, Copy)]
struct AdaptiveSlot {
    weight: u64,
    pending_reward_points: u128,
    pending_cost_units: u128,
    uses: u64,
    deterministic_cost: u64,
    generated: u64,
    evaluated: u64,
    cpu_nanos: u64,
    improvements: u64,
    global_bests: u64,
    positive_rewards: u64,
}

#[derive(Clone, Copy, Default)]
struct AuxiliaryProfile {
    uses: u64,
    work_units: u64,
    cpu_nanos: u64,
    productive: u64,
}

impl AuxiliaryProfile {
    fn record(&mut self, work_units: u64, cpu_nanos: u64, productive: bool) {
        self.uses = self.uses.saturating_add(1);
        self.work_units = self.work_units.saturating_add(work_units);
        self.cpu_nanos = self.cpu_nanos.saturating_add(cpu_nanos);
        self.productive = self.productive.saturating_add(u64::from(productive));
    }

    fn metrics(self, name: &str) -> RoutingAuxiliaryMetrics {
        RoutingAuxiliaryMetrics {
            name: name.to_owned(),
            uses: self.uses,
            work_units: self.work_units,
            cpu_nanos: self.cpu_nanos,
            productive: self.productive,
        }
    }
}

impl Default for AdaptiveSlot {
    fn default() -> Self {
        Self {
            weight: WEIGHT_SCALE,
            pending_reward_points: 0,
            pending_cost_units: 0,
            uses: 0,
            deterministic_cost: 0,
            generated: 0,
            evaluated: 0,
            cpu_nanos: 0,
            improvements: 0,
            global_bests: 0,
            positive_rewards: 0,
        }
    }
}

struct AdaptiveSelector<const N: usize> {
    slots: [AdaptiveSlot; N],
    costs: [CostModel; N],
    recorded_observations: u64,
    selections: u64,
}

impl<const N: usize> AdaptiveSelector<N> {
    fn new(costs: [CostModel; N]) -> Self {
        assert!(N > 0, "an adaptive selector needs at least one operator");
        Self { slots: [AdaptiveSlot::default(); N], costs, recorded_observations: 0, selections: 0 }
    }

    fn choose(&mut self, seed: u64) -> usize {
        let selection = self.selections;
        self.selections = self.selections.saturating_add(1);
        if self.slots.iter().any(|slot| slot.uses == 0) {
            let offset = (mix64(seed) % N as u64) as usize;
            if let Some(index) = (0..N).map(|step| (offset + step) % N).find(|&index| self.slots[index].uses == 0) {
                return index;
            }
        }

        // Roulette weights retain a small absolute floor, while this sparse,
        // rotating choice guarantees that even a badly scored operator cannot
        // become unreachable. One forced choice every interval keeps the
        // exploration tax bounded and fully deterministic.
        if selection > 0 && selection.is_multiple_of(GUARANTEED_EXPLORATION_INTERVAL) {
            let completed_intervals = selection / GUARANTEED_EXPLORATION_INTERVAL;
            return (completed_intervals.saturating_sub(1) % N as u64) as usize;
        }

        let total = self.slots.iter().map(|slot| slot.weight.max(MIN_WEIGHT)).sum::<u64>().max(1);
        let mut pick = mix64(seed) % total;
        for (index, slot) in self.slots.iter().enumerate() {
            let weight = slot.weight.max(MIN_WEIGHT);
            if pick < weight {
                return index;
            }
            pick -= weight;
        }
        N.saturating_sub(1)
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        index: usize,
        work: u64,
        generated: u64,
        evaluated: u64,
        cpu_nanos: u64,
        improved: bool,
        global_best: bool,
        accepted: bool,
    ) {
        let quality_reward: u64 = if global_best {
            20_000
        } else if improved {
            8_000
        } else if accepted {
            3_000
        } else {
            0
        };
        // The sparse deterministic exploration in `choose` is sufficient to
        // keep every operator reachable. Reward the cost of an observation
        // only when the operator actually produced candidate work. Otherwise
        // a completed cursor would repeatedly earn a high reward rate from
        // its tiny fixed cost and eventually dominate useful neighborhoods.
        let observed_candidates = generated > 0 || evaluated > 0;
        let adaptive_reward = quality_reward.saturating_add(if observed_candidates { COST_EXPLORATION_REWARD } else { 0 });
        let deterministic_cost = self.costs[index].estimate(work, generated, evaluated);
        let slot = &mut self.slots[index];
        slot.uses = slot.uses.saturating_add(1);
        slot.deterministic_cost = slot.deterministic_cost.saturating_add(deterministic_cost);
        slot.generated = slot.generated.saturating_add(generated);
        slot.evaluated = slot.evaluated.saturating_add(evaluated);
        slot.cpu_nanos = slot.cpu_nanos.saturating_add(cpu_nanos);
        slot.improvements = slot.improvements.saturating_add(u64::from(improved));
        slot.global_bests = slot.global_bests.saturating_add(u64::from(global_best));
        slot.positive_rewards = slot.positive_rewards.saturating_add(u64::from(quality_reward > 0));
        slot.pending_cost_units = slot.pending_cost_units.saturating_add(u128::from(deterministic_cost));
        slot.pending_reward_points = slot.pending_reward_points.saturating_add(u128::from(adaptive_reward));
        self.recorded_observations = self.recorded_observations.saturating_add(1);
        if self.recorded_observations.is_multiple_of(N as u64 * 2) {
            self.react();
        }
    }

    fn react(&mut self) {
        for slot in &mut self.slots {
            if slot.pending_cost_units == 0 {
                continue;
            }
            let observed = fixed_point_reward_rate(slot.pending_reward_points, slot.pending_cost_units);
            let retained = u128::from(WEIGHT_SCALE - REACTION_MILLI).saturating_mul(u128::from(slot.weight));
            let learned = u128::from(REACTION_MILLI).saturating_mul(u128::from(observed));
            let blended = retained.saturating_add(learned) / u128::from(WEIGHT_SCALE);
            slot.weight = u64::try_from(blended).unwrap_or(MAX_OBSERVED_WEIGHT);
            slot.weight = slot.weight.max(MIN_WEIGHT);
            slot.pending_reward_points = 0;
            slot.pending_cost_units = 0;
        }
    }

    fn metrics(&mut self, names: &[&str; N]) -> Vec<NeighborhoodSearchMetrics> {
        self.react();
        names
            .iter()
            .zip(self.slots)
            .map(|(&name, slot)| NeighborhoodSearchMetrics {
                name: name.to_owned(),
                uses: slot.uses,
                generated: slot.generated,
                evaluated: slot.evaluated,
                cpu_nanos: slot.cpu_nanos,
                improvements: slot.improvements,
                global_bests: slot.global_bests,
                positive_rewards: slot.positive_rewards,
                weight_milli: slot.weight,
            })
            .collect()
    }
}

fn fixed_point_reward_rate(reward_points: u128, cost_units: u128) -> u64 {
    let scaled = reward_points.saturating_mul(u128::from(REWARD_RATE_SCALE)) / cost_units.max(1);
    u64::try_from(scaled.min(u128::from(MAX_OBSERVED_WEIGHT))).unwrap_or(MAX_OBSERVED_WEIGHT)
}

struct CheckpointTracker {
    next: usize,
    records: Vec<AnytimeCheckpointMetrics>,
    last: CheckpointSnapshot,
}

struct CheckpointSnapshot {
    feasible: bool,
    objectives: Vec<i64>,
    fleet: Option<usize>,
    candidates: u64,
}

impl CheckpointTracker {
    fn new() -> Self {
        Self {
            next: 0,
            records: Vec::with_capacity(CHECKPOINTS_NANOS.len()),
            last: CheckpointSnapshot { feasible: false, objectives: Vec::new(), fleet: None, candidates: 0 },
        }
    }

    fn observe(&mut self, elapsed: Duration, feasible: bool, objectives: &[i64], fleet: usize, candidates: u64) {
        let observed_nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        while self.next < CHECKPOINTS_NANOS.len() && CHECKPOINTS_NANOS[self.next] <= observed_nanos {
            self.records.push(AnytimeCheckpointMetrics {
                target_nanos: CHECKPOINTS_NANOS[self.next],
                observed_nanos,
                feasible: self.last.feasible,
                objectives: self.last.objectives.clone(),
                fleet: self.last.fleet,
                candidates: self.last.candidates,
            });
            self.next += 1;
        }
        self.last = if feasible {
            CheckpointSnapshot { feasible, objectives: objectives.to_vec(), fleet: Some(fleet), candidates }
        } else {
            CheckpointSnapshot { feasible, objectives: Vec::new(), fleet: None, candidates }
        };
    }
}

pub(super) struct RoutingSearchControl {
    scheduler: SliceScheduler,
    neighborhoods: AdaptiveSelector<6>,
    macros: AdaptiveSelector<4>,
    relink: AdaptiveSelector<1>,
    elite_archive: AuxiliaryProfile,
    elite_selection: AuxiliaryProfile,
    checkpoints: Option<CheckpointTracker>,
    metrics: RoutingSearchMetrics,
    candidates: u64,
}

impl RoutingSearchControl {
    pub(super) fn new(profile: bool, construction_candidates: u64) -> Self {
        Self {
            scheduler: SliceScheduler::new(),
            neighborhoods: AdaptiveSelector::new(NEIGHBORHOOD_COSTS),
            macros: AdaptiveSelector::new(MACRO_COSTS),
            relink: AdaptiveSelector::new(RELINK_COSTS),
            elite_archive: AuxiliaryProfile::default(),
            elite_selection: AuxiliaryProfile::default(),
            checkpoints: profile.then(CheckpointTracker::new),
            metrics: RoutingSearchMetrics::default(),
            candidates: construction_candidates,
        }
    }

    pub(super) fn next_slice(&mut self, stagnant: u64, has_relink_target: bool) -> SliceKind {
        self.scheduler.next(stagnant, has_relink_target)
    }

    pub(super) fn choose_neighborhood(&mut self, seed: u64) -> NeighborhoodKind {
        NeighborhoodKind::ALL[self.neighborhoods.choose(seed)]
    }

    pub(super) fn choose_macro(&mut self, seed: u64) -> MacroOperator {
        MacroOperator::ALL[self.macros.choose(seed)]
    }

    pub(super) fn checkpoints_enabled(&self) -> bool {
        self.checkpoints.is_some()
    }

    pub(super) fn observe_checkpoint(&mut self, elapsed: Duration, feasible: bool, objectives: &[i64], fleet: usize) {
        self.observe_checkpoint_with_candidates(elapsed, feasible, objectives, fleet, self.candidates);
    }

    pub(super) fn observe_checkpoint_with_candidates(
        &mut self,
        elapsed: Duration,
        feasible: bool,
        objectives: &[i64],
        fleet: usize,
        candidates: u64,
    ) {
        if let Some(checkpoints) = &mut self.checkpoints {
            checkpoints.observe(elapsed, feasible, objectives, fleet, candidates);
        }
    }

    pub(super) fn record_descent(
        &mut self,
        kind: NeighborhoodKind,
        run: NeighborhoodRun,
        improved: bool,
        global_best: bool,
        accepted: bool,
    ) {
        self.metrics.slices = self.metrics.slices.saturating_add(1);
        self.metrics.descent_slices = self.metrics.descent_slices.saturating_add(1);
        self.candidates = self.candidates.saturating_add(run.evaluated);
        self.neighborhoods.record(
            kind.index(),
            run.generation_work,
            run.generated,
            run.evaluated,
            run.cpu_nanos,
            improved,
            global_best,
            accepted,
        );
    }

    pub(super) fn record_global(
        &mut self,
        kind: NeighborhoodKind,
        run: NeighborhoodRun,
        improved: bool,
        global_best: bool,
        accepted: bool,
    ) {
        self.metrics.slices = self.metrics.slices.saturating_add(1);
        self.metrics.global_scan_slices = self.metrics.global_scan_slices.saturating_add(1);
        self.candidates = self.candidates.saturating_add(run.evaluated);
        self.neighborhoods.record(
            kind.index(),
            run.generation_work,
            run.generated,
            run.evaluated,
            run.cpu_nanos,
            improved,
            global_best,
            accepted,
        );
    }

    pub(super) fn record_alns(&mut self, evaluated: u64) {
        self.metrics.slices = self.metrics.slices.saturating_add(1);
        self.metrics.alns_slices = self.metrics.alns_slices.saturating_add(1);
        self.candidates = self.candidates.saturating_add(evaluated);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_macro(
        &mut self,
        operator: MacroOperator,
        status: AlnsBuildStatus,
        generated: u64,
        evaluated: u64,
        cpu_nanos: u64,
        improved: bool,
        global_best: bool,
        accepted: bool,
    ) {
        self.metrics.slices = self.metrics.slices.saturating_add(1);
        self.candidates = self.candidates.saturating_add(evaluated);
        match operator {
            MacroOperator::RouteElimination => {
                self.metrics.route_elimination_attempts = self.metrics.route_elimination_attempts.saturating_add(1);
            }
            MacroOperator::EjectionChain => {
                self.metrics.ejection_chain_attempts = self.metrics.ejection_chain_attempts.saturating_add(1);
            }
            MacroOperator::ChainRelocate => {
                self.metrics.chain_relocate_attempts = self.metrics.chain_relocate_attempts.saturating_add(1);
            }
            MacroOperator::GuidedSegmentExchange => {
                self.metrics.guided_segment_exchange_attempts = self.metrics.guided_segment_exchange_attempts.saturating_add(1);
            }
        }
        self.metrics.macro_candidates_built =
            self.metrics.macro_candidates_built.saturating_add(u64::from(status == AlnsBuildStatus::Built));
        self.metrics.macro_budget_exhaustions =
            self.metrics.macro_budget_exhaustions.saturating_add(u64::from(status == AlnsBuildStatus::BudgetExhausted));
        self.macros.record(
            operator.index(),
            generated.saturating_add(evaluated),
            generated,
            evaluated,
            cpu_nanos,
            improved,
            global_best,
            accepted,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_relink(
        &mut self,
        status: PathRelinkStatus,
        steps: usize,
        work_units: u64,
        generated: u64,
        evaluated: u64,
        cpu_nanos: u64,
        improved: bool,
        global_best: bool,
        accepted: bool,
    ) {
        self.metrics.slices = self.metrics.slices.saturating_add(1);
        self.metrics.relink_slices = self.metrics.relink_slices.saturating_add(1);
        self.metrics.path_relink_attempts = self.metrics.path_relink_attempts.saturating_add(1);
        self.metrics.path_relink_steps = self.metrics.path_relink_steps.saturating_add(u64::try_from(steps).unwrap_or(u64::MAX));
        self.metrics.path_relink_budget_exhaustions =
            self.metrics.path_relink_budget_exhaustions.saturating_add(u64::from(status == PathRelinkStatus::BudgetExhausted));
        self.candidates = self.candidates.saturating_add(evaluated);
        debug_assert_eq!(work_units, generated.saturating_add(evaluated));
        self.relink.record(0, work_units, generated, evaluated, cpu_nanos, improved, global_best, accepted);
    }

    pub(super) fn record_relink_slice_without_target(&mut self) {
        self.metrics.slices = self.metrics.slices.saturating_add(1);
        self.metrics.relink_slices = self.metrics.relink_slices.saturating_add(1);
    }

    pub(super) fn record_elite_archive(&mut self, status: EliteOperationStatus, inserted: bool, work_units: u64, cpu_nanos: u64) {
        let completed = status == EliteOperationStatus::Complete;
        self.elite_archive.record(work_units, cpu_nanos, completed && inserted);
        if completed {
            if inserted {
                self.metrics.elite_insertions = self.metrics.elite_insertions.saturating_add(1);
            } else {
                self.metrics.elite_rejections = self.metrics.elite_rejections.saturating_add(1);
            }
        }
    }

    pub(super) fn record_elite_selection(&mut self, status: EliteOperationStatus, target_selected: bool, work_units: u64, cpu_nanos: u64) {
        self.elite_selection.record(work_units, cpu_nanos, status == EliteOperationStatus::Complete && target_selected);
    }

    pub(super) fn finish(mut self) -> RoutingSearchMetrics {
        const NEIGHBORHOOD_NAMES: [&str; 6] = ["relocate", "swap", "or-opt", "two-opt-star", "cross-exchange", "reverse"];
        const MACRO_NAMES: [&str; 4] = ["route-elimination", "ejection-chain", "chain-relocate", "guided-segment-exchange"];
        const RELINK_NAME: [&str; 1] = ["path-relink"];
        self.metrics.neighborhoods = self.neighborhoods.metrics(&NEIGHBORHOOD_NAMES);
        self.metrics.neighborhoods.extend(self.macros.metrics(&MACRO_NAMES));
        self.metrics.neighborhoods.extend(self.relink.metrics(&RELINK_NAME));
        self.metrics.auxiliary = vec![self.elite_archive.metrics("elite-archive"), self.elite_selection.metrics("elite-selection")];
        self.metrics.checkpoints = self.checkpoints.map_or_else(Vec::new, |tracker| tracker.records);
        self.metrics
    }
}

#[cfg(test)]
pub(crate) fn audit_routing_activity_counters() -> bool {
    let mut control = RoutingSearchControl::new(false, 0);
    control.record_macro(MacroOperator::RouteElimination, AlnsBuildStatus::Built, 10, 2, 100, false, false, false);
    control.record_macro(MacroOperator::EjectionChain, AlnsBuildStatus::BudgetExhausted, 20, 3, 200, false, false, false);
    control.record_macro(MacroOperator::ChainRelocate, AlnsBuildStatus::Infeasible, 30, 4, 300, false, false, false);
    control.record_macro(MacroOperator::GuidedSegmentExchange, AlnsBuildStatus::Interrupted, 40, 5, 400, false, false, false);
    control.record_relink_slice_without_target();
    control.record_relink(PathRelinkStatus::Complete, 3, 18, 12, 6, 500, false, false, false);
    control.record_relink(PathRelinkStatus::BudgetExhausted, 2, 21, 14, 7, 600, false, false, false);
    control.record_elite_archive(EliteOperationStatus::Complete, true, 50, 700);
    control.record_elite_archive(EliteOperationStatus::BudgetExhausted, false, 25, 300);
    control.record_elite_selection(EliteOperationStatus::Complete, true, 30, 200);
    control.record_elite_selection(EliteOperationStatus::Interrupted, false, 10, 100);

    let metrics = control.finish();
    let profile_uses = |name: &str| metrics.neighborhoods.iter().find(|entry| entry.name == name).map(|entry| entry.uses);
    let auxiliary = |name: &str| metrics.auxiliary.iter().find(|entry| entry.name == name);
    metrics.slices == 7
        && metrics.relink_slices == 3
        && metrics.route_elimination_attempts == 1
        && metrics.ejection_chain_attempts == 1
        && metrics.chain_relocate_attempts == 1
        && metrics.guided_segment_exchange_attempts == 1
        && metrics.macro_candidates_built == 1
        && metrics.macro_budget_exhaustions == 1
        && metrics.path_relink_attempts == 2
        && metrics.path_relink_steps == 5
        && metrics.path_relink_budget_exhaustions == 1
        && metrics.elite_insertions == 1
        && metrics.elite_rejections == 0
        && profile_uses("route-elimination") == Some(1)
        && profile_uses("ejection-chain") == Some(1)
        && profile_uses("chain-relocate") == Some(1)
        && profile_uses("guided-segment-exchange") == Some(1)
        && profile_uses("path-relink") == Some(2)
        && auxiliary("elite-archive")
            .is_some_and(|entry| entry.uses == 2 && entry.work_units == 75 && entry.cpu_nanos == 1_000 && entry.productive == 1)
        && auxiliary("elite-selection")
            .is_some_and(|entry| entry.uses == 2 && entry.work_units == 40 && entry.cpu_nanos == 300 && entry.productive == 1)
}

#[cfg(test)]
pub(crate) fn audit_scheduler_prefix(stagnant: u64, has_relink_target: bool, length: usize) -> Vec<SliceKind> {
    let mut scheduler = SliceScheduler::new();
    (0..length).map(|_| scheduler.next(stagnant, has_relink_target)).collect()
}

#[cfg(test)]
pub(crate) fn audit_checkpoint_history() -> bool {
    let mut early = CheckpointTracker::new();
    early.observe(Duration::from_millis(100), true, &[10], 3, 20);
    early.observe(Duration::from_millis(300), true, &[5], 3, 40);

    let mut late = CheckpointTracker::new();
    late.observe(Duration::from_millis(300), true, &[10], 3, 20);
    late.observe(Duration::from_millis(1_100), true, &[5], 3, 40);

    let mut improved_between_observations = CheckpointTracker::new();
    improved_between_observations.observe(Duration::from_millis(200), true, &[10], 3, 20);
    improved_between_observations.observe(Duration::from_millis(201), true, &[8], 3, 21);
    improved_between_observations.observe(Duration::from_millis(300), true, &[8], 3, 21);

    let mut infeasible = CheckpointTracker::new();
    infeasible.observe(Duration::from_millis(100), true, &[10], 3, 20);
    infeasible.observe(Duration::from_millis(200), false, &[999], 99, 30);
    infeasible.observe(Duration::from_millis(300), false, &[999], 99, 40);

    early.records.len() == 1
        && early.records[0].feasible
        && early.records[0].objectives == [10]
        && early.records[0].candidates == 20
        && late.records.len() == 2
        && !late.records[0].feasible
        && late.records[0].objectives.is_empty()
        && late.records[1].feasible
        && late.records[1].objectives == [10]
        && late.records[1].candidates == 20
        && improved_between_observations.records.len() == 1
        && improved_between_observations.records[0].objectives == [8]
        && improved_between_observations.records[0].candidates == 21
        && infeasible.records.len() == 1
        && !infeasible.records[0].feasible
        && infeasible.records[0].objectives.is_empty()
        && infeasible.records[0].fleet.is_none()
        && infeasible.records[0].candidates == 30
}

#[cfg(test)]
pub(crate) fn audit_neighborhood_learning() -> bool {
    const ROUTING_WORK: u64 = 1_600;
    const DRAWS: u64 = 8_192;

    let costs = [NEIGHBORHOOD_COSTS[0], NEIGHBORHOOD_COSTS[0]];
    let realistic_cost = costs[0].estimate(ROUTING_WORK, ROUTING_WORK, ROUTING_WORK);
    let integer_rate_would_vanish = 8_000u64.saturating_add(COST_EXPLORATION_REWARD) / realistic_cost == 0;
    let mut reference_timing = AdaptiveSelector::<2>::new(costs);
    let mut inverted_timing = AdaptiveSelector::<2>::new(costs);
    for observation in 0u64..64 {
        let productive = observation.is_multiple_of(2);
        reference_timing.record(0, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 25_000_000, productive, false, productive);
        reference_timing.record(1, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 1, false, false, false);
        inverted_timing.record(0, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 1, productive, false, productive);
        inverted_timing.record(1, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 25_000_000, false, false, false);
    }

    let same_weights = reference_timing.slots.iter().zip(inverted_timing.slots).all(|(left, right)| left.weight == right.weight);
    let mut reference_draws = [0u64; 2];
    let mut inverted_draws = [0u64; 2];
    let mut same_choices = true;
    for seed in 0..DRAWS {
        let reference_choice = reference_timing.choose(seed);
        let inverted_choice = inverted_timing.choose(seed);
        same_choices &= reference_choice == inverted_choice;
        reference_draws[reference_choice] = reference_draws[reference_choice].saturating_add(1);
        inverted_draws[inverted_choice] = inverted_draws[inverted_choice].saturating_add(1);
    }
    let guaranteed_idle_draws = DRAWS.saturating_sub(1) / (GUARANTEED_EXPLORATION_INTERVAL * 2);
    let reference_metrics = reference_timing.metrics(&["productive", "idle"]);
    let inverted_metrics = inverted_timing.metrics(&["productive", "idle"]);

    integer_rate_would_vanish
        && same_weights
        && same_choices
        && reference_metrics[0].weight_milli > reference_metrics[1].weight_milli
        && reference_metrics[0].weight_milli == inverted_metrics[0].weight_milli
        && reference_metrics[1].weight_milli == inverted_metrics[1].weight_milli
        && reference_draws == inverted_draws
        && reference_draws[0] > reference_draws[1]
        && reference_draws[1] >= guaranteed_idle_draws
        && reference_metrics[0].positive_rewards > 0
        && reference_metrics[0].cpu_nanos != inverted_metrics[0].cpu_nanos
        && reference_metrics[1].cpu_nanos != inverted_metrics[1].cpu_nanos
}

#[cfg(test)]
pub(crate) fn audit_guaranteed_exploration() -> bool {
    let mut selector = AdaptiveSelector::<4>::new([NEIGHBORHOOD_COSTS[0]; 4]);
    for slot in &mut selector.slots {
        slot.uses = 1;
        slot.weight = MIN_WEIGHT;
    }
    selector.slots[0].weight = MAX_OBSERVED_WEIGHT;

    let total = selector.slots.iter().map(|slot| slot.weight.max(MIN_WEIGHT)).sum::<u64>();
    let seed = (0..u64::MAX)
        .find(|&candidate| mix64(candidate) % total < selector.slots[0].weight)
        .expect("a roulette seed selecting the dominant operator");

    (1..=4).all(|interval| {
        selector.selections = GUARANTEED_EXPLORATION_INTERVAL * interval;
        selector.choose(seed) == usize::try_from(interval - 1).unwrap()
    })
}

#[cfg(test)]
pub(crate) fn audit_timing_independent_cost_learning() -> bool {
    const ROUTING_WORK: u64 = 1_600;
    let costs = [CostModel::new(2, 1, 1, 2), CostModel::new(12, 3, 2, 8)];
    let mut first_timeline = AdaptiveSelector::<2>::new(costs);
    let mut second_timeline = AdaptiveSelector::<2>::new(costs);
    for _ in 0..8 {
        first_timeline.record(0, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 100_000, true, false, true);
        first_timeline.record(1, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 10_000_000, true, false, true);
        second_timeline.record(0, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 20_000_000, true, false, true);
        second_timeline.record(1, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 1, true, false, true);
    }
    let same_weights = first_timeline.slots.iter().zip(second_timeline.slots).all(|(left, right)| left.weight == right.weight);
    let same_choices = (0..4_096).all(|seed| first_timeline.choose(seed) == second_timeline.choose(seed));
    let first = first_timeline.metrics(&["cheap", "expensive"]);
    let second = second_timeline.metrics(&["cheap", "expensive"]);
    same_weights
        && same_choices
        && first[0].weight_milli > first[1].weight_milli
        && first[0].weight_milli == second[0].weight_milli
        && first[1].weight_milli == second[1].weight_milli
        && first[0].cpu_nanos != second[0].cpu_nanos
        && first[1].cpu_nanos != second[1].cpu_nanos
}

#[cfg(test)]
pub(crate) fn audit_unproductive_cost_balance() -> bool {
    const ROUTING_WORK: u64 = 1_600;
    let costs = [NEIGHBORHOOD_COSTS[0], NEIGHBORHOOD_COSTS[4]];
    let mut selector = AdaptiveSelector::<2>::new(costs);
    for _ in 0..8 {
        selector.record(0, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 10_000_000, false, false, false);
        selector.record(1, ROUTING_WORK, ROUTING_WORK, ROUTING_WORK, 1, false, false, false);
    }
    let mut uses = [0u64; 2];
    for draw in 0..20_000 {
        uses[selector.choose(draw)] += 1;
    }
    let projected = [
        uses[0].saturating_mul(costs[0].estimate(ROUTING_WORK, ROUTING_WORK, ROUTING_WORK)),
        uses[1].saturating_mul(costs[1].estimate(ROUTING_WORK, ROUTING_WORK, ROUTING_WORK)),
    ];
    let total = projected[0].saturating_add(projected[1]);
    projected[1].saturating_mul(100) < total.saturating_mul(70)
}

#[cfg(test)]
pub(crate) fn audit_exhausted_neighborhood_learning() -> bool {
    const OBSERVATIONS: u64 = 64;
    const DRAWS: u64 = 8_192;
    const CANDIDATE_WORK: u64 = 64;

    let costs = [NEIGHBORHOOD_COSTS[0]; 2];
    let mut single_observation = AdaptiveSelector::<2>::new(costs);
    single_observation.record(0, 0, 0, 0, 1, false, false, false);
    let empty_reward_withheld =
        single_observation.slots[0].pending_reward_points == 0 && single_observation.slots[0].pending_cost_units > 0;

    let mut selector = AdaptiveSelector::<2>::new(costs);
    for _ in 0..OBSERVATIONS {
        selector.record(0, 0, 0, 0, 1, false, false, false);
        selector.record(1, CANDIDATE_WORK, CANDIDATE_WORK, CANDIDATE_WORK, 1, false, false, false);
    }

    let exhausted_weight = selector.slots[0].weight;
    let productive_weight = selector.slots[1].weight;
    let mut draws = [0u64; 2];
    for seed in 0..DRAWS {
        let choice = selector.choose(seed);
        draws[choice] = draws[choice].saturating_add(1);
    }

    empty_reward_withheld
        && exhausted_weight < WEIGHT_SCALE
        && exhausted_weight < productive_weight
        && selector.slots[0].positive_rewards == 0
        && draws.into_iter().sum::<u64>() == DRAWS
        && draws[0] < draws[1]
}
