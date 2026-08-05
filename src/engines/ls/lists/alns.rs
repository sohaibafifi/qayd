use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

use smallvec::SmallVec;

use super::incremental::{EvalScratch, InsertView, RemoveView};
use super::local_search::{score_scalar, score_with_replaced_list, tier_values, CandidateNeighbors, ListScore, PerList, Score, State};
use super::metrics::{AdaptiveOperatorMetrics, AlnsSearchMetrics};
use super::moves::{apply_move, best_improving_move, trial_list_score_view, SearchMemory};
use crate::mix64;
use crate::model::list::CollectionModel;

const DESTROY_OPERATORS: usize = 3;
const REPAIR_OPERATORS: usize = 3;
const WEIGHT_SCALE: u64 = 1_000;
const MIN_WEIGHT: u64 = 100;

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
}

#[derive(Clone, Copy)]
pub(super) enum DestroyOperator {
    Shaw = 0,
    Segments = 1,
    Worst = 2,
}

impl DestroyOperator {
    const ALL: [Self; DESTROY_OPERATORS] = [Self::Shaw, Self::Segments, Self::Worst];

    const fn name(self) -> &'static str {
        match self {
            Self::Shaw => "shaw",
            Self::Segments => "segments",
            Self::Worst => "worst",
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum RepairOperator {
    Greedy = 0,
    Regret2 = 1,
    Regret3 = 2,
}

#[derive(Clone, Copy)]
pub(super) struct AlnsChoice {
    pub(super) destroy: DestroyOperator,
    pub(super) repair: RepairOperator,
    pub(super) target: usize,
}

impl RepairOperator {
    const ALL: [Self; REPAIR_OPERATORS] = [Self::Greedy, Self::Regret2, Self::Regret3];

    const fn name(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Regret2 => "regret-2",
            Self::Regret3 => "regret-3",
        }
    }

    const fn regret(self) -> Option<usize> {
        match self {
            Self::Greedy => None,
            Self::Regret2 => Some(2),
            Self::Regret3 => Some(3),
        }
    }
}

#[derive(Clone, Copy)]
struct OperatorState {
    weight: u64,
    pending_reward: u64,
    pending_uses: u64,
    uses: u64,
    accepted: u64,
    improvements: u64,
    global_bests: u64,
}

impl Default for OperatorState {
    fn default() -> Self {
        Self { weight: WEIGHT_SCALE, pending_reward: 0, pending_uses: 0, uses: 0, accepted: 0, improvements: 0, global_bests: 0 }
    }
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
    fn from_env(items: usize, profile: SearchProfile) -> Self {
        let root = integer_sqrt(items.max(1));
        let (temperature, cooling, late_window, destroy_min, destroy_max) = match profile {
            SearchProfile::Sequential | SearchProfile::Balanced => (0.12, 0.997, root.saturating_mul(2), 6, 45),
            SearchProfile::Intensify => (0.05, 0.995, root, 3, 28),
            SearchProfile::Diversify => (0.30, 0.999, root.saturating_mul(4), 15, 65),
        };
        let reaction_milli = env_u64("QAYD_LS_ALNS_REACTION", 200).clamp(1, WEIGHT_SCALE);
        let segment_length = env_u64("QAYD_LS_ALNS_SEGMENT", root as u64).max(1);
        let initial_temperature = env_f64("QAYD_LS_ALNS_TEMPERATURE", temperature).clamp(1.0e-6, 100.0);
        let cooling = env_f64("QAYD_LS_ALNS_COOLING", cooling).clamp(0.5, 1.0);
        let late_window = env_usize("QAYD_LS_LATE_WINDOW", late_window).max(1);
        let min_destroy_percent = env_usize("QAYD_LS_DESTROY_MIN", destroy_min).clamp(1, 100);
        let max_destroy_percent = env_usize("QAYD_LS_DESTROY_MAX", destroy_max).clamp(min_destroy_percent, 100);
        Self { reaction_milli, segment_length, initial_temperature, cooling, late_window, min_destroy_percent, max_destroy_percent }
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).filter(|value: &f64| value.is_finite()).unwrap_or(default)
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
        let config = AlnsConfig::from_env(items, profile);
        Self {
            temperature: config.initial_temperature,
            late_scores: vec![initial.clone(); config.late_window],
            config,
            destroy: [OperatorState::default(); DESTROY_OPERATORS],
            repair: [OperatorState::default(); REPAIR_OPERATORS],
            late_cursor: 0,
            segment_uses: 0,
            metrics: AlnsSearchMetrics::default(),
        }
    }

    pub(super) fn select(&self, seed: u64, iteration: u64) -> (DestroyOperator, RepairOperator) {
        let destroy = roulette(&self.destroy, seed ^ mix64(iteration));
        let repair = roulette(&self.repair, seed ^ mix64(iteration ^ 0xA076_1D64_78BD_642F));
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

    pub(super) fn record(
        &mut self,
        destroy: DestroyOperator,
        repair: RepairOperator,
        acceptance: AcceptanceKind,
        improved_current: bool,
        global_best: bool,
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
        update_operator(&mut self.destroy[destroy as usize], reward, accepted, improved_current, global_best);
        update_operator(&mut self.repair[repair as usize], reward, accepted, improved_current, global_best);
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
            react(&mut self.destroy, self.config.reaction_milli);
            react(&mut self.repair, self.config.reaction_milli);
            self.segment_uses = 0;
        }
    }

    pub(super) fn record_failed(&mut self, destroy: DestroyOperator, repair: RepairOperator) {
        self.record(destroy, repair, AcceptanceKind::Rejected, false, false);
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
        react(&mut self.destroy, self.config.reaction_milli);
        react(&mut self.repair, self.config.reaction_milli);
        self.metrics.destroy =
            DestroyOperator::ALL.iter().enumerate().map(|(idx, operator)| operator_metrics(operator.name(), self.destroy[idx])).collect();
        self.metrics.repair =
            RepairOperator::ALL.iter().enumerate().map(|(idx, operator)| operator_metrics(operator.name(), self.repair[idx])).collect();
        self.metrics
    }
}

fn roulette<const N: usize>(operators: &[OperatorState; N], seed: u64) -> usize {
    let total: u64 = operators.iter().map(|operator| operator.weight).sum();
    let mut pick = mix64(seed) % total.max(1);
    for (idx, operator) in operators.iter().enumerate() {
        if pick < operator.weight {
            return idx;
        }
        pick -= operator.weight;
    }
    N - 1
}

fn update_operator(operator: &mut OperatorState, reward: u64, accepted: bool, improved: bool, global_best: bool) {
    operator.uses = operator.uses.saturating_add(1);
    operator.pending_uses = operator.pending_uses.saturating_add(1);
    operator.pending_reward = operator.pending_reward.saturating_add(reward);
    operator.accepted = operator.accepted.saturating_add(u64::from(accepted));
    operator.improvements = operator.improvements.saturating_add(u64::from(improved));
    operator.global_bests = operator.global_bests.saturating_add(u64::from(global_best));
}

fn react<const N: usize>(operators: &mut [OperatorState; N], reaction_milli: u64) {
    for operator in operators {
        if operator.pending_uses == 0 {
            continue;
        }
        let observed = operator.pending_reward / operator.pending_uses;
        operator.weight =
            ((WEIGHT_SCALE - reaction_milli).saturating_mul(operator.weight).saturating_add(reaction_milli.saturating_mul(observed)))
                / WEIGHT_SCALE;
        operator.weight = operator.weight.max(MIN_WEIGHT);
        operator.pending_reward = 0;
        operator.pending_uses = 0;
    }
}

fn operator_metrics(name: &str, state: OperatorState) -> AdaptiveOperatorMetrics {
    AdaptiveOperatorMetrics {
        name: name.to_owned(),
        uses: state.uses,
        accepted: state.accepted,
        improvements: state.improvements,
        global_bests: state.global_bests,
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

pub(super) fn build_candidate(
    model: &CollectionModel,
    per: &PerList,
    current: &State,
    choice: AlnsChoice,
    seed: u64,
    stop: &AtomicBool,
) -> Option<State> {
    if choice.target == 0 || stop.load(Ordering::Relaxed) {
        return None;
    }
    let mut lists = current.lists.clone();
    let removed = match choice.destroy {
        DestroyOperator::Shaw => destroy_shaw(&mut lists, per.candidates.as_ref(), choice.target, seed),
        DestroyOperator::Segments => destroy_segments(&mut lists, choice.target, seed),
        DestroyOperator::Worst => destroy_worst(per, current, &mut lists, choice.target, seed, stop),
    };
    if removed.is_empty() || stop.load(Ordering::Relaxed) {
        return None;
    }
    let mut state = State::from_lists(model, per, lists);
    let repaired = match choice.repair.regret() {
        Some(k) => repair_regret(per, &mut state, &removed, k, seed ^ 0xA076_1D64_78BD_642F, stop),
        None => repair_greedy(per, &mut state, &removed, seed ^ 0xA076_1D64_78BD_642F, stop),
    };
    if !repaired {
        return None;
    }
    state = State::from_lists(model, per, state.lists);
    let polish_steps = removed.len().saturating_mul(choice.repair.regret().unwrap_or(1));
    polish(per, &mut state, stop, polish_steps);
    Some(state)
}

fn destroy_segments(lists: &mut [Vec<i32>], target: usize, seed: u64) -> Vec<i32> {
    let mut removed = Vec::with_capacity(target);
    let mut step = 0u64;
    while removed.len() < target {
        let total: usize = lists.iter().map(Vec::len).sum();
        if total == 0 {
            break;
        }
        let mut pick = (mix64(seed ^ mix64(step)) % total as u64) as usize;
        let mut route = 0usize;
        let mut pos = 0usize;
        for (idx, list) in lists.iter().enumerate() {
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
        removed.extend(lists[route].drain(start..start + len));
        step = step.wrapping_add(1);
    }
    shuffle_values(&mut removed, seed ^ 0xD1B5_4A32_D192_ED03);
    removed
}

fn destroy_shaw(lists: &mut [Vec<i32>], candidates: Option<&CandidateNeighbors>, target: usize, seed: u64) -> Vec<i32> {
    let present: HashSet<i32> = lists.iter().flatten().copied().collect();
    if present.is_empty() {
        return Vec::new();
    }
    let all: Vec<i32> = lists.iter().flatten().copied().collect();
    let mut removed = HashSet::with_capacity(target);
    let mut order = Vec::with_capacity(target);
    let first = all[(mix64(seed) % all.len() as u64) as usize];
    removed.insert(first);
    order.push(first);
    let mut step = 0u64;
    while removed.len() < target && removed.len() < present.len() {
        let pivot = order[(mix64(seed ^ mix64(step)) % order.len() as u64) as usize];
        let related = candidates
            .and_then(|neighbors| neighbors.nearest_present(pivot, &removed, &present))
            .or_else(|| positional_neighbor(lists, pivot, &removed))
            .or_else(|| {
                let available: Vec<i32> = all.iter().copied().filter(|item| !removed.contains(item)).collect();
                (!available.is_empty()).then(|| available[(mix64(seed ^ step) % available.len() as u64) as usize])
            });
        let Some(next) = related else { break };
        removed.insert(next);
        order.push(next);
        step = step.wrapping_add(1);
    }
    let mut result = Vec::with_capacity(removed.len());
    for list in lists {
        list.retain(|item| {
            if removed.contains(item) {
                result.push(*item);
                false
            } else {
                true
            }
        });
    }
    shuffle_values(&mut result, seed ^ 0xD1B5_4A32_D192_ED03);
    result
}

fn positional_neighbor(lists: &[Vec<i32>], pivot: i32, removed: &HashSet<i32>) -> Option<i32> {
    for list in lists {
        let Some(position) = list.iter().position(|&item| item == pivot) else { continue };
        for distance in 1..list.len() {
            if let Some(&item) = position.checked_sub(distance).and_then(|index| list.get(index)) {
                if !removed.contains(&item) {
                    return Some(item);
                }
            }
            if let Some(&item) = list.get(position + distance) {
                if !removed.contains(&item) {
                    return Some(item);
                }
            }
        }
    }
    None
}

fn destroy_worst(per: &PerList, state: &State, lists: &mut [Vec<i32>], target: usize, seed: u64, stop: &AtomicBool) -> Vec<i32> {
    let mut ranked: Vec<(Score, u64, i32)> = Vec::new();
    let mut list_scratch = EvalScratch::default();
    let mut score_scratch = EvalScratch::default();
    let mut polled = 0u32;
    for (list, contents) in state.lists.iter().enumerate() {
        for pos in 0..contents.len() {
            polled = polled.wrapping_add(1);
            if polled.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                return Vec::new();
            }
            let view = RemoveView::new(contents, pos);
            per.metrics.record_candidate();
            let replacement = trial_list_score_view(per, state, list, &view, None, &mut list_scratch);
            let score = score_with_replaced_list(per, state, list, &replacement, &view, 0, &mut score_scratch);
            let item = contents[pos];
            ranked.push((score, mix64(seed ^ item as i64 as u64), item));
        }
    }
    ranked.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let selected: HashSet<i32> = ranked.into_iter().take(target).map(|(_, _, item)| item).collect();
    let mut removed = Vec::with_capacity(selected.len());
    for list in lists {
        list.retain(|item| {
            if selected.contains(item) {
                removed.push(*item);
                false
            } else {
                true
            }
        });
    }
    removed
}

fn repair_greedy(per: &PerList, state: &mut State, removed: &[i32], seed: u64, stop: &AtomicBool) -> bool {
    let mut scratch = EvalScratch::default();
    let mut polled = 0u32;
    for &item in removed {
        let mut best: Option<(Score, u64, usize, usize, ListScore)> = None;
        for list in 0..state.lists.len() {
            for pos in 0..=state.lists[list].len() {
                polled = polled.wrapping_add(1);
                if polled.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                    return false;
                }
                let candidate = InsertView::new(&state.lists[list], pos, item);
                per.metrics.record_candidate();
                let next = trial_list_score_view(per, state, list, &candidate, None, &mut scratch);
                let global_delta = per.globals.delta(&state.item_list, &[(item, list)]);
                let score = score_with_replaced_list(per, state, list, &next, &candidate, global_delta, &mut scratch);
                let tie = mix64(seed ^ item as i64 as u64 ^ ((list as u64) << 32) ^ pos as u64);
                if best
                    .as_ref()
                    .is_none_or(|(best_score, best_tie, _, _, _)| score < *best_score || (score == *best_score && tie < *best_tie))
                {
                    best = Some((score, tie, list, pos, next));
                }
            }
        }
        let Some((_, _, list, pos, expected)) = best else { return false };
        state.lists[list].insert(pos, item);
        state.rescore(per, list);
        debug_assert_eq!(state.scores[list].violation, expected.violation);
        state.set_item_list(per, item, list);
        state.global_viol = per.globals.total(&state.item_list);
    }
    true
}

fn repair_regret(per: &PerList, state: &mut State, removed: &[i32], k: usize, seed: u64, stop: &AtomicBool) -> bool {
    let mut pending = removed.to_vec();
    let mut scratch = EvalScratch::default();
    let mut polled = 0u32;
    while !pending.is_empty() {
        let mut choice: Option<(usize, i128, u64, usize, usize, ListScore)> = None;
        for (idx, &item) in pending.iter().enumerate() {
            let mut topk: SmallVec<[i128; 4]> = SmallVec::with_capacity(k);
            let mut best_place: Option<(i128, u64, usize, usize, ListScore)> = None;
            for list in 0..state.lists.len() {
                for pos in 0..=state.lists[list].len() {
                    polled = polled.wrapping_add(1);
                    if polled.is_multiple_of(1024) && stop.load(Ordering::Relaxed) {
                        return false;
                    }
                    let candidate = InsertView::new(&state.lists[list], pos, item);
                    per.metrics.record_candidate();
                    let next = trial_list_score_view(per, state, list, &candidate, None, &mut scratch);
                    let global_delta = per.globals.delta(&state.item_list, &[(item, list)]);
                    let cost = score_scalar(&score_with_replaced_list(per, state, list, &next, &candidate, global_delta, &mut scratch));
                    let tie = mix64(seed ^ item as i64 as u64 ^ ((list as u64) << 32) ^ pos as u64);
                    let insert = topk.partition_point(|&candidate_cost| candidate_cost <= cost);
                    if insert < k {
                        topk.insert(insert, cost);
                        topk.truncate(k);
                    }
                    if best_place
                        .as_ref()
                        .is_none_or(|&(best_cost, best_tie, _, _, _)| cost < best_cost || (cost == best_cost && tie < best_tie))
                    {
                        best_place = Some((cost, tie, list, pos, next));
                    }
                }
            }
            let Some((best_cost, best_tie, best_list, best_pos, best_score)) = best_place else { return false };
            let mut regret = 0i128;
            for rank in 1..k {
                let alternative = topk.get(rank).copied().unwrap_or_else(|| best_cost.saturating_add(1i128 << 60));
                regret = regret.saturating_add(alternative.saturating_sub(best_cost));
            }
            if choice
                .as_ref()
                .is_none_or(|&(_, old_regret, old_tie, _, _, _)| regret > old_regret || (regret == old_regret && best_tie < old_tie))
            {
                choice = Some((idx, regret, best_tie, best_list, best_pos, best_score));
            }
        }
        let (idx, _, _, list, pos, expected) = choice.expect("pending is non-empty");
        let item = pending.swap_remove(idx);
        state.lists[list].insert(pos, item);
        state.rescore(per, list);
        debug_assert_eq!(state.scores[list].violation, expected.violation);
        state.set_item_list(per, item, list);
        state.global_viol = per.globals.total(&state.item_list);
    }
    true
}

fn polish(per: &PerList, state: &mut State, stop: &AtomicBool, steps: usize) {
    let mut memory = SearchMemory::new(state.lists.len());
    for _ in 0..steps {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let Some(mv) = best_improving_move(per, state, stop, &mut memory) else { return };
        apply_move(per, state, mv);
        memory.reset_touched(mv);
    }
}

fn shuffle_values(values: &mut [i32], seed: u64) {
    for index in (1..values.len()).rev() {
        let other = (mix64(seed.wrapping_add(index as u64)) % (index as u64 + 1)) as usize;
        values.swap(index, other);
    }
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
