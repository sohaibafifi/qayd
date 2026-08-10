use std::cell::RefCell;
use std::fmt;
use std::time::{Duration, Instant};

use crate::model::list::{Iterable, ReduceOp, Reduction};

const ITERABLE_KINDS: usize = 6;
const REDUCE_OP_KINDS: usize = 6;
const REDUCTION_KINDS: usize = ITERABLE_KINDS * REDUCE_OP_KINDS;

/// Statistics for one adaptive destroy or repair operator.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdaptiveOperatorMetrics {
    pub name: String,
    pub uses: u64,
    /// Candidates produced before feasibility and granular filters.
    pub generated: u64,
    /// Complete candidates whose score was evaluated.
    pub evaluated: u64,
    /// Deterministic generated plus evaluated work, retained for audit.
    pub work_units: u64,
    /// Observed thread CPU time, exported as telemetry only.
    pub cpu_nanos: u64,
    pub accepted: u64,
    pub improvements: u64,
    pub global_bests: u64,
    pub positive_rewards: u64,
    /// Final roulette weight, scaled by 1,000.
    pub weight_milli: u64,
}

/// One internal anytime checkpoint. Checkpoints describe the incumbent that
/// existed at `target_nanos`; they are never fed back into search control.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct AnytimeCheckpointMetrics {
    pub target_nanos: u64,
    pub observed_nanos: u64,
    pub feasible: bool,
    pub objectives: Vec<i64>,
    pub fleet: Option<usize>,
    pub candidates: u64,
}

/// Work and reward profile for one bounded routing neighborhood.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct NeighborhoodSearchMetrics {
    pub name: String,
    pub uses: u64,
    pub generated: u64,
    pub evaluated: u64,
    /// Observed thread CPU time, exported as telemetry only.
    pub cpu_nanos: u64,
    pub improvements: u64,
    pub global_bests: u64,
    pub positive_rewards: u64,
    /// Final adaptive weight, scaled by 1,000.
    pub weight_milli: u64,
}

/// CPU and deterministic work spent by routing support mechanisms that are
/// outside an adaptive candidate neighborhood.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RoutingAuxiliaryMetrics {
    pub name: String,
    pub uses: u64,
    pub work_units: u64,
    pub cpu_nanos: u64,
    pub productive: u64,
}

/// Routing scheduler, neighborhood, and elite-search diagnostics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RoutingSearchMetrics {
    pub slices: u64,
    pub descent_slices: u64,
    pub alns_slices: u64,
    pub relink_slices: u64,
    pub global_scan_slices: u64,
    pub route_elimination_attempts: u64,
    pub ejection_chain_attempts: u64,
    pub chain_relocate_attempts: u64,
    pub guided_segment_exchange_attempts: u64,
    pub macro_candidates_built: u64,
    pub macro_budget_exhaustions: u64,
    pub elite_insertions: u64,
    pub elite_rejections: u64,
    pub path_relink_attempts: u64,
    pub path_relink_steps: u64,
    pub path_relink_budget_exhaustions: u64,
    pub checkpoints: Vec<AnytimeCheckpointMetrics>,
    pub neighborhoods: Vec<NeighborhoodSearchMetrics>,
    pub auxiliary: Vec<RoutingAuxiliaryMetrics>,
}

/// Adaptive large-neighborhood-search statistics for one list search.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AlnsSearchMetrics {
    pub iterations: u64,
    pub accepted: u64,
    pub improving: u64,
    pub global_bests: u64,
    pub simulated_annealing_accepts: u64,
    pub late_acceptance_accepts: u64,
    pub gls_updates: u64,
    pub gls_penalties: u64,
    /// Strict improvements published to a shared list-search incumbent.
    pub shared_publications: u64,
    /// Better shared incumbents injected into this worker.
    pub shared_injections: u64,
    pub destroy: Vec<AdaptiveOperatorMetrics>,
    pub repair: Vec<AdaptiveOperatorMetrics>,
}

/// Iterable family reported by the ordered-list local-search profiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListIterableKind {
    Items,
    SetItems,
    Edges,
    Pairs,
    Scan,
    Windows,
}

impl ListIterableKind {
    const ALL: [Self; ITERABLE_KINDS] = [Self::Items, Self::SetItems, Self::Edges, Self::Pairs, Self::Scan, Self::Windows];

    const fn index(self) -> usize {
        match self {
            Self::Items => 0,
            Self::SetItems => 1,
            Self::Edges => 2,
            Self::Pairs => 3,
            Self::Scan => 4,
            Self::Windows => 5,
        }
    }
}

impl fmt::Display for ListIterableKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Items => "items",
            Self::SetItems => "set_items",
            Self::Edges => "edges",
            Self::Pairs => "pairs",
            Self::Scan => "scan",
            Self::Windows => "windows",
        })
    }
}

/// Aggregate operator reported by the ordered-list local-search profiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListReduceOpKind {
    Sum,
    Min,
    Max,
    Count,
    Used,
    SelectKth,
}

impl ListReduceOpKind {
    const ALL: [Self; REDUCE_OP_KINDS] = [Self::Sum, Self::Min, Self::Max, Self::Count, Self::Used, Self::SelectKth];

    const fn index(self) -> usize {
        match self {
            Self::Sum => 0,
            Self::Min => 1,
            Self::Max => 2,
            Self::Count => 3,
            Self::Used => 4,
            Self::SelectKth => 5,
        }
    }
}

impl fmt::Display for ListReduceOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::Count => "count",
            Self::Used => "used",
            Self::SelectKth => "select_kth",
        })
    }
}

/// Timing and work counters for one iterable/operator combination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionSearchMetrics {
    pub iterable: ListIterableKind,
    pub op: ListReduceOpKind,
    pub full_calls: u64,
    pub full_nanos: u64,
    /// Number of emitted iterable values visited by full evaluations.
    pub full_steps: u64,
    pub delta_calls: u64,
    pub delta_nanos: u64,
}

impl ReductionSearchMetrics {
    pub fn average_full_nanos(&self) -> f64 {
        ratio(self.full_nanos, self.full_calls)
    }

    pub fn average_delta_nanos(&self) -> f64 {
        ratio(self.delta_nanos, self.delta_calls)
    }

    pub fn average_full_nanos_per_step(&self) -> f64 {
        ratio(self.full_nanos, self.full_steps)
    }
}

/// Opt-in profile of one ordered-list local-search run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ListSearchMetrics {
    pub elapsed_nanos: u64,
    /// Initial solution strategy selected by the bounded constructor portfolio.
    pub constructor: Option<String>,
    pub construction_nanos: u64,
    pub time_to_first_feasible_nanos: Option<u64>,
    pub construction_candidates: u64,
    pub constructor_fleet: Option<usize>,
    pub constructor_cost: Option<i64>,
    /// Complete move or repair placements whose score was compared.
    pub candidates: u64,
    /// Candidate list contributions evaluated either incrementally or in full.
    pub trial_list_evaluations: u64,
    pub incremental_trial_list_evaluations: u64,
    pub full_recompute_trial_list_evaluations: u64,
    /// Number of scan ranges evaluated and total positions recomputed. Phase 1
    /// records complete scans; later incremental scans can record suffixes here.
    pub scan_recomputations: u64,
    pub scan_recomputed_steps: u64,
    pub alns: AlnsSearchMetrics,
    pub(crate) routing: RoutingSearchMetrics,
    pub reductions: Vec<ReductionSearchMetrics>,
}

impl ListSearchMetrics {
    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.elapsed_nanos)
    }

    pub fn candidates_per_second(&self) -> f64 {
        if self.elapsed_nanos == 0 {
            0.0
        } else {
            self.candidates as f64 * 1_000_000_000.0 / self.elapsed_nanos as f64
        }
    }

    pub fn full_recompute_percentage(&self) -> f64 {
        if self.trial_list_evaluations == 0 {
            0.0
        } else {
            self.full_recompute_trial_list_evaluations as f64 * 100.0 / self.trial_list_evaluations as f64
        }
    }

    pub fn average_recomputed_scan_steps(&self) -> f64 {
        ratio(self.scan_recomputed_steps, self.scan_recomputations)
    }

    pub fn reduction(&self, iterable: ListIterableKind, op: ListReduceOpKind) -> Option<&ReductionSearchMetrics> {
        self.reductions.iter().find(|metrics| metrics.iterable == iterable && metrics.op == op)
    }

    pub(crate) fn routing(&self) -> &RoutingSearchMetrics {
        &self.routing
    }

    pub(crate) fn anytime_checkpoints_metadata(&self) -> String {
        let mut encoded = String::from("1");
        for checkpoint in &self.routing.checkpoints {
            encoded.push(';');
            encoded.push_str(&checkpoint.target_nanos.to_string());
            encoded.push(',');
            encoded.push_str(&checkpoint.observed_nanos.to_string());
            encoded.push(',');
            encoded.push(if checkpoint.feasible { '1' } else { '0' });
            encoded.push(',');
            if let Some(fleet) = checkpoint.fleet {
                encoded.push_str(&fleet.to_string());
            }
            encoded.push(',');
            encoded.push_str(&checkpoint.candidates.to_string());
            encoded.push(',');
            for (index, objective) in checkpoint.objectives.iter().enumerate() {
                if index > 0 {
                    encoded.push(':');
                }
                encoded.push_str(&objective.to_string());
            }
        }
        encoded
    }

    pub(crate) fn neighborhoods_metadata(&self) -> String {
        let mut encoded = String::from("1");
        for neighborhood in &self.routing.neighborhoods {
            append_neighborhood_metadata(
                &mut encoded,
                &neighborhood.name,
                [
                    neighborhood.uses,
                    neighborhood.generated,
                    neighborhood.evaluated,
                    neighborhood.cpu_nanos,
                    neighborhood.improvements,
                    neighborhood.global_bests,
                    neighborhood.positive_rewards,
                    neighborhood.weight_milli,
                ],
            );
        }
        for auxiliary in &self.routing.auxiliary {
            // Preserve the stable v1 wire format used by the launchers. For
            // these explicitly named support records, `generated` carries
            // deterministic work units and `evaluated` remains zero.
            append_neighborhood_metadata(
                &mut encoded,
                &auxiliary.name,
                [auxiliary.uses, auxiliary.work_units, 0, auxiliary.cpu_nanos, auxiliary.productive, 0, auxiliary.productive, 1_000],
            );
        }
        for (family, operators) in [("alns_destroy", &self.alns.destroy), ("alns_repair", &self.alns.repair)] {
            for operator in operators {
                append_neighborhood_metadata(
                    &mut encoded,
                    &format!("{family}/{}", operator.name),
                    [
                        operator.uses,
                        operator.generated,
                        operator.evaluated,
                        operator.cpu_nanos,
                        operator.improvements,
                        operator.global_bests,
                        operator.positive_rewards,
                        operator.weight_milli,
                    ],
                );
            }
        }
        encoded
    }
}

impl fmt::Display for ListSearchMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "qayd list-LS metrics: elapsed={:.6}s candidates={} candidates/s={:.1}",
            self.elapsed_nanos as f64 / 1_000_000_000.0,
            self.candidates,
            self.candidates_per_second()
        )?;
        writeln!(
            f,
            "  constructor={} construction={:.6}s first-feasible={} construction-candidates={} fleet={} cost={}",
            self.constructor.as_deref().unwrap_or("none"),
            self.construction_nanos as f64 / 1_000_000_000.0,
            self.time_to_first_feasible_nanos.map_or_else(|| "none".to_owned(), |value| format!("{:.6}s", value as f64 / 1_000_000_000.0)),
            self.construction_candidates,
            self.constructor_fleet.map_or_else(|| "none".to_owned(), |value| value.to_string()),
            self.constructor_cost.map_or_else(|| "none".to_owned(), |value| value.to_string()),
        )?;
        writeln!(
            f,
            "  trial-list-evals={} incremental={} full-fallback={} full-fallback%={:.2}",
            self.trial_list_evaluations,
            self.incremental_trial_list_evaluations,
            self.full_recompute_trial_list_evaluations,
            self.full_recompute_percentage()
        )?;
        writeln!(
            f,
            "  scan-recomputations={} scan-steps={} average-recomputed-scan-steps={:.2}",
            self.scan_recomputations,
            self.scan_recomputed_steps,
            self.average_recomputed_scan_steps()
        )?;
        writeln!(
            f,
            "  alns-iterations={} accepted={} improving={} global-bests={} sa-accepts={} late-accepts={} gls-updates={} gls-penalties={} shared-publications={} shared-injections={}",
            self.alns.iterations,
            self.alns.accepted,
            self.alns.improving,
            self.alns.global_bests,
            self.alns.simulated_annealing_accepts,
            self.alns.late_acceptance_accepts,
            self.alns.gls_updates,
            self.alns.gls_penalties,
            self.alns.shared_publications,
            self.alns.shared_injections
        )?;
        for operator in self.alns.destroy.iter().chain(&self.alns.repair) {
            writeln!(
                f,
                "  alns-operator={} uses={} generated={} evaluated={} work-units={} cpu-ns={} accepted={} improvements={} global-bests={} positive-rewards={} weight={:.3}",
                operator.name,
                operator.uses,
                operator.generated,
                operator.evaluated,
                operator.work_units,
                operator.cpu_nanos,
                operator.accepted,
                operator.improvements,
                operator.global_bests,
                operator.positive_rewards,
                operator.weight_milli as f64 / 1_000.0
            )?;
        }
        writeln!(
            f,
            "  routing-slices={} descent={} alns={} relink={} global={} route-elimination-attempts={} ejection-chain-attempts={} chain-relocate-attempts={} guided-segment-exchange-attempts={} macro-candidates-built={} macro-budget-exhaustions={} elite-insertions={} elite-rejections={} path-relink-attempts={} path-relink-steps={} path-relink-budget-exhaustions={}",
            self.routing.slices,
            self.routing.descent_slices,
            self.routing.alns_slices,
            self.routing.relink_slices,
            self.routing.global_scan_slices,
            self.routing.route_elimination_attempts,
            self.routing.ejection_chain_attempts,
            self.routing.chain_relocate_attempts,
            self.routing.guided_segment_exchange_attempts,
            self.routing.macro_candidates_built,
            self.routing.macro_budget_exhaustions,
            self.routing.elite_insertions,
            self.routing.elite_rejections,
            self.routing.path_relink_attempts,
            self.routing.path_relink_steps,
            self.routing.path_relink_budget_exhaustions,
        )?;
        for checkpoint in &self.routing.checkpoints {
            writeln!(
                f,
                "  checkpoint-target-ns={} observed-ns={} feasible={} objectives={:?} fleet={} candidates={}",
                checkpoint.target_nanos,
                checkpoint.observed_nanos,
                checkpoint.feasible,
                checkpoint.objectives,
                checkpoint.fleet.map_or_else(|| "none".to_owned(), |value| value.to_string()),
                checkpoint.candidates,
            )?;
        }
        for neighborhood in &self.routing.neighborhoods {
            writeln!(
                f,
                "  neighborhood={} uses={} generated={} evaluated={} cpu-ns={} improvements={} global-bests={} positive-rewards={} weight={:.3}",
                neighborhood.name,
                neighborhood.uses,
                neighborhood.generated,
                neighborhood.evaluated,
                neighborhood.cpu_nanos,
                neighborhood.improvements,
                neighborhood.global_bests,
                neighborhood.positive_rewards,
                neighborhood.weight_milli as f64 / 1_000.0,
            )?;
        }
        for auxiliary in &self.routing.auxiliary {
            writeln!(
                f,
                "  routing-auxiliary={} uses={} work-units={} cpu-ns={} productive={}",
                auxiliary.name, auxiliary.uses, auxiliary.work_units, auxiliary.cpu_nanos, auxiliary.productive,
            )?;
        }
        for reduction in &self.reductions {
            writeln!(
                f,
                "  reduction={}/{} full-calls={} full-ns={} full-avg-ns={:.1} full-steps={} full-ns/step={:.1} delta-calls={} delta-ns={} delta-avg-ns={:.1}",
                reduction.iterable,
                reduction.op,
                reduction.full_calls,
                reduction.full_nanos,
                reduction.average_full_nanos(),
                reduction.full_steps,
                reduction.average_full_nanos_per_step(),
                reduction.delta_calls,
                reduction.delta_nanos,
                reduction.average_delta_nanos()
            )?;
        }
        Ok(())
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn encode_metadata_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.') {
            encoded.push(char::from(byte));
        } else {
            use std::fmt::Write as _;
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

fn append_neighborhood_metadata(encoded: &mut String, name: &str, values: [u64; 8]) {
    encoded.push(';');
    encoded.push_str(&encode_metadata_component(name));
    for value in values {
        encoded.push(',');
        encoded.push_str(&value.to_string());
    }
}

#[derive(Clone, Copy, Default)]
struct RawReductionMetrics {
    full_calls: u64,
    full_nanos: u64,
    full_steps: u64,
    delta_calls: u64,
    delta_nanos: u64,
}

struct RawMetrics {
    candidates: u64,
    constructor: Option<String>,
    construction_nanos: u64,
    time_to_first_feasible_nanos: Option<u64>,
    construction_candidates: u64,
    constructor_fleet: Option<usize>,
    constructor_cost: Option<i64>,
    incremental_trials: u64,
    full_trials: u64,
    scan_recomputations: u64,
    scan_recomputed_steps: u64,
    alns: AlnsSearchMetrics,
    routing: RoutingSearchMetrics,
    reductions: [RawReductionMetrics; REDUCTION_KINDS],
}

impl Default for RawMetrics {
    fn default() -> Self {
        Self {
            candidates: 0,
            constructor: None,
            construction_nanos: 0,
            time_to_first_feasible_nanos: None,
            construction_candidates: 0,
            constructor_fleet: None,
            constructor_cost: None,
            incremental_trials: 0,
            full_trials: 0,
            scan_recomputations: 0,
            scan_recomputed_steps: 0,
            alns: AlnsSearchMetrics::default(),
            routing: RoutingSearchMetrics::default(),
            reductions: [RawReductionMetrics::default(); REDUCTION_KINDS],
        }
    }
}

/// Kept as `None` when profiling is disabled, so hot-path recording avoids both
/// clock reads and mutable counter traffic in normal solver runs.
pub(super) struct MetricsRecorder {
    raw: Option<RefCell<RawMetrics>>,
}

impl MetricsRecorder {
    pub(super) fn new(enabled: bool) -> Self {
        Self { raw: enabled.then(|| RefCell::new(RawMetrics::default())) }
    }

    pub(super) fn measure_full<T>(&self, reduction: &Reduction, contents_len: usize, evaluate: impl FnOnce() -> T) -> T {
        let Some(raw) = &self.raw else {
            return evaluate();
        };
        let started = Instant::now();
        let result = evaluate();
        let elapsed = nanos(started.elapsed());
        let (iterable, op) = reduction_kind(reduction);
        let steps = iterable_steps(&reduction.iterable, contents_len);
        let mut raw = raw.borrow_mut();
        let metrics = &mut raw.reductions[kind_index(iterable, op)];
        metrics.full_calls = metrics.full_calls.saturating_add(1);
        metrics.full_nanos = metrics.full_nanos.saturating_add(elapsed);
        metrics.full_steps = metrics.full_steps.saturating_add(steps);
        if iterable == ListIterableKind::Scan {
            raw.scan_recomputations = raw.scan_recomputations.saturating_add(1);
            raw.scan_recomputed_steps = raw.scan_recomputed_steps.saturating_add(steps);
        }
        result
    }

    pub(super) fn measure_delta<T>(&self, reduction: &Reduction, evaluate: impl FnOnce() -> T) -> T {
        let Some(raw) = &self.raw else {
            return evaluate();
        };
        let started = Instant::now();
        let result = evaluate();
        let elapsed = nanos(started.elapsed());
        let (iterable, op) = reduction_kind(reduction);
        let mut raw = raw.borrow_mut();
        let metrics = &mut raw.reductions[kind_index(iterable, op)];
        metrics.delta_calls = metrics.delta_calls.saturating_add(1);
        metrics.delta_nanos = metrics.delta_nanos.saturating_add(elapsed);
        result
    }

    pub(super) fn record_candidate(&self) {
        if let Some(raw) = &self.raw {
            let mut raw = raw.borrow_mut();
            raw.candidates = raw.candidates.saturating_add(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_construction(
        &self,
        constructor: impl Into<String>,
        elapsed: Duration,
        first_feasible: Option<Duration>,
        candidates: u64,
        fleet: Option<usize>,
        cost: Option<i64>,
    ) {
        if let Some(raw) = &self.raw {
            let mut raw = raw.borrow_mut();
            raw.constructor = Some(constructor.into());
            raw.construction_nanos = nanos(elapsed);
            raw.time_to_first_feasible_nanos = first_feasible.map(nanos);
            raw.construction_candidates = candidates;
            raw.constructor_fleet = fleet;
            raw.constructor_cost = cost;
        }
    }

    pub(super) fn record_incremental_trial(&self) {
        if let Some(raw) = &self.raw {
            let mut raw = raw.borrow_mut();
            raw.incremental_trials = raw.incremental_trials.saturating_add(1);
        }
    }

    pub(super) fn record_incremental_scan(&self, recomputed_steps: u64) {
        if let Some(raw) = &self.raw {
            let mut raw = raw.borrow_mut();
            raw.scan_recomputations = raw.scan_recomputations.saturating_add(1);
            raw.scan_recomputed_steps = raw.scan_recomputed_steps.saturating_add(recomputed_steps);
        }
    }

    pub(super) fn record_alns(&self, metrics: AlnsSearchMetrics) {
        if let Some(raw) = &self.raw {
            raw.borrow_mut().alns = metrics;
        }
    }

    /// Atomically stores the scheduler profile after search. Building this
    /// aggregate happens outside the recorder, keeping candidate hot paths free
    /// of `RefCell` traffic. Disabled profiling performs no allocation or clock
    /// read here.
    pub(super) fn record_routing(&self, metrics: RoutingSearchMetrics) {
        if let Some(raw) = &self.raw {
            raw.borrow_mut().routing = metrics;
        }
    }

    pub(super) fn snapshot(&self, elapsed: Duration) -> ListSearchMetrics {
        let Some(raw) = &self.raw else {
            return ListSearchMetrics::default();
        };
        let raw = raw.borrow();
        let mut reductions = Vec::new();
        for iterable in ListIterableKind::ALL {
            for op in ListReduceOpKind::ALL {
                let metrics = raw.reductions[kind_index(iterable, op)];
                if metrics.full_calls == 0 && metrics.delta_calls == 0 {
                    continue;
                }
                reductions.push(ReductionSearchMetrics {
                    iterable,
                    op,
                    full_calls: metrics.full_calls,
                    full_nanos: metrics.full_nanos,
                    full_steps: metrics.full_steps,
                    delta_calls: metrics.delta_calls,
                    delta_nanos: metrics.delta_nanos,
                });
            }
        }
        ListSearchMetrics {
            elapsed_nanos: nanos(elapsed),
            constructor: raw.constructor.clone(),
            construction_nanos: raw.construction_nanos,
            time_to_first_feasible_nanos: raw.time_to_first_feasible_nanos,
            construction_candidates: raw.construction_candidates,
            constructor_fleet: raw.constructor_fleet,
            constructor_cost: raw.constructor_cost,
            candidates: raw.candidates,
            trial_list_evaluations: raw.incremental_trials.saturating_add(raw.full_trials),
            incremental_trial_list_evaluations: raw.incremental_trials,
            full_recompute_trial_list_evaluations: raw.full_trials,
            scan_recomputations: raw.scan_recomputations,
            scan_recomputed_steps: raw.scan_recomputed_steps,
            alns: raw.alns.clone(),
            routing: raw.routing.clone(),
            reductions,
        }
    }
}

fn reduction_kind(reduction: &Reduction) -> (ListIterableKind, ListReduceOpKind) {
    let iterable = match reduction.iterable {
        Iterable::Items(_) => ListIterableKind::Items,
        Iterable::SetItems(_) => ListIterableKind::SetItems,
        Iterable::Edges { .. } => ListIterableKind::Edges,
        Iterable::Pairs(_) => ListIterableKind::Pairs,
        Iterable::Scan { .. } => ListIterableKind::Scan,
        Iterable::Windows { .. } => ListIterableKind::Windows,
    };
    let op = match reduction.op {
        ReduceOp::Sum => ListReduceOpKind::Sum,
        ReduceOp::Min => ListReduceOpKind::Min,
        ReduceOp::Max => ListReduceOpKind::Max,
        ReduceOp::Count => ListReduceOpKind::Count,
        ReduceOp::Used => ListReduceOpKind::Used,
        ReduceOp::SelectKth(_) => ListReduceOpKind::SelectKth,
    };
    (iterable, op)
}

const fn kind_index(iterable: ListIterableKind, op: ListReduceOpKind) -> usize {
    iterable.index() * REDUCE_OP_KINDS + op.index()
}

fn iterable_steps(iterable: &Iterable, contents_len: usize) -> u64 {
    let steps = match iterable {
        Iterable::Items(_) | Iterable::SetItems(_) => contents_len,
        Iterable::Edges { .. } => contents_len.saturating_add(1),
        Iterable::Pairs(_) => contents_len.saturating_mul(contents_len),
        Iterable::Scan { end, .. } => contents_len.saturating_add(usize::from(end.is_some())),
        Iterable::Windows { size, .. } => {
            if *size > 0 && contents_len >= *size {
                contents_len - *size + 1
            } else {
                0
            }
        }
    };
    u64::try_from(steps).unwrap_or(u64::MAX)
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
