use std::collections::BTreeSet;
use std::fmt;
use std::time::Duration;

use crate::search_policy::SearchPolicy;

pub use crate::search::SolveStats as SearchStats;

/// User-visible solving policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolveMode {
    #[default]
    Auto,
    Exact,
    LocalSearch,
}

/// JSSP local-search strategy selected by an explicit solve request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScheduleJsspSearch {
    /// Preserve the established persistent tabu implementation.
    #[default]
    Legacy,
    /// Run the bounded TSAB-inspired candidate-ranked N5 pilot.
    TsabCandidate,
    /// Run the bounded TSAB-inspired candidate-ranked N5 pilot on every worker.
    TsabMultiCandidate,
}

/// Stable engine identity used by events and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EngineKind {
    IntegerExact,
    IntegerLocalSearch,
    ListExact,
    RoutingExact,
    ListLocalSearch,
    RoutingLocalSearch,
    ScheduleExact,
    ScheduleLocalSearch,
    Linear,
    Verifier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFamily {
    Integer,
    ListAssignment,
    Routing,
    Sequencing,
    Schedule,
}

impl ModelFamily {
    pub fn name(self) -> &'static str {
        match self {
            Self::Integer => "integer",
            Self::ListAssignment => "list assignment / packing",
            Self::Routing => "routing",
            Self::Sequencing => "sequencing",
            Self::Schedule => "schedule",
        }
    }
}

impl EngineKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::IntegerExact => "integer-exact",
            Self::IntegerLocalSearch => "integer-ls",
            Self::ListExact => "list-exact",
            Self::RoutingExact => "routing-exact",
            Self::ListLocalSearch => "list-ls",
            Self::RoutingLocalSearch => "routing-ls",
            Self::ScheduleExact => "schedule-exact",
            Self::ScheduleLocalSearch => "schedule-ls",
            Self::Linear => "linear",
            Self::Verifier => "verifier",
        }
    }

    /// Whether this engine is authorized to turn exhaustive termination into
    /// an optimality or infeasibility proof. Heuristic engines can publish
    /// feasible candidates and bounds, but never a completion claim.
    pub fn can_prove_complete(self) -> bool {
        matches!(self, Self::IntegerExact | Self::ListExact | Self::RoutingExact | Self::ScheduleExact | Self::Linear)
    }
}

/// Limits shared by parsing-independent solve phases.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SolveLimits {
    pub time: Option<Duration>,
    pub memory_bytes: Option<u64>,
    pub conflicts: Option<u64>,
    pub iterations: Option<u64>,
}

/// Routing controls that must be reproducible through API and CLI arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutingControls {
    pub two_way: bool,
    pub nearest_neighbor: bool,
    pub warm_start: bool,
}

/// Optional continuous relaxation used by exact integer CP search and by
/// certified root bounds for compatible collection engines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LinearBackendMode {
    /// Use the best linked advisory backend, if any.
    #[default]
    Auto,
    /// Keep the native CP bounds only.
    Native,
    /// Use the lightweight Amthal simplex backend.
    Amthal,
}

/// Resource and model-size limits for the advisory linear relaxation.
/// All controls are ordinary request data, so API and CLI executions are
/// reproducible without process-wide environment variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearControls {
    pub backend: LinearBackendMode,
    pub root_time: Duration,
    pub node_time: Duration,
    pub node_depth_interval: usize,
    pub max_variables: usize,
    pub max_rows: usize,
    pub max_nonzeros: usize,
    pub min_coverage_percent: usize,
    /// Maximum physical CP model size eligible for LP primal phase guidance.
    pub phase_max_variables: usize,
    /// Number of remembered customers in each routing ng-neighborhood. A
    /// value of one recovers q-route pricing.
    pub route_ng_size: usize,
    /// Hard label cap for exact ng-route pricing. Reaching it declines the LP
    /// certificate instead of pruning labels heuristically.
    pub route_max_labels: usize,
    /// Weight of the restricted-master dual in stabilized routing pricing.
    pub route_dual_stabilization_percent: usize,
}

impl Default for LinearControls {
    fn default() -> Self {
        Self {
            backend: LinearBackendMode::Auto,
            root_time: Duration::from_millis(50),
            node_time: Duration::ZERO,
            node_depth_interval: 8,
            max_variables: 2_000,
            max_rows: 1_000,
            max_nonzeros: 100_000,
            min_coverage_percent: 1,
            phase_max_variables: 1_000,
            route_ng_size: 8,
            route_max_labels: 2_000_000,
            route_dual_stabilization_percent: 75,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpControls {
    pub split: bool,
    pub probes: usize,
    pub lns: usize,
    pub no_learn_csp: bool,
    pub force_scope_reasons: bool,
    pub shared_pool_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SatBackendMode {
    Native,
    Linear,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SatPreprocess {
    Off,
    #[default]
    Basic,
    Full,
}

/// SAT-specific lowering controls. `backend=None` leaves Boolean models on the
/// general CP compiler; a DIMACS frontend opts into the specialized SAT plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SatControls {
    pub backend: Option<SatBackendMode>,
    pub preprocess: SatPreprocess,
    /// Optional DRAT output path owned by the SAT engine. Frontends only pass
    /// the explicit user request and never open or stream the artifact.
    pub proof_path: Option<String>,
}

impl Default for CpControls {
    fn default() -> Self {
        Self { split: false, probes: 0, lns: 0, no_learn_csp: false, force_scope_reasons: false, shared_pool_capacity: 1 << 14 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticAssumptionOp {
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticAssumption {
    pub variable: usize,
    pub operation: SemanticAssumptionOp,
    pub value: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProofRequest {
    /// Do not impose an additional proof requirement. Completion claims are
    /// still mandatory for `OPTIMAL` and `UNSATISFIABLE` and are always
    /// validated by the canonical result contract.
    #[default]
    None,
    /// Accept only a result carrying a verified completion claim.
    Require,
}

impl Default for RoutingControls {
    fn default() -> Self {
        Self { two_way: true, nearest_neighbor: true, warm_start: true }
    }
}

/// Complete frontend-neutral solve request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolveRequest {
    pub mode: SolveMode,
    pub seed: u64,
    pub threads: usize,
    pub limits: SolveLimits,
    pub profile: bool,
    pub schedule_cdcl: bool,
    /// Enable bounded guide-directed path relinking for scheduling profile 6.
    /// The feature remains inactive without profiling, seven workers, and a
    /// validated global schedule archive containing at least two entries.
    pub schedule_path_relink: bool,
    /// Run bounded exact JSSP repairs in measurement-only shadow mode. The
    /// orchestrator activates this only for wall-clock searches without a
    /// deterministic iteration limit; repairs never publish an incumbent.
    pub schedule_lns_shadow: bool,
    /// Reserve the final scheduling worker for deterministic multi-start
    /// construction. The initial C1 experiment supports exactly one reserved
    /// worker in a seven-worker JSSP local-search portfolio.
    pub schedule_constructor_workers: usize,
    /// Select the JSSP local-search candidate-ranking strategy.
    pub schedule_jssp_search: ScheduleJsspSearch,
    pub routing: RoutingControls,
    pub linear: LinearControls,
    pub cp: CpControls,
    pub sat: SatControls,
    pub assumptions: Vec<SemanticAssumption>,
    pub hints: Vec<(usize, i32)>,
    /// Optional ordered-list incumbent seed, one sequence per semantic list.
    pub list_hint: Option<Vec<Vec<i32>>>,
    /// Optional semantic integer-variable scope that must be exhausted before
    /// exact CP search branches on completion variables. `None` preserves the
    /// engine's legacy all-variable heuristic; `Some([])` requests an explicit
    /// completion-only phase.
    pub primary_branch_scope: Option<Vec<usize>>,
    /// Exact variable priorities within the active branching phase. When a
    /// primary scope is explicit, every entry must belong to that scope and no
    /// entry can promote a completion variable ahead of it.
    pub branch_order: Vec<usize>,
    /// Ordered semantic exact-search phases. The compiled engine always adds
    /// an `Auto` fallback over every remaining physical variable.
    pub search_policy: SearchPolicy,
    /// Request verified assignment snapshots for improving incumbents. This is
    /// intentionally opt-in because replaying every improvement is unsuitable
    /// for high-frequency heuristic loops.
    pub publish_incumbent_assignments: bool,
    pub proof: ProofRequest,
}

impl Default for SolveRequest {
    fn default() -> Self {
        Self {
            mode: SolveMode::Auto,
            seed: 0,
            threads: 1,
            limits: SolveLimits::default(),
            profile: false,
            schedule_cdcl: false,
            schedule_path_relink: false,
            schedule_lns_shadow: false,
            schedule_constructor_workers: 0,
            schedule_jssp_search: ScheduleJsspSearch::Legacy,
            routing: RoutingControls::default(),
            linear: LinearControls::default(),
            cp: CpControls::default(),
            sat: SatControls::default(),
            assumptions: Vec::new(),
            hints: Vec::new(),
            list_hint: None,
            primary_branch_scope: None,
            branch_order: Vec::new(),
            search_policy: SearchPolicy::default(),
            publish_incumbent_assignments: false,
            proof: ProofRequest::None,
        }
    }
}

impl SolveRequest {
    pub fn validate(&self) -> Result<(), SolveError> {
        if self.threads == 0 {
            return Err(SolveError::InvalidRequest("threads must be positive".to_string()));
        }
        if self.schedule_constructor_workers > 1 {
            return Err(SolveError::InvalidRequest("schedule_constructor_workers currently supports only 0 or 1".to_string()));
        }
        if self.schedule_constructor_workers == 1 && self.threads != 7 {
            return Err(SolveError::InvalidRequest("schedule_constructor_workers=1 currently requires threads=7".to_string()));
        }
        if self.schedule_constructor_workers != 0 && (self.schedule_path_relink || self.schedule_lns_shadow) {
            return Err(SolveError::InvalidRequest(
                "schedule_constructor_workers cannot yet be combined with schedule_path_relink or schedule_lns_shadow".to_string(),
            ));
        }
        let tsab_strategy = match self.schedule_jssp_search {
            ScheduleJsspSearch::Legacy => None,
            ScheduleJsspSearch::TsabCandidate => Some("tsab-candidate"),
            ScheduleJsspSearch::TsabMultiCandidate => Some("tsab-multi-candidate"),
        };
        if let Some(strategy) = tsab_strategy {
            let unsupported_path_relink =
                self.schedule_path_relink && !matches!(self.schedule_jssp_search, ScheduleJsspSearch::TsabMultiCandidate);
            if self.schedule_constructor_workers != 0 || unsupported_path_relink || self.schedule_lns_shadow {
                return Err(SolveError::InvalidRequest(format!(
                    "schedule_jssp_search='{}' cannot yet be combined with schedule_constructor_workers, schedule_path_relink, or schedule_lns_shadow",
                    strategy
                )));
            }
            if self.threads != 7 {
                return Err(SolveError::InvalidRequest(format!("schedule_jssp_search='{strategy}' currently requires threads=7")));
            }
        }
        if self.limits.memory_bytes == Some(0) {
            return Err(SolveError::InvalidRequest("memory limit must be positive when provided".to_string()));
        }
        if self.cp.shared_pool_capacity == 0 {
            return Err(SolveError::InvalidRequest("shared clause capacity must be positive".to_string()));
        }
        if self.linear.max_variables == 0 || self.linear.max_rows == 0 || self.linear.max_nonzeros == 0 {
            return Err(SolveError::InvalidRequest("linear relaxation size limits must be positive".to_string()));
        }
        if self.linear.node_depth_interval == 0 {
            return Err(SolveError::InvalidRequest("linear relaxation node depth interval must be positive".to_string()));
        }
        if self.linear.min_coverage_percent > 100 {
            return Err(SolveError::InvalidRequest("linear relaxation minimum coverage must be between 0 and 100 percent".to_string()));
        }
        if self.linear.backend == LinearBackendMode::Amthal && !cfg!(feature = "lp-relaxation") {
            return Err(SolveError::InvalidRequest("linear backend 'amthal' requires the lp-relaxation Cargo feature".to_string()));
        }
        if self.sat.proof_path.is_some() && self.sat.backend != Some(SatBackendMode::Native) {
            return Err(SolveError::InvalidRequest("SAT proof output requires the native SAT backend".to_string()));
        }
        if self.sat.proof_path.as_deref().is_some_and(str::is_empty) {
            return Err(SolveError::InvalidRequest("SAT proof output path must not be empty".to_string()));
        }
        if self.proof == ProofRequest::Require && self.mode == SolveMode::LocalSearch {
            return Err(SolveError::InvalidRequest("proof=Require cannot be combined with a local-search-only request".to_string()));
        }
        self.search_policy.validate().map_err(SolveError::InvalidRequest)?;
        if !self.search_policy.is_auto() {
            if self.mode == SolveMode::LocalSearch {
                return Err(SolveError::InvalidRequest("search_policy requires an exact CP search".to_string()));
            }
            if self.primary_branch_scope.is_some() || !self.branch_order.is_empty() || !self.hints.is_empty() {
                return Err(SolveError::InvalidRequest(
                    "search_policy cannot be combined with hints, primary_branch_scope, or branch_order".to_string(),
                ));
            }
        }
        if let Some(scope) = &self.primary_branch_scope {
            let mut seen = BTreeSet::new();
            for &variable in scope {
                if !seen.insert(variable) {
                    return Err(SolveError::InvalidRequest(format!("integer variable {variable} appears twice in primary_branch_scope")));
                }
            }
            if let Some(&variable) = self.branch_order.iter().find(|variable| !seen.contains(variable)) {
                return Err(SolveError::InvalidRequest(format!("branch_order variable {variable} is outside primary_branch_scope")));
            }
        }
        Ok(())
    }
}

/// Final semantic status. Only the orchestrator constructs this status from
/// engine claims and verified candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveStatus {
    Optimal,
    Satisfiable,
    Unsatisfiable,
    Unknown,
    Unsupported,
}

impl SolveStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Optimal => "OPTIMAL",
            Self::Satisfiable => "SATISFIABLE",
            Self::Unsatisfiable => "UNSATISFIABLE",
            Self::Unknown => "UNKNOWN",
            Self::Unsupported => "UNSUPPORTED",
        }
    }
}

/// Objective direction for one lexicographic tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

/// One interval value in a typed assignment arena.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntervalValue {
    pub start: Option<i64>,
    pub present: bool,
    pub machine: Option<usize>,
    /// Stable semantic mode index, when the interval has alternatives.
    pub mode: Option<usize>,
}

/// Assignment arenas. Empty arenas mean that the corresponding variable kind
/// does not occur in the model; their contents are never interpreted by shape.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Assignment {
    pub integers: Vec<Option<i64>>,
    pub sets: Vec<Vec<i32>>,
    pub lists: Vec<Vec<i32>>,
    pub intervals: Vec<IntervalValue>,
}

/// Strength of validation already applied to a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationLevel {
    /// Private engine-local state. Local candidates never cross the event bus
    /// and cannot be published as a final result.
    Local,
    /// Replayed at an engine boundary and safe to seed another plan.
    Transfer,
    /// Replayed for publication in the final result.
    Final,
}

/// Feasible primal candidate. The event bus accepts only canonically verified
/// candidates; engines keep high-frequency local candidates private.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSolution {
    assignment: Assignment,
    objectives: Vec<i64>,
    source: EngineKind,
    verification: VerificationLevel,
}

impl CandidateSolution {
    pub(super) fn verified(assignment: Assignment, objectives: Vec<i64>, source: EngineKind, verification: VerificationLevel) -> Self {
        Self { assignment, objectives, source, verification }
    }

    /// Construct a candidate only after the canonical semantic replay accepts
    /// the complete assignment and its claimed objective vector. This is the
    /// safe boundary for workspace frontends with a specialized parser.
    pub fn replayed(
        model: &crate::model::Model,
        assignment: Assignment,
        objectives: Vec<i64>,
        source: EngineKind,
        verification: VerificationLevel,
    ) -> Result<Self, SolveError> {
        super::verify_semantic_assignment(model, &assignment, &objectives)?;
        Ok(Self::verified(assignment, objectives, source, verification))
    }

    pub fn assignment(&self) -> &Assignment {
        &self.assignment
    }

    pub fn objectives(&self) -> &[i64] {
        &self.objectives
    }

    pub fn source(&self) -> EngineKind {
        self.source
    }

    pub fn verification(&self) -> VerificationLevel {
        self.verification
    }

    pub fn transferable(&self) -> bool {
        matches!(self.verification, VerificationLevel::Transfer | VerificationLevel::Final)
    }
}

/// Certified dual bound for one objective tier.
#[derive(Clone, Debug, PartialEq)]
pub struct Bound {
    pub tier: usize,
    pub value: i64,
    pub method: String,
}

/// Canonical primal-dual gap for one lexicographic tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimalityGap {
    pub absolute: u64,
    pub relative: f64,
}

/// Public description of the evidence behind a sealed proof claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofKind {
    /// The named exact engine exhausted its complete search space.
    CompleteSearch { engine: EngineKind, objective_tiers: usize },
    /// Exact tier closure used one or more independently certified dual
    /// bounds. `methods` contains one honest evidence description per tier.
    CertifiedBounds { engine: EngineKind, methods: Vec<String>, objective_tiers: usize },
    /// A proof artifact was checked by the named verifier.
    VerifiedArtifact { format: String, verifier: String },
    /// Independent submodels were proved separately and combined by the
    /// orchestrator's disjoint-assignment merge rule.
    Decomposed { components: Vec<ProofClaim>, objective_tiers: usize },
}

/// Evidence attached to an exact completion claim.
///
/// Its fields are private so a frontend cannot manufacture optimality or
/// infeasibility. Only the orchestrator can seal a complete-search claim after
/// an authorized exact engine terminates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofClaim {
    kind: ProofKind,
    conclusion: ProvenConclusion,
}

impl ProofClaim {
    pub(super) fn complete_search(engine: EngineKind, conclusion: ProvenConclusion, objective_tiers: usize) -> Self {
        Self { kind: ProofKind::CompleteSearch { engine, objective_tiers }, conclusion }
    }

    pub(super) fn certified_bounds(engine: EngineKind, methods: Vec<String>, objective_tiers: usize) -> Result<Self, SolveError> {
        if objective_tiers == 0 || methods.len() != objective_tiers {
            return Err(SolveError::InvalidResult(format!(
                "certified-bound proof has {} methods for {objective_tiers} objective tiers",
                methods.len()
            )));
        }
        if methods.iter().any(String::is_empty) {
            return Err(SolveError::InvalidResult("certified-bound proof contains an empty method".to_string()));
        }
        Ok(Self { kind: ProofKind::CertifiedBounds { engine, methods, objective_tiers }, conclusion: ProvenConclusion::Optimal })
    }

    pub(super) fn decomposed(components: Vec<Self>, conclusion: ProvenConclusion, objective_tiers: usize) -> Result<Self, SolveError> {
        if components.is_empty() {
            return Err(SolveError::InvalidResult("a decomposed proof must contain component evidence".to_string()));
        }
        match conclusion {
            ProvenConclusion::Optimal => {
                if components.iter().any(|proof| proof.conclusion != ProvenConclusion::Optimal) {
                    return Err(SolveError::InvalidResult(
                        "a decomposed optimality proof contains non-optimal component evidence".to_string(),
                    ));
                }
                let covered = components.iter().try_fold(0usize, |covered, proof| {
                    proof
                        .objective_tiers()
                        .and_then(|tiers| covered.checked_add(tiers))
                        .ok_or_else(|| SolveError::InvalidResult("decomposed optimality proof tier count overflowed".to_string()))
                })?;
                if covered != objective_tiers {
                    return Err(SolveError::InvalidResult(format!(
                        "decomposed optimality proof covers {covered} objective tiers but the model has {objective_tiers}"
                    )));
                }
            }
            ProvenConclusion::Unsatisfiable => {
                if !components.iter().any(|proof| proof.conclusion == ProvenConclusion::Unsatisfiable) {
                    return Err(SolveError::InvalidResult(
                        "a decomposed unsatisfiability proof has no unsatisfiable component evidence".to_string(),
                    ));
                }
            }
        }
        for proof in &components {
            proof.validate_authority()?;
        }
        Ok(Self { kind: ProofKind::Decomposed { components, objective_tiers }, conclusion })
    }

    pub fn kind(&self) -> &ProofKind {
        &self.kind
    }

    pub fn conclusion(&self) -> ProvenConclusion {
        self.conclusion
    }

    pub fn engine(&self) -> Option<EngineKind> {
        match &self.kind {
            ProofKind::CompleteSearch { engine, .. } | ProofKind::CertifiedBounds { engine, .. } => Some(*engine),
            ProofKind::VerifiedArtifact { .. } | ProofKind::Decomposed { .. } => None,
        }
    }

    pub fn objective_tiers(&self) -> Option<usize> {
        match &self.kind {
            ProofKind::CompleteSearch { objective_tiers, .. }
            | ProofKind::CertifiedBounds { objective_tiers, .. }
            | ProofKind::Decomposed { objective_tiers, .. } => Some(*objective_tiers),
            ProofKind::VerifiedArtifact { .. } => None,
        }
    }

    fn validate_authority(&self) -> Result<(), SolveError> {
        match &self.kind {
            ProofKind::CompleteSearch { engine, .. } => {
                if !engine.can_prove_complete() {
                    return Err(SolveError::InvalidResult(format!(
                        "engine {} is not authorized to issue complete-search proofs",
                        engine.name()
                    )));
                }
            }
            ProofKind::CertifiedBounds { engine, methods, objective_tiers } => {
                if !engine.can_prove_complete() {
                    return Err(SolveError::InvalidResult(format!(
                        "engine {} is not authorized to issue certified-bound proofs",
                        engine.name()
                    )));
                }
                if self.conclusion != ProvenConclusion::Optimal
                    || *objective_tiers == 0
                    || methods.len() != *objective_tiers
                    || methods.iter().any(String::is_empty)
                {
                    return Err(SolveError::InvalidResult("malformed certified-bound proof".to_string()));
                }
            }
            ProofKind::VerifiedArtifact { .. } => {}
            ProofKind::Decomposed { components, objective_tiers } => {
                if components.is_empty() {
                    return Err(SolveError::InvalidResult("a decomposed proof has no component evidence".to_string()));
                }
                for proof in components {
                    proof.validate_authority()?;
                }
                if self.conclusion == ProvenConclusion::Optimal {
                    let covered = components.iter().try_fold(0usize, |covered, proof| {
                        if proof.conclusion != ProvenConclusion::Optimal {
                            return Err(SolveError::InvalidResult(
                                "a decomposed optimality proof contains non-optimal component evidence".to_string(),
                            ));
                        }
                        proof
                            .objective_tiers()
                            .and_then(|tiers| covered.checked_add(tiers))
                            .ok_or_else(|| SolveError::InvalidResult("decomposed optimality proof tier count overflowed".to_string()))
                    })?;
                    if covered != *objective_tiers {
                        return Err(SolveError::InvalidResult(format!(
                            "decomposed optimality proof covers {covered} objective tiers but declares {objective_tiers}"
                        )));
                    }
                } else if !components.iter().any(|proof| proof.conclusion == ProvenConclusion::Unsatisfiable) {
                    return Err(SolveError::InvalidResult(
                        "a decomposed unsatisfiability proof has no unsatisfiable component evidence".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProvenConclusion {
    Optimal,
    Unsatisfiable,
}

/// Per-engine execution report. Search statistics stay in their native compact
/// representation and are aggregated without frontend involvement.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineReport {
    pub engine: Option<EngineKind>,
    pub search: SearchStats,
    pub elapsed: Duration,
    pub improvements: u64,
    pub metadata: Vec<(String, String)>,
}

/// Low-frequency facts crossing engine boundaries.
#[derive(Clone, Debug, PartialEq)]
pub enum SolveEvent {
    Candidate(CandidateSolution),
    Bound(Bound),
    Proof(ProofClaim),
    /// A concrete engine stage is about to run. Emitted before the stage so a
    /// verbose consumer can announce the phase in real time (e.g. a warm-start
    /// local search feeding an exact stage). `warm_start` is true when this stage
    /// only seeds the stage after it.
    StageStarted {
        engine: EngineKind,
        warm_start: bool,
    },
    Progress {
        engine: EngineKind,
        objectives: Vec<i64>,
        /// Time since the orchestrator created the shared solve budget. This
        /// keeps protocol renderers from owning clocks or deadlines.
        elapsed: Duration,
    },
    Finished(EngineReport),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventControl {
    Continue,
    Stop,
}

/// Consumer of solve events. Clause exchange deliberately uses its own
/// high-frequency engine-private channel and never this interface.
pub trait EventSink {
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError>;
}

pub struct EventCallback<F>(pub F);

impl<F> EventSink for EventCallback<F>
where
    F: FnMut(SolveEvent) -> Result<EventControl, SolveError>,
{
    fn emit(&mut self, event: SolveEvent) -> Result<EventControl, SolveError> {
        (self.0)(event)
    }
}

#[derive(Default)]
pub struct IgnoreEvents;

impl EventSink for IgnoreEvents {
    fn emit(&mut self, _event: SolveEvent) -> Result<EventControl, SolveError> {
        Ok(EventControl::Continue)
    }
}

/// Final result shared by native and language frontends.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveResult {
    pub(super) status: SolveStatus,
    pub(super) primal: Option<CandidateSolution>,
    pub(super) bounds: Vec<Bound>,
    pub(super) proof: Option<ProofClaim>,
    pub(super) reports: Vec<EngineReport>,
    pub(super) message: Option<String>,
}

impl SolveResult {
    pub fn unknown() -> Self {
        Self { status: SolveStatus::Unknown, primal: None, bounds: Vec::new(), proof: None, reports: Vec::new(), message: None }
    }

    pub fn status(&self) -> SolveStatus {
        self.status
    }

    pub fn primal(&self) -> Option<&CandidateSolution> {
        self.primal.as_ref()
    }

    pub fn bounds(&self) -> &[Bound] {
        &self.bounds
    }

    pub fn proof(&self) -> Option<&ProofClaim> {
        self.proof.as_ref()
    }

    pub fn reports(&self) -> &[EngineReport] {
        &self.reports
    }

    /// Canonical aggregate over all engine and component reports.
    pub fn aggregate_search_stats(&self) -> SearchStats {
        self.reports.iter().fold(SearchStats::default(), |mut total, report| {
            super::merge_search_stats(&mut total, report.search);
            total
        })
    }

    /// Canonical elapsed summary for renderers. Portfolio workers are already
    /// collapsed into one engine report, while sequential stages and
    /// decomposed components retain one report each.
    pub fn elapsed(&self) -> Duration {
        self.reports.iter().fold(Duration::ZERO, |total, report| total.saturating_add(report.elapsed))
    }

    pub fn improvements(&self) -> u64 {
        self.reports.iter().fold(0, |total, report| total.saturating_add(report.improvements))
    }

    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    pub fn into_primal(self) -> Option<CandidateSolution> {
        self.primal
    }

    /// Compute the shared gap convention used by every renderer. The relative
    /// denominator is `max(abs(primal), abs(bound), 1)` and an inconsistent
    /// bound direction is rejected.
    pub fn optimality_gap(&self, tier: usize, minimizing: bool) -> Option<OptimalityGap> {
        let primal = *self.primal.as_ref()?.objectives().get(tier)?;
        let bound = self.bounds.iter().find(|bound| bound.tier == tier)?;
        let signed = if minimizing { i128::from(primal) - i128::from(bound.value) } else { i128::from(bound.value) - i128::from(primal) };
        if signed < 0 {
            return None;
        }
        let absolute = u64::try_from(signed).unwrap_or(u64::MAX);
        let denominator = primal.unsigned_abs().max(bound.value.unsigned_abs()).max(1) as f64;
        Some(OptimalityGap { absolute, relative: absolute as f64 / denominator })
    }

    pub fn validate_contract(&self) -> Result<(), SolveError> {
        if self.primal.as_ref().is_some_and(|candidate| candidate.verification() != VerificationLevel::Final) {
            return Err(SolveError::InvalidResult("final primal was not verified for publication".to_string()));
        }
        match self.status {
            SolveStatus::Optimal => {
                if self.primal.is_none() || self.proof.is_none() {
                    return Err(SolveError::InvalidResult("OPTIMAL requires a verified primal and a proof claim".to_string()));
                }
                if self.proof.as_ref().is_some_and(|proof| proof.conclusion() != ProvenConclusion::Optimal) {
                    return Err(SolveError::InvalidResult("OPTIMAL requires an optimality proof".to_string()));
                }
            }
            SolveStatus::Satisfiable if self.primal.is_none() || self.proof.is_some() => {
                return Err(SolveError::InvalidResult(
                    "SATISFIABLE requires a verified primal and cannot carry a completion proof".to_string(),
                ));
            }
            SolveStatus::Unsatisfiable if self.primal.is_some() || self.proof.is_none() => {
                return Err(SolveError::InvalidResult("UNSATISFIABLE requires no primal and a proof claim".to_string()));
            }
            SolveStatus::Unknown | SolveStatus::Unsupported if self.primal.is_some() || self.proof.is_some() => {
                return Err(SolveError::InvalidResult("UNKNOWN or UNSUPPORTED cannot carry a primal or completion proof".to_string()));
            }
            _ => {}
        }
        if self.status == SolveStatus::Unsatisfiable
            && self.proof.as_ref().is_some_and(|proof| proof.conclusion() != ProvenConclusion::Unsatisfiable)
        {
            return Err(SolveError::InvalidResult("UNSATISFIABLE requires an unsatisfiability proof".to_string()));
        }
        if let Some(proof) = &self.proof {
            proof.validate_authority()?;
        }
        Ok(())
    }

    pub(crate) fn validate_model_contract(&self, model: &crate::model::Model) -> Result<(), SolveError> {
        self.validate_contract()?;
        let objective_tiers = model.objectives().len();
        if let Some(candidate) = &self.primal {
            if candidate.objectives().len() != objective_tiers {
                return Err(SolveError::InvalidResult(format!(
                    "candidate has {} objective tiers but the semantic model has {objective_tiers}",
                    candidate.objectives().len()
                )));
            }
        }
        if let Some(proof_tiers) = self.proof.as_ref().and_then(ProofClaim::objective_tiers) {
            if proof_tiers != objective_tiers {
                return Err(SolveError::InvalidResult(format!(
                    "proof covers {proof_tiers} objective tiers but the semantic model has {objective_tiers}"
                )));
            }
        }
        let mut bound_tiers = std::collections::BTreeSet::new();
        for bound in &self.bounds {
            if bound.method.trim().is_empty() {
                return Err(SolveError::InvalidResult("bound method must not be empty".to_string()));
            }
            if bound.tier >= objective_tiers {
                return Err(SolveError::InvalidResult(format!(
                    "bound references objective tier {} but the semantic model has {objective_tiers} tiers",
                    bound.tier
                )));
            }
            if !bound_tiers.insert(bound.tier) {
                return Err(SolveError::InvalidResult(format!("duplicate bound for objective tier {}", bound.tier)));
            }
            if self.primal.is_some() && self.optimality_gap(bound.tier, model.objectives()[bound.tier].is_minimize()).is_none() {
                return Err(SolveError::InvalidResult(format!(
                    "bound for objective tier {} is inconsistent with the verified primal",
                    bound.tier
                )));
            }
        }
        if let Some(engine) = self.proof.as_ref().and_then(ProofClaim::engine) {
            if !self.reports.iter().any(|report| report.engine == Some(engine)) {
                return Err(SolveError::InvalidResult(format!("proof from engine {} has no matching engine report", engine.name())));
            }
        }
        if self.status == SolveStatus::Optimal {
            for bound in &self.bounds {
                let minimizing = model.objectives()[bound.tier].is_minimize();
                if self.optimality_gap(bound.tier, minimizing).is_some_and(|gap| gap.absolute != 0) {
                    return Err(SolveError::InvalidResult(format!(
                        "OPTIMAL carries a non-closing bound for objective tier {}",
                        bound.tier
                    )));
                }
            }
        }
        if let Some(candidate) = &self.primal {
            let source = candidate.source();
            if source != EngineKind::Verifier && !self.reports.iter().any(|report| report.engine == Some(source)) {
                return Err(SolveError::InvalidResult(format!("primal from engine {} has no matching engine report", source.name())));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolveError {
    InvalidRequest(String),
    Unsupported(String),
    Compile(String),
    Engine(String),
    InvalidResult(String),
    /// Cooperative cancellation observed at an interruptible boundary. Only
    /// this variant may be converted into an UNKNOWN solve result.
    Interrupted(String),
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(f, "invalid solve request: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported model: {message}"),
            Self::Compile(message) => write!(f, "model compilation failed: {message}"),
            Self::Engine(message) => write!(f, "engine failed: {message}"),
            Self::InvalidResult(message) => write!(f, "invalid solve result: {message}"),
            Self::Interrupted(message) => write!(f, "solve interrupted: {message}"),
        }
    }
}

impl std::error::Error for SolveError {}
